from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from konsilium import (
    case_review,
    doctor_letter,
    guidelines_lookup,
    ingest_text,
    monitor_review,
    render_doctor_letter,
    pubmed_search,
    semanticscholar_search,
)
from konsilium.egress import EgressViolation
from konsilium.knowledge import SearchResult
from konsilium.memory import PatientMemory
from konsilium.research import paper_deepen_request, retraction_sweep_request
from konsilium.smoke import knowledge_smoke


class StagesOneToThreeTest(unittest.TestCase):
    def test_ingest_structures_patient_files_and_searches_external_sources(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ingest_text(
                "case-1",
                "\n".join(
                    [
                        "Patient: Anna Mueller",
                        "2026-07-01 HbA1c 8.1, started Metformin 500 mg.",
                        "Problem: Type 2 diabetes",
                    ]
                ),
                root,
                allow_synthetic=True,
            )

            patient_dir = root / "patients" / "case-1"
            self.assertIn("2026-07-01", (patient_dir / "timeline" / "events.md").read_text())
            self.assertIn("Type 2 diabetes", (patient_dir / "problems.md").read_text())
            self.assertIn("Metformin 500 mg", (patient_dir / "meds.md").read_text())
            self.assertIn("HbA1c 8.1", (patient_dir / "labs" / "labs.md").read_text())

            fetched_urls: list[str] = []

            def fake_fetch(url: str) -> str:
                fetched_urls.append(url)
                return json.dumps(
                    {
                        "data": [
                            {
                                "title": "Metformin review",
                                "url": "https://example.test/paper",
                                "year": 2025,
                            }
                        ]
                    }
                )

            results = semanticscholar_search("metformin hba1c", fetch=fake_fetch)
            self.assertEqual(results[0].title, "Metformin review")
            self.assertIn("semanticscholar", fetched_urls[0])

            with self.assertRaises(EgressViolation):
                guidelines_lookup("diabetes [PATIENT_1]")

    def test_knowledge_tools_use_configured_api_keys_without_exposing_pii(self) -> None:
        pubmed_urls: list[str] = []

        def fake_pubmed(url: str) -> str:
            pubmed_urls.append(url)
            return json.dumps({"esearchresult": {"idlist": ["12345"]}})

        old_ncbi = os.environ.get("NCBI_API_KEY")
        os.environ["NCBI_API_KEY"] = "test_ncbi_key"
        try:
            pubmed = pubmed_search("metformin hba1c", fetch=fake_pubmed)
        finally:
            if old_ncbi is None:
                os.environ.pop("NCBI_API_KEY", None)
            else:
                os.environ["NCBI_API_KEY"] = old_ncbi

        self.assertEqual(pubmed[0].url, "https://pubmed.ncbi.nlm.nih.gov/12345/")
        self.assertIn("api_key=test_ncbi_key", pubmed_urls[0])

        seen_headers: list[dict] = []

        def fake_semantic(url: str, headers: dict | None = None) -> str:
            seen_headers.append(headers or {})
            return json.dumps({"data": [{"title": "Paper", "url": "https://paper.test"}]})

        old_s2 = os.environ.get("S2_API_KEY")
        os.environ["S2_API_KEY"] = "test_s2_key"
        try:
            semanticscholar_search("metformin hba1c", fetch=fake_semantic)
        finally:
            if old_s2 is None:
                os.environ.pop("S2_API_KEY", None)
            else:
                os.environ["S2_API_KEY"] = old_s2

        self.assertEqual(seen_headers[0]["x-api-key"], "test_s2_key")

    def test_knowledge_smoke_reports_safe_offline_urls(self) -> None:
        report = knowledge_smoke("metformin hba1c")

        self.assertTrue(report["passed"])
        self.assertIn("pubmed", report["sources"])
        self.assertIn("semanticscholar", report["sources"])

    def test_patient_memory_search_get_filters_by_patient_id(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ingest_text(
                "case-1",
                "Patient: Anna Mueller\n2026-07-01 HbA1c 8.1\nProblem: Type 2 diabetes",
                root,
                allow_synthetic=True,
            )
            ingest_text(
                "case-2",
                "Patient: Bob Smith\n2026-06-01 LDL 190\nProblem: Hyperlipidemia",
                root,
                allow_synthetic=True,
            )

            memory = PatientMemory(root)
            hits = memory.search("HbA1c diabetes", patient_id="case-1", limit=3)
            text = "\n".join(memory.get(hit["path"]) for hit in hits)

            self.assertTrue(hits)
            self.assertIn("case-1", {hit["patient_id"] for hit in hits})
            self.assertIn("HbA1c", text)
            self.assertNotIn("LDL 190", text)

    def test_reviews_letters_monitoring_and_research_hooks_cover_stages_two_and_three(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ingest_text(
                "case-1",
                "\n".join(
                    [
                        "Patient: Anna Mueller",
                        "Adresse: Hauptstrasse 7, Berlin",
                        "2026-07-01 HbA1c 8.1, started Metformin 500 mg.",
                        "Problem: Type 2 diabetes",
                    ]
                ),
                root,
                allow_synthetic=True,
            )
            ingest_text(
                "case-2",
                "\n".join(
                    [
                        "Patient: Bob Smith",
                        "2026-06-01 LDL 190, atorvastatin discussed.",
                        "Problem: Hyperlipidemia",
                    ]
                ),
                root,
                allow_synthetic=True,
            )

            knowledge = [SearchResult("AWMF diabetes guideline", "guidelines:awmf", "https://awmf.org")]
            report = case_review(root, "case-1", roles=["internist", "endocrinologist"], knowledge=knowledge)
            self.assertEqual(report["patient_id"], "case-1")
            self.assertIn("internist", report["perspectives"])
            self.assertIn("endocrinologist", report["perspectives"])
            self.assertTrue(report["chair_summary"])
            self.assertGreaterEqual(len(report["claims"]), 2)
            self.assertEqual(report["disagreements"], [])
            self.assertIn("case-1", (root / "patients" / "case-1" / "consilium" / "latest.json").read_text())
            self.assertNotIn("Bob Smith", json.dumps(report))

            draft_path = doctor_letter(root, "case-1", language="de")
            draft = draft_path.read_text(encoding="utf-8")
            self.assertIn("[PATIENT_1]", draft)
            self.assertNotIn("Anna Mueller", draft)

            rendered = render_doctor_letter(root, "case-1", draft)
            self.assertIn("Anna Mueller", rendered)
            self.assertIn("Hauptstrasse 7", rendered)

            english = doctor_letter(root, "case-1", language="en").read_text(encoding="utf-8")
            russian = doctor_letter(root, "case-1", language="ru").read_text(encoding="utf-8")
            self.assertIn("Dear doctor", english)
            self.assertIn("Уважаемый врач", russian)
            self.assertIn("[PATIENT_1]", english)
            self.assertIn("[PATIENT_1]", russian)

            with self.assertRaises(ValueError):
                doctor_letter(root, "case-1", language="fr")

            monitor = monitor_review(root, ["case-1", "case-2"])
            self.assertEqual({item["patient_id"] for item in monitor["patients"]}, {"case-1", "case-2"})
            self.assertTrue((root / "monitor_schedule.json").exists())

            deepen = paper_deepen_request("case-1", "10.1000/example")
            sweep = retraction_sweep_request("case-1", ["10.1000/example"])
            self.assertEqual(deepen["target_agent"], "research-agent")
            self.assertEqual(sweep["tool"], "retraction_sweep")

    def test_case_review_can_use_model_client_without_raw_pii(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.calls = []

            def build_kwargs(self, messages, system_prompt, tools):
                call = {"messages": messages, "system_prompt": system_prompt, "tools": tools}
                self.calls.append(call)
                return call

            def call(self, kwargs):
                if "Chair synthesis" in kwargs["system_prompt"]:
                    return SimpleNamespace(
                        content=json.dumps(
                            {
                                "chair_summary": "Endocrine risk needs specialist follow-up.",
                                "claims": ["Chair synthesized the role outputs."],
                                "open_questions": ["Should endocrinology adjust medication?"],
                                "disagreements": ["internist vs endocrinologist: urgency differs."],
                                "recommended_next_step": "Discuss HbA1c trend with the treating physician.",
                            }
                        )
                    )
                role = "endocrinologist" if "endocrinologist" in kwargs["system_prompt"] else "internist"
                return SimpleNamespace(
                    content=json.dumps(
                        {
                            "claims": [f"{role} role claim."],
                            "open_questions": [f"{role} question?"],
                        }
                    )
                )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ingest_text(
                "case-1",
                "\n".join(
                    [
                        "Patient: Anna Mueller",
                        "Adresse: Hauptstrasse 7, Berlin",
                        "2026-07-01 HbA1c 8.1, started Metformin 500 mg.",
                        "Problem: Type 2 diabetes",
                    ]
                ),
                root,
                allow_synthetic=True,
            )
            ingest_text(
                "case-2",
                "Patient: Bob Smith\n2026-06-01 LDL 190\nProblem: Hyperlipidemia",
                root,
                allow_synthetic=True,
            )
            model = FakeModel()

            report = case_review(
                root,
                "case-1",
                roles=["internist", "endocrinologist"],
                knowledge=[SearchResult("Guideline", "guidelines:awmf", "https://awmf.org")],
                model_client=model,
            )

            self.assertEqual(len(model.calls), 3)
            self.assertEqual(report["perspectives"]["internist"]["claims"], ["internist role claim."])
            self.assertEqual(report["perspectives"]["endocrinologist"]["claims"], ["endocrinologist role claim."])
            self.assertEqual(report["chair_summary"], "Endocrine risk needs specialist follow-up.")
            self.assertEqual(report["disagreements"], ["internist vs endocrinologist: urgency differs."])
            prompt_payload = json.dumps(model.calls, ensure_ascii=False)
            self.assertIn("[PATIENT_1]", prompt_payload)
            self.assertNotIn("Anna Mueller", prompt_payload)
            self.assertNotIn("Hauptstrasse", prompt_payload)
            self.assertNotIn("Bob Smith", prompt_payload)

    def test_case_review_falls_back_when_model_response_is_not_json(self) -> None:
        class BadModel:
            def build_kwargs(self, messages, system_prompt, tools):
                return {"messages": messages, "system_prompt": system_prompt, "tools": tools}

            def call(self, kwargs):
                return SimpleNamespace(content="not json")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ingest_text(
                "case-1",
                "Patient: Anna Mueller\nProblem: Type 2 diabetes",
                root,
                allow_synthetic=True,
            )

            report = case_review(root, "case-1", model_client=BadModel())

            self.assertEqual(report["model_status"], "fallback")
            self.assertTrue(report["claims"])

    def test_case_review_loads_markdown_role_profiles(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.kwargs = {}

            def build_kwargs(self, messages, system_prompt, tools):
                self.kwargs = {"messages": messages, "system_prompt": system_prompt, "tools": tools}
                return self.kwargs

            def call(self, kwargs):
                return SimpleNamespace(content='{"claims":["ok"],"open_questions":[]}')

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            roles_dir = root / "roles"
            roles_dir.mkdir()
            (roles_dir / "internist.md").write_text(
                "# Internist\n\nFocus: coordination and medication review.\n",
                encoding="utf-8",
            )
            ingest_text(
                "case-1",
                "Patient: Anna Mueller\nProblem: Type 2 diabetes",
                root,
                allow_synthetic=True,
            )
            model = FakeModel()

            report = case_review(
                root,
                "case-1",
                roles=["internist"],
                roles_dir=roles_dir,
                model_client=model,
            )

            self.assertEqual(report["role_profiles"]["internist"]["title"], "Internist")
            prompt_payload = json.dumps(model.kwargs, ensure_ascii=False)
            self.assertIn("coordination and medication review", prompt_payload)
            self.assertNotIn("Anna Mueller", prompt_payload)


if __name__ == "__main__":
    unittest.main()
