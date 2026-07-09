from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from konsilium import (
    EgressViolation,
    assert_safe_knowledge_query,
    deidentify,
    ingest_document,
    ingest_patient_file,
    ingest_patient_document,
    ingest_text,
    stage1_smoke,
)
from konsilium.config import Config
from konsilium.deid import PiiEntity
from konsilium.ingest import _ollama_detector, extract_text_with_stats
from konsilium.ollama_deid import OllamaPiiDetector


class Stage1Test(unittest.TestCase):
    def test_ingest_keeps_pii_out_of_patient_memory(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            patient_dir = ingest_text(
                "case-1",
                "\n".join(
                    [
                        "Patient: Anna Mueller",
                        "Geburtsdatum: 12.03.1974",
                        "Adresse: Hauptstrasse 7, Berlin",
                        "KVNR: X123456789",
                        "Visit date: 2026-07-01",
                        "HbA1c elevated, metformin discussed.",
                    ]
                ),
                root,
                allow_synthetic=True,
            )

            patient_text = (patient_dir / "documents.md").read_text(encoding="utf-8")
            self.assertIn("[PATIENT_1]", patient_text)
            self.assertRegex(patient_text, r"age \d+")
            self.assertNotIn("[DOB_1]", patient_text)
            self.assertIn("Visit date: 2026-07-01", patient_text)
            self.assertNotIn("Anna Mueller", patient_text)
            self.assertNotIn("12.03.1974", patient_text)
            self.assertNotIn("Hauptstrasse", patient_text)

            vault = json.loads((root / "identity_vault" / "case-1.json").read_text(encoding="utf-8"))
            self.assertEqual(vault["[PATIENT_1]"], "Anna Mueller")
            self.assertEqual(vault["[DOB_1]"], "12.03.1974")

    def test_regex_deid_catches_german_insurance_numbers_without_model(self) -> None:
        document = deidentify(
            "\n".join(
                [
                    "Versichertennummer A987654321",
                    "Die Karte A987654321 wurde vorgelegt.",
                    "HbA1c 8.1, LDL 140, CRP 5",
                ]
            )
        )

        self.assertIn("[INSURANCE_1]", document.text)
        self.assertEqual(document.text.count("[INSURANCE_1]"), 2)
        self.assertNotIn("A987654321", document.text)
        self.assertIn("HbA1c 8.1", document.text)
        self.assertIn("LDL 140", document.text)
        self.assertEqual(document.vault["[INSURANCE_1]"], "A987654321")

    def test_egress_guard_rejects_tokens_and_raw_pii(self) -> None:
        assert_safe_knowledge_query("metformin hba1c older adults guideline")

        with self.assertRaises(EgressViolation):
            assert_safe_knowledge_query("metformin for [PATIENT_1]")

        with self.assertRaises(EgressViolation):
            assert_safe_knowledge_query("diabetes Geburtsdatum: 12.03.1974")

    def test_ingest_appends_documents_and_preserves_existing_tokens(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ingest_text(
                "case-1",
                "Patient: Anna Mueller\n2026-07-01 HbA1c 8.1",
                root,
                allow_synthetic=True,
            )
            ingest_text(
                "case-1",
                "Patient: Anna Mueller\n2026-07-08 LDL 140",
                root,
                allow_synthetic=True,
            )

            patient_text = (root / "patients" / "case-1" / "documents.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("2026-07-01 HbA1c 8.1", patient_text)
            self.assertIn("2026-07-08 LDL 140", patient_text)
            self.assertEqual(patient_text.count("[PATIENT_1]"), 2)

            vault = json.loads((root / "identity_vault" / "case-1.json").read_text())
            self.assertEqual(vault, {"[PATIENT_1]": "Anna Mueller"})

    def test_ingest_document_uses_pdf_extractor_hook(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "synthetic.pdf"
            pdf.write_bytes(b"%PDF synthetic fixture")

            patient_dir = ingest_document(
                "case-1",
                pdf,
                root,
                extractor=lambda path: "Patient: Anna Mueller\n2026-07-01 HbA1c 8.1",
                allow_synthetic=True,
            )

            patient_text = (patient_dir / "documents.md").read_text(encoding="utf-8")
            self.assertIn("[PATIENT_1]", patient_text)
            self.assertNotIn("Anna Mueller", patient_text)

    def test_pdf_extraction_tracks_text_layer_ocr_and_mixed_pages(self) -> None:
        class Page:
            def __init__(self, text: str):
                self.text = text

            def extract_text(self) -> str:
                return self.text

        def reader(path: Path):
            if path.name == "text.pdf":
                return [Page("Patient: Anna Mueller\nProblem: Diabetes")]
            if path.name == "scan.pdf":
                return [Page("")]
            return [Page("Patient: Anna Mueller"), Page("")]

        def ocr(path: Path, pages: list[int]) -> dict[int, str]:
            return {page: f"OCR page {page}: Anna Mueller HbA1c 8.1" for page in pages}

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("text.pdf", "scan.pdf", "mixed.pdf"):
                (root / name).write_bytes(b"%PDF fixture")

            text = extract_text_with_stats(root / "text.pdf", pdf_reader=reader, ocr_pages=ocr)
            scan = extract_text_with_stats(root / "scan.pdf", pdf_reader=reader, ocr_pages=ocr)
            mixed = extract_text_with_stats(root / "mixed.pdf", pdf_reader=reader, ocr_pages=ocr)

            self.assertEqual(text.stats["text_layer_pages"], [1])
            self.assertEqual(text.stats["ocr_pages"], [])
            self.assertEqual(scan.stats["text_layer_pages"], [])
            self.assertEqual(scan.stats["ocr_pages"], [1])
            self.assertEqual(mixed.stats["text_layer_pages"], [1])
            self.assertEqual(mixed.stats["ocr_pages"], [2])
            self.assertIn("OCR page 2", mixed.text)

    def test_pdf_ocr_output_is_deidentified_and_empty_pages_fail_loudly(self) -> None:
        class Page:
            def extract_text(self) -> str:
                return ""

        def reader(path: Path):
            return [Page()]

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "scan.pdf"
            pdf.write_bytes(b"%PDF fixture")

            def ocr(path: Path, pages: list[int]) -> dict[int, str]:
                return {1: "Patient: Anna Mueller\nProblem: Diabetes"}

            patient_dir = ingest_document(
                "case-1",
                pdf,
                root,
                extractor=lambda path: extract_text_with_stats(path, pdf_reader=reader, ocr_pages=ocr).text,
                allow_synthetic=True,
            )
            patient_text = (patient_dir / "documents.md").read_text(encoding="utf-8")
            self.assertIn("[PATIENT_1]", patient_text)
            self.assertNotIn("Anna Mueller", patient_text)

            with self.assertRaisesRegex(RuntimeError, "pages without text after OCR: 1"):
                extract_text_with_stats(pdf, pdf_reader=reader, ocr_pages=lambda path, pages: {1: ""})

            with self.assertRaisesRegex(RuntimeError, "PDF has no pages"):
                extract_text_with_stats(pdf, pdf_reader=lambda path: [])

            def corrupt_reader(path: Path):
                raise RuntimeError("corrupt PDF")

            with self.assertRaisesRegex(RuntimeError, "corrupt PDF"):
                extract_text_with_stats(pdf, pdf_reader=corrupt_reader)

            with patch("konsilium.ingest.shutil.which", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "OCR required but not available"):
                    extract_text_with_stats(pdf, pdf_reader=reader)

    def test_local_detector_removes_free_text_german_and_english_pii(self) -> None:
        def detector(text: str) -> list[PiiEntity]:
            return [
                PiiEntity("PATIENT", "Frau Mueller"),
                PiiEntity("ADDR", "Hauptstrasse 7"),
                PiiEntity("PATIENT", "John Smith"),
                PiiEntity("ADDR", "742 Evergreen Terrace"),
            ]

        document = deidentify(
            "\n".join(
                [
                    "Frau Mueller berichtet über Kopfschmerzen in Hauptstrasse 7.",
                    "John Smith reports nausea at 742 Evergreen Terrace.",
                ]
            ),
            pii_detector=detector,
        )

        self.assertIn("[PATIENT_1]", document.text)
        self.assertIn("[ADDR_1]", document.text)
        self.assertIn("[PATIENT_2]", document.text)
        self.assertIn("[ADDR_2]", document.text)
        self.assertNotIn("Frau Mueller", document.text)
        self.assertNotIn("Hauptstrasse 7", document.text)
        self.assertNotIn("John Smith", document.text)
        self.assertNotIn("742 Evergreen Terrace", document.text)

    def test_ingest_passes_local_detector_into_deid_pipeline(self) -> None:
        def detector(text: str) -> list[PiiEntity]:
            return [PiiEntity("PATIENT", "Frau Mueller")]

        def structure_model(messages, system_prompt):
            return {"timeline": [], "problems": ["Kopfschmerzen"], "meds": [], "labs": []}

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            patient_dir = ingest_text(
                "case-1",
                "Frau Mueller berichtet über Kopfschmerzen.",
                root,
                pii_detector=detector,
                structure_model=structure_model,
            )

            patient_text = (patient_dir / "documents.md").read_text(encoding="utf-8")
            self.assertIn("[PATIENT_1]", patient_text)
            self.assertNotIn("Frau Mueller", patient_text)

    def test_ingest_without_detector_is_not_implicit_regex_only(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                ingest_text("case-1", "Patient: Anna Mueller", Path(tmp))
            with self.assertRaises(RuntimeError):
                ingest_text("case-1", "Patient: Anna Mueller", Path(tmp), pii_detector=lambda text: [])

    def test_structuring_model_extracts_deidentified_german_and_english_text(self) -> None:
        class FakeStructureModel:
            def __init__(self) -> None:
                self.calls = []

            def build_kwargs(self, messages, system_prompt, tools, *, json_mode=False):
                call = {
                    "messages": messages,
                    "system_prompt": system_prompt,
                    "stream": not json_mode,
                    "max_tokens": 4096,
                }
                if json_mode:
                    call["response_format"] = {"type": "json_object"}
                self.calls.append(call)
                return call

            def call(self, kwargs):
                return SimpleNamespace(content=json.dumps({
                    "timeline": ["2026-07-02 nephrology review"],
                    "problems": ["chronische Nierenkrankheit", "iron deficiency anemia"],
                    "meds": ["Apixaban 5 mg zweimal taeglich", "rivaroxaban 20 mg nightly"],
                    "labs": ["Kreatinin 1.8 mg/dl", "ferritin 12 ng/mL"],
                }))

        def detector(text: str) -> list[PiiEntity]:
            return [
                PiiEntity("PATIENT", "Frau Mueller"),
                PiiEntity("PATIENT", "John Smith"),
                PiiEntity("ADDR", "Hauptstrasse 7"),
            ]

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = FakeStructureModel()
            patient_dir = ingest_text(
                "case-1",
                "\n".join(
                    [
                        "Patient: Anna Mueller",
                        "Frau Mueller wohnt in Hauptstrasse 7.",
                        "2026-07-02 Kreatinin 1.8 mg/dl, Apixaban 5 mg zweimal taeglich.",
                        "John Smith reports ferritin 12 ng/mL and rivaroxaban 20 mg nightly.",
                    ]
                ),
                root,
                pii_detector=detector,
                structure_model=model,
            )

            labs = (patient_dir / "labs" / "labs.md").read_text(encoding="utf-8")
            meds = (patient_dir / "meds.md").read_text(encoding="utf-8")
            problems = (patient_dir / "problems.md").read_text(encoding="utf-8")
            prompt_payload = json.dumps(model.calls, ensure_ascii=False)

            self.assertIn("Kreatinin 1.8 mg/dl", labs)
            self.assertIn("ferritin 12 ng/mL", labs)
            self.assertIn("Apixaban 5 mg", meds)
            self.assertIn("rivaroxaban 20 mg", meds)
            self.assertIn("chronische Nierenkrankheit", problems)
            self.assertIn("iron deficiency anemia", problems)
            self.assertNotIn("Anna Mueller", prompt_payload)
            self.assertNotIn("Frau Mueller", prompt_payload)
            self.assertNotIn("John Smith", prompt_payload)
            self.assertNotIn("Hauptstrasse", prompt_payload)
            self.assertFalse(model.calls[0]["stream"])
            self.assertEqual(model.calls[0]["response_format"], {"type": "json_object"})

    def test_runtime_ingest_requires_real_docs_enabled_model_and_reachable_detector(self) -> None:
        config = Config()
        with TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                ingest_patient_document(config, "case-1", "Patient: Anna Mueller", Path(tmp))

            def extractor(path: Path) -> str:
                raise AssertionError("real file extractor must not run before runtime gate")

            blocked_pdf = Path(tmp) / "blocked.pdf"
            blocked_pdf.write_bytes(b"%PDF fixture")
            with self.assertRaises(RuntimeError):
                ingest_patient_file(config, "case-1", blocked_pdf, Path(tmp), extractor=extractor)

        config = Config.model_validate(
            {
                "runtime": {"allow_real_patient_docs": True},
                "deidentification": {"ollama_model": "local-med-ner"},
            }
        )

        def unreachable(text: str) -> list[PiiEntity]:
            raise RuntimeError("ollama unavailable")

        with TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                ingest_patient_document(
                    config,
                    "case-1",
                    "Patient: Anna Mueller",
                    Path(tmp),
                    pii_detector=unreachable,
                )

        def detector(text: str) -> list[PiiEntity]:
            return [PiiEntity("PATIENT", "Anna Mueller")]

        def empty_structure(messages, system_prompt):
            return {"timeline": [], "problems": [], "meds": [], "labs": []}

        with TemporaryDirectory() as tmp:
            patient_dir = ingest_patient_document(
                config,
                "case-1",
                "Patient: Anna Mueller",
                Path(tmp),
                pii_detector=detector,
                structure_model=empty_structure,
            )
            patient_text = (patient_dir / "documents.md").read_text(encoding="utf-8")
            self.assertIn("[PATIENT_1]", patient_text)
            self.assertNotIn("Anna Mueller", patient_text)

    def test_ollama_detector_parses_local_model_entities(self) -> None:
        calls = []
        shapes = [
            json.dumps({"entities": [{"kind": "PERSON", "value": "Frau Mueller"}]}),
            "```json\n{\"entities\":[{\"kind\":\"ADDRESS\",\"value\":\"Hauptstrasse 7\"}]}\n```",
            json.dumps([{"kind": "EMAIL", "value": "frau@example.test"}]),
            json.dumps({"kind": "PHONE", "value": "+491234"}),
        ]

        def fake_fetch(url: str, payload: dict, timeout_s: float) -> str:
            calls.append((url, payload, timeout_s))
            return json.dumps({"response": shapes.pop(0)})

        detector = OllamaPiiDetector(
            model="local-med-ner",
            base_url="http://127.0.0.1:11434",
            timeout_s=321,
            fetch=fake_fetch,
        )

        entities = []
        for _ in range(4):
            entities.extend(detector("Frau Mueller lebt in Hauptstrasse 7."))

        self.assertEqual(
            entities,
            [
                PiiEntity("PERSON", "Frau Mueller"),
                PiiEntity("ADDRESS", "Hauptstrasse 7"),
                PiiEntity("EMAIL", "frau@example.test"),
                PiiEntity("PHONE", "+491234"),
            ],
        )
        self.assertEqual(calls[0][0], "http://127.0.0.1:11434/api/generate")
        self.assertEqual(calls[0][1]["model"], "local-med-ner")
        self.assertEqual(calls[0][1]["format"]["required"], ["entities"])
        self.assertEqual(calls[0][1]["format"]["properties"]["entities"]["items"]["required"], ["kind", "value"])
        self.assertIn("PERSON includes every named human", calls[0][1]["prompt"])
        self.assertIn("physicians", calls[0][1]["prompt"])
        self.assertIn("Dr. med.", calls[0][1]["prompt"])
        self.assertIs(calls[0][1]["think"], False)
        self.assertEqual(calls[0][1]["options"], {"temperature": 0})
        self.assertEqual({call[2] for call in calls}, {321})

        configured = _ollama_detector(
            Config.model_validate({"deidentification": {"ollama_model": "gemma3:4b", "timeout_s": 123}})
        )
        self.assertEqual(configured.timeout_s, 123)

    def test_stage1_smoke_reports_pii_boundary(self) -> None:
        with TemporaryDirectory() as tmp:
            report = stage1_smoke(Path(tmp))

            self.assertTrue(report["passed"])
            self.assertEqual(report["patient_id"], "synthetic-stage1")
            self.assertTrue(report["vault_tokens"])
            self.assertTrue(report["case_review"]["evidence_refs"])


if __name__ == "__main__":
    unittest.main()
