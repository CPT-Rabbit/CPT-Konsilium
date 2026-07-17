from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from konsilium import ingest_text
from konsilium.memory import PatientMemory


class MemorySafetyTest(unittest.TestCase):
    def test_get_confined_to_patients_subtree(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "identity_vault").mkdir(parents=True)
            (root / "identity_vault" / "case-1.json").write_text('{"[PATIENT_1]": "Real Name"}')
            doc = root / "patients" / "case-1" / "documents"
            doc.mkdir(parents=True)
            (doc / "d.md").write_text("clinical body")
            mem = PatientMemory(root)

            self.assertEqual(mem.get("patients/case-1/documents/d.md"), "clinical body")
            with self.assertRaises(ValueError):
                mem.get("identity_vault/case-1.json")  # re-identification vault is unreachable
            with self.assertRaises(ValueError):
                mem.get("../../etc/passwd")

    def test_sync_drops_deleted_documents_from_search(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ingest_text(
                "case-1",
                "Problem: Type 2 diabetes\n2026-07-01 HbA1c 8.1",
                root,
                allow_synthetic=True,
            )
            self.assertTrue(PatientMemory(root).search("diabetes", patient_id="case-1"))

            shutil.rmtree(root / "patients" / "case-1")
            mem = PatientMemory(root)
            mem.sync()
            self.assertEqual(mem.search("diabetes", patient_id="case-1"), [])


class SafeIdTest(unittest.TestCase):
    def test_rejects_dots_only_patient_id(self) -> None:
        from konsilium.ingest import _safe_id

        self.assertEqual(_safe_id("case-1"), "case-1")
        for bad in ("", ".", "..", "..."):
            with self.assertRaises(ValueError):
                _safe_id(bad)


class ModelEgressGuardTest(unittest.TestCase):
    def test_blocks_residual_pii_before_model_send(self) -> None:
        from konsilium.deid import ResidueError
        from konsilium.review import _assert_model_egress_safe

        # De-identified clinical body passes.
        _assert_model_egress_safe({"patients/case-1/documents/d.md": "Problem: diabetes, HbA1c 8.1"})
        # A residual birth marker + date (a de-id regression) must fail-closed before egress.
        with self.assertRaises(ResidueError):
            _assert_model_egress_safe({"patients/case-1/documents/d.md": "geb. 12.08.2015 Diagnose Epilepsie"})


class DisagreementGroundingTest(unittest.TestCase):
    def test_keeps_only_disagreements_attributed_to_contributing_roles(self) -> None:
        from konsilium.review import _ground_disagreements

        perspectives = {
            "internist": {"claims": ["a"]},
            "endocrinologist": {"claims": ["b"]},
            "neurologist": {"claims": []},  # produced no claims
        }
        out = _ground_disagreements(
            [
                "internist vs endocrinologist: urgency differs.",  # both contributing -> keep
                "endocrinologist vs neurologist: dosing.",         # neurologist has no claims -> drop
                "the specialists broadly agree but with nuance",   # names no roles -> drop
                "internist has residual concerns",                 # only one role -> drop
            ],
            perspectives,
        )
        self.assertEqual(out, ["internist vs endocrinologist: urgency differs."])


if __name__ == "__main__":
    unittest.main()
