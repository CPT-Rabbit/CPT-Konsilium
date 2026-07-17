from __future__ import annotations

import unittest

from konsilium.ingest import _structure_document


class StructuringFallbackTest(unittest.TestCase):
    def test_falls_back_to_prepass_when_model_errors(self) -> None:
        # A transient structuring failure (e.g. gateway 408) must degrade to the
        # deterministic prepass, not abort the ingest.
        class Boom:
            def build_kwargs(self, *args, **kwargs):
                return {}

            def call(self, *args, **kwargs):
                raise RuntimeError("gateway 408")

        out = _structure_document(
            "Problem: Type 2 diabetes\n2026-07-01 HbA1c 8.1",
            structure_model=Boom(),
        )
        self.assertIn("Type 2 diabetes", " ".join(out["problems"]))
        self.assertIn("2026-07-01", " ".join(out["timeline"]))


class TransientStatusTest(unittest.TestCase):
    def test_408_is_retryable_but_404_is_not(self) -> None:
        try:
            from konsilium.model_client import _TRANSIENT_STATUS
        except ModuleNotFoundError as error:
            raise unittest.SkipTest(f"model runtime unavailable: {error.name}") from error
        self.assertIn(408, _TRANSIENT_STATUS)
        self.assertNotIn(404, _TRANSIENT_STATUS)


if __name__ == "__main__":
    unittest.main()
