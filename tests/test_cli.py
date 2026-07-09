from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from konsilium.__main__ import main


class CliTest(unittest.TestCase):
    def test_operator_synthetic_flow(self) -> None:
        class BadModel:
            def build_kwargs(self, messages, system_prompt, tools, *, json_mode=False):
                return {}

            def call(self, kwargs):
                return SimpleNamespace(content="not json")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "model": {"base_url": "https://gateway.test/compat"},
                        "runtime": {"patient_root": str(root / "memory")},
                        "auth": {"default_path": str(root / "auth.json")},
                    }
                ),
                encoding="utf-8",
            )
            source = root / "letter.txt"
            source.write_text(
                "Patient: Anna Mueller\n2026-07-01 HbA1c 8.1\nProblem: Type 2 diabetes",
                encoding="utf-8",
            )

            ingest_out = _run("--config", config, "ingest", "--patient", "case-1", "--file", source, "--synthetic")
            with patch("konsilium.__main__._model_client", return_value=BadModel()):
                review_out = _run("--config", config, "review", "--patient", "case-1", "--roles", "internist")
            letter_out = _run("--config", config, "letter", "--patient", "case-1", "--language", "de")
            rendered = _run("--config", config, "letter-render", "--patient", "case-1", "--file", letter_out.strip())
            monitor_out = _run("--config", config, "monitor", "--patients", "case-1")
            memory_out = _run(
                "--config",
                config,
                "memory-search",
                "--patient",
                "case-1",
                "--query",
                "HbA1c diabetes",
            )

            self.assertEqual(json.loads(ingest_out)["extraction"]["kind"], "text")
            self.assertIn("report=", review_out)
            self.assertIn("Chair synthesis", review_out)
            self.assertIn("Anna Mueller", rendered)
            self.assertIn("report=", monitor_out)
            self.assertIn("documents.md", memory_out)
            self.assertNotIn("Anna Mueller", memory_out)


def _run(*args) -> str:
    out = io.StringIO()
    with redirect_stdout(out):
        main([str(arg) for arg in args])
    return out.getvalue()


if __name__ == "__main__":
    unittest.main()
