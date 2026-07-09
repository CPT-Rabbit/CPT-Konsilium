from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from konsilium.config import Config
from konsilium.mcp_server import MCP_TOOL_NAMES, KonsiliumOps


class McpServerTest(unittest.TestCase):
    def test_mcp_surface_does_not_return_vault_contents(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "synthetic.txt"
            source.write_text(
                "Patient: Anna Mueller\n2026-07-01 HbA1c 8.1\nProblem: Type 2 diabetes",
                encoding="utf-8",
            )
            ops = KonsiliumOps(Config.model_validate({"runtime": {"patient_root": root}}))

            ingest = ops.ingest_document("case-1", str(source), synthetic=True)
            preview = ops.deid_preview(str(source))
            draft = ops.doctor_letter("case-1")
            vault = json.loads((root / "identity_vault" / "case-1.json").read_text(encoding="utf-8"))

            self.assertNotIn("letter-render", MCP_TOOL_NAMES)
            self.assertIn("deid_preview", MCP_TOOL_NAMES)
            self.assertEqual(ingest["extraction"]["kind"], "text")
            self.assertNotIn("Anna Mueller", json.dumps(ingest["extraction"], ensure_ascii=False))
            self.assertNotIn("Anna Mueller", json.dumps(preview, ensure_ascii=False))
            self.assertIn("Anna Mueller", Path(preview["vault_path"]).read_text(encoding="utf-8"))
            self.assertIn("[PATIENT_1]", draft["content"])
            self.assertNotIn("Anna Mueller", draft["content"])
            with self.assertRaises(ValueError):
                ops.memory_get("identity_vault/case-1.json")
            self.assertEqual(vault["[PATIENT_1]"], "Anna Mueller")


if __name__ == "__main__":
    unittest.main()
