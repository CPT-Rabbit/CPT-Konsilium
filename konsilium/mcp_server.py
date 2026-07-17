from __future__ import annotations

from pathlib import Path

from .config import Config

MCP_TOOL_NAMES = {
    "ingest_document",
    "deid_preview",
    "case_review",
    "doctor_letter",
    "memory_search",
    "memory_get",
    "monitor_review",
    "list_patients",
}


class KonsiliumOps:
    def __init__(self, config: Config):
        self.config = config
        self.root = config.runtime.patient_root

    def ingest_document(self, patient_id: str, path: str, synthetic: bool = False) -> dict:
        from .ingest import extract_text_with_stats, ingest_patient_file, ingest_text

        if synthetic:
            extracted = extract_text_with_stats(path)
            document_path = ingest_text(patient_id, extracted.text, self.root, allow_synthetic=True)
            stats = extracted.stats
        else:
            document_path, stats = ingest_patient_file(
                self.config,
                patient_id,
                path,
                self.root,
            )
        return {
            "patient_id": patient_id,
            "patient_dir": str(document_path.parents[1]),
            "document_path": str(document_path),
            "extraction": stats,
        }

    def deid_preview(self, path: str) -> dict:
        from .ingest import deid_preview

        return deid_preview(self.config, path, self.root)

    def case_review(
        self,
        patient_id: str,
        roles: list[str] | str | None = None,
        question: str | None = None,
    ) -> dict:
        from .review import case_review

        return case_review(
            self.root,
            patient_id,
            roles=_roles(roles),
            question=question,
            model_client=_model_client(self.config),
            residue_policy=self.config.deidentification.residue,
        )

    def doctor_letter(self, patient_id: str) -> dict:
        from .letters import doctor_letter

        path = doctor_letter(self.root, patient_id)
        return {"path": str(path), "content": path.read_text(encoding="utf-8")}

    def memory_search(self, patient_id: str, query: str) -> list[dict]:
        from .memory import PatientMemory

        return PatientMemory(self.root).search(query, patient_id=patient_id)

    def memory_get(self, path: str) -> str:
        from .memory import PatientMemory

        # Single source of truth for the patients/-only guard lives in the primitive.
        return PatientMemory(self.root).get(path)

    def monitor_review(self, patient_ids: list[str] | str) -> dict:
        from .monitor import monitor_review

        return monitor_review(self.root, _roles(patient_ids) or [])

    def list_patients(self) -> list[str]:
        patients = self.root / "patients"
        if not patients.exists():
            return []
        return sorted(path.name for path in patients.iterdir() if path.is_dir())


def run_stdio(config: Config) -> None:
    from mcp.server.fastmcp import FastMCP

    ops = KonsiliumOps(config)
    server = FastMCP("konsilium")

    server.tool()(ops.ingest_document)
    server.tool()(ops.deid_preview)
    server.tool()(ops.case_review)
    server.tool()(ops.doctor_letter)
    server.tool()(ops.memory_search)
    server.tool()(ops.memory_get)
    server.tool()(ops.monitor_review)
    server.tool()(ops.list_patients)
    server.run(transport="stdio")


def _model_client(config: Config):
    from .model_client import ModelClient
    from .providers.base import build_provider

    return ModelClient(
        build_provider(config.model),
        request_timeout_s=config.model.request_timeout_s,
        stream=config.model.stream,
    )


def _roles(value: list[str] | str | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]
