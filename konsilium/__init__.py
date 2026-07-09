"""Konsilium stage-1 bootstrap."""

from .deid import DeidentifiedDocument, PiiEntity, ResidueError, deidentify, residue_report
from .egress import EgressViolation, assert_safe_knowledge_query
from .ingest import deid_preview, ingest_document, ingest_patient_document, ingest_patient_file, ingest_text
from .knowledge import guidelines_lookup, pubmed_search, semanticscholar_search
from .letters import doctor_letter, render_doctor_letter
from .monitor import monitor_review
from .ollama_deid import OllamaPiiDetector
from .review import case_review
from .smoke import stage1_smoke

__all__ = [
    "DeidentifiedDocument",
    "EgressViolation",
    "OllamaPiiDetector",
    "PiiEntity",
    "ResidueError",
    "assert_safe_knowledge_query",
    "case_review",
    "deidentify",
    "deid_preview",
    "doctor_letter",
    "guidelines_lookup",
    "ingest_document",
    "ingest_patient_document",
    "ingest_patient_file",
    "ingest_text",
    "monitor_review",
    "pubmed_search",
    "render_doctor_letter",
    "residue_report",
    "semanticscholar_search",
    "stage1_smoke",
]
