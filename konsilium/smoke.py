from __future__ import annotations

import json
from pathlib import Path

from .ingest import ingest_text
from .knowledge import SearchResult
from .knowledge import guidelines_lookup, pubmed_search, semanticscholar_search
from .review import case_review


def stage1_smoke(root: str | Path) -> dict:
    root = Path(root)
    patient_id = "synthetic-stage1"
    document_path = ingest_text(
        patient_id,
        "\n".join(
            [
                "Patient: Anna Mueller",
                "Geburtsdatum: 12.03.1974",
                "Adresse: Hauptstrasse 7, Berlin",
                "KVNR: X123456789",
                "2026-07-01 HbA1c 8.1, started Metformin 500 mg.",
                "Problem: Type 2 diabetes",
            ]
        ),
        root,
        allow_synthetic=True,
    )
    patient_dir = document_path.parents[1]
    patient_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in patient_dir.rglob("*.md")
        if "identity_vault" not in path.parts
    )
    vault_path = root / "identity_vault" / f"{patient_id}.json"
    vault = json.loads(vault_path.read_text(encoding="utf-8"))
    leaked_values = [value for value in vault.values() if value and value in patient_text]
    report = case_review(
        root,
        patient_id,
        roles=["internist"],
        knowledge=[SearchResult("Synthetic guideline evidence", "smoke", "local://stage1")],
    )
    return {
        "passed": not leaked_values and bool(vault),
        "patient_id": patient_id,
        "patient_dir": str(patient_dir),
        "document_path": str(document_path),
        "vault_tokens": sorted(vault),
        "leaked_values": leaked_values,
        "case_review": report,
    }


def knowledge_smoke(query: str) -> dict:
    calls = []

    def fake_fetch(url: str, headers: dict | None = None) -> str:
        calls.append({"url": url, "headers": headers or {}})
        if "semanticscholar" in url:
            return '{"data":[{"title":"Synthetic Semantic Scholar result","url":"local://s2"}]}'
        return '{"esearchresult":{"idlist":["0"]}}'

    pubmed = pubmed_search(query, fetch=fake_fetch)
    semantic = semanticscholar_search(query, fetch=fake_fetch)
    guidelines = guidelines_lookup(query)
    return {
        "passed": bool(pubmed and semantic and guidelines),
        "sources": sorted({pubmed[0].source, semantic[0].source, guidelines[0].source}),
        "calls": calls,
    }
