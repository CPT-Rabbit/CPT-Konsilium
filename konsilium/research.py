"""Delegation payloads for an external research agent."""

from __future__ import annotations


def paper_deepen_request(patient_id: str, doi: str) -> dict:
    return {"target_agent": "research-agent", "tool": "paper_deepen", "patient_id": patient_id, "doi": doi}


def retraction_sweep_request(patient_id: str, dois: list[str]) -> dict:
    return {"target_agent": "research-agent", "tool": "retraction_sweep", "patient_id": patient_id, "dois": dois}
