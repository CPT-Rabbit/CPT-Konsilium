from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .knowledge import SearchResult, guidelines_lookup
from .memory import PatientMemory
from .roles import load_role_profiles
from .util import json_block

_ROLE_FOCUS = {
    "internist": "overall differential and primary-care coordination",
    "endocrinologist": "metabolic risk, diabetes and hormone-related patterns",
    "neurologist": "neurologic symptoms and escalation questions",
}


def case_review(
    root: str | Path,
    patient_id: str,
    *,
    roles: list[str] | None = None,
    question: str | None = None,
    knowledge: list[SearchResult] | None = None,
    model_client=None,
    roles_dir: str | Path = "roles",
) -> dict:
    root = Path(root)
    patient_dir = root / "patients" / patient_id
    selected_roles = roles or ["internist"]
    role_profiles = load_role_profiles(selected_roles, roles_dir)
    evidence = _patient_evidence(patient_dir)
    external = knowledge if knowledge is not None else guidelines_lookup(question or _first_problem(patient_dir))
    if model_client is not None:
        try:
            report = _model_report(
                model_client,
                patient_dir,
                patient_id,
                selected_roles,
                question,
                evidence,
                external,
                role_profiles,
            )
        except Exception as error:  # noqa: BLE001 - model path must not break local review
            report = _deterministic_report(
                patient_id,
                selected_roles,
                question,
                evidence,
                external,
                role_profiles,
            )
            report["model_status"] = "fallback"
            report["model_error"] = error.__class__.__name__
        _write_report(patient_dir, report)
        return report

    report = _deterministic_report(
        patient_id,
        selected_roles,
        question,
        evidence,
        external,
        role_profiles,
    )
    _write_report(patient_dir, report)
    return report


def _deterministic_report(
    patient_id: str,
    selected_roles: list[str],
    question: str | None,
    evidence: list[dict],
    external: list[SearchResult],
    role_profiles: dict[str, dict],
) -> dict:
    perspectives = _deterministic_perspectives(patient_id, selected_roles)
    claims = [
        claim
        for perspective in perspectives.values()
        for claim in perspective["claims"]
    ]
    return {
        "artifact_type": "consilium_report",
        "patient_id": patient_id,
        "created_at": datetime.now(UTC).isoformat(),
        "roles": selected_roles,
        "role_profiles": role_profiles,
        "perspectives": perspectives,
        "chair_summary": _chair_summary(perspectives),
        "claims": claims,
        "evidence_refs": evidence + [item.to_dict() for item in external],
        "open_questions": [question or "Which next clinical question should the physician resolve?"],
        "disagreements": [],
        "recommended_next_step": "Review this report with a licensed physician.",
        "model_status": "not_used",
    }


def _model_report(
    model_client,
    patient_dir: Path,
    patient_id: str,
    selected_roles: list[str],
    question: str | None,
    evidence: list[dict],
    external: list[SearchResult],
    role_profiles: dict[str, dict],
) -> dict:
    memory = _patient_memory(patient_dir, patient_id, question or _first_problem(patient_dir))
    external_evidence = [item.to_dict() for item in external]
    perspectives = {}
    for role in selected_roles:
        payload = _model_json(
            model_client,
            [
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "patient_id": patient_id,
                            "role": role,
                            "role_profile": role_profiles.get(role, {}),
                            "question": question,
                            "patient_memory": memory,
                            "external_evidence": external_evidence,
                        },
                        ensure_ascii=False,
                    ),
                }
            ],
            (
                f"Role perspective: {role}. Return strict JSON with claims and open_questions. "
                "Do not diagnose or prescribe."
            ),
        )
        claims = payload.get("claims") or []
        open_questions = payload.get("open_questions") or []
        if not isinstance(claims, list) or not isinstance(open_questions, list):
            raise ValueError("model role response has invalid fields")
        perspectives[role] = {
            "claims": [str(item) for item in claims],
            "open_questions": [str(item) for item in open_questions],
        }

    chair = _model_json(
        model_client,
        [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "patient_id": patient_id,
                        "roles": selected_roles,
                        "role_profiles": role_profiles,
                        "perspectives": perspectives,
                        "question": question,
                        "external_evidence": external_evidence,
                    },
                    ensure_ascii=False,
                ),
            }
        ],
        (
            "Chair synthesis. Return strict JSON with chair_summary, claims, "
            "open_questions, disagreements, and recommended_next_step. "
            "Disagreements must reflect actual divergence between role outputs; use [] if none."
        ),
    )
    claims = chair.get("claims") or [
        claim
        for perspective in perspectives.values()
        for claim in perspective.get("claims", [])
    ]
    open_questions = chair.get("open_questions") or []
    disagreements = chair.get("disagreements") or []
    if not isinstance(claims, list) or not isinstance(open_questions, list):
        raise ValueError("model chair response has invalid report fields")
    if not isinstance(disagreements, list):
        raise ValueError("model chair response has invalid disagreements")
    return {
        "artifact_type": "consilium_report",
        "patient_id": patient_id,
        "created_at": datetime.now(UTC).isoformat(),
        "roles": selected_roles,
        "role_profiles": role_profiles,
        "perspectives": perspectives,
        "chair_summary": str(chair.get("chair_summary") or _chair_summary(perspectives)),
        "claims": [str(item) for item in claims],
        "evidence_refs": evidence + [item.to_dict() for item in external],
        "open_questions": [str(item) for item in open_questions],
        "disagreements": [str(item) for item in disagreements],
        "recommended_next_step": str(
            chair.get("recommended_next_step") or "Review this report with a licensed physician."
        ),
        "model_status": "used",
    }


def _model_json(model_client, messages: list[dict], system_prompt: str) -> dict:
    kwargs = model_client.build_kwargs(messages, system_prompt, [], json_mode=True)
    response = model_client.call(kwargs)
    payload = json.loads(json_block(response.content))
    if not isinstance(payload, dict):
        raise ValueError("model response is not an object")
    return payload


def _write_report(patient_dir: Path, report: dict) -> None:
    out = patient_dir / "consilium"
    out.mkdir(exist_ok=True)
    (out / "latest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def _deterministic_perspectives(patient_id: str, roles: list[str]) -> dict[str, dict]:
    return {
        role: {
            "claims": [
                f"{role}: review {patient_id} for {_ROLE_FOCUS.get(role, 'the assigned clinical perspective')}."
            ],
            "open_questions": ["What should the treating physician clarify next?"],
        }
        for role in roles
    }


def _chair_summary(perspectives: dict[str, dict]) -> str:
    roles = ", ".join(perspectives) or "no roles"
    return f"Chair synthesis required across: {roles}."


def _patient_memory(patient_dir: Path, patient_id: str, query: str) -> dict[str, str]:
    root = patient_dir.parents[1]
    memory = PatientMemory(root)
    hits = memory.search(query, patient_id=patient_id, limit=5)
    return {hit["path"]: memory.get(hit["path"]) for hit in hits}


def _patient_evidence(patient_dir: Path) -> list[dict]:
    refs = []
    for path in sorted((patient_dir / "documents").glob("*.md")):
        name = str(path.relative_to(patient_dir))
        refs.append({"source": f"patient_memory:{name}", "url": str(path)})
    for name in ("timeline/events.md", "problems.md", "meds.md", "labs/labs.md"):
        path = patient_dir / name
        if path.exists():
            refs.append({"source": f"patient_memory:{name}", "url": str(path)})
    return refs


def _first_problem(patient_dir: Path) -> str:
    problems = patient_dir / "problems.md"
    if not problems.exists():
        return "clinical guideline"
    for line in problems.read_text(encoding="utf-8").splitlines():
        if line.startswith("- ") and "No structured" not in line:
            return line[2:]
    return "clinical guideline"
