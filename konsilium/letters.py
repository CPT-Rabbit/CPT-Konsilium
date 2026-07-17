from __future__ import annotations

import json
from datetime import date
from pathlib import Path

# Letters are drafted with identity tokens only (never PII on disk); the
# deterministic renderer substitutes real values from the central per-patient
# vault at the very end, into stdout only. Layout follows DIN 5008 (German
# business correspondence): recipient block, right place/date line, a Betreff
# without the word "Betreff", salutation, body, Grußformel, signature.

_CHANNELS = ("paper", "email")


def doctor_letter(
    root: str | Path,
    patient_id: str,
    *,
    language: str = "de",
    channel: str = "paper",
    recipient: str | None = None,
    subject: str | None = None,
    body: str | None = None,
    sender: str | None = None,
    place: str = "[ORT]",
    today: date | None = None,
) -> Path:
    if language != "de":
        raise ValueError("only German doctor letters are supported")
    if channel not in _CHANNELS:
        raise ValueError(f"unknown letter channel: {channel}")
    patient_dir = Path(root) / "patients" / patient_id
    letters_dir = patient_dir / "letters"
    letters_dir.mkdir(exist_ok=True)
    tokens = _vault(Path(root), patient_id)
    patient = _first(tokens, "PATIENT")
    subject = subject or f"Ärztliche Einschätzung zu {patient}"
    body = body or _default_body(patient_dir, patient)
    draft = _render_template(
        channel,
        sender=sender or "[ABSENDER]",
        recipient=recipient or "[EMPFÄNGER]",
        place=place,
        letter_date=(today or date.today()).strftime("%d.%m.%Y"),
        subject=subject,
        patient=patient,
        body=body.strip(),
    )
    path = letters_dir / f"doctor_letter_de_{channel}.md"
    path.write_text(draft, encoding="utf-8")
    return path


def render_doctor_letter(root: str | Path, patient_id: str, draft: str) -> str:
    """Deterministic local substitution: every [token] -> real value from the
    patient's central vault. Longest tokens first so [1_PATIENT_10] is replaced
    before [1_PATIENT_1]. Result is returned, never written to disk."""
    rendered = draft
    for token, value in sorted(_vault(Path(root), patient_id).items(), key=lambda kv: -len(kv[0])):
        rendered = rendered.replace(token, value)
    return rendered


def _render_template(channel: str, **fields: str) -> str:
    return (_EMAIL if channel == "email" else _PAPER).format(**fields)


def _first(tokens: dict[str, str], kind: str) -> str:
    import re

    from .deid import _token_kind

    def index(token: str) -> int:
        match = re.search(r"_(\d+)\]$", token)
        return int(match.group(1)) if match else 0

    matches = sorted((key for key in tokens if _token_kind(key) == kind), key=index)
    return matches[0] if matches else f"[{kind}_1]"


def _default_body(patient_dir: Path, patient: str) -> str:
    problems = _read(patient_dir / "problems.md").strip()
    return (
        f"wir betreuen {patient} und bitten um Ihre ärztliche Einschätzung.\n\n"
        f"Zusammenfassung der bekannten Befunde:\n\n{problems}\n\n"
        "Über eine Rückmeldung zu weiterführender Diagnostik würden wir uns freuen."
    )


def _vault(root: Path, patient_id: str) -> dict[str, str]:
    path = root / "identity_vault" / f"{patient_id}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


# DIN 5008 Geschäftsbrief (Form B, Fensterumschlag): recipient block top-left,
# place/date right-aligned, Betreff in bold without the label word.
_PAPER = (
    "{sender}\n\n"
    "{recipient}\n\n"
    "{place}, {letter_date}\n\n"
    "**{subject}**\n\n"
    "Sehr geehrte Damen und Herren,\n\n"
    "{body}\n\n"
    "Mit freundlichen Grüßen\n\n\n"
    "{sender}\n\n"
    "---\n"
    "Dieses Schreiben ist ein vorbereitender Entwurf und ersetzt keine ärztliche Beratung.\n"
)

# DIN 5008 for e-mail: subject in the header line, no address block, compact
# signature. The Betreff line becomes the mail subject.
_EMAIL = (
    "Betreff: {subject}\n\n"
    "Sehr geehrte Damen und Herren,\n\n"
    "{body}\n\n"
    "Mit freundlichen Grüßen\n"
    "{sender}\n\n"
    "---\n"
    "Dieses Schreiben ist ein vorbereitender Entwurf und ersetzt keine ärztliche Beratung.\n"
)
