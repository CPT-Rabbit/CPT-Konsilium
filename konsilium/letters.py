from __future__ import annotations

import json
from pathlib import Path


def doctor_letter(root: str | Path, patient_id: str, *, language: str = "de") -> Path:
    if language != "de":
        raise ValueError("only German doctor letters are supported")
    patient_dir = Path(root) / "patients" / patient_id
    letters_dir = patient_dir / "letters"
    letters_dir.mkdir(exist_ok=True)
    tokens = _vault(Path(root), patient_id)
    patient = next((key for key in tokens if key.startswith("[PATIENT_")), "[PATIENT_1]")
    address = next((key for key in tokens if key.startswith("[ADDR_")), "[ADDR_1]")
    problems = _read(patient_dir / "problems.md")
    draft = _TEMPLATE.format(patient=patient, address=address, problems=problems)
    path = letters_dir / "doctor_letter_de.md"
    path.write_text(draft, encoding="utf-8")
    return path


def render_doctor_letter(root: str | Path, patient_id: str, draft: str) -> str:
    rendered = draft
    for token, value in _vault(Path(root), patient_id).items():
        rendered = rendered.replace(token, value)
    return rendered


def _vault(root: Path, patient_id: str) -> dict[str, str]:
    path = root / "identity_vault" / f"{patient_id}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


_TEMPLATE = (
    "Sehr geehrte Damen und Herren,\n\n"
    "ich bitte um ärztliche Einschätzung für {patient}, {address}.\n\n"
    "Zusammenfassung:\n{problems}\n\n"
    "Dieses Schreiben ist ein vorbereitender Entwurf und ersetzt keine ärztliche Beratung.\n"
)
