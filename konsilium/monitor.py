from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


def monitor_review(root: str | Path, patient_ids: list[str]) -> dict:
    root = Path(root)
    patients = []
    for patient_id in patient_ids:
        patient_dir = root / "patients" / patient_id
        patients.append(
            {
                "patient_id": patient_id,
                "latest_timeline": _read(patient_dir / "timeline" / "events.md"),
                "strategy_path": str(patient_dir / "strategy.md"),
            }
        )
    report = {"created_at": datetime.now(UTC).isoformat(), "patients": patients}
    monitor_dir = root / "monitor"
    monitor_dir.mkdir(exist_ok=True)
    (monitor_dir / "latest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (root / "monitor_schedule.json").write_text(
        json.dumps(
            {"job": "konsilium.monitor_review", "cadence": "manual_or_scheduled", "patient_ids": patient_ids},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""
