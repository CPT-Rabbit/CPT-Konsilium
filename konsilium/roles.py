from __future__ import annotations

from pathlib import Path


def load_role_profiles(role_names: list[str], roles_dir: str | Path = "roles") -> dict[str, dict]:
    roles_path = Path(roles_dir)
    profiles = {}
    for name in role_names:
        path = roles_path / f"{name}.md"
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        profiles[name] = {
            "title": _title(text) or name,
            "content": text,
            "path": str(path) if path.exists() else None,
        }
    return profiles


def _title(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""
