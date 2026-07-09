from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Callable


@dataclass(frozen=True)
class DeidentifiedDocument:
    text: str
    vault: dict[str, str]


@dataclass(frozen=True)
class PiiEntity:
    kind: str
    value: str


PiiDetector = Callable[[str], list[PiiEntity]]


_FIELD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("PATIENT", re.compile(r"\b(?:Patient|Name|Patientin|Patient name):\s*([^\n,;]+)", re.I)),
    ("DOB", re.compile(r"\b(?:DOB|Date of birth|Geboren|Geburtsdatum):\s*([0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})", re.I)),
    ("ADDR", re.compile(r"\b(?:Address|Adresse):\s*([^\n]+)", re.I)),
    (
        "INSURANCE",
        re.compile(
            r"\b(?:Insurance|Versicherung|Versichertennummer|Versichertennr|Versicherten-Nr|KVNR):?\s*([A-Z0-9 -]{6,})",
            re.I,
        ),
    ),
    ("INSURANCE", re.compile(r"\b([A-Z][0-9]{9})\b")),
    ("EMAIL", re.compile(r"\b([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\b", re.I)),
    (
        "PHONE",
        re.compile(
            r"\b((?!\d{4}-\d{2}-\d{2}\b)(?!\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b)"
            r"\+?[0-9][0-9 ()/-]{7,}[0-9])\b"
        ),
    ),
)


def deidentify(
    text: str,
    existing_vault: dict[str, str] | None = None,
    *,
    pii_detector: PiiDetector | None = None,
    today: date | None = None,
) -> DeidentifiedDocument:
    vault = dict(existing_vault or {})
    counts: dict[str, int] = _counts(vault)
    today = today or date.today()

    def token(kind: str, value: str) -> str:
        for existing, stored in vault.items():
            if stored == value:
                return existing
        counts[kind] = counts.get(kind, 0) + 1
        new_token = f"[{kind}_{counts[kind]}]"
        vault[new_token] = value
        return new_token

    clean = text
    for kind, pattern in _FIELD_PATTERNS:
        clean = pattern.sub(
            lambda match: _replace_match(match, kind, token, today),
            clean,
        )
    if pii_detector is not None:
        for entity in pii_detector(clean):
            kind = _normal_kind(entity.kind)
            value = entity.value.strip()
            if kind and value:
                clean = clean.replace(value, token(kind, value))

    return DeidentifiedDocument(text=clean, vault=vault)


def _replace_match(match: re.Match[str], kind: str, token, today: date) -> str:
    value = match.group(1).strip()
    if kind == "DOB":
        token(kind, value)
        replacement = f"age {_age(value, today)}"
    else:
        replacement = token(kind, value)
    return match.group(0).replace(match.group(1), replacement)


def _age(value: str, today: date) -> int:
    day, month, year = _parse_dob(value)
    birthday_passed = (today.month, today.day) >= (month, day)
    return today.year - year - (0 if birthday_passed else 1)


def _parse_dob(value: str) -> tuple[int, int, int]:
    parts = re.split(r"[./-]", value)
    day = int(parts[0])
    month = int(parts[1])
    year = int(parts[2])
    if year < 100:
        year += 1900 if year > 30 else 2000
    return day, month, year


def _normal_kind(kind: str) -> str:
    aliases = {
        "PERSON": "PATIENT",
        "NAME": "PATIENT",
        "PATIENT": "PATIENT",
        "ADDRESS": "ADDR",
        "ADDR": "ADDR",
        "DOB": "DOB",
        "INSURANCE": "INSURANCE",
        "EMAIL": "EMAIL",
        "PHONE": "PHONE",
    }
    return aliases.get(kind.strip().upper(), "")


def _counts(vault: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in vault:
        match = re.fullmatch(r"\[([A-Z]+)_(\d+)\]", key)
        if match:
            counts[match.group(1)] = max(counts.get(match.group(1), 0), int(match.group(2)))
    return counts
