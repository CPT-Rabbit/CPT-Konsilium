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

DEFAULT_RESIDUE_POLICY = {
    "DOB": "block",
    "STREET": "block",
    "PLZ_CITY": "block",
    "PHONE": "block",
    "DIGIT_RUN": "block",
    "KVNR": "block",
    "CASE_NUMBER": "block",
}


@dataclass(frozen=True)
class ResidueHit:
    line: int
    pattern: str
    action: str


class ResidueError(RuntimeError):
    def __init__(self, hits: list[ResidueHit]):
        self.hits = hits
        grouped: dict[str, list[int]] = {}
        for hit in hits:
            grouped.setdefault(hit.pattern, []).append(hit.line)
        detail = "; ".join(f"{name} lines {','.join(map(str, lines))}" for name, lines in grouped.items())
        super().__init__(f"de-identification residue blocked ingest: {detail}")


_FIELD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("PATIENT", re.compile(r"\b(?:Patient|Name|Patientin|Patient name):\s*([^\n,;]+)", re.I)),
    (
        "DOB",
        re.compile(
            r"\b(?:DOB|Date of birth|Geboren(?:\s+am)?|Geburtsdatum|geb\s*\.?):?\s*"
            r"([0-9]{1,2}\s*[./-]\s*[0-9]{1,2}\s*[./-]\s*[0-9]{2,4})",
            re.I,
        ),
    ),
    ("ADDR", re.compile(r"\b(?:Address|Adresse):\s*([^\n]+)", re.I)),
    (
        "CASE_NUMBER",
        re.compile(r"\b(?:Fall-Nr\.?|Fallnummer|Patienten-Nr\.?|Pat\.?-Nr\.?)\s*:?\s*([A-Z0-9][A-Z0-9./-]{3,})", re.I),
    ),
    (
        "INSURANCE",
        re.compile(
            r"\b(?:Insurance|Versicherung|Versichertennummer|Versichertennr|Versicherten-Nr|KVNR):?\s*([A-Z0-9 -]{6,})",
            re.I,
        ),
    ),
    ("INSURANCE", re.compile(r"\b([A-Z][0-9]{9})\b")),
    (
        "ADDR",
        re.compile(
            r"\b((?:[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.-]*\s+){0,3}"
            r"[A-Za-zÄÖÜäöüß.-]*(?:straße|strasse|weg|allee|platz|damm|ring|gasse)\s+\d{1,5}[A-Za-z]?)\b",
            re.I,
        ),
    ),
    ("ADDR", re.compile(r"\b(\d{5}\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.-]*(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.-]*){0,3})\b")),
    ("EMAIL", re.compile(r"\b([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\b", re.I)),
    (
        "PHONE",
        re.compile(
            r"\b(?:Tel(?:efon)?|Fax)\.?\s*:?\s*"
            r"(\+?[0-9][0-9 ()/-]{5,}[0-9])",
            re.I,
        ),
    ),
    (
        "PHONE",
        re.compile(
            r"(?<!\w)(?!\d{4}-\d{2}-\d{2}\b)(?!\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b)"
            r"(\+?[0-9][0-9 ()/-]{7,}[0-9])(?!\w)"
        ),
    ),
)

_RESIDUE_PATTERNS: dict[str, re.Pattern[str]] = {
    "DOB": re.compile(
        r"\b(?:geb\s*\.?|geboren(?:\s+am)?|geburtsdatum)\s*:?[ \t]*"
        r"\d{1,2}\s*[./-]\s*\d{1,2}\s*[./-]\s*\d{2,4}\b",
        re.I,
    ),
    "STREET": re.compile(
        r"\b(?:[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.-]*\s+){0,3}"
        r"[A-Za-zÄÖÜäöüß.-]*(?:straße|strasse|weg|allee|platz|damm|ring|gasse)\s+\d{1,5}[A-Za-z]?\b",
        re.I,
    ),
    "PLZ_CITY": re.compile(r"\b\d{5}\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.-]*(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.-]*){0,3}\b"),
    "PHONE": re.compile(r"\b(?:Tel(?:efon)?|Fax)\.?\s*:?\s*\+?[0-9][0-9 ()/-]{5,}[0-9]", re.I),
    "DIGIT_RUN": re.compile(r"(?<!\d)\d{6,}(?!\d)"),
    "KVNR": re.compile(r"\b[A-Z][0-9]{9}\b"),
    "CASE_NUMBER": re.compile(
        r"\b(?:Fall-Nr\.?|Fallnummer|Patienten-Nr\.?|Pat\.?-Nr\.?)\s*:?\s*[A-Z0-9][A-Z0-9./-]{3,}",
        re.I,
    ),
}


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


def residue_report(text: str, policy: dict[str, str] | None = None) -> list[ResidueHit]:
    configured = {**DEFAULT_RESIDUE_POLICY, **{str(key).upper(): str(value).lower() for key, value in (policy or {}).items()}}
    hits = []
    for line_number, line in enumerate(text.splitlines(), 1):
        for name, pattern in _RESIDUE_PATTERNS.items():
            action = configured.get(name, "block")
            if action not in {"block", "report", "ignore"}:
                raise ValueError(f"invalid residue action for {name}: {action}")
            if action != "ignore" and pattern.search(line):
                hits.append(ResidueHit(line_number, name, action))
    return hits


def assert_no_blocking_residue(text: str, policy: dict[str, str] | None = None) -> list[ResidueHit]:
    hits = residue_report(text, policy)
    blocking = [hit for hit in hits if hit.action == "block"]
    if blocking:
        raise ResidueError(blocking)
    return hits


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
        match = re.fullmatch(r"\[([A-Z_]+)_(\d+)\]", key)
        if match:
            counts[match.group(1)] = max(counts.get(match.group(1), 0), int(match.group(2)))
    return counts
