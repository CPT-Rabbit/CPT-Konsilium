from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Callable


@dataclass(frozen=True)
class DeidentifiedDocument:
    text: str
    vault: dict[str, str]
    retained_institutional_emails: tuple[str, ...] = ()


@dataclass(frozen=True)
class PiiEntity:
    kind: str
    value: str


PiiDetector = Callable[[str], list[PiiEntity]]

DEFAULT_RESIDUE_POLICY = {
    "CORRUPTED_TOKEN": "block",
    "DOB": "block",
    "PERSON_HEADER": "block",
    "STREET": "block",
    "PLZ_CITY": "block",
    "PHONE": "block",
    "DIGIT_RUN": "block",
    "KVNR": "block",
    "CASE_NUMBER": "block",
    "DOB_MARKER": "block",
    "EMAIL": "block",
}

_MIN_ENTITY_CHARS = 3
_TOKEN_SPAN = re.compile(r"\[[A-Z][A-Z_]*_\d+\]")
_STREET_SUFFIXES = r"straße|strasse|str\.?|weg|allee|platz|damm|ring|gasse|stieg|twiete|kamp|redder|chaussee|deich|brook|horst|wall|steig"
_GERMAN_MONTHS = {
    "januar": 1, "februar": 2, "märz": 3, "maerz": 3, "april": 4, "mai": 5, "juni": 6,
    "juli": 7, "august": 8, "september": 9, "oktober": 10, "november": 11, "dezember": 12,
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
            r"\b(?:DOB|Date of birth|Geboren(?:\s+am)?|Geburtsdatum|geb\s*\.?)\s*:?(?:\s+am)?\s*"
            r"([0-9]{1,2}\s*[.,/-]\s*[0-9]{1,2}\s*[.,/-]\s*[0-9]{2,4}(?:\s+\d)?)",
            re.I,
        ),
    ),
    (
        "DOB",
        re.compile(
            r"\b(?:Geburtsdatum|Geboren(?:\s+am)?|geb\s*\.?)\s*:?(?:\s+am)?[ \t]*"
            r"((?:(?:Montag|Dienstag|Mittwoch|Donnerstag|Freitag|Samstag|Sonntag),?\s+)?"
            r"[0-9]{1,2}\.\s*(?:Januar|Februar|März|Maerz|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s+\d{4})",
            re.I,
        ),
    ),
    ("PATIENT", re.compile(r"\bPatienten\s+([A-ZÄÖÜ][A-Za-zÄÖÜäöüß.-]+)\s*,", re.I)),
    ("PATIENT", re.compile(r"\bSeite\s+\d+\s+von\s+\d+\s*,\s*([A-ZÄÖÜ][A-Za-zÄÖÜäöüß.-]+)\s*,", re.I)),
    (
        "PATIENT",
        re.compile(r"\b((?:(?:Frau|Herr)\s+)?Dr\.?\s*(?:med\.?\s*)?[A-ZÄÖÜ][.\w-]*[a-zäöüß])\b"),
    ),
    ("ADDR", re.compile(r"\b(?:Address|Adresse|wh\.?|wohnhaft(?:\s+in)?):?\s*([^\n]+)", re.I)),
    (
        "CASE_NUMBER",
        re.compile(
            r"\b(?:Fall-Nr\.?|Fallnummer|Patienten-Nr\.?|Pat\.?\s*-?\s*Nr\.?|Aufn\.?\s*-?\s*Nr\.?)\s*:?\s*"
            r"([A-Z0-9][A-Z0-9./-]{3,})",
            re.I,
        ),
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
            rf"[A-Za-zÄÖÜäöüß.-]*(?:{_STREET_SUFFIXES})\s+\d{{1,5}}[A-Za-z]?)\b",
            re.I,
        ),
    ),
    ("ADDR", re.compile(r"\b(\d{5}\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.-]*(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.-]*){0,3})\b")),
    (
        "EMAIL",
        re.compile(r"(?<!\w)(\w[\w.%+-]*@[\w-]+[\w .-]*\.?\s?(?:de|com|org|net|eu))(?!\w)", re.I),
    ),
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
    "CORRUPTED_TOKEN": re.compile(r"\[[A-Z_]+\[|\][A-Z_]"),
    "DOB": re.compile(
        r"\b(?:geb\s*\.?|geboren(?:\s+am)?|geburtsdatum)\s*:?(?:\s+am)?[ \t]*"
        r"(?:\d{1,2}\s*[.,/-]\s*\d{1,2}\s*[.,/-]\s*\d{2,4}(?:\s+\d)?|"
        r"(?:(?:Montag|Dienstag|Mittwoch|Donnerstag|Freitag|Samstag|Sonntag),?\s+)?"
        r"\d{1,2}\.\s*(?:Januar|Februar|März|Maerz|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s+\d{4})\b",
        re.I,
    ),
    "PERSON_HEADER": re.compile(
        r"\b(?:Patienten\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.-]+\s*,|Seite\s+\d+\s+von\s+\d+\s*,\s*[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.-]+\s*,)",
        re.I,
    ),
    "STREET": re.compile(
        r"\b(?:[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.-]*\s+){0,3}"
        rf"[A-Za-zÄÖÜäöüß.-]*(?:{_STREET_SUFFIXES})\s+\d{{1,5}}[A-Za-z]?\b",
        re.I,
    ),
    "PLZ_CITY": re.compile(r"\b\d{5}\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.-]*(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.-]*){0,3}\b"),
    "PHONE": re.compile(r"\b(?:Tel(?:efon)?|Fax)\.?\s*:?\s*\+?[0-9][0-9 ()/-]{5,}[0-9]", re.I),
    "EMAIL": re.compile(r"(?<=\w)@(?=\w)"),
    "DIGIT_RUN": re.compile(r"(?<!\d)\d{6,}(?!\d)"),
    "KVNR": re.compile(r"\b[A-Z][0-9]{9}\b"),
    "CASE_NUMBER": re.compile(
        r"\b(?:Fall-Nr\.?|Fallnummer|Patienten-Nr\.?|Pat\.?\s*-?\s*Nr\.?|Aufn\.?\s*-?\s*Nr\.?)\s*:?\s*[A-Z0-9][A-Z0-9./-]{3,}",
        re.I,
    ),
}

_INSTITUTION_MARKERS = re.compile(
    r"\b(?:Krankenhaus|Klinik|Klinikum|Zentrum|SPZ|Praxis|gGmbH|e\.?\s*V\.?|"
    r"Stiftung|Postfach|Akademisches|Institut|Ambulanz)\b|www\.|\b(?:IBAN|IK-?Nr|Ust-?ID)\b",
    re.I,
)
_PRIVATE_ADDRESS_MARKERS = re.compile(
    r"\b(?:geb\s*\.?|geboren(?:\s+am)?|wh\.?|wohnhaft|patient(?:in)?\s*:|name\s*:)",
    re.I,
)
_CONTACT_MARKER = re.compile(r"\b(?:Tel(?:efon)?|Fax)\.?\s*:?", re.I)
_INSTITUTION_NUMBER_MARKER = re.compile(r"\b(?:Ust-?ID|IK-?Nr|IBAN)\b", re.I)
_DOB_MARKER = re.compile(r"\b(?:Geburtsdatum\b|geb\s*\.?)", re.I)


def deidentify(
    text: str,
    existing_vault: dict[str, str] | None = None,
    *,
    pii_detector: PiiDetector | None = None,
    today: date | None = None,
) -> DeidentifiedDocument:
    vault = dict(existing_vault or {})
    counts: dict[str, int] = _counts(vault)
    retained_emails: set[str] = set()
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
            lambda match: _replace_match(match, kind, token, today, retained_emails),
            clean,
        )
    if pii_detector is not None:
        entities = [
            (_normal_kind(entity.kind), entity.value.strip())
            for entity in pii_detector(clean)
            if _normal_kind(entity.kind) and _valid_entity_value(entity.value)
        ]
        for kind, value in sorted(entities, key=lambda item: len(item[1]), reverse=True):
            clean = _replace_entity(clean, kind, value, token, retained_emails)
    for value in sorted(
        (stored for key, stored in vault.items() if key.startswith("[PATIENT_") and _valid_entity_value(stored)),
        key=len,
        reverse=True,
    ):
        clean = _replace_entity(clean, "PATIENT", value, token, retained_emails)

    return DeidentifiedDocument(text=clean, vault=vault, retained_institutional_emails=tuple(sorted(retained_emails)))


def residue_report(
    text: str,
    policy: dict[str, str] | None = None,
    *,
    retained_institutional_emails: tuple[str, ...] = (),
) -> list[ResidueHit]:
    configured = {**DEFAULT_RESIDUE_POLICY, **{str(key).upper(): str(value).lower() for key, value in (policy or {}).items()}}
    hits = []
    offset = 0
    for line_number, raw_line in enumerate(text.splitlines(keepends=True), 1):
        line = raw_line.rstrip("\r\n")
        for name, pattern in _RESIDUE_PATTERNS.items():
            action = configured.get(name, "block")
            if action not in {"block", "report", "ignore"}:
                raise ValueError(f"invalid residue action for {name}: {action}")
            matches = list(pattern.finditer(line))
            if action != "ignore" and matches and not _allowed_residue(
                name, text, offset, matches, retained_institutional_emails
            ):
                hits.append(ResidueHit(line_number, name, action))
        marker = _DOB_MARKER.search(line)
        marker_action = configured["DOB_MARKER"]
        if marker_action != "ignore" and marker and not re.search(r"\bage\s+\d+\b", line[marker.end():marker.end() + 24], re.I):
            hits.append(ResidueHit(line_number, "DOB_MARKER", marker_action))
        offset += len(raw_line)
    if text.count("[") != text.count("]") and configured["CORRUPTED_TOKEN"] != "ignore":
        hits.append(ResidueHit(1, "CORRUPTED_TOKEN", configured["CORRUPTED_TOKEN"]))
    return hits


def assert_no_blocking_residue(
    text: str,
    policy: dict[str, str] | None = None,
    *,
    retained_institutional_emails: tuple[str, ...] = (),
) -> list[ResidueHit]:
    hits = residue_report(text, policy, retained_institutional_emails=retained_institutional_emails)
    blocking = [hit for hit in hits if hit.action == "block"]
    if blocking:
        raise ResidueError(blocking)
    return hits


def _replace_match(match: re.Match[str], kind: str, token, today: date, retained_emails: set[str]) -> str:
    value = match.group(1).strip()
    if kind in {"ADDR", "PHONE", "EMAIL"} and _is_institutional_address(match.string, match.start(1)):
        if kind == "EMAIL":
            retained_emails.add(value)
        return match.group(0)
    if kind == "DOB":
        token(kind, value)
        replacement = f"age {_age(value, today)}"
    else:
        replacement = token(kind, value)
    return match.group(0).replace(match.group(1), replacement)


def _replace_entity(text: str, kind: str, value: str, token, retained_emails: set[str]) -> str:
    pattern = re.compile(rf"(?<!\w){re.escape(value)}(?!\w)")
    parts = []
    start = 0
    for protected in _TOKEN_SPAN.finditer(text):
        parts.append(_replace_entity_segment(text[start:protected.start()], pattern, kind, value, token, text, start, retained_emails))
        parts.append(protected.group(0))
        start = protected.end()
    parts.append(_replace_entity_segment(text[start:], pattern, kind, value, token, text, start, retained_emails))
    return "".join(parts)


def _replace_entity_segment(
    segment: str,
    pattern: re.Pattern[str],
    kind: str,
    value: str,
    token,
    text: str,
    offset: int,
    retained_emails: set[str],
) -> str:
    def replace(match: re.Match[str]) -> str:
        position = offset + match.start()
        if kind in {"ADDR", "PHONE", "EMAIL"} and _is_institutional_address(text, position):
            if kind == "EMAIL":
                retained_emails.add(value)
            return match.group(0)
        return token(kind, value)

    return pattern.sub(replace, segment)


def _valid_entity_value(value: str) -> bool:
    return len(re.sub(r"\W", "", value or "")) >= _MIN_ENTITY_CHARS


def _is_institutional_address(text: str, position: int) -> bool:
    lines = text.splitlines()
    line_index = text[:position].count("\n")
    context = "\n".join(lines[max(0, line_index - 4):line_index + 5])
    if _PRIVATE_ADDRESS_MARKERS.search(context):
        return False
    return bool(_INSTITUTION_MARKERS.search(context))


def _allowed_residue(
    name: str,
    text: str,
    offset: int,
    matches: list[re.Match[str]],
    retained_institutional_emails: tuple[str, ...],
) -> bool:
    if name in {"STREET", "PLZ_CITY", "PHONE"}:
        return all(_is_institutional_address(text, offset + match.start()) for match in matches)
    if name == "EMAIL":
        return all(_email_at(text, offset + match.start()) in retained_institutional_emails for match in matches)
    if name == "DIGIT_RUN":
        return all(
            _is_institutional_address(text, offset + match.start())
            and (
                _CONTACT_MARKER.search(_line_at(text, offset + match.start()))
                or _INSTITUTION_NUMBER_MARKER.search(_line_at(text, offset + match.start()))
            )
            for match in matches
        )
    return False


def _line_at(text: str, position: int) -> str:
    start = text.rfind("\n", 0, position) + 1
    end = text.find("\n", position)
    return text[start:] if end < 0 else text[start:end]


def _email_at(text: str, position: int) -> str:
    line = _line_at(text, position)
    match = re.search(r"\w[\w.%+-]*@[\w-]+[\w .-]*\.?\s?(?:de|com|org|net|eu)", line, re.I)
    return match.group(0).strip() if match else ""


def _age(value: str, today: date) -> int:
    day, month, year = _parse_dob(value)
    birthday_passed = (today.month, today.day) >= (month, day)
    return today.year - year - (0 if birthday_passed else 1)


def _parse_dob(value: str) -> tuple[int, int, int]:
    spelled = re.search(r"(\d{1,2})\.\s*([A-Za-zÄÖÜäöüß]+)\s+(\d{4})", value, re.I)
    if spelled:
        month = _GERMAN_MONTHS.get(spelled.group(2).lower())
        if month:
            return int(spelled.group(1)), month, int(spelled.group(3))
    parts = re.split(r"[.,/-]", value)
    day = int(parts[0])
    month = int(parts[1])
    year = int(re.sub(r"\s+", "", parts[2]))
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
