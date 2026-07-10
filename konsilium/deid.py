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
    rejected_entities: tuple["RejectedEntity", ...] = ()


@dataclass(frozen=True)
class PiiEntity:
    kind: str
    value: str


@dataclass(frozen=True)
class RejectedEntity:
    kind: str
    reason: str


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
    "PARTIAL_NAME": "block",
}

_MIN_ENTITY_CHARS = 3
_PROTECTED_SPAN = re.compile(r"\[[A-Z][A-Z_]*_\d+\]|\bage\s+\d{1,3}\b", re.I)
_STREET_SUFFIXES = r"straße|strasse|str\.?|weg|allee|platz|damm|ring|gasse|stieg|twiete|kamp|redder|chaussee|deich|brook|horst|wall|steig"
_ADDRESS_MATERIAL = re.compile(rf"\d|(?:{_STREET_SUFFIXES})\b", re.I)
_ADDRESS_DENYLIST = {
    "keine", "kein", "keinen", "keinem", "keiner", "keines", "ohne", "nicht", "sehr",
    "und", "oder", "der", "die", "das", "den", "dem", "ein", "eine", "einer", "eines",
    "mit", "von", "zu", "im", "in", "am", "an",
}
_TRAILING_ADDRESS_WORDS = re.compile(
    r"(?:[ \t]+(?:Sehr|Mit|Liebe|Lieber|Guten|Freundliche[nrms]?))+$",
    re.I,
)
_ADDRESS_WORD_EXCLUSION = r"Sehr|Mit|Liebe|Lieber|Guten|Freundliche[nrms]?"
_LETTERHEAD_PLACE_DATE = re.compile(
    r"^\s*[A-ZÄÖÜ][^,/\n]{1,60}/\s*[A-ZÄÖÜ][^,\n]{1,60},\s*\d{1,2}\.\d{1,2}\.\d{4}\s*$"
)
_GERMAN_MONTHS = {
    "januar": 1, "februar": 2, "märz": 3, "maerz": 3, "april": 4, "mai": 5, "juni": 6,
    "juli": 7, "juëi": 7, "august": 8, "september": 9, "oktober": 10, "november": 11, "dezember": 12,
}
_GENERIC_INSTITUTION_EMAIL_LOCAL_PARTS = {
    "info", "kontakt", "praxis", "zentrum", "office", "sekretariat", "verwaltung", "spz", "empfang",
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
            r"[0-9]{1,2}\.\s*\S{2,16}\s+\d{4})",
            re.I,
        ),
    ),
    ("CASE_NUMBER", re.compile(r"\bage\s+\d{1,3}\s*,\s*(\d{6})\b", re.I)),
    ("PATIENT", re.compile(r"\bPatienten\s+([A-ZÄÖÜ][A-Za-zÄÖÜäöüß.-]+)\s*,", re.I)),
    ("PATIENT", re.compile(r"\bSeite\s+\d+\s+von\s+\d+\s*,\s*([A-ZÄÖÜ][A-Za-zÄÖÜäöüß.-]+)\s*,", re.I)),
    (
        "PATIENT",
        re.compile(
            r"\b((?:(?:Frau|Herr)[ \t]+)?Dr\.?[ \t]*(?:med\.?[ \t]*)?"
            r"[A-ZÄÖÜ][.\w-]*[a-zäöüß]"
            r"(?:[ \t]+(?!(?:Geburtsdatum|Geboren|Patient(?:in)?|Fallnummer)\b)"
            r"[A-ZÄÖÜ][\w-]*[a-zäöüß])*)\b"
        ),
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
            r"\b((?:[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.-]*[ \t]+){0,3}"
            rf"[A-Za-zÄÖÜäöüß.-]*(?:{_STREET_SUFFIXES})[ \t]+\d{{1,5}}[A-Za-z]?)\b",
            re.I,
        ),
    ),
    (
        "ADDR",
        re.compile(
            rf"\b(\d{{2}}[ \t]?\d{{3}}[ \t]+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.-]*"
            rf"(?:[ \t]+(?!(?:{_ADDRESS_WORD_EXCLUSION})\b)[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.-]*){{0,3}})\b"
        ),
    ),
    (
        "EMAIL",
        re.compile(r"(?<!\w)([.\w][\w.%+-]*@[\w-]+[\w .-]*\.?\s?(?:de|com|org|net|eu))(?!\w)", re.I),
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
        r"\b(?:[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.-]*[ \t]+){0,3}"
        rf"[A-Za-zÄÖÜäöüß.-]*(?:{_STREET_SUFFIXES})[ \t]+\d{{1,5}}[A-Za-z]?\b",
        re.I,
    ),
    "PLZ_CITY": re.compile(
        rf"\b\d{{2}}[ \t]?\d{{3}}[ \t]+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.-]*"
        rf"(?:[ \t]+(?!(?:{_ADDRESS_WORD_EXCLUSION})\b)[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.-]*){{0,3}}\b"
    ),
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
    r"\b(?:geb\s*\.?|geboren(?:\s+am)?|wh\.?|wohnhaft|nachrichtlich|empfänger|patient(?:in)?\s*:|name\s*:)",
    re.I,
)
_RECIPIENT_LINE = re.compile(r"^\s*(?:Frau|Herr|An|Empfänger|Empfaenger)\s*:?\s*$", re.I)
_CONTACT_MARKER = re.compile(r"\b(?:Tel(?:efon)?|Fax)\.?\s*:?", re.I)
_INSTITUTION_NUMBER_MARKER = re.compile(r"\b(?:Ust-?ID|IK-?Nr|IBAN)\b", re.I)
_DOB_MARKER = re.compile(r"\b(?:Geburtsdatum\b|geb\s*\.?)", re.I)
_PARTIAL_NAME = re.compile(
    r"\[PATIENT_\d+\]\s+(?!(?:geb|geboren|Geburtsdatum|age)\b)\w{3,}",
    re.I,
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
    retained_emails: set[str] = set()
    rejected_entities: list[RejectedEntity] = []
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
            lambda match: _replace_match(match, kind, token, today, retained_emails, rejected_entities),
            clean,
        )
    if pii_detector is not None:
        entities = []
        for entity in pii_detector(clean):
            kind = _normal_kind(entity.kind)
            if not kind:
                continue
            for value in _model_entity_values(kind, entity.value):
                if not _valid_entity_value(value):
                    continue
                rejection = _model_entity_rejection(kind, value, clean)
                if rejection:
                    rejected_entities.append(RejectedEntity(kind, rejection))
                    continue
                entities.append((kind, value))
        for kind, value in sorted(entities, key=lambda item: len(item[1]), reverse=True):
            clean = _replace_entity(clean, kind, value, token, today, retained_emails)
    for value in sorted(
        (stored for key, stored in vault.items() if key.startswith("[PATIENT_") and _valid_entity_value(stored)),
        key=len,
        reverse=True,
    ):
        clean = _replace_entity(clean, "PATIENT", value, token, today, retained_emails)

    return DeidentifiedDocument(
        text=clean,
        vault=vault,
        retained_institutional_emails=tuple(sorted(retained_emails)),
        rejected_entities=tuple(rejected_entities),
    )


def residue_report(
    text: str,
    policy: dict[str, str] | None = None,
    *,
    retained_institutional_emails: tuple[str, ...] = (),
) -> list[ResidueHit]:
    configured = {**DEFAULT_RESIDUE_POLICY, **{str(key).upper(): str(value).lower() for key, value in (policy or {}).items()}}
    for name, action in configured.items():
        if action not in {"block", "report", "ignore"}:
            raise ValueError(f"invalid residue action for {name}: {action}")
    hits = []
    offset = 0
    for line_number, raw_line in enumerate(text.splitlines(keepends=True), 1):
        line = raw_line.rstrip("\r\n")
        for name, pattern in _RESIDUE_PATTERNS.items():
            action = configured.get(name, "block")
            matches = list(pattern.finditer(line))
            if action != "ignore" and matches and not _allowed_residue(
                name, text, offset, matches, retained_institutional_emails
            ):
                hits.append(ResidueHit(line_number, name, action))
        marker = _DOB_MARKER.search(line)
        marker_action = configured["DOB_MARKER"]
        if marker_action != "ignore" and marker and not re.search(r"\bage\s+\d+\b", line[marker.end():marker.end() + 24], re.I):
            hits.append(ResidueHit(line_number, "DOB_MARKER", marker_action))
        partial_action = configured["PARTIAL_NAME"]
        if partial_action != "ignore" and marker and _PARTIAL_NAME.search(line):
            hits.append(ResidueHit(line_number, "PARTIAL_NAME", partial_action))
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


def _replace_match(
    match: re.Match[str],
    kind: str,
    token,
    today: date,
    retained_emails: set[str],
    rejected_entities: list[RejectedEntity],
) -> str:
    value = match.group(1).strip()
    if "\n" in value or "\r" in value:
        rejected_entities.append(RejectedEntity(kind, "cross_line_span"))
        return match.group(0)
    if _retain_institutional_value(kind, value, match.string, match.start(1)):
        if kind == "EMAIL":
            retained_emails.add(value)
        return match.group(0)
    replacement = _entity_replacement(kind, value, token, today)
    return match.group(0).replace(match.group(1), replacement)


def _replace_entity(text: str, kind: str, value: str, token, today: date, retained_emails: set[str]) -> str:
    if "\n" in value or "\r" in value:
        return text
    pattern = re.compile(rf"(?<!\w){re.escape(value)}(?!\w)")
    parts = []
    start = 0
    for protected in _PROTECTED_SPAN.finditer(text):
        parts.append(_replace_entity_segment(text[start:protected.start()], pattern, kind, value, token, today, text, start, retained_emails))
        parts.append(protected.group(0))
        start = protected.end()
    parts.append(_replace_entity_segment(text[start:], pattern, kind, value, token, today, text, start, retained_emails))
    return "".join(parts)


def _replace_entity_segment(
    segment: str,
    pattern: re.Pattern[str],
    kind: str,
    value: str,
    token,
    today: date,
    text: str,
    offset: int,
    retained_emails: set[str],
) -> str:
    def replace(match: re.Match[str]) -> str:
        position = offset + match.start()
        if kind == "DOB" and not _is_birth_context(text, position):
            return match.group(0)
        if _retain_institutional_value(kind, value, text, position):
            if kind == "EMAIL":
                retained_emails.add(value)
            return match.group(0)
        return _entity_replacement(kind, value, token, today)

    return pattern.sub(replace, segment)


def _entity_replacement(kind: str, value: str, token, today: date) -> str:
    replacement = token(kind, value)
    if kind == "DOB":
        try:
            replacement = f"age {_age(value, today)}"
        except (IndexError, ValueError):
            pass
    return replacement


def _valid_entity_value(value: str) -> bool:
    return len(re.sub(r"\W", "", value or "")) >= _MIN_ENTITY_CHARS


def _model_entity_rejection(kind: str, value: str, text: str) -> str | None:
    if kind == "DOB":
        return None if _has_birth_context(value, text) else "not_birth_context"
    if kind != "ADDR":
        return None
    normalized = value.strip(" .,;:").lower()
    if normalized in _ADDRESS_DENYLIST:
        return "generic_word"
    if _ADDRESS_MATERIAL.search(value) or _patient_linked_place(value, text):
        return None
    return "not_address_like"


def _model_entity_values(kind: str, value: str) -> list[str]:
    values = value.splitlines()
    if kind == "ADDR":
        values = [_TRAILING_ADDRESS_WORDS.sub("", item) for item in values]
    return [item.strip() for item in values if item.strip()]


def _has_birth_context(value: str, text: str) -> bool:
    pattern = re.compile(rf"(?<!\w){re.escape(value)}(?!\w)", re.I)
    return any(_is_birth_context(text, match.start()) for match in pattern.finditer(text))


def _is_birth_context(text: str, position: int) -> bool:
    line_start = text.rfind("\n", 0, position) + 1
    prefix = text[line_start:position]
    return bool(re.search(
        r"(?:\bDOB\b|Date of birth|Geburtsdatum|geboren(?:\s+am)?|geb\s*\.?)"
        r"[^\n]{0,64}$",
        prefix,
        re.I,
    ))


def _patient_linked_place(value: str, text: str) -> bool:
    pattern = re.compile(rf"(?<!\w){re.escape(value)}(?!\w)", re.I)
    for match in pattern.finditer(text):
        line = _line_at(text, match.start())
        if re.search(
            r"(?:\[PATIENT_\d+\]|\bPatient(?:in)?\b).{0,80}"
            r"(?:wohnt|wohnhaft|lebt|Wohnort|Wohnsitz).{0,80}$|"
            r"\b(?:Wohnort|Wohnsitz|wohnhaft)\s*:?[^\n]*$",
            line,
            re.I,
        ):
            return True
    return False


def institutional_email_allowlist(text: str) -> tuple[str, ...]:
    pattern = re.compile(r"(?<!\w)([.\w][\w.%+-]*@[\w-]+[\w .-]*\.?\s?(?:de|com|org|net|eu))(?!\w)", re.I)
    return tuple(sorted({
        match.group(1).strip()
        for match in pattern.finditer(text)
        if _retain_institutional_value("EMAIL", match.group(1).strip(), text, match.start(1))
    }))


def _retain_institutional_value(kind: str, value: str, text: str, position: int) -> bool:
    if kind not in {"ADDR", "PHONE", "EMAIL"} or not _is_institutional_address(text, position):
        return False
    if kind != "EMAIL":
        return True
    local_part = value.lstrip(".").split("@", 1)[0].lower()
    return local_part in _GENERIC_INSTITUTION_EMAIL_LOCAL_PARTS


def _is_institutional_address(text: str, position: int) -> bool:
    lines = text.splitlines()
    line_index = text[:position].count("\n")
    context = "\n".join(lines[max(0, line_index - 4):line_index + 5])
    if _is_recipient_address(lines, line_index):
        return False
    if _LETTERHEAD_PLACE_DATE.fullmatch(_line_at(text, position)):
        return True
    if _PRIVATE_ADDRESS_MARKERS.search(context):
        return False
    return bool(_INSTITUTION_MARKERS.search(context))


def _is_recipient_address(lines: list[str], line_index: int) -> bool:
    preceding = lines[max(0, line_index - 3):line_index]
    if any(_RECIPIENT_LINE.fullmatch(line) for line in preceding):
        return True
    adjacent = "\n".join(lines[max(0, line_index - 2):line_index + 1])
    if not re.search(r"\[PATIENT_\d+\]", adjacent):
        return False
    affiliation = "\n".join(lines[max(0, line_index - 4):line_index + 1])
    return not _INSTITUTION_MARKERS.search(affiliation)


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
    match = re.search(r"[.\w][\w.%+-]*@[\w-]+[\w .-]*\.?\s?(?:de|com|org|net|eu)", line, re.I)
    return match.group(0).strip() if match else ""


def _age(value: str, today: date) -> int:
    day, month, year = _parse_dob(value)
    birthday_passed = (today.month, today.day) >= (month, day)
    return today.year - year - (0 if birthday_passed else 1)


def _parse_dob(value: str) -> tuple[int, int, int]:
    spelled = re.search(r"(\d{1,2})\.\s*(\S{2,16})\s+(\d{4})", value, re.I)
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
