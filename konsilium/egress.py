from __future__ import annotations

import re


class EgressViolation(ValueError):
    pass


_PII_TOKEN = re.compile(r"\[(?:PATIENT|DOB|ADDR|INSURANCE|EMAIL|PHONE)_\d+\]")
_PHONE = re.compile(
    r"\b(?!\d{4}-\d{2}-\d{2}\b)(?!\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b)(?!\d{4}[/-]\d{4}\b)"
    r"\+?[0-9][0-9 ()/-]{7,}[0-9]\b"
)

_RAW_PII = (
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    re.compile(r"\b(?:DOB|Date of birth|Geboren|Geburtsdatum):", re.I),
    re.compile(r"\b(?:Address|Adresse|Insurance|Versicherung|KVNR):", re.I),
    _PHONE,
)


def assert_safe_knowledge_query(query: str) -> None:
    if _PII_TOKEN.search(query):
        raise EgressViolation("knowledge query contains a de-identification token")
    if any(pattern.search(query) for pattern in _RAW_PII):
        raise EgressViolation("knowledge query contains likely raw PII")
