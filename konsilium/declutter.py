"""Deterministic clinical-body cleanup.

Strips institutional boilerplate that pollutes analysis context — banking and
registry details, licences, per-page footers, page markers, contact lines —
without ever rewriting clinical prose. The issuing institution and date are
preserved separately as document metadata (structuring), so provenance is kept
in the header while the body carries only clinical content.

Never touches clinical text: it only drops lines matching institution-only
markers (IBAN/BIC/HRB/Amtsgericht/... — never present in clinical prose),
page markers, or contact-only footer lines.
"""
from __future__ import annotations

import re
from collections import Counter

_PAGE = re.compile(r"^\s*Seite\s+\d+\s*(?:von|/)\s*\d+\b", re.I)
_REQS = re.compile(
    r"\b(?:IBAN|BIC|SWIFT|Ust-?\s?Id\s?Nr|Ust-?ID|USt-?tdNr|St[.-]?Nr|Steuer-?Nr|Steuernummer|"
    r"IK-?NR|HRB|Amtsgericht|Registergericht|Sitz der Gesellschaft|Gesch[aä]ftsf[uü]hr|"
    r"Postfach|Vorsitzende|Aufsichtsrat|Bankverbindung|Spendenkonto|Sparkasse|Volksbank|"
    r"Commercial Bank|Handelsregister)\b",
    re.I,
)
_CONTACT = re.compile(r"www\.|https?://|\bTel(?:efon)?\.?\s*:?\s*\+?\d|\bFax\.?\s*:?\s*\+?[\d\[]|@", re.I)
_ADDR_INST = re.compile(r"\b\d{5}\s+[A-ZÄÖÜ]|\bPostfach\b|stra[sß]e\s+\d+", re.I)
# a line still carrying clinical prose if it holds >=3 real words after stripping contacts
_PROSE = re.compile(r"[A-Za-zÄÖÜäöüß]{4,}")


def _prose_words(line: str) -> int:
    stripped = re.sub(r"www\.\S+|https?://\S+|\S+@\S+", "", line)
    return len(_PROSE.findall(stripped))


def _drop_reason(line: str, repeated: bool) -> str | None:
    text = line.strip()
    if not text:
        return None
    if _PAGE.search(text):
        return "page"
    if _REQS.search(text):
        return "reqs"
    if _CONTACT.search(text) and _prose_words(text) < 3:
        return "contact"
    if repeated and (_REQS.search(text) or _CONTACT.search(text) or _ADDR_INST.search(text)):
        return "repeat"
    return None


def declutter(text: str) -> str:
    lines = text.splitlines()
    counts = Counter(line.strip() for line in lines if len(line.strip()) > 12)
    kept = [line for line in lines if _drop_reason(line, counts[line.strip()] >= 3) is None]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip() + "\n"


def declutter_stats(text: str) -> dict[str, int]:
    lines = text.splitlines()
    counts = Counter(line.strip() for line in lines if len(line.strip()) > 12)
    dropped: Counter[str] = Counter()
    for line in lines:
        reason = _drop_reason(line, counts[line.strip()] >= 3)
        if reason:
            dropped[reason] += 1
    return dict(dropped)
