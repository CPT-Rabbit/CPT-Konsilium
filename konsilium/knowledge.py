from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from .egress import assert_safe_knowledge_query


@dataclass(frozen=True)
class SearchResult:
    title: str
    source: str
    url: str
    year: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def pubmed_search(query: str, *, limit: int = 5, fetch=None) -> list[SearchResult]:
    assert_safe_knowledge_query(query)
    api_key = os.environ.get("NCBI_API_KEY")
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=pubmed&retmode=json&retmax={limit}&term={quote_plus(query)}"
    )
    if api_key:
        url += f"&api_key={quote_plus(api_key)}"
    data = _json(fetch or _fetch, url)
    ids = data.get("esearchresult", {}).get("idlist", [])
    return [
        SearchResult(f"PubMed PMID {pmid}", "pubmed", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
        for pmid in ids[:limit]
    ]


def semanticscholar_search(query: str, *, limit: int = 5, fetch=None) -> list[SearchResult]:
    assert_safe_knowledge_query(query)
    headers = {}
    if os.environ.get("S2_API_KEY"):
        headers["x-api-key"] = os.environ["S2_API_KEY"]
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/search"
        f"?limit={limit}&fields=title,url,year&query={quote_plus(query)}"
    )
    data = _json(fetch or _fetch, url, headers=headers)
    return [
        SearchResult(
            item.get("title") or "Untitled paper",
            "semanticscholar",
            item.get("url") or "",
            item.get("year"),
        )
        for item in data.get("data", [])[:limit]
    ]


def guidelines_lookup(query: str, *, limit: int = 5) -> list[SearchResult]:
    assert_safe_knowledge_query(query)
    return [
        SearchResult(
            f"AWMF guideline search: {query}",
            "guidelines:awmf",
            f"https://register.awmf.org/de/suche?search={quote_plus(query)}",
        )
    ][:limit]


def _fetch(url: str, headers: dict | None = None) -> str:
    request = Request(url, headers=headers or {})
    with urlopen(request, timeout=20) as response:
        return response.read().decode("utf-8")


def _json(fetch, url: str, headers: dict | None = None) -> dict:
    try:
        raw = fetch(url, headers)
    except TypeError:
        raw = fetch(url)
    return json.loads(raw)
