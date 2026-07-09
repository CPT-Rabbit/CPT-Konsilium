from __future__ import annotations

import json
import re
from typing import Callable
from urllib.request import Request, urlopen

from .deid import PiiEntity
from .util import json_block

Fetch = Callable[[str, dict, float], str]

_ROLE_WORDS = {
    "eltern", "mutter", "vater", "kind", "sohn", "tochter", "familie", "patient", "patientin",
    "ehefrau", "ehemann", "geschwister", "junge", "mädchen", "parents", "mother", "father", "child",
}
_CAPITALIZED_WORD = re.compile(r"\b[A-ZÄÖÜ][a-zäöüß]{2,}\b")

_ENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["PERSON", "ADDRESS", "DOB", "INSURANCE", "EMAIL", "PHONE"]},
                    "value": {"type": "string"},
                },
                "required": ["kind", "value"],
            },
        }
    },
    "required": ["entities"],
}


class OllamaPiiDetector:
    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout_s: float = 300.0,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
        fetch: Fetch | None = None,
    ):
        if not model:
            raise ValueError("Ollama PII detector requires a configured model")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("Ollama PII detector requires 0 <= chunk_overlap < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.fetch = fetch or _fetch

    def __call__(self, text: str) -> list[PiiEntity]:
        entities = []
        seen = set()
        for chunk in _chunks(text, self.chunk_size, self.chunk_overlap):
            for entity in self._detect_chunk(chunk):
                key = (entity.kind.strip().upper(), entity.value.strip())
                if key not in seen:
                    seen.add(key)
                    entities.append(entity)
        return entities

    def _detect_chunk(self, text: str) -> list[PiiEntity]:
        payload = {
            "model": self.model,
            "stream": False,
            "format": _ENTITY_SCHEMA,
            "think": False,
            "options": {"temperature": 0},
            "prompt": _prompt(text),
        }
        raw = self.fetch(f"{self.base_url}/api/generate", payload, self.timeout_s)
        response = json.loads(raw).get("response", "[]")
        entities = _entities(response)
        return [
            PiiEntity(str(item.get("kind") or ""), str(item.get("value") or ""))
            for item in entities
            if isinstance(item, dict) and _valid_entity(item)
        ]

    def healthcheck(self) -> None:
        self("Detector reachability check.")


def _fetch(url: str, payload: dict, timeout_s: float) -> str:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_s) as response:
        return response.read().decode("utf-8")


def _chunks(text: str, size: int, overlap: int):
    step = size - overlap
    for start in range(0, len(text), step):
        yield text[start:start + size]
        if start + size >= len(text):
            break


def _valid_entity(item: dict) -> bool:
    kind = str(item.get("kind") or "").strip().upper()
    value = str(item.get("value") or "").strip()
    if not kind or not value:
        return False
    if kind != "PERSON":
        return True
    return value.lower() not in _ROLE_WORDS and bool(_CAPITALIZED_WORD.search(value))


def _entities(response) -> list[dict]:
    if isinstance(response, str):
        response = json.loads(json_block(response))
    if isinstance(response, dict):
        if isinstance(response.get("entities"), list):
            return response["entities"]
        if response.get("kind") and response.get("value"):
            return [response]
    if isinstance(response, list):
        return response
    raise ValueError("Ollama PII detector response must contain entities")


def _prompt(text: str) -> str:
    return (
        "Extract ALL personally identifying information from this medical text. "
        "PERSON includes every named human: patients, physicians, doctors (Dr., Dr. med., Prof.), "
        "nurses, relatives, contacts. "
        "Extract proper names only; never generic role words like Eltern, Mutter, Patient. "
        "Return strict JSON only matching the provided schema. "
        "Allowed kinds: PERSON, ADDRESS, DOB, INSURANCE, EMAIL, PHONE. "
        "Do not include diagnoses, labs, medications, dates of medical events, or symptoms.\n\n"
        f"TEXT:\n{text}"
    )
