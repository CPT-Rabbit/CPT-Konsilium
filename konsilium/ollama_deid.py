from __future__ import annotations

import json
import re
from typing import Callable
from urllib.request import Request, urlopen

from .deid import PiiEntity

Fetch = Callable[[str, dict, float], str]

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
        fetch: Fetch | None = None,
    ):
        if not model:
            raise ValueError("Ollama PII detector requires a configured model")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.fetch = fetch or _fetch

    def __call__(self, text: str) -> list[PiiEntity]:
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
            if isinstance(item, dict) and item.get("kind") and item.get("value")
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


def _entities(response) -> list[dict]:
    if isinstance(response, str):
        response = json.loads(_json_block(response))
    if isinstance(response, dict):
        if isinstance(response.get("entities"), list):
            return response["entities"]
        if response.get("kind") and response.get("value"):
            return [response]
    if isinstance(response, list):
        return response
    raise ValueError("Ollama PII detector response must contain entities")


def _json_block(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S | re.I)
    return match.group(1) if match else text


def _prompt(text: str) -> str:
    return (
        "Extract ALL personally identifying information from this medical text. "
        "PERSON includes every named human: patients, physicians, doctors (Dr., Dr. med., Prof.), "
        "nurses, relatives, contacts. "
        "Return strict JSON only matching the provided schema. "
        "Allowed kinds: PERSON, ADDRESS, DOB, INSURANCE, EMAIL, PHONE. "
        "Do not include diagnoses, labs, medications, dates of medical events, or symptoms.\n\n"
        f"TEXT:\n{text}"
    )
