from __future__ import annotations

import json
from typing import Callable
from urllib.request import Request, urlopen

from .deid import PiiEntity

Fetch = Callable[[str, dict], str]


class OllamaPiiDetector:
    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        fetch: Fetch | None = None,
    ):
        if not model:
            raise ValueError("Ollama PII detector requires a configured model")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.fetch = fetch or _fetch

    def __call__(self, text: str) -> list[PiiEntity]:
        payload = {
            "model": self.model,
            "stream": False,
            "prompt": _prompt(text),
        }
        raw = self.fetch(f"{self.base_url}/api/generate", payload)
        response = json.loads(raw).get("response", "[]")
        entities = json.loads(response)
        if not isinstance(entities, list):
            raise ValueError("Ollama PII detector response must be a JSON list")
        return [
            PiiEntity(str(item.get("kind") or ""), str(item.get("value") or ""))
            for item in entities
            if isinstance(item, dict) and item.get("kind") and item.get("value")
        ]

    def healthcheck(self) -> None:
        self("Detector reachability check.")


def _fetch(url: str, payload: dict) -> str:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def _prompt(text: str) -> str:
    return (
        "Extract personally identifying information from this medical text. "
        "Return strict JSON only: an array of objects with kind and value. "
        "Allowed kinds: PERSON, ADDRESS, DOB, INSURANCE, EMAIL, PHONE. "
        "Do not include diagnoses, labs, medications, dates of medical events, or symptoms.\n\n"
        f"TEXT:\n{text}"
    )
