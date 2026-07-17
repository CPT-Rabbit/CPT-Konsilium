from __future__ import annotations

from .deid import PiiEntity

DEFAULT_GLINER_MODEL = "urchade/gliner_multi_pii-v1"

_LABEL_KINDS = {
    "person": "PERSON",
    "address": "ADDRESS",
    "date of birth": "DOB",
    "phone number": "PHONE",
    "email": "EMAIL",
    "insurance number": "INSURANCE",
}


class GlinerPiiDetector:
    """Token-classification NER detector (primary recall layer, fully local)."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_GLINER_MODEL,
        threshold: float = 0.3,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
    ):
        try:
            from gliner import GLiNER
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "GLiNER detector requires the 'gliner' package (pip install konsilium[ner])"
            ) from error
        if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("GLiNER detector requires 0 <= chunk_overlap < chunk_size")
        self.model = GLiNER.from_pretrained(model)
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def __call__(self, text: str) -> list[PiiEntity]:
        seen: set[tuple[str, str]] = set()
        entities: list[PiiEntity] = []
        step = self.chunk_size - self.chunk_overlap
        for start in range(0, max(len(text), 1), step):
            chunk = text[start:start + self.chunk_size]
            for entity in self.model.predict_entities(chunk, list(_LABEL_KINDS), threshold=self.threshold):
                kind = _LABEL_KINDS[entity["label"]]
                value = entity["text"].strip()
                if value and (kind, value) not in seen:
                    seen.add((kind, value))
                    entities.append(PiiEntity(kind, value))
            if start + self.chunk_size >= len(text):
                break
        return entities


def composite_detector(*detectors):
    """Union of several PII detectors; recall stacks, dedup by (kind, value)."""

    def detect(text: str) -> list[PiiEntity]:
        seen: set[tuple[str, str]] = set()
        entities: list[PiiEntity] = []
        for detector in detectors:
            for entity in detector(text):
                key = (entity.kind.strip().upper(), entity.value.strip())
                if key not in seen:
                    seen.add(key)
                    entities.append(entity)
        return entities

    return detect
