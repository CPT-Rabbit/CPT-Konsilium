from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

_DIM = 128


class PatientMemory:
    def __init__(self, root: str | Path, *, lancedb_path: str | Path | None = None, collection: str = "konsilium_memory"):
        self.root = Path(root)
        self.docs_root = self.root / "patients"
        self.index_path = self.root / "memory_index.json"
        self.lancedb_path = Path(lancedb_path or self.root / "lance")
        self.collection = collection
        self._table = None

    def sync(self) -> None:
        # ponytail: full reindex; switch to source_sha256 incremental sync when patient history size makes this slow.
        records = self._records()
        if self._has_lancedb():
            table = self._lance_table()
            for record in records:
                table.delete(f"path = {_quote(record['path'])}")
            if records:
                table.add(records)
        self.index_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")

    def _records(self) -> list[dict]:
        records = []
        for path in sorted(self.docs_root.glob("*/*.md")) + sorted(self.docs_root.glob("*/*/*.md")):
            if not path.is_file():
                continue
            rel = str(path.relative_to(self.root))
            patient_id = path.relative_to(self.docs_root).parts[0]
            text = path.read_text(encoding="utf-8")
            records.append(
                {
                    "path": rel,
                    "patient_id": patient_id,
                    "title": path.stem,
                    "summary": _summary(text),
                    "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "vector": _embed(f"{rel}\n{text}"),
                }
            )
        return records

    def search(self, query: str, *, patient_id: str, limit: int = 5) -> list[dict]:
        if not self.index_path.exists() and not self._has_lancedb():
            self.sync()
        if self._has_lancedb():
            return self._search_lance(query, patient_id=patient_id, limit=limit)
        if not self.index_path.exists():
            self.sync()
        qvec = _embed(query)
        hits = []
        for record in json.loads(self.index_path.read_text(encoding="utf-8")):
            if record.get("patient_id") != patient_id:
                continue
            hits.append((sum(a * b for a, b in zip(qvec, record["vector"])), record))
        hits.sort(key=lambda item: item[0], reverse=True)
        return [
            {key: record[key] for key in ("path", "patient_id", "title", "summary", "source_sha256")}
            for _, record in hits[:limit]
        ]

    def get(self, path: str) -> str:
        resolved = (self.root / path).resolve()
        root = self.root.resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(f"path outside memory root: {path}")
        return resolved.read_text(encoding="utf-8")

    def _has_lancedb(self) -> bool:
        if os.environ.get("KONSILIUM_DISABLE_LANCEDB") == "1":
            return False
        try:
            import lancedb  # noqa: F401
        except ModuleNotFoundError:
            return False
        return True

    def _lance_table(self):
        if self._table is not None:
            return self._table
        import lancedb

        self.lancedb_path.mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(str(self.lancedb_path))
        listed = db.list_tables()
        table_names = getattr(listed, "tables", listed)
        if self.collection in table_names:
            self._table = db.open_table(self.collection)
            return self._table
        self._table = db.create_table(self.collection, data=[_placeholder()])
        self._table.delete("path = '__seed__'")
        return self._table

    def _search_lance(self, query: str, *, patient_id: str, limit: int) -> list[dict]:
        table = self._lance_table()
        rows = table.search(_embed(query)).where(f"patient_id = {_quote(patient_id)}").limit(limit).to_list()
        out = []
        for record in rows:
            if record.get("path") == "__seed__":
                continue
            out.append(
                {key: record[key] for key in ("path", "patient_id", "title", "summary", "source_sha256")}
            )
        return out


def _embed(text: str) -> list[float]:
    vector = [0.0] * _DIM
    for token in re.findall(r"[\w-]+", text.lower()):
        digest = int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16)
        vector[digest % _DIM] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _summary(text: str, limit: int = 240) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _placeholder() -> dict:
    return {
        "path": "__seed__",
        "patient_id": "",
        "title": "",
        "summary": "",
        "source_sha256": "",
        "vector": [0.0] * _DIM,
    }


def _quote(value: str) -> str:
    return "'" + value.replace("'", "\\'") + "'"
