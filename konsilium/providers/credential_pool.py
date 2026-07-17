"""M3 — Credential store and pool.

A minimal credential pool + auth-store. We need little: hold per-provider entries
(access/refresh/expires), serve the active one, rotate on exhaustion, repair
(refresh) on expiry. PKCE login is done once outside the runtime (a CLI tool);
tokens arrive here via auth.json.

Reference pattern: `PooledCredential` (provider/id/access_token/refresh_token/expires/
priority/source) + `_is_expiring(expires_at, skew)`. Our implementation.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


def _parse_iso(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


@dataclass
class Credential:
    provider: str
    access_token: str = ""
    refresh_token: str | None = None
    expires_at: str | None = None       # ISO8601
    base_url: str | None = None
    priority: int = 0
    source: str = "manual"
    request_count: int = 0
    extra: dict = field(default_factory=dict)

    def is_expiring(self, skew_seconds: int = 120) -> bool:
        exp = _parse_iso(self.expires_at)
        return exp is not None and exp - time.time() <= skew_seconds

    @classmethod
    def from_dict(cls, provider: str, d: dict) -> "Credential":
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(provider=provider, extra=extra,
                   **{k: v for k, v in d.items() if k in known and k != "provider"})

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "extra"}
        d.update(self.extra)
        return d


class AuthStore:
    """JSON file `auth.json`: { "providers": { "<name>": [cred, ...] } }.

    Atomic write. The file is a secret (in .gitignore), mounted into the container separately.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.environ.get("KONSILIUM_AUTH_FILE", "auth.json"))

    def _read(self) -> dict:
        if not self.path.exists():
            return {"providers": {}}
        return json.loads(self.path.read_text(encoding="utf-8")) or {"providers": {}}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self.path)
        finally:
            Path(tmp).unlink(missing_ok=True)

    def load(self, provider: str) -> list[Credential]:
        raw = self._read().get("providers", {}).get(provider, [])
        return [Credential.from_dict(provider, c) for c in raw]

    def save(self, provider: str, creds: list[Credential]) -> None:
        data = self._read()
        data.setdefault("providers", {})[provider] = [c.to_dict() for c in creds]
        self._write(data)


class CredentialPool:
    """One provider's entries, sorted by priority. The current one = first
    not exhausted. Rotation moves the exhausted one to the end."""

    def __init__(self, provider: str, store: AuthStore):
        self.provider = provider
        self.store = store
        self.creds = sorted(store.load(provider), key=lambda c: c.priority)
        self._exhausted: set[int] = set()

    def current(self) -> Credential | None:
        for i, c in enumerate(self.creds):
            if i not in self._exhausted:
                return c
        return None

    def rotate(self) -> Credential | None:
        """Mark the current one exhausted and return the next."""
        cur = self.current()
        if cur is not None:
            self._exhausted.add(self.creds.index(cur))
        return self.current()

    def persist(self) -> None:
        self.store.save(self.provider, self.creds)
