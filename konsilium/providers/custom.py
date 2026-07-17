"""`custom` provider: Cloudflare / OpenAI-compatible.

Use a base_url on an OpenAI-compatible `/compat` endpoint and a static env key.
There is nothing to refresh, so on_auth_error returns False.
"""

from __future__ import annotations

import os

from .base import Provider, Resolved


class CustomProvider(Provider):
    kind = "custom"

    def __init__(self, *, base_url: str, model: str, api_key_env: str = "OPENAI_API_KEY"):
        if not base_url:
            raise ValueError("custom provider requires base_url")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key_env = api_key_env

    def resolve(self) -> Resolved:
        key = os.environ.get(self._api_key_env, "")
        if not key:
            raise RuntimeError(f"no key in env {self._api_key_env!r}")
        return Resolved(base_url=self._base_url, api_key=key, model=self._model)

    def on_auth_error(self) -> bool:
        # static key: nothing to refresh
        return False
