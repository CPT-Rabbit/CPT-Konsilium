"""M3 — Model providers: generic interface.

Hermes node #2 (`auxiliary_client.resolve_provider_client`) is a big
if/elif over the `provider` string. We fold it into one pattern: each provider
can yield (base_url, creds) and build an OpenAI-compatible client; rotation/refresh
live in the pool. A new subscription = one more Provider, no loop changes.

Reference pattern: `agent/auxiliary_client.py::resolve_provider_client`,
`agent/credential_pool.py` (CredentialPool/PooledCredential, refresh-on-401).
Concrete implementations: providers/custom.py, providers/xai_oauth.py, providers/codex.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class Resolved:
    """Ready-made provider connection parameters for the current call."""

    base_url: str
    api_key: str
    model: str
    extra_headers: dict[str, str] | None = None


@runtime_checkable
class Provider(Protocol):
    """One path to connect to the model (custom / subscription)."""

    kind: str  # "custom" | "xai-oauth" | "codex"

    def resolve(self) -> Resolved:
        """Return base_url + the active key/token (accounting for refresh).

        custom    → base_url from config + key from env.
        xai-oauth → token from the OAuth pool (PKCE login already done; refresh on expiry).
        codex     → token from the codex OAuth pool.
        """
        ...

    def on_auth_error(self) -> bool:
        """The call returned 401/403. Try to refresh creds / switch pool entry.

        Returns True if there is something to retry with (refreshed token/next entry),
        otherwise False. Reference: `_try_refresh_*_credentials` + CredentialPool rotation.
        """
        ...


def build_provider(model_cfg, *, auth_store=None) -> Provider:  # noqa: ANN001 — types in config.py
    """Factory by `model_cfg.provider`.

    custom     → key from env (auth_store not needed).
    xai-oauth / codex → OAuth pool from auth_store (auth.json).
    claude-cli → local Claude Code CLI subscription on PATH.
    """
    kind = model_cfg.provider
    if kind == "custom":
        from .custom import CustomProvider

        return CustomProvider(
            base_url=model_cfg.base_url,
            model=model_cfg.model,
            api_key_env=model_cfg.api_key_env,
        )
    if kind in ("xai-oauth", "codex"):
        from .credential_pool import AuthStore

        store = auth_store or AuthStore()
        if kind == "xai-oauth":
            from . import xai_oauth

            return xai_oauth.make(model_cfg.model, store)
        from . import codex

        return codex.make(model_cfg.model, store)
    if kind == "claude-cli":
        from . import claude_cli

        return claude_cli.make(model_cfg.model)
    raise ValueError(f"unknown provider: {kind!r}")
