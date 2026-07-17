"""Shared base for subscription OAuth providers (Grok, ChatGPT/Codex).

A subscription = model access via an OAuth token, not an API key. A one-time
PKCE login (outside the runtime, via a CLI tool) places access/refresh into auth.json; here
is the runtime part: take the active token, refresh it via refresh_token on expiry,
and on 401 rotate the pool entry.

Standard OAuth2 refresh over the shared credential pool.

⚠ Air-gap: subscription paths go NOT to the gateway but to the provider endpoint
(inference + token refresh). These hosts must be added to the tinyproxy egress allowlist
(as the CF-gateway is allowed now). See DESIGN.md.
"""

from __future__ import annotations

import time

from .base import Provider, Resolved
from .credential_pool import CredentialPool


class OAuthProvider(Provider):
    """Base subscription provider on top of CredentialPool."""

    def __init__(
        self,
        *,
        kind: str,
        model: str,
        inference_base_url: str,
        token_url: str,
        client_id: str,
        pool: CredentialPool,
    ):
        self.kind = kind
        self._model = model
        self._inference_base_url = inference_base_url.rstrip("/")
        self._token_url = token_url
        self._client_id = client_id
        self._pool = pool

    def resolve(self) -> Resolved:
        cred = self._pool.current()
        if cred is None:
            raise RuntimeError(f"{self.kind}: no credentials in the pool (PKCE login required)")
        if cred.is_expiring():
            self._refresh(cred)
        return Resolved(
            base_url=cred.base_url or self._inference_base_url,
            api_key=cred.access_token,
            model=self._model,
        )

    def on_auth_error(self) -> bool:
        cred = self._pool.current()
        if cred is not None and cred.refresh_token:
            try:
                self._refresh(cred)
                return True
            except Exception:
                pass
        # refresh didn't help — try the next pool entry
        return self._pool.rotate() is not None

    def _refresh(self, cred) -> None:
        """OAuth2 refresh_token grant → update access/expires in the pool."""
        import httpx

        if not cred.refresh_token:
            raise RuntimeError(f"{self.kind}: no refresh_token")
        resp = httpx.post(
            self._token_url,
            data={
                "grant_type": "refresh_token",
                "refresh_token": cred.refresh_token,
                "client_id": self._client_id,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        tok = resp.json()
        cred.access_token = tok["access_token"]
        if tok.get("refresh_token"):
            cred.refresh_token = tok["refresh_token"]
        if tok.get("expires_in"):
            cred.expires_at = _iso_in(int(tok["expires_in"]))
        cred.extra["last_refresh"] = _iso_in(0)
        self._pool.persist()


def _iso_in(seconds: int) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(time.time() + seconds, tz=timezone.utc).isoformat()
