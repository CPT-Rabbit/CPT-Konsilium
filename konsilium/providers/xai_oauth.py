"""M3 — `xai-oauth` provider: Grok subscription (super Grok).

A thin wrapper over OAuthProvider with xAI endpoints.
Reference pattern: `XAI_OAUTH_CLIENT_ID`, `auth.x.ai`, `api.x.ai/v1` (hermes_cli/auth.py).
"""

from __future__ import annotations

from ._oauth import OAuthProvider
from .credential_pool import AuthStore, CredentialPool

XAI_INFERENCE_BASE_URL = "https://api.x.ai/v1"
XAI_TOKEN_URL = "https://auth.x.ai/oauth2/token"  # TODO: confirm the exact path
XAI_OAUTH_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"


def make(model: str, store: AuthStore) -> OAuthProvider:
    return OAuthProvider(
        kind="xai-oauth",
        model=model,
        inference_base_url=XAI_INFERENCE_BASE_URL,
        token_url=XAI_TOKEN_URL,
        client_id=XAI_OAUTH_CLIENT_ID,
        pool=CredentialPool("xai-oauth", store),
    )
