"""M3 — `codex` provider: ChatGPT subscription (Codex backend).

A thin wrapper over OAuthProvider with OpenAI/Codex endpoints.
Reference pattern: ProviderConfig `openai-codex`, `auth.openai.com/oauth/token`,
`chatgpt.com/backend`.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable, Any

from ._oauth import OAuthProvider
from .credential_pool import AuthStore, Credential, CredentialPool

CODEX_INFERENCE_BASE_URL = "https://chatgpt.com/backend-api/codex"
CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_ISSUER = "https://auth.openai.com"


def make(model: str, store: AuthStore) -> OAuthProvider:
    return OAuthProvider(
        kind="codex",
        model=model,
        inference_base_url=CODEX_INFERENCE_BASE_URL,
        token_url=CODEX_TOKEN_URL,
        client_id=CODEX_OAUTH_CLIENT_ID,
        pool=CredentialPool("codex", store),
    )


def codex_device_login(
    store: AuthStore,
    *,
    notify: Callable[[dict[str, str]], None],
    sleep: Callable[[float], None] = time.sleep,
    client_factory: Callable[[], Any] | None = None,
    timeout_seconds: float = 900.0,
) -> Credential:
    """Authorize an independent Codex session through a browser device code."""
    if client_factory is None:
        import httpx

        factory = lambda: httpx.Client(timeout=httpx.Timeout(15.0))
    else:
        factory = client_factory
    with factory() as client:
        response = client.post(
            f"{CODEX_ISSUER}/api/accounts/deviceauth/usercode",
            json={"client_id": CODEX_OAUTH_CLIENT_ID},
            headers={"Content-Type": "application/json"},
        )
        if response.status_code != 200:
            raise RuntimeError(f"Codex device-code request failed ({response.status_code})")
        payload = response.json()
        user_code = str(payload.get("user_code") or "")
        device_auth_id = str(payload.get("device_auth_id") or "")
        if not user_code or not device_auth_id:
            raise RuntimeError("Codex device-code response is incomplete")
        interval = max(0.0, float(payload.get("interval") or 5))

        notify({
            "verification_url": f"{CODEX_ISSUER}/codex/device",
            "user_code": user_code,
        })
        deadline = time.monotonic() + timeout_seconds
        authorization = None
        while time.monotonic() < deadline:
            sleep(interval)
            poll = client.post(
                f"{CODEX_ISSUER}/api/accounts/deviceauth/token",
                json={"device_auth_id": device_auth_id, "user_code": user_code},
                headers={"Content-Type": "application/json"},
            )
            if poll.status_code == 200:
                authorization = poll.json()
                break
            if poll.status_code not in {403, 404}:
                raise RuntimeError(f"Codex device authorization failed ({poll.status_code})")
        if authorization is None:
            raise RuntimeError("Codex device authorization timed out")

        authorization_code = str(authorization.get("authorization_code") or "")
        code_verifier = str(authorization.get("code_verifier") or "")
        if not authorization_code or not code_verifier:
            raise RuntimeError("Codex device authorization response is incomplete")
        token = client.post(
            CODEX_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": authorization_code,
                "redirect_uri": f"{CODEX_ISSUER}/deviceauth/callback",
                "client_id": CODEX_OAUTH_CLIENT_ID,
                "code_verifier": code_verifier,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if token.status_code != 200:
            raise RuntimeError(f"Codex token exchange failed ({token.status_code})")
        token_payload = token.json()

    access_token = str(token_payload.get("access_token") or "")
    refresh_token = str(token_payload.get("refresh_token") or "")
    if not access_token or not refresh_token:
        raise RuntimeError("Codex token exchange did not return complete credentials")
    expires_in = int(token_payload.get("expires_in") or 3600)
    expires_at = datetime.fromtimestamp(
        time.time() + expires_in,
        tz=timezone.utc,
    ).isoformat()
    credential = Credential(
        provider="codex",
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=expires_at,
        base_url=CODEX_INFERENCE_BASE_URL,
        source="device-code",
    )
    store.save("codex", [credential])
    return credential
