from __future__ import annotations

import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from konsilium.providers.codex import codex_device_login
from konsilium.providers.claude_cli import ClaudeCliProvider
from konsilium.providers.codex_responses import build_responses_kwargs, normalize_stream_events
from konsilium.providers.credential_pool import AuthStore


class _Response:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _Client:
    def __init__(self, responses: list[_Response]):
        self.responses = iter(responses)
        self.requests = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def post(self, url: str, **kwargs):
        self.requests.append((url, kwargs))
        return next(self.responses)


class _SlowCompletions:
    def __init__(self, delay_s: float):
        self.delay_s = delay_s

    def create(self, **kwargs):
        time.sleep(self.delay_s)
        message = SimpleNamespace(content='{"ok":true}', tool_calls=[])
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


class _SlowModelClient:
    def __init__(self, delay_s: float):
        self.chat = SimpleNamespace(completions=_SlowCompletions(delay_s))
        self.responses = SimpleNamespace(create=lambda **kwargs: time.sleep(delay_s))


def _model_runtime():
    try:
        from konsilium.model_client import ModelClient, ModelTimeout
    except ModuleNotFoundError as error:
        raise unittest.SkipTest(f"model runtime dependency is unavailable: {error.name}") from error
    return ModelClient, ModelTimeout


class ModelProviderTest(unittest.TestCase):
    def test_blocking_call_ignores_stream_stale_threshold(self) -> None:
        ModelClient, _ = _model_runtime()

        model = ModelClient(
            SimpleNamespace(kind="custom"),
            request_timeout_s=0.3,
            stale_timeout_s=0.01,
        )

        response = model._call_blocking(_SlowModelClient(0.06), {"stream": False})

        self.assertEqual(response.content, '{"ok":true}')

    def test_blocking_call_honors_absolute_deadline(self) -> None:
        ModelClient, ModelTimeout = _model_runtime()

        model = ModelClient(SimpleNamespace(kind="custom"), request_timeout_s=0.03)
        started = time.monotonic()

        with self.assertRaisesRegex(ModelTimeout, "absolute deadline"):
            model._call_blocking(_SlowModelClient(0.2), {"stream": False})

        self.assertLess(time.monotonic() - started, 0.15)

    def test_responses_blocking_call_ignores_stream_stale_threshold(self) -> None:
        ModelClient, _ = _model_runtime()

        normalized = SimpleNamespace(
            content='{"ok":true}',
            tool_calls=[],
            finish_reason="stop",
            reasoning_summary="",
            reasoning_items=[],
            usage={},
        )
        model = ModelClient(
            SimpleNamespace(kind="codex"),
            request_timeout_s=0.3,
            stale_timeout_s=0.01,
        )

        with patch("konsilium.providers.codex_responses.normalize_response", return_value=normalized):
            response = model._call_responses_blocking(_SlowModelClient(0.06), {"stream": False})

        self.assertEqual(response.content, '{"ok":true}')

    def test_auth_store_defaults_to_konsilium_env(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "auth.json"
            previous = os.environ.get("KONSILIUM_AUTH_FILE")
            os.environ["KONSILIUM_AUTH_FILE"] = str(path)
            try:
                store = AuthStore()
                store.save("codex", [])
                self.assertEqual(store.path, path)
                self.assertTrue(path.exists())
            finally:
                if previous is None:
                    os.environ.pop("KONSILIUM_AUTH_FILE", None)
                else:
                    os.environ["KONSILIUM_AUTH_FILE"] = previous

    def test_codex_device_login_saves_subscription_tokens(self) -> None:
        with TemporaryDirectory() as tmp:
            client = _Client(
                [
                    _Response(200, {"user_code": "ABCD", "device_auth_id": "device-1", "interval": 0}),
                    _Response(200, {"authorization_code": "code-1", "code_verifier": "verifier-1"}),
                    _Response(
                        200,
                        {"access_token": "test_access_token", "refresh_token": "test_refresh_token"},
                    ),
                ]
            )
            notices = []
            store = AuthStore(Path(tmp) / "auth.json")

            credential = codex_device_login(
                store,
                notify=notices.append,
                sleep=lambda _: None,
                client_factory=lambda: client,
                timeout_seconds=10,
            )

            self.assertEqual(notices[0]["verification_url"], "https://auth.openai.com/codex/device")
            self.assertEqual(credential.access_token, "test_access_token")
            self.assertEqual(store.load("codex")[0].source, "device-code")
            self.assertEqual(client.requests[-1][1]["data"]["code_verifier"], "verifier-1")

    def test_codex_responses_adapter_keeps_tools_and_reasoning_summary(self) -> None:
        kwargs = build_responses_kwargs(
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "ping"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {"name": "case_review", "arguments": "{}"},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call-1", "content": "{}"},
                ],
                "tools": [{"function": {"name": "case_review", "parameters": {"type": "object"}}}],
                "model": "gpt-5.5",
            }
        )
        self.assertEqual(kwargs["instructions"], "system")
        self.assertEqual(kwargs["tools"][0]["name"], "case_review")
        self.assertEqual(kwargs["input"][2]["type"], "function_call_output")

        json_kwargs = build_responses_kwargs({
            "messages": [],
            "max_tokens": 4096,
            "response_format": {"type": "json_object"},
        })
        self.assertEqual(json_kwargs["max_output_tokens"], 4096)
        self.assertEqual(json_kwargs["text"], {"format": {"type": "json_object"}})

        result = normalize_stream_events(
            [
                SimpleNamespace(type="response.reasoning_summary_text.delta", delta="Checked "),
                SimpleNamespace(type="response.reasoning_summary_text.delta", delta="case."),
                SimpleNamespace(type="response.output_text.delta", delta="OK"),
                SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(status="completed", usage=None),
                ),
            ]
        )
        self.assertEqual(result.content, "OK")
        self.assertEqual(result.reasoning_summary, "Checked case.")

    def test_claude_cli_provider_uses_headless_json_output(self) -> None:
        with TemporaryDirectory() as tmp:
            exe = Path(tmp) / "claude"
            exe.write_text("#!/bin/sh\nprintf '%s\\n' '{\"result\":\"OK\"}'\n", encoding="utf-8")
            exe.chmod(0o700)

            with patch.dict(os.environ, {"PATH": tmp}):
                content = ClaudeCliProvider("sonnet").complete(
                    {"messages": [{"role": "user", "content": "ping"}]},
                    timeout_s=10,
                )

            self.assertEqual(content, "OK")

    def test_claude_cli_provider_fails_loudly_when_missing(self) -> None:
        with patch.dict(os.environ, {"PATH": ""}):
            with self.assertRaises(RuntimeError):
                ClaudeCliProvider("sonnet").complete({"messages": []}, timeout_s=10)


if __name__ == "__main__":
    unittest.main()
