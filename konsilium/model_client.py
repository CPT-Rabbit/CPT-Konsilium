"""M4/M5 — Model client: request assembly + call with OUR reliability control.

The main lesson from dissecting Hermes: a provider can "hang" — send the first chunk
and go silent. The default httpx timeout is huge (up to 1800s), so without our own
**stale detector** the call hangs, piles into retries and kills the trading cycle (exactly
the v0.15.1 regression). Here we control that ourselves: kill the call on a short silence
threshold and retry with jittered backoff; on 401 — refresh/rotate via the provider.

Reference pattern: `_interruptible_api_call` / `_interruptible_streaming_api_call`
(stale detector) + `_build_api_kwargs` + `retry_utils.jittered_backoff`. Our code.
"""

from __future__ import annotations

import threading
import time
from queue import Empty, Full, Queue
from dataclasses import dataclass, field
from typing import Any

import openai

from .providers.base import Provider
from .util import jittered_backoff, strip_think_blocks


@dataclass
class ModelResponse:
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str = "stop"
    reasoning: str = ""
    reasoning_items: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


class ModelTimeout(Exception):
    """Provider exceeded the streaming-silence or absolute request deadline."""


class ModelBadRequest(Exception):
    """Provider rejected the assembled request; the loop may compact and retry once."""


class ModelClient:
    def __init__(
        self,
        provider: Provider,
        *,
        max_tokens: int = 4096,
        request_timeout_s: float = 120.0,
        stale_timeout_s: float = 45.0,
        max_retries: int = 4,
        retry_base_delay_s: float = 5.0,
        rate_limit_base_delay_s: float = 30.0,
        retry_max_delay_s: float = 180.0,
        stream: bool = True,
        event_sink=None,
    ):
        self.provider = provider
        self.max_tokens = max_tokens
        self.request_timeout_s = request_timeout_s
        self.stale_timeout_s = stale_timeout_s   # silence threshold — short, that's the whole point
        self.max_retries = max_retries
        self.retry_base_delay_s = retry_base_delay_s
        self.rate_limit_base_delay_s = rate_limit_base_delay_s
        self.retry_max_delay_s = retry_max_delay_s
        self.stream = stream
        self.event_sink = event_sink

    # ── M5: request assembly ──────────────────────────────────────────
    def build_kwargs(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[dict],
        *,
        json_mode: bool = False,
    ) -> dict:
        api_messages = [{"role": "system", "content": system_prompt}, *messages]
        kwargs: dict[str, Any] = {
            "messages": api_messages,
            "max_tokens": 4096 if json_mode else self.max_tokens,
            "stream": False if json_mode else self.stream,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if tools:
            kwargs["tools"] = tools
        return kwargs

    # ── M4: call with retries + stale detector ─────────────────────
    def call(self, kwargs: dict) -> ModelResponse:
        last_exc: Exception | None = None
        deadline = time.monotonic() + self.request_timeout_s
        call_started = time.monotonic()
        self._emit("model_call_started", self._request_metrics(kwargs))
        for attempt in range(1, self.max_retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                error = ModelTimeout(f"model call absolute deadline > {self.request_timeout_s}s")
                self._emit_failure(error, call_started)
                raise error
            if getattr(self.provider, "kind", "") == "claude-cli":
                try:
                    content = self.provider.complete(kwargs, timeout_s=remaining)
                    response = ModelResponse(content=strip_think_blocks(content))
                    self._emit("model_call_finished", {
                        "duration_ms": int((time.monotonic() - call_started) * 1000),
                        "attempt": attempt,
                        "finish_reason": response.finish_reason,
                        "content_chars": len(response.content or ""),
                        "tool_call_count": 0,
                    })
                    return response
                except Exception as e:
                    self._emit_failure(e, call_started)
                    raise
            resolved = self.provider.resolve()       # fresh token/key on every attempt
            client = openai.OpenAI(
                base_url=resolved.base_url,
                api_key=resolved.api_key,
                timeout=max(0.1, remaining),
                max_retries=0,                        # we retry OURSELVES (controlled)
                default_headers=resolved.extra_headers or None,
            )
            call_kwargs = {**kwargs, "model": resolved.model}
            try:
                self._emit("model_call_attempt_started", {"attempt": attempt, "remaining_ms": int(remaining * 1000)})
                if getattr(self.provider, "kind", "custom") == "codex":
                    response = self._call_responses_blocking(client, call_kwargs, deadline=deadline)
                elif call_kwargs.get("stream"):
                    response = self._call_streaming(client, call_kwargs, deadline=deadline)
                else:
                    response = self._call_blocking(client, call_kwargs, deadline=deadline)
                self._emit("model_call_finished", {
                    "duration_ms": int((time.monotonic() - call_started) * 1000),
                    "attempt": attempt,
                    "finish_reason": response.finish_reason,
                    "content_chars": len(response.content or ""),
                    "tool_call_count": len(response.tool_calls or []),
                })
                return response
            except openai.AuthenticationError as e:
                last_exc = e
                if not self.provider.on_auth_error():
                    self._emit_failure(e, call_started)
                    raise
                continue                              # token refreshed — retry without backoff
            except openai.BadRequestError as e:
                error = ModelBadRequest(str(e))
                self._emit_failure(error, call_started)
                raise error from e
            except (ModelTimeout, openai.APITimeoutError, openai.APIConnectionError,
                    openai.RateLimitError, openai.InternalServerError) as e:
                last_exc = e
                if attempt < self.max_retries:
                    delay = min(self._retry_delay(attempt, e), max(0.0, deadline - time.monotonic()))
                    if delay > 0:
                        time.sleep(delay)
                continue
        error = last_exc or RuntimeError("model call failed")
        self._emit_failure(error, call_started)
        raise error

    def _emit(self, event_type: str, payload: dict | None = None) -> None:
        if self.event_sink is not None:
            self.event_sink.emit(event_type, payload or {})

    def _emit_failure(self, error: Exception, started: float) -> None:
        self._emit("model_call_failed", {
            "duration_ms": int((time.monotonic() - started) * 1000),
            "error_type": error.__class__.__name__,
            "retryable": isinstance(error, (
                ModelTimeout,
                ModelBadRequest,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.RateLimitError,
                openai.InternalServerError,
            )),
        })

    def _request_metrics(self, kwargs: dict) -> dict:
        messages = kwargs.get("messages") or []
        tools = kwargs.get("tools") or []
        system_chars = 0
        message_chars = 0
        for message in messages:
            content = str(message.get("content") or "")
            message_chars += len(content)
            if message.get("role") == "system":
                system_chars += len(content)
        tool_schema_chars = sum(len(str(tool)) for tool in tools)
        return {
            "message_count": len(messages),
            "message_chars": message_chars,
            "system_chars": system_chars,
            "tool_count": len(tools),
            "tool_schema_chars": tool_schema_chars,
            "approx_input_units": (message_chars + tool_schema_chars) // 4,
            "stream": bool(kwargs.get("stream")),
            "max_output_units": kwargs.get("max_tokens"),
        }

    def _retry_delay(self, attempt: int, error: Exception) -> float:
        base_delay = (
            self.rate_limit_base_delay_s
            if isinstance(error, openai.RateLimitError)
            else self.retry_base_delay_s
        )
        return jittered_backoff(
            attempt,
            base_delay=base_delay,
            max_delay=self.retry_max_delay_s,
        )

    def _call_streaming(self, client, kwargs: dict, *, deadline: float | None = None) -> ModelResponse:
        """Stream with silence and absolute-deadline detectors."""
        stream = client.chat.completions.create(**kwargs)
        deadline = deadline or (time.monotonic() + self.request_timeout_s)
        content_parts: list[str] = []
        tool_acc: dict[int, dict] = {}
        finish = "stop"
        last_chunk = time.monotonic()
        chunks: Queue = Queue(maxsize=64)
        done = object()
        stop_reader = threading.Event()

        def _read_stream() -> None:
            try:
                for item in stream:
                    if stop_reader.is_set():
                        break
                    while not stop_reader.is_set():
                        try:
                            chunks.put(item, timeout=0.1)
                            break
                        except Full:
                            continue
            except Exception as error:  # noqa: BLE001 - propagate from reader thread
                if not stop_reader.is_set():
                    chunks.put(error)
            finally:
                if not stop_reader.is_set():
                    chunks.put(done)

        threading.Thread(target=_read_stream, daemon=True).start()
        while True:
            now = time.monotonic()
            remaining_absolute = deadline - now
            remaining_stale = self.stale_timeout_s - (now - last_chunk)
            if remaining_absolute <= 0:
                stop_reader.set()
                try:
                    stream.close()
                except Exception:
                    pass
                raise ModelTimeout("stream absolute deadline exceeded")
            if remaining_stale <= 0:
                stop_reader.set()
                try:
                    stream.close()
                except Exception:
                    pass
                raise ModelTimeout(f"stream stale > {self.stale_timeout_s}s")
            try:
                chunk = chunks.get(timeout=min(remaining_absolute, remaining_stale, 0.5))
            except Empty:
                continue
            if chunk is done:
                break
            if isinstance(chunk, Exception):
                stop_reader.set()
                raise chunk
            last_chunk = time.monotonic()
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                content_parts.append(delta.content)
            if delta and delta.tool_calls:
                self._accumulate_tool_calls(tool_acc, delta.tool_calls)
            if chunk.choices[0].finish_reason:
                finish = chunk.choices[0].finish_reason
        return self._normalize("".join(content_parts), list(tool_acc.values()), finish)

    def _call_blocking(self, client, kwargs: dict, *, deadline: float | None = None) -> ModelResponse:
        """Non-stream: call in a background thread with an absolute-deadline watchdog."""
        box: dict[str, Any] = {}

        def _run():
            try:
                box["resp"] = client.chat.completions.create(**kwargs)
            except Exception as e:  # noqa: BLE001 — propagate to the main thread
                box["err"] = e
            finally:
                box["completed_at"] = time.monotonic()

        t = threading.Thread(target=_run, daemon=True)
        deadline = deadline or (time.monotonic() + self.request_timeout_s)
        t.start()
        while t.is_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ModelTimeout("blocking call absolute deadline exceeded")
            t.join(timeout=min(0.3, remaining))
        if box["completed_at"] > deadline:
            raise ModelTimeout("blocking call absolute deadline exceeded")
        if "err" in box:
            raise box["err"]
        resp = box["resp"]
        msg = resp.choices[0].message
        tcs = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in (msg.tool_calls or [])
        ]
        return self._normalize(msg.content or "", tcs, resp.choices[0].finish_reason or "stop")

    def _call_responses_blocking(
        self,
        client,
        kwargs: dict,
        *,
        deadline: float | None = None,
    ) -> ModelResponse:
        """Codex subscription backend uses Responses API, not Chat Completions."""
        from .providers.codex_responses import build_responses_kwargs, normalize_response, normalize_stream_events

        box: dict[str, Any] = {}

        def _run():
            try:
                request = build_responses_kwargs(kwargs)
                if kwargs.get("stream"):
                    with client.responses.stream(**request) as stream:
                        box["normalized"] = normalize_stream_events(stream)
                else:
                    box["normalized"] = normalize_response(client.responses.create(**request))
            except Exception as error:  # noqa: BLE001 - propagate to the main thread
                box["err"] = error
            finally:
                box["completed_at"] = time.monotonic()

        thread = threading.Thread(target=_run, daemon=True)
        deadline = deadline or (time.monotonic() + self.request_timeout_s)
        thread.start()
        while thread.is_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ModelTimeout("Responses call absolute deadline exceeded")
            thread.join(timeout=min(0.3, remaining))
        if box["completed_at"] > deadline:
            raise ModelTimeout("Responses call absolute deadline exceeded")
        if "err" in box:
            raise box["err"]
        normalized = box["normalized"]
        return ModelResponse(
            content=normalized.content,
            tool_calls=normalized.tool_calls,
            finish_reason=normalized.finish_reason,
            reasoning=normalized.reasoning_summary,
            reasoning_items=normalized.reasoning_items,
            usage=normalized.usage,
        )

    @staticmethod
    def _accumulate_tool_calls(acc: dict[int, dict], deltas) -> None:
        for tc in deltas:
            slot = acc.setdefault(tc.index, {"id": None, "type": "function",
                                             "function": {"name": "", "arguments": ""}})
            if tc.id:
                slot["id"] = tc.id
            if tc.function and tc.function.name:
                slot["function"]["name"] = tc.function.name
            if tc.function and tc.function.arguments:
                slot["function"]["arguments"] += tc.function.arguments

    def _normalize(self, raw_content: str, tool_calls: list[dict], finish: str) -> ModelResponse:
        # tool_calls instead of text → finish_reason="tool_calls" (routers sometimes drop it)
        if tool_calls and finish not in ("tool_calls", "length"):
            finish = "tool_calls"
        return ModelResponse(
            content=strip_think_blocks(raw_content),   # M5: strip think blocks
            tool_calls=tool_calls,
            finish_reason=finish,
        )

    # ── accessors for loop.py ───────────────────────────────────────
    @staticmethod
    def finish_reason(resp: ModelResponse) -> str:
        return resp.finish_reason

    @staticmethod
    def assistant_message(resp: ModelResponse) -> dict:
        msg: dict[str, Any] = {"role": "assistant", "content": resp.content}
        if resp.tool_calls:
            msg["tool_calls"] = resp.tool_calls
        if resp.reasoning:
            msg["reasoning_summary"] = resp.reasoning
        if resp.reasoning_items:
            msg["_codex_reasoning_items"] = resp.reasoning_items
        return msg
