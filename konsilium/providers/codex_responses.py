"""Small Chat Completions → Codex Responses API compatibility layer.

Only observable summaries are exposed. Encrypted reasoning is retained as an
opaque provider item solely for multi-step continuity and is never decoded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ResponsesResult:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"
    reasoning_summary: str = ""
    reasoning_items: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        str(part.get("text") or "")
        for part in content
        if isinstance(part, dict) and part.get("type") in {"text", "input_text", "output_text"}
    )


def _replay_reasoning(message: dict[str, Any]) -> list[dict[str, Any]]:
    replay = []
    for item in message.get("_codex_reasoning_items") or []:
        if not isinstance(item, dict) or not item.get("encrypted_content"):
            continue
        clean = {
            "type": "reasoning",
            "encrypted_content": str(item["encrypted_content"]),
        }
        summary = item.get("summary")
        if isinstance(summary, list):
            clean["summary"] = [
                {"type": "summary_text", "text": str(part.get("text") or "")}
                for part in summary
                if isinstance(part, dict) and part.get("text")
            ]
        replay.append(clean)
    return replay


def _input_items(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    instructions = []
    items = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            instructions.append(_content_text(message.get("content")))
            continue
        if role == "assistant":
            items.extend(_replay_reasoning(message))
            content = _content_text(message.get("content"))
            if content:
                items.append({"role": "assistant", "content": content})
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                arguments = function.get("arguments") or "{}"
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                items.append({
                    "type": "function_call",
                    "call_id": str(call.get("call_id") or call.get("id") or ""),
                    "name": str(function.get("name") or ""),
                    "arguments": arguments,
                })
            continue
        if role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": str(message.get("tool_call_id") or ""),
                "output": _content_text(message.get("content")),
            })
            continue
        items.append({"role": str(role or "user"), "content": _content_text(message.get("content"))})
    return "\n\n".join(part for part in instructions if part), items


def _tools(chat_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for tool in chat_tools:
        function = tool.get("function") or {}
        converted = {
            "type": "function",
            "name": function.get("name"),
            "parameters": function.get("parameters") or {"type": "object", "properties": {}},
        }
        if function.get("description"):
            converted["description"] = function["description"]
        if "strict" in function:
            converted["strict"] = bool(function["strict"])
        result.append(converted)
    return result


def build_responses_kwargs(chat_kwargs: dict[str, Any]) -> dict[str, Any]:
    instructions, items = _input_items(chat_kwargs.get("messages") or [])
    result: dict[str, Any] = {
        "input": items,
        "instructions": instructions,
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "reasoning": {"summary": "auto"},
    }
    if chat_kwargs.get("model"):
        result["model"] = chat_kwargs["model"]
    if chat_kwargs.get("max_tokens"):
        result["max_output_tokens"] = chat_kwargs["max_tokens"]
    if chat_kwargs.get("response_format") == {"type": "json_object"}:
        result["text"] = {"format": {"type": "json_object"}}
    if chat_kwargs.get("tools"):
        result["tools"] = _tools(chat_kwargs["tools"])
    return result


def _attr(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def normalize_response(response: Any) -> ResponsesResult:
    content = []
    tool_calls = []
    summaries = []
    reasoning_items = []
    for item in _attr(response, "output", []) or []:
        item_type = _attr(item, "type", "")
        if item_type == "message":
            for part in _attr(item, "content", []) or []:
                if _attr(part, "type", "") in {"output_text", "text"}:
                    text = _attr(part, "text", "")
                    if text:
                        content.append(str(text))
        elif item_type == "function_call":
            tool_calls.append({
                "id": str(_attr(item, "call_id", "")),
                "type": "function",
                "function": {
                    "name": str(_attr(item, "name", "")),
                    "arguments": str(_attr(item, "arguments", "{}")),
                },
            })
        elif item_type == "reasoning":
            summary = []
            for part in _attr(item, "summary", []) or []:
                text = _attr(part, "text", "")
                if text:
                    summaries.append(str(text))
                    summary.append({"type": "summary_text", "text": str(text)})
            encrypted = _attr(item, "encrypted_content", "")
            if encrypted:
                reasoning_items.append({
                    "type": "reasoning",
                    "encrypted_content": str(encrypted),
                    "summary": summary,
                })

    usage = _attr(response, "usage", None)
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    if not isinstance(usage, dict):
        usage = {}
    return ResponsesResult(
        content="\n".join(content).strip(),
        tool_calls=tool_calls,
        finish_reason="tool_calls" if tool_calls else "stop",
        reasoning_summary="\n".join(summaries).strip(),
        reasoning_items=reasoning_items,
        usage=usage,
    )


def normalize_stream_events(events: Any) -> ResponsesResult:
    """Aggregate the Codex backend SSE shape without relying on SDK final state."""
    text_parts = []
    reasoning_summary_parts = []
    output_items = []
    completed_response = None
    saw_completed = False
    for event in events:
        event_type = _attr(event, "type", "")
        if event_type == "response.output_text.delta":
            delta = _attr(event, "delta", "")
            if delta:
                text_parts.append(str(delta))
        elif event_type == "response.reasoning_summary_text.delta":
            delta = _attr(event, "delta", "")
            if delta:
                reasoning_summary_parts.append(str(delta))
        elif event_type == "response.output_item.done":
            item = _attr(event, "item", None)
            if item is not None:
                output_items.append(item)
        elif event_type == "response.completed":
            completed_response = _attr(event, "response", None)
            saw_completed = True
        elif event_type == "response.incomplete":
            response = _attr(event, "response", {}) or {}
            details = _attr(response, "incomplete_details", {}) or {}
            reason = _attr(details, "reason", "unknown")
            raise RuntimeError(f"Codex response incomplete: {reason}")
        elif event_type == "response.failed":
            response = _attr(event, "response", {}) or {}
            error = _attr(response, "error", {}) or {}
            code = str(_attr(error, "code", "") or "").strip()
            message = str(_attr(error, "message", "") or "").strip()
            detail = ": ".join(part for part in (code, message) if part) or "unknown error"
            raise RuntimeError(f"Codex response failed: {detail}")

    if not saw_completed:
        raise RuntimeError("Codex stream closed before response.completed")

    response = completed_response or {}
    normalized = normalize_response({
        "output": output_items,
        "status": _attr(response, "status", "completed"),
        "usage": _attr(response, "usage", None),
    })
    streamed_text = "".join(text_parts).strip()
    streamed_summary = "".join(reasoning_summary_parts).strip()
    content = normalized.content or streamed_text
    reasoning_summary = normalized.reasoning_summary or streamed_summary
    if content == normalized.content and reasoning_summary == normalized.reasoning_summary:
        return normalized
    return ResponsesResult(
        content=content,
        tool_calls=normalized.tool_calls,
        finish_reason=normalized.finish_reason,
        reasoning_summary=reasoning_summary,
        reasoning_items=normalized.reasoning_items,
        usage=normalized.usage,
    )
