from __future__ import annotations

import json
import shutil
import subprocess


class ClaudeCliProvider:
    kind = "claude-cli"

    def __init__(self, model: str):
        self._model = model

    def complete(self, kwargs: dict, *, timeout_s: float) -> str:
        exe = shutil.which("claude")
        if not exe:
            raise RuntimeError("claude-cli provider requires `claude` on PATH")
        system_prompt, prompt = _prompt(kwargs.get("messages", []))
        cmd = [
            exe,
            "-p",
            prompt,
            "--output-format",
            "json",
            "--no-session-persistence",
            "--tools",
            "",
        ]
        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])
        if self._model:
            cmd.extend(["--model", self._model])
        run = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_s, check=False)
        if run.returncode:
            detail = (run.stderr or run.stdout).strip().splitlines()[-1:]
            raise RuntimeError(f"claude-cli failed: {detail[0] if detail else run.returncode}")
        return _content(run.stdout)

    def on_auth_error(self) -> bool:
        return False


def make(model: str) -> ClaudeCliProvider:
    return ClaudeCliProvider(model)


def _prompt(messages: list[dict]) -> tuple[str, str]:
    system = []
    parts = []
    for message in messages:
        role = message.get("role", "user")
        if role == "system":
            system.append(str(message.get("content", "")))
            continue
        parts.append(f"{role.upper()}:\n{message.get('content', '')}")
    return "\n\n".join(system), "\n\n".join(parts)


def _content(stdout: str) -> str:
    data = json.loads(stdout)
    for key in ("result", "content", "text"):
        if isinstance(data.get(key), str):
            return data[key]
    message = data.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    raise RuntimeError("claude-cli returned JSON without text content")
