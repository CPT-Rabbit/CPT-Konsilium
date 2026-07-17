"""Small pure utilities, no dependencies."""

from __future__ import annotations

import random
import re
import time


def jittered_backoff(
    attempt: int,
    *,
    base_delay: float = 5.0,
    max_delay: float = 120.0,
    jitter_ratio: float = 0.5,
) -> float:
    """Retry delay: min(base*2^(attempt-1), max) + random jitter.

    Jitter decorrelates parallel retries. Reference: `agent/retry_utils.jittered_backoff`.
    """
    exponent = max(0, attempt - 1)
    if exponent >= 63 or base_delay <= 0:
        delay = max_delay
    else:
        delay = min(base_delay * (2 ** exponent), max_delay)
    seed = (time.time_ns() ^ (attempt * 0x9E3779B9)) & 0xFFFFFFFF
    jitter = random.Random(seed).uniform(0, jitter_ratio * delay)
    return delay + jitter


_THINK_PAIRS = [
    r"<think>.*?</think>",
    r"<thinking>.*?</thinking>",
    r"<reasoning>.*?</reasoning>",
    r"<thought>.*?</thought>",
    r"<REASONING_SCRATCHPAD>.*?</REASONING_SCRATCHPAD>",
]
# Unclosed opening tag → cut from it to the end (the model didn't close the block).
_THINK_OPEN_TO_END = r"<(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)\b[^>]*>.*\Z"
# Dangling orphan tag (opening or closing) — remove the tag itself.
_THINK_DANGLING = r"</?(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)\b[^>]*>"


def strip_think_blocks(content: str | None) -> str:
    """Strip reasoning/think blocks, keep the visible text.

    Cures "kimi burns tokens in reasoning_content": thinking models wrap
    reasoning in tags — they are not needed in the journal/answer.
    Reference: `run_agent.py::_strip_think_blocks` (ported almost verbatim).
    """
    if not content:
        return ""
    for pat in _THINK_PAIRS:
        content = re.sub(pat, "", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(_THINK_OPEN_TO_END, "", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(_THINK_DANGLING, "", content, flags=re.IGNORECASE)
    return content.strip()


def json_block(text: str) -> str:
    """Return JSON from a fenced model response, preserving bare JSON."""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S | re.I)
    return match.group(1) if match else text
