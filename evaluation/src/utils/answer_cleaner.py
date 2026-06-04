"""Answer text cleanup shared by answer-generating adapters."""

from __future__ import annotations

import re
from typing import Any


HARMONY_FINAL_MARKER = "<|channel|>final<|message|>"
HARMONY_END_MARKER = "<|end|>"
HARMONY_START_ROLE_RE = re.compile(
    r"<\|start\|>\s*(?:assistant|user|system|developer)",
    flags=re.IGNORECASE,
)
HARMONY_TOKEN_RE = re.compile(r"<\|[^|]*\|>")


def clean_answer_text(answer: Any) -> str:
    """Normalize model answers without changing ordinary natural-language text.

    Some OpenAI-compatible gpt-oss deployments can return Harmony-style channel
    tokens. For benchmark answers, only the final-channel message should be
    scored. Plain model outputs are left unchanged apart from whitespace and the
    pre-existing ``FINAL ANSWER:`` prefix handling.
    """
    text = str(answer or "").strip()
    if not text:
        return ""

    if HARMONY_FINAL_MARKER in text:
        text = text.rsplit(HARMONY_FINAL_MARKER, maxsplit=1)[-1].strip()

    if HARMONY_END_MARKER in text:
        text = text.split(HARMONY_END_MARKER, maxsplit=1)[0].strip()

    text = HARMONY_START_ROLE_RE.sub("", text).strip()
    text = HARMONY_TOKEN_RE.sub("", text).strip()

    if "FINAL ANSWER:" in text:
        text = text.split("FINAL ANSWER:")[-1].strip()

    return text
