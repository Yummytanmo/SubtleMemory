from __future__ import annotations

import re
from typing import Any


TOKEN_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\s]",
    re.UNICODE,
)


def count_conversation_turns(messages: list[Any]) -> int:
    return len(messages) // 2


def count_conversation_tokens(messages: list[Any]) -> int:
    return sum(count_text_tokens(message_content(message)) for message in messages)


def count_text_tokens(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(TOKEN_RE.findall(value))
    if isinstance(value, list):
        return sum(count_text_tokens(item) for item in value)
    if isinstance(value, dict):
        return sum(count_text_tokens(item) for item in value.values())
    return len(TOKEN_RE.findall(str(value)))


def message_content(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)
