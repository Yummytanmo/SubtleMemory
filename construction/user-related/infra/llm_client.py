from __future__ import annotations

import re
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Iterable

from infra.config import mask_secret
from infra.logging_utils import NullLogger


@dataclass
class StreamResult:
    text: str
    stream_completed: bool
    chunks: int = 0
    error: str | None = None


DEFAULT_LLM_MAX_RETRIES = 3
DEFAULT_LLM_RETRY_INITIAL_DELAY = 1.0
DEFAULT_LLM_RETRY_MAX_DELAY = 20.0
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


class OpenAIStreamingClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        reasoning_effort: str | None = None,
        logger: Any | None = None,
        max_retries: int = DEFAULT_LLM_MAX_RETRIES,
        retry_initial_delay: float = DEFAULT_LLM_RETRY_INITIAL_DELAY,
        retry_max_delay: float = DEFAULT_LLM_RETRY_MAX_DELAY,
        sleep_fn: Callable[[float], None] = time.sleep,
        client: Any | None = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            self.client = OpenAI(base_url=base_url, api_key=api_key)
        else:
            self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.logger = logger or NullLogger()
        self.max_retries = max(0, max_retries)
        self.retry_initial_delay = max(0.0, retry_initial_delay)
        self.retry_max_delay = max(0.0, retry_max_delay)
        self.sleep_fn = sleep_fn

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        phase: str,
        temperature: float = 0.2,
        extra_body: dict[str, Any] | None = None,
    ) -> StreamResult:
        body = dict(extra_body or {})
        if self.reasoning_effort:
            body.setdefault("reasoning_effort", self.reasoning_effort)

        sanitized_messages = sanitize_json_value(messages)
        max_attempts = self.max_retries + 1
        last_text = ""
        last_chunks = 0
        for attempt in range(1, max_attempts + 1):
            self.logger.debug(
                phase,
                "llm_stream_start",
                model=self.model,
                stream_completed=False,
                attempt=attempt,
                max_attempts=max_attempts,
                counts={"messages": len(sanitized_messages)},
            )
            pieces: list[str] = []
            chunks = 0
            try:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=sanitized_messages,
                    temperature=temperature,
                    stream=True,
                    extra_body=body or None,
                )
                for chunk in stream:
                    chunks += 1
                    piece = extract_stream_text(chunk)
                    if piece:
                        pieces.append(piece)
                text = "".join(pieces)
                self.logger.info(
                    phase,
                    "llm_stream_complete",
                    model=self.model,
                    stream_completed=True,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    counts={"chunks": chunks, "characters": len(text)},
                )
                return StreamResult(text=text, stream_completed=True, chunks=chunks)
            except Exception as exc:  # pragma: no cover - provider specific
                last_text = "".join(pieces)
                last_chunks = chunks
                message = sanitize_error_message(exc, self.client)
                retryable = is_retryable_llm_error(exc, message)
                if retryable and attempt < max_attempts:
                    delay = retry_delay_seconds(attempt, self.retry_initial_delay, self.retry_max_delay)
                    self.logger.warning(
                        phase,
                        "llm_stream_retry",
                        model=self.model,
                        stream_completed=False,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        retry_delay_seconds=delay,
                        error=message,
                        endpoint=mask_secret(str(getattr(self.client, "base_url", ""))),
                        counts={"chunks": chunks, "characters": len(last_text)},
                    )
                    if delay > 0:
                        self.sleep_fn(delay)
                    continue
                self.logger.error(
                    phase,
                    "llm_stream_error",
                    model=self.model,
                    stream_completed=False,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    error=message,
                    endpoint=mask_secret(str(getattr(self.client, "base_url", ""))),
                    counts={"chunks": chunks, "characters": len(last_text)},
                )
                return StreamResult(text=last_text, stream_completed=False, chunks=last_chunks, error=message)

        return StreamResult(text=last_text, stream_completed=False, chunks=last_chunks, error="LLM stream failed")


class StaticStreamingClient:
    def __init__(self, responses: Iterable[str | StreamResult]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self._lock = Lock()

    def stream_chat(self, messages: list[dict[str, Any]], *, phase: str, **kwargs: Any) -> StreamResult:
        with self._lock:
            self.calls.append({"messages": messages, "phase": phase, **kwargs})
            if not self._responses:
                return StreamResult(text="", stream_completed=False, error="no static responses left")
            response = self._responses.pop(0)
        if isinstance(response, StreamResult):
            return response
        return StreamResult(text=response, stream_completed=True, chunks=1)


def extract_stream_text(chunk: Any) -> str:
    if isinstance(chunk, dict):
        choices = chunk.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            return delta.get("content") or ""
        return ""

    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return ""
    choice = choices[0]
    delta = getattr(choice, "delta", None)
    if delta is None and isinstance(choice, dict):
        delta = choice.get("delta")
    if isinstance(delta, dict):
        return delta.get("content") or ""
    return getattr(delta, "content", None) or ""


def sanitize_error_message(exc: Exception, client: Any) -> str:
    message = str(exc)
    api_key = str(getattr(client, "api_key", "") or "")
    if api_key:
        message = message.replace(api_key, "***")
    return message


def is_retryable_llm_error(exc: Exception, message: str) -> bool:
    status_code = extract_status_code(exc)
    lowered = message.lower()
    if status_code in {408, 409, 425, 429}:
        return True
    if status_code is not None and status_code >= 500:
        return True
    if status_code == 400 and "could not parse the json body" in lowered:
        return True

    retryable_fragments = (
        "could not parse the json body",
        "connection",
        "temporarily unavailable",
        "timeout",
        "timed out",
        "rate limit",
        "too many requests",
        "server error",
        "service unavailable",
        "overloaded",
        "try again",
    )
    return any(fragment in lowered for fragment in retryable_fragments)


def extract_status_code(exc: Exception) -> int | None:
    for value in (
        getattr(exc, "status_code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    ):
        if isinstance(value, int):
            return value
    return None


def retry_delay_seconds(attempt: int, initial_delay: float, max_delay: float) -> float:
    if initial_delay <= 0 or max_delay <= 0:
        return 0.0
    return min(max_delay, initial_delay * (2 ** max(0, attempt - 1)))


def sanitize_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_json_text(value)
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_json_value(item) for key, item in value.items()}
    return value


def sanitize_json_text(text: str) -> str:
    text = SURROGATE_RE.sub("", text)
    return CONTROL_CHARS_RE.sub("", text)
