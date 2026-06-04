from __future__ import annotations

import inspect
from typing import Any

from openai import OpenAI


def _apply_reasoning_effort(
    kwargs: dict[str, Any],
    create_method: Any,
    reasoning_effort: str | None,
) -> None:
    if not reasoning_effort:
        return

    try:
        signature = inspect.signature(create_method)
    except (TypeError, ValueError):
        signature = None

    if signature and "reasoning_effort" in signature.parameters:
        kwargs["reasoning_effort"] = reasoning_effort
        return

    extra_body = dict(kwargs.get("extra_body") or {})
    extra_body["reasoning_effort"] = reasoning_effort
    kwargs["extra_body"] = extra_body


class LLMClient:
    def __init__(self, config: dict[str, Any]):
        api_cfg = config["api"]
        model_cfg = config["model"]
        self.client = OpenAI(
            api_key=api_cfg["api_key"],
            base_url=api_cfg.get("base_url"),
            timeout=api_cfg.get("timeout_seconds", 180),
        )
        self.model_name = model_cfg["name"]
        self.temperature = model_cfg.get("temperature")
        self.reasoning_effort = model_cfg.get("reasoning_effort")
        self.max_tokens = model_cfg.get("max_tokens", 3000)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
        }

        if temperature is None:
            temperature = self.temperature
        if temperature is not None:
            kwargs["temperature"] = temperature

        if reasoning_effort is None:
            reasoning_effort = self.reasoning_effort
        create_method = self.client.chat.completions.create
        _apply_reasoning_effort(kwargs, create_method, reasoning_effort)

        try:
            response = create_method(**kwargs)
        except Exception as exc:
            message = str(exc)
            if "reasoning_effort" in message:
                kwargs.pop("reasoning_effort", None)
                extra_body = dict(kwargs.get("extra_body") or {})
                if "reasoning_effort" in extra_body:
                    extra_body.pop("reasoning_effort", None)
                    if extra_body:
                        kwargs["extra_body"] = extra_body
                    else:
                        kwargs.pop("extra_body", None)
                response = create_method(**kwargs)
            elif "temperature" in message:
                kwargs.pop("temperature", None)
                response = create_method(**kwargs)
            elif "maxOutputTokens" in message or "max_tokens" in message:
                kwargs["max_tokens"] = min(int(kwargs.get("max_tokens") or self.max_tokens), 65536)
                response = create_method(**kwargs)
            else:
                raise

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Model returned empty content.")
        return content
