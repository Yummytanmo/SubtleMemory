"""Minimal OpenAI-compatible LLM provider used by standalone evaluation."""

from __future__ import annotations

import os
from typing import Any

from openai import AsyncOpenAI


class LLMProvider:
    """Thin async wrapper compatible with the original evaluation call sites."""

    def __init__(
        self,
        provider_type: str = "openai",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 16384,
    ):
        self.provider_type = provider_type
        self.model = (
            model
            or os.environ.get("ANSWER_LLM_MODEL")
            or os.environ.get("LLM_MODEL")
            or "gpt-4o-mini"
        )
        self.api_key = (
            api_key
            or os.environ.get("ANSWER_LLM_API_KEY")
            or os.environ.get("LLM_API_KEY")
            or ""
        )
        self.base_url = (
            base_url
            or os.environ.get("ANSWER_LLM_BASE_URL")
            or os.environ.get("LLM_BASE_URL")
            or None
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def generate(
        self, prompt: str, temperature: float | None = None, **kwargs: Any
    ) -> str:
        """Generate text from a single prompt using an OpenAI-compatible chat API."""

        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=int(kwargs.get("max_tokens", self.max_tokens)),
        )
        content = response.choices[0].message.content
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                text_value = getattr(part, "text", None)
                if text_value:
                    text_parts.append(str(text_value))
                elif isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
                else:
                    text_parts.append(str(part))
            return "".join(text_parts)
        return str(content or "")
