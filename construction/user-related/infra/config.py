from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


GENERATION_MODEL = "gpt-5.4"
FILTER_MODEL = "gemini-3.1-pro-preview-thinking"
GENERATION_REASONING_EFFORT = "medium"
DEFAULT_ENV_PATH = Path(".env")
if DEFAULT_ENV_PATH.exists():
    load_dotenv(DEFAULT_ENV_PATH, override=False)
DEFAULT_INPUT_DIR = Path(os.getenv("DATA_CONSTRUCTION_PERSONAMEM_INPUT_DIR", "data/personamem-raw"))
DEFAULT_OUTPUT_DIR = Path(os.getenv("DATA_CONSTRUCTION_USER_RELATED_OUTPUT_DIR", "data/user-related"))


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class AppConfig:
    generation_base_url: str | None
    generation_api_key: str | None
    filter_base_url: str | None
    filter_api_key: str | None
    generation_model: str = GENERATION_MODEL
    filter_model: str = FILTER_MODEL
    generation_reasoning_effort: str = GENERATION_REASONING_EFFORT

    @property
    def secrets(self) -> list[str]:
        return [
            value
            for value in [
                self.generation_api_key,
                self.filter_api_key,
                self.generation_base_url,
                self.filter_base_url,
            ]
            if value
        ]


def load_config(env_path: str | Path = ".env", require_secrets: bool = True) -> AppConfig:
    env_file = Path(env_path)
    if env_file.exists():
        load_dotenv(env_file, override=False)

    config = AppConfig(
        generation_base_url=os.getenv("GENERATION_BASE_URL"),
        generation_api_key=os.getenv("GENERATION_API_KEY"),
        filter_base_url=os.getenv("FILTER_BASE_URL"),
        filter_api_key=os.getenv("FILTER_API_KEY"),
    )
    if require_secrets:
        missing = [
            name
            for name, value in [
                ("GENERATION_BASE_URL", config.generation_base_url),
                ("GENERATION_API_KEY", config.generation_api_key),
                ("FILTER_BASE_URL", config.filter_base_url),
                ("FILTER_API_KEY", config.filter_api_key),
            ]
            if not value
        ]
        if missing:
            raise ConfigError(f"Missing required .env values: {', '.join(missing)}")
    return config


def mask_secret(value: str | None) -> str | None:
    if value is None:
        return None
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def redact_value(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, str):
        redacted = value
        for secret in secrets:
            if secret:
                redacted = redacted.replace(secret, mask_secret(secret) or "***")
        return redacted
    if isinstance(value, dict):
        return {key: redact_value(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_value(item, secrets) for item in value]
    return value


def masked_config_summary(config: AppConfig) -> dict[str, Any]:
    return {
        "generation_model": config.generation_model,
        "filter_model": config.filter_model,
        "generation_reasoning_effort": config.generation_reasoning_effort,
        "generation_base_url_configured": bool(config.generation_base_url),
        "generation_api_key_configured": bool(config.generation_api_key),
        "filter_base_url_configured": bool(config.filter_base_url),
        "filter_api_key_configured": bool(config.filter_api_key),
    }
