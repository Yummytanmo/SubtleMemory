from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from infra.config import redact_value


LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
}

CONSOLE_KEYS = (
    "model",
    "stream_completed",
    "persona_id",
    "topic_preference",
    "case_id",
    "fact_id",
    "attempt",
    "max_attempts",
    "retry_delay_seconds",
    "counts",
    "pass_rate",
    "output_dir",
    "case_concurrency",
    "filter_concurrency",
    "conversation_concurrency",
    "qa_concurrency",
    "llm_max_retries",
    "llm_retry_initial_delay",
    "llm_retry_max_delay",
    "error",
)

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
COLORS = {
    "DEBUG": "\033[2m",
    "INFO": "\033[36m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
}
STATUS_COLORS = {
    "ok": "\033[32m",
    "completed": "\033[32m",
    "running": "\033[36m",
    "warning": "\033[33m",
    "error": "\033[31m",
    "failed": "\033[31m",
    "interrupted": "\033[33m",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class JsonlLogger:
    def __init__(
        self,
        run_id: str,
        *,
        level: str = "INFO",
        log_file: str | Path | None = None,
        secrets: list[str] | None = None,
        stream: TextIO | None = None,
        console_format: str = "text",
        color: bool | None = None,
    ) -> None:
        self.run_id = run_id
        self.level = level.upper()
        self.secrets = secrets or []
        self.stream = stream or sys.stdout
        self.console_format = console_format
        self.color = should_use_color(self.stream) if color is None else color
        self._file_handle: TextIO | None = None
        if log_file:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = path.open("a", encoding="utf-8")

    def close(self) -> None:
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def emit(self, phase: str, event: str, status: str, level: str = "INFO", **fields: Any) -> dict[str, Any]:
        record = {
            "run_id": self.run_id,
            "timestamp": utc_now_iso(),
            "phase": phase,
            "level": level.upper(),
            "event": event,
            "status": status,
        }
        record.update(fields)
        record = redact_value(record, self.secrets)
        if LEVELS.get(record["level"], 20) < LEVELS.get(self.level, 20):
            return record
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        console_line = line if self.console_format == "json" else format_console_record(record, color=self.color)
        print(console_line, file=self.stream, flush=True)
        if self._file_handle:
            print(line, file=self._file_handle, flush=True)
        return record

    def debug(self, phase: str, event: str, status: str = "ok", **fields: Any) -> dict[str, Any]:
        return self.emit(phase, event, status, "DEBUG", **fields)

    def info(self, phase: str, event: str, status: str = "ok", **fields: Any) -> dict[str, Any]:
        return self.emit(phase, event, status, "INFO", **fields)

    def warning(self, phase: str, event: str, status: str = "warning", **fields: Any) -> dict[str, Any]:
        return self.emit(phase, event, status, "WARNING", **fields)

    def error(self, phase: str, event: str, status: str = "error", **fields: Any) -> dict[str, Any]:
        return self.emit(phase, event, status, "ERROR", **fields)


class NullLogger:
    run_id = "null"

    def info(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    def warning(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    def error(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    def debug(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        return None


def should_use_color(stream: TextIO) -> bool:
    if os.getenv("NO_COLOR"):
        return False
    if os.getenv("FORCE_COLOR"):
        return True
    if os.getenv("TERM") == "dumb":
        return False
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


def colorize(text: str, code: str | None, enabled: bool) -> str:
    if not enabled or not code:
        return text
    return f"{code}{text}{RESET}"


def format_console_record(record: dict[str, Any], *, color: bool = False) -> str:
    level = str(record.get("level", "INFO"))
    status = str(record.get("status", "ok"))
    phase = str(record.get("phase", "unknown"))
    event = str(record.get("event", "event"))
    parts = [
        colorize(str(record.get("timestamp", "")), DIM, color),
        colorize(level.ljust(7), COLORS.get(level), color),
        colorize(f"{phase}.{event}", BOLD, color),
        f"status={colorize(format_console_value(status), STATUS_COLORS.get(status), color)}",
    ]
    config = record.get("config")
    if isinstance(config, dict):
        for key in (
            "generation_model",
            "filter_model",
            "generation_reasoning_effort",
            "generation_base_url_configured",
            "generation_api_key_configured",
            "filter_base_url_configured",
            "filter_api_key_configured",
        ):
            if key in config:
                parts.append(f"{key}={format_console_value(config[key])}")

    for key in CONSOLE_KEYS:
        if key in record:
            parts.append(f"{key}={format_console_value(record[key])}")
    return " ".join(part for part in parts if part)


def format_console_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if not text or any(char.isspace() for char in text):
        return json.dumps(text, ensure_ascii=False)
    return text
