"""Standalone environment loading helpers.

This keeps the copied evaluation project compatible with the original
``common_utils.load_env.setup_environment`` call site while removing the hard
dependency on the parent SubtleMemory application bootstrap.
"""

from __future__ import annotations

from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency fallback
    load_dotenv = None


def setup_environment(
    load_env_file_name: str = ".env",
    check_env_var: str | None = None,
) -> None:
    """Best-effort standalone env loading.

    The original project required specific SubtleMemory infrastructure variables.
    Standalone evaluation only needs API keys when the chosen adapter/stage uses
    them, so this helper intentionally does not enforce ``check_env_var``.
    """

    del check_env_var

    if load_dotenv is None:
        return

    project_root = Path(__file__).resolve().parent.parent
    candidate_paths = [
        project_root / load_env_file_name,
        Path.cwd() / load_env_file_name,
    ]

    seen: set[Path] = set()
    for candidate in candidate_paths:
        resolved = candidate.resolve()
        if resolved in seen or not candidate.exists():
            continue
        load_dotenv(candidate, override=False)
        seen.add(resolved)
