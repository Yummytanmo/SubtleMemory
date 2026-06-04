from __future__ import annotations

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
_DATA_ROOT = _REPO_ROOT / "data" / "user-unrelated"
_OUTPUTS_ROOT = _DATA_ROOT / "outputs"
_OUTPUTS_TO_FILTER_ROOT = _DATA_ROOT / "outputs_to_filter"
_OUTPUTS_SELECTED_ROOT = _DATA_ROOT / "outputs_selected"


def repo_root() -> Path:
    return _REPO_ROOT


def workflow_root() -> Path:
    return _WORKFLOW_ROOT


def data_root() -> Path:
    return _DATA_ROOT


def workflow_name() -> str:
    return "user-unrelated"


def config_root() -> Path:
    return data_root() / "config"


def source_data_root() -> Path:
    return data_root() / "source_data"


def archives_root() -> Path:
    return data_root() / "archives"


def outputs_root() -> Path:
    return _OUTPUTS_ROOT


def outputs_to_filter_root() -> Path:
    return _OUTPUTS_TO_FILTER_ROOT


def outputs_selected_root() -> Path:
    return _OUTPUTS_SELECTED_ROOT


def runtime_roots() -> tuple[Path, ...]:
    return (
        outputs_root(),
        outputs_to_filter_root(),
        outputs_selected_root(),
    )


def ensure_runtime_layout() -> tuple[Path, ...]:
    created: list[Path] = []
    for root in runtime_roots():
        root.mkdir(parents=True, exist_ok=True)
        created.append(root)
    return tuple(created)


def relative_to_repo(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(repo_root()))
    except ValueError:
        return str(resolved)


def display_path(path: str | Path) -> str:
    return relative_to_repo(path)
