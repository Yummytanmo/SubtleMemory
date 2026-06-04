from __future__ import annotations

from typing import Any, Dict


async def clear_group_data_in_context(
    group_id: str, verbose: bool = True
) -> Dict[str, Any]:
    del group_id, verbose
    raise RuntimeError(
        "clean_groups is not supported in the standalone evaluation package. "
        "It requires SubtleMemory application storage dependencies that are excluded "
        "with the local EverMemOS adapter."
    )


async def clear_group_data(group_id: str, verbose: bool = True) -> Dict[str, Any]:
    return await clear_group_data_in_context(group_id=group_id, verbose=verbose)
