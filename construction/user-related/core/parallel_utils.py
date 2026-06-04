from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Sequence, TypeVar, cast


T = TypeVar("T")
R = TypeVar("R")
_MISSING = object()


def map_ordered(items: Sequence[T], worker: Callable[[T], R], max_workers: int = 1) -> list[R]:
    if max_workers <= 1 or len(items) <= 1:
        return [worker(item) for item in items]

    results: list[R | object] = [_MISSING] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, item): index for index, item in enumerate(items)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return [cast(R, item) for item in results if item is not _MISSING]


def normalize_concurrency(value: int | None) -> int:
    if value is None:
        return 1
    if value < 1:
        raise ValueError("concurrency must be >= 1")
    return value
