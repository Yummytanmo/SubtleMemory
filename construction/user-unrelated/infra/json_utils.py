from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

try:
    from json_repair import repair_json
except ImportError:  # pragma: no cover
    repair_json = None


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(path: str | Path) -> dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a JSON object.")
    return data


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: str | Path, data: Any, *, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def read_jsonl_ids(path: str | Path, key: str = "sample_id") -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()

    sample_ids: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = record.get(key)
            if isinstance(value, str):
                sample_ids.add(value)
    return sample_ids


def extract_json_from_response(text: str) -> Any:
    candidate = text.strip()
    fenced_match = re.search(r"```json\s*(.*?)\s*```", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        candidate = fenced_match.group(1).strip()
    else:
        generic_fence = re.search(r"```(?:\w+)?\s*(.*?)\s*```", candidate, flags=re.DOTALL)
        if generic_fence:
            candidate = generic_fence.group(1).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        if repair_json is not None:
            repaired = repair_json(candidate)
            return json.loads(repaired)

    start_positions = [idx for idx in (candidate.find("{"), candidate.find("[")) if idx != -1]
    if not start_positions:
        raise ValueError("Could not find JSON object or array in model response.")

    start = min(start_positions)
    tail = candidate[start:]
    for end in range(len(tail), 0, -1):
        snippet = tail[:end]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            continue

    raise ValueError("Unable to parse JSON from model response.")


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def stable_answer_text(answer: Any) -> str:
    if isinstance(answer, (dict, list)):
        return json.dumps(answer, ensure_ascii=False, sort_keys=True)
    return str(answer)


def make_seeded_rng(seed: int, key: str):
    import random

    return random.Random(f"{seed}:{key}")
