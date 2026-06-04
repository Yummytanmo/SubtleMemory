"""
SubtleMemory converter - convert case-based persona bundles to LoCoMo format.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from evaluation.src.converters.base import BaseConverter
from evaluation.src.converters.registry import register_converter


REQUIRED_QA_METADATA_FIELDS = (
    "correct_answers",
    "incorrect_answers",
    "case",
    "case_id",
    "instance_id",
    "session_ids",
    "facts",
    "relation_type",
    "relation_subtype",
    "topic",
    "persona_str",
    "persona_id",
    "source",
)


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _parse_iso_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None

    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _format_locomo_timestamp(value: Any) -> str:
    parsed = _parse_iso_timestamp(value)
    if parsed is None:
        return "Unknown"

    hour = parsed.strftime("%I").lstrip("0") or "0"
    minute = parsed.strftime("%M")
    meridiem = parsed.strftime("%p").lower()
    month = parsed.strftime("%B")
    return f"{hour}:{minute} {meridiem} on {parsed.day} {month}, {parsed.year}"


def _normalize_role(role: Any) -> str:
    return "user" if str(role or "").strip().lower() == "user" else "assistant"


def _sanitize_identifier_component(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"

    sanitized_chars = []
    previous_was_separator = False
    for char in text:
        if char.isalnum():
            sanitized_chars.append(char.lower())
            previous_was_separator = False
            continue
        if not previous_was_separator:
            sanitized_chars.append("_")
            previous_was_separator = True

    sanitized = "".join(sanitized_chars).strip("_")
    return sanitized or "unknown"


def _bundle_key(bundle_dir: Path, dataset_root: Path) -> str:
    resolved_bundle = bundle_dir.resolve()
    resolved_root = dataset_root.resolve()
    relative_path = resolved_bundle.relative_to(resolved_root)
    if relative_path == Path("."):
        return "root"
    return "__".join(
        _sanitize_identifier_component(part) for part in relative_path.parts
    )


def _normalize_sort_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _discover_bundle_dirs(dataset_root: Path) -> List[Path]:
    candidates = []

    if (dataset_root / "bench_instances.json").is_file() and (
        dataset_root / "history_sessions.json"
    ).is_file():
        candidates.append(dataset_root)

    for path in dataset_root.rglob("*"):
        if not path.is_dir():
            continue
        if (path / "bench_instances.json").is_file() and (
            path / "history_sessions.json"
        ).is_file():
            candidates.append(path)

    unique_paths = {path.resolve() for path in candidates}
    return sorted(unique_paths, key=lambda path: str(path))


def _session_sort_key(session: Dict[str, Any]) -> tuple[int, Any, str]:
    order = session.get("order")
    if order is not None:
        try:
            return (0, int(order), str(session.get("session_id", "")))
        except (TypeError, ValueError):
            pass

    parsed_timestamp = _parse_iso_timestamp(session.get("timestamp"))
    if parsed_timestamp is not None:
        return (
            1,
            _normalize_sort_timestamp(parsed_timestamp),
            str(session.get("session_id", "")),
        )

    return (2, str(session.get("session_id", "")), str(session.get("case_id", "")))


def _iter_clean_strings(values: Iterable[Any]) -> List[str]:
    cleaned: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text:
            cleaned.append(text)
    return cleaned


def _qa_rows_have_required_schema(rows: List[Dict[str, Any]]) -> bool:
    for row in rows:
        if not isinstance(row, dict):
            return False
        conversation = row.get("conversation")
        if not isinstance(conversation, dict):
            return False
        for value in conversation.values():
            if not isinstance(value, list):
                continue
            for message in value:
                if not isinstance(message, dict):
                    return False
                if "session_source" not in message or "session_timestamp" not in message:
                    return False
        qa_rows = row.get("qa")
        if not isinstance(qa_rows, list):
            return False
        for qa in qa_rows:
            if not isinstance(qa, dict):
                return False
            for key in REQUIRED_QA_METADATA_FIELDS:
                if key not in qa:
                    return False
    return True


def _build_locomo_entry(
    bundle_dir: Path,
    bundle_key: str,
    sessions: List[Dict[str, Any]],
    bench_instances: List[Dict[str, Any]],
) -> Dict[str, Any]:
    conversation: Dict[str, Any] = {"speaker_a": "user", "speaker_b": "assistant"}

    for session_idx, session in enumerate(sessions):
        conversation[f"session_{session_idx}_date_time"] = _format_locomo_timestamp(
            session.get("timestamp")
        )
        conversation[f"session_{session_idx}"] = []

        for msg_idx, message in enumerate(session.get("history", []) or []):
            conversation[f"session_{session_idx}"].append(
                {
                    "speaker": _normalize_role(message.get("role")),
                    "text": str(message.get("content") or "").strip(),
                    "dia_id": f"D{session_idx}:{msg_idx}",
                    "session_id": str(session.get("session_id") or "").strip(),
                    "case_id": str(session.get("case_id") or "").strip(),
                    "session_order": session.get("order"),
                    "session_source": str(session.get("source") or "").strip(),
                    "session_timestamp": str(session.get("timestamp") or "").strip(),
                }
            )

    qa_entries = []
    case_ids: List[str] = []
    instance_ids: List[str] = []
    persona_id = None

    for instance in bench_instances:
        case_id = str(instance.get("case_id") or "").strip()
        instance_id = str(instance.get("instance_id") or "").strip()
        if case_id:
            case_ids.append(case_id)
        if instance_id:
            instance_ids.append(instance_id)
        if persona_id is None and instance.get("persona_id") is not None:
            persona_id = instance.get("persona_id")

        question_prefix = "__".join(
            [
                bundle_key,
                _sanitize_identifier_component(case_id or "casebench"),
                _sanitize_identifier_component(instance_id or case_id or "casebench"),
            ]
        )
        relation_type = str(instance.get("relation_type") or "")
        session_ids = _iter_clean_strings(instance.get("session_ids", []) or [])

        for qa_idx, qa in enumerate(instance.get("qas", []) or []):
            correct_answers = _iter_clean_strings(qa.get("correct_answers", []) or [])
            incorrect_answers = _iter_clean_strings(
                qa.get("incorrect_answers", []) or []
            )
            qa_entries.append(
                {
                    "question_id": f"{question_prefix}_qa{qa_idx}",
                    "question": str(qa.get("query") or "").strip(),
                    "answer": correct_answers[0] if correct_answers else "",
                    "category": relation_type,
                    "evidence": [],
                    "correct_answers": correct_answers,
                    "incorrect_answers": incorrect_answers,
                    "case": str(instance.get("case") or "").strip(),
                    "case_id": case_id,
                    "instance_id": instance_id,
                    "session_ids": session_ids,
                    "facts": _iter_clean_strings(instance.get("facts", []) or []),
                    "relation_type": relation_type,
                    "relation_subtype": str(
                        instance.get("relation_subtype") or ""
                    ).strip(),
                    "topic": str(instance.get("topic") or "").strip(),
                    "persona_str": str(instance.get("persona_str") or "").strip(),
                    "persona_id": (
                        ""
                        if instance.get("persona_id") is None
                        else str(instance.get("persona_id"))
                    ),
                    "source": str(instance.get("source") or "").strip(),
                }
            )

    return {
        "persona_id": persona_id,
        "case_ids": case_ids,
        "instance_ids": instance_ids,
        "bundle_case_count": len(bench_instances),
        "source_bundle_dir": str(bundle_dir),
        "qa": qa_entries,
        "conversation": conversation,
    }


@register_converter("subtlememory")
class SubtleMemoryConverter(BaseConverter):
    """Convert recursive persona casebench bundles into one LoCoMo-style file."""

    def needs_conversion(self, data_dir: Path) -> bool:
        dataset_root = data_dir.resolve()
        output_file = self.get_converted_path(data_dir)
        bundle_dirs = _discover_bundle_dirs(dataset_root)

        if not output_file.exists() or not bundle_dirs:
            return True

        current_bundle_paths = {str(path.resolve()) for path in bundle_dirs}
        try:
            with open(output_file, "r", encoding="utf-8") as file:
                converted_rows = json.load(file)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return True

        converted_bundle_paths = {
            str(row.get("source_bundle_dir", "")).strip()
            for row in converted_rows
            if isinstance(row, dict) and str(row.get("source_bundle_dir", "")).strip()
        }
        if not _qa_rows_have_required_schema(converted_rows):
            return True
        if len(converted_rows) != len(bundle_dirs):
            return True
        if converted_bundle_paths != current_bundle_paths:
            return True

        output_mtime = output_file.stat().st_mtime_ns
        for bundle_dir in bundle_dirs:
            for filename in ("bench_instances.json", "history_sessions.json"):
                source_path = bundle_dir / filename
                if source_path.stat().st_mtime_ns > output_mtime:
                    return True

        return False

    def get_input_files(self) -> Dict[str, str]:
        return {"root": "."}

    def get_output_filename(self) -> str:
        return "subtlememory_locomo_style.json"

    def convert(self, input_paths: Dict[str, str], output_path: str) -> None:
        dataset_root = Path(input_paths["root"]).resolve()
        bundle_dirs = _discover_bundle_dirs(dataset_root)

        if not bundle_dirs:
            raise FileNotFoundError(
                f"No sample bundles found under {dataset_root}. Expected directories containing both bench_instances.json and history_sessions.json."
            )

        print("🔄 Converting SubtleMemory to LoCoMo format...")
        print(f"   Dataset root: {dataset_root}")
        print(f"   Discovered bundle directories: {len(bundle_dirs)}")

        locomo_entries: List[Dict[str, Any]] = []

        for bundle_dir in bundle_dirs:
            bench_instances = _load_json(bundle_dir / "bench_instances.json")
            history_sessions = _load_json(bundle_dir / "history_sessions.json")
            bundle_path_key = _bundle_key(bundle_dir, dataset_root)

            ordered_sessions = sorted(history_sessions, key=_session_sort_key)
            if not ordered_sessions:
                print(f"   ⚠️  Skipping bundle without sessions: {bundle_dir}")
                continue

            locomo_entries.append(
                _build_locomo_entry(
                    bundle_dir=bundle_dir,
                    bundle_key=bundle_path_key,
                    sessions=ordered_sessions,
                    bench_instances=bench_instances,
                )
            )

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(locomo_entries, file, indent=2, ensure_ascii=False)

        print(f"   ✅ Saved {len(locomo_entries)} entries to {output_path}")
        print(
            f"   Total questions: {sum(len(entry.get('qa', [])) for entry in locomo_entries)}"
        )
