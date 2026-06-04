from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv


RELATED_ROOT = Path(__file__).resolve().parents[1] / "user-related"
if str(RELATED_ROOT) not in sys.path:
    sys.path.insert(0, str(RELATED_ROOT))

from infra.llm_client import OpenAIStreamingClient  # noqa: E402
from ingestion.topic_builder import clean_topic, extract_topic_after_output  # noqa: E402
from prompts import build_categorize_preference_topic_prompt  # noqa: E402


DEFAULT_INPUT = Path("data/user-unrelated/runs/remaining-for-50/outputs_selected/all_passed_samples.json")
DEFAULT_OUTPUT_NAME = "unrelated_topic_classification.json"
DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_WORKERS = 1
DEFAULT_MAX_FACTS = 12
DEFAULT_MAX_QAS = 3


class TopicClassificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class TopicConfig:
    base_url: str
    api_key: str
    model: str
    reasoning_effort: str | None = None
    source_prefix: str = ""


@dataclass(frozen=True)
class TopicAssignment:
    item_id: str
    sample_id: str
    category: str
    source_dataset: str
    topic: str
    raw_response: str
    input_text: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "item_id": self.item_id,
            "sample_id": self.sample_id,
            "category": self.category,
            "source_dataset": self.source_dataset,
            "topic": self.topic,
            "raw_response": self.raw_response,
        }
        if self.input_text is not None:
            payload["input_text"] = self.input_text
        return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify user-unrelated cases into related-style topic labels by reusing "
            "construction/user-related/prompts.py::build_categorize_preference_topic_prompt."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--merged-dir",
        type=Path,
        default=None,
        help=(
            "Classify source=user-unrelated cases from a merged dataset root. "
            "Initial existing topics are taken from source=user-related cases in the same merged dataset."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Defaults to INPUT.parent / {DEFAULT_OUTPUT_NAME}, or MERGED_DIR / {DEFAULT_OUTPUT_NAME}.",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--base-url", default=None, help="Override TOPIC_BASE_URL / DOMAIN_BASE_URL.")
    parser.add_argument("--api-key", default=None, help="Override TOPIC_API_KEY / DOMAIN_API_KEY.")
    parser.add_argument("--model", default=None, help="Override TOPIC_MODEL / DOMAIN_MODEL.")
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--max-retries", type=non_negative_int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--max-workers", type=positive_int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--limit", type=positive_int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-facts", type=positive_int, default=DEFAULT_MAX_FACTS)
    parser.add_argument("--max-qas", type=positive_int, default=DEFAULT_MAX_QAS)
    parser.add_argument(
        "--save-input-text",
        action="store_true",
        help="Store the exact compact text passed into the related topic prompt for each case.",
    )
    parser.add_argument(
        "--update-merge-report",
        action="store_true",
        help=(
            "Only valid with --merged-dir. Replace merge_report.json's user-unrelated topic counts "
            "with the newly classified labels and recompute combined topic counts."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_topic_config(
            args.env_file,
            base_url_override=args.base_url,
            api_key_override=args.api_key,
            model_override=args.model,
            reasoning_effort_override=args.reasoning_effort,
        )
        client = OpenAIStreamingClient(
            base_url=config.base_url,
            api_key=config.api_key,
            model=config.model,
            reasoning_effort=config.reasoning_effort,
            max_retries=args.max_retries,
        )
        if args.merged_dir is not None:
            output_path = args.output or args.merged_dir / DEFAULT_OUTPUT_NAME
            stats = classify_merged_unrelated_topics(
                merged_dir=args.merged_dir,
                output_path=output_path,
                client=client,
                model=config.model,
                config_source=config.source_prefix,
                resume=args.resume,
                limit=args.limit,
                max_workers=args.max_workers,
                max_facts=args.max_facts,
                max_qas=args.max_qas,
                save_input_text=args.save_input_text,
                update_merge_report=args.update_merge_report,
            )
        else:
            if args.update_merge_report:
                raise TopicClassificationError("--update-merge-report requires --merged-dir")
            output_path = args.output or args.input.parent / DEFAULT_OUTPUT_NAME
            stats = classify_unrelated_topics(
                input_path=args.input,
                output_path=output_path,
                client=client,
                model=config.model,
                config_source=config.source_prefix,
                resume=args.resume,
                limit=args.limit,
                max_workers=args.max_workers,
                max_facts=args.max_facts,
                max_qas=args.max_qas,
                save_input_text=args.save_input_text,
            )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "output_path": stats["metadata"]["output_path"],
                "input_path": stats["metadata"]["input_path"],
                "model": stats["metadata"]["model"],
                "status": stats["metadata"]["status"],
                "completed_cases": stats["metadata"]["completed_cases"],
                "expected_total_cases": stats["metadata"]["expected_total_cases"],
                "topic_counts": stats["statistics"]["topic_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def classify_unrelated_topics(
    *,
    input_path: Path,
    output_path: Path,
    client: OpenAIStreamingClient,
    model: str,
    config_source: str,
    resume: bool,
    limit: int | None,
    max_workers: int,
    max_facts: int,
    max_qas: int,
    save_input_text: bool,
) -> dict[str, Any]:
    items = load_items(input_path)
    return classify_topic_items(
        input_path=input_path,
        output_path=output_path,
        items=items,
        initial_topics=[],
        client=client,
        model=model,
        config_source=config_source,
        resume=resume,
        limit=limit,
        max_workers=max_workers,
        max_facts=max_facts,
        max_qas=max_qas,
        save_input_text=save_input_text,
        extra_metadata={"input_mode": "unrelated_samples"},
    )


def classify_merged_unrelated_topics(
    *,
    merged_dir: Path,
    output_path: Path,
    client: OpenAIStreamingClient,
    model: str,
    config_source: str,
    resume: bool,
    limit: int | None,
    max_workers: int,
    max_facts: int,
    max_qas: int,
    save_input_text: bool,
    update_merge_report: bool,
) -> dict[str, Any]:
    records = load_merged_bench_records(merged_dir)
    initial_topics = related_topics_from_merged_records(records)
    unrelated_items = [item for item in records if item.get("source") == "user-unrelated"]
    stats = classify_topic_items(
        input_path=merged_dir,
        output_path=output_path,
        items=unrelated_items,
        initial_topics=initial_topics,
        client=client,
        model=model,
        config_source=config_source,
        resume=resume,
        limit=limit,
        max_workers=max_workers,
        max_facts=max_facts,
        max_qas=max_qas,
        save_input_text=save_input_text,
        extra_metadata={
            "input_mode": "merged_user_unrelated",
            "merged_dir": str(merged_dir),
            "initial_topics_source": "source=user-related topics from merged bench_instances.json files",
            "initial_topic_count": len(initial_topics),
            "initial_topics": initial_topics,
        },
    )
    if update_merge_report:
        if stats["metadata"]["status"] != "complete":
            raise TopicClassificationError("cannot update merge_report.json from partial topic classification")
        update_merge_report_topics(
            merged_dir=merged_dir,
            classification_output=output_path,
            stats=stats,
            records=records,
        )
    return stats


def classify_topic_items(
    *,
    input_path: Path,
    output_path: Path,
    items: list[dict[str, Any]],
    initial_topics: list[str],
    client: OpenAIStreamingClient,
    model: str,
    config_source: str,
    resume: bool,
    limit: int | None,
    max_workers: int,
    max_facts: int,
    max_qas: int,
    save_input_text: bool,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if limit is not None:
        items = items[:limit]
    expected_total = len(items)
    completed = load_completed_assignments(output_path, model=model) if resume else {}
    assignments_by_index: list[TopicAssignment | None] = [None] * len(items)
    existing_topics: list[str] = list(initial_topics)

    for index, item in enumerate(items):
        item_id = item_identifier(item)
        if item_id in completed:
            assignment = completed[item_id]
            assignments_by_index[index] = assignment
            append_topic(existing_topics, assignment.topic)

    emit_log(
        f"topic_classify.start input={input_path} output={output_path} model={model} "
        f"cases={expected_total} resume={resume} resumed={len(completed)} "
        f"max_workers={max_workers} session_mode=none"
    )
    latest_stats: dict[str, Any] | None = None

    def ordered_assignments() -> list[TopicAssignment]:
        return [assignment for assignment in assignments_by_index if assignment is not None]

    def checkpoint() -> None:
        nonlocal latest_stats
        assignments = ordered_assignments()
        latest_stats = build_stats_payload(
            input_path=input_path,
            output_path=output_path,
            model=model,
            config_source=config_source,
            assignments=assignments,
            expected_total_cases=expected_total,
            resume_enabled=resume,
            max_workers=max_workers,
            session_mode="none",
            max_facts=max_facts,
            max_qas=max_qas,
            max_session_chars=0,
            extra_metadata=extra_metadata,
        )
        write_json(output_path, latest_stats)
        metadata = latest_stats["metadata"]
        emit_log(
            f"topic_classify.progress status={metadata['status']} "
            f"completed={metadata['completed_cases']}/{metadata['expected_total_cases']} "
            f"pending={metadata['pending_cases']}"
        )

    completed_ids = {assignment.item_id for assignment in ordered_assignments()}
    pending_indices = [index for index, item in enumerate(items) if item_identifier(item) not in completed_ids]
    errors: list[str] = []
    try:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(pending_indices)) or 1) as executor:
            next_pending = 0
            futures: dict[Any, tuple[int, str]] = {}

            def submit_next() -> None:
                nonlocal next_pending
                if next_pending >= len(pending_indices):
                    return
                index = pending_indices[next_pending]
                next_pending += 1
                item = items[index]
                preference_text = build_unrelated_preference_text(
                    item,
                    session_mode="none",
                    max_facts=max_facts,
                    max_qas=max_qas,
                    max_session_chars=0,
                )
                future = executor.submit(classify_one_topic, client, preference_text, list(existing_topics))
                futures[future] = (index, preference_text)

            for _ in range(min(max_workers, len(pending_indices))):
                submit_next()

            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    index, preference_text = futures.pop(future)
                    item = items[index]
                    item_id = item_identifier(item)
                    try:
                        topic, raw_response = future.result()
                    except Exception as exc:
                        errors.append(f"{item_id}: {exc}")
                        emit_log(f"topic_classify.case_error item_id={item_id} error={exc}")
                        submit_next()
                        continue

                    topic = matching_existing_topic(topic, existing_topics)
                    append_topic(existing_topics, topic)
                    assignments_by_index[index] = TopicAssignment(
                        item_id=item_id,
                        sample_id=string_value(item.get("sample_id") or item.get("case_id")),
                        category=string_value(
                            item.get("category") or item.get("relationship") or item.get("relation_type")
                        ),
                        source_dataset=string_value(item.get("source_dataset") or item.get("source")),
                        topic=topic,
                        raw_response=raw_response,
                        input_text=preference_text if save_input_text else None,
                    )
                    completed_ids.add(item_id)
                    checkpoint()
                    submit_next()
    except Exception as exc:
        if ordered_assignments():
            checkpoint()
        emit_log(f"topic_classify.error completed={len(ordered_assignments())}/{expected_total} error={exc}")
        raise
    if errors:
        if ordered_assignments():
            checkpoint()
        raise TopicClassificationError(
            "topic classification failed for some cases; saved completed results. " + "; ".join(errors[:10])
        )

    if latest_stats is None or latest_stats["metadata"]["status"] != "complete":
        checkpoint()
    emit_log(f"topic_classify.complete completed={latest_stats['metadata']['completed_cases']}")
    return latest_stats


def classify_one_topic(
    client: OpenAIStreamingClient,
    preference_text: str,
    existing_topics: list[str],
) -> tuple[str, str]:
    prompt = build_categorize_preference_topic_prompt(preference_text, existing_topics)
    result = client.stream_chat(
        [{"role": "user", "content": prompt}],
        phase="unrelated_topic_categorization",
        temperature=0.0,
    )
    if not result.stream_completed:
        raise TopicClassificationError(f"topic categorization stream failed: {result.error}")
    raw_topic = extract_topic_after_output(result.text)
    cleaned = clean_topic(raw_topic)
    return matching_existing_topic(cleaned, existing_topics), result.text


def build_unrelated_preference_text(
    item: dict[str, Any],
    *,
    session_mode: str,
    max_facts: int,
    max_qas: int,
    max_session_chars: int,
) -> str:
    parts: list[str] = []
    add_part(parts, "Case", item.get("case"))

    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    add_part(parts, "Source question", metadata.get("source_question"))
    add_part(parts, "Source answer", metadata.get("source_answer_text") or stable_text(metadata.get("source_answer")))

    facts = extract_fact_texts(item)[:max_facts]
    if facts:
        add_part(parts, "Facts", "; ".join(facts))

    qa_texts = extract_qa_texts(item)[:max_qas]
    if qa_texts:
        add_part(parts, "QA", " | ".join(qa_texts))

    if session_mode != "none":
        session_text = extract_session_text(item, mode=session_mode, max_chars=max_session_chars)
        add_part(parts, "Sessions", session_text)

    return compact_text(" | ".join(parts))


def extract_fact_texts(item: dict[str, Any]) -> list[str]:
    if isinstance(item.get("facts"), list):
        return [compact_text(value) for value in item["facts"] if compact_text(value)]

    texts: list[str] = []
    for fact in item.get("memory_facts") or []:
        if isinstance(fact, dict):
            value = fact.get("fact_statement") or fact.get("fact_text") or fact.get("statement")
            if value:
                texts.append(compact_text(value))
            source_memory = fact.get("source_memory")
            if isinstance(source_memory, dict):
                subquestion = source_memory.get("subquestion")
                subanswer = source_memory.get("subanswer_text") or stable_text(source_memory.get("subanswer"))
                if subquestion:
                    texts.append(compact_text(f"{subquestion} -> {subanswer}"))
        elif fact:
            texts.append(compact_text(fact))
    return dedupe_preserve_order(texts)


def extract_qa_texts(item: dict[str, Any]) -> list[str]:
    raw_qas = item.get("qa_pairs") or item.get("qas") or []
    texts: list[str] = []
    for qa in raw_qas:
        if not isinstance(qa, dict):
            continue
        question = qa.get("question") or qa.get("query")
        answer = first_answer_text(qa.get("correct_answers") or qa.get("answers") or qa.get("answer"))
        if question and answer:
            texts.append(compact_text(f"{question} -> {answer}"))
        elif question:
            texts.append(compact_text(question))
    return texts


def extract_session_text(item: dict[str, Any], *, mode: str, max_chars: int) -> str:
    snippets: list[str] = []
    if mode == "summary":
        for plan in item.get("session_plans") or []:
            if not isinstance(plan, dict):
                continue
            bits = [
                plan.get("scenario_label"),
                plan.get("event_summary"),
                plan.get("user_goal"),
                plan.get("fact_integration_plan"),
            ]
            text = compact_text(" ".join(str(bit) for bit in bits if bit))
            if text:
                snippets.append(text)
    elif mode == "messages":
        for session in item.get("sessions") or []:
            if not isinstance(session, dict):
                continue
            conversation = session.get("conversation") or session.get("messages") or []
            turns: list[str] = []
            for message in conversation:
                if not isinstance(message, dict):
                    continue
                role = message.get("role")
                content = compact_text(message.get("content"))
                if role and content:
                    turns.append(f"{role}: {content}")
            if turns:
                snippets.append(" ".join(turns))

    joined = compact_text(" || ".join(snippets))
    return joined[:max_chars].rstrip()


def first_answer_text(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            return ""
        first = value[0]
        if isinstance(first, dict):
            return compact_text(first.get("text") or stable_text(first))
        return compact_text(first)
    if isinstance(value, dict):
        return compact_text(value.get("text") or stable_text(value))
    return compact_text(value)


def build_stats_payload(
    *,
    input_path: Path,
    output_path: Path,
    model: str,
    config_source: str,
    assignments: list[TopicAssignment],
    expected_total_cases: int,
    resume_enabled: bool,
    session_mode: str,
    max_facts: int,
    max_qas: int,
    max_session_chars: int,
    max_workers: int | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    completed = len(assignments)
    pending = max(expected_total_cases - completed, 0)
    case_topics = [assignment.to_payload() for assignment in assignments]
    metadata = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "config_source": config_source,
        "prompt_source": "construction/user-related/prompts.py::build_categorize_preference_topic_prompt",
        "parser_source": "construction/user-related/ingestion/topic_builder.py::extract_topic_after_output + clean_topic",
        "status": "complete" if pending == 0 else "partial",
        "expected_total_cases": expected_total_cases,
        "completed_cases": completed,
        "pending_cases": pending,
        "resume_enabled": resume_enabled,
        "max_workers": max_workers,
        "session_mode": session_mode,
        "max_facts": max_facts,
        "max_qas": max_qas,
        "max_session_chars": max_session_chars,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return {
        "metadata": metadata,
        "case_topics": case_topics,
        "statistics": {
            "topic_counts": dict(Counter(item.topic for item in assignments).most_common()),
            "topic_counts_by_source_dataset": nested_counts(assignments, "source_dataset"),
            "topic_counts_by_category": nested_counts(assignments, "category"),
        },
    }


def nested_counts(assignments: list[TopicAssignment], field: str) -> dict[str, dict[str, int]]:
    result: dict[str, Counter[str]] = {}
    for assignment in assignments:
        group = getattr(assignment, field) or ""
        result.setdefault(group, Counter())[assignment.topic] += 1
    return {key: dict(counter.most_common()) for key, counter in sorted(result.items())}


def load_merged_bench_records(merged_dir: Path) -> list[dict[str, Any]]:
    if not merged_dir.exists():
        raise FileNotFoundError(f"merged directory does not exist: {merged_dir}")
    records: list[dict[str, Any]] = []
    for bench_path in sorted(merged_dir.glob("persona_*/bench_instances.json")):
        payload = json.loads(bench_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON array in {bench_path}")
        persona_id = bench_path.parent.name.removeprefix("persona_")
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError(f"Expected object at index {index} in {bench_path}")
            copied = dict(item)
            copied.setdefault("persona_id", persona_id)
            copied["_bench_path"] = str(bench_path)
            item_identifier(copied)
            records.append(copied)
    if not records:
        raise FileNotFoundError(f"no persona_*/bench_instances.json files found under {merged_dir}")
    return records


def related_topics_from_merged_records(records: list[dict[str, Any]]) -> list[str]:
    topics: list[str] = []
    for item in records:
        if item.get("source") != "user-related":
            continue
        topic = string_value(item.get("topic"))
        if topic:
            append_topic(topics, topic)
    if not topics:
        raise TopicClassificationError("no source=user-related topics found in merged bench records")
    return topics


def update_merge_report_topics(
    *,
    merged_dir: Path,
    classification_output: Path,
    stats: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    report_path = merged_dir / "merge_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"merge_report.json does not exist: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise TopicClassificationError(f"merge report must be a JSON object: {report_path}")

    assignments_by_id = {
        string_value(item.get("item_id")): string_value(item.get("topic"))
        for item in stats.get("case_topics", [])
        if isinstance(item, dict)
    }
    unrelated_records = [item for item in records if item.get("source") == "user-unrelated"]
    missing = [
        string_value(item.get("instance_id") or item_identifier(item))
        for item in unrelated_records
        if string_value(item.get("instance_id") or item_identifier(item)) not in assignments_by_id
    ]
    if missing:
        raise TopicClassificationError(
            f"classification output is missing {len(missing)} merged unrelated cases; first missing={missing[:5]}"
        )

    related_counts = Counter(
        string_value(item.get("topic"))
        for item in records
        if item.get("source") == "user-related" and string_value(item.get("topic"))
    )
    new_unrelated_counts = Counter(
        assignments_by_id[string_value(item.get("instance_id") or item_identifier(item))]
        for item in unrelated_records
    )
    combined_counts = related_counts + new_unrelated_counts

    overview = report.setdefault("overview", {})
    if not isinstance(overview, dict):
        raise TopicClassificationError("merge report overview must be an object")
    bench = overview.setdefault("bench_instances", {})
    if not isinstance(bench, dict):
        raise TopicClassificationError("merge report overview.bench_instances must be an object")
    by_source = bench.setdefault("topic_counts_by_source", {})
    if not isinstance(by_source, dict):
        raise TopicClassificationError("merge report overview.bench_instances.topic_counts_by_source must be an object")

    previous_user_unrelated = by_source.get("user-unrelated", {})
    bench["topic_counts"] = dict(sorted(combined_counts.items()))
    by_source["user-related"] = dict(sorted(related_counts.items()))
    by_source["user-unrelated"] = dict(sorted(new_unrelated_counts.items()))
    bench["topic_reclassification"] = {
        "source": "user-unrelated",
        "method": "related_prompt_case_only",
        "classification_output": str(classification_output),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "model": stats.get("metadata", {}).get("model"),
        "prompt_source": stats.get("metadata", {}).get("prompt_source"),
        "initial_topics_source": stats.get("metadata", {}).get("initial_topics_source"),
        "initial_topic_count": stats.get("metadata", {}).get("initial_topic_count"),
        "classified_cases": len(unrelated_records),
        "previous_user_unrelated_topic_counts": previous_user_unrelated,
        "new_user_unrelated_topic_counts": dict(sorted(new_unrelated_counts.items())),
    }

    write_json(report_path, report)
    emit_log(f"merge_report.updated path={report_path} user_unrelated_topics={len(new_unrelated_counts)}")


def load_completed_assignments(output_path: Path, *, model: str) -> dict[str, TopicAssignment]:
    if not output_path.exists():
        return {}
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TopicClassificationError(f"resume output must be a JSON object: {output_path}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise TopicClassificationError(f"resume output missing metadata: {output_path}")
    if metadata.get("model") != model:
        raise TopicClassificationError(
            f"resume model mismatch: existing={metadata.get('model')!r}, current={model!r}. "
            "Use the same model, delete the output, or rerun without --resume."
        )

    completed: dict[str, TopicAssignment] = {}
    for raw in payload.get("case_topics") or []:
        if not isinstance(raw, dict):
            continue
        item_id = string_value(raw.get("item_id"))
        topic = string_value(raw.get("topic"))
        if not item_id or not topic:
            continue
        completed[item_id] = TopicAssignment(
            item_id=item_id,
            sample_id=string_value(raw.get("sample_id")),
            category=string_value(raw.get("category")),
            source_dataset=string_value(raw.get("source_dataset")),
            topic=topic,
            raw_response=string_value(raw.get("raw_response")),
            input_text=string_value(raw.get("input_text")) or None,
        )
    return completed


def load_topic_config(
    env_path: Path,
    *,
    base_url_override: str | None,
    api_key_override: str | None,
    model_override: str | None,
    reasoning_effort_override: str | None,
) -> TopicConfig:
    if env_path.exists():
        load_dotenv(env_path, override=False)

    prefixes = ("TOPIC", "DOMAIN", "GENERATION")
    for prefix in prefixes:
        base_url = base_url_override or os.getenv(f"{prefix}_BASE_URL")
        api_key = api_key_override or os.getenv(f"{prefix}_API_KEY")
        model = model_override or os.getenv(f"{prefix}_MODEL")
        if base_url and api_key and model:
            return TopicConfig(
                base_url=base_url,
                api_key=api_key,
                model=model,
                reasoning_effort=reasoning_effort_override or os.getenv(f"{prefix}_REASONING_EFFORT") or None,
                source_prefix=prefix,
            )

    missing = []
    if not (base_url_override or any(os.getenv(f"{prefix}_BASE_URL") for prefix in prefixes)):
        missing.append("TOPIC_BASE_URL or DOMAIN_BASE_URL or GENERATION_BASE_URL")
    if not (api_key_override or any(os.getenv(f"{prefix}_API_KEY") for prefix in prefixes)):
        missing.append("TOPIC_API_KEY or DOMAIN_API_KEY or GENERATION_API_KEY")
    if not (model_override or any(os.getenv(f"{prefix}_MODEL") for prefix in prefixes)):
        missing.append("TOPIC_MODEL or DOMAIN_MODEL or GENERATION_MODEL")
    raise TopicClassificationError(f"Missing required model config: {', '.join(missing)}")


def load_items(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON array in {path}")
    items: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Expected object at index {index} in {path}")
        item_identifier(item)
        items.append(item)
    return items


def item_identifier(item: dict[str, Any]) -> str:
    value = item.get("global_sample_id") or item.get("instance_id") or item.get("sample_id") or item.get("case_id")
    value = string_value(value)
    if not value:
        raise ValueError("unrelated item is missing global_sample_id, instance_id, sample_id, and case_id")
    return value


def append_topic(existing_topics: list[str], topic: str) -> None:
    if all(topic.lower() != existing.lower() for existing in existing_topics):
        existing_topics.append(topic)


def matching_existing_topic(topic: str, existing_topics: list[str]) -> str:
    for existing in existing_topics:
        if topic.lower() == existing.lower():
            return existing
    return topic


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def stable_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def compact_text(value: Any) -> str:
    return " ".join(stable_text(value).split())


def string_value(value: Any) -> str:
    return "" if value is None else str(value).strip()


def add_part(parts: list[str], label: str, value: Any) -> None:
    text = compact_text(value)
    if text:
        parts.append(f"{label}: {text}")


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)


def emit_log(message: str) -> None:
    print(f"{datetime.now(timezone.utc).isoformat()} {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
