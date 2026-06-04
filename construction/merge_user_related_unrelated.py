from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


DEFAULT_EMBEDDING_THRESHOLD = 0.4
DEFAULT_EMBEDDING_BATCH_SIZE = 128
FALLBACK_START = datetime(2025, 4, 1, 8, 0, 0, tzinfo=timezone(timedelta(hours=8)))
FALLBACK_END = datetime(2026, 4, 1, 22, 59, 59, tzinfo=timezone(timedelta(hours=8)))
RELATION_TYPE_BALANCE_ORDER = ("nuanced", "contradictory", "complementary")
UNSET_TOPIC = "Unset"
TOKEN_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\s]",
    re.UNICODE,
)


@dataclass
class QARecord:
    query: str
    correct_answers: list[str]
    incorrect_answers: list[str]

    def to_payload(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "correct_answers": list(self.correct_answers),
            "incorrect_answers": list(self.incorrect_answers),
        }


@dataclass
class SessionRecord:
    persona_id: str
    session_id: str
    timestamp: str
    source: str
    case_id: str
    conversation_type: str | None
    conversation_flow: str | None
    persona_signal_level: str | None
    history: list[dict[str, Any]]

    def to_payload(self, order: int) -> dict[str, Any]:
        return {
            "order": order,
            "persona_id": self.persona_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "source": self.source,
            "case_id": self.case_id,
            "conversation_type": self.conversation_type,
            "conversation_flow": self.conversation_flow,
            "persona_signal_level": self.persona_signal_level,
            "history": self.history,
        }


@dataclass
class BenchInstance:
    instance_id: str
    persona_id: str
    persona_str: str
    source: str
    case_id: str
    topic: str
    relation_type: str
    relation_subtype: str
    case: str
    facts: list[str]
    qas: list[QARecord]
    session_ids: list[str]

    def to_payload(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "persona_id": self.persona_id,
            "persona_str": self.persona_str,
            "source": self.source,
            "case_id": self.case_id,
            "topic": self.topic,
            "relation_type": self.relation_type,
            "relation_subtype": self.relation_subtype,
            "case": self.case,
            "facts": list(self.facts),
            "qas": [qa.to_payload() for qa in self.qas],
            "session_ids": list(self.session_ids),
        }


@dataclass
class PersonaDataset:
    persona_key: str
    persona_id: str
    persona_str: str
    related_sessions: list[SessionRecord] = field(default_factory=list)
    related_bench: list[BenchInstance] = field(default_factory=list)


@dataclass
class UnrelatedSession:
    source_session_id: str
    conversation_type: str | None
    conversation_flow: str | None
    history: list[dict[str, Any]]


@dataclass
class UnrelatedCase:
    sample_id: str
    topic: str
    relation_type: str
    relation_subtype: str
    case: str
    facts: list[str]
    qas: list[QARecord]
    sessions: list[UnrelatedSession]


@dataclass(frozen=True)
class EmbeddingConfig:
    base_url: str
    api_key: str
    model: str


class OpenAIEmbeddingClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        client: Any | None = None,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    ) -> None:
        if client is None:
            from openai import OpenAI

            self.client = OpenAI(base_url=base_url, api_key=api_key)
        else:
            self.client = client
        self.model = model
        self.batch_size = max(1, batch_size)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            records = embedding_records(response)
            if len(records) != len(batch):
                raise ValueError(
                    f"embedding response returned {len(records)} vectors for {len(batch)} inputs"
                )
            embeddings.extend(records)
        return embeddings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge user-related run artifacts with user-unrelated cases into persona-scoped bench files."
    )
    parser.add_argument("--related-run-dir", type=Path)
    parser.add_argument("--unrelated-input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-threshold", type=float, default=DEFAULT_EMBEDDING_THRESHOLD)
    parser.add_argument("--similarity-threshold", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--qa-mode", type=str, default=None)
    args = parser.parse_args(argv)
    if args.related_run_dir is None and args.unrelated_input is None:
        parser.error("at least one of --related-run-dir or --unrelated-input is required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        merged, report = build_merged_datasets_with_report(
            related_run_dir=args.related_run_dir,
            unrelated_input=args.unrelated_input,
            seed=args.seed,
            embedding_threshold=args.embedding_threshold,
            similarity_threshold=args.similarity_threshold,
            qa_mode=args.qa_mode,
            log_fn=log_info,
        )
        write_merged_outputs(args.output_dir, merged, report=report)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    summary = {
        "output_dir": str(args.output_dir),
        "report_path": str(args.output_dir / "merge_report.json"),
        "merge_mode": report.get("merge_mode"),
        "embedding_threshold": report.get("embedding_threshold"),
        "counts": report["counts"],
        "overview": report.get("overview"),
        "personas": {
            persona_key: {
                "history_sessions": len(payload["history_sessions"]),
                "bench_instances": len(payload["bench_instances"]),
                "user_unrelated_instances": sum(
                    1 for item in payload["bench_instances"] if item["source"] == "user-unrelated"
                ),
            }
            for persona_key, payload in merged.items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_merged_datasets(
    *,
    related_run_dir: Path | None = None,
    unrelated_input: Path | None = None,
    seed: int,
    embedding_threshold: float = DEFAULT_EMBEDDING_THRESHOLD,
    similarity_threshold: float | None = None,
    qa_mode: str | None = None,
    embedding_client: Any | None = None,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    merged, _ = build_merged_datasets_with_report(
        related_run_dir=related_run_dir,
        unrelated_input=unrelated_input,
        seed=seed,
        embedding_threshold=embedding_threshold,
        similarity_threshold=similarity_threshold,
        qa_mode=qa_mode,
        embedding_client=embedding_client,
    )
    return merged


def build_merged_datasets_with_report(
    *,
    related_run_dir: Path | None = None,
    unrelated_input: Path | None = None,
    seed: int,
    embedding_threshold: float = DEFAULT_EMBEDDING_THRESHOLD,
    similarity_threshold: float | None = None,
    qa_mode: str | None = None,
    embedding_client: Any | None = None,
    log_fn: Any | None = None,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    if related_run_dir is None and unrelated_input is None:
        raise ValueError("at least one of related_run_dir or unrelated_input is required")
    if related_run_dir is None:
        assert unrelated_input is not None
        return build_unrelated_only_datasets_with_report(
            unrelated_input=unrelated_input,
            seed=seed,
            qa_mode=qa_mode,
            log_fn=log_fn,
        )
    if unrelated_input is None:
        assert related_run_dir is not None
        return build_related_only_datasets_with_report(
            related_run_dir=related_run_dir,
            seed=seed,
            qa_mode=qa_mode,
            log_fn=log_fn,
        )

    threshold = resolve_embedding_threshold(embedding_threshold, similarity_threshold)
    personas = load_related_run(related_run_dir, qa_mode=qa_mode)
    unrelated_cases = load_unrelated_cases(unrelated_input)
    log_merge_event(
        log_fn,
        "merge.start",
        personas=len(personas),
        related_cases=sum(len(persona.related_bench) for persona in personas.values()),
        related_sessions=sum(len(persona.related_sessions) for persona in personas.values()),
        unrelated_cases=len(unrelated_cases),
        threshold=threshold,
    )
    if unrelated_cases:
        related_case_fact_texts = collect_related_case_fact_texts(personas)
        if not related_case_fact_texts:
            raise ValueError("related run has no case facts to compare against")
        if embedding_client is None:
            embedding_client = load_embedding_client()
        vectors_by_text = embed_texts_by_value(
            [*related_case_fact_texts, *collect_unrelated_case_fact_texts(unrelated_cases)],
            embedding_client,
        )
        similarity_scores = score_unrelated_cases_by_embedding(
            unrelated_cases,
            related_case_fact_texts,
            vectors_by_text,
        )
        eligible_cases = filter_unrelated_cases_by_embedding(
            unrelated_cases,
            related_case_fact_texts,
            vectors_by_text,
            threshold,
            similarity_scores=similarity_scores,
        )
        rejected_cases = [case for case in unrelated_cases if case.sample_id not in {item.sample_id for item in eligible_cases}]
        similarity_summary = summarize_similarity_scores(similarity_scores)
        log_merge_event(
            log_fn,
            "merge.embedding_filter",
            related_case_pool=len(related_case_fact_texts),
            unrelated_cases=len(unrelated_cases),
            eligible_cases=len(eligible_cases),
            rejected_cases=len(rejected_cases),
            threshold=threshold,
            min_cosine=similarity_summary["min_max_cosine"],
            avg_cosine=similarity_summary["avg_max_cosine"],
            max_cosine=similarity_summary["max_max_cosine"],
        )
    else:
        eligible_cases = []
        rejected_cases = []
        similarity_scores = {}
        similarity_summary = empty_similarity_summary()

    assignments = assign_unrelated_cases(
        eligible_cases,
        personas,
        seed=seed,
    )

    merged = build_assigned_persona_payloads(personas, assignments, log_fn=log_fn)
    report = build_merge_report(
        output_dir=None,
        merge_mode="merged",
        merged=merged,
        personas=personas,
        unrelated_cases=unrelated_cases,
        eligible_cases=eligible_cases,
        rejected_cases=rejected_cases,
        assignments=assignments,
        similarity_scores=similarity_scores,
        similarity_summary=similarity_summary,
        threshold=threshold,
        seed=seed,
        qa_mode=qa_mode,
    )
    log_merge_event(
        log_fn,
        "merge.completed",
        selected_unrelated_cases=report["counts"]["selected_unrelated_cases"],
        selected_unrelated_sessions=report["counts"]["selected_unrelated_sessions"],
        personas=len(merged),
    )
    return merged, report


def build_related_only_datasets_with_report(
    *,
    related_run_dir: Path,
    seed: int,
    qa_mode: str | None,
    log_fn: Any | None = None,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    personas = load_related_run(related_run_dir, qa_mode=qa_mode)
    assignments = {persona_key: [] for persona_key in personas}
    log_merge_event(
        log_fn,
        "merge.direct_related.start",
        personas=len(personas),
        related_cases=sum(len(persona.related_bench) for persona in personas.values()),
        related_sessions=sum(len(persona.related_sessions) for persona in personas.values()),
    )
    merged = build_assigned_persona_payloads(personas, assignments, log_fn=log_fn)
    report = build_merge_report(
        output_dir=None,
        merge_mode="related_only",
        merged=merged,
        personas=personas,
        unrelated_cases=[],
        eligible_cases=[],
        rejected_cases=[],
        assignments=assignments,
        similarity_scores={},
        similarity_summary=empty_similarity_summary(),
        threshold=None,
        seed=seed,
        qa_mode=qa_mode,
    )
    log_merge_event(
        log_fn,
        "merge.completed",
        selected_unrelated_cases=0,
        selected_unrelated_sessions=0,
        personas=len(merged),
    )
    return merged, report


def build_unrelated_only_datasets_with_report(
    *,
    unrelated_input: Path,
    seed: int,
    qa_mode: str | None,
    log_fn: Any | None = None,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    unrelated_cases = load_unrelated_cases(unrelated_input)
    persona = synthetic_unrelated_persona()
    personas = {persona.persona_key: persona}
    assignments = {persona.persona_key: list(unrelated_cases)}
    log_merge_event(
        log_fn,
        "merge.direct_unrelated.start",
        personas=len(personas),
        unrelated_cases=len(unrelated_cases),
    )
    merged = build_assigned_persona_payloads(personas, assignments, log_fn=log_fn)
    report = build_merge_report(
        output_dir=None,
        merge_mode="unrelated_only",
        merged=merged,
        personas=personas,
        unrelated_cases=unrelated_cases,
        eligible_cases=unrelated_cases,
        rejected_cases=[],
        assignments=assignments,
        similarity_scores={},
        similarity_summary=empty_similarity_summary(),
        threshold=None,
        seed=seed,
        qa_mode=qa_mode,
    )
    log_merge_event(
        log_fn,
        "merge.completed",
        selected_unrelated_cases=report["counts"]["selected_unrelated_cases"],
        selected_unrelated_sessions=report["counts"]["selected_unrelated_sessions"],
        personas=len(merged),
    )
    return merged, report


def synthetic_unrelated_persona() -> PersonaDataset:
    return PersonaDataset(
        persona_key="persona_0",
        persona_id="persona_0",
        persona_str="Synthetic persona for user-unrelated-only export.",
    )


def build_assigned_persona_payloads(
    personas: dict[str, PersonaDataset],
    assignments: dict[str, list[UnrelatedCase]],
    *,
    log_fn: Any | None = None,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    merged: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for persona_key in sorted(personas):
        selected_cases = assignments.get(persona_key, [])
        history, bench = build_persona_payloads(personas[persona_key], selected_cases)
        validate_persona_payload(persona_key, history, bench)
        merged[persona_key] = {
            "history_sessions": history,
            "bench_instances": bench,
        }
        log_merge_event(
            log_fn,
            "merge.persona_selection",
            persona_key=persona_key,
            related_cases=len(personas[persona_key].related_bench),
            selected_unrelated_cases=len(selected_cases),
            selected_relation_types=relation_type_counts(selected_cases),
        )
    return merged


def load_related_run(run_dir: Path, *, qa_mode: str | None = None) -> dict[str, PersonaDataset]:
    if not run_dir.exists():
        raise FileNotFoundError(f"related run directory does not exist: {run_dir}")
    if (run_dir / "sessions.json").exists() and (run_dir / "evaluation_instances.json").exists():
        sessions_payload, evaluation_payload, sanitized_payload = load_related_run_payloads(run_dir, qa_mode=qa_mode)
    else:
        selected_qa_mode = qa_mode or select_related_run_qa_mode(run_dir)
        if related_export_exists(run_dir, selected_qa_mode):
            return load_related_run_from_export(run_dir, qa_mode=selected_qa_mode)
        sessions_payload, evaluation_payload, sanitized_payload = load_related_run_payloads(run_dir, qa_mode=selected_qa_mode)
    if not isinstance(sessions_payload, list):
        raise ValueError("related sessions payload must contain a list")
    if not isinstance(evaluation_payload, list):
        raise ValueError("related evaluation payload must contain a list")

    persona_ids = persona_ids_from_related_payload(sanitized_payload, sessions_payload, evaluation_payload)
    if not persona_ids:
        raise ValueError("related run contains no persona_id values")
    persona_keys = build_persona_key_map(persona_ids)
    persona_strings = persona_strings_from_profiles(sanitized_payload)
    personas = {
        persona_keys[persona_id]: PersonaDataset(
            persona_key=persona_keys[persona_id],
            persona_id=persona_id,
            persona_str=persona_strings.get(persona_id) or fallback_persona_str(persona_id),
        )
        for persona_id in persona_ids
    }

    sessions_by_persona_case: dict[tuple[str, str], list[SessionRecord]] = {}
    for item in sessions_payload:
        if not isinstance(item, dict):
            raise ValueError("sessions.json items must be objects")
        persona_id = as_non_empty_str(item.get("persona_id"), "sessions[].persona_id")
        persona_key = persona_keys[persona_id]
        session = SessionRecord(
            persona_id=persona_id,
            session_id=as_non_empty_str(item.get("session_id") or item.get("conversation_id"), "session_id"),
            timestamp=as_non_empty_str(item.get("timestamp"), "sessions[].timestamp"),
            source="user-related",
            case_id=as_non_empty_str(item.get("case_id"), "sessions[].case_id"),
            conversation_type=optional_str(item.get("selected_conversation_type") or item.get("conversation_type")),
            conversation_flow=optional_str(item.get("selected_conversation_flow") or item.get("conversation_flow")),
            persona_signal_level=optional_str(item.get("persona_signal_level")),
            history=normalize_messages(item.get("messages") or item.get("history"), "sessions[].messages"),
        )
        parse_timestamp(session.timestamp)
        personas[persona_key].related_sessions.append(session)
        sessions_by_persona_case.setdefault((persona_id, session.case_id), []).append(session)

    case_groups: dict[tuple[str, str], dict[str, Any]] = {}
    for item in evaluation_payload:
        if not isinstance(item, dict):
            raise ValueError("evaluation_instances.json items must be objects")
        persona_id = as_non_empty_str(item.get("persona_id"), "evaluation_instances[].persona_id")
        persona_key = persona_keys[persona_id]
        case_id = as_non_empty_str(item.get("case_id"), "evaluation_instances[].case_id")
        group_key = (persona_id, case_id)
        case_payload = item.get("case") if isinstance(item.get("case"), dict) else {}
        facts = fact_texts(item.get("facts") or case_payload.get("facts"))
        if not facts:
            raise ValueError(f"evaluation instance has no facts for case {case_id}")
        group = case_groups.setdefault(
            group_key,
            {
                "persona_id": persona_id,
                "source": "user-related",
                "case_id": case_id,
                "topic": str(item.get("topic_preference") or case_payload.get("topic_preference") or ""),
                "relation_type": str(item.get("relation_type") or case_payload.get("relation_type") or ""),
                "relation_subtype": str(item.get("relation_subtype") or case_payload.get("relation_subtype") or ""),
                "case": case_description(item.get("case")),
                "facts": facts,
                "qas": [],
                "seen_questions": set(),
            },
        )
        question_key = (
            optional_str(item.get("qa_id")),
            optional_str(item.get("question_id")),
            as_non_empty_str(item.get("query"), "evaluation_instances[].query"),
        )
        if question_key not in group["seen_questions"]:
            group["qas"].append(
                QARecord(
                    query=question_key[2],
                    correct_answers=answer_texts(item.get("correct_answers")),
                    incorrect_answers=answer_texts(item.get("incorrect_answers")),
                )
            )
            group["seen_questions"].add(question_key)

    for (persona_id, case_id), group in sorted(case_groups.items()):
        persona_key = persona_keys[persona_id]
        case_sessions = sorted(
            sessions_by_persona_case.get((persona_id, case_id), []),
            key=session_sort_key,
        )
        if not case_sessions:
            raise ValueError(f"related case {case_id} for persona {persona_id} has no matching sessions")
        personas[persona_key].related_bench.append(
            BenchInstance(
                instance_id=stable_id("related", persona_id, case_id),
                persona_id=persona_id,
                persona_str=personas[persona_key].persona_str,
                source="user-related",
                case_id=case_id,
                topic=group["topic"],
                relation_type=group["relation_type"],
                relation_subtype=group["relation_subtype"],
                case=group["case"],
                facts=group["facts"],
                qas=group["qas"],
                session_ids=[session.session_id for session in case_sessions],
            )
        )

    return personas


def related_export_exists(run_dir: Path, qa_mode: str) -> bool:
    export_dir = run_dir / "export" / qa_mode
    return export_dir.exists() and any(export_dir.glob("persona_*.json"))


def load_related_run_from_export(run_dir: Path, *, qa_mode: str) -> dict[str, PersonaDataset]:
    export_dir = run_dir / "export" / qa_mode
    persona_dirs = sorted(path for path in run_dir.glob("persona_*") if path.is_dir())
    if not persona_dirs:
        raise FileNotFoundError(f"no persona directories found under {run_dir} for export-based merge")

    personas: dict[str, PersonaDataset] = {}
    for persona_dir in persona_dirs:
        persona_key = persona_dir.name
        persona_payload = read_json(persona_dir / "00_profile" / "persona.json")
        if not isinstance(persona_payload, dict):
            raise ValueError(f"persona profile must be an object: {persona_dir / '00_profile' / 'persona.json'}")
        persona_id = as_non_empty_str(persona_payload.get("persona_id"), f"{persona_key}.persona_id")
        personas[persona_key] = PersonaDataset(
            persona_key=persona_key,
            persona_id=persona_id,
            persona_str=persona_str_from_profile(persona_payload) or fallback_persona_str(persona_id),
        )

    for persona_key in sorted(personas):
        persona = personas[persona_key]
        export_path = export_dir / f"{persona_key}.json"
        if not export_path.exists():
            raise FileNotFoundError(f"missing export artifact for {persona_key}: {export_path}")
        export_payload = read_json(export_path)
        if not isinstance(export_payload, dict):
            raise ValueError(f"export payload must be an object: {export_path}")
        export_persona_str = persona_str_from_profile(export_payload.get("persona"))
        if export_persona_str:
            persona.persona_str = export_persona_str
        cases_payload = export_payload.get("cases")
        if not isinstance(cases_payload, list):
            raise ValueError(f"export payload cases must be a list: {export_path}")

        for case_index, case_item in enumerate(cases_payload):
            if not isinstance(case_item, dict):
                raise ValueError(f"export case must be an object: {export_path} cases[{case_index}]")
            topic = as_non_empty_str(case_item.get("topic"), f"{export_path} cases[{case_index}].topic")
            relation_type = as_non_empty_str(case_item.get("relation_type"), f"{export_path} cases[{case_index}].relation_type")
            relation_subtype = as_non_empty_str(case_item.get("relation_subtype"), f"{export_path} cases[{case_index}].relation_subtype")
            case_text = as_non_empty_str(case_item.get("case"), f"{export_path} cases[{case_index}].case")
            facts = fact_texts(case_item.get("facts"))
            if not facts:
                raise ValueError(f"export case has no facts: {export_path} cases[{case_index}]")
            case_id = stable_id(
                "related-export",
                persona.persona_id,
                persona_key,
                str(case_index),
                topic,
                relation_type,
                relation_subtype,
                case_text,
                *facts,
            )
            qas = export_case_qas(case_item, export_path=export_path, case_index=case_index)
            sessions = export_case_sessions(
                case_item,
                export_path=export_path,
                case_index=case_index,
                persona_id=persona.persona_id,
                persona_key=persona_key,
                case_id=case_id,
            )
            persona.related_sessions.extend(sessions)
            persona.related_bench.append(
                BenchInstance(
                    instance_id=stable_id("related", persona.persona_id, case_id),
                    persona_id=persona.persona_id,
                    persona_str=persona.persona_str,
                    source="user-related",
                    case_id=case_id,
                    topic=topic,
                    relation_type=relation_type,
                    relation_subtype=relation_subtype,
                    case=case_text,
                    facts=facts,
                    qas=qas,
                    session_ids=[session.session_id for session in sessions],
                )
            )
    return personas


def build_related_export_case_id_map(evaluation_payload: list[Any]) -> dict[tuple[str, str, str, str, tuple[str, ...]], str]:
    case_map: dict[tuple[str, str, str, str, tuple[str, ...]], str] = {}
    grouped: dict[str, tuple[str, str, str, str, tuple[str, ...]]] = {}
    for item in evaluation_payload:
        if not isinstance(item, dict):
            raise ValueError("evaluation_instances.json items must be objects")
        case_id = as_non_empty_str(item.get("case_id"), "evaluation_instances[].case_id")
        case_payload = item.get("case") if isinstance(item.get("case"), dict) else {}
        facts = tuple(fact_texts(item.get("facts") or case_payload.get("facts")))
        signature = related_case_signature(
            str(item.get("topic_preference") or case_payload.get("topic_preference") or ""),
            str(item.get("relation_type") or case_payload.get("relation_type") or ""),
            str(item.get("relation_subtype") or case_payload.get("relation_subtype") or ""),
            case_description(item.get("case")),
            list(facts),
        )
        existing = grouped.get(case_id)
        if existing is None:
            grouped[case_id] = signature
        elif existing != signature:
            raise ValueError(f"evaluation case {case_id} has inconsistent export signature")
    for case_id, signature in grouped.items():
        if signature in case_map and case_map[signature] != case_id:
            raise ValueError(f"duplicate export signature for related cases {case_map[signature]} and {case_id}")
        case_map[signature] = case_id
    return case_map


def related_case_signature(
    topic: str,
    relation_type: str,
    relation_subtype: str,
    case_text: str,
    facts: list[str],
) -> tuple[str, str, str, str, tuple[str, ...]]:
    return (
        topic.strip(),
        relation_type.strip(),
        relation_subtype.strip(),
        case_text.strip(),
        tuple(fact.strip() for fact in facts),
    )


def export_case_qas(case_item: dict[str, Any], *, export_path: Path, case_index: int) -> list[QARecord]:
    qa_payload = case_item.get("qa")
    if not isinstance(qa_payload, list) or not qa_payload:
        raise ValueError(f"export case has no qa entries: {export_path} cases[{case_index}]")
    qas: list[QARecord] = []
    for qa_index, item in enumerate(qa_payload):
        if not isinstance(item, dict):
            raise ValueError(f"export qa must be an object: {export_path} cases[{case_index}].qa[{qa_index}]")
        query = as_non_empty_str(item.get("query"), f"{export_path} cases[{case_index}].qa[{qa_index}].query")
        correct_answers = answer_texts(item.get("correct_answers"))
        incorrect_answers = answer_texts(item.get("incorrect_answers"))
        if not correct_answers or not incorrect_answers:
            raise ValueError(f"export qa must contain correct and incorrect answers: {export_path} cases[{case_index}].qa[{qa_index}]")
        qas.append(QARecord(query=query, correct_answers=correct_answers, incorrect_answers=incorrect_answers))
    return qas


def export_case_sessions(
    case_item: dict[str, Any],
    *,
    export_path: Path,
    case_index: int,
    persona_id: str,
    persona_key: str,
    case_id: str,
) -> list[SessionRecord]:
    sessions_payload = case_item.get("sessions")
    if not isinstance(sessions_payload, list) or not sessions_payload:
        raise ValueError(f"export case has no sessions: {export_path} cases[{case_index}]")
    sessions: list[SessionRecord] = []
    for session_index, item in enumerate(sessions_payload):
        if not isinstance(item, dict):
            raise ValueError(f"export session must be an object: {export_path} cases[{case_index}].sessions[{session_index}]")
        timestamp = as_non_empty_str(item.get("timestamp"), f"{export_path} cases[{case_index}].sessions[{session_index}].timestamp")
        parse_timestamp(timestamp)
        session_id = f"related-{persona_key}-{slug(case_id)}-s{session_index + 1}"
        sessions.append(
            SessionRecord(
                persona_id=persona_id,
                session_id=session_id,
                timestamp=timestamp,
                source="user-related",
                case_id=case_id,
                conversation_type=optional_str(item.get("conversation_type")),
                conversation_flow=optional_str(item.get("conversation_flow")),
                persona_signal_level=optional_str(item.get("persona_signal_level")),
                history=normalize_messages(item.get("history"), f"{export_path} cases[{case_index}].sessions[{session_index}].history"),
            )
        )
    return sorted(sessions, key=session_sort_key)


def load_related_run_payloads(run_dir: Path, *, qa_mode: str | None = None) -> tuple[list[Any], list[Any], list[Any]]:
    sessions_path = run_dir / "sessions.json"
    evaluation_path = run_dir / "evaluation_instances.json"
    sanitized_path = run_dir / "sanitized_personas.json"
    if sessions_path.exists() and evaluation_path.exists():
        sessions_payload = read_json(sessions_path)
        evaluation_payload = read_json(evaluation_path)
        sanitized_payload = optional_list_json(sanitized_path)
        return sessions_payload, evaluation_payload, sanitized_payload

    persona_dirs = sorted(path for path in run_dir.glob("persona_*") if path.is_dir())
    if not persona_dirs:
        raise FileNotFoundError(f"no readable related run artifacts found under {run_dir}")

    selected_qa_mode = qa_mode or select_related_run_qa_mode(run_dir)
    sessions_payload: list[Any] = []
    evaluation_payload: list[Any] = []
    sanitized_payload: list[Any] = []
    for persona_dir in persona_dirs:
        persona_path = persona_dir / "00_profile" / "persona.json"
        sessions_persona_path = persona_dir / "02_sessions" / "sessions.json"
        evaluation_persona_path = persona_dir / "04_evaluation" / selected_qa_mode / "evaluation_instances.json"
        if not sessions_persona_path.exists():
            raise FileNotFoundError(f"missing persona sessions artifact: {sessions_persona_path}")
        if not evaluation_persona_path.exists():
            raise FileNotFoundError(f"missing persona evaluation artifact: {evaluation_persona_path}")
        sanitized_payload.extend(optional_single_json_as_list(persona_path))
        sessions_payload.extend(expect_list_json(sessions_persona_path))
        evaluation_payload.extend(expect_list_json(evaluation_persona_path))
    return sessions_payload, evaluation_payload, sanitized_payload


def select_related_run_qa_mode(run_dir: Path) -> str:
    run_manifest_path = run_dir / "run_manifest.json"
    if run_manifest_path.exists():
        payload = read_json(run_manifest_path)
        qa_modes = payload.get("qa_modes")
        if isinstance(qa_modes, list):
            modes = [str(item) for item in qa_modes if str(item).strip()]
            if len(modes) == 1:
                return modes[0]
            if len(modes) > 1:
                raise ValueError(
                    f"related run contains multiple qa_modes {sorted(modes)}; pass --qa-mode explicitly"
                )
    modes = sorted(
        {
            path.name
            for persona_dir in run_dir.glob("persona_*")
            if persona_dir.is_dir()
            for evaluation_dir in [persona_dir / "04_evaluation"]
            if evaluation_dir.exists()
            for path in evaluation_dir.iterdir()
            if path.is_dir()
        }
    )
    if not modes:
        raise FileNotFoundError(f"no mode-scoped evaluation artifacts found under {run_dir}")
    if len(modes) > 1:
        raise ValueError(f"related run contains multiple qa_modes {modes}; pass --qa-mode explicitly")
    return modes[0]


def expect_list_json(path: Path) -> list[Any]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON array in {path}")
    return payload


def optional_single_json_as_list(path: Path) -> list[Any]:
    if not path.exists():
        return []
    payload = read_json(path)
    return [payload] if isinstance(payload, dict) else []


def persona_ids_from_related_payload(sanitized_profiles: list[Any], sessions: list[Any], evaluations: list[Any]) -> list[str]:
    persona_ids: set[str] = set()
    for profile in sanitized_profiles:
        if isinstance(profile, dict) and profile.get("persona_id") is not None:
            persona_ids.add(str(profile["persona_id"]))
    for item in sessions:
        if isinstance(item, dict) and item.get("persona_id") is not None:
            persona_ids.add(str(item["persona_id"]))
    for item in evaluations:
        if isinstance(item, dict) and item.get("persona_id") is not None:
            persona_ids.add(str(item["persona_id"]))
    return sorted(persona_ids, key=natural_key)


def persona_strings_from_profiles(sanitized_profiles: list[Any]) -> dict[str, str]:
    persona_strings: dict[str, str] = {}
    for profile in sanitized_profiles:
        if not isinstance(profile, dict) or profile.get("persona_id") is None:
            continue
        persona_id = str(profile["persona_id"])
        persona_str = persona_str_from_profile(profile)
        if persona_str:
            persona_strings[persona_id] = persona_str
    return persona_strings


def persona_str_from_profile(profile: Any) -> str:
    if not isinstance(profile, dict):
        return ""
    persona_str = optional_str(profile.get("persona_str"))
    if persona_str:
        return persona_str
    nested_profile = profile.get("profile")
    if isinstance(nested_profile, dict) and nested_profile:
        return json.dumps(nested_profile, ensure_ascii=False)
    if profile:
        return json.dumps(profile, ensure_ascii=False)
    return ""


def fallback_persona_str(persona_id: str) -> str:
    return json.dumps({"persona_id": persona_id}, ensure_ascii=False)


def load_unrelated_cases(path: Path) -> list[UnrelatedCase]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError("unrelated input must be the raw FanOutQA-style list, not a persona export object")
    cases = [build_unrelated_case(item, index) for index, item in enumerate(payload)]
    if not cases:
        return []
    seen: set[str] = set()
    for case in cases:
        if case.sample_id in seen:
            raise ValueError(f"duplicate unrelated sample_id: {case.sample_id}")
        seen.add(case.sample_id)
    return cases


def build_unrelated_case(item: Any, index: int) -> UnrelatedCase:
    if not isinstance(item, dict):
        raise ValueError("unrelated input items must be objects")
    sample_id = str(
        item.get("sample_id")
        or nested_get(item, ["metadata", "source_record", "seed_id"])
        or f"unrelated-{index}"
    )
    sample_id = sample_id.strip()
    if not sample_id:
        raise ValueError(f"unrelated item at index {index} has an empty sample_id")

    facts = unrelated_fact_texts(item)
    if not facts:
        raise ValueError(f"unrelated sample {sample_id} has no facts")

    qas = unrelated_qas(item, sample_id)

    topic = unrelated_topic(item)
    source_dataset = optional_str(item.get("source_dataset")) or "user-unrelated"
    source_question = unrelated_source_question(item, qas[0].query)
    case_text = unrelated_case_text(item, source_dataset, source_question)
    sessions = unrelated_sessions(item, sample_id)
    return UnrelatedCase(
        sample_id=sample_id,
        topic=topic,
        relation_type=normalize_relation_type(item.get("relationship") or item.get("task_family")),
        relation_subtype=normalize_relation_subtype(
            item.get("relation_subtype")
            or item.get("subtype")
            or item.get("complementary_subtype")
            or item.get("complementary_major_type")
            or item.get("complementary_source_subtype")
            or nested_get(item, ["category_details", "complementary_subtype"])
            or nested_get(item, ["category_details", "complementary_major_type"])
            or nested_get(item, ["category_details", "complementary_source_subtype"])
            or nested_get(item, ["category_details", "contradictory_subtype"])
            or nested_get(item, ["category_details", "nuanced_subtype"])
        ),
        case=case_text,
        facts=facts,
        qas=qas,
        sessions=sessions,
    )


def unrelated_fact_texts(item: dict[str, Any]) -> list[str]:
    for key in (
        "selected_complementary_facts",
        "selected_conflicting_facts",
        "selected_temporal_facts",
        "selected_context_facts",
        "memory_facts",
    ):
        facts = fact_texts(item.get(key))
        if facts:
            return facts

    memory_items = nested_get(item, ["metadata", "memory_items"]) or nested_get(item, ["metadata", "source_record", "memory_items"])
    if isinstance(memory_items, list):
        facts: list[str] = []
        for memory in memory_items:
            if not isinstance(memory, dict):
                continue
            direct_text = optional_str(memory.get("fact_statement") or memory.get("text"))
            if direct_text:
                facts.append(direct_text)
                continue
            subquestion = optional_str(memory.get("subquestion"))
            subanswer = optional_str(memory.get("subanswer_text") or memory.get("subanswer"))
            if subquestion and subanswer:
                facts.append(f"{subquestion} {subanswer}")
        if facts:
            return facts
    return []


def unrelated_qas(item: dict[str, Any], sample_id: str) -> list[QARecord]:
    qa_pairs = item.get("qa_pairs")
    if isinstance(qa_pairs, list) and qa_pairs:
        qas: list[QARecord] = []
        for pair in qa_pairs:
            if not isinstance(pair, dict):
                continue
            query = optional_str(pair.get("question")) or optional_str(pair.get("query"))
            if not query:
                continue
            correct_answers = answer_texts(pair.get("correct_answers") or pair.get("canonical_answer"))
            incorrect_answers = answer_texts(pair.get("incorrect_answers"))
            if not correct_answers or not incorrect_answers:
                continue
            qas.append(
                QARecord(
                    query=query,
                    correct_answers=correct_answers,
                    incorrect_answers=incorrect_answers,
                )
            )
        if qas:
            return qas

    query = optional_str(item.get("question")) or optional_str(item.get("complementary_question"))
    if not query:
        raise ValueError(f"unrelated sample {sample_id} has no question")
    correct_answers = answer_texts(item.get("correct_answers") or item.get("canonical_answer"))
    incorrect_answers = answer_texts(item.get("incorrect_answers"))
    if not correct_answers:
        raise ValueError(f"unrelated sample {sample_id} has no correct answers")
    if not incorrect_answers:
        raise ValueError(f"unrelated sample {sample_id} has no incorrect answers")
    return [QARecord(query=query, correct_answers=correct_answers, incorrect_answers=incorrect_answers)]


def unrelated_source_question(item: dict[str, Any], fallback_query: str) -> str:
    return (
        optional_str(item.get("complementary_question"))
        or optional_str(item.get("question"))
        or optional_str(nested_get(item, ["metadata", "source_question"]))
        or optional_str(nested_get(item, ["metadata", "context_question"]))
        or fallback_query
    )


def unrelated_case_text(item: dict[str, Any], source_dataset: str, source_question: str) -> str:
    return (
        optional_str(item.get("case"))
        or optional_str(item.get("case_text"))
        or optional_str(nested_get(item, ["metadata", "source_question"]))
        or optional_str(nested_get(item, ["metadata", "context_question"]))
        or f"External knowledge case derived from {source_dataset}: {source_question}"
    )


def unrelated_topic(item: dict[str, Any]) -> str:
    topic = optional_str(item.get("topic"))
    if topic:
        return topic
    categories = nested_get(item, ["metadata", "source_record", "metadata", "categories"])
    if isinstance(categories, list):
        category_texts = [str(category).strip() for category in categories if str(category).strip()]
        if category_texts:
            return " / ".join(category_texts)
    return UNSET_TOPIC


def unrelated_sessions(item: dict[str, Any], sample_id: str) -> list[UnrelatedSession]:
    plans_by_id = {
        str(plan.get("session_id")): plan
        for plan in item.get("session_plans", [])
        if isinstance(plan, dict) and plan.get("session_id") is not None
    }
    sessions_payload = item.get("sessions")
    if not isinstance(sessions_payload, list) or not sessions_payload:
        raise ValueError(f"unrelated sample {sample_id} has no sessions")

    sessions: list[UnrelatedSession] = []
    for index, session in enumerate(sessions_payload):
        if not isinstance(session, dict):
            raise ValueError(f"unrelated sample {sample_id} has a non-object session")
        source_session_id = str(session.get("session_id") or f"s{index + 1}").strip()
        if not source_session_id:
            source_session_id = f"s{index + 1}"
        plan = plans_by_id.get(source_session_id, {})
        sessions.append(
            UnrelatedSession(
                source_session_id=source_session_id,
                conversation_type=optional_str(
                    session.get("conversation_type")
                    or session.get("chosen_conversation_type")
                    or plan.get("chosen_conversation_type")
                ),
                conversation_flow=optional_str(
                    session.get("conversation_flow")
                    or session.get("chosen_conversation_flow")
                    or plan.get("chosen_conversation_flow")
                ),
                history=normalize_messages(
                    session.get("conversation") or session.get("history") or session.get("messages"),
                    f"unrelated sample {sample_id} session {source_session_id}",
                ),
            )
        )
    return sessions


def assign_unrelated_cases(
    unrelated_cases: list[UnrelatedCase],
    personas: dict[str, PersonaDataset],
    *,
    seed: int,
) -> dict[str, list[UnrelatedCase]]:
    assignments: dict[str, list[UnrelatedCase]] = {persona_key: [] for persona_key in personas}
    if not unrelated_cases:
        return assignments
    if not personas:
        raise ValueError("cannot assign unrelated cases without personas")

    remaining_cases = list(unrelated_cases)
    for persona_key in sorted(personas):
        persona = personas[persona_key]
        target_count = len(persona.related_bench)
        selected, remaining_cases = select_cases_for_persona(
            persona_key,
            remaining_cases,
            target_count=target_count,
            seed=seed,
        )
        assignments[persona_key] = selected
    return assignments


def select_cases_for_persona(
    persona_key: str,
    remaining_cases: list[UnrelatedCase],
    *,
    target_count: int,
    seed: int,
) -> tuple[list[UnrelatedCase], list[UnrelatedCase]]:
    if target_count < 1:
        return [], list(remaining_cases)

    target_by_type = relation_type_target_counts(target_count)
    rng = random.Random(stable_selection_seed(seed, persona_key))
    selected_ids: set[str] = set()
    selected_cases: list[UnrelatedCase] = []

    for relation_type in RELATION_TYPE_BALANCE_ORDER:
        needed = target_by_type[relation_type]
        if needed < 1:
            continue
        candidates = [case for case in remaining_cases if case.relation_type == relation_type]
        if len(candidates) < needed:
            raise ValueError(
                f"{persona_key} needs {needed} unrelated {relation_type} cases, but only {len(candidates)} remain"
            )
        candidates = sorted(candidates, key=lambda case: natural_key(case.sample_id))
        rng.shuffle(candidates)
        picks = candidates[:needed]
        selected_cases.extend(picks)
        selected_ids.update(case.sample_id for case in picks)

    rng.shuffle(selected_cases)
    next_remaining = [case for case in remaining_cases if case.sample_id not in selected_ids]
    return selected_cases, next_remaining


def build_persona_payloads(
    persona: PersonaDataset,
    unrelated_cases: list[UnrelatedCase],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unrelated_sessions: list[SessionRecord] = []
    unrelated_bench: list[BenchInstance] = []
    for case in unrelated_cases:
        case_id = f"unrelated-{slug(case.sample_id)}"
        session_ids: list[str] = []
        seen_session_ids: set[str] = set()
        for index, source_session in enumerate(case.sessions):
            session_id = f"unrelated-{persona.persona_key}-{slug(case.sample_id)}-{slug(source_session.source_session_id)}"
            if session_id in seen_session_ids:
                session_id = f"{session_id}-{index}"
            seen_session_ids.add(session_id)
            session_ids.append(session_id)
            unrelated_sessions.append(
                SessionRecord(
                    persona_id=persona.persona_id,
                    session_id=session_id,
                    timestamp="",
                    source="user-unrelated",
                    case_id=case_id,
                    conversation_type=source_session.conversation_type,
                    conversation_flow=source_session.conversation_flow,
                    persona_signal_level=None,
                    history=source_session.history,
                )
            )
        unrelated_bench.append(
            BenchInstance(
                instance_id=stable_id("unrelated", persona.persona_id, case.sample_id),
                persona_id=persona.persona_id,
                persona_str=persona.persona_str,
                source="user-unrelated",
                case_id=case_id,
                topic=case.topic,
                relation_type=case.relation_type,
                relation_subtype=case.relation_subtype,
                case=case.case,
                facts=case.facts,
                qas=case.qas,
                session_ids=session_ids,
            )
        )

    timestamps = assign_unrelated_timestamps(persona.related_sessions, len(unrelated_sessions))
    for session, timestamp in zip(unrelated_sessions, timestamps):
        session.timestamp = timestamp

    all_sessions = sorted([*persona.related_sessions, *unrelated_sessions], key=session_sort_key)
    history_payload = [session.to_payload(order=index) for index, session in enumerate(all_sessions)]
    bench_payload = [
        instance.to_payload()
        for instance in sorted([*persona.related_bench, *unrelated_bench], key=bench_sort_key)
    ]
    return history_payload, bench_payload


def assign_unrelated_timestamps(related_sessions: list[SessionRecord], count: int) -> list[str]:
    if count < 1:
        return []
    related_times = sorted(parse_timestamp(session.timestamp) for session in related_sessions if session.timestamp)
    if len(related_times) >= 2:
        start = related_times[0]
        end = related_times[-1]
    else:
        start = FALLBACK_START
        end = FALLBACK_END
    if end <= start:
        end = start + timedelta(days=1)
    span = end - start
    return [
        (start + span * ((index + 1) / (count + 1))).isoformat(timespec="seconds")
        for index in range(count)
    ]


def validate_persona_payload(
    persona_key: str,
    history_sessions: list[dict[str, Any]],
    bench_instances: list[dict[str, Any]],
) -> None:
    session_ids = [session["session_id"] for session in history_sessions]
    if len(session_ids) != len(set(session_ids)):
        raise ValueError(f"{persona_key} has duplicate session_id values")
    session_id_set = set(session_ids)
    timestamps = [parse_timestamp(session["timestamp"]) for session in history_sessions]
    if timestamps != sorted(timestamps):
        raise ValueError(f"{persona_key} history_sessions are not timestamp sorted")
    for instance in bench_instances:
        if not optional_str(instance.get("persona_str")):
            raise ValueError(f"{persona_key} bench instance {instance['instance_id']} has no persona_str")
        missing = [session_id for session_id in instance["session_ids"] if session_id not in session_id_set]
        if missing:
            raise ValueError(f"{persona_key} bench instance {instance['instance_id']} references missing sessions: {missing}")
        if not instance["qas"]:
            raise ValueError(f"{persona_key} bench instance {instance['instance_id']} has no qas")


def write_merged_outputs(
    output_dir: Path,
    merged: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    report: dict[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for persona_key, payload in merged.items():
        persona_dir = output_dir / persona_key
        persona_dir.mkdir(parents=True, exist_ok=True)
        write_json(persona_dir / "history_sessions.json", payload["history_sessions"])
        write_json(persona_dir / "bench_instances.json", payload["bench_instances"])
    if report is not None:
        report_payload = dict(report)
        report_payload["output_dir"] = str(output_dir)
        write_json(output_dir / "merge_report.json", report_payload)


def load_embedding_config(env_path: str | Path = ".env") -> EmbeddingConfig:
    env_file = Path(env_path)
    if env_file.exists():
        load_dotenv(env_file, override=False)
    values = {
        "EMBEDDING_BASE_URL": os.getenv("EMBEDDING_BASE_URL"),
        "EMBEDDING_API_KEY": os.getenv("EMBEDDING_API_KEY"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL"),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError(f"missing required embedding .env values: {', '.join(missing)}")
    return EmbeddingConfig(
        base_url=str(values["EMBEDDING_BASE_URL"]),
        api_key=str(values["EMBEDDING_API_KEY"]),
        model=str(values["EMBEDDING_MODEL"]),
    )


def load_embedding_client(env_path: str | Path = ".env", *, client: Any | None = None) -> OpenAIEmbeddingClient:
    config = load_embedding_config(env_path)
    return OpenAIEmbeddingClient(
        base_url=config.base_url,
        api_key=config.api_key,
        model=config.model,
        client=client,
    )


def resolve_embedding_threshold(
    embedding_threshold: float | None,
    similarity_threshold: float | None,
) -> float:
    threshold = similarity_threshold if similarity_threshold is not None else embedding_threshold
    if threshold is None:
        threshold = DEFAULT_EMBEDDING_THRESHOLD
    if threshold < 0 or threshold > 1:
        raise ValueError("embedding threshold must be between 0 and 1")
    return float(threshold)


def case_fact_embedding_text(facts: list[str]) -> str:
    return "\n".join(fact.strip() for fact in facts if fact and fact.strip())


def collect_related_case_fact_texts(personas: dict[str, PersonaDataset]) -> list[str]:
    seen: set[str] = set()
    texts: list[str] = []
    for persona_key in sorted(personas):
        for instance in personas[persona_key].related_bench:
            case_text = case_fact_embedding_text(instance.facts)
            if case_text and case_text not in seen:
                seen.add(case_text)
                texts.append(case_text)
    return texts


def collect_unrelated_case_fact_texts(unrelated_cases: list[UnrelatedCase]) -> list[str]:
    seen: set[str] = set()
    texts: list[str] = []
    for case in unrelated_cases:
        case_text = case_fact_embedding_text(case.facts)
        if case_text and case_text not in seen:
            seen.add(case_text)
            texts.append(case_text)
    return texts


def embed_texts_by_value(texts: list[str], embedding_client: Any) -> dict[str, tuple[float, ...]]:
    unique_texts: list[str] = []
    seen: set[str] = set()
    for text in texts:
        if text and text not in seen:
            seen.add(text)
            unique_texts.append(text)
    vectors = embedding_client.embed_texts(unique_texts)
    if len(vectors) != len(unique_texts):
        raise ValueError(f"embedding client returned {len(vectors)} vectors for {len(unique_texts)} unique texts")
    return {
        text: tuple(float(value) for value in vector)
        for text, vector in zip(unique_texts, vectors)
    }


def score_unrelated_cases_by_embedding(
    unrelated_cases: list[UnrelatedCase],
    related_case_fact_texts: list[str],
    vectors_by_text: dict[str, tuple[float, ...]],
) -> dict[str, float]:
    return {
        case.sample_id: sample_max_case_similarity(case, related_case_fact_texts, vectors_by_text)
        for case in unrelated_cases
    }


def filter_unrelated_cases_by_embedding(
    unrelated_cases: list[UnrelatedCase],
    related_case_fact_texts: list[str],
    vectors_by_text: dict[str, tuple[float, ...]],
    threshold: float,
    *,
    similarity_scores: dict[str, float] | None = None,
) -> list[UnrelatedCase]:
    scores = similarity_scores or score_unrelated_cases_by_embedding(
        unrelated_cases,
        related_case_fact_texts,
        vectors_by_text,
    )
    eligible: list[UnrelatedCase] = []
    for case in unrelated_cases:
        if scores[case.sample_id] < threshold:
            eligible.append(case)
    return eligible


def sample_max_case_similarity(
    case: UnrelatedCase,
    related_case_fact_texts: list[str],
    vectors_by_text: dict[str, tuple[float, ...]],
) -> float:
    case_text = case_fact_embedding_text(case.facts)
    case_vector = vectors_by_text.get(case_text)
    if case_vector is None:
        raise ValueError(f"missing embedding for unrelated case facts: {case.sample_id}")

    max_score = 0.0
    for related_case_text in related_case_fact_texts:
        related_vector = vectors_by_text.get(related_case_text)
        if related_vector is None:
            raise ValueError(f"missing embedding for related case facts: {related_case_text}")
        score = cosine_similarity(case_vector, related_vector)
        if score > max_score:
            max_score = score
    return max_score


def relation_type_target_counts(total: int) -> dict[str, int]:
    base, remainder = divmod(total, len(RELATION_TYPE_BALANCE_ORDER))
    counts = {relation_type: base for relation_type in RELATION_TYPE_BALANCE_ORDER}
    for relation_type in RELATION_TYPE_BALANCE_ORDER[:remainder]:
        counts[relation_type] += 1
    return counts


def stable_selection_seed(seed: int, persona_key: str) -> int:
    digest = hashlib.sha1(f"{seed}|{persona_key}".encode("utf-8")).hexdigest()[:16]
    return int(digest, 16)


def embedding_records(response: Any) -> list[list[float]]:
    data = response.get("data") if isinstance(response, dict) else getattr(response, "data", None)
    if not isinstance(data, list):
        raise ValueError("embedding response must contain a data list")
    ordered = sorted(data, key=embedding_record_index)
    records: list[list[float]] = []
    for item in ordered:
        embedding = item.get("embedding") if isinstance(item, dict) else getattr(item, "embedding", None)
        if not isinstance(embedding, list) or not embedding:
            raise ValueError("embedding record must contain a non-empty embedding list")
        records.append([float(value) for value in embedding])
    return records


def embedding_record_index(item: Any) -> int:
    value = item.get("index") if isinstance(item, dict) else getattr(item, "index", 0)
    try:
        return int(value)
    except Exception:
        return 0


def cosine_similarity(left: tuple[float, ...] | list[float], right: tuple[float, ...] | list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding vectors must have the same dimension")
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def relation_type_counts(cases: list[UnrelatedCase]) -> dict[str, int]:
    counts = {relation_type: 0 for relation_type in RELATION_TYPE_BALANCE_ORDER}
    for case in cases:
        counts[case.relation_type] = counts.get(case.relation_type, 0) + 1
    return counts


def summarize_similarity_scores(scores: dict[str, float]) -> dict[str, float | None]:
    if not scores:
        return empty_similarity_summary()
    values = list(scores.values())
    return {
        "min_max_cosine": min(values),
        "avg_max_cosine": sum(values) / len(values),
        "max_max_cosine": max(values),
    }


def empty_similarity_summary() -> dict[str, float | None]:
    return {
        "min_max_cosine": None,
        "avg_max_cosine": None,
        "max_max_cosine": None,
    }


def build_merge_report(
    *,
    output_dir: Path | None,
    merge_mode: str,
    merged: dict[str, dict[str, list[dict[str, Any]]]],
    personas: dict[str, PersonaDataset],
    unrelated_cases: list[UnrelatedCase],
    eligible_cases: list[UnrelatedCase],
    rejected_cases: list[UnrelatedCase],
    assignments: dict[str, list[UnrelatedCase]],
    similarity_scores: dict[str, float],
    similarity_summary: dict[str, float | None],
    threshold: float | None,
    seed: int,
    qa_mode: str | None,
) -> dict[str, Any]:
    selected_cases = [case for persona_cases in assignments.values() for case in persona_cases]
    overview = build_merged_overview(merged)
    token_counts = overview["history_sessions"]["token_counts"]
    return {
        "schema": "merged_user_memory_report_v1",
        "merge_mode": merge_mode,
        "output_dir": str(output_dir) if output_dir is not None else None,
        "embedding_threshold": threshold,
        "seed": seed,
        "qa_mode": qa_mode,
        "counts": {
            "personas": len(personas),
            "related_cases": sum(len(persona.related_bench) for persona in personas.values()),
            "related_sessions": sum(len(persona.related_sessions) for persona in personas.values()),
            "unrelated_input_cases": len(unrelated_cases),
            "eligible_unrelated_cases": len(eligible_cases),
            "rejected_unrelated_cases": len(rejected_cases),
            "selected_unrelated_cases": len(selected_cases),
            "selected_unrelated_sessions": sum(len(case.sessions) for case in selected_cases),
            "session_tokens": token_counts["total"],
            "related_session_tokens": token_counts["by_source"].get("user-related", 0),
            "unrelated_session_tokens": token_counts["by_source"].get("user-unrelated", 0),
        },
        "similarity_summary": similarity_summary,
        "overview": overview,
        "case_similarity_scores": [
            {
                "sample_id": case.sample_id,
                "relation_type": case.relation_type,
                "max_case_cosine": similarity_scores.get(case.sample_id),
                "eligible": case.sample_id in {eligible.sample_id for eligible in eligible_cases},
                "selected": case.sample_id in {selected.sample_id for selected in selected_cases},
            }
            for case in sorted(unrelated_cases, key=lambda item: natural_key(item.sample_id))
        ],
        "personas": {
            persona_key: {
                "persona_id": personas[persona_key].persona_id,
                "related_cases": len(personas[persona_key].related_bench),
                "related_sessions": len(personas[persona_key].related_sessions),
                "selected_unrelated_cases": len(assignments.get(persona_key, [])),
                "selected_unrelated_sessions": sum(len(case.sessions) for case in assignments.get(persona_key, [])),
                "selected_relation_type_counts": relation_type_counts(assignments.get(persona_key, [])),
                "selected_case_ids": [case.sample_id for case in assignments.get(persona_key, [])],
            }
            for persona_key in sorted(personas)
        },
        "top_rejected_cases": [
            {
                "sample_id": case.sample_id,
                "relation_type": case.relation_type,
                "max_case_cosine": similarity_scores.get(case.sample_id),
            }
            for case in sorted(
                rejected_cases,
                key=lambda item: (-(similarity_scores.get(item.sample_id) or 0.0), natural_key(item.sample_id)),
            )[:20]
        ],
        "top_selected_cases": [
            {
                "sample_id": case.sample_id,
                "relation_type": case.relation_type,
                "max_case_cosine": similarity_scores.get(case.sample_id),
            }
            for case in sorted(
                selected_cases,
                key=lambda item: (-(similarity_scores.get(item.sample_id) or 0.0), natural_key(item.sample_id)),
            )[:20]
        ],
        "merged_output_counts": {
            persona_key: {
                "history_sessions": len(payload["history_sessions"]),
                "bench_instances": len(payload["bench_instances"]),
                "user_unrelated_instances": sum(
                    1 for item in payload["bench_instances"] if item["source"] == "user-unrelated"
                ),
            }
            for persona_key, payload in merged.items()
        },
    }


def build_merged_overview(merged: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    bench_items = [item for payload in merged.values() for item in payload["bench_instances"]]
    history_items = [item for payload in merged.values() for item in payload["history_sessions"]]
    return {
        "bench_instances": {
            "total": len(bench_items),
            "by_source": count_key(bench_items, "source"),
            "topic_counts": count_key(bench_items, "topic"),
            "topic_counts_by_source": count_nested_by_source(bench_items, source_key="source", value_key="topic"),
            "relation_type_counts": count_key(bench_items, "relation_type"),
            "relation_type_counts_by_source": count_nested_by_source(bench_items, source_key="source", value_key="relation_type"),
        },
        "history_sessions": {
            "total": len(history_items),
            "by_source": count_key(history_items, "source"),
            "token_counts": session_token_counts(history_items),
            "conversation_type_counts": count_key(history_items, "conversation_type", missing_label="unknown"),
            "conversation_type_counts_by_source": count_nested_by_source(
                history_items,
                source_key="source",
                value_key="conversation_type",
                missing_label="unknown",
            ),
        },
    }


def session_token_counts(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, int] = {}
    total = 0
    for session in sessions:
        token_count = count_conversation_tokens(session.get("history"))
        total += token_count
        source = normalize_count_label(session.get("source"), missing_label="unknown")
        by_source[source] = by_source.get(source, 0) + token_count
    return {
        "total": total,
        "by_source": dict(sorted(by_source.items(), key=lambda pair: natural_key(pair[0]))),
    }


def count_conversation_tokens(messages: Any) -> int:
    if not isinstance(messages, list):
        return 0
    return sum(count_text_tokens(message_content(message)) for message in messages)


def count_text_tokens(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(TOKEN_RE.findall(value))
    if isinstance(value, list):
        return sum(count_text_tokens(item) for item in value)
    if isinstance(value, dict):
        return sum(count_text_tokens(item) for item in value.values())
    return len(TOKEN_RE.findall(str(value)))


def message_content(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def count_key(items: list[dict[str, Any]], key: str, *, missing_label: str = "") -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = item.get(key)
        label = normalize_count_label(value, missing_label=missing_label)
        if not label:
            continue
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda pair: natural_key(pair[0])))


def count_nested_by_source(
    items: list[dict[str, Any]],
    *,
    source_key: str,
    value_key: str,
    missing_label: str = "",
) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        source = normalize_count_label(item.get(source_key), missing_label="unknown")
        grouped.setdefault(source, []).append(item)
    return {
        source: count_key(grouped[source], value_key, missing_label=missing_label)
        for source in sorted(grouped, key=natural_key)
    }


def normalize_count_label(value: Any, *, missing_label: str = "") -> str:
    text = optional_str(value)
    if text:
        return text
    return missing_label


def log_info(message: str) -> None:
    print(message, file=sys.stderr)


def log_merge_event(log_fn: Any | None, event: str, **fields: Any) -> None:
    if log_fn is None:
        return
    parts = [event]
    for key, value in fields.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.6f}")
        elif isinstance(value, dict):
            parts.append(f"{key}={json.dumps(value, ensure_ascii=False, sort_keys=True)}")
        else:
            parts.append(f"{key}={value}")
    log_fn(" ".join(parts))


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"missing required JSON file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def optional_list_json(path: Path) -> list[Any]:
    if not path.exists():
        return []
    payload = read_json(path)
    return payload if isinstance(payload, list) else []


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_persona_key_map(persona_ids: list[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    used: set[str] = set()
    for persona_id in persona_ids:
        if persona_id.startswith("persona_"):
            base = slug(persona_id)
        elif persona_id.isdigit():
            base = f"persona_{persona_id}"
        else:
            base = f"persona_{slug(persona_id)}"
        candidate = base
        if candidate in used:
            candidate = f"{base}-{stable_digest(persona_id)}"
        output[persona_id] = candidate
        used.add(candidate)
    return output


def normalize_messages(value: Any, location: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location} must contain a non-empty message list")
    messages: list[dict[str, Any]] = []
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            raise ValueError(f"{location}[{index}] must be an object")
        role = as_non_empty_str(message.get("role"), f"{location}[{index}].role")
        content = message.get("content")
        if isinstance(content, str):
            if not content.strip():
                raise ValueError(f"{location}[{index}].content must be non-empty")
            normalized_content: str | list[Any] = content
        elif isinstance(content, list) and content:
            normalized_content = content
        else:
            raise ValueError(f"{location}[{index}].content must be a non-empty string or list")
        messages.append({"role": role, "content": normalized_content})
    return messages


def answer_texts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        text = optional_str(value.get("text") or value.get("answer") or value.get("content"))
        return [text] if text else []
    if isinstance(value, list):
        texts: list[str] = []
        for item in value:
            texts.extend(answer_texts(item))
        return texts
    text = str(value).strip()
    return [text] if text else []


def fact_texts(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    facts: list[str] = []
    for item in value:
        if isinstance(item, dict):
            source_question = optional_str(item.get("source_subquestion") or item.get("source_question") or item.get("subquestion"))
            source_answer = optional_str(item.get("answer_text") or item.get("subanswer_text") or item.get("subanswer"))
            if source_question and source_answer:
                text = f"{source_question} {source_answer}"
            else:
                text = optional_str(
                    item.get("text")
                    or item.get("fact_statement")
                    or item.get("answer_text")
                    or item.get("context_condition")
                )
        else:
            text = optional_str(item)
        if text:
            facts.append(text)
    return facts


def case_description(value: Any) -> str:
    if isinstance(value, dict):
        text = optional_str(value.get("description") or value.get("case") or value.get("case_description"))
        if text:
            return text
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError("evaluation instance case must include a description")


def normalize_relation_type(value: Any) -> str:
    text = str(value or "complementary").strip().lower()
    if text in {"complementary", "nuanced", "contradictory"}:
        return text
    if "contradict" in text:
        return "contradictory"
    if "nuance" in text or "temporal" in text or "context" in text:
        return "nuanced"
    return "complementary"


def normalize_relation_subtype(value: Any) -> str:
    text = str(value or "K>1").strip()
    lowered = text.lower().replace("-", "_").replace(" ", "_")
    if lowered in {"k_gt_1", "k>1", "k_of_n", "type1_k_of_n"}:
        return "K>1"
    if lowered in {"k_1", "k=1", "k_eq_1", "single"}:
        return "K=1"
    if lowered in {"any_one", "anyone", "any_one_of_n"}:
        return "any_one"
    if lowered == "temporal":
        return "Temporal"
    if lowered == "context":
        return "Context"
    if "contradict" in lowered:
        return text
    return text


def session_sort_key(session: SessionRecord) -> tuple[datetime, str, str, str]:
    return (parse_timestamp(session.timestamp), session.source, session.case_id, session.session_id)


def bench_sort_key(instance: BenchInstance) -> tuple[str, str, str]:
    source_rank = "0" if instance.source == "user-related" else "1"
    return (source_rank, instance.case_id, instance.instance_id)


def parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"timestamp must be ISO-8601: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def nested_get(value: Any, keys: list[str]) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def as_non_empty_str(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} is required")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    return text


def optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def stable_id(prefix: str, *parts: str) -> str:
    return f"{prefix}-{stable_digest('|'.join(parts))}"


def stable_digest(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    text = text.strip("-._")
    return text or stable_digest(value)


def natural_key(value: str) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", value)
    return tuple(int(part) if part.isdigit() else part for part in parts)


if __name__ == "__main__":
    raise SystemExit(main())
