from __future__ import annotations

import argparse
from collections import Counter
import json
import random
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, TypeVar

from core.schemas import (
    AcceptedCase,
    AcceptedCaseQA,
    CaseRelationPlanItem,
    CaseQA,
    ConversationSession,
    EvaluationInstance,
    MemoryCase,
    QAMode,
    RejectedCase,
    RejectedCaseQA,
    RejectedConversation,
    RunReport,
    RunStatus,
    SanitizedPersonaProfile,
    TimestampedSession,
    TopicPreferenceGroup,
)
from core.validators import (
    assert_no_secrets,
    calculate_pass_rate,
    filter_report_from_decisions,
    validate_duplicate_accepted_cases,
    validate_rejected_cases_excluded,
)
from filtering.case_filter import filter_memory_cases
from filtering.case_qa_filter import (
    DEFAULT_CASE_QA_FILTER_BATCH_SIZE,
    CaseQAFilterBatchResult,
    apply_existing_qa_filter_decision,
    build_case_qa_filter_batches,
    build_case_qa_filter_report,
    build_qa_filter_decision,
    filter_case_qa,
)
from filtering.conversation_filter import (
    DEFAULT_CONVERSATION_FILTER_BATCH_SIZE,
    ConversationFilterBatchResult,
    build_conversation_filter_batches,
    build_conversation_filter_decision,
    build_conversation_filter_report,
    filter_conversations,
)
from generation.case_generator import generate_memory_cases, plan_case_relations
from generation.case_qa_generator import DEFAULT_CASE_QA_QUESTION_COUNT, generate_case_qa
from generation.conversation_generator import DEFAULT_CONVERSATION_VALIDATION_RETRIES, generate_conversations_with_rejections
from infra.config import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, AppConfig, ConfigError, load_config, masked_config_summary
from infra.llm_client import (
    DEFAULT_LLM_MAX_RETRIES,
    DEFAULT_LLM_RETRY_INITIAL_DELAY,
    DEFAULT_LLM_RETRY_MAX_DELAY,
    OpenAIStreamingClient,
)
from infra.logging_utils import JsonlLogger, NullLogger, utc_now_iso
from ingestion.persona_cache import (
    is_persona_cache_dir,
    load_or_build_persona_artifacts,
    load_persona_caches,
    persona_cache_dir,
)
from ingestion.source_loader import load_source_personas
from outputs.evaluation_instances import build_evaluation_instances
from outputs.persona_outputs import (
    LEGACY_PERSONA_ARTIFACT_PATHS,
    build_export_manifest,
    build_persona_output_bundles,
    legacy_persona_artifact_path,
    persona_artifact_path,
)
from outputs.session_timeline import build_timestamped_sessions, conversation_metric_totals


T = TypeVar("T")

RUN_QA_ARTIFACT_FILENAMES = {
    "generated_case_qa": "generated_case_qa.json",
    "generated_case_qa_partial": "generated_case_qa.partial.json",
    "accepted_case_qa": "accepted_case_qa.json",
    "rejected_case_qa": "rejected_case_qa.json",
    "qa_filter_report": "qa_filter_report.json",
    "qa_question_sampling_report": "qa_question_sampling_report.json",
    "evaluation_instances": "evaluation_instances.json",
}

ROOT_INTERMEDIATE_FILENAMES = {
    "sanitized_personas.json",
    "topic_preferences.json",
    "case_relation_plan.json",
    "case_relation_plan_report.json",
    "generated_cases.json",
    "generated_cases.partial.json",
    "filter_report.json",
    "accepted_cases.json",
    "qa_eligible_cases.json",
    "rejected_cases.json",
    "generated_conversations.partial.json",
    "conversations.json",
    "qa_conversations.json",
    "sessions.json",
    "rejected_conversations.json",
    "dropped_cases_after_conversation.json",
    "conversation_filter_report.json",
    "generated_case_qa.json",
    "generated_case_qa.partial.json",
    "accepted_case_qa.json",
    "rejected_case_qa.json",
    "qa_filter_report.json",
    "qa_question_sampling_report.json",
    "evaluation_instances.json",
}

ROOT_INTERMEDIATE_DIRNAMES = {
    "03_qa",
    "04_evaluation",
}

BUILD_CONFIG_CLI_OPTION_DESTINATIONS = {
    "--input-dir": "input_dir",
    "--output-dir": "output_dir",
    "--limit": "limit",
    "--preference-limit": "preference_limit",
    "--seed": "seed",
    "--qa-mode": "qa_mode",
    "--qa-question-count": "qa_question_count",
    "--dry-run": "dry_run",
}

SUPPORTED_BUILD_CONFIG_FIELDS = frozenset(
    {
        "input_dir",
        "output_dir",
        "persona_ids",
        "limit",
        "preference_limit",
        "seed",
        "qa_mode",
        "qa_question_count",
        "dry_run",
    }
)

BRANCH_STAGE_SEQUENCE = (
    "case_relation_plan",
    "case_generation",
    "case_filter",
    "conversation_generation",
    "conversation_pruning",
    "sessions",
    "case_qa_generation",
    "case_qa_filter",
    "evaluation",
)

BRANCH_STAGE_OUTPUT_STEMS = {
    "case_relation_plan": ("case_relation_plan",),
    "case_generation": ("generated_cases",),
    "case_filter": ("accepted_cases", "rejected_cases"),
    "conversation_generation": ("conversations", "rejected_conversations"),
    "conversation_pruning": ("qa_eligible_cases", "qa_conversations", "dropped_cases_after_conversation"),
    "sessions": ("sessions",),
    "case_qa_generation": ("generated_case_qa",),
    "case_qa_filter": ("accepted_case_qa", "rejected_case_qa"),
    "evaluation": ("evaluation_instances",),
}

BRANCH_BASE_STEMS = ("persona", "topic_preferences")

QA_RANDOM_SAMPLING_RELATION_TYPES = {"complementary", "contradictory"}
QA_RANDOM_SAMPLING_REJECT_CATEGORY = "random_question_sampling"


def add_shared_build_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--to-stage",
        choices=BRANCH_STAGE_SEQUENCE,
        default=None,
        help="Stop after the selected workflow stage and leave root artifacts for inspection.",
    )
    parser.add_argument("--llm-concurrency", type=int, default=32)
    parser.add_argument("--case-concurrency", type=int, default=None)
    parser.add_argument("--filter-concurrency", type=int, default=None)
    parser.add_argument("--conversation-concurrency", type=int, default=None)
    parser.add_argument("--qa-concurrency", type=int, default=None)
    parser.add_argument(
        "--qa-mode",
        choices=[item.value for item in QAMode],
        default=QAMode.QUESTION.value,
        help="Choose whether QA generation produces direct user questions or realistic task prompts.",
    )
    parser.add_argument(
        "--qa-question-count",
        type=positive_int,
        default=DEFAULT_CASE_QA_QUESTION_COUNT,
        help="Number of QA prompts to generate per case.",
    )
    parser.add_argument("--llm-max-retries", type=int, default=DEFAULT_LLM_MAX_RETRIES)
    parser.add_argument("--llm-retry-initial-delay", type=float, default=DEFAULT_LLM_RETRY_INITIAL_DELAY)
    parser.add_argument("--llm-retry-max-delay", type=float, default=DEFAULT_LLM_RETRY_MAX_DELAY)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="Construct user-related SubtleMemory data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Run the user-related data construction workflow.")
    build_parser.add_argument("--config", type=Path, default=None)
    build_parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    build_parser.add_argument("--limit", type=positive_int, default=1)
    build_parser.add_argument(
        "--preference-limit",
        type=positive_int,
        default=None,
        help="Maximum source preferences to use per persona for this run.",
    )
    build_parser.add_argument("--resume", type=str, default=None)
    build_parser.add_argument("--dry-run", action="store_true")
    add_shared_build_runtime_args(build_parser)

    build_from_parser = subparsers.add_parser(
        "build-from",
        help="Create a new run from an existing run directory and rerun from a selected stage onward.",
    )
    build_from_parser.add_argument("--source-run", type=Path, required=True)
    build_from_parser.add_argument("--from-stage", choices=BRANCH_STAGE_SEQUENCE, required=True)
    build_from_parser.add_argument("--run-id", type=str, default=None)
    add_shared_build_runtime_args(build_from_parser)

    organize_parser = subparsers.add_parser("organize-run", help="Move legacy persona_i flat artifacts into grouped directories.")
    organize_parser.add_argument("run_dir", type=Path)

    args = parser.parse_args(argv)
    if args.command == "build-from" and args.to_stage is not None:
        if branch_stage_index(args.to_stage) < branch_stage_index(args.from_stage):
            parser.error("--to-stage must be the same as or later than --from-stage")
    if args.command == "build":
        apply_build_config_overrides(args, argv, parser)
    else:
        args.persona_ids = None
        args.limit_explicit = False
    return args


def apply_build_config_overrides(args: argparse.Namespace, argv: list[str], parser: argparse.ArgumentParser) -> None:
    args.persona_ids = None
    explicit_destinations = explicit_build_cli_destinations(argv)
    args.limit_explicit = "limit" in explicit_destinations
    if args.config is None:
        return
    try:
        config_values = load_build_config(args.config)
    except ValueError as exc:
        parser.error(f"invalid build config {args.config}: {exc}")

    for field, value in config_values.items():
        if field == "persona_ids":
            args.persona_ids = value
            continue
        if field in explicit_destinations:
            continue
        setattr(args, field, value)
        if field == "limit":
            args.limit_explicit = True


def selected_persona_ids_for_run(args: argparse.Namespace) -> list[str] | None:
    selected_persona_ids = getattr(args, "persona_ids", None)
    if selected_persona_ids is None:
        return None
    if getattr(args, "limit_explicit", False):
        return list(selected_persona_ids[: args.limit])
    return list(selected_persona_ids)


def persona_selection_limit_applied(args: argparse.Namespace) -> bool:
    selected_persona_ids = getattr(args, "persona_ids", None)
    return (
        selected_persona_ids is not None
        and getattr(args, "limit_explicit", False)
        and len(selected_persona_ids_for_run(args) or []) < len(selected_persona_ids)
    )


def explicit_build_cli_destinations(argv: list[str]) -> set[str]:
    explicit: set[str] = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        for option, destination in BUILD_CONFIG_CLI_OPTION_DESTINATIONS.items():
            if token == option or token.startswith(f"{option}="):
                explicit.add(destination)
                break
    return explicit


def load_build_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ValueError(f"config file does not exist: {config_path}")
    try:
        payload = read_json(config_path)
    except json.JSONDecodeError as exc:
        raise ValueError(f"config file is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("config file must contain a JSON object")

    unknown_fields = sorted(set(payload) - SUPPORTED_BUILD_CONFIG_FIELDS)
    if unknown_fields:
        raise ValueError(f"unknown build config fields: {', '.join(unknown_fields)}")

    config: dict[str, Any] = {}
    if "input_dir" in payload:
        config["input_dir"] = validate_build_config_path(payload["input_dir"], field="input_dir")
    if "output_dir" in payload:
        config["output_dir"] = validate_build_config_path(payload["output_dir"], field="output_dir")
    if "persona_ids" in payload:
        config["persona_ids"] = validate_build_config_persona_ids(payload["persona_ids"])
    if "limit" in payload:
        config["limit"] = validate_positive_build_config_int(payload["limit"], field="limit")
    if "preference_limit" in payload:
        config["preference_limit"] = validate_positive_build_config_int(payload["preference_limit"], field="preference_limit")
    if "seed" in payload:
        config["seed"] = validate_build_config_int(payload["seed"], field="seed")
    if "qa_mode" in payload:
        config["qa_mode"] = validate_build_config_qa_mode(payload["qa_mode"])
    if "qa_question_count" in payload:
        config["qa_question_count"] = validate_positive_build_config_int(
            payload["qa_question_count"],
            field="qa_question_count",
        )
    if "dry_run" in payload:
        config["dry_run"] = validate_build_config_bool(payload["dry_run"], field="dry_run")
    return config


def validate_build_config_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string path")
    return Path(value)


def validate_positive_build_config_int(value: Any, *, field: str) -> int:
    parsed = validate_build_config_int(value, field=field)
    if parsed < 1:
        raise ValueError(f"{field} must be a positive integer")
    return parsed


def validate_build_config_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def validate_build_config_qa_mode(value: Any) -> str:
    if not isinstance(value, str) or value not in {item.value for item in QAMode}:
        raise ValueError(f"qa_mode must be one of: {', '.join(item.value for item in QAMode)}")
    return value


def validate_build_config_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def validate_build_config_persona_ids(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("persona_ids must be a non-empty array")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, str)):
            raise ValueError("persona_ids items must be strings or integers")
        persona_id = str(item).strip()
        if not persona_id:
            raise ValueError("persona_ids items must not be empty")
        if persona_id in seen:
            raise ValueError(f"duplicate persona_id in config: {persona_id}")
        normalized.append(persona_id)
        seen.add(persona_id)
    return normalized


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def normalize_build_from_args(args: argparse.Namespace) -> None:
    source_run = Path(args.source_run)
    args.source_run = source_run
    args.input_dir = source_run_input_dir(source_run)
    args.limit = 1
    args.preference_limit = None
    args.resume = args.run_id
    args.dry_run = False
    args.persona_ids = None


def source_run_input_dir(source_run: Path) -> Path:
    report_path = source_run / "run_report.json"
    if report_path.exists():
        report = read_json(report_path)
        input_dir = report.get("input_dir")
        if isinstance(input_dir, str) and input_dir.strip():
            return Path(input_dir)
    return source_run


def branch_stage_prerequisite_stems(from_stage: str) -> list[str]:
    stems = list(BRANCH_BASE_STEMS)
    for stage in BRANCH_STAGE_SEQUENCE:
        if stage == from_stage:
            return stems
        stems.extend(BRANCH_STAGE_OUTPUT_STEMS[stage])
    raise KeyError(f"unknown branch stage: {from_stage}")


def branch_stage_index(stage: str) -> int:
    return BRANCH_STAGE_SEQUENCE.index(stage)


def should_stop_at_stage(args: argparse.Namespace, stage: str) -> bool:
    return getattr(args, "to_stage", None) == stage


def prepare_branch_run(
    *,
    source_run: Path,
    target_run: Path,
    from_stage: str,
    qa_mode: str,
    logger: JsonlLogger,
) -> None:
    source_root = source_run.resolve()
    target_root = target_run.resolve()
    if source_root == target_root:
        raise ValueError("build-from target run must differ from source run")
    if not source_root.exists():
        raise FileNotFoundError(f"source run does not exist: {source_root}")
    if not source_root.is_dir():
        raise ValueError(f"source run is not a directory: {source_root}")

    persona_dirs = iter_persona_dirs(source_root)
    if not persona_dirs:
        raise ValueError(f"source run has no persona directories: {source_root}")

    extra_entries = sorted(
        path.name
        for path in target_root.iterdir()
        if path.name != "run.log.jsonl"
    )
    if extra_entries:
        raise ValueError(
            f"target run directory already contains artifacts: {', '.join(extra_entries)}"
        )

    required_stems = branch_stage_prerequisite_stems(from_stage)
    missing: list[str] = []
    copy_plan: list[tuple[Path, Path]] = []
    copied = 0
    for source_persona_dir in persona_dirs:
        target_persona_dir = target_root / source_persona_dir.name
        for stem in required_stems:
            source_path = existing_persona_artifact_path(source_persona_dir, stem, qa_mode=qa_mode)
            if not source_path.exists():
                missing.append(f"{source_persona_dir.name}:{stem}:{source_path}")
                continue
            target_path = target_persona_dir / persona_artifact_path(stem, qa_mode=qa_mode)
            copy_plan.append((source_path, target_path))
    if missing:
        raise ValueError(
            "source run is missing required prerequisite artifacts for "
            f"{from_stage} ({qa_mode}): " + "; ".join(missing)
        )
    for source_path, target_path in copy_plan:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        copied += 1

    logger.info(
        "build_from",
        "branch_prerequisites_copied",
        source_run=str(source_root),
        from_stage=from_stage,
        qa_mode=qa_mode,
        counts={
            "personas": len(persona_dirs),
            "required_stems": len(required_stems),
            "copied_files": copied,
        },
    )


def qa_mode_value(qa_mode: str | QAMode) -> str:
    return QAMode(qa_mode).value


def mode_scoped_run_artifact_path(run_dir: Path, stem: str, *, qa_mode: str | QAMode) -> Path:
    filename = RUN_QA_ARTIFACT_FILENAMES[stem]
    mode = qa_mode_value(qa_mode)
    if stem == "evaluation_instances":
        return run_dir / "04_evaluation" / mode / filename
    return run_dir / "03_qa" / mode / filename


def legacy_run_artifact_path(run_dir: Path, stem: str) -> Path:
    return run_dir / RUN_QA_ARTIFACT_FILENAMES[stem]


def existing_run_artifact_path(run_dir: Path, stem: str, *, qa_mode: str | QAMode) -> Path:
    mode_path = mode_scoped_run_artifact_path(run_dir, stem, qa_mode=qa_mode)
    if mode_path.exists():
        return mode_path
    legacy_path = legacy_run_artifact_path(run_dir, stem)
    if legacy_path.exists():
        return legacy_path
    return mode_path


def iter_persona_dirs(run_dir: Path) -> list[Path]:
    return sorted(path for path in run_dir.glob("persona_*") if path.is_dir())


def existing_persona_artifact_path(persona_dir: Path, stem: str, *, qa_mode: str | QAMode) -> Path:
    canonical_path = persona_dir / persona_artifact_path(stem, qa_mode=qa_mode)
    if canonical_path.exists():
        return canonical_path
    legacy_grouped_path = persona_dir / legacy_persona_artifact_path(stem)
    if legacy_grouped_path.exists():
        return legacy_grouped_path
    legacy_flat_path = persona_dir / f"{stem}.json"
    if legacy_flat_path.exists():
        return legacy_flat_path
    return canonical_path


def aggregate_persona_list_models(run_dir: Path, stem: str, model_cls: type[T], *, qa_mode: str | QAMode) -> list[T]:
    items: list[T] = []
    for persona_dir in iter_persona_dirs(run_dir):
        path = existing_persona_artifact_path(persona_dir, stem, qa_mode=qa_mode)
        if not path.exists():
            continue
        items.extend(read_json_models(path, model_cls))
    return items


def aggregate_persona_object_models(run_dir: Path, stem: str, model_cls: type[T], *, qa_mode: str | QAMode) -> list[T]:
    items: list[T] = []
    for persona_dir in iter_persona_dirs(run_dir):
        path = existing_persona_artifact_path(persona_dir, stem, qa_mode=qa_mode)
        if not path.exists():
            continue
        items.append(model_cls.model_validate(read_json(path)))
    return items


def aggregate_persona_list_payloads(run_dir: Path, stem: str, *, qa_mode: str | QAMode) -> list[Any]:
    items: list[Any] = []
    for persona_dir in iter_persona_dirs(run_dir):
        path = existing_persona_artifact_path(persona_dir, stem, qa_mode=qa_mode)
        if not path.exists():
            continue
        payload = read_json(path)
        if not isinstance(payload, list):
            raise ValueError(f"expected JSON array in {path}")
        items.extend(payload)
    return items


def persona_artifact_exists(run_dir: Path, stem: str, *, qa_mode: str | QAMode) -> bool:
    return any(existing_persona_artifact_path(persona_dir, stem, qa_mode=qa_mode).exists() for persona_dir in iter_persona_dirs(run_dir))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command in {"build", "build-from"}:
        return run_build(args)
    if args.command == "organize-run":
        report = organize_persona_output_dirs(args.run_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    return 1


def run_build(
    args: argparse.Namespace,
    *,
    generation_client: Any | None = None,
    filter_client: Any | None = None,
) -> int:
    if getattr(args, "command", None) == "build-from":
        normalize_build_from_args(args)
    source_run = getattr(args, "source_run", None)
    from_stage = getattr(args, "from_stage", None)
    run_id = getattr(args, "resume", None) or getattr(args, "run_id", None) or make_run_id()
    if getattr(args, "command", None) == "build-from":
        args.resume = run_id
    started_at = utc_now_iso()
    output_base = args.output_dir
    output_base.mkdir(parents=True, exist_ok=True)

    try:
        config = load_config(require_secrets=not args.dry_run)
    except ConfigError as exc:
        logger = JsonlLogger(run_id, level=args.log_level)
        logger.error("config", "load_config", error=str(exc))
        logger.close()
        return 1

    if args.dry_run:
        return run_dry_run(args, config, run_id)

    run_dir = output_base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(run_id, level=args.log_level, log_file=run_dir / "run.log.jsonl", secrets=config.secrets)
    counts = initial_counts()
    errors: list[dict[str, Any]] = []

    def stop_if_requested(stage: str) -> int | None:
        if not should_stop_at_stage(args, stage):
            return None
        return complete_run(
            run_id,
            started_at,
            args,
            config,
            counts,
            errors,
            run_dir,
            logger,
            source_run=source_run,
            from_stage=from_stage,
            stopped_at_stage=stage,
        )

    try:
        if source_run is not None and from_stage is not None:
            prepare_branch_run(
                source_run=Path(source_run),
                target_run=run_dir,
                from_stage=str(from_stage),
                qa_mode=args.qa_mode,
                logger=logger,
            )
        generation = generation_client or make_generation_client(config, logger, args)
        filtering = filter_client or make_filter_client(config, logger, args)
        effective_persona_ids = selected_persona_ids_for_run(args)
        logger.info(
            "config",
            "config_loaded",
            config=masked_config_summary(config),
            output_dir=str(run_dir),
            config_path=str(args.config.resolve()) if getattr(args, "config", None) is not None else None,
            persona_ids=list(effective_persona_ids or []),
            configured_persona_ids=list(getattr(args, "persona_ids", None) or []),
            persona_selection_limit_applied=persona_selection_limit_applied(args),
            persona_limit=args.limit if getattr(args, "limit_explicit", False) else None,
            preference_limit=args.preference_limit,
            case_concurrency=case_concurrency(args),
            filter_concurrency=filter_concurrency(args),
            conversation_concurrency=conversation_concurrency(args),
            qa_concurrency=qa_concurrency(args),
            qa_mode=args.qa_mode,
            qa_question_count=args.qa_question_count,
            llm_max_retries=args.llm_max_retries,
            llm_retry_initial_delay=args.llm_retry_initial_delay,
            llm_retry_max_delay=args.llm_retry_max_delay,
            source_run=str(Path(source_run).resolve()) if source_run is not None else None,
            from_stage=str(from_stage) if from_stage is not None else None,
            to_stage=str(args.to_stage) if getattr(args, "to_stage", None) is not None else None,
        )

        sanitized, topic_groups, persona_counts = load_or_resume_persona_artifacts(args, output_base, run_dir, generation, logger)
        personas = {profile.persona_id: profile for profile in sanitized}
        counts.update(persona_counts)
        counts["processed_personas"] = len(sanitized)
        counts["topics"] = len(topic_groups)
        counts["source_preferences"] = count_topic_group_preferences(topic_groups)
        logger.info(
            "source",
            "source_loaded",
            counts={
                "personas": len(sanitized),
                "topics": len(topic_groups),
                "source_preferences": counts["source_preferences"],
                **persona_counts,
            },
        )

        case_relation_plan, case_relation_plan_report = load_or_plan_case_relations(
            args,
            run_dir,
            topic_groups,
            personas,
            generation,
            config,
            logger,
        )
        counts["case_relation_plan_items"] = len(case_relation_plan)
        logger.info("case_relation_planning", "plan_complete", counts=case_relation_plan_report)
        if (code := stop_if_requested("case_relation_plan")) is not None:
            return code

        generated_cases = load_or_generate_cases(
            args,
            run_dir,
            topic_groups,
            personas,
            generation,
            config,
            logger,
            case_relation_plan,
        )
        counts["generated_cases"] = len(generated_cases)
        if (code := stop_if_requested("case_generation")) is not None:
            return code

        accepted_cases, rejected_cases, filter_report = load_or_filter_cases(
            args,
            run_dir,
            generated_cases,
            personas,
            filtering,
            config,
            logger,
        )
        counts["accepted_cases"] = len(accepted_cases)
        counts["rejected_cases"] = len(rejected_cases)
        logger.info("case_filter", "filter_complete", counts=filter_report)
        if (code := stop_if_requested("case_filter")) is not None:
            return code

        conversations, rejected_conversations, conversation_filter_report = load_or_generate_conversations(
            args,
            run_dir,
            accepted_cases,
            rejected_cases,
            personas,
            generation,
            filtering,
            config,
            logger,
        )
        counts["generated_conversations"] = conversation_filter_report.get("total_generated_conversations", len(conversations))
        counts["accepted_conversations"] = len(conversations)
        counts["rejected_conversations"] = len(rejected_conversations)
        logger.info("conversation_filter", "filter_complete", counts=conversation_filter_report)
        if (code := stop_if_requested("conversation_generation")) is not None:
            return code

        qa_eligible_cases, qa_conversations, dropped_cases_after_conversation, pruning_report = (
            load_or_prune_case_conversation_coverage(
                args,
                run_dir,
                accepted_cases,
                conversations,
                rejected_conversations,
                logger,
            )
        )
        counts["qa_eligible_cases"] = len(qa_eligible_cases)
        counts["qa_conversations"] = len(qa_conversations)
        counts["conversation_dropped_cases"] = len(dropped_cases_after_conversation)
        conversation_filter_report = write_conversation_pruning_report(run_dir, conversation_filter_report, pruning_report)
        logger.info("case_conversation_pruning", "pruning_complete", counts=pruning_report)
        if (code := stop_if_requested("conversation_pruning")) is not None:
            return code

        sessions = load_or_build_sessions(args, run_dir, qa_conversations, logger)
        counts.update(conversation_metric_totals(sessions))
        if (code := stop_if_requested("sessions")) is not None:
            return code

        generated_case_qa = load_or_generate_case_qa(
            args,
            run_dir,
            qa_eligible_cases,
            qa_conversations,
            personas,
            generation,
            config,
            logger,
        )
        counts["generated_case_qa"] = len(generated_case_qa)
        if (code := stop_if_requested("case_qa_generation")) is not None:
            return code

        accepted_case_qa, rejected_case_qa, qa_filter_report = load_or_filter_case_qa(
            args,
            run_dir,
            generated_case_qa,
            qa_eligible_cases,
            qa_conversations,
            personas,
            filtering,
            config,
            logger,
        )
        accepted_case_qa, qa_filter_report, qa_question_sampling_report = apply_qa_question_sampling(
            args,
            run_dir,
            generated_case_qa,
            accepted_case_qa,
            rejected_case_qa,
            qa_eligible_cases,
            qa_conversations,
            personas,
            config,
        )
        counts["accepted_case_qa"] = len(accepted_case_qa)
        counts["rejected_case_qa"] = len(rejected_case_qa)
        counts["accepted_case_qa_questions"] = sum(len(item.qa.questions) for item in accepted_case_qa)
        counts["qa_question_sampling_removed"] = int(
            qa_question_sampling_report.get("counts", {}).get("questions_removed", 0)
        )
        logger.info("case_qa_filter", "filter_complete", counts=qa_filter_report)
        logger.info("case_qa_sampling", "sampling_complete", counts=qa_question_sampling_report["counts"])
        if (code := stop_if_requested("case_qa_filter")) is not None:
            return code

        evaluation_instances = load_or_build_evaluation_instances(
            args,
            run_dir,
            accepted_case_qa,
            qa_eligible_cases,
            qa_conversations,
            personas,
            logger,
        )
        counts["evaluation_instances"] = len(evaluation_instances)
        if (code := stop_if_requested("evaluation")) is not None:
            return code

        persona_extra_artifacts = build_persona_extra_artifacts(
            personas=personas,
            topic_groups=topic_groups,
            case_relation_plan=case_relation_plan,
            generated_cases=generated_cases,
            accepted_cases=accepted_cases,
            rejected_cases=rejected_cases,
            filter_report=filter_report,
            conversations=conversations,
            qa_eligible_cases=qa_eligible_cases,
            qa_conversations=qa_conversations,
            rejected_conversations=rejected_conversations,
            dropped_cases_after_conversation=dropped_cases_after_conversation,
            conversation_filter_report=conversation_filter_report,
            generated_case_qa=generated_case_qa,
            accepted_case_qa=accepted_case_qa,
            rejected_case_qa=rejected_case_qa,
            qa_filter_report=qa_filter_report,
        )
        write_persona_outputs(
            run_dir=run_dir,
            qa_mode=args.qa_mode,
            personas=personas,
            topic_groups=topic_groups,
            extra_artifacts_by_persona=persona_extra_artifacts,
            generated_cases=generated_cases,
            accepted_cases=accepted_cases,
            qa_eligible_cases=qa_eligible_cases,
            rejected_cases=rejected_cases,
            conversations=conversations,
            qa_conversations=qa_conversations,
            sessions=sessions,
            rejected_conversations=rejected_conversations,
            dropped_cases_after_conversation=dropped_cases_after_conversation,
            generated_case_qa=generated_case_qa,
            accepted_case_qa=accepted_case_qa,
            rejected_case_qa=rejected_case_qa,
            evaluation_instances=evaluation_instances,
        )
        return complete_run(
            run_id,
            started_at,
            args,
            config,
            counts,
            errors,
            run_dir,
            logger,
            source_run=source_run,
            from_stage=from_stage,
            cleanup_intermediate=True,
        )
    except KeyboardInterrupt:
        report = build_run_report(run_id, started_at, args, config, counts, errors, RunStatus.INTERRUPTED)
        write_json(run_dir / "run_report.json", report)
        logger.error("run", "run_interrupted", status="interrupted", counts=counts)
        return 2
    except Exception as exc:
        errors.append({"phase": "run", "message": str(exc)})
        counts["validation_failures"] = counts.get("validation_failures", 0) + 1
        report = build_run_report(run_id, started_at, args, config, counts, errors, RunStatus.FAILED)
        write_json(run_dir / "run_report.json", report)
        logger.error("run", "run_failed", error=str(exc), counts=counts)
        return 1
    finally:
        logger.close()


def complete_run(
    run_id: str,
    started_at: str,
    args: argparse.Namespace,
    config: AppConfig,
    counts: dict[str, int],
    errors: list[dict[str, Any]],
    run_dir: Path,
    logger: JsonlLogger,
    *,
    source_run: Path | None,
    from_stage: str | None,
    stopped_at_stage: str | None = None,
    cleanup_intermediate: bool = False,
) -> int:
    if cleanup_intermediate:
        remove_run_intermediate_artifacts(run_dir)
    write_json(
        run_dir / "run_manifest.json",
        build_run_manifest(
            run_dir,
            source_run=str(Path(source_run).resolve()) if source_run is not None else None,
            from_stage=str(from_stage) if from_stage is not None else None,
            to_stage=stopped_at_stage,
        ),
    )
    report = build_run_report(run_id, started_at, args, config, counts, errors, RunStatus.COMPLETED)
    assert_no_secrets(report.model_dump(mode="json"), config.secrets)
    write_json(run_dir / "run_report.json", report)
    if stopped_at_stage is not None:
        logger.info("run", "run_stopped_at_stage", counts=counts, pass_rate=report.pass_rate, to_stage=stopped_at_stage)
    else:
        logger.info("run", "run_complete", counts=counts, pass_rate=report.pass_rate)
    return 0


def load_or_resume_persona_artifacts(
    args: argparse.Namespace,
    output_base: Path,
    run_dir: Path,
    generation_client: Any,
    logger: JsonlLogger,
) -> tuple[list[SanitizedPersonaProfile], list[TopicPreferenceGroup], dict[str, int]]:
    sanitized_path = run_dir / "sanitized_personas.json"
    topic_path = run_dir / "topic_preferences.json"
    expected_persona_ids = selected_persona_ids_for_run(args)
    if args.resume and sanitized_path.exists() and topic_path.exists():
        sanitized = read_json_models(sanitized_path, SanitizedPersonaProfile)
        topic_groups = read_json_models(topic_path, TopicPreferenceGroup)
        if persona_artifacts_match_selection(sanitized, topic_groups, expected_persona_ids):
            return sanitized, topic_groups, {"cache_hits": 0, "cache_misses": 0}
        logger.info(
            "source",
            "resume_persona_artifacts_mismatch_rebuild",
            counts={"existing_personas": len(sanitized), "expected_personas": len(expected_persona_ids or [])},
        )
    if args.resume and persona_artifact_exists(run_dir, "persona", qa_mode=QAMode.QUESTION):
        sanitized = aggregate_persona_object_models(run_dir, "persona", SanitizedPersonaProfile, qa_mode=QAMode.QUESTION)
        topic_groups = aggregate_persona_list_models(run_dir, "topic_preferences", TopicPreferenceGroup, qa_mode=QAMode.QUESTION)
        if sanitized and topic_groups and persona_artifacts_match_selection(sanitized, topic_groups, expected_persona_ids):
            return sanitized, topic_groups, {"cache_hits": 0, "cache_misses": 0}

    selected_persona_ids = expected_persona_ids
    if is_persona_cache_dir(args.input_dir):
        sanitized, topic_groups = load_persona_caches(
            args.input_dir,
            None if selected_persona_ids is not None else args.limit,
            persona_ids=selected_persona_ids,
        )
        cache_counts = {"cache_hits": len(sanitized), "cache_misses": 0}
    else:
        source_records = load_source_personas(
            args.input_dir,
            None if selected_persona_ids is not None else args.limit,
            exclude_dirs=[output_base],
            persona_ids=selected_persona_ids,
        )
        if not source_records:
            raise ValueError(f"No source records discovered in {args.input_dir}")
        sanitized, topic_groups, cache_counts = load_or_build_persona_artifacts(
            source_records,
            persona_cache_dir(output_base),
            generation_client,
        )
    if selected_persona_ids is not None:
        logger.info(
            "source",
            "persona_selection_applied",
            persona_ids=selected_persona_ids,
            configured_persona_ids=list(getattr(args, "persona_ids", None) or []),
            limit_applied=persona_selection_limit_applied(args),
            persona_limit=args.limit if getattr(args, "limit_explicit", False) else None,
        )
    original_preference_count = count_topic_group_preferences(topic_groups)
    topic_groups = limit_topic_group_preferences(topic_groups, args.preference_limit, random.Random(args.seed))
    limited_preference_count = count_topic_group_preferences(topic_groups)

    write_json(sanitized_path, sanitized)
    write_json(topic_path, topic_groups)
    logger.info(
        "source",
        "persona_artifacts_written",
        counts={
            "personas": len(sanitized),
            "topics": len(topic_groups),
            "source_preferences": limited_preference_count,
            "source_preferences_before_limit": original_preference_count,
            "preference_limit_per_persona": args.preference_limit or 0,
            **cache_counts,
        },
    )
    return sanitized, topic_groups, cache_counts


def persona_artifacts_match_selection(
    sanitized: list[SanitizedPersonaProfile],
    topic_groups: list[TopicPreferenceGroup],
    expected_persona_ids: list[str] | None,
) -> bool:
    if expected_persona_ids is None:
        return True
    observed_persona_ids = [profile.persona_id for profile in sanitized]
    if observed_persona_ids != expected_persona_ids:
        return False
    expected_set = set(expected_persona_ids)
    return all(group.persona_id in expected_set for group in topic_groups)


def limit_topic_group_preferences(
    topic_groups: list[TopicPreferenceGroup],
    preference_limit: int | None,
    rng: random.Random,
) -> list[TopicPreferenceGroup]:
    if preference_limit is None:
        return topic_groups

    selected_by_persona = sample_preference_ids_by_persona(topic_groups, preference_limit, rng)
    limited_groups: list[TopicPreferenceGroup] = []
    for group in topic_groups:
        selected_ids = selected_by_persona.get(group.persona_id, set())
        preferences = [preference for preference in group.preferences if preference.preference_id in selected_ids]
        if preferences:
            limited_groups.append(
                group.model_copy(
                    update={
                        "preferences": preferences,
                        "source_count": len(preferences),
                    }
                )
            )
    return limited_groups


def sample_preference_ids_by_persona(
    topic_groups: list[TopicPreferenceGroup],
    preference_limit: int,
    rng: random.Random,
) -> dict[str, set[str]]:
    preference_ids_by_persona: dict[str, list[str]] = {}
    for group in topic_groups:
        for preference in group.preferences:
            preference_ids_by_persona.setdefault(group.persona_id, []).append(preference.preference_id)
    return {
        persona_id: set(preference_ids if len(preference_ids) <= preference_limit else rng.sample(preference_ids, preference_limit))
        for persona_id, preference_ids in preference_ids_by_persona.items()
    }


def count_topic_group_preferences(topic_groups: list[TopicPreferenceGroup]) -> int:
    return sum(len(group.preferences) for group in topic_groups)


def load_or_plan_case_relations(
    args: argparse.Namespace,
    run_dir: Path,
    topic_groups: list[TopicPreferenceGroup],
    personas: dict[str, SanitizedPersonaProfile],
    generation_client: Any,
    config: AppConfig,
    logger: JsonlLogger,
) -> tuple[list[CaseRelationPlanItem], dict[str, Any]]:
    plan_path = run_dir / "case_relation_plan.json"
    report_path = run_dir / "case_relation_plan_report.json"
    if args.resume and plan_path.exists() and report_path.exists():
        plan_items = read_json_models(plan_path, CaseRelationPlanItem)
        if case_relation_plan_matches_preferences(plan_items, topic_groups):
            report = read_json(report_path)
            logger.info(
                "case_relation_planning",
                "resume_existing_plan",
                counts={"case_relation_plan_items": len(plan_items)},
            )
            return plan_items, report
        logger.info(
            "case_relation_planning",
            "resume_plan_mismatch_rebuild",
            counts={
                "existing_plan_items": len(plan_items),
                "source_preferences": count_topic_group_preferences(topic_groups),
            },
        )
    if args.resume and persona_artifact_exists(run_dir, "case_relation_plan", qa_mode=QAMode.QUESTION):
        plan_items = aggregate_persona_list_models(run_dir, "case_relation_plan", CaseRelationPlanItem, qa_mode=QAMode.QUESTION)
        if plan_items and case_relation_plan_matches_preferences(plan_items, topic_groups):
            report = build_case_relation_plan_report_from_items(plan_items)
            logger.info(
                "case_relation_planning",
                "resume_existing_persona_plan",
                counts={"case_relation_plan_items": len(plan_items)},
            )
            return plan_items, report

    plan_items, report = plan_case_relations(
        topic_groups,
        personas,
        generation_client,
        generation_model=config.generation_model,
        concurrency=case_concurrency(args),
        rng=random.Random(args.seed),
    )
    write_json(plan_path, plan_items)
    write_json(report_path, report)
    return plan_items, report


def case_relation_plan_matches_preferences(
    plan_items: list[CaseRelationPlanItem],
    topic_groups: list[TopicPreferenceGroup],
) -> bool:
    expected: dict[tuple[str, str], str] = {}
    for group in topic_groups:
        for preference in group.preferences:
            key = (preference.persona_id, preference.preference_id)
            if key in expected:
                return False
            expected[key] = group.topic_preference

    observed: dict[tuple[str, str], str] = {}
    for item in plan_items:
        key = (item.persona_id, item.preference_id)
        if key in observed:
            return False
        observed[key] = item.topic_preference
    return observed == expected


def load_or_generate_cases(
    args: argparse.Namespace,
    run_dir: Path,
    topic_groups: list[TopicPreferenceGroup],
    personas: dict[str, SanitizedPersonaProfile],
    generation_client: Any,
    config: AppConfig,
    logger: JsonlLogger,
    case_relation_plan: list[CaseRelationPlanItem] | None = None,
) -> list[MemoryCase]:
    generated_path = run_dir / "generated_cases.json"
    partial_path = run_dir / "generated_cases.partial.json"
    if args.resume and generated_path.exists():
        cases = read_json_models(generated_path, MemoryCase)
        if case_relation_plan is not None and not generated_cases_match_relation_plan(cases, case_relation_plan):
            logger.info(
                "case_generation",
                "resume_cases_mismatch_rebuild",
                counts={"existing_generated_cases": len(cases), "expected_plan_items": len(case_relation_plan)},
            )
        else:
            logger.info("case_generation", "resume_existing_cases", counts={"generated_cases": len(cases)})
            return cases
    if args.resume and persona_artifact_exists(run_dir, "generated_cases", qa_mode=QAMode.QUESTION):
        cases = aggregate_persona_list_models(run_dir, "generated_cases", MemoryCase, qa_mode=QAMode.QUESTION)
        if cases:
            if case_relation_plan is not None and not generated_cases_match_relation_plan(cases, case_relation_plan):
                logger.info(
                    "case_generation",
                    "resume_persona_cases_mismatch_rebuild",
                    counts={"existing_generated_cases": len(cases), "expected_plan_items": len(case_relation_plan)},
                )
            else:
                logger.info("case_generation", "resume_existing_persona_cases", counts={"generated_cases": len(cases)})
                return cases

    cases = generate_memory_cases(
        topic_groups,
        personas,
        generation_client,
        generation_model=config.generation_model,
        concurrency=case_concurrency(args),
        rng=random.Random(args.seed),
        checkpoint_path=partial_path,
        logger=logger,
        relation_plan=case_relation_plan,
    )
    write_json(generated_path, cases)
    remove_file_if_exists(partial_path)
    return cases


def generated_cases_match_relation_plan(
    cases: list[MemoryCase],
    case_relation_plan: list[CaseRelationPlanItem],
) -> bool:
    plan_by_key: dict[tuple[str, str], CaseRelationPlanItem] = {}
    for item in case_relation_plan:
        key = (item.persona_id, item.preference_id)
        if key in plan_by_key:
            return False
        plan_by_key[key] = item

    case_keys = {(case.persona_id, case.source_preference_id) for case in cases}
    if case_keys != set(plan_by_key):
        return False

    for case in cases:
        plan_item = plan_by_key[(case.persona_id, case.source_preference_id)]
        if plan_item.topic_preference != case.topic_preference:
            return False
        if str(plan_item.relation_type) != str(case.relation_type):
            return False
        if str(plan_item.relation_subtype) != str(case.relation_subtype):
            return False
    return True


def load_or_filter_cases(
    args: argparse.Namespace,
    run_dir: Path,
    generated_cases: list[MemoryCase],
    personas: dict[str, SanitizedPersonaProfile],
    filter_client: Any,
    config: AppConfig,
    logger: JsonlLogger,
) -> tuple[list[AcceptedCase], list[RejectedCase], dict[str, Any]]:
    accepted_path = run_dir / "accepted_cases.json"
    rejected_path = run_dir / "rejected_cases.json"
    report_path = run_dir / "filter_report.json"
    if args.resume and accepted_path.exists() and rejected_path.exists() and report_path.exists():
        accepted_cases = read_json_models(accepted_path, AcceptedCase)
        rejected_cases = read_json_models(rejected_path, RejectedCase)
        if case_filter_artifacts_match_generated(generated_cases, accepted_cases, rejected_cases):
            report = read_json(report_path)
            logger.info("case_filter", "resume_existing_filter", counts={"accepted": len(accepted_cases), "rejected": len(rejected_cases)})
            return accepted_cases, rejected_cases, report
        logger.info(
            "case_filter",
            "resume_filter_mismatch_rebuild",
            counts={
                "existing_cases": len(accepted_cases) + len(rejected_cases),
                "expected_generated_cases": len(generated_cases),
            },
        )
    if args.resume and persona_artifact_exists(run_dir, "accepted_cases", qa_mode=QAMode.QUESTION):
        accepted_cases = aggregate_persona_list_models(run_dir, "accepted_cases", AcceptedCase, qa_mode=QAMode.QUESTION)
        rejected_cases = aggregate_persona_list_models(run_dir, "rejected_cases", RejectedCase, qa_mode=QAMode.QUESTION)
        if (accepted_cases or rejected_cases) and case_filter_artifacts_match_generated(generated_cases, accepted_cases, rejected_cases):
            report = build_aggregated_case_filter_report(
                generated_cases=generated_cases,
                accepted_cases=accepted_cases,
                rejected_cases=rejected_cases,
                filter_model=config.filter_model,
            )
            logger.info("case_filter", "resume_existing_persona_filter", counts={"accepted": len(accepted_cases), "rejected": len(rejected_cases)})
            return accepted_cases, rejected_cases, report

    accepted_cases, rejected_cases, report = filter_memory_cases(
        generated_cases,
        personas,
        filter_client,
        filter_model=config.filter_model,
        concurrency=filter_concurrency(args),
    )
    validate_duplicate_accepted_cases(accepted_cases)
    write_json(report_path, report)
    write_json(accepted_path, accepted_cases)
    write_json(rejected_path, rejected_cases)
    return accepted_cases, rejected_cases, report


def case_filter_artifacts_match_generated(
    generated_cases: list[MemoryCase],
    accepted_cases: list[AcceptedCase],
    rejected_cases: list[RejectedCase],
) -> bool:
    generated_case_ids = {case.case_id for case in generated_cases}
    accepted_case_ids = {accepted.case.case_id for accepted in accepted_cases}
    rejected_case_ids = {rejected.case.case_id for rejected in rejected_cases}
    return not (accepted_case_ids & rejected_case_ids) and accepted_case_ids | rejected_case_ids == generated_case_ids


def load_or_generate_conversations(
    args: argparse.Namespace,
    run_dir: Path,
    accepted_cases: list[AcceptedCase],
    rejected_cases: list[RejectedCase],
    personas: dict[str, SanitizedPersonaProfile],
    generation_client: Any,
    filter_client: Any,
    config: AppConfig,
    logger: JsonlLogger,
) -> tuple[list[ConversationSession], list[RejectedConversation], dict[str, Any]]:
    conversations_path = run_dir / "conversations.json"
    rejected_path = run_dir / "rejected_conversations.json"
    report_path = run_dir / "conversation_filter_report.json"
    partial_path = run_dir / "generated_conversations.partial.json"
    if args.resume and conversations_path.exists() and rejected_path.exists() and report_path.exists():
        conversations = read_json_models(conversations_path, ConversationSession)
        rejected = read_json_models(rejected_path, RejectedConversation)
        if conversation_artifacts_match_cases(accepted_cases, conversations, rejected):
            report = read_json(report_path)
            report = add_conversation_type_metrics(report, conversations, rejected)
            write_json(report_path, report)
            logger.info("conversation_filter", "resume_existing_filter", counts={"accepted": len(conversations), "rejected": len(rejected)})
            return conversations, rejected, report
        logger.info(
            "conversation_filter",
            "resume_conversations_mismatch_rebuild",
            counts={
                "existing_conversations": len(conversations) + len(rejected),
                "expected_fact_units": count_accepted_case_facts(accepted_cases),
            },
        )
    if args.resume and persona_artifact_exists(run_dir, "conversations", qa_mode=QAMode.QUESTION):
        conversations = aggregate_persona_list_models(run_dir, "conversations", ConversationSession, qa_mode=QAMode.QUESTION)
        rejected = aggregate_persona_list_models(run_dir, "rejected_conversations", RejectedConversation, qa_mode=QAMode.QUESTION)
        if (conversations or rejected) and conversation_artifacts_match_cases(accepted_cases, conversations, rejected):
            report = build_aggregated_conversation_filter_report(
                accepted_cases=accepted_cases,
                conversations=conversations,
                rejected_conversations=rejected,
                personas=personas,
                filter_model=config.filter_model,
            )
            logger.info("conversation_filter", "resume_existing_persona_filter", counts={"accepted": len(conversations), "rejected": len(rejected)})
            return conversations, rejected, report

    generated, local_rejected = generate_conversations_with_rejections(
        accepted_cases,
        personas,
        generation_client,
        rng=random.Random(args.seed),
        generation_model=config.generation_model,
        concurrency=conversation_concurrency(args),
        validation_retries=DEFAULT_CONVERSATION_VALIDATION_RETRIES,
        checkpoint_path=partial_path,
        logger=logger,
    )
    validate_rejected_cases_excluded(generated, rejected_cases)
    conversations, filter_rejected, report = filter_conversations(
        generated,
        accepted_cases,
        personas,
        filter_client,
        filter_model=config.filter_model,
        concurrency=filter_concurrency(args),
    )
    rejected = [*local_rejected, *filter_rejected]
    report = merge_local_conversation_rejections(report, local_rejected)
    report = add_conversation_type_metrics(report, conversations, rejected)
    write_json(conversations_path, conversations)
    write_json(rejected_path, rejected)
    write_json(report_path, report)
    remove_file_if_exists(partial_path)
    return conversations, rejected, report


def count_accepted_case_facts(accepted_cases: list[AcceptedCase]) -> int:
    return sum(len(accepted.case.facts) for accepted in accepted_cases)


def conversation_artifacts_match_cases(
    accepted_cases: list[AcceptedCase],
    conversations: list[ConversationSession],
    rejected_conversations: list[RejectedConversation],
) -> bool:
    expected_fact_keys = {
        (accepted.case.case_id, fact.fact_id)
        for accepted in accepted_cases
        for fact in accepted.case.facts
    }
    accepted_fact_keys = {(conversation.case_id, conversation.fact_id) for conversation in conversations}
    rejected_fact_keys = {
        (rejected.conversation.case_id, rejected.conversation.fact_id)
        for rejected in rejected_conversations
    }
    return not (accepted_fact_keys & rejected_fact_keys) and accepted_fact_keys | rejected_fact_keys == expected_fact_keys


def add_conversation_type_metrics(
    report: dict[str, Any],
    accepted_conversations: list[ConversationSession],
    rejected_conversations: list[RejectedConversation],
) -> dict[str, Any]:
    rejected_sessions = [item.conversation for item in rejected_conversations]
    all_sessions = [*accepted_conversations, *rejected_sessions]
    preferred_used_count = sum(1 for session in all_sessions if session.preferred_conversation_type_used is True)
    fallback_sessions = [
        session
        for session in all_sessions
        if session.preferred_conversation_type and session.preferred_conversation_type_used is False
    ]
    merged = dict(report)
    merged["conversation_type_metrics"] = {
        "preferred_conversation_type_distribution": count_conversation_attr(all_sessions, "preferred_conversation_type"),
        "selected_conversation_type_distribution": count_conversation_attr(all_sessions, "selected_conversation_type"),
        "accepted_selected_conversation_type_distribution": count_conversation_attr(
            accepted_conversations,
            "selected_conversation_type",
        ),
        "rejected_selected_conversation_type_distribution": count_conversation_attr(
            rejected_sessions,
            "selected_conversation_type",
        ),
        "preferred_conversation_type_used": preferred_used_count,
        "preferred_conversation_type_fallbacks": len(fallback_sessions),
        "preferred_conversation_type_used_rate": preferred_used_count / len(all_sessions) if all_sessions else 0.0,
        "fallback_selected_conversation_type_counts": count_fallback_conversation_type_pairs(fallback_sessions),
    }
    return merged


def count_conversation_attr(sessions: list[ConversationSession], attr: str) -> dict[str, int]:
    return dict(sorted(Counter(conversation_type_value(getattr(session, attr, None)) for session in sessions if getattr(session, attr, None)).items()))


def count_fallback_conversation_type_pairs(sessions: list[ConversationSession]) -> dict[str, int]:
    counts = Counter(
        f"{conversation_type_value(session.preferred_conversation_type)}->{conversation_type_value(session.selected_conversation_type)}"
        for session in sessions
    )
    return dict(sorted(counts.items()))


def conversation_type_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def load_or_prune_case_conversation_coverage(
    args: argparse.Namespace,
    run_dir: Path,
    accepted_cases: list[AcceptedCase],
    conversations: list[ConversationSession],
    rejected_conversations: list[RejectedConversation],
    logger: JsonlLogger,
) -> tuple[list[AcceptedCase], list[ConversationSession], list[dict[str, Any]], dict[str, Any]]:
    eligible_path = run_dir / "qa_eligible_cases.json"
    qa_conversations_path = run_dir / "qa_conversations.json"
    dropped_path = run_dir / "dropped_cases_after_conversation.json"
    if args.resume and eligible_path.exists() and qa_conversations_path.exists() and dropped_path.exists():
        qa_eligible_cases = read_json_models(eligible_path, AcceptedCase)
        qa_conversations = read_json_models(qa_conversations_path, ConversationSession)
        dropped_cases = read_json(dropped_path)
        if pruning_artifacts_match_inputs(accepted_cases, conversations, qa_eligible_cases, qa_conversations, dropped_cases):
            report = build_case_conversation_pruning_report(accepted_cases, conversations, qa_eligible_cases, qa_conversations, dropped_cases)
            logger.info(
                "case_conversation_pruning",
                "resume_existing_pruning",
                counts=report,
            )
            return qa_eligible_cases, qa_conversations, dropped_cases, report
        logger.info(
            "case_conversation_pruning",
            "resume_pruning_mismatch_rebuild",
            counts={
                "accepted_cases": len(accepted_cases),
                "conversations": len(conversations),
                "existing_qa_eligible_cases": len(qa_eligible_cases),
                "existing_qa_conversations": len(qa_conversations),
            },
        )
    if args.resume and persona_artifact_exists(run_dir, "qa_eligible_cases", qa_mode=QAMode.QUESTION):
        qa_eligible_cases = aggregate_persona_list_models(run_dir, "qa_eligible_cases", AcceptedCase, qa_mode=QAMode.QUESTION)
        qa_conversations = aggregate_persona_list_models(run_dir, "qa_conversations", ConversationSession, qa_mode=QAMode.QUESTION)
        dropped_cases = aggregate_persona_list_payloads(run_dir, "dropped_cases_after_conversation", qa_mode=QAMode.QUESTION)
        if pruning_artifacts_match_inputs(accepted_cases, conversations, qa_eligible_cases, qa_conversations, dropped_cases):
            report = build_case_conversation_pruning_report(accepted_cases, conversations, qa_eligible_cases, qa_conversations, dropped_cases)
            logger.info(
                "case_conversation_pruning",
                "resume_existing_persona_pruning",
                counts=report,
            )
            return qa_eligible_cases, qa_conversations, dropped_cases, report

    qa_eligible_cases, qa_conversations, dropped_cases, report = prune_cases_without_full_conversation_coverage(
        accepted_cases,
        conversations,
        rejected_conversations,
    )
    write_json(eligible_path, qa_eligible_cases)
    write_json(qa_conversations_path, qa_conversations)
    write_json(dropped_path, dropped_cases)
    return qa_eligible_cases, qa_conversations, dropped_cases, report


def prune_cases_without_full_conversation_coverage(
    accepted_cases: list[AcceptedCase],
    conversations: list[ConversationSession],
    rejected_conversations: list[RejectedConversation],
) -> tuple[list[AcceptedCase], list[ConversationSession], list[dict[str, Any]], dict[str, Any]]:
    accepted_fact_ids_by_case: dict[str, set[str]] = {}
    accepted_conversation_ids_by_case: dict[str, list[str]] = {}
    for conversation in conversations:
        accepted_fact_ids_by_case.setdefault(conversation.case_id, set()).add(conversation.fact_id)
        accepted_conversation_ids_by_case.setdefault(conversation.case_id, []).append(conversation.conversation_id)

    rejected_by_case: dict[str, list[RejectedConversation]] = {}
    for rejected in rejected_conversations:
        rejected_by_case.setdefault(rejected.conversation.case_id, []).append(rejected)

    qa_eligible_cases: list[AcceptedCase] = []
    dropped_cases: list[dict[str, Any]] = []
    for accepted in accepted_cases:
        case = accepted.case
        expected_fact_ids = [fact.fact_id for fact in case.facts]
        covered_fact_ids = accepted_fact_ids_by_case.get(case.case_id, set())
        missing_fact_ids = [fact_id for fact_id in expected_fact_ids if fact_id not in covered_fact_ids]
        if not missing_fact_ids:
            qa_eligible_cases.append(accepted)
            continue

        rejected_for_case = rejected_by_case.get(case.case_id, [])
        dropped_cases.append(
            {
                "case_id": case.case_id,
                "persona_id": case.persona_id,
                "topic_preference": case.topic_preference,
                "relation_type": case.relation_type,
                "relation_subtype": case.relation_subtype,
                "reason": "missing accepted conversation coverage for one or more facts",
                "missing_fact_ids": missing_fact_ids,
                "accepted_conversation_ids": accepted_conversation_ids_by_case.get(case.case_id, []),
                "rejected_conversation_ids": [
                    item.conversation.conversation_id
                    for item in rejected_for_case
                    if item.conversation.fact_id in missing_fact_ids
                ],
                "rejected_conversations": [
                    {
                        "conversation_id": item.conversation.conversation_id,
                        "fact_id": item.conversation.fact_id,
                        "reason": item.filter_decision.reason,
                        "reject_categories": item.filter_decision.reject_categories,
                    }
                    for item in rejected_for_case
                    if item.conversation.fact_id in missing_fact_ids
                ],
            }
        )

    eligible_case_ids = {item.case.case_id for item in qa_eligible_cases}
    qa_conversations = [conversation for conversation in conversations if conversation.case_id in eligible_case_ids]
    report = build_case_conversation_pruning_report(
        accepted_cases,
        conversations,
        qa_eligible_cases,
        qa_conversations,
        dropped_cases,
    )
    return qa_eligible_cases, qa_conversations, dropped_cases, report


def build_case_conversation_pruning_report(
    accepted_cases: list[AcceptedCase],
    conversations: list[ConversationSession],
    qa_eligible_cases: list[AcceptedCase],
    qa_conversations: list[ConversationSession],
    dropped_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "accepted_cases_before_pruning": len(accepted_cases),
        "accepted_conversations_before_pruning": len(conversations),
        "qa_eligible_cases": len(qa_eligible_cases),
        "qa_conversations": len(qa_conversations),
        "dropped_cases_after_conversation": len(dropped_cases),
        "dropped_accepted_conversations_after_case_pruning": len(conversations) - len(qa_conversations),
        "missing_fact_count": sum(len(item.get("missing_fact_ids", [])) for item in dropped_cases),
    }


def pruning_artifacts_match_inputs(
    accepted_cases: list[AcceptedCase],
    conversations: list[ConversationSession],
    qa_eligible_cases: list[AcceptedCase],
    qa_conversations: list[ConversationSession],
    dropped_cases: Any,
) -> bool:
    if not isinstance(dropped_cases, list) or not all(isinstance(item, dict) for item in dropped_cases):
        return False

    accepted_case_ids = {item.case.case_id for item in accepted_cases}
    eligible_case_ids = {item.case.case_id for item in qa_eligible_cases}
    dropped_case_ids = {str(item.get("case_id")) for item in dropped_cases}
    if eligible_case_ids & dropped_case_ids:
        return False
    if eligible_case_ids | dropped_case_ids != accepted_case_ids:
        return False

    conversation_ids = {item.conversation_id for item in conversations}
    qa_conversation_ids = {item.conversation_id for item in qa_conversations}
    if not qa_conversation_ids <= conversation_ids:
        return False
    if any(item.case_id not in eligible_case_ids for item in qa_conversations):
        return False

    covered_fact_ids_by_case: dict[str, set[str]] = {}
    for conversation in qa_conversations:
        covered_fact_ids_by_case.setdefault(conversation.case_id, set()).add(conversation.fact_id)
    for accepted in qa_eligible_cases:
        expected_fact_ids = {fact.fact_id for fact in accepted.case.facts}
        if not expected_fact_ids <= covered_fact_ids_by_case.get(accepted.case.case_id, set()):
            return False
    return True


def write_conversation_pruning_report(run_dir: Path, report: dict[str, Any], pruning_report: dict[str, Any]) -> dict[str, Any]:
    merged = dict(report)
    merged["case_pruning"] = pruning_report
    write_json(run_dir / "conversation_filter_report.json", merged)
    return merged


def load_or_build_sessions(
    args: argparse.Namespace,
    run_dir: Path,
    conversations: list[ConversationSession],
    logger: JsonlLogger | NullLogger | None = None,
) -> list[TimestampedSession]:
    sessions_path = run_dir / "sessions.json"
    if args.resume and sessions_path.exists():
        sessions = read_json_models(sessions_path, TimestampedSession)
        if session_artifacts_match_conversations(sessions, conversations):
            return sessions
        if logger is not None:
            logger.info(
                "session_timeline",
                "resume_sessions_mismatch_rebuild",
                counts={"existing_sessions": len(sessions), "expected_sessions": len(conversations)},
            )
    if args.resume and persona_artifact_exists(run_dir, "sessions", qa_mode=QAMode.QUESTION):
        sessions = aggregate_persona_list_models(run_dir, "sessions", TimestampedSession, qa_mode=QAMode.QUESTION)
        if session_artifacts_match_conversations(sessions, conversations):
            return sessions
        if logger is not None:
            logger.info(
                "session_timeline",
                "resume_persona_sessions_mismatch_rebuild",
                counts={"existing_sessions": len(sessions), "expected_sessions": len(conversations)},
            )
    sessions = build_timestamped_sessions(conversations, seed=args.seed)
    write_json(sessions_path, sessions)
    return sessions


def load_or_generate_case_qa(
    args: argparse.Namespace,
    run_dir: Path,
    accepted_cases: list[AcceptedCase],
    conversations: list[ConversationSession],
    personas: dict[str, SanitizedPersonaProfile],
    generation_client: Any,
    config: AppConfig,
    logger: JsonlLogger,
) -> list[CaseQA]:
    qa_mode = getattr(args, "qa_mode", QAMode.QUESTION.value)
    generated_path = mode_scoped_run_artifact_path(run_dir, "generated_case_qa", qa_mode=qa_mode)
    partial_path = existing_run_artifact_path(run_dir, "generated_case_qa_partial", qa_mode=qa_mode)
    legacy_generated_path = legacy_run_artifact_path(run_dir, "generated_case_qa")
    legacy_partial_path = legacy_run_artifact_path(run_dir, "generated_case_qa_partial")
    existing_generated_path = existing_run_artifact_path(run_dir, "generated_case_qa", qa_mode=qa_mode)
    if args.resume and existing_generated_path.exists():
        case_qa = read_json_models(existing_generated_path, CaseQA)
        if case_qa_artifacts_match_inputs(
            case_qa,
            accepted_cases,
            conversations,
            qa_mode=qa_mode,
            question_count=args.qa_question_count,
        ):
            logger.info("case_qa_generation", "resume_existing_case_qa", counts={"generated_case_qa": len(case_qa)})
            return case_qa
        logger.info(
            "case_qa_generation",
            "resume_case_qa_mismatch_rebuild",
            counts={"existing_case_qa": len(case_qa), "expected_case_qa": len(accepted_cases)},
        )
    if args.resume and persona_artifact_exists(run_dir, "generated_case_qa", qa_mode=qa_mode):
        case_qa = aggregate_persona_list_models(run_dir, "generated_case_qa", CaseQA, qa_mode=qa_mode)
        if case_qa and case_qa_artifacts_match_inputs(
            case_qa,
            accepted_cases,
            conversations,
            qa_mode=qa_mode,
            question_count=args.qa_question_count,
        ):
            logger.info("case_qa_generation", "resume_existing_persona_case_qa", counts={"generated_case_qa": len(case_qa)})
            return case_qa

    case_qa = generate_case_qa(
        accepted_cases,
        conversations,
        personas,
        generation_client,
        generation_model=config.generation_model,
        concurrency=qa_concurrency(args),
        checkpoint_path=partial_path,
        logger=logger,
        question_count=args.qa_question_count,
        qa_mode=qa_mode,
    )
    write_json(generated_path, case_qa)
    write_json(legacy_generated_path, case_qa)
    remove_file_if_exists(partial_path)
    remove_file_if_exists(legacy_partial_path)
    return case_qa


def load_or_filter_case_qa(
    args: argparse.Namespace,
    run_dir: Path,
    generated_case_qa: list[CaseQA],
    accepted_cases: list[AcceptedCase],
    conversations: list[ConversationSession],
    personas: dict[str, SanitizedPersonaProfile],
    filter_client: Any,
    config: AppConfig,
    logger: JsonlLogger,
) -> tuple[list[AcceptedCaseQA], list[RejectedCaseQA], dict[str, Any]]:
    qa_mode = getattr(args, "qa_mode", QAMode.QUESTION.value)
    accepted_path = mode_scoped_run_artifact_path(run_dir, "accepted_case_qa", qa_mode=qa_mode)
    rejected_path = mode_scoped_run_artifact_path(run_dir, "rejected_case_qa", qa_mode=qa_mode)
    report_path = mode_scoped_run_artifact_path(run_dir, "qa_filter_report", qa_mode=qa_mode)
    legacy_accepted_path = legacy_run_artifact_path(run_dir, "accepted_case_qa")
    legacy_rejected_path = legacy_run_artifact_path(run_dir, "rejected_case_qa")
    legacy_report_path = legacy_run_artifact_path(run_dir, "qa_filter_report")
    existing_accepted_path = existing_run_artifact_path(run_dir, "accepted_case_qa", qa_mode=qa_mode)
    existing_rejected_path = existing_run_artifact_path(run_dir, "rejected_case_qa", qa_mode=qa_mode)
    existing_report_path = existing_run_artifact_path(run_dir, "qa_filter_report", qa_mode=qa_mode)
    if args.resume and existing_accepted_path.exists() and existing_rejected_path.exists() and existing_report_path.exists():
        accepted = read_json_models(existing_accepted_path, AcceptedCaseQA)
        rejected = read_json_models(existing_rejected_path, RejectedCaseQA)
        report = read_json(existing_report_path)
        if case_qa_filter_artifacts_match_generated(generated_case_qa, accepted, rejected):
            logger.info("case_qa_filter", "resume_existing_filter", counts={"accepted": len(accepted), "rejected": len(rejected)})
            return accepted, rejected, report
        logger.info(
            "case_qa_filter",
            "resume_filter_mismatch_rebuild",
            counts={
                "existing_case_qa": len({item.qa.qa_id for item in accepted} | {item.original_qa.qa_id for item in rejected}),
                "expected_case_qa": len(generated_case_qa),
            },
        )
    if args.resume and persona_artifact_exists(run_dir, "accepted_case_qa", qa_mode=qa_mode):
        accepted = aggregate_persona_list_models(run_dir, "accepted_case_qa", AcceptedCaseQA, qa_mode=qa_mode)
        rejected = aggregate_persona_list_models(run_dir, "rejected_case_qa", RejectedCaseQA, qa_mode=qa_mode)
        if accepted or rejected:
            if case_qa_filter_artifacts_match_generated(generated_case_qa, accepted, rejected):
                report = build_aggregated_case_qa_filter_report(
                    generated_case_qa=generated_case_qa,
                    accepted_case_qa=accepted,
                    rejected_case_qa=rejected,
                    accepted_cases=accepted_cases,
                    conversations=conversations,
                    personas=personas,
                    filter_model=config.filter_model,
                )
                logger.info("case_qa_filter", "resume_existing_persona_filter", counts={"accepted": len(accepted), "rejected": len(rejected)})
                return accepted, rejected, report
            logger.info(
                "case_qa_filter",
                "resume_persona_filter_mismatch_rebuild",
                counts={
                    "existing_case_qa": len({item.qa.qa_id for item in accepted} | {item.original_qa.qa_id for item in rejected}),
                    "expected_case_qa": len(generated_case_qa),
                },
            )

    accepted, rejected, report = filter_case_qa(
        generated_case_qa,
        accepted_cases,
        conversations,
        personas,
        filter_client,
        filter_model=config.filter_model,
        concurrency=filter_concurrency(args),
        qa_mode=qa_mode,
    )
    write_json(accepted_path, accepted)
    write_json(rejected_path, rejected)
    write_json(report_path, report)
    write_json(legacy_accepted_path, accepted)
    write_json(legacy_rejected_path, rejected)
    write_json(legacy_report_path, report)
    return accepted, rejected, report


def apply_qa_question_sampling(
    args: argparse.Namespace,
    run_dir: Path,
    generated_case_qa: list[CaseQA],
    accepted_case_qa: list[AcceptedCaseQA],
    rejected_case_qa: list[RejectedCaseQA],
    accepted_cases: list[AcceptedCase],
    conversations: list[ConversationSession],
    personas: dict[str, SanitizedPersonaProfile],
    config: AppConfig,
) -> tuple[list[AcceptedCaseQA], dict[str, Any], dict[str, Any]]:
    qa_mode = getattr(args, "qa_mode", QAMode.QUESTION.value)
    sampled_case_qa, sampling_report = sample_accepted_case_qa_questions(
        accepted_case_qa,
        seed=args.seed,
    )
    qa_filter_report = build_aggregated_case_qa_filter_report(
        generated_case_qa=generated_case_qa,
        accepted_case_qa=sampled_case_qa,
        rejected_case_qa=rejected_case_qa,
        accepted_cases=accepted_cases,
        conversations=conversations,
        personas=personas,
        filter_model=config.filter_model,
    )
    write_case_qa_sampling_artifacts(
        run_dir,
        qa_mode=qa_mode,
        accepted_case_qa=sampled_case_qa,
        qa_filter_report=qa_filter_report,
        sampling_report=sampling_report,
    )
    return sampled_case_qa, qa_filter_report, sampling_report


def sample_accepted_case_qa_questions(
    accepted_case_qa: list[AcceptedCaseQA],
    *,
    seed: int,
) -> tuple[list[AcceptedCaseQA], dict[str, Any]]:
    sampled: list[AcceptedCaseQA] = []
    selected_questions: list[dict[str, Any]] = []
    question_counts_before_by_relation = Counter()
    question_counts_after_by_relation = Counter()
    case_counts_by_relation = Counter()
    sampled_case_counts_by_relation = Counter()

    for item in accepted_case_qa:
        relation_type = schema_value(item.qa.relation_type)
        relation_subtype = schema_value(item.qa.relation_subtype)
        existing_sampling_rejected_questions = [
            rejected_question
            for rejected_question in item.filter_decision.rejected_questions
            if QA_RANDOM_SAMPLING_REJECT_CATEGORY in (rejected_question.get("reject_categories") or [])
        ]
        current_question_count = len(item.qa.questions)
        question_count_before = current_question_count + len(existing_sampling_rejected_questions)
        case_counts_by_relation[relation_type] += 1
        question_counts_before_by_relation[relation_type] += question_count_before

        if relation_type in QA_RANDOM_SAMPLING_RELATION_TYPES and current_question_count > 1:
            selected_index = seeded_question_sample_index(seed, item.qa.qa_id, current_question_count)
            selected_question = item.qa.questions[selected_index]
            rejected_questions = [
                {
                    "question_id": question.question_id,
                    "reason": (
                        "Random QA sampling kept one question for final complementary/contradictory data."
                    ),
                    "reject_categories": [QA_RANDOM_SAMPLING_REJECT_CATEGORY],
                }
                for index, question in enumerate(item.qa.questions)
                if index != selected_index
            ]
            sampling_rejected_questions = [*existing_sampling_rejected_questions, *rejected_questions]
            filtered_qa, decision = build_qa_filter_decision(
                item.qa,
                rejected_questions=[*item.filter_decision.rejected_questions, *rejected_questions],
                removed_answers=list(item.filter_decision.removed_answers),
                filter_model=item.filter_decision.filter_model,
                stream_completed=item.filter_decision.stream_completed,
            )
            if filtered_qa is None:
                raise ValueError(f"random QA sampling removed all questions for {item.qa.qa_id}")
            sampled_item = AcceptedCaseQA(qa=filtered_qa, filter_decision=decision)
            sampled.append(sampled_item)
            sampled_case_counts_by_relation[relation_type] += 1
            selected_questions.append(
                {
                    "qa_id": item.qa.qa_id,
                    "persona_id": item.qa.persona_id,
                    "case_id": item.qa.case_id,
                    "relation_type": relation_type,
                    "relation_subtype": relation_subtype,
                    "questions_before": question_count_before,
                    "questions_after": len(filtered_qa.questions),
                    "selected_index": selected_index,
                    "selected_question_id": selected_question.question_id,
                    "removed_question_ids": [entry["question_id"] for entry in sampling_rejected_questions],
                }
            )
        else:
            sampled.append(item)
            if relation_type in QA_RANDOM_SAMPLING_RELATION_TYPES and existing_sampling_rejected_questions:
                sampled_case_counts_by_relation[relation_type] += 1
                selected_questions.append(
                    {
                        "qa_id": item.qa.qa_id,
                        "persona_id": item.qa.persona_id,
                        "case_id": item.qa.case_id,
                        "relation_type": relation_type,
                        "relation_subtype": relation_subtype,
                        "questions_before": question_count_before,
                        "questions_after": current_question_count,
                        "selected_index": None,
                        "selected_question_id": item.qa.questions[0].question_id if item.qa.questions else None,
                        "removed_question_ids": [
                            str(entry.get("question_id") or "")
                            for entry in existing_sampling_rejected_questions
                        ],
                        "already_sampled": True,
                    }
                )

        question_counts_after_by_relation[relation_type] += len(sampled[-1].qa.questions)

    questions_before = sum(question_counts_before_by_relation.values())
    questions_after = sum(question_counts_after_by_relation.values())
    target_questions_before = sum(
        question_counts_before_by_relation[relation_type]
        for relation_type in QA_RANDOM_SAMPLING_RELATION_TYPES
    )
    target_questions_after = sum(
        question_counts_after_by_relation[relation_type]
        for relation_type in QA_RANDOM_SAMPLING_RELATION_TYPES
    )
    report = {
        "schema": "user_related_qa_question_sampling_report_v1",
        "selection_seed": seed,
        "target_relation_types": sorted(QA_RANDOM_SAMPLING_RELATION_TYPES),
        "counts": {
            "accepted_case_qa": len(accepted_case_qa),
            "target_case_qa": sum(case_counts_by_relation[relation_type] for relation_type in QA_RANDOM_SAMPLING_RELATION_TYPES),
            "sampled_case_qa": sum(sampled_case_counts_by_relation.values()),
            "questions_before": questions_before,
            "questions_after": questions_after,
            "questions_removed": questions_before - questions_after,
            "target_questions_before": target_questions_before,
            "target_questions_after": target_questions_after,
            "target_questions_removed": target_questions_before - target_questions_after,
        },
        "case_counts_by_relation_type": dict(sorted(case_counts_by_relation.items())),
        "sampled_case_counts_by_relation_type": dict(sorted(sampled_case_counts_by_relation.items())),
        "question_counts_before_by_relation_type": dict(sorted(question_counts_before_by_relation.items())),
        "question_counts_after_by_relation_type": dict(sorted(question_counts_after_by_relation.items())),
        "selected_questions": selected_questions,
    }
    return sampled, report


def seeded_question_sample_index(seed: int, qa_id: str, question_count: int) -> int:
    if question_count < 1:
        raise ValueError("question_count must be positive")
    rng = random.Random(f"{seed}:qa-question-sampling:{qa_id}")
    return rng.randrange(question_count)


def schema_value(value: Any) -> str:
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def write_case_qa_sampling_artifacts(
    run_dir: Path,
    *,
    qa_mode: str | QAMode,
    accepted_case_qa: list[AcceptedCaseQA],
    qa_filter_report: dict[str, Any],
    sampling_report: dict[str, Any],
) -> None:
    accepted_path = mode_scoped_run_artifact_path(run_dir, "accepted_case_qa", qa_mode=qa_mode)
    qa_filter_report_path = mode_scoped_run_artifact_path(run_dir, "qa_filter_report", qa_mode=qa_mode)
    sampling_report_path = mode_scoped_run_artifact_path(run_dir, "qa_question_sampling_report", qa_mode=qa_mode)
    write_json(accepted_path, accepted_case_qa)
    write_json(qa_filter_report_path, qa_filter_report)
    write_json(sampling_report_path, sampling_report)
    write_json(legacy_run_artifact_path(run_dir, "accepted_case_qa"), accepted_case_qa)
    write_json(legacy_run_artifact_path(run_dir, "qa_filter_report"), qa_filter_report)
    write_json(legacy_run_artifact_path(run_dir, "qa_question_sampling_report"), sampling_report)


def load_or_build_evaluation_instances(
    args: argparse.Namespace,
    run_dir: Path,
    accepted_case_qa: list[AcceptedCaseQA],
    accepted_cases: list[AcceptedCase],
    conversations: list[ConversationSession],
    personas: dict[str, SanitizedPersonaProfile],
    logger: JsonlLogger | NullLogger | None = None,
) -> list[EvaluationInstance]:
    qa_mode = getattr(args, "qa_mode", QAMode.QUESTION.value)
    path = mode_scoped_run_artifact_path(run_dir, "evaluation_instances", qa_mode=qa_mode)
    legacy_path = legacy_run_artifact_path(run_dir, "evaluation_instances")
    existing_path = existing_run_artifact_path(run_dir, "evaluation_instances", qa_mode=qa_mode)
    if args.resume and existing_path.exists():
        instances = read_json_models(existing_path, EvaluationInstance)
        if evaluation_instances_match_accepted_case_qa(instances, accepted_case_qa):
            return instances
        if logger is not None:
            logger.info(
                "evaluation",
                "resume_evaluation_instances_mismatch_rebuild",
                counts={
                    "existing_instances": len(instances),
                    "expected_instances": sum(len(item.qa.questions) for item in accepted_case_qa),
                },
            )
    if args.resume and persona_artifact_exists(run_dir, "evaluation_instances", qa_mode=qa_mode):
        instances = aggregate_persona_list_models(run_dir, "evaluation_instances", EvaluationInstance, qa_mode=qa_mode)
        if evaluation_instances_match_accepted_case_qa(instances, accepted_case_qa):
            return instances
        if logger is not None:
            logger.info(
                "evaluation",
                "resume_persona_evaluation_instances_mismatch_rebuild",
                counts={
                    "existing_instances": len(instances),
                    "expected_instances": sum(len(item.qa.questions) for item in accepted_case_qa),
                },
            )
    instances = build_evaluation_instances(accepted_case_qa, accepted_cases, conversations, personas)
    write_json(path, instances)
    write_json(legacy_path, instances)
    return instances


def evaluation_instances_match_accepted_case_qa(
    instances: list[EvaluationInstance],
    accepted_case_qa: list[AcceptedCaseQA],
) -> bool:
    expected = {}
    for item in accepted_case_qa:
        qa = item.qa
        for question in qa.questions:
            expected[(qa.qa_id, question.question_id)] = {
                "persona_id": qa.persona_id,
                "case_id": qa.case_id,
                "qa_mode": qa.qa_mode,
                "session_ids": list(qa.session_ids),
                "query": question.question,
                "task_form": question.task_form,
                "correct_answers": [answer.model_dump(mode="json") for answer in question.correct_answers],
                "incorrect_answers": [answer.model_dump(mode="json") for answer in question.incorrect_answers],
            }
    existing = {
        (item.qa_id, item.question_id): {
            "persona_id": item.persona_id,
            "case_id": item.case_id,
            "qa_mode": item.qa_mode,
            "session_ids": [session.conversation_id for session in item.sessions],
            "query": item.query,
            "task_form": item.task_form,
            "correct_answers": [answer.model_dump(mode="json") for answer in item.correct_answers],
            "incorrect_answers": [answer.model_dump(mode="json") for answer in item.incorrect_answers],
        }
        for item in instances
    }
    return existing == expected


def session_artifacts_match_conversations(
    sessions: list[TimestampedSession],
    conversations: list[ConversationSession],
) -> bool:
    session_ids = {session.conversation_id for session in sessions}
    conversation_ids = {conversation.conversation_id for conversation in conversations}
    return len(sessions) == len(conversations) and session_ids == conversation_ids


def case_qa_artifacts_match_inputs(
    case_qa: list[CaseQA],
    accepted_cases: list[AcceptedCase],
    conversations: list[ConversationSession],
    *,
    qa_mode: str,
    question_count: int,
) -> bool:
    expected_case_ids = {accepted.case.case_id for accepted in accepted_cases}
    existing_case_ids = {item.case_id for item in case_qa}
    if existing_case_ids != expected_case_ids:
        return False

    conversation_ids_by_case: dict[str, set[str]] = {}
    for conversation in conversations:
        conversation_ids_by_case.setdefault(conversation.case_id, set()).add(conversation.conversation_id)
    return all(
        item.qa_mode == qa_mode
        and len(item.questions) == question_count
        and set(item.session_ids) == conversation_ids_by_case.get(item.case_id, set())
        for item in case_qa
    )


def case_qa_filter_artifacts_match_generated(
    generated_case_qa: list[CaseQA],
    accepted_case_qa: list[AcceptedCaseQA],
    rejected_case_qa: list[RejectedCaseQA],
) -> bool:
    generated_by_id = {item.qa_id: item for item in generated_case_qa}
    existing_qa_ids = {item.qa.qa_id for item in accepted_case_qa} | {item.original_qa.qa_id for item in rejected_case_qa}
    if existing_qa_ids != set(generated_by_id):
        return False

    for item in rejected_case_qa:
        generated = generated_by_id.get(item.original_qa.qa_id)
        if generated is None or item.original_qa.model_dump(mode="json") != generated.model_dump(mode="json"):
            return False

    for item in accepted_case_qa:
        generated = generated_by_id.get(item.qa.qa_id)
        if generated is None:
            return False
        filtered_qa, decision = apply_existing_qa_filter_decision(generated, item.filter_decision)
        if not decision.accepted or filtered_qa is None:
            return False
        if filtered_qa.model_dump(mode="json") != item.qa.model_dump(mode="json"):
            return False
    return True


def format_relation_pair_counter(counts: Counter[tuple[str, str]]) -> dict[str, int]:
    return {f"{relation_type}/{relation_subtype}": count for (relation_type, relation_subtype), count in sorted(counts.items())}


def build_case_relation_plan_report_from_items(plan_items: list[CaseRelationPlanItem]) -> dict[str, Any]:
    pair_counts = Counter((str(item.relation_type), str(item.relation_subtype)) for item in plan_items)
    type_counts = Counter(str(item.relation_type) for item in plan_items)
    return {
        "personas": 1 if plan_items else 0,
        "planned_relation_items": len(plan_items),
        "target_relation_type_distribution": dict(sorted(type_counts.items())),
        "target_relation_subtype_distribution": format_relation_pair_counter(pair_counts),
        "planned_relation_type_distribution": dict(sorted(type_counts.items())),
        "planned_relation_subtype_distribution": dict(sorted(Counter(str(item.relation_subtype) for item in plan_items).items())),
        "planned_relation_pair_distribution": format_relation_pair_counter(pair_counts),
    }


def build_aggregated_case_filter_report(
    *,
    generated_cases: list[MemoryCase],
    accepted_cases: list[AcceptedCase],
    rejected_cases: list[RejectedCase],
    filter_model: str,
) -> dict[str, Any]:
    accepted_by_id = {item.case.case_id: item.filter_decision for item in accepted_cases}
    rejected_by_id = {item.case.case_id: item.filter_decision for item in rejected_cases}
    decisions = [
        accepted_by_id.get(case.case_id) or rejected_by_id[case.case_id]
        for case in generated_cases
    ]
    report = filter_report_from_decisions(decisions)
    report["filter_model"] = filter_model
    report["stream_completed"] = all(decision.stream_completed for decision in decisions)
    report["topic_filter_units"] = len({(case.persona_id, case.topic_preference) for case in generated_cases})
    generated_cases_by_persona = Counter(case.persona_id for case in generated_cases)
    report["persona_filter_units"] = sum(1 for count in generated_cases_by_persona.values() if count > 1)
    report["persona_filter_errors"] = []
    return report


def build_aggregated_conversation_filter_report(
    *,
    accepted_cases: list[AcceptedCase],
    conversations: list[ConversationSession],
    rejected_conversations: list[RejectedConversation],
    personas: dict[str, SanitizedPersonaProfile],
    filter_model: str,
) -> dict[str, Any]:
    generated_conversations = [*conversations, *(item.conversation for item in rejected_conversations)]
    rejected_by_id = {item.conversation.conversation_id: item.filter_decision for item in rejected_conversations}
    decisions = [
        rejected_by_id.get(conversation.conversation_id)
        or build_conversation_filter_decision(
            conversation.conversation_id,
            {"accepted": True, "reason": "accepted", "reject_categories": []},
            filter_model=filter_model,
            stream_completed=True,
        )
        for conversation in generated_conversations
    ]
    batches = build_conversation_filter_batches(
        generated_conversations,
        accepted_cases,
        personas,
        batch_size=DEFAULT_CONVERSATION_FILTER_BATCH_SIZE,
    )
    dummy_results = [
        ConversationFilterBatchResult(
            batch_id=batch.batch_id,
            decisions=[],
            ignored_decisions=[],
            stream_completed=all(decision.stream_completed for decision in decisions),
        )
        for batch in batches
    ]
    report = build_conversation_filter_report(
        generated_conversations,
        decisions,
        dummy_results,
        filter_model,
        DEFAULT_CONVERSATION_FILTER_BATCH_SIZE,
    )
    report["local_validation_rejections"] = sum(
        1 for item in rejected_conversations if "local_validation" in item.filter_decision.reject_categories
    )
    return add_conversation_type_metrics(report, conversations, rejected_conversations)


def build_aggregated_case_qa_filter_report(
    *,
    generated_case_qa: list[CaseQA],
    accepted_case_qa: list[AcceptedCaseQA],
    rejected_case_qa: list[RejectedCaseQA],
    accepted_cases: list[AcceptedCase],
    conversations: list[ConversationSession],
    personas: dict[str, SanitizedPersonaProfile],
    filter_model: str,
) -> dict[str, Any]:
    accepted_by_id = {item.qa.qa_id: item.filter_decision for item in accepted_case_qa}
    rejected_by_id = {item.original_qa.qa_id: item.filter_decision for item in rejected_case_qa}
    decisions = [
        accepted_by_id.get(item.qa_id) or rejected_by_id[item.qa_id]
        for item in generated_case_qa
    ]
    batches = build_case_qa_filter_batches(
        generated_case_qa,
        accepted_cases,
        conversations,
        personas,
        batch_size=DEFAULT_CASE_QA_FILTER_BATCH_SIZE,
    )
    dummy_results = [
        CaseQAFilterBatchResult(
            batch_id=batch.batch_id,
            decisions=[],
            ignored_decisions=[],
            stream_completed=all(decision.stream_completed for decision in decisions),
        )
        for batch in batches
    ]
    return build_case_qa_filter_report(
        generated_case_qa,
        decisions,
        dummy_results,
        filter_model,
        DEFAULT_CASE_QA_FILTER_BATCH_SIZE,
    )


def build_persona_extra_artifacts(
    *,
    personas: dict[str, SanitizedPersonaProfile],
    topic_groups: list[TopicPreferenceGroup],
    case_relation_plan: list[CaseRelationPlanItem],
    generated_cases: list[MemoryCase],
    accepted_cases: list[AcceptedCase],
    rejected_cases: list[RejectedCase],
    filter_report: dict[str, Any],
    conversations: list[ConversationSession],
    qa_eligible_cases: list[AcceptedCase],
    qa_conversations: list[ConversationSession],
    rejected_conversations: list[RejectedConversation],
    dropped_cases_after_conversation: list[dict[str, Any]],
    conversation_filter_report: dict[str, Any],
    generated_case_qa: list[CaseQA],
    accepted_case_qa: list[AcceptedCaseQA],
    rejected_case_qa: list[RejectedCaseQA],
    qa_filter_report: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    extras_by_persona: dict[str, dict[str, Any]] = {}
    for persona_id in sorted(personas):
        persona_topic_groups = [item for item in topic_groups if item.persona_id == persona_id]
        persona_case_relation_plan = [item for item in case_relation_plan if item.persona_id == persona_id]
        persona_generated_cases = [item for item in generated_cases if item.persona_id == persona_id]
        persona_accepted_cases = [item for item in accepted_cases if item.case.persona_id == persona_id]
        persona_rejected_cases = [item for item in rejected_cases if item.case.persona_id == persona_id]
        persona_conversations = [item for item in conversations if item.persona_id == persona_id]
        persona_qa_eligible_cases = [item for item in qa_eligible_cases if item.case.persona_id == persona_id]
        persona_qa_conversations = [item for item in qa_conversations if item.persona_id == persona_id]
        persona_rejected_conversations = [
            item for item in rejected_conversations if item.conversation.persona_id == persona_id
        ]
        persona_dropped_cases = [item for item in dropped_cases_after_conversation if item.get("persona_id") == persona_id]
        persona_generated_case_qa = [item for item in generated_case_qa if item.persona_id == persona_id]
        persona_accepted_case_qa = [item for item in accepted_case_qa if item.qa.persona_id == persona_id]
        persona_rejected_case_qa = [item for item in rejected_case_qa if item.original_qa.persona_id == persona_id]

        persona_conversation_filter_report = build_aggregated_conversation_filter_report(
            accepted_cases=persona_accepted_cases,
            conversations=persona_conversations,
            rejected_conversations=persona_rejected_conversations,
            personas={persona_id: personas[persona_id]},
            filter_model=str(conversation_filter_report.get("filter_model") or ""),
        )
        persona_conversation_filter_report["case_pruning"] = build_case_conversation_pruning_report(
            persona_accepted_cases,
            persona_conversations,
            persona_qa_eligible_cases,
            persona_qa_conversations,
            persona_dropped_cases,
        )

        extras_by_persona[persona_id] = {
            "case_relation_plan": persona_case_relation_plan,
            "case_relation_plan_report": build_case_relation_plan_report_from_items(persona_case_relation_plan),
            "filter_report": build_aggregated_case_filter_report(
                generated_cases=persona_generated_cases,
                accepted_cases=persona_accepted_cases,
                rejected_cases=persona_rejected_cases,
                filter_model=str(filter_report.get("filter_model") or ""),
            ),
            "conversation_filter_report": persona_conversation_filter_report,
            "qa_filter_report": build_aggregated_case_qa_filter_report(
                generated_case_qa=persona_generated_case_qa,
                accepted_case_qa=persona_accepted_case_qa,
                rejected_case_qa=persona_rejected_case_qa,
                accepted_cases=persona_qa_eligible_cases,
                conversations=persona_qa_conversations,
                personas={persona_id: personas[persona_id]},
                filter_model=str(qa_filter_report.get("filter_model") or ""),
            ),
        }
    return extras_by_persona


def build_run_manifest(
    run_dir: Path,
    *,
    source_run: str | None = None,
    from_stage: str | None = None,
    to_stage: str | None = None,
) -> dict[str, Any]:
    export_dir = run_dir / "export"
    qa_modes = sorted(path.name for path in export_dir.iterdir() if path.is_dir()) if export_dir.exists() else []
    personas: list[dict[str, Any]] = []
    for persona_dir in iter_persona_dirs(run_dir):
        manifest_path = persona_dir / "manifest.json"
        persona_manifest = read_json(manifest_path) if manifest_path.exists() else {}
        persona_key = persona_dir.name
        personas.append(
            {
                "persona_key": persona_key,
                "persona_id": persona_manifest.get("persona_id"),
                "manifest": f"{persona_key}/manifest.json",
                "export_files": {
                    qa_mode: f"export/{qa_mode}/{persona_key}.json"
                    for qa_mode in qa_modes
                    if (export_dir / qa_mode / f"{persona_key}.json").exists()
                },
            }
        )
    return {
        "schema": "user_related_run_manifest_v1",
        "branched_from": source_run is not None,
        "source_run": source_run,
        "from_stage": from_stage,
        "to_stage": to_stage,
        "qa_modes": qa_modes,
        "personas": personas,
        "root_files": {
            "run_report": "run_report.json",
            "run_log": "run.log.jsonl",
            "export_manifests": {
                qa_mode: f"export/{qa_mode}/manifest.json"
                for qa_mode in qa_modes
                if (export_dir / qa_mode / "manifest.json").exists()
            },
        },
    }


def remove_run_intermediate_artifacts(run_dir: Path) -> None:
    for filename in ROOT_INTERMEDIATE_FILENAMES:
        remove_file_if_exists(run_dir / filename)
    for dirname in ROOT_INTERMEDIATE_DIRNAMES:
        remove_tree_if_exists(run_dir / dirname)
    export_dir = run_dir / "export"
    if export_dir.exists():
        for path in export_dir.iterdir():
            if path.is_file():
                path.unlink()


def write_persona_outputs(
    *,
    run_dir: Path,
    qa_mode: str,
    personas: dict[str, SanitizedPersonaProfile],
    topic_groups: list[TopicPreferenceGroup],
    extra_artifacts_by_persona: dict[str, dict[str, Any]] | None = None,
    generated_cases: list[MemoryCase],
    accepted_cases: list[AcceptedCase],
    qa_eligible_cases: list[AcceptedCase] | None = None,
    rejected_cases: list[RejectedCase],
    conversations: list[ConversationSession],
    qa_conversations: list[ConversationSession] | None = None,
    sessions: list[TimestampedSession],
    rejected_conversations: list[RejectedConversation],
    dropped_cases_after_conversation: list[dict[str, Any]] | None = None,
    generated_case_qa: list[CaseQA],
    accepted_case_qa: list[AcceptedCaseQA],
    rejected_case_qa: list[RejectedCaseQA],
    evaluation_instances: list[EvaluationInstance],
) -> None:
    qa_eligible_cases = qa_eligible_cases if qa_eligible_cases is not None else accepted_cases
    qa_conversations = qa_conversations if qa_conversations is not None else conversations
    dropped_cases_after_conversation = dropped_cases_after_conversation or []
    bundles = build_persona_output_bundles(
        personas=personas,
        topic_groups=topic_groups,
        extra_artifacts_by_persona=extra_artifacts_by_persona,
        generated_cases=generated_cases,
        accepted_cases=accepted_cases,
        qa_eligible_cases=qa_eligible_cases,
        rejected_cases=rejected_cases,
        conversations=conversations,
        qa_conversations=qa_conversations,
        sessions=sessions,
        rejected_conversations=rejected_conversations,
        dropped_cases_after_conversation=dropped_cases_after_conversation,
        generated_case_qa=generated_case_qa,
        accepted_case_qa=accepted_case_qa,
        rejected_case_qa=rejected_case_qa,
        evaluation_instances=evaluation_instances,
        qa_mode=qa_mode,
    )
    for bundle in bundles:
        persona_dir = run_dir / bundle.persona_key
        for stem, artifact in bundle.artifacts.items():
            path = persona_dir / persona_artifact_path(stem, qa_mode=qa_mode)
            write_json(path, artifact)
        write_json(run_dir / "export" / qa_mode_value(qa_mode) / f"{bundle.persona_key}.json", bundle.export_payload, sort_keys=False)
    manifest = build_export_manifest(bundles, qa_mode=qa_mode)
    write_json(run_dir / "export" / qa_mode_value(qa_mode) / "manifest.json", manifest)


def merge_local_conversation_rejections(
    report: dict[str, Any],
    local_rejected: list[RejectedConversation],
) -> dict[str, Any]:
    if not local_rejected:
        report["local_validation_rejections"] = 0
        return report
    merged = dict(report)
    accepted = int(merged.get("accepted_conversations", 0))
    total = int(merged.get("total_generated_conversations", 0)) + len(local_rejected)
    rejected = int(merged.get("rejected_conversations", 0)) + len(local_rejected)
    merged["total_generated_conversations"] = total
    merged["rejected_conversations"] = rejected
    merged["retention_rate"] = accepted / total if total else 0.0
    merged["local_validation_rejections"] = len(local_rejected)
    details = list(merged.get("rejected_conversation_details", []))
    details.extend(
        {
            "conversation_id": item.conversation.conversation_id,
            "case_id": item.conversation.case_id,
            "fact_id": item.conversation.fact_id,
            "reason": item.filter_decision.reason,
            "reject_categories": item.filter_decision.reject_categories,
        }
        for item in local_rejected
    )
    merged["rejected_conversation_details"] = details
    return merged


def organize_persona_output_dirs(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    if not root.exists():
        raise FileNotFoundError(f"run directory does not exist: {root}")
    moved = 0
    updated_manifests = 0
    persona_dirs = sorted(path for path in root.glob("persona_*") if path.is_dir())
    for persona_dir in persona_dirs:
        artifact_files: dict[str, str] = {}
        manifest_path = persona_dir / "manifest.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        if isinstance(manifest.get("artifact_files"), dict):
            artifact_files.update({str(key): str(value) for key, value in manifest["artifact_files"].items()})

        for stem, relative_path in LEGACY_PERSONA_ARTIFACT_PATHS.items():
            legacy_path = persona_dir / f"{stem}.json"
            target_path = persona_dir / relative_path
            if legacy_path.exists() and legacy_path != target_path:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                legacy_path.replace(target_path)
                moved += 1
            if target_path.exists():
                artifact_files[stem] = relative_path.as_posix()

        if artifact_files:
            manifest["artifact_files"] = {
                stem: artifact_files[stem]
                for stem in LEGACY_PERSONA_ARTIFACT_PATHS
                if stem in artifact_files
            }
            write_json(manifest_path, manifest)
            updated_manifests += 1

    return {
        "run_dir": str(root),
        "counts": {
            "persona_dirs": len(persona_dirs),
            "moved": moved,
            "updated_manifests": updated_manifests,
        },
    }


def run_dry_run(args: argparse.Namespace, config: AppConfig, run_id: str) -> int:
    logger = JsonlLogger(run_id, level=args.log_level, secrets=config.secrets)
    try:
        selected_persona_ids = selected_persona_ids_for_run(args)
        if is_persona_cache_dir(args.input_dir):
            profiles, groups = load_persona_caches(
                args.input_dir,
                None if selected_persona_ids is not None else args.limit,
                persona_ids=selected_persona_ids,
            )
            groups = limit_topic_group_preferences(groups, args.preference_limit, random.Random(args.seed))
            counts = {
                "source_records": len(profiles),
                "topics": len(groups),
                "source_preferences": count_topic_group_preferences(groups),
            }
        else:
            source_records = load_source_personas(
                args.input_dir,
                None if selected_persona_ids is not None else args.limit,
                exclude_dirs=[args.output_dir],
                persona_ids=selected_persona_ids,
            )
            if not source_records:
                raise ValueError(f"No source records discovered in {args.input_dir}")
            counts = {"source_records": len(source_records), "preference_limit_per_persona": args.preference_limit or 0}
        logger.info(
            "dry_run",
            "dry_run_complete",
            config=masked_config_summary(config),
            counts=counts,
            output_dir=str(args.output_dir),
            config_path=str(args.config.resolve()) if getattr(args, "config", None) is not None else None,
            persona_ids=list(selected_persona_ids or []),
            configured_persona_ids=list(getattr(args, "persona_ids", None) or []),
            persona_selection_limit_applied=persona_selection_limit_applied(args),
            persona_limit=args.limit if getattr(args, "limit_explicit", False) else None,
            qa_mode=args.qa_mode,
        )
        return 0
    except Exception as exc:
        logger.error("dry_run", "dry_run_failed", error=str(exc))
        return 1
    finally:
        logger.close()


def make_generation_client(config: AppConfig, logger: JsonlLogger | NullLogger, args: argparse.Namespace) -> OpenAIStreamingClient:
    return OpenAIStreamingClient(
        base_url=config.generation_base_url or "",
        api_key=config.generation_api_key or "",
        model=config.generation_model,
        reasoning_effort=config.generation_reasoning_effort,
        logger=logger,
        max_retries=args.llm_max_retries,
        retry_initial_delay=args.llm_retry_initial_delay,
        retry_max_delay=args.llm_retry_max_delay,
    )


def make_filter_client(config: AppConfig, logger: JsonlLogger | NullLogger, args: argparse.Namespace) -> OpenAIStreamingClient:
    return OpenAIStreamingClient(
        base_url=config.filter_base_url or "",
        api_key=config.filter_api_key or "",
        model=config.filter_model,
        logger=logger,
        max_retries=args.llm_max_retries,
        retry_initial_delay=args.llm_retry_initial_delay,
        retry_max_delay=args.llm_retry_max_delay,
    )


def initial_counts() -> dict[str, int]:
    return {
        "processed_personas": 0,
        "topics": 0,
        "source_preferences": 0,
        "case_relation_plan_items": 0,
        "generated_cases": 0,
        "accepted_cases": 0,
        "rejected_cases": 0,
        "generated_conversations": 0,
        "accepted_conversations": 0,
        "rejected_conversations": 0,
        "qa_eligible_cases": 0,
        "qa_conversations": 0,
        "conversation_dropped_cases": 0,
        "generated_case_qa": 0,
        "accepted_case_qa": 0,
        "rejected_case_qa": 0,
        "evaluation_instances": 0,
        "conversation_total_turns": 0,
        "conversation_total_tokens": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "validation_failures": 0,
    }


def build_run_report(
    run_id: str,
    started_at: str,
    args: argparse.Namespace,
    config: AppConfig,
    counts: dict[str, int],
    errors: list[dict[str, Any]],
    status: RunStatus,
) -> RunReport:
    return RunReport(
        run_id=run_id,
        started_at=started_at,
        completed_at=utc_now_iso(),
        status=status,
        input_dir=str(args.input_dir),
        output_dir=str(args.output_dir / run_id),
        config_path=str(args.config.resolve()) if getattr(args, "config", None) is not None else None,
        generation_model=config.generation_model,
        filter_model=config.filter_model,
        qa_mode=args.qa_mode,
        persona_ids=list(selected_persona_ids_for_run(args) or []),
        branched_from=getattr(args, "source_run", None) is not None,
        source_run=str(Path(args.source_run).resolve()) if getattr(args, "source_run", None) is not None else None,
        from_stage=str(getattr(args, "from_stage", None)) if getattr(args, "from_stage", None) is not None else None,
        to_stage=str(getattr(args, "to_stage", None)) if getattr(args, "to_stage", None) is not None else None,
        counts=counts,
        pass_rate=calculate_pass_rate(counts.get("accepted_cases", 0), counts.get("generated_cases", 0)),
        errors=errors,
    )


def case_concurrency(args: argparse.Namespace) -> int:
    return args.case_concurrency or args.llm_concurrency


def filter_concurrency(args: argparse.Namespace) -> int:
    return args.filter_concurrency or args.llm_concurrency


def conversation_concurrency(args: argparse.Namespace) -> int:
    return args.conversation_concurrency or args.llm_concurrency


def qa_concurrency(args: argparse.Namespace) -> int:
    return args.qa_concurrency or args.llm_concurrency


def write_json(path: Path, value: Any, *, sort_keys: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = to_jsonable(value)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=sort_keys)
        handle.write("\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_models(path: Path, model_cls: type[T]) -> list[T]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON array in {path}")
    return [model_cls.model_validate(item) for item in payload]


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    return value


def remove_file_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def remove_tree_if_exists(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def make_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


if __name__ == "__main__":
    sys.exit(main())
