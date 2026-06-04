from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from generation.prompts import nuanced_temporal as prompts
from infra.json_utils import clean_text, ensure_dir, extract_json_from_response, load_json, make_seeded_rng
from infra.llm_client import LLMClient


SOURCE_FILES = {
    "temporal_light": "source_data/nuanced/light/hoh_temporal.json",
}


TEMPORAL_QUESTION_CUES = (
    "after",
    "at the time",
    "before",
    "by the time",
    "during",
    "draft",
    "earlier",
    "final",
    "latest before",
    "note",
    "older",
    "revision",
    "session",
    "snapshot",
    "update",
    "version",
    "year",
    "years",
    "when",
    "date",
    "dates",
    "timeline",
    "time",
    "period",
    "periods",
    "span",
    "spans",
    "across",
    "over time",
    "chronolog",
    "order",
    "sequence",
    "later",
    "first aired",
    "first air",
)


TIMELINE_AGGREGATION_MARKERS = (
    "timeline",
    "chronological",
    "chronologically",
    "change over time",
    "over time",
    "full picture",
    "full timeline",
    "all the",
    "everything we discussed",
    "lay out",
    "put everything",
)


@dataclass
class GenerationResult:
    record: dict[str, Any]


class NuancedTemporalPipeline:
    def __init__(self, root_dir: str | Path, config: dict[str, Any], llm: LLMClient | None = None):
        self.root_dir = Path(root_dir)
        self.config = config
        self.llm = llm or LLMClient(config)
        self.generation_cfg = config["generation"]
        self.shared_conversation_generation_cfg = self._shared_conversation_generation_cfg()
        self.session_candidate_count = int(
            self.generation_cfg.get("conversation_type_candidate_count")
            or self.shared_conversation_generation_cfg.get("conversation_type_candidate_count", 4)
        )
        self.conversation_type_descriptions = self._load_conversation_type_descriptions()
        self.conversation_type_flows = self._load_conversation_type_flows()
        self.available_conversation_types = [
            type_name for type_name in self.conversation_type_descriptions if self.conversation_type_flows.get(type_name)
        ]
        if not self.available_conversation_types:
            raise ValueError(
                "Nuanced temporal generation requires non-empty conversation type descriptions and flows."
            )
        ensure_dir(self.root_dir / self.config["paths"]["output_dir"])

    def _shared_conversation_generation_cfg(self) -> dict[str, Any]:
        explicit_cfg_path = self.generation_cfg.get("conversation_type_config_path")
        if explicit_cfg_path:
            data = load_json(self.root_dir / str(explicit_cfg_path))
            if not isinstance(data, dict):
                raise ValueError(f"Expected JSON object in conversation type config: {explicit_cfg_path}")
            return data

        shared_cfg_path = self.root_dir / "config" / "contradictory_generation.json"
        if not shared_cfg_path.exists():
            return {}
        data = load_json(shared_cfg_path)
        if not isinstance(data, dict):
            return {}
        generation = data.get("generation")
        return generation if isinstance(generation, dict) else {}

    def _load_conversation_type_descriptions(self) -> dict[str, str]:
        raw = self.generation_cfg.get("CONVERSATION_TYPE_DESCRIPTIONS") or self.shared_conversation_generation_cfg.get(
            "CONVERSATION_TYPE_DESCRIPTIONS",
            {},
        )
        if not isinstance(raw, dict):
            raise ValueError("generation.CONVERSATION_TYPE_DESCRIPTIONS must be a JSON object.")
        normalized: dict[str, str] = {}
        for key, value in raw.items():
            type_name = clean_text(str(key))
            description = clean_text(str(value))
            if type_name and description:
                normalized[type_name] = description
        return normalized

    def _load_conversation_type_flows(self) -> dict[str, list[str]]:
        raw = self.generation_cfg.get("CONVERSATION_TYPE_FLOW_DESCRIPTIONS") or self.shared_conversation_generation_cfg.get(
            "CONVERSATION_TYPE_FLOW_DESCRIPTIONS",
            {},
        )
        if not isinstance(raw, dict):
            raise ValueError("generation.CONVERSATION_TYPE_FLOW_DESCRIPTIONS must be a JSON object.")
        normalized: dict[str, list[str]] = {}
        for key, value in raw.items():
            type_name = clean_text(str(key))
            if not type_name:
                continue
            if not isinstance(value, list):
                raise ValueError("Each conversation type flow entry must be a list of strings.")
            flows = [clean_text(str(item)) for item in value if clean_text(str(item))]
            if flows:
                normalized[type_name] = flows
        return normalized

    def _choose_primary_alias(self, aliases: list[str]) -> str:
        if not aliases:
            return ""
        return max(aliases, key=lambda value: (len(value), value))

    def _date_part(self, raw_time: str) -> str:
        raw_time = clean_text(str(raw_time))
        if "T" in raw_time:
            raw_time = raw_time.split("T", 1)[0]
        return raw_time

    def resolve_input_paths(self, sources: list[str] | None = None) -> list[Path]:
        if not sources or sources == ["all"]:
            sources = list(SOURCE_FILES.keys())
        paths: list[Path] = []
        for source in sources:
            if source not in SOURCE_FILES:
                raise ValueError(f"Unknown source '{source}'. Valid options: {sorted(SOURCE_FILES)}")
            paths.append(self.root_dir / SOURCE_FILES[source])
        return paths

    def load_samples(self, sources: list[str] | None = None) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for path in self.resolve_input_paths(sources):
            data = load_json(path)
            if not isinstance(data, list):
                raise ValueError(f"Expected list in {path}, got {type(data).__name__}")
            for idx, item in enumerate(data):
                samples.append(self.normalize_sample(item, sample_index=idx, source_path=path))
        return samples

    def normalize_sample(self, item: dict[str, Any], sample_index: int, source_path: Path) -> dict[str, Any]:
        source_question = clean_text(str(item["question"]))
        document = item.get("document") or {}
        snapshots: list[dict[str, Any]] = []
        for outdated in item.get("outdated_infos", []):
            snapshots.append(
                {
                    "snapshot_role": "outdated",
                    "last_modified_time": self._date_part(outdated.get("last_modified_time", "")),
                    "answer": clean_text(str(outdated.get("answer", ""))),
                    "evidence": clean_text(str(outdated.get("evidence", ""))),
                }
            )
        snapshots.append(
            {
                "snapshot_role": "current",
                "last_modified_time": self._date_part(item.get("last_modified_time", "")),
                "answer": clean_text(str(item.get("answer", ""))),
                "evidence": clean_text(str(item.get("evidence", ""))),
            }
        )
        snapshots = [snapshot for snapshot in snapshots if snapshot["last_modified_time"] and snapshot["answer"]]
        snapshots.sort(key=lambda snapshot: snapshot["last_modified_time"])
        unique_snapshots: list[dict[str, Any]] = []
        seen_answers: set[str] = set()
        for snapshot in snapshots:
            answer_key = self._canonicalize_text(snapshot["answer"])
            if answer_key in seen_answers:
                continue
            unique_snapshots.append(snapshot)
            seen_answers.add(answer_key)
        snapshots = unique_snapshots
        if len(snapshots) < 2:
            raise ValueError(f"HoH temporal source item {sample_index} must contain at least two dated snapshots.")

        memory_items: list[dict[str, Any]] = []
        for idx, snapshot in enumerate(snapshots, start=1):
            time_condition = f"as of {snapshot['last_modified_time']}"
            answer_text = snapshot["answer"]
            memory_items.append(
                {
                    "memory_id": f"nu-temp-{sample_index + 1:05d}-m{idx:02d}",
                    "base_question": source_question,
                    "time_specific_question": f"{source_question} ({time_condition})",
                    "time_condition": time_condition,
                    "last_modified_time": snapshot["last_modified_time"],
                    "snapshot_role": snapshot["snapshot_role"],
                    "answer_aliases": [answer_text],
                    "answer_text": answer_text,
                    "evidence": snapshot["evidence"],
                    "document": document,
                }
            )

        return {
            "sample_id": f"nu-temp-{sample_index + 1:05d}",
            "relationship": "nuanced",
            "task_family": "nuanced",
            "nuanced_subtype": "temporal",
            "source_dataset": "HoH-Temporal-Light",
            "source_path": str(source_path),
            "source_question": source_question,
            "document": document,
            "memory_items": memory_items,
            "source_record": item,
        }

    def _session_count_for_fact_count(self, fact_count: int) -> int:
        return 2 if fact_count <= 4 else 3

    def _conversation_candidates_for_session(self, sample_id: str, session_id: str) -> dict[str, Any]:
        rng = make_seeded_rng(
            self.generation_cfg["seed"],
            f"temporal-conversation-type-candidates:{sample_id}:{session_id}",
        )
        candidate_count = min(self.session_candidate_count, len(self.available_conversation_types))
        sampled_types = rng.sample(self.available_conversation_types, k=candidate_count)
        preferred_type = rng.choice(sampled_types)

        candidates: list[dict[str, str]] = []
        for type_name in sampled_types:
            flow = rng.choice(self.conversation_type_flows[type_name])
            candidates.append(
                {
                    "conversation_type": type_name,
                    "type_description": self.conversation_type_descriptions[type_name],
                    "conversation_flow": flow,
                    "candidate_tier": "preferred" if type_name == preferred_type else "fallback",
                }
            )

        preferred_candidate = next(candidate for candidate in candidates if candidate["candidate_tier"] == "preferred")
        fallback_candidates = [candidate for candidate in candidates if candidate["candidate_tier"] == "fallback"]
        return {
            "preferred_candidate": preferred_candidate,
            "fallback_candidates": fallback_candidates,
            "all_candidates": [preferred_candidate] + fallback_candidates,
        }

    def _build_session_plan_context(self, sample: dict[str, Any]) -> list[dict[str, Any]]:
        session_count = self._session_count_for_fact_count(len(sample["selected_temporal_facts"]))
        contexts: list[dict[str, Any]] = []
        for idx in range(session_count):
            session_id = f"s{idx + 1}"
            contexts.append(
                {
                    "session_id": session_id,
                    "conversation_candidates": self._conversation_candidates_for_session(sample["sample_id"], session_id),
                    "fact_capacity": "1-3 facts",
                }
            )
        return contexts

    def _select_target_fact(self, sample: dict[str, Any], session_plans: list[dict[str, Any]]) -> dict[str, Any]:
        rng = make_seeded_rng(self.generation_cfg["seed"], f"temporal-target:{sample['sample_id']}")
        fact_by_id = {fact["memory_id"]: fact for fact in sample["selected_temporal_facts"]}
        preferred_ids: list[str] = []
        for plan in session_plans:
            if len(plan["assigned_memory_ids"]) >= 2:
                preferred_ids.extend(plan["assigned_memory_ids"])
        candidate_ids = preferred_ids or [fact["memory_id"] for fact in sample["selected_temporal_facts"]]
        chosen_id = rng.choice(candidate_ids)
        return fact_by_id[chosen_id]

    def generate_sample(self, sample: dict[str, Any]) -> GenerationResult:
        sample = self.generate_fact_bundle(sample)
        session_plans = self.generate_session_plans(sample)
        target_temporal_fact = self._select_target_fact(sample, session_plans)
        enriched_sample = {
            **sample,
            "session_plans": session_plans,
            "target_temporal_fact": target_temporal_fact,
        }
        session_bundles = self.generate_sessions(enriched_sample, session_plans)
        question_bundle = self.generate_questions(enriched_sample, session_bundles)
        qa_pairs = self.generate_qa_pairs(enriched_sample, session_bundles, question_bundle["questions"])
        primary_qa = qa_pairs[0]

        record = {
            "sample_id": sample["sample_id"],
            "relationship": "nuanced",
            "task_family": "nuanced",
            "nuanced_subtype": "temporal",
            "source_dataset": sample["source_dataset"],
            "temporal_question": sample["temporal_question"],
            "selected_temporal_facts": sample["selected_temporal_facts"],
            "target_temporal_fact": target_temporal_fact,
            "session_plans": session_plans,
            "sessions": [
                {
                    "session_id": plan["session_id"],
                    "session_scenario": bundle["chosen_scenario"],
                    "covered_memory_ids": plan["assigned_memory_ids"],
                    "conversation": bundle["conversation"],
                }
                for plan, bundle in zip(session_plans, session_bundles)
            ],
            "qa_pairs": qa_pairs,
            "question": primary_qa["question"],
            "correct_answers": primary_qa["correct_answers"],
            "incorrect_answers": primary_qa["incorrect_answers"],
            "metadata": {
                "source_question": sample["source_question"],
                "memory_items": sample["memory_items"],
                "fact_selection_notes": sample["fact_selection_notes"],
                "session_coverage_notes": [
                    {
                        "session_id": plan["session_id"],
                        "coverage_notes": bundle.get("coverage_notes", []),
                    }
                    for plan, bundle in zip(session_plans, session_bundles)
                ],
                "source_record": sample["source_record"],
            },
        }
        return GenerationResult(record=record)

    def generate_fact_bundle(self, sample: dict[str, Any]) -> dict[str, Any]:
        prompt = prompts.build_fact_selection_prompt(sample)
        messages = [
            {"role": "system", "content": prompts.FACT_SELECTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        last_error: Exception | None = None
        for attempt in range(3):
            if attempt > 0:
                messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            f"The previous output was invalid: {last_error}. "
                            "Return corrected JSON only. Keep every source fact, normalize the temporal question, "
                            "and preserve explicit time conditions for each fact."
                        ),
                    }
                ]
            raw = self.llm.chat(messages)
            try:
                parsed = extract_json_from_response(raw)
                return self._validate_fact_bundle(parsed, sample)
            except Exception as exc:
                last_error = exc
        raise ValueError(f"Failed to generate valid temporal fact bundle for {sample['sample_id']}: {last_error}")

    def generate_session_plans(self, sample: dict[str, Any]) -> list[dict[str, Any]]:
        session_plan_context = self._build_session_plan_context(sample)
        prompt = prompts.build_session_plan_prompt(sample, session_plan_context)
        messages = [
            {"role": "system", "content": prompts.SESSION_PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        last_error: Exception | None = None
        for attempt in range(3):
            if attempt > 0:
                messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            f"The previous output was invalid: {last_error}. "
                            "Return corrected JSON only. Produce the required number of session plans, "
                            "assign every temporal fact exactly once, and keep each session within the 1-3 fact limit."
                        ),
                    }
                ]
            raw = self.llm.chat(messages)
            try:
                parsed = extract_json_from_response(raw)
                return self._validate_session_plan_bundle(parsed, sample, session_plan_context)
            except Exception as exc:
                last_error = exc
        raise ValueError(f"Failed to generate valid session plans for {sample['sample_id']}: {last_error}")

    def generate_sessions(self, sample: dict[str, Any], session_plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not session_plans:
            return []
        bundles: list[dict[str, Any] | None] = [None] * len(session_plans)
        max_workers = min(len(session_plans), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self.generate_session, sample, session_plan): idx
                for idx, session_plan in enumerate(session_plans)
            }
            for future, idx in future_to_index.items():
                bundles[idx] = future.result()
        return [bundle for bundle in bundles if bundle is not None]

    def generate_session(self, sample: dict[str, Any], session_plan: dict[str, Any]) -> dict[str, Any]:
        min_rounds = int(self.generation_cfg["conversation_round_min"])
        max_rounds = int(self.generation_cfg["conversation_round_max"])
        fact_by_id = {fact["memory_id"]: fact for fact in sample["selected_temporal_facts"]}
        assigned_facts = [fact_by_id[memory_id] for memory_id in session_plan["assigned_memory_ids"]]
        prompt = prompts.build_session_prompt(sample, session_plan, assigned_facts, min_rounds, max_rounds)
        messages = [
            {"role": "system", "content": prompts.CONVERSATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        last_error: Exception | None = None
        for attempt in range(3):
            if attempt > 0:
                messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            f"The previous output was invalid: {last_error}. "
                            f"Return corrected JSON only. The session must contain at least {min_rounds} rounds, "
                            f"should stay around {min_rounds}-{max_rounds} rounds, and must strictly alternate user and assistant."
                        ),
                    }
                ]
            raw = self.llm.chat(messages)
            try:
                parsed = extract_json_from_response(raw)
                self._validate_session_bundle(parsed, assigned_facts, min_rounds, max_rounds)
                return parsed
            except Exception as exc:
                last_error = exc
        raise ValueError(f"Failed to generate a valid session for {sample['sample_id']} {session_plan['session_id']}: {last_error}")

    def generate_questions(self, sample: dict[str, Any], session_bundles: list[dict[str, Any]]) -> dict[str, Any]:
        session_payload = self._session_payload(sample, session_bundles)
        prompt = prompts.build_question_prompt(sample, session_payload)
        messages = [
            {"role": "system", "content": prompts.QUESTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        last_error: Exception | None = None
        for attempt in range(3):
            if attempt > 0:
                messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            f"The previous output was invalid: {last_error}. "
                            "Return corrected JSON only with exactly 3 question objects. Include one easy, one medium, and one hard question, "
                            "and make each question resolve to exactly one target temporal fact."
                        ),
                    }
                ]
            raw = self.llm.chat(messages)
            try:
                parsed = extract_json_from_response(raw)
                questions = self._validate_question_bundle(parsed, sample)
                return {"questions": questions}
            except Exception as exc:
                last_error = exc
        raise ValueError(f"Failed to generate valid questions for {sample['sample_id']}: {last_error}")

    def generate_qa_pairs(
        self,
        sample: dict[str, Any],
        session_bundles: list[dict[str, Any]],
        questions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not questions:
            return []
        qa_pairs: list[dict[str, Any] | None] = [None] * len(questions)
        max_workers = min(len(questions), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self.generate_answers, sample, session_bundles, question_item): idx
                for idx, question_item in enumerate(questions)
            }
            for future, idx in future_to_index.items():
                answer_bundle = future.result()
                qa_pairs[idx] = {
                    **questions[idx],
                    "correct_answers": answer_bundle["correct_answers"],
                    "incorrect_answers": answer_bundle["incorrect_answers"],
                }
        return [qa_pair for qa_pair in qa_pairs if qa_pair is not None]

    def generate_answers(
        self,
        sample: dict[str, Any],
        session_bundles: list[dict[str, Any]],
        question_item: dict[str, Any],
    ) -> dict[str, Any]:
        session_payload = self._session_payload(sample, session_bundles)
        prompt = prompts.build_answer_prompt(sample, session_payload, question_item)
        messages = [
            {"role": "system", "content": prompts.ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        last_error: Exception | None = None
        for attempt in range(3):
            if attempt > 0:
                messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            f"The previous output was invalid: {last_error}. "
                            "Return corrected JSON only with exactly 3 correct_answers and 3 incorrect_answers."
                        ),
                    }
                ]
            raw = self.llm.chat(messages)
            try:
                parsed = extract_json_from_response(raw)
                correct_answers = self._normalize_answer_candidates(parsed.get("correct_answers"))
                incorrect_answers = self._normalize_answer_candidates(parsed.get("incorrect_answers"))
                if len(correct_answers) != 3 or len(incorrect_answers) != 3:
                    raise ValueError("Expected exactly 3 correct answers and 3 incorrect answers.")
                target_fact = self._fact_by_id(sample, question_item["target_memory_id"])
                self._validate_answer_candidate_quality(
                    correct_answers,
                    incorrect_answers,
                    target_fact,
                    sample["selected_temporal_facts"],
                )
                return {
                    "correct_answers": correct_answers,
                    "incorrect_answers": incorrect_answers,
                }
            except Exception as exc:
                last_error = exc
        raise ValueError(f"Failed to generate valid answer candidates for {sample['sample_id']}: {last_error}")

    def _validate_fact_bundle(self, bundle: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(bundle, dict):
            raise ValueError("Temporal fact bundle must be a JSON object.")

        temporal_question = clean_text(str(bundle.get("temporal_question", "")))
        if not temporal_question:
            raise ValueError("Temporal fact bundle must include a non-empty temporal_question.")

        raw_facts = bundle.get("temporal_facts")
        if not isinstance(raw_facts, list):
            raise ValueError("Temporal fact bundle must include a temporal_facts list.")
        if len(raw_facts) != len(sample["memory_items"]):
            raise ValueError(
                f"Expected {len(sample['memory_items'])} temporal facts, got {len(raw_facts)}."
            )

        memory_by_id = {item["memory_id"]: item for item in sample["memory_items"]}
        normalized: list[dict[str, Any]] = []
        seen_memory_ids: set[str] = set()
        for raw_fact in raw_facts:
            if not isinstance(raw_fact, dict):
                raise ValueError("Each temporal fact must be a JSON object.")
            memory_id = clean_text(str(raw_fact.get("memory_id", "")))
            if memory_id not in memory_by_id:
                raise ValueError(f"Unknown memory_id in temporal_facts: {memory_id!r}")
            if memory_id in seen_memory_ids:
                raise ValueError(f"Duplicate temporal fact for {memory_id}.")

            source_memory = memory_by_id[memory_id]
            time_condition = clean_text(str(raw_fact.get("time_condition", "")))
            answer_text = clean_text(str(raw_fact.get("answer_text", "")))
            fact_statement = clean_text(str(raw_fact.get("fact_statement", "")))
            if not time_condition or not answer_text or not fact_statement:
                raise ValueError(f"Temporal fact {memory_id} is missing a required non-empty field.")
            if source_memory["last_modified_time"] not in time_condition:
                raise ValueError(f"Temporal fact {memory_id} must preserve the source snapshot date.")
            if not self._answer_matches_source_aliases(answer_text, source_memory["answer_aliases"]):
                raise ValueError(f"Temporal fact {memory_id} answer {answer_text!r} is not grounded in source aliases.")

            normalized.append(
                {
                    "memory_id": memory_id,
                    "base_question": source_memory["base_question"],
                    "time_specific_question": source_memory["time_specific_question"],
                    "time_condition": time_condition,
                    "last_modified_time": source_memory["last_modified_time"],
                    "snapshot_role": source_memory["snapshot_role"],
                    "answer_aliases": source_memory["answer_aliases"],
                    "answer_text": answer_text,
                    "evidence": source_memory["evidence"],
                    "document": source_memory["document"],
                    "fact_statement": fact_statement,
                }
            )
            seen_memory_ids.add(memory_id)

        notes = clean_text(str(bundle.get("fact_selection_notes", "")))
        if not notes:
            raise ValueError("Temporal fact bundle must include non-empty fact_selection_notes.")

        ordered_facts = [next(fact for fact in normalized if fact["memory_id"] == item["memory_id"]) for item in sample["memory_items"]]
        return {
            **sample,
            "temporal_question": temporal_question,
            "selected_temporal_facts": ordered_facts,
            "fact_selection_notes": notes,
        }

    def _validate_session_plan_bundle(
        self,
        bundle: dict[str, Any],
        sample: dict[str, Any],
        session_plan_context: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not isinstance(bundle, dict):
            raise ValueError("Session-plan bundle must be a JSON object.")
        raw_plans = bundle.get("session_plans")
        if not isinstance(raw_plans, list):
            raise ValueError("Session-plan bundle must include a 'session_plans' list.")

        expected_session_ids = [context["session_id"] for context in session_plan_context]
        if len(raw_plans) != len(expected_session_ids):
            raise ValueError(
                f"Expected {len(expected_session_ids)} session plans, got {len(raw_plans)}."
            )

        context_by_id = {context["session_id"]: context for context in session_plan_context}
        fact_ids = [fact["memory_id"] for fact in sample["selected_temporal_facts"]]
        fact_id_set = set(fact_ids)
        normalized_by_id: dict[str, dict[str, Any]] = {}
        assigned_fact_ids: list[str] = []
        required_fields = [
            "chosen_conversation_type",
            "chosen_conversation_flow",
            "scenario_label",
            "event_signature",
            "event_summary",
            "opening_situation",
            "user_goal",
            "assistant_role",
            "temporal_grouping_rationale",
            "fact_integration_plan",
            "distinct_from_other_sessions",
        ]

        for raw_plan in raw_plans:
            if not isinstance(raw_plan, dict):
                raise ValueError("Each session plan must be a JSON object.")
            session_id = clean_text(str(raw_plan.get("session_id", "")))
            if session_id not in expected_session_ids:
                raise ValueError(f"Unexpected session_id in session plans: {session_id!r}")
            if session_id in normalized_by_id:
                raise ValueError(f"Duplicate session plan for {session_id}.")

            normalized_plan: dict[str, Any] = {"session_id": session_id}
            context = context_by_id[session_id]
            candidate_map = {
                candidate["conversation_type"]: candidate
                for candidate in context["conversation_candidates"]["all_candidates"]
            }
            for field in required_fields:
                value = clean_text(str(raw_plan.get(field, "")))
                if not value:
                    raise ValueError(f"Session plan {session_id} is missing non-empty field '{field}'.")
                normalized_plan[field] = value

            assigned_ids = raw_plan.get("assigned_memory_ids")
            if not isinstance(assigned_ids, list):
                raise ValueError(f"Session plan {session_id} must include assigned_memory_ids as a list.")
            cleaned_assigned_ids = [clean_text(str(memory_id)) for memory_id in assigned_ids if clean_text(str(memory_id))]
            if not cleaned_assigned_ids or len(cleaned_assigned_ids) > 3:
                raise ValueError(f"Session plan {session_id} must assign between 1 and 3 facts.")
            if len(set(cleaned_assigned_ids)) != len(cleaned_assigned_ids):
                raise ValueError(f"Session plan {session_id} has duplicate memory ids.")
            if any(memory_id not in fact_id_set for memory_id in cleaned_assigned_ids):
                raise ValueError(f"Session plan {session_id} references unknown fact ids.")
            normalized_plan["assigned_memory_ids"] = cleaned_assigned_ids

            chosen_type = normalized_plan["chosen_conversation_type"]
            chosen_flow = normalized_plan["chosen_conversation_flow"]
            if chosen_type not in candidate_map:
                raise ValueError(
                    f"Session plan {session_id} chose unsupported conversation type {chosen_type!r}."
                )
            candidate = candidate_map[chosen_type]
            if chosen_flow != candidate["conversation_flow"]:
                raise ValueError(
                    f"Session plan {session_id} chose a flow that does not match the provided candidate for {chosen_type!r}."
                )

            normalized_plan["chosen_candidate_tier"] = candidate["candidate_tier"]
            normalized_plan["chosen_conversation_type_description"] = candidate["type_description"]
            normalized_by_id[session_id] = normalized_plan
            assigned_fact_ids.extend(cleaned_assigned_ids)

        if set(assigned_fact_ids) != fact_id_set:
            raise ValueError("Session plans must cover every temporal fact exactly once.")
        if len(assigned_fact_ids) != len(fact_ids):
            raise ValueError("Temporal facts must not be assigned to more than one session.")

        ordered_plans = [normalized_by_id[session_id] for session_id in expected_session_ids]
        scenario_labels = [plan["scenario_label"].casefold() for plan in ordered_plans]
        if len(set(scenario_labels)) != len(scenario_labels):
            raise ValueError("Session plans must use different scenario_label values.")
        event_signatures = [plan["event_signature"].casefold() for plan in ordered_plans]
        if len(set(event_signatures)) != len(event_signatures):
            raise ValueError("Session plans must use different event_signature values.")
        normalized_event_summaries = [
            re.sub(r"[^a-z0-9]+", " ", plan["event_summary"].casefold()).strip()
            for plan in ordered_plans
        ]
        if len(set(normalized_event_summaries)) != len(normalized_event_summaries):
            raise ValueError("Session plans must use different event_summary values.")
        return ordered_plans

    def _session_payload(self, sample: dict[str, Any], session_bundles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        plan_by_id = {plan["session_id"]: plan for plan in sample["session_plans"]} if "session_plans" in sample else {}
        payload: list[dict[str, Any]] = []
        if not plan_by_id:
            return payload
        fact_by_id = {fact["memory_id"]: fact for fact in sample["selected_temporal_facts"]}
        for bundle in session_bundles:
            session_id = clean_text(str(bundle.get("session_id", "")))
            if not session_id or session_id not in plan_by_id:
                continue
            plan = plan_by_id[session_id]
            payload.append(
                {
                    "session_id": session_id,
                    "chosen_scenario": bundle["chosen_scenario"],
                    "covered_facts": [fact_by_id[memory_id] for memory_id in plan["assigned_memory_ids"]],
                    "conversation": bundle["conversation"],
                }
            )
        return payload

    def _validate_session_bundle(
        self,
        bundle: dict[str, Any],
        assigned_facts: list[dict[str, Any]],
        min_rounds: int,
        max_rounds: int,
    ) -> None:
        if not isinstance(bundle, dict):
            raise ValueError("Session bundle must be a JSON object.")
        chosen_scenario = clean_text(str(bundle.get("chosen_scenario", "")))
        session_id = clean_text(str(bundle.get("session_id", "")))
        if not chosen_scenario:
            raise ValueError("Session bundle must include a non-empty chosen_scenario.")
        if not session_id:
            raise ValueError("Session bundle must include a non-empty session_id.")
        bundle["chosen_scenario"] = chosen_scenario
        bundle["session_id"] = session_id

        conversation = bundle.get("conversation")
        if not isinstance(conversation, list):
            raise ValueError("Conversation must be a list.")
        min_messages = min_rounds * 2
        max_messages = max_rounds * 2
        if len(conversation) < min_messages:
            raise ValueError(f"Conversation must contain at least {min_messages} messages, got {len(conversation)}.")
        if len(conversation) > max_messages:
            raise ValueError(
                f"Conversation should stay around {min_rounds}-{max_rounds} rounds; got {len(conversation)} messages."
            )
        for idx, message in enumerate(conversation):
            if not isinstance(message, dict):
                raise ValueError(f"Conversation message at index {idx} must be an object.")
            if message.get("role") not in {"user", "assistant"}:
                raise ValueError(f"Invalid role at index {idx}: {message.get('role')}")
            expected_role = "user" if idx % 2 == 0 else "assistant"
            if message.get("role") != expected_role:
                raise ValueError(
                    f"Conversation must alternate user/assistant and start with user; index {idx} has role {message.get('role')}."
                )
            content = clean_text(str(message.get("content", "")))
            if not content:
                raise ValueError(f"Empty content at message index {idx}.")
            message["content"] = content

        coverage_notes = bundle.get("coverage_notes", [])
        if coverage_notes is None:
            coverage_notes = []
        if not isinstance(coverage_notes, list):
            raise ValueError("coverage_notes must be a list when present.")
        assigned_ids = {fact["memory_id"] for fact in assigned_facts}
        for note in coverage_notes:
            if not isinstance(note, dict):
                raise ValueError("Each coverage note must be an object.")
            memory_id = clean_text(str(note.get("memory_id", "")))
            if memory_id and memory_id not in assigned_ids:
                raise ValueError("coverage_notes referenced a fact not assigned to this session.")

    def _normalize_answer_candidates(self, answers: Any) -> list[dict[str, str]]:
        if not isinstance(answers, list):
            raise ValueError("Answer candidates must be a list.")
        normalized: list[dict[str, str]] = []
        for answer in answers:
            if isinstance(answer, dict):
                text = clean_text(str(answer.get("text", "")))
            else:
                text = clean_text(str(answer))
            if not text:
                raise ValueError("Encountered empty answer candidate.")
            normalized.append({"text": text})
        return normalized

    def _fact_by_id(self, sample: dict[str, Any], memory_id: str) -> dict[str, Any]:
        fact_by_id = {fact["memory_id"]: fact for fact in sample["selected_temporal_facts"]}
        if memory_id not in fact_by_id:
            raise ValueError(f"Unknown target_memory_id: {memory_id!r}")
        return fact_by_id[memory_id]

    def _validate_question_bundle(self, bundle: dict[str, Any], sample: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(bundle, dict):
            raise ValueError("Question bundle must be a JSON object.")
        raw_questions = bundle.get("questions")
        if not isinstance(raw_questions, list):
            raise ValueError("Question bundle must include a questions list.")
        if len(raw_questions) != 3:
            raise ValueError(f"Expected exactly 3 questions, got {len(raw_questions)}.")

        allowed_difficulties = {"easy", "medium", "hard"}
        allowed_anchor_types = {
            "direct_date",
            "event_anchor",
            "session_task_anchor",
            "version_anchor",
            "relative_time",
            "latest_before",
        }
        normalized: list[dict[str, Any]] = []
        seen_questions: set[str] = set()
        difficulties: set[str] = set()
        non_direct_count = 0

        for idx, raw_question in enumerate(raw_questions, start=1):
            if not isinstance(raw_question, dict):
                raise ValueError("Each question item must be a JSON object.")
            question = clean_text(str(raw_question.get("question", "")))
            target_memory_id = clean_text(str(raw_question.get("target_memory_id", "")))
            difficulty = clean_text(str(raw_question.get("difficulty", ""))).casefold()
            anchor_type = clean_text(str(raw_question.get("temporal_anchor_type", ""))).casefold()
            design_notes = clean_text(str(raw_question.get("question_design_notes", "")))
            if difficulty not in allowed_difficulties:
                raise ValueError(f"Question {idx} has unsupported difficulty {difficulty!r}.")
            if anchor_type not in allowed_anchor_types:
                raise ValueError(f"Question {idx} has unsupported temporal_anchor_type {anchor_type!r}.")
            target_fact = self._fact_by_id(sample, target_memory_id)
            self._validate_generated_question(question, target_fact, difficulty, anchor_type)

            question_key = self._canonicalize_text(question)
            if question_key in seen_questions:
                raise ValueError("Generated questions must be distinct.")
            seen_questions.add(question_key)
            difficulties.add(difficulty)
            if anchor_type != "direct_date":
                non_direct_count += 1

            normalized.append(
                {
                    "qa_id": f"{sample['sample_id']}-q{idx:02d}",
                    "difficulty": difficulty,
                    "temporal_anchor_type": anchor_type,
                    "target_memory_id": target_memory_id,
                    "target_temporal_fact": target_fact,
                    "question": question,
                    "question_design_notes": design_notes,
                }
            )

        if difficulties != {"easy", "medium", "hard"}:
            raise ValueError("Question set must include exactly one easy, one medium, and one hard question.")
        if non_direct_count < 2:
            raise ValueError("Question set must include at least two non-direct-date temporal anchors.")
        return normalized

    def _canonicalize_text(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()

    def _answer_matches_source_aliases(self, answer_text: str, aliases: list[str]) -> bool:
        answer_key = self._canonicalize_text(answer_text)
        alias_keys = {self._canonicalize_text(alias) for alias in aliases}
        return answer_key in alias_keys

    def _candidate_mentions_fact(self, candidate_text: str, fact: dict[str, Any]) -> bool:
        candidate_key = self._canonicalize_text(candidate_text)
        answer_key = self._canonicalize_text(fact["answer_text"])
        if answer_key and answer_key in candidate_key:
            return True
        candidate_tokens = set(candidate_key.split())
        stop_tokens = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "against",
            "is",
            "it",
            "match",
            "matches",
            "of",
            "operates",
            "operate",
            "operated",
            "one",
            "the",
            "to",
            "which",
        }
        answer_tokens = [token for token in answer_key.split() if token and token not in stop_tokens]
        if answer_tokens and all(token in candidate_tokens for token in answer_tokens):
            return True
        if len(answer_tokens) >= 4:
            overlap = sum(1 for token in answer_tokens if token in candidate_tokens)
            if overlap / len(answer_tokens) >= 0.6:
                return True
        return False

    def _years_from_text(self, text: str) -> list[str]:
        return re.findall(r"\b(?:19|20)\d{2}\b", text)

    def _date_variants_from_time_condition(self, time_condition: str) -> list[str]:
        match = re.search(r"\b((?:19|20)\d{2})-(\d{2})-(\d{2})\b", time_condition)
        if not match:
            return []
        year, month, day = match.groups()
        month_names = {
            "01": "january",
            "02": "february",
            "03": "march",
            "04": "april",
            "05": "may",
            "06": "june",
            "07": "july",
            "08": "august",
            "09": "september",
            "10": "october",
            "11": "november",
            "12": "december",
        }
        day_int = str(int(day))
        month_name = month_names.get(month, "")
        variants = [
            f"{year}-{month}-{day}",
            f"{year}/{month}/{day}",
            f"{month}/{day}/{year}",
            f"{year}-{month}",
        ]
        if month_name:
            variants.extend(
                [
                    f"{month_name} {day_int}, {year}",
                    f"{month_name} {day_int} {year}",
                    f"{month_name} {year}",
                ]
            )
        return variants

    def _validate_generated_question(
        self,
        question: str,
        target_fact: dict[str, Any],
        difficulty: str,
        anchor_type: str,
    ) -> None:
        lowered = question.casefold()
        if not re.search(r"[?？]$", question):
            raise ValueError("Generated question must end with a question mark.")
        if not re.search(r"\b(?:19|20)\d{2}\b", question) and not any(cue in lowered for cue in TEMPORAL_QUESTION_CUES):
            raise ValueError("Generated question is not explicitly time-related enough.")
        if any(marker in lowered for marker in TIMELINE_AGGREGATION_MARKERS):
            raise ValueError("Generated question reverted to a full-timeline / all-periods question.")
        date_variants = self._date_variants_from_time_condition(target_fact["time_condition"])
        mentions_exact_target_date = bool(date_variants and any(variant in lowered for variant in date_variants))
        if anchor_type == "direct_date":
            if not mentions_exact_target_date:
                raise ValueError("Direct-date question must preserve the exact target snapshot date clearly enough.")
            if "as of" not in lowered and "snapshot" not in lowered and "version" not in lowered:
                raise ValueError("Direct-date question must ask for the answer as of a dated snapshot/version.")
        else:
            if difficulty == "hard" and mentions_exact_target_date:
                raise ValueError("Hard questions should not be answerable by direct target-date string matching.")
            if anchor_type == "latest_before" and "before" not in lowered:
                raise ValueError("latest_before questions must include a before-style temporal anchor.")
            if anchor_type == "relative_time" and not any(cue in lowered for cue in ("before", "after", "earlier", "later")):
                raise ValueError("relative_time questions must use before/after/earlier/later wording.")

    def _validate_answer_candidate_quality(
        self,
        correct_answers: list[dict[str, str]],
        incorrect_answers: list[dict[str, str]],
        target_fact: dict[str, Any],
        facts: list[dict[str, Any]],
    ) -> None:
        correct_texts = [candidate["text"] for candidate in correct_answers]
        incorrect_texts = [candidate["text"] for candidate in incorrect_answers]
        if len({self._canonicalize_text(text) for text in correct_texts}) != len(correct_texts):
            raise ValueError("Correct answers must be distinct.")
        if len({self._canonicalize_text(text) for text in incorrect_texts}) != len(incorrect_texts):
            raise ValueError("Incorrect answers must be distinct.")
        if {self._canonicalize_text(text) for text in correct_texts} & {
            self._canonicalize_text(text) for text in incorrect_texts
        }:
            raise ValueError("Correct and incorrect answers must not overlap.")

        alternative_facts = [fact for fact in facts if fact["memory_id"] != target_fact["memory_id"]]
        for candidate in correct_answers:
            candidate_text = candidate["text"]
            if not self._candidate_mentions_fact(candidate_text, target_fact):
                raise ValueError("Each correct answer must match the target temporal fact.")

        for candidate in incorrect_answers:
            candidate_text = candidate["text"]
            if self._candidate_mentions_fact(candidate_text, target_fact):
                raise ValueError("Incorrect answers must not restate the target temporal fact.")

        if alternative_facts and not any(
            any(self._candidate_mentions_fact(candidate["text"], fact) for fact in alternative_facts)
            for candidate in incorrect_answers
        ):
            raise ValueError("At least one incorrect answer should reflect another time slice from the same topic.")
