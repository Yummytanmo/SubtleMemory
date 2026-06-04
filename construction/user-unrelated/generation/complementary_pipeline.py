from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from generation.prompts import complementary as prompts
from infra.json_utils import (
    clean_text,
    ensure_dir,
    extract_json_from_response,
    load_json,
    make_seeded_rng,
    stable_answer_text,
)
from infra.llm_client import LLMClient


SOURCE_FILES = {
    "fanoutqa_kgt1": "source_data/complementary/fanoutqa_complementary_k_of_n.json",
    "musique_kgt1": "source_data/complementary/musique_complementary_k_of_n.json",
    "qacc_any_one": "source_data/complementary/qacc_complementary_any_one_of_n.json",
}


MAX_SELECTED_FACTS = 6


@dataclass
class GenerationResult:
    record: dict[str, Any]


class ComplementaryPipeline:
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
            raise ValueError("Complementary generation requires non-empty conversation type descriptions and flows.")
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
            for item in data:
                samples.append(self.normalize_sample(item, source_path=path))
        return samples

    def normalize_sample(self, item: dict[str, Any], source_path: Path) -> dict[str, Any]:
        subtype = item["complementary_subtype"]
        k = item.get("k")
        if subtype == "k_of_n":
            major_type = "type1_k_of_n"
            subtype_label = "k_eq_1" if k == 1 else "k_gt_1"
        elif subtype == "any_one_of_n":
            major_type = "type2_any_one_of_n"
            subtype_label = "any_one_of_n"
        else:
            raise ValueError(f"Unexpected complementary subtype: {subtype}")

        return {
            "sample_id": item["seed_id"],
            "relationship": "complementary",
            "task_family": "complementary",
            "complementary_major_type": major_type,
            "complementary_subtype": subtype,
            "complementary_subtype_label": subtype_label,
            "k": k,
            "source_dataset": item.get("source_dataset"),
            "source_path": str(source_path),
            "source_question": clean_text(str(item["question"])),
            "source_answer": item.get("answer"),
            "source_answer_text": clean_text(item.get("answer_text") or stable_answer_text(item.get("answer"))),
            "gold_memory_ids": list(item.get("gold_memory_ids", [])),
            "distractor_memory_ids": list(item.get("distractor_memory_ids", [])),
            "memory_items": item.get("memory_items", []),
            "memory_prompt_items": [self._memory_item_for_prompt(memory_item) for memory_item in item.get("memory_items", [])],
            "source_record": item,
        }

    def _session_count_for_fact_count(self, fact_count: int) -> int:
        return 2 if fact_count <= 4 else 3

    def _conversation_candidates_for_session(self, sample_id: str, session_id: str) -> dict[str, Any]:
        rng = make_seeded_rng(
            self.generation_cfg["seed"],
            f"complementary-conversation-type-candidates:{sample_id}:{session_id}",
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
        session_count = self._session_count_for_fact_count(len(sample["selected_complementary_facts"]))
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

    def generate_sample(self, sample: dict[str, Any]) -> GenerationResult:
        sample = self.generate_fact_bundle(sample)
        session_plans = self.generate_session_plans(sample)
        enriched_sample = {**sample, "session_plans": session_plans}
        session_bundles = self.generate_sessions(enriched_sample, session_plans)
        question_bundle = self.generate_question(enriched_sample, session_bundles)
        answer_bundle = self.generate_answers(enriched_sample, session_bundles, question_bundle["question"])

        record = {
            "sample_id": sample["sample_id"],
            "relationship": "complementary",
            "task_family": "complementary",
            "complementary_major_type": sample["complementary_major_type"],
            "complementary_source_subtype": sample["complementary_subtype"],
            "complementary_subtype": sample["complementary_subtype_label"],
            "k": sample["k"],
            "effective_k": sample["effective_k"],
            "source_dataset": sample["source_dataset"],
            "complementary_question": sample["complementary_question"],
            "canonical_answer": sample["canonical_answer"],
            "selected_complementary_facts": sample["selected_complementary_facts"],
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
            "question": question_bundle["question"],
            "correct_answers": answer_bundle["correct_answers"],
            "incorrect_answers": answer_bundle["incorrect_answers"],
            "metadata": {
                "source_question": sample["source_question"],
                "source_answer": sample["source_answer"],
                "source_answer_text": sample["source_answer_text"],
                "gold_memory_ids": sample["gold_memory_ids"],
                "distractor_memory_ids": sample["distractor_memory_ids"],
                "fact_selection_notes": sample["fact_selection_notes"],
                "memory_items": sample["memory_items"],
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
                            "Return corrected JSON only. Keep the fact bundle complementary, keep the selected facts manageable, "
                            "and make sure the subtype rules about k=1, k>1, or any-one-of-n are satisfied."
                        ),
                    }
                ]
            raw = self.llm.chat(messages)
            try:
                parsed = extract_json_from_response(raw)
                return self._validate_fact_bundle(parsed, sample)
            except Exception as exc:
                last_error = exc
        raise ValueError(f"Failed to generate valid complementary fact bundle for {sample['sample_id']}: {last_error}")

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
                            "assign every selected fact exactly once, keep each session within the 1-3 fact limit, "
                            "and keep the session events clearly different."
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
        fact_by_id = {fact["memory_id"]: fact for fact in sample["selected_complementary_facts"]}
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
                self._validate_session_bundle(parsed, sample, assigned_facts, min_rounds, max_rounds)
                return parsed
            except Exception as exc:
                last_error = exc
        raise ValueError(f"Failed to generate a valid session for {sample['sample_id']} {session_plan['session_id']}: {last_error}")

    def generate_question(self, sample: dict[str, Any], session_bundles: list[dict[str, Any]]) -> dict[str, str]:
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
                            "Return corrected JSON only with one non-empty complementary question."
                        ),
                    }
                ]
            raw = self.llm.chat(messages)
            try:
                parsed = extract_json_from_response(raw)
                question = clean_text(str(parsed.get("question", "")))
                if not question:
                    raise ValueError("Generated question is empty.")
                self._validate_generated_question(question)
                return {"question": question}
            except Exception as exc:
                last_error = exc
        raise ValueError(f"Failed to generate valid question for {sample['sample_id']}: {last_error}")

    def generate_answers(
        self,
        sample: dict[str, Any],
        session_bundles: list[dict[str, Any]],
        question: str,
    ) -> dict[str, Any]:
        session_payload = self._session_payload(sample, session_bundles)
        prompt = prompts.build_answer_prompt(sample, session_payload, question)
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
                self._validate_answer_candidate_quality(correct_answers, incorrect_answers)
                return {"correct_answers": correct_answers, "incorrect_answers": incorrect_answers}
            except Exception as exc:
                last_error = exc
        raise ValueError(f"Failed to generate valid answer candidates for {sample['sample_id']}: {last_error}")

    def _validate_fact_bundle(self, bundle: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(bundle, dict):
            raise ValueError("Complementary fact bundle must be a JSON object.")

        complementary_question = clean_text(str(bundle.get("complementary_question", "")))
        canonical_answer = clean_text(str(bundle.get("canonical_answer", "")))
        notes = clean_text(str(bundle.get("fact_selection_notes", "")))
        raw_facts = bundle.get("selected_facts")
        raw_effective_k = bundle.get("effective_k")

        if not complementary_question:
            raise ValueError("Complementary fact bundle must include a non-empty complementary_question.")
        if not canonical_answer:
            raise ValueError("Complementary fact bundle must include a non-empty canonical_answer.")
        if not notes:
            raise ValueError("Complementary fact bundle must include non-empty fact_selection_notes.")
        if not isinstance(raw_facts, list):
            raise ValueError("Complementary fact bundle must include a selected_facts list.")
        if len(raw_facts) < 2:
            raise ValueError("Complementary fact bundle must contain at least 2 selected facts.")
        if len(raw_facts) > MAX_SELECTED_FACTS:
            raise ValueError(f"Complementary fact bundle must contain at most {MAX_SELECTED_FACTS} selected facts.")

        try:
            effective_k = int(raw_effective_k)
        except (TypeError, ValueError):
            raise ValueError("Complementary fact bundle must include an integer effective_k.") from None

        memory_by_id = {memory_item["memory_id"]: memory_item for memory_item in sample["memory_items"]}
        selected_facts: list[dict[str, Any]] = []
        seen_memory_ids: set[str] = set()
        for raw_fact in raw_facts:
            if not isinstance(raw_fact, dict):
                raise ValueError("Each selected fact must be a JSON object.")
            memory_id = clean_text(str(raw_fact.get("memory_id", "")))
            fact_role = clean_text(str(raw_fact.get("fact_role", "")))
            fact_statement = clean_text(str(raw_fact.get("fact_statement", "")))
            answer_contribution = clean_text(str(raw_fact.get("answer_contribution", "")))
            selection_rationale = clean_text(str(raw_fact.get("selection_rationale", "")))
            if memory_id not in memory_by_id:
                raise ValueError(f"Unknown memory_id in selected_facts: {memory_id!r}")
            if memory_id in seen_memory_ids:
                raise ValueError(f"Duplicate selected fact for {memory_id}.")
            if not fact_role or not fact_statement or not answer_contribution or not selection_rationale:
                raise ValueError(f"Selected fact {memory_id} is missing a required non-empty field.")
            selected_facts.append(
                {
                    "memory_id": memory_id,
                    "fact_role": fact_role,
                    "fact_statement": fact_statement,
                    "answer_contribution": answer_contribution,
                    "selection_rationale": selection_rationale,
                    "source_memory": memory_by_id[memory_id],
                }
            )
            seen_memory_ids.add(memory_id)

        subtype = sample["complementary_subtype_label"]
        roles = [fact["fact_role"] for fact in selected_facts]
        if subtype == "k_eq_1":
            if effective_k != 1:
                raise ValueError("k_eq_1 fact bundle must have effective_k=1.")
            if roles.count("required") != 1:
                raise ValueError("k_eq_1 fact bundle must contain exactly one required fact.")
            if any(role not in {"required", "distractor"} for role in roles):
                raise ValueError("k_eq_1 fact bundle may only use required and distractor fact roles.")
            if roles.count("distractor") < 1:
                raise ValueError("k_eq_1 fact bundle must contain at least one distractor fact.")
        elif subtype == "k_gt_1":
            if effective_k < 2:
                raise ValueError("k_gt_1 fact bundle must have effective_k>=2.")
            required_count = roles.count("required")
            if required_count < 2:
                raise ValueError("k_gt_1 fact bundle must contain at least two required facts.")
            if effective_k > required_count:
                raise ValueError("effective_k cannot exceed the number of required facts.")
            if any(role not in {"required", "distractor"} for role in roles):
                raise ValueError("k_gt_1 fact bundle may only use required and distractor fact roles.")
        elif subtype == "any_one_of_n":
            if effective_k != 1:
                raise ValueError("any_one_of_n fact bundle must have effective_k=1.")
            if any(role != "equivalent" for role in roles):
                raise ValueError("any_one_of_n fact bundle must contain only equivalent facts.")
        else:
            raise ValueError(f"Unknown complementary subtype label: {subtype}")

        ordered_facts = [next(fact for fact in selected_facts if fact["memory_id"] == memory_id) for memory_id in seen_memory_ids]
        ordered_facts = sorted(
            ordered_facts,
            key=lambda fact: next(
                idx for idx, memory_item in enumerate(sample["memory_items"]) if memory_item["memory_id"] == fact["memory_id"]
            ),
        )
        return {
            **sample,
            "complementary_question": complementary_question,
            "canonical_answer": canonical_answer,
            "effective_k": effective_k,
            "selected_complementary_facts": ordered_facts,
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
            raise ValueError(f"Expected {len(expected_session_ids)} session plans, got {len(raw_plans)}.")

        context_by_id = {context["session_id"]: context for context in session_plan_context}
        fact_ids = [fact["memory_id"] for fact in sample["selected_complementary_facts"]]
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
            "grouping_rationale",
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
                raise ValueError(f"Session plan {session_id} chose unsupported conversation type {chosen_type!r}.")
            candidate = candidate_map[chosen_type]
            if chosen_flow != candidate["conversation_flow"]:
                raise ValueError(
                    f"Session plan {session_id} chose a flow that does not match the provided candidate for {chosen_type!r}."
                )

            normalized_plan["chosen_candidate_tier"] = candidate["candidate_tier"]
            normalized_plan["chosen_conversation_type_description"] = candidate["type_description"]
            normalized_plan["preferred_candidate"] = context["conversation_candidates"]["preferred_candidate"]
            normalized_plan["fallback_candidates"] = context["conversation_candidates"]["fallback_candidates"]
            normalized_by_id[session_id] = normalized_plan
            assigned_fact_ids.extend(cleaned_assigned_ids)

        if set(assigned_fact_ids) != fact_id_set:
            raise ValueError("Session plans must cover every selected fact exactly once.")
        if len(assigned_fact_ids) != len(fact_ids):
            raise ValueError("Selected facts must not be assigned to more than one session.")

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

        if sample["complementary_subtype_label"] == "k_gt_1" and sample["effective_k"] > 1:
            required_ids = {
                fact["memory_id"]
                for fact in sample["selected_complementary_facts"]
                if fact["fact_role"] == "required"
            }
            sessions_with_required = [
                plan for plan in ordered_plans if required_ids.intersection(plan["assigned_memory_ids"])
            ]
            if len(required_ids) > 1 and len(sessions_with_required) < 2:
                raise ValueError("k_gt_1 session plans must spread required facts across at least two sessions.")
            if any(required_ids.issubset(set(plan["assigned_memory_ids"])) for plan in ordered_plans):
                raise ValueError("k_gt_1 session plans must not place all required facts into a single session.")

        return ordered_plans

    def _session_payload(self, sample: dict[str, Any], session_bundles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        plan_by_id = {plan["session_id"]: plan for plan in sample["session_plans"]} if "session_plans" in sample else {}
        payload: list[dict[str, Any]] = []
        if not plan_by_id:
            return payload
        fact_by_id = {fact["memory_id"]: fact for fact in sample["selected_complementary_facts"]}
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
        sample: dict[str, Any],
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

        conversation_text = " ".join(message["content"] for message in conversation)
        conversation_key = self._canonicalize_text(conversation_text)
        canonical_answer_key = self._canonicalize_text(sample["canonical_answer"])
        subtype = sample["complementary_subtype_label"]
        answer_bearing_roles = {"required", "equivalent"}
        has_answer_bearing_fact = any(fact["fact_role"] in answer_bearing_roles for fact in assigned_facts)
        if subtype in {"k_eq_1", "any_one_of_n"} and not has_answer_bearing_fact:
            if canonical_answer_key and canonical_answer_key in conversation_key:
                raise ValueError("A non-answer-bearing session leaked the canonical answer.")
        if subtype == "k_gt_1":
            required_ids = {
                fact["memory_id"]
                for fact in sample["selected_complementary_facts"]
                if fact["fact_role"] == "required"
            }
            if required_ids and not required_ids.issubset(assigned_ids):
                if canonical_answer_key and canonical_answer_key in conversation_key:
                    raise ValueError("A partial k>1 session leaked the full canonical answer.")

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

    def _canonicalize_text(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()

    def _validate_generated_question(self, question: str) -> None:
        if not re.search(r"[?？]$", question):
            raise ValueError("Generated question must end with a question mark.")

    def _validate_answer_candidate_quality(
        self,
        correct_answers: list[dict[str, str]],
        incorrect_answers: list[dict[str, str]],
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

    def _memory_item_for_prompt(self, memory_item: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "memory_id": memory_item.get("memory_id"),
            "role": memory_item.get("role"),
            "representation": memory_item.get("representation"),
        }
        if memory_item.get("representation") == "sub_qa":
            payload["subquestion"] = memory_item.get("subquestion")
            payload["subanswer_text"] = memory_item.get("subanswer_text") or stable_answer_text(memory_item.get("subanswer"))
            payload["depends_on"] = memory_item.get("depends_on", [])
            payload["depth"] = memory_item.get("depth")
        else:
            if "context_label" in memory_item:
                payload["context_label"] = memory_item.get("context_label")
            if "title" in memory_item:
                payload["title"] = memory_item.get("title")
            if "text" in memory_item:
                payload["text"] = clean_text(str(memory_item.get("text", "")))
            if "source" in memory_item:
                payload["source"] = memory_item.get("source")
            if "paragraph_idx" in memory_item:
                payload["paragraph_idx"] = memory_item.get("paragraph_idx")
        return payload
