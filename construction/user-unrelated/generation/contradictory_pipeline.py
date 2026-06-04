from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from generation.prompts import contradictory as prompts
from infra.json_utils import clean_text, ensure_dir, extract_json_from_response, load_json, make_seeded_rng
from infra.llm_client import LLMClient


SOURCE_FILES = {
    "contradictory_source": "source_data/contradictory/contradictory_source.json",
}


DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "sports": (
        "nba",
        "nfl",
        "ncaa",
        "ligue",
        "psg",
        "football",
        "basketball",
        "soccer",
        "baseball",
        "championship",
        "mvp",
        "touchdown",
        "playoff",
        "world cup",
        "fifa",
        "team",
        "season",
        "tournament",
        "super bowl",
        "passing",
    ),
    "entertainment": (
        "movie",
        "film",
        "voice",
        "star wars",
        "jungle book",
        "actor",
        "actress",
        "episode",
        "series",
        "show",
        "broadway",
        "screen",
        "premiere",
        "tv",
        "character",
    ),
    "music": (
        "song",
        "sing",
        "sang",
        "singer",
        "album",
        "music",
        "record",
        "recorded",
        "karaoke",
        "playlist",
    ),
    "software_tech": (
        "adobe",
        "dreamweaver",
        "software",
        "version",
        "tool",
        "app",
        "install",
        "update",
        "release",
        "browser",
        "windows",
        "iphone",
        "android",
    ),
    "study_definition": (
        "means what",
        "mean",
        "definition",
        "creed",
        "consubstantial",
        "translation",
        "translations",
        "term",
        "phrase",
    ),
    "travel_geo_history": (
        "where",
        "location",
        "located",
        "capital",
        "country",
        "state",
        "city",
        "travel",
        "empire",
        "held",
        "founded",
        "india",
        "population",
    ),
}


CONFLICT_ACK_MARKERS = (
    "conflict",
    "conflicts",
    "contradict",
    "contradiction",
    "disagree",
    "unresolved",
    "unclear",
    "clarif",
    "can't give",
    "cannot give",
    "need to know",
    "need clarification",
    "which one",
    "which version",
    "which film",
)


CANONICAL_QUESTION_DISALLOWED_MARKERS = (
    "as of ",
    " tv series",
    " television series",
    "live-action",
    "animated version",
    " remake",
    " reboot",
    "classic version",
)


@dataclass
class GenerationResult:
    record: dict[str, Any]


class ContradictoryPipeline:
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
                "Contradictory generation config must define non-empty CONVERSATION_TYPE_DESCRIPTIONS "
                "and CONVERSATION_TYPE_FLOW_DESCRIPTIONS."
            )
        ensure_dir(self.root_dir / self.config["paths"]["output_dir"])

    def _shared_conversation_generation_cfg(self) -> dict[str, Any]:
        explicit_cfg_path = self.generation_cfg.get("conversation_type_config_path")
        if not explicit_cfg_path:
            return {}
        data = load_json(self.root_dir / str(explicit_cfg_path))
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object in conversation type config: {explicit_cfg_path}")
        return data

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
                raise ValueError(
                    "Each generation.CONVERSATION_TYPE_FLOW_DESCRIPTIONS entry must be a list of strings."
                )
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
            for idx, item in enumerate(data):
                samples.append(self.normalize_sample(item, sample_index=idx, source_path=path))
        return samples

    def normalize_sample(self, item: dict[str, Any], sample_index: int, source_path: Path) -> dict[str, Any]:
        qa_pair_bag = item["annotations"]["qaPairs"][0]
        raw_questions = qa_pair_bag["question"]
        raw_answers = qa_pair_bag["answer"]
        if len(raw_questions) != len(raw_answers):
            raise ValueError(f"Mismatched contradictory QA lengths for source id {item.get('id')}")

        memory_items: list[dict[str, Any]] = []
        for idx, (question, answers) in enumerate(zip(raw_questions, raw_answers), start=1):
            aliases = [clean_text(str(v)) for v in answers]
            answer_text = clean_text(str(answers[0] if answers else ""))
            if not answer_text:
                raise ValueError(f"Empty answer for source id {item.get('id')} at position {idx}")
            memory_items.append(
                {
                    "memory_id": f"ctr-src-{sample_index + 1:05d}-m{idx:02d}",
                    "source_subquestion": clean_text(question),
                    "answer_aliases": aliases,
                    "answer_text": answer_text,
                }
            )

        prompt_payload = {
            "source_dataset": "ContradictorySource",
            "source_id": item.get("id"),
            "ambiguous_question": clean_text(item["question"]),
            "sub_qa_pairs": memory_items,
        }

        return {
            "sample_id": f"ctr-src-{sample_index + 1:05d}",
            "source_dataset": "ContradictorySource",
            "source_path": str(source_path),
            "source_question": clean_text(item["question"]),
            "memory_items": memory_items,
            "prompt_payload": prompt_payload,
            "source_record": item,
        }

    def materialize_subtype_sample(self, base_sample: dict[str, Any], subtype: str) -> dict[str, Any]:
        subtype_short = {
            "a_user_vs_user": "a",
            "b_user_vs_non_user": "b",
            "c_non_user_vs_non_user": "c",
        }[subtype]
        return {
            **base_sample,
            "sample_id": f"ctr-{subtype_short}-{base_sample['sample_id'].split('-')[-1]}",
            "relationship": "contradictory",
            "task_family": "contradictory",
            "contradictory_subtype": subtype,
        }

    def _roles_for_subtype(self, subtype: str) -> list[str]:
        if subtype == "a_user_vs_user":
            return ["user", "user"]
        if subtype == "b_user_vs_non_user":
            return ["user", "assistant"]
        if subtype == "c_non_user_vs_non_user":
            return ["assistant", "assistant"]
        raise ValueError(f"Unknown contradictory subtype: {subtype}")

    def _infer_domain(self, sample: dict[str, Any], target_fact: dict[str, Any]) -> str:
        text = " ".join(
            [
                sample["source_question"],
                target_fact["source_subquestion"],
                target_fact["answer_text"],
            ]
        ).casefold()
        text = re.sub(r"\s+", " ", text)
        best_domain = "generic"
        best_score = 0
        for domain, keywords in DOMAIN_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > best_score:
                best_domain = domain
                best_score = score
        return best_domain

    def _conversation_candidates_for_fact(self, target_fact: dict[str, Any]) -> dict[str, Any]:
        rng = make_seeded_rng(
            self.generation_cfg["seed"],
            f"conversation-type-candidates:{target_fact['session_id']}:{target_fact['memory_id']}",
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
        contexts: list[dict[str, Any]] = []

        for target_fact in sample["selected_conflicting_facts"]:
            rng = make_seeded_rng(
                self.generation_cfg["seed"],
                f"session-plan-context:{sample['sample_id']}:{target_fact['session_id']}",
            )
            domain = self._infer_domain(sample, target_fact)
            contexts.append(
                {
                    "session_id": target_fact["session_id"],
                    "fact_source_role": target_fact["fact_source_role"],
                    "domain": domain,
                    "conflict_question": sample["conflict_question"],
                    "target_answer_text": target_fact["answer_text"],
                    "conversation_candidates": self._conversation_candidates_for_fact(target_fact),
                }
            )
        return contexts

    def generate_sample(self, sample: dict[str, Any]) -> GenerationResult:
        sample = self.generate_conflict_setup(sample)
        session_plans = self.generate_session_plans(sample)
        session_bundles = self.generate_sessions(sample, session_plans)
        question_bundle = self.generate_question(sample, session_bundles)
        answer_bundle = self.generate_answers(sample, session_bundles, question_bundle["question"])

        record = {
            "sample_id": sample["sample_id"],
            "relationship": "contradictory",
            "task_family": "contradictory",
            "contradictory_subtype": sample["contradictory_subtype"],
            "source_dataset": sample["source_dataset"],
            "canonical_conflict_question": sample["conflict_question"],
            "selected_conflicting_facts": sample["selected_conflicting_facts"],
            "session_plans": session_plans,
            "sessions": [
                {
                    "session_id": fact["session_id"],
                    "fact_source_role": fact["fact_source_role"],
                    "session_scenario": bundle["chosen_scenario"],
                    "conversation": bundle["conversation"],
                }
                for fact, bundle in zip(sample["selected_conflicting_facts"], session_bundles)
            ],
            "question": question_bundle["question"],
            "correct_answers": answer_bundle["correct_answers"],
            "incorrect_answers": answer_bundle["incorrect_answers"],
            "metadata": {
                "source_question": sample["source_question"],
                "fact_selection_rationale": sample["conflict_rationale"],
                "memory_items": sample["memory_items"],
                "session_coverage_notes": [
                    {
                        "session_id": fact["session_id"],
                        "coverage_notes": bundle.get("coverage_notes", []),
                    }
                    for fact, bundle in zip(sample["selected_conflicting_facts"], session_bundles)
                ],
                "source_record": sample["source_record"],
            },
        }
        return GenerationResult(record=record)

    def generate_conflict_setup(self, sample: dict[str, Any]) -> dict[str, Any]:
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
                            "Return corrected JSON only. Select exactly two source entries, create one canonical conflict question, "
                            "and make sure the result is contradictory rather than temporal/contextual nuance."
                        ),
                    }
                ]
            raw = self.llm.chat(messages)
            try:
                parsed = extract_json_from_response(raw)
                return self._validate_conflict_setup(parsed, sample)
            except Exception as exc:
                last_error = exc
        raise ValueError(f"Failed to generate valid conflict setup for {sample['sample_id']}: {last_error}")

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
                            "Return corrected JSON only with exactly one session plan per selected fact, "
                            "make sure each plan chooses one of its provided conversation candidates exactly, "
                            "and make sure the plans describe clearly different scenarios and different events."
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
        jobs = list(zip(sample["selected_conflicting_facts"], session_plans))
        if len(jobs) <= 1:
            return [self.generate_session(sample, target_fact, session_plan) for target_fact, session_plan in jobs]

        bundles: list[dict[str, Any] | None] = [None] * len(jobs)
        max_workers = min(len(jobs), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self.generate_session, sample, target_fact, session_plan): idx
                for idx, (target_fact, session_plan) in enumerate(jobs)
            }
            for future, idx in future_to_index.items():
                bundles[idx] = future.result()

        return [bundle for bundle in bundles if bundle is not None]

    def generate_session(
        self,
        sample: dict[str, Any],
        target_fact: dict[str, Any],
        session_plan: dict[str, Any],
    ) -> dict[str, Any]:
        min_rounds = int(self.generation_cfg["conversation_round_min"])
        max_rounds = int(self.generation_cfg["conversation_round_max"])
        prompt = prompts.build_session_prompt(
            sample,
            target_fact,
            session_plan,
            min_rounds,
            max_rounds,
        )
        messages = [
            {"role": "system", "content": prompts.CONVERSATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        last_error: Exception | None = None
        for attempt in range(3):
            if attempt > 0:
                retry_note = (
                    f"The previous output was invalid: {last_error}. "
                    f"Return corrected JSON only. The session must contain at least {min_rounds} rounds "
                    f"and should stay around {min_rounds}-{max_rounds} rounds. It must strictly alternate user and assistant."
                )
                messages = messages + [{"role": "user", "content": retry_note}]
            raw = self.llm.chat(messages)
            try:
                parsed = extract_json_from_response(raw)
                self._validate_session_bundle(parsed, min_rounds, max_rounds)
                return parsed
            except Exception as exc:
                last_error = exc
        raise ValueError(f"Failed to generate a valid session for {sample['sample_id']} {target_fact['session_id']}: {last_error}")

    def generate_question(
        self,
        sample: dict[str, Any],
        session_bundles: list[dict[str, Any]],
    ) -> dict[str, str]:
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
                            "Return corrected JSON only with one non-empty 'question' field."
                        ),
                    }
                ]
            raw = self.llm.chat(messages)
            try:
                parsed = extract_json_from_response(raw)
                question = clean_text(str(parsed.get("question", "")))
                if not question:
                    raise ValueError("Generated question is empty.")
                self._validate_generated_question(question, sample["conflict_question"])
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

        expected_facts = {fact["session_id"]: fact for fact in sample["selected_conflicting_facts"]}
        context_by_id = {context["session_id"]: context for context in session_plan_context}
        if len(raw_plans) != len(expected_facts):
            raise ValueError(
                f"Expected {len(expected_facts)} session plans, got {len(raw_plans)}."
            )

        normalized_by_id: dict[str, dict[str, str]] = {}
        required_fields = [
            "chosen_conversation_type",
            "chosen_conversation_flow",
            "scenario_label",
            "event_signature",
            "event_summary",
            "opening_situation",
            "user_goal",
            "assistant_role",
            "fact_integration_plan",
            "distinct_from_other_session",
        ]
        for raw_plan in raw_plans:
            if not isinstance(raw_plan, dict):
                raise ValueError("Each session plan must be a JSON object.")
            session_id = clean_text(str(raw_plan.get("session_id", "")))
            if session_id not in expected_facts:
                raise ValueError(f"Unexpected session_id in session plans: {session_id!r}")
            if session_id in normalized_by_id:
                raise ValueError(f"Duplicate session plan for {session_id}.")

            normalized_plan = {"session_id": session_id}
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
            normalized_plan["fact_source_role"] = expected_facts[session_id]["fact_source_role"]
            normalized_plan["chosen_candidate_tier"] = candidate["candidate_tier"]
            normalized_plan["chosen_conversation_type_description"] = candidate["type_description"]
            normalized_plan["preferred_candidate"] = context["conversation_candidates"]["preferred_candidate"]
            normalized_plan["fallback_candidates"] = context["conversation_candidates"]["fallback_candidates"]
            normalized_by_id[session_id] = normalized_plan

        ordered_plans = [normalized_by_id[fact["session_id"]] for fact in sample["selected_conflicting_facts"]]

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

    def _validate_conflict_setup(self, bundle: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(bundle, dict):
            raise ValueError("Conflict setup must be a JSON object.")

        conflict_question = clean_text(str(bundle.get("conflict_question", "")))
        if not conflict_question:
            raise ValueError("Conflict setup must include a non-empty conflict_question.")
        self._validate_canonical_conflict_question(conflict_question)

        raw_facts = bundle.get("selected_facts")
        if not isinstance(raw_facts, list):
            raise ValueError("Conflict setup must include a selected_facts list.")
        if len(raw_facts) != 2:
            raise ValueError(f"Conflict setup must select exactly 2 facts, got {len(raw_facts)}.")

        memory_by_id = {item["memory_id"]: item for item in sample["memory_items"]}
        roles = self._roles_for_subtype(sample["contradictory_subtype"])
        selected_facts: list[dict[str, Any]] = []
        seen_memory_ids: set[str] = set()
        seen_answers: set[str] = set()

        for idx, (raw_fact, role) in enumerate(zip(raw_facts, roles), start=1):
            if not isinstance(raw_fact, dict):
                raise ValueError("Each selected fact must be a JSON object.")
            memory_id = clean_text(str(raw_fact.get("memory_id", "")))
            if memory_id not in memory_by_id:
                raise ValueError(f"Unknown memory_id in selected_facts: {memory_id!r}")
            if memory_id in seen_memory_ids:
                raise ValueError("selected_facts must use two different source memory items.")

            source_memory = memory_by_id[memory_id]
            contradictory_answer = clean_text(str(raw_fact.get("contradictory_answer", "")))
            if not contradictory_answer:
                raise ValueError(f"selected_facts entry for {memory_id} is missing contradictory_answer.")
            if not self._answer_matches_source_aliases(contradictory_answer, source_memory["answer_aliases"]):
                raise ValueError(
                    f"contradictory_answer {contradictory_answer!r} is not grounded in aliases for {memory_id}."
                )

            answer_key = self._canonicalize_text(contradictory_answer)
            if answer_key in seen_answers:
                raise ValueError("selected_facts must contain two distinct contradictory answers.")

            selected_facts.append(
                {
                    "session_id": f"s{idx}",
                    "memory_id": memory_id,
                    "source_subquestion": source_memory["source_subquestion"],
                    "source_answer_text": source_memory["answer_text"],
                    "answer_aliases": source_memory["answer_aliases"],
                    "answer_text": contradictory_answer,
                    "fact_source_role": role,
                }
            )
            seen_memory_ids.add(memory_id)
            seen_answers.add(answer_key)

        conflict_rationale = clean_text(str(bundle.get("conflict_rationale", "")))
        if not conflict_rationale:
            raise ValueError("Conflict setup must include a non-empty conflict_rationale.")

        return {
            **sample,
            "conflict_question": conflict_question,
            "conflict_rationale": conflict_rationale,
            "selected_conflicting_facts": selected_facts,
        }

    def _session_payload(self, sample: dict[str, Any], session_bundles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for fact, bundle in zip(sample["selected_conflicting_facts"], session_bundles):
            payload.append(
                {
                    "session_id": fact["session_id"],
                    "fact_source_role": fact["fact_source_role"],
                    "chosen_scenario": bundle["chosen_scenario"],
                    "conversation": bundle["conversation"],
                }
            )
        return payload

    def _canonicalize_text(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()

    def _answer_matches_source_aliases(self, answer_text: str, aliases: list[str]) -> bool:
        answer_key = self._canonicalize_text(answer_text)
        alias_keys = {self._canonicalize_text(alias) for alias in aliases}
        return answer_key in alias_keys

    def _validate_canonical_conflict_question(self, question: str) -> None:
        lowered = question.casefold()
        if re.search(r"\b(?:19|20)\d{2}\b", question):
            raise ValueError("Canonical conflict question still contains an explicit year qualifier.")
        if any(marker in lowered for marker in CANONICAL_QUESTION_DISALLOWED_MARKERS):
            raise ValueError("Canonical conflict question still contains qualifier markers that explain away the contradiction.")

    def _validate_generated_question(self, question: str, canonical_conflict_question: str) -> None:
        if not question:
            raise ValueError("Generated question is empty.")
        if not re.search(r"[?？]$", question):
            raise ValueError("Generated question must end with a question mark.")
        if not re.search(r"\b(?:19|20)\d{2}\b", canonical_conflict_question) and re.search(r"\b(?:19|20)\d{2}\b", question):
            raise ValueError("Generated question reintroduced an explicit year qualifier.")

    def _validate_session_bundle(self, bundle: dict[str, Any], min_rounds: int, max_rounds: int) -> None:
        if not isinstance(bundle, dict):
            raise ValueError("Session bundle must be a JSON object.")
        chosen_scenario = clean_text(str(bundle.get("chosen_scenario", "")))
        if not chosen_scenario:
            raise ValueError("Session bundle must include a non-empty chosen_scenario.")
        bundle["chosen_scenario"] = chosen_scenario
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

    def _validate_answer_candidate_quality(
        self,
        correct_answers: list[dict[str, str]],
        incorrect_answers: list[dict[str, str]],
    ) -> None:
        def canonicalize(text: str) -> str:
            return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()

        correct_texts = [candidate["text"] for candidate in correct_answers]
        incorrect_texts = [candidate["text"] for candidate in incorrect_answers]
        if len({canonicalize(text) for text in correct_texts}) != len(correct_texts):
            raise ValueError("Correct answers must be distinct.")
        if len({canonicalize(text) for text in incorrect_texts}) != len(incorrect_texts):
            raise ValueError("Incorrect answers must be distinct.")
        if {canonicalize(text) for text in correct_texts} & {canonicalize(text) for text in incorrect_texts}:
            raise ValueError("Correct and incorrect answers must not overlap.")
        if not any(any(marker in text.casefold() for marker in CONFLICT_ACK_MARKERS) for text in correct_texts):
            raise ValueError("At least one correct answer must explicitly acknowledge the contradiction or need for clarification.")
