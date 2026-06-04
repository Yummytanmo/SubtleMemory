from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import random
from pathlib import Path
from threading import Lock
from typing import Any

from infra.config import GENERATION_MODEL
from infra.logging_utils import utc_now_iso
from core.parallel_utils import map_ordered, normalize_concurrency
from prompts import (
    ACTIVE_CASE_RELATION_SUBTYPE_ORDER,
    ACTIVE_CASE_SPECS,
    CASE_RELATION_TYPE_ORDER,
    CASE_SPECS,
    PROMPT_VERSION,
    build_case_relation_plan_prompt,
    build_case_selection_prompt,
)
from core.schemas import (
    CaseRelationPlanItem,
    Fact,
    MemoryCase,
    PreferenceItem,
    SanitizedPersonaProfile,
    TopicPreferenceGroup,
)
from core.validators import parse_json_after_output


CASE_RELATION_CANDIDATE_COUNT = 4


def plan_case_relations(
    groups: list[TopicPreferenceGroup],
    personas: dict[str, SanitizedPersonaProfile],
    llm_client: Any,
    *,
    generation_model: str = GENERATION_MODEL,
    concurrency: int = 1,
    rng: random.Random | None = None,
) -> tuple[list[CaseRelationPlanItem], dict[str, Any]]:
    rng = rng or random.Random()
    jobs = build_case_relation_planning_jobs(groups, personas, rng)
    planned_batches = map_ordered(
        jobs,
        lambda job: generate_case_relation_plan_for_job(job, llm_client, generation_model=generation_model),
        max_workers=normalize_concurrency(concurrency),
    )
    plan_items = [item for batch in planned_batches for item in batch]
    report = build_case_relation_plan_report(plan_items, jobs)
    return plan_items, report


def generate_memory_cases(
    groups: list[TopicPreferenceGroup],
    personas: dict[str, SanitizedPersonaProfile],
    llm_client: Any,
    *,
    generation_model: str = GENERATION_MODEL,
    concurrency: int = 1,
    rng: random.Random | None = None,
    checkpoint_path: Path | None = None,
    logger: Any | None = None,
    relation_plan: list[CaseRelationPlanItem] | None = None,
) -> list[MemoryCase]:
    rng = rng or random.Random()
    jobs = build_case_generation_jobs(groups, personas, rng, relation_plan=relation_plan)
    checkpoint = CaseGenerationCheckpoint(checkpoint_path, jobs)
    cached_count = checkpoint.cached_count()
    if logger is not None and checkpoint_path is not None:
        logger.info(
            "case_generation",
            "case_generation_checkpoint_loaded",
            checkpoint_path=str(checkpoint_path),
            counts={"cached_cases": cached_count, "total_jobs": len(jobs)},
        )
    pending_jobs = [job for job in jobs if not checkpoint.has(job)]
    if pending_jobs:
        map_ordered(
            pending_jobs,
            lambda job: generate_and_checkpoint_case(job, llm_client, checkpoint, generation_model=generation_model),
            max_workers=normalize_concurrency(concurrency),
        )
    cases = checkpoint.ordered_cases()
    if len(cases) != len(jobs):
        raise RuntimeError(f"case generation checkpoint incomplete: {len(cases)}/{len(jobs)} cases")
    if logger is not None and checkpoint_path is not None:
        logger.info(
            "case_generation",
            "case_generation_checkpoint_complete",
            checkpoint_path=str(checkpoint_path),
            counts={"cached_cases": cached_count, "generated_cases": len(pending_jobs), "total_cases": len(cases)},
        )
    return cases


def generate_cases_for_group(
    group: TopicPreferenceGroup,
    persona: SanitizedPersonaProfile,
    llm_client: Any,
    *,
    generation_model: str = GENERATION_MODEL,
    concurrency: int = 1,
    rng: random.Random | None = None,
) -> list[MemoryCase]:
    rng = rng or random.Random()
    jobs = build_case_generation_jobs([group], {persona.persona_id: persona}, rng)
    return map_ordered(
        jobs,
        lambda job: generate_case_for_job(job, llm_client, generation_model=generation_model),
        max_workers=normalize_concurrency(concurrency),
    )


class CaseRelationPlanningJob:
    def __init__(
        self,
        *,
        persona: SanitizedPersonaProfile,
        preferences: list[PreferenceItem],
        target_distribution: list[dict[str, Any]],
    ) -> None:
        self.persona = persona
        self.preferences = preferences
        self.target_distribution = target_distribution


class CaseGenerationJob:
    def __init__(
        self,
        *,
        group: TopicPreferenceGroup,
        persona: SanitizedPersonaProfile,
        preference: PreferenceItem,
        preferred_relation: dict[str, str],
        candidate_relations: list[dict[str, str]],
    ) -> None:
        self.group = group
        self.persona = persona
        self.preference = preference
        self.preferred_relation = preferred_relation
        self.candidate_relations = candidate_relations


def build_case_relation_planning_jobs(
    groups: list[TopicPreferenceGroup],
    personas: dict[str, SanitizedPersonaProfile],
    rng: random.Random,
) -> list[CaseRelationPlanningJob]:
    preferences_by_persona = flatten_preferences_by_persona(groups)
    jobs: list[CaseRelationPlanningJob] = []
    for persona_id, preferences in preferences_by_persona.items():
        jobs.append(
            CaseRelationPlanningJob(
                persona=personas[persona_id],
                preferences=preferences,
                target_distribution=case_relation_target_distribution(len(preferences), rng),
            )
        )
    return jobs


def flatten_preferences_by_persona(groups: list[TopicPreferenceGroup]) -> dict[str, list[PreferenceItem]]:
    preferences_by_persona: dict[str, list[PreferenceItem]] = {}
    for group in groups:
        for preference in group.preferences:
            preferences_by_persona.setdefault(group.persona_id, []).append(preference)
    return preferences_by_persona


def generate_case_relation_plan_for_job(
    job: CaseRelationPlanningJob,
    llm_client: Any,
    *,
    generation_model: str = GENERATION_MODEL,
) -> list[CaseRelationPlanItem]:
    prompt = build_case_relation_plan_prompt(
        job.persona.persona_str,
        job.persona.persona_id,
        [preference.model_dump(mode="json") for preference in job.preferences],
        job.target_distribution,
    )
    result = llm_client.stream_chat(
        [{"role": "user", "content": prompt}],
        phase="case_relation_planning",
        temperature=0.1,
    )
    if not result.stream_completed:
        raise RuntimeError(f"case relation planning stream failed for {job.persona.persona_id}: {result.error}")
    payload = parse_json_after_output(result.text)
    return build_case_relation_plan_items(
        job.persona,
        job.preferences,
        payload,
        job.target_distribution,
        generation_model=generation_model,
        stream_completed=result.stream_completed,
    )


def build_case_relation_plan_items(
    persona: SanitizedPersonaProfile,
    preferences: list[PreferenceItem],
    payload: dict[str, Any],
    target_distribution: list[dict[str, Any]],
    *,
    generation_model: str = GENERATION_MODEL,
    stream_completed: bool = True,
) -> list[CaseRelationPlanItem]:
    raw_items = payload.get("planned_relations")
    if not isinstance(raw_items, list):
        raise ValueError("case relation plan must include planned_relations list")

    preferences_by_id = {preference.preference_id: preference for preference in preferences}
    if len(preferences_by_id) != len(preferences):
        raise ValueError(f"duplicate preference_id in relation planning input for {persona.persona_id}")

    expected_counts = case_relation_distribution_counter(target_distribution)
    if sum(expected_counts.values()) != len(preferences):
        raise ValueError("case relation target distribution count does not match preference count")

    plan_by_preference: dict[str, CaseRelationPlanItem] = {}
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            raise ValueError("each planned relation must be an object")
        preference_id = str(raw_item.get("preference_id") or "").strip()
        if preference_id not in preferences_by_id:
            raise ValueError(f"planned relation contains unknown preference_id: {preference_id}")
        if preference_id in plan_by_preference:
            raise ValueError(f"planned relation contains duplicate preference_id: {preference_id}")

        relation_type = str(raw_item.get("relation_type") or "").strip()
        relation_subtype = str(raw_item.get("relation_subtype") or "").strip()
        if (relation_type, relation_subtype) not in CASE_SPECS:
            raise ValueError(f"unknown planned relation {relation_type}/{relation_subtype} for {preference_id}")
        preference = preferences_by_id[preference_id]
        plan_by_preference[preference_id] = CaseRelationPlanItem(
            persona_id=persona.persona_id,
            preference_id=preference_id,
            topic_preference=preference.topic_preference,
            relation_type=relation_type,
            relation_subtype=relation_subtype,
            planning_reason=str(raw_item.get("planning_reason") or "").strip(),
            planning_model=generation_model,
            stream_completed=stream_completed,
            created_at=utc_now_iso(),
        )

    missing_ids = sorted(set(preferences_by_id) - set(plan_by_preference))
    if missing_ids:
        raise ValueError(f"case relation plan is missing preference IDs: {missing_ids}")

    actual_counts = Counter((str(item.relation_type), str(item.relation_subtype)) for item in plan_by_preference.values())
    if actual_counts != expected_counts:
        raise ValueError(
            "case relation plan does not match target distribution: "
            f"expected {dict(sorted(expected_counts.items()))}, actual {dict(sorted(actual_counts.items()))}"
        )

    return [plan_by_preference[preference.preference_id] for preference in preferences]


def build_case_relation_plan_report(
    plan_items: list[CaseRelationPlanItem],
    jobs: list[CaseRelationPlanningJob],
) -> dict[str, Any]:
    target_counts: Counter[tuple[str, str]] = Counter()
    target_type_counts: Counter[str] = Counter()
    for job in jobs:
        job_target_counts = case_relation_distribution_counter(job.target_distribution)
        target_counts.update(job_target_counts)
        target_type_counts.update(case_relation_type_counter(job_target_counts))
    planned_counts = Counter((str(item.relation_type), str(item.relation_subtype)) for item in plan_items)
    return {
        "personas": len(jobs),
        "planned_relation_items": len(plan_items),
        "target_relation_type_distribution": dict(sorted(target_type_counts.items())),
        "target_relation_subtype_distribution": format_relation_counter(target_counts),
        "planned_relation_type_distribution": dict(sorted(Counter(str(item.relation_type) for item in plan_items).items())),
        "planned_relation_subtype_distribution": dict(sorted(Counter(str(item.relation_subtype) for item in plan_items).items())),
        "planned_relation_pair_distribution": format_relation_counter(planned_counts),
    }


def build_case_generation_jobs(
    groups: list[TopicPreferenceGroup],
    personas: dict[str, SanitizedPersonaProfile],
    rng: random.Random,
    *,
    relation_plan: list[CaseRelationPlanItem] | None = None,
) -> list[CaseGenerationJob]:
    raw_jobs: list[tuple[TopicPreferenceGroup, SanitizedPersonaProfile, PreferenceItem]] = []
    indices_by_persona: dict[str, list[int]] = defaultdict(list)
    for group in groups:
        if not group.preferences:
            continue
        persona = personas[group.persona_id]
        for preference in group.preferences:
            indices_by_persona[group.persona_id].append(len(raw_jobs))
            raw_jobs.append((group, persona, preference))

    jobs: list[CaseGenerationJob | None] = [None] * len(raw_jobs)
    if relation_plan is not None:
        plan_by_key = case_relation_plan_by_key(relation_plan)
        expected_keys = {(preference.persona_id, preference.preference_id) for _, _, preference in raw_jobs}
        missing_keys = sorted(expected_keys - set(plan_by_key))
        extra_keys = sorted(set(plan_by_key) - expected_keys)
        if missing_keys or extra_keys:
            raise ValueError(f"case relation plan does not match generation jobs: missing={missing_keys}, extra={extra_keys}")
        for index, (group, persona, preference) in enumerate(raw_jobs):
            plan_item = plan_by_key[(preference.persona_id, preference.preference_id)]
            if plan_item.topic_preference != group.topic_preference:
                raise ValueError(f"case relation plan topic mismatch for {preference.preference_id}")
            planned_relation = relation_dict(str(plan_item.relation_type), str(plan_item.relation_subtype))
            jobs[index] = CaseGenerationJob(
                group=group,
                persona=persona,
                preference=preference,
                preferred_relation=planned_relation,
                candidate_relations=sample_case_relation_candidates(rng, preferred_relation=planned_relation),
            )
        return [job for job in jobs if job is not None]

    for persona_id, indices in indices_by_persona.items():
        preferred_relations = balanced_case_relation_preferences(len(indices), rng)
        for index, preferred_relation in zip(indices, preferred_relations):
            group, persona, preference = raw_jobs[index]
            if persona.persona_id != persona_id:
                raise ValueError("persona grouping mismatch while building case generation jobs")
            jobs[index] = CaseGenerationJob(
                group=group,
                persona=persona,
                preference=preference,
                preferred_relation=preferred_relation,
                candidate_relations=sample_case_relation_candidates(rng, preferred_relation=preferred_relation),
            )
    return [job for job in jobs if job is not None]


class CaseGenerationCheckpoint:
    def __init__(self, path: Path | None, jobs: list[CaseGenerationJob]) -> None:
        self.path = path
        self.job_keys = [case_generation_job_key(job) for job in jobs]
        self.cases_by_key = load_case_generation_checkpoint(path)
        self._lock = Lock()

    def has(self, job: CaseGenerationJob) -> bool:
        return case_generation_job_key(job) in self.cases_by_key

    def cached_count(self) -> int:
        return sum(1 for key in self.job_keys if key in self.cases_by_key)

    def record(self, job: CaseGenerationJob, memory_case: MemoryCase) -> None:
        key = case_generation_job_key(job)
        with self._lock:
            self.cases_by_key[key] = memory_case
            self.flush_locked()

    def ordered_cases(self) -> list[MemoryCase]:
        return [
            self.cases_by_key[key]
            for key in self.job_keys
            if key in self.cases_by_key
        ]

    def flush_locked(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [case.model_dump(mode="json") for case in self.ordered_cases()]
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temp_path.replace(self.path)


def load_case_generation_checkpoint(path: Path | None) -> dict[str, MemoryCase]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"case generation checkpoint must be a JSON array: {path}")
    cases: dict[str, MemoryCase] = {}
    for item in payload:
        memory_case = MemoryCase.model_validate(item)
        key = case_generation_case_key(memory_case)
        if key not in cases:
            cases[key] = memory_case
    return cases


def generate_and_checkpoint_case(
    job: CaseGenerationJob,
    llm_client: Any,
    checkpoint: CaseGenerationCheckpoint,
    *,
    generation_model: str = GENERATION_MODEL,
) -> MemoryCase:
    memory_case = generate_case_for_job(job, llm_client, generation_model=generation_model)
    checkpoint.record(job, memory_case)
    return memory_case


def generate_case_for_job(
    job: CaseGenerationJob,
    llm_client: Any,
    *,
    generation_model: str = GENERATION_MODEL,
) -> MemoryCase:
    prompt = build_case_selection_prompt(
        job.persona.persona_str,
        job.preference.model_dump(mode="json"),
        job.group.topic_preference,
        job.candidate_relations,
        preferred_relation=job.preferred_relation,
    )
    result = llm_client.stream_chat(
        [{"role": "user", "content": prompt}],
        phase="case_generation",
        temperature=0.2,
    )
    if not result.stream_completed:
        raise RuntimeError(f"case generation stream failed for {job.group.topic_id}: {result.error}")
    payload = parse_json_after_output(result.text)
    relation_type = str(payload.get("relation_type", "")).strip()
    relation_subtype = str(payload.get("relation_subtype", "")).strip()
    ensure_selected_relation_is_candidate(relation_type, relation_subtype, job.candidate_relations)
    return build_memory_case(
        payload,
        persona_id=job.group.persona_id,
        topic_preference=job.group.topic_preference,
        source_preference_id=job.preference.preference_id,
        relation_type=relation_type,
        relation_subtype=relation_subtype,
        preferred_relation=job.preferred_relation,
        candidate_relations=job.candidate_relations,
        generation_model=generation_model,
    )


def case_generation_job_key(job: CaseGenerationJob) -> str:
    return "|".join(
        [
            job.group.persona_id,
            job.group.topic_preference,
            job.preference.preference_id,
            job.preferred_relation["relation_type"],
            job.preferred_relation["relation_subtype"],
        ]
    )


def case_generation_case_key(memory_case: MemoryCase) -> str:
    relation_type = str(getattr(memory_case, "preferred_relation_type", None) or memory_case.relation_type)
    relation_subtype = str(getattr(memory_case, "preferred_relation_subtype", None) or memory_case.relation_subtype)
    return "|".join(
        [
            memory_case.persona_id,
            memory_case.topic_preference,
            memory_case.source_preference_id,
            relation_type,
            relation_subtype,
        ]
    )


def case_relation_target_distribution(job_count: int, rng: random.Random) -> list[dict[str, Any]]:
    del rng
    counts = case_relation_pair_target_counts(job_count)
    return [
        {
            "relation_type": relation_type,
            "relation_subtype": relation_subtype,
            "target_count": counts[(relation_type, relation_subtype)],
        }
        for relation_type, relation_subtype in ACTIVE_CASE_SPECS
    ]


def case_relation_distribution_counter(target_distribution: list[dict[str, Any]]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for item in target_distribution:
        relation_type = str(item.get("relation_type") or "").strip()
        relation_subtype = str(item.get("relation_subtype") or "").strip()
        if (relation_type, relation_subtype) not in CASE_SPECS:
            raise ValueError(f"unknown target relation {relation_type}/{relation_subtype}")
        target_count = int(item.get("target_count") or 0)
        if target_count < 0:
            raise ValueError("target_count must be non-negative")
        if target_count:
            counts[(relation_type, relation_subtype)] += target_count
    return counts


def format_relation_counter(counts: Counter[tuple[str, str]]) -> dict[str, int]:
    return {f"{relation_type}/{relation_subtype}": count for (relation_type, relation_subtype), count in sorted(counts.items())}


def case_relation_type_counter(counts: Counter[tuple[str, str]]) -> Counter[str]:
    type_counts: Counter[str] = Counter()
    for (relation_type, _), count in counts.items():
        if count:
            type_counts[relation_type] += count
    return type_counts


def case_relation_plan_by_key(
    relation_plan: list[CaseRelationPlanItem],
) -> dict[tuple[str, str], CaseRelationPlanItem]:
    plan_by_key: dict[tuple[str, str], CaseRelationPlanItem] = {}
    for item in relation_plan:
        key = (item.persona_id, item.preference_id)
        if key in plan_by_key:
            raise ValueError(f"duplicate case relation plan item for {key[0]}/{key[1]}")
        plan_by_key[key] = item
    return plan_by_key


def balanced_case_relation_preferences(job_count: int, rng: random.Random) -> list[dict[str, str]]:
    if job_count <= 0:
        return []
    target_counts = case_relation_pair_target_counts(job_count)
    preferred_specs: list[tuple[str, str]] = []
    for relation_type, relation_subtype in ACTIVE_CASE_SPECS:
        preferred_specs.extend([(relation_type, relation_subtype)] * target_counts[(relation_type, relation_subtype)])
    rng.shuffle(preferred_specs)
    return [relation_dict(relation_type, relation_subtype) for relation_type, relation_subtype in preferred_specs]


def case_relation_pair_target_counts(job_count: int) -> Counter[tuple[str, str]]:
    if job_count <= 0:
        return Counter()
    counts: Counter[tuple[str, str]] = Counter()
    type_counts = _balanced_counts(job_count, CASE_RELATION_TYPE_ORDER)
    for relation_type in CASE_RELATION_TYPE_ORDER:
        subtype_counts = _balanced_counts(type_counts[relation_type], ACTIVE_CASE_RELATION_SUBTYPE_ORDER[relation_type])
        for relation_subtype, count in subtype_counts.items():
            if count:
                counts[(relation_type, relation_subtype)] = count
    return counts


def _balanced_counts(total: int, ordered_labels: tuple[str, ...]) -> dict[str, int]:
    base_count, remainder = divmod(total, len(ordered_labels))
    counts = {label: base_count for label in ordered_labels}
    for label in ordered_labels[:remainder]:
        counts[label] += 1
    return counts


def sample_case_relation_candidates(
    rng: random.Random,
    preferred_relation: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    if preferred_relation is None:
        sampled = rng.sample(ACTIVE_CASE_SPECS, CASE_RELATION_CANDIDATE_COUNT)
        return [relation_dict(relation_type, relation_subtype) for relation_type, relation_subtype in sampled]

    preferred_spec = relation_tuple(preferred_relation)
    if preferred_spec not in CASE_SPECS:
        raise ValueError(f"unknown preferred relation {preferred_spec[0]}/{preferred_spec[1]}")
    return [relation_dict(preferred_spec[0], preferred_spec[1])]


def relation_dict(relation_type: str, relation_subtype: str) -> dict[str, str]:
    return {"relation_type": relation_type, "relation_subtype": relation_subtype}


def relation_tuple(relation: dict[str, str]) -> tuple[str, str]:
    return str(relation["relation_type"]), str(relation["relation_subtype"])


def ensure_selected_relation_is_candidate(
    relation_type: str,
    relation_subtype: str,
    candidates: list[dict[str, str]],
) -> None:
    if not any(
        relation_type == candidate["relation_type"] and relation_subtype == candidate["relation_subtype"]
        for candidate in candidates
    ):
        raise ValueError(f"selected relation {relation_type}/{relation_subtype} is not in sampled candidates")


def build_memory_case(
    payload: dict[str, Any],
    *,
    persona_id: str,
    topic_preference: str,
    source_preference_id: str,
    relation_type: str,
    relation_subtype: str,
    preferred_relation: dict[str, str] | None = None,
    candidate_relations: list[dict[str, str]] | None = None,
    generation_model: str = GENERATION_MODEL,
) -> MemoryCase:
    description = str(payload.get("description", "")).strip()
    raw_facts = payload.get("facts", [])
    if isinstance(raw_facts, str):
        raw_facts = [raw_facts]
    case_id = make_case_id(persona_id, topic_preference, source_preference_id, relation_type, relation_subtype, description)
    facts = [
        Fact(
            fact_id=f"{case_id}-fact-{idx}",
            case_id=case_id,
            text=_fact_text(item),
            order=idx,
        )
        for idx, item in enumerate(raw_facts)
    ]
    return MemoryCase(
        case_id=case_id,
        persona_id=persona_id,
        topic_preference=topic_preference,
        source_preference_id=source_preference_id,
        relation_type=relation_type,
        relation_subtype=relation_subtype,
        description=description,
        facts=facts,
        generation_model=generation_model,
        prompt_version=PROMPT_VERSION,
        created_at=utc_now_iso(),
        preferred_relation_type=preferred_relation["relation_type"] if preferred_relation else None,
        preferred_relation_subtype=preferred_relation["relation_subtype"] if preferred_relation else None,
        candidate_relations=candidate_relations or [],
        relation_choice_reason=str(payload.get("relation_choice_reason", "")).strip(),
        preferred_relation_used=(
            relation_tuple(preferred_relation) == (relation_type, relation_subtype) if preferred_relation else False
        ),
    )


def make_case_id(
    persona_id: str,
    topic_preference: str,
    source_preference_id: str,
    relation_type: str,
    relation_subtype: str,
    description: str,
) -> str:
    digest = hashlib.sha1(
        f"{persona_id}|{topic_preference}|{source_preference_id}|{relation_type}|{relation_subtype}|{description}".encode(
            "utf-8"
        )
    ).hexdigest()[:14]
    return f"case-{digest}"


def _fact_text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("text") or item.get("fact") or "").strip()
    return str(item).strip()
