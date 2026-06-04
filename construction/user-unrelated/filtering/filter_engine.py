from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import time
from typing import Any

from infra.json_utils import clean_text, extract_json_from_response
from infra.llm_client import LLMClient


class FilterEngine:
    def __init__(self, root_dir: str | Path, config: dict[str, Any]):
        self.root_dir = Path(root_dir)
        self.llm = LLMClient(config)
        self.prompt_modules = {
            "complementary": self._load_prompt_module("complementary"),
            "contradictory": self._load_prompt_module("contradictory"),
            "nuanced": self._load_prompt_module("nuanced"),
        }

    def _load_prompt_module(self, category: str):
        module_path = self.root_dir / "filtering" / "prompts" / f"{category}.py"
        spec = spec_from_file_location(f"filter_prompts_{category}", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load filter prompt module for category {category} from {module_path}")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def filter_sample(self, category: str, sample: dict[str, Any]) -> dict[str, Any]:
        if category not in self.prompt_modules:
            raise ValueError(f"Unsupported category for filtering: {category}")
        prompt_module = self.prompt_modules[category]

        conversation_result = self._run_stage(
            prompt_module.CONVERSATION_FILTER_SYSTEM_PROMPT,
            prompt_module.build_conversation_filter_prompt(sample),
        )
        if conversation_result["decision"] != "yes":
            return {
                "overall_pass": False,
                "conversation": conversation_result,
                "question": {"decision": "skipped", "reason": "Skipped because conversation filter failed."},
                "answer": {"decision": "skipped", "reason": "Skipped because conversation filter failed."},
            }

        if self._use_combined_qa_filter(prompt_module, sample):
            qa_result = self._run_stage(
                prompt_module.QA_FILTER_SYSTEM_PROMPT,
                prompt_module.build_qa_filter_prompt(sample),
            )
            overall_pass = qa_result["decision"] == "yes"
            return {
                "overall_pass": overall_pass,
                "conversation": conversation_result,
                "question": qa_result,
                "answer": qa_result,
                "qa": qa_result,
            }

        question_result = self._run_stage(
            prompt_module.QUESTION_FILTER_SYSTEM_PROMPT,
            prompt_module.build_question_filter_prompt(sample),
        )
        if question_result["decision"] != "yes":
            return {
                "overall_pass": False,
                "conversation": conversation_result,
                "question": question_result,
                "answer": {"decision": "skipped", "reason": "Skipped because question filter failed."},
            }

        answer_result = self._run_stage(
            prompt_module.ANSWER_FILTER_SYSTEM_PROMPT,
            prompt_module.build_answer_filter_prompt(sample),
        )
        overall_pass = answer_result["decision"] == "yes"
        return {
            "overall_pass": overall_pass,
            "conversation": conversation_result,
            "question": question_result,
            "answer": answer_result,
        }

    def _use_combined_qa_filter(self, prompt_module: Any, sample: dict[str, Any]) -> bool:
        return (
            hasattr(prompt_module, "QA_FILTER_SYSTEM_PROMPT")
            and hasattr(prompt_module, "build_qa_filter_prompt")
            and isinstance(sample.get("qa_pairs"), list)
            and bool(sample.get("qa_pairs"))
        )

    def _run_stage(self, system_prompt: str, user_prompt: str) -> dict[str, str]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        last_error: Exception | None = None
        add_correction_prompt = False
        max_attempts = 5
        for attempt in range(max_attempts):
            if add_correction_prompt:
                messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            f"The previous output was invalid: {last_error}. "
                            "Return corrected JSON only with fields decision and reason."
                        ),
                    }
                ]
                add_correction_prompt = False
            try:
                raw = self.llm.chat(messages)
            except Exception as exc:
                last_error = exc
                if attempt == max_attempts - 1:
                    raise
                time.sleep(min(2 ** attempt, 20))
                continue
            try:
                parsed = extract_json_from_response(raw)
                return self._validate_stage_result(parsed)
            except Exception as exc:
                last_error = exc
                add_correction_prompt = True
        raise ValueError(f"Failed to obtain a valid filter judgment: {last_error}")

    def _validate_stage_result(self, data: Any) -> dict[str, str]:
        if not isinstance(data, dict):
            raise ValueError("Filter judgment must be a JSON object.")
        decision = clean_text(str(data.get("decision", ""))).casefold()
        reason = clean_text(str(data.get("reason", "")))
        if decision not in {"yes", "no"}:
            raise ValueError("Filter judgment decision must be yes or no.")
        if not reason:
            raise ValueError("Filter judgment reason must be non-empty.")
        return {"decision": decision, "reason": reason}
