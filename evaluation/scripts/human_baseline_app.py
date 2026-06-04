#!/usr/bin/env python3
"""Local web app for a human SubtleMemory persona0 baseline.

The app presents each question with the same oracle-scoped session transcript
used by ``OracleContextAdapter`` and exports standard evaluation artifacts:
``search_results.json`` and ``answer_results.json``. The exported answers can
then be judged by the existing ``evaluation.cli --stages evaluate`` path.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.parse import parse_qs, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = PROJECT_ROOT / "evaluation"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.src.adapters.baselines.oracle_context_adapter import OracleContextAdapter
from evaluation.src.core.data_models import Dataset, QAPair, SearchResult
from evaluation.src.core.loaders import load_dataset


DEFAULT_DATASET = "subtlememory"
DEFAULT_DATA_DIR = EVALUATION_ROOT / "data" / DEFAULT_DATASET
DEFAULT_RUN_NAME = "persona0-human-oracle-sessions"
DEFAULT_OUTPUT_ROOT = (
    EVALUATION_ROOT / "results" / "human_baseline_persona0" / DEFAULT_RUN_NAME
)
DEFAULT_SYSTEM = "human-baseline-oracle-sessions"
DEFAULT_SEARCH_MODE = "api"
FORBIDDEN_CLIENT_KEYS = {
    "golden_answer",
    "correct_answers",
    "incorrect_answers",
    "facts",
    "case",
    "relation_type",
    "relation_subtype",
    "topic",
    "source",
    "judge_detail",
    "evaluation",
}


@dataclass(frozen=True)
class AnnotationItem:
    index: int
    question_id: str
    conversation_id: str
    question: str
    answer_query: str
    golden_answer: str
    category: Optional[str]
    metadata: Dict[str, Any]
    evidence: List[str]
    evidence_source_unit_ids: List[str]
    search_result: SearchResult

    @property
    def formatted_context(self) -> str:
        return str(
            self.search_result.retrieval_metadata.get("formatted_context") or ""
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        Path(tmp_name).replace(path)
    except Exception:
        try:
            Path(tmp_name).unlink(missing_ok=True)
        finally:
            raise


def build_mode_dir(output_root: Path, search_mode: str = DEFAULT_SEARCH_MODE) -> Path:
    return output_root / f"{output_root.name}-{search_mode}"


def answer_query_for_qa(qa: QAPair) -> str:
    query = qa.question
    options = (qa.metadata or {}).get("all_options")
    if not isinstance(options, dict):
        return query

    options_text = "\n".join(f"{key} {value}" for key, value in options.items())
    return f"""{qa.question}

OPTIONS:
{options_text}

IMPORTANT: This is a multiple-choice question. You MUST analyze the context and select the BEST option. In your FINAL ANSWER, return ONLY the option letter like (a), (b), (c), or (d), nothing else."""


def select_dataset_slice(dataset: Dataset, from_conv: int, to_conv: Optional[int]) -> Dataset:
    selected_conversations = dataset.conversations[from_conv:to_conv]
    selected_ids = {conversation.conversation_id for conversation in selected_conversations}
    selected_qas = [
        qa
        for qa in dataset.qa_pairs
        if str((qa.metadata or {}).get("conversation_id") or "") in selected_ids
    ]
    return Dataset(
        dataset_name=dataset.dataset_name,
        conversations=selected_conversations,
        qa_pairs=selected_qas,
        metadata={**(dataset.metadata or {}), "selected_conversation_ids": sorted(selected_ids)},
    )


def validate_persona0_slice(dataset: Dataset) -> None:
    if len(dataset.conversations) != 1:
        raise ValueError(
            "Human persona0 baseline expects exactly one selected conversation; "
            f"found {len(dataset.conversations)}."
        )
    conversation_id = dataset.conversations[0].conversation_id
    if conversation_id != "subtlememory_0":
        raise ValueError(
            "Human persona0 baseline expected conversation_id='subtlememory_0', "
            f"found {conversation_id!r}."
        )

    persona_ids = {
        str((qa.metadata or {}).get("persona_id") or "").strip()
        for qa in dataset.qa_pairs
        if str((qa.metadata or {}).get("persona_id") or "").strip()
    }
    if persona_ids and persona_ids != {"0"}:
        raise ValueError(
            "Human persona0 baseline expected only persona_id='0', "
            f"found {sorted(persona_ids)!r}."
        )


async def build_oracle_search_results(
    dataset: Dataset,
    *,
    output_dir: Path,
    run_name: str,
    system_id: str = DEFAULT_SYSTEM,
) -> List[SearchResult]:
    adapter = OracleContextAdapter(
        {
            "name": system_id,
            "adapter": "oracle_context",
            "llm": {
                "provider": "openai",
                "model": "not-used-for-human-baseline",
                "api_key": "not-used",
                "base_url": "http://localhost/unused",
            },
        },
        output_dir=output_dir,
    )
    adapter.set_run_context(
        {
            "run_id": run_name,
            "run_name": run_name,
            "dataset_id": dataset.dataset_name,
            "system_id": system_id,
            "benchmark_mode": "category1",
            "requested_stages": ["search", "answer"],
            "stage_order": ["add", "finalize", "search", "answer", "evaluate"],
        }
    )
    adapter.build_lazy_index(dataset.conversations, output_dir)
    conversation_map = {
        conversation.conversation_id: conversation for conversation in dataset.conversations
    }

    search_results: List[SearchResult] = []
    for qa in dataset.qa_pairs:
        conversation_id = str((qa.metadata or {}).get("conversation_id") or "")
        conversation = conversation_map.get(conversation_id)
        if conversation is None:
            raise ValueError(
                f"Question {qa.question_id!r} points to missing conversation "
                f"{conversation_id!r}."
            )
        result = await adapter.search(
            qa.question,
            conversation_id,
            None,
            conversation=conversation,
            question_id=qa.question_id,
            question_metadata=qa.metadata,
        )
        search_results.append(result)
    return search_results


def build_annotation_items_from_dataset(
    dataset: Dataset,
    *,
    output_dir: Path,
    run_name: str = DEFAULT_RUN_NAME,
    system_id: str = DEFAULT_SYSTEM,
    validate_persona0: bool = True,
) -> List[AnnotationItem]:
    if validate_persona0:
        validate_persona0_slice(dataset)
    if not dataset.qa_pairs:
        raise ValueError("No questions found for the selected dataset slice.")

    search_results = asyncio.run(
        build_oracle_search_results(
            dataset,
            output_dir=output_dir,
            run_name=run_name,
            system_id=system_id,
        )
    )
    search_map = {result.question_id: result for result in search_results}

    items: List[AnnotationItem] = []
    for index, qa in enumerate(dataset.qa_pairs):
        search_result = search_map.get(qa.question_id)
        if search_result is None:
            raise ValueError(f"Missing oracle search result for {qa.question_id!r}.")
        if search_result.retrieval_status != "ok":
            detail = search_result.retrieval_metadata.get("error") or "unknown error"
            raise ValueError(
                f"Oracle context failed for {qa.question_id!r}: {detail}"
            )
        items.append(
            AnnotationItem(
                index=index,
                question_id=qa.question_id,
                conversation_id=str((qa.metadata or {}).get("conversation_id") or ""),
                question=qa.question,
                answer_query=answer_query_for_qa(qa),
                golden_answer=qa.answer,
                category=qa.category,
                metadata=dict(qa.metadata or {}),
                evidence=list(qa.evidence or []),
                evidence_source_unit_ids=list(
                    getattr(qa, "evidence_source_unit_ids", []) or []
                ),
                search_result=search_result,
            )
        )
    return items


def load_annotation_items(
    *,
    dataset_name: str,
    data_dir: Path,
    output_dir: Path,
    run_name: str,
    system_id: str,
    from_conv: int,
    to_conv: Optional[int],
    validate_persona0: bool = True,
) -> List[AnnotationItem]:
    dataset = load_dataset(dataset_name, str(data_dir))
    dataset = select_dataset_slice(dataset, from_conv, to_conv)
    return build_annotation_items_from_dataset(
        dataset,
        output_dir=output_dir,
        run_name=run_name,
        system_id=system_id,
        validate_persona0=validate_persona0,
    )


def search_result_to_dict(result: SearchResult) -> Dict[str, Any]:
    return {
        "question_id": result.question_id,
        "query": result.query,
        "conversation_id": result.conversation_id,
        "results": result.results,
        "retrieval_metadata": result.retrieval_metadata,
        "retrieval_status": result.retrieval_status,
        "timing_ms": result.timing_ms,
    }


def answer_result_to_dict(item: AnnotationItem, answer: str) -> Dict[str, Any]:
    metadata = dict(item.metadata or {})
    metadata.update(
        {
            "answer_query": item.answer_query,
            "answer_prompt": None,
            "answer_prompt_available": False,
            "answer_context_char_count": len(item.formatted_context),
            "retrieved_item_count": len(item.search_result.results),
            "answer_source": "human",
            "human_baseline": True,
            "human_context_condition": "oracle_query_sessions",
        }
    )
    return {
        "question_id": item.question_id,
        "question": item.question,
        "answer": answer.strip(),
        "golden_answer": item.golden_answer,
        "category": item.category,
        "conversation_id": item.conversation_id,
        "formatted_context": item.formatted_context,
        "search_results": item.search_result.results,
        "latency_ms": 0.0,
        "errors": [],
        "metadata": metadata,
    }


def item_to_client_payload(
    item: AnnotationItem,
    *,
    answer: str = "",
    total: Optional[int] = None,
) -> Dict[str, Any]:
    payload = {
        "index": item.index,
        "total": total,
        "question_id": item.question_id,
        "conversation_id": item.conversation_id,
        "question": item.question,
        "answer_query": item.answer_query,
        "formatted_context": item.formatted_context,
        "retrieval_status": item.search_result.retrieval_status,
        "message_count": item.search_result.retrieval_metadata.get("message_count"),
        "scope_session_ids": item.search_result.retrieval_metadata.get(
            "scope_session_ids", []
        ),
        "answer": answer,
    }
    for key in FORBIDDEN_CLIENT_KEYS:
        payload.pop(key, None)
    return payload


class AnnotationState:
    def __init__(
        self,
        path: Path,
        *,
        question_ids: Sequence[str],
        annotator: str = "human",
    ):
        self.path = path
        self.question_ids = list(question_ids)
        self.question_id_set = set(question_ids)
        self.annotator = annotator
        self.answers: Dict[str, Dict[str, Any]] = {}
        self.created_at = utc_now()
        self.updated_at = self.created_at
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        answers = payload.get("answers", {}) or {}
        self.answers = {
            str(question_id): dict(record)
            for question_id, record in answers.items()
            if str(question_id) in self.question_id_set and isinstance(record, dict)
        }
        self.created_at = str(payload.get("created_at") or self.created_at)
        self.updated_at = str(payload.get("updated_at") or self.updated_at)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "human_baseline_annotation_state_v1",
            "annotator": self.annotator,
            "question_ids": self.question_ids,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "answers": self.answers,
        }

    def save(self) -> None:
        atomic_write_json(self.path, self.to_dict())

    def get_answer(self, question_id: str) -> str:
        record = self.answers.get(question_id) or {}
        return str(record.get("answer") or "")

    def save_answer(self, question_id: str, answer: str) -> None:
        if question_id not in self.question_id_set:
            raise KeyError(f"Unknown question_id: {question_id}")
        now = utc_now()
        clean_answer = str(answer or "").strip()
        if clean_answer:
            self.answers[question_id] = {"answer": clean_answer, "updated_at": now}
        else:
            self.answers.pop(question_id, None)
        self.updated_at = now
        self.save()

    def answered_count(self) -> int:
        return sum(
            1
            for question_id in self.question_ids
            if self.get_answer(question_id).strip()
        )

    def missing_question_ids(self) -> List[str]:
        return [
            question_id
            for question_id in self.question_ids
            if not self.get_answer(question_id).strip()
        ]


def judge_command_text(
    *,
    dataset_name: str,
    system_id: str,
    run_name: str,
    output_root: Path,
    from_conv: int,
    to_conv: Optional[int],
) -> str:
    parts = [
        "uv",
        "run",
        "python",
        "-m",
        "evaluation.cli",
        "--dataset",
        dataset_name,
        "--system",
        system_id,
        "--stages",
        "evaluate",
        "--from-conv",
        str(from_conv),
    ]
    if to_conv is not None:
        parts.extend(["--to-conv", str(to_conv)])
    parts.extend(["--run-name", run_name, "--output-dir", str(output_root)])
    return " ".join(parts)


def export_artifacts(
    *,
    items: Sequence[AnnotationItem],
    state: AnnotationState,
    output_dir: Path,
    dataset_name: str,
    system_id: str,
    run_name: str,
    output_root: Path,
    from_conv: int,
    to_conv: Optional[int],
    allow_incomplete: bool = False,
) -> Dict[str, Any]:
    missing = state.missing_question_ids()
    if missing and not allow_incomplete:
        raise ValueError(
            f"Cannot export: {len(missing)} unanswered questions remain."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    search_rows = [search_result_to_dict(item.search_result) for item in items]
    answer_rows = [
        answer_result_to_dict(item, state.get_answer(item.question_id))
        for item in items
        if state.get_answer(item.question_id).strip() or allow_incomplete
    ]

    search_path = output_dir / "search_results.json"
    answer_path = output_dir / "answer_results.json"
    manifest_path = output_dir / "human_annotation_export.json"
    atomic_write_json(search_path, search_rows)
    atomic_write_json(answer_path, answer_rows)
    manifest = {
        "schema_version": "human_baseline_export_v1",
        "exported_at": utc_now(),
        "dataset": dataset_name,
        "system": system_id,
        "run_name": run_name,
        "conversation_id": items[0].conversation_id if items else "",
        "total_questions": len(items),
        "answered_questions": state.answered_count(),
        "context_condition": "oracle_query_sessions",
        "search_results": str(search_path),
        "answer_results": str(answer_path),
        "judge_command": judge_command_text(
            dataset_name=dataset_name,
            system_id=system_id,
            run_name=run_name,
            output_root=output_root,
            from_conv=from_conv,
            to_conv=to_conv,
        ),
    }
    atomic_write_json(manifest_path, manifest)
    return {
        "search_results": str(search_path),
        "answer_results": str(answer_path),
        "manifest": str(manifest_path),
        "judge_command": manifest["judge_command"],
    }


class HumanBaselineApp:
    def __init__(
        self,
        *,
        items: Sequence[AnnotationItem],
        state: AnnotationState,
        dataset_name: str,
        system_id: str,
        run_name: str,
        output_root: Path,
        output_dir: Path,
        from_conv: int,
        to_conv: Optional[int],
    ):
        self.items = list(items)
        self.items_by_question_id = {item.question_id: item for item in self.items}
        self.state = state
        self.dataset_name = dataset_name
        self.system_id = system_id
        self.run_name = run_name
        self.output_root = output_root
        self.output_dir = output_dir
        self.from_conv = from_conv
        self.to_conv = to_conv

    def state_payload(self) -> Dict[str, Any]:
        return {
            "total": len(self.items),
            "answered": self.state.answered_count(),
            "unanswered": len(self.state.missing_question_ids()),
            "state_path": str(self.state.path),
            "output_dir": str(self.output_dir),
            "questions": [
                {
                    "index": item.index,
                    "question_id": item.question_id,
                    "answered": bool(self.state.get_answer(item.question_id).strip()),
                }
                for item in self.items
            ],
        }

    def question_payload(self, index: int) -> Dict[str, Any]:
        if index < 0 or index >= len(self.items):
            raise IndexError(f"Question index out of range: {index}")
        item = self.items[index]
        return item_to_client_payload(
            item,
            answer=self.state.get_answer(item.question_id),
            total=len(self.items),
        )

    def save_answer(self, question_id: str, answer: str) -> Dict[str, Any]:
        if question_id not in self.items_by_question_id:
            raise KeyError(f"Unknown question_id: {question_id}")
        self.state.save_answer(question_id, answer)
        return self.state_payload()

    def export(self) -> Dict[str, Any]:
        return export_artifacts(
            items=self.items,
            state=self.state,
            output_dir=self.output_dir,
            dataset_name=self.dataset_name,
            system_id=self.system_id,
            run_name=self.run_name,
            output_root=self.output_root,
            from_conv=self.from_conv,
            to_conv=self.to_conv,
        )


def html_page() -> bytes:
    return HTML.encode("utf-8")


def make_handler(app: HumanBaselineApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "HumanBaselineApp/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(
                f"{self.address_string()} - - "
                f"[{self.log_date_time_string()}] {fmt % args}"
            )

        def send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_error_json(self, status: int, message: str) -> None:
            self.send_json({"error": message}, status=status)

        def read_json_body(self) -> Dict[str, Any]:
            content_length = int(self.headers.get("Content-Length") or "0")
            body = self.rfile.read(content_length)
            if not body:
                return {}
            return json.loads(body.decode("utf-8"))

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    body = html_page()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if parsed.path == "/api/state":
                    self.send_json(app.state_payload())
                    return
                if parsed.path == "/api/question":
                    params = parse_qs(parsed.query)
                    index = int((params.get("index") or ["0"])[0])
                    self.send_json(app.question_payload(index))
                    return
                self.send_error_json(HTTPStatus.NOT_FOUND, "Not found")
            except (ValueError, IndexError, KeyError) as exc:
                self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            except Exception as exc:
                self.send_error_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"{type(exc).__name__}: {exc}",
                )

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/answer":
                    payload = self.read_json_body()
                    question_id = str(payload.get("question_id") or "")
                    answer = str(payload.get("answer") or "")
                    self.send_json(app.save_answer(question_id, answer))
                    return
                if parsed.path == "/api/export":
                    self.send_json(app.export())
                    return
                self.send_error_json(HTTPStatus.NOT_FOUND, "Not found")
            except json.JSONDecodeError as exc:
                self.send_error_json(HTTPStatus.BAD_REQUEST, f"Invalid JSON: {exc}")
            except (ValueError, IndexError, KeyError) as exc:
                self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            except Exception as exc:
                self.send_error_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"{type(exc).__name__}: {exc}",
                )

    return Handler


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local human baseline annotation app."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--from-conv", type=int, default=0)
    parser.add_argument("--to-conv", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--allow-non-persona0",
        action="store_true",
        help="Disable the fail-fast persona0 validation guard.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_root = Path(args.output_dir).expanduser().resolve()
    output_dir = build_mode_dir(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "human_annotation_state.json"

    print("Loading annotation packet...")
    started = time.perf_counter()
    items = load_annotation_items(
        dataset_name=args.dataset,
        data_dir=Path(args.data_dir).expanduser().resolve(),
        output_dir=output_dir,
        run_name=args.run_name,
        system_id=args.system,
        from_conv=args.from_conv,
        to_conv=args.to_conv,
        validate_persona0=not args.allow_non_persona0,
    )
    state = AnnotationState(
        state_path,
        question_ids=[item.question_id for item in items],
        annotator="human",
    )
    app = HumanBaselineApp(
        items=items,
        state=state,
        dataset_name=args.dataset,
        system_id=args.system,
        run_name=args.run_name,
        output_root=output_root,
        output_dir=output_dir,
        from_conv=args.from_conv,
        to_conv=args.to_conv,
    )
    elapsed = time.perf_counter() - started
    print(
        f"Loaded {len(items)} questions in {elapsed:.1f}s; "
        f"{state.answered_count()} already answered."
    )
    print(f"State file: {state_path}")
    print(f"Export dir: {output_dir}")

    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    url = f"http://{args.host}:{args.port}"
    print(f"Serving human baseline app at {url}")
    print("Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Human Baseline Annotation</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fb;
      --panel: #ffffff;
      --panel-2: #f0f3f7;
      --border: #d8dee9;
      --text: #1d2430;
      --muted: #657083;
      --accent: #286f6c;
      --accent-2: #0f766e;
      --danger: #b42318;
      --shadow: 0 1px 2px rgba(20, 30, 45, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    .shell {
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      height: 100vh;
      min-height: 620px;
    }
    aside {
      display: flex;
      flex-direction: column;
      border-right: 1px solid var(--border);
      background: var(--panel);
      min-width: 0;
    }
    header {
      padding: 18px 18px 12px;
      border-bottom: 1px solid var(--border);
    }
    h1 {
      margin: 0 0 10px;
      font-size: 18px;
      line-height: 1.25;
      font-weight: 700;
    }
    .meta, .status {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }
    .toolbar {
      display: flex;
      gap: 8px;
      padding: 12px 18px;
      border-bottom: 1px solid var(--border);
    }
    button {
      border: 1px solid var(--border);
      background: var(--panel);
      color: var(--text);
      border-radius: 6px;
      min-height: 34px;
      padding: 7px 11px;
      font: inherit;
      font-size: 13px;
      cursor: pointer;
    }
    button:hover { border-color: #aeb8c7; }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }
    button.primary:hover { background: var(--accent-2); }
    button:disabled {
      color: #9aa4b2;
      cursor: not-allowed;
      background: #eef1f5;
    }
    .question-list {
      overflow: auto;
      padding: 8px;
    }
    .question-row {
      width: 100%;
      display: grid;
      grid-template-columns: 44px minmax(0, 1fr) 16px;
      gap: 8px;
      align-items: center;
      border: 1px solid transparent;
      background: transparent;
      text-align: left;
      border-radius: 6px;
      padding: 9px 8px;
      margin: 1px 0;
    }
    .question-row.active {
      background: #e8f3f1;
      border-color: #a8d4cf;
    }
    .qid {
      overflow: hidden;
      white-space: nowrap;
      text-overflow: ellipsis;
      color: var(--muted);
      font-size: 12px;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: #c4ccd8;
    }
    .dot.done { background: var(--accent); }
    main {
      min-width: 0;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      height: 100vh;
    }
    .question-head {
      padding: 22px 28px 16px;
      border-bottom: 1px solid var(--border);
      background: var(--panel);
    }
    .question-title {
      margin: 0;
      font-size: 20px;
      line-height: 1.35;
      font-weight: 700;
    }
    .content {
      overflow: auto;
      padding: 20px 28px;
    }
    .transcript {
      max-width: 1040px;
      white-space: pre-wrap;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 13px;
      line-height: 1.55;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 6px;
      box-shadow: var(--shadow);
      padding: 18px;
    }
    .answer-bar {
      border-top: 1px solid var(--border);
      background: var(--panel);
      padding: 16px 28px 18px;
      display: grid;
      gap: 10px;
    }
    textarea {
      width: 100%;
      min-height: 100px;
      max-height: 220px;
      resize: vertical;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px;
      font: inherit;
      font-size: 15px;
      line-height: 1.45;
      color: var(--text);
      background: #fff;
    }
    textarea:focus {
      outline: 2px solid rgba(40, 111, 108, 0.2);
      border-color: var(--accent);
    }
    .actions {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .left-actions, .right-actions {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }
    .message {
      color: var(--muted);
      font-size: 13px;
      min-height: 18px;
    }
    .message.error { color: var(--danger); }
    @media (max-width: 820px) {
      .shell { grid-template-columns: 1fr; height: auto; min-height: 100vh; }
      aside { height: 260px; border-right: none; border-bottom: 1px solid var(--border); }
      main { height: auto; min-height: calc(100vh - 260px); }
      .question-head, .content, .answer-bar { padding-left: 16px; padding-right: 16px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <header>
        <h1>Human Baseline Annotation</h1>
        <div class="meta" id="progress">Loading...</div>
      </header>
      <div class="toolbar">
        <button id="allBtn">All</button>
        <button id="openBtn">Unanswered</button>
      </div>
      <div class="question-list" id="questionList"></div>
    </aside>
    <main>
      <section class="question-head">
        <div class="status" id="questionMeta"></div>
        <h2 class="question-title" id="questionText">Loading...</h2>
      </section>
      <section class="content">
        <div class="transcript" id="transcript"></div>
      </section>
      <section class="answer-bar">
        <textarea id="answer" placeholder="Answer"></textarea>
        <div class="actions">
          <div class="left-actions">
            <button id="prevBtn">Previous</button>
            <button id="nextBtn">Next</button>
            <span class="message" id="saveStatus"></span>
          </div>
          <div class="right-actions">
            <button class="primary" id="exportBtn">Export</button>
          </div>
        </div>
      </section>
    </main>
  </div>
  <script>
    let state = null;
    let currentIndex = 0;
    let onlyOpen = false;
    let currentQuestionId = null;
    let saveTimer = null;

    const els = {
      progress: document.getElementById("progress"),
      questionList: document.getElementById("questionList"),
      questionMeta: document.getElementById("questionMeta"),
      questionText: document.getElementById("questionText"),
      transcript: document.getElementById("transcript"),
      answer: document.getElementById("answer"),
      saveStatus: document.getElementById("saveStatus"),
      prevBtn: document.getElementById("prevBtn"),
      nextBtn: document.getElementById("nextBtn"),
      exportBtn: document.getElementById("exportBtn"),
      allBtn: document.getElementById("allBtn"),
      openBtn: document.getElementById("openBtn")
    };

    async function api(path, options) {
      const response = await fetch(path, options);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || response.statusText);
      return payload;
    }

    async function loadState() {
      state = await api("/api/state");
      renderState();
    }

    function renderState() {
      els.progress.textContent = `${state.answered}/${state.total} answered`;
      els.exportBtn.disabled = state.unanswered !== 0;
      els.questionList.innerHTML = "";
      for (const item of state.questions) {
        if (onlyOpen && item.answered) continue;
        const row = document.createElement("button");
        row.className = "question-row" + (item.index === currentIndex ? " active" : "");
        row.type = "button";
        row.onclick = () => loadQuestion(item.index);
        row.innerHTML = `
          <span>${item.index + 1}</span>
          <span class="qid" title="${item.question_id}">${item.question_id}</span>
          <span class="dot ${item.answered ? "done" : ""}"></span>
        `;
        els.questionList.appendChild(row);
      }
    }

    async function loadQuestion(index) {
      const item = await api(`/api/question?index=${index}`);
      currentIndex = item.index;
      currentQuestionId = item.question_id;
      els.questionMeta.textContent = `${item.index + 1}/${item.total}  ${item.question_id}`;
      els.questionText.textContent = item.answer_query || item.question;
      els.transcript.textContent = item.formatted_context || "(empty context)";
      els.answer.value = item.answer || "";
      els.saveStatus.textContent = "";
      els.saveStatus.className = "message";
      renderState();
    }

    async function saveCurrentAnswer() {
      if (!currentQuestionId) return;
      els.saveStatus.textContent = "Saving...";
      els.saveStatus.className = "message";
      try {
        state = await api("/api/answer", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({question_id: currentQuestionId, answer: els.answer.value})
        });
        els.saveStatus.textContent = "Saved";
        renderState();
      } catch (error) {
        els.saveStatus.textContent = error.message;
        els.saveStatus.className = "message error";
      }
    }

    function scheduleSave() {
      clearTimeout(saveTimer);
      saveTimer = setTimeout(saveCurrentAnswer, 350);
    }

    function nextVisible(delta) {
      if (!state) return currentIndex;
      let index = currentIndex;
      for (let step = 0; step < state.total; step += 1) {
        index = Math.max(0, Math.min(state.total - 1, index + delta));
        const row = state.questions[index];
        if (!onlyOpen || !row.answered) return index;
        if (index === 0 || index === state.total - 1) return index;
      }
      return currentIndex;
    }

    els.answer.addEventListener("input", scheduleSave);
    els.prevBtn.onclick = () => loadQuestion(nextVisible(-1));
    els.nextBtn.onclick = () => loadQuestion(nextVisible(1));
    els.allBtn.onclick = () => { onlyOpen = false; renderState(); };
    els.openBtn.onclick = () => { onlyOpen = true; renderState(); };
    els.exportBtn.onclick = async () => {
      await saveCurrentAnswer();
      try {
        const payload = await api("/api/export", {method: "POST"});
        els.saveStatus.textContent = `Exported. Judge: ${payload.judge_command}`;
        els.saveStatus.className = "message";
      } catch (error) {
        els.saveStatus.textContent = error.message;
        els.saveStatus.className = "message error";
      }
    };

    (async function boot() {
      try {
        await loadState();
        await loadQuestion(0);
      } catch (error) {
        els.questionText.textContent = "Failed to load";
        els.transcript.textContent = error.message;
      }
    })();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
