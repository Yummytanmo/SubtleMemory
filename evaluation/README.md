# Evaluation Guide: Adding a Memory System

This guide explains how to evaluate a new memory system on SubtleMemory.
It is intended for adapter authors. If you only want to run an already
configured system, see the root [README](../README.md#evaluation).

SubtleMemory uses one benchmark protocol for hosted APIs, local memory
libraries, native agent runtimes, OpenClaw-style plugins, and baselines:

```text
add -> finalize -> search -> answer -> evaluate
```

The memory system is responsible for ingesting the conversation history and
retrieving relevant memories. The benchmark framework handles data loading,
question selection, answer generation, judging, artifact writing, and run
validation.

## Integration Overview

To add a new memory system:

1. Add a system config under `evaluation/config/systems/<system>.yaml`.
2. Implement an adapter under `evaluation/src/adapters/`.
3. Register the adapter in `evaluation/src/adapters/core/registry.py`.
4. Run a smoke slice with `evaluation.cli`.
5. Validate the output with `evaluation.validate_run`.
6. Run the full benchmark after the smoke artifacts pass validation.

The CLI resolves `--system <system>` to
`evaluation/config/systems/<system>.yaml`. The YAML `adapter` field then maps
to a Python adapter through the registry.

For example, `--system mem0` loads:

```text
evaluation/config/systems/mem0.yaml
  adapter: "mem0"

evaluation/src/adapters/core/registry.py
  "mem0": "evaluation.src.adapters.providers.mem0_adapter"

evaluation/src/adapters/providers/mem0_adapter.py
  @register_adapter("mem0")
  class Mem0Adapter(...)
```

## Benchmark Flow

| Stage | Adapter or framework responsibility | Main artifacts |
| --- | --- | --- |
| `add` | Adapter ingests `Conversation` objects into the memory system. | `source_units.jsonl`, `import_manifest.jsonl` |
| `finalize` | Adapter confirms or refreshes write readiness. Async providers should resolve pending writes here. | `finalize_report.json`, refreshed `import_manifest.jsonl` |
| `search` | Adapter retrieves memories for each benchmark question and returns `SearchResult`. In normal benchmark runs this is the adapter's API retrieval path. | `search_results.json` |
| `answer` | Framework builds context from search results and calls `adapter.answer(...)` (a required adapter method; see [Adapter Contract](#adapter-contract)). Most online systems reuse the common answer path. | `answer_results.json`, `qa_results.jsonl` |
| `evaluate` | Framework evaluator judges generated answers against SubtleMemory references and relation metadata. | `evaluation_results.jsonl`, `score_summary.json` |

## Search Modes

SubtleMemory has two evaluation-level search modes:

| Mode | Required for a new system? | Pipeline call | Meaning |
| --- | --- | --- | --- |
| `api` | Yes | `adapter.search(...)` | The normal benchmark path. The memory system retrieves evidence through its own search API, retrieval engine, agent recall path, or native query interface. |
| `readback` | No | `adapter.search_from_readback(...)` | Optional diagnostic path. It does not run query search; it rebuilds evidence from stored memory objects linked to the question's source sessions. |

In `readback` mode, the pipeline passes question metadata, including
`session_ids`, to the adapter. The adapter should use whatever
provider-specific handles it has, such as session ids, namespace or user ids,
memory ids, local file paths, or import-manifest receipts, to locate memory
objects produced from those source sessions and format them as search context.

In the current SubtleMemory runner, `search` and `answer` are gated on completed
add/finalize artifacts. A separate readback rerun therefore commonly reuses
`import_manifest.jsonl` and `finalize_report.json` from the matching API-mode
run, but that is a runner artifact requirement rather than the conceptual
definition of readback.

`api` is the default. A missing `search.mode`, `search.mode: api`, or a
provider-internal value other than explicit `readback` runs the normal API path.
The CLI can also set it directly:

```bash
uv run python -m evaluation.cli ... --search-mode api
uv run python -m evaluation.cli ... --search-mode readback
```

Do not treat `readback` as an integration requirement. If an adapter does not
implement readback hooks, the base adapter returns
`retrieval_status="unsupported"`, which is acceptable for API-only benchmark
integration. Readback is useful when you want to diagnose whether target
evidence was written but not retrieved by normal search; it is not a replacement
for the main benchmark retrieval path.

## Data Contract

The SubtleMemory dataset lives under `data/subtlememory/` and can be copied to
the evaluation data path:

```bash
mkdir -p evaluation/data/subtlememory
rsync -a data/subtlememory/ evaluation/data/subtlememory/
```

Adapter authors do not need to parse the raw `bench_instances.json` and
`history_sessions.json` files. The loader converts them into standard runtime
objects:

- `Conversation`: one persona-level history, with ordered messages and message
  metadata such as source session id, source case id, timestamp, and source
  unit id.
- `QAPair`: one benchmark question, with metadata such as `correct_answers`,
  `incorrect_answers`, `session_ids`, `facts`, `relation_type`,
  `relation_subtype`, `topic`, `persona_id`, and `source`.
- `SearchResult`: the adapter-normalized retrieval result for one question.

The pipeline also creates `source_units.jsonl`, which is the canonical mapping
from source messages to stable source-unit ids. If your adapter records provider
write receipts, include these ids in `import_manifest.jsonl` so validation and
readback can trace benchmark evidence through your system.

## System Config

Create a YAML file under `evaluation/config/systems/`. Keep credentials in
`.env`; YAML values support `${VAR}` and `${VAR:default}` interpolation.

Minimal example:

```yaml
name: "my-memory-system"
version: "1.0"
description: "My memory system"

adapter: "my_memory_system"
benchmark_mode: "category1"
semantic_chunk_policy: "native_system"
transport_batch_policy: "batch_messages"

api_key: "${MY_MEMORY_API_KEY}"
api_url: "${MY_MEMORY_API_URL:https://api.example.com}"
num_workers: 4

readiness:
  budget_seconds: 180
  poll_interval_seconds: 10

llm:
  provider: "openai"
  model: "${ANSWER_LLM_MODEL:gpt-4o-mini}"
  api_key: "${ANSWER_LLM_API_KEY}"
  base_url: "${ANSWER_LLM_BASE_URL:https://api.openai.com/v1}"
  temperature: 0
  max_tokens: 16384

search:
  mode: api
  top_k: 20
  num_workers: 8

answer:
  max_retries: 3
  timeout_seconds: 120
```

The dataset config supplies the evaluator. For SubtleMemory, the default judge
uses `JUDGE_LLM_API_KEY`, `JUDGE_LLM_BASE_URL`, and `JUDGE_LLM_MODEL`.

Keep `search.mode` unset or set to `api` for the normal benchmark. Set it to
`readback` only for an explicit diagnostic rerun, usually with
`--reuse-add-artifacts-from` so the readback run can reuse the manifest and
provider write receipts from an earlier API-mode add/finalize run.

## Adapter Contract

All adapters ultimately implement `BaseAdapter` from
`evaluation/src/adapters/core/base.py`.

Required methods:

| Method | Purpose |
| --- | --- |
| `add(conversations, **kwargs)` | Write benchmark history into the memory system and return an index or metadata object. |
| `search(query, conversation_id, index, **kwargs)` | Retrieve relevant memories and return a `SearchResult`. |
| `answer(query, context, **kwargs)` | Generate the final answer from the retrieved context. The answer stage calls this unconditionally, so it must exist. `OnlineAPIAdapter` provides a shared implementation through the configured `llm`, so hosted-API adapters inherit it for free. Adapters that subclass `BaseAdapter` directly must implement it, because `BaseAdapter` does not. |

Recommended methods:

| Method | Purpose |
| --- | --- |
| `get_import_manifest_records()` | Return provenance rows mapping source units to provider writes. Strongly recommended for validation and readback. |
| `finalize_imports(import_manifest_rows, ...)` | Confirm async provider writes are ready, update statuses, and backfill memory refs when possible. |
| `build_lazy_index(conversations, output_dir)` | Rebuild local runtime/index metadata when resuming later stages without rerunning `add`. |
| `prepare(conversations, **kwargs)` | Optional setup before add, such as cleaning isolated namespaces for debug runs. |

Optional readback methods:

| Method | Purpose |
| --- | --- |
| `get_storage_readback(...)` | Read provider storage objects and normalize them for provenance-driven analysis. Only needed if you want `--search-mode readback`. |
| `search_from_readback(...)` | Build a `SearchResult` from readback objects instead of the provider search API. Only needed if you want `--search-mode readback`. |

The default `BaseAdapter` implementation already returns an unsupported
readback result. Do not add a fake readback implementation unless the system can
actually map benchmark `session_ids` or import-manifest receipts back to stored
memory objects.

For hosted APIs, prefer inheriting `OnlineAPIAdapter` from
`evaluation/src/adapters/shared/online_base.py`. It already handles:

- conversation-level concurrency
- single-speaker and dual-perspective message formatting
- per-run namespace/user id isolation
- import-manifest recording
- answer generation through the shared LLM path

An online API adapter usually implements:

- `_add_user_messages(...)`
- `_search_single_user(...)`
- `_build_single_search_result(...)`
- `_build_dual_search_result(...)`

For local/native systems, inherit `BaseAdapter` directly when the system needs
custom runtime management, local indexes, SQLite files, subprocesses, or native
agent answering. In that case you must implement `answer(...)` yourself, since
`BaseAdapter` does not provide one. The bundled `amem`, `mirix`, and `metaclaw`
adapters each define their own `answer`.

## Registering the Adapter

Add a module mapping in `evaluation/src/adapters/core/registry.py`:

```python
_ADAPTER_MODULES = {
    ...
    "my_memory_system": "evaluation.src.adapters.providers.my_memory_system_adapter",
}
```

Then decorate the adapter class:

```python
from evaluation.src.adapters.core.registry import register_adapter


@register_adapter("my_memory_system")
class MyMemorySystemAdapter(...):
    ...
```

The `adapter` value in your system YAML must match the registry name:

```yaml
adapter: "my_memory_system"
```

## Worked Example: Mem0

Mem0 is the clearest example of a hosted memory API adapter.

Config: `evaluation/config/systems/mem0.yaml`

- `adapter: "mem0"` selects the registered adapter.
- `api_key: "${MEM0_API_KEY}"` and `host: "${MEM0_HOST:https://api.mem0.ai}"`
  resolve from `.env`.
- `batch_size`, `add_interval`, `num_workers`, and `search.search_interval`
  control provider load.
- `readiness` controls finalize-stage polling budget.
- `llm` controls answer generation, not memory ingestion.
- `search.top_k` controls provider retrieval depth.

Adapter: `evaluation/src/adapters/providers/mem0_adapter.py`

- `Mem0Adapter` inherits `OnlineAPIAdapter`.
- `__init__` creates `AsyncMemoryClient` from the resolved config.
- `_add_user_messages` writes messages to Mem0 in batches without crossing
  source session boundaries.
- Add receipts include provider status, event ids, source session ids,
  source-unit ids, and memory refs when available.
- `finalize_imports` refreshes write status and backfills stable memory ids when
  the provider exposes them.

Mem0 API mode:

- This is the normal benchmark path and the only required path for Mem0
  leaderboard-style evaluation.
- `mem0.yaml` does not set `search.mode: readback`, so Mem0 runs in `api` mode
  unless the CLI or an override explicitly changes it.
- The search stage calls `adapter.search(...)`, which uses
  `_search_single_user`.
- `_search_single_user` calls `AsyncMemoryClient.search(...)` with `query`,
  `top_k`, and `filters={"user_id": user_id}`.
- Search results are normalized into dictionaries with `content`, `score`, and
  `metadata`, then wrapped in `SearchResult`.

Mem0 readback mode:

- This is optional diagnostic support, not a requirement for benchmarking Mem0.
- Enable it with `--search-mode readback` and reuse the earlier API-mode
  add/finalize artifacts with `--reuse-add-artifacts-from`.
- The search stage calls `search_from_readback(...)` instead of
  `adapter.search(...)`.
- `search_from_readback` reads `question_metadata["session_ids"]`. Without
  session IDs, it returns `retrieval_status="unsupported"`.
- It passes the question id, conversation id, session ids, and
  `import_manifest.jsonl` rows into `get_storage_readback(...)`.
- `get_storage_readback` uses manifest memory refs, namespace receipts, and
  source session ids to find the Mem0 `user_id` values and memory-id-to-session
  mapping for the target sessions.
- The adapter then lists Mem0 memories for those users/sessions, normalizes each
  storage object, orders target-session objects before extras, and builds a
  `SearchResult` whose results have score `1.0` and whose metadata includes
  `search_mode: readback`, `session_ids`, `user_ids`, `formatted_context`, and a
  serialized `readback` payload.

Mem0 readback therefore answers a different diagnostic question: "Were memory
objects from the target sessions present in Mem0 storage?" It does not call
Mem0's query search API and should not be reported as the main Mem0 retrieval
score.

This pattern is reusable for other hosted APIs: keep provider-specific logic
inside the adapter, and return framework-standard artifacts.

## Running a Smoke Test

From the repository root, install dependencies and configure `.env`:

```bash
uv sync
cp env.template .env
```

Set at least:

```text
ANSWER_LLM_API_KEY=...
JUDGE_LLM_API_KEY=...
MY_MEMORY_API_KEY=...
```

For Mem0, use:

```text
MEM0_API_KEY=...
MEM0_HOST=https://api.mem0.ai
```

Run a small Mem0 smoke slice:

```bash
uv run python -m evaluation.cli \
  --dataset subtlememory \
  --system mem0 \
  --run-name mem0-smoke \
  --from-conv 4 \
  --to-conv 5 \
  --smoke \
  --smoke-messages 10 \
  --smoke-questions 1
```

Validate the run:

```bash
uv run python -m evaluation.validate_run \
  --output-dir evaluation/results/subtlememory-mem0-mem0-smoke/subtlememory-mem0-mem0-smoke-api \
  --expected-dataset subtlememory \
  --expected-system mem0
```

For a new system, replace the system and run name:

```bash
uv run python -m evaluation.cli \
  --dataset subtlememory \
  --system my-memory-system \
  --run-name my-memory-system-smoke \
  --from-conv 4 \
  --to-conv 5 \
  --smoke \
  --smoke-messages 10 \
  --smoke-questions 1
```

The corresponding output directory will be:

```text
evaluation/results/subtlememory-my-memory-system-my-memory-system-smoke/
  subtlememory-my-memory-system-my-memory-system-smoke-api/
```

Validate it with:

```bash
uv run python -m evaluation.validate_run \
  --output-dir evaluation/results/subtlememory-my-memory-system-my-memory-system-smoke/subtlememory-my-memory-system-my-memory-system-smoke-api \
  --expected-dataset subtlememory \
  --expected-system my-memory-system
```

## Running the Full Benchmark

After the smoke run validates, run the configured system on the full dataset:

```bash
uv run python -m evaluation.cli \
  --dataset subtlememory \
  --system my-memory-system \
  --run-name my-memory-system-main
```

If ingestion has already completed and `finalize_report.json` is ready, later
stages can be rerun without adding memories again:

```bash
uv run python -m evaluation.cli \
  --dataset subtlememory \
  --system my-memory-system \
  --run-name my-memory-system-main \
  --stages search answer evaluate
```

If you want a separate answer/evaluate experiment using the same search results,
reuse search artifacts:

```bash
uv run python -m evaluation.cli \
  --dataset subtlememory \
  --system my-memory-system \
  --run-name my-memory-system-answer-rerun \
  --reuse-search-artifacts-from evaluation/results/<source-run>/<source-run>-api \
  --stages answer evaluate
```

## Readback Mode

Readback mode is optional. It is useful for diagnosing whether the memory system
wrote the target evidence but failed to retrieve it through normal search. The
normal API benchmark does not require this mode.

Run `api` mode first so the benchmark writes memories, finalizes provider
readiness, and saves `import_manifest.jsonl`:

```bash
uv run python -m evaluation.cli \
  --dataset subtlememory \
  --system my-memory-system \
  --run-name my-memory-system-readback-demo \
  --from-conv 4 \
  --to-conv 5 \
  --smoke \
  --smoke-messages 10 \
  --smoke-questions 1
```

Then run readback using the same add/finalize artifacts:

```bash
uv run python -m evaluation.cli \
  --dataset subtlememory \
  --system my-memory-system \
  --run-name my-memory-system-readback-demo \
  --search-mode readback \
  --reuse-add-artifacts-from evaluation/results/subtlememory-my-memory-system-my-memory-system-readback-demo/subtlememory-my-memory-system-my-memory-system-readback-demo-api \
  --from-conv 4 \
  --to-conv 5 \
  --smoke \
  --smoke-messages 10 \
  --smoke-questions 1 \
  --stages search answer evaluate
```

Use `mem0` in place of `my-memory-system` to run the same diagnostic on Mem0.
For Mem0, the readback run uses the target QA `session_ids` and the reused
manifest rows to locate Mem0 user ids and storage objects; it does not call
Mem0's query search API.

If the adapter does not implement readback hooks, the default result is
`retrieval_status="unsupported"`. That is acceptable for API-only integration,
but readback diagnostics and storage-level evidence coverage will be unavailable.

## Artifact Checklist

Use this checklist when validating a new adapter:

- `run_config_snapshot.json` records the intended dataset, system,
  benchmark mode, stage order, concurrency, and redacted adapter config.
- `source_units.jsonl` is non-empty and has unique `source_unit_id` values.
- `import_manifest.jsonl` is non-empty after `add`.
- Manifest rows include source-unit coverage and, when possible, provider
  memory refs.
- `finalize_report.json` has `ready=true` before search/answer/evaluate.
- `search_results.json` has one row per evaluated question.
- `answer_results.json` and `qa_results.jsonl` align with search question ids.
- `evaluation_results.jsonl` has one row per QA row.
- `score_summary.json` totals match `evaluation_results.jsonl`.

The validation CLI checks these invariants:

```bash
uv run python -m evaluation.validate_run \
  --output-dir <run-mode-output-dir> \
  --expected-dataset subtlememory \
  --expected-system <system-name>
```

The main aggregate score is in `score_summary.json`. For question-level
diagnosis, inspect `qa_results.jsonl`, `search_results.json`, and
`evaluation_results.jsonl`.
