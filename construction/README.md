# SubtleMemory Data Construction

This directory contains the code used to construct the SubtleMemory benchmark.
The core construction idea is to embed latent relation-controlled semantic
artifacts implicitly into user histories, then evaluate whether memory systems
can recover and use those relations through later questions. Construction
creates persona-scoped benchmark bundles, while `evaluation/` runs memory
systems on those bundles.

## Pipeline

The benchmark construction process has five stages:

1. **Semantic seed selection:** collect high-quality user-related and
   user-unrelated source facts from open-source seed data.
2. **Semantic artifact creation:** generate latent relation-controlled semantic
   artifacts for complementary, nuanced, and contradictory cases.
3. **Session construction:** embed those artifacts implicitly into natural
   multi-turn user histories while preserving relation-critical context.
4. **Evaluation instance construction:** generate questions, correct answers,
   incorrect answers, source facts, and target session provenance.
5. **User-history assembly:** merge user-related and user-unrelated sessions
   into chronological persona histories and export final bench bundles.

The final released bundle format is:

```text
data/subtlememory/
  persona_0/
    bench_instances.json
    history_sessions.json
  ...
  persona_9/
    bench_instances.json
    history_sessions.json
```

## Layout

| Path | Purpose |
| --- | --- |
| `user-related/` | Builds persona-grounded relation cases from PersonaMem-style user profiles and preferences. |
| `user-unrelated/` | Builds non-user factual relation cases from QA and contextual seed data. |
| `merge_user_related_unrelated.py` | Merges completed user-related runs and passed user-unrelated cases into persona-scoped bench bundles. |
| `user-related/DATA_FLOW.md` | Detailed user-related construction flow and artifact layout. |
| `user-unrelated/DATA_FLOW.md` | Detailed user-unrelated command flow and artifact layout. |
| `user-related/PROMPTS.md` | User-related generation and filtering prompt inventory. |
| `user-related/FILTER_STANDARDS.md` | User-related filtering criteria. |
| `user-unrelated/FILTER_STANDARDS.md` | User-unrelated filtering criteria. |

## Environment

Copy the root template and fill in OpenAI-compatible endpoints:

```bash
cp env.template .env
```

The construction code uses:

```text
GENERATION_BASE_URL
GENERATION_API_KEY
FILTER_BASE_URL
FILTER_API_KEY
DATA_CONSTRUCTION_PERSONAMEM_INPUT_DIR
DATA_CONSTRUCTION_USER_RELATED_OUTPUT_DIR
```

`DATA_CONSTRUCTION_PERSONAMEM_INPUT_DIR` defaults to `data/personamem-raw`.
`DATA_CONSTRUCTION_USER_RELATED_OUTPUT_DIR` defaults to `data/user-related`.
These are local construction working paths. The repository release data should
remain under `data/subtlememory`.

## User-Related Construction

Run from the repository root:

```bash
uv run python construction/user-related/main.py build \
  --config construction/user-related/examples/build-config.related-10-diverse-personas.json
```

Useful checks and variants:

```bash
uv run python construction/user-related/main.py build --dry-run

uv run python construction/user-related/main.py build \
  --config construction/user-related/examples/build-config.persona0.json

uv run python construction/user-related/main.py build-from \
  --source-run data/user-related/<source_run> \
  --from-stage conversation_generation \
  --run-id <new_run_id>
```

The user-related workflow produces persona profiles, topic groups, relation
plans, generated cases, filtered cases, sessions, QA pairs, and evaluation
instances. See `user-related/DATA_FLOW.md` for the full artifact contract.

## User-Unrelated Construction

The user-unrelated workflow expects local construction inputs under:

```text
data/user-unrelated/config/
data/user-unrelated/source_data/
```

Run generation, filtering, and passed-dataset assembly from the repository root:

```bash
uv run python construction/user-unrelated/main.py generate-to-filter \
  --categories all \
  --count 10

uv run python construction/user-unrelated/main.py filter \
  --categories all

uv run python construction/user-unrelated/main.py build-passed-dataset
```

The normalized passed file is written to:

```text
data/user-unrelated/outputs_selected/all_passed_samples.json
```

See `user-unrelated/DATA_FLOW.md` for the command-level data flow and
intermediate artifact paths.

## Merge Into Benchmark Bundles

After completing a user-related run and preparing passed user-unrelated cases,
merge them into persona-scoped bench bundles:

```bash
uv run python construction/merge_user_related_unrelated.py \
  --related-run-dir data/user-related/<run_id> \
  --unrelated-input data/user-unrelated/outputs_selected/all_passed_samples.json \
  --output-dir data/merged-user-memory/<merge_id> \
  --qa-mode task
```

After manual review or final selection, copy the selected `persona_*` folders
into the released bench location:

```bash
mkdir -p data/subtlememory
rsync -a data/merged-user-memory/<merge_id>/persona_* data/subtlememory/
```

For evaluation, copy the released bench data into the evaluation data directory:

```bash
mkdir -p evaluation/data/subtlememory
rsync -a data/subtlememory/ evaluation/data/subtlememory/
```

## Validation

Basic entry-point checks:

```bash
uv run python construction/user-related/main.py --help
uv run python construction/user-unrelated/main.py --help
uv run python construction/merge_user_related_unrelated.py --help
uv run python -m compileall -q construction
```
