# Public Aggregated Result Data

These files are the aggregated data used by the SubtleMemory project page.
They are static public assets and are copied by Vite into the deployed site as
`/data/*.json`.

- `leaderboard.json`: one row per reported paper result, including the main table
  and GPT-5.4 Perfect Retrieval table values.
- `integration_effect.json`: Base vs OpenClaw integration deltas.
- `diagnostic_waterfall.json`: aggregate Memory Preservation Success and
  Retrieval Success Given Preservation tables rendered by the webpage.
- `manifest.json`: generation metadata, public-data policy, and site assets.

The files intentionally contain aggregate metrics only. They do not include
case-level questions, model answers, gold answers, judge rationales, or local
source artifact paths.
