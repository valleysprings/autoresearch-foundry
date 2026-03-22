# autoresearch-with-experience-replay

Strict LLM-required autoresearch for direct Python code generation.

The active flywheel is:

`strategy memory retrieval -> LLM candidate generation -> deterministic materialize/test/benchmark -> winner selection -> memory write-back -> artifacts/UI`

There is no degraded mode:

- no deterministic local fallback
- no offline mode
- no secondary model rotation
- any config or model failure aborts the run

## Theme

This repo is a small autoresearch workbench around one idea:

- the model proposes code mutations
- the verifier stays deterministic and external
- memory stores reusable success and failure experience
- every successful run emits handoff-ready artifacts

The current implementation is intentionally biased toward pure-function, deterministic tasks because that keeps correctness, benchmarking, and experience replay legible.

The benchmark source of truth lives under [benchmark/](/Users/david/coding/2026/autoresearcher-MA/benchmark).

The active research lane is dataset-first:

- each comparable benchmark task is a real local dataset such as `olymmath`, `math-500`, `aime`, `amc`, `planbench`, `sciq`, `qasc`, or `scienceqa`
- each dataset fans out into independent question-runs
- each question-run evolves its own solver trajectory and emits its own artifacts

Only [benchmark/registry.json](/Users/david/coding/2026/autoresearcher-MA/benchmark/registry.json) is intended to sync by default. Dataset assets under `benchmark/`, plus `paper/` and `references/`, stay local.

## 5-Layer Architecture

1. **UI / workbench**
   React + TypeScript + Vite under [ui/](/Users/david/coding/2026/autoresearcher-MA/ui). The selected run is the main focus, with live job polling, artifacts, charts, and memory fragments.
2. **HTTP / job orchestration**
   [app/entries/server.py](/Users/david/coding/2026/autoresearcher-MA/app/entries/server.py) exposes `/api/tasks`, `/api/latest-run`, `/api/runtime`, async job endpoints, and artifact serving.
3. **Codegen engine**
   [app/codegen/](/Users/david/coding/2026/autoresearcher-MA/app/codegen) contains the task catalog, strict config loader, LLM runtime, trainer, verifier, reporting, and handoff helpers.
4. **Memory / reporting / artifacts**
   [app/memory/](/Users/david/coding/2026/autoresearcher-MA/app/memory) persists prompt-ready strategy experiences as JSON + markdown; `runs/` stores payloads, traces, reports, and materialized candidates.
5. **Config / runtime**
   Repo-root `.env` plus shell env override the runtime model settings. The configured model is required for every proposal and memory reflection step.

The active implementation path is [app/codegen/](/Users/david/coding/2026/autoresearcher-MA/app/codegen).

Detailed implementation notes now live next to the code:

- engine, verifier, selection, and handoff design:
  [app/codegen/README.md](/Users/david/coding/2026/autoresearcher-MA/app/codegen/README.md)
- memory layout and write-back rules:
  [app/memory/README.md](/Users/david/coding/2026/autoresearcher-MA/app/memory/README.md)
- benchmark registry and task layout:
  [benchmark/README.md](/Users/david/coding/2026/autoresearcher-MA/benchmark/README.md)

## Configuration

Local configuration is loaded from repo-root `.env`. Shell environment variables override `.env`.

Required keys:

- `AUTORESEARCH_API_KEY`
- `AUTORESEARCH_API_BASE`
- `AUTORESEARCH_PRIMARY_MODEL`
- `AUTORESEARCH_TEMPERATURE`
- `AUTORESEARCH_MAX_TOKENS`
- `AUTORESEARCH_TIMEOUT_S`

Optional:

- `AUTORESEARCH_AVAILABLE_MODELS`
  comma-separated allowlist for the frontend model picker

See [.env.example](/Users/david/coding/2026/autoresearcher-MA/.env.example).

## Benchmarks

The active registry lives in [benchmark/registry.json](/Users/david/coding/2026/autoresearcher-MA/benchmark/registry.json).

For benchmark layout, dataset-task metadata, and how to add a new real dataset, see:
[benchmark/README.md](/Users/david/coding/2026/autoresearcher-MA/benchmark/README.md)

You can cap how many real dataset items a run fans out into:

```bash
python3 -m app.entries.discrete_demo --task olymmath --max-items 100
```

## API And Artifacts

Useful endpoints:

```bash
curl http://127.0.0.1:8000/api/tasks
curl http://127.0.0.1:8000/api/latest-run
curl http://127.0.0.1:8000/api/runtime
curl -X POST "http://127.0.0.1:8000/api/run-task?task_id=olymmath&max_items=100"
```

Successful runs emit artifacts under `runs/`, including:

- dataset-level payload and manifest JSON
- per-question `items/<item_id>.json` summaries
- per-question `item_runs/<item_id>/trace.jsonl`, `llm_trace.jsonl`, `memory.md`, `objective_curve.json`, and `result.json`
- materialized candidate source trees under `runs/workspace/`

## Run The Project

Run one task from the CLI:

```bash
python3 -m app.entries.discrete_demo --task olymmath --max-items 25
```

Run another math benchmark task:

```bash
python3 -m app.entries.discrete_demo --task math-500 --max-items 25
```
