# autoresearch-foundry

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![UI](https://img.shields.io/badge/UI-React%20%2B%20Vite-61dafb)

### Overview Panel

![Autoresearch panel](assets/panel.png)

## Intro

This repo is a benchmark-native workbench for verification-gated agent search.

The core idea is simple:

- the model proposes mutations
- the verifier stays external and deterministic
- selection is gated by task validity first, then by the task-native objective
- memory stores reusable success and failure experience
- every successful run emits handoff-ready artifacts

The web UI is a workbench, not the platform boundary. The platform boundary is the task contract plus the CLI/API surfaces that execute it.

## Task Contract

Each task declares three orthogonal dimensions in `task.json`:

- `task_mode`
  what the editable file represents:
  - `answer`
  - `artifact`
  - `agent`
- `optimization_scope`
  what is allowed to change:
  - `prompt`
  - `wrapper`
  - `implementation`
- `runtime_backend`
  active registry tasks currently execute under `dataset`
  `external` remains a legacy disabled path in the codebase; direct local tasks and dataset fan-out both use `dataset`

The task-facing metric is dataset-specific.

- `answer_metric` names the benchmark-native metric
- `objective_spec` defines its display name, direction, unit, and formula
- `primary_score` is just the normalized comparison scalar used by selection
- `tie_break_score` is optional and should only break near-ties

Read [references/evaluation-contract.md](references/evaluation-contract.md) before changing metric semantics, adding tie-break logic, or writing new benchmark wrappers.

## Benchmark Tiers

The benchmark source of truth lives under [benchmark/](benchmark/), with active registration in [benchmark/registry.json](benchmark/registry.json).

Two tiers are supported:

- `comparable`
  main comparison tasks with local or relatively stable evaluation paths
- `experiment`
  useful heavier tasks that are not yet part of the main headline comparison

Current `comparable` tasks:

- math: `olymmath`, `math-500`, `aime-2024`, `aime-2025`, `aime-2026`
- science QA: `sciq`, `qasc`, `scienceqa`, `openbookqa`
- reasoning: `planbench`, `arc-challenge`
- long-context: `longbench-v2`
- coding: `livecodebench`

Current `experiment` tasks:

- OR and optimization tasks: `co-bench`

Current enabled experiment-task setup note:

- `co-bench` uses dataset fan-out with the checked-in official evaluator plus local dataset assets

Checked-in but currently disabled external-harness tasks:

- `terminal-bench` requires Harbor plus a working local Docker daemon
- `tau-bench-*` requires isolated environment setup, but not the Harbor Docker path
- `nl4opt` and `industryor` require local `coptpy` execution

## Dataset Preparation

Preparing benchmark-local datasets is a prerequisite after clone.

Run:

```bash
python benchmark/prepare_datasets.py
```

If a benchmark is missing local assets, the runtime should now point you at the matching `python benchmark/prepare_datasets.py --task-id ...` command, and OR benchmark setup is materialized under each task's own `benchmark/.../data/` directory.

## CLI And API

The CLI is JSON-first and mirrors the workbench API surfaces.

- `python -m app tasks`
  mirrors `/api/tasks`
- `python -m app runtime`
  mirrors `/api/runtime`
- `python -m app latest-run --task-id livecodebench`
  mirrors `/api/latest-run?task_id=livecodebench`
- `python -m app run-task --task-id livecodebench --max-items 1`
  mirrors `/api/run-task?task_id=livecodebench&max_items=1`, but runs synchronously and prints the final payload
- `python -m app run-sequence --max-items 25`
  mirrors `/api/run-sequence?max_items=25`, but runs synchronously and prints the final payload

Useful flags:

- `--model`
- `--generation-budget`
- `--candidate-budget`
- `--branching-factor`
- `--item-workers`
- `--max-items`
- `--external-config '{"n_tasks": 3}'`
  only on `run-task`, matching the server-side `external_config` request body for legacy external-harness tasks and fixtures
- `--pretty`
  render human-readable summaries instead of JSON

Examples:

```bash
python -m app tasks --pretty
python -m app tasks --task-id livecodebench --pretty
python -m app runtime
python -m app run-task --task-id livecodebench --max-items 3 --pretty
python -m app latest-run --task-id livecodebench --pretty
```

## Web Workbench

The UI consumes the same backend payload shapes and is mainly for live inspection.

### Live Execution View

![Live execution view](assets/live%20execution.png)

Build and serve the UI:

```bash
cd ui && npm install && npm run build
python -m app serve
```

If port `8000` is occupied, `serve` defaults to:

- reusing `8000` after stopping a stale autoresearch server process
- otherwise moving to the next free port

Explicit overrides:

```bash
python -m app serve --port 8001
python -m app serve --port-conflict next
python -m app serve --port-conflict kill
```

## Artifacts

Successful runs emit artifacts under `runs/`, including:

- dataset-level payload and manifest JSON
- per-question `items/<item_id>.json` summaries
- per-question `item_runs/<item_id>/trace.jsonl`, `llm_trace.jsonl`, `memory.md`, `objective_curve.json`, and `result.json`
- materialized candidate source trees under `runs/workspace/`

The shared task runner lives in [app/entries/runner.py](app/entries/runner.py).

## 5-Layer Architecture

1. **UI / workbench**
   React + TypeScript + Vite under [ui/](ui/).
2. **HTTP / job orchestration**
   [app/entries/server.py](app/entries/server.py) exposes `/api/tasks`, `/api/runtime`, `/api/latest-run`, async job endpoints, and artifact serving.
3. **Codegen engine**
   [app/codegen/](app/codegen/) contains the task catalog, runtime, verifier logic, selection policy, trainer, and handoff helpers.
4. **Memory / reporting / artifacts**
   [app/memory/](app/memory/) persists strategy experiences; `runs/` stores traces, reports, payloads, and materialized workspaces.
5. **Config / runtime**
   repo-root `.env` carries secrets and model identity; versioned Python config carries the default runtime knobs.

Detailed implementation notes:

- engine, verifier, selection, and handoff design:
  [app/codegen/README.md](app/codegen/README.md)
- memory layout and write-back rules:
  [app/memory/README.md](app/memory/README.md)
- benchmark registry and task layout:
  [benchmark/README.md](benchmark/README.md)
- repo map and execution flow:
  [references/README.md](references/README.md)

## Configuration

Local configuration is loaded from repo-root `.env`. Shell environment variables override `.env`.

Required keys:

- `AUTORESEARCH_API_KEY`
- `AUTORESEARCH_API_BASE`

See `.env.example`.

## Adding Benchmarks

For benchmark layout, dataset-task metadata, and how to add a new task, read:

- [benchmark/README.md](benchmark/README.md)
- [references/evaluation-contract.md](references/evaluation-contract.md)

The important rule is that each dataset must state one clear headline metric and one clear verifier gate. Do not let `primary_score`, `tie_break_score`, and implementation preferences blur the actual purpose of the dataset.

## Inspirations

This repo is not a fork of any one project, but a few nearby repos were especially useful as reference points:

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- [algorithmicsuperintelligence/openevolve](https://github.com/algorithmicsuperintelligence/openevolve)
- [JARVIS-Xs/SE-Agent](https://github.com/JARVIS-Xs/SE-Agent)
- [evo-eval/evoeval](https://github.com/evo-eval/evoeval)

## License

This repository is released under the [MIT License](LICENSE).
