# EvoSkill Foundry 😎

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![UI](https://img.shields.io/badge/UI-React%20%2B%20Vite-61dafb)

### Overview Panel

![EvoSkill Foundry panel](assets/panel.png)

## Intro

This repo is a benchmark-native workbench for verification-gated agent search.

The core idea is simple:

- the model proposes mutations
- the verifier stays external and deterministic
- selection is gated by task validity first, then by the task-native objective
- memory stores reusable success and failure experience
- every successful run emits handoff-ready artifacts

The web UI is a workbench, not the platform boundary. The platform boundary is the task contract plus the CLI/API surfaces that execute it.

## Setup

Bootstrap the environment with the team-default Conda env:

```bash
conda activate autoresearch
uv sync --active
cp llm_profiles.example.toml llm_profiles.toml
cp .env.example .env
```

The default workflow is to keep Python dependencies in the active `autoresearch` Conda environment and let `uv.lock` pin the exact resolved set. If dependency ranges change, run `uv lock`, then refresh the env with `uv sync --active`.

After `uv sync --active`, you can run commands directly from the active Conda env or prefix them with `uv run`.

## Task Contract

Each active task declares two orthogonal dimensions in `task.json`:

- `task_mode`
  what the candidate entrypoint represents at evaluation time:
  - `answer`
  - `artifact`
- `interaction_mode`
  how interaction unfolds at evaluation time:
  - `single_turn`
  - `multi_turn`

The task-facing metric is dataset-specific.

- `answer_metric` names the benchmark-native metric
- `objective_spec` defines its display name, direction, unit, and formula
- `primary_score` is the normalized comparison scalar used by selection
- `tie_break_score` is optional and should only break near-ties

For `answer` tasks, `editable.py` is only the repo's internal candidate entrypoint for the search/eval harness. It does not mean the benchmark itself is an implementation or coding task.

Read [references/evaluation-contract.md](references/evaluation-contract.md) before changing metric semantics, adding tie-break logic, or changing benchmark task contracts.

## Benchmark Tiers

The benchmark source of truth lives under [benchmark/](benchmark/), with active registration in [benchmark/registry.json](benchmark/registry.json).

Enabled registry entries now form the current maintained runnable benchmark task set:

- math: `olymmath`, `math-500`, `aime-2024`, `aime-2025`, `aime-2026`
- reasoning: `planbench-t1`, `planbench-t2`, `planbench-t3`, `arc-challenge`, `bbh`, `mmlu-pro`
- long-context: `longbench-v2`
- science QA: `sciq`, `qasc`, `scienceqa`, `openbookqa`, `gpqa-diamond`
- coding: `livecodebench-v1` to `livecodebench-v6`
- text-to-SQL: `spider`, `bird`, `chase`
- Optimization: `co-bench`
- personalization: `incharacter`, `characterbench`, `socialbench`, `personafeedback`, `personamem-32k`, `alpsbench-extraction`, `alpsbench-update`, `alpsbench-retrieval`, `alpsbench-utilization`, `alpbench`
- safety: `xstest-refusal-calibration`, `harmbench-text-harmful`, `jailbreakbench-harmful`, `or-bench-hard-1k`, `or-bench-toxic`, `longsafety`

Active benchmark membership is defined by `benchmark/registry.json`. Tasks that should stay out of the main set should set `enabled: false` in the registry or `included_in_main_comparison: false` in their task spec.

Current setup notes:

- PlanBench shared verifier support lives in `app/bench/planbench_support.py`, with shared validator assets under `benchmark/reasoning_verified/planbench-shared/official/`
- all enabled tasks above are the current maintained benchmark set

## Dataset Preparation

Preparing benchmark-local datasets is a prerequisite after clone.

Run:

```bash
conda run -n autoresearch python benchmark/prepare_datasets.py
```

If a benchmark is missing local assets, the runtime should now point you at the matching `python benchmark/prepare_datasets.py --task-id ...` command, and OR benchmark setup is materialized under each task's own `benchmark/.../data/` directory.

## CLI And API

The CLI is JSON-first and mirrors the workbench API surfaces.

- `python -m app tasks`
  mirrors `/api/tasks`
- `python -m app runtime`
  mirrors `/api/runtime`
- `python -m app latest-run --task-id livecodebench-v6`
  mirrors `/api/latest-run?task_id=livecodebench-v6`
- `python -m app run-task --task-id livecodebench-v6 --max-items 1`
  mirrors `/api/run-task?task_id=livecodebench-v6&max_items=1`, but runs synchronously and prints the final payload
- `python -m app run-sequence --max-items 25`
  mirrors `/api/run-sequence?max_items=25`, but runs synchronously and prints the final payload

If you are not already inside `conda activate autoresearch`, prefix those commands with `conda run -n autoresearch`.

Useful flags:

- `--model`
- `--llm-concurrency`
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
uv run python -m app tasks --pretty
uv run python -m app tasks --task-id livecodebench-v6 --pretty
uv run python -m app runtime
uv run python -m app run-task --task-id livecodebench-v6 --max-items 3 --pretty
uv run python -m app latest-run --task-id livecodebench-v6 --pretty
```

## Web Workbench

The UI consumes the same backend payload shapes and is mainly for live inspection.

### Live Execution View

![Live execution view](assets/live%20execution.png)

Build and serve the UI:

```bash
cd ui && npm install && npm run build
uv run python -m app serve
```

If port `8000` is occupied, `serve` defaults to:

- reusing `8000` after stopping a stale autoresearch server process
- otherwise moving to the next free port

Explicit overrides:

```bash
uv run python -m app serve --port 8001
uv run python -m app serve --port-conflict next
uv run python -m app serve --port-conflict kill
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
   repo-root `.env` carries secrets; repo-root `llm_profiles.toml` selects the active endpoint profile and transport knobs.

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

Runtime configuration is loaded from repo-root `llm_profiles.toml`.

Secrets are loaded from repo-root `.env`, and shell environment variables override `.env`.

Required files:

- `llm_profiles.toml`
- `.env` when a profile uses `api_key_env`

Optional environment override:

- `AUTORESEARCH_LLM_PROFILE` to force a different profile than `active_profile`

When dependency ranges change, regenerate the lockfile with `uv lock`.

See `.env.example` and `llm_profiles.example.toml`.

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
