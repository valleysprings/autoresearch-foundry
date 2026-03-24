# autoresearch-foundry

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![UI](https://img.shields.io/badge/UI-React%20%2B%20Vite-61dafb)

### Overview Panel

![Autoresearch panel](fig/panel.png)

## Intro

This repo is a small autoresearch workbench around one idea:

- the model proposes code mutations
- the verifier stays deterministic and external
- memory stores reusable success and failure experience
- every successful run emits handoff-ready artifacts

The current implementation is intentionally biased toward pure-function, deterministic tasks because that keeps correctness, benchmarking, and experience replay legible. This repo was built with substantial help from OpenAI Codex for scaffolding, debugging, refactors, verification, and documentation cleanup.

The benchmark source of truth lives under [benchmark/](benchmark/).

- math comparable benchmarks are real local datasets such as `olymmath`, `math-500`, `aime-2024`, `aime-2025`, and `aime-2026`
- science QA comparable benchmarks are real local datasets such as `sciq`, `qasc`, `scienceqa`, and `openbookqa`
- reasoning benchmarks are currently represented by `planbench` and `arc-challenge`
- coding is now represented by `livecodebench`, wired as a lazy local cache over the full `release_v6` coding set
- long-context benchmarks are currently represented by `longbench-v2`, wired as a lazy local cache over the official 503-question set
- there is no standalone `leetcode` task id; LeetCode-style functional problems run as one branch inside `livecodebench`
- each dataset fans out into independent question-runs
- each question-run evolves its own solver trajectory and emits its own artifacts

Only [benchmark/registry.json](benchmark/registry.json) is intended to sync by default. Dataset assets materialized under `benchmark/`, plus notes under `references/`, stay local.

## 5-Layer Architecture

1. **UI / workbench**
   React + TypeScript + Vite under [ui/](ui/). The selected run is the main focus, with live job polling, artifacts, charts, and memory fragments.
2. **HTTP / job orchestration**
   [app/entries/server.py](app/entries/server.py) exposes `/api/tasks`, `/api/latest-run`, `/api/runtime`, async job endpoints, and artifact serving.
3. **Codegen engine**
   [app/codegen/](app/codegen/) contains the task catalog, strict config loader, LLM runtime, trainer, verifier, reporting, and handoff helpers.
4. **Memory / reporting / artifacts**
   [app/memory/](app/memory/) persists prompt-ready strategy experiences as JSON + markdown; `runs/` stores payloads, traces, reports, and materialized candidates.
5. **Config / runtime**
   Repo-root `.env` carries secrets and model identity; versioned Python config carries the default runtime knobs. The configured model is required for every proposal and memory reflection step.

The active implementation path is [app/codegen/](app/codegen/).

Detailed implementation notes now live next to the code:

- engine, verifier, selection, and handoff design:
  [app/codegen/README.md](app/codegen/README.md)
- memory layout and write-back rules:
  [app/memory/README.md](app/memory/README.md)
- benchmark registry and task layout:
  [benchmark/README.md](benchmark/README.md)
- backend/frontend module map and runtime flow:
  [references/README.md](references/README.md)

## Configuration

Local configuration is loaded from repo-root `.env`. Shell environment variables override `.env`.

Required keys:

- `AUTORESEARCH_API_KEY`
- `AUTORESEARCH_API_BASE`

See [.env.example](.env.example).

## Benchmarks

The active registry lives in [benchmark/registry.json](benchmark/registry.json).

For benchmark layout, dataset-task metadata, and how to add a new real dataset, see:
[benchmark/README.md](benchmark/README.md)

You can cap how many real dataset items a run fans out into:

```bash
python3 -m app --task olymmath --max-items 100
```

`livecodebench` uses the same `--max-items` switch. The first run materializes only the requested prefix of items into a local cache under [benchmark/coding_verified/livecodebench/data/](benchmark/coding_verified/livecodebench/data/), so small runs avoid pulling the full 1055-problem release at once.

LeetCode-style functional items are included inside `livecodebench`; they are not exposed as a separate top-level benchmark. The verifier switches automatically between `stdin` and `class Solution` execution depending on the cached problem metadata.

## Run Surfaces

There are two user-facing ways to run the same backend:

- CLI mode writes artifacts directly under `runs/`
- Web mode serves the UI plus the same run orchestration over HTTP

The shared task runner lives in [app/entries/runner.py](app/entries/runner.py).

The internal [app/entries/](app/entries/) package just means process entrypoints. It is not the recommended user-facing module path anymore.

### Live Execution View

![Live execution view](fig/live%20execution.png)

## API And Artifacts

Useful endpoints:

```bash
python3 -m app serve
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

List available tasks from the CLI:

```bash
python3 -m app --list-tasks
```

Run one task from the CLI:

```bash
python3 -m app --task olymmath --max-items 25
```

Run the coding benchmark from the CLI:

```bash
python3 -m app --task livecodebench --max-items 1
```

Run another math benchmark task:

```bash
python3 -m app --task math-500 --max-items 25
```

Start the web UI and API on `127.0.0.1:8000`:

```bash
cd ui && npm install && npm run build
python3 -m app serve
```

If port `8000` is already occupied, `serve` now defaults to:

- reusing `8000` after stopping a stale autoresearch server process
- otherwise moving to the next free port

You can override that behavior explicitly:

```bash
python3 -m app serve --port 8001
python3 -m app serve --port-conflict next
python3 -m app serve --port-conflict kill
```

## Inspirations

This repo is not a fork of any one project, but a few nearby repos were especially useful as reference points for framing and tradeoff comparison:

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch): agent-driven automated research loop around a tightly scoped editable core
- [algorithmicsuperintelligence/openevolve](https://github.com/algorithmicsuperintelligence/openevolve): open-source evolutionary search framing
- [JARVIS-Xs/SE-Agent](https://github.com/JARVIS-Xs/SE-Agent): self-evolution ideas for code agents
- [evo-eval/evoeval](https://github.com/evo-eval/evoeval): evolving coding benchmarks and evaluation-oriented task design

## License

This repository is released under the [MIT License](LICENSE).
