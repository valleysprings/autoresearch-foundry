# autoresearch-foundry

## Motivation

This repo is a workbench for one question: can an LLM keep improving Python solutions when generation stays inside the model, but checking and selection stay outside it?

The project is intentionally strict about that boundary. The model is not an optional helper, and the verifier is not allowed to drift into another opaque model call. Every run depends on a configured LLM, every candidate is materialized locally, and every comparison is decided by deterministic tests or benchmarks.

That constraint matters because the useful artifact is not just the latest answer. It is the trail of strategies, failures, and write-backs that can be reused in the next run. The goal is to make iterative code research inspectable instead of magical.

This repo was built with substantial help from OpenAI Codex for scaffolding, debugging, refactors, verification, and documentation cleanup. It deliberately was not built with Claude in this iteration because the cost profile and usage limits were a worse fit for repeated local agent loops.

## Theme

This repo is a small autoresearch workbench around one idea:

- the model proposes code mutations
- the verifier stays deterministic and external
- memory stores reusable success and failure experience
- every successful run emits handoff-ready artifacts

The current implementation is intentionally biased toward pure-function, deterministic tasks because that keeps correctness, benchmarking, and experience replay legible.

The benchmark source of truth lives under [benchmark/](benchmark/).

The active research lane is dataset-first:

- each comparable benchmark task is a real local dataset such as `olymmath`, `math-500`, `aime-2024`, `aime-2025`, `aime-2026`, `planbench`, `sciq`, `qasc`, or `scienceqa`
- coding is now represented by `livecodebench`, wired as a lazy local cache over the full `release_v6` coding set
- there is no standalone `leetcode` task id; LeetCode-style functional problems run as one branch inside `livecodebench`
- each dataset fans out into independent question-runs
- each question-run evolves its own solver trajectory and emits its own artifacts

Only [benchmark/registry.json](benchmark/registry.json) is intended to sync by default. Dataset assets under `benchmark/`, plus notes under `references/`, stay local.

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

Optional:

- `AUTORESEARCH_PRIMARY_MODEL`
- `AUTORESEARCH_AVAILABLE_MODELS`
  temporary shell overrides; the committed defaults live in [app/configs/runtime.py](app/configs/runtime.py)
- `AUTORESEARCH_TEMPERATURE`
- `AUTORESEARCH_MAX_TOKENS`
- `AUTORESEARCH_TIMEOUT_S`
  temporary shell overrides; the committed defaults live in [app/configs/runtime.py](app/configs/runtime.py)

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
