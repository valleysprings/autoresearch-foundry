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

[app/engine.py](/Users/david/coding/2026/autoresearcher-MA/app/engine.py) and [app/evaluator.py](/Users/david/coding/2026/autoresearcher-MA/app/evaluator.py) are compatibility shims. The active implementation path is [app/codegen/](/Users/david/coding/2026/autoresearcher-MA/app/codegen).

## Engine Design

### Proposal runtime

- loads strict model config from `.env` or shell env
- sends one-model OpenAI-compatible chat requests
- expects strict JSON candidates with `function_body`, not operator plans

### Trainer

- retrieves prompt-ready memory fragments
- asks the model for candidate function bodies
- materializes and verifies each candidate
- writes back new success or failure experiences for the current run

### Verifier

- builds a concrete Python module from the candidate body
- imports and executes the target function
- runs fixed correctness tests first
- benchmarks only verified candidates
- computes `objective`, `speed_score`, `stability`, `complexity`, and `J`

### Reporting and handoff

- writes payload JSON, traces, `llm_trace.jsonl`, markdown memory, improvement table, and SVG report
- exposes those artifacts to the UI and to downstream handoff consumers

## Objective Design

The runner uses two related scores:

- **objective**
  the task-facing metric, currently `speedup_vs_baseline`
- **J**
  the deterministic selection score used by the engine

Current `J` formula:

`J = 1.20 * correctness + 0.95 * speed_score + 0.20 * memory_bonus + 0.15 * stability - 0.18 * complexity - 0.05 * (line_count / 10)`

Selection logic:

- correctness is gated first
- failing or erroring candidates do not enter the benchmark lane as winners
- generations mutate a selected frontier parent, not only the global incumbent
- a generation is accepted when it beats its selected parent by `epsilon`
- the global best updates only when a frontier winner also beats the current best by `epsilon`
- passing-but-stagnant candidates do not get written back as failure memory

## Memory Design

Memory is prompt-ready, not a generic log dump.

Each experience stores fields such as:

- `failure_pattern`
- `strategy_hypothesis`
- `successful_strategy`
- `prompt_fragment`
- `tool_trace_summary`
- `delta_J`
- `proposal_model`
- `candidate_summary`
- `experience_outcome`
- `verifier_status`

Important properties:

- memory persists across runs and is not reset to seeds each time
- retrieval prefers **success** experiences and caps failure fragments
- failure memory is reserved for informative verifier failures or execution errors
- duplicate memory fragments are suppressed before write-back
- markdown output is an auditable ledger, but the UI also surfaces run-local fragments directly

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

## Current Examples

Current embedded examples live in [app/codegen/catalog.py](/Users/david/coding/2026/autoresearcher-MA/app/codegen/catalog.py).

They are all pure-function deterministic tasks:

- `contains-duplicates`
- `first-repeated-value`
- `has-overlap`
- `most-frequent-item`
- `deduplicate-preserve-order`
- `missing-number`
- `count-primes-up-to`
- `count-change-ways`
- `count-n-queens`

Why they fit well:

- one function signature
- fixed tests
- fixed deterministic benchmark inputs
- objective and correctness are easy to evaluate without human judgment

## Add A New Example / Task

To add a new embedded task in the current engine, define:

- `id`, `title`, `description`, `family`
- `function_name`, `function_signature`
- `objective_label`, `objective_direction`
- `task_signature`
- `generation_budget`, `candidate_budget`, `epsilon`
- `baseline_imports`, `baseline_body`, `baseline_summary`
- `benchmark`
- `tests`

Use the current engine when the task can be represented as:

- a single Python function body
- deterministic tests
- deterministic benchmark inputs
- a measurable objective like runtime, quality, or both

Add a new benchmark kind when the task needs a new deterministic input generator or score fixture in [app/codegen/verifier.py](/Users/david/coding/2026/autoresearcher-MA/app/codegen/verifier.py).

For longer experiments, you can override budgets from the CLI without editing the catalog:

```bash
python3 -m app.entries.discrete_demo --task count-n-queens --generation-budget 20
```

The current runtime is especially good for:

- heuristic improvement
- constructive heuristics
- local search steps
- order-sensitive data transforms
- search / counting / numeric kernels

## Next Task Families

### Directly compatible with the current engine

- **Route / tour**
  `tsp_nearest_neighbor_tour`, `tsp_two_opt_pass`, `tour_length`
- **Scheduling**
  `interval_schedule`, `list_schedule_makespan`, `weighted_job_order`
- **Selection / portfolio**
  `knapsack_select`, `budgeted_project_pick`, `portfolio_greedy_rebalance`
- **Search / graph / numeric**
  `connected_components_label`, `top_k_frequent`, `merge_intervals`, `shortest_path_relaxation_step`

### Require runtime v2

These are not “just another task” in the current engine:

- tiny-dataset pretrain / fine-tune with `loss` or `PPL`
- model-eval-driven tasks with checkpoints
- tasks that need dataset loading, training steps, optimizer state, or runtime budgeting beyond a pure function call

Those need a different verifier contract:

- training harness
- dataset loader
- checkpoint/temp workspace lifecycle
- objective based on `loss`, `PPL`, or eval score
- runtime and cost budgets that are explicit in the task schema

## Example Specs For Near-Term Roadmap

### `tsp_two_opt_pass(route, distance_matrix)`

- goal: reduce total tour length while preserving tour validity
- correctness: output is a legal permutation / tour
- objective: relative tour-length improvement with runtime tradeoff
- benchmark: fixed matrix sizes and deterministic candidate tours

### `interval_schedule(intervals)`

- goal: maximize the number of non-overlapping intervals
- correctness: returned intervals are valid and mutually non-overlapping
- objective: throughput plus runtime

### `list_schedule_makespan(job_lengths, machine_count)`

- goal: minimize makespan
- correctness: each job is assigned exactly once
- objective: schedule quality plus runtime

### `knapsack_select(items, capacity)`

- goal: maximize value without exceeding capacity
- correctness: feasible subset only
- objective: value quality plus runtime

## API And Artifacts

Useful endpoints:

```bash
curl http://127.0.0.1:8000/api/tasks
curl http://127.0.0.1:8000/api/latest-run
curl http://127.0.0.1:8000/api/runtime
curl -X POST "http://127.0.0.1:8000/api/run-task?task_id=contains-duplicates"
curl -X POST "http://127.0.0.1:8000/api/run-sequence"
```

Successful runs emit artifacts under `runs/`, including:

- payload JSON
- `trace.jsonl`
- `llm_trace.jsonl`
- `memory.md`
- `objective_curve.json`
- task-specific SVG improvement report
- task-specific improvement table JSON
- materialized candidate source trees

## Run The Project

Run one task from the CLI:

```bash
python3 -m app.entries.discrete_demo --task contains-duplicates
```

Run the full sequence:

```bash
python3 -m app.entries.discrete_demo
```

Build the frontend:

```bash
cd ui
npm install
npm run build
```

Frontend-only development:

```bash
cd ui
npm run dev
```

Run the Python server:

```bash
python3 -m app.entries.server
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Current Direction After This Commit

The next visual pass will keep backend-generated SVG reports as canonical artifacts, while modernizing the frontend around:

- better run summary charts
- memory growth / fragment visualizations
- repo-aligned titles and section naming
- light / dark mode
- a more intentional research-workbench layout
