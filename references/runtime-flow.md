# Runtime Flow

This file explains the actual execution path of the current app.

## 1. Serve Path

1. build the frontend with `cd ui && npm run build`
2. start the HTTP server with `python -m app serve`
3. `app/entries/server.py` serves static files from `ui/dist`
4. the same server also exposes JSON endpoints under `/api/*`

The backend refuses to start if `ui/dist/index.html` is missing.

## 2. Read-Only UI Path

The simplest page load is:

1. browser loads `index.html`
2. `ui/src/main.tsx` mounts `App`
3. `ui/src/App.tsx` asks `ui/src/api.ts` for:
   - `/api/runtime`
   - `/api/tasks`
   - `/api/latest-run`
4. `app/entries/server.py` answers those requests from:
   - `ProposalRuntime.describe()`
   - `list_codegen_task_summaries()`
   - cached run payload JSON on disk

No mutation happens on this path.

## 3. Run-Task Path

When the user clicks run in the UI:

1. `ui/src/api.ts::startJob()` sends `POST /api/run-task?...`
2. `app/entries/server.py` validates query params and creates a background job
3. the background thread calls `write_discrete_artifacts(...)`
4. `app/entries/runner.py::generate_discrete_payload(...)` loads task metadata from the catalog
5. execution splits into one of two branches:

### Branch A: dataset task

Used for `olymmath`, `math-500`, `aime`, `amc`, `planbench-t1`, `planbench-t2`, `planbench-t3`, `sciq`, `qasc`, `scienceqa`, and other dataset-backed tasks.

1. `app/codegen/dataset_runner.py::run_dataset_task(...)`
2. `app/codegen/dataset_support.py::load_question_manifest(...)`
3. build one micro-task per question with `build_micro_task(...)`
4. for each item, call `app/codegen/trainer.py::run_codegen_task(...)`
5. aggregate item-level outcomes back into dataset-level summary fields

The key idea is: the benchmark task is the whole dataset, but evolution happens one question at a time.

### Branch B: non-dataset task

Used for small function benchmarks such as snapshot QA or terminal tasks that are still evaluated as one task at a time.

1. `app/codegen/trainer.py::run_codegen_task(...)`
2. load baseline editable file
3. retrieve memory fragments
4. ask the model for candidate rewrites
5. materialize and verify candidates
6. choose a generation winner
7. optionally write strategy memory

## 4. Single-Task Codegen Loop

`app/codegen/trainer.py::run_codegen_task(...)` is the main optimization loop.

Its logic is:

1. load the checked-in baseline candidate
2. retrieve prompt-ready memories from `MemoryStore`
3. ask `app/codegen/llm.py` for candidate file rewrites
4. materialize each candidate into an isolated workspace
5. send each materialized file to the verifier
6. rank by verifier gate, then `primary_score`, then `tie_break_score`
7. update the frontier only if the winner beats its selected parent by `epsilon`
8. reflect successful or informative failures back into memory

The trainer is stateful across generations through:

- `current_best`
- `frontier`
- `accepted_history`
- `memory store`

## 5. Model Call Path

`app/codegen/llm.py` owns outbound model requests.

Proposal path:

1. build proposal prompt from task metadata, parent candidate, and retrieved memories
2. send one OpenAI-compatible chat request
3. require strict JSON output
4. normalize candidates into a fixed internal shape

Reflection path:

1. take the generation winner or failure case
2. ask for a compact reusable memory summary
3. require strict JSON fields such as `failure_pattern` and `prompt_fragment`

Important constraints:

- no degraded mode
- transport failures and malformed JSON are terminal errors
- requests are throttled through a shared dispatcher keyed by API base

## 6. Verification Path

`app/codegen/verifier.py` owns candidate materialization and scoring.

The generic verifier does:

1. write the candidate file into a workspace directory
2. import the declared entry symbol
3. block forbidden network imports when browsing is disabled
4. run deterministic correctness checks first
5. run benchmark timing only for passing candidates
6. compute:
   - task-facing `objective`
   - internal `objective_score`
   - internal `primary_score`
   - internal `tie_break_score`

Dataset tasks can swap in specialized grading logic:

- science / multiple-choice tasks use `choice_answer_matches(...)`
- math tasks use `app/bench/math_grading.py`
- planning tasks can normalize plan text before exact comparison

## 7. Memory Path

Memory is not raw chat history.

`app/memory/store.py` stores reusable strategy experiences with fields like:

- `failure_pattern`
- `strategy_hypothesis`
- `successful_strategy`
- `prompt_fragment`
- `delta_primary_score`

Retrieval prefers:

- task-signature overlap
- family match
- strong `delta_primary_score`
- successful experiences over noisy failure spam

## 8. Artifact Path

After a run finishes, `app/entries/runner.py` writes a payload under `runs/`.

That payload is what the UI renders.

Artifacts include:

- top-level payload JSON
- per-item summaries for dataset runs
- traces and LLM traces
- markdown memory ledgers
- materialized candidate workspaces

## 9. Why The Structure Looks This Way

The repo is intentionally split by responsibility:

- `app/run.py` selects CLI mode vs web mode
- `server.py` handles HTTP only
- `runner.py` handles whole-run orchestration
- `trainer.py` handles one optimization loop
- `verifier.py` handles correctness and scoring
- `dataset_runner.py` handles dataset fanout
- `MemoryStore` handles durable strategy memory

That separation keeps the flywheel legible:

`request -> task selection -> proposal -> verification -> winner selection -> memory write-back -> artifact emission -> UI`
