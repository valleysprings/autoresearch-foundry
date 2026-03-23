# Backend Modules

This file is the ownership map for the backend.

## Top-Level Rule

The active implementation lives under `app/`.

Use this mental split:

- `app/run.py`
  public command entry that chooses CLI mode vs web mode
- `app/entries/*`
  internal process entrypoints and orchestration
- `app/codegen/*`
  benchmark loading, proposal generation, verification, and selection
- `app/memory/*`
  durable strategy memory
- `app/configs/*`
  constants, env parsing, and prompt text

## Entry Modules

### `app/run.py`

What it does:

- is the public `python -m app` / `python -m app.run` entry
- dispatches to CLI mode by default
- dispatches to web mode when called as `python -m app serve`

Why it exists:

- users should not need to know internal module names under `app.entries`
- it keeps CLI mode and web mode under one Python entry surface

### `app/entries/server.py`

What it does:

- serves built frontend assets from `ui/dist`
- exposes `/api/tasks`, `/api/runtime`, `/api/latest-run`, `/api/run-task`, `/api/run-sequence`, `/api/job`
- manages background job state in memory

What it does not do:

- it does not run selection logic itself
- it does not know benchmark semantics

Main dependency chain:

`server.py -> runner.py -> trainer.py / dataset_runner.py`

### `app/entries/runner.py`

What it does:

- loads task catalog entries
- runs one task or the full comparable sequence
- chooses dataset vs non-dataset execution path
- assembles the final payload JSON written under `runs/`

Think of it as:

the run-level conductor above the task-level trainer.

## Config Modules

### `app/configs/paths.py`

Central path constants:

- repo root
- benchmark root
- registry path
- runs root

Everything else imports from here instead of hardcoding paths.

### `app/configs/runtime.py`

Defines environment keys and runtime defaults such as:

- required API env vars
- available-models env key
- LLM concurrency default

### `app/configs/prompts.py`

Prompt contract constants for:

- proposal generation
- reflection generation
- JSON shape requirements
- truncation and preview limits

This is the prompt policy layer.

### `app/configs/codegen.py`

Central constants for the optimization loop:

- seed strategy memories
- `J` scoring weights
- dataset instructions
- network-block rules
- complexity heuristics

If you want to understand selection behavior, start here.

## Catalog And Task Loading

### `app/codegen/catalog.py`

What it owns:

- reading `benchmark/registry.json`
- loading each task's `task.json`
- validating required fields
- deriving task summaries for the UI
- sorting tasks by benchmark tier, track, and preferred order

Why it matters:

- everything else assumes the catalog has already normalized task metadata
- if a task is missing local assets, the loader can skip it instead of crashing the whole UI

## Runtime Config And Model Transport

### `app/codegen/config.py`

What it owns:

- `.env` parsing
- environment validation
- runtime config normalization
- model selection overrides

This module turns shell state into a strict `RuntimeConfig`.

### `app/codegen/llm.py`

What it owns:

- request construction for proposal and reflection calls
- transport dispatching with bounded concurrency
- JSON extraction and validation
- truncated-response handling
- normalized candidate and reflection payloads

This is the only module that should talk to the model API directly.

## Optimization And Selection

### `app/codegen/trainer.py`

What it owns:

- baseline loading
- generation loop
- frontier management
- parent selection for branches
- candidate ranking and winner selection
- memory write-back decisions

Core idea:

`trainer.py` is the one-task evolution loop.

### `app/codegen/verifier.py`

What it owns:

- materializing candidate files into a workspace
- importing the entry symbol
- running deterministic tests
- benchmarking passing candidates
- computing final metrics including `objective_score` and `J`

It is the gatekeeper between model output and accepted progress.

### `app/codegen/errors.py`

Typed runtime errors used across backend and UI transport:

- `ConfigError`
- `LlmTransportError`
- `LlmResponseError`
- `VerificationError`

These are designed to serialize cleanly into API payloads.

## Dataset-Specific Execution

### `app/codegen/dataset_support.py`

What it owns:

- loading normalized question manifests
- shaping public question payloads
- building item-level micro-tasks
- dataset summary aggregation helpers

This module translates:

`dataset task -> question micro-task`

### `app/codegen/dataset_runner.py`

What it owns:

- fanout across dataset items
- per-item memory stores
- optional parallel item execution
- re-aggregation back into one dataset-level run record

If `trainer.py` is the single-task loop, `dataset_runner.py` is the dataset wrapper around that loop.

### `app/codegen/benchmark_support.py`

Shared grading helpers:

- text normalization
- numeric normalization
- multiple-choice answer matching
- set-style answer matching
- hiding expected answers from solver-visible payloads

It is the shared benchmark math for many verifiers.

### `app/codegen/math_grading.py`

Math-specific grading layer.

What it adds on top of `benchmark_support.py`:

- choice grading for math multiple-choice items
- canonical numeric matching
- symbolic equivalence checks via `math-verify`

Use this when a math dataset cannot be graded by plain string equality.

## Memory Modules

### `app/memory/store.py`

What it owns:

- reading and writing memory JSON
- deduplicating experiences
- seeding default experiences
- ranking retrieval candidates

Important design choice:

it stores prompt-ready strategy fragments, not generic event logs.

### `app/memory/markdown.py`

What it owns:

- rendering the memory ledger into human-readable markdown

This is the audit-friendly view of the same memory store.

## Benchmark Folder

### `benchmark/registry.json`

This is the source of truth for which tasks are active.

### `benchmark/<track>/<task>/task.json`

Defines:

- task identity
- answer metric
- benchmark tier
- editable file
- verifier file
- dataset manifest path when applicable

### `benchmark/<track>/<task>/prepare.py`

Converts raw external data into the shared local manifest shape.

### `benchmark/<track>/<task>/verifier.py`

Task-specific correctness logic when the generic verifier is not enough.

## Quick Ownership Summary

- need to add or remove a benchmark task: `benchmark/registry.json`, `benchmark/.../task.json`, `app/codegen/catalog.py`
- need to change model request format: `app/codegen/llm.py`, `app/configs/prompts.py`
- need to change selection logic: `app/codegen/trainer.py`, `app/configs/codegen.py`
- need to change candidate scoring: `app/codegen/verifier.py`, `app/codegen/benchmark_support.py`, `app/codegen/math_grading.py`
- need to change memory retrieval/write-back: `app/memory/store.py`, `app/codegen/trainer.py`
- need to change API behavior: `app/entries/server.py`
