# Evaluation Contract

This note defines how the repo evaluates tasks. It is meant to keep task-facing metrics clear across `answer`, `artifact`, and `agent` modes.

## 1. Core Contract

Every benchmark task should declare three orthogonal dimensions in `task.json`:

- `task_mode`
  what the editable file represents at evaluation time:
  - `answer`: candidate code returns the final answer for one item
  - `artifact`: candidate code defines or produces an executable artifact that the verifier runs
  - `agent`: candidate code defines a policy, wrapper, or harness config for a multi-step environment
- `optimization_scope`
  what is allowed to change:
  - `prompt`: mutate prompt templates only
  - `wrapper`: mutate the wrapper, policy shell, prompting, and run config
  - `implementation`: mutate the solver or program itself
- `runtime_backend`
  where the task is executed:
  - `dataset`: local runtime; `local_dataset_only=true` means dataset fan-out with per-item runs
  - `external`: legacy custom benchmark harness execution, currently disabled in `benchmark/registry.json`

These fields say what is being optimized. They do not define the headline metric by themselves.

## 2. Primary Metric

The headline metric is always dataset-specific.

- `answer_metric` names the task-native metric, for example `accuracy`, `exact_plan_match_rate`, `test_pass_rate`, `success_rate`, or `avg_test_score`
- `objective_spec` explains that metric in task-facing terms:
  - `display_name`
  - `direction`
  - `unit`
  - `formula`
- `primary_score` is not a separate benchmark concept like F1 or accuracy
  it is the normalized comparison scalar used by selection
- in the current implementation:
  - `primary_score = objective_score`
  - `objective_score` is just the task objective normalized so that "higher is better"

So the correct question is not "is primary F1 or accuracy?" The correct question is "what is this task's `objective_spec`, and how is it normalized into `primary_score`?"

## 3. Gate Before Tie-Break

Selection is layered.

- `gate_passed`
  verifier-side eligibility check; candidates that fail the task gate cannot win
- `primary_score`
  task-native objective used for real ranking
- `tie_break_score`
  optional weak preference used only after primary scores are effectively tied

Tie-break metrics should stay secondary. They are not the benchmark headline.

Current examples:

- most tasks use the `objective_only` profile
- `planbench` uses `plan_length`
  shorter valid plans only matter after solved ratio is tied
- optimization-style profiles may include weak preferences like `stability`, `complexity`, or `line_count`

`complexity` should therefore be read as a weak implementation preference, not as the main scientific claim of a benchmark result.

## 4. Mode-Specific Guidance

### `answer`

Use `answer` when `editable.py` returns the final answer for a single item.

Typical repo examples:

- math benchmarks
- science QA
- reasoning tasks such as `planbench` and `arc-challenge`
- long-context QA

Expected metric pattern:

- gate: answer is valid or exactly verifiable
- primary: accuracy, exact match, answer-level score, or another dataset-native answer metric
- tie-break: usually none; use only when the dataset's purpose supports it

### `artifact`

Use `artifact` when `editable.py` defines executable code or another artifact that is itself evaluated.

Typical repo examples:

- `livecodebench`
  editable Python program artifact
- `nl4opt` and `industryor`
  generated `coptpy` artifact
- `co-bench`
  generated `solve(**kwargs)` artifact scored by the checked-in official evaluator

Expected metric pattern:

- gate: artifact compiles, executes, or is verifier-valid
- primary: pass rate, exact optimal-value match rate, normalized benchmark score, or another artifact-level result metric
- tie-break: only weak preferences such as artifact simplicity or stability, if needed

### `agent`

Use `agent` when `editable.py` defines a policy or harness wrapper that is evaluated over a multi-step environment.

Typical repo examples:

- `terminal-bench`
- `tau-bench-retail`
- `tau-bench-airline`

Expected metric pattern:

- gate: harness run is valid and produces usable task results
- primary: task success rate or reward-like benchmark metric
- tie-break: only secondary operational preferences, and only when they do not distort the benchmark's purpose

## 5. Comparable vs Experiment Tasks

This repo uses two benchmark tiers:

- `comparable`
  local, relatively stable tasks included in the main comparison set
- `experiment`
  heavier or not-yet-normalized tasks that are useful, but not yet part of the main headline comparison

Current mapping:

- `comparable`
  math, science QA, reasoning, long-context, and `livecodebench`
- `experiment`
  `co-bench`

Checked-in but currently disabled external-harness tasks:

- `terminal-bench`
- `tau-bench-retail`
- `tau-bench-airline`
- `nl4opt`
- `industryor`

This separation is deliberate: some tasks are scientifically interesting before they are cheap, stable, or normalized enough for headline comparison.

## 6. Dependency Burden

Not every checked-in non-comparable benchmark has the same operational cost.

- `terminal-bench`
  requires Harbor plus a working local Docker daemon
- `tau-bench-*`
  requires isolated environment setup, but not the Harbor Docker path
- `nl4opt` and `industryor`
  require local `coptpy` execution
- `co-bench`
  uses dataset fan-out plus the checked-in official evaluation framework and local dataset assets

`external` in this repo is about execution shape, not about whether the task is local, cloned, or missing dependencies. `co-bench` is a useful counterexample: it is still an experiment benchmark, but it now uses the generic dataset runner with a benchmark-specific checked-in evaluator and remains enabled while the harness-backed tasks stay disabled.

Do not describe these as if they all have the same setup burden.

## 7. Checklist For New Tasks

When adding a benchmark task, write down:

1. what the benchmark is trying to measure
2. which `task_mode` it belongs to
3. what the verifier gate is
4. what the single headline metric is
5. how that metric is encoded in `objective_spec`
6. whether any tie-break is truly needed
7. whether the task belongs in `comparable` or `experiment`

If these seven items are not explicit, the benchmark contract is still underspecified.
