# Evaluation Contract

This note defines how the repo evaluates tasks. It keeps task-facing semantics clear across `answer` and `artifact` task modes plus `single_turn` and `multi_turn` interaction modes.

## 1. Core Contract

Every active benchmark task should declare two task-kind dimensions in `task.json`:

- `task_mode`
  what the candidate entrypoint represents at evaluation time:
  - `answer`: the candidate entrypoint returns the final answer or output for one evaluated item
  - `artifact`: the returned or generated program, policy, script, or other artifact is itself what the verifier runs or consumes; multi-turn policies also live here
- `interaction_mode`
  how interaction unfolds at evaluation time:
  - `single_turn`: one candidate invocation returns the final result for each item
  - `multi_turn`: candidate code interacts over repeated observations, actions, and state updates inside an episode

For the active local benchmark lane, `local_dataset_only=true` means the repo fans evaluation out across local dataset items or episodes. Legacy fields such as `runtime_backend` and `optimization_scope` are no longer active contract surface.

These fields say what kind of task is being evaluated. They do not define the headline metric by themselves.

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
- `planbench-t1` and `planbench-t2` use `plan_length`
  shorter valid plans only matter after solved ratio is tied
- optimization-style profiles may include weak preferences like `stability`, `complexity`, or `line_count`

`complexity` should therefore be read as a weak implementation preference, not as the main scientific claim of a benchmark result.

## 4. Mode-Specific Guidance

### `answer`

Use `answer` when the candidate entrypoint returns the final answer or output for a single item.

Typical repo examples:

- math benchmarks
- science QA
- reasoning tasks such as `planbench-t1`, `planbench-t2`, `planbench-t3`, and `arc-challenge`
- long-context QA

Important repo detail:

- `editable.py` is only the repo's compatibility entrypoint for search/eval
- it does not mean the benchmark itself is an implementation or coding task
- the verifier judges the returned answer/output, not the existence of a standalone program artifact

Expected metric pattern:

- gate: answer is valid or exactly verifiable
- primary: accuracy, exact match, answer-level score, or another dataset-native answer metric
- tie-break: usually none; use only when the dataset's purpose supports it

### `artifact`

Use `artifact` when the candidate defines or produces executable code or another artifact that is itself evaluated.

Typical repo examples:

- `livecodebench`
  returned Python program artifact
- `co-bench`
  generated `solve(**kwargs)` artifact scored by the checked-in official evaluator

Expected metric pattern:

- gate: artifact compiles, executes, or is verifier-valid
- primary: pass rate, exact optimal-value match rate, normalized benchmark score, or another artifact-level result metric
- tie-break: only weak preferences such as artifact simplicity or stability, if needed

## 5. Active Benchmark Task Set

Enabled registry entries now form one active benchmark task set.

- dataset-backed tasks span math, reasoning, text-to-SQL, long-context, personalization, safety, science QA, coding, and OR
- active multi-turn environment tasks should also be dataset-backed item fan-outs rather than a separate benchmark-adapter lane

`benchmark_tier` is retained as compatibility metadata, but the active catalog is no longer split into separate comparable and experiment lanes. If a task should stay out of the active benchmark set, mark it explicitly with `included_in_main_comparison: false`.

## 6. Dependency Burden

Some benchmark families still carry heavier setup requirements:

- `co-bench`
  uses dataset fan-out plus the checked-in official evaluation framework and local dataset assets

## 7. Checklist For New Tasks

When adding a benchmark task, write down:

1. what the benchmark is trying to measure
2. which `task_mode` it belongs to
3. what the verifier gate is
4. what the single headline metric is
5. how that metric is encoded in `objective_spec`
6. whether any tie-break is truly needed
7. whether the task should stay in the active benchmark set or opt out with `included_in_main_comparison: false`

If these seven items are not explicit, the benchmark contract is still underspecified.
