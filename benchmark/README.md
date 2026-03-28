# Benchmark Registry

This repo treats `benchmark/` as the source of truth for local research datasets and their task wrappers.

Two tiers are supported:

- `benchmark_tier=experiment`
  reserved for local or test-only fixtures outside the active benchmark registry
- `benchmark_tier=comparable`
  real dataset tasks used for research runs and dataset fan-out execution

The active registry contains both `comparable` and `experiment` tasks, but all enabled tasks currently use the dataset runtime.

The active tasks live in `benchmark/registry.json`. Each task directory includes:

- `task.json`
- `editable.py`
- `verifier.py`
- `prepare.py` for local dataset normalization when the benchmark ships as raw local assets
- `data/`
- `README.md`

Each task is expected to declare a clear contract:

- `runtime_backend`
  use `dataset` for active registry tasks. `external` remains a legacy/internal runtime value and is not enabled right now
- `task_mode`
  what the editable file represents: `answer`, `artifact`, or `agent`
- `optimization_scope`
  what is allowed to change: `prompt`, `wrapper`, or `implementation`
- `answer_metric`
  the task-native headline metric name
- `objective_spec`
  display name, direction, unit, and formula for that metric

For `task_mode=answer`, also make the verifier style explicit in task-local docs and prompt context:

- `exact_match`
  the returned answer is compared directly against a reference answer or label set
- `adapter` / `semantic`
  the returned answer is first parsed, normalized, or adapted by the verifier and then judged by a semantic checker or task-native evaluator

Read [references/evaluation-contract.md](../references/evaluation-contract.md) for the metric and selection semantics behind these fields.

Only `benchmark/registry.json` is intended to sync by default. Dataset directories under `benchmark/` stay local.

## Prepare Local Datasets

Preparing local datasets is a prerequisite after clone.

Use the shared entrypoint to materialize every benchmark-local dataset setup that exposes a `prepare.py`:

```bash
python benchmark/prepare_datasets.py
```

Useful variants:

```bash
python benchmark/prepare_datasets.py --list
python benchmark/prepare_datasets.py --task-id co-bench
```

The shared script just dispatches each task's own `prepare.py`, so task-specific customization still lives beside the task itself. When a task is missing local assets, runtime errors should point back to this command.

## Add A New Benchmark Dataset

1. add a task directory under the appropriate research track such as `math_verified/`, `science_verified/`, `reasoning_verified/`, or `longcontext_verified/`
2. register the dataset task in `benchmark/registry.json`
3. provide:
   - `task.json`
   - `editable.py`
   - `verifier.py`
   - `prepare.py` when raw local data must be normalized or lazily materialized into the shared manifest shape
   - `data/questions.json` or another local question manifest
   - `README.md`

Required `task.json` fields for dataset tasks:

- `benchmark_tier`
- `track`
- `dataset_id`
- `dataset_size`
- `local_dataset_only`
- `item_manifest`
- `runtime_backend`
- `task_mode`
- `optimization_scope`
- `answer_metric`
- `objective_spec`
- `editable_file`
- `entry_symbol`

Recommended conventions:

- use the real dataset name for `id`, directory name, and `dataset_id`
- put subset details like `validation`, `test`, `numeric_verified`, or `local_eval` in `split`
- keep one dataset-native headline metric; do not make `tie_break_score` or code-shape preferences the headline claim
- for answer tasks, state whether the verifier is `exact_match` or `adapter/semantic`; this belongs in the task README and prompt context even if it is not a dedicated schema field
- normalize raw dataset rows into the shared question schema:
  - `item_id`
  - `prompt`
  - optional `context`
  - optional `choices`
  - `expected_answer`
  - `metadata`
- for very large per-item payloads, a manifest row can also declare `item_file`, pointing to a local JSON object that is merged into that row at load time
- math datasets should declare `metadata.answer_format` as `symbolic`, `numeric`, or `choice`
- choice-form math datasets should also declare `metadata.correct_choice_index`
- treat the dataset as the benchmark task and each question as an independent question-run
