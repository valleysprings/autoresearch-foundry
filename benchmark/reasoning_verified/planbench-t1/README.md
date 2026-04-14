# planbench-t1

Comparable reasoning benchmark backed by the public `tasksource/planbench` dataset.

This local task tracks the official direct `task_1_plan_generation` setting as closely as practical while keeping benchmark-side logic self-contained:

- input: the original natural-language planning prompt
- output: a directly generated plan in a format accepted by upstream `text_to_plan`
- scoring: upstream-compatible Task 1 parsing plus semantic validation with `VAL`
- unified repo contract: `task_mode=answer`, `interaction_mode=single_turn`
- verifier style: adapter/semantic validation, not exact string match or permissive local fallback parsing

Current local normalization target:

- config: `task_1_plan_generation`
- split: `train`

Prepare locally with:

```bash
python3 prepare.py
```

Shared official evaluation assets live under:

- `app/bench/planbench_support.py`
- `benchmark/reasoning_verified/planbench-shared/official/plan-bench`
- `benchmark/reasoning_verified/planbench-shared/official/VAL/build/bin/{validate,Validate}`

This task does not use `external/` clones elsewhere in the repo. The verifier reads only the shared vendored PlanBench assets above.

You can override those paths with:

- `PLANBENCH_OFFICIAL_ROOT`
- `PLANBENCH_VAL_BINARY`
