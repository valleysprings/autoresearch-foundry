# planbench-t2

Comparable reasoning benchmark backed by the public `tasksource/planbench` dataset.

This local task tracks the official direct `task_2_plan_optimality` setting:

- input: the original natural-language optimal-planning prompt
- output: a directly generated plan in a format accepted by upstream `text_to_plan`
- scoring: semantic validation with `VAL`, then optimality by exact plan-step count against the official reference plan
- unified repo contract: `task_mode=answer`, `interaction_mode=single_turn`

Current local normalization target:

- config: `task_2_plan_optimality`
- split: `train`

Prepare locally with:

```bash
python3 prepare.py
```

Shared official evaluation assets live under:

- `app/bench/planbench_support.py`
- `benchmark/reasoning_verified/planbench-shared/official/plan-bench`
- `benchmark/reasoning_verified/planbench-shared/official/VAL/build/bin/{validate,Validate}`
