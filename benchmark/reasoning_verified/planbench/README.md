# planbench

Comparable reasoning benchmark backed by the public `tasksource/planbench` dataset.

This local task tracks the official direct `task_1_plan_generation` setting:

- input: the original natural-language planning prompt
- output: a directly generated plan
- scoring: official-style plan extraction plus semantic validation with `VAL`
- unified repo contract: `runtime_backend=dataset`, `task_mode=answer`, `optimization_scope=wrapper`
- verifier style: adapter/semantic validation, not exact string match

Current local normalization target:

- config: `task_1_plan_generation`
- split: `train`

Prepare locally with:

```bash
python3 prepare.py
```

Official assets are resolved from:

- `external/LLMs-Planning/plan-bench`
- `external/VAL/build/bin/Validate`

You can override those paths with:

- `PLANBENCH_OFFICIAL_ROOT`
- `PLANBENCH_VAL_BINARY`
