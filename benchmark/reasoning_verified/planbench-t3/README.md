# planbench-t3

Comparable reasoning benchmark backed by the public `tasksource/planbench` dataset.

This local task tracks the official `task_3_plan_verification` split while normalizing the output contract to a binary verdict:

- input: the original natural-language plan-verification prompt
- output: `yes` if the plan is valid, otherwise `no`
- scoring: direct verdict match against prepare-time normalized labels
- unified repo contract: `task_mode=answer`, `interaction_mode=single_turn`

Current local normalization target:

- config: `task_3_plan_verification`
- split: `train`

Prepare locally with:

```bash
python3 prepare.py
```
