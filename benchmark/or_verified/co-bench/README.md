# co-bench

CO-Bench now runs through the normal dataset fan-out path. Each manifest item is one official CO-Bench problem, and each item run still uses the checked-in official evaluator.

`python benchmark/prepare_datasets.py --task-id co-bench` is the expected prerequisite after clone.

You can also run `python prepare.py` directly to download the dataset snapshot into `data/` and write `data/questions.json`.

Evaluation contract:

- the wrapper prompt generates Python that must define `def solve(**kwargs)`
- the official evaluator executes `solve(**instance)` for each benchmark instance
- the returned dict is scored by the task's `eval_func(**instance, **solution)`
- this repo uses the official `test_score` and reports the mean normalized test score across fan-out item runs
- the checked-in `evaluation/*` directory under this task is the only upstream framework code used by this repo
- no upstream `agents/*` code is imported by our verifier path

Operational notes:

- `evaluation/` is checked into this repo directly under the task, rather than hidden behind `external/` or another nested checkout
- `prepare.py` writes a manifest where each item is one official problem; use `--max-items` or item selection at runtime to limit fan-out
- Docker is not required for the current official evaluator path; the upstream README explicitly says Docker support is still "coming soon"
- the upstream controller uses `TSP` and `MIS` aliases, but the dataset folders use the full names; this integration normalizes them automatically
- `editable.py` still owns prompt construction and timeout settings, but dataset selection now comes from the manifest rather than `RUN_CONFIG.problem_names`
- the default search budget is intentionally conservative: `generation_budget=1`, `candidate_budget=1`, `branching_factor=2`, `item_workers=10`
