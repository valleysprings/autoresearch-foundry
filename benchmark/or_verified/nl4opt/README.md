# nl4opt

OR benchmark task for `CardinalOperations/NL4OPT` using the custom external runtime backend.

It prompts the configured model for `coptpy` code, executes the generated program locally, and scores exact optimal-value matches against the benchmark answers.

Here `runtime_backend=external` means this task runs through a custom benchmark harness instead of the generic dataset fan-out path. It does not mean the benchmark is remote-only.

`python benchmark/prepare_datasets.py --task-id nl4opt` is the expected prerequisite after clone.

You can also run `python prepare.py` directly to materialize the benchmark question manifest into `data/questions.json`.
