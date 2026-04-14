# LiveCodeBench

Aggregate LiveCodeBench coding task spanning the official `v1` through `v6` shards.

- `prepare.py` materializes one local manifest plus cached per-problem JSON under `data/problems/`
- `verifier.py` reuses the official-style local execution semantics from `app.bench.livecodebench_official_support`
- `runtime_split_selector` exposes single-select release filtering in the UI without changing the wire format
