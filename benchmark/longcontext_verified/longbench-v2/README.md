# LongBench v2

Local dataset-task entry for LongBench v2.

- benchmark task unit: the full dataset
- evolution unit: one question per independent question-run
- `prepare.py` materializes the local question manifest and per-item context cache under `data/`
- the full long context for each item is stored in a per-item local file under `data/items/`
- prompt construction uses a bounded context preview, while verifier execution still passes the full cached context to `solve(question)`

This task wrapper is versioned. Materialized dataset files under `data/` stay local and should not be uploaded to GitHub.

Prepare locally with:

```bash
python3 ../../prepare_datasets.py --task-id longbench-v2
```
