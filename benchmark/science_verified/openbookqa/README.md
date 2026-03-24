# OpenBookQA

Local dataset-task entry for the OpenBookQA validation split using the official `additional` config.

- benchmark task unit: the full dataset
- evolution unit: one question per independent question-run
- `prepare.py` lazily materializes only the requested local prefix
- the local context includes the official supporting `fact1` field

This task wrapper is versioned. Materialized dataset files under `data/` stay local and should not be uploaded to GitHub.

Prepare locally with:

```bash
python3 prepare.py --items 50
```
