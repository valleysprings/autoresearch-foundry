# ARC-Challenge

Local dataset-task entry for the AI2 ARC Challenge validation split.

- benchmark task unit: the full dataset
- evolution unit: one question per independent question-run
- `prepare.py` lazily materializes only the requested local prefix
- the wrapper lives under the reasoning lane, while solver family stays aligned with science multiple-choice QA

This task wrapper is versioned. Materialized dataset files under `data/` stay local and should not be uploaded to GitHub.

Prepare locally with:

```bash
python3 prepare.py --items 50
```
