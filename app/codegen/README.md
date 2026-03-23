# Codegen Engine

This directory contains the active benchmark-driven codegen loop.

## Design

### Proposal runtime

- loads strict runtime config from repo-root `.env` or shell env
- sends one-model OpenAI-compatible chat requests
- expects strict JSON candidates with `file_body`

### Trainer

- retrieves prompt-ready memory fragments
- asks the model for candidate file rewrites
- materializes and verifies each candidate
- writes back reusable success or failure experiences

### Verifier

- materializes a single editable file from `file_body`
- imports the declared entry symbol
- runs deterministic correctness checks first
- benchmarks only verified candidates
- computes task-facing `objective` plus layered selection metrics

### Reporting and handoff

- emits payload JSON, traces, `llm_trace.jsonl`, markdown memory, and report artifacts
- exposes those artifacts to the UI and downstream handoff consumers

## Objective and Selection

The runner now uses a layered selection contract instead of one global `J`:

- `gate_passed`
  verifier-side eligibility check; candidates that fail the gate cannot win
- `primary_score`
  task-native objective score, normalized so that higher is always better
- `tie_break_score`
  track-specific weak preference score used only when primary scores are effectively tied
- `archive_features`
  optional diversity metadata for downstream sampling or archive logic; not part of winner selection

Selection rules:

- correctness or task-validity is gated first
- failing or erroring candidates do not enter the winner lane
- generations mutate a selected frontier parent, not only the global incumbent
- a generation is accepted when it beats its selected parent on `primary_score` by `epsilon`
- the global best updates only when a frontier winner also beats the current best on `primary_score`
- passing but stagnant candidates do not get written back as failure memory
