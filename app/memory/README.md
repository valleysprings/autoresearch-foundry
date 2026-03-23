# Memory

This directory contains the strategy-memory layer used by the benchmark loop.

## Design

Memory is prompt-ready, not a generic event log.

Each experience stores fields such as:

- `failure_pattern`
- `strategy_hypothesis`
- `successful_strategy`
- `prompt_fragment`
- `tool_trace_summary`
- `delta_primary_score`
- `proposal_model`
- `candidate_summary`
- `experience_outcome`
- `verifier_status`

Important properties:

- memory persists across runs and is not reset to seeds each time
- retrieval prefers success experiences and caps failure fragments
- failure memory is reserved for informative verifier failures or execution errors
- duplicate memory fragments are suppressed before write-back
- markdown output is an auditable ledger, while the UI also surfaces run-local fragments directly
