# Framework design

## Core claim

The platform does not organize conversation. It organizes **validated experience units**.

Each run produces candidate solutions from a small set of specialized agents. A deterministic evaluator ranks the candidates. Only a winning candidate with measurable gain is consolidated into memory.

## Objects

Let the task be `x`. Let the retrieved experience set be `M_k(x)`. Let candidate solutions be `c_i`.

Candidate generation:

`c_i ~ pi_i(. | x, M_k(x))`

Selection:

`c* = argmax_i J(c_i; x)`

Evaluator:

`J(c; x) = alpha * success + beta * test_pass - gamma * cost - lambda * steps`

Write-back:

`if J(c*; x) - J(base; x) > epsilon, then M <- M union {e}`

Experience unit:

`e = (task_signature, failure_pattern, successful_strategy, tool_trace_summary, delta_J)`

## System modules

1. Task router: parse task and compute task signature.
2. Memory retriever: fetch top-k prior experience units with similar signatures.
3. Planner: propose a short plan and agent assignments.
4. Solver agents: produce competing candidate solutions.
5. Critic/evaluator: run deterministic checks and compute `J`.
6. Consolidator: summarize the winning trajectory and decide whether to write it back.
7. Run logger/UI: expose the trace, scores, and memory updates.

## Why this fits the interview brief

This design answers three questions directly:

- What is being organized: tasks, candidate results, and reusable experience units.
- Why the mechanism works: deterministic selection plus selective write-back prevents noisy memory growth.
- Why agents participate: each agent competes to produce the winning candidate and increases its reputation or utility when selected.

## Minimal local implementation

Use JSON files for tasks, memory, and run logs. Keep the first demo in a single domain such as code fix, text transformation, or structured extraction. Avoid open-ended creative tasks because evaluation becomes subjective.
