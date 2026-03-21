# Framework note

## Core claim

The platform organizes **validated research experience**, not free-form agent conversation.

Each task is a small research decision problem:

- which experiment proposal should run next
- whether the local device can support it
- whether the proposal should become reusable experience

## Objects

Let the incoming task be `x`.

Let retrieved memory be `M_k(x)`.

Let candidate proposals be `c_i`.

Each candidate is a structured experiment plan containing:

- target device
- attention backend
- replay usage
- logging policy
- write-back policy
- expected utility and cost estimates

Selection is:

`c* = argmax_i J(c_i; x)`

The evaluator is deterministic and reward-shapes for:

- success under budget
- required capability checks
- replay alignment
- reproducibility
- scale readiness

with penalties for cost, complexity, excessive steps, and unsupported kernels.

Write-back remains selective:

`if J(c*; x) - J(base; x) > epsilon, then M <- M union {e}`

where:

`e = (task_signature, failure_pattern, successful_strategy, tool_trace_summary, delta_J, reusable_rules)`

## Why this design fits the brief

- It preserves the **autoresearch** intuition of repeated experiment proposals and selection.
- It works on **macOS first**, so the prototype can run locally without waiting for cluster infrastructure.
- It borrows **evolution pressure** from OpenEvolve by keeping multiple candidate lanes instead of a single linear proposal stream.
- It keeps the organized object small and reusable: the memory contains rules that future tasks can retrieve and compose.

## System modules

1. Task planner: restates the task and required checks.
2. Memory retriever: scores prior experience by signature and device.
3. Proposal lanes: `local-optimizer`, `replay-synthesizer`, `evolution-scout`, and `scale-bridge`.
4. Deterministic evaluator: computes `J` from explicit metrics.
5. Consolidator: writes back experience only when `delta_J` is real.
6. Local backend/UI: emits and renders the run artifact for the demo.
