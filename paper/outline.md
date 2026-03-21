# A local flywheel for autoresearch with experience replay

## Abstract

We present a local-first autoresearch prototype that organizes validated experience rather than agent dialogue. The system retrieves prior experience, generates competing experiment proposals, scores them with a deterministic evaluator, and writes back only winners whose score clears a baseline by `epsilon`. The prototype is designed for macOS-first execution but exposes an explicit handoff path into an H200 training lane. The contribution is not a chat wrapper but a runnable mechanism for proposal selection, memory consolidation, and future scale-up.

## 1. Problem setup

- Karpathy-style autoresearch assumes repeated experiment selection.
- Local Mac execution imposes device and memory constraints.
- H200 scale-up should reuse local lessons instead of restarting search from scratch.

## 2. Design objective

`max E[J(c; x)]`

subject to:

- deterministic evaluation
- bounded memory growth
- local execution for the first loop
- cluster handoff compatibility later

## 3. Method

### 3.1 Task representation

Represent each task with:

- task signature
- target device
- budget constraints
- required capability checks
- a baseline proposal

### 3.2 Proposal competition

`c_i ~ pi_i(. | x, M_k(x))`

with lanes such as:

- local-optimizer
- replay-synthesizer
- evolution-scout
- scale-bridge

### 3.3 Deterministic evaluation

`J(c; x)` rewards success, required-check coverage, replay alignment, reproducibility, and scale readiness while penalizing cost and complexity.

### 3.4 Selective consolidation

`if J(c*; x) - J(base; x) > epsilon, then M <- M union {e}`

### 3.5 Limits

- proposal simulation rather than live training
- JSON-backed memory store
- fixed candidate lanes

## 4. Demo implementation

- Python stdlib backend
- static frontend dashboard
- three tasks covering local bootstrap, replay benefit, and H200 handoff

## 5. Qualitative results

- first local task discovers an MPS-safe replay-guided strategy
- second local task retrieves it and reduces mutation churn
- H200 handoff packages the validated local pattern into a cluster-ready bundle

## 6. Ablations

- no memory retrieval
- no selective write-back
- single candidate lane
- no device-specific compatibility checks

## 7. Failure cases

- memory pollution from weak write-back gates
- evaluator misspecification
- overfitting to the proposal simulator
- poor handoff assumptions between local and cluster lanes

## 8. Conclusion

The central contribution is a lightweight flywheel that turns local, validated research decisions into reusable experience for later large-scale training.
