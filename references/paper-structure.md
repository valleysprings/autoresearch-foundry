# Paper structure

## Title

A local flywheel for multi-agent experience consolidation

## Abstract

State the problem, the mechanism, and the demo claim in 4 to 6 sentences. Emphasize that the prototype organizes validated experience units through deterministic evaluation and selective write-back.

## 1. Problem setup

Describe the interview brief. Clarify that the goal is not a chat product or a tool wrapper, but a runnable mechanism for organizing agents.

## 2. Design objective

Define the design objective as:

`max expected task utility under local cost and reliability constraints`

A simple formulation is:

`max E[J(c; x)] subject to local execution, deterministic evaluation, and bounded memory growth`

## 3. Method

### 3.1 Task representation
Define the task signature.

### 3.2 Agent competition
Use:

`c_i ~ pi_i(. | x, M_k(x))`

### 3.3 Deterministic evaluation
Use:

`J(c; x) = alpha * success + beta * test_pass - gamma * cost - lambda * steps`

### 3.4 Selective experience consolidation
Use:

`if J(c*; x) - J(base; x) > epsilon, then M <- M union {e}`

### 3.5 Complexity and limits
Mention the current prototype uses file-backed JSON memory and fixed agents.

## 4. Demo implementation

Describe the local macOS stack, repo layout, and one task domain. Mention that no GPU is required.

## 5. Qualitative results

Report a first task and a similar follow-up task. Show that retrieval changes the winning strategy or reduces steps/cost.

## 6. Ablations

Keep these simple:

- without memory retrieval
- without write-back threshold
- single-agent baseline

## 7. Failure cases

Discuss memory pollution, evaluator misspecification, and overfitting to the demo domain.

## 8. Conclusion

State that the main contribution is a lightweight, local-first platform loop that turns validated trajectories into reusable experience.
