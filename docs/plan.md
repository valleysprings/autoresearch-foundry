# Implementation plan

## Success criteria

Build a local-first autoresearch demo that feels structurally close to `karpathy/autoresearch`, runs on macOS without GPU requirements, exposes a deterministic selection loop, and shows how the local loop can later hand off to an H200 training lane.

## Reference mapping

- `karpathy/autoresearch`: fixed-budget experiment loop, program-centric mutation, results logging, and explicit keep/discard decisions.
- `miolini/autoresearch-macos`: MPS-safe execution path, smaller local budgets, and Apple Silicon compatibility.
- `openevolve`: diversity across candidate lanes, mutation pressure, and evaluator-driven selection.

## Chosen mechanism

The prototype organizes **validated experience units**, not free-form chat. Each task:

1. Retrieves prior experience by task signature and target device.
2. Generates competing research proposals from specialist lanes.
3. Scores proposals with a deterministic evaluator.
4. Selects the highest-scoring candidate against a baseline.
5. Writes back a new experience only if `delta_J > epsilon`.

## Scope lock

The first implementation deliberately avoids full training, cluster orchestration, and live LLM calls. Instead it simulates the research loop over structured experiment proposals so the flywheel is visible, deterministic, and demoable on any Mac.

## Phase plan

### Phase 1

Implement the local macOS loop:

- file-backed memory retrieval
- deterministic evaluator
- proposal competition across `local-optimizer`, `replay-synthesizer`, and `evolution-scout`

### Phase 2

Show experience replay:

- first local win creates an MPS-specific memory
- second local task retrieves it
- replay-guided proposal wins with fewer steps and better reliability

### Phase 3

Add the H200 bridge:

- package winning local patterns as handoff bundles
- keep evaluator and logging schema stable across devices
- preserve selective write-back for cluster experiments

## Demo acceptance

- `python3 -m app.demo_run` produces `runs/latest_run.json`
- `python3 -m app.server` serves the dashboard locally
- the UI shows memory retrieval, candidate scores, winner selection, and write-back
- the roadmap clearly separates local Mac execution from later H200 expansion
