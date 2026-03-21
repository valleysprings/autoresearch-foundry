# Demo wireframe

## Goal

Show one closed loop in under two minutes:

1. A task arrives.
2. The system retrieves relevant prior experience.
3. Two or three agents generate candidates.
4. The evaluator scores each candidate.
5. One candidate is selected.
6. A new experience unit is written back.
7. A second similar task benefits from retrieval.

## Screen layout

### Left column: task and retrieved memory

- Task card
- Parsed task signature
- Top-k retrieved experience units

### Center column: agent competition

- Planner note
- Candidate A card
- Candidate B card
- Candidate C card
- Each card shows: strategy summary, estimated steps, tool usage, raw output snippet

### Right column: evaluator and consolidation

- Score table with `success`, `test_pass`, `cost`, `steps`, `J`
- Winner banner
- Write-back decision
- Newly created experience unit

## Demo narration

Say this in order:

1. "The platform is organizing reusable experience, not agent chat."
2. "These agents compete over the same task under a shared evaluator."
3. "Only validated improvement is written back into memory."
4. "On the next similar task, the platform starts with a better prior."

## Suggested local implementation

- Generate `runs/latest_run.json` from the backend.
- Let the dashboard render that JSON.
- Keep the UI read-only for the first version.

## Visual cues

- Green badge for selected candidate
- Gray badge for rejected candidates
- Explicit `delta_J` line before memory write-back
- Before/after memory count
