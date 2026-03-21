# autoresearch-with-experience-replay

A local-first autoresearch flywheel that borrows the fixed-budget experiment loop from `karpathy/autoresearch`, the Apple Silicon execution posture from `autoresearch-macos`, and the candidate-selection pressure of `openevolve`.

The demo does not try to train a real model yet. Instead it simulates the part we need to prove first:

`task -> retrieved experience -> competing experiment proposals -> deterministic evaluation -> winner selection -> selective write-back`

That gives you a runnable macOS prototype today and a clean handoff path to an H200-backed training lane later.

## What this repo shows

- A macOS-first local research loop with deterministic scoring.
- Experience replay as the central organizing object.
- Proposal competition across multiple agent lanes.
- A handoff story from local MPS experimentation to H200 cluster execution.
- A read-only frontend demo that visualizes the full loop.

## Run the demo

Generate the demo artifact:

```bash
python3 -m app.demo_run
```

Start the local server:

```bash
python3 -m app.server
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Repo map

- `app/engine.py`: planner, proposal lanes, selection logic, and write-back policy.
- `app/evaluator.py`: deterministic evaluator for local and H200-targeted proposals.
- `app/memory_store.py`: file-backed retrieval and memory append logic.
- `app/demo_run.py`: end-to-end demo artifact generation.
- `app/server.py`: tiny local backend that serves the UI and latest run JSON.
- `data/tasks.json`: the local-to-H200 scenario definitions.
- `data/experiences.json`: seed experience memory.
- `docs/plan.md`: implementation plan and scope lock.
- `docs/framework.md`: core mechanism and system view.
- `docs/demo.md`: narration for the frontend demo.
- `paper/outline.md`: short paper structure for the concept.

## Design choices

- **Validated experience over chat transcripts.** The reusable unit is a scored experience, not a conversation.
- **Deterministic evaluation first.** Keep/discard decisions stay stable and inspectable.
- **Local-first constraints.** The first loop must make sense on a Mac without GPU assumptions.
- **H200 as a later lane.** Cluster scale-up is modeled as a handoff bundle, not a dependency of the local demo.

## Reference repos cloned locally

These were cloned into `external/` for planning and comparison and are intentionally ignored by git:

- `external/autoresearch`
- `external/autoresearch-macos`
- `external/openevolve`
