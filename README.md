# autoresearch-with-experience-replay

A local-first autoresearch flywheel that borrows the fixed-budget experiment loop from `karpathy/autoresearch`, the Apple Silicon execution posture from `autoresearch-macos`, and the candidate-selection pressure of `openevolve`.

The project now runs concrete local tasks instead of only rendering a blueprint:

`baseline Python program -> candidate mutations -> fixed tests -> benchmark -> winner selection -> selective memory write-back`

That gives you a runnable macOS prototype today. Once this loop is stable, the same scaffold can be retargeted to prompt optimization or a tiny NAS search.

## What this repo shows

- A macOS-first local runner that executes real Python optimization tasks.
- Experience replay as the central organizing object.
- Proposal competition across multiple agent lanes.
- Deterministic test-first evaluation before benchmark-based selection.
- A frontend that can trigger tasks and inspect winners on localhost.

## Run the task runner

List available tasks:

```bash
python3 -m app.demo_run --list-tasks
```

Run `task1` directly:

```bash
python3 -m app.demo_run --task contains-duplicates
```

Run the full sequence:

```bash
python3 -m app.demo_run
```

Start the local server:

```bash
python3 -m app.server
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

Direct API calls:

```bash
curl http://127.0.0.1:8000/api/tasks
curl http://127.0.0.1:8000/api/latest-run?task_id=contains-duplicates
curl -X POST http://127.0.0.1:8000/api/run-task?task_id=contains-duplicates
curl -X POST http://127.0.0.1:8000/api/run-sequence
```

## Repo map

- `app/engine.py`: runner orchestration, candidate selection, and write-back policy.
- `app/evaluator.py`: actual code execution, correctness tests, and benchmark scoring.
- `app/memory_store.py`: file-backed retrieval and memory append logic.
- `app/demo_run.py`: end-to-end demo artifact generation.
- `app/server.py`: tiny local backend that serves the UI and latest run JSON.
- `app/task_catalog.py`: runnable task definitions and candidate mutations.
- `data/tasks.json`: the local task catalog.
- `data/experiences.json`: seed experience memory.
- `examples/evolve/*/initial_program.py`: baseline functions for real tasks.
- `docs/plan.md`: implementation plan and scope lock.
- `docs/framework.md`: core mechanism and system view.
- `docs/demo.md`: narration for the frontend demo.
- `paper/outline.md`: short paper structure for the concept.

## Design choices

- **Validated experience over chat transcripts.** The reusable unit is a scored experience, not a conversation.
- **Deterministic evaluation first.** Keep/discard decisions stay stable and inspectable.
- **Local-first constraints.** The first loop must make sense on a Mac without GPU assumptions.
- **Real code first.** The current runner executes concrete Python tasks before expanding to prompt optimization or tiny NAS.

## Reference repos cloned locally

These were cloned into `external/` for planning and comparison and are intentionally ignored by git:

- `external/autoresearch`
- `external/autoresearch-macos`
- `external/openevolve`
