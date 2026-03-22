---
name: agent-flywheel-prototype
description: build a local macos multi-agent prototype for interview tasks about agent coordination, evaluation, selection, and experience consolidation. use when chatgpt or codex needs to scaffold a no-gpu demo, turn a task brief into a runnable local repo, design a flywheel around deterministic evaluation plus memory consolidation, or produce the companion framework note, demo wireframe, and short paper outline.
---

# Agent flywheel prototype

Build a small local-first prototype for a multi-agent platform where the core novelty is a runnable mechanism, not a chat shell.

Prefer a no-GPU stack that runs on macOS with Python 3.11 and simple local web assets. Treat deterministic evaluation and validated experience write-back as the center of the design.

## Default output

Produce four artifacts inside the working repo:

1. A runnable local prototype.
2. A short framework note explaining what is being organized.
3. A demo script and local UI/wireframe.
4. A paper outline with claims, formulas, and ablations.

## Workflow

Follow this sequence.

1. Read the task brief and restate the success criteria in one paragraph.
2. Lock the scope to one core mechanism. Default to: candidate generation -> deterministic evaluation -> selection -> experience write-back.
3. Run `scripts/bootstrap_local_demo.py --output <repo-path>` to scaffold the starter repo.
4. Edit the generated files to match the user's task domain, but keep the local-first and no-GPU constraints unless the user explicitly asks otherwise.
5. Run the generated demo locally and verify that one full flow works end-to-end.
6. Fill or refine the framework note, demo note, and paper outline using the references bundled with this skill.

## Mechanism to implement

Organize **experience units**, not agent chat turns.

Use the following default objects:

- Task `x`
- Retrieved memory set `M_k(x)`
- Candidate solutions `c_i`
- Evaluator score `J(c_i; x)`
- Consolidated experience `e`

Use these formulas unless the user provides a better task-specific one.

Candidate generation:

`c_i ~ pi_i(. | x, M_k(x))`

Selection:

`c* = argmax_i J(c_i; x)`

Evaluator:

`J(c; x) = alpha * success + beta * test_pass - gamma * cost - lambda * steps`

Write-back rule:

`if J(c*; x) - J(base; x) > epsilon, then M <- M union {e}`

Experience unit structure:

`e = (task_signature, failure_pattern, successful_strategy, tool_trace_summary, delta_J)`

Keep the initial prototype simple. Do not implement model training, policy gradient updates, or heavyweight long-term memory systems.

## Allowed simplifications

Prefer these simplifications:

- Use two or three specialist agents at most: planner, solver, critic, consolidator.
- Use deterministic or mocked agent outputs when needed for the demo.
- Use unit tests, regex checks, JSON schema checks, or exact-match validators as the evaluator.
- Use JSON files for memory storage and run logs.
- Use a static HTML dashboard or a tiny local server for visualization.

## Do not build

Do not build any of the following unless the user explicitly requests them:

- multi-agent social feeds or free-form chat rooms
- generic tool wrappers with no mechanism
- GPU training loops
- external SaaS dependencies for the critical path
- clones of existing products

## Local stack

Default stack:

- Python 3.11
- Standard library first; minimal extra packages only if clearly justified
- Static HTML/CSS/JS or a tiny local server
- JSON for tasks, memory, and run artifacts

If the user needs a local demo fast, keep the UI read-only and let the backend generate `runs/latest_run.json` for the page to render.

## Files to create or maintain

The generated repo should usually contain:

- `app/codegen/`
- `app/memory_store.py`
- `app/demo_run.py`
- `ui/index.html`
- `ui/app.js`
- `data/tasks.json`
- `data/experiences.json`
- `paper/outline.md`
- `docs/framework.md`
- `docs/demo.md`

Use the references in this skill for the intended content of `docs/framework.md`, `docs/demo.md`, and `paper/outline.md`.

## Quality bar

Before finishing, verify all of the following:

- One complete flow runs locally on macOS.
- The mechanism is visible from logs or UI.
- The evaluator is deterministic.
- Memory is written only after a measurable improvement.
- The paper outline states what is organized, what the mechanism is, and why it should work.

## Resources

- Framework design: `references/framework.md`
- Demo wireframe and narration: `references/demo-wireframe.md`
- Paper structure: `references/paper-structure.md`
- Codex execution plan: `references/codex-runbook.md`
- Starter repo generator: `scripts/bootstrap_local_demo.py`
