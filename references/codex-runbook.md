# Codex runbook

## Goal

Use Codex on macOS to scaffold and refine a local demo quickly without relying on GPU execution.

## Working mode

Ask Codex to operate as a repo builder, not as an ideation partner. Give it a target directory and require concrete file creation and local verification.

## Suggested initial instruction

"Build a local multi-agent prototype in this repo for an interview task. Organize validated experience units, not agent chat. Use Python 3.11 and a minimal static dashboard. Implement task -> candidate generation -> deterministic evaluation -> winner selection -> selective memory write-back. Keep everything runnable on macOS without GPU. Start by creating the repo structure, then implement a demo run and a local dashboard."

## Execution order

1. Run `python scripts/bootstrap_local_demo.py --output <target-repo>` from the skill folder.
2. Open the generated repo in Codex.
3. Ask Codex to refine the task domain while preserving the evaluation loop.
4. Run the backend demo.
5. Serve the UI locally.
6. Patch only what breaks.
7. Finalize docs and paper outline.

## Guardrails to give Codex

- Do not replace the deterministic evaluator with subjective LLM judging.
- Do not add GPU or cloud dependencies.
- Do not turn the project into a general chat assistant.
- Prefer JSON artifacts and explicit logs.
- Keep the first task domain narrow and evaluable.

## Minimal acceptance checks

- `python app/demo_run.py` succeeds.
- `runs/latest_run.json` is created.
- `ui/index.html` renders the run artifact.
- The docs clearly state the organized object, mechanism, and rationale.
