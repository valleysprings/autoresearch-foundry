# autoresearch-with-experience-replay

A strict LLM-required outer loop for direct Python code generation.

The active path is:

`strategy memory retrieval -> llm candidate generation -> deterministic materialize/test/benchmark -> winner selection -> strategy-memory write-back`

There is no fallback mode:

- no deterministic local fallback
- no offline mode
- no secondary model rotation
- any config or LLM failure aborts the run

## Configuration

Local configuration is loaded from repo-root `.env`, with shell environment variables taking precedence.

Required keys:

- `AUTORESEARCH_API_KEY`
- `AUTORESEARCH_API_BASE`
- `AUTORESEARCH_PRIMARY_MODEL`
- `AUTORESEARCH_TEMPERATURE`
- `AUTORESEARCH_MAX_TOKENS`
- `AUTORESEARCH_TIMEOUT_S`

See [.env.example](./.env.example).

## Run the codegen runner

List tasks:

```bash
python3 -m app.entries.discrete_demo --list-tasks
```

Run one task:

```bash
python3 -m app.entries.discrete_demo --task contains-duplicates
```

Run the full sequence:

```bash
python3 -m app.entries.discrete_demo
```

Successful runs write artifacts under `runs/`, including:

- payload JSON
- `memory.md`
- `trace.jsonl`
- `llm_trace.jsonl`
- materialized candidate workspaces

## Run the local server

```bash
python3 -m app.entries.server
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

API endpoints:

```bash
curl http://127.0.0.1:8000/api/tasks
curl http://127.0.0.1:8000/api/latest-run
curl http://127.0.0.1:8000/api/latest-run?task_id=contains-duplicates
curl -X POST http://127.0.0.1:8000/api/run-task?task_id=missing-number
curl -X POST http://127.0.0.1:8000/api/run-sequence
```

If config or the model call fails, the API returns a terminal JSON error and the job stops.

## Repo map

- `app/codegen/`: embedded task specs, strict config loading, LLM runtime, verifier, trainer, and handoff helpers
- `app/entries/discrete_demo.py`: compatibility entrypoint now backed by the codegen runner
- `app/entries/server.py`: local HTTP server and job API
- `app/memory/`: JSON + markdown strategy memory store
- `tests/`: config, runtime, verifier, integration, server, and UI smoke coverage

## Design choices

- **Direct code generation only.** The model returns candidate function bodies, not operator plans.
- **Deterministic verification outside the model.** Tests and benchmarks run on materialized candidate files.
- **Prompt-ready memory.** Stored experience is shaped for direct injection into future prompts.
- **Strict failure semantics.** Missing config, transport failure, invalid JSON, or malformed candidates all abort the run.
