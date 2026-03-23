# Frontend Modules

This file explains the current `ui/src/*` layout.

## Top-Level Rule

The frontend is intentionally thin.

It does not own benchmark logic or scoring logic.
It only:

- fetches backend payloads
- starts jobs
- polls job state
- renders task, generation, dataset, and memory views

## `ui/src/main.tsx`

Responsibility:

- bootstraps React
- mounts `App`

It is the entry point and nothing more.

## `ui/src/api.ts`

Responsibility:

- wraps all fetch calls
- normalizes request query parameters
- turns non-2xx responses into thrown errors with attached payloads

Main calls:

- `loadRuntime()`
- `loadTasks()`
- `loadLatestRun()`
- `startJob()`
- `loadJob()`

If an API contract changes, this is the first frontend file to inspect.

## `ui/src/types.ts`

Responsibility:

- defines the TypeScript shapes for backend payloads
- keeps frontend rendering code aligned with JSON produced by the backend

This file is the contract layer between backend and frontend.

## `ui/src/errorPayload.ts`

Responsibility:

- normalize backend error payloads into one UI-friendly shape
- make mixed error sources printable

This prevents the UI from having to guess whether an error came from:

- the backend payload
- a thrown `Error`
- some unknown object

## `ui/src/App.tsx`

Responsibility:

- almost all screen state and rendering
- task selection
- runtime and payload loading
- job polling
- dataset summary rendering
- generation and candidate detail rendering
- error presentation

Mental model:

`App.tsx` is a read-mostly dashboard container, not a component library.

It keeps a few major state groups:

- runtime and task catalog
- latest completed payload
- currently selected task/run
- active background job and live event stream
- modal state such as dataset intro

## `ui/src/styles.css`

Responsibility:

- global layout
- panel styling
- metric cards
- typography and theme variables

This repo does not currently split styling into many component-local CSS files; the app uses one main stylesheet.

## Frontend Logic In One Pass

1. app bootstraps through `main.tsx`
2. `App.tsx` loads runtime, tasks, and latest payload
3. user starts a run
4. `api.ts` sends a POST request
5. `App.tsx` polls `/api/job`
6. when the job completes, `App.tsx` swaps from live job state back to completed payload rendering

## What The Frontend Does Not Do

- it does not compute `primary_score` or `tie_break_score`
- it does not evaluate candidates
- it does not decide benchmark winners
- it does not mutate memory directly

All of that lives in the backend payload it renders.
