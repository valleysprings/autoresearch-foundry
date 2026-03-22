import { useEffect, useMemo, useRef, useState } from "react";

import { loadJob, loadLatestRun, loadRuntime, loadTasks, startJob } from "./api";
import type {
  Branch,
  Candidate,
  ErrorPayload,
  Generation,
  JSpec,
  JobState,
  LiveEvent,
  ObjectiveSpec,
  Payload,
  Run,
  RuntimeInfo,
  TaskSummary,
} from "./types";

type ThemePreference = "system" | "light" | "dark";

type LiveBranchCard = {
  branchId: string;
  branchIndex: number;
  status: "queued" | "running" | "completed";
  parentCandidate: string;
  selectedModel: string | null;
  candidateMessages: string[];
  memoryMessage: string | null;
  acceptedToFrontier: boolean;
  improvedGlobalBest: boolean;
  memoryDelta: number;
};

type LiveGenerationCard = {
  generation: number;
  status: "queued" | "running" | "completed";
  summary: string | null;
  acceptedCount: number;
  memoryDelta: number;
  branches: LiveBranchCard[];
};

type LiveTaskCard = {
  taskId: string;
  title: string;
  description: string;
  model: string;
  branchingFactor: number;
  generationBudget: number;
  currentBest: string | null;
  status: "queued" | "running" | "completed";
  generations: LiveGenerationCard[];
  events: LiveEvent[];
};

function shortPath(path?: string | null): string {
  return path ? path.replace(/^runs\//, "") : "n/a";
}

function artifactUrl(path?: string | null): string | null {
  if (!path) {
    return null;
  }
  return `/api/artifact?path=${encodeURIComponent(path)}`;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function numeric(value: string | number | undefined | null): number {
  if (typeof value === "number") {
    return value;
  }
  const parsed = Number(value ?? 0);
  return Number.isFinite(parsed) ? parsed : 0;
}

function formatValue(value: string | number | undefined | null, unit = ""): string {
  if (typeof value === "number") {
    return `${value}${unit}`;
  }
  if (typeof value === "string" && value.length) {
    return `${value}${unit}`;
  }
  return "n/a";
}

function formatSigned(value: string | number | undefined | null, digits = 3): string {
  const amount = numeric(value);
  return `${amount >= 0 ? "+" : ""}${amount.toFixed(digits)}`;
}

function directionCopy(direction: string): string {
  return direction === "min" ? "Lower is better" : "Higher is better";
}

function emptyRuntime(): RuntimeInfo {
  return {
    mode: "llm-required",
    primary_model: "n/a",
    active_model: "n/a",
    available_models: [],
    api_base: "n/a",
    temperature: "n/a",
    max_tokens: "n/a",
    timeout_s: "n/a",
  };
}

function emptyJSpec(): JSpec {
  return {
    display_name: "Internal selection score J",
    direction: "max",
    summary_template: "J is the always-max internal score used to compare verified candidates.",
    formula: "",
    delta_template: "",
  };
}

function emptyPayload(taskCatalog: TaskSummary[] = []): Payload {
  return {
    summary: {
      generated_at: "n/a",
      run_mode: "llm-required",
      active_model: "n/a",
      num_tasks: 0,
      total_runs: 0,
      experiment_runs: 0,
      total_generations: 0,
      initial_memory_count: 0,
      memory_size_after_run: 0,
      write_backs: 0,
      experiment_write_backs: 0,
      source_repo: "n/a",
      git_commit: "n/a",
      upstream_target: "n/a",
      flywheel: [],
      proposal_engine: emptyRuntime(),
    },
    formulas: {
      J: "",
      objective: "",
      delta_J: "",
      run_delta_J: "",
    },
    j_spec: emptyJSpec(),
    audit: {
      workspace_root: "n/a",
      session_id: "n/a",
    },
    task_catalog: taskCatalog,
    runs: [],
  };
}

function normalizePayload(payload: Payload, fallbackCatalog: TaskSummary[]): Payload {
  const jSpec = payload.j_spec ?? emptyJSpec();
  return {
    ...payload,
    j_spec: jSpec,
    task_catalog: payload.task_catalog?.length ? payload.task_catalog : fallbackCatalog,
    runs: Array.isArray(payload.runs) ? payload.runs.map((run) => ({ ...run, j_spec: run.j_spec ?? jSpec })) : [],
  };
}

function asErrorPayload(error: unknown): ErrorPayload {
  const payload = (error as { payload?: ErrorPayload })?.payload;
  if (payload && typeof payload.error_type === "string") {
    return payload;
  }
  return {
    terminal: true,
    error_type: "runtime_error",
    error: String(error),
    model: null,
  };
}

function metric(label: string, value: string | number) {
  return (
    <div className="metric-card" key={label}>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
    </div>
  );
}

function objectiveLabel(spec: ObjectiveSpec): string {
  return spec.display_name || "Objective";
}

function benchmarkTierLabel(includedInMainComparison: boolean): string {
  return includedInMainComparison ? "main benchmark" : "small experiment";
}

function trackLabel(track: string): string {
  return track.replace(/_/g, " ");
}

function summarizeLiveTasks(
  events: LiveEvent[],
  taskCatalog: TaskSummary[],
  liveJob: JobState | null,
  runs: Run[],
): LiveTaskCard[] {
  const taskMap = new Map<string, LiveTaskCard>();

  function getTask(taskId: string): LiveTaskCard {
    const catalogTask = taskCatalog.find((task) => task.id === taskId);
    const completedRun = runs.find((run) => run.task.id === taskId);
    const existing = taskMap.get(taskId);
    if (existing) {
      return existing;
    }
    const entry: LiveTaskCard = {
      taskId,
      title: catalogTask?.title ?? completedRun?.task.title ?? taskId,
      description: catalogTask?.description ?? completedRun?.task.description ?? "Task description unavailable.",
      model: liveJob?.model ?? completedRun?.active_model ?? "n/a",
      branchingFactor:
        liveJob?.branching_factor ?? catalogTask?.branching_factor ?? completedRun?.task.branching_factor ?? 1,
      generationBudget: catalogTask?.generation_budget ?? completedRun?.task.generation_budget ?? 0,
      currentBest: null,
      status: "queued",
      generations: [],
      events: [],
    };
    taskMap.set(taskId, entry);
    return entry;
  }

  for (const event of events) {
    const taskId = event.task_id ?? liveJob?.task_id ?? liveJob?.taskId ?? null;
    if (!taskId) {
      continue;
    }
    const task = getTask(taskId);
    task.events.push(event);
    const generationNumber = event.generation;
    if (typeof generationNumber !== "number") {
      continue;
    }
    let generation = task.generations.find((item) => item.generation === generationNumber);
    if (!generation) {
      generation = {
        generation: generationNumber,
        status: "queued",
        summary: null,
        acceptedCount: 0,
        memoryDelta: 0,
        branches: [],
      };
      task.generations.push(generation);
    }
    if (event.phase === "generation_started") {
      generation.status = "running";
      generation.summary = event.message ?? null;
      task.status = "running";
    }
    if (event.phase === "generation_finished") {
      generation.status = "completed";
      generation.summary = event.message ?? generation.summary;
      task.currentBest = event.message ?? task.currentBest;
      task.status = liveJob?.status === "completed" ? "completed" : "running";
    }
    if (event.accepted_to_frontier) {
      generation.acceptedCount += 1;
    }
    generation.memoryDelta += event.memory_delta ?? 0;

    if (!event.branch_id) {
      continue;
    }
    let branch = generation.branches.find((item) => item.branchId === event.branch_id);
    if (!branch) {
      branch = {
        branchId: event.branch_id,
        branchIndex: event.branch_index ?? generation.branches.length + 1,
        status: "queued",
        parentCandidate: event.parent_candidate ?? "n/a",
        selectedModel: null,
        candidateMessages: [],
        memoryMessage: null,
        acceptedToFrontier: false,
        improvedGlobalBest: false,
        memoryDelta: 0,
      };
      generation.branches.push(branch);
    }
    branch.parentCandidate = event.parent_candidate ?? branch.parentCandidate;
    if (event.phase === "branch_started") {
      branch.status = "running";
    } else if (event.phase === "proposal_generated") {
      branch.status = "running";
      branch.selectedModel = event.candidate ?? branch.selectedModel;
    } else if (event.phase === "candidate_verified" && event.message) {
      branch.candidateMessages = [...branch.candidateMessages, event.message];
    } else if (event.phase === "memory_writeback" || event.phase === "memory_skipped") {
      branch.status = "completed";
      branch.memoryMessage = event.message ?? branch.memoryMessage;
      branch.acceptedToFrontier = Boolean(event.accepted_to_frontier);
      branch.improvedGlobalBest = Boolean(event.improved_global_best);
      branch.memoryDelta = event.memory_delta ?? branch.memoryDelta;
    }
  }

  return [...taskMap.values()]
    .map((task) => ({
      ...task,
      generations: task.generations
        .map((generation) => ({
          ...generation,
          branches: [...generation.branches].sort((left, right) => left.branchIndex - right.branchIndex),
        }))
        .sort((left, right) => left.generation - right.generation),
    }))
    .sort((left, right) => left.taskId.localeCompare(right.taskId));
}

function deltaChart(run: Run) {
  const points = run.objective_curve ?? [];
  if (!points.length) {
    return null;
  }
  const spec = run.task.objective_spec;
  const label = objectiveLabel(spec);
  const baselineObjective = numeric(points[0].objective);
  const baselineJ = numeric(points[0].J);
  const chartPoints = points.map((point, index) => ({
    generation: point.generation,
    index,
    objectiveDelta: numeric(point.objective) - baselineObjective,
    jDelta: numeric(point.J) - baselineJ,
    acceptedCount: numeric(point.accepted_count ?? 0),
  }));
  const width = 700;
  const height = 260;
  const padding = 26;
  const plotWidth = width - padding * 2;
  const plotHeight = height - padding * 2;
  const values = chartPoints.flatMap((point) => [point.objectiveDelta, point.jDelta, 0]);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const x = (index: number) => padding + (chartPoints.length === 1 ? plotWidth / 2 : (plotWidth * index) / (chartPoints.length - 1));
  const y = (value: number) => padding + plotHeight - ((value - min) / range) * plotHeight;
  const pathFor = (key: "objectiveDelta" | "jDelta") =>
    chartPoints.map((point, index) => `${index === 0 ? "M" : "L"} ${x(point.index)} ${y(point[key])}`).join(" ");

  return (
    <div className="chart-wrap">
      <svg className="chart-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Generation deltas chart">
        {[0.2, 0.4, 0.6, 0.8].map((ratio) => {
          const lineY = padding + plotHeight * ratio;
          return <line key={ratio} className="chart-grid-line" x1={padding} x2={width - padding} y1={lineY} y2={lineY} />;
        })}
        <line className="chart-axis" x1={padding} x2={width - padding} y1={y(0)} y2={y(0)} />
        <path className="chart-line objective-line" d={pathFor("objectiveDelta")} />
        <path className="chart-line j-line" d={pathFor("jDelta")} />
        {chartPoints.map((point) => (
          <g key={`chart-${run.task.id}-${point.generation}`}>
            <circle className={`chart-point ${point.acceptedCount > 0 ? "accepted" : "candidate"}`} cx={x(point.index)} cy={y(point.objectiveDelta)} r="4.6" />
            <circle className={`chart-point j-point ${point.acceptedCount > 0 ? "accepted" : "candidate"}`} cx={x(point.index)} cy={y(point.jDelta)} r="4.6" />
            <text className="chart-label" x={x(point.index)} y={height - 8} textAnchor="middle">
              g{point.generation}
            </text>
          </g>
        ))}
      </svg>
      <div className="legend">
        <span className="legend-item">
          <span className="legend-swatch objective-line-swatch" />
          {label} delta
        </span>
        <span className="legend-item">
          <span className="legend-swatch j-line-swatch" />
          J delta
        </span>
        <span className="legend-item">
          <span className="legend-swatch accepted-swatch" />
          accepted branches in generation
        </span>
      </div>
    </div>
  );
}

function memoryDeltaChart(run: Run) {
  const rows = run.generations.map((generation) => ({
    generation: generation.generation,
    memoryDelta: numeric(generation.memory_delta ?? 0),
  }));
  if (!rows.length) {
    return null;
  }
  const width = 700;
  const height = 230;
  const padding = 28;
  const plotWidth = width - padding * 2;
  const plotHeight = height - padding * 2;
  const extent = Math.max(...rows.map((row) => Math.abs(row.memoryDelta)), 1);
  const zeroY = padding + plotHeight / 2;
  const barWidth = plotWidth / Math.max(rows.length * 1.4, 2);
  const x = (index: number) => padding + index * (barWidth + 18);
  const barHeight = (value: number) => (Math.abs(value) / extent) * (plotHeight / 2 - 16);

  return (
    <div className="chart-wrap">
      <svg className="chart-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Memory delta chart">
        <line className="chart-axis" x1={padding} x2={width - padding} y1={zeroY} y2={zeroY} />
        {rows.map((row, index) => {
          const heightValue = barHeight(row.memoryDelta);
          const positive = row.memoryDelta >= 0;
          return (
            <g key={`memory-${run.task.id}-${row.generation}`}>
              <rect
                className={`memory-bar ${positive ? "positive-bar" : "negative-bar"}`}
                x={x(index)}
                y={positive ? zeroY - heightValue : zeroY}
                width={Math.max(barWidth, 18)}
                height={heightValue}
                rx="10"
              />
              <text className="chart-label" x={x(index) + Math.max(barWidth, 18) / 2} y={height - 8} textAnchor="middle">
                g{row.generation}
              </text>
              <text className="chart-value" x={x(index) + Math.max(barWidth, 18) / 2} y={positive ? zeroY - heightValue - 8 : zeroY + heightValue + 16} textAnchor="middle">
                {row.memoryDelta >= 0 ? `+${row.memoryDelta}` : row.memoryDelta}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="legend">
        <span className="legend-item">
          <span className="legend-swatch positive-swatch" />
          positive write-back
        </span>
        <span className="legend-item">
          <span className="legend-swatch negative-swatch" />
          negative write-back
        </span>
      </div>
    </div>
  );
}

function metricTemplate(spec: ObjectiveSpec, jSpec: JSpec) {
  return (
    <section className="subpanel">
      <div className="subpanel-header">
        <p className="eyebrow">metric template</p>
        <div className="badge-row">
          <span className="badge">{directionCopy(spec.direction)}</span>
          <span className="badge">J always max</span>
        </div>
      </div>
      <div className="template-grid">
        <article className="template-card">
          <div className="template-label">{objectiveLabel(spec)}</div>
          <p className="small">{spec.summary_template}</p>
          <p className="template-formula">{spec.formula}</p>
        </article>
        <article className="template-card">
          <div className="template-label">{jSpec.display_name}</div>
          <p className="small">{jSpec.summary_template}</p>
          <p className="template-formula">{jSpec.formula}</p>
          <p className="small">{jSpec.delta_template}</p>
        </article>
      </div>
    </section>
  );
}

function candidateCard(candidate: Candidate, objectiveSpec: ObjectiveSpec, tone: "winner" | "candidate" = "candidate") {
  return (
    <details className={`detail-card ${tone}`} key={candidate.candidate_id ?? candidate.agent}>
      <summary className="detail-summary">
        <div>
          <strong>{candidate.label}</strong>
          <div className="detail-summary-copy">{candidate.candidate_summary}</div>
        </div>
        <div className="badge-row">
          <span className={`badge ${tone === "winner" ? "good" : ""}`}>{tone}</span>
          <span className="badge">{candidate.proposal_model ?? "baseline"}</span>
          <span className="badge">{candidate.metrics.verifier_status ?? "n/a"}</span>
        </div>
      </summary>
      <div className="detail-body">
        <div className="metric-grid compact-metrics">
          {metric(objectiveLabel(objectiveSpec), formatValue(candidate.metrics.objective, objectiveSpec.unit ? ` ${objectiveSpec.unit}` : ""))}
          {metric("objective_score", formatValue(candidate.metrics.objective_score))}
          {metric("J", formatValue(candidate.metrics.J))}
          {metric("benchmark", candidate.metrics.benchmark_ms == null ? "n/a" : `${candidate.metrics.benchmark_ms} ms`)}
          {metric("tests", `${candidate.metrics.passed_tests ?? "n/a"}/${candidate.metrics.total_tests ?? "n/a"}`)}
          {metric("workspace", shortPath(candidate.workspace_path))}
        </div>
        <p className="muted">{candidate.strategy}</p>
        <p className="small">{candidate.rationale}</p>
        <pre className="code-block"><code>{candidate.source_code}</code></pre>
      </div>
    </details>
  );
}

function branchCard(branch: Branch, objectiveSpec: ObjectiveSpec, openByDefault = false) {
  return (
    <details className="detail-card branch-card" key={branch.branch_id} open={openByDefault}>
      <summary className="detail-summary">
        <div>
          <strong>{branch.branch_id}</strong>
          <div className="detail-summary-copy">
            parent {branch.parent_candidate.agent} → winner {branch.winner.label}
          </div>
        </div>
        <div className="badge-row">
          <span className={`badge ${branch.winner_accepted ? "good" : "warn"}`}>{branch.winner_accepted ? "accepted" : "rejected"}</span>
          <span className={`badge ${branch.memory_delta > 0 ? "good" : branch.memory_delta < 0 ? "warn" : ""}`}>
            memory {branch.memory_delta > 0 ? `+${branch.memory_delta}` : branch.memory_delta}
          </span>
          <span className="badge">delta_J {formatSigned(branch.delta_J, 4)}</span>
        </div>
      </summary>
      <div className="detail-body stack">
        <div className="metric-grid compact-metrics">
          {metric("parent", branch.parent_candidate.agent)}
          {metric(objectiveLabel(objectiveSpec), formatValue(branch.winner.metrics.objective, objectiveSpec.unit ? ` ${objectiveSpec.unit}` : ""))}
          {metric("J", formatValue(branch.winner.metrics.J))}
          {metric("improved global best", String(Boolean(branch.winner_improved_global_best)))}
        </div>
        {branch.rejection_reason ? <p className="small muted">{branch.rejection_reason}</p> : null}
        <div>
          <div className="section-label">Retrieved memory</div>
          <ul className="dense-list">
            {branch.retrieved_memories.length ? (
              branch.retrieved_memories.map((memory) => (
                <li key={memory.experience_id}>
                  <strong>{memory.experience_id}</strong>
                  <div className="small">{memory.prompt_fragment || memory.strategy_hypothesis || "No summary."}</div>
                </li>
              ))
            ) : (
              <li>No retrieved memory.</li>
            )}
          </ul>
        </div>
        <div className="stack">
          {branch.candidates.map((candidate) =>
            candidateCard(candidate, objectiveSpec, candidate.candidate_id === branch.winner.candidate_id ? "winner" : "candidate"),
          )}
        </div>
      </div>
    </details>
  );
}

function generationCard(generation: Generation, objectiveSpec: ObjectiveSpec, openByDefault = false) {
  return (
    <details className="detail-card generation-card" key={generation.generation} open={openByDefault}>
      <summary className="detail-summary">
        <div>
          <strong>Generation {generation.generation}</strong>
          <div className="detail-summary-copy">{generation.winner.candidate_summary}</div>
        </div>
        <div className="badge-row">
          <span className={`badge ${generation.accepted_count ? "good" : "warn"}`}>{generation.accepted_count ?? 0} accepts</span>
          <span className={`badge ${numeric(generation.memory_delta) > 0 ? "good" : numeric(generation.memory_delta) < 0 ? "warn" : ""}`}>
            memory {numeric(generation.memory_delta) > 0 ? `+${generation.memory_delta}` : generation.memory_delta ?? 0}
          </span>
          <span className="badge">delta_J {formatSigned(generation.delta_J, 4)}</span>
        </div>
      </summary>
      <div className="detail-body stack">
        <div className="metric-grid compact-metrics">
          {metric("parents", generation.parents?.length ?? generation.branches.length)}
          {metric("winner", generation.winner.label)}
          {metric(objectiveLabel(objectiveSpec), formatValue(generation.winner.metrics.objective, objectiveSpec.unit ? ` ${objectiveSpec.unit}` : ""))}
          {metric("winner J", formatValue(generation.winner.metrics.J))}
          {metric("positive write-backs", generation.positive_writebacks ?? 0)}
          {metric("negative write-backs", generation.negative_writebacks ?? 0)}
        </div>
        <div className="stack">
          {generation.branches.map((branch, index) => branchCard(branch, objectiveSpec, index === 0))}
        </div>
      </div>
    </details>
  );
}

function liveTaskSection(task: LiveTaskCard, isOpen: boolean, onToggle: () => void) {
  return (
    <article className="task-card" key={task.taskId}>
      <button className="accordion-toggle" onClick={onToggle} type="button">
        <div className="accordion-copy">
          <p className="eyebrow">live task</p>
          <h3>{task.title}</h3>
          <p className="muted">{task.description}</p>
        </div>
        <div className="accordion-meta">
          <span className={`badge ${task.status === "completed" ? "good" : task.status === "running" ? "warn" : ""}`}>{task.status}</span>
          <span className="badge">{task.model}</span>
          <span className="badge">branching {task.branchingFactor}</span>
          <span className="badge">
            g{task.generations.length}/{task.generationBudget || "?"}
          </span>
        </div>
      </button>
      <div className="task-summary-row">
        <span className="summary-pill">{task.taskId}</span>
        <span className="summary-pill">{task.currentBest ?? "Current best pending"}</span>
      </div>
      {isOpen ? (
        <div className="accordion-body stack">
          {task.generations.length ? (
            task.generations.map((generation) => (
              <article className="live-generation-card" key={`${task.taskId}-${generation.generation}`}>
                <div className="panel-header">
                  <div>
                    <div className="section-label">Generation {generation.generation}</div>
                    <p className="small">{generation.summary ?? "Waiting for generation summary."}</p>
                  </div>
                  <div className="badge-row">
                    <span className={`badge ${generation.status === "completed" ? "good" : generation.status === "running" ? "warn" : ""}`}>{generation.status}</span>
                    <span className="badge">accepts {generation.acceptedCount}</span>
                    <span className={`badge ${generation.memoryDelta > 0 ? "good" : generation.memoryDelta < 0 ? "warn" : ""}`}>
                      memory {generation.memoryDelta > 0 ? `+${generation.memoryDelta}` : generation.memoryDelta}
                    </span>
                  </div>
                </div>
                <div className="branch-grid">
                  {generation.branches.map((branch) => (
                    <article className="branch-pill-card" key={branch.branchId}>
                      <div className="badge-row">
                        <span className="badge">{branch.branchId}</span>
                        <span className={`badge ${branch.acceptedToFrontier ? "good" : ""}`}>{branch.selectedModel ?? "awaiting model"}</span>
                      </div>
                      <p className="small">
                        parent <strong>{branch.parentCandidate}</strong>
                      </p>
                      <ul className="dense-list compact-list">
                        {branch.candidateMessages.length ? (
                          branch.candidateMessages.map((message) => <li key={`${branch.branchId}-${message}`}>{message}</li>)
                        ) : (
                          <li>No verified candidates yet.</li>
                        )}
                      </ul>
                      {branch.memoryMessage ? <p className="small">{branch.memoryMessage}</p> : null}
                    </article>
                  ))}
                </div>
              </article>
            ))
          ) : (
            <p className="small">Waiting for task events.</p>
          )}
          <details className="detail-card">
            <summary className="detail-summary">
              <div>
                <strong>Event log</strong>
                <div className="detail-summary-copy">Raw task-scoped backend events.</div>
              </div>
            </summary>
            <div className="detail-body">
              <ul className="dense-list">
                {task.events.map((event, index) => (
                  <li key={`${task.taskId}-${event.timestamp ?? "t"}-${index}`}>
                    <strong>{event.phase ?? event.event_type ?? "event"}</strong>
                    <div className="small">
                      {[event.timestamp, event.branch_id, event.message].filter(Boolean).join(" · ")}
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </details>
        </div>
      ) : null}
    </article>
  );
}

function runCard(run: Run, jSpec: JSpec, isOpen: boolean, onToggle: () => void) {
  const reportSvg = artifactUrl(run.handoff_bundle?.manifest?.artifact_paths.report_svg);
  const objectiveSpec = run.task.objective_spec;
  return (
    <article className="task-card completed-card" key={run.task.id}>
      <button className="accordion-toggle" onClick={onToggle} type="button">
        <div className="accordion-copy">
          <p className="eyebrow">completed task</p>
          <h3>{run.task.title}</h3>
          <p className="muted">{run.task.description}</p>
        </div>
        <div className="accordion-meta">
          <span className="badge">{run.active_model}</span>
          <span className="badge">branching {run.task.branching_factor}</span>
          <span className="badge">
            {objectiveLabel(objectiveSpec)} {formatValue(run.winner.metrics.objective, objectiveSpec.unit ? ` ${objectiveSpec.unit}` : "")}
          </span>
          <span className={`badge ${numeric(run.run_delta_J ?? run.delta_J) >= 0 ? "good" : "warn"}`}>
            run_delta_J {formatSigned(run.run_delta_J ?? run.delta_J, 4)}
          </span>
        </div>
      </button>
      <div className="task-summary-row">
        <span className="summary-pill">{run.task.id}</span>
        <span className="summary-pill">{benchmarkTierLabel(run.included_in_main_comparison)}</span>
        <span className="summary-pill">{trackLabel(run.track)}</span>
        <span className="summary-pill">{directionCopy(objectiveSpec.direction)}</span>
        <span className="summary-pill">{run.selection_reason}</span>
      </div>
      {isOpen ? (
        <div className="accordion-body stack">
          {metricTemplate(objectiveSpec, run.j_spec ?? jSpec)}

          <div className="metric-grid">
            {metric("baseline objective", formatValue(run.baseline.metrics.objective, objectiveSpec.unit ? ` ${objectiveSpec.unit}` : ""))}
            {metric("winner objective", formatValue(run.winner.metrics.objective, objectiveSpec.unit ? ` ${objectiveSpec.unit}` : ""))}
            {metric("run_delta_J", formatSigned(run.run_delta_J ?? run.delta_J, 4))}
            {metric("generations", run.generations.length)}
            {metric("write-backs", run.added_experiences?.length ?? 0)}
            {metric("memory", `${run.memory_before_count ?? "n/a"} → ${run.memory_after_count ?? "n/a"}`)}
          </div>

          <div className="split-grid">
            {candidateCard(run.baseline, objectiveSpec, "candidate")}
            {candidateCard(run.winner, objectiveSpec, "winner")}
          </div>

          <div className="split-grid report-grid">
            <section className="subpanel">
              <div className="subpanel-header">
                <div>
                  <p className="eyebrow">objective + J</p>
                  <h4>Generational deltas</h4>
                </div>
              </div>
              {deltaChart(run)}
            </section>
            <section className="subpanel">
              <div className="subpanel-header">
                <div>
                  <p className="eyebrow">memory delta</p>
                  <h4>Per-generation net change</h4>
                </div>
              </div>
              {memoryDeltaChart(run)}
            </section>
          </div>

          <section className="subpanel">
            <div className="subpanel-header">
              <div>
                <p className="eyebrow">report</p>
                <h4>SVG summary</h4>
              </div>
            </div>
            {reportSvg ? <img className="report-figure" src={reportSvg} alt={`${run.task.id} report`} /> : deltaChart(run)}
          </section>

          <section className="subpanel">
            <div className="subpanel-header">
              <div>
                <p className="eyebrow">artifacts</p>
                <h4>Manifest and memory ledger</h4>
              </div>
            </div>
            <div className="artifact-grid">
              {metric("manifest", shortPath(run.handoff_bundle?.manifest_path))}
              {metric("payload", shortPath(run.handoff_bundle?.manifest?.artifact_paths.payload))}
              {metric("trace", shortPath(run.handoff_bundle?.manifest?.artifact_paths.trace))}
              {metric("llm trace", shortPath(run.handoff_bundle?.manifest?.artifact_paths.llm_trace_jsonl))}
              {metric("memory markdown", shortPath(run.handoff_bundle?.manifest?.artifact_paths.memory_markdown))}
              {metric("report svg", shortPath(run.handoff_bundle?.manifest?.artifact_paths.report_svg))}
            </div>
            <pre className="code-block compact"><code>{run.memory_markdown}</code></pre>
          </section>

          <section className="stack">
            {run.generations.map((generation, index) => generationCard(generation, objectiveSpec, index === run.generations.length - 1))}
          </section>
        </div>
      ) : null}
    </article>
  );
}

function themeChoices(): ThemePreference[] {
  return ["system", "light", "dark"];
}

export function App() {
  const [runtimeInfo, setRuntimeInfo] = useState<RuntimeInfo>(emptyRuntime());
  const [payload, setPayload] = useState<Payload>(emptyPayload());
  const [selectedTaskId, setSelectedTaskId] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [branchingFactorInput, setBranchingFactorInput] = useState("4");
  const [themePreference, setThemePreference] = useState<ThemePreference>("system");
  const [liveJob, setLiveJob] = useState<JobState | null>({
    status: "loading",
    events: [{ phase: "boot", message: "Loading runtime and task catalog." }],
  });
  const [error, setError] = useState<ErrorPayload | null>(null);
  const [openLiveTasks, setOpenLiveTasks] = useState<Record<string, boolean>>({});
  const [openCompletedTasks, setOpenCompletedTasks] = useState<Record<string, boolean>>({});
  const pollToken = useRef(0);

  const selectedTask = useMemo(
    () => payload.task_catalog.find((task) => task.id === selectedTaskId) ?? payload.task_catalog[0] ?? null,
    [payload.task_catalog, selectedTaskId],
  );

  const comparableTasks = useMemo(
    () => payload.task_catalog.filter((task) => task.included_in_main_comparison),
    [payload.task_catalog],
  );

  const experimentTasks = useMemo(
    () => payload.task_catalog.filter((task) => !task.included_in_main_comparison),
    [payload.task_catalog],
  );

  const liveTasks = useMemo(
    () => summarizeLiveTasks(liveJob?.events ?? [], payload.task_catalog, liveJob, payload.runs),
    [liveJob, payload.task_catalog, payload.runs],
  );

  const comparableRuns = useMemo(
    () => payload.runs.filter((run) => run.included_in_main_comparison),
    [payload.runs],
  );

  const experimentRuns = useMemo(
    () => payload.runs.filter((run) => !run.included_in_main_comparison),
    [payload.runs],
  );

  useEffect(() => {
    const stored = window.localStorage.getItem("autoresearch-theme");
    if (stored === "system" || stored === "light" || stored === "dark") {
      setThemePreference(stored);
    }
  }, []);

  useEffect(() => {
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const applyTheme = () => {
      const resolvedTheme = themePreference === "system" ? (mediaQuery.matches ? "dark" : "light") : themePreference;
      document.documentElement.dataset.theme = resolvedTheme;
    };
    window.localStorage.setItem("autoresearch-theme", themePreference);
    applyTheme();
    mediaQuery.addEventListener("change", applyTheme);
    return () => mediaQuery.removeEventListener("change", applyTheme);
  }, [themePreference]);

  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      try {
        const [runtime, tasks] = await Promise.all([loadRuntime(), loadTasks()]);
        if (cancelled) {
          return;
        }
        setRuntimeInfo(runtime);
        setSelectedModel(runtime.active_model);
        setPayload(emptyPayload(tasks));
        const defaultTask = tasks.find((task) => task.included_in_main_comparison) ?? tasks[0] ?? null;
        setSelectedTaskId(defaultTask?.id ?? "");
        setBranchingFactorInput(String(defaultTask?.branching_factor ?? 4));
        setLiveJob({
          status: "loading",
          events: [{ phase: "boot", message: "Loading latest cached run." }],
        });
        const latest = await loadLatestRun();
        if (cancelled) {
          return;
        }
        const normalized = normalizePayload(latest, tasks);
        setPayload(normalized);
        setOpenCompletedTasks(
          normalized.runs[0]
            ? {
                [normalized.runs[0].task.id]: true,
              }
            : {},
        );
        setLiveJob(null);
        setError(null);
      } catch (caught) {
        if (cancelled) {
          return;
        }
        setError(asErrorPayload(caught));
        setLiveJob({ status: "failed", events: [] });
      }
    }

    void bootstrap();
    return () => {
      cancelled = true;
      pollToken.current += 1;
    };
  }, []);

  useEffect(() => {
    if (!selectedTask) {
      return;
    }
    setBranchingFactorInput(String(selectedTask.branching_factor ?? 4));
  }, [selectedTask?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (liveJob?.status === "running") {
      return undefined;
    }

    const timer = window.setInterval(async () => {
      try {
        const latest = await loadLatestRun();
        setPayload((previous) => normalizePayload(latest, previous.task_catalog));
      } catch {
        return;
      }
    }, 5000);

    return () => {
      window.clearInterval(timer);
    };
  }, [liveJob?.status]);

  useEffect(() => {
    if (!liveTasks.length) {
      return;
    }
    setOpenLiveTasks((previous) => {
      const next = { ...previous };
      for (const task of liveTasks) {
        if (!(task.taskId in next) && task.status !== "queued") {
          next[task.taskId] = true;
        }
      }
      return next;
    });
  }, [liveTasks]);

  function toggleLiveTask(taskId: string) {
    setOpenLiveTasks((previous) => ({ ...previous, [taskId]: !previous[taskId] }));
  }

  function toggleCompletedTask(taskId: string) {
    setOpenCompletedTasks((previous) => ({ ...previous, [taskId]: !previous[taskId] }));
  }

  async function runTask(taskId: string | null) {
    const model = selectedModel || runtimeInfo.active_model;
    const branchingFactor = Math.max(1, Math.floor(numeric(branchingFactorInput || selectedTask?.branching_factor || 4)));
    pollToken.current += 1;
    const token = pollToken.current;
    setError(null);
    setLiveJob({
      status: "running",
      taskId,
      model,
      branching_factor: branchingFactor,
      events: [{ phase: "queued", message: `Starting ${taskId ?? "full sequence"} with ${model}.` }],
    });

    try {
      const start = await startJob(taskId, model, branchingFactor);
      let job = await loadJob(start.job_id);
      while (job.status === "running" && token === pollToken.current) {
        setLiveJob(job);
        await sleep(280);
        job = await loadJob(start.job_id);
      }
      if (token !== pollToken.current) {
        return;
      }
      setLiveJob(job);
      if (job.status === "failed") {
        setError(asErrorPayload(job));
        return;
      }
      if (job.payload) {
        const normalized = normalizePayload(job.payload, payload.task_catalog);
        setPayload(normalized);
        setOpenCompletedTasks((previous) => {
          const next = { ...previous };
          for (const run of normalized.runs) {
            if (taskId === run.task.id || (!taskId && !(run.task.id in next))) {
              next[run.task.id] = true;
            }
          }
          return next;
        });
        setError(null);
      }
    } catch (caught) {
      if (token !== pollToken.current) {
        return;
      }
      setError(asErrorPayload(caught));
      setLiveJob({
        status: "failed",
        taskId,
        model,
        branching_factor: Math.max(1, Math.floor(numeric(branchingFactorInput || 4))),
        events: [],
      });
    }
  }

  const taskJSpec = payload.j_spec ?? emptyJSpec();

  return (
    <main className="app-shell">
      <section className="topbar">
        <div>
          <p className="eyebrow">autoresearch</p>
          <strong className="topbar-title">Task-centered evolution workbench</strong>
        </div>
        <div className="theme-toggle" role="tablist" aria-label="Theme mode">
          {themeChoices().map((choice) => (
            <button
              key={choice}
              className={`theme-chip ${themePreference === choice ? "active" : ""}`}
              onClick={() => setThemePreference(choice)}
              type="button"
            >
              {choice}
            </button>
          ))}
        </div>
      </section>

      <section className="panel control-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">control room</p>
            <h1>Branching evolution, task by task.</h1>
            <p className="muted hero-copy">
              Each task carries its own description, objective template, and branchable generation trace. The interface stays collapsed until a specific task actually moves.
            </p>
          </div>
          <div className="badge-row">
            <span className="badge">{runtimeInfo.mode}</span>
            <span className="badge">runtime {runtimeInfo.active_model}</span>
            <span className="badge">memory {payload.summary.memory_size_after_run}</span>
          </div>
        </div>

        <div className="control-grid triple">
          <label className="field">
            <span className="field-label">Task</span>
            <select className="control" value={selectedTask?.id ?? ""} onChange={(event) => setSelectedTaskId(event.target.value)}>
              {comparableTasks.length ? (
                <optgroup label="Main benchmark">
                  {comparableTasks.map((task) => (
                    <option key={task.id} value={task.id}>
                      {task.id}
                    </option>
                  ))}
                </optgroup>
              ) : null}
              {experimentTasks.length ? (
                <optgroup label="Small experiments">
                  {experimentTasks.map((task) => (
                    <option key={task.id} value={task.id}>
                      {task.id}
                    </option>
                  ))}
                </optgroup>
              ) : null}
            </select>
          </label>
          <label className="field">
            <span className="field-label">Model</span>
            <select className="control" value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)} disabled={!runtimeInfo.available_models.length}>
              {runtimeInfo.available_models.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span className="field-label">Branching Factor</span>
            <input className="control" type="number" min={1} step={1} value={branchingFactorInput} onChange={(event) => setBranchingFactorInput(event.target.value)} />
          </label>
        </div>

        <div className="button-row">
          <button className="action primary" onClick={() => void runTask(selectedTask?.id ?? null)} type="button">
            Run selected task
          </button>
          <button className="action" onClick={() => void runTask(null)} type="button">
            Run main benchmark sequence
          </button>
        </div>
        <p className="small muted">The default sequence runs only comparable benchmark tasks. Small experiments stay manual and out of the main comparison lane.</p>

        {selectedTask ? (
          <div className="task-preview">
            <div className="task-summary-row">
              <span className="summary-pill">{benchmarkTierLabel(selectedTask.included_in_main_comparison)}</span>
              <span className="summary-pill">{trackLabel(selectedTask.track)}</span>
              <span className="summary-pill">{selectedTask.answer_metric}</span>
              <span className="summary-pill">{selectedTask.function_name}</span>
              <span className="summary-pill">
                {selectedTask.generation_budget} generations × {selectedTask.candidate_budget} candidates × branching {branchingFactorInput}
              </span>
            </div>
            <p className="muted">{selectedTask.description}</p>
            {metricTemplate(selectedTask.objective_spec, taskJSpec)}
          </div>
        ) : null}
      </section>

      {error ? (
        <section className="panel error-panel">
          <p className="eyebrow">terminal failure</p>
          <h2>{error.error_type}</h2>
          <p className="muted">{error.error}</p>
          <div className="metric-grid">
            {metric("terminal", String(Boolean(error.terminal)))}
            {metric("model", error.model ?? "n/a")}
          </div>
        </section>
      ) : null}

      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">runtime</p>
            <h2>Verifier and proposal runtime</h2>
          </div>
          <div className="badge-row">
            <span className="badge">primary {runtimeInfo.primary_model}</span>
            <span className="badge">selected {selectedModel || runtimeInfo.active_model}</span>
          </div>
        </div>
        <div className="metric-grid">
          {metric("cached runs", payload.runs.length)}
          {metric("main runs", comparableRuns.length)}
          {metric("experiment runs", experimentRuns.length)}
          {metric("latest generated", payload.summary.generated_at)}
          {metric("memory size", payload.summary.memory_size_after_run)}
          {metric("main write backs", payload.summary.write_backs)}
          {metric("experiment write backs", payload.summary.experiment_write_backs)}
          {metric("temperature", runtimeInfo.temperature)}
          {metric("max tokens", runtimeInfo.max_tokens)}
        </div>
      </section>

      <section className="panel stack">
        <div className="panel-header">
          <div>
            <p className="eyebrow">live tasks</p>
            <h2>Task-scoped runtime trace</h2>
          </div>
          <div className="badge-row">
            <span className={`badge ${liveJob?.status === "completed" ? "good" : liveJob?.status === "running" ? "warn" : ""}`}>{liveJob?.status ?? "idle"}</span>
            <span className="badge">{liveJob?.model ?? selectedModel ?? "n/a"}</span>
          </div>
        </div>
        {liveTasks.length ? (
          liveTasks.map((task) => liveTaskSection(task, Boolean(openLiveTasks[task.taskId]), () => toggleLiveTask(task.taskId)))
        ) : (
          <section className="empty-state">
            <h3>No live task yet</h3>
            <p className="muted">Start a task or a full sequence and the frontend will group all live events by task, then by generation and branch.</p>
          </section>
        )}
      </section>

      <section className="panel stack">
        <div className="panel-header">
          <div>
            <p className="eyebrow">completed tasks</p>
            <h2>Task summaries and generation details</h2>
          </div>
          <div className="badge-row">
            <span className="badge">{payload.summary.active_model}</span>
            <span className="badge">{payload.summary.num_tasks} main tasks</span>
            <span className="badge">{payload.summary.experiment_runs} experiment runs</span>
          </div>
        </div>
        <section className="subpanel stack">
          <div className="subpanel-header">
            <div>
              <p className="eyebrow">main benchmark</p>
              <h4>Main benchmark comparison</h4>
            </div>
            <div className="badge-row">
              <span className="badge">{comparableTasks.length} registered</span>
              <span className="badge">{comparableRuns.length} cached</span>
            </div>
          </div>
          {comparableRuns.length ? (
            comparableRuns.map((run) => runCard(run, taskJSpec, Boolean(openCompletedTasks[run.task.id]), () => toggleCompletedTask(run.task.id)))
          ) : (
            <section className="empty-state">
              <h3>No main benchmark run yet</h3>
              <p className="muted">Full-sequence runs land here, and only comparable tasks contribute to the default benchmark lane.</p>
            </section>
          )}
        </section>

        <section className="subpanel stack">
          <div className="subpanel-header">
            <div>
              <p className="eyebrow">small experiments</p>
              <h4>Small Experiments</h4>
            </div>
            <div className="badge-row">
              <span className="badge">{experimentTasks.length} registered</span>
              <span className="badge">{experimentRuns.length} cached</span>
            </div>
          </div>
          {experimentRuns.length ? (
            experimentRuns.map((run) => runCard(run, taskJSpec, Boolean(openCompletedTasks[run.task.id]), () => toggleCompletedTask(run.task.id)))
          ) : (
            <section className="empty-state">
              <h3>No small experiment run yet</h3>
              <p className="muted">Manual smoke and regression runs stay here and never enter the default comparable benchmark summary.</p>
            </section>
          )}
        </section>
      </section>
    </main>
  );
}

export default App;
