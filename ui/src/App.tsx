import { useEffect, useMemo, useRef, useState } from "react";

import { loadJob, loadLatestRun, loadRuntime, loadTasks, startJob } from "./api";
import type {
  AddedExperience,
  Candidate,
  ErrorPayload,
  Generation,
  LiveEvent,
  JobState,
  Payload,
  Run,
  RuntimeInfo,
  TaskSummary,
} from "./types";

type ThemePreference = "system" | "light" | "dark";

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

function emptyPayload(taskCatalog: TaskSummary[] = []): Payload {
  return {
    summary: {
      generated_at: "n/a",
      run_mode: "llm-required",
      active_model: "n/a",
      num_tasks: 0,
      total_generations: 0,
      initial_memory_count: 0,
      memory_size_after_run: 0,
      write_backs: 0,
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
    },
    audit: {
      workspace_root: "n/a",
      session_id: "n/a",
    },
    task_catalog: taskCatalog,
    runs: [],
  };
}

function normalizePayload(payload: Payload, fallbackCatalog: TaskSummary[]): Payload {
  return {
    ...payload,
    task_catalog: payload.task_catalog?.length ? payload.task_catalog : fallbackCatalog,
    runs: Array.isArray(payload.runs) ? payload.runs : [],
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

type LiveGenerationCard = {
  generation: number;
  status: "queued" | "running" | "completed";
  retrievedMemories: string | null;
  selectedModel: string | null;
  candidateMessages: string[];
  memoryWriteback: string | null;
  completion: string | null;
};

function numeric(value: string | number | undefined): number {
  if (typeof value === "number") {
    return value;
  }
  const parsed = Number(value ?? 0);
  return Number.isFinite(parsed) ? parsed : 0;
}

function themeChoices(): ThemePreference[] {
  return ["system", "light", "dark"];
}

function summarizeLiveEvents(events: LiveEvent[]): LiveGenerationCard[] {
  const cards = new Map<number, LiveGenerationCard>();
  for (const event of events) {
    const generation = typeof event.generation === "number" ? event.generation : null;
    if (generation == null) {
      continue;
    }
    const current =
      cards.get(generation) ??
      {
        generation,
        status: "queued",
        retrievedMemories: null,
        selectedModel: null,
        candidateMessages: [],
        memoryWriteback: null,
        completion: null,
      };
    if (event.phase === "generation_started") {
      current.status = "running";
      current.retrievedMemories = event.message ?? null;
    } else if (event.phase === "proposal_generated") {
      current.selectedModel = event.candidate ?? null;
    } else if (event.phase === "candidate_verified" && event.message) {
      current.candidateMessages = [...current.candidateMessages, event.message];
    } else if (event.phase === "memory_writeback") {
      current.memoryWriteback = event.message ?? null;
    } else if (event.phase === "generation_finished") {
      current.status = "completed";
      current.completion = event.message ?? null;
    }
    cards.set(generation, current);
  }
  return [...cards.values()].sort((left, right) => left.generation - right.generation);
}

function runOverviewChart(run: Run) {
  const points = run.objective_curve ?? [];
  if (!points.length) {
    return null;
  }

  const baselineObjective = numeric(points[0].objective);
  const baselineJ = numeric(points[0].J);
  const chartPoints = points.map((point, index) => ({
    generation: point.generation,
    index,
    objectiveDelta: numeric(point.objective) - baselineObjective,
    jDelta: numeric(point.J) - baselineJ,
    accepted: point.accepted,
  }));

  const width = 520;
  const height = 220;
  const padding = 24;
  const plotWidth = width - padding * 2;
  const plotHeight = height - padding * 2;
  const values = chartPoints.flatMap((point) => [point.objectiveDelta, point.jDelta, 0]);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const x = (index: number) =>
    padding + (chartPoints.length === 1 ? plotWidth / 2 : (plotWidth * index) / (chartPoints.length - 1));
  const y = (value: number) => padding + plotHeight - ((value - min) / range) * plotHeight;
  const linePath = (key: "objectiveDelta" | "jDelta") =>
    chartPoints.map((point, index) => `${index === 0 ? "M" : "L"} ${x(point.index)} ${y(point[key])}`).join(" ");

  return (
    <svg className="chart-svg compact-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Run overview chart">
      {[0.2, 0.4, 0.6, 0.8].map((ratio) => {
        const lineY = padding + plotHeight * ratio;
        return <line key={ratio} className="chart-grid-line" x1={padding} x2={width - padding} y1={lineY} y2={lineY} />;
      })}
      <line className="chart-axis" x1={padding} x2={width - padding} y1={y(0)} y2={y(0)} />
      <path className="chart-line objective-line" d={linePath("objectiveDelta")} />
      <path className="chart-line j-line" d={linePath("jDelta")} />
      {chartPoints.map((point) => (
        <g key={`overview-${point.generation}`}>
          <circle
            className={`chart-point ${point.accepted ? "accepted" : "candidate"}`}
            cx={x(point.index)}
            cy={y(point.objectiveDelta)}
            r="4.5"
          />
          <circle
            className={`chart-point j-point ${point.accepted ? "accepted" : "candidate"}`}
            cx={x(point.index)}
            cy={y(point.jDelta)}
            r="4.5"
          />
          <text className="chart-label" x={x(point.index)} y={height - 6} textAnchor="middle">
            g{point.generation}
          </text>
        </g>
      ))}
    </svg>
  );
}

function memoryGrowthChart(run: Run) {
  const generationCount = run.generations.length;
  if (!generationCount) {
    return null;
  }

  const grouped = new Map<number, { positive: number; negative: number }>();
  for (let generation = 1; generation <= generationCount; generation += 1) {
    grouped.set(generation, { positive: 0, negative: 0 });
  }
  for (const experience of run.added_experiences ?? []) {
    const row = grouped.get(experience.generation) ?? { positive: 0, negative: 0 };
    if (experience.experience_outcome === "success") {
      row.positive += 1;
    } else if (experience.experience_outcome === "failure") {
      row.negative += 1;
    }
    grouped.set(experience.generation, row);
  }

  let positiveRunning = 0;
  let negativeRunning = 0;
  const rows = [...grouped.entries()].map(([generation, value]) => {
    positiveRunning += value.positive;
    negativeRunning += value.negative;
    return { generation, positive: positiveRunning, negative: negativeRunning };
  });

  const width = 520;
  const height = 220;
  const padding = 24;
  const plotWidth = width - padding * 2;
  const plotHeight = height - padding * 2;
  const maxValue = Math.max(...rows.flatMap((row) => [row.positive, row.negative]), 1);
  const barWidth = plotWidth / Math.max(rows.length * 2, 2);
  const x = (index: number, offset: number) => padding + index * barWidth * 2 + offset * barWidth;
  const barHeight = (value: number) => (value / maxValue) * (plotHeight - 18);

  return (
    <svg className="chart-svg compact-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Memory growth chart">
      {[0.2, 0.4, 0.6, 0.8].map((ratio) => {
        const lineY = padding + plotHeight * ratio;
        return <line key={ratio} className="chart-grid-line" x1={padding} x2={width - padding} y1={lineY} y2={lineY} />;
      })}
      {rows.map((row, index) => {
        const positiveHeight = barHeight(row.positive);
        const negativeHeight = barHeight(row.negative);
        return (
          <g key={`memory-growth-${row.generation}`}>
            <rect
              className="memory-bar positive-bar"
              x={x(index, 0)}
              y={height - padding - positiveHeight}
              width={Math.max(barWidth - 8, 10)}
              height={positiveHeight}
              rx="8"
            />
            <rect
              className="memory-bar negative-bar"
              x={x(index, 1)}
              y={height - padding - negativeHeight}
              width={Math.max(barWidth - 8, 10)}
              height={negativeHeight}
              rx="8"
            />
            <text className="chart-label" x={x(index, 0.5)} y={height - 6} textAnchor="middle">
              g{row.generation}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function fallbackImprovementChart(run: Run) {
  const points = run.objective_curve ?? [];
  if (!points.length) {
    return null;
  }

  const baselineObjective = numeric(points[0].objective);
  const baselineJ = numeric(points[0].J);
  const chartPoints = points.map((point, index) => ({
    generation: point.generation,
    xIndex: index,
    objectiveDelta: numeric(point.objective) - baselineObjective,
    jDelta: numeric(point.J) - baselineJ,
    accepted: point.accepted,
  }));

  const width = 680;
  const height = 260;
  const padding = 26;
  const plotWidth = width - padding * 2;
  const plotHeight = height - padding * 2;
  const values = chartPoints.flatMap((point) => [point.objectiveDelta, point.jDelta, 0]);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  const x = (index: number) =>
    padding + (chartPoints.length === 1 ? plotWidth / 2 : (plotWidth * index) / (chartPoints.length - 1));
  const y = (value: number) => padding + plotHeight - ((value - min) / range) * plotHeight;

  const makePath = (key: "objectiveDelta" | "jDelta") =>
    chartPoints.map((point, index) => `${index === 0 ? "M" : "L"} ${x(point.xIndex)} ${y(point[key])}`).join(" ");

  return (
    <>
      <svg className="chart-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Improvement curve">
        {[0.25, 0.5, 0.75].map((ratio) => {
          const lineY = padding + plotHeight * ratio;
          return <line key={ratio} className="chart-grid-line" x1={padding} x2={width - padding} y1={lineY} y2={lineY} />;
        })}
        <line className="chart-axis" x1={padding} x2={width - padding} y1={y(0)} y2={y(0)} />
        <path className="chart-line objective-line" d={makePath("objectiveDelta")} />
        <path className="chart-line j-line" d={makePath("jDelta")} />
        {chartPoints.map((point) => (
          <g key={`objective-${point.generation}`}>
            <circle
              className={`chart-point ${point.accepted ? "accepted" : "candidate"}`}
              cx={x(point.xIndex)}
              cy={y(point.objectiveDelta)}
              r="4.5"
            />
            <circle
              className={`chart-point j-point ${point.accepted ? "accepted" : "candidate"}`}
              cx={x(point.xIndex)}
              cy={y(point.jDelta)}
              r="4.5"
            />
            <text className="chart-label" x={x(point.xIndex)} y={height - 6} textAnchor="middle">
              g{point.generation}
            </text>
          </g>
        ))}
      </svg>

      <div className="legend">
        <span className="legend-item">
          <span className="legend-swatch objective-line-swatch" />
          objective delta
        </span>
        <span className="legend-item">
          <span className="legend-swatch j-line-swatch" />
          J delta
        </span>
        <span className="legend-item">
          <span className="legend-swatch accepted-swatch" />
          accepted generation
        </span>
      </div>
    </>
  );
}

function improvementPanel(run: Run) {
  const reportSvg = artifactUrl(run.handoff_bundle?.manifest?.artifact_paths.report_svg);
  const improvementRows = run.improvement_table ?? [];
  return (
    <section className="panel chart-card">
      <div className="panel-header">
        <div>
          <p className="eyebrow">improvement</p>
          <h3>Task-specific report</h3>
        </div>
        <div className="badge-row">
          <span className="badge">{run.generated_at ?? "n/a"}</span>
          <span className="badge">{run.active_model}</span>
        </div>
      </div>
      {reportSvg ? (
        <img className="report-figure" src={reportSvg} alt={`${run.task.id} improvement report`} />
      ) : (
        fallbackImprovementChart(run)
      )}
      {improvementRows.length ? (
        <div className="improvement-table-wrap">
          <table className="improvement-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              {improvementRows.map((row) => (
                <tr key={row.label}>
                  <td>{row.label}</td>
                  <td>{row.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </section>
  );
}

function candidateCard(candidate: Candidate, tone: "winner" | "candidate" = "candidate") {
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
          <span className="badge">{candidate.verifier_status ?? candidate.metrics.verifier_status ?? "n/a"}</span>
        </div>
      </summary>
      <div className="detail-body">
        <div className="metric-grid">
          {metric("objective", candidate.metrics.objective)}
          {metric("J", candidate.metrics.J)}
          {metric("benchmark", candidate.metrics.benchmark_ms == null ? "n/a" : `${candidate.metrics.benchmark_ms} ms`)}
          {metric("speedup", candidate.metrics.speedup_vs_baseline ?? "n/a")}
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

function generationSection(generation: Generation, openByDefault: boolean) {
  return (
    <details className="detail-card generation-card" key={generation.generation} open={openByDefault}>
      <summary className="detail-summary">
        <div>
          <strong>Generation {generation.generation}</strong>
          <div className="detail-summary-copy">{generation.winner.candidate_summary}</div>
        </div>
        <div className="badge-row">
          <span className={`badge ${generation.winner_accepted ? "good" : "warn"}`}>
            {generation.winner_accepted ? "accepted" : "rejected"}
          </span>
          <span className={`badge ${generation.wrote_memory ? "good" : ""}`}>
            {generation.wrote_memory ? "memory write-back" : "no write-back"}
          </span>
          <span className="badge">delta_J {generation.delta_J}</span>
        </div>
      </summary>
      <div className="detail-body stack">
        <div className="metric-grid">
          {metric("winner objective", generation.winner.metrics.objective)}
          {metric("winner J", generation.winner.metrics.J)}
          {metric("delta_J", generation.delta_J)}
        </div>
        <div>
          <div className="section-label">Retrieved memory</div>
          <ul className="dense-list">
            {generation.retrieved_memories.length ? (
              generation.retrieved_memories.map((memory) => (
                <li key={memory.experience_id}>
                  <strong>{memory.experience_id}</strong>
                  <div className="small">{memory.prompt_fragment || memory.strategy_hypothesis || "No prompt fragment."}</div>
                </li>
              ))
            ) : (
              <li>No retrieved memory.</li>
            )}
          </ul>
        </div>
        <div className="stack">
          {generation.candidates.map((candidate) =>
            candidateCard(
              candidate,
              candidate.candidate_id === generation.winner.candidate_id ? "winner" : "candidate",
            ),
          )}
        </div>
      </div>
    </details>
  );
}

function addedExperienceCard(experience: AddedExperience) {
  const positive = experience.experience_outcome === "success";
  return (
    <article className={`memory-fragment-card ${positive ? "positive" : "negative"}`} key={experience.experience_id}>
      <div className="badge-row">
        <span className={`badge ${positive ? "good" : "warn"}`}>{positive ? "positive" : "negative"}</span>
        <span className="badge">g{experience.generation}</span>
        <span className="badge">{experience.verifier_status}</span>
        <span className="badge">{experience.proposal_model ?? "n/a"}</span>
      </div>
      <p className="memory-fragment-summary">{experience.candidate_summary}</p>
      <p className="small"><strong>prompt</strong> {experience.prompt_fragment}</p>
      <p className="small"><strong>hypothesis</strong> {experience.strategy_hypothesis}</p>
      <p className="small"><strong>delta_J</strong> {experience.delta_J}</p>
    </article>
  );
}

function memoryFragmentsPanel(run: Run) {
  const addedExperiences = run.added_experiences ?? [];
  const positiveExperiences = addedExperiences.filter((experience) => experience.experience_outcome === "success");
  const negativeExperiences = addedExperiences.filter((experience) => experience.experience_outcome === "failure");
  const positiveCount = numeric(run.positive_experiences_added ?? positiveExperiences.length);
  const negativeCount = numeric(run.negative_experiences_added ?? negativeExperiences.length);
  const memoryBefore = run.memory_before_count ?? "n/a";
  const memoryAfter = run.memory_after_count ?? "n/a";

  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">memory fragments</p>
          <h3>Memory fragments written in this run</h3>
          <p className="muted">Positive and negative fragments are grouped separately so you can see both the winning moves and the dead ends that were recorded for reuse.</p>
        </div>
        <div className="badge-row">
          <span className="badge">initial {memoryBefore}</span>
          <span className="badge">after {memoryAfter}</span>
          <span className="badge good">+positive {positiveCount}</span>
          <span className="badge warn">+negative {negativeCount}</span>
        </div>
      </div>

      <div className="metric-grid memory-metric-grid">
        {metric("initial memory", memoryBefore)}
        {metric("memory after run", memoryAfter)}
        {metric("+positive", positiveCount)}
        {metric("+negative", negativeCount)}
      </div>

      <div className="memory-fragment-grid">
        <section className="memory-column positive-column">
          <div className="memory-column-header">
            <strong>Positive fragments</strong>
            <span className="badge good">{positiveCount}</span>
          </div>
          <div className="memory-scroll">
            {positiveExperiences.length ? (
              positiveExperiences.map((experience) => addedExperienceCard(experience))
            ) : (
              <p className="small">No positive fragments were written in this run.</p>
            )}
          </div>
        </section>

        <section className="memory-column negative-column">
          <div className="memory-column-header">
            <strong>Negative fragments</strong>
            <span className="badge warn">{negativeCount}</span>
          </div>
          <div className="memory-scroll">
            {negativeExperiences.length ? (
              negativeExperiences.map((experience) => addedExperienceCard(experience))
            ) : (
              <p className="small">No negative fragments were written in this run.</p>
            )}
          </div>
        </section>
      </div>
    </section>
  );
}

function runOverviewPanels(run: Run) {
  return (
    <div className="split-grid overview-grid">
      <section className="panel chart-card">
        <div className="panel-header">
          <div>
            <p className="eyebrow">run overview</p>
            <h3>Objective and J drift by generation</h3>
          </div>
          <div className="badge-row">
            <span className="badge">{run.task.objective_label}</span>
            <span className="badge">{run.winner.agent}</span>
          </div>
        </div>
        {runOverviewChart(run)}
      </section>

      <section className="panel chart-card">
        <div className="panel-header">
          <div>
            <p className="eyebrow">memory growth</p>
            <h3>Cumulative positive and negative write-backs</h3>
          </div>
          <div className="badge-row">
            <span className="badge good">+positive {run.positive_experiences_added ?? 0}</span>
            <span className="badge warn">+negative {run.negative_experiences_added ?? 0}</span>
          </div>
        </div>
        {memoryGrowthChart(run)}
      </section>
    </div>
  );
}

function runDetail(run: Run | null) {
  if (!run) {
    return (
      <section className="panel empty-state">
        <p className="eyebrow">latest run</p>
        <h2>No completed run yet</h2>
        <p className="muted">Choose a task, pick a model, and start a run. The frontend will poll the backend job and swap the result into the workbench automatically when it finishes.</p>
      </section>
    );
  }

  const manifest = run.handoff_bundle?.manifest;

  return (
    <section className="panel stack">
      <div className="panel-header">
        <div>
          <p className="eyebrow">{run.task.id}</p>
          <h2>{run.task.title}</h2>
          <p className="muted">{run.task.description}</p>
        </div>
        <div className="badge-row">
          <span className="badge">{run.run_mode}</span>
          <span className="badge">{run.active_model}</span>
          <span className="badge">{run.session_id ?? "session n/a"}</span>
          <span className="badge">{run.llm_traces.length} llm traces</span>
        </div>
      </div>

      <div className="split-grid">
        {candidateCard(run.baseline, "candidate")}
        {candidateCard(run.winner, "winner")}
      </div>

      <div className="metric-grid">
        {metric("winner objective", run.winner.metrics.objective)}
        {metric("winner J", run.winner.metrics.J)}
        {metric("delta_J", run.delta_J)}
        {metric("function", run.task.function_name)}
        {metric("signature", run.task.function_signature)}
        {metric("source type", run.task.source_type)}
      </div>

      <div className="metric-grid">
        {metric("generated", run.generated_at ?? "n/a")}
        {metric("session", run.session_id ?? "n/a")}
        {metric("winner candidate", run.winner.agent)}
      </div>

      {runOverviewPanels(run)}

      {improvementPanel(run)}

      {memoryFragmentsPanel(run)}

      <div className="panel inset-panel">
        <div className="section-label">Selection reason</div>
        <p className="muted">{run.selection_reason}</p>
      </div>

      <details className="detail-card">
        <summary className="detail-summary">
          <div>
            <strong>Artifacts and ledger</strong>
            <div className="detail-summary-copy">Manifest, trace, llm trace, and prompt-ready memory.</div>
          </div>
        </summary>
        <div className="detail-body stack">
          <div className="artifact-grid">
            {metric("manifest", shortPath(run.handoff_bundle?.manifest_path))}
            {metric("payload", shortPath(manifest?.artifact_paths.payload))}
            {metric("trace", shortPath(manifest?.artifact_paths.trace))}
            {metric("llm trace", shortPath(manifest?.artifact_paths.llm_trace_jsonl))}
            {metric("memory markdown", shortPath(manifest?.artifact_paths.memory_markdown))}
            {metric("report svg", shortPath(manifest?.artifact_paths.report_svg))}
          </div>
          <pre className="code-block compact"><code>{run.memory_markdown}</code></pre>
        </div>
      </details>

      <div className="stack">
        {run.generations.map((generation, index) =>
          generationSection(generation, index === run.generations.length - 1),
        )}
      </div>
    </section>
  );
}

export function App() {
  const [runtimeInfo, setRuntimeInfo] = useState<RuntimeInfo>(emptyRuntime());
  const [payload, setPayload] = useState<Payload>(emptyPayload());
  const [selectedTaskId, setSelectedTaskId] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedRunId, setSelectedRunId] = useState("");
  const [themePreference, setThemePreference] = useState<ThemePreference>("system");
  const [liveJob, setLiveJob] = useState<JobState | null>({
    status: "loading",
    events: [{ phase: "boot", message: "Loading runtime and task catalog." }],
  });
  const [error, setError] = useState<ErrorPayload | null>(null);
  const pollToken = useRef(0);

  const selectedTask = useMemo(
    () => payload.task_catalog.find((task) => task.id === selectedTaskId) ?? payload.task_catalog[0] ?? null,
    [payload.task_catalog, selectedTaskId],
  );

  const selectedRun = useMemo(
    () => payload.runs.find((run) => run.task.id === selectedRunId) ?? payload.runs[0] ?? null,
    [payload.runs, selectedRunId],
  );
  const liveGenerationCards = useMemo(() => summarizeLiveEvents(liveJob?.events ?? []), [liveJob?.events]);

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
        setSelectedTaskId(tasks[0]?.id ?? "");
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
        setSelectedRunId(normalized.runs[0]?.task.id ?? "");
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

  async function runTask(taskId: string | null) {
    const model = selectedModel || runtimeInfo.active_model;
    pollToken.current += 1;
    const token = pollToken.current;
    setError(null);
    setLiveJob({
      status: "running",
      taskId: taskId,
      model,
      events: [{ phase: "queued", message: `Starting ${taskId ?? "full sequence"} with ${model}.` }],
    });

    try {
      const start = await startJob(taskId, model);
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
        setSelectedRunId(taskId ?? normalized.runs[0]?.task.id ?? "");
        setError(null);
      }
    } catch (caught) {
      if (token !== pollToken.current) {
        return;
      }
      setError(asErrorPayload(caught));
      setLiveJob({
        status: "failed",
        taskId: taskId,
        model,
        events: [],
      });
    }
  }

  return (
    <main className="app-shell">
      <section className="topbar">
        <div>
          <p className="eyebrow">autoresearch</p>
          <strong className="topbar-title">LLM-required codegen flywheel</strong>
        </div>
        <div className="theme-toggle" role="tablist" aria-label="Theme mode">
          {themeChoices().map((choice) => (
            <button
              key={choice}
              className={`theme-chip ${themePreference === choice ? "active" : ""}`}
              onClick={() => setThemePreference(choice)}
            >
              {choice}
            </button>
          ))}
        </div>
      </section>

      <section className="hero panel">
        <div>
          <p className="eyebrow">strategy memory + deterministic verifier</p>
          <h1>LLM-required autoresearch around direct code generation.</h1>
          <p className="muted hero-copy">
            Launch a selected task, let the backend outer loop run, and inspect the result through one focused workbench:
            runtime state, live generations, prompt-ready memory, artifacts, and deterministic reports.
          </p>
        </div>
        <div className="hero-side">
          <div className="control-grid">
            <label className="field">
              <span className="field-label">Task</span>
              <select
                className="control"
                value={selectedTask?.id ?? ""}
                onChange={(event) => setSelectedTaskId(event.target.value)}
              >
                {payload.task_catalog.map((task) => (
                  <option key={task.id} value={task.id}>
                    {task.id}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span className="field-label">Model</span>
              <select
                className="control"
                value={selectedModel}
                onChange={(event) => setSelectedModel(event.target.value)}
                disabled={!runtimeInfo.available_models.length}
              >
                {runtimeInfo.available_models.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="button-row">
            <button className="action primary" onClick={() => void runTask(selectedTask?.id ?? null)}>
              Run selected task
            </button>
            <button className="action" onClick={() => void runTask(null)}>
              Run full sequence
            </button>
          </div>
          {selectedTask ? (
            <div className="task-brief">
              <div className="badge-row">
                <span className="badge">{selectedTask.family}</span>
                <span className="badge">{selectedTask.function_name}</span>
                <span className="badge">
                  {selectedTask.generation_budget} x {selectedTask.candidate_budget}
                </span>
              </div>
              <p className="small">{selectedTask.description}</p>
            </div>
          ) : null}
        </div>
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
            <h2>Deterministic verifier and model runtime</h2>
          </div>
          <div className="badge-row">
            <span className="badge">{runtimeInfo.mode}</span>
            <span className="badge">primary {runtimeInfo.primary_model}</span>
            <span className="badge">selected {selectedModel || runtimeInfo.active_model}</span>
          </div>
        </div>
        <div className="metric-grid">
          {metric("cached runs", payload.runs.length)}
          {metric("latest generated", payload.summary.generated_at)}
          {metric("memory size", payload.summary.memory_size_after_run)}
          {metric("write backs", payload.summary.write_backs)}
          {metric("temperature", runtimeInfo.temperature)}
          {metric("max tokens", runtimeInfo.max_tokens)}
        </div>
        <div className="split-grid">
          <div className="panel inset-panel">
            <div className="section-label">Enabled models</div>
            <div className="badge-row">
              {runtimeInfo.available_models.map((model) => (
                <span className="badge" key={model}>
                  {model}
                </span>
              ))}
            </div>
            <p className="small">Mode is intentionally fixed to <code>llm-required</code>. Model choice is dynamic per run, but there is still no automatic fallback.</p>
          </div>
          <div className="panel inset-panel">
            <div className="section-label">Audit</div>
            <ul className="dense-list">
              <li>workspace: {payload.audit.workspace_root}</li>
              <li>session: {payload.audit.session_id ?? "n/a"}</li>
              <li>api base: {runtimeInfo.api_base}</li>
              <li>git commit: {payload.summary.git_commit}</li>
            </ul>
          </div>
        </div>
      </section>

      {liveJob ? (
        <section className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">live run</p>
              <h2>{liveJob.status === "running" ? "Backend job in progress" : liveJob.status === "loading" ? "Bootstrapping workbench" : "Last job state"}</h2>
            </div>
            <div className="badge-row">
              <span className="badge">{liveJob.status}</span>
              <span className="badge">{liveJob.model ?? selectedModel ?? "n/a"}</span>
              <span className="badge">{liveJob.taskId ?? liveJob.task_id ?? "full sequence"}</span>
            </div>
          </div>
          {liveGenerationCards.length ? (
            <div className="live-generation-grid">
              {liveGenerationCards.map((card) => (
                <article className={`live-generation-card ${card.status}`} key={card.generation}>
                  <div className="badge-row">
                    <span className="badge">g{card.generation}</span>
                    <span className={`badge ${card.status === "completed" ? "good" : card.status === "running" ? "warn" : ""}`}>
                      {card.status}
                    </span>
                    <span className="badge">{card.selectedModel ?? "awaiting proposal"}</span>
                  </div>
                  <p className="small">{card.retrievedMemories ?? "Waiting for generation start."}</p>
                  <ul className="dense-list compact-list">
                    {card.candidateMessages.length ? (
                      card.candidateMessages.map((message) => <li key={message}>{message}</li>)
                    ) : (
                      <li>No verified candidates yet.</li>
                    )}
                  </ul>
                  {card.memoryWriteback ? <p className="small">write-back: {card.memoryWriteback}</p> : null}
                  {card.completion ? <p className="small">done: {card.completion}</p> : null}
                </article>
              ))}
            </div>
          ) : null}
          <details className="detail-card">
            <summary className="detail-summary">
              <div>
                <strong>Event log</strong>
                <div className="detail-summary-copy">Raw backend events for this job.</div>
              </div>
            </summary>
            <div className="detail-body">
              <ul className="dense-list">
                {(liveJob.events || []).length ? (
                  liveJob.events.map((event, index) => (
                    <li key={`${event.timestamp ?? "t"}-${index}`}>
                      <strong>{event.phase ?? event.event_type ?? "event"}</strong>
                      <div className="small">
                        {[event.timestamp, event.message, event.candidate].filter(Boolean).join(" · ")}
                      </div>
                    </li>
                  ))
                ) : (
                  <li>No events yet.</li>
                )}
              </ul>
            </div>
          </details>
        </section>
      ) : null}

      {payload.runs.length ? (
        <section className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">run history</p>
              <h2>Focus one run at a time</h2>
            </div>
          </div>
          <div className="run-selector">
            {payload.runs.map((run) => (
              <button
                key={run.task.id}
                className={`selector-chip ${selectedRun?.task.id === run.task.id ? "active" : ""}`}
                onClick={() => setSelectedRunId(run.task.id)}
              >
                <strong>{run.task.id}</strong>
                <span>{run.winner.metrics.objective}</span>
              </button>
            ))}
          </div>
        </section>
      ) : null}

      {runDetail(selectedRun)}

    </main>
  );
}
