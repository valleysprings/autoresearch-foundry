import { useEffect, useMemo, useRef, useState } from "react";

import { loadJob, loadLatestRun, loadRuntime, loadTasks, startJob } from "./api";
import { normalizeErrorPayload, stringifyUnknown } from "./errorPayload";
import { initialTaskId, mergeTaskCatalogs, taskScopedPayload } from "./reportCache";
import type {
  Branch,
  Candidate,
  CandidateTestResult,
  DatasetWarning,
  ItemRun,
  ErrorPayload,
  Generation,
  JobState,
  LiveEvent,
  ObjectiveSpec,
  Payload,
  Run,
  RuntimeInfo,
  SelectionSpec,
  TaskSummary,
} from "./types";

type ThemePreference = "system" | "light" | "dark";

type RetryState = {
  attempt: number;
  maxAttempts: number;
};

type LiveItemCard = {
  itemKey: string;
  itemId: string;
  displayName: string;
  status: "queued" | "running" | "failed" | "completed";
  latestGeneration: number;
  branchCount: number;
  passCount: number;
  failCount: number;
  errorCount: number;
  acceptCount: number;
  memoryDelta: number;
  bestObjective: number | null;
  itemBrief: string | null;
  expectedAnswer: string | null;
  latestResponseOutput: string | null;
  latestResponseStatus: string | null;
  responseOutput: string | null;
  responseStatus: string | null;
  latestMessage: string | null;
  retryLabel: string | null;
  startedAtMs: number | null;
  latestEventAtMs: number | null;
  finishedAtMs: number | null;
};

type LiveTaskCard = {
  taskId: string;
  title: string;
  description: string;
  objectiveLabel: string;
  objectiveUnit: string | null;
  model: string;
  branchingFactor: number;
  generationBudget: number;
  candidateBudget: number;
  itemWorkers: number | null;
  maxItems: number | null;
  selectedItemIds: string[] | null;
  usesMaxItems: boolean;
  defaultMaxItems: number | null;
  scheduledItems: number | null;
  currentBest: string | null;
  status: "queued" | "running" | "failed" | "completed";
  totalItems: number;
  completedItems: number;
  passItems: number;
  acceptedCount: number;
  memoryDelta: number;
  items: LiveItemCard[];
  events: LiveEvent[];
};

type MutableLiveItemCard = LiveItemCard & {
  acceptedKeys: Set<string>;
  branchIds: Set<string>;
  retryStates: Map<string, RetryState>;
};

type MutableLiveTaskCard = Omit<LiveTaskCard, "items" | "totalItems" | "completedItems" | "passItems" | "acceptedCount" | "memoryDelta"> & {
  itemsMap: Map<string, MutableLiveItemCard>;
  acceptedCount: number;
  memoryDelta: number;
};

type TaskGroup = {
  track: string;
  label: string;
  tasks: TaskSummary[];
};

const IDLE_LATEST_RUN_POLL_MS = 15000;
const LIVE_JOB_POLL_MS = 500;
const LIVE_JOB_BACKGROUND_POLL_MS = 1500;
const DEFAULT_FRONTEND_BRANCHING_FACTOR = 1;
const DEFAULT_FRONTEND_GENERATION_BUDGET = 3;
const DEFAULT_FRONTEND_CANDIDATE_BUDGET = 1;
const DEFAULT_FRONTEND_ITEM_WORKERS = 5;

function shortPath(path?: string | null): string {
  return path ? path.replace(/^runs\//, "") : "n/a";
}

function questionPreview(prompt: string | undefined | null, limit = 140): string {
  const text = String(prompt ?? "").replace(/\s+/g, " ").trim();
  if (text.length <= limit) {
    return text || "Prompt preview unavailable.";
  }
  return `${text.slice(0, limit - 3).trimEnd()}...`;
}

function stringValue(value: unknown): string | null {
  if (value == null) {
    return null;
  }
  const text = String(value).trim();
  return text.length ? text : null;
}

function humanizeItemName(name: string | undefined | null, itemId?: string | null): string {
  const base = stringValue(name) ?? stringValue(itemId) ?? "Dataset item";
  const seedMatch = base.match(/^(.*?)(?:\s+seed)\s+(\d+)$/i);
  if (!seedMatch) {
    return base;
  }
  const prefix = seedMatch[1].trim() || "Dataset";
  return `${prefix} Question ${Number(seedMatch[2])}`;
}

function firstTestResult(candidate: Candidate | undefined | null): CandidateTestResult | null {
  const results = candidate?.metrics.test_results;
  if (!Array.isArray(results) || !results.length) {
    return null;
  }
  return results[0] ?? null;
}

function firstTestResultReason(candidate: Candidate | undefined | null): string | null {
  const result = firstTestResult(candidate);
  if (!result || typeof result !== "object") {
    return null;
  }
  return stringValue((result as Record<string, unknown>).reason);
}

function candidateResponseOutput(candidate: Candidate | undefined | null): string | null {
  const result = firstTestResult(candidate);
  return stringValue(result?.actual) ?? stringValue(result?.actual_raw);
}

function liveCandidateOutput(value: string | undefined | null): string | null {
  const text = stringValue(value);
  if (!text) {
    return null;
  }
  return text === "[]" || text === "{}" ? null : text;
}

function candidateDisplayOutput(candidate: Candidate | undefined | null): string | null {
  return candidateResponseOutput(candidate) ?? (firstTestResultReason(candidate) ? "parsing error" : null);
}

function latestAttemptedCandidate(itemRun: ItemRun | undefined | null): Candidate | null {
  if (!itemRun) {
    return null;
  }
  for (let generationIndex = itemRun.generations.length - 1; generationIndex >= 0; generationIndex -= 1) {
    const generation = itemRun.generations[generationIndex];
    const candidates = Array.isArray(generation?.candidates) ? generation.candidates : [];
    for (let candidateIndex = candidates.length - 1; candidateIndex >= 0; candidateIndex -= 1) {
      const candidate = candidates[candidateIndex];
      if (candidateResponseOutput(candidate) || candidateResponseStatus(candidate)) {
        return candidate;
      }
    }
  }
  return null;
}

function candidateResponseStatus(candidate: Candidate | undefined | null): string | null {
  return stringValue(candidate?.metrics.verifier_status) ?? stringValue(candidate?.metrics.status);
}

function verifierTone(status: string | undefined | null): "" | "good" | "warn" {
  const normalized = String(status ?? "").toLowerCase();
  if (normalized === "pass") {
    return "good";
  }
  if (normalized === "fail" || normalized === "error") {
    return "warn";
  }
  return "";
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function pageVisible(): boolean {
  return document.visibilityState === "visible";
}

function numeric(value: string | number | undefined | null): number {
  if (typeof value === "number") {
    return value;
  }
  const parsed = Number(value ?? 0);
  return Number.isFinite(parsed) ? parsed : 0;
}

function parseTimestampMs(value: string | undefined | null): number | null {
  if (!value) {
    return null;
  }
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatElapsedDuration(ms: number | null): string | null {
  if (ms == null || !Number.isFinite(ms)) {
    return null;
  }
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) {
    return `${hours}h ${String(minutes).padStart(2, "0")}m`;
  }
  if (minutes > 0) {
    return `${minutes}m ${String(seconds).padStart(2, "0")}s`;
  }
  return `${seconds}s`;
}

function itemElapsedDuration(item: Pick<LiveItemCard, "status" | "startedAtMs" | "latestEventAtMs" | "finishedAtMs">, nowMs: number): string | null {
  if (item.startedAtMs == null) {
    return null;
  }
  const endAtMs =
    item.status === "running"
      ? Math.max(nowMs, item.latestEventAtMs ?? item.startedAtMs)
      : item.finishedAtMs ?? item.latestEventAtMs;
  if (endAtMs == null) {
    return null;
  }
  return formatElapsedDuration(endAtMs - item.startedAtMs);
}

function statusWithDuration(status: string | null | undefined, duration: string | null): string {
  return duration ? `${status ?? "pending"} ${duration}` : status ?? "pending";
}

function eventBranchKey(event: Pick<LiveEvent, "branch_id" | "generation">, fallback: string): string {
  if (event.branch_id) {
    return event.branch_id;
  }
  if (typeof event.generation === "number") {
    return `g${event.generation}`;
  }
  return fallback;
}

function summarizeRetryStates(retryStates: Map<string, RetryState>): string | null {
  if (!retryStates.size) {
    return null;
  }
  if (retryStates.size > 1) {
    return `retries ${retryStates.size}`;
  }
  const current = retryStates.values().next().value as RetryState | undefined;
  if (!current) {
    return null;
  }
  return `retry ${current.attempt}/${current.maxAttempts}`;
}

function parseExternalConfigInput(
  input: string,
  fallback: Record<string, unknown> | null = null,
): { config: Record<string, unknown> | null; error: string | null } {
  if (!input.trim()) {
    return { config: fallback, error: null };
  }
  try {
    const parsed = JSON.parse(input);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return { config: null, error: "External RUN_CONFIG must be a JSON object." };
    }
    return { config: parsed as Record<string, unknown>, error: null };
  } catch {
    return { config: null, error: "External RUN_CONFIG is not valid JSON." };
  }
}

function parseItemIdsInput(input: string): string[] | null {
  const selected = input
    .replace(/\n/g, ",")
    .split(",")
    .map((token) => token.trim())
    .filter(Boolean);
  return selected.length ? selected : null;
}

function prettyJson(value: unknown): string {
  return JSON.stringify(value ?? {}, null, 2);
}

function inferExternalDefaultMaxItems(config: Record<string, unknown> | null | undefined): number | null {
  if (!config) {
    return null;
  }
  for (const key of ["n_tasks", "cases"]) {
    const value = config[key];
    if (typeof value === "number" && Number.isFinite(value) && value > 0) {
      return Math.max(1, Math.floor(value));
    }
    if (typeof value === "string") {
      const parsed = Number(value);
      if (Number.isFinite(parsed) && parsed > 0) {
        return Math.max(1, Math.floor(parsed));
      }
    }
  }
  for (const key of ["problem_names", "task_names", "tasks"]) {
    const value = config[key];
    if (Array.isArray(value) && value.length) {
      return value.length;
    }
  }
  return null;
}

function average(values: number[]): number | null {
  if (!values.length) {
    return null;
  }
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function firstRoundWinner(run: { generations?: Generation[] | null }): Candidate | null {
  if (!Array.isArray(run.generations) || !run.generations.length) {
    return null;
  }
  return run.generations[0]?.winner ?? null;
}

function improvementRatio(finalValue: string | number | undefined | null, anchorValue: string | number | undefined | null): number | null {
  const anchor = numeric(anchorValue);
  if (Math.abs(anchor) < 1e-9) {
    return null;
  }
  return numeric(finalValue) / anchor;
}

function itemRunImprovementRatio(itemRun: ItemRun): number | null {
  const roundOne = firstRoundWinner(itemRun);
  return roundOne ? improvementRatio(itemRun.winner.metrics.primary_score, roundOne.metrics.primary_score) : null;
}

function runImprovementRatio(run: Run): number | null {
  if (Array.isArray(run.item_runs) && run.item_runs.length) {
    const ratios = run.item_runs
      .map((itemRun) => itemRunImprovementRatio(itemRun))
      .filter((value): value is number => value != null);
    return average(ratios);
  }
  const roundOne = firstRoundWinner(run);
  return roundOne ? improvementRatio(run.winner.metrics.primary_score, roundOne.metrics.primary_score) : null;
}

function formatMultiplier(value: number | null, digits = 4): string {
  return value == null ? "n/a" : `${value.toFixed(digits)}x`;
}

function ratioTone(value: number | null): "" | "good" | "warn" {
  if (value == null) {
    return "";
  }
  if (value > 1.000001) {
    return "good";
  }
  if (value < 0.999999) {
    return "warn";
  }
  return "";
}

function passedCandidate(candidate: Candidate | undefined | null): boolean {
  if (!candidate) {
    return false;
  }
  const status = candidate.metrics.verifier_status ?? candidate.metrics.status;
  if (status === "pass") {
    return true;
  }
  const passedTests = numeric(candidate.metrics.passed_tests);
  const totalTests = numeric(candidate.metrics.total_tests);
  return totalTests > 0 && passedTests === totalTests;
}

function datasetTransitionSummary(run: Run): {
  improved: number;
  regressed: number;
  unchangedPass: number;
  unchangedFail: number;
} {
  const counts = {
    improved: 0,
    regressed: 0,
    unchangedPass: 0,
    unchangedFail: 0,
  };
  for (const itemRun of run.item_runs ?? []) {
    const baselinePassed = passedCandidate(itemRun.baseline);
    const winnerPassed = passedCandidate(itemRun.winner);
    if (!baselinePassed && winnerPassed) {
      counts.improved += 1;
    } else if (baselinePassed && !winnerPassed) {
      counts.regressed += 1;
    } else if (baselinePassed && winnerPassed) {
      counts.unchangedPass += 1;
    } else {
      counts.unchangedFail += 1;
    }
  }
  return counts;
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

function sameMetricValue(left: string | number | undefined | null, right: string | number | undefined | null, epsilon = 1e-9): boolean {
  return Math.abs(numeric(left) - numeric(right)) <= epsilon;
}

function hasTieBreakMetrics(selectionSpec: SelectionSpec): boolean {
  return Boolean(selectionSpec.tie_break_formula) && !selectionSpec.tie_break_formula.toLowerCase().includes("no auxiliary tie-break metrics configured");
}

function replacePrimaryTerms(text: string | undefined | null): string {
  return String(text ?? "")
    .replace(/\bprimary scores\b/gi, "selection scores")
    .replace(/\bprimary score\b/gi, "selection score")
    .replace(/\bprimary_score\b/gi, "selection score");
}

function selectionPipelineLabel(selectionSpec: SelectionSpec): string {
  return hasTieBreakMetrics(selectionSpec) ? "gate -> score -> tie-break" : "gate -> score";
}

function selectionFormula(selectionSpec: SelectionSpec): string {
  const formula = replacePrimaryTerms(selectionSpec.primary_formula);
  if (formula.toLowerCase().includes("selection score = objective_score")) {
    return "selection score = normalized objective score";
  }
  return formula;
}

function parseGenerationBestValue(message?: string | null): number | null {
  const text = String(message ?? "");
  const match = text.match(/:\s*([-+]?\d+(?:\.\d+)?)(?:\s|\(|$)/);
  if (!match) {
    return null;
  }
  return Number(match[1]);
}

function generationSummaryPill(event: LiveEvent, objectiveLabelText: string, objectiveUnit?: string | null): string | null {
  const generation = event.generation;
  const bestValue = parseGenerationBestValue(event.message);
  if (generation == null || bestValue == null || !Number.isFinite(bestValue)) {
    return stringValue(event.message);
  }
  return `best ${objectiveLabelText} ${formatObjectiveValueFromUnit(bestValue, objectiveUnit)} at gen ${generation}`;
}

function formatLiveEventMessage(event: LiveEvent, objectiveLabelText: string, objectiveUnit?: string | null): string | null {
  if (event.phase === "candidate_verified") {
    const metrics = parseCandidateMetrics(event.message);
    const fragments = [`Candidate ${metrics.status ?? "checked"}`];
    if (metrics.objective != null) {
      fragments.push(`${objectiveLabelText} ${formatObjectiveValueFromUnit(metrics.objective, objectiveUnit)}`);
    }
    if (metrics.primaryScore != null && !sameMetricValue(metrics.primaryScore, metrics.objective)) {
      fragments.push(`score ${formatValue(metrics.primaryScore)}`);
    }
    return `${fragments.join(", ")}.`;
  }
  if (event.phase === "generation_finished") {
    const generation = event.generation;
    const bestValue = parseGenerationBestValue(event.message);
    if (generation == null || bestValue == null || !Number.isFinite(bestValue)) {
      return stringValue(event.message);
    }
    if (event.improved_global_best) {
      return `Generation ${generation} found a new best ${objectiveLabelText}: ${formatObjectiveValueFromUnit(bestValue, objectiveUnit)}.`;
    }
    return `Generation ${generation} finished. Best ${objectiveLabelText} remains ${formatObjectiveValueFromUnit(bestValue, objectiveUnit)}.`;
  }
  return stringValue(event.message);
}

function directionCopy(direction: string): string {
  return direction === "min" ? "Lower score wins" : "Higher score wins";
}

function themeChoiceLabel(choice: ThemePreference): string {
  return choice.charAt(0).toUpperCase() + choice.slice(1);
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
    llm_concurrency: "n/a",
  };
}

function emptySelectionSpec(): SelectionSpec {
  return {
    display_name: "Layered selection policy",
    summary_template: "Candidates must satisfy the verifier gate, improve the selection score, and only use tie-break metrics when scores are effectively tied.",
    primary_metric: "objective_score",
    primary_label: "Normalized objective score",
    primary_direction: "max",
    primary_formula: "",
    gate_summary: "",
    tie_break_formula: "",
    delta_template: "",
    archive_summary: "",
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
      flywheel: [],
      proposal_engine: emptyRuntime(),
    },
    formulas: {
      objective: "",
      primary_score: "",
      tie_break_score: "",
      delta_primary_score: "",
      run_delta_primary_score: "",
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
  const taskCatalog = mergeTaskCatalogs(fallbackCatalog, payload.task_catalog);
  const activeTaskIds = new Set(taskCatalog.map((task) => task.id));
  return {
    ...payload,
    task_catalog: taskCatalog,
    runs: Array.isArray(payload.runs)
      ? payload.runs
          .filter((run) => activeTaskIds.has(String(run.task?.id ?? "")))
          .map((run) => {
            const catalogTask = taskCatalog.find((task) => task.id === run.task?.id);
            const selectionSpec =
              run.selection_spec ?? run.task?.selection_spec ?? catalogTask?.selection_spec ?? emptySelectionSpec();
            return {
              ...run,
              selection_spec: selectionSpec,
              task: {
                ...run.task,
                selection_spec: run.task?.selection_spec ?? selectionSpec,
              },
            };
          })
      : [],
  };
}

function scopedPayloadOrEmpty(payload: Payload, taskId: string | null | undefined, fallbackCatalog: TaskSummary[]): Payload {
  const scoped = taskScopedPayload(payload, taskId);
  return scoped ? scoped : emptyPayload(fallbackCatalog);
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
  return spec.display_name || "Benchmark objective";
}

function benchmarkTierLabel(includedInMainComparison: boolean): string {
  return includedInMainComparison ? "benchmark task" : "auxiliary task";
}

function trackLabel(track: string): string {
  const labels: Record<string, string> = {
    math_verified: "Mathematics",
    reasoning_verified: "Reasoning",
    longcontext_verified: "Long Context",
    browse_snapshot: "Browse",
    science_verified: "Science Reasoning",
    agent_verified: "Agent Benchmarks",
    coding_verified: "Coding",
    or_verified: "Operations Research",
  };
  return labels[track] ?? track.replace(/_/g, " ");
}

function taskModeLabel(mode: string | undefined | null): string {
  const normalized = String(mode ?? "").trim().toLowerCase();
  if (normalized === "answer") {
    return "mode answer";
  }
  if (normalized === "artifact") {
    return "mode artifact";
  }
  if (normalized === "agent") {
    return "mode agent";
  }
  return "mode unknown";
}

function optimizationScopeLabel(scope: string | undefined | null): string {
  const normalized = String(scope ?? "").trim().toLowerCase();
  if (normalized === "prompt") {
    return "scope prompt";
  }
  if (normalized === "wrapper") {
    return "scope wrapper";
  }
  if (normalized === "implementation") {
    return "scope implementation";
  }
  return "scope unknown";
}

function groupTasksByTrack(tasks: TaskSummary[]): TaskGroup[] {
  const groups = new Map<string, TaskGroup>();
  for (const task of tasks) {
    const track = task.track;
    const existing = groups.get(track);
    if (existing) {
      existing.tasks.push(task);
      continue;
    }
    groups.set(track, {
      track,
      label: trackLabel(track),
      tasks: [task],
    });
  }
  return [...groups.values()];
}

function datasetIntroCopy(task: TaskSummary): string {
  const parts = [task.description];
  if (typeof task.dataset_size === "number" && task.dataset_size > 0) {
    parts.push(`Local mirror: ${task.dataset_size} items.`);
  }
  if (task.split) {
    parts.push(`Split: ${task.split}.`);
  }
  parts.push(
    task.included_in_main_comparison
      ? "Included in the benchmark task set used for direct task runs."
      : "Reserved for targeted or auxiliary runs, outside the active benchmark task set.",
  );
  parts.push("A dataset run opens one item at a time, evolves code against that prompt, verifies locally, and then aggregates item outcomes into one report.");
  return parts.join(" ");
}

function parseCandidateMetrics(message?: string | null): {
  status: string | null;
  objective: number | null;
  primaryScore: number | null;
} {
  const text = String(message ?? "");
  const statusMatch = text.match(/status=([a-z]+)/i);
  const objectiveMatch = text.match(/objective=([-+]?\d+(?:\.\d+)?)/i);
  const primaryScoreMatch = text.match(/primary_score=([-+]?\d+(?:\.\d+)?)/i);
  return {
    status: statusMatch ? statusMatch[1].toLowerCase() : null,
    objective: objectiveMatch ? Number(objectiveMatch[1]) : null,
    primaryScore: primaryScoreMatch ? Number(primaryScoreMatch[1]) : null,
  };
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function isRatioUnit(unit?: string | null): boolean {
  return String(unit ?? "").trim().toLowerCase() === "ratio";
}

function formatObjectiveValueFromUnit(value: string | number | undefined | null, unit?: string | null): string {
  if (isRatioUnit(unit)) {
    return formatPercent(numeric(value));
  }
  return formatValue(value, unit ? ` ${unit}` : "");
}

function formatObjectiveValue(value: string | number | undefined | null, spec: ObjectiveSpec): string {
  return formatObjectiveValueFromUnit(value, spec.unit);
}

function formatPercentagePointDelta(value: string | number | undefined | null, digits = 1): string {
  const amount = numeric(value) * 100;
  return `${amount >= 0 ? "+" : ""}${amount.toFixed(digits)} pp`;
}

function formatObjectiveDelta(value: string | number | undefined | null, spec: ObjectiveSpec): string {
  if (isRatioUnit(spec.unit)) {
    return formatPercentagePointDelta(value);
  }
  return `${formatSigned(value, 4)}${spec.unit ? ` ${spec.unit}` : ""}`;
}

function formatSolvedFraction(solved: number | string | undefined | null, total: number | string | undefined | null): string {
  const solvedCount = numeric(solved);
  const totalCount = numeric(total);
  if (totalCount <= 0) {
    return `${solvedCount}/${totalCount}`;
  }
  return `${solvedCount}/${totalCount} (${formatPercent(solvedCount / totalCount)})`;
}

function summarizeLiveTasks(
  events: LiveEvent[],
  taskCatalog: TaskSummary[],
  liveJob: JobState | null,
  runs: Run[],
): LiveTaskCard[] {
  const taskMap = new Map<string, MutableLiveTaskCard>();

  function completedRunForTask(taskId: string): Run | undefined {
    return runs.find((run) => run.task.id === taskId);
  }

  function getTask(taskId: string): MutableLiveTaskCard {
    const catalogTask = taskCatalog.find((task) => task.id === taskId);
    const completedRun = completedRunForTask(taskId);
    const existing = taskMap.get(taskId);
    if (existing) {
      return existing;
    }
    const usesMaxItems = Boolean(catalogTask?.supports_max_items ?? catalogTask?.local_dataset_only ?? completedRun?.task.local_dataset_only);
    const datasetSize = catalogTask?.dataset_size ?? completedRun?.task.dataset_size ?? null;
    const defaultMaxItems = catalogTask?.default_max_items ?? completedRun?.task.default_max_items ?? null;
    const requestedItems = liveJob?.max_items ?? null;
    const selectedItemIds = Array.isArray(liveJob?.item_ids) && liveJob?.item_ids.length ? liveJob.item_ids : null;
    const scheduledItems = selectedItemIds
      ? selectedItemIds.length
      : usesMaxItems
      ? typeof requestedItems === "number" && requestedItems > 0
        ? typeof datasetSize === "number" && datasetSize > 0
          ? Math.min(requestedItems, datasetSize)
          : requestedItems
        : typeof datasetSize === "number" && datasetSize > 0
          ? datasetSize
          : defaultMaxItems
      : 1;
    const entry: LiveTaskCard = {
      taskId,
      title: catalogTask?.title ?? completedRun?.task.title ?? taskId,
      description: catalogTask?.description ?? completedRun?.task.description ?? "Benchmark description unavailable.",
      objectiveLabel: objectiveLabel(catalogTask?.objective_spec ?? completedRun?.task.objective_spec ?? { display_name: "Benchmark objective", direction: "max", summary_template: "", formula: "" }),
      objectiveUnit: catalogTask?.objective_spec?.unit ?? completedRun?.task.objective_spec?.unit ?? null,
      model: liveJob?.model ?? completedRun?.active_model ?? "n/a",
      branchingFactor:
        liveJob?.branching_factor ?? catalogTask?.branching_factor ?? completedRun?.task.branching_factor ?? 1,
      generationBudget:
        liveJob?.generation_budget ?? catalogTask?.generation_budget ?? completedRun?.task.generation_budget ?? 0,
      candidateBudget:
        liveJob?.candidate_budget ?? catalogTask?.candidate_budget ?? completedRun?.task.candidate_budget ?? 0,
      itemWorkers: liveJob?.item_workers ?? catalogTask?.item_workers ?? completedRun?.task.item_workers ?? null,
      maxItems: liveJob?.max_items ?? null,
      selectedItemIds,
      usesMaxItems,
      defaultMaxItems,
      scheduledItems,
      currentBest: null,
      status:
        liveJob?.status === "completed"
          ? "completed"
          : liveJob?.status === "failed"
            ? "failed"
            : liveJob?.status === "running"
              ? "running"
              : "queued",
      totalItems: 0,
      completedItems: 0,
      passItems: 0,
      acceptedCount: 0,
      memoryDelta: 0,
      items: [],
      events: [],
    };
    const mutableEntry: MutableLiveTaskCard = {
      ...entry,
      itemsMap: new Map<string, MutableLiveItemCard>(),
    };
    taskMap.set(taskId, mutableEntry);
    return mutableEntry;
  }

  function getItem(task: MutableLiveTaskCard, event: LiveEvent): MutableLiveItemCard | null {
    if (!event.item_id && task.usesMaxItems) {
      return null;
    }
    const itemKey = event.item_id ?? task.taskId;
    const existing = task.itemsMap.get(itemKey);
    if (existing) {
      return existing;
    }
    const item: MutableLiveItemCard = {
      itemKey,
      itemId: event.item_id ?? task.taskId,
      displayName: humanizeItemName(event.item_name ?? event.item_id ?? task.title, itemKey),
      status: "queued",
      latestGeneration: 0,
      branchCount: 0,
      passCount: 0,
      failCount: 0,
      errorCount: 0,
      acceptCount: 0,
      memoryDelta: 0,
      bestObjective: null,
      itemBrief: event.item_brief ?? null,
      expectedAnswer: event.expected_answer ?? null,
      latestResponseOutput: liveCandidateOutput(event.candidate_actual),
      latestResponseStatus: event.candidate_status ?? null,
      responseOutput: null,
      responseStatus: null,
      latestMessage: null,
      retryLabel: null,
      startedAtMs: null,
      latestEventAtMs: null,
      finishedAtMs: null,
      acceptedKeys: new Set<string>(),
      branchIds: new Set<string>(),
      retryStates: new Map<string, RetryState>(),
    };
    task.itemsMap.set(itemKey, item);
    return item;
  }

  for (const event of events) {
    const taskId = event.task_id ?? liveJob?.task_id ?? liveJob?.taskId ?? null;
    if (!taskId) {
      continue;
    }
    const task = getTask(taskId);
    task.events.push(event);
    const displayMessage = formatLiveEventMessage(event, task.objectiveLabel, task.objectiveUnit);
    const item = getItem(task, event);
    const eventTimestampMs = parseTimestampMs(event.timestamp);
    const isGenerationSummary = event.phase === "generation_finished";
    const eventResponseOutput = liveCandidateOutput(event.candidate_actual);
    if (item) {
      item.displayName = humanizeItemName(event.item_name ?? item.displayName, item.itemId);
      item.itemBrief = event.item_brief ?? item.itemBrief;
      item.expectedAnswer = event.expected_answer ?? item.expectedAnswer;
      if (!isGenerationSummary) {
        item.latestResponseOutput = eventResponseOutput ?? item.latestResponseOutput;
        item.latestResponseStatus = event.candidate_status ?? item.latestResponseStatus;
      }
      item.latestMessage = displayMessage ?? item.latestMessage;
      if (eventTimestampMs != null) {
        item.startedAtMs = item.startedAtMs == null ? eventTimestampMs : Math.min(item.startedAtMs, eventTimestampMs);
        item.latestEventAtMs = item.latestEventAtMs == null ? eventTimestampMs : Math.max(item.latestEventAtMs, eventTimestampMs);
      }
      if (typeof event.generation === "number") {
        item.latestGeneration = Math.max(item.latestGeneration, event.generation);
      }
      if (event.branch_id) {
        item.branchIds.add(event.branch_id);
        item.branchCount = item.branchIds.size;
      }
    }
    if (event.phase === "generation_started" || event.phase === "branch_started" || event.phase === "proposal_generated") {
      task.status = "running";
      if (item) {
        item.status = "running";
        if (event.phase === "branch_started" || event.phase === "proposal_generated") {
          item.retryStates.delete(eventBranchKey(event, item.itemKey));
          item.retryLabel = summarizeRetryStates(item.retryStates);
        }
      }
    }
    if (event.phase === "llm_retry" && item) {
      if (typeof event.retry_attempt === "number" && typeof event.max_attempts === "number") {
        item.retryStates.set(eventBranchKey(event, item.itemKey), { attempt: event.retry_attempt, maxAttempts: event.max_attempts });
        item.retryLabel = summarizeRetryStates(item.retryStates);
      }
    }
    if (event.phase === "candidate_verified" && item) {
      const metrics = parseCandidateMetrics(event.message);
      item.retryStates.delete(eventBranchKey(event, item.itemKey));
      item.retryLabel = summarizeRetryStates(item.retryStates);
      if (metrics.status === "pass") {
        item.passCount += 1;
      } else if (metrics.status === "fail") {
        item.failCount += 1;
      } else if (metrics.status === "error") {
        item.errorCount += 1;
      }
      if (typeof metrics.objective === "number" && Number.isFinite(metrics.objective)) {
        item.bestObjective = item.bestObjective == null ? metrics.objective : Math.max(item.bestObjective, metrics.objective);
      }
      if (metrics.status) {
        item.latestResponseStatus = metrics.status;
        if (!eventResponseOutput && metrics.status !== "pass") {
          item.latestResponseOutput = "parsing error";
        }
      }
    }
    if (event.phase === "generation_finished" && item) {
      item.responseOutput =
        eventResponseOutput
        ?? item.latestResponseOutput
        ?? ((item.latestResponseStatus && item.latestResponseStatus !== "pass") ? "parsing error" : null);
      item.responseStatus = event.candidate_status ?? item.latestResponseStatus ?? item.responseStatus;
      item.retryStates.clear();
      item.retryLabel = null;
    }
    if (event.accepted_to_frontier) {
      if (item) {
        const acceptedKey =
          event.branch_id ?? (typeof event.generation === "number" ? `g${event.generation}` : `${event.phase ?? "event"}-${task.events.length}`);
        item.acceptedKeys.add(acceptedKey);
        item.acceptCount = item.acceptedKeys.size;
      }
    }
    if (item) {
      item.memoryDelta += event.memory_delta ?? 0;
    }
    task.memoryDelta += event.memory_delta ?? 0;
    if (event.phase === "generation_finished") {
      task.currentBest = generationSummaryPill(event, task.objectiveLabel, task.objectiveUnit) ?? task.currentBest;
      if (item && item.latestGeneration >= task.generationBudget) {
        item.status = "completed";
        item.finishedAtMs = eventTimestampMs ?? item.latestEventAtMs ?? item.finishedAtMs;
      }
    }
  }

  return [...taskMap.values()]
    .map((task) => {
      const completedRun = completedRunForTask(task.taskId);
      const completedItemRuns = new Map((completedRun?.item_runs ?? []).map((itemRun) => [itemRun.item_id, itemRun]));
      const itemKeys = new Set<string>([...task.itemsMap.keys(), ...completedItemRuns.keys()]);
      const items = [...itemKeys]
        .map((itemKey) => {
          const item = task.itemsMap.get(itemKey);
          const completedItemRun = completedItemRuns.get(itemKey);
          const latestAttempt = latestAttemptedCandidate(completedItemRun);
          const latestResponseOutput =
            item?.latestResponseOutput
            ?? candidateDisplayOutput(latestAttempt)
            ?? null;
          const latestResponseStatus =
            item?.latestResponseStatus
            ?? candidateResponseStatus(latestAttempt)
            ?? null;
          const winnerResponseOutput = candidateDisplayOutput(completedItemRun?.winner);
          const fallbackResponseOutput =
            item?.responseOutput
            ?? candidateDisplayOutput(latestAttempt)
            ?? null;
          const responseOutput = winnerResponseOutput ?? fallbackResponseOutput;
          const winnerResponseStatus = candidateResponseStatus(completedItemRun?.winner);
          const fallbackResponseStatus =
            item?.responseStatus
            ?? candidateResponseStatus(latestAttempt)
            ?? null;
          const responseStatus =
            winnerResponseOutput != null
              ? (winnerResponseStatus ?? fallbackResponseStatus)
              : (fallbackResponseStatus ?? winnerResponseStatus);
          const winnerPassed = passedCandidate(completedItemRun?.winner);
          const winnerStatus = completedItemRun?.winner.metrics.verifier_status ?? completedItemRun?.winner.metrics.status ?? null;
          const latestGeneration = item?.latestGeneration ?? completedItemRun?.generations.length ?? 0;
          const itemStatus =
            liveJob?.status === "completed" || completedItemRun || latestGeneration >= task.generationBudget
              ? "completed"
              : liveJob?.status === "failed" && item
                ? "failed"
              : item?.status ?? "queued";
          const finishedAtMs =
            itemStatus === "running"
              ? null
              : item?.finishedAtMs ?? item?.latestEventAtMs ?? null;
          return {
            itemKey,
            itemId: completedItemRun?.item_id ?? item?.itemId ?? itemKey,
            displayName: humanizeItemName(completedItemRun?.item_name ?? item?.displayName ?? itemKey, itemKey),
            status: itemStatus,
            latestGeneration,
            branchCount: item?.branchIds.size ?? 0,
            passCount: item?.passCount ?? (winnerPassed ? 1 : 0),
            failCount: item?.failCount ?? (completedItemRun && winnerStatus === "fail" ? 1 : 0),
            errorCount: item?.errorCount ?? (completedItemRun && winnerStatus === "error" ? 1 : 0),
            acceptCount: item?.acceptCount ?? 0,
            memoryDelta: item?.memoryDelta ?? 0,
            bestObjective: item?.bestObjective ?? (completedItemRun ? numeric(completedItemRun.winner.metrics.objective) : null),
            itemBrief: completedItemRun ? questionPreview(completedItemRun.question.prompt, 240) : item?.itemBrief ?? null,
            expectedAnswer: completedItemRun ? String(completedItemRun.question.expected_answer) : item?.expectedAnswer ?? null,
            latestResponseOutput,
            latestResponseStatus,
            responseOutput,
            responseStatus,
            latestMessage: item?.latestMessage ?? completedItemRun?.selection_reason ?? null,
            retryLabel: item?.retryLabel ?? null,
            startedAtMs: item?.startedAtMs ?? null,
            latestEventAtMs: item?.latestEventAtMs ?? null,
            finishedAtMs,
          };
        })
        .sort((left, right) => left.itemId.localeCompare(right.itemId));
      const totalItems = Math.max(items.length, task.scheduledItems ?? items.length);
      return {
        ...task,
        status:
          liveJob?.status === "completed"
            ? "completed"
            : liveJob?.status === "failed"
              ? "failed"
              : task.status,
        acceptedCount: items.reduce((count, item) => count + item.acceptCount, 0),
        totalItems,
        completedItems: items.filter((item) => item.status === "completed").length,
        passItems: items.filter((item) => item.responseStatus === "pass").length,
        items,
      };
    })
    .sort((left, right) => left.taskId.localeCompare(right.taskId));
}

function statusTone(status: "loading" | "queued" | "running" | "failed" | "completed" | null | undefined): "" | "good" | "warn" | "bad" {
  if (status === "completed") {
    return "good";
  }
  if (status === "running") {
    return "warn";
  }
  if (status === "failed") {
    return "bad";
  }
  return "";
}

function deltaChart(run: Run) {
  const points = run.objective_curve ?? [];
  if (!points.length) {
    return null;
  }
  const spec = run.task.objective_spec;
  const label = objectiveLabel(spec);
  const anchorPoint = points.find((point) => numeric(point.generation) > 0) ?? points[0];
  const anchorLabel = numeric(anchorPoint.generation) > 0 ? "round 1" : "checked-in baseline";
  const anchorObjective = numeric(anchorPoint.objective);
  const anchorPrimaryScore = numeric(anchorPoint.primary_score);
  const chartPoints = points.map((point, index) => ({
    generation: point.generation,
    index,
    objectiveDelta: numeric(point.objective) - anchorObjective,
    primaryScoreDelta: numeric(point.primary_score) - anchorPrimaryScore,
    acceptedCount: numeric(point.accepted_count ?? 0),
  }));
  const showSelectionScoreTrace = chartPoints.some((point) => !sameMetricValue(point.objectiveDelta, point.primaryScoreDelta));
  const width = 700;
  const height = 260;
  const padding = 26;
  const plotWidth = width - padding * 2;
  const plotHeight = height - padding * 2;
  const values = chartPoints.flatMap((point) => [point.objectiveDelta, point.primaryScoreDelta, 0]);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const x = (index: number) => padding + (chartPoints.length === 1 ? plotWidth / 2 : (plotWidth * index) / (chartPoints.length - 1));
  const y = (value: number) => padding + plotHeight - ((value - min) / range) * plotHeight;
  const pathFor = (key: "objectiveDelta" | "primaryScoreDelta") =>
    chartPoints.map((point, index) => `${index === 0 ? "M" : "L"} ${x(point.index)} ${y(point[key])}`).join(" ");

  return (
    <div className="chart-wrap">
      <svg className="chart-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Generation score trace">
        {[0.2, 0.4, 0.6, 0.8].map((ratio) => {
          const lineY = padding + plotHeight * ratio;
          return <line key={ratio} className="chart-grid-line" x1={padding} x2={width - padding} y1={lineY} y2={lineY} />;
        })}
        <line className="chart-axis" x1={padding} x2={width - padding} y1={y(0)} y2={y(0)} />
        <path className="chart-line objective-line" d={pathFor("objectiveDelta")} />
        {showSelectionScoreTrace ? <path className="chart-line primary-line" d={pathFor("primaryScoreDelta")} /> : null}
        {chartPoints.map((point) => (
          <g key={`chart-${run.task.id}-${point.generation}`}>
            <circle className={`chart-point ${point.acceptedCount > 0 ? "accepted" : "candidate"}`} cx={x(point.index)} cy={y(point.objectiveDelta)} r="4.6" />
            {showSelectionScoreTrace ? (
              <circle className={`chart-point primary-point ${point.acceptedCount > 0 ? "accepted" : "candidate"}`} cx={x(point.index)} cy={y(point.primaryScoreDelta)} r="4.6" />
            ) : null}
            <text className="chart-label" x={x(point.index)} y={height - 8} textAnchor="middle">
              g{point.generation}
            </text>
          </g>
        ))}
      </svg>
      <div className="legend">
        <span className="legend-item">
          <span className="legend-swatch objective-line-swatch" />
          {label} vs {anchorLabel}
        </span>
        {showSelectionScoreTrace ? (
          <span className="legend-item">
            <span className="legend-swatch primary-line-swatch" />
            selection score vs {anchorLabel}
          </span>
        ) : null}
        <span className="legend-item">
          <span className="legend-swatch accepted-swatch" />
          frontier accepts
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
      <svg className="chart-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Memory write trace">
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
          positive memory writes
        </span>
        <span className="legend-item">
          <span className="legend-swatch negative-swatch" />
          negative memory writes
        </span>
      </div>
    </div>
  );
}

function metricTemplate(spec: ObjectiveSpec, selectionSpec: SelectionSpec) {
  return (
    <section className="subpanel">
      <div className="subpanel-header">
        <p className="eyebrow">selection rules</p>
        <div className="badge-row">
          <span className="badge">{directionCopy(spec.direction)}</span>
          <span className="badge">{selectionPipelineLabel(selectionSpec)}</span>
        </div>
      </div>
      <div className="template-grid">
        <article className="template-card">
          <div className="template-label">{objectiveLabel(spec)}</div>
          <p className="small">{spec.summary_template}</p>
          <p className="template-formula">{spec.formula}</p>
        </article>
        <article className="template-card">
          <div className="template-label">{selectionSpec.display_name}</div>
          <p className="small">{replacePrimaryTerms(selectionSpec.summary_template)}</p>
          <p className="template-formula">{selectionSpec.gate_summary}</p>
          <p className="template-formula">{selectionFormula(selectionSpec)}</p>
          <p className="template-formula">{selectionSpec.tie_break_formula}</p>
          <p className="small">{selectionSpec.delta_template}</p>
          <p className="small">{selectionSpec.archive_summary}</p>
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
          <span className={`badge ${tone === "winner" ? "good" : ""}`}>{tone === "winner" ? "selected" : "candidate"}</span>
          <span className="badge">{candidate.proposal_model ?? "baseline"}</span>
          <span className="badge">{candidate.metrics.verifier_status ?? "n/a"}</span>
        </div>
      </summary>
      <div className="detail-body">
        <div className="metric-grid compact-metrics">
          {metric(objectiveLabel(objectiveSpec), formatObjectiveValue(candidate.metrics.objective, objectiveSpec))}
          {metric("normalized score", formatValue(candidate.metrics.objective_score))}
          {metric("tie-break score", formatValue(candidate.metrics.tie_break_score))}
          {metric("verifier time", candidate.metrics.benchmark_ms == null ? "n/a" : `${candidate.metrics.benchmark_ms} ms`)}
          {metric("tests passed", `${candidate.metrics.passed_tests ?? "n/a"}/${candidate.metrics.total_tests ?? "n/a"}`)}
          {metric("workspace path", shortPath(candidate.workspace_path))}
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
            seeded from {branch.parent_candidate.agent} {"->"} {branch.winner.label}
          </div>
        </div>
        <div className="badge-row">
          <span className={`badge ${branch.winner_accepted ? "good" : "warn"}`}>{branch.winner_accepted ? "accepted" : "rejected"}</span>
          <span className={`badge ${branch.memory_delta > 0 ? "good" : branch.memory_delta < 0 ? "warn" : ""}`}>
            memory {branch.memory_delta > 0 ? `+${branch.memory_delta}` : branch.memory_delta}
          </span>
          <span className="badge">evolve metric {formatSigned(branch.delta_primary_score, 4)}</span>
        </div>
      </summary>
      <div className="detail-body stack">
        <div className="metric-grid compact-metrics">
          {metric("parent agent", branch.parent_candidate.agent)}
          {metric(objectiveLabel(objectiveSpec), formatObjectiveValue(branch.winner.metrics.objective, objectiveSpec))}
          {metric("winner score", formatValue(branch.winner.metrics.primary_score))}
          {metric("winner tie-break", formatValue(branch.winner.metrics.tie_break_score))}
          {metric("beats run best", String(Boolean(branch.winner_improved_global_best)))}
        </div>
        {branch.rejection_reason ? <p className="small muted">{branch.rejection_reason}</p> : null}
        <div>
          <div className="section-label">Retrieved memories</div>
          <ul className="dense-list">
            {branch.retrieved_memories.length ? (
              branch.retrieved_memories.map((memory) => (
                <li key={memory.experience_id}>
                  <strong>{memory.experience_id}</strong>
                  <div className="small">{memory.prompt_fragment || memory.strategy_hypothesis || "No summary available."}</div>
                </li>
              ))
            ) : (
              <li>No retrieved memories.</li>
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
          <span className={`badge ${generation.accepted_count ? "good" : "warn"}`}>{generation.accepted_count ?? 0} frontier accepts</span>
          <span className={`badge ${numeric(generation.memory_delta) > 0 ? "good" : numeric(generation.memory_delta) < 0 ? "warn" : ""}`}>
            memory {numeric(generation.memory_delta) > 0 ? `+${generation.memory_delta}` : generation.memory_delta ?? 0}
          </span>
          <span className="badge">evolve metric {formatSigned(generation.delta_primary_score, 4)}</span>
        </div>
      </summary>
      <div className="detail-body stack">
        <div className="metric-grid compact-metrics">
          {metric("parent pool", generation.parents?.length ?? generation.branches.length)}
          {metric("selected candidate", generation.winner.label)}
          {metric(objectiveLabel(objectiveSpec), formatObjectiveValue(generation.winner.metrics.objective, objectiveSpec))}
          {metric("winner score", formatValue(generation.winner.metrics.primary_score))}
          {metric("winner tie-break", formatValue(generation.winner.metrics.tie_break_score))}
          {metric("positive writes", generation.positive_writebacks ?? 0)}
          {metric("negative writes", generation.negative_writebacks ?? 0)}
        </div>
        <div className="stack">
          {generation.branches.map((branch, index) => branchCard(branch, objectiveSpec, index === 0))}
        </div>
      </div>
    </details>
  );
}

function liveTaskSection(task: LiveTaskCard, nowMs: number) {
  const completedRatio = task.totalItems ? task.completedItems / task.totalItems : 0;
  const passRatio = task.totalItems ? task.passItems / task.totalItems : 0;
  const itemScopeSummary = task.selectedItemIds?.length
    ? task.selectedItemIds.length <= 3
      ? `items ${task.selectedItemIds.join(",")}`
      : `selected items ${task.selectedItemIds.length}`
    : task.usesMaxItems
      ? task.maxItems
        ? `item cap ${task.maxItems}`
        : task.defaultMaxItems
          ? `item cap default ${task.defaultMaxItems}`
          : "item cap default"
      : "item cap off";
  return (
    <article className="task-card live-task-card" key={task.taskId}>
      <div className="panel-header">
        <div>
          <p className="eyebrow">active task</p>
          <h3>{task.title}</h3>
          <p className="muted">{task.description}</p>
        </div>
        <div className="accordion-meta">
          <span className={`badge ${statusTone(task.status)}`}>{task.status}</span>
          <span className="badge">{task.model}</span>
        </div>
      </div>
      <div className="task-summary-row">
        <span className="summary-pill">{task.taskId}</span>
        {task.branchingFactor > 0 ? <span className="summary-pill">frontier parents {task.branchingFactor}</span> : null}
        {task.itemWorkers && task.itemWorkers > 0 ? <span className="summary-pill">workers {task.itemWorkers}</span> : null}
        {task.candidateBudget > 0 ? <span className="summary-pill">candidates/branch {task.candidateBudget}</span> : null}
        {task.generationBudget > 0 ? <span className="summary-pill">max rounds {task.generationBudget}</span> : null}
        <span className="summary-pill">{itemScopeSummary}</span>
        <span className="summary-pill">{task.currentBest ?? "awaiting first selection"}</span>
      </div>
      <div className="metric-grid compact-metrics">
        {metric("items scheduled", task.totalItems)}
        {metric("items complete", `${task.completedItems}/${task.totalItems || "?"}`)}
        {metric("completion", formatPercent(completedRatio))}
        {metric("items solved", `${task.passItems}/${task.totalItems || "?"}`)}
        {metric("solve rate", formatPercent(passRatio))}
        {metric("frontier accepts", task.acceptedCount)}
        {metric("memory delta", task.memoryDelta > 0 ? `+${task.memoryDelta}` : task.memoryDelta)}
      </div>
      <div className="live-scroll">
        {task.items.length ? (
          task.items.map((item) => {
            const responseTone = verifierTone(item.responseStatus ?? item.latestResponseStatus);
            const itemDuration = itemElapsedDuration(item, nowMs);
            const isFinalItem = item.status === "completed" || item.status === "failed";
            const finalResponseDuration = isFinalItem ? itemDuration : null;
            return (
            <article className="live-item-row" key={item.itemKey}>
              <div className="panel-header">
                <div className="live-item-main">
                  <strong>{item.displayName}</strong>
                  <div className="detail-summary-copy live-item-id">{item.itemId}</div>
                </div>
                <div className="badge-row">
                  <span className={`badge ${statusTone(item.status)}`}>{statusWithDuration(item.status, itemDuration)}</span>
                  {task.generationBudget > 0 ? <span className="badge">Generation {item.latestGeneration || 0}/{task.generationBudget || "?"}</span> : null}
                  <span className="badge">branches {item.branchCount}</span>
                  <span className="badge">passes {item.passCount}</span>
                  <span className="badge">fail {item.failCount}</span>
                  <span className="badge">accepted {item.acceptCount}</span>
                  {item.retryLabel ? <span className="badge warn">{item.retryLabel}</span> : null}
                </div>
              </div>
              <div className="split-grid report-grid">
                <section className="subpanel brief-panel">
                  <div className="section-label">Brief</div>
                  <p className="brief-question"><strong>Question.</strong> {item.itemBrief ?? "Question brief is still loading."}</p>
                  <div className="badge-row">
                    <span className="badge">Answer {item.expectedAnswer ?? "n/a"}</span>
                  </div>
                </section>
                <section className={`subpanel response-panel ${responseTone}`}>
                  <div className="section-label">Response</div>
                  <div className="response-stack">
                    <div className="response-entry">
                      <div className="response-caption">Latest Event</div>
                      <div className={`response-value compact ${verifierTone(item.latestResponseStatus)}`}>
                        {item.latestResponseOutput ?? (item.status === "completed" ? "No event output captured." : "Waiting for the latest candidate.")}
                      </div>
                      <div className="badge-row">
                        <span className={`badge ${verifierTone(item.latestResponseStatus)}`}>
                          {statusWithDuration(item.latestResponseStatus, item.latestResponseStatus && item.latestResponseStatus !== "pending" ? finalResponseDuration : null)}
                        </span>
                      </div>
                    </div>
                    <div className="response-entry">
                      <div className="response-caption">Winner</div>
                      <div className={`response-value compact ${verifierTone(item.responseStatus)}`}>
                        {item.responseOutput ?? (item.status === "completed" ? "No winner output captured." : "Waiting for the selected candidate.")}
                      </div>
                      <div className="badge-row">
                        <span className={`badge ${verifierTone(item.responseStatus)}`}>
                          {statusWithDuration(item.responseStatus, item.responseStatus && item.responseStatus !== "pending" ? finalResponseDuration : null)}
                        </span>
                      </div>
                    </div>
                  </div>
                </section>
              </div>
              {item.latestMessage ? <p className="small live-item-message">{item.latestMessage}</p> : null}
            </article>
            );
          })
        ) : (
          <p className="small">Waiting for task events.</p>
        )}
      </div>
    </article>
  );
}

function itemRunCard(itemRun: ItemRun) {
  const displayName = humanizeItemName(itemRun.item_name, itemRun.item_id);
  const latestAttempt = latestAttemptedCandidate(itemRun);
  const winnerResponseOutput = candidateDisplayOutput(itemRun.winner);
  const responseOutput = winnerResponseOutput ?? candidateDisplayOutput(latestAttempt);
  const winnerResponseStatus = candidateResponseStatus(itemRun.winner);
  const responseStatus =
    winnerResponseOutput != null
      ? (winnerResponseStatus ?? candidateResponseStatus(latestAttempt))
      : (candidateResponseStatus(latestAttempt) ?? winnerResponseStatus);
  const baselineStatus = itemRun.baseline.metrics.verifier_status ?? itemRun.baseline.metrics.status ?? "n/a";
  const finalStatus = itemRun.winner.metrics.verifier_status ?? itemRun.winner.metrics.status ?? "n/a";
  return (
    <details className="detail-card generation-card" key={itemRun.item_id}>
      <summary className="detail-summary">
        <div>
          <strong>{displayName}</strong>
          <div className="detail-summary-copy">{questionPreview(itemRun.question.prompt)}</div>
        </div>
        <div className="badge-row">
          <span className={`badge ${finalStatus === "pass" ? "good" : "warn"}`}>
            {finalStatus}
          </span>
          {baselineStatus !== finalStatus ? <span className="badge">{`baseline ${baselineStatus} -> final ${finalStatus}`}</span> : null}
        </div>
      </summary>
      <div className="detail-body stack">
        <div className="split-grid report-grid">
          <section className="subpanel brief-panel">
            <div className="section-label">Brief</div>
            <p className="brief-question"><strong>Question.</strong> {itemRun.question.prompt}</p>
            <div className="badge-row">
              <span className="badge">Answer {String(itemRun.question.expected_answer)}</span>
            </div>
          </section>
          <section className={`subpanel response-panel ${verifierTone(responseStatus)}`}>
            <div className="section-label">Response</div>
            <div className={`response-value ${verifierTone(responseStatus)}`}>
              {responseOutput ?? "No verified output captured."}
            </div>
            <div className="badge-row">
              <span className={`badge ${verifierTone(responseStatus)}`}>{responseStatus ?? "n/a"}</span>
            </div>
          </section>
        </div>
        <div className="metric-grid compact-metrics">
          {metric("item key", itemRun.item_id)}
          {metric("baseline status", baselineStatus)}
          {metric("final status", finalStatus)}
          {metric("generations used", itemRun.generations.length)}
          {metric("memory ledger", `${itemRun.memory_before_count ?? "n/a"} → ${itemRun.memory_after_count ?? "n/a"}`)}
        </div>
        <p className="small">{itemRun.selection_reason}</p>
        {itemRun.question.context ? (
          <pre className="code-block compact"><code>{JSON.stringify(itemRun.question.context, null, 2)}</code></pre>
        ) : null}
        {itemRun.question.choices?.length ? (
          <div className="badge-row">
            {itemRun.question.choices.map((choice) => (
              <span className="badge" key={`${itemRun.item_id}-${choice}`}>
                {choice}
              </span>
            ))}
          </div>
        ) : null}
        <div className="split-grid">
          {candidateCard(itemRun.baseline, objectiveSpec, "candidate")}
          {candidateCard(itemRun.winner, objectiveSpec, "winner")}
        </div>
      </div>
    </details>
  );
}

function runCard(run: Run, defaultSelectionSpec: SelectionSpec, isOpen: boolean, onToggle: () => void) {
  const objectiveSpec = run.task.objective_spec;
  const selectionSpec = run.selection_spec ?? run.task.selection_spec ?? defaultSelectionSpec;
  const isDatasetRun = Array.isArray(run.item_runs) && run.item_runs.length > 0;
  const transitions = isDatasetRun ? datasetTransitionSummary(run) : null;
  const roundOneWinner = firstRoundWinner(run);
  const improvement = runImprovementRatio(run);
  const totalItems = run.dataset_summary?.total_items ?? run.item_runs?.length ?? 0;
  const baselineSolved = run.dataset_summary?.baseline_passed ?? 0;
  const finalSolved = run.dataset_summary?.winner_passed ?? 0;
  const baselineSolveRate = totalItems ? baselineSolved / totalItems : 0;
  const finalSolveRate = totalItems ? finalSolved / totalItems : 0;
  const solveRateGain = finalSolveRate - baselineSolveRate;
  return (
    <article className="task-card completed-card" key={run.task.id}>
      <button className="accordion-toggle" onClick={onToggle} type="button">
        <div className="accordion-copy">
          <p className="eyebrow">cached report</p>
          <h3>{run.task.title}</h3>
          <p className="muted">{run.task.description}</p>
        </div>
        <div className="accordion-meta">
          <span className="badge">{run.active_model}</span>
          <span className="badge">branches {run.task.branching_factor}</span>
          {isDatasetRun ? (
            <>
              <span className="badge">solved {finalSolved}/{totalItems}</span>
              <span className="badge">solve rate {formatPercent(finalSolveRate)}</span>
            </>
          ) : (
            <>
              <span className="badge">
                {objectiveLabel(objectiveSpec)} {formatObjectiveValue(run.winner.metrics.objective, objectiveSpec)}
              </span>
              <span className={`badge ${ratioTone(improvement)}`}>
                improvement {formatMultiplier(improvement)}
              </span>
            </>
          )}
        </div>
      </button>
      <div className="task-summary-row">
        <span className="summary-pill">{run.task.id}</span>
        <span className="summary-pill">{benchmarkTierLabel(run.included_in_main_comparison)}</span>
        <span className="summary-pill">{trackLabel(run.track, run.task.id)}</span>
        <span className="summary-pill">{directionCopy(objectiveSpec.direction)}</span>
        <span className="summary-pill">{run.selection_reason}</span>
      </div>
      {isOpen ? (
        <div className="accordion-body stack">
          {metricTemplate(objectiveSpec, selectionSpec)}

          <div className="metric-grid">
            {isDatasetRun ? (
              <>
                {metric("dataset size", totalItems)}
                {metric("baseline solved", formatSolvedFraction(baselineSolved, totalItems))}
                {metric("final solved", formatSolvedFraction(finalSolved, totalItems))}
                {metric("solve-rate gain", formatPercentagePointDelta(solveRateGain))}
                {metric("new memories", run.added_experiences?.length ?? 0)}
                {metric("memory ledger", `${run.memory_before_count ?? "n/a"} → ${run.memory_after_count ?? "n/a"}`)}
              </>
            ) : (
              <>
                {metric("checked-in objective", formatObjectiveValue(run.baseline.metrics.objective, objectiveSpec))}
                {metric("round 1 objective", roundOneWinner ? formatObjectiveValue(roundOneWinner.metrics.objective, objectiveSpec) : "n/a")}
                {metric("selected objective", formatObjectiveValue(run.winner.metrics.objective, objectiveSpec))}
                {metric("objective delta", formatObjectiveDelta(run.run_delta_objective ?? 0, objectiveSpec))}
                {metric("improvement vs round 1", formatMultiplier(improvement))}
                {metric("generations", run.generations.length)}
                {metric("new memories", run.added_experiences?.length ?? 0)}
                {metric("memory ledger", `${run.memory_before_count ?? "n/a"} → ${run.memory_after_count ?? "n/a"}`)}
              </>
            )}
          </div>

          {isDatasetRun ? (
            <section className="subpanel">
              <div className="subpanel-header">
                <div>
                  <p className="eyebrow">dataset summary</p>
                  <h4>Item-level outcomes</h4>
                </div>
              </div>
              <div className="metric-grid compact-metrics">
                {metric("dataset size", totalItems)}
                {metric("baseline solved", formatSolvedFraction(baselineSolved, totalItems))}
                {metric("final solved", formatSolvedFraction(finalSolved, totalItems))}
                {metric("fail -> pass", transitions?.improved ?? 0)}
                {metric("pass -> fail", transitions?.regressed ?? 0)}
                {metric("solve-rate gain", formatPercentagePointDelta(solveRateGain))}
                {metric("still failing", run.dataset_summary?.failure_count ?? 0)}
              </div>
              <section className="stack">
                {run.item_runs?.map((itemRun) => itemRunCard(itemRun))}
              </section>
            </section>
          ) : (
            <>
              <div className="split-grid">
                {candidateCard(run.baseline, objectiveSpec, "candidate")}
                {candidateCard(run.winner, objectiveSpec, "winner")}
              </div>

              <div className="split-grid report-grid">
                <section className="subpanel">
                  <div className="subpanel-header">
                    <div>
                      <p className="eyebrow">score trace</p>
                      <h4>Generation score trace</h4>
                    </div>
                  </div>
                  {deltaChart(run)}
                </section>
                <section className="subpanel">
                  <div className="subpanel-header">
                    <div>
                      <p className="eyebrow">memory trace</p>
                      <h4>Memory write trace</h4>
                    </div>
                  </div>
                  {memoryDeltaChart(run)}
                </section>
              </div>
            </>
          )}

          {!isDatasetRun ? (
            <section className="subpanel">
              <div className="subpanel-header">
                <div>
                  <p className="eyebrow">strategy memory</p>
                  <h4>Run memory log</h4>
                </div>
              </div>
              <pre className="code-block compact"><code>{run.memory_markdown}</code></pre>
            </section>
          ) : null}

          {!isDatasetRun ? (
            <section className="stack">
              {run.generations.map((generation, index) => generationCard(generation, objectiveSpec, index === run.generations.length - 1))}
            </section>
          ) : null}
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
  const [branchingFactorInput, setBranchingFactorInput] = useState(String(DEFAULT_FRONTEND_BRANCHING_FACTOR));
  const [generationBudgetInput, setGenerationBudgetInput] = useState(String(DEFAULT_FRONTEND_GENERATION_BUDGET));
  const [candidateBudgetInput, setCandidateBudgetInput] = useState(String(DEFAULT_FRONTEND_CANDIDATE_BUDGET));
  const [itemWorkersInput, setItemWorkersInput] = useState(String(DEFAULT_FRONTEND_ITEM_WORKERS));
  const [maxItemsInput, setMaxItemsInput] = useState("");
  const [selectedItemIdsInput, setSelectedItemIdsInput] = useState("");
  const [externalConfigInput, setExternalConfigInput] = useState("");
  const [themePreference, setThemePreference] = useState<ThemePreference>("system");
  const [datasetIntroTaskId, setDatasetIntroTaskId] = useState<string | null>(null);
  const [datasetWarnings, setDatasetWarnings] = useState<DatasetWarning[]>([]);
  const [datasetWarningOpen, setDatasetWarningOpen] = useState(false);
  const [nowMs, setNowMs] = useState(() => Date.now());
  const [liveJob, setLiveJob] = useState<JobState | null>({
    status: "loading",
    events: [{ phase: "boot", message: "Loading runtime config and task registry." }],
  });
  const [error, setError] = useState<ErrorPayload | null>(null);
  const [openCompletedTasks, setOpenCompletedTasks] = useState<Record<string, boolean>>({});
  const pollToken = useRef(0);
  const hydratedTaskId = useRef<string | null>(null);

  const selectedTask = useMemo(
    () => payload.task_catalog.find((task) => task.id === selectedTaskId) ?? payload.task_catalog[0] ?? null,
    [payload.task_catalog, selectedTaskId],
  );

  const taskGroups = useMemo(
    () => groupTasksByTrack(payload.task_catalog),
    [payload.task_catalog],
  );

  const datasetIntroTask = useMemo(
    () => payload.task_catalog.find((task) => task.id === datasetIntroTaskId) ?? null,
    [payload.task_catalog, datasetIntroTaskId],
  );

  const liveTasks = useMemo(
    () => summarizeLiveTasks(liveJob?.events ?? [], payload.task_catalog, liveJob, liveJob?.payload?.runs ?? []),
    [liveJob, payload.task_catalog],
  );

  const selectedTaskRun = useMemo(() => {
    if (!selectedTask) {
      return payload.runs[0] ?? null;
    }
    return payload.runs.find((run) => run.task.id === selectedTask.id) ?? null;
  }, [payload.runs, selectedTask]);

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
    if (!datasetIntroTask && !datasetWarningOpen) {
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setDatasetIntroTaskId(null);
        setDatasetWarningOpen(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [datasetIntroTask, datasetWarningOpen]);

  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      try {
        const [runtime, tasksPayload] = await Promise.all([loadRuntime(), loadTasks()]);
        if (cancelled) {
          return;
        }
        const tasks = tasksPayload.tasks;
        const startupWarnings = Array.isArray(tasksPayload.dataset_warnings) ? tasksPayload.dataset_warnings : [];
        setRuntimeInfo(runtime);
        setSelectedModel(runtime.active_model);
        setDatasetWarnings(startupWarnings);
        setDatasetWarningOpen(startupWarnings.length > 0);
        setPayload(emptyPayload(tasks));
        setLiveJob({
          status: "loading",
          events: [{ phase: "boot", message: "Loading cached reports." }],
        });
        const latest = normalizePayload(await loadLatestRun(), tasks);
        if (cancelled) {
          return;
        }
        const defaultTaskId = initialTaskId(tasks, latest);
        setSelectedTaskId(defaultTaskId);
        const scopedLatest = scopedPayloadOrEmpty(latest, defaultTaskId, tasks);
        if (defaultTaskId && scopedLatest.runs.length) {
          hydratedTaskId.current = defaultTaskId;
          setPayload(scopedLatest);
        } else {
          const taskPayload = defaultTaskId ? normalizePayload(await loadLatestRun(defaultTaskId), tasks) : emptyPayload(tasks);
          if (cancelled) {
            return;
          }
          hydratedTaskId.current = defaultTaskId || null;
          setPayload(scopedPayloadOrEmpty(taskPayload, defaultTaskId, tasks));
        }
        setOpenCompletedTasks({});
        setLiveJob(null);
        setError(null);
      } catch (caught) {
        if (cancelled) {
          return;
        }
        setError(normalizeErrorPayload(caught));
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
    if (!selectedModel) {
      return;
    }
    let cancelled = false;

    async function refreshRuntimeForModel() {
      try {
        const runtime = await loadRuntime(selectedModel);
        if (!cancelled) {
          setRuntimeInfo(runtime);
        }
      } catch {
        if (!cancelled) {
          return;
        }
      }
    }

    void refreshRuntimeForModel();
    return () => {
      cancelled = true;
    };
  }, [selectedModel]);

  useEffect(() => {
    setNowMs(Date.now());
    if (liveJob?.status !== "running") {
      return;
    }
    const timer = window.setInterval(() => setNowMs(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [liveJob?.status]);

  useEffect(() => {
    if (!selectedTask) {
      return;
    }
    setBranchingFactorInput(String(DEFAULT_FRONTEND_BRANCHING_FACTOR));
    setGenerationBudgetInput(String(DEFAULT_FRONTEND_GENERATION_BUDGET));
    setCandidateBudgetInput(String(DEFAULT_FRONTEND_CANDIDATE_BUDGET));
    setItemWorkersInput(selectedTask.runtime_backend === "external" ? "" : String(DEFAULT_FRONTEND_ITEM_WORKERS));
    setMaxItemsInput("");
    setSelectedItemIdsInput("");
    setExternalConfigInput(selectedTask.supports_runtime_config ? prettyJson(selectedTask.external_run_config ?? {}) : "");
  }, [selectedTask?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!selectedTaskId || liveJob?.status === "running" || liveJob?.status === "loading") {
      return;
    }
    if (hydratedTaskId.current === selectedTaskId) {
      hydratedTaskId.current = null;
      return;
    }
    let cancelled = false;
    async function loadSelectedTaskReport() {
      try {
        if (cancelled) {
          return;
        }
        const latest = await loadLatestRun(selectedTaskId);
        setPayload((previous) => {
          const normalized = normalizePayload(latest, previous.task_catalog);
          return scopedPayloadOrEmpty(normalized, selectedTaskId, previous.task_catalog);
        });
      } catch {
        if (cancelled) {
          return;
        }
        setPayload((previous) => emptyPayload(previous.task_catalog));
      }
    }
    void loadSelectedTaskReport();
    return () => {
      cancelled = true;
    };
  }, [selectedTaskId, liveJob?.status]);

  useEffect(() => {
    if (!selectedTaskId || liveJob?.status === "running" || liveJob?.status === "loading") {
      return undefined;
    }

    let cancelled = false;
    let timer: number | null = null;

    const clearTimer = () => {
      if (timer !== null) {
        window.clearTimeout(timer);
        timer = null;
      }
    };

    const schedule = (delayMs: number) => {
      clearTimer();
      timer = window.setTimeout(() => {
        void tick();
      }, delayMs);
    };

    const tick = async () => {
      if (cancelled || !pageVisible()) {
        return;
      }
      try {
        const latest = await loadLatestRun(selectedTaskId);
        if (cancelled) {
          return;
        }
        setPayload((previous) => {
          const normalized = normalizePayload(latest, previous.task_catalog);
          return scopedPayloadOrEmpty(normalized, selectedTaskId, previous.task_catalog);
        });
      } catch {
        return;
      } finally {
        if (!cancelled && pageVisible()) {
          schedule(IDLE_LATEST_RUN_POLL_MS);
        }
      }
    };

    const onVisibilityChange = () => {
      if (cancelled) {
        return;
      }
      if (!pageVisible()) {
        clearTimer();
        return;
      }
      schedule(0);
    };

    if (pageVisible()) {
      schedule(IDLE_LATEST_RUN_POLL_MS);
    }
    document.addEventListener("visibilitychange", onVisibilityChange);

    return () => {
      cancelled = true;
      clearTimer();
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, [liveJob?.status, selectedTaskId]);

  function toggleCompletedTask(taskId: string) {
    setOpenCompletedTasks((previous) => ({ ...previous, [taskId]: !previous[taskId] }));
  }

  async function runTask(taskId: string | null) {
    const model = selectedModel || runtimeInfo.active_model;
    const isExternalTask = selectedTask?.runtime_backend === "external";
    const isDatasetTask = selectedTask?.runtime_backend === "dataset";
    const supportsMaxItems = Boolean(selectedTask?.supports_max_items);
    const branchingFactor = Math.max(
      1,
      Math.floor(numeric(branchingFactorInput || DEFAULT_FRONTEND_BRANCHING_FACTOR)),
    );
    const generationBudget = Math.max(
      1,
      Math.floor(numeric(generationBudgetInput || DEFAULT_FRONTEND_GENERATION_BUDGET)),
    );
    const candidateBudget = Math.max(
      1,
      Math.floor(numeric(candidateBudgetInput || DEFAULT_FRONTEND_CANDIDATE_BUDGET)),
    );
    const itemWorkers = isExternalTask
      ? null
      : Math.max(1, Math.floor(numeric(itemWorkersInput || DEFAULT_FRONTEND_ITEM_WORKERS)));
    const selectedItemIds = isDatasetTask ? parseItemIdsInput(selectedItemIdsInput) : null;
    const maxItems = selectedItemIds ? null : supportsMaxItems && maxItemsInput.trim() ? Math.max(1, Math.floor(numeric(maxItemsInput))) : null;
    const externalConfig = selectedTask?.supports_runtime_config
      ? parseExternalConfigInput(externalConfigInput, selectedTask.external_run_config ?? null)
      : { config: null, error: null };
    if (externalConfig.error) {
      setError({
        terminal: true,
        error_type: "config_error",
        error: externalConfig.error,
        model,
      });
      return;
    }
    pollToken.current += 1;
    const token = pollToken.current;
    setError(null);
    setLiveJob({
      status: "running",
      taskId,
      model,
      branching_factor: branchingFactor,
      generation_budget: generationBudget,
      candidate_budget: candidateBudget,
      item_workers: itemWorkers,
      max_items: maxItems,
      item_ids: selectedItemIds,
      external_config: externalConfig.config,
      events: [
        {
          phase: "queued",
          message: isExternalTask
            ? `Launching ${taskId ?? "selected task"} on ${model} ` +
              `(gen=${generationBudget}, candidates/branch=${candidateBudget}, branches=${branchingFactor}` +
              `${selectedItemIds ? `, items=${selectedItemIds.join(",")}` : maxItems ? `, item cap=${maxItems}` : ""}).`
            : `Launching ${taskId ?? "selected task"} on ${model} ` +
              `(gen=${generationBudget}, candidates/branch=${candidateBudget}, branches=${branchingFactor}, workers=${itemWorkers}` +
              `${selectedItemIds ? `, items=${selectedItemIds.join(",")}` : maxItems ? `, item cap=${maxItems}` : ""}).`,
        },
      ],
    });

    try {
      const start = await startJob(taskId, model, {
        branchingFactor,
        generationBudget,
        candidateBudget,
        itemWorkers,
        maxItems,
        itemIds: selectedItemIds,
        externalConfig: externalConfig.config,
      });
      let job = await loadJob(start.job_id);
      while (job.status === "running" && token === pollToken.current) {
        setLiveJob(job);
        await sleep(pageVisible() ? LIVE_JOB_POLL_MS : LIVE_JOB_BACKGROUND_POLL_MS);
        job = await loadJob(start.job_id);
      }
      if (token !== pollToken.current) {
        return;
      }
      setLiveJob(job);
      if (job.status === "failed") {
        setError(normalizeErrorPayload(job));
        return;
      }
      let completedPayload = job.payload ?? null;
      try {
        completedPayload = await loadLatestRun(taskId ?? undefined);
      } catch {
        completedPayload = job.payload ?? null;
      }
      if (token !== pollToken.current) {
        return;
      }
      if (completedPayload) {
        if (taskId) {
          hydratedTaskId.current = taskId;
        }
        setLiveJob({ ...job, payload: completedPayload });
        setPayload((previous) => {
          const normalized = normalizePayload(completedPayload, previous.task_catalog);
          if (taskId) {
            return scopedPayloadOrEmpty(normalized, taskId, previous.task_catalog);
          }
          return normalized;
        });
        if (taskId) {
          setSelectedTaskId(taskId);
        }
        setError(null);
      }
    } catch (caught) {
      if (token !== pollToken.current) {
        return;
      }
      setError(normalizeErrorPayload(caught));
      setLiveJob({
        status: "failed",
        taskId,
        model,
        branching_factor: Math.max(1, Math.floor(numeric(branchingFactorInput || DEFAULT_FRONTEND_BRANCHING_FACTOR))),
        generation_budget: Math.max(1, Math.floor(numeric(generationBudgetInput || DEFAULT_FRONTEND_GENERATION_BUDGET))),
        candidate_budget: Math.max(1, Math.floor(numeric(candidateBudgetInput || DEFAULT_FRONTEND_CANDIDATE_BUDGET))),
        item_workers: isExternalTask ? null : Math.max(1, Math.floor(numeric(itemWorkersInput || DEFAULT_FRONTEND_ITEM_WORKERS))),
        item_ids: isDatasetTask ? parseItemIdsInput(selectedItemIdsInput) : null,
        events: [],
      });
    }
  }

  const defaultSelectionSpec = selectedTask?.selection_spec ?? payload.task_catalog[0]?.selection_spec ?? emptySelectionSpec();
  const externalConfigDraft = selectedTask?.supports_runtime_config
    ? parseExternalConfigInput(externalConfigInput, selectedTask.external_run_config ?? null)
    : { config: null, error: null };
  const selectedTaskDefaultMaxItems = selectedTask?.local_dataset_only
    ? selectedTask.dataset_size ?? null
    : inferExternalDefaultMaxItems(externalConfigDraft.config ?? selectedTask?.external_run_config ?? null) ?? selectedTask?.default_max_items ?? null;
  const selectedTaskUsesMaxItems = Boolean(selectedTask?.supports_max_items);
  const selectedTaskHasDatasetIntro = Boolean(selectedTask?.local_dataset_only);
  const selectedTaskIsDataset = selectedTask?.runtime_backend === "dataset";
  const selectedTaskIsExternal = selectedTask?.runtime_backend === "external";
  const selectedTaskIsCoding = selectedTask?.track === "coding_verified";
  const parsedSelectedItemIds = selectedTaskIsDataset ? parseItemIdsInput(selectedItemIdsInput) : null;
  const parsedMaxItems = parsedSelectedItemIds ? null : selectedTaskUsesMaxItems && maxItemsInput.trim() ? Math.max(1, Math.floor(numeric(maxItemsInput))) : null;
  const requestedItemCount = selectedTaskUsesMaxItems ? (parsedMaxItems ?? selectedTaskDefaultMaxItems ?? null) : null;
  const showDemoMaxItemsWarning = Boolean(selectedTaskIsCoding && requestedItemCount && requestedItemCount > 50);
  const maxItemsLabel = selectedTaskUsesMaxItems && selectedTask?.dataset_size
    ? `${selectedTaskIsCoding ? "Problem Count" : "Item Cap"} (full dataset = ${selectedTask.dataset_size})`
    : selectedTaskUsesMaxItems
      ? selectedTaskDefaultMaxItems
        ? `Task Cap (default = ${selectedTaskDefaultMaxItems})`
        : "Task Cap"
      : "Item Cap";
  const maxItemsHelper = selectedTaskUsesMaxItems && selectedTask?.dataset_size
    ? `Dataset size: ${selectedTask.dataset_size} ${selectedTaskIsCoding ? "coding problems" : "items"}`
    : selectedTaskUsesMaxItems
      ? selectedTaskDefaultMaxItems
        ? `Blank uses the current RUN_CONFIG default of ${selectedTaskDefaultMaxItems} tasks.`
        : "Blank uses the current RUN_CONFIG default task subset."
      : "Single-item task: cap disabled";

  return (
    <main className="app-shell">
      <section className="topbar">
        <div>
          <p className="eyebrow">verified benchmark search</p>
          <strong className="topbar-title">Auto Research Console</strong>
        </div>
        <div className="theme-toggle" role="tablist" aria-label="Theme">
          {themeChoices().map((choice) => (
            <button
              key={choice}
              className={`theme-chip ${themePreference === choice ? "active" : ""}`}
              onClick={() => setThemePreference(choice)}
              type="button"
            >
              {themeChoiceLabel(choice)}
            </button>
          ))}
        </div>
      </section>

      <section className="panel control-panel">
        <div className="panel-header">
          <div className="hero-block">
            <p className="eyebrow">run launcher</p>
            <h1 className="hero-title">Auto Research Operations Console</h1>
            <p className="muted hero-copy">
              Each task runs a bounded evolutionary search, verifies candidates locally, and keeps only accepted gains and memory writes.
            </p>
          </div>
        </div>

        <div className="control-grid triple">
          <label className="field">
            <span className="field-label">Benchmark Task</span>
            <select className="control" value={selectedTask?.id ?? ""} onChange={(event) => setSelectedTaskId(event.target.value)}>
              {taskGroups.map((group) => (
                <optgroup key={group.track} label={group.label}>
                  {group.tasks.map((task) => (
                    <option key={task.id} value={task.id}>
                      {task.id}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </label>
          <label className="field">
            <span className="field-label">Proposal Model</span>
            <select className="control" value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)} disabled={!runtimeInfo.available_models.length}>
              {runtimeInfo.available_models.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span className="field-label">Frontier Parents</span>
            <input
              className="control"
              type="number"
              min={1}
              step={1}
              value={branchingFactorInput}
              onChange={(event) => setBranchingFactorInput(event.target.value)}
            />
          </label>
          <label className="field">
            <span className="field-label">Max Search Rounds</span>
            <input
              className="control"
              type="number"
              min={1}
              step={1}
              value={generationBudgetInput}
              onChange={(event) => setGenerationBudgetInput(event.target.value)}
            />
          </label>
          <label className="field">
            <span className="field-label">Candidates Count per Branch</span>
            <input
              className="control"
              type="number"
              min={1}
              step={1}
              value={candidateBudgetInput}
              onChange={(event) => setCandidateBudgetInput(event.target.value)}
            />
          </label>
          <label className="field">
            <span className="field-label">Parallel Item Workers</span>
            <input
              className="control"
              type="number"
              min={1}
              step={1}
              value={itemWorkersInput}
              onChange={(event) => setItemWorkersInput(event.target.value)}
              disabled={selectedTaskIsExternal}
              placeholder={selectedTaskIsExternal ? "n/a" : undefined}
            />
          </label>
          <label className="field">
            <span className="field-label">{maxItemsLabel}</span>
            <input
              className="control"
              type="number"
              min={1}
              step={1}
              placeholder={selectedTaskUsesMaxItems ? (selectedTask?.dataset_size ? "all" : "default") : "n/a"}
              value={maxItemsInput}
              onChange={(event) => setMaxItemsInput(event.target.value)}
              disabled={!selectedTaskUsesMaxItems || Boolean(parsedSelectedItemIds)}
            />
            <span className="small muted">
              {parsedSelectedItemIds ? "Specific item ids selected: Item Cap is ignored for this run." : maxItemsHelper}
            </span>
            {showDemoMaxItemsWarning ? (
              <span className="small launcher-warning">
                Demo warning: running more than 50 LiveCodeBench items here is not recommended.
              </span>
            ) : null}
          </label>
          {selectedTaskIsDataset ? (
            <label className="field">
              <span className="field-label">Item IDs (Optional)</span>
              <input
                className="control"
                type="text"
                placeholder="1,2,3 or aime-2026-01,aime-2026-10"
                value={selectedItemIdsInput}
                onChange={(event) => setSelectedItemIdsInput(event.target.value)}
              />
              <span className="small muted">Comma-separated manifest ids or 1-based question numbers. When set, this overrides Item Cap.</span>
            </label>
          ) : null}
          {selectedTask?.supports_runtime_config ? (
            <label className="field field-span-full">
              <span className="field-label">External RUN_CONFIG</span>
              <textarea
                className="control code-textarea"
                rows={14}
                value={externalConfigInput}
                onChange={(event) => setExternalConfigInput(event.target.value)}
                spellCheck={false}
              />
              <span className="small muted">
                Per-run JSON override for the external benchmark harness. This does not rewrite the checked-in `editable.py`.
              </span>
              {externalConfigDraft.error ? <span className="small launcher-warning">{externalConfigDraft.error}</span> : null}
            </label>
          ) : null}
        </div>

        <div className="button-row">
          <button className="action primary" onClick={() => void runTask(selectedTask?.id ?? null)} type="button">
            Start verified run
          </button>
        </div>
        <p className="small muted">
          Runs execute one task at a time. For dataset and external benchmark tasks, Item Cap limits how many local items or harness cases are expanded in this session.
          {selectedTaskIsExternal ? " External benchmark tasks also use the standard search loop; the JSON RUN_CONFIG above defines the wrapper baseline for this run." : ""}
          {selectedTaskIsCoding ? " LiveCodeBench lazily caches only the requested prefix on first use." : ""}
        </p>

        {selectedTask ? (
          <div className="task-preview">
            <div className="task-summary-row">
              <span className="summary-pill">{benchmarkTierLabel(selectedTask.included_in_main_comparison)}</span>
              <span className="summary-pill">{trackLabel(selectedTask.track)}</span>
              <span className="summary-pill">{selectedTask.answer_metric}</span>
              <span className="summary-pill">{selectedTask.function_name}</span>
              <span className="summary-pill">{taskModeLabel(selectedTask.task_mode)}</span>
              <span className="summary-pill">{optimizationScopeLabel(selectedTask.optimization_scope)}</span>
              <span className="summary-pill">
                cap {generationBudgetInput || DEFAULT_FRONTEND_GENERATION_BUDGET} rounds | {candidateBudgetInput || DEFAULT_FRONTEND_CANDIDATE_BUDGET} candidates/branch | {branchingFactorInput || DEFAULT_FRONTEND_BRANCHING_FACTOR} frontier parents
              </span>
              {selectedTaskIsExternal ? <span className="summary-pill">configured via RUN_CONFIG</span> : <span className="summary-pill">workers {itemWorkersInput || DEFAULT_FRONTEND_ITEM_WORKERS}</span>}
              {parsedSelectedItemIds ? <span className="summary-pill">items {parsedSelectedItemIds.join(", ")}</span> : null}
              {selectedTaskIsExternal ? (
                <span className="summary-pill">
                  {parsedMaxItems ? `task cap ${parsedMaxItems}` : selectedTaskDefaultMaxItems ? `task cap default ${selectedTaskDefaultMaxItems}` : "task cap default"}
                </span>
              ) : null}
            </div>
            <p className="muted">{selectedTask.description}</p>
            {selectedTaskHasDatasetIntro ? (
              <div className="button-row intro-button-row">
                <button className="action" onClick={() => setDatasetIntroTaskId(selectedTask.id)} type="button">
                  Open dataset brief
                </button>
              </div>
            ) : null}
          </div>
        ) : null}
      </section>

      {error ? (
        <section className="panel error-panel">
          <p className="eyebrow">run failure</p>
          <h2>{error.error_type}</h2>
          <p className="muted">{error.error}</p>
          {error.details != null ? (
            <pre className="code-block compact"><code>{stringifyUnknown(error.details)}</code></pre>
          ) : null}
          <div className="metric-grid">
            {metric("terminal-backed", String(Boolean(error.terminal)))}
            {metric("model", error.model ?? "n/a")}
          </div>
        </section>
      ) : null}

      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">runtime profile</p>
            <h2>Runtime and verifier configuration</h2>
          </div>
          <div className="badge-row">
            <span className="badge">active {selectedModel || runtimeInfo.active_model}</span>
          </div>
        </div>
        <div className="metric-grid">
          {metric("temperature", runtimeInfo.temperature)}
          {metric("max tokens", runtimeInfo.max_tokens)}
          {metric("timeout", `${runtimeInfo.timeout_s}s`)}
        </div>
      </section>

      <section className="panel stack">
        <div className="panel-header">
          <div>
            <p className="eyebrow">live execution</p>
            <h2>Live execution trace</h2>
          </div>
        </div>
        {liveTasks.length ? (
          liveTasks.map((task) => liveTaskSection(task, nowMs))
        ) : (
          <section className="empty-state">
            <h3>No active run</h3>
            <p className="muted">Start a task to stream item-level events, verifier results, and frontier decisions here.</p>
          </section>
        )}
      </section>

      <section className="panel stack">
        <div className="panel-header">
          <div>
            <p className="eyebrow">selected task report</p>
            <h2>Latest cached report</h2>
          </div>
          <div className="badge-row">
            <span className="badge">{selectedTask?.id ?? "no task selected"}</span>
            <span className="badge">{payload.summary.active_model}</span>
          </div>
        </div>
        <div className="history-scroll">
          {selectedTaskRun ? (
            runCard(
              selectedTaskRun,
              defaultSelectionSpec,
              Boolean(openCompletedTasks[selectedTaskRun.task.id]),
              () => toggleCompletedTask(selectedTaskRun.task.id),
            )
          ) : (
            <section className="empty-state">
              <h3>No cached report for this task yet</h3>
            </section>
          )}
        </div>
      </section>

      {datasetWarningOpen && datasetWarnings.length ? (
        <section className="modal-overlay" onClick={() => setDatasetWarningOpen(false)} role="presentation">
          <article
            className="modal-card"
            role="dialog"
            aria-modal="true"
            aria-labelledby="dataset-warning-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="panel-header">
              <div>
                <p className="eyebrow">warning</p>
                <h2 id="dataset-warning-title">Local dataset setup incomplete</h2>
              </div>
              <button className="action" onClick={() => setDatasetWarningOpen(false)} type="button">
                Dismiss
              </button>
            </div>
            <p className="muted modal-copy">
              Some enabled dataset tasks are missing local files. They may be unavailable until you run the matching prepare command.
            </p>
            <section className="stack">
              {datasetWarnings.map((warning) => (
                <article className="subpanel error-panel" key={`${warning.task_id}:${warning.manifest_path}`}>
                  <div className="subpanel-header">
                    <div>
                      <h3>{warning.title}</h3>
                      <p className="muted">{trackLabel(warning.track)} · {warning.task_id}</p>
                    </div>
                  </div>
                  <p className="muted modal-copy">{warning.message}</p>
                  <pre className="code-block compact"><code>{warning.prepare_command}</code></pre>
                </article>
              ))}
            </section>
          </article>
        </section>
      ) : null}

      {datasetIntroTask ? (
        <section className="modal-overlay" onClick={() => setDatasetIntroTaskId(null)} role="presentation">
          <article
            className="modal-card"
            role="dialog"
            aria-modal="true"
            aria-labelledby="dataset-intro-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="panel-header">
              <div>
                <p className="eyebrow">dataset brief</p>
                <h2 id="dataset-intro-title">{datasetIntroTask.title}</h2>
              </div>
              <button className="action" onClick={() => setDatasetIntroTaskId(null)} type="button">
                Dismiss
              </button>
            </div>
            <p className="muted modal-copy">{datasetIntroCopy(datasetIntroTask)}</p>
            <div className="task-summary-row">
              <span className="summary-pill">{benchmarkTierLabel(datasetIntroTask.included_in_main_comparison)}</span>
              <span className="summary-pill">{trackLabel(datasetIntroTask.track)}</span>
              <span className="summary-pill">{datasetIntroTask.answer_metric}</span>
              <span className="summary-pill">{datasetIntroTask.objective_label}</span>
              <span className="summary-pill">{datasetIntroTask.function_name}</span>
              <span className="summary-pill">{taskModeLabel(datasetIntroTask.task_mode)}</span>
              <span className="summary-pill">{optimizationScopeLabel(datasetIntroTask.optimization_scope)}</span>
            </div>
            <div className="metric-grid compact-metrics">
              {metric("local items", datasetIntroTask.dataset_size ?? "n/a")}
              {metric("source", datasetIntroTask.dataset_id)}
              {metric("split", datasetIntroTask.split ?? "local")}
              {metric("editable file", datasetIntroTask.editable_file)}
            </div>
          </article>
        </section>
      ) : null}
    </main>
  );
}

export default App;
