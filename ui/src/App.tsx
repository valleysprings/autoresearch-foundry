import { useEffect, useMemo, useRef, useState } from "react";

import { loadJob, loadLatestRun, loadRuntime, loadTasks, startJob } from "./api";
import { displayErrorType, normalizeErrorPayload, stringifyUnknown } from "./errorPayload";
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
  PersonalizationReferenceBenchmark,
  QuestionRecord,
  Run,
  RunTask,
  RuntimeInfo,
  TaskSkill,
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
  displayOrder: number | null;
  status: "queued" | "running" | "failed" | "completed";
  latestGeneration: number;
  branchCount: number;
  acceptCount: number;
  memoryDelta: number;
  bestObjective: number | null;
  latestObjective: number | null;
  responseObjective: number | null;
  itemBrief: string | null;
  expectedAnswer: string | null;
  testCaseCount: number | null;
  latestPassedTests: number | null;
  latestTotalTests: number | null;
  responsePassedTests: number | null;
  responseTotalTests: number | null;
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

type SearchRoundSummary = {
  generation: number;
  averageObjective: number;
  sampleCount: number;
};

type LiveTaskCard = {
  taskId: string;
  title: string;
  description: string;
  interactionMode: string;
  taskShape: string | null;
  scoringMode: string | null;
  objectiveSpec: ObjectiveSpec;
  objectiveLabel: string;
  objectiveUnit: string | null;
  model: string;
  branchingFactor: number;
  generationBudget: number;
  candidateBudget: number;
  llmConcurrency: number | null;
  itemWorkers: number | null;
  maxItems: number | null;
  maxEpisodes: number | null;
  selectedItemIds: string[] | null;
  usesMaxItems: boolean;
  usesMaxEpisodes: boolean;
  defaultMaxItems: number | null;
  defaultMaxEpisodes: number | null;
  scheduledItems: number | null;
  roundSummaries: SearchRoundSummary[];
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
  roundObjectives: Map<number, Map<string, number>>;
  acceptedCount: number;
  memoryDelta: number;
};

type BrowseModeGroup = {
  mode: string;
  label: string;
  tasks: TaskSummary[];
};

type InteractionGroup = {
  mode: string;
  label: string;
  tasks: TaskSummary[];
};

type CategoryGroup = {
  key: string;
  label: string;
  tasks: TaskSummary[];
};

type PersonalizationReferenceSourceGroup = {
  key: string;
  sourceLabel: string;
  sourceUrl: string;
  mirrorSlug: string | null;
  references: PersonalizationReferenceBenchmark[];
};

const IDLE_LATEST_RUN_POLL_MS = 15000;
const LIVE_JOB_POLL_MS = 500;
const LIVE_JOB_BACKGROUND_POLL_MS = 1500;
const DEFAULT_FRONTEND_BRANCHING_FACTOR = 1;
const DEFAULT_FRONTEND_GENERATION_BUDGET = 3;
const DEFAULT_FRONTEND_CANDIDATE_BUDGET = 1;
const DEFAULT_FRONTEND_LLM_CONCURRENCY = 20;
const DEFAULT_FRONTEND_ITEM_WORKERS = 5;
const DEFAULT_BROWSE_MODE = "general_intelligence";
const DEFAULT_INTERACTION_MODE = "single_turn";
const SAFETY_BROWSER_MODE = "safety";
const PERSONALIZATION_BROWSER_MODE = "personalization";
const PERSONALIZATION_CATEGORY_ORDER: Record<string, number> = {
  character_knowledge: 0,
  character_portrayal: 1,
  consistency_robustness: 2,
  user_personalization: 3,
  agentic_personalization: 4,
  uncategorized: 5,
};
const SAFETY_CATEGORY_ORDER: Record<string, number> = {
  jailbreak_attack: 0,
  should_refuse: 1,
  over_refusal: 2,
  factuality_hallucination: 3,
  policy_drift: 4,
  benign_utility: 5,
  safety_degradation: 6,
  uncategorized: 7,
};
const SAFETY_FOCUS_ORDER: Record<string, number> = {
  jailbreak_attack: 0,
  should_refuse: 1,
  over_refusal: 2,
  factuality_hallucination: 3,
  policy_drift: 4,
  benign_utility: 5,
  safety_degradation: 6,
  uncategorized: 7,
};

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

function dialogueTurnText(turn: unknown): string | null {
  if (!turn || typeof turn !== "object") {
    return null;
  }
  const record = turn as Record<string, unknown>;
  const speaker = stringValue(record.speaker) ?? stringValue(record.role) ?? stringValue(record.from) ?? "speaker";
  const text = stringValue(record.text) ?? stringValue(record.content) ?? stringValue(record.value);
  if (!text) {
    return null;
  }
  return `${speaker}: ${text}`;
}

function dialogueSnippet(turns: unknown, limit = 240): string | null {
  if (!Array.isArray(turns) || !turns.length) {
    return null;
  }
  const parts = turns
    .map((turn) => dialogueTurnText(turn))
    .filter((value): value is string => Boolean(value));
  if (!parts.length) {
    return null;
  }
  return questionPreview(parts.slice(-3).join(" "), limit);
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

function nonNegativeInteger(value: unknown): number | null {
  if (value == null || typeof value === "boolean") {
    return null;
  }
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 0) {
    return null;
  }
  return parsed;
}

function itemOrderFromQuestion(question: QuestionRecord | null | undefined): number | null {
  if (!question || typeof question !== "object") {
    return null;
  }
  return nonNegativeInteger(question.metadata?.source_index);
}

function itemOrderFromItemRun(itemRun: ItemRun | null | undefined): number | null {
  if (!itemRun || typeof itemRun !== "object") {
    return null;
  }
  return nonNegativeInteger(itemRun.item_source_index) ?? itemOrderFromQuestion(itemRun.question);
}

function compareItemDisplayOrder(
  left: { displayOrder: number | null; itemId: string },
  right: { displayOrder: number | null; itemId: string },
): number {
  if (left.displayOrder != null && right.displayOrder != null && left.displayOrder !== right.displayOrder) {
    return left.displayOrder - right.displayOrder;
  }
  if (left.displayOrder != null) {
    return -1;
  }
  if (right.displayOrder != null) {
    return 1;
  }
  return left.itemId.localeCompare(right.itemId);
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
  return (
    stringValue((result as Record<string, unknown>).error)
    ?? stringValue((result as Record<string, unknown>).reason)
  );
}

function candidateResponseOutput(candidate: Candidate | undefined | null): string | null {
  const result = firstTestResult(candidate);
  return stringValue(result?.actual_display) ?? stringValue(result?.actual) ?? stringValue(result?.actual_raw);
}

function normalizeResponseText(value: unknown): string {
  return String(value ?? "").normalize("NFKC").trim().replace(/\s+/g, " ").toLowerCase();
}

function previewResponseText(value: string, limit = 140): string {
  const text = value.trim();
  if (text.length <= limit) {
    return text;
  }
  return `${text.slice(0, limit - 3).trimEnd()}...`;
}

function questionLabelAlias(question: QuestionRecord | undefined | null, label: string): string | null {
  const metadata = (question?.metadata ?? {}) as Record<string, unknown>;
  const rawAliases = metadata.label_aliases;
  if (!rawAliases || typeof rawAliases !== "object") {
    return null;
  }
  const aliases = (rawAliases as Record<string, unknown>)[label];
  if (!Array.isArray(aliases)) {
    return null;
  }
  const normalizedLabel = normalizeResponseText(label);
  for (const alias of aliases) {
    const text = stringValue(alias);
    if (!text) {
      continue;
    }
    const normalizedAlias = normalizeResponseText(text);
    if (!normalizedAlias || normalizedAlias === normalizedLabel) {
      continue;
    }
    if (normalizedAlias.length === 1 && normalizedAlias >= "a" && normalizedAlias <= "z") {
      continue;
    }
    return text;
  }
  return null;
}

function choiceLabelForIndex(index: number): string {
  return index >= 0 && index < 26 ? String.fromCharCode("A".charCodeAt(0) + index) : String(index + 1);
}

function liveCandidateOutput(value: string | undefined | null): string | null {
  const text = stringValue(value);
  if (!text) {
    return null;
  }
  return text === "[]" || text === "{}" ? null : text;
}

function queuedItemsLabel(visibleItems: number, totalItems: number): string | null {
  if (!(totalItems > visibleItems)) {
    return null;
  }
  return `showing ${visibleItems} active/completed; ${totalItems - visibleItems} queued`;
}

function candidateDisplayOutput(candidate: Candidate | undefined | null, question?: QuestionRecord | null): string | null {
  const result = firstTestResult(candidate);
  const actualDisplay = stringValue(result?.actual_display);
  if (actualDisplay) {
    return actualDisplay;
  }

  const actual = stringValue(result?.actual);
  const actualRaw = stringValue(result?.actual_raw);
  const metadata = (question?.metadata ?? {}) as Record<string, unknown>;
  const allowedLabels = Array.isArray(metadata.allowed_labels)
    ? metadata.allowed_labels.map((value) => stringValue(value)).filter((value): value is string => Boolean(value))
    : [];
  if (actual && allowedLabels.includes(actual)) {
    const alias = questionLabelAlias(question, actual);
    if (alias) {
      return `${actual} -> ${previewResponseText(alias)}`;
    }
    return actual;
  }

  const choices = Array.isArray(question?.choices) ? question?.choices : [];
  if (actual && choices.length) {
    const actualKey = normalizeResponseText(actual);
    const choiceIndex = choices.findIndex((choice) => normalizeResponseText(choice) === actualKey);
    if (choiceIndex >= 0) {
      return `${choiceLabelForIndex(choiceIndex)} -> ${previewResponseText(String(choices[choiceIndex]))}`;
    }
  }

  if (actualRaw) {
    return previewResponseText(actualRaw);
  }
  if (actual) {
    return previewResponseText(actual);
  }
  const metricError = stringValue(candidate?.metrics?.error);
  if (metricError) {
    return previewResponseText(metricError);
  }
  const resultReason = firstTestResultReason(candidate);
  return resultReason ? previewResponseText(resultReason) : null;
}

function latestAttemptedCandidate(itemRun: ItemRun | undefined | null): Candidate | null {
  if (!itemRun) {
    return null;
  }
  const generations = itemRun.generations ?? [];
  for (let generationIndex = generations.length - 1; generationIndex >= 0; generationIndex -= 1) {
    const generation = generations[generationIndex];
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

function candidateVerifierStatus(candidate: Candidate | undefined | null): string | null {
  return stringValue(candidate?.metrics.verifier_status) ?? stringValue(candidate?.metrics.status);
}

function isUnavailableStatus(status: string | undefined | null): boolean {
  const normalized = String(status ?? "").trim().toLowerCase();
  return normalized === "error" || normalized === "not-run";
}

function candidateObjectiveUnavailable(candidate: Candidate | undefined | null): boolean {
  return isUnavailableStatus(candidateVerifierStatus(candidate));
}

function objectiveReachedFullScore(
  objective: string | number | undefined | null,
  objectiveSpec: ObjectiveSpec | undefined | null,
): boolean | null {
  if (!objectiveSpec || String(objectiveSpec.unit ?? "").trim().toLowerCase() !== "ratio") {
    return null;
  }
  const value = numeric(objective);
  const direction = String(objectiveSpec.direction ?? "max").trim().toLowerCase();
  if (direction === "min") {
    return value <= 1e-9;
  }
  return value >= 1 - 1e-9;
}

function objectiveStatus(
  objective: string | number | undefined | null,
  objectiveSpec: ObjectiveSpec | undefined | null,
): "pass" | "partial" | "fail" | null {
  const fullObjective = objectiveReachedFullScore(objective, objectiveSpec);
  if (fullObjective === true) {
    return "pass";
  }
  if (fullObjective === false) {
    return numeric(objective) > 0 ? "partial" : "fail";
  }
  return null;
}

function candidatePassedAllTests(candidate: Candidate | undefined | null): boolean | null {
  if (!candidate) {
    return null;
  }
  const results = Array.isArray(candidate.metrics.test_results) ? candidate.metrics.test_results : [];
  const explicitPassed = results
    .map((result) => (typeof result?.passed === "boolean" ? result.passed : null))
    .filter((value): value is boolean => value != null);
  if (explicitPassed.length > 0) {
    return explicitPassed.every(Boolean);
  }
  const passedTests = numeric(candidate.metrics.passed_tests);
  const totalTests = numeric(candidate.metrics.total_tests);
  if (totalTests > 0) {
    return passedTests === totalTests;
  }
  return null;
}

function explicitCheckCount(results: CandidateTestResult[] | undefined | null): number {
  return Array.isArray(results) ? results.length : 0;
}

function multiCheckCount(
  totalTests: string | number | undefined | null,
  results: CandidateTestResult[] | undefined | null,
): number {
  return Math.max(numeric(totalTests), explicitCheckCount(results));
}

function hasMultiChecks(
  totalTests: string | number | undefined | null,
  results: CandidateTestResult[] | undefined | null,
): boolean {
  return multiCheckCount(totalTests, results) > 1;
}

function candidateHasMultiChecks(candidate: Candidate | undefined | null): boolean {
  return hasMultiChecks(candidate?.metrics.total_tests, candidate?.metrics.test_results);
}

function testProgressStatus(
  passedTests: string | number | undefined | null,
  totalTests: string | number | undefined | null,
): string | null {
  const total = numeric(totalTests);
  if (!(total > 0)) {
    return null;
  }
  const passed = Math.max(0, Math.min(total, numeric(passedTests)));
  if (passed >= total) {
    return "pass";
  }
  if (passed <= 0) {
    return "fail";
  }
  return "partial";
}

function candidateResponseStatus(candidate: Candidate | undefined | null, objectiveSpec?: ObjectiveSpec | null): string | null {
  const rawStatus = candidateVerifierStatus(candidate);
  if (!candidate || !rawStatus) {
    return null;
  }
  if (isUnavailableStatus(rawStatus)) {
    return "fail";
  }

  const fullObjective = objectiveReachedFullScore(
    candidate.metrics.objective_score ?? candidate.metrics.objective,
    objectiveSpec,
  );
  if (fullObjective === true) {
    return "pass";
  }
  if (fullObjective === false) {
    return numeric(candidate.metrics.objective_score ?? candidate.metrics.objective) > 0 ? "partial" : "fail";
  }

  const byTests = testProgressStatus(candidate.metrics.passed_tests, candidate.metrics.total_tests);
  if (byTests) {
    return byTests;
  }

  const passedAllTests = candidatePassedAllTests(candidate);
  if (passedAllTests === true) {
    return "pass";
  }
  if (passedAllTests === false) {
    return "fail";
  }

  if (rawStatus === "pass") {
    return "pass";
  }
  return stringValue(candidateResponseOutput(candidate)) ? "partial" : "fail";
}

function testProgress(
  passedTests: string | number | undefined | null,
  totalTests: string | number | undefined | null,
  objective: string | number | undefined | null = null,
  objectiveSpec: ObjectiveSpec | undefined | null = null,
): { passed: number; total: number; status: "pass" | "partial" | "fail"; label: string } | null {
  const total = numeric(totalTests);
  const passed = Math.max(0, Math.min(total, numeric(passedTests)));
  const byObjective = objectiveStatus(objective, objectiveSpec);
  if (byObjective === "partial") {
    if (total > 0 && passed >= total) {
      return { passed, total, status: "partial", label: "Partial score" };
    }
    if (total > 0) {
      return { passed, total, status: "partial", label: `${passed}/${total} checks passed; partial score` };
    }
    return { passed: 0, total: 0, status: "partial", label: "Partial score" };
  }
  if (byObjective === "fail") {
    if (total > 0) {
      return { passed, total, status: "fail", label: `${passed}/${total} checks passed; score 0` };
    }
    return { passed: 0, total: 0, status: "fail", label: "Score 0" };
  }
  if (!(total > 0)) {
    return null;
  }
  if (passed >= total) {
    return { passed, total, status: "pass", label: "ALL test cases passed" };
  }
  if (passed <= 0) {
    return { passed, total, status: "fail", label: `0/${total} tests passed` };
  }
  return { passed, total, status: "partial", label: `${passed}/${total} tests passed` };
}

function formatCandidateObjectiveValue(candidate: Candidate | undefined | null, spec: ObjectiveSpec): string {
  if (!candidate) {
    return "n/a";
  }
  if (candidateObjectiveUnavailable(candidate)) {
    return "unavailable";
  }
  return formatObjectiveValue(candidate.metrics.objective, spec);
}

function formatCandidateMetricValue(candidate: Candidate | undefined | null, value: string | number | undefined | null): string {
  if (!candidate) {
    return "n/a";
  }
  if (candidateObjectiveUnavailable(candidate)) {
    return "unavailable";
  }
  return formatValue(value);
}

function verifierTone(status: string | undefined | null): "" | "good" | "warn" | "bad" {
  const normalized = String(status ?? "").toLowerCase();
  if (normalized === "pass") {
    return "good";
  }
  if (normalized === "partial") {
    return "warn";
  }
  if (normalized === "fail" || normalized === "error" || normalized === "not-run") {
    return "bad";
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

function formatSkillOptionLabel(skill: TaskSkill): string {
  const parts = [
    skill.source_model || "unknown model",
    typeof skill.source_items === "number" ? `${skill.source_items} items` : null,
  ].filter(Boolean);
  const generatedAt = stringValue(skill.generated_at);
  if (generatedAt) {
    parts.push(generatedAt.replace("T", " ").replace(/([+-]\d\d:\d\d|Z)$/, ""));
  }
  return parts.length ? parts.join(" · ") : skill.filename;
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

function inferSuiteDefaultMaxItems(config: Record<string, unknown> | null | undefined): number | null {
  if (!config) {
    return null;
  }
  for (const key of ["task_limit", "n_tasks", "cases"]) {
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
  for (const key of ["task_ids", "problem_names", "task_names", "tasks", "inline_episodes"]) {
    const value = config[key];
    if (Array.isArray(value) && value.length) {
      return value.length;
    }
  }
  return null;
}

function inferSuiteDefaultMaxEpisodes(config: Record<string, unknown> | null | undefined): number | null {
  if (!config) {
    return null;
  }
  for (const key of ["episode_limit", "n_episodes", "max_episodes", "task_limit"]) {
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
  for (const key of ["episode_ids", "episodes", "inline_episodes"]) {
    const value = config[key];
    if (Array.isArray(value) && value.length) {
      return value.length;
    }
  }
  return null;
}

function inferSuiteDefaultMaxTurns(config: Record<string, unknown> | null | undefined): number | null {
  if (!config) {
    return null;
  }
  const value = config.max_turns;
  if (typeof value === "number" && Number.isFinite(value) && value > 0) {
    return Math.max(1, Math.floor(value));
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed) && parsed > 0) {
      return Math.max(1, Math.floor(parsed));
    }
  }
  return null;
}

function familyLabel(family: string | undefined | null): string {
  const normalized = String(family ?? "").trim().toLowerCase();
  const labels: Record<string, string> = {
    "agent-benchmark": "Agent",
    coding: "Code",
    math: "Math",
    "operations-research": "OR",
    planning: "Planning",
    qa: "QA",
    reasoning: "Reasoning",
    science: "Science",
    text2sql: "Text-to-SQL",
  };
  return labels[normalized] ?? (normalized ? normalized.replace(/-/g, " ") : "Task");
}

function isLocalDatasetTask(task: TaskSummary | RunTask | null | undefined): boolean {
  if (!task) {
    return false;
  }
  return Boolean(task.local_dataset_only || task.supports_max_items || task.supports_max_episodes);
}

function taskSupportsParallelWorkers(task: TaskSummary | null | undefined): boolean {
  if (!task) {
    return false;
  }
  if (isLocalDatasetTask(task)) {
    return true;
  }
  return Boolean(task.supports_max_items || task.supports_max_episodes || task.interaction_mode === "multi_turn");
}

function defaultParallelWorkers(task: TaskSummary | null | undefined): number {
  if (!task) {
    return DEFAULT_FRONTEND_ITEM_WORKERS;
  }
  const configured = Number(task.item_workers ?? 0);
  if (Number.isFinite(configured) && configured > 0) {
    return Math.max(1, Math.floor(configured));
  }
  if (task.interaction_mode === "multi_turn") {
    return 1;
  }
  return DEFAULT_FRONTEND_ITEM_WORKERS;
}

function defaultLlmConcurrency(runtime: RuntimeInfo | null | undefined): number {
  const configured = Number(runtime?.llm_concurrency ?? 0);
  if (Number.isFinite(configured) && configured > 0) {
    return Math.max(1, Math.floor(configured));
  }
  return DEFAULT_FRONTEND_LLM_CONCURRENCY;
}

function resolvedLlmConcurrency(
  requestedLlmConcurrency: number | null,
  runtimeDefault: number,
  parallelWorkers: number | null,
): number {
  if (requestedLlmConcurrency != null) {
    return requestedLlmConcurrency;
  }
  if (parallelWorkers != null) {
    return parallelWorkers;
  }
  return runtimeDefault;
}

function average(values: number[]): number | null {
  if (!values.length) {
    return null;
  }
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function firstRoundBaseline(run: { generations?: Generation[] | null }): Candidate | null {
  if (!Array.isArray(run.generations) || !run.generations.length) {
    return null;
  }
  const firstGeneration = run.generations[0];
  const firstCandidate = Array.isArray(firstGeneration?.candidates) ? (firstGeneration.candidates[0] ?? null) : null;
  return firstCandidate ?? firstGeneration?.winner ?? null;
}

function improvementRatio(finalValue: string | number | undefined | null, anchorValue: string | number | undefined | null): number | null {
  const anchor = numeric(anchorValue);
  if (Math.abs(anchor) < 1e-9) {
    return null;
  }
  return numeric(finalValue) / anchor;
}

function itemRunImprovementRatio(itemRun: ItemRun): number | null {
  const baseline = firstRoundBaseline(itemRun);
  return baseline && itemRun.winner ? improvementRatio(itemRun.winner.metrics.primary_score, baseline.metrics.primary_score) : null;
}

function runImprovementRatio(run: Run): number | null {
  if (Array.isArray(run.item_runs) && run.item_runs.length) {
    const ratios = run.item_runs
      .map((itemRun) => itemRunImprovementRatio(itemRun))
      .filter((value): value is number => value != null);
    return average(ratios);
  }
  const baseline = firstRoundBaseline(run);
  return baseline ? improvementRatio(run.winner.metrics.primary_score, baseline.metrics.primary_score) : null;
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

function passedCandidate(candidate: Candidate | undefined | null, objectiveSpec?: ObjectiveSpec | null): boolean {
  return candidateResponseStatus(candidate, objectiveSpec) === "pass";
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
    const baselinePassed = passedCandidate(firstRoundBaseline(itemRun), run.task.objective_spec);
    const winnerPassed = passedCandidate(itemRun.winner, run.task.objective_spec);
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

function roundAverageTone(
  roundSummary: SearchRoundSummary,
  previousSummary: SearchRoundSummary | null,
): "" | "good" | "warn" {
  if (!previousSummary) {
    return "";
  }
  if (roundSummary.averageObjective > previousSummary.averageObjective + 1e-9) {
    return "good";
  }
  if (roundSummary.averageObjective < previousSummary.averageObjective - 1e-9) {
    return "warn";
  }
  return "";
}

function generationObjectiveFromSummary(generation: Generation): number | null {
  const bestAfterGeneration = (generation as Generation & { best_after_generation?: Candidate }).best_after_generation;
  const objective = bestAfterGeneration?.metrics?.objective ?? generation.winner?.metrics?.objective;
  const parsed = numeric(objective);
  return Number.isFinite(parsed) ? parsed : null;
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
    profile: "n/a",
    provider: "n/a",
    transport: "n/a",
    default_model: "n/a",
    active_model: "n/a",
    available_models: [],
    base_url: "n/a",
    temperature: "n/a",
    max_tokens: "n/a",
    timeout_s: "n/a",
    llm_concurrency: "n/a",
    supports_tools: "n/a",
    supports_json_mode: "n/a",
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
      policy_model: "n/a",
      eval_model: null,
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
      policy_model: "n/a",
      eval_model: null,
    },
    task_catalog: taskCatalog,
    runs: [],
  };
}

function normalizePayload(payload: Payload, fallbackCatalog: TaskSummary[]): Payload {
  const taskCatalog = mergeTaskCatalogs(fallbackCatalog, payload.task_catalog);
  const activeTaskIds = new Set(taskCatalog.map((task) => task.id));
  const normalizedSummary = {
    ...payload.summary,
    policy_model: payload.summary?.policy_model ?? payload.summary?.active_model ?? "n/a",
    eval_model: payload.summary?.eval_model ?? null,
  };
  return {
    ...payload,
    summary: normalizedSummary,
    audit: {
      ...payload.audit,
      policy_model: payload.audit?.policy_model ?? normalizedSummary.policy_model ?? null,
      eval_model: payload.audit?.eval_model ?? normalizedSummary.eval_model ?? null,
    },
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
              policy_model: run.policy_model ?? run.active_model ?? normalizedSummary.policy_model,
              eval_model: run.eval_model ?? normalizedSummary.eval_model ?? null,
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
  return includedInMainComparison ? "active benchmark task" : "inactive task";
}

function trackLabel(track: string): string {
  const labels: Record<string, string> = {
    math_verified: "Mathematics",
    reasoning_verified: "Reasoning",
    text2sql_verified: "Text-to-SQL",
    longcontext_verified: "Long Context",
    personalization_verified: "Personalization",
    safety_verified: "Safety",
    browse_snapshot: "Browse",
    science_verified: "Science Reasoning",
    coding_verified: "Coding",
    or_verified: "Operations Research",
    agent_verified: "Agent Verified",
  };
  return labels[track] ?? track.replace(/_/g, " ");
}

function browseModeLabel(mode: string | undefined | null): string {
  const normalized = String(mode ?? "").trim().toLowerCase();
  if (normalized === DEFAULT_BROWSE_MODE || normalized === "general") {
    return "General Intelligence";
  }
  if (normalized === PERSONALIZATION_BROWSER_MODE) {
    return "Personalization";
  }
  if (normalized === SAFETY_BROWSER_MODE) {
    return "Safety";
  }
  return "Unknown";
}

function interactionModeLabel(mode: string | undefined | null): string {
  const normalized = String(mode ?? "").trim().toLowerCase();
  if (normalized === "single_turn") {
    return "Single Turn";
  }
  if (normalized === "multi_turn") {
    return "Multi Turn";
  }
  return "Unknown";
}

function personalizationCategoryLabel(category: string | undefined | null): string {
  const normalized = String(category ?? "").trim().toLowerCase();
  if (normalized === "role_play") {
    return "Role Play";
  }
  if (normalized === "user_persona") {
    return "User Persona";
  }
  if (normalized === "trait_behavior") {
    return "Trait / Behavior";
  }
  if (normalized === "explicit_character_persona") {
    return "Explicit Character Persona";
  }
  if (normalized === "user_persona_personalization") {
    return "User Persona / Personalization";
  }
  return "Uncategorized";
}

function personaBenchmarkCategoryLabel(category: string | undefined | null): string {
  const normalized = String(category ?? "").trim().toLowerCase();
  if (normalized === "character_knowledge") {
    return "Character Knowledge";
  }
  if (normalized === "character_portrayal") {
    return "Character Portrayal";
  }
  if (normalized === "consistency_robustness") {
    return "Consistency / Robustness";
  }
  if (normalized === "user_personalization") {
    return "User Personalization";
  }
  if (normalized === "agentic_personalization") {
    return "Agentic Personalization";
  }
  return "Uncategorized";
}

function personalizationSecondaryCategoryLabel(category: string | undefined | null): string {
  const normalized = String(category ?? "").trim().toLowerCase();
  const labels: Record<string, string> = {
    character_knowledge: "character knowledge",
    personality_fidelity: "personality fidelity",
    style_fidelity: "style fidelity",
    attribute_control: "attribute control",
    social_drift: "social drift",
    memory_consistency: "memory consistency",
    self_awareness: "self awareness",
    temporal_consistency: "temporal consistency",
    dialogue_consistency: "dialogue consistency",
    behavior_consistency: "behavior consistency",
    instruction_resolution: "instruction resolution",
    literary_dialogue: "literary dialogue",
    decision_alignment: "decision alignment",
    persona_conditioning: "persona conditioning",
    preference_following: "preference following",
    persona_memory: "persona memory",
    long_horizon_personalization: "long-horizon personalization",
    latent_trait_inference: "latent trait inference",
    agentic_personalization: "agentic personalization",
    task_success: "task success",
    factual_reliability: "factual reliability",
    consistency_stress_test: "consistency stress test",
    anime_acg_slice: "anime / ACG slice",
    user_intent_alignment: "user intent alignment",
  };
  return labels[normalized] ?? normalized.replace(/[_-]+/g, " ");
}

function safetyCategoryKey(task: TaskSummary | RunTask | null | undefined): string {
  const normalized = String(task?.safety_category ?? "").trim().toLowerCase();
  return normalized || "uncategorized";
}

function safetyCategoryLabel(category: string | undefined | null): string {
  const normalized = String(category ?? "").trim().toLowerCase();
  const labels: Record<string, string> = {
    jailbreak_attack: "Jailbreak / Attack",
    should_refuse: "Should Refuse",
    over_refusal: "Over-refusal",
    factuality_hallucination: "Factuality / Hallucination",
    policy_drift: "Policy Drift",
    benign_utility: "Benign Utility",
    safety_degradation: "Safety Degradation",
  };
  return labels[normalized] ?? "Uncategorized";
}

function safetyFocusKey(task: TaskSummary | RunTask | null | undefined): string {
  const normalized = String(task?.safety_focus ?? "").trim().toLowerCase();
  return normalized || "uncategorized";
}

function safetyFocusLabel(focus: string | undefined | null): string {
  return safetyCategoryLabel(focus);
}

function safetyBenchmarkCategoryKey(task: TaskSummary | RunTask | null | undefined): string {
  return safetyCategoryKey(task);
}

function safetyBenchmarkCategoryLabel(task: TaskSummary | RunTask | null | undefined): string {
  return safetyCategoryLabel(safetyCategoryKey(task));
}

function subjectDomainLabel(domain: string | undefined | null): string {
  const normalized = String(domain ?? "").trim().toLowerCase();
  const labels: Record<string, string> = {
    anime_acg: "anime / ACG",
    games: "games",
    literary_fiction: "literary fiction",
    movie_tv: "movie / TV",
    general_fiction: "general fiction",
    celebrity_real_person: "celebrity / real person",
    assistant_task_oriented: "assistant / task-oriented",
  };
  return labels[normalized] ?? normalized.replaceAll("_", " ");
}

function officialMetricBackendLabel(backend: string | undefined | null): string {
  const normalized = String(backend ?? "").trim().toLowerCase();
  const labels: Record<string, string> = {
    deterministic_local: "deterministic local",
    llm_judge: "LLM judge",
    reward_model: "reward model",
    hybrid: "hybrid",
  };
  return labels[normalized] ?? normalized.replaceAll("_", " ");
}

function metricFidelityLabel(fidelity: string | undefined | null): string {
  const normalized = String(fidelity ?? "").trim().toLowerCase();
  const labels: Record<string, string> = {
    official: "official",
    adapted_local: "adapted local",
    proxy_local: "proxy local",
    reference_only: "reference only",
  };
  return labels[normalized] ?? normalized.replaceAll("_", " ");
}

function officialMetricGranularityLabel(granularity: string | undefined | null): string {
  const normalized = String(granularity ?? "").trim().toLowerCase();
  const labels: Record<string, string> = {
    item_level: "item level",
    turn_level: "turn level",
    dialogue_level: "dialogue level",
    episode_level: "episode level",
    benchmark_level: "benchmark level",
  };
  return labels[normalized] ?? normalized.replaceAll("_", " ");
}

function referenceStatusLabel(status: string | undefined | null): string {
  const normalized = String(status ?? "").trim().toLowerCase();
  if (normalized === "local_task") {
    return "local task";
  }
  if (normalized === "planned_task") {
    return "planned task";
  }
  if (normalized === "external_reference") {
    return "external reference";
  }
  return "reference";
}

function referenceStatusTone(status: string | undefined | null): "" | "good" | "warn" {
  const normalized = String(status ?? "").trim().toLowerCase();
  if (normalized === "local_task") {
    return "good";
  }
  if (normalized === "planned_task") {
    return "warn";
  }
  return "";
}

function referenceImplementationStatusKey(status: string | undefined | null): "running" | "planned" | "blocked" {
  const normalized = String(status ?? "").trim().toLowerCase();
  if (normalized === "running") {
    return "running";
  }
  if (normalized === "blocked") {
    return "blocked";
  }
  return "planned";
}

function referenceImplementationStatusLabel(status: string | undefined | null): string {
  const key = referenceImplementationStatusKey(status);
  if (key === "running") {
    return "running";
  }
  if (key === "blocked") {
    return "blocked";
  }
  return "planned";
}

function referenceImplementationStatusTone(status: string | undefined | null): "good" | "warn" | "bad" {
  const key = referenceImplementationStatusKey(status);
  if (key === "running") {
    return "good";
  }
  if (key === "blocked") {
    return "bad";
  }
  return "warn";
}

function taskShapeLabel(shape: string | undefined | null): string {
  const normalized = String(shape ?? "").trim().toLowerCase();
  if (!normalized) {
    return "shape unknown";
  }
  return normalized.replaceAll("_", " ");
}

function scoringModeLabel(mode: string | undefined | null): string {
  const normalized = String(mode ?? "").trim().toLowerCase();
  if (!normalized) {
    return "scoring unknown";
  }
  return normalized.replaceAll("_", " ");
}

function referenceCategoryKey(reference: PersonalizationReferenceBenchmark): string {
  return String(reference.primary_category ?? "").trim().toLowerCase();
}

function referenceSourceKey(reference: PersonalizationReferenceBenchmark): string {
  const mirrorSlug = String(reference.mirror_slug ?? "").trim().toLowerCase();
  if (mirrorSlug) {
    return `mirror:${mirrorSlug}`;
  }
  const sourceUrl = String(reference.source_url ?? "").trim().toLowerCase();
  if (sourceUrl) {
    return `source:${sourceUrl}`;
  }
  return `label:${String(reference.source_label ?? "").trim().toLowerCase()}`;
}

function referenceTaskIds(reference: PersonalizationReferenceBenchmark): string[] {
  const values: string[] = [];
  for (const rawValue of Array.isArray(reference.task_ids) ? reference.task_ids : []) {
    const value = String(rawValue ?? "").trim();
    if (value && !values.includes(value)) {
      values.push(value);
    }
  }
  const taskId = String(reference.task_id ?? "").trim();
  if (taskId && !values.includes(taskId)) {
    values.push(taskId);
  }
  return values;
}

function primaryCategoryForTask(
  task: TaskSummary | RunTask | null | undefined,
  references: PersonalizationReferenceBenchmark[] = [],
): string {
  if (!task) {
    return "uncategorized";
  }
  for (const reference of references) {
    if (!referenceTaskIds(reference).includes(task.id)) {
      continue;
    }
    const primaryCategory = String(reference.primary_category ?? "").trim().toLowerCase();
    if (primaryCategory) {
      return primaryCategory;
    }
  }
  const legacyCategory = String(task.personalization_category ?? "").trim().toLowerCase();
  if (legacyCategory === "role_play") {
    return "character_portrayal";
  }
  if (legacyCategory === "user_persona") {
    return "user_personalization";
  }
  return "uncategorized";
}

function browserModeForTask(task: TaskSummary | RunTask | null | undefined): string {
  if (!task) {
    return DEFAULT_BROWSE_MODE;
  }
  const researchLine = String(task.research_line ?? "").trim().toLowerCase();
  if (researchLine === PERSONALIZATION_BROWSER_MODE) {
    return PERSONALIZATION_BROWSER_MODE;
  }
  if (researchLine === SAFETY_BROWSER_MODE) {
    return SAFETY_BROWSER_MODE;
  }
  if (researchLine === DEFAULT_BROWSE_MODE) {
    return DEFAULT_BROWSE_MODE;
  }
  if (String(task.track ?? "").trim().toLowerCase() === "safety_verified") {
    return SAFETY_BROWSER_MODE;
  }
  return DEFAULT_BROWSE_MODE;
}

function taskModeLabel(mode: string | undefined | null): string {
  const normalized = String(mode ?? "").trim().toLowerCase();
  if (normalized === "answer") {
    return "Answer task";
  }
  if (normalized === "artifact") {
    return "Artifact task";
  }
  return "Unknown task";
}

function groupTasksByBrowseMode(tasks: TaskSummary[]): BrowseModeGroup[] {
  const groups = new Map<string, BrowseModeGroup>();
  for (const task of tasks) {
    const mode = browserModeForTask(task);
    const existing = groups.get(mode);
    if (existing) {
      existing.tasks.push(task);
      continue;
    }
    groups.set(mode, {
      mode,
      label: browseModeLabel(mode),
      tasks: [task],
    });
  }
  return [DEFAULT_BROWSE_MODE, PERSONALIZATION_BROWSER_MODE, SAFETY_BROWSER_MODE]
    .map((mode) => groups.get(mode))
    .filter((group): group is BrowseModeGroup => Boolean(group));
}

function groupTasksByInteractionMode(
  tasks: TaskSummary[],
  browseMode: string,
  references: PersonalizationReferenceBenchmark[] = [],
): InteractionGroup[] {
  const groups = new Map<string, InteractionGroup>();
  for (const task of tasks) {
    const mode = String(task.interaction_mode ?? DEFAULT_INTERACTION_MODE).trim().toLowerCase() || DEFAULT_INTERACTION_MODE;
    const existing = groups.get(mode);
    if (existing) {
      existing.tasks.push(task);
      continue;
    }
    groups.set(mode, {
      mode,
      label: interactionModeLabel(mode),
      tasks: [task],
    });
  }
  if (browseMode === PERSONALIZATION_BROWSER_MODE) {
    for (const reference of references) {
      const mode = String(reference.interaction_mode ?? DEFAULT_INTERACTION_MODE).trim().toLowerCase() || DEFAULT_INTERACTION_MODE;
      if (groups.has(mode)) {
        continue;
      }
      groups.set(mode, {
        mode,
        label: interactionModeLabel(mode),
        tasks: [],
      });
    }
  }
  return [DEFAULT_INTERACTION_MODE, "multi_turn"]
    .map((mode) => groups.get(mode))
    .filter((group): group is InteractionGroup => Boolean(group));
}

function taskCategoryKey(
  task: TaskSummary,
  browseMode: string,
  references: PersonalizationReferenceBenchmark[] = [],
): string {
  if (browseMode === PERSONALIZATION_BROWSER_MODE) {
    return primaryCategoryForTask(task, references);
  }
  if (browseMode === SAFETY_BROWSER_MODE) {
    return safetyBenchmarkCategoryKey(task);
  }
  return String(task.track || "").trim();
}

function taskCategoryLabel(
  task: TaskSummary | RunTask,
  browseMode: string,
  references: PersonalizationReferenceBenchmark[] = [],
): string {
  if (browseMode === PERSONALIZATION_BROWSER_MODE) {
    return personaBenchmarkCategoryLabel(primaryCategoryForTask(task, references));
  }
  if (browseMode === SAFETY_BROWSER_MODE) {
    return safetyBenchmarkCategoryLabel(task);
  }
  return trackLabel(String(task.track || ""));
}

function groupTasksByCategory(
  tasks: TaskSummary[],
  browserMode: string,
  interactionMode: string,
  references: PersonalizationReferenceBenchmark[] = [],
): CategoryGroup[] {
  const groups = new Map<string, CategoryGroup>();
  for (const task of tasks) {
    const key = taskCategoryKey(task, browserMode, references);
    if (!key) {
      continue;
    }
    const existing = groups.get(key);
    if (existing) {
      existing.tasks.push(task);
      continue;
    }
    groups.set(key, {
      key,
      label: taskCategoryLabel(task, browserMode, references),
      tasks: [task],
    });
  }
  if (browserMode === PERSONALIZATION_BROWSER_MODE) {
    for (const reference of references) {
      const mode = String(reference.interaction_mode ?? DEFAULT_INTERACTION_MODE).trim().toLowerCase();
      if (mode !== interactionMode) {
        continue;
      }
      const key = referenceCategoryKey(reference);
      if (!key || groups.has(key)) {
        continue;
      }
      groups.set(key, {
        key,
        label: personaBenchmarkCategoryLabel(key),
        tasks: [],
      });
    }
  }
  return Array.from(groups.values()).sort((left, right) => {
    if (browserMode === PERSONALIZATION_BROWSER_MODE) {
      const leftOrder = PERSONALIZATION_CATEGORY_ORDER[left.key] ?? Number.MAX_SAFE_INTEGER;
      const rightOrder = PERSONALIZATION_CATEGORY_ORDER[right.key] ?? Number.MAX_SAFE_INTEGER;
      if (leftOrder !== rightOrder) {
        return leftOrder - rightOrder;
      }
    }
    if (browserMode === SAFETY_BROWSER_MODE) {
      const leftCategoryOrder = SAFETY_CATEGORY_ORDER[left.key || "uncategorized"] ?? Number.MAX_SAFE_INTEGER;
      const rightCategoryOrder = SAFETY_CATEGORY_ORDER[right.key || "uncategorized"] ?? Number.MAX_SAFE_INTEGER;
      if (leftCategoryOrder !== rightCategoryOrder) {
        return leftCategoryOrder - rightCategoryOrder;
      }
      const leftFocusOrder = SAFETY_FOCUS_ORDER[left.key || "uncategorized"] ?? Number.MAX_SAFE_INTEGER;
      const rightFocusOrder = SAFETY_FOCUS_ORDER[right.key || "uncategorized"] ?? Number.MAX_SAFE_INTEGER;
      if (leftFocusOrder !== rightFocusOrder) {
        return leftFocusOrder - rightFocusOrder;
      }
    }
    return left.label.localeCompare(right.label);
  });
}

function inferSuiteSplitLabel(config: Record<string, unknown> | null | undefined): string | null {
  if (!config) {
    return null;
  }
  for (const key of ["episode_split", "task_split", "split"]) {
    const value = config[key];
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return null;
}

function inferSuiteToolCount(config: Record<string, unknown> | null | undefined): number | null {
  if (!config) {
    return null;
  }
  const tools = config.tools;
  if (!Array.isArray(tools) || !tools.length) {
    return null;
  }
  return tools.length;
}

function taskBriefCaption(task: TaskSummary): string {
  return task.local_dataset_only ? "dataset brief" : "task brief";
}

function taskBriefCopy(task: TaskSummary): string {
  const parts = [task.description];
  const defaultMaxTurns = inferSuiteDefaultMaxTurns(task.suite_run_config ?? null);
  const defaultMaxEpisodes = inferSuiteDefaultMaxEpisodes(task.suite_run_config ?? null) ?? task.default_max_episodes ?? null;
  const suiteSplit = inferSuiteSplitLabel(task.suite_run_config ?? null);

  if (typeof task.dataset_size === "number" && task.dataset_size > 0) {
    parts.push(
      task.supports_max_episodes
        ? `Catalog size: ${task.dataset_size} episodes.`
        : `Local mirror: ${task.dataset_size} items.`,
    );
  }
  if (task.split) {
    parts.push(`Split: ${task.split}.`);
  } else if (suiteSplit) {
    parts.push(`Suite slice: ${suiteSplit}.`);
  }
  if (task.task_mode === "answer") {
    parts.push(
      "This is an answer task: the checked-in editable file is just the repo's candidate entrypoint, and the verifier scores the returned answer/output for each item.",
    );
  } else if (task.task_mode === "artifact") {
    parts.push(
      "This is an artifact task: the returned or generated program/policy artifact is itself what the verifier runs or consumes.",
    );
  }
  parts.push(
    task.included_in_main_comparison
      ? "Included in the active benchmark task set used for direct task runs."
      : "Present in catalog metadata, but not included in the active benchmark task set.",
  );
  if (task.supports_max_episodes) {
    if (typeof defaultMaxEpisodes === "number" && defaultMaxEpisodes > 0) {
      parts.push(`Default eval cap: ${defaultMaxEpisodes} episodes.`);
    }
    if (typeof defaultMaxTurns === "number" && defaultMaxTurns > 0) {
      parts.push(`Default max turns per episode: ${defaultMaxTurns}.`);
    }
    parts.push(
      "A multi-turn benchmark run opens one episode at a time, executes the shared agent contract against the suite-owned environment, and then aggregates episode outcomes into one report.",
    );
  } else {
    parts.push(
      "A dataset run opens one item at a time, evolves code against that prompt, verifies locally, and then aggregates item outcomes into one report.",
    );
  }
  return parts.join(" ");
}

function parseCandidateMetrics(message?: string | null): {
  status: string | null;
  objective: number | null;
  primaryScore: number | null;
  passedTests: number | null;
  totalTests: number | null;
} {
  const text = String(message ?? "");
  const statusMatch = text.match(/status=([a-z]+)/i);
  const objectiveMatch = text.match(/objective=([-+]?\d+(?:\.\d+)?)/i);
  const primaryScoreMatch = text.match(/primary_score=([-+]?\d+(?:\.\d+)?)/i);
  const passedTestsMatch = text.match(/passed_tests=(\d+)/i);
  const totalTestsMatch = text.match(/total_tests=(\d+)/i);
  return {
    status: statusMatch ? statusMatch[1].toLowerCase() : null,
    objective: objectiveMatch ? Number(objectiveMatch[1]) : null,
    primaryScore: primaryScoreMatch ? Number(primaryScoreMatch[1]) : null,
    passedTests: passedTestsMatch ? Number(passedTestsMatch[1]) : null,
    totalTests: totalTestsMatch ? Number(totalTestsMatch[1]) : null,
  };
}

function eventResponseStatus(
  rawStatus: string | null | undefined,
  objective: number | null | undefined,
  output: string | null | undefined,
  objectiveSpec: ObjectiveSpec | undefined | null,
): string | null {
  const status = stringValue(rawStatus);
  if (!status) {
    return null;
  }
  if (isUnavailableStatus(status)) {
    return "fail";
  }
  const fullObjective = objectiveReachedFullScore(objective, objectiveSpec);
  if (fullObjective === true) {
    return "pass";
  }
  const objectiveValue = numeric(objective);
  const hasOutput = Boolean(stringValue(output));
  if (fullObjective === false) {
    return objectiveValue > 0 ? "partial" : "fail";
  }
  if (status === "pass") {
    return "pass";
  }
  return hasOutput ? "partial" : "fail";
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

function seededEpisodeItems(task: TaskSummary | RunTask | null | undefined): Array<{ itemId: string; itemName: string; itemBrief: string; expectedAnswer: string; displayOrder: number }> {
  const config = task?.suite_run_config;
  const rows = Array.isArray(config?.inline_episodes) ? config.inline_episodes : [];
  return rows
    .filter((row): row is Record<string, unknown> => Boolean(row) && typeof row === "object")
    .map((row, index) => {
      const itemId = stringValue(row.episode_id) ?? `${task?.id ?? "episode"}-${index + 1}`;
      const instruction = stringValue(row.instruction) ?? "Episode instruction unavailable.";
      return {
        itemId,
        itemName: itemId,
        itemBrief: instruction,
        expectedAnswer: "Solve episode",
        displayOrder: index,
      };
    });
}

function outcomeItemRuns(run: Run | undefined | null): ItemRun[] {
  if (!run) {
    return [];
  }
  if (Array.isArray(run.item_runs) && run.item_runs.length) {
    return run.item_runs;
  }
  return Array.isArray(run.winner?.metrics?.item_runs) ? run.winner.metrics.item_runs : [];
}

function toolCallDigest(toolCall: Record<string, unknown> | undefined | null): string | null {
  const name = stringValue(toolCall?.name);
  if (!name) {
    return null;
  }
  const argumentsValue = (toolCall?.arguments ?? {}) as Record<string, unknown>;
  const detail = stringValue(argumentsValue.command) ?? stringValue(argumentsValue.message);
  return detail ? `${name}(${detail})` : name;
}

function turnActionSummary(turn: Record<string, unknown> | undefined | null): string | null {
  const action = (turn?.action ?? {}) as Record<string, unknown>;
  const toolCalls = Array.isArray(action.tool_calls) ? action.tool_calls as Array<Record<string, unknown>> : [];
  const toolSummary = toolCalls.map((call) => toolCallDigest(call)).filter(Boolean).join(", ");
  const message = stringValue(action.message);
  const done = action.done === true ? "done" : null;
  const parts = [toolSummary, message, done].filter(Boolean);
  return parts.length ? parts.join(" | ") : null;
}

function turnObservationSummary(turn: Record<string, unknown> | undefined | null): string | null {
  const observation = (turn?.observation ?? {}) as Record<string, unknown>;
  return (
    stringValue(observation.text)
    ?? stringValue(observation.instruction)
    ?? stringValue(observation.hint)
    ?? null
  );
}

function itemRunPrompt(itemRun: ItemRun): string {
  return (
    stringValue(itemRun.question?.prompt)
    ?? stringValue(itemRun.payload?.instruction)
    ?? turnObservationSummary((itemRun.turns ?? [])[0] as Record<string, unknown> | undefined)
    ?? "Prompt preview unavailable."
  );
}

function rawContextBrief(rawContext: unknown): string | null {
  if (!rawContext || typeof rawContext !== "object") {
    return null;
  }
  const context = rawContext as Record<string, unknown>;
  const promptText =
    stringValue(context.query)
    ?? stringValue(context.latest_query)
    ?? stringValue(context.question_text)
    ?? stringValue(context.latest_user_message)
    ?? stringValue(context.target_utterance)
    ?? stringValue(context.user_message)
    ?? stringValue(context.question)
    ?? stringValue(context.instruction)
    ?? null;
  if (!promptText) {
    return (
      dialogueSnippet(context.dialogue)
      ?? dialogueSnippet(context.new_dialogue)
      ?? dialogueSnippet(context.old_dialogue)
      ?? null
    );
  }
  const benchmark = stringValue(context.benchmark)?.toLowerCase() ?? "";
  const roleName = stringValue(context.role_name);
  if (benchmark === "socialbench" && roleName) {
    return `${roleName}: ${promptText}`;
  }
  return promptText;
}

function promptBadgeLabel(
  taskLike: { task_shape?: string | null; scoring_mode?: string | null; usesMaxEpisodes?: boolean } | null | undefined,
  hasQuestionRecord = true,
): string {
  if (!taskLike) {
    return hasQuestionRecord ? "Question." : "Instruction.";
  }
  if ("usesMaxEpisodes" in taskLike && taskLike.usesMaxEpisodes) {
    return "Instruction.";
  }
  const taskShape = String(taskLike.task_shape ?? "").trim().toLowerCase();
  const scoringMode = String(taskLike.scoring_mode ?? "").trim().toLowerCase();
  if (scoringMode === "rubric_score" || taskShape === "agentic_open_ended" || taskShape === "dialogue_judgement" || taskShape === "dialogue_next_turn") {
    return "Prompt.";
  }
  return hasQuestionRecord ? "Question." : "Instruction.";
}

function expectedBadgeLabel(
  taskLike: { task_shape?: string | null; scoring_mode?: string | null; usesMaxEpisodes?: boolean } | null | undefined,
  hasQuestionRecord = true,
): string {
  if (!taskLike) {
    return hasQuestionRecord ? "Answer" : "Target";
  }
  if ("usesMaxEpisodes" in taskLike && taskLike.usesMaxEpisodes) {
    return "Target";
  }
  const taskShape = String(taskLike.task_shape ?? "").trim().toLowerCase();
  const scoringMode = String(taskLike.scoring_mode ?? "").trim().toLowerCase();
  if (scoringMode === "rubric_score" || taskShape === "agentic_open_ended" || taskShape === "dialogue_judgement" || taskShape === "dialogue_next_turn") {
    return "Reference";
  }
  return hasQuestionRecord ? "Answer" : "Target";
}

function itemRunBrief(itemRun: ItemRun): string {
  return (
    stringValue(itemRun.item_brief)
    ?? rawContextBrief(itemRun.question?.raw_context)
    ?? questionPreview(itemRunPrompt(itemRun), 240)
  );
}

function itemRunExpected(itemRun: ItemRun): string | null {
  if (itemRun.question?.expected_answer != null) {
    return String(itemRun.question.expected_answer);
  }
  return stringValue(itemRun.payload?.expected_answer) ?? "Solve episode";
}

function itemRunLatestEventOutput(itemRun: ItemRun): string | null {
  const latestTurn = Array.isArray(itemRun.turns) && itemRun.turns.length
    ? itemRun.turns[itemRun.turns.length - 1] as Record<string, unknown>
    : null;
  return turnActionSummary(latestTurn) ?? null;
}

function itemRunWinnerOutput(itemRun: ItemRun): string | null {
  const summary = itemRunLatestEventOutput(itemRun);
  if (summary) {
    return summary;
  }
  if (typeof itemRun.success === "boolean") {
    return itemRun.success ? "Episode solved" : "Episode not solved";
  }
  return null;
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
    const interactionMode = catalogTask?.interaction_mode ?? completedRun?.task.interaction_mode ?? DEFAULT_INTERACTION_MODE;
    const usesMaxItems = Boolean(catalogTask?.supports_max_items ?? completedRun?.task.supports_max_items ?? catalogTask?.local_dataset_only ?? completedRun?.task.local_dataset_only);
    const usesMaxEpisodes = Boolean(catalogTask?.supports_max_episodes ?? completedRun?.task.supports_max_episodes);
    const datasetSize = catalogTask?.dataset_size ?? completedRun?.task.dataset_size ?? null;
    const defaultMaxItems = catalogTask?.default_max_items ?? completedRun?.task.default_max_items ?? null;
    const defaultMaxEpisodes = catalogTask?.default_max_episodes ?? completedRun?.task.default_max_episodes ?? null;
    const requestedItems = liveJob?.max_items ?? null;
    const requestedEpisodes = liveJob?.max_episodes ?? null;
    const selectedItemIds = Array.isArray(liveJob?.item_ids) && liveJob?.item_ids.length ? liveJob.item_ids : null;
    const scheduledItems = selectedItemIds
      ? selectedItemIds.length
      : usesMaxEpisodes
      ? typeof requestedEpisodes === "number" && requestedEpisodes > 0
        ? requestedEpisodes
        : defaultMaxEpisodes
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
      interactionMode,
      taskShape: catalogTask?.task_shape ?? completedRun?.task.task_shape ?? null,
      scoringMode: catalogTask?.scoring_mode ?? completedRun?.task.scoring_mode ?? null,
      objectiveSpec: catalogTask?.objective_spec ?? completedRun?.task.objective_spec ?? { display_name: "Benchmark objective", direction: "max", summary_template: "", formula: "" },
      objectiveLabel: objectiveLabel(catalogTask?.objective_spec ?? completedRun?.task.objective_spec ?? { display_name: "Benchmark objective", direction: "max", summary_template: "", formula: "" }),
      objectiveUnit: catalogTask?.objective_spec?.unit ?? completedRun?.task.objective_spec?.unit ?? null,
      model: liveJob?.policy_model ?? liveJob?.model ?? completedRun?.policy_model ?? completedRun?.active_model ?? "n/a",
      branchingFactor:
        liveJob?.branching_factor ?? catalogTask?.branching_factor ?? completedRun?.task.branching_factor ?? 1,
      generationBudget:
        liveJob?.generation_budget ?? catalogTask?.generation_budget ?? completedRun?.task.generation_budget ?? 0,
      candidateBudget:
        liveJob?.candidate_budget ?? catalogTask?.candidate_budget ?? completedRun?.task.candidate_budget ?? 0,
      llmConcurrency: liveJob?.llm_concurrency ?? null,
      itemWorkers: liveJob?.item_workers ?? catalogTask?.item_workers ?? completedRun?.task.item_workers ?? null,
      maxItems: liveJob?.max_items ?? null,
      maxEpisodes: liveJob?.max_episodes ?? null,
      selectedItemIds,
      usesMaxItems,
      usesMaxEpisodes,
      defaultMaxItems,
      defaultMaxEpisodes,
      scheduledItems,
      roundSummaries: [],
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
      roundObjectives: new Map<number, Map<string, number>>(),
    };
    for (const seededItem of seededEpisodeItems(catalogTask ?? completedRun?.task)) {
      mutableEntry.itemsMap.set(seededItem.itemId, {
        itemKey: seededItem.itemId,
        itemId: seededItem.itemId,
        displayName: humanizeItemName(seededItem.itemName, seededItem.itemId),
        displayOrder: seededItem.displayOrder,
        status: liveJob?.status === "running" ? "running" : "queued",
        latestGeneration: 0,
        branchCount: 0,
        acceptCount: 0,
        memoryDelta: 0,
        bestObjective: null,
        latestObjective: null,
        responseObjective: null,
        itemBrief: seededItem.itemBrief,
        expectedAnswer: seededItem.expectedAnswer,
        testCaseCount: null,
        latestPassedTests: null,
        latestTotalTests: null,
        responsePassedTests: null,
        responseTotalTests: null,
        latestResponseOutput: null,
        latestResponseStatus: null,
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
      });
    }
    taskMap.set(taskId, mutableEntry);
    return mutableEntry;
  }

  function getItem(task: MutableLiveTaskCard, event: LiveEvent): MutableLiveItemCard | null {
    if (!event.item_id && (task.usesMaxItems || task.usesMaxEpisodes)) {
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
      displayOrder: nonNegativeInteger(event.item_source_index),
      status: "queued",
      latestGeneration: 0,
      branchCount: 0,
      acceptCount: 0,
      memoryDelta: 0,
      bestObjective: null,
      latestObjective: null,
      responseObjective: null,
      itemBrief: event.item_brief ?? null,
      expectedAnswer: event.expected_answer ?? null,
      testCaseCount: null,
      latestPassedTests: null,
      latestTotalTests: null,
      responsePassedTests: null,
      responseTotalTests: null,
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
      const eventItemOrder = nonNegativeInteger(event.item_source_index);
      if (eventItemOrder != null) {
        item.displayOrder = item.displayOrder == null ? eventItemOrder : Math.min(item.displayOrder, eventItemOrder);
      }
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
      const semanticStatus = eventResponseStatus(metrics.status, metrics.objective, eventResponseOutput, task.objectiveSpec);
      item.retryStates.delete(eventBranchKey(event, item.itemKey));
      item.retryLabel = summarizeRetryStates(item.retryStates);
      if (metrics.totalTests != null && metrics.totalTests > 0) {
        item.latestPassedTests = metrics.passedTests ?? 0;
        item.latestTotalTests = metrics.totalTests;
        item.testCaseCount = metrics.totalTests;
      }
      if (typeof metrics.objective === "number" && Number.isFinite(metrics.objective)) {
        item.latestObjective = metrics.objective;
        item.bestObjective = item.bestObjective == null ? metrics.objective : Math.max(item.bestObjective, metrics.objective);
      }
      if (semanticStatus) {
        item.latestResponseStatus = semanticStatus;
        if (!eventResponseOutput && semanticStatus !== "pass") {
          item.latestResponseOutput = "No candidate output captured.";
        }
      }
    }
    if (event.phase === "generation_finished" && item) {
      const metrics = parseCandidateMetrics(event.message);
      item.responseOutput =
        eventResponseOutput
        ?? item.latestResponseOutput
        ?? ((item.latestResponseStatus && item.latestResponseStatus !== "pass") ? "No candidate output captured." : null);
      item.responseStatus = eventResponseStatus(
        event.candidate_status ?? item.latestResponseStatus,
        item.bestObjective,
        item.responseOutput,
        task.objectiveSpec,
      ) ?? item.latestResponseStatus ?? item.responseStatus;
      if (metrics.totalTests != null && metrics.totalTests > 0) {
        item.responsePassedTests = metrics.passedTests ?? 0;
        item.responseTotalTests = metrics.totalTests;
        item.testCaseCount = metrics.totalTests;
      } else if (item.latestTotalTests != null && item.latestTotalTests > 0) {
        item.responsePassedTests = item.latestPassedTests ?? 0;
        item.responseTotalTests = item.latestTotalTests;
      }
      if (typeof metrics.objective === "number" && Number.isFinite(metrics.objective)) {
        item.responseObjective = metrics.objective;
      } else {
        item.responseObjective = item.latestObjective;
      }
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
      if (item && typeof event.generation === "number") {
        const objective = parseGenerationBestValue(event.message);
        if (objective != null && Number.isFinite(objective)) {
          const generationObjectives = task.roundObjectives.get(event.generation) ?? new Map<string, number>();
          generationObjectives.set(item.itemKey, objective);
          task.roundObjectives.set(event.generation, generationObjectives);
        }
      }
      if (item && item.latestGeneration >= task.generationBudget) {
        item.status = "completed";
        item.finishedAtMs = eventTimestampMs ?? item.latestEventAtMs ?? item.finishedAtMs;
      }
    }
  }

  return [...taskMap.values()]
    .map((task) => {
      const completedRun = completedRunForTask(task.taskId);
      const completedItemRuns = new Map(outcomeItemRuns(completedRun).map((itemRun) => [itemRun.item_id, itemRun]));
      const itemKeys = new Set<string>([...task.itemsMap.keys(), ...completedItemRuns.keys()]);
      const items = [...itemKeys]
        .map((itemKey) => {
          const item = task.itemsMap.get(itemKey);
          const completedItemRun = completedItemRuns.get(itemKey);
          const latestAttempt = latestAttemptedCandidate(completedItemRun);
          const latestResponseOutput =
            item?.latestResponseOutput
            ?? candidateDisplayOutput(latestAttempt, completedItemRun?.question)
            ?? null;
          const latestResponseStatus =
            item?.latestResponseStatus
            ?? candidateResponseStatus(latestAttempt, task.objectiveSpec)
            ?? null;
          const winnerResponseOutput = candidateDisplayOutput(completedItemRun?.winner, completedItemRun?.question);
          const fallbackResponseOutput =
            item?.responseOutput
            ?? candidateDisplayOutput(latestAttempt, completedItemRun?.question)
            ?? null;
          const responseOutput = winnerResponseOutput ?? fallbackResponseOutput;
          const winnerResponseStatus = candidateResponseStatus(completedItemRun?.winner, task.objectiveSpec);
          const fallbackResponseStatus =
            item?.responseStatus
            ?? candidateResponseStatus(latestAttempt, task.objectiveSpec)
            ?? null;
          const responseStatus =
            winnerResponseOutput != null
              ? (winnerResponseStatus ?? fallbackResponseStatus)
              : (fallbackResponseStatus ?? winnerResponseStatus);
          const winnerPassed = passedCandidate(completedItemRun?.winner, task.objectiveSpec);
          const winnerStatus = candidateResponseStatus(completedItemRun?.winner, task.objectiveSpec);
          const latestGeneration = item?.latestGeneration ?? completedItemRun?.generations?.length ?? 0;
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
          const displayOrder = item?.displayOrder ?? itemOrderFromItemRun(completedItemRun);
          return {
            itemKey,
            itemId: completedItemRun?.item_id ?? item?.itemId ?? itemKey,
            displayName: humanizeItemName(completedItemRun?.item_name ?? item?.displayName ?? itemKey, itemKey),
            displayOrder,
            status: itemStatus,
            latestGeneration,
            branchCount: item?.branchIds.size ?? 0,
            acceptCount: item?.acceptCount ?? 0,
            memoryDelta: item?.memoryDelta ?? 0,
            bestObjective: item?.bestObjective ?? (completedItemRun?.winner ? numeric(completedItemRun.winner.metrics.objective) : null),
            latestObjective:
              item?.latestObjective
              ?? (latestAttempt ? numeric(latestAttempt.metrics.objective_score ?? latestAttempt.metrics.objective) : null),
            responseObjective:
              item?.responseObjective
              ?? (completedItemRun?.winner ? numeric(completedItemRun.winner.metrics.objective_score ?? completedItemRun.winner.metrics.objective) : null),
            itemBrief: item?.itemBrief ?? (completedItemRun ? itemRunBrief(completedItemRun) : null),
            expectedAnswer: completedItemRun ? itemRunExpected(completedItemRun) : item?.expectedAnswer ?? null,
            testCaseCount:
              item?.testCaseCount
              ?? (completedItemRun?.winner ? numeric(completedItemRun.winner.metrics.total_tests) : null)
              ?? (latestAttempt ? numeric(latestAttempt.metrics.total_tests) : null),
            latestPassedTests:
              item?.latestPassedTests
              ?? (latestAttempt ? numeric(latestAttempt.metrics.passed_tests) : null),
            latestTotalTests:
              item?.latestTotalTests
              ?? (latestAttempt ? numeric(latestAttempt.metrics.total_tests) : null),
            responsePassedTests:
              item?.responsePassedTests
              ?? (completedItemRun?.winner ? numeric(completedItemRun.winner.metrics.passed_tests) : null),
            responseTotalTests:
              item?.responseTotalTests
              ?? (completedItemRun?.winner ? numeric(completedItemRun.winner.metrics.total_tests) : null),
            latestResponseOutput: latestResponseOutput ?? (completedItemRun ? itemRunLatestEventOutput(completedItemRun) : null),
            latestResponseStatus,
            responseOutput: responseOutput ?? (completedItemRun ? itemRunWinnerOutput(completedItemRun) : null),
            responseStatus: responseStatus ?? (typeof completedItemRun?.success === "boolean" ? (completedItemRun.success ? "pass" : "fail") : null),
            latestMessage: item?.latestMessage ?? completedItemRun?.selection_reason ?? null,
            retryLabel: item?.retryLabel ?? null,
            startedAtMs: item?.startedAtMs ?? null,
            latestEventAtMs: item?.latestEventAtMs ?? null,
            finishedAtMs,
          };
        })
        .sort(compareItemDisplayOrder);
      const roundObjectives = new Map<number, Map<string, number>>();
      for (const [generation, objectives] of task.roundObjectives.entries()) {
        roundObjectives.set(generation, new Map(objectives));
      }
      for (const [itemKey, itemRun] of completedItemRuns.entries()) {
        for (const generation of itemRun.generations ?? []) {
          if (typeof generation.generation !== "number") {
            continue;
          }
          const objective = generationObjectiveFromSummary(generation);
          if (objective == null) {
            continue;
          }
          const generationObjectives = roundObjectives.get(generation.generation) ?? new Map<string, number>();
          if (!generationObjectives.has(itemKey)) {
            generationObjectives.set(itemKey, objective);
          }
          roundObjectives.set(generation.generation, generationObjectives);
        }
      }
      const roundSummaries = [...roundObjectives.entries()]
        .sort(([left], [right]) => left - right)
        .map(([generation, objectives]) => ({
          generation,
          averageObjective: average([...objectives.values()]) ?? 0,
          sampleCount: objectives.size,
        }))
        .filter((summary) => summary.sampleCount > 0);
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
        roundSummaries,
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
  const candidateError = stringValue(candidate.metrics.error);
  const showTestMetrics = candidateHasMultiChecks(candidate);
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
          {metric(objectiveLabel(objectiveSpec), formatCandidateObjectiveValue(candidate, objectiveSpec))}
          {metric("normalized score", formatCandidateMetricValue(candidate, candidate.metrics.objective_score))}
          {metric("tie-break score", formatCandidateMetricValue(candidate, candidate.metrics.tie_break_score))}
          {metric("verifier time", candidate.metrics.benchmark_ms == null ? "n/a" : `${candidate.metrics.benchmark_ms} ms`)}
          {showTestMetrics ? metric("tests passed", `${candidate.metrics.passed_tests ?? "n/a"}/${candidate.metrics.total_tests ?? "n/a"}`) : null}
          {metric("workspace path", shortPath(candidate.workspace_path))}
        </div>
        <p className="muted">{candidate.strategy}</p>
        <p className="small">{candidate.rationale}</p>
        {candidateError ? <p className="small muted">{candidateError}</p> : null}
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
  const queuedLabel = queuedItemsLabel(task.items.length, task.totalItems);
  const recentTaskEvents = task.events
    .filter((event) => !event.item_id && stringValue(event.message))
    .slice(-6);
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
      <div className="metric-grid compact-metrics">
        {metric(task.usesMaxEpisodes ? "episodes scheduled" : "items scheduled", task.totalItems)}
        {metric(task.usesMaxEpisodes ? "episodes complete" : "items complete", `${task.completedItems}/${task.totalItems || "?"}`)}
        {metric("completion", formatPercent(completedRatio))}
        {metric(task.usesMaxEpisodes ? "episodes solved" : "items solved", `${task.passItems}/${task.totalItems || "?"}`)}
        {metric("solve rate", formatPercent(passRatio))}
        {metric("frontier accepts", task.acceptedCount)}
        {metric("memory delta", task.memoryDelta > 0 ? `+${task.memoryDelta}` : task.memoryDelta)}
      </div>
      {queuedLabel ? <p className="small muted">{queuedLabel}</p> : null}
      {task.roundSummaries.length || recentTaskEvents.length ? (
        <section className="subpanel">
          <div className="subpanel-header">
            <div>
              <p className="eyebrow">evolution trace</p>
              <h4>Current Search State</h4>
            </div>
          </div>
          <div className="badge-row">
            {task.roundSummaries.map((summary, index) => (
              <span
                className={`badge ${roundAverageTone(summary, index > 0 ? task.roundSummaries[index - 1] : null)}`}
                key={`${task.taskId}-round-${summary.generation}`}
                title={`${summary.sampleCount} item${summary.sampleCount === 1 ? "" : "s"} contributed to round ${summary.generation}.`}
              >
                {`round ${summary.generation} avg ${formatObjectiveValueFromUnit(summary.averageObjective, task.objectiveUnit)}`}
              </span>
            ))}
            {task.generationBudget > 0 ? <span className="badge">max rounds {task.generationBudget}</span> : null}
          </div>
          {recentTaskEvents.length ? (
            <ul className="dense-list compact-list">
              {recentTaskEvents.map((event, index) => (
                <li key={`${task.taskId}-event-${index}`}>{event.message}</li>
              ))}
            </ul>
          ) : null}
        </section>
      ) : null}
      <div className="live-scroll">
        {task.items.length ? (
          task.items.map((item) => {
            const responseTone = verifierTone(item.responseStatus ?? item.latestResponseStatus);
            const itemDuration = itemElapsedDuration(item, nowMs);
            const latestHasMultiChecks = hasMultiChecks(item.latestTotalTests, null);
            const winnerHasMultiChecks = hasMultiChecks(item.responseTotalTests, null);
            const latestTests = latestHasMultiChecks
              ? testProgress(item.latestPassedTests, item.latestTotalTests, item.latestObjective, task.objectiveSpec)
              : null;
            const winnerTests = winnerHasMultiChecks
              ? testProgress(item.responsePassedTests, item.responseTotalTests, item.responseObjective, task.objectiveSpec)
              : null;
            const visibleTests = winnerTests ?? latestTests;
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
                  {visibleTests ? <span className={`badge ${verifierTone(visibleTests.status)}`}>{visibleTests.label}</span> : null}
                  <span className="badge">accepted {item.acceptCount}</span>
                  {item.retryLabel ? <span className="badge warn">{item.retryLabel}</span> : null}
                </div>
              </div>
              <div className="split-grid report-grid">
                <section className="subpanel brief-panel">
                  <div className="section-label">Brief</div>
                  <p className="brief-question"><strong>{promptBadgeLabel({ usesMaxEpisodes: task.usesMaxEpisodes, task_shape: task.taskShape, scoring_mode: task.scoringMode })}</strong> {item.itemBrief ?? "Prompt brief is still loading."}</p>
                  <div className="badge-row">
                    <span className="badge">{expectedBadgeLabel({ usesMaxEpisodes: task.usesMaxEpisodes, task_shape: task.taskShape, scoring_mode: task.scoringMode })} {item.expectedAnswer ?? "n/a"}</span>
                    {item.testCaseCount != null && item.testCaseCount > 1 ? <span className="badge">{`test cases ${item.testCaseCount}`}</span> : null}
                  </div>
                </section>
                <section className={`subpanel response-panel ${responseTone}`}>
                  <div className="section-label">Response</div>
                  <div className="response-stack">
                    <div className="response-entry">
                      <div className="response-caption">Latest Candidate</div>
                      <div className={`response-value compact ${verifierTone(item.latestResponseStatus)}`}>
                        {item.latestResponseOutput ?? (item.status === "completed" ? "No event output captured." : "Waiting for the latest candidate.")}
                      </div>
                      {latestTests ? <div className="detail-summary-copy">{latestTests.label}</div> : null}
                    </div>
                    <div className="response-entry">
                      <div className="response-caption">Winner</div>
                      <div className={`response-value compact ${verifierTone(item.responseStatus)}`}>
                        {item.responseOutput ?? (item.status === "completed" ? "No winner output captured." : "Waiting for the selected candidate.")}
                      </div>
                      {winnerTests ? <div className="detail-summary-copy">{winnerTests.label}</div> : null}
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

function itemRunCard(itemRun: ItemRun, objectiveSpec: ObjectiveSpec) {
  const displayName = humanizeItemName(itemRun.item_name, itemRun.item_id);
  const questionPrompt = itemRunBrief(itemRun);
  const questionMetadata = itemRun.question?.metadata as Record<string, unknown> | undefined;
  const itemRunTaskLike = {
    task_shape: stringValue(questionMetadata?.task_shape) ?? null,
    scoring_mode: stringValue(questionMetadata?.scoring_mode) ?? null,
  };
  const baselineCandidate = firstRoundBaseline(itemRun);
  const latestAttempt = latestAttemptedCandidate(itemRun);
  const latestResponseOutput = candidateDisplayOutput(latestAttempt, itemRun.question) ?? itemRunLatestEventOutput(itemRun);
  const latestResponseStatus = candidateResponseStatus(latestAttempt, objectiveSpec);
  const winnerResponseOutput = candidateDisplayOutput(itemRun.winner, itemRun.question) ?? itemRunWinnerOutput(itemRun);
  const responseOutput = winnerResponseOutput ?? latestResponseOutput;
  const winnerResponseStatus = candidateResponseStatus(itemRun.winner, objectiveSpec) ?? (typeof itemRun.success === "boolean" ? (itemRun.success ? "pass" : "fail") : null);
  const responseStatus =
    winnerResponseOutput != null
      ? (winnerResponseStatus ?? candidateResponseStatus(latestAttempt, objectiveSpec))
      : (candidateResponseStatus(latestAttempt, objectiveSpec) ?? winnerResponseStatus);
  const baselineStatus = candidateResponseStatus(baselineCandidate, objectiveSpec) ?? "n/a";
  const finalStatus = candidateResponseStatus(itemRun.winner, objectiveSpec) ?? responseStatus ?? "n/a";
  const generationsUsed = itemRun.generations?.length ?? 0;
  const latestTests = candidateHasMultiChecks(latestAttempt)
    ? testProgress(
        latestAttempt?.metrics.passed_tests,
        latestAttempt?.metrics.total_tests,
        latestAttempt?.metrics.objective_score ?? latestAttempt?.metrics.objective,
        objectiveSpec,
      )
    : null;
  const winnerTests = candidateHasMultiChecks(itemRun.winner)
    ? testProgress(
        itemRun.winner?.metrics.passed_tests,
        itemRun.winner?.metrics.total_tests,
        itemRun.winner?.metrics.objective_score ?? itemRun.winner?.metrics.objective,
        objectiveSpec,
      )
    : null;
  const totalTests = multiCheckCount(
    itemRun.winner?.metrics.total_tests ?? latestAttempt?.metrics.total_tests,
    itemRun.winner?.metrics.test_results ?? latestAttempt?.metrics.test_results,
  );
  return (
    <article className="live-item-row" key={itemRun.item_id}>
      <div className="panel-header">
        <div className="live-item-main">
          <strong>{displayName}</strong>
          <div className="detail-summary-copy live-item-id">{itemRun.item_id}</div>
        </div>
        <div className="badge-row">
          <span className={`badge ${verifierTone(finalStatus)}`}>
            {finalStatus}
          </span>
          {winnerTests ? <span className={`badge ${verifierTone(winnerTests.status)}`}>{winnerTests.label}</span> : null}
          {baselineStatus !== finalStatus ? <span className="badge">{`baseline ${baselineStatus} -> final ${finalStatus}`}</span> : null}
          <span className="badge">{`generations ${generationsUsed}`}</span>
        </div>
      </div>
      <div className="split-grid report-grid">
        <section className="subpanel brief-panel">
          <div className="section-label">Brief</div>
          <p className="brief-question"><strong>{promptBadgeLabel(itemRunTaskLike, Boolean(itemRun.question))}</strong> {questionPrompt}</p>
          {itemRunExpected(itemRun) ? (
            <div className="badge-row">
              <span className="badge">{expectedBadgeLabel(itemRunTaskLike, Boolean(itemRun.question))} {String(itemRunExpected(itemRun))}</span>
              {totalTests > 1 ? <span className="badge">{`test cases ${totalTests}`}</span> : null}
            </div>
          ) : null}
        </section>
        <section className={`subpanel response-panel ${verifierTone(responseStatus)}`}>
          <div className="section-label">Response</div>
          <div className="response-stack">
            <div className="response-entry">
              <div className="response-caption">Latest Candidate</div>
              <div className={`response-value compact ${verifierTone(latestResponseStatus)}`}>
                {latestResponseOutput ?? "No event output captured."}
              </div>
              {latestTests ? <div className="detail-summary-copy">{latestTests.label}</div> : null}
            </div>
            <div className="response-entry">
              <div className="response-caption">Winner</div>
              <div className={`response-value compact ${verifierTone(responseStatus)}`}>
                {responseOutput ?? "No winner output captured."}
              </div>
              {winnerTests ? <div className="detail-summary-copy">{winnerTests.label}</div> : null}
            </div>
          </div>
        </section>
      </div>
      {itemRun.selection_reason ? <p className="small live-item-message">{itemRun.selection_reason}</p> : null}
      {Array.isArray(itemRun.turns) && itemRun.turns.length ? (
        <details className="detail-card">
          <summary className="detail-summary">
            <div>
              <strong>Episode trace</strong>
              <div className="detail-summary-copy">{itemRun.turns.length} turns</div>
            </div>
          </summary>
          <div className="detail-body">
            <ul className="dense-list compact-list">
              {itemRun.turns.map((turn, index) => (
                <li key={`${itemRun.item_id}-turn-${index}`}>
                  <strong>t{index}</strong> {turnObservationSummary(turn) ?? "Observation unavailable."}
                  {turnActionSummary(turn) ? ` -> ${turnActionSummary(turn)}` : ""}
                </li>
              ))}
            </ul>
          </div>
        </details>
      ) : null}
    </article>
  );
}

function runCard(run: Run, defaultSelectionSpec: SelectionSpec, isOpen: boolean, onToggle: () => void) {
  const objectiveSpec = run.task.objective_spec;
  const selectionSpec = run.selection_spec ?? run.task.selection_spec ?? defaultSelectionSpec;
  const policyModel = run.policy_model ?? run.active_model;
  const evalModel = run.eval_model ?? null;
  const itemOutcomes = outcomeItemRuns(run);
  const hasItemOutcomes = itemOutcomes.length > 0;
  const hasEvolutionTrace = Array.isArray(run.generations) && run.generations.length > 0;
  const isDatasetRun = hasItemOutcomes;
  const transitions = isDatasetRun ? datasetTransitionSummary(run) : null;
  const baselineCandidate = firstRoundBaseline(run);
  const improvement = runImprovementRatio(run);
  const winnerStatus = candidateResponseStatus(run.winner, objectiveSpec);
  const winnerUnavailable = candidateObjectiveUnavailable(run.winner);
  const baselineUnavailable = candidateObjectiveUnavailable(baselineCandidate);
  const runError = stringValue(run.winner.metrics.error);
  const totalItems = run.dataset_summary?.total_items ?? itemOutcomes.length ?? 0;
  const baselineSolved = isDatasetRun
    ? itemOutcomes.filter((itemRun) => passedCandidate(firstRoundBaseline(itemRun), objectiveSpec)).length
    : (run.dataset_summary?.baseline_passed ?? 0);
  const finalSolved = isDatasetRun
    ? itemOutcomes.filter((itemRun) => passedCandidate(itemRun.winner, objectiveSpec) || itemRun.success === true).length
    : (run.dataset_summary?.winner_passed ?? 0);
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
          <span className="badge">policy {policyModel}</span>
          {evalModel ? <span className="badge">eval {evalModel}</span> : null}
          <span className="badge">branches {run.task.branching_factor}</span>
          <span className={`badge ${verifierTone(winnerStatus)}`}>{winnerStatus ?? "n/a"}</span>
          {isDatasetRun ? (
            <>
              <span className="badge">solved {finalSolved}/{totalItems}</span>
              <span className="badge">solve rate {formatPercent(finalSolveRate)}</span>
            </>
          ) : (
            <>
              <span className="badge">
                {objectiveLabel(objectiveSpec)} {formatCandidateObjectiveValue(run.winner, objectiveSpec)}
              </span>
              <span className={`badge ${ratioTone(improvement)}`}>
                improvement {winnerUnavailable || baselineUnavailable ? "n/a" : formatMultiplier(improvement)}
              </span>
            </>
          )}
        </div>
      </button>
      {isOpen ? (
        <div className="accordion-body stack">
          {metricTemplate(objectiveSpec, selectionSpec)}
          {runError ? (
            <section className="subpanel error-panel">
              <div className="subpanel-header">
                <div>
                  <p className="eyebrow">run failure</p>
                  <h4>Verifier reported an unavailable runtime</h4>
                </div>
              </div>
              <p className="muted">{runError}</p>
            </section>
          ) : null}

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
                {metric("baseline objective", baselineCandidate ? formatCandidateObjectiveValue(baselineCandidate, objectiveSpec) : "n/a")}
                {metric("selected objective", formatCandidateObjectiveValue(run.winner, objectiveSpec))}
                {metric("objective delta", winnerUnavailable || baselineUnavailable ? "n/a" : formatObjectiveDelta(run.run_delta_objective ?? 0, objectiveSpec))}
                {metric("improvement vs baseline", winnerUnavailable || baselineUnavailable ? "n/a" : formatMultiplier(improvement))}
                {metric("generations", run.generations.length)}
                {metric("new memories", run.added_experiences?.length ?? 0)}
                {metric("memory ledger", `${run.memory_before_count ?? "n/a"} → ${run.memory_after_count ?? "n/a"}`)}
              </>
            )}
          </div>

          {hasItemOutcomes ? (
            <section className="subpanel">
              <div className="subpanel-header">
                <div>
                  <p className="eyebrow">{run.task.interaction_mode === "multi_turn" ? "episode summary" : "dataset summary"}</p>
                  <h4>{run.task.interaction_mode === "multi_turn" ? "Episode-level outcomes" : "Item-level outcomes"}</h4>
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
                {itemOutcomes.map((itemRun) => itemRunCard(itemRun, objectiveSpec))}
              </section>
            </section>
          ) : null}

          {hasEvolutionTrace ? (
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
          ) : null}

          {hasEvolutionTrace ? (
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

          {hasEvolutionTrace ? (
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

function debugModeEnabled(): boolean {
  if (typeof window === "undefined") {
    return false;
  }
  const rawValue = new URLSearchParams(window.location.search).get("debug");
  if (rawValue === null) {
    return false;
  }
  const normalized = rawValue.trim().toLowerCase();
  return normalized === "" || normalized === "1" || normalized === "true" || normalized === "yes" || normalized === "on";
}

export function App() {
  const [runtimeInfo, setRuntimeInfo] = useState<RuntimeInfo>(emptyRuntime());
  const [payload, setPayload] = useState<Payload>(emptyPayload());
  const [selectedBrowseMode, setSelectedBrowseMode] = useState(DEFAULT_BROWSE_MODE);
  const [selectedInteractionMode, setSelectedInteractionMode] = useState(DEFAULT_INTERACTION_MODE);
  const [selectedCategory, setSelectedCategory] = useState("");
  const [selectedTaskId, setSelectedTaskId] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedEvalModel, setSelectedEvalModel] = useState("");
  const [branchingFactorInput, setBranchingFactorInput] = useState("");
  const [generationBudgetInput, setGenerationBudgetInput] = useState("");
  const [candidateBudgetInput, setCandidateBudgetInput] = useState("");
  const [llmConcurrencyInput, setLlmConcurrencyInput] = useState("");
  const [itemWorkersInput, setItemWorkersInput] = useState(String(DEFAULT_FRONTEND_ITEM_WORKERS));
  const [maxItemsInput, setMaxItemsInput] = useState("");
  const [maxEpisodesInput, setMaxEpisodesInput] = useState("");
  const [maxTurnsInput, setMaxTurnsInput] = useState("");
  const [selectedRuntimeSplit, setSelectedRuntimeSplit] = useState("");
  const [selectedSkillId, setSelectedSkillId] = useState("");
  const [recordSkillEnabled, setRecordSkillEnabled] = useState(false);
  const [themePreference, setThemePreference] = useState<ThemePreference>("system");
  const [taskBriefTaskId, setTaskBriefTaskId] = useState<string | null>(null);
  const [datasetWarnings, setDatasetWarnings] = useState<DatasetWarning[]>([]);
  const [personalizationReferenceBenchmarks, setPersonalizationReferenceBenchmarks] = useState<PersonalizationReferenceBenchmark[]>([]);
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
  const showTrackedBenchmarks = useMemo(() => debugModeEnabled(), []);

  const taskBrowserTasks = useMemo(
    () => payload.task_catalog.filter((task) => task.included_in_main_comparison),
    [payload.task_catalog],
  );

  const browseModeGroups = useMemo(
    () => groupTasksByBrowseMode(taskBrowserTasks),
    [taskBrowserTasks],
  );

  const browseModeTasks = useMemo(() => {
    const filtered = taskBrowserTasks.filter((task) => browserModeForTask(task) === selectedBrowseMode);
    return filtered.length ? filtered : taskBrowserTasks;
  }, [taskBrowserTasks, selectedBrowseMode]);

  const strictBrowseModeTasks = useMemo(
    () => taskBrowserTasks.filter((task) => browserModeForTask(task) === selectedBrowseMode),
    [taskBrowserTasks, selectedBrowseMode],
  );

  const interactionGroups = useMemo(
    () => groupTasksByInteractionMode(strictBrowseModeTasks, selectedBrowseMode, personalizationReferenceBenchmarks),
    [strictBrowseModeTasks, selectedBrowseMode, personalizationReferenceBenchmarks],
  );

  const strictInteractionTasks = useMemo(
    () => strictBrowseModeTasks.filter(
      (task) => String(task.interaction_mode ?? DEFAULT_INTERACTION_MODE).trim().toLowerCase() === selectedInteractionMode,
    ),
    [strictBrowseModeTasks, selectedInteractionMode],
  );

  const categoryGroups = useMemo(
    () => groupTasksByCategory(strictInteractionTasks, selectedBrowseMode, selectedInteractionMode, personalizationReferenceBenchmarks),
    [strictInteractionTasks, selectedBrowseMode, selectedInteractionMode, personalizationReferenceBenchmarks],
  );

  const strictVisibleTasks = useMemo(() => {
    if (!selectedCategory) {
      return strictInteractionTasks;
    }
    return strictInteractionTasks.filter(
      (task) => taskCategoryKey(task, selectedBrowseMode, personalizationReferenceBenchmarks) === selectedCategory,
    );
  }, [strictInteractionTasks, selectedBrowseMode, selectedCategory, personalizationReferenceBenchmarks]);

  const visibleTasks = useMemo(() => strictVisibleTasks, [strictVisibleTasks]);

  const filteredPersonalizationReferences = useMemo(() => {
    if (selectedBrowseMode !== PERSONALIZATION_BROWSER_MODE) {
      return [];
    }
    return personalizationReferenceBenchmarks
      .filter((reference) => String(reference.interaction_mode ?? "").trim().toLowerCase() === selectedInteractionMode)
      .filter((reference) => !selectedCategory || referenceCategoryKey(reference) === selectedCategory)
      .sort((left, right) => {
        const leftStatus = referenceImplementationStatusKey(left.implementation_status);
        const rightStatus = referenceImplementationStatusKey(right.implementation_status);
        const leftOrder = leftStatus === "running" ? 0 : leftStatus === "planned" ? 1 : 2;
        const rightOrder = rightStatus === "running" ? 0 : rightStatus === "planned" ? 1 : 2;
        if (leftOrder !== rightOrder) {
          return leftOrder - rightOrder;
        }
        const leftLocal = left.status === "local_task" ? 0 : 1;
        const rightLocal = right.status === "local_task" ? 0 : 1;
        if (leftLocal !== rightLocal) {
          return leftLocal - rightLocal;
        }
        return left.title.localeCompare(right.title);
      });
  }, [personalizationReferenceBenchmarks, selectedBrowseMode, selectedInteractionMode, selectedCategory]);

  const filteredPersonalizationSourceGroups = useMemo(() => {
    const groups = new Map<string, PersonalizationReferenceSourceGroup>();
    for (const reference of filteredPersonalizationReferences) {
      const key = referenceSourceKey(reference);
      const existing = groups.get(key);
      if (existing) {
        existing.references.push(reference);
        continue;
      }
      groups.set(key, {
        key,
        sourceLabel: reference.source_label,
        sourceUrl: reference.source_url,
        mirrorSlug: reference.mirror_slug ?? null,
        references: [reference],
      });
    }
    return Array.from(groups.values()).sort((left, right) => left.sourceLabel.localeCompare(right.sourceLabel));
  }, [filteredPersonalizationReferences]);

  const personalizationTaskById = useMemo(
    () => new Map(payload.task_catalog.map((task) => [task.id, task] as const)),
    [payload.task_catalog],
  );

  const filteredPersonalizationLocalCount = useMemo(
    () => filteredPersonalizationReferences.filter((reference) => reference.status === "local_task").length,
    [filteredPersonalizationReferences],
  );

  const filteredPersonalizationExternalCount = filteredPersonalizationReferences.length - filteredPersonalizationLocalCount;
  const filteredPersonalizationSourceCount = filteredPersonalizationSourceGroups.length;
  const filteredPersonalizationDerivedTaskCount = useMemo(
    () => filteredPersonalizationReferences.reduce((total, reference) => total + referenceTaskIds(reference).length, 0),
    [filteredPersonalizationReferences],
  );
  const filteredPersonalizationRunningCount = useMemo(
    () =>
      filteredPersonalizationReferences.filter(
        (reference) => referenceImplementationStatusKey(reference.implementation_status) === "running",
      ).length,
    [filteredPersonalizationReferences],
  );
  const filteredPersonalizationPlannedCount = useMemo(
    () =>
      filteredPersonalizationReferences.filter(
        (reference) => referenceImplementationStatusKey(reference.implementation_status) === "planned",
      ).length,
    [filteredPersonalizationReferences],
  );
  const filteredPersonalizationBlockedCount = useMemo(
    () =>
      filteredPersonalizationReferences.filter(
        (reference) => referenceImplementationStatusKey(reference.implementation_status) === "blocked",
      ).length,
    [filteredPersonalizationReferences],
  );

  const selectedTask = useMemo(
    () => visibleTasks.find((task) => task.id === selectedTaskId) ?? visibleTasks[0] ?? null,
    [visibleTasks, selectedTaskId],
  );

  const taskBriefTask = useMemo(
    () => payload.task_catalog.find((task) => task.id === taskBriefTaskId) ?? null,
    [payload.task_catalog, taskBriefTaskId],
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
    if (!datasetWarningOpen && !taskBriefTask) {
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setTaskBriefTaskId(null);
        setDatasetWarningOpen(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [datasetWarningOpen, taskBriefTask]);

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
        const startupReferences = Array.isArray(tasksPayload.personalization_reference_benchmarks)
          ? tasksPayload.personalization_reference_benchmarks
          : [];
        setRuntimeInfo(runtime);
        setSelectedModel(runtime.active_model);
        setSelectedEvalModel("");
        setDatasetWarnings(startupWarnings);
        setPersonalizationReferenceBenchmarks(startupReferences);
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
        const defaultTask = tasks.find((task) => task.id === defaultTaskId) ?? tasks[0] ?? null;
        const defaultBrowserMode = browserModeForTask(defaultTask);
        setSelectedBrowseMode(defaultBrowserMode);
        setSelectedInteractionMode(String(defaultTask?.interaction_mode ?? DEFAULT_INTERACTION_MODE));
        setSelectedCategory(defaultTask ? taskCategoryKey(defaultTask, defaultBrowserMode, startupReferences) : "");
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
    setCandidateBudgetInput("");
    setItemWorkersInput(taskSupportsParallelWorkers(selectedTask) ? String(defaultParallelWorkers(selectedTask)) : "");
    setMaxItemsInput("");
    setMaxEpisodesInput("");
    setMaxTurnsInput("");
    setSelectedRuntimeSplit(selectedTask.runtime_split_selector?.default_value || "");
    setSelectedSkillId("");
    setRecordSkillEnabled(false);
    if (selectedTask.supports_eval_model) {
      setSelectedEvalModel(selectedTask.default_eval_model || selectedModel || runtimeInfo.active_model);
      return;
    }
    setSelectedEvalModel("");
  }, [selectedTask?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!selectedTask?.supports_eval_model) {
      return;
    }
    const nextEvalModel = selectedTask?.default_eval_model || selectedEvalModel || selectedModel || runtimeInfo.active_model;
    if (nextEvalModel && nextEvalModel !== selectedEvalModel) {
      setSelectedEvalModel(nextEvalModel);
    }
  }, [
    runtimeInfo.active_model,
    selectedEvalModel,
    selectedModel,
    selectedTask?.default_eval_model,
    selectedTask?.supports_eval_model,
  ]);

  useEffect(() => {
    if (!selectedTask) {
      return;
    }
    const availableSkills = Array.isArray(selectedTask.available_skills) ? selectedTask.available_skills : [];
    if (!availableSkills.some((skill) => skill.id === selectedSkillId)) {
      setSelectedSkillId("");
    }
  }, [selectedTask, selectedSkillId]);

  useEffect(() => {
    if (!browseModeGroups.length) {
      return;
    }
    if (browseModeGroups.some((group) => group.mode === selectedBrowseMode)) {
      return;
    }
    setSelectedBrowseMode(browseModeGroups[0].mode);
  }, [browseModeGroups, selectedBrowseMode]);

  useEffect(() => {
    if (!interactionGroups.length) {
      return;
    }
    if (interactionGroups.some((group) => group.mode === selectedInteractionMode)) {
      return;
    }
    setSelectedInteractionMode(interactionGroups[0].mode);
  }, [interactionGroups, selectedInteractionMode]);

  useEffect(() => {
    if (!categoryGroups.length) {
      return;
    }
    if (categoryGroups.some((group) => group.key === selectedCategory)) {
      return;
    }
    setSelectedCategory(categoryGroups[0].key);
  }, [categoryGroups, selectedCategory]);

  useEffect(() => {
    if (!visibleTasks.length) {
      return;
    }
    if (visibleTasks.some((task) => task.id === selectedTaskId)) {
      return;
    }
    setSelectedTaskId(visibleTasks[0].id);
  }, [visibleTasks, selectedTaskId]);

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
    const evalModel = selectedTask?.supports_eval_model
      ? selectedEvalModel || selectedTask?.default_eval_model || model
      : null;
    const runtimeLlmConcurrencyDefault = defaultLlmConcurrency(runtimeInfo);
    const isDatasetTask = isLocalDatasetTask(selectedTask);
    const supportsParallelWorkers = taskSupportsParallelWorkers(selectedTask);
    const supportsMaxItems = Boolean(selectedTask?.supports_max_items);
    const supportsMaxEpisodes = Boolean(selectedTask?.supports_max_episodes);
    const branchingFactor = parsedBranchingFactor ?? selectedTaskBranchingFactorDefault;
    const generationBudget = parsedGenerationBudget ?? selectedTaskGenerationBudgetDefault;
    const requestedCandidateBudget = candidateBudgetInput.trim()
      ? Math.max(1, Math.floor(numeric(candidateBudgetInput)))
      : null;
    const candidateBudget = requestedCandidateBudget ?? selectedTaskCandidateBudgetDefault;
    const itemWorkers = supportsParallelWorkers
      ? Math.max(1, Math.floor(numeric(itemWorkersInput || defaultParallelWorkers(selectedTask))))
      : null;
    const requestedLlmConcurrency = llmConcurrencyInput.trim()
      ? Math.max(1, Math.floor(numeric(llmConcurrencyInput)))
      : null;
    const llmConcurrency = resolvedLlmConcurrency(requestedLlmConcurrency, runtimeLlmConcurrencyDefault, itemWorkers);
    const chosenSkillId = isDatasetTask && selectedSkillId.trim() ? selectedSkillId.trim() : null;
    const recordSkill = isDatasetTask ? recordSkillEnabled : false;
    const maxItems = supportsMaxItems && maxItemsInput.trim() ? Math.max(1, Math.floor(numeric(maxItemsInput))) : null;
    const maxEpisodes = supportsMaxEpisodes && maxEpisodesInput.trim() ? Math.max(1, Math.floor(numeric(maxEpisodesInput))) : null;
    const maxTurns = supportsMaxEpisodes && maxTurnsInput.trim() ? Math.max(1, Math.floor(numeric(maxTurnsInput))) : null;
    const suiteConfigEntries: Record<string, unknown> = {};
    if (selectedTask?.runtime_split_selector) {
      const selectedSplit = selectedRuntimeSplit || selectedTask.runtime_split_selector.default_value;
      if (selectedSplit) {
        suiteConfigEntries.split = selectedSplit;
      }
    }
    if (selectedTask?.supports_runtime_config && maxTurns != null) {
      suiteConfigEntries.max_turns = maxTurns;
    }
    const suiteConfig = Object.keys(suiteConfigEntries).length ? suiteConfigEntries : null;
    pollToken.current += 1;
    const token = pollToken.current;
    setError(null);
    setLiveJob({
      status: "running",
      taskId,
      model,
      policy_model: model,
      eval_model: evalModel,
      branching_factor: branchingFactor,
      generation_budget: generationBudget,
      candidate_budget: candidateBudget,
      llm_concurrency: llmConcurrency,
      item_workers: itemWorkers,
      max_items: maxItems,
      max_episodes: maxEpisodes,
      events: [
        {
          phase: "queued",
          message: `Launching ${taskId ?? "selected task"} on ${model} ` +
            `${evalModel ? `[eval ${evalModel}] ` : ""}` +
            `(gen=${generationBudget}, proposal-calls/branch=${candidateBudget}, branches=${branchingFactor}, llm-concurrency=${llmConcurrency}` +
            `${itemWorkers ? `, parallel-eval=${itemWorkers}` : ""}` +
            `${maxEpisodes ? `, episode count=${maxEpisodes}` : maxItems ? `, item cap=${maxItems}` : ""}` +
            `${maxTurns ? `, max-turns/episode=${maxTurns}` : ""}).`,
        },
      ],
    });

    try {
        const start = await startJob(taskId, model, {
          branchingFactor,
          generationBudget,
          candidateBudget,
          evalModel,
          llmConcurrency,
          itemWorkers,
          maxItems,
          maxEpisodes,
          suiteConfig,
          recordSkill,
          selectedSkillId: chosenSkillId,
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
        policy_model: model,
        eval_model: evalModel,
        branching_factor: parsedBranchingFactor ?? selectedTaskBranchingFactorDefault,
        generation_budget: parsedGenerationBudget ?? selectedTaskGenerationBudgetDefault,
        candidate_budget:
          (candidateBudgetInput.trim() ? Math.max(1, Math.floor(numeric(candidateBudgetInput))) : null) ??
          selectedTaskCandidateBudgetDefault,
        llm_concurrency:
          (llmConcurrencyInput.trim() ? Math.max(1, Math.floor(numeric(llmConcurrencyInput))) : null) ??
          runtimeLlmConcurrencyDefault,
        item_workers: supportsParallelWorkers ? Math.max(1, Math.floor(numeric(itemWorkersInput || defaultParallelWorkers(selectedTask)))) : null,
        max_episodes: supportsMaxEpisodes && maxEpisodesInput.trim() ? Math.max(1, Math.floor(numeric(maxEpisodesInput))) : null,
        events: [],
      });
    }
  }

  const defaultSelectionSpec = selectedTask?.selection_spec ?? payload.task_catalog[0]?.selection_spec ?? emptySelectionSpec();
  const selectedTaskDefaultMaxItems = selectedTask?.local_dataset_only
    ? selectedTask.dataset_size ?? null
    : inferSuiteDefaultMaxItems(selectedTask?.suite_run_config ?? null) ?? selectedTask?.default_max_items ?? null;
  const selectedTaskDefaultMaxEpisodes =
    inferSuiteDefaultMaxEpisodes(selectedTask?.suite_run_config ?? null) ?? selectedTask?.default_max_episodes ?? null;
  const selectedTaskDefaultMaxTurns = inferSuiteDefaultMaxTurns(selectedTask?.suite_run_config ?? null);
  const selectedTaskEpisodeDatasetSize = selectedTask?.supports_max_episodes ? selectedTask?.dataset_size ?? null : null;
  const selectedTaskUsesMaxItems = Boolean(selectedTask?.supports_max_items);
  const selectedTaskUsesMaxEpisodes = Boolean(selectedTask?.supports_max_episodes);
  const selectedTaskSupportsParallelWorkers = taskSupportsParallelWorkers(selectedTask);
  const selectedTaskSupportsEvalModel = Boolean(selectedTask?.supports_eval_model);
  const selectedTaskRequiresEvalModel = Boolean(selectedTask?.requires_eval_model);
  const selectedTaskDefaultEvalModel = selectedTask?.default_eval_model ?? null;
  const selectedTaskIsDataset = isLocalDatasetTask(selectedTask);
  const selectedTaskIsCoding = selectedTask?.track === "coding_verified";
  const selectedTaskAvailableSkills = selectedTask?.available_skills ?? [];
  const showUseSkillField = selectedTaskIsDataset && selectedTaskAvailableSkills.length > 0;
  const selectedTaskBranchingFactorDefault = Math.max(
    1,
    Math.floor(numeric(selectedTask?.branching_factor ?? DEFAULT_FRONTEND_BRANCHING_FACTOR)),
  );
  const selectedTaskGenerationBudgetDefault = Math.max(
    1,
    Math.floor(numeric(selectedTask?.generation_budget ?? DEFAULT_FRONTEND_GENERATION_BUDGET)),
  );
  const selectedTaskCandidateBudgetDefault = Math.max(
    1,
    Math.floor(numeric(selectedTask?.candidate_budget ?? DEFAULT_FRONTEND_CANDIDATE_BUDGET)),
  );
  const selectedTaskParallelWorkerDefault = defaultParallelWorkers(selectedTask);
  const selectedTaskLlmConcurrencyDefault = defaultLlmConcurrency(runtimeInfo);
  const selectedTaskRuntimeSplitSelector = selectedTask?.runtime_split_selector ?? null;
  const selectedTaskSupportsRuntimeSplit = Boolean(selectedTaskRuntimeSplitSelector);
  const selectedTaskRuntimeSplitOptions = selectedTaskRuntimeSplitSelector?.options ?? [];
  const effectiveRuntimeSplit = selectedTaskSupportsRuntimeSplit
    ? selectedRuntimeSplit || selectedTaskRuntimeSplitSelector?.default_value || ""
    : "";
  const selectedTaskRuntimeSplitOption = selectedTaskRuntimeSplitOptions.find((option) => option.value === effectiveRuntimeSplit) ?? null;
  const selectedTaskVisibleDatasetSize = selectedTaskRuntimeSplitOption?.item_count ?? selectedTask?.dataset_size ?? null;
  const selectedTaskEvalLimitUnitLabel = selectedTask?.eval_limit_unit_label ?? (selectedTaskIsCoding ? "problems" : "items");
  const selectedTaskRuntimeSplitHelp = selectedTask?.runtime_split_help ?? null;
  const parsedMaxItems = selectedTaskUsesMaxItems && maxItemsInput.trim() ? Math.max(1, Math.floor(numeric(maxItemsInput))) : null;
  const parsedMaxEpisodes = selectedTaskUsesMaxEpisodes && maxEpisodesInput.trim() ? Math.max(1, Math.floor(numeric(maxEpisodesInput))) : null;
  const parsedMaxTurns = selectedTaskUsesMaxEpisodes && maxTurnsInput.trim() ? Math.max(1, Math.floor(numeric(maxTurnsInput))) : null;
  const parsedBranchingFactor = branchingFactorInput.trim() ? Math.max(1, Math.floor(numeric(branchingFactorInput))) : null;
  const parsedGenerationBudget = generationBudgetInput.trim() ? Math.max(1, Math.floor(numeric(generationBudgetInput))) : null;
  const parsedCandidateBudget = candidateBudgetInput.trim() ? Math.max(1, Math.floor(numeric(candidateBudgetInput))) : null;
  const parsedLlmConcurrency = llmConcurrencyInput.trim() ? Math.max(1, Math.floor(numeric(llmConcurrencyInput))) : null;
  const effectiveBranchingFactor = parsedBranchingFactor ?? selectedTaskBranchingFactorDefault;
  const effectiveGenerationBudget = parsedGenerationBudget ?? selectedTaskGenerationBudgetDefault;
  const effectiveCandidateBudget = parsedCandidateBudget ?? selectedTaskCandidateBudgetDefault;
  const effectiveParallelWorkers = selectedTaskSupportsParallelWorkers
    ? Math.max(1, Math.floor(numeric(itemWorkersInput || selectedTaskParallelWorkerDefault)))
    : null;
  const effectiveLlmConcurrency = resolvedLlmConcurrency(
    parsedLlmConcurrency,
    selectedTaskLlmConcurrencyDefault,
    effectiveParallelWorkers,
  );
  const effectiveEvalModel = selectedTaskSupportsEvalModel
    ? selectedEvalModel || selectedTaskDefaultEvalModel || selectedModel || runtimeInfo.active_model
    : null;
  const requestedEvalCount = selectedTaskUsesMaxEpisodes
    ? parsedMaxEpisodes ?? selectedTaskDefaultMaxEpisodes ?? selectedTaskEpisodeDatasetSize ?? null
    : selectedTaskUsesMaxItems
      ? parsedMaxItems ?? selectedTaskVisibleDatasetSize ?? selectedTaskDefaultMaxItems ?? null
      : null;
  const showDemoCodingWarning = Boolean(selectedTaskIsCoding && requestedEvalCount && requestedEvalCount > 50);
  const evalLimitLabel = "Eval Limit";
  const maxItemsLabel = selectedTaskUsesMaxItems && selectedTaskVisibleDatasetSize
    ? `${evalLimitLabel} (full dataset = ${selectedTaskVisibleDatasetSize})`
    : selectedTaskUsesMaxItems
      ? selectedTaskDefaultMaxItems
        ? `${evalLimitLabel} (default = ${selectedTaskDefaultMaxItems})`
        : evalLimitLabel
      : evalLimitLabel;
  const maxItemsHelper = selectedTaskUsesMaxItems && selectedTaskVisibleDatasetSize
    ? parsedMaxItems
      ? `The first ${parsedMaxItems} ${selectedTaskEvalLimitUnitLabel} will be used for eval.`
      : `Blank uses the full dataset. The first ${selectedTaskVisibleDatasetSize} ${selectedTaskEvalLimitUnitLabel} will be used for eval.`
    : selectedTaskUsesMaxItems
      ? parsedMaxItems
        ? `The first ${parsedMaxItems} tasks will be used for eval.`
        : selectedTaskDefaultMaxItems
          ? `Blank uses the current default. The first ${selectedTaskDefaultMaxItems} tasks will be used for eval.`
          : "Blank uses the current default task subset for eval."
      : "Single-item task: cap disabled";
  const maxEpisodesLabel = selectedTaskEpisodeDatasetSize
    ? `${evalLimitLabel} (full dataset = ${selectedTaskEpisodeDatasetSize})`
    : selectedTaskDefaultMaxEpisodes
      ? `${evalLimitLabel} (default = ${selectedTaskDefaultMaxEpisodes})`
      : evalLimitLabel;
  const maxEpisodesHelper = parsedMaxEpisodes
    ? `The first ${parsedMaxEpisodes} episodes/problems will be used for eval.`
    : selectedTaskEpisodeDatasetSize && selectedTaskDefaultMaxEpisodes
      ? `Blank uses the current default of ${selectedTaskDefaultMaxEpisodes}. The full dataset contains ${selectedTaskEpisodeDatasetSize} episodes/problems.`
      : selectedTaskEpisodeDatasetSize
        ? `Blank uses the full dataset. The first ${selectedTaskEpisodeDatasetSize} episodes/problems will be used for eval.`
        : selectedTaskDefaultMaxEpisodes
          ? `Blank uses the current default of ${selectedTaskDefaultMaxEpisodes} episodes/problems.`
          : "Set how many episodes/problems to use for eval in this run.";
  const maxTurnsLabel = selectedTaskDefaultMaxTurns
    ? `Max Turns per Episode (default = ${selectedTaskDefaultMaxTurns})`
    : "Max Turns per Episode";
  const maxTurnsHelper = parsedMaxTurns
    ? `Each episode will stop after ${parsedMaxTurns} environment/API turns unless it finishes earlier.`
    : selectedTaskDefaultMaxTurns
      ? `Blank uses the current default of ${selectedTaskDefaultMaxTurns} turns per episode.`
      : "Cap how many environment/API turns each episode can take.";
  const skillSelectHelper = selectedTaskAvailableSkills.length
    ? `${selectedTaskAvailableSkills.length} distilled skill${selectedTaskAvailableSkills.length === 1 ? "" : "s"} available for this dataset.`
    : "No distilled skill recorded for this dataset yet.";
  const skillRecordHelper = recordSkillEnabled
    ? "After the run, all completed item memories from this eval prefix will be distilled into a reusable markdown skill."
    : "Leave unchecked to skip recording a distilled skill artifact for this run.";
  const parallelEvalHelper = selectedTaskUsesMaxEpisodes
    ? "How many episodes to evaluate in parallel for this run."
    : "How many items/questions to evaluate in parallel for this run.";
  const candidateBudgetHelper = parsedCandidateBudget
    ? `This run will issue ${parsedCandidateBudget} independent proposal requests per branch.`
    : `Blank uses the task default of ${selectedTaskCandidateBudgetDefault} independent proposal request${selectedTaskCandidateBudgetDefault === 1 ? "" : "s"} per branch.`;
  const llmConcurrencyHelper = parsedLlmConcurrency
    ? `This run will allow up to ${parsedLlmConcurrency} LLM requests in flight at once.`
    : effectiveParallelWorkers
      ? `Blank follows Parallel Eval and uses ${effectiveLlmConcurrency} in-flight LLM requests for this run (runtime default: ${selectedTaskLlmConcurrencyDefault}).`
      : `Blank uses the current runtime default of ${selectedTaskLlmConcurrencyDefault} in-flight LLM requests.`;
  const searchRoundsSummary = `${effectiveGenerationBudget} rounds`;
  const proposalCallsSummary = `${effectiveCandidateBudget} proposal calls/branch`;
  const llmConcurrencySummary = `${effectiveLlmConcurrency} reqs`;
  const evalModelSummary = selectedTaskSupportsEvalModel
    ? effectiveEvalModel || (selectedTaskRequiresEvalModel ? "required" : "optional")
    : "Not used";
  const frontierParentsSummary = `${effectiveBranchingFactor} frontier parents`;
  const evalScopeSummary = selectedTaskUsesMaxEpisodes
    ? parsedMaxEpisodes
      ? `First ${parsedMaxEpisodes} episodes`
      : selectedTaskDefaultMaxEpisodes
        ? `First ${selectedTaskDefaultMaxEpisodes} episodes`
        : "Task episodes"
    : selectedTaskUsesMaxItems
      ? parsedMaxItems
        ? `First ${parsedMaxItems} ${selectedTaskEvalLimitUnitLabel}`
        : selectedTaskDefaultMaxItems
          ? `First ${selectedTaskDefaultMaxItems} ${selectedTaskEvalLimitUnitLabel}`
          : "Task subset"
      : "Single item";
  const parallelismSummary = selectedTaskSupportsParallelWorkers
    ? `${itemWorkersInput || selectedTaskParallelWorkerDefault} evals`
    : "Backend managed";
  const selectedPersonaCategoryLabel = selectedCategory ? personaBenchmarkCategoryLabel(selectedCategory) : "Uncategorized";
  const selectedPersonaTurnLabel = interactionModeLabel(selectedInteractionMode);
  const personalizationSliceHasLocalTask = strictVisibleTasks.length > 0;
  const taskBrowserEmptyMessage = useMemo(() => {
    if (visibleTasks.length) {
      return null;
    }
    if (selectedBrowseMode === PERSONALIZATION_BROWSER_MODE) {
      return `No runnable local tasks match ${selectedPersonaTurnLabel} / ${selectedPersonaCategoryLabel}.`;
    }
    return "No runnable local tasks match the current filters.";
  }, [visibleTasks.length, selectedBrowseMode, selectedPersonaTurnLabel, selectedPersonaCategoryLabel]);

  return (
    <main className="app-shell">
      <section className="topbar">
        <div>
          <strong className="topbar-title">EvoSkill Foundry 😎</strong>
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
            <h2>Launch Controls</h2>
          </div>
        </div>

        <div className="stack launcher-panel-stack">
          <section className="subpanel">
            <div className="subpanel-header">
              <div>
                <p className="eyebrow">TASK SELECTION</p>
              </div>
            </div>
            <div className="stack">
              <div className="control-grid triple">
                <label className="field">
                  <span className="field-label">Browse Mode</span>
                  <select className="control" value={selectedBrowseMode} onChange={(event) => setSelectedBrowseMode(event.target.value)}>
                    {browseModeGroups.map((group) => (
                      <option key={group.mode} value={group.mode}>
                        {group.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="field">
                  <span className="field-label">Turn Mode</span>
                  <select className="control" value={selectedInteractionMode} onChange={(event) => setSelectedInteractionMode(event.target.value)}>
                    {interactionGroups.map((group) => (
                      <option key={group.mode} value={group.mode}>
                        {group.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="field">
                  <span className="field-label">Category</span>
                  <select className="control" value={selectedCategory} onChange={(event) => setSelectedCategory(event.target.value)}>
                    {categoryGroups.map((group) => (
                      <option key={group.key} value={group.key}>
                        {group.label}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="field field-span-full">
                  <span className="field-label">Task Browser</span>
                  {visibleTasks.length ? (
                    <div className="task-picker-grid">
                      {visibleTasks.map((task) => {
                        const active = selectedTask?.id === task.id;
                        return (
                          <article
                            key={task.id}
                            className={`task-picker-card ${active ? "active" : ""}`}
                            onClick={() => setSelectedTaskId(task.id)}
                            onKeyDown={(event) => {
                              if (event.key === "Enter" || event.key === " ") {
                                event.preventDefault();
                                setSelectedTaskId(task.id);
                              }
                            }}
                            role="button"
                            tabIndex={0}
                          >
                            <div className="task-picker-header">
                              <div className="task-picker-copy">
                                <strong>{task.title}</strong>
                                <span className="small muted">{task.description}</span>
                              </div>
                              <div className="task-picker-actions">
                                <button
                                  className="action subtle"
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    setTaskBriefTaskId(task.id);
                                  }}
                                  type="button"
                                >
                                  Open brief
                                </button>
                              </div>
                            </div>
                          </article>
                        );
                      })}
                    </div>
                  ) : (
                    <div className="empty-state">
                      <h3>No tasks in this slice</h3>
                      <p className="muted">{taskBrowserEmptyMessage}</p>
                    </div>
                  )}
                </div>
              </div>
              {showTrackedBenchmarks && selectedBrowseMode === PERSONALIZATION_BROWSER_MODE && filteredPersonalizationReferences.length ? (
                <details className="detail-card" open>
                  <summary className="detail-summary">
                    <div>
                      <strong>Tracked Benchmarks</strong>
                      <div className="detail-summary-copy">
                        {selectedPersonaTurnLabel} / {selectedPersonaCategoryLabel} slice with {filteredPersonalizationReferences.length} tracked benchmark
                        {filteredPersonalizationReferences.length === 1 ? "" : "s"} from {filteredPersonalizationSourceCount} source suite
                        {filteredPersonalizationSourceCount === 1 ? "" : "s"}.
                        {!personalizationSliceHasLocalTask ? " No runnable local task is wired for this slice yet." : ""}
                      </div>
                    </div>
                    <div className="badge-row">
                      <span className="badge">{filteredPersonalizationReferences.length} tracked benchmarks</span>
                      <span className="badge">{filteredPersonalizationSourceCount} source suites</span>
                      <span className="badge">{filteredPersonalizationDerivedTaskCount} local tasks</span>
                      <span className="badge good">{filteredPersonalizationRunningCount} running</span>
                      <span className="badge warn">{filteredPersonalizationPlannedCount} planned</span>
                      <span className="badge bad">{filteredPersonalizationBlockedCount} blocked</span>
                      <span className="badge">{filteredPersonalizationLocalCount} local benchmarks</span>
                      <span className="badge">{filteredPersonalizationExternalCount} reference-only</span>
                    </div>
                  </summary>
                  <div className="detail-body">
                    <ul className="dense-list compact-list">
                      {filteredPersonalizationSourceGroups.map((group) => (
                        <li key={group.key}>
                          <div className="badge-row">
                            <strong>{group.sourceLabel}</strong>
                            <span className="badge">{group.references.length} benchmark{group.references.length === 1 ? "" : "s"}</span>
                            {group.mirrorSlug ? <span className="badge">mirror {group.mirrorSlug}</span> : null}
                          </div>
                          <a className="text-link small" href={group.sourceUrl} target="_blank" rel="noreferrer">
                            {group.sourceUrl}
                          </a>
                          <ul className="dense-list compact-list">
                            {group.references.map((reference) => {
                              const localTaskIds = referenceTaskIds(reference);
                              return (
                                <li key={reference.id}>
                                  <div className="badge-row">
                                    <strong>{reference.title}</strong>
                                    <span className={`badge ${referenceImplementationStatusTone(reference.implementation_status)}`}>
                                      {referenceImplementationStatusLabel(reference.implementation_status)}
                                    </span>
                                    <span className={`badge ${referenceStatusTone(reference.status)}`}>{referenceStatusLabel(reference.status)}</span>
                                    <span className="badge">{personalizationCategoryLabel(reference.benchmark_category)}</span>
                                    <span className="badge">{personaBenchmarkCategoryLabel(reference.primary_category)}</span>
                                    {localTaskIds.length ? (
                                      <span className="badge">{localTaskIds.length} local task{localTaskIds.length === 1 ? "" : "s"}</span>
                                    ) : null}
                                    <span className="badge">{taskShapeLabel(reference.task_shape)}</span>
                                    <span className="badge">{scoringModeLabel(reference.scoring_mode)}</span>
                                    <span className="badge">{officialMetricBackendLabel(reference.official_metric_backend)}</span>
                                    <span className="badge">{metricFidelityLabel(reference.metric_fidelity)}</span>
                                    {reference.requires_eval_model ? <span className="badge warn">eval model required</span> : null}
                                  </div>
                                  <div className="small">{reference.focus}</div>
                                  <div className="small">{reference.summary}</div>
                                  <div className="small muted">
                                    Official metric: {reference.official_metric_name} · {officialMetricBackendLabel(reference.official_metric_backend)} · {officialMetricGranularityLabel(reference.official_metric_granularity)} · {metricFidelityLabel(reference.metric_fidelity)}.
                                  </div>
                                  <div className="small muted">Protocol: {reference.protocol_summary}</div>
                                  <div className="small muted">Current wave: {reference.implementation_note}</div>
                                  {reference.secondary_categories?.length ? (
                                    <div className="badge-row">
                                      {reference.secondary_categories.map((category) => (
                                        <span key={`${reference.id}:secondary:${category}`} className="badge">
                                          {personalizationSecondaryCategoryLabel(category)}
                                        </span>
                                      ))}
                                    </div>
                                  ) : null}
                                  {reference.official_dimensions?.length ? (
                                    <div className="badge-row">
                                      {reference.official_dimensions.map((dimension) => (
                                        <span key={`${reference.id}:dimension:${dimension}`} className="badge">
                                          {dimension}
                                        </span>
                                      ))}
                                    </div>
                                  ) : null}
                                  {reference.subject_domains?.length ? (
                                    <div className="badge-row">
                                      {reference.subject_domains.map((domain) => (
                                        <span key={`${reference.id}:domain:${domain}`} className="badge">
                                          {subjectDomainLabel(domain)}
                                        </span>
                                      ))}
                                    </div>
                                  ) : null}
                                  {reference.blocking_reason ? <div className="small muted">Blocked: {reference.blocking_reason}</div> : null}
                                  {localTaskIds.length ? (
                                    <ul className="dense-list compact-list">
                                      {localTaskIds.map((taskId) => {
                                        const localTask = personalizationTaskById.get(taskId) ?? null;
                                        return (
                                          <li key={`${reference.id}:${taskId}`}>
                                            <div className="badge-row">
                                              <strong>{localTask?.title ?? taskId}</strong>
                                              <span className="badge">{taskId}</span>
                                              {typeof localTask?.dataset_size === "number" && localTask.dataset_size > 0 ? (
                                                <span className="badge">{localTask.dataset_size} items</span>
                                              ) : null}
                                              {localTask?.split ? <span className="badge">{localTask.split}</span> : null}
                                            </div>
                                          </li>
                                        );
                                      })}
                                    </ul>
                                  ) : null}
                                </li>
                              );
                            })}
                          </ul>
                        </li>
                      ))}
                    </ul>
                  </div>
                </details>
              ) : null}
            </div>
          </section>

          <section className="subpanel">
            <div className="subpanel-header">
              <div>
                <p className="eyebrow">TASK CONFIGURATION</p>
              </div>
            </div>
            <div className="control-grid triple">
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
              {selectedTaskSupportsEvalModel ? (
                <label className="field">
                  <span className="field-label">Eval Model</span>
                  <select
                    className="control"
                    value={effectiveEvalModel ?? ""}
                    onChange={(event) => setSelectedEvalModel(event.target.value)}
                    disabled={!runtimeInfo.available_models.length}
                  >
                    {runtimeInfo.available_models.map((model) => (
                      <option key={`eval-${model}`} value={model}>
                        {model}
                      </option>
                    ))}
                  </select>
                  <span className="small muted">
                    {selectedTaskRequiresEvalModel
                      ? "Used for judge/reward scoring. The policy model remains the benchmarked model."
                      : "Optional evaluator for judge/reward metrics. The policy model remains the benchmarked model."}
                  </span>
                </label>
              ) : null}
              <label className="field">
                <span className="field-label">Frontier Parents</span>
                <input
                  className="control"
                  type="number"
                  min={1}
                  step={1}
                  placeholder={String(selectedTaskBranchingFactorDefault)}
                  value={branchingFactorInput}
                  onChange={(event) => setBranchingFactorInput(event.target.value)}
                />
              </label>
              <label className="field">
                <span className="field-label">Max Evolve Rounds</span>
                <input
                  className="control"
                  type="number"
                  min={1}
                  step={1}
                  placeholder={String(selectedTaskGenerationBudgetDefault)}
                  value={generationBudgetInput}
                  onChange={(event) => setGenerationBudgetInput(event.target.value)}
                />
              </label>
              <label className="field">
                <span className="field-label">Proposal Calls per Branch</span>
                <input
                  className="control"
                  type="number"
                  min={1}
                  step={1}
                  placeholder={String(selectedTaskCandidateBudgetDefault)}
                  value={candidateBudgetInput}
                  onChange={(event) => setCandidateBudgetInput(event.target.value)}
                />
                <span className="small muted">{candidateBudgetHelper}</span>
              </label>
              <label className="field">
                <span className="field-label">LLM Concurrency</span>
                <input
                  className="control"
                  type="number"
                  min={1}
                  step={1}
                  placeholder={String(selectedTaskLlmConcurrencyDefault)}
                  value={llmConcurrencyInput}
                  onChange={(event) => setLlmConcurrencyInput(event.target.value)}
                />
                <span className="small muted">{llmConcurrencyHelper}</span>
              </label>
              <label className="field">
                <span className="field-label">Parallel Eval</span>
                <input
                  className="control"
                  type="number"
                  min={1}
                  step={1}
                  value={itemWorkersInput}
                  onChange={(event) => setItemWorkersInput(event.target.value)}
                  disabled={!selectedTaskSupportsParallelWorkers}
                  placeholder={selectedTaskSupportsParallelWorkers ? String(selectedTaskParallelWorkerDefault) : "n/a"}
                />
                <span className="small muted">
                  {selectedTaskSupportsParallelWorkers ? parallelEvalHelper : "This task uses its own backend-specific scheduling."}
                </span>
              </label>
              {selectedTaskSupportsRuntimeSplit ? (
                <label className="field">
                  <span className="field-label">{selectedTaskRuntimeSplitSelector?.label || "Split"}</span>
                  <select
                    className="control"
                    value={effectiveRuntimeSplit}
                    onChange={(event) => setSelectedRuntimeSplit(event.target.value)}
                  >
                    {selectedTaskRuntimeSplitOptions.map((option) => (
                      <option key={`runtime-split-${option.value}`} value={option.value}>
                        {option.title}
                      </option>
                    ))}
                  </select>
                  <span className="small muted">
                    {selectedTaskRuntimeSplitOption?.description ||
                      selectedTaskRuntimeSplitHelp ||
                      (selectedTaskVisibleDatasetSize
                        ? `This split exposes ${selectedTaskVisibleDatasetSize} ${selectedTaskEvalLimitUnitLabel} for eval.`
                        : "Choose the dataset slice to evaluate.")}
                  </span>
                </label>
              ) : null}
              {selectedTaskUsesMaxItems ? (
                <label className="field">
                  <span className="field-label">{maxItemsLabel}</span>
                  <input
                  className="control"
                  type="number"
                  min={1}
                  step={1}
                  placeholder={selectedTaskUsesMaxItems ? (selectedTaskVisibleDatasetSize ? "all" : "default") : "n/a"}
                  value={maxItemsInput}
                  onChange={(event) => setMaxItemsInput(event.target.value)}
                  disabled={!selectedTaskUsesMaxItems}
                  />
                  <span className="small muted">{maxItemsHelper}</span>
                </label>
              ) : null}
              {selectedTaskUsesMaxEpisodes ? (
                <label className="field">
                  <span className="field-label">{maxEpisodesLabel}</span>
                  <input
                    className="control"
                    type="number"
                    min={1}
                    step={1}
                    placeholder="default"
                    value={maxEpisodesInput}
                    onChange={(event) => setMaxEpisodesInput(event.target.value)}
                  />
                  <span className="small muted">{maxEpisodesHelper}</span>
                </label>
              ) : null}
              {selectedTaskUsesMaxEpisodes ? (
                <label className="field">
                  <span className="field-label">{maxTurnsLabel}</span>
                  <input
                    className="control"
                    type="number"
                    min={1}
                    step={1}
                    placeholder={selectedTaskDefaultMaxTurns ? String(selectedTaskDefaultMaxTurns) : "default"}
                    value={maxTurnsInput}
                    onChange={(event) => setMaxTurnsInput(event.target.value)}
                  />
                  <span className="small muted">{maxTurnsHelper}</span>
                </label>
              ) : null}
              {showDemoCodingWarning ? (
                <div className="field field-span-full">
                  <span className="small launcher-warning">
                    Demo warning: running more than 50 coding items here is not recommended.
                  </span>
                </div>
              ) : null}
            </div>
          </section>

          {selectedTaskIsDataset ? (
            <section className="subpanel">
              <div className="subpanel-header">
                <div>
                  <p className="eyebrow">SKILL RECORD/LOAD</p>
                </div>
              </div>
              <div className="control-grid">
                {showUseSkillField ? (
                  <label className="field field-row-start">
                    <span className="field-label">Use Skill</span>
                    <select
                      className="control"
                      value={selectedSkillId}
                      onChange={(event) => setSelectedSkillId(event.target.value)}
                    >
                      <option value="">No prior skill</option>
                      {selectedTaskAvailableSkills.map((skill) => (
                        <option key={skill.id} value={skill.id}>
                          {formatSkillOptionLabel(skill)}
                        </option>
                      ))}
                    </select>
                    <span className="small muted">{skillSelectHelper}</span>
                  </label>
                ) : null}
                <div className={`field ${showUseSkillField ? "" : "field-span-full"}`}>
                  <span className="field-label">Record Skill from This Run</span>
                  <label className="toggle-card">
                    <span className="toggle-row">
                      <input
                        type="checkbox"
                        checked={recordSkillEnabled}
                        onChange={(event) => setRecordSkillEnabled(event.target.checked)}
                      />
                      <span className="toggle-title">Record a distilled skill artifact for this run</span>
                    </span>
                    <span className="small muted toggle-description">
                      Save a distilled markdown skill beside this dataset&apos;s verified runs.
                    </span>
                    <span className="small muted toggle-description">{skillRecordHelper}</span>
                  </label>
                </div>
              </div>
            </section>
          ) : null}
        </div>

        <div className="button-row launcher-actions">
          <button className="action primary" onClick={() => void runTask(selectedTask?.id ?? null)} type="button">
            Start verified run
          </button>
        </div>
      </section>

      {error ? (
        <section className="panel error-panel">
          <p className="caption">run failure</p>
          <h2>{displayErrorType(error.error_type)}</h2>
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
          <div className="hero-block">
            <h2>Runtime Profile</h2>
          </div>
        </div>
        <div className="stack">
          <section className="subpanel">
            <div className="subpanel-header">
              <div>
                <p className="eyebrow">llm</p>
              </div>
            </div>
            <div className="metric-grid">
              {metric("policy model", selectedModel || runtimeInfo.active_model)}
              {selectedTaskSupportsEvalModel ? metric("eval model", effectiveEvalModel ?? "required") : null}
              {metric("endpoint", runtimeInfo.base_url)}
              {metric("temperature", runtimeInfo.temperature)}
              {metric("llm concurrency", runtimeInfo.llm_concurrency)}
              {metric("max tokens", runtimeInfo.max_tokens)}
              {metric("timeout", `${runtimeInfo.timeout_s}s`)}
            </div>
          </section>
          {selectedTask ? (
            <section className="subpanel">
              <div className="subpanel-header">
                <div>
                  <p className="eyebrow">task specification</p>
                </div>
              </div>
              <div className="metric-grid">
                {metric("task", selectedTask.title)}
                {metric("browse mode", browseModeLabel(browserModeForTask(selectedTask)))}
                {metric("category", taskCategoryLabel(selectedTask, browserModeForTask(selectedTask), personalizationReferenceBenchmarks))}
                {selectedTask.personalization_category ? metric("persona lane", personalizationCategoryLabel(selectedTask.personalization_category)) : null}
                {selectedTask.safety_category ? metric("safety category", safetyCategoryLabel(selectedTask.safety_category)) : null}
                {metric("turn mode", interactionModeLabel(selectedTask.interaction_mode))}
                {metric("metric", selectedTask.answer_metric)}
                {selectedTaskSupportsEvalModel ? metric("eval model", evalModelSummary) : null}
                {metric("task kind", taskModeLabel(selectedTask.task_mode))}
                {metric("search rounds", searchRoundsSummary)}
                {metric("proposal calls", proposalCallsSummary)}
                {metric("llm concurrency", llmConcurrencySummary)}
                {metric("frontier parents", frontierParentsSummary)}
                {metric("parallel eval", parallelismSummary)}
                {metric("eval scope", evalScopeSummary)}
              </div>
            </section>
          ) : null}
        </div>
      </section>

      <section className="panel stack">
        <div className="panel-header">
          <div className="hero-block">
            <h2>Active Run Stream</h2>
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
          <div className="hero-block">
            <h2>Latest Cached Report</h2>
          </div>
          <div className="badge-row">
            <span className="badge">{selectedTask?.id ?? "no task selected"}</span>
            <span className="badge">policy {payload.summary.policy_model ?? payload.summary.active_model}</span>
            {payload.summary.eval_model ? <span className="badge">eval {payload.summary.eval_model}</span> : null}
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
              <div className="hero-block">
                <p className="caption">warning</p>
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

      {taskBriefTask ? (
        <section className="modal-overlay" onClick={() => setTaskBriefTaskId(null)} role="presentation">
          <article
            className="modal-card"
            role="dialog"
            aria-modal="true"
            aria-labelledby="dataset-intro-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="panel-header">
              <div className="hero-block">
                <p className="caption">{taskBriefCaption(taskBriefTask)}</p>
                <h2 id="dataset-intro-title">{taskBriefTask.title}</h2>
              </div>
              <button className="action" onClick={() => setTaskBriefTaskId(null)} type="button">
                Dismiss
              </button>
            </div>
            <p className="muted modal-copy">{taskBriefCopy(taskBriefTask)}</p>
            <div className="task-summary-row">
              <span className="summary-pill">{benchmarkTierLabel(taskBriefTask.included_in_main_comparison)}</span>
              <span className="summary-pill">{trackLabel(taskBriefTask.track)}</span>
              <span className="summary-pill">{taskBriefTask.answer_metric}</span>
              <span className="summary-pill">{taskBriefTask.objective_label}</span>
              <span className="summary-pill">{taskBriefTask.function_name}</span>
              <span className="summary-pill">{taskModeLabel(taskBriefTask.task_mode)}</span>
            </div>
            <div className="metric-grid compact-metrics">
              {metric(
                taskBriefTask.supports_max_episodes ? "catalog episodes" : "local items",
                taskBriefTask.dataset_size ?? "n/a",
              )}
              {metric("source", taskBriefTask.dataset_id)}
              {metric(
                "split",
                taskBriefTask.split ?? inferSuiteSplitLabel(taskBriefTask.suite_run_config ?? null) ?? "local",
              )}
              {taskBriefTask.selected_runtime_split ? metric("selected split", taskBriefTask.selected_runtime_split) : null}
              {metric(taskBriefTask.task_mode === "answer" ? "candidate entrypoint" : "candidate file", taskBriefTask.editable_file)}
              {metric("turn mode", interactionModeLabel(taskBriefTask.interaction_mode))}
              {taskBriefTask.supports_max_episodes
                ? metric(
                    "default episode cap",
                    inferSuiteDefaultMaxEpisodes(taskBriefTask.suite_run_config ?? null) ??
                      taskBriefTask.default_max_episodes ??
                      "n/a",
                  )
                : null}
              {taskBriefTask.supports_max_episodes
                ? metric("max turns / episode", inferSuiteDefaultMaxTurns(taskBriefTask.suite_run_config ?? null) ?? "n/a")
                : null}
              {taskBriefTask.supports_runtime_config
                ? metric("tool surface", inferSuiteToolCount(taskBriefTask.suite_run_config ?? null) ?? "backend-owned")
                : null}
            </div>
            {taskBriefTask.supports_runtime_config && taskBriefTask.suite_run_config ? (
              <details className="detail-card" open>
                <summary className="detail-summary">
                  <div>
                    <strong>Runtime slice</strong>
                    <div className="detail-summary-copy">
                      Backend-owned split, limits, and tool schema for this task.
                    </div>
                  </div>
                </summary>
                <div className="detail-body">
                  <pre className="code-block compact"><code>{JSON.stringify(taskBriefTask.suite_run_config, null, 2)}</code></pre>
                </div>
              </details>
            ) : null}
          </article>
        </section>
      ) : null}

    </main>
  );
}

export default App;
