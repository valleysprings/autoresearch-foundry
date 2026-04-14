import type { JobState, Payload, RuntimeInfo, TaskCatalogPayload } from "./types";
import { displayErrorType } from "./errorPayload";

type StartJobOptions = {
  branchingFactor?: number | null;
  generationBudget?: number | null;
  candidateBudget?: number | null;
  evalModel?: string | null;
  llmConcurrency?: number | null;
  itemWorkers?: number | null;
  maxItems?: number | null;
  maxEpisodes?: number | null;
  suiteConfig?: Record<string, unknown> | null;
  recordSkill?: boolean;
  selectedSkillId?: string | null;
};

type JsonOptions = RequestInit | undefined;

async function fetchJson<T>(url: string, options?: JsonOptions): Promise<T> {
  const response = await fetch(url, options);
  const text = await response.text();
  const payload = text ? JSON.parse(text) : {};
  if (!response.ok) {
    const message =
      typeof payload.error_type === "string"
        ? `${displayErrorType(payload.error_type)}: ${payload.error ?? ""}`.trim()
        : `request failed with status ${response.status}`;
    const error = new Error(message) as Error & { payload?: unknown };
    error.payload = payload;
    throw error;
  }
  return payload as T;
}

export async function loadRuntime(model?: string): Promise<RuntimeInfo> {
  const query = model ? `?model=${encodeURIComponent(model)}` : "";
  return fetchJson<RuntimeInfo>(`/api/runtime${query}`);
}

export async function loadTasks(): Promise<TaskCatalogPayload> {
  return fetchJson<TaskCatalogPayload>("/api/tasks");
}

export async function loadLatestRun(taskId?: string): Promise<Payload> {
  const query = taskId ? `?task_id=${encodeURIComponent(taskId)}` : "";
  return fetchJson<Payload>(`/api/latest-run${query}`, { cache: "no-store" });
}

export async function startJob(
  taskId: string | null,
  model: string,
  options: StartJobOptions = {},
): Promise<{ job_id: string; model: string; policy_model?: string | null; eval_model?: string | null }> {
  const params = new URLSearchParams();
  if (model) {
    params.set("model", model);
  }
  if (taskId) {
    params.set("task_id", taskId);
  }
  const { branchingFactor, generationBudget, candidateBudget, evalModel, llmConcurrency, itemWorkers, maxItems, maxEpisodes } = options;
  const suiteConfig = options.suiteConfig;
  const recordSkill = options.recordSkill;
  const selectedSkillId = options.selectedSkillId;
  if (typeof branchingFactor === "number" && Number.isFinite(branchingFactor)) {
    params.set("branching_factor", String(Math.max(1, Math.floor(branchingFactor))));
  }
  if (typeof generationBudget === "number" && Number.isFinite(generationBudget)) {
    params.set("generation_budget", String(Math.max(1, Math.floor(generationBudget))));
  }
  if (typeof candidateBudget === "number" && Number.isFinite(candidateBudget)) {
    params.set("candidate_budget", String(Math.max(1, Math.floor(candidateBudget))));
  }
  if (typeof evalModel === "string" && evalModel.trim()) {
    params.set("eval_model", evalModel.trim());
  }
  if (typeof llmConcurrency === "number" && Number.isFinite(llmConcurrency)) {
    params.set("llm_concurrency", String(Math.max(1, Math.floor(llmConcurrency))));
  }
  if (typeof itemWorkers === "number" && Number.isFinite(itemWorkers)) {
    params.set("item_workers", String(Math.max(1, Math.floor(itemWorkers))));
  }
  if (typeof maxItems === "number" && Number.isFinite(maxItems)) {
    params.set("max_items", String(Math.max(1, Math.floor(maxItems))));
  }
  if (typeof maxEpisodes === "number" && Number.isFinite(maxEpisodes)) {
    params.set("max_episodes", String(Math.max(1, Math.floor(maxEpisodes))));
  }
  const suffix = params.toString() ? `?${params.toString()}` : "";
  const url = taskId ? `/api/run-task${suffix}` : `/api/run-sequence${suffix}`;
  const request: RequestInit = { method: "POST" };
  const requestBody: Record<string, unknown> = {};
  if (suiteConfig && taskId) {
    requestBody.suite_config = suiteConfig;
  }
  if (taskId && typeof recordSkill === "boolean") {
    requestBody.record_skill = recordSkill;
  }
  if (taskId && typeof selectedSkillId === "string" && selectedSkillId.trim()) {
    requestBody.selected_skill_id = selectedSkillId.trim();
  }
  if (taskId && Object.keys(requestBody).length) {
    request.headers = { "Content-Type": "application/json" };
    request.body = JSON.stringify(requestBody);
  }
  return fetchJson<{ job_id: string; model: string; policy_model?: string | null; eval_model?: string | null }>(url, request);
}

export async function loadJob(jobId: string): Promise<JobState> {
  return fetchJson<JobState>(`/api/job?job_id=${encodeURIComponent(jobId)}`, { cache: "no-store" });
}
