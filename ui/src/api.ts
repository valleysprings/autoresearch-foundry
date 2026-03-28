import type { JobState, Payload, RuntimeInfo, TaskCatalogPayload } from "./types";

type StartJobOptions = {
  branchingFactor?: number | null;
  generationBudget?: number | null;
  candidateBudget?: number | null;
  itemWorkers?: number | null;
  maxItems?: number | null;
  itemIds?: string[] | null;
  externalConfig?: Record<string, unknown> | null;
};

type JsonOptions = RequestInit | undefined;

async function fetchJson<T>(url: string, options?: JsonOptions): Promise<T> {
  const response = await fetch(url, options);
  const text = await response.text();
  const payload = text ? JSON.parse(text) : {};
  if (!response.ok) {
    const message =
      typeof payload.error_type === "string"
        ? `${payload.error_type}: ${payload.error ?? ""}`.trim()
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
): Promise<{ job_id: string; model: string }> {
  const params = new URLSearchParams();
  if (model) {
    params.set("model", model);
  }
  if (taskId) {
    params.set("task_id", taskId);
  }
  const { branchingFactor, generationBudget, candidateBudget, itemWorkers, maxItems, itemIds } = options;
  const externalConfig = options.externalConfig;
  if (typeof branchingFactor === "number" && Number.isFinite(branchingFactor)) {
    params.set("branching_factor", String(Math.max(1, Math.floor(branchingFactor))));
  }
  if (typeof generationBudget === "number" && Number.isFinite(generationBudget)) {
    params.set("generation_budget", String(Math.max(1, Math.floor(generationBudget))));
  }
  if (typeof candidateBudget === "number" && Number.isFinite(candidateBudget)) {
    params.set("candidate_budget", String(Math.max(1, Math.floor(candidateBudget))));
  }
  if (typeof itemWorkers === "number" && Number.isFinite(itemWorkers)) {
    params.set("item_workers", String(Math.max(1, Math.floor(itemWorkers))));
  }
  if (typeof maxItems === "number" && Number.isFinite(maxItems)) {
    params.set("max_items", String(Math.max(1, Math.floor(maxItems))));
  }
  if (Array.isArray(itemIds) && itemIds.length) {
    params.set("item_ids", itemIds.join(","));
  }
  const suffix = params.toString() ? `?${params.toString()}` : "";
  const url = taskId ? `/api/run-task${suffix}` : `/api/run-sequence${suffix}`;
  const request: RequestInit = { method: "POST" };
  if (externalConfig && taskId) {
    request.headers = { "Content-Type": "application/json" };
    request.body = JSON.stringify({ external_config: externalConfig });
  }
  return fetchJson<{ job_id: string; model: string }>(url, request);
}

export async function loadJob(jobId: string): Promise<JobState> {
  return fetchJson<JobState>(`/api/job?job_id=${encodeURIComponent(jobId)}`, { cache: "no-store" });
}
