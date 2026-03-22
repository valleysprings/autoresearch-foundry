import type { JobState, Payload, RuntimeInfo, TaskSummary } from "./types";

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

export async function loadRuntime(): Promise<RuntimeInfo> {
  return fetchJson<RuntimeInfo>("/api/runtime");
}

export async function loadTasks(): Promise<TaskSummary[]> {
  const payload = await fetchJson<{ tasks: TaskSummary[] }>("/api/tasks");
  return payload.tasks;
}

export async function loadLatestRun(taskId?: string): Promise<Payload> {
  const query = taskId ? `?task_id=${encodeURIComponent(taskId)}` : "";
  return fetchJson<Payload>(`/api/latest-run${query}`);
}

export async function startJob(
  taskId: string | null,
  model: string,
  branchingFactor?: number | null,
): Promise<{ job_id: string; model: string }> {
  const params = new URLSearchParams();
  if (model) {
    params.set("model", model);
  }
  if (taskId) {
    params.set("task_id", taskId);
  }
  if (typeof branchingFactor === "number" && Number.isFinite(branchingFactor)) {
    params.set("branching_factor", String(Math.max(1, Math.floor(branchingFactor))));
  }
  const suffix = params.toString() ? `?${params.toString()}` : "";
  const url = taskId ? `/api/run-task${suffix}` : `/api/run-sequence${suffix}`;
  return fetchJson<{ job_id: string; model: string }>(url, { method: "POST" });
}

export async function loadJob(jobId: string): Promise<JobState> {
  return fetchJson<JobState>(`/api/job?job_id=${encodeURIComponent(jobId)}`);
}
