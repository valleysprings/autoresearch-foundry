import type { Payload, TaskSummary } from "./types";

export function mergeTaskCatalogs(fallbackCatalog: TaskSummary[], cachedCatalog: TaskSummary[] | undefined): TaskSummary[] {
  if (!Array.isArray(cachedCatalog) || !cachedCatalog.length) {
    return fallbackCatalog;
  }
  const cachedById = new Map(cachedCatalog.map((task) => [task.id, task]));
  return fallbackCatalog.map((task) => {
    const cachedTask = cachedById.get(task.id);
    if (!cachedTask) {
      return task;
    }
    return {
      ...cachedTask,
      ...task,
      id: task.id,
    };
  });
}

export function latestTaskIdFromPayload(payload: Payload | null | undefined): string | null {
  if (!payload || !Array.isArray(payload.runs)) {
    return null;
  }
  for (const run of payload.runs) {
    const taskId = typeof run?.task?.id === "string" ? run.task.id.trim() : "";
    if (taskId) {
      return taskId;
    }
  }
  return null;
}

export function initialTaskId(tasks: TaskSummary[], latestPayload: Payload | null | undefined): string {
  const latestTaskId = latestTaskIdFromPayload(latestPayload);
  if (latestTaskId && tasks.some((task) => task.id === latestTaskId)) {
    return latestTaskId;
  }
  return tasks[0]?.id ?? "";
}

export function taskScopedPayload(payload: Payload, taskId: string | null | undefined): Payload | null {
  if (!taskId || !Array.isArray(payload.runs)) {
    return null;
  }
  const runs = payload.runs.filter((run) => run.task?.id === taskId);
  if (!runs.length) {
    return null;
  }
  return {
    ...payload,
    runs,
  };
}
