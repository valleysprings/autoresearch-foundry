import type { ErrorPayload } from "./types.ts";

function asRecord(value: unknown): Record<string, unknown> | null {
  if (value && typeof value === "object") {
    return value as Record<string, unknown>;
  }
  return null;
}

export function stringifyUnknown(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  if (value instanceof Error) {
    return value.message || String(value);
  }
  if (value == null) {
    return "";
  }
  try {
    const serialized = JSON.stringify(value, null, 2);
    if (serialized) {
      return serialized;
    }
  } catch {
    return String(value);
  }
  return String(value);
}

function buildPayload(candidate: Record<string, unknown>, fallbackMessage: string): ErrorPayload {
  const error = candidate.error;
  const details = candidate.details;
  return {
    terminal: typeof candidate.terminal === "boolean" ? candidate.terminal : true,
    error_type: typeof candidate.error_type === "string" ? candidate.error_type : "runtime_error",
    error: stringifyUnknown(error ?? details ?? fallbackMessage),
    model: typeof candidate.model === "string" ? candidate.model : null,
    details,
  };
}

export function normalizeErrorPayload(error: unknown): ErrorPayload {
  const direct = asRecord(error);
  const nested = asRecord(direct?.payload);

  if (nested && typeof nested.error_type === "string") {
    return buildPayload(nested, stringifyUnknown(error));
  }

  if (direct && typeof direct.error_type === "string") {
    return buildPayload(direct, stringifyUnknown(error));
  }

  if (error instanceof Error) {
    return {
      terminal: true,
      error_type: "runtime_error",
      error: error.message || String(error),
      model: null,
    };
  }

  return {
    terminal: true,
    error_type: "runtime_error",
    error: stringifyUnknown(error),
    model: null,
  };
}
