import test from "node:test";
import assert from "node:assert/strict";

import { normalizeErrorPayload } from "./errorPayload.ts";

test("normalizes a raw failed job object", () => {
  const payload = normalizeErrorPayload({
    status: "failed",
    terminal: true,
    error_type: "llm_response_error",
    error: "Model response did not contain valid JSON.",
    model: "deepseek-chat",
    details: { attempt: 3, parse_status: "invalid_json" },
    events: [],
  });

  assert.equal(payload.error_type, "llm_response_error");
  assert.equal(payload.error, "Model response did not contain valid JSON.");
  assert.equal(payload.model, "deepseek-chat");
  assert.deepEqual(payload.details, { attempt: 3, parse_status: "invalid_json" });
});

test("stringifies nested object errors", () => {
  const payload = normalizeErrorPayload({
    terminal: true,
    error_type: "runtime_error",
    error: { code: "planner_failed", reason: "empty frontier" },
    model: null,
  });

  assert.equal(payload.error_type, "runtime_error");
  assert.match(payload.error, /planner_failed/);
  assert.match(payload.error, /empty frontier/);
});

test("prefers payload details from thrown errors", () => {
  const error = new Error("request failed");
  (error as Error & { payload?: unknown }).payload = {
    terminal: true,
    error_type: "verification_error",
    error: "solve(problem) must return list[str]",
    model: "deepseek-chat",
    details: { task_id: "planbench-lite" },
  };

  const payload = normalizeErrorPayload(error);

  assert.equal(payload.error_type, "verification_error");
  assert.equal(payload.error, "solve(problem) must return list[str]");
  assert.equal(payload.model, "deepseek-chat");
  assert.deepEqual(payload.details, { task_id: "planbench-lite" });
});
