import test from "node:test";
import assert from "node:assert/strict";

import { initialTaskId, latestTaskIdFromPayload, mergeTaskCatalogs, taskScopedPayload } from "./reportCache.ts";
import type { Payload, TaskSummary } from "./types.ts";

function task(id: string): TaskSummary {
  return {
    id,
    title: id,
    description: `${id} description`,
    family: "coding",
    function_name: "solve",
    entry_symbol: "solve",
    editable_file: "editable.py",
    answer_metric: "test_pass_rate",
    objective_label: "Test pass rate",
    objective_direction: "max",
    objective_spec: {
      display_name: "Test pass rate",
      direction: "max",
      summary_template: "Higher is better.",
      formula: "test_pass_rate = passed / total",
    },
    selection_spec: {
      display_name: "Layered selection policy",
      summary_template: "Gate first, then primary, then tie-break.",
      primary_metric: "objective_score",
      primary_label: "Normalized objective score",
      primary_direction: "max",
      primary_formula: "primary_score = objective_score",
      gate_summary: "gate: verifier_status == 'pass'",
      tie_break_formula: "tie_break_score = 0.0",
      delta_template: "delta_primary_score compares winners against parents.",
      archive_summary: "archive_features = none",
    },
    generation_budget: 1,
    candidate_budget: 1,
    branching_factor: 1,
    item_workers: 1,
    benchmark_tier: "comparable",
    track: "coding_verified",
    dataset_id: id,
    dataset_size: 1,
    local_dataset_only: true,
    split: "test",
    runtime_backend: "dataset",
    task_mode: "artifact",
    optimization_scope: "implementation",
    included_in_main_comparison: true,
    supports_runtime_config: false,
    external_run_config: null,
    supports_max_items: true,
    default_max_items: 1,
  };
}

function payload(taskIds: string[]): Payload {
  return {
    summary: {
      generated_at: "now",
      run_mode: "llm-required",
      active_model: "deepseek-chat",
      num_tasks: taskIds.length,
      total_runs: taskIds.length,
      experiment_runs: 0,
      total_generations: 0,
      initial_memory_count: 0,
      memory_size_after_run: 0,
      write_backs: 0,
      experiment_write_backs: 0,
      source_repo: "origin",
      git_commit: "abc123",
      flywheel: [],
      proposal_engine: {
        mode: "llm-required",
        primary_model: "deepseek-chat",
        active_model: "deepseek-chat",
        available_models: ["deepseek-chat"],
        api_base: "http://example.test/v1",
        temperature: 0.2,
        max_tokens: 4096,
        timeout_s: 45,
        llm_concurrency: 20,
      },
    },
    formulas: {
      objective: "",
      primary_score: "",
      tie_break_score: "",
      delta_primary_score: "",
      run_delta_primary_score: "",
    },
    audit: {
      workspace_root: "runs/workspace/current",
      session_id: "session-test",
    },
    task_catalog: taskIds.map((id) => task(id)),
    runs: taskIds.map((id) => ({
      run_mode: "llm-required",
      active_model: "deepseek-chat",
      benchmark_tier: "comparable",
      track: "coding_verified",
      dataset_id: id,
      included_in_main_comparison: true,
      task: {
        ...task(id),
        runtime_backend: "dataset",
      },
      baseline: {
        agent: "baseline",
        label: "baseline",
        strategy: "baseline",
        rationale: "baseline",
        candidate_summary: "baseline",
        source_code: "",
        metrics: { objective: 0, primary_score: 0, tie_break_score: 0, gate_passed: true },
      },
      winner: {
        agent: "winner",
        label: "winner",
        strategy: "winner",
        rationale: "winner",
        candidate_summary: "winner",
        source_code: "",
        metrics: { objective: 1, primary_score: 1, tie_break_score: 0, gate_passed: true },
      },
      objective_curve: [],
      llm_traces: [],
      generations: [],
      memory_markdown: "",
      selection_reason: `${id} selected`,
      delta_primary_score: 1,
    })),
  };
}

test("extracts the latest task id from the first cached run", () => {
  assert.equal(latestTaskIdFromPayload(payload(["livecodebench", "olymmath"])), "livecodebench");
});

test("prefers the latest cached task over the default first task", () => {
  const tasks = [task("olymmath"), task("livecodebench")];
  assert.equal(initialTaskId(tasks, payload(["livecodebench"])), "livecodebench");
});

test("returns only the selected task payload when the cached report matches", () => {
  const scoped = taskScopedPayload(payload(["livecodebench", "olymmath"]), "livecodebench");
  assert.ok(scoped);
  assert.deepEqual(scoped?.runs.map((run) => run.task.id), ["livecodebench"]);
});

test("returns null when the cached report belongs to another task", () => {
  assert.equal(taskScopedPayload(payload(["olymmath"]), "livecodebench"), null);
});

test("prefers current task catalog metadata over stale cached task catalog entries", () => {
  const current = {
    ...task("co-bench"),
    track: "or_verified",
    benchmark_tier: "experiment",
    included_in_main_comparison: false,
    runtime_backend: "dataset",
    task_mode: "artifact",
    optimization_scope: "wrapper",
    supports_runtime_config: false,
    local_dataset_only: true,
    dataset_size: 36,
    default_max_items: 36,
  };
  const staleCached = {
    ...current,
    runtime_backend: "external",
    task_mode: "agent",
    optimization_scope: "prompt",
    supports_runtime_config: true,
    local_dataset_only: false,
    dataset_size: undefined,
    default_max_items: 3,
  };

  const [merged] = mergeTaskCatalogs([current], [staleCached]);
  assert.equal(merged.runtime_backend, "dataset");
  assert.equal(merged.task_mode, "artifact");
  assert.equal(merged.optimization_scope, "wrapper");
  assert.equal(merged.supports_runtime_config, false);
  assert.equal(merged.local_dataset_only, true);
  assert.equal(merged.dataset_size, 36);
  assert.equal(merged.default_max_items, 36);
});
