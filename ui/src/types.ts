export type RuntimeInfo = {
  mode: string;
  profile: string;
  provider: string;
  transport: string;
  default_model: string;
  active_model: string;
  available_models: string[];
  base_url: string;
  temperature: number | string;
  max_tokens: number | string;
  timeout_s: number | string;
  llm_concurrency: number | string;
  supports_tools: boolean | string;
  supports_json_mode: boolean | string;
};

export type ObjectiveSpec = {
  display_name: string;
  direction: "max" | "min" | string;
  unit?: string;
  summary_template: string;
  formula: string;
};

export type SelectionSpec = {
  profile?: string;
  display_name: string;
  summary_template: string;
  primary_metric?: string;
  primary_label?: string;
  primary_direction?: "max" | "min" | string;
  primary_formula: string;
  gate_summary: string;
  tie_break_formula: string;
  delta_template: string;
  archive_summary: string;
};

export type TaskSkill = {
  id: string;
  filename: string;
  title: string;
  path?: string;
  task_id?: string;
  dataset_id?: string;
  source_model?: string | null;
  source_items?: number | null;
  generated_at?: string | null;
};

export type RuntimeSplitOption = {
  value: string;
  title: string;
  description?: string | null;
  item_count?: number | null;
  match_tags_any?: string[] | null;
};

export type RuntimeSplitSelector = {
  label: string;
  default_value: string;
  options: RuntimeSplitOption[];
};

export type TaskSummary = {
  id: string;
  title: string;
  description: string;
  family: string;
  function_name: string;
  entry_symbol: string;
  editable_file: string;
  answer_metric: string;
  objective_label: string;
  objective_direction: string;
  objective_spec: ObjectiveSpec;
  selection_spec: SelectionSpec;
  generation_budget: number;
  candidate_budget: number;
  branching_factor: number;
  item_workers: number;
  benchmark_tier: string;
  track: string;
  dataset_id: string;
  dataset_size?: number;
  local_dataset_only?: boolean;
  split?: string | null;
  task_mode: string;
  interaction_mode: string;
  task_shape?: string | null;
  scoring_mode?: string | null;
  research_line?: string;
  personalization_category?: string | null;
  personalization_focus?: string | null;
  safety_category?: string | null;
  safety_focus?: string | null;
  supports_eval_model?: boolean;
  requires_eval_model?: boolean;
  default_eval_model?: string | null;
  included_in_main_comparison: boolean;
  supports_runtime_config: boolean;
  suite_run_config?: Record<string, unknown> | null;
  runtime_split_selector?: RuntimeSplitSelector | null;
  selected_runtime_split?: string | null;
  supports_max_items: boolean;
  default_max_items?: number | null;
  supports_max_episodes: boolean;
  default_max_episodes?: number | null;
  available_skills?: TaskSkill[];
};

export type DatasetWarning = {
  task_id: string;
  title: string;
  track: string;
  manifest_path: string;
  prepare_command: string;
  message: string;
};

export type PersonalizationReferenceBenchmark = {
  id: string;
  title: string;
  status: "local_task" | "external_reference" | string;
  task_id?: string | null;
  task_ids?: string[] | null;
  interaction_mode: "single_turn" | "multi_turn" | string;
  benchmark_category: string;
  primary_category: string;
  secondary_categories?: string[] | null;
  subject_domains?: string[] | null;
  implementation_status: "running" | "phase1" | "phase2" | "blocked" | string;
  task_shape: string;
  scoring_mode: string;
  supports_eval_model?: boolean;
  requires_eval_model?: boolean;
  default_eval_model?: string | null;
  official_metric_name: string;
  official_metric_backend: string;
  official_metric_granularity: string;
  metric_fidelity: string;
  official_dimensions: string[];
  protocol_summary: string;
  implementation_note: string;
  required_runtime_roles: string[];
  blocking_reason?: string | null;
  focus: string;
  summary: string;
  source_label: string;
  source_url: string;
  mirror_slug?: string | null;
  mirror_url?: string | null;
};

export type TaskCatalogPayload = {
  tasks: TaskSummary[];
  dataset_warnings?: DatasetWarning[];
  personalization_reference_benchmarks?: PersonalizationReferenceBenchmark[];
};

export type CandidateMetrics = {
  objective: number | string;
  objective_score?: number | string;
  primary_score?: number | string;
  tie_break_score?: number | string;
  gate_passed?: boolean;
  benchmark_ms?: number | null;
  speedup_vs_baseline?: number | string;
  passed_tests?: number | string;
  total_tests?: number | string;
  verifier_status?: string;
  status?: string;
  error?: string | null;
  test_results?: CandidateTestResult[];
  item_runs?: ItemRun[];
};

export type CandidateTestResult = {
  name?: string;
  expected?: unknown;
  actual_display?: unknown;
  actual?: unknown;
  actual_raw?: unknown;
  answer_format?: string;
  passed?: boolean;
};

export type Candidate = {
  candidate_id?: string;
  agent: string;
  label: string;
  strategy: string;
  rationale: string;
  candidate_summary: string;
  proposal_model?: string | null;
  verifier_status?: string;
  workspace_path?: string;
  source_code: string;
  metrics: CandidateMetrics;
};

export type MemoryFragment = {
  experience_id: string;
  prompt_fragment?: string;
  strategy_hypothesis?: string;
};

export type AddedExperience = {
  experience_id: string;
  generation: number;
  experience_outcome: string;
  verifier_status: string;
  delta_primary_score: number | string;
  prompt_fragment: string;
  strategy_hypothesis: string;
  candidate_summary: string;
  proposal_model?: string | null;
};

export type Branch = {
  branch_id: string;
  branch_index: number;
  parent_candidate: Candidate;
  retrieved_memories: MemoryFragment[];
  candidates: Candidate[];
  winner: Candidate;
  winner_accepted: boolean;
  winner_improved_global_best: boolean;
  delta_primary_score: number | string;
  global_best_delta_primary_score?: number | string;
  experience_outcome: string;
  wrote_memory: boolean;
  memory_delta: number;
  rejection_reason?: string | null;
};

export type Generation = {
  generation: number;
  winner_accepted: boolean;
  wrote_memory: boolean;
  delta_primary_score: number | string;
  winner: Candidate;
  parent_candidate: Candidate;
  parents?: Candidate[];
  retrieved_memories: MemoryFragment[];
  candidates: Candidate[];
  branches: Branch[];
  accepted_count?: number;
  memory_delta?: number;
  positive_writebacks?: number;
  negative_writebacks?: number;
};

export type ObjectivePoint = {
  generation: number;
  objective: number | string;
  objective_score?: number | string;
  candidate_objective: number | string;
  candidate_objective_score?: number | string;
  primary_score: number | string;
  candidate_primary_score: number | string;
  tie_break_score?: number | string;
  candidate_tie_break_score?: number | string;
  accepted: boolean;
  accepted_count?: number;
  improved_global_best?: boolean;
  memory_delta?: number;
};

export type RunTask = {
  id: string;
  title: string;
  description: string;
  family: string;
  function_name: string;
  entry_symbol: string;
  editable_file: string;
  answer_metric: string;
  objective_label: string;
  objective_direction: string;
  objective_spec: ObjectiveSpec;
  selection_spec: SelectionSpec;
  generation_budget: number;
  candidate_budget: number;
  branching_factor: number;
  item_workers: number;
  benchmark_tier: string;
  track: string;
  dataset_id: string;
  dataset_size?: number;
  local_dataset_only?: boolean;
  split?: string | null;
  included_in_main_comparison: boolean;
  task_mode?: string;
  interaction_mode?: string;
  task_shape?: string | null;
  scoring_mode?: string | null;
  research_line?: string;
  personalization_category?: string | null;
  personalization_focus?: string | null;
  safety_category?: string | null;
  safety_focus?: string | null;
  supports_eval_model?: boolean;
  requires_eval_model?: boolean;
  default_eval_model?: string | null;
  supports_runtime_config?: boolean;
  suite_run_config?: Record<string, unknown> | null;
  runtime_split_selector?: RuntimeSplitSelector | null;
  selected_runtime_split?: string | null;
  supports_max_items?: boolean;
  default_max_items?: number | null;
  supports_max_episodes?: boolean;
  default_max_episodes?: number | null;
  available_skills?: TaskSkill[];
};

export type DatasetSummary = {
  total_items: number;
  baseline_passed: number;
  winner_passed: number;
  failure_count: number;
  solved_ratio: number | string;
  avg_baseline_objective: number | string;
  avg_winner_objective: number | string;
  avg_delta_primary_score: number | string;
};

export type QuestionRecord = {
  item_id: string;
  id?: string;
  question_id?: string;
  name: string;
  prompt: string;
  raw_prompt?: string;
  context?: unknown;
  raw_context?: unknown;
  choices?: string[];
  raw_choices?: string[];
  expected_answer: unknown;
  raw_expected_answer?: unknown;
  metadata?: Record<string, unknown>;
};

export type ItemRun = {
  item_id: string;
  item_name: string;
  item_source_index?: number;
  item_brief?: string;
  question?: QuestionRecord;
  baseline?: Candidate;
  winner?: Candidate;
  delta_primary_score?: number | string;
  run_delta_primary_score?: number | string;
  run_delta_objective?: number | string;
  generations?: Generation[];
  objective_curve?: ObjectivePoint[];
  llm_traces?: Array<Record<string, unknown>>;
  memory_before_count?: number | string;
  memory_after_count?: number | string;
  positive_experiences_added?: number | string;
  negative_experiences_added?: number | string;
  added_experiences?: AddedExperience[];
  memory_markdown?: string;
  selection_reason?: string;
  payload?: Record<string, unknown>;
  turns?: Array<Record<string, unknown>>;
  success?: boolean;
  reward?: number | string;
  raw_artifact_path?: string | null;
  messages?: Array<Record<string, unknown>>;
};

export type Run = {
  run_mode: string;
  active_model: string;
  policy_model?: string;
  eval_model?: string | null;
  benchmark_tier: string;
  track: string;
  dataset_id: string;
  included_in_main_comparison: boolean;
  session_id?: string;
  generated_at?: string;
  task: RunTask;
  baseline: Candidate;
  winner: Candidate;
  dataset_summary?: DatasetSummary;
  suite_summary?: Record<string, unknown>;
  item_runs?: ItemRun[];
  delta_primary_score: number | string;
  run_delta_primary_score?: number | string;
  run_delta_objective?: number | string;
  selection_spec?: SelectionSpec;
  objective_curve: ObjectivePoint[];
  llm_traces: Array<Record<string, unknown>>;
  generations: Generation[];
  memory_before_count?: number | string;
  memory_after_count?: number | string;
  positive_experiences_added?: number | string;
  negative_experiences_added?: number | string;
  added_experiences?: AddedExperience[];
  memory_markdown: string;
  selection_reason: string;
};

export type PayloadSummary = {
  generated_at: string;
  run_mode: string;
  active_model: string;
  policy_model?: string;
  eval_model?: string | null;
  num_tasks: number;
  total_runs: number;
  experiment_runs: number;
  total_generations: number;
  initial_memory_count: number;
  memory_size_after_run: number;
  write_backs: number;
  experiment_write_backs: number;
  source_repo: string;
  git_commit: string;
  flywheel: string[];
  proposal_engine: RuntimeInfo;
};

export type Payload = {
  summary: PayloadSummary;
  formulas: {
    objective: string;
    primary_score: string;
    tie_break_score: string;
    delta_primary_score: string;
    run_delta_primary_score?: string;
  };
  audit: {
    workspace_root: string;
    session_id?: string | null;
    policy_model?: string | null;
    eval_model?: string | null;
    max_items?: number | null;
    max_episodes?: number | null;
  };
  task_catalog: TaskSummary[];
  runs: Run[];
};

export type ErrorPayload = {
  terminal: boolean;
  error_type: string;
  error: string;
  model: string | null;
  details?: unknown;
};

export type LiveEvent = {
  phase?: string;
  event_type?: string;
  task_id?: string;
  question_task_id?: string;
  item_id?: string;
  item_name?: string;
  item_source_index?: number;
  item_brief?: string;
  expected_answer?: string;
  generation?: number;
  branch_id?: string;
  branch_index?: number;
  parent_candidate?: string;
  candidate?: string;
  candidate_actual?: string;
  candidate_status?: string;
  retry_attempt?: number;
  max_attempts?: number;
  timestamp?: string;
  message?: string;
  accepted_to_frontier?: boolean;
  improved_global_best?: boolean;
  memory_delta?: number;
};

export type JobState = {
  status: "loading" | "running" | "failed" | "completed";
  task_id?: string | null;
  taskId?: string | null;
  branching_factor?: number | null;
  generation_budget?: number | null;
  candidate_budget?: number | null;
  llm_concurrency?: number | null;
  item_workers?: number | null;
  max_items?: number | null;
  max_episodes?: number | null;
  item_ids?: string[] | null;
  terminal?: boolean;
  error_type?: string | null;
  error?: string | null;
  model?: string | null;
  policy_model?: string | null;
  eval_model?: string | null;
  details?: unknown;
  suite_config?: Record<string, unknown> | null;
  events: LiveEvent[];
  payload?: Payload;
};
