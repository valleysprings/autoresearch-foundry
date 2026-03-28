from __future__ import annotations

import re
from typing import Any

from app.codegen.selection import (
    DELTA_PRIMARY_SCORE_FORMULA,
    LINE_COUNT_NORMALIZER,
    PRIMARY_SCORE_FORMULA,
    PROPOSAL_SELECTION_GUIDANCE,
    RUN_DELTA_PRIMARY_SCORE_FORMULA,
    TIE_BREAK_SCORE_FORMULA,
)


DEFAULT_BRANCHING_FACTOR = 4
DEFAULT_EDITABLE_FILE = "editable.py"
DEFAULT_ENTRY_SYMBOL = "solve"
VALID_BENCHMARK_TIERS = {"experiment", "comparable"}
REQUIRED_TASK_FIELDS = (
    "benchmark_tier",
    "track",
    "answer_metric",
    "runtime_backend",
    "task_mode",
    "optimization_scope",
    "editable_file",
    "entry_symbol",
    "verifier",
)
DEFAULT_SPEEDUP_OBJECTIVE_SPEC: dict[str, str] = {
    "display_name": "Speedup vs baseline",
    "direction": "max",
    "unit": "x",
    "summary_template": "Higher speedup is better. This task maximizes runtime gain over the checked-in baseline.",
    "formula": "speedup_vs_baseline = baseline_ms / candidate_ms",
}
SEED_STRATEGY_EXPERIENCES: list[dict[str, Any]] = [
    {
        "experience_id": "exp-correctness-first",
        "experience_type": "strategy_experience",
        "experience_outcome": "success",
        "source_task": "seed",
        "source_session_id": "seed-catalog",
        "family": "agnostic",
        "task_signature": ["editable-file-optimization", "deterministic-eval", "correctness-first"],
        "verifier_status": "pass",
        "rejection_reason": "",
        "failure_pattern": "benchmark-only selection promoted fast but incorrect candidates",
        "strategy_hypothesis": "Run deterministic correctness gates before trusting optimization gains.",
        "successful_strategy": "preserve the public contract first, then optimize the editable file under the verifier",
        "prompt_fragment": "Preserve correctness first, then optimize only under the deterministic verifier contract.",
        "tool_trace_summary": "materialize candidate file -> deterministic verifier -> score -> selective write-back",
        "delta_primary_score": 0.18,
        "proposal_model": "seed",
        "candidate_summary": "Valid candidates only enter the comparison lane.",
        "reusable_rules": ["correctness_first", "deterministic_scoring", "editable_file_mutation"],
        "supporting_memory_ids": [],
    },
    {
        "experience_id": "exp-semantics-before-shortcuts",
        "experience_type": "strategy_experience",
        "experience_outcome": "failure",
        "source_task": "seed",
        "source_session_id": "seed-catalog",
        "family": "agnostic",
        "task_signature": ["editable-file-optimization", "deterministic-eval", "semantics-preservation"],
        "verifier_status": "fail",
        "rejection_reason": "A shortcut violated the contract and failed the deterministic checks.",
        "failure_pattern": "Aggressive rewrites improved one metric while breaking task semantics.",
        "strategy_hypothesis": "When the verifier is strict, local semantics-preserving rewrites dominate speculative shortcuts.",
        "successful_strategy": "Keep the contract fixed and prefer rewrites that remain faithful to the benchmark spec.",
        "prompt_fragment": "Do not trade away task semantics for a shortcut; keep the benchmark contract intact.",
        "tool_trace_summary": "shortcut rewrite -> deterministic failure -> reject -> prefer semantics-preserving rewrite",
        "delta_primary_score": -0.24,
        "proposal_model": "seed",
        "candidate_summary": "A fast-looking rewrite that violated the verifier contract.",
        "reusable_rules": ["preserve_semantics", "respect_verifier_contract"],
        "supporting_memory_ids": [],
    },
]

QUESTION_PREVIEW_LIMIT = 180
DATASET_NETWORK_ACCESS_INSTRUCTION = "Do not use browsing, web search, HTTP requests, or any external network access."
DATASET_SINGLE_QUESTION_INSTRUCTION = (
    "This run evaluates exactly one dataset question. Preserve the declared entry symbol and solve this question only."
)

DEFAULT_SESSION_ID = "session-current"
DEFAULT_MEMORY_RETRIEVAL_TOP_K = 4
DEFAULT_FRONTIER_SIZE = 8
WORKING_MEMORY_NAME = "codegen_working_memory.json"
WORKING_MEMORY_MD_NAME = "codegen_working_memory.md"
WORKING_MEMORY_TITLE = "Codegen Strategy Memory"
ITEM_MEMORY_DIR_NAME = "item_memory"
ITEM_MEMORY_JSON_NAME = "memory.json"
ITEM_MEMORY_MD_NAME = "memory.md"
FLYWHEEL_STEPS = [
    "load strict llm config from shell env or repo-root .env",
    "retrieve strategy memory fragments",
    "ask the configured model for candidate function bodies",
    "materialize candidates into an ignored workspace",
    "run deterministic tests and benchmarks",
    "select winners and write back reusable strategy experience",
    "emit payload, memory ledger, trace, and llm_trace artifacts",
]

OBJECTIVE_FORMULA = "objective is task-specific; see task.objective_spec.formula"
DELTA_FORMULA = DELTA_PRIMARY_SCORE_FORMULA
RUN_DELTA_FORMULA = RUN_DELTA_PRIMARY_SCORE_FORMULA
PRIMARY_FORMULA = PRIMARY_SCORE_FORMULA
TIE_BREAK_FORMULA = TIE_BREAK_SCORE_FORMULA

TEXT_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
    }
)
BOXED_PATTERN = re.compile(r"\\boxed\s*\{\s*([^{}]+)\s*\}")
LATEX_FRAC_PATTERN = re.compile(r"\\frac\s*\{\s*([+-]?\d+)\s*\}\s*\{\s*([+-]?\d+)\s*\}")
NUMERIC_FRAGMENT_PATTERN = re.compile(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:\s*/\s*[+-]?\d+)?")
PUBLIC_QUESTION_HIDDEN_METADATA_KEYS = {
    "answer_aliases",
    "correct_choice_index",
    "expected_answer",
    "raw_expected_answer",
}

COMPLEXITY_BASE = 0.12
COMPLEXITY_MAX = 0.95
COMPLEXITY_LINE_DIVISOR = 28.0
COMPLEXITY_IMPORT_COST = 0.04
COMPLEXITY_FOR_COST = 0.03
COMPLEXITY_WHILE_COST = 0.04
COMPLEXITY_IF_COST = 0.02
BENCHMARK_SAMPLE_COUNT = 3
SPEED_SCORE_CAP = 8.0
FORBIDDEN_NETWORK_PATTERNS = (
    r"(^|\n)\s*import\s+requests\b",
    r"(^|\n)\s*from\s+requests\b",
    r"(^|\n)\s*import\s+urllib\b",
    r"(^|\n)\s*from\s+urllib\b",
    r"(^|\n)\s*import\s+httpx\b",
    r"(^|\n)\s*from\s+httpx\b",
    r"(^|\n)\s*import\s+socket\b",
    r"(^|\n)\s*from\s+socket\b",
    r"(^|\n)\s*import\s+http\b",
    r"(^|\n)\s*from\s+http\b",
    r"(^|\n)\s*import\s+webbrowser\b",
    r"(^|\n)\s*from\s+webbrowser\b",
)


def speedup_objective_spec() -> dict[str, str]:
    return dict(DEFAULT_SPEEDUP_OBJECTIVE_SPEC)
