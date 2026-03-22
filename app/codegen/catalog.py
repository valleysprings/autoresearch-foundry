from __future__ import annotations

import textwrap
from typing import Any


def _body(source: str) -> str:
    return textwrap.dedent(source).strip("\n")


SEED_STRATEGY_EXPERIENCES: list[dict[str, Any]] = [
    {
        "experience_id": "exp-correctness-first",
        "experience_type": "strategy_experience",
        "experience_outcome": "success",
        "source_task": "seed",
        "source_session_id": "seed-catalog",
        "family": "agnostic",
        "task_signature": ["python-codegen", "deterministic-eval", "correctness-first"],
        "verifier_status": "pass",
        "rejection_reason": "",
        "failure_pattern": "benchmark-only selection promoted fast but incorrect candidates",
        "strategy_hypothesis": "Run deterministic correctness gates before trusting benchmark improvements.",
        "successful_strategy": "compile the candidate, run fixed correctness tests, and benchmark only the passing variants",
        "prompt_fragment": "Preserve task semantics first, then optimize runtime only on candidates that pass every fixed test.",
        "tool_trace_summary": "candidate source -> compile -> deterministic tests -> median benchmark -> score -> selective write-back",
        "delta_J": 0.18,
        "proposal_model": "seed",
        "candidate_summary": "Valid candidates only enter the benchmark lane.",
        "reusable_rules": ["correctness_first", "benchmark_after_tests", "deterministic_scoring"],
        "supporting_memory_ids": [],
    },
    {
        "experience_id": "exp-semantics-before-shortcuts",
        "experience_type": "strategy_experience",
        "experience_outcome": "failure",
        "source_task": "seed",
        "source_session_id": "seed-catalog",
        "family": "agnostic",
        "task_signature": ["python-codegen", "deterministic-eval", "semantics-preservation"],
        "verifier_status": "fail",
        "rejection_reason": "A fast shortcut changed ordering semantics and failed the deterministic tests.",
        "failure_pattern": "Sort- or set-based shortcuts broke stable-order or first-hit semantics while looking faster on the benchmark.",
        "strategy_hypothesis": "When the task depends on encounter order, optimize with streaming state instead of reordering the input.",
        "successful_strategy": "Prefer a streaming hash-based or counted approach that preserves the original traversal semantics.",
        "prompt_fragment": "Do not trade away order-sensitive semantics for a reordering shortcut; keep the original traversal contract intact.",
        "tool_trace_summary": "reordering shortcut -> deterministic test failure -> reject -> switch to stateful streaming strategy",
        "delta_J": -0.24,
        "proposal_model": "seed",
        "candidate_summary": "A seemingly fast reordering shortcut that violated the task contract.",
        "reusable_rules": ["preserve_semantics", "avoid_order_breaking_shortcuts"],
        "supporting_memory_ids": [],
    },
]


CODEGEN_TASKS: list[dict[str, Any]] = [
    {
        "id": "contains-duplicates",
        "title": "Optimize contains_duplicates",
        "description": "Improve a naive quadratic duplicate detector with deterministic correctness checks and a real benchmark.",
        "family": "set-logic",
        "function_name": "contains_duplicates",
        "function_signature": "def contains_duplicates(values):",
        "objective_label": "speedup_vs_baseline",
        "objective_direction": "max",
        "task_signature": ["python-codegen", "set-logic", "duplicate-detection"],
        "source_type": "embedded-codegen-task",
        "generation_budget": 3,
        "candidate_budget": 3,
        "epsilon": 0.20,
        "baseline_imports": [],
        "baseline_body": _body(
            """
            for index, left in enumerate(values):
                for right in values[index + 1 :]:
                    if left == right:
                        return True
            return False
            """
        ),
        "baseline_summary": "Nested-loop duplicate detection with O(n^2) behavior.",
        "benchmark": {"kind": "contains_duplicates", "repeats": 14},
        "tests": [
            {"name": "unique-values", "args": [[1, 2, 3, 4]], "expected": False},
            {"name": "has-duplicate", "args": [[1, 2, 3, 2]], "expected": True},
            {"name": "empty", "args": [[]], "expected": False},
            {"name": "pair", "args": [[7, 7]], "expected": True},
        ],
    },
    {
        "id": "first-repeated-value",
        "title": "Optimize first_repeated_value",
        "description": "Preserve correctness for the first repeated item while replacing the quadratic scan with a streaming strategy.",
        "family": "set-logic",
        "function_name": "first_repeated_value",
        "function_signature": "def first_repeated_value(values):",
        "objective_label": "speedup_vs_baseline",
        "objective_direction": "max",
        "task_signature": ["python-codegen", "set-logic", "first-repeat"],
        "source_type": "embedded-codegen-task",
        "generation_budget": 3,
        "candidate_budget": 3,
        "epsilon": 0.20,
        "baseline_imports": [],
        "baseline_body": _body(
            """
            for index, left in enumerate(values):
                for right in values[index + 1 :]:
                    if left == right:
                        return left
            return None
            """
        ),
        "baseline_summary": "Quadratic scan that returns the first value with a later duplicate.",
        "benchmark": {"kind": "first_repeated_value", "repeats": 12},
        "tests": [
            {"name": "basic-repeat", "args": [[1, 2, 3, 2, 4]], "expected": 2},
            {"name": "first-repeat-matters", "args": [[5, 1, 5, 1]], "expected": 5},
            {"name": "no-repeat", "args": [[1, 2, 3]], "expected": None},
            {"name": "repeat-at-start", "args": [[9, 9, 1]], "expected": 9},
        ],
    },
    {
        "id": "has-overlap",
        "title": "Optimize has_overlap",
        "description": "Turn a nested-loop overlap check into a real set-based benchmarked optimization.",
        "family": "set-logic",
        "function_name": "has_overlap",
        "function_signature": "def has_overlap(left, right):",
        "objective_label": "speedup_vs_baseline",
        "objective_direction": "max",
        "task_signature": ["python-codegen", "set-logic", "overlap-detection"],
        "source_type": "embedded-codegen-task",
        "generation_budget": 3,
        "candidate_budget": 3,
        "epsilon": 0.20,
        "baseline_imports": [],
        "baseline_body": _body(
            """
            for left_value in left:
                for right_value in right:
                    if left_value == right_value:
                        return True
            return False
            """
        ),
        "baseline_summary": "Quadratic overlap check over both collections.",
        "benchmark": {"kind": "has_overlap", "repeats": 16},
        "tests": [
            {"name": "disjoint", "args": [[1, 2, 3], [4, 5]], "expected": False},
            {"name": "has-overlap", "args": [[1, 2, 3], [3, 4]], "expected": True},
            {"name": "empty-left", "args": [[], [1]], "expected": False},
            {"name": "single-match", "args": [[5], [5]], "expected": True},
        ],
    },
    {
        "id": "most-frequent-item",
        "title": "Optimize most_frequent_item",
        "description": "Replace a quadratic mode finder with counted architectures and compare one-pass versus two-pass strategies.",
        "family": "counting",
        "function_name": "most_frequent_item",
        "function_signature": "def most_frequent_item(values):",
        "objective_label": "speedup_vs_baseline",
        "objective_direction": "max",
        "task_signature": ["python-codegen", "counting", "mode-finding"],
        "source_type": "embedded-codegen-task",
        "generation_budget": 3,
        "candidate_budget": 3,
        "epsilon": 0.20,
        "baseline_imports": [],
        "baseline_body": _body(
            """
            best_value = None
            best_count = -1
            for value in values:
                count = 0
                for other in values:
                    if other == value:
                        count += 1
                if count > best_count:
                    best_value = value
                    best_count = count
            return best_value
            """
        ),
        "baseline_summary": "Quadratic recounting loop for every candidate value.",
        "benchmark": {"kind": "most_frequent_item", "repeats": 10},
        "tests": [
            {"name": "simple-mode", "args": [[1, 2, 2, 3]], "expected": 2},
            {"name": "string-mode", "args": [["a", "b", "b", "a", "b"]], "expected": "b"},
            {"name": "single-item", "args": [[7]], "expected": 7},
            {"name": "late-mode", "args": [[3, 3, 2, 2, 2]], "expected": 2},
        ],
    },
    {
        "id": "deduplicate-preserve-order",
        "title": "Optimize deduplicate_preserve_order",
        "description": "Keep original order while removing duplicates, and compare hash-based versus order-breaking architectures.",
        "family": "set-logic",
        "function_name": "deduplicate_preserve_order",
        "function_signature": "def deduplicate_preserve_order(values):",
        "objective_label": "speedup_vs_baseline",
        "objective_direction": "max",
        "task_signature": ["python-codegen", "set-logic", "stable-deduplication"],
        "source_type": "embedded-codegen-task",
        "generation_budget": 3,
        "candidate_budget": 3,
        "epsilon": 0.20,
        "baseline_imports": [],
        "baseline_body": _body(
            """
            result = []
            for value in values:
                seen = False
                for existing in result:
                    if existing == value:
                        seen = True
                        break
                if not seen:
                    result.append(value)
            return result
            """
        ),
        "baseline_summary": "Nested-loop stable deduplication that rescans the output buffer.",
        "benchmark": {"kind": "deduplicate_preserve_order", "repeats": 12},
        "tests": [
            {"name": "basic-dedup", "args": [[1, 2, 1, 3, 2]], "expected": [1, 2, 3]},
            {"name": "already-unique", "args": [[1, 2, 3]], "expected": [1, 2, 3]},
            {"name": "strings", "args": [["a", "b", "a", "c"]], "expected": ["a", "b", "c"]},
            {"name": "empty", "args": [[]], "expected": []},
        ],
    },
    {
        "id": "missing-number",
        "title": "Optimize missing_number",
        "description": "Find the missing integer with architecture families that mimic search, sorting, and arithmetic reasoning.",
        "family": "numeric",
        "function_name": "missing_number",
        "function_signature": "def missing_number(values):",
        "objective_label": "speedup_vs_baseline",
        "objective_direction": "max",
        "task_signature": ["python-codegen", "numeric", "missing-value"],
        "source_type": "embedded-codegen-task",
        "generation_budget": 3,
        "candidate_budget": 3,
        "epsilon": 0.20,
        "baseline_imports": [],
        "baseline_body": _body(
            """
            upper = len(values) + 1
            for candidate in range(upper):
                present = False
                for value in values:
                    if value == candidate:
                        present = True
                        break
                if not present:
                    return candidate
            return None
            """
        ),
        "baseline_summary": "Repeated full-list membership scans to find the missing value.",
        "benchmark": {"kind": "missing_number", "repeats": 16},
        "tests": [
            {"name": "missing-middle", "args": [[0, 1, 3]], "expected": 2},
            {"name": "missing-zero", "args": [[1, 2, 3]], "expected": 0},
            {"name": "missing-end", "args": [[0, 1, 2]], "expected": 3},
            {"name": "unordered-input", "args": [[4, 2, 1, 0]], "expected": 3},
        ],
    },
    {
        "id": "count-primes-up-to",
        "title": "Optimize count_primes_up_to",
        "description": "Count primes up to a limit on a deterministic math benchmark where trial division and sieve families compete.",
        "family": "math",
        "function_name": "count_primes_up_to",
        "function_signature": "def count_primes_up_to(limit):",
        "objective_label": "speedup_vs_baseline",
        "objective_direction": "max",
        "task_signature": ["python-codegen", "math", "prime-counting"],
        "source_type": "embedded-codegen-task",
        "generation_budget": 6,
        "candidate_budget": 3,
        "epsilon": 0.15,
        "baseline_imports": [],
        "baseline_body": _body(
            """
            count = 0
            for candidate in range(2, limit + 1):
                is_prime = True
                divisor = 2
                while divisor < candidate:
                    if candidate % divisor == 0:
                        is_prime = False
                        break
                    divisor += 1
                if is_prime:
                    count += 1
            return count
            """
        ),
        "baseline_summary": "Naive trial division for every candidate integer.",
        "benchmark": {"kind": "count_primes_up_to", "repeats": 8},
        "tests": [
            {"name": "below-two", "args": [1], "expected": 0},
            {"name": "two", "args": [2], "expected": 1},
            {"name": "ten", "args": [10], "expected": 4},
            {"name": "thirty", "args": [30], "expected": 10},
        ],
    },
    {
        "id": "count-change-ways",
        "title": "Optimize count_change_ways",
        "description": "Tackle a harder combinatorics problem where exponential recursion should evolve toward dynamic programming.",
        "family": "math",
        "function_name": "count_change_ways",
        "function_signature": "def count_change_ways(total, coins):",
        "objective_label": "speedup_vs_baseline",
        "objective_direction": "max",
        "task_signature": ["python-codegen", "math", "coin-change"],
        "source_type": "embedded-codegen-task",
        "generation_budget": 6,
        "candidate_budget": 3,
        "epsilon": 0.15,
        "baseline_imports": [],
        "baseline_body": _body(
            """
            def search(remaining, index):
                if remaining == 0:
                    return 1
                if remaining < 0 or index >= len(coins):
                    return 0
                return search(remaining - coins[index], index) + search(remaining, index + 1)

            return search(total, 0)
            """
        ),
        "baseline_summary": "Exponential recursive coin-change counting without memoization.",
        "benchmark": {"kind": "count_change_ways", "repeats": 5},
        "tests": [
            {"name": "zero-total", "args": [0, [1, 2]], "expected": 1},
            {"name": "simple", "args": [4, [1, 2, 3]], "expected": 4},
            {"name": "classic", "args": [10, [2, 5, 3, 6]], "expected": 5},
            {"name": "single-coin", "args": [5, [5]], "expected": 1},
        ],
    },
    {
        "id": "count-n-queens",
        "title": "Optimize count_n_queens",
        "description": "Benchmark combinatorial search on a harder math workload where better state encoding should unlock deeper improvements.",
        "family": "math",
        "function_name": "count_n_queens",
        "function_signature": "def count_n_queens(size):",
        "objective_label": "speedup_vs_baseline",
        "objective_direction": "max",
        "task_signature": ["python-codegen", "math", "n-queens"],
        "source_type": "embedded-codegen-task",
        "generation_budget": 6,
        "candidate_budget": 3,
        "epsilon": 0.15,
        "baseline_imports": [],
        "baseline_body": _body(
            """
            placements = []

            def conflicts(column):
                row = len(placements)
                for previous_row, previous_column in enumerate(placements):
                    if previous_column == column or abs(previous_column - column) == row - previous_row:
                        return True
                return False

            def search():
                row = len(placements)
                if row == size:
                    return 1
                total = 0
                for column in range(size):
                    if conflicts(column):
                        continue
                    placements.append(column)
                    total += search()
                    placements.pop()
                return total

            return search()
            """
        ),
        "baseline_summary": "Plain backtracking with linear conflict checks at every placement.",
        "benchmark": {"kind": "count_n_queens", "repeats": 4},
        "tests": [
            {"name": "one", "args": [1], "expected": 1},
            {"name": "four", "args": [4], "expected": 2},
            {"name": "five", "args": [5], "expected": 10},
            {"name": "six", "args": [6], "expected": 4},
        ],
    },
]


def load_codegen_tasks() -> list[dict[str, Any]]:
    return [dict(task) for task in CODEGEN_TASKS]


def list_codegen_task_summaries() -> list[dict[str, Any]]:
    return [
        {
            "id": task["id"],
            "title": task["title"],
            "description": task["description"],
            "family": task["family"],
            "function_name": task["function_name"],
            "objective_label": task["objective_label"],
            "objective_direction": task["objective_direction"],
            "generation_budget": task["generation_budget"],
            "candidate_budget": task["candidate_budget"],
        }
        for task in load_codegen_tasks()
    ]


def seed_strategy_experiences() -> list[dict[str, Any]]:
    return [dict(item) for item in SEED_STRATEGY_EXPERIENCES]
