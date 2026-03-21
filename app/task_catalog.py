from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _code(source: str) -> str:
    return textwrap.dedent(source).strip() + "\n"


CANDIDATE_SPECS: dict[str, list[dict[str, Any]]] = {
    "contains-duplicates": [
        {
            "agent": "sort-optimizer",
            "label": "Sort and scan neighbors",
            "strategy": "Replace the nested loop with a sorted pass and an adjacent comparison.",
            "code": _code(
                """
                def contains_duplicates(values):
                    ordered = sorted(values)
                    for index in range(1, len(ordered)):
                        if ordered[index] == ordered[index - 1]:
                            return True
                    return False
                """
            ),
            "complexity": 0.35,
            "uses_memory": False,
            "required_rules": [],
            "reusable_rules": ["sort_then_scan"],
            "notes": [
                "Correct and deterministic.",
                "Cuts the search from O(n^2) to O(n log n).",
            ],
        },
        {
            "agent": "hash-optimizer",
            "label": "Set cardinality check",
            "strategy": "Materialize a set and compare its size with the original list length.",
            "code": _code(
                """
                def contains_duplicates(values):
                    return len(values) != len(set(values))
                """
            ),
            "complexity": 0.18,
            "uses_memory": False,
            "required_rules": [],
            "reusable_rules": ["hash_membership", "cardinality_check"],
            "notes": [
                "Uses the Python runtime's optimized hash table.",
                "Usually the fastest option when the whole container must be scanned anyway.",
            ],
        },
        {
            "agent": "replay-synthesizer",
            "label": "Streaming seen-set",
            "strategy": "Use a seen set so later tasks can reuse the same pattern for first-repeat and overlap problems.",
            "code": _code(
                """
                def contains_duplicates(values):
                    seen = set()
                    for value in values:
                        if value in seen:
                            return True
                        seen.add(value)
                    return False
                """
            ),
            "complexity": 0.26,
            "uses_memory": True,
            "required_rules": ["correctness_first"],
            "reusable_rules": ["hash_membership", "streaming_seen_set"],
            "notes": [
                "Slightly more code than cardinality check but more reusable across task families.",
                "This is the memory-friendly pattern for later tasks.",
            ],
        },
    ],
    "first-repeated-value": [
        {
            "agent": "count-optimizer",
            "label": "Count then rescan",
            "strategy": "Build counts once, then rescan to find the first value whose count exceeds one.",
            "code": _code(
                """
                def first_repeated_value(values):
                    counts = {}
                    for value in values:
                        counts[value] = counts.get(value, 0) + 1
                    for value in values:
                        if counts[value] > 1:
                            return value
                    return None
                """
            ),
            "complexity": 0.31,
            "uses_memory": False,
            "required_rules": [],
            "reusable_rules": ["count_then_scan"],
            "notes": [
                "Correct and clearly faster than the nested baseline.",
                "Needs two passes through the list.",
            ],
        },
        {
            "agent": "sort-optimizer",
            "label": "Sorted adjacency guess",
            "strategy": "Sort the values and take the first adjacent duplicate, which is fast but wrong for the original order semantics.",
            "code": _code(
                """
                def first_repeated_value(values):
                    ordered = sorted(values)
                    for index in range(1, len(ordered)):
                        if ordered[index] == ordered[index - 1]:
                            return ordered[index]
                    return None
                """
            ),
            "complexity": 0.28,
            "uses_memory": False,
            "required_rules": [],
            "reusable_rules": ["sort_then_scan"],
            "notes": [
                "This candidate is intentionally incorrect for first-occurrence semantics.",
                "The evaluator should reject it even if it is fast.",
            ],
        },
        {
            "agent": "replay-synthesizer",
            "label": "Replay-guided streaming set",
            "strategy": "Reuse the duplicate-detection memory and return on the first repeated value seen in order.",
            "code": _code(
                """
                def first_repeated_value(values):
                    seen = set()
                    for value in values:
                        if value in seen:
                            return value
                        seen.add(value)
                    return None
                """
            ),
            "complexity": 0.22,
            "uses_memory": True,
            "required_rules": ["hash_membership", "streaming_seen_set"],
            "reusable_rules": ["hash_membership", "streaming_seen_set", "preserve_order"],
            "notes": [
                "Carries a concrete memory forward from the first task.",
                "Fast and preserves the original traversal order.",
            ],
        },
    ],
    "has-overlap": [
        {
            "agent": "merge-optimizer",
            "label": "Sort and merge",
            "strategy": "Sort both collections and walk them with two indices.",
            "code": _code(
                """
                def has_overlap(left, right):
                    ordered_left = sorted(left)
                    ordered_right = sorted(right)
                    left_index = 0
                    right_index = 0

                    while left_index < len(ordered_left) and right_index < len(ordered_right):
                        left_value = ordered_left[left_index]
                        right_value = ordered_right[right_index]
                        if left_value == right_value:
                            return True
                        if left_value < right_value:
                            left_index += 1
                        else:
                            right_index += 1
                    return False
                """
            ),
            "complexity": 0.39,
            "uses_memory": False,
            "required_rules": [],
            "reusable_rules": ["sort_then_scan"],
            "notes": [
                "A solid O(n log n) fallback when hashability is uncertain.",
                "More moving parts than a hash-based check.",
            ],
        },
        {
            "agent": "hash-optimizer",
            "label": "Two-set intersection",
            "strategy": "Materialize both sides as sets and test whether the intersection is non-empty.",
            "code": _code(
                """
                def has_overlap(left, right):
                    return bool(set(left) & set(right))
                """
            ),
            "complexity": 0.20,
            "uses_memory": False,
            "required_rules": [],
            "reusable_rules": ["hash_membership", "set_intersection"],
            "notes": [
                "Concise and usually fast for medium-sized collections.",
                "Builds two sets even when one side is much smaller.",
            ],
        },
        {
            "agent": "replay-synthesizer",
            "label": "Replay-guided smaller-side set",
            "strategy": "Use replay memory to materialize only the smaller side as a set and stream the other collection through it.",
            "code": _code(
                """
                def has_overlap(left, right):
                    if len(left) <= len(right):
                        seen = set(left)
                        probe = right
                    else:
                        seen = set(right)
                        probe = left
                    for value in probe:
                        if value in seen:
                            return True
                    return False
                """
            ),
            "complexity": 0.27,
            "uses_memory": True,
            "required_rules": ["hash_membership"],
            "reusable_rules": ["hash_membership", "materialize_smaller_side"],
            "notes": [
                "Generalizes the hash-membership lesson from earlier tasks.",
                "Avoids unnecessary work when one side is much smaller.",
            ],
        },
    ],
}


def load_tasks() -> list[dict[str, Any]]:
    raw_tasks = json.loads((DATA / "tasks.json").read_text())
    tasks = []
    for task in raw_tasks:
        baseline_path = ROOT / task["baseline_path"]
        enriched = dict(task)
        enriched["baseline_path"] = str(baseline_path)
        enriched["baseline_code"] = baseline_path.read_text()
        enriched["candidate_specs"] = CANDIDATE_SPECS[task["id"]]
        tasks.append(enriched)
    return tasks


def list_task_summaries() -> list[dict[str, str]]:
    return [
        {
            "id": task["id"],
            "title": task["title"],
            "description": task["description"],
            "baseline_path": task["baseline_path"],
        }
        for task in load_tasks()
    ]
