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
            "architecture_family": "sort-based",
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
            "architecture_family": "hash-based",
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
            "architecture_family": "replay-conditioned",
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
            "architecture_family": "counting",
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
            "architecture_family": "sort-based",
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
            "architecture_family": "replay-conditioned",
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
            "architecture_family": "sort-based",
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
            "architecture_family": "hash-based",
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
            "architecture_family": "replay-conditioned",
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
    "most-frequent-item": [
        {
            "agent": "count-optimizer",
            "label": "Two-pass counting table",
            "architecture_family": "counting",
            "strategy": "Build a frequency table and then select the maximum count.",
            "code": _code(
                """
                def most_frequent_item(values):
                    counts = {}
                    for value in values:
                        counts[value] = counts.get(value, 0) + 1
                    best_value = None
                    best_count = -1
                    for value, count in counts.items():
                        if count > best_count:
                            best_value = value
                            best_count = count
                    return best_value
                """
            ),
            "complexity": 0.32,
            "uses_memory": False,
            "required_rules": [],
            "reusable_rules": ["frequency_table", "count_then_select"],
            "notes": [
                "A direct architectural upgrade over the quadratic baseline.",
                "Separates counting from selection.",
            ],
        },
        {
            "agent": "collections-optimizer",
            "label": "Counter most_common",
            "architecture_family": "library-assisted",
            "strategy": "Use Counter and delegate the selection logic to the standard library.",
            "code": _code(
                """
                from collections import Counter

                def most_frequent_item(values):
                    return Counter(values).most_common(1)[0][0]
                """
            ),
            "complexity": 0.21,
            "uses_memory": False,
            "required_rules": [],
            "reusable_rules": ["counter_library"],
            "notes": [
                "Short and expressive.",
                "Usually wins when library overhead is dominated by the quadratic baseline.",
            ],
        },
        {
            "agent": "replay-synthesizer",
            "label": "One-pass running best",
            "architecture_family": "replay-conditioned",
            "strategy": "Reuse counted-structure intuition and maintain the current best answer during the counting pass.",
            "code": _code(
                """
                def most_frequent_item(values):
                    counts = {}
                    best_value = None
                    best_count = -1
                    for value in values:
                        counts[value] = counts.get(value, 0) + 1
                        if counts[value] > best_count:
                            best_value = value
                            best_count = counts[value]
                    return best_value
                """
            ),
            "complexity": 0.27,
            "uses_memory": True,
            "required_rules": ["count_then_scan"],
            "reusable_rules": ["frequency_table", "one_pass_best"],
            "notes": [
                "Moves selection into the same pass as counting.",
                "Acts like a lightweight architecture search over counting styles.",
            ],
        },
    ],
    "deduplicate-preserve-order": [
        {
            "agent": "dict-optimizer",
            "label": "dict.fromkeys",
            "architecture_family": "hash-based",
            "strategy": "Exploit insertion ordering in dict to preserve order and remove duplicates.",
            "code": _code(
                """
                def deduplicate_preserve_order(values):
                    return list(dict.fromkeys(values))
                """
            ),
            "complexity": 0.17,
            "uses_memory": False,
            "required_rules": [],
            "reusable_rules": ["ordered_hashing"],
            "notes": [
                "Compact and usually very fast.",
                "Uses language-level insertion ordering as the architecture trick.",
            ],
        },
        {
            "agent": "sort-optimizer",
            "label": "Sorted set",
            "architecture_family": "sort-based",
            "strategy": "Sort a set of values, which is fast but breaks the original order guarantee.",
            "code": _code(
                """
                def deduplicate_preserve_order(values):
                    return sorted(set(values))
                """
            ),
            "complexity": 0.14,
            "uses_memory": False,
            "required_rules": [],
            "reusable_rules": ["sort_then_scan"],
            "notes": [
                "This candidate intentionally violates task semantics.",
                "The evaluator should reject it like Karpathy would discard a faster but invalid run.",
            ],
        },
        {
            "agent": "replay-synthesizer",
            "label": "Streaming seen-set dedup",
            "architecture_family": "replay-conditioned",
            "strategy": "Reuse hash-membership memory and explicitly preserve order with a seen set and output buffer.",
            "code": _code(
                """
                def deduplicate_preserve_order(values):
                    seen = set()
                    result = []
                    for value in values:
                        if value not in seen:
                            seen.add(value)
                            result.append(value)
                    return result
                """
            ),
            "complexity": 0.24,
            "uses_memory": True,
            "required_rules": ["hash_membership"],
            "reusable_rules": ["hash_membership", "preserve_order"],
            "notes": [
                "More explicit than dict.fromkeys and easier to generalize.",
                "This is the replay version of stable deduplication.",
            ],
        },
    ],
    "missing-number": [
        {
            "agent": "sort-optimizer",
            "label": "Sort and detect gap",
            "architecture_family": "sort-based",
            "strategy": "Sort the values once and scan for the missing integer gap.",
            "code": _code(
                """
                def missing_number(values):
                    ordered = sorted(values)
                    if not ordered or ordered[0] != 0:
                        return 0
                    for index in range(1, len(ordered)):
                        if ordered[index] - ordered[index - 1] > 1:
                            return ordered[index - 1] + 1
                    return ordered[-1] + 1
                """
            ),
            "complexity": 0.33,
            "uses_memory": False,
            "required_rules": [],
            "reusable_rules": ["sort_then_scan"],
            "notes": [
                "A classical sorting architecture.",
                "Correct but heavier than arithmetic reasoning.",
            ],
        },
        {
            "agent": "formula-optimizer",
            "label": "Arithmetic checksum",
            "architecture_family": "analytic",
            "strategy": "Subtract the observed sum from the expected arithmetic progression.",
            "code": _code(
                """
                def missing_number(values):
                    upper = len(values)
                    expected = upper * (upper + 1) // 2
                    return expected - sum(values)
                """
            ),
            "complexity": 0.16,
            "uses_memory": False,
            "required_rules": [],
            "reusable_rules": ["closed_form_reasoning"],
            "notes": [
                "An architecture-level change from search to algebra.",
                "Most faithful to Karpathy's idea of genuinely trying a different architecture.",
            ],
        },
        {
            "agent": "replay-synthesizer",
            "label": "Replay-guided set lookup",
            "architecture_family": "replay-conditioned",
            "strategy": "Reuse hash-membership memory and check membership with a set instead of scanning the list repeatedly.",
            "code": _code(
                """
                def missing_number(values):
                    seen = set(values)
                    for candidate in range(len(values) + 1):
                        if candidate not in seen:
                            return candidate
                    return None
                """
            ),
            "complexity": 0.22,
            "uses_memory": True,
            "required_rules": ["hash_membership"],
            "reusable_rules": ["hash_membership", "set_lookup"],
            "notes": [
                "Bridges the set-logic family into a numeric task.",
                "Not the most elegant architecture, but replay makes it available.",
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
            "family": task["family"],
        }
        for task in load_tasks()
    ]
