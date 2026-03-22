from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.memory.store import MemoryStore


SUCCESS_MEMORY = {
    "experience_id": "exp-success",
    "experience_type": "strategy_experience",
    "experience_outcome": "success",
    "source_task": "contains-duplicates",
    "source_session_id": "session-a",
    "family": "set-logic",
    "task_signature": ["python-codegen", "set-logic"],
    "verifier_status": "pass",
    "rejection_reason": "",
    "failure_pattern": "Quadratic rescans wasted time.",
    "strategy_hypothesis": "Hash membership will dominate nested scans.",
    "successful_strategy": "Use a set-based duplicate detector.",
    "prompt_fragment": "Prefer a hash-based duplicate detector that preserves semantics.",
    "tool_trace_summary": "tests pass -> benchmark improves -> write back",
    "delta_J": 0.4,
    "proposal_model": "deepseek-chat",
    "candidate_summary": "Set membership duplicate detection.",
    "supporting_memory_ids": [],
}

FAILURE_MEMORY = {
    "experience_id": "exp-failure",
    "experience_type": "strategy_experience",
    "experience_outcome": "failure",
    "source_task": "deduplicate-preserve-order",
    "source_session_id": "session-b",
    "family": "set-logic",
    "task_signature": ["python-codegen", "set-logic"],
    "verifier_status": "fail",
    "rejection_reason": "Sorting changed order semantics.",
    "failure_pattern": "A reordering shortcut broke stable output order.",
    "strategy_hypothesis": "Order-sensitive tasks should optimize with streaming state, not sorting.",
    "successful_strategy": "Use a seen set while preserving the original traversal order.",
    "prompt_fragment": "Do not use a reordering shortcut when stable order is part of the task contract.",
    "tool_trace_summary": "sorted candidate -> deterministic failure -> reject",
    "delta_J": -0.7,
    "proposal_model": "deepseek-chat",
    "candidate_summary": "Sort-first shortcut that changed semantics.",
    "supporting_memory_ids": [],
}

NEW_SEED_MEMORY = {
    "experience_id": "exp-seed-extra",
    "experience_type": "strategy_experience",
    "experience_outcome": "success",
    "source_task": "seed",
    "source_session_id": "seed-catalog",
    "family": "agnostic",
    "task_signature": ["python-codegen"],
    "verifier_status": "pass",
    "rejection_reason": "",
    "failure_pattern": "Candidates skipped correctness gating.",
    "strategy_hypothesis": "Correctness-first selection prevents fast but wrong winners.",
    "successful_strategy": "Run fixed tests before trusting benchmarks.",
    "prompt_fragment": "Preserve semantics first, benchmark only verified candidates.",
    "tool_trace_summary": "seed memory",
    "delta_J": 0.2,
    "proposal_model": "seed",
    "candidate_summary": "Correctness-first gating.",
    "supporting_memory_ids": [],
}


class MemoryStoreTest(unittest.TestCase):
    def test_ensure_seed_records_preserves_existing_memories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = MemoryStore(Path(tmp_dir) / "memory.json", markdown_path=Path(tmp_dir) / "memory.md")
            store.ensure_seed_records([SUCCESS_MEMORY])
            self.assertTrue(store.append(FAILURE_MEMORY))

            merged = store.ensure_seed_records([SUCCESS_MEMORY, NEW_SEED_MEMORY])

            self.assertEqual(len(merged), 3)
            self.assertEqual(store.count(), 3)
            self.assertTrue(any(item["experience_id"] == "exp-failure" for item in merged))
            self.assertTrue(any(item["experience_id"] == "exp-seed-extra" for item in merged))

    def test_retrieve_and_markdown_include_success_and_failure_experiences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = MemoryStore(Path(tmp_dir) / "memory.json", markdown_path=Path(tmp_dir) / "memory.md")
            store.seed_from_records([SUCCESS_MEMORY, FAILURE_MEMORY])

            retrieved = store.retrieve(task_signature=["python-codegen", "set-logic"], family="set-logic", top_k=2)
            markdown = store.load_markdown()

            self.assertEqual(len(retrieved), 2)
            self.assertEqual(retrieved[0]["experience_outcome"], "success")
            self.assertEqual(retrieved[1]["experience_outcome"], "failure")
            self.assertIn("success_memories: 1", markdown)
            self.assertIn("failure_memories: 1", markdown)
            self.assertIn("experience_outcome: failure", markdown)
            self.assertIn("strategy_hypothesis:", markdown)
            self.assertIn("prompt_fragment:", markdown)

    def test_append_deduplicates_equivalent_memory_fragments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = MemoryStore(Path(tmp_dir) / "memory.json", markdown_path=Path(tmp_dir) / "memory.md")
            store.seed_from_records([SUCCESS_MEMORY])

            duplicate = dict(SUCCESS_MEMORY)
            duplicate["experience_id"] = "exp-success-duplicate"

            self.assertFalse(store.append(duplicate))
            self.assertEqual(store.count(), 1)


if __name__ == "__main__":
    unittest.main()
