from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.memory.markdown import render_memory_markdown


class MemoryStore:
    def __init__(self, path: Path, markdown_path: Path | None = None, title: str = "Working Memory"):
        self.path = path
        self.markdown_path = markdown_path
        self.title = title

    def seed_from_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seed_memories = [dict(record) for record in records]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(seed_memories, indent=2))
        self._write_markdown(seed_memories)
        return seed_memories

    def ensure_seed_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        memories = self.load()
        if not memories:
            return self.seed_from_records(records)
        existing_ids = {item.get("experience_id") for item in memories}
        new_records = [dict(record) for record in records if record.get("experience_id") not in existing_ids]
        if not new_records:
            self._write_markdown(memories)
            return memories
        merged = memories + new_records
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(merged, indent=2))
        self._write_markdown(merged)
        return merged

    def seed_from(self, source_path: Path) -> list[dict[str, Any]]:
        return self.seed_from_records(json.loads(source_path.read_text()))

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text())

    def count(self) -> int:
        return len(self.load())

    def retrieve(
        self,
        *,
        task_signature: list[str],
        family: str,
        top_k: int = 3,
        failure_top_k: int = 1,
    ) -> list[dict[str, Any]]:
        success_scored: list[tuple[float, dict[str, Any]]] = []
        failure_scored: list[tuple[float, dict[str, Any]]] = []
        target = set(task_signature)
        for item in self.load():
            overlap = len(target & set(item.get("task_signature", [])))
            family_name = item.get("family", "agnostic")
            family_bonus = 2.0 if family_name == family else 1.0 if family_name == "agnostic" else 0.0
            impact_bonus = min(abs(self._delta_primary_score(item)), 1.0)
            outcome = item.get("experience_outcome", "success")
            verifier_status = item.get("verifier_status", "")
            if outcome == "failure" and verifier_status == "pass":
                # Passing-but-stagnant attempts are usually prompt noise, not reusable avoidance memory.
                continue
            outcome_bonus = 0.25 if outcome == "success" else 0.05 if outcome == "failure" else 0.0
            score = overlap * 3.0 + family_bonus + impact_bonus + outcome_bonus
            if score <= 0:
                continue
            enriched = dict(item)
            enriched["retrieval_score"] = round(score, 3)
            if outcome == "failure":
                failure_scored.append((score, enriched))
            else:
                success_scored.append((score, enriched))

        def _sort_key(pair: tuple[float, dict[str, Any]]) -> tuple[float, float]:
            return pair[0], abs(self._delta_primary_score(pair[1]))

        success_scored.sort(key=_sort_key, reverse=True)
        failure_scored.sort(key=_sort_key, reverse=True)
        selected = [item for _, item in success_scored[:top_k]]
        remaining = max(top_k - len(selected), 0)
        if remaining > 0:
            selected.extend(item for _, item in failure_scored[: min(failure_top_k, remaining)])

        if len(selected) < top_k:
            leftovers = success_scored[top_k:] + failure_scored[min(failure_top_k, remaining) :]
            leftovers.sort(key=_sort_key, reverse=True)
            selected.extend(item for _, item in leftovers[: top_k - len(selected)])
        return selected[:top_k]

    def append(self, experience: dict[str, Any]) -> bool:
        memories = self.load()
        existing_ids = {item.get("experience_id") for item in memories}
        if experience.get("experience_id") in existing_ids:
            return False
        signature = self._signature(experience)
        if signature and signature in {self._signature(item) for item in memories}:
            return False
        memories.append(dict(experience))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(memories, indent=2))
        self._write_markdown(memories)
        return True

    def load_markdown(self) -> str:
        if self.markdown_path is None or not self.markdown_path.exists():
            return ""
        return self.markdown_path.read_text()

    def _write_markdown(self, memories: list[dict[str, Any]]) -> None:
        if self.markdown_path is None:
            return
        self.markdown_path.parent.mkdir(parents=True, exist_ok=True)
        self.markdown_path.write_text(render_memory_markdown(memories, title=self.title))

    @staticmethod
    def _signature(experience: dict[str, Any]) -> str:
        parts = [
            str(experience.get("source_task", "")).strip().lower(),
            str(experience.get("experience_outcome", "")).strip().lower(),
            str(experience.get("verifier_status", "")).strip().lower(),
            str(experience.get("failure_pattern", "")).strip().lower(),
            str(experience.get("successful_strategy", "")).strip().lower(),
            str(experience.get("prompt_fragment", "")).strip().lower(),
            str(experience.get("candidate_summary", "")).strip().lower(),
        ]
        normalized = " | ".join(part for part in parts if part)
        return normalized[:640]

    @staticmethod
    def _delta_primary_score(experience: dict[str, Any]) -> float:
        raw_value = experience.get("delta_primary_score", experience.get("delta_J", 0.0))
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return 0.0
