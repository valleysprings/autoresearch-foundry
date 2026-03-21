from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MemoryStore:
    def __init__(self, path: Path):
        self.path = path

    def seed_from(self, source_path: Path) -> list[dict[str, Any]]:
        seed_memories = json.loads(source_path.read_text())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(seed_memories, indent=2))
        return seed_memories

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text())

    def count(self) -> int:
        return len(self.load())

    def retrieve(
        self,
        task_signature: list[str],
        target_device: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        scored: list[tuple[float, dict[str, Any]]] = []
        target = set(task_signature)

        for item in self.load():
            overlap = len(target & set(item.get("task_signature", [])))
            device = item.get("target_device", "agnostic")
            device_bonus = 2.0 if device == target_device else 1.0 if device == "agnostic" else 0.0
            impact_bonus = min(float(item.get("delta_J", 0.0)), 1.0)
            score = overlap * 3.0 + device_bonus + impact_bonus
            if score <= 0:
                continue
            enriched = dict(item)
            enriched["retrieval_score"] = round(score, 3)
            scored.append((score, enriched))

        scored.sort(key=lambda pair: (pair[0], pair[1].get("delta_J", 0.0)), reverse=True)
        return [item for _, item in scored[:top_k]]

    def append(self, experience: dict[str, Any]) -> bool:
        memories = self.load()
        existing_ids = {item.get("experience_id") for item in memories}
        if experience.get("experience_id") in existing_ids:
            return False
        memories.append(experience)
        self.path.write_text(json.dumps(memories, indent=2))
        return True
