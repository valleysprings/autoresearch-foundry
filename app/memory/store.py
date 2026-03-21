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
    ) -> list[dict[str, Any]]:
        scored: list[tuple[float, dict[str, Any]]] = []
        target = set(task_signature)
        for item in self.load():
            overlap = len(target & set(item.get("task_signature", [])))
            family_name = item.get("family", "agnostic")
            family_bonus = 2.0 if family_name == family else 1.0 if family_name == "agnostic" else 0.0
            impact_bonus = min(float(item.get("delta_J", 0.0)), 1.0)
            score = overlap * 3.0 + family_bonus + impact_bonus
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
        memories.append(dict(experience))
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
