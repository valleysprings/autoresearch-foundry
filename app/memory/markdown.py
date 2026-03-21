from __future__ import annotations

from datetime import datetime
from typing import Any


def render_memory_markdown(
    memories: list[dict[str, Any]],
    *,
    title: str,
    generated_at: str | None = None,
) -> str:
    timestamp = generated_at or datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        f"# {title}",
        "",
        f"- generated_at: {timestamp}",
        f"- num_memories: {len(memories)}",
        "",
        "## Experience Units",
        "",
    ]

    for item in memories:
        lines.extend(
            [
                f"### {item.get('experience_id', 'unknown-memory')}",
                "",
                f"- source_task: {item.get('source_task', 'unknown')}",
                f"- family: {item.get('family', 'agnostic')}",
                f"- delta_J: {item.get('delta_J', 0.0)}",
                f"- task_signature: {', '.join(item.get('task_signature', []))}",
                f"- failure_pattern: {item.get('failure_pattern', '')}",
                f"- strategy_hypothesis: {item.get('strategy_hypothesis', '')}",
                f"- successful_strategy: {item.get('successful_strategy', '')}",
                f"- prompt_fragment: {item.get('prompt_fragment', item.get('successful_strategy', ''))}",
                f"- candidate_summary: {item.get('candidate_summary', item.get('code_pattern', ''))}",
                f"- tool_trace_summary: {item.get('tool_trace_summary', '')}",
                f"- proposal_model: {item.get('proposal_model', '')}",
                f"- reusable_rules: {', '.join(item.get('reusable_rules', []))}",
                f"- supporting_memory_ids: {', '.join(item.get('supporting_memory_ids', [])) or 'none'}",
                "",
            ]
        )

    return "\n".join(lines).strip() + "\n"
