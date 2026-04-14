from __future__ import annotations


def solve(question: dict) -> str:
    choices = question.get("choices") or []
    if choices:
        return "A"
    return "yes"
