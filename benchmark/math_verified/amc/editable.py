def solve(question: dict) -> str:
    choices = question.get("choices") or []
    if choices:
        return str(choices[0])
    return ""
