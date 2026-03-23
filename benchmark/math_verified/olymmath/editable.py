def solve(question: dict) -> str:
    prompt = str(question.get("prompt") or "").strip()
    if not prompt:
        return ""
    choices = question.get("choices") or []
    if choices:
        return str(choices[0])
    return "0"
