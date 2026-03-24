def solve(question: dict) -> str:
    choices = question.get("choices") or []
    if choices:
        return str(choices[0])
    prompt = str(question.get("prompt") or "").strip()
    if not prompt:
        return ""
    return prompt.split()[0]
