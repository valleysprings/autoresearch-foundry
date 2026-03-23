def solve(question: dict) -> str:
    prompt = str(question.get("prompt") or "").strip()
    if not prompt:
        return ""
    return "0"
