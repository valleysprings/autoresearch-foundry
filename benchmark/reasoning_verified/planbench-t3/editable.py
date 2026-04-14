SYSTEM_PROMPT = (
    "You are a planning verifier. "
    "Return only yes or no, with no extra commentary."
)


def build_user_prompt(question: dict) -> str:
    return str(question.get("prompt") or "").strip()


def solve(question: dict) -> str:
    _ = SYSTEM_PROMPT
    _ = build_user_prompt(question)
    return ""
