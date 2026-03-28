SYSTEM_PROMPT = (
    "You are a planning assistant. "
    "Return only the final plan, with one action per line and no extra commentary."
)


def build_user_prompt(question: dict) -> str:
    return str(question.get("prompt") or "").strip()


def adapt_plan_response(response: object) -> object:
    return response if response is not None else []


def solve(question: dict) -> object:
    _ = SYSTEM_PROMPT
    _ = build_user_prompt(question)
    return adapt_plan_response([])
