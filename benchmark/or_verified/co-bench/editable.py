def build_run_config() -> dict:
    return {
        "timeout_s": 10,
        "download_max_workers": 2,
    }


RUN_CONFIG = build_run_config()

SYSTEM_PROMPT = (
    "You are an expert in combinatorial optimization and algorithm design. "
    "Return only a JSON object with keys algorithm_summary and python_code. "
    "python_code must define the exact function required by the prompt, without Markdown fences."
)


def build_user_prompt(problem_description: str) -> str:
    return (
        "Solve the following CO-Bench task by writing Python code for the required solve function.\n\n"
        f"{problem_description.strip()}\n"
    )
