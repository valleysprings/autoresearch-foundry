from __future__ import annotations


MODEL_COMPLETION_MAX_ATTEMPTS = 3
TRIM_DEFAULT_LIMIT = 240
REQUEST_PREVIEW_LIMIT = 280
RAW_PREVIEW_LIMIT = 280
CANDIDATE_LABEL_LIMIT = 72
REFLECTION_FIELD_LIMIT = 320
CANDIDATE_RESPONSE_REQUIRED_FIELDS = (
    "name",
    "strategy",
    "rationale",
    "file_body",
    "candidate_summary",
)
REFLECTION_REQUIRED_FIELDS = (
    "failure_pattern",
    "strategy_hypothesis",
    "successful_strategy",
    "prompt_fragment",
    "tool_trace_summary",
)
PROPOSAL_SYSTEM_PROMPT = (
    "You are the only proposal model in a strict outer-loop editable-file Python optimization system. "
    "Return strict JSON with shape "
    '{"candidates":[{"name":"short label","strategy":"one sentence","rationale":"why it should win",'
    '"file_body":"full editable file contents","candidate_summary":"brief code summary"}]}.'
)
PROPOSAL_JSON_ONLY_INSTRUCTION = (
    "Return only a JSON object. Do not include Markdown code fences or any text before or after the JSON object."
)
PROPOSAL_CONCISE_FIELDS_INSTRUCTION = (
    "Keep name, strategy, rationale, and candidate_summary concise. file_body is the only field that may be long."
)
PROPOSAL_CANDIDATE_COUNT_TEMPLATE = "Return exactly {count} {noun}."
PROPOSAL_RESULT_INSTRUCTION = (
    "Return full editable-file rewrites that preserve the public contract, improve the selected parent, and avoid repeating recent no-op rewrites."
)
SUCCESS_REFLECTION_SYSTEM_PROMPT = (
    "You compress successful Python code mutations into reusable strategy memory. "
    "Return strict JSON with fields failure_pattern, strategy_hypothesis, successful_strategy, prompt_fragment, tool_trace_summary."
)
FAILURE_REFLECTION_SYSTEM_PROMPT = (
    "You compress failed or rejected Python code mutations into reusable avoidance memory. "
    "Return strict JSON with fields failure_pattern, strategy_hypothesis, successful_strategy, prompt_fragment, tool_trace_summary. "
    "successful_strategy must describe the corrective strategy to prefer next time, not the failed attempt."
)
SUCCESS_REFLECTION_OUTCOME_INSTRUCTIONS = "Write the prompt_fragment as a reusable success hint for the next proposal prompt."
FAILURE_REFLECTION_OUTCOME_INSTRUCTIONS = (
    "Write the prompt_fragment as a concise warning plus corrective strategy that the next proposal prompt can reuse."
)
REFLECTION_FRAGMENT_INSTRUCTION = "Write a compact strategy fragment that can be pasted into the next proposal prompt."
