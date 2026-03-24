from __future__ import annotations

from copy import deepcopy
from typing import Any


PRIMARY_SCORE_FORMULA = (
    "primary_score = objective_score "
    "(the task-facing objective normalized so that higher is always better)"
)
TIE_BREAK_SCORE_FORMULA = "tie_break_score is task-specific; see task.selection_spec.tie_break_formula"
DELTA_PRIMARY_SCORE_FORMULA = "delta_primary_score = primary_score(generation_winner) - primary_score(selected_parent)"
RUN_DELTA_PRIMARY_SCORE_FORMULA = "run_delta_primary_score = primary_score(final_winner) - primary_score(baseline)"
PROPOSAL_SELECTION_GUIDANCE = (
    "Selection is layered: satisfy the verifier gate first, then improve primary_score. "
    "Tie-break metrics only matter among candidates with essentially the same primary_score."
)

LINE_COUNT_NORMALIZER = 10.0
PLAN_STEP_NORMALIZER = 20.0
COMMAND_COUNT_NORMALIZER = 20.0


def _gate_rule(metric: str, op: str, threshold: object, *, label: str | None = None) -> dict[str, Any]:
    return {
        "metric": metric,
        "op": op,
        "threshold": threshold,
        "label": label or f"{metric} {op} {threshold!r}",
    }


def _tie_break_metric(
    metric: str,
    *,
    direction: str,
    weight: float,
    normalizer: float = 1.0,
    label: str | None = None,
) -> dict[str, Any]:
    return {
        "metric": metric,
        "direction": direction,
        "weight": float(weight),
        "normalizer": float(normalizer),
        "label": label or metric,
    }


SELECTION_PROFILES: dict[str, dict[str, Any]] = {
    "objective_only": {
        "summary_template": (
            "Candidates must satisfy the verifier gate. Winner selection uses primary_score only; "
            "no auxiliary tie-break metrics are configured."
        ),
        "gate": [_gate_rule("verifier_status", "==", "pass", label="verified pass")],
        "tie_break_metrics": [],
        "archive_features": [],
    },
    "optimization": {
        "summary_template": (
            "Candidates must satisfy the verifier gate. Accepted mutations improve primary_score first; "
            "stability and code-shape metrics only break ties among valid optimizations."
        ),
        "gate": [_gate_rule("verifier_status", "==", "pass", label="verified pass")],
        "tie_break_metrics": [
            _tie_break_metric("stability", direction="max", weight=0.20, label="stability"),
            _tie_break_metric("complexity", direction="min", weight=0.18, label="complexity"),
            _tie_break_metric(
                "line_count",
                direction="min",
                weight=0.05,
                normalizer=LINE_COUNT_NORMALIZER,
                label=f"line_count / {int(LINE_COUNT_NORMALIZER)}",
            ),
        ],
        "archive_features": ["complexity", "line_count"],
    },
    "terminal_commands": {
        "summary_template": (
            "Candidates must satisfy the verifier gate. Primary selection uses task success first; "
            "fewer generated commands only break ties among equally successful candidates."
        ),
        "gate": [_gate_rule("verifier_status", "==", "pass", label="verified pass")],
        "tie_break_metrics": [
            _tie_break_metric(
                "avg_command_count",
                direction="min",
                weight=1.0,
                normalizer=COMMAND_COUNT_NORMALIZER,
                label=f"avg_command_count / {int(COMMAND_COUNT_NORMALIZER)}",
            )
        ],
        "archive_features": ["avg_command_count"],
    },
    "plan_length": {
        "summary_template": (
            "Candidates must satisfy the verifier gate. Primary selection uses solved ratio first; "
            "shorter valid plans only break ties among equally successful candidates."
        ),
        "gate": [_gate_rule("verifier_status", "==", "pass", label="verified pass")],
        "tie_break_metrics": [
            _tie_break_metric(
                "avg_plan_steps",
                direction="min",
                weight=1.0,
                normalizer=PLAN_STEP_NORMALIZER,
                label=f"avg_plan_steps / {int(PLAN_STEP_NORMALIZER)}",
            )
        ],
        "archive_features": ["avg_plan_steps"],
    },
}


def _default_profile_name(task: dict[str, Any]) -> str:
    answer_metric = str(task.get("answer_metric") or "").strip().lower()
    track = str(task.get("track") or "").strip().lower()
    if answer_metric == "speedup_vs_baseline":
        return "optimization"
    if track == "planning_verified":
        return "plan_length"
    if track == "terminal_verified":
        return "terminal_commands"
    return "objective_only"


def _normalize_gate_rules(raw_rules: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw_rule in raw_rules or []:
        if not isinstance(raw_rule, dict):
            continue
        metric = str(raw_rule.get("metric") or "").strip()
        op = str(raw_rule.get("op") or "").strip()
        if not metric or not op:
            continue
        normalized.append(
            _gate_rule(
                metric,
                op,
                raw_rule.get("threshold"),
                label=str(raw_rule.get("label") or "").strip() or None,
            )
        )
    return normalized


def _normalize_tie_break_metrics(raw_metrics: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw_metric in raw_metrics or []:
        if not isinstance(raw_metric, dict):
            continue
        metric = str(raw_metric.get("metric") or "").strip()
        direction = str(raw_metric.get("direction") or "").strip().lower()
        if not metric or direction not in {"max", "min"}:
            continue
        normalized.append(
            _tie_break_metric(
                metric,
                direction=direction,
                weight=float(raw_metric.get("weight") or 0.0),
                normalizer=float(raw_metric.get("normalizer") or 1.0),
                label=str(raw_metric.get("label") or "").strip() or None,
            )
        )
    return normalized


def _format_threshold(value: object) -> str:
    if isinstance(value, str):
        return repr(value)
    return str(value)


def _render_gate_summary(gate_rules: list[dict[str, Any]]) -> str:
    if not gate_rules:
        return "gate: none"
    fragments = [f"{rule['metric']} {rule['op']} {_format_threshold(rule.get('threshold'))}" for rule in gate_rules]
    return "gate: " + "; ".join(fragments)


def _render_tie_break_formula(tie_break_metrics: list[dict[str, Any]]) -> str:
    if not tie_break_metrics:
        return "tie_break_score = 0.0 (no auxiliary tie-break metrics configured)"
    parts: list[str] = []
    for item in tie_break_metrics:
        metric = str(item["metric"])
        direction = str(item["direction"])
        weight = float(item["weight"])
        normalizer = float(item.get("normalizer") or 1.0)
        sign = "+" if direction == "max" else "-"
        if normalizer != 1.0:
            metric_text = f"({metric} / {normalizer:g})"
        else:
            metric_text = metric
        parts.append(f"{sign} {weight:.2f} * {metric_text}")
    rendered = " ".join(parts)
    if rendered.startswith("+ "):
        rendered = rendered[2:]
    return f"tie_break_score = {rendered}"


def _render_archive_summary(archive_features: list[str]) -> str:
    if not archive_features:
        return "archive_features = none"
    return "archive_features = " + ", ".join(archive_features)


def selection_spec_for_task(task: dict[str, Any]) -> dict[str, Any]:
    raw_override = dict(task.get("selection_spec") or {})
    profile_name = str(raw_override.get("profile") or _default_profile_name(task)).strip().lower()
    profile = deepcopy(SELECTION_PROFILES.get(profile_name) or SELECTION_PROFILES["objective_only"])

    if "summary_template" in raw_override:
        profile["summary_template"] = str(raw_override.get("summary_template") or "").strip() or profile["summary_template"]
    if "gate" in raw_override:
        profile["gate"] = _normalize_gate_rules(raw_override.get("gate"))
    else:
        profile["gate"] = _normalize_gate_rules(profile.get("gate"))
    if "tie_break_metrics" in raw_override:
        profile["tie_break_metrics"] = _normalize_tie_break_metrics(raw_override.get("tie_break_metrics"))
    else:
        profile["tie_break_metrics"] = _normalize_tie_break_metrics(profile.get("tie_break_metrics"))
    if "archive_features" in raw_override:
        archive_features = raw_override.get("archive_features") or []
        profile["archive_features"] = [str(item).strip() for item in archive_features if str(item).strip()]
    else:
        profile["archive_features"] = [str(item).strip() for item in profile.get("archive_features", []) if str(item).strip()]

    profile["profile"] = profile_name
    profile["display_name"] = str(raw_override.get("display_name") or "Layered selection policy")
    profile["primary_metric"] = "objective_score"
    profile["primary_label"] = "Normalized objective score"
    profile["primary_direction"] = "max"
    profile["primary_formula"] = PRIMARY_SCORE_FORMULA
    profile["gate_summary"] = _render_gate_summary(profile["gate"])
    profile["tie_break_formula"] = _render_tie_break_formula(profile["tie_break_metrics"])
    profile["delta_template"] = (
        "evolve metric compares the generation winner against the selected parent. "
        "Latest report improvement is shown separately against the round-1 winner."
    )
    profile["archive_summary"] = _render_archive_summary(profile["archive_features"])
    return profile


def metric_value(metrics: dict[str, Any], metric: str) -> Any:
    if metric in metrics:
        return metrics[metric]
    return None


def evaluate_gate(selection_spec: dict[str, Any], metrics: dict[str, Any]) -> bool:
    gate_rules = list(selection_spec.get("gate") or [])
    if not gate_rules:
        return True
    for rule in gate_rules:
        metric = str(rule.get("metric") or "")
        op = str(rule.get("op") or "")
        threshold = rule.get("threshold")
        value = metric_value(metrics, metric)
        if op == "==":
            if value != threshold:
                return False
        elif op == "!=":
            if value == threshold:
                return False
        else:
            try:
                numeric_value = float(value)
                numeric_threshold = float(threshold)
            except (TypeError, ValueError):
                return False
            if op == ">=" and not (numeric_value >= numeric_threshold):
                return False
            if op == ">" and not (numeric_value > numeric_threshold):
                return False
            if op == "<=" and not (numeric_value <= numeric_threshold):
                return False
            if op == "<" and not (numeric_value < numeric_threshold):
                return False
    return True


def compute_tie_break_score(selection_spec: dict[str, Any], metrics: dict[str, Any]) -> float:
    score = 0.0
    for rule in selection_spec.get("tie_break_metrics") or []:
        value = metric_value(metrics, str(rule.get("metric") or ""))
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        normalizer = float(rule.get("normalizer") or 1.0) or 1.0
        weight = float(rule.get("weight") or 0.0)
        direction = str(rule.get("direction") or "max")
        signed = numeric_value / normalizer
        score += weight * signed if direction == "max" else -weight * signed
    return round(score, 6)


def status_rank(status: object) -> int:
    text = str(status or "").strip().lower()
    if text == "pass":
        return 2
    if text == "fail":
        return 1
    return 0


def metrics_rank(metrics: dict[str, Any]) -> tuple[int, float, float, int]:
    return (
        1 if bool(metrics.get("gate_passed")) else 0,
        float(metrics.get("primary_score") or 0.0),
        float(metrics.get("tie_break_score") or 0.0),
        status_rank(metrics.get("verifier_status") or metrics.get("status")),
    )


def prompt_summary(selection_spec: dict[str, Any]) -> str:
    return (
        f"Selection profile: {selection_spec.get('profile')}\n"
        f"{selection_spec.get('gate_summary')}\n"
        f"{selection_spec.get('primary_formula')}\n"
        f"{selection_spec.get('tie_break_formula')}\n"
        f"{selection_spec.get('archive_summary')}"
    )
