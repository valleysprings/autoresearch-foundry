from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt


FAMILY_STYLE = {
    "set-logic": {
        "objective": "#c4581c",
        "j": "#165f4f",
        "accent": "#8f3c12",
        "subtitle": "Set membership and duplicate-detection workloads reward fast hash-based transitions.",
        "secondary_title": "Retrieved memory by generation",
    },
    "counting": {
        "objective": "#9b3d8a",
        "j": "#1a6d8a",
        "accent": "#6b2b61",
        "subtitle": "Counting workloads expose whether a candidate consolidates repeated work into one linear pass.",
        "secondary_title": "Generation winner objective",
    },
    "numeric": {
        "objective": "#3366cc",
        "j": "#1f8f6d",
        "accent": "#244b99",
        "subtitle": "Numeric search tasks highlight whether arithmetic structure replaces repeated scans.",
        "secondary_title": "Candidate frontier",
    },
    "math": {
        "objective": "#8c4f1c",
        "j": "#2f6f3e",
        "accent": "#5f3411",
        "subtitle": "Harder combinatorics and number-theory tasks reward stepping-stone improvements, not only one-shot wins.",
        "secondary_title": "Candidate frontier",
    },
}


def _numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_improvement_table(run: dict[str, Any]) -> list[dict[str, str]]:
    baseline_metrics = run["baseline"]["metrics"]
    winner_metrics = run["winner"]["metrics"]
    baseline_objective = _numeric(baseline_metrics.get("objective"))
    winner_objective = _numeric(winner_metrics.get("objective"))
    baseline_j = _numeric(baseline_metrics.get("J"))
    winner_j = _numeric(winner_metrics.get("J"))
    baseline_benchmark = baseline_metrics.get("benchmark_ms")
    winner_benchmark = winner_metrics.get("benchmark_ms")
    accepted_generations = sum(1 for point in run.get("objective_curve", [])[1:] if point.get("accepted"))
    best_improvements = sum(1 for point in run.get("objective_curve", [])[1:] if point.get("improved_global_best"))

    return [
        {"label": "Task", "value": str(run["task"]["id"])},
        {"label": "Family", "value": str(run["task"]["family"])},
        {"label": "Session", "value": str(run.get("session_id", "n/a"))},
        {"label": "Winner", "value": str(run["winner"]["label"])},
        {"label": "Model", "value": str(run.get("active_model", "n/a"))},
        {"label": "Baseline objective", "value": f"{baseline_objective:.4f}"},
        {"label": "Winner objective", "value": f"{winner_objective:.4f}"},
        {"label": "Objective gain", "value": f"{winner_objective - baseline_objective:+.4f}"},
        {"label": "Baseline J", "value": f"{baseline_j:.4f}"},
        {"label": "Winner J", "value": f"{winner_j:.4f}"},
        {"label": "J gain", "value": f"{winner_j - baseline_j:+.4f}"},
        {
            "label": "Baseline benchmark",
            "value": "n/a" if baseline_benchmark is None else f"{_numeric(baseline_benchmark):.3f} ms",
        },
        {
            "label": "Winner benchmark",
            "value": "n/a" if winner_benchmark is None else f"{_numeric(winner_benchmark):.3f} ms",
        },
        {"label": "Frontier accepts", "value": str(accepted_generations)},
        {"label": "Global-best improves", "value": str(best_improvements)},
        {"label": "Write-backs", "value": str(len(run.get("memory_events", [])))},
    ]


def _family_style(run: dict[str, Any]) -> dict[str, str]:
    family = str(run["task"].get("family", "set-logic"))
    return FAMILY_STYLE.get(family, FAMILY_STYLE["set-logic"])


def _render_secondary_panel(run: dict[str, Any], axis: Any, style: dict[str, str]) -> None:
    family = str(run["task"].get("family", "set-logic"))
    axis.set_title(style["secondary_title"], loc="left", fontsize=12, pad=10, fontweight="bold")
    if family == "set-logic":
        generations = [generation["generation"] for generation in run.get("generations", [])]
        memory_counts = [len(generation.get("retrieved_memories", [])) for generation in run.get("generations", [])]
        writeback_flags = [1 if generation.get("wrote_memory") else 0 for generation in run.get("generations", [])]
        axis.bar(generations, memory_counts, color="#ecd6c8", edgecolor=style["accent"], linewidth=1.1)
        axis.plot(generations, writeback_flags, color=style["j"], marker="o", linewidth=2.0, label="write-back")
        axis.set_xlabel("Generation")
        axis.set_ylabel("Memory count")
        axis.legend(frameon=False, loc="upper left")
        return
    if family == "counting":
        generations = [generation["generation"] for generation in run.get("generations", [])]
        winner_objectives = [_numeric(generation["winner"]["metrics"].get("objective")) for generation in run.get("generations", [])]
        axis.bar(generations, winner_objectives, color=style["objective"], alpha=0.84, width=0.56)
        axis.set_xlabel("Generation")
        axis.set_ylabel("Objective")
        for generation, objective in zip(generations, winner_objectives):
            axis.text(generation, objective, f"{objective:.1f}", ha="center", va="bottom", fontsize=9)
        return

    candidate_objectives = [_numeric(point.get("candidate_objective")) for point in run.get("objective_curve", [])[1:]]
    candidate_js = [_numeric(point.get("candidate_J")) for point in run.get("objective_curve", [])[1:]]
    generations = [int(point["generation"]) for point in run.get("objective_curve", [])[1:]]
    scatter = axis.scatter(candidate_objectives, candidate_js, c=generations, cmap="viridis", s=80, edgecolors="white", linewidths=1.0)
    axis.set_xlabel("Candidate objective")
    axis.set_ylabel("Candidate J")
    colorbar = plt.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Generation")


def write_improvement_report_svg(run: dict[str, Any], out_path: Path) -> None:
    points = run.get("objective_curve", [])
    if not points:
        raise ValueError("Run did not include an objective curve.")

    generations = [int(point["generation"]) for point in points]
    baseline_objective = _numeric(points[0]["objective"])
    baseline_j = _numeric(points[0]["J"])
    objective_deltas = [_numeric(point["objective"]) - baseline_objective for point in points]
    j_deltas = [_numeric(point["J"]) - baseline_j for point in points]
    accepted = [bool(point.get("accepted")) for point in points]

    style = _family_style(run)
    baseline_benchmark = _numeric(run["baseline"]["metrics"].get("benchmark_ms"))
    winner_benchmark = _numeric(run["winner"]["metrics"].get("benchmark_ms"))
    summary_rows = build_improvement_table(run)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(13.2, 8.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[2.6, 1.45], width_ratios=[1.2, 1.15, 1.05])
    curve_ax = fig.add_subplot(grid[0, :])
    secondary_ax = fig.add_subplot(grid[1, 0])
    benchmark_ax = fig.add_subplot(grid[1, 1])
    table_ax = fig.add_subplot(grid[1, 2])

    objective_color = style["objective"]
    j_color = style["j"]

    curve_ax.plot(generations, objective_deltas, color=objective_color, marker="o", linewidth=3.0, label="objective delta")
    curve_ax.plot(generations, j_deltas, color=j_color, marker="o", linewidth=3.0, label="J delta")
    curve_ax.fill_between(generations, objective_deltas, 0.0, color=objective_color, alpha=0.10)
    curve_ax.fill_between(generations, j_deltas, 0.0, color=j_color, alpha=0.08)
    curve_ax.axhline(0.0, color="#5d6470", linewidth=1.1, linestyle="--", alpha=0.6)
    for generation, objective_delta, j_delta, is_accepted in zip(generations, objective_deltas, j_deltas, accepted):
        if not is_accepted:
            continue
        curve_ax.scatter([generation], [objective_delta], color=objective_color, s=70, edgecolors="white", linewidths=1.3, zorder=5)
        curve_ax.scatter([generation], [j_delta], color=j_color, s=70, edgecolors="white", linewidths=1.3, zorder=5)
    curve_ax.set_title(f"{run['task']['title']}  |  Improvement vs baseline", loc="left", fontsize=15, pad=12, fontweight="bold")
    curve_ax.set_xlabel("Generation")
    curve_ax.set_ylabel("Delta")
    curve_ax.legend(frameon=False, loc="upper left")
    curve_ax.text(
        0.01,
        1.03,
        style["subtitle"],
        transform=curve_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#5d6470",
    )
    curve_ax.text(
        0.99,
        0.04,
        f"Winner: {run['winner']['label']}  |  model={run.get('active_model', 'n/a')}  |  session={run.get('session_id', 'n/a')}",
        transform=curve_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="#5d6470",
    )

    _render_secondary_panel(run, secondary_ax, style)

    benchmark_ax.barh(["baseline", "winner"], [baseline_benchmark, winner_benchmark], color=["#d6c9b9", j_color])
    benchmark_ax.set_title("Benchmark time (ms)", loc="left", fontsize=12, pad=10, fontweight="bold")
    benchmark_ax.set_xlabel("Milliseconds")
    for index, value in enumerate([baseline_benchmark, winner_benchmark]):
        benchmark_ax.text(value, index, f" {value:.3f}", va="center", ha="left", fontsize=9.5, color="#18212a")

    table_ax.axis("off")
    table = table_ax.table(
        cellText=[[row["label"], row["value"]] for row in summary_rows],
        colLabels=["Metric", "Value"],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.7)
    table.scale(1.0, 1.18)
    for (row_index, _col_index), cell in table.get_celld().items():
        cell.set_edgecolor("#d8ccbb")
        if row_index == 0:
            cell.set_facecolor("#efe2d2")
            cell.set_text_props(weight="bold", color="#18212a")
        else:
            cell.set_facecolor("#fff9f1")

    fig.patch.set_facecolor("#fff9f1")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg", facecolor=fig.get_facecolor())
    plt.close(fig)
