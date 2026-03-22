from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt


FAMILY_STYLE = {
    "set-logic": {"objective": "#c4581c", "j": "#165f4f", "accent": "#8f3c12", "memory": "#e4c2ad"},
    "counting": {"objective": "#ac6234", "j": "#245f7d", "accent": "#7a4320", "memory": "#d9c3aa"},
    "numeric": {"objective": "#3265b0", "j": "#1d7a69", "accent": "#22467b", "memory": "#bfd3f2"},
    "math": {"objective": "#8c4f1c", "j": "#2f6f3e", "accent": "#5f3411", "memory": "#dcc7ad"},
}


def _numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _family_style(run: dict[str, Any]) -> dict[str, str]:
    family = str(run["task"].get("family", "set-logic"))
    return FAMILY_STYLE.get(family, FAMILY_STYLE["set-logic"])


def write_improvement_report_svg(run: dict[str, Any], out_path: Path) -> None:
    points = run.get("objective_curve", [])
    if not points:
        raise ValueError("Run did not include an objective curve.")

    style = _family_style(run)
    objective_spec = run["task"].get("objective_spec") or {}
    objective_name = objective_spec.get("display_name") or run["task"].get("objective_label") or "objective"
    objective_direction = objective_spec.get("direction") or run["task"].get("objective_direction") or "max"
    direction_copy = "higher is better" if objective_direction == "max" else "lower is better"

    generations = [int(point["generation"]) for point in points]
    baseline_objective = _numeric(points[0]["objective"])
    baseline_j = _numeric(points[0]["J"])
    objective_deltas = [_numeric(point["objective"]) - baseline_objective for point in points]
    j_deltas = [_numeric(point["J"]) - baseline_j for point in points]
    branch_accepts = [int(point.get("accepted_count") or 0) for point in points]
    memory_deltas = [int(point.get("memory_delta") or 0) for point in points]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(13.2, 7.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[2.6, 1.3], width_ratios=[1.2, 1.0])
    curve_ax = fig.add_subplot(grid[0, :])
    memory_ax = fig.add_subplot(grid[1, 0])
    summary_ax = fig.add_subplot(grid[1, 1])

    curve_ax.plot(generations, objective_deltas, color=style["objective"], marker="o", linewidth=3.0, label=f"{objective_name} delta")
    curve_ax.plot(generations, j_deltas, color=style["j"], marker="o", linewidth=3.0, label="J delta")
    curve_ax.axhline(0.0, color="#6b6f77", linewidth=1.1, linestyle="--", alpha=0.55)
    for generation, accepted_count, objective_delta, j_delta in zip(generations, branch_accepts, objective_deltas, j_deltas):
        if accepted_count <= 0:
            continue
        curve_ax.scatter([generation], [objective_delta], color=style["objective"], s=80, edgecolors="white", linewidths=1.2, zorder=5)
        curve_ax.scatter([generation], [j_delta], color=style["j"], s=80, edgecolors="white", linewidths=1.2, zorder=5)
    curve_ax.set_title(f"{run['task']['title']} | generational deltas", loc="left", fontsize=15, pad=12, fontweight="bold")
    curve_ax.set_xlabel("Generation")
    curve_ax.set_ylabel("Delta vs baseline")
    curve_ax.legend(frameon=False, loc="upper left")
    curve_ax.text(
        0.01,
        1.03,
        f"{run['task']['description']} {objective_name}: {direction_copy}.",
        transform=curve_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#5d6470",
    )

    memory_ax.axhline(0.0, color="#6b6f77", linewidth=1.1, linestyle="--", alpha=0.55)
    bar_colors = [style["j"] if value >= 0 else style["accent"] for value in memory_deltas]
    memory_ax.bar(generations, memory_deltas, color=bar_colors, width=0.58)
    memory_ax.set_title("Memory delta by generation", loc="left", fontsize=12, pad=10, fontweight="bold")
    memory_ax.set_xlabel("Generation")
    memory_ax.set_ylabel("Net write-back")
    for generation, value, accepts in zip(generations, memory_deltas, branch_accepts):
        memory_ax.text(generation, value + (0.07 if value >= 0 else -0.16), f"{value:+d} | a{accepts}", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)

    summary_ax.axis("off")
    baseline_objective_value = _numeric(run["baseline"]["metrics"].get("objective"))
    winner_objective_value = _numeric(run["winner"]["metrics"].get("objective"))
    baseline_j_value = _numeric(run["baseline"]["metrics"].get("J"))
    winner_j_value = _numeric(run["winner"]["metrics"].get("J"))
    summary_lines = [
        run["task"]["description"],
        f"{objective_name}: {direction_copy}",
        f"Objective formula: {objective_spec.get('formula') or 'n/a'}",
        "J: internal selection score, always maximized",
        f"Winner: {run['winner']['label']} ({run['winner']['agent']})",
        f"Objective {baseline_objective_value:.3f} -> {winner_objective_value:.3f}",
        f"J {baseline_j_value:.4f} -> {winner_j_value:.4f}",
        f"run_delta_J {run.get('run_delta_J', run.get('delta_J', 0.0)):+.4f}",
        f"Accepted branches: {sum(branch_accepts)}",
        f"Write-backs: {len(run.get('memory_events', []))}",
        f"Model: {run.get('active_model', 'n/a')}",
        f"Session: {run.get('session_id', 'n/a')}",
    ]
    summary_ax.text(
        0.0,
        1.0,
        "\n".join(summary_lines),
        transform=summary_ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.3,
        color="#18212a",
        linespacing=1.55,
    )

    fig.patch.set_facecolor("#fff9f1")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg", facecolor=fig.get_facecolor())
    plt.close(fig)
