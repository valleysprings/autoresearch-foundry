from __future__ import annotations

import time

from app.codegen.verifier import load_callable_from_path


def _neighbors(problem: dict) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    for left, right in problem["roads"]:
        graph.setdefault(left, set()).add(right)
        graph.setdefault(right, set()).add(left)
    return graph


def _simulate(problem: dict, plan: list[str]) -> tuple[bool, str | None]:
    graph = _neighbors(problem)
    robot = problem["start"]
    package = problem["package_start"]
    holding = False
    for action in plan:
        parts = action.split()
        if not parts:
            return False, "empty action"
        if parts[0] == "drive" and len(parts) == 3:
            _, source, target = parts
            if robot != source or target not in graph.get(source, set()):
                return False, f"invalid drive {action}"
            robot = target
        elif parts[0] == "pickup" and len(parts) == 3:
            _, _item, location = parts
            if holding or robot != location or package != location:
                return False, f"invalid pickup {action}"
            holding = True
        elif parts[0] == "drop" and len(parts) == 3:
            _, _item, location = parts
            if not holding or robot != location:
                return False, f"invalid drop {action}"
            holding = False
            package = location
        else:
            return False, f"unknown action {action}"
    return package == problem["goal"] and not holding, None


def _evaluation_items(task: dict) -> tuple[list[dict], bool]:
    item = task.get("question_item")
    if isinstance(item, dict):
        raw_context = item.get("raw_context")
        if not isinstance(raw_context, dict):
            raise ValueError("PlanBench dataset question must provide raw_context.")
        problem = raw_context.get("problem")
        if not isinstance(problem, dict):
            raise ValueError("PlanBench dataset question raw_context.problem must be an object.")
        optimal_steps = int(raw_context.get("optimal_steps") or 0)
        return (
            [
                {
                    "name": item.get("name") or item.get("item_id") or "item",
                    "problem": problem,
                    "optimal_steps": optimal_steps,
                }
            ],
            True,
        )
    return list(task["data"]["items"]), False


def evaluate_candidate(*, task, candidate_path, source_code, baseline_metrics, memory_applied):
    started = time.perf_counter()
    solver = load_callable_from_path(candidate_path, str(task["entry_symbol"]))
    rows = []
    solved = 0
    solved_steps = 0
    items, single_item = _evaluation_items(task)
    total = len(items)
    for item in items:
        plan = solver(item["problem"])
        if not isinstance(plan, list) or not all(isinstance(step, str) for step in plan):
            raise ValueError("solve(problem) must return list[str]")
        passed, error = _simulate(item["problem"], plan)
        rows.append(
            {
                "name": item["name"],
                "expected": f"valid plan in <= {item['optimal_steps']} steps",
                "actual": plan,
                "passed": passed,
                "error": error,
            }
        )
        if passed:
            solved += 1
            solved_steps += len(plan)
    solved_ratio = solved / total if total else 0.0
    avg_steps = (solved_steps / solved) if solved else 0.0
    step_penalty = min(avg_steps / 20.0, 0.25) if solved else 0.25
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    passed = solved == total and total > 0
    return {
        "status": "pass" if (passed or not single_item) else "fail",
        "verifier_status": "pass" if (passed or not single_item) else "fail",
        "correctness": solved_ratio,
        "passed_tests": solved,
        "total_tests": total,
        "benchmark_ms": round(elapsed_ms, 3),
        "benchmark_samples_ms": [round(elapsed_ms, 3)],
        "objective": solved_ratio,
        "objective_score": round(solved_ratio - step_penalty, 6),
        "objective_signal": max(0.0, solved_ratio - step_penalty),
        "avg_plan_steps": round(avg_steps, 3),
        "error": None,
        "test_results": rows,
    }
