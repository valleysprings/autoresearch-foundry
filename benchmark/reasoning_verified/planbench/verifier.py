from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from app.codegen.benchmark_support import public_question_payload
from app.codegen.verifier import load_callable_from_path


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_-]+")
NUMBERED_PREFIX_PATTERN = re.compile(r"^\s*\d+\.\s*")
PARENTHESIZED_ACTION_PATTERN = re.compile(r"^\(\s*([a-zA-Z0-9_-]+)(?:\s+([^()]+))?\s*\)$")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLANBENCH_ROOT = REPO_ROOT / "external/LLMs-Planning/plan-bench"
DEFAULT_VAL_BINARIES = (
    REPO_ROOT / "external/VAL/build/bin/Validate",
    REPO_ROOT / "external/VAL/build/bin/validate",
)

BLOCKSWORLD_ACTIONS = {
    "pick-up": "pick up the {}",
    "put-down": "put down the {}",
    "stack": "stack the {} on top of the {}",
    "unstack": "unstack the {} from on top of the {}",
}
BLOCKSWORLD_OBJECTS = {
    "a": "red block",
    "b": "blue block",
    "c": "orange block",
    "d": "yellow block",
    "e": "white block",
    "f": "magenta block",
    "g": "black block",
    "h": "cyan block",
    "i": "green block",
    "j": "violet block",
    "k": "silver block",
    "l": "gold block",
}
MYSTERY_BLOCKSWORLD_ACTIONS = {
    "attack": "attack {}",
    "succumb": "succumb {}",
    "overcome": "overcome {} from {}",
    "feast": "feast {} from {}",
}
MYSTERY_BLOCKSWORLD_OBJECTS = {
    "a": "object a",
    "b": "object b",
    "c": "object c",
    "d": "object d",
    "e": "object e",
    "f": "object f",
    "g": "object g",
    "h": "object h",
    "i": "object i",
    "j": "object j",
    "k": "object k",
    "l": "object l",
}

DOMAIN_SPECS: dict[str, dict[str, Any]] = {
    "blocksworld": {
        "domain_file": "blocksworld/generated_domain.pddl",
        "instance_dir": "blocksworld/generated_basic",
        "instances_template": "instance-{}.pddl",
        "actions": BLOCKSWORLD_ACTIONS,
        "encoded_objects": BLOCKSWORLD_OBJECTS,
    },
    "blocksworld_3": {
        "domain_file": "blocksworld/generated_domain.pddl",
        "instance_dir": "blocksworld/generated_basic_3",
        "instances_template": "instance-{}.pddl",
        "actions": BLOCKSWORLD_ACTIONS,
        "encoded_objects": BLOCKSWORLD_OBJECTS,
    },
    "mystery_blocksworld": {
        "domain_file": "blocksworld/mystery/generated_domain.pddl",
        "instance_dir": "blocksworld/mystery/generated_basic",
        "instances_template": "instance-{}.pddl",
        "actions": MYSTERY_BLOCKSWORLD_ACTIONS,
        "encoded_objects": MYSTERY_BLOCKSWORLD_OBJECTS,
    },
    "mystery_blocksworld_3": {
        "domain_file": "blocksworld/mystery/generated_domain.pddl",
        "instance_dir": "blocksworld/mystery/generated_basic_3",
        "instances_template": "instance-{}.pddl",
        "actions": MYSTERY_BLOCKSWORLD_ACTIONS,
        "encoded_objects": MYSTERY_BLOCKSWORLD_OBJECTS,
    },
    "logistics": {
        "domain_file": "logistics/generated_domain.pddl",
        "instance_dir": "logistics/generated_basic",
        "instances_template": "instance-{}.pddl",
        "actions": {
            "load-truck": "load {} into {} at {}",
            "load-airplane": "load {} into {} at {}",
            "unload-truck": "unload {} from {} at {}",
            "unload-airplane": "unload {} from {} at {}",
            "drive-truck": "drive {} from {} to {} in {}",
            "fly-airplane": "fly {} from {} to {}",
        },
    },
    "depots": {
        "domain_file": "depots/generated_domain.pddl",
        "instance_dir": "depots/generated_basic",
        "instances_template": "instance-{}.pddl",
        "actions": {
            "drive": "drive {} from {} to {}",
            "lift": "Use {} to lift {} from {} at {}",
            "drop": "Use {} to drop {} to {} at {}",
            "load": "Use {} to load {} into {} at {}",
            "unload": "Use {} to unload {} from {} at {}",
        },
    },
    "obfuscated_deceptive_logistics": {
        "domain_file": "obfuscated_deceptive_logistics/generated_domain.pddl",
        "instance_dir": "obfuscated_deceptive_logistics/generated_basic",
        "instances_template": "instance-{}.pddl",
        "actions": {
            "clip": "clip {} {} {}",
            "memory": "memory {} {} {}",
            "paltry": "paltry {} {} {}",
            "sip": "sip {} {} {}",
            "tightfisted": "tightfisted {} {} {}",
            "wretched": "wretched {} {} {} {}",
        },
    },
}


def _raw_context(item: dict[str, Any]) -> dict[str, Any]:
    raw_context = item.get("raw_context")
    if isinstance(raw_context, dict):
        return raw_context
    context = item.get("context")
    if isinstance(context, dict):
        return context
    return {}


def _domain_name(item: dict[str, Any]) -> str:
    context = _raw_context(item)
    domain = context.get("domain")
    if isinstance(domain, str) and domain.strip():
        return domain.strip().lower()
    metadata = dict(item.get("metadata") or {})
    fallback = metadata.get("domain")
    if isinstance(fallback, str) and fallback.strip():
        return fallback.strip().lower()
    raise ValueError("PlanBench question is missing context.domain.")


def _instance_id(item: dict[str, Any]) -> int:
    context = _raw_context(item)
    raw_instance_id = context.get("instance_id")
    if raw_instance_id is None:
        raise ValueError("PlanBench question is missing context.instance_id.")
    return int(raw_instance_id)


def _domain_spec(item: dict[str, Any]) -> dict[str, Any]:
    domain = _domain_name(item)
    spec = DOMAIN_SPECS.get(domain)
    if spec is None:
        raise ValueError(f"Unsupported PlanBench domain: {domain!r}")
    return spec


def _planbench_root() -> Path:
    explicit = os.getenv("PLANBENCH_OFFICIAL_ROOT")
    root = Path(explicit).expanduser() if explicit else DEFAULT_PLANBENCH_ROOT
    if (root / "instances").exists():
        return root
    nested = root / "plan-bench"
    if (nested / "instances").exists():
        return nested
    return root


def _val_binary() -> Path:
    explicit = os.getenv("PLANBENCH_VAL_BINARY")
    if explicit:
        return Path(explicit).expanduser()
    val_env = os.getenv("VAL")
    if val_env:
        candidate = Path(val_env).expanduser()
        if candidate.is_dir():
            for name in ("validate", "Validate"):
                binary = candidate / name
                if binary.exists():
                    return binary
        if candidate.exists():
            return candidate
    for binary in DEFAULT_VAL_BINARIES:
        if binary.exists():
            return binary
    return DEFAULT_VAL_BINARIES[0]


def _plan_text(value: object) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value or "")


def _has_digit(text: str) -> bool:
    return any(char.isdigit() for char in text)


def _count_action_args(template: str) -> int:
    return template.count("{}")


def _strip_numbering(line: str) -> str:
    return NUMBERED_PREFIX_PATTERN.sub("", line).strip()


def _is_symbolic_arg(token: str) -> bool:
    return "_" not in token and bool(re.fullmatch(r"[a-z][a-z0-9-]*", token))


def _direct_action_plan(text: str, *, actions: dict[str, str]) -> str:
    steps: list[str] = []
    for raw_line in text.splitlines():
        line = _strip_numbering(raw_line)
        if not line or "[cost]" in line.lower():
            continue
        match = PARENTHESIZED_ACTION_PATTERN.fullmatch(line)
        if match:
            action = match.group(1).lower()
            args = [token.lower() for token in TOKEN_PATTERN.findall(match.group(2) or "")]
        else:
            tokens = [token.lower() for token in TOKEN_PATTERN.findall(line)]
            if not tokens:
                continue
            action, *args = tokens
        if action not in actions:
            continue
        if len(args) != _count_action_args(actions[action]):
            continue
        if not all(_is_symbolic_arg(arg) for arg in args):
            continue
        steps.append(f"({action}{' ' if args else ''}{' '.join(args)})")
    return "\n".join(steps)


def _get_ordered_objects(object_names: list[str], line: str) -> list[str]:
    objects: list[tuple[int, str]] = []
    for name in object_names:
        if name in line:
            objects.append((line.index(name), name))
    objects.sort()
    return [name for _, name in objects]


def _extract_blocksworld_plan(text: str, *, actions: dict[str, str], encoded_objects: dict[str, str]) -> str:
    object_names = [name.lower() for name in encoded_objects.values()]
    reverse_objects = {value: key for key, value in encoded_objects.items()}
    action_aliases: dict[str, str] = {}
    for action, template in actions.items():
        first_word = template.split(" ", 1)[0]
        action_aliases[action] = action.replace("-", " ") if first_word in action else first_word

    normalized_text = text.lower().strip()
    for raw_action, text_action in action_aliases.items():
        normalized_text = normalized_text.replace(text_action, raw_action)

    plan_lines: list[str] = []
    for raw_line in normalized_text.splitlines():
        line = _strip_numbering(raw_line)
        if not line or "[cost]" in line.lower():
            continue
        tokens = line.split()
        matching_actions = [action for action in actions if action in tokens]
        if not matching_actions:
            continue
        action = matching_actions[0]
        ordered_objects = _get_ordered_objects(object_names, line)
        if len(ordered_objects) != _count_action_args(actions[action]):
            continue
        args = [reverse_objects[name] for name in ordered_objects]
        plan_lines.append(f"({action} {' '.join(args)})")
    return "\n".join(plan_lines)


def _logistics_symbol(token: str) -> str:
    parts = token.split("_")
    return token[0].lower() + "-".join(parts[1:])


def _extract_logistics_plan(text: str, *, actions: dict[str, str]) -> str:
    raw_actions = [name.split("-", 1)[0].lower() for name in actions]
    plan_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().lower()
        if not line:
            continue
        if "[cost]" in line:
            break
        line = _strip_numbering(line).replace(".", "")
        tokens = line.split()
        if not tokens or tokens[0] not in raw_actions:
            continue
        action = tokens[0]
        objects = [_logistics_symbol(token) for token in tokens if _has_digit(token)]
        if not objects:
            continue
        vehicle = objects[1] if action in {"load", "unload"} and len(objects) > 1 else objects[0]
        if vehicle.startswith("a"):
            action = f"{action}-airplane"
        elif vehicle.startswith("t"):
            action = f"{action}-truck"
        else:
            continue
        if action == "drive-truck" and len(objects) == 3:
            city_digits = "".join(char for char in objects[1] if char.isdigit())
            if city_digits:
                objects.append(f"c{city_digits[0]}")
        if len(objects) != _count_action_args(actions[action]):
            continue
        plan_lines.append(f"({action} {' '.join(objects)})")
    return "\n".join(plan_lines)


def _extract_depots_plan(text: str, *, actions: dict[str, str]) -> str:
    raw_actions = [name.lower() for name in actions]
    plan_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().lower()
        if not line:
            continue
        if "[cost]" in line:
            break
        line = line.lstrip("0123456789").replace(".", "").strip()
        objects = [token for token in line.split() if _has_digit(token)]
        matching_actions = [action for action in raw_actions if action in line]
        if not matching_actions:
            continue
        action = matching_actions[0]
        if len(objects) != _count_action_args(actions[action]):
            continue
        plan_lines.append(f"({action} {' '.join(objects)})")
    return "\n".join(plan_lines)


def _extract_obfuscated_plan(text: str, *, actions: dict[str, str]) -> str:
    raw_actions = list(actions.keys())
    plan_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().lower()
        if not line:
            continue
        if "[cost]" in line:
            break
        tokens = line.split()
        matching_actions = [action for action in raw_actions if action in tokens]
        object_ids = [token.strip() for token in line.split("object_") if token.strip().isdigit()]
        if not matching_actions or not object_ids:
            continue
        action = matching_actions[0]
        arity = _count_action_args(actions[action])
        objects = [f"o{value}" for value in object_ids]
        if len(objects) != arity:
            continue
        plan_lines.append(f"({action} {' '.join(objects)})")
    if plan_lines:
        return "\n".join(plan_lines)

    tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
    scanned_lines: list[str] = []
    index = 0
    while index < len(tokens):
        action = tokens[index]
        if action not in actions:
            index += 1
            continue
        arity = _count_action_args(actions[action])
        objects: list[str] = []
        cursor = index + 1
        while cursor < len(tokens) and len(objects) < arity:
            match = re.fullmatch(r"object_(\d+)", tokens[cursor])
            if match:
                objects.append(f"o{match.group(1)}")
            cursor += 1
        if len(objects) == arity:
            scanned_lines.append(f"({action} {' '.join(objects)})")
            index = cursor
            continue
        index += 1
    return "\n".join(scanned_lines)


def _extract_plan_text(raw_plan: object, item: dict[str, Any]) -> str:
    text = _plan_text(raw_plan)
    spec = _domain_spec(item)
    direct_plan = _direct_action_plan(text, actions=dict(spec["actions"]))
    if direct_plan:
        return direct_plan

    domain = _domain_name(item)
    actions = dict(spec["actions"])
    if domain == "logistics":
        return _extract_logistics_plan(text, actions=actions)
    if domain == "depots":
        return _extract_depots_plan(text, actions=actions)
    if domain == "obfuscated_deceptive_logistics":
        return _extract_obfuscated_plan(text, actions=actions)
    if "blocksworld" in domain:
        return _extract_blocksworld_plan(
            text,
            actions=actions,
            encoded_objects=dict(spec["encoded_objects"]),
        )
    raise ValueError(f"Unsupported PlanBench domain: {domain!r}")


def _plan_paths(item: dict[str, Any]) -> tuple[Path, Path]:
    root = _planbench_root()
    spec = _domain_spec(item)
    instance_id = _instance_id(item)
    domain_path = root / "instances" / str(spec["domain_file"])
    instance_path = root / "instances" / str(spec["instance_dir"]) / str(spec["instances_template"]).format(instance_id)
    return domain_path, instance_path


def _validate_plan(domain_path: Path, instance_path: Path, plan_text: str) -> tuple[bool, str]:
    binary = _val_binary()
    if not domain_path.exists():
        raise FileNotFoundError(f"Missing official PlanBench domain file: {domain_path}")
    if not instance_path.exists():
        raise FileNotFoundError(f"Missing official PlanBench instance file: {instance_path}")
    if not binary.exists():
        raise FileNotFoundError(f"Missing VAL validator binary: {binary}")

    with tempfile.NamedTemporaryFile("w", suffix=".plan", delete=False) as handle:
        handle.write(plan_text.rstrip() + "\n")
        plan_path = Path(handle.name)
    try:
        completed = subprocess.run(
            [str(binary), str(domain_path), str(instance_path), str(plan_path)],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        plan_path.unlink(missing_ok=True)

    output = "\n".join(part for part in (completed.stdout.strip(), completed.stderr.strip()) if part)
    return "plan valid" in output.lower(), output


def _plan_step_count(plan_text: str) -> int:
    return sum(1 for line in plan_text.splitlines() if line.strip())


def _display_plan(plan_text: str) -> str:
    rendered: list[str] = []
    for line in plan_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = PARENTHESIZED_ACTION_PATTERN.fullmatch(stripped)
        if match:
            action = match.group(1)
            args = " ".join(TOKEN_PATTERN.findall(match.group(2) or ""))
            rendered.append(f"{action}{' ' if args else ''}{args}")
        else:
            rendered.append(stripped)
    return "\n".join(rendered)


def evaluate_candidate(*, task, candidate_path, source_code, baseline_metrics, memory_applied):
    item = task.get("question_item")
    if not isinstance(item, dict):
        raise ValueError("PlanBench dataset task must provide question_item.")

    started = time.perf_counter()
    solver = load_callable_from_path(candidate_path, str(task["entry_symbol"]))
    raw_actual = solver(public_question_payload(item))
    extracted_plan = _extract_plan_text(raw_actual, item)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    if not extracted_plan.strip():
        row = {
            "name": item.get("name") or item["item_id"],
            "expected": item.get("expected_answer"),
            "actual": "",
            "actual_raw": raw_actual,
            "passed": False,
            "reason": "plan extraction failed",
        }
        return {
            "status": "fail",
            "verifier_status": "fail",
            "correctness": 0.0,
            "passed_tests": 0,
            "total_tests": 1,
            "benchmark_ms": round(elapsed_ms, 3),
            "benchmark_samples_ms": [round(elapsed_ms, 3)],
            "objective": 0.0,
            "objective_score": 0.0,
            "objective_signal": 0.0,
            "plan_steps": 0,
            "avg_plan_steps": 0.0,
            "error": None,
            "test_results": [row],
        }

    domain_path, instance_path = _plan_paths(item)
    passed, validator_output = _validate_plan(domain_path, instance_path, extracted_plan)
    step_count = _plan_step_count(extracted_plan)
    row = {
        "name": item.get("name") or item["item_id"],
        "expected": item.get("expected_answer"),
        "actual": _display_plan(extracted_plan),
        "actual_raw": raw_actual,
        "passed": passed,
        "validator_output": validator_output,
        "actual_plan_pddl": extracted_plan,
    }
    return {
        "status": "pass" if passed else "fail",
        "verifier_status": "pass" if passed else "fail",
        "correctness": 1.0 if passed else 0.0,
        "passed_tests": 1 if passed else 0,
        "total_tests": 1,
        "benchmark_ms": round(elapsed_ms, 3),
        "benchmark_samples_ms": [round(elapsed_ms, 3)],
        "objective": 1.0 if passed else 0.0,
        "objective_score": 1.0 if passed else 0.0,
        "objective_signal": 1.0 if passed else 0.0,
        "plan_steps": step_count,
        "avg_plan_steps": float(step_count),
        "error": None,
        "test_results": [row],
    }
