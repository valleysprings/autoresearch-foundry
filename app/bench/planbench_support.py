from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.bench.benchmark_support import answer_aliases, canonical_text


REPO_ROOT = Path(__file__).resolve().parents[2]
PLANBENCH_SHARED_ROOT = REPO_ROOT / "benchmark" / "reasoning_verified" / "planbench-shared"
TASK_LOCAL_ASSETS_ROOT = PLANBENCH_SHARED_ROOT / "official"
TASK_LOCAL_OFFICIAL_ROOT = TASK_LOCAL_ASSETS_ROOT / "plan-bench"
TASK_LOCAL_VAL_ROOT = TASK_LOCAL_ASSETS_ROOT / "VAL"
DEFAULT_VAL_BINARIES = (
    TASK_LOCAL_VAL_ROOT / "build" / "bin" / "validate",
    TASK_LOCAL_VAL_ROOT / "build" / "bin" / "Validate",
)


class PlanExtractionError(ValueError):
    pass


@dataclass(frozen=True)
class PlanBenchConfig:
    domain_name: str
    domain_file: str
    instance_dir: str
    instances_template: str
    actions: dict[str, str]
    encoded_objects: dict[str, str]


def _raw_context(item: dict[str, Any]) -> dict[str, Any]:
    raw_context = item.get("raw_context")
    if isinstance(raw_context, dict):
        return raw_context
    context = item.get("context")
    if isinstance(context, dict):
        return context
    return {}


def domain_name(item: dict[str, Any]) -> str:
    context = _raw_context(item)
    domain = context.get("domain")
    if isinstance(domain, str) and domain.strip():
        return domain.strip().lower()
    metadata = dict(item.get("metadata") or {})
    fallback = metadata.get("domain")
    if isinstance(fallback, str) and fallback.strip():
        return fallback.strip().lower()
    raise ValueError("PlanBench question is missing context.domain.")


def instance_id(item: dict[str, Any]) -> int:
    context = _raw_context(item)
    raw_instance_id = context.get("instance_id")
    if raw_instance_id is None:
        raise ValueError("PlanBench question is missing context.instance_id.")
    return int(raw_instance_id)


def _resolve_nested_official_root(root: Path) -> Path:
    for candidate in (root, root / "plan-bench"):
        if (candidate / "configs").exists() or (candidate / "instances").exists():
            return candidate
    return root


def resolve_official_root() -> Path:
    explicit = os.getenv("PLANBENCH_OFFICIAL_ROOT")
    if explicit:
        return _resolve_nested_official_root(Path(explicit).expanduser())

    for candidate in (TASK_LOCAL_OFFICIAL_ROOT, TASK_LOCAL_ASSETS_ROOT):
        resolved = _resolve_nested_official_root(candidate)
        if (resolved / "configs").exists() or (resolved / "instances").exists():
            return resolved
    return TASK_LOCAL_OFFICIAL_ROOT


def resolve_val_binary() -> Path:
    explicit = os.getenv("PLANBENCH_VAL_BINARY")
    if explicit:
        return Path(explicit).expanduser()

    val_env = os.getenv("VAL")
    if val_env:
        candidate = Path(val_env).expanduser()
        if candidate.is_file():
            return candidate
        if candidate.is_dir():
            for relative in ("validate", "Validate", "build/bin/validate", "build/bin/Validate"):
                binary = candidate / relative
                if binary.exists():
                    return binary
            return candidate / "validate"

    for binary in DEFAULT_VAL_BINARIES:
        if binary.exists():
            return binary
    return DEFAULT_VAL_BINARIES[0]


def _config_path(item: dict[str, Any]) -> Path:
    return resolve_official_root() / "configs" / f"{domain_name(item)}.yaml"


def load_config(item: dict[str, Any]) -> PlanBenchConfig:
    config_path = _config_path(item)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing PlanBench official config file: {config_path}")
    payload = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid PlanBench config payload: {config_path}")

    return PlanBenchConfig(
        domain_name=str(payload.get("domain_name") or domain_name(item)).strip().lower(),
        domain_file=str(payload["domain_file"]),
        instance_dir=str(payload["instance_dir"]),
        instances_template=str(payload["instances_template"]),
        actions={str(key): str(value) for key, value in dict(payload.get("actions") or {}).items()},
        encoded_objects={str(key): str(value) for key, value in dict(payload.get("encoded_objects") or {}).items()},
    )


def domain_and_instance_paths(item: dict[str, Any]) -> tuple[Path, Path]:
    config = load_config(item)
    root = resolve_official_root()
    domain_path = root / "instances" / config.domain_file
    instance_path = root / "instances" / config.instance_dir / config.instances_template.format(instance_id(item))
    return domain_path, instance_path


def _plan_text(raw_plan: object) -> str:
    if isinstance(raw_plan, list):
        return "\n".join(str(item) for item in raw_plan)
    return str(raw_plan or "")


def _contains_cost_marker(line: str) -> bool:
    return "[cost]" in line.lower()


def _has_digit(value: str) -> bool:
    return any(char.isdigit() for char in value)


def _ordered_objects(object_names: list[str], line: str) -> list[str]:
    objects: list[tuple[int, str]] = []
    for name in object_names:
        if name in line:
            objects.append((line.index(name), name))
    objects.sort()
    return [name for _, name in objects]


def _format_plan_line(action: str, args: list[str]) -> str:
    return f"({action}{' ' if args else ''}{' '.join(args)})"


def _logistics_symbol(token: str) -> str:
    pieces = token.split("_")
    return token[0] + "-".join(pieces[1:])


def _extract_logistics_plan(text: str, config: PlanBenchConfig) -> str:
    raw_actions = [name.split("-", 1)[0].lower() for name in config.actions]
    plan_lines: list[str] = []
    for line in [value.strip().lower() for value in text.split("\n")]:
        if not line:
            continue
        if _contains_cost_marker(line):
            break

        if len(line) >= 2 and line[0].isdigit() and line[1] == ".":
            line = line[2:]
            line = line.replace(".", "")
        elif len(line) >= 3 and line[0].isdigit() and line[1].isdigit() and line[2] == ".":
            line = line[3:]
            line = line.replace(".", "")

        tokens = line.split()
        if not tokens or tokens[0] not in raw_actions:
            continue

        objects = [_logistics_symbol(token) for token in tokens if _has_digit(token)]
        if "load" in tokens[0] or "unload" in tokens[0]:
            if len(objects) < 2:
                raise PlanExtractionError(f"Malformed logistics step: {line}")
            to_check = objects[1]
        else:
            if not objects:
                raise PlanExtractionError(f"Malformed logistics step: {line}")
            to_check = objects[0]

        action = tokens[0]
        if "a" in to_check:
            action += "-airplane"
        elif "t" in to_check:
            action += "-truck"
        else:
            raise PlanExtractionError(f"Unable to infer logistics vehicle type: {line}")

        if action == "drive-truck" and len(objects) == 3:
            city_digits = [char for char in objects[1] if char.isdigit()]
            if city_digits:
                objects.append(f"c{city_digits[0]}")

        plan_lines.append(_format_plan_line(action, objects))
    return "\n".join(plan_lines)


def _extract_depots_plan(text: str, config: PlanBenchConfig) -> str:
    raw_actions = [name.lower() for name in config.actions]
    plan_lines: list[str] = []
    for line in [value.strip().lower() for value in text.split("\n")]:
        if not line:
            continue
        if _contains_cost_marker(line):
            break

        line = line.lstrip("0123456789").replace(".", "")
        objects = [token for token in line.split() if _has_digit(token)]

        action = next((name for name in raw_actions if name in line), None)
        if action is None:
            continue

        plan_lines.append(_format_plan_line(action, objects))
    return "\n".join(plan_lines)


def _extract_obfuscated_plan(text: str, config: PlanBenchConfig) -> str:
    raw_actions = list(config.actions.keys())
    object_prefix = next(iter(config.encoded_objects.values()), "object_{}").split("{", 1)[0]
    plan_lines: list[str] = []
    for line in [value.strip() for value in text.split("\n")]:
        if not line:
            continue
        if _contains_cost_marker(line):
            break

        action = next((name for name in raw_actions if name in line.split()), None)
        if action is None:
            continue

        action_arity = config.actions[action].count("{}")
        object_ids = [token.strip() for token in line.split(object_prefix) if token.strip().isdigit()]
        if len(object_ids) != action_arity:
            continue

        plan_lines.append(_format_plan_line(action, [f"o{value}" for value in object_ids]))
    return "\n".join(plan_lines)


def _extract_blocksworld_plan(text: str, config: PlanBenchConfig) -> str:
    encoded_objects = dict(config.encoded_objects)
    reverse_objects = {value.lower(): key for key, value in encoded_objects.items()}
    action_aliases: dict[str, str] = {}
    for action, template in config.actions.items():
        first_word = template.split(" ", 1)[0]
        action_aliases[action] = action.replace("-", " ") if first_word in action else first_word

    normalized_text = text.lower().strip()
    for raw_action, text_action in action_aliases.items():
        normalized_text = normalized_text.replace(text_action, raw_action)

    object_names = [value.lower() for value in encoded_objects.values()]
    plan_lines: list[str] = []
    for line in [value.strip() for value in normalized_text.split("\n")]:
        if not line:
            continue
        if _contains_cost_marker(line):
            break

        action = next((name for name in config.actions if name in line.split()), None)
        if action is None:
            continue

        expected_args = config.actions[action].count("{}")
        objects = _ordered_objects(object_names, line)
        if len(objects) != expected_args:
            continue

        plan_lines.append(_format_plan_line(action, [reverse_objects[name] for name in objects]))
    return "\n".join(plan_lines)


def extract_plan(raw_plan: object, item: dict[str, Any]) -> str:
    config = load_config(item)
    text = _plan_text(raw_plan)
    if "obfuscated" in config.domain_name:
        return _extract_obfuscated_plan(text, config)
    if config.domain_name == "logistics":
        return _extract_logistics_plan(text, config)
    if "blocksworld" in config.domain_name:
        return _extract_blocksworld_plan(text, config)
    if "depots" in config.domain_name:
        return _extract_depots_plan(text, config)
    raise ValueError(f"Unsupported PlanBench domain: {config.domain_name!r}")


def validate_plan(item: dict[str, Any], plan_text: str) -> tuple[bool, str]:
    domain_path, instance_path = domain_and_instance_paths(item)
    binary = resolve_val_binary()

    if not domain_path.exists():
        raise FileNotFoundError(f"Missing PlanBench domain file: {domain_path}")
    if not instance_path.exists():
        raise FileNotFoundError(f"Missing PlanBench instance file: {instance_path}")
    if not binary.exists():
        raise FileNotFoundError(f"Missing PlanBench VAL validator binary: {binary}")

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


def plan_step_count(plan_text: str) -> int:
    return sum(1 for line in plan_text.splitlines() if line.strip())


def display_plan(plan_text: str) -> str:
    rendered: list[str] = []
    for line in plan_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("(") and stripped.endswith(")"):
            rendered.append(stripped[1:-1].strip())
        elif stripped:
            rendered.append(stripped)
    return "\n".join(rendered)


PLAN_BLOCK_PATTERN = re.compile(r"\[PLAN\](.*?)\[PLAN END\]", re.IGNORECASE | re.DOTALL)


def extract_final_query_plan(query: object) -> str:
    matches = [match.strip() for match in PLAN_BLOCK_PATTERN.findall(str(query or "")) if match.strip()]
    if not matches:
        raise PlanExtractionError("Unable to locate a final [PLAN] ... [PLAN END] block in the PlanBench prompt.")
    return matches[-1]


def derive_verification_verdict_from_query(query: object, item: dict[str, Any]) -> tuple[str, str]:
    extracted_plan = extract_plan(extract_final_query_plan(query), item)
    if not extracted_plan.strip():
        raise PlanExtractionError("Unable to extract a canonical plan from the final PlanBench prompt block.")
    is_valid, validator_output = validate_plan(item, extracted_plan)
    normalized_output = canonical_text(validator_output, lowercase=True)
    if "plan valid" in normalized_output:
        detail = "Plan valid"
    elif "plan invalid" in normalized_output:
        detail = "Plan invalid"
    else:
        detail = "Semantic validation completed"
    return ("yes" if is_valid else "no"), detail


def normalize_verification_verdict(value: object) -> str:
    normalized = canonical_text(value, lowercase=True)
    if "plan is invalid" in normalized or normalized.startswith("the above plan is invalid"):
        return "no"
    if "plan is valid" in normalized or normalized.startswith("the above plan is valid"):
        return "yes"
    raise ValueError(f"Unable to infer PlanBench verification verdict from: {value!r}")


def verification_answer_aliases(verdict: str) -> list[str]:
    normalized = canonical_text(verdict, lowercase=True)
    if normalized == "yes":
        aliases = answer_aliases("yes", "valid", "plan valid", "the above plan is valid", "the plan is valid")
    elif normalized == "no":
        aliases = answer_aliases("no", "invalid", "plan invalid", "the above plan is invalid", "the plan is invalid")
    else:
        raise ValueError(f"Unsupported verification verdict: {verdict!r}")
    return sorted(aliases)
