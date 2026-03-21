from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from app.codegen.errors import ConfigError


ROOT = Path(__file__).resolve().parents[2]
REQUIRED_ENV_KEYS = (
    "AUTORESEARCH_API_KEY",
    "AUTORESEARCH_API_BASE",
    "AUTORESEARCH_PRIMARY_MODEL",
    "AUTORESEARCH_TEMPERATURE",
    "AUTORESEARCH_MAX_TOKENS",
    "AUTORESEARCH_TIMEOUT_S",
)


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ConfigError(f"Malformed .env line {line_number}: expected KEY=VALUE.")
        key, value = line.split("=", 1)
        normalized_key = key.strip()
        if not normalized_key:
            raise ConfigError(f"Malformed .env line {line_number}: empty key.")
        values[normalized_key] = _strip_quotes(value.strip())
    return values


def load_repo_env(root: Path | None = None) -> Path:
    resolved_root = root or ROOT
    env_path = resolved_root / ".env"
    for key, value in parse_dotenv(env_path).items():
        os.environ.setdefault(key, value)
    return env_path


def _require_non_empty(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise ConfigError(f"Missing required environment variable {name}.")
    return value.strip()


def _parse_float(name: str) -> float:
    value = _require_non_empty(name)
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name} must be a float.") from exc
    if not 0.0 <= parsed <= 2.0:
        raise ConfigError(f"Environment variable {name} must be between 0.0 and 2.0.")
    return parsed


def _parse_int(name: str) -> int:
    value = _require_non_empty(name)
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name} must be an integer.") from exc
    if parsed <= 0:
        raise ConfigError(f"Environment variable {name} must be greater than zero.")
    return parsed


def _parse_api_base(name: str) -> str:
    value = _require_non_empty(name)
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigError(f"Environment variable {name} must be a valid http(s) URL.")
    return value.rstrip("/")


@dataclass(slots=True)
class RuntimeConfig:
    api_key: str
    api_base: str
    primary_model: str
    temperature: float
    max_tokens: int
    timeout_s: int

    def describe(self) -> dict[str, object]:
        return {
            "mode": "llm-required",
            "active_model": self.primary_model,
            "api_base": self.api_base,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_s": self.timeout_s,
        }


def load_runtime_config(root: Path | None = None) -> RuntimeConfig:
    load_repo_env(root or ROOT)
    return RuntimeConfig(
        api_key=_require_non_empty("AUTORESEARCH_API_KEY"),
        api_base=_parse_api_base("AUTORESEARCH_API_BASE"),
        primary_model=_require_non_empty("AUTORESEARCH_PRIMARY_MODEL"),
        temperature=_parse_float("AUTORESEARCH_TEMPERATURE"),
        max_tokens=_parse_int("AUTORESEARCH_MAX_TOKENS"),
        timeout_s=_parse_int("AUTORESEARCH_TIMEOUT_S"),
    )
