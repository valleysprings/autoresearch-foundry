from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlparse

from app.configs.runtime import (
    AVAILABLE_MODELS_ENV_KEY,
    DEFAULT_AVAILABLE_MODELS,
    DEFAULT_LLM_CONCURRENCY,
    DEFAULT_PRIMARY_MODEL,
    DEFAULT_RUNTIME_MAX_TOKENS,
    DEFAULT_RUNTIME_TEMPERATURE,
    DEFAULT_RUNTIME_TIMEOUT_S,
    LLM_CONCURRENCY_ENV_KEY,
    PRIMARY_MODEL_ENV_KEY,
    REQUIRED_ENV_KEYS,
    ROOT,
)
from app.codegen.errors import ConfigError


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


def _parse_float(name: str, *, default: float | None = None) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        if default is None:
            raise ConfigError(f"Missing required environment variable {name}.")
        return default
    value = value.strip()
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name} must be a float.") from exc
    if not 0.0 <= parsed <= 2.0:
        raise ConfigError(f"Environment variable {name} must be between 0.0 and 2.0.")
    return parsed


def _parse_int(name: str, *, default: int | None = None) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        if default is None:
            raise ConfigError(f"Missing required environment variable {name}.")
        return default
    value = value.strip()
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name} must be an integer.") from exc
    if parsed <= 0:
        raise ConfigError(f"Environment variable {name} must be greater than zero.")
    return parsed


def _parse_optional_positive_int(name: str, *, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
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


def _parse_available_models(primary_model: str) -> tuple[str, ...]:
    raw_value = os.getenv(AVAILABLE_MODELS_ENV_KEY, "")
    models: list[str] = []
    source = raw_value.split(",") if raw_value.strip() else list(DEFAULT_AVAILABLE_MODELS)
    for item in source:
        model = item.strip()
        if model and model not in models:
            models.append(model)
    if primary_model not in models:
        models.insert(0, primary_model)
    return tuple(models)


def _primary_model() -> str:
    value = os.getenv(PRIMARY_MODEL_ENV_KEY)
    if value is None or not value.strip():
        return DEFAULT_PRIMARY_MODEL
    return value.strip()


@dataclass(slots=True)
class RuntimeConfig:
    api_key: str
    api_base: str
    primary_model: str
    available_models: tuple[str, ...]
    temperature: float
    max_tokens: int
    timeout_s: int
    llm_concurrency: int
    selected_model: str | None = None

    @property
    def active_model(self) -> str:
        return self.selected_model or self.primary_model

    def with_model(self, model: str | None) -> "RuntimeConfig":
        if model is None or model == "":
            return replace(self, selected_model=None)
        if model not in self.available_models:
            raise ConfigError(
                f"Model {model} is not enabled. Choose one of: {', '.join(self.available_models)}."
            )
        return replace(self, selected_model=model)

    def describe(self) -> dict[str, object]:
        return {
            "mode": "llm-required",
            "primary_model": self.primary_model,
            "active_model": self.active_model,
            "available_models": list(self.available_models),
            "api_base": self.api_base,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_s": self.timeout_s,
            "llm_concurrency": self.llm_concurrency,
        }


def load_runtime_config(root: Path | None = None) -> RuntimeConfig:
    load_repo_env(root or ROOT)
    primary_model = _primary_model()
    return RuntimeConfig(
        api_key=_require_non_empty("AUTORESEARCH_API_KEY"),
        api_base=_parse_api_base("AUTORESEARCH_API_BASE"),
        primary_model=primary_model,
        available_models=_parse_available_models(primary_model),
        temperature=_parse_float("AUTORESEARCH_TEMPERATURE", default=DEFAULT_RUNTIME_TEMPERATURE),
        max_tokens=_parse_int("AUTORESEARCH_MAX_TOKENS", default=DEFAULT_RUNTIME_MAX_TOKENS),
        timeout_s=_parse_int("AUTORESEARCH_TIMEOUT_S", default=DEFAULT_RUNTIME_TIMEOUT_S),
        llm_concurrency=_parse_optional_positive_int(LLM_CONCURRENCY_ENV_KEY, default=DEFAULT_LLM_CONCURRENCY),
    )
