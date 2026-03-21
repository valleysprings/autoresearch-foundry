from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.codegen.config import load_runtime_config
from app.codegen.errors import ConfigError


class CodegenConfigTest(unittest.TestCase):
    def test_dotenv_loads_required_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {}, clear=True):
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "AUTORESEARCH_API_KEY=test-key",
                        "AUTORESEARCH_API_BASE=https://api.example.com/v1",
                        "AUTORESEARCH_PRIMARY_MODEL=deepseek-chat",
                        "AUTORESEARCH_TEMPERATURE=0.2",
                        "AUTORESEARCH_MAX_TOKENS=1400",
                        "AUTORESEARCH_TIMEOUT_S=45",
                    ]
                )
            )
            config = load_runtime_config(Path(tmp_dir))
            self.assertEqual(config.api_key, "test-key")
            self.assertEqual(config.primary_model, "deepseek-chat")
            self.assertEqual(config.max_tokens, 1400)

    def test_shell_env_overrides_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(
            os.environ,
            {"AUTORESEARCH_PRIMARY_MODEL": "shell-model"},
            clear=True,
        ):
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "AUTORESEARCH_API_KEY=test-key",
                        "AUTORESEARCH_API_BASE=https://api.example.com/v1",
                        "AUTORESEARCH_PRIMARY_MODEL=deepseek-chat",
                        "AUTORESEARCH_TEMPERATURE=0.2",
                        "AUTORESEARCH_MAX_TOKENS=1400",
                        "AUTORESEARCH_TIMEOUT_S=45",
                    ]
                )
            )
            config = load_runtime_config(Path(tmp_dir))
            self.assertEqual(config.primary_model, "shell-model")

    def test_missing_required_key_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {}, clear=True):
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "AUTORESEARCH_API_KEY=test-key",
                        "AUTORESEARCH_API_BASE=https://api.example.com/v1",
                        "AUTORESEARCH_PRIMARY_MODEL=deepseek-chat",
                        "AUTORESEARCH_TEMPERATURE=0.2",
                        "AUTORESEARCH_TIMEOUT_S=45",
                    ]
                )
            )
            with self.assertRaises(ConfigError):
                load_runtime_config(Path(tmp_dir))

    def test_invalid_url_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {}, clear=True):
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "AUTORESEARCH_API_KEY=test-key",
                        "AUTORESEARCH_API_BASE=not-a-url",
                        "AUTORESEARCH_PRIMARY_MODEL=deepseek-chat",
                        "AUTORESEARCH_TEMPERATURE=0.2",
                        "AUTORESEARCH_MAX_TOKENS=1400",
                        "AUTORESEARCH_TIMEOUT_S=45",
                    ]
                )
            )
            with self.assertRaises(ConfigError):
                load_runtime_config(Path(tmp_dir))


if __name__ == "__main__":
    unittest.main()
