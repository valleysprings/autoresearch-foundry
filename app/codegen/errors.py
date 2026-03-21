from __future__ import annotations


class AutoresearchError(RuntimeError):
    error_type = "runtime_error"
    terminal = True

    def __init__(self, message: str, *, model: str | None = None):
        super().__init__(message)
        self.model = model

    def as_payload(self) -> dict[str, object]:
        return {
            "terminal": True,
            "error_type": self.error_type,
            "error": str(self),
            "model": self.model,
        }


class ConfigError(AutoresearchError):
    error_type = "config_error"


class LlmTransportError(AutoresearchError):
    error_type = "llm_transport_error"


class LlmResponseError(AutoresearchError):
    error_type = "llm_response_error"


class VerificationError(AutoresearchError):
    error_type = "verification_error"
