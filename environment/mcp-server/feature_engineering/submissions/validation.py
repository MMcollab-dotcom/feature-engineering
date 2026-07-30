"""Source validation for submitted feature-engineering model code."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

SUBMITTED_MODULE_PREFIX = "submitted_feature_model_"


@dataclass(frozen=True, slots=True)
class CodeValidationError:
    error_code: str
    message: str
    details: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "error_code": self.error_code,
            "message": self.message[:500],
            "recoverable": True,
        }
        if self.details:
            payload["details"] = dict(self.details)
        return payload


def validate_model_code(
    code: str,
    *,
    max_code_bytes: int,
) -> CodeValidationError | None:
    if not isinstance(code, str) or not code.strip():
        return CodeValidationError(
            "missing_model_code",
            "train_model requires a non-empty model_code string.",
        )
    if len(code.encode("utf-8")) > int(max_code_bytes):
        return CodeValidationError(
            "model_code_too_large",
            "Submitted model_code exceeds the task byte limit.",
            details={"max_model_code_bytes": int(max_code_bytes)},
        )
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return CodeValidationError(
            "invalid_model_code_syntax",
            exc.msg,
            details={"line": exc.lineno or 0, "offset": exc.offset or 0},
        )

    if not any(
        isinstance(node, ast.FunctionDef) and node.name == "train_model"
        for node in tree.body
    ):
        return CodeValidationError(
            "missing_model_code_function",
            "Submitted model_code must define train_model(X, y).",
            details={"missing": ["train_model"]},
        )
    return None


def submitted_module_name(source_hash: str) -> str:
    """Return the stable module identity for one exact submitted source blob."""

    return f"{SUBMITTED_MODULE_PREFIX}{source_hash}"


def redact_submitted_identity(message: str, *, source_hash: str) -> str:
    """Remove only the exact synthetic identity bound to this worker request."""

    return message.replace(
        submitted_module_name(source_hash),
        "submitted_feature_model",
    ).replace(source_hash, "<source-hash>")


__all__ = [
    "CodeValidationError",
    "redact_submitted_identity",
    "submitted_module_name",
    "validate_model_code",
]
