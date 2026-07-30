"""Validation for the model-selected portfolio gross cap."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

MAX_ERROR_TEXT = 500


@dataclass(frozen=True, slots=True)
class StrategyError:
    error_code: str
    message: str
    details: dict[str, Any] | None = None
    contract_failure: bool = False

    def to_payload(self, **extra: Any) -> dict[str, Any]:
        payload = {
            "ok": False,
            "error_code": self.error_code,
            "message": self.message[:MAX_ERROR_TEXT],
            "recoverable": True,
        }
        if self.details:
            payload["details"] = self.details
        payload.update(extra)
        return payload


@dataclass(frozen=True, slots=True)
class CompiledStrategy:
    model_id: str
    max_gross_exposure: float
    settings_hash: str

    @property
    def settings(self) -> dict[str, float]:
        return {"max_gross_exposure": self.max_gross_exposure}


def compile_model_strategy(
    *,
    model_id: str,
    max_gross_exposure: Any,
) -> CompiledStrategy | StrategyError:
    if isinstance(max_gross_exposure, bool):
        return StrategyError(
            "invalid_max_gross_exposure",
            "max_gross_exposure must be a finite number.",
        )
    try:
        requested = float(max_gross_exposure)
    except (TypeError, ValueError):
        return StrategyError(
            "invalid_max_gross_exposure",
            "max_gross_exposure must be a finite number.",
        )
    if not math.isfinite(requested):
        return StrategyError(
            "invalid_max_gross_exposure",
            "max_gross_exposure must be a finite number.",
        )
    if requested < 0.0:
        return StrategyError(
            "invalid_max_gross_exposure",
            "max_gross_exposure must be non-negative.",
        )

    payload = {"max_gross_exposure": requested}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return CompiledStrategy(
        model_id=model_id,
        max_gross_exposure=requested,
        settings_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


__all__ = ["CompiledStrategy", "StrategyError", "compile_model_strategy"]
