"""Runtime request records and structured errors."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class ProtocolError:
    error_code: str
    message: str
    recoverable: bool = True
    details: dict[str, Any] | None = None
    suggested_correction: str | None = None

    def to_payload(self, **extra: Any) -> dict[str, Any]:
        payload = {
            "ok": False,
            "error_code": self.error_code,
            "message": self.message,
            "recoverable": self.recoverable,
        }
        if self.details:
            payload["details"] = self.details
        if self.suggested_correction:
            payload["suggested_correction"] = self.suggested_correction
        payload.update(extra)
        return payload


@dataclass(frozen=True, slots=True)
class DatetimeFilter:
    start_datetime: datetime | None = None
    end_datetime: datetime | None = None

    def to_payload(self) -> dict[str, str]:
        return {
            key: value.isoformat().replace("+00:00", "Z")
            for key, value in {
                "start_datetime": self.start_datetime,
                "end_datetime": self.end_datetime,
            }.items()
            if value is not None
        }


@dataclass(frozen=True, slots=True)
class BacktestRequest:
    model_id: str
    max_gross_exposure: float
    label: str | None = None
    backtest_filter: DatetimeFilter | None = None


@dataclass(frozen=True, slots=True)
class TrainModelRequest:
    model_code: str
    label: str | None = None
    train_filter: DatetimeFilter | None = None


@dataclass(frozen=True, slots=True)
class SubmitStrategyRequest:
    strategy_id: str
    rationale: str | None = None
