"""Strict assistant JSON protocol parsing and structured errors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
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
class GetTrainModelResultRequest:
    training_id: str


@dataclass(frozen=True, slots=True)
class GetBacktestResultRequest:
    backtest_id: str


@dataclass(frozen=True, slots=True)
class SubmitStrategyRequest:
    strategy_id: str
    rationale: str | None = None


def parse_assistant_response(
    message: str,
) -> (
    BacktestRequest
    | TrainModelRequest
    | GetTrainModelResultRequest
    | GetBacktestResultRequest
    | SubmitStrategyRequest
    | ProtocolError
):
    try:
        payload = json.loads(message)
    except json.JSONDecodeError as exc:
        return ProtocolError(
            error_code="malformed_json",
            message="Assistant response must be strict JSON.",
            details={"position": exc.pos, "reason": exc.msg},
            suggested_correction="Return a supported feature-engineering tool payload.",
        )
    if not isinstance(payload, dict):
        return ProtocolError(
            error_code="invalid_payload",
            message="Assistant response JSON must be an object.",
        )
    action = payload.get("action")
    if action == "train_model":
        return _parse_train_model(payload)
    if action == "get_train_model_result":
        return _parse_result_query(payload, "training_id", GetTrainModelResultRequest)
    if action == "backtest":
        return _parse_backtest(payload)
    if action == "get_backtest_result":
        return _parse_result_query(payload, "backtest_id", GetBacktestResultRequest)
    if action == "submit_strategy":
        return _parse_submit_strategy(payload)
    return ProtocolError(
        error_code="unknown_action",
        message=(
            "Unknown action. Expected 'train_model', 'get_train_model_result', "
            "'backtest', 'get_backtest_result', or 'submit_strategy'."
        ),
        details={"action": action},
    )


def _parse_train_model(payload: dict[str, Any]) -> TrainModelRequest | ProtocolError:
    model_code = payload.get("model_code")
    if not isinstance(model_code, str) or not model_code.strip():
        return ProtocolError(
            error_code="missing_model_code",
            message="train_model action requires a model_code string.",
            suggested_correction=(
                "Submit Python code whose harness entry point is train_model(X, y), "
                "returning a fitted source-bound sklearn estimator with "
                "feature_names_in_. Module-scope implementation helpers are allowed."
            ),
        )
    train_filter = _parse_datetime_filter(payload.get("train_filter"), "train_filter")
    if isinstance(train_filter, ProtocolError):
        return train_filter
    label = payload.get("label")
    return TrainModelRequest(
        model_code=model_code,
        label=label if isinstance(label, str) else None,
        train_filter=train_filter,
    )


def _parse_backtest(payload: dict[str, Any]) -> BacktestRequest | ProtocolError:
    unknown = sorted(
        set(payload)
        - {
            "action",
            "model_id",
            "max_gross_exposure",
            "label",
            "backtest_filter",
        }
    )
    if unknown:
        return ProtocolError(
            error_code="invalid_backtest_fields",
            message="backtest received unsupported fields.",
            details={"unknown_fields": unknown},
        )
    model_id = payload.get("model_id")
    if model_id is None:
        return ProtocolError(
            error_code="missing_model_id",
            message=(
                "Backtest action requires a model_id from a successful "
                "train_model action."
            ),
            suggested_correction=(
                "Call train_model, query get_train_model_result, then call backtest "
                "with the returned model_id."
            ),
        )
    if not isinstance(model_id, str) or not model_id:
        return ProtocolError(
            error_code="invalid_model_id",
            message="Backtest model_id must be a non-empty string when provided.",
        )
    max_gross_exposure = payload.get("max_gross_exposure")
    if isinstance(max_gross_exposure, bool) or not isinstance(
        max_gross_exposure, (int, float)
    ):
        return ProtocolError(
            error_code="invalid_max_gross_exposure",
            message="Backtest max_gross_exposure must be a finite number.",
        )
    backtest_filter = _parse_datetime_filter(
        payload.get("backtest_filter"),
        "backtest_filter",
    )
    if isinstance(backtest_filter, ProtocolError):
        return backtest_filter
    label = payload.get("label")
    return BacktestRequest(
        model_id=model_id,
        max_gross_exposure=float(max_gross_exposure),
        label=label if isinstance(label, str) else None,
        backtest_filter=backtest_filter,
    )


def _parse_result_query(
    payload: dict[str, Any],
    id_field: str,
    request_type: type[GetTrainModelResultRequest] | type[GetBacktestResultRequest],
) -> GetTrainModelResultRequest | GetBacktestResultRequest | ProtocolError:
    unknown = sorted(set(payload) - {"action", id_field})
    operation_id = payload.get(id_field)
    if unknown or not isinstance(operation_id, str) or not operation_id:
        return ProtocolError(
            error_code=f"invalid_{id_field}",
            message=f"{payload.get('action')} requires only a non-empty {id_field}.",
            details={"unknown_fields": unknown} if unknown else None,
        )
    return request_type(operation_id)


def _parse_submit_strategy(
    payload: dict[str, Any],
) -> SubmitStrategyRequest | ProtocolError:
    strategy_id = payload.get("strategy_id")
    if not isinstance(strategy_id, str) or not strategy_id:
        return ProtocolError(
            error_code="missing_strategy_id",
            message="submit_strategy requires an existing successful strategy_id.",
        )
    rationale = payload.get("rationale")
    if rationale is not None and not isinstance(rationale, str):
        return ProtocolError(
            error_code="invalid_rationale",
            message="submit_strategy rationale must be a string when provided.",
        )
    return SubmitStrategyRequest(strategy_id=strategy_id, rationale=rationale)


def _parse_datetime_filter(
    value: Any, field_name: str
) -> DatetimeFilter | ProtocolError | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        return ProtocolError(
            error_code=f"invalid_{field_name}",
            message=(
                f"{field_name} must be an object with optional "
                "start_datetime/end_datetime ISO-8601 strings."
            ),
        )
    unknown = sorted(set(value) - {"start_datetime", "end_datetime"})
    if unknown:
        return ProtocolError(
            error_code=f"invalid_{field_name}",
            message=f"{field_name} only supports start_datetime and end_datetime.",
            details={"unknown_fields": unknown},
        )
    parsed: dict[str, datetime | None] = {}
    for key in ("start_datetime", "end_datetime"):
        raw = value.get(key)
        try:
            item = datetime.fromisoformat(raw.replace("Z", "+00:00")) if raw else None
        except (AttributeError, ValueError):
            item = None
        if raw is not None and (item is None or item.tzinfo is None):
            return ProtocolError(
                error_code=f"invalid_{field_name}",
                message=f"{field_name}.{key} must be a timezone-aware ISO-8601 string.",
            )
        parsed[key] = item.astimezone(UTC) if item else None
    return DatetimeFilter(**parsed)
