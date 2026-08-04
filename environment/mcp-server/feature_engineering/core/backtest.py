"""Fixed-window backtest entrypoint for strategy attempts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pandas as pd

from evalenv_shared.worker import WorkerProtocolError
from feature_engineering.config import (
    MetricRoundingConfig,
    TaskConfig,
)
from feature_engineering.core.data import SupervisedData
from feature_engineering.core.granularity import forecast_origin_end_datetime
from feature_engineering.core.portfolio import execute_backtest
from feature_engineering.submissions.causal_audit import audit_summary
from feature_engineering.submissions.dataframes import (
    build_prediction_frame,
    write_dataframe,
)
from feature_engineering.submissions.registry import TrainedModelRegistry
from feature_engineering.submissions.runner import (
    WorkerExecutionError,
    is_submitted_prediction_failure,
    predict_artifact,
)
from feature_engineering.submissions.strategy import CompiledStrategy, StrategyError


@dataclass(frozen=True, slots=True)
class BacktestResult:
    ok: bool
    metrics: dict[str, float]
    trace: dict[str, Any]
    model_visible: dict[str, Any]
    audit: dict[str, Any] | None = None
    error: StrategyError | None = None


def _prepare_backtest_payload(
    *,
    config: TaskConfig,
    public_data: SupervisedData,
    lower: pd.Timestamp,
    origin_end: pd.Timestamp,
    operation_directory: Any,
) -> tuple[Any, dict[str, Any]]:
    X = _build_backtest_prediction_frame(
        config,
        public_data,
        lower,
        origin_end,
    )
    X_payload = write_dataframe(operation_directory / "batch-X.arrow", X)
    return X, X_payload


async def run_backtest(
    *,
    config: TaskConfig,
    public_data: SupervisedData,
    strategy: CompiledStrategy,
    registry: TrainedModelRegistry,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    audit_visibility: str = "public_detailed",
    worker_host: Any = None,
) -> BacktestResult:
    lower = public_data.start_datetime if start is None else pd.Timestamp(start)
    upper = public_data.end_datetime if end is None else pd.Timestamp(end)
    origin_end = forecast_origin_end_datetime(upper, config.data.granularity)
    model = registry.get(strategy.model_id)
    with registry.operation_directory(strategy.model_id) as operation_directory:
        X, X_payload = await asyncio.to_thread(
            _prepare_backtest_payload,
            config=config,
            public_data=public_data,
            lower=lower,
            origin_end=origin_end,
            operation_directory=operation_directory,
        )
        predicted = await predict_artifact(
            code=model.model_code,
            expected_source_hash=model.model_code_sha256,
            allowed_imports=tuple(config.prediction.allowed_model_packages),
            max_code_bytes=config.prediction.max_model_code_bytes,
            X=X_payload,
            artifact_path=operation_directory / "model.joblib",
            expected_artifact_hash=model.artifact_sha256,
            target_names=model.target_names,
            configured_feature_names=tuple(config.data.features),
            expected_inference_columns=model.inference_columns,
            audit_visibility=audit_visibility,
            timeout_seconds=config.execution.timeout_seconds,
            worker_host=worker_host,
        )
        if isinstance(predicted, WorkerExecutionError):
            if not is_submitted_prediction_failure(predicted.error_code):
                raise WorkerProtocolError(
                    f"Trusted artifact prediction failed: {predicted.error_code}."
                )
            audit = (
                dict(predicted.details["causal_audit"])
                if predicted.details
                and isinstance(predicted.details.get("causal_audit"), dict)
                else audit_summary(
                    visibility=audit_visibility,
                    status="rejected",
                    error_code=predicted.error_code,
                )
            )
            details = dict(predicted.details or {})
            details["causal_audit"] = audit
            error = StrategyError(
                predicted.error_code,
                predicted.message,
                details=details,
                contract_failure=True,
            )
            result = BacktestResult(
                ok=False,
                metrics={},
                trace={
                    "backtest_engine": config.backtest.engine,
                    "target_norm_weight": config.backtest.target_norm_weight,
                    "steps": [],
                    "prediction_records": [],
                },
                error=error,
                model_visible=error.to_payload(),
                audit=audit,
            )
        else:
            portfolio = await asyncio.to_thread(
                execute_backtest,
                config=config,
                public_data=public_data,
                start=lower,
                end=upper,
                strategy=strategy,
                predictions=predicted.frame,
                forecast_scale=model.forecast_scale,
                median_signal_size=model.median_signal_size,
            )
            audit = dict(predicted.audit)
            visible = (
                {
                    "ok": True,
                    "metrics": _model_visible_metrics(config, portfolio.metrics),
                }
                if portfolio.ok
                else portfolio.error.to_payload()
            )
            result = BacktestResult(
                ok=portfolio.ok,
                metrics=portfolio.metrics,
                trace=portfolio.trace,
                error=portfolio.error,
                model_visible=visible,
                audit=audit,
            )
    return result


def _build_backtest_prediction_frame(
    config: TaskConfig,
    public_data: SupervisedData,
    lower: pd.Timestamp,
    origin_end: pd.Timestamp,
) -> Any:
    rows = public_data.frame.loc[
        (public_data.frame[config.data.datetime_column] >= lower)
        & (public_data.frame[config.data.datetime_column] <= origin_end)
        & public_data.frame[config.data.symbol_column].isin(public_data.symbols)
    ]
    return build_prediction_frame(config=config, rows=rows)


def _model_visible_metrics(
    config: TaskConfig, metrics: dict[str, float]
) -> dict[str, float]:
    rounding = config.backtest.model_visible_metric_rounding
    return {
        "annualized_sharpe": _round_metric(metrics, "annualized_sharpe", rounding),
        "cagr": _round_metric(metrics, "cagr", rounding),
        "cumulative_after_cost_return": _round_metric(
            metrics, "cumulative_after_cost_return", rounding
        ),
        "max_drawdown": _round_metric(metrics, "max_drawdown", rounding),
        "turnover": _round_metric(metrics, "turnover", rounding),
        "correlation": _round_metric(metrics, "correlation", rounding),
        "mse": _round_metric(metrics, "mse", rounding),
        "mean_abs_prediction_error": _round_metric(
            metrics, "mean_abs_prediction_error", rounding
        ),
        "directional_accuracy": _round_metric(
            metrics, "directional_accuracy", rounding
        ),
        "prediction_return_rank_correlation": _round_metric(
            metrics, "prediction_return_rank_correlation", rounding
        ),
    }


def _round_metric(
    metrics: dict[str, float],
    key: str,
    rounding: MetricRoundingConfig,
) -> float:
    value = float(metrics.get(key, 0.0))
    if key == "annualized_sharpe":
        decimals = rounding.sharpe_decimals
    elif key in {
        "cumulative_after_cost_return",
        "cagr",
        "max_drawdown",
        "turnover",
    }:
        decimals = rounding.return_decimals
    elif key == "directional_accuracy":
        decimals = rounding.rate_decimals
    elif key == "mse":
        decimals = rounding.mse_decimals
    elif key in {"correlation", "prediction_return_rank_correlation"}:
        decimals = rounding.correlation_decimals
    else:
        decimals = rounding.error_decimals
    return round(value, int(decimals))
