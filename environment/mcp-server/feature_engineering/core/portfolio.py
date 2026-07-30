"""Single-horizon portfolio backtest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from feature_engineering.config import TaskConfig
from feature_engineering.core.costs import linear_fee_rate
from feature_engineering.core.ema_smoothed_engine import EmaSmoothedPortfolioEngine
from feature_engineering.core.fixed_data import SupervisedData
from feature_engineering.core.granularity import (
    forecast_origin_end_datetime,
    granularity_delta,
)
from feature_engineering.core.portfolio_engine import RebalanceRequest
from feature_engineering.core.reward import rank_correlation
from feature_engineering.submissions.strategy import CompiledStrategy, StrategyError

FORECAST_SCALE_EPSILON = 1.0e-8
FORECAST_SCALE_SHRINKAGE = 0.65


@dataclass(frozen=True, slots=True)
class BacktestSimulation:
    ok: bool
    metrics: dict[str, float]
    trace: dict[str, Any]
    error: StrategyError | None = None


def calculate_forecast_beta(
    forecasts: np.ndarray,
    training_targets: np.ndarray,
) -> float:
    """Fit the zero-intercept slope from forecasts to training targets."""

    forecast = np.asarray(forecasts, dtype="float64").reshape(-1)
    target = np.asarray(training_targets, dtype="float64").reshape(-1)
    if len(forecast) != len(target):
        raise ValueError("Forecasts and training targets must have equal length.")
    with np.errstate(over="ignore", invalid="ignore"):
        beta = float(forecast @ target) / (
            float(forecast @ forecast) + FORECAST_SCALE_EPSILON
        )
    if not np.isfinite(beta):
        raise ValueError("Training forecast beta is not finite.")
    return beta


def calculate_forecast_scale(
    forecasts: np.ndarray,
    training_targets: np.ndarray,
) -> float:
    """Return the shrunk non-negative training forecast beta."""

    return forecast_scale_from_beta(
        calculate_forecast_beta(forecasts, training_targets)
    )


def forecast_scale_from_beta(forecast_beta: float) -> float:
    """Shrink one finite training beta without reversing forecast signs."""

    if not np.isfinite(forecast_beta):
        raise ValueError("Training forecast beta is not finite.")
    return FORECAST_SCALE_SHRINKAGE * max(0.0, float(forecast_beta))


def calculate_median_signal_size(
    scaled_forecasts: np.ndarray,
    market_betas: np.ndarray,
) -> float:
    """Return the median beta-neutral gross signal across training timestamps."""

    forecasts = np.asarray(scaled_forecasts, dtype="float64")
    betas = np.asarray(market_betas, dtype="float64")
    if forecasts.ndim != 2 or forecasts.shape != betas.shape or not forecasts.size:
        raise ValueError(
            "Training forecasts and market betas must be equal non-empty matrices."
        )
    if not np.isfinite(forecasts).all() or not np.isfinite(betas).all():
        raise ValueError("Training forecasts and market betas must be finite.")
    beta_norm_squared = np.sum(betas * betas, axis=1)
    projection_scale = np.divide(
        np.sum(betas * forecasts, axis=1),
        beta_norm_squared,
        out=np.zeros(len(forecasts), dtype="float64"),
        where=beta_norm_squared > np.finfo("float64").eps,
    )
    projected_forecasts = forecasts - betas * projection_scale[:, None]
    median_signal_size = float(np.median(np.abs(projected_forecasts).sum(axis=1)))
    if not np.isfinite(median_signal_size) or median_signal_size < 0.0:
        raise ValueError("Training median signal size must be finite and non-negative.")
    return median_signal_size


def execute_backtest(
    *,
    config: TaskConfig,
    public_data: SupervisedData,
    start: pd.Timestamp,
    end: pd.Timestamp,
    strategy: CompiledStrategy,
    predictions: pd.DataFrame,
    forecast_scale: float,
    median_signal_size: float,
) -> BacktestSimulation:
    """Replay one prediction batch through the scheduled single-horizon LP."""

    data = config.data
    symbols = public_data.symbols
    last_origin = forecast_origin_end_datetime(end, data.granularity)
    datetimes = public_data.datetimes[
        (public_data.datetimes >= start) & (public_data.datetimes <= last_origin)
    ]
    if not len(datetimes):
        raise ValueError("Backtest window needs at least one forecast origin.")

    expected_index = pd.MultiIndex.from_product(
        [datetimes, symbols],
        names=[data.datetime_column, data.symbol_column],
    )
    prediction_index = predictions.index
    if (
        not isinstance(prediction_index, pd.MultiIndex)
        or prediction_index.nlevels != 2
        or list(prediction_index.names) != list(expected_index.names)
        or not prediction_index.is_unique
        or len(prediction_index) != len(expected_index)
    ):
        raise ValueError(
            "Prediction frame must exactly cover the backtest datetime/symbol keys."
        )
    if list(predictions.columns) != list(data.targets):
        raise ValueError(
            "Prediction frame columns must exactly equal configured targets."
        )

    if prediction_index.equals(expected_index):
        prediction_positions: slice | np.ndarray = slice(None)
    else:
        prediction_positions = prediction_index.get_indexer(expected_index)
        if np.any(prediction_positions < 0):
            raise ValueError(
                "Prediction frame must exactly cover the backtest datetime/symbol keys."
            )
    raw_forecasts = predictions[data.targets[0]].to_numpy(dtype="float64")[
        prediction_positions
    ]

    frame = public_data.frame
    frame_datetimes = frame[data.datetime_column]
    frame_symbols = frame[data.symbol_column].to_numpy(copy=False)
    in_window = (
        (frame_datetimes >= start)
        & (frame_datetimes <= last_origin)
        & frame[data.symbol_column].isin(symbols)
    )
    candidate_positions = np.flatnonzero(in_window.to_numpy())
    candidate_index = pd.MultiIndex.from_arrays(
        (
            frame_datetimes.array[candidate_positions],
            frame_symbols[candidate_positions],
        ),
        names=expected_index.names,
    )
    if candidate_index.equals(expected_index):
        row_positions = candidate_positions
    else:
        frame_index = pd.MultiIndex.from_arrays(
            (
                frame_datetimes.array,
                frame_symbols,
            ),
            names=expected_index.names,
        )
        if not frame_index.is_unique:
            raise ValueError("Backtest data datetime/symbol keys must be unique.")
        row_positions = frame_index.get_indexer(expected_index)
        if np.any(row_positions < 0):
            raise ValueError("Backtest data does not cover the requested window.")

    step_count = len(datetimes)
    symbol_count = len(symbols)

    def column_matrix(column: str) -> np.ndarray:
        return (
            frame[column]
            .to_numpy(dtype="float64")[row_positions]
            .reshape(step_count, symbol_count)
        )

    forecast_scale = float(forecast_scale)
    if not np.isfinite(forecast_scale) or forecast_scale < 0.0:
        raise ValueError("forecast_scale must be finite and non-negative.")
    realized_targets = column_matrix(data.targets[0])
    raw_forecasts = raw_forecasts.reshape(step_count, symbol_count)
    scaled_forecasts = (
        raw_forecasts * forecast_scale * config.backtest.target_norm_weight
    )
    market_betas = column_matrix(data.market_beta_column)
    tradable_returns = column_matrix(data.tradable_return_column)
    fee_rate = linear_fee_rate(config.costs)
    engine = EmaSmoothedPortfolioEngine(
        config=config.backtest,
        symbol_count=symbol_count,
        median_signal_size=median_signal_size,
        fee_rate=fee_rate,
        max_gross_exposure=strategy.max_gross_exposure,
    )

    nav = float(config.execution.initial_capital)
    weights = np.zeros(symbol_count, dtype="float64")
    previous_weights = np.empty((step_count, symbol_count), dtype="float64")
    target_weights = np.empty_like(previous_weights)
    end_weights = np.empty_like(previous_weights)
    step_returns = np.empty(step_count, dtype="float64")
    nav_values = np.empty(step_count, dtype="float64")
    fee_values = np.empty(step_count, dtype="float64")
    traded_notionals = np.empty(step_count, dtype="float64")
    beta_exposures = np.empty(step_count, dtype="float64")
    gross_values = np.empty(step_count, dtype="float64")
    target_gross_values = np.empty(step_count, dtype="float64")
    net_values = np.empty(step_count, dtype="float64")
    did_rebalance_values = np.zeros(step_count, dtype="bool")
    optimisations: list[dict[str, float | int | bool] | None] = [None] * step_count
    total_turnover = 0.0
    total_fee = 0.0
    rebalance_count = 0
    completed_steps = 0
    period = granularity_delta(data.granularity)
    datetime_strings = tuple(value.isoformat() for value in datetimes)
    execution_datetime_strings = tuple(
        (value + period).isoformat() for value in datetimes
    )
    realization_datetime_strings = tuple(
        (value + 2 * period).isoformat() for value in datetimes
    )

    def materialize_trace(
        count: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return (
            _materialize_trace_steps(
                count=count,
                symbols=symbols,
                target_name=data.targets[0],
                datetime_strings=datetime_strings,
                execution_datetime_strings=execution_datetime_strings,
                realization_datetime_strings=realization_datetime_strings,
                raw_forecasts=raw_forecasts,
                scaled_forecasts=scaled_forecasts,
                market_betas=market_betas,
                realized_targets=realized_targets,
                previous_weights=previous_weights,
                target_weights=target_weights,
                end_weights=end_weights,
                did_rebalance=did_rebalance_values,
                beta_exposures=beta_exposures,
                nav_values=nav_values,
                step_returns=step_returns,
                fee_values=fee_values,
                traded_notionals=traded_notionals,
                optimisations=optimisations,
            ),
            _materialize_prediction_records(
                count=count,
                symbols=symbols,
                datetime_strings=datetime_strings,
                raw_forecasts=raw_forecasts,
                scaled_forecasts=scaled_forecasts,
                realized_targets=realized_targets,
            ),
        )

    for step_index in range(step_count):
        previous_weights[step_index] = weights
        did_rebalance = step_index % config.backtest.rebalance_freq == 0
        did_rebalance_values[step_index] = did_rebalance
        optimisation: dict[str, float | int | bool] | None = None
        if did_rebalance:
            rebalance_count += 1
            try:
                decision = engine.rebalance(
                    RebalanceRequest(
                        step_index=step_index,
                        scaled_forecast=scaled_forecasts[step_index],
                        market_beta=market_betas[step_index],
                        pretrade_weights=weights,
                    )
                )
            except RuntimeError as exc:
                trace_steps, prediction_records = materialize_trace(completed_steps)
                return _failed_backtest(
                    StrategyError(
                        "portfolio_optimisation_failed",
                        str(exc),
                        contract_failure=True,
                    ),
                    forecast_scale=forecast_scale,
                    rebalance_freq=config.backtest.rebalance_freq,
                    backtest_engine=config.backtest.engine,
                    target_norm_weight=config.backtest.target_norm_weight,
                    steps=trace_steps,
                    prediction_records=prediction_records,
                )
            target = decision.target_weights
            optimisation = decision.diagnostics
        else:
            # No rebalance means no trade: the return-drifted weights carry into
            # this step unchanged until the next scheduled solve.
            target = weights.copy()

        turnover_weight = float(np.abs(target - weights).sum())
        if optimisation is not None:
            optimisation["estimated_execution_cost"] = fee_rate * turnover_weight
        optimisations[step_index] = optimisation
        traded_notional = turnover_weight * nav
        fee = traded_notional * fee_rate
        gross_return = float(target @ tradable_returns[step_index])
        step_return = gross_return - fee / nav
        growth = 1.0 + step_return
        next_nav = nav * growth
        if not np.isfinite(next_nav) or growth <= 0.0:
            trace_steps, prediction_records = materialize_trace(completed_steps)
            return _failed_backtest(
                StrategyError(
                    "portfolio_insolvent",
                    "Chosen exposure made portfolio NAV non-positive or non-finite.",
                    details={"datetime": datetime_strings[step_index]},
                    contract_failure=True,
                ),
                forecast_scale=forecast_scale,
                rebalance_freq=config.backtest.rebalance_freq,
                backtest_engine=config.backtest.engine,
                target_norm_weight=config.backtest.target_norm_weight,
                steps=trace_steps,
                prediction_records=prediction_records,
            )
        end = target * (1.0 + tradable_returns[step_index]) / growth

        target_weights[step_index] = target
        end_weights[step_index] = end
        target_gross_values[step_index] = float(np.abs(target).sum())
        gross_values[step_index] = float(np.abs(end).sum())
        net_values[step_index] = abs(float(end.sum()))
        step_returns[step_index] = step_return
        nav_values[step_index] = next_nav
        fee_values[step_index] = fee
        traded_notionals[step_index] = traded_notional
        beta_exposures[step_index] = float(target @ market_betas[step_index])
        total_turnover += turnover_weight
        total_fee += fee
        completed_steps = step_index + 1
        nav = float(next_nav)
        weights = end

    try:
        metrics = backtest_metrics(
            initial_nav=float(config.execution.initial_capital),
            final_nav=nav,
            step_returns=step_returns,
            turnover=total_turnover,
            gross_values=gross_values,
            target_gross_values=target_gross_values,
            net_values=net_values,
            fee_paid=total_fee,
            periods_per_year=config.reward.periods_per_year,
            prediction_records=None,
            forecast_scale=forecast_scale,
            rebalance_count=rebalance_count,
            prediction_values=scaled_forecasts.reshape(-1),
            realized_target_values=realized_targets.reshape(-1),
        )
    except ValueError as exc:
        trace_steps, prediction_records = materialize_trace(completed_steps)
        return _failed_backtest(
            StrategyError(
                "backtest_metrics_non_finite",
                str(exc),
                contract_failure=True,
            ),
            forecast_scale=forecast_scale,
            rebalance_freq=config.backtest.rebalance_freq,
            backtest_engine=config.backtest.engine,
            target_norm_weight=config.backtest.target_norm_weight,
            steps=trace_steps,
            prediction_records=prediction_records,
        )
    trace_steps, prediction_records = materialize_trace(completed_steps)
    return BacktestSimulation(
        ok=True,
        metrics=metrics,
        trace={
            "backtest_engine": config.backtest.engine,
            "forecast_scale": forecast_scale,
            "rebalance_freq": config.backtest.rebalance_freq,
            "target_norm_weight": config.backtest.target_norm_weight,
            "steps": trace_steps,
            "prediction_records": prediction_records,
        },
    )


def _symbol_values(symbols: tuple[str, ...], values: np.ndarray) -> dict[str, float]:
    return {symbol: float(values[index]) for index, symbol in enumerate(symbols)}


def _materialize_trace_steps(
    *,
    count: int,
    symbols: tuple[str, ...],
    target_name: str,
    datetime_strings: tuple[str, ...],
    execution_datetime_strings: tuple[str, ...],
    realization_datetime_strings: tuple[str, ...],
    raw_forecasts: np.ndarray,
    scaled_forecasts: np.ndarray,
    market_betas: np.ndarray,
    realized_targets: np.ndarray,
    previous_weights: np.ndarray,
    target_weights: np.ndarray,
    end_weights: np.ndarray,
    did_rebalance: np.ndarray,
    beta_exposures: np.ndarray,
    nav_values: np.ndarray,
    step_returns: np.ndarray,
    fee_values: np.ndarray,
    traded_notionals: np.ndarray,
    optimisations: list[dict[str, float | int | bool] | None],
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for index in range(count):
        steps.append(
            {
                "datetime": datetime_strings[index],
                "execution_datetime": execution_datetime_strings[index],
                "realization_datetime": realization_datetime_strings[index],
                "did_rebalance": bool(did_rebalance[index]),
                "raw_forecasts": _symbol_values(symbols, raw_forecasts[index]),
                "scaled_forecasts": _symbol_values(symbols, scaled_forecasts[index]),
                "market_betas": _symbol_values(symbols, market_betas[index]),
                "beta_exposure": float(beta_exposures[index]),
                "previous_weights": _symbol_values(symbols, previous_weights[index]),
                "target_weights": _symbol_values(symbols, target_weights[index]),
                "end_weights": _symbol_values(symbols, end_weights[index]),
                "nav": float(nav_values[index]),
                "step_return": float(step_returns[index]),
                "fee_paid": float(fee_values[index]),
                "traded_notional": float(traded_notionals[index]),
                "realized_targets": {
                    "target_name": target_name,
                    "values": _symbol_values(symbols, realized_targets[index]),
                },
                "optimisation": optimisations[index],
            }
        )
    return steps


def _materialize_prediction_records(
    *,
    count: int,
    symbols: tuple[str, ...],
    datetime_strings: tuple[str, ...],
    raw_forecasts: np.ndarray,
    scaled_forecasts: np.ndarray,
    realized_targets: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "datetime": datetime_strings[step],
            "symbol": symbol,
            "raw_prediction": float(raw_forecasts[step, symbol_index]),
            "prediction": float(scaled_forecasts[step, symbol_index]),
            "realized_target": float(realized_targets[step, symbol_index]),
        }
        for step in range(count)
        for symbol_index, symbol in enumerate(symbols)
    ]


def backtest_metrics(
    *,
    initial_nav: float,
    final_nav: float,
    step_returns: list[float] | np.ndarray,
    turnover: float,
    gross_values: list[float] | np.ndarray,
    target_gross_values: list[float] | np.ndarray,
    net_values: list[float] | np.ndarray,
    fee_paid: float,
    periods_per_year: int,
    prediction_records: list[dict[str, Any]] | None,
    forecast_scale: float,
    rebalance_count: int,
    prediction_values: np.ndarray | None = None,
    realized_target_values: np.ndarray | None = None,
) -> dict[str, float]:
    returns = np.asarray(step_returns, dtype="float64")
    mean = float(returns.mean()) if len(returns) else 0.0
    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe = 0.0 if std <= 1.0e-12 else mean / std * np.sqrt(periods_per_year)
    growth = final_nav / initial_nav
    with np.errstate(over="ignore", invalid="ignore"):
        cagr = float(np.expm1(np.log(growth) * periods_per_year / len(returns)))
        nav_curve = initial_nav * np.concatenate(
            (np.ones(1), np.cumprod(1.0 + returns))
        )
    running_max = np.maximum.accumulate(nav_curve)
    drawdowns = nav_curve / running_max - 1.0
    metrics = {
        "annualized_sharpe": float(sharpe),
        "cagr": cagr,
        "cumulative_after_cost_return": float(growth - 1.0),
        "max_drawdown": float(drawdowns.min()),
        "turnover": float(turnover),
        "average_gross_exposure": (
            float(np.mean(gross_values)) if len(gross_values) else 0.0
        ),
        "average_target_gross_exposure": (
            float(np.mean(target_gross_values)) if len(target_gross_values) else 0.0
        ),
        "average_net_exposure": (
            float(np.mean(net_values)) if len(net_values) else 0.0
        ),
        "fee_paid": float(fee_paid),
        "forecast_scale": float(forecast_scale),
        "rebalance_count": float(rebalance_count),
        **(
            _prediction_metrics(prediction_records or [])
            if prediction_values is None and realized_target_values is None
            else _prediction_metrics_from_arrays(
                prediction_values,
                realized_target_values,
            )
        ),
    }
    metrics["pearson_ic"] = float(metrics["correlation"])
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("Backtest metrics are not finite.")
    return metrics


def _prediction_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {
            "correlation": 0.0,
            "mse": 0.0,
            "mean_abs_prediction_error": 0.0,
            "directional_accuracy": 0.0,
            "prediction_return_rank_correlation": 0.0,
            "prediction_label_count": 0.0,
        }
    predictions = np.asarray([record["prediction"] for record in records])
    realized = np.asarray([record["realized_target"] for record in records])
    return _prediction_metrics_from_arrays(predictions, realized)


def _prediction_metrics_from_arrays(
    predictions: np.ndarray | None,
    realized: np.ndarray | None,
) -> dict[str, float]:
    if predictions is None or realized is None:
        raise ValueError(
            "Prediction and realized-target arrays must be supplied together."
        )
    predictions = np.asarray(predictions, dtype="float64").reshape(-1)
    realized = np.asarray(realized, dtype="float64").reshape(-1)
    if len(predictions) != len(realized):
        raise ValueError(
            "Prediction and realized-target arrays must have equal length."
        )
    if not len(predictions):
        return {
            "correlation": 0.0,
            "mse": 0.0,
            "mean_abs_prediction_error": 0.0,
            "directional_accuracy": 0.0,
            "prediction_return_rank_correlation": 0.0,
            "prediction_label_count": 0.0,
        }
    residual = predictions - realized
    if (
        len(predictions) < 2
        or np.std(predictions) <= 1.0e-12
        or np.std(realized) <= 1.0e-12
    ):
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(predictions, realized)[0, 1])
    return {
        "correlation": correlation,
        "mse": float(np.mean(residual**2)),
        "mean_abs_prediction_error": float(np.mean(np.abs(residual))),
        "directional_accuracy": float(
            np.mean(np.sign(predictions) == np.sign(realized))
        ),
        "prediction_return_rank_correlation": float(
            rank_correlation(predictions, realized)
        ),
        "prediction_label_count": float(len(predictions)),
    }


def _failed_backtest(
    error: StrategyError,
    *,
    forecast_scale: float | None,
    rebalance_freq: int,
    backtest_engine: str,
    target_norm_weight: float,
    steps: list[dict[str, Any]] | None = None,
    prediction_records: list[dict[str, Any]] | None = None,
) -> BacktestSimulation:
    return BacktestSimulation(
        ok=False,
        metrics={},
        trace={
            "backtest_engine": backtest_engine,
            "forecast_scale": forecast_scale,
            "rebalance_freq": rebalance_freq,
            "target_norm_weight": target_norm_weight,
            "steps": list(steps or []),
            "prediction_records": list(prediction_records or []),
            "error": error.to_payload(),
        },
        error=error,
    )


__all__ = [
    "BacktestSimulation",
    "backtest_metrics",
    "calculate_forecast_beta",
    "calculate_forecast_scale",
    "calculate_median_signal_size",
    "execute_backtest",
    "forecast_scale_from_beta",
]
