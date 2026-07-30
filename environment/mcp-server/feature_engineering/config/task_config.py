"""The one frozen configuration shipped with this Harbor task."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class WorkspaceConfig:
    profile: str = "feature_engineering"


@dataclass(frozen=True, slots=True)
class DataConfig:
    manifest_path: str
    granularity: str = "minutely"
    features: tuple[str, ...] = (
        "open", "high", "low", "close", "volume", "quote_asset_volume",
        "number_of_trades", "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume", "weight_std_dollar_vol",
    )
    targets: tuple[str, ...] = ("target_horizon_1",)
    scoring_columns: tuple[str, ...] = ("tradable_return", "beta_10d_fwd_1")
    tradable_return_column: str = "tradable_return"
    market_beta_column: str = "beta_10d_fwd_1"
    datetime_column: str = "datetime"
    symbol_column: str = "symbol"
    index_columns: tuple[str, str] = ("datetime", "symbol")


@dataclass(frozen=True, slots=True)
class MetricRoundingConfig:
    sharpe_decimals: int = 2
    return_decimals: int = 4
    rate_decimals: int = 3
    exposure_decimals: int = 3
    error_decimals: int = 5
    correlation_decimals: int = 3


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    granularity: str = "minutely"
    rebalance_freq: int = 1
    engine: str = "ema_smoothed"
    portfolio_ema_hl_steps: int = 7
    portfolio_ema_tail_hl_steps: int = 4
    portfolio_ema_switch_steps: int = 7
    target_norm_weight: float = 0.1
    model_visible_metric_rounding: MetricRoundingConfig = field(default_factory=MetricRoundingConfig)


@dataclass(frozen=True, slots=True)
class AgentConfig:
    max_research_attempts: int = 50
    response_error_budget: int = 10


@dataclass(frozen=True, slots=True)
class PredictionConfig:
    allowed_model_packages: tuple[str, ...] = ("math", "statistics", "numpy", "pandas", "sklearn")
    max_model_code_bytes: int = 20_000
    require_prediction_output: bool = True


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    initial_capital: float = 100_000.0
    timeout_seconds: float = 1_800.0


@dataclass(frozen=True, slots=True)
class CostsConfig:
    linear_fee_bps: float = 1.0


@dataclass(frozen=True, slots=True)
class RewardConfig:
    model: str = "annualized_sharpe"
    periods_per_year: int = 525_600
    primary_metric: str = "sharpe"
    reported_metrics: tuple[str, ...] = (
        "sharpe",
        "cagr",
        "max_drawdown",
        "pearson_ic",
    )


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    enabled: bool = False


@dataclass(frozen=True, slots=True)
class TaskConfig:
    workspace: WorkspaceConfig
    data: DataConfig
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    costs: CostsConfig = field(default_factory=CostsConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)


def load_task_config(path: str | Path) -> TaskConfig:
    """Validate and return the one immutable task configuration."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw != _EXPECTED_FROZEN_CONFIG:
        differences = _config_differences(_EXPECTED_FROZEN_CONFIG, raw)
        raise ValueError(
            "task_config.yaml differs from the frozen feature-engineering task: "
            + ", ".join(differences[:20])
        )
    return TaskConfig(
        workspace=WorkspaceConfig(),
        data=DataConfig(manifest_path=str(config_path.parent / "data_manifest.json")),
    )


_EXPECTED_FROZEN_CONFIG: dict[str, Any] = {
    "workspace": {"profile": "feature_engineering"},
    "data": {
        "granularity": "minutely",
        "features": [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "weight_std_dollar_vol",
        ],
        "targets": ["target_horizon_1"],
        "scoring": {
            "tradable_return": "tradable_return",
            "market_beta": "beta_10d_fwd_1",
        },
    },
    "backtest": {
        "engine": "ema_smoothed",
        "rebalance_freq": 1,
        "portfolio_ema_hl_steps": 7,
        "portfolio_ema_tail_hl_steps": 4,
        "portfolio_ema_switch_steps": 7,
        "target_norm_weight": 0.1,
    },
    "agent": {
        "max_research_attempts": 50,
        "response_error_budget": 10,
    },
    "prediction": {
        "allowed_model_packages": [
            "math",
            "statistics",
            "numpy",
            "pandas",
            "sklearn",
        ],
        "max_model_code_bytes": 20_000,
    },
    "execution": {
        "initial_capital": 100_000.0,
        "timeout_seconds": 1_800.0,
    },
    "costs": {"linear_fee_bps": 1.0},
    "reward": {
        "model": "annualized_sharpe",
        "periods_per_year": 525_600,
        "primary_metric": "sharpe",
        "reported_metrics": [
            "sharpe",
            "cagr",
            "max_drawdown",
            "pearson_ic",
        ],
    },
    "harbor": {
        "submission_path": "/app/submission",
        "primary_metric": "sharpe",
    },
}


def _config_differences(expected: Any, actual: Any, path: str = "") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            child = f"{path}.{key}" if path else str(key)
            if key not in expected:
                differences.append(f"unexpected {child}")
            elif key not in actual:
                differences.append(f"missing {child}")
            else:
                differences.extend(
                    _config_differences(expected[key], actual[key], child)
                )
        return differences
    if expected != actual:
        return [f"{path} expected {expected!r}, got {actual!r}"]
    return []


__all__ = [
    "AgentConfig", "BacktestConfig", "BaselineConfig", "CostsConfig", "DataConfig",
    "ExecutionConfig", "MetricRoundingConfig", "PredictionConfig", "RewardConfig",
    "TaskConfig", "WorkspaceConfig", "load_task_config",
]
