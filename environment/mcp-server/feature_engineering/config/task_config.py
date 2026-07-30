"""Typed configuration loaded from the task's canonical YAML file."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class FrozenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WorkspaceConfig(FrozenConfig):
    profile: Literal["feature_engineering"]


class ScoringConfig(FrozenConfig):
    tradable_return: str
    market_beta: str


class DataConfig(FrozenConfig):
    manifest_path: str
    granularity: Literal["minutely", "hourly", "daily"]
    features: tuple[str, ...]
    targets: tuple[str, ...]
    scoring: ScoringConfig
    datetime_column: str
    symbol_column: str
    index_columns: tuple[str, str]

    @property
    def scoring_columns(self) -> tuple[str, str]:
        return (self.scoring.tradable_return, self.scoring.market_beta)

    @property
    def tradable_return_column(self) -> str:
        return self.scoring.tradable_return

    @property
    def market_beta_column(self) -> str:
        return self.scoring.market_beta


class MetricRoundingConfig(FrozenConfig):
    sharpe_decimals: int = Field(ge=0)
    return_decimals: int = Field(ge=0)
    rate_decimals: int = Field(ge=0)
    error_decimals: int = Field(ge=0)
    mse_decimals: int = Field(ge=0)
    correlation_decimals: int = Field(ge=0)


class BacktestConfig(FrozenConfig):
    rebalance_freq: int = Field(gt=0)
    engine: Literal["ema_smoothed"]
    portfolio_ema_hl_steps: int = Field(gt=0)
    portfolio_ema_tail_hl_steps: int = Field(gt=0)
    portfolio_ema_switch_steps: int = Field(gt=0)
    target_norm_weight: float = Field(gt=0.0)
    model_visible_metric_rounding: MetricRoundingConfig


class AgentConfig(FrozenConfig):
    max_research_attempts: int = Field(gt=0)
    response_error_budget: int = Field(ge=0)


class PredictionConfig(FrozenConfig):
    allowed_model_packages: tuple[str, ...]
    max_model_code_bytes: int = Field(gt=0)


class ExecutionConfig(FrozenConfig):
    initial_capital: float = Field(gt=0.0)
    timeout_seconds: float = Field(gt=0.0)


class CostsConfig(FrozenConfig):
    linear_fee_bps: float = Field(ge=0.0)


class RewardConfig(FrozenConfig):
    model: Literal["annualized_sharpe"]
    periods_per_year: int = Field(gt=0)
    primary_metric: Literal["sharpe"]
    reported_metrics: tuple[str, ...]


class TaskConfig(FrozenConfig):
    workspace: WorkspaceConfig
    data: DataConfig
    backtest: BacktestConfig
    agent: AgentConfig
    prediction: PredictionConfig
    execution: ExecutionConfig
    costs: CostsConfig
    reward: RewardConfig


def load_task_config(path: str | Path) -> TaskConfig:
    """Load the immutable task configuration from its sole value authority."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("task_config.yaml must contain a mapping.")
    data = raw.get("data")
    if not isinstance(data, dict):
        raise ValueError("task_config.yaml data must contain a mapping.")
    payload: dict[str, Any] = {
        **raw,
        "data": {
            **data,
            "manifest_path": str(config_path.parent / "data_manifest.json"),
        },
    }
    try:
        return TaskConfig.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid task_config.yaml: {exc}") from exc


__all__ = [
    "AgentConfig",
    "BacktestConfig",
    "CostsConfig",
    "DataConfig",
    "ExecutionConfig",
    "MetricRoundingConfig",
    "PredictionConfig",
    "RewardConfig",
    "ScoringConfig",
    "TaskConfig",
    "WorkspaceConfig",
    "load_task_config",
]
