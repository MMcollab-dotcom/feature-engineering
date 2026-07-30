"""Linear execution cost shared by the optimiser and backtest."""

from __future__ import annotations

from feature_engineering.config import CostsConfig


def linear_fee_rate(config: CostsConfig) -> float:
    """Return the one-way fee per unit of traded notional."""

    return float(config.linear_fee_bps) / 10_000.0


__all__ = ["linear_fee_rate"]
