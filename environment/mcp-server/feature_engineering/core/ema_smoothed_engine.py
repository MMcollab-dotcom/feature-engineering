"""EMA-smoothed portfolio engine."""

from __future__ import annotations

import numpy as np

from feature_engineering.config import BacktestConfig
from feature_engineering.core.dual_optimiser import optimise_linear_position
from feature_engineering.core.portfolio_engine import (
    RebalanceDecision,
    RebalanceRequest,
)


class EmaSmoothedPortfolioEngine:
    def __init__(
        self,
        *,
        config: BacktestConfig,
        scaled_forecasts: np.ndarray,
        market_betas: np.ndarray,
        fee_rate: float,
        max_gross_exposure: float,
    ) -> None:
        if (
            config.portfolio_ema_hl_steps is None
            or config.portfolio_ema_tail_hl_steps is None
            or config.portfolio_ema_switch_steps is None
        ):
            raise ValueError("EMA-smoothed portfolio parameters are required.")

        symbol_count = scaled_forecasts.shape[1]
        beta_norm_squared = np.sum(market_betas * market_betas, axis=1)
        projected_forecasts = (
            scaled_forecasts
            - market_betas
            * (np.sum(market_betas * scaled_forecasts, axis=1) / beta_norm_squared)[
                :, None
            ]
        )
        self.median_signal_size = float(
            np.median(np.abs(projected_forecasts).sum(axis=1))
        )
        if max_gross_exposure > 0.0 and self.median_signal_size > 0.0:
            self.quadratic_penalty = self.median_signal_size / max_gross_exposure
        else:
            self.quadratic_penalty = 1.0

        ema_decay = 2.0 ** (-1.0 / config.portfolio_ema_hl_steps)
        ema_alpha = 1.0 - ema_decay
        self.ema_decay = ema_decay
        self.ema_tail_decay = 2.0 ** (-1.0 / config.portfolio_ema_tail_hl_steps)
        switch_steps = config.portfolio_ema_switch_steps
        ema_kernel_mass = (
            1.0
            - ema_decay ** (switch_steps + 1)
            + ema_alpha
            * ema_decay**switch_steps
            * self.ema_tail_decay
            / (1.0 - self.ema_tail_decay)
        )
        self.ema_entry_weight = ema_alpha / ema_kernel_mass
        self.ema_front = np.zeros(
            (switch_steps + 1, symbol_count),
            dtype="float64",
        )
        self.ema_tail = np.zeros(symbol_count, dtype="float64")
        self.previous_dual = np.zeros(2 * symbol_count, dtype="float64")
        self.identity = np.eye(symbol_count, dtype="float64")
        self.fee_rate = fee_rate
        self.max_gross_exposure = max_gross_exposure
        self.rebalance_count = 0

    def rebalance(self, request: RebalanceRequest) -> RebalanceDecision:
        self.rebalance_count += 1
        beta = request.market_beta
        projection = self.identity - np.outer(beta, beta) / float(beta @ beta)
        slot = (self.rebalance_count - 1) % len(self.ema_front)
        exiting = self.ema_front[slot].copy()
        self.ema_front[slot] = 0.0
        self.ema_front = self.ema_decay * (self.ema_front @ projection.T)
        self.ema_tail = self.ema_tail_decay * (projection @ (self.ema_tail + exiting))
        older = self.ema_front.sum(axis=0) + self.ema_tail
        position_map = self.ema_entry_weight * projection
        solved = optimise_linear_position(
            expected_returns=projection @ request.scaled_forecast,
            older_position=older,
            position_map=position_map,
            pretrade_position=request.pretrade_weights,
            market_beta=beta,
            linear_execution_cost=self.fee_rate,
            future_decay_cost=self.fee_rate * self.ema_entry_weight,
            quadratic_penalty=self.quadratic_penalty,
            gross_budget=self.max_gross_exposure,
            initial_dual=self.previous_dual,
        )
        self.previous_dual = solved.dual
        self.ema_front[slot] = position_map @ solved.allocation
        target = older + self.ema_front[slot]
        gross = float(np.abs(target).sum())
        gross_overlay = min(1.0, self.max_gross_exposure / gross) if gross else 1.0
        target = gross_overlay * target
        self.ema_front *= gross_overlay
        self.ema_tail *= gross_overlay
        return RebalanceDecision(
            target_weights=target,
            diagnostics={
                "expected_return": solved.expected_return,
                "turnover": solved.turnover,
                "objective_execution_cost": solved.execution_cost,
                "objective_future_decay_cost": solved.future_decay_cost,
                "future_decay_cost_rate": self.fee_rate * self.ema_entry_weight,
                "objective_value": solved.objective_value,
                "iterations": solved.iterations,
                "quadratic_penalty": self.quadratic_penalty,
                "median_signal_size": self.median_signal_size,
                "gross_constraint_active": solved.gross_constraint_active,
                "desired_gross_exposure": float(np.abs(solved.allocation).sum()),
                "desired_beta_exposure": float(beta @ solved.allocation),
                "gross_overlay": gross_overlay,
            },
        )


__all__ = ["EmaSmoothedPortfolioEngine"]
