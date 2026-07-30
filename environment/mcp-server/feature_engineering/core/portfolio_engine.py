"""Shared request and decision boundary for portfolio engines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RebalanceRequest:
    step_index: int
    scaled_forecast: np.ndarray
    market_beta: np.ndarray
    pretrade_weights: np.ndarray


@dataclass(frozen=True, slots=True)
class RebalanceDecision:
    target_weights: np.ndarray
    diagnostics: dict[str, float | int | bool]


__all__ = ["RebalanceDecision", "RebalanceRequest"]
