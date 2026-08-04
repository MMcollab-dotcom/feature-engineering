"""Reward and metric helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rank_correlation(
    predictions: list[float] | np.ndarray,
    realized: list[float] | np.ndarray,
) -> float:
    x_values = np.asarray(predictions, dtype=float)
    y_values = np.asarray(realized, dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[finite]
    y_values = y_values[finite]
    if len(x_values) < 2:
        return 0.0
    x_rank = pd.Series(x_values).rank(method="average").to_numpy()
    y_rank = pd.Series(y_values).rank(method="average").to_numpy()
    if len(set(x_rank.tolist())) < 2 or len(set(y_rank.tolist())) < 2:
        return 0.0
    x_centered = x_rank - float(x_rank.mean())
    y_centered = y_rank - float(y_rank.mean())
    denominator = float(np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2)))
    if denominator <= 1.0e-12:
        return 0.0
    return float(np.sum(x_centered * y_centered) / denominator)
