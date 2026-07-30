"""Task-owned future-suffix probes for batch-dependent inference."""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import permutations
from typing import Sequence

import numpy as np
import pandas as pd

from feature_engineering.config import TaskConfig
from feature_engineering.core.fixed_data import SupervisedData
from feature_engineering.core.granularity import forecast_origin_end_datetime
from feature_engineering.submissions.dataframes import build_prediction_frame

POLICY = "future_suffix_v1"
PROBE_COUNT = 3
RTOL = 1.0e-7
ATOL = 1.0e-9
MAX_BUILD_ATTEMPTS = 32
MIN_CHANGED_FRACTION = 0.8
MIN_SCALED_DISPLACEMENT = 0.5


class AuditInputInsufficient(ValueError):
    pass


class AuditUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CausalProbe:
    X: pd.DataFrame
    past_mask: np.ndarray


def build_future_suffix_probes(
    X: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    seed: int,
) -> tuple[CausalProbe, ...]:
    datetimes = pd.Index(X.index.get_level_values(0)).unique().sort_values()
    if len(datetimes) < 5:
        raise AuditInputInsufficient("Temporal audit requires at least five datetimes.")
    columns = [
        name
        for name in feature_names
        if name in X.columns and _finite_unique_count(X[name]) > 1
    ]
    if not columns:
        raise AuditInputInsufficient(
            "Temporal audit requires one materially variable configured feature."
        )

    rng = random.Random(seed)
    interior = list(datetimes[1:-1])
    for cutoffs in _sample_cutoff_assignments(interior, rng=rng):
        probes: list[CausalProbe] = []
        for cutoff, transform in zip(cutoffs, ("low", "high", "bootstrap")):
            probe = _build_probe(
                X,
                columns=columns,
                cutoff=cutoff,
                transform=transform,
                rng=rng,
            )
            if probe is None:
                break
            probes.append(probe)
        if len(probes) == PROBE_COUNT:
            return tuple(probes)
    raise AuditUnavailable("Could not construct three material future-suffix probes.")


def audit_delta(
    candidate: np.ndarray,
    probe: np.ndarray,
    *,
    past_mask: np.ndarray,
) -> tuple[bool, float, float]:
    expected = candidate[past_mask]
    actual = probe[past_mask]
    difference = np.abs(expected - actual)
    max_abs = float(difference.max(initial=0.0))
    denominator = np.maximum(np.abs(expected), ATOL)
    max_rel = float((difference / denominator).max(initial=0.0))
    return (
        bool(np.allclose(expected, actual, rtol=RTOL, atol=ATOL, equal_nan=False)),
        max_abs,
        max_rel,
    )


def audit_summary(
    *,
    visibility: str,
    status: str,
    error_code: str | None,
    failed_probe_count: int = 0,
    max_abs_delta: float = 0.0,
    max_rel_delta: float = 0.0,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "policy": POLICY,
        "status": status,
        "probe_count": PROBE_COUNT,
        "rtol": RTOL,
        "atol": ATOL,
        "error_code": error_code,
    }
    if visibility == "public_detailed":
        summary.update(
            {
                "failed_probe_count": int(failed_probe_count),
                "max_abs_delta": float(max_abs_delta),
                "max_rel_delta": float(max_rel_delta),
            }
        )
    elif visibility != "hidden_fixed":
        raise ValueError("Unknown causal-audit visibility.")
    return summary


def validate_fixed_audit_frame(
    X: pd.DataFrame,
    *,
    feature_names: Sequence[str],
) -> None:
    """Fail trusted task/split validation when a fixed window cannot be audited."""

    try:
        build_future_suffix_probes(X, feature_names=feature_names, seed=0)
    except (AuditInputInsufficient, AuditUnavailable) as exc:
        raise ValueError(
            f"Fixed prediction window is not causally auditable: {exc}"
        ) from exc


def validate_fixed_prediction_window(
    *,
    config: TaskConfig,
    data: SupervisedData,
    start: object,
    end: object,
) -> None:
    origin_end = forecast_origin_end_datetime(
        pd.Timestamp(end), config.data.granularity
    )
    rows = data.frame.loc[
        (data.frame[config.data.datetime_column] >= pd.Timestamp(start))
        & (data.frame[config.data.datetime_column] <= origin_end)
        & data.frame[config.data.symbol_column].isin(data.symbols)
    ]
    validate_fixed_audit_frame(
        build_prediction_frame(config=config, rows=rows),
        feature_names=config.data.features,
    )


def _build_probe(
    X: pd.DataFrame,
    *,
    columns: list[str],
    cutoff: object,
    transform: str,
    rng: random.Random,
) -> CausalProbe | None:
    datetimes = X.index.get_level_values(0)
    past_mask = np.asarray(datetimes <= cutoff, dtype=bool)
    future_mask = ~past_mask
    if not future_mask.any():
        return None
    probe = X.copy(deep=True)
    if transform in {"low", "high"}:
        quantile = 0.05 if transform == "low" else 0.95
        for column in columns:
            past = X.loc[past_mask, column].to_numpy(dtype="float64")
            finite_past = past[np.isfinite(past)]
            if not len(finite_past):
                return None
            donor = float(np.quantile(finite_past, quantile, method="nearest"))
            probe.loc[future_mask, column] = donor
    else:
        symbols = np.asarray(X.index.get_level_values(1))
        column_positions = probe.columns.get_indexer(columns)
        values = X.iloc[:, column_positions].to_numpy(dtype="float64")
        finite_vectors = np.isfinite(values).all(axis=1)
        for _ in range(MAX_BUILD_ATTEMPTS):
            for symbol in pd.unique(symbols[future_mask]):
                donor_values = values[past_mask & finite_vectors & (symbols == symbol)]
                if not len(donor_values):
                    return None
                future_positions = np.flatnonzero(future_mask & (symbols == symbol))
                selected = rng.choices(
                    range(len(donor_values)),
                    k=len(future_positions),
                )
                probe.iloc[future_positions, column_positions] = donor_values[selected]
            if _is_material_change(
                X,
                probe,
                columns=columns,
                future_mask=future_mask,
            ):
                return CausalProbe(X=probe, past_mask=past_mask)
        return None
    if not _is_material_change(X, probe, columns=columns, future_mask=future_mask):
        return None
    return CausalProbe(X=probe, past_mask=past_mask)


def _is_material_change(
    original: pd.DataFrame,
    probe: pd.DataFrame,
    *,
    columns: list[str],
    future_mask: np.ndarray,
) -> bool:
    before = original.loc[future_mask, columns].to_numpy(dtype="float64")
    after = probe.loc[future_mask, columns].to_numpy(dtype="float64")
    eligible = np.isfinite(before) & np.isfinite(after)
    if not eligible.any():
        return False
    changed = eligible & (before != after)
    # Materiality is about perturbing the future observations seen by a
    # whole-batch transform. Sparse indicators legitimately remain unchanged
    # in most cells, so a per-cell threshold can reject a probe even when every
    # future row has several continuous features replaced. Require most
    # eligible rows to change in at least one configured feature instead.
    eligible_rows = eligible.any(axis=1)
    changed_rows = changed.any(axis=1)
    if float(changed_rows[eligible_rows].mean()) < MIN_CHANGED_FRACTION:
        return False
    scales = _feature_scales(original, columns)
    displacement = np.abs(after - before) / scales[future_mask]
    return float(displacement[eligible].mean()) >= MIN_SCALED_DISPLACEMENT


def _feature_scales(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    symbols = np.asarray(frame.index.get_level_values(1))
    scales = np.empty((len(frame), len(columns)), dtype="float64")
    for symbol in pd.unique(symbols):
        symbol_mask = symbols == symbol
        for column_index, column in enumerate(columns):
            values = frame.loc[symbol_mask, column].to_numpy(dtype="float64")
            values = values[np.isfinite(values)]
            if not len(values):
                scales[symbol_mask, column_index] = np.nan
                continue
            q25, q75 = np.quantile(values, [0.25, 0.75])
            scales[symbol_mask, column_index] = max(
                float(q75 - q25),
                0.01 * float(values.max() - values.min()),
                1e-12,
            )
    return scales


def _finite_unique_count(series: pd.Series) -> int:
    values = series.to_numpy(dtype="float64")
    return int(np.unique(values[np.isfinite(values)]).size)


def _sample_cutoff_assignments(
    interior: list[object],
    *,
    rng: random.Random,
) -> list[tuple[object, object, object]]:
    assignment_count = len(interior) * (len(interior) - 1) * (len(interior) - 2)
    if assignment_count <= MAX_BUILD_ATTEMPTS:
        assignments = list(permutations(interior, PROBE_COUNT))
        rng.shuffle(assignments)
        return assignments

    assignments = []
    seen: set[tuple[object, object, object]] = set()
    while len(assignments) < MAX_BUILD_ATTEMPTS:
        assignment = tuple(rng.sample(interior, PROBE_COUNT))
        if assignment not in seen:
            seen.add(assignment)
            assignments.append(assignment)
    return assignments


__all__ = [
    "ATOL",
    "AuditInputInsufficient",
    "AuditUnavailable",
    "CausalProbe",
    "POLICY",
    "PROBE_COUNT",
    "RTOL",
    "audit_delta",
    "audit_summary",
    "build_future_suffix_probes",
    "validate_fixed_audit_frame",
    "validate_fixed_prediction_window",
]
