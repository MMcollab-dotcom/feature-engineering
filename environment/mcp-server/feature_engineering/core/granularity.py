"""Time-unit primitives for feature data and portfolio accounting."""

from __future__ import annotations

import pandas as pd

_GRANULARITY_DELTAS = {
    "minutely": pd.Timedelta(minutes=1),
    "hourly": pd.Timedelta(hours=1),
    "daily": pd.Timedelta(days=1),
}


def granularity_delta(granularity: str) -> pd.Timedelta:
    try:
        return _GRANULARITY_DELTAS[str(granularity)]
    except KeyError as exc:
        supported = ", ".join(_GRANULARITY_DELTAS)
        raise ValueError(
            f"Unsupported granularity {granularity!r}; supported values: {supported}."
        ) from exc


def periods_per_day(granularity: str) -> int:
    return int(pd.Timedelta(days=1) / granularity_delta(granularity))


def forecast_origin_end_datetime(
    end_datetime: pd.Timestamp, granularity: str
) -> pd.Timestamp:
    """Return the final origin whose t+2 realization fits in the datetime window."""

    return end_datetime - 2 * granularity_delta(granularity)


__all__ = [
    "forecast_origin_end_datetime",
    "granularity_delta",
    "periods_per_day",
]
