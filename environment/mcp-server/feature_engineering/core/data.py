"""Load the packaged public task data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from feature_engineering.config import TaskConfig
from feature_engineering.core.granularity import forecast_origin_end_datetime


@dataclass(frozen=True, slots=True)
class SupervisedData:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    datetimes: pd.Index
    symbols: tuple[str, ...]
    start_datetime: pd.Timestamp
    end_datetime: pd.Timestamp


def load_supervised_data(config: TaskConfig) -> SupervisedData:
    manifest_path = Path(config.data.manifest_path)
    manifest = _read_manifest(manifest_path)
    public_path = (manifest_path.parent / manifest["public_path"]).resolve()
    frame = pd.read_parquet(public_path)
    return normalize_supervised_frame(
        config,
        frame,
        start_datetime=pd.Timestamp(manifest["public_first_forecast_origin_datetime"]),
        end_datetime=pd.Timestamp(manifest["public_last_realization_datetime"]),
    )


def normalize_supervised_frame(
    config: TaskConfig,
    frame: pd.DataFrame,
    *,
    start_datetime: pd.Timestamp,
    end_datetime: pd.Timestamp,
) -> SupervisedData:
    data = config.data
    required = {
        *data.index_columns,
        *data.features,
        *data.targets,
        *data.scoring_columns,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Supervised data missing column(s): {', '.join(missing)}")
    normalized = frame.copy()
    normalized[data.datetime_column] = pd.to_datetime(
        normalized[data.datetime_column], utc=True, errors="raise"
    )
    start_datetime = pd.Timestamp(start_datetime).tz_convert("UTC")
    end_datetime = pd.Timestamp(end_datetime).tz_convert("UTC")
    origin_end = forecast_origin_end_datetime(end_datetime, data.granularity)
    normalized = normalized[
        normalized[data.symbol_column].astype(str).str.fullmatch(r"symbol_\d{2}")
        & (normalized[data.datetime_column] >= start_datetime)
        & (normalized[data.datetime_column] <= origin_end)
    ].sort_values(list(data.index_columns))
    if normalized.duplicated(list(data.index_columns)).any():
        raise ValueError("Supervised data contains duplicate index keys.")
    symbols = tuple(sorted(normalized[data.symbol_column].astype(str).unique()))
    if not symbols or symbols != tuple(
        f"symbol_{i:02d}" for i in range(1, len(symbols) + 1)
    ):
        raise ValueError(
            "Supervised data symbols must be the complete symbol_NN sequence."
        )
    for column in dict.fromkeys((*data.features, *data.targets, *data.scoring_columns)):
        normalized[column] = normalized[column].astype(float)
    return SupervisedData(
        frame=normalized,
        feature_columns=tuple(data.features),
        target_columns=tuple(data.targets),
        datetimes=pd.Index(normalized[data.datetime_column].drop_duplicates()),
        symbols=symbols,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )


def _read_manifest(path: Path) -> dict[str, str]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Data manifest must be a JSON object.")
    return payload


__all__ = ["SupervisedData", "load_supervised_data", "normalize_supervised_frame"]
