"""Canonical pandas frames and private file-backed worker Arrow IPC."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from feature_engineering.config import TaskConfig
from feature_engineering.core.fixed_data import SupervisedData
from feature_engineering.core.granularity import forecast_origin_end_datetime

MAX_DATAFRAME_ARROW_BYTES = 2 * 1024 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def build_training_frames(
    *,
    config: TaskConfig,
    public_data: SupervisedData,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the exact submitted-code X/y contract from canonical task data."""
    data = config.data
    feature_names = tuple(public_data.feature_columns)
    target_names = tuple(data.targets)
    columns = (data.datetime_column, data.symbol_column, *feature_names)
    if len(columns) != len(set(columns)):
        raise ValueError("Model identity and feature column names must be unique.")
    if len(target_names) != len(set(target_names)):
        raise ValueError("Model target column names must be unique.")

    lower = public_data.start_datetime if start is None else pd.Timestamp(start)
    upper = forecast_origin_end_datetime(
        public_data.end_datetime if end is None else pd.Timestamp(end),
        data.granularity,
    )
    frame = public_data.frame.loc[
        (public_data.frame[data.datetime_column] >= lower)
        & (public_data.frame[data.datetime_column] <= upper)
        & public_data.frame[data.symbol_column].isin(public_data.symbols)
    ].copy()
    if frame.empty:
        raise ValueError("No training rows are visible in the selected public window.")

    target_values = frame.loc[:, list(target_names)].astype("float64")
    valid_targets = np.isfinite(target_values.to_numpy(dtype="float64")).all(axis=1)
    frame = frame.loc[valid_targets].copy()
    if frame.empty:
        raise ValueError("No training rows have complete finite targets.")

    datetimes = pd.to_datetime(frame[data.datetime_column], utc=True)
    symbols = frame[data.symbol_column].map(str).astype(object)
    index = pd.MultiIndex.from_arrays(
        [datetimes, symbols],
        names=[data.datetime_column, data.symbol_column],
    )
    if not index.is_unique:
        raise ValueError(
            "Training frame index must contain unique datetime/symbol keys."
        )

    X = pd.DataFrame(index=index)
    X[data.datetime_column] = pd.Series(datetimes.array, index=index)
    X[data.symbol_column] = pd.Series(
        symbols.to_numpy(dtype=object), index=index, dtype=object
    )
    for name in feature_names:
        X[name] = pd.Series(
            frame[name].to_numpy(dtype="float64"),
            index=index,
            dtype="float64",
        )

    y = pd.DataFrame(index=index)
    for name in target_names:
        y[name] = pd.Series(
            frame[name].to_numpy(dtype="float64"),
            index=index,
            dtype="float64",
        )
    return X, y


def build_prediction_frame(
    *,
    config: TaskConfig,
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """Build canonical X for a public or hidden prediction batch."""
    data = config.data
    feature_names = tuple(data.features)
    columns = (data.datetime_column, data.symbol_column, *feature_names)
    if len(columns) != len(set(columns)):
        raise ValueError("Model identity and feature column names must be unique.")
    missing = [name for name in columns if name not in rows.columns]
    if missing:
        raise ValueError(f"Prediction data missing column(s): {', '.join(missing)}")

    datetimes = pd.to_datetime(rows[data.datetime_column], utc=True)
    symbols = rows[data.symbol_column].map(str).astype(object)
    index = pd.MultiIndex.from_arrays(
        [datetimes, symbols],
        names=[data.datetime_column, data.symbol_column],
    )
    if not index.is_unique:
        raise ValueError(
            "Prediction frame index must contain unique datetime/symbol keys."
        )

    X = pd.DataFrame(index=index)
    X[data.datetime_column] = pd.Series(datetimes.array, index=index)
    X[data.symbol_column] = pd.Series(
        symbols.to_numpy(dtype=object), index=index, dtype=object
    )
    for name in feature_names:
        X[name] = pd.Series(
            rows[name].to_numpy(dtype="float64"),
            index=index,
            dtype="float64",
        )
    return X


def write_dataframe(path: str | Path, frame: pd.DataFrame) -> dict[str, Any]:
    """Atomically write one pandas DataFrame as Arrow IPC."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Worker DataFrame attachment requires a pandas DataFrame.")
    destination = Path(path)
    descriptor_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.pending-",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor_fd, "wb") as handle:
            table = pa.Table.from_pandas(frame, preserve_index=True, safe=True)
            with ipc.new_file(handle, table.schema) as writer:
                writer.write_table(table)
            handle.flush()
            os.fsync(handle.fileno())
        byte_count = _regular_file_stat(temporary).st_size
        if byte_count > MAX_DATAFRAME_ARROW_BYTES:
            raise ValueError("Worker DataFrame Arrow file exceeds the allowed size.")
        sha256 = _sha256_file(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "name": destination.name,
        "byte_count": byte_count,
        "sha256": sha256,
        "row_count": len(frame),
    }


def read_dataframe(
    directory: str | Path,
    payload: Any,
    *,
    expected_rows: int | None = None,
    expected_columns: int | None = None,
    max_bytes: int | None = None,
) -> pd.DataFrame:
    """Read one hash-verified Arrow IPC DataFrame from a private directory."""
    descriptor = _dataframe_descriptor(payload)
    if expected_rows is not None and descriptor["row_count"] != expected_rows:
        raise ValueError("Worker DataFrame row count does not match the expected rows.")
    if max_bytes is not None and descriptor["byte_count"] > max_bytes:
        raise ValueError(
            "Worker DataFrame Arrow file exceeds the operation-specific limit."
        )
    source = Path(directory) / descriptor["name"]
    source_stat = _regular_file_stat(source)
    if source_stat.st_size != descriptor["byte_count"]:
        raise ValueError("Worker DataFrame byte count does not match its descriptor.")
    if _sha256_file(source) != descriptor["sha256"]:
        raise ValueError("Worker DataFrame failed hash verification.")
    with source.open("rb") as handle:
        with ipc.open_file(handle) as reader:
            table = reader.read_all()
    if table.num_rows != descriptor["row_count"]:
        raise ValueError("Worker DataFrame row count does not match its descriptor.")
    if expected_columns is not None and table.num_columns != expected_columns:
        raise ValueError(
            "Worker DataFrame column count does not match the expected schema."
        )
    frame = table.to_pandas()
    _restore_object_dtypes(frame, table.schema.pandas_metadata or {})
    return frame


def canonical_prediction_frame(
    result: Any,
    *,
    X: pd.DataFrame,
    target_names: list[str],
) -> pd.DataFrame:
    try:
        values = np.asarray(result, dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ValueError("Estimator predictions must be numeric.") from exc
    if values.ndim == 1 and len(target_names) == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape != (len(X), len(target_names)):
        raise ValueError("Estimator predictions have the wrong shape.")
    if not np.isfinite(values).all():
        raise ValueError("Estimator predictions must all be finite.")
    return pd.DataFrame(values, index=X.index, columns=target_names, dtype="float64")


def _dataframe_descriptor(payload: Any) -> dict[str, Any]:
    keys = {"name", "byte_count", "sha256", "row_count"}
    if not isinstance(payload, dict) or set(payload) != keys:
        raise ValueError("Worker DataFrame descriptor has an invalid schema.")
    name = payload["name"]
    byte_count = payload["byte_count"]
    sha256 = payload["sha256"]
    row_count = payload["row_count"]
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("Worker DataFrame name must be a private local filename.")
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 0
        or byte_count > MAX_DATAFRAME_ARROW_BYTES
    ):
        raise ValueError("Worker DataFrame byte count is invalid.")
    if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
        raise ValueError("Worker DataFrame sha256 is invalid.")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0:
        raise ValueError("Worker DataFrame row count is invalid.")
    return payload


def _restore_object_dtypes(frame: pd.DataFrame, metadata: dict[str, Any]) -> None:
    for column in metadata["columns"]:
        name = column.get("name")
        if column.get("numpy_type") == "object" and name in frame.columns:
            frame[name] = frame[name].astype(object)
    if isinstance(frame.index, pd.MultiIndex):
        levels = list(frame.index.levels)
        levels[1] = levels[1].astype(object)
        frame.index = frame.index.set_levels(levels)


def _regular_file_stat(path: Path) -> os.stat_result:
    source_stat = path.lstat()
    if not stat.S_ISREG(source_stat.st_mode):
        raise ValueError("Worker DataFrame must be a regular file.")
    return source_stat


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "build_prediction_frame",
    "build_training_frames",
    "canonical_prediction_frame",
    "read_dataframe",
    "write_dataframe",
]
