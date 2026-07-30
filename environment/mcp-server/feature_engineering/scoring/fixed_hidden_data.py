"""Load the verifier-only hidden split from the fixed task bundle."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from feature_engineering.config import TaskConfig
from feature_engineering.core.fixed_data import SupervisedData
from feature_engineering.submissions.causal_audit import validate_fixed_prediction_window


def load_hidden_supervised_data(
    *,
    public_config: TaskConfig,
    public_data: SupervisedData,
    scoring_config,
) -> SupervisedData:
    del scoring_config
    manifest_path = Path(public_config.data.manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    hidden_path = (manifest_path.parent / payload["hidden_path"]).resolve()
    hidden = pd.read_parquet(hidden_path)
    data = public_config.data
    required = {*data.index_columns, *data.features, *data.targets, *data.scoring_columns}
    missing = sorted(required - set(hidden.columns))
    if missing:
        raise ValueError(f"Hidden data missing column(s): {', '.join(missing)}")
    hidden[data.datetime_column] = pd.to_datetime(
        hidden[data.datetime_column], utc=True, errors="raise"
    )
    hidden = hidden.sort_values(list(data.index_columns))
    for column in dict.fromkeys((*data.targets, *data.scoring_columns)):
        hidden[column] = hidden[column].astype(float)
    hidden_data = SupervisedData(
        frame=hidden,
        feature_columns=public_data.feature_columns,
        target_columns=public_data.target_columns,
        datetimes=pd.Index(hidden[data.datetime_column].drop_duplicates()),
        symbols=public_data.symbols,
        start_datetime=pd.Timestamp(payload["hidden_start_datetime"]),
        end_datetime=pd.Timestamp(payload["hidden_end_datetime"]),
        manifest_sha256=public_data.manifest_sha256,
    )
    validate_fixed_prediction_window(
        config=public_config,
        data=hidden_data,
        start=hidden_data.start_datetime,
        end=hidden_data.end_datetime,
    )
    return hidden_data


__all__ = ["load_hidden_supervised_data"]
