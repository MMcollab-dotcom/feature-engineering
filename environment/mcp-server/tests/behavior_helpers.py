from __future__ import annotations

import hashlib
import json
import platform
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml

from feature_engineering.config import TaskConfig, load_task_config
from feature_engineering.core.data import SupervisedData, load_supervised_data
from feature_engineering.submissions.registry import StoredModel, TrainedModelRegistry
from feature_engineering.submissions.validation import submitted_module_name

FEATURE_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "weight_std_dollar_vol",
)
TARGET_COLUMNS = ("target_horizon_1",)
SYMBOLS = tuple(f"symbol_{number:02d}" for number in range(1, 5))

PAST_ONLY_MODEL_CODE = """\
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline


class PastOnlyFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        return X.loc[:, ["open", "close"]].to_numpy(dtype=float)


def train_model(X, y):
    model = Pipeline(
        [
            ("features", PastOnlyFeatures()),
            (
                "elasticnet",
                ElasticNet(alpha=1.0e-6, l1_ratio=0.5, max_iter=10000),
            ),
        ]
    )
    return model.fit(X, np.asarray(y).reshape(-1))
"""

FUTURE_DEPENDENT_MODEL_CODE = """\
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline


class FutureDependentFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        values = X.loc[:, ["open"]].to_numpy(dtype=float)
        return values - values.mean(axis=0)


def train_model(X, y):
    model = Pipeline(
        [
            ("features", FutureDependentFeatures()),
            (
                "elasticnet",
                ElasticNet(alpha=1.0e-6, l1_ratio=0.5, max_iter=10000),
            ),
        ]
    )
    return model.fit(X, np.asarray(y).reshape(-1))
"""


@dataclass(frozen=True, slots=True)
class SyntheticWorkspace:
    config_path: Path
    submission_root: Path
    runtime_root: Path
    config: TaskConfig
    public_data: SupervisedData


def build_synthetic_workspace(
    root: Path,
    *,
    research_attempts: int = 3,
    response_error_budget: int = 3,
    public_steps: int = 24,
    hidden_steps: int = 16,
) -> SyntheticWorkspace:
    """Create a complete, deterministic fixed-split workspace for behavior tests."""

    root = Path(root).resolve()
    if public_steps < 7 or hidden_steps < 7:
        raise ValueError("Synthetic splits need at least five forecast origins.")
    root.mkdir(parents=True, exist_ok=True)
    public_path = root / "data" / "runtime_public.parquet"
    hidden_path = root / "tests" / "hidden_data" / "hidden.parquet"
    submission_root = root / "task_outputs"
    runtime_root = root / "runtime"
    for directory in (
        public_path.parent,
        hidden_path.parent,
        submission_root,
        runtime_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    public_start = pd.Timestamp("2024-01-01T00:00:00Z")
    hidden_start = public_start + pd.Timedelta(minutes=public_steps)
    public_frame = _minute_panel(public_start, public_steps)
    hidden_frame = _minute_panel(hidden_start, hidden_steps)
    public_frame.to_parquet(public_path, index=False)
    hidden_frame.to_parquet(hidden_path, index=False)

    manifest = {
        "schema_version": "feature_engineering_fixed_split_v2",
        "license": "Synthetic test data",
        "source_type": "deterministic synthetic minute supervised-learning panel",
        "source_location": "generated in a temporary test workspace",
        "sha256": {
            "public_runtime": _sha256_file(public_path),
            "hidden_verifier": _sha256_file(hidden_path),
        },
        "public_path": public_path.relative_to(root).as_posix(),
        "hidden_path": hidden_path.relative_to(root).as_posix(),
        **_split_endpoints("public", public_start, public_steps),
        **_split_endpoints("hidden", hidden_start, hidden_steps),
        "symbols": list(SYMBOLS),
        "source_symbols": ["SYNTH_A", "SYNTH_B", "SYNTH_C", "SYNTH_D"],
        "features": list(FEATURE_COLUMNS),
        "target": TARGET_COLUMNS[0],
        "scoring_columns": ["tradable_return", "beta_10d_fwd_1"],
    }
    (root / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    config_path = root / "task_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            _config_payload(
                research_attempts=research_attempts,
                response_error_budget=response_error_budget,
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config = load_task_config(config_path)
    public_data = load_supervised_data(config)
    return SyntheticWorkspace(
        config_path=config_path,
        submission_root=submission_root,
        runtime_root=runtime_root,
        config=config,
        public_data=public_data,
    )


def register_dummy_model(
    registry: TrainedModelRegistry,
    *,
    model_code: str = PAST_ONLY_MODEL_CODE,
    forecast_scale: float = 1.0,
    median_signal_size: float = 1.0,
) -> StoredModel:
    """Commit a tiny fitted estimator whose artifact matches submitted source."""

    X = pd.DataFrame(
        {
            name: np.asarray([row + column / 10.0 for row in range(8)], dtype="float64")
            for column, name in enumerate(FEATURE_COLUMNS)
        }
    )
    y = pd.DataFrame(
        {TARGET_COLUMNS[0]: np.linspace(-0.02, 0.03, len(X), dtype="float64")}
    )
    source_hash = hashlib.sha256(model_code.encode("utf-8")).hexdigest()
    module_name = submitted_module_name(source_hash)
    module = types.ModuleType(module_name)
    module.__dict__["__builtins__"] = __builtins__
    sys.modules[module_name] = module
    exec(compile(model_code, "<synthetic_feature_model>", "exec"), module.__dict__)
    estimator = module.train_model(X, y)

    with registry.staging_directory() as staging_directory:
        artifact_path = staging_directory / "model.joblib"
        joblib.dump(estimator, artifact_path, compress=0, protocol=5)
        return registry.register(
            artifact_path,
            model_code=model_code,
            model_code_sha256=source_hash,
            inference_columns=FEATURE_COLUMNS,
            target_names=TARGET_COLUMNS,
            package_versions={
                "python": platform.python_version(),
                "sklearn": sklearn.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "joblib": joblib.__version__,
            },
            training_filter={
                "start_datetime": "2024-01-01T00:00:00+00:00",
                "end_datetime": "2024-01-01T00:05:00+00:00",
            },
            training_row_count=len(X),
            forecast_scale=forecast_scale,
            median_signal_size=median_signal_size,
        )


def _config_payload(
    *, research_attempts: int, response_error_budget: int
) -> dict[str, object]:
    return {
        "workspace": {"profile": "feature_engineering"},
        "data": {
            "granularity": "minutely",
            "datetime_column": "datetime",
            "symbol_column": "symbol",
            "index_columns": ["datetime", "symbol"],
            "features": list(FEATURE_COLUMNS),
            "targets": list(TARGET_COLUMNS),
            "scoring": {
                "tradable_return": "tradable_return",
                "market_beta": "beta_10d_fwd_1",
            },
        },
        "backtest": {
            "engine": "ema_smoothed",
            "rebalance_freq": 1,
            "portfolio_ema_hl_steps": 7,
            "portfolio_ema_tail_hl_steps": 4,
            "portfolio_ema_switch_steps": 7,
            "target_norm_weight": 0.1,
            "model_visible_metric_rounding": {
                "sharpe_decimals": 2,
                "return_decimals": 4,
                "rate_decimals": 3,
                "error_decimals": 5,
                "mse_decimals": 8,
                "correlation_decimals": 3,
            },
        },
        "agent": {
            "max_research_attempts": research_attempts,
            "response_error_budget": response_error_budget,
        },
        "prediction": {
            "allowed_model_packages": [
                "math",
                "statistics",
                "numpy",
                "pandas",
                "sklearn",
            ],
            "max_model_code_bytes": 20000,
        },
        "execution": {"initial_capital": 100000.0, "timeout_seconds": 30.0},
        "costs": {"linear_fee_bps": 1.0},
        "reward": {
            "periods_per_year": 525600,
        },
    }


def _minute_panel(start: pd.Timestamp, steps: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for step in range(steps):
        timestamp = start + pd.Timedelta(minutes=step)
        for symbol_number, symbol in enumerate(SYMBOLS, start=1):
            open_price = _open_price(step, symbol_number)
            close_price = _close_price(step, symbol_number)
            execution_price = _close_price(step + 1, symbol_number)
            realization_price = _close_price(step + 2, symbol_number)
            volume = 1000.0 + 17.0 * step + 11.0 * symbol_number
            taker_fraction = 0.42 + 0.01 * ((step + symbol_number) % 5)
            rows.append(
                {
                    "datetime": timestamp,
                    "symbol": symbol,
                    "open": open_price,
                    "high": max(open_price, close_price) + 0.08,
                    "low": min(open_price, close_price) - 0.08,
                    "close": close_price,
                    "volume": volume,
                    "quote_asset_volume": volume * close_price,
                    "number_of_trades": 50.0 + 2.0 * step + symbol_number,
                    "taker_buy_base_asset_volume": volume * taker_fraction,
                    "taker_buy_quote_asset_volume": (
                        volume * taker_fraction * close_price
                    ),
                    "weight_std_dollar_vol": (
                        0.5
                        + 0.03 * step
                        + 0.02 * symbol_number
                        + 0.005 * ((step * symbol_number) % 3)
                    ),
                    "target_horizon_1": realization_price / execution_price - 1.0,
                    "tradable_return": realization_price / execution_price - 1.0,
                    "beta_10d_fwd_1": (0.65 + 0.08 * symbol_number + 0.01 * (step % 4)),
                }
            )
    return pd.DataFrame(rows)


def _open_price(step: int, symbol_number: int) -> float:
    return (
        100.0 + 3.0 * symbol_number + 0.2 * step + 0.03 * ((step + symbol_number) % 4)
    )


def _close_price(step: int, symbol_number: int) -> float:
    return _open_price(step, symbol_number) + 0.015 * (
        (step + 2 * symbol_number) % 5 - 2
    )


def _split_endpoints(prefix: str, start: pd.Timestamp, steps: int) -> dict[str, str]:
    last_origin = start + pd.Timedelta(minutes=steps - 3)
    return {
        f"{prefix}_first_forecast_origin_datetime": start.isoformat(),
        f"{prefix}_last_forecast_origin_datetime": last_origin.isoformat(),
        f"{prefix}_last_execution_datetime": (
            last_origin + pd.Timedelta(minutes=1)
        ).isoformat(),
        f"{prefix}_last_realization_datetime": (
            last_origin + pd.Timedelta(minutes=2)
        ).isoformat(),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "FUTURE_DEPENDENT_MODEL_CODE",
    "PAST_ONLY_MODEL_CODE",
    "SyntheticWorkspace",
    "build_synthetic_workspace",
    "register_dummy_model",
]
