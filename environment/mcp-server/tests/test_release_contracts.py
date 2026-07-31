from __future__ import annotations

import asyncio
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evalenv_shared.worker.socket_server import _socket_is_live, serve
from feature_engineering.config import load_task_config
from feature_engineering.core.backtest import _model_visible_metrics
from feature_engineering.core.ema_smoothed_engine import EmaSmoothedPortfolioEngine
from feature_engineering.core.portfolio import calculate_median_signal_size
from feature_engineering.submissions.strategy import (
    StrategyError,
    compile_model_strategy,
)
from feature_engineering.submissions.worker import _validate_estimator

TASK_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = TASK_ROOT / "task_config.yaml"
MANIFEST_PATH = TASK_ROOT / "data_manifest.json"


class ConfigurationContractTests(unittest.TestCase):
    def test_yaml_values_are_the_runtime_authority(self) -> None:
        payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        payload["agent"]["max_research_attempts"] = 7
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "task_config.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            config = load_task_config(path)
        self.assertEqual(config.agent.max_research_attempts, 7)

    def test_unknown_config_fields_are_rejected(self) -> None:
        payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        payload["unexpected"] = True
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "task_config.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected"):
                load_task_config(path)

    def test_small_mse_remains_visible(self) -> None:
        config = load_task_config(CONFIG_PATH)
        visible = _model_visible_metrics(config, {"mse": 2.75e-6})
        self.assertEqual(visible["mse"], 2.75e-6)


class EstimatorFamilyContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.X = pd.DataFrame(
            {
                "open": np.linspace(1.0, 2.0, 12),
                "close": np.linspace(1.1, 2.2, 12),
            }
        )
        self.y = np.linspace(-0.02, 0.03, len(self.X))

    def _fit(self, estimator):
        return estimator.fit(self.X, self.y)

    def _assert_rejected(self, estimator) -> None:
        result = _validate_estimator(self._fit(estimator))
        self.assertIsInstance(result, dict)
        self.assertEqual(result["error_code"], "invalid_fitted_estimator")
        self.assertIn("exact sklearn.linear_model.ElasticNet", result["message"])

    def test_direct_exact_elastic_net_is_accepted(self) -> None:
        result = _validate_estimator(self._fit(ElasticNet(alpha=0.01)))
        self.assertEqual(result, ("open", "close"))

    def test_pipeline_ending_in_exact_elastic_net_is_accepted(self) -> None:
        model = make_pipeline(StandardScaler(), ElasticNet(alpha=0.01))
        result = _validate_estimator(self._fit(model))
        self.assertEqual(result, ("open", "close"))

    def test_random_forest_is_rejected(self) -> None:
        self._assert_rejected(RandomForestRegressor(n_estimators=2, random_state=0))

    def test_pipeline_ending_in_another_estimator_is_rejected(self) -> None:
        self._assert_rejected(make_pipeline(StandardScaler(), Ridge(alpha=0.01)))

    def test_elastic_net_subclass_is_rejected(self) -> None:
        class ElasticNetSubclass(ElasticNet):
            pass

        self._assert_rejected(ElasticNetSubclass(alpha=0.01))


class ManifestContractTests(unittest.TestCase):
    def test_split_endpoints_name_each_forecast_stage(self) -> None:
        payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        for split in ("public", "hidden"):
            origin = pd.Timestamp(payload[f"{split}_last_forecast_origin_datetime"])
            execution = pd.Timestamp(payload[f"{split}_last_execution_datetime"])
            realization = pd.Timestamp(payload[f"{split}_last_realization_datetime"])
            self.assertEqual(execution, origin + pd.Timedelta(minutes=1))
            self.assertEqual(realization, origin + pd.Timedelta(minutes=2))
            self.assertNotIn(f"{split}_end_datetime", payload)


class StrategyContractTests(unittest.TestCase):
    def test_zero_max_gross_exposure_is_rejected(self) -> None:
        result = compile_model_strategy(
            model_id="model-1",
            max_gross_exposure=0.0,
        )

        self.assertIsInstance(result, StrategyError)
        self.assertEqual(result.error_code, "invalid_max_gross_exposure")
        self.assertIn("positive", result.message)


class PortfolioScalingTests(unittest.TestCase):
    def test_training_median_uses_beta_neutral_cross_sections(self) -> None:
        forecasts = np.asarray([[1.0, 2.0], [5.0, 1.0]])
        betas = np.asarray([[1.0, 0.0], [1.0, 1.0]])

        self.assertEqual(calculate_median_signal_size(forecasts, betas), 3.0)

    def test_engine_uses_frozen_training_median(self) -> None:
        config = load_task_config(CONFIG_PATH)
        engine = EmaSmoothedPortfolioEngine(
            config=config.backtest,
            symbol_count=4,
            median_signal_size=0.25,
            fee_rate=0.0001,
            max_gross_exposure=0.5,
        )

        self.assertEqual(engine.median_signal_size, 0.25)
        self.assertEqual(engine.quadratic_penalty, 0.5)


class SocketLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def _wait_until_live(self, path: Path) -> None:
        for _ in range(100):
            if path.exists() and await _socket_is_live(path):
                return
            await asyncio.sleep(0.01)
        self.fail("Worker socket did not become live.")

    async def test_stale_socket_is_replaced_and_cleaned_up(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            runtime = root / "runtime"
            runtime.mkdir()
            socket_path = root / "worker.sock"
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stale:
                stale.bind(str(socket_path))
            environment = {
                "FEATURE_WORKER_ROOT": str(runtime),
                "FEATURE_WORKER_SOCKET": str(socket_path),
            }
            with patch.dict(os.environ, environment):
                task = asyncio.create_task(serve())
                await self._wait_until_live(socket_path)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            self.assertFalse(socket_path.exists())

    async def test_live_socket_is_not_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            runtime = root / "runtime"
            runtime.mkdir()
            socket_path = root / "worker.sock"
            environment = {
                "FEATURE_WORKER_ROOT": str(runtime),
                "FEATURE_WORKER_SOCKET": str(socket_path),
            }
            with patch.dict(os.environ, environment):
                task = asyncio.create_task(serve())
                await self._wait_until_live(socket_path)
                with self.assertRaisesRegex(RuntimeError, "already serving"):
                    await serve()
                self.assertTrue(await _socket_is_live(socket_path))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task


if __name__ == "__main__":
    unittest.main()
