from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from behavior_helpers import PAST_ONLY_MODEL_CODE, build_synthetic_workspace

from evalenv_shared.worker.process import SubprocessWorkerHost
from feature_engineering.core import backtest as backtest_module
from feature_engineering.evaluation.verifier import verify_submission
from feature_engineering.runtime.protocol import (
    BacktestRequest,
    SubmitStrategyRequest,
    TrainModelRequest,
)
from feature_engineering.runtime.task import FeatureEngineeringRuntime
from feature_engineering.submissions import modeling as modeling_module
from feature_engineering.submissions.registry import TrainedModelRegistry


class EndToEndBehaviorTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_training_submission_and_verification_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = build_synthetic_workspace(Path(temporary_directory))
            reward_path = Path(temporary_directory) / "verifier" / "reward.json"
            metrics_path = Path(temporary_directory) / "verifier" / "metrics.json"
            main_thread_id = threading.get_ident()
            training_preparation_threads: list[int] = []
            backtest_preparation_threads: list[int] = []
            original_training_preparation = modeling_module._prepare_training_payloads
            original_backtest_preparation = backtest_module._prepare_backtest_payload

            def observe_training_preparation(*args, **kwargs):
                training_preparation_threads.append(threading.get_ident())
                return original_training_preparation(*args, **kwargs)

            def observe_backtest_preparation(*args, **kwargs):
                backtest_preparation_threads.append(threading.get_ident())
                return original_backtest_preparation(*args, **kwargs)

            training_preparation_patch = patch.object(
                modeling_module,
                "_prepare_training_payloads",
                side_effect=observe_training_preparation,
            )
            backtest_preparation_patch = patch.object(
                backtest_module,
                "_prepare_backtest_payload",
                side_effect=observe_backtest_preparation,
            )
            training_preparation_patch.start()
            backtest_preparation_patch.start()
            self.addCleanup(training_preparation_patch.stop)
            self.addCleanup(backtest_preparation_patch.stop)

            with TrainedModelRegistry(workspace.runtime_root) as registry:
                runtime = FeatureEngineeringRuntime.from_config_path(
                    workspace.config_path,
                    registry=registry,
                    worker_host=SubprocessWorkerHost(),
                    task_outputs=workspace.submission_root,
                )

                training_started = runtime.start_training(
                    TrainModelRequest(
                        model_code=PAST_ONLY_MODEL_CODE,
                        label="deterministic-model",
                    )
                )
                self.assertEqual(training_started["training_id"], "training_001")
                self.assertEqual(training_started["status"], "running")
                self.assertEqual(
                    runtime.get_train_model_result("training_001"),
                    {
                        "type": "training_status",
                        "ok": True,
                        "training_id": "training_001",
                        "status": "running",
                    },
                )

                await runtime.execute_training("training_001")
                training_result = runtime.get_train_model_result("training_001")
                self.assertEqual(training_result["type"], "training_result")
                self.assertEqual(training_result["training_id"], "training_001")
                self.assertTrue(training_result["ok"])
                self.assertEqual(training_result["status"], "succeeded")
                self.assertEqual(training_result["model_id"], "model_001")
                self.assertEqual(registry.get("model_001").model_id, "model_001")

                backtest_started = runtime.start_backtest(
                    BacktestRequest(
                        model_id="model_001",
                        max_gross_exposure=0.25,
                        label="deterministic-strategy",
                    )
                )
                self.assertEqual(backtest_started["backtest_id"], "backtest_001")
                self.assertEqual(backtest_started["status"], "running")
                self.assertEqual(
                    runtime.get_backtest_result("backtest_001"),
                    {
                        "type": "backtest_status",
                        "ok": True,
                        "backtest_id": "backtest_001",
                        "status": "running",
                    },
                )

                await runtime.execute_backtest("backtest_001")
                backtest_result = runtime.get_backtest_result("backtest_001")
                self.assertEqual(backtest_result["type"], "backtest_result")
                self.assertEqual(backtest_result["backtest_id"], "backtest_001")
                self.assertTrue(backtest_result["ok"])
                self.assertEqual(backtest_result["status"], "succeeded")
                self.assertEqual(backtest_result["strategy_id"], "strategy_001")
                self.assertIn("strategy_001", runtime.strategies)

                submission_result = await runtime.submit_strategy(
                    SubmitStrategyRequest(
                        strategy_id="strategy_001",
                        rationale="deterministic end-to-end exercise",
                    )
                )
                self.assertEqual(
                    submission_result,
                    {
                        "type": "final_submission_result",
                        "ok": True,
                        "action": "submit_strategy",
                        "strategy_id": "strategy_001",
                        "accepted": True,
                        "done": True,
                    },
                )

            bundle_manifests = sorted(
                (
                    workspace.submission_root / "feature_engineering" / "submissions"
                ).glob("*/manifest.json")
            )
            self.assertEqual(len(bundle_manifests), 1)

            with (
                patch.dict(
                    os.environ,
                    {"FEATURE_VERIFIER_RUNTIME_ROOT": str(workspace.runtime_root)},
                ),
                patch(
                    "evalenv_shared.worker.socket_host.UnixSocketWorkerHost",
                    return_value=SubprocessWorkerHost(),
                ) as worker_host_constructor,
            ):
                rewards = await verify_submission(
                    workspace.config_path,
                    task_root=Path(temporary_directory),
                    submission_root=workspace.submission_root,
                    reward_path=reward_path,
                    metrics_path=metrics_path,
                )

            worker_host_constructor.assert_called_once()
            self.assertEqual(
                set(rewards),
                {
                    "primary_score",
                    "reward",
                    "sharpe",
                    "cagr",
                    "max_drawdown",
                    "pearson_ic",
                },
            )
            for value in rewards.values():
                self.assertIs(type(value), float)
                self.assertTrue(math.isfinite(value))
            self.assertEqual(rewards["reward"], rewards["sharpe"])
            self.assertEqual(rewards["primary_score"], rewards["sharpe"])

            self.assertTrue(reward_path.is_file())
            self.assertTrue(metrics_path.is_file())
            self.assertEqual(
                json.loads(reward_path.read_text(encoding="utf-8")), rewards
            )
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertTrue(metrics["ok"])
            self.assertEqual(metrics["primary_score"], rewards["primary_score"])
            self.assertEqual(
                metrics["metrics"]["annualized_sharpe"],
                rewards["sharpe"],
            )
            self.assertIn("fit_diagnostics", metrics)
            self.assertNotEqual(metrics, rewards)
            self.assertTrue(training_preparation_threads)
            self.assertTrue(backtest_preparation_threads)
            self.assertTrue(
                all(
                    thread_id != main_thread_id
                    for thread_id in training_preparation_threads
                )
            )
            self.assertTrue(
                all(
                    thread_id != main_thread_id
                    for thread_id in backtest_preparation_threads
                )
            )


if __name__ == "__main__":
    unittest.main()
