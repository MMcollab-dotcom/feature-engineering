from __future__ import annotations

import asyncio
import copy
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from behavior_helpers import (
    PAST_ONLY_MODEL_CODE,
    SyntheticWorkspace,
    build_synthetic_workspace,
    register_dummy_model,
)

from feature_engineering.core.backtest import BacktestResult
from feature_engineering.runtime.protocol import (
    BacktestRequest,
    DatetimeFilter,
    TrainModelRequest,
)
from feature_engineering.runtime.task import (
    FeatureEngineeringRuntime,
    OperationInfrastructureError,
)
from feature_engineering.submissions.modeling import TrainingResult
from feature_engineering.submissions.registry import (
    StoredModel,
    TrainedModelRegistry,
)


class RuntimeBehaviorTestCase(unittest.IsolatedAsyncioTestCase):
    def make_workspace(
        self,
        *,
        research_attempts: int = 3,
        response_error_budget: int = 3,
    ) -> SyntheticWorkspace:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        return build_synthetic_workspace(
            Path(temporary_directory.name),
            research_attempts=research_attempts,
            response_error_budget=response_error_budget,
        )

    def make_runtime(
        self, workspace: SyntheticWorkspace
    ) -> tuple[FeatureEngineeringRuntime, StoredModel]:
        registry = TrainedModelRegistry(workspace.runtime_root)
        self.addCleanup(registry.close)
        model = register_dummy_model(registry)
        runtime = FeatureEngineeringRuntime.from_config_path(
            workspace.config_path,
            registry=registry,
            worker_host=object(),
            task_outputs=workspace.submission_root,
        )
        return runtime, model

    @staticmethod
    def training_result(model: StoredModel) -> TrainingResult:
        return TrainingResult(
            model=model,
            diagnostics={"mse": 0.0001, "forecast_scale": model.forecast_scale},
            row_count=32,
            feature_names=model.inference_columns,
            target_names=model.target_names,
        )

    async def complete_training(
        self,
        runtime: FeatureEngineeringRuntime,
        model: StoredModel,
    ) -> str:
        started = runtime.start_training(TrainModelRequest(PAST_ONLY_MODEL_CODE))
        training_id = started["training_id"]

        async def succeed(**_kwargs: object) -> TrainingResult:
            return self.training_result(model)

        with patch(
            "feature_engineering.runtime.task.evaluate_model_code_async",
            new=succeed,
        ):
            await runtime.execute_training(training_id)
        return training_id


class AsyncOperationLifecycleTests(RuntimeBehaviorTestCase):
    async def test_training_running_and_terminal_queries_are_repeatable(self) -> None:
        workspace = self.make_workspace()
        runtime, model = self.make_runtime(workspace)
        boundary_started = asyncio.Event()
        boundary_release = asyncio.Event()

        async def controlled_boundary(**_kwargs: object) -> TrainingResult:
            boundary_started.set()
            await boundary_release.wait()
            return self.training_result(model)

        started = runtime.start_training(
            TrainModelRequest(PAST_ONLY_MODEL_CODE, label="stable lifecycle")
        )
        self.assertEqual(started["training_id"], "training_001")
        self.assertEqual(started["status"], "running")
        self.assertEqual(started["research_budget"]["remaining_research_attempts"], 2)

        with patch(
            "feature_engineering.runtime.task.evaluate_model_code_async",
            new=controlled_boundary,
        ):
            task = asyncio.create_task(runtime.execute_training("training_001"))
            await asyncio.wait_for(boundary_started.wait(), timeout=5.0)
            running = runtime.get_train_model_result("training_001")
            self.assertEqual(running, runtime.get_train_model_result("training_001"))
            self.assertEqual(
                running,
                {
                    "type": "training_status",
                    "ok": True,
                    "training_id": "training_001",
                    "status": "running",
                },
            )
            boundary_release.set()
            await task

        terminal = runtime.get_train_model_result("training_001")
        expected = copy.deepcopy(terminal)
        terminal["diagnostics"]["mse"] = 999.0
        terminal["filter"]["start_datetime"] = "mutated"
        terminal["research_budget"]["remaining_research_attempts"] = 999
        self.assertEqual(runtime.get_train_model_result("training_001"), expected)
        self.assertEqual(expected["training_id"], started["training_id"])
        self.assertEqual(expected["status"], "succeeded")
        self.assertIsNone(runtime.active_training_id)

    async def test_training_infrastructure_failure_releases_slot_and_is_stable(
        self,
    ) -> None:
        workspace = self.make_workspace()
        runtime, _model = self.make_runtime(workspace)
        started = runtime.start_training(TrainModelRequest(PAST_ONLY_MODEL_CODE))

        async def fail_boundary(**_kwargs: object) -> TrainingResult:
            raise OSError("synthetic worker transport failed")

        with patch(
            "feature_engineering.runtime.task.evaluate_model_code_async",
            new=fail_boundary,
        ):
            await runtime.execute_training(started["training_id"])

        self.assertIsNone(runtime.active_training_id)
        failures: list[tuple[str, str, str]] = []
        for _ in range(2):
            with self.assertRaises(OperationInfrastructureError) as raised:
                runtime.get_train_model_result(started["training_id"])
            failures.append(
                (
                    raised.exception.operation_id,
                    raised.exception.error_code,
                    raised.exception.message,
                )
            )
        self.assertEqual(failures[0], failures[1])
        self.assertEqual(
            failures[0][:2],
            ("training_001", "training_infrastructure_failure"),
        )
        next_started = runtime.start_training(TrainModelRequest(PAST_ONLY_MODEL_CODE))
        self.assertEqual(next_started["training_id"], "training_002")

    async def test_backtest_running_and_terminal_queries_are_repeatable(self) -> None:
        workspace = self.make_workspace(research_attempts=4)
        runtime, model = self.make_runtime(workspace)
        await self.complete_training(runtime, model)
        boundary_started = asyncio.Event()
        boundary_release = asyncio.Event()

        async def controlled_boundary(**_kwargs: object) -> BacktestResult:
            boundary_started.set()
            await boundary_release.wait()
            return BacktestResult(
                ok=True,
                metrics={"annualized_sharpe": 1.25},
                trace={"steps": [], "prediction_records": []},
                model_visible={
                    "ok": True,
                    "metrics": {
                        "annualized_sharpe": 1.25,
                        "cumulative_after_cost_return": 0.01,
                    },
                },
                audit={"policy": "synthetic_boundary", "status": "passed"},
            )

        started = runtime.start_backtest(
            BacktestRequest(model.model_id, max_gross_exposure=0.5)
        )
        self.assertEqual(started["backtest_id"], "backtest_001")
        with patch(
            "feature_engineering.runtime.task.run_backtest", new=controlled_boundary
        ):
            task = asyncio.create_task(runtime.execute_backtest("backtest_001"))
            await asyncio.wait_for(boundary_started.wait(), timeout=5.0)
            running = runtime.get_backtest_result("backtest_001")
            self.assertEqual(running, runtime.get_backtest_result("backtest_001"))
            self.assertEqual(running["status"], "running")
            boundary_release.set()
            await task

        terminal = runtime.get_backtest_result("backtest_001")
        expected = copy.deepcopy(terminal)
        terminal["metrics"]["annualized_sharpe"] = -99.0
        terminal["filter"]["start_datetime"] = "mutated"
        self.assertEqual(runtime.get_backtest_result("backtest_001"), expected)
        self.assertEqual(expected["backtest_id"], started["backtest_id"])
        self.assertEqual(expected["status"], "succeeded")
        self.assertIsNone(runtime.active_backtest_id)

    async def test_backtest_infrastructure_failure_releases_slot_and_is_stable(
        self,
    ) -> None:
        workspace = self.make_workspace(research_attempts=4)
        runtime, model = self.make_runtime(workspace)
        await self.complete_training(runtime, model)
        started = runtime.start_backtest(
            BacktestRequest(model.model_id, max_gross_exposure=0.5)
        )

        async def fail_boundary(**_kwargs: object) -> BacktestResult:
            raise ConnectionError("synthetic backtest boundary failed")

        with patch("feature_engineering.runtime.task.run_backtest", new=fail_boundary):
            await runtime.execute_backtest(started["backtest_id"])

        self.assertIsNone(runtime.active_backtest_id)
        failures: list[tuple[str, str, str]] = []
        for _ in range(2):
            with self.assertRaises(OperationInfrastructureError) as raised:
                runtime.get_backtest_result(started["backtest_id"])
            failures.append(
                (
                    raised.exception.operation_id,
                    raised.exception.error_code,
                    raised.exception.message,
                )
            )
        self.assertEqual(failures[0], failures[1])
        self.assertEqual(
            failures[0][:2],
            ("backtest_001", "backtest_infrastructure_failure"),
        )
        next_started = runtime.start_backtest(
            BacktestRequest(model.model_id, max_gross_exposure=0.25)
        )
        self.assertEqual(next_started["backtest_id"], "backtest_002")


class BudgetBoundaryTests(RuntimeBehaviorTestCase):
    async def test_only_valid_reservations_consume_the_exact_attempt_limit(
        self,
    ) -> None:
        workspace = self.make_workspace(
            research_attempts=2,
            response_error_budget=5,
        )
        runtime, model = self.make_runtime(workspace)
        outside_start = (
            workspace.public_data.start_datetime.to_pydatetime() - timedelta(minutes=1)
        )
        invalid = runtime.start_training(
            TrainModelRequest(
                PAST_ONLY_MODEL_CODE,
                train_filter=DatetimeFilter(start_datetime=outside_start),
            )
        )
        self.assertEqual(invalid["error_code"], "filter_out_of_public_range")
        self.assertEqual(
            invalid["research_budget"],
            {"remaining_research_attempts": 2, "response_errors_remaining": 4},
        )

        unknown_model = runtime.start_backtest(
            BacktestRequest("model_missing", max_gross_exposure=0.5)
        )
        self.assertEqual(unknown_model["error_code"], "unknown_model_id")
        self.assertEqual(
            unknown_model["research_budget"],
            {"remaining_research_attempts": 2, "response_errors_remaining": 3},
        )

        first = runtime.start_training(TrainModelRequest(PAST_ONLY_MODEL_CODE))
        self.assertEqual(first["research_budget"]["remaining_research_attempts"], 1)
        duplicate = runtime.start_training(TrainModelRequest(PAST_ONLY_MODEL_CODE))
        self.assertEqual(duplicate["error_code"], "training_already_running")
        self.assertEqual(
            duplicate["research_budget"],
            {"remaining_research_attempts": 1, "response_errors_remaining": 3},
        )

        async def succeed(**_kwargs: object) -> TrainingResult:
            return self.training_result(model)

        with patch(
            "feature_engineering.runtime.task.evaluate_model_code_async", new=succeed
        ):
            await runtime.execute_training(first["training_id"])
            second = runtime.start_training(TrainModelRequest(PAST_ONLY_MODEL_CODE))
            self.assertEqual(
                second["research_budget"]["remaining_research_attempts"], 0
            )
            await runtime.execute_training(second["training_id"])

        exhausted = runtime.start_training(TrainModelRequest(PAST_ONLY_MODEL_CODE))
        self.assertEqual(exhausted["error_code"], "research_attempt_budget_exhausted")
        self.assertEqual(
            exhausted["research_budget"],
            {"remaining_research_attempts": 0, "response_errors_remaining": 2},
        )
        self.assertEqual(runtime.research_attempts_consumed, 2)
        self.assertEqual(runtime.response_errors, 3)

    async def test_error_budget_waits_for_active_work_then_completes_done(self) -> None:
        workspace = self.make_workspace(
            research_attempts=3,
            response_error_budget=2,
        )
        runtime, model = self.make_runtime(workspace)
        started = runtime.start_training(TrainModelRequest(PAST_ONLY_MODEL_CODE))

        first_unknown = runtime.get_train_model_result("training_missing_1")
        self.assertEqual(first_unknown["error_code"], "unknown_training_id")
        self.assertFalse(runtime.done)
        self.assertFalse(runtime.termination_pending)
        second_unknown = runtime.get_backtest_result("backtest_missing_2")
        self.assertEqual(second_unknown["error_code"], "unknown_backtest_id")
        self.assertFalse(runtime.done)
        self.assertTrue(runtime.termination_pending)
        self.assertEqual(runtime.active_training_id, started["training_id"])

        blocked = runtime.start_backtest(
            BacktestRequest(model.model_id, max_gross_exposure=0.5)
        )
        self.assertEqual(blocked["error_code"], "rollout_terminating")
        self.assertFalse(blocked["recoverable"])
        self.assertEqual(runtime.research_attempts_consumed, 1)
        self.assertEqual(runtime.response_errors, 2)

        async def succeed(**_kwargs: object) -> TrainingResult:
            return self.training_result(model)

        with patch(
            "feature_engineering.runtime.task.evaluate_model_code_async", new=succeed
        ):
            await runtime.execute_training(started["training_id"])

        self.assertTrue(runtime.done)
        self.assertTrue(runtime.termination_pending)
        self.assertIsNone(runtime.active_training_id)
        terminal = runtime.get_train_model_result(started["training_id"])
        self.assertEqual(terminal["status"], "succeeded")
        self.assertTrue(runtime.projection(terminal)["done"])
        rejected = runtime.start_training(TrainModelRequest(PAST_ONLY_MODEL_CODE))
        self.assertEqual(rejected["error_code"], "rollout_terminating")
        self.assertEqual(runtime.research_attempts_consumed, 1)


if __name__ == "__main__":
    unittest.main()
