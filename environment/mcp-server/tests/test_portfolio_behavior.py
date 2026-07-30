from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from behavior_helpers import build_synthetic_workspace

from feature_engineering.core.granularity import granularity_delta
from feature_engineering.core.portfolio import execute_backtest
from feature_engineering.core.portfolio_engine import RebalanceDecision
from feature_engineering.submissions.strategy import CompiledStrategy


class _ControlledPortfolioEngine:
    targets: dict[int, np.ndarray] = {}

    def __init__(self, **_kwargs: object) -> None:
        pass

    def rebalance(self, request: object) -> RebalanceDecision:
        step_index = int(request.step_index)  # type: ignore[attr-defined]
        return RebalanceDecision(
            target_weights=self.targets[step_index].copy(),
            diagnostics={},
        )


class PortfolioBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.workspace = build_synthetic_workspace(
            Path(self._temporary_directory.name),
            public_steps=7,
            hidden_steps=7,
        )

    def _config(self, *, rebalance_freq: int):
        config = self.workspace.config
        return config.model_copy(
            update={
                "backtest": config.backtest.model_copy(
                    update={"rebalance_freq": rebalance_freq}
                ),
                "execution": config.execution.model_copy(
                    update={"initial_capital": 1_000.0}
                ),
                "costs": config.costs.model_copy(update={"linear_fee_bps": 100.0}),
                "reward": config.reward.model_copy(update={"periods_per_year": 2}),
            }
        )

    def _public_data_with_returns(
        self,
        returns_by_origin: dict[pd.Timestamp, np.ndarray],
    ):
        public_data = self.workspace.public_data
        frame = public_data.frame.copy()
        data = self.workspace.config.data
        for origin, returns in returns_by_origin.items():
            for symbol, value in zip(public_data.symbols, returns, strict=True):
                row = (frame[data.datetime_column] == origin) & (
                    frame[data.symbol_column] == symbol
                )
                self.assertEqual(int(row.sum()), 1)
                frame.loc[row, data.tradable_return_column] = float(value)
                frame.loc[row, data.targets[0]] = float(value)
        return public_data.__class__(
            frame=frame,
            feature_columns=public_data.feature_columns,
            target_columns=public_data.target_columns,
            datetimes=public_data.datetimes,
            symbols=public_data.symbols,
            start_datetime=public_data.start_datetime,
            end_datetime=public_data.end_datetime,
            manifest_sha256=public_data.manifest_sha256,
        )

    def _predictions(self, origins: list[pd.Timestamp]) -> pd.DataFrame:
        config = self.workspace.config
        index = pd.MultiIndex.from_product(
            [origins, self.workspace.public_data.symbols],
            names=[config.data.datetime_column, config.data.symbol_column],
        )
        values = np.arange(1.0, len(index) + 1.0)
        return pd.DataFrame({config.data.targets[0]: values}, index=index)

    def _execute(
        self,
        *,
        origins: list[pd.Timestamp],
        public_data: object,
        targets: dict[int, np.ndarray],
        rebalance_freq: int,
        max_gross_exposure: float = 1.0,
    ):
        period = granularity_delta(self.workspace.config.data.granularity)
        _ControlledPortfolioEngine.targets = targets
        strategy = CompiledStrategy(
            model_id="controlled-model",
            max_gross_exposure=max_gross_exposure,
            settings_hash="controlled-settings",
        )
        with patch(
            "feature_engineering.core.portfolio.EmaSmoothedPortfolioEngine",
            _ControlledPortfolioEngine,
        ):
            return execute_backtest(
                config=self._config(rebalance_freq=rebalance_freq),
                public_data=public_data,
                start=origins[0],
                end=origins[-1] + 2 * period,
                strategy=strategy,
                predictions=self._predictions(origins),
                forecast_scale=1.0,
                median_signal_size=1.0,
            )

    def test_forecast_execution_realization_alignment_and_exact_accounting(
        self,
    ) -> None:
        origins = list(self.workspace.public_data.datetimes[:2])
        origin = origins[0]
        forward_returns = np.asarray([0.10, 0.05, -0.02, 0.03])
        sentinel_next_returns = np.asarray([-0.70, 0.60, 0.50, -0.40])
        public_data = self._public_data_with_returns(
            {
                origin: forward_returns,
                origins[1]: sentinel_next_returns,
            }
        )
        target = np.asarray([0.40, -0.20, 0.10, -0.10])

        result = self._execute(
            origins=[origin],
            public_data=public_data,
            targets={0: target},
            rebalance_freq=1,
        )

        self.assertTrue(result.ok)
        self.assertEqual(len(result.trace["steps"]), 1)
        step = result.trace["steps"][0]
        period = granularity_delta(self.workspace.config.data.granularity)
        self.assertEqual(step["datetime"], origin.isoformat())
        self.assertEqual(step["execution_datetime"], (origin + period).isoformat())
        self.assertEqual(
            step["realization_datetime"], (origin + 2 * period).isoformat()
        )

        symbols = self.workspace.public_data.symbols
        expected_realized = dict(zip(symbols, forward_returns, strict=True))
        self.assertEqual(step["realized_targets"]["target_name"], "target_horizon_1")
        self.assertEqual(step["realized_targets"]["values"], expected_realized)
        prediction_records = result.trace["prediction_records"]
        self.assertEqual(
            [record["realized_target"] for record in prediction_records],
            forward_returns.tolist(),
        )
        self.assertEqual(step["previous_weights"], dict.fromkeys(symbols, 0.0))
        np.testing.assert_allclose(
            list(step["target_weights"].values()), target, rtol=0.0, atol=0.0
        )

        # 100 bps is one percent. Trading from zero to 0.8 gross therefore
        # costs 8 on a 1,000 NAV before the origin row's forward return realizes.
        expected_turnover = 0.80
        expected_traded_notional = 800.0
        expected_fee = 8.0
        expected_gross_return = 0.025
        expected_step_return = 0.017
        expected_nav = 1_017.0
        expected_end_weights = target * (1.0 + forward_returns) / 1.017

        self.assertAlmostEqual(result.metrics["turnover"], expected_turnover, places=14)
        self.assertAlmostEqual(result.metrics["fee_paid"], expected_fee, places=14)
        self.assertAlmostEqual(
            step["traded_notional"], expected_traded_notional, places=14
        )
        self.assertAlmostEqual(step["fee_paid"], expected_fee, places=14)
        self.assertAlmostEqual(
            step["step_return"] + step["fee_paid"] / 1_000.0,
            expected_gross_return,
            places=14,
        )
        self.assertAlmostEqual(step["step_return"], expected_step_return, places=14)
        self.assertAlmostEqual(step["nav"], expected_nav, places=12)
        np.testing.assert_allclose(
            list(step["end_weights"].values()),
            expected_end_weights,
            rtol=0.0,
            atol=1.0e-14,
        )

    def test_no_rebalance_carries_return_drifted_weights_without_cost(self) -> None:
        origins = list(self.workspace.public_data.datetimes[:2])
        first_returns = np.asarray([0.10, 0.05, -0.02, 0.03])
        second_returns = np.asarray([0.02, -0.04, 0.01, -0.03])
        public_data = self._public_data_with_returns(
            {origins[0]: first_returns, origins[1]: second_returns}
        )
        first_target = np.asarray([0.40, -0.20, 0.10, -0.10])

        result = self._execute(
            origins=origins,
            public_data=public_data,
            targets={0: first_target},
            rebalance_freq=2,
            max_gross_exposure=1.0,
        )

        self.assertTrue(result.ok)
        first, carried = result.trace["steps"]
        self.assertTrue(first["did_rebalance"])
        self.assertFalse(carried["did_rebalance"])
        np.testing.assert_allclose(
            list(carried["previous_weights"].values()),
            list(first["end_weights"].values()),
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(carried["target_weights"], carried["previous_weights"])
        self.assertEqual(carried["traded_notional"], 0.0)
        self.assertEqual(carried["fee_paid"], 0.0)
        self.assertIsNone(carried["optimisation"])

        carried_target = np.asarray(list(carried["target_weights"].values()))
        expected_gross_return = float(carried_target @ second_returns)
        expected_nav = first["nav"] * (1.0 + expected_gross_return)
        expected_end_weights = (
            carried_target * (1.0 + second_returns) / (1.0 + expected_gross_return)
        )
        self.assertAlmostEqual(carried["step_return"], expected_gross_return, places=14)
        self.assertAlmostEqual(carried["nav"], expected_nav, places=12)
        np.testing.assert_allclose(
            list(carried["end_weights"].values()),
            expected_end_weights,
            rtol=0.0,
            atol=1.0e-14,
        )
        self.assertLessEqual(float(np.abs(carried_target).sum()), 1.0)
        self.assertLessEqual(float(np.abs(expected_end_weights).sum()), 1.0)
        self.assertAlmostEqual(result.metrics["turnover"], 0.8, places=14)
        self.assertAlmostEqual(result.metrics["fee_paid"], 8.0, places=14)

    def test_rebalance_turnover_uses_return_drifted_pretrade_weights(self) -> None:
        origins = list(self.workspace.public_data.datetimes[:2])
        first_returns = np.asarray([0.10, 0.05, -0.02, 0.03])
        second_returns = np.asarray([0.02, -0.04, 0.01, -0.03])
        public_data = self._public_data_with_returns(
            {origins[0]: first_returns, origins[1]: second_returns}
        )
        first_target = np.asarray([0.40, -0.20, 0.10, -0.10])
        second_target = np.asarray([0.20, -0.10, 0.30, -0.20])

        result = self._execute(
            origins=origins,
            public_data=public_data,
            targets={0: first_target, 1: second_target},
            rebalance_freq=1,
        )

        self.assertTrue(result.ok)
        first, second = result.trace["steps"]
        expected_pretrade = first_target * (1.0 + first_returns) / 1.017
        expected_second_turnover = float(
            np.abs(second_target - expected_pretrade).sum()
        )
        expected_second_notional = expected_second_turnover * 1_017.0
        expected_second_fee = expected_second_notional * 0.01
        expected_second_gross_return = float(second_target @ second_returns)
        expected_second_step_return = (
            expected_second_gross_return - expected_second_turnover * 0.01
        )
        expected_final_nav = 1_017.0 * (1.0 + expected_second_step_return)

        np.testing.assert_allclose(
            list(second["previous_weights"].values()),
            expected_pretrade,
            rtol=0.0,
            atol=1.0e-14,
        )
        np.testing.assert_allclose(
            list(second["target_weights"].values()),
            second_target,
            rtol=0.0,
            atol=0.0,
        )
        self.assertAlmostEqual(
            second["traded_notional"], expected_second_notional, places=12
        )
        self.assertAlmostEqual(second["fee_paid"], expected_second_fee, places=12)
        self.assertAlmostEqual(
            second["step_return"] + second["fee_paid"] / first["nav"],
            expected_second_gross_return,
            places=14,
        )
        self.assertAlmostEqual(
            second["step_return"], expected_second_step_return, places=14
        )
        self.assertAlmostEqual(second["nav"], expected_final_nav, places=12)
        self.assertAlmostEqual(
            result.metrics["turnover"],
            0.8 + expected_second_turnover,
            places=14,
        )
        self.assertAlmostEqual(
            result.metrics["fee_paid"],
            8.0 + expected_second_fee,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
