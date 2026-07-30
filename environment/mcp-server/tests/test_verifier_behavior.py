from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pandas as pd
from behavior_helpers import build_synthetic_workspace, register_dummy_model

from feature_engineering.core.backtest import BacktestResult, run_backtest
from feature_engineering.evaluation.verifier import (
    _require_bundle_file,
    verify_submission,
)
from feature_engineering.scoring.fixed_hidden_data import load_hidden_supervised_data
from feature_engineering.submissions.bundle import (
    SCHEMA_VERSION,
    promote_submission_bundle,
)
from feature_engineering.submissions.dataframes import read_dataframe
from feature_engineering.submissions.modeling import TrainingResult
from feature_engineering.submissions.registry import TrainedModelRegistry
from feature_engineering.submissions.runner import ArtifactPrediction
from feature_engineering.submissions.strategy import compile_model_strategy


class VerifierBehaviorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = build_synthetic_workspace(self.root)

        with TrainedModelRegistry(self.workspace.runtime_root) as registry:
            model = register_dummy_model(registry)
            bundle_path, _ = promote_submission_bundle(
                task_outputs=self.workspace.submission_root,
                task_name="feature_engineering",
                data_split="public",
                registry=registry,
                model=model,
                strategy_id="strategy-1",
                strategy_settings={"max_gross_exposure": 1.0},
                strategy_hash="0" * 64,
                rationale=None,
                public_metrics={},
                public_filter={},
                public_audit=None,
                fit_diagnostics={},
                accepted=True,
                official_scoring={},
                hidden_audit=None,
            )
        self.bundle = self.workspace.submission_root / bundle_path
        self.manifest_path = self.bundle / "manifest.json"

    async def _verify(self, *, submission_root: Path | None = None) -> dict[str, float]:
        with patch.dict(
            os.environ,
            {"FEATURE_VERIFIER_RUNTIME_ROOT": str(self.workspace.runtime_root)},
        ):
            return await verify_submission(
                self.workspace.config_path,
                task_root=self.root,
                submission_root=submission_root or self.workspace.submission_root,
                reward_path=self.root / "reward.json",
                metrics_path=self.root / "metrics.json",
            )

    def _copied_submission_root(
        self,
        name: str,
        *,
        bundle_count: int,
        manifest_updates: dict[str, object] | None = None,
    ) -> Path:
        root = self.root / name
        submissions = root / "feature_engineering" / "submissions"
        submissions.mkdir(parents=True)
        for number in range(bundle_count):
            copied = submissions / f"bundle-{number}"
            shutil.copytree(self.bundle, copied)
            if manifest_updates:
                manifest_path = copied / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest.update(manifest_updates)
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return root

    def test_exact_bundle_source_and_artifact_match_manifest(self) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], SCHEMA_VERSION)
        for section in ("source", "artifact"):
            with self.subTest(section=section):
                entry = manifest[section]
                _require_bundle_file(
                    self.bundle / entry["filename"],
                    bundle=self.bundle,
                    expected=entry["sha256"],
                )

    async def test_tampered_source_and_artifact_are_rejected_before_scoring(
        self,
    ) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        for section in ("source", "artifact"):
            with self.subTest(section=section):
                path = self.bundle / manifest[section]["filename"]
                original = path.read_bytes()
                path.write_bytes(original + b"tampered")
                score = AsyncMock()
                try:
                    with patch(
                        "feature_engineering.scoring.official.score_official_strategy",
                        new=score,
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError,
                            "hash did not match",
                        ):
                            await self._verify()
                finally:
                    path.write_bytes(original)
                score.assert_not_awaited()

    async def test_manifest_paths_cannot_escape_the_bundle(self) -> None:
        original_manifest = self.manifest_path.read_bytes()
        for section in ("source", "artifact"):
            with self.subTest(section=section):
                outside = self.bundle.parent / f"outside-{section}.bin"
                outside.write_bytes(f"outside {section}".encode())
                manifest = json.loads(original_manifest)
                manifest[section] = {
                    **manifest[section],
                    "filename": f"../{outside.name}",
                    "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
                }
                self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                score = AsyncMock()
                try:
                    with patch(
                        "feature_engineering.scoring.official.score_official_strategy",
                        new=score,
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError,
                            "outside its bundle",
                        ):
                            await self._verify()
                finally:
                    self.manifest_path.write_bytes(original_manifest)
                score.assert_not_awaited()

    async def test_bundle_cardinality_schema_and_public_acceptance_precede_scoring(
        self,
    ) -> None:
        cases = (
            (
                "zero",
                self._copied_submission_root("zero", bundle_count=0),
                "Expected one submission bundle, found 0",
            ),
            (
                "multiple",
                self._copied_submission_root("multiple", bundle_count=2),
                "Expected one submission bundle, found 2",
            ),
            (
                "unsupported_schema",
                self._copied_submission_root(
                    "unsupported-schema",
                    bundle_count=1,
                    manifest_updates={"schema_version": "future-schema"},
                ),
                "schema is unsupported",
            ),
            (
                "publicly_rejected",
                self._copied_submission_root(
                    "publicly-rejected",
                    bundle_count=1,
                    manifest_updates={"accepted": False},
                ),
                "was not accepted by the public runtime",
            ),
        )
        for name, submission_root, message in cases:
            with self.subTest(case=name):
                score = AsyncMock()
                with patch(
                    "feature_engineering.scoring.official.score_official_strategy",
                    new=score,
                ):
                    with self.assertRaisesRegex(RuntimeError, message):
                        await self._verify(submission_root=submission_root)
                score.assert_not_awaited()

    async def test_verifier_refits_selected_source_and_scores_replacement_model(
        self,
    ) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        selected_source = (self.bundle / manifest["source"]["filename"]).read_text(
            encoding="utf-8"
        )
        replacement_bytes = b"trusted full-public refit artifact"
        calls: dict[str, object] = {}

        async def refit(**kwargs: object) -> TrainingResult:
            registry = kwargs["registry"]
            self.assertIsInstance(registry, TrainedModelRegistry)
            replace_model_id = kwargs["replace_model_id"]
            self.assertIsInstance(replace_model_id, str)
            selected = registry.get(replace_model_id)
            calls["selected"] = selected
            calls["model_code"] = kwargs["model_code"]
            calls["refit_public_data"] = kwargs["public_data"]
            calls["training_filter"] = kwargs["training_filter"]
            calls["replace_model_id"] = replace_model_id
            with registry.staging_directory() as staging:
                staged = staging / "model.joblib"
                staged.write_bytes(replacement_bytes)
                replacement = registry.register(
                    staged,
                    model_code=selected.model_code,
                    model_code_sha256=selected.model_code_sha256,
                    inference_columns=selected.inference_columns,
                    target_names=selected.target_names,
                    package_versions=dict(selected.package_versions),
                    training_filter=kwargs["training_filter"],
                    training_row_count=len(self.workspace.public_data.frame),
                    forecast_scale=selected.forecast_scale,
                    median_signal_size=selected.median_signal_size,
                    replace_model_id=replace_model_id,
                )
            calls["replacement"] = replacement
            return TrainingResult(
                model=replacement,
                diagnostics={"mse": 0.0},
                row_count=len(self.workspace.public_data.frame),
                feature_names=replacement.inference_columns,
                target_names=replacement.target_names,
            )

        def load_hidden(**kwargs: object):
            calls["hidden_public_config"] = kwargs["public_config"]
            calls["hidden_public_data"] = kwargs["public_data"]
            return self.workspace.public_data

        async def score_hidden(**kwargs: object) -> BacktestResult:
            registry = kwargs["registry"]
            strategy = kwargs["strategy"]
            replacement = registry.get(strategy.model_id)
            calls["hidden_model"] = replacement
            calls["hidden_strategy_model_id"] = strategy.model_id
            with registry.operation_directory(strategy.model_id) as operation:
                calls["hidden_artifact"] = (operation / "model.joblib").read_bytes()
            return BacktestResult(
                ok=True,
                metrics={
                    "annualized_sharpe": 1.25,
                    "cagr": 0.2,
                    "max_drawdown": -0.1,
                    "correlation": 0.3,
                    "pearson_ic": 0.3,
                },
                trace={},
                model_visible={"ok": True},
                audit={"visibility": "hidden_fixed"},
            )

        with (
            patch(
                "feature_engineering.scoring.official.evaluate_model_code_async",
                new=refit,
            ),
            patch(
                "feature_engineering.scoring.official.load_hidden_supervised_data",
                new=load_hidden,
            ),
            patch(
                "feature_engineering.scoring.official.run_backtest",
                new=score_hidden,
            ),
        ):
            rewards = await self._verify()

        expected_filter = {
            "start_datetime": self.workspace.public_data.start_datetime.isoformat(),
            "end_datetime": self.workspace.public_data.end_datetime.isoformat(),
        }
        selected = calls["selected"]
        replacement = calls["replacement"]
        self.assertEqual(calls["model_code"], selected_source)
        self.assertEqual(calls["training_filter"], expected_filter)
        self.assertEqual(calls["replace_model_id"], selected.model_id)
        self.assertEqual(replacement.model_id, selected.model_id)
        self.assertNotEqual(replacement.artifact_sha256, selected.artifact_sha256)
        self.assertEqual(calls["hidden_model"], replacement)
        self.assertEqual(calls["hidden_strategy_model_id"], replacement.model_id)
        self.assertEqual(calls["hidden_artifact"], replacement_bytes)
        self.assertIs(calls["hidden_public_data"], calls["refit_public_data"])
        self.assertEqual(rewards["primary_score"], 1.25)

    async def test_hidden_prediction_attachment_contains_no_outcome_columns(
        self,
    ) -> None:
        hidden = load_hidden_supervised_data(
            public_config=self.workspace.config,
            public_data=self.workspace.public_data,
            scoring_config=None,
        )
        captured: dict[str, object] = {}

        with TrainedModelRegistry(self.workspace.runtime_root) as registry:
            model = register_dummy_model(registry)
            strategy = compile_model_strategy(
                model_id=model.model_id,
                max_gross_exposure=1.0,
            )
            self.assertTrue(hasattr(strategy, "model_id"))

            async def predict(**kwargs: object) -> ArtifactPrediction:
                artifact_path = Path(kwargs["artifact_path"])
                X = read_dataframe(artifact_path.parent, kwargs["X"])
                captured["columns"] = tuple(X.columns)
                captured["index_names"] = tuple(X.index.names)
                predictions = pd.DataFrame(
                    0.0,
                    index=X.index,
                    columns=kwargs["target_names"],
                    dtype="float64",
                )
                return ArtifactPrediction(
                    frame=predictions,
                    inference_columns=tuple(kwargs["expected_inference_columns"]),
                    audit={"visibility": "hidden_fixed", "status": "accepted"},
                    serialization_policy="joblib",
                    package_versions={},
                )

            def execute(**kwargs: object):
                captured["prediction_rows"] = len(kwargs["predictions"])
                return SimpleNamespace(
                    ok=True,
                    metrics={},
                    trace={},
                    error=None,
                )

            with (
                patch(
                    "feature_engineering.core.backtest.predict_artifact",
                    new=predict,
                ),
                patch(
                    "feature_engineering.core.backtest.execute_backtest",
                    new=execute,
                ),
            ):
                result = await run_backtest(
                    config=self.workspace.config,
                    public_data=hidden,
                    strategy=strategy,
                    registry=registry,
                    start=hidden.start_datetime,
                    end=hidden.end_datetime,
                    audit_visibility="hidden_fixed",
                )

        data = self.workspace.config.data
        expected_columns = (
            data.datetime_column,
            data.symbol_column,
            *data.features,
        )
        self.assertTrue(result.ok)
        self.assertGreater(captured["prediction_rows"], 0)
        self.assertEqual(captured["columns"], expected_columns)
        self.assertEqual(
            captured["index_names"],
            (data.datetime_column, data.symbol_column),
        )
        for forbidden in (
            "target_horizon_1",
            "tradable_return",
            "beta_10d_fwd_1",
        ):
            self.assertNotIn(forbidden, captured["columns"])


if __name__ == "__main__":
    unittest.main()
