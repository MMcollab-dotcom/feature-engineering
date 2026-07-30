from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from behavior_helpers import (
    FUTURE_DEPENDENT_MODEL_CODE,
    PAST_ONLY_MODEL_CODE,
    SyntheticWorkspace,
    build_synthetic_workspace,
    register_dummy_model,
)

from evalenv_shared.worker.process import SubprocessWorkerHost
from feature_engineering.submissions.dataframes import (
    build_prediction_frame,
    write_dataframe,
)
from feature_engineering.submissions.registry import (
    StoredModel,
    TrainedModelRegistry,
)
from feature_engineering.submissions.runner import (
    ArtifactPrediction,
    WorkerExecutionError,
    predict_artifact,
)

AUDIT_SEED = 0xCA55A1
SAFE_AUDIT_FIELDS = frozenset(
    {"policy", "status", "probe_count", "rtol", "atol", "error_code"}
)


class CausalAuditBehaviorTests(unittest.IsolatedAsyncioTestCase):
    async def _predict_registered_model(
        self,
        *,
        workspace: SyntheticWorkspace,
        registry: TrainedModelRegistry,
        model: StoredModel,
        visibility: str,
    ) -> ArtifactPrediction | WorkerExecutionError:
        X = build_prediction_frame(
            config=workspace.config,
            rows=workspace.public_data.frame,
        )
        with registry.operation_directory(model.model_id) as operation_directory:
            X_payload = write_dataframe(operation_directory / "batch-X.arrow", X)
            return await predict_artifact(
                code=model.model_code,
                expected_source_hash=model.model_code_sha256,
                allowed_imports=tuple(
                    workspace.config.prediction.allowed_model_packages
                ),
                max_code_bytes=workspace.config.prediction.max_model_code_bytes,
                X=X_payload,
                artifact_path=operation_directory / "model.joblib",
                expected_artifact_hash=model.artifact_sha256,
                target_names=model.target_names,
                configured_feature_names=tuple(workspace.config.data.features),
                expected_inference_columns=model.inference_columns,
                audit_visibility=visibility,
                audit_seed=AUDIT_SEED,
                timeout_seconds=workspace.config.execution.timeout_seconds,
                worker_host=SubprocessWorkerHost(),
            )

    async def test_past_only_model_passes_every_future_suffix_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = build_synthetic_workspace(Path(temporary_directory))
            with TrainedModelRegistry(workspace.runtime_root) as registry:
                model = register_dummy_model(
                    registry,
                    model_code=PAST_ONLY_MODEL_CODE,
                )

                result = await self._predict_registered_model(
                    workspace=workspace,
                    registry=registry,
                    model=model,
                    visibility="public_detailed",
                )

        self.assertIsInstance(result, ArtifactPrediction)
        assert isinstance(result, ArtifactPrediction)
        self.assertEqual(result.audit["policy"], "future_suffix_v1")
        self.assertEqual(result.audit["status"], "passed")
        self.assertIsNone(result.audit["error_code"])
        self.assertEqual(result.audit["probe_count"], 3)
        self.assertEqual(result.audit["failed_probe_count"], 0)
        self.assertEqual(result.audit["max_abs_delta"], 0.0)
        self.assertEqual(result.audit["max_rel_delta"], 0.0)

    async def test_future_dependent_model_is_rejected_without_hidden_deltas(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = build_synthetic_workspace(Path(temporary_directory))
            with TrainedModelRegistry(workspace.runtime_root) as registry:
                model = register_dummy_model(
                    registry,
                    model_code=FUTURE_DEPENDENT_MODEL_CODE,
                )

                public_result = await self._predict_registered_model(
                    workspace=workspace,
                    registry=registry,
                    model=model,
                    visibility="public_detailed",
                )
                hidden_result = await self._predict_registered_model(
                    workspace=workspace,
                    registry=registry,
                    model=model,
                    visibility="hidden_fixed",
                )

        self.assertIsInstance(public_result, WorkerExecutionError)
        self.assertIsInstance(hidden_result, WorkerExecutionError)
        assert isinstance(public_result, WorkerExecutionError)
        assert isinstance(hidden_result, WorkerExecutionError)
        self.assertEqual(
            public_result.error_code,
            "temporal_batch_dependency_detected",
        )
        self.assertEqual(
            hidden_result.error_code,
            "temporal_batch_dependency_detected",
        )

        self.assertIsNotNone(public_result.details)
        self.assertIsNotNone(hidden_result.details)
        assert public_result.details is not None
        assert hidden_result.details is not None
        public_audit = public_result.details["causal_audit"]
        hidden_audit = hidden_result.details["causal_audit"]

        self.assertEqual(public_audit["status"], "failed")
        self.assertEqual(
            public_audit["error_code"],
            "temporal_batch_dependency_detected",
        )
        self.assertGreater(public_audit["failed_probe_count"], 0)
        self.assertGreater(public_audit["max_abs_delta"], 0.0)
        self.assertGreater(public_audit["max_rel_delta"], 0.0)

        self.assertEqual(set(hidden_audit), SAFE_AUDIT_FIELDS)
        self.assertEqual(
            hidden_audit,
            {field: public_audit[field] for field in SAFE_AUDIT_FIELDS},
        )


if __name__ == "__main__":
    unittest.main()
