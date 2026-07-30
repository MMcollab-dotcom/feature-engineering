"""Transactional training for submitted sklearn model artifacts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from evalenv_shared.worker import WorkerProtocolError

from feature_engineering.config import TaskConfig
from feature_engineering.core.fixed_data import SupervisedData
from feature_engineering.core.portfolio import (
    calculate_forecast_beta,
    forecast_scale_from_beta,
)
from feature_engineering.submissions.dataframes import (
    build_training_frames,
    write_dataframe,
)
from feature_engineering.submissions.registry import (
    MAX_ARTIFACT_BYTES,
    MAX_COMMITTED_BYTES,
    ArtifactTooLargeError,
    RegistryCapacityError,
    StoredModel,
    TrainedModelRegistry,
)
from feature_engineering.submissions.runner import (
    WorkerExecutionError,
    fit_submitted_model,
    predict_artifact,
)
from feature_engineering.submissions.validation import validate_model_code


@dataclass(frozen=True, slots=True)
class ModelError:
    error_code: str
    stage: str
    message: str
    details: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "error_code": self.error_code,
            "stage": self.stage,
            "message": self.message[:500],
            "recoverable": True,
        }
        if self.details:
            payload["details"] = dict(self.details)
        return payload


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model: StoredModel | None
    diagnostics: dict[str, float]
    row_count: int
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    error: ModelError | None = None


def evaluate_model_code(
    *,
    config: TaskConfig,
    public_data: SupervisedData,
    registry: TrainedModelRegistry,
    model_code: str,
    training_filter: dict[str, str],
    start: datetime | None = None,
    end: datetime | None = None,
    timeout_seconds: float = 2.0,
    worker_host: Any = None,
    replace_model_id: str | None = None,
) -> TrainingResult:
    return asyncio.run(
        evaluate_model_code_async(
            config=config,
            public_data=public_data,
            registry=registry,
            model_code=model_code,
            training_filter=training_filter,
            start=start,
            end=end,
            timeout_seconds=timeout_seconds,
            worker_host=worker_host,
            replace_model_id=replace_model_id,
        )
    )


async def evaluate_model_code_async(
    *,
    config: TaskConfig,
    public_data: SupervisedData,
    registry: TrainedModelRegistry,
    model_code: str,
    training_filter: dict[str, str],
    start: datetime | None = None,
    end: datetime | None = None,
    timeout_seconds: float = 2.0,
    worker_host: Any = None,
    replace_model_id: str | None = None,
) -> TrainingResult:
    feature_names = tuple(public_data.feature_columns)
    target_names = tuple(config.data.targets)
    validation_error = validate_model_code(
        model_code,
        max_code_bytes=config.prediction.max_model_code_bytes,
    )
    if validation_error is not None:
        return _failed(
            feature_names=feature_names,
            target_names=target_names,
            error=ModelError(
                "model_code_validation_failed",
                "source_validation",
                validation_error.message,
                details={"reason": validation_error.error_code},
            ),
        )

    model_code_sha256 = hashlib.sha256(model_code.encode("utf-8")).hexdigest()
    try:
        X, y = build_training_frames(
            config=config,
            public_data=public_data,
            start=start,
            end=end,
        )
    except (TypeError, ValueError) as exc:
        return _failed(
            feature_names=feature_names,
            target_names=target_names,
            error=ModelError(
                "training_data_invalid",
                "dataframe_construction",
                f"{type(exc).__name__}: {exc}",
            ),
        )

    allowed_imports = tuple(config.prediction.allowed_model_packages)
    replaced_artifact_bytes = (
        registry.get(replace_model_id).artifact_bytes
        if replace_model_id is not None
        else 0
    )
    with registry.staging_directory() as staging_directory:
        artifact_path = staging_directory / "model.joblib"
        X_payload = write_dataframe(staging_directory / "training-X.arrow", X)
        y_payload = write_dataframe(
            staging_directory / "training-y.arrow",
            y,
        )
        fitted = await fit_submitted_model(
            code=model_code,
            expected_source_hash=model_code_sha256,
            X=X_payload,
            y=y_payload,
            artifact_path=artifact_path,
            allowed_imports=allowed_imports,
            max_code_bytes=config.prediction.max_model_code_bytes,
            timeout_seconds=timeout_seconds,
            worker_host=worker_host,
        )
        if isinstance(fitted, WorkerExecutionError):
            return _failed(
                feature_names=feature_names,
                target_names=target_names,
                row_count=len(X),
                error=_worker_model_error(fitted, operation="fit"),
            )

        artifact_path.stat()
        fitted_bytes = artifact_path.stat().st_size
        if fitted_bytes > MAX_ARTIFACT_BYTES:
            return _failed(
                feature_names=feature_names,
                target_names=target_names,
                row_count=len(X),
                error=ModelError(
                    "model_artifact_too_large",
                    "artifact_limit",
                    "Serialized estimator exceeds the per-model artifact limit.",
                    details={
                        "artifact_bytes": fitted_bytes,
                        "limit_bytes": MAX_ARTIFACT_BYTES,
                    },
                ),
            )
        committed_bytes = registry.committed_bytes - replaced_artifact_bytes
        if committed_bytes + fitted_bytes > MAX_COMMITTED_BYTES:
            return _failed(
                feature_names=feature_names,
                target_names=target_names,
                row_count=len(X),
                error=ModelError(
                    "model_registry_capacity_exceeded",
                    "registry_capacity",
                    "Serialized estimator would exceed rollout registry capacity.",
                    details={
                        "artifact_bytes": fitted_bytes,
                        "committed_bytes": committed_bytes,
                        "limit_bytes": MAX_COMMITTED_BYTES,
                    },
                ),
            )

        artifact_hash = _sha256_file(artifact_path)
        fitted_columns = fitted.get("inference_columns")
        if not isinstance(fitted_columns, list) or not all(
            isinstance(name, str) for name in fitted_columns
        ):
            raise WorkerProtocolError(
                "Training worker returned malformed inference columns."
            )
        validated = await predict_artifact(
            code=model_code,
            expected_source_hash=model_code_sha256,
            allowed_imports=allowed_imports,
            max_code_bytes=config.prediction.max_model_code_bytes,
            X=X_payload,
            artifact_path=artifact_path,
            expected_artifact_hash=artifact_hash,
            target_names=target_names,
            configured_feature_names=tuple(config.data.features),
            expected_inference_columns=tuple(fitted_columns),
            audit_visibility="public_detailed",
            run_causal_audit=False,
            timeout_seconds=timeout_seconds,
            worker_host=worker_host,
        )
        if isinstance(validated, WorkerExecutionError):
            return _failed(
                feature_names=feature_names,
                target_names=target_names,
                row_count=len(X),
                error=_worker_model_error(validated, operation="validation"),
            )
        artifact_bytes = artifact_path.stat().st_size
        if artifact_bytes > MAX_ARTIFACT_BYTES:
            return _failed(
                feature_names=feature_names,
                target_names=target_names,
                row_count=len(X),
                error=ModelError(
                    "model_artifact_too_large",
                    "artifact_limit",
                    "Serialized estimator exceeds the per-model artifact limit.",
                    details={
                        "artifact_bytes": artifact_bytes,
                        "limit_bytes": MAX_ARTIFACT_BYTES,
                    },
                ),
            )
        committed_bytes = registry.committed_bytes - replaced_artifact_bytes
        if committed_bytes + artifact_bytes > MAX_COMMITTED_BYTES:
            return _failed(
                feature_names=feature_names,
                target_names=target_names,
                row_count=len(X),
                error=ModelError(
                    "model_registry_capacity_exceeded",
                    "registry_capacity",
                    "Serialized estimator would exceed rollout registry capacity.",
                    details={
                        "artifact_bytes": artifact_bytes,
                        "committed_bytes": committed_bytes,
                        "limit_bytes": MAX_COMMITTED_BYTES,
                    },
                ),
            )
        if fitted.get("serialization_policy") != validated.serialization_policy:
            raise WorkerProtocolError(
                "Training and validation workers returned incompatible serialization policies."
            )
        fit_versions = fitted.get("package_versions")
        validation_versions = validated.package_versions
        if not isinstance(fit_versions, dict) or fit_versions != validation_versions:
            raise WorkerProtocolError(
                "Training and validation workers returned incompatible package versions."
            )

        predictions = validated.frame
        prediction_values = predictions.to_numpy(dtype="float64").reshape(-1)
        target_values = y.to_numpy(dtype="float64").reshape(-1)
        try:
            forecast_beta = calculate_forecast_beta(
                prediction_values,
                target_values,
            )
        except ValueError as exc:
            return _failed(
                feature_names=feature_names,
                target_names=target_names,
                row_count=len(X),
                error=ModelError(
                    "forecast_scale_calculation_failed",
                    "training_calibration",
                    str(exc),
                ),
            )
        forecast_scale = forecast_scale_from_beta(forecast_beta)
        diagnostics = {
            **_diagnostics(
                prediction_values,
                target_values,
            ),
            "fit_forecast_beta": forecast_beta,
            "forecast_scale": forecast_scale,
            "model_artifact_bytes": float(artifact_bytes),
        }

        try:
            model = registry.register(
                artifact_path,
                model_code=model_code,
                model_code_sha256=model_code_sha256,
                inference_columns=validated.inference_columns,
                target_names=target_names,
                package_versions={str(k): str(v) for k, v in fit_versions.items()},
                training_filter=training_filter,
                training_row_count=len(X),
                forecast_scale=forecast_scale,
                replace_model_id=replace_model_id,
            )
        except (ArtifactTooLargeError, RegistryCapacityError) as exc:
            stage = (
                "artifact_limit"
                if isinstance(exc, ArtifactTooLargeError)
                else "registry_capacity"
            )
            return _failed(
                feature_names=feature_names,
                target_names=target_names,
                row_count=len(X),
                error=ModelError(
                    exc.error_code,
                    stage,
                    str(exc),
                    details=exc.details,
                ),
            )
        return TrainingResult(
            model=model,
            diagnostics=diagnostics,
            row_count=len(X),
            feature_names=validated.inference_columns,
            target_names=target_names,
        )

def _worker_model_error(error: WorkerExecutionError, *, operation: str) -> ModelError:
    if error.error_code == "invalid_fitted_estimator":
        stage = "estimator_validation"
    elif error.error_code == "model_serialization_failed":
        stage = "serialization"
    elif error.error_code == "model_deserialization_failed":
        stage = "deserialization"
    elif error.error_code == "model_prediction_validation_failed":
        stage = "smoke_prediction"
    else:
        stage = "fit" if operation == "fit" else "smoke_prediction"
    return ModelError(error.error_code, stage, error.message, details=error.details)


def _failed(
    *,
    feature_names: tuple[str, ...],
    target_names: tuple[str, ...],
    error: ModelError,
    row_count: int = 0,
) -> TrainingResult:
    return TrainingResult(
        model=None,
        diagnostics={},
        row_count=row_count,
        feature_names=feature_names,
        target_names=target_names,
        error=error,
    )


def _diagnostics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    residual = predictions - targets
    baseline = targets - float(targets.mean())
    if (
        len(predictions) < 2
        or np.std(predictions) <= 1.0e-12
        or np.std(targets) <= 1.0e-12
    ):
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(predictions, targets)[0, 1])
    return {
        "fit_mse": round(float(np.mean(residual**2)), 6),
        "fit_baseline_mse": round(float(np.mean(baseline**2)), 6),
        "fit_correlation": round(correlation, 3),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ModelError",
    "TrainingResult",
    "evaluate_model_code",
    "evaluate_model_code_async",
]
