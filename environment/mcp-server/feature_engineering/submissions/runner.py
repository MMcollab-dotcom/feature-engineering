"""Parent-side worker client for feature-engineering model artifacts."""

from __future__ import annotations

import asyncio
import hashlib
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from evalenv_shared.worker import (
    WorkerProtocolError,
    WorkerRemoteError,
    WorkerTimeoutError,
)
from evalenv_shared.worker.process import SubprocessWorkerHost
from feature_engineering.submissions.causal_audit import (
    AuditInputInsufficient,
    audit_delta,
    audit_summary,
    build_future_suffix_probes,
)
from feature_engineering.submissions.dataframes import (
    canonical_prediction_frame,
    read_dataframe,
    write_dataframe,
)
from feature_engineering.submissions.validation import redact_submitted_identity

SUBMITTED_PREDICTION_ERROR_CODES = frozenset(
    {
        "file_io_blocked",
        "inference_schema_changed",
        "model_prediction_validation_failed",
        "temporal_audit_input_insufficient",
        "temporal_batch_dependency_detected",
        "temporal_probe_rejected",
    }
)
SUBMITTED_WORKER_ERROR_CODES = frozenset(
    {
        *SUBMITTED_PREDICTION_ERROR_CODES,
        "invalid_fitted_estimator",
        "invalid_model_code_syntax",
        "missing_model_code",
        "missing_model_code_function",
        "model_code_execution_failed",
        "model_code_too_large",
        "model_deserialization_failed",
        "model_fit_failed",
        "model_serialization_failed",
        "inference_columns_missing",
    }
)


@dataclass(frozen=True, slots=True)
class WorkerExecutionError:
    error_code: str
    message: str
    details: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "error_code": self.error_code,
            "message": self.message[:500],
            "recoverable": True,
        }
        if self.details:
            payload["details"] = dict(self.details)
        return payload


@dataclass(frozen=True, slots=True)
class ArtifactPrediction:
    frame: pd.DataFrame
    inference_columns: tuple[str, ...]
    audit: dict[str, Any]
    serialization_policy: str
    package_versions: dict[str, str]


async def fit_submitted_model(
    *,
    code: str,
    expected_source_hash: str,
    X: dict[str, Any],
    y: dict[str, Any],
    artifact_path: Path,
    allowed_imports: tuple[str, ...],
    max_code_bytes: int,
    timeout_seconds: float,
    worker_host: Any = None,
) -> dict[str, Any] | WorkerExecutionError:
    return await _run_one_shot(
        {
            "mode": "fit",
            "code": code,
            "expected_source_hash": expected_source_hash,
            "X": X,
            "y": y,
            "artifact_name": artifact_path.name,
            "allowed_imports": list(allowed_imports),
            "max_code_bytes": int(max_code_bytes),
        },
        workdir=artifact_path.parent,
        timeout_seconds=timeout_seconds,
        timeout_code="model_fit_timeout",
        worker_host=worker_host,
    )


async def predict_artifact(
    *,
    code: str,
    expected_source_hash: str,
    allowed_imports: tuple[str, ...],
    max_code_bytes: int,
    X: dict[str, Any],
    artifact_path: Path,
    expected_artifact_hash: str,
    target_names: tuple[str, ...],
    configured_feature_names: tuple[str, ...],
    expected_inference_columns: tuple[str, ...] | None = None,
    audit_visibility: str = "public_detailed",
    audit_seed: int | None = None,
    run_causal_audit: bool = True,
    timeout_seconds: float,
    worker_host: Any = None,
) -> ArtifactPrediction | WorkerExecutionError:
    X_frame = read_dataframe(artifact_path.parent, X)
    init_request: dict[str, Any] = {
        "mode": "init_prediction",
        "code": code,
        "expected_source_hash": expected_source_hash,
        "allowed_imports": list(allowed_imports),
        "max_code_bytes": int(max_code_bytes),
        "artifact_name": artifact_path.name,
        "expected_artifact_hash": expected_artifact_hash,
        "target_names": list(target_names),
    }
    host = worker_host or SubprocessWorkerHost()
    session = await host.start(
        timeout_s=float(timeout_seconds),
        cwd=str(artifact_path.parent),
        env=_worker_env(artifact_path.parent),
    )
    source_initializing = [True]
    try:
        result = await asyncio.wait_for(
            _run_prediction_operation(
                session=session,
                init_request=init_request,
                source_hash=expected_source_hash,
                X=X_frame,
                X_payload=X,
                artifact_path=artifact_path,
                expected_artifact_hash=expected_artifact_hash,
                target_names=target_names,
                configured_feature_names=configured_feature_names,
                expected_inference_columns=expected_inference_columns,
                audit_visibility=audit_visibility,
                audit_seed=int(
                    audit_seed if audit_seed is not None else secrets.randbits(64)
                ),
                run_causal_audit=run_causal_audit,
                source_initializing=source_initializing,
            ),
            timeout=float(timeout_seconds),
        )
    except TimeoutError:
        await session.close()
        return WorkerExecutionError(
            (
                "model_code_execution_failed"
                if source_initializing[0]
                else "model_prediction_validation_failed"
            ),
            "Submitted model code exceeded the task timeout.",
            details={"timeout_seconds": float(timeout_seconds)},
        )
    except BaseException:
        await session.close()
        raise
    if isinstance(result, WorkerExecutionError):
        await session.close()
        return result
    try:
        await session.shutdown()
    except BaseException:
        await session.close()
        raise
    return result


async def _run_prediction_operation(
    *,
    session: Any,
    init_request: dict[str, Any],
    source_hash: str,
    X: pd.DataFrame,
    X_payload: dict[str, Any],
    artifact_path: Path,
    expected_artifact_hash: str,
    target_names: tuple[str, ...],
    configured_feature_names: tuple[str, ...],
    expected_inference_columns: tuple[str, ...] | None,
    audit_visibility: str,
    audit_seed: int,
    run_causal_audit: bool,
    source_initializing: list[bool],
) -> ArtifactPrediction | WorkerExecutionError:
    initialized = await _prediction_request(
        session,
        init_request,
        source_hash=source_hash,
        timeout_code="model_code_execution_failed",
    )
    if isinstance(initialized, WorkerExecutionError):
        return (
            _audit_rejection(initialized, visibility=audit_visibility)
            if run_causal_audit
            else initialized
        )
    _require_artifact_hash(artifact_path, expected_artifact_hash)
    source_initializing[0] = False
    try:
        policy = str(initialized["serialization_policy"])
        versions = {
            str(key): str(value)
            for key, value in initialized["package_versions"].items()
        }
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise WorkerProtocolError(
            "Submitted model worker returned malformed initialization metadata."
        ) from exc

    candidate = await _predict_batch(
        session=session,
        source_hash=source_hash,
        X=X,
        X_payload=X_payload,
        artifact_path=artifact_path,
        artifact_hash=expected_artifact_hash,
        target_names=target_names,
        expected_inference_columns=expected_inference_columns,
        prediction_error_code="model_prediction_validation_failed",
    )
    if isinstance(candidate, WorkerExecutionError):
        return (
            _audit_rejection(candidate, visibility=audit_visibility)
            if run_causal_audit
            else candidate
        )
    candidate_frame, inference_columns = candidate
    if not run_causal_audit:
        return ArtifactPrediction(
            frame=candidate_frame,
            inference_columns=inference_columns,
            audit={},
            serialization_policy=policy,
            package_versions=versions,
        )

    perturbable = [
        name for name in configured_feature_names if name in inference_columns
    ]
    try:
        probes = build_future_suffix_probes(
            X,
            feature_names=perturbable,
            seed=audit_seed,
        )
    except AuditInputInsufficient as exc:
        return _audit_rejection(
            WorkerExecutionError(
                "temporal_audit_input_insufficient",
                f"{type(exc).__name__}: {exc}",
            ),
            visibility=audit_visibility,
        )

    candidate_values = candidate_frame.to_numpy(dtype="float64")
    failed = 0
    max_abs = 0.0
    max_rel = 0.0
    for probe in probes:
        probe_payload = write_dataframe(
            artifact_path.parent / "batch-X.arrow",
            probe.X,
        )
        predicted = await _predict_batch(
            session=session,
            source_hash=source_hash,
            X=probe.X,
            X_payload=probe_payload,
            artifact_path=artifact_path,
            artifact_hash=expected_artifact_hash,
            target_names=target_names,
            expected_inference_columns=inference_columns,
            prediction_error_code="temporal_probe_rejected",
        )
        if isinstance(predicted, WorkerExecutionError):
            return _audit_rejection(
                predicted,
                visibility=audit_visibility,
                failed_probe_count=1,
            )
        probe_frame, _ = predicted
        matches, probe_abs, probe_rel = audit_delta(
            candidate_values,
            probe_frame.to_numpy(dtype="float64"),
            past_mask=probe.past_mask,
        )
        max_abs = max(max_abs, probe_abs)
        max_rel = max(max_rel, probe_rel)
        failed += int(not matches)
    if failed:
        summary = audit_summary(
            visibility=audit_visibility,
            status="failed",
            error_code="temporal_batch_dependency_detected",
            failed_probe_count=failed,
            max_abs_delta=max_abs,
            max_rel_delta=max_rel,
        )
        return WorkerExecutionError(
            "temporal_batch_dependency_detected",
            "The fitted estimator did not satisfy causal prediction requirements.",
            details={"causal_audit": summary},
        )
    return ArtifactPrediction(
        frame=candidate_frame,
        inference_columns=inference_columns,
        audit=audit_summary(
            visibility=audit_visibility,
            status="passed",
            error_code=None,
            max_abs_delta=max_abs,
            max_rel_delta=max_rel,
        ),
        serialization_policy=policy,
        package_versions=versions,
    )


async def _predict_batch(
    *,
    session: Any,
    source_hash: str,
    X: pd.DataFrame,
    X_payload: dict[str, Any],
    artifact_path: Path,
    artifact_hash: str,
    target_names: tuple[str, ...],
    expected_inference_columns: tuple[str, ...] | None,
    prediction_error_code: str,
) -> tuple[pd.DataFrame, tuple[str, ...]] | WorkerExecutionError:
    request: dict[str, Any] = {
        "mode": "predict_batch",
        "X": X_payload,
        "expected_artifact_hash": artifact_hash,
        "prediction_error_code": prediction_error_code,
    }
    if expected_inference_columns is not None:
        request["expected_inference_columns"] = list(expected_inference_columns)
    payload = await _prediction_request(
        session,
        request,
        source_hash=source_hash,
    )
    _require_artifact_hash(artifact_path, artifact_hash)
    if isinstance(payload, WorkerExecutionError):
        return payload
    try:
        wire_frame = read_dataframe(
            artifact_path.parent,
            payload["predictions"],
            expected_rows=len(X),
            expected_columns=len(target_names) + 2,
            max_bytes=_prediction_arrow_limit(len(X), len(target_names)),
        )
        raw_columns = payload["inference_columns"]
        if (
            not isinstance(raw_columns, list)
            or not raw_columns
            or not all(isinstance(value, str) and value for value in raw_columns)
            or len(set(raw_columns)) != len(raw_columns)
        ):
            raise TypeError("invalid inference columns")
        columns = tuple(raw_columns)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise WorkerProtocolError(
            "Submitted model worker returned malformed artifact prediction data."
        ) from exc
    if expected_inference_columns is not None and columns != expected_inference_columns:
        return WorkerExecutionError(
            "inference_schema_changed",
            "Estimator feature_names_in_ changed after smoke validation.",
        )
    missing = [name for name in columns if name not in X.columns]
    if missing:
        return WorkerExecutionError(
            "inference_columns_missing",
            f"Prediction data is missing inference column(s): {', '.join(missing)}",
        )
    try:
        frame = canonical_prediction_frame(
            wire_frame.to_numpy(dtype="float64"),
            X=X,
            target_names=list(target_names),
        )
    except (TypeError, ValueError) as exc:
        return WorkerExecutionError(
            prediction_error_code,
            f"{type(exc).__name__}: {exc}",
        )
    return frame, columns


def _prediction_arrow_limit(row_count: int, target_count: int) -> int:
    return 1024 * 1024 + row_count * (target_count + 2) * 64


async def _prediction_request(
    session: Any,
    request: dict[str, Any],
    *,
    source_hash: str,
    timeout_code: str = "model_prediction_validation_failed",
) -> dict[str, Any] | WorkerExecutionError:
    try:
        payload = await session.request(
            {"type": "feature_engineering_request", **request}
        )
    except (WorkerRemoteError, WorkerTimeoutError) as exc:
        return _submitted_execution_error(
            exc,
            timeout_code=timeout_code,
            source_hash=source_hash,
        )
    return _require_result_payload(
        payload,
        operation=str(request.get("mode") or "request"),
    )


def _audit_rejection(
    error: WorkerExecutionError,
    *,
    visibility: str,
    failed_probe_count: int = 0,
) -> WorkerExecutionError:
    details = dict(error.details or {})
    details["causal_audit"] = audit_summary(
        visibility=visibility,
        status="rejected",
        error_code=error.error_code,
        failed_probe_count=failed_probe_count,
    )
    message = (
        "Hidden artifact prediction was rejected."
        if visibility == "hidden_fixed"
        else error.message
    )
    return WorkerExecutionError(error.error_code, message, details=details)


def _require_artifact_hash(path: Path, expected_hash: str) -> None:
    if _sha256_file(path) != expected_hash:
        raise WorkerProtocolError(
            "Submitted model worker changed the bound artifact bytes."
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def _run_one_shot(
    request: dict[str, Any],
    *,
    workdir: Path,
    timeout_seconds: float,
    timeout_code: str,
    worker_host: Any,
) -> dict[str, Any] | WorkerExecutionError:
    host = worker_host or SubprocessWorkerHost()
    session = await host.start(
        timeout_s=float(timeout_seconds),
        cwd=str(workdir),
        env=_worker_env(workdir),
    )
    try:
        payload = await session.request(
            {"type": "feature_engineering_request", **request}
        )
    except (WorkerRemoteError, WorkerTimeoutError) as exc:
        await session.close()
        return _submitted_execution_error(
            exc,
            timeout_code=timeout_code,
            source_hash=str(request.get("expected_source_hash") or ""),
        )
    except BaseException:
        await session.close()
        raise
    try:
        await session.shutdown()
    except BaseException:
        await session.close()
        raise
    return _require_result_payload(
        payload, operation=str(request.get("mode") or "request")
    )


def _require_result_payload(payload: Any, *, operation: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise WorkerProtocolError(
            f"Submitted model worker returned a malformed {operation} result payload."
        )
    return payload


def _submitted_execution_error(
    exc: WorkerRemoteError | WorkerTimeoutError,
    *,
    timeout_code: str,
    source_hash: str,
) -> WorkerExecutionError:
    if isinstance(exc, WorkerRemoteError):
        payload = dict(exc.error)
        error_code = str(
            payload.get("error_code")
            or payload.get("code")
            or "worker_rejected_model_code"
        )
        if error_code not in SUBMITTED_WORKER_ERROR_CODES:
            raise WorkerProtocolError(
                "Submitted model worker returned unrecognized error code: "
                f"{error_code!r}."
            ) from exc
        private_messages = {
            "temporal_audit_input_insufficient": (
                "The fitted estimator did not satisfy causal prediction requirements."
            ),
            "temporal_batch_dependency_detected": (
                "The fitted estimator did not satisfy causal prediction requirements."
            ),
            "temporal_probe_rejected": (
                "The fitted estimator did not satisfy causal prediction requirements."
            ),
        }
        return WorkerExecutionError(
            error_code,
            private_messages.get(
                error_code,
                _public_error_message(payload, source_hash=source_hash),
            ),
            details=payload.get("details")
            if isinstance(payload.get("details"), dict)
            else None,
        )
    return WorkerExecutionError(
        timeout_code,
        "Submitted model code exceeded the task timeout.",
        details={"timeout_seconds": float(exc.timeout_s)},
    )


def is_submitted_prediction_failure(error_code: str) -> bool:
    return error_code in SUBMITTED_PREDICTION_ERROR_CODES


def _public_error_message(payload: dict[str, Any], *, source_hash: str) -> str:
    message = str(payload.get("message") or "Submitted model code was rejected.")
    if source_hash:
        message = redact_submitted_identity(message, source_hash=source_hash)
    return message[:500]


def _worker_env(workdir: Path) -> dict[str, str]:
    task_root = Path(__file__).resolve().parents[2]
    return {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(task_root),
        "TMPDIR": str(workdir),
        "TEMP": str(workdir),
        "TMP": str(workdir),
    }


__all__ = [
    "ArtifactPrediction",
    "SUBMITTED_PREDICTION_ERROR_CODES",
    "SUBMITTED_WORKER_ERROR_CODES",
    "WorkerExecutionError",
    "fit_submitted_model",
    "is_submitted_prediction_failure",
    "predict_artifact",
]
