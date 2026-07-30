"""Durable terminal submission bundles."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from feature_engineering.submissions.registry import (
    StoredModel,
    TrainedModelRegistry,
    _copy_and_hash,
    _remove_tree,
)

SCHEMA_VERSION = "feature_engineering_submission_v1"
BUNDLE_PREFIX = PurePosixPath("feature_engineering/submissions")
MODEL_SOURCE_NAME = "model.py"
MODEL_ARTIFACT_NAME = "model.joblib"
MANIFEST_NAME = "manifest.json"


class FeatureEngineeringArtifactError(RuntimeError):
    """Raised when durable feature-engineering outputs are absent or invalid."""


def promote_submission_bundle(
    *,
    task_outputs: Path | None,
    task_name: str,
    data_split: str,
    registry: TrainedModelRegistry,
    model: StoredModel,
    strategy_id: str,
    strategy_settings: Mapping[str, Any],
    strategy_hash: str,
    rationale: str | None,
    public_metrics: Mapping[str, Any],
    public_filter: Mapping[str, str] | None,
    public_audit: Mapping[str, Any] | None,
    fit_diagnostics: Mapping[str, Any],
    accepted: bool,
    official_scoring: Mapping[str, Any],
    hidden_audit: Mapping[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    """Atomically promote one exact registry entry into the durable sink."""

    root = _require_task_outputs(task_outputs)
    submissions_root = root.joinpath(*BUNDLE_PREFIX.parts)
    submissions_root.mkdir(parents=True, exist_ok=True)
    submission_id = str(uuid4())
    final_directory = submissions_root / submission_id
    temporary_directory = Path(
        tempfile.mkdtemp(prefix=".submission-", dir=submissions_root)
    )
    temporary_directory.chmod(0o755)
    renamed = False
    try:
        source_path = temporary_directory / MODEL_SOURCE_NAME
        source_bytes = model.model_code.encode("utf-8")
        _write_bytes(source_path, source_bytes)
        if _sha256_bytes(source_bytes) != model.model_code_sha256:
            raise FeatureEngineeringArtifactError(
                "Selected model source hash does not match registry metadata."
            )

        artifact_path = temporary_directory / MODEL_ARTIFACT_NAME
        with registry.operation_directory(model.model_id) as operation_directory:
            copied_bytes, copied_hash = _copy_and_hash(
                operation_directory / MODEL_ARTIFACT_NAME,
                artifact_path,
            )
        artifact_path.chmod(0o644)
        if copied_bytes != model.artifact_bytes or copied_hash != model.artifact_sha256:
            raise FeatureEngineeringArtifactError(
                "Selected model artifact does not match registry metadata."
            )

        manifest = _build_manifest(
            submission_id=submission_id,
            task_name=task_name,
            data_split=data_split,
            model=model,
            strategy_id=strategy_id,
            strategy_settings=strategy_settings,
            strategy_hash=strategy_hash,
            rationale=rationale,
            public_metrics=public_metrics,
            public_filter=public_filter,
            public_audit=public_audit,
            fit_diagnostics=fit_diagnostics,
            accepted=accepted,
            official_scoring=official_scoring,
            hidden_audit=hidden_audit,
        )
        _write_json(temporary_directory / MANIFEST_NAME, manifest)
        _fsync_directory(temporary_directory)
        try:
            os.replace(temporary_directory, final_directory)
        except OSError as exc:
            raise FeatureEngineeringArtifactError(
                "Could not atomically install the feature-engineering "
                "submission bundle."
            ) from exc
        renamed = True
        _fsync_directory(submissions_root)
        relative_path = (BUNDLE_PREFIX / submission_id).as_posix()
        return relative_path, manifest_projection(manifest)
    except BaseException:
        if renamed:
            _remove_tree(final_directory)
        raise
    finally:
        _remove_tree(temporary_directory)


def remove_submission_bundle(task_outputs: Path | None, bundle_path: str) -> None:
    """Best-effort compensating deletion before terminal trace synchronization."""

    try:
        root = _require_task_outputs(task_outputs)
        directory = _resolve_bundle_path(root, bundle_path)
        _remove_tree(directory)
    except (OSError, FeatureEngineeringArtifactError):
        pass


def manifest_projection(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the fixed safe subset stored in trace and eval JSON."""

    fields = (
        "schema_version",
        "submission_id",
        "task",
        "model_id",
        "strategy_id",
        "accepted",
        "official_scoring",
        "source",
        "artifact",
        "inference_columns",
        "target_names",
    )
    return _json_copy({name: manifest[name] for name in fields})


def _build_manifest(
    *,
    submission_id: str,
    task_name: str,
    data_split: str,
    model: StoredModel,
    strategy_id: str,
    strategy_settings: Mapping[str, Any],
    strategy_hash: str,
    rationale: str | None,
    public_metrics: Mapping[str, Any],
    public_filter: Mapping[str, int] | None,
    public_audit: Mapping[str, Any] | None,
    fit_diagnostics: Mapping[str, Any],
    accepted: bool,
    official_scoring: Mapping[str, Any],
    hidden_audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "submission_id": submission_id,
        "task": {"name": task_name, "data_split": data_split},
        "model_id": model.model_id,
        "strategy_id": strategy_id,
        "accepted": bool(accepted),
        "official_scoring": dict(official_scoring),
        "source": {
            "filename": MODEL_SOURCE_NAME,
            "sha256": model.model_code_sha256,
        },
        "artifact": {
            "filename": MODEL_ARTIFACT_NAME,
            "sha256": model.artifact_sha256,
            "bytes": model.artifact_bytes,
        },
        "inference_columns": list(model.inference_columns),
        "target_names": list(model.target_names),
        "package_versions": dict(model.package_versions),
        "training": {
            "filter": dict(model.training_filter),
            "row_count": model.training_row_count,
            "forecast_scale": model.forecast_scale,
            "median_signal_size": model.median_signal_size,
            "fit_diagnostics": dict(fit_diagnostics),
        },
        "strategy": {
            "settings": dict(strategy_settings),
            "sha256": strategy_hash,
            "rationale": rationale,
            "public_metrics": dict(public_metrics),
            "public_filter": dict(public_filter or {}),
            "causal_audit": dict(public_audit or {}),
        },
    }
    if hidden_audit is not None:
        manifest["hidden_causal_audit"] = dict(hidden_audit)
    return _json_copy(manifest)


def _require_task_outputs(task_outputs: Path | None) -> Path:
    if task_outputs is None:
        raise FeatureEngineeringArtifactError(
            "The runner did not bind a durable task_outputs directory."
        )
    root = Path(task_outputs).resolve()
    if not root.is_dir():
        raise FeatureEngineeringArtifactError(
            "The bound durable task_outputs path is not a directory."
        )
    return root


def _resolve_bundle_path(root: Path, bundle_path: str) -> Path:
    relative = PurePosixPath(bundle_path)
    if relative.is_absolute() or relative.parts[:2] != BUNDLE_PREFIX.parts:
        raise FeatureEngineeringArtifactError(
            f"Invalid feature-engineering submission bundle path: {bundle_path}"
        )
    if len(relative.parts) != 3 or relative.parts[2] in {"", ".", ".."}:
        raise FeatureEngineeringArtifactError(
            f"Invalid feature-engineering submission bundle path: {bundle_path}"
        )
    directory = root.joinpath(*relative.parts).resolve()
    if not directory.is_relative_to(root):
        raise FeatureEngineeringArtifactError(
            "Feature-engineering submission bundle path escapes "
            f"task_outputs: {bundle_path}"
        )
    return directory


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o644)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_bytes(path, encoded)


def _json_copy(payload: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, allow_nan=False, sort_keys=True))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "FeatureEngineeringArtifactError",
    "manifest_projection",
    "promote_submission_bundle",
    "remove_submission_bundle",
]
