"""Rollout-local storage for opaque trained-model artifacts."""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence
from uuid import uuid4

MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024
MAX_COMMITTED_BYTES = 8 * 1024 * 1024 * 1024
COPY_CHUNK_BYTES = 1024 * 1024


class ArtifactTooLargeError(Exception):
    """Raised when one staged artifact exceeds the exact per-model limit."""

    error_code = "model_artifact_too_large"

    def __init__(self, artifact_bytes: int) -> None:
        self.details = {
            "artifact_bytes": artifact_bytes,
            "limit_bytes": MAX_ARTIFACT_BYTES,
        }
        super().__init__(
            f"Model artifact is {artifact_bytes} bytes; "
            f"limit is {MAX_ARTIFACT_BYTES} bytes."
        )


class RegistryCapacityError(Exception):
    """Raised when committing an artifact would exceed rollout capacity."""

    error_code = "model_registry_capacity_exceeded"

    def __init__(self, *, committed_bytes: int, artifact_bytes: int) -> None:
        self.details = {
            "committed_bytes": committed_bytes,
            "artifact_bytes": artifact_bytes,
            "limit_bytes": MAX_COMMITTED_BYTES,
        }
        super().__init__(
            "Model registry capacity would be exceeded: "
            f"{committed_bytes} + {artifact_bytes} > {MAX_COMMITTED_BYTES} bytes."
        )


class UnknownModelError(LookupError):
    """Raised when a rollout-local model ID is not registered."""


@dataclass(frozen=True, slots=True)
class StoredModel:
    """Immutable metadata safe for trusted task code to inspect or persist."""

    model_id: str
    artifact_sha256: str
    artifact_bytes: int
    model_code: str
    model_code_sha256: str
    inference_columns: tuple[str, ...]
    target_names: tuple[str, ...]
    package_versions: tuple[tuple[str, str], ...]
    training_filter: tuple[tuple[str, str], ...]
    training_row_count: int
    forecast_scale: float


@dataclass(frozen=True, slots=True)
class _RegistryEntry:
    """Registry-private record that binds safe metadata to opaque bytes."""

    metadata: StoredModel
    artifact_path: Path


class TrainedModelRegistry:
    """Own exact trained-model artifact bytes for the lifetime of one rollout.

    ``register`` takes ownership of the supplied staging file and removes it on
    both success and failure. Artifact contents are never deserialized here.
    """

    def __init__(self, runtime_workspace: str | Path) -> None:
        workspace = Path(runtime_workspace)
        if not workspace.is_dir():
            raise NotADirectoryError(workspace)
        self._root = Path(
            tempfile.mkdtemp(prefix="trained-model-registry-", dir=workspace)
        )
        self._root.chmod(0o700)
        self._entries: dict[str, _RegistryEntry] = {}
        self._committed_bytes = 0
        self._next_model_number = 1
        self._closed = False

    def __enter__(self) -> TrainedModelRegistry:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def committed_bytes(self) -> int:
        return self._committed_bytes

    @contextmanager
    def staging_directory(self) -> Iterator[Path]:
        """Yield an empty registry-owned training directory and always remove it."""

        self._require_open()
        directory = Path(tempfile.mkdtemp(prefix="model-training-", dir=self._root))
        directory.chmod(0o700)
        try:
            yield directory
        finally:
            _remove_tree(directory)

    @contextmanager
    def operation_directory(self, model_id: str) -> Iterator[Path]:
        """Yield a private directory containing only the selected artifact."""

        self._require_open()
        entry = self._entry(model_id)
        directory = Path(tempfile.mkdtemp(prefix="model-operation-", dir=self._root))
        directory.chmod(0o700)
        pending: Path | None = None
        try:
            pending = _temporary_path(directory, ".model-pending-")
            copied_bytes, artifact_sha256 = _copy_and_hash(entry.artifact_path, pending)
            if (
                copied_bytes != entry.metadata.artifact_bytes
                or artifact_sha256 != entry.metadata.artifact_sha256
            ):
                raise RuntimeError(
                    "Registered model artifact changed before operation staging."
                )
            os.replace(pending, directory / "model.joblib")
            yield directory
        finally:
            if pending is not None:
                pending.unlink(missing_ok=True)
            _remove_tree(directory)

    def register(
        self,
        staged_artifact: str | Path,
        *,
        model_code: str,
        model_code_sha256: str,
        inference_columns: Sequence[str],
        target_names: Sequence[str],
        package_versions: Mapping[str, str],
        training_filter: Mapping[str, str],
        training_row_count: int,
        forecast_scale: float,
        replace_model_id: str | None = None,
    ) -> StoredModel:
        """Atomically commit or replace one completed model artifact."""

        source = Path(staged_artifact)
        pending: Path | None = None
        installed: Path | None = None
        committed = False
        try:
            self._require_open()
            actual_source_hash = hashlib.sha256(model_code.encode("utf-8")).hexdigest()
            if actual_source_hash != model_code_sha256:
                raise RuntimeError("Model source hash changed before registry commit.")
            if not math.isfinite(forecast_scale) or forecast_scale < 0.0:
                raise ValueError("forecast_scale must be finite and non-negative.")
            if not source.resolve().is_relative_to(self._root.resolve()):
                raise RuntimeError(
                    "The staged model artifact is outside the private registry root."
                )
            source_stat = source.stat()
            if not stat.S_ISREG(source_stat.st_mode):
                raise RuntimeError("The staged model artifact must be a regular file.")

            artifact_bytes = source_stat.st_size
            if artifact_bytes > MAX_ARTIFACT_BYTES:
                raise ArtifactTooLargeError(artifact_bytes)
            replaced_entry = (
                self._entry(replace_model_id) if replace_model_id is not None else None
            )
            committed_bytes = self._committed_bytes - (
                replaced_entry.metadata.artifact_bytes if replaced_entry else 0
            )
            if committed_bytes + artifact_bytes > MAX_COMMITTED_BYTES:
                raise RegistryCapacityError(
                    committed_bytes=committed_bytes,
                    artifact_bytes=artifact_bytes,
                )

            pending = self._new_pending_path()
            copied_bytes, artifact_sha256 = _copy_and_hash(source, pending)
            if copied_bytes != artifact_bytes:
                raise RuntimeError(
                    "Model artifact size changed while it was being registered."
                )
            model_id = replace_model_id or f"model_{self._next_model_number:03d}"
            metadata = StoredModel(
                model_id=model_id,
                artifact_sha256=artifact_sha256,
                artifact_bytes=artifact_bytes,
                model_code=model_code,
                model_code_sha256=model_code_sha256,
                inference_columns=tuple(inference_columns),
                target_names=tuple(target_names),
                package_versions=tuple(package_versions.items()),
                training_filter=tuple(training_filter.items()),
                training_row_count=training_row_count,
                forecast_scale=float(forecast_scale),
            )
            installed = self._root / f"artifact-{uuid4().hex}.joblib"
            os.replace(pending, installed)
            pending = None
            self._entries[model_id] = _RegistryEntry(metadata, installed)
            self._committed_bytes = committed_bytes + artifact_bytes
            if replaced_entry is None:
                self._next_model_number += 1
            else:
                try:
                    replaced_entry.artifact_path.unlink(missing_ok=True)
                except OSError:
                    pass
            committed = True
            return metadata
        finally:
            if pending is not None:
                pending.unlink(missing_ok=True)
            if installed is not None and not committed:
                installed.unlink(missing_ok=True)
            source.unlink(missing_ok=True)

    def get(self, model_id: str) -> StoredModel:
        """Return immutable safe metadata without exposing the registry path."""

        self._require_open()
        return self._entry(model_id).metadata

    def close(self) -> None:
        """Forget all models and remove the private registry root, idempotently."""

        if self._closed:
            return
        _remove_tree(self._root)
        self._entries.clear()
        self._committed_bytes = 0
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("The trained model registry is closed.")

    def _entry(self, model_id: str) -> _RegistryEntry:
        try:
            return self._entries[model_id]
        except KeyError:
            raise UnknownModelError(f"Unknown model ID: {model_id}") from None

    def _new_pending_path(self) -> Path:
        return _temporary_path(self._root, ".artifact-pending-")


def _temporary_path(directory: Path, prefix: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=prefix, dir=directory)
    os.fchmod(descriptor, 0o600)
    os.close(descriptor)
    return Path(name)


def _remove_tree(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass


def _copy_and_hash(source: Path, destination: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    copied_bytes = 0
    with source.open("rb") as source_file, destination.open("wb") as destination_file:
        while chunk := source_file.read(COPY_CHUNK_BYTES):
            destination_file.write(chunk)
            digest.update(chunk)
            copied_bytes += len(chunk)
        destination_file.flush()
        os.fsync(destination_file.fileno())
    destination.chmod(0o600)
    return copied_bytes, digest.hexdigest()
