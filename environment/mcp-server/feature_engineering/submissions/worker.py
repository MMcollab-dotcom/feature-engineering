"""Subprocess entrypoint for fitted feature-engineering model artifacts."""

from __future__ import annotations

import builtins
from dataclasses import dataclass
import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
import sys
import types
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from evalenv_shared.worker.program import WorkerProgram
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from feature_engineering.submissions.dataframes import (
    canonical_prediction_frame,
    read_dataframe,
    write_dataframe,
)
from feature_engineering.submissions.validation import (
    submitted_module_name,
    validate_model_code,
)


SERIALIZATION_POLICY = "joblib-v1-protocol5-uncompressed"
SAFE_BUILTINS = {
    "__build_class__": builtins.__build_class__,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "object": object,
    "pow": pow,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "ValueError": ValueError,
    "Exception": Exception,
}


class FileIOBlocked(RuntimeError):
    pass


class ModelCodeExecutionFailed(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _PredictionState:
    artifact_path: Path
    target_names: list[str]


class FeatureEngineeringWorkerHandler:
    def __init__(self) -> None:
        self._prediction_state: _PredictionState | None = None

    def handle(self, message: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            response = self._handle_request(dict(message))
        except FileIOBlocked as exc:
            return _error("file_io_blocked", exc)
        except ModelCodeExecutionFailed as exc:
            return _error("model_code_execution_failed", exc)
        except Exception as exc:
            return _internal_error(exc)
        if response.get("ok") is False:
            return {"ok": False, "error": dict(response)}
        return {"ok": True, "value": response}

    def _handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        mode = request.get("mode")
        if mode in {"fit", "init_prediction"}:
            code = request.get("code")
            if not isinstance(code, str):
                raise ValueError("Worker request missing model code.")
            validation_error = validate_model_code(
                code,
                max_code_bytes=int(request.get("max_code_bytes", 20000)),
            )
            if validation_error is not None:
                return validation_error.to_payload()
        if mode == "fit":
            return self._fit(request, code)
        if mode == "init_prediction":
            return self._init_prediction(request, code)
        if mode == "predict_batch":
            return self._predict_batch(request)
        raise ValueError(f"Unsupported worker mode: {mode!r}")

    def _fit(self, request: dict[str, Any], code: str) -> dict[str, Any]:
        X = read_dataframe(Path.cwd(), request.get("X"))
        y = read_dataframe(Path.cwd(), request.get("y"))
        if not y.index.equals(X.index):
            raise ValueError("Worker training DataFrame indexes do not match.")
        allowed_imports = tuple(str(value) for value in request.get("allowed_imports", ()))
        module = _load_submitted_module(
            code,
            expected_source_hash=str(request.get("expected_source_hash") or ""),
            allowed_imports=allowed_imports,
        )
        try:
            model = module.train_model(X, y)
        except FileIOBlocked:
            raise
        except Exception as exc:
            return _failure("model_fit_failed", exc)
        validation = _validate_estimator(model)
        if isinstance(validation, dict):
            return validation
        artifact_path = _artifact_path(request.get("artifact_name"))
        try:
            joblib.dump(model, artifact_path, compress=0, protocol=5)
        except OSError:
            raise
        except Exception as exc:
            return _failure("model_serialization_failed", exc)
        return {
            "ok": True,
            "inference_columns": list(validation),
            "serialization_policy": SERIALIZATION_POLICY,
            "package_versions": _package_versions(),
        }

    def _init_prediction(
        self,
        request: dict[str, Any],
        code: str,
    ) -> dict[str, Any]:
        target_names = _string_list(request.get("target_names"), "target_names")
        if self._prediction_state is not None:
            raise RuntimeError("Prediction worker was initialized more than once.")
        _load_submitted_module(
            code,
            expected_source_hash=str(request.get("expected_source_hash") or ""),
            allowed_imports=tuple(
                str(value) for value in request.get("allowed_imports", ())
            ),
        )
        artifact_path = _artifact_path(request.get("artifact_name"))
        expected_hash = str(request.get("expected_artifact_hash") or "")
        if _sha256_file(artifact_path) != expected_hash:
            raise RuntimeError("Staged artifact hash does not match registry metadata.")
        self._prediction_state = _PredictionState(
            artifact_path=artifact_path,
            target_names=target_names,
        )
        return {
            "ok": True,
            "serialization_policy": SERIALIZATION_POLICY,
            "package_versions": _package_versions(),
        }

    def _predict_batch(self, request: dict[str, Any]) -> dict[str, Any]:
        state = self._require_prediction_state()
        self._verify_artifact_hash(request, state.artifact_path)
        X = read_dataframe(Path.cwd(), request.get("X"))
        expected_columns = request.get("expected_inference_columns")
        if expected_columns is not None:
            expected_columns = _string_list(
                expected_columns,
                "expected_inference_columns",
            )
        prediction_error_code = str(
            request.get("prediction_error_code")
            or "model_prediction_validation_failed"
        )
        if prediction_error_code not in {
            "model_prediction_validation_failed",
            "temporal_probe_rejected",
        }:
            raise ValueError("Worker request has an invalid prediction error code.")
        predicted = _predict_once(
            state.artifact_path,
            X=X,
            target_names=state.target_names,
            expected_columns=expected_columns,
            prediction_error_code=prediction_error_code,
        )
        if isinstance(predicted, dict):
            return predicted
        frame, inference_columns = predicted
        return {
            "ok": True,
            "predictions": write_dataframe(Path.cwd() / "predictions.arrow", frame),
            "inference_columns": inference_columns,
        }

    def _require_prediction_state(self) -> _PredictionState:
        if self._prediction_state is None:
            raise RuntimeError("Prediction worker has not been initialized.")
        return self._prediction_state

    @staticmethod
    def _verify_artifact_hash(request: dict[str, Any], artifact_path: Path) -> None:
        expected_hash = str(request.get("expected_artifact_hash") or "")
        if _sha256_file(artifact_path) != expected_hash:
            raise RuntimeError("Staged artifact hash does not match trusted metadata.")


def _predict_once(
    artifact_path: Path,
    *,
    X: pd.DataFrame,
    target_names: list[str],
    expected_columns: list[str] | None,
    prediction_error_code: str,
) -> tuple[pd.DataFrame, list[str]] | dict[str, Any]:
    try:
        model = joblib.load(artifact_path)
    except OSError:
        raise
    except Exception as exc:
        return _failure("model_deserialization_failed", exc)
    validation = _validate_estimator(model)
    if isinstance(validation, dict):
        return validation
    columns = list(validation)
    if expected_columns is not None and columns != expected_columns:
        return _failure(
            "inference_schema_changed",
            ValueError("Estimator feature_names_in_ changed after smoke validation."),
        )
    missing = [name for name in columns if name not in X.columns]
    if missing:
        return _failure(
            "inference_columns_missing",
            ValueError(f"Prediction data is missing inference column(s): {', '.join(missing)}"),
        )

    prediction: pd.DataFrame | None = None
    prediction_error: Exception | None = None
    try:
        raw = model.predict(X.loc[:, columns].copy(deep=True))
        prediction = canonical_prediction_frame(raw, X=X, target_names=target_names)
    except Exception as exc:
        prediction_error = exc
    if prediction_error is not None:
        return _failure(prediction_error_code, prediction_error)
    assert prediction is not None
    return prediction, columns


def _validate_estimator(model: Any) -> tuple[str, ...] | dict[str, Any]:
    if not isinstance(model, BaseEstimator):
        return _failure(
            "invalid_fitted_estimator",
            ValueError("train_model(X, y) must return one sklearn BaseEstimator."),
        )
    try:
        check_is_fitted(model)
    except Exception as exc:
        return _failure("invalid_fitted_estimator", exc)
    actual = getattr(model, "feature_names_in_", None)
    values = np.asarray(actual, dtype=object) if actual is not None else np.asarray([])
    if (
        values.ndim != 1
        or not len(values)
        or not all(isinstance(value, str) and value for value in values.tolist())
        or len(set(values.tolist())) != len(values)
    ):
        return _failure(
            "invalid_fitted_estimator",
            ValueError(
                "Fitted estimator must expose unique ordered string feature_names_in_."
            ),
        )
    return tuple(str(value) for value in values.tolist())


def _artifact_path(value: Any) -> Path:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError("Worker artifact name must be a private local filename.")
    return Path.cwd() / value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Worker request {name} must be a non-empty string list.")
    if len(set(value)) != len(value):
        raise ValueError(f"Worker request {name} must contain unique values.")
    return list(value)


def _package_versions() -> dict[str, str]:
    return {
        "python": __import__("platform").python_version(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }


def _load_submitted_module(
    code: str,
    *,
    expected_source_hash: str,
    allowed_imports: tuple[str, ...],
) -> types.ModuleType:
    source_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
    if source_hash != expected_source_hash:
        raise RuntimeError("Submitted model source hash does not match trusted metadata.")
    module_name = submitted_module_name(source_hash)
    module = types.ModuleType(module_name)
    module.__dict__.update(
        {
            "__builtins__": {
                **SAFE_BUILTINS,
                "__import__": _restricted_import(allowed_imports),
            },
            "math": math,
        }
    )
    sys.modules[module_name] = module
    try:
        compiled = compile(code, "<submitted_feature_model>", "exec")
        exec(compiled, module.__dict__, module.__dict__)
    except Exception as exc:
        raise ModelCodeExecutionFailed(f"{type(exc).__name__}: {exc}") from exc
    return module


def _restricted_import(allowed_imports: tuple[str, ...]):
    def import_allowed(
        name: str,
        globals_: dict[str, Any] | None = None,
        locals_: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        del globals_, locals_
        if level:
            raise ImportError("relative imports are not allowed")
        if not any(
            name == allowed or name.startswith(f"{allowed}.")
            for allowed in allowed_imports
        ):
            raise ImportError(f"import {name!r} is not allowed")
        module = __import__(name, fromlist=fromlist, level=level)
        _patch_library_file_io(module)
        return module

    return import_allowed


def _patch_library_file_io(module: Any) -> None:
    root = str(getattr(module, "__name__", "")).split(".", 1)[0]
    if root == "pandas":
        for name in (
            "ExcelFile", "HDFStore", "read_clipboard", "read_csv", "read_excel",
            "read_feather", "read_fwf", "read_gbq", "read_hdf", "read_html",
            "read_json", "read_orc", "read_parquet", "read_pickle", "read_sas",
            "read_spss", "read_sql", "read_sql_query", "read_sql_table", "read_stata",
            "read_table", "read_xml", "to_pickle",
        ):
            if hasattr(pd, name):
                setattr(pd, name, _blocked_file_io)
        for pandas_type in (pd.DataFrame, pd.Series):
            for method_name in (
                "to_csv", "to_excel", "to_feather", "to_hdf", "to_json",
                "to_parquet", "to_pickle", "to_sql", "to_stata",
            ):
                if hasattr(pandas_type, method_name):
                    setattr(pandas_type, method_name, _blocked_file_io)
        if hasattr(pd, "io") and hasattr(pd.io, "common"):
            pd.io.common.get_handle = _blocked_file_io
    elif root == "numpy":
        for name in (
            "fromfile", "genfromtxt", "load", "loadtxt", "memmap", "save",
            "savetxt", "savez", "savez_compressed",
        ):
            if hasattr(np, name):
                setattr(np, name, _blocked_file_io)
    elif root == "sklearn":
        try:
            import sklearn.datasets as datasets
        except ImportError:
            return
        for name in (
            "fetch_20newsgroups", "fetch_california_housing", "fetch_covtype",
            "fetch_kddcup99", "fetch_lfw_pairs", "fetch_lfw_people",
            "fetch_olivetti_faces", "fetch_openml", "fetch_rcv1",
            "fetch_species_distributions", "load_files", "load_svmlight_file",
            "load_svmlight_files",
        ):
            if hasattr(datasets, name):
                setattr(datasets, name, _blocked_file_io)


def _blocked_file_io(*args: Any, **kwargs: Any) -> None:
    del args, kwargs
    raise FileIOBlocked("Submitted model code may not read or write files.")


def _failure(
    error_code: str,
    exc: Exception,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": False,
        "error_code": error_code,
        "message": f"{type(exc).__name__}: {exc}"[:500],
        "recoverable": True,
    }
    if details is not None:
        payload["details"] = details
    return payload


def _error(error_code: str, exc: Exception) -> dict[str, Any]:
    return {"ok": False, "error": _failure(error_code, exc)}


def _internal_error(exc: Exception) -> dict[str, Any]:
    return {
        "ok": False,
        "error": {
            "ok": False,
            "error_code": "worker_internal_error",
            "message": f"{type(exc).__name__}: {exc}"[:500],
            "recoverable": False,
        },
    }


def build_worker_handler() -> FeatureEngineeringWorkerHandler:
    return FeatureEngineeringWorkerHandler()


if __name__ == "__main__":  # pragma: no cover
    WorkerProgram(build_worker_handler()).run()
