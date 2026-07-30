"""Rollout state machine for the feature-engineering task."""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from evalenv_shared.worker.process import SubprocessWorkerHost
from feature_engineering.config import TaskConfig, load_task_config
from feature_engineering.core.backtest import run_backtest
from feature_engineering.core.fixed_data import (
    SupervisedData,
    load_supervised_data,
)
from feature_engineering.core.granularity import granularity_delta
from feature_engineering.runtime.protocol import (
    BacktestRequest,
    DatetimeFilter,
    ProtocolError,
    SubmitStrategyRequest,
    TrainModelRequest,
)
from feature_engineering.submissions.bundle import (
    promote_submission_bundle,
    remove_submission_bundle,
)
from feature_engineering.submissions.causal_audit import (
    validate_fixed_prediction_window,
)
from feature_engineering.submissions.modeling import (
    evaluate_model_code_async,
)
from feature_engineering.submissions.registry import TrainedModelRegistry
from feature_engineering.submissions.strategy import (
    CompiledStrategy,
    StrategyError,
    compile_model_strategy,
)

_CAUSAL_VALIDATION_ERROR_CODES = frozenset(
    {
        "temporal_audit_input_insufficient",
        "temporal_batch_dependency_detected",
        "temporal_probe_rejected",
    }
)

_LOGGER = logging.getLogger(__name__)

_INTERNAL_TRAINING_DIAGNOSTICS = frozenset(
    {"fit_forecast_beta", "forecast_scale"}
)


def _model_visible_training_diagnostics(
    diagnostics: dict[str, float],
) -> dict[str, float]:
    return {
        key: value
        for key, value in diagnostics.items()
        if key not in _INTERNAL_TRAINING_DIAGNOSTICS
    }


@dataclass(slots=True)
class StrategyAttempt:
    attempt_id: str
    strategy_id: str | None
    model_id: str | None
    label: str | None
    ok: bool
    strategy_settings: dict[str, Any] | None
    strategy_settings_hash: str | None
    metrics: dict[str, Any] | None = None
    causal_audit: dict[str, Any] | None = None
    filter: dict[str, str] | None = None
    error: dict[str, Any] | None = None

    def model_summary(self) -> dict[str, Any]:
        payload = {
            "attempt_id": self.attempt_id,
            "strategy_id": self.strategy_id,
            "model_id": self.model_id,
            "label": self.label,
            "ok": self.ok,
        }
        if self.metrics is not None:
            payload["metrics"] = self.metrics
        if self.filter is not None:
            payload["filter"] = dict(self.filter)
        if self.error is not None:
            payload["error"] = _model_visible_error(self.error)
        return payload


@dataclass(slots=True)
class ModelAttempt:
    attempt_id: str
    model_id: str | None
    label: str | None
    ok: bool
    model_code_hash: str | None
    model_artifact_hash: str | None
    model_artifact_bytes: int | None = None
    diagnostics: dict[str, float] | None = None
    row_count: int = 0
    feature_names: tuple[str, ...] = ()
    target_names: tuple[str, ...] = ()
    package_versions: tuple[tuple[str, str], ...] = ()
    filter: dict[str, str] | None = None
    error: dict[str, Any] | None = None

    def model_summary(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "attempt_id": self.attempt_id,
            "model_id": self.model_id,
            "label": self.label,
            "ok": self.ok,
            "model_code_hash": self.model_code_hash,
            "row_count": self.row_count,
            "feature_names": list(self.feature_names),
            "target_names": list(self.target_names),
        }
        if self.diagnostics is not None:
            payload["diagnostics"] = _model_visible_training_diagnostics(
                self.diagnostics
            )
        if self.filter is not None:
            payload["filter"] = dict(self.filter)
        if self.error is not None:
            payload["error"] = _model_visible_error(self.error)
        return payload


class OperationInfrastructureError(RuntimeError):
    """Stable failure returned by every query of a broken operation."""

    def __init__(self, operation_id: str, error_code: str, message: str) -> None:
        self.operation_id = operation_id
        self.error_code = error_code
        self.message = message
        super().__init__(message)


@dataclass(slots=True)
class TrainingOperation:
    training_id: str
    attempt_id: str
    request: TrainModelRequest
    start: datetime
    end: datetime
    filter: dict[str, str]
    result: dict[str, Any] | None = None
    failure: tuple[str, str] | None = None
    publication_failure: tuple[str, str] | None = None

    def summary(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "training_id": self.training_id,
            "attempt_id": self.attempt_id,
            "status": "running",
        }
        failure = self.failure or self.publication_failure
        if failure is not None:
            payload["status"] = "infrastructure_failed"
            payload["error_code"] = failure[0]
        elif self.result is not None:
            payload["status"] = "completed"
            payload["outcome"] = str(self.result["status"])
            payload["ok"] = bool(self.result["ok"])
            if "model_id" in self.result:
                payload["model_id"] = self.result["model_id"]
        return payload


@dataclass(slots=True)
class BacktestOperation:
    backtest_id: str
    attempt_id: str
    request: BacktestRequest
    start: datetime
    end: datetime
    filter: dict[str, str]
    strategy: CompiledStrategy | StrategyError
    result: dict[str, Any] | None = None
    failure: tuple[str, str] | None = None
    publication_failure: tuple[str, str] | None = None

    def summary(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "backtest_id": self.backtest_id,
            "attempt_id": self.attempt_id,
            "status": "running",
        }
        failure = self.failure or self.publication_failure
        if failure is not None:
            payload["status"] = "infrastructure_failed"
            payload["error_code"] = failure[0]
        elif self.result is not None:
            payload["status"] = "completed"
            payload["outcome"] = str(self.result["status"])
            payload["ok"] = bool(self.result["ok"])
            if "strategy_id" in self.result:
                payload["strategy_id"] = self.result["strategy_id"]
        return payload


@dataclass(slots=True)
class FeatureEngineeringRuntime:
    config: TaskConfig
    public_data: SupervisedData
    worker_host: Any
    registry: TrainedModelRegistry
    official_scoring_enabled: bool = False
    task_outputs: Path | None = None
    task_name: str = "feature-engineering"
    data_split: str = "train"
    attempts: list[StrategyAttempt] = field(default_factory=list)
    model_attempts: list[ModelAttempt] = field(default_factory=list)
    strategies: dict[str, CompiledStrategy] = field(default_factory=dict)
    official_primary_score: float | None = None
    official_metrics: dict[str, float] = field(default_factory=dict)
    official_scoring_status: dict[str, Any] = field(default_factory=dict)
    last_payload: dict[str, Any] = field(default_factory=dict)
    done: bool = False
    response_errors: int = 0
    research_attempts_consumed: int = 0
    active_training_id: str | None = None
    active_backtest_id: str | None = None
    termination_pending: bool = False
    training_operations: dict[str, TrainingOperation] = field(default_factory=dict)
    backtest_operations: dict[str, BacktestOperation] = field(default_factory=dict)
    _next_training_number: int = 1
    _next_backtest_number: int = 1
    trace: dict[str, Any] = field(
        default_factory=lambda: {
            "attempts": [],
            "model_attempts": [],
            "public_feature_columns": [],
            "research_attempts_consumed": 0,
        }
    )

    @classmethod
    def from_config_path(
        cls,
        path: str | Path,
        *,
        registry: TrainedModelRegistry,
        worker_host: Any = None,
        official_scoring_enabled: bool = False,
        task_outputs: str | Path | None = None,
        task_name: str = "feature-engineering",
        data_split: str = "train",
    ) -> "FeatureEngineeringRuntime":
        config = load_task_config(path)
        public_data = load_supervised_data(config)
        validate_fixed_prediction_window(
            config=config,
            data=public_data,
            start=public_data.start_datetime,
            end=public_data.end_datetime,
        )
        runtime = cls(
            config=config,
            public_data=public_data,
            worker_host=worker_host or SubprocessWorkerHost(),
            registry=registry,
            official_scoring_enabled=official_scoring_enabled,
            task_outputs=Path(task_outputs).resolve() if task_outputs is not None else None,
            task_name=task_name,
            data_split=data_split,
        )
        runtime.trace["public_feature_columns"] = list(public_data.feature_columns)
        runtime.trace["workspace_profile"] = config.workspace.profile
        return runtime

    def record_protocol_error(self, error: ProtocolError) -> dict[str, Any]:
        self._consume_response_error()
        return {
            "type": "error",
            **error.to_payload(research_budget=self._budget_payload()),
        }

    def current_metrics(self, strategy_id: str | None = None) -> dict[str, float]:
        if strategy_id is None:
            submission = self.trace.get("final_submission")
            if isinstance(submission, dict):
                submitted_strategy_id = submission.get("strategy_id")
                if isinstance(submitted_strategy_id, str):
                    strategy_id = submitted_strategy_id

        source = next(
            (
                attempt.metrics
                for attempt in reversed(self.attempts)
                if attempt.ok
                and attempt.metrics is not None
                and (strategy_id is None or attempt.strategy_id == strategy_id)
            ),
            None,
        )
        metrics = dict(source or {})
        metrics.setdefault("annualized_sharpe", 0.0)
        metrics.setdefault("sharpe", metrics["annualized_sharpe"])
        metrics.setdefault("cagr", 0.0)
        metrics.setdefault("max_drawdown", 0.0)
        metrics.setdefault("pearson_ic", metrics.get("correlation", 0.0))
        metrics.setdefault("cumulative_after_cost_return", 0.0)
        metrics["attempt_count"] = float(len(self.attempts))
        metrics["successful_strategy_count"] = float(len(self.strategies))
        metrics["model_attempt_count"] = float(len(self.model_attempts))
        metrics["successful_model_count"] = float(
            sum(attempt.ok for attempt in self.model_attempts)
        )
        metrics["reward"] = float(metrics.get("annualized_sharpe", 0.0))
        return metrics

    def start_training(self, request: TrainModelRequest) -> dict[str, Any]:
        """Validate and reserve one training operation without starting it."""

        if self.active_training_id is not None:
            return self._active_operation_error(
                "training_already_running", "training_id", self.active_training_id
            )
        blocked = self._new_work_blocked_error()
        if blocked is not None:
            return blocked
        window = self._public_filter_window(request.train_filter)
        if isinstance(window, ProtocolError):
            return self.record_protocol_error(window)
        if self._research_budget_exhausted():
            return self._research_budget_error("train_model")

        number = self._next_training_number
        self._next_training_number += 1
        training_id = f"training_{number:03d}"
        attempt_id = f"model_attempt_{number:03d}"
        start, end = window
        filter_payload = _filter_payload(start, end)
        self._consume_research_attempt()
        self.active_training_id = training_id
        self.training_operations[training_id] = TrainingOperation(
            training_id=training_id,
            attempt_id=attempt_id,
            request=request,
            start=start,
            end=end,
            filter=filter_payload,
        )
        return {
            "type": "training_started",
            "ok": True,
            "training_id": training_id,
            "status": "running",
            "research_budget": self._budget_payload(),
        }

    async def execute_training(self, training_id: str) -> None:
        operation = self.training_operations[training_id]
        try:
            result = await evaluate_model_code_async(
                config=self.config,
                public_data=self.public_data,
                registry=self.registry,
                model_code=operation.request.model_code,
                training_filter=operation.filter,
                start=operation.start,
                end=operation.end,
                timeout_seconds=self.config.execution.timeout_seconds,
                worker_host=self.worker_host,
            )
            if result.error is not None or result.model is None:
                error = result.error.to_payload() if result.error else {"ok": False}
                attempt = ModelAttempt(
                    attempt_id=operation.attempt_id,
                    model_id=None,
                    label=operation.request.label,
                    ok=False,
                    model_code_hash=None,
                    model_artifact_hash=None,
                    row_count=result.row_count,
                    feature_names=result.feature_names,
                    target_names=result.target_names,
                    filter=operation.filter,
                    error=error,
                )
                payload = {
                    "type": "training_result",
                    "ok": False,
                    "training_id": training_id,
                    "status": "failed",
                    "attempt_id": operation.attempt_id,
                    **_model_visible_error(error),
                    "row_count": result.row_count,
                    "feature_names": list(result.feature_names),
                    "target_names": list(result.target_names),
                    "filter": dict(operation.filter),
                }
            else:
                model = result.model
                attempt = ModelAttempt(
                    attempt_id=operation.attempt_id,
                    model_id=model.model_id,
                    label=operation.request.label,
                    ok=True,
                    model_code_hash=model.model_code_sha256,
                    model_artifact_hash=model.artifact_sha256,
                    model_artifact_bytes=model.artifact_bytes,
                    diagnostics=result.diagnostics,
                    row_count=result.row_count,
                    feature_names=result.feature_names,
                    target_names=result.target_names,
                    package_versions=model.package_versions,
                    filter=operation.filter,
                )
                payload = {
                    "type": "training_result",
                    "ok": True,
                    "training_id": training_id,
                    "status": "succeeded",
                    "attempt_id": operation.attempt_id,
                    "model_id": model.model_id,
                    "diagnostics": _model_visible_training_diagnostics(
                        result.diagnostics
                    ),
                    "row_count": result.row_count,
                    "feature_names": list(result.feature_names),
                    "target_names": list(result.target_names),
                    "filter": dict(operation.filter),
                }
            self._record_model_attempt(attempt)
            operation.result = payload
        except Exception:
            _LOGGER.exception("Training operation %s failed", training_id)
            operation.failure = (
                "training_infrastructure_failure",
                "Training failed because of an environment infrastructure error.",
            )
        finally:
            # Commit and slot release are one local transition; state-channel
            # publication happens later and cannot change this outcome.
            if operation.result is not None or operation.failure is not None:
                if self.active_training_id != training_id:
                    raise RuntimeError("Training completion does not own the active slot.")
                self.active_training_id = None
                self._finish_pending_termination_if_idle()

    def get_train_model_result(self, training_id: str) -> dict[str, Any]:
        operation = self.training_operations.get(training_id)
        if operation is None:
            return self._unknown_operation_error("training", training_id)
        if operation.failure is not None:
            raise OperationInfrastructureError(training_id, *operation.failure)
        if operation.publication_failure is not None:
            raise OperationInfrastructureError(
                training_id, *operation.publication_failure
            )
        if operation.result is None:
            return {
                "type": "training_status",
                "ok": True,
                "training_id": training_id,
                "status": "running",
            }
        return {**deepcopy(operation.result), "research_budget": self._budget_payload()}

    def start_backtest(self, request: BacktestRequest) -> dict[str, Any]:
        """Validate and reserve one backtest operation without starting it."""

        if self.active_backtest_id is not None:
            return self._active_operation_error(
                "backtest_already_running", "backtest_id", self.active_backtest_id
            )
        blocked = self._new_work_blocked_error()
        if blocked is not None:
            return blocked
        window = self._public_filter_window(request.backtest_filter)
        if isinstance(window, ProtocolError):
            return self.record_protocol_error(window)
        available_model_ids = sorted(
            str(operation.result["model_id"])
            for operation in self.training_operations.values()
            if operation.result is not None
            and operation.result.get("status") == "succeeded"
        )
        if request.model_id not in available_model_ids:
            self._consume_response_error()
            return {
                "type": "error",
                "ok": False,
                "error_code": "unknown_model_id",
                "message": "Backtest model_id must refer to a successful train_model action.",
                "recoverable": True,
                "model_id": request.model_id,
                "available_model_ids": available_model_ids,
                "research_budget": self._budget_payload(),
            }
        self.registry.get(request.model_id)
        if self._research_budget_exhausted():
            return self._research_budget_error("backtest")

        compiled: CompiledStrategy | StrategyError = compile_model_strategy(
            model_id=request.model_id,
            max_gross_exposure=request.max_gross_exposure,
        )
        number = self._next_backtest_number
        self._next_backtest_number += 1
        backtest_id = f"backtest_{number:03d}"
        attempt_id = f"attempt_{number:03d}"
        filter_payload = _filter_payload(*window)
        self._consume_research_attempt()
        self.active_backtest_id = backtest_id
        self.backtest_operations[backtest_id] = BacktestOperation(
            backtest_id=backtest_id,
            attempt_id=attempt_id,
            request=request,
            start=window[0],
            end=window[1],
            filter=filter_payload,
            strategy=compiled,
        )
        return {
            "type": "backtest_started",
            "ok": True,
            "backtest_id": backtest_id,
            "status": "running",
            "research_budget": self._budget_payload(),
        }

    async def execute_backtest(self, backtest_id: str) -> None:
        operation = self.backtest_operations[backtest_id]
        try:
            if isinstance(operation.strategy, StrategyError):
                compiled = operation.strategy
                attempt = StrategyAttempt(
                    attempt_id=operation.attempt_id,
                    strategy_id=None,
                    model_id=operation.request.model_id,
                    label=operation.request.label,
                    ok=False,
                    strategy_settings={
                        "max_gross_exposure": operation.request.max_gross_exposure
                    },
                    strategy_settings_hash=None,
                    filter=operation.filter,
                    error=compiled.to_payload(),
                )
                self._record_attempt(attempt)
                operation.result = {
                    "type": "backtest_result",
                    "ok": False,
                    "backtest_id": backtest_id,
                    "status": "failed",
                    "attempt_id": operation.attempt_id,
                    **compiled.to_payload(),
                    "filter": dict(operation.filter),
                }
                return

            compiled = operation.strategy
            backtest = await run_backtest(
                config=self.config,
                public_data=self.public_data,
                strategy=compiled,
                registry=self.registry,
                start=operation.start,
                end=operation.end,
                worker_host=self.worker_host,
            )
            if backtest.ok:
                strategy_id = f"strategy_{len(self.strategies) + 1:03d}"
                self.strategies[strategy_id] = compiled
                attempt = StrategyAttempt(
                    attempt_id=operation.attempt_id,
                    strategy_id=strategy_id,
                    model_id=operation.request.model_id,
                    label=operation.request.label,
                    ok=True,
                    strategy_settings=compiled.settings,
                    strategy_settings_hash=compiled.settings_hash,
                    metrics=backtest.model_visible["metrics"],
                    causal_audit=backtest.audit,
                    filter=operation.filter,
                )
                self._record_attempt(attempt, trace=backtest.trace)
                operation.result = {
                    "type": "backtest_result",
                    "ok": True,
                    "backtest_id": backtest_id,
                    "status": "succeeded",
                    "attempt_id": operation.attempt_id,
                    "strategy_id": strategy_id,
                    **backtest.model_visible,
                    "filter": dict(operation.filter),
                }
                return

            error_payload = backtest.model_visible
            attempt = StrategyAttempt(
                attempt_id=operation.attempt_id,
                strategy_id=None,
                model_id=operation.request.model_id,
                label=operation.request.label,
                ok=False,
                strategy_settings=compiled.settings,
                strategy_settings_hash=compiled.settings_hash,
                filter=operation.filter,
                error=error_payload,
                causal_audit=backtest.audit,
            )
            self._record_attempt(attempt, trace=backtest.trace)
            operation.result = {
                "type": "backtest_result",
                "ok": False,
                "backtest_id": backtest_id,
                "status": "failed",
                "attempt_id": operation.attempt_id,
                **_model_visible_error(error_payload),
                "filter": dict(operation.filter),
            }
        except Exception:
            _LOGGER.exception("Backtest operation %s failed", backtest_id)
            operation.failure = (
                "backtest_infrastructure_failure",
                "Backtest failed because of an environment infrastructure error.",
            )
        finally:
            # Commit and slot release are one local transition; state-channel
            # publication happens later and cannot change this outcome.
            if operation.result is not None or operation.failure is not None:
                if self.active_backtest_id != backtest_id:
                    raise RuntimeError("Backtest completion does not own the active slot.")
                self.active_backtest_id = None
                self._finish_pending_termination_if_idle()

    def get_backtest_result(self, backtest_id: str) -> dict[str, Any]:
        operation = self.backtest_operations.get(backtest_id)
        if operation is None:
            return self._unknown_operation_error("backtest", backtest_id)
        if operation.failure is not None:
            raise OperationInfrastructureError(backtest_id, *operation.failure)
        if operation.publication_failure is not None:
            raise OperationInfrastructureError(
                backtest_id, *operation.publication_failure
            )
        if operation.result is None:
            return {
                "type": "backtest_status",
                "ok": True,
                "backtest_id": backtest_id,
                "status": "running",
            }
        return {**deepcopy(operation.result), "research_budget": self._budget_payload()}

    def record_publication_failure(self, kind: str, operation_id: str) -> None:
        operations = (
            self.training_operations if kind == "training" else self.backtest_operations
        )
        operation = operations[operation_id]
        failure = (
            f"{kind}_state_publication_failed",
            f"{kind.title()} state publication failed.",
        )
        operation.publication_failure = failure
        if operation.result is None and operation.failure is None:
            operation.failure = failure
            active_field = f"active_{kind}_id"
            if getattr(self, active_field) != operation_id:
                raise RuntimeError(f"{kind.title()} does not own the active slot.")
            setattr(self, active_field, None)
            self._finish_pending_termination_if_idle()

    async def submit_strategy(
        self,
        request: SubmitStrategyRequest,
    ) -> dict[str, Any]:
        strategy = self.strategies.get(request.strategy_id)
        if strategy is None:
            self._consume_response_error()
            return {
                "type": "error",
                "ok": False,
                "error_code": "unknown_strategy_id",
                "message": (
                    "Strategy not found. It may still be in backtest; query the "
                    "corresponding backtest_id before submitting."
                ),
                "recoverable": True,
                "strategy_id": request.strategy_id,
                "available_strategy_ids": sorted(self.strategies),
                "research_budget": self._budget_payload(),
            }
        if self.active_training_id is not None or self.active_backtest_id is not None:
            return {
                "type": "error",
                "ok": False,
                "error_code": "operations_still_running",
                "message": "Wait for active training and backtest operations to finish.",
                "recoverable": True,
                "active_training_id": self.active_training_id,
                "active_backtest_id": self.active_backtest_id,
                "research_budget": self._budget_payload(),
            }
        submission: dict[str, Any] = {
            "strategy_id": request.strategy_id,
            "model_id": strategy.model_id,
            "strategy_settings_hash": strategy.settings_hash,
            "rationale": request.rationale,
            "metrics_at_submission": self.current_metrics(request.strategy_id),
        }
        official_fit_diagnostics: dict[str, float] | None = None
        try:
            if self.official_scoring_enabled:
                from feature_engineering.scoring.official import (
                    OfficialSubmittedCodeError,
                    score_official_strategy,
                )

                try:
                    official = await score_official_strategy(
                        public_config=self.config,
                        public_data=self.public_data,
                        strategy=strategy,
                        registry=self.registry,
                        worker_host=self.worker_host,
                    )
                except OfficialSubmittedCodeError as exc:
                    return self._reject_official_submission(
                        request=request,
                        strategy=strategy,
                        submission=submission,
                        error_code=exc.error_code,
                        audit=exc.audit,
                        fit_diagnostics=exc.fit_diagnostics,
                        reason="hidden_execution_failed",
                    )
                if not official.get("ok", False):
                    return self._reject_official_submission(
                        request=request,
                        strategy=strategy,
                        submission=submission,
                        error_code=str(
                            official.get("error_code", "official_scoring_failed")
                        ),
                        audit=official.get("causal_audit"),
                        fit_diagnostics=official.get("fit_diagnostics"),
                        reason="official_scoring_failed",
                    )
                self.official_primary_score = float(official["primary_score"])
                self.official_metrics = dict(official["metrics"])
                official_fit_diagnostics = dict(official["fit_diagnostics"])
                submission["causal_audit"] = dict(official.get("causal_audit") or {})
                self.official_scoring_status = {
                    "ok": True,
                    "isolation": "full_public_refit_then_source_aware_hidden_batch",
                    "scoring_components": list(official["scoring_components"]),
                }
                official_scoring = {
                    "status": "succeeded",
                    "isolation": "full_public_refit_then_source_aware_hidden_batch",
                }
            else:
                self.official_scoring_status = {
                    "ok": False,
                    "skipped": True,
                    "reason": "official_scoring_disabled",
                }
                official_scoring = {
                    "status": "skipped",
                    "reason": "official_scoring_disabled",
                }
            submission["accepted_for_official_scoring"] = True
            bundle_path, projection = self._promote_submission(
                request=request,
                strategy=strategy,
                accepted=True,
                official_scoring=official_scoring,
                hidden_audit=(
                    _hidden_audit_projection(official.get("causal_audit") or {})
                    if self.official_scoring_enabled
                    else None
                ),
                fit_diagnostics=official_fit_diagnostics,
            )
            submission.update(
                {
                    "submission_id": projection["submission_id"],
                    "accepted": True,
                    "bundle_path": bundle_path,
                    "manifest": projection,
                }
            )
            try:
                self.trace["final_submission"] = submission
            except BaseException:
                remove_submission_bundle(self.task_outputs, bundle_path)
                raise
            self.done = True
            return {
                "type": "final_submission_result",
                "ok": True,
                "action": "submit_strategy",
                "strategy_id": request.strategy_id,
                "accepted": True,
                "done": True,
            }
        finally:
            self.registry.close()

    def _reject_official_submission(
        self,
        *,
        request: SubmitStrategyRequest,
        strategy: CompiledStrategy,
        submission: dict[str, Any],
        error_code: str,
        audit: dict[str, Any] | None,
        fit_diagnostics: dict[str, Any] | None,
        reason: str,
        official_scoring_extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist a terminal rejected submission without exposing hidden metrics."""

        hidden_audit = _hidden_audit_projection(
            dict(audit or {}), error_code=error_code
        )
        official_scoring = {
            "status": "rejected",
            "error_code": error_code,
            **dict(official_scoring_extra or {}),
        }
        submission["accepted_for_official_scoring"] = False
        submission["rejection_reason"] = error_code
        submission["causal_audit"] = hidden_audit
        self.official_primary_score = 0.0
        self.official_metrics = {}
        self.official_scoring_status = {
            "ok": False,
            "submitted_code_failure": reason == "hidden_execution_failed",
            "reason": reason,
        }
        if official_scoring_extra:
            self.official_scoring_status.update(official_scoring_extra)
        bundle_path, projection = self._promote_submission(
            request=request,
            strategy=strategy,
            accepted=False,
            official_scoring=official_scoring,
            hidden_audit=hidden_audit,
            fit_diagnostics=fit_diagnostics,
        )
        submission.update(
            {
                "submission_id": projection["submission_id"],
                "accepted": False,
                "bundle_path": bundle_path,
                "manifest": projection,
            }
        )
        try:
            self.trace["final_submission"] = submission
        except BaseException:
            remove_submission_bundle(self.task_outputs, bundle_path)
            raise
        self.done = True
        return {
            "type": "final_submission_result",
            "ok": False,
            "action": "submit_strategy",
            "strategy_id": request.strategy_id,
            "accepted": False,
            "done": True,
            "recoverable": False,
            "message": "The submitted strategy could not be accepted.",
        }

    def _promote_submission(
        self,
        *,
        request: SubmitStrategyRequest,
        strategy: CompiledStrategy,
        accepted: bool,
        official_scoring: dict[str, Any],
        hidden_audit: dict[str, Any] | None,
        fit_diagnostics: dict[str, float] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        model = self.registry.get(strategy.model_id)
        public_attempt = next(
            attempt
            for attempt in reversed(self.attempts)
            if attempt.strategy_id == request.strategy_id and attempt.ok
        )
        model_attempt = next(
            attempt
            for attempt in reversed(self.model_attempts)
            if attempt.model_id == strategy.model_id and attempt.ok
        )
        return promote_submission_bundle(
            task_outputs=self.task_outputs,
            task_name=self.task_name,
            data_split=self.data_split,
            registry=self.registry,
            model=model,
            strategy_id=request.strategy_id,
            strategy_settings=strategy.settings,
            strategy_hash=strategy.settings_hash,
            rationale=request.rationale,
            public_metrics=public_attempt.metrics or {},
            public_filter=public_attempt.filter,
            public_audit=public_attempt.causal_audit,
            fit_diagnostics=(
                fit_diagnostics
                if fit_diagnostics is not None
                else model_attempt.diagnostics or {}
            ),
            accepted=accepted,
            official_scoring=official_scoring,
            hidden_audit=hidden_audit,
        )

    def projection(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Build the fields for one authoritative state publication."""

        done = self.done or (
            self.termination_pending
            and self.active_training_id is None
            and self.active_backtest_id is None
        )
        metrics = self.current_metrics()
        reward = float(
            self.official_primary_score
            if self.official_primary_score is not None
            else metrics.get("reward", 0.0)
        )
        return {
            "done": done,
            "reward": reward,
            "metrics": metrics,
            "trace_data": deepcopy(self.trace),
            "last_payload": deepcopy(payload),
            "official_primary_score": self.official_primary_score,
            "official_metrics": dict(self.official_metrics),
            "official_scoring_status": dict(self.official_scoring_status),
            "active_training_id": self.active_training_id,
            "active_backtest_id": self.active_backtest_id,
            "training_operations": {
                key: operation.summary()
                for key, operation in self.training_operations.items()
            },
            "backtest_operations": {
                key: operation.summary()
                for key, operation in self.backtest_operations.items()
            },
            "research_attempts_consumed": self.research_attempts_consumed,
            "termination_pending": self.termination_pending,
        }

    def record_publication(self, payload: dict[str, Any]) -> None:
        self.last_payload = deepcopy(payload)

    def _research_budget_exhausted(self) -> bool:
        return (
            self.research_attempts_consumed >= self.config.agent.max_research_attempts
        )

    def _research_budget_error(self, action: str) -> dict[str, Any]:
        self._consume_response_error()
        return {
            "type": "error",
            "ok": False,
            "error_code": "research_attempt_budget_exhausted",
            "message": f"{action} research attempt budget is exhausted.",
            "recoverable": True,
            "research_budget": self._budget_payload(),
        }

    def _new_work_blocked_error(self) -> dict[str, Any] | None:
        if not self.done and not self.termination_pending:
            return None
        return {
            "type": "error",
            "ok": False,
            "error_code": "rollout_terminating",
            "message": "No new research operation may start while the rollout terminates.",
            "recoverable": False,
            "research_budget": self._budget_payload(),
        }

    def _active_operation_error(
        self,
        error_code: str,
        id_field: str,
        operation_id: str,
    ) -> dict[str, Any]:
        return {
            "type": "error",
            "ok": False,
            "error_code": error_code,
            "message": f"Operation {operation_id} is already running.",
            "recoverable": True,
            id_field: operation_id,
            "research_budget": self._budget_payload(),
        }

    def _unknown_operation_error(
        self,
        operation_type: str,
        operation_id: str,
    ) -> dict[str, Any]:
        self._consume_response_error()
        return {
            "type": "error",
            "ok": False,
            "error_code": f"unknown_{operation_type}_id",
            "message": f"Unknown {operation_type} operation: {operation_id}.",
            "recoverable": True,
            f"{operation_type}_id": operation_id,
            "research_budget": self._budget_payload(),
        }

    def _consume_research_attempt(self) -> None:
        self.research_attempts_consumed += 1
        self.trace["research_attempts_consumed"] = self.research_attempts_consumed

    def _consume_response_error(self) -> None:
        self.response_errors += 1
        if self.response_errors < self.config.agent.response_error_budget:
            return
        if self.active_training_id is not None or self.active_backtest_id is not None:
            self.termination_pending = True
        else:
            self.done = True

    def _finish_pending_termination_if_idle(self) -> None:
        if (
            self.termination_pending
            and self.active_training_id is None
            and self.active_backtest_id is None
        ):
            self.done = True

    def _public_filter_window(
        self,
        datetime_filter: DatetimeFilter | None,
    ) -> tuple[datetime, datetime] | ProtocolError:
        public_start = self.public_data.start_datetime
        public_end = self.public_data.end_datetime
        start = (
            datetime_filter.start_datetime
            if datetime_filter and datetime_filter.start_datetime is not None
            else public_start
        )
        end = (
            datetime_filter.end_datetime
            if datetime_filter and datetime_filter.end_datetime is not None
            else public_end
        )
        period = granularity_delta(self.config.data.granularity)
        if (
            start < public_start
            or end > public_end
            or end <= start
            or (start - public_start) % period
            or (end - public_start) % period
        ):
            return ProtocolError(
                error_code="filter_out_of_public_range",
                message=(
                    "Datetime filters must stay inside the public dataset window "
                    "and have end_datetime after start_datetime."
                ),
                details={"public": _filter_payload(public_start, public_end)},
            )
        return start, end

    def _budget_payload(self) -> dict[str, int]:
        return {
            "remaining_research_attempts": max(
                0,
                self.config.agent.max_research_attempts
                - self.research_attempts_consumed,
            ),
            "response_errors_remaining": max(
                0,
                self.config.agent.response_error_budget - self.response_errors,
            ),
        }

    def _record_attempt(
        self,
        attempt: StrategyAttempt,
        trace: dict[str, Any] | None = None,
    ) -> None:
        self.attempts.append(attempt)
        payload = {
            **attempt.model_summary(),
            "strategy_settings": attempt.strategy_settings,
            "strategy_settings_hash": attempt.strategy_settings_hash,
            "research_budget": self._budget_payload(),
        }
        if trace is not None:
            # Keep transient Verifiers state bounded; detailed traces stay local
            # to the backtest result and must not cross the state bridge.
            payload["trace_summary"] = {
                "step_count": len(trace.get("steps", [])),
                "prediction_record_count": len(trace.get("prediction_records", [])),
            }
        self.trace["attempts"].append(payload)

    def _record_model_attempt(
        self,
        attempt: ModelAttempt,
    ) -> None:
        self.model_attempts.append(attempt)
        self.trace["model_attempts"].append(
            {
                **attempt.model_summary(),
                "model_artifact_hash": attempt.model_artifact_hash,
                "model_artifact_bytes": attempt.model_artifact_bytes,
                "package_versions": dict(attempt.package_versions),
            }
        )


def _filter_payload(start: datetime, end: datetime) -> dict[str, str]:
    return {
        "start_datetime": start.isoformat(),
        "end_datetime": end.isoformat(),
    }


def _audit_from_error(error: dict[str, Any]) -> dict[str, Any] | None:
    details = error.get("details")
    if not isinstance(details, dict):
        return None
    audit = details.get("causal_audit")
    return dict(audit) if isinstance(audit, dict) else None


def _model_visible_error(error: dict[str, Any]) -> dict[str, Any]:
    visible = dict(error)
    details = visible.get("details")
    if isinstance(details, dict) and "causal_audit" in details:
        remaining_details = {
            key: value for key, value in details.items() if key != "causal_audit"
        }
        if remaining_details:
            visible["details"] = remaining_details
        else:
            visible.pop("details")

    if visible.get("error_code") in _CAUSAL_VALIDATION_ERROR_CODES:
        visible.pop("details", None)
        visible["error_code"] = "causal_validation_failed"
        visible["message"] = (
            "The fitted estimator did not satisfy causal prediction requirements."
        )
    elif visible.get("error_code") == "forecast_scale_calculation_failed":
        visible.pop("details", None)
        visible["error_code"] = "training_validation_failed"
        visible["stage"] = "training_validation"
        visible["message"] = "Training diagnostics were not finite."
    return visible


def _hidden_audit_projection(
    audit: dict[str, Any],
    *,
    error_code: str | None = None,
) -> dict[str, Any]:
    projected = {
        key: audit[key]
        for key in ("policy", "status", "probe_count", "rtol", "atol", "error_code")
        if key in audit
    }
    if error_code is not None:
        projected["error_code"] = error_code
    return projected

