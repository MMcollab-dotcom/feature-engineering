"""Trusted synchronous scoring against one live rollout registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from feature_engineering.config import TaskConfig
from feature_engineering.core.backtest import run_backtest
from feature_engineering.core.fixed_data import SupervisedData
from feature_engineering.scoring.fixed_hidden_data import load_hidden_supervised_data
from feature_engineering.submissions.modeling import evaluate_model_code_async
from feature_engineering.submissions.registry import TrainedModelRegistry
from feature_engineering.submissions.strategy import CompiledStrategy


class OfficialSubmittedCodeError(RuntimeError):
    """Sanitized terminal classification for official submitted-code failures."""

    def __init__(
        self,
        error_code: str,
        audit: dict[str, Any] | None = None,
        fit_diagnostics: dict[str, float] | None = None,
    ) -> None:
        self.error_code = error_code
        self.audit = dict(audit or {})
        self.fit_diagnostics = (
            dict(fit_diagnostics) if fit_diagnostics is not None else None
        )
        super().__init__("Official submitted-code execution failed.")


async def score_official_strategy(
    *,
    public_config: TaskConfig,
    public_data: SupervisedData,
    strategy: CompiledStrategy,
    registry: TrainedModelRegistry,
    worker_host: Any,
) -> dict[str, Any]:
    manifest = json.loads(
        Path(public_config.data.manifest_path).read_text(encoding="utf-8")
    )
    selected_model = registry.get(strategy.model_id)
    full_public_filter = {
        "start_datetime": public_data.start_datetime.isoformat(),
        "end_datetime": public_data.end_datetime.isoformat(),
    }
    refit = await evaluate_model_code_async(
        config=public_config,
        public_data=public_data,
        registry=registry,
        model_code=selected_model.model_code,
        training_filter=full_public_filter,
        timeout_seconds=public_config.execution.timeout_seconds,
        worker_host=worker_host,
        replace_model_id=strategy.model_id,
    )
    if refit.error is not None or refit.model is None:
        if refit.error is not None and refit.error.error_code == "training_data_invalid":
            raise RuntimeError("Trusted full-public training data was invalid.")
        raise OfficialSubmittedCodeError(
            refit.error.error_code if refit.error is not None else "model_refit_failed"
        )
    hidden_data = load_hidden_supervised_data(
        public_config=public_config,
        public_data=public_data,
        scoring_config=None,
    )
    backtest = await run_backtest(
        config=public_config,
        public_data=hidden_data,
        strategy=strategy,
        registry=registry,
        start=manifest["hidden_start_datetime"],
        end=manifest["hidden_end_datetime"],
        audit_visibility="hidden_fixed",
        worker_host=worker_host,
    )
    if not backtest.ok:
        error = backtest.error
        if error is not None and error.contract_failure:
            raise OfficialSubmittedCodeError(
                error.error_code,
                backtest.audit,
                refit.diagnostics,
            )
        raise RuntimeError("Trusted hidden scoring failed outside submitted code.")

    metrics = dict(backtest.metrics)
    scoring_components = [
        "hidden_annualized_sharpe_after_costs",
        "hidden_cagr_after_costs",
        "hidden_max_drawdown",
        "hidden_prediction_correlations",
    ]
    return {
        "ok": True,
        "primary_score": float(metrics["annualized_sharpe"]),
        "metrics": metrics,
        "causal_audit": dict(backtest.audit or {}),
        "fit_diagnostics": dict(refit.diagnostics),
        "scoring_components": scoring_components,
    }


__all__ = ["OfficialSubmittedCodeError", "score_official_strategy"]
