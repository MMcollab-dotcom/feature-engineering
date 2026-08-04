"""Harbor-facing FastMCP server for the fixed feature-engineering task."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from fastmcp.server.lifespan import lifespan

from feature_engineering.runtime.protocol import (
    BacktestRequest,
    DatetimeFilter,
    ProtocolError,
    SubmitStrategyRequest,
    TrainModelRequest,
)
from feature_engineering.runtime.task import FeatureEngineeringRuntime
from feature_engineering.submissions.registry import TrainedModelRegistry

LOGGER = logging.getLogger(__name__)
TASK_CONFIG_PATH = Path(os.environ.get("FEATURE_TASK_CONFIG", "/app/task_config.yaml"))
RUNTIME_ROOT = Path(os.environ.get("FEATURE_RUNTIME_ROOT", "/app/runtime"))
SUBMISSION_ROOT = Path(os.environ.get("FEATURE_SUBMISSION_ROOT", "/app/submission"))


_runtime: FeatureEngineeringRuntime | None = None
_background_tasks: set[asyncio.Task[None]] = set()


_DATETIME_FILTER_FIELDS = frozenset({"start_datetime", "end_datetime"})


def _datetime_filter(
    value: dict[str, str] | None,
    *,
    argument_name: str,
) -> DatetimeFilter | ProtocolError | None:
    if value is None:
        return None
    unknown_fields = sorted(set(value) - _DATETIME_FILTER_FIELDS)
    if unknown_fields:
        return ProtocolError(
            error_code="invalid_datetime_filter_fields",
            message=f"{argument_name} contains unsupported fields: {unknown_fields}.",
            details={
                "unknown_fields": unknown_fields,
                "allowed_fields": sorted(_DATETIME_FILTER_FIELDS),
            },
            suggested_correction=(
                f"Use only start_datetime and end_datetime in {argument_name}."
            ),
        )

    from datetime import datetime

    parsed: dict[str, datetime | None] = {}
    for key in _DATETIME_FILTER_FIELDS:
        raw = value.get(key)
        try:
            parsed[key] = datetime.fromisoformat(raw) if raw else None
        except (TypeError, ValueError):
            return ProtocolError(
                error_code="invalid_datetime_filter_value",
                message=f"{argument_name}.{key} must be an ISO-8601 datetime.",
                details={"field": key, "value": raw},
                suggested_correction=(
                    "Use an ISO-8601 datetime such as 2023-07-01T00:00:00Z."
                ),
            )
    return DatetimeFilter(**parsed)


async def _initialize_runtime() -> None:
    global _runtime
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    SUBMISSION_ROOT.mkdir(parents=True, exist_ok=True)
    _runtime = await asyncio.to_thread(
        FeatureEngineeringRuntime.from_config_path,
        TASK_CONFIG_PATH,
        registry=TrainedModelRegistry(RUNTIME_ROOT),
        official_scoring_enabled=False,
        task_outputs=SUBMISSION_ROOT,
        task_name="feature-engineering",
        data_split="public",
    )


def _required_runtime() -> FeatureEngineeringRuntime:
    if _runtime is None:
        raise RuntimeError("Feature-engineering runtime is not initialized.")
    return _runtime

@lifespan
async def _server_lifespan(_server: FastMCP):
    await _initialize_runtime()
    yield {}


mcp = FastMCP("feature-engineering", lifespan=_server_lifespan)


def _start_background(coro: Any, *, name: str) -> None:
    task = asyncio.create_task(coro, name=name)
    _background_tasks.add(task)

    def observe(completed: asyncio.Task[None]) -> None:
        _background_tasks.discard(completed)
        try:
            completed.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            LOGGER.exception("Background MCP operation failed: %s", name)

    task.add_done_callback(observe)


@mcp.tool()
async def train_model(
    model_code: str,
    label: str | None = None,
    train_filter: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Start one asynchronous model fit and immediately return a training ID."""

    runtime = _required_runtime()
    parsed_filter = _datetime_filter(train_filter, argument_name="train_filter")
    if isinstance(parsed_filter, ProtocolError):
        return runtime.record_protocol_error(parsed_filter)
    result = runtime.start_training(
        TrainModelRequest(
            model_code=model_code,
            label=label,
            train_filter=parsed_filter,
        )
    )
    if result.get("ok") is True:
        training_id = str(result["training_id"])
        _start_background(
            runtime.execute_training(training_id),
            name=f"feature-engineering-{training_id}",
        )
    return result


@mcp.tool()
async def get_train_model_result(training_id: str) -> dict[str, Any]:
    """Return running status or the immutable terminal training result."""

    result = _required_runtime().get_train_model_result(training_id)
    return {**result, "done": result.get("status") != "running"}


@mcp.tool()
async def backtest(
    model_id: str,
    max_gross_exposure: float,
    label: str | None = None,
    backtest_filter: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Start one asynchronous public backtest and return a backtest ID."""

    runtime = _required_runtime()
    parsed_filter = _datetime_filter(backtest_filter, argument_name="backtest_filter")
    if isinstance(parsed_filter, ProtocolError):
        return runtime.record_protocol_error(parsed_filter)
    result = runtime.start_backtest(
        BacktestRequest(
            model_id=model_id,
            max_gross_exposure=max_gross_exposure,
            label=label,
            backtest_filter=parsed_filter,
        )
    )
    if result.get("ok") is True:
        backtest_id = str(result["backtest_id"])
        _start_background(
            runtime.execute_backtest(backtest_id),
            name=f"feature-engineering-{backtest_id}",
        )
    return result


@mcp.tool()
async def get_backtest_result(backtest_id: str) -> dict[str, Any]:
    """Return running status or the immutable terminal backtest result."""

    result = _required_runtime().get_backtest_result(backtest_id)
    return {**result, "done": result.get("status") != "running"}


@mcp.tool()
async def submit_strategy(
    strategy_id: str,
    rationale: str | None = None,
) -> dict[str, Any]:
    """Validate and persist the one final accepted strategy submission."""

    return await _required_runtime().submit_strategy(
        SubmitStrategyRequest(strategy_id=strategy_id, rationale=rationale)
    )


def main() -> None:
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
