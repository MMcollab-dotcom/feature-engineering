"""Harbor-facing FastMCP server for the fixed feature-engineering task."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from feature_engineering.runtime.protocol import (
    BacktestRequest,
    DatetimeFilter,
    SubmitStrategyRequest,
    TrainModelRequest,
)
from feature_engineering.runtime.task import FeatureEngineeringRuntime
from feature_engineering.submissions.registry import TrainedModelRegistry

LOGGER = logging.getLogger(__name__)
TASK_CONFIG_PATH = Path(os.environ.get("FEATURE_TASK_CONFIG", "/app/task_config.yaml"))
RUNTIME_ROOT = Path(os.environ.get("FEATURE_RUNTIME_ROOT", "/app/runtime"))
SUBMISSION_ROOT = Path(os.environ.get("FEATURE_SUBMISSION_ROOT", "/app/submission"))

mcp = FastMCP("feature-engineering")

_runtime: FeatureEngineeringRuntime | None = None
_runtime_lock: asyncio.Lock | None = None
_background_tasks: set[asyncio.Task[None]] = set()


def _datetime_filter(value: dict[str, str] | None) -> DatetimeFilter | None:
    if value is None:
        return None
    from datetime import datetime

    parsed: dict[str, datetime | None] = {}
    for key in ("start_datetime", "end_datetime"):
        raw = value.get(key)
        parsed[key] = datetime.fromisoformat(raw) if raw else None
    return DatetimeFilter(**parsed)


async def _get_runtime() -> FeatureEngineeringRuntime:
    global _runtime, _runtime_lock
    if _runtime is not None:
        return _runtime
    if _runtime_lock is None:
        _runtime_lock = asyncio.Lock()
    async with _runtime_lock:
        if _runtime is None:
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
    return _runtime


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

    runtime = await _get_runtime()
    result = runtime.start_training(
        TrainModelRequest(
            model_code=model_code,
            label=label,
            train_filter=_datetime_filter(train_filter),
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

    result = (await _get_runtime()).get_train_model_result(training_id)
    return {**result, "done": result.get("status") != "running"}


@mcp.tool()
async def backtest(
    model_id: str,
    max_gross_exposure: float,
    label: str | None = None,
    backtest_filter: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Start one asynchronous public backtest and return a backtest ID."""

    runtime = await _get_runtime()
    result = runtime.start_backtest(
        BacktestRequest(
            model_id=model_id,
            max_gross_exposure=max_gross_exposure,
            label=label,
            backtest_filter=_datetime_filter(backtest_filter),
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

    result = (await _get_runtime()).get_backtest_result(backtest_id)
    return {**result, "done": result.get("status") != "running"}


@mcp.tool()
async def submit_strategy(
    strategy_id: str,
    rationale: str | None = None,
) -> dict[str, Any]:
    """Validate and persist the one final accepted strategy submission."""

    return await (await _get_runtime()).submit_strategy(
        SubmitStrategyRequest(strategy_id=strategy_id, rationale=rationale)
    )


def main() -> None:
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
