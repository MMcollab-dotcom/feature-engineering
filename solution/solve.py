"""Reference agent that exercises the task exclusively through MCP."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MCP_URL = "http://mcp-server:8000/mcp"
MODEL_CODE = r"""import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class LagReturnFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        close = X["close"].astype("float64")
        grouped = close.groupby(level="symbol", sort=False)
        values = []
        for lag in (1, 2, 3, 5):
            ret = close / grouped.shift(lag) - 1.0
            ret = ret.clip(-0.2, 0.2).fillna(0.0)
            values.append(ret.to_numpy())
        return np.column_stack(values)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(
            ["return_1", "return_2", "return_3", "return_5"],
            dtype=object,
        )


def train_model(X, y):
    weight = X["weight_std_dollar_vol"].astype("float64").to_numpy()
    model = make_pipeline(
        LagReturnFeatures(),
        StandardScaler(),
        ElasticNet(
            alpha=2.0e-6,
            l1_ratio=0.05,
            max_iter=3000,
            tol=1.0e-6,
            selection="cyclic",
        ),
    )
    model.fit(
        X.loc[:, ["close"]],
        y.to_numpy().ravel(),
        elasticnet__sample_weight=weight,
    )
    return model
"""


def _payload(result) -> dict:
    if result.isError:
        raise RuntimeError(str(result.content))
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured
    for block in result.content:
        text = getattr(block, "text", None)
        if text:
            value = json.loads(text)
            if isinstance(value, dict):
                return value
    raise RuntimeError("MCP tool returned no JSON object")


async def _wait(session: ClientSession, tool: str, id_name: str, value: str) -> dict:
    while True:
        result = _payload(await session.call_tool(tool, {id_name: value}))
        if result.get("done") is True:
            return result
        await asyncio.sleep(2)


async def solve() -> None:
    async with (
        streamable_http_client(MCP_URL) as (read, write, *_),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        training = _payload(
            await session.call_tool(
                "train_model",
                {
                    "model_code": MODEL_CODE,
                    "label": "sol-short-reversal-elastic-net",
                    "train_filter": {
                        "start_datetime": "2022-01-01T00:00:00+00:00",
                        "end_datetime": "2022-12-31T23:59:00+00:00",
                    },
                },
            )
        )
        if not training.get("ok"):
            raise RuntimeError(json.dumps(training, sort_keys=True))
        trained = await _wait(
            session,
            "get_train_model_result",
            "training_id",
            training["training_id"],
        )
        if not trained.get("ok"):
            raise RuntimeError(json.dumps(trained, sort_keys=True))

        started = _payload(
            await session.call_tool(
                "backtest",
                {
                    "model_id": trained["model_id"],
                    "max_gross_exposure": 1.0,
                    "label": "sol-short-reversal-elastic-net",
                    "backtest_filter": {
                        "start_datetime": "2023-01-01T00:00:00+00:00",
                        "end_datetime": "2023-12-31T23:57:00+00:00",
                    },
                },
            )
        )
        if not started.get("ok"):
            raise RuntimeError(json.dumps(started, sort_keys=True))
        backtested = await _wait(
            session,
            "get_backtest_result",
            "backtest_id",
            started["backtest_id"],
        )
        if not backtested.get("ok"):
            raise RuntimeError(json.dumps(backtested, sort_keys=True))

        submitted = _payload(
            await session.call_tool(
                "submit_strategy",
                {
                    "strategy_id": backtested["strategy_id"],
                    "rationale": (
                        "Public-only four-lag return-reversal Elastic Net baseline. "
                        "Hyperparameters "
                        "were selected with expanding chronological folds in 2022; the "
                        "official training and backtest windows are disjoint."
                    ),
                },
            )
        )
        if not submitted.get("ok"):
            raise RuntimeError(json.dumps(submitted, sort_keys=True))
        Path("/app/reference-result.json").write_text(
            json.dumps(submitted, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    asyncio.run(solve())
