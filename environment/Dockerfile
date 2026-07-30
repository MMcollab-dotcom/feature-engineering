# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.12.0@sha256:606e70c71c852d03f611b1e56a195d08648507018a7057fab82c4974c4eae105 AS uv

FROM python:3.12.13-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b AS runtime


ENV PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY environment/mcp-server/requirements.lock /app/requirements.lock
RUN --mount=type=bind,from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=type=cache,target=/root/.cache \
    uv pip install --link-mode=copy --system --require-hashes --no-deps \
    -r /app/requirements.lock

FROM runtime AS public-data

COPY environment/mcp-server/runtime_public.parquet /tmp/public.parquet
COPY environment/mcp-server/task_config.yaml /tmp/task_config.yaml
RUN python -c "from pathlib import Path; import pandas as pd, yaml; data=yaml.safe_load(Path('/tmp/task_config.yaml').read_text())['data']; columns=['datetime','symbol',*data['features'],*data['targets']]; pd.read_parquet('/tmp/public.parquet', columns=columns).to_parquet('/tmp/public_train.parquet', index=False, compression='zstd')"

FROM runtime

COPY --chmod=0444 environment/agent/AGENTS.md /app/AGENTS.md
COPY environment/agent/context /app/context
COPY --chmod=0444 --from=public-data \
    /tmp/public_train.parquet /app/data/public_train.parquet

RUN mkdir -p /app/research /logs/agent \
 && chmod 444 /app/context/*.md

CMD ["sleep", "infinity"]
