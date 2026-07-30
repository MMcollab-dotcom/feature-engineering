# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.10.7@sha256:edd1fd89f3e5b005814cc8f777610445d7b7e3ed05361f9ddfae67bebfe8456a AS uv

FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7 AS runtime


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
