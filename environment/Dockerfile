# syntax=docker/dockerfile:1.7
FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7 AS runtime

COPY --from=ghcr.io/astral-sh/uv:0.10.7 /uv /bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY environment/mcp-server/requirements.lock /app/requirements.lock
RUN uv pip install --system --require-hashes --no-deps -r /app/requirements.lock

FROM runtime AS public-data

COPY environment/mcp-server/runtime_public.parquet /tmp/public.parquet
COPY environment/mcp-server/task_config.yaml /tmp/task_config.yaml
RUN python -c "from pathlib import Path; import pandas as pd, yaml; data=yaml.safe_load(Path('/tmp/task_config.yaml').read_text())['data']; columns=['datetime','symbol',*data['features'],*data['targets']]; pd.read_parquet('/tmp/public.parquet', columns=columns).to_parquet('/tmp/public_train.parquet', index=False, compression='zstd')"

FROM runtime

COPY environment/agent/AGENTS.md /app/AGENTS.md
COPY environment/agent/context /app/context
COPY --from=public-data /tmp/public_train.parquet /app/data/public_train.parquet

RUN mkdir -p /app/research /logs/agent \
 && chmod 444 /app/AGENTS.md /app/data/public_train.parquet /app/context/*.md

CMD ["sleep", "infinity"]
