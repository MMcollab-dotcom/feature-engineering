# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.10.7@sha256:edd1fd89f3e5b005814cc8f777610445d7b7e3ed05361f9ddfae67bebfe8456a AS uv

FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7 AS runtime


ENV PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY environment/mcp-server/requirements.lock /app/requirements.lock
RUN --mount=type=bind,from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=type=cache,target=/root/.cache \
    uv pip install --link-mode=copy --system --require-hashes --no-deps \
    -r /app/requirements.lock

COPY environment/mcp-server/pyproject.toml environment/LICENSE /app/
COPY environment/mcp-server/feature_engineering /app/feature_engineering
COPY environment/mcp-server/evalenv_shared/worker /app/evalenv_shared/worker
COPY environment/mcp-server/evalenv_shared/__init__.py /app/evalenv_shared/__init__.py
RUN --mount=type=bind,from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=type=cache,target=/root/.cache \
    uv pip install --link-mode=copy --system --no-deps --no-build-isolation /app

FROM runtime AS worker
RUN groupadd --gid 10001 worker \
 && useradd --uid 10001 --gid 10001 --no-create-home worker
USER 10001:10001

CMD ["feature-engineering-worker-server"]

FROM runtime AS tools

COPY environment/mcp-server/task_config.yaml environment/mcp-server/data_manifest.json /app/
COPY --chown=10001:10001 --chmod=0600 environment/mcp-server/runtime_public.parquet \
    /app/data/runtime_public.parquet

RUN mkdir -p /app/runtime /app/submission /logs/verifier \
 && chown -R 10001:10001 /app/runtime /app/submission /logs/verifier

EXPOSE 8000

CMD ["feature-engineering-mcp"]

FROM tools AS verifier

COPY tests/test.sh /tests/test.sh
COPY tests/hidden_data /app/tests/hidden_data

CMD ["sleep", "infinity"]
