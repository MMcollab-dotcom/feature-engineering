# syntax=docker/dockerfile:1.7
FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7 AS runtime

COPY --from=ghcr.io/astral-sh/uv:0.10.7 /uv /bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY environment/mcp-server/requirements.lock /app/requirements.lock
RUN uv pip install --system --require-hashes --no-deps -r /app/requirements.lock

COPY environment/mcp-server/pyproject.toml environment/LICENSE /app/
COPY environment/mcp-server/feature_engineering /app/feature_engineering
COPY environment/mcp-server/evalenv_shared/worker /app/evalenv_shared/worker
COPY environment/mcp-server/evalenv_shared/__init__.py /app/evalenv_shared/__init__.py
RUN uv pip install --system --no-deps --no-build-isolation /app

COPY environment/mcp-server/task_config.yaml environment/mcp-server/data_manifest.json /app/
COPY environment/mcp-server/runtime_public.parquet /app/data/runtime_public.parquet

RUN mkdir -p /app/runtime /app/submission \
 && chmod 600 /app/data/runtime_public.parquet

FROM runtime AS verifier

COPY tests /tests
COPY tests/hidden_data /app/tests/hidden_data

CMD ["sleep", "infinity"]

FROM runtime AS tools

EXPOSE 8000

CMD ["feature-engineering-mcp"]
