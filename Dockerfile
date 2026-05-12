# syntax=docker/dockerfile:1.7
#
# Two-stage build per Astral's canonical uv-in-Docker pattern
# (https://docs.astral.sh/uv/guides/integration/docker/).
#
# Stage 1 (builder): resolves uv.lock and installs deps + project into /app/.venv.
# Stage 2 (runtime): copies only the venv onto a clean python:3.11-slim base.
# Result: no apt build tooling, no uv binary, no sources in the runtime image.

FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_PROGRESS=1

# Step 1: install deps without the project. This layer only invalidates when
# pyproject.toml or uv.lock change, so source-only edits skip the dep install.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# Step 2: install the project itself, non-editable so /app/.venv is
# self-contained and can be copied without src/ tagging along.
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable


FROM python:3.11-slim

# libgomp1: xgboost native runtime. curl: compose healthcheck. ca-certs: HTTPS.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:${PATH}" \
    BENTOML_HOME=/var/lib/bentoml \
    PYTHONUNBUFFERED=1
RUN mkdir -p /var/lib/bentoml

EXPOSE 3000

CMD ["uvicorn", "intelligence.api.service:app", "--host", "0.0.0.0", "--port", "3000"]
