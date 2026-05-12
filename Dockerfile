# syntax=docker/dockerfile:1.7

FROM python:3.11-slim

# libgomp1: xgboost native runtime. ca-certificates: HTTPS to Prometheus / HF.
# curl: used by compose healthchecks against /healthz.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# uv resolves and installs Python deps. Pinned for build reproducibility.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

# pyproject + sources are everything hatchling needs to build the wheel.
# Copy README too — pyproject references it as the package readme.
COPY pyproject.toml README.md ./
COPY src/ ./src/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_PROGRESS=1 \
    UV_CACHE_DIR=/root/.cache/uv

# BuildKit cache mount keeps uv's archive cache out of the image layer —
# the venv is copied in (UV_LINK_MODE=copy), the cache itself stays mounted.
# Without this the cache duplicates every wheel inside the final image.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

ENV PATH="/app/.venv/bin:${PATH}" \
    BENTOML_HOME=/var/lib/bentoml \
    PYTHONUNBUFFERED=1
RUN mkdir -p /var/lib/bentoml

EXPOSE 3000

# uvicorn serves the FastAPI app. bentoml.Service mounts the same app for
# its model-store machinery; the runner isn't used, so `bentoml serve` adds
# no value over uvicorn here.
CMD ["uvicorn", "intelligence.api.service:app", "--host", "0.0.0.0", "--port", "3000"]
