# syntax=docker/dockerfile:1.7
#
# Two-stage build per Astral's canonical uv-in-Docker pattern
# (https://docs.astral.sh/uv/guides/integration/docker/).
#
# Stage 1 (builder): resolves uv.lock and installs deps + project into /app/.venv.
# Stage 2 (runtime): copies only the venv onto a clean python:3.11-slim base.
# Result: no apt build tooling, no uv binary, no sources in the runtime image.

FROM python:3.11-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_PROGRESS=1

# Step 1: install deps without the project. This layer only invalidates when
# pyproject.toml or uv.lock change, so source-only edits skip the dep install.
# --extra drift bundles nannyml so the shipped image can serve drift tasks
# out of the box. Pilots who don't run drift can build their own slimmer
# image by dropping that flag.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev --extra drift

# Step 2: install the project itself, non-editable so /app/.venv is
# self-contained and can be copied without src/ tagging along.
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --extra drift


FROM python:3.11-slim

# libgomp1: xgboost native runtime. curl: compose healthcheck. ca-certs: HTTPS.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade the base image's system pip + setuptools + wheel to clear
# CVEs in the versions python:3.11-slim ships (jaraco.context inside
# setuptools/_vendor, the top-level wheel install). The runtime never
# uses these — the venv is uv-built and self-contained — but trivy
# flags the vendored copies regardless. Must run before PATH points
# at /app/.venv/bin so the *system* pip is what executes.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:${PATH}" \
    BENTOML_HOME=/var/lib/bentoml \
    PYTHONUNBUFFERED=1

# Non-root runtime. UID 1000 matches the Helm chart's
# podSecurityContext.runAsUser default, so both deployment paths run
# with the same identity.
RUN groupadd --system --gid 1000 app \
 && useradd --system --uid 1000 --gid app --home-dir /app --shell /usr/sbin/nologin app \
 && mkdir -p /var/lib/bentoml \
 && chown -R app:app /var/lib/bentoml /app

USER app

EXPOSE 3000

# Curl is already present (installed above for compose probes). Healthcheck
# hits the cheap liveness endpoint so `docker run` users get container
# health out of the box without needing to wire k8s probes.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:3000/healthz || exit 1

CMD ["uvicorn", "intelligence.api.service:app", "--host", "0.0.0.0", "--port", "3000"]
