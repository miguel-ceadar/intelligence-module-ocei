.PHONY: install install-dev install-legacy lint format test test-fast test-integration clean \
        up-demo down-demo logs-demo smoke e2e stress e2e-stress \
        up-dev down-dev logs-dev e2e-dev e2e-stress-dev chart-lint chart-template

# The compose stack is a DEMO only (image + bundled prometheus + node-exporter
# for "see it work in 3 minutes"). Pilots deploy via the Helm chart (k8s) or
# `docker run` against THEIR Prometheus — neither path involves compose.
COMPOSE_DEMO := docker compose -f docker-compose.demo.yml
COMPOSE_DEV  := docker compose -f docker-compose.demo.yml -f docker-compose.dev.yml

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-legacy:
	pip install -e ".[dev,legacy]"

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff format src tests
	ruff check --fix src tests

test:
	pytest

test-fast:
	pytest -m "not integration and not slow"

test-integration:
	pytest -m integration

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	rm -rf build dist *.egg-info .coverage htmlcov

# --- Demo compose stack -------------------------------------------------------
# Pulls the published image from GHCR + spins up prometheus + node-exporter.
# Pin the tag via `INTELLIGENCE_TAG=v0.1.0 make up-demo`.

up-demo:
	$(COMPOSE_DEMO) up -d --wait

down-demo:
	$(COMPOSE_DEMO) down -v

logs-demo:
	$(COMPOSE_DEMO) logs -f intelligence

# `smoke` assumes a stack is already up — iterate without rebuilding.
# `e2e` is the one-shot: demo stack + smoke, dump logs on failure.
smoke:
	uv run pytest -m smoke -v

e2e: up-demo
	uv run pytest -m smoke -v || ($(COMPOSE_DEMO) logs intelligence; exit 1)

# Stress runs against the same compose stack as `make e2e` — the demo
# Prometheus retention is 12h, so leave the stack up for at least 15
# minutes before running stress (some tests train against 6h of
# history). Tune via STRESS_PREDICT_LOOPS, STRESS_LONG_WINDOW etc; see
# the docstring on tests/smoke/test_stress.py.
stress:
	uv run pytest -m stress -v -s

# One-shot: boot the stack, run quick smoke, then heavier stress. For
# overnight: `make up-demo`, leave for hours, then `make stress` when
# you want to exercise the long-window paths.
e2e-stress: up-demo
	uv run pytest -m smoke -v || ($(COMPOSE_DEMO) logs intelligence; exit 1)
	uv run pytest -m stress -v -s || ($(COMPOSE_DEMO) logs intelligence; exit 1)

# --- Dev overlay (build image from local sources) -----------------------------
# Contributor path: rebuild the image and run the demo stack against the
# local build instead of the GHCR image. Use this whenever the local
# schema diverges from the published image (between releases).
#
# `smoke` and `stress` are stack-agnostic — they just hit localhost:3000
# — so they work against either the demo stack (up-demo) or the dev
# stack (up-dev). Only the orchestration targets care which compose
# files are in play; that's what the `-dev` variants here are for.

up-dev:
	$(COMPOSE_DEV) up -d --build --wait

down-dev:
	$(COMPOSE_DEV) down -v

logs-dev:
	$(COMPOSE_DEV) logs -f intelligence

e2e-dev: up-dev
	uv run pytest -m smoke -v || ($(COMPOSE_DEV) logs intelligence; exit 1)

# Dev counterpart to `e2e-stress`: rebuild local image, then run smoke
# then stress. For overnight on the local build, swap to `make up-dev`
# and leave it warming before invoking `make stress`.
e2e-stress-dev: up-dev
	uv run pytest -m smoke -v || ($(COMPOSE_DEV) logs intelligence; exit 1)
	uv run pytest -m stress -v -s || ($(COMPOSE_DEV) logs intelligence; exit 1)

# --- Helm chart ---------------------------------------------------------------

chart-lint:
	helm lint helm/intelligence

chart-template:
	helm template icos-intelligence-ocei helm/intelligence
