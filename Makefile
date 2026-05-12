.PHONY: install install-dev install-legacy lint format test test-fast test-integration clean \
        up down logs smoke up-demo down-demo logs-demo e2e

COMPOSE := docker compose
COMPOSE_DEMO := docker compose -f docker-compose.yml -f docker-compose.demo.yml

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

# --- Docker compose -----------------------------------------------------------
# Standalone deployment — `up` runs just the intelligence service, expecting
# you to point it at your own Prometheus (see `.env.example`).

up:
	$(COMPOSE) up -d --build --wait

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f intelligence

# Demo overlay — adds an in-stack Prometheus + node-exporter so the service
# has something to train against without touching your real infra. Used by
# the smoke suite.

up-demo:
	$(COMPOSE_DEMO) up -d --build --wait

down-demo:
	$(COMPOSE_DEMO) down -v

logs-demo:
	$(COMPOSE_DEMO) logs -f intelligence

# `smoke` assumes a stack is already up — iterate without rebuilding.
# `e2e` is the one-shot: demo overlay + smoke, dump logs on failure.
smoke:
	uv run pytest -m smoke -v

e2e: up-demo
	uv run pytest -m smoke -v || ($(COMPOSE_DEMO) logs intelligence; exit 1)

# --- Helm chart ---------------------------------------------------------------

chart-lint:
	helm lint helm/intelligence

chart-template:
	helm template intelligence helm/intelligence

.PHONY: chart-lint chart-template
