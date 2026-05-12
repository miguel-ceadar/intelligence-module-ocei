.PHONY: install install-dev install-legacy lint format test test-fast test-integration clean \
        up down down-clean logs smoke e2e

COMPOSE := docker compose

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

# --- Docker compose stack -----------------------------------------------------
# `up` builds + waits for healthchecks; `down` keeps the bento volume so
# trained models persist across restarts; `down-clean` drops it.

up:
	$(COMPOSE) up -d --build --wait

down:
	$(COMPOSE) down

down-clean:
	$(COMPOSE) down -v

logs:
	$(COMPOSE) logs -f intelligence

# `smoke` assumes the stack is already up (iterate without rebuilding).
# `e2e` is the one-shot: up + smoke, dump logs on failure.
smoke:
	pytest -m smoke -v

e2e: up
	pytest -m smoke -v || ($(COMPOSE) logs intelligence; exit 1)
