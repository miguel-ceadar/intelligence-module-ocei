.PHONY: install install-dev install-legacy lint format test test-fast test-integration clean

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
