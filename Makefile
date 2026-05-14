.PHONY: install install-dev install-legacy lint format test test-fast test-integration clean \
        up-demo down-demo logs-demo smoke e2e stress e2e-stress \
        up-dev down-dev logs-dev e2e-dev e2e-stress-dev \
        chart-lint chart-template chart-template-matrix \
        chart-e2e-up chart-e2e chart-e2e-down

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

# Build is its own step so output is visible — `--build --wait` from
# `up` can swallow build output behind the wait spinner. `--force-recreate`
# guarantees the container is replaced even when compose's image-hash
# check decides the existing container is still valid.
up-dev:
	$(COMPOSE_DEV) build intelligence
	$(COMPOSE_DEV) up -d --force-recreate --wait

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

# Render the chart against every CI overlay. Mirrors the `helm` job in
# .github/workflows/ci.yml — keep the two in sync when adding overlays.
chart-template-matrix:
	helm template icos-intelligence-ocei helm/intelligence
	helm template icos-intelligence-ocei helm/intelligence -f helm/intelligence/ci/minimal-values.yaml
	helm template icos-intelligence-ocei helm/intelligence -f helm/intelligence/ci/full-values.yaml

# End-to-end install into a local kind cluster — k8s counterpart to
# `make e2e` (which boots compose + runs the smoke suite). Same shape:
# `chart-e2e-up` boots the stack and leaves it running; `chart-e2e`
# runs the smoke suite against it; `chart-e2e-down` tears it all down.
#
# Prereqs: docker, kind, kubectl, helm, uv. Not in CI — too slow and
# pulls in tooling the runners don't always have. Run before tagging a
# release or after touching helm/intelligence/templates/**.
KIND_CLUSTER ?= icos-intelligence-test
CHART_E2E_IMAGE := icos-intelligence-ocei:chart-e2e

# Boot kind, build the local image into the cluster, deploy the
# Prometheus + node-exporter pair that mirrors the compose stack, then
# helm-install the chart against that Prometheus. Leaves everything
# running. The Prom + node-exporter manifests live next to the values
# overlay in helm/intelligence/ci/ — keep them in sync with compose/.
chart-e2e-up:
	@kind get clusters | grep -q "^$(KIND_CLUSTER)$$" || kind create cluster --name $(KIND_CLUSTER)
	docker build -t $(CHART_E2E_IMAGE) .
	kind load docker-image $(CHART_E2E_IMAGE) --name $(KIND_CLUSTER)
	kubectl apply -f helm/intelligence/ci/prom-stack.yaml
	kubectl rollout status deploy/prometheus    --timeout=180s
	kubectl rollout status deploy/node-exporter --timeout=180s
	helm upgrade --install intelligence helm/intelligence \
		--set fullnameOverride=intelligence \
		--set image.repository=icos-intelligence-ocei \
		--set image.tag=chart-e2e \
		--set image.pullPolicy=Never \
		-f helm/intelligence/ci/e2e-values.yaml \
		--wait --timeout 5m

# Port-forward the chart's Service and the in-cluster Prometheus so the
# smoke suite hits the same localhost:3000 / localhost:9090 endpoints
# it uses against compose, then run it. Both forwards are killed on
# exit; the release stays up for `make chart-e2e-down` (mirrors how
# `make e2e` leaves compose up for `make down-demo`).
chart-e2e: chart-e2e-up
	@bash -c 'set -e; \
		kubectl port-forward svc/intelligence 3000:3000 >/dev/null 2>&1 & PF_I=$$!; \
		kubectl port-forward svc/prometheus   9090:9090 >/dev/null 2>&1 & PF_P=$$!; \
		trap "kill $$PF_I $$PF_P 2>/dev/null || true" EXIT; \
		sleep 3; \
		uv run pytest -m smoke -v || (kubectl logs deploy/intelligence; exit 1)'
	@echo "chart-e2e passed. Stack left running — 'make chart-e2e-down' to tear it all down."

chart-e2e-down:
	-helm uninstall intelligence 2>/dev/null
	-kubectl delete -f helm/intelligence/ci/prom-stack.yaml 2>/dev/null
	kind delete cluster --name $(KIND_CLUSTER)
