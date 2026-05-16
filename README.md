# icos-intelligence-ocei

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3721889.3721929-blue.svg)](https://doi.org/10.1145/3721889.3721929)

> Train forecasting and drift-detection models against your existing
> Prometheus metrics, served as a small HTTP API.

Point the service at your Prometheus, list the PromQL queries for the
metrics you care about, and you get `/train` + `/predict` endpoints per
metric. ARIMA (with native confidence intervals), XGBoost, LSTM, and
NannyML drift detection ship out of the box. Adding a new
`(metric × algorithm)` pairing is one YAML block under `tasks:`.

Originally developed by **CeADAR Ireland** for the
[ICOS metaOS project](https://github.com/icos-project/intelligence-module).
This iteration is the O-CEI continuum take — relaxes the ICOS-specific
coupling so the same service runs on any vanilla Kubernetes +
Prometheus stack, beside or instead of the original ICOS deployment.

> **New here?** Follow the [Getting Started](docs/getting-started.md)
> walkthrough — install the chart, point at a Prometheus, train, and
> predict in about 10 minutes. The rest of this README is reference.

## What's included

Four algorithm **kinds** you compose into tasks via YAML:

| Kind | Model | Forecast shape |
|---|---|---|
| `arima` | statsmodels ARIMA | Single or multi-step; native 95 % confidence intervals |
| `xgb` | XGBoost regressor | Sliding-window, recursive multi-step (no CI) |
| `lstm` | PyTorch LSTM | Sliding-window, direct multi-step (no CI) |
| `drift` | NannyML | Univariate drift alert on a chunk, paired with a forecaster |

A *task* is one PromQL query + one kind. Declare as many as you want
under `tasks:` in the config — no Python edits. The
[`examples/`](examples/) directory has ready-to-run configs for CPU,
memory, k8s cluster metrics, and energy forecasting.

Two data sources, picked once per deployment:

- **`prometheus`** — PromQL `/api/v1/query_range` against your Prometheus or Thanos.
- **`static`** — CSVs from the bundled `samples/` directory; used for the in-repo demo and tests.

## Quick start

The deployable artifact is the container image at
`ghcr.io/miguel-ceadar/icos-intelligence-ocei`. Two ways to consume
it: Helm for Kubernetes, raw `docker run` for a single host. A
self-contained demo runs against a bundled Prometheus + node-exporter
for quick local evaluation — it is not a deployment template.

### Deploy on Kubernetes (Helm)

The chart is published as an OCI artifact. Pin to a release version
(avoid `latest`, which is not stable):

```bash
helm install icos-intelligence-ocei \
  oci://ghcr.io/miguel-ceadar/charts/icos-intelligence-ocei \
  --version 0.2.10 \
  -f your-values.yaml
```

Cluster-side it deploys a Deployment + Service + ConfigMap (the YAML
config) + Secret (tokens) + PVC (Bento store) + optional ServiceMonitor
and retraining CronJob. See the [chart README](helm/intelligence/README.md)
for values and the multi-replica caveat.

### Deploy on a single host (`docker run`)

```bash
docker run -d --name icos-intelligence-ocei \
  -p 3000:3000 \
  -e INTELLIGENCE_CONFIG=/etc/intelligence/config.yaml \
  -e INTELLIGENCE_TELEMETRY__PROMETHEUS__ENDPOINT=https://prom.example.com \
  -v "$PWD/config.yaml:/etc/intelligence/config.yaml:ro" \
  -v intelligence-bentoml:/var/lib/bentoml \
  ghcr.io/miguel-ceadar/icos-intelligence-ocei:0.2.10

curl http://localhost:3000/healthz
```

Place a `config.yaml` next to the command — start from any
[`examples/*/config.yaml`](examples/) and edit the PromQL queries.
Any field in the config can also be overridden via an environment
variable (`INTELLIGENCE_<SECTION>__<FIELD>` — see
[Configuration](#configuration)).

### See it work (in-repo demo, 3 minutes)

```bash
make e2e
```

Pulls the published image, spins up a bundled Prometheus +
node-exporter, exercises train + predict on all four tasks, dumps
logs on failure. `make down-demo` tears it down. The compose file is
for the demo only; real deployments use Helm or `docker run` against
your own Prometheus.

### Calling the API

Train + predict against the Prometheus you pointed the service at:

```bash
curl -X POST http://localhost:3000/tasks/cpu_forecast_arima/train \
  -H 'Content-Type: application/json' \
  -d '{"data_source": {"kind": "prometheus", "window": "24h", "step": "1m"}}'

curl -X POST http://localhost:3000/tasks/cpu_forecast_arima/predict \
  -H 'Content-Type: application/json' \
  -d '{"input_series": {"cpu": [0.85]}}'
```

The `data_source.kind` on the request must match the deployment's
`telemetry.source` config (`prometheus` here). To train without a
Prometheus, switch to `source: static` and pass a CSV name — see
[`examples/cpu_forecast/README.md`](examples/cpu_forecast/README.md#try-it-in-static-mode)
or just run `make e2e`, which spins up the bundled Prometheus +
node-exporter.

`cpu_forecast_xgb` and `cpu_forecast_lstm` expect a 6-value `input_series`
window; `cpu_forecast_arima_drift` expects 12. Otherwise identical
envelope.

## API

The full HTTP surface — train, predict, list models, version pinning,
HF push/pull, Prometheus `/metrics` — is documented at `/docs` (Swagger
UI) and `/redoc` once the service is running.

Each task declares an input contract (feature count, window length,
expected value range); mismatched requests get `422` before reaching
the model. Predict serves `:latest` by default — pin a specific
version via the request's `model_version` or the task config's
`pinned_version` (request wins; see [Pretrained models](#pretrained-models)
for the validation rules around pulled models).

Predict requests accept an optional `horizon` (default `1`). The
response's `prediction` field is a list of `{value, lower, upper}`
points of length `horizon`; `lower`/`upper` carry a 95 % confidence
interval when the model exposes one (ARIMA does; recursive XGB and
direct-output LSTM leave them empty).

### Observability

`/metrics` exposes Prometheus counters and histograms (HTTP, per-task
train/predict). Logs are JSON to stdout, each request tagged with a
short `request_id` for grepping across one call's trace.

## Configuration

The service reads a YAML config from `INTELLIGENCE_CONFIG`. Any field
can be overridden by an env var — `INTELLIGENCE_<SECTION>__<FIELD>`
(double underscore separates nested sections, env wins over YAML).

### Minimal — static, bundled samples

```yaml
intelligence:
  telemetry:
    source: static
  tasks:
    cpu_forecast_arima:
      kind: arima
      steps_back: 1
      features:
        - name: cpu
          value_range: [0.0, 1.0]
```

### Prometheus

```yaml
intelligence:
  telemetry:
    source: prometheus
    prometheus:
      endpoint: http://prometheus.monitoring.svc:9090
      token_env: PROM_TOKEN          # or token_file: /var/run/secrets/prom
      tls_skip_verify: false
    allow_endpoint_override: false   # set true to allow `data_source.endpoint`
                                     # overrides on train requests (SSRF risk)
  tasks:
    cpu_forecast_arima:
      kind: arima
      steps_back: 1
      features:
        - name: cpu
          value_range: [0.0, 1.0]
          query: 'avg(rate(node_cpu_seconds_total{mode="user"}[1m]))'
```

Tasks declare their inputs under `features:` — the first entry is the
target, any additional entries are covariates (XGB/LSTM consume them;
ARIMA refuses multi-feature configs). Each feature carries its own
PromQL on `query:` when `telemetry.source: prometheus`. Train request
shape:

```bash
curl -X POST http://localhost:3000/tasks/cpu_forecast_arima/train \
  -H 'Content-Type: application/json' \
  -d '{"data_source": {"kind": "prometheus", "window": "24h", "step": "1m"}}'
```

`source: prometheus` without a `prometheus:` block fails at startup,
not at first request. Same for a prom-mode task without `query:`.

#### Per-request endpoint override

When `allow_endpoint_override: true`, a train request can flip the
Prometheus endpoint for that single call:

```bash
curl -X POST http://localhost:3000/tasks/cpu_forecast_arima/train \
  -d '{"data_source": {"kind": "prometheus", "window": "24h", "step": "1m",
                       "endpoint": "https://other-prom.example:9090"}}'
```

The configured auth + TLS settings carry through to the override
(per-request token rotation is not supported). Off by default,
because an authenticated `POST /train` can otherwise redirect
outbound traffic to arbitrary hosts (SSRF). Enable only for trusted
clients.

### Auth (bearer token on the HTTP API)

Off by default — the smoke stack and local dev stay frictionless, and
the service logs a `HTTP auth disabled` warning at startup. To require
`Authorization: Bearer <token>` on `/train`, `/predict`, `/tasks` and
friends, point the service at an env var holding the expected value:

```yaml
intelligence:
  auth:
    token_env: API_TOKEN
```

`/healthz`, `/readyz`, `/metrics`, and `/docs` stay open regardless so
kubelet probes and API discovery still work without a credential. The
Helm chart pairs this with `existingSecretName` for production
deployments — see the
[energy walkthrough](docs/energy-walkthrough.md#step-1--pre-create-the-secret)
for the end-to-end recipe.

### Auto-train on startup

Opt a task in to run its first training in the background as the
service boots:

```yaml
intelligence:
  tasks:
    cpu_forecast_arima:
      bootstrap:
        auto_train_on_startup: true
        dataset_name: cpu_sample_dataset_orangepi.csv   # for source: static
        # window: 24h                                   # for source: prometheus
        # step: 1m
```

`/readyz` returns `503` while bootstrap runs, then `200`. Default is
off — operators opt in explicitly so a misconfigured query doesn't
silently block startup.

### Retraining

The library itself has no built-in scheduler. The Helm chart ships
one — set `retraining.enabled: true` and list tasks under
`retraining.tasks:` (see the [chart README](helm/intelligence/README.md#retraining)).
For non-Kubernetes deployments, point any external cron at
`POST /tasks/{task}/train`.

### Hugging Face push / pull

```yaml
intelligence:
  model_repo:
    hf_enabled: true
    repo_id: <org>/<repo>
```

The HF token is read from the `HF_TOKEN` environment variable at
request time — never stored in config. Request shape for
`POST /models/sync` is at `/docs`.

### Pretrained models

A pulled model becomes available to a task only if its stored
`input_spec` matches the task's spec (same `n_features`,
`feature_names`, `steps_back`). Mismatched models are refused at
predict time — the pull still completes and the artifact lands in
the local store, but `predict` will not serve from it. Verification
is always enforced; the `allow_unverified_models` flag on `BaseTask`
exists only as a debugging hook and has no YAML surface.

## Extending

A task is a `(data source × model algorithm)` pairing declared in
YAML. The lib ships four algorithm **kinds**; everything else — which
metric, which PromQL, which task name — is config. To get any new
forecast wired up, you add a block under `tasks:` and pick a `kind:`.

### Add a task (no code)

Drop a block into your `config.yaml`:

```yaml
intelligence:
  tasks:
    mem_forecast_arima:
      kind: arima
      steps_back: 1
      features:
        - name: mem
          value_range: [0.0, 1.0]
          query: '1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)'
```

That's the whole change — no Python edits, no rebuild, no factory file.
Restart the service and `/tasks/mem_forecast_arima/{train,predict}`
are live. See [`examples/`](examples/) for full configs covering CPU,
memory, and energy forecasting end-to-end.

### Kind reference

Every task carries a `features:` list — the first entry is the
target, additional entries are covariates. Each feature has `name`,
optional `value_range: [lo, hi]`, and (for `telemetry.source:
prometheus`) `query:`. Kind-specific knobs:

| `kind` | What it does | Common optional |
|---|---|---|
| `arima` | Univariate forecast (single or multi-step, native 95 % CI) | `steps_back`, `model_params: {p, d, q}` |
| `xgb` | Sliding-window forecast (recursive multi-step, no CI) | `steps_back` (window length, default 6), `model_params: {n_estimators, max_depth, eta, ...}` |
| `lstm` | Sliding-window forecast (direct multi-step, PyTorch) | `steps_back`, `batch_size`, `horizon` (trained max, default 1), `model_params: {hidden_size, num_epochs, ...}` |
| `drift` | NannyML univariate drift alerting on a chunk | `forecaster` (sibling task name, required), `chunk_size` (default 12), `metric` (default `jensen_shannon`) |

### Add a new model algorithm (lib-side change)

A new kind — say `prophet` or `transformer` — is three local edits:

1. New model class in `src/intelligence/ml/models/<kind>.py`
   implementing the `Model` protocol (`train` + `predict`).
2. New per-kind config schema in `src/intelligence/config/settings.py`
   alongside the existing `ArimaTaskConfig` / `XgbTaskConfig` /
   `LstmTaskConfig` / `DriftTaskConfig`, included in the
   `TaskInstanceConfig` discriminated union.
3. New builder in `src/intelligence/tasks/builders/<kind>.py` that
   constructs a `BaseTask` from the config block. Register it in
   `BUILDERS` in `tasks/builders/__init__.py`.

Users then enable it with `kind: <new>` in YAML.

### Add a new data source (lib-side change)

Sources like Kafka or OpenTelemetry are heavier — they extend
`TelemetryConfig.source`, add a new `TelemetrySource` Protocol
implementation, and add a branch in `build_loader_for_task`. See
`src/intelligence/telemetry/` for the existing two (static, prometheus).

## Layout

```
src/intelligence/            the library
├── api/                     FastAPI service + Pydantic schemas + HF push/pull
├── tasks/                   Task protocol + BaseTask + registry + loaders
│   ├── builders/            one builder per algorithm kind (arima/xgb/lstm/drift)
│   └── contracts/           per-task InputSpec
├── telemetry/               TelemetrySource Protocol + StaticSource + PrometheusSource
├── ml/
│   ├── models/              Model protocol + ArimaModel / XgbModel / LstmModel / DriftModel
│   └── trainers/            ModelTrainer + LSTM defs + metrics
├── config/                  typed config (pydantic-settings + YAML)
└── data/samples/            bundled sample CSVs (ship with the wheel)

examples/                    ready-to-run task configurations
├── cpu_forecast/            four kinds against a CPU PromQL
├── mem_forecast/            four kinds against a memory PromQL
├── k8s_cluster_metrics/     node + pod metrics for kube-prometheus-stack
└── energy_forecast/         template for Kepler-style power metrics

helm/intelligence/           Helm chart (deployment surface)
docker-compose.demo.yml      in-repo demo stack (image + bundled prom)
docker-compose.dev.yml       contributor overlay (build image from src)
compose/                     demo-only configs (intelligence.demo.yaml, prometheus.yml)
.github/workflows/           release pipeline (image + OCI chart → GHCR)
```

## Development

For contributors hacking on the code rather than deploying it:

```bash
uv sync --extra dev                  # install lib + dev deps
uv run uvicorn intelligence.api.service:app --port 3000   # run the service directly

make test                            # pytest (excludes smoke)
make lint                            # ruff check + format
make chart-lint                      # helm lint the chart
make e2e-dev                         # demo stack with the image rebuilt from local src
```

macOS arm64 quick notes:
- `brew install libomp` once — xgboost dlopens its native lib.

## Credits & funding

Originally built at **CeADAR Ireland** (University College Dublin) as
part of the [ICOS metaOS](https://www.icos-project.eu/) initiative.

- **CeADAR Ireland, UCD**: Jaydeep Samanta, Sebastian Cajas Ordoñez,
  Romila Ghosh, Dr. Andrés L. Suárez-Cetrulo, Dr. Ricardo Simón Carbajo.
- **National and Kapodistrian University of Athens (NKUA)**: contributors
  to the original ICOS intelligence module.

🇪🇺 Funded by the European Union's HORIZON research and innovation
programme under grant agreement No. 101070177.

## License

Apache 2.0. See [LICENSE](LICENSE).

## Citation

If you use this work, please cite the original ICOS paper:

```bibtex
@inproceedings{ICOS-paper,
  title     = {{ICOS An Intelligent MetaOS for the Continuum}},
  author    = {Garcia, Jordi and Masip-Bruin, Xavi and Giannopoulos, Anastasios and Trakadas, Panagiotis and Cajas Ordoñez, Sebastián A. and Samanta, Jaydeep and Suárez-Cetrulo, Andrés L. and Simón Carbajo, Ricardo and Michalke, Marc and Admela, Jukan and Jaworski, Artur and Kotliński, Marcin and Giammatteo, Gabriele and D'Andria, Francesco},
  year      = {2025},
  isbn      = {9798400715600},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  url       = {https://doi.org/10.1145/3721889.3721929},
  doi       = {10.1145/3721889.3721929},
  booktitle = {Proceedings of the 2nd International Workshop on MetaOS for the Cloud-Edge-IoT Continuum},
  pages     = {53–59},
  numpages  = {7},
  location  = {Rotterdam, Netherlands},
  series    = {MECC '25}
}
```
