# ICOS Intelligence Module

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3721889.3721929-blue.svg)](https://doi.org/10.1145/3721889.3721929)

> Train forecasting and drift-detection models against your existing
> Prometheus metrics, served as a small HTTP API. Plugs into the
> monitoring you already run — no glue code, no separate training infra.

Point the service at your Prometheus, list the PromQL queries for the
metrics you care about, and you get `/train` + `/predict` endpoints per
metric. ARIMA, XGBoost, LSTM, and NannyML drift detection ship out of
the box; adding a new `(metric × algorithm)` pairing is one factory line
in `src/intelligence/tasks/catalog.py`.

Originally developed by **CeADAR Ireland** for the
[ICOS metaOS project](https://github.com/icos-project/intelligence-module).
This iteration relaxes the ICOS-specific coupling so the same service
runs on any vanilla Kubernetes + Prometheus stack, beside or instead of
the original ICOS deployment.

## What's included

Four built-in tasks (opt in via config):

| Task | Model | Use |
|---|---|---|
| `cpu_forecast_arima` | ARIMA | 1-step CPU forecast off the last observation |
| `cpu_forecast_xgb` | XGBoost | 1-step CPU forecast from a 6-observation window |
| `cpu_forecast_lstm` | PyTorch LSTM | 1-step CPU forecast from a 6-observation window |
| `cpu_forecast_arima_drift` | NannyML | Univariate drift detection paired with the ARIMA forecaster |

Two data sources, picked once per deployment:

- **`prometheus`** — PromQL `/api/v1/query_range` against your Prometheus or Thanos.
- **`static`** — CSVs from the bundled `samples/` directory; used for demos and tests.

## Quick start

### See it work (no setup)

```bash
make e2e
```

Brings up the service alongside a sample Prometheus + node-exporter in
Docker (~3 min the first time, image build included) and exercises
train + predict on all four tasks. Use this to feel out the API before
wiring it to your own infra. `make down-demo` tears it down.

### Deploy against your Prometheus

```bash
cp .env.example .env
# edit .env — set INTELLIGENCE_TELEMETRY__PROMETHEUS__ENDPOINT, plus
# TOKEN_ENV + the token if your Prometheus needs auth.
docker compose up -d --build --wait
curl http://localhost:3000/healthz
```

`compose/intelligence.yaml` carries the task set and the PromQL queries.
Edit `endpoint:` there for a permanent setting, or leave the placeholder
and override via the env var — env wins.

### Local dev (no Docker)

```bash
uv sync
uv run uvicorn intelligence.api.service:app --port 3000
```

### Kubernetes (Helm)

A Helm chart at [`helm/intelligence`](helm/intelligence/) installs the
service alongside ConfigMap / Secret / PVC / optional ServiceMonitor +
retraining CronJob. See the [chart README](helm/intelligence/README.md)
for values and the multi-replica caveat.

```bash
helm install intelligence ./helm/intelligence -f your-values.yaml
```

### Calling the API

Train + predict on the bundled CSV sample — works without a Prometheus:

```bash
curl -X POST http://localhost:3000/tasks/cpu_forecast_arima/train \
  -H 'Content-Type: application/json' \
  -d '{"data_source": {"kind": "static", "name": "cpu_sample_dataset_orangepi.csv"}}'

curl -X POST http://localhost:3000/tasks/cpu_forecast_arima/predict \
  -H 'Content-Type: application/json' \
  -d '{"input_series": {"cpu": [0.85]}}'
```

Same call, training from a Prometheus window instead:

```bash
... -d '{"data_source": {"kind": "prometheus", "window": "24h", "step": "1m"}}'
```

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
`pinned_version` (request wins; see [Pretrained Bentos](#pretrained-bentos)
for the validation rules around pulled models).

Predict requests accept an optional `horizon` (default `1`). The
response's `prediction` field is a list of `{value, lower, upper}`
points of length `horizon`; `lower`/`upper` carry a 95 % confidence
interval when the model exposes one (ARIMA does; recursive XGB and
direct-output LSTM leave them empty).

### Observability

`/metrics` exposes Prometheus-format counters and histograms: HTTP
request count + latency (route-normalised so per-task paths collapse
to `/tasks/{task}/...`), per-task train and predict counts + durations,
and a gauge for registered tasks. Probe endpoints (`/healthz`,
`/readyz`) and `/metrics` itself are excluded to keep the time series
honest.

Logs are emitted as JSON to stdout (`{timestamp, level, logger,
message, request_id, ...}`). Every HTTP request carries a short
`request_id` you can grep across logs for one call's full trace.

## Configuration

The service reads a YAML config from `INTELLIGENCE_CONFIG`. Any field
can be overridden by an env var — `INTELLIGENCE_<SECTION>__<FIELD>`
(double underscore separates nested sections, env wins over YAML).

### Minimal — static, bundled samples

```yaml
intelligence:
  enabled_tasks:
    - cpu_forecast_arima
    - cpu_forecast_xgb
    - cpu_forecast_lstm
    - cpu_forecast_arima_drift
  telemetry:
    source: static
```

### Prometheus

```yaml
intelligence:
  enabled_tasks:
    - cpu_forecast_arima
  telemetry:
    source: prometheus
    prometheus:
      endpoint: http://prometheus.monitoring.svc:9090
      token_env: PROM_TOKEN          # or token_file: /var/run/secrets/prom
      tls_skip_verify: false
      queries:
        cpu_forecast_arima: 'avg(rate(node_cpu_seconds_total{mode="user"}[1m]))'
```

Train request switches to:

```bash
curl -X POST http://localhost:3000/tasks/cpu_forecast_arima/train \
  -H 'Content-Type: application/json' \
  -d '{"data_source": {"kind": "prometheus", "window": "24h", "step": "1m"}}'
```

`enabled_tasks` is validated against the registered catalog at startup
— typos fail loudly, not at first request. Same for `source: prometheus`
without a `prometheus:` block.

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

Stays external. The service ships no in-process scheduler at this stage. Point a
`CronJob` at `POST /tasks/{task}/train` on whatever cadence makes
sense for the deployment.

### Hugging Face push / pull

```yaml
intelligence:
  model_repo:
    hf_enabled: true
    repo_id: ICOS-AI/ICOS-AI_icos_models
```

The HF token is read from `HF_TOKEN` at request time — not stored in
config. Then:

```bash
export HF_TOKEN=...
curl -X POST http://localhost:3000/models/sync \
  -H 'Content-Type: application/json' \
  -d '{"action": "push", "model_tag": "metrics_utilization_model_arima:latest"}'

curl -X POST http://localhost:3000/models/sync \
  -H 'Content-Type: application/json' \
  -d '{"action": "pull", "model_tag": "metrics_utilization_model_arima:abc123"}'
```

### Pretrained Bentos

A pulled Bento becomes available to a task only if its stored
`input_spec` matches the task's spec (same `n_features`,
`feature_names`, `steps_back`). Older Bentos that predate the contract
are refused at predict time — the pull still works (the artifact lands
in the local store), but `predict` won't serve from it. Override per
task with `allow_unverified_models=True` when intentionally accepting
the risk.

## Extending

A task is a `(data source × model algorithm)` pairing declared in
YAML. The lib ships four algorithm **kinds**; everything else — which
metric, which PromQL, which task name — is config. To get any new
forecast wired up, you add a block under `tasks:` and pick a `kind:`.

### Add a task (no code)

Drop a block into your `config.yaml` (or `compose/intelligence.yaml`):

```yaml
intelligence:
  tasks:
    mem_forecast_arima:
      kind: arima
      feature: mem
      value_range: [0.0, 1.0]
      steps_back: 1
      query: '1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)'
```

That's the whole change — no Python edits, no rebuild, no factory file.
Restart the service and `/tasks/mem_forecast_arima/{train,predict}`
are live. See [`examples/`](examples/) for full configs covering CPU,
memory, and energy forecasting end-to-end.

### Kind reference

| `kind` | What it does | Required | Common optional |
|---|---|---|---|
| `arima` | Univariate forecast (single or multi-step, native 95 % CI) | `feature`, `query`† | `steps_back`, `value_range`, `model_params: {p, d, q}` |
| `xgb` | Sliding-window forecast (recursive multi-step, no CI) | `feature`, `query`† | `steps_back` (window length, default 6), `value_range`, `model_params: {n_estimators, max_depth, eta, ...}` |
| `lstm` | Sliding-window forecast (direct multi-step, PyTorch) | `feature`, `query`† | `steps_back`, `batch_size`, `horizon` (trained max, default 1), `model_params: {hidden_size, num_epochs, ...}` |
| `drift` | NannyML univariate drift alerting on a chunk | `feature`, `forecaster` (name of a sibling task), `query`† | `chunk_size` (default 12), `value_range`, `metric` (default `jensen_shannon`) |

† `query:` required when `telemetry.source: prometheus`; ignored for
`static` (the train request body supplies the CSV name).

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

Pilots then enable it with `kind: <new>` in YAML.

### Add a new data source (lib-side change)

Sources like Kafka or OpenTelemetry are heavier — they extend
`TelemetryConfig.source`, add a new `TelemetrySource` Protocol
implementation, and add a branch in `build_loader_for_task`. See
`src/intelligence/telemetry/` for the existing two (static, prometheus).

## Layout

```
src/intelligence/         the library
├── api/                  FastAPI service + Pydantic schemas + HF push/pull
├── tasks/                Task protocol + BaseTask + registry + loaders
│   ├── builders/         one builder per algorithm kind (arima/xgb/lstm/drift)
│   └── contracts/        per-task InputSpec
├── telemetry/            TelemetrySource Protocol + StaticSource + PrometheusSource
├── ml/
│   ├── models/           Model protocol + ArimaModel / XgbModel / LstmModel
│   └── trainers/         ModelTrainer + LSTM defs + metrics + MLflow helper
├── config/               typed config (pydantic-settings + YAML)
└── data/samples/         bundled sample CSVs (ship with the wheel)

examples/                 ready-to-run task configurations
├── cpu_forecast/         four kinds against a CPU PromQL
├── mem_forecast/         four kinds against a memory PromQL
└── energy_forecast/      template for Kepler-style power metrics
```

The four user-facing domains (tasks, api, telemetry, ml) map to the
architecture sketch in `modules.png`.

## Development

```bash
make install-dev               # uv sync --extra dev
make test                      # pytest (excludes smoke)
make lint                      # ruff check + format
make up / make down            # standalone compose (intelligence only)
make up-demo / make down-demo  # demo overlay (adds in-stack prom + node-exporter)
make smoke                     # pytest -m smoke against whichever stack is running
make e2e                       # one-shot: demo overlay + smoke, dumps logs on failure
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
