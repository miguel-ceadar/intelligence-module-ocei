# ICOS Intelligence Coordination API — O-CEI Utility

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3721889.3721929-blue.svg)](https://doi.org/10.1145/3721889.3721929)

> **CeADAR's adaptation of the [ICOS intelligence-module](https://github.com/icos-project/intelligence-module)
> for the O-CEI continuum.** Same AI capabilities (forecasting, drift,
> SHAP) — relaxed coupling so the service runs against generic
> Prometheus-scraped telemetry on a vanilla k8s cluster, alongside (or
> instead of) the original ICOS metaOS deployment.

A small AI coordination service for the edge-cloud continuum: train and
serve models against telemetry, exposed as a single HTTP API. Tasks
compose a *data loader* and a *model* — adding a new (domain × algorithm)
pairing is one factory line.

## What ships

Four registered tasks (opt in via config):

| Task | Model | Use |
|---|---|---|
| `cpu_forecast_arima` | ARIMA | 1-step CPU forecast off the last observation |
| `cpu_forecast_xgb` | XGBoost | 1-step CPU forecast from a 6-observation window |
| `cpu_forecast_lstm` | PyTorch LSTM | 1-step CPU forecast from a 6-observation window |
| `cpu_forecast_arima_drift` | NannyML | Univariate drift detection paired with the ARIMA forecaster |

Two data sources, picked once per deployment in config:

- **`static`** — CSVs from the bundled `samples/` directory (dev / demo / tests).
- **`prometheus`** — PromQL `/api/v1/query_range` against Prometheus or Thanos.

## Quick start

### With Docker (recommended for a first look)

Boots the service plus a local Prometheus + node-exporter so the
`prometheus` data source works out of the box:

```bash
docker compose up -d --build --wait
curl http://localhost:3000/healthz
```

End-to-end exercise of all four tasks against the live stack
(Prometheus needs ~2 minutes of warmup before the suite trains):

```bash
make e2e        # = make up && pytest -m smoke
```

`make down` keeps the bento volume so trained models persist across
restarts; `make down-clean` drops it.

### With uv (local dev, no Docker)

```bash
uv sync
uv run uvicorn intelligence.api.service:app --port 3000
# or, for the full BentoML serving stack:
uv run bentoml serve intelligence.api.service:svc
```

Train against the bundled sample:

```bash
curl -X POST http://localhost:3000/tasks/cpu_forecast_arima/train \
  -H 'Content-Type: application/json' \
  -d '{"data_source": {"kind": "static", "name": "cpu_sample_dataset_orangepi.csv"}}'
```

Predict:

```bash
curl -X POST http://localhost:3000/tasks/cpu_forecast_arima/predict \
  -H 'Content-Type: application/json' \
  -d '{"input_series": {"cpu": [0.85]}}'
```

`cpu_forecast_xgb` and `cpu_forecast_lstm` need a 6-value window:

```bash
curl -X POST http://localhost:3000/tasks/cpu_forecast_xgb/predict \
  -H 'Content-Type: application/json' \
  -d '{"input_series": {"cpu": [0.30, 0.31, 0.29, 0.32, 0.30, 0.31]}}'
```

Drift detection (12-value chunk):

```bash
curl -X POST http://localhost:3000/tasks/cpu_forecast_arima_drift/train \
  -H 'Content-Type: application/json' \
  -d '{"data_source": {"kind": "static", "name": "cpu_sample_dataset_orangepi.csv"}}'

curl -X POST http://localhost:3000/tasks/cpu_forecast_arima_drift/predict \
  -H 'Content-Type: application/json' \
  -d '{"input_series": {"cpu": [0.30,0.31,0.29,0.32,0.30,0.31,0.30,0.29,0.31,0.32,0.30,0.31]}}'
```

## API

| Method | Path | |
|---|---|---|
| `GET`  | `/healthz`              | liveness |
| `GET`  | `/readyz`               | readiness — registry + bento store + per-task probes + bootstrap state |
| `GET`  | `/tasks`                | list enabled tasks |
| `POST` | `/tasks/{task}/train`   | train from a data source, persist a Bento |
| `POST` | `/tasks/{task}/predict` | predict from the trained Bento |
| `GET`  | `/models`               | list Bento models in the local store |
| `POST` | `/models/sync`          | push to or pull from a Hugging Face repo (opt-in) |

Each task carries an `InputSpec` (feature count, window length, value
range, units). Bad requests get a `422` before anything reaches the
model. Bentos saved by a task carry their `InputSpec` in
`custom_objects`; pulled / pretrained Bentos that lack a matching spec
are refused at predict time (override via `allow_unverified_models` per
task — see [Pretrained Bentos](#pretrained-bentos)).

## Configuration

YAML file at `INTELLIGENCE_CONFIG`. Env vars override individual fields
via `INTELLIGENCE_<SECTION>__<FIELD>` (double-underscore separates
nested sections).

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

Stays external. The service ships no in-process scheduler. Point a
`CronJob` at `POST /tasks/{task}/train` on whatever cadence makes
sense for the deployment.

### Hugging Face push / pull

```yaml
intelligence:
  model_repo:
    hf_enabled: true
    repo_id: CeADAR/intelligence-bentos
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

## Adding a task

Tasks compose `(data_loader, model)` and register by name. A new
variant is one factory in `src/intelligence/tasks/catalog.py`:

```python
@register_builtin("mem_forecast_arima")
def make_mem_forecast_arima(cfg: IntelligenceConfig) -> BaseTask:
    from intelligence.ml.models.arima import ArimaModel
    from intelligence.tasks.contracts import InputSpec

    return BaseTask(
        name="mem_forecast_arima",
        model=ArimaModel(p=3, d=1, q=0),
        data_loader=build_loader_for_task(cfg, "mem_forecast_arima", value_col="MEM"),
        input_spec=InputSpec(
            n_features=1,
            feature_names=["mem"],
            steps_back=1,
            value_range={"mem": (0.0, 1.0)},
        ),
    )
```

For an XGB or LSTM variant, pass a custom `prepare` so the loader
produces the supervised-structure components the trainer expects:

```python
data_loader=build_loader_for_task(
    cfg, "cpu_forecast_xgb",
    prepare=make_xgb_prepare(look_back=6, num_variables=1),
),
```

A new model algorithm is one new file under `src/intelligence/ml/models/`
implementing the `Model` protocol, plus one factory using it.

A drift task is one new factory using `DriftDetectionTask` and
`make_drift_prepare`:

```python
@register_builtin("mem_forecast_arima_drift")
def make_mem_forecast_arima_drift(cfg: IntelligenceConfig) -> BaseTask:
    from intelligence.tasks.contracts import InputSpec
    from intelligence.tasks.drift import DriftDetectionTask, make_drift_prepare

    return DriftDetectionTask(
        name="mem_forecast_arima_drift",
        forecaster_task_name="mem_forecast_arima",
        model=None,
        data_loader=build_loader_for_task(
            cfg, "mem_forecast_arima_drift",
            prepare=make_drift_prepare(value_col="MEM"),
        ),
        chunk_size=12,
        input_spec=InputSpec(n_features=1, feature_names=["mem"], steps_back=12),
    )
```

## Layout

```
src/intelligence/
├── api/           FastAPI service + Pydantic schemas + HF push/pull
├── tasks/         Task protocol, BaseTask, registry, catalog, loaders,
│   │               bootstrap, DriftDetectionTask
│   └── contracts/ per-task InputSpec
├── telemetry/     TelemetrySource protocol + StaticSource + PrometheusSource
├── ml/
│   ├── models/    Model protocol + ArimaModel / XgbModel / LstmModel
│   └── trainers/  ModelTrainer + LSTM defs + metrics + MLflow helper
├── config/        typed config (pydantic-settings + YAML)
└── data/samples/  bundled sample CSVs (ship with the wheel)
```

The four user-facing domains (tasks, api, telemetry, ml) map to the
architecture sketch in `modules.png`.

## Development

```bash
make install-dev    # uv sync --extra dev
make test           # pytest (excludes smoke)
make lint           # ruff check + format
make up / make down # docker compose lifecycle
make smoke          # pytest -m smoke against the running stack
make e2e            # one-shot: up + smoke, dumps logs on failure
```

Tests: 126 passing, 2 skipped. Integration tests spin up the service
in-process via `httpx`'s ASGI client — no subprocess boot needed.
Smoke tests run against a live compose stack (see Quick start).

macOS arm64 quick notes:
- `brew install libomp` once — xgboost dlopens its native lib.
- `xgboost==1.7.6` is pinned because 2.x segfaults during numpy interop.
- `kaleido>=1.0` is forced via `[tool.uv] override-dependencies` because
  nannyml pins a kaleido that has no ARM-Mac wheel.

## Provenance

Built on the [ICOS intelligence-module](https://github.com/icos-project/intelligence-module),
part of the [ICOS metaOS](https://www.icos-project.eu/) initiative. The
original work was funded by the EU HORIZON programme under grant
agreement No. 101070177.

### Credits

- **CeADAR Ireland** (University College Dublin): Jaydeep Samanta,
  Sebastian Cajas Ordoñez, Romila Ghosh, Dr. Andrés L. Suárez-Cetrulo,
  Dr. Ricardo Simón Carbajo.
- **National and Kapodistrian University of Athens (NKUA)**: contributors
  to the original ICOS intelligence module.

🇪🇺 This work has received funding from the European Union's HORIZON
research and innovation programme under grant agreement No. 101070177.

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
