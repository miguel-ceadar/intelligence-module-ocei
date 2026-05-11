# ICOS Intelligence Coordination API

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3721889.3721929-blue.svg)](https://doi.org/10.1145/3721889.3721929)

> **This is an adapted version of the [ICOS intelligence-module](https://github.com/icos-project/intelligence-module),
> reworked for the O-CEI continuum.** The original was built for the
> ICOS metaOS; this version drops the ICOS-specific coupling
> (meta-kernel, DataClay, agent-telemetry shapes) so the same AI
> capabilities run against generic sources on a vanilla k8s cluster.

A small AI coordination service for the edge-cloud continuum: train and
serve models against telemetry, exposed as a single HTTP API.

## What it does

A *task* trains a model from a data source and serves predictions. The
shipped task is `cpu_forecast_arima` — 1-step-ahead CPU forecasting via
ARIMA. Tasks are opt-in: only configured ones load. Each is composed
from a *data loader* and a *model adapter*, so adding a new domain or
algorithm is a small factory file. See [Adding a task](#adding-a-task).

## Quick start

```bash
uv sync
uv run bentoml serve intelligence.api.service:svc
# or, as a plain ASGI app:
uv run uvicorn intelligence.api.service:app
```

Train against a sample dataset:

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

## API

| Method | Path | |
|---|---|---|
| `GET`  | `/healthz`              | liveness |
| `GET`  | `/readyz`               | readiness (registry + bento store + per-task probes) |
| `GET`  | `/tasks`                | list enabled tasks |
| `POST` | `/tasks/{task}/train`   | train from a data source, persist a Bento |
| `POST` | `/tasks/{task}/predict` | predict from the trained Bento |
| `GET`  | `/models`               | list Bento models in the local store |

Each task carries an `InputSpec` (feature count, window length, value
range, units). Bad requests get a `422` with a useful message before
anything reaches the model.

## Configuration

YAML file at `INTELLIGENCE_CONFIG`. Env vars override individual fields
via `INTELLIGENCE_<SECTION>__<FIELD>` (double-underscore separates
nested sections).

```yaml
intelligence:
  enabled_tasks:
    - cpu_forecast_arima
  mlflow:
    tracking_uri: http://localhost:5000
  telemetry:
    source: static
```

`enabled_tasks` is validated against the registered catalog at
startup — typos fail loudly, not at first request.

## Adding a task

Tasks compose `(data_loader, model_adapter)` and register by name. A
new variant is one factory in `src/intelligence/tasks/catalog.py`:

```python
@register_builtin("mem_forecast_arima")
def make_mem_forecast_arima() -> BaseTask:
    from intelligence.models.arima import ArimaAdapter
    return BaseTask(
        name="mem_forecast_arima",
        model_adapter=ArimaAdapter(p=3, d=1, q=0),
        data_loader=static_csv_loader(value_col="MEM"),
    )
```

A new model algorithm is one new file under `src/intelligence/models/`
implementing the `ModelAdapter` protocol, plus one factory using it.

## Layout

```
src/intelligence/
├── adapters/   external-system clients
├── api/        FastAPI service + Pydantic schemas
├── config/     typed config (pydantic-settings + YAML)
├── contracts/  InputSpec — per-task input contract
├── models/     ModelAdapter implementations (ARIMA, ...)
├── tasks/      Task protocol, registry, BaseTask, builtin catalog
└── trainers/   framework-specific training methods
```

The `oasis/` directory holds the original ICOS implementation. New
development happens under `src/intelligence/`.

## Development

```bash
make install-dev    # uv sync --extra dev
make test           # pytest
make lint           # ruff check + format
```

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
