# Examples

Concrete `(metric × model × use case)` configurations you can run as-is
or copy and edit. Each subdirectory is self-contained: a `README.md`
walks through what the example demonstrates, and a `config.yaml` is a
complete file you can mount into the service.

| Example | What it shows |
|---|---|
| [`cpu_forecast/`](cpu_forecast/) | CPU utilisation forecasting + drift detection. The canonical example — four task kinds against the same Prometheus query. |
| [`mem_forecast/`](mem_forecast/) | Memory utilisation forecasting. Same model kinds, different PromQL — illustrates "swap the metric, keep the algorithms." |
| [`k8s_cluster_metrics/`](k8s_cluster_metrics/) | Cluster-wide and namespace-scoped metrics from a typical kube-prometheus-stack deployment. PromQL recipes for node + pod CPU / memory. |
| [`energy_forecast/`](energy_forecast/) | Energy-consumption forecasting pattern. Stub — needs real data; ships the configuration shape only. |

## How to run one

Pick the example you want, mount its config into the container:

```bash
INTELLIGENCE_CONFIG_FILE=./examples/mem_forecast/config.yaml \
  docker compose up -d --build --wait
```

Or for local dev without Docker:

```bash
INTELLIGENCE_CONFIG=./examples/mem_forecast/config.yaml \
  uv run uvicorn intelligence.api.service:app --port 3000
```

The example configs use placeholder Prometheus endpoints. Override at
deploy time:

```bash
INTELLIGENCE_TELEMETRY__PROMETHEUS__ENDPOINT=https://my-prom.example \
INTELLIGENCE_CONFIG_FILE=./examples/cpu_forecast/config.yaml \
  docker compose up -d --build --wait
```

## How to combine examples

Each example's `config.yaml` declares its tasks under `tasks:`. To run
two examples simultaneously, copy both files' `tasks:` entries into
one config — task names are independent and the registry handles
arbitrarily many. No code change required.

## How to make a new example

1. Create a directory under `examples/` named after your domain.
2. Write a `config.yaml` declaring one or more tasks under `tasks:`,
   each picking a `kind:` (`arima` / `xgb` / `lstm` / `drift`) and a
   `feature:` matching the metric you care about.
3. Add a `README.md` describing the metric, the PromQL queries, and
   anything pilot-specific.

If you need a new algorithm or a new data source, see the project
[README](../README.md#extending) — those need lib-side additions, not
just config.
