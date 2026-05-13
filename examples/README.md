# Examples

Concrete `(metric × model × use case)` configurations you can run as-is
or copy and edit. Each subdirectory is self-contained: a `README.md`
walks through what the example demonstrates, and a `config.yaml` is a
complete file you can mount into the service.

> If you're new, start with the
> [Getting Started walkthrough](../docs/getting-started.md) — it uses
> `cpu_forecast/` as the anchor and takes you end-to-end from `helm
> install` to a `/predict` response.

| Example | What it shows |
|---|---|
| [`cpu_forecast/`](cpu_forecast/) | CPU utilisation forecasting + drift detection. The canonical example — four task kinds against the same Prometheus query. |
| [`mem_forecast/`](mem_forecast/) | Memory utilisation forecasting. Same model kinds, different PromQL — illustrates "swap the metric, keep the algorithms." |
| [`k8s_cluster_metrics/`](k8s_cluster_metrics/) | Cluster-wide and namespace-scoped metrics from a typical kube-prometheus-stack deployment. PromQL recipes for node + pod CPU / memory. |
| [`energy_forecast/`](energy_forecast/) | Energy-consumption forecasting against any watts-producing exporter. Ships a PromQL recipe table covering Kepler, IPMI, RAPL, PDU SNMP, smart-plugs and Redfish. |

## How to run one

Pick the example, mount its `config.yaml` into the published image,
and point at your Prometheus:

```bash
docker run -d --name icos-intelligence-ocei \
  -p 3000:3000 \
  -e INTELLIGENCE_CONFIG=/etc/intelligence/config.yaml \
  -e INTELLIGENCE_TELEMETRY__PROMETHEUS__ENDPOINT=https://my-prom.example \
  -v "$PWD/examples/mem_forecast/config.yaml:/etc/intelligence/config.yaml:ro" \
  -v intelligence-bentoml:/var/lib/bentoml \
  ghcr.io/miguel-ceadar/icos-intelligence-ocei:0.1.0
```

On Kubernetes via Helm, the same config goes under `config.intelligence`
in your values file — see the [chart README](../helm/intelligence/README.md).

For local dev without Docker (the contributor path), point the config
env var at the file and run uvicorn:

```bash
INTELLIGENCE_CONFIG=./examples/mem_forecast/config.yaml \
  uv run uvicorn intelligence.api.service:app --port 3000
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
   `features:` list. The first feature is the target (what gets
   forecast); additional entries are covariates.
3. Add a `README.md` describing the metric, the PromQL queries, and
   anything pilot-specific.

If you need a new algorithm or a new data source, see the project
[README](../README.md#extending) — those need lib-side additions, not
just config.
