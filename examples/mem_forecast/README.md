# Memory forecasting

Same task kinds as `cpu_forecast/`, swapped onto a memory metric. The
point of this example: adding a new `(metric × model)` pairing is
config, not code. The lib's algorithm kinds don't know whether `cpu`
or `mem` is being forecast — they read whatever feature you declare.

| Task | Kind |
|---|---|
| `mem_forecast_arima` | arima |
| `mem_forecast_xgb` | xgb |
| `mem_forecast_lstm` | lstm |
| `mem_forecast_arima_drift` | drift |

## PromQL recipes

Memory typically isn't a rate metric — it's a level (bytes available)
divided by a level (total bytes) to make it fractional. Examples:

| Stack | Query (fraction of memory used) |
|---|---|
| node-exporter | `1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)` |
| cAdvisor (per-container) | `sum(container_memory_working_set_bytes{namespace="my-ns"}) / sum(machine_memory_bytes)` |
| Kubernetes pod | `container_memory_working_set_bytes{pod="my-pod"} / container_spec_memory_limit_bytes{pod="my-pod"}` |

The `value_range: [0.0, 1.0]` in the config matches fractional memory
use. If your query returns raw bytes instead, widen or remove
`value_range` — InputSpec validation enforces the range at predict time.

## Run it

```bash
docker run -d --name icos-intelligence-ocei \
  -p 3000:3000 \
  -e INTELLIGENCE_CONFIG=/etc/intelligence/config.yaml \
  -e INTELLIGENCE_TELEMETRY__PROMETHEUS__ENDPOINT=https://your-prom \
  -v "$PWD/examples/mem_forecast/config.yaml:/etc/intelligence/config.yaml:ro" \
  -v intelligence-bentoml:/var/lib/bentoml \
  ghcr.io/miguel-ceadar/icos-intelligence-ocei:0.1.0

curl -X POST http://localhost:3000/tasks/mem_forecast_arima/train \
  -H 'Content-Type: application/json' \
  -d '{"data_source": {"kind": "prometheus", "window": "24h", "step": "1m"}}'

curl -X POST http://localhost:3000/tasks/mem_forecast_arima/predict \
  -H 'Content-Type: application/json' \
  -d '{"input_series": {"mem": [0.61]}}'
```

## Try it in static mode

The bundled `node_3_utilisation_sample_dataset.csv` carries CPU+MEM
side-by-side. The loader picks the `MEM` column automatically when
`feature: mem` is set:

```bash
curl -X POST http://localhost:3000/tasks/mem_forecast_arima/train \
  -H 'Content-Type: application/json' \
  -d '{"data_source": {"kind": "static", "name": "node_3_utilisation_sample_dataset.csv"}}'
```

## What changed vs. `cpu_forecast/`

Comparing `config.yaml`:

- `feature: cpu` → `feature: mem`
- Task names renamed to `mem_forecast_*`
- `query:` swapped for a memory PromQL expression
- Drift task's `forecaster:` points at `mem_forecast_arima`

Nothing else. No new Python, no new builders, no new model classes.
That's the M+N composition claim in practice.
