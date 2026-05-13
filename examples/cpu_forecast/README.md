# CPU forecasting + drift detection

Four tasks against the same PromQL CPU expression — one per model kind
plus a NannyML drift detector paired with the ARIMA forecaster. This
is the canonical end-to-end example.

| Task | Kind | What it predicts |
|---|---|---|
| `cpu_forecast_arima` | arima | Next observation, single-step from the latest value |
| `cpu_forecast_xgb` | xgb | Next observation, from a 6-step window |
| `cpu_forecast_lstm` | lstm | Next observation, from a 6-step window (PyTorch) |
| `cpu_forecast_arima_drift` | drift | Drift alert on a 12-observation chunk |

## PromQL recipes

The shipped `config.yaml` uses node-exporter:

```promql
avg(rate(node_cpu_seconds_total{mode!="idle"}[30s]))
```

Common alternatives for other Prometheus setups:

| Stack | Query |
|---|---|
| node-exporter (bare-metal / VM) | `avg(rate(node_cpu_seconds_total{mode!="idle"}[30s]))` |
| cAdvisor (per-container) | `sum(rate(container_cpu_usage_seconds_total{namespace="my-ns"}[30s]))` |
| kube-state + metrics-server | `avg(rate(container_cpu_usage_seconds_total{pod=~"my-pod-.*"}[30s]))` |
| Cloud CPU credit / utilisation | provider-specific; needs your exporter's metric name |

Edit `query:` on each task block to match what your Prometheus actually
exposes. The same expression appears on all four tasks because they
forecast the same metric with different algorithms.

## Run it

```bash
docker run -d --name icos-intelligence-ocei \
  -p 3000:3000 \
  -e INTELLIGENCE_CONFIG=/etc/intelligence/config.yaml \
  -e INTELLIGENCE_TELEMETRY__PROMETHEUS__ENDPOINT=https://your-prom \
  -v "$PWD/examples/cpu_forecast/config.yaml:/etc/intelligence/config.yaml:ro" \
  -v intelligence-bentoml:/var/lib/bentoml \
  ghcr.io/miguel-ceadar/icos-intelligence-ocei:0.1.0

curl -X POST http://localhost:3000/tasks/cpu_forecast_arima/train \
  -H 'Content-Type: application/json' \
  -d '{"data_source": {"kind": "prometheus", "window": "24h", "step": "1m"}}'

curl -X POST http://localhost:3000/tasks/cpu_forecast_arima/predict \
  -H 'Content-Type: application/json' \
  -d '{"input_series": {"cpu": [0.42]}}'
```

## Try it in static mode (no Prometheus)

The bundled `cpu_sample_dataset_orangepi.csv` carries a univariate CPU
trace. Switch the config to static source and train from that file:

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

```bash
curl -X POST http://localhost:3000/tasks/cpu_forecast_arima/train \
  -H 'Content-Type: application/json' \
  -d '{"data_source": {"kind": "static", "name": "cpu_sample_dataset_orangepi.csv"}}'
```
