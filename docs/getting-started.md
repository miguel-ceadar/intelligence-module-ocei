# Getting started — from zero to forecast

A linear walkthrough: install the service on a Kubernetes cluster, point
it at an existing Prometheus, register one forecasting task, train it,
and call `/predict`.

We use the [`cpu_forecast`](../examples/cpu_forecast/) example as the
anchor because it works against a vanilla `node-exporter` deployment,
which most Prometheus stacks already scrape. Once you've reached
`/predict` once, swapping the PromQL for memory, energy, or any other
metric is one line.

## Before you start

You need:

- A Kubernetes cluster (kubectl context configured).
- A reachable Prometheus inside that cluster — note its in-cluster URL
  (typically `http://prometheus.<namespace>.svc:9090`).
- Helm 3.
- `node-exporter` (or any exporter publishing `node_cpu_seconds_total`)
  being scraped by that Prometheus. Confirm with:

  ```bash
  kubectl -n monitoring port-forward svc/prometheus 9090:9090 &
  curl -s 'http://localhost:9090/api/v1/query?query=node_cpu_seconds_total' \
    | head -c 200
  ```

  If you get a non-empty `"result"`, you're set. If your Prometheus
  doesn't expose `node_cpu_seconds_total`, see
  [PromQL recipes](../examples/cpu_forecast/README.md#promql-recipes)
  for alternatives — pick one and substitute it everywhere `query:`
  appears below.

## Step 1 — Install the chart

The chart is published as an OCI artifact at
`oci://ghcr.io/miguel-ceadar/charts/icos-intelligence-ocei`. Pin to a
released version (avoid `latest`).

Create a minimal values file:

```yaml
# values.yaml
config:
  intelligence:
    telemetry:
      source: prometheus
      prometheus:
        endpoint: http://prometheus.monitoring.svc:9090   # ← your Prometheus
    tasks:
      cpu_forecast_arima:
        kind: arima
        steps_back: 1
        features:
          - name: cpu
            value_range: [0.0, 1.0]
            query: 'avg(rate(node_cpu_seconds_total{mode!="idle"}[30s]))'

persistence:
  enabled: true
  size: 5Gi
```

Install:

```bash
helm install icos-intelligence-ocei \
  oci://ghcr.io/miguel-ceadar/charts/icos-intelligence-ocei \
  --version 0.1.0 \
  -f values.yaml
```

What the chart deploys: a Deployment, Service, ConfigMap (your tasks),
Secret (empty here — no Prometheus token needed for an unauthenticated
in-cluster Prometheus), and a PVC for the BentoML model store.

If your Prometheus requires a bearer token, add it to `secretEnv` in
`values.yaml` and reference it from the telemetry block:

```yaml
config:
  intelligence:
    telemetry:
      prometheus:
        endpoint: https://prometheus.example.com
        token_env: PROM_TOKEN

secretEnv:
  PROM_TOKEN: "your-bearer-token"
```

## Step 2 — Verify it's up

Wait for the pod to become ready, then port-forward:

```bash
kubectl wait --for=condition=ready pod \
  -l app.kubernetes.io/name=icos-intelligence-ocei --timeout=120s

kubectl port-forward svc/icos-intelligence-ocei 3000:3000 &

curl http://localhost:3000/healthz
# {"status":"ok"}

curl http://localhost:3000/tasks
# [{"name":"cpu_forecast_arima","model_type":"arima","has_drift":false}]
```

If `/tasks` is empty or returns a different name, your values file's
`tasks:` block didn't make it into the pod's ConfigMap — re-check the
helm release with `helm get values icos-intelligence-ocei`.

The full HTTP surface lives at `http://localhost:3000/docs` (Swagger
UI) once the service is up — useful for poking around past this
walkthrough.

## Step 3 — Train

Tell the service to pull the last 24 hours of CPU data from Prometheus
and fit an ARIMA model on it:

```bash
curl -X POST http://localhost:3000/tasks/cpu_forecast_arima/train \
  -H 'Content-Type: application/json' \
  -d '{"data_source": {"kind": "prometheus", "window": "24h", "step": "1m"}}'
```

Successful response:

```json
{
  "model_tag": "cpu_forecast_arima:2nrgmtnnxsfopjqf",
  "metrics": {"mse": 0.0012, "rmse": 0.035, "mape": 0.08, "mae": 0.02, "smape": 7.9}
}
```

The model is saved to the PVC under `/var/lib/bentoml` as
`cpu_forecast_arima:<version>`. `:latest` will resolve to this version
until you train again — pass `model_version: "<version>"` on a predict
request to pin to this specific one. Metric keys vary slightly per
kind (drift returns a different shape entirely).

Errors you might hit and what they mean:

| Status | Detail | Likely cause |
|---|---|---|
| `404` | `unknown task: ...` | The task name in the URL isn't registered — check `/tasks`. |
| `422` | (validation message) | Request body doesn't match `TrainRequest` — e.g. wrong `data_source.kind`, missing `window` / `step`. |
| `502` | `upstream telemetry error: ...` | Prometheus unreachable, refused the connection, or returned a 5xx. Test the endpoint from inside the cluster. |
| `501` | drift not trainable | You hit `/train` on a `kind: drift` task — drift consumes its forecaster's output, no separate fit. |

Two operational notes:

- **Concurrent `/train` calls succeed independently.** Both produce a
  new model version with its own tag; the most recent timestamp wins
  `:latest`. Duplicate POSTs from a misbehaving client pay the full
  cost twice.
- **Training shares the pod with `/predict`.** Long trains add
  `/predict` tail latency and can push memory past the chart's default
  4 Gi limit. For slow-training tasks, raise `resources.limits.memory`
  in your values file and uncomment the `startupProbe` block to give
  the first auto-train enough budget.

## Step 4 — Predict

Forecast the next CPU value, given the latest observation:

```bash
curl -X POST http://localhost:3000/tasks/cpu_forecast_arima/predict \
  -H 'Content-Type: application/json' \
  -d '{"input_series": {"cpu": [0.42]}}'
```

Response:

```json
{
  "prediction": [{"value": 0.45, "lower": 0.38, "upper": 0.52}],
  "metric_type": null,
  "model_version": "2nrgmtnnxsfopjqf"
}
```

`lower` / `upper` carry a 95 % confidence interval — ARIMA is one of
the kinds that exposes one natively. Recursive XGB and direct-output
LSTM leave both bounds `null`. `metric_type` is populated only for
drift tasks; for forecasting tasks it's `null`.

If the value you pass in `input_series.cpu` falls outside the task's
`value_range`, predict returns `422` with a message like
`feature 'cpu' value 1.4 at index 0 outside trained range [0.0, 1.0]`.
NaN / ±Inf values are rejected the same way. That's the contract check
catching a unit, scaling, or upstream-data mismatch before the model
sees it.

For a multi-step forecast, pass `horizon`:

```bash
curl -X POST http://localhost:3000/tasks/cpu_forecast_arima/predict \
  -H 'Content-Type: application/json' \
  -d '{"input_series": {"cpu": [0.42]}, "horizon": 5}'
```

`prediction` becomes a 5-element list. ARIMA and XGB accept arbitrary
horizons but cost scales with the value; LSTM refuses horizons above
the trained `output_size` with `422`.

Errors on `/predict`:

| Status | Detail | Likely cause |
|---|---|---|
| `404` | `unknown task: ...` | Task name in the URL isn't registered. |
| `422` | (validation message) | Request body doesn't match `PredictRequest`, or InputSpec rejected the window — wrong feature names, wrong length, NaN/Inf, value outside `value_range`, or horizon above trained `max_horizon`. |
| `503` | `no Bento ... in the local store; POST /tasks/.../train first` | No model trained yet (or the pinned version was deleted). |

## Step 5 — Where to go next

You now have one task running end-to-end. From here:

- **Swap the metric.** Edit `query:` in your values file, `helm upgrade`,
  re-train. The four examples under [`examples/`](../examples/) show the
  shape for CPU, memory, k8s cluster metrics, and energy.
- **Add more kinds.** Drop `cpu_forecast_xgb` / `cpu_forecast_lstm` /
  `cpu_forecast_arima_drift` blocks alongside the ARIMA one — see
  [`examples/cpu_forecast/config.yaml`](../examples/cpu_forecast/config.yaml).
  Each is independently trainable.
- **Schedule retraining.** Set `retraining.enabled: true` and
  `retraining.tasks: [cpu_forecast_arima]` in values — the chart adds
  a CronJob that POSTs to `/train` on the schedule you set.
- **Wire `/metrics` into Prometheus.** If you run kube-prometheus-stack,
  flip `serviceMonitor.enabled: true` to scrape per-task train/predict
  latency and error counters.
- **Pin a model version.** Once you have a model you trust, set
  `pinned_version:` on the task or pass `model_version` in the predict
  request. Reference: [README → Pretrained models](../README.md#pretrained-models).

If something didn't behave the way this walkthrough describes, open an
issue with the failing step number and the curl output.