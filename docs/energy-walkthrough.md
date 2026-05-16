# Energy on Kubernetes — a production-flavoured helm walkthrough

A second walkthrough, picking up where
[`getting-started.md`](getting-started.md) leaves off and layering on
the patterns you typically reach for once a single task is live:

- Pre-created Secret for bearer tokens (not the demo-only `secretEnv`).
- Two tasks declared up front: a forecast and a drift verdict against
  the same series.
- `value_range` tuned to the hardware, demonstrably catching a
  unit-mismatch on `/predict`.
- A trained model version pinned in values so a future train can't
  silently change what `:latest` resolves to.
- A third task added post-install via `helm upgrade -f`.

Energy is the worked example: the synthetic exporter shipped under
[`examples/energy_forecast/k8s/`](../examples/energy_forecast/k8s/) emits a
sinusoidal watts gauge so you can run this end-to-end without real
hardware. Swap the PromQL and the units and the same flow drives CPU,
memory, or any other metric you scrape.

## Before you start

You need:

- A Kubernetes cluster (kubectl context configured).
- Helm 3.
- A Prometheus reachable from the cluster, scraping a watts-shaped
  energy metric. The PromQL recipes for Kepler, IPMI, RAPL,
  smart-plugs, PDU SNMP and Redfish are in
  [`examples/energy_forecast/README.md`](../examples/energy_forecast/README.md#example-energy-promql-recipes).

If you don't have either of those, the synthetic stack under
[`examples/energy_forecast/k8s/synthetic-stack.yaml`](../examples/energy_forecast/k8s/synthetic-stack.yaml)
gives you both — a tiny python-based exporter emitting
`synthetic_energy_watts` as a 10-minute sinusoid plus noise, and a
Prometheus scraping it every 5 s. This walkthrough uses it as the
reference data source.

```bash
kubectl apply -f examples/energy_forecast/k8s/synthetic-stack.yaml
kubectl -n energy-forecast wait --for=condition=ready \
  pod -l app=prometheus --timeout=60s
```

Give it ~5 minutes to accumulate samples before Step 4 — training
reads a 10-minute window.

## Step 1 — Pre-create the Secret

The walkthrough's values file references its Secret by name. Create it in the
same namespace before installing the chart — Helm refuses to render
when `existingSecretName` points at a Secret that doesn't exist.

```bash
API_TOKEN=$(openssl rand -hex 32)
kubectl -n energy-forecast create secret generic intelligence-secrets \
  --from-literal=API_TOKEN="$API_TOKEN" \
  --from-literal=PROM_TOKEN=unused-in-this-test

echo "$API_TOKEN"   # save it — clients need this for /train, /predict, /tasks
```

`API_TOKEN` is enforced: every request to `/train`, `/predict`, and
`/tasks` must carry `Authorization: Bearer $API_TOKEN`. `/healthz`
and `/readyz` stay unauthenticated so kubelet probes still work.

`PROM_TOKEN` is mounted on the pod and sent as `Authorization: Bearer`
to Prometheus. The synthetic Prometheus does not enforce it — vanilla
Prom has no built-in bearer auth on read endpoints — but the wiring
is in place for production where Prom typically sits behind an
auth-aware reverse proxy.

In production, generate this Secret via sealed-secrets, external-secrets,
or Vault rather than `kubectl create secret` — cleartext should never
touch a values file or your shell history.
[`secret.example.yaml`](../examples/energy_forecast/k8s/secret.example.yaml)
is a template you can feed those tools.

## Step 2 — Install the chart

```bash
helm install intelligence-energy \
  oci://ghcr.io/miguel-ceadar/charts/icos-intelligence-ocei \
  --version 0.2.10 -n energy-forecast \
  -f examples/energy_forecast/k8s/values-walkthrough.yaml
```

The walkthrough's values file declares two tasks against
`synthetic_energy_watts` with `value_range: [50.0, 300.0]`:

- `energy_forecast_arima` — one-step-ahead ARIMA with native 95 % CI.
- `energy_forecast_arima_drift` — NannyML drift verdict on the same
  series, paired to the forecast task via `forecaster:` so alerts can
  route together.

It also flips on `auth.token_env: API_TOKEN`, points at the in-cluster
Prometheus from Step 0, and references `intelligence-secrets` for both
bearer tokens. If you're aiming this at your own Prometheus, edit
`telemetry.prometheus.endpoint`; if your hardware draws more than
300 W, widen `value_range`.

## Step 3 — Verify

```bash
kubectl -n energy-forecast wait --for=condition=ready pod \
  -l app.kubernetes.io/instance=intelligence-energy --timeout=180s

kubectl -n energy-forecast port-forward \
  svc/intelligence-energy-icos-intelligence-ocei 3000:3000 &

curl http://localhost:3000/healthz
# {"status":"ok","version":"0.2.10"}

curl http://localhost:3000/readyz
# {"status":"ready","tasks":2,"version":"0.2.10"}

curl -i http://localhost:3000/tasks
# HTTP/1.1 401 Unauthorized

curl -s -H "Authorization: Bearer $API_TOKEN" http://localhost:3000/tasks
# {"tasks":[
#   {"name":"energy_forecast_arima","model_type":"arima","has_drift":true,"is_loaded":false},
#   {"name":"energy_forecast_arima_drift","model_type":"none","has_drift":false,"is_loaded":false}
# ]}
```

`/tasks` returning 401 without the bearer is `auth.token_env` doing
its job. `/healthz` and `/readyz` stay open so probes work without
having to inject the token via downward API. `is_loaded: false` means
neither task has a trained model yet — that's Step 4.

## Step 4 — Train both tasks

Pull a 10-minute window at 5-second resolution. With the synthetic
exporter on a 10-minute sinusoid, this captures one full cycle plus
boundary samples.

```bash
curl -s -X POST http://localhost:3000/tasks/energy_forecast_arima/train \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $API_TOKEN" \
  -d '{"data_source": {"kind": "prometheus", "window": "10m", "step": "5s"}}'
```

```json
{
  "model_tag": "energy_forecast_arima:nqgzbdcpuslzfex2",
  "metrics": {"mse": 53.83, "rmse": 7.34, "mape": 0.038, "mae": 6.32, "smape": 3.9, "r2": 0.84}
}
```

Save the version tag (`nqgzbdcpuslzfex2` above) — Step 7 pins it.

```bash
curl -s -X POST http://localhost:3000/tasks/energy_forecast_arima_drift/train \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $API_TOKEN" \
  -d '{"data_source": {"kind": "prometheus", "window": "10m", "step": "5s"}}'
```

```json
{"model_tag": "energy_forecast_arima_drift:nraqkuspuslzfex2", "metrics": {"reference_size": 121}}
```

`reference_size: 121` = 10 min ÷ 5 s + 1 boundary sample. Drift train
establishes a reference distribution; subsequent `/predict` calls
compare incoming windows against it.

Train against the longest window you can in production — 24 h is the
default in
[`examples/cpu_forecast/`](../examples/cpu_forecast/) and gives the model
the full diurnal pattern that 10 min cannot.

## Step 5 — Predict

Forecast `horizon=5` from the latest sample. Your input value should
be a recent reading (`curl http://localhost:9090/api/v1/query?query=synthetic_energy_watts`
against the Prometheus port-forward gives you one):

```bash
curl -s -X POST http://localhost:3000/tasks/energy_forecast_arima/predict \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $API_TOKEN" \
  -d '{"input_series": {"energy": [165.4]}, "horizon": 5}'
```

```json
{
  "prediction": [
    {"value": 178.44, "lower": 164.94, "upper": 191.94},
    {"value": 175.21, "lower": 159.74, "upper": 190.69},
    {"value": 172.49, "lower": 154.97, "upper": 190.01},
    {"value": 170.90, "lower": 150.25, "upper": 191.55},
    {"value": 169.38, "lower": 145.94, "upper": 192.83}
  ],
  "model_version": "nqgzbdcpuslzfex2"
}
```

Drift verdict on a 24-sample window (matches the task's `chunk_size`):

```bash
curl -s -X POST http://localhost:3000/tasks/energy_forecast_arima_drift/predict \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $API_TOKEN" \
  -d '{"input_series": {"energy": [148, 152, 159, 161, 168, 172, 175, 178, 182, 184, 187, 188, 190, 191, 192, 193, 192, 191, 188, 184, 180, 175, 168, 162]}}'
```

```json
{
  "prediction": {
    "drift_detected": false,
    "n_chunks": 1,
    "metric": "jensen_shannon",
    "forecaster": "energy_forecast_arima"
  },
  "model_version": "nraqkuspuslzfex2"
}
```

`forecaster` echoes the sibling forecast task — route alerts to
whoever owns the forecast. `metric` is whichever NannyML metric was
configured (Jensen-Shannon divergence by default).

## Step 6 — `value_range` catches a unit error

The walkthrough's values file declared `value_range: [50.0, 300.0]`. Send a
999 W reading and the request never reaches the model:

```bash
curl -s -X POST http://localhost:3000/tasks/energy_forecast_arima/predict \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $API_TOKEN" \
  -d '{"input_series": {"energy": [999.0]}}'
```

```
HTTP/1.1 422 Unprocessable Entity
{"detail":"feature 'energy' value 999.0 at index 0 outside trained range [50.0, 300.0]"}
```

That's the InputSpec contract refusing input that doesn't match the
trained envelope — almost always a unit confusion (kW vs W) or a
scaling bug upstream. NaN and ±Inf are rejected the same way.

Tune `value_range` to actual hardware draw on day one. The default in
[`examples/energy_forecast/config.yaml`](../examples/energy_forecast/config.yaml)
is `[0.0, 500.0]` — fine for a single node, far too narrow for a rack PDU.

## Step 7 — Pin a model version

Lock predict to the model from Step 4 so a future train can't move
`:latest` under live traffic. Edit
[`values-walkthrough.yaml`](../examples/energy_forecast/k8s/values-walkthrough.yaml)
and add `pinned_version:` to the arima task block:

```yaml
config:
  intelligence:
    tasks:
      energy_forecast_arima:
        kind: arima
        steps_back: 1
        pinned_version: nqgzbdcpuslzfex2   # ← tag from Step 4
        features:
          - name: energy
            value_range: [50.0, 300.0]
            query: 'synthetic_energy_watts'
```

```bash
helm upgrade intelligence-energy \
  oci://ghcr.io/miguel-ceadar/charts/icos-intelligence-ocei \
  --version 0.2.10 -n energy-forecast \
  -f examples/energy_forecast/k8s/values-walkthrough.yaml
```

The chart annotates the deployment with a checksum of the rendered
ConfigMap, so the pod rolls automatically when task config changes.

Train a second time to verify the pin holds — the new tag becomes
`:latest`, but predict should still return the pinned one:

```bash
curl -s -X POST http://localhost:3000/tasks/energy_forecast_arima/train \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $API_TOKEN" \
  -d '{"data_source": {"kind": "prometheus", "window": "10m", "step": "5s"}}'
# {"model_tag":"energy_forecast_arima:s66wbbspusqjmfrr", ...}   ← new :latest

curl -s -X POST http://localhost:3000/tasks/energy_forecast_arima/predict \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $API_TOKEN" \
  -d '{"input_series": {"energy": [165.4]}}'
# {"prediction":[...], "model_version":"nqgzbdcpuslzfex2"}      ← still pinned
```

You can also pin per-request without changing values — pass
`model_version: "<tag>"` on the predict body. Useful for A/B-ing two
trained versions side by side without rolling the pod.

## Step 8 — Add a task post-install

Adding tasks doesn't need a reinstall — `helm upgrade -f` with an
extra values file is enough. The overlay file
[`values-with-xgb.yaml`](../examples/energy_forecast/k8s/values-with-xgb.yaml)
declares an XGBoost task against the same metric:

```bash
helm upgrade intelligence-energy \
  oci://ghcr.io/miguel-ceadar/charts/icos-intelligence-ocei \
  --version 0.2.10 -n energy-forecast \
  -f examples/energy_forecast/k8s/values-walkthrough.yaml \
  -f examples/energy_forecast/k8s/values-with-xgb.yaml
```

`/tasks` reflects the change after the pod rolls:

```bash
curl -s -H "Authorization: Bearer $API_TOKEN" http://localhost:3000/tasks
# {"tasks":[..., {"name":"energy_forecast_xgb","model_type":"xgb","has_drift":true,"is_loaded":false}]}
```

Train and predict it like any other task — XGB takes a window of
`steps_back: 6` samples on predict:

```bash
curl -s -X POST http://localhost:3000/tasks/energy_forecast_xgb/train \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $API_TOKEN" \
  -d '{"data_source": {"kind": "prometheus", "window": "10m", "step": "5s"}}'

curl -s -X POST http://localhost:3000/tasks/energy_forecast_xgb/predict \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $API_TOKEN" \
  -d '{"input_series": {"energy": [148, 152, 159, 161, 168, 172]}}'
# {"prediction":[{"value":171.27,"lower":null,"upper":null}], ...}
```

`lower` / `upper` stay `null` — recursive XGB and direct-output LSTM
don't ship native confidence intervals; ARIMA is the only kind in the
catalog that does.

## Where to go next

You've shipped a forecast task, a drift task, an XGBoost peer task, a
pinned model, and a bearer-token gate — broadly the production
loadout minus retraining and observability. From here:

- **Recurring retraining** — flip `retraining.enabled: true` and list
  the tasks under `retraining.tasks:`. The chart emits a CronJob that
  POSTs `/train` on the schedule with the bearer token automatically
  attached. See the
  [chart README](../helm/intelligence/README.md#retraining).
- **/metrics → Prometheus Operator** — set
  `serviceMonitor.enabled: true` to wire per-task train / predict
  latency and error counters into kube-prometheus-stack.
- **NetworkPolicy** — `networkPolicy.enabled: true` restricts pod
  egress to DNS plus what you explicitly list (your Prometheus, the
  HuggingFace hub for `/models/sync`, etc.).
- **Multi-replica** — the chart README's
  [multi-replica section](../helm/intelligence/README.md#multi-replica-considerations)
  walks through the three valid patterns (RWX volume, HF-pushed model
  store, partition-per-task).
- **Real exporter** — swap the synthetic data source for one of the
  recipes in
  [`examples/energy_forecast/README.md`](../examples/energy_forecast/README.md#example-energy-promql-recipes)
  (Kepler, IPMI, RAPL, smart-plug, PDU). Only `query:` and
  `value_range:` change.

## Tear-down

```bash
helm uninstall intelligence-energy -n energy-forecast
kubectl delete pvc -n energy-forecast --all
kubectl delete -f examples/energy_forecast/k8s/synthetic-stack.yaml
```
