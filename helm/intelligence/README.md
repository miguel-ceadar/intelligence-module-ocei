# icos-intelligence-ocei — Helm chart

Deploys the intelligence service to Kubernetes: Deployment, Service,
ConfigMap (for the task YAML), optional Secret, optional PVC for the
BentoML model store, optional ServiceMonitor + retraining CronJob.

## Quick install

The chart is published as an OCI artifact alongside the image. Pin to
a release (`latest` drifts under you):

```bash
helm install icos-intelligence-ocei \
  oci://ghcr.io/miguel-ceadar/charts/icos-intelligence-ocei \
  --version 0.1.0 \
  --set config.intelligence.telemetry.prometheus.endpoint=https://prom.example
```

Or with a local checkout + a values file:

```yaml
# values.prod.yaml
image:
  repository: ghcr.io/miguel-ceadar/icos-intelligence-ocei
  tag: "0.1.0"   # leave empty to follow .Chart.AppVersion

config:
  intelligence:
    telemetry:
      source: prometheus
      prometheus:
        endpoint: https://prom.yourorg.com
        token_env: PROM_TOKEN
    tasks:
      cpu_forecast_arima:
        kind: arima
        feature: cpu
        value_range: [0.0, 1.0]
        steps_back: 1
        query: 'avg(rate(node_cpu_seconds_total{mode!="idle"}[30s]))'

secretEnv:
  PROM_TOKEN: "your-bearer-token"

persistence:
  enabled: true
  size: 20Gi
```

```bash
helm install icos-intelligence-ocei ./helm/intelligence -f values.prod.yaml
```

Every value is documented inline in `values.yaml`. The most common
overrides:

| Path | What |
|---|---|
| `image.repository` / `image.tag` | which image to pull |
| `config.intelligence.tasks.*` | which tasks to register and how |
| `config.intelligence.telemetry.prometheus.endpoint` | where to scrape from |
| `secretEnv.PROM_TOKEN` / `secretEnv.HF_TOKEN` | bearer tokens (rendered into a Secret) |
| `persistence.enabled` / `size` | local BentoML model store volume |
| `serviceMonitor.enabled` | wire `/metrics` to Prometheus Operator |
| `retraining.enabled` / `retraining.schedule` | CronJob hitting `/tasks/*/train` |

## Multi-replica considerations

The chart defaults to `replicaCount: 1` because the BentoML model store
is a local filesystem under `/var/lib/bentoml` — one pod owns it.
Running more than one replica needs one of:

- **ReadWriteMany volume** — set `persistence.accessMode: ReadWriteMany`
  (NFS, EFS, CephFS, Filestore, etc.). All replicas share one store.
  Concurrent `/train` calls across replicas race, so stagger external
  triggers.
- **HF push after each train** — turn on `model_repo.hf_enabled` +
  `HF_TOKEN`, and have each replica push to a shared HF repo. Pull-on-
  startup fills the local store. Eventually consistent, not real-time.
- **One replica per task** — partition tasks across deployments. Most
  pilots don't need this; reach for it only when training throughput
  matters more than operational simplicity.

Pick one before scaling past `replicaCount: 1`.

## Retraining

`retraining.enabled: true` creates a CronJob that POSTs to
`/tasks/{name}/train` on a schedule. The data source comes from
`retraining.dataSource` (typically `kind: prometheus` with a window
and step). Use `concurrencyPolicy: Forbid` to skip an overlapping
run if a previous one is still going.

For one-shot first-train-on-boot (rather than periodic), set
`bootstrap.auto_train_on_startup: true` on each task in
`config.intelligence.tasks.*.bootstrap` instead.

## Observability

`/metrics` exposes Prometheus counters and histograms. With
`serviceMonitor.enabled: true` and Prometheus Operator running in the
cluster, the chart wires up a ServiceMonitor that scrapes the pod
every 30 s by default.

## Verification

Dry-render the chart to see what would be installed:

```bash
helm template intelligence ./helm/intelligence -f values.prod.yaml | less
```

Lint:

```bash
helm lint ./helm/intelligence
```

Install onto a test cluster (kind / minikube / k3d):

```bash
helm install --dry-run --debug intelligence ./helm/intelligence
```
