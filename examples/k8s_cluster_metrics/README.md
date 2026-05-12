# Kubernetes cluster metrics

PromQL recipes for the metrics a typical k8s cluster exposes via
[kube-prometheus-stack](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack)
(node-exporter, kube-state-metrics, cAdvisor). Same task kinds as
`cpu_forecast/` — the only thing that changes is the PromQL queries
and the feature names. M+N, illustrated.

The shipped `config.yaml` registers four tasks:

| Task | Slice | Metric |
|---|---|---|
| `cluster_cpu_arima` | Cluster-wide | Fraction of node CPU in use |
| `cluster_mem_arima` | Cluster-wide | Fraction of node memory in use |
| `pod_cpu_arima` | Namespace-scoped | Pod CPU rate in a chosen namespace |
| `pod_mem_arima` | Namespace-scoped | Pod memory fraction in a chosen namespace |

Each is `kind: arima` so a fresh reader can compare to `cpu_forecast/`
side by side. Swap in `xgb` or `lstm` per task to use windowed models.

## How to adapt for your cluster

The placeholders to edit:

- **Namespace filter** — `pod_cpu_arima` and `pod_mem_arima` use
  `{namespace="my-app"}`. Replace with the namespace you care about.
  Drop the filter entirely for cluster-wide aggregates.
- **Pod selector** — for a single deployment, add e.g.
  `{pod=~"my-app-.*"}` to scope down further.
- **Total reference** — pod CPU is shown as a *rate* (cores), not a
  fraction. Memory uses `machine_memory_bytes` as the cluster total;
  for per-node fractions divide by `node_memory_MemTotal_bytes` instead.

## Common variations

```yaml
# Node CPU — cluster-wide fraction in use
'1 - avg(rate(node_cpu_seconds_total{mode="idle"}[5m]))'

# Node memory — cluster-wide fraction in use
'1 - (sum(node_memory_MemAvailable_bytes) / sum(node_memory_MemTotal_bytes))'

# Pod CPU — sum of all pod CPU rates in a namespace (in cores)
'sum(rate(container_cpu_usage_seconds_total{namespace="my-app"}[5m]))'

# Pod memory — fraction of cluster memory used by a namespace's pods
'sum(container_memory_working_set_bytes{namespace="my-app"}) / sum(machine_memory_bytes)'

# Per-pod CPU — keyed by pod label, for one deployment
'sum by (pod) (rate(container_cpu_usage_seconds_total{pod=~"my-app-.*"}[5m]))'
```

## What you'll need on the cluster side

- **node-exporter** for `node_*` metrics — ships with kube-prometheus-stack.
- **cAdvisor** for `container_*` metrics — kubelet exposes this natively.
- **kube-state-metrics** for pod-/namespace-level inventory metrics (not
  strictly needed for the examples above, but useful for richer queries
  like "pods in CrashLoopBackOff").

If your cluster runs a different Prometheus stack (e.g. Thanos with
relabeled metric names), substitute accordingly — the task kinds don't
care what the metric is called, only that the PromQL returns one
time series per task.

## Caveat — value ranges

Node CPU and memory are fractional, so `value_range: [0.0, 1.0]` fits.
Pod CPU is a *rate in cores* and can exceed 1.0 on busy nodes — the
config widens it to `[0.0, 32.0]` to cover typical cluster sizes. For
your environment, pick a ceiling slightly above your peak observed
value; predict requests outside the range get rejected with `422`.
