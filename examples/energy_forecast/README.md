# Energy forecasting (template — needs real data)

The same `(data × model)` pattern applied to power consumption. **This
example ships only the configuration shape; we don't yet have real
energy data captured for end-to-end testing.** Use it as a starting
point if you have an energy-exporter in your cluster.

| Task | Kind |
|---|---|
| `energy_forecast_arima` | arima |
| `energy_forecast_xgb` | xgb |
| `energy_forecast_lstm` | lstm |
| `energy_forecast_arima_drift` | drift |

## Where energy metrics come from

- **Kepler** (k8s-native, eBPF-based attribution) — `kepler_container_joules_total`
  and `kepler_node_package_joules_total` are the typical metrics. Both
  are counters; use `rate(...)` to get watts.
- **node-exporter RAPL collector** — `node_rapl_*_joules_total` on
  bare-metal Linux with Intel RAPL (`rapl` kernel module loaded).
- **Cloud-provider energy estimates** — typically counter-based; same
  `rate(...)` pattern.

Example PromQL (Kepler):

```promql
sum(rate(kepler_node_package_joules_total[30s]))
```

Units are watts (joules / second). If your cluster's typical idle
draw is ~50 W and peak is ~250 W, set `value_range: [0.0, 500.0]` as
a generous bound — InputSpec validation enforces it at predict time.

## What changed vs. `cpu_forecast/`

- `feature: energy`
- `value_range: [0.0, 500.0]` (watts, not a fraction)
- Task names renamed
- `query:` swapped for energy PromQL

## What's still needed

- A representative CSV sample for static-mode demo. Should carry a
  realistic power trace from a Kepler-instrumented node or a RAPL-
  capable bare-metal host. Once available, drop it into
  `src/intelligence/data/samples/` and reference it from this README.
- End-to-end smoke against a Prometheus that's actually scraping
  Kepler.

Until those land, treat this directory as a reference for shape, not
a runnable example.
