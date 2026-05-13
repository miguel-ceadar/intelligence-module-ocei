# Energy forecasting

Same task kinds as `cpu_forecast/`, swapped onto an energy metric, useful as an example, details may differ in your exact setup. The
shipped `config.yaml` defaults to a Kepler-based PromQL expression
(eBPF energy attribution), but the model kinds don't care which
exporter produces the watts, substitute the query with whatever one your stack already runs.

| Task | Kind |
|---|---|
| `energy_forecast_arima` | arima |
| `energy_forecast_xgb` | xgb |
| `energy_forecast_lstm` | lstm |
| `energy_forecast_arima_drift` | drift |

## Example energy PromQL recipes

Energy exporters split into two camps:

- **Level (watts)** — direct power reading. Use the metric as-is.
- **Counter (joules)** — cumulative energy. Wrap in `rate(...[Xs])` to
  get watts (joules/second).

Common exporters and the query each one needs:

| Source | Query | Unit out | Scope |
|---|---|---|---|
| Kepler (eBPF) | `sum(rate(kepler_node_package_joules_total[30s]))` | W | Node — CPU package |
| RAPL via node-exporter | `rate(node_rapl_package_joules_total[30s])` | W | Host — CPU package |
| IPMI exporter (DCMI) | `ipmi_dcmi_power_consumption_watts` | W | Chassis total |
| Redfish exporter | `redfish_chassis_power_powerconsumedwatts` | W | Chassis total |
| PDU via snmp_exporter | `apc_rPDU2_phase_status_power_watts` | W | PDU phase/outlet draw |
| Shelly smart-plug | `shelly_switch_power_watts` | W | Per-plug load |
| Tasmota smart-plug | `tasmota_sensor_energy_power_watts` | W | Per-plug load |

Exact metric names depend on your exporter version and configuration —
confirm with a Prometheus `label_values(__name__)` query against your
endpoint before pasting.

## Set `value_range` to your hardware.

Energy magnitudes span four orders of magnitude across the table above
(a smart-plug measures ~10–2000 W per device; a rack PDU aggregates
to several kW). The default `value_range: [0.0, 500.0]` in
`config.yaml` is a placeholder. If your samples fall outside it
at `/predict` time, the request will be rejected with `422` —
that's the check catching a unit / scale mismatch.
Widen the range to match the actual draw of whatever your query
covers, or remove it entirely if you'd rather not enforce one.

## Run it

```bash
docker run -d --name icos-intelligence-ocei \
  -p 3000:3000 \
  -e INTELLIGENCE_CONFIG=/etc/intelligence/config.yaml \
  -e INTELLIGENCE_TELEMETRY__PROMETHEUS__ENDPOINT=https://your-prom \
  -v "$PWD/examples/energy_forecast/config.yaml:/etc/intelligence/config.yaml:ro" \
  -v intelligence-bentoml:/var/lib/bentoml \
  ghcr.io/miguel-ceadar/icos-intelligence-ocei:0.2.2

curl -X POST http://localhost:3000/tasks/energy_forecast_arima/train \
  -H 'Content-Type: application/json' \
  -d '{"data_source": {"kind": "prometheus", "window": "24h", "step": "1m"}}'

curl -X POST http://localhost:3000/tasks/energy_forecast_arima/predict \
  -H 'Content-Type: application/json' \
  -d '{"input_series": {"energy": [180.4]}}'
```

The `feature: energy` declaration in `config.yaml` is a logical
identifier — it names the column that `/predict` accepts, not a
specific exporter. As long as the value you pass in `input_series.energy`
is in the same units as your training data (watts, in the recipes
above), the model is unit-agnostic.

## Diff against `cpu_forecast/`

Comparing `config.yaml`:

- `feature: cpu` → `feature: energy`
- Task names renamed to `energy_forecast_*`
- `query:` swapped for an energy PromQL expression
- `value_range` widened from `[0.0, 1.0]` (a CPU fraction) to
  `[0.0, 500.0]` (a node-class watts envelope)
- Drift task's `forecaster:` points at `energy_forecast_arima`
