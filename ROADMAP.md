# Roadmap

Forward plan for `icos-intelligence-ocei` after the pickle migration
(shipped 2026-05-13) and the upcoming end-to-end pilot-style test.

Ordering reflects two principles:

1. **Dependency before flourish** — multivariate before transformer-era
   models, GPU plumbing before training a transformer, MLflow before
   the explainability artefacts that live in it.
2. **Pilot impact early** — items that improve the deploy / first-forecast
   experience come before model breadth.

Energy forecasting is treated as a first-class use case throughout, not a
separate track.

## Phase 1 — QoL improvements - DONE

Cheap, doc-side changes.

- **End-to-end "from zero to forecast" walkthrough.** *Shipped 2026-05-13
  as [`docs/getting-started.md`](docs/getting-started.md).* Linear path
  from `helm install` to first `/predict`, anchored on the `cpu_forecast`
  example.
- **More energy examples beyond Kepler.** *Shipped 2026-05-13.* IPMI
  exporter, RAPL via node-exporter, PDU SNMP exporter, smart-plug
  exporters, DCIM-style metrics (Redfish). Each is one PromQL row in
  the [`examples/energy_forecast/`](examples/energy_forecast/) recipe
  table — no new bundled data, no library changes.

## Phase 2 — Backend compatibility - DONE

No code change required. `PrometheusSource` speaks any
Prom-API-compatible backend (Thanos Query, Mimir, Cortex) by virtue
of the shared `/api/v1/query_range` and `/api/v1/query` paths.
Future tutorials will demonstrate the long-history case by pointing
at a Thanos endpoint.

## Phase 3 — Multivariate

Foundation that unlocks Phase 5.

- Extend `feature` to accept a list of feature names.
- Extend `InputSpec` and the contract verification (HF pull validator,
  predict-time `n_features` check, version-pinning verifier).
- Extend XGB and LSTM trainers; refresh sliding-window logic.
- Allow N PromQL queries per task in the YAML schema.
- Update the energy examples to demonstrate covariates
  (energy ~ temperature + workload + hour-of-day).

Blast-radius warning: this touches input contracts, version pinning,
HF push/pull validation and every example. Own branch, contract tests
as the gate.

## Phase 4 — GPU mode

Infrastructure prerequisite for Phase 5.

- **GPU as an extension.** PyTorch GPU device selection in the trainer,
  optional `torch[cuda]` redirect, Dockerfile variant (or build arg) for
  a CUDA base, Helm `resources.limits.nvidia.com/gpu` plumbing. Larger
  image size but unlocks faster training and inference of complex models
  (transformer training, Chronos inference). CPU path stays the default.

## Phase 5 — Modern models

Lands as one coherent chapter behind multivariate and GPU.

- **`kind: chronos`** (Hugging Face Chronos 2 / Chronos-Bolt). Fits the
  existing builder pattern; model pulled on demand; no `/train` call
  required for inference. Changes the pitch from "train a model on
  your data" to "point it at your metric and forecast." Especially
  strong on bursty / regime-shifting signals — energy is one of the
  clearest beneficiaries. Behind a `[chronos]` extra.
- **`kind: tft`** (Temporal Fusion Transformer via
  `pytorch-forecasting`). Multivariate-native, attention weights
  provide explainability for free, native confidence intervals.
  Best ROI of the bespoke transformer architectures. Behind a
  `[transformers]` extra.
- **`kind: patchtst`** as a lighter univariate alternative if TFT
  proves heavy for pilot hardware.

Optional follow-up: `kind: timesfm` once Chronos has settled.

## Phase 6 — Tracking, explainability and backtesting

- **MLflow integration.** Log params, metrics and tags per `/train`
  call. Defensive import — MLflow stays disabled when `tracking_uri`
  is unset. Restored as a `[mlflow]` optional extra. BentoML remains
  the artefact store; MLflow holds the metadata. Prerequisite for the
  rest of this phase.
- **Explainability artefacts at train time.** SHAP for XGB, TFT
  attention / variable importance, ARIMA decomposition. Stored as
  MLflow artefacts plus a `/tasks/{task}/explain` endpoint.
- **Walk-forward backtesting** as `/tasks/{task}/backtest`. Returns
  per-horizon error metrics. Pilots will ask "how good is this
  forecast?" — this is the answer.

## Phase 7 — Adjacent improvements

Only when a pilot actually asks for one of these:

- **Drift-triggered auto-retrain loop.** Drift fires → train fires
  automatically. Currently external (operator schedules via CronJob);
  closing the loop couples the service more tightly.
- **Other source types** (Kafka, cloud metric stores). Each is a new
  `TelemetrySource` plus a `TelemetryConfig.source` literal extension
  plus a branch in `build_loader_for_task`. At three sources, consider
  extracting a source registry mirroring the kind dispatch.
- **`kind: prophet`** as a low-priority addition; the Chronos
  zero-shot pitch covers most of the same ground.
- **Default Grafana dashboard JSON.** Operator-facing polish. ConfigMap
  shipped via the Helm chart with the kube-prometheus-stack sidecar
  label; panels source from the existing `/metrics` endpoint
  (predict / train latency, per-task error rate, registered-tasks
  gauge). Lands when a pilot operator actually wants one.

## Explicitly out of scope

- A bespoke UI. Helm + YAML is good enough for this level unless asked.
- An in-process retraining scheduler. Deferred for future.
- Stronger drift detection. Deferred as well.
- Multi-replica work beyond the existing chart caveat unless a pilot
  drives the need.
