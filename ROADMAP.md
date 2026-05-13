# Roadmap

Forward plan for `icos-intelligence-ocei` after the pickle migration
(shipped 2026-05-13) and the upcoming end-to-end pilot-style test.

Ordering reflects two principles:

1. **Dependency before flourish** — multivariate before per-variable
   explainability, MLflow before backtesting artefacts, etc.
2. **Pilot impact early** — items that improve the deploy / first-forecast
   experience come before model breadth.

Energy forecasting is treated as a first-class use case throughout, not a
separate track.

## Phase 1 — QoL improvements

Cheap, doc-side changes.

- **End-to-end "from zero to forecast" walkthrough.** Single tutorial
  layered over `examples/`: deploy via Helm, point at a Prometheus,
  define one task, train, predict. Not new code — new doc.
- **More energy examples beyond Kepler.** IPMI/RAPL exporter,
  PDU SNMP exporter, smart-plug exporter, DCIM-style metrics. Each is
  one YAML file plus a short README paragraph.
- **MLflow integration.** Log params, metrics and artefacts per `/train`
  call. Unblocks Phase 5 (explainability artefacts need somewhere to
  live; backtesting results too). answers
  "why did this model do X?" without code spelunking.
- **Default Grafana dashboard JSON.** Shipped in the Helm chart.
  Predict latency, train durations, per-task error rate sourced from
  the existing `/metrics` endpoint. Trivial, high perceived polish.


## Phase 2 — Adjacent ingestion and minor models

Independent track.

- **OpenTelemetry source.** Lib-side change in `src/intelligence/telemetry/`,
  isolated, low risk. Extends `TelemetryConfig.source` and adds a
  `TelemetrySource` implementation alongside `static` and `prometheus`.

  - Thanos connection?


## Phase 3 — Zero-shot foundation model

Biggest ease-of-use and modernness lever in the whole roadmap.

- **`kind: chronos`** (Hugging Face Chronos 2 / Chronos-Bolt). Fits the
  existing builder pattern; model pulled on demand; no `/train` call
  required for inference. Changes the pitch from "train a model on
  your data" to "point it at your metric and forecast." Especially
  strong on bursty / regime-shifting signals — energy is one of the
  clearest beneficiaries.
- Optional follow-up: `kind: timesfm` once Chronos has settled.

## Phase 4 — Multivariate

Foundation that unlocks Phase 4 and Phase 5.

- Extend `feature` to accept a list of feature names.
- Extend `InputSpec` and the contract verification (HF pull validator,
  predict-time `n_features` check, version-pinning verifier).
- Extend XGB and LSTM trainers; refresh sliding-window logic.
- Allow N PromQL queries per task in the YAML schema.
- Update all four energy examples to demonstrate covariates
  (energy ~ temperature + workload + hour-of-day).

Blast-radius warning: this touches input contracts, version pinning,
HF push/pull validation and every example. Own branch, contract tests
as the gate.

## Phase 5 — Transformers

Worth doing only after multivariate.

- **TFT first.** Multivariate-native, attention weights provide
  explainability for free, has native confidence intervals. Best ROI
  of the modern architectures.
- **PatchTST** as a lighter univariate alternative if TFT proves heavy
  for pilot hardware.

## Phase 6 — Explainability and backtesting

MLflow and multivariate are prerequisites; both exist by this phase.

- **Explainability artefacts at train time:** SHAP for XGB, TFT
  attention / variable importance, ARIMA decomposition. Stored as
  MLflow artefacts plus a `/tasks/{task}/explain` endpoint.
- **Walk-forward backtesting** as `/tasks/{task}/backtest`. Returns
  per-horizon error metrics. Pilots will ask "how good is this
  forecast?" — this is the answer.

## Phase 7 - GPU support
- **GPU as an extension** simple wiring for pytorch on gpu mode, larger image size but allows for faster training and inference of more complex models (Chronos inference + transformers-based models train/predict)

## Explicitly out of scope

- A bespoke UI. Helm + YAML is good enough for this level unless asked.
- An in-process retraining scheduler. Deferred for future.
- Stronger drift detection. Deferred as well.
- Multi-replica work beyond the existing chart caveat unless a pilot
  drives the need.
