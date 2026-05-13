"""Train + predict roundtrip for ``DriftDetectionTask``.

Drift task is a ``BaseTask`` subclass that fits a NannyML
``UnivariateDriftCalculator`` on the input feature distribution at train
time, and at predict time runs the calculator on a fresh chunk to
flag drift.

The ``forecaster_task_name`` field carries the identity of the
forecaster this drift task is paired with — used for URL naming and as
a hook for future prediction-drift extensions; not loaded at this stage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from intelligence.api.schemas import (
    PredictRequest,
    StaticDataSource,
    TrainRequest,
)


def _stationary_cpu(n: int, mean: float = 0.5, std: float = 0.05, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"cpu": rng.normal(mean, std, n).clip(0.0, 1.0)})


def _drift_task(reference_df: pd.DataFrame, **kwargs):
    from intelligence.tasks.contracts import InputSpec
    from intelligence.tasks.drift import DriftDetectionTask

    def loader(_descriptor):
        return {"reference_df": reference_df, "drift_columns": ["cpu"]}

    return DriftDetectionTask(
        name="t_drift",
        forecaster_task_name="t_forecaster",
        model=None,
        data_loader=loader,
        input_spec=InputSpec(
            n_features=1,
            feature_names=["cpu"],
            steps_back=12,
            value_range={"cpu": (0.0, 1.0)},
            units={"cpu": "fraction"},
        ),
        chunk_size=12,
        **kwargs,
    )


def test_drift_train_saves_bento_with_calculator(tmp_path, monkeypatch):
    monkeypatch.setenv("BENTOML_HOME", str(tmp_path / "bentoml"))
    reference = _stationary_cpu(300)
    task = _drift_task(reference)

    result = task.train(
        TrainRequest(
            data_source=StaticDataSource(kind="static", name="ignored"),
            model_parameters={},
        )
    )
    assert result.metrics["reference_size"] == 300
    loaded, served_tag = task._load_drift_artifact()
    assert loaded is not None
    assert "drift_calculator" in loaded
    assert loaded["forecaster_task_name"] == "t_forecaster"
    assert served_tag is not None


def test_drift_predict_on_similar_chunk_reports_no_drift(tmp_path, monkeypatch):
    monkeypatch.setenv("BENTOML_HOME", str(tmp_path / "bentoml"))
    reference = _stationary_cpu(300, mean=0.5, std=0.05, seed=1)
    task = _drift_task(reference)
    task.train(
        TrainRequest(
            data_source=StaticDataSource(kind="static", name="ignored"),
            model_parameters={},
        )
    )

    # Same distribution as reference — should not alert.
    similar_chunk = _stationary_cpu(12, mean=0.5, std=0.05, seed=2)
    resp = task.predict(PredictRequest(input_series={"cpu": similar_chunk["cpu"].tolist()}))
    assert resp.prediction["drift_detected"] is False


def test_drift_predict_on_shifted_chunk_reports_drift(tmp_path, monkeypatch):
    monkeypatch.setenv("BENTOML_HOME", str(tmp_path / "bentoml"))
    reference = _stationary_cpu(300, mean=0.5, std=0.05, seed=1)
    task = _drift_task(reference)
    task.train(
        TrainRequest(
            data_source=StaticDataSource(kind="static", name="ignored"),
            model_parameters={},
        )
    )

    # Strongly shifted distribution (mean 0.5 -> 0.9, much tighter std).
    shifted_chunk = _stationary_cpu(12, mean=0.9, std=0.02, seed=3)
    resp = task.predict(PredictRequest(input_series={"cpu": shifted_chunk["cpu"].tolist()}))
    assert resp.prediction["drift_detected"] is True


def test_drift_multivariate_alerts_on_any_feature_drift(tmp_path, monkeypatch):
    """Multivariate drift monitors each feature independently; any one
    shifting raises the alert. NannyML's ``UnivariateDriftCalculator``
    already operates per-column, so the lib just needs to surface a
    list of column names through the prepare path."""
    from intelligence.tasks.contracts import InputSpec
    from intelligence.tasks.drift import DriftDetectionTask

    monkeypatch.setenv("BENTOML_HOME", str(tmp_path / "bentoml"))

    rng = np.random.default_rng(1)
    reference_mv = pd.DataFrame(
        {
            "cpu": rng.normal(0.5, 0.05, 300).clip(0.0, 1.0),
            "mem": rng.normal(0.6, 0.04, 300).clip(0.0, 1.0),
        }
    )

    def loader(_descriptor):
        return {"reference_df": reference_mv, "drift_columns": ["cpu", "mem"]}

    task = DriftDetectionTask(
        name="cpu_mem_drift",
        forecaster_task_name="cpu_mem_forecaster",
        model=None,
        data_loader=loader,
        input_spec=InputSpec(
            n_features=2,
            feature_names=["cpu", "mem"],
            steps_back=12,
            value_range={"cpu": (0.0, 1.0), "mem": (0.0, 1.0)},
        ),
        chunk_size=12,
    )
    task.train(
        TrainRequest(
            data_source=StaticDataSource(kind="static", name="ignored"),
            model_parameters={},
        )
    )

    rng2 = np.random.default_rng(7)
    # cpu stationary; mem shifted up. Drift should fire on mem alone.
    chunk = {
        "cpu": rng2.normal(0.5, 0.05, 12).clip(0.0, 1.0).tolist(),
        "mem": rng2.normal(0.95, 0.01, 12).clip(0.0, 1.0).tolist(),
    }
    resp = task.predict(PredictRequest(input_series=chunk))
    assert resp.prediction["drift_detected"] is True


def test_drift_task_is_registered_via_builder(tmp_path, monkeypatch):
    """A drift task block under ``cfg.tasks`` registers via the drift
    builder and carries its forecaster reference.
    """
    from intelligence.config.settings import (
        ArimaTaskConfig,
        DriftTaskConfig,
        FeatureSpec,
        IntelligenceConfig,
    )
    from intelligence.tasks import build_registry_from_config

    monkeypatch.setenv("BENTOML_HOME", str(tmp_path / "bentoml"))
    cfg = IntelligenceConfig(
        tasks={
            "cpu_forecast_arima": ArimaTaskConfig(kind="arima", features=[FeatureSpec(name="cpu")]),
            "cpu_forecast_arima_drift": DriftTaskConfig(
                kind="drift",
                features=[FeatureSpec(name="cpu")],
                forecaster="cpu_forecast_arima",
            ),
        },
    )
    reg = build_registry_from_config(cfg)
    task = reg.get("cpu_forecast_arima_drift")
    assert getattr(task, "forecaster_task_name", None) == "cpu_forecast_arima"


# ---- New protocol: fit / save_artifacts / load_artifacts ------------------
#
# Drift has no native serialisable artefact (NannyML's calculator
# carries fitted-distribution state with no documented non-pickle save).
# Strategy: persist the reference DataFrame as parquet plus a small
# JSON config, and refit the calculator at load time. The fit is
# cheap — a one-pass histogram over the reference — and only happens
# once per artefact load.


@pytest.fixture
def drift_artifacts_fit():
    """A fresh ``(task, artifacts, metrics, reference)`` quad from
    ``DriftDetectionTask.fit``."""
    from intelligence.tasks.drift import DriftDetectionTask

    reference = _stationary_cpu(300, mean=0.5, std=0.05, seed=1)
    task = DriftDetectionTask(
        name="t_drift_new_protocol",
        forecaster_task_name="t_forecaster",
        model=None,
        data_loader=lambda _: {},
        chunk_size=12,
        metric="jensen_shannon",
    )

    artifacts, metrics = task.fit({"reference_df": reference, "drift_columns": ["cpu"]})
    return task, artifacts, metrics, reference


def test_drift_fit_returns_artifacts_and_metrics(drift_artifacts_fit):
    _task, artifacts, metrics, _ref = drift_artifacts_fit
    assert isinstance(artifacts, dict)
    assert isinstance(metrics, dict)
    assert metrics["reference_size"] == 300

    # The artefacts hold what's needed to refit + use the calculator.
    assert "reference_df" in artifacts
    assert "column_names" in artifacts
    assert "chunk_size" in artifacts
    assert "metric" in artifacts
    assert "forecaster_task_name" in artifacts


def test_drift_save_artifacts_writes_parquet_no_pickle(drift_artifacts_fit, tmp_path):
    task, artifacts, _, _ = drift_artifacts_fit
    files = task.save_artifacts(artifacts, tmp_path)

    # Reference data lives in parquet (compact, no pickle).
    assert files["reference"] == "reference.parquet"
    assert (tmp_path / "reference.parquet").exists()
    # Configuration JSON declares column_names + chunk_size + metric.
    assert (tmp_path / "drift.json").exists()
    assert (tmp_path / "metrics.json").exists()

    # No pickle whatsoever.
    assert not list(tmp_path.glob("*.pkl"))
    assert not list(tmp_path.glob("*.pickle"))


def test_drift_save_artifacts_includes_input_spec_when_present(drift_artifacts_fit, tmp_path):
    from intelligence.tasks.contracts import InputSpec

    task, artifacts, _, _ = drift_artifacts_fit
    artifacts["input_spec"] = InputSpec(n_features=1, feature_names=["cpu"], steps_back=12)
    files = task.save_artifacts(artifacts, tmp_path)
    assert files.get("input_spec") == "input_spec.json"
    assert (tmp_path / "input_spec.json").exists()


def test_drift_load_artifacts_refits_calculator(drift_artifacts_fit, tmp_path):
    """Load reconstructs the NannyML calculator from the persisted
    reference data. The refit is what replaces a pickled calculator —
    NannyML has no documented non-pickle save."""
    task, artifacts, _, _ = drift_artifacts_fit
    task.save_artifacts(artifacts, tmp_path)
    loaded = task.load_artifacts(tmp_path)

    assert "drift_calculator" in loaded
    assert loaded["column_names"] == artifacts["column_names"]
    assert loaded["chunk_size"] == artifacts["chunk_size"]
    assert loaded["metric"] == artifacts["metric"]
    assert loaded["forecaster_task_name"] == artifacts["forecaster_task_name"]

    # The refitted calculator should run on a fresh chunk and return
    # a result without raising — equivalent contract to the freshly-
    # fitted calculator in the legacy path.
    similar_chunk = _stationary_cpu(12, mean=0.5, std=0.05, seed=2)
    result = loaded["drift_calculator"].calculate(similar_chunk[["cpu"]])
    assert result is not None


def test_drift_load_artifacts_detects_shifted_chunk(drift_artifacts_fit, tmp_path):
    """End-to-end parity: the refit-on-load calculator must flag a
    shifted distribution the same way a freshly-fitted one would."""
    task, artifacts, _, _ = drift_artifacts_fit
    task.save_artifacts(artifacts, tmp_path)
    loaded = task.load_artifacts(tmp_path)

    shifted_chunk = _stationary_cpu(12, mean=0.9, std=0.02, seed=3)
    results = loaded["drift_calculator"].calculate(shifted_chunk[["cpu"]])
    chunks = results.filter(period="analysis").to_df()

    alert_col = ("cpu", loaded["metric"], "alert")
    assert alert_col in chunks.columns
    assert bool(chunks[alert_col].any())


def test_drift_load_artifacts_restores_input_spec(drift_artifacts_fit, tmp_path):
    from intelligence.tasks.contracts import InputSpec

    task, artifacts, _, _ = drift_artifacts_fit
    artifacts["input_spec"] = InputSpec(n_features=1, feature_names=["cpu"], steps_back=12)
    task.save_artifacts(artifacts, tmp_path)

    loaded = task.load_artifacts(tmp_path)
    assert isinstance(loaded["input_spec"], InputSpec)
    assert loaded["input_spec"].feature_names == ["cpu"]


def test_drift_files_map_declares_only_safe_extensions(drift_artifacts_fit, tmp_path):
    from pathlib import Path

    from intelligence.ml.artifact.manifest import ALLOWED_EXTENSIONS

    task, artifacts, _, _ = drift_artifacts_fit
    files = task.save_artifacts(artifacts, tmp_path)
    for role, fname in files.items():
        assert Path(fname).suffix.lower() in ALLOWED_EXTENSIONS, (
            f"role {role!r} declares {fname!r} with disallowed extension"
        )
