"""Train + predict roundtrip for a drift task.

A drift task is a plain ``BaseTask`` wired with a ``DriftModel`` —
NannyML's ``UnivariateDriftCalculator`` fits the reference
distribution at train time, and at predict time runs the calculator
on a fresh chunk to flag drift.

The model-level protocol (fit / save / load / predict in isolation)
is covered by ``test_drift_model.py``; this file exercises the task
plumbing — bento save+load, predict response shape, multivariate
feature handling, builder wiring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from intelligence.api.schemas import (
    PredictRequest,
    StaticDataSource,
    TrainRequest,
)
from tests._synthetic import stationary_cpu


def _drift_task(reference_df: pd.DataFrame, **drift_model_kwargs):
    """Build a BaseTask carrying a DriftModel, with a fake loader that
    returns the supplied reference frame on demand."""
    from intelligence.ml.models.drift import DriftModel
    from intelligence.tasks.base import BaseTask
    from intelligence.tasks.contracts import InputSpec

    def loader(_descriptor):
        return {"reference_df": reference_df, "drift_columns": ["cpu"]}

    return BaseTask(
        name="t_drift",
        model=DriftModel(
            chunk_size=12,
            metric=drift_model_kwargs.get("metric", "jensen_shannon"),
            forecaster_task_name=drift_model_kwargs.get("forecaster_task_name", "t_forecaster"),
        ),
        data_loader=loader,
        input_spec=InputSpec(
            n_features=1,
            feature_names=["cpu"],
            steps_back=12,
            value_range={"cpu": (0.0, 1.0)},
            units={"cpu": "fraction"},
        ),
    )


def test_drift_train_saves_bento_with_calculator(tmp_path, monkeypatch):
    monkeypatch.setenv("BENTOML_HOME", str(tmp_path / "bentoml"))
    reference = stationary_cpu(300)
    task = _drift_task(reference)

    result = task.train(
        TrainRequest(
            data_source=StaticDataSource(kind="static", name="ignored"),
            model_parameters={},
        )
    )
    assert result.metrics["reference_size"] == 300
    loaded, served_tag = task._load_artifact()
    assert loaded is not None
    assert "drift_calculator" in loaded
    assert loaded["forecaster_task_name"] == "t_forecaster"
    assert served_tag is not None


def test_drift_predict_on_similar_chunk_reports_no_drift(tmp_path, monkeypatch):
    monkeypatch.setenv("BENTOML_HOME", str(tmp_path / "bentoml"))
    reference = stationary_cpu(300, mean=0.5, std=0.05, seed=1)
    task = _drift_task(reference)
    task.train(
        TrainRequest(
            data_source=StaticDataSource(kind="static", name="ignored"),
            model_parameters={},
        )
    )

    # Same distribution as reference — should not alert.
    similar_chunk = stationary_cpu(12, mean=0.5, std=0.05, seed=2)
    resp = task.predict(PredictRequest(input_series={"cpu": similar_chunk["cpu"].tolist()}))
    assert resp.prediction.drift_detected is False


def test_drift_predict_on_shifted_chunk_reports_drift(tmp_path, monkeypatch):
    monkeypatch.setenv("BENTOML_HOME", str(tmp_path / "bentoml"))
    reference = stationary_cpu(300, mean=0.5, std=0.05, seed=1)
    task = _drift_task(reference)
    task.train(
        TrainRequest(
            data_source=StaticDataSource(kind="static", name="ignored"),
            model_parameters={},
        )
    )

    # Strongly shifted distribution (mean 0.5 -> 0.9, much tighter std).
    shifted_chunk = stationary_cpu(12, mean=0.9, std=0.02, seed=3)
    resp = task.predict(PredictRequest(input_series={"cpu": shifted_chunk["cpu"].tolist()}))
    assert resp.prediction.drift_detected is True


def test_drift_multivariate_alerts_on_any_feature_drift(tmp_path, monkeypatch):
    """Multivariate drift monitors each feature independently; any one
    shifting raises the alert. NannyML's ``UnivariateDriftCalculator``
    already operates per-column, so the lib just needs to surface a
    list of column names through the prepare path."""
    from intelligence.ml.models.drift import DriftModel
    from intelligence.tasks.base import BaseTask
    from intelligence.tasks.contracts import InputSpec

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

    task = BaseTask(
        name="cpu_mem_drift",
        model=DriftModel(chunk_size=12, forecaster_task_name="cpu_mem_forecaster"),
        data_loader=loader,
        input_spec=InputSpec(
            n_features=2,
            feature_names=["cpu", "mem"],
            steps_back=12,
            value_range={"cpu": (0.0, 1.0), "mem": (0.0, 1.0)},
        ),
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
    assert resp.prediction.drift_detected is True


def test_drift_task_is_registered_via_builder(tmp_path, monkeypatch):
    """A drift task block under ``cfg.tasks`` registers via the drift
    builder. ``forecaster_task_name`` lives on the wrapped DriftModel
    (the task is a plain BaseTask).
    """
    from intelligence.config.settings import (
        ArimaTaskConfig,
        DriftTaskConfig,
        FeatureSpec,
        IntelligenceConfig,
    )
    from intelligence.ml.models.drift import DriftModel
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
    assert isinstance(task.model, DriftModel)
    assert task.model.forecaster_task_name == "cpu_forecast_arima"
