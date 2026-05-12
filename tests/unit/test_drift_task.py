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

    result = task.train(TrainRequest(
        data_source=StaticDataSource(kind="static", name="ignored"),
        model_parameters={},
    ))
    assert result.metrics["reference_size"] == 300
    bento = task._load_bento()
    assert bento is not None
    assert "drift_calculator" in bento.custom_objects
    assert bento.custom_objects["forecaster_task_name"] == "t_forecaster"


def test_drift_predict_on_similar_chunk_reports_no_drift(tmp_path, monkeypatch):
    monkeypatch.setenv("BENTOML_HOME", str(tmp_path / "bentoml"))
    reference = _stationary_cpu(300, mean=0.5, std=0.05, seed=1)
    task = _drift_task(reference)
    task.train(TrainRequest(
        data_source=StaticDataSource(kind="static", name="ignored"),
        model_parameters={},
    ))

    # Same distribution as reference — should not alert.
    similar_chunk = _stationary_cpu(12, mean=0.5, std=0.05, seed=2)
    resp = task.predict(PredictRequest(input_series={"cpu": similar_chunk["cpu"].tolist()}))
    assert resp.prediction["drift_detected"] is False


def test_drift_predict_on_shifted_chunk_reports_drift(tmp_path, monkeypatch):
    monkeypatch.setenv("BENTOML_HOME", str(tmp_path / "bentoml"))
    reference = _stationary_cpu(300, mean=0.5, std=0.05, seed=1)
    task = _drift_task(reference)
    task.train(TrainRequest(
        data_source=StaticDataSource(kind="static", name="ignored"),
        model_parameters={},
    ))

    # Strongly shifted distribution (mean 0.5 -> 0.9, much tighter std).
    shifted_chunk = _stationary_cpu(12, mean=0.9, std=0.02, seed=3)
    resp = task.predict(PredictRequest(input_series={"cpu": shifted_chunk["cpu"].tolist()}))
    assert resp.prediction["drift_detected"] is True


def test_drift_task_is_registered_via_factory(tmp_path, monkeypatch):
    """The drift factory should compose with ``build_loader_for_task`` and
    register under a name that ties it to its forecaster.
    """
    from intelligence.config.settings import IntelligenceConfig
    from intelligence.tasks import build_registry_from_config

    monkeypatch.setenv("BENTOML_HOME", str(tmp_path / "bentoml"))
    cfg = IntelligenceConfig(enabled_tasks=["cpu_forecast_arima_drift"])
    reg = build_registry_from_config(cfg)
    task = reg.get("cpu_forecast_arima_drift")
    assert getattr(task, "forecaster_task_name", None) == "cpu_forecast_arima"
