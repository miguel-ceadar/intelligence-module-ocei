"""``ModelTrainer`` contract — return shapes the ``Model`` adapters expect."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

from intelligence.ml.trainers import ModelTrainer


@pytest.fixture
def univariate_components():
    """Minimal data_components dict the ARIMA/XGB trainers expect.

    Synthetic random-walk rather than a real CSV so the unit test stays
    sub-second — ARIMA refits once per test sample in the legacy loop.
    """
    rng = np.random.default_rng(42)
    series = np.cumsum(rng.standard_normal(80)).reshape(-1, 1)
    split = 60
    scaler = MinMaxScaler().fit(series[:split])
    return {
        "X_train": scaler.transform(series[:split]),
        "X_test": scaler.transform(series[split:]),
        "y_train": series[:split].ravel(),
        "y_test": series[split:].ravel(),
        "scaler_obj": scaler,
        "model_parameters": {"p": 2, "d": 1, "q": 0},  # tiny ARIMA for test speed
    }


def test_train_arima_returns_metrics_and_history(univariate_components):
    """``train_arima`` returns ``(metrics, history)`` — the walk-forward
    history is the only by-product the ARIMA artifact persists; the
    fitted model object is discarded because predict refits on each call.
    """
    trainer = ModelTrainer(univariate_components)
    metrics, history = trainer.train_arima()
    assert "mae" in metrics and "rmse" in metrics
    assert isinstance(history, list) and len(history) > 0


def test_train_xgb_returns_metrics_and_model(univariate_components):
    components = {
        **univariate_components,
        "model_parameters": {"n_estimators": 20, "max_depth": 3, "eta": 0.1},
    }
    trainer = ModelTrainer(components)
    metrics, model = trainer.train_xgb()
    assert "mae" in metrics
    assert hasattr(model, "predict")
