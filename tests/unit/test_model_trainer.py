"""``ModelTrainer`` contract.

Regression guards:
- It's a plain class — no ``DataClayObject`` base in the MRO.
- Methods are not ``@activemethod``-decorated.
- ``train_xgb`` / ``train_arima`` / ``train_pytorch`` return the
  shapes the ``Model`` adapters in ``intelligence.ml.models`` expect.
"""

from __future__ import annotations

from pathlib import Path  # noqa: F401 — kept for the slow LSTM test below

import pytest

trainers = pytest.importorskip("intelligence.ml.trainers", reason="phase-1 §2.2 pending")


def _maybe(name: str):
    obj = getattr(trainers, name, None)
    if obj is None:
        pytest.skip(f"intelligence.ml.trainers.{name} not implemented yet")
    return obj


def test_model_trainer_is_a_plain_class():
    ModelTrainer = _maybe("ModelTrainer")
    bases = {b.__name__ for b in ModelTrainer.__mro__}
    assert "DataClayObject" not in bases, (
        "ModelTrainer must not inherit from DataClayObject — that's the "
        "structural coupling we're removing."
    )


def test_model_trainer_methods_have_no_activemethod_marker():
    ModelTrainer = _maybe("ModelTrainer")
    for method_name in ("train_xgb", "train_arima", "train_pytorch"):
        method = getattr(ModelTrainer, method_name, None)
        if method is None:
            continue
        # `@activemethod` from dataclay sets a sentinel attribute on the wrapped
        # function. Whatever the exact attribute name, it should not survive.
        assert not getattr(method, "__activemethod__", False)
        assert not getattr(method, "_dc_active", False)


@pytest.fixture
def univariate_components():
    """Minimal data_components dict the ARIMA/XGB trainers expect.

    Synthetic random-walk rather than a real CSV so the unit test stays
    sub-second — ARIMA refits once per test sample in the legacy loop.
    """
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

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


def test_train_arima_returns_expected_tuple(univariate_components):
    ModelTrainer = _maybe("ModelTrainer")
    trainer = ModelTrainer(univariate_components)
    metrics, model, history, y_test, y_pred = trainer.train_arima()
    assert "mae" in metrics and "rmse" in metrics
    assert hasattr(model, "fit")
    assert isinstance(history, list) and len(history) > 0
    assert len(y_pred) == len(y_test)


def test_train_xgb_returns_metrics_and_model(univariate_components):
    ModelTrainer = _maybe("ModelTrainer")
    components = {**univariate_components,
                  "model_parameters": {"n_estimators": 20, "max_depth": 3, "eta": 0.1}}
    trainer = ModelTrainer(components)
    metrics, model = trainer.train_xgb()
    assert "mae" in metrics
    assert hasattr(model, "predict")


@pytest.mark.slow
def test_train_pytorch_returns_metrics_and_model(sample_csv_multivariate: Path):
    """PyTorch path needs (X_train, X_test, y_train, y_test) as tensors plus
    a DataLoader-friendly dataset. Marked slow because it actually runs an
    LSTM for a couple of epochs."""
    pytest.importorskip("torch")
    # Placeholder until the multivariate tensor fixture is built out;
    # the LSTM happy-path is covered by ``test_lstm_model.py`` via
    # ``make_lstm_prepare`` on synthetic data.
    pytest.skip("pending: multivariate tensor fixture for ModelTrainer.train_pytorch")
