"""Phase-1 §2.3 / §2.6: lazy model loading on ``BaseTask``.

Construction is side-effect free. The BentoML model is fetched on first
predict and cached for subsequent calls. Training invalidates the cache
so the next predict picks up the freshly-saved model.

Unit-level — uses mocks for the model, loader, and ``bentoml.get``;
no real BentoML store interaction. The end-to-end sanity check (real
ARIMA train + predict round-trip) lives in
``tests/integration/test_api_endpoints.py``.
"""

from __future__ import annotations

from unittest import mock

import pytest

from intelligence.api.schemas import (
    PredictRequest,
    StaticDataSource,
    TrainRequest,
)
from intelligence.tasks import BaseTask


def _make_model(predict_return: float = 0.42):
    m = mock.MagicMock()
    m.name = "fake"
    m.has_drift = False
    m.predict.return_value = predict_return
    fake_bento = mock.MagicMock()
    fake_bento.tag = "fake:v1"
    m.train.return_value = (fake_bento, {"mae": 0.1})
    return m


def _make_loader():
    return mock.MagicMock(return_value={
        "X_train": [[1.0]],
        "X_test": [[2.0]],
        "scaler_obj": mock.MagicMock(),
    })


@pytest.fixture
def task():
    return BaseTask(
        name="t_lazy",
        model=_make_model(),
        data_loader=_make_loader(),
    )


def test_construction_does_not_load_model(task):
    """``is_loaded()`` is False on a fresh task — no Bento fetched yet."""
    assert not task.is_loaded()


def test_first_predict_triggers_exactly_one_bentoml_get(task):
    with mock.patch("bentoml.picklable_model.get") as get:
        get.return_value = mock.MagicMock(custom_objects={})
        task.predict(PredictRequest(input_series={"x": [1.0]}))
        get.assert_called_once_with("t_lazy:latest")
    assert task.is_loaded()


def test_subsequent_predicts_use_cached_model(task):
    """Multiple predicts share one ``bentoml.get`` call."""
    with mock.patch("bentoml.picklable_model.get") as get:
        get.return_value = mock.MagicMock(custom_objects={})
        for v in (1.0, 2.0, 3.0):
            task.predict(PredictRequest(input_series={"x": [v]}))
    assert get.call_count == 1


def test_train_invalidates_cache(task):
    """After ``train``, the next ``predict`` must re-fetch from the store."""
    with mock.patch("bentoml.picklable_model.get") as get:
        get.return_value = mock.MagicMock(custom_objects={})

        task.predict(PredictRequest(input_series={"x": [1.0]}))
        assert get.call_count == 1
        assert task.is_loaded()

        task.train(TrainRequest(
            data_source=StaticDataSource(kind="static", name="x.csv"),
            model_parameters={},
        ))
        assert not task.is_loaded(), "train should invalidate the cached model"

        task.predict(PredictRequest(input_series={"x": [1.0]}))
        assert get.call_count == 2


def test_train_no_longer_raises_for_prometheus_descriptor():
    """``BaseTask.train`` must not hardcode a ``StaticDataSource`` check —
    dispatch is the loader's job. The loader either accepts the descriptor
    or raises ``ValueError`` (which the API translates to 422).
    """
    from intelligence.api.schemas import PrometheusDataSource

    # A loader that accepts any descriptor and returns trivial components.
    fake_loader = mock.MagicMock(return_value={
        "X_train": [[1.0]],
        "X_test": [[2.0]],
        "scaler_obj": mock.MagicMock(),
    })

    t = BaseTask(name="t", model=_make_model(), data_loader=fake_loader)
    result = t.train(TrainRequest(
        data_source=PrometheusDataSource(kind="prometheus", window="1h", step="1m"),
        model_parameters={},
    ))
    assert result.model_tag == "fake:v1"
    fake_loader.assert_called_once()


def test_predict_raises_filenotfound_when_no_model_in_store(task):
    """Without a saved Bento, predict must raise FileNotFoundError —
    the API layer translates that to 503 (see service.predict)."""
    import bentoml
    with mock.patch(
        "bentoml.picklable_model.get",
        side_effect=bentoml.exceptions.NotFound("no such model"),
    ):
        with pytest.raises(FileNotFoundError, match="no trained model"):
            task.predict(PredictRequest(input_series={"x": [1.0]}))
    assert not task.is_loaded()
