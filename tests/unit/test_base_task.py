"""Phase-1 §2.3 / §2.6: lazy model loading on ``BaseTask``.

Construction is side-effect free. The artefact is fetched on first
predict and cached for subsequent calls. Training invalidates the
``:latest`` cache slot so the next predict picks up the freshly-saved
artefact.

Unit-level — uses mocks for the model, loader, and the artefact-store
helpers in ``intelligence.tasks.base``. The end-to-end sanity check
(real train + predict round-trip) lives in
``tests/integration/test_api_endpoints.py``.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from intelligence.api.schemas import (
    PredictRequest,
    StaticDataSource,
    TrainRequest,
)
from intelligence.ml.artifact import SavedArtifact
from intelligence.ml.artifact.manifest import Manifest
from intelligence.tasks import BaseTask


def _make_model(predict_return=None, loaded_return: dict | None = None):
    from intelligence.api.schemas import ForecastPoint

    m = mock.MagicMock()
    m.name = "fake"
    m.has_drift = False
    # PredictResponse.prediction is ``list[ForecastPoint] | DriftPrediction`` —
    # mocks that returned a bare float used to slip through ``prediction: Any``.
    m.predict.return_value = (
        predict_return if predict_return is not None else [ForecastPoint(value=0.42)]
    )
    m.fit.return_value = ({"any": "thing"}, {"mae": 0.1})
    m.save_artifacts.return_value = {"thing": "thing.json"}
    m.load_artifacts.return_value = loaded_return if loaded_return is not None else {}
    return m


def _make_loader():
    return mock.MagicMock(
        return_value={
            "X_train": [[1.0]],
            "X_test": [[2.0]],
            "scaler_obj": mock.MagicMock(),
        }
    )


def _fake_saved(tag: str = "t_lazy:v1") -> SavedArtifact:
    name, version = tag.split(":", 1)
    return SavedArtifact(
        tag=tag,
        name=name,
        version=version,
        path=Path("/fake/path"),
        manifest=Manifest(
            schema_version=1,
            kind="fake",
            created_at="2026-05-13T10:00:00+00:00",
            files={},
        ),
        created_at="2026-05-13T10:00:00+00:00",
    )


@pytest.fixture
def task():
    return BaseTask(
        name="t_lazy",
        model=_make_model(),
        data_loader=_make_loader(),
    )


def test_construction_does_not_load_model(task):
    """``is_loaded()`` is False on a fresh task — no artefact fetched yet."""
    assert not task.is_loaded()


def test_first_predict_triggers_exactly_one_store_lookup(task):
    with mock.patch("intelligence.tasks.base.get_artifact_by_tag") as get:
        get.return_value = _fake_saved("t_lazy:v1")
        task.predict(PredictRequest(input_series={"x": [1.0]}))
        get.assert_called_once_with("t_lazy:latest")
    assert task.is_loaded()


def test_subsequent_predicts_use_cached_artifact(task):
    """Multiple predicts share one store lookup + one load_artifacts call."""
    with mock.patch("intelligence.tasks.base.get_artifact_by_tag") as get:
        get.return_value = _fake_saved("t_lazy:v1")
        for v in (1.0, 2.0, 3.0):
            task.predict(PredictRequest(input_series={"x": [v]}))
    assert get.call_count == 1
    assert task.model.load_artifacts.call_count == 1


def test_train_invalidates_cache(task):
    """After ``train``, the next ``predict`` must re-fetch from the store."""
    with (
        mock.patch("intelligence.tasks.base.get_artifact_by_tag") as get,
        mock.patch("intelligence.tasks.base.save_artifact") as save,
    ):
        get.return_value = _fake_saved("t_lazy:v1")
        save.return_value = _fake_saved("t_lazy:v2")

        task.predict(PredictRequest(input_series={"x": [1.0]}))
        assert get.call_count == 1
        assert task.is_loaded()

        task.train(
            TrainRequest(
                data_source=StaticDataSource(kind="static", name="x.csv"),
                model_parameters={},
            )
        )
        assert not task.is_loaded(), "train should invalidate the cached artefact"

        task.predict(PredictRequest(input_series={"x": [1.0]}))
        assert get.call_count == 2


def test_train_threads_prometheus_descriptor_through_loader():
    """``BaseTask.train`` must not hardcode a ``StaticDataSource`` check —
    dispatch is the loader's job. The loader either accepts the descriptor
    or raises ``ValueError`` (which the API translates to 422).
    """
    from intelligence.api.schemas import PrometheusDataSource

    fake_loader = mock.MagicMock(
        return_value={
            "X_train": [[1.0]],
            "X_test": [[2.0]],
            "scaler_obj": mock.MagicMock(),
        }
    )

    t = BaseTask(name="t", model=_make_model(), data_loader=fake_loader)
    with mock.patch("intelligence.tasks.base.save_artifact") as save:
        save.return_value = _fake_saved("fake:v1")
        result = t.train(
            TrainRequest(
                data_source=PrometheusDataSource(kind="prometheus", window="1h", step="1m"),
                model_parameters={},
            )
        )
    assert result.model_tag == "fake:v1"
    fake_loader.assert_called_once()


def test_predict_threads_horizon_into_model(task):
    """``PredictRequest.horizon`` flows through to ``Model.predict``."""
    from intelligence.api import schemas as api_schemas

    with mock.patch("intelligence.tasks.base.get_artifact_by_tag") as get:
        get.return_value = _fake_saved("t_lazy:v1")
        task.predict(api_schemas.PredictRequest(input_series={"x": [1.0]}, horizon=4))

    call = task.model.predict.call_args
    assert call.kwargs.get("horizon") == 4 or (len(call.args) >= 3 and call.args[2] == 4)


def test_predict_rejects_horizon_above_input_spec_max():
    """When ``input_spec.max_horizon`` is set, requests above it get 422
    (``ContractViolation``) at the API boundary — no store fetch needed.
    """
    from intelligence.api import schemas as api_schemas
    from intelligence.tasks.contracts import ContractViolation, InputSpec

    spec = InputSpec(n_features=1, feature_names=["x"], steps_back=1, max_horizon=2)
    t = BaseTask(
        name="t",
        model=_make_model(),
        data_loader=_make_loader(),
        input_spec=spec,
    )
    with pytest.raises(ContractViolation, match="horizon"):
        t.predict(api_schemas.PredictRequest(input_series={"x": [0.5]}, horizon=5))


def test_predict_allows_horizon_within_max(task):
    """``horizon`` at or below ``max_horizon`` passes through cleanly."""
    from intelligence.api import schemas as api_schemas
    from intelligence.tasks.contracts import InputSpec

    spec = InputSpec(n_features=1, feature_names=["x"], steps_back=1, max_horizon=3)
    # ``allow_unverified_models=True`` skips the input_spec check — the
    # fake loaded artefact here doesn't carry one.
    t = BaseTask(
        name="t",
        model=_make_model(),
        data_loader=_make_loader(),
        input_spec=spec,
        allow_unverified_models=True,
    )
    with mock.patch("intelligence.tasks.base.get_artifact_by_tag") as get:
        get.return_value = _fake_saved("t:v1")
        t.predict(api_schemas.PredictRequest(input_series={"x": [0.5]}, horizon=3))
    t.model.predict.assert_called_once()


def test_predict_raises_filenotfound_when_no_artefact_in_store(task):
    """Without a saved artefact, predict must raise FileNotFoundError —
    the API layer translates that to 503 (see service.predict)."""
    with (
        mock.patch("intelligence.tasks.base.get_artifact_by_tag") as get,
        pytest.raises(FileNotFoundError, match="no Bento"),
    ):
        get.return_value = None  # nothing in the store
        task.predict(PredictRequest(input_series={"x": [1.0]}))
    assert not task.is_loaded()
