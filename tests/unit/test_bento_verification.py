"""Pretrained-Bento policy: refuse predict from Bentos whose saved
``input_spec`` is missing or doesn't match the task's spec.

The contract travels with the model — it's written into the Bento's
``custom_objects`` at train time (``BaseTask.train`` injects it via
``extras``). At predict time we check the loaded Bento's spec against
the task's current spec. Mismatches surface as the same 503 response
shape as "no trained model", since operationally the situation is the
same: the caller needs to train (or pull a matching Bento).

Override: ``BaseTask.allow_unverified_models=True`` lets predict
proceed with a warning. Intended for debugging or for someone who has
intentionally accepted the risk of a pulled HF Bento that predates the
contract.
"""

from __future__ import annotations

from unittest import mock

import pytest

from intelligence.api.schemas import PredictRequest
from intelligence.tasks import BaseTask
from intelligence.tasks.contracts import InputSpec


def _spec(n_features: int = 1, steps_back: int = 1, feature_names=("cpu",)) -> InputSpec:
    return InputSpec(
        n_features=n_features,
        feature_names=list(feature_names),
        steps_back=steps_back,
    )


def _model():
    m = mock.MagicMock()
    m.name = "fake"
    m.has_drift = False
    m.predict.return_value = 0.42
    return m


def _task(input_spec=None, allow_unverified_models: bool = False) -> BaseTask:
    return BaseTask(
        name="t_verify",
        model=_model(),
        data_loader=mock.MagicMock(),
        input_spec=input_spec,
        allow_unverified_models=allow_unverified_models,
    )


def _fake_bento(custom_objects: dict):
    b = mock.MagicMock()
    b.custom_objects = custom_objects
    return b


def test_predict_succeeds_when_bento_spec_matches():
    spec = _spec()
    task = _task(input_spec=spec)
    bento = _fake_bento({"input_spec": spec})

    with mock.patch("bentoml.picklable_model.get", return_value=bento):
        result = task.predict(PredictRequest(input_series={"cpu": [0.5]}))

    assert result.prediction == 0.42


def test_predict_refuses_bento_without_input_spec_by_default():
    task = _task(input_spec=_spec())
    bento = _fake_bento({})  # no input_spec — legacy pretrained Bento

    with mock.patch("bentoml.picklable_model.get", return_value=bento):
        with pytest.raises(FileNotFoundError, match="unverified|input_spec"):
            task.predict(PredictRequest(input_series={"cpu": [0.5]}))


def test_predict_refuses_bento_with_mismatched_n_features():
    task = _task(input_spec=_spec(n_features=1, feature_names=("cpu",)))
    bento = _fake_bento({"input_spec": _spec(n_features=3, feature_names=("cpu", "mem", "io"))})

    with mock.patch("bentoml.picklable_model.get", return_value=bento):
        with pytest.raises(FileNotFoundError, match="n_features|mismatch"):
            task.predict(PredictRequest(input_series={"cpu": [0.5]}))


def test_predict_refuses_bento_with_mismatched_feature_names():
    task = _task(input_spec=_spec(feature_names=("cpu",)))
    bento = _fake_bento({"input_spec": _spec(feature_names=("memory",))})

    with mock.patch("bentoml.picklable_model.get", return_value=bento):
        with pytest.raises(FileNotFoundError, match="feature_names|mismatch"):
            task.predict(PredictRequest(input_series={"cpu": [0.5]}))


def test_predict_refuses_bento_with_mismatched_steps_back():
    task = _task(input_spec=_spec(steps_back=1))
    bento = _fake_bento({"input_spec": _spec(steps_back=10)})

    with mock.patch("bentoml.picklable_model.get", return_value=bento):
        with pytest.raises(FileNotFoundError, match="steps_back|mismatch"):
            task.predict(PredictRequest(input_series={"cpu": [0.5]}))


def test_override_lets_missing_spec_through():
    task = _task(input_spec=_spec(), allow_unverified_models=True)
    bento = _fake_bento({})  # no input_spec

    with mock.patch("bentoml.picklable_model.get", return_value=bento):
        # Must not raise — override is on.
        result = task.predict(PredictRequest(input_series={"cpu": [0.5]}))
    assert result.prediction == 0.42


def test_override_lets_mismatched_spec_through():
    task = _task(
        input_spec=_spec(feature_names=("cpu",)),
        allow_unverified_models=True,
    )
    bento = _fake_bento({"input_spec": _spec(feature_names=("memory",))})

    with mock.patch("bentoml.picklable_model.get", return_value=bento):
        result = task.predict(PredictRequest(input_series={"cpu": [0.5]}))
    assert result.prediction == 0.42


def test_task_without_input_spec_skips_verification():
    """A task that doesn't declare an InputSpec can't verify anything —
    accept whatever the Bento has.
    """
    task = _task(input_spec=None)
    bento = _fake_bento({})  # anything

    with mock.patch("bentoml.picklable_model.get", return_value=bento):
        result = task.predict(PredictRequest(input_series={"cpu": [0.5]}))
    assert result.prediction == 0.42


def test_bento_spec_stored_as_dict_is_accepted():
    """Older Bentos may pickle the spec as a plain dict rather than an
    ``InputSpec`` instance. Verify by field, not by class.
    """
    task = _task(input_spec=_spec())
    bento = _fake_bento({"input_spec": {
        "n_features": 1,
        "feature_names": ["cpu"],
        "steps_back": 1,
    }})

    with mock.patch("bentoml.picklable_model.get", return_value=bento):
        result = task.predict(PredictRequest(input_series={"cpu": [0.5]}))
    assert result.prediction == 0.42
