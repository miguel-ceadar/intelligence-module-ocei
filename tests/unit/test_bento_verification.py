"""Pretrained-artefact policy: refuse predict from artefacts whose
stored ``input_spec`` is missing or doesn't match the task's spec.

The contract travels with the model — it's written into
``input_spec.json`` at train time (``BaseTask.train`` injects
``self.input_spec`` into the artefacts dict; the per-kind
``save_artifacts`` persists it via the sidecar helper). At predict
time we check the loaded artefact's spec against the task's current
spec. Mismatches surface as the same 503 response shape as "no trained
model", since operationally the situation is the same: the caller
needs to train (or pull a matching artefact).

Override: ``BaseTask.allow_unverified_models=True`` lets predict
proceed with a warning. Intended for debugging or for someone who has
intentionally accepted the risk of a pulled artefact that predates
the contract.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from intelligence.api.schemas import PredictRequest
from intelligence.ml.artifact import SavedArtifact
from intelligence.ml.artifact.manifest import Manifest
from intelligence.tasks import BaseTask
from intelligence.tasks.contracts import InputSpec


def _spec(n_features: int = 1, steps_back: int = 1, feature_names=("cpu",)) -> InputSpec:
    return InputSpec(
        n_features=n_features,
        feature_names=list(feature_names),
        steps_back=steps_back,
    )


def _model(loaded: dict):
    m = mock.MagicMock()
    m.name = "fake"
    m.has_drift = False
    m.predict.return_value = 0.42
    m.load_artifacts.return_value = loaded
    return m


def _task(loaded: dict, input_spec=None, allow_unverified_models: bool = False) -> BaseTask:
    return BaseTask(
        name="t_verify",
        model=_model(loaded),
        data_loader=mock.MagicMock(),
        input_spec=input_spec,
        allow_unverified_models=allow_unverified_models,
    )


def _fake_saved() -> SavedArtifact:
    return SavedArtifact(
        tag="t_verify:v1",
        name="t_verify",
        version="v1",
        path=Path("/fake/path"),
        manifest=Manifest(
            schema_version=1,
            kind="fake",
            created_at="2026-05-13T10:00:00+00:00",
            files={},
        ),
        created_at="2026-05-13T10:00:00+00:00",
    )


def test_predict_succeeds_when_artefact_spec_matches():
    spec = _spec()
    task = _task(loaded={"input_spec": spec}, input_spec=spec)

    with mock.patch("intelligence.tasks.base.get_artifact_by_tag") as get:
        get.return_value = _fake_saved()
        result = task.predict(PredictRequest(input_series={"cpu": [0.5]}))

    assert result.prediction == 0.42


def test_predict_refuses_artefact_without_input_spec_by_default():
    task = _task(loaded={}, input_spec=_spec())  # no input_spec in loaded dict

    with (
        mock.patch("intelligence.tasks.base.get_artifact_by_tag") as get,
        pytest.raises(FileNotFoundError, match=r"unverified|input_spec"),
    ):
        get.return_value = _fake_saved()
        task.predict(PredictRequest(input_series={"cpu": [0.5]}))


def test_predict_refuses_artefact_with_mismatched_n_features():
    task = _task(
        loaded={"input_spec": _spec(n_features=3, feature_names=("cpu", "mem", "io"))},
        input_spec=_spec(n_features=1, feature_names=("cpu",)),
    )
    with (
        mock.patch("intelligence.tasks.base.get_artifact_by_tag") as get,
        pytest.raises(FileNotFoundError, match=r"n_features|mismatch"),
    ):
        get.return_value = _fake_saved()
        task.predict(PredictRequest(input_series={"cpu": [0.5]}))


def test_predict_refuses_artefact_with_mismatched_feature_names():
    task = _task(
        loaded={"input_spec": _spec(feature_names=("memory",))},
        input_spec=_spec(feature_names=("cpu",)),
    )
    with (
        mock.patch("intelligence.tasks.base.get_artifact_by_tag") as get,
        pytest.raises(FileNotFoundError, match=r"feature_names|mismatch"),
    ):
        get.return_value = _fake_saved()
        task.predict(PredictRequest(input_series={"cpu": [0.5]}))


def test_predict_refuses_artefact_with_mismatched_steps_back():
    task = _task(
        loaded={"input_spec": _spec(steps_back=10)},
        input_spec=_spec(steps_back=1),
    )
    with (
        mock.patch("intelligence.tasks.base.get_artifact_by_tag") as get,
        pytest.raises(FileNotFoundError, match=r"steps_back|mismatch"),
    ):
        get.return_value = _fake_saved()
        task.predict(PredictRequest(input_series={"cpu": [0.5]}))


def test_override_lets_missing_spec_through():
    task = _task(loaded={}, input_spec=_spec(), allow_unverified_models=True)

    with mock.patch("intelligence.tasks.base.get_artifact_by_tag") as get:
        get.return_value = _fake_saved()
        result = task.predict(PredictRequest(input_series={"cpu": [0.5]}))
    assert result.prediction == 0.42


def test_override_lets_mismatched_spec_through():
    task = _task(
        loaded={"input_spec": _spec(feature_names=("memory",))},
        input_spec=_spec(feature_names=("cpu",)),
        allow_unverified_models=True,
    )
    with mock.patch("intelligence.tasks.base.get_artifact_by_tag") as get:
        get.return_value = _fake_saved()
        result = task.predict(PredictRequest(input_series={"cpu": [0.5]}))
    assert result.prediction == 0.42


def test_task_without_input_spec_skips_verification():
    """A task that doesn't declare an InputSpec can't verify anything —
    accept whatever the artefact has.
    """
    task = _task(loaded={}, input_spec=None)

    with mock.patch("intelligence.tasks.base.get_artifact_by_tag") as get:
        get.return_value = _fake_saved()
        result = task.predict(PredictRequest(input_series={"cpu": [0.5]}))
    assert result.prediction == 0.42
