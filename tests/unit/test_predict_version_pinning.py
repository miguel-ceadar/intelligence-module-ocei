"""Predict-time version pinning.

Three resolution paths:
- request.model_version (explicit pin per call)
- task.pinned_version (operator-set per task in config)
- ``:latest`` (default — most recent trained model)

Request wins over task pin; task pin wins over latest.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from intelligence.api.schemas import PredictRequest
from intelligence.ml.artifact import SavedArtifact
from intelligence.ml.artifact.manifest import Manifest
from intelligence.tasks import BaseTask


@pytest.fixture(autouse=True)
def _isolated_bento_home(tmp_path, monkeypatch):
    monkeypatch.setenv("BENTOML_HOME", str(tmp_path / "bentoml"))


def _make_model():
    from intelligence.api.schemas import ForecastPoint

    m = mock.MagicMock()
    m.name = "fake"
    m.has_drift = False
    m.predict.return_value = [ForecastPoint(value=0.5)]
    m.load_artifacts.return_value = {}
    return m


def _fake_saved(version: str) -> SavedArtifact:
    return SavedArtifact(
        tag=f"t:{version}",
        name="t",
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


def test_request_pin_overrides_everything():
    """An explicit request.model_version always wins."""
    task = BaseTask(
        name="t",
        model=_make_model(),
        data_loader=mock.MagicMock(),
        pinned_version="task_pin_v1",
    )
    with mock.patch("intelligence.tasks.base.get_artifact_by_tag") as get:
        get.return_value = _fake_saved("request_pin_v2")
        resp = task.predict(
            PredictRequest(
                input_series={"x": [1.0]},
                model_version="request_pin_v2",
            )
        )
    get.assert_called_once_with("t:request_pin_v2")
    assert resp.model_version == "request_pin_v2"


def test_task_pin_used_when_no_request_pin():
    task = BaseTask(
        name="t",
        model=_make_model(),
        data_loader=mock.MagicMock(),
        pinned_version="task_pin_v1",
    )
    with mock.patch("intelligence.tasks.base.get_artifact_by_tag") as get:
        get.return_value = _fake_saved("task_pin_v1")
        resp = task.predict(PredictRequest(input_series={"x": [1.0]}))
    get.assert_called_once_with("t:task_pin_v1")
    assert resp.model_version == "task_pin_v1"


def test_latest_used_when_no_pin_at_all():
    task = BaseTask(name="t", model=_make_model(), data_loader=mock.MagicMock())
    with mock.patch("intelligence.tasks.base.get_artifact_by_tag") as get:
        get.return_value = _fake_saved("abc123")
        resp = task.predict(PredictRequest(input_series={"x": [1.0]}))
    get.assert_called_once_with("t:latest")
    # Response carries the *actual* version that served, not "latest".
    assert resp.model_version == "abc123"


def test_pinned_versions_cached_separately_from_latest():
    """Different versions should hit different cache slots — a pinned
    cache entry must not be invalidated by a fresh train."""
    task = BaseTask(name="t", model=_make_model(), data_loader=mock.MagicMock())

    def by_tag(tag: str) -> SavedArtifact:
        return _fake_saved("v0") if tag.endswith(":v0") else _fake_saved("v1")

    with mock.patch("intelligence.tasks.base.get_artifact_by_tag", side_effect=by_tag) as get:
        task.predict(PredictRequest(input_series={"x": [1.0]}))  # caches latest=v1
        task.predict(PredictRequest(input_series={"x": [1.0]}, model_version="v0"))  # caches v0
        # _invalidate clears only :latest, not v0
        task._invalidate()
        task.predict(
            PredictRequest(input_series={"x": [1.0]}, model_version="v0")
        )  # served from cache
        task.predict(PredictRequest(input_series={"x": [1.0]}))  # latest re-fetched
    # 3 actual store reads: initial latest, initial v0, post-invalidate latest.
    assert get.call_count == 3


def test_missing_pinned_version_raises_filenotfound_with_helpful_message():
    task = BaseTask(name="t", model=_make_model(), data_loader=mock.MagicMock())
    with (
        mock.patch("intelligence.tasks.base.get_artifact_by_tag") as get,
        pytest.raises(FileNotFoundError, match="t:abc-does-not-exist"),
    ):
        get.return_value = None
        task.predict(
            PredictRequest(
                input_series={"x": [1.0]},
                model_version="abc-does-not-exist",
            )
        )
