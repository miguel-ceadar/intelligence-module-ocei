"""Predict-time version pinning.

Three resolution paths:
- request.model_version (explicit pin per call)
- task.pinned_version (operator-set per task in config)
- ``:latest`` (default — most recent trained model)

Request wins over task pin; task pin wins over latest.
"""

from __future__ import annotations

from unittest import mock

import bentoml
import pytest

from intelligence.api.schemas import PredictRequest
from intelligence.tasks import BaseTask


@pytest.fixture(autouse=True)
def _isolated_bento_home(tmp_path, monkeypatch):
    monkeypatch.setenv("BENTOML_HOME", str(tmp_path / "bentoml"))


def _make_model():
    m = mock.MagicMock()
    m.name = "fake"
    m.has_drift = False
    m.predict.return_value = 0.5
    return m


def _fake_bento(version: str):
    b = mock.MagicMock()
    b.tag = f"t:{version}"
    b.custom_objects = {}
    return b


def test_request_pin_overrides_everything():
    """An explicit request.model_version always wins."""
    task = BaseTask(
        name="t",
        model=_make_model(),
        data_loader=mock.MagicMock(),
        pinned_version="task_pin_v1",
    )
    fake = _fake_bento("request_pin_v2")
    with mock.patch("bentoml.picklable_model.get", return_value=fake) as get:
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
    fake = _fake_bento("task_pin_v1")
    with mock.patch("bentoml.picklable_model.get", return_value=fake) as get:
        resp = task.predict(PredictRequest(input_series={"x": [1.0]}))
    get.assert_called_once_with("t:task_pin_v1")
    assert resp.model_version == "task_pin_v1"


def test_latest_used_when_no_pin_at_all():
    task = BaseTask(name="t", model=_make_model(), data_loader=mock.MagicMock())
    fake = _fake_bento("abc123")
    with mock.patch("bentoml.picklable_model.get", return_value=fake) as get:
        resp = task.predict(PredictRequest(input_series={"x": [1.0]}))
    get.assert_called_once_with("t:latest")
    # The response carries the *actual* version that served, not "latest".
    assert resp.model_version == "abc123"


def test_pinned_versions_cached_separately_from_latest():
    """Different versions should hit different cache slots — a pinned
    cache entry must not be invalidated by a fresh train."""
    task = BaseTask(name="t", model=_make_model(), data_loader=mock.MagicMock())
    fake_latest = _fake_bento("v1")
    fake_pinned = _fake_bento("v0")

    def by_tag(tag: str):
        return fake_pinned if tag.endswith(":v0") else fake_latest

    with mock.patch("bentoml.picklable_model.get", side_effect=by_tag) as get:
        task.predict(PredictRequest(input_series={"x": [1.0]}))  # caches latest=v1
        task.predict(PredictRequest(input_series={"x": [1.0]}, model_version="v0"))  # caches v0
        # _invalidate clears only :latest, not v0
        task._invalidate()
        task.predict(
            PredictRequest(input_series={"x": [1.0]}, model_version="v0")
        )  # served from cache
        task.predict(PredictRequest(input_series={"x": [1.0]}))  # latest re-fetched
    # 3 actual store reads: initial latest, initial v0, post-invalidate latest
    assert get.call_count == 3


def test_missing_pinned_version_raises_filenotfound_with_helpful_message():
    task = BaseTask(name="t", model=_make_model(), data_loader=mock.MagicMock())
    with (
        mock.patch(
            "bentoml.picklable_model.get",
            side_effect=bentoml.exceptions.NotFound("not in store"),
        ),
        pytest.raises(FileNotFoundError, match="t:abc-does-not-exist"),
    ):
        task.predict(
            PredictRequest(
                input_series={"x": [1.0]},
                model_version="abc-does-not-exist",
            )
        )
