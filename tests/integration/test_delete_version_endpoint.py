"""``DELETE /tasks/{task}/versions/{version}`` endpoint.

Reclaims PVC space by deleting a specific stored version. Guards against
deleting ``latest`` or the currently pinned version of a task.
"""

from __future__ import annotations

from unittest import mock

import pytest

from intelligence.api import service as api

pytestmark = pytest.mark.integration


class _FakeTask:
    """Minimal task stub: only the attributes the DELETE handler reads."""

    def __init__(
        self,
        name: str,
        bento_name: str | None = None,
        pinned_version: str | None = None,
    ) -> None:
        self.name = name
        self.bento_name = bento_name or name
        self.pinned_version = pinned_version
        self._cached_artifacts: dict[str, object] = {}


@pytest.fixture
def app():
    return api.app


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient

    return TestClient(app)


@pytest.fixture
def registered_task(monkeypatch):
    """Drop a fake task into the live registry for the duration of one test."""
    from intelligence.tasks.base import TaskRegistry

    reg = TaskRegistry()
    task = _FakeTask("cpu_forecast_arima")
    reg.register(task)
    monkeypatch.setattr(api, "registry", reg)
    return task


def test_delete_version_404_for_unknown_task(client):
    resp = client.delete("/tasks/does_not_exist/versions/abc123")
    assert resp.status_code == 404


def test_delete_version_400_on_latest(client, registered_task):
    """``latest`` must be rejected explicitly — operators can't accidentally
    drop the most recent model with a `:latest` URL."""
    resp = client.delete("/tasks/cpu_forecast_arima/versions/latest")
    assert resp.status_code == 400
    assert "latest" in resp.json()["detail"]


def test_delete_version_409_on_pinned_version(client, monkeypatch):
    """Refuse to delete a version that's currently pinned for the task —
    would immediately break /predict."""
    from intelligence.tasks.base import TaskRegistry

    reg = TaskRegistry()
    reg.register(_FakeTask("cpu_forecast_arima", pinned_version="pinned123"))
    monkeypatch.setattr(api, "registry", reg)

    resp = client.delete("/tasks/cpu_forecast_arima/versions/pinned123")
    assert resp.status_code == 409
    assert "pinned" in resp.json()["detail"]


def test_delete_version_happy_path(client, registered_task):
    """Successful delete returns 200 with the deleted version, calls
    bentoml.models.delete with the right tag, and invalidates cached
    entries (the deleted version + ``latest``)."""
    registered_task._cached_artifacts["abc123"] = ("loaded", "tag")
    registered_task._cached_artifacts["latest"] = ("loaded", "tag")

    with mock.patch("bentoml.models.delete") as mock_delete:
        resp = client.delete("/tasks/cpu_forecast_arima/versions/abc123")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == {"task": "cpu_forecast_arima", "deleted": "abc123"}
    mock_delete.assert_called_once_with("cpu_forecast_arima:abc123")
    assert "abc123" not in registered_task._cached_artifacts
    assert "latest" not in registered_task._cached_artifacts


def test_delete_version_404_when_bentoml_doesnt_have_it(client, registered_task):
    """A version string the local store doesn't know about returns 404
    instead of a generic 500."""
    from bentoml.exceptions import NotFound

    with mock.patch("bentoml.models.delete", side_effect=NotFound("nope")):
        resp = client.delete("/tasks/cpu_forecast_arima/versions/ghost")

    assert resp.status_code == 404
    assert "ghost" in resp.json()["detail"]
