"""GET /tasks/{task}/versions — list locally-stored Bento versions."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest import mock

import pytest

pytestmark = pytest.mark.integration

api = pytest.importorskip("intelligence.api.service")


@pytest.fixture
def app():
    return api.app


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


def _fake_model(name: str, version: str, when: datetime):
    m = mock.MagicMock()
    m.tag = mock.MagicMock(name=name, version=version)
    m.tag.name = name
    m.tag.version = version
    m.tag.__str__ = lambda _self: f"{name}:{version}"
    m.info.creation_time = when
    return m


def test_versions_endpoint_404_for_unknown_task(client):
    resp = client.get("/tasks/does_not_exist/versions")
    assert resp.status_code == 404


def test_versions_endpoint_returns_empty_list_when_no_training_has_happened(client):
    """A registered task with no trained models should still respond 200
    with an empty list (not 404). Pilots checking before their first
    train shouldn't see an error."""
    if "cpu_forecast_arima" not in api.registry:
        pytest.skip("cpu_forecast_arima not registered in this config")
    with mock.patch("bentoml.models.list", return_value=[]):
        resp = client.get("/tasks/cpu_forecast_arima/versions")
    assert resp.status_code == 200
    body = resp.json()
    assert body["task"] == "cpu_forecast_arima"
    assert body["versions"] == []
    # pinned_version key is present even when None — explicit in the contract.
    assert "pinned_version" in body


def test_versions_endpoint_sorts_newest_first(client):
    if "cpu_forecast_arima" not in api.registry:
        pytest.skip("cpu_forecast_arima not registered in this config")

    older = _fake_model("cpu_forecast_arima", "abc111", datetime(2026, 1, 1, tzinfo=timezone.utc))
    newer = _fake_model("cpu_forecast_arima", "abc222", datetime(2026, 5, 1, tzinfo=timezone.utc))
    unrelated = _fake_model("mem_forecast_arima", "abc333", datetime(2026, 6, 1, tzinfo=timezone.utc))

    with mock.patch("bentoml.models.list", return_value=[older, newer, unrelated]):
        resp = client.get("/tasks/cpu_forecast_arima/versions")
    assert resp.status_code == 200
    body = resp.json()
    versions = body["versions"]
    assert [v["version"] for v in versions] == ["abc222", "abc111"]
    # Unrelated bento_name didn't leak in.
    assert "abc333" not in [v["version"] for v in versions]
