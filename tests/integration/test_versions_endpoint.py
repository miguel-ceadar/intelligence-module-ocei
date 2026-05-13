"""GET /tasks/{task}/versions — list locally-stored artefact versions."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from intelligence.ml.artifact import SavedArtifact
from intelligence.ml.artifact.manifest import Manifest

pytestmark = pytest.mark.integration

api = pytest.importorskip("intelligence.api.service")


@pytest.fixture
def app():
    return api.app


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient

    return TestClient(app)


def _fake_artifact(name: str, version: str, when: str) -> SavedArtifact:
    return SavedArtifact(
        tag=f"{name}:{version}",
        name=name,
        version=version,
        path=Path("/fake"),
        manifest=Manifest(
            schema_version=1,
            kind="arima",
            created_at=when,
            files={},
        ),
        created_at=when,
    )


def test_versions_endpoint_404_for_unknown_task(client):
    resp = client.get("/tasks/does_not_exist/versions")
    assert resp.status_code == 404


def test_versions_endpoint_returns_empty_list_when_no_training_has_happened(client):
    """A registered task with no trained models should still respond 200
    with an empty list (not 404). Pilots checking before their first
    train shouldn't see an error."""
    if "cpu_forecast_arima" not in api.registry:
        pytest.skip("cpu_forecast_arima not registered in this config")
    with mock.patch(
        "intelligence.api.service.list_artifacts_by_name", return_value=[]
    ):
        resp = client.get("/tasks/cpu_forecast_arima/versions")
    assert resp.status_code == 200
    body = resp.json()
    assert body["task"] == "cpu_forecast_arima"
    assert body["versions"] == []
    # pinned_version key is present even when None — explicit in the contract.
    assert "pinned_version" in body


def test_versions_endpoint_sorts_newest_first(client):
    """``list_artifacts_by_name`` returns newest first; the endpoint
    preserves that order in the response.
    """
    if "cpu_forecast_arima" not in api.registry:
        pytest.skip("cpu_forecast_arima not registered in this config")

    # list_artifacts_by_name already filters by name and sorts newest first;
    # the endpoint just maps to JSON.
    newer = _fake_artifact("cpu_forecast_arima", "abc222", "2026-05-01T00:00:00+00:00")
    older = _fake_artifact("cpu_forecast_arima", "abc111", "2026-01-01T00:00:00+00:00")

    with mock.patch(
        "intelligence.api.service.list_artifacts_by_name",
        return_value=[newer, older],
    ):
        resp = client.get("/tasks/cpu_forecast_arima/versions")
    assert resp.status_code == 200
    body = resp.json()
    assert [v["version"] for v in body["versions"]] == ["abc222", "abc111"]
