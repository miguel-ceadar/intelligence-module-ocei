"""``POST /models/sync`` endpoint.

Wiring tests — verifies the endpoint exists, dispatches to the right
push/pull helper, and surfaces missing-token / disabled-config as
useful HTTP errors.
"""

from __future__ import annotations

from unittest import mock

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def app():
    from intelligence.api.service import app as _app

    return _app


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient

    return TestClient(app)


def test_sync_endpoint_403_when_hf_disabled(client):
    """Default config has ``hf_enabled=False`` — sync requests must 403,
    not silently succeed."""
    resp = client.post(
        "/models/sync",
        json={"action": "push", "model_tag": "x:latest", "repo_id": "CeADAR/bentos"},
    )
    assert resp.status_code == 403


def test_sync_endpoint_dispatches_push(monkeypatch, client):
    """When hf_enabled and HF_TOKEN are set, the endpoint calls
    ``push_to_hf`` and returns the returned tag.
    """
    from intelligence.api import service

    # Flip the toggle on the live service config.
    monkeypatch.setattr(
        service.config.intelligence.model_repo,
        "hf_enabled",
        True,
    )
    monkeypatch.setenv("HF_TOKEN", "fake")

    with mock.patch("intelligence.api.service.push_to_hf", return_value="x:v1") as mock_push:
        resp = client.post(
            "/models/sync",
            json={"action": "push", "model_tag": "x:latest", "repo_id": "CeADAR/bentos"},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["action"] == "push"
    assert body["model_tag"] == "x:v1"
    mock_push.assert_called_once()


def test_sync_endpoint_401_when_token_missing(monkeypatch, client):
    """``HF_TOKEN`` unset → 401 from the endpoint."""
    from intelligence.api import service

    monkeypatch.setattr(
        service.config.intelligence.model_repo,
        "hf_enabled",
        True,
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)

    resp = client.post(
        "/models/sync",
        json={"action": "push", "model_tag": "x:latest", "repo_id": "CeADAR/bentos"},
    )
    assert resp.status_code == 401
    assert "HF_TOKEN" in resp.json()["detail"]
