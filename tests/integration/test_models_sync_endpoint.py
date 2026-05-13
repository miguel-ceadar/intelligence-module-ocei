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


def _make_hf_error(status_code: int, message: str):
    """Build an HfHubHTTPError that surfaces the right ``response.status_code``.

    The real exception is constructed by the SDK with a requests Response;
    we mimic that shape with a stub so the handler's status-code-based
    branching can be exercised without making a network call.
    """
    from huggingface_hub.errors import HfHubHTTPError

    class _StubResponse:
        def __init__(self, code: int) -> None:
            self.status_code = code
            self.headers: dict[str, str] = {}
            self.request = None

    return HfHubHTTPError(message, response=_StubResponse(status_code))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "upstream_status,expected_status",
    [(401, 401), (403, 403), (404, 404), (500, 502), (503, 502)],
)
def test_sync_endpoint_translates_hf_http_errors(
    upstream_status, expected_status, monkeypatch, client
):
    """Bad-token / repo-not-found / gated / 5xx from the HF API used to
    propagate as opaque 500s. Translate them: auth-class statuses pass
    through; everything else surfaces as 502 (upstream error)."""
    from intelligence.api import service

    monkeypatch.setattr(service.config.intelligence.model_repo, "hf_enabled", True)
    monkeypatch.setenv("HF_TOKEN", "fake")

    err = _make_hf_error(upstream_status, "upstream said no")
    with mock.patch("intelligence.api.service.pull_from_hf", side_effect=err):
        resp = client.post(
            "/models/sync",
            json={"action": "pull", "model_tag": "x:v1", "repo_id": "CeADAR/bentos"},
        )

    assert resp.status_code == expected_status, resp.text
    assert "upstream HF" in resp.json()["detail"]


def test_sync_endpoint_translates_transport_errors(monkeypatch, client):
    """Connection refused / timeout / DNS failure from inside the HF SDK
    must surface as 502 with a useful message, not an opaque 500."""
    import requests

    from intelligence.api import service

    monkeypatch.setattr(service.config.intelligence.model_repo, "hf_enabled", True)
    monkeypatch.setenv("HF_TOKEN", "fake")

    with mock.patch(
        "intelligence.api.service.pull_from_hf",
        side_effect=requests.ConnectionError("connection refused"),
    ):
        resp = client.post(
            "/models/sync",
            json={"action": "pull", "model_tag": "x:v1", "repo_id": "CeADAR/bentos"},
        )

    assert resp.status_code == 502
    assert "transport error" in resp.json()["detail"]
