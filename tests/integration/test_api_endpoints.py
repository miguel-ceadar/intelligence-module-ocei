"""Per-task HTTP endpoints.

Routes:
    GET    /healthz
    GET    /readyz
    GET    /tasks
    POST   /tasks/{task}/train
    POST   /tasks/{task}/predict
    GET    /models
    DELETE /models/{tag}
    POST   /models/sync

Tests use the FastAPI app via ``httpx``'s TestClient so no subprocess
boot is required.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from intelligence.api import service as api

pytestmark = pytest.mark.integration


@pytest.fixture
def client():
    return TestClient(api.app)


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_readyz(client):
    resp = client.get("/readyz")
    # 200 if all probes pass; 503 with failure list otherwise. Either is a
    # real answer; what we don't want is 404 (endpoint missing) or 500.
    assert resp.status_code in (200, 503)
    body = resp.json()
    if resp.status_code == 200:
        assert body["status"] == "ready"
        assert body["tasks"] >= 1
    else:
        assert body["status"] == "not_ready"
        assert isinstance(body["failures"], list) and body["failures"]


def test_readiness_helper_empty_registry_fails():
    """Direct unit test for the readiness aggregator — verifies the
    not-ready path that's hard to trigger via the live registry."""
    from intelligence.api.service import compute_readiness
    from intelligence.tasks import TaskRegistry

    ok, failures = compute_readiness(TaskRegistry())
    assert not ok
    assert any(f["check"] == "registry" for f in failures)


def test_list_tasks(client):
    resp = client.get("/tasks")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, (list, dict))


def test_unknown_task_returns_404(client):
    # Send a structurally-valid body so the 4xx is the route's task-lookup,
    # not pydantic's request-body validation.
    resp = client.post(
        "/tasks/does_not_exist/predict",
        json={"input_series": {"x": [0.1, 0.2]}},
    )
    assert resp.status_code == 404


def test_predict_input_spec_mismatch_returns_422(client):
    """Wrong feature count or steps_back gets rejected by the InputSpec
    validator before the artifact load — 422, not a deeper 500 from a
    numpy shape error.
    """
    resp = client.post(
        "/tasks/cpu_forecast_arima/predict",
        json={"input_series": {"cpu": [0.0, 0.0]}},
    )
    if resp.status_code == 404:
        pytest.skip("cpu_forecast_arima task not registered in this config")
    assert resp.status_code in (422, 503)


def test_models_list(client):
    resp = client.get("/models")
    assert resp.status_code == 200


def test_train_endpoint_accepts_static_data_source(client):
    """Train contract: body carries a ``data_source`` descriptor.

    ``kind: "static"`` reads from the configured samples directory;
    ``kind: "prometheus"`` pulls a window from PromQL.
    """
    body = {
        "data_source": {
            "kind": "static",
            "name": "cpu_sample_dataset_orangepi.csv",
        },
        "model_parameters": {"p": 2, "d": 1, "q": 0},
    }
    resp = client.post("/tasks/cpu_forecast_arima/train", json=body)
    if resp.status_code == 404:
        pytest.skip("cpu_forecast_arima task not registered in this config")
    assert resp.status_code in (200, 202), resp.text
    payload = resp.json()
    # Whatever the field name (``model_tag`` / ``bento_tag`` / ``trained_model``),
    # something identifying the saved model has to come back.
    assert any(k in payload for k in ("model_tag", "bento_tag", "trained_model", "tag"))


def test_train_endpoint_rejects_unknown_data_source_kind(client):
    """Unknown ``kind`` should fail at the API boundary, not deeper."""
    body = {"data_source": {"kind": "kafka"}, "model_parameters": {}}
    resp = client.post("/tasks/cpu_forecast_arima/train", json=body)
    if resp.status_code == 404:
        pytest.skip("cpu_forecast_arima task not registered in this config")
    assert resp.status_code == 422


def test_train_endpoint_maps_upstream_prometheus_failure_to_502(client):
    """A flaky upstream Prometheus (5xx / timeout / connection refused)
    raises ``requests.RequestException`` from inside the loader. The
    service must surface that as 502, not a generic 500 — pilots need to
    distinguish "my data source is down" from "the model crashed"."""
    import requests

    from intelligence.api import service as svc_mod
    from intelligence.tasks.base import BaseTask

    def _boom(_req):
        raise requests.ConnectionError("connection refused")

    fake = BaseTask(
        name="_test_upstream_502",
        model=None,
        data_loader=lambda _ds: {},
        bento_name="_test_upstream_502",
    )
    fake.train = _boom  # type: ignore[method-assign]
    svc_mod.registry.register(fake)
    try:
        resp = client.post(
            "/tasks/_test_upstream_502/train",
            json={"data_source": {"kind": "static", "name": "any.csv"}},
        )
        assert resp.status_code == 502, resp.text
        assert "upstream" in resp.json()["detail"].lower()
    finally:
        svc_mod.registry._tasks.pop("_test_upstream_502", None)
