"""Phase-1 §2.3: per-task endpoints replace the mega ``/predict``.

Routes:
    GET    /healthz
    GET    /readyz
    GET    /tasks
    POST   /tasks/{task}/train
    POST   /tasks/{task}/predict
    POST   /tasks/{task}/drift           (404 if task lacks drift support)
    GET    /models
    DELETE /models/{tag}
    POST   /models/sync

Tests use the BentoML service's ASGI app via ``httpx`` so no subprocess
boot is required.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

api = pytest.importorskip("intelligence.api.service", reason="phase-1 §2.3 pending")
httpx = pytest.importorskip("httpx")


@pytest.fixture
def app():
    svc = getattr(api, "svc", None) or getattr(api, "build_service", lambda: None)()
    if svc is None:
        pytest.skip("intelligence.api.service.svc / build_service not implemented yet")
    asgi = getattr(svc, "asgi_app", None)
    if asgi is None:
        pytest.skip("BentoML version does not expose `svc.asgi_app`; revise harness")
    return asgi


@pytest.fixture
def client(app):
    transport = httpx.ASGITransport(app=app)
    with httpx.Client(transport=transport, base_url="http://testserver") as c:
        yield c


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200


def test_readyz(client):
    resp = client.get("/readyz")
    # 200 if all required tasks loadable; 503 otherwise. Either is a real answer;
    # what we don't want is 404 (endpoint missing).
    assert resp.status_code in (200, 503)


def test_list_tasks(client):
    resp = client.get("/tasks")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, (list, dict))


def test_unknown_task_returns_404(client):
    resp = client.post("/tasks/does_not_exist/predict", json={})
    assert resp.status_code == 404


def test_predict_input_spec_mismatch_returns_422(client):
    """Wrong feature count or steps_back should produce a 4xx (ideally 422),
    not a 500 from a numpy shape error inside the runner."""
    # Use a task that's expected to be registered post-refactor.
    resp = client.post(
        "/tasks/cpu_forecast_arima/predict",
        json={"input_series": {"cpu": [0.0, 0.0]}},  # too few steps
    )
    if resp.status_code == 404:
        pytest.skip("cpu_forecast_arima task not registered in this config")
    assert resp.status_code == 422


def test_models_list(client):
    resp = client.get("/models")
    assert resp.status_code == 200


def test_train_endpoint_accepts_static_data_source(client):
    """Phase-1 train contract: body carries a `data_source` descriptor.

    Phase 1 only implements ``kind: "static"`` (read from ``oasis/dataset/``).
    Phase 2 adds ``kind: "prometheus"`` against PromQL — same endpoint
    signature, no breaking change for callers.
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


def test_train_endpoint_prometheus_kind_unimplemented_in_phase_1(client):
    """``kind: "prometheus"`` is the phase-2 add. Phase 1 should refuse
    cleanly (501 Not Implemented or 422), not pretend to work."""
    body = {
        "data_source": {"kind": "prometheus", "window": "24h", "step": "1m"},
        "model_parameters": {},
    }
    resp = client.post("/tasks/cpu_forecast_arima/train", json=body)
    if resp.status_code == 404:
        pytest.skip("cpu_forecast_arima task not registered in this config")
    assert resp.status_code in (501, 422), resp.text
