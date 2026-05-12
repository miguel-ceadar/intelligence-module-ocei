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
pytest.importorskip("fastapi")  # TestClient lives here


@pytest.fixture
def app():
    # Prefer the FastAPI app directly when exported (faster + simpler than
    # going through the bentoml.Service ASGI wrapper).
    fastapi_app = getattr(api, "app", None)
    if fastapi_app is not None:
        return fastapi_app
    svc = getattr(api, "svc", None) or getattr(api, "build_service", lambda: None)()
    if svc is None:
        pytest.skip("intelligence.api.service.{app,svc} not implemented yet")
    asgi = getattr(svc, "asgi_app", None)
    if asgi is None:
        pytest.skip("BentoML version does not expose `svc.asgi_app`; revise harness")
    return asgi


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


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
    """Wrong feature count or steps_back should produce a 4xx (ideally 422),
    not a 500 from a numpy shape error inside the runner.

    Pre-task-#11 (no InputSpec validation) the request hits a
    model-not-found path and gets 503 — also a real 4xx/5xx answer, not
    a crash. Once #11 lands, validation rejects pre-load and the
    response narrows to 422.
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
