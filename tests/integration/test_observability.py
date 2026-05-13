"""Internal observability — /metrics endpoint + structured JSON logs."""

from __future__ import annotations

import json
import logging

import pytest

pytestmark = pytest.mark.integration

api = pytest.importorskip("intelligence.api.service")


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    return TestClient(api.app)


def test_metrics_endpoint_returns_prometheus_text(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    # Counter families are present even before any traffic, with no samples.
    body = resp.text
    assert "intelligence_http_requests_total" in body
    assert "intelligence_train_total" in body
    assert "intelligence_predict_total" in body


def test_http_counter_increments_on_request(client):
    from intelligence.api.observability import HTTP_REQUESTS

    before = HTTP_REQUESTS.labels(route="/tasks", method="GET", status="200")._value.get()
    client.get("/tasks")
    after = HTTP_REQUESTS.labels(route="/tasks", method="GET", status="200")._value.get()
    assert after == before + 1


def test_metrics_endpoint_is_not_self_instrumented(client):
    """Scraping /metrics shouldn't add to /metrics counters — otherwise
    every scrape interval would generate noise and the rate would
    self-amplify on the very metrics meant to measure real traffic."""
    from intelligence.api.observability import HTTP_REQUESTS

    before = HTTP_REQUESTS.labels(route="/metrics", method="GET", status="200")._value.get()
    client.get("/metrics")
    client.get("/metrics")
    after = HTTP_REQUESTS.labels(route="/metrics", method="GET", status="200")._value.get()
    assert after == before


def test_healthz_is_skipped_too(client):
    """Liveness probes get hammered by k8s; they'd dominate the metric
    series if instrumented."""
    from intelligence.api.observability import HTTP_REQUESTS

    before = HTTP_REQUESTS.labels(route="/healthz", method="GET", status="200")._value.get()
    client.get("/healthz")
    after = HTTP_REQUESTS.labels(route="/healthz", method="GET", status="200")._value.get()
    assert after == before


def test_dynamic_paths_normalize_to_route_pattern(client):
    """A POST to /tasks/does_not_exist/predict should record against
    /tasks/{task}/predict, not against the literal path. Keeps label
    cardinality bounded."""
    from intelligence.api.observability import HTTP_REQUESTS

    # Use a path that's guaranteed to 404 (no task registered in this test config).
    client.post("/tasks/no_such_task/predict", json={"input_series": {"x": [0.1]}})
    # The "{task}" template should have at least one sample now.
    value = HTTP_REQUESTS.labels(
        route="/tasks/{task}/predict",
        method="POST",
        status="404",
    )._value.get()
    assert value >= 1


def test_request_id_filter_injects_id_onto_log_records():
    from intelligence.api.observability import RequestIdFilter, _request_id

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hi",
        args=(),
        exc_info=None,
    )
    token = _request_id.set("deadbeef")
    try:
        RequestIdFilter().filter(record)
        assert record.request_id == "deadbeef"
    finally:
        _request_id.reset(token)


def test_json_formatter_produces_well_formed_object_with_extras():
    from intelligence.api.observability import JsonFormatter

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    record.task_name = "cpu_forecast_arima"
    record.duration_ms = 12.3

    out = JsonFormatter().format(record)
    payload = json.loads(out)
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test"
    assert payload["message"] == "hello world"
    assert payload["task_name"] == "cpu_forecast_arima"
    assert payload["duration_ms"] == 12.3
    assert "timestamp" in payload
