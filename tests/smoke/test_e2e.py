"""End-to-end smoke against a running compose stack.

Marked ``smoke`` so it doesn't run under plain ``pytest`` / ``make test``.
Use ``make e2e`` to boot the stack and run this suite, or
``pytest -m smoke`` against any already-running deployment.

Environment:
    INTELLIGENCE_SMOKE_URL  default http://localhost:3000
    PROMETHEUS_URL          default http://localhost:9090
    SMOKE_MAX_WAIT          default 180 (seconds for prometheus to fill)
"""

from __future__ import annotations

import os
import time

import httpx
import pytest

pytestmark = pytest.mark.smoke

INTELLIGENCE_URL = os.environ.get("INTELLIGENCE_SMOKE_URL", "http://localhost:3000").rstrip("/")
PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9090").rstrip("/")
MAX_WAIT_S = int(os.environ.get("SMOKE_MAX_WAIT", "180"))


def _poll(url: str, *, timeout: float, interval: float = 1.0) -> None:
    deadline = time.time() + timeout
    last: object = "no attempt yet"
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=5.0)
            if r.status_code == 200:
                return
            last = f"HTTP {r.status_code}"
        except httpx.HTTPError as e:
            last = e
        time.sleep(interval)
    pytest.fail(f"{url} not ready after {timeout}s (last: {last})")


def _wait_for_samples(min_samples: int, timeout: float) -> None:
    """Wait until prometheus has scraped enough cpu samples to train against."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(
                f"{PROMETHEUS_URL}/api/v1/query",
                params={"query": 'count_over_time(node_cpu_seconds_total{mode="user"}[5m])'},
                timeout=5.0,
            )
            r.raise_for_status()
            results = r.json().get("data", {}).get("result", [])
            if results and float(results[0]["value"][1]) >= min_samples:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    pytest.fail(f"prometheus did not accumulate >= {min_samples} samples in {timeout}s")


@pytest.fixture(scope="module", autouse=True)
def _stack_ready() -> None:
    _poll(f"{INTELLIGENCE_URL}/healthz", timeout=60)
    _poll(f"{PROMETHEUS_URL}/-/healthy", timeout=60)
    # 60+ samples at 2s scrape ≈ 2 minutes of data — enough for ARIMA on
    # last-observation, XGB/LSTM with look_back=6 + 80/20 split, and drift
    # with chunk_size=12.
    _wait_for_samples(60, timeout=MAX_WAIT_S)


# (task, train_window, predict_input_series). ``predict_input_series`` is
# the body that goes under ``input_series`` — a dict so multivariate
# tasks can declare multiple feature windows.
_CPU_WINDOW = [0.3, 0.4, 0.5, 0.4, 0.3, 0.5]
_MEM_WINDOW = [0.6, 0.6, 0.6, 0.7, 0.7, 0.7]
TASKS: list[tuple[str, str, dict[str, list[float]]]] = [
    # Univariate baseline
    ("cpu_forecast_arima", "2m", {"cpu": [0.5]}),
    ("cpu_forecast_xgb", "2m", {"cpu": _CPU_WINDOW}),
    ("cpu_forecast_lstm", "2m", {"cpu": _CPU_WINDOW}),
    ("cpu_forecast_arima_drift", "2m", {"cpu": [0.3] * 12}),
    # Multivariate (Phase 3)
    ("cpu_mem_forecast_xgb", "2m", {"cpu": _CPU_WINDOW, "mem": _MEM_WINDOW}),
    ("cpu_mem_forecast_lstm", "2m", {"cpu": _CPU_WINDOW, "mem": _MEM_WINDOW}),
    ("cpu_mem_drift", "2m", {"cpu": [0.3] * 12, "mem": [0.6] * 12}),
]


@pytest.mark.parametrize("task,window,predict_input_series", TASKS, ids=[t[0] for t in TASKS])
def test_train_then_predict(
    task: str, window: str, predict_input_series: dict[str, list[float]]
) -> None:
    train = httpx.post(
        f"{INTELLIGENCE_URL}/tasks/{task}/train",
        json={"data_source": {"kind": "prometheus", "window": window, "step": "2s"}},
        timeout=180.0,
    )
    assert train.status_code == 200, f"train {task}: {train.status_code} {train.text}"
    assert "model_tag" in train.json()

    predict = httpx.post(
        f"{INTELLIGENCE_URL}/tasks/{task}/predict",
        json={"input_series": predict_input_series},
        timeout=30.0,
    )
    assert predict.status_code == 200, f"predict {task}: {predict.status_code} {predict.text}"
    body = predict.json()
    assert "prediction" in body
    # Forecast tasks return list[ForecastPoint] of length horizon (default 1);
    # drift task keeps a dict-shaped prediction.
    if task.endswith("_drift"):
        assert isinstance(body["prediction"], dict)
    else:
        assert isinstance(body["prediction"], list) and body["prediction"]
        assert "value" in body["prediction"][0]


def test_arima_multi_horizon_returns_confidence_intervals() -> None:
    """ARIMA exposes native 95 % CIs — every point should carry both bounds."""
    r = httpx.post(
        f"{INTELLIGENCE_URL}/tasks/cpu_forecast_arima/predict",
        json={"input_series": {"cpu": [0.5]}, "horizon": 4},
        timeout=30.0,
    )
    assert r.status_code == 200, r.text
    points = r.json()["prediction"]
    assert isinstance(points, list) and len(points) == 4
    for p in points:
        assert p.get("lower") is not None and p.get("upper") is not None
        assert p["lower"] <= p["value"] <= p["upper"]


def test_readyz_after_bootstrap() -> None:
    r = httpx.get(f"{INTELLIGENCE_URL}/readyz", timeout=5.0)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ready"
    # 4 univariate + 3 multivariate = 7 registered tasks in the demo config.
    assert body["tasks"] >= 7


def test_arima_rejects_multivariate_predict_input() -> None:
    """ARIMA is single-feature by design. Sending a multi-feature
    ``input_series`` to an ARIMA task must fail at the contract boundary
    (422) — the InputSpec's ``n_features`` is 1 and validation rejects
    the extra key.
    """
    r = httpx.post(
        f"{INTELLIGENCE_URL}/tasks/cpu_forecast_arima/predict",
        json={"input_series": {"cpu": [0.5], "mem": [0.6]}},
        timeout=10.0,
    )
    assert r.status_code == 422, r.text


def test_multivariate_xgb_predict_rejects_missing_covariate() -> None:
    """A multivariate task must receive every feature it was trained
    on. Sending only the target should fail validation (422) rather
    than silently dropping the covariate."""
    r = httpx.post(
        f"{INTELLIGENCE_URL}/tasks/cpu_mem_forecast_xgb/predict",
        json={"input_series": {"cpu": _CPU_WINDOW}},
        timeout=10.0,
    )
    assert r.status_code == 422, r.text
