"""Stress suite — runs against the same demo stack as ``test_e2e.py``
but exercises the heavier paths a quick smoke skips. Marked ``stress``
so it stays out of the default ``pytest`` and the quick ``make e2e``.

Run with ``make stress`` against an already-up demo stack. The stack's
Prometheus retention is 12 hours by default, so the long-window
training tests need the stack to have been up for at least that
duration's worth of data — letting it warm for 15+ minutes after
``make up-demo`` is enough to exercise everything except the truly
long-window case.

Environment overrides (mirror ``test_e2e.py``):
    INTELLIGENCE_SMOKE_URL  default http://localhost:3000
    PROMETHEUS_URL          default http://localhost:9090
    STRESS_PREDICT_LOOPS    default 200 (predict iterations per task)
    STRESS_DRIFT_LOOPS      default 60  (drift-over-real-time iterations)
    STRESS_DRIFT_INTERVAL_S default 1.0 (sleep between drift predicts)
    STRESS_LONG_WINDOW      default "6h" (long-window train target)
    STRESS_P95_BUDGET_MS    default 250 (p95 predict latency budget)
"""

from __future__ import annotations

import os
import statistics
import time

import httpx
import pytest

pytestmark = pytest.mark.stress

INTELLIGENCE_URL = os.environ.get("INTELLIGENCE_SMOKE_URL", "http://localhost:3000").rstrip("/")
PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9090").rstrip("/")

PREDICT_LOOPS = int(os.environ.get("STRESS_PREDICT_LOOPS", "200"))
DRIFT_LOOPS = int(os.environ.get("STRESS_DRIFT_LOOPS", "60"))
DRIFT_INTERVAL_S = float(os.environ.get("STRESS_DRIFT_INTERVAL_S", "1.0"))
LONG_WINDOW = os.environ.get("STRESS_LONG_WINDOW", "6h")
P95_BUDGET_MS = int(os.environ.get("STRESS_P95_BUDGET_MS", "250"))


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


@pytest.fixture(scope="module", autouse=True)
def _stack_ready() -> None:
    _poll(f"{INTELLIGENCE_URL}/healthz", timeout=60)
    _poll(f"{PROMETHEUS_URL}/-/healthy", timeout=60)


# ---- High predict volume -------------------------------------------------

_CPU_WINDOW = [0.3, 0.4, 0.5, 0.4, 0.3, 0.5]
_MEM_WINDOW = [0.6, 0.6, 0.6, 0.7, 0.7, 0.7]

_PREDICT_TASKS: list[tuple[str, dict[str, list[float]]]] = [
    ("cpu_forecast_xgb", {"cpu": _CPU_WINDOW}),
    ("cpu_forecast_lstm", {"cpu": _CPU_WINDOW}),
    ("cpu_mem_forecast_xgb", {"cpu": _CPU_WINDOW, "mem": _MEM_WINDOW}),
    ("cpu_mem_forecast_lstm", {"cpu": _CPU_WINDOW, "mem": _MEM_WINDOW}),
]


def _ensure_trained(task: str, window: str = "5m") -> None:
    """Train the task once if it hasn't been trained yet — the stress
    tests are runnable standalone, not just after ``make e2e``."""
    r = httpx.post(
        f"{INTELLIGENCE_URL}/tasks/{task}/predict",
        json={"input_series": {"cpu": _CPU_WINDOW, "mem": _MEM_WINDOW}},
        timeout=10.0,
    )
    if r.status_code == 200:
        return
    # 503 means no trained model; train and retry once.
    if r.status_code == 503:
        train = httpx.post(
            f"{INTELLIGENCE_URL}/tasks/{task}/train",
            json={"data_source": {"kind": "prometheus", "window": window, "step": "2s"}},
            timeout=300.0,
        )
        assert train.status_code == 200, f"train {task}: {train.status_code} {train.text}"


@pytest.mark.parametrize("task,input_series", _PREDICT_TASKS, ids=[t[0] for t in _PREDICT_TASKS])
def test_predict_loop_holds_latency(task: str, input_series: dict[str, list[float]]) -> None:
    """Predict in a tight loop; assert all return 200 and p95 stays
    within the configured budget. Catches caching / GC regressions
    that only show up under load.
    """
    _ensure_trained(task)

    latencies_ms: list[float] = []
    with httpx.Client(timeout=10.0) as client:
        for _ in range(PREDICT_LOOPS):
            t0 = time.perf_counter()
            r = client.post(
                f"{INTELLIGENCE_URL}/tasks/{task}/predict",
                json={"input_series": input_series},
            )
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            assert r.status_code == 200, f"{task}: {r.status_code} {r.text}"

    p50 = statistics.median(latencies_ms)
    p95 = statistics.quantiles(latencies_ms, n=20)[18]  # 95th percentile
    p99 = statistics.quantiles(latencies_ms, n=100)[98]
    print(f"\n{task}: n={PREDICT_LOOPS} p50={p50:.1f}ms p95={p95:.1f}ms p99={p99:.1f}ms")
    assert p95 <= P95_BUDGET_MS, f"{task}: p95 {p95:.1f}ms > budget {P95_BUDGET_MS}ms"


# ---- Long-window training ------------------------------------------------


_LONG_WINDOW_TASKS = [
    "cpu_forecast_arima",
    "cpu_forecast_xgb",
    "cpu_forecast_lstm",
    "cpu_mem_forecast_xgb",
]


@pytest.mark.parametrize("task", _LONG_WINDOW_TASKS)
def test_long_window_training_completes(task: str) -> None:
    """Train against ``STRESS_LONG_WINDOW`` of Prometheus history (default
    6h). Exercises the join + scale path on a meaningful data volume and
    surfaces any memory regressions in the loader. Requires the stack to
    have been up long enough — fails clearly on insufficient retention.
    """
    r = httpx.post(
        f"{INTELLIGENCE_URL}/tasks/{task}/train",
        json={"data_source": {"kind": "prometheus", "window": LONG_WINDOW, "step": "30s"}},
        timeout=600.0,
    )
    if r.status_code != 200:
        # min_points-style errors are actionable: the stack hasn't been
        # up long enough. Surface the body so the operator knows.
        pytest.fail(
            f"long-window train {task!r} with window={LONG_WINDOW!r} returned "
            f"{r.status_code}: {r.text}. If this says 'usable point(s)', let "
            f"the stack warm up further before re-running."
        )
    assert "model_tag" in r.json()


# ---- Drift over real time ------------------------------------------------


def test_drift_over_real_time_remains_healthy() -> None:
    """Train cpu_mem_drift once, then repeatedly /predict against the
    same chunk_size of recent observations. The host's actual cpu+mem
    load drifts naturally over the loop — we don't assert on the drift
    outcome, only that the service stays healthy across many predicts
    and that NannyML's calculator doesn't degrade with sustained use.
    """
    _ensure_trained("cpu_mem_drift", window="5m")

    # Build a deterministic chunk; the realism comes from the loop length,
    # not the input values.
    chunk = {"cpu": [0.3] * 12, "mem": [0.6] * 12}

    drift_responses: list[bool] = []
    failures: list[tuple[int, int, str]] = []
    with httpx.Client(timeout=10.0) as client:
        for i in range(DRIFT_LOOPS):
            r = client.post(
                f"{INTELLIGENCE_URL}/tasks/cpu_mem_drift/predict",
                json={"input_series": chunk},
            )
            if r.status_code != 200:
                failures.append((i, r.status_code, r.text))
                continue
            drift_responses.append(bool(r.json()["prediction"]["drift_detected"]))
            time.sleep(DRIFT_INTERVAL_S)

    assert not failures, f"{len(failures)}/{DRIFT_LOOPS} drift predicts failed: {failures[:3]}"
    print(
        f"\ndrift over {DRIFT_LOOPS} iterations: {sum(drift_responses)}/{len(drift_responses)} "
        f"flagged drift"
    )

    # /readyz must still return healthy after the loop.
    health = httpx.get(f"{INTELLIGENCE_URL}/readyz", timeout=5.0)
    assert health.status_code == 200, health.text
    assert health.json()["status"] == "ready"
