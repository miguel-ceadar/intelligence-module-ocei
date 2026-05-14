"""Unit tests for ``PrometheusSource``.

HTTP is mocked at the module level (``intelligence.telemetry.
prometheus.requests``) so these tests don't touch the network. They pin:
  - PromQL ``query_range`` response parsing (single + multi-series).
  - Bearer-token auth from env var and from file.
  - TLS skip-verify is forwarded to ``requests.get``.
  - Error paths: HTTP errors, ``status: error`` payloads.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _matrix_response(series: list[dict]) -> dict:
    return {
        "status": "success",
        "data": {"resultType": "matrix", "result": series},
    }


def _mock_get(json_payload, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_payload
    resp.raise_for_status.return_value = None
    return resp


@patch("intelligence.telemetry.prometheus.requests.get")
def test_fetch_range_parses_single_series(mock_get):
    from intelligence.telemetry import PrometheusSource

    mock_get.return_value = _mock_get(
        _matrix_response(
            [
                {
                    "metric": {"__name__": "node_cpu"},
                    "values": [[1700000000, "0.42"], [1700000060, "0.43"]],
                }
            ]
        )
    )

    src = PrometheusSource(endpoint="http://prom:9090")
    df = src.fetch_range(
        "node_cpu",
        start=datetime(2023, 11, 14, tzinfo=UTC),
        end=datetime(2023, 11, 15, tzinfo=UTC),
        step=timedelta(minutes=1),
    )

    assert list(df.columns) == ["timestamp", "value"]
    assert len(df) == 2
    assert df["value"].tolist() == [0.42, 0.43]
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


@patch("intelligence.telemetry.prometheus.requests.get")
def test_fetch_range_merges_multi_series_on_timestamp(mock_get):
    from intelligence.telemetry import PrometheusSource

    mock_get.return_value = _mock_get(
        _matrix_response(
            [
                {"metric": {"id": "a"}, "values": [[1700000000, "1.0"], [1700000060, "1.1"]]},
                {"metric": {"id": "b"}, "values": [[1700000000, "2.0"], [1700000060, "2.1"]]},
            ]
        )
    )

    src = PrometheusSource(endpoint="http://prom:9090")
    df = src.fetch_range(
        "irrelevant",
        start=datetime(2023, 11, 14, tzinfo=UTC),
        end=datetime(2023, 11, 15, tzinfo=UTC),
        step=timedelta(minutes=1),
    )

    assert "timestamp" in df.columns
    value_cols = [c for c in df.columns if c != "timestamp"]
    assert len(value_cols) == 2
    assert len(df) == 2


@patch("intelligence.telemetry.prometheus.requests.get")
def test_bearer_token_from_env_is_forwarded(mock_get, monkeypatch):
    from intelligence.telemetry import PrometheusSource

    monkeypatch.setenv("PROMTOKEN", "abc123")
    mock_get.return_value = _mock_get(_matrix_response([]))

    src = PrometheusSource(endpoint="http://prom:9090", token_env="PROMTOKEN")
    src.fetch_range(
        "up",
        start=datetime(2023, 11, 14, tzinfo=UTC),
        end=datetime(2023, 11, 15, tzinfo=UTC),
        step=timedelta(minutes=1),
    )

    _args, kwargs = mock_get.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer abc123"


@patch("intelligence.telemetry.prometheus.requests.get")
def test_bearer_token_from_file_is_forwarded(mock_get, tmp_path):
    from intelligence.telemetry import PrometheusSource

    token_file = tmp_path / "token"
    token_file.write_text("fromfile\n")
    mock_get.return_value = _mock_get(_matrix_response([]))

    src = PrometheusSource(endpoint="http://prom:9090", token_file=str(token_file))
    src.fetch_range(
        "up",
        start=datetime(2023, 11, 14, tzinfo=UTC),
        end=datetime(2023, 11, 15, tzinfo=UTC),
        step=timedelta(minutes=1),
    )

    _args, kwargs = mock_get.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer fromfile"


def test_token_file_missing_fails_at_construction(tmp_path):
    from intelligence.telemetry import PrometheusSource

    missing = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        PrometheusSource(endpoint="http://prom:9090", token_file=str(missing))


@patch("intelligence.telemetry.prometheus.requests.get")
def test_tls_skip_verify_is_forwarded(mock_get):
    from intelligence.telemetry import PrometheusSource

    mock_get.return_value = _mock_get(_matrix_response([]))
    src = PrometheusSource(endpoint="https://prom:9090", tls_skip_verify=True)
    src.fetch_range(
        "up",
        start=datetime(2023, 11, 14, tzinfo=UTC),
        end=datetime(2023, 11, 15, tzinfo=UTC),
        step=timedelta(minutes=1),
    )
    _args, kwargs = mock_get.call_args
    assert kwargs["verify"] is False


@patch("intelligence.telemetry.prometheus.requests.get")
def test_error_status_payload_raises(mock_get):
    from intelligence.telemetry import PrometheusSource

    mock_get.return_value = _mock_get(
        {
            "status": "error",
            "errorType": "bad_data",
            "error": "syntax error in query",
        }
    )

    src = PrometheusSource(endpoint="http://prom:9090")
    with pytest.raises(RuntimeError, match="syntax error in query"):
        src.fetch_range(
            "bad query",
            start=datetime(2023, 11, 14, tzinfo=UTC),
            end=datetime(2023, 11, 15, tzinfo=UTC),
            step=timedelta(minutes=1),
        )


@patch("intelligence.telemetry.prometheus.requests.get")
def test_is_ready_probes_healthy_endpoint(mock_get):
    from intelligence.telemetry import PrometheusSource

    mock_get.return_value = _mock_get({}, status_code=200)
    src = PrometheusSource(endpoint="http://prom:9090")
    ok, _msg = src.is_ready()
    assert ok is True
    called_url = mock_get.call_args.args[0]
    assert called_url.endswith("/-/healthy")


@patch("intelligence.telemetry.prometheus.requests.get")
def test_is_ready_returns_failure_on_non_200(mock_get):
    from intelligence.telemetry import PrometheusSource

    mock_get.return_value = _mock_get({}, status_code=503)
    src = PrometheusSource(endpoint="http://prom:9090")
    ok, msg = src.is_ready()
    assert ok is False
    assert "503" in msg


def test_satisfies_telemetry_source_protocol():
    from intelligence.telemetry import PrometheusSource, TelemetrySource

    assert isinstance(PrometheusSource(endpoint="http://x"), TelemetrySource)
