"""``PrometheusSource`` — PromQL-backed ``TelemetrySource``.

Speaks the Prometheus HTTP API (``/api/v1/query_range`` for windowed
training data, ``/api/v1/query`` for live predict). Compatible with
Thanos Query — same endpoints.

Auth: pass a token via ``token_env`` (env var name) or ``token_file``
(filesystem path). The token is read at call time, not at construction,
so rotating the secret doesn't require recreating the source.
``token_file`` is checked for existence at construction so a typo'd
path fails the service at startup, not at first request.

TLS: ``tls_skip_verify=True`` disables certificate validation. Only use
inside a trusted cluster.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class PrometheusSource:
    def __init__(
        self,
        endpoint: str,
        token_env: str | None = None,
        token_file: str | None = None,
        tls_skip_verify: bool = False,
        timeout: float = 30.0,
    ) -> None:
        if token_env and token_file:
            raise ValueError("specify token_env or token_file, not both")
        if token_file and not Path(token_file).is_file():
            raise FileNotFoundError(
                f"prometheus token_file {token_file!r} does not exist or is not a regular file"
            )
        self.endpoint = endpoint.rstrip("/")
        self.token_env = token_env
        self.token_file = token_file
        self.tls_skip_verify = tls_skip_verify
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        import os

        token: str | None = None
        if self.token_env:
            token = os.environ.get(self.token_env)
        elif self.token_file:
            try:
                token = Path(self.token_file).read_text().strip()
            except OSError as e:
                raise RuntimeError(
                    f"failed to read prometheus token_file {self.token_file!r}: {e}"
                ) from e
        return {"Authorization": f"Bearer {token}"} if token else {}

    def _request(self, path: str, params: dict | None = None, *, timeout: float | None = None):
        url = f"{self.endpoint}{path}"
        return requests.get(
            url,
            params=params,
            headers=self._headers(),
            verify=not self.tls_skip_verify,
            timeout=timeout if timeout is not None else self.timeout,
        )

    def fetch_range(
        self,
        query: str,
        start: datetime | None = None,
        end: datetime | None = None,
        step: timedelta | None = None,
    ) -> pd.DataFrame:
        if start is None or end is None or step is None:
            raise ValueError("PrometheusSource.fetch_range requires start, end, and step")
        params = {
            "query": query,
            "start": start.timestamp(),
            "end": end.timestamp(),
            "step": int(step.total_seconds()),
        }
        resp = self._request("/api/v1/query_range", params=params)
        resp.raise_for_status()
        return _parse_query_response(resp.json(), expected="matrix")

    def fetch_instant(self, query: str) -> pd.DataFrame:
        resp = self._request("/api/v1/query", params={"query": query})
        resp.raise_for_status()
        return _parse_query_response(resp.json(), expected="vector")

    def is_ready(self) -> tuple[bool, str]:
        try:
            resp = self._request("/-/healthy", timeout=5.0)
        except requests.RequestException as e:
            return False, f"healthy probe failed: {e}"
        if resp.status_code == 200:
            return True, "ok"
        return False, f"healthy probe returned HTTP {resp.status_code}"


def _parse_query_response(payload: dict, *, expected: str) -> pd.DataFrame:
    if payload.get("status") != "success":
        raise RuntimeError(f"prometheus query failed: {payload.get('error', 'unknown error')}")
    data = payload.get("data", {})
    result_type = data.get("resultType")
    if result_type != expected:
        raise ValueError(f"expected {expected!r} result, got {result_type!r}")

    series = data.get("result", [])
    if not series:
        return pd.DataFrame(columns=["timestamp", "value"])

    if expected == "matrix":
        return _matrix_to_dataframe(series)
    return _vector_to_dataframe(series)


def _matrix_to_dataframe(series: list[dict]) -> pd.DataFrame:
    """Matrix = one series per metric, each with a list of [ts, val] pairs.

    Single series → ``[timestamp, value]``. Multi-series → ``[timestamp,
    value_0, value_1, ...]`` joined on timestamp.
    """
    frames: list[pd.DataFrame] = []
    for i, s in enumerate(series):
        col = "value" if len(series) == 1 else f"value_{i}"
        df = pd.DataFrame(s["values"], columns=["timestamp", col])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df[col] = df[col].astype(float)
        frames.append(df.sort_values("timestamp"))

    out = frames[0]
    for f in frames[1:]:
        out = pd.merge_asof(out, f, on="timestamp")
    return out


def _vector_to_dataframe(series: list[dict]) -> pd.DataFrame:
    """Vector = one ``[ts, val]`` per series (instantaneous)."""
    rows = []
    for i, s in enumerate(series):
        ts, val = s["value"]
        rows.append(
            {
                "series": s.get("metric", {}).get("__name__", f"s{i}"),
                "timestamp": pd.to_datetime(ts, unit="s", utc=True),
                "value": float(val),
            }
        )
    return pd.DataFrame(rows)
