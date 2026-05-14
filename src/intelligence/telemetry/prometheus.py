"""``PrometheusSource`` — PromQL-backed ``TelemetrySource``.

Speaks the Prometheus HTTP API (``/api/v1/query_range`` for windowed
training data, ``/api/v1/query`` for live predict). Compatible with
Thanos Query — same endpoints.

Auth: pass a token via ``token_env`` (env var name) or ``token_file``
(filesystem path). The token is read at call time, not at construction,
so rotating the secret doesn't require recreating the source. Both
forms are validated at construction (env var must resolve to a
non-empty value; file must exist) so a typo fails the service at
startup, not at first request.

TLS: ``tls_skip_verify=True`` disables certificate validation. Only use
inside a trusted cluster.
"""

from __future__ import annotations

import logging
import os
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
        if token_env and not os.environ.get(token_env):
            # Symmetric to the token_file check: a typo'd env var name
            # would otherwise produce an empty Authorization header at
            # first request, with no log line.
            raise ValueError(f"prometheus token_env {token_env!r} is not set in the environment")
        self.endpoint = endpoint.rstrip("/")
        self.token_env = token_env
        self.token_file = token_file
        self.tls_skip_verify = tls_skip_verify
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
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
        if start.tzinfo is None or end.tzinfo is None:
            # ``datetime.timestamp()`` on a naïve datetime applies the
            # host's local timezone, silently shifting the queried window
            # by the local UTC offset. Reject upfront.
            raise ValueError(
                "PrometheusSource.fetch_range requires tz-aware datetimes for start and end"
            )
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
    value_0, value_1, ...]`` inner-joined on timestamp.

    Per-series duplicate timestamps (Thanos with overlapping stores, or
    a recording rule replayed across an evaluation boundary) are
    collapsed via ``keep="last"`` so the join key stays unique. The
    multi-series join is an exact inner-join because all series in a
    single matrix response share the same evaluation grid by
    construction; any row missing from one series is dropped rather
    than nearest-matched (which would silently fabricate alignment).
    """
    frames: list[pd.DataFrame] = []
    for i, s in enumerate(series):
        col = "value" if len(series) == 1 else f"value_{i}"
        df = pd.DataFrame(s["values"], columns=["timestamp", col])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df[col] = df[col].astype(float)
        df = df.drop_duplicates(subset="timestamp", keep="last")
        frames.append(df.sort_values("timestamp").reset_index(drop=True))

    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="timestamp", how="inner")
    return out.reset_index(drop=True)


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
