"""Data loaders that compose a ``TelemetrySource`` with a ``prepare``
callable to produce the components dict each ``Model.fit`` expects.

Two seams:
  - **source**: where data comes from (``StaticSource``,
    ``PrometheusSource``, …). Returns a raw DataFrame.
  - **prepare**: how the DataFrame becomes training components
    (split, scale, window). The default is univariate; multivariate
    tasks pass their own.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from intelligence.api.schemas import PrometheusDataSource, StaticDataSource
from intelligence.telemetry import PrometheusSource, StaticSource, TelemetrySource

if TYPE_CHECKING:
    from intelligence.config.settings import IntelligenceConfig


# Hostnames we always refuse regardless of DNS — these are explicit
# operator footguns. DNS-resolution-based filtering is intentionally not
# done here: an authed /train POST runs in a sync handler on a thread,
# but resolving every override would add a blocking call per request and
# is still defeatable with rebinding. Cluster-level egress controls
# (NetworkPolicy, mesh sidecar) are the real defense; this gate catches
# honest mistakes.
_FORBIDDEN_HOSTNAMES = frozenset({"localhost", "localhost.localdomain"})


def _validate_override_endpoint(url: str) -> None:
    """Refuse override URLs that would let an authed /train POST probe
    loopback / metadata / private / link-local addresses.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(
            f"endpoint override must use https://, got scheme {parsed.scheme!r} in {url!r}"
        )
    host = parsed.hostname
    if not host:
        raise ValueError(f"endpoint override has no host: {url!r}")
    if host.lower() in _FORBIDDEN_HOSTNAMES:
        raise ValueError(f"endpoint override host {host!r} is loopback — rejected")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        # Hostname literal, not an IP. Accept — see module note above.
        return
    if (
        ip.is_loopback
        or ip.is_link_local  # includes 169.254.169.254 cloud metadata
        or ip.is_private  # RFC1918 + IPv6 unique-local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        raise ValueError(
            f"endpoint override IP {ip} is loopback/private/link-local/metadata — rejected"
        )


class StaticCsvLoader:
    """Compose a ``TelemetrySource`` with a ``prepare`` callable.

    Args:
        source: where data comes from. Defaults to a ``StaticSource``
            rooted at ``base_dir`` (or the bundled samples dir if
            neither is given).
        prepare: ``DataFrame -> components dict``. Defaults to a
            univariate split + MinMax-scaler used by the ARIMA /
            XGB paths.
        value_col: which column the default prepare picks. Ignored when
            a custom ``prepare`` is supplied.
        base_dir: convenience for constructing a default ``StaticSource``.
            Ignored when ``source`` is supplied.
        min_points: minimum row count after NaN drop. Below this the
            loader raises with a clear message rather than letting a
            tiny dataset crash the scaler or the supervised reshape.
    """

    def __init__(
        self,
        source: TelemetrySource | None = None,
        prepare: Callable[[pd.DataFrame], dict] | None = None,
        value_col: str | None = None,
        base_dir: Path | None = None,
        min_points: int = 30,
    ) -> None:
        self.source = source if source is not None else StaticSource(base_dir=base_dir)
        self.value_col = value_col
        self.prepare = prepare if prepare is not None else _make_univariate_prepare(value_col)
        self.min_points = int(min_points)

    def __call__(self, descriptor: StaticDataSource) -> dict:
        if not isinstance(descriptor, StaticDataSource):
            raise ValueError(
                f"StaticCsvLoader expects StaticDataSource, got {type(descriptor).__name__}"
            )
        df = self.source.fetch_range(descriptor.name)
        if self.value_col is not None:
            df = _select_value_column(df, self.value_col)
        if len(df) < self.min_points:
            raise ValueError(
                f"static dataset {descriptor.name!r} has {len(df)} usable "
                f"row(s) — need at least {self.min_points}."
            )
        return self.prepare(df)

    def is_ready(self) -> tuple[bool, str]:
        probe = getattr(self.source, "is_ready", None)
        if probe is None:
            return True, "ok"
        return probe()


def static_csv_loader(
    value_col: str | None = None,
    base_dir: Path | None = None,
    prepare: Callable[[pd.DataFrame], dict] | None = None,
    min_points: int = 30,
) -> StaticCsvLoader:
    """Build a ``StaticCsvLoader``.

    Pass ``prepare`` to swap the default univariate split+scale logic
    (e.g. ``make_xgb_prepare(look_back=6)`` or ``make_lstm_prepare(...)``).
    """
    return StaticCsvLoader(
        value_col=value_col, base_dir=base_dir, prepare=prepare, min_points=min_points
    )


def _make_univariate_prepare(
    value_col: str | None,
) -> Callable[[pd.DataFrame], dict]:
    """Default prepare: pick a numeric column, 80/20 split, MinMax-scale.

    The timestamp column (if any) is ignored — the trainer indexes by
    row position, not by time. Sparse or irregularly-spaced inputs will
    silently train as if regularly-spaced; validate upstream if that
    matters for your metric.
    """

    def prepare(df: pd.DataFrame) -> dict:
        col = value_col or _autodetect_value_column(df)
        series = df[col].astype(float).values.reshape(-1, 1)
        split = int(len(series) * 0.8)
        scaler = MinMaxScaler().fit(series[:split])
        return {
            "X_train": scaler.transform(series[:split]),
            "X_test": scaler.transform(series[split:]),
            "y_train": series[:split].ravel(),
            "y_test": series[split:].ravel(),
            "scaler_obj": scaler,
        }

    return prepare


class PrometheusLoader:
    """Compose a ``PrometheusSource`` (or other windowed source) with a
    ``prepare`` callable.

    The PromQL query is fixed at task-registration time — operators don't
    pick arbitrary PromQL on every request. The descriptor only supplies
    the window and step (and optionally an endpoint override when the
    deployment opts in via ``allow_endpoint_override``).

    NaN-valued stale markers and ±Inf points are stripped before the
    prepare runs (sklearn scalers reject them opaquely). Prometheus
    *missing* points are sparse — timestamps with no data are simply
    omitted, not NaN'd — and downstream prepares index by row position,
    so heavily-sparse windows silently train on irregularly-spaced data;
    pick a window with consistent metric presence.

    Args:
        source: Prometheus-shaped source. Required (no useful default).
        query: PromQL expression evaluated at fetch time.
        prepare: ``DataFrame -> components dict``. Default is the
            univariate split + scaler used for ARIMA / XGB.
        value_col: name of the value column for the default prepare.
            For single-series PromQL results, the source returns
            ``["timestamp", "value"]`` — ``"value"`` is the default.
        allow_endpoint_override: whether ``data_source.endpoint`` on the
            request is honoured. Off by default (SSRF defense).
        source_kwargs: how to rebuild a ``PrometheusSource`` for the
            override path — configured auth + TLS carry through. ``None``
            means override is not supported.
        min_points: minimum usable rows after NaN/Inf drop. Below this
            the loader raises with a clear message rather than letting
            a tiny window crash the scaler or supervised reshape.
    """

    def __init__(
        self,
        source: TelemetrySource,
        query: str,
        prepare: Callable[[pd.DataFrame], dict] | None = None,
        value_col: str | None = None,
        allow_endpoint_override: bool = False,
        source_kwargs: dict | None = None,
        min_points: int = 30,
    ) -> None:
        self.source = source
        self.query = query
        self.value_col = value_col
        self.prepare = (
            prepare if prepare is not None else _make_univariate_prepare(value_col or "value")
        )
        self.allow_endpoint_override = allow_endpoint_override
        self._source_kwargs = source_kwargs or {}
        self.min_points = int(min_points)

    def __call__(self, descriptor: PrometheusDataSource) -> dict:
        if not isinstance(descriptor, PrometheusDataSource):
            raise ValueError(
                f"PrometheusLoader expects PrometheusDataSource, got {type(descriptor).__name__}"
            )

        source = self._resolve_source(descriptor)
        end = datetime.now(UTC)
        start = end - _parse_duration(descriptor.window)
        step = _parse_duration(descriptor.step)
        df = source.fetch_range(self.query, start=start, end=end, step=step)
        if df.empty:
            # The most common operator mistake is a query that matches no
            # series over the requested window. Surface this before it
            # turns into an opaque numpy IndexError inside the trainer.
            raise ValueError(
                f"PromQL returned no data — query {self.query!r} matched "
                f"zero series over the {descriptor.window} window. Check "
                f"the query against the live Prometheus and that the "
                f"window covers a period where data exists."
            )
        value_cols = [c for c in df.columns if c.lower() != "timestamp"]
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=value_cols)
        if len(df) < self.min_points:
            raise ValueError(
                f"PromQL returned {len(df)} usable point(s) after stripping "
                f"NaN/Inf — need at least {self.min_points}. Widen the "
                f"window, shrink the step, or check that the metric isn't "
                f"sparse over the requested range. Query: {self.query!r}"
            )
        # Single-series PromQL responses always come back with a literal
        # "value" column. Rename to the task's feature name so downstream
        # prepares — and any column-name round-trip into a Bento's stored
        # contract, e.g. drift's column_names — match the task's InputSpec.
        if self.value_col is not None and "value" in df.columns:
            df = df.rename(columns={"value": self.value_col})
        return self.prepare(df)

    def _resolve_source(self, descriptor: PrometheusDataSource) -> TelemetrySource:
        """Pick the source to query: configured by default, override if
        the request set one *and* the deployment opted in."""
        override = getattr(descriptor, "endpoint", None)
        if override is None:
            return self.source
        if not self.allow_endpoint_override:
            raise ValueError(
                "data_source.endpoint override sent, but this deployment "
                "has telemetry.allow_endpoint_override=false. Flip it on "
                "in the service config to enable per-request overrides "
                "(SSRF: only do this on trusted clients)."
            )
        if not self._source_kwargs:
            # Configured source wasn't built via the factory (e.g. injected
            # in a unit test) — can't honour the override without knowing
            # how to construct a fresh source.
            raise ValueError(
                "endpoint override requested but no source kwargs are "
                "available to rebuild a PrometheusSource"
            )
        _validate_override_endpoint(override)
        return PrometheusSource(**{**self._source_kwargs, "endpoint": override})

    def is_ready(self) -> tuple[bool, str]:
        probe = getattr(self.source, "is_ready", None)
        if probe is None:
            return True, "ok"
        return probe()


def prometheus_loader(
    source: TelemetrySource,
    query: str,
    prepare: Callable[[pd.DataFrame], dict] | None = None,
    value_col: str | None = None,
    allow_endpoint_override: bool = False,
    source_kwargs: dict | None = None,
    min_points: int = 30,
) -> PrometheusLoader:
    """Build a ``PrometheusLoader``. Thin wrapper for symmetry with
    ``static_csv_loader``."""
    return PrometheusLoader(
        source=source,
        query=query,
        prepare=prepare,
        value_col=value_col,
        allow_endpoint_override=allow_endpoint_override,
        source_kwargs=source_kwargs,
        min_points=min_points,
    )


def build_loader_for_task(
    cfg: IntelligenceConfig,
    task_name: str,
    value_col: str | None = None,
    prepare: Callable[[pd.DataFrame], dict] | None = None,
    query: str | None = None,
    min_points: int = 30,
) -> StaticCsvLoader | PrometheusLoader:
    """Pick the right loader for ``task_name`` based on ``cfg.telemetry``.

    Operators flip ``telemetry.source`` to switch the whole deployment
    between CSV-backed (dev/test) and PromQL-backed (prod).

    For ``source == "prometheus"`` the per-task PromQL ``query`` is
    required. The per-kind builders pass it through from each task's
    own ``query:`` config field; missing it is a startup error so a
    misconfigured task fails loudly at registry build time, not at
    first request.
    """
    if cfg.telemetry.source == "prometheus":
        prom = cfg.telemetry.prometheus
        if prom is None:  # already enforced by TelemetryConfig validator, defensive
            raise ValueError("telemetry.source='prometheus' requires telemetry.prometheus block")
        if query is None:
            raise ValueError(
                f"no PromQL query for task {task_name!r}; set `query:` on the task config block"
            )
        source_kwargs = {
            "endpoint": prom.endpoint,
            "token_env": prom.token_env,
            "token_file": prom.token_file,
            "tls_skip_verify": prom.tls_skip_verify,
            "timeout": prom.timeout,
        }
        source = PrometheusSource(**source_kwargs)
        return prometheus_loader(
            source=source,
            query=query,
            value_col=value_col,
            prepare=prepare,
            allow_endpoint_override=cfg.telemetry.allow_endpoint_override,
            source_kwargs=source_kwargs,
            min_points=min_points,
        )

    return static_csv_loader(value_col=value_col, prepare=prepare, min_points=min_points)


# Duration parsing: matches the subset of PromQL durations we actually
# accept in train requests. Years/months excluded — too imprecise for
# the windows we use here and pandas can't represent them as timedelta.
_DURATION_UNITS = {
    "s": "seconds",
    "m": "minutes",
    "h": "hours",
    "d": "days",
    "w": "weeks",
}
_DURATION_RE = re.compile(r"(\d+)([smhdw])")


def _parse_duration(spec: str) -> timedelta:
    """Parse a PromQL duration: ``30s``, ``5m``, ``2h``, ``1d``, ``1w``."""
    match = _DURATION_RE.fullmatch(spec.strip())
    if not match:
        raise ValueError(
            f"invalid duration: {spec!r} (expected like '30s', '5m', '1h', '1d', '1w')"
        )
    n, unit = int(match.group(1)), match.group(2)
    return timedelta(**{_DURATION_UNITS[unit]: n})


def _autodetect_value_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if col.lower() in {"time", "timestamp", "date"}:
            continue
        try:
            pd.to_numeric(df[col])
            return col
        except (ValueError, TypeError):
            continue
    raise ValueError(f"no numeric column found in dataset; columns={list(df.columns)}")


_TIMESTAMP_COLS = {"time", "timestamp", "date"}


def _select_value_column(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Normalise a CSV-shaped DataFrame to a single value column named
    ``value_col``, mirroring what ``PrometheusLoader`` does for single-series
    PromQL responses.

    Univariate input → rename the lone value column.
    Multivariate input → pick the column whose name matches ``value_col``
    (case-insensitive) and drop the rest, keeping any timestamp column.
    Pass-through if it already matches.
    """
    value_cols = [c for c in df.columns if c.lower() not in _TIMESTAMP_COLS]
    if not value_cols:
        return df

    if len(value_cols) == 1:
        if value_cols[0] != value_col:
            df = df.rename(columns={value_cols[0]: value_col})
        return df

    # Multivariate — pick the matching column. Case-insensitive so CSVs
    # with header "CPU" or "MEM" line up with kebab-flavored YAML keys.
    match = next((c for c in value_cols if c.lower() == value_col.lower()), None)
    if match is None:
        return df  # let downstream prepare decide (it may want all columns)
    ts_col = next((c for c in df.columns if c.lower() in _TIMESTAMP_COLS), None)
    keep = [ts_col, match] if ts_col else [match]
    df = df[keep]
    if match != value_col:
        df = df.rename(columns={match: value_col})
    return df
