"""Data loaders that compose a ``TelemetrySource`` with a ``prepare``
callable to produce the components dict each ``Model.fit`` expects.

Two seams:
  - **source**: where data comes from (``StaticSource``,
    ``PrometheusSource``, …). Returns a raw DataFrame.
  - **prepare**: how the DataFrame becomes training components
    (split, scale, window). The default is univariate; multivariate
    tasks pass their own.

A task's features are expressed as parallel ``value_cols`` /
``queries`` lists at the loader API. The first entry is the target
(what gets forecast); the rest are exogenous covariates. Loaders
are otherwise model-agnostic — per-kind shape decisions live in the
``prepare`` callable each builder supplies.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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
        value_cols: which columns the default prepare picks. The first
            entry is the target. ``None`` triggers autodetection of a
            single numeric column. Ignored when a custom ``prepare``
            is supplied.
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
        value_cols: list[str] | None = None,
        base_dir: Path | None = None,
        min_points: int = 30,
    ) -> None:
        self.source = source if source is not None else StaticSource(base_dir=base_dir)
        self.value_cols = list(value_cols) if value_cols else None
        target = self.value_cols[0] if self.value_cols else None
        self.prepare = prepare if prepare is not None else _make_univariate_prepare(target)
        self.min_points = int(min_points)

    def __call__(self, descriptor: StaticDataSource) -> dict:
        if not isinstance(descriptor, StaticDataSource):
            raise ValueError(
                f"StaticCsvLoader expects StaticDataSource, got {type(descriptor).__name__}"
            )
        df = self.source.fetch_range(descriptor.name)
        if self.value_cols:
            df = _select_value_columns(df, self.value_cols)
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
    value_cols: list[str] | None = None,
    base_dir: Path | None = None,
    prepare: Callable[[pd.DataFrame], dict] | None = None,
    min_points: int = 30,
) -> StaticCsvLoader:
    """Build a ``StaticCsvLoader``.

    Pass ``prepare`` to swap the default univariate split+scale logic
    (e.g. ``make_xgb_prepare(look_back=6)`` or ``make_lstm_prepare(...)``).
    """
    return StaticCsvLoader(
        value_cols=value_cols, base_dir=base_dir, prepare=prepare, min_points=min_points
    )


def _make_univariate_prepare(
    value_col: str | None,
) -> Callable[[pd.DataFrame], dict]:
    """Default prepare: pick a numeric column, 80/20 split, MinMax-scale.

    The timestamp column (if any) is ignored — the trainer indexes by
    row position, not by time. Sparse or irregularly-spaced inputs will
    silently train as if regularly-spaced; validate upstream if that
    matters for your metric.

    For multivariate tasks, the target (first feature) is consumed
    here; covariates load into the DataFrame but the default prepare
    ignores them. Kinds that use covariates (xgb, lstm) ship their
    own prepare.
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

    ``queries`` and ``value_cols`` are parallel lists — entry ``i``
    fetches its data with ``queries[i]`` and the resulting ``value``
    column is renamed to ``value_cols[i]``. The first feature is the
    target; the rest are covariates. Multi-feature tasks run their
    queries in parallel (small thread pool — IO-bound) and join the
    results on timestamp via ``merge_asof``.

    The PromQL query for each feature is fixed at task-registration
    time — operators don't pick arbitrary PromQL on every request. The
    descriptor only supplies the window and step (and optionally an
    endpoint override when the deployment opts in via
    ``allow_endpoint_override``).

    NaN-valued stale markers and ±Inf points are stripped before the
    prepare runs (sklearn scalers reject them opaquely). Prometheus
    *missing* points are sparse — timestamps with no data are simply
    omitted, not NaN'd — and downstream prepares index by row position,
    so heavily-sparse windows silently train on irregularly-spaced data;
    pick a window with consistent metric presence.

    Args:
        source: Prometheus-shaped source. Required (no useful default).
        queries: one PromQL expression per feature. Length ≥ 1.
        prepare: ``DataFrame -> components dict``. Default is the
            univariate split + scaler used for ARIMA / XGB.
        value_cols: column names after the per-query ``value`` rename.
            Must pair 1:1 with ``queries``. ``None`` is allowed only
            for legacy single-query tests that don't care about the
            column name.
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
        queries: list[str],
        prepare: Callable[[pd.DataFrame], dict] | None = None,
        value_cols: list[str] | None = None,
        allow_endpoint_override: bool = False,
        source_kwargs: dict | None = None,
        min_points: int = 30,
    ) -> None:
        queries = list(queries)
        if not queries:
            raise ValueError("PrometheusLoader requires at least one query")
        if value_cols is not None and len(value_cols) != len(queries):
            raise ValueError(
                f"value_cols ({len(value_cols)}) must pair 1:1 with queries ({len(queries)})"
            )
        self.source = source
        self.queries = queries
        self.value_cols = list(value_cols) if value_cols is not None else None
        target = (self.value_cols[0] if self.value_cols else None) or "value"
        self.prepare = prepare if prepare is not None else _make_univariate_prepare(target)
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
        df = self._fetch_all(source, start=start, end=end, step=step)

        value_cols = [c for c in df.columns if c.lower() != "timestamp"]
        # NaN/Inf in any feature drops the row. Per-feature imputation
        # would let one sparse covariate poison its row alone; for now
        # the conservative policy is uniform.
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=value_cols)
        if len(df) < self.min_points:
            queries_str = ", ".join(repr(q) for q in self.queries)
            raise ValueError(
                f"PromQL returned {len(df)} usable point(s) after stripping "
                f"NaN/Inf — need at least {self.min_points}. Widen the "
                f"window, shrink the step, or check that the metric isn't "
                f"sparse over the requested range. Queries: {queries_str}"
            )
        return self.prepare(df)

    def _fetch_all(
        self,
        source: TelemetrySource,
        *,
        start: datetime,
        end: datetime,
        step: timedelta,
    ) -> pd.DataFrame:
        """Run all queries; join on timestamp for N > 1.

        Single-query path skips the thread pool to keep the n=1 case
        identical in shape to the pre-multivariate code. For N > 1
        each query may land on a slightly different evaluation grid
        (different recording rules, Thanos store skew), so we join
        with ``merge_asof(direction="nearest", tolerance=step/2)`` —
        rows within half a step are treated as the same tick, rows
        further apart are dropped rather than nearest-matched across
        an arbitrary gap.
        """
        if len(self.queries) == 1:
            df = source.fetch_range(self.queries[0], start=start, end=end, step=step)
            self._check_nonempty(df, self.queries[0])
            return self._rename_value_column(df, 0)

        def fetch_i(i: int) -> pd.DataFrame:
            return source.fetch_range(self.queries[i], start=start, end=end, step=step)

        with ThreadPoolExecutor(max_workers=min(4, len(self.queries))) as pool:
            results = list(pool.map(fetch_i, range(len(self.queries))))

        renamed: list[pd.DataFrame] = []
        for i, df in enumerate(results):
            self._check_nonempty(df, self.queries[i])
            renamed.append(self._rename_value_column(df, i))

        tolerance = step / 2
        out = renamed[0].sort_values("timestamp")
        for other in renamed[1:]:
            out = pd.merge_asof(
                out,
                other.sort_values("timestamp"),
                on="timestamp",
                direction="nearest",
                tolerance=tolerance,
            )
        # ``merge_asof`` keeps the left frame's rows; rows that couldn't
        # join produce NaN on the right side. Drop those — the alternative
        # is forwarding NaN into the prepare and crashing the scaler.
        value_cols = [c for c in out.columns if c != "timestamp"]
        return out.dropna(subset=value_cols).reset_index(drop=True)

    def _check_nonempty(self, df: pd.DataFrame, query: str) -> None:
        if df.empty:
            # The most common operator mistake is a query that matches no
            # series over the requested window. Surface this before it
            # turns into an opaque numpy IndexError inside the trainer.
            raise ValueError(
                f"PromQL returned no data — query {query!r} matched "
                f"zero series. Check the query against the live "
                f"Prometheus and that the window covers a period where "
                f"data exists."
            )

    def _rename_value_column(self, df: pd.DataFrame, i: int) -> pd.DataFrame:
        """Rename the lone ``value`` column to the i-th feature name."""
        if self.value_cols is None:
            return df
        target_name = self.value_cols[i]
        if "value" in df.columns and target_name != "value":
            return df.rename(columns={"value": target_name})
        return df

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
    queries: list[str],
    prepare: Callable[[pd.DataFrame], dict] | None = None,
    value_cols: list[str] | None = None,
    allow_endpoint_override: bool = False,
    source_kwargs: dict | None = None,
    min_points: int = 30,
) -> PrometheusLoader:
    """Build a ``PrometheusLoader``. Thin wrapper for symmetry with
    ``static_csv_loader``."""
    return PrometheusLoader(
        source=source,
        queries=queries,
        prepare=prepare,
        value_cols=value_cols,
        allow_endpoint_override=allow_endpoint_override,
        source_kwargs=source_kwargs,
        min_points=min_points,
    )


def build_loader_for_task(
    cfg: IntelligenceConfig,
    task_name: str,
    value_cols: list[str] | None = None,
    prepare: Callable[[pd.DataFrame], dict] | None = None,
    queries: list[str] | None = None,
    min_points: int = 30,
) -> StaticCsvLoader | PrometheusLoader:
    """Pick the right loader for ``task_name`` based on ``cfg.telemetry``.

    Operators flip ``telemetry.source`` to switch the whole deployment
    between CSV-backed (dev/test) and PromQL-backed (prod).

    For ``source == "prometheus"`` every feature needs a non-null
    ``query`` — the per-kind builders pass them through from each
    task's ``features[*].query`` block. A missing query is a startup
    error so a misconfigured task fails loudly at registry build time,
    not at first request.
    """
    if cfg.telemetry.source == "prometheus":
        prom = cfg.telemetry.prometheus
        if prom is None:  # already enforced by TelemetryConfig validator, defensive
            raise ValueError("telemetry.source='prometheus' requires telemetry.prometheus block")
        if not queries or any(q is None for q in queries):
            raise ValueError(
                f"no PromQL query for task {task_name!r}; set `query:` on every feature"
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
            queries=queries,
            value_cols=value_cols,
            prepare=prepare,
            allow_endpoint_override=cfg.telemetry.allow_endpoint_override,
            source_kwargs=source_kwargs,
            min_points=min_points,
        )

    return static_csv_loader(value_cols=value_cols, prepare=prepare, min_points=min_points)


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
    if n <= 0:
        # Zero or negative durations make ``query_range`` return HTTP 400
        # with an opaque error; reject at the contract boundary instead.
        raise ValueError(f"duration must be positive: {spec!r}")
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


def _select_value_columns(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Normalise a multi-column DataFrame to the given value columns.

    For each entry in ``value_cols`` the loader tries — in order —
    an exact column match, then a case-insensitive match (renaming to
    the requested casing). Columns not in ``value_cols`` are dropped;
    any timestamp column is preserved.

    Any requested name that doesn't match (exact or case-insensitive)
    is a configuration error — silently passing through would let a
    typo train on whatever column the autodetect grabs first. Raise
    with both the requested and available names so the operator can
    fix the YAML.
    """
    available = list(df.columns)
    available_lower = {c.lower(): c for c in available}
    ts_col = next((c for c in available if c.lower() in _TIMESTAMP_COLS), None)

    keep: list[str] = [ts_col] if ts_col else []
    renames: dict[str, str] = {}
    missing: list[str] = []

    for name in value_cols:
        if name in available:
            keep.append(name)
            continue
        match = available_lower.get(name.lower())
        if match is not None and match not in keep:
            renames[match] = name
            keep.append(name)
            continue
        missing.append(name)

    if missing:
        raise ValueError(
            f"value_cols {missing!r} not found in dataset; available columns: {available!r}"
        )

    if renames:
        df = df.rename(columns=renames)
    return df[keep]
