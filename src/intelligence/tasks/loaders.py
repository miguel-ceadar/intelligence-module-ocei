"""Static data loaders — compose a ``TelemetrySource`` with a ``prepare``
callable to produce the components dict that ``Model.train``
expects.

Two seams in one class:
  - **source**: where data comes from (``StaticSource``,
    ``PrometheusSource``, ...). Returns a raw DataFrame.
  - **prepare**: how the DataFrame becomes training components
    (split, scale, window). The default is univariate; multivariate
    tasks pass their own.

Public factories keep their pre-phase-2 signatures so existing task
factories in ``catalog.py`` don't churn.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from intelligence.api.schemas import PrometheusDataSource, StaticDataSource
from intelligence.telemetry import PrometheusSource, StaticSource, TelemetrySource

if TYPE_CHECKING:
    from intelligence.config.settings import IntelligenceConfig


class StaticCsvLoader:
    """Compose a ``TelemetrySource`` with a ``prepare`` callable.

    Args:
        source: where data comes from. Defaults to a ``StaticSource``
            rooted at ``base_dir`` (or the bundled samples dir if
            neither is given).
        prepare: ``DataFrame -> components dict``. Defaults to the
            univariate split + MinMax-scaler used by the legacy ARIMA /
            XGB paths.
        value_col: which column the default prepare picks. Ignored when
            a custom ``prepare`` is supplied.
        base_dir: convenience for constructing a default ``StaticSource``.
            Ignored when ``source`` is supplied.
    """

    def __init__(
        self,
        source: TelemetrySource | None = None,
        prepare: Callable[[pd.DataFrame], dict] | None = None,
        value_col: str | None = None,
        base_dir: Path | None = None,
    ) -> None:
        self.source = source if source is not None else StaticSource(base_dir=base_dir)
        self.prepare = prepare if prepare is not None else _make_univariate_prepare(value_col)

    def __call__(self, descriptor: StaticDataSource) -> dict:
        if not isinstance(descriptor, StaticDataSource):
            raise ValueError(
                f"StaticCsvLoader expects StaticDataSource, got {type(descriptor).__name__}"
            )
        df = self.source.fetch_range(descriptor.name)
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
) -> StaticCsvLoader:
    """Build a ``StaticCsvLoader``.

    Pass ``prepare`` to swap the default univariate split+scale logic
    (e.g. ``make_xgb_prepare(look_back=6)`` or ``make_lstm_prepare(...)``).
    """
    return StaticCsvLoader(value_col=value_col, base_dir=base_dir, prepare=prepare)


def _make_univariate_prepare(
    value_col: str | None,
) -> Callable[[pd.DataFrame], dict]:
    """Default prepare: pick a numeric column, 80/20 split, MinMax-scale."""

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
    the window and step (and could be extended later with overrides like
    ``end_offset``).

    Args:
        source: Prometheus-shaped source. Required (no useful default).
        query: PromQL expression evaluated at fetch time.
        prepare: ``DataFrame -> components dict``. Default is the
            univariate split + scaler used for ARIMA / XGB.
        value_col: name of the value column for the default prepare.
            For single-series PromQL results, the source returns
            ``["timestamp", "value"]`` — ``"value"`` is the default.
    """

    def __init__(
        self,
        source: TelemetrySource,
        query: str,
        prepare: Callable[[pd.DataFrame], dict] | None = None,
        value_col: str | None = None,
    ) -> None:
        self.source = source
        self.query = query
        self.prepare = (
            prepare if prepare is not None else _make_univariate_prepare(value_col or "value")
        )

    def __call__(self, descriptor: PrometheusDataSource) -> dict:
        if not isinstance(descriptor, PrometheusDataSource):
            raise ValueError(
                f"PrometheusLoader expects PrometheusDataSource, "
                f"got {type(descriptor).__name__}"
            )
        end = datetime.now(UTC)
        start = end - _parse_duration(descriptor.window)
        step = _parse_duration(descriptor.step)
        df = self.source.fetch_range(self.query, start=start, end=end, step=step)
        return self.prepare(df)

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
) -> PrometheusLoader:
    """Build a ``PrometheusLoader``. Thin wrapper for symmetry with
    ``static_csv_loader``."""
    return PrometheusLoader(source=source, query=query, prepare=prepare, value_col=value_col)


def build_loader_for_task(
    cfg: "IntelligenceConfig",
    task_name: str,
    value_col: str | None = None,
    prepare: Callable[[pd.DataFrame], dict] | None = None,
) -> StaticCsvLoader | PrometheusLoader:
    """Pick the right loader for ``task_name`` based on ``cfg.telemetry``.

    Every task factory in ``catalog.py`` goes through this — operators
    flip ``telemetry.source`` to switch the whole deployment between
    CSV-backed (dev/test) and PromQL-backed (prod).

    For ``source == "prometheus"``: ``cfg.telemetry.prometheus.queries``
    must contain a PromQL expression keyed by ``task_name``. A missing
    query is a config error and raised here so the failure surfaces at
    registry build (i.e. service startup), not at first request.
    """
    if cfg.telemetry.source == "prometheus":
        prom = cfg.telemetry.prometheus
        if prom is None:  # already enforced by TelemetryConfig validator, defensive
            raise ValueError(
                "telemetry.source='prometheus' requires telemetry.prometheus block"
            )
        query = prom.queries.get(task_name)
        if query is None:
            raise ValueError(
                f"telemetry.prometheus.queries[{task_name!r}] is not set; "
                "add a PromQL expression for this task to your config"
            )
        source = PrometheusSource(
            endpoint=prom.endpoint,
            token_env=prom.token_env,
            token_file=prom.token_file,
            tls_skip_verify=prom.tls_skip_verify,
            timeout=prom.timeout,
        )
        return prometheus_loader(source=source, query=query, value_col=value_col, prepare=prepare)

    return static_csv_loader(value_col=value_col, prepare=prepare)


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
