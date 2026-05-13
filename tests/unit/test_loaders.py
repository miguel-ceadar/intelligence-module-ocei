"""Unit tests for ``StaticCsvLoader`` — verifies the new internal
``source + prepare`` composition and backwards compat of the
``static_csv_loader()`` factory.
"""

from __future__ import annotations

import pandas as pd
import pytest

from intelligence.api.schemas import StaticDataSource


def test_default_prepare_produces_legacy_components(samples_dir):
    """The default prepare callable must match what phase-1's loader produced:
    a univariate split + MinMax-scaled X_train/X_test/y_train/y_test/scaler_obj.
    """
    from intelligence.tasks.loaders import static_csv_loader

    loader = static_csv_loader(base_dir=samples_dir)
    components = loader(StaticDataSource(kind="static", name="cpu_sample_dataset_orangepi.csv"))

    assert set(components) >= {"X_train", "X_test", "y_train", "y_test", "scaler_obj"}
    assert components["X_train"].shape[1] == 1, "univariate default expected"
    assert components["X_train"].ndim == 2


def test_custom_prepare_is_invoked(samples_dir):
    """A user-supplied ``prepare`` callable receives the source's DataFrame
    and its return value is the loader's output verbatim — that's the (B)
    flavour of the seam.
    """
    from intelligence.tasks.loaders import StaticCsvLoader
    from intelligence.telemetry import StaticSource

    seen: dict = {}

    def custom_prepare(df: pd.DataFrame) -> dict:
        seen["columns"] = list(df.columns)
        seen["nrows"] = len(df)
        return {"marker": "custom", "nrows": len(df)}

    loader = StaticCsvLoader(
        source=StaticSource(base_dir=samples_dir),
        prepare=custom_prepare,
    )
    out = loader(StaticDataSource(kind="static", name="cpu_sample_dataset_orangepi.csv"))

    assert out == {"marker": "custom", "nrows": seen["nrows"]}
    assert seen["nrows"] > 0


def test_loader_rejects_non_static_descriptor(samples_dir):
    """A ``PrometheusDataSource`` should be rejected by ``StaticCsvLoader``
    with a clear error — wrong loader for the descriptor kind.
    """
    from intelligence.api.schemas import PrometheusDataSource
    from intelligence.tasks.loaders import static_csv_loader

    loader = static_csv_loader(base_dir=samples_dir)
    with pytest.raises(ValueError, match="StaticDataSource"):
        loader(PrometheusDataSource(kind="prometheus", window="24h", step="1m"))


def test_loader_is_ready_delegates_to_source(tmp_path):
    """The ``is_ready`` probe must reflect the underlying source's state
    (e.g. directory missing) — that's how /readyz surfaces source health.
    """
    from intelligence.tasks.loaders import StaticCsvLoader
    from intelligence.telemetry import StaticSource

    loader = StaticCsvLoader(source=StaticSource(base_dir=tmp_path / "nope"))
    ok, msg = loader.is_ready()
    assert ok is False
    assert "nope" in msg or "missing" in msg


def test_static_loader_raises_clear_error_on_too_few_rows(tmp_path):
    """A CSV smaller than ``min_points`` must be refused with a clear
    message rather than letting the scaler / supervised reshape produce
    an opaque error inside the trainer."""
    from intelligence.tasks.loaders import StaticCsvLoader
    from intelligence.telemetry import StaticSource

    tiny = tmp_path / "tiny.csv"
    tiny.write_text("time,value\n2024-01-01,0.1\n2024-01-02,0.2\n2024-01-03,0.3\n")

    loader = StaticCsvLoader(source=StaticSource(base_dir=tmp_path))
    with pytest.raises(ValueError, match=r"'tiny.csv'.*3 usable.*need at least 30"):
        loader(StaticDataSource(kind="static", name="tiny.csv"))


def test_factory_signature_supports_optional_value_cols(samples_dir):
    """``static_csv_loader(value_cols=..., base_dir=...)`` produces a
    working loader; passing ``value_cols=None`` triggers autodetection
    of a single numeric column.
    """
    from intelligence.tasks.loaders import static_csv_loader

    loader = static_csv_loader(value_cols=None, base_dir=samples_dir)
    out = loader(StaticDataSource(kind="static", name="cpu_sample_dataset_orangepi.csv"))
    assert "scaler_obj" in out


# ---- PrometheusLoader ----------------------------------------------------


class _FakeSource:
    """In-memory ``TelemetrySource`` stub for loader tests.

    Records the args ``fetch_range`` was called with so tests can assert
    that ``PrometheusLoader`` translated the descriptor's ``window`` /
    ``step`` into the right datetime / timedelta arguments.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self.calls: list[dict] = []

    def fetch_range(self, query, start=None, end=None, step=None):
        self.calls.append({"query": query, "start": start, "end": end, "step": step})
        return self._df

    def is_ready(self):
        return True, "ok"


def _fake_promql_df(n: int = 40) -> pd.DataFrame:
    """Build a fake matrix response. Default size sits above the
    loader's ``min_points`` floor so unit tests exercise the happy path
    without each having to thread an override."""
    ts = pd.to_datetime([1700000000 + 60 * i for i in range(n)], unit="s", utc=True)
    return pd.DataFrame({"timestamp": ts, "value": [0.1 + 0.01 * i for i in range(n)]})


def test_prometheus_loader_passes_query_to_source():
    from intelligence.api.schemas import PrometheusDataSource
    from intelligence.tasks.loaders import PrometheusLoader

    source = _FakeSource(_fake_promql_df())
    loader = PrometheusLoader(source=source, queries=["rate(node_cpu_seconds_total[5m])"])

    loader(PrometheusDataSource(kind="prometheus", window="1h", step="1m"))

    assert len(source.calls) == 1
    assert source.calls[0]["query"] == "rate(node_cpu_seconds_total[5m])"


def test_prometheus_loader_translates_window_step_to_datetime():
    from datetime import timedelta

    from intelligence.api.schemas import PrometheusDataSource
    from intelligence.tasks.loaders import PrometheusLoader

    source = _FakeSource(_fake_promql_df())
    loader = PrometheusLoader(source=source, queries=["up"])
    loader(PrometheusDataSource(kind="prometheus", window="2h", step="30s"))

    call = source.calls[0]
    assert call["step"] == timedelta(seconds=30)
    assert call["end"] is not None and call["start"] is not None
    span = call["end"] - call["start"]
    assert span == timedelta(hours=2), f"expected 2h window, got {span}"


def test_prometheus_loader_runs_default_prepare():
    from intelligence.api.schemas import PrometheusDataSource
    from intelligence.tasks.loaders import PrometheusLoader

    source = _FakeSource(_fake_promql_df())
    loader = PrometheusLoader(source=source, queries=["up"])
    out = loader(PrometheusDataSource(kind="prometheus", window="1h", step="1m"))

    assert {"X_train", "X_test", "y_train", "y_test", "scaler_obj"} <= set(out)


def test_prometheus_loader_accepts_custom_prepare():
    from intelligence.api.schemas import PrometheusDataSource
    from intelligence.tasks.loaders import PrometheusLoader

    seen: dict = {}

    def custom_prepare(df):
        seen["rows"] = len(df)
        return {"marker": "custom"}

    fake_df = _fake_promql_df()
    source = _FakeSource(fake_df)
    loader = PrometheusLoader(source=source, queries=["up"], prepare=custom_prepare)
    out = loader(PrometheusDataSource(kind="prometheus", window="1h", step="1m"))

    assert out == {"marker": "custom"}
    assert seen["rows"] == len(fake_df)


def test_prometheus_loader_rejects_wrong_descriptor():
    from intelligence.tasks.loaders import PrometheusLoader

    source = _FakeSource(_fake_promql_df())
    loader = PrometheusLoader(source=source, queries=["up"])
    with pytest.raises(ValueError, match="PrometheusDataSource"):
        loader(StaticDataSource(kind="static", name="x.csv"))


def test_prometheus_loader_strips_nan_and_inf_before_prepare():
    """NaN values (stale markers, histogram_quantile on empty buckets) and
    ±Inf (division-by-zero PromQL) crash sklearn scalers with opaque
    errors. The loader must drop them before the prepare runs.
    """
    import numpy as np

    from intelligence.api.schemas import PrometheusDataSource
    from intelligence.tasks.loaders import PrometheusLoader

    base = _fake_promql_df(n=40)
    # Punch holes: a NaN, a +Inf, a -Inf scattered through the response.
    base.loc[5, "value"] = np.nan
    base.loc[12, "value"] = np.inf
    base.loc[20, "value"] = -np.inf

    seen: dict = {}

    def custom_prepare(df):
        seen["rows"] = len(df)
        seen["has_nan"] = bool(df["value"].isna().any())
        seen["has_inf"] = bool(np.isinf(df["value"]).any())
        return {"marker": "ok"}

    loader = PrometheusLoader(source=_FakeSource(base), queries=["up"], prepare=custom_prepare)
    loader(PrometheusDataSource(kind="prometheus", window="1h", step="1m"))

    assert seen["rows"] == 37  # 40 minus the three holes
    assert seen["has_nan"] is False
    assert seen["has_inf"] is False


def test_prometheus_loader_raises_clear_error_on_too_few_points():
    """A response with fewer usable points than ``min_points`` should be
    refused at the loader with a message that tells the operator what
    happened, rather than letting sklearn's MinMaxScaler raise on a
    too-small array. Default min_points is 30."""
    from intelligence.api.schemas import PrometheusDataSource
    from intelligence.tasks.loaders import PrometheusLoader

    source = _FakeSource(_fake_promql_df(n=5))
    loader = PrometheusLoader(source=source, queries=["rate(metric[1m])"])

    with pytest.raises(ValueError, match=r"5 usable point.* need at least 30"):
        loader(PrometheusDataSource(kind="prometheus", window="1h", step="1m"))


def test_prometheus_loader_min_points_is_configurable():
    """A custom ``min_points`` overrides the default — useful when a
    builder knows its kind tolerates very short series."""
    from intelligence.api.schemas import PrometheusDataSource
    from intelligence.tasks.loaders import PrometheusLoader

    source = _FakeSource(_fake_promql_df(n=5))
    loader = PrometheusLoader(source=source, queries=["up"], min_points=2)
    out = loader(PrometheusDataSource(kind="prometheus", window="1h", step="1m"))
    assert "scaler_obj" in out


def test_prometheus_loader_raises_clear_error_on_empty_response():
    """The most common operator mistake is a PromQL that matches no series
    over the requested window. Without this gate, the loader hands an
    empty DataFrame to a model trainer and the user sees an opaque numpy
    IndexError. Surface a clear error with the query text instead."""
    from intelligence.api.schemas import PrometheusDataSource
    from intelligence.tasks.loaders import PrometheusLoader

    empty_df = pd.DataFrame(columns=["timestamp", "value"])
    source = _FakeSource(empty_df)
    loader = PrometheusLoader(source=source, queries=["nonsense_metric"])

    with pytest.raises(ValueError, match=r"no data.*nonsense_metric"):
        loader(PrometheusDataSource(kind="prometheus", window="1h", step="1m"))


def test_prometheus_loader_propagates_transport_errors():
    """Network errors from the source (timeout, connection refused, 5xx)
    must propagate so the service layer can map them to 502. The loader
    shouldn't swallow or rewrap them."""
    import requests

    from intelligence.api.schemas import PrometheusDataSource
    from intelligence.tasks.loaders import PrometheusLoader

    class _FlakySource:
        def fetch_range(self, *a, **kw):
            raise requests.ConnectionError("connection refused")

        def is_ready(self):
            return False, "down"

    loader = PrometheusLoader(source=_FlakySource(), queries=["up"])
    with pytest.raises(requests.ConnectionError):
        loader(PrometheusDataSource(kind="prometheus", window="1h", step="1m"))


def test_prometheus_loader_rejects_empty_queries_list():
    """A loader with no queries can't fetch anything — fail loudly at
    construction rather than producing a useless instance."""
    from intelligence.tasks.loaders import PrometheusLoader

    with pytest.raises(ValueError, match="at least one query"):
        PrometheusLoader(source=_FakeSource(_fake_promql_df()), queries=[])


def test_prometheus_loader_rejects_mismatched_value_cols():
    """``value_cols`` must pair 1:1 with ``queries`` — anything else is
    a builder bug surfaced loudly."""
    from intelligence.tasks.loaders import PrometheusLoader

    with pytest.raises(ValueError, match="pair 1:1"):
        PrometheusLoader(
            source=_FakeSource(_fake_promql_df()),
            queries=["up", "down"],
            value_cols=["cpu"],
        )


def test_prometheus_loader_multivariate_joins_on_timestamp():
    """Two queries → two features. Results join on timestamp; the
    DataFrame fed to ``prepare`` has both feature columns named after
    ``value_cols``.
    """
    from intelligence.api.schemas import PrometheusDataSource
    from intelligence.tasks.loaders import PrometheusLoader

    class _DualSource:
        def __init__(self, by_query: dict[str, pd.DataFrame]) -> None:
            self._by_query = by_query

        def fetch_range(self, query, start=None, end=None, step=None):
            return self._by_query[query].copy()

    cpu_df = _fake_promql_df(n=40)
    mem_df = _fake_promql_df(n=40)
    mem_df["value"] = [0.6 + 0.005 * i for i in range(40)]

    seen: dict = {}

    def custom_prepare(df):
        seen["columns"] = list(df.columns)
        seen["rows"] = len(df)
        return {"marker": "ok"}

    loader = PrometheusLoader(
        source=_DualSource({"cpu_q": cpu_df, "mem_q": mem_df}),
        queries=["cpu_q", "mem_q"],
        value_cols=["cpu", "mem"],
        prepare=custom_prepare,
    )
    loader(PrometheusDataSource(kind="prometheus", window="1h", step="1m"))

    assert seen["columns"] == ["timestamp", "cpu", "mem"]
    assert seen["rows"] == 40


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("30s", 30),
        ("5m", 300),
        ("2h", 7200),
        ("1d", 86400),
        ("1w", 7 * 86400),
    ],
)
def test_parse_duration_understands_promql_units(spec, expected):
    from intelligence.tasks.loaders import _parse_duration

    assert _parse_duration(spec).total_seconds() == expected


def test_parse_duration_rejects_unknown_units():
    from intelligence.tasks.loaders import _parse_duration

    with pytest.raises(ValueError):
        _parse_duration("1y")  # PromQL allows it but ARIMA windows shouldn't need it
    with pytest.raises(ValueError):
        _parse_duration("nonsense")
