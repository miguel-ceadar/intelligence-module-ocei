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


def test_factory_signature_is_backwards_compatible(samples_dir):
    """``static_csv_loader(value_col=..., base_dir=...)`` must keep
    producing a working loader — existing factories in catalog.py rely on
    this signature.
    """
    from intelligence.tasks.loaders import static_csv_loader

    loader = static_csv_loader(value_col=None, base_dir=samples_dir)
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


def _fake_promql_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime([1700000000, 1700000060, 1700000120], unit="s", utc=True),
            "value": [0.1, 0.2, 0.3],
        }
    )


def test_prometheus_loader_passes_query_to_source():
    from intelligence.api.schemas import PrometheusDataSource
    from intelligence.tasks.loaders import PrometheusLoader

    source = _FakeSource(_fake_promql_df())
    loader = PrometheusLoader(source=source, query='rate(node_cpu_seconds_total[5m])')

    loader(PrometheusDataSource(kind="prometheus", window="1h", step="1m"))

    assert len(source.calls) == 1
    assert source.calls[0]["query"] == 'rate(node_cpu_seconds_total[5m])'


def test_prometheus_loader_translates_window_step_to_datetime():
    from datetime import timedelta

    from intelligence.api.schemas import PrometheusDataSource
    from intelligence.tasks.loaders import PrometheusLoader

    source = _FakeSource(_fake_promql_df())
    loader = PrometheusLoader(source=source, query="up")
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
    loader = PrometheusLoader(source=source, query="up")
    out = loader(PrometheusDataSource(kind="prometheus", window="1h", step="1m"))

    assert {"X_train", "X_test", "y_train", "y_test", "scaler_obj"} <= set(out)


def test_prometheus_loader_accepts_custom_prepare():
    from intelligence.api.schemas import PrometheusDataSource
    from intelligence.tasks.loaders import PrometheusLoader

    seen: dict = {}

    def custom_prepare(df):
        seen["rows"] = len(df)
        return {"marker": "custom"}

    source = _FakeSource(_fake_promql_df())
    loader = PrometheusLoader(source=source, query="up", prepare=custom_prepare)
    out = loader(PrometheusDataSource(kind="prometheus", window="1h", step="1m"))

    assert out == {"marker": "custom"}
    assert seen["rows"] == 3


def test_prometheus_loader_rejects_wrong_descriptor():
    from intelligence.tasks.loaders import PrometheusLoader

    source = _FakeSource(_fake_promql_df())
    loader = PrometheusLoader(source=source, query="up")
    with pytest.raises(ValueError, match="PrometheusDataSource"):
        loader(StaticDataSource(kind="static", name="x.csv"))


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
