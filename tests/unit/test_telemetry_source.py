"""Unit tests for the ``TelemetrySource`` Protocol + ``StaticSource``.

``TelemetrySource`` is the seam between data fetching (CSV / PromQL /
future OTel) and feature preparation. Tests here pin the contract:
implementations must accept a ``query`` (filename for static, PromQL for
Prometheus) and return a ``pandas.DataFrame``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from intelligence.telemetry import StaticSource, TelemetrySource


def test_static_source_satisfies_protocol():
    assert isinstance(StaticSource(), TelemetrySource), (
        "StaticSource must structurally satisfy TelemetrySource"
    )


def test_static_source_fetch_range_returns_dataframe(samples_dir):
    src = StaticSource(base_dir=samples_dir)
    df = src.fetch_range("cpu_sample_dataset_orangepi.csv")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_static_source_ignores_time_window_args(samples_dir):
    """start/end/step are no-ops for a static CSV — the file is what it is."""
    src = StaticSource(base_dir=samples_dir)
    df_full = src.fetch_range("cpu_sample_dataset_orangepi.csv")
    df_windowed = src.fetch_range(
        "cpu_sample_dataset_orangepi.csv",
        start=datetime(2020, 1, 1, tzinfo=UTC),
        end=datetime(2030, 1, 1, tzinfo=UTC),
        step=timedelta(minutes=1),
    )
    pd.testing.assert_frame_equal(df_full, df_windowed)


def test_static_source_missing_file_raises(samples_dir):
    src = StaticSource(base_dir=samples_dir)
    with pytest.raises(FileNotFoundError):
        src.fetch_range("does_not_exist.csv")


def test_static_source_is_ready_reports_missing_dir(tmp_path):
    src = StaticSource(base_dir=tmp_path / "missing")
    ok, msg = src.is_ready()
    assert ok is False
    assert "missing" in msg
