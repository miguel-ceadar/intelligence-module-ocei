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


def test_static_source_sorts_by_timestamp_when_column_present(tmp_path):
    """Out-of-order rows in a CSV silently break downstream row-position
    splits (train/test, sliding windows). The source must return rows
    sorted ascending by any standard timestamp column so the prepare
    contract — "rows are in chronological order" — holds upstream.
    """
    p = tmp_path / "shuffled.csv"
    p.write_text(
        "time,value\n2024-01-03,3\n2024-01-01,1\n2024-01-05,5\n2024-01-02,2\n2024-01-04,4\n"
    )
    df = StaticSource(base_dir=tmp_path).fetch_range("shuffled.csv")
    assert df["value"].tolist() == [1, 2, 3, 4, 5]
    assert list(df.index) == [0, 1, 2, 3, 4], "index must be reset after sort"


def test_static_source_sort_recognises_case_variants(tmp_path):
    """Timestamp column detection is case-insensitive across the standard
    names so a CSV with ``Timestamp`` or ``DATE`` headers still sorts.
    """
    p = tmp_path / "cased.csv"
    p.write_text("Timestamp,value\n2024-01-02,2\n2024-01-01,1\n")
    df = StaticSource(base_dir=tmp_path).fetch_range("cased.csv")
    assert df["value"].tolist() == [1, 2]


def test_static_source_leaves_order_alone_when_no_timestamp_column(tmp_path):
    """Without a recognised timestamp column the source has no basis for
    sorting; row order is preserved as-is."""
    p = tmp_path / "no_ts.csv"
    p.write_text("a,b\n3,30\n1,10\n2,20\n")
    df = StaticSource(base_dir=tmp_path).fetch_range("no_ts.csv")
    assert df["a"].tolist() == [3, 1, 2]
