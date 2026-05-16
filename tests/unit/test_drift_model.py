"""Tests for ``DriftModel`` — NannyML-backed drift implementing the
``Model`` protocol.

The drift task moved from a bespoke ``DriftDetectionTask`` subclass of
``BaseTask`` to a regular ``Model`` plugged into plain ``BaseTask``.
These tests exercise the model in isolation; end-to-end task tests
live in ``test_drift_task.py`` (which constructs a ``BaseTask`` around
this model).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests._synthetic import stationary_cpu


def test_drift_model_advertises_drift_kind():
    """``name`` is the dispatch key + manifest tag. ``has_drift`` flag
    on a model means "this forecaster has a paired drift detector" —
    for the drift detector itself that's ``False`` (it IS the
    detector, doesn't pair with another)."""
    from intelligence.ml.models.drift import DriftModel

    model = DriftModel()
    assert model.name == "drift"
    assert model.has_drift is False


def test_drift_fit_returns_artifacts_and_metrics():
    """``fit`` packages the reference frame + config into an artifacts
    dict and returns ``(artifacts, metrics)``. The NannyML calculator
    itself is refit at ``load_artifacts`` time (no pickle-free save)."""
    from intelligence.ml.models.drift import DriftModel

    model = DriftModel(chunk_size=12, metric="jensen_shannon", forecaster_task_name="t_forecaster")
    reference = stationary_cpu(300)
    artifacts, metrics = model.fit({"reference_df": reference, "drift_columns": ["cpu"]})

    assert metrics["reference_size"] == 300
    assert artifacts["column_names"] == ["cpu"]
    assert artifacts["chunk_size"] == 12
    assert artifacts["metric"] == "jensen_shannon"
    assert artifacts["forecaster_task_name"] == "t_forecaster"


def test_drift_save_artifacts_writes_parquet_no_pickle(tmp_path):
    """Reference data persists as parquet + a JSON config sidecar. No
    pickle on disk — drift's calculator state is regenerable from the
    reference + config."""
    from intelligence.ml.models.drift import DriftModel

    model = DriftModel(chunk_size=12, metric="jensen_shannon")
    artifacts, _ = model.fit({"reference_df": stationary_cpu(300), "drift_columns": ["cpu"]})
    files = model.save_artifacts(artifacts, tmp_path)

    assert files["reference"] == "reference.parquet"
    assert (tmp_path / "reference.parquet").exists()
    assert (tmp_path / "drift.json").exists()
    assert (tmp_path / "metrics.json").exists()
    assert not list(tmp_path.glob("*.pkl"))
    assert not list(tmp_path.glob("*.pickle"))


def test_drift_save_artifacts_includes_input_spec_when_present(tmp_path):
    from intelligence.ml.models.drift import DriftModel
    from intelligence.tasks.contracts import InputSpec

    model = DriftModel(chunk_size=12)
    artifacts, _ = model.fit({"reference_df": stationary_cpu(300), "drift_columns": ["cpu"]})
    artifacts["input_spec"] = InputSpec(n_features=1, feature_names=["cpu"], steps_back=12)
    files = model.save_artifacts(artifacts, tmp_path)

    assert files.get("input_spec") == "input_spec.json"
    assert (tmp_path / "input_spec.json").exists()


def test_drift_load_artifacts_refits_calculator(tmp_path):
    """``load_artifacts`` reads the persisted reference + config and
    refits a fresh NannyML calculator — the artifact dict it returns
    is the same shape ``fit`` produced, plus the live calculator."""
    from intelligence.ml.models.drift import DriftModel

    model = DriftModel(chunk_size=12, metric="jensen_shannon")
    artifacts, _ = model.fit({"reference_df": stationary_cpu(300), "drift_columns": ["cpu"]})
    model.save_artifacts(artifacts, tmp_path)
    loaded = model.load_artifacts(tmp_path)

    assert "drift_calculator" in loaded
    assert loaded["column_names"] == artifacts["column_names"]
    assert loaded["chunk_size"] == artifacts["chunk_size"]
    assert loaded["metric"] == artifacts["metric"]


def test_drift_load_artifacts_restores_input_spec(tmp_path):
    from intelligence.ml.models.drift import DriftModel
    from intelligence.tasks.contracts import InputSpec

    model = DriftModel(chunk_size=12)
    artifacts, _ = model.fit({"reference_df": stationary_cpu(300), "drift_columns": ["cpu"]})
    artifacts["input_spec"] = InputSpec(n_features=1, feature_names=["cpu"], steps_back=12)
    model.save_artifacts(artifacts, tmp_path)

    loaded = model.load_artifacts(tmp_path)
    assert isinstance(loaded["input_spec"], InputSpec)
    assert loaded["input_spec"].feature_names == ["cpu"]


def test_drift_predict_returns_dict_on_similar_chunk(tmp_path):
    """Drift's predict returns a dict with the alert flag, chunk count,
    metric, and forecaster name. Same-distribution input should not
    raise an alert."""
    from intelligence.ml.models.drift import DriftModel

    model = DriftModel(chunk_size=12, forecaster_task_name="t_forecaster")
    artifacts, _ = model.fit(
        {
            "reference_df": stationary_cpu(300, mean=0.5, std=0.05, seed=1),
            "drift_columns": ["cpu"],
        }
    )
    model.save_artifacts(artifacts, tmp_path)
    loaded = model.load_artifacts(tmp_path)

    similar = stationary_cpu(12, mean=0.5, std=0.05, seed=2)
    result = model.predict(loaded, {"cpu": similar["cpu"].tolist()}, horizon=1)

    assert result["drift_detected"] is False
    assert result["metric"] == "jensen_shannon"
    assert result["forecaster"] == "t_forecaster"


def test_drift_predict_flags_shifted_distribution(tmp_path):
    from intelligence.ml.models.drift import DriftModel

    model = DriftModel(chunk_size=12)
    artifacts, _ = model.fit(
        {
            "reference_df": stationary_cpu(300, mean=0.5, std=0.05, seed=1),
            "drift_columns": ["cpu"],
        }
    )
    model.save_artifacts(artifacts, tmp_path)
    loaded = model.load_artifacts(tmp_path)

    shifted = stationary_cpu(12, mean=0.9, std=0.02, seed=3)
    result = model.predict(loaded, {"cpu": shifted["cpu"].tolist()})
    assert result["drift_detected"] is True


def test_drift_predict_ignores_horizon():
    """Drift has no notion of forecast horizon. The arg is accepted to
    fit the ``Model`` protocol signature; the value is ignored.
    """
    # Synthetic minimal-load artifacts dict so we don't have to round-trip
    # through save/load. The calculator-bearing dict is what ``predict``
    # consumes; building it here directly tests the predict semantics.
    import nannyml as nml

    from intelligence.ml.models.drift import DriftModel

    reference = stationary_cpu(300)
    calc = nml.UnivariateDriftCalculator(column_names=["cpu"], chunk_size=12).fit(
        reference[["cpu"]]
    )
    artifacts = {
        "drift_calculator": calc,
        "column_names": ["cpu"],
        "chunk_size": 12,
        "metric": "jensen_shannon",
        "forecaster_task_name": "",
    }
    model = DriftModel(chunk_size=12)
    similar = stationary_cpu(12, seed=2)
    out_h1 = model.predict(artifacts, {"cpu": similar["cpu"].tolist()}, horizon=1)
    out_h7 = model.predict(artifacts, {"cpu": similar["cpu"].tolist()}, horizon=7)
    assert out_h1 == out_h7


def test_drift_predict_raises_when_input_columns_diverge_from_artifact(tmp_path):
    """If the saved artifact's ``column_names`` differs from the
    request keys (e.g. an old artifact loaded under a spec that has
    since added a covariate, surfaced through ``allow_unverified_models``),
    the predict path must raise. The previous implementation silently
    intersected and would return a meaningless drift verdict over a
    smaller column set.
    """
    from intelligence.ml.models.drift import DriftModel

    model = DriftModel(chunk_size=12)
    artifacts, _ = model.fit(
        {
            "reference_df": stationary_cpu(300),
            "drift_columns": ["cpu"],  # artifact only knows about 'cpu'
        }
    )
    model.save_artifacts(artifacts, tmp_path)
    loaded = model.load_artifacts(tmp_path)

    # Request supplies 'mem' instead of 'cpu' — a clear contract mismatch
    # that BaseTask's InputSpec check would normally catch. Bypassing it
    # here (no InputSpec) mimics the legacy-artifact path through
    # allow_unverified_models=True.
    with pytest.raises(ValueError, match=r"cpu"):
        model.predict(loaded, {"mem": stationary_cpu(12)["cpu"].tolist()})


def test_make_drift_prepare_emits_reference_and_columns():
    """``make_drift_prepare`` is the data-shaping factory the builder
    pairs with the loader. It produces the dict shape ``DriftModel.fit``
    consumes — a reference DataFrame and the list of columns to monitor.
    """
    from intelligence.ml.models.drift import make_drift_prepare

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
            "cpu": np.linspace(0.1, 0.9, 100),
            "mem": np.linspace(0.4, 0.5, 100),
        }
    )
    prep = make_drift_prepare()
    out = prep(df)

    assert "reference_df" in out
    assert out["drift_columns"] == ["cpu", "mem"]  # timestamp dropped


def test_make_drift_prepare_respects_value_col_when_set():
    from intelligence.ml.models.drift import make_drift_prepare

    df = pd.DataFrame({"cpu": [0.5] * 30, "mem": [0.6] * 30})
    prep = make_drift_prepare(value_col="cpu")
    out = prep(df)
    assert out["drift_columns"] == ["cpu"]
    assert list(out["reference_df"].columns) == ["cpu"]


def test_make_drift_prepare_raises_when_value_col_missing():
    from intelligence.ml.models.drift import make_drift_prepare

    df = pd.DataFrame({"mem": [0.6] * 30})
    prep = make_drift_prepare(value_col="cpu")
    with pytest.raises(ValueError, match="cpu"):
        prep(df)


def test_drift_model_save_uses_only_allowed_extensions(tmp_path: Path):
    from intelligence.ml.artifact.manifest import ALLOWED_EXTENSIONS
    from intelligence.ml.models.drift import DriftModel

    model = DriftModel(chunk_size=12)
    artifacts, _ = model.fit({"reference_df": stationary_cpu(300), "drift_columns": ["cpu"]})
    files = model.save_artifacts(artifacts, tmp_path)
    for role, fname in files.items():
        assert Path(fname).suffix.lower() in ALLOWED_EXTENSIONS, (
            f"role {role!r} declares {fname!r} with disallowed extension"
        )
