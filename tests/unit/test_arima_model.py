"""Train + predict roundtrip for ``ArimaModel``, including multi-horizon
with confidence intervals (Wave 1 #2).

Single-step predict was previously covered only by the API integration
tests; this file pulls the contract into a focused unit test so multi-
horizon + CI changes can be validated without spinning the full service.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _synthetic_cpu(n: int = 120, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.standard_normal(n) * 0.02) + 0.5
    return pd.DataFrame({"timestamp": np.arange(n), "cpu": walk.clip(0.05, 0.95)})


def _components(df: pd.DataFrame) -> dict:
    """ARIMA prepare: 80/20 split, MinMax-scaled, model_parameters carries
    the order. Matches what ``_make_univariate_prepare`` produces."""
    from sklearn.preprocessing import MinMaxScaler

    series = df["cpu"].astype(float).values.reshape(-1, 1)
    split = int(len(series) * 0.8)
    scaler = MinMaxScaler().fit(series[:split])
    return {
        "X_train": scaler.transform(series[:split]),
        "X_test": scaler.transform(series[split:]),
        "y_train": series[:split].ravel(),
        "y_test": series[split:].ravel(),
        "scaler_obj": scaler,
        "model_parameters": {"p": 2, "d": 1, "q": 0},
    }


def test_arima_predict_multi_horizon_returns_list_of_forecast_points():
    """horizon=N returns a list of length N. Each entry is a
    ForecastPoint with a ``value`` plus 95 % CI bounds from
    ``get_forecast(steps=N)``.
    """
    pytest.importorskip("statsmodels")
    from intelligence.ml.models.arima import ArimaModel

    comps = _components(_synthetic_cpu(n=150))
    model = ArimaModel()
    artifacts, _ = model.fit(comps)

    out = model.predict(artifacts, {"cpu": [0.42]}, horizon=5)
    assert isinstance(out, list) and len(out) == 5
    for point in out:
        assert point.value is not None
        # ARIMA ships native CIs — both bounds populated.
        assert point.lower is not None and point.upper is not None
        assert point.lower <= point.value <= point.upper


def test_arima_predict_horizon_one_returns_single_point():
    """``horizon=1`` is the default; predict returns a list of length 1."""
    from intelligence.ml.models.arima import ArimaModel

    comps = _components(_synthetic_cpu(n=150))
    model = ArimaModel()
    artifacts, _ = model.fit(comps)
    out = model.predict(artifacts, {"cpu": [0.42]}, horizon=1)
    assert isinstance(out, list) and len(out) == 1


# ---- New protocol: fit / save_artifacts / load_artifacts ------------------
#
# These exercise the pickle-free path. They don't touch BentoML — the
# directory is the artefact, and round-trip equivalence is checked by
# comparing the loaded dict to what ``fit`` produced. The shim in the
# parity test stays only until step 9, when ``predict`` is refactored
# to take the artefacts dict directly.


@pytest.fixture
def arima_artifacts_fit():
    """A fresh ``(model, artifacts, metrics)`` triple from ``ArimaModel.fit``."""
    from intelligence.ml.models.arima import ArimaModel

    comps = _components(_synthetic_cpu(n=150))
    model = ArimaModel()
    artifacts, metrics = model.fit(comps)
    return model, artifacts, metrics


def test_arima_fit_returns_artifacts_and_metrics(arima_artifacts_fit):
    _model, artifacts, metrics = arima_artifacts_fit

    assert isinstance(artifacts, dict)
    assert isinstance(metrics, dict)
    # Everything predict needs lives in the artifacts dict.
    assert "scaler_obj" in artifacts
    assert "historical_data" in artifacts
    assert "arima_order" in artifacts
    assert "model_metrics" in artifacts


def test_arima_save_artifacts_writes_pickle_free_files(arima_artifacts_fit, tmp_path):
    model, artifacts, _ = arima_artifacts_fit
    files = model.save_artifacts(artifacts, tmp_path)

    # Manifest files map declares the model file and its sidecars.
    assert "model" in files
    assert files["model"] == "arima.json"

    # Physical files present.
    assert (tmp_path / "arima.json").exists()
    assert (tmp_path / "scaler.json").exists()
    assert (tmp_path / "scaler.npz").exists()
    assert (tmp_path / "metrics.json").exists()

    # No pickle leakage anywhere.
    assert not list(tmp_path.glob("*.pkl"))
    assert not list(tmp_path.glob("*.pickle"))


def test_arima_save_artifacts_includes_input_spec_when_present(arima_artifacts_fit, tmp_path):
    from intelligence.tasks.contracts import InputSpec

    model, artifacts, _ = arima_artifacts_fit
    artifacts["input_spec"] = InputSpec(
        n_features=1,
        feature_names=["cpu"],
        steps_back=1,
        value_range={"cpu": (0.0, 1.0)},
    )

    files = model.save_artifacts(artifacts, tmp_path)
    assert files.get("input_spec") == "input_spec.json"
    assert (tmp_path / "input_spec.json").exists()


def test_arima_save_artifacts_omits_input_spec_when_absent(arima_artifacts_fit, tmp_path):
    model, artifacts, _ = arima_artifacts_fit
    artifacts.pop("input_spec", None)

    files = model.save_artifacts(artifacts, tmp_path)
    assert "input_spec" not in files
    assert not (tmp_path / "input_spec.json").exists()


def test_arima_load_artifacts_round_trips_full_state(arima_artifacts_fit, tmp_path):
    """Round-trip equivalence — the loaded dict carries scaler, history,
    and order identical to the originals."""
    model, artifacts, _ = arima_artifacts_fit
    model.save_artifacts(artifacts, tmp_path)
    loaded = model.load_artifacts(tmp_path)

    assert set(loaded.keys()) >= {
        "scaler_obj",
        "historical_data",
        "arima_order",
        "model_metrics",
    }

    sample = np.array([[0.42]])
    np.testing.assert_allclose(
        loaded["scaler_obj"].transform(sample),
        artifacts["scaler_obj"].transform(sample),
    )
    np.testing.assert_allclose(
        np.asarray(loaded["historical_data"], dtype=float),
        np.asarray(artifacts["historical_data"], dtype=float),
    )
    assert tuple(loaded["arima_order"]) == tuple(artifacts["arima_order"])


def test_arima_load_artifacts_restores_input_spec(arima_artifacts_fit, tmp_path):
    from intelligence.tasks.contracts import InputSpec

    model, artifacts, _ = arima_artifacts_fit
    artifacts["input_spec"] = InputSpec(
        n_features=1, feature_names=["cpu"], steps_back=1
    )
    model.save_artifacts(artifacts, tmp_path)

    loaded = model.load_artifacts(tmp_path)
    assert isinstance(loaded["input_spec"], InputSpec)
    assert loaded["input_spec"].feature_names == ["cpu"]


def test_arima_predict_via_loaded_artifacts_matches_original(arima_artifacts_fit, tmp_path):
    """End-to-end parity: predict on the loaded artefacts produces the
    same value as predict on the in-memory artefacts."""
    model, artifacts, _ = arima_artifacts_fit

    original = model.predict(artifacts, {"cpu": [0.42]})

    model.save_artifacts(artifacts, tmp_path)
    loaded = model.load_artifacts(tmp_path)
    via_loaded = model.predict(loaded, {"cpu": [0.42]})

    assert original[0].value == via_loaded[0].value


def test_arima_files_map_declares_only_safe_extensions(arima_artifacts_fit, tmp_path):
    """Sanity: every filename the save returns is an extension the
    manifest layer would accept. Catches typos before they hit the
    manifest validation."""
    from intelligence.ml.artifact.manifest import ALLOWED_EXTENSIONS

    model, artifacts, _ = arima_artifacts_fit
    files = model.save_artifacts(artifacts, tmp_path)
    for role, fname in files.items():
        from pathlib import Path as _P

        assert _P(fname).suffix.lower() in ALLOWED_EXTENSIONS, (
            f"role {role!r} declares {fname!r} with disallowed extension"
        )
