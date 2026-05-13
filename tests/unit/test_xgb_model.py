"""Train + predict roundtrip for ``XgbModel``.

Uses a synthetic random walk so the test is sub-second and
deterministic. Verifies:
  - ``make_xgb_prepare`` produces the components shape ``train_xgb`` expects.
  - ``XgbModel.train`` saves a Bento with the X and y scalers populated.
  - ``XgbModel.predict`` returns a float from the saved Bento using a
    window of observations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _synthetic_cpu(n: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.standard_normal(n) * 0.02) + 0.5
    return pd.DataFrame({"timestamp": np.arange(n), "cpu": walk.clip(0.05, 0.95)})


def test_make_xgb_prepare_yields_expected_components():
    from intelligence.ml.models.xgb import make_xgb_prepare

    prep = make_xgb_prepare(look_back=4, num_variables=1)
    comps = prep(_synthetic_cpu())

    for key in ("X_train", "X_test", "y_train", "y_test", "scaler_obj", "scaler_X", "look_back"):
        assert key in comps, f"missing component: {key}"
    assert comps["look_back"] == 4
    assert comps["X_train"].shape[1] == 4  # look_back lag features
    assert hasattr(comps["scaler_X"], "transform")
    assert hasattr(comps["scaler_obj"], "transform")


def test_xgb_predict_rejects_short_window():
    """Contract: window shorter than ``look_back`` is rejected before
    any model call. No training needed — construct minimal artefacts
    and assert the size check fires.
    """
    from sklearn.preprocessing import StandardScaler

    from intelligence.ml.models.xgb import XgbModel

    artifacts = {
        "regressor": object(),  # never reached — error fires on window length
        "scaler_obj": StandardScaler().fit(np.array([[0.0], [1.0]])),
        "scaler_X": StandardScaler().fit(np.zeros((2, 6))),
        "look_back": 6,
    }
    with pytest.raises(ValueError, match="at least 6"):
        XgbModel().predict(artifacts, {"cpu": [0.5, 0.5]})


def test_xgb_predict_recursive_multi_horizon():
    """``horizon=N`` returns a list of length N. XGB is recursive — no
    native confidence intervals, so ``lower``/``upper`` stay ``None``.
    """
    from intelligence.ml.models.xgb import XgbModel, make_xgb_prepare

    prep = make_xgb_prepare(look_back=6, num_variables=1)
    comps = prep(_synthetic_cpu(n=200))
    comps["model_parameters"] = {"n_estimators": 20}
    model = XgbModel()
    artifacts, _ = model.fit(comps)

    window = _synthetic_cpu(n=8).iloc[-6:]["cpu"].tolist()
    out = model.predict(artifacts, {"cpu": window}, horizon=4)
    assert isinstance(out, list) and len(out) == 4
    for point in out:
        assert point.value is not None
        assert point.lower is None
        assert point.upper is None


def test_xgb_predict_horizon_one_returns_single_element_list():
    from intelligence.ml.models.xgb import XgbModel, make_xgb_prepare

    prep = make_xgb_prepare(look_back=6, num_variables=1)
    comps = prep(_synthetic_cpu(n=200))
    comps["model_parameters"] = {"n_estimators": 20}
    model = XgbModel()
    artifacts, _ = model.fit(comps)
    out = model.predict(
        artifacts,
        {"cpu": _synthetic_cpu(n=8).iloc[-6:]["cpu"].tolist()},
        horizon=1,
    )
    assert isinstance(out, list) and len(out) == 1


# ---- New protocol: fit / save_artifacts / load_artifacts ------------------


@pytest.fixture
def xgb_artifacts_fit():
    """A fresh ``(model, artifacts, metrics, comps)`` quad from ``XgbModel.fit``."""
    from intelligence.ml.models.xgb import XgbModel, make_xgb_prepare

    prep = make_xgb_prepare(look_back=6, num_variables=1)
    comps = prep(_synthetic_cpu(n=200))
    comps["model_parameters"] = {"n_estimators": 20, "max_depth": 3, "eta": 0.1}
    model = XgbModel()
    artifacts, metrics = model.fit(comps)
    return model, artifacts, metrics, comps


def test_xgb_fit_returns_artifacts_and_metrics(xgb_artifacts_fit):
    _model, artifacts, metrics, _ = xgb_artifacts_fit
    assert isinstance(artifacts, dict)
    assert isinstance(metrics, dict)
    assert "mae" in metrics
    # Everything predict needs is in the artifacts dict — including the
    # regressor itself, not just metadata.
    assert "regressor" in artifacts
    assert "scaler_X" in artifacts
    assert "scaler_obj" in artifacts  # y-scaler
    assert "look_back" in artifacts


def test_xgb_save_artifacts_writes_native_ubj_no_pickle(xgb_artifacts_fit, tmp_path):
    model, artifacts, _, _ = xgb_artifacts_fit
    files = model.save_artifacts(artifacts, tmp_path)

    # Native xgboost UBJ as the model file — not pickle.
    assert files["model"] == "xgb.ubj"
    assert (tmp_path / "xgb.ubj").exists()

    # Two scaler sidecars (X for input feature columns, y for the target).
    assert (tmp_path / "scaler_x.json").exists()
    assert (tmp_path / "scaler_x.npz").exists()
    assert (tmp_path / "scaler_y.json").exists()
    assert (tmp_path / "scaler_y.npz").exists()
    assert (tmp_path / "xgb_meta.json").exists()
    assert (tmp_path / "metrics.json").exists()

    assert not list(tmp_path.glob("*.pkl"))
    assert not list(tmp_path.glob("*.pickle"))


def test_xgb_save_artifacts_includes_input_spec_when_present(xgb_artifacts_fit, tmp_path):
    from intelligence.tasks.contracts import InputSpec

    model, artifacts, _, _ = xgb_artifacts_fit
    artifacts["input_spec"] = InputSpec(
        n_features=1,
        feature_names=["cpu"],
        steps_back=6,
        value_range={"cpu": (0.0, 1.0)},
    )
    files = model.save_artifacts(artifacts, tmp_path)
    assert files.get("input_spec") == "input_spec.json"
    assert (tmp_path / "input_spec.json").exists()


def test_xgb_load_artifacts_round_trips_full_state(xgb_artifacts_fit, tmp_path):
    """Round-trip equivalence — the loaded regressor predicts identically
    on the same input, and the scalers transform identically."""
    model, artifacts, _, _ = xgb_artifacts_fit
    model.save_artifacts(artifacts, tmp_path)
    loaded = model.load_artifacts(tmp_path)

    assert {"regressor", "scaler_X", "scaler_obj", "look_back"} <= set(loaded.keys())
    assert loaded["look_back"] == artifacts["look_back"]

    # Build a sample window in the scaler's expected column layout.
    cols = getattr(artifacts["scaler_X"], "feature_names_in_", None)
    sample = np.array([[0.5] * artifacts["look_back"]])
    if cols is not None:
        sample = pd.DataFrame(sample, columns=cols)

    np.testing.assert_allclose(
        artifacts["scaler_X"].transform(sample),
        loaded["scaler_X"].transform(sample),
    )
    sample_y = np.array([[0.5]])
    np.testing.assert_allclose(
        artifacts["scaler_obj"].transform(sample_y),
        loaded["scaler_obj"].transform(sample_y),
    )

    # Native xgboost UBJ round-trip preserves prediction output exactly.
    np.testing.assert_allclose(
        artifacts["regressor"].predict(
            artifacts["scaler_X"].transform(sample)
        ),
        loaded["regressor"].predict(loaded["scaler_X"].transform(sample)),
    )


def test_xgb_load_artifacts_restores_input_spec(xgb_artifacts_fit, tmp_path):
    from intelligence.tasks.contracts import InputSpec

    model, artifacts, _, _ = xgb_artifacts_fit
    artifacts["input_spec"] = InputSpec(
        n_features=1, feature_names=["cpu"], steps_back=6
    )
    model.save_artifacts(artifacts, tmp_path)

    loaded = model.load_artifacts(tmp_path)
    assert isinstance(loaded["input_spec"], InputSpec)
    assert loaded["input_spec"].feature_names == ["cpu"]


def test_xgb_files_map_declares_only_safe_extensions(xgb_artifacts_fit, tmp_path):
    from pathlib import Path as _P

    from intelligence.ml.artifact.manifest import ALLOWED_EXTENSIONS

    model, artifacts, _, _ = xgb_artifacts_fit
    files = model.save_artifacts(artifacts, tmp_path)
    for role, fname in files.items():
        assert _P(fname).suffix.lower() in ALLOWED_EXTENSIONS, (
            f"role {role!r} declares {fname!r} with disallowed extension"
        )
