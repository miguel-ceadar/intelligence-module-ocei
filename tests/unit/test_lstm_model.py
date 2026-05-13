"""Train + predict roundtrip for ``LstmModel``.

Uses a synthetic random walk + tiny model (hidden_size=4, 2 epochs) so
the test stays under a couple of seconds. Verifies:
  - ``make_lstm_prepare`` produces 3-D tensors + ``TimeSeriesDataset`` instances.
  - ``LstmModel.train`` saves a Bento with the scaler + window metadata.
  - ``LstmModel.predict`` returns a float from the saved Bento.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _synthetic_cpu(n: int = 200, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.standard_normal(n) * 0.02) + 0.5
    return pd.DataFrame({"timestamp": np.arange(n), "cpu": walk.clip(0.05, 0.95)})


def test_make_lstm_prepare_yields_3d_tensors():
    pytest.importorskip("torch")
    from intelligence.ml.models.lstm import make_lstm_prepare

    prep = make_lstm_prepare(look_back=5, num_variables=1, batch_size=8)
    comps = prep(_synthetic_cpu())

    for key in (
        "X_train",
        "X_test",
        "y_train",
        "y_test",
        "train_dataset",
        "test_dataset",
        "batch_size",
        "scaler_obj",
        "look_back",
        "num_variables",
    ):
        assert key in comps, f"missing component: {key}"
    assert comps["X_train"].dim() == 3
    assert comps["X_train"].shape[1] == 5  # look_back
    assert comps["X_train"].shape[2] == 1  # num_variables


def test_lstm_predict_rejects_short_window():
    """Contract test — no training, just construct minimal artefacts
    and assert predict rejects windows shorter than ``look_back``."""
    pytest.importorskip("torch")
    from sklearn.preprocessing import MinMaxScaler

    from intelligence.ml.models.lstm import LstmModel

    artifacts = {
        "network": None,  # never reached — size check fires first
        "look_back": 6,
        "num_variables": 1,
        "scaler_obj": MinMaxScaler().fit(np.array([[0.0], [1.0]])),
    }
    with pytest.raises(ValueError, match="at least 6"):
        LstmModel().predict(artifacts, {"cpu": [0.5, 0.5]})


@pytest.mark.slow
def test_lstm_predict_multi_horizon():
    """LSTM is direct multi-output — ``output_size`` is set at train
    time and equals the maximum horizon predict can serve.
    """
    pytest.importorskip("torch")
    from intelligence.ml.models.lstm import LstmModel, make_lstm_prepare

    horizon = 3
    prep = make_lstm_prepare(look_back=6, num_variables=1, batch_size=16, horizon=horizon)
    comps = prep(_synthetic_cpu(n=300))
    comps["model_parameters"] = {
        "input_size": 1,
        "output_size": horizon,
        "hidden_size": 4,
        "num_epochs": 2,
    }

    model = LstmModel()
    artifacts, _ = model.fit(comps)
    assert artifacts["output_size"] == horizon

    out = model.predict(
        artifacts,
        {"cpu": _synthetic_cpu(n=8).iloc[-6:]["cpu"].tolist()},
        horizon=horizon,
    )
    assert isinstance(out, list) and len(out) == horizon
    for point in out:
        assert point.value is not None
        # LSTM ships without CIs (MC-dropout deferred per memory note).
        assert point.lower is None


def test_lstm_predict_rejects_horizon_greater_than_trained_output_size():
    """LSTM is direct — request horizon > trained output_size is a 422."""
    pytest.importorskip("torch")
    from sklearn.preprocessing import MinMaxScaler

    from intelligence.ml.models.lstm import LstmModel

    artifacts = {
        "network": None,  # never reached — horizon check fires first
        "look_back": 6,
        "num_variables": 1,
        "scaler_obj": MinMaxScaler().fit(np.array([[0.0], [1.0]])),
        "output_size": 2,  # trained max horizon
        "horizon": 2,
    }
    window = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    with pytest.raises(ValueError, match="horizon"):
        LstmModel().predict(artifacts, {"cpu": window}, horizon=5)


# ---- New protocol: fit / save_artifacts / load_artifacts ------------------


@pytest.fixture
def lstm_artifacts_fit():
    """A fresh ``(model, artifacts, metrics)`` triple from ``LstmModel.fit``.

    Tiny network + 2 epochs keeps the test in the seconds range.
    """
    pytest.importorskip("torch")
    from intelligence.ml.models.lstm import LstmModel, make_lstm_prepare

    prep = make_lstm_prepare(look_back=6, num_variables=1, batch_size=16)
    comps = prep(_synthetic_cpu(n=300))
    comps["model_parameters"] = {
        "input_size": 1,
        "output_size": 1,
        "hidden_size": 4,
        "num_epochs": 2,
    }
    model = LstmModel()
    artifacts, metrics = model.fit(comps)
    return model, artifacts, metrics


@pytest.mark.slow
def test_lstm_fit_returns_artifacts_and_metrics(lstm_artifacts_fit):
    _model, artifacts, metrics = lstm_artifacts_fit
    assert isinstance(artifacts, dict)
    assert isinstance(metrics, dict)
    # Everything predict needs
    assert "network" in artifacts  # the nn.Module
    assert "scaler_obj" in artifacts
    assert "look_back" in artifacts
    assert "num_variables" in artifacts
    assert "input_size" in artifacts and "hidden_size" in artifacts
    assert "output_size" in artifacts


@pytest.mark.slow
def test_lstm_save_artifacts_writes_safetensors_no_pickle(lstm_artifacts_fit, tmp_path):
    model, artifacts, _ = lstm_artifacts_fit
    files = model.save_artifacts(artifacts, tmp_path)

    # safetensors holds the state_dict — no pickle anywhere.
    assert files["model"] == "lstm.safetensors"
    assert (tmp_path / "lstm.safetensors").exists()
    # arch.json carries the constructor arguments needed to rebuild
    # the nn.Module before loading the state_dict into it.
    assert (tmp_path / "arch.json").exists()
    assert (tmp_path / "scaler.json").exists()
    assert (tmp_path / "scaler.npz").exists()
    assert (tmp_path / "metrics.json").exists()

    assert not list(tmp_path.glob("*.pkl"))
    assert not list(tmp_path.glob("*.pickle"))
    assert not list(tmp_path.glob("*.pt"))  # torch.save default — also pickle-backed
    assert not list(tmp_path.glob("*.pth"))


@pytest.mark.slow
def test_lstm_save_artifacts_includes_input_spec_when_present(lstm_artifacts_fit, tmp_path):
    from intelligence.tasks.contracts import InputSpec

    model, artifacts, _ = lstm_artifacts_fit
    artifacts["input_spec"] = InputSpec(
        n_features=1, feature_names=["cpu"], steps_back=6, max_horizon=1
    )
    files = model.save_artifacts(artifacts, tmp_path)
    assert files.get("input_spec") == "input_spec.json"
    assert (tmp_path / "input_spec.json").exists()


@pytest.mark.slow
def test_lstm_load_artifacts_round_trips_full_state(lstm_artifacts_fit, tmp_path):
    """Round-trip equivalence: the loaded network produces identical
    forward-pass output for the same input."""
    import torch

    model, artifacts, _ = lstm_artifacts_fit
    model.save_artifacts(artifacts, tmp_path)
    loaded = model.load_artifacts(tmp_path)

    assert {"network", "scaler_obj", "look_back", "num_variables"} <= set(loaded.keys())
    assert loaded["look_back"] == artifacts["look_back"]
    assert loaded["output_size"] == artifacts["output_size"]

    # Numerical equivalence: same input through both networks → same output.
    artifacts["network"].eval()
    loaded["network"].eval()
    x = torch.zeros(1, artifacts["look_back"], artifacts["num_variables"])
    with torch.no_grad():
        y_orig = artifacts["network"](x).numpy()
        y_load = loaded["network"](x).numpy()
    np.testing.assert_allclose(y_orig, y_load, rtol=1e-5)


@pytest.mark.slow
def test_lstm_load_artifacts_restores_input_spec(lstm_artifacts_fit, tmp_path):
    from intelligence.tasks.contracts import InputSpec

    model, artifacts, _ = lstm_artifacts_fit
    artifacts["input_spec"] = InputSpec(
        n_features=1, feature_names=["cpu"], steps_back=6, max_horizon=1
    )
    model.save_artifacts(artifacts, tmp_path)

    loaded = model.load_artifacts(tmp_path)
    assert isinstance(loaded["input_spec"], InputSpec)
    assert loaded["input_spec"].feature_names == ["cpu"]


@pytest.mark.slow
def test_lstm_files_map_declares_only_safe_extensions(lstm_artifacts_fit, tmp_path):
    from pathlib import Path

    from intelligence.ml.artifact.manifest import ALLOWED_EXTENSIONS

    model, artifacts, _ = lstm_artifacts_fit
    files = model.save_artifacts(artifacts, tmp_path)
    for role, fname in files.items():
        assert Path(fname).suffix.lower() in ALLOWED_EXTENSIONS, (
            f"role {role!r} declares {fname!r} with disallowed extension"
        )
