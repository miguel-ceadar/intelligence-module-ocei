"""Typed sidecar persistence — InputSpec, sklearn scalers, JSON dicts.

These are the helpers each per-kind ``save_artifacts`` reaches for. The
guarantees that matter:

  - Numerical round-trips are exact (a scaler that's saved and loaded
    transforms identically; ``transform()`` is the only contract that
    matters at predict time).
  - No pickle anywhere. NPZ archives are opened with
    ``allow_pickle=False`` so a tampered ``scaler.npz`` cannot smuggle
    pickle in. The sklearn class is restored via a strict allowlist —
    no arbitrary import path.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from intelligence.ml.artifact.sidecars import (
    load_input_spec,
    load_json,
    load_sklearn_scaler,
    save_input_spec,
    save_json,
    save_sklearn_scaler,
)
from intelligence.tasks.contracts import InputSpec


def test_input_spec_roundtrip(tmp_path):
    spec = InputSpec(
        n_features=2,
        feature_names=["cpu", "mem"],
        steps_back=6,
        max_horizon=4,
        value_range={"cpu": (0.0, 1.0), "mem": (0.0, 1.0)},
        units={"cpu": "fraction"},
    )
    save_input_spec(tmp_path, spec)

    restored = load_input_spec(tmp_path)
    assert restored.n_features == spec.n_features
    assert restored.feature_names == spec.feature_names
    assert restored.steps_back == spec.steps_back
    assert restored.max_horizon == spec.max_horizon
    assert restored.value_range == spec.value_range
    assert restored.units == spec.units


def test_input_spec_value_range_preserves_tuple_shape(tmp_path):
    """``value_range`` is declared as ``dict[str, tuple[float, float]]`` —
    JSON-encodes lists, but pydantic must restore tuples or the API
    range-check (which destructures into ``(lo, hi)``) would behave
    differently between fresh-trained and loaded models."""
    spec = InputSpec(
        n_features=1, feature_names=["cpu"], steps_back=1,
        value_range={"cpu": (0.0, 1.0)},
    )
    save_input_spec(tmp_path, spec)
    restored = load_input_spec(tmp_path)
    lo, hi = restored.value_range["cpu"]  # must destructure cleanly
    assert (lo, hi) == (0.0, 1.0)


def test_standard_scaler_roundtrip_transforms_identically(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    scaler = StandardScaler().fit(X)

    save_sklearn_scaler(tmp_path, "scaler", scaler)
    restored = load_sklearn_scaler(tmp_path, "scaler")

    fresh_in = rng.normal(size=(10, 3))
    np.testing.assert_allclose(scaler.transform(fresh_in), restored.transform(fresh_in))


def test_minmax_scaler_roundtrip_transforms_identically(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 2))
    scaler = MinMaxScaler().fit(X)

    save_sklearn_scaler(tmp_path, "scaler", scaler)
    restored = load_sklearn_scaler(tmp_path, "scaler")

    fresh_in = rng.normal(size=(10, 2))
    np.testing.assert_allclose(scaler.transform(fresh_in), restored.transform(fresh_in))
    np.testing.assert_allclose(
        scaler.inverse_transform(scaler.transform(fresh_in)),
        restored.inverse_transform(restored.transform(fresh_in)),
    )


def test_scaler_preserves_feature_names_in(tmp_path):
    """XGB's ``scaler_X`` is fit on a DataFrame; ``feature_names_in_``
    must survive a round-trip or predict will fail with a column-name
    mismatch error."""
    import pandas as pd

    rng = np.random.default_rng(2)
    df = pd.DataFrame(rng.normal(size=(30, 3)), columns=["a", "b", "c"])
    scaler = StandardScaler().fit(df)
    assert list(scaler.feature_names_in_) == ["a", "b", "c"]

    save_sklearn_scaler(tmp_path, "scaler", scaler)
    restored = load_sklearn_scaler(tmp_path, "scaler")
    assert list(restored.feature_names_in_) == ["a", "b", "c"]


def test_scaler_writes_two_named_files(tmp_path):
    scaler = StandardScaler().fit(np.array([[1.0], [2.0], [3.0]]))
    save_sklearn_scaler(tmp_path, "scaler_y", scaler)
    assert (tmp_path / "scaler_y.json").exists()
    assert (tmp_path / "scaler_y.npz").exists()


def test_scaler_loader_rejects_unknown_class(tmp_path):
    """Tampered JSON points to a class we don't ship — refuse."""
    (tmp_path / "scaler.json").write_text(
        json.dumps({
            "class": "RobustScaler",
            "module": "sklearn.preprocessing",
            "params": {},
            "attrs": {},
        })
    )
    np.savez(tmp_path / "scaler.npz")
    with pytest.raises(ValueError, match="unsupported scaler"):
        load_sklearn_scaler(tmp_path, "scaler")


def test_scaler_loader_rejects_non_sklearn_module(tmp_path):
    """Tampered JSON points outside sklearn — refuse before importing."""
    (tmp_path / "scaler.json").write_text(
        json.dumps({
            "class": "StandardScaler",
            "module": "os",
            "params": {},
            "attrs": {},
        })
    )
    np.savez(tmp_path / "scaler.npz")
    with pytest.raises(ValueError, match="module"):
        load_sklearn_scaler(tmp_path, "scaler")


def test_npz_loaded_without_pickle(tmp_path, monkeypatch):
    """``numpy.load`` defaults to ``allow_pickle=False`` in modern NumPy,
    but we assert it explicitly — a tampered npz with object arrays
    must raise rather than execute pickled state.
    """
    scaler = StandardScaler().fit(np.array([[1.0], [2.0]]))
    save_sklearn_scaler(tmp_path, "scaler", scaler)

    # Replace the npz with one that would only load under allow_pickle=True.
    obj = np.empty((1,), dtype=object)
    obj[0] = {"sneaky": 1}
    np.savez(tmp_path / "scaler.npz", obj_array=obj)

    with pytest.raises(ValueError, match=r"(?i)pickle|object array"):
        load_sklearn_scaler(tmp_path, "scaler")


def test_json_roundtrip(tmp_path):
    data = {"alpha": 1.5, "beta": [1, 2, 3], "gamma": "x"}
    save_json(tmp_path, "metrics.json", data)
    assert load_json(tmp_path, "metrics.json") == data
