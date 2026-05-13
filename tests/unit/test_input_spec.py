"""Phase-1 §2.4: explicit input contract per task.

InputSpec captures shape (n_features, steps_back, feature_names) and
optional value_range / units. Validation happens at the API boundary so
mismatches return 422, not silent garbage from a misaligned scaler.
"""

from __future__ import annotations

import pickle

import pytest

contracts = pytest.importorskip("intelligence.tasks.contracts", reason="phase-1 §2.4 pending")


def _import_or_skip(name: str):
    obj = getattr(contracts, name, None)
    if obj is None:
        pytest.skip(f"intelligence.tasks.contracts.{name} not implemented yet")
    return obj


def test_input_spec_is_constructable():
    InputSpec = _import_or_skip("InputSpec")
    spec = InputSpec(
        n_features=2,
        feature_names=["cpu", "mem"],
        steps_back=6,
        value_range={"cpu": (0.0, 1.0), "mem": (0.0, 1.0)},
        units={"cpu": "fraction", "mem": "fraction"},
    )
    assert spec.n_features == 2
    assert spec.feature_names == ["cpu", "mem"]


def test_input_spec_rejects_wrong_feature_count():
    InputSpec = _import_or_skip("InputSpec")
    ContractViolation = _import_or_skip("ContractViolation")
    spec = InputSpec(n_features=2, feature_names=["cpu", "mem"], steps_back=6)
    with pytest.raises(ContractViolation, match="features"):
        spec.validate({"cpu": [0.5] * 6})  # missing 'mem'


def test_input_spec_rejects_wrong_steps_back():
    InputSpec = _import_or_skip("InputSpec")
    ContractViolation = _import_or_skip("ContractViolation")
    spec = InputSpec(n_features=1, feature_names=["cpu"], steps_back=6)
    with pytest.raises(ContractViolation, match=r"steps|window|length"):
        spec.validate({"cpu": [0.5] * 4})


def test_input_spec_rejects_out_of_range():
    InputSpec = _import_or_skip("InputSpec")
    ContractViolation = _import_or_skip("ContractViolation")
    spec = InputSpec(
        n_features=1,
        feature_names=["cpu"],
        steps_back=6,
        value_range={"cpu": (0.0, 1.0)},
    )
    with pytest.raises(ContractViolation, match=r"range|outside"):
        spec.validate({"cpu": [0.0, 0.5, 87.3, 0.5, 0.5, 0.5]})  # percent vs fraction


def test_input_spec_passes_valid_input():
    InputSpec = _import_or_skip("InputSpec")
    spec = InputSpec(
        n_features=1,
        feature_names=["cpu"],
        steps_back=6,
        value_range={"cpu": (0.0, 1.0)},
    )
    spec.validate({"cpu": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})  # no exception


def test_input_spec_pickle_roundtrip():
    """BentoML stashes contract objects in ``custom_objects`` via pickle."""
    InputSpec = _import_or_skip("InputSpec")
    spec = InputSpec(n_features=2, feature_names=["a", "b"], steps_back=6)
    restored = pickle.loads(pickle.dumps(spec))
    assert restored == spec
