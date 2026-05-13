"""Phase-1 §2.4: explicit input contract per task.

InputSpec captures shape (n_features, steps_back, feature_names) and
optional value_range / units. Validation happens at the API boundary so
mismatches return 422, not silent garbage from a misaligned scaler.
"""

from __future__ import annotations

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


def test_input_spec_rejects_nan_values():
    """NaN bypasses ``<``/``>`` comparisons silently, so a NaN would
    pass the value_range check and crash the scaler downstream with an
    opaque error. Reject explicitly at the contract boundary."""
    InputSpec = _import_or_skip("InputSpec")
    ContractViolation = _import_or_skip("ContractViolation")
    spec = InputSpec(
        n_features=1,
        feature_names=["cpu"],
        steps_back=6,
        value_range={"cpu": (0.0, 1.0)},
    )
    with pytest.raises(ContractViolation, match=r"not finite"):
        spec.validate({"cpu": [0.1, 0.2, float("nan"), 0.4, 0.5, 0.6]})


def test_input_spec_rejects_inf_values():
    """+/-Inf would technically be caught by the value_range check but
    with a confusing 'outside range' message. Reject explicitly so the
    operator sees 'not finite' instead."""
    InputSpec = _import_or_skip("InputSpec")
    ContractViolation = _import_or_skip("ContractViolation")
    spec = InputSpec(n_features=1, feature_names=["cpu"], steps_back=3)
    with pytest.raises(ContractViolation, match=r"not finite"):
        spec.validate({"cpu": [0.5, float("inf"), 0.5]})
    with pytest.raises(ContractViolation, match=r"not finite"):
        spec.validate({"cpu": [0.5, float("-inf"), 0.5]})


def test_input_spec_rejects_non_numeric_values():
    """A string or None slipping through pydantic's parsing should fail
    at the contract boundary with a clear message rather than at
    float(v) deep in the scaler."""
    InputSpec = _import_or_skip("InputSpec")
    ContractViolation = _import_or_skip("ContractViolation")
    spec = InputSpec(n_features=1, feature_names=["cpu"], steps_back=3)
    with pytest.raises(ContractViolation, match=r"not numeric"):
        spec.validate({"cpu": [0.5, "oops", 0.5]})  # type: ignore[list-item]


def test_input_spec_passes_valid_input():
    InputSpec = _import_or_skip("InputSpec")
    spec = InputSpec(
        n_features=1,
        feature_names=["cpu"],
        steps_back=6,
        value_range={"cpu": (0.0, 1.0)},
    )
    spec.validate({"cpu": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})  # no exception


def test_input_spec_json_roundtrip():
    """``InputSpec`` is persisted as ``input_spec.json`` via pydantic's
    JSON round-trip (not pickle). Tuples in ``value_range`` must survive
    the trip so the predict path can still destructure ``(lo, hi)``."""
    InputSpec = _import_or_skip("InputSpec")
    spec = InputSpec(
        n_features=2,
        feature_names=["a", "b"],
        steps_back=6,
        value_range={"a": (0.0, 1.0)},
    )
    restored = InputSpec.model_validate_json(spec.model_dump_json())
    assert restored == spec
    # Tuple in value_range is preserved (pydantic restores list → tuple).
    assert restored.value_range["a"] == (0.0, 1.0)
