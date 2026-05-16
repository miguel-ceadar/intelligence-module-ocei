"""Explicit input contract per task.

InputSpec captures shape (n_features, steps_back, feature_names) and
optional value_range / units. Validation happens at the API boundary so
mismatches return 422, not silent garbage from a misaligned scaler.
"""

from __future__ import annotations

import pytest

from intelligence.tasks.contracts import ContractViolation, InputSpec


def test_input_spec_is_constructable():
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
    spec = InputSpec(n_features=2, feature_names=["cpu", "mem"], steps_back=6)
    with pytest.raises(ContractViolation, match="features"):
        spec.validate({"cpu": [0.5] * 6})  # missing 'mem'


def test_input_spec_rejects_wrong_steps_back():
    spec = InputSpec(n_features=1, feature_names=["cpu"], steps_back=6)
    with pytest.raises(ContractViolation, match=r"steps|window|length"):
        spec.validate({"cpu": [0.5] * 4})


def test_input_spec_rejects_out_of_range():
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
    spec = InputSpec(n_features=1, feature_names=["cpu"], steps_back=3)
    with pytest.raises(ContractViolation, match=r"not finite"):
        spec.validate({"cpu": [0.5, float("inf"), 0.5]})
    with pytest.raises(ContractViolation, match=r"not finite"):
        spec.validate({"cpu": [0.5, float("-inf"), 0.5]})


def test_input_spec_rejects_non_numeric_values():
    """A string or None slipping through pydantic's parsing should fail
    at the contract boundary with a clear message rather than at
    float(v) deep in the scaler."""
    spec = InputSpec(n_features=1, feature_names=["cpu"], steps_back=3)
    with pytest.raises(ContractViolation, match=r"not numeric"):
        spec.validate({"cpu": [0.5, "oops", 0.5]})  # type: ignore[list-item]


def test_input_spec_passes_valid_input():
    spec = InputSpec(
        n_features=1,
        feature_names=["cpu"],
        steps_back=6,
        value_range={"cpu": (0.0, 1.0)},
    )
    spec.validate({"cpu": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})  # no exception


def test_input_spec_rejects_zero_n_features():
    """A spec with ``n_features=0`` would accept any input as valid (the
    empty-dict request would match), bypassing the whole contract. Reject
    at construction so a tampered or mis-written ``input_spec.json``
    can't slip past."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="n_features"):
        InputSpec(n_features=0, feature_names=["cpu"], steps_back=6)


def test_input_spec_rejects_empty_feature_names():
    """Symmetric guard to ``n_features > 0``: ``feature_names=[]`` is the
    same loophole expressed through the other field."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="feature_names"):
        InputSpec(n_features=1, feature_names=[], steps_back=6)


def test_input_spec_rejects_count_name_mismatch():
    """``n_features`` and ``len(feature_names)`` encode the same fact; a
    spec where they disagree is corrupt. Caught by a model_validator
    that pydantic's field-level constraints can't express."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match=r"n_features|feature_names"):
        InputSpec(n_features=2, feature_names=["cpu"], steps_back=6)


def test_input_spec_json_roundtrip():
    """``InputSpec`` is persisted as ``input_spec.json`` via pydantic's
    JSON round-trip (not pickle). Tuples in ``value_range`` must survive
    the trip so the predict path can still destructure ``(lo, hi)``."""
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
