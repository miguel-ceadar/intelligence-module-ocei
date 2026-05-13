"""``InputSpec`` — explicit input contract per task.

Today every model carries an *implicit* contract via its saved scaler
plus shape assumptions baked into the relevant predict path. Without
validation, mismatches either crash mid-numpy with a shape error or
produce silent garbage from a misaligned scaler.

``InputSpec`` makes the contract explicit. Each task carries one; the
API validates incoming requests against it before dispatching to the
runner. Mismatches return 422 with a useful message.

The spec is also written into the saved Bento's ``custom_objects`` at
train time, so future runs (or pulled Bentos) carry their contract
with them.
"""

from __future__ import annotations

import math

from pydantic import BaseModel


class ContractViolation(ValueError):
    """Raised when a request violates a task's ``InputSpec``.

    Subclasses ``ValueError`` so the API service's existing error
    handler translates it to HTTP 422.
    """


class InputSpec(BaseModel):
    """Per-task input contract.

    Captures shape (``n_features``, ``feature_names``, ``steps_back``)
    and optional semantic metadata (``value_range``, ``units``) that
    catches unit mismatches — e.g. passing CPU as a percent (87.3) when
    the model was trained on fractional CPU (0.87).

    ``max_horizon`` clamps how many steps ahead the task can serve.
    ``None`` is unbounded (ARIMA refits each call; XGB recursive walks
    forward as long as requested). LSTM is direct multi-output and sets
    this to the trained ``output_size`` — request horizons above that
    are refused at the API boundary.
    """

    n_features: int
    feature_names: list[str]
    steps_back: int
    max_horizon: int | None = None
    value_range: dict[str, tuple[float, float]] = {}
    units: dict[str, str] = {}

    def validate(self, input_series: dict[str, list[float]]) -> None:
        """Raise ``ContractViolation`` on shape, name, or range mismatch.

        Note: this overrides pydantic's deprecated v1-style ``validate``
        classmethod alias. The instance method we want is ``check``ing
        runtime input — pydantic's class-level data validation is unaffected.
        """
        # Feature count
        if len(input_series) != self.n_features:
            raise ContractViolation(
                f"expected {self.n_features} features ({self.feature_names}), "
                f"got {len(input_series)} ({list(input_series.keys())})"
            )

        # Feature names (set comparison)
        expected = set(self.feature_names)
        actual = set(input_series.keys())
        if expected != actual:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            parts = []
            if missing:
                parts.append(f"missing features: {missing}")
            if unexpected:
                parts.append(f"unexpected features: {unexpected}")
            raise ContractViolation(f"feature names mismatch — {'; '.join(parts)}")

        # Steps_back (window length per feature)
        for name, values in input_series.items():
            if len(values) != self.steps_back:
                raise ContractViolation(
                    f"feature {name!r}: expected steps_back={self.steps_back} "
                    f"timesteps, got window length {len(values)}"
                )

        # Numeric finiteness. NaN bypasses <, > comparisons silently, so
        # the value_range check below would let it through; +/-Inf would
        # be caught there but the message would be confusing. Reject both
        # explicitly before the semantic check.
        for name, values in input_series.items():
            for i, v in enumerate(values):
                try:
                    finite = math.isfinite(float(v))
                except (TypeError, ValueError) as e:
                    raise ContractViolation(
                        f"feature {name!r} value {v!r} at index {i} is not numeric"
                    ) from e
                if not finite:
                    raise ContractViolation(
                        f"feature {name!r} value {v!r} at index {i} is not finite (NaN/Inf)"
                    )

        # Value ranges (semantic check — catches unit mismatches)
        for name, (lo, hi) in self.value_range.items():
            if name not in input_series:
                continue
            for i, v in enumerate(input_series[name]):
                if v < lo or v > hi:
                    unit = self.units.get(name)
                    suffix = f" (units: {unit})" if unit else ""
                    raise ContractViolation(
                        f"feature {name!r} value {v} at index {i} outside trained "
                        f"range [{lo}, {hi}]{suffix}"
                    )

    def validate_horizon(self, horizon: int) -> None:
        """Raise ``ContractViolation`` when ``horizon`` exceeds the task's
        ``max_horizon`` clamp. ``None`` means unbounded.
        """
        if self.max_horizon is not None and horizon > self.max_horizon:
            raise ContractViolation(
                f"horizon {horizon} exceeds trained max_horizon={self.max_horizon}; "
                f"retrain with a larger output window or request a shorter horizon"
            )
