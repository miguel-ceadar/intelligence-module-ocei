"""``ModelAdapter`` protocol — train + predict for one ML algorithm family.

A ``ModelAdapter`` is *stateless* with respect to the data domain. Given
prepared training components, it trains a model and saves a Bento.
Given a saved Bento and an input series, it predicts. The same adapter
can serve any task whose data shape it supports — ``cpu_forecast``,
``mem_forecast``, ``energy_forecast`` all reuse one ``ArimaAdapter``.

Phase 1 ships ``ArimaAdapter``. ``XgbAdapter`` and ``LstmAdapter`` follow
the same contract; each is one new file plus one factory line in
``intelligence.tasks.catalog``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ModelAdapter(Protocol):
    """Train one model family, persist as a Bento, predict from one."""

    name: str          # short tag — 'arima', 'xgb', 'lstm'
    has_drift: bool    # whether this adapter ships drift detection

    def train(
        self,
        components: dict,
        bento_name: str,
        extras: dict | None = None,
    ) -> tuple[Any, dict]:
        """Train, save Bento, return ``(bento_model, metrics_dict)``.

        ``extras`` is merged into the Bento's ``custom_objects`` — that's
        how ``BaseTask`` injects task-level metadata (e.g. ``input_spec``)
        without the adapter knowing what it is.
        """
        ...

    def predict(self, bento_model: Any, input_series: dict[str, list[float]]) -> Any:
        """Predict from a saved Bento + new observations."""
        ...
