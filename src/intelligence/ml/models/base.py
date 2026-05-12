"""``Model`` protocol — train + predict for one ML algorithm family.

A ``Model`` is *stateless* with respect to the data domain. Given
prepared training components, it trains and saves a Bento. Given a saved
Bento and an input series, it predicts. The same ``Model`` can serve any
task whose data shape it supports — ``cpu_forecast``, ``mem_forecast``,
``energy_forecast`` all reuse one ``ArimaModel``.

Phase 1 ships ``ArimaModel``. ``XgbModel`` and ``LstmModel`` follow the
same contract; each is one new file plus one factory line in
``intelligence.tasks.catalog``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Model(Protocol):
    """Train one model family, persist as a Bento, predict from one."""

    name: str          # short tag — 'arima', 'xgb', 'lstm'
    has_drift: bool    # whether this model ships drift detection

    def train(
        self,
        components: dict,
        bento_name: str,
        extras: dict | None = None,
    ) -> tuple[Any, dict]:
        """Train, save Bento, return ``(bento_model, metrics_dict)``.

        ``extras`` is merged into the Bento's ``custom_objects`` — that's
        how ``BaseTask`` injects task-level metadata (e.g. ``input_spec``)
        without the model knowing what it is.
        """
        ...

    def predict(self, bento_model: Any, input_series: dict[str, list[float]]) -> Any:
        """Predict from a saved Bento + new observations."""
        ...
