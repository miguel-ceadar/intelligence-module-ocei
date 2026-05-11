"""``ForecastTask`` — generic time-series forecast task.

Composes a data loader (where to get training data) with a model
adapter (which ML algorithm). Adding a new (domain × model) combination
is a one-line factory in ``catalog.py`` — no new class required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from intelligence.api.schemas import (
    PredictRequest,
    PredictResponse,
    StaticDataSource,
    TrainRequest,
    TrainResponse,
)
from intelligence.models.base import ModelAdapter

logger = logging.getLogger(__name__)


@dataclass
class ForecastTask:
    """Time-series forecast task.

    Attributes:
        name: URL segment under ``/tasks/{name}/...``.
        model_adapter: train/predict implementation for one ML algorithm.
        data_loader: ``StaticDataSource → dict`` (training components).
        bento_name: BentoML storage key. Kept explicit so legacy Bentos
            can be reused during phase 1's transition (see
            ``intelligence.tasks.catalog``).
    """

    name: str
    model_adapter: ModelAdapter
    data_loader: Callable[[StaticDataSource], dict]
    bento_name: str
    _cached_model: Any = field(default=None, init=False, repr=False)

    @property
    def model_type(self) -> str:
        return self.model_adapter.name

    @property
    def has_drift(self) -> bool:
        return bool(getattr(self.model_adapter, "has_drift", False))

    def is_loaded(self) -> bool:
        return self._cached_model is not None

    def is_ready(self) -> tuple[bool, str]:
        """Readiness probe: can the task serve requests right now?

        Delegates to the data_loader's ``is_ready`` if it exposes one
        (e.g. ``StaticCsvLoader`` checks the dataset directory exists).
        Phase 2 will also include a model-loadable check for tasks
        configured with ``bootstrap.auto_train_on_startup=true``.
        """
        loader_check = getattr(self.data_loader, "is_ready", None)
        if loader_check is not None:
            try:
                ok, msg = loader_check()
                if not ok:
                    return False, f"data_loader: {msg}"
            except Exception as e:
                return False, f"data_loader probe raised: {e}"
        return True, "ok"

    def _load_model(self) -> Any:
        if self._cached_model is not None:
            return self._cached_model
        import bentoml
        try:
            self._cached_model = bentoml.picklable_model.get(f"{self.bento_name}:latest")
        except bentoml.exceptions.NotFound:
            self._cached_model = None
        return self._cached_model

    def _invalidate(self) -> None:
        self._cached_model = None

    def train(self, req: TrainRequest) -> TrainResponse:
        if not isinstance(req.data_source, StaticDataSource):
            raise NotImplementedError(
                f"phase 1 supports kind='static' only; got kind={req.data_source.kind!r}"
            )
        components = self.data_loader(req.data_source)
        components["model_parameters"] = req.model_parameters
        bento, metrics = self.model_adapter.train(components, self.bento_name)
        self._invalidate()
        return TrainResponse(model_tag=str(bento.tag), metrics=metrics)

    def predict(self, req: PredictRequest) -> PredictResponse:
        model = self._load_model()
        if model is None:
            raise FileNotFoundError(
                f"no trained model for {self.name}; "
                f"POST /tasks/{self.name}/train first"
            )
        prediction = self.model_adapter.predict(model, req.input_series)
        return PredictResponse(prediction=prediction)
