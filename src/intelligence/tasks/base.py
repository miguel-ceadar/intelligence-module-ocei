"""Task protocol + registry + the concrete ``BaseTask`` class.

A ``Task`` (the Protocol) is the contract — anything in the registry has
``name``, ``train(req)``, ``predict(req)``, ``is_loaded()``. Most tasks
will be instances of the concrete ``BaseTask`` dataclass below, which
composes ``(data_loader, model_adapter)`` and handles the lifecycle —
caching, lazy load, readiness probing, request glue.

If a task type ever needs different lifecycle (e.g. anomaly tasks want
a custom drift wiring), subclass ``BaseTask`` and override the relevant
method. Don't subclass eagerly — empty subclasses are noise.

Tasks are looked up by name. ``build_registry_from_config`` constructs
a registry from a list of names by calling factories registered via
``@register_builtin`` (see ``catalog.py``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from intelligence.api.schemas import (
    PredictRequest,
    PredictResponse,
    StaticDataSource,
    TrainRequest,
    TrainResponse,
)
from intelligence.models.base import ModelAdapter

logger = logging.getLogger(__name__)


@runtime_checkable
class Task(Protocol):
    """Duck-typed task contract.

    Concrete tasks may expose more (``model_type``, ``has_drift``,
    ``input_spec``, ``drift``); the registry surfaces what's there via
    ``getattr`` defaults.
    """

    name: str

    def train(self, req: Any) -> Any: ...
    def predict(self, req: Any) -> Any: ...
    def is_loaded(self) -> bool: ...


class TaskRegistry:
    """Name → Task lookup. Cheap to introspect; doesn't touch any model."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def register(self, task: Task) -> None:
        if task.name in self._tasks:
            raise ValueError(f"task already registered: {task.name}")
        self._tasks[task.name] = task

    def get(self, name: str) -> Task:
        if name not in self._tasks:
            raise KeyError(name)
        return self._tasks[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tasks

    def __iter__(self):
        return iter(self._tasks)

    def __len__(self) -> int:
        return len(self._tasks)

    def list_info(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "model_type": getattr(t, "model_type", "unknown"),
                "has_drift": bool(getattr(t, "has_drift", False)),
                "is_loaded": _safely(getattr(t, "is_loaded", lambda: False)),
            }
            for t in self._tasks.values()
        ]


def _safely(fn) -> bool:
    try:
        return bool(fn())
    except Exception:
        return False


# ---- Concrete task --------------------------------------------------------


@dataclass
class BaseTask:
    """Generic task: composes a data loader with a model adapter.

    Works for any (task domain × model algorithm) pairing — forecast,
    anomaly, classification — as long as the adapter and loader follow
    their contracts. Subclass only when a task type needs different
    lifecycle (e.g. a custom drift method); don't subclass for naming.

    Attributes:
        name: URL segment under ``/tasks/{name}/...``.
        model_adapter: train/predict implementation for one ML algorithm.
        data_loader: maps a ``StaticDataSource`` to training components.
        bento_name: BentoML storage key. Defaults to ``name``; override
            only to share a Bento name with legacy code.
    """

    name: str
    model_adapter: ModelAdapter
    data_loader: Callable[[StaticDataSource], dict]
    bento_name: str | None = None
    _cached_model: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.bento_name is None:
            self.bento_name = self.name

    @property
    def model_type(self) -> str:
        return self.model_adapter.name

    @property
    def has_drift(self) -> bool:
        return bool(getattr(self.model_adapter, "has_drift", False))

    def is_loaded(self) -> bool:
        return self._cached_model is not None

    def is_ready(self) -> tuple[bool, str]:
        """Readiness probe — delegates to the data_loader if it has one."""
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


# ---- Builtin factory catalog ------------------------------------------

_BUILTIN_FACTORIES: dict[str, Callable[[], Task]] = {}


def register_builtin(name: str) -> Callable[[Callable[[], Task]], Callable[[], Task]]:
    """Decorator: register a factory function under ``name``.

    The factory body is what runs at registry-build time (which is when
    the dependency imports happen). Defining a factory does not import
    anything heavy.
    """

    def wrap(factory: Callable[[], Task]) -> Callable[[], Task]:
        if name in _BUILTIN_FACTORIES:
            raise ValueError(f"builtin task factory already registered: {name}")
        _BUILTIN_FACTORIES[name] = factory
        return factory

    return wrap


def builtin_task_factory(name: str) -> Callable[[], Task]:
    if name not in _BUILTIN_FACTORIES:
        raise KeyError(f"no builtin task named: {name}")
    return _BUILTIN_FACTORIES[name]


def build_registry_from_config(enabled_tasks: list[str]) -> TaskRegistry:
    """Build a registry containing exactly the named tasks, in order.

    Imports ``catalog.py`` lazily — that module's ``@register_builtin``
    decorators populate ``_BUILTIN_FACTORIES``. Each factory body lazy-
    imports its model adapter, so unconfigured tasks don't pull their deps.
    """
    import intelligence.tasks.catalog  # noqa: F401  populates _BUILTIN_FACTORIES

    reg = TaskRegistry()
    for name in enabled_tasks:
        factory = builtin_task_factory(name)
        reg.register(factory())
    return reg
