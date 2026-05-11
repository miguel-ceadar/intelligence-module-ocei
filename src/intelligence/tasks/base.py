"""Task protocol + registry.

A ``Task`` is a registered capability — a name, a ``train(req)``, and a
``predict(req)``. Concrete implementations (``ForecastTask`` in phase 1;
classification / anomaly task types later) compose data loaders with
model adapters so the (task × model) matrix doesn't explode into
subclasses.

Tasks are looked up by name. ``build_registry_from_config`` constructs
a registry from a list of names by calling factories registered via
``@register_builtin`` (see ``catalog.py``).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Protocol, runtime_checkable

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
