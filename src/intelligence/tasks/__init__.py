"""Task abstraction — see ``base.Task`` (Protocol) and ``base.BaseTask`` (concrete).

Concrete task instances are built by factories in ``catalog``. Importing
this package does NOT trigger the catalog; that happens lazily when
``build_registry_from_config`` is called.
"""

from intelligence.tasks.base import (
    BaseTask,
    Task,
    TaskRegistry,
    build_registry_from_config,
    builtin_task_factory,
    register_builtin,
)

__all__ = [
    "BaseTask",
    "Task",
    "TaskRegistry",
    "build_registry_from_config",
    "builtin_task_factory",
    "register_builtin",
]
