"""Task abstraction — see ``base.Task`` (Protocol) and ``base.BaseTask`` (concrete).

Task instances are built from typed config blocks by per-kind builders
under ``intelligence.tasks.builders``. ``build_registry_from_config``
walks ``cfg.tasks`` and dispatches each block to its builder.
"""

from intelligence.tasks.base import (
    BaseTask,
    Task,
    TaskRegistry,
    build_registry_from_config,
)

__all__ = [
    "BaseTask",
    "Task",
    "TaskRegistry",
    "build_registry_from_config",
]
