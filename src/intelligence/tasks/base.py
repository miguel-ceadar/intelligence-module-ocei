"""Task protocol, registry, and the concrete ``BaseTask``.

A ``Task`` exposes ``name``, ``train(req)``, ``predict(req)``,
``is_loaded()``. ``BaseTask`` composes a data loader with a model and
handles caching, lazy load, and readiness. Tasks are looked up by
name; ``build_registry_from_config`` dispatches each typed config
block to a per-kind builder.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from intelligence.api.schemas import (
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
)
from intelligence.ml.artifact import get_artifact_by_tag, save_artifact
from intelligence.ml.models import Model
from intelligence.tasks.contracts import InputSpec

if TYPE_CHECKING:
    from intelligence.api.schemas import DataSource
    from intelligence.config.settings import IntelligenceConfig

logger = logging.getLogger(__name__)

# Cap on per-task cached artifact entries. Realistic upper bound for
# pinned/rollback candidates plus ``:latest`` — well above any normal
# workload and small enough that an adversarial probe of distinct
# versions can't grow memory unboundedly.
_CACHE_MAXSIZE = 8


@runtime_checkable
class Task(Protocol):
    """Duck-typed task contract. Concrete tasks may expose more
    attributes (``model_type``, ``has_drift``, ``input_spec``); the
    registry surfaces them via ``getattr`` defaults."""

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


@dataclass
class BaseTask:
    """Composes a data loader with a model.

    Works for any (data source x model algorithm) pairing as long as
    both follow their contracts. Subclass only when the task needs
    different lifecycle behaviour (see ``DriftDetectionTask``).

    Attributes:
        name: URL segment under ``/tasks/{name}/...``.
        model: implementation of the ``Model`` protocol, or ``None``
            for tasks (like drift) that handle persistence themselves.
        data_loader: maps a ``DataSource`` to training components.
        bento_name: artifact-store key; defaults to ``name``.
    """

    name: str
    model: Model | None
    data_loader: Callable[[DataSource], dict]
    bento_name: str | None = None
    input_spec: InputSpec | None = None
    # When True, predict still serves an artifact whose stored
    # input_spec is missing or doesn't match the task's spec.
    allow_unverified_models: bool = False
    pinned_version: str | None = None
    # LRU cache of resolved artifacts, keyed by version string.
    # ``"latest"`` is invalidated on each train; pinned versions are
    # immutable. Bounded so a client probing many distinct
    # ``model_version`` values can't grow RSS without limit — once we
    # exceed ``_CACHE_MAXSIZE``, the least-recently-used entry is
    # evicted (it'll reload from the BentoML store on next access).
    _cached_artifacts: OrderedDict[str, tuple[dict, str]] = field(
        default_factory=OrderedDict, init=False, repr=False
    )
    bootstrap_state: str = field(default="pending", init=False, repr=False)
    bootstrap_error: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.bento_name is None:
            self.bento_name = self.name

    @property
    def model_type(self) -> str:
        return getattr(self.model, "name", "none") if self.model is not None else "none"

    @property
    def has_drift(self) -> bool:
        return bool(getattr(self.model, "has_drift", False)) if self.model is not None else False

    def is_loaded(self) -> bool:
        return bool(self._cached_artifacts)

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

    def _resolve_version(self, requested: str | None) -> str:
        """Resolve the version to load. Precedence: request → task pin → latest."""
        return requested or self.pinned_version or "latest"

    def _load_artifact(self, version: str | None = None) -> tuple[dict | None, str | None]:
        """Resolve a version and load its artifact into a dict.

        Returns ``(loaded_dict, served_tag)`` on success, or
        ``(None, None)`` when the artifact isn't in the local store.
        Results are cached by resolved version up to ``_CACHE_MAXSIZE``;
        the least-recently-used entry is evicted past the cap.
        """
        resolved = self._resolve_version(version)
        if resolved in self._cached_artifacts:
            self._cached_artifacts.move_to_end(resolved)
            return self._cached_artifacts[resolved]
        saved = get_artifact_by_tag(f"{self.bento_name}:{resolved}")
        if saved is None:
            return None, None
        loaded = self.model.load_artifacts(saved.path)
        self._cached_artifacts[resolved] = (loaded, saved.tag)
        if len(self._cached_artifacts) > _CACHE_MAXSIZE:
            self._cached_artifacts.popitem(last=False)
        return loaded, saved.tag

    def _invalidate(self) -> None:
        # Only ``:latest`` shifts on a new train; pinned versions are immutable.
        self._cached_artifacts.pop("latest", None)

    def train(self, req: TrainRequest) -> TrainResponse:
        # A wrong-kind data_source raises ValueError inside ``data_loader``,
        # which the API translates to HTTP 422.
        components = self.data_loader(req.data_source)
        components["model_parameters"] = req.model_parameters

        artifacts, metrics = self.model.fit(components)
        if self.input_spec is not None:
            artifacts["input_spec"] = self.input_spec

        kind = getattr(self.model, "name", "unknown")
        saved = save_artifact(
            self.bento_name,
            kind,
            lambda dest: self.model.save_artifacts(artifacts, dest),
        )
        self._invalidate()
        return TrainResponse(model_tag=saved.tag, metrics=metrics)

    def predict(self, req: PredictRequest) -> PredictResponse:
        # Validate the request before touching the store. Horizon is O(1)
        # and rejects the cheap mistake before we scan the full series.
        if self.input_spec is not None:
            self.input_spec.validate_horizon(req.horizon)
            self.input_spec.validate(req.input_series)

        loaded, served_tag = self._load_artifact(version=req.model_version)
        if loaded is None:
            resolved = self._resolve_version(req.model_version)
            raise FileNotFoundError(
                f"no Bento {self.bento_name}:{resolved} in the local store; "
                f"POST /tasks/{self.name}/train first, or pin to an existing version"
            )
        self._verify_artifact(loaded)
        prediction = self.model.predict(loaded, req.input_series, horizon=req.horizon)
        served = str(served_tag).split(":")[-1] if served_tag else None
        return PredictResponse(prediction=prediction, model_version=served)

    def _verify_artifact(self, loaded: dict) -> None:
        """Refuse predict if the loaded artifact's stored ``input_spec``
        doesn't match the task's. ``allow_unverified_models=True``
        downgrades refusal to a warning.
        """
        if self.input_spec is None:
            return

        stored = loaded.get("input_spec")
        if stored is None:
            if self.allow_unverified_models:
                logger.warning(
                    "task %s: serving unverified artifact (no input_spec)",
                    self.name,
                )
                return
            raise FileNotFoundError(
                f"artifact for task {self.name!r} has no input_spec; "
                f"train a fresh model or pull one that carries an input_spec."
            )

        mismatch = _spec_mismatch(stored, self.input_spec)
        if mismatch is not None:
            if self.allow_unverified_models:
                logger.warning(
                    "task %s: serving artifact with mismatched input_spec (%s)",
                    self.name,
                    mismatch,
                )
                return
            raise FileNotFoundError(
                f"artifact for task {self.name!r} has a mismatched input_spec: "
                f"{mismatch}. Train a fresh model or pull a matching one."
            )


def _spec_mismatch(stored: InputSpec, expected: InputSpec) -> str | None:
    """First shape-affecting field that disagrees, or ``None`` if all
    three (``n_features``, ``feature_names``, ``steps_back``) match.
    """
    if stored.n_features != expected.n_features:
        return f"n_features {stored.n_features} != {expected.n_features}"
    if list(stored.feature_names) != list(expected.feature_names):
        return f"feature_names {list(stored.feature_names)} != {list(expected.feature_names)}"
    if stored.steps_back != expected.steps_back:
        return f"steps_back {stored.steps_back} != {expected.steps_back}"
    return None


def build_registry_from_config(cfg: IntelligenceConfig) -> TaskRegistry:
    """Build a registry by dispatching each ``cfg.tasks`` block to its
    per-kind builder. Heavy imports happen inside each builder, so
    unused kinds don't pull their model dependencies.
    """
    from intelligence.tasks.builders import get_builder

    reg = TaskRegistry()
    for name, task_cfg in cfg.tasks.items():
        builder = get_builder(task_cfg.kind)
        reg.register(builder(name, task_cfg, cfg))
    return reg
