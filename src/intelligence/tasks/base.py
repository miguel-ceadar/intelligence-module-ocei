"""Task protocol + registry + the concrete ``BaseTask`` class.

A ``Task`` (the Protocol) is the contract — anything in the registry has
``name``, ``train(req)``, ``predict(req)``, ``is_loaded()``. Most tasks
will be instances of the concrete ``BaseTask`` dataclass below, which
composes ``(data_loader, model)`` and handles the lifecycle — caching,
lazy load, readiness probing, request glue.

If a task type ever needs different lifecycle (e.g. anomaly tasks want
a custom drift wiring), subclass ``BaseTask`` and override the relevant
method. Don't subclass eagerly — empty subclasses are noise.

Tasks are looked up by name. ``build_registry_from_config`` walks the
typed ``cfg.tasks`` dict and dispatches each block to a per-kind
builder under ``intelligence.tasks.builders``.
"""

from __future__ import annotations

import logging
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
    """Generic task: composes a data loader with a model.

    Works for any (task domain x model algorithm) pairing — forecast,
    anomaly, classification — as long as the model and loader follow
    their contracts. Subclass only when a task type needs different
    lifecycle (e.g. a custom drift method); don't subclass for naming.

    Attributes:
        name: URL segment under ``/tasks/{name}/...``.
        model: the per-algorithm implementation of the ``Model``
            protocol (``fit`` / ``save_artifacts`` / ``load_artifacts``
            / ``predict``).
        data_loader: maps a ``DataSource`` to training components.
        bento_name: artefact-store key. Defaults to ``name``; override
            only to share a name with legacy code.
    """

    name: str
    # ``model`` is the per-algorithm Model implementation. Optional
    # because subclasses (e.g. ``DriftDetectionTask``) override both
    # train and predict and don't use a Model at all.
    model: Model | None
    data_loader: Callable[[DataSource], dict]
    bento_name: str | None = None
    input_spec: InputSpec | None = None
    # When True, predict serves artefacts whose stored input_spec is
    # missing or doesn't match the task's spec. Default False — pulled
    # artefacts that predate the contract are refused.
    allow_unverified_models: bool = False
    # Pin this task's predict path to a specific version. A request's
    # ``model_version`` overrides this.
    pinned_version: str | None = None
    # Cache resolved artefacts by version string, holding ``(loaded_dict,
    # served_tag)``. ``"latest"`` is invalidated on each train (a new
    # train shifts what ``latest`` means); pinned versions are immutable
    # so their cache entries stay valid.
    _cached_artifacts: dict[str, tuple[dict, str]] = field(
        default_factory=dict, init=False, repr=False
    )
    # Bootstrap-on-startup state (see ``intelligence.tasks.bootstrap``).
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
        """Resolve a version and load its artefact into a dict.

        Returns ``(loaded_dict, served_tag)`` on success, or
        ``(None, None)`` when the artefact isn't in the local store.
        The loaded dict is cached by the resolved version; ``latest``
        is invalidated by :meth:`_invalidate` whenever ``train`` runs.
        """
        resolved = self._resolve_version(version)
        if resolved in self._cached_artifacts:
            return self._cached_artifacts[resolved]
        saved = get_artifact_by_tag(f"{self.bento_name}:{resolved}")
        if saved is None:
            return None, None
        loaded = self.model.load_artifacts(saved.path)
        self._cached_artifacts[resolved] = (loaded, saved.tag)
        return loaded, saved.tag

    def _invalidate(self) -> None:
        # A new train only shifts what ``:latest`` means; pinned versions
        # are immutable so their cache entries stay valid.
        self._cached_artifacts.pop("latest", None)

    def train(self, req: TrainRequest) -> TrainResponse:
        # Descriptor dispatch is the loader's job — wrong kind raises
        # ValueError inside ``data_loader``, which the API translates to 422.
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
        # Validate request against the contract BEFORE loading the
        # artefact — cheap rejection of obviously-bad requests, no
        # store fetch needed.
        if self.input_spec is not None:
            self.input_spec.validate(req.input_series)
            self.input_spec.validate_horizon(req.horizon)

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
        """Refuse predict if the loaded artefact's stored contract
        doesn't match this task's ``input_spec``.

        The contract is what was written to ``input_spec.json`` at
        train time (see :meth:`train` → ``model.save_artifacts``).
        Pulled artefacts from before the contract existed have no
        ``input_spec`` and are refused by default.

        ``allow_unverified_models=True`` downgrades refusal to a
        warning. Operationally a refusal is the same shape as "no
        trained model" (503 from the API), so the caller knows to
        train fresh or pull a matching artefact.
        """
        if self.input_spec is None:
            return  # task has no contract to verify against

        stored = loaded.get("input_spec")
        if stored is None:
            if self.allow_unverified_models:
                logger.warning(
                    "task %s: serving unverified artefact (no input_spec)",
                    self.name,
                )
                return
            raise FileNotFoundError(
                f"unverified Bento for task {self.name!r}: stored artefact has no "
                f"input_spec (saved before the contract existed). Train a fresh "
                f"model or set allow_unverified_models=true (debugging only)."
            )

        mismatch = _spec_mismatch(stored, self.input_spec)
        if mismatch is not None:
            if self.allow_unverified_models:
                logger.warning(
                    "task %s: serving artefact with mismatched input_spec (%s)",
                    self.name,
                    mismatch,
                )
                return
            raise FileNotFoundError(
                f"Bento for task {self.name!r} has a mismatched input_spec: "
                f"{mismatch}. Train a fresh model or set allow_unverified_models=true."
            )


def _spec_mismatch(stored: InputSpec, expected: InputSpec) -> str | None:
    """Return a short description of the first structural field that
    doesn't match, or ``None`` if all three agree.

    Only the fields that affect tensor shape are compared —
    ``n_features``, ``feature_names``, ``steps_back``. ``value_range``
    and ``units`` are descriptive and don't block.
    """
    if stored.n_features != expected.n_features:
        return f"n_features {stored.n_features} != {expected.n_features}"
    if list(stored.feature_names) != list(expected.feature_names):
        return f"feature_names {list(stored.feature_names)} != {list(expected.feature_names)}"
    if stored.steps_back != expected.steps_back:
        return f"steps_back {stored.steps_back} != {expected.steps_back}"
    return None


def build_registry_from_config(cfg: IntelligenceConfig) -> TaskRegistry:
    """Build a registry from the typed ``cfg.tasks`` dict.

    Iterates each ``(name, task_cfg)`` pair, dispatches on
    ``task_cfg.kind`` via the builder registry, and registers the
    resulting ``BaseTask``. Each builder body is where heavy imports
    happen — unused kinds don't pull their model deps.
    """
    from intelligence.tasks.builders import get_builder

    reg = TaskRegistry()
    for name, task_cfg in cfg.tasks.items():
        builder = get_builder(task_cfg.kind)
        reg.register(builder(name, task_cfg, cfg))
    return reg
