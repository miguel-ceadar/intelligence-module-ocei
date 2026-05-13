"""Task protocol + registry + the concrete ``BaseTask`` class.

A ``Task`` (the Protocol) is the contract — anything in the registry has
``name``, ``train(req)``, ``predict(req)``, ``is_loaded()``. Most tasks
will be instances of the concrete ``BaseTask`` dataclass below, which
composes ``(data_loader, model)`` and handles the lifecycle —
caching, lazy load, readiness probing, request glue.

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
        model: train/predict implementation for one ML algorithm.
        data_loader: maps a ``StaticDataSource`` to training components.
        bento_name: BentoML storage key. Defaults to ``name``; override
            only to share a Bento name with legacy code.
    """

    name: str
    # ``model`` is the per-algorithm ``train`` + ``predict`` implementation.
    # Optional because subclasses (e.g. ``DriftDetectionTask``) may
    # override both lifecycle methods and not use a Model at all.
    model: Model | None
    data_loader: Callable[[DataSource], dict]
    bento_name: str | None = None
    input_spec: InputSpec | None = None
    # When True, predict serves Bentos whose stored input_spec is missing
    # or doesn't match the task's spec. Default False — pulled/pretrained
    # Bentos that predate the contract are refused. Override for
    # debugging or accepted-risk situations.
    allow_unverified_models: bool = False
    # Pin this task's predict path to a specific Bento version. Useful
    # for staged rollouts or rolling back a bad model without touching
    # client code. A request's ``model_version`` overrides this.
    pinned_version: str | None = None
    # Cache resolved Bentos by version string. ``"latest"`` is invalidated
    # on each train (a new train shifts what ``latest`` means); pinned
    # versions are immutable once written, so their cache entries stay.
    _cached_bentos: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    # Bootstrap-on-startup state (see ``intelligence.tasks.bootstrap``).
    # ``pending`` is the initial state; the coroutine flips it to
    # ``running`` → ``complete`` / ``failed``. ``/readyz`` only blocks
    # when a configured-as-bootstrap task is not yet ``complete``.
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
        return bool(self._cached_bentos)

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

    def _load_bento(self, version: str | None = None) -> Any:
        resolved = self._resolve_version(version)
        if resolved in self._cached_bentos:
            return self._cached_bentos[resolved]
        import bentoml

        try:
            bento = bentoml.picklable_model.get(f"{self.bento_name}:{resolved}")
        except bentoml.exceptions.NotFound:
            return None
        self._cached_bentos[resolved] = bento
        return bento

    def _invalidate(self) -> None:
        # A new train only shifts what ``:latest`` means; pinned versions
        # are immutable so their cache entries stay valid.
        self._cached_bentos.pop("latest", None)

    def train(self, req: TrainRequest) -> TrainResponse:
        # Descriptor dispatch is the loader's job — wrong kind raises
        # ValueError inside ``data_loader``, which the API translates to 422.
        components = self.data_loader(req.data_source)
        components["model_parameters"] = req.model_parameters
        # Inject the input_spec into the saved Bento's custom_objects so
        # the contract travels with the model — see plan §2.4.
        extras = {"input_spec": self.input_spec} if self.input_spec is not None else None
        bento, metrics = self.model.train(components, self.bento_name, extras)
        self._invalidate()
        return TrainResponse(model_tag=str(bento.tag), metrics=metrics)

    def predict(self, req: PredictRequest) -> PredictResponse:
        # Validate request against the contract BEFORE loading the bento —
        # cheap rejection of obviously-bad requests, no Bento fetch needed.
        if self.input_spec is not None:
            self.input_spec.validate(req.input_series)
            self.input_spec.validate_horizon(req.horizon)

        bento = self._load_bento(version=req.model_version)
        if bento is None:
            resolved = self._resolve_version(req.model_version)
            raise FileNotFoundError(
                f"no Bento {self.bento_name}:{resolved} in the local store; "
                f"POST /tasks/{self.name}/train first, or pin to an existing version"
            )
        self._verify_bento(bento)
        prediction = self.model.predict(bento, req.input_series, horizon=req.horizon)
        served = str(getattr(bento, "tag", "")).split(":")[-1] or None
        return PredictResponse(prediction=prediction, model_version=served)

    def _verify_bento(self, bento: Any) -> None:
        """Refuse predict if the loaded Bento's stored contract doesn't
        match this task's ``input_spec``.

        The contract is what was written to ``custom_objects['input_spec']``
        at train time (see ``BaseTask.train`` → ``Model.train(..., extras)``).
        Pulled HF Bentos from before the contract existed have no
        ``input_spec`` and are refused by default.

        ``allow_unverified_models=True`` downgrades refusal to a warning.
        Operationally a refusal is the same shape as "no trained model"
        (503 from the API), so the caller knows to train fresh or pull
        a matching Bento.
        """
        if self.input_spec is None:
            return  # task has no contract to verify against

        stored = getattr(bento, "custom_objects", {}).get("input_spec")
        if stored is None:
            if self.allow_unverified_models:
                logger.warning(
                    "task %s: serving unverified Bento (no input_spec in custom_objects)",
                    self.name,
                )
                return
            raise FileNotFoundError(
                f"unverified Bento for task {self.name!r}: stored model has no "
                f"input_spec (saved before the contract existed). Train a fresh "
                f"model or set allow_unverified_models=true (debugging only)."
            )

        mismatch = _spec_mismatch(stored, self.input_spec)
        if mismatch is not None:
            if self.allow_unverified_models:
                logger.warning(
                    "task %s: serving Bento with mismatched input_spec (%s)",
                    self.name,
                    mismatch,
                )
                return
            raise FileNotFoundError(
                f"Bento for task {self.name!r} has a mismatched input_spec: "
                f"{mismatch}. Train a fresh model or set allow_unverified_models=true."
            )


def _spec_fields(spec: Any) -> tuple[int | None, list[str] | None, int | None]:
    """Extract ``(n_features, feature_names, steps_back)`` from either an
    ``InputSpec`` instance or a plain dict (older Bentos may have pickled
    the spec as a dict).
    """
    if isinstance(spec, dict):
        return (
            spec.get("n_features"),
            list(spec["feature_names"]) if spec.get("feature_names") is not None else None,
            spec.get("steps_back"),
        )
    return (
        getattr(spec, "n_features", None),
        list(getattr(spec, "feature_names", []) or []) or None,
        getattr(spec, "steps_back", None),
    )


def _spec_mismatch(stored: Any, expected: InputSpec) -> str | None:
    """Return a short description of the first field that doesn't match,
    or ``None`` if all structural fields agree.

    Only the three fields that affect tensor shape are compared —
    ``n_features``, ``feature_names``, ``steps_back``. ``value_range``
    and ``units`` are descriptive and don't block.
    """
    s_n, s_names, s_back = _spec_fields(stored)
    e_n, e_names, e_back = _spec_fields(expected)
    if s_n != e_n:
        return f"n_features {s_n} != {e_n}"
    if s_names != e_names:
        return f"feature_names {s_names} != {e_names}"
    if s_back != e_back:
        return f"steps_back {s_back} != {e_back}"
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
