"""HTTP service exposing per-task endpoints.

A FastAPI app handles routing + request validation; a ``bentoml.Service``
mounts the FastAPI app for deployment via ``bentoml serve``. Routes:

    GET    /healthz
    GET    /readyz
    GET    /tasks
    POST   /tasks/{task}/train
    POST   /tasks/{task}/predict

Importing this module does NOT load any Bento models — task instances
defer model fetch to first use (see ``tests/integration/test_lazy_loading.py``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from intelligence import __version__
from intelligence.api.auth import BearerTokenMiddleware, resolve_expected_token
from intelligence.api.model_repo import pull_from_hf, push_to_hf
from intelligence.api.observability import (
    PREDICT_DURATION,
    PREDICT_TOTAL,
    REGISTERED_TASKS,
    TRAIN_DURATION,
    TRAIN_TOTAL,
    ObservabilityMiddleware,
    configure_logging,
    metrics_response,
)
from intelligence.api.schemas import (
    ModelSyncRequest,
    ModelSyncResponse,
    PredictRequest,
    TrainRequest,
)
from intelligence.config import load_config
from intelligence.ml.artifact import list_artifacts_by_name
from intelligence.tasks import TaskRegistry, build_registry_from_config

configure_logging()

logger = logging.getLogger(__name__)


def _load_app_config():
    """Load typed config from ``INTELLIGENCE_CONFIG`` (path) or defaults."""
    cfg_path = os.environ.get("INTELLIGENCE_CONFIG")
    return load_config(Path(cfg_path) if cfg_path else None)


config = _load_app_config()
registry = build_registry_from_config(config.intelligence)
REGISTERED_TASKS.set(len(registry))

# asyncio only weak-refs tasks; hold strong refs so bootstrap coroutines
# aren't garbage-collected mid-run. The done_callback drops each entry.
_bootstrap_tasks: set = set()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Spawn bootstrap coroutines for every task whose config has
    ``auto_train_on_startup: true``. State is flipped to ``running``
    synchronously so ``/readyz`` returns 503 even before the event loop
    ticks the coroutine.

    """
    from intelligence.tasks.bootstrap import bootstrap_task

    for name in registry:
        task = registry.get(name)
        task_cfg = config.intelligence.tasks.get(name)
        if task_cfg is None or not task_cfg.bootstrap.auto_train_on_startup:
            continue
        task.bootstrap_state = "running"
        coro = asyncio.create_task(bootstrap_task(task, config.intelligence))
        _bootstrap_tasks.add(coro)
        coro.add_done_callback(_bootstrap_tasks.discard)

    yield


app = FastAPI(
    title="ICOS Intelligence (O-CEI)",
    version=__version__,
    lifespan=lifespan,
)
# Observability runs outside auth so 401s show up on /metrics.
app.add_middleware(ObservabilityMiddleware)
app.add_middleware(
    BearerTokenMiddleware,
    expected_token=resolve_expected_token(config.intelligence.auth.token_env),
)


@app.get("/metrics")
def metrics():
    """Prometheus-format metrics for scraping. Excluded from its own
    instrumentation to avoid self-amplifying counters."""
    return metrics_response()


@app.get("/healthz")
def healthz() -> dict:
    """Liveness probe: process is alive. Cheap, no external calls."""
    return {"status": "ok", "version": __version__}


def compute_readiness(reg: TaskRegistry) -> tuple[bool, list[dict]]:
    """Aggregate readiness checks. Returns ``(ok, failures)``.

    Probes:
      - registry has at least one task (otherwise the service is useless)
      - bento store is queryable (otherwise train/predict will fail)
      - each task's own ``is_ready`` (e.g. data source reachable)
      - each task's bootstrap state — ``running``/``failed`` blocks
        readiness (``pending`` means bootstrap isn't configured for
        that task, which is fine).
    """
    failures: list[dict] = []

    if len(reg) == 0:
        failures.append({"check": "registry", "detail": "no tasks enabled"})

    try:
        import bentoml

        bentoml.models.list()
    except Exception as e:
        failures.append({"check": "bento_store", "detail": str(e)})

    for name in reg:
        task = reg.get(name)
        boot_state = getattr(task, "bootstrap_state", None)
        if boot_state == "running":
            failures.append(
                {
                    "check": f"task:{name}",
                    "detail": "bootstrap in progress",
                }
            )
        elif boot_state == "failed":
            err = getattr(task, "bootstrap_error", "unknown")
            failures.append(
                {
                    "check": f"task:{name}",
                    "detail": f"bootstrap failed: {err}",
                }
            )

        probe = getattr(task, "is_ready", None)
        if probe is None:
            continue
        try:
            ok, msg = probe()
            if not ok:
                failures.append({"check": f"task:{name}", "detail": msg})
        except Exception as e:
            failures.append({"check": f"task:{name}", "detail": f"probe raised: {e}"})

    return len(failures) == 0, failures


@app.get("/readyz")
def readyz():
    """Readiness probe: is this process able to serve requests right now?"""
    ok, failures = compute_readiness(registry)
    if not ok:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "failures": failures},
        )
    return {"status": "ready", "tasks": len(registry), "version": __version__}


@app.get("/tasks")
def list_tasks() -> dict:
    return {"tasks": registry.list_info()}


@app.get("/tasks/{task_name}/versions")
def list_task_versions(task_name: str) -> dict:
    """Locally-stored artifact versions for this task, newest first.

    Useful for rollback: pick a known-good version and send it as
    ``model_version`` on the next predict request, or set it as
    ``pinned_version:`` in config. Artifacts without a readable
    manifest are filtered out.
    """
    if task_name not in registry:
        raise HTTPException(status_code=404, detail=f"unknown task: {task_name}")
    task = registry.get(task_name)
    bento_name = getattr(task, "bento_name", task_name)

    artifacts = list_artifacts_by_name(bento_name)

    return {
        "task": task_name,
        "bento_name": bento_name,
        "pinned_version": getattr(task, "pinned_version", None),
        "versions": [
            {
                "tag": a.tag,
                "version": a.version,
                "created_at": a.created_at,
            }
            for a in artifacts
        ],
    }


@app.delete("/tasks/{task_name}/versions/{version}")
def delete_task_version(task_name: str, version: str) -> dict:
    """Remove a specific stored version of this task from the local
    BentoML store. Reclaims PVC space — the store is otherwise
    append-only and grows with every train.

    Guards:

    - ``version="latest"`` is refused: pass a concrete version string
      so an operator can't accidentally drop the most recent.
    - The currently ``pinned_version`` is refused: deleting it would
      immediately break ``/predict`` for that task.
    """
    import bentoml
    from bentoml.exceptions import NotFound

    if task_name not in registry:
        raise HTTPException(status_code=404, detail=f"unknown task: {task_name}")
    if version == "latest":
        raise HTTPException(
            status_code=400,
            detail="refuse to delete 'latest' — pass a concrete version string",
        )
    task = registry.get(task_name)
    if getattr(task, "pinned_version", None) == version:
        raise HTTPException(
            status_code=409,
            detail=(
                f"version {version!r} is pinned for task {task_name!r}; "
                f"unpin it in config or delete a different version"
            ),
        )
    bento_name = getattr(task, "bento_name", task_name)
    tag = f"{bento_name}:{version}"
    try:
        bentoml.models.delete(tag)
    except NotFound as e:
        raise HTTPException(
            status_code=404,
            detail=f"version {version!r} not found for task {task_name!r}",
        ) from e
    # Drop cached entries that may have pointed at the deleted tag.
    cached = getattr(task, "_cached_artifacts", None)
    if isinstance(cached, dict):
        cached.pop(version, None)
        cached.pop("latest", None)
    return {"task": task_name, "deleted": version}


@app.get("/models")
def list_models() -> dict:
    """List Bento models in the local store.

    ``bentoml.models.list()`` reads metadata only — does not load weights,
    so this endpoint is cheap and doesn't break lazy loading.
    """
    import bentoml

    models = bentoml.models.list()
    return {
        "count": len(models),
        "models": [{"name": m.tag.name, "version": m.tag.version} for m in models],
    }


@app.post("/tasks/{task_name}/train")
def train(task_name: str, req: TrainRequest):
    import requests

    if task_name not in registry:
        raise HTTPException(status_code=404, detail=f"unknown task: {task_name}")
    task = registry.get(task_name)
    start = time.monotonic()
    try:
        result = task.train(req)
    except NotImplementedError as e:
        TRAIN_TOTAL.labels(task=task_name, status="unsupported").inc()
        return JSONResponse(status_code=501, content={"detail": str(e)})
    except FileNotFoundError as e:
        TRAIN_TOTAL.labels(task=task_name, status="not_found").inc()
        raise HTTPException(status_code=404, detail=str(e)) from e
    except (ValueError, TypeError) as e:
        TRAIN_TOTAL.labels(task=task_name, status="invalid").inc()
        raise HTTPException(status_code=422, detail=str(e)) from e
    except requests.RequestException as e:
        # Upstream Prometheus failure (5xx, timeout, connection refused).
        # 502 distinguishes "my data source is broken" from "the model
        # crashed" so dashboards can route the page correctly.
        TRAIN_TOTAL.labels(task=task_name, status="upstream_error").inc()
        raise HTTPException(
            status_code=502, detail=f"upstream telemetry error: {type(e).__name__}: {e}"
        ) from e
    except Exception:
        TRAIN_TOTAL.labels(task=task_name, status="error").inc()
        raise
    TRAIN_TOTAL.labels(task=task_name, status="ok").inc()
    TRAIN_DURATION.labels(task=task_name).observe(time.monotonic() - start)
    return result.model_dump()


@app.post("/models/sync")
def sync_model(req: ModelSyncRequest):
    """Push a local model to Hugging Face or pull one into the local store.

    Requires ``model_repo.hf_enabled`` in config and ``HF_TOKEN`` in
    the environment. Pulled models still need to match the task's
    ``input_spec`` to be served (see ``BaseTask._verify_artifact``).
    """
    import requests
    from huggingface_hub.errors import HfHubHTTPError

    cfg = config.intelligence.model_repo
    if not cfg.hf_enabled:
        raise HTTPException(
            status_code=403,
            detail="model_repo.hf_enabled is false in config",
        )
    repo_id = req.repo_id or cfg.repo_id
    if not repo_id:
        raise HTTPException(
            status_code=422,
            detail="repo_id missing — set it in config or in the request body",
        )
    try:
        if req.action == "push":
            tag = push_to_hf(req.model_tag, repo_id, commit_message=req.commit_message)
        else:
            tag = pull_from_hf(req.model_tag, repo_id)
    except PermissionError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except HfHubHTTPError as e:
        # Bad token (401), repo not found (404), gated repo (403), etc.
        # Forward the upstream status so the client doesn't see an opaque 500.
        upstream_status = getattr(e.response, "status_code", None)
        if upstream_status in (401, 403, 404):
            raise HTTPException(
                status_code=upstream_status,
                detail=f"upstream HF error: {type(e).__name__}: {e}",
            ) from e
        raise HTTPException(
            status_code=502,
            detail=f"upstream HF error: {type(e).__name__}: {e}",
        ) from e
    except requests.RequestException as e:
        # Connection refused / timeout / DNS — Hub unreachable.
        raise HTTPException(
            status_code=502,
            detail=f"upstream HF transport error: {type(e).__name__}: {e}",
        ) from e
    return ModelSyncResponse(action=req.action, model_tag=tag, repo_id=repo_id).model_dump()


@app.post("/tasks/{task_name}/predict")
def predict(task_name: str, req: PredictRequest):
    if task_name not in registry:
        raise HTTPException(status_code=404, detail=f"unknown task: {task_name}")
    task = registry.get(task_name)
    start = time.monotonic()
    try:
        result = task.predict(req)
    except FileNotFoundError as e:
        PREDICT_TOTAL.labels(task=task_name, status="no_model").inc()
        return JSONResponse(status_code=503, content={"detail": str(e)})
    except (ValueError, TypeError) as e:
        PREDICT_TOTAL.labels(task=task_name, status="invalid").inc()
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception:
        PREDICT_TOTAL.labels(task=task_name, status="error").inc()
        raise
    PREDICT_TOTAL.labels(task=task_name, status="ok").inc()
    PREDICT_DURATION.labels(task=task_name).observe(time.monotonic() - start)
    return result.model_dump()


# Expose the FastAPI app as a BentoML Service so ``bentoml serve``
# works alongside ``uvicorn intelligence.api.service:app``. The class-
# based Service constructor moved to ``bentoml.legacy`` in 1.2 when
# the decorator-based @bentoml.service became the recommended pattern;
# the legacy form is preserved through 1.4.x for backward compat.
from bentoml.legacy import Service  # noqa: E402

svc = Service(name="intelligence")
svc.mount_asgi_app(app)
