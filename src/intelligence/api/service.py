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

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from intelligence import __version__
from intelligence.api.schemas import PredictRequest, TrainRequest
from intelligence.config import load_config
from intelligence.tasks import TaskRegistry, build_registry_from_config

logger = logging.getLogger(__name__)


def _load_app_config():
    """Load typed config from ``INTELLIGENCE_CONFIG`` (path) or defaults."""
    cfg_path = os.environ.get("INTELLIGENCE_CONFIG")
    return load_config(Path(cfg_path) if cfg_path else None)


config = _load_app_config()
registry = build_registry_from_config(config.intelligence.enabled_tasks)

# Apply MLflow tracking URI if configured.
if config.intelligence.mlflow.tracking_uri:
    try:
        import mlflow
        mlflow.set_tracking_uri(config.intelligence.mlflow.tracking_uri)
    except ImportError:
        logger.warning("mlflow not installed; ignoring mlflow.tracking_uri config")

app = FastAPI(title="Intelligence Utility", version="0.1.0.dev0")


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
    if task_name not in registry:
        raise HTTPException(status_code=404, detail=f"unknown task: {task_name}")
    task = registry.get(task_name)
    try:
        result = task.train(req)
    except NotImplementedError as e:
        # Phase-2 features (e.g. kind='prometheus' before TelemetrySource lands).
        return JSONResponse(status_code=501, content={"detail": str(e)})
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return result.model_dump()


@app.post("/tasks/{task_name}/predict")
def predict(task_name: str, req: PredictRequest):
    if task_name not in registry:
        raise HTTPException(status_code=404, detail=f"unknown task: {task_name}")
    task = registry.get(task_name)
    try:
        result = task.predict(req)
    except FileNotFoundError as e:
        # No trained model yet — caller needs to train first.
        return JSONResponse(status_code=503, content={"detail": str(e)})
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return result.model_dump()


# ---- BentoML hosting --------------------------------------------------
# A ``bentoml.Service`` is what ``bentoml serve`` runs. We mount the
# FastAPI app into it so routing stays clean while deployment uses
# bentoml's runner / config / observability machinery.

import bentoml  # noqa: E402  — kept low to avoid pulling bentoml at import-time on type-only paths

svc = bentoml.Service(name="intelligence")
svc.mount_asgi_app(app)
