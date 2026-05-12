"""Internal observability.

Two outputs, both standard:

- **``GET /metrics``** in Prometheus text format. HTTP-level counters
  + histograms come from a path-normalising ASGI middleware (label
  cardinality stays bounded by route patterns, not by task names).
  Task-level counters for ``train`` / ``predict`` are recorded inside
  the handlers in ``service.py``.
- **Structured JSON logs** on stdout. Each line is a JSON object with
  a stable ``timestamp / level / logger / message`` envelope plus
  whatever ``extra={...}`` the caller passed. A request-scoped
  ``request_id`` is auto-injected via a logging Filter.

Neither output requires bentoml's runner machinery (we serve via
uvicorn). ``prometheus_client`` is the only new runtime dep.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from contextvars import ContextVar

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# A dedicated registry — avoids polluting prometheus_client's default
# global registry (which bentoml may also touch) and makes test isolation
# straightforward.
REGISTRY = CollectorRegistry()

# HTTP layer ---------------------------------------------------------------

HTTP_REQUESTS = Counter(
    "intelligence_http_requests_total",
    "HTTP requests by route pattern, method, and status.",
    ["route", "method", "status"],
    registry=REGISTRY,
)
HTTP_DURATION = Histogram(
    "intelligence_http_request_duration_seconds",
    "HTTP request latency by route pattern.",
    ["route", "method"],
    registry=REGISTRY,
)

# Task layer (low cardinality — bounded by configured tasks) ---------------

TRAIN_TOTAL = Counter(
    "intelligence_train_total",
    "Train attempts by task and outcome.",
    ["task", "status"],
    registry=REGISTRY,
)
TRAIN_DURATION = Histogram(
    "intelligence_train_duration_seconds",
    "Train duration by task.",
    ["task"],
    registry=REGISTRY,
)
PREDICT_TOTAL = Counter(
    "intelligence_predict_total",
    "Predict requests by task and outcome.",
    ["task", "status"],
    registry=REGISTRY,
)
PREDICT_DURATION = Histogram(
    "intelligence_predict_duration_seconds",
    "Predict latency by task.",
    ["task"],
    registry=REGISTRY,
)
REGISTERED_TASKS = Gauge(
    "intelligence_registered_tasks",
    "Number of tasks registered in the live registry.",
    registry=REGISTRY,
)

# Skip-list for paths that get polled constantly (probes, the metrics
# endpoint itself). Recording them would dominate the time series and
# the /metrics scrape would self-amplify.
_SKIP_INSTRUMENTATION: set[str] = {"/healthz", "/readyz", "/metrics"}

# Route-normalising regexes — keep label cardinality independent of the
# number of tasks (which is config-bounded but still better not exploded).
_DYNAMIC_SEGMENTS = [
    (re.compile(r"^/tasks/[^/]+/train$"), "/tasks/{task}/train"),
    (re.compile(r"^/tasks/[^/]+/predict$"), "/tasks/{task}/predict"),
    (re.compile(r"^/tasks/[^/]+/versions$"), "/tasks/{task}/versions"),
    (re.compile(r"^/models/[^/]+$"), "/models/{tag}"),
]


def _normalise_route(path: str) -> str:
    for pattern, replacement in _DYNAMIC_SEGMENTS:
        if pattern.match(path):
            return replacement
    return path


# Request ID context ------------------------------------------------------

_request_id: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    return _request_id.get()


class RequestIdFilter(logging.Filter):
    """Inject the current request_id from the ContextVar onto every log
    record. Records emitted outside a request scope carry an empty
    string."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = _request_id.get()
        return True


# JSON formatter ----------------------------------------------------------

# Standard logging-record attributes we don't want to spam into every log
# line. Everything else on the record becomes a top-level JSON field.
_STD_LOG_RECORD_FIELDS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName",
}


class JsonFormatter(logging.Formatter):
    """Minimal JSON log formatter — no extra deps.

    Output shape: ``{"timestamp": ..., "level": ..., "logger": ...,
    "message": ..., ...extras}``.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _STD_LOG_RECORD_FIELDS or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Replace the root logger's handlers with a single JSON-emitting one.

    Idempotent: safe to call from module-import time. Tests can call
    again to reset state.
    """
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RequestIdFilter())
    root.addHandler(handler)
    root.setLevel(level)


# ASGI middleware ---------------------------------------------------------

logger = logging.getLogger(__name__)


class ObservabilityMiddleware:
    """ASGI middleware: assigns a request_id, times each request, records
    HTTP-level Prometheus metrics, and emits one structured log line per
    request.

    Skips the probe endpoints + the metrics endpoint itself — recording
    those would drown signal in noise.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "/")
        if path in _SKIP_INSTRUMENTATION:
            await self.app(scope, receive, send)
            return

        req_id = uuid.uuid4().hex[:8]
        token = _request_id.set(req_id)
        method: str = scope.get("method", "GET")
        route = _normalise_route(path)
        start = time.monotonic()
        # Mutable holder so the wrapper closure can write back the status.
        status_holder = {"code": 500}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_holder["code"] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            duration = time.monotonic() - start
            HTTP_REQUESTS.labels(route=route, method=method, status="500").inc()
            HTTP_DURATION.labels(route=route, method=method).observe(duration)
            logger.exception(
                "request raised",
                extra={
                    "path": path, "method": method, "route": route,
                    "status": 500, "duration_ms": round(duration * 1000, 2),
                },
            )
            raise
        else:
            duration = time.monotonic() - start
            status = status_holder["code"]
            HTTP_REQUESTS.labels(route=route, method=method, status=str(status)).inc()
            HTTP_DURATION.labels(route=route, method=method).observe(duration)
            logger.info(
                "request",
                extra={
                    "path": path, "method": method, "route": route,
                    "status": status, "duration_ms": round(duration * 1000, 2),
                },
            )
        finally:
            _request_id.reset(token)


# Helpers used by the route handlers --------------------------------------


def metrics_response():
    """Return a Starlette/FastAPI Response object carrying the current
    Prometheus exposition. Caller is the ``/metrics`` route handler.
    """
    from fastapi import Response

    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
