"""Optional bearer-token auth.

When ``intelligence.auth.token_env`` is set to the name of an env var
holding the expected token, every protected request must carry
``Authorization: Bearer <token>`` matching that env var's value at
service start.

Probes (``/healthz``, ``/readyz``, ``/metrics``) and the auto-generated
API docs (``/docs``, ``/redoc``, ``/openapi.json``) stay open regardless
— k8s probes and API discovery shouldn't need credentials.

Default behaviour is auth disabled (the config field is ``None``).
Local dev and the smoke stack stay frictionless.
"""

from __future__ import annotations

import os

from fastapi.responses import JSONResponse

_UNAUTHENTICATED_PATHS: frozenset[str] = frozenset(
    {
        "/healthz",
        "/readyz",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
)


class BearerTokenMiddleware:
    """ASGI middleware that checks ``Authorization: Bearer <token>`` on
    protected paths. No-op when ``expected_token`` is ``None``.
    """

    def __init__(self, app, *, expected_token: str | None) -> None:
        self.app = app
        self.expected = expected_token

    async def __call__(self, scope, receive, send):
        if self.expected is None or scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "/")
        if path in _UNAUTHENTICATED_PATHS:
            await self.app(scope, receive, send)
            return

        auth_header = ""
        for name, value in scope.get("headers", []):
            if name == b"authorization":
                auth_header = value.decode("latin-1", errors="replace")
                break

        if auth_header.startswith("Bearer ") and auth_header[7:] == self.expected:
            await self.app(scope, receive, send)
            return

        response = JSONResponse(
            {"detail": "missing or invalid bearer token"},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
        await response(scope, receive, send)


def resolve_expected_token(token_env: str | None) -> str | None:
    """Read the expected token from the configured env var. ``None`` (or
    an unset env var) disables auth — the middleware passes everything
    through.
    """
    if not token_env:
        return None
    return os.environ.get(token_env)
