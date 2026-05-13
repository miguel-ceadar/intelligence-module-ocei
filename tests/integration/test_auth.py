"""Optional bearer-token auth — opt-in via config.

The middleware is unit-testable in isolation; we mount it on a
throwaway FastAPI app rather than wrestling with the production app's
module-load-time config.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from intelligence.api.auth import BearerTokenMiddleware, resolve_expected_token


def _build_app(expected_token: str | None) -> FastAPI:
    app = FastAPI()
    app.add_middleware(BearerTokenMiddleware, expected_token=expected_token)

    @app.get("/protected")
    def protected():
        return {"ok": True}

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics():
        return "# fake metrics"

    return app


def test_auth_disabled_lets_everything_through():
    client = TestClient(_build_app(expected_token=None))
    assert client.get("/protected").status_code == 200
    assert client.get("/healthz").status_code == 200


def test_protected_path_without_header_returns_401():
    client = TestClient(_build_app(expected_token="s3cret"))
    resp = client.get("/protected")
    assert resp.status_code == 401
    assert "bearer" in resp.json()["detail"].lower()
    # WWW-Authenticate header tells the client what scheme to use.
    assert resp.headers.get("www-authenticate") == "Bearer"


def test_protected_path_with_wrong_token_returns_401():
    client = TestClient(_build_app(expected_token="s3cret"))
    resp = client.get("/protected", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


def test_protected_path_with_correct_token_passes():
    client = TestClient(_build_app(expected_token="s3cret"))
    resp = client.get("/protected", headers={"Authorization": "Bearer s3cret"})
    assert resp.status_code == 200


def test_protected_path_with_token_longer_than_expected_returns_401():
    # Pins the constant-time comparison: different-length strings must
    # still reject cleanly (compare_digest used to raise on differing
    # types/lengths in older Pythons; modern compare_digest handles it
    # but the contract matters either way).
    client = TestClient(_build_app(expected_token="s3cret"))
    resp = client.get("/protected", headers={"Authorization": "Bearer s3cretwithextra"})
    assert resp.status_code == 401


def test_protected_path_with_empty_bearer_value_returns_401():
    # "Bearer " with a trailing space and no token used to compare an
    # empty string against the expected — would have passed if expected
    # was also empty. Pin the rejection.
    client = TestClient(_build_app(expected_token="s3cret"))
    resp = client.get("/protected", headers={"Authorization": "Bearer "})
    assert resp.status_code == 401


def test_protected_path_with_non_bearer_scheme_returns_401():
    client = TestClient(_build_app(expected_token="s3cret"))
    resp = client.get("/protected", headers={"Authorization": "Basic dXNlcjpwYXNz"})
    assert resp.status_code == 401


def test_protected_path_with_bearer_no_space_returns_401():
    client = TestClient(_build_app(expected_token="s3cret"))
    resp = client.get("/protected", headers={"Authorization": "Bearer"})
    assert resp.status_code == 401


def test_probe_endpoints_stay_open_even_with_auth_enabled():
    """k8s liveness probes shouldn't need credentials — operationally
    fragile if they did."""
    client = TestClient(_build_app(expected_token="s3cret"))
    assert client.get("/healthz").status_code == 200
    assert client.get("/metrics").status_code == 200


def test_resolve_expected_token_reads_from_env(monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "abc123")
    assert resolve_expected_token("MY_TOKEN") == "abc123"


def test_resolve_expected_token_returns_none_when_env_var_missing(monkeypatch):
    monkeypatch.delenv("MY_TOKEN", raising=False)
    assert resolve_expected_token("MY_TOKEN") is None


def test_resolve_expected_token_returns_none_when_env_var_is_empty(monkeypatch):
    # An env var set to "" must not be treated as a valid token —
    # otherwise the middleware would happily accept "Bearer " and let
    # everyone in.
    monkeypatch.setenv("MY_TOKEN", "")
    assert resolve_expected_token("MY_TOKEN") is None


def test_resolve_expected_token_returns_none_when_config_field_unset():
    assert resolve_expected_token(None) is None
    assert resolve_expected_token("") is None
