"""Manifest schema + global extension allowlist.

The manifest is self-describing: each artefact's ``manifest.json``
declares which files it contains, and the validator enforces that
nothing else is present in the directory. Adding a new model kind
should not require an edit to ``manifest.py``; security is purely
structural — no executable types, no pickle, no path traversal.
"""

from __future__ import annotations

import pytest

from intelligence.ml.artifact.manifest import (
    ALLOWED_EXTENSIONS,
    ALWAYS_ALLOWED,
    SCHEMA_VERSION,
    ManifestError,
    read_manifest,
    validate_artifact_directory,
    validate_manifest_dict,
    write_manifest,
)


def test_schema_version_is_one():
    assert SCHEMA_VERSION == 1


def test_allowed_extensions_excludes_pickle_and_executables():
    """Defense in depth: even a future regression that tried to write a
    pickle or native shared object into the artefact dir would fail at
    manifest-write time."""
    for forbidden in {".pkl", ".pickle", ".py", ".so", ".dylib", ".exe", ".sh", ".dll"}:
        assert forbidden not in ALLOWED_EXTENSIONS


def test_always_allowed_covers_manifest_and_bentoml_yaml():
    """``manifest.json`` is ours; ``model.yaml`` is bentoml-generated —
    both ride alongside whatever the kind declares."""
    assert "manifest.json" in ALWAYS_ALLOWED
    assert "model.yaml" in ALWAYS_ALLOWED


def test_validate_rejects_old_schema_version():
    with pytest.raises(ManifestError, match=r"schema_version"):
        validate_manifest_dict(
            {
                "schema_version": 0,
                "kind": "arima",
                "created_at": "2026-05-13T10:00:00+00:00",
                "files": {},
            }
        )


def test_validate_rejects_future_schema_version():
    with pytest.raises(ManifestError, match=r"schema_version"):
        validate_manifest_dict(
            {
                "schema_version": 999,
                "kind": "arima",
                "created_at": "2026-05-13T10:00:00+00:00",
                "files": {},
            }
        )


def test_validate_rejects_missing_required_field():
    with pytest.raises(ManifestError):
        validate_manifest_dict({"schema_version": SCHEMA_VERSION, "kind": "arima"})


def test_validate_rejects_pickle_extension_in_files():
    with pytest.raises(ManifestError, match="disallowed extension"):
        validate_manifest_dict(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "arima",
                "created_at": "2026-05-13T10:00:00+00:00",
                "files": {"model": "model.pkl"},
            }
        )


def test_validate_rejects_path_traversal_in_filename():
    with pytest.raises(ManifestError, match=r"traversal|path separator"):
        validate_manifest_dict(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "arima",
                "created_at": "2026-05-13T10:00:00+00:00",
                "files": {"model": "../etc/passwd"},
            }
        )


def test_validate_rejects_hidden_filename():
    with pytest.raises(ManifestError, match="hidden"):
        validate_manifest_dict(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "arima",
                "created_at": "2026-05-13T10:00:00+00:00",
                "files": {"model": ".secret.json"},
            }
        )


def test_validate_rejects_executable_extension():
    with pytest.raises(ManifestError, match="disallowed extension"):
        validate_manifest_dict(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "arima",
                "created_at": "2026-05-13T10:00:00+00:00",
                "files": {"shim": "loader.py"},
            }
        )


def test_validate_directory_accepts_declared_files_plus_always_allowed(tmp_path):
    manifest = write_manifest(
        tmp_path,
        "arima",
        {"model": "arima.json", "scaler_meta": "scaler.json", "scaler_arrays": "scaler.npz"},
    )
    (tmp_path / "arima.json").write_text("{}")
    (tmp_path / "scaler.json").write_text("{}")
    (tmp_path / "scaler.npz").write_bytes(b"x")
    # ``model.yaml`` is in ALWAYS_ALLOWED — simulate the bento layer
    (tmp_path / "model.yaml").write_text("...")
    validate_artifact_directory(tmp_path, manifest)  # no raise


def test_validate_directory_rejects_undeclared_stowaway(tmp_path):
    manifest = write_manifest(tmp_path, "arima", {"model": "arima.json"})
    (tmp_path / "arima.json").write_text("{}")
    (tmp_path / "stowaway.json").write_text("{}")
    with pytest.raises(ManifestError, match=r"stowaway\.json"):
        validate_artifact_directory(tmp_path, manifest)


def test_validate_directory_rejects_missing_declared_file(tmp_path):
    manifest = write_manifest(tmp_path, "arima", {"model": "arima.json"})
    # we deliberately don't create arima.json
    with pytest.raises(ManifestError, match="missing"):
        validate_artifact_directory(tmp_path, manifest)


def test_validate_directory_rejects_subdirectory(tmp_path):
    manifest = write_manifest(tmp_path, "arima", {})
    (tmp_path / "nested").mkdir()
    with pytest.raises(ManifestError, match="subdirectory"):
        validate_artifact_directory(tmp_path, manifest)


def test_write_manifest_refuses_unsafe_filename(tmp_path):
    """A buggy ``save_artifacts`` that tried to declare a ``.pkl`` file
    would be caught here before the manifest hits disk."""
    with pytest.raises(ManifestError, match="disallowed extension"):
        write_manifest(tmp_path, "arima", {"model": "model.pkl"})


def test_read_manifest_roundtrips_through_disk(tmp_path):
    written = write_manifest(
        tmp_path,
        "xgb",
        {"model": "xgb.ubj", "scaler_y": "scaler_y.json", "scaler_y_arrays": "scaler_y.npz"},
    )
    read_back = read_manifest(tmp_path)
    assert read_back == written


def test_read_manifest_raises_when_missing(tmp_path):
    with pytest.raises(ManifestError, match=r"manifest\.json missing"):
        read_manifest(tmp_path)


def test_read_manifest_raises_on_corrupt_json(tmp_path):
    (tmp_path / "manifest.json").write_text("{not valid json")
    with pytest.raises(ManifestError):
        read_manifest(tmp_path)
