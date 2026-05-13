"""HF push/pull functions — manifest-validated, pickle-free.

``HF_TOKEN`` is read at call time. ``pull_from_hf`` validates the
manifest schema + the artefact directory's filename allowlist *before
any file is opened*. A pulled directory carrying a stowaway pickle,
a path-traversing filename, or a mismatched schema_version is refused
at the boundary — by construction no malicious payload ever reaches
``pickle.load`` (which we no longer import).
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest import mock

import pytest

from intelligence.ml.artifact.manifest import ManifestError, write_manifest


def test_module_does_not_import_pickle():
    """Defence in depth: ``intelligence.api.model_repo`` must never
    import ``pickle``. A grep-friendly invariant — a regression that
    re-introduced pickle would trip this test."""
    from intelligence.api import model_repo

    source = inspect.getsource(model_repo)
    assert "import pickle" not in source, "model_repo must not import pickle"
    assert "from pickle" not in source, "model_repo must not import from pickle"


def test_push_raises_permission_error_without_hf_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    from intelligence.api.model_repo import push_to_hf

    with pytest.raises(PermissionError, match="HF_TOKEN"):
        push_to_hf("metrics_utilization_model_arima:latest", "CeADAR/intelligence-bentos")


def test_pull_raises_permission_error_without_hf_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    from intelligence.api.model_repo import pull_from_hf

    with pytest.raises(PermissionError, match="HF_TOKEN"):
        pull_from_hf("metrics_utilization_model_arima:abc123", "CeADAR/intelligence-bentos")


@mock.patch("intelligence.api.model_repo.HfApi")
def test_push_uploads_local_artefact_folder(mock_hf_api_cls, tmp_path, monkeypatch):
    """Push re-validates the artefact before upload and forwards the
    auth token to HfApi."""
    from intelligence.api.model_repo import push_to_hf
    from intelligence.ml.artifact import save_artifact

    monkeypatch.setenv("HF_TOKEN", "fake-token")
    monkeypatch.setenv("BENTOML_HOME", str(tmp_path / "bentoml"))

    # Save a real manifest-described artefact in the local store.
    def write_fn(path):
        (path / "thing.json").write_text('{"x": 1}')
        return {"thing": "thing.json"}

    saved = save_artifact("push_me_test", "arima", write_fn)

    instance = mock_hf_api_cls.return_value
    tag = push_to_hf(saved.tag, "CeADAR/intelligence-bentos", commit_message="hello")

    mock_hf_api_cls.assert_called_once_with(token="fake-token")
    upload_args = instance.upload_folder.call_args
    assert upload_args.kwargs["repo_id"] == "CeADAR/intelligence-bentos"
    assert upload_args.kwargs["repo_type"] == "model"
    assert upload_args.kwargs["commit_message"] == "hello"
    assert Path(upload_args.kwargs["folder_path"]).exists()
    assert tag == saved.tag


@mock.patch("intelligence.api.model_repo.snapshot_download")
def test_pull_validates_manifest_and_imports(mock_snapshot, tmp_path, monkeypatch):
    """Happy path: well-formed manifest-described artefact is
    validated, imported into the local store, and the new tag returned.
    """
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    monkeypatch.setenv("BENTOML_HOME", str(tmp_path / "bentoml"))

    def fake_snapshot(repo_id, repo_type, allow_patterns, local_dir, token):
        name_version = allow_patterns.removesuffix("/**")
        name, version = name_version.split("/")
        out = Path(local_dir) / name / version
        out.mkdir(parents=True, exist_ok=True)
        (out / "thing.json").write_text('{"x": 1}')
        write_manifest(out, "arima", {"thing": "thing.json"})
        return str(out)

    mock_snapshot.side_effect = fake_snapshot

    from intelligence.api.model_repo import pull_from_hf
    from intelligence.ml.artifact import get_artifact_by_tag

    new_tag = pull_from_hf("pulled_model:abc123", "CeADAR/intelligence-bentos")

    # snapshot_download saw the auth token + correct allow_patterns.
    snap_kwargs = mock_snapshot.call_args.kwargs
    assert snap_kwargs["token"] == "fake-token"
    assert snap_kwargs["repo_id"] == "CeADAR/intelligence-bentos"
    assert snap_kwargs["allow_patterns"] == "pulled_model/abc123/**"

    # The new tag resolves to a usable artefact in the local store.
    fetched = get_artifact_by_tag(new_tag)
    assert fetched is not None
    assert fetched.manifest.kind == "arima"
    assert (fetched.path / "thing.json").exists()


@mock.patch("intelligence.api.model_repo.snapshot_download")
def test_pull_rejects_missing_manifest(mock_snapshot, tmp_path, monkeypatch):
    """An artefact directory without ``manifest.json`` is refused
    before any other file is opened — read_manifest raises ManifestError."""
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    monkeypatch.setenv("BENTOML_HOME", str(tmp_path / "bentoml"))

    def fake_snapshot(repo_id, repo_type, allow_patterns, local_dir, token):
        name_version = allow_patterns.removesuffix("/**")
        name, version = name_version.split("/")
        out = Path(local_dir) / name / version
        out.mkdir(parents=True, exist_ok=True)
        # No manifest.json — the boundary check fires here.
        (out / "thing.json").write_text("{}")
        return str(out)

    mock_snapshot.side_effect = fake_snapshot

    from intelligence.api.model_repo import pull_from_hf

    with pytest.raises(ManifestError, match="manifest.json missing"):
        pull_from_hf("missing:nope", "CeADAR/intelligence-bentos")


@mock.patch("intelligence.api.model_repo.snapshot_download")
def test_pull_rejects_stowaway_pickle_file(mock_snapshot, tmp_path, monkeypatch):
    """A pulled directory carrying a stowaway ``.pkl`` file alongside
    a valid manifest is refused. The manifest doesn't declare the
    pickle file, so ``validate_artifact_directory`` flags it before
    anything is opened.
    """
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    monkeypatch.setenv("BENTOML_HOME", str(tmp_path / "bentoml"))

    def fake_snapshot(repo_id, repo_type, allow_patterns, local_dir, token):
        name_version = allow_patterns.removesuffix("/**")
        name, version = name_version.split("/")
        out = Path(local_dir) / name / version
        out.mkdir(parents=True, exist_ok=True)
        (out / "thing.json").write_text("{}")
        write_manifest(out, "arima", {"thing": "thing.json"})
        # Stowaway pickle file alongside the legitimate ones.
        (out / "malicious.pkl").write_bytes(b"unsafe payload")
        return str(out)

    mock_snapshot.side_effect = fake_snapshot

    from intelligence.api.model_repo import pull_from_hf

    with pytest.raises(ManifestError, match="malicious.pkl"):
        pull_from_hf("stowaway:nope", "CeADAR/intelligence-bentos")


@mock.patch("intelligence.api.model_repo.snapshot_download")
def test_pull_rejects_wrong_schema_version(mock_snapshot, tmp_path, monkeypatch):
    """A manifest claiming an unsupported schema_version is refused —
    no fallback to legacy parsing."""
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    monkeypatch.setenv("BENTOML_HOME", str(tmp_path / "bentoml"))

    def fake_snapshot(repo_id, repo_type, allow_patterns, local_dir, token):
        name_version = allow_patterns.removesuffix("/**")
        name, version = name_version.split("/")
        out = Path(local_dir) / name / version
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 999,  # not supported by this version
                    "kind": "arima",
                    "created_at": "2026-05-13T10:00:00+00:00",
                    "files": {},
                }
            )
        )
        return str(out)

    mock_snapshot.side_effect = fake_snapshot

    from intelligence.api.model_repo import pull_from_hf

    with pytest.raises(ManifestError, match="schema_version"):
        pull_from_hf("wrong_schema:nope", "CeADAR/intelligence-bentos")


@mock.patch("intelligence.api.model_repo.snapshot_download")
def test_pull_rejects_manifest_with_traversal_filename(mock_snapshot, tmp_path, monkeypatch):
    """A manifest whose ``files`` map declares a path-traversing
    filename is refused at parse time."""
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    monkeypatch.setenv("BENTOML_HOME", str(tmp_path / "bentoml"))

    def fake_snapshot(repo_id, repo_type, allow_patterns, local_dir, token):
        name_version = allow_patterns.removesuffix("/**")
        name, version = name_version.split("/")
        out = Path(local_dir) / name / version
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "arima",
                    "created_at": "2026-05-13T10:00:00+00:00",
                    "files": {"model": "../etc/passwd"},
                }
            )
        )
        return str(out)

    mock_snapshot.side_effect = fake_snapshot

    from intelligence.api.model_repo import pull_from_hf

    with pytest.raises(ManifestError, match="traversal|path separator"):
        pull_from_hf("traversal:nope", "CeADAR/intelligence-bentos")


@mock.patch("intelligence.api.model_repo.snapshot_download")
def test_pull_raises_when_artifact_missing(mock_snapshot, tmp_path, monkeypatch):
    """The snapshot download succeeded but the expected subdirectory
    isn't there — surface as FileNotFoundError."""
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    monkeypatch.setenv("BENTOML_HOME", str(tmp_path / "bentoml"))

    def fake_snapshot(repo_id, repo_type, allow_patterns, local_dir, token):
        return str(local_dir)  # never creates the expected subdir

    mock_snapshot.side_effect = fake_snapshot

    from intelligence.api.model_repo import pull_from_hf

    with pytest.raises(FileNotFoundError):
        pull_from_hf("missing:nope", "CeADAR/intelligence-bentos")
