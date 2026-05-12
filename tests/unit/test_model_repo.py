"""HF push/pull functions.

``HF_TOKEN`` is read from the environment at call time (not at module
load) so a missing token is a 401-shaped runtime error, not an import
crash. HF API calls are mocked at module level so the tests don't talk
to the network.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest import mock

import pytest


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
def test_push_uploads_local_bento_folder(mock_hf_api_cls, tmp_path, monkeypatch):
    import bentoml

    from intelligence.api.model_repo import push_to_hf

    monkeypatch.setenv("HF_TOKEN", "fake-token")
    monkeypatch.setenv("BENTOML_HOME", str(tmp_path / "bentoml"))

    # Save a real Bento in the local store so we can exercise the path lookup.
    bento = bentoml.picklable_model.save_model(
        "push_me_test", {"weights": [1, 2, 3]}, custom_objects={"meta": "ok"},
    )

    instance = mock_hf_api_cls.return_value
    tag = push_to_hf(str(bento.tag), "CeADAR/intelligence-bentos", commit_message="hello")

    mock_hf_api_cls.assert_called_once_with(token="fake-token")
    upload_args = instance.upload_folder.call_args
    assert upload_args.kwargs["repo_id"] == "CeADAR/intelligence-bentos"
    assert upload_args.kwargs["repo_type"] == "model"
    assert upload_args.kwargs["commit_message"] == "hello"
    assert Path(upload_args.kwargs["folder_path"]).exists()
    assert tag == str(bento.tag)


@mock.patch("intelligence.api.model_repo.snapshot_download")
def test_pull_downloads_and_resaves_as_picklable_model(
    mock_snapshot, tmp_path, monkeypatch,
):
    """``pull`` stages a snapshot from HF, loads the pickled model and
    custom_objects, then re-saves into the local Bento store via
    ``picklable_model`` (consistent with how the new tree saves models).
    """
    import bentoml

    monkeypatch.setenv("HF_TOKEN", "fake-token")
    monkeypatch.setenv("BENTOML_HOME", str(tmp_path / "bentoml"))

    # Simulate what snapshot_download writes to ``local_dir``: a
    # ``{name}/{version}/{saved_model.pkl, custom_objects.pkl}`` layout.
    def fake_snapshot(repo_id, repo_type, allow_patterns, local_dir, token):
        # parse the model name + version out of the pattern so we mirror the layout
        name_version = allow_patterns.removesuffix("/**")
        name, version = name_version.split("/")
        out = Path(local_dir) / name / version
        out.mkdir(parents=True, exist_ok=True)
        with (out / "saved_model.pkl").open("wb") as f:
            pickle.dump({"weights": [9, 8, 7]}, f)
        with (out / "custom_objects.pkl").open("wb") as f:
            pickle.dump({"meta": "from_hf"}, f)
        return str(out)

    mock_snapshot.side_effect = fake_snapshot

    from intelligence.api.model_repo import pull_from_hf

    new_tag = pull_from_hf("pulled_model:abc123", "CeADAR/intelligence-bentos")

    # snapshot_download was called with token from env
    snap_kwargs = mock_snapshot.call_args.kwargs
    assert snap_kwargs["token"] == "fake-token"
    assert snap_kwargs["repo_id"] == "CeADAR/intelligence-bentos"
    assert snap_kwargs["allow_patterns"] == "pulled_model/abc123/**"

    # And the local store now has a Bento with the pulled custom_objects.
    bento = bentoml.picklable_model.get(new_tag)
    assert bento.custom_objects == {"meta": "from_hf"}


@mock.patch("intelligence.api.model_repo.snapshot_download")
def test_pull_raises_when_artifact_missing(mock_snapshot, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    monkeypatch.setenv("BENTOML_HOME", str(tmp_path / "bentoml"))

    def fake_snapshot(repo_id, repo_type, allow_patterns, local_dir, token):
        return str(local_dir)  # but doesn't create the expected subdir

    mock_snapshot.side_effect = fake_snapshot

    from intelligence.api.model_repo import pull_from_hf

    with pytest.raises(FileNotFoundError):
        pull_from_hf("missing:nope", "CeADAR/intelligence-bentos")
