"""Artefact store — thin wrapper over bentoml's tag/version store.

Wraps ``bentoml.models.create`` / ``get`` / ``list`` so the rest of the
codebase calls a narrow, manifest-aware API. The bentoml-specific bits
(``Tag``, ``NotFound``, ``ModelContext``) stay inside ``store.py``;
call-sites work in terms of :class:`SavedArtifact` + string tags.

Each test isolates a fresh ``BENTOML_HOME`` via the fixture so saves
don't bleed across cases.
"""

from __future__ import annotations

import time

import pytest

from intelligence.ml.artifact import (
    SavedArtifact,
    get_artifact_by_tag,
    import_artifact,
    list_artifacts_by_name,
    save_artifact,
)
from intelligence.ml.artifact.manifest import ManifestError, write_manifest


@pytest.fixture(autouse=True)
def _isolated_bentoml_home(tmp_path, monkeypatch):
    monkeypatch.setenv("BENTOML_HOME", str(tmp_path / "bentoml"))
    yield


def test_save_round_trips_through_tag():
    def write_fn(path):
        (path / "thing.json").write_text('{"x": 1}')
        return {"thing": "thing.json"}

    saved = save_artifact("save_round_trip_model", "arima", write_fn)

    assert isinstance(saved, SavedArtifact)
    assert saved.name == "save_round_trip_model"
    assert saved.manifest.kind == "arima"
    assert (saved.path / "thing.json").exists()
    assert (saved.path / "manifest.json").exists()

    fetched = get_artifact_by_tag(saved.tag)
    assert fetched is not None
    assert fetched.tag == saved.tag
    assert fetched.path == saved.path


def test_get_missing_tag_returns_none():
    assert get_artifact_by_tag("nonexistent_model:badversion") is None


def test_get_latest_resolves_to_newest():
    """``:latest`` is bentoml's canonical alias for the most recent
    version under that name — BaseTask relies on this for default
    predict-time resolution."""
    save_artifact("latest_check", "arima", lambda _p: {})
    time.sleep(0.01)
    second = save_artifact("latest_check", "arima", lambda _p: {})

    latest = get_artifact_by_tag("latest_check:latest")
    assert latest is not None
    assert latest.tag == second.tag


def test_list_artifacts_returns_newest_first():
    first = save_artifact("ranked", "arima", lambda _p: {})
    time.sleep(0.01)
    second = save_artifact("ranked", "arima", lambda _p: {})

    result = list_artifacts_by_name("ranked")
    assert [a.tag for a in result] == [second.tag, first.tag]


def test_list_artifacts_filters_by_name():
    save_artifact("name_a", "arima", lambda _p: {})
    save_artifact("name_b", "arima", lambda _p: {})

    a = list_artifacts_by_name("name_a")
    b = list_artifacts_by_name("name_b")
    assert [x.name for x in a] == ["name_a"]
    assert [x.name for x in b] == ["name_b"]


def test_save_refuses_undeclared_stowaway():
    def write_fn(path):
        (path / "thing.json").write_text("{}")
        (path / "stowaway.json").write_text("{}")  # not declared in returned map
        return {"thing": "thing.json"}

    with pytest.raises(ManifestError, match="stowaway.json"):
        save_artifact("stowaway_check", "arima", write_fn)


def test_save_refuses_unsafe_extension_in_declaration():
    """A buggy save_fn that declared a pickle file is caught before the
    bento commits — the model never lands in the local store."""

    def write_fn(path):
        (path / "x.pkl").write_bytes(b"unsafe")
        return {"thing": "x.pkl"}

    with pytest.raises(ManifestError, match="disallowed extension"):
        save_artifact("ext_check", "arima", write_fn)


def test_tag_is_unique_per_save():
    """Each save produces a fresh version; both remain retrievable."""
    s1 = save_artifact("immut", "arima", lambda _p: {})
    s2 = save_artifact("immut", "arima", lambda _p: {})
    assert s1.tag != s2.tag
    assert get_artifact_by_tag(s1.tag) is not None
    assert get_artifact_by_tag(s2.tag) is not None


def test_import_artifact_copies_vetted_directory(tmp_path):
    """HF pull workflow: caller has vetted a directory; import_artifact
    copies it into the local store under a new tag."""
    source = tmp_path / "vetted"
    source.mkdir()
    (source / "thing.json").write_text("{}")
    write_manifest(source, "arima", {"thing": "thing.json"})

    imported = import_artifact("pulled", source)
    assert imported.name == "pulled"
    assert imported.manifest.kind == "arima"
    assert (imported.path / "thing.json").exists()


def test_import_artifact_rejects_stowaway_in_source(tmp_path):
    """Defense in depth: import_artifact re-validates even though the
    HF pull path is expected to validate first."""
    source = tmp_path / "with_stowaway"
    source.mkdir()
    (source / "thing.json").write_text("{}")
    (source / "stray.json").write_text("{}")
    write_manifest(source, "arima", {"thing": "thing.json"})

    with pytest.raises(ManifestError, match="stray.json"):
        import_artifact("rejected", source)


def test_saved_artifact_carries_created_at():
    saved = save_artifact("with_time", "arima", lambda _p: {})
    fetched = get_artifact_by_tag(saved.tag)
    assert fetched is not None
    assert fetched.created_at  # non-empty ISO timestamp


def test_metadata_records_kind():
    """The ``kind`` is recorded both in our manifest *and* in bentoml's
    metadata — the second copy lets ``list_artifacts_by_name`` skip
    non-ours bentos cheaply without reading the manifest."""
    import bentoml

    saved = save_artifact("kind_meta_check", "arima", lambda _p: {})
    m = bentoml.models.get(saved.tag)
    assert m.info.metadata.get("kind") == "arima"
