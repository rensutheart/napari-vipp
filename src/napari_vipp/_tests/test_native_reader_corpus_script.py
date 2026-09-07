"""Network-free checks for the explicit native reader qualification cache."""

import hashlib
import io
from pathlib import Path

import pytest

from scripts import cache_native_reader_corpus as corpus


def _artifact(data=b"frozen"):
    return {
        "relative_path": "acceptance/test/source.bin",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "url": "https://example.invalid/frozen.bin",
    }


def test_native_subset_preserves_the_frozen_corpus_and_scope():
    artifacts = corpus.native_artifacts()
    assert len(artifacts) == 8
    assert sum(item["bytes"] for _name, item in artifacts) == 84_216_131
    assert not any("vsi" in name or "ims" in name for name, _item in artifacts)
    assert all(item["url"].startswith("https://") for _name, item in artifacts)


def test_missing_cache_never_downloads_without_explicit_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(
        corpus, "urlopen", lambda *_a, **_k: pytest.fail("Network used")
    )
    with pytest.raises(FileNotFoundError, match="use --download"):
        corpus.cache_artifact(tmp_path, _artifact())


def test_corrupt_existing_cache_is_not_refreshed_to_hide_drift(tmp_path, monkeypatch):
    artifact = _artifact()
    path = tmp_path / artifact["relative_path"]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"edited")
    monkeypatch.setattr(
        corpus, "urlopen", lambda *_a, **_k: pytest.fail("Network used")
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        corpus.cache_artifact(tmp_path, artifact, download=True)
    assert path.read_bytes() == b"edited"


@pytest.mark.parametrize("response_bytes", [b"frozen", b"edited", b"too long"])
def test_download_verifies_bounded_bytes_before_publication(
    tmp_path, monkeypatch, response_bytes
):
    def response(url, *, timeout):
        assert timeout == 60 and url.startswith("https://")
        stream = io.BytesIO(response_bytes)
        stream.url = url
        return stream

    monkeypatch.setattr(corpus, "urlopen", response)
    artifact = _artifact()
    if response_bytes == b"frozen":
        path = corpus.cache_artifact(tmp_path, artifact, download=True)
        assert path.read_bytes() == response_bytes
    else:
        with pytest.raises(ValueError):
            corpus.cache_artifact(tmp_path, artifact, download=True)
        assert not (tmp_path / artifact["relative_path"]).exists()
    assert not list(tmp_path.rglob(".vipp-corpus-*"))


def test_artifact_paths_cannot_escape_the_cache(tmp_path):
    artifact = {**_artifact(), "relative_path": "../elsewhere.bin"}
    with pytest.raises(ValueError, match="escapes"):
        corpus.cache_artifact(tmp_path, artifact, download=True)


def test_macos_corpus_gate_is_explicit_installed_and_no_skip():
    workflow = (
        Path(__file__).resolve().parents[3] / ".github/workflows/macos-installer.yml"
    ).read_text(encoding="utf-8")
    assert "qualify_native_readers:" in workflow
    assert (
        "github.event_name == 'workflow_dispatch' && inputs.qualify_native_readers"
        in workflow
    )
    assert 'VIPP_PUBLIC_DATA_STRICT: "1"' in workflow
    assert '--cache "$VIPP_PUBLIC_DATA_ROOT" --download' in workflow
    assert "--system-site-packages" in workflow
    assert "installed.is_relative_to" in workflow
    assert "--noconftest --import-mode=importlib" in workflow
    assert '"tests": 16, "failures": 0, "errors": 0, "skipped": 0' in workflow
    assert "dist/macos-development/reader-qualification" in workflow
    assert "scripts/smoke_mesh_install.py --require-installed" in workflow
