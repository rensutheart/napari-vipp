from __future__ import annotations

import os

import pytest

from napari_vipp.core.source_identity import (
    SourceChangedError,
    capture_local_source_identity,
    verify_local_source_identity,
)


def test_local_file_identity_hashes_exact_bytes(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"scientific source\x00bytes")

    identity = capture_local_source_identity(source)

    assert identity.kind == "file"
    assert identity.regular_file_count == 1
    assert identity.size_bytes == len(source.read_bytes())
    assert verify_local_source_identity(source, identity) == identity

    source.write_bytes(b"scientific source\x00BYTES")

    with pytest.raises(SourceChangedError, match="source changed"):
        verify_local_source_identity(source, identity)


def test_directory_identity_hashes_relative_paths_and_bytes_not_root_stat(tmp_path):
    source = tmp_path / "source.ome.zarr"
    (source / "0").mkdir(parents=True)
    (source / ".zattrs").write_text('{"multiscales": []}', encoding="utf-8")
    chunk = source / "0" / "0.0"
    chunk.write_bytes(b"chunk-A")
    root_stat = source.stat()
    identity = capture_local_source_identity(source)

    chunk.write_bytes(b"chunk-B")
    os.utime(
        source,
        ns=(root_stat.st_atime_ns, root_stat.st_mtime_ns),
    )

    assert source.stat().st_mtime_ns == root_stat.st_mtime_ns
    changed = capture_local_source_identity(source)
    assert changed.sha256 != identity.sha256
    assert changed.size_bytes == identity.size_bytes
    with pytest.raises(SourceChangedError, match="source changed"):
        verify_local_source_identity(source, identity)


def test_directory_identity_includes_regular_file_relative_path(tmp_path):
    source = tmp_path / "source.zarr"
    first = source / "0" / "chunk"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"same bytes")
    original = capture_local_source_identity(source)

    renamed = source / "1" / "chunk"
    renamed.parent.mkdir()
    first.rename(renamed)

    assert capture_local_source_identity(source).sha256 != original.sha256
