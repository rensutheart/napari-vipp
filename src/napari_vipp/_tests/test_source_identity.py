from __future__ import annotations

import os
import threading

import pytest

from napari_vipp.core.progress import OperationCancelled
from napari_vipp.core.source_identity import (
    SourceChangedError,
    capture_local_source_bundle,
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


def test_source_identity_reports_byte_progress_and_cancels_between_chunks(tmp_path):
    source = tmp_path / "large.bin"
    source.write_bytes(b"x" * (3 * 1024 * 1024 + 17))
    cancel_event = threading.Event()
    updates = []

    def progress(current, total, message):
        updates.append((current, total, message))
        if current >= 1024 * 1024:
            cancel_event.set()

    with pytest.raises(OperationCancelled, match="source identity"):
        capture_local_source_identity(
            source,
            cancel_callback=cancel_event.is_set,
            progress_callback=progress,
        )

    assert updates[0][0] == 0
    assert updates[0][1] == source.stat().st_size
    assert any(current >= 1024 * 1024 for current, _total, _message in updates)


def test_vsi_identity_binds_primary_file_and_ets_companion_tree(tmp_path):
    source = tmp_path / "sample.vsi"
    source.write_bytes(b"vsi metadata")
    companion = tmp_path / "_sample_" / "stack1"
    companion.mkdir(parents=True)
    ets = companion / "frame_t_0.ets"
    ets.write_bytes(b"pixel payload A")

    bundle = capture_local_source_bundle(source, source_format="olympus-vsi")
    identity = capture_local_source_identity(source)

    assert bundle.revision.kind == "multifile"
    assert identity.kind == "multifile"
    assert bundle.revision.sha256 == identity.sha256
    assert [member.key for member in bundle.members] == [
        ".",
        "_sample_/stack1/frame_t_0.ets",
    ]
    assert [member.role for member in bundle.members] == ["primary", "companion"]

    ets.write_bytes(b"pixel payload B")
    with pytest.raises(SourceChangedError, match="source changed"):
        verify_local_source_identity(source, identity)


@pytest.mark.parametrize(
    ("source_name", "companion_name"),
    (("sample.vsi", "_sample_"), ("sample.oif", "sample.files")),
)
def test_multifile_source_requires_its_companion_directory(
    tmp_path,
    source_name,
    companion_name,
):
    source = tmp_path / source_name
    source.write_bytes(b"container metadata")

    with pytest.raises(FileNotFoundError, match="companion directory is missing"):
        capture_local_source_bundle(source)

    companion = tmp_path / companion_name
    companion.mkdir()
    with pytest.raises(FileNotFoundError, match="contains no readable files"):
        capture_local_source_bundle(source)
