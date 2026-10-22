from __future__ import annotations

import pytest

from napari_vipp.installer import file_ops


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_atomic_replace_waits_for_windows_lock_without_deleting_old_file(
    tmp_path, monkeypatch, winerror,
):
    source = tmp_path / "new.tmp"
    target = tmp_path / "owned.json"
    source.write_bytes(b"new")
    target.write_bytes(b"previous")
    original = file_ops.os.replace
    attempts, delays = [], []

    def briefly_locked(source_path, destination_path):
        attempts.append(source_path)
        if len(attempts) < 4:
            assert target.read_bytes() == b"previous"
            error = PermissionError(13, "temporarily open")
            error.winerror = winerror
            raise error
        return original(source_path, destination_path)

    monkeypatch.setattr(file_ops.os, "name", "nt")
    monkeypatch.setattr(file_ops.os, "replace", briefly_locked)
    monkeypatch.setattr(file_ops.time, "sleep", delays.append)
    file_ops.replace_with_retry(source, target)

    assert target.read_bytes() == b"new"
    assert not source.exists()
    assert len(attempts) == 4
    assert delays == list(file_ops.ATOMIC_REPLACE_RETRY_DELAYS[:3])


@pytest.mark.parametrize(
    ("platform", "error_number"), [("nt", 13), ("nt", 22), ("posix", 13)],
)
def test_permanent_or_unrelated_error_preserves_both_files(
    tmp_path, monkeypatch, platform, error_number,
):
    source = tmp_path / "new.tmp"
    target = tmp_path / "owned.json"
    source.write_bytes(b"new")
    target.write_bytes(b"previous")
    attempts = []

    def cannot_replace(*_args):
        attempts.append(True)
        raise OSError(error_number, "cannot replace")

    monkeypatch.setattr(file_ops.os, "name", platform)
    monkeypatch.setattr(file_ops.os, "replace", cannot_replace)
    monkeypatch.setattr(file_ops.time, "sleep", lambda _seconds: None)
    with pytest.raises(OSError):
        file_ops.replace_with_retry(source, target)

    assert target.read_bytes() == b"previous"
    assert source.read_bytes() == b"new"
    expected = len(file_ops.ATOMIC_REPLACE_RETRY_DELAYS) + 1
    assert len(attempts) == (expected if platform == "nt" and error_number == 13 else 1)
