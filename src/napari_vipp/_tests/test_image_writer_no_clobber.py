"""No-overwrite writes publish complete outputs without check/use clobbering."""

from __future__ import annotations

import errno
import os
from types import SimpleNamespace

import numpy as np
import pytest

from napari_vipp.core.io import registry


@pytest.mark.parametrize(
    ("format", "filename"),
    [("npy", "out.npy"), ("tiff", "out.tif"), ("png", "out.png")],
)
def test_new_output_round_trip_and_original_unchanged(format, filename, tmp_path):
    data = np.arange(30, dtype=np.uint8).reshape(5, 6)
    data.flags.writeable = False
    original = data.copy()
    target = tmp_path / filename
    assert registry.write_image(data, target, format=format, overwrite=False) == target
    np.testing.assert_array_equal(registry.read_image(target).data, original)
    np.testing.assert_array_equal(data, original)
    assert sorted(tmp_path.iterdir()) == [target]


def test_racing_file_survives_with_complete_original_bytes(monkeypatch, tmp_path):
    target = tmp_path / "out.npy"
    normal = registry.write_numpy

    def racing_writer(data, staged):
        saved = normal(data, staged)
        target.write_bytes(b"another writer's complete result")
        return saved

    monkeypatch.setattr(registry, "write_numpy", racing_writer)
    with pytest.raises(FileExistsError):
        registry.write_image(np.arange(4), target, format="npy", overwrite=False)
    assert target.read_bytes() == b"another writer's complete result"
    assert sorted(tmp_path.iterdir()) == [target]


def test_serialization_failure_never_leaves_final_output(monkeypatch, tmp_path):
    target = tmp_path / "out.npy"

    def fail(data, staged):
        staged.write_bytes(b"partial")
        raise ValueError("synthetic serializer failure")

    monkeypatch.setattr(registry, "write_numpy", fail)
    with pytest.raises(ValueError, match="serializer"):
        registry.write_image(np.arange(4), target, format="npy", overwrite=False)
    assert list(tmp_path.iterdir()) == []


def test_hardlink_unsupported_fails_without_unsafe_replace(monkeypatch, tmp_path):
    target = tmp_path / "out.npy"

    def unsupported(source, destination):
        raise OSError(errno.ENOTSUP, "synthetic no hard links")

    monkeypatch.setattr(registry.os, "link", unsupported)
    with pytest.raises(OSError, match="hard links"):
        registry.write_image(np.arange(4), target, format="npy", overwrite=False)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("directory", [False, True])
def test_existing_file_or_store_not_touched(directory, tmp_path):
    target = tmp_path / "out.npy"
    if directory:
        target.mkdir()
        (target / "content").write_bytes(b"existing")
    else:
        target.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        registry.write_image(np.arange(4), target, format="npy", overwrite=False)
    assert (target / "content" if directory else target).read_bytes() == b"existing"


def test_dangling_symlink_not_touched(tmp_path):
    target = tmp_path / "out.npy"
    original = tmp_path / "missing.npy"
    try:
        target.symlink_to(original)
    except OSError:
        pytest.skip("Creating a symlink requires OS permission.")
    with pytest.raises(FileExistsError):
        registry.write_image(np.arange(4), target, format="npy", overwrite=False)
    assert target.is_symlink()
    assert not original.exists()


def _synthetic_store_writer(data, path, **kwargs):
    path.mkdir()
    (path / "zarr.json").write_text('{"zarr_format": 3}')
    (path / "synthetic_pixels").write_bytes(np.asarray(data).tobytes())
    return path


def test_new_store_promoted_as_one_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(registry, "write_ome_zarr", _synthetic_store_writer)
    target = tmp_path / "out.ome.zarr"
    data = np.arange(4, dtype=np.uint8)
    registry.write_image(data, target, format="ome-zarr-0.5", overwrite=False)
    assert (target / "synthetic_pixels").read_bytes() == data.tobytes()
    assert sorted(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("race_kind", ["directory", "empty-directory", "file"])
def test_racing_store_or_file_never_deleted(monkeypatch, tmp_path, race_kind):
    target = tmp_path / "out.ome.zarr"

    def racing_writer(data, path, **kwargs):
        saved = _synthetic_store_writer(data, path, **kwargs)
        if race_kind in {"directory", "empty-directory"}:
            target.mkdir()
            if race_kind == "directory":
                (target / "original").write_bytes(b"other writer")
        else:
            target.write_bytes(b"other writer")
        return saved

    monkeypatch.setattr(registry, "write_ome_zarr", racing_writer)
    with pytest.raises(OSError):
        registry.write_image(np.arange(4), target, format="ome-zarr", overwrite=False)
    if race_kind == "empty-directory":
        assert target.is_dir()
        assert list(target.iterdir()) == []
    else:
        assert (
            target / "original" if target.is_dir() else target
        ).read_bytes() == b"other writer"
    assert sorted(tmp_path.iterdir()) == [target]


def test_explicit_overwrite_retains_ordinary_behavior(tmp_path):
    target = tmp_path / "out.npy"
    target.write_bytes(b"replace me")
    data = np.arange(6)
    registry.write_image(data, target, format="npy", overwrite=True)
    np.testing.assert_array_equal(np.load(target), data)
    assert sorted(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize(
    ("platform", "function_name", "flags"),
    [("linux", "renameat2", 1), ("darwin", "renamex_np", 4)],
)
def test_posix_uses_native_exclusive_rename(
    monkeypatch, platform, function_name, flags, tmp_path
):
    calls = []

    class NativeFunction:
        def __call__(self, *args):
            calls.append(args)
            return 0

    library = SimpleNamespace(**{function_name: NativeFunction()})
    monkeypatch.setattr(
        registry, "os", SimpleNamespace(name="posix", fsencode=os.fsencode)
    )
    monkeypatch.setattr(registry.sys, "platform", platform)
    monkeypatch.setattr(registry.ctypes, "CDLL", lambda *args, **kwargs: library)
    registry._rename_directory_no_replace(tmp_path / "staged", tmp_path / "target")
    assert calls[0][-1] == flags
    assert os.fsencode(tmp_path / "staged") in calls[0]
    assert os.fsencode(tmp_path / "target") in calls[0]


def test_unsupported_directory_publication_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(registry, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(registry.sys, "platform", "unqualified-platform")
    monkeypatch.setattr(
        registry.ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace()
    )
    with pytest.raises(OSError) as failure:
        registry._rename_directory_no_replace(tmp_path / "staged", tmp_path / "target")
    assert failure.value.errno == errno.ENOTSUP


def test_native_directory_collision_preserves_errno(monkeypatch, tmp_path):
    class NativeFunction:
        def __call__(self, *args):
            return -1

    monkeypatch.setattr(
        registry, "os", SimpleNamespace(name="posix", fsencode=os.fsencode)
    )
    monkeypatch.setattr(registry.sys, "platform", "linux")
    monkeypatch.setattr(
        registry.ctypes,
        "CDLL",
        lambda *args, **kwargs: SimpleNamespace(renameat2=NativeFunction()),
    )
    monkeypatch.setattr(registry.ctypes, "get_errno", lambda: errno.EEXIST)
    with pytest.raises(FileExistsError):
        registry._rename_directory_no_replace(tmp_path / "staged", tmp_path / "target")
