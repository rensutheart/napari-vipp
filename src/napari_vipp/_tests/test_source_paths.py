from __future__ import annotations

import os
from pathlib import Path

import pytest

import napari_vipp.core.file_sources as file_sources
import napari_vipp.core.source_inspection as source_inspection
from napari_vipp.core.io import (
    normalize_local_image_source_path,
    validate_local_image_source_path,
)


def test_normalize_accepts_quoted_path_with_spaces(tmp_path: Path) -> None:
    source = tmp_path / "folder with spaces" / "sample image.tif"

    assert normalize_local_image_source_path(f'  "{source}"  ') == source


def test_validate_decodes_percent_escaped_file_uri(tmp_path: Path) -> None:
    source = tmp_path / "folder with spaces" / "sample image.tif"
    source.parent.mkdir()
    source.write_bytes(b"image")

    assert "%20" in source.as_uri()
    assert validate_local_image_source_path(source.as_uri()) == source.resolve()


@pytest.mark.skipif(os.name != "nt", reason="Windows separator semantics")
def test_normalize_accepts_mixed_windows_separators() -> None:
    source = normalize_local_image_source_path(r"C:\scientific/data\sample.ome.tif")

    assert source == Path(r"C:\scientific\data\sample.ome.tif")


def test_normalize_preserves_unc_authority_from_file_uri() -> None:
    source = normalize_local_image_source_path(
        "file://microscope-server/share%20one/sample.czi"
    )

    normalized = str(source).replace("\\", "/")
    assert normalized.startswith("//microscope-server/")
    assert normalized.endswith("/share one/sample.czi")


@pytest.mark.parametrize("uri", ["https://example.test/image.tif", "s3://data/x"])
def test_normalize_rejects_non_file_uri(uri: str) -> None:
    with pytest.raises(ValueError, match="local paths and file: URIs only"):
        normalize_local_image_source_path(uri)


def test_validate_accepts_regular_file_and_zarr_directory(tmp_path: Path) -> None:
    image = tmp_path / "sample.tif"
    image.write_bytes(b"image")
    store = tmp_path / "sample.ome.zarr"
    store.mkdir()

    assert validate_local_image_source_path(image) == image.resolve()
    assert validate_local_image_source_path(store) == store.resolve()


def test_validate_rejects_ordinary_directory_and_zarr_file(tmp_path: Path) -> None:
    ordinary_directory = tmp_path / "images"
    ordinary_directory.mkdir()
    false_store = tmp_path / "sample.zarr"
    false_store.write_bytes(b"not a directory store")

    with pytest.raises(IsADirectoryError, match="ordinary directory"):
        validate_local_image_source_path(ordinary_directory)
    with pytest.raises(ValueError, match="directory-backed store"):
        validate_local_image_source_path(false_store)


def test_validate_can_normalize_a_missing_path_without_accepting_it() -> None:
    missing = Path("missing folder") / "sample.tif"

    normalized = validate_local_image_source_path(missing, require_exists=False)
    assert normalized.is_absolute()
    with pytest.raises(FileNotFoundError, match="Local image source not found"):
        validate_local_image_source_path(missing)


@pytest.mark.parametrize(
    ("module", "entry_point"),
    [
        (source_inspection, source_inspection.inspect_local_source_item),
        (file_sources, file_sources.load_frozen_file_source_snapshot),
    ],
)
def test_core_entry_points_reject_directory_before_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module,
    entry_point,
) -> None:
    ordinary_directory = tmp_path / "unrelated-files"
    ordinary_directory.mkdir()

    def unexpected_hash(*_args, **_kwargs):
        pytest.fail("ordinary directory reached recursive identity hashing")

    monkeypatch.setattr(module, "capture_local_source_bundle", unexpected_hash)

    with pytest.raises(IsADirectoryError, match="ordinary directory"):
        entry_point(ordinary_directory)
