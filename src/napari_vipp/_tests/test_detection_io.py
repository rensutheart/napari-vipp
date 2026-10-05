"""Lossless template-pair persistence and explicit unsupported-format failures."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import zarr

from napari_vipp.core.detection import find_peaks, template_match
from napari_vipp.core.io import inspect_image_state, read_image, write_image
from napari_vipp.core.io.detection_state import restore_template_state
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array


def _pair(rank=2, unit="micrometer", singleton_z=False):
    shape = (4, 5) if rank == 2 else (3, 4, 5)
    template = np.random.default_rng(818).normal(size=shape)
    search_shape = tuple(size * 3 for size in shape)
    if singleton_z:
        search_shape = (shape[0], *search_shape[1:])
    search = np.zeros(search_shape)
    search[tuple(slice(0, size) for size in shape)] = template

    def state(array):
        axes = "yx" if rank == 2 else "zyx"
        return image_state_from_array(
            array,
            axes=tuple(
                AxisMetadata(axis, "space", unit, 0.5 * (i + 1), -4.0 + i)
                for i, axis in enumerate(axes)
            ),
            source_name="test source",
        )

    return template_match(
        search, template, search_state=state(search), template_state=state(template)
    )


@pytest.mark.parametrize(
    "format,suffix",
    [
        ("tiff", ".tif"),
        ("ome-tiff", ".ome.tif"),
        ("ome-zarr-0.4", ".zarr"),
        ("ome-zarr-0.5", ".zarr"),
    ],
)
@pytest.mark.parametrize("rank", [2, 3])
def test_supported_formats_preserve_exact_pair_dtype_kind_grid_and_evidence(
    tmp_path, format, suffix, rank
):
    scores, valid, ss, ms = _pair(rank)
    loaded = []
    for index, (data, state) in enumerate(((scores, ss), (valid, ms))):
        path = write_image(
            data, tmp_path / f"{index}{suffix}", format=format, image_state=state
        )
        metadata_only = inspect_image_state(path)
        dataset = read_image(path)
        actual = np.asarray(dataset.data)
        assert actual.dtype == data.dtype
        np.testing.assert_array_equal(actual, data)
        for restored in (metadata_only, dataset.image_state):
            assert restored.template_match_metadata == state.template_match_metadata
            assert restored.kind == state.kind
            assert restored.axes == state.axes
        loaded.append((actual, dataset.image_state))
    table = find_peaks(
        loaded[0][0],
        loaded[1][0],
        image_state=loaded[0][1],
        mask_state=loaded[1][1],
        minimum_value=0.999,
    )
    expected = find_peaks(
        scores, valid, image_state=ss, mask_state=ms, minimum_value=0.999
    )
    assert table.rows == expected.rows
    assert table.detection_metadata == expected.detection_metadata


@pytest.mark.parametrize(
    "format,suffix",
    [("tiff", ".tif"), ("ome-tiff", ".ome.tif"), ("ome-zarr-0.4", ".zarr")],
)
def test_pixel_calibration_is_not_manufactured_as_physical_units(
    tmp_path, format, suffix
):
    scores, _, state, _ = _pair(unit="pixel")
    path = write_image(
        scores, tmp_path / f"scores{suffix}", format=format, image_state=state
    )
    restored = read_image(path).image_state
    assert tuple(axis.unit for axis in restored.axes) == ("pixel", "pixel")


def test_imagej_rejects_tagged_outputs_before_its_lossy_dtype_path(tmp_path):
    scores, valid, ss, ms = _pair()
    for index, (data, state) in enumerate(((scores, ss), (valid, ms))):
        with pytest.raises(ValueError, match="ImageJ TIFF cannot preserve"):
            write_image(
                data, tmp_path / f"{index}.tif", format="imagej-tiff", image_state=state
            )


@pytest.mark.parametrize("format", ["tiff", "ome-tiff"])
def test_tiff_refuses_singleton_z_while_zarr_retains_rank(tmp_path, format):
    scores, _, ss, _ = _pair(3, singleton_z=True)
    with pytest.raises(ValueError, match="singleton Z"):
        write_image(scores, tmp_path / "score.ome.tif", format=format, image_state=ss)
    path = write_image(
        scores, tmp_path / "score.zarr", format="ome-zarr-0.4", image_state=ss
    )
    assert read_image(path).image_state.shape == scores.shape


@pytest.mark.parametrize("change", ["dtype", "shape", "unit", "missing_tag"])
def test_corrupt_carried_state_cannot_be_silently_dropped(change):
    scores, _, ss, _ = _pair()
    payload = deepcopy(ss.to_dict())
    if change == "dtype":
        payload["dtype"] = "float32"
    elif change == "shape":
        payload["shape"][0] += 1
    elif change == "unit":
        payload["axes"][0]["unit"] = "nm"
    else:
        payload.pop("template_match_metadata")
    with pytest.raises(ValueError):
        restore_template_state(payload, ss, scores)


def test_native_ngff_translation_is_present_and_conflicting_edits_fail(tmp_path):
    scores, _, ss, _ = _pair()
    path = write_image(
        scores, tmp_path / "scores.zarr", format="ome-zarr-0.4", image_state=ss
    )
    group = zarr.open_group(str(path), mode="a")
    multiscales = group.attrs["multiscales"]
    transforms = multiscales[0]["datasets"][0]["coordinateTransformations"]
    translation = next(item for item in transforms if item["type"] == "translation")
    assert translation["translation"] == [axis.translation for axis in ss.axes]
    translation["translation"][0] += 5
    group.attrs["multiscales"] = multiscales
    with pytest.raises(ValueError, match="calibration conflicts"):
        read_image(path)


def test_changed_output_grid_is_not_persisted_with_stale_detection_tag(tmp_path):
    scores, _, ss, _ = _pair()
    changed = replace(ss, axes=(replace(ss.axes[0], translation=99), ss.axes[1]))
    with pytest.raises(ValueError, match="calibration changed"):
        write_image(scores, tmp_path / "scores.tif", format="tiff", image_state=changed)
