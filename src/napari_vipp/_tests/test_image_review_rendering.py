"""Mask presentation preserves exact source buffers and logical foreground."""

import numpy as np
import pytest
from napari.components import ViewerModel
from napari.layers import Image

from napari_vipp.core.host_memory import HostMemorySnapshot
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.review_images import prepare_review_input
from napari_vipp.ui.image_review_rendering import (
    _canonical_boolean_storage,
    add_mask_review_layer,
    apply_mask_review_style,
    prepare_mask_review_data,
)


def _mask_input(data):
    axes = tuple(AxisMetadata(name, "space") for name in "zyx"[-data.ndim :])
    return prepare_review_input(data, image_state_from_array(data, axes=axes))


def test_volume_mask_borrows_uint8_storage_and_keeps_native_grid():
    data = np.zeros((5, 7, 9), bool)
    data[1:4, 2:5, 3:6] = True
    before = data.tobytes()
    image = _mask_input(data)
    layer = add_mask_review_layer(
        ViewerModel(ndisplay=3), image, display=3, scale=(2, 0.5, 0.4)
    )
    assert isinstance(layer, Image)
    assert layer.data.dtype == np.dtype(np.uint8)
    assert layer.data.shape == data.shape
    assert np.shares_memory(layer.data, data)
    assert not layer.data.flags.writeable and data.flags.writeable
    assert layer.rendering == "iso" and layer.iso_threshold == 0.5
    assert layer.interpolation3d == "linear"
    assert layer.interpolation2d == "nearest"
    assert tuple(layer.contrast_limits) == (0, 1)
    assert tuple(layer.contrast_limits_range) == (0, 255)
    np.testing.assert_array_equal(layer.scale, (2, 0.5, 0.4))
    apply_mask_review_style(layer, color="#00FF00", opacity=0.4, display=3)
    assert layer.opacity == 0.4 and layer.blending == "translucent"
    shaded = layer.colormap.map([0.2, 0.8])
    assert shaded[0, 1] < shaded[1, 1]
    assert np.all(shaded[:, 3] == 1)
    assert data.tobytes() == before


def test_slice_mask_remains_boolean_nearest_and_background_transparent():
    data = np.array([[False, True], [True, False]])
    layer = add_mask_review_layer(ViewerModel(), _mask_input(data), display=2)
    apply_mask_review_style(layer, color="#FF0000", opacity=0.7, display=2)
    assert layer.data.dtype == bool and np.shares_memory(layer.data, data)
    assert not layer.data.flags.writeable and data.flags.writeable
    assert layer.interpolation2d == "nearest"
    np.testing.assert_array_equal(
        layer.colormap.map([0, 1]), [[0, 0, 0, 0], [1, 0, 0, 1]]
    )


def test_noncanonical_true_bytes_have_a_separate_binary_presentation_buffer(
    monkeypatch,
):
    from napari_vipp.ui import image_review_rendering as rendering

    snapshot = HostMemorySnapshot(
        platform="win32",
        source="windows_global_memory_status_ex",
        physical_total_bytes=8 * 1024**3,
        physical_available_bytes=7 * 1024**3,
        commit_limit_bytes=16 * 1024**3,
        commit_available_bytes=12 * 1024**3,
    )
    monkeypatch.setattr(rendering, "capture_host_memory", lambda: snapshot)
    raw = np.zeros((3, 4, 5), np.uint8)
    raw[1, 1, 1] = 1
    raw[1, 2, 2] = 255
    data = raw.view(bool)
    before = raw.tobytes()
    layer = add_mask_review_layer(ViewerModel(ndisplay=3), _mask_input(data), display=3)
    assert isinstance(layer, Image) and layer.rendering == "iso"
    assert not np.shares_memory(layer.data, data)
    assert layer.data.dtype == np.uint8 and not layer.data.flags.writeable
    assert layer.data.nbytes == data.size
    np.testing.assert_array_equal(layer.data, data.astype(np.uint8))
    apply_mask_review_style(layer, color="#00FF00", opacity=0.5, display=3)
    assert raw.tobytes() == before
    assert bool(data[1, 1, 1]) and bool(data[1, 2, 2])


def test_mask_presentation_buffer_refuses_unavailable_memory_before_allocating(
    monkeypatch,
):
    from napari_vipp.ui import image_review_rendering as rendering

    data = np.full((3, 4, 5), 255, np.uint8).view(bool)
    image = _mask_input(data)
    monkeypatch.setattr(
        rendering,
        "capture_host_memory",
        lambda: HostMemorySnapshot.unavailable("win32", "deterministic test"),
    )

    def forbidden_allocation(*_args, **_kwargs):
        raise AssertionError("A denied display allocation must not be attempted")

    monkeypatch.setattr(rendering.np, "empty", forbidden_allocation)
    with pytest.raises(ValueError, match="Free memory or review a smaller crop"):
        add_mask_review_layer(None, image, display=3)
    assert data.view(np.uint8).tobytes() == bytes([255]) * data.size


def test_mask_presentation_allocation_failure_is_actionable(monkeypatch):
    from napari_vipp.ui import image_review_rendering as rendering

    data = np.full((3, 4, 5), 255, np.uint8).view(bool)
    image = _mask_input(data)
    snapshot = HostMemorySnapshot(
        platform="win32",
        source="windows_global_memory_status_ex",
        physical_total_bytes=8 * 1024**3,
        physical_available_bytes=7 * 1024**3,
        commit_limit_bytes=16 * 1024**3,
        commit_available_bytes=12 * 1024**3,
    )
    monkeypatch.setattr(rendering, "capture_host_memory", lambda: snapshot)

    def failed_allocation(*_args, **_kwargs):
        raise MemoryError("deterministic allocation failure")

    monkeypatch.setattr(rendering.np, "empty", failed_allocation)
    with pytest.raises(ValueError, match="one-byte-per-voxel display conversion"):
        add_mask_review_layer(None, image, display=3)
    assert data.view(np.uint8).tobytes() == bytes([255]) * data.size


def test_boolean_storage_check_handles_strided_arrays_in_bounded_chunks(monkeypatch):
    from napari_vipp.ui import image_review_rendering as rendering

    monkeypatch.setattr(rendering, "_STORAGE_CHUNK_SIZE", 7)
    data = np.zeros((20, 30, 40), np.uint8)
    data[5, 7, 9] = 255
    assert not _canonical_boolean_storage(data.view(bool).transpose(2, 0, 1))
    data[5, 7, 9] = 1
    assert _canonical_boolean_storage(data.view(bool).transpose(2, 0, 1))


def test_prepared_mask_display_buffer_is_reused_across_panes_and_mode_switches(
    monkeypatch,
):
    from napari_vipp.ui import image_review_rendering as rendering

    data = np.full((3, 4, 5), 255, np.uint8).view(bool)
    image = _mask_input(data)
    snapshot = HostMemorySnapshot(
        platform="win32",
        source="windows_global_memory_status_ex",
        physical_total_bytes=8 * 1024**3,
        physical_available_bytes=7 * 1024**3,
        commit_limit_bytes=16 * 1024**3,
        commit_available_bytes=12 * 1024**3,
    )
    monkeypatch.setattr(rendering, "capture_host_memory", lambda: snapshot)
    prepared = prepare_mask_review_data(image)

    def forbidden_preparation(*_args, **_kwargs):
        raise AssertionError("The window's presentation buffer must be reused")

    monkeypatch.setattr(rendering, "prepare_mask_review_data", forbidden_preparation)
    for display in (3, 3, 2, 3):
        layer = add_mask_review_layer(
            ViewerModel(ndisplay=display),
            image,
            display=display,
            prepared_data=prepared,
        )
        assert layer.data is prepared if display == 3 else np.shares_memory(
            layer.data, data
        )
    assert not prepared.flags.writeable
    assert data.view(np.uint8).tobytes() == bytes([255]) * data.size
