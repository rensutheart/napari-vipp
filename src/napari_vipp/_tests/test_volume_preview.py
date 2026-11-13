"""Volume display admission never changes the scientific array."""

from __future__ import annotations

import numpy as np
import pytest

from napari_vipp.ui.volume_preview import plan_volume_preview, preview_data

_CELL4_SHAPE = (70, 1900, 1904)
_GIB = 1024**3


def _volume(dtype):
    # Shape-accurate, no multi-GiB allocation.
    return np.broadcast_to(np.zeros((), dtype=dtype), _CELL4_SHAPE)


def test_cell4_uses_full_resolution_when_renderer_accepts():
    image = _volume(np.float32)
    plan = plan_volume_preview(
        image,
        (0, 1, 2),
        max_texture_axis=2048,
        max_texture_bytes=_GIB,
        admit=lambda xyz, size: xyz == (1904, 1900, 70) and size == 4,
    )
    assert plan.can_display_3d and not plan.is_reduced
    assert plan.shape == image.shape


def test_cell4_llvmpipe_reduces_only_preview_xy_after_proxy_rejection():
    image = _volume(np.float32)
    probes = []

    def admit(xyz, size):
        probes.append((xyz, size))
        return xyz != (1904, 1900, 70)

    plan = plan_volume_preview(
        image,
        (0, 1, 2),
        max_texture_axis=2048,
        max_texture_bytes=_GIB,
        admit=admit,
    )
    assert plan.can_display_3d and plan.is_reduced
    assert plan.strides == (1, 2, 2)
    assert plan.shape == (70, 950, 952)
    assert probes == [((1904, 1900, 70), 4), ((952, 950, 70), 4)]
    view = preview_data(image, plan)
    assert view.shape == plan.shape
    assert view.dtype == image.dtype
    assert np.shares_memory(view, image)
    assert not view.flags.writeable
    assert image.shape == _CELL4_SHAPE


def test_binary_mask_uses_one_byte_and_never_downsamples():
    mask = _volume(bool)
    limits = {"max_texture_axis": 2048, "max_texture_bytes": _GIB}
    native = plan_volume_preview(
        mask,
        (0, 1, 2),
        is_binary_mask=True,
        **limits,
        admit=lambda xyz, size: xyz == (1904, 1900, 70) and size == 1,
    )
    assert native.can_display_3d and native.shape == _CELL4_SHAPE
    refused = plan_volume_preview(
        mask,
        (0, 1, 2),
        is_binary_mask=True,
        **limits,
        admit=lambda *_: False,
    )
    assert not refused.can_display_3d and refused.shape == _CELL4_SHAPE
    with pytest.raises(ValueError, match="3D texture budget"):
        preview_data(mask, refused)


def test_cell4_labels_use_native_color_texture_budget_but_keep_their_exact_grid():
    labels = _volume(np.int32)
    requests = []

    def admit(xyz, size):
        requests.append((xyz, size))
        return size in (1, 2)

    options = dict(
        allow_reduction=False,
        max_texture_axis=2048,
        max_texture_bytes=_GIB // 2,
        admit=admit,
    )
    assert not plan_volume_preview(labels, (0, 1, 2), **options).can_display_3d
    plan = plan_volume_preview(labels, (0, 1, 2), texture_bytes_per_voxel=1, **options)

    assert requests == [((1904, 1900, 70), 1)]
    assert plan.can_display_3d and not plan.is_reduced and plan.shape == labels.shape
    view = preview_data(labels, plan)
    assert view.dtype == np.int32
    assert np.shares_memory(view, labels)


def test_only_spatial_axes_are_sampled_and_input_values_are_untouched():
    image = np.arange(2 * 3 * 4 * 6 * 8, dtype=np.float32).reshape((2, 3, 4, 6, 8))
    before = image.copy()
    plan = plan_volume_preview(image, (2, 3, 4), max_texture_axis=4)
    assert plan.strides == (1, 1, 1, 2, 2)
    assert plan.shape == (2, 3, 4, 3, 4)
    view = preview_data(image, plan)
    np.testing.assert_array_equal(view, before[:, :, :, ::2, ::2])
    np.testing.assert_array_equal(image, before)
    assert image.flags.writeable and not view.flags.writeable


def test_unavailable_proxy_keeps_byte_guard_and_categorical_grid():
    image = _volume(np.float64)
    plan = plan_volume_preview(image, (0, 1, 2), admit=lambda *_: None)
    assert plan.can_display_3d and plan.is_reduced
    exact = plan_volume_preview(image, (0, 1, 2), allow_reduction=False)
    assert not exact.can_display_3d and not exact.is_reduced
