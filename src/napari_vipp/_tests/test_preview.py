from __future__ import annotations

import numpy as np

from napari_vipp.core.operations import gaussian_blur, otsu_threshold
from napari_vipp.core.preview import make_preview, normalize_thumbnail


def test_slice_preview_reduces_3d_to_2d():
    data = np.arange(5 * 6 * 7).reshape(5, 6, 7)

    preview = make_preview(data, mode="slice")

    assert preview.shape == (6, 7)


def test_mip_preview_reduces_3d_to_2d():
    data = np.zeros((3, 4, 5), dtype=np.float32)
    data[1, 2, 3] = 10

    preview = make_preview(data, mode="mip")

    assert preview.shape == (4, 5)
    assert preview[2, 3] == 10


def test_thumbnail_is_rgb_uint8():
    data = np.arange(32 * 48, dtype=np.float32).reshape(32, 48)

    thumb = normalize_thumbnail(data)

    assert thumb.dtype == np.uint8
    assert thumb.ndim == 3
    assert thumb.shape[-1] == 3
    assert thumb.shape[0] <= 110
    assert thumb.shape[1] <= 180


def test_gaussian_and_otsu_pipeline_outputs_mask():
    data = np.zeros((12, 12), dtype=np.float32)
    data[4:8, 4:8] = 1.0

    blurred = gaussian_blur(data, sigma=1.0)
    mask = otsu_threshold(blurred)

    assert blurred.shape == data.shape
    assert mask.dtype == bool
    assert mask.shape == data.shape
    assert mask.any()
