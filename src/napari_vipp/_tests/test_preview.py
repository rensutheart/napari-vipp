from __future__ import annotations

import numpy as np

from napari_vipp.core.metadata import image_state_from_array
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


def test_multichannel_state_preview_renders_fluorescence_composite():
    data = np.zeros((3, 4, 12, 14), dtype=np.uint16)
    data[0, :, 2:8, 3:9] = 2000
    data[1, :, 4:10, 5:11] = 4000
    data[2, :, 6:12, 7:13] = 6000
    state = image_state_from_array(data, layer_metadata={"axes": "CZYX"})

    preview = make_preview(data, mode="slice", state=state)
    thumbnail = normalize_thumbnail(preview)

    assert preview.shape == (12, 14, 3)
    assert thumbnail is not None
    assert thumbnail.shape[-1] == 3


def test_rgb_channel_last_state_preview_preserves_color_axis():
    data = np.zeros((4, 12, 14, 3), dtype=np.float32)
    data[:, 2:8, 3:9, 0] = 1.0
    data[:, 4:10, 5:11, 1] = 0.5
    state = image_state_from_array(
        data,
        layer_metadata={
            "axes": [
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
                {"name": "rgb", "type": "channel"},
            ],
        },
    )

    preview = make_preview(data, mode="slice", state=state)

    assert preview.shape == (12, 14, 3)
    assert preview[..., 0].max() == 1.0
    assert preview[..., 1].max() == 0.5


def test_gaussian_and_otsu_pipeline_outputs_mask():
    data = np.zeros((12, 12), dtype=np.float32)
    data[4:8, 4:8] = 1.0

    blurred = gaussian_blur(data, sigma=1.0)
    mask = otsu_threshold(blurred)

    assert blurred.shape == data.shape
    assert mask.dtype == bool
    assert mask.shape == data.shape
    assert mask.any()
