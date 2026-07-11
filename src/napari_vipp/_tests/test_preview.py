from __future__ import annotations

import numpy as np

from napari_vipp._sample_data import make_sample_data
from napari_vipp.core.metadata import (
    ChannelMetadata,
    image_state_from_array,
    transform_image_state,
    transform_split_output_state,
)
from napari_vipp.core.operations import composite_to_rgb, gaussian_blur, otsu_threshold
from napari_vipp.core.preview import (
    make_preview,
    normalize_thumbnail,
    normalize_thumbnail_with_colormap,
    thumbnail_channel_contrast_limits,
    thumbnail_contrast_limits,
)


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


def test_thumbnail_contrast_modes_include_minmax_and_raw():
    data = np.asarray([[1000, 2000]], dtype=np.uint16)

    raw = normalize_thumbnail_with_colormap(
        data,
        size=(2, 1),
        colormap="Gray",
        contrast_mode="Raw",
    )
    minmax = normalize_thumbnail_with_colormap(
        data,
        size=(2, 1),
        colormap="Gray",
        contrast_mode="Min-max",
    )

    assert raw is not None
    assert minmax is not None
    assert raw[0, 0, 0] < 10
    assert raw[0, 1, 0] < 10
    assert minmax[0, 0, 0] == 0
    assert minmax[0, 1, 0] == 255


def test_raw_float_thumbnail_scales_relative_to_finite_data_range():
    data = np.asarray(
        [
            [0.0, 5000.0],
            [2500.0, np.nan],
        ],
        dtype=np.float32,
    )

    thumb = normalize_thumbnail_with_colormap(
        data,
        size=(2, 2),
        colormap="Gray",
        contrast_mode="Raw",
    )

    assert thumb is not None
    assert thumb[0, 0, 0] == 0
    assert thumb[0, 1, 0] == 255
    assert 120 <= thumb[1, 0, 0] <= 130
    assert thumb[1, 1, 0] == 0


def test_raw_float_thumbnail_can_use_stack_contrast_reference():
    stack = np.zeros((2, 2, 2), dtype=np.float32)
    stack[0, 0, 0] = 1.0
    stack[1, 0, 0] = 10.0
    slice_view = stack[0]

    slice_scaled = normalize_thumbnail_with_colormap(
        slice_view,
        size=(2, 2),
        colormap="Gray",
        contrast_mode="Raw",
        contrast_reference=slice_view,
    )
    stack_scaled = normalize_thumbnail_with_colormap(
        slice_view,
        size=(2, 2),
        colormap="Gray",
        contrast_mode="Raw",
        contrast_reference=stack,
    )

    assert slice_scaled is not None
    assert stack_scaled is not None
    assert slice_scaled[0, 0, 0] == 255
    assert 20 <= stack_scaled[0, 0, 0] <= 30


def test_raw_float_thumbnail_can_use_cached_stack_contrast_limits():
    stack = np.zeros((2, 2, 2), dtype=np.float32)
    stack[0, 0, 0] = 1.0
    stack[1, 0, 0] = 10.0
    slice_view = stack[0]
    limits = thumbnail_contrast_limits(stack, contrast_mode="Raw")

    stack_scaled = normalize_thumbnail_with_colormap(
        slice_view,
        size=(2, 2),
        colormap="Gray",
        contrast_mode="Raw",
        contrast_limits=limits,
    )

    assert stack_scaled is not None
    assert limits == (0.0, 10.0)
    assert 20 <= stack_scaled[0, 0, 0] <= 30


def test_mask_thumbnail_contrast_limits_do_not_scan_the_stack():
    class UnreadableMask:
        def __array__(self):
            raise AssertionError("mask pixels should not be inspected")

    assert thumbnail_contrast_limits(UnreadableMask(), data_kind="mask") == (
        0.0,
        1.0,
    )


def test_percentile_thumbnail_keeps_sparse_foreground_visible_over_low_ramp():
    yy, xx = np.indices((64, 64), dtype=np.uint16)
    data = ((yy + xx) % 32).astype(np.uint16)
    data[18:23, 18:26] = 34_000

    thumb = normalize_thumbnail_with_colormap(
        data,
        size=(64, 64),
        colormap="Gray",
        contrast_mode="Percentile",
    )

    assert thumb is not None
    assert thumb[0, :, 0].max() < 5
    assert thumb[20, 20, 0] > 200


def test_mesh_morphology_empty_z_slice_thumbnail_stays_dark():
    data, layer_kwargs, _layer_type = next(
        sample
        for sample in make_sample_data()
        if sample[1]["name"] == "VIPP synthetic 3D mesh morphology"
    )
    state = image_state_from_array(
        data,
        layer_metadata=layer_kwargs["metadata"],
    )

    preview = make_preview(
        data,
        mode="slice",
        current_step=(0, 0, 0),
        current_step_nsteps=data.shape,
        state=state,
    )
    thumb = normalize_thumbnail_with_colormap(
        preview,
        size=(64, 64),
        colormap="Gray",
        contrast_mode="Percentile",
    )

    assert thumb is not None
    assert thumb[..., :3].max() == 0


def test_signed_difference_thumbnail_renders_zero_background_black():
    # A Subtract of two boolean masks yields a signed difference image with
    # values in {-1, 0, 1}. The zero background must render black (like the
    # inspector view) rather than mid-grey.
    data = np.zeros((20, 20), dtype=np.float32)
    data[8:12, :] = 1.0  # foreground band
    data[0:4, 8:12] = -1.0  # negative lobe

    thumb = normalize_thumbnail(data, size=(20, 20))

    assert thumb is not None
    assert tuple(thumb[15, 2]) == (0, 0, 0)  # zero background -> black
    assert tuple(thumb[2, 10]) == (0, 0, 0)  # negative lobe -> black
    assert thumb[10, 2, 0] > 200  # positive band -> bright


def test_label_thumbnail_ignores_raw_int32_scaling():
    labels = np.zeros((8, 9), dtype=np.int32)
    labels[2:5, 2:5] = 1
    labels[5:7, 5:8] = 2

    thumb = normalize_thumbnail_with_colormap(
        labels,
        size=(9, 8),
        colormap="Gray",
        contrast_mode="Raw",
        data_kind="labels",
    )

    assert thumb is not None
    assert tuple(thumb[0, 0]) == (0, 0, 0)
    assert thumb[3, 3].sum() > 0
    assert thumb[6, 6].sum() > 0
    assert not np.array_equal(thumb[3, 3], thumb[6, 6])


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


def test_multichannel_state_preview_uses_requested_channel_colours():
    data = np.zeros((2, 8, 9), dtype=np.uint16)
    data[0, 2:5, 2:5] = 2000
    data[1, 4:7, 5:8] = 2000
    state = image_state_from_array(data, layer_metadata={"axes": "CYX"})

    preview = make_preview(
        data,
        mode="slice",
        state=state,
        channel_colors=["Yellow", "Cyan"],
    )

    assert preview.shape == (8, 9, 3)
    yellow_pixel = preview[3, 3]
    cyan_pixel = preview[5, 6]
    assert yellow_pixel[0] > 0
    assert yellow_pixel[1] > 0
    assert yellow_pixel[2] == 0
    assert cyan_pixel[0] == 0
    assert cyan_pixel[1] > 0
    assert cyan_pixel[2] > 0


def test_multichannel_state_preview_uses_metadata_channel_colours():
    data = np.zeros((2, 8, 9), dtype=np.uint16)
    data[0, 2:5, 2:5] = 2000
    data[1, 4:7, 5:8] = 2000
    state = image_state_from_array(
        data,
        layer_metadata={"axes": "CYX"},
        channels=(
            ChannelMetadata(name="first", color=0xFFFF00),
            ChannelMetadata(name="second", color=0x00FFFF),
        ),
    )

    preview = make_preview(data, mode="slice", state=state)

    assert preview.shape == (8, 9, 3)
    yellow_pixel = preview[3, 3]
    cyan_pixel = preview[5, 6]
    assert yellow_pixel[0] > 0
    assert yellow_pixel[1] > 0
    assert yellow_pixel[2] == 0
    assert cyan_pixel[0] == 0
    assert cyan_pixel[1] > 0
    assert cyan_pixel[2] > 0


def test_monochrome_thumbnail_colormap_changes_single_channel_colour():
    data = np.zeros((8, 9), dtype=np.uint16)
    data[2:5, 2:5] = 2000

    gray = normalize_thumbnail_with_colormap(data, size=(9, 8), colormap="Gray")
    green = normalize_thumbnail_with_colormap(data, size=(9, 8), colormap="Green")

    assert gray is not None
    assert green is not None
    assert gray[3, 3, 0] == gray[3, 3, 1] == gray[3, 3, 2]
    assert green[3, 3, 0] == 0
    assert green[3, 3, 1] > 0
    assert green[3, 3, 2] == 0


def test_requested_blue_channel_stays_blue_in_thumbnail():
    data = np.zeros((2, 8, 9), dtype=np.uint16)
    data[0, 2:5, 2:5] = 2000
    data[1, 4:7, 5:8] = 2000
    state = image_state_from_array(data, layer_metadata={"axes": "CYX"})

    preview = make_preview(
        data,
        mode="slice",
        state=state,
        channel_colors=["Blue", "Green"],
    )
    thumbnail = normalize_thumbnail(preview, size=(9, 8))

    assert thumbnail is not None
    blue_pixel = thumbnail[3, 3]
    green_pixel = thumbnail[5, 6]
    assert blue_pixel[0] == 0
    assert blue_pixel[1] == 0
    assert blue_pixel[2] > 0
    assert green_pixel[0] == 0
    assert green_pixel[1] > 0
    assert green_pixel[2] == 0


def test_multichannel_stack_contrast_scope_uses_full_channel_range():
    data = np.zeros((1, 2, 8, 9), dtype=np.float32)
    data[0, 0, 2:5, 2:5] = 1.0
    data[0, 1, 2:5, 2:5] = 10.0
    state = image_state_from_array(data, layer_metadata={"axes": "CZYX"})

    slice_scaled = make_preview(
        data,
        mode="slice",
        current_step=(0, 0, 0, 0),
        state=state,
        channel_colors=["Green"],
        contrast_mode="Raw",
        contrast_scope="Slice",
    )
    stack_scaled = make_preview(
        data,
        mode="slice",
        current_step=(0, 0, 0, 0),
        state=state,
        channel_colors=["Green"],
        contrast_mode="Raw",
        contrast_scope="Stack",
    )

    assert slice_scaled is not None
    assert stack_scaled is not None
    assert slice_scaled[3, 3, 1] > 0.95
    assert 0.05 < stack_scaled[3, 3, 1] < 0.15


def test_multichannel_preview_can_use_cached_stack_contrast_limits():
    data = np.zeros((1, 2, 8, 9), dtype=np.float32)
    data[0, 0, 2:5, 2:5] = 1.0
    data[0, 1, 2:5, 2:5] = 10.0
    state = image_state_from_array(data, layer_metadata={"axes": "CZYX"})
    limits = thumbnail_channel_contrast_limits(
        data,
        channel_axis=0,
        contrast_mode="Raw",
    )

    stack_scaled = make_preview(
        data,
        mode="slice",
        current_step=(0, 0, 0, 0),
        state=state,
        channel_colors=["Green"],
        contrast_mode="Raw",
        contrast_scope="Stack",
        contrast_limits=limits,
    )

    assert stack_scaled is not None
    assert limits == ((0.0, 10.0),)
    assert 0.05 < stack_scaled[3, 3, 1] < 0.15


def test_time_lapse_multichannel_preview_follows_current_step():
    data = np.zeros((2, 3, 4, 8, 9), dtype=np.uint16)
    data[0, 0, 0, 2, 3] = 2000
    data[1, 0, 3, 5, 6] = 4000
    state = image_state_from_array(data, layer_metadata={"axes": "TCZYX"})

    first = make_preview(data, mode="slice", current_step=(0, 0, 0, 0, 0), state=state)
    second = make_preview(
        data,
        mode="slice",
        current_step=(1, 0, 3, 0, 0),
        state=state,
    )

    assert first.shape == (8, 9, 3)
    assert second.shape == (8, 9, 3)
    assert first[2, 3].sum() > 0
    assert second[5, 6].sum() > 0
    assert first[5, 6].sum() == 0
    assert second[2, 3].sum() == 0


def test_composite_to_rgb_preview_uses_source_z_axis_when_rgb_axis_is_hidden():
    data = np.zeros((3, 4, 8, 9), dtype=np.uint16)
    data[:, 0, 2, 3] = 2000
    data[:, 3, 5, 6] = 4000
    source_state = image_state_from_array(data, layer_metadata={"axes": "CZYX"})
    rgb = composite_to_rgb(data)
    rgb_state = transform_image_state(
        rgb,
        source_state,
        operation_id="composite_to_rgb",
        operation_title="Composite to RGB",
        params={},
    )

    first = make_preview(
        rgb,
        mode="slice",
        current_step=(0, 0, 0, 0),
        state=rgb_state,
    )
    second = make_preview(
        rgb,
        mode="slice",
        current_step=(0, 3, 0, 0),
        state=rgb_state,
    )

    assert rgb_state.axes[0].name == "z"
    assert rgb_state.axes[0].source_axis == 1
    assert first[2, 3].sum() > 0
    assert second[5, 6].sum() > 0
    assert first[5, 6].sum() == 0
    assert second[2, 3].sum() == 0


def test_split_channel_preview_maps_z_to_source_viewer_axis():
    source = np.zeros((2, 3, 4, 8, 9), dtype=np.uint16)
    red = source[:, 0]
    red[0, 0, 2, 3] = 2000
    red[0, 3, 5, 6] = 4000
    source_state = image_state_from_array(source, layer_metadata={"axes": "TCZYX"})
    split_state = transform_split_output_state(
        red,
        source_state,
        operation_id="split_channels",
        operation_title="Split Channels",
        port_name="Ch 1",
        params={},
    )
    mask = red > 0
    mask_state = transform_image_state(
        mask,
        split_state,
        operation_id="otsu_threshold",
        operation_title="Otsu Threshold",
        params={},
    )

    first = make_preview(
        mask,
        mode="slice",
        current_step=(0, 0, 0, 0, 0),
        state=mask_state,
    )
    second = make_preview(
        mask,
        mode="slice",
        current_step=(0, 0, 3, 0, 0),
        state=mask_state,
    )

    assert mask_state.axes[1].name == "z"
    assert mask_state.axes[1].source_axis == 2
    assert first[2, 3]
    assert second[5, 6]
    assert not first[5, 6]
    assert not second[2, 3]


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
