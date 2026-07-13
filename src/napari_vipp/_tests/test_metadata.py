from __future__ import annotations

import numpy as np
import pytest

from napari_vipp.core.metadata import (
    AXIS_CONFIDENCE_EXPLICIT,
    AXIS_CONFIDENCE_INFERRED,
    AXIS_CONFIDENCE_MIXED,
    DEFERRED_VALUE_RANGE,
    AxisMetadata,
    ChannelMetadata,
    ImageState,
    image_state_from_array,
    infer_axis_metadata_from_shape,
    transform_image_state,
    transform_multi_input_image_state,
    transform_split_output_state,
    with_channel_colors,
)


def test_shape_only_channel_guess_is_marked_inferred_not_explicit():
    data = np.zeros((5, 7, 3), dtype=np.uint16)

    inferred = image_state_from_array(data)
    explicit_volume = image_state_from_array(
        data,
        layer_metadata={"axes": "ZYX"},
    )
    explicit_channels = image_state_from_array(
        data,
        layer_metadata={"axes": "YXC"},
    )
    explicit_rgb = image_state_from_array(
        data,
        layer_metadata={
            "axes": [
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
                {"name": "rgb", "type": "channel"},
            ]
        },
    )

    assert inferred.axis_order == "YXC"
    assert inferred.axis_confidence == AXIS_CONFIDENCE_INFERRED
    assert not inferred.axes_explicit
    assert not inferred.axes[-1].is_explicit
    assert explicit_volume.axis_order == "ZYX"
    assert explicit_volume.axis_confidence == AXIS_CONFIDENCE_EXPLICIT
    assert explicit_volume.axes_explicit
    assert explicit_volume.kind == "intensity image"
    assert explicit_channels.kind == "multi-channel image"
    assert explicit_rgb.axis_order == "Y,X,rgb"
    assert explicit_rgb.axes_explicit
    assert explicit_rgb.kind == "RGB image"


@pytest.mark.parametrize("ndim", range(11))
def test_inferred_axis_metadata_preserves_rank_and_unique_names(ndim):
    axes = infer_axis_metadata_from_shape((5,) * ndim)

    assert len(axes) == ndim
    assert len({axis.name for axis in axes}) == ndim
    assert all(axis.confidence == AXIS_CONFIDENCE_INFERRED for axis in axes)


def test_high_rank_shape_inferred_channel_state_constructs_without_axis_overflow():
    data = np.zeros((2, 5, 6, 7, 8, 9, 3), dtype=np.uint8)

    state = image_state_from_array(data)

    assert len(state.axes) == data.ndim
    assert state.axes[-1].name == "c"
    assert state.kind == "multi-channel image"


def test_explicit_ome_and_carried_axis_confidence_survive_roundtrip():
    data = np.zeros((2, 5, 7), dtype=np.float32)
    ome_metadata = {
        "multiscales": [
            {
                "axes": [
                    {"name": "z", "type": "space"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ]
            }
        ]
    }

    explicit = image_state_from_array(data, layer_metadata=ome_metadata)
    carried_explicit = image_state_from_array(
        data,
        layer_metadata={"vipp_image_state": explicit.to_dict()},
    )
    inferred = image_state_from_array(data)
    carried_inferred = image_state_from_array(
        data,
        layer_metadata={"vipp_image_state": inferred.to_dict()},
    )

    assert explicit.metadata_source == "OME-NGFF multiscales"
    assert explicit.axes_explicit
    assert carried_explicit.metadata_source == "VIPP carried state"
    assert carried_explicit.axes_explicit
    assert carried_inferred.metadata_source == "VIPP carried state"
    assert carried_inferred.axis_confidence == AXIS_CONFIDENCE_INFERRED
    assert not carried_inferred.axes_explicit


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("scale", 0.0, "scale must be a finite positive number"),
        ("scale", np.nan, "scale must be a finite positive number"),
        ("scale", "not-a-number", "scale must be a finite positive number"),
        ("translation", np.inf, "translation must be finite"),
        ("translation", "not-a-number", "translation must be finite"),
    ),
)
def test_axis_metadata_rejects_invalid_calibration(field, value, message):
    document = {"name": "x", "type": "space", field: value}

    with pytest.raises(ValueError, match=message):
        AxisMetadata.from_dict(document)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("scale", -1.0, "scale must be a finite positive number"),
        ("translation", np.nan, "translation must be finite"),
    ),
)
def test_direct_axis_metadata_rejects_invalid_calibration(field, value, message):
    kwargs = {field: value}

    with pytest.raises(ValueError, match=message):
        AxisMetadata("x", "space", **kwargs)


def test_invalid_carried_calibration_is_not_silently_reinferred():
    data = np.zeros((4, 5), dtype=np.uint16)
    carried = image_state_from_array(
        data,
        layer_metadata={"axes": "YX"},
    ).to_dict()
    carried["axes"][0]["scale"] = np.nan

    with pytest.raises(ValueError, match="invalid carried image state"):
        image_state_from_array(
            data,
            layer_metadata={"vipp_image_state": carried},
        )


def test_stale_carried_shape_is_not_silently_reused_or_reinferred():
    carried = image_state_from_array(
        np.zeros((4, 5), dtype=np.uint16),
        layer_metadata={"axes": "YX"},
    ).to_dict()

    with pytest.raises(ValueError, match="shape does not match the array"):
        image_state_from_array(
            np.zeros((5, 4), dtype=np.uint16),
            layer_metadata={"vipp_image_state": carried},
        )


@pytest.mark.parametrize(
    "layer_metadata",
    (
        {"axes": "ZYX"},
        {"axis_order": ["y"]},
        {"multiscales": []},
        {"multiscales": [{"axes": ["y"], "datasets": []}]},
    ),
)
def test_declared_axis_metadata_is_not_silently_replaced_when_malformed(
    layer_metadata,
):
    with pytest.raises(ValueError):
        image_state_from_array(
            np.zeros((4, 5), dtype=np.uint16),
            layer_metadata=layer_metadata,
        )


def test_explicit_axis_argument_rank_mismatch_is_rejected():
    with pytest.raises(ValueError, match="Axis metadata rank does not match"):
        image_state_from_array(
            np.zeros((4, 5), dtype=np.uint16),
            axes=(AxisMetadata("x", "space"),),
        )


def test_channel_colours_cannot_create_channels_absent_from_the_array():
    scalar_state = image_state_from_array(
        np.zeros((4, 5), dtype=np.uint16),
        layer_metadata={"axes": "YX"},
    )
    channel_state = image_state_from_array(
        np.zeros((2, 4, 5), dtype=np.uint16),
        layer_metadata={"axes": "CYX"},
    )

    with pytest.raises(ValueError, match="require a declared channel axis"):
        with_channel_colors(scalar_state, "red")
    with pytest.raises(ValueError, match="array's channel axis contains 2"):
        with_channel_colors(channel_state, "red,green,blue")


def test_partial_channel_colours_match_the_array_channel_count_exactly():
    state = image_state_from_array(
        np.zeros((3, 4, 5), dtype=np.uint16),
        layer_metadata={"axes": "CYX"},
    )

    colored = with_channel_colors(state, "yellow,cyan")

    assert colored is not None
    assert len(colored.channels) == 3
    assert colored.channels[0].color == 0xFFFF00
    assert colored.channels[1].color == 0x00FFFF
    assert colored.channels[2].color is None


@pytest.mark.parametrize(
    ("transform", "message"),
    (
        (
            {"type": "scale", "scale": [0.25]},
            "scale transform must contain exactly 2 values",
        ),
        (
            {"type": "scale", "scale": [0.25, 0.0]},
            "scale must be a finite positive number",
        ),
        (
            {"type": "scale", "scale": [0.25, np.nan]},
            "scale must be a finite positive number",
        ),
        (
            {"type": "translation", "translation": [1.0]},
            "translation transform must contain exactly 2 values",
        ),
        (
            {"type": "translation", "translation": [1.0, np.inf]},
            "translation must be finite",
        ),
    ),
)
def test_ngff_calibration_rejects_invalid_transforms(transform, message):
    metadata = {
        "multiscales": [
            {
                "axes": [
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": [
                    {"coordinateTransformations": [transform]},
                ],
            }
        ]
    }

    with pytest.raises(ValueError, match=message):
        image_state_from_array(
            np.zeros((4, 5), dtype=np.uint16),
            layer_metadata=metadata,
        )


def test_legacy_carried_state_recovers_axis_confidence_conservatively():
    explicit = image_state_from_array(
        np.zeros((3, 4), dtype=np.uint8),
        layer_metadata={"axes": "YX"},
    ).to_dict()
    inferred = image_state_from_array(
        np.zeros((3, 4), dtype=np.uint8),
    ).to_dict()
    for document in (explicit, inferred):
        document.pop("axis_confidence")
        for axis in document["axes"]:
            axis.pop("confidence")

    restored_explicit = ImageState.from_dict(explicit)
    restored_inferred = ImageState.from_dict(inferred)

    assert restored_explicit is not None and restored_explicit.axes_explicit
    assert restored_inferred is not None
    assert restored_inferred.axis_confidence == AXIS_CONFIDENCE_INFERRED


@pytest.mark.parametrize(
    "metadata_source",
    (
        "OME-NGFF multiscales",
        "OME-NGFF multiscales; napari layer transform",
        "OME-XML",
        "OME-Zarr 0.5 metadata",
        "napari layer axes metadata",
    ),
)
def test_legacy_authoritative_axis_sources_remain_explicit(metadata_source):
    document = image_state_from_array(
        np.zeros((3, 4), dtype=np.uint8),
        layer_metadata={"axes": "YX"},
    ).to_dict()
    document["metadata_source"] = metadata_source
    document.pop("axis_confidence")
    for axis in document["axes"]:
        axis.pop("confidence")

    restored = ImageState.from_dict(document)

    assert restored is not None
    assert restored.axis_confidence == AXIS_CONFIDENCE_EXPLICIT


@pytest.mark.parametrize(
    "metadata_source",
    (
        "VIPP transformed metadata",
        "VIPP carried state",
        "inferred after Invert",
        "unrecognized reader metadata",
    ),
)
def test_legacy_non_authoritative_axis_sources_remain_inferred(metadata_source):
    document = image_state_from_array(
        np.zeros((3, 4), dtype=np.uint8),
        layer_metadata={"axes": "YX"},
    ).to_dict()
    document["metadata_source"] = metadata_source
    document.pop("axis_confidence")
    for axis in document["axes"]:
        axis.pop("confidence")

    restored = ImageState.from_dict(document)

    assert restored is not None
    assert restored.axis_confidence == AXIS_CONFIDENCE_INFERRED


def test_transforms_preserve_inferred_axis_confidence():
    data = np.zeros((2, 5, 7), dtype=np.float32)
    inferred = image_state_from_array(data)

    transformed = transform_image_state(
        data.copy(),
        inferred,
        operation_id="invert",
        operation_title="Invert",
        params={},
    )

    assert transformed.axis_order == "ZYX"
    assert transformed.axis_confidence == AXIS_CONFIDENCE_INFERRED
    assert not transformed.axes_explicit


def test_transform_created_projection_axes_derive_input_confidence():
    data = np.zeros((2, 5, 7), dtype=np.float32)
    montage = np.zeros((7, 12), dtype=np.float32)
    inferred = image_state_from_array(data)
    explicit = image_state_from_array(
        data,
        layer_metadata={"axes": "ZYX"},
    )

    inferred_output = transform_image_state(
        montage,
        inferred,
        operation_id="orthogonal_projection",
        operation_title="Orthogonal Projection",
        params={},
    )
    explicit_output = transform_image_state(
        montage,
        explicit,
        operation_id="orthogonal_projection",
        operation_title="Orthogonal Projection",
        params={},
    )

    assert inferred_output.axis_order == "YX"
    assert inferred_output.axis_confidence == AXIS_CONFIDENCE_INFERRED
    assert explicit_output.axis_order == "YX"
    assert explicit_output.axis_confidence == AXIS_CONFIDENCE_EXPLICIT


def test_reorder_select_and_project_preserve_per_axis_confidence():
    data = np.zeros((2, 5, 7), dtype=np.float32)
    inferred = image_state_from_array(data)

    reordered = transform_image_state(
        np.transpose(data, (2, 1, 0)),
        inferred,
        operation_id="reorder_axes",
        operation_title="Reorder Axes",
        params={"order": "2,1,0"},
    )
    selected = transform_image_state(
        data[:, 2, :],
        inferred,
        operation_id="select_axis_slice",
        operation_title="Select Axis Slice",
        params={"axes": "1", "indices": "2"},
    )
    projected = transform_image_state(
        data.max(axis=0),
        inferred,
        operation_id="project_image",
        operation_title="Project Image",
        params={"axes": "axis:0", "method": "Maximum"},
    )

    assert reordered.axis_confidence == AXIS_CONFIDENCE_INFERRED
    assert selected.axis_order == "ZX"
    assert selected.axis_confidence == AXIS_CONFIDENCE_INFERRED
    assert projected.axis_order == "YX"
    assert projected.axis_confidence == AXIS_CONFIDENCE_INFERRED


def test_reorder_axes_moves_semantics_and_source_mapping_with_pixels():
    data = np.zeros((2, 3, 4), dtype=np.float32)
    state = image_state_from_array(data, layer_metadata={"axes": "ZYX"})

    reordered = transform_image_state(
        np.transpose(data, (2, 0, 1)),
        state,
        operation_id="reorder_axes",
        operation_title="Reorder Axes",
        params={"order": "XZY"},
    )

    assert reordered.shape == (4, 2, 3)
    assert reordered.axis_order == "XZY"
    assert [axis.source_axis for axis in reordered.axes] == [2, 0, 1]


def test_split_axis_metadata_normalizes_a_valid_negative_axis_once():
    input_state = image_state_from_array(
        np.zeros((3, 4), dtype=np.uint16),
        layer_metadata={"axes": "YX"},
    )

    output_state = transform_split_output_state(
        np.zeros((3,), dtype=np.uint16),
        input_state,
        operation_id="split_axis",
        operation_title="Split Axis",
        port_name="x_1",
        params={"axis": "axis:-1"},
    )

    assert output_state is not None
    assert output_state.axis_order == "Y"
    assert output_state.axes[0] == input_state.axes[0]


@pytest.mark.parametrize(
    ("params", "message"),
    (
        ({}, "must use axis:N"),
        ({"axis": 0}, "must use axis:N"),
        ({"axis": "axis: 0"}, "must use axis:N"),
        ({"axis": "axis:2"}, "out of range for 2D input"),
        ({"axis": "axis:-3"}, "out of range for 2D input"),
    ),
)
def test_split_axis_metadata_rejects_invalid_persisted_axis(params, message):
    input_state = image_state_from_array(
        np.zeros((3, 4), dtype=np.uint16),
        layer_metadata={"axes": "YX"},
    )

    with pytest.raises(ValueError, match=message):
        transform_split_output_state(
            np.zeros((4,), dtype=np.uint16),
            input_state,
            operation_id="split_axis",
            operation_title="Split Axis",
            port_name="axis_1",
            params=params,
        )


def test_split_axis_metadata_rejects_axis_selection_for_scalar_input():
    input_state = image_state_from_array(np.asarray(1, dtype=np.uint16))

    with pytest.raises(ValueError, match="cannot select an axis from 0D input"):
        transform_split_output_state(
            np.asarray(1, dtype=np.uint16),
            input_state,
            operation_id="split_axis",
            operation_title="Split Axis",
            port_name="axis_1",
            params={"axis": "axis:0"},
        )


def test_composite_creates_explicit_rgb_axis_without_promoting_spatial_guesses():
    data = np.zeros((3, 5, 7), dtype=np.float32)
    inferred = image_state_from_array(data)

    composite = transform_image_state(
        np.zeros((5, 7, 3), dtype=np.float32),
        inferred,
        operation_id="composite_to_rgb",
        operation_title="Composite to RGB",
        params={"channel_axis": 0},
    )
    restored = ImageState.from_dict(composite.to_dict())

    assert composite.axis_order == "Y,X,rgb"
    assert composite.axis_confidence == AXIS_CONFIDENCE_MIXED
    assert [axis.confidence for axis in composite.axes] == [
        AXIS_CONFIDENCE_INFERRED,
        AXIS_CONFIDENCE_INFERRED,
        AXIS_CONFIDENCE_EXPLICIT,
    ]
    assert restored is not None
    assert [axis.confidence for axis in restored.axes] == [
        AXIS_CONFIDENCE_INFERRED,
        AXIS_CONFIDENCE_INFERRED,
        AXIS_CONFIDENCE_EXPLICIT,
    ]


@pytest.mark.parametrize("channel_axis", [True, "0", 4, -5])
def test_combine_channels_metadata_rejects_repaired_insertion_axis(channel_axis):
    data = np.zeros((5, 7), dtype=np.float32)
    state = image_state_from_array(data, layer_metadata={"axes": "YX"})

    with pytest.raises(ValueError, match="insertion axis"):
        transform_multi_input_image_state(
            np.zeros((2, 5, 7), dtype=np.float32),
            [state, state],
            operation_id="combine_channels",
            operation_title="Combine Channels",
            params={"channel_axis": channel_axis},
        )


@pytest.mark.parametrize("operation_id", ["binary_threshold", "sobel_filter"])
def test_explicit_luma_reduction_removes_the_selected_numeric_axis(operation_id):
    data = np.zeros((3, 5, 7), dtype=np.float32)
    state = image_state_from_array(
        data,
        channels=tuple(ChannelMetadata(name=name) for name in ("R", "G", "B")),
    )

    reduced = transform_image_state(
        np.zeros((5, 7), dtype=bool),
        state,
        operation_id=operation_id,
        operation_title="Explicit luma operation",
        params={"channel_axis": 0},
    )

    assert reduced.axis_order == "YX"
    assert reduced.axis_confidence == AXIS_CONFIDENCE_INFERRED
    assert reduced.channels == ()


def test_scalar_threshold_retains_explicit_channel_axis_and_channel_metadata():
    data = np.zeros((5, 7, 3), dtype=np.float32)
    channels = tuple(ChannelMetadata(name=name) for name in ("R", "G", "B"))
    state = image_state_from_array(
        data,
        layer_metadata={"axes": "YXC"},
        channels=channels,
    )

    scalar = transform_image_state(
        np.zeros(data.shape, dtype=bool),
        state,
        operation_id="binary_threshold",
        operation_title="Binary Threshold",
        params={"threshold": 0.5, "channel_axis": -1},
    )

    assert scalar.axis_order == "YXC"
    assert scalar.axes_explicit
    assert scalar.channels == channels


def test_large_metadata_value_range_and_pattern_use_all_finite_values():
    data = np.zeros(600_123, dtype=np.float32)
    data[500_001] = -17.0
    data[511_111] = 7.0
    data[-1] = 1_000.0
    data[522_222] = np.nan

    state = image_state_from_array(data)

    assert state.value_range == "-17 to 1000"
    assert state.value_pattern == ""


def test_large_two_level_numeric_metadata_remains_binary_valued():
    data = np.zeros(600_123, dtype=np.uint16)
    data[500_001:] = 65_535

    state = image_state_from_array(data)

    assert state.value_range == "0 to 65535"
    assert state.value_pattern == "binary-valued"


def test_metadata_value_range_preserves_wide_integer_extrema_exactly():
    base = 2**60
    data = np.array([base, base + 1, base + 3], dtype=np.int64)

    state = image_state_from_array(data)

    assert state.value_range == f"{base} to {base + 3}"


def test_metadata_statistics_can_be_deferred_until_background_execution():
    data = np.arange(16, dtype=np.float32)

    state = image_state_from_array(data, defer_statistics=True)

    assert state.value_range == DEFERRED_VALUE_RANGE
    assert state.value_pattern == ""


@pytest.mark.parametrize(
    ("data", "expected_mode"),
    [
        (
            np.zeros((3, 4), dtype=np.uint16),
            "native integer levels (uint16; Float histogram bins ignored)",
        ),
        (
            np.zeros((3, 4), dtype=np.float32),
            "4096 equal-width float bins (float32)",
        ),
    ],
)
def test_histogram_threshold_history_records_scope_and_dtype_mode(
    data,
    expected_mode,
):
    input_state = image_state_from_array(data, layer_metadata={"axes": "YX"})

    output_state = transform_image_state(
        np.zeros(data.shape, dtype=bool),
        input_state,
        operation_id="otsu_threshold",
        operation_title="Otsu Threshold",
        params={
            "threshold_scope": "Slice histogram",
            "histogram_bins": 4_096,
        },
    )

    history = output_state.history[-1]
    assert history.startswith("Otsu Threshold: Slice histogram; ")
    assert expected_mode in history
    assert history.endswith("non-finite values excluded and set to background")


def test_boolean_threshold_history_records_passthrough_instead_of_algorithm():
    data = np.zeros((3, 4), dtype=bool)
    input_state = image_state_from_array(data, layer_metadata={"axes": "YX"})

    output_state = transform_image_state(
        data.copy(),
        input_state,
        operation_id="triangle_threshold",
        operation_title="Triangle Threshold",
        params={"threshold_scope": "Stack histogram", "histogram_bins": 256},
    )

    assert output_state.history[-1] == (
        "Triangle Threshold: existing boolean segmentation preserved; "
        "automatic threshold bypassed"
    )


def test_li_threshold_history_records_raw_finite_value_policy():
    data = np.zeros((3, 4), dtype=np.float32)
    input_state = image_state_from_array(data, layer_metadata={"axes": "YX"})

    output_state = transform_image_state(
        np.zeros(data.shape, dtype=bool),
        input_state,
        operation_id="li_threshold",
        operation_title="Li Threshold",
        params={"threshold_scope": "Stack histogram"},
    )

    assert output_state.history[-1] == (
        "Li Threshold: Stack histogram; raw finite-value iteration; "
        "non-finite values excluded and set to background"
    )


def test_threshold_history_never_infers_rgb_conversion_from_x_size():
    data = np.zeros((5, 7, 3), dtype=np.float32)
    scalar_state = image_state_from_array(data, layer_metadata={"axes": "ZYX"})
    color_state = image_state_from_array(data, layer_metadata={"axes": "YXC"})

    scalar = transform_image_state(
        np.zeros(data.shape, dtype=bool),
        scalar_state,
        operation_id="otsu_threshold",
        operation_title="Otsu Threshold",
        params={
            "threshold_scope": "Stack histogram",
            "histogram_bins": 256,
            "channel_axis": -1,
        },
    )
    color = transform_image_state(
        np.zeros(data.shape[:-1], dtype=bool),
        color_state,
        operation_id="otsu_threshold",
        operation_title="Otsu Threshold",
        params={
            "threshold_scope": "Stack histogram",
            "histogram_bins": 256,
            "channel_axis": 2,
        },
    )

    assert "RGB" not in scalar.history[-1]
    assert "BT.601 RGB/RGBA luma from channel axis 2" in color.history[-1]


@pytest.mark.parametrize(
    ("operation_id", "title", "params", "expected"),
    [
        (
            "rescale_intensity",
            "Rescale Intensity",
            {
                "cutoff_mode": "Percentiles",
                "in_low_percentile": 1.0,
                "in_high_percentile": 99.0,
                "in_low_value": -5.0,
                "in_high_value": 5.0,
                "out_min": 0.0,
                "out_max": 255.0,
            },
            "Rescale Intensity: finite-value percentiles 1..99; output 0..255",
        ),
        (
            "rescale_intensity",
            "Rescale Intensity",
            {
                "cutoff_mode": "Values",
                "in_low_percentile": 0.0,
                "in_high_percentile": 100.0,
                "in_low_value": -5.0,
                "in_high_value": 5.0,
                "out_min": -1.0,
                "out_max": 1.0,
            },
            "Rescale Intensity: explicit input values -5..5; output -1..1",
        ),
        (
            "clip_intensity",
            "Clip",
            {"cutoff_mode": "Data range", "minimum": 2.0, "maximum": 4.0},
            "Clip: full data range preserved (no clipping)",
        ),
        (
            "clip_intensity",
            "Clip",
            {"cutoff_mode": "Values", "minimum": 2.0, "maximum": 4.0},
            "Clip: explicit values 2..4",
        ),
    ],
)
def test_intensity_cutoff_history_records_only_active_mode(
    operation_id,
    title,
    params,
    expected,
):
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    input_state = image_state_from_array(data, layer_metadata={"axes": "YX"})

    output_state = transform_image_state(
        data.copy(),
        input_state,
        operation_id=operation_id,
        operation_title=title,
        params=params,
    )

    assert output_state.history[-1] == expected
