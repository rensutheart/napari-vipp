from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from scipy import ndimage as ndi

from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.operations import (
    average_blur,
    crop_stack,
    difference_of_gaussians_filter,
    gaussian_blur,
    gaussian_blur_3d,
    median_filter,
    rolling_ball_background,
    sigma_filter,
    subtract_background,
)
from napari_vipp.core.pipeline import (
    NODE_LIBRARY_BY_ID,
    PrototypePipeline,
    operation_call_parameter_value,
)

Operation = Callable[..., np.ndarray]


XY_FILTERS: tuple[tuple[str, Operation, dict[str, object]], ...] = (
    ("average_blur", average_blur, {"size": 3}),
    ("gaussian_blur", gaussian_blur, {"sigma": 0.8}),
    (
        "difference_of_gaussians",
        difference_of_gaussians_filter,
        {"low_sigma": 0.6, "high_sigma": 1.2},
    ),
    ("median_filter", median_filter, {"size": 3}),
    ("sigma_filter", sigma_filter, {"radius": 1.5}),
)

ROLLING_BALL_OPERATIONS: tuple[tuple[str, Operation, dict[str, object]], ...] = (
    (
        "rolling_ball_background",
        rolling_ball_background,
        {"radius": 2, "disable_smoothing": True},
    ),
    (
        "subtract_background",
        subtract_background,
        {"radius": 2, "disable_smoothing": True},
    ),
)

ALL_AXIS_AWARE_OPERATIONS: tuple[tuple[str, Operation, dict[str, object]], ...] = (
    (
        "crop_stack",
        crop_stack,
        {"top": 1, "bottom": 1, "left": 1, "right": 1},
    ),
    *XY_FILTERS,
    (
        "gaussian_blur_3d",
        gaussian_blur_3d,
        {"sigma_z": 0.6, "sigma_y": 0.7, "sigma_x": 0.8},
    ),
    *ROLLING_BALL_OPERATIONS,
)


def _scalar_zyx_data(x_size: int) -> np.ndarray:
    rng = np.random.default_rng(842 + x_size)
    values = rng.uniform(2.0, 18.0, size=(2, 9, x_size)).astype(np.float32)
    values[:, 4, x_size // 2] += 40.0
    return values


@pytest.mark.parametrize("x_size", [3, 4])
@pytest.mark.parametrize(
    ("_operation_id", "operation", "kwargs"),
    XY_FILTERS,
    ids=[case[0] for case in XY_FILTERS],
)
def test_xy_filters_treat_rgb_sized_trailing_dimension_as_scalar_x(
    x_size: int,
    _operation_id: str,
    operation: Operation,
    kwargs: dict[str, object],
):
    data = _scalar_zyx_data(x_size)

    result = operation(data, **kwargs)
    expected = np.stack([operation(plane, **kwargs) for plane in data], axis=0)

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("x_size", [3, 4])
def test_crop_treats_rgb_sized_trailing_dimension_as_scalar_x(x_size: int):
    data = _scalar_zyx_data(x_size)

    result = crop_stack(
        data,
        top=1,
        bottom=1,
        left=1,
        right=1,
    )

    np.testing.assert_array_equal(result, data[:, 1:-1, 1:-1])


def test_crop_uses_declared_noncanonical_yx_axes():
    data = np.arange(3 * 7 * 5 * 9).reshape(3, 7, 5, 9)

    result = crop_stack(
        data,
        top=1,
        bottom=2,
        left=2,
        right=1,
        channel_axis=0,
        axis_names=("c", "y", "z", "x"),
    )

    np.testing.assert_array_equal(result, data[:, 1:-2, :, 2:-1])


def test_pipeline_crop_does_not_promote_shape_inferred_yxc_names():
    data = np.arange(7 * 9 * 3, dtype=np.uint16).reshape(7, 9, 3)
    state = image_state_from_array(data)
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    crop = pipeline.add_node("crop_stack")
    assert pipeline.connect("input", crop.id).success
    for name, value in {
        "top": 1,
        "bottom": 2,
        "left": 1,
        "right": 1,
    }.items():
        pipeline.set_param(crop.id, name, value)

    call = pipeline.prepare_node_call(crop.id, (data,), (state,))
    outputs = pipeline.run(data)

    assert call.kwargs["axis_names"] == ("axis:0", "y", "x")
    assert call.kwargs["channel_axis"] is None
    np.testing.assert_array_equal(outputs[crop.id], data[:, 1:-2, 1:-1])
    assert tuple(
        axis.translation for axis in pipeline.output_states[crop.id].axes
    ) == (0.0, 1.0, 1.0)


def test_pipeline_crop_preserves_explicit_yxc_semantics():
    data = np.arange(7 * 9 * 3, dtype=np.uint16).reshape(7, 9, 3)
    state = image_state_from_array(data, layer_metadata={"axes": "YXC"})
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    crop = pipeline.add_node("crop_stack")
    assert pipeline.connect("input", crop.id).success
    for name, value in {
        "top": 1,
        "bottom": 2,
        "left": 1,
        "right": 1,
    }.items():
        pipeline.set_param(crop.id, name, value)

    call = pipeline.prepare_node_call(crop.id, (data,), (state,))
    outputs = pipeline.run(data, input_metadata={"axes": "YXC"})

    assert call.kwargs["axis_names"] == ("y", "x", "c")
    assert call.kwargs["channel_axis"] == 2
    np.testing.assert_array_equal(outputs[crop.id], data[1:-2, 1:-1, :])
    assert tuple(
        axis.translation for axis in pipeline.output_states[crop.id].axes
    ) == (1.0, 1.0, 0.0)


def test_pipeline_crop_protects_explicit_non_xy_axes_with_inferred_yx():
    data = np.arange(2 * 7 * 3 * 9 * 5, dtype=np.uint16).reshape(2, 7, 3, 9, 5)
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("t", "time"),
            AxisMetadata("y", "space", confidence="shape-inferred"),
            AxisMetadata("c", "channel"),
            AxisMetadata("x", "space", confidence="shape-inferred"),
            AxisMetadata("z", "space"),
        ),
    )
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    crop = pipeline.add_node("crop_stack")
    assert pipeline.connect("input", crop.id).success
    pipeline.set_param(crop.id, "top", 1)
    pipeline.set_param(crop.id, "left", 2)
    pipeline.set_param(crop.id, "z_start", 1)

    call = pipeline.prepare_node_call(crop.id, (data,), (state,))
    outputs = pipeline.run(
        data,
        input_metadata={"vipp_image_state": state.to_dict()},
    )

    assert call.kwargs["axis_names"] == ("t", "y", "c", "x", "z")
    assert call.kwargs["channel_axis"] == 2
    assert call.kwargs["z_axis_explicit"] is True
    np.testing.assert_array_equal(outputs[crop.id], data[:, 1:, :, 2:, 1:])
    assert tuple(
        axis.translation for axis in pipeline.output_states[crop.id].axes
    ) == (0.0, 1.0, 0.0, 2.0, 1.0)


@pytest.mark.parametrize(
    ("shape", "axes"),
    (
        (
            (4, 5, 6),
            (
                AxisMetadata("y", "space"),
                AxisMetadata("y", "space"),
                AxisMetadata("x", "space"),
            ),
        ),
        (
            (4, 3),
            (
                AxisMetadata("t", "time"),
                AxisMetadata("c", "channel"),
            ),
        ),
    ),
)
def test_pipeline_crop_fails_closed_without_one_safe_yx_pair(shape, axes):
    data = np.zeros(shape, dtype=np.uint16)
    state = image_state_from_array(data, axes=axes)
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    crop = pipeline.add_node("crop_stack")
    assert pipeline.connect("input", crop.id).success
    pipeline.set_param(crop.id, "top", 1)

    with pytest.raises(ValueError, match="cannot resolve two safe Y/X axes"):
        pipeline.run(
            data,
            input_metadata={"vipp_image_state": state.to_dict()},
        )


@pytest.mark.parametrize(
    ("shape", "axis_names", "channel_axis", "expected_slices"),
    (
        pytest.param(
            (6, 7, 8),
            ("z", "y", "x"),
            None,
            (slice(1, -2), slice(1, -1), slice(2, -1)),
            id="zyx",
        ),
        pytest.param(
            (2, 3, 6, 7, 8),
            ("t", "c", "z", "y", "x"),
            1,
            (slice(None), slice(None), slice(1, -2), slice(1, -1), slice(2, -1)),
            id="tczyx",
        ),
        pytest.param(
            (2, 7, 3, 8, 6),
            ("t", "y", "c", "x", "z"),
            2,
            (slice(None), slice(1, -1), slice(None), slice(2, -1), slice(1, -2)),
            id="noncanonical",
        ),
    ),
)
def test_crop_crops_exact_named_z_without_reordering_other_axes(
    shape,
    axis_names,
    channel_axis,
    expected_slices,
):
    data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)

    result = crop_stack(
        data,
        z_start=1,
        z_end=2,
        top=1,
        bottom=1,
        left=2,
        right=1,
        channel_axis=channel_axis,
        axis_names=axis_names,
    )

    np.testing.assert_array_equal(result, data[expected_slices])
    assert result.dtype == data.dtype
    assert result.flags.c_contiguous


@pytest.mark.parametrize("axis_names", [("z", "y"), ("z", "a", "x")])
def test_crop_rejects_malformed_declared_yx_axes(axis_names):
    with pytest.raises(ValueError, match="axis names|exactly one"):
        crop_stack(np.zeros((3, 5, 7)), axis_names=axis_names)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"top": -1}, "non-negative"),
        ({"left": 2.5}, "integers"),
        ({"left": 2, "right": 2}, "remove every sample"),
    ),
)
def test_crop_rejects_margins_that_would_be_silently_repaired(kwargs, message):
    with pytest.raises(ValueError, match=message):
        crop_stack(np.zeros((4, 4), dtype=np.uint8), **kwargs)


@pytest.mark.parametrize(
    ("shape", "axis_names", "channel_axis", "kwargs", "message"),
    (
        (
            (2, 3, 5, 7),
            ("z", "z", "y", "x"),
            None,
            {"z_start": 1},
            "exactly one explicitly declared Z axis",
        ),
        (
            (4, 5, 7),
            ("z", "y", "x"),
            0,
            {"z_start": 1},
            "cannot also be the Z spatial axis",
        ),
        (
            (4, 5, 7),
            ("z", "y", "x"),
            None,
            {"z_start": 2, "z_end": 2},
            "remove every sample",
        ),
    ),
)
def test_crop_rejects_ambiguous_or_empty_z_crops(
    shape,
    axis_names,
    channel_axis,
    kwargs,
    message,
):
    with pytest.raises(ValueError, match=message):
        crop_stack(
            np.zeros(shape, dtype=np.uint8),
            axis_names=axis_names,
            channel_axis=channel_axis,
            **kwargs,
        )


@pytest.mark.parametrize("name", ("z_start", "z_end"))
def test_crop_z_margin_set_param_rejects_negative_values(name):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    crop = pipeline.add_node("crop_stack")

    with pytest.raises(ValueError, match="non-negative"):
        pipeline.set_param(crop.id, name, -1)

    assert pipeline.nodes[crop.id].params[name] == 0


@pytest.mark.parametrize("x_size", [3, 4])
def test_gaussian_3d_filters_rgb_sized_trailing_scalar_x(x_size: int):
    data = np.zeros((2, 9, x_size), dtype=np.float32)
    data[0, 4, x_size // 2] = 1.0

    result = gaussian_blur_3d(
        data,
        sigma_z=0.0,
        sigma_y=0.0,
        sigma_x=0.8,
    )
    expected = ndi.gaussian_filter(data, sigma=(0.0, 0.0, 0.8))

    np.testing.assert_allclose(result, expected)
    assert np.count_nonzero(result[0, 4]) > 1


@pytest.mark.parametrize("x_size", [3, 4])
@pytest.mark.parametrize(
    ("_operation_id", "operation", "kwargs"),
    ROLLING_BALL_OPERATIONS,
    ids=[case[0] for case in ROLLING_BALL_OPERATIONS],
)
def test_rolling_ball_treats_rgb_sized_trailing_dimension_as_scalar_x(
    x_size: int,
    _operation_id: str,
    operation: Operation,
    kwargs: dict[str, object],
):
    data = _scalar_zyx_data(x_size)

    result = operation(data, **kwargs)
    expected = np.stack([operation(plane, **kwargs) for plane in data], axis=0)

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("_operation_id", "operation", "kwargs"),
    ALL_AXIS_AWARE_OPERATIONS,
    ids=[case[0] for case in ALL_AXIS_AWARE_OPERATIONS],
)
def test_filters_support_an_explicit_nontrailing_channel_axis(
    _operation_id: str,
    operation: Operation,
    kwargs: dict[str, object],
):
    scalar_channels = [
        _scalar_zyx_data(4) + np.float32(channel * 100.0) for channel in range(3)
    ]
    data = np.stack(scalar_channels, axis=2)

    result = operation(data, channel_axis=2, **kwargs)
    expected = np.stack(
        [operation(channel, **kwargs) for channel in scalar_channels],
        axis=2,
    )

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "operation_id",
    [case[0] for case in ALL_AXIS_AWARE_OPERATIONS],
)
def test_axis_aware_filter_nodes_expose_scalar_default_contract(operation_id: str):
    spec = NODE_LIBRARY_BY_ID[operation_id]
    parameter = next(
        parameter for parameter in spec.parameters if parameter.name == "channel_axis"
    )

    assert parameter.label == "Channel axis (-1 = scalar)"
    assert parameter.default == -1
    assert operation_call_parameter_value(operation_id, "channel_axis", -1) is None


@pytest.mark.parametrize("x_size", [3, 4])
@pytest.mark.parametrize(
    ("operation_id", "operation", "kwargs"),
    ALL_AXIS_AWARE_OPERATIONS,
    ids=[case[0] for case in ALL_AXIS_AWARE_OPERATIONS],
)
def test_pipeline_scalar_contract_matches_direct_zyx_operation(
    x_size: int,
    operation_id: str,
    operation: Operation,
    kwargs: dict[str, object],
):
    data = _scalar_zyx_data(x_size)
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    node = pipeline.add_node(operation_id)
    pipeline.connect("input", node.id)
    for name, value in kwargs.items():
        pipeline.set_param(node.id, name, value)

    outputs = pipeline.run(data, input_metadata={"axes": "ZYX"})
    expected = operation(data, **kwargs)
    state = pipeline.output_states[node.id]

    np.testing.assert_allclose(outputs[node.id], expected)
    assert state is not None
    assert state.shape == expected.shape
    assert tuple(axis.name for axis in state.axes) == ("z", "y", "x")
    if operation_id == "crop_stack":
        assert tuple(axis.translation for axis in state.axes) == (0.0, 1.0, 1.0)


@pytest.mark.parametrize(
    ("operation_id", "operation", "kwargs"),
    ALL_AXIS_AWARE_OPERATIONS,
    ids=[case[0] for case in ALL_AXIS_AWARE_OPERATIONS],
)
def test_pipeline_nontrailing_channel_contract_matches_direct_operation(
    operation_id: str,
    operation: Operation,
    kwargs: dict[str, object],
):
    data = np.stack(
        [_scalar_zyx_data(4) + np.float32(channel * 100.0) for channel in range(3)],
        axis=2,
    )
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    node = pipeline.add_node(operation_id)
    pipeline.connect("input", node.id)
    for name, value in kwargs.items():
        pipeline.set_param(node.id, name, value)
    pipeline.set_param(node.id, "channel_axis", 2)

    outputs = pipeline.run(data, input_metadata={"axes": "ZYCX"})
    expected = operation(data, channel_axis=2, **kwargs)
    state = pipeline.output_states[node.id]

    np.testing.assert_allclose(outputs[node.id], expected)
    assert state is not None
    assert state.shape == expected.shape
    assert tuple(axis.name for axis in state.axes) == ("z", "y", "c", "x")
    if operation_id == "crop_stack":
        assert tuple(axis.translation for axis in state.axes) == (
            0.0,
            1.0,
            0.0,
            1.0,
        )
