from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from scipy import ndimage as ndi

from napari_vipp.core.operations import (
    average_blur,
    crop_stack,
    difference_of_gaussians_filter,
    gaussian_blur,
    gaussian_blur_3d,
    median_filter,
    rolling_ball_background,
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
