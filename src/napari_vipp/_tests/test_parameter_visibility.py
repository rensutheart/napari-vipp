from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from napari_vipp.core.metadata import AxisMetadata, ImageState, image_state_from_array
from napari_vipp.core.pipeline import (
    HISTOGRAM_BINS_PARAMETER,
    NODE_LIBRARY,
    PARAMETER_VISIBILITY_ALWAYS,
    PARAMETER_VISIBILITY_FLOATING_INPUT,
    PARAMETER_VISIBILITY_MULTICHANNEL_INPUT,
    PARAMETER_VISIBILITY_RGB_OR_RGBA_INPUT,
    PARAMETER_VISIBILITY_SPATIAL_MODE_RELEVANT,
    PARAMETER_VISIBILITY_STACK_SCOPE_RELEVANT,
    SCALAR_CHANNEL_AXIS_PARAMETER,
    SCALAR_LUMA_CHANNEL_AXIS_PARAMETER,
    SPATIAL_MODE_PARAMETER,
    THRESHOLD_SCOPE_PARAMETER,
    ParameterSpec,
    ParameterVisibilityContext,
    PrototypePipeline,
    resolve_parameter_visibility,
    validate_parameter_visibility_spec,
)


def _state(shape, dtype=np.float32, axes: str | None = None) -> ImageState:
    metadata = {"axes": axes} if axes is not None else None
    return image_state_from_array(
        np.zeros(shape, dtype=dtype),
        layer_metadata=metadata,
    )


def _context(
    *states: ImageState | None,
    params: dict | None = None,
    ports: tuple[str, ...] | None = None,
    connected: frozenset[str] | None = None,
) -> ParameterVisibilityContext:
    names = ports or tuple(f"input_{index + 1}" for index in range(len(states)))
    connected_ports = connected
    if connected_ports is None:
        connected_ports = frozenset(
            name for name, state in zip(names, states, strict=True) if state is not None
        )
    return ParameterVisibilityContext(
        operation_id="test",
        parameter_values=params or {},
        input_state_by_port=dict(zip(names, states, strict=True)),
        declared_input_ports=names,
        connected_input_ports=connected_ports,
    )


@pytest.mark.parametrize(
    ("dtype", "floating", "negative_possible"),
    (
        (np.float32, True, True),
        (np.float64, True, True),
        (np.uint8, False, False),
        (np.uint16, False, False),
        (np.int16, False, True),
        (np.int32, False, True),
        (np.bool_, False, False),
    ),
)
def test_dtype_visibility_rules(dtype, floating, negative_possible):
    state = _state((4, 5), dtype=dtype, axes="YX")
    context = _context(state)
    negative_guard = ParameterSpec(
        "clip_negative",
        "Clip negative",
        "bool",
        True,
        0,
        1,
        1,
        visibility="negative_values_possible",
    )

    assert (
        resolve_parameter_visibility(HISTOGRAM_BINS_PARAMETER, context=context).visible
        is floating
    )
    assert (
        resolve_parameter_visibility(negative_guard, context=context).visible
        is negative_possible
    )


@pytest.mark.parametrize(
    ("shape", "axes", "ordinary", "encoded"),
    (
        ((8, 9), "YX", False, False),
        ((3, 8, 9), "CYX", True, False),
        ((4, 3, 8, 9), "ZCYX", True, False),
        ((8, 9, 3), "Y,X,rgb", True, True),
        ((8, 9, 4), "Y,X,rgba", True, True),
    ),
)
def test_channel_rules_use_explicit_semantics(shape, axes, ordinary, encoded):
    context = _context(_state(shape, axes=axes))

    assert (
        resolve_parameter_visibility(
            SCALAR_CHANNEL_AXIS_PARAMETER,
            context=context,
        ).visible
        is ordinary
    )
    assert (
        resolve_parameter_visibility(
            SCALAR_LUMA_CHANNEL_AXIS_PARAMETER,
            context=context,
        ).visible
        is encoded
    )


@pytest.mark.parametrize("last_size", (3, 4))
def test_shape_only_color_like_arrays_remain_unresolved(last_size):
    context = _context(_state((8, 9, last_size)))

    assert resolve_parameter_visibility(
        SCALAR_CHANNEL_AXIS_PARAMETER,
        context=context,
    ).visible
    assert resolve_parameter_visibility(
        SCALAR_LUMA_CHANNEL_AXIS_PARAMETER,
        context=context,
    ).visible


def test_explicit_channel_names_remain_safe_with_incomplete_axis_types():
    cyx = replace(
        _state((3, 8, 9), axes="CYX"),
        axes=(
            AxisMetadata("C", "unknown"),
            AxisMetadata("Y", "space"),
            AxisMetadata("X", "space"),
        ),
    )
    rgb = replace(
        _state((8, 9, 3), axes="Y,X,rgb"),
        axes=(
            AxisMetadata("Y", "space"),
            AxisMetadata("X", "space"),
            AxisMetadata("rgb", "unknown"),
        ),
    )

    assert resolve_parameter_visibility(
        SCALAR_CHANNEL_AXIS_PARAMETER,
        context=_context(cyx),
    ).visible
    assert resolve_parameter_visibility(
        SCALAR_LUMA_CHANNEL_AXIS_PARAMETER,
        context=_context(rgb),
    ).visible


@pytest.mark.parametrize(
    ("axes", "mode", "visible"),
    (
        ("YX", "Auto from axes", False),
        ("YX", "2D YX", False),
        ("YX", "3D ZYX", True),
        ("ZYX", "Auto from axes", True),
        ("CYX", "Auto from axes", True),
        ("ZCYX", "Auto from axes", True),
        ("TYX", "Auto from axes", True),
    ),
)
def test_spatial_mode_visibility_is_semantic_and_preserves_invalid_mode(
    axes,
    mode,
    visible,
):
    shape = tuple(3 for _ in axes)
    context = _context(
        _state(shape, axes=axes),
        params={"spatial_mode": mode},
    )

    assert (
        resolve_parameter_visibility(
            SPATIAL_MODE_PARAMETER,
            context=context,
        ).visible
        is visible
    )


def test_spatial_mode_visibility_considers_every_declared_input_port():
    ports = ("image", "psf")
    both_yx = _context(
        _state((8, 9), axes="YX"),
        _state((5, 5), axes="YX"),
        params={"spatial_mode": "Auto from axes"},
        ports=ports,
    )
    mixed = _context(
        _state((3, 8, 9), axes="ZYX"),
        _state((5, 5), axes="YX"),
        params={"spatial_mode": "Auto from axes"},
        ports=ports,
    )
    missing_psf = _context(
        _state((8, 9), axes="YX"),
        None,
        params={"spatial_mode": "Auto from axes"},
        ports=ports,
    )

    assert not resolve_parameter_visibility(
        SPATIAL_MODE_PARAMETER,
        context=both_yx,
    ).visible
    assert resolve_parameter_visibility(
        SPATIAL_MODE_PARAMETER,
        context=mixed,
    ).visible
    assert resolve_parameter_visibility(
        SPATIAL_MODE_PARAMETER,
        context=missing_psf,
    ).visible


@pytest.mark.parametrize(
    ("shape", "axes", "visible"),
    (
        ((8, 9), "YX", False),
        ((3, 8, 9), "CYX", False),
        ((3, 8, 9), "ZYX", True),
        ((4, 8, 9), "TYX", True),
        ((8, 9, 3), "Y,X,rgb", False),
    ),
)
def test_stack_scope_rule_excludes_only_explicit_plane_and_channel_axes(
    shape,
    axes,
    visible,
):
    context = _context(_state(shape, axes=axes))
    assert (
        resolve_parameter_visibility(
            THRESHOLD_SCOPE_PARAMETER,
            context=context,
        ).visible
        is visible
    )


def test_stack_scope_shape_only_2d_remains_visible():
    assert resolve_parameter_visibility(
        THRESHOLD_SCOPE_PARAMETER,
        context=_context(_state((8, 9))),
    ).visible


def test_parameter_dependency_rules_are_conservative_and_composable():
    spec = ParameterSpec(
        "manual_value",
        "Manual value",
        "float",
        1.0,
        0.0,
        2.0,
        0.1,
        visibility="parameter_in",
        visibility_parameter="mode",
        visibility_values=("Manual",),
    )

    assert resolve_parameter_visibility(spec, context=_context()).visible
    assert resolve_parameter_visibility(
        spec,
        context=_context(params={"mode": "Manual"}),
    ).visible
    assert not resolve_parameter_visibility(
        spec,
        context=_context(params={"mode": "Auto"}),
    ).visible


def test_unknown_or_incomplete_visibility_rules_fail_clearly():
    unknown = replace(HISTOGRAM_BINS_PARAMETER, visibility="not_a_rule")
    incomplete = replace(
        HISTOGRAM_BINS_PARAMETER,
        visibility="parameter_in",
    )

    with pytest.raises(ValueError, match="unknown visibility rule"):
        resolve_parameter_visibility(unknown)
    with pytest.raises(ValueError, match="requires"):
        resolve_parameter_visibility(incomplete)


def test_catalog_visibility_rules_and_shared_families_are_complete():
    for operation in NODE_LIBRARY:
        parameter_names = {parameter.name for parameter in operation.parameters}
        declared_ports = {port.name for port in operation.input_ports}
        for parameter in operation.parameters:
            validate_parameter_visibility_spec(parameter)
            if parameter.visibility_parameter:
                assert parameter.visibility_parameter in parameter_names
            if parameter.visibility_ports:
                assert set(parameter.visibility_ports) <= declared_ports

    ordinary = {
        operation.id
        for operation in NODE_LIBRARY
        for parameter in operation.parameters
        if parameter.name == "channel_axis"
        and parameter.visibility == PARAMETER_VISIBILITY_MULTICHANNEL_INPUT
    }
    assert ordinary == {
        "crop_stack",
        "average_blur",
        "gaussian_blur",
        "gaussian_blur_3d",
        "median_filter",
        "bilateral_filter",
        "non_local_means_filter",
        "rolling_ball_background",
        "subtract_background",
        "difference_of_gaussians",
        "unsharp_mask",
    }
    encoded = {
        operation.id
        for operation in NODE_LIBRARY
        for parameter in operation.parameters
        if parameter.name == "channel_axis"
        and parameter.visibility == PARAMETER_VISIBILITY_RGB_OR_RGBA_INPUT
    }
    assert encoded == {
        "sobel_filter",
        "canny_edges",
        "laplace_filter",
        "otsu_threshold",
        "triangle_threshold",
        "li_threshold",
        "yen_threshold",
        "isodata_threshold",
        "minimum_threshold",
        "binary_threshold",
        "hysteresis_threshold",
        "adaptive_mean_threshold",
        "adaptive_gaussian_threshold",
        "sauvola_threshold",
        "niblack_threshold",
    }
    assert all(
        parameter.visibility == PARAMETER_VISIBILITY_FLOATING_INPUT
        for operation in NODE_LIBRARY
        for parameter in operation.parameters
        if parameter.name == "histogram_bins"
    )
    assert all(
        parameter.visibility == PARAMETER_VISIBILITY_STACK_SCOPE_RELEVANT
        for operation in NODE_LIBRARY
        for parameter in operation.parameters
        if parameter.name == "threshold_scope"
    )
    spatial_exceptions = {
        operation.id
        for operation in NODE_LIBRARY
        for parameter in operation.parameters
        if parameter.name == "spatial_mode"
        and parameter.visibility == PARAMETER_VISIBILITY_ALWAYS
    }
    assert spatial_exceptions == {"born_wolf_psf", "measure_3d_mesh_morphology"}
    assert all(
        parameter.visibility == PARAMETER_VISIBILITY_SPATIAL_MODE_RELEVANT
        for operation in NODE_LIBRARY
        if operation.id not in spatial_exceptions
        for parameter in operation.parameters
        if parameter.name == "spatial_mode"
    )


def test_all_costes_threshold_parameters_have_manual_mode_dependency():
    threshold_parameters = [
        (operation.id, parameter)
        for operation in NODE_LIBRARY
        for parameter in operation.parameters
        if parameter.name in {"channel_1_threshold", "channel_2_threshold"}
    ]
    assert len(threshold_parameters) == 14
    assert all(
        parameter.visibility == "parameter_in"
        and parameter.visibility_parameter == "threshold_mode"
        and parameter.visibility_values == ("Manual",)
        for _, parameter in threshold_parameters
    )


def test_pipeline_context_uses_dynamic_input_and_output_ports():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    second_source = pipeline.add_node("input")
    add = pipeline.add_node("add_images")
    assert pipeline.connect("input", add.id, 0).success
    assert pipeline.connect(second_source.id, add.id, 1).success

    input_context = pipeline.parameter_visibility_context(add.id)
    assert input_context.declared_input_ports == ("input_1", "input_2")
    assert input_context.connected_input_ports == frozenset(
        {"input_1", "input_2"}
    )

    split = pipeline.add_node("split_channels")
    downstream = pipeline.add_node("gaussian_blur")
    assert pipeline.connect("input", split.id).success
    assert pipeline.connect(split.id, downstream.id, source_port=1).success

    output_context = pipeline.parameter_visibility_context(split.id)
    assert output_context.used_output_ports == frozenset({1})
