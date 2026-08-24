from __future__ import annotations

import numpy as np
import pytest

import napari_vipp.core.operations as operations
from napari_vipp.core.metadata import (
    AXIS_CONFIDENCE_EXPLICIT,
    AXIS_CONFIDENCE_INFERRED,
    AXIS_CONFIDENCE_MIXED,
    DEFERRED_VALUE_RANGE,
    AcquisitionMetadata,
    AmbiguousAxisError,
    AxisDeclaration,
    AxisMetadata,
    ChannelMetadata,
    apply_axis_declaration,
    image_state_from_array,
)
from napari_vipp.core.operations import (
    COMPOSITE_RGB_AUTO,
    COMPOSITE_RGB_MANUAL,
    COMPOSITE_RGB_PERCENTILE_1_99,
)
from napari_vipp.core.pipeline import (
    HISTOGRAM_BINS_PARAMETER,
    SCALAR_LUMA_CHANNEL_AXIS_PARAMETER,
    PrototypePipeline,
    SourcePayload,
    resolve_parameter_visibility,
)


def _pipeline_with(operation_id: str) -> tuple[PrototypePipeline, str]:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    node = pipeline.add_node(operation_id)
    assert pipeline.connect("input", node.id).success
    return pipeline, node.id


def _inferred_named_state(
    data,
    names: str,
    *,
    scales: tuple[float, ...] | None = None,
    translations: tuple[float, ...] | None = None,
    source_axes: tuple[int, ...] | None = None,
):
    """Build carried named axes whose semantic provenance remains inferred."""
    normalized = tuple(name.casefold() for name in names)
    axis_types = {
        "t": "time",
        "c": "channel",
        "z": "space",
        "y": "space",
        "x": "space",
    }
    count = len(normalized)
    scales = scales or (1.0,) * count
    translations = translations or (0.0,) * count
    source_axes = source_axes or tuple(range(count))
    axes = tuple(
        AxisMetadata(
            name,
            axis_types.get(name, "unknown"),
            scale=scales[index],
            translation=translations[index],
            source_axis=source_axes[index],
            confidence=AXIS_CONFIDENCE_INFERRED,
        )
        for index, name in enumerate(normalized)
    )
    return image_state_from_array(
        data,
        axes=axes,
        metadata_source="carried inferred test axes",
    )


@pytest.mark.parametrize(
    ("dtype", "expected"),
    (
        (np.float32, True),
        (np.float64, True),
        (np.uint8, False),
        (np.uint16, False),
        (np.int16, False),
        (np.bool_, False),
    ),
)
def test_float_parameter_visibility_uses_resolved_input_dtype(dtype, expected):
    data = np.zeros((4, 5), dtype=dtype)

    result = resolve_parameter_visibility(
        HISTOGRAM_BINS_PARAMETER,
        input_data=data,
    )

    assert result.visible is expected


def test_float_parameter_visibility_uses_state_dtype_without_materialized_data():
    state = image_state_from_array(
        np.zeros((4, 5), dtype=np.uint16),
        layer_metadata={"axes": "YX"},
    )

    result = resolve_parameter_visibility(
        HISTOGRAM_BINS_PARAMETER,
        input_state=state,
    )

    assert not result.visible


def test_unresolved_parameter_inputs_remain_visible():
    assert resolve_parameter_visibility(HISTOGRAM_BINS_PARAMETER).visible
    assert resolve_parameter_visibility(SCALAR_LUMA_CHANNEL_AXIS_PARAMETER).visible


def test_rgb_parameter_visibility_requires_explicit_rgb_semantics():
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    explicit_rgb = image_state_from_array(
        rgb,
        layer_metadata={"axes": "Y,X,rgb"},
    )
    explicit_scalar = image_state_from_array(
        np.zeros((3, 4, 5), dtype=np.uint8),
        layer_metadata={"axes": "ZYX"},
    )
    inferred_from_shape = image_state_from_array(rgb)

    assert resolve_parameter_visibility(
        SCALAR_LUMA_CHANNEL_AXIS_PARAMETER,
        input_data=rgb,
        input_state=explicit_rgb,
    ).visible
    assert not resolve_parameter_visibility(
        SCALAR_LUMA_CHANNEL_AXIS_PARAMETER,
        input_state=explicit_scalar,
    ).visible
    assert resolve_parameter_visibility(
        SCALAR_LUMA_CHANNEL_AXIS_PARAMETER,
        input_data=rgb,
        input_state=inferred_from_shape,
    ).visible


def test_auto_spatial_mode_rejects_shape_inference_but_explicit_mode_runs():
    data = np.zeros((3, 8, 9), dtype=bool)
    automatic, automatic_id = _pipeline_with("hysteresis_threshold")

    with pytest.raises(AmbiguousAxisError, match="Auto from axes"):
        automatic.run(data)

    explicit, explicit_id = _pipeline_with("hysteresis_threshold")
    explicit.set_param(explicit_id, "spatial_mode", "2D YX")
    explicit.run(data)

    assert automatic.nodes[automatic_id].params["spatial_mode"] == "Auto from axes"
    assert explicit.nodes[explicit_id].params["resolved_spatial_ndim"] == 2
    assert explicit.outputs[explicit_id].shape == data.shape


def test_auto_spatial_mode_preserves_explicit_layer_axes():
    data = np.zeros((3, 8, 9), dtype=bool)
    pipeline, node_id = _pipeline_with("hysteresis_threshold")

    pipeline.run(data, input_metadata={"axes": "ZYX"})

    assert pipeline.nodes[node_id].params["resolved_spatial_ndim"] == 3


def test_auto_spatial_mode_accepts_unambiguous_two_dimensional_array():
    data = np.zeros((8, 9), dtype=bool)
    pipeline, node_id = _pipeline_with("hysteresis_threshold")

    pipeline.run(data)

    assert pipeline.nodes[node_id].params["resolved_spatial_ndim"] == 2


def test_project_auto_requires_explicit_axes_but_index_selection_is_safe():
    data = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    automatic, _automatic_id = _pipeline_with("project_image")

    with pytest.raises(AmbiguousAxisError, match="projection axes"):
        automatic.run(data)

    indexed, indexed_id = _pipeline_with("project_image")
    indexed.set_param(indexed_id, "axes", "axis:0")
    indexed.run(data)

    np.testing.assert_array_equal(indexed.outputs[indexed_id], data.max(axis=0))

    named, named_id = _pipeline_with("project_image")
    named.run(data, input_metadata={"axes": "ZYX"})
    np.testing.assert_array_equal(named.outputs[named_id], data.max(axis=0))


def test_project_non_yx_spatial_uses_named_z_after_reorder():
    data = np.arange(2 * 4 * 3 * 5, dtype=np.float32).reshape(2, 4, 3, 5)
    pipeline, node_id = _pipeline_with("project_image")
    pipeline.set_param(node_id, "axes", "non_yx_spatial")

    pipeline.run(data, input_metadata={"axes": "CYZX"})

    np.testing.assert_array_equal(pipeline.outputs[node_id], data.max(axis=2))
    assert pipeline.output_states[node_id].axis_order == "CYX"


@pytest.mark.parametrize(
    "operation_id",
    ("gaussian_blur", "gaussian_blur_3d", "hysteresis_threshold"),
)
def test_positional_spatial_operations_reject_noncanonical_explicit_axes(
    operation_id,
):
    data = np.zeros((2, 4, 3, 5), dtype=np.float32)
    pipeline, _node_id = _pipeline_with(operation_id)

    with pytest.raises(AmbiguousAxisError, match="positional.*processing"):
        pipeline.run(data, input_metadata={"axes": "CYZX"})


def test_qyx_volume_needs_declaration_before_sequential_zyx_processing():
    data = np.arange(3 * 8 * 9, dtype=np.float32).reshape(3, 8, 9)

    def workflow() -> tuple[PrototypePipeline, str]:
        pipeline = PrototypePipeline()
        pipeline.reset_empty_graph()
        subtract = pipeline.add_node("subtract_background")
        pipeline.set_param(subtract.id, "spatial_mode", "3D ZYX")
        pipeline.set_param(subtract.id, "radius", 1.0)
        assert pipeline.connect("input", subtract.id).success
        blur = pipeline.add_node("gaussian_blur_3d")
        assert pipeline.connect(subtract.id, blur.id).success
        return pipeline, blur.id

    raw_state = image_state_from_array(data, layer_metadata={"axes": "QYX"})
    rejected, _rejected_output = workflow()
    with pytest.raises(AmbiguousAxisError, match="Declare axes.*QYX -> ZYX"):
        rejected.run(
            None,
            source_payloads={
                "input": SourcePayload(data, image_state=raw_state),
            },
        )

    declared_state = apply_axis_declaration(
        raw_state,
        AxisDeclaration("QYX", "ZYX"),
        declaration_source="test config",
    )
    accepted, accepted_output = workflow()
    accepted.run(
        None,
        source_payloads={
            "input": SourcePayload(data, image_state=declared_state),
        },
    )

    assert accepted.outputs[accepted_output].shape == data.shape
    assert accepted.output_states[accepted_output].axis_order == "ZYX"


def _skeletonize_pipeline() -> tuple[PrototypePipeline, str]:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    assert pipeline.connect("input", threshold.id).success
    skeletonize = pipeline.add_node("skeletonize")
    assert pipeline.connect(threshold.id, skeletonize.id).success
    return pipeline, skeletonize.id


def test_skeletonize_auto_rejects_ambiguous_qyx_before_pixel_execution(
    monkeypatch,
):
    data = np.ones((3, 8, 9), dtype=np.uint8)
    raw_state = image_state_from_array(data, layer_metadata={"axes": "QYX"})
    pipeline, _skeletonize_id = _skeletonize_pipeline()

    def unexpected_kernel(*_args, **_kwargs):
        raise AssertionError("axis preflight must not execute pixel kernels")

    monkeypatch.setattr(
        "napari_vipp.core.pipeline.execute_prepared_node_call",
        unexpected_kernel,
    )

    with pytest.raises(
        AmbiguousAxisError,
        match="QYX.*unknown spatial meaning.*QYX -> ZYX",
    ):
        pipeline.preflight_axis_contract(
            {"input": SourcePayload(data, image_state=raw_state)}
        )


def test_skeletonize_auto_rejects_trailing_qyx_below_leading_time_axis(
    monkeypatch,
):
    data = np.ones((2, 3, 8, 9), dtype=np.uint8)
    raw_state = image_state_from_array(data, layer_metadata={"axes": "TQYX"})
    pipeline, _skeletonize_id = _skeletonize_pipeline()

    def unexpected_kernel(*_args, **_kwargs):
        raise AssertionError("axis preflight must not execute pixel kernels")

    monkeypatch.setattr(
        "napari_vipp.core.pipeline.execute_prepared_node_call",
        unexpected_kernel,
    )

    with pytest.raises(
        AmbiguousAxisError,
        match="TQYX.*unknown spatial meaning.*TQYX -> TZYX",
    ) as raised:
        pipeline.preflight_axis_contract(
            {"input": SourcePayload(data, image_state=raw_state)}
        )

    assert raised.value.detected_axes == "TQYX"
    assert raised.value.required_axes == "TZYX"


def test_tqyx_declaration_runs_one_volumetric_lee_block_per_timepoint(
    monkeypatch,
):
    data = np.zeros((2, 3, 8, 9), dtype=np.uint8)
    data[:, :, 2:6, 2:7] = 1
    raw_state = image_state_from_array(data, layer_metadata={"axes": "TQYX"})
    pipeline, skeletonize_id = _skeletonize_pipeline()
    pipeline.set_param("input", "axis_declaration", "TQYX -> TZYX")

    original = operations.morphology.skeletonize
    calls: list[tuple[tuple[int, ...], str | None]] = []

    def record_call(block, *, method=None):
        calls.append((tuple(block.shape), method))
        return original(block, method=method)

    monkeypatch.setattr(operations.morphology, "skeletonize", record_call)
    pipeline.run(
        None,
        source_payloads={"input": SourcePayload(data, image_state=raw_state)},
    )

    assert calls == [((3, 8, 9), "lee"), ((3, 8, 9), "lee")]
    assert pipeline.nodes[skeletonize_id].params["resolved_spatial_ndim"] == 3
    assert pipeline.output_states[skeletonize_id].axis_order == "TZYX"


def test_image_source_qyx_declaration_runs_one_volumetric_lee_block(monkeypatch):
    data = np.zeros((3, 8, 9), dtype=np.uint8)
    data[:, 2:6, 2:7] = 1
    raw_state = image_state_from_array(
        data,
        axes=(
            AxisMetadata(
                "q",
                "unknown",
                unit="micrometer",
                scale=2.0,
                translation=10.0,
            ),
            AxisMetadata(
                "y",
                "space",
                unit="micrometer",
                scale=0.5,
                translation=20.0,
            ),
            AxisMetadata(
                "x",
                "space",
                unit="micrometer",
                scale=0.25,
                translation=30.0,
            ),
        ),
    )
    pipeline, skeletonize_id = _skeletonize_pipeline()
    pipeline.set_param("input", "axis_declaration", "QYX -> ZYX")

    original = operations.morphology.skeletonize
    calls: list[tuple[tuple[int, ...], str | None]] = []

    def record_call(block, *, method=None):
        calls.append((tuple(block.shape), method))
        return original(block, method=method)

    monkeypatch.setattr(operations.morphology, "skeletonize", record_call)
    pipeline.run(
        None,
        source_payloads={"input": SourcePayload(data, image_state=raw_state)},
    )

    output_state = pipeline.output_states[skeletonize_id]
    assert calls == [((3, 8, 9), "lee")]
    assert pipeline.nodes[skeletonize_id].params["resolved_spatial_ndim"] == 3
    assert output_state.axis_order == "ZYX"
    assert output_state.axes == pipeline.output_states["input"].axes
    assert [axis.scale for axis in output_state.axes] == [2.0, 0.5, 0.25]
    assert [axis.translation for axis in output_state.axes] == [10.0, 20.0, 30.0]
    assert [axis.unit for axis in output_state.axes] == ["micrometer"] * 3
    assert "method=Lee (Auto-resolved)" in output_state.history[-1]
    assert "resolved=3D ZYX volumetric block" in output_state.history[-1]
    assert "3x3x3 neighborhood" in output_state.history[-1]
    assert "boundary=background" in output_state.history[-1]


def test_skeletonize_explicit_2d_keeps_q_as_independent_leading_blocks(
    monkeypatch,
):
    data = np.ones((3, 8, 9), dtype=np.uint8)
    raw_state = image_state_from_array(data, layer_metadata={"axes": "QYX"})
    pipeline, skeletonize_id = _skeletonize_pipeline()
    pipeline.set_param(skeletonize_id, "spatial_mode", "2D YX")

    original = operations.morphology.skeletonize
    calls: list[tuple[tuple[int, ...], str | None]] = []

    def record_call(block, *, method=None):
        calls.append((tuple(block.shape), method))
        return original(block, method=method)

    monkeypatch.setattr(operations.morphology, "skeletonize", record_call)
    pipeline.run(
        None,
        source_payloads={"input": SourcePayload(data, image_state=raw_state)},
    )

    assert calls == [((8, 9), "zhang")] * 3
    assert pipeline.nodes[skeletonize_id].params["resolved_spatial_ndim"] == 2
    assert pipeline.output_states[skeletonize_id].axis_order == "QYX"


def test_skeletonize_preflight_rejects_zhang_for_empty_zyx_without_kernels(
    monkeypatch,
):
    data = np.zeros((3, 8, 9), dtype=np.uint8)
    state = image_state_from_array(data, layer_metadata={"axes": "ZYX"})
    pipeline, skeletonize_id = _skeletonize_pipeline()
    pipeline.set_param(skeletonize_id, "method", "Zhang 2D")

    def unexpected_kernel(*_args, **_kwargs):
        raise AssertionError("scientific preflight must not execute pixel kernels")

    monkeypatch.setattr(
        "napari_vipp.core.pipeline.execute_prepared_node_call",
        unexpected_kernel,
    )

    with pytest.raises(
        ValueError,
        match="Zhang 2D.*cannot process a 3D ZYX volume",
    ):
        pipeline.preflight_axis_contract(
            {"input": SourcePayload(data, image_state=state)}
        )


def test_axis_preflight_propagates_through_reorder_and_convert_without_kernels(
    monkeypatch,
):
    data = np.zeros((3, 8, 9), dtype=np.uint16)
    raw_state = image_state_from_array(data, layer_metadata={"axes": "QYX"})
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    reorder = pipeline.add_node("reorder_axes")
    assert pipeline.connect("input", reorder.id).success
    convert = pipeline.add_node("convert_dtype")
    pipeline.set_param(convert.id, "output_dtype", "float32")
    assert pipeline.connect(reorder.id, convert.id).success
    subtract = pipeline.add_node("subtract_background")
    pipeline.set_param(subtract.id, "spatial_mode", "3D ZYX")
    assert pipeline.connect(convert.id, subtract.id).success

    def unexpected_kernel(*_args, **_kwargs):
        raise AssertionError("axis preflight must not execute pixel kernels")

    monkeypatch.setattr(
        "napari_vipp.core.pipeline.execute_prepared_node_call",
        unexpected_kernel,
    )

    with pytest.raises(AmbiguousAxisError, match="Declare axes.*QYX -> ZYX"):
        pipeline.preflight_axis_contract(
            {"input": SourcePayload(data, image_state=raw_state)}
        )


def test_axis_preflight_never_scans_lazy_source_statistics(monkeypatch):
    class MetadataOnlyArray:
        shape = (40, 1536, 1763)
        ndim = 3
        dtype = np.dtype(np.uint16)

        def compute(self):
            raise AssertionError("axis preflight must not compute source pixels")

    source = MetadataOnlyArray()
    state = image_state_from_array(source, layer_metadata={"axes": "ZYX"})
    assert state is not None
    assert state.value_range == "not computed (lazy)"
    pipeline, node_id = _pipeline_with("subtract_background")
    pipeline.set_param(node_id, "spatial_mode", "3D ZYX")

    def unexpected_statistics(*_args, **_kwargs):
        raise AssertionError("axis preflight must not calculate value statistics")

    monkeypatch.setattr(
        "napari_vipp.core.metadata._value_range_label",
        unexpected_statistics,
    )

    pipeline.preflight_axis_contract(
        {"input": SourcePayload(source, image_state=state)}
    )


def test_targeted_axis_preflight_projects_only_declared_gaussian_branch(
    monkeypatch,
):
    data = np.zeros((3, 8, 9), dtype=np.uint16)
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    pipeline.set_param("input", "axis_declaration", "QYX -> ZYX")
    subtract = pipeline.add_node("subtract_background")
    pipeline.set_param(subtract.id, "spatial_mode", "2D YX")
    assert pipeline.connect("input", subtract.id).success
    rescale = pipeline.add_node("rescale_intensity")
    assert pipeline.connect(subtract.id, rescale.id).success
    gaussian = pipeline.add_node("gaussian_blur_3d")
    assert pipeline.connect(rescale.id, gaussian.id).success
    unrelated = pipeline.add_node("reorder_axes")
    pipeline.set_param(unrelated.id, "order", "invalid")
    assert pipeline.connect("input", unrelated.id).success

    def unexpected_statistics(*_args, **_kwargs):
        raise AssertionError("metadata projection must not calculate statistics")

    def unexpected_kernel(*_args, **_kwargs):
        raise AssertionError("metadata projection must not execute pixel kernels")

    monkeypatch.setattr(
        "napari_vipp.core.metadata._value_range_label",
        unexpected_statistics,
    )
    monkeypatch.setattr(
        "napari_vipp.core.pipeline.execute_prepared_node_call",
        unexpected_kernel,
    )

    pipeline.preflight_axis_contract(
        {"input": SourcePayload(data, {"axes": "QYX"})},
        target_node_ids={gaussian.id},
    )

    assert [
        pipeline.output_states[node_id].axis_order
        for node_id in ("input", subtract.id, rescale.id, gaussian.id)
    ] == ["ZYX"] * 4
    assert pipeline.output_states["input"].value_range == DEFERRED_VALUE_RANGE
    assert pipeline.output_states[unrelated.id] is None


def test_axis_preflight_propagates_qyx_through_axis_slice():
    data = np.zeros((3, 8, 9), dtype=np.uint16)
    raw_state = image_state_from_array(data, layer_metadata={"axes": "QYX"})
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    select = pipeline.add_node("select_axis_slice")
    pipeline.set_param(select.id, "axis", 1)
    assert pipeline.connect("input", select.id).success
    subtract = pipeline.add_node("subtract_background")
    pipeline.set_param(subtract.id, "spatial_mode", "2D YX")
    assert pipeline.connect(select.id, subtract.id).success

    with pytest.raises(AmbiguousAxisError, match="effective axis order is QX"):
        pipeline.preflight_axis_contract(
            {"input": SourcePayload(data, image_state=raw_state)}
        )


def test_axis_preflight_propagates_rank_reduction_before_3d_operation():
    data = np.zeros((3, 8, 9), dtype=np.uint16)
    state = image_state_from_array(data, layer_metadata={"axes": "ZYX"})
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    select = pipeline.add_node("select_axis_slice")
    pipeline.set_param(select.id, "axis", 0)
    assert pipeline.connect("input", select.id).success
    subtract = pipeline.add_node("subtract_background")
    pipeline.set_param(subtract.id, "spatial_mode", "3D ZYX")
    assert pipeline.connect(select.id, subtract.id).success

    with pytest.raises(ValueError, match="requested 3D.*2D input"):
        pipeline.preflight_axis_contract(
            {"input": SourcePayload(data, image_state=state)}
        )


def test_axis_preflight_propagates_through_multi_output_image_node():
    data = np.zeros((3, 8, 9), dtype=np.uint8)
    state = image_state_from_array(data, layer_metadata={"axes": "QYX"})
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    assert pipeline.connect("input", threshold.id).success
    keypoints = pipeline.add_node("skeleton_keypoints")
    pipeline.set_param(keypoints.id, "spatial_mode", "2D YX")
    assert pipeline.connect(threshold.id, keypoints.id).success
    blur = pipeline.add_node("gaussian_blur_3d")
    assert pipeline.connect(keypoints.id, blur.id, source_port=0).success

    with pytest.raises(AmbiguousAxisError, match="effective axis order is QYX"):
        pipeline.preflight_axis_contract(
            {"input": SourcePayload(data, image_state=state)}
        )

    assert [
        output_state.axis_order
        for output_state in pipeline.node_output_states[keypoints.id]
    ] == ["QYX", "QYX", "QYX"]


def test_skeleton_keypoints_axis_preflight_metadata_matches_execution():
    data = np.zeros((3, 4, 5), dtype=np.uint8)
    channels = tuple(
        ChannelMetadata(name=name) for name in ("red", "green", "blue")
    )
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("c", "channel"),
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
        ),
        channels=channels,
    )

    def workflow():
        pipeline = PrototypePipeline()
        pipeline.reset_empty_graph()
        threshold = pipeline.add_node("binary_threshold")
        assert pipeline.connect("input", threshold.id).success
        keypoints = pipeline.add_node("skeleton_keypoints")
        pipeline.set_param(keypoints.id, "spatial_mode", "2D YX")
        assert pipeline.connect(threshold.id, keypoints.id).success
        return pipeline, keypoints.id

    preflight, preflight_node = workflow()
    preflight.preflight_axis_contract(
        {"input": SourcePayload(data, image_state=state)}
    )
    executed, executed_node = workflow()
    executed.run(
        None,
        source_payloads={"input": SourcePayload(data, image_state=state)},
    )

    def signatures(pipeline, node_id):
        return tuple(
            (
                output_state.shape,
                output_state.dtype,
                output_state.axes,
                output_state.channels,
            )
            for output_state in pipeline.node_output_states[node_id]
        )

    assert signatures(preflight, preflight_node) == signatures(
        executed,
        executed_node,
    )
    assert all(
        output_state.channels == channels
        for output_state in preflight.node_output_states[preflight_node]
    )


@pytest.mark.parametrize(
    ("operation_id", "shape", "dtype", "axes", "params"),
    (
        ("select_axis_slice", (2, 3, 4), np.uint8, "ZYX", (("axis", 0),)),
        ("mip", (2, 3, 4), np.uint8, "ZYX", (("axis", 0),)),
        (
            "project_image",
            (2, 3, 4),
            np.uint8,
            "ZYX",
            (("axes", "axis:0"), ("method", "Mean")),
        ),
        ("orthogonal_projection", (2, 3, 4), np.uint8, "ZYX", ()),
        ("extract_channel", (3, 4, 5), np.uint8, "CYX", (("channel", 1),)),
        ("composite_to_rgb", (3, 4, 5), np.uint8, "CYX", ()),
        ("split_axis", (2, 3, 4), np.uint8, "ZYX", (("axis", "axis:0"),)),
        ("split_channels", (3, 4, 5), np.uint8, "CYX", ()),
        ("otsu_threshold", (3, 4, 5), np.uint8, "CYX", (("channel_axis", 0),)),
        ("sobel_filter", (3, 4, 5), np.uint16, "CYX", (("channel_axis", 0),)),
        ("laplace_filter", (3, 4, 5), np.uint16, "CYX", (("channel_axis", 0),)),
        ("laplace_filter", (3, 4, 5), np.float64, "CYX", (("channel_axis", 0),)),
        (
            "born_wolf_psf",
            (4, 5),
            np.float32,
            "YX",
            (
                ("spatial_mode", "2D YX"),
                ("auto_parameters", False),
                ("wavelength_nm", 520.0),
                ("numerical_aperture", 1.0),
                ("refractive_index", 1.33),
                ("pixel_size_xy_um", 0.1),
                ("z_step_um", 0.2),
                ("xy_size", 9),
                ("z_size", 3),
                ("pupil_samples", 16),
            ),
        ),
    ),
)
def test_deterministic_axis_preflight_metadata_matches_execution(
    operation_id,
    shape,
    dtype,
    axes,
    params,
):
    data = np.zeros(shape, dtype=dtype)
    state = image_state_from_array(data, layer_metadata={"axes": axes})

    def workflow():
        pipeline, node_id = _pipeline_with(operation_id)
        for name, value in params:
            pipeline.set_param(node_id, name, value)
        return pipeline, node_id

    preflight, preflight_node = workflow()
    preflight.preflight_axis_contract(
        {"input": SourcePayload(data, image_state=state)}
    )
    executed, executed_node = workflow()
    executed.run(
        None,
        source_payloads={"input": SourcePayload(data, image_state=state)},
    )

    def signatures(pipeline, node_id):
        return tuple(
            None
            if output_state is None
            else (
                output_state.shape,
                output_state.dtype,
                output_state.axes,
                output_state.channels,
            )
            for output_state in pipeline.node_output_states[node_id]
        )

    assert signatures(preflight, preflight_node) == signatures(
        executed,
        executed_node,
    )


def test_combine_channels_axis_preflight_metadata_matches_execution():
    first = np.zeros((4, 5), dtype=np.uint8)
    second = np.zeros((4, 5), dtype=np.uint16)
    first_state = image_state_from_array(first, layer_metadata={"axes": "YX"})
    second_state = image_state_from_array(second, layer_metadata={"axes": "YX"})

    def workflow():
        pipeline = PrototypePipeline()
        pipeline.reset_empty_graph()
        second_source = pipeline.add_node("input")
        combine = pipeline.add_node("combine_channels")
        pipeline.set_param(combine.id, "input_count", 2)
        assert pipeline.connect("input", combine.id, target_port=0).success
        assert pipeline.connect(second_source.id, combine.id, target_port=1).success
        return pipeline, second_source.id, combine.id

    preflight, preflight_source, preflight_node = workflow()
    preflight.preflight_axis_contract(
        {
            "input": SourcePayload(first, image_state=first_state),
            preflight_source: SourcePayload(second, image_state=second_state),
        }
    )
    executed, executed_source, executed_node = workflow()
    executed.run(
        None,
        source_payloads={
            "input": SourcePayload(first, image_state=first_state),
            executed_source: SourcePayload(second, image_state=second_state),
        },
    )

    preflight_state = preflight.output_states[preflight_node]
    executed_state = executed.output_states[executed_node]
    assert preflight_state is not None
    assert executed_state is not None
    assert (
        preflight_state.shape,
        preflight_state.dtype,
        preflight_state.axes,
        preflight_state.channels,
    ) == (
        executed_state.shape,
        executed_state.dtype,
        executed_state.axes,
        executed_state.channels,
    )


def test_combine_channels_axis_preflight_rejects_misaligned_grids():
    first = np.zeros((4, 5), dtype=np.uint8)
    second = np.zeros((4, 5), dtype=np.uint8)
    first_state = image_state_from_array(
        first,
        axes=(
            AxisMetadata("y", "space", unit="micrometer", scale=0.2),
            AxisMetadata("x", "space", unit="micrometer", scale=0.2),
        ),
    )
    second_state = image_state_from_array(
        second,
        axes=(
            AxisMetadata("y", "space", unit="micrometer", scale=0.4),
            AxisMetadata("x", "space", unit="micrometer", scale=0.4),
        ),
    )
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    second_source = pipeline.add_node("input")
    combine = pipeline.add_node("combine_channels")
    pipeline.set_param(combine.id, "input_count", 2)
    assert pipeline.connect("input", combine.id, target_port=0).success
    assert pipeline.connect(second_source.id, combine.id, target_port=1).success

    with pytest.raises(ValueError, match="scale|grid"):
        pipeline.preflight_axis_contract(
            {
                "input": SourcePayload(first, image_state=first_state),
                second_source.id: SourcePayload(
                    second,
                    image_state=second_state,
                ),
            }
        )


def test_skeleton_overlay_axis_preflight_preserves_existing_channel_axis():
    data = np.zeros((3, 5, 7), dtype=np.uint8)
    state = image_state_from_array(data, layer_metadata={"axes": "CYX"})

    def workflow():
        pipeline = PrototypePipeline()
        pipeline.reset_empty_graph()
        threshold = pipeline.add_node("binary_threshold")
        assert pipeline.connect("input", threshold.id).success
        overlay = pipeline.add_node("skeleton_graph_overlay")
        pipeline.set_param(overlay.id, "spatial_mode", "2D YX")
        assert pipeline.connect(threshold.id, overlay.id).success
        return pipeline, overlay.id

    preflight, preflight_node = workflow()
    preflight.preflight_axis_contract(
        {"input": SourcePayload(data, image_state=state)}
    )
    executed, executed_node = workflow()
    executed.run(
        None,
        source_payloads={"input": SourcePayload(data, image_state=state)},
    )

    assert preflight.output_states[preflight_node].axis_order == "C,Y,X,rgb"
    assert (
        preflight.output_states[preflight_node].axes
        == executed.output_states[executed_node].axes
    )


def test_axis_removal_updates_channel_metadata_for_projection_and_slice():
    data = np.zeros((3, 5, 7), dtype=np.uint8)
    channels = tuple(ChannelMetadata(name=name) for name in ("red", "green", "blue"))
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("c", "channel"),
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
        ),
        channels=channels,
    )

    projection, projection_node = _pipeline_with("mip")
    projection.set_param(projection_node, "axis", 0)
    projection.run(
        None,
        source_payloads={"input": SourcePayload(data, image_state=state)},
    )
    assert projection.output_states[projection_node].axis_order == "YX"
    assert projection.output_states[projection_node].channels == ()

    selection, selection_node = _pipeline_with("select_axis_slice")
    selection.set_param(selection_node, "axis", 0)
    selection.set_param(selection_node, "index", 1)
    selection.run(
        None,
        source_payloads={"input": SourcePayload(data, image_state=state)},
    )
    assert selection.output_states[selection_node].axis_order == "YX"
    assert tuple(
        channel.name for channel in selection.output_states[selection_node].channels
    ) == ("green",)


@pytest.mark.parametrize(
    ("shape", "channel_axis", "message"),
    (
        ((3, 4, 5), 5, "out of range"),
        ((2, 4, 5), 0, "exactly 3 RGB or 4 RGBA"),
        ((3, 5), 0, "at least two spatial dimensions"),
    ),
)
def test_axis_preflight_validates_explicit_luma_channel_contract(
    shape,
    channel_axis,
    message,
):
    data = np.zeros(shape, dtype=np.uint16)
    axes = "CYX" if len(shape) == 3 else "CY"
    state = image_state_from_array(data, layer_metadata={"axes": axes})
    pipeline, node_id = _pipeline_with("otsu_threshold")
    pipeline.set_param(node_id, "channel_axis", channel_axis)

    with pytest.raises(ValueError, match=message):
        pipeline.preflight_axis_contract(
            {"input": SourcePayload(data, image_state=state)}
        )


def test_born_wolf_axis_preflight_requires_resolved_metadata():
    data = np.zeros((3, 8, 9), dtype=np.uint16)
    state = image_state_from_array(data, layer_metadata={"axes": "ZYX"})
    pipeline, _node_id = _pipeline_with("born_wolf_psf")

    with pytest.raises(ValueError, match="unresolved parameter"):
        pipeline.preflight_axis_contract(
            {"input": SourcePayload(data, image_state=state)}
        )


def test_multichannel_born_wolf_axis_preflight_resolves_each_output():
    data = np.zeros((2, 3, 8, 9), dtype=np.uint16)
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("c", "channel"),
            AxisMetadata("z", "space", unit="micrometer", scale=0.4),
            AxisMetadata("y", "space", unit="micrometer", scale=0.08),
            AxisMetadata("x", "space", unit="micrometer", scale=0.08),
        ),
        channels=(
            ChannelMetadata(
                name="green",
                emission_wavelength=520.0,
                emission_wavelength_unit="nanometer",
            ),
            ChannelMetadata(
                name="red",
                emission_wavelength=620.0,
                emission_wavelength_unit="nanometer",
            ),
        ),
        acquisition=AcquisitionMetadata(
            objective_na=1.2,
            refractive_index=1.33,
        ),
    )
    pipeline, node_id = _pipeline_with("born_wolf_psf")
    pipeline.set_param(node_id, "xy_size", 9)
    pipeline.set_param(node_id, "z_size", 3)
    pipeline.set_param(node_id, "pupil_samples", 16)

    pipeline.preflight_axis_contract(
        {"input": SourcePayload(data, image_state=state)}
    )

    output_states = pipeline.node_output_states[node_id]
    assert [output.shape for output in output_states] == [(3, 9, 9), (3, 9, 9)]
    assert [output.axis_order for output in output_states] == ["ZYX", "ZYX"]
    assert [output.channels[0].name for output in output_states] == ["green", "red"]
    assert "wavelength 620" in output_states[1].history[-1]


@pytest.mark.parametrize("axes", ("TYX", "CYX"))
def test_explicit_nonspatial_leading_axis_is_not_assumed_to_be_z(axes):
    data = np.zeros((3, 8, 9), dtype=np.float32)
    pipeline, node_id = _pipeline_with("subtract_background")
    pipeline.set_param(node_id, "spatial_mode", "3D ZYX")

    with pytest.raises(AmbiguousAxisError, match="Declare axes"):
        pipeline.run(data, input_metadata={"axes": axes})


@pytest.mark.parametrize("axes", ("TYX", "CYX"))
def test_explicit_noncanonical_axis_requires_recorded_declaration(axes):
    data = np.zeros((3, 8, 9), dtype=np.float32)
    raw_state = image_state_from_array(data, layer_metadata={"axes": axes})
    declared_state = apply_axis_declaration(
        raw_state,
        AxisDeclaration(axes, "ZYX"),
        declaration_source="reviewed batch config",
    )
    pipeline, node_id = _pipeline_with("subtract_background")
    pipeline.set_param(node_id, "spatial_mode", "3D ZYX")

    pipeline.run(
        None,
        source_payloads={
            "input": SourcePayload(data, image_state=declared_state),
        },
    )

    assert pipeline.output_states[node_id].axis_order == "ZYX"
    assert "data order unchanged" in pipeline.output_states[node_id].history[0]
    assert pipeline.output_states[node_id].kind == "intensity image"


def test_multi_input_positional_operation_rejects_noncanonical_explicit_axes():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    psf_source = pipeline.add_node("input")
    deconvolution = pipeline.add_node("richardson_lucy_deconvolution")
    pipeline.set_param(deconvolution.id, "spatial_mode", "2D YX")
    assert pipeline.connect("input", deconvolution.id, target_port=0).success
    assert pipeline.connect(psf_source.id, deconvolution.id, target_port=1).success
    data = np.zeros((2, 4, 3, 5), dtype=np.float32)

    with pytest.raises(AmbiguousAxisError, match="positional YX processing"):
        pipeline.run(
            data,
            input_metadata={"axes": "CYZX"},
            source_payloads={
                psf_source.id: SourcePayload(data, {"axes": "CYZX"}),
            },
        )


@pytest.mark.parametrize(
    "operation_id",
    ("orthogonal_projection", "set_pixel_size"),
)
def test_named_spatial_role_operations_require_explicit_spatial_axes(operation_id):
    data = np.zeros((3, 8, 9), dtype=np.float32)
    inferred, _inferred_id = _pipeline_with(operation_id)

    with pytest.raises(AmbiguousAxisError, match="explicit spatial ax"):
        inferred.run(data)

    explicit_but_nonspatial, _nonspatial_id = _pipeline_with(operation_id)
    with pytest.raises(AmbiguousAxisError, match="explicit spatial ax"):
        explicit_but_nonspatial.run(data, input_metadata={"axes": "ABC"})


def test_rescale_axes_accepts_inferred_yx_and_records_the_authored_plane():
    data = np.arange(8 * 10, dtype=np.uint16).reshape(8, 10)
    input_state = image_state_from_array(data)
    pipeline, node_id = _pipeline_with("rescale_axes")
    pipeline.set_param(node_id, "lock_xy", False)
    pipeline.set_param(node_id, "x_scale", 2.0)
    pipeline.set_param(node_id, "y_scale", 0.5)

    pipeline.run(
        None,
        source_payloads={
            "input": SourcePayload(data, image_state=input_state),
        },
    )

    output_state = pipeline.output_states[node_id]
    assert output_state is not None
    assert pipeline.outputs[node_id].shape == (4, 20)
    assert output_state.axis_order == "YX"
    assert output_state.axis_confidence == AXIS_CONFIDENCE_EXPLICIT
    assert [axis.confidence for axis in output_state.axes] == [
        AXIS_CONFIDENCE_EXPLICIT,
        AXIS_CONFIDENCE_EXPLICIT,
    ]
    assert [axis.scale for axis in output_state.axes] == [2.0, 0.5]
    history = output_state.history[-1].casefold()
    assert "inferred" in history
    assert "x/y" in history or "y/x" in history
    assert "confidence" in history


def test_rescale_axes_true_noop_preserves_inferred_axis_confidence():
    data = np.arange(6 * 7, dtype=np.float32).reshape(6, 7)
    input_state = image_state_from_array(data)
    pipeline, node_id = _pipeline_with("rescale_axes")

    pipeline.run(
        None,
        source_payloads={
            "input": SourcePayload(data, image_state=input_state),
        },
    )

    output_state = pipeline.output_states[node_id]
    assert output_state is not None
    assert pipeline.outputs[node_id].shape == data.shape
    assert output_state.axes == input_state.axes
    assert output_state.axis_confidence == AXIS_CONFIDENCE_INFERRED
    history = output_state.history[-1].casefold()
    assert "inferred y/x" in history or "inferred x/y" in history
    assert "retained" in history
    assert "sizes did not change" in history


def test_rescale_axes_changes_inferred_yx_but_preserves_q_metadata():
    data = np.arange(3 * 8 * 10, dtype=np.uint16).reshape(3, 8, 10)
    input_state = _inferred_named_state(
        data,
        "QYX",
        scales=(7.0, 0.5, 0.25),
        translations=(11.0, 22.0, 33.0),
        source_axes=(4, 5, 6),
    )
    pipeline, node_id = _pipeline_with("rescale_axes")
    pipeline.set_param(node_id, "lock_xy", False)
    pipeline.set_param(node_id, "x_scale", 2.0)
    pipeline.set_param(node_id, "y_scale", 0.5)

    pipeline.run(
        None,
        source_payloads={
            "input": SourcePayload(data, image_state=input_state),
        },
    )

    output_state = pipeline.output_states[node_id]
    assert output_state is not None
    assert pipeline.outputs[node_id].shape == (3, 4, 20)
    assert output_state.axis_order == "QYX"
    assert output_state.axes[0] == input_state.axes[0]
    assert output_state.axis_confidence == AXIS_CONFIDENCE_MIXED
    assert [axis.confidence for axis in output_state.axes] == [
        AXIS_CONFIDENCE_INFERRED,
        AXIS_CONFIDENCE_EXPLICIT,
        AXIS_CONFIDENCE_EXPLICIT,
    ]
    assert [axis.scale for axis in output_state.axes] == [7.0, 1.0, 0.125]
    assert [axis.translation for axis in output_state.axes] == [11.0, 22.0, 33.0]
    assert [axis.source_axis for axis in output_state.axes] == [4, 5, 6]
    history = output_state.history[-1].casefold()
    assert "inferred" in history
    assert "x/y" in history or "y/x" in history
    assert "confidence" in history


@pytest.mark.parametrize(
    ("resize_mode", "parameter", "value"),
    (
        ("Scale factor", "z_scale", 1.0001),
        ("Scale factor", "z_scale", 2.0),
        ("Output size", "z_size", 6),
    ),
)
def test_rescale_axes_never_treats_explicit_q_as_z(
    resize_mode,
    parameter,
    value,
):
    data = np.zeros((3, 8, 10), dtype=np.uint16)
    pipeline, node_id = _pipeline_with("rescale_axes")
    pipeline.set_param(node_id, "resize_mode", resize_mode)
    pipeline.set_param(node_id, parameter, value)

    with pytest.raises(AmbiguousAxisError, match=r"Q.*Z|Z.*Q"):
        pipeline.run(data, input_metadata={"axes": "QYX"})

    assert pipeline.outputs[node_id] is None


@pytest.mark.parametrize(
    ("resize_mode", "parameter", "value"),
    (
        ("Scale factor", "z_scale", 2.0),
        ("Output size", "z_size", 6),
    ),
)
def test_rescale_axes_blocks_nontrivial_inferred_z(
    resize_mode,
    parameter,
    value,
):
    data = np.zeros((3, 8, 10), dtype=np.float32)
    input_state = image_state_from_array(data)
    pipeline, node_id = _pipeline_with("rescale_axes")
    pipeline.set_param(node_id, "resize_mode", resize_mode)
    pipeline.set_param(node_id, parameter, value)

    with pytest.raises(AmbiguousAxisError, match=r"explicit Z|Z.*explicit"):
        pipeline.run(
            None,
            source_payloads={
                "input": SourcePayload(data, image_state=input_state),
            },
        )

    assert pipeline.outputs[node_id] is None


def test_rescale_axes_rejects_an_incomplete_inferred_yx_pair():
    data = np.zeros((3, 8, 10), dtype=np.float32)
    input_state = _inferred_named_state(data, "QAX")
    pipeline, node_id = _pipeline_with("rescale_axes")
    pipeline.set_param(node_id, "x_scale", 2.0)

    with pytest.raises(AmbiguousAxisError, match=r"unique.*Y.*X|Y.*X.*unique"):
        pipeline.run(
            None,
            source_payloads={
                "input": SourcePayload(data, image_state=input_state),
            },
        )

    assert pipeline.outputs[node_id] is None


def test_rescale_axes_leaves_inferred_z_unpromoted_during_yx_rescale():
    data = np.zeros((3, 8, 10), dtype=np.float32)
    input_state = image_state_from_array(data)
    pipeline, node_id = _pipeline_with("rescale_axes")
    pipeline.set_param(node_id, "lock_xy", False)
    pipeline.set_param(node_id, "x_scale", 2.0)
    pipeline.set_param(node_id, "y_scale", 0.5)

    pipeline.run(
        None,
        source_payloads={
            "input": SourcePayload(data, image_state=input_state),
        },
    )

    output_state = pipeline.output_states[node_id]
    assert output_state is not None
    assert pipeline.outputs[node_id].shape == (3, 4, 20)
    assert output_state.axes[0] == input_state.axes[0]
    assert [axis.confidence for axis in output_state.axes] == [
        AXIS_CONFIDENCE_INFERRED,
        AXIS_CONFIDENCE_EXPLICIT,
        AXIS_CONFIDENCE_EXPLICIT,
    ]
    assert output_state.axis_confidence == AXIS_CONFIDENCE_MIXED


def test_rescale_axes_preserves_inferred_tcz_while_recording_yx():
    data = np.zeros((2, 3, 4, 6, 8), dtype=np.uint16)
    input_state = _inferred_named_state(data, "TCZYX")
    pipeline, node_id = _pipeline_with("rescale_axes")
    pipeline.set_param(node_id, "lock_xy", False)
    pipeline.set_param(node_id, "x_scale", 1.5)
    pipeline.set_param(node_id, "y_scale", 0.5)

    pipeline.run(
        None,
        source_payloads={
            "input": SourcePayload(data, image_state=input_state),
        },
    )

    output_state = pipeline.output_states[node_id]
    assert output_state is not None
    assert pipeline.outputs[node_id].shape == (2, 3, 4, 3, 12)
    assert output_state.axis_order == "TCZYX"
    assert output_state.axes[:3] == input_state.axes[:3]
    assert [axis.confidence for axis in output_state.axes] == [
        AXIS_CONFIDENCE_INFERRED,
        AXIS_CONFIDENCE_INFERRED,
        AXIS_CONFIDENCE_INFERRED,
        AXIS_CONFIDENCE_EXPLICIT,
        AXIS_CONFIDENCE_EXPLICIT,
    ]
    assert output_state.axis_confidence == AXIS_CONFIDENCE_MIXED


def test_rescale_axes_follows_inferred_yx_names_after_reorder():
    data = np.zeros((3, 8, 10), dtype=np.uint16)
    input_state = _inferred_named_state(
        data,
        "QYX",
        scales=(7.0, 0.5, 0.25),
        translations=(11.0, 22.0, 33.0),
        source_axes=(4, 5, 6),
    )
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    reorder = pipeline.add_node("reorder_axes")
    rescale = pipeline.add_node("rescale_axes")
    assert pipeline.connect("input", reorder.id).success
    assert pipeline.connect(reorder.id, rescale.id).success
    pipeline.set_param(reorder.id, "order", "YQX")
    pipeline.set_param(rescale.id, "lock_xy", False)
    pipeline.set_param(rescale.id, "x_scale", 2.0)
    pipeline.set_param(rescale.id, "y_scale", 0.5)

    pipeline.run(
        None,
        source_payloads={
            "input": SourcePayload(data, image_state=input_state),
        },
    )

    output_state = pipeline.output_states[rescale.id]
    assert output_state is not None
    assert pipeline.outputs[rescale.id].shape == (4, 3, 20)
    assert output_state.axis_order == "YQX"
    assert output_state.axes[1] == input_state.axes[0]
    assert [axis.source_axis for axis in output_state.axes] == [5, 4, 6]
    assert [axis.confidence for axis in output_state.axes] == [
        AXIS_CONFIDENCE_EXPLICIT,
        AXIS_CONFIDENCE_INFERRED,
        AXIS_CONFIDENCE_EXPLICIT,
    ]


def test_rescale_axes_axis_preflight_matches_execution_after_inferred_yx():
    data = np.zeros((6, 7), dtype=np.float32)
    input_state = image_state_from_array(data)

    def workflow():
        pipeline = PrototypePipeline()
        pipeline.reset_empty_graph()
        rescale = pipeline.add_node("rescale_axes")
        calibrate = pipeline.add_node("set_pixel_size")
        assert pipeline.connect("input", rescale.id).success
        assert pipeline.connect(rescale.id, calibrate.id).success
        pipeline.set_param(rescale.id, "x_scale", 2.0)
        return pipeline, rescale.id, calibrate.id

    preflight, preflight_rescale, preflight_calibrate = workflow()
    preflight.preflight_axis_contract(
        {"input": SourcePayload(data, image_state=input_state)}
    )
    executed, executed_rescale, executed_calibrate = workflow()
    executed.run(
        None,
        source_payloads={
            "input": SourcePayload(data, image_state=input_state),
        },
    )

    projected_rescale = preflight.output_states[preflight_rescale]
    actual_rescale = executed.output_states[executed_rescale]
    projected_calibrate = preflight.output_states[preflight_calibrate]
    actual_calibrate = executed.output_states[executed_calibrate]
    assert projected_rescale is not None and actual_rescale is not None
    assert projected_calibrate is not None and actual_calibrate is not None
    assert projected_rescale.shape == actual_rescale.shape == (12, 14)
    assert projected_rescale.axes == actual_rescale.axes
    assert projected_rescale.history == actual_rescale.history
    assert projected_calibrate.axis_confidence == AXIS_CONFIDENCE_EXPLICIT
    assert actual_calibrate.axis_confidence == AXIS_CONFIDENCE_EXPLICIT


def test_channel_operations_reject_shape_only_rgb_guess():
    data = np.zeros((8, 9, 3), dtype=np.uint16)
    inferred, _inferred_id = _pipeline_with("split_channels")

    with pytest.raises(AmbiguousAxisError, match="explicit channel axis"):
        inferred.run(data)

    explicit_rgb, split_id = _pipeline_with("split_channels")
    explicit_rgb.run(data, input_metadata={"axes": "YXC"})

    assert len(explicit_rgb.node_outputs[split_id]) == 3
    assert all(output.shape == (8, 9) for output in explicit_rgb.node_outputs[split_id])


def test_explicit_volume_with_three_columns_is_not_treated_as_rgb():
    data = np.zeros((8, 9, 3), dtype=np.uint16)
    pipeline, _split_id = _pipeline_with("split_channels")

    with pytest.raises(AmbiguousAxisError, match="explicit channel axis"):
        pipeline.run(data, input_metadata={"axes": "ZYX"})


def test_explicit_numeric_channel_axis_bypasses_shape_axis_guess():
    data = np.zeros((3, 8, 9), dtype=np.uint16)
    data[0, 2:4, 2:4] = 100
    pipeline, composite_id = _pipeline_with("composite_to_rgb")
    pipeline.set_param(composite_id, "channel_axis_mode", "Manual")
    pipeline.set_param(composite_id, "channel_axis", 0)

    pipeline.run(data)

    assert pipeline.outputs[composite_id].shape == (8, 9, 3)
    state = pipeline.output_states[composite_id]
    assert state.axis_order == "Y,X,rgb"
    assert state.axis_confidence == AXIS_CONFIDENCE_MIXED
    assert not state.axes[0].is_explicit
    assert not state.axes[1].is_explicit
    assert state.axes[2].is_explicit


@pytest.mark.parametrize("width", [3, 4])
def test_composite_auto_does_not_promote_scalar_x_by_shape(width):
    data = np.arange(6 * width, dtype=np.float32).reshape(6, width)
    pipeline, _composite_id = _pipeline_with("composite_to_rgb")

    with pytest.raises(AmbiguousAxisError, match="explicit channel axis"):
        pipeline.run(data, input_metadata={"axes": "YX"})


def test_composite_auto_uses_explicit_nontrailing_channel_semantics():
    data = np.zeros((4, 3, 5), dtype=np.float32)
    data[:, 0, :] = 2.0
    data[:, 1, :] = 3.0
    data[:, 2, :] = 5.0
    pipeline, composite_id = _pipeline_with("composite_to_rgb")

    pipeline.run(data, input_metadata={"axes": "YCX"})

    output = pipeline.outputs[composite_id]
    assert output.shape == (4, 5, 3)
    assert np.all(output[..., 0] == 5.0)
    assert np.all(output[..., 1] == 3.0)
    assert np.all(output[..., 2] == 2.0)
    state = pipeline.output_states[composite_id]
    assert state.axis_order == "Y,X,rgb"
    assert [channel.name for channel in state.channels] == ["Red", "Green", "Blue"]
    assert "c axis (1)" in state.history[-1]
    assert "native intensity scale retained" in state.history[-1]
    assert "no normalization or clipping" in state.history[-1]


def test_composite_rejects_multiple_declared_channel_axes():
    data = np.zeros((2, 3, 4, 5), dtype=np.float32)
    pipeline, _composite_id = _pipeline_with("composite_to_rgb")

    with pytest.raises(AmbiguousAxisError, match="exactly one explicit channel"):
        pipeline.run(data, input_metadata={"axes": "CCYX"})


def test_composite_auto_ignores_saved_numeric_axis_and_uses_metadata():
    data = np.zeros((3, 4, 5), dtype=np.float32)
    data[2, 1, 1] = 7.0
    pipeline, composite_id = _pipeline_with("composite_to_rgb")
    pipeline.set_param(composite_id, "channel_axis", 1)
    pipeline.set_param(composite_id, "channel_axis_mode", COMPOSITE_RGB_AUTO)

    pipeline.run(data, input_metadata={"axes": "CYX"})

    assert pipeline.outputs[composite_id].shape == (4, 5, 3)
    assert pipeline.outputs[composite_id][1, 1, 0] == 7.0


def test_composite_manual_accepts_a_deliberate_nonchannel_axis():
    data = np.zeros((4, 5, 6), dtype=np.float32)
    data[0, 1, 1] = 2.0
    data[1, 1, 2] = 3.0
    data[2, 1, 3] = 5.0
    data[3, 1, 4] = 7.0
    pipeline, composite_id = _pipeline_with("composite_to_rgb")
    pipeline.set_param(composite_id, "channel_axis_mode", COMPOSITE_RGB_MANUAL)
    pipeline.set_param(composite_id, "channel_axis", 0)
    pipeline.set_param(composite_id, "mapping_mode", COMPOSITE_RGB_MANUAL)
    pipeline.set_param(
        composite_id,
        "channel_colors",
        "Red,Green,Unassigned,Blue",
    )

    pipeline.run(data, input_metadata={"axes": "ZYX"})

    output = pipeline.outputs[composite_id]
    assert output.shape == (5, 6, 3)
    assert output[1, 1].tolist() == [2.0, 0.0, 0.0]
    assert output[1, 2].tolist() == [0.0, 3.0, 0.0]
    assert output[1, 3].tolist() == [0.0, 0.0, 0.0]
    assert output[1, 4].tolist() == [0.0, 0.0, 7.0]
    assert "z axis (0)" in pipeline.output_states[composite_id].history[-1]


def test_composite_auto_ignores_stale_manual_assignments_without_metadata_colors():
    data = np.zeros((2, 4, 5), dtype=np.float32)
    data[0, 1, 1] = 2.0
    data[1, 1, 2] = 3.0
    pipeline, composite_id = _pipeline_with("composite_to_rgb")
    pipeline.set_param(composite_id, "channel_axis_mode", COMPOSITE_RGB_AUTO)
    pipeline.set_param(composite_id, "mapping_mode", COMPOSITE_RGB_AUTO)
    pipeline.set_param(
        composite_id,
        "channel_colors",
        "Unassigned,Unassigned",
    )

    pipeline.run(data, input_metadata={"axes": "CYX"})

    output = pipeline.outputs[composite_id]
    assert output[1, 1].tolist() == [0.0, 0.0, 2.0]
    assert output[1, 2].tolist() == [0.0, 3.0, 0.0]


def test_composite_manual_assignments_are_not_replaced_by_metadata_colors():
    data = np.zeros((2, 4, 5), dtype=np.float32)
    data[0, 1, 1] = 2.0
    data[1, 1, 2] = 30.0
    state = image_state_from_array(
        data,
        layer_metadata={"axes": "CYX"},
        channels=(
            ChannelMetadata(name="first", color=0xFF0000),
            ChannelMetadata(name="second", color=0x00FF00),
        ),
    )
    pipeline, composite_id = _pipeline_with("composite_to_rgb")
    pipeline.set_param(composite_id, "channel_axis_mode", COMPOSITE_RGB_AUTO)
    pipeline.set_param(composite_id, "mapping_mode", COMPOSITE_RGB_MANUAL)
    pipeline.set_param(composite_id, "channel_colors", "Blue,Unassigned")

    pipeline.run(
        data,
        source_payloads={"input": SourcePayload(data, image_state=state)},
    )

    output = pipeline.outputs[composite_id]
    assert output[1, 1].tolist() == [0.0, 0.0, 2.0]
    assert output[1, 2].tolist() == [0.0, 0.0, 0.0]
    history = pipeline.output_states[composite_id].history[-1]
    assert "manual additive colour table" in history
    assert "channel 1=Unassigned" in history


def test_composite_auto_preserves_order_only_for_declared_rgb_semantics():
    data = np.zeros((4, 5, 3), dtype=np.float32)
    data[..., 0] = 2.0
    data[..., 1] = 3.0
    data[..., 2] = 5.0

    encoded, encoded_id = _pipeline_with("composite_to_rgb")
    encoded.run(data, input_metadata={"axes": "Y,X,rgb"})
    fluorescence, fluorescence_id = _pipeline_with("composite_to_rgb")
    fluorescence.run(data, input_metadata={"axes": "YXC"})

    assert np.all(encoded.outputs[encoded_id][..., 0] == 2.0)
    assert np.all(encoded.outputs[encoded_id][..., 2] == 5.0)
    assert np.all(fluorescence.outputs[fluorescence_id][..., 0] == 5.0)
    assert np.all(fluorescence.outputs[fluorescence_id][..., 2] == 2.0)
    assert "declared encoded RGB order" in encoded.output_states[encoded_id].history[-1]


def test_composite_pipeline_runs_legacy_lossy_mode_only_when_selected():
    data = np.zeros((3, 4, 5), dtype=np.uint16)
    data[0] = 1000
    data[1] = 2000
    data[2] = 3000
    pipeline, composite_id = _pipeline_with("composite_to_rgb")
    pipeline.set_param(
        composite_id,
        "intensity_mapping",
        COMPOSITE_RGB_PERCENTILE_1_99,
    )

    pipeline.run(data, input_metadata={"axes": "CYX"})

    output = pipeline.outputs[composite_id]
    assert np.all(output == 1.0)
    history = pipeline.output_states[composite_id].history[-1]
    assert "1st-99th percentile normalization" in history
    assert "clipped to [0, 1]" in history
