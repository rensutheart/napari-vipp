from __future__ import annotations

import numpy as np
import pytest

from napari_vipp.core.metadata import (
    AXIS_CONFIDENCE_MIXED,
    AmbiguousAxisError,
)
from napari_vipp.core.operations import COMPOSITE_RGB_PERCENTILE_1_99
from napari_vipp.core.pipeline import PrototypePipeline, SourcePayload


def _pipeline_with(operation_id: str) -> tuple[PrototypePipeline, str]:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    node = pipeline.add_node(operation_id)
    assert pipeline.connect("input", node.id).success
    return pipeline, node.id


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
    ("orthogonal_projection", "rescale_axes", "set_pixel_size"),
)
def test_named_spatial_role_operations_require_explicit_spatial_axes(operation_id):
    data = np.zeros((3, 8, 9), dtype=np.float32)
    inferred, _inferred_id = _pipeline_with(operation_id)

    with pytest.raises(AmbiguousAxisError, match="explicit spatial ax"):
        inferred.run(data)

    explicit_but_nonspatial, _nonspatial_id = _pipeline_with(operation_id)
    with pytest.raises(AmbiguousAxisError, match="explicit spatial ax"):
        explicit_but_nonspatial.run(data, input_metadata={"axes": "ABC"})


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


def test_composite_rejects_numeric_axis_that_conflicts_with_metadata():
    data = np.zeros((3, 4, 5), dtype=np.float32)
    pipeline, composite_id = _pipeline_with("composite_to_rgb")
    pipeline.set_param(composite_id, "channel_axis", 1)

    with pytest.raises(AmbiguousAxisError, match="conflicts.*declared channel axis"):
        pipeline.run(data, input_metadata={"axes": "CYX"})


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
