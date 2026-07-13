from __future__ import annotations

import numpy as np
import pytest

from napari_vipp.core.metadata import (
    AXIS_CONFIDENCE_MIXED,
    AmbiguousAxisError,
)
from napari_vipp.core.pipeline import PrototypePipeline


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
