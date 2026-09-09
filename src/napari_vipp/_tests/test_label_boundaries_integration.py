"""Label-boundary graph, scientific metadata, UI, and export contracts."""

import numpy as np
import pytest

from napari_vipp.core.compute import ComputeMode, ComputeRequest
from napari_vipp.core.compute_specs import compute_specs_for
from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
from napari_vipp.core.export import export_pipeline_to_python
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.operations import find_label_boundaries
from napari_vipp.core.pipeline import NODE_LIBRARY_BY_ID, PrototypePipeline
from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow
from napari_vipp.ui.inspector import (
    FILTER_RESULT_SECTION,
    MASK_SUMMARY_SECTION,
    inspector_profile,
)


def _pipeline(*, labels=True, mode="Auto from axes", placement="Inside objects"):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    assert pipeline.connect("input", threshold.id).success
    upstream = threshold.id
    if labels:
        labeler = pipeline.add_node("label_connected_components")
        pipeline.set_param(labeler.id, "spatial_mode", mode)
        assert pipeline.connect(upstream, labeler.id).success
        upstream = labeler.id
    boundaries = pipeline.add_node("find_label_boundaries")
    pipeline.set_param(boundaries.id, "spatial_mode", mode)
    pipeline.set_param(boundaries.id, "boundary_placement", placement)
    assert pipeline.connect(upstream, boundaries.id).success
    return pipeline, boundaries.id, upstream


def _image(shape):
    data = np.zeros(shape, dtype=np.float32)
    data[..., 2:6, 2:6] = 1
    data[..., 7:9, 7:9] = 1
    return data


@pytest.mark.parametrize("labels", [False, True])
@pytest.mark.parametrize(
    "axes,shape,ndim",
    [("YX", (10, 10), 2), ("ZYX", (3, 10, 10), 3),
     ("CYX", (2, 10, 10), 2), ("TZYX", (2, 3, 10, 10), 3),
     ("TCZYX", (2, 2, 3, 10, 10), 3)],
)
def test_auto_axes_shape_mask_kind_and_history(labels, axes, shape, ndim):
    pipeline, node_id, upstream = _pipeline(labels=labels)
    data = _image(shape)
    data.setflags(write=False)
    outputs = pipeline.run(data, input_metadata={"axes": axes})
    state = pipeline.output_states[node_id]
    assert state.axes == pipeline.output_states[upstream].axes
    assert state.kind == "binary mask"
    assert outputs[node_id].shape == data.shape
    assert outputs[node_id].dtype == bool
    assert not np.shares_memory(outputs[node_id], outputs[upstream])
    assert pipeline.nodes[node_id].params["resolved_spatial_ndim"] == ndim
    assert "Inside objects" in state.history[-1]
    assert "Face connected" in state.history[-1]
    assert "background label 0" in state.history[-1]
    assert "no exterior padding" in state.history[-1]
    assert f"{ndim}D" in state.history[-1]
    np.testing.assert_array_equal(
        outputs[node_id],
        find_label_boundaries(outputs[upstream], resolved_spatial_ndim=ndim),
    )


@pytest.mark.parametrize(
    "placement", ["Inside objects", "Outside objects", "Both sides"]
)
@pytest.mark.parametrize("mode", ["2D YX", "3D ZYX"])
def test_roundtrip_and_generated_python_match_current_graph(placement, mode):
    pipeline, node_id, _upstream = _pipeline(mode=mode, placement=placement)
    data = _image((3, 10, 10))
    expected = pipeline.run(data, input_metadata={"axes": "ZYX"})[node_id]
    payload = deserialize_workflow(serialize_workflow(pipeline))
    restored = PrototypePipeline()
    restored.restore_graph(payload["nodes"], payload["connections"])
    np.testing.assert_array_equal(
        restored.run(data, input_metadata={"axes": "ZYX"})[node_id], expected,
    )
    namespace = {"__name__": "exported_pipeline"}
    exec(compile(export_pipeline_to_python(pipeline), "<exported>", "exec"), namespace)
    exported = namespace["run_pipeline"](data, input_metadata={"axes": "ZYX"})
    np.testing.assert_array_equal(exported[node_id], expected)
    assert restored.nodes[node_id].params["boundary_placement"] == placement


def test_shared_cpu_executor_preserves_calibration_and_reports_provider():
    pipeline, node_id, _upstream = _pipeline()
    data = _image((3, 10, 10))
    axes = tuple(
        AxisMetadata(name, "space", "micrometer", scale, translation, source_axis=index)
        for index, (name, scale, translation) in enumerate(
            zip("zyx", (2.0, 0.3, 0.2), (5.0, 8.0, -2.0), strict=True)
        )
    )
    state = image_state_from_array(data, axes=axes)
    result = execute_pipeline_request(PipelineRunRequest(
        run_id=1, workflow=serialize_workflow(pipeline), input_data=data,
        input_metadata={"vipp_image_state": state.to_dict()},
        input_name="test", source_payloads={},
        compute_request=ComputeRequest(mode=ComputeMode.CPU),
    ))
    assert result.error == ""
    assert result.pipeline is not None
    assert result.pipeline.output_states[node_id].axes == axes
    assert result.pipeline.outputs[node_id].dtype == bool
    assert result.execution_report is not None
    assert result.execution_report.cleanup_succeeded is True


@pytest.mark.parametrize(
    "axes,shape,mode",
    [(None, (3, 10, 10), "Auto from axes"),
     ("CYX", (2, 10, 10), "3D ZYX"),
     ("ZCYX", (3, 2, 10, 10), "3D ZYX"),
     ("QYX", (3, 10, 10), "Auto from axes"),
     ("TCQYX", (2, 2, 3, 10, 10), "Auto from axes"),
     ("YZX", (10, 3, 10), "3D ZYX")],
)
def test_ambiguous_or_incompatible_axes_fail_instead_of_mixing_channels(
    axes, shape, mode,
):
    pipeline, _node_id, _upstream = _pipeline(labels=False, mode=mode)
    with pytest.raises(ValueError):
        pipeline.run(_image(shape), input_metadata={"axes": axes} if axes else None)


def test_unknown_stack_axis_requires_a_deliberate_spatial_choice():
    pipeline, node_id, upstream = _pipeline(labels=False, mode="2D YX")
    outputs = pipeline.run(_image((3, 10, 10)), input_metadata={"axes": "QYX"})
    np.testing.assert_array_equal(
        outputs[node_id],
        find_label_boundaries(outputs[upstream], spatial_mode="2D YX"),
    )


def test_declaration_provider_and_inspector_semantics():
    spec = NODE_LIBRARY_BY_ID["find_label_boundaries"]
    assert spec.title == "Find Label Boundaries"
    assert spec.category == "Label Operations"
    assert spec.input_type == "mask_or_labels"
    assert spec.output_type == "mask"
    assert not spec.preserves_input_type
    assert len(compute_specs_for(spec.id)) == 1
    assert compute_specs_for(spec.id)[0].runtime_id == "cpu-numpy"
    profile = inspector_profile(spec)
    assert MASK_SUMMARY_SECTION in profile.primary_sections
    assert FILTER_RESULT_SECTION not in profile.section_order


def test_boundary_mask_connects_to_mask_consumers_not_label_only_consumers():
    pipeline, node_id, _upstream = _pipeline()
    dilation = pipeline.add_node("dilate")
    label_filter = pipeline.add_node("filter_labels_by_volume")
    assert pipeline.connect(node_id, dilation.id).success
    assert not pipeline.connect(node_id, label_filter.id).success


def test_inspector_exposes_placement_and_connectivity(qtbot):
    from napari_vipp._tests.test_ui_inspector_widget_integration import (
        _publish_array_output,
        _select,
        _widget,
    )
    from napari_vipp._tests.test_widget import _palette_item

    widget = _widget(qtbot, _image((3, 10, 10)), axes="ZYX")
    node = widget.add_node_from_palette("find_label_boundaries")
    widget._connect_nodes("threshold", node.id)
    _publish_array_output(widget, "threshold", _image((3, 10, 10)) > 0, axes="ZYX")
    _select(widget, node.id)
    assert _palette_item(widget, node.operation_id) is not None
    for name, expected in (
        ("boundary_placement", ["Inside objects", "Outside objects", "Both sides"]),
        ("connectivity", ["Face connected", "Full connectivity"]),
    ):
        control = widget._parameter_widgets[name]
        combo = getattr(control, "combo", control)
        assert [combo.itemText(i) for i in range(combo.count())] == expected
        assert combo.toolTip() or control.toolTip()
