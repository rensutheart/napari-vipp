"""Seeded propagation contracts across graph, metadata, export and batch."""

import json
from dataclasses import replace
from importlib.metadata import version

import numpy as np
import pytest

from napari_vipp.core.batch import run_batch
from napari_vipp.core.compute import ComputeMode, ComputeRequest
from napari_vipp.core.compute_cache import required_scientific_dependency_ids
from napari_vipp.core.compute_specs import compute_specs_for
from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
from napari_vipp.core.export import export_pipeline_to_python
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.pipeline import NODE_LIBRARY_BY_ID, PrototypePipeline
from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow


def _pipeline():
    graph = PrototypePipeline()
    graph.reset_empty_graph()
    seed_mask = graph.add_node("binary_threshold")
    graph.set_param(seed_mask.id, "threshold", 0.9)
    labels = graph.add_node("label_connected_components")
    graph.set_param(labels.id, "spatial_mode", "2D YX")
    mask = graph.add_node("binary_threshold")
    graph.set_param(mask.id, "threshold", 0.1)
    grow = graph.add_node("cellprofiler_propagation")
    for source, target, port in (
        ("input", seed_mask.id, 0), (seed_mask.id, labels.id, 0),
        ("input", mask.id, 0), ("input", grow.id, 0),
        (labels.id, grow.id, 1), (mask.id, grow.id, 2),
    ):
        assert graph.connect(source, target, target_port=port).success
    return graph, grow.id, labels.id, mask.id


def _image():
    data = np.full((13, 21), 0.3, dtype=np.float32)
    data[6, 4] = 1
    data[6, 16] = 1
    data[0] = 0
    data.setflags(write=False)
    return data


def test_registration_manual_cpu_and_backend_cache_identity():
    spec = NODE_LIBRARY_BY_ID["cellprofiler_propagation"]
    assert spec.title == "Grow Regions from Seeds — CellProfiler Propagation"
    assert spec.execution_policy == "manual"
    assert spec.output_type == "labels"
    assert [port.input_type for port in spec.inputs] == ["array", "labels", "mask"]
    implementations = compute_specs_for(spec.id)
    assert len(implementations) == 1
    assert implementations[0].runtime_id == "cpu-numpy"
    assert implementations[0].implementation_library_id == "centrosome"
    assert implementations[0].supported_spatial_ndims == (2,)
    assert "centrosome" in required_scientific_dependency_ids(implementations[0])


def test_graph_export_and_roundtrip_match_exact_reference():
    from centrosome.propagate import propagate

    graph, node_id, seeds_id, mask_id = _pipeline()
    data = _image()
    outputs = graph.run(data, input_metadata={"axes": "YX"})
    reference, _ = propagate(data, outputs[seeds_id], outputs[mask_id], 0.05)
    np.testing.assert_array_equal(outputs[node_id], reference)
    assert not np.shares_memory(outputs[node_id], data)
    state = graph.output_states[node_id]
    assert state.kind == "label image"
    assert state.axes == graph.output_states["input"].axes
    assert f"centrosome {version('centrosome')}" in state.history[-1]
    assert "regularization 0.05" in state.history[-1]
    # Branches creating the seeds and mask remain visible in carried history.
    assert any("0.9" in entry for entry in state.history)
    assert any("0.1" in entry for entry in state.history)
    restored = PrototypePipeline()
    payload = deserialize_workflow(serialize_workflow(graph))
    restored.restore_graph(payload["nodes"], payload["connections"])
    np.testing.assert_array_equal(restored.run(data)[node_id], reference)
    namespace = {"__name__": "exported_pipeline"}
    exec(compile(export_pipeline_to_python(graph), "<exported>", "exec"), namespace)
    np.testing.assert_array_equal(namespace["run_pipeline"](data)[node_id], reference)


def test_shared_executor_preserves_calibration_and_centrosome_provenance():
    graph, node_id, _, _ = _pipeline()
    data = _image()
    axes = (
        AxisMetadata("y", "space", "micrometer", 0.2, 4, source_axis=0),
        AxisMetadata("x", "space", "micrometer", 0.3, -3, source_axis=1),
    )
    state = image_state_from_array(data, axes=axes)
    result = execute_pipeline_request(PipelineRunRequest(
        run_id=1, workflow=serialize_workflow(graph), input_data=data,
        input_metadata={"vipp_image_state": state.to_dict()}, input_name="phantom",
        source_payloads={}, compute_request=ComputeRequest(mode=ComputeMode.CPU),
        manual_node_ids=frozenset({node_id}),
    ))
    assert result.error == ""
    assert result.pipeline.output_states[node_id].axes == axes
    assert result.pipeline.outputs[node_id].dtype == np.int32
    assert result.execution_report.cleanup_succeeded is True
    assert "centrosome" in str(result.execution_report.as_dict())


@pytest.mark.parametrize("axes,shape", [
    ("ZYX", (1, 13, 21)), ("TYX", (1, 13, 21)),
    ("CYX", (1, 13, 21)), ("QYX", (1, 13, 21)), ("XY", (13, 21)),
    ("TX", (13, 21)),
])
def test_never_silently_slices_stacks_or_interprets_non_yx(axes, shape):
    graph, _, _, _ = _pipeline()
    with pytest.raises(ValueError):
        graph.run(_image().reshape(shape), input_metadata={"axes": axes})


@pytest.mark.parametrize("input_index", [0, 1, 2])
def test_every_input_must_be_on_same_physical_grid(input_index):
    graph, node_id, _, _ = _pipeline()
    image = _image()
    axes = tuple(AxisMetadata(name, "space", "micrometer", 0.2, 0,
                              source_axis=index)
                 for index, name in enumerate("yx"))
    state = image_state_from_array(image, axes=axes)
    different = replace(state, axes=(replace(axes[0], translation=5), axes[1]))
    states = [state, state, state]
    states[input_index] = different
    with pytest.raises(ValueError, match="(grid|translation|origin|align)"):
        graph._validate_multi_input_grids(graph.nodes[node_id], states, {})


def test_actual_batch_outputs_and_manifest(tmp_path):
    from napari_vipp._tests.test_batch import _batch_config

    graph, node_id, _, _ = _pipeline()
    output = graph.add_node("batch_output")
    graph.set_param(output.id, "tag", "grown")
    graph.set_param(output.id, "format", "npy")
    assert graph.connect(node_id, output.id).success
    expected = graph.run(_image())[node_id]
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    np.save(inputs / "phantom.npy", _image())
    workflow = serialize_workflow(graph)
    config = _batch_config(workflow, inputs, tmp_path / "outputs", (output.id,))
    result = run_batch(workflow, config)
    assert not result.has_failures
    np.testing.assert_array_equal(np.load(result.saved_paths[0]), expected)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["runtime"]["packages"]["centrosome"] == version("centrosome")
    assert "centrosome" in json.dumps(manifest["items"][0]["execution"])


def test_palette_and_inspector_expose_seed_growth_controls(qtbot):
    from napari_vipp._tests.test_ui_inspector_widget_integration import _select, _widget
    from napari_vipp._tests.test_widget import _palette_item

    widget = _widget(qtbot, _image(), axes="YX")
    node = widget.add_node_from_palette("cellprofiler_propagation")
    _select(widget, node.id)
    assert _palette_item(widget, node.operation_id) is not None
    assert "regularization" in widget._parameter_widgets
    assert widget.pipeline.is_manual_node(node.id)
    assert [port.label for port in widget.pipeline.input_ports(node.id)] == [
        "Guidance image", "Seed labels", "Foreground mask",
    ]
