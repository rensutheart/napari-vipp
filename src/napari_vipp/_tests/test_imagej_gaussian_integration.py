"""ImageJ Gaussian graph, calibration, execution, workflow and export contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from napari_vipp.core.compute import ComputeMode, ComputeRequest
from napari_vipp.core.compute_specs import compute_specs_for
from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
from napari_vipp.core.export import export_pipeline_to_python
from napari_vipp.core.imagej_gaussian import imagej_gaussian_blur
from napari_vipp.core.metadata import (
    AcquisitionMetadata,
    AmbiguousAxisError,
    AxisMetadata,
    ChannelMetadata,
    image_state_from_array,
)
from napari_vipp.core.pipeline import NODE_LIBRARY_BY_ID, PrototypePipeline
from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow


def _pipeline(sigma=1.5):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    node = pipeline.add_node("imagej_gaussian_blur")
    pipeline.set_param(node.id, "sigma", sigma)
    assert pipeline.connect("input", node.id).success
    return pipeline, node.id


def _java_reference(dtype):
    path = Path(__file__).parent / "fixtures" / "imagej_gaussian_reference_v1.json"
    reference = json.loads(path.read_text(encoding="utf-8"))
    case = next(
        case for case in reference["cases"] if case["id"] == f"{dtype}_7x9_1.5"
    )
    source = np.asarray(case["input_values"], dtype=dtype).reshape(case["shape"])
    expected = np.asarray(case["output_values"], dtype=dtype).reshape(case["shape"])
    return source, expected


def _calibrated_state(data):
    axes = (
        AxisMetadata("c", "channel", source_axis=0),
        AxisMetadata("z", "space", "micrometer", 0.6, 2.0, source_axis=1),
        AxisMetadata("y", "space", "micrometer", 0.2, 3.0, source_axis=2),
        AxisMetadata("x", "space", "micrometer", 0.3, -4.0, source_axis=3),
    )
    return image_state_from_array(
        data,
        axes=axes,
        source_name="calibrated independent ImageJ phantom",
        history=("Acquisition calibrated before filtering",),
        channels=(ChannelMetadata(name="Hoechst"), ChannelMetadata(name="Control")),
        acquisition=AcquisitionMetadata(
            objective="60x", objective_na=1.4, instrument="Reference microscope"
        ),
    )


def _execute(pipeline, data, metadata):
    return execute_pipeline_request(
        PipelineRunRequest(
            run_id=1,
            workflow=serialize_workflow(pipeline),
            input_data=data,
            input_metadata=metadata,
            input_name="ImageJ phantom",
            source_payloads={},
            compute_request=ComputeRequest(mode=ComputeMode.CPU),
        )
    )


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float32"])
def test_shared_executor_matches_java_and_preserves_all_calibration_metadata(dtype):
    source, expected = _java_reference(dtype)
    data = np.broadcast_to(source, (2, 3, *source.shape)).copy()
    before = data.copy()
    data.setflags(write=False)
    input_state = _calibrated_state(data)
    pipeline, node_id = _pipeline()
    result = _execute(pipeline, data, {"vipp_image_state": input_state.to_dict()})

    assert result.error == ""
    assert result.pipeline is not None
    output = result.pipeline.outputs[node_id]
    np.testing.assert_array_equal(output, np.broadcast_to(expected, data.shape))
    assert output.dtype == data.dtype
    assert not np.shares_memory(output, data)
    np.testing.assert_array_equal(data, before)

    output_state = result.pipeline.output_states[node_id]
    assert output_state.axes == input_state.axes
    assert output_state.channels == input_state.channels
    assert output_state.acquisition == input_state.acquisition
    assert output_state.source == input_state.source
    upstream = result.pipeline.output_states["input"]
    assert output_state.source_name == upstream.source_name == "ImageJ phantom"
    assert output_state.history[:-1] == upstream.history
    method = output_state.history[-1]
    for detail in (
        "ImageJ", "1.54p", "sigma", "1.5", "YX", "X then Y", "nearest", "direct"
    ):
        assert detail in method

    assert result.execution_report is not None
    assert result.execution_report.cleanup_succeeded is True
    decision = next(
        decision
        for decision in result.execution_report.actual_decisions
        if decision.node_id == node_id
    )
    assert decision.operation_id == "imagej_gaussian_blur"
    assert decision.runtime_id == "cpu-numpy"
    assert decision.implementation_id == "cpu-imagej_gaussian_blur-v1"


@pytest.mark.parametrize("sigma", [0, 0.5, 1.5, 8.5])
def test_serialized_workflow_and_generated_python_share_executor_and_results(sigma):
    source, _expected = _java_reference("float32")
    data = np.stack((source, np.zeros_like(source), source))
    data.setflags(write=False)
    metadata = {"axes": "ZYX"}
    pipeline, node_id = _pipeline(sigma=sigma)
    expected = imagej_gaussian_blur(data, sigma=sigma)

    workflow = serialize_workflow(pipeline)
    payload = deserialize_workflow(json.loads(json.dumps(workflow)))
    restored = PrototypePipeline()
    restored.restore_graph(payload["nodes"], payload["connections"])
    result = _execute(restored, data, metadata)
    assert result.error == ""
    assert result.pipeline is not None
    assert restored.nodes[node_id].params["sigma"] == sigma
    np.testing.assert_array_equal(result.pipeline.outputs[node_id], expected)

    code = export_pipeline_to_python(
        restored, compute_request=ComputeRequest(mode=ComputeMode.CPU)
    )
    namespace = {"__name__": "exported_imagej_gaussian_pipeline"}
    exec(compile(code, "<exported ImageJ Gaussian>", "exec"), namespace)
    exported = namespace["run_pipeline"](
        data, input_metadata=metadata, input_name="ImageJ phantom"
    )
    np.testing.assert_array_equal(exported[node_id], expected)
    assert exported.output_states[node_id].to_dict() == (
        result.pipeline.output_states[node_id].to_dict()
    )
    assert exported.execution_report.cleanup_succeeded is True
    actual = next(
        item["actual_implementation"]
        for item in exported.execution_provenance["nodes"]
        if item["node_id"] == node_id
    )
    assert actual["identity_complete"] is True
    assert actual["implementation_id"] == "cpu-imagej_gaussian_blur-v1"


@pytest.mark.parametrize(
    "axes,shape",
    [("XY", (7, 9)), ("YZX", (7, 2, 9)), ("YXZ", (7, 9, 2)),
     ("YXC", (7, 9, 3)), ("ZYXC", (2, 7, 9, 3))],
)
def test_explicit_incompatible_axes_fail_before_silent_spatial_filtering(axes, shape):
    data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    before = data.copy()
    data.setflags(write=False)
    pipeline, node_id = _pipeline()
    with pytest.raises(AmbiguousAxisError, match="positional YX"):
        pipeline.run(data, input_metadata={"axes": axes})
    assert pipeline.outputs.get(node_id) is None
    np.testing.assert_array_equal(data, before)


def test_shared_executor_and_generated_python_retain_axis_failure():
    data = np.arange(7 * 9, dtype=np.uint16).reshape(7, 9)
    pipeline, node_id = _pipeline()
    result = _execute(pipeline, data, {"axes": "XY"})
    assert "positional YX" in result.error
    assert result.failure is not None
    assert result.failure.error_type == "AmbiguousAxisError"

    namespace = {"__name__": "exported_invalid_imagej_gaussian_pipeline"}
    exec(compile(export_pipeline_to_python(pipeline), "<exported>", "exec"), namespace)
    with pytest.raises(AmbiguousAxisError, match="positional YX"):
        namespace["run_pipeline"](data, input_metadata={"axes": "XY"})
    assert pipeline.outputs.get(node_id) is None


def test_declaration_is_cpu_only_and_limits_sigma_without_changing_native_gaussian():
    imagej_spec = NODE_LIBRARY_BY_ID["imagej_gaussian_blur"]
    native_spec = NODE_LIBRARY_BY_ID["gaussian_blur"]
    assert imagej_spec.function is imagej_gaussian_blur
    assert native_spec.function is not imagej_gaussian_blur
    implementations = compute_specs_for(imagej_spec.id)
    assert len(implementations) == 1
    assert implementations[0].runtime_id == "cpu-numpy"
    parameters = {parameter.name: parameter for parameter in imagej_spec.parameters}
    assert set(parameters) == {"sigma"}
    assert parameters["sigma"].default == 1.5
    assert parameters["sigma"].minimum == 0.0
    assert parameters["sigma"].maximum == 8.5
