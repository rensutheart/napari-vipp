"""Median acceleration stays exact through shared scientific execution paths."""

from __future__ import annotations

import json
from dataclasses import replace
from importlib.metadata import version

import numpy as np
import pytest
from scipy import ndimage as ndi

from napari_vipp.core import simpleitk_filters as adapters
from napari_vipp.core.batch import run_batch
from napari_vipp.core.compute import ComputeMode, ComputeRequest
from napari_vipp.core.compute_cache import required_scientific_dependency_ids
from napari_vipp.core.compute_specs import compute_specs_for
from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
from napari_vipp.core.export import export_pipeline_to_python
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow

_CPU = ComputeRequest(mode=ComputeMode.CPU)
_LAYOUTS = (
    ("YX", (512, 512)),
    ("ZYX", (2, 512, 512)),
    ("CYX", (2, 512, 512)),
    ("TCZYX", (2, 2, 2, 512, 512)),
)


def _pipeline():
    graph = PrototypePipeline()
    graph.reset_empty_graph()
    median = graph.add_node("median_filter")
    graph.set_param(median.id, "size", 5)
    assert graph.connect("input", median.id).success
    return graph, median.id


def _image(shape, dtype):
    rng = np.random.default_rng(260922)
    image = rng.integers(0, 65_536, size=shape, dtype=np.uint16).astype(dtype)
    if np.dtype(dtype).kind == "f":
        image = image / np.dtype(dtype).type(64) - np.dtype(dtype).type(250)
    image.flags.writeable = False
    return image


def _reference(image):
    # Independent, pre-adapter CPU call: do not route the oracle through VIPP.
    return ndi.median_filter(image, size=(1,) * (image.ndim - 2) + (5, 5))


def _assert_exact(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


@pytest.fixture
def itk_calls(monkeypatch):
    calls = []
    implementation = adapters._simpleitk_median_filter

    def observed(array, *, size, xy_axes):
        calls.append((array.shape, array.dtype, size, xy_axes))
        return implementation(array, size=size, xy_axes=xy_axes)

    monkeypatch.setattr(adapters, "_simpleitk_median_filter", observed)
    return calls


@pytest.mark.parametrize("axes,shape", _LAYOUTS)
@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_graph_roundtrip_and_generated_python_match_original_scipy(
    axes,
    shape,
    dtype,
    itk_calls,
):
    graph, node_id = _pipeline()
    image = _image(shape, dtype)
    original = image.tobytes()
    expected = _reference(image)
    metadata = {"axes": axes}

    direct = graph.run(image, input_metadata=metadata)[node_id]
    _assert_exact(direct, expected)
    assert len(itk_calls) == 1
    assert not np.shares_memory(direct, image)
    assert graph.output_states[node_id].axes == graph.output_states["input"].axes

    document = serialize_workflow(graph, compute_request=_CPU)
    payload = deserialize_workflow(document)
    restored = PrototypePipeline()
    restored.restore_graph(payload["nodes"], payload["connections"])
    roundtrip = restored.run(image, input_metadata=metadata)[node_id]
    _assert_exact(roundtrip, expected)
    assert len(itk_calls) == 2

    namespace = {"__name__": "exported_pipeline"}
    exec(
        compile(
            export_pipeline_to_python(graph, compute_request=_CPU), "<exported>", "exec"
        ),
        namespace,
    )
    exported = namespace["run_pipeline"](image, input_metadata=metadata)
    _assert_exact(exported[node_id], expected)
    assert len(itk_calls) == 3
    assert exported.execution_report.cleanup_succeeded is True
    spec = next(
        spec
        for spec in compute_specs_for("median_filter")
        if spec.runtime_id == "cpu-numpy"
    )
    identity = exported.node_compute_provenance[node_id].actual_implementation
    assert identity.implementation_id == spec.implementation_id
    assert identity.implementation_version == spec.implementation_version
    assert image.tobytes() == original
    assert not image.flags.writeable


def test_shared_cpu_executor_preserves_calibration_and_current_provider(itk_calls):
    graph, node_id = _pipeline()
    image = _image((2, 2, 2, 512, 512), np.uint16)
    original = image.tobytes()
    axes = (
        AxisMetadata("t", "time", "second", 5, 10, source_axis=0),
        AxisMetadata("c", "channel", source_axis=1),
        AxisMetadata("z", "space", "micrometer", 2.0, 5.0, source_axis=2),
        AxisMetadata("y", "space", "micrometer", 0.3, 8.0, source_axis=3),
        AxisMetadata("x", "space", "micrometer", 0.2, -2.0, source_axis=4),
    )
    state = image_state_from_array(image, axes=axes)
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=1,
            workflow=serialize_workflow(graph, compute_request=_CPU),
            input_data=image,
            input_metadata={"vipp_image_state": state.to_dict()},
            input_name="median phantom",
            source_payloads={},
            compute_request=_CPU,
        )
    )

    assert result.error == ""
    assert result.pipeline is not None
    assert result.execution_report is not None
    assert len(itk_calls) == 1
    actual = result.pipeline.outputs[node_id]
    _assert_exact(actual, _reference(image))
    assert not np.shares_memory(actual, image)
    assert result.pipeline.output_states[node_id].axes == axes
    assert result.execution_report.cleanup_succeeded is True
    spec = next(
        spec
        for spec in compute_specs_for("median_filter")
        if spec.runtime_id == "cpu-numpy"
    )
    decision = next(
        item
        for item in result.execution_report.actual_decisions
        if item.node_id == node_id
    )
    assert decision.runtime_id == "cpu-numpy"
    assert decision.implementation_id == spec.implementation_id
    assert decision.implementation_version == spec.implementation_version
    assert "simpleitk" in required_scientific_dependency_ids(spec)
    assert image.tobytes() == original
    assert not image.flags.writeable


def test_batch_exports_exact_median_and_records_backend_dependency(tmp_path, itk_calls):
    from napari_vipp._tests.test_batch import _batch_config

    graph, node_id = _pipeline()
    output = graph.add_node("batch_output")
    graph.set_param(output.id, "tag", "median")
    graph.set_param(output.id, "format", "npy")
    assert graph.connect(node_id, output.id).success
    image = _image((512, 512), np.uint16)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    np.save(inputs / "phantom.npy", image)
    workflow = serialize_workflow(graph, compute_request=_CPU)
    config = replace(
        _batch_config(workflow, inputs, tmp_path / "outputs", (output.id,)),
        compute_request=_CPU,
    )

    result = run_batch(workflow, config)

    assert not result.has_failures
    assert len(itk_calls) == 1
    _assert_exact(np.load(result.saved_paths[0]), _reference(image))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["runtime"]["packages"]["simpleitk"] == version("SimpleITK")
    spec = next(
        spec
        for spec in compute_specs_for("median_filter")
        if spec.runtime_id == "cpu-numpy"
    )
    assert spec.implementation_id in json.dumps(manifest["items"][0]["execution"])
