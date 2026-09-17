"""Native graph contracts for the explicit CellProfiler compartment stages."""

from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest

from napari_vipp.core import metadata as metadata_module
from napari_vipp.core.cellprofiler_contracts import (
    CELLPROFILER_COMPARTMENT_OPERATION_IDS,
)
from napari_vipp.core.compute import ComputeMode, ComputeRequest
from napari_vipp.core.compute_cache import required_scientific_dependency_ids
from napari_vipp.core.compute_specs import compute_specs_for
from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
from napari_vipp.core.export import export_pipeline_to_python
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.pipeline import (
    NODE_LIBRARY_BY_ID,
    PrototypePipeline,
    SourcePayload,
)
from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow


@pytest.fixture
def reference():
    """Outputs were produced by the official CP4.2.6 executable, not VIPP."""
    path = (
        Path(__file__).parent
        / "fixtures/cellprofiler_compartments/cellprofiler-4.2.6-fixture.npz"
    )
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    for array in result.values():
        array.setflags(write=False)
    return result


def _workflow():
    graph = PrototypePipeline()
    graph.reset_empty_graph()
    ids = {"DNA": "input", "Actin": graph.add_node("input").id}

    def add(role, operation, source=None):
        ids[role] = graph.add_node(operation).id
        if source is not None:
            wire(source, role)

    def wire(source, target, source_port=0, target_port=0):
        result = graph.connect(
            ids[source], ids[target], source_port=source_port, target_port=target_port
        )
        assert result.success, result.message

    add("SmoothedDNA", "cellprofiler_smooth", "DNA")
    add("SmoothedActin", "cellprofiler_smooth", "Actin")
    add("Nuclei", "cellprofiler_primary_objects", "SmoothedDNA")
    add("Mask", "cellprofiler_threshold", "SmoothedActin")
    add("Seeds", "cellprofiler_propagation_seeds")
    wire("Nuclei", "Seeds", source_port=1)
    wire("Nuclei", "Seeds", target_port=1)
    add("Grown", "cellprofiler_propagation")
    wire("SmoothedActin", "Grown")
    wire("Seeds", "Grown", target_port=1)
    wire("Mask", "Grown", target_port=2)
    add("Cells", "cellprofiler_finish_cells", "Grown")
    wire("Nuclei", "Cells", target_port=1)
    add("Cytoplasm", "cellprofiler_cytoplasm", "Cells")
    wire("Nuclei", "Cytoplasm", target_port=1)
    return graph, ids


def _sources(ids, reference):
    axes = (
        AxisMetadata("y", "space", "micrometer", 0.2, 4, source_axis=0),
        AxisMetadata("x", "space", "micrometer", 0.3, -3, source_axis=1),
    )
    return {
        ids[key]: SourcePayload(
            reference[key],
            image_state=image_state_from_array(reference[key], axes=axes),
        )
        for key in ("DNA", "Actin")
    }


def _assert_reference(graph, ids, reference):
    for role, reference_key in {
        "SmoothedDNA": "SmoothedDNA",
        "SmoothedActin": "SmoothedActin",
        "Nuclei": "Nuclei_segmented",
        "Mask": "Cells_threshold_mask_replayed",
        "Cells": "Cells_segmented",
        "Cytoplasm": "Cytoplasm_segmented",
    }.items():
        np.testing.assert_array_equal(
            graph.node_outputs[ids[role]][0], reference[reference_key]
        )
    np.testing.assert_array_equal(
        graph.node_outputs[ids["Nuclei"]][1], reference["Nuclei_unedited_segmented"]
    )


@pytest.mark.parametrize("operation", sorted(CELLPROFILER_COMPARTMENT_OPERATION_IDS))
def test_compartment_registry_is_explicit_cpu_2d_with_centrosome_cache_dependency(
    operation,
):
    spec = NODE_LIBRARY_BY_ID[operation]
    implementations = compute_specs_for(operation)
    assert len(implementations) == 1
    implementation = implementations[0]
    assert implementation.runtime_id == "cpu-numpy"
    assert implementation.implementation_library_id == "centrosome"
    assert implementation.supported_spatial_ndims == (2,)
    assert "centrosome" in required_scientific_dependency_ids(implementation)
    assert "2D YX" in spec.stack_processing_note
    if operation == "cellprofiler_primary_objects":
        assert spec.execution_policy == "manual"
        assert [port.name for port in spec.output_ports] == ["retained", "unedited"]


@pytest.mark.parametrize("operation", sorted(CELLPROFILER_COMPARTMENT_OPERATION_IDS))
def test_compartment_history_records_scientific_settings_without_runtime_objects(
    operation,
):
    class RuntimeOnly:
        def __str__(self):
            raise AssertionError("Runtime-only objects must not enter image history")

        __repr__ = __str__

    spec = NODE_LIBRARY_BY_ID[operation]
    params = {parameter.name: parameter.default for parameter in spec.parameters}
    state = image_state_from_array(np.zeros((5, 7), dtype=np.float32))
    assert state is not None
    expected = metadata_module._operation_history(state, operation, spec.title, params)
    for name, value in params.items():
        assert f"{name}={value}" in expected
    for _ in range(2):
        runtime_params = {
            **params,
            "context": RuntimeOnly(),
            "inputs": RuntimeOnly(),
            "progress_callback": RuntimeOnly(),
            "cancel_callback": RuntimeOnly(),
            "_vipp_internal": RuntimeOnly(),
        }
        actual = metadata_module._operation_history(
            state, operation, spec.title, runtime_params
        )
        assert actual == expected
        if len(spec.inputs) > 1:
            actual_multi_input = metadata_module._multi_input_history(
                [state] * len(spec.inputs), operation, spec.title, runtime_params
            )
            assert actual_multi_input == expected


def test_native_graph_matches_official_cp_fixture_preserves_sources_and_port_metadata(
    reference,
):
    graph, ids = _workflow()
    sources = _sources(ids, reference)
    originals = {key: reference[key].copy() for key in ("DNA", "Actin")}
    graph.run(None, source_payloads=sources)
    _assert_reference(graph, ids, reference)
    for key, before in originals.items():
        np.testing.assert_array_equal(reference[key], before)
        assert not reference[key].flags.writeable
    primary_states = graph.node_output_states[ids["Nuclei"]]
    assert len(primary_states) == 2
    for state, label in zip(
        primary_states, ["Retained nuclei", "Before filtering"], strict=True
    ):
        assert state.axes == sources["input"].image_state.axes
        assert state.kind == "label image"
        assert np.dtype(state.dtype) == np.dtype(np.int32)
        assert label in state.history[-1]
        assert "CellProfiler 4.2.6" in state.history[-1]
        assert f"centrosome {version('centrosome')}" in state.history[-1]
    assert not np.shares_memory(*graph.node_outputs[ids["Nuclei"]])
    for role in ["Seeds", "Cells", "Cytoplasm"]:
        state = graph.output_states[ids[role]]
        assert state.axes == sources["input"].image_state.axes
        assert "pixel units" in state.history[-1]
        assert any("Before filtering" in item for item in state.history)


def test_primary_two_port_preflight_matches_execution_without_running_kernels(
    reference, monkeypatch
):
    graph, ids = _workflow()
    sources = _sources(ids, reference)
    graph.run(None, source_payloads=sources)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("Axis preflight must not execute a pixel kernel")

    for operation in CELLPROFILER_COMPARTMENT_OPERATION_IDS | {
        "cellprofiler_propagation"
    }:
        spec = NODE_LIBRARY_BY_ID[operation]
        monkeypatch.setitem(
            NODE_LIBRARY_BY_ID, operation, replace(spec, function=fail_if_called)
        )
    preflight, preflight_ids = _workflow()
    preflight.preflight_axis_contract(_sources(preflight_ids, reference))
    for role in ["Nuclei", "Seeds", "Cells", "Cytoplasm"]:
        actual = graph.node_output_states[ids[role]]
        predicted = preflight.node_output_states[preflight_ids[role]]
        assert len(actual) == len(predicted)
        for left, right in zip(actual, predicted, strict=True):
            assert (left.shape, left.dtype, left.axes, left.kind) == (
                right.shape,
                right.dtype,
                right.axes,
                right.kind,
            )


def test_shared_executor_and_export_replay_official_fixture(reference):
    graph, ids = _workflow()
    sources = _sources(ids, reference)
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=1,
            workflow=serialize_workflow(graph),
            input_data=None,
            input_metadata={},
            input_name="official synthetic fixture",
            source_payloads=sources,
            compute_request=ComputeRequest(mode=ComputeMode.CPU),
            manual_node_ids=graph.manual_node_ids(),
        )
    )
    assert result.error == ""
    _assert_reference(result.pipeline, ids, reference)
    assert result.execution_report.cleanup_succeeded
    assert "centrosome" in str(result.execution_report.as_dict())
    namespace = {"__name__": "exported_compartment_workflow"}
    exec(
        compile(export_pipeline_to_python(graph), "<exported compartments>", "exec"),
        namespace,
    )
    exported = namespace["run_pipeline"](
        source_payloads=sources, compute_request=ComputeRequest(mode=ComputeMode.CPU)
    )
    for role, key in [
        ("Nuclei", "Nuclei_segmented"),
        ("Cells", "Cells_segmented"),
        ("Cytoplasm", "Cytoplasm_segmented"),
    ]:
        np.testing.assert_array_equal(exported[ids[role]], reference[key])
    document = deserialize_workflow(serialize_workflow(graph))
    restored = PrototypePipeline()
    restored.restore_graph(document["nodes"], document["connections"])
    restored.run(None, source_payloads=sources)
    _assert_reference(restored, ids, reference)


@pytest.mark.parametrize("operation", sorted(CELLPROFILER_COMPARTMENT_OPERATION_IDS))
@pytest.mark.parametrize(
    "axes,shape",
    [("TX", (11, 12)), ("XY", (11, 12)), ("ZYX", (1, 11, 12)), ("TYX", (1, 11, 12))],
)
def test_each_compartment_node_rejects_wrong_axes_and_3d(
    operation, axes, shape, monkeypatch
):
    if NODE_LIBRARY_BY_ID[operation].input_type == "labels":
        # Isolate this node's axis guard using an explicitly typed label source;
        # an upstream segmentation guard must not be the one rejecting the axes.
        monkeypatch.setitem(
            NODE_LIBRARY_BY_ID,
            "input",
            replace(NODE_LIBRARY_BY_ID["input"], output_type="labels"),
        )
    graph = PrototypePipeline()
    graph.reset_empty_graph()
    node = graph.add_node(operation)
    for port in range(len(NODE_LIBRARY_BY_ID[operation].inputs) or 1):
        assert graph.connect("input", node.id, target_port=port).success
    data = np.zeros(
        shape,
        np.float32
        if operation
        in {
            "cellprofiler_smooth",
            "cellprofiler_threshold",
            "cellprofiler_primary_objects",
        }
        else np.int32,
    )
    source = SourcePayload(data, metadata={"axes": axes})
    with pytest.raises(ValueError, match="(YX|2D|axes|axis)"):
        graph.preflight_axis_contract({"input": source})
    with pytest.raises(ValueError, match="(YX|2D|axes|axis)"):
        graph.run(None, source_payloads={"input": source})


@pytest.mark.parametrize(
    "operation",
    [
        "cellprofiler_propagation_seeds",
        "cellprofiler_finish_cells",
        "cellprofiler_cytoplasm",
    ],
)
@pytest.mark.parametrize("port", [0, 1])
@pytest.mark.parametrize(
    "attribute,value",
    [("scale", 0.5), ("translation", 3.0), ("unit", "nanometer"), ("name", "z")],
)
def test_both_label_inputs_require_equal_physical_grid(
    operation, port, attribute, value
):
    graph = PrototypePipeline()
    graph.reset_empty_graph()
    node = graph.add_node(operation)
    axes = (
        AxisMetadata("y", "space", "micrometer", 0.2, 0, source_axis=0),
        AxisMetadata("x", "space", "micrometer", 0.2, 0, source_axis=1),
    )
    state = image_state_from_array(np.zeros((11, 12), np.int32), axes=axes)
    states = [state, state]
    states[port] = replace(
        state, axes=(replace(axes[0], **{attribute: value}), axes[1])
    )
    with pytest.raises(
        ValueError, match="(grid|align|axis|axes|spatial|scale|translation|unit)"
    ):
        graph._validate_multi_input_grids(node, states, {})
