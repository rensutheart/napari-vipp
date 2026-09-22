from __future__ import annotations

import csv
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tifffile

from napari_vipp.core.batch import (
    BatchConfig,
    BatchOutputConfig,
    BatchSourceConfig,
    run_batch,
    scientific_workflow_hash,
)
from napari_vipp.core.compute import ComputeMode, ComputeRequest
from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
from napari_vipp.core.export import export_pipeline_to_python
from napari_vipp.core.metadata import (
    AmbiguousAxisError,
    AxisMetadata,
    image_state_from_array,
)
from napari_vipp.core.operations import analyze_skeleton, merge_tables
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.progress import ProgressContext
from napari_vipp.core.tables import TableData, TableState, table_from_columns
from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow


def _pipeline(*, supplied_skeleton: bool = False):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    labels = pipeline.add_node("label_connected_components")
    skeleton = pipeline.add_node("skeletonize_labels")
    analysis = pipeline.add_node("analyze_skeleton_per_label")
    assert pipeline.connect("input", threshold.id).success
    assert pipeline.connect(threshold.id, labels.id).success
    assert pipeline.connect(labels.id, skeleton.id).success
    assert pipeline.connect(labels.id, analysis.id, target_port=0).success
    if supplied_skeleton:
        assert pipeline.connect(skeleton.id, analysis.id, target_port=1).success
    return pipeline, labels, skeleton, analysis


def _mask():
    data = np.zeros((2, 9, 13), dtype=bool)
    data[0, 2, 1:5] = True
    data[0, 6, 9] = True
    data[1, 2, 1:7] = True
    data[1, 6, 9:12] = True
    data.flags.writeable = False
    return data


def _state(data, axes="TYX", *, name="cell", history=()):
    axis_types = {"T": "time", "C": "channel", "Q": "unknown", "P": "unknown"}
    state = image_state_from_array(
        data,
        axes=tuple(
            AxisMetadata(
                axis.lower(),
                axis_types.get(axis, "space"),
                scale=0.5 if axis in "XYZ" else 1.0,
                unit="micrometer" if axis in "XYZ" else "",
            )
            for axis in axes
        ),
        source_name=name,
    )
    assert state is not None
    return replace(state, history=history)


def _execute_call(pipeline, node, arrays, states):
    call = pipeline.prepare_node_call(node.id, tuple(arrays), tuple(states))
    assert call is not None
    output = call.cpu_function(call.positional_input(), **call.keyword_arguments())
    return pipeline.finalize_node_call(call, output)


@pytest.mark.parametrize("supplied_skeleton", [False, True])
def test_label_skeleton_optional_port_executes_and_keeps_both_tables(supplied_skeleton):
    pipeline, labels, skeleton, analysis = _pipeline(
        supplied_skeleton=supplied_skeleton
    )
    data = _mask()
    pipeline.run(data, input_metadata={"axes": "TYX"}, input_name="cell")

    assert pipeline.input_port_count(analysis.id) == 2
    assert [port.name for port in pipeline.output_ports(analysis.id)] == [
        "out",
        "components",
    ]
    summary, components = pipeline.node_outputs[analysis.id]
    assert isinstance(summary, TableData) and isinstance(components, TableData)
    assert summary.row_count == components.row_count == 4
    assert {(row["t_index"], row["label_id"]) for row in summary.records()} == {
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
    }
    assert [row["isolated_node_count"] for row in summary.records()] == [0, 1, 0, 0]
    assert [row["skeleton_component_count"] for row in summary.records()] == [1] * 4
    assert all(
        isinstance(state, TableState)
        for state in pipeline.node_output_states[analysis.id]
    )
    assert all(
        state.source_name == "cell"
        for state in pipeline.node_output_states[analysis.id]
    )
    assert pipeline.output_states[skeleton.id].kind == "label image"
    assert (
        pipeline.output_states[skeleton.id].axes
        == pipeline.output_states[labels.id].axes
    )
    assert not data.flags.writeable


def test_label_skeleton_only_optional_connection_is_not_executable():
    pipeline, _labels, skeleton, analysis = _pipeline()
    pipeline.connections = [
        connection
        for connection in pipeline.connections
        if connection.target_id != analysis.id
    ]
    assert pipeline.connect(skeleton.id, analysis.id, target_port=1).success
    assert pipeline.prepare_node_call(analysis.id) is None
    pipeline.run(_mask(), input_metadata={"axes": "TYX"})
    assert analysis.id not in pipeline.completed_node_ids
    assert pipeline.outputs.get(analysis.id) is None


def test_label_skeleton_computed_and_supplied_results_match():
    outputs = []
    for supplied in (False, True):
        pipeline, _labels, _skeleton, analysis = _pipeline(supplied_skeleton=supplied)
        pipeline.run(_mask(), input_metadata={"axes": "TYX"})
        outputs.append(pipeline.node_outputs[analysis.id])
    assert outputs[0] == outputs[1]


def test_supplied_skeleton_tables_keep_calibration_and_both_input_histories():
    pipeline, _labels, _skeleton, analysis = _pipeline(supplied_skeleton=True)
    labels = np.zeros((7, 10), dtype=np.uint16)
    labels[2, 1:5] = 42
    labels[5, 8] = 99
    labels.flags.writeable = False
    label_state = _state(labels, "YX", history=("Filtered original labels",))
    skeleton_state = _state(labels, "YX", history=("Supplied labeled skeleton",))
    results = _execute_call(
        pipeline, analysis, (labels, labels), (label_state, skeleton_state)
    )
    summary, components = (data for data, _state in results)
    assert [row["label_id"] for row in summary.records()] == [42, 99]
    assert summary.records()[0]["skeleton_length_physical"] == 1.5
    assert summary.records()[1]["isolated_node_count"] == 1
    assert components.records()[0]["label_id"] == 42
    for _data, state in results:
        assert state.source_name == "cell"
        assert "Filtered original labels" in state.history
        assert "Supplied labeled skeleton" in state.history
    assert not labels.flags.writeable


@pytest.mark.parametrize(
    "change", [{"scale": 2.0}, {"translation": 7.0}, {"unit": "pixel"}, {"name": "z"}]
)
@pytest.mark.parametrize("preflight_only", [False, True])
def test_label_skeleton_rejects_equal_shape_different_physical_grid(
    change, preflight_only
):
    pipeline, _labels, _skeleton, analysis = _pipeline(supplied_skeleton=True)
    labels = _mask().astype(np.uint16)
    original = _state(labels)
    altered = replace(
        original, axes=(*original.axes[:-1], replace(original.axes[-1], **change))
    )
    with pytest.raises(ValueError):
        pipeline.prepare_node_call(
            analysis.id,
            (labels, labels),
            (original, altered),
            axis_contract_only=preflight_only,
        )


@pytest.mark.parametrize(
    "operation", ["skeletonize_labels", "analyze_skeleton_per_label"]
)
def test_label_skeleton_rejects_ambiguous_qyx_volume(operation):
    pipeline, _labels, skeleton, analysis = _pipeline()
    node = skeleton if operation == "skeletonize_labels" else analysis
    labels = _mask().astype(np.uint16)
    with pytest.raises(AmbiguousAxisError, match="QYX|Q.*YX|ambiguous"):
        pipeline.prepare_node_call(node.id, (labels,), (_state(labels, "QYX"),))


@pytest.mark.parametrize(
    "operation", ["skeletonize_labels", "analyze_skeleton_per_label"]
)
def test_label_skeleton_uses_semantic_nontrailing_axes(operation):
    pipeline, _labels, skeleton, analysis = _pipeline()
    node = skeleton if operation == "skeletonize_labels" else analysis
    labels = _mask().astype(np.uint16)
    # Make two distinct, persistent object identities in each time frame.
    labels[:, 6, :] *= 42
    canonical = _execute_call(pipeline, node, (labels,), (_state(labels),))
    transposed = labels.transpose(2, 0, 1)
    reordered = _execute_call(
        pipeline, node, (transposed,), (_state(transposed, "XTY"),)
    )
    if operation == "skeletonize_labels":
        np.testing.assert_array_equal(
            reordered[0][0].transpose(1, 2, 0), canonical[0][0]
        )
        assert reordered[0][1].axis_order == "XTY"
    else:
        assert [table.records() for table, _state in reordered] == [
            table.records() for table, _state in canonical
        ]


def test_label_skeleton_merge_uses_label_and_all_leading_indices():
    pipeline, labels_node, _skeleton, analysis = _pipeline()
    morphology = pipeline.add_node("measure_objects")
    assert pipeline.connect(labels_node.id, morphology.id).success
    labels = np.zeros((2, 7, 10), dtype=np.uint16)
    labels[0, 2, 1:5] = 7
    labels[1, 2, 1:7] = 7
    state = _state(labels, "PYX")
    skeleton_table = _execute_call(pipeline, analysis, (labels,), (state,))[0][0]
    morphology_table = _execute_call(pipeline, morphology, (labels,), (state,))[0][0]
    merged = merge_tables((morphology_table, skeleton_table))
    assert merged.row_count == 2
    rows = merged.records()
    assert [
        (row["p_index"], row["label_id"], row["skeleton_voxel_count"]) for row in rows
    ] == [(0, 7, 4), (1, 7, 6)]


@pytest.mark.parametrize("units", [{}, {"fragmentation_index": "components/voxel"}])
def test_auto_merge_does_not_treat_fragmentation_measurement_as_identity(units):
    first = table_from_columns(
        {"label_id": [7], "fragmentation_index": [0.25]},
        column_units={"label_id": "label", **units},
    )
    second = table_from_columns(
        {"label_id": [7], "fragmentation_index": [0.5], "skeleton_voxel_count": [8]},
        column_units={"label_id": "label", **units},
    )
    merged = merge_tables((first, second))
    assert merged.records()[0]["skeleton_voxel_count"] == 8


@pytest.mark.parametrize("supplied_skeleton", [False, True])
def test_label_skeleton_roundtrip_and_export_preserve_connections_and_results(
    supplied_skeleton,
):
    pipeline, _labels, _skeleton, analysis = _pipeline(
        supplied_skeleton=supplied_skeleton
    )
    component_output = pipeline.add_node("batch_output")
    assert pipeline.connect(analysis.id, component_output.id, source_port=1).success
    document = serialize_workflow(pipeline)
    restored = deserialize_workflow(document)
    reopened = PrototypePipeline()
    reopened.restore_graph(
        restored["nodes"], restored["connections"], restored["output_tunnels"]
    )
    reopened.run(_mask(), input_metadata={"axes": "TYX"})
    expected = reopened.node_outputs[analysis.id]
    assert len(expected) == 2

    namespace = {"__name__": "exported_label_skeleton"}
    code = export_pipeline_to_python(reopened)
    exec(compile(code, "<exported-label-skeleton>", "exec"), namespace)
    outputs = namespace["run_pipeline"](_mask(), input_metadata={"axes": "TYX"})
    assert outputs[analysis.id] == expected[0]
    assert outputs[component_output.id] == expected[1]


@pytest.mark.parametrize("mode", [ComputeMode.CPU, ComputeMode.AUTO])
@pytest.mark.parametrize("supplied_skeleton", [False, True])
def test_label_skeleton_shared_executor_reports_progress_and_cpu_provenance(
    mode,
    supplied_skeleton,
):
    pipeline, _labels, skeleton, analysis = _pipeline(
        supplied_skeleton=supplied_skeleton
    )
    updates = []
    request = PipelineRunRequest(
        run_id=1,
        workflow=serialize_workflow(pipeline),
        input_data=_mask(),
        input_metadata={"axes": "TYX"},
        input_name="cell",
        source_payloads={},
        compute_request=ComputeRequest(mode=mode),
        manual_node_ids=frozenset({analysis.id}),
    )
    result = execute_pipeline_request(
        request, progress_callback=lambda *args: updates.append(args)
    )
    assert result.error == ""
    assert not result.cancelled
    assert result.pipeline is not None
    for node in (skeleton, analysis):
        assert node.id in result.pipeline.completed_node_ids
        assert any(update[0] == node.id for update in updates)
        assert (
            result.pipeline.node_compute_provenance[
                node.id
            ].actual_implementation.runtime_id
            == "cpu-numpy"
        )
    assert len(result.pipeline.node_outputs[analysis.id]) == 2


def test_label_skeleton_cancellation_does_not_publish_partial_tables():
    pipeline, _labels, _skeleton, analysis = _pipeline()
    cancel = threading.Event()
    finished = []

    def report(node_id, current, _total, _message):
        if node_id == analysis.id and current > 0:
            cancel.set()

    request = PipelineRunRequest(
        run_id=1,
        workflow=serialize_workflow(pipeline),
        input_data=_mask(),
        input_metadata={"axes": "TYX"},
        input_name="cell",
        source_payloads={},
        cancel_event=cancel,
        manual_node_ids=frozenset({analysis.id}),
    )
    result = execute_pipeline_request(
        request, progress_callback=report, node_finished_callback=finished.append
    )
    assert result.cancelled
    assert result.pipeline is None
    assert all(node.node_id != analysis.id for node in finished)


def test_label_skeleton_analysis_prepared_call_receives_progress_context():
    pipeline, _labels, _skeleton, analysis = _pipeline()
    data = _mask().astype(np.uint16)
    call = pipeline.prepare_node_call(analysis.id, (data,), (_state(data),))
    assert call is not None
    assert isinstance(call.kwargs["progress"], ProgressContext)


def test_existing_binary_skeleton_analysis_retains_independent_component_ids():
    data = np.zeros((7, 10), dtype=np.uint16)
    data[2, 1:5] = 42
    data[5, 8] = 99
    table = analyze_skeleton(data, spatial_mode="2D YX")
    assert "label_id" not in table.columns
    assert [row["component_id"] for row in table.records()] == [1, 2]


def test_supplied_label_skeleton_rejects_shape_mismatch():
    pipeline, _labels, _skeleton, analysis = _pipeline(supplied_skeleton=True)
    labels = _mask().astype(np.uint16)
    skeleton = labels[:, :, :-1]
    with pytest.raises(ValueError, match="shape"):
        _execute_call(
            pipeline, analysis, (labels, skeleton), (_state(labels), _state(skeleton))
        )


@pytest.mark.parametrize("supplied_skeleton", [False, True])
def test_label_skeleton_batch_exports_both_ports_with_cpu_provenance(
    tmp_path,
    supplied_skeleton,
):
    pipeline, _labels, _skeleton, analysis = _pipeline(
        supplied_skeleton=supplied_skeleton
    )
    outputs = []
    for port, tag in enumerate(("objects", "components")):
        output = pipeline.add_node("batch_output")
        assert pipeline.connect(analysis.id, output.id, source_port=port).success
        pipeline.set_param(output.id, "tag", tag)
        outputs.append(
            BatchOutputConfig(
                output.id,
                "Batch Output",
                tag,
                "table",
                "batch default",
                "",
                "{source_stem}__{tag}",
            )
        )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    tifffile.imwrite(
        inputs / "cell.ome.tif",
        _mask().astype(np.uint8),
        ome=True,
        photometric="minisblack",
        metadata={"axes": "TYX"},
    )
    document = serialize_workflow(pipeline)
    config = BatchConfig(
        workflow_file=Path("workflow.json"),
        workflow_sha256=scientific_workflow_hash(document),
        output_dir=tmp_path / "outputs",
        sources=(BatchSourceConfig("input", "Input", inputs, "*.ome.tif"),),
        outputs=tuple(outputs),
        save_python_script=False,
    )
    result = run_batch(document, config)
    assert result.summary["completed"] == 1
    assert len(result.saved_paths) == 2
    exported = {}
    for path in result.saved_paths:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        exported["components" if path.stem.endswith("components") else "objects"] = rows
        assert len(rows) == 4
        assert {(row["t_index"], row["label_id"]) for row in rows} == {
            ("0", "1"),
            ("0", "2"),
            ("1", "1"),
            ("1", "2"),
        }
    assert "skeleton_status" in exported["objects"][0]
    assert "component_id" in exported["components"][0]
    node_record = next(
        node
        for node in result.manifest.items[0].execution["nodes"]
        if node["node_id"] == analysis.id
    )
    assert node_record["actual_implementation"]["runtime_id"] == "cpu-numpy"
