"""The label-first example joins the same objects and preserves an isolate."""

from pathlib import Path

import numpy as np
import pytest

from napari_vipp._sample_data import _skeleton_network_sample
from napari_vipp.core.compute import ComputeRequest
from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
from napari_vipp.core.export import export_pipeline_to_python
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.result_plots import PlotData
from napari_vipp.core.workflow import load_workflow, serialize_workflow
from napari_vipp.ui.examples import _example_workflow_by_id, _example_workflow_path


def _pipeline():
    spec = _example_workflow_by_id("per-label-skeleton")
    assert spec is not None
    document = load_workflow(_example_workflow_path(spec))
    pipeline = PrototypePipeline()
    pipeline.restore_graph(
        document["nodes"], document["connections"], document["output_tunnels"]
    )
    return pipeline, document


def test_example_has_one_shared_object_population_and_no_file_writer():
    pipeline, document = _pipeline()
    root = Path(__file__).resolve().parents[3]
    filename = "synthetic-per-label-skeleton.json"
    assert (root / "examples" / filename).read_bytes() == (
        root / "src" / "napari_vipp" / "examples" / filename
    ).read_bytes()
    assert set(document["positions"]) == set(pipeline.nodes)
    assert len(document["notes"]) == 4
    assert {
        edge.target_id
        for edge in pipeline.connections
        if edge.source_id == "filtered_labels"
    } == {"morphology", "skeleton_labels", "skeleton_measurements"}
    assert pipeline.nodes["merged"].params["join_keys"] == "label_id"
    assert not any(
        node.operation_id in {"save_output", "batch_output", "table_source"}
        for node in pipeline.nodes.values()
    )


@pytest.mark.parametrize("minimum_volume, expected_ids", [(1, {1, 2, 3}), (2, {1, 2})])
def test_example_joins_calibrated_rows_and_filters_both_branches_together(
    minimum_volume, expected_ids, tmp_path, monkeypatch
):
    pipeline, _document = _pipeline()
    pipeline.nodes["filtered_labels"].params["min_volume"] = minimum_volume
    data, kwargs, _kind = _skeleton_network_sample()
    before = data.copy()
    data.setflags(write=False)
    monkeypatch.chdir(tmp_path)
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=1,
            workflow=serialize_workflow(pipeline),
            input_data=data,
            input_metadata=kwargs["metadata"],
            input_name=kwargs["name"],
            source_payloads={},
            compute_request=ComputeRequest(mode="cpu"),
            manual_node_ids=frozenset(pipeline.manual_node_ids()),
        ),
        raise_errors=True,
    )
    assert not result.error and not result.cancelled
    outputs = result.pipeline.outputs
    original = outputs["filtered_labels"]
    skeleton = outputs["skeleton_labels"]
    np.testing.assert_array_equal(skeleton[skeleton > 0], original[skeleton > 0])
    for node_id in ("morphology", "skeleton_measurements", "merged", "annotated"):
        records = outputs[node_id].records()
        assert len(records) == len(expected_ids)
        assert {row["label_id"] for row in records} == expected_ids
    rows = {row["label_id"]: row for row in outputs["merged"].records()}
    for row in rows.values():
        assert row["skeleton_status"] == "ok"
        assert row["skeleton_component_count"] == 1
        assert row["volume_physical"] == pytest.approx(row["volume_voxels"] * 0.45**3)
        assert row["skeleton_length_physical"] == pytest.approx(
            row["skeleton_length_voxels"] * 0.45
        )
    assert rows[2]["skeleton_length_voxels"] == 6
    assert rows[2]["endpoint_voxel_count"] == 2
    if 3 in expected_ids:
        assert rows[3]["isolated_node_count"] == 1
        assert rows[3]["skeleton_length_voxels"] == 0
        assert rows[3]["skeleton_voxel_count"] == 1
    components = result.pipeline.node_outputs["skeleton_measurements"][1]
    assert {row["label_id"] for row in components.records()} == expected_ids
    assert all(row["component_id"] == 1 for row in components.records())
    plot = outputs["plot_volume_length"]
    assert isinstance(plot, PlotData)
    assert plot.counts.plotted_points == len(expected_ids)
    assert plot.counts.excluded_rows == 0
    assert "micrometer" in plot.x_label and "micrometer" in plot.y_label
    np.testing.assert_array_equal(data, before)
    assert not list(tmp_path.iterdir())


def test_exported_example_keeps_the_label_join_and_isolate(tmp_path, monkeypatch):
    pipeline, _document = _pipeline()
    data, kwargs, _kind = _skeleton_network_sample()
    namespace = {"__name__": "per_label_skeleton_example"}
    exec(
        compile(export_pipeline_to_python(pipeline), "<per-label example>", "exec"),
        namespace,
    )
    monkeypatch.chdir(tmp_path)
    outputs = namespace["run_pipeline"](
        data, input_metadata=kwargs["metadata"], input_name=kwargs["name"]
    )
    rows = outputs["merged"].records()
    assert {row["label_id"] for row in rows} == {1, 2, 3}
    assert sum(row["isolated_node_count"] for row in rows) == 1
    assert outputs["plot_volume_length"].counts.plotted_points == 3
    assert {path.name for path in tmp_path.iterdir()} <= {".napari-vipp-test-state"}
