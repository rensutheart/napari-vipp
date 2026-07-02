from __future__ import annotations

from pathlib import Path

import numpy as np

from napari_vipp._sample_data import make_sample_data
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import load_workflow

EXAMPLE_WORKFLOW = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "otsu-red-channel-labels.json"
)
INTENSITY_EXAMPLE_WORKFLOW = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "red-channel-object-intensity-measurements.json"
)
MERGED_TABLE_EXAMPLE_WORKFLOW = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "red-channel-merged-measurement-table.json"
)
SUMMARY_EXAMPLE_WORKFLOW = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "synthetic-measurement-summary.json"
)
SKELETON_QC_EXAMPLE_WORKFLOW = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "synthetic-skeleton-qc.json"
)
ADVANCED_SKELETON_EXAMPLE_WORKFLOW = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "synthetic-advanced-skeleton-network.json"
)


def test_otsu_red_channel_label_workflow_loads_and_runs():
    workflow = load_workflow(EXAMPLE_WORKFLOW)
    pipeline = PrototypePipeline()
    pipeline.restore_graph(workflow["nodes"], workflow["connections"])

    split_ports = pipeline.output_ports("split_channels_1")
    assert [port.output_type for port in split_ports] == ["mask", "mask", "mask"]
    assert any(
        connection.source_id == "split_channels_1"
        and connection.source_port == 0
        and connection.target_id == "fill_holes_1"
        for connection in pipeline.connections
    )

    data, layer_kwargs, _layer_type = make_sample_data()[1]
    outputs = pipeline.run(
        data,
        input_metadata=layer_kwargs["metadata"],
        input_name=layer_kwargs["name"],
    )

    filled = outputs["fill_holes_1"]
    labels = outputs["label_connected_components_1"]
    cleared = outputs["clear_border_objects_1"]
    filtered = outputs["filter_labels_by_volume_1"]
    assert filled.shape == data.shape[1:]
    assert filled.dtype == bool
    assert labels.shape == filled.shape
    assert labels.dtype == np.int32
    assert labels.max() > 0
    assert cleared.shape == labels.shape
    assert cleared.dtype == np.int32
    assert np.count_nonzero(cleared) <= np.count_nonzero(labels)
    assert filtered.shape == labels.shape
    assert filtered.dtype == np.int32
    assert np.count_nonzero(filtered) <= np.count_nonzero(cleared)


def test_red_channel_intensity_measurement_workflow_loads_and_runs():
    workflow = load_workflow(INTENSITY_EXAMPLE_WORKFLOW)
    pipeline = PrototypePipeline()
    pipeline.restore_graph(workflow["nodes"], workflow["connections"])

    input_ports = pipeline.input_ports("measure_objects_intensity_1")
    assert [port.name for port in input_ports] == ["labels", "intensity"]
    assert any(
        connection.source_id == "filter_labels_by_volume_1"
        and connection.target_id == "measure_objects_intensity_1"
        and connection.target_port == 0
        for connection in pipeline.connections
    )
    assert any(
        connection.source_id == "split_channels_1"
        and connection.target_id == "measure_objects_intensity_1"
        and connection.target_port == 1
        for connection in pipeline.connections
    )

    data, layer_kwargs, _layer_type = make_sample_data()[1]
    outputs = pipeline.run(
        data,
        input_metadata=layer_kwargs["metadata"],
        input_name=layer_kwargs["name"],
    )
    table = outputs["measure_objects_intensity_1"]

    assert table.row_count > 0
    assert "volume_voxels" in table.columns
    assert "intensity_mean" in table.columns
    assert "intensity_sum" in table.columns


def test_red_channel_merged_measurement_table_workflow_loads_and_runs():
    workflow = load_workflow(MERGED_TABLE_EXAMPLE_WORKFLOW)
    pipeline = PrototypePipeline()
    pipeline.restore_graph(workflow["nodes"], workflow["connections"])

    input_ports = pipeline.input_ports("merge_tables_1")
    assert len(input_ports) == 2
    assert all(port.input_type == "table" for port in input_ports)
    assert any(
        connection.source_id == "measure_objects_1"
        and connection.target_id == "merge_tables_1"
        and connection.target_port == 0
        for connection in pipeline.connections
    )
    assert any(
        connection.source_id == "measure_objects_intensity_1"
        and connection.target_id == "merge_tables_1"
        and connection.target_port == 1
        for connection in pipeline.connections
    )

    data, layer_kwargs, _layer_type = make_sample_data()[1]
    outputs = pipeline.run(
        data,
        input_metadata=layer_kwargs["metadata"],
        input_name=layer_kwargs["name"],
    )
    table = outputs["add_metadata_columns_1"]

    assert table.row_count > 0
    assert "volume_voxels" in table.columns
    assert "intensity_mean" in table.columns
    assert "condition" in table.columns
    assert "replicate" in table.columns
    assert table.records()[0]["condition"] == "example"


def test_synthetic_measurement_summary_workflow_loads_and_runs():
    workflow = load_workflow(SUMMARY_EXAMPLE_WORKFLOW)
    pipeline = PrototypePipeline()
    pipeline.restore_graph(workflow["nodes"], workflow["connections"])

    data, layer_kwargs, _layer_type = next(
        sample
        for sample in make_sample_data()
        if sample[1]["name"] == "VIPP synthetic measurement summary"
    )
    outputs = pipeline.run(
        data,
        input_metadata=layer_kwargs["metadata"],
        input_name=layer_kwargs["name"],
    )
    measurements = outputs["measure_objects_1"]
    summary = outputs["summarize_measurements_1"]
    records = summary.records()

    assert measurements.row_count == 6
    assert summary.columns == (
        "condition",
        "replicate",
        "t_index",
        "row_count",
        "area_pixels_count",
        "area_pixels_mean",
        "area_pixels_min",
        "area_pixels_max",
    )
    assert summary.row_count == 3
    assert [record["row_count"] for record in records] == [2, 3, 1]
    assert [record["area_pixels_count"] for record in records] == [2, 3, 1]
    assert [record["area_pixels_mean"] for record in records] == [27.0, 20.0, 40.0]
    assert [record["area_pixels_min"] for record in records] == [24.0, 12.0, 40.0]
    assert [record["area_pixels_max"] for record in records] == [30.0, 28.0, 40.0]
    assert all(record["condition"] == "summary_validation" for record in records)


def test_synthetic_skeleton_qc_workflow_loads_and_runs():
    workflow = load_workflow(SKELETON_QC_EXAMPLE_WORKFLOW)
    pipeline = PrototypePipeline()
    pipeline.restore_graph(workflow["nodes"], workflow["connections"])

    keypoint_ports = pipeline.output_ports("skeleton_keypoints_1")
    assert [port.label for port in keypoint_ports] == [
        "Endpoints",
        "Junctions",
        "Isolated nodes",
    ]
    graph_table_ports = pipeline.output_ports("skeleton_graph_tables_1")
    assert [port.label for port in graph_table_ports] == [
        "Graph nodes",
        "Graph edges",
    ]

    data, layer_kwargs, _layer_type = next(
        sample
        for sample in make_sample_data()
        if sample[1]["name"] == "VIPP synthetic skeleton network"
    )
    outputs = pipeline.run(
        data,
        input_metadata=layer_kwargs["metadata"],
        input_name=layer_kwargs["name"],
    )

    keypoint_counts = [
        int(output.sum()) for output in pipeline.node_outputs["skeleton_keypoints_1"]
    ]
    before = outputs["analyze_skeleton_1"]
    after = outputs["analyze_skeleton_2"]
    graph_nodes, graph_edges = pipeline.node_outputs["skeleton_graph_tables_1"]
    summary = outputs["measure_overall_skeleton_network_1"]

    assert keypoint_counts == [8, 1, 1]
    assert outputs["label_skeleton_components_1"].max() == 3
    assert outputs["label_skeleton_branches_1"].max() == 7
    assert outputs["prune_skeleton_branches_1"].sum() == np.count_nonzero(data) - 2
    assert before.row_count == 3
    assert after.row_count == 2
    assert graph_nodes.row_count == 10
    assert graph_edges.row_count == 7
    assert summary.row_count == 1
    assert summary.records()[0]["component_count"] == 3
    assert summary.records()[0]["branch_count"] == 7
    assert sum(record["branch_count"] for record in before.records()) == 7
    assert sum(record["branch_count"] for record in after.records()) == 6


def test_synthetic_advanced_skeleton_workflow_loads_and_runs():
    workflow = load_workflow(ADVANCED_SKELETON_EXAMPLE_WORKFLOW)
    pipeline = PrototypePipeline()
    pipeline.restore_graph(workflow["nodes"], workflow["connections"])

    graph_table_ports = pipeline.output_ports("skeleton_graph_tables_1")
    assert [port.label for port in graph_table_ports] == [
        "Graph nodes",
        "Graph edges",
    ]

    data, layer_kwargs, _layer_type = next(
        sample
        for sample in make_sample_data()
        if sample[1]["name"] == "VIPP synthetic advanced skeleton network"
    )
    outputs = pipeline.run(
        data,
        input_metadata=layer_kwargs["metadata"],
        input_name=layer_kwargs["name"],
    )

    binary = outputs["binary_threshold_1"]
    keypoint_counts = [
        int(output.sum()) for output in pipeline.node_outputs["skeleton_keypoints_1"]
    ]
    graph_nodes, graph_edges = pipeline.node_outputs["skeleton_graph_tables_1"]
    branch_table = outputs["measure_skeleton_branches_1"]
    pruned_branch_table = outputs["measure_skeleton_branches_2"]
    summary = outputs["measure_overall_skeleton_network_1"]
    pruned_summary = outputs["measure_overall_skeleton_network_2"]
    summary_records = summary.records()
    pruned_records = pruned_summary.records()

    assert binary.shape == data.shape
    assert outputs["skeleton_graph_overlay_1"].shape == data.shape + (3,)
    assert outputs["skeleton_graph_overlay_2"].shape == data.shape + (3,)
    assert outputs["label_skeleton_components_1"].max() >= 6
    assert outputs["label_skeleton_branches_1"].max() >= 20
    assert all(count > 0 for count in keypoint_counts)
    assert summary.row_count == 2
    assert pruned_summary.row_count == 2
    assert "t_index" in summary.columns
    assert "skeleton_length_physical" in summary.columns
    assert graph_nodes.row_count > 30
    assert graph_edges.row_count > 20
    assert branch_table.row_count >= graph_edges.row_count
    assert pruned_branch_table.row_count <= branch_table.row_count
    assert sum(record["cycle_count"] for record in summary_records) >= 3
    assert sum(record["isolated_component_count"] for record in summary_records) >= 3
    assert sum(record["skeleton_voxel_count"] for record in pruned_records) < int(
        np.count_nonzero(binary)
    )
    assert all(record["physical_unit"] == "micrometer" for record in summary_records)
