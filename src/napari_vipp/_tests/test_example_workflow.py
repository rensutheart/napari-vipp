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
DERIVED_MORPHOLOGY_EXAMPLE_WORKFLOW = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "synthetic-derived-object-morphology.json"
)
MESH_MORPHOLOGY_EXAMPLE_WORKFLOW = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "synthetic-3d-mesh-morphology.json"
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
COLOCALIZATION_EXAMPLE_WORKFLOW = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "synthetic-colocalization-racc.json"
)
OBJECT_COLOCALIZATION_EXAMPLE_WORKFLOW = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "synthetic-object-colocalization-association.json"
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


def test_synthetic_derived_object_morphology_workflow_loads_and_runs():
    workflow = load_workflow(DERIVED_MORPHOLOGY_EXAMPLE_WORKFLOW)
    pipeline = PrototypePipeline()
    pipeline.restore_graph(workflow["nodes"], workflow["connections"])

    data, layer_kwargs, _layer_type = next(
        sample
        for sample in make_sample_data()
        if sample[1]["name"] == "VIPP synthetic object morphology"
    )
    outputs = pipeline.run(
        data,
        input_metadata=layer_kwargs["metadata"],
        input_name=layer_kwargs["name"],
    )
    measurements = outputs["measure_objects_1"]
    selected = outputs["select_table_columns_1"]
    measurement_records = measurements.records()

    assert outputs["label_connected_components_1"].max() == 4
    assert measurements.row_count == 4
    for column in (
        "axis_ratio_major_minor",
        "bbox_axis_0_length_pixels",
        "bbox_axis_1_length_pixels",
        "bbox_fill_fraction",
        "circularity",
        "perimeter_area_ratio",
        "hu_moment_0",
    ):
        assert column in measurements.columns
    assert selected.columns == (
        "label_id",
        "area_pixels",
        "circularity",
        "axis_ratio_major_minor",
        "bbox_fill_fraction",
        "perimeter_area_ratio",
        "hu_moment_0",
    )
    assert selected.row_count == measurements.row_count
    assert all(record["circularity"] > 0 for record in measurement_records)
    assert all(record["bbox_fill_fraction"] > 0 for record in measurement_records)


def test_synthetic_3d_mesh_morphology_workflow_loads_and_runs():
    workflow = load_workflow(MESH_MORPHOLOGY_EXAMPLE_WORKFLOW)
    pipeline = PrototypePipeline()
    pipeline.restore_graph(workflow["nodes"], workflow["connections"])

    data, layer_kwargs, _layer_type = next(
        sample
        for sample in make_sample_data()
        if sample[1]["name"] == "VIPP synthetic 3D mesh morphology"
    )
    outputs = pipeline.run(
        data,
        input_metadata=layer_kwargs["metadata"],
        input_name=layer_kwargs["name"],
    )
    labels = outputs["label_connected_components_1"]
    measurements = outputs["measure_objects_1"]
    mesh = outputs["measure_3d_mesh_morphology_1"]
    merged = outputs["merge_tables_1"]
    selected = outputs["select_table_columns_1"]
    mesh_records = mesh.records()
    ok_records = [
        record for record in mesh_records if record["mesh_status"] == "ok"
    ]

    assert labels.max() == 5
    assert measurements.row_count == 5
    assert mesh.row_count == 5
    assert merged.row_count == 5
    assert selected.row_count == 5
    assert "voxel_volume_physical" in mesh.columns
    assert "mesh_surface_area_physical" in mesh.columns
    assert "convex_hull_volume_physical" in mesh.columns
    assert any(
        record["mesh_status"] == "skipped_too_few_voxels"
        for record in mesh_records
    )
    assert len(ok_records) == 4
    assert all(record["mesh_volume_physical"] > 0 for record in ok_records)
    assert all(record["mesh_surface_area_physical"] > 0 for record in ok_records)
    assert max(record["sphericity"] for record in ok_records) > min(
        record["sphericity"] for record in ok_records
    )
    assert min(record["solidity_3d"] for record in ok_records) < 0.95
    assert selected.columns == (
        "label_id",
        "volume_voxels",
        "volume_physical",
        "voxel_volume_physical",
        "mesh_volume_physical",
        "mesh_surface_area_physical",
        "sphericity",
        "solidity_3d",
        "surface_area_to_volume",
        "mesh_status",
    )


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
    branch_summary = outputs["summarize_skeleton_branches_1"]

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
    assert branch_summary.row_count == 1
    assert branch_summary.records()[0]["branch_count"] == 8
    assert branch_summary.records()[0]["branch_type_endpoint_to_junction_count"] == 6
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
    branch_summary = outputs["summarize_skeleton_branches_1"]
    pruned_branch_summary = outputs["summarize_skeleton_branches_2"]
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
    assert branch_summary.row_count == 2
    assert pruned_branch_summary.row_count == 2
    assert "branch_length_physical_total" in branch_summary.columns
    assert sum(record["branch_count"] for record in branch_summary.records()) == (
        branch_table.row_count
    )
    assert sum(
        record["branch_count"] for record in pruned_branch_summary.records()
    ) == pruned_branch_table.row_count
    assert "branches_per_skeleton_length" in summary.columns
    assert "branches_per_physical_length" in summary.columns
    assert sum(record["cycle_count"] for record in summary_records) >= 3
    assert sum(record["isolated_component_count"] for record in summary_records) >= 3
    assert sum(record["skeleton_voxel_count"] for record in pruned_records) < int(
        np.count_nonzero(binary)
    )
    assert all(record["physical_unit"] == "micrometer" for record in summary_records)


def test_synthetic_colocalization_workflow_loads_and_runs():
    workflow = load_workflow(COLOCALIZATION_EXAMPLE_WORKFLOW)
    pipeline = PrototypePipeline()
    pipeline.restore_graph(workflow["nodes"], workflow["connections"])

    data, layer_kwargs, _layer_type = next(
        sample
        for sample in make_sample_data()
        if sample[1]["name"] == "VIPP synthetic colocalization"
    )
    outputs = pipeline.run(
        data,
        input_metadata=layer_kwargs["metadata"],
        input_name=layer_kwargs["name"],
    )

    overlay = outputs["colocalized_voxels_1"]
    metrics = outputs["colocalization_metrics_1"]
    racc = outputs["racc_index_1"]
    roi_mask = outputs["binary_threshold_1"]
    masked_overlay = outputs["masked_colocalized_voxels_1"]
    masked_metrics = outputs["masked_colocalization_metrics_1"]
    masked_racc = outputs["masked_racc_index_1"]
    record = metrics.records()[0]
    masked_record = masked_metrics.records()[0]

    assert (
        pipeline.input_ports("colocalization_metrics_1")[0].label
        == "Channel 1 image"
    )
    assert (
        pipeline.input_ports("masked_colocalization_metrics_1")[2].label
        == "ROI mask"
    )
    assert overlay.shape == data.shape[1:] + (3,)
    assert overlay.dtype == np.float32
    assert roi_mask.shape == data.shape[1:]
    assert roi_mask.dtype == bool
    assert masked_overlay.shape == data.shape[1:] + (3,)
    assert metrics.row_count == 1
    assert masked_metrics.row_count == 1
    assert record["colocalized_voxels"] > 0
    assert record["channel_1_positive_voxels"] > record["colocalized_voxels"]
    assert record["channel_2_positive_voxels"] > record["colocalized_voxels"]
    assert "manders_m1" in metrics.columns
    assert masked_record["mask_restricted"] is True
    assert masked_record["total_voxels"] == int(np.count_nonzero(roi_mask))
    assert masked_record["colocalized_voxels"] > 0
    assert racc.shape == data.shape[1:]
    assert racc.dtype == np.float32
    assert float(racc.max()) > 0.0
    assert masked_racc.shape == data.shape[1:]
    assert masked_racc.dtype == np.float32
    assert float(masked_racc.max()) > 0.0
    assert not np.isclose(
        pipeline.nodes["colocalization_metrics_1"].params["channel_1_threshold"],
        35,
    )
    assert not np.isclose(
        pipeline.nodes["masked_colocalization_metrics_1"].params[
            "channel_1_threshold"
        ],
        35,
    )


def test_synthetic_object_colocalization_workflow_loads_and_runs():
    workflow = load_workflow(OBJECT_COLOCALIZATION_EXAMPLE_WORKFLOW)
    pipeline = PrototypePipeline()
    pipeline.restore_graph(workflow["nodes"], workflow["connections"])

    data, layer_kwargs, _layer_type = next(
        sample
        for sample in make_sample_data()
        if sample[1]["name"] == "VIPP synthetic colocalization"
    )
    outputs = pipeline.run(
        data,
        input_metadata=layer_kwargs["metadata"],
        input_name=layer_kwargs["name"],
    )

    labels_1 = outputs["label_connected_components_1"]
    labels_2 = outputs["label_connected_components_2"]
    measurements = outputs["measure_objects_1"]
    object_metrics = outputs["object_colocalization_metrics_1"]
    overlaps = outputs["label_overlap_association_1"]
    distances = outputs["nearest_object_distance_1"]
    localization = outputs["event_localization_1"]
    merged = outputs["merge_tables_1"]

    assert labels_1.max() == 3
    assert labels_2.max() == 3
    assert measurements.row_count == 3
    assert object_metrics.table_kind == "per-object colocalization metrics"
    assert object_metrics.row_count == 3
    assert "manders_m1" in object_metrics.columns
    assert all(
        record["object_voxels"] > 0 for record in object_metrics.records()
    )
    assert overlaps.table_kind == "label overlap association"
    assert overlaps.row_count == 2
    assert all(record["overlap_voxels"] > 0 for record in overlaps.records())
    assert distances.table_kind == "nearest object distance"
    assert distances.row_count == 3
    assert "centroid_distance_physical" in distances.columns
    assert all(
        record["nearest_label_id"] > 0 for record in distances.records()
    )
    assert localization.table_kind == "event localization"
    assert localization.row_count == 3
    assert any(record["in_region"] is True for record in localization.records())
    assert any(record["in_region"] is False for record in localization.records())
    assert merged.row_count == measurements.row_count
    assert "volume_voxels" in merged.columns
    assert "colocalized_voxels" in merged.columns
