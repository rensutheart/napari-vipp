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
