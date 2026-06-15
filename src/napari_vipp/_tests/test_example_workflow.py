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


def test_otsu_red_channel_label_workflow_loads_and_runs():
    workflow = load_workflow(EXAMPLE_WORKFLOW)
    pipeline = PrototypePipeline()
    pipeline.restore_graph(workflow["nodes"], workflow["connections"])

    split_ports = pipeline.output_ports("split_channels_1")
    assert [port.output_type for port in split_ports] == ["mask", "mask", "mask"]
    assert any(
        connection.source_id == "split_channels_1"
        and connection.source_port == 0
        and connection.target_id == "label_connected_components_1"
        for connection in pipeline.connections
    )

    data, layer_kwargs, _layer_type = make_sample_data()[1]
    outputs = pipeline.run(
        data,
        input_metadata=layer_kwargs["metadata"],
        input_name=layer_kwargs["name"],
    )

    labels = outputs["label_connected_components_1"]
    cleared = outputs["clear_border_objects_1"]
    filtered = outputs["filter_labels_by_volume_1"]
    assert labels.shape == data.shape[1:]
    assert labels.dtype == np.int32
    assert labels.max() > 0
    assert cleared.shape == labels.shape
    assert cleared.dtype == np.int32
    assert np.count_nonzero(cleared) <= np.count_nonzero(labels)
    assert filtered.shape == labels.shape
    assert filtered.dtype == np.int32
    assert np.count_nonzero(filtered) <= np.count_nonzero(cleared)
