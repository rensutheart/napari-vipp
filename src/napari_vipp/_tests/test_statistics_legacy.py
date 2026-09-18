"""Restored pre-version workflows retain the complete original summary result."""

from copy import deepcopy

import numpy as np
import pytest

from napari_vipp.core.operations import summarize_measurements
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow


@pytest.mark.parametrize("group_by", ["", "auto", "t_index"])
@pytest.mark.parametrize("explicit_version", [False, True])
def test_restored_legacy_summary_matches_direct_function(group_by, explicit_version):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    threshold.params["threshold"] = 5
    labels = pipeline.add_node("label_connected_components")
    measurements = pipeline.add_node("measure_objects")
    summary = pipeline.add_node("summarize_measurements")
    for source, target in (
        ("input", threshold.id),
        (threshold.id, labels.id),
        (labels.id, measurements.id),
        (measurements.id, summary.id),
    ):
        assert pipeline.connect(source, target).success

    document = serialize_workflow(pipeline)
    legacy_params = {
        "group_by": group_by,
        "value_columns": "area_pixels",
        "statistics": "count,mean,median,std,min,max,q25,q75",
    }
    saved_summary = next(node for node in document["nodes"] if node["id"] == summary.id)
    saved_summary["params"] = dict(legacy_params)
    if explicit_version:
        saved_summary["params"]["summary_version"] = 1
    original_document = deepcopy(document)
    restored_document = deserialize_workflow(document)
    restored = PrototypePipeline()
    restored.restore_graph(restored_document["nodes"], restored_document["connections"])
    image = np.zeros((2, 16, 16), dtype=np.float32)
    image[0, 1:3, 1:3] = image[0, 7:11, 7:11] = 10  # object areas 4 and 16
    image[1, 2:8, 2:8] = 10  # singleton group, area 36
    image.setflags(write=False)
    before = image.copy()
    restored.run(image, input_metadata={"axes": "TYX"})

    upstream_table = restored.outputs[measurements.id]
    expected = summarize_measurements(upstream_table, **legacy_params)
    actual = restored.outputs[summary.id]
    # Full equality covers columns/order, rows, units, names and source metadata.
    assert actual == expected
    assert actual.columns == (
        "t_index",
        "row_count",
        "area_pixels_count",
        "area_pixels_mean",
        "area_pixels_median",
        "area_pixels_std",
        "area_pixels_min",
        "area_pixels_max",
        "area_pixels_q25",
        "area_pixels_q75",
    )
    rows = actual.records()
    # Legacy empty group_by still means auto, never modern ungrouped output.
    assert [row["t_index"] for row in rows] == [0, 1]
    assert [row["area_pixels_count"] for row in rows] == [2, 1]
    assert [row["area_pixels_mean"] for row in rows] == [10, 36]
    assert rows[1]["area_pixels_std"] == 0.0
    assert restored.nodes[summary.id].params["summary_version"] == 1
    assert document == original_document
    np.testing.assert_array_equal(image, before)
    assert not image.flags.writeable
