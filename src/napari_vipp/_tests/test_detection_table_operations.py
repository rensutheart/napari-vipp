"""Annotation/selection preserve detection evidence only for unchanged fields."""

import numpy as np
import pytest

from napari_vipp.core.detection import find_peaks, template_match
from napari_vipp.core.measurement_collection import table_measurement_metadata
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.operations import add_metadata_columns, select_table_columns
from napari_vipp.core.tables import table_state_from_data


def _state(array):
    return image_state_from_array(
        array,
        axes=(
            AxisMetadata("y", "space", "um", 2),
            AxisMetadata("x", "space", "um", 0.5),
        ),
        source_name="source",
    )


def _detections():
    template = np.array([[0.0, 1.0], [3.0, 2.0]])
    search = np.zeros((10, 12))
    search[1:3, 1:3] = search[6:8, 8:10] = template
    score, mask, score_state, mask_state = template_match(
        search, template, search_state=_state(search), template_state=_state(template)
    )
    return find_peaks(
        score,
        mask,
        image_state=score_state,
        mask_state=mask_state,
        minimum_value=0.99,
        maximum_detections=1,
    )


@pytest.mark.parametrize("metadata_columns", ["", "condition=treated"])
def test_annotation_keeps_source_coordinates_cap_and_export_evidence(metadata_columns):
    table = _detections()
    result = add_metadata_columns(table, metadata_columns=metadata_columns)
    assert result.detection_metadata is table.detection_metadata
    assert result.detection_metadata.truncated
    assert result.detection_metadata.accepted_count == 2
    assert table_state_from_data(result).detection_metadata is table.detection_metadata
    assert table_measurement_metadata(result)["detection_metadata"]["truncated"] is True
    for column in table.columns:
        assert result.records()[0][column] == table.records()[0][column]


def test_column_reorder_and_unrelated_annotation_removal_keep_evidence():
    table = add_metadata_columns(_detections(), metadata_columns="condition=treated")
    result = select_table_columns(table, columns=",".join(reversed(table.columns[:-1])))
    assert result.columns == tuple(reversed(table.columns[:-1]))
    assert result.detection_metadata is table.detection_metadata
    assert select_table_columns(table).detection_metadata is table.detection_metadata
    assert result.records()[0] == {
        key: value for key, value in table.records()[0].items() if key != "condition"
    }


@pytest.mark.parametrize(
    "column",
    [
        "detection_id",
        "score",
        "y_index",
        "x_physical",
        "template_y_start",
        "template_x_stop",
    ],
)
def test_explicit_measurement_overwrite_clears_evidence_and_marks_table_type(column):
    table = _detections()
    original = table.records()
    result = add_metadata_columns(
        table, metadata_columns=f"{column}=99", overwrite="yes"
    )
    assert result.detection_metadata is None
    assert (
        "detection evidence cleared: protected fields overwritten" in result.table_kind
    )
    assert "detection_metadata" not in table_measurement_metadata(result)
    assert table.records() == original and table.detection_metadata is not None


@pytest.mark.parametrize(
    "column",
    [
        "detection_id",
        "score",
        "y_index",
        "x_physical",
        "template_y_start",
        "template_x_stop",
    ],
)
def test_explicit_measurement_removal_clears_evidence_and_marks_table_type(column):
    table = _detections()
    result = select_table_columns(
        table, columns=column, selection_mode="Drop listed columns"
    )
    assert result.detection_metadata is None
    assert "detection evidence cleared: protected fields removed" in result.table_kind
    assert column not in result.columns
    assert table.detection_metadata is not None


def test_overwriting_unrelated_annotation_preserves_detection_evidence():
    table = add_metadata_columns(_detections(), metadata_columns="condition=control")
    result = add_metadata_columns(
        table, metadata_columns="condition=treated", overwrite="yes"
    )
    assert result.detection_metadata is table.detection_metadata
    assert result.records()[0]["condition"] == "treated"
