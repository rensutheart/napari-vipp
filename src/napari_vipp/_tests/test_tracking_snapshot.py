"""Collection snapshots validate each source's retained scientific cells."""

from copy import deepcopy
from dataclasses import replace

import pytest

from napari_vipp._tests.test_measurement_collection import _manifest
from napari_vipp._tests.test_tracking import series
from napari_vipp.core import tracking
from napari_vipp.core.batch_resume import seal_document
from napari_vipp.core.measurement_collection import (
    CollectionLimits,
    MeasurementCollectionError,
    _load_snapshot,
    _snapshot_cell,
    _snapshot_document,
    collect_measurements,
    inspect_collection,
)
from napari_vipp.core.progress import OperationCancelled
from napari_vipp.core.tracking import build_tracks
from napari_vipp.core.tracking_metadata import FramePopulation


def _collection(tmp_path, kind, *, empty=False, exclude_second=False):
    tables = []
    for identity in ("a", "b"):
        observations = series(
            [] if empty else [(0, 1, 1, 1), (2, 1, 1, 2)],
            frame_count=4,
            source_frame=f"source-{identity}",
            source_revision=identity * 64,
        )
        if kind == "capped":
            populations = list(observations.observation_metadata.frame_populations)
            populations[0] = FramePopulation(0, 2, 0 if empty else 1, True)
            observations = replace(
                observations,
                observation_metadata=replace(
                    observations.observation_metadata,
                    frame_populations=tuple(populations),
                ),
            )
        if kind in {"tracks", "summary"}:
            table = build_tracks(observations, maximum_gap=1)[kind == "summary"]
        else:
            table = observations
        tables.append(table)
    preview = inspect_collection(_manifest(tmp_path, tables), "batch_output_1")
    return collect_measurements(
        preview,
        included_ids=["item-1"] if exclude_second else None,
        reviewed_exclusions=exclude_second,
        annotations={"item-1": {"condition": "control"}},
    )


@pytest.mark.parametrize("kind", ["observations", "capped", "tracks", "summary"])
@pytest.mark.parametrize("empty", [False, True])
def test_valid_per_source_snapshot_evidence_survives_without_relinking(
    tmp_path, monkeypatch, kind, empty
):
    collection = _collection(tmp_path, kind, empty=empty)
    document = _snapshot_document(collection)

    def forbidden(*args, **kwargs):
        pytest.fail("Snapshot loading must not rerun scientific assignment")

    monkeypatch.setattr(tracking, "build_tracks", forbidden)
    monkeypatch.setattr(tracking, "linear_sum_assignment", forbidden)
    loaded = _load_snapshot(document, CollectionLimits())
    assert loaded.table.rows == collection.table.rows
    assert loaded.table.observation_metadata is loaded.table.tracking_metadata is None
    for actual, expected in zip(loaded.items, collection.items, strict=True):
        assert actual.observation_metadata == expected.observation_metadata
        assert actual.tracking_metadata == expected.tracking_metadata


@pytest.mark.parametrize(
    "kind,column,row,value",
    [
        ("observations", "t_index", 1, 1),
        ("observations", "x_index", 0, -1),
        ("capped", "x_index", 0, 999),
        ("tracks", "track_id", 0, -9),
        ("tracks", "previous_frame", 1, 1),
        ("tracks", "gap_frames", 1, 0),
        ("tracks", "displacement", 1, 999.0),
        ("tracks", "speed", 1, 999.0),
        ("summary", "track_id", 0, -9),
        ("summary", "observation_count", 0, 9),
        ("summary", "duration", 0, 999.0),
        ("summary", "path_length", 0, -1.0),
    ],
)
def test_resealed_snapshot_with_invalid_scientific_cells_is_rejected(
    tmp_path, kind, column, row, value
):
    document = deepcopy(_snapshot_document(_collection(tmp_path, kind)))
    index = document["table"]["columns"].index(column)
    document["table"]["rows"][row][index] = _snapshot_cell(value)
    with pytest.raises(MeasurementCollectionError, match="time-series snapshot cells"):
        _load_snapshot(seal_document(document), CollectionLimits())


@pytest.mark.parametrize(
    "kind,column",
    [("observations", "x_index"), ("tracks", "track_id"), ("summary", "path_length")],
)
def test_resealed_snapshot_cannot_drop_protected_measurement_columns(
    tmp_path, kind, column
):
    document = deepcopy(_snapshot_document(_collection(tmp_path, kind)))
    index = document["table"]["columns"].index(column)
    document["table"]["columns"].pop(index)
    document["provenance"]["measurement_columns"].remove(column)
    document["table"]["column_units"] = [
        pair for pair in document["table"]["column_units"] if pair[0] != column
    ]
    for row in document["table"]["rows"]:
        row.pop(index)
    with pytest.raises(MeasurementCollectionError, match="time-series snapshot cells"):
        _load_snapshot(seal_document(document), CollectionLimits())


def test_included_item_evidence_is_not_validated_against_other_source_rows(tmp_path):
    collection = _collection(tmp_path, "tracks", exclude_second=True)
    loaded = _load_snapshot(_snapshot_document(collection), CollectionLimits())
    assert loaded.table.row_count == 2
    assert loaded.items[0].included
    assert not loaded.items[1].included
    assert loaded.items[1].row_count == 2
    assert loaded.items[1].tracking_metadata is not None


def test_per_item_validation_keeps_cooperative_cancellation(tmp_path, monkeypatch):
    document = _snapshot_document(_collection(tmp_path, "tracks"))
    original = tracking.validate_tracking_table
    cancelled = False

    def validate(*args, **kwargs):
        nonlocal cancelled
        cancelled = True
        return original(*args, **kwargs)

    monkeypatch.setattr(tracking, "validate_tracking_table", validate)
    with pytest.raises(OperationCancelled):
        _load_snapshot(document, CollectionLimits(), cancellation=lambda: cancelled)
