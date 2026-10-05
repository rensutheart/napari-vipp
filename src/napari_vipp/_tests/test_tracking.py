"""Analytical, malformed-data and bounded-resource contracts for tracking."""

import itertools
import json
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from napari_vipp.core import tracking
from napari_vipp.core.progress import OperationCancelled, ProgressContext
from napari_vipp.core.tables import TableData
from napari_vipp.core.tracking import build_tracks
from napari_vipp.core.tracking_metadata import (
    FramePopulation,
    ObservationSeriesMetadata,
    TrackingMetadata,
)


def series(rows, *, rank=2, frame_count=None, **overrides):
    axes = ("y", "x") if rank == 2 else ("z", "y", "x")
    frame_count = frame_count or max((row[0] for row in rows), default=0) + 1
    populations = tuple(
        FramePopulation(
            i, sum(row[0] == i for row in rows), sum(row[0] == i for row in rows)
        )
        for i in range(frame_count)
    )
    params = dict(
        spatial_axes=axes,
        source_shape=(100,) * rank,
        source_scale=(1.0,) * rank,
        source_origin=(0.0,) * rank,
        source_units=(None,) * rank,
        source_frame="synthetic-series",
        source_revision="a" * 64,
        frame_count=frame_count,
        time_scale=1.0,
        time_origin=0.0,
        time_unit=None,
        coordinate_columns=tuple(f"{axis}_index" for axis in axes),
        id_column="detection_id",
        frame_populations=populations,
    )
    params.update(overrides)
    metadata = ObservationSeriesMetadata(**params)
    return TableData(
        columns=("t_index", "detection_id", *metadata.coordinate_columns),
        rows=tuple(tuple(row) for row in rows),
        observation_metadata=metadata,
    )


def records(table):
    return table.records()


@pytest.mark.parametrize("rank", [2, 3])
def test_known_linear_motion_preserves_all_rows_and_input(rank):
    coordinates = [(2, 3), (5, 7), (8, 11)]
    if rank == 3:
        coordinates = [(10, *p) for p in coordinates]
    array = np.array([(t, 9 - t, *point) for t, point in enumerate(coordinates)])
    array.setflags(write=False)
    original = array.copy()
    table = series(array, rank=rank)
    before = table.rows
    observations, summary = build_tracks(table, maximum_displacement=5)
    assert np.array_equal(array, original)
    assert table.rows == before
    assert observations.columns[: len(table.columns)] == table.columns
    assert [r["track_id"] for r in records(observations)] == [1, 1, 1]
    assert [r["displacement"] for r in records(observations)] == [None, 5.0, 5.0]
    assert [r["speed"] for r in records(observations)] == [None, 5.0, 5.0]
    assert records(summary)[0] == dict(
        track_id=1,
        observation_count=3,
        first_frame=0,
        last_frame=2,
        duration=2.0,
        path_length=10.0,
        net_displacement=10.0,
        gap_count=0,
        missing_frame_count=0,
        review_flag=False,
    )
    assert summary.unit_for("duration") == "frames"
    assert observations.unit_for("speed") == "pixels/frames"


def test_maximum_cardinality_precedes_shortest_individual_link():
    table = series([(0, 1, 5, 1), (0, 2, 5, 3), (1, 1, 5, 1.1), (1, 2, 5, 0)])
    observations, summary = build_tracks(table, maximum_displacement=2)
    assert [r["track_id"] for r in records(observations)] == [1, 2, 2, 1]
    assert summary.row_count == 2
    assert all(r["review_flag"] for r in records(observations))


def test_distance_optimum_matches_independent_small_permutation_oracle():
    left = np.array([(4, 2), (5, 8), (10, 5)])
    right = np.array([(11, 5), (5, 2), (6, 9)])
    table = series(
        [(0, i + 1, *p) for i, p in enumerate(left)]
        + [(1, i + 1, *p) for i, p in enumerate(right)]
    )
    observations, _ = build_tracks(table, maximum_displacement=20)
    expected = min(
        itertools.permutations(range(3)),
        key=lambda order: sum(
            np.linalg.norm(left[i] - right[j]) for i, j in enumerate(order)
        ),
    )
    actual = [r["track_id"] for r in records(observations)[3:]]
    assert actual == [expected.index(j) + 1 for j in range(3)]


def test_anisotropic_mixed_length_units_and_time_seconds():
    table = series(
        [(0, 1, 1, 1, 1), (1, 1, 2, 5, 1)],
        rank=3,
        source_scale=(5, 500, 0.001),
        source_units=("um", "nm", "mm"),
        source_origin=(100, 5000, 0.2),
        time_scale=500,
        time_origin=40,
        time_unit="ms",
    )
    observations, summary = build_tracks(
        table, maximum_displacement=6, distance_units="Physical (micrometers)"
    )
    distance = np.sqrt(29)
    assert records(observations)[1]["displacement"] == pytest.approx(distance)
    assert records(observations)[1]["speed"] == pytest.approx(2 * distance)
    assert records(summary)[0]["duration"] == 0.5
    assert observations.unit_for("displacement") == "micrometers"
    assert observations.unit_for("speed") == "micrometers/seconds"
    assert summary.tracking_metadata == observations.tracking_metadata


def test_gap_is_missing_frames_with_elapsed_frame_distance_gate():
    table = series([(0, 1, 5, 1), (2, 1, 5, 3)], frame_count=4)
    assert build_tracks(table, maximum_displacement=1, maximum_gap=0)[1].row_count == 2
    observations, summary = build_tracks(table, maximum_displacement=1, maximum_gap=1)
    final = records(observations)[1]
    assert (final["track_id"], final["previous_frame"], final["gap_frames"]) == (
        1,
        0,
        1,
    )
    assert final["displacement"] == 2 and final["speed"] == 1
    assert observations.row_count == 2  # no synthetic gap row
    assert records(summary)[0]["missing_frame_count"] == 1


def test_recent_track_has_priority_over_closer_old_endpoint():
    table = series([(0, 1, 5, 2.1), (1, 1, 5, 4), (2, 1, 5, 3)])
    observations, _ = build_tracks(table, maximum_displacement=1, maximum_gap=1)
    assert [r["track_id"] for r in records(observations)] == [1, 2, 2]


def test_crossing_ambiguity_and_input_permutation_stability():
    rows = [(0, 7, 5, 2), (0, 3, 5, 0), (1, 8, 5, 1), (1, 2, 5, 1)]
    reference = build_tracks(series(rows), maximum_displacement=1)
    assert all(r["review_flag"] for r in records(reference[0]))
    for permutation in (rows[::-1], rows[2:] + rows[:2], rows[::2] + rows[1::2]):
        assert build_tracks(series(permutation), maximum_displacement=1) == reference


def test_empty_series_and_empty_boundary_frames_have_exact_evidence():
    observations, summary = build_tracks(series([], frame_count=5))
    assert observations.row_count == summary.row_count == 0
    assert len(observations.observation_metadata.frame_populations) == 5
    assert observations.tracking_metadata.source_observations.frame_count == 5
    observations, summary = build_tracks(series([(2, 1, 1, 1)], frame_count=5))
    assert records(summary)[0]["first_frame"] == 2
    assert records(summary)[0]["duration"] == 0
    assert records(observations)[0]["previous_frame"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("maximum_displacement", 0),
        ("maximum_displacement", -1),
        ("maximum_displacement", np.nan),
        ("maximum_displacement", True),
        ("maximum_gap", 1.5),
        ("maximum_gap", -1),
        ("maximum_gap", True),
        ("distance_units", "nanometers"),
    ],
)
def test_invalid_settings_fail(field, value):
    with pytest.raises(ValueError):
        build_tracks(series([(0, 1, 1, 1)]), **{field: value})


@pytest.mark.parametrize(
    "row",
    [
        (-1, 1, 1, 1),
        (3, 1, 1, 1),
        (0.0, 1, 1, 1),
        (True, 1, 1, 1),
        (0, 0, 1, 1),
        (0, 1.5, 1, 1),
        (0, True, 1, 1),
        (0, 1, np.nan, 1),
        (0, 1, np.inf, 1),
        (0, 1, -0.5, 1),
        (0, 1, 100, 1),
        (0, 1, True, 1),
        (0, 1, "1", 1),
        (0, 1, 1),
    ],
)
def test_invalid_rows_are_not_silently_dropped(row):
    table = replace(series([(0, 1, 1, 1)], frame_count=3), rows=(row,))
    with pytest.raises(ValueError):
        build_tracks(table)


def test_duplicate_identity_population_mismatch_missing_metadata_and_truncation():
    table = series([(0, 1, 1, 1)])
    for invalid in (
        replace(table, rows=table.rows * 2),
        replace(table, rows=()),
        replace(table, observation_metadata=None),
        replace(table, columns=("t_index", "detection_id", "y_index", "y_index")),
        replace(
            table,
            observation_metadata=replace(
                table.observation_metadata,
                frame_populations=(FramePopulation(0, 2, 1, True),),
            ),
        ),
    ):
        with pytest.raises(ValueError):
            build_tracks(invalid)


@pytest.mark.parametrize(
    "updates",
    [
        {"source_scale": (0, 1)},
        {"source_origin": (np.nan, 0)},
        {"time_scale": 0},
        {"time_origin": np.inf},
        {"source_units": ("um", None)},
        {"source_units": ("banana", "banana")},
        {"time_unit": "banana"},
        {"time_unit": "um"},
    ],
)
def test_invalid_calibration_fails(updates):
    with pytest.raises(ValueError):
        build_tracks(series([(0, 1, 1, 1)], **updates))


def test_physical_linking_requires_length_calibration():
    with pytest.raises(ValueError, match="known length"):
        build_tracks(series([(0, 1, 1, 1)]), distance_units="Physical (micrometers)")


def test_frame_local_wide_integer_ids_are_preserved_exactly():
    large = 2**63 + 11
    table = series([(0, large, 1, 1), (1, large + 1, 1, 2)])
    observations, _ = build_tracks(table)
    assert [r["detection_id"] for r in records(observations)] == [large, large + 1]


def test_resources_refused_before_assignment(monkeypatch):
    table = series([(0, 1, 1, 1), (0, 2, 1, 2), (1, 1, 1, 1), (1, 2, 1, 2)])

    def forbidden(*args, **kwargs):
        pytest.fail("Assignment must not run after admission refusal")

    monkeypatch.setattr(tracking, "linear_sum_assignment", forbidden)
    monkeypatch.setattr(tracking, "MAX_CANDIDATE_EDGES", 2)
    with pytest.raises(MemoryError, match="candidate limit"):
        build_tracks(table)
    monkeypatch.setattr(tracking, "MAX_CANDIDATE_EDGES", 100)
    monkeypatch.setattr(tracking, "MAX_COMPONENT_MATRIX_ELEMENTS", 2)
    with pytest.raises(MemoryError, match="component"):
        build_tracks(table)


def test_host_memory_refusal_and_cancellation_return_no_results(monkeypatch):
    table = series([(0, 1, 1, 1), (1, 1, 1, 2)])
    with pytest.raises(OperationCancelled):
        build_tracks(table, progress_context=ProgressContext(cancelled=lambda: True))

    class Refusal:
        allowed = False
        reason = "synthetic memory refusal"

    monkeypatch.setattr(
        tracking, "preflight_host_allocation", lambda *a, **k: Refusal()
    )
    with pytest.raises(MemoryError, match="synthetic memory refusal"):
        build_tracks(table)


def test_cancel_after_native_assignment(monkeypatch):
    table = series([(0, 1, 1, 1), (1, 1, 1, 2)])
    original = tracking.linear_sum_assignment
    cancelled = False

    def assignment(*args, **kwargs):
        nonlocal cancelled
        result = original(*args, **kwargs)
        cancelled = True
        return result

    monkeypatch.setattr(tracking, "linear_sum_assignment", assignment)
    with pytest.raises(OperationCancelled):
        build_tracks(
            table, progress_context=ProgressContext(cancelled=lambda: cancelled)
        )


def test_immutable_schema_strict_json_round_trip():
    metadata = series([(0, 1, 1, 1)]).observation_metadata
    assert (
        ObservationSeriesMetadata.from_dict(json.loads(json.dumps(metadata.to_dict())))
        == metadata
    )
    tracks = build_tracks(series([(0, 1, 1, 1)]))[0].tracking_metadata
    assert (
        TrackingMetadata.from_dict(json.loads(json.dumps(tracks.to_dict()))) == tracks
    )
    with pytest.raises(FrozenInstanceError):
        metadata.frame_count = 3
    for data in (
        {**metadata.to_dict(), "schema_version": True},
        {**metadata.to_dict(), "unknown": 1},
        {**metadata.to_dict(), "frame_populations": []},
    ):
        with pytest.raises(ValueError):
            ObservationSeriesMetadata.from_dict(data)


@pytest.mark.parametrize(
    "population",
    [
        (0, 1, 2, False),
        (0, 2, 1, False),
        (0, 1, 1, True),
        (False, 1, 1, False),
    ],
)
def test_inconsistent_population_metadata_fails(population):
    with pytest.raises(ValueError):
        FramePopulation(*population)


def test_structural_validation_preserves_caps_and_completed_tracks():
    table = series([(0, 1, 1, 1)])
    capped = replace(
        table,
        observation_metadata=replace(
            table.observation_metadata,
            frame_populations=(FramePopulation(0, 2, 1, True),),
        ),
    )
    assert tracking.validate_observation_table(capped, for_linking=False).truncated
    observations, _ = build_tracks(table)
    assert (
        tracking.validate_observation_table(observations, for_linking=False)
        == table.observation_metadata
    )
    with pytest.raises(ValueError, match="tracking output"):
        build_tracks(observations)


def test_conflicting_coordinate_units_are_rejected():
    table = replace(series([(0, 1, 1, 1)]), column_units=(("x_index", "um"),))
    with pytest.raises(ValueError, match="source-index|Source-index"):
        build_tracks(table)


def test_sparse_independent_components_never_allocate_a_full_pair_matrix(monkeypatch):
    rows = [
        (t, identity + 1, 5, 3 * identity) for t in range(2) for identity in range(30)
    ]
    monkeypatch.setattr(tracking, "MAX_COMPONENT_MATRIX_ELEMENTS", 2)
    observations, summary = build_tracks(series(rows), maximum_displacement=1)
    assert summary.row_count == 30
    assert observations.row_count == 60


def test_unsafe_integer_coordinates_fail_without_restricting_wide_ids():
    table = series([(0, 1, 1, 2**53 + 1)], source_shape=(2, 2**53 + 2))
    with pytest.raises(ValueError, match="exact float64"):
        build_tracks(table)


def _replace_cell(table, row, column, value):
    rows = [list(values) for values in table.rows]
    rows[row][table.columns.index(column)] = value
    return replace(table, rows=tuple(tuple(values) for values in rows))


@pytest.mark.parametrize("empty", [False, True])
def test_valid_tracking_outputs_pass_structural_validation(empty):
    table = series([] if empty else [(0, 1, 1, 1), (2, 1, 1, 2)], frame_count=4)
    for output in build_tracks(table, maximum_gap=1):
        assert tracking.validate_tracking_table(output) == output.tracking_metadata


@pytest.mark.parametrize(
    "column,row,value",
    [
        ("track_id", 0, -1),
        ("track_id", 1, 99),
        ("review_flag", 0, "False"),
        ("previous_frame", 1, None),
        ("previous_frame", 1, 1),
        ("gap_frames", 1, 0),
        ("displacement", 0, 0.0),
        ("displacement", 1, 4.0),
        ("speed", 1, 4.0),
    ],
)
def test_malformed_tracked_rows_fail_structural_validation(column, row, value):
    output = build_tracks(series([(0, 1, 1, 1), (2, 1, 1, 2)]), maximum_gap=1)[0]
    with pytest.raises(ValueError):
        tracking.validate_tracking_table(_replace_cell(output, row, column, value))


@pytest.mark.parametrize(
    "column,value",
    [
        ("track_id", -9),
        ("observation_count", 0),
        ("observation_count", 3),
        ("first_frame", 1),
        ("last_frame", 9),
        ("duration", 99),
        ("path_length", -1),
        ("net_displacement", 9),
        ("gap_count", 0),
        ("missing_frame_count", 0),
        ("review_flag", 1),
    ],
)
def test_malformed_summary_rows_fail_structural_validation(column, value):
    output = build_tracks(series([(0, 1, 1, 1), (2, 1, 1, 2)]), maximum_gap=1)[1]
    with pytest.raises(ValueError):
        tracking.validate_tracking_table(_replace_cell(output, 0, column, value))


def test_tracking_schema_units_and_duplicate_summary_ids_fail():
    outputs = build_tracks(series([(0, 1, 1, 1), (0, 2, 5, 5)]))
    for output in outputs:
        with pytest.raises(ValueError, match="protected"):
            tracking.validate_tracking_table(
                replace(output, columns=("track_id",), rows=((1,),))
            )
        with pytest.raises(ValueError, match="units"):
            tracking.validate_tracking_table(replace(output, column_units=()))
    with pytest.raises(ValueError, match="unique"):
        tracking.validate_tracking_table(_replace_cell(outputs[1], 1, "track_id", 1))


def test_impossible_summary_endpoint_and_span_populations_fail():
    original = series([(0, 1, 1, 1), (1, 1, 1, 1), (1, 2, 3, 3), (2, 1, 1, 1)])
    output = build_tracks(original)[1]
    invalid = replace(
        output,
        rows=(
            (1, 2, 0, 1, 1.0, 0.0, 0.0, 0, 0, False),
            (2, 2, 0, 1, 1.0, 0.0, 0.0, 0, 0, False),
        ),
    )
    with pytest.raises(ValueError, match="endpoints|spans"):
        tracking.validate_tracking_table(invalid)
