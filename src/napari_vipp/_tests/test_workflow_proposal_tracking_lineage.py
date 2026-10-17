"""Tracking proposals retain only known complete-series column identities.

These are pixel-free graph/metadata checks on synthetic sources. Actual table
rows, truncation and numerical tracking remain executor-time responsibilities.
"""

from __future__ import annotations

from dataclasses import replace
from functools import wraps

import pytest

from napari_vipp._tests.test_workflow_proposal_full_library import (
    full_node,
    full_response,
    full_source,
)
from napari_vipp.core.pipeline import NODE_LIBRARY_BY_ID
from napari_vipp.core.workflow_proposals import (
    ProposalError,
    capture_proposal_context,
    validate_proposal,
)


@pytest.fixture(autouse=True)
def prohibit_scientific_functions(monkeypatch):
    def guarded(function):
        @wraps(function)
        def fail(*_args, **_kwargs):
            pytest.fail("Tracking proposal validation called a scientific kernel.")

        return fail

    for operation_id, spec in tuple(NODE_LIBRARY_BY_ID.items()):
        if spec.function is not None:
            monkeypatch.setitem(
                NODE_LIBRARY_BY_ID,
                operation_id,
                replace(spec, function=guarded(spec.function)),
            )


def detection_context():
    return capture_proposal_context(full_source(axes="tyx"), "input")


def detection_then(context, middle, *, source_port=0):
    nodes = [full_node("detect", "detect_spots_per_frame")]
    nodes.extend(middle)
    nodes.append(full_node("tracks", "build_tracks"))
    links = [("input", "detect", 0)]
    previous = "detect"
    for node in middle:
        links.append((previous, node["key"], 0))
        previous = node["key"]
    links.append((previous, "tracks", 0, source_port))
    return full_response(context, nodes, links)


def test_direct_complete_detection_series_is_authorable():
    context = detection_context()
    assert validate_proposal(detection_then(context, []), context).can_apply


@pytest.mark.parametrize("summary_version", [1, 2])
def test_summary_does_not_carry_detection_series(summary_version):
    context = detection_context()
    raw = detection_then(
        context,
        [
            full_node(
                "summary", "summarize_measurements", summary_version=summary_version
            )
        ],
    )
    with pytest.raises(ProposalError, match="Build Tracks"):
        validate_proposal(raw, context)


def test_merged_table_does_not_inherit_series_from_any_one_ancestor():
    context = detection_context()
    raw = full_response(
        context,
        [
            full_node("detect", "detect_spots_per_frame"),
            full_node("merged", "merge_tables", input_count=2),
            full_node("tracks", "build_tracks"),
        ],
        [
            ("input", "detect", 0),
            ("detect", "merged", 0),
            ("detect", "merged", 1),
            ("merged", "tracks", 0),
        ],
    )
    with pytest.raises(ProposalError, match="Build Tracks"):
        validate_proposal(raw, context)


def test_track_summary_output_does_not_carry_observation_series():
    context = detection_context()
    raw = detection_then(
        context, [full_node("first_tracks", "build_tracks")], source_port=1
    )
    with pytest.raises(ProposalError, match="Build Tracks"):
        validate_proposal(raw, context)


def test_track_observation_output_cannot_be_linked_again():
    context = detection_context()
    raw = detection_then(context, [full_node("first_tracks", "build_tracks")])
    with pytest.raises(ProposalError, match="Build Tracks|already.*track"):
        validate_proposal(raw, context)


@pytest.mark.parametrize("column", ["t_index", "detection_id", "y_index", "x_index"])
def test_overwritten_protected_detection_columns_clear_tracking_eligibility(column):
    context = detection_context()
    raw = detection_then(
        context,
        [
            full_node(
                "annotation",
                "add_metadata_columns",
                metadata_columns=f"{column}=1",
                overwrite="yes",
            )
        ],
    )
    with pytest.raises(ProposalError, match="Build Tracks"):
        validate_proposal(raw, context)


@pytest.mark.parametrize("column", ["t_index", "detection_id", "y_index", "x_index"])
def test_dropping_protected_detection_columns_clears_tracking_eligibility(column):
    context = detection_context()
    kept_columns = ",".join(
        name
        for name in ("t_index", "detection_id", "y_index", "x_index")
        if name != column
    )
    raw = detection_then(
        context,
        [
            full_node(
                "columns",
                "select_table_columns",
                columns=kept_columns,
            )
        ],
    )
    with pytest.raises(ProposalError, match="Build Tracks"):
        validate_proposal(raw, context)


@pytest.mark.parametrize("columns", ["detection_id", "none", "<none>"])
def test_incomplete_kept_column_set_clears_tracking_eligibility(columns):
    context = detection_context()
    raw = detection_then(
        context, [full_node("columns", "select_table_columns", columns=columns)]
    )
    with pytest.raises(ProposalError, match="Build Tracks"):
        validate_proposal(raw, context)


def test_safe_annotation_and_column_reordering_retain_series_with_default_controls():
    context = detection_context()
    raw = detection_then(
        context,
        [
            full_node(
                "annotation", "add_metadata_columns", metadata_columns="condition=test"
            ),
            full_node(
                "columns",
                "select_table_columns",
                columns="detection_id,x_index,t_index,y_index,condition",
            ),
        ],
    )
    assert validate_proposal(raw, context).can_apply


@pytest.mark.parametrize(
    "undeclared",
    [{"append_unlisted": "yes"}, {"selection_mode": "Drop listed columns"}],
)
def test_kernel_only_column_controls_are_not_model_authorable(undeclared):
    context = detection_context()
    raw = detection_then(
        context,
        [
            full_node(
                "columns",
                "select_table_columns",
                columns="detection_id",
                **undeclared,
            )
        ],
    )
    with pytest.raises(ProposalError, match="declared"):
        validate_proposal(raw, context)


@pytest.mark.parametrize(("axes", "mode"), [("tyx", "2D YX"), ("tzyx", "3D ZYX")])
def test_complete_object_measurement_series_can_feed_tracking(axes, mode):
    context = capture_proposal_context(full_source(axes=axes), "input")
    raw = full_response(
        context,
        [
            full_node("mask", "binary_threshold"),
            full_node("labels", "label_connected_components", spatial_mode=mode),
            full_node("objects", "measure_objects", spatial_mode=mode),
            full_node("tracks", "build_tracks"),
        ],
        [
            ("input", "mask", 0),
            ("mask", "labels", 0),
            ("labels", "objects", 0),
            ("objects", "tracks", 0),
        ],
    )
    assert validate_proposal(raw, context).can_apply


def test_per_plane_measurements_of_spatial_volume_are_not_whole_volume_series():
    context = capture_proposal_context(full_source(axes="tzyx"), "input")
    raw = full_response(
        context,
        [
            full_node("mask", "binary_threshold"),
            full_node("labels", "label_connected_components", spatial_mode="3D ZYX"),
            full_node("objects", "measure_objects", spatial_mode="2D YX"),
            full_node("tracks", "build_tracks"),
        ],
        [
            ("input", "mask", 0),
            ("mask", "labels", 0),
            ("labels", "objects", 0),
            ("objects", "tracks", 0),
        ],
    )
    with pytest.raises(ProposalError, match="Build Tracks"):
        validate_proposal(raw, context)


@pytest.mark.parametrize(
    "operation",
    ["invert", "normalize_image", "rescale_intensity", "convert_dtype", "average_blur"],
)
def test_metadata_described_preprocessing_can_feed_per_frame_detection(operation):
    context = detection_context()
    raw = full_response(
        context,
        [
            full_node("process", operation),
            full_node("detect", "detect_spots_per_frame"),
            full_node("tracks", "build_tracks"),
        ],
        [("input", "process", 0), ("process", "detect", 0), ("detect", "tracks", 0)],
    )
    assert validate_proposal(raw, context).can_apply
