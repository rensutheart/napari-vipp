"""Analytical evidence fixtures: exact counts, bounded resources, no provider."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from napari_vipp._tests.test_workflow_assessment import (
    commit_cache,
    context_for,
    fixture,
    state_for,
)
from napari_vipp.core import workflow_inspection_metrics as metrics
from napari_vipp.core.metadata import AxisMetadata
from napari_vipp.core.tables import TableData, table_state_from_data
from napari_vipp.core.workflow_assessment import (
    MAX_TABLE_COLUMNS,
    AssessmentError,
    _table_summary,
    capture_assessment_snapshot,
)
from napari_vipp.core.workflow_inspection_metrics import (
    InspectionMetricsCancelled,
    InspectionMetricsError,
    foreground_change_summary,
    foreground_span_summary,
    mesh_topology_summary,
    public_role_binding,
)


def tetrahedron():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
    vertices.setflags(write=False)
    faces.setflags(write=False)
    return vertices, faces


def test_tetrahedron_topology_counts_used_vertices_not_unused_display_buffer():
    vertices, faces = tetrahedron()
    vertices = np.vstack((vertices, [100, 100, 100]))
    original = vertices.copy(), faces.copy()
    vertices.setflags(write=False)
    result = mesh_topology_summary(vertices, faces)
    assert result["used_vertex_count"] == 4
    assert result["unused_vertex_count"] == 1
    assert result["unique_edge_count"] == 6
    assert result["surface_component_count"] == 1
    assert result["euler_characteristic"] == 2
    for name in (
        "boundary_edge_count",
        "nonmanifold_edge_count",
        "inconsistent_winding_edge_count",
        "duplicate_triangle_count",
        "repeated_vertex_triangle_count",
    ):
        assert result[name] == 0
    np.testing.assert_array_equal(vertices, original[0])
    np.testing.assert_array_equal(faces, original[1])
    assert not vertices.flags.writeable and not faces.flags.writeable
    assert "genus" in result["topology_limitations"]


def test_triangle_components_require_a_shared_edge_not_only_a_vertex():
    vertices = np.zeros((5, 3), dtype=float)
    faces = np.array([[0, 1, 2], [0, 3, 4]], dtype=np.int64)
    result = mesh_topology_summary(vertices, faces)
    assert result["surface_component_count"] == 2
    assert result["boundary_edge_count"] == 6
    assert result["euler_characteristic"] == 1
    assert "complete indexed edge" in result["component_definition"]


def test_mesh_open_nonmanifold_winding_and_invalid_indexed_facets_are_separate():
    vertices, faces = tetrahedron()
    assert mesh_topology_summary(vertices, faces[:-1])["boundary_edge_count"] == 3
    reversed_face = faces.copy()
    reversed_face[0] = reversed_face[0, ::-1]
    assert (
        mesh_topology_summary(vertices, reversed_face)[
            "inconsistent_winding_edge_count"
        ]
        == 3
    )
    duplicate = mesh_topology_summary(vertices, np.vstack((faces, faces[0])))
    assert duplicate["duplicate_triangle_count"] == 1
    assert duplicate["nonmanifold_edge_count"] == 3
    assert duplicate["euler_characteristic"] is None
    assert duplicate["euler_count_v_minus_e_plus_f"] == 3
    repeated = mesh_topology_summary(vertices, np.array([[0, 0, 1]], dtype=np.int64))
    assert repeated["repeated_vertex_triangle_count"] == 1
    assert repeated["euler_characteristic"] is None


def test_empty_mesh_is_not_invented_as_one_surface():
    result = mesh_topology_summary(np.empty((0, 3)), np.empty((0, 3), dtype=np.int64))
    assert result["surface_component_count"] == 0
    assert result["used_vertex_count"] == result["unique_edge_count"] == 0
    assert result["euler_characteristic"] == 0


@pytest.mark.parametrize("bad", ["negative", "outside", "nonfinite", "float_faces"])
def test_invalid_mesh_indices_or_geometry_fail_without_repair(bad):
    vertices, faces = (array.copy() for array in tetrahedron())
    if bad == "negative":
        faces[0, 0] = -1
    elif bad == "outside":
        faces[0, 0] = len(vertices)
    elif bad == "nonfinite":
        vertices[0, 0] = np.nan
    else:
        faces = faces.astype(float)
    with pytest.raises(InspectionMetricsError):
        mesh_topology_summary(vertices, faces)


def test_mesh_limits_and_cancellation_remain_bounded(monkeypatch):
    vertices, faces = tetrahedron()
    monkeypatch.setattr(metrics, "MAX_METRIC_TRIANGLES", 3)
    with pytest.raises(InspectionMetricsError, match="bounds"):
        mesh_topology_summary(vertices, faces)
    monkeypatch.setattr(metrics, "MAX_METRIC_TRIANGLES", 200_000)
    checks = iter([False, False, True])
    with pytest.raises(InspectionMetricsCancelled):
        mesh_topology_summary(vertices, faces, cancelled=lambda: next(checks))


def test_spans_are_native_axis_cell_extents_with_gaps_not_thickness():
    labels = np.zeros((2, 6, 4), dtype=np.uint64)
    labels[:, (1, 4), 2] = 2**63 + 1
    axes = tuple(
        AxisMetadata(name, "space", "um", scale=scale)
        for name, scale in zip("zyx", (3, 2, 0.5), strict=True)
    )
    state = state_for(labels, kind="label image", axes=axes)
    labels.setflags(write=False)
    result = foreground_span_summary(labels, state)
    z, y, x = result["axes"]
    assert y["native_cell_span"]["minimum"] == y["native_cell_span"]["maximum"] == 4
    assert y["calibrated_span"]["median"] == 8
    assert z["calibrated_span"]["maximum"] == 6
    assert x["native_cell_span"]["maximum"] == 1
    assert y["nonempty_line_count"] == 2
    assert "not perpendicular thickness" in result["meaning"]
    assert not labels.flags.writeable and labels.max() == 2**63 + 1


def test_empty_spans_and_extreme_calibration_are_defined_without_nonfinite_json():
    empty = np.zeros((2, 3), dtype=bool)
    result = foreground_span_summary(empty, state_for(empty))
    assert all(item["native_cell_span"]["maximum"] is None for item in result["axes"])
    full = np.ones((2, 3), dtype=bool)
    state = state_for(
        full,
        axes=(
            AxisMetadata("y", "space", "um", scale=1e308),
            AxisMetadata("x", "space", "um"),
        ),
    )
    result = foreground_span_summary(full, state)
    assert result["axes"][0]["calibrated_span"]["maximum"] is None
    assert "overflow" in result["axes"][0]["calibration_status"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("bad", ["shape", "dtype", "privateunit", "nonfinite"])
def test_spans_reject_mismatched_or_unsafe_interpretations(bad):
    data = np.ones((2, 3), dtype=float)
    state = state_for(data)
    if bad == "shape":
        state = replace(state, shape=(3, 2))
    elif bad == "dtype":
        state = replace(state, dtype="uint8")
    elif bad == "privateunit":
        state = replace(
            state, axes=(AxisMetadata("y", "space", "PRIVATE"), state.axes[1])
        )
    else:
        data[0, 0] = np.inf
    with pytest.raises(InspectionMetricsError):
        foreground_span_summary(data, state)


def test_membership_change_is_exact_for_wide_labels_and_empty_is_undefined():
    left = np.array([[0, 2**63 + 1], [7, 0]], dtype=np.uint64)
    right = np.array([[0, 1], [0, 1]], dtype=bool)
    result = foreground_change_summary(left, right)
    assert (
        result["intersection_count"]
        == result["added_count"]
        == result["removed_count"]
        == 1
    )
    assert result["left_retained_fraction"] == 0.5
    assert result["intersection_over_union"] == pytest.approx(1 / 3)
    empty = np.zeros_like(right)
    result = foreground_change_summary(empty, empty)
    assert result["intersection_over_union"] is None
    assert result["left_retained_fraction"] is None
    assert "not ground truth" in result["meaning"]


def test_membership_rejects_nonfinite_shape_limits_and_honors_cancel(monkeypatch):
    with pytest.raises(InspectionMetricsError, match="equal shapes"):
        foreground_change_summary(np.ones((2, 3)), np.ones((3, 2)))
    with pytest.raises(InspectionMetricsError, match="Non-finite"):
        foreground_change_summary(np.full((2, 2), np.nan), np.zeros((2, 2)))
    with pytest.raises(InspectionMetricsCancelled):
        foreground_change_summary(
            np.ones((2, 2)), np.zeros((2, 2)), cancelled=lambda: True
        )
    monkeypatch.setattr(metrics, "MAX_METRIC_ELEMENTS", 3)
    with pytest.raises(InspectionMetricsError):
        foreground_change_summary(np.ones((2, 2)), np.zeros((2, 2)))


def test_explicit_public_semantic_binding_has_stable_port_and_content_identity():
    result = public_role_binding("band mesh", "protected_structure", "n17", 1, "a" * 64)
    digest = result.pop("binding_sha256")
    assert (
        digest
        == hashlib.sha256(
            json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert result["node_key"] == "protected_structure" and result["node_alias"] == "n17"
    assert result["port"] == 1
    assert (
        public_role_binding("band mesh", "protected_structure", "n17", 0, "a" * 64)[
            "binding_sha256"
        ]
        != digest
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ("", "b1", "n1", 0, "a" * 64),
        ("private/path", "b1", "n1", 0, "a" * 64),
        ("band", "C:/private", "n1", 0, "a" * 64),
        ("band", "b1", "n1", True, "a" * 64),
        ("band", "b1", "n1", -1, "a" * 64),
        ("band", "b1", "n1", 0, "not-a-hash"),
    ],
)
def test_semantic_binding_does_not_guess_or_accept_paths(arguments):
    with pytest.raises(InspectionMetricsError):
        public_role_binding(*arguments)


def mesh_table():
    columns = (
        "vertex_count",
        "triangle_count",
        "mesh_status",
        "mesh_error",
        "watertight",
        "physical_unit",
        "mesh_volume_physical",
        "mesh_surface_area_physical",
        "mesh_extent_z_physical",
        "mesh_extent_y_physical",
        "mesh_extent_x_physical",
        "sphericity",
        "mesh_extent_ratio_z_y",
        "mesh_extent_ratio_z_x",
        "mesh_extent_ratio_y_x",
        "equivalent_sphere_radius_physical",
        "equivalent_sphere_diameter_physical",
        "solidity_3d",
        "PRIVATE_COLUMN",
    )
    row = (
        8,
        12,
        "open_or_invalid_surface",
        "PRIVATE_PATH/error",
        False,
        "um",
        np.nan,
        12,
        1,
        2,
        3,
        np.nan,
        0.5,
        1 / 3,
        2 / 3,
        np.nan,
        np.nan,
        np.nan,
        "PRIVATE_ROW",
    )
    return TableData(
        columns,
        (row,),
        table_kind="3D mesh morphology",
        column_units=(
            ("mesh_volume_physical", "um^3"),
            ("mesh_surface_area_physical", "PRIVATE_UNIT"),
        ),
    )


def test_registered_mesh_schema_exposes_types_units_status_and_booleans_not_text():
    result = _table_summary(
        mesh_table(), None, operation_id="measure_3d_mesh_morphology"
    )
    columns = {item["column"]: item for item in result["column_summaries"]}
    assert len(columns) == MAX_TABLE_COLUMNS and result["columns_omitted"] == 3
    assert columns["watertight"]["false_count"] == 1
    assert columns["watertight"]["true_count"] == 0
    assert columns["mesh_status"]["value_counts"] == {"open_or_invalid_surface": 1}
    assert columns["mesh_volume_physical"]["unit"] == "um^3"
    assert columns["mesh_volume_physical"]["nonfinite_count"] == 1
    assert columns["mesh_volume_physical"]["minimum"] is None
    assert columns["mesh_surface_area_physical"]["unit"] is None
    assert "PRIVATE" not in json.dumps(result, allow_nan=False)
    assert "solidity_3d" in result["omitted_standard_columns"]


@pytest.mark.parametrize(
    "operation,kind",
    [
        (None, "3D mesh morphology"),
        ("plot_results", "3D mesh morphology"),
        ("measure_3d_mesh_morphology", "arbitrary table"),
    ],
)
def test_known_names_do_not_unredact_arbitrary_table_without_registered_provenance(
    operation, kind
):
    table = replace(mesh_table(), table_kind=kind)
    result = _table_summary(table, None, operation_id=operation)
    assert "schema" not in result
    assert "mesh_status" not in json.dumps(result)
    assert "PRIVATE" not in json.dumps(result)


def test_unknown_status_values_stay_redacted_and_actual_assessment_passes_producer():
    table = replace(
        mesh_table(),
        columns=("mesh_status", "watertight"),
        rows=(("PRIVATE_STATUS", np.bool_(True)),),
    )
    result = _table_summary(table, None, operation_id="measure_3d_mesh_morphology")
    status = next(
        item for item in result["column_summaries"] if item["column"] == "mesh_status"
    )
    assert status["redacted_value_count"] == 1
    assert "PRIVATE" not in json.dumps(result)
    pipeline, nodes = fixture(mesh=True)
    node = pipeline.add_node("measure_3d_mesh_morphology")
    assert pipeline.connect(nodes[-1], node.id).success
    commit_cache(pipeline, node.id, table, table_state_from_data(table))
    context = context_for(pipeline, (*nodes, node.id))
    actual = context.public_request("goal")["selected_outputs"][-1]["diagnostics"]
    assert actual["schema"] == "registered 3D mesh morphology measurements"
    assert any(item.get("true_count") == 1 for item in actual["column_summaries"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("column_units", (("mesh_volume_physical", "mm^3"),)),
        ("table_kind", "untrusted altered kind"),
    ],
)
def test_newly_interpreted_table_schema_and_units_are_bound_to_freshness(field, value):
    pipeline, nodes = fixture(mesh=True)
    node = pipeline.add_node("measure_3d_mesh_morphology")
    assert pipeline.connect(nodes[-1], node.id).success
    table = mesh_table()
    commit_cache(pipeline, node.id, table, table_state_from_data(table))
    snapshot = capture_assessment_snapshot(pipeline, "input", (*nodes, node.id), "run")
    # Exercise hostile mutation even though TableData itself is frozen. A stale
    # schema must not retain authority because a separate TableState was unchanged.
    object.__setattr__(table, field, value)
    with pytest.raises(AssessmentError, match="changed"):
        snapshot.assert_current(pipeline)
