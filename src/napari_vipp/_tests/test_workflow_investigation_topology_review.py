"""Existing mesh facts stay explicit; no geometry, providers or user data."""

from copy import deepcopy

import pytest

from napari_vipp.core.workflow_inspection_metrics import public_role_binding
from napari_vipp.core.workflow_investigation_topology import topology_review_payload


def evidence(*, role="surface", digest="a", **changes):
    counts = {
        "kind": "3D surface mesh",
        "vertex_count": 4,
        "triangle_count": 4,
        "boundary_edge_count": 0,
        "nonmanifold_edge_count": 0,
        "duplicate_triangle_count": 0,
        "repeated_vertex_triangle_count": 0,
        "inconsistent_winding_edge_count": 0,
        "surface_component_count": 1,
        "euler_count_v_minus_e_plus_f": 2,
        "euler_characteristic": 2,
        "euler_status": "indexed_complex_count",
    } | changes
    binding = public_role_binding(role, "mesh", "n0", 0, digest * 64)
    return {
        "outputs": [{"role": role, "sha256": digest * 64, "diagnostics": counts}],
        "role_bindings": [binding],
    }


def test_closed_edges_do_not_hide_exact_topology_change_or_infer_genus():
    before = evidence()
    after = evidence(digest="b", euler_count_v_minus_e_plus_f=0, euler_characteristic=0)
    snapshots = deepcopy((before, after))
    payload = topology_review_payload("candidate0", before, "candidate1", after)
    row = payload["meshes"][0]
    assert row["baseline"]["counts"]["boundary_edge_count"] == 0
    assert row["candidate"]["counts"]["boundary_edge_count"] == 0
    assert row["baseline"]["counts"]["euler_count_v_minus_e_plus_f"] == 2
    assert row["candidate"]["counts"]["euler_count_v_minus_e_plus_f"] == 0
    assert "topology_changed" in {warning["code"] for warning in row["warnings"]}
    assert (
        "genus" in payload["limitations"] and "do not prove" in payload["limitations"]
    )
    assert "winner" not in payload and "accuracy" not in row
    assert (before, after) == snapshots


def test_initial_and_unchanged_inherited_defects_remain_visible():
    current = evidence(
        nonmanifold_edge_count=3,
        duplicate_triangle_count=2,
        surface_component_count=3,
        euler_characteristic=None,
        euler_status="invalid_indexed_facets",
    )
    row = topology_review_payload("candidate0", current, "candidate0", current)[
        "meshes"
    ][0]
    assert row["changed_fields"] == []
    assert {warning["code"] for warning in row["warnings"]} >= {
        "nonmanifold_edge_count",
        "duplicate_triangle_count",
        "surface_components",
        "invalid_indexed_facets",
    }


def test_missing_role_does_not_become_an_empty_or_unchanged_mesh():
    before, after = evidence(), evidence(role="new surface", digest="b")
    rows = topology_review_payload("candidate0", before, "candidate1", after)["meshes"]
    assert {row["role"] for row in rows} == {"surface", "new surface"}
    assert all(row["baseline"] is None or row["candidate"] is None for row in rows)
    assert all(row["warnings"][0]["code"] == "unmatched_role" for row in rows)


@pytest.mark.parametrize(
    "field,value",
    [
        ("boundary_edge_count", -1),
        ("triangle_count", True),
        ("euler_count_v_minus_e_plus_f", 0.5),
        ("surface_component_count", None),
    ],
)
def test_unknown_or_invalid_existing_counts_are_not_coerced(field, value):
    with pytest.raises(ValueError, match="integral topology"):
        topology_review_payload(
            "candidate0", evidence(), "candidate1", evidence(**{field: value})
        )


def test_changed_or_cross_role_binding_fails_closed():
    bad = evidence()
    bad["outputs"][0]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="exact measured role"):
        topology_review_payload("candidate0", evidence(), "candidate1", bad)


def test_no_mesh_means_no_fabricated_review():
    empty = {"outputs": [], "role_bindings": []}
    assert topology_review_payload("candidate0", empty, "candidate1", empty) is None


def test_empty_mesh_is_not_passed_as_a_closed_solid():
    empty = evidence(vertex_count=0, triangle_count=0, surface_component_count=0)
    row = topology_review_payload("candidate0", evidence(), "candidate1", empty)[
        "meshes"
    ][0]
    assert "empty_mesh" in {warning["code"] for warning in row["warnings"]}
