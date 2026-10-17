"""Bounded review of existing role-bound mesh measurements; no geometry work.

Changed indexed counts and existing defects are review warnings, not an inferred
genus, a scientific failure policy, an automatic repair or a candidate ranking.
The engine alone registers the complete payload at candidate completion and
grants citation authority only when it is actually sent.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy

from napari_vipp.core.workflow_inspection_metrics import public_role_binding

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_COUNTS = (
    "vertex_count",
    "triangle_count",
    "boundary_edge_count",
    "nonmanifold_edge_count",
    "duplicate_triangle_count",
    "repeated_vertex_triangle_count",
    "inconsistent_winding_edge_count",
    "surface_component_count",
    "euler_count_v_minus_e_plus_f",
    "euler_characteristic",
)
_TOPOLOGY = _COUNTS[2:]
_DEFECTS = {
    "boundary_edge_count": "open boundary edges",
    "nonmanifold_edge_count": "nonmanifold edges",
    "duplicate_triangle_count": "duplicate triangles",
    "repeated_vertex_triangle_count": "triangles with repeated vertices",
    "inconsistent_winding_edge_count": "edges with inconsistent winding",
}


def _measurements(evidence):
    if type(evidence) is not dict or type(evidence.get("outputs")) is not list:
        raise ValueError("A completed public candidate evidence record is required.")
    bindings = evidence.get("role_bindings", [])
    if type(bindings) is not list or len(bindings) > 8:
        raise ValueError("Mesh review requires the existing finite role bindings.")
    result = {}
    for output in evidence["outputs"]:
        diagnostics = output.get("diagnostics", {})
        if diagnostics.get("kind") != "3D surface mesh":
            continue
        role = output["role"]
        binding = next((row for row in bindings if row.get("role") == role), None)
        if binding is None or binding != public_role_binding(
            role,
            binding["node_key"],
            binding["node_alias"],
            binding["port"],
            output["sha256"],
        ):
            raise ValueError("Mesh review requires an exact measured role binding.")
        if role in result or len(result) == 8 or not _SHA.fullmatch(output["sha256"]):
            raise ValueError("Mesh review role identity is invalid or duplicated.")
        counts = {field: diagnostics.get(field) for field in _COUNTS}
        for field, value in counts.items():
            if field == "euler_characteristic" and value is None:
                continue
            if (
                type(value) is not int
                or abs(value) > 2**53 - 1
                or (not field.startswith("euler_") and value < 0)
            ):
                raise ValueError(
                    "Exact existing integral topology counts are required."
                )
        status = diagnostics.get("euler_status")
        if status not in {"invalid_indexed_facets", "indexed_complex_count"}:
            raise ValueError("The existing indexed topology status is required.")
        counts["euler_status"] = status
        result[role] = {"binding": deepcopy(binding), "counts": counts}
    return result


def topology_review_payload(
    baseline_id, baseline_evidence, candidate_id, candidate_evidence
):
    """Copy exact existing facts and flag changes, without recalculating meshes."""
    if any(
        type(value) is not str or not re.fullmatch(r"candidate[0-9]{1,3}", value)
        for value in (baseline_id, candidate_id)
    ):
        raise ValueError("Same-session candidate aliases are required for mesh review.")
    before, after = _measurements(baseline_evidence), _measurements(candidate_evidence)
    roles = list(dict.fromkeys((*after, *before)))
    if not roles:
        return None
    if len(roles) > 8:
        raise ValueError("Mesh review exceeds eight explicit roles.")
    rows = []
    for role in roles:
        previous, current = before.get(role), after.get(role)
        warnings = []
        changed = []
        if previous is None or current is None:
            warnings.append(
                {
                    "code": "unmatched_role",
                    "message": (
                        "This explicit mesh role has no exact before/after counterpart."
                    ),
                }
            )
        if previous is not None and current is not None:
            changed = [
                field
                for field in (*_TOPOLOGY, "euler_status")
                if previous["counts"][field] != current["counts"][field]
            ]
            if changed:
                warnings.append(
                    {
                        "code": "topology_changed",
                        "message": (
                            "Indexed topology counts changed. Inspect the exact "
                            "before/after masks and meshes; closed edges do not "
                            "cancel this change or establish the desired shape."
                        ),
                    }
                )
        if current is not None:
            counts = current["counts"]
            for field, label in _DEFECTS.items():
                if counts[field]:
                    warnings.append(
                        {
                            "code": field,
                            "message": f"Current mesh has {counts[field]} {label}.",
                        }
                    )
            if counts["triangle_count"] == 0:
                warnings.append(
                    {"code": "empty_mesh", "message": "Current mesh is empty."}
                )
            if counts["surface_component_count"] != 1:
                warnings.append(
                    {
                        "code": "surface_components",
                        "message": (
                            f"Current mesh has {counts['surface_component_count']} "
                            "edge-connected surfaces; display groups are not "
                            "biological objects."
                        ),
                    }
                )
            if counts["euler_status"] != "indexed_complex_count":
                warnings.append(
                    {
                        "code": "invalid_indexed_facets",
                        "message": (
                            "Invalid indexed facets prevent a valid "
                            "Euler-characteristic claim."
                        ),
                    }
                )
        rows.append(
            {
                "role": role,
                "baseline": previous,
                "candidate": current,
                "changed_fields": changed,
                "warnings": warnings,
            }
        )
    payload = {
        "kind": "mesh-topology-review",
        "version": 1,
        "baseline_candidate_id": baseline_id,
        "candidate_id": candidate_id,
        "meshes": rows,
        "limitations": (
            "Exact existing role-bound indexed counts only. Zero boundary edges "
            "do not prove a simple solid, self-intersection freedom, genus, correct "
            "reconstruction or anatomical accuracy. Unchanged counts also do not "
            "prove unchanged geometry. Warnings do not repair or rank candidates."
        ),
    }
    if (
        len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
        > 12 * 1024
    ):
        raise ValueError("The complete topology review exceeds its finite allowance.")
    return payload
