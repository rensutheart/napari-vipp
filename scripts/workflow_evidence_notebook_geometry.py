"""Check the user's closed-end requirement without repairing or ranking meshes."""

from __future__ import annotations

from workflow_microedit_evidence import candidate_key_mapping


class ClosedMeshGeometryError(ValueError):
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


def verify_closed_mesh_deliverables(candidate):
    """Only nonempty actual bound geometry with zero boundary edges may lock.

    This is NOT a complete solid/manifold/anatomical validity assertion. Other
    geometric diagnostics stay visible but do not introduce a hidden criterion.
    """
    from napari_vipp.core.meshes import MeshData, MeshState
    from napari_vipp.core.workflow_assessment import (
        _data_hash,
        _MeshSnapshot,
        _require_owned_readonly,
        _state_hash,
    )
    from napari_vipp.core.workflow_inspection_metrics import mesh_topology_summary

    receipt = {
        "candidate_id": candidate.candidate_id,
        "candidate_run_id": candidate.run_id,
        "passed": False,
        "criteria": (
            "Both explicitly bound delivered meshes are nonempty "
            "and have zero boundary edges."
        ),
        "scientific_success_established": False,
        "full_solid_manifold_or_anatomical_validity_claimed": False,
        "geometry_modified_or_repaired": False,
        "meshes": [],
    }

    def fail(message):
        receipt["failure"] = message
        raise ClosedMeshGeometryError(message, receipt)

    if candidate.status != "completed":
        fail("Closed-end checks require the actual completed nominated candidate.")
    mapping = candidate_key_mapping(candidate)
    roles = candidate.output_roles
    if len({row.get("role") for row in roles}) != len(roles):
        fail("Mesh deliverable roles must be explicitly unique.")
    outputs = list(candidate._outputs)
    for role in ("band_mesh", "circle_mesh"):
        nominated = [row for row in roles if row.get("role") == role]
        bound = [binding for name, binding in outputs if name == role]
        if len(nominated) != 1 or len(bound) != 1:
            fail(
                "Both band_mesh and circle_mesh require exact delivered role bindings."
            )
        chosen, binding = nominated[0], bound[0]
        key, port = chosen.get("node"), chosen.get("port")
        if (
            type(key) is not str
            or key not in mapping
            or type(port) is not int
            or port < 0
            or binding.node_id != mapping[key]
            or type(binding.port) is not int
            or binding.port != port
        ):
            fail("A mesh role's model key, actual node or typed port differs.")
        mesh = binding.data
        if (
            type(mesh) not in {MeshData, _MeshSnapshot}
            or type(binding.state) is not MeshState
            or type(mesh.state) is not MeshState
            or _data_hash(mesh) != binding.fingerprint
            or _state_hash(binding.state) != binding.state_fingerprint
            or _state_hash(mesh.state) != binding.state_fingerprint
        ):
            fail(
                "A delivered role lacks its exact actual mesh geometry "
                "and carried MeshState."
            )
        for array in (mesh.vertices, mesh.faces, mesh.face_object_ids):
            _require_owned_readonly(array)
        if not len(mesh.vertices) or not len(mesh.faces):
            fail("A delivered mesh is empty, so closed ends cannot be established.")
        topology = mesh_topology_summary(mesh.vertices, mesh.faces)
        limitations = []
        for name in (
            "nonmanifold_edge_count",
            "duplicate_triangle_count",
            "repeated_vertex_triangle_count",
            "inconsistent_winding_edge_count",
        ):
            if topology[name]:
                limitations.append({"diagnostic": name, "count": topology[name]})
        if topology["surface_component_count"] != 1:
            limitations.append(
                {
                    "diagnostic": "surface_component_count",
                    "count": topology["surface_component_count"],
                }
            )
        receipt["meshes"].append(
            {
                "role": role,
                "model_key": key,
                "actual_node_id": binding.node_id,
                "port": port,
                "output_sha256": binding.fingerprint,
                "state_sha256": binding.state_fingerprint,
                "carried_mesh_state": binding.state.to_dict(),
                "vertex_count": len(mesh.vertices),
                "triangle_count": len(mesh.faces),
                "topology": topology,
                "other_diagnostics_not_additional_gates": limitations,
            }
        )
        if topology["boundary_edge_count"] != 0:
            fail(
                "A delivered mesh has boundary edges; "
                "the closed-end requirement is unmet."
            )
    receipt["passed"] = True
    return receipt
