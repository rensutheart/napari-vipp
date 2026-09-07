"""Object-aware, non-destructive triangle-mesh collection operations.

Object identity is explicit per triangle, not inferred from vertex colours or
surface containment. Combining never welds vertices or performs geometric union.
Splitting uses shared edges, so a cavity's disconnected inner surface becomes a
separate surface object; extract connected foreground objects before meshing when
cavity shells must remain part of one material object.
"""

from __future__ import annotations

import colorsys
from dataclasses import replace

import numpy as np

from napari_vipp.core.grid import _unit_dimension_and_factor
from napari_vipp.core.meshes import MeshData, _mesh_calibration

MEASUREMENT_PROPERTIES = (
    "mesh_volume_physical",
    "mesh_surface_area_physical",
    "sphericity",
    "mesh_extent_z_physical",
    "mesh_extent_y_physical",
    "mesh_extent_x_physical",
    "vertex_count",
    "triangle_count",
    "mesh_id",
)


def object_id_color(object_id):
    """Deterministic categorical colours; no measurement-dependent scaling."""
    hue = ((int(object_id) * 11400714819323198485) % (1 << 64)) / (1 << 64)
    return (*colorsys.hsv_to_rgb(hue, 0.7, 0.95), 1.0)


def _require_mesh(mesh):
    if not isinstance(mesh, MeshData):
        raise TypeError("This operation requires a 3D mesh.")


def _select_faces(mesh, selected, *, records=None, state=None):
    selected_faces = mesh.faces[selected]
    ids = mesh.face_object_ids[selected]
    used, inverse = np.unique(selected_faces, return_inverse=True)
    vertices = mesh.vertices[used]
    faces = inverse.reshape((-1, 3))
    selected_ids = set(int(v) for v in np.unique(ids))
    records = tuple(
        item
        for item in (mesh.objects if records is None else records)
        if item.object_id in selected_ids
    )
    return MeshData(vertices, faces, state or mesh.state, ids, records)


def iter_mesh_objects(mesh):
    """Yield compact single-object meshes in stable ID order in the same frame."""
    _require_mesh(mesh)
    for item, selected in _object_face_groups(mesh):
        yield _select_faces(mesh, selected, records=(item,))


def _object_face_groups(mesh):
    """Group faces once, avoiding an entire-face scan for every object."""
    ordered = np.argsort(mesh.face_object_ids, kind="stable")
    sorted_ids = mesh.face_object_ids[ordered]
    bounds = np.concatenate(
        ([0], np.flatnonzero(np.diff(sorted_ids)) + 1, [len(ordered)])
    )
    for index, item in enumerate(mesh.objects):
        yield item, ordered[bounds[index] : bounds[index + 1]]


def _calibrated_frame(state):
    _mesh_calibration(state)
    if tuple(axis.name.lower() for axis in state.spatial_axes) != tuple("zyx") or any(
        axis.type != "space" or not axis.is_explicit for axis in state.spatial_axes
    ):
        raise ValueError("Mesh operations require explicit Z/Y/X coordinate metadata.")
    dimensions = [_unit_dimension_and_factor(axis.unit) for axis in state.spatial_axes]
    factors = np.asarray([value for _dimension, value in dimensions])
    scales = np.asarray([axis.scale for axis in state.spatial_axes]) * factors
    origins = np.asarray([axis.translation for axis in state.spatial_axes]) * factors
    if (
        not np.isfinite(scales).all()
        or np.any(scales <= 0)
        or not np.isfinite(origins).all()
    ):
        raise ValueError(
            "Mesh calibration requires finite positive scales and finite origins."
        )
    return scales, origins, dimensions[0][0]


def _new_id(reserved):
    candidate = max(reserved, default=0) + 1
    if candidate > np.iinfo(np.int64).max:
        candidate = 1
        while candidate in reserved:
            candidate += 1
    reserved.add(candidate)
    return candidate


def _combined_source_history(record, state):
    """Preserve repeated scientific steps; deduplicating text loses provenance."""
    original = record.source_history
    current = state.history
    if not current:
        return original
    if current[: len(original)] == original:
        return current
    return (
        *original,
        f"Combined input history: {state.source_name or 'unnamed mesh'}",
        *current,
    )


def combine_meshes(meshes, input_count=None, *, progress=None):
    """Collect objects in the first input's physical frame, with no union/weld."""
    meshes = tuple(meshes)
    if input_count is not None and (
        isinstance(input_count, (bool, np.bool_))
        or not isinstance(input_count, (int, np.integer))
        or input_count != len(meshes)
    ):
        raise ValueError(
            "Input meshes must be an integer matching the supplied mesh count."
        )
    if not meshes:
        raise ValueError("Combine Meshes requires at least one mesh.")
    for mesh in meshes:
        _require_mesh(mesh)
    reference = meshes[0].state
    target_scales, target_origins, target_dimension = _calibrated_frame(reference)
    reserved = {item.object_id for mesh in meshes for item in mesh.objects}
    used_ids = set()
    vertices, faces, ids, records = [], [], [], []
    vertex_offset = 0
    for index, mesh in enumerate(meshes):
        if progress is not None:
            progress.check_cancelled()
            progress.report(
                index, len(meshes), "Combining mesh objects without union or welding"
            )
        scales, origins, dimension = _calibrated_frame(mesh.state)
        if dimension != target_dimension:
            raise ValueError(
                "Combine Meshes cannot mix calibrated and uncalibrated meshes "
                "or incompatible units."
            )
        # Work with the difference of origins to avoid losing small structures
        # unnecessarily when both frames share a large physical translation.
        transformed = (
            mesh.vertices.copy()
            if np.array_equal(scales, target_scales)
            and np.array_equal(origins, target_origins)
            else (mesh.vertices * scales + (origins - target_origins)) / target_scales
        )
        if not np.isfinite(transformed).all():
            raise ValueError("Combined mesh coordinates exceed finite numeric range.")
        vertices.append(transformed)
        faces.append(mesh.faces + vertex_offset)
        vertex_offset += len(mesh.vertices)
        remapped_ids = []
        for item in mesh.objects:
            new_id = _new_id(reserved) if item.object_id in used_ids else item.object_id
            used_ids.add(new_id)
            remapped_ids.append(new_id)
            records.append(
                replace(
                    item,
                    object_id=new_id,
                    source_object_id=item.source_object_id or item.object_id,
                    source_name=item.source_name or mesh.state.source_name,
                    source_axes=item.source_axes
                    or tuple(a.name for a in mesh.state.spatial_axes),
                    source_scale=item.source_scale
                    or tuple(a.scale for a in mesh.state.spatial_axes),
                    source_units=item.source_units
                    or tuple(a.unit for a in mesh.state.spatial_axes),
                    source_origin=item.source_origin
                    or tuple(a.translation for a in mesh.state.spatial_axes),
                    source_history=_combined_source_history(item, mesh.state),
                )
            )
        original_ids = np.asarray(
            [item.object_id for item in mesh.objects], dtype=np.int64
        )
        mapping = np.asarray(remapped_ids, dtype=np.int64)
        ids.append(mapping[np.searchsorted(original_ids, mesh.face_object_ids)])
    entry = (
        f"Combine Meshes: {len(meshes)} inputs; retained separate objects and colours; "
        "converted compatible units into first input frame; "
        "no welding or geometric union"
    )
    result = MeshData(
        np.concatenate(vertices),
        np.concatenate(faces),
        replace(
            reference,
            history=reference.history + (entry,),
            processing_history=reference.processing_history + (entry,),
        ),
        np.concatenate(ids),
        tuple(records),
    )
    if progress is not None:
        progress.check_cancelled()
        progress.report(
            len(meshes), len(meshes), f"Combined {result.object_count} objects"
        )
    return result


def _edge_components(faces):
    """One component per connected triangle set; touching at a vertex is not enough."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    edges = np.sort(
        np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1
    )
    face_ids = np.tile(np.arange(len(faces)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    sorted_edges = edges[order]
    matches = np.all(sorted_edges[1:] == sorted_edges[:-1], axis=1)
    left, right = face_ids[order[:-1]][matches], face_ids[order[1:]][matches]
    graph = coo_matrix(
        (np.ones(len(left), dtype=bool), (left, right)), shape=(len(faces), len(faces))
    )
    _count, labels = connected_components(graph, directed=False)
    return labels


def split_mesh_objects(mesh, *, progress=None):
    """Split shared-edge components within each object; never infer containment."""
    _require_mesh(mesh)
    ids = mesh.face_object_ids.copy()
    reserved = {item.object_id for item in mesh.objects}
    records = []
    for index, (item, selection) in enumerate(_object_face_groups(mesh)):
        if progress is not None:
            progress.check_cancelled()
            progress.report(
                index, max(1, mesh.object_count), "Separating edge-connected surfaces"
            )
        labels = _edge_components(mesh.faces[selection])
        component_values, first_positions = np.unique(labels, return_index=True)
        components = component_values[np.argsort(first_positions)]
        if len(components) == 1:
            records.append(item)
            continue
        ordered = np.argsort(labels, kind="stable")
        sorted_labels = labels[ordered]
        for component_index, component in enumerate(components):
            new_id = item.object_id if component_index == 0 else _new_id(reserved)
            start = np.searchsorted(sorted_labels, component, side="left")
            stop = np.searchsorted(sorted_labels, component, side="right")
            ids[selection[ordered[start:stop]]] = new_id
            records.append(
                replace(
                    item,
                    object_id=new_id,
                    name=f"{item.name} · part {component_index + 1}",
                    source_object_id=item.source_object_id or item.object_id,
                    parent_object_id=item.object_id,
                )
            )
    entry = (
        "Split Mesh Objects: shared-edge triangle connectivity within each object; "
        "disconnected cavity shells become separate objects; no containment inference"
    )
    result = MeshData(
        mesh.vertices,
        mesh.faces,
        replace(
            mesh.state,
            history=mesh.state.history + (entry,),
            processing_history=mesh.state.processing_history + (entry,),
        ),
        ids,
        tuple(records),
    )
    if progress is not None:
        progress.check_cancelled()
        progress.report(
            max(1, mesh.object_count),
            max(1, mesh.object_count),
            f"Split into {result.object_count} objects",
        )
    return result


def _object_measurements(mesh, property_name, progress):
    from napari_vipp.core.meshes import measure_mesh_geometry

    if property_name not in MEASUREMENT_PROPERTIES:
        raise ValueError(f"Unknown mesh measurement {property_name!r}.")
    table = measure_mesh_geometry(
        mesh, include_convex_hull_metrics=False, progress=progress
    )
    index, id_index = table.columns.index(property_name), table.columns.index("mesh_id")
    convert = (
        int if property_name in {"mesh_id", "vertex_count", "triangle_count"} else float
    )
    return {int(row[id_index]): convert(row[index]) for row in table.rows}


def color_mesh_objects(
    mesh, color_by="Object ID", color_map="viridis", *, progress=None
):
    """Colour stable identities or current-geometry measurements; invalids are grey."""
    _require_mesh(mesh)
    if progress is not None:
        progress.check_cancelled()
    if color_by == "Object ID":
        colors = {
            item.object_id: object_id_color(item.object_id) for item in mesh.objects
        }
        detail = "deterministic categorical object-ID colours"
    else:
        from matplotlib import colormaps

        try:
            cmap = colormaps[color_map]
        except KeyError as error:
            raise ValueError(f"Unknown mesh colour map {color_map!r}.") from error
        values = _object_measurements(mesh, color_by, progress)
        finite = [value for value in values.values() if np.isfinite(value)]
        low, high = (min(finite), max(finite)) if finite else (0.0, 0.0)
        colors = {}
        for item in mesh.objects:
            value = values.get(item.object_id, float("nan"))
            colors[item.object_id] = (
                tuple(
                    float(v)
                    for v in cmap(0.5 if high == low else (value - low) / (high - low))
                )
                if np.isfinite(value)
                else (0.5, 0.5, 0.5, 1.0)
            )
        detail = (
            f"{color_by}, {color_map}; finite range {low:g} to {high:g}; "
            "missing/non-finite values grey"
        )
    entry = f"Colour Mesh Objects: {detail}; geometry unchanged"
    result = MeshData(
        mesh.vertices,
        mesh.faces,
        replace(
            mesh.state,
            history=mesh.state.history + (entry,),
            processing_history=mesh.state.processing_history + (entry,),
        ),
        mesh.face_object_ids,
        tuple(replace(item, color=colors[item.object_id]) for item in mesh.objects),
    )
    if progress is not None:
        progress.check_cancelled()
    return result


def filter_mesh_objects(
    mesh,
    property_name="mesh_volume_physical",
    minimum=0.0,
    maximum=1e12,
    keep="In range",
    *,
    progress=None,
):
    """Keep inclusive finite measurement ranges (or their outside complement)."""
    _require_mesh(mesh)
    if isinstance(minimum, (bool, np.bool_)) or isinstance(maximum, (bool, np.bool_)):
        raise ValueError("Mesh filter bounds must be finite numbers, not booleans.")
    minimum = int(minimum) if isinstance(minimum, (int, np.integer)) else float(minimum)
    maximum = int(maximum) if isinstance(maximum, (int, np.integer)) else float(maximum)
    if not np.isfinite((minimum, maximum)).all() or minimum > maximum:
        raise ValueError("Mesh filter bounds must be finite with minimum <= maximum.")
    if keep not in {"In range", "Outside range"}:
        raise ValueError("Mesh filtering requires In range or Outside range.")
    values = _object_measurements(mesh, property_name, progress)
    kept = []
    for item in mesh.objects:
        value = values.get(item.object_id, float("nan"))
        if not np.isfinite(value):
            continue
        inside = minimum <= value <= maximum
        if inside == (keep == "In range"):
            kept.append(item.object_id)
    entry = (
        f"Filter Mesh Objects: {property_name}; {keep}; "
        f"inclusive bounds {minimum:g} to {maximum:g}; "
        "fresh geometry measurements; missing/non-finite measurements excluded"
    )
    state = replace(
        mesh.state,
        history=mesh.state.history + (entry,),
        processing_history=mesh.state.processing_history + (entry,),
    )
    result = _select_faces(mesh, np.isin(mesh.face_object_ids, kept), state=state)
    if progress is not None:
        progress.check_cancelled()
    return result
