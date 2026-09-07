"""Explicit, non-destructive refinement of calibrated triangle objects.

Smoothing uses the two-pass filter of Taubin (SIGGRAPH 1995,
doi:10.1145/218380.218473), with fixed inverse initial physical-edge-length
weights. Simplification uses fast-simplification's quadric-error edge-collapse
implementation. Neither operation promises exact volume or shape preservation;
both return a new mesh and leave upstream geometry available for comparison.
"""

from __future__ import annotations

import importlib.metadata
import threading
from dataclasses import replace

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from napari_vipp.core.meshes import MeshData, _mesh_calibration

# The provider stores mesh buffers in process-global native state. Distinct
# workflow workers must not interleave load/simplify/return calls.
_SIMPLIFICATION_LOCK = threading.Lock()


def _check_progress(progress, current=None, total=None, message=""):
    if progress is not None:
        progress.check_cancelled()
        if current is not None:
            progress.report(current, total, message)


def _finite_number(value, name, low, high, *, include_low=True):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number, not a boolean.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    lower_ok = result >= low if include_low else result > low
    if not np.isfinite(result) or not lower_ok or result > high:
        relation = "at least" if include_low else "greater than"
        raise ValueError(f"{name} must be {relation} {low} and at most {high}.")
    return result


def _boundary_flag(value):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError("Preserve boundary must be a boolean.")
    return bool(value)


def _physical_frame(mesh):
    if not isinstance(mesh, MeshData):
        raise TypeError("Mesh refinement requires a 3D mesh, not an image or table.")
    axes = mesh.state.spatial_axes
    if tuple(axis.name.lower() for axis in axes) != ("z", "y", "x") or any(
        axis.type != "space" for axis in axes
    ):
        raise ValueError("Mesh refinement requires explicit Z/Y/X spatial calibration.")
    factors, unit, _dimension = _mesh_calibration(mesh.state)
    scales = np.array([axis.scale for axis in axes], dtype=float) * factors
    origins = np.array([axis.translation for axis in axes], dtype=float) * factors
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Mesh refinement requires finite, positive spatial scales.")
    if not np.isfinite(origins).all():
        raise ValueError("Mesh refinement requires finite spatial origins.")
    return scales, unit


def _surface_contract(vertices, faces):
    """Reject unsafe triangles instead of repairing a scientific surface.

    Edge incidence/winding checks are not a general self-intersection test.
    Return the unique edges, counts, closed flag and Euler characteristic.
    """
    if not len(faces):
        return np.empty((0, 2), np.int64), np.empty(0, np.int64), False, 0
    if not np.isfinite(vertices).all():
        raise ValueError("non-finite coordinates")
    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    lengths = np.linalg.norm(normals, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError("finite, non-degenerate triangles are required")
    duplicate_count = len(faces) - len(np.unique(np.sort(faces, axis=1), axis=0))
    if duplicate_count:
        noun = "face" if duplicate_count == 1 else "faces"
        raise ValueError(
            f"duplicate triangles ({duplicate_count} repeated {noun} "
            "with the same three vertex IDs)"
        )
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges, inverse, counts = np.unique(
        np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    orientation = np.bincount(
        inverse, weights=np.where(directed[:, 0] < directed[:, 1], 1, -1)
    )
    if np.any(counts > 2) or np.any((counts == 2) & (orientation != 0)):
        raise ValueError("consistently wound manifold edges are required")
    euler = len(np.unique(faces)) - len(edges) + len(faces)
    return edges, counts, bool(np.all(counts == 2)), euler


def _refinement_failure(operation, object_id, detail, *, settings=None, component=None):
    """Distinguish an invalid input from a rejected, newly generated candidate."""
    location = f"{operation}: object {object_id}"
    if component is not None:
        location += f", component {component}"
    detail = str(detail).rstrip(".")
    if settings is None:
        message = (
            f"{location} has an invalid input mesh: {detail}. "
            "Changing this node's settings will not repair the input. "
            "Inspect the upstream mesh or regenerate it from the source mask/labels. "
        )
    else:
        advice = (
            "Increase Triangles to keep (%) or try lower Aggressiveness. "
            "The safe reduction depends on the mesh. "
            if operation == "Simplify Mesh"
            else "Reduce Strength or Iterations and retry. "
        )
        message = (
            f"{location} passed input validation, but {settings} produced an "
            f"invalid result: {detail}. {advice}"
        )
    return ValueError(message + "No new mesh was produced; the input is unchanged.")


def _input_surface_contract(vertices, faces, operation, object_id):
    try:
        return _surface_contract(vertices, faces)
    except ValueError as exc:
        raise _refinement_failure(operation, object_id, exc) from exc


def _signed_volume(vertices, faces):
    centred = vertices - vertices.mean(axis=0)
    triangles = centred[faces]
    return (
        np.einsum(
            "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
        ).sum()
        / 6
    )


def _check_refined_topology(before_vertices, before_faces, after_vertices, after_faces):
    _edges, _counts, was_closed, before_euler = _surface_contract(
        before_vertices, before_faces
    )
    _edges, _counts, is_closed, after_euler = _surface_contract(
        after_vertices, after_faces
    )
    if not len(after_faces) or was_closed != is_closed or before_euler != after_euler:
        raise ValueError(
            "refinement changed the surface topology or removed a component"
        )
    if was_closed:
        before_volume = _signed_volume(before_vertices, before_faces)
        after_volume = _signed_volume(after_vertices, after_faces)
        if (
            not np.isfinite(after_volume)
            or after_volume == 0
            or np.sign(after_volume) != np.sign(before_volume)
        ):
            raise ValueError("refinement collapsed or inverted an enclosed surface")


def _component_indices(vertex_count, faces, edges):
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    cols = np.concatenate((edges[:, 1], edges[:, 0]))
    adjacency = sparse.csr_matrix(
        (np.ones(len(rows), np.uint8), (rows, cols)), shape=(vertex_count, vertex_count)
    )
    _count, labels = connected_components(adjacency, directed=False)
    face_labels = labels[faces[:, 0]]
    # Faces joined through an edge or vertex stay in their original component;
    # no spatial proximity or tolerance is used to weld distinct vertices.
    ordered = np.argsort(face_labels, kind="stable")
    ordered_labels = face_labels[ordered]
    bounds = np.concatenate(
        ([0], np.flatnonzero(np.diff(ordered_labels)) + 1, [len(ordered)])
    )
    for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
        yield ordered[start:stop]


def _assemble(mesh, pieces, description):
    vertices, faces, ids = [], [], []
    offset = 0
    for points, triangles, object_id in pieces:
        vertices.append(points)
        faces.append(triangles + offset)
        ids.append(np.full(len(triangles), object_id, np.int64))
        offset += len(points)
    state = replace(
        mesh.state,
        history=mesh.state.history + (description,),
        processing_history=mesh.state.processing_history + (description,),
    )
    return MeshData(
        np.concatenate(vertices) if vertices else np.empty((0, 3), float),
        np.concatenate(faces) if faces else np.empty((0, 3), np.int64),
        state,
        np.concatenate(ids) if ids else np.empty(0, np.int64),
        mesh.objects,
    )


def smooth_mesh(
    mesh, iterations=10, strength=0.5, preserve_boundary=True, *, progress=None
):
    """Taubin-style smoothing per object, in common physical Z/Y/X units.

    Each iteration applies lambda=0.5*strength then
    mu=-lambda/(1-0.1*lambda), using one fixed inverse-edge-length weight
    matrix from the original calibrated geometry. Boundary vertices are locked
    by default. The method reduces shrinkage but does not conserve volume or
    guarantee freedom from self-intersection. No faces are added or removed.
    """
    from napari_vipp.core.mesh_objects import iter_mesh_objects

    _check_progress(progress)
    if (
        isinstance(iterations, (bool, np.bool_))
        or not isinstance(iterations, (int, np.integer))
        or not 1 <= iterations <= 1000
    ):
        raise ValueError("Smoothing iterations must be an integer from 1 to 1000.")
    strength = _finite_number(strength, "Smoothing strength", 0, 1)
    preserve_boundary = _boundary_flag(preserve_boundary)
    scales, unit = _physical_frame(mesh)
    lam = 0.5 * strength
    mu = -lam / (1 - 0.1 * lam)
    total = max(mesh.object_count * int(iterations), 1)
    pieces = []
    for object_index, part in enumerate(iter_mesh_objects(mesh)):
        _check_progress(progress)
        object_id = part.objects[0].object_id
        settings = f"Strength {strength:g}, Iterations {iterations}"
        # Omit the global origin and centre before arithmetic. Refinement is
        # translation invariant and remains stable for very distant origins.
        centre = part.vertices.mean(axis=0)
        points = (part.vertices - centre) * scales
        edges, counts, _closed, _euler = _input_surface_contract(
            points, part.faces, "Smooth Mesh", object_id
        )
        distances = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
        weights = 1 / distances
        rows = np.concatenate((edges[:, 0], edges[:, 1]))
        cols = np.concatenate((edges[:, 1], edges[:, 0]))
        adjacency = sparse.csr_matrix(
            (np.tile(weights, 2), (rows, cols)), shape=(len(points), len(points))
        )
        degrees = np.asarray(adjacency.sum(axis=1)).ravel()
        if not np.isfinite(degrees).all() or np.any(degrees <= 0):
            raise ValueError("Mesh smoothing cannot resolve finite neighbour weights.")
        weighted_mean = sparse.diags(1 / degrees) @ adjacency
        locked = np.unique(edges[counts == 1]) if preserve_boundary else []
        work = points.copy()
        for iteration in range(iterations):
            _check_progress(
                progress,
                object_index * iterations + iteration,
                total,
                f"Smoothing object {part.objects[0].object_id}; "
                f"iteration {iteration + 1}/{iterations}",
            )
            if strength:
                for coefficient in (lam, mu):
                    displacement = weighted_mean @ work - work
                    displacement[locked] = 0
                    work += coefficient * displacement
                if not np.isfinite(work).all():
                    raise ValueError("Mesh smoothing exceeded finite numeric range.")
        for component_index, face_indices in enumerate(
            _component_indices(len(points), part.faces, edges), start=1
        ):
            component = part.faces[face_indices]
            used = np.unique(component)
            try:
                _check_refined_topology(
                    points[used],
                    np.searchsorted(used, component),
                    work[used],
                    np.searchsorted(used, component),
                )
            except ValueError as exc:
                raise _refinement_failure(
                    "Smooth Mesh",
                    object_id,
                    exc,
                    settings=settings,
                    component=component_index,
                ) from exc
        restored = work / scales + centre if strength else part.vertices.copy()
        if preserve_boundary:
            restored[locked] = part.vertices[locked]
        pieces.append((restored, part.faces, part.objects[0].object_id))
    description = (
        f"Smooth Mesh: Taubin-style inverse initial physical-edge-length weights; "
        f"{iterations} iterations; strength {strength:g}; "
        f"lambda {lam:g}; mu {mu:.12g}; "
        f"pass band 0.1; preserve boundary={preserve_boundary}; physical unit={unit}; "
        "approximate geometry; no exact volume preservation; original mesh unchanged"
    )
    result = _assemble(mesh, pieces, description)
    _check_progress(
        progress, total, total, "Smoothed mesh ready; original retained upstream"
    )
    return result


def _simplify_component(
    points, faces, target, aggressiveness, preserve_boundary, progress
):
    import fast_simplification

    while not _SIMPLIFICATION_LOCK.acquire(timeout=0.1):
        _check_progress(progress)
    try:
        _check_progress(progress)
        # Owned buffers also protect callers from future changes in the native
        # provider's input-ownership contract.
        result = fast_simplification.simplify(
            points.copy(),
            faces.copy(),
            target_count=target,
            agg=aggressiveness,
            preserve_border=preserve_boundary,
        )
        _check_progress(progress)
        return np.array(result[0], dtype=float, copy=True), np.array(
            result[1], copy=True
        )
    finally:
        _SIMPLIFICATION_LOCK.release()


def simplify_mesh(
    mesh,
    target_percent=50.0,
    aggressiveness=7.0,
    preserve_boundary=True,
    *,
    progress=None,
):
    """Approximate QEM triangle reduction, independently per object/component.

    target_percent is the percentage to KEEP (not remove). Component minimums,
    locked boundaries and provider quality checks may leave more triangles than
    requested. No component is intentionally erased. Target=100 is an exact
    geometry no-op. Native simplification notices cancellation between library
    calls, not during one component. No welding, repair or geometric union.
    """
    from napari_vipp.core.mesh_objects import iter_mesh_objects

    _check_progress(progress)
    target_percent = _finite_number(
        target_percent, "Triangles to keep (%)", 0, 100, include_low=False
    )
    aggressiveness = _finite_number(
        aggressiveness, "Simplification aggressiveness", 0, 10
    )
    preserve_boundary = _boundary_flag(preserve_boundary)
    scales, unit = _physical_frame(mesh)
    total = max(mesh.object_count, 1)
    pieces = []
    for object_index, part in enumerate(iter_mesh_objects(mesh)):
        object_id = part.objects[0].object_id
        settings = (
            f"keeping {target_percent:g}% of triangles "
            f"(Aggressiveness {aggressiveness:g})"
        )
        centre = part.vertices.mean(axis=0)
        points = (part.vertices - centre) * scales
        edges, _counts, _closed, _euler = _input_surface_contract(
            points, part.faces, "Simplify Mesh", object_id
        )
        for component_index, face_indices in enumerate(
            _component_indices(len(points), part.faces, edges), start=1
        ):
            _check_progress(
                progress,
                object_index,
                total,
                f"Simplifying object {part.objects[0].object_id}; this native call "
                "finishes before cancellation can be observed",
            )
            original_faces = part.faces[face_indices]
            used, inverse = np.unique(original_faces, return_inverse=True)
            faces = inverse.reshape((-1, 3))
            vertices = points[used]
            component_edges, edge_counts, closed, _euler = _surface_contract(
                vertices, faces
            )
            target = max(
                4 if closed else 1, int(np.ceil(len(faces) * target_percent / 100))
            )
            if target >= len(faces):
                pieces.append((part.vertices[used], faces, part.objects[0].object_id))
                continue
            reduced_points, reduced_faces = _simplify_component(
                vertices, faces, target, aggressiveness, preserve_boundary, progress
            )
            if (
                reduced_points.ndim != 2
                or reduced_points.shape[1] != 3
                or reduced_faces.ndim != 2
                or reduced_faces.shape[1] != 3
                or not np.issubdtype(reduced_faces.dtype, np.integer)
                or (
                    reduced_faces.size
                    and (
                        reduced_faces.min() < 0
                        or reduced_faces.max() >= len(reduced_points)
                    )
                )
                or len(reduced_faces) > len(faces)
            ):
                raise _refinement_failure(
                    "Simplify Mesh",
                    object_id,
                    "invalid triangle geometry",
                    settings=settings,
                    component=component_index,
                )
            try:
                _check_refined_topology(vertices, faces, reduced_points, reduced_faces)
            except ValueError as exc:
                raise _refinement_failure(
                    "Simplify Mesh",
                    object_id,
                    exc,
                    settings=settings,
                    component=component_index,
                ) from exc
            restored = reduced_points / scales + centre
            if preserve_boundary:
                boundary_indices = np.unique(component_edges[edge_counts == 1])
                boundary = vertices[boundary_indices]
                present = {
                    tuple(point): index for index, point in enumerate(reduced_points)
                }
                if any(tuple(point) not in present for point in boundary):
                    raise _refinement_failure(
                        "Simplify Mesh",
                        object_id,
                        "a locked boundary vertex moved",
                        settings=settings,
                        component=component_index,
                    )
                # Keep locked coordinates bit-for-bit in the authored frame,
                # not just up to a scale/division round trip.
                for index, point in zip(boundary_indices, boundary, strict=True):
                    restored[present[tuple(point)]] = part.vertices[used[index]]
            pieces.append((restored, reduced_faces, part.objects[0].object_id))
    count = sum(len(faces) for _vertices, faces, _object_id in pieces)
    try:
        provider_version = importlib.metadata.version("fast-simplification")
    except importlib.metadata.PackageNotFoundError:
        # Empty and 100%-keep requests do not invoke the native provider.
        provider_version = "not-installed (no native simplification invoked)"
    description = (
        f"Simplify Mesh: fast-simplification {provider_version}; "
        "quadric-error edge collapse; "
        f"target keep {target_percent:g}%; aggressiveness {aggressiveness:g}; "
        f"preserve boundary={preserve_boundary}; physical unit={unit}; "
        f"{len(mesh.faces)} to {count} triangles achieved; per object/component; "
        "approximate geometry; no exact volume preservation; original mesh unchanged"
    )
    result = _assemble(mesh, pieces, description)
    _check_progress(
        progress, total, total, f"Simplified mesh ready: {count:,} triangles"
    )
    return result
