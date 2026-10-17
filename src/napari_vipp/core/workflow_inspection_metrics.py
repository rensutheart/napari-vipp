"""Bounded read-only descriptive evidence, never scientific processing or scores.

Callers retain responsibility for consent, source/grid identity, freshness and
semantic output selection. These helpers neither choose a region nor repair,
resample, cast or change a scientific array. Array membership is exact nonzero
membership; axis spans include gaps and are not perpendicular object thickness.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from napari_vipp.core.metadata import ImageState
from napari_vipp.core.workflow_proposals import _SAFE_UNITS

MAX_METRIC_ELEMENTS = 4_000_000
MAX_METRIC_BYTES = 64 * 1024 * 1024
MAX_METRIC_TRIANGLES = 200_000
_PUBLIC_KEY = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,39}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class InspectionMetricsError(ValueError):
    """The requested evidence has no supported exact bounded interpretation."""


class InspectionMetricsCancelled(InspectionMetricsError):
    """Cancelled evidence must not be published as complete."""


def _check_cancel(cancelled):
    if cancelled is not None and cancelled():
        raise InspectionMetricsCancelled("Evidence preparation was cancelled.")


def _membership(data, cancelled):
    if (
        type(data) is not np.ndarray
        or data.ndim not in {2, 3}
        or not data.size
        or data.size > MAX_METRIC_ELEMENTS
        or data.nbytes > MAX_METRIC_BYTES
        or data.dtype.kind not in "biuf"
        or data.dtype.itemsize > 8
    ):
        raise InspectionMetricsError(
            "Membership evidence requires a non-empty real YX/ZYX array of at "
            "most 4,000,000 elements and 64 MiB."
        )
    _check_cancel(cancelled)
    if not np.isfinite(data).all():
        raise InspectionMetricsError("Non-finite values cannot define foreground.")
    return data != 0


def foreground_change_summary(
    left: np.ndarray,
    right: np.ndarray,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> dict:
    """Compare aligned caller-selected masks, without treating either as truth.

    The caller MUST check matching source, physical grid and mask/label meaning
    before calling. Equal shape alone is not evidence of a common physical grid.
    Empty-reference retention and empty-union IoU are undefined, not perfect.
    """
    a, b = _membership(left, cancelled), _membership(right, cancelled)
    if a.shape != b.shape:
        raise InspectionMetricsError("Membership comparison requires equal shapes.")
    _check_cancel(cancelled)
    left_count, right_count = int(a.sum()), int(b.sum())
    intersection = int(np.count_nonzero(a & b))
    union = left_count + right_count - intersection
    return {
        "sampling": "exact full nonzero membership",
        "left_foreground_count": left_count,
        "right_foreground_count": right_count,
        "intersection_count": intersection,
        "added_count": right_count - intersection,
        "removed_count": left_count - intersection,
        "union_count": union,
        "left_retained_fraction": intersection / left_count if left_count else None,
        "intersection_over_union": intersection / union if union else None,
        "meaning": (
            "Descriptive agreement on a caller-verified common source/grid, "
            "not ground truth, accuracy or a preference for less foreground."
        ),
    }


def foreground_span_summary(
    data: np.ndarray,
    state: ImageState,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> dict:
    """Summarize first-to-last nonzero cell spans on each native spatial axis.

    Statistics use only nonempty axis-parallel lines and include gaps between
    foreground cells. Native axis calibration is carried, never inferred from
    another axis. No line profiles, coordinates or raw image values are emitted.
    """
    foreground = _membership(data, cancelled)
    if (
        not isinstance(state, ImageState)
        or not state.axes_explicit
        or state.axis_order not in {"YX", "ZYX"}
        or tuple(state.shape) != tuple(data.shape)
        or np.dtype(state.dtype) != data.dtype
        or any(
            axis.type != "space" or axis.unit not in _SAFE_UNITS for axis in state.axes
        )
    ):
        raise InspectionMetricsError(
            "Spans require matching explicit spatial metadata."
        )
    axes = []
    for index, axis in enumerate(state.axes):
        _check_cancel(cancelled)
        present = foreground.any(axis=index)
        first = foreground.argmax(axis=index)
        last = foreground.shape[index] - np.flip(foreground, axis=index).argmax(
            axis=index
        )
        spans = (last - first)[present]
        quantiles = (
            np.quantile(spans, [0, 0.25, 0.5, 0.75, 1]).tolist()
            if spans.size
            else [None] * 5
        )
        names = ("minimum", "q25", "median", "q75", "maximum")
        calibrated = {
            key: None if value is None else value * axis.scale
            for key, value in zip(names, quantiles, strict=True)
        }
        calibration_overflow = any(
            value is not None and not np.isfinite(value)
            for value in calibrated.values()
        )
        axes.append(
            {
                "axis": axis.name,
                "nonempty_line_count": int(spans.size),
                "native_cell_span": dict(zip(names, quantiles, strict=True)),
                "calibrated_span": dict.fromkeys(names)
                if calibration_overflow
                else calibrated,
                "calibration_status": "numeric overflow; unavailable"
                if calibration_overflow
                else "carried axis scale",
                "axis_scale": axis.scale,
                "axis_unit": axis.unit or "unspecified",
            }
        )
    _check_cancel(cancelled)
    return {
        "sampling": "exact full nonzero membership; linear-interpolated quantiles",
        "axes": axes,
        "meaning": (
            "Axis-parallel first-to-last foreground cell spans, including gaps; "
            "not perpendicular thickness, boundary accuracy or a thinness score."
        ),
    }


def mesh_topology_summary(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> dict:
    """Count indexed triangle connectivity without merging or repairing geometry.

    Components connect triangles sharing a complete indexed edge, not merely a
    vertex. Euler uses only referenced vertices. Duplicate facets or repeated
    vertex indices invalidate its surface interpretation; the raw V-E+F count
    is still reported. No genus, geometric degeneracy, self-intersection or
    biological completeness test is implied.
    """
    if (
        type(vertices) is not np.ndarray
        or type(faces) is not np.ndarray
        or vertices.ndim != 2
        or faces.ndim != 2
        or vertices.shape[1] != 3
        or faces.shape[1] != 3
        or vertices.dtype.kind not in "iuf"
        or vertices.dtype.itemsize > 8
        or faces.dtype.kind not in "iu"
        or faces.dtype.itemsize > 8
        or len(faces) > MAX_METRIC_TRIANGLES
        or vertices.size + faces.size > MAX_METRIC_ELEMENTS
        or vertices.nbytes + faces.nbytes > MAX_METRIC_BYTES
    ):
        raise InspectionMetricsError("Mesh topology exceeds supported geometry bounds.")
    _check_cancel(cancelled)
    if not np.isfinite(vertices).all() or (
        faces.size and (int(faces.min()) < 0 or int(faces.max()) >= len(vertices))
    ):
        raise InspectionMetricsError("Mesh vertices/triangle indices are invalid.")
    directed = np.concatenate((faces[:, :2], faces[:, 1:], faces[:, [2, 0]]))
    signs = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    edges = np.sort(directed, axis=1)
    _check_cancel(cancelled)
    unique_edges, inverse, counts = np.unique(
        edges, axis=0, return_inverse=True, return_counts=True
    )
    used_count = int(len(np.unique(faces)))
    winding = np.bincount(inverse, weights=signs, minlength=len(unique_edges))
    repeated = int(
        np.count_nonzero(
            (faces[:, 0] == faces[:, 1])
            | (faces[:, 1] == faces[:, 2])
            | (faces[:, 2] == faces[:, 0])
        )
    )
    duplicates = len(faces) - len(np.unique(np.sort(faces, axis=1), axis=0))
    if len(faces):
        # An edge's incident triangles form a star, not an unbounded clique.
        face_ids = np.tile(np.arange(len(faces)), 3)
        first = np.full(len(unique_edges), len(faces), dtype=np.int64)
        np.minimum.at(first, inverse, face_ids)
        adjacency = sparse.coo_matrix(
            (np.ones(len(face_ids), dtype=np.uint8), (first[inverse], face_ids)),
            shape=(len(faces), len(faces)),
        ).tocsr()
        _check_cancel(cancelled)
        components = int(
            connected_components(adjacency, directed=False, return_labels=False)
        )
    else:
        components = 0
    _check_cancel(cancelled)
    euler_count = used_count - len(unique_edges) + len(faces)
    return {
        "used_vertex_count": used_count,
        "unused_vertex_count": int(len(vertices)) - used_count,
        "unique_edge_count": int(len(unique_edges)),
        "boundary_edge_count": int(np.count_nonzero(counts == 1)),
        "nonmanifold_edge_count": int(np.count_nonzero(counts > 2)),
        "inconsistent_winding_edge_count": int(
            np.count_nonzero((counts == 2) & (winding != 0))
        ),
        "repeated_vertex_triangle_count": repeated,
        "duplicate_triangle_count": int(duplicates),
        "surface_component_count": components,
        "component_definition": (
            "Triangle components joined by a shared complete indexed edge."
        ),
        "euler_count_v_minus_e_plus_f": int(euler_count),
        "euler_characteristic": None if repeated or duplicates else int(euler_count),
        "euler_status": "invalid_indexed_facets"
        if repeated or duplicates
        else "indexed_complex_count",
        "topology_limitations": (
            "Coincident vertex positions are not merged. No geometric degeneracy, "
            "self-intersection, genus or anatomical correctness is established. "
            "Closed edges alone do not imply a simple solid or one surface."
        ),
    }


def public_role_binding(
    role: str,
    node_key: str,
    node_alias: str,
    port: int,
    output_sha256: str,
) -> dict:
    """Bind explicit trusted semantic identity; never infer it from array order.

    Roles are explicit bounded public descriptions (not inferred local names).
    Recipe keys/aliases are identifiers, never runtime IDs or paths. The hash
    binds this mapping, not its anatomical interpretation.
    """
    if (
        type(role) is not str
        or not role.strip()
        or len(role) > 80
        or any(not character.isprintable() or character in "/\\" for character in role)
        or any(
            type(value) is not str or not _PUBLIC_KEY.fullmatch(value)
            for value in (node_key, node_alias)
        )
    ):
        raise InspectionMetricsError("Semantic bindings require bounded public keys.")
    if type(port) is not int or not 0 <= port <= 255:
        raise InspectionMetricsError(
            "Semantic binding port must be explicit and bounded."
        )
    if type(output_sha256) is not str or not _SHA256.fullmatch(output_sha256):
        raise InspectionMetricsError(
            "Semantic binding requires the exact output SHA-256."
        )
    binding = {
        "role": role,
        "node_key": node_key,
        "node_alias": node_alias,
        "port": port,
        "output_sha256": output_sha256,
    }
    binding["binding_sha256"] = hashlib.sha256(
        json.dumps(binding, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return binding
