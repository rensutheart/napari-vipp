"""Bounded CPU presentation of actual assessment mesh geometry, without Qt.

Every supplied triangle is drawn, with no remeshing, welding, hole filling or
decimation. Fixed orthographic cameras and a source-grid frame make successive
views comparable when their grids agree. Triangle-centroid depth sorting is an
explicit display approximation, not a topology or geometric-accuracy test.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

from napari_vipp.core.grid import _unit_dimension_and_factor
from napari_vipp.core.meshes import MeshState, _mesh_calibration
from napari_vipp.core.metadata import ImageState
from napari_vipp.core.workflow_proposal_images import MAX_PREVIEW_EDGE, ImagePreview

MAX_MESH_PREVIEW_TRIANGLES = 200_000
MAX_MESH_PREVIEW_ELEMENTS = 4_000_000
MESH_PREVIEW_VIEWS = ("oblique A", "opposing oblique B")
_BACKGROUND = (15, 20, 29)


class MeshPreviewError(ValueError):
    """The actual geometry cannot honestly enter this bounded display boundary."""


def _calibrated_geometry(vertices, state, *, relative_to=None):
    if (
        type(state) is not MeshState
        or len(state.spatial_axes) != 3
        or tuple(axis.name.upper() for axis in state.spatial_axes) != ("Z", "Y", "X")
        or any(
            axis.type != "space" or not axis.is_explicit for axis in state.spatial_axes
        )
    ):
        raise MeshPreviewError(
            "Mesh previews require explicit carried Z/Y/X calibration."
        )
    try:
        factors, _unit, dimension = _mesh_calibration(state)
    except ValueError as exc:
        raise MeshPreviewError(str(exc)) from exc
    scales = np.array([axis.scale for axis in state.spatial_axes]) * factors
    offsets = np.array([axis.translation for axis in state.spatial_axes]) * factors
    with np.errstate(over="ignore", invalid="ignore"):
        if relative_to is not None:
            offsets = offsets - relative_to
        coordinates = vertices * scales + offsets
    if not np.isfinite(coordinates).all():
        raise MeshPreviewError(
            "Calibrated mesh coordinates exceed finite display range."
        )
    return coordinates, dimension


def _grid_corners(axes, shape, target_unit, dimension, relative_to):
    if (
        len(axes) != 3
        or len(shape) != 3
        or tuple(axis.name.upper() for axis in axes) != ("Z", "Y", "X")
        or any(axis.type != "space" or not axis.is_explicit for axis in axes)
        or any(size <= 0 for size in shape)
    ):
        raise MeshPreviewError(
            "A mesh display frame needs an explicit nonempty ZYX grid."
        )
    dimensions = [_unit_dimension_and_factor(axis.unit) for axis in axes]
    if any(value[0] != dimension for value in dimensions):
        raise MeshPreviewError(
            "The source frame and mesh have incompatible spatial units."
        )
    target_factor = _unit_dimension_and_factor(target_unit)[1]
    scales = np.array([axis.scale for axis in axes]) * np.array(
        [factor / target_factor for _, factor in dimensions]
    )
    offsets = np.array([axis.translation for axis in axes]) * np.array(
        [factor / target_factor for _, factor in dimensions]
    )
    corners = np.array(list(itertools.product(*[(-0.5, size - 0.5) for size in shape])))
    with np.errstate(over="ignore", invalid="ignore"):
        offsets = offsets - relative_to
        corners = corners * scales + offsets
    if not np.isfinite(corners).all():
        raise MeshPreviewError("Calibrated source bounds exceed finite display range.")
    return corners


def _camera(view_index):
    # Coordinates are conventional XYZ for the camera, not scientific cache data.
    towards = np.array((1.0, -1.0, 0.8)) * (1 if view_index == 0 else -1)
    towards /= np.linalg.norm(towards)
    right = np.cross(np.array((0.0, 0.0, 1.0)), towards)
    right /= np.linalg.norm(right)
    up = np.cross(towards, right)
    return right, up, towards


def render_mesh_preview(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_object_ids: np.ndarray,
    object_ids: Sequence[int],
    colors: Sequence[Sequence[float]],
    state: MeshState,
    *,
    label: str,
    view_index: int,
    source_state: ImageState | None = None,
    check_cancel: Callable[[], None] | None = None,
) -> ImagePreview:
    """Render a detached display PNG of all faces, retaining carried aspect.

    This function never reads source intensity, files, lazy arrays or the viewer.
    It uses at most a 512-square canvas and 200,000 triangles. No scientific
    array is changed; callers own output-preview consent and total PNG budgets.
    ``check_cancel`` may raise the caller's cancellation exception at bounded
    preparation and every 256-triangle drawing checkpoint.
    """
    check = check_cancel or (lambda: None)
    check()
    if (
        type(vertices) is not np.ndarray
        or type(faces) is not np.ndarray
        or type(face_object_ids) is not np.ndarray
        or vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or vertices.dtype.kind != "f"
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or faces.dtype.kind not in "iu"
        or face_object_ids.shape != (len(faces),)
        or face_object_ids.dtype.kind not in "iu"
        or len(faces) > MAX_MESH_PREVIEW_TRIANGLES
        or vertices.size + faces.size + face_object_ids.size > MAX_MESH_PREVIEW_ELEMENTS
        or type(view_index) is not int
        or view_index not in (0, 1)
        or not np.isfinite(vertices).all()
        or (faces.size and (faces.min() < 0 or faces.max() >= len(vertices)))
    ):
        raise MeshPreviewError(
            "Mesh previews require bounded finite actual triangle geometry."
        )
    palette = np.asarray(colors, dtype=np.float64)
    ids = np.asarray(object_ids, dtype=np.int64)
    if not len(ids) and not palette.size:
        palette = palette.reshape((0, 4))
    if (
        palette.shape != (len(ids), 4)
        or not np.isfinite(palette).all()
        or np.any((palette < 0) | (palette > 1))
        or len(np.unique(ids)) != len(ids)
        or not np.isin(face_object_ids, ids).all()
    ):
        raise MeshPreviewError(
            "Mesh display colours must match the actual face object IDs."
        )
    # Subtract one common carried origin before adding small voxel offsets.
    # This keeps a large shared absolute origin from erasing local detail; the
    # source grid uses the identical reference, so relative positions survive.
    _unused, dimension = _calibrated_geometry(np.empty((0, 3)), state)
    factors, _unit, _dimension = _mesh_calibration(state)
    relative_to = np.array([axis.translation for axis in state.spatial_axes]) * factors
    coordinates, dimension = _calibrated_geometry(
        vertices, state, relative_to=relative_to
    )
    target_unit = state.spatial_axes[2].unit
    if source_state is not None:
        corners = _grid_corners(
            source_state.axes, source_state.shape, target_unit, dimension, relative_to
        )
        framing = "fixed carried source-grid frame"
    else:
        corners = _grid_corners(
            state.spatial_axes, state.source_shape, target_unit, dimension, relative_to
        )
        framing = "fixed carried mesh-source-grid frame"
    lower, upper = corners.min(axis=0), corners.max(axis=0)
    if len(coordinates) and (
        np.any(coordinates < lower) or np.any(coordinates > upper)
    ):
        framing += (
            "; expanded to retain out-of-frame geometry (framing may differ by run)"
        )
        lower = np.minimum(lower, coordinates.min(axis=0))
        upper = np.maximum(upper, coordinates.max(axis=0))
        corners = np.array(list(itertools.product(*zip(lower, upper, strict=True))))
    center = lower * 0.5 + upper * 0.5
    # Center before projection; a large carried absolute origin does not erase
    # the local display detail. Any overflow is rejected, never normalized away.
    with np.errstate(over="ignore", invalid="ignore"):
        coordinates = (coordinates - center)[:, ::-1]
        corners = (corners - center)[:, ::-1]
    if not np.isfinite(coordinates).all() or not np.isfinite(corners).all():
        raise MeshPreviewError("Mesh display extent exceeds finite numeric range.")
    # A common display-only factor prevents overflow in camera dot products and
    # lighting normals without changing aspect, framing or scientific geometry.
    normalization = max(float(np.max(np.abs(corners))), np.finfo(float).tiny)
    coordinates = coordinates / normalization
    corners = corners / normalization
    right, up, towards = _camera(view_index)
    projected = np.column_stack((coordinates @ right, -(coordinates @ up)))
    frame = np.column_stack((corners @ right, -(corners @ up)))
    extent = frame.max(axis=0) - frame.min(axis=0)
    scale = (MAX_PREVIEW_EDGE - 64) / max(float(extent.max()), np.finfo(float).tiny)
    if not np.isfinite(scale):
        raise MeshPreviewError("Mesh display scale exceeds finite numeric range.")
    screen = projected * scale + MAX_PREVIEW_EDGE * 0.5
    check()
    image = Image.new("RGB", (MAX_PREVIEW_EDGE, MAX_PREVIEW_EDGE), _BACKGROUND)
    drawing = ImageDraw.Draw(image, "RGBA")
    if len(faces):
        triangles = coordinates[faces]
        depth = np.mean(triangles @ towards, axis=1)
        order = np.argsort(depth, kind="stable")
        # Double-sided flat lighting changes presentation only. The absolute
        # normal avoids interpreting winding as a claimed outside direction.
        normal = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        lengths = np.linalg.norm(normal, axis=1)
        shading = np.full(len(faces), 0.4)
        valid = lengths > 0
        shading[valid] += 0.6 * np.abs(normal[valid] @ towards) / lengths[valid]
        colors_by_face = np.empty((len(faces), 4), dtype=np.uint8)
        # Sorted lookup is bounded O(F log O), not one full-face scan per
        # display object; a fragmented candidate may have one object per face.
        sorted_objects = np.argsort(ids, kind="stable")
        palette_indices = sorted_objects[
            np.searchsorted(ids[sorted_objects], face_object_ids)
        ]
        face_palette = palette[palette_indices]
        colors_by_face[:, :3] = np.rint(
            face_palette[:, :3] * shading[:, None] * 255
        ).astype(np.uint8)
        colors_by_face[:, 3] = np.rint(face_palette[:, 3] * 255).astype(np.uint8)
        screen_triangles = screen[faces]
        for position, face_index in enumerate(order):
            if position % 256 == 0:
                check()
            drawing.polygon(
                [tuple(point) for point in screen_triangles[face_index]],
                fill=tuple(int(value) for value in colors_by_face[face_index]),
            )
    else:
        drawing.text((20, 20), "Empty mesh: no triangles", fill=(210, 220, 230, 255))
    # A fixed, labelled camera triad prevents two opposing views being mistaken
    # for changed anatomy. Labels are constants, never private mesh/source text.
    anchor = np.array((46.0, MAX_PREVIEW_EDGE - 46.0))
    for axis_index, (name, color) in enumerate(
        (
            ("X", (245, 108, 108, 255)),
            ("Y", (110, 224, 140, 255)),
            ("Z", (95, 188, 255, 255)),
        )
    ):
        direction = np.eye(3)[axis_index]
        tip = anchor + np.array((direction @ right, -(direction @ up))) * 27
        drawing.line((tuple(anchor), tuple(tip)), fill=color, width=2)
        drawing.text(tuple(tip + 2), name, fill=color)
    check()
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    calibration = (
        "recognized carried length calibration"
        if dimension == "length_micrometer"
        else "carried index/identical-unit scale, not asserted physical distances"
    )
    description = (
        f"{label}: actual mesh, {MESH_PREVIEW_VIEWS[view_index]}; fixed orthographic "
        f"camera; {len(faces)} / {len(faces)} triangles drawn with no geometry "
        "simplification. Carried compatible Z/Y/X scale and origin converted to "
        f"a common axis unit ({calibration}); aspect preserved; {framing}. "
        "Actual per-object RGBA colours with double-sided flat lighting; "
        "triangle-centroid back-to-front ordering and transparency are approximate "
        "and can obscure or misorder intersecting surfaces. Subpixel triangles "
        "may not be visible. No source-intensity overlay. This rendered view is "
        "not proof of mesh topology, closure, completeness or biological accuracy. "
        "Scientific geometry unchanged."
    )
    return ImagePreview(
        f"{label} · mesh {MESH_PREVIEW_VIEWS[view_index]}",
        buffer.getvalue(),
        description,
    )


__all__ = ["MeshPreviewError", "MESH_PREVIEW_VIEWS", "render_mesh_preview"]
