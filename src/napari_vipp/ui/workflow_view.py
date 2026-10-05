"""Runtime-only napari view snapshots and selected-layer framing.

These helpers never change layer data, scientific metadata, or workflow files.
The widget owns when a tab's snapshot is captured and restored; ordinary node
refreshes should not call them and therefore retain the user's current view.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from napari_vipp.ui.napari_compat import viewer_camera


@dataclass(frozen=True, slots=True)
class ViewerState:
    """An immutable, deliberately non-serialized presentation snapshot."""

    order: tuple[int, ...] | None = None
    ndisplay: int | None = None
    point: tuple[float, ...] | None = None
    center: tuple[float, ...] | None = None
    zoom: float | None = None
    angles: tuple[float, ...] | None = None
    perspective: float | None = None


def _get(obj, name, default=None):
    try:
        return getattr(obj, name, default)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return default


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _numbers(value, *, lengths=None):
    try:
        result = tuple(_number(item) for item in value)
    except TypeError:
        return None
    if not result or any(item is None for item in result):
        return None
    if lengths is not None and len(result) not in lengths:
        return None
    return result


def _camera(viewer):
    try:
        return viewer_camera(viewer)
    except (AttributeError, RuntimeError):
        return None


def _set(obj, name, value):
    if value is None or _get(obj, name) is None:
        return False
    try:
        setattr(obj, name, value)
        return True
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def _order(dims):
    values = _numbers(_get(dims, "order"))
    if values is None or any(value != int(value) for value in values):
        return None
    result = tuple(int(value) for value in values)
    return result if set(result) == set(range(len(result))) else None


def capture_viewer_state(viewer) -> ViewerState | None:
    """Capture available view fields, tolerating minimal/part-built viewers."""
    dims, camera = _get(viewer, "dims"), _camera(viewer)
    ndisplay = _get(dims, "ndisplay")
    zoom = _number(_get(camera, "zoom"))
    state = ViewerState(
        order=_order(dims),
        ndisplay=int(ndisplay) if ndisplay in (2, 3) else None,
        point=_numbers(_get(dims, "point")),
        center=_numbers(_get(camera, "center"), lengths=(2, 3)),
        zoom=zoom if zoom is not None and zoom > 0 else None,
        angles=_numbers(_get(camera, "angles"), lengths=(3,)),
        perspective=_number(_get(camera, "perspective")),
    )
    available = any(
        getattr(state, field) is not None
        for field in (
            "order",
            "ndisplay",
            "point",
            "center",
            "zoom",
            "angles",
            "perspective",
        )
    )
    return state if available else None


def _ndim(dims):
    number = _number(_get(dims, "ndim"))
    if number is not None and number > 0 and number == int(number):
        return int(number)
    for name in ("order", "point", "nsteps"):
        try:
            size = len(_get(dims, name))
        except TypeError:
            continue
        if size:
            return size
    return 0


def _adapt_order(order, ndim):
    """Napari aligns lower-rank layers to the trailing global dimensions."""
    if order is None or not ndim:
        return None
    offset = ndim - len(order)
    mapped = tuple(axis + offset for axis in order if axis + offset >= 0)
    result = tuple(range(max(offset, 0))) + mapped
    return result if set(result) == set(range(ndim)) else None


def _adapt_point(point, dims, ndim):
    if point is None or not ndim:
        return None
    current = _numbers(_get(dims, "point")) or (0.0,) * ndim
    result = list(current[-ndim:])
    result = [0.0] * (ndim - len(result)) + result
    offset = ndim - len(point)
    ranges = _get(dims, "range", ())
    for axis in range(ndim):
        source_axis = axis - offset
        if 0 <= source_axis < len(point):
            result[axis] = point[source_axis]
        try:
            low, high = float(ranges[axis][0]), float(ranges[axis][1])
            if math.isfinite(low) and math.isfinite(high) and low <= high:
                result[axis] = min(max(result[axis], low), high)
        except (IndexError, TypeError, ValueError):
            pass
    return tuple(result)


def restore_viewer_state(viewer, state: ViewerState | None, *, layer=None) -> bool:
    """Restore valid dimensions first, then camera after synchronous callbacks.

    Dropped/added leading dimensions are right-aligned. World-coordinate slice
    points are clamped to current ranges, not copied into an obsolete grid.
    If a target layer is supplied, its nondisplayed axes are also clamped to
    sample-centre bounds so a smaller replacement cannot show an empty slice.
    Napari's ndisplay callbacks can change camera and point, so camera is last.
    """
    if not isinstance(state, ViewerState):
        return False
    dims, camera = _get(viewer, "dims"), _camera(viewer)
    ndim = _ndim(dims)
    changed = _set(dims, "order", _adapt_order(state.order, ndim))
    mode = 2 if 0 < ndim < 3 and state.ndisplay == 3 else state.ndisplay
    changed = _set(dims, "ndisplay", mode) or changed
    changed = _set(dims, "point", _adapt_point(state.point, dims, ndim)) or changed
    if layer is not None:
        changed = _clamp_layer_slice_point(viewer, layer) or changed
    for field in ("center", "zoom", "angles", "perspective"):
        changed = _set(camera, field, getattr(state, field)) or changed
    return changed


def _canvas_size(viewer):
    canvas = _get(viewer, "canvas")
    viewbox = _get(canvas, "viewbox_size")
    if callable(viewbox):
        try:
            size = _numbers(viewbox(_get(viewer, "layers")), lengths=(2,))
            if size is not None and min(size) > 0:
                return np.asarray(size)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    # Older supported napari has no public Canvas model. Bound its historical
    # canvas-size fallback here rather than traversing any Qt/vispy object graph.
    for candidate in (_get(canvas, "size"), _get(viewer, "_canvas_size")):
        size = _numbers(candidate, lengths=(2,))
        if size is not None and min(size) > 0:
            return np.asarray(size)
    return np.asarray((800.0, 600.0))


def _layer_bounds(layer, *, pixel_edges=True):
    extent = _get(layer, "extent")
    try:
        bounds = np.array(extent.world, dtype=float, copy=True)
    except (AttributeError, TypeError, ValueError):
        return None
    if (
        bounds.ndim != 2
        or bounds.shape[0] != 2
        or not bounds.shape[1]
        or not np.isfinite(bounds).all()
        or np.any(bounds[1] < bounds[0])
    ):
        return None
    metadata = _get(layer, "metadata", {})
    display_kind = metadata.get("display_kind") if isinstance(metadata, dict) else None
    # Image/Labels extent.world describes sample centres. Include half a pixel
    # on each side; other geometry's public world extent is already its bounds.
    from napari.layers import Image, Labels

    if pixel_edges and (
        isinstance(layer, (Image, Labels)) or display_kind in {"image", "labels"}
    ):
        ndim = bounds.shape[1]
        transform = _get(layer, "data_to_world")
        widths = None
        if callable(transform):
            try:
                zero = np.asarray(transform((0.0,) * ndim), dtype=float)
                basis = [
                    np.asarray(transform(tuple(float(i == axis) for i in range(ndim))))
                    - zero
                    for axis in range(ndim)
                ]
                widths = np.sum(np.abs(basis), axis=0)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        if widths is None:
            widths = np.asarray(_get(extent, "step", (1.0,) * ndim), dtype=float)
        if widths.shape == (ndim,) and np.isfinite(widths).all():
            bounds[0] -= np.abs(widths) / 2
            bounds[1] += np.abs(widths) / 2
    return bounds


def _projected_size(camera, size):
    up = _numbers(_get(camera, "up_direction"), lengths=(3,))
    view = _numbers(_get(camera, "view_direction"), lengths=(3,))
    if up is None or view is None:
        from scipy.spatial.transform import Rotation

        angles = _numbers(_get(camera, "angles"), lengths=(3,)) or (0.0, 0.0, 0.0)
        rotation = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
        view, up = -rotation[0], -rotation[1]
    right = np.cross(view, up)
    return np.asarray((np.dot(np.abs(up), size), np.dot(np.abs(right), size)))


def _scene_bounds(viewer, layer, bounds):
    """Match public layer coordinates to the units napari renders in."""
    missing = object()
    units = _get(_get(viewer, "layers"), "units", missing)
    if units is missing:
        units = _get(_get(viewer, "dims"), "units")
    layer_units = _get(layer, "units")
    # None means that napari found incompatible scene units and renders raw
    # values. Older viewers without unit support also use unconverted values.
    if units is None or layer_units is None:
        return bounds
    try:
        ndim = bounds.shape[1]
        scene_units = tuple(units)[-ndim:]
        source_units = tuple(layer_units)[-ndim:]
        if len(scene_units) != ndim or len(source_units) != ndim:
            return None
        factors = np.asarray(
            [
                float((1.0 * source).to(target).magnitude)
                for source, target in zip(source_units, scene_units, strict=True)
            ]
        )
    except (AttributeError, TypeError, ValueError):
        return None
    if not np.isfinite(factors).all() or np.any(factors <= 0):
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        converted = bounds * factors
    return converted if np.isfinite(converted).all() else None


def _clamp_layer_slice_point(viewer, layer):
    """Clamp target slice axes only; displayed point values are not moved."""
    bounds = _layer_bounds(layer, pixel_edges=False)
    if bounds is None:
        return False
    bounds = _scene_bounds(viewer, layer, bounds)
    if bounds is None:
        return False
    dims = _get(viewer, "dims")
    ndim, layer_ndim = _ndim(dims), bounds.shape[1]
    point = _numbers(_get(dims, "point"))
    if point is None or len(point) != ndim or layer_ndim > ndim:
        return False
    mode = _get(dims, "ndisplay", 2)
    mode = mode if mode in (2, 3) else 2
    order = _order(dims) or tuple(range(ndim))
    offset = ndim - layer_ndim
    clamped = list(point)
    for axis in order[:-mode]:
        local_axis = axis - offset
        if local_axis >= 0:
            clamped[axis] = min(
                max(clamped[axis], bounds[0, local_axis]), bounds[1, local_axis]
            )
    return _set(dims, "point", tuple(clamped)) if tuple(clamped) != point else False


def fit_layer_view(viewer, layer, *, margin=0.05) -> bool:
    """Frame one layer without including hidden or unrelated global layers.

    Use the calibrated world AABB, including affine-transformed pixel edges for
    Image/Labels. As in napari, 3D zoom projects this AABB onto camera up/right;
    camera angles and perspective are preserved. The fit is orthographic framing
    (not a perspective-frustum solver). No layer data is read or changed.
    """
    margin = _number(margin)
    if margin is None or not 0 <= margin < 1:
        raise ValueError("View margin must be finite and between 0 and 1.")
    bounds, camera = _layer_bounds(layer), _camera(viewer)
    if bounds is None or camera is None:
        return False
    bounds = _scene_bounds(viewer, layer, bounds)
    if bounds is None:
        return False
    dims = _get(viewer, "dims")
    ndim, layer_ndim = _ndim(dims), bounds.shape[1]
    if not ndim or layer_ndim > ndim or layer_ndim < 2:
        return False
    mode = _get(dims, "ndisplay", 2)
    mode = mode if mode in (2, 3) else 2
    if mode > layer_ndim:
        mode = 2
        _set(dims, "ndisplay", mode)
    order = _order(dims) or tuple(range(ndim))
    offset = ndim - layer_ndim
    if any(axis < offset for axis in order[-mode:]):
        order = tuple(axis for axis in order if axis < offset) + tuple(
            axis for axis in order if axis >= offset
        )
        _set(dims, "order", order)
    _clamp_layer_slice_point(viewer, layer)
    displayed = [axis - offset for axis in order[-mode:]]
    selected = bounds[:, displayed]
    center = selected[0] / 2 + selected[1] / 2
    size = selected[1] - selected[0]
    projected = size if mode == 2 else _projected_size(camera, size)
    # Ignore only genuinely zero projected spans: real subpixel/calibrated
    # lengths must not be replaced by an arbitrary world-unit floor.
    nonzero = projected > 0
    canvas_size = _canvas_size(viewer)
    zoom = (1 - margin) * float(
        np.min(canvas_size[nonzero] / projected[nonzero])
        if np.any(nonzero)
        else np.min(canvas_size)
    )
    if not math.isfinite(zoom) or zoom <= 0:
        return False
    camera_center = tuple([0.0] * (3 - len(center)) + center.tolist())
    centered = _set(camera, "center", camera_center)
    return _set(camera, "zoom", zoom) and centered
