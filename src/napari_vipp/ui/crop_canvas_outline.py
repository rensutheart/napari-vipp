"""Live crop overlays painted by Qt, outside the image's OpenGL scene."""

from __future__ import annotations

from contextlib import suppress
from math import isfinite
from weakref import ref

import numpy as np
from qtpy.QtCore import QLineF, QPointF, QRectF, Qt, QTimer
from qtpy.QtGui import QColor, QPainter, QPen, QPolygonF
from qtpy.QtWidgets import QWidget

# Vertices are in itertools.product order: each edge flips exactly one bit.
_BOX_EDGES = tuple((a, a ^ b) for a in range(8) for b in (1, 2, 4) if not a & b)
_RECT_EDGES = ((0, 1), (1, 3), (3, 2), (2, 0))
_Segment = tuple[float, float, float, float]
_Polygon = tuple[tuple[float, float], ...]


def _clip_edges(points, rect, edges=_BOX_EDGES) -> tuple[_Segment, ...]:
    """Clip homogeneous canvas coords *before* dividing by perspective depth."""
    left, top, right, bottom = rect
    x, y, z, w = np.asarray(points, dtype=float).T
    # VisPy's viewport maps normalised X/Y to logical pixels, leaving Z/W in
    # OpenGL clip space. Intersect each edge with the six viewing half-spaces.
    inside = np.column_stack(
        (x - left * w, right * w - x, y - top * w, bottom * w - y, w + z, w - z)
    )
    result = []
    for a, b in edges:
        av, bv = inside[a], inside[b]
        if not np.all(np.isfinite((av, bv))) or np.any((av < 0) & (bv < 0)):
            continue
        first, last = 0.0, 1.0
        enters, leaves = av < 0, bv < 0
        if np.any(enters):
            first = float(np.max(av[enters] / (av[enters] - bv[enters])))
        if np.any(leaves):
            last = float(np.min(av[leaves] / (av[leaves] - bv[leaves])))
        if first > last:
            continue
        start = points[a] + first * (points[b] - points[a])
        end = points[a] + last * (points[b] - points[a])
        if min(start[3], end[3]) <= 0:
            continue
        xy = np.concatenate((start[:2] / start[3], end[:2] / end[3]))
        # Ignore imperceptible floating point jitter when deciding to repaint.
        result.append(tuple(float(v) for v in np.round(xy, 3)))
    return tuple(result)


def _clip_plane_fill(points, rect) -> _Polygon:
    """Clip the 2D tint without stroking new edges at the viewport boundary."""
    if not np.isfinite(points).all() or (points[:, 3] <= 0).any():
        return ()
    # This is a single plane in napari's orthographic PanZoom view. Never
    # project a clipped 3D face backwards through the camera.
    if (np.abs(points[:, 2]) > points[:, 3]).any():
        return ()
    xy = points[:, :2] / points[:, 3, None]
    polygon = QPolygonF([QPointF(*xy[i]) for i, _ in _RECT_EDGES])
    left, top, right, bottom = rect
    boundary = QPolygonF(QRectF(left, top, right - left, bottom - top))
    return tuple(
        (round(p.x(), 3), round(p.y(), 3)) for p in polygon.intersected(boundary)
    )


def _to_canvas(node, root, points):
    """Walk already-existing transforms, never constructing a scene visual."""
    while node is not root:
        if node is None:
            raise ValueError("Image is not attached to this canvas.")
        points = node.transform.map(points)
        node = node.parent
    return np.asarray(points)


def _project(canvas, layer, corners):
    visual = canvas.layer_to_visual.get(layer)
    node = getattr(visual, "node", None)
    scene = getattr(node, "parent", None)
    # An empty slice, including a refused over-budget volume, has no visible
    # visual even though the layer's visibility checkbox may still be on.
    if scene is None or not layer.visible or not getattr(node, "visible", True):
        return None
    view, root = scene.parent, canvas._scene_canvas.scene
    # Coordinates are calibrated in the layer's units. napari can display
    # multiple layers in a common unit different from this particular layer.
    xyz = corners[:, ::-1]
    factors = getattr(visual, "_world_to_layer_units_scale", None)
    if factors is not None:
        axes = layer._slice_input.displayed
        xyz = xyz * np.asarray(factors)[list(axes)][::-1]
    projected = _to_canvas(scene, root, xyz)
    # Limit the ink to this volume's viewbox, including in napari grid view.
    size = view.size
    viewport = _to_canvas(view, root, [[0, 0, 0, 1], [size[0], size[1], 0, 1]])
    rect = (*viewport[0, :2], *viewport[1, :2])
    if corners.shape[1] == 2:
        return _clip_edges(projected, rect, _RECT_EDGES), _clip_plane_fill(
            projected, rect
        )
    return _clip_edges(projected, rect), ()


class _OutlineWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("vipp_crop_canvas_outline")
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setFocusPolicy(Qt.NoFocus)
        self.segments: tuple[_Segment, ...] = ()
        self.fill: _Polygon = ()
        self.retained = True
        self.line_width = 2.0

    def set_strokes(self, segments, fill, retained, width):
        strokes = (segments, fill, retained, width)
        if (self.segments, self.fill, self.retained, self.line_width) == strokes:
            return
        self.segments, self.fill, self.retained, self.line_width = strokes
        self.update()

    def paintEvent(self, event):  # noqa: N802
        if not self.segments and not self.fill:
            return
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            color = QColor("#ffbe45" if self.retained else "#ef4444")
            if self.fill:
                fill = QColor(color)
                fill.setAlpha(32)
                painter.setPen(Qt.NoPen)
                painter.setBrush(fill)
                painter.drawPolygon(QPolygonF([QPointF(*p) for p in self.fill]))
            lines = [QLineF(*p) for p in self.segments]
            # Dual contrast keeps the same box legible on dark and bright voxels.
            for stroke, width in (
                (QColor(32, 23, 5, 235), self.line_width + min(self.line_width, 2)),
                (color, self.line_width),
            ):
                pen = QPen(stroke)
                pen.setWidthF(width)
                painter.setPen(pen)
                painter.drawLines(lines)
        finally:
            painter.end()


class CropCanvasOutline:
    """Attach a passive raster overlay to the actual napari canvas on demand.

    Corners are in calibrated world coordinates and displayed axis order,
    ordered as itertools.product(low/high for each axis). No image pixels
    are read, no scene object is installed, and the camera is never switched.
    """

    def __init__(self, viewer):
        self._viewer = viewer
        self._canvas = self._layer = self._corners = None
        self._widget = self._timer = None
        self._width = 2.0
        self._retained = True

    def _viewer_canvas(self):
        qt_viewer = getattr(getattr(self._viewer, "window", None), "_qt_viewer", None)
        return getattr(qt_viewer, "canvas", None)

    @property
    def available(self) -> bool:
        """Whether this viewer exposes the actual Qt canvas needed for an outline."""
        canvas = self._viewer_canvas()
        return isinstance(getattr(canvas, "native", None), QWidget)

    def set_box(self, layer, world_corners, *, line_width=2.0) -> bool:
        """Show the twelve edges of a 3D crop box."""
        return self._set_shape(layer, world_corners, (8, 3), True, line_width)

    def set_rectangle(
        self, layer, world_corners, *, retained=True, line_width=2.0
    ) -> bool:
        """Outline the retained region on this plane (red for an excluded slice)."""
        return self._set_shape(layer, world_corners, (4, 2), retained, line_width)

    def _set_shape(self, layer, world_corners, expected_shape, retained, line_width):
        corners = np.asarray(world_corners, dtype=float)
        canvas = self._viewer_canvas()
        if (
            canvas is None
            or not isinstance(getattr(canvas, "native", None), QWidget)
            or layer is None
            or corners.shape != expected_shape
            or not np.isfinite(corners).all()
        ):
            self.clear()
            return False
        if self._canvas is not canvas:
            self.close()
            self._canvas = canvas
            self._widget = _OutlineWidget(canvas.native)
            self._timer = QTimer(self._widget)
            self._timer.setSingleShot(True)
            self._timer.timeout.connect(self._refresh)
            # Draws cover rotation, pan, zoom, resize and grid/layer changes.
            # Project after VisPy has finished updating its camera matrices.
            canvas.events.draw.connect(self._after_draw, position="last")
        # A canvas outline must not keep a removed volume's pixel array alive.
        self._layer = ref(layer)
        self._corners = corners.copy()
        self._retained = bool(retained)
        width = float(line_width)
        self._width = width if isfinite(width) and width > 0 else 2.0
        return self._refresh()

    def _after_draw(self, event):
        if self._layer is not None and not self._timer.isActive():
            self._timer.start(0)

    def _refresh(self) -> bool:
        if self._layer is None:
            return False
        try:
            projected = None
            layer = self._layer()
            if layer is None:
                self.clear()
                return False
            if self._viewer.dims.ndisplay == self._corners.shape[1]:
                projected = _project(self._canvas, layer, self._corners)
            widget = self._widget
            if projected is None:
                widget.hide()
                return False
            rect = self._canvas.native.rect()
            if widget.geometry() != rect:
                widget.setGeometry(rect)
            widget.set_strokes(*projected, self._retained, self._width)
            if widget.isHidden():
                widget.raise_()
                widget.show()
            return True
        except (AttributeError, IndexError, TypeError, RuntimeError, ValueError):
            # During canvas teardown / layer reparenting never leave stale ink.
            with suppress(RuntimeError):
                self._widget.hide()
            return False

    def clear(self):
        """Remove the drawing immediately without altering any viewer layer."""
        self._layer = self._corners = None
        if self._widget is not None:
            with suppress(RuntimeError):
                self._timer.stop()
                self._widget.hide()

    def close(self):
        self.clear()
        if self._canvas is not None:
            with suppress(RuntimeError):
                self._canvas.events.draw.disconnect(self._after_draw)
        if self._widget is not None:
            with suppress(RuntimeError):
                self._widget.deleteLater()
        self._canvas = self._widget = self._timer = None
