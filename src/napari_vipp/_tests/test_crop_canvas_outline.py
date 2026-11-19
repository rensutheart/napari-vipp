"""Regression checks for projected crop overlays that never touch image data."""

from itertools import product
from types import SimpleNamespace as NS

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QWidget
from vispy.util.event import EventEmitter
from vispy.visuals.transforms import STTransform

from napari_vipp.ui.crop_canvas_outline import CropCanvasOutline, _clip_edges


class Layer:
    visible = True
    _slice_input = NS(displayed=(0, 1, 2))


def _viewer(qtbot):
    native = QWidget()
    qtbot.addWidget(native)
    native.resize(240, 180)
    native.show()
    root = object()
    # A viewbox offset inside the canvas (like napari's grid) and an
    # orthographic camera. XYZ pixels in this view increase toward bottom-right.
    view = NS(parent=root, transform=STTransform(translate=(8, 12)), size=(180, 150))
    scene = NS(parent=view, transform=STTransform(scale=(4, 5, 0.01)))
    layer = Layer()
    canvas = NS(
        native=native,
        _scene_canvas=NS(scene=root),
        layer_to_visual={
            layer: NS(node=NS(parent=scene), _world_to_layer_units_scale=(1, 2, 3))
        },
        events=NS(draw=EventEmitter(type="draw")),
    )
    viewer = NS(window=NS(_qt_viewer=NS(canvas=canvas)), dims=NS(ndisplay=3))
    return viewer, layer, canvas, view


def _box(x1=10):
    return np.array(list(product((1, 8), (2, 10), (3, x1))))


def test_live_wireframe_projects_into_the_volume_view_and_tracks_camera(qtbot):
    viewer, layer, canvas, view = _viewer(qtbot)
    outline = CropCanvasOutline(viewer)
    assert outline.available
    assert outline.set_box(layer, _box())
    widget = canvas.native.findChild(QWidget, "vipp_crop_canvas_outline")
    assert widget.isVisible() and widget.testAttribute(Qt.WA_TransparentForMouseEvents)
    assert len(widget.segments) == 12
    # x world [3, 10] * unit 3 * zoom 4 + x offset 8 => [44, 128].
    # y world [2, 10] * unit 2 * zoom 5 + y offset 12 => [32, 112].
    np.testing.assert_equal(np.asarray(widget.segments).min(axis=0)[:2], (44, 32))
    np.testing.assert_equal(np.asarray(widget.segments).max(axis=0)[2:], (128, 112))
    im = widget.grab().toImage()
    dpr = im.devicePixelRatio()
    assert im.pixelColor(round(80 * dpr), round(32 * dpr)) == QColor("#ffbe45")
    for width in (0.1, 16.0):
        assert outline.set_box(layer, _box(), line_width=width)
        assert widget.line_width == width

    # Updating a draft changes the drawn box *before* a scene draw or commit.
    assert outline.set_box(layer, _box(7), line_width=3)
    assert max(p[2] for p in widget.segments) == 92
    # Native draws project again, but identical geometry must not enqueue a
    # redundant paint (repainting the child can itself redraw an OpenGL parent).
    before = widget.segments
    canvas.events.draw()
    qtbot.waitUntil(lambda: not outline._timer.isActive())
    assert widget.segments is before

    view.transform.translate = (4, 17)
    canvas.native.resize(300, 160)
    canvas.events.draw()
    qtbot.waitUntil(lambda: max(p[2] for p in widget.segments) == 88)
    assert widget.geometry() == canvas.native.rect()
    assert min(p[1] for p in widget.segments) == 37

    layer.visible = False
    canvas.events.draw()
    qtbot.waitUntil(widget.isHidden)
    layer.visible = True
    canvas.events.draw()
    qtbot.waitUntil(widget.isVisible)
    canvas.layer_to_visual[layer].node.visible = False
    canvas.events.draw()
    qtbot.waitUntil(widget.isHidden)
    outline.clear()
    assert widget.isHidden()
    canvas.events.draw()
    assert not outline._timer.isActive()
    outline.close()


def test_missing_camera_invalid_bounds_or_non_3d_hide_previous_ink(qtbot):
    assert not CropCanvasOutline(NS()).available
    assert not CropCanvasOutline(NS()).set_box(Layer(), _box())
    viewer, layer, canvas, _ = _viewer(qtbot)
    outline = CropCanvasOutline(viewer)
    assert outline.set_box(layer, _box())
    widget = outline._widget
    assert not outline.set_box(layer, np.full((8, 3), np.nan))
    assert widget.isHidden()
    assert outline.set_box(layer, _box())
    viewer.dims.ndisplay = 2
    canvas.events.draw()
    qtbot.waitUntil(widget.isHidden)
    del canvas.layer_to_visual[layer]
    viewer.dims.ndisplay = 3
    assert not outline.set_box(layer, _box())
    assert widget.isHidden()
    outline.close()


def test_rectangle_follows_plane_margins_and_highlights_excluded_slices(qtbot):
    viewer, layer, canvas, view = _viewer(qtbot)
    outline = CropCanvasOutline(viewer)
    assert outline.set_box(layer, _box())
    widget = outline._widget
    assert widget.fill == ()
    viewer.dims.ndisplay = 2
    layer._slice_input = NS(displayed=(1, 2))
    # Full-source data edges Y=[2,10], X=[3,10] in the layer's physical
    # units, independent of scan-axis order or the currently displayed Z.
    corners = np.array(list(product((2.0, 10.0), (3.0, 10.0))))
    assert outline.set_rectangle(layer, corners)
    assert outline._widget is widget and widget.isVisible()
    assert len(widget.segments) == 4
    np.testing.assert_equal(
        set(widget.segments),
        {
            (44, 32, 128, 32),
            (128, 32, 128, 112),
            (128, 112, 44, 112),
            (44, 112, 44, 32),
        },
    )

    def pixel(x, y):
        im = widget.grab().toImage()
        return im.pixelColor(
            round(x * im.devicePixelRatio()), round(y * im.devicePixelRatio())
        )

    assert pixel(70, 32) == QColor("#ffbe45")
    assert pixel(70, 80) != pixel(25, 80)

    # Capture the held margin's new location before any draw/commit. Cross
    # an excluded Z plane with the SAME XY footprint: style must repaint.
    corners[:2, 0] = 5
    assert outline.set_rectangle(layer, corners)
    assert pixel(70, 62) == QColor("#ffbe45")
    assert outline.set_rectangle(layer, corners, retained=False)
    assert pixel(70, 62) == QColor("#ef4444")
    assert pixel(70, 32) != QColor("#ef4444")
    view.transform.translate = (10, 20)
    canvas.events.draw()
    qtbot.waitUntil(lambda: min(s[1] for s in widget.segments) == 70)
    assert pixel(70, 70) == QColor("#ef4444")

    viewer.dims.ndisplay = 3
    layer._slice_input = NS(displayed=(0, 1, 2))
    assert outline.set_box(layer, _box())
    assert widget.retained and not widget.fill and len(widget.segments) == 12
    outline.close()


def test_zoom_inside_rectangle_keeps_only_tint_in_relevant_viewbox(qtbot):
    viewer, layer, _canvas, _view = _viewer(qtbot)
    viewer.dims.ndisplay = 2
    layer._slice_input = NS(displayed=(1, 2))
    outline = CropCanvasOutline(viewer)
    assert outline.set_rectangle(
        layer, np.array(list(product((-100, 100), (-100, 100))))
    )
    widget = outline._widget
    assert widget.segments == ()  # No false edge invented at the crop of the viewport
    assert set(widget.fill) == {(8, 12), (188, 12), (188, 162), (8, 162)}
    assert not outline.set_rectangle(layer, np.full((4, 2), np.nan))
    assert widget.isHidden()
    outline.close()


def test_edges_clipped_at_near_plane_before_perspective_division():
    # The four near-plane-side points are BEHIND the eye (w<0). The surviving
    # front face must not be joined to inverted, spurious screen coordinates.
    points = np.array(
        [
            (x * (1 + z / 2), y * (1 + z / 2), z, 1 + z / 2)
            for z, y, x in product((-4, 0), (20, 60), (30, 90))
        ],
        dtype=float,
    )
    segments = _clip_edges(points, (0, 0, 100, 100))
    assert len(segments) == 8  # 4 intact + 4 entering at the near plane
    assert all(20 <= v <= 90 for seg in segments for v in seg)
    points[:, 2] = -100
    assert _clip_edges(points, (0, 0, 100, 100)) == ()

    # An edge from (40,40,-2,1) to (40,40,0,2) crosses Z=-W at t=1/3.
    # The near point in logical pixels is (30,30), not the perspective
    # projection of either original end or their screen-space midpoint.
    points = np.array(
        [
            (x, y, -2 if front == 0 else 0, 1 + front)
            for front, y, x in product((0, 1), (40, 60), (40, 60))
        ],
        dtype=float,
    )
    assert (30.0, 30.0, 20.0, 20.0) in _clip_edges(points, (0, 0, 100, 100))


def test_edges_never_spill_out_of_the_relevant_grid_cell():
    corners = np.asarray(list(product((-0.1, 0.1), (0, 30), (5, 25))))
    points = np.column_stack((corners[:, ::-1], np.ones(8)))
    segments = _clip_edges(points, (10, 0, 20, 30))
    assert len(segments) == 4
    assert all(s[0] == 10 and s[2] == 20 for s in segments)
