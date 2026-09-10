from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from qtpy.QtCore import QPointF, QRect, QRectF, Qt
from qtpy.QtGui import QBrush, QColor, QPainterPath, QPen

from napari_vipp.ui.plots import DetailedHistogramPlot


@dataclass
class _Paint:
    path: QPainterPath
    pen: QPen
    brush: QBrush

    @property
    def is_fill(self) -> bool:
        return self.brush.style() != Qt.NoBrush

    @property
    def is_stroke(self) -> bool:
        return self.pen.style() != Qt.NoPen


class _PainterRecorder:
    """Inspect actual drawing primitives without platform-specific raster edges."""

    def __init__(self):
        self.pen = QPen()
        self.brush = QBrush(Qt.NoBrush)
        self.paints: list[_Paint] = []

    def setPen(self, pen) -> None:  # noqa: N802
        self.pen = QPen(pen)

    def setBrush(self, brush) -> None:  # noqa: N802
        self.brush = QBrush(brush)

    def drawPath(self, path) -> None:  # noqa: N802
        self.paints.append(
            _Paint(QPainterPath(path), QPen(self.pen), QBrush(self.brush))
        )

    def fillPath(self, path, brush) -> None:  # noqa: N802
        self.paints.append(
            _Paint(QPainterPath(path), QPen(Qt.NoPen), QBrush(brush))
        )

    def strokePath(self, path, pen) -> None:  # noqa: N802
        self.paints.append(
            _Paint(QPainterPath(path), QPen(pen), QBrush(Qt.NoBrush))
        )

    def drawRect(self, rect) -> None:  # noqa: N802
        path = QPainterPath()
        path.addRect(QRectF(rect))
        self.drawPath(path)


def _configured_plot(qtbot, *, edges, values, **kwargs):
    plot = DetailedHistogramPlot()
    qtbot.addWidget(plot)
    plot.resize(800, 450)
    series_count = np.asarray(values).shape[0]
    plot.set_histogram(
        edges,
        values,
        colors=[QColor("red"), QColor("lime")][:series_count],
        **kwargs,
    )
    return plot


def _paints(plot, rect):
    recorder = _PainterRecorder()
    plot._draw_bars(recorder, rect)
    return recorder.paints


def _strokes(paints, color):
    return [
        paint.path
        for paint in paints
        if paint.is_stroke and paint.pen.color().name() == QColor(color).name()
    ]


def _segments(path):
    previous = None
    for index in range(path.elementCount()):
        element = path.elementAt(index)
        point = QPointF(element.x, element.y)
        assert not element.isCurveTo(), "Histogram outlines must not smooth the data"
        if element.isLineTo() and previous is not None:
            yield previous, point
        previous = point


@pytest.mark.parametrize("x_scale", ["linear", "log10"])
@pytest.mark.parametrize("y_scale", ["linear", "log10"])
def test_overlay_draws_all_fills_before_exact_stepped_outlines(
    qtbot, x_scale, y_scale
):
    edges = np.asarray([1.0, 3.0, 20.0, 70.0, 100.0])
    values = np.asarray([[4.0, 8.0, 4.0, 8.0], [40.0, 80.0, 40.0, 80.0]])
    plot = _configured_plot(
        qtbot,
        edges=edges,
        values=values,
        x_scale=x_scale,
        y_scale=y_scale,
    )
    rect = QRect(20, 20, 700, 300)

    paints = _paints(plot, rect)

    fills = [index for index, paint in enumerate(paints) if paint.is_fill]
    strokes = [index for index, paint in enumerate(paints) if paint.is_stroke]
    assert fills and strokes
    assert max(fills) < min(strokes), "Later fills must not veil earlier outlines"
    assert all(0 < paints[index].brush.color().alpha() < 128 for index in fills)
    assert not any(paint.is_fill and paint.is_stroke for paint in paints)

    for series_index, color in enumerate(("red", "lime")):
        paths = _strokes(paints, color)
        assert len(paths) == 1, "Draw one top outline, not one rectangle per bin"
        segments = list(_segments(paths[0]))
        assert segments
        # All positive heights are above the baseline in these fixtures. A
        # closed bar outline would introduce unwanted vertical baseline strokes.
        assert all(
            point.y() < rect.bottom()
            for segment in segments
            for point in segment
        )
        assert all(
            left.x() == right.x() or left.y() == right.y()
            for left, right in segments
        )
        edge_pixels = plot._x_pixels(edges, rect)
        for bin_index in range(values.shape[1]):
            midpoint = (edge_pixels[bin_index] + edge_pixels[bin_index + 1]) / 2
            expected_y = plot._y_pixel(float(values[series_index, bin_index]), rect)
            assert any(
                left.x() <= midpoint <= right.x()
                and left.y() == pytest.approx(expected_y, abs=0.5)
                and right.y() == pytest.approx(expected_y, abs=0.5)
                for left, right in segments
            ), "Every nonuniform bin must retain its own transformed width and height"


def test_log_y_zero_bins_leave_gaps_in_outline_and_fill(qtbot):
    edges = np.asarray([1.0, 2.0, 4.0, 8.0, 16.0])
    plot = _configured_plot(
        qtbot,
        edges=edges,
        values=np.asarray([[4.0, 0.0, 8.0, 0.0], [40.0, 60.0, 80.0, 100.0]]),
        x_scale="log10",
        y_scale="log10",
    )
    rect = QRect(20, 20, 600, 240)
    paints = _paints(plot, rect)
    segments = [
        segment
        for path in _strokes(paints, "red")
        for segment in _segments(path)
    ]
    red_fills = [
        paint.path
        for paint in paints
        if paint.is_fill and paint.brush.color().name() == QColor("red").name()
    ]
    edge_pixels = plot._x_pixels(edges, rect)

    for index in (1, 3):
        midpoint = float(edge_pixels[index] + edge_pixels[index + 1]) / 2
        assert not any(
            min(left.x(), right.x()) < midpoint < max(left.x(), right.x())
            for left, right in segments
        ), "A zero bin has no logarithm; do not connect across its gap"
        assert not any(
            path.contains(QPointF(midpoint, rect.bottom() - 2.0))
            for path in red_fills
        )


@pytest.mark.parametrize("x_scale", ["linear", "log10"])
@pytest.mark.parametrize("y_scale", ["linear", "log10"])
def test_dense_overlay_retains_a_narrow_peak_and_original_hover_data(
    qtbot, x_scale, y_scale
):
    bin_count = 4096
    edges = np.geomspace(1.0, 1000.0, bin_count + 1)
    values = np.ones((2, bin_count), dtype=np.float64)
    peak_index = 1703
    values[0, peak_index] = 100.0
    values[1, :] = 10.0
    edges.flags.writeable = False
    values.flags.writeable = False
    plot = _configured_plot(
        qtbot,
        edges=edges,
        values=values,
        x_scale=x_scale,
        y_scale=y_scale,
        series_labels=("First channel", "Second channel"),
        hover_details={"Original counts": values},
    )
    rect = QRect(20, 20, 180, 160)
    original_tooltip = plot._bin_tooltip(peak_index)

    paints = _paints(plot, rect)

    paths = _strokes(paints, "red")
    assert paths
    peak_y = plot._y_pixel(100.0, rect)
    peak_edges = plot._x_pixels(edges[peak_index : peak_index + 2], rect)
    peak_x = float(np.mean(peak_edges))
    assert any(
        left.y() == pytest.approx(peak_y, abs=0.5)
        and right.y() == pytest.approx(peak_y, abs=0.5)
        and min(left.x(), right.x()) - 2 <= peak_x <= max(left.x(), right.x()) + 2
        for path in paths
        for left, right in _segments(path)
    ), "Pixel aggregation must preserve a narrow peak, not average it away"
    assert sum(path.elementCount() for path in paths) <= rect.width() * 5
    np.testing.assert_array_equal(plot.values, values)
    np.testing.assert_array_equal(plot.bin_edges, edges)
    assert not plot.values.flags.writeable
    assert not plot.bin_edges.flags.writeable
    assert plot._bin_tooltip(peak_index) == original_tooltip
    assert "First channel: Count: 100" in original_tooltip
    assert "Original counts: First channel 100, Second channel 10" in original_tooltip


def test_single_series_keeps_existing_outlined_bars(qtbot):
    plot = _configured_plot(
        qtbot,
        edges=np.asarray([0.0, 1.0, 2.0, 3.0]),
        values=np.asarray([[4.0, 8.0, 6.0]]),
    )

    paints = _paints(plot, QRect(20, 20, 360, 180))

    assert len(paints) == 3
    assert all(paint.is_fill and paint.is_stroke for paint in paints)
    assert all(paint.pen.color().alpha() == 255 for paint in paints)
    assert all(paint.brush.color().alpha() == 185 for paint in paints)
