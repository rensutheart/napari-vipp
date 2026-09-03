from __future__ import annotations

import numpy as np
from qtpy.QtCore import QPoint, QRect, Qt
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QApplication

from napari_vipp import _widget
from napari_vipp.ui import plots


def _configured_histogram_plot(qtbot):
    plot = plots.HistogramPlot()
    qtbot.addWidget(plot)
    plot.resize(240, 160)
    plot.set_histogram(
        np.ones(32, dtype=np.float32),
        log_scale=False,
        x_range=(0.0, 100.0),
        markers=[("threshold", 50.0, QColor("#f59e0b"))],
        draggable_markers={"threshold"},
    )
    plot.show()
    qtbot.waitExposed(plot)
    return plot


def _histogram_point(plot, fraction: float) -> QPoint:
    rect = plot._plot_rect()
    return QPoint(
        rect.left() + int(round(fraction * rect.width())),
        rect.center().y(),
    )


class _BarPaintRecorder:
    """Record the colors supplied to the histogram drawing primitives."""

    def __init__(self):
        self.pen_colors: list[QColor] = []
        self.brush_colors: list[QColor] = []

    def setPen(self, pen) -> None:  # noqa: N802
        self.pen_colors.append(QColor(pen.color()))

    def setBrush(self, brush) -> None:  # noqa: N802
        if isinstance(brush, QColor):
            self.brush_colors.append(QColor(brush))

    def drawLine(self, *_args) -> None:  # noqa: N802
        return None

    def drawRect(self, *_args) -> None:  # noqa: N802
        return None


def _record_compact_histogram_paint(qtbot, *, series_count: int, bin_count: int):
    plot = plots.HistogramPlot()
    qtbot.addWidget(plot)
    values = np.ones((series_count, bin_count), dtype=np.float32)
    plot.set_histogram(
        values,
        log_scale=False,
        colors=[QColor("#d946ef"), QColor("#06b6d4")][:series_count],
    )
    recorder = _BarPaintRecorder()
    plot._draw_histogram_series(recorder, QRect(10, 10, 180, 90), values, 1.0)
    return recorder


def _record_detailed_histogram_paint(qtbot, *, series_count: int):
    plot = plots.DetailedHistogramPlot()
    qtbot.addWidget(plot)
    values = np.ones((series_count, 3), dtype=np.float64)
    plot.set_histogram(
        np.asarray([0.0, 1.0, 2.0, 3.0]),
        values,
        colors=[QColor("#d946ef"), QColor("#06b6d4")][:series_count],
    )
    recorder = _BarPaintRecorder()
    plot._draw_bars(recorder, QRect(10, 10, 180, 90))
    return recorder


def test_widget_module_reexports_extracted_plot_symbols():
    assert _widget.HistogramPlot is plots.HistogramPlot
    assert _widget.ColocalizationScatterPlot is plots.ColocalizationScatterPlot
    assert _widget._event_position is plots._event_position
    assert _widget._qcolor_from_channel_color is plots._qcolor_from_channel_color
    assert _widget._histogram_series_colors is plots._histogram_series_colors
    assert _widget._format_histogram_label is plots._format_histogram_label
    assert _widget.COLOCALIZATION_SCATTER_BINS == plots.COLOCALIZATION_SCATTER_BINS
    assert (
        _widget.COLOCALIZATION_SCATTER_COLORMAPS
        == plots.COLOCALIZATION_SCATTER_COLORMAPS
    )


def test_widget_scatter_density_facade_preserves_extracted_result():
    channel_1 = np.arange(16, dtype=np.float32).reshape(4, 4)
    channel_2 = np.flip(channel_1, axis=1).copy()
    kwargs = {
        "threshold_1": 4.0,
        "threshold_2": 6.0,
        "roi_mask": channel_1 % 2 == 0,
        "intensity_max": 15.0,
        "bins": 32,
    }

    facade_result = _widget._prepare_colocalization_scatter_density(
        channel_1,
        channel_2,
        **kwargs,
    )
    extracted_result = plots._prepare_colocalization_scatter_density(
        channel_1,
        channel_2,
        **kwargs,
    )

    np.testing.assert_array_equal(facade_result[0], extracted_result[0])
    assert facade_result[1:] == extracted_result[1:]


def test_scatter_pending_thresholds_preserve_only_compatible_density(qtbot):
    plot = plots.ColocalizationScatterPlot()
    qtbot.addWidget(plot)
    plot.set_density(
        np.ones((16, 16), dtype=np.float64),
        threshold_1=25.0,
        threshold_2=30.0,
        intensity_min=-20.0,
        intensity_max=4_000.0,
    )
    density_image = plot._image

    plot.set_pending_thresholds(
        threshold_1=70.0,
        threshold_2=80.0,
        preserve_density=True,
    )

    assert plot._image is density_image
    assert plot._threshold_1 == 70.0
    assert plot._threshold_2 == 80.0
    assert plot._intensity_min == -20.0
    assert plot._intensity_max == 4_000.0
    assert plot._summary == "Calculating exact count..."

    plot.set_pending_thresholds(
        threshold_1=90.0,
        threshold_2=100.0,
        preserve_density=False,
    )

    assert plot._image is None
    assert plot._summary == "Calculating exact count..."


def test_scatter_colormap_recolors_retained_density_without_state_change(qtbot):
    plot = plots.ColocalizationScatterPlot()
    qtbot.addWidget(plot)
    density = np.zeros((16, 16), dtype=np.float64)
    density[3:10, 5:12] = np.arange(49, dtype=np.float64).reshape(7, 7)
    plot.set_density(
        density,
        threshold_1=25.0,
        threshold_2=30.0,
        intensity_min=0.0,
        intensity_max=255.0,
        colormap="Viridis",
        summary="Calculating exact count...",
    )
    retained_density = plot._density_counts
    original_image = plot._image

    plot.set_colormap("Gray")

    assert plot._density_counts is retained_density
    assert plot._image is not original_image
    assert plot._image != original_image
    assert plot._threshold_1 == 25.0
    assert plot._threshold_2 == 30.0
    assert plot._summary == "Calculating exact count..."
    assert plot._colormap == "Gray"


def _configured_scatter_plot(qtbot):
    plot = plots.ColocalizationScatterPlot()
    qtbot.addWidget(plot)
    plot.resize(360, 340)
    plot.set_density(
        np.ones((16, 16), dtype=np.float64),
        threshold_1=25.0,
        threshold_2=70.0,
        intensity_min=0.0,
        intensity_max=100.0,
    )
    plot.show()
    return plot


def _scatter_threshold_points(plot):
    rect = plot._plot_rect()
    vertical = QPoint(
        plot._x_from_value(plot._threshold_1, rect),
        rect.bottom() - 20,
    )
    horizontal = QPoint(
        rect.right() - 20,
        plot._y_from_value(plot._threshold_2, rect),
    )
    return vertical, horizontal, rect.center()


def test_scatter_threshold_hover_uses_directional_resize_cursors(qtbot):
    plot = _configured_scatter_plot(qtbot)
    vertical, horizontal, away = _scatter_threshold_points(plot)

    qtbot.mouseMove(plot, pos=vertical)
    assert plot.cursor().shape() == Qt.SizeHorCursor

    qtbot.mouseMove(plot, pos=horizontal)
    assert plot.cursor().shape() == Qt.SizeVerCursor

    qtbot.mouseMove(plot, pos=away)
    assert plot.cursor().shape() == Qt.ArrowCursor


def test_scatter_threshold_cursor_and_axis_stay_locked_during_drag(qtbot):
    plot = _configured_scatter_plot(qtbot)
    vertical, horizontal, _away = _scatter_threshold_points(plot)
    emitted = []
    committed = []
    plot.thresholdChanged.connect(lambda axis, value: emitted.append((axis, value)))
    plot.thresholdCommitted.connect(
        lambda axis, value: committed.append((axis, value))
    )

    qtbot.mousePress(plot, Qt.LeftButton, pos=vertical)
    qtbot.mouseMove(plot, pos=horizontal)

    assert plot.cursor().shape() == Qt.SizeHorCursor
    assert emitted and {axis for axis, _value in emitted} == {1}
    assert committed == []

    qtbot.mouseRelease(plot, Qt.LeftButton, pos=horizontal)
    assert len(committed) == 1
    assert committed[0][0] == 1
    assert committed[0][1] == plot._threshold_1
    rect = plot._plot_rect()
    horizontal_only = QPoint(
        rect.left() + 20,
        plot._y_from_value(plot._threshold_2, rect),
    )
    qtbot.mouseMove(plot, pos=horizontal_only)

    assert plot.cursor().shape() == Qt.SizeVerCursor


def test_scatter_drag_requires_left_button_near_a_guide_and_clear_resets_cursor(
    qtbot,
):
    plot = _configured_scatter_plot(qtbot)
    vertical, _horizontal, away = _scatter_threshold_points(plot)
    gestures = []
    plot.gestureStarted.connect(lambda: gestures.append("started"))

    try:
        qtbot.mousePress(plot, Qt.RightButton, pos=vertical)
        qtbot.mousePress(plot, Qt.LeftButton, pos=away)

        assert gestures == []
        assert plot._drag_axis is None

        qtbot.mouseMove(plot, pos=vertical)
        assert plot.cursor().shape() == Qt.SizeHorCursor
        plot.clear()
        assert plot.cursor().shape() == Qt.ArrowCursor
    finally:
        qtbot.mouseRelease(plot, Qt.LeftButton, pos=away)
        qtbot.mouseRelease(plot, Qt.RightButton, pos=vertical)

    assert QApplication.mouseButtons() == Qt.NoButton


def test_scatter_threshold_hit_area_remains_usable_at_plot_boundaries(qtbot):
    plot = _configured_scatter_plot(qtbot)
    plot.set_density(
        np.ones((16, 16), dtype=np.float64),
        threshold_1=0.0,
        threshold_2=100.0,
        intensity_min=0.0,
        intensity_max=100.0,
    )
    rect = plot._plot_rect()

    qtbot.mouseMove(plot, pos=QPoint(rect.left() - 4, rect.center().y()))
    assert plot.cursor().shape() == Qt.SizeHorCursor

    qtbot.mouseMove(plot, pos=QPoint(rect.center().x(), rect.top() - 4))
    assert plot.cursor().shape() == Qt.SizeVerCursor


def test_scatter_density_reset_cancels_an_active_threshold_gesture(qtbot):
    plot = _configured_scatter_plot(qtbot)
    vertical, _horizontal, _away = _scatter_threshold_points(plot)
    events = []
    plot.gestureStarted.connect(lambda: events.append("started"))
    plot.gestureFinished.connect(lambda: events.append("finished"))

    try:
        qtbot.mousePress(plot, Qt.LeftButton, pos=vertical)
        assert plot.cursor().shape() == Qt.SizeHorCursor
        plot.set_pending_thresholds(
            threshold_1=25.0,
            threshold_2=70.0,
            preserve_density=False,
        )

        assert events == ["started", "finished"]
        assert plot._drag_axis is None
        assert plot.cursor().shape() == Qt.ArrowCursor
    finally:
        qtbot.mouseRelease(plot, Qt.LeftButton, pos=vertical)

    assert QApplication.mouseButtons() == Qt.NoButton


def test_histogram_marker_drag_emits_one_gesture_pair(qtbot):
    plot = _configured_histogram_plot(qtbot)
    events = []
    plot.gestureStarted.connect(lambda: events.append("started"))
    plot.gestureFinished.connect(lambda: events.append("finished"))

    start = _histogram_point(plot, 0.5)
    middle = _histogram_point(plot, 0.65)
    end = _histogram_point(plot, 0.75)
    qtbot.mousePress(plot, Qt.LeftButton, pos=start)
    qtbot.mouseMove(plot, pos=middle)
    qtbot.mouseMove(plot, pos=end)

    assert events == ["started"]

    qtbot.mouseRelease(plot, Qt.LeftButton, pos=end)
    qtbot.mouseRelease(plot, Qt.LeftButton, pos=end)

    assert events == ["started", "finished"]


def test_histogram_plot_labels_use_compact_in_frame_geometry(qtbot):
    plot = _configured_histogram_plot(qtbot)
    unlabelled_rect = plot._plot_rect()

    plot.set_plot_labels(
        title=" Input ",
        x_axis_label=" Intensity (a.u.) ",
        y_axis_label=" Voxels ",
    )
    labelled_rect = plot._plot_rect()

    assert plot._title == "Input"
    assert plot._x_axis_label == "Intensity (a.u.)"
    assert plot._y_axis_label == "Voxels"
    assert labelled_rect.left() > unlabelled_rect.left()
    assert labelled_rect.top() >= (
        unlabelled_rect.top() + plot.fontMetrics().height() + 8
    )
    # Range ticks and the axis label intentionally share one bottom band.
    assert labelled_rect.bottom() == unlabelled_rect.bottom()
    assert labelled_rect.width() > 100
    assert labelled_rect.height() >= 60
    assert not plot.grab().isNull()


def test_histogram_plot_labels_preserve_marker_value_mapping(qtbot):
    plot = _configured_histogram_plot(qtbot)
    plot.set_plot_labels(
        title="Input",
        x_axis_label="Intensity (a.u.)",
        y_axis_label="Voxels",
    )
    emitted = []
    plot.markerChanged.connect(
        lambda label, value: emitted.append((label, value))
    )

    start = _histogram_point(plot, 0.5)
    end = _histogram_point(plot, 0.75)
    qtbot.mousePress(plot, Qt.LeftButton, pos=start)
    qtbot.mouseMove(plot, pos=end)
    qtbot.mouseRelease(plot, Qt.LeftButton, pos=end)

    assert emitted
    assert emitted[-1][0] == "threshold"
    assert np.isclose(emitted[-1][1], 75.0, atol=0.5)
    assert np.isclose(plot.marker_values()["threshold"], 75.0, atol=0.5)


def test_binary_histogram_draws_inset_visible_bars(qtbot):
    plot = plots.HistogramPlot()
    qtbot.addWidget(plot)
    plot.resize(240, 160)
    plot.set_plot_labels(
        title="Output",
        x_axis_label="Mask value",
        y_axis_label="Voxels",
    )
    plot.set_histogram(
        np.asarray([30, 70], dtype=np.float32),
        log_scale=False,
        x_range=(0.0, 1.0),
    )
    plot.show()
    qtbot.wait(10)

    plot_rect = plot._plot_rect()
    image = plot.grab().toImage()
    device_ratio = image.devicePixelRatio()
    slot_width = plot_rect.width() / 2.0
    sample_y = plot_rect.bottom() - 3
    bar_samples = [
        image.pixelColor(
            int(
                (plot_rect.left() + int((index + 0.5) * slot_width))
                * device_ratio
            ),
            int(sample_y * device_ratio),
        )
        for index in range(2)
    ]
    background = image.pixelColor(
        int(plot_rect.center().x() * device_ratio),
        int((plot_rect.top() + 3) * device_ratio),
    )

    assert all(sample != background for sample in bar_samples)


def test_compact_histogram_only_reduces_multiseries_stroke_opacity(qtbot):
    single = _record_compact_histogram_paint(qtbot, series_count=1, bin_count=16)
    multiseries = _record_compact_histogram_paint(
        qtbot,
        series_count=2,
        bin_count=16,
    )

    assert [color.alpha() for color in single.pen_colors] == [255]
    assert len(multiseries.pen_colors) == 2
    assert all(color.alpha() <= 128 for color in multiseries.pen_colors)
    assert all(
        color.alpha() < single.pen_colors[0].alpha() for color in multiseries.pen_colors
    )


def test_compact_discrete_histogram_only_reduces_multiseries_fill_opacity(qtbot):
    single = _record_compact_histogram_paint(qtbot, series_count=1, bin_count=3)
    multiseries = _record_compact_histogram_paint(
        qtbot,
        series_count=2,
        bin_count=3,
    )

    assert [color.alpha() for color in single.brush_colors] == [255]
    assert len(multiseries.brush_colors) == 2
    assert all(color.alpha() <= 80 for color in multiseries.brush_colors)
    assert all(
        color.alpha() < single.brush_colors[0].alpha()
        for color in multiseries.brush_colors
    )


def test_detailed_histogram_only_reduces_multiseries_fill_opacity(qtbot):
    single = _record_detailed_histogram_paint(qtbot, series_count=1)
    multiseries = _record_detailed_histogram_paint(qtbot, series_count=2)

    assert [color.alpha() for color in single.brush_colors] == [185]
    assert len(multiseries.brush_colors) == 2
    assert all(color.alpha() <= 80 for color in multiseries.brush_colors)
    assert all(
        color.alpha() < single.brush_colors[0].alpha()
        for color in multiseries.brush_colors
    )
    assert [color.alpha() for color in single.pen_colors] == [255]
    assert [color.alpha() for color in multiseries.pen_colors] == [255, 255]


def test_scatter_drag_hidden_mid_gesture_finishes_once(qtbot):
    plot = plots.ColocalizationScatterPlot()
    qtbot.addWidget(plot)
    plot.resize(360, 340)
    plot.set_density(
        np.ones((16, 16), dtype=np.float64),
        threshold_1=25.0,
        threshold_2=30.0,
        intensity_min=0.0,
        intensity_max=100.0,
    )
    plot.show()
    events = []
    plot.gestureStarted.connect(lambda: events.append("started"))
    plot.gestureFinished.connect(lambda: events.append("finished"))

    rect = plot._plot_rect()
    start = QPoint(plot._x_from_value(25.0, rect), rect.center().y())
    moved = QPoint(start.x() + 20, start.y())
    pressed = False
    try:
        qtbot.mousePress(plot, Qt.LeftButton, pos=start)
        pressed = True
        qtbot.mouseMove(plot, pos=moved)

        assert events == ["started"]

        plot.hide()
        plot.hide()

        assert events == ["started", "finished"]
    finally:
        # Hiding must finish the plot's semantic gesture, but QtTest still
        # requires the physical release paired with mousePress.  Without it,
        # QApplication retains LeftButton globally and unrelated later slider
        # tests can mis-detect or hang in an active drag.
        if pressed:
            qtbot.mouseRelease(plot, Qt.LeftButton, pos=moved)

    assert QApplication.mouseButtons() == Qt.NoButton


def _plot_palette(*, base: str, text: str) -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.Base, QColor(base))
    palette.setColor(QPalette.Text, QColor(text))
    palette.setColor(QPalette.AlternateBase, QColor(base))
    return palette


def _rendered_plot_surface(plot) -> QColor:
    image = plot.grab().toImage()
    # Stay clear of the antialiased one-pixel frame at any device-pixel ratio.
    return image.pixelColor(30, 30)


def test_histogram_surface_follows_runtime_palette_changes(qtbot):
    plot = plots.HistogramPlot()
    qtbot.addWidget(plot)
    plot.resize(240, 160)
    plot.show()

    plot.setPalette(_plot_palette(base="#ffffff", text="#111827"))
    qtbot.wait(10)
    assert _rendered_plot_surface(plot).name() == "#ffffff"

    plot.setPalette(_plot_palette(base="#111827", text="#f8fafc"))
    qtbot.wait(10)
    assert _rendered_plot_surface(plot).name() == "#111827"


def test_scatter_surface_follows_runtime_palette_changes(qtbot):
    plot = plots.ColocalizationScatterPlot()
    qtbot.addWidget(plot)
    plot.resize(360, 340)
    plot.show()

    plot.setPalette(_plot_palette(base="#ffffff", text="#111827"))
    qtbot.wait(10)
    assert _rendered_plot_surface(plot).name() == "#ffffff"

    plot.setPalette(_plot_palette(base="#111827", text="#f8fafc"))
    qtbot.wait(10)
    assert _rendered_plot_surface(plot).name() == "#111827"
