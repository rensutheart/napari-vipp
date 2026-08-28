from __future__ import annotations

import numpy as np
from qtpy.QtCore import QPoint, Qt
from qtpy.QtGui import QColor

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
    return plot


def _histogram_point(plot, fraction: float) -> QPoint:
    rect = plot._plot_rect()
    return QPoint(
        rect.left() + int(round(fraction * rect.width())),
        rect.center().y(),
    )


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
    qtbot.mousePress(plot, Qt.LeftButton, pos=start)
    qtbot.mouseMove(plot, pos=moved)

    assert events == ["started"]

    plot.hide()
    plot.hide()

    assert events == ["started", "finished"]
