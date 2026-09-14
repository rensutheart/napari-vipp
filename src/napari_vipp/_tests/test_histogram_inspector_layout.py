"""Embedded histogram height reserves data space after text and legend bands."""

import numpy as np
import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont, QFontMetrics
from qtpy.QtWidgets import QApplication, QScrollArea, QVBoxLayout, QWidget

from napari_vipp.ui.plots import DetailedHistogramPlot


def _flush_layout():
    # Font/resize changes and the resulting parent-layout request are queued
    # separately. Flush a bounded number of event passes, without timed sleeps.
    for _ in range(3):
        QApplication.processEvents()


def _font(plot, points):
    font = QFont(plot.font())
    font.setPointSizeF(points)
    plot.setFont(font)


def _populate(plot, *, label="Count", channels=3):
    edges = np.array([1.0, 10.0, 100.0, 1000.0])
    values = np.array([[1, 3, 2], [2, 4, 1], [4, 2, 3]], dtype=float)[:channels]
    plot.set_histogram(
        edges,
        values * 1e6,
        title="Intensity Histogram",
        x_axis_label="Input value (a.u.)",
        y_axis_label=label,
        series_labels=tuple(f"Channel {i + 1}" for i in range(channels)),
    )


def _compact_at_width(plot, width):
    plot.resize(width, 1)
    _flush_layout()
    # A wider resize may lower the minimum without shrinking current geometry.
    # Ask again for the compact layout rather than testing the old tall size.
    plot.resize(width, 1)
    _flush_layout()
    assert plot.width() == width
    assert plot.height() == plot.minimumHeight()


def _assert_readable(plot, label):
    rect = plot._plot_rect()
    metrics = QFontMetrics(plot._axis_label_font())
    assert rect.height() >= 160
    assert rect.height() >= metrics.horizontalAdvance(label) + 8
    assert metrics.elidedText(label, Qt.ElideRight, rect.height()) == label
    assert rect.width() > 0
    assert plot.rect().contains(rect)


@pytest.mark.parametrize(
    ("width", "points", "label"),
    [
        (260, 9, "Count"),
        (260, 13, "Probability density"),
        (520, 9, "Cumulative fraction"),
        (520, 13, "Probability density"),
    ],
)
def test_compact_inspector_retains_data_height_and_full_y_label(
    qtbot, width, points, label
):
    plot = DetailedHistogramPlot(minimum_plot_height=160)
    qtbot.addWidget(plot)
    _font(plot, points)
    _populate(plot, label=label)
    plot.show()
    _compact_at_width(plot, width)

    _assert_readable(plot, label)
    metrics = QFontMetrics(plot._tick_label_font())
    if width == 260:
        assert plot._legend_row_count(metrics, plot._plot_rect().width()) >= 2


def test_widening_releases_only_unneeded_legend_height_and_keeps_cached_data(qtbot):
    plot = DetailedHistogramPlot(minimum_plot_height=160)
    qtbot.addWidget(plot)
    _font(plot, 9)
    _populate(plot)
    plot.set_scales(x_scale="log10", y_scale="log10")
    before_edges, before_values = plot.bin_edges.copy(), plot.values.copy()
    retained_edges, retained_values = plot.bin_edges, plot.values
    plot.show()
    _compact_at_width(plot, 260)
    narrow_height = plot.minimumHeight()
    _compact_at_width(plot, 520)

    assert plot.minimumHeight() < narrow_height
    _assert_readable(plot, "Count")
    _compact_at_width(plot, 260)
    assert plot.minimumHeight() == narrow_height
    _assert_readable(plot, "Count")
    np.testing.assert_array_equal(plot.bin_edges, before_edges)
    np.testing.assert_array_equal(plot.values, before_values)
    assert np.shares_memory(plot.bin_edges, retained_edges)
    assert np.shares_memory(plot.values, retained_values)
    assert not plot.bin_edges.flags.writeable and not plot.values.flags.writeable
    assert plot.x_logarithmic and plot.y_logarithmic


def test_font_and_style_changes_recompute_embedded_height(qtbot):
    plot = DetailedHistogramPlot(minimum_plot_height=160)
    qtbot.addWidget(plot)
    _font(plot, 9)
    _populate(plot, label="Cumulative fraction")
    plot.show()
    _compact_at_width(plot, 260)
    original = plot.minimumHeight()

    _font(plot, 13)
    _compact_at_width(plot, 260)
    assert plot.minimumHeight() > original
    _assert_readable(plot, "Cumulative fraction")
    plot.setStyleSheet("font-size: 18pt;")
    _compact_at_width(plot, 260)
    _assert_readable(plot, "Cumulative fraction")


def test_scale_grid_and_clear_keep_height_current_without_replacing_values(qtbot):
    plot = DetailedHistogramPlot(minimum_plot_height=160)
    qtbot.addWidget(plot)
    _font(plot, 13)
    _populate(plot)
    plot.show()
    _compact_at_width(plot, 260)
    retained = plot.values
    plot.set_scales(y_scale="log10")
    plot.set_grid_divisions(x=12, y=12)
    _compact_at_width(plot, 260)
    populated_height = plot.minimumHeight()
    _assert_readable(plot, "Count")
    assert np.shares_memory(plot.values, retained)
    assert plot.y_logarithmic
    assert (plot.x_grid_divisions, plot.y_grid_divisions) == (12, 12)

    plot.clear()
    _compact_at_width(plot, 260)
    assert plot.minimumHeight() < populated_height
    _assert_readable(plot, "Count")
    assert plot.values.size == 0
    _populate(plot)
    _compact_at_width(plot, 260)
    _assert_readable(plot, "Count")


def test_histogram_populated_while_hidden_is_readable_on_first_show(qtbot):
    parent = QWidget()
    qtbot.addWidget(parent)
    layout = QVBoxLayout(parent)
    plot = DetailedHistogramPlot(parent, minimum_plot_height=160)
    layout.addWidget(plot)
    parent.resize(260, 120)
    _font(plot, 13)
    _populate(plot, label="Probability density")
    assert not plot.isVisible()

    parent.show()
    _flush_layout()

    assert plot.isVisible()
    _assert_readable(plot, "Probability density")
    assert plot.height() >= plot.minimumHeight()


def test_short_inspector_scrolls_vertically_instead_of_squeezing_plot(qtbot):
    scroll = QScrollArea()
    qtbot.addWidget(scroll)
    scroll.setWidgetResizable(True)
    body = QWidget()
    layout = QVBoxLayout(body)
    layout.setContentsMargins(0, 0, 0, 0)
    plot = DetailedHistogramPlot(body, minimum_plot_height=160)
    layout.addWidget(plot)
    scroll.setWidget(body)
    scroll.resize(280, 220)
    _font(plot, 13)
    _populate(plot, label="Probability density")
    scroll.show()
    _flush_layout()

    assert plot.minimumHeight() > scroll.viewport().height()
    assert scroll.verticalScrollBar().maximum() > 0
    assert scroll.horizontalScrollBar().maximum() == 0
    _assert_readable(plot, "Probability density")


def test_real_inspector_uses_drawable_height_contract(qtbot):
    from napari_vipp._tests.test_widget import _Viewer
    from napari_vipp._widget import VippWidget

    widget = VippWidget(_Viewer(np.zeros((4, 5))), defer_initial_run=True)
    qtbot.addWidget(widget)
    widget.run_pipeline = lambda *args, **kwargs: None
    plot = widget.histogram_result_plot
    _font(plot, 13)
    plot.resize(260, 1)
    _populate(plot, label="Probability density")

    # Check the real object while its inspector is hidden: the old 165-pixel
    # override must not survive, nor may a standalone default disguise it.
    assert plot.minimumHeight() > 220
    _assert_readable(plot, "Probability density")


def test_detached_plot_and_dialog_keep_existing_minimums(qtbot):
    from napari_vipp.ui.histogram_dialog import HistogramDialog

    plot = DetailedHistogramPlot()
    qtbot.addWidget(plot)
    dialog = HistogramDialog()
    qtbot.addWidget(dialog)
    for target, minimum in ((plot, 220), (dialog.plot, 280)):
        _font(target, 13)
        _populate(target, label="Probability density")
        target.resize(520, minimum)
        _flush_layout()
        assert target.minimumHeight() == minimum


def test_narrow_embedded_plot_uses_full_bands_for_horizontal_labels(qtbot):
    from napari_vipp._tests.test_histogram_dialog import (
        _RecordingPainter,
        _text_vertical_bounds,
    )

    plot = DetailedHistogramPlot(minimum_plot_height=160)
    qtbot.addWidget(plot)
    _font(plot, 13)
    _populate(plot)
    plot.show()
    _compact_at_width(plot, 260)
    outer = plot.rect().adjusted(8, 8, -8, -8)
    painter = _RecordingPainter(plot.font())

    plot._draw_title_and_labels(painter, outer, plot._plot_rect())

    text_calls = {item[2]: item for item in painter.text_calls}
    for expected in ("Intensity Histogram", "Input value (a.u.)"):
        assert expected in text_calls
        call = text_calls[expected]
        x, _baseline, _text, font = call
        metrics = QFontMetrics(font)
        assert x >= outer.left()
        assert x + metrics.horizontalAdvance(expected) <= outer.right() + 1
        top, bottom = _text_vertical_bounds(call)
        assert top >= outer.top()
        assert bottom <= outer.bottom()
