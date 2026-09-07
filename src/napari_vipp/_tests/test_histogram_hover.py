from __future__ import annotations

import numpy as np
import pytest
from qtpy.QtCore import QEvent, QPoint, Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QApplication

from napari_vipp.ui import plots


def _plot(qtbot, counts, **kwargs):
    plot = plots.HistogramPlot()
    qtbot.addWidget(plot)
    plot.resize(600, 180)
    plot.set_plot_labels(x_axis_label="Volume (voxels)", y_axis_label="Objects")
    plot.set_histogram(counts, log_scale=kwargs.pop("log_scale", False), **kwargs)
    plot.show()
    qtbot.waitExposed(plot)
    return plot


def _point(plot, index):
    rect = plot._plot_rect()
    bins = plot._hover_counts.shape[1]
    fraction = (index + 0.5) / bins if bins <= 8 else index / (bins - 1)
    return QPoint(
        min(rect.left() + int(fraction * rect.width()), rect.right()),
        rect.top() + 3,
    )


@pytest.mark.parametrize("bins", [1, 2, 8, 9, 64, 256])
def test_hover_bin_lookup_matches_discrete_bars_and_dense_strokes(qtbot, bins):
    plot = _plot(qtbot, np.arange(bins))
    for index in range(bins):
        assert plot._bin_at_point(_point(plot, index)) == index
    assert plot._bin_at_point(QPoint(0, 0)) is None


@pytest.mark.parametrize("log_scale", [False, True])
def test_counts_are_original_integers_and_do_not_alias_input(qtbot, log_scale):
    counts = np.array([2**24 + 1, 2**53 + 1], dtype=np.uint64)
    counts.flags.writeable = False
    plot = _plot(qtbot, counts, log_scale=log_scale)
    assert "Objects: 16,777,217" in plot._bin_tooltip(0)
    assert "Objects: 9,007,199,254,740,993" in plot._bin_tooltip(1)
    assert not np.shares_memory(counts, plot._hover_counts)
    np.testing.assert_array_equal(counts, [2**24 + 1, 2**53 + 1])


@pytest.mark.parametrize(
    "endpoint", [None, 9735.0, np.nextafter(9735.0, 0), np.nextafter(9735.0, np.inf)]
)
def test_log_size_hover_uses_original_units_not_logarithms(qtbot, endpoint):
    edges = np.expm1(np.linspace(0, np.log1p(9735), 9))
    if endpoint is not None:
        edges[-1] = endpoint
    plot = _plot(
        qtbot,
        np.array([23, 0, 1, 0, 1, 0, 0, 2]),
        x_range=(0, 9735),
        x_scale="log",
        bin_edges=edges,
    )
    assert "Objects: 23" in plot._bin_tooltip(0)
    assert "Volume (voxels): [0, " in plot._bin_tooltip(0)
    # expm1/log1p can end one ULP away from an integer on different platforms.
    # Both grouped integer and ungrouped decimal text represent the original
    # voxel units; assert the value and closed endpoint, not the grouping.
    upper = plot._bin_tooltip(7).splitlines()[1].rsplit(", ", 1)[1]
    assert upper.endswith("]")
    assert float(upper[:-1].replace(",", "")) == pytest.approx(9735, rel=0, abs=1e-8)
    assert "Objects: 0" in plot._bin_tooltip(1)
    assert plot._bin_at_point(_point(plot, 7)) == 7
    np.testing.assert_array_equal(plot._bin_edges, edges)


def test_ranges_use_half_open_intervals_with_closed_final_bin(qtbot):
    plot = _plot(qtbot, [3, 5], bin_edges=np.array([0, 10, 20]))
    assert "[0, 10)" in plot._bin_tooltip(0)
    assert "[10, 20]" in plot._bin_tooltip(1)


def test_unknown_edges_are_not_invented_from_endpoint_labels(qtbot):
    plot = _plot(qtbot, [3, 5], x_range=(0, 1))
    assert plot._bin_tooltip(0) == "Bin 1 of 2\nObjects: 3"


@pytest.mark.parametrize("edges", [[0, 1e-12, 2e-12], [1, 1 + 1e-10, 1 + 2e-10]])
def test_fine_property_bin_bounds_remain_distinguishable(qtbot, edges):
    plot = _plot(qtbot, [3, 5], bin_edges=np.array(edges))
    text = plot._bin_tooltip(0).splitlines()[1]
    left, right = text.split("[")[1].rstrip(")").split(", ")
    assert float(left) < float(right)


def test_multiseries_counts_remain_separate(qtbot):
    plot = _plot(qtbot, [[3, 5], [7, 11]])
    assert plot._bin_tooltip(0) == (
        "Bin 1 of 2\nSeries 1 · Objects: 3\nSeries 2 · Objects: 7"
    )


def test_hover_reuses_cached_values_and_only_updates_on_a_new_bin(qtbot, monkeypatch):
    plot = _plot(qtbot, [23, 0, 2])
    shown = []
    monkeypatch.setattr(plots.QToolTip, "showText", lambda *args: shown.append(args[1]))

    def no_recalculation(*args, **kwargs):
        pytest.fail("Hover must not calculate another histogram")

    monkeypatch.setattr(np, "histogram", no_recalculation)
    plot._show_bin_hover(_point(plot, 0))
    plot._show_bin_hover(_point(plot, 0) + QPoint(1, 1))
    assert len(shown) == 1
    assert plot.toolTip().endswith("Objects: 23")
    # A zero-height bar is reachable well above its baseline, just like a tall bar.
    plot._show_bin_hover(_point(plot, 1))
    assert len(shown) == 2
    assert plot.toolTip().endswith("Objects: 0")
    plot._show_bin_hover(QPoint(0, 0))
    assert plot.toolTip() == ""


def test_reduced_display_does_not_mislabel_a_merged_stroke(qtbot):
    plot = _plot(qtbot, np.arange(256))
    plot.resize(200, 180)
    assert plot._bin_at_point(plot._plot_rect().center()) is None
    plot.resize(600, 180)
    assert plot._bin_at_point(_point(plot, 127)) == 127


@pytest.mark.parametrize("action", ["replace", "clear", "resize", "hide", "leave"])
def test_hover_is_cleared_when_data_or_geometry_becomes_stale(qtbot, action):
    plot = _plot(qtbot, [3, 5], bin_edges=np.array([0, 10, 20]))
    plot._show_bin_hover(_point(plot, 0))
    assert plot.toolTip()
    if action == "replace":
        plot.set_histogram([8, 9], False)
        assert plot._bin_edges is None
    elif action == "clear":
        plot.set_histogram(None, False)
        assert plot._bin_at_point(plot._plot_rect().center()) is None
    elif action == "resize":
        plot.resize(300, 180)
    elif action == "hide":
        plot.hide()
    else:
        QApplication.sendEvent(plot, QEvent(QEvent.Leave))
    assert plot.toolTip() == ""
    assert plot._hovered_bin is None


def test_marker_drag_takes_precedence_over_bin_hover(qtbot):
    plot = _plot(
        qtbot,
        np.ones(8),
        x_range=(0, 100),
        markers=[("min", 50, QColor("orange"))],
        draggable_markers={"min"},
    )
    plot._show_bin_hover(_point(plot, 0))
    marker = plot._plot_rect().center()
    qtbot.mouseMove(plot, marker)
    assert plot.toolTip() == ""
    with qtbot.waitSignal(plot.markerChanged):
        qtbot.mousePress(plot, Qt.LeftButton, pos=marker)
        qtbot.mouseMove(plot, _point(plot, 6))
        assert plot.toolTip() == ""
        qtbot.mouseRelease(plot, Qt.LeftButton, pos=_point(plot, 6))
    assert plot._drag_marker is None


@pytest.mark.parametrize("edges", [[0, 1], [0, 0, 1], [0, 2, 1], [0, 1, np.nan]])
def test_incorrect_hover_edges_are_rejected(qtbot, edges):
    plot = _plot(qtbot, [3, 5])
    with pytest.raises(ValueError, match="edge per bin boundary"):
        plot.set_histogram([3, 5], False, bin_edges=np.array(edges))
