"""Crowded categorical axes remain readable without changing measurement data."""

import io
import warnings

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from napari_vipp.core.plot_rendering import build_plot_figure
from napari_vipp.core.result_plots import build_plot_result
from napari_vipp.core.tables import TableData


def _crowded_result(*, strings=False, count=60):
    groups = tuple(
        ("Shared very long image identifier " * 5 + str(index))
        if strings
        else 19.987654321 + index * 0.123456789
        for index in range(count)
    )
    return _result_for_groups(groups)


def _result_for_groups(groups):
    table = TableData(
        ("label_id", "minor_axis_length_pixels", "major_axis_length_pixels"),
        tuple(
            (index + 1, group, 24 + index % 25) for index, group in enumerate(groups)
        ),
        column_units=(
            ("minor_axis_length_pixels", "pixels"),
            ("major_axis_length_pixels", "pixels"),
        ),
    )
    return build_plot_result(
        table,
        y_column="major_axis_length_pixels",
        group_column="minor_axis_length_pixels",
    )


def _assert_readable(figure, renderer):
    axes = figure.axes[0]
    # Calling get_xticklabels/get_xticks here would invoke the adaptive locator
    # again and could conceal a first-draw layout problem. Inspect its rendered
    # tick objects and formatter locations without requesting another update.
    locations = tuple(axes.xaxis.major.formatter.locs)
    labels = [
        tick.label1
        for tick in axes.xaxis.majorTicks[: len(locations)]
        if tick.label1.get_visible()
    ]
    boxes = [label.get_window_extent(renderer) for label in labels if label.get_text()]
    for left, right in zip(boxes, boxes[1:], strict=False):
        assert left.x1 <= right.x0 + 0.5, (left.bounds, right.bounds)
    xlabel_box = axes.xaxis.label.get_window_extent(renderer)
    note_boxes = [
        text.get_window_extent(renderer) for text in figure.texts if text.get_text()
    ]
    decorations = boxes + [xlabel_box] + note_boxes
    for box in decorations:
        assert box.x0 >= figure.bbox.x0 - 1
        assert box.x1 <= figure.bbox.x1 + 1
        assert box.y0 >= figure.bbox.y0 - 1
        assert box.y1 <= figure.bbox.y1 + 1
    for box in boxes:
        assert not box.overlaps(xlabel_box)
    for box in note_boxes:
        assert box.y1 <= xlabel_box.y0 + 0.5
    for artist in axes.collections:
        positions = axes.transData.transform(artist.get_offsets())
        assert np.all(positions[:, 0] >= axes.bbox.x0 - 0.5)
        assert np.all(positions[:, 0] <= axes.bbox.x1 + 0.5)
    return locations


def _assert_all_data_present(figure, result):
    axes = figure.axes[0]
    assert len(axes.collections) == len(result.series)
    for artist, series in zip(axes.collections, result.series, strict=True):
        assert artist.get_label() == series.name
        assert artist.vipp_source_rows == series.source_rows
        np.testing.assert_array_equal(artist.get_offsets()[:, 1], series.y)
    assert sum(len(artist.get_offsets()) for artist in axes.collections) == 60


@pytest.mark.parametrize("strings", [False, True])
@pytest.mark.parametrize("size,compact", [((3, 2.8), True), ((9, 6), False)])
def test_crowded_group_labels_fit_without_changing_data(strings, size, compact):
    result = _crowded_result(strings=strings)
    figure = build_plot_figure(result, size_inches=size, compact=compact)
    canvas = FigureCanvasAgg(figure)
    with warnings.catch_warnings(record=True) as caught:
        canvas.draw()
    assert not [warning for warning in caught if "constrained_layout" in str(warning)]
    ticks = _assert_readable(figure, canvas.get_renderer())
    assert 1 <= len(ticks) < 60
    assert "All 60 groups plotted" in figure.texts[0].get_text()
    labels = figure.axes[0].xaxis.get_major_formatter().labels
    assert len(labels) == len(set(labels)) == 60
    _assert_all_data_present(figure, result)


def test_categorical_labels_adapt_both_ways_on_same_figure_resize():
    result = _crowded_result()
    figure = build_plot_figure(result, size_inches=(9, 6), compact=True)
    canvas = FigureCanvasAgg(figure)
    layouts = []
    for size in ((9, 6), (3, 2.8), (9, 6)):
        figure.set_size_inches(size)
        canvas.draw()
        layouts.append(_assert_readable(figure, canvas.get_renderer()))
        _assert_all_data_present(figure, result)
    assert len(layouts[1]) < len(layouts[0])
    assert layouts[2] == layouts[0]


def test_dpi_alone_does_not_change_categorical_label_budget():
    layouts = []
    for dpi in (100, 300):
        figure = build_plot_figure(_crowded_result(), size_inches=(9, 6), dpi=dpi)
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        layouts.append(_assert_readable(figure, canvas.get_renderer()))
    assert layouts[0] == layouts[1]


@pytest.mark.parametrize("format_name", ["png", "svg", "pdf"])
@pytest.mark.parametrize("strings,compact", [(False, False), (True, True)])
def test_first_export_draw_adapts_labels_for_its_actual_backend(
    format_name, strings, compact
):
    result = _crowded_result(strings=strings)
    figure = build_plot_figure(
        result,
        size_inches=(3, 2.8) if compact else (4, 3),
        dpi=100,
        publication=True,
        compact=compact,
    )
    canvas = FigureCanvasAgg(figure)
    layouts = []

    def inspect_draw(event):
        layouts.append(_assert_readable(figure, event.renderer))
        _assert_all_data_present(figure, result)

    canvas.mpl_connect("draw_event", inspect_draw)
    with warnings.catch_warnings(record=True) as caught:
        figure.savefig(io.BytesIO(), format=format_name)
    assert not [warning for warning in caught if "constrained_layout" in str(warning)]
    assert layouts
    assert all(1 <= len(ticks) < 60 for ticks in layouts)


def test_adjacent_float_groups_are_not_merged_by_display_rounding():
    groups = (1.0, float(np.nextafter(1.0, 2.0)), float(np.nextafter(1.0, 0.0)))
    result = _result_for_groups(groups)
    figure = build_plot_figure(result)
    FigureCanvasAgg(figure).draw()
    labels = figure.axes[0].xaxis.get_major_formatter().labels
    assert len(set(labels)) == len(groups)
    assert tuple(float(label) for label in labels) == groups
    assert tuple(series.name for series in result.series) == tuple(map(str, groups))


def test_numeric_looking_string_identifiers_remain_literal():
    groups = ("001", "1", "1.0", "0.0001", "1e-04", "$5")
    result = _result_for_groups(groups)
    figure = build_plot_figure(result)
    FigureCanvasAgg(figure).draw()
    formatter = figure.axes[0].xaxis.get_major_formatter()
    assert tuple(formatter.labels) == groups
    assert all(not label.get_parse_math() for label in figure.axes[0].get_xticklabels())


def test_wide_integer_group_identifiers_do_not_lose_precision():
    groups = (2**100, 2**100 + 1, 2**100 + 2)
    result = _result_for_groups(groups)
    figure = build_plot_figure(result)
    FigureCanvasAgg(figure).draw()
    labels = figure.axes[0].xaxis.get_major_formatter().labels
    assert len(set(labels)) == len(groups)
    assert tuple(label.replace("\n", "") for label in labels) == tuple(map(str, groups))
    assert tuple(series.name for series in result.series) == tuple(map(str, groups))
