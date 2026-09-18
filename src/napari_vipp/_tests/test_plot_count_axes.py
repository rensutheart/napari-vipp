"""Count axis labels express integer observations without altering plot data."""

import re
from dataclasses import replace

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from napari_vipp.core.plot_rendering import build_plot_figure, export_plot_result
from napari_vipp.core.result_plots import build_plot_result
from napari_vipp.core.tables import TableData


def _histogram(counts=(12, 5, 1), **params):
    table = TableData(
        ("area",),
        tuple(
            (index + 0.25,) for index, count in enumerate(counts) for _ in range(count)
        ),
        column_units=(("area", "pixels"),),
    )
    return build_plot_result(
        table, plot_type="Distribution", y_column="area", bins=len(counts), **params
    )


def _assert_integer_axis(axis):
    ticks = axis.get_majorticklocs()
    assert len(ticks)
    np.testing.assert_array_equal(ticks, np.floor(ticks))
    assert not len(axis.get_minorticklocs())
    labels = [label.get_text() for label in axis.get_ticklabels()]
    assert all(re.fullmatch(r"-?[\d,]+", label) for label in labels)
    assert not axis.get_offset_text().get_text()


@pytest.mark.parametrize("counts", [(1, 1), (1, 2), (12, 5, 1)])
@pytest.mark.parametrize("log_y", [False, True])
@pytest.mark.parametrize("compact", [False, True])
def test_histogram_counts_have_integer_ticks_at_all_sizes(counts, log_y, compact):
    result = _histogram(counts, log_y=log_y)
    snapshot = result.source_table.rows, result.series, result.plotted_table.rows
    figure = build_plot_figure(result, compact=compact, display_only=compact)
    canvas = FigureCanvasAgg(figure)
    for size in ((6.8, 4.6), (3.2, 2.8), (9, 6)):
        figure.set_size_inches(size)
        canvas.draw()
        axis = figure.axes[0].yaxis
        _assert_integer_axis(axis)
        if log_y:
            assert np.all(axis.get_majorticklocs() >= 1)
        if max(counts) <= 2:
            assert max(counts) in axis.get_majorticklocs()
        np.testing.assert_array_equal(
            figure.axes[0].patches[0].get_data().values, counts
        )
    assert snapshot == (
        result.source_table.rows,
        result.series,
        result.plotted_table.rows,
    )


def test_large_count_labels_do_not_use_decimal_scaled_offsets():
    result = _histogram()
    # Test display range independently of allocating millions of source rows.
    series = replace(result.series[0], histogram_values=(1_000_000, 250_000, 1))
    figure = build_plot_figure(replace(result, series=(series,)))
    FigureCanvasAgg(figure).draw()
    _assert_integer_axis(figure.axes[0].yaxis)
    assert any("," in label.get_text() for label in figure.axes[0].get_yticklabels())


@pytest.mark.parametrize("distribution", ["Histogram", "Cumulative"])
def test_percentage_axes_retain_fractional_ticks(distribution):
    result = _histogram((1,) * 60, normalization="Percent", distribution=distribution)
    figure = build_plot_figure(result)
    if distribution == "Cumulative":
        figure.axes[0].set_ylim(99, 100)
    FigureCanvasAgg(figure).draw()
    ticks = figure.axes[0].get_yticks()
    assert np.any(ticks != np.floor(ticks))


def _measurements(
    *,
    units="count",
    point_unit="Objects",
    plot_type="Scatter",
    logarithmic=False,
    values=((1, 1), (2, 2)),
):
    return build_plot_result(
        TableData(
            ("image_id", "count_a", "count_b"),
            tuple(("one", y, x) for y, x in values),
            column_units=(("count_a", units), ("count_b", units)),
        ),
        y_column="count_a",
        x_column="count_b",
        plot_type=plot_type,
        point_unit=point_unit,
        image_column="image_id",
        log_x=logarithmic,
        log_y=logarithmic,
    )


@pytest.mark.parametrize("plot_type", ["Scatter", "Compare groups", "Distribution"])
@pytest.mark.parametrize("logarithmic", [False, True])
def test_explicit_raw_count_units_use_integer_measurement_axes(plot_type, logarithmic):
    result = _measurements(plot_type=plot_type, logarithmic=logarithmic)
    figure = build_plot_figure(result)
    FigureCanvasAgg(figure).draw()
    axes = figure.axes[0]
    _assert_integer_axis(axes.xaxis if plot_type == "Distribution" else axes.yaxis)
    if plot_type == "Scatter":
        _assert_integer_axis(axes.xaxis)


@pytest.mark.parametrize("units", ["", "count/mm", "ratio"])
def test_names_and_integer_values_or_count_densities_do_not_imply_count_axes(units):
    figure = build_plot_figure(_measurements(units=units))
    FigureCanvasAgg(figure).draw()
    for axis in (figure.axes[0].xaxis, figure.axes[0].yaxis):
        ticks = axis.get_majorticklocs()
        assert np.any(ticks != np.floor(ticks))


@pytest.mark.parametrize("plot_type", ["Scatter", "Compare groups", "Distribution"])
def test_fractional_values_with_inherited_count_units_keep_fractional_axes(plot_type):
    result = _measurements(values=((1.25, 1.5), (2.25, 2.5)), plot_type=plot_type)
    figure = build_plot_figure(result)
    FigureCanvasAgg(figure).draw()
    axes = figure.axes[0]
    measurement_axis = axes.xaxis if plot_type == "Distribution" else axes.yaxis
    ticks = measurement_axis.get_majorticklocs()
    assert np.any(ticks != np.floor(ticks))
    if plot_type == "Distribution":
        # Frequencies count rows even when the measured values are fractional.
        _assert_integer_axis(axes.yaxis)
    elif plot_type == "Scatter":
        ticks = axes.xaxis.get_majorticklocs()
        assert np.any(ticks != np.floor(ticks))
        np.testing.assert_array_equal(
            axes.collections[0].get_offsets(), [[1.5, 1.25], [2.5, 2.25]]
        )
    assert result.series[0].y == (1.25, 2.25)


def test_fractional_count_guard_applies_independently_to_each_measurement_axis():
    figure = build_plot_figure(_measurements(values=((1, 1.5), (2, 2.5))))
    FigureCanvasAgg(figure).draw()
    _assert_integer_axis(figure.axes[0].yaxis)
    ticks = figure.axes[0].get_xticks()
    assert np.any(ticks != np.floor(ticks))


def test_mean_counts_remain_fractional_values_and_axes():
    result = _measurements(point_unit="Mean per image")
    assert result.series[0].x == result.series[0].y == (1.5,)
    figure = build_plot_figure(result)
    FigureCanvasAgg(figure).draw()
    for axis in (figure.axes[0].xaxis, figure.axes[0].yaxis):
        ticks = axis.get_majorticklocs()
        assert np.any(ticks != np.floor(ticks))
    np.testing.assert_array_equal(
        figure.axes[0].collections[0].get_offsets(), [[1.5, 1.5]]
    )


def test_histogram_of_mean_counts_keeps_fractional_values_but_integer_frequencies():
    result = _measurements(point_unit="Mean per image", plot_type="Distribution")
    figure = build_plot_figure(result)
    FigureCanvasAgg(figure).draw()
    _assert_integer_axis(figure.axes[0].yaxis)
    ticks = figure.axes[0].get_xticks()
    assert np.any(ticks != np.floor(ticks))
    assert result.series[0].y == (1.5,)


@pytest.mark.parametrize("log_y", [False, True])
def test_exported_count_axis_uses_same_integer_policy(tmp_path, log_y):
    result = _histogram((1, 2), log_y=log_y)
    target = tmp_path / "counts.svg"
    export_plot_result(result, target, width_mm=101.6, height_mm=76.2, dpi=150)
    svg = target.read_text(encoding="utf-8")
    y_axis = svg.split('id="matplotlib.axis_2"', 1)[1].split('id="patch_', 1)[0]
    labels = re.findall(r"<!-- ([\d.,]+) -->", y_axis)
    assert labels
    assert all(re.fullmatch(r"[\d,]+", label) for label in labels)
    assert "2" in labels
