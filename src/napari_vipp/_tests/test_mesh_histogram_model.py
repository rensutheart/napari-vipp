"""Exact mesh-filter summaries and fractional calibrated plot coordinates."""

import numpy as np
import pytest

from napari_vipp.core.tables import TableData
from napari_vipp.ui.mesh_histogram import mesh_filter_decimals, mesh_filter_histogram
from napari_vipp.ui.plots import HistogramPlot, _format_histogram_label


def _table(values, unit="nm^3", prop="mesh_volume_physical"):
    return TableData((prop,), tuple((v,) for v in values), column_units=((prop, unit),))


@pytest.mark.parametrize("log", [False, True])
def test_input_distribution_does_not_expand_to_default_maximum(log):
    table = _table([0.001, 0.01, 0.25, float("nan")])
    result = mesh_filter_histogram(
        table, "mesh_volume_physical", 0.01, 1e12, "In range", log=log
    )
    assert result.label == "Volume (nm³)"
    assert result.x_range == (0.0, 0.25)
    assert result.counts.sum() == 3
    assert result.omitted == 1
    assert "4 input objects · 2 match · 1 unavailable" in result.summary
    assert result.edges[-1] == 0.25
    assert result.edges.size == result.counts.size + 1


def test_outside_rule_excludes_endpoints_and_unavailable_measurements():
    result = mesh_filter_histogram(
        _table([1, 8, 27, float("nan")], "voxel^3"),
        "mesh_volume_physical",
        1,
        8,
        "Outside range",
    )
    assert "4 input objects · 1 match · 1 unavailable" in result.summary
    assert "not a foreground voxel count" in result.summary
    assert result.counts.sum() == 3


def test_wide_object_id_summary_does_not_round_integer_choices():
    big = 2**53
    result = mesh_filter_histogram(
        _table([big, big + 1, big + 2], "", "mesh_id"),
        "mesh_id",
        big + 1,
        big + 1,
        "In range",
    )
    assert "3 input objects · 1 match" in result.summary
    assert "exact plot precision" in result.summary
    assert not result.counts.size


@pytest.mark.parametrize("values", [[], [float("nan")]])
def test_empty_and_unavailable_meshes_have_no_invented_distribution(values):
    result = mesh_filter_histogram(
        _table(values), "mesh_volume_physical", 0, 1e12, "In range"
    )
    assert not result.counts.size
    assert result.omitted == len(values)


@pytest.mark.parametrize("minimum,maximum", [(8, 1), (False, 8), (0, float("inf"))])
def test_invalid_limits_do_not_claim_a_matching_filter(minimum, maximum):
    with pytest.raises(ValueError, match="limits"):
        mesh_filter_histogram(
            _table([1]), "mesh_volume_physical", minimum, maximum, "In range"
        )


def test_fractional_log_axis_markers_reach_full_plot_width(qtbot):
    plot = HistogramPlot()
    qtbot.addWidget(plot)
    plot.resize(420, 160)
    plot.set_histogram(np.array([1, 2]), False, x_range=(0.0, 0.25), x_scale="log")
    rect = plot._plot_rect()
    assert plot._x_fraction(0.25) == pytest.approx(1.0)
    for value in (0.0, 0.001, 0.1, 0.25):
        x = rect.left() + rect.width() * plot._x_fraction(value)
        assert plot._value_from_x(x, rect) == pytest.approx(value)


def test_mesh_limits_adapt_decimal_precision_for_small_physical_units():
    assert mesh_filter_decimals(_table([1, 8]), "mesh_volume_physical") == 6
    assert mesh_filter_decimals(_table([1e-12]), "mesh_volume_physical") == 15
    assert mesh_filter_decimals(None, "mesh_volume_physical", [1e-8, 100]) == 11
    assert mesh_filter_decimals(None, "mesh_id", [1, 100]) == 0


def test_physical_volume_labels_remain_readable_at_tiny_and_distant_bounds():
    assert _format_histogram_label(1e-12) == "1e-12"
    assert _format_histogram_label(1e12) == "1e+12"
    assert _format_histogram_label(2**63 + 1) == str(2**63 + 1)
