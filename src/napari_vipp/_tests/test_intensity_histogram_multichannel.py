from __future__ import annotations

import numpy as np
import pytest

from napari_vipp._tests.test_widget import _Viewer
from napari_vipp._widget import VippWidget
from napari_vipp.core.channel_colors import color_value_to_rgb
from napari_vipp.core.operations import (
    intensity_histogram,
    intensity_histogram_table_columns,
)

SCALAR_HISTOGRAM_COLUMNS = (
    "bin_index",
    "bin_left",
    "bin_right",
    "bin_center",
    "bin_width",
    "count",
    "fraction",
    "density",
    "cumulative_count",
    "cumulative_fraction",
)

MULTICHANNEL_HISTOGRAM_COLUMNS = SCALAR_HISTOGRAM_COLUMNS + (
    "series_index",
    "series_name",
    "series_color",
)

RGB_NAMES = ("Red", "Green", "Blue")
RGB_COLORS = (0xFF0000, 0x00FF00, 0x0000FF)


def _rgb_fixture() -> np.ndarray:
    """Return YXrgb data with a deliberately different distribution per channel."""

    return np.asarray(
        [
            [[0, 1, 3], [0, 2, 3]],
            [[1, 2, 3], [1, 3, 3]],
        ],
        dtype=np.uint8,
    )


def _column(table, name: str, *, dtype=None) -> np.ndarray:
    index = table.columns.index(name)
    return np.asarray([row[index] for row in table.rows], dtype=dtype)


def _series_rows(table, series_index: int) -> tuple[tuple[object, ...], ...]:
    index_column = table.columns.index("series_index")
    return tuple(row for row in table.rows if int(row[index_column]) == series_index)


def _series_column(table, series_index: int, name: str, *, dtype=None):
    value_column = table.columns.index(name)
    return np.asarray(
        [row[value_column] for row in _series_rows(table, series_index)],
        dtype=dtype,
    )


def _assert_rgb_colors(values) -> None:
    colors = [
        np.asarray(value.getRgbF()[:3], dtype=np.float32)
        if hasattr(value, "getRgbF")
        else color_value_to_rgb(value)
        for value in values
    ]
    assert all(color is not None for color in colors)
    np.testing.assert_allclose(
        np.asarray(colors, dtype=np.float64),
        np.eye(3, dtype=np.float64),
        atol=1 / 255,
    )


def _assert_plot_is_rgb(plot) -> None:
    np.testing.assert_array_equal(
        np.asarray(plot.values, dtype=np.float64),
        np.asarray(
            [
                [2, 2, 0, 0],
                [0, 1, 2, 1],
                [0, 0, 0, 4],
            ],
            dtype=np.float64,
        ),
    )
    assert plot._series_labels == RGB_NAMES
    _assert_rgb_colors(plot._series_colors)


def test_explicit_rgb_histogram_exports_one_independent_series_per_channel():
    table = intensity_histogram(
        _rgb_fixture(),
        bin_count=4,
        range_mode="Custom range",
        custom_min=0.0,
        custom_max=4.0,
        bin_spacing="Linear",
        channel_axis=2,
        channel_axis_name="rgb",
        channel_names=RGB_NAMES,
        channel_colors=RGB_COLORS,
    )

    assert table.columns == MULTICHANNEL_HISTOGRAM_COLUMNS
    assert table.row_count == 12
    np.testing.assert_array_equal(
        _column(table, "series_index", dtype=np.int64),
        np.repeat(np.arange(3, dtype=np.int64), 4),
    )
    assert tuple(dict.fromkeys(_column(table, "series_name").tolist())) == RGB_NAMES
    _assert_rgb_colors(tuple(dict.fromkeys(_column(table, "series_color").tolist())))

    expected_counts = (
        np.asarray([2, 2, 0, 0]),
        np.asarray([0, 1, 2, 1]),
        np.asarray([0, 0, 0, 4]),
    )
    expected_edges = np.arange(5, dtype=np.float64)
    for channel_index, counts in enumerate(expected_counts):
        np.testing.assert_array_equal(
            _series_column(table, channel_index, "count", dtype=np.int64),
            counts,
        )
        left = _series_column(table, channel_index, "bin_left", dtype=np.float64)
        right = _series_column(
            table,
            channel_index,
            "bin_right",
            dtype=np.float64,
        )
        np.testing.assert_allclose(
            np.concatenate((left[:1], right)),
            expected_edges,
        )
        assert _series_column(
            table,
            channel_index,
            "fraction",
            dtype=np.float64,
        ).sum() == pytest.approx(1.0)
        assert _series_column(
            table,
            channel_index,
            "cumulative_fraction",
            dtype=np.float64,
        )[-1] == pytest.approx(1.0)

    metadata = table.histogram_metadata
    assert metadata is not None
    assert metadata.input_value_count == _rgb_fixture().size
    assert metadata.binned_value_count == _rgb_fixture().size


def test_explicit_c_axis_preserves_identity_and_per_series_exclusion_counts():
    data = np.asarray(
        [
            [[0.0, 1.0], [np.nan, 9.0]],
            [[0.0, 1.0], [2.0, 4.0]],
        ],
        dtype=np.float64,
    )
    names = ("DNA", "Actin")
    colors = (0xFF00FF, 0x00FFFF)

    table = intensity_histogram(
        data,
        bin_count=4,
        range_mode="Custom range",
        custom_min=0.0,
        custom_max=4.0,
        bin_spacing="Linear",
        channel_axis=0,
        channel_axis_name="c",
        channel_names=names,
        channel_colors=colors,
    )

    assert table.columns == MULTICHANNEL_HISTOGRAM_COLUMNS
    assert tuple(dict.fromkeys(_column(table, "series_name").tolist())) == names
    table_colors = tuple(dict.fromkeys(_column(table, "series_color").tolist()))
    np.testing.assert_allclose(
        np.asarray([color_value_to_rgb(value) for value in table_colors]),
        np.asarray(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        ),
        atol=1 / 255,
    )
    np.testing.assert_array_equal(
        _series_column(table, 0, "count", dtype=np.int64),
        [1, 1, 0, 0],
    )
    np.testing.assert_array_equal(
        _series_column(table, 1, "count", dtype=np.int64),
        [1, 1, 1, 1],
    )

    metadata = table.histogram_metadata
    assert metadata is not None
    assert len(metadata.series) == 2
    first, second = metadata.series
    assert (
        first.series_index,
        first.series_name,
        first.input_value_count,
        first.finite_value_count,
        first.nan_value_count,
        first.binned_value_count,
        first.overflow_count,
    ) == (0, "DNA", 4, 3, 1, 2, 1)
    assert (
        second.series_index,
        second.series_name,
        second.input_value_count,
        second.finite_value_count,
        second.nan_value_count,
        second.binned_value_count,
        second.overflow_count,
    ) == (1, "Actin", 4, 4, 0, 4, 0)
    np.testing.assert_allclose(
        np.asarray(
            [
                color_value_to_rgb(first.series_color),
                color_value_to_rgb(second.series_color),
            ]
        ),
        np.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
        atol=1 / 255,
    )
    assert metadata.input_value_count == 8
    assert metadata.nan_value_count == 1
    assert metadata.binned_value_count == 6
    assert metadata.overflow_count == 1


def test_rgb_histogram_inspector_and_popout_show_the_same_labeled_series(qtbot):
    data = _rgb_fixture()
    widget = VippWidget(
        _Viewer(data, metadata={"axes": "Y,X,rgb"}),
    )
    widget._should_run_pipeline_in_background = lambda *_args, **_kwargs: False
    qtbot.addWidget(widget)
    input_state = widget.pipeline.output_states["input"]
    assert input_state.axis_order == "Y,X,rgb"
    assert input_state.axes[-1].is_explicit

    histogram = widget.add_node_from_palette("intensity_histogram")
    for name, value in {
        "bin_count": 4,
        "range_mode": "Custom range",
        "custom_min": 0.0,
        "custom_max": 4.0,
        "bin_spacing": "Linear",
    }.items():
        widget.pipeline.set_param(histogram.id, name, value)
    widget._connect_nodes("input", histogram.id)
    assert any(
        connection.source_id == "input" and connection.target_id == histogram.id
        for connection in widget.pipeline.connections
    )
    assert widget.pipeline.outputs["input"] is not None
    source_payloads, _source_layers = widget._source_payloads_for_pipeline()
    payload_state = source_payloads["input"].image_state
    assert payload_state is not None
    assert payload_state.axes[-1].is_explicit
    prepared = widget.pipeline.prepare_node_call(histogram.id)
    assert prepared is not None
    assert prepared.input_states == (input_state,)
    assert np.asarray(prepared.inputs[0]).shape == (2, 2, 3)
    assert prepared.kwargs["channel_axis"] == 2
    widget.pipeline.discard_cached_results({histogram.id})
    widget.pipeline.run(
        None,
        source_payloads=source_payloads,
        dirty_node_ids={histogram.id},
        manual_node_ids={histogram.id},
        target_node_ids={histogram.id},
    )
    output = widget.pipeline.outputs[histogram.id]
    assert output is not None, widget._node_execution_ui_state(histogram.id)
    assert output.row_count == 12
    widget.graph_view.select_node(histogram.id)

    _assert_plot_is_rgb(widget.histogram_result_plot)
    assert "3 channels" in widget.histogram_semantic_summary.text().casefold()

    widget._open_histogram_dialog()
    dialog = widget._histogram_dialog
    assert dialog is not None
    assert dialog.table is widget.pipeline.outputs[histogram.id]
    _assert_plot_is_rgb(dialog.plot)


def test_shape_alone_does_not_split_scalar_histogram_or_change_legacy_schema(qtbot):
    table = intensity_histogram(
        _rgb_fixture(),
        bin_count=4,
        range_mode="Custom range",
        custom_min=0.0,
        custom_max=4.0,
        bin_spacing="Linear",
    )

    assert intensity_histogram_table_columns() == SCALAR_HISTOGRAM_COLUMNS
    assert table.columns == SCALAR_HISTOGRAM_COLUMNS
    assert table.row_count == 4
    np.testing.assert_array_equal(_column(table, "count"), [2, 3, 2, 5])

    from napari_vipp.ui.histogram_dialog import HistogramDialog

    dialog = HistogramDialog()
    qtbot.addWidget(dialog)
    dialog.set_histogram(table)

    assert dialog.plot.values.shape == (1, 4)
    np.testing.assert_array_equal(dialog.plot.values, [[2.0, 3.0, 2.0, 5.0]])
    assert dialog.plot._series_labels == ("Histogram",)
