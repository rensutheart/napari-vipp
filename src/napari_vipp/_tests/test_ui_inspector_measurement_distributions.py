from __future__ import annotations

import numpy as np

import napari_vipp._widget as widget_module
from napari_vipp._tests.test_ui_inspector_widget_integration import (
    _assert_visible_order,
    _publish_array_output,
    _publish_table_output,
    _select,
    _widget,
)
from napari_vipp.core.pipeline import EXECUTION_READY
from napari_vipp.core.tables import TableData, TableState


def _labels() -> np.ndarray:
    labels = np.zeros((8, 10), dtype=np.int32)
    labels[1:3, 1:4] = 1
    labels[4:7, 5:9] = 2
    return labels


def test_measure_objects_shows_exact_area_input_before_results(qtbot):
    widget = _widget(qtbot, np.zeros((8, 10), dtype=np.uint16), axes="YX")
    labels_node = widget.add_node_from_palette("label_connected_components")
    measurements = widget.add_node_from_palette("measure_objects")
    labels = _labels()
    _publish_array_output(widget, labels_node.id, labels, axes="YX")
    widget._connect_nodes(labels_node.id, measurements.id)
    _publish_table_output(widget, measurements.id)

    _select(widget, measurements.id)

    _assert_visible_order(
        widget,
        widget.parameter_group,
        widget.histograms_section,
        widget.table_group,
    )
    assert widget.histograms_section.title() == "Input distributions"
    assert widget.histograms_section.summary_label.text() == "Labels"
    assert not widget.measurement_object_size_histogram_group.isHidden()
    assert widget.measurement_intensity_histogram_group.isHidden()
    size_plot = widget.measurement_object_size_histogram_plot
    assert size_plot._title == "Object area"
    assert size_plot._x_axis_label == "Area (pixels)"
    assert size_plot._y_axis_label == "Objects"
    assert int(np.asarray(size_plot._counts).sum()) == 2
    assert size_plot._x_range == (0.0, 12.0)
    assert 0 < widget.table_preview.height() < 180


def test_measure_objects_intensity_adds_exact_full_intensity_input(qtbot):
    intensity = np.arange(80, dtype=np.uint16).reshape(8, 10)
    widget = _widget(qtbot, intensity, axes="YX")
    labels_node = widget.add_node_from_palette("label_connected_components")
    measurements = widget.add_node_from_palette("measure_objects_intensity")
    labels = _labels()
    _publish_array_output(widget, "input", intensity, axes="YX")
    _publish_array_output(widget, labels_node.id, labels, axes="YX")
    widget._connect_nodes(labels_node.id, measurements.id, target_port=0)
    widget._connect_nodes("input", measurements.id, target_port=1)
    _publish_table_output(widget, measurements.id)

    _select(widget, measurements.id)

    assert widget.histograms_section.summary_label.text() == "Labels + intensity"
    assert not widget.measurement_object_size_histogram_group.isHidden()
    assert not widget.measurement_intensity_histogram_group.isHidden()
    assert not widget.histogram_controls_row.isHidden()
    assert widget.histogram_scope_combo.isHidden()
    assert not widget.histogram_log_checkbox.isHidden()
    intensity_plot = widget.measurement_intensity_histogram_plot
    assert intensity_plot._title == "Intensity input"
    assert intensity_plot._x_axis_label == "Intensity (a.u.)"
    assert intensity_plot._y_axis_label == "Voxels"
    assert int(np.asarray(intensity_plot._counts).sum()) == intensity.size
    assert intensity_plot._x_range is not None
    assert intensity_plot._x_range[0] <= float(intensity.min())
    assert intensity_plot._x_range[1] >= float(intensity.max())

    widget.histogram_log_checkbox.setChecked(True)
    assert widget.measurement_object_size_histogram_plot._log_scale
    assert widget.measurement_intensity_histogram_plot._log_scale


def test_measurement_results_table_grows_to_a_capped_scrollable_height(qtbot):
    widget = _widget(qtbot, np.zeros((8, 10), dtype=np.uint16), axes="YX")
    labels_node = widget.add_node_from_palette("label_connected_components")
    measurements = widget.add_node_from_palette("measure_objects")
    _publish_array_output(widget, labels_node.id, _labels(), axes="YX")
    widget._connect_nodes(labels_node.id, measurements.id)

    rows = tuple((index, index + 1) for index in range(240))
    table = TableData(("label", "area_pixels"), rows, name="Many objects")
    state = TableState(
        table.row_count,
        table.column_count,
        table.columns,
        source_name=table.name,
    )
    widget.pipeline.outputs[measurements.id] = table
    widget.pipeline.output_states[measurements.id] = state
    widget.pipeline.node_outputs[measurements.id] = [table]
    widget.pipeline.node_output_states[measurements.id] = [state]
    widget.pipeline.completed_node_ids.add(measurements.id)
    widget.pipeline.node_execution_states[measurements.id] = EXECUTION_READY

    _select(widget, measurements.id)

    assert widget.table_preview.rowCount() == 200
    assert widget.table_preview.height() == 360


def test_result_preview_reserves_space_for_its_horizontal_scrollbar(qtbot):
    widget = _widget(qtbot)
    summary = widget.add_node_from_palette("summarize_skeleton_branches")
    columns = (
        "branch_length_voxels_q25",
        "branch_length_voxels_q75",
        "branch_tortuosity_mean",
        "branch_tortuosity_median",
        "branch_tortuosity_q25",
        "branch_tortuosity_q75",
    )
    table = TableData(
        columns,
        (tuple(float(index) for index in range(len(columns))),),
        name="Skeleton branch summary",
        table_kind="Skeleton branch summary",
    )
    widget.table_preview.setFixedWidth(420)
    _publish_table_output(widget, summary.id, table)

    _select(widget, summary.id)
    qtbot.waitUntil(
        lambda: widget.table_preview.horizontalScrollBar().maximum() > 0,
        timeout=1_000,
    )
    widget._sync_table_preview_geometry()

    header_height = max(
        widget.table_preview.horizontalHeader().height(),
        widget.table_preview.horizontalHeader().sizeHint().height(),
    )
    content_height = (
        header_height
        + widget.table_preview.rowHeight(0)
        + widget.table_preview.frameWidth() * 2
        + widget.table_preview.horizontalScrollBar().sizeHint().height()
    )
    assert widget.table_preview.height() == content_height
    assert widget.table_preview.height() < 360


def test_measurement_area_distribution_honors_explicit_nontrailing_spatial_axes(
    qtbot,
):
    # TYX and YXT carry the same two objects at two independent time points.
    # The inspector must follow the node's explicit Y/X axes rather than
    # treating the trailing X/T dimensions as a spatial plane.
    labels_tyx = np.stack((_labels(), _labels()), axis=0)
    labels_yxt = np.moveaxis(labels_tyx, 0, 2)
    widget = _widget(qtbot, np.zeros_like(labels_yxt), axes="YXT")
    labels_node = widget.add_node_from_palette("label_connected_components")
    measurements = widget.add_node_from_palette("measure_objects")
    _publish_array_output(widget, labels_node.id, labels_yxt, axes="YXT")
    widget._connect_nodes(labels_node.id, measurements.id)

    _select(widget, measurements.id)

    size_plot = widget.measurement_object_size_histogram_plot
    assert size_plot._title == "Object area"
    assert int(np.asarray(size_plot._counts).sum()) == 4
    assert size_plot._x_range == (0.0, 12.0)


def test_large_measurement_inputs_use_existing_background_diagnostic_paths(
    qtbot,
    monkeypatch,
):
    intensity = np.arange(80, dtype=np.uint16).reshape(8, 10)
    widget = _widget(qtbot, intensity, axes="YX")
    labels_node = widget.add_node_from_palette("label_connected_components")
    measurements = widget.add_node_from_palette("measure_objects_intensity")
    labels = _labels()
    _publish_array_output(widget, "input", intensity, axes="YX")
    _publish_array_output(widget, labels_node.id, labels, axes="YX")
    widget._connect_nodes(labels_node.id, measurements.id, target_port=0)
    widget._connect_nodes("input", measurements.id, target_port=1)

    queued_sizes = []
    queued_intensity = []
    monkeypatch.setattr(
        widget_module,
        "_should_auto_background_data",
        lambda _data: True,
    )
    monkeypatch.setattr(
        widget,
        "_queue_label_volume_request",
        lambda **kwargs: queued_sizes.append(kwargs),
    )
    monkeypatch.setattr(
        widget,
        "_queue_input_histogram",
        lambda **kwargs: queued_intensity.append(kwargs),
    )

    _select(widget, measurements.id)

    assert len(queued_sizes) == 1
    assert queued_sizes[0]["node_id"] == measurements.id
    assert queued_sizes[0]["spatial_ndim"] == 2
    assert queued_sizes[0]["connectivity"] == "Label IDs"
    assert len(queued_intensity) == 1
    assert queued_intensity[0]["node_id"] == measurements.id
    assert queued_intensity[0]["data"] is intensity
    assert queued_intensity[0]["scope"] == "Stack histogram"
    assert "full labels input" in (
        widget.measurement_object_size_histogram_status.text()
    )
    assert "full-input intensity" in (
        widget.measurement_intensity_histogram_status.text()
    )
