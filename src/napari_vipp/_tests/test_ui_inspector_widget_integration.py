from __future__ import annotations

import numpy as np
from qtpy.QtCore import QPoint, Qt
from qtpy.QtWidgets import QComboBox

from napari_vipp._tests.test_widget import _Viewer
from napari_vipp._widget import VippWidget
from napari_vipp.core.metadata import image_state_from_array
from napari_vipp.core.pipeline import (
    EXECUTION_NOT_CALCULATED,
    EXECUTION_READY,
    EXECUTION_RUNNING,
    EXECUTION_STALE,
)
from napari_vipp.core.tables import TableData, TableState
from napari_vipp.ui.controls import (
    ImageSourceControl,
    ImageSourceResolutionPresentation,
)


def _widget(qtbot, data=None, *, axes="CYX") -> VippWidget:
    if data is None:
        data = np.arange(3 * 10 * 12, dtype=np.uint16).reshape(3, 10, 12)
    widget = VippWidget(
        _Viewer(data, metadata={"axes": axes}),
        defer_initial_run=True,
    )
    qtbot.addWidget(widget)
    widget.run_pipeline = lambda *args, **kwargs: None
    return widget


def _publish_array_output(widget, node_id, data, *, axes=None) -> None:
    values = np.asarray(data)
    metadata = {"axes": axes} if axes else None
    state = image_state_from_array(values, layer_metadata=metadata)
    widget.pipeline.outputs[node_id] = values
    widget.pipeline.output_states[node_id] = state
    widget.pipeline.node_outputs[node_id] = [values]
    widget.pipeline.node_output_states[node_id] = [state]
    widget.pipeline.completed_node_ids.add(node_id)
    widget.pipeline.node_execution_states[node_id] = EXECUTION_READY


def _publish_table_output(
    widget,
    node_id,
    table: TableData | None = None,
) -> None:
    table = table or TableData(
        ("label", "mean_intensity"),
        ((1, 12.5), (2, 23.0)),
        name="Object intensities",
    )
    state = TableState(
        table.row_count,
        table.column_count,
        table.columns,
        source_name=table.name,
        table_kind=table.table_kind,
    )
    widget.pipeline.outputs[node_id] = table
    widget.pipeline.output_states[node_id] = state
    widget.pipeline.node_outputs[node_id] = [table]
    widget.pipeline.node_output_states[node_id] = [state]
    widget.pipeline.completed_node_ids.add(node_id)
    widget.pipeline.node_execution_states[node_id] = EXECUTION_READY


def _select(widget, node_id: str) -> None:
    widget._select_node(node_id)


def _assert_visible_order(widget, *sections) -> None:
    indexes = []
    for section in sections:
        assert not section.isHidden()
        index = widget._inspector_layout.indexOf(section)
        assert index >= 0
        indexes.append(index)
    assert indexes == sorted(indexes)
    assert len(indexes) == len(set(indexes))


def _connection_snapshot(widget) -> tuple[tuple[str, int, str, int], ...]:
    return tuple(
        (
            connection.source_id,
            int(connection.source_port),
            connection.target_id,
            int(connection.target_port),
        )
        for connection in widget.pipeline.connections
    )


def test_representative_nodes_render_semantic_sections_in_order(qtbot):
    widget = _widget(qtbot)
    data = np.arange(3 * 10 * 12, dtype=np.uint16).reshape(3, 10, 12)

    crop = widget.add_node_from_palette("crop_stack")
    threshold = widget.add_node_from_palette("binary_threshold")
    labels = widget.add_node_from_palette("label_connected_components")
    label_filter = widget.add_node_from_palette("filter_labels_by_volume")
    measurements = widget.add_node_from_palette("measure_objects_intensity")
    coloc = widget.add_node_from_palette("colocalized_voxels")
    writer = widget.add_node_from_palette("save_output")
    split = widget.add_node_from_palette("split_channels")

    _publish_array_output(widget, "input", data, axes="CYX")
    _publish_array_output(widget, crop.id, data[:, 1:9, 1:11], axes="CYX")
    _publish_array_output(widget, threshold.id, data > 100, axes="CYX")
    label_values = (data > 100).astype(np.int32)
    _publish_array_output(widget, labels.id, label_values, axes="CYX")
    _publish_array_output(widget, label_filter.id, label_values, axes="CYX")
    _publish_table_output(widget, measurements.id)
    _publish_array_output(widget, coloc.id, data, axes="CYX")

    split_outputs = [np.asarray(data[index]) for index in range(data.shape[0])]
    split_states = [
        image_state_from_array(values, layer_metadata={"axes": "YX"})
        for values in split_outputs
    ]
    widget.pipeline.outputs[split.id] = split_outputs[0]
    widget.pipeline.output_states[split.id] = split_states[0]
    widget.pipeline.node_outputs[split.id] = split_outputs
    widget.pipeline.node_output_states[split.id] = split_states
    widget.pipeline.completed_node_ids.add(split.id)
    widget.pipeline.node_execution_states[split.id] = EXECUTION_READY

    _select(widget, crop.id)
    _assert_visible_order(
        widget,
        widget.parameter_group,
        widget.histograms_section,
    )
    assert widget.label_volume_group.isHidden()
    assert widget.colocalization_scatter_group.isHidden()

    _select(widget, threshold.id)
    _assert_visible_order(
        widget,
        widget.parameter_group,
        widget.histograms_section,
        widget.mask_summary_section,
    )
    assert widget.histograms_section.title() == "Histograms"
    assert (
        widget.histograms_section.summary_label.text()
        == "Intensity → binary mask"
    )
    assert widget.parameter_group.summary_label.text() == "2 values"
    assert widget.mask_summary_section.title() == "Mask summary"
    assert not widget.rescale_input_histogram_group.isHidden()
    assert not widget.histogram_group.isHidden()

    _select(widget, label_filter.id)
    _assert_visible_order(
        widget,
        widget.parameter_group,
        widget.label_volume_group,
    )
    assert widget.label_volume_group.title() == "Input Object Volume Distribution"
    assert widget.histograms_section.isHidden()

    _select(widget, measurements.id)
    _assert_visible_order(
        widget,
        widget.parameter_group,
        widget.histograms_section,
        widget.table_group,
    )
    assert widget.parameter_group.title() == "Measurements"
    assert widget.histograms_section.title() == "Input distributions"
    assert widget.history_group.isHidden()
    assert widget.history_group.summary_label.text() == ""

    _select(widget, coloc.id)
    _assert_visible_order(
        widget,
        widget.parameter_group,
        widget.colocalization_scatter_group,
        widget.histograms_section,
    )
    assert widget.parameter_group.title() == "Colocalization"
    assert widget.histograms_section.summary_label.text() == (
        "Channel 1 + channel 2"
    )

    _select(widget, "input")
    _assert_visible_order(
        widget,
        widget.parameter_group,
        widget.histograms_section,
    )
    assert widget.source_representation_section.isHidden()
    assert widget.parameter_group.title() == "Source & data representations"
    assert widget.histograms_section.summary_label.text() == (
        "Source data · Level 0"
    )
    assert widget.histogram_plot._title == "Source data"

    _select(widget, writer.id)
    _assert_visible_order(
        widget,
        widget.parameter_group,
        widget.writer_status_section,
    )
    assert widget.parameter_group.title() == "Output settings"
    assert widget.histograms_section.isHidden()

    _select(widget, split.id)
    _assert_visible_order(
        widget,
        widget.output_selector_section,
        widget.histograms_section,
    )
    assert widget.parameter_group.isHidden()
    assert widget.output_selector_combo.count() == 3


def test_connected_inputs_are_read_only_and_threshold_guidance_is_visible(qtbot):
    widget = _widget(qtbot)
    labels = widget.add_node_from_palette("label_connected_components")
    measurements = widget.add_node_from_palette("measure_objects_intensity")
    threshold = widget.add_node_from_palette("binary_threshold")

    widget._connect_nodes("input", labels.id)
    widget._connect_nodes(labels.id, measurements.id, target_port=0)
    widget._connect_nodes("input", measurements.id, target_port=1)
    _publish_array_output(
        widget,
        labels.id,
        np.zeros((2, 8, 10), dtype=np.uint32),
        axes="ZYX",
    )
    _publish_array_output(
        widget,
        "input",
        np.zeros((2, 8, 10), dtype=np.uint16),
        axes="ZYX",
    )

    _select(widget, measurements.id)

    assert not widget.connected_inputs_panel.isHidden()
    assert widget.connected_inputs_panel.findChildren(QComboBox) == []
    assert widget.connected_inputs_form.rowCount() == 2
    assert widget.connected_inputs_panel.title_label.text() == "Connected inputs"
    rows = widget.connected_inputs_panel.rows
    assert {row.role_label.text() for row in rows} == {
        "Labels",
        "Intensity image",
    }
    assert all(
        row.source_label.textInteractionFlags() & Qt.TextSelectableByMouse
        for row in rows
    )
    assert all(row.role_label.wordWrap() for row in rows)
    assert all(row.source_label.wordWrap() for row in rows)
    assert all(not row.icon_label.pixmap().isNull() for row in rows)
    assert any(labels.title in row.source_label.text() for row in rows)
    assert any("Image Source" in row.source_label.text() for row in rows)
    assert any("out" in row.source_label.text() for row in rows)
    rows_by_role = {row.role_label.text(): row for row in rows}
    assert "ZYX: 2 × 8 × 10" in (
        rows_by_role["Labels"].scientific_label.text()
    )
    assert "uint32 (32-bit integer)" in (
        rows_by_role["Labels"].scientific_label.text()
    )
    assert "uint16 (16-bit integer)" in (
        rows_by_role["Intensity image"].scientific_label.text()
    )

    widget._connect_nodes("input", threshold.id)
    _select(widget, threshold.id)

    assert not widget.histogram_interaction_hint.isHidden()
    assert widget.histogram_interaction_hint.text() == (
        "Drag the orange threshold line to tune the threshold."
    )
    assert widget.histogram_interaction_hint.accessibleName() == (
        "Interactive histogram guidance"
    )


def _metadata_values(widget) -> dict[str, str]:
    return {
        widget.metadata_table.item(row, 0).text(): widget.metadata_table.item(
            row,
            1,
        ).text()
        for row in range(widget.metadata_table.rowCount())
    }


def test_uncalculated_measurement_has_projected_metadata_and_no_empty_sections(
    qtbot,
):
    widget = _widget(qtbot, np.zeros((8, 10), dtype=np.uint16), axes="YX")
    labels = widget.add_node_from_palette("label_connected_components")
    measurements = widget.add_node_from_palette("measure_objects_intensity")
    label_data = np.zeros((8, 10), dtype=np.int32)
    label_data[1:3, 1:4] = 1
    label_data[4:7, 5:9] = 2
    _publish_array_output(widget, labels.id, label_data, axes="YX")
    widget._connect_nodes(labels.id, measurements.id, target_port=0)
    widget._connect_nodes("input", measurements.id, target_port=1)

    _select(widget, measurements.id)

    metadata = _metadata_values(widget)
    assert metadata["Status"] == "Not calculated"
    assert metadata["Kind"] == "Object measurement table"
    assert metadata["Expected rows"] == "2"
    assert int(metadata["Expected fields"]) >= 10
    assert "label_id" in metadata["Expected field names"]
    assert "intensity_mean" in metadata["Expected field names"]
    assert metadata["NaN values"] == "Available after calculation"
    assert metadata["Infinite values"] == "Available after calculation"
    assert widget.history_group.isHidden()
    assert widget.behavior_section.isHidden()
    assert widget.isolated_tuning_checkbox.isHidden()
    assert widget.node_bypass_checkbox.isHidden()
    assert widget.keep_cached_checkbox.isHidden()
    assert not widget.table_group.isHidden()
    assert widget.table_summary.text() == "No result yet."
    assert widget.table_preview.rowCount() == 0
    assert widget.table_preview.height() == 0
    assert not widget.table_popout_button.isEnabled()
    assert not widget.table_calculate_button.isHidden()
    assert widget.table_calculate_button.text() == "Calculate"
    assert widget.table_calculate_button.isEnabled()
    assert widget.table_calculate_button.property("attentionRequired") is True
    assert widget.table_calculate_button.styleSheet()


def test_stale_measurement_keeps_rows_and_offers_local_recalculation(qtbot):
    widget = _widget(qtbot)
    measurements = widget.add_node_from_palette("measure_objects_intensity")
    _publish_table_output(widget, measurements.id)
    _select(widget, measurements.id)

    assert widget.table_calculate_button.isHidden()
    assert widget.table_preview.rowCount() == 2
    widget.table_popout_button.click()
    dialog = widget._result_table_dialog
    assert dialog is not None and dialog.isVisible()

    calls = []
    crop_draft_commits = []
    widget.run_pipeline = lambda *args, **kwargs: calls.append((args, kwargs))
    widget._commit_crop_draft = lambda *, schedule_run: (
        crop_draft_commits.append(schedule_run)
    )
    widget.pipeline.node_execution_states[measurements.id] = EXECUTION_STALE
    widget._sync_execution_ui()

    assert widget.table_preview.rowCount() == 2
    assert widget.table_popout_button.isEnabled()
    assert dialog.isVisible()
    assert dialog.model.rowCount() == 2
    assert widget.table_summary.text().startswith("Stale cached result · 2 rows")
    assert "not current" in widget.table_popout_button.toolTip()
    assert not widget.table_calculate_button.isHidden()
    assert widget.table_calculate_button.text() == "Recalculate"
    assert widget.table_calculate_button.isEnabled()
    assert widget.table_calculate_button.property("attentionRequired") is True
    assert widget.table_calculate_button.styleSheet()
    assert not dialog.result_status_panel.isHidden()
    assert "stale cached result" in dialog.result_status_label.text().lower()
    assert dialog.result_action_button.text() == "Recalculate"
    assert dialog.result_action_button.isEnabled()
    assert dialog.result_action_button.property("attentionRequired") is True
    assert dialog.result_action_button.styleSheet()
    assert "not current" in dialog.export_button.toolTip()

    dialog.result_action_button.click()

    assert crop_draft_commits == [False]
    assert calls == [((), {"manual_node_ids": {measurements.id}})]
    assert (
        widget.pipeline.node_execution_states[measurements.id]
        == EXECUTION_RUNNING
    )
    assert widget.table_preview.rowCount() == 2
    assert widget.table_calculate_button.text() == "Calculating…"
    assert not widget.table_calculate_button.isEnabled()
    assert (
        widget.table_calculate_button.property("attentionRequired") is False
    )
    assert widget.table_calculate_button.styleSheet() == ""
    assert "previous cached result" in dialog.result_status_label.text().lower()
    assert dialog.result_action_button.text() == "Calculating…"
    assert not dialog.result_action_button.isEnabled()


def test_uncalculated_table_result_action_uses_selected_node_calculation(qtbot):
    widget = _widget(qtbot)
    measurements = widget.add_node_from_palette("measure_objects")
    _select(widget, measurements.id)
    calls = []
    widget.run_pipeline = lambda *args, **kwargs: calls.append((args, kwargs))

    assert (
        widget.pipeline.node_execution_states[measurements.id]
        == EXECUTION_NOT_CALCULATED
    )
    widget.table_calculate_button.click()

    assert calls == [((), {"manual_node_ids": {measurements.id}})]
    assert (
        widget.pipeline.node_execution_states[measurements.id]
        == EXECUTION_RUNNING
    )


def test_calculated_measurement_shows_quality_metadata_and_real_history(qtbot):
    widget = _widget(qtbot, np.zeros((8, 10), dtype=np.uint16), axes="YX")
    measurements = widget.add_node_from_palette("measure_objects_intensity")
    table = TableData(
        ("label_id", "mean", "ratio", "note"),
        (
            (1, 3.5, float("nan"), "ok"),
            (2, float("inf"), 0.5, None),
        ),
        name="Object measurements",
        table_kind="Basic morphology + intensity",
    )
    state = TableState(
        table.row_count,
        table.column_count,
        table.columns,
        source_name=table.name,
        table_kind=table.table_kind,
        history=(
            "Image Source: imported YX data",
            "Measure Objects + Intensity: measured 2 objects",
        ),
    )
    widget.pipeline.outputs[measurements.id] = table
    widget.pipeline.output_states[measurements.id] = state
    widget.pipeline.node_outputs[measurements.id] = [table]
    widget.pipeline.node_output_states[measurements.id] = [state]
    widget.pipeline.completed_node_ids.add(measurements.id)
    widget.pipeline.node_execution_states[measurements.id] = EXECUTION_READY

    _select(widget, measurements.id)

    metadata = _metadata_values(widget)
    assert metadata["Rows"] == "2"
    assert metadata["Fields"] == "4"
    assert metadata["NaN values"] == "1"
    assert metadata["Infinite values"] == "1"
    assert metadata["Missing values"] == "1"
    assert metadata["Rows with NaN/Inf"] == "2"
    assert metadata["Fields with NaN/Inf"] == "2 (mean, ratio)"
    assert not widget.history_group.isHidden()
    assert widget.history_group.summary_label.text() == "2 steps"
    assert "measured 2 objects" in widget.history_label.text()
    assert widget.behavior_section.isHidden()


def test_long_table_metadata_rows_use_current_width_without_vertical_padding(qtbot):
    widget = _widget(qtbot)
    summary = widget.add_node_from_palette("summarize_skeleton_branches")
    columns = tuple(
        f"branch_length_physical_{statistic}"
        for statistic in (
            "total",
            "mean",
            "median",
            "std",
            "min",
            "max",
            "q25",
            "q75",
        )
    )
    table = TableData(
        columns,
        (tuple(float(index) for index in range(len(columns))),),
        name="Skeleton branch summary",
        table_kind="Skeleton branch summary",
    )
    _publish_table_output(widget, summary.id, table)

    _select(widget, summary.id)
    widget.metadata_group.setExpanded(False)
    widget.metadata_table.resize(900, 320)
    field_names_row = next(
        row
        for row in range(widget.metadata_table.rowCount())
        if widget.metadata_table.item(row, 0).text() == "Field names"
    )
    widget.metadata_table.setRowHeight(field_names_row, 300)

    widget.metadata_group.setExpanded(True)
    qtbot.waitUntil(
        lambda: widget.metadata_table.rowHeight(field_names_row) < 300,
        timeout=1_000,
    )

    expected_alignment = int(Qt.AlignLeft | Qt.AlignTop)
    assert (
        widget.metadata_table.item(field_names_row, 0).textAlignment()
        == expected_alignment
    )
    assert (
        widget.metadata_table.item(field_names_row, 1).textAlignment()
        == expected_alignment
    )


def test_source_writer_and_result_actions_are_contextual(qtbot):
    widget = _widget(qtbot)
    data = np.arange(3 * 10 * 12, dtype=np.uint16).reshape(3, 10, 12)
    threshold = widget.add_node_from_palette("binary_threshold")
    measurements = widget.add_node_from_palette("measure_objects_intensity")
    writer = widget.add_node_from_palette("save_output")
    _publish_array_output(widget, "input", data, axes="CYX")
    _publish_array_output(widget, threshold.id, data > 100, axes="CYX")
    _publish_table_output(widget, measurements.id)

    _select(widget, "input")
    source_control = widget._parameter_widgets["image_source"]
    assert isinstance(source_control, ImageSourceControl)
    assert widget.source_representation_label.isHidden()
    assert widget.source_representation_label.text() == ""
    assert "level 0" in source_control.resolution_panel.toolTip().lower()
    assert "presentation only" in source_control.resolution_panel.toolTip().lower()
    assert "processing and export" in source_control.viewer_display_combo.toolTip()
    assert not widget.pin_button.isHidden()
    assert widget.pin_button.text() == "Pin source"
    assert not widget.save_button.isHidden()
    assert widget.save_button.text() == "Save source…"
    assert widget.save_button.isEnabled()

    _select(widget, threshold.id)
    assert not widget.pin_button.isHidden()
    assert widget.pin_button.text() == "Pin node"
    assert widget.save_button.text() == "Save mask…"
    assert widget.save_button.isEnabled()

    _select(widget, measurements.id)
    assert widget.pin_button.isHidden()
    assert not widget.save_button.isHidden()
    assert widget.save_button.text() == "Export table…"
    assert widget.save_button.isEnabled()

    _select(widget, writer.id)
    assert widget.pin_button.isHidden()
    assert widget.save_button.isHidden()
    assert not widget.writer_status_section.isHidden()


def test_complete_table_window_uses_exact_selected_multi_output_port(qtbot):
    widget = _widget(qtbot)
    graph_tables = widget.add_node_from_palette("skeleton_graph_tables")
    nodes = TableData(
        ("node_id", "degree"),
        tuple((index, index % 4) for index in range(250)),
        name="Graph nodes",
        table_kind="skeleton graph nodes",
    )
    edges = TableData(
        ("edge_id", "length"),
        ((10, 4.5), (2, 9.0), (7, 2.0)),
        name="Graph edges",
        table_kind="skeleton graph edges",
    )
    states = [
        TableState(
            table.row_count,
            table.column_count,
            table.columns,
            source_name=table.name,
            table_kind=table.table_kind,
        )
        for table in (nodes, edges)
    ]
    widget.pipeline.outputs[graph_tables.id] = nodes
    widget.pipeline.output_states[graph_tables.id] = states[0]
    widget.pipeline.node_outputs[graph_tables.id] = [nodes, edges]
    widget.pipeline.node_output_states[graph_tables.id] = states
    widget.pipeline.completed_node_ids.add(graph_tables.id)
    widget.pipeline.node_execution_states[graph_tables.id] = EXECUTION_READY

    _select(widget, graph_tables.id)

    assert widget.table_preview.rowCount() == 200
    assert widget.table_popout_button.isEnabled()
    widget.table_popout_button.click()
    dialog = widget._result_table_dialog
    assert dialog is not None
    assert dialog.isVisible()
    assert not dialog.isModal()
    assert dialog.table is nodes
    assert dialog.model.rowCount() == 250
    assert dialog.context_key == (graph_tables.id, 0)
    assert "Graph nodes" in dialog.windowTitle()

    widget.output_selector_combo.setCurrentIndex(1)
    widget.table_popout_button.click()

    assert widget._result_table_dialog is dialog
    assert dialog.table is edges
    assert dialog.model.rowCount() == 3
    assert dialog.context_key == (graph_tables.id, 1)
    assert "Graph edges" in dialog.windowTitle()


def test_generic_table_output_opens_and_live_window_refreshes_off_selection(
    qtbot,
):
    widget = _widget(qtbot)
    batch_output = widget.add_node_from_palette("batch_output")
    original = TableData(
        ("sample", "score"),
        (("sample2", 2.0), ("sample1", 1.0)),
        name="Batch table",
        table_kind="batch results",
    )
    _publish_table_output(widget, batch_output.id, original)

    _select(widget, batch_output.id)

    assert widget._node_output_type(batch_output.id) == "table"
    _assert_visible_order(
        widget,
        widget.table_group,
        widget.writer_status_section,
    )
    assert widget.table_popout_button.isEnabled()
    widget.table_popout_button.click()
    dialog = widget._result_table_dialog
    assert dialog is not None
    assert dialog.table is original
    assert dialog.context_key == (batch_output.id, 0)

    _select(widget, "input")
    refreshed = TableData(
        ("sample", "score"),
        (("sample3", 3.0),),
        name="Updated batch table",
        table_kind="batch results",
    )
    _publish_table_output(widget, batch_output.id, refreshed)
    widget._refresh_node_presentation_surfaces({batch_output.id})

    assert dialog.isVisible()
    assert dialog.table is refreshed
    assert dialog.model.rowCount() == 1
    assert dialog.context_key == (batch_output.id, 0)

    widget._delete_node(batch_output.id)

    assert not dialog.isVisible()


def test_colocalization_threshold_scrub_preserves_inspector_allocation(qtbot):
    data = np.zeros((80, 100), dtype=np.uint8)
    data[:, 50:] = 200
    widget = _widget(qtbot, data, axes="YX")
    coloc = widget.add_node_from_palette("racc_index")
    widget.pipeline.set_param(coloc.id, "threshold_mode", "Manual")
    widget.pipeline.set_param(coloc.id, "channel_1_threshold", 80.0)
    widget.pipeline.set_param(coloc.id, "channel_2_threshold", 90.0)
    widget.resize(1_200, 640)
    widget.show()
    _select(widget, coloc.id)
    plot = widget.colocalization_scatter_plot
    density = np.ones((32, 32), dtype=np.float64)
    plot.set_density(
        density,
        threshold_1=80.0,
        threshold_2=90.0,
        intensity_min=0.0,
        intensity_max=255.0,
        summary="Exact: 2,000/8,000 (25.0%)",
    )
    widget.colocalization_scatter_group.show()
    widget._sync_inspector_presentation()
    panel = widget.inspector_panel
    panel.ensureWidgetVisible(plot)
    qtbot.wait(20)
    locked_widgets = (
        widget.inspector_content,
        widget.colocalization_scatter_group,
        widget.colocalization_scatter_summary,
        plot,
    )
    original_constraints = {
        id(candidate): (candidate.minimumHeight(), candidate.maximumHeight())
        for candidate in locked_widgets
    }
    widget._update_colocalization_scatter = lambda: None

    def allocation_snapshot():
        viewport_position = plot.mapTo(panel.viewport(), QPoint(0, 0))
        group = widget.colocalization_scatter_group
        return (
            plot.x(),
            plot.y(),
            plot.width(),
            plot.height(),
            viewport_position.x(),
            viewport_position.y(),
            group.x(),
            group.y(),
            group.width(),
            group.height(),
            widget.colocalization_scatter_summary.height(),
            panel.verticalScrollBar().value(),
        )

    before = allocation_snapshot()
    plot.gestureStarted.emit()

    try:
        assert widget._colocalization_inspector_height_lock
        assert not widget._inspector_layout.isEnabled()
        for threshold, summary in (
            (
                120.0,
                "Manual thresholds. Calculating the exact full-ROI counts and "
                "visible scatter density in the background while every scientific "
                "result remains live.",
            ),
            (160.0, "Exact colocalized count: 1,000/8,000 (12.5%)."),
            ):
            widget._on_colocalization_scatter_threshold_changed(1, threshold)
            plot.set_pending_thresholds(
                threshold_1=threshold,
                threshold_2=90.0,
                preserve_density=True,
                summary="Calculating exact count...",
            )
            widget.colocalization_scatter_summary.setText(summary)
            widget._sync_inspector_presentation()
            qtbot.wait(10)

            assert plot._threshold_1 == threshold
            assert widget.pipeline.nodes[coloc.id].params[
                "channel_1_threshold"
            ] == threshold
            assert widget._colocalization_inspector_sync_deferred
            after = allocation_snapshot()
            assert after == before
    finally:
        plot.gestureFinished.emit()
        widget._permit_incomplete_startup_discard()
    qtbot.wait(10)

    assert widget._colocalization_inspector_height_lock == ()
    assert not widget._colocalization_inspector_sync_deferred
    assert widget._inspector_layout.isEnabled()
    assert panel.verticalScrollBar().value() == before[-1]
    for candidate in locked_widgets:
        assert (
            candidate.minimumHeight(),
            candidate.maximumHeight(),
        ) == original_constraints[id(candidate)]


def test_source_representation_panel_moves_with_selection_without_stale_ownership(
    qtbot,
):
    widget = _widget(qtbot)
    crop = widget.add_node_from_palette("crop_stack")

    _select(widget, "input")
    first_control = widget._parameter_widgets["image_source"]
    assert isinstance(first_control, ImageSourceControl)
    first_panel = first_control.resolution_panel
    host = widget.source_representation_section.content_widget
    assert first_control.source_representation_host is host
    assert first_panel.parentWidget() is host
    assert widget.source_representation_layout.indexOf(first_panel) >= 0

    _select(widget, crop.id)

    assert first_control.source_representation_host is None
    assert first_panel.parentWidget() is first_control._source_representation_home
    assert widget.source_representation_layout.indexOf(first_panel) == -1
    assert widget.source_representation_section.isHidden()

    _select(widget, "input")
    second_control = widget._parameter_widgets["image_source"]
    assert isinstance(second_control, ImageSourceControl)
    assert second_control is not first_control
    assert second_control.source_representation_host is host
    assert second_control.resolution_panel.parentWidget() is host
    assert (
        widget.source_representation_layout.indexOf(second_control.resolution_panel)
        >= 0
    )
    assert widget.source_representation_layout.indexOf(first_panel) == -1


def test_empty_source_representation_section_tracks_live_resolution_content(
    qtbot,
    monkeypatch,
):
    widget = _widget(qtbot)
    _select(widget, "input")
    control = widget._parameter_widgets["image_source"]
    assert isinstance(control, ImageSourceControl)

    assert control.resolution_panel.isHidden()
    assert widget.source_representation_section.isHidden()
    assert widget.source_representation_section.summary_label.text() == ""

    control.mode_combo.blockSignals(True)
    control.mode_combo.setCurrentText("file path")
    control.mode_combo.blockSignals(False)
    presentation = ImageSourceResolutionPresentation(
        analysis_axes="ZYX",
        analysis_shape=(12, 128, 160),
        level_shapes=((12, 128, 160), (6, 64, 80)),
    )
    monkeypatch.setattr(
        widget,
        "_source_resolution_presentation",
        lambda _node: presentation,
    )

    widget._refresh_source_resolution_control("input")

    assert not control.resolution_panel.isHidden()
    assert not widget.source_representation_section.isHidden()
    assert (
        widget.source_representation_section.summary_label.text()
        == "Processing uses level 0"
    )

    monkeypatch.setattr(
        widget,
        "_source_resolution_presentation",
        lambda _node: ImageSourceResolutionPresentation(),
    )
    widget._refresh_source_resolution_control("input")

    assert control.resolution_panel.isHidden()
    assert widget.source_representation_section.isHidden()
    assert widget.source_representation_section.summary_label.text() == ""


def test_multi_output_selector_changes_display_only_not_graph_connections(qtbot):
    data = np.arange(3 * 8 * 9, dtype=np.uint16).reshape(3, 8, 9)
    widget = _widget(qtbot, data)
    split = widget.add_node_from_palette("split_channels")
    widget._connect_nodes("input", split.id)

    outputs = [np.asarray(data[index]) for index in range(data.shape[0])]
    states = [
        image_state_from_array(values, layer_metadata={"axes": "YX"})
        for values in outputs
    ]
    widget.pipeline.outputs[split.id] = outputs[0]
    widget.pipeline.output_states[split.id] = states[0]
    widget.pipeline.node_outputs[split.id] = outputs
    widget.pipeline.node_output_states[split.id] = states
    widget.pipeline.completed_node_ids.add(split.id)
    widget.pipeline.node_execution_states[split.id] = EXECUTION_READY
    _select(widget, split.id)

    assert not widget.output_selector_section.isHidden()
    assert widget.output_selector_combo.isEnabled()
    assert widget.output_selector_combo.count() == 3
    assert "Workflow connections keep their authored output ports" in (
        widget.output_selector_note.text()
    )
    connections_before = _connection_snapshot(widget)
    first_data, _first_state, first_port = widget._node_display_payload(split.id)
    np.testing.assert_array_equal(first_data, outputs[0])
    assert first_port == 0

    widget.output_selector_combo.setCurrentIndex(1)

    second_data, _second_state, second_port = widget._node_display_payload(split.id)
    np.testing.assert_array_equal(second_data, outputs[1])
    assert second_port == 1
    assert widget.pipeline.nodes[split.id].params["preview_channel"] == 1
    assert _connection_snapshot(widget) == connections_before
    assert not widget.save_button.isHidden()
    assert widget.save_button.text() == "Save image…"
