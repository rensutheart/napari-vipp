from __future__ import annotations

import numpy as np
import pytest
from qtpy.QtCore import QPoint, Qt

from napari_vipp._tests.test_ui_inspector_widget_integration import (
    _connection_snapshot,
    _publish_array_output,
    _publish_table_output,
    _select,
    _widget,
)
from napari_vipp._tests.test_widget import _QueuedThreadPool
from napari_vipp._widget import INSPECTOR_LABEL_LOADING_CONTENT_HEIGHT
from napari_vipp.core.diagnostics import object_sizes
from napari_vipp.core.metadata import image_state_from_array
from napari_vipp.core.pipeline import EXECUTION_ERROR, EXECUTION_READY
from napari_vipp.core.tables import TableData, TableState
from napari_vipp.ui.inspector import MASK_SUMMARY_SECTION


def _publish_multi_output(widget, node_id, outputs, *, axes="YX") -> None:
    values = [np.asarray(output) for output in outputs]
    states = [
        image_state_from_array(output, layer_metadata={"axes": axes})
        for output in values
    ]
    widget.pipeline.outputs[node_id] = values[0]
    widget.pipeline.output_states[node_id] = states[0]
    widget.pipeline.node_outputs[node_id] = values
    widget.pipeline.node_output_states[node_id] = states
    widget.pipeline.completed_node_ids.add(node_id)
    widget.pipeline.node_execution_states[node_id] = EXECUTION_READY


def test_node_switch_retires_previous_parameter_form_before_new_identity(qtbot):
    widget = _widget(qtbot)
    label_filter = widget.add_node_from_palette("filter_labels_by_volume")
    labels = widget.add_node_from_palette("label_connected_components")
    _select(widget, label_filter.id)

    retired_widgets = [
        widget.parameter_form.itemAt(index).widget()
        for index in range(widget.parameter_form.count())
        if widget.parameter_form.itemAt(index).widget() is not None
    ]
    assert retired_widgets
    assert all(not candidate.isHidden() for candidate in retired_widgets)
    original_sync_preview = widget._sync_preview_ui
    transition = {}

    def observe_first_new_node_sync():
        transition["title"] = widget.selected_title.text()
        transition["retired"] = tuple(
            (candidate.isHidden(), candidate.parent() is None)
            for candidate in retired_widgets
        )
        transition["old_sections_hidden"] = all(
            section.isHidden() for section in widget._inspector_sections.values()
        )
        transition["metadata_rows"] = widget.metadata_table.rowCount()
        original_sync_preview()

    widget._sync_preview_ui = observe_first_new_node_sync

    _select(widget, labels.id)

    assert transition["title"] == "Label Connected Components"
    assert all(hidden and detached for hidden, detached in transition["retired"])
    assert transition["old_sections_hidden"]
    assert transition["metadata_rows"] == 0
    assert "min_volume" not in widget._parameter_widgets
    assert "max_volume" not in widget._parameter_widgets
    assert {"spatial_mode", "connectivity"} <= set(widget._parameter_widgets)


def test_embedded_card_switch_defers_secondary_refresh_until_new_form_is_ready(
    qtbot,
    monkeypatch,
):
    widget = _widget(qtbot)
    label_filter = widget.add_node_from_palette("filter_labels_by_volume")
    labels = widget.add_node_from_palette("label_connected_components")
    _select(widget, label_filter.id)
    completed = []

    def record_refresh(node_id, *, select_layer):
        completed.append(
            (node_id, bool(select_layer), frozenset(widget._parameter_widgets))
        )

    monkeypatch.setattr(
        widget,
        "_refresh_selected_inspector_after_selection",
        record_refresh,
    )

    # NodeCard is an embedded QWidget and has a separate press route from the
    # surrounding QGraphicsProxyWidget. Exercise that exact user-facing path.
    widget.graph_view._cards[labels.id].selected.emit(labels.id)

    assert widget._selected_node_id == labels.id
    assert widget.selected_title.text() == "Label Connected Components"
    assert {"spatial_mode", "connectivity"} <= set(widget._parameter_widgets)
    assert "min_volume" not in widget._parameter_widgets
    assert not widget.parameter_group.isHidden()
    assert completed == []

    qtbot.waitUntil(lambda: bool(completed), timeout=1_000)

    assert completed == [
        (
            labels.id,
            True,
            frozenset({"spatial_mode", "connectivity"}),
        )
    ]


@pytest.mark.parametrize(
    ("operation_id", "source_id", "expected_controls"),
    (
        (
            "composite_to_rgb",
            "input",
            frozenset({"channel_axis_mode", "mapping_mode"}),
        ),
        (
            "label_connected_components",
            "threshold",
            frozenset({"spatial_mode", "connectivity"}),
        ),
    ),
)
def test_dragging_selected_node_preserves_inspector_and_scroll(
    qtbot,
    monkeypatch,
    operation_id,
    source_id,
    expected_controls,
):
    widget = _widget(qtbot)
    widget._permit_incomplete_startup_discard()
    node = widget.add_node_from_palette(operation_id)
    widget._connect_nodes(source_id, node.id)
    widget.resize(1_300, 640)
    widget.show()
    qtbot.waitExposed(widget)
    widget.graph_view.select_node(node.id)
    qtbot.wait(40)
    assert expected_controls <= set(widget._parameter_widgets)

    control_ids = {
        name: id(widget._parameter_widgets[name]) for name in expected_controls
    }
    connected_row_ids = tuple(id(row) for row in widget.connected_inputs_panel.rows)
    panel = widget.inspector_panel
    widget.inspector_content.setMinimumHeight(panel.viewport().height() + 500)
    qtbot.waitUntil(lambda: panel.verticalScrollBar().maximum() > 0)
    panel.verticalScrollBar().setValue(panel.verticalScrollBar().maximum())
    scroll_before = panel.verticalScrollBar().value()
    assert scroll_before > 0

    calls = {
        "_render_parameters": 0,
        "_update_histogram": 0,
        "_update_metadata_panel": 0,
        "_inspect_selected_node": 0,
        "_sync_inspector_presentation": 0,
    }
    for method_name in tuple(calls):
        original = getattr(widget, method_name)

        def tracked(*args, _name=method_name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(widget, method_name, tracked)

    proxy = widget.graph_view._proxies[node.id]
    position_before = proxy.pos()
    start = widget.graph_view.mapFromScene(proxy.sceneBoundingRect().center())
    end = start + QPoint(55, -35)
    qtbot.mousePress(widget.graph_view.viewport(), Qt.LeftButton, pos=start)
    qtbot.mouseMove(widget.graph_view.viewport(), pos=end)
    qtbot.wait(40)

    assert proxy.pos() != position_before
    assert calls == {name: 0 for name in calls}
    assert {
        name: id(widget._parameter_widgets[name]) for name in expected_controls
    } == control_ids
    assert tuple(id(row) for row in widget.connected_inputs_panel.rows) == (
        connected_row_ids
    )
    assert panel.verticalScrollBar().value() == scroll_before

    qtbot.mouseRelease(widget.graph_view.viewport(), Qt.LeftButton, pos=end)
    qtbot.wait(40)

    assert not widget.graph_view.node_drag_in_progress()
    assert calls == {name: 0 for name in calls}
    assert panel.verticalScrollBar().value() == scroll_before


def test_new_node_drag_defers_secondary_inspector_work_until_release(
    qtbot,
    monkeypatch,
):
    widget = _widget(qtbot)
    widget._permit_incomplete_startup_discard()
    node = widget.add_node_from_palette("label_connected_components")
    widget._connect_nodes("threshold", node.id)
    widget.resize(1_300, 640)
    widget.show()
    qtbot.waitExposed(widget)
    widget.graph_view.select_node("gaussian")
    qtbot.wait(40)
    completed = []
    monkeypatch.setattr(
        widget,
        "_refresh_selected_inspector_after_selection",
        lambda node_id, *, select_layer: completed.append(
            (node_id, bool(select_layer))
        ),
    )

    proxy = widget.graph_view._proxies[node.id]
    start = widget.graph_view.mapFromScene(proxy.sceneBoundingRect().center())
    end = start + QPoint(50, -30)
    qtbot.mousePress(widget.graph_view.viewport(), Qt.LeftButton, pos=start)
    qtbot.mouseMove(widget.graph_view.viewport(), pos=end)
    qtbot.wait(50)

    assert widget._selected_node_id == node.id
    assert {"spatial_mode", "connectivity"} <= set(widget._parameter_widgets)
    assert widget.graph_view.node_drag_in_progress()
    assert completed == []

    qtbot.mouseRelease(widget.graph_view.viewport(), Qt.LeftButton, pos=end)
    qtbot.waitUntil(lambda: bool(completed), timeout=1_000)

    assert completed == [(node.id, True)]


def test_composite_press_hold_defers_cached_inspector_until_drag_release(
    qtbot,
    monkeypatch,
):
    widget = _widget(
        qtbot,
        data=np.zeros((3, 4, 32, 32), dtype=np.uint16),
        axes="CZYX",
    )
    widget._permit_incomplete_startup_discard()
    composite = widget.add_node_from_palette("composite_to_rgb")
    widget._connect_nodes("input", composite.id)
    _publish_array_output(
        widget,
        composite.id,
        np.zeros((4, 32, 32, 3), dtype=np.uint16),
        axes="ZYXC",
    )
    widget.resize(1_300, 640)
    widget.show()
    qtbot.waitExposed(widget)
    widget.graph_view.select_node("gaussian")
    qtbot.wait(40)

    scheduled = []
    monkeypatch.setattr(
        "napari_vipp._widget.QTimer.singleShot",
        lambda delay, callback: scheduled.append((int(delay), callback)),
    )
    completed = []
    monkeypatch.setattr(
        widget,
        "_refresh_selected_inspector_after_selection",
        lambda node_id, *, select_layer: completed.append(
            (node_id, bool(select_layer))
        ),
    )

    proxy = widget.graph_view._proxies[composite.id]
    position_before = proxy.pos()
    start = widget.graph_view.mapFromScene(proxy.sceneBoundingRect().center())
    end = start + QPoint(55, -35)
    qtbot.mousePress(widget.graph_view.viewport(), Qt.LeftButton, pos=start)

    generation = widget._selected_inspector_refresh_generation
    assert widget.graph_view.node_pointer_gesture_active()
    assert not widget.graph_view.node_drag_in_progress()
    assert scheduled
    widget._finish_selected_inspector_refresh(
        generation,
        composite.id,
        select_layer=True,
    )
    assert completed == []

    qtbot.mouseMove(widget.graph_view.viewport(), pos=end)

    assert proxy.pos() != position_before
    assert widget.graph_view.node_drag_in_progress()
    assert completed == []

    qtbot.mouseRelease(widget.graph_view.viewport(), Qt.LeftButton, pos=end)

    assert not widget.graph_view.node_pointer_gesture_active()
    assert not widget.graph_view.node_drag_in_progress()
    widget._finish_selected_inspector_refresh(
        generation,
        composite.id,
        select_layer=True,
    )
    assert completed == [(composite.id, True)]


def test_rapid_embedded_card_switch_publishes_only_latest_inspector(qtbot, monkeypatch):
    widget = _widget(qtbot)
    label_filter = widget.add_node_from_palette("filter_labels_by_volume")
    labels = widget.add_node_from_palette("label_connected_components")
    completed = []
    monkeypatch.setattr(
        widget,
        "_refresh_selected_inspector_after_selection",
        lambda node_id, *, select_layer: completed.append(
            (node_id, bool(select_layer))
        ),
    )

    widget.graph_view._cards[label_filter.id].selected.emit(label_filter.id)
    widget.graph_view._cards[labels.id].selected.emit(labels.id)

    assert completed == []
    assert widget._selected_node_id == labels.id
    assert {"spatial_mode", "connectivity"} <= set(widget._parameter_widgets)

    qtbot.waitUntil(lambda: bool(completed), timeout=1_000)

    assert completed == [(labels.id, True)]


def test_repeated_embedded_label_switch_keeps_parameter_rows_and_height_stable(
    qtbot,
):
    widget = _widget(qtbot)
    label_filter = widget.add_node_from_palette("filter_labels_by_volume")
    labels = widget.add_node_from_palette("label_connected_components")
    widget.resize(1_400, 800)
    widget._permit_incomplete_startup_discard()
    widget.show()
    widget.splitter.setSizes((250, 700, 450))
    qtbot.wait(25)
    widget._sync_inspector_responsive_layout()
    assert widget.inspector_viewport.width() >= 350
    expected_geometry = {}

    for _cycle in range(3):
        for node in (label_filter, labels):
            widget.graph_view._cards[node.id].selected.emit(node.id)
            # This regression is about the first parameter frame. Keep the
            # slower diagnostics out of the comparison while allowing QLabel's
            # queued height-for-width pass to run.
            widget._cancel_selected_inspector_refresh()

            form_widgets = tuple(
                item.widget()
                for index in range(widget.parameter_form.count())
                if (item := widget.parameter_form.itemAt(index)).widget()
                is not None
            )
            assert form_widgets
            assert all(not control.isHidden() for control in form_widgets)
            immediate = (
                widget.parameter_form.rowCount(),
                widget.parameter_group.height(),
                widget.parameter_form_widget.height(),
            )
            expected_geometry.setdefault(node.operation_id, immediate)
            assert immediate == expected_geometry[node.operation_id]

            qtbot.wait(25)

            settled = (
                widget.parameter_form.rowCount(),
                widget.parameter_group.height(),
                widget.parameter_form_widget.height(),
            )
            assert settled == immediate

    assert expected_geometry["filter_labels_by_volume"][0] == 4
    assert expected_geometry["label_connected_components"][0] == 3


def test_embedded_label_switch_prioritizes_parameters_while_distribution_loads(
    qtbot,
    monkeypatch,
):
    labels_data = np.zeros((24, 28), dtype=np.int32)
    labels_data[1:7, 2:9] = 1
    labels_data[10:22, 14:26] = 9
    widget, _labels, filtered = _large_label_filter_inspector(qtbot, labels_data)
    pool = _QueuedThreadPool()
    widget._label_volume_thread_pool = pool
    widget._clear_label_volume_cache()
    calls = []

    def exact_volumes(values, spatial_ndim, connectivity):
        calls.append((values, int(spatial_ndim), connectivity))
        return np.asarray([42, 144], dtype=np.int64)

    monkeypatch.setattr(
        type(widget),
        "_object_sizes",
        staticmethod(exact_volumes),
    )
    widget._permit_incomplete_startup_discard()
    widget.show()
    qtbot.wait(25)

    widget.graph_view._cards[filtered.id].selected.emit(filtered.id)

    assert {"min_volume", "max_volume"} <= set(widget._parameter_widgets)
    assert all(
        not widget.parameter_form.itemAt(index).widget().isHidden()
        for index in range(widget.parameter_form.count())
        if widget.parameter_form.itemAt(index).widget() is not None
    )
    parameter_height = widget.parameter_form_widget.height()
    assert widget.label_volume_group.isBusy()
    assert not widget.label_volume_group.isHidden()
    assert widget.label_volume_group.content_widget.minimumHeight() == (
        INSPECTOR_LABEL_LOADING_CONTENT_HEIGHT
    )
    assert calls == []
    # A completion from the previously selected node may arrive during the
    # one-frame delay. It must not clear this selection's primed placeholder.
    widget._sync_inspector_diagnostic_busy_state()
    assert widget.label_volume_group.isBusy()

    qtbot.waitUntil(lambda: len(pool.workers) == 1, timeout=1_000)

    assert widget.label_volume_group.isBusy()
    assert widget.parameter_form_widget.height() == parameter_height
    assert calls == []

    pool.workers[0].run()

    assert len(calls) == 1
    assert not widget.label_volume_group.isBusy()
    assert widget.label_volume_group.content_widget.minimumHeight() == 0
    assert widget.parameter_form_widget.height() == parameter_height
    assert widget.label_volume_plot._counts.sum() == 2


def test_embedded_measurement_switch_primes_results_and_distributions(qtbot):
    data = np.arange(12 * 14, dtype=np.uint16).reshape(12, 14)
    labels_data = np.zeros_like(data, dtype=np.int32)
    labels_data[1:5, 2:7] = 1
    labels_data[7:11, 8:13] = 2
    widget = _widget(qtbot, data, axes="YX")
    labels = widget.add_node_from_palette("label_connected_components")
    measurement = widget.add_node_from_palette("measure_objects_intensity")
    widget._connect_nodes(labels.id, measurement.id, target_port=0)
    widget._connect_nodes("input", measurement.id, target_port=1)
    _publish_array_output(widget, labels.id, labels_data, axes="YX")
    _publish_table_output(widget, measurement.id)
    _select(widget, "input")
    widget.table_summary.setText("99 old rows")

    widget.graph_view._cards[measurement.id].selected.emit(measurement.id)

    assert widget.histograms_section.isBusy()
    assert not widget.histograms_section.isHidden()
    assert widget.histograms_section.content_widget.minimumHeight() > 0
    assert widget.table_group.isBusy()
    assert not widget.table_group.isHidden()
    assert widget.table_group.summary_label.text() == "Loading…"
    assert widget.table_summary.text() == "Loading results…"
    widget._cancel_selected_inspector_refresh()


def test_embedded_switch_does_not_claim_missing_diagnostics_are_loading(qtbot):
    widget = _widget(qtbot)
    measurement = widget.add_node_from_palette("measure_objects_intensity")
    colocalization = widget.add_node_from_palette("colocalization_metrics")

    widget.graph_view._cards[measurement.id].selected.emit(measurement.id)

    assert not widget.histograms_section.isBusy()
    assert not widget.table_group.isBusy()
    widget._cancel_selected_inspector_refresh()

    widget.graph_view._cards[colocalization.id].selected.emit(colocalization.id)

    assert not widget.colocalization_scatter_group.isBusy()
    widget._cancel_selected_inspector_refresh()


def test_label_filter_form_bounds_never_scan_labels_on_selection(qtbot, monkeypatch):
    labels_data = np.zeros((24, 28), dtype=np.int32)
    labels_data[1:7, 2:9] = 1
    labels_data[10:22, 14:26] = 9
    widget, _labels, filtered = _large_label_filter_inspector(qtbot, labels_data)
    widget._clear_label_volume_cache()

    def unexpected_scan(*_args, **_kwargs):
        raise AssertionError("parameter-form construction scanned every label")

    monkeypatch.setattr(
        type(widget),
        "_object_sizes",
        staticmethod(unexpected_scan),
    )

    widget.graph_view._cards[filtered.id].selected.emit(filtered.id)
    widget._cancel_selected_inspector_refresh()

    assert {"min_volume", "max_volume"} <= set(widget._parameter_widgets)


def test_selected_viewer_refresh_coalesces_transient_dims_events(qtbot, monkeypatch):
    widget = _widget(qtbot)
    calls = []

    def emit_transient_dims_events():
        for _index in range(6):
            widget._on_dims_changed()

    monkeypatch.setattr(widget, "_inspect_selected_node", emit_transient_dims_events)
    monkeypatch.setattr(
        widget,
        "_apply_selected_viewer_surface",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(widget, "_update_crop_roi_presentation", lambda *_args: None)
    monkeypatch.setattr(
        widget,
        "_restore_selected_output_for_interactive_cache",
        lambda *_args: None,
    )
    monkeypatch.setattr(widget, "_sync_view_dims_bar", lambda: calls.append("dims"))
    monkeypatch.setattr(
        widget,
        "_update_thumbnails",
        lambda: calls.append("thumbnails"),
    )
    monkeypatch.setattr(
        widget,
        "_update_metadata_panel",
        lambda: calls.append("metadata"),
    )
    monkeypatch.setattr(widget, "_update_histogram", lambda: calls.append("histogram"))

    widget._finish_selected_viewer_refresh(
        widget._selected_viewer_refresh_generation,
        widget._selected_node_id,
        select_layer=False,
    )

    assert calls == ["dims", "thumbnails", "metadata", "histogram"]
    assert widget._selected_viewer_dims_refresh_pending is False


def test_object_colocalization_results_keep_scatter_visible_after_table(qtbot):
    channel_1 = np.zeros((12, 12), dtype=np.uint16)
    channel_1[2:9, 2:9] = 80
    channel_2 = np.zeros_like(channel_1)
    channel_2[5:11, 5:11] = 120
    labels_data = np.zeros_like(channel_1, dtype=np.int32)
    labels_data[2:7, 2:7] = 1
    labels_data[7:11, 7:11] = 2
    widget = _widget(qtbot, channel_1, axes="YX")
    labels = widget.add_node_from_palette("label_connected_components")
    second_channel = widget.add_node_from_palette("calculate_weighted_image")
    coloc = widget.add_node_from_palette("object_colocalization_metrics")

    _publish_array_output(widget, "input", channel_1, axes="YX")
    _publish_array_output(widget, labels.id, labels_data, axes="YX")
    _publish_array_output(widget, second_channel.id, channel_2, axes="YX")
    widget._connect_nodes(labels.id, coloc.id, target_port=0)
    widget._connect_nodes("input", coloc.id, target_port=1)
    widget._connect_nodes(second_channel.id, coloc.id, target_port=2)
    _publish_table_output(widget, coloc.id)

    _select(widget, coloc.id)

    assert not widget.table_group.isHidden()
    assert not widget.colocalization_scatter_group.isHidden()
    assert widget._inspector_layout.indexOf(widget.table_group) < (
        widget._inspector_layout.indexOf(widget.colocalization_scatter_group)
    )
    assert widget.colocalization_scatter_plot._image is not None
    assert "voxels" in widget.colocalization_scatter_summary.text().casefold()


def test_threshold_has_a_real_mask_summary_after_input_histogram(qtbot):
    data = np.arange(100, dtype=np.uint16).reshape(10, 10)
    widget = _widget(qtbot, data, axes="YX")
    threshold = widget.add_node_from_palette("binary_threshold")
    _publish_array_output(widget, "input", data, axes="YX")
    _publish_array_output(widget, threshold.id, data > 40, axes="YX")
    widget._connect_nodes("input", threshold.id)

    _select(widget, threshold.id)

    mask_summary = widget._inspector_sections.get(MASK_SUMMARY_SECTION)
    assert mask_summary is not None
    assert mask_summary is not widget.histograms_section
    assert not widget.histograms_section.isHidden()
    assert not mask_summary.isHidden()
    assert widget._inspector_layout.indexOf(widget.histograms_section) < (
        widget._inspector_layout.indexOf(mask_summary)
    )
    assert not widget.rescale_input_histogram_group.isHidden()


def test_filter_labels_never_presents_label_ids_as_an_output_histogram(qtbot):
    source = np.zeros((12, 12), dtype=np.uint16)
    labels_data = np.zeros_like(source, dtype=np.int32)
    labels_data[1:5, 1:5] = 1
    labels_data[7:11, 7:11] = 17
    widget = _widget(qtbot, source, axes="YX")
    labels = widget.add_node_from_palette("label_connected_components")
    label_filter = widget.add_node_from_palette("filter_labels_by_volume")
    _publish_array_output(widget, labels.id, labels_data, axes="YX")
    _publish_array_output(widget, label_filter.id, labels_data, axes="YX")
    widget._connect_nodes(labels.id, label_filter.id)

    _select(widget, label_filter.id)

    assert not widget.label_volume_group.isHidden()
    assert widget.histogram_group.isHidden()
    assert widget.histograms_section.isHidden()
    assert np.asarray(widget.histogram_plot._counts).size == 0


def test_property_filter_prioritizes_connected_measurement_distribution(
    qtbot,
    monkeypatch,
):
    source = np.zeros((12, 12), dtype=np.uint16)
    labels_data = np.zeros_like(source, dtype=np.int32)
    labels_data[1:4, 1:4] = 1
    labels_data[5:8, 5:8] = 2
    labels_data[8:11, 8:11] = 3
    widget = _widget(qtbot, source, axes="YX")
    labels = widget.add_node_from_palette("label_connected_components")
    measurements = widget.add_node_from_palette("measure_objects")
    filtered = widget.add_node_from_palette("filter_labels_by_property")
    widget._connect_nodes(labels.id, filtered.id, target_port=0)
    widget._connect_nodes(measurements.id, filtered.id, target_port=1)
    _publish_array_output(widget, labels.id, labels_data, axes="YX")
    table = TableData(
        ("label_id", "intensity_mean", "note"),
        (
            (1, 5.0, "dim"),
            (2, 20.0, "mid"),
            (3, 40.0, "bright"),
            (4, None, "missing"),
        ),
        name="Object measurements",
        column_units=(("intensity_mean", "a.u."),),
    )
    table_state = TableState(
        table.row_count,
        table.column_count,
        table.columns,
        source_name=table.name,
        column_units=table.column_units,
    )
    widget.pipeline.outputs[measurements.id] = table
    widget.pipeline.output_states[measurements.id] = table_state
    widget.pipeline.node_outputs[measurements.id] = [table]
    widget.pipeline.node_output_states[measurements.id] = [table_state]
    widget.pipeline.completed_node_ids.add(measurements.id)
    widget.pipeline.node_execution_states[measurements.id] = EXECUTION_READY
    _publish_array_output(widget, filtered.id, labels_data, axes="YX")
    widget.pipeline.set_param(filtered.id, "property_column", "auto")
    widget.pipeline.set_param(filtered.id, "min_value", 10.0)
    widget.pipeline.set_param(filtered.id, "max_value", 30.0)

    def unexpected_label_scan(*_args, **_kwargs):
        raise AssertionError("property filtering must not scan label volumes")

    monkeypatch.setattr(
        type(widget),
        "_object_sizes",
        staticmethod(unexpected_label_scan),
    )
    _select(widget, filtered.id)

    assert not widget.label_volume_group.isHidden()
    assert widget.label_volume_group.title() == "Measurement Property Distribution"
    assert widget.label_volume_group.summary_label.text() == "intensity_mean (a.u.)"
    assert "Auto selected" in widget.label_volume_summary.text()
    assert "3 finite values from 4 rows" in widget.label_volume_summary.text()
    assert "1 finite rows match the current rule" in widget.label_volume_summary.text()
    assert widget.label_volume_plot._counts.sum() == 3
    assert widget.label_volume_plot.marker_values() == {"min": 10.0, "max": 30.0}
    assert widget.label_volume_log_checkbox.isHidden()
    assert not widget.label_volume_interaction_hint.isHidden()
    assert widget.histogram_group.isHidden()
    assert widget.histograms_section.isHidden()

    widget._on_label_volume_marker_changed("min", 12.0)

    assert widget.pipeline.nodes[filtered.id].params["min_value"] == 12.0
    assert widget.label_volume_plot.marker_values()["min"] == 12.0


def test_property_filter_reuses_bounded_identity_column_value_cache(
    qtbot,
    monkeypatch,
):
    widget = _widget(qtbot)
    table = TableData(
        ("label_id", "area"),
        ((1, 5.0), (2, 20.0), (3, float("nan"))),
    )
    original = type(widget)._extract_property_filter_column_values
    calls = []

    def counted_extract(candidate, column):
        calls.append((candidate, column))
        return original(candidate, column)

    monkeypatch.setattr(
        type(widget),
        "_extract_property_filter_column_values",
        staticmethod(counted_extract),
    )

    first = widget._property_filter_table_values(table, "area")
    second = widget._property_filter_table_values(table, "area")

    assert calls == [(table, "area")]
    assert first[1] is second[1]
    assert first[1].tolist() == [5.0, 20.0]

    retained_tables = []
    for index in range(20):
        candidate = TableData(("area",), ((float(index),),))
        retained_tables.append(candidate)
        widget._property_filter_table_values(candidate, "area")
    assert len(widget._property_filter_value_cache) <= 16


def test_split_selector_can_show_unused_port_and_refreshes_active_pin(qtbot):
    data = np.arange(3 * 8 * 9, dtype=np.uint16).reshape(3, 8, 9)
    widget = _widget(qtbot, data)
    split = widget.add_node_from_palette("split_channels")
    first_consumer = widget.add_node_from_palette("gaussian_blur")
    second_consumer = widget.add_node_from_palette("median_filter")
    widget._connect_nodes("input", split.id)
    widget._connect_nodes(split.id, first_consumer.id, source_port=0)
    widget._connect_nodes(split.id, second_consumer.id, source_port=1)
    outputs = [np.asarray(data[index]) for index in range(data.shape[0])]
    _publish_multi_output(widget, split.id, outputs)
    _select(widget, split.id)

    assert widget.output_selector_combo.isEnabled()
    assert widget._used_split_channel_ports(split.id) == (0, 1)
    connections_before = _connection_snapshot(widget)

    widget.output_selector_combo.setCurrentIndex(2)

    selected_data, _state, selected_port = widget._node_display_payload(split.id)
    assert selected_port == 2
    np.testing.assert_array_equal(selected_data, outputs[2])
    assert _connection_snapshot(widget) == connections_before

    widget.pin_node(split.id)
    pinned = widget._active_pinned_layer()
    assert pinned is not None
    assert pinned.metadata["output_port"] == 2
    np.testing.assert_array_equal(pinned.data, outputs[2])

    widget.output_selector_combo.setCurrentIndex(0)

    pinned = widget._active_pinned_layer()
    assert pinned is not None
    assert pinned.metadata["output_port"] == 0
    np.testing.assert_array_equal(pinned.data, outputs[0])
    assert _connection_snapshot(widget) == connections_before


def test_dynamic_output_port_refresh_rebuilds_visible_selector(qtbot):
    three_channels = np.arange(3 * 6 * 7, dtype=np.uint16).reshape(3, 6, 7)
    widget = _widget(qtbot, three_channels)
    split = widget.add_node_from_palette("split_channels")
    widget._connect_nodes("input", split.id)
    _publish_array_output(widget, "input", three_channels, axes="CYX")
    _publish_multi_output(
        widget,
        split.id,
        [three_channels[index] for index in range(3)],
    )
    _select(widget, split.id)
    assert widget.output_selector_combo.count() == 3

    five_channels = np.arange(5 * 6 * 7, dtype=np.uint16).reshape(5, 6, 7)
    _publish_array_output(widget, "input", five_channels, axes="CYX")
    _publish_multi_output(
        widget,
        split.id,
        [five_channels[index] for index in range(5)],
    )
    assert len(widget.pipeline.output_ports(split.id)) == 5

    widget._refresh_dynamic_output_ports()

    assert widget.output_selector_combo.count() == 5
    assert [
        widget.output_selector_combo.itemData(index)
        for index in range(widget.output_selector_combo.count())
    ] == [0, 1, 2, 3, 4]


def test_writer_status_surfaces_execution_detail_and_summary(qtbot):
    widget = _widget(qtbot)
    writer = widget.add_node_from_palette("save_output")
    widget.pipeline.node_execution_states[writer.id] = EXECUTION_ERROR
    widget.pipeline.node_execution_messages[writer.id] = (
        "Output already exists; choose overwrite or a new path."
    )

    _select(widget, writer.id)

    assert not widget.writer_status_section.isHidden()
    assert widget.writer_status_section.summary_label.text() == "Error"
    assert widget.writer_status_label.text() == (
        "Error\nOutput already exists; choose overwrite or a new path."
    )
    assert widget.writer_status_label.wordWrap()


def test_batch_output_status_uses_scannable_field_rows(qtbot):
    widget = _widget(qtbot)
    writer = widget.add_node_from_palette("batch_output")
    writer.params.update(
        {
            "tag": "Remove_Small_Objects-remove_small_objects_1",
            "format": "batch default",
            "subfolder": "labels/cleaned",
            "filename_template": "{source_stem}__{tag}",
        }
    )
    widget.pipeline.node_execution_states[writer.id] = EXECUTION_READY

    _select(widget, writer.id)

    assert widget.writer_status_section.summary_label.text() == ("Ready · Batch only")
    assert widget.writer_status_label.isHidden()
    assert not widget.batch_output_status_panel.isHidden()
    assert widget.batch_output_status_description.text() == (
        "Written when this workflow runs in the Batch workspace."
    )
    assert {
        name: value.text()
        for name, (_label, value) in widget.batch_output_status_rows.items()
    } == {
        "tag": "Remove_Small_Objects-remove_small_objects_1",
        "format": "Batch default",
        "folder": "labels/cleaned",
        "filename": "{source_stem}__{tag}",
    }
    assert all(
        value.wordWrap() for _label, value in widget.batch_output_status_rows.values()
    )
    assert "Filename: {source_stem}__{tag}." in (
        widget.writer_status_section.accessibleDescription()
    )


def test_batch_output_status_omits_empty_folder_row(qtbot):
    widget = _widget(qtbot)
    writer = widget.add_node_from_palette("batch_output")
    writer.params["subfolder"] = ""
    widget.pipeline.node_execution_states[writer.id] = EXECUTION_READY

    _select(widget, writer.id)

    folder_label, folder_value = widget.batch_output_status_rows["folder"]
    assert folder_label.isHidden()
    assert folder_value.isHidden()
    assert "Folder:" not in widget.writer_status_section.accessibleDescription()


def test_clip_mode_change_updates_histogram_drag_guidance(qtbot):
    data = np.arange(100, dtype=np.uint16).reshape(10, 10)
    widget = _widget(qtbot, data, axes="YX")
    clip = widget.add_node_from_palette("clip_intensity")
    _publish_array_output(widget, "input", data, axes="YX")
    _publish_array_output(widget, clip.id, data, axes="YX")
    widget._connect_nodes("input", clip.id)
    _select(widget, clip.id)

    assert widget.pipeline.nodes[clip.id].params["cutoff_mode"] == "Data range"
    assert widget.histogram_interaction_hint.isHidden()

    widget._on_param_changed("cutoff_mode", "Values")
    widget._debounce_timer.stop()

    assert not widget.histogram_interaction_hint.isHidden()
    assert widget.histogram_interaction_hint.text() == (
        "Drag either cutoff line to tune the input range."
    )

    widget._on_param_changed("cutoff_mode", "Data range")
    widget._debounce_timer.stop()

    assert widget.histogram_interaction_hint.isHidden()


def _large_label_filter_inspector(qtbot, labels_data):
    widget = _widget(qtbot, np.zeros(labels_data.shape, dtype=np.uint16), axes="YX")
    labels = widget.add_node_from_palette("label_connected_components")
    filtered = widget.add_node_from_palette("filter_labels_by_volume")
    widget._connect_nodes(labels.id, filtered.id)
    _publish_array_output(widget, labels.id, labels_data, axes="YX")
    _publish_array_output(widget, filtered.id, labels_data, axes="YX")
    _select(widget, "input")
    return widget, labels, filtered


def test_large_label_volume_inspector_queues_then_renders_and_reuses_cache(
    qtbot,
    monkeypatch,
):
    labels_data = np.zeros((24, 28), dtype=np.int32)
    labels_data[1:7, 2:9] = 1
    labels_data[10:22, 14:26] = 9
    widget, _labels, filtered = _large_label_filter_inspector(qtbot, labels_data)
    pool = _QueuedThreadPool()
    widget._label_volume_thread_pool = pool
    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_BYTES", 1)
    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_ELEMENTS", 1)
    calls = []

    def exact_volumes(values, spatial_ndim, connectivity):
        calls.append((values, int(spatial_ndim), connectivity))
        return np.asarray([42, 144], dtype=np.int64)

    monkeypatch.setattr(
        type(widget),
        "_object_sizes",
        staticmethod(exact_volumes),
    )
    widget._clear_label_volume_cache()

    _select(widget, filtered.id)

    # Parameter construction and inspector presentation must not perform the
    # exact all-label scan on the GUI thread for a large input.
    assert calls == []
    assert len(pool.workers) == 1
    assert widget._active_label_volume_run_id is not None
    assert widget._pending_label_volume_request is None
    assert "calculating exact object sizes" in (
        widget.label_volume_summary.text().casefold()
    )
    assert np.asarray(widget.label_volume_plot._counts).size == 0
    assert widget._parameter_widgets["min_volume"]._bounds.maximum >= 1
    assert widget._parameter_widgets["max_volume"]._bounds.maximum >= 1

    pool.workers[0].run()

    assert widget._active_label_volume_run_id is None
    assert len(calls) == 1
    assert calls[0][0] is labels_data
    assert calls[0][1] == 2
    assert calls[0][2] == "Label IDs"
    assert widget.label_volume_plot._counts.sum() == 2
    assert "2 objects" in widget.label_volume_summary.text()
    assert widget._current_label_volume_key in widget._label_volume_cache

    widget._update_label_volume_histogram()
    widget._update_label_volume_histogram()

    assert len(pool.workers) == 1
    assert len(calls) == 1
    assert widget.label_volume_plot._counts.sum() == 2


def test_large_label_volume_replacement_cancels_stale_request_and_renders_latest(
    qtbot,
    monkeypatch,
):
    first = np.zeros((24, 28), dtype=np.int32)
    first[1:7, 2:9] = 1
    replacement = np.zeros_like(first)
    replacement[2:5, 3:8] = 2
    replacement[10:18, 15:23] = 3
    widget, labels, filtered = _large_label_filter_inspector(qtbot, first)
    pool = _QueuedThreadPool()
    widget._label_volume_thread_pool = pool
    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_BYTES", 1)
    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_ELEMENTS", 1)
    calls = []

    def exact_volumes(values, _spatial_ndim, connectivity):
        del connectivity
        calls.append(values)
        if values is first:
            return np.asarray([42], dtype=np.int64)
        assert values is replacement
        return np.asarray([15, 64], dtype=np.int64)

    monkeypatch.setattr(
        type(widget),
        "_object_sizes",
        staticmethod(exact_volumes),
    )
    widget._clear_label_volume_cache()
    _select(widget, filtered.id)
    first_worker = pool.workers[0]
    first_key = first_worker.request.key
    cancel_event = first_worker.request.cancel_event
    assert cancel_event is not None and not cancel_event.is_set()

    _publish_array_output(widget, labels.id, replacement, axes="YX")
    widget._update_label_volume_histogram()

    assert cancel_event.is_set()
    assert widget._pending_label_volume_request is not None
    assert widget._pending_label_volume_request.data is replacement
    assert widget._pending_label_volume_request.key != first_key
    assert len(pool.workers) == 1

    # Completing the cancelled request must not calculate, cache, or paint it;
    # it only hands off to the coalesced request for the current input object.
    first_worker.run()

    assert calls == []
    assert first_key not in widget._label_volume_cache
    assert len(pool.workers) == 2
    latest_worker = pool.workers[1]
    assert latest_worker.request.data is replacement
    assert widget._active_label_volume_run_id == latest_worker.request.run_id

    latest_worker.run()

    assert len(calls) == 1
    assert calls[0] is replacement
    assert widget._active_label_volume_run_id is None
    assert widget._pending_label_volume_request is None
    assert widget.label_volume_plot._counts.sum() == 2
    assert "2 objects" in widget.label_volume_summary.text()
    assert widget._current_label_volume_key == latest_worker.request.key
    assert latest_worker.request.key in widget._label_volume_cache


def test_large_remove_small_mask_queues_exact_connectivity_and_keeps_marker(
    qtbot,
    monkeypatch,
):
    mask = np.zeros((24, 28), dtype=bool)
    mask[1, 1] = True
    mask[2, 2] = True
    mask[10:12, 14:17] = True
    filtered = mask.copy()
    filtered[1, 1] = False
    filtered[2, 2] = False
    widget = _widget(qtbot, mask, axes="YX")
    mask_source = widget.add_node_from_palette("binary_threshold")
    remove = widget.add_node_from_palette("remove_small_objects")
    widget._connect_nodes(mask_source.id, remove.id)
    _publish_array_output(widget, mask_source.id, mask, axes="YX")
    _publish_array_output(widget, remove.id, filtered, axes="YX")
    widget.pipeline.set_param(remove.id, "min_size", 2)
    pool = _QueuedThreadPool()
    widget._label_volume_thread_pool = pool
    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_BYTES", 1)
    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_ELEMENTS", 1)
    calls = []

    def exact_sizes(values, spatial_ndim, connectivity):
        calls.append((values, int(spatial_ndim), connectivity))
        return object_sizes(values, spatial_ndim, connectivity)

    monkeypatch.setattr(
        type(widget),
        "_object_sizes",
        staticmethod(exact_sizes),
    )
    widget._clear_label_volume_cache()

    _select(widget, remove.id)

    assert calls == []
    assert len(pool.workers) == 1
    assert pool.workers[0].request.connectivity == "Face connected"
    assert "calculating exact object sizes" in (
        widget.label_volume_summary.text().casefold()
    )

    pool.workers[0].run()

    assert calls == [(mask, 2, "Face connected")]
    assert "3 objects" in widget.label_volume_summary.text()
    assert widget.label_volume_plot.marker_values()["min"] == 2.0
    face_key = widget._current_label_volume_key

    widget.pipeline.set_param(remove.id, "connectivity", "Full connectivity")
    widget._update_label_volume_histogram()

    assert len(pool.workers) == 2
    assert pool.workers[1].request.connectivity == "Full connectivity"
    assert pool.workers[1].request.key != face_key
    pool.workers[1].run()

    assert calls[-1] == (mask, 2, "Full connectivity")
    assert "2 objects" in widget.label_volume_summary.text()
    assert widget.label_volume_plot.marker_values()["min"] == 2.0
