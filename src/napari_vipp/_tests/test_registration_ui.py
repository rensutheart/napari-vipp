"""Registration is a typed workflow result, never a pretend image preview."""

from types import SimpleNamespace

import numpy as np
import pytest

from napari_vipp.core.pipeline import (
    EXECUTION_READY,
    EXECUTION_STALE,
    NODE_LIBRARY_BY_ID,
)
from napari_vipp.core.tables import TableData, TableState
from napari_vipp.core.transforms import (
    RegistrationGrid,
    TransformData,
    TransformState,
    load_transform,
)
from napari_vipp.ui.connected_inputs import connected_input_scientific_summary
from napari_vipp.ui.inspector import (
    HISTOGRAMS_SECTION,
    NEXT_STEP_SECTION,
    REGISTRATION_RESULTS_SECTION,
    TABLE_RESULTS_SECTION,
    inspector_profile,
)
from napari_vipp.ui.output_formats import writer_format_choices


def _transform():
    grid = RegistrationGrid(("y", "x"), (16, 18), (1.0, 1.0), (0.0, 0.0),
                            "pixel", "synthetic-frame")
    state = TransformState("Translation", ("y", "x"), 1, "pixel",
                           "Synthetic moving", "Synthetic reference")
    return TransformData((np.eye(3),), grid, grid, state)


@pytest.fixture
def registration_layout_compat(request, monkeypatch):
    if getattr(request, "param", "native") == "portable":
        from napari_vipp.ui import inspector

        monkeypatch.setattr(
            inspector, "_independent_layout_constraints_available",
            lambda _layout: False,
        )
    return getattr(request, "param", "native")


@pytest.fixture
def registration_widget(qtbot, registration_layout_compat):
    from napari_vipp._tests.test_widget import _Viewer
    from napari_vipp._widget import VippWidget

    widget = VippWidget(_Viewer(np.zeros((16, 18), np.float32)),
                        defer_initial_run=True)
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("estimate_registration")
    widget._debounce_timer.stop()
    return widget, node


def _publish(widget, node):
    transform = _transform()
    diagnostics = TableData(("time", "overlap"), ((0, 0.95),),
                            table_kind="registration diagnostics")
    table_state = TableState(1, 2, diagnostics.columns,
                             table_kind=diagnostics.table_kind)
    widget.pipeline.outputs[node.id] = transform
    widget.pipeline.node_outputs[node.id] = [transform, diagnostics]
    widget.pipeline.output_states[node.id] = transform.state
    widget.pipeline.node_output_states[node.id] = [transform.state, table_state]
    widget.pipeline.node_execution_states[node.id] = EXECUTION_READY
    widget._selected_node_id = node.id
    return transform


def test_transform_inspector_profile_has_no_image_actions_or_histograms():
    profile = inspector_profile(NODE_LIBRARY_BY_ID["estimate_registration"],
                                effective_output_type="transform")
    assert not profile.supports_pin
    assert profile.output_action_kind == "multi_transform"
    assert HISTOGRAMS_SECTION not in profile.primary_sections
    assert not profile.show_output_selector
    assert TABLE_RESULTS_SECTION in profile.primary_sections
    assert NEXT_STEP_SECTION in profile.primary_sections
    assert REGISTRATION_RESULTS_SECTION in profile.primary_sections
    assert profile.supports_all_outputs_action
    assert profile.execution_is_manual


def test_connected_transform_summary_is_readable_and_directional():
    transform = _transform()
    summary = connected_input_scientific_summary(None, transform.state)
    assert "Translation" in summary
    assert "YX" in summary
    assert "1 transform" in summary
    assert "moving → reference" in summary


def test_transform_card_has_no_thumbnail_before_or_after_calculation(
    registration_widget,
):
    widget, node = registration_widget
    assert not widget.graph_view._cards[node.id]._preview_enabled
    transform = _publish(widget, node)
    widget._update_node_thumbnail(node.id, transform, transform.state, 0,
                                  queue_stack_contrast=False)
    assert not widget.graph_view._cards[node.id]._preview_enabled
    assert not widget._node_preview_enabled(node.id)
    assert not widget._node_can_pin(node.id)
    assert widget._data_kind(transform) == "transform"
    assert not widget._display_rgb(transform)
    assert not widget._can_save_selected_output_as_raster(node.id)
    with pytest.raises(ValueError, match="Non-image"):
        widget._display_data(transform)


def test_select_transform_keeps_viewer_and_histograms_safe(registration_widget):
    widget, node = registration_widget
    _publish(widget, node)
    original_layer_count = len(widget.viewer.layers)
    widget._select_node(node.id)
    widget._update_histogram()
    widget.inspect_node(node.id)
    assert len(widget.viewer.layers) == original_layer_count
    message = widget.status_label.text()
    assert "registration transform" in message
    assert "Review the Diagnostics table in the inspector." in message
    assert "select Diagnostics" not in message
    assert widget.histogram_group.isHidden()
    assert widget.save_button.isHidden()
    assert widget.pin_button.isHidden()
    assert widget.output_selector_section.isHidden()
    assert widget.table_preview.rowCount() == 1
    assert widget._registration_results.export_transform.isEnabled()


def test_diagnostics_port_remains_an_ordinary_table(registration_widget):
    widget, node = registration_widget
    _publish(widget, node)
    widget._select_node(node.id)
    assert widget._node_output_type(node.id) == "transform"
    assert widget.table_preview.rowCount() == 1
    assert widget.table_group.title() == "Diagnostics"
    assert not widget.table_group.isHidden()
    widget._open_result_table_dialog()
    assert widget._result_table_dialog.context_key == (node.id, 1)
    assert widget._registration_results.export_diagnostics.isEnabled()
    assert not widget._node_can_pin(node.id)


def test_registration_shows_both_metadata_and_ignores_old_output_selection(
    registration_widget,
):
    widget, node = registration_widget
    transform = _publish(widget, node)
    widget._inspector_output_port_by_node[node.id] = 1
    widget._select_node(node.id)
    assert widget._node_display_payload(node.id) == (transform, transform.state, 0)
    assert widget.table_preview.rowCount() == 1
    labels = [widget.metadata_table.item(row, 0).text()
              for row in range(widget.metadata_table.rowCount())]
    assert "Transform · Motion model" in labels
    assert "Diagnostics · Rows" in labels
    assert "Translation" in widget._registration_results.summary.text()


def test_registration_diagnostics_export_uses_exact_port_without_selection(
    registration_widget, tmp_path,
):
    widget, node = registration_widget
    _publish(widget, node)
    widget._select_node(node.id)
    target = tmp_path / "diagnostics.csv"
    assert widget._save_node_output(
        node.id, str(target), format="csv", output_port=1,
    ) == target
    assert target.read_text().splitlines() == ["time,overlap", "0,0.95"]
    assert widget._node_display_payload(node.id)[2] == 0
    widget.pipeline.node_execution_states[node.id] = EXECUTION_STALE
    widget._sync_execution_ui()
    assert not widget._registration_results.export_diagnostics.isEnabled()
    assert not widget._registration_results.export_transform.isEnabled()
    assert not widget._registration_results.export_both.isEnabled()
    other = tmp_path / "stale.csv"
    assert widget._save_node_output(
        node.id, str(other), format="csv", output_port=1,
    ) is None
    assert not other.exists()


def test_registration_export_actions_target_named_ports(
    registration_widget, monkeypatch, tmp_path,
):
    widget, node = registration_widget
    _publish(widget, node)
    widget._select_node(node.id)
    from napari_vipp.ui import registration_results

    captured = []
    monkeypatch.setattr(
        widget, "_save_node_output",
        lambda *args, **kwargs: captured.append((args, kwargs)),
    )
    monkeypatch.setattr(registration_results.QFileDialog, "getSaveFileName",
                        lambda *args: (str(tmp_path / "transform.json"), ""))
    monkeypatch.setattr(registration_results, "choose_table_export_target",
                        lambda *args, **kwargs: (tmp_path / "diagnostics.tsv", "tsv"))
    widget._registration_results.export_transform.click()
    widget._registration_results.export_diagnostics.click()
    assert [(entry[1]["output_port"], entry[1]["format"]) for entry in captured] == [
        (0, "json"), (1, "tsv"),
    ]
    assert widget._node_display_payload(node.id)[2] == 0


def test_registration_diagnostics_popout_refreshes_its_port_off_selection(
    registration_widget,
):
    widget, node = registration_widget
    _publish(widget, node)
    widget._select_node(node.id)
    widget._open_result_table_dialog()
    dialog = widget._result_table_dialog
    widget._select_node("input")
    updated = TableData(("time", "overlap"), ((0, 0.99), (1, 0.97)),
                        table_kind="registration diagnostics")
    widget.pipeline.node_outputs[node.id][1] = updated
    widget.pipeline.node_output_states[node.id][1] = TableState(
        2, 2, updated.columns, table_kind=updated.table_kind,
    )
    widget._refresh_node_presentation_surfaces({node.id})
    assert dialog.isVisible()
    assert dialog.context_key == (node.id, 1)
    assert dialog.table is updated
    assert dialog.model.rowCount() == 2
    assert widget._selected_node_id == "input"


def test_registration_popout_export_guard_is_bound_to_its_result(
    registration_widget, tmp_path,
):
    widget, node = registration_widget
    _publish(widget, node)
    widget._select_node(node.id)
    widget._open_result_table_dialog()
    dialog = widget._result_table_dialog
    widget._select_node("input")
    target = tmp_path / "current.csv"
    assert dialog.export_table(target) == target
    widget.pipeline.node_execution_states[node.id] = EXECUTION_STALE
    widget._sync_result_table_dialog_attention()
    assert not dialog.export_button.isEnabled()
    with pytest.raises(ValueError, match="not current"):
        dialog.export_table(tmp_path / "stale.csv")
    ordinary = widget.add_node_from_palette("measure_objects")
    widget._debounce_timer.stop()
    widget._sync_result_table_dialog(dialog.table, 0, node_id=ordinary.id)
    assert dialog.export_guard is None
    assert dialog.export_table(tmp_path / "ordinary.csv").is_file()


def test_transform_export_requires_current_result(registration_widget, tmp_path):
    widget, node = registration_widget
    transform = _publish(widget, node)
    target = tmp_path / "motion.json"
    assert widget._save_node_output(node.id, str(target), format="json") == target
    assert load_transform(target).to_dict() == transform.to_dict()
    target.unlink()
    widget.pipeline.node_execution_states[node.id] = EXECUTION_STALE
    assert widget._save_node_output(node.id, str(target), format="json") is None
    assert not target.exists()
    assert "Calculate the updated registration" in widget.status_label.text()


def test_registration_controls_explain_scope_and_update_mode(registration_widget):
    widget, node = registration_widget
    assert "reference_channel" in widget._parameter_widgets
    assert "reference_time" not in widget._parameter_widgets
    assert "iterations" not in widget._parameter_widgets
    assert "all channels" in widget._parameter_widgets["channel"].toolTip()
    widget._on_param_changed("mode", "Time series")
    widget._debounce_timer.stop()
    assert "reference_time" in widget._parameter_widgets
    assert "reference_channel" not in widget._parameter_widgets
    assert len(widget.pipeline.input_ports(node.id)) == 1
    assert "Z slices are never" in widget._operation_help_note(node.id)
    widget._on_param_changed("model", "Affine")
    widget._debounce_timer.stop()
    assert "iterations" in widget._parameter_widgets
    assert "alter object shape and size" in widget._operation_help_note(node.id)
    assert widget._operation_help_note_status(node.id) == "Warning"
    assert widget._parameter_widgets["operation_notice"].property(
        "vippTextTone"
    ) == "warning"
    assert widget.parameter_group.summary_label.text() == "9 values"


def test_registration_name_summary_only_describes_active_mode(registration_widget):
    from napari_vipp.ui.node_labels import build_node_presentations

    widget, node = registration_widget
    summary = build_node_presentations(widget.pipeline, {})[node.id].summary
    assert "reference channel" in summary
    assert "reference time" not in summary
    widget.pipeline.set_param(node.id, "mode", "Time series")
    summary = build_node_presentations(widget.pipeline, {})[node.id].summary
    assert "reference time" in summary
    assert "reference channel" not in summary


def test_initial_graph_ports_follow_time_series_mode():
    from napari_vipp._graph import (
        _node_input_port_count,
        _node_input_port_labels,
        _node_input_port_types,
    )
    node = SimpleNamespace(operation_id="estimate_registration", has_input=True,
                           params={"mode": "Time series"})
    assert _node_input_port_count(node) == 1
    assert _node_input_port_labels(node) == ["Time series"]
    assert _node_input_port_types(node) == ["array"]


def test_transform_batch_format_choices_only_offer_json():
    pipeline = SimpleNamespace(
        nodes={"out": SimpleNamespace(operation_id="batch_output")},
        output_ports=lambda node: [SimpleNamespace(output_type="transform")],
    )
    assert writer_format_choices(pipeline, "out", (
        "batch default", "ome-tiff", "csv", "json", "obj"
    )) == ("batch default", "json")


@pytest.mark.parametrize("identity", ("", "source_uuid", "uri"))
def test_connect_registration_refreshes_only_anonymous_cached_sources(
    registration_widget, monkeypatch, identity,
):
    from dataclasses import replace

    from napari_vipp.core.metadata import SourceMetadata, image_state_from_array

    widget, node = registration_widget
    source_data = np.zeros((16, 18), np.float32)
    state = image_state_from_array(source_data, layer_metadata={"axes": "YX"})
    if identity:
        state = replace(state, source=SourceMetadata(**{identity: "known-frame"}))
    widget.pipeline.outputs["input"] = source_data
    widget.pipeline.output_states["input"] = state
    widget._pending_dirty_node_ids.clear()
    calls = []
    monkeypatch.setattr(widget, "run_pipeline", lambda: calls.append(True))
    widget._connect_nodes("input", node.id, target_port=0)
    assert ("input" in widget._pending_dirty_node_ids) is (not identity)
    assert widget.pipeline.is_manual_node(node.id)
    assert widget.pipeline.outputs.get(node.id) is None
    assert calls == [True]


def test_registration_operations_always_leave_the_gui_thread():
    from napari_vipp._widget import BACKGROUND_PIPELINE_OPERATIONS
    assert {"estimate_registration", "apply_transform", "compare_images"} <= (
        BACKGROUND_PIPELINE_OPERATIONS
    )


def test_compare_coverage_control_updates_graph_ports(registration_widget):
    from napari_vipp._graph import (
        _node_input_port_count,
        _node_input_port_labels,
        _node_input_port_types,
    )

    widget, _estimate = registration_widget
    node = widget.add_node_from_palette("compare_images")
    assert _node_input_port_count(node) == 2
    assert len(widget.pipeline.input_ports(node.id)) == 2
    widget._on_param_changed("use_mask", True)
    widget._debounce_timer.stop()
    assert _node_input_port_count(node) == 3
    assert _node_input_port_labels(node)[2] == "Valid coverage"
    assert _node_input_port_types(node)[2] == "mask"
    assert len(widget.pipeline.input_ports(node.id)) == 3
    widget._connect_nodes("threshold", node.id, target_port=2)
    assert 2 in widget.pipeline.input_data_by_port_for_node(node.id)
    widget._on_param_changed("use_mask", False)
    widget._debounce_timer.stop()
    assert len(widget.pipeline.input_ports(node.id)) == 2
    assert 2 not in widget.pipeline.input_data_by_port_for_node(node.id)
    assert "same physical grid" in widget._operation_help_note(node.id)


@pytest.mark.parametrize("reason", ("superseded", "edited", "cancelled"))
def test_old_transform_worker_callback_cannot_replace_current_result(
    registration_widget, reason,
):
    import threading

    from napari_vipp.core.execution import PipelineNodeResult

    widget, node = registration_widget
    current = _publish(widget, node)
    older = _transform()
    widget._active_pipeline_run_id = 47
    widget._pending_dirty_node_ids.clear()
    widget._pipeline_cancel_events[47] = threading.Event()
    if reason == "edited":
        widget._pending_dirty_node_ids.add(node.id)
    elif reason == "cancelled":
        widget._cancel_background_pipeline_run()
        assert widget._pipeline_cancel_events[47].is_set()
        assert "Stopping calculation" in widget.pipeline_busy_label.text()
    result = PipelineNodeResult(
        46 if reason == "superseded" else 47, node.id, node.operation_id,
        older, older.state, (older,), (older.state,), EXECUTION_READY,
    )
    widget._on_background_pipeline_node_finished(result)
    assert node.id not in widget._background_node_result_overrides
    assert node.id not in widget._background_execution_state_overrides
    assert widget.pipeline.outputs[node.id] is current
    widget._active_pipeline_run_id = None
    widget._pipeline_user_cancel_requested_run_id = None
    widget._pipeline_cancel_events.clear()


@pytest.mark.parametrize("theme", ("dark", "light"))
@pytest.mark.parametrize(
    "registration_layout_compat", ("native", "portable"), indirect=True,
)
@pytest.mark.parametrize(
    "view", ("pair", "time", "affine", "diagnostics", "apply", "compare"),
)
def test_registration_inspectors_render_at_narrow_width(
    qtbot, monkeypatch, registration_widget, registration_layout_compat, theme, view,
):
    import os
    from pathlib import Path

    from napari.qt import get_stylesheet
    from qtpy.QtGui import QTextDocument
    from qtpy.QtWidgets import QMainWindow

    widget, estimate = registration_widget
    monkeypatch.setattr(widget, "_confirm_close_dirty_workflow_tabs", lambda: True)
    host = QMainWindow()
    host.setCentralWidget(widget)
    qtbot.addWidget(host)
    host.resize(1240, 1000)
    host.setStyleSheet(get_stylesheet(theme, extra_variables={"font_size": "10pt"}))
    host.show()
    qtbot.waitUntil(lambda: widget.property("vippColorScheme") == theme)
    if view == "time":
        widget._on_param_changed("mode", "Time series")
    elif view == "affine":
        widget._on_param_changed("model", "Affine")
    elif view in {"apply", "compare"}:
        node = widget.add_node_from_palette(
            "apply_transform" if view == "apply" else "compare_images"
        )
        if view == "compare":
            widget._on_param_changed("use_mask", True)
        else:
            _publish(widget, estimate)
            widget._connect_nodes("input", node.id, target_port=0)
            widget._connect_nodes(estimate.id, node.id, target_port=1)
        widget._select_node(node.id)
    else:
        _publish(widget, estimate)
        widget._select_node(estimate.id)
        if view == "diagnostics":
            widget._on_inspector_output_selector_changed(1)
    widget._debounce_timer.stop()
    widget.splitter.setSizes([210, 650, 380])
    qtbot.wait(200)
    assert widget.inspector_panel.width() <= 400
    assert widget.inspector_content.width() <= widget.inspector_panel.width()
    assert not widget.inspector_panel.horizontalScrollBar().isVisible()
    note = widget._parameter_widgets["operation_notice"]
    assert note.geometry().bottom() < widget.parameter_form_widget.height()
    document = QTextDocument()
    document.setDocumentMargin(0)
    document.setDefaultFont(note.font())
    document.setHtml(note.text())
    document.setTextWidth(note.contentsRect().width())
    assert note.height() >= document.size().height(), (
        note.size(), document.size(), widget.parameter_form_widget.size(),
        note.parentWidget().size(),
    )
    child = note
    while child.parentWidget() is not widget.inspector_content:
        container = child.parentWidget()
        assert child.geometry().bottom() < container.height(), (
            type(child).__name__, child.geometry(), type(container).__name__,
            container.geometry(), container.minimumSize(),
            container.layout().minimumSize(),
            container.layout().totalHeightForWidth(container.width()),
        )
        child = container
    screenshot_dir = os.environ.get("VIPP_REGISTRATION_UI_SCREENSHOTS")
    if screenshot_dir:
        target = Path(screenshot_dir)
        target.mkdir(parents=True, exist_ok=True)
        filename = f"registration-{view}-{theme}-{registration_layout_compat}.png"
        assert widget.inspector_content.grab().save(
            str(target / filename)
        )
        assert note.geometry().bottom() < widget.parameter_form_widget.height()
        assert note.height() >= document.size().height(), (
            note.size(), document.size(), widget.parameter_form_widget.size(),
            note.parentWidget().size(),
        )
        child = note
        while child.parentWidget() is not widget.inspector_content:
            container = child.parentWidget()
            assert child.geometry().bottom() < container.height(), (
                type(child).__name__, child.geometry(), type(container).__name__,
                container.geometry(), container.minimumSize(),
                container.layout().minimumSize(),
                container.layout().totalHeightForWidth(container.width()),
            )
            child = container
    sections = [
        item.widget()
        for index in range(widget._inspector_layout.count())
        if (item := widget._inspector_layout.itemAt(index)).widget() is not None
        and not item.widget().isHidden()
    ]
    for first, second in zip(sections, sections[1:], strict=False):
        assert first.geometry().bottom() < second.geometry().top(), (
            first.objectName(), first.geometry(),
            second.objectName(), second.geometry(),
        )


@pytest.mark.parametrize(
    "registration_layout_compat", ("native", "portable"), indirect=True,
)
def test_registration_guidance_commits_ancestor_minimums_without_another_event(
    qtbot, monkeypatch, registration_widget, registration_layout_compat,
):
    from qtpy.QtWidgets import QApplication

    widget, _node = registration_widget
    monkeypatch.setattr(widget, "_confirm_close_dirty_workflow_tabs", lambda: True)
    widget.resize(1240, 900)
    widget.show()
    widget.splitter.setSizes([210, 650, 380])
    QApplication.processEvents()
    note = widget._parameter_widgets["operation_notice"]
    for repeats in (30, 2, 45):
        note.setText(
            "Registration uses one transform for the complete volume. " * repeats
        )
        note._sync_wrapped_minimum_height()
        widget._sync_parameter_form_height()
        # Check synchronously: a queued LayoutRequest may already have been
        # handled/coalesced before this later wrapped-height change occurred.
        for layout in widget._registration_guidance_layouts():
            assert (
                layout.parentWidget().minimumHeight()
                >= layout.totalMinimumSize().height()
            )
