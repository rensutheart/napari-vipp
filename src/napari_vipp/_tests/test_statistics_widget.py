"""Statistics uses workflow ownership, undo and the existing result-table surface."""

from napari_vipp._tests.test_statistics_ui import _table
from napari_vipp._tests.test_ui_inspector_widget_integration import (
    _publish_table_output,
    _select,
    _widget,
)
from napari_vipp.core.pipeline import EXECUTION_ERROR
from napari_vipp.core.statistics import summarize_statistics


def _statistics_widget(qtbot, *, legacy=False):
    widget = _widget(qtbot)
    source = widget.add_node_from_palette("table_source")
    node = widget.add_node_from_palette("summarize_measurements")
    table = _table()
    _publish_table_output(widget, source.id, table)
    widget._connect_nodes(source.id, node.id)
    node.params.update(
        summary_version=1 if legacy else 2,
        group_by="condition",
        value_columns="area",
        statistics="count,mean,std",
    )
    result = summarize_statistics(table, **node.params)
    _publish_table_output(widget, node.id, result)
    widget._pending_dirty_node_ids.clear()
    widget._debounce_timer.stop()
    _select(widget, node.id)
    panel = widget._statistics.panels[widget._statistics._context(node.id)]
    return widget, source, node, table, result, panel


def test_statistics_uses_compact_existing_preview_and_complete_popout(qtbot):
    widget, _source, node, _table, result, panel = _statistics_widget(qtbot)
    widget._update_metadata_panel()
    assert not widget.table_group.isHidden()
    assert widget.table_preview.columnCount() == 4
    assert result.column_count > widget.table_preview.columnCount()
    assert "valid n" in widget.table_preview.horizontalHeaderItem(1).text()
    assert "—" == widget.table_preview.item(0, 3).text()
    assert "Undefined" in widget.table_preview.item(0, 3).toolTip()
    assert panel.result is result
    assert not panel.stale
    assert not widget._node_can_pin(node.id)
    widget._open_result_table_dialog()
    assert widget._result_table_dialog.isVisible()
    assert widget.pipeline.outputs[node.id] is result


def test_recipe_edit_is_one_undo_step_and_preserves_upstream(qtbot):
    widget, source, node, table, _result, panel = _statistics_widget(qtbot)
    before = dict(node.params)
    history_count = len(widget._history.undo_stack)
    panel.params_changed.emit({**before, "statistics": "mean,median", "group_by": ""})
    assert node.params["statistics"] == "mean,median"
    assert len(widget._history.undo_stack) == history_count + 1
    assert widget.pipeline.outputs[source.id] is table
    assert source.id not in widget._pending_dirty_node_ids
    assert node.id in widget._pending_dirty_node_ids
    assert panel.stale
    widget.undo()
    assert widget.pipeline.nodes[node.id].params == before
    assert widget.pipeline.outputs[source.id] is table
    widget.redo()
    assert widget.pipeline.nodes[node.id].params["statistics"] == "mean,median"


def test_upgrade_is_explicit_atomic_undoable_and_retains_raw_choices(qtbot):
    widget, source, node, table, _result, panel = _statistics_widget(qtbot, legacy=True)
    before = dict(node.params)
    _select(widget, source.id)
    _select(widget, node.id)
    assert node.params == before
    assert not panel.legacy.isHidden()
    history_count = len(widget._history.undo_stack)
    panel.upgrade_button.click()
    assert node.params["summary_version"] == 2
    assert node.params["statistics"] == before["statistics"]
    assert len(widget._history.undo_stack) == history_count + 1
    assert widget.pipeline.outputs[source.id] is table
    widget.undo()
    assert widget.pipeline.nodes[node.id].params == before
    assert not panel.legacy.isHidden()


def test_generic_signal_cannot_implicitly_upgrade_legacy(qtbot):
    widget, _source, node, _table, _result, panel = _statistics_widget(
        qtbot, legacy=True
    )
    before = dict(node.params)
    panel.params_changed.emit({**before, "summary_version": 2})
    assert node.params == before
    assert "explicit upgrade" in widget.status_label.text()


def test_stale_authored_revision_is_rejected_without_overwriting_new_values(qtbot):
    widget, _source, node, _table, _result, panel = _statistics_widget(qtbot)
    old_values = dict(node.params)
    node.params["statistics"] = "median"
    panel.params_changed.emit({**old_values, "group_by": ""})
    assert node.params["statistics"] == "median"
    assert node.params["group_by"] == "condition"
    assert panel.params["statistics"] == "median"


def test_upstream_edit_and_failure_do_not_show_old_inclusion_as_current(qtbot):
    widget, source, node, _table, result, panel = _statistics_widget(qtbot)
    assert panel.inclusion_note.text()
    widget._mark_pipeline_dirty(source.id)
    assert panel.stale
    assert panel.inclusion_note.text() == ""
    widget.pipeline.set_node_execution_error(node.id, "Statistics need review.")
    widget._statistics.refresh()
    assert panel.failed
    assert widget.pipeline.node_execution_states[node.id] == EXECUTION_ERROR
    assert panel.result is result
    assert panel.inclusion_note.text() == ""


def test_workflow_switch_retains_panel_and_rejects_cross_workflow_edits(qtbot):
    widget, _source, node, _table, _result, panel = _statistics_widget(qtbot)
    context = widget._statistics._context(node.id)
    before = dict(node.params)
    widget._new_workflow()
    assert widget._workflow_tabs.current.session_id != context[0]
    panel.params_changed.emit({**before, "statistics": "median"})
    assert node.params == before
    assert panel.stale
    assert not panel.isEnabled()
    assert widget._activate_workflow_tab(0, check_safety=False)
    _select(widget, node.id)
    assert widget._statistics.panels[context] is panel
    assert panel.isEnabled()
    assert panel.params == before


def test_refresh_and_selection_never_calculate_or_modify_recipe(qtbot, monkeypatch):
    widget, source, node, table, _result, panel = _statistics_widget(qtbot)
    before = dict(node.params)
    calls = []
    monkeypatch.setattr(
        widget.pipeline, "run", lambda *args, **kwargs: calls.append(True)
    )
    for _ in range(2):
        _select(widget, source.id)
        _select(widget, node.id)
        widget._statistics.refresh()
    assert calls == []
    assert node.params == before
    assert widget.pipeline.outputs[source.id] is table
    assert widget._statistics.panels[widget._statistics._context(node.id)] is panel


def test_statistics_results_are_reachable_in_narrow_scrolling_inspector(qtbot, qapp):
    widget, _source, _node, _table, _result, controls = _statistics_widget(qtbot)
    widget._update_metadata_panel()
    inspector = widget.inspector_panel
    inspector.setParent(None)
    qtbot.addWidget(inspector)
    inspector.resize(340, 650)
    inspector.show()
    qapp.processEvents()
    assert inspector.verticalScrollBar().maximum() > 0
    inspector.ensureWidgetVisible(controls.inclusion_note)
    qapp.processEvents()
    assert inspector.viewport().rect().contains(
        controls.inclusion_note.mapTo(
            inspector.viewport(), controls.inclusion_note.rect().center()
        )
    )
    inspector.ensureWidgetVisible(widget.table_preview)
    qapp.processEvents()
    assert inspector.viewport().rect().contains(
        widget.table_preview.mapTo(
            inspector.viewport(), widget.table_preview.rect().center()
        )
    )
