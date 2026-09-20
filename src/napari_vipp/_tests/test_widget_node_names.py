"""Names survive normal editing while calculated data remain untouched."""

import json
from copy import deepcopy

from qtpy.QtCore import QPointF, Qt
from qtpy.QtWidgets import QApplication

from napari_vipp._tests.test_results_workspace_cross_workflows import _two_workflows
from napari_vipp._tests.test_results_workspace_widget import _plot, _settle
from napari_vipp._tests.test_statistics_widget import _statistics_widget
from napari_vipp.ui.workflow_save_settings import WorkflowSavePolicy


def _clean(widget):
    _settle(widget)
    widget._history.clear()
    session = widget._workflow_tabs.current
    session.mark_clean(
        widget._current_history_snapshot(),
        persistence_token=widget._workflow_tab_persistence_token(),
    )
    return session


def _science(widget):
    return (
        deepcopy(
            [
                (node.id, node.title, node.params)
                for node in widget.pipeline.nodes.values()
            ]
        ),
        tuple(widget.pipeline.connections),
        dict(widget.pipeline.node_execution_states),
        {node_id: id(value) for node_id, value in widget.pipeline.outputs.items()},
    )


def test_rename_reset_undo_redo_preserve_calculated_tables_and_dirty_state(qtbot):
    widget, source, node, table, result, _panel = _statistics_widget(qtbot)
    session = _clean(widget)
    automatic = widget._node_title(node.id)
    science = _science(widget)

    assert widget._set_node_name(node.id, "Nuclear intensity by well")
    assert widget._node_title(node.id) == "Nuclear intensity by well"
    assert session.dirty
    assert len(widget._history.undo_stack) == 1
    assert _science(widget) == science
    assert not widget._pending_dirty_node_ids
    assert not widget._debounce_timer.isActive()

    widget.undo()
    assert widget._node_title(node.id) == automatic
    assert node.id not in widget._node_names
    assert not session.dirty
    assert _science(widget) == science
    widget.redo()
    assert widget._node_title(node.id) == "Nuclear intensity by well"
    assert _science(widget) == science

    widget.node_name_editor.reset_button.click()
    assert widget._node_title(node.id) == automatic
    assert node.id not in widget._node_names
    assert len(widget._history.undo_stack) == 2
    widget.undo()
    assert widget._node_title(node.id) == "Nuclear intensity by well"
    assert widget.pipeline.outputs[source.id] is table
    assert widget.pipeline.outputs[node.id] is result
    assert _science(widget) == science


def test_inspector_name_edit_updates_graph_results_and_search(qtbot):
    widget, source, node, _table, result, _panel = _statistics_widget(qtbot)
    dialog = widget._results_workspace.open_node(node.id)
    widget._select_node(node.id)
    editor = widget.node_name_editor
    editor.name_edit.setFocus()
    qtbot.keyClicks(editor.name_edit, "Well comparison")
    qtbot.keyClick(editor.name_edit, Qt.Key_Return)

    assert widget._node_title(node.id) == "Well comparison"
    assert widget.graph_view._cards[node.id].title_label._full_text == "Well comparison"
    assert dialog.summary_selector.currentText() == "Well comparison"
    assert "Statistics" in editor.operation_label.text()
    assert "Grouped by condition" in editor.summary_label.text()
    assert widget.pipeline.outputs[node.id] is result
    widget.graph_search_edit.setText("Well comparison")
    assert [match.node_id for match in widget._graph_search_matches] == [node.id]
    widget.graph_search_edit.setText("summarize_measurements")
    assert node.id in {match.node_id for match in widget._graph_search_matches}
    assert dialog.data_selector.currentData() == (source.id, 0)


def test_save_and_load_roundtrip_keeps_names_outside_scientific_nodes(
    qtbot, monkeypatch, tmp_path
):
    widget, source, node, _table, _result, _panel = _statistics_widget(qtbot)
    original_title = node.title
    original_params = deepcopy(node.params)
    widget._set_node_name(source.id, "All cell measurements")
    widget._set_node_name(node.id, "Comparison by well")
    path = tmp_path / "named-workflow.json"
    widget._workflow_save_policy = WorkflowSavePolicy.OVERWRITE
    monkeypatch.setattr(widget, "_choose_workflow_save_path", lambda *_: path)
    monkeypatch.setattr(widget, "_include_batch_workspace_with_workflow", lambda: False)
    assert widget._save_workflow_dialog(force_choose_path=True)
    assert not widget._workflow_tabs.current.dirty
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["metadata"]["vipp"]["node_names"] == {
        source.id: "All cell measurements",
        node.id: "Comparison by well",
    }
    saved = next(saved for saved in document["nodes"] if saved["id"] == node.id)
    assert "title" not in saved
    assert "name" not in saved
    assert saved["params"] == original_params

    widget.load_workflow_file(path)
    assert widget._node_title(source.id) == "All cell measurements"
    assert widget._node_title(node.id) == "Comparison by well"
    assert widget.pipeline.nodes[node.id].title == original_title
    assert widget.pipeline.nodes[node.id].params == original_params


def test_names_are_owned_by_workflow_even_when_node_ids_collide(qtbot):
    widget, first, second = _two_workflows(qtbot)
    assert first.summary.id == second.summary.id
    widget._set_node_name(first.summary.id, "YAP well summary")
    assert widget._activate_workflow_tab(1)
    assert widget._node_title(second.summary.id) != "YAP well summary"
    widget._set_node_name(second.summary.id, "Fascin well summary")
    widget.undo()
    assert widget._node_title(second.summary.id) != "Fascin well summary"
    widget.redo()
    assert widget._node_title(second.summary.id) == "Fascin well summary"
    assert widget._activate_workflow_tab(0)
    assert widget._node_title(first.summary.id) == "YAP well summary"
    assert widget.pipeline.outputs[first.summary.id] is first.result
    assert widget._activate_workflow_tab(1)
    assert widget._node_title(second.summary.id) == "Fascin well summary"
    assert widget.pipeline.outputs[second.summary.id] is second.result


def test_pending_name_commits_to_original_workflow_before_switching(qtbot):
    widget, first, second = _two_workflows(qtbot)
    widget._select_node(first.summary.id)
    editor = widget.node_name_editor
    qtbot.keyClicks(editor.name_edit, "First workflow only")
    assert first.summary.id not in widget._node_names

    assert widget._activate_workflow_tab(1)
    assert widget._node_title(second.summary.id) != "First workflow only"
    assert second.summary.id not in widget._node_names
    assert widget._activate_workflow_tab(0)
    assert widget._node_title(first.summary.id) == "First workflow only"
    assert widget.pipeline.outputs[first.summary.id] is first.result


def test_duplicate_delete_and_undo_preserve_names_and_distinguish_copies(qtbot):
    widget, _source, node, _table, _result, _panel = _statistics_widget(qtbot)
    widget._set_node_name(node.id, "Well summary")
    original_ids = set(widget.pipeline.nodes)
    widget._duplicate_node(node.id)
    (copy_id,) = set(widget.pipeline.nodes) - original_ids

    assert widget._node_names[copy_id] == "Well summary"
    assert widget._node_title(copy_id) != widget._node_title(node.id)
    assert widget._node_title(copy_id).startswith("Well summary")
    assert widget.pipeline.nodes[copy_id].params == node.params
    widget._delete_node(copy_id)
    assert copy_id not in widget._node_names
    widget.undo()
    assert widget._node_names[copy_id] == "Well summary"
    assert widget._node_title(copy_id) != widget._node_title(node.id)


def test_clipboard_carries_names_to_another_workflow_and_undo(qtbot):
    widget, _source, node, _table, _result, _panel = _statistics_widget(qtbot)
    widget._set_node_name(node.id, "Well summary")
    QApplication.clipboard().clear()
    widget._copy_graph_nodes((node.id,))
    widget._new_workflow()
    widget._history.clear()
    (pasted_id,) = widget._paste_graph_fragment(QPointF(350, 250))

    assert widget._node_names[pasted_id] == "Well summary"
    assert widget._node_title(pasted_id) == "Well summary"
    assert widget.pipeline.nodes[pasted_id].params == node.params
    widget.undo()
    assert pasted_id not in widget.pipeline.nodes
    assert pasted_id not in widget._node_names
    widget.redo()
    assert widget._node_title(pasted_id) == "Well summary"


def test_automatic_labels_and_summaries_follow_live_parameters(qtbot):
    widget, source, node, table, _result, panel = _statistics_widget(qtbot)
    plot, plotted = _plot(widget, source, table)
    widget._sync_node_names()
    before = widget._node_presentation(node.id)
    assert "condition" in before.name
    panel.params_changed.emit(
        {**node.params, "group_by": "image_id", "statistics": "median"}
    )

    changed = widget._node_presentation(node.id)
    assert "image_id" in changed.name
    assert "Median" in changed.summary
    assert "Mean" not in changed.summary
    widget._set_node_name(node.id, "My comparison")
    panel.params_changed.emit({**node.params, "statistics": "mean"})
    assert widget._node_title(node.id) == "My comparison"
    assert "Mean" in widget._node_presentation(node.id).summary
    plot_params = deepcopy(plot.params)
    widget._set_node_name(plot.id, "Publication panel A")
    assert plot.params == plot_params
    assert widget.pipeline.outputs[plot.id] is plotted


def test_existing_search_and_graph_name_refresh_when_statistics_settings_change(qtbot):
    widget, _source, node, _table, _result, panel = _statistics_widget(qtbot)
    widget.graph_search_edit.setText("Grouped by condition")
    assert node.id in {match.node_id for match in widget._graph_search_matches}

    panel.params_changed.emit({**node.params, "group_by": "image_id"})

    assert "image_id" in widget.graph_view._cards[node.id].title_label._full_text
    assert node.id not in {match.node_id for match in widget._graph_search_matches}


def test_save_commits_a_pending_inspector_name(qtbot, monkeypatch, tmp_path):
    widget, _source, node, _table, _result, _panel = _statistics_widget(qtbot)
    qtbot.keyClicks(widget.node_name_editor.name_edit, "Saved draft alias")
    path = tmp_path / "named-draft.json"
    widget._workflow_save_policy = WorkflowSavePolicy.OVERWRITE
    monkeypatch.setattr(widget, "_choose_workflow_save_path", lambda *_: path)
    monkeypatch.setattr(widget, "_include_batch_workspace_with_workflow", lambda: False)

    assert widget._save_workflow_dialog(force_choose_path=True)

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["metadata"]["vipp"]["node_names"][node.id] == "Saved draft alias"
    assert not widget._workflow_tabs.current.dirty


def test_paste_values_retains_destination_name_and_undo(qtbot):
    widget, _source, node, _table, _result, _panel = _statistics_widget(qtbot)
    widget._set_node_name(node.id, "Copied recipe")
    destination = widget.add_node_from_palette("summarize_measurements")
    widget._set_node_name(destination.id, "Destination analysis")
    before = deepcopy(destination.params)
    widget._copy_graph_nodes((node.id,))

    assert widget._paste_graph_node_values(destination.id)

    assert widget._node_title(destination.id) == "Destination analysis"
    assert destination.params == node.params
    widget.undo()
    assert widget._node_title(destination.id) == "Destination analysis"
    assert widget.pipeline.nodes[destination.id].params == before


def test_changed_table_source_does_not_reuse_previous_dataset_title(qtbot):
    widget, source, _node, _table, _result, _panel = _statistics_widget(qtbot)
    widget._select_node(source.id)
    old_title = widget._node_title(source.id)

    widget._on_param_changed("dataset_path", "C:/study/replacement.vipp-results.json")

    assert widget._node_title(source.id) == "replacement"
    assert widget._node_title(source.id) != old_title


def test_renaming_refreshes_an_already_open_complete_table_window(qtbot):
    widget, _source, node, _table, result, _panel = _statistics_widget(qtbot)
    widget._open_result_table_dialog()
    dialog = widget._result_table_dialog

    widget._set_node_name(node.id, "Well results")

    assert dialog.windowTitle() == "Well results — Result table"
    assert widget.pipeline.outputs[node.id] is result
