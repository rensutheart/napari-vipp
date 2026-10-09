"""Batch review navigation and bulk selection remain read-only and scoped."""

from __future__ import annotations

from dataclasses import replace
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
from qtpy.QtCore import QSignalBlocker, Qt, QUrl

from napari_vipp._tests.test_ui_batch import _actions, _preview_result
from napari_vipp._widget import VippWidget
from napari_vipp.ui.batch import CollectionBatchDialog


def _blocked_plan(tmp_path, *, count=3, blocked_position=0):
    plan = _preview_result(tmp_path, count=count)
    output_config = plan.config.outputs[0]
    config = replace(
        plan.config,
        outputs=(output_config, replace(output_config, node_id="output_2")),
    )
    items = list(plan.items)
    item = items[blocked_position]
    output = replace(item.outputs[0], duplicate=True)
    items[blocked_position] = replace(
        item,
        outputs=(output, replace(output, node_id="output_2")),
    )
    rows = tuple(
        replace(
            row,
            outputs=[output.path for output in item.outputs],
            output_statuses=tuple(output.status_text for output in item.outputs),
        )
        for row, item in zip(plan.rows, items, strict=True)
    )
    return replace(
        plan, config=config, items=tuple(items), rows=rows, collision_count=2
    )


def _dialog(qtbot, plan, *, focused=None, previewed=None):
    previewed = [] if previewed is None else previewed
    actions = _actions(plan, previewed)
    if focused is not None:
        actions = replace(actions, focus_problem_node=focused.append)
    dialog = CollectionBatchDialog(actions=actions)
    qtbot.addWidget(dialog)
    dialog.apply_preview_result(plan, preview_representative=False)
    return dialog


def test_select_and_deselect_all_include_items_on_other_pages(qtbot, tmp_path):
    full = _preview_result(tmp_path, count=120)
    plan = replace(full, rows=full.rows[:25])
    dialog = _dialog(qtbot, plan)
    assert dialog.preview_table.rowCount() == 50

    dialog.select_all_items_button.click()
    assert dialog._checked_items == set(range(120))
    assert "120 selected" in dialog.item_range_label.text()
    assert not dialog.select_all_items_button.isEnabled()
    dialog.items_next_button.click()
    assert dialog.preview_table.item(0, 0).data(Qt.UserRole) == 50
    assert all(
        dialog.preview_table.item(row, 0).checkState() == Qt.Checked
        for row in range(dialog.preview_table.rowCount())
    )

    dialog.deselect_all_items_button.click()
    assert dialog._checked_items == set()
    assert not dialog.deselect_all_items_button.isEnabled()
    assert dialog.select_all_items_button.isEnabled()
    assert dialog._preview_result is plan


def test_filtered_bulk_selection_preserves_hidden_item_checks(qtbot, tmp_path):
    plan = _preview_result(tmp_path, count=120)
    dialog = _dialog(qtbot, plan)
    dialog._checked_items = {49}
    dialog.item_search.setText("field-1")
    matching = set(dialog._matching_item_positions())
    assert 49 not in matching
    assert len(matching) > 20

    dialog.select_all_items_button.click()
    assert dialog._checked_items == matching | {49}
    assert "hidden by filters" in dialog.item_range_label.text()
    dialog.deselect_all_items_button.click()
    assert dialog._checked_items == {49}
    dialog.item_search.clear()
    assert dialog.preview_table.item(49, 0).checkState() == Qt.Checked
    assert "all pages" in dialog.select_all_items_button.toolTip().lower()
    assert "hidden" in dialog.deselect_all_items_button.toolTip().lower()


def test_empty_filter_disables_bulk_actions_without_changing_hidden_selection(
    qtbot, tmp_path
):
    dialog = _dialog(qtbot, _preview_result(tmp_path))
    before = set(dialog._checked_items)
    dialog.item_search.setText("no matching sample")
    assert not dialog.select_all_items_button.isEnabled()
    assert not dialog.deselect_all_items_button.isEnabled()
    dialog._set_matching_items_checked(False)
    assert dialog._checked_items == before


def test_bulk_selection_syncs_exact_override_sample_identities(qtbot, tmp_path):
    from napari_vipp._tests.test_ui_batch_parameter_overrides import (
        _parameter,
        _source_item,
    )
    from napari_vipp.core.batch_parameters import batch_source_item_override_key
    from napari_vipp.ui.batch_overrides import BatchOverrideSourceItem

    plan = _preview_result(tmp_path)
    sources = [_source_item(str(index)) for index in range(3)]
    keys = [batch_source_item_override_key("input", source) for source in sources]
    plan = replace(
        plan,
        items=tuple(
            replace(item, parameter_override_source_item_key=key)
            for item, key in zip(plan.items, keys, strict=True)
        ),
    )
    dialog = _dialog(qtbot, plan)
    dialog.configure_parameter_overrides(
        [
            BatchOverrideSourceItem("input", str(index), sources[index])
            for index in (2, 1, 0)
        ],
        [_parameter("threshold", "threshold", "float", 0.0, 100.0)],
        overrides=(),
    )

    dialog.select_all_items_button.click()
    assert set(dialog.parameter_override_editor.selected_source_keys()) == set(keys)
    dialog.deselect_all_items_button.click()
    assert dialog.parameter_override_editor.selected_source_keys() == ()


def test_find_problem_finds_one_duplicate_node_outside_current_filter(qtbot, tmp_path):
    plan = _blocked_plan(tmp_path, count=60, blocked_position=52)
    focused = []
    previewed = []
    dialog = _dialog(qtbot, plan, focused=focused, previewed=previewed)
    dialog.item_search.setText("0001_field-1")
    assert dialog.preview_table.rowCount() == 1
    dialog.show()

    dialog._find_output_problem()
    assert focused == ["output"]
    assert previewed == []
    assert dialog.isHidden()
    assert dialog._preview_result is plan
    assert not plan.config.output_dir.exists()


@pytest.mark.parametrize(
    "state",
    ["_checking_plan", "_run_in_progress", "_representative_pending", "_run_preparing"],
)
def test_review_actions_refuse_navigation_and_bulk_changes_while_busy(
    qtbot, tmp_path, state
):
    focused = []
    plan = _blocked_plan(tmp_path)
    dialog = _dialog(qtbot, plan, focused=focused)
    before = set(dialog._checked_items)
    setattr(dialog, state, True)
    dialog._sync_workspace()

    dialog._find_output_problem()
    dialog._focus_output_problem("output")
    dialog._set_matching_items_checked(True)
    assert focused == []
    assert dialog._checked_items == before
    assert not dialog.select_all_items_button.isEnabled()
    assert not dialog.deselect_all_items_button.isEnabled()
    assert not dialog.find_problem_button.isEnabled()


def test_stale_plan_cannot_navigate_to_historical_problem(qtbot, tmp_path):
    focused = []
    plan = _blocked_plan(tmp_path)
    dialog = _dialog(qtbot, plan, focused=focused)
    dialog._invalidate_preview_plan()

    dialog._find_output_problem()
    dialog._focus_output_problem("output")
    assert focused == []
    assert dialog._display_plan is plan
    assert dialog._preview_result is None
    assert dialog.find_problem_button.isHidden()


def test_problem_focus_accepts_only_current_blocked_output_nodes(qtbot, tmp_path):
    focused = []
    plan = _blocked_plan(tmp_path)
    dialog = _dialog(qtbot, plan, focused=focused)
    dialog._focus_output_problem("input")
    dialog._focus_output_problem("missing")
    assert focused == []

    dialog._focus_output_problem("output_2")
    assert focused == ["output_2"]
    dialog.apply_preview_result(_preview_result(tmp_path), preview_representative=False)
    dialog._focus_output_problem("output")
    assert focused == ["output_2"]


def test_problem_detail_links_address_only_current_blocked_outputs(qtbot, tmp_path):
    focused = []
    plan = _blocked_plan(tmp_path, blocked_position=1)
    dialog = _dialog(qtbot, plan, focused=focused)
    dialog.select_preview_item(1)
    assert 'href="output-node:1"' in dialog.item_details.toHtml()
    for address in ("output-node:-1", "output-node:2", "output-node:no-index"):
        dialog.item_details.anchorClicked.emit(QUrl(address))
    assert focused == []
    dialog.item_details.anchorClicked.emit(QUrl("output-node:1"))
    assert focused == ["output_2"]

    dialog.select_preview_item(0)
    dialog.item_details.anchorClicked.emit(QUrl("output-node:0"))
    assert focused == ["output_2"]
    dialog._invalidate_preview_plan()
    dialog.item_details.anchorClicked.emit(QUrl("output-node:0"))
    assert focused == ["output_2"]


def test_failed_problem_focus_keeps_batch_window_and_configuration(qtbot, tmp_path):
    plan = _blocked_plan(tmp_path)
    actions = replace(_actions(plan, []), focus_problem_node=lambda _node: False)
    dialog = CollectionBatchDialog(actions=actions)
    qtbot.addWidget(dialog)
    dialog.apply_preview_result(plan, preview_representative=False)
    before = dialog.values()
    dialog.show()

    dialog._find_output_problem()
    assert not dialog.isHidden()
    assert dialog._preview_result is plan
    assert dialog.values() == before


def _focus_host(dialog):
    calls = []
    host = SimpleNamespace(
        _active_collection_batch_dialog=dialog,
        _workflow_tabs=SimpleNamespace(current=SimpleNamespace(session_id="origin")),
        _collection_batch_running=False,
        _pending_collection_batch_start=None,
        _workflow_load_selection_in_progress=False,
        pipeline=SimpleNamespace(nodes={"output": object(), "output_2": object()}),
        _engage_collection_batch_workspace=lambda active: (
            calls.append(("engage", active))
        ),
        _commit_crop_draft=lambda **options: calls.append(("crop", options)),
        _node_title=lambda node_id: f"Batch Output ({node_id})",
        _set_status=lambda text, **options: calls.append(("status", text)),
        run_pipeline=lambda *_args, **_kwargs: (
            pytest.fail("Navigation calculated pixels")
        ),
    )
    host._workflow_tab_is_active = MethodType(VippWidget._workflow_tab_is_active, host)

    def focus(node_id):
        assert host._workflow_load_selection_in_progress
        VippWidget._restore_selected_output_for_interactive_cache(host, node_id)
        calls.append(("focus", node_id))

    host.graph_view = SimpleNamespace(focus_node=focus)
    window = SimpleNamespace(
        windowState=lambda: Qt.WindowNoState,
        show=lambda: calls.append(("window", "show")),
        raise_=lambda: calls.append(("window", "raise")),
        activateWindow=lambda: calls.append(("window", "activate")),
    )
    host.window = lambda: window
    return host, calls


def test_host_focus_suppresses_calculation_and_retains_dialog_config(qtbot, tmp_path):
    plan = _blocked_plan(tmp_path)
    dialog = _dialog(qtbot, plan)
    host, calls = _focus_host(dialog)
    before = dialog.values()
    rejected = []
    dialog.rejected.connect(lambda: rejected.append(True))
    dialog.show()

    assert VippWidget._focus_collection_batch_problem_node(host, "output", "origin")
    assert ("crop", {"schedule_run": False}) in calls
    assert ("focus", "output") in calls
    assert not host._workflow_load_selection_in_progress
    assert host._active_collection_batch_dialog is dialog
    assert not dialog.isHidden()
    assert dialog._preview_result is plan
    assert dialog.values() == before
    assert rejected == []
    assert not plan.config.output_dir.exists()


@pytest.mark.parametrize(
    "reason", ["other_workflow", "busy", "stale", "missing", "healthy"]
)
def test_host_rejects_unavailable_or_unrelated_problem(qtbot, tmp_path, reason):
    plan = _blocked_plan(tmp_path)
    dialog = _dialog(qtbot, plan)
    host, calls = _focus_host(dialog)
    node_id = "output"
    origin = "origin"
    if reason == "other_workflow":
        origin = "other"
    elif reason == "busy":
        host._pending_collection_batch_start = object()
    elif reason == "stale":
        dialog._preview_result = None
    elif reason == "missing":
        node_id = "missing"
    elif reason == "healthy":
        dialog._preview_result = _preview_result(tmp_path)

    assert not VippWidget._focus_collection_batch_problem_node(host, node_id, origin)
    assert calls == []


def test_real_host_opens_output_inspector_without_running_and_retains_workspace(
    qtbot, tmp_path, monkeypatch
):
    from napari_vipp._tests.test_widget import _Viewer
    from napari_vipp._widget import CACHE_MODE_SMART

    widget = VippWidget(_Viewer(), defer_initial_run=True)
    qtbot.addWidget(
        widget,
        before_close_func=lambda closing: setattr(
            closing, "_discard_incomplete_startup_on_close", True
        ),
    )
    runs = []
    monkeypatch.setattr(
        widget, "run_pipeline", lambda *args, **kwargs: runs.append((args, kwargs))
    )
    output = widget.pipeline.add_node("batch_output")
    assert widget.pipeline.connect("input", output.id).success
    widget._build_graph_from_pipeline()
    with QSignalBlocker(widget.cache_mode_combo):
        widget.cache_mode_combo.setCurrentText(CACHE_MODE_SMART)
    plan = _blocked_plan(tmp_path)
    plan = replace(
        plan,
        config=replace(
            plan.config,
            outputs=(replace(plan.config.outputs[0], node_id=output.id),),
        ),
        items=tuple(
            replace(
                item,
                outputs=tuple(
                    replace(planned_output, node_id=output.id)
                    for planned_output in item.outputs
                ),
            )
            for item in plan.items
        ),
    )
    dialog = widget._batch_collection_dialog(preview_config=False)
    dialog.apply_preview_result(plan, preview_representative=False)
    before = dialog.values()
    assert widget.pipeline.outputs.get(output.id) is None

    widget.showMinimized()
    dialog.find_problem_button.click()
    qtbot.wait(20)
    assert widget._selected_node_id == output.id
    assert "tag" in widget._parameter_widgets
    assert widget.selected_title.text() == widget._node_title(output.id)
    assert runs == []
    assert not widget.isMinimized()
    assert not widget._workflow_load_selection_in_progress
    assert widget._active_collection_batch_dialog is dialog
    assert dialog.isHidden()
    assert dialog.values() == before
    assert dialog._preview_result is plan
    assert not plan.config.output_dir.exists()
    assert widget._batch_collection_dialog(preview_config=False) is dialog
    assert dialog.isVisible()


def test_output_tag_edit_retains_checked_batch_settings_and_requires_recheck(
    qtbot, tmp_path, monkeypatch
):
    from napari_vipp._tests.test_widget import _Viewer

    widget = VippWidget(_Viewer(), defer_initial_run=True)
    qtbot.addWidget(
        widget,
        before_close_func=lambda closing: setattr(
            closing, "_discard_incomplete_startup_on_close", True
        ),
    )
    runs = []
    monkeypatch.setattr(
        widget, "run_pipeline", lambda *args, **kwargs: runs.append((args, kwargs))
    )
    widget.pipeline.nodes["input"].params["binding_mode"] = "collection"
    outputs = [widget.pipeline.add_node("batch_output") for _ in range(2)]
    for output in outputs:
        output.params["format"] = "npy"
        assert widget.pipeline.connect("input", output.id).success
    widget._build_graph_from_pipeline()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for index in range(2):
        np.save(inputs / f"field-{index}.npy", np.full((4, 5), index, np.uint8))
    destination = tmp_path / "outputs"
    dialog = widget._batch_collection_dialog(preview_config=False)
    dialog.input_edit.setText(str(inputs))
    dialog.output_edit.setText(str(destination))
    dialog.pattern_edit.setText("*.npy")
    dialog.format_combo.setCurrentText("npy")

    dialog.next_button.click()
    qtbot.waitUntil(lambda: not dialog._checking_plan, timeout=10000)
    assert "4 outputs share planned file paths" in dialog.preview_status.text()
    before = dialog.values()
    dialog.find_problem_button.click()
    qtbot.wait(20)
    assert dialog.isHidden() and runs == []

    tag_edit = widget._parameter_widgets["tag"].edit
    tag_edit.selectAll()
    qtbot.keyClicks(tag_edit, "processed")
    assert tag_edit.text() == "processed"
    assert widget._batch_collection_dialog(preview_config=False) is dialog
    assert dialog.isVisible()
    assert dialog.values() == before
    assert "Needs recheck" == dialog.preview_table.item(0, 4).text()
    assert dialog.find_problem_button.isHidden()
    assert not destination.exists()

    dialog.preview_button.click()
    qtbot.waitUntil(lambda: not dialog._checking_plan, timeout=10000)
    assert "4 to create" in dialog.preview_status.text()
    assert "Ready" == dialog.preview_table.item(0, 4).text()
    assert dialog.values() == before
    assert not destination.exists()
