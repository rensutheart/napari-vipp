"""Recovery rebuilds safe previews without writing files or granting authority."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from napari.components import ViewerModel
from qtpy.QtCore import QCoreApplication, QEvent, Qt
from qtpy.QtGui import QKeyEvent
from qtpy.QtWidgets import (
    QAbstractItemView,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QWidget,
)

from napari_vipp._widget import VippWidget
from napari_vipp.core.compute import ComputeMode
from napari_vipp.core.pipeline import EXECUTION_READY
from napari_vipp.core.workflow import save_workflow
from napari_vipp.core.workflow_recovery import RecoveryStore
from napari_vipp.ui import workflow_recovery as recovery_ui


@pytest.fixture
def recovery_widget(qtbot):
    viewer = ViewerModel()
    widget = VippWidget(
        viewer, defer_initial_run=True, initial_compute_mode=ComputeMode.CPU
    )
    qtbot.addWidget(widget)
    widget.pipeline.reset_empty_graph()
    widget._build_graph_from_pipeline()
    widget._sync_current_workflow_tab_state()
    yield widget
    widget._discard_incomplete_startup_on_close = True


def _edit(widget, threshold=0.63):
    node = widget.pipeline.add_node("binary_threshold")
    assert widget.pipeline.connect("input", node.id).success
    node.params["threshold"] = threshold
    widget._build_graph_from_pipeline()
    widget._sync_current_workflow_tab_state()
    return node


def _wait_for_recovery_preview(widget, qtbot):
    qtbot.waitUntil(
        lambda: (
            not widget._recovery_preview_pending
            and widget._active_source_load_id is None
            and widget._active_pipeline_run_id is None
        ),
        timeout=15_000,
    )


def test_normal_autosave_keeps_original_file_and_dirty_state(
    recovery_widget, tmp_path, qtbot
):
    widget = recovery_widget
    node = _edit(widget, 0.25)
    original = save_workflow(tmp_path / "original.json", widget.pipeline)
    before = original.read_bytes()
    widget._workflow_tabs.current.mark_saved(original)
    node.params["threshold"] = 0.63
    widget._sync_current_workflow_tab_state()
    controller = widget._workflow_recovery
    assert controller.timer.interval() == 2000
    qtbot.waitUntil(lambda: controller.last_checkpoint is not None, timeout=15_000)
    checkpoint = controller.last_checkpoint
    assert checkpoint.tabs[0]["workflow"]["nodes"][-1]["params"]["threshold"] == 0.63
    assert widget._workflow_tabs.current.dirty
    assert original.read_bytes() == before
    assert not widget.pipeline.completed_node_ids
    assert controller.store.recoverable() == []  # Never recover a live owner.


def test_recovered_copy_retains_graph_and_calculates_only_safe_automatic_nodes(
    recovery_widget, qtbot, tmp_path
):
    widget = recovery_widget
    node = _edit(widget)
    widget.pipeline.nodes["input"].params.update(
        source_mode="sample", sample_name="VIPP synthetic volume"
    )
    manual = widget.pipeline.add_node("intensity_histogram")
    widget.pipeline.set_node_auto_recalculate(manual.id, True)
    assert widget.pipeline.connect(node.id, manual.id).success
    downstream = widget.pipeline.add_node("select_table_columns")
    assert widget.pipeline.connect(manual.id, downstream.id).success
    writer = widget.pipeline.add_node("save_output")
    output_path = tmp_path / "must-not-be-published.npy"
    writer.params.update(enabled="on", path=str(output_path))
    assert widget.pipeline.connect(node.id, writer.id).success
    widget._build_graph_from_pipeline()
    widget._sync_current_workflow_tab_state()
    positions = widget.graph_view.node_positions()
    original = save_workflow(tmp_path / "source.json", widget.pipeline)
    widget._workflow_tabs.current.mark_saved(original)
    before = original.read_bytes()
    checkpoint = widget._workflow_recovery.checkpoint_now(required=True)
    ids = widget._restore_recovery_workspace(checkpoint)
    assert len(ids) == 1
    _wait_for_recovery_preview(widget, qtbot)
    recovered = widget._workflow_tabs.current
    assert recovered.title.startswith("Recovered") and recovered.path is None
    assert recovered.dirty and recovered.session_id == ids[0]
    assert node.id in widget.pipeline.nodes
    assert widget.pipeline.nodes[node.id].params["threshold"] == 0.63
    assert widget.graph_view.node_positions() == positions
    source_data = widget._sample_payloads()["VIPP synthetic volume"].data
    np.testing.assert_array_equal(widget.pipeline.outputs["input"], source_data)
    np.testing.assert_array_equal(widget.pipeline.outputs[node.id], source_data > 0.63)
    assert widget.pipeline.node_execution_states[node.id] == EXECUTION_READY
    assert {"input", node.id} <= widget.pipeline.completed_node_ids
    for paused in (manual.id, downstream.id, writer.id):
        assert paused not in widget.pipeline.completed_node_ids
        assert widget.pipeline.outputs.get(paused) is None
        assert widget.pipeline.node_execution_states[paused] != EXECUTION_READY
    assert widget.pipeline.nodes[manual.id].params["_vipp_auto_recalculate"] is True
    assert widget.pipeline.nodes[writer.id].params["enabled"] == "on"
    assert not output_path.exists()
    assert not recovered.history.can_undo
    assert original.read_bytes() == before


def test_failed_preedit_checkpoint_aborts_without_graph_changes(
    recovery_widget, monkeypatch
):
    widget = recovery_widget
    _edit(widget)
    controller = widget._workflow_recovery
    before = widget._capture_recovery_workspace()

    def denied(*_, **__):
        raise PermissionError("Synthetic disk failure")

    monkeypatch.setattr(controller.store, "checkpoint", denied)
    with pytest.raises(ValueError, match="checkpoint failed"):
        controller.checkpoint_now("before AI patch", required=True)
    assert widget._capture_recovery_workspace() == before


def test_recovery_base_folder_is_not_changed_to_checkpoint_directory(
    recovery_widget, tmp_path
):
    widget = recovery_widget
    tab = widget._capture_recovery_workspace()[0][0]
    assert Path(tab["base_dir"]).is_absolute()
    assert Path(tab["base_dir"]) != Path(widget._workflow_recovery.store.root)


def test_relative_image_binding_survives_changed_working_directory(
    recovery_widget, tmp_path, monkeypatch, qtbot
):
    widget = recovery_widget
    original = tmp_path / "image-folder"
    original.mkdir()
    monkeypatch.chdir(original)
    widget.pipeline.nodes["input"].params.update(
        source_mode="file path", file_path="image.ome.tif"
    )
    checkpoint = widget._workflow_recovery.checkpoint_now(required=True)
    monkeypatch.chdir(tmp_path)
    widget._restore_recovery_workspace(checkpoint)
    _wait_for_recovery_preview(widget, qtbot)
    assert widget.pipeline.nodes["input"].params["file_path"] == str(
        original / "image.ome.tif"
    )
    assert not widget.pipeline.completed_node_ids  # The source file does not exist.
    assert widget._last_pipeline_source_signature is None


def test_opened_recovery_not_duplicated_when_bookkeeping_fails(
    recovery_widget, tmp_path, monkeypatch
):
    widget = recovery_widget
    controller = widget._workflow_recovery
    tabs, active = widget._capture_recovery_workspace()
    crashed = RecoveryStore(controller.store.root)
    crashed.checkpoint(tabs, active)
    crashed.close(clean=False)

    def denied(*_, **__):
        raise PermissionError("Synthetic disk full after opening recovered tabs")

    monkeypatch.setattr(controller, "checkpoint_now", denied)
    controller.show_recovery()
    dialog = controller._dialog
    from qtpy.QtWidgets import QPushButton

    button = next(
        b
        for b in dialog.findChildren(QPushButton)
        if b.text().startswith("Recover selected")
    )
    before = len(widget._workflow_tabs)
    choices = dialog.findChild(QListWidget, "recoverySessions")
    choices.item(0).setCheckState(Qt.Checked)
    button.click()
    assert len(widget._workflow_tabs) == before + 1
    assert "Recovered tabs are open" in widget.status_label.text()
    assert controller._dialog is None
    controller.show_recovery()
    assert controller._dialog is None
    assert len(widget._workflow_tabs) == before + 1
    assert len(controller.store.recoverable()) == 1  # Old copy retained, not erased.


def test_folder_change_requires_durable_new_copy(
    recovery_widget, tmp_path, monkeypatch
):
    import napari_vipp.ui.workflow_recovery as ui

    widget = recovery_widget
    controller = widget._workflow_recovery
    original = controller.store
    selected = tmp_path / "new-recovery-folder"
    selected.mkdir()
    monkeypatch.setattr(
        ui.QFileDialog, "getExistingDirectory", lambda *a: str(selected)
    )
    real_checkpoint = RecoveryStore.checkpoint

    def checkpoint(store, *args, **kwargs):
        if store.root == selected:
            raise PermissionError("Synthetic unavailable destination")
        return real_checkpoint(store, *args, **kwargs)

    monkeypatch.setattr(RecoveryStore, "checkpoint", checkpoint)
    controller.choose_folder()
    assert controller.store is original
    assert controller.last_checkpoint is not None
    assert "current folder is kept" in widget.status_label.text()
    assert original.checkpoint(*widget._capture_recovery_workspace())


@pytest.mark.parametrize("legacy_relative_copy", [False, True])
def test_portable_recipe_recovery_keeps_original_bindings_and_source_proofs(
    recovery_widget, tmp_path, monkeypatch, legacy_relative_copy, qtbot
):
    """A new unsaved recovered tab must not use the recovery folder as its base."""
    import hashlib
    import shutil

    from napari_vipp._tests.test_workflow_paths import _project
    from napari_vipp.core.workflow import load_workflow
    from napari_vipp.core.workflow_paths import transform_workflow_paths

    pipeline, original, _item = _project(tmp_path / "original-project")
    save_workflow(original, pipeline)
    moved = tmp_path / "relocated-project"
    shutil.copytree(original.parent.parent, moved)
    saved_path = moved / "workflows" / original.name
    before_file = saved_path.read_bytes()
    source = moved / "data" / "cells.npy"
    dataset = moved / "data" / "measurements.json"
    before_source, before_dataset = source.read_bytes(), dataset.read_bytes()
    unrelated = tmp_path / "unrelated-working-directory"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    loaded = load_workflow(saved_path)
    widget = recovery_widget
    widget.pipeline.restore_graph(
        loaded["nodes"], loaded["connections"], loaded["output_tunnels"]
    )
    widget._build_graph_from_pipeline()
    widget._sync_current_workflow_tab_state()
    widget._workflow_tabs.current.mark_saved(saved_path)
    controller = widget._workflow_recovery
    if legacy_relative_copy:
        # Working copies predating portable-path integration may retain
        # relative bindings together with their original containing folder.
        tabs, active = widget._capture_recovery_workspace()
        tabs[0]["workflow"] = transform_workflow_paths(
            tabs[0]["workflow"], saved_path.parent, relative=True
        )
        checkpoint = controller.store.checkpoint(tabs, active)
    else:
        checkpoint = controller.checkpoint_now(required=True)
    ids = widget._restore_recovery_workspace(checkpoint)
    assert len(ids) == 1
    _wait_for_recovery_preview(widget, qtbot)
    params = widget.pipeline.nodes["input"].params
    assert Path(params["file_path"]) == source
    proof = params["_vipp_source_item"]["container"]
    assert Path(proof["uri"]) == source
    expected_proof = loaded["nodes"][0].params["_vipp_source_item"]["container"]
    assert proof == expected_proof
    table = next(
        node
        for node in widget.pipeline.nodes.values()
        if node.operation_id == "table_source"
    )
    output = next(
        node
        for node in widget.pipeline.nodes.values()
        if node.operation_id == "save_output"
    )
    assert Path(table.params["dataset_path"]) == dataset
    assert table.params["dataset_sha256"] == hashlib.sha256(before_dataset).hexdigest()
    assert Path(output.params["path"]) == moved / "results" / "mask.tif"
    assert output.params["enabled"] == "off"
    assert widget._workflow_tabs.current.path is None
    assert widget._workflow_tabs.current.dirty
    # This provenance-only fixture deliberately contains non-NPY marker bytes.
    # Automatic preview must reject those bytes rather than invent a result.
    assert not widget.pipeline.completed_node_ids
    assert widget._last_pipeline_source_signature is None
    assert output.id not in widget.pipeline.completed_node_ids
    assert widget.pipeline.outputs.get(output.id) is None
    assert saved_path.read_bytes() == before_file
    assert source.read_bytes() == before_source
    assert dataset.read_bytes() == before_dataset
    assert not (moved / "results").exists()


class _RecoveryDialogStore:
    issues = ()

    def __init__(self, checkpoints, events, durable):
        self.pending = list(checkpoints)
        self.events, self.durable = events, durable
        self.dismiss_failures = set()
        self.mark_failures = set()
        self.on_dismiss = None

    def recoverable(self):
        return list(self.pending)

    def mark_recovered(self, checkpoint_id):
        assert checkpoint_id in self.durable
        self.events.append(("mark", checkpoint_id))
        if checkpoint_id in self.mark_failures:
            raise PermissionError("injected mark failure")
        self.pending = [p for p in self.pending if p.checkpoint_id != checkpoint_id]

    def dismiss(self, checkpoint_id):
        self.events.append(("dismiss", checkpoint_id))
        if self.on_dismiss:
            self.on_dismiss()
        if checkpoint_id in self.dismiss_failures:
            raise PermissionError("injected dismissal failure")
        self.pending = [p for p in self.pending if p.checkpoint_id != checkpoint_id]

    def close(self, **kwargs):
        pass


@pytest.fixture
def recovery_dialog(qtbot, tmp_path, monkeypatch):
    """Controller/UI only: no recovery directory is read or written."""
    checkpoints = [
        SimpleNamespace(
            checkpoint_id=f"checkpoint-{index}",
            tabs=[{"title": f"Synthetic session {index}"}],
            created_utc=f"2026-10-25T12:00:0{index}Z",
            fallback_used=False,
        )
        for index in range(3)
    ]
    events, durable, messages = [], set(), []
    store = _RecoveryDialogStore(checkpoints, events, durable)
    monkeypatch.setattr(recovery_ui, "RecoveryStore", lambda _: store)
    monkeypatch.setattr(recovery_ui, "recovery_root", lambda: tmp_path / "unused-store")
    parent = QWidget()
    qtbot.addWidget(parent)
    state = SimpleNamespace(
        restore_failures=set(), checkpoint_failures=set(), last=None, on_restore=None
    )

    def restore(checkpoint):
        identifier = checkpoint.checkpoint_id
        events.append(("restore", identifier))
        if state.on_restore:
            state.on_restore()
        if identifier in state.restore_failures:
            raise ValueError("injected restore failure")
        state.last = identifier

    controller = recovery_ui.WorkflowRecoveryController(
        parent, capture=lambda: ([], ""), restore=restore, announce=messages.append
    )
    controller.periodic.stop()

    def checkpoint(reason, *, required=False):
        assert required and reason == "recovered workspace"
        assert state.last in controller._restored
        events.append(("checkpoint", state.last))
        if state.last in state.checkpoint_failures:
            raise PermissionError("injected durable checkpoint failure")
        durable.add(state.last)
        return SimpleNamespace(created_utc="durable synthetic checkpoint")

    monkeypatch.setattr(controller, "checkpoint_now", checkpoint)
    monkeypatch.setattr(controller, "changed", lambda: None)
    controller.show_recovery()
    yield SimpleNamespace(
        controller=controller,
        store=store,
        events=events,
        durable=durable,
        messages=messages,
        checkpoints=checkpoints,
        state=state,
    )
    controller.close()


def _dialog_controls(case):
    dialog = case.controller._dialog
    assert dialog is not None
    return SimpleNamespace(
        dialog=dialog,
        choices=dialog.findChild(QListWidget, "recoverySessions"),
        recover=dialog.findChild(QPushButton, "recoverSelected"),
        discard=dialog.findChild(QPushButton, "discardSelectedRecovery"),
        all=dialog.findChild(QPushButton, "recoverySelectAll"),
        clear=dialog.findChild(QPushButton, "recoveryClearSelection"),
        summary=dialog.findChild(QLabel, "recoverySelectionSummary"),
    )


def _check_rows(choices, rows):
    for row in range(choices.count()):
        choices.item(row).setCheckState(Qt.Unchecked)
    for row in rows:
        choices.item(row).setCheckState(Qt.Checked)


def _checked_rows(choices):
    return [
        row
        for row in range(choices.count())
        if choices.item(row).checkState() == Qt.Checked
    ]


def _check_indicator_rect(choices, row):
    return choices.check_indicator_rect(choices.model().index(row, 0))


def test_recovery_selection_counts_select_all_clear_and_no_selection(recovery_dialog):
    case = recovery_dialog
    controls = _dialog_controls(case)
    assert controls.choices.selectionMode() == QAbstractItemView.NoSelection
    assert all(
        controls.choices.item(row).flags() & Qt.ItemIsUserCheckable
        for row in range(controls.choices.count())
    )
    for row, checkpoint in enumerate(case.checkpoints):
        item = controls.choices.item(row)
        assert checkpoint.tabs[0]["title"] in item.text()
        assert "1 workflow tab" in item.text()
        assert checkpoint.created_utc in item.toolTip()
    assert "Space" in controls.choices.accessibleDescription()
    assert _checked_rows(controls.choices) == []
    assert controls.summary.text() == "0 of 3 sessions selected"
    assert not controls.recover.isEnabled() and not controls.discard.isEnabled()
    _check_rows(controls.choices, [0])
    assert controls.summary.text() == "1 of 3 sessions selected"
    assert "(1)" in controls.recover.text() and "(1)" in controls.discard.text()
    controls.all.click()
    assert _checked_rows(controls.choices) == [0, 1, 2]
    assert controls.summary.text() == "3 of 3 sessions selected"
    assert "(3)" in controls.recover.text() and "(3)" in controls.discard.text()
    controls.clear.click()
    assert _checked_rows(controls.choices) == []
    assert controls.summary.text() == "0 of 3 sessions selected"
    assert controls.recover.text() == "Recover selected (0)"
    assert controls.discard.text() == "Discard selected (0)…"
    assert not controls.recover.isEnabled() and not controls.discard.isEnabled()
    controls.recover.click()
    controls.discard.click()
    assert case.events == []
    assert case.controller._dialog is controls.dialog


@pytest.mark.parametrize("direction", [Qt.LeftToRight, Qt.RightToLeft])
def test_recovery_checkbox_click_adds_but_body_click_selects_only_that_session(
    qtbot, recovery_dialog, direction
):
    controls = _dialog_controls(recovery_dialog)
    choices = controls.choices
    choices.setLayoutDirection(direction)
    qtbot.waitUntil(lambda: choices.visualItemRect(choices.item(2)).isValid())
    first_indicator = _check_indicator_rect(choices, 0)
    third_indicator = _check_indicator_rect(choices, 2)
    assert first_indicator.isValid() and third_indicator.isValid()
    qtbot.mouseClick(
        choices.viewport(),
        Qt.LeftButton,
        pos=first_indicator.center(),
    )
    qtbot.mouseClick(
        choices.viewport(),
        Qt.LeftButton,
        pos=third_indicator.center(),
    )
    assert _checked_rows(choices) == [0, 2]
    assert controls.summary.text() == "2 of 3 sessions selected"
    qtbot.mouseClick(
        choices.viewport(),
        Qt.LeftButton,
        pos=first_indicator.center(),
    )
    assert _checked_rows(choices) == [2]
    body = choices.visualItemRect(choices.item(1)).center()
    assert not _check_indicator_rect(choices, 1).contains(body)
    qtbot.mouseClick(
        choices.viewport(),
        Qt.LeftButton,
        pos=body,
    )
    assert _checked_rows(choices) == [1]
    assert choices.currentRow() == 1
    assert controls.summary.text() == "1 of 3 sessions selected"
    qtbot.mouseClick(choices.viewport(), Qt.LeftButton, pos=body)
    assert _checked_rows(choices) == [1]


@pytest.mark.parametrize(
    ("press_row", "press_checkbox", "release_row", "release_checkbox"),
    [
        (0, True, 0, False),
        (0, False, 0, True),
        (0, True, 1, True),
        (0, False, 1, False),
    ],
)
def test_recovery_press_release_must_match_row_and_checkbox_region(
    qtbot, recovery_dialog, press_row, press_checkbox, release_row, release_checkbox
):
    controls = _dialog_controls(recovery_dialog)
    choices = controls.choices
    _check_rows(choices, [2])

    def position(row, checkbox):
        rect = (
            _check_indicator_rect(choices, row)
            if checkbox
            else choices.visualItemRect(choices.item(row))
        )
        assert rect.isValid()
        return rect.center()

    qtbot.mousePress(
        choices.viewport(), Qt.LeftButton, pos=position(press_row, press_checkbox)
    )
    qtbot.mouseRelease(
        choices.viewport(), Qt.LeftButton, pos=position(release_row, release_checkbox)
    )
    assert _checked_rows(choices) == [2]
    assert recovery_dialog.events == []


def test_recovery_double_click_and_return_never_start_recovery(qtbot, recovery_dialog):
    controls = _dialog_controls(recovery_dialog)
    choices = controls.choices
    qtbot.mouseDClick(
        choices.viewport(),
        Qt.LeftButton,
        pos=choices.visualItemRect(choices.item(0)).center(),
    )
    qtbot.keyClick(choices, Qt.Key_Return)
    assert recovery_dialog.events == []
    assert recovery_dialog.controller._dialog is controls.dialog


@pytest.mark.parametrize("removed_row", [0, 2])
def test_recovery_removed_pressed_row_cannot_check_another_row_on_release(
    qtbot, recovery_dialog, removed_row
):
    controls = _dialog_controls(recovery_dialog)
    choices = controls.choices
    _check_rows(choices, [1])
    position = _check_indicator_rect(choices, removed_row).center()
    qtbot.mousePress(choices.viewport(), Qt.LeftButton, pos=position)
    removed = choices.takeItem(removed_row)
    assert removed is not None
    # Removing the last row makes the release index invalid too; removing the
    # first puts a different, still checked session under the release position.
    if removed_row == 2:
        assert not choices.indexAt(position).isValid()
    qtbot.mouseRelease(choices.viewport(), Qt.LeftButton, pos=position)
    assert [choices.item(row).data(Qt.UserRole) for row in _checked_rows(choices)] == [
        1
    ]
    assert recovery_dialog.events == []


def test_recovery_keyboard_space_toggles_and_arrow_focus_preserves_other_checks(
    qtbot, recovery_dialog
):
    controls = _dialog_controls(recovery_dialog)
    choices = controls.choices
    choices.setCurrentRow(0)
    choices.setFocus()
    assert _checked_rows(choices) == []
    qtbot.keyClick(choices, Qt.Key_Space)
    assert _checked_rows(choices) == [0]
    qtbot.keyClick(choices, Qt.Key_Down)
    assert choices.currentRow() == 1
    assert _checked_rows(choices) == [0]
    qtbot.keyClick(choices, Qt.Key_Space)
    assert _checked_rows(choices) == [0, 1]
    qtbot.keyClick(choices, Qt.Key_Up)
    assert choices.currentRow() == 0
    assert _checked_rows(choices) == [0, 1]
    qtbot.keyClick(choices, Qt.Key_Space)
    assert _checked_rows(choices) == [1]
    assert controls.summary.text() == "1 of 3 sessions selected"


def test_recovery_held_space_does_not_repeat_checkbox_toggles(recovery_dialog, qtbot):
    controls = _dialog_controls(recovery_dialog)
    choices = controls.choices
    _check_rows(choices, [1])
    choices.setCurrentRow(0)

    def repeated_space():
        event = QKeyEvent(QEvent.KeyPress, Qt.Key_Space, Qt.NoModifier, " ", True, 1)
        assert event.isAutoRepeat()
        QCoreApplication.sendEvent(choices, event)

    repeated_space()
    assert _checked_rows(choices) == [1]
    qtbot.keyClick(choices, Qt.Key_Space)
    assert _checked_rows(choices) == [0, 1]
    repeated_space()
    assert _checked_rows(choices) == [0, 1]
    assert recovery_dialog.events == []


@pytest.mark.parametrize("rows", [[1], [2, 0]])
def test_selected_recovery_uses_display_order_and_durable_per_item_boundaries(
    recovery_dialog, rows
):
    case = recovery_dialog
    controls = _dialog_controls(case)
    _check_rows(controls.choices, rows)
    controls.recover.click()
    expected = [case.checkpoints[index].checkpoint_id for index in sorted(rows)]
    assert case.events == [
        (kind, identifier)
        for identifier in expected
        for kind in ("restore", "checkpoint", "mark")
    ]
    assert case.controller._restored == set(expected)
    assert case.durable == set(expected)
    assert controls.choices.count() == 3 - len(rows)
    assert _checked_rows(controls.choices) == []
    assert not controls.recover.isEnabled() and not controls.discard.isEnabled()
    assert case.controller._dialog is controls.dialog


def test_recover_all_closes_only_after_every_selected_session_is_protected(
    recovery_dialog,
):
    case = recovery_dialog
    controls = _dialog_controls(case)
    controls.all.click()
    controls.recover.click()
    assert case.controller._dialog is None
    assert case.durable == {p.checkpoint_id for p in case.checkpoints}
    assert case.store.pending == []


def test_partial_restore_failure_retries_only_failed_rows(
    recovery_dialog,
):
    case = recovery_dialog
    controls = _dialog_controls(case)
    failed = case.checkpoints[1].checkpoint_id
    case.state.restore_failures.add(failed)
    controls.all.click()
    controls.recover.click()
    assert case.events == [
        ("restore", "checkpoint-0"),
        ("checkpoint", "checkpoint-0"),
        ("mark", "checkpoint-0"),
        ("restore", failed),
        ("restore", "checkpoint-2"),
        ("checkpoint", "checkpoint-2"),
        ("mark", "checkpoint-2"),
    ]
    assert controls.choices.count() == 1 and _checked_rows(controls.choices) == [0]
    assert case.controller._restored == {"checkpoint-0", "checkpoint-2"}
    case.state.restore_failures.clear()
    controls.recover.click()
    assert case.controller._dialog is None
    assert [identifier for kind, identifier in case.events if kind == "restore"] == [
        "checkpoint-0",
        failed,
        "checkpoint-2",
        failed,
    ]


@pytest.mark.parametrize("failure", ["checkpoint", "mark"])
def test_opened_batch_items_are_never_restored_twice_after_bookkeeping_failure(
    recovery_dialog, failure
):
    case = recovery_dialog
    controls = _dialog_controls(case)
    failed = case.checkpoints[0].checkpoint_id
    failures = (
        case.state.checkpoint_failures
        if failure == "checkpoint"
        else case.store.mark_failures
    )
    failures.add(failed)
    _check_rows(controls.choices, [0, 1])
    controls.recover.click()
    assert case.controller._restored == {"checkpoint-0", "checkpoint-1"}
    assert controls.choices.count() == 1 and _checked_rows(controls.choices) == []
    assert failed in {p.checkpoint_id for p in case.store.pending}
    assert any("Recovered tabs are open" in message for message in case.messages)
    if failure == "checkpoint":
        assert ("mark", failed) not in case.events
    controls.dialog.reject()
    case.controller.show_recovery()
    reopened = _dialog_controls(case)
    assert reopened.choices.count() == 1
    assert "Synthetic session 2" in reopened.choices.item(0).text()
    assert [identifier for kind, identifier in case.events if kind == "restore"] == [
        "checkpoint-0",
        "checkpoint-1",
    ]


def test_recovery_actions_are_disabled_and_reentrant_clicks_do_not_duplicate(
    recovery_dialog,
):
    case = recovery_dialog
    controls = _dialog_controls(case)
    _check_rows(controls.choices, [0])

    def during_restore():
        assert not controls.choices.isEnabled()
        for button in controls.dialog.findChildren(QPushButton):
            assert not button.isEnabled()
        controls.recover.clicked.emit(False)
        controls.discard.clicked.emit(False)

    case.state.on_restore = during_restore
    controls.recover.click()
    assert case.events == [
        ("restore", "checkpoint-0"),
        ("checkpoint", "checkpoint-0"),
        ("mark", "checkpoint-0"),
    ]


@pytest.mark.parametrize("approve", [False, True])
def test_discard_selection_asks_once_with_count_and_defaults_to_no(
    recovery_dialog, monkeypatch, approve
):
    case = recovery_dialog
    controls = _dialog_controls(case)
    _check_rows(controls.choices, [0, 2])
    questions = []

    def question(*args, **kwargs):
        questions.append((args, kwargs))
        return QMessageBox.Yes if approve else QMessageBox.No

    monkeypatch.setattr(recovery_ui.QMessageBox, "question", question)
    controls.discard.click()
    assert len(questions) == 1
    args, kwargs = questions[0]
    assert "2" in args[2]
    assert len(args[2].splitlines()) <= 3
    assert len(args[2]) < 300
    assert (args[3] if len(args) > 3 else kwargs["buttons"]) == (
        QMessageBox.Yes | QMessageBox.No
    )
    assert (
        kwargs.get("defaultButton", args[4] if len(args) > 4 else None)
        == QMessageBox.No
    )
    assert case.events == (
        [("dismiss", "checkpoint-0"), ("dismiss", "checkpoint-2")] if approve else []
    )
    assert controls.choices.count() == (1 if approve else 3)
    assert case.controller._restored == set()


def test_partial_discard_failure_keeps_failure_and_unselected_rows_for_retry(
    recovery_dialog, monkeypatch
):
    case = recovery_dialog
    controls = _dialog_controls(case)
    _check_rows(controls.choices, [0, 1])
    case.store.dismiss_failures.add("checkpoint-1")
    monkeypatch.setattr(
        recovery_ui.QMessageBox, "question", lambda *args: QMessageBox.Yes
    )
    controls.discard.click()
    assert case.events == [("dismiss", "checkpoint-0"), ("dismiss", "checkpoint-1")]
    assert controls.choices.count() == 2 and _checked_rows(controls.choices) == [0]
    assert "Synthetic session 1" in controls.choices.item(0).text()
    assert "Synthetic session 2" in controls.choices.item(1).text()
    case.store.dismiss_failures.clear()
    controls.discard.click()
    assert case.events[-1] == ("dismiss", "checkpoint-1")
    assert controls.choices.count() == 1 and _checked_rows(controls.choices) == []


def test_discard_batch_blocks_reentrancy_and_reenables_remaining_controls(
    recovery_dialog, monkeypatch
):
    case = recovery_dialog
    controls = _dialog_controls(case)
    _check_rows(controls.choices, [0, 1])
    questions = []

    def question(*args):
        questions.append(args)
        assert not controls.choices.isEnabled()
        assert all(
            not button.isEnabled()
            for button in controls.dialog.findChildren(QPushButton)
        )
        controls.discard.clicked.emit(False)
        return QMessageBox.Yes

    def during_dismiss():
        assert not controls.choices.isEnabled()
        controls.recover.clicked.emit(False)
        controls.discard.clicked.emit(False)

    case.store.on_dismiss = during_dismiss
    monkeypatch.setattr(recovery_ui.QMessageBox, "question", question)
    controls.discard.click()
    assert len(questions) == 1
    assert case.events == [("dismiss", "checkpoint-0"), ("dismiss", "checkpoint-1")]
    assert controls.choices.count() == 1
    assert controls.choices.isEnabled() and controls.all.isEnabled()
    assert _checked_rows(controls.choices) == []


def test_no_recovery_dialog_is_created_when_no_interrupted_sessions_remain(
    recovery_dialog,
):
    case = recovery_dialog
    _dialog_controls(case).dialog.reject()
    case.store.pending.clear()
    case.controller.show_recovery()
    assert case.controller._dialog is None
    assert case.events == []
    assert (
        case.messages[-1] == "No interrupted VIPP workflows are waiting for recovery."
    )


def test_two_real_recovery_copies_open_unsaved_with_safe_independent_previews(
    recovery_widget, qtbot, tmp_path, monkeypatch
):
    widget = recovery_widget
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    node = _edit(widget)
    manual = widget.pipeline.add_node("intensity_histogram")
    widget.pipeline.set_node_auto_recalculate(manual.id, True)
    assert widget.pipeline.connect(node.id, manual.id).success
    writer = widget.pipeline.add_node("save_output")
    assert widget.pipeline.connect(node.id, writer.id).success
    controller = widget._workflow_recovery
    original_bytes, source_bytes, expected = {}, {}, {}
    checkpoint_ids = set()
    for index, threshold in enumerate((0.25, 0.75)):
        source = tmp_path / "data" / f"source-{index}.npy"
        data = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8) + index
        np.save(source, data)
        source_bytes[source] = source.read_bytes()
        widget.pipeline.nodes["input"].params.update(
            source_mode="file path", file_path=f"data/{source.name}"
        )
        widget.pipeline.nodes[node.id].params["threshold"] = threshold
        destination = tmp_path / f"must-not-write-{index}.npy"
        widget.pipeline.nodes[writer.id].params.update(
            enabled="on", path=str(destination)
        )
        widget._build_graph_from_pipeline()
        widget._sync_current_workflow_tab_state()
        original = save_workflow(tmp_path / f"workflow-{index}.json", widget.pipeline)
        widget._workflow_tabs.current.mark_saved(original)
        original_bytes[original] = original.read_bytes()
        tabs, active = widget._capture_recovery_workspace()
        assert len(tabs) == 1
        companion = deepcopy(tabs[0])
        companion["tab_id"] = f"companion-tab-{index}"
        companion["title"] = f"Companion workflow {index}"
        tabs.append(companion)
        crashed = RecoveryStore(controller.store.root)
        checkpoint = crashed.checkpoint(tabs, active)
        crashed.close(clean=False)
        checkpoint_ids.add(checkpoint.checkpoint_id)
        expected[str(original)] = (source, data, threshold, destination)
    original_session = widget._workflow_tabs.current
    controller.show_recovery()
    dialog = controller._dialog
    assert dialog is not None
    choices = dialog.findChild(QListWidget, "recoverySessions")
    assert choices.count() == 2
    dialog.findChild(QPushButton, "recoverySelectAll").click()
    dialog.findChild(QPushButton, "recoverSelected").click()
    qtbot.waitUntil(lambda: controller._dialog is None, timeout=15_000)
    assert len(widget._workflow_tabs) == 5
    assert checkpoint_ids <= controller._restored
    assert controller.store.recoverable() == []
    _wait_for_recovery_preview(widget, qtbot)
    sessions = [
        session for session in widget._workflow_tabs if session is not original_session
    ]
    assert len(sessions) == 4
    for original in expected:
        assert (
            sum(
                session.runtime_cache["_recovery_original_path"] == original
                for session in sessions
            )
            == 2
        )
    for session in sessions:
        assert widget._activate_workflow_tab(
            widget._workflow_tabs.index_of(session.session_id), check_safety=True
        )
        _wait_for_recovery_preview(widget, qtbot)
        source, data, threshold, destination = expected[
            session.runtime_cache["_recovery_original_path"]
        ]
        assert (
            session.path is None
            and session.dirty
            and session.title.startswith("Recovered")
        )
        assert Path(widget.pipeline.nodes["input"].params["file_path"]) == source
        np.testing.assert_array_equal(widget.pipeline.outputs["input"], data)
        np.testing.assert_array_equal(
            widget.pipeline.outputs[node.id], data > threshold
        )
        assert not session.history.can_undo
        for paused in (manual.id, writer.id):
            assert paused not in widget.pipeline.completed_node_ids
            assert widget.pipeline.outputs.get(paused) is None
            assert widget.pipeline.node_execution_states[paused] != EXECUTION_READY
        assert widget.pipeline.nodes[manual.id].params["_vipp_auto_recalculate"] is True
        assert widget.pipeline.nodes[writer.id].params["enabled"] == "on"
        assert not destination.exists()
    assert all(path.read_bytes() == raw for path, raw in original_bytes.items())
    assert all(path.read_bytes() == raw for path, raw in source_bytes.items())


def test_new_relative_recipe_bindings_are_frozen_before_cwd_changes(
    recovery_widget, tmp_path, monkeypatch, qtbot
):
    widget = recovery_widget
    authored = tmp_path / "authoring-folder"
    authored.mkdir()
    monkeypatch.chdir(authored)
    widget.pipeline.nodes["input"].params.update(
        source_mode="file path", file_path="input.ome.tif"
    )
    table = widget.pipeline.add_node("table_source")
    table.params.update(dataset_path="measurements.json", dataset_sha256="a" * 64)
    output = widget.pipeline.add_node("save_output")
    output.params["path"] = "results/mask.tif"
    widget._build_graph_from_pipeline()
    widget._sync_current_workflow_tab_state()
    checkpoint = widget._workflow_recovery.checkpoint_now(required=True)
    monkeypatch.chdir(tmp_path)
    widget._restore_recovery_workspace(checkpoint)
    _wait_for_recovery_preview(widget, qtbot)
    assert Path(widget.pipeline.nodes["input"].params["file_path"]) == (
        authored / "input.ome.tif"
    )
    assert Path(widget.pipeline.nodes[table.id].params["dataset_path"]) == (
        authored / "measurements.json"
    )
    assert widget.pipeline.nodes[table.id].params["dataset_sha256"] == "a" * 64
    assert Path(widget.pipeline.nodes[output.id].params["path"]) == (
        authored / "results" / "mask.tif"
    )
    assert not widget.pipeline.completed_node_ids  # No source files were created.
    assert widget._last_pipeline_source_signature is None
