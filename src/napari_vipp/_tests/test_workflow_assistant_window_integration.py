"""Application seams for the assistant workspace; no inference or calculations."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from napari.components import ViewerModel
from qtpy.QtWidgets import QTabWidget

from napari_vipp import _widget
from napari_vipp._widget import VippWidget
from napari_vipp.core.compute import ComputeMode
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.core.workflow_proposal_outputs import (
    OUTPUT_DIRECTORY_PLACEHOLDER,
    OutputDestination,
)
from napari_vipp.core.workflow_proposals import (
    ProposalError,
    reference_proposal,
    validate_proposal,
)


@pytest.fixture
def editor(qtbot, monkeypatch):
    from napari_vipp.core import workflow_proposal_codex

    monkeypatch.setattr(
        workflow_proposal_codex,
        "discover_codex_runtime",
        lambda: SimpleNamespace(
            supported=True, authenticated=False, detail="Sign in to Codex."
        ),
    )
    model_request = Mock(side_effect=AssertionError("Navigation must not request AI"))
    monkeypatch.setattr(
        workflow_proposal_codex,
        "generate_codex_proposal",
        model_request,
    )
    widget = VippWidget(
        ViewerModel(), defer_initial_run=True, initial_compute_mode=ComputeMode.CPU
    )
    qtbot.addWidget(widget)
    monkeypatch.setattr(widget, "_confirm_close_dirty_workflow_tabs", lambda: True)
    widget.pipeline.reset_empty_graph()
    widget._build_graph_from_pipeline()
    data = np.arange(24 * 32, dtype=np.float32).reshape(24, 32).copy()
    data.flags.writeable = False
    state = image_state_from_array(
        data, axes=(AxisMetadata("y", "space"), AxisMetadata("x", "space"))
    )
    widget.pipeline.outputs["input"] = data
    widget.pipeline.node_outputs["input"] = [data]
    widget.pipeline.output_states["input"] = state
    widget.pipeline.node_output_states["input"] = [state]
    widget.pipeline.completed_node_ids.add("input")
    calls = []
    widget._assistant_test_model_request = model_request
    monkeypatch.setattr(widget, "run_pipeline", lambda **kwargs: calls.append(kwargs))
    widget._history.clear()
    return widget, data, calls


def add_plan(widget, *, include_images=False):
    context, owner = widget._capture_workflow_proposal_context(
        "input", include_images=include_images
    )
    prepared = validate_proposal(reference_proposal(context), context)
    return widget._apply_workflow_proposal(prepared, owner)


def prepare_save_plan(widget, directory):
    destination = OutputDestination.from_directory(directory)
    context, owner = widget._capture_workflow_proposal_context(
        "input", output_destination=destination
    )
    raw = reference_proposal(context)
    raw["nodes"].append(
        {
            "key": "save",
            "operation_id": "save_output",
            "params": {
                "enabled": "on",
                "path": OUTPUT_DIRECTORY_PLACEHOLDER,
                "format": "npy",
                "overwrite": "no",
            },
            "reason": "Save only when the user explicitly calculates.",
        }
    )
    raw["connections"].append(
        {"source": "blur", "source_port": 0, "target": "save", "target_port": 0}
    )
    return validate_proposal(raw, context), owner


def test_save_add_keeps_binding_local_manual_and_does_not_write(editor, tmp_path):
    widget, data, calls = editor
    prepared, owner = prepare_save_plan(widget, tmp_path)
    assert str(tmp_path) not in str(prepared._context.public_request("Save my result"))
    ids = widget._apply_workflow_proposal(prepared, owner)
    save_id = ids[-1]
    assert widget.pipeline.nodes[save_id].params["path"] == str(
        tmp_path / "vipp-save.npy"
    )
    assert widget.pipeline.is_manual_node(save_id)
    assert not widget.pipeline.node_auto_recalculate(save_id)
    assert widget._inspector_profile_for_node(save_id).execution_is_manual
    assert (
        widget._workflow_assistant_run_revision.context.output_destination is not None
    )
    assert widget.pipeline.outputs["input"] is data
    assert calls == [] and list(tmp_path.iterdir()) == []
    widget._run_workflow_assistant_plan(ids)
    assert len(calls) == 1
    assert save_id in calls[0]["manual_node_ids"]
    assert list(tmp_path.iterdir()) == []  # Run dispatch is spied, not executed.


def test_save_collision_before_add_leaves_live_graph_unchanged(editor, tmp_path):
    widget, _data, calls = editor
    prepared, owner = prepare_save_plan(widget, tmp_path)
    before = GraphSnapshot.from_pipeline(widget.pipeline)
    target = tmp_path / "vipp-save.npy"
    target.write_bytes(b"existing scientific result")
    with pytest.raises(ValueError, match="already exists"):
        widget._apply_workflow_proposal(prepared, owner)
    assert GraphSnapshot.from_pipeline(widget.pipeline) == before
    assert target.read_bytes() == b"existing scientific result"
    assert calls == []


def test_save_collision_before_run_does_not_dispatch_or_overwrite(editor, tmp_path):
    widget, _data, calls = editor
    prepared, owner = prepare_save_plan(widget, tmp_path)
    ids = widget._apply_workflow_proposal(prepared, owner)
    target = tmp_path / "vipp-save.npy"
    target.write_bytes(b"another result arrived after Add")
    with pytest.raises(ValueError, match="already exists"):
        widget._run_workflow_assistant_plan(ids)
    assert target.read_bytes() == b"another result arrived after Add"
    assert calls == []


def test_same_input_folder_callback_uses_only_physical_source_binding(editor, tmp_path):
    widget, _data, _calls = editor
    assert widget._workflow_assistant_output_sources("input") == ()
    source = tmp_path / "source.tif"
    source.write_bytes(b"synthetic source identity only; not loaded")
    widget.pipeline.nodes["input"].params.update(
        source_mode="file path", file_path=str(source)
    )
    paths = widget._workflow_assistant_output_sources("input")
    destination = OutputDestination.same_folder_as_input(paths)
    assert destination.directory == str(tmp_path)
    assert destination.source_paths == (str(source),)
    assert widget._workflow_assistant_output_sources("not-a-source") == ()


def test_singleton_open_uses_workspace_reveal_without_run(editor, monkeypatch):
    from napari_vipp.ui import workflow_assistant

    widget, _data, calls = editor
    shown = []
    constructor_arguments = []
    original_dialog = workflow_assistant.WorkflowAssistantDialog

    def construct(*args, **kwargs):
        constructor_arguments.append(kwargs)
        return original_dialog(*args, **kwargs)

    monkeypatch.setattr(workflow_assistant, "WorkflowAssistantDialog", construct)
    monkeypatch.setattr(_widget, "show_workspace_window", shown.append)
    widget._populate_workflow_actions_menu()
    action = next(
        item
        for item in widget.workflow_actions_menu.actions()
        if item.text() == "Workflow assistant…"
    )
    action.trigger()
    dialog = widget._workflow_assistant_dialog
    refresh_sources = Mock(wraps=dialog.refresh_sources)
    monkeypatch.setattr(dialog, "refresh_sources", refresh_sources)
    widget._open_workflow_assistant()
    assert shown == [dialog, dialog]
    assert widget._workflow_assistant_dialog is dialog
    refresh_sources.assert_called_once_with()
    assert len(constructor_arguments) == 1
    assert constructor_arguments[0]["run"] == widget._run_workflow_assistant_plan
    assert widget._workflow_assistant_run_revision is None
    assert calls == []
    assert widget._assistant_test_model_request.call_count == 0
    dialog.close()


def test_task_tabs_and_back_next_do_not_add_or_run(editor, monkeypatch, qtbot):
    widget, _data, calls = editor
    run_requests = []
    monkeypatch.setattr(widget, "_run_workflow_assistant_plan", run_requests.append)
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    before = GraphSnapshot.from_pipeline(widget.pipeline)
    tabs = next(
        control
        for control in dialog.findChildren(QTabWidget)
        if [
            control.tabText(index).split("·", 1)[-1].strip()
            for index in range(control.count())
        ]
        == ["Goal", "Discussion", "Plans"]
    )
    for index in (1, 2, 0):
        tabs.setCurrentIndex(index)
    assert dialog.next_button.isHidden()
    for button in (dialog.next_button, dialog.back_button):
        if button.isEnabled():
            button.click()
    assert GraphSnapshot.from_pipeline(widget.pipeline) == before
    assert widget._workflow_assistant_run_revision is None
    assert run_requests == calls == []
    assert widget._assistant_test_model_request.call_count == 0
    dialog.close()


def test_run_requires_a_successful_explicit_add(editor):
    widget, _data, calls = editor
    with pytest.raises(ProposalError, match="explicitly added"):
        widget._run_workflow_assistant_plan(("unaccepted",))
    assert calls == []


def test_add_keeps_pixels_private_and_run_joins_normal_calculate_all(editor):
    widget, data, calls = editor
    ids = add_plan(widget, include_images=True)
    revision = widget._workflow_assistant_run_revision
    assert revision.added_node_ids == ids
    assert revision.context.image_previews == ()
    assert revision.context.baseline == GraphSnapshot.from_pipeline(widget.pipeline)
    assert revision.owner[0] == widget._current_workflow_session_id()
    assert revision.owner[1] == widget._current_history_snapshot().workflow
    assert calls == []
    manual = widget._manual_node_ids_needing_calculation()
    widget._run_workflow_assistant_plan(ids)
    assert calls == [{"manual_node_ids": manual}]
    assert widget.pipeline.outputs["input"] is data
    assert all(widget.pipeline.outputs[node_id] is None for node_id in ids)
    assert not data.flags.writeable


@pytest.mark.parametrize("bad_ids", [(), ("old-plan",), ["old-plan"], None])
def test_wrong_or_mutable_plan_identity_cannot_run(editor, bad_ids):
    widget, _data, calls = editor
    add_plan(widget)
    with pytest.raises(ProposalError, match="latest plan"):
        widget._run_workflow_assistant_plan(bad_ids)
    assert calls == []


def test_historical_added_ids_cannot_run_after_a_new_add(editor):
    widget, _data, calls = editor
    old_ids = add_plan(widget)
    latest_ids = add_plan(widget)
    assert old_ids != latest_ids
    assert set(old_ids) <= set(widget.pipeline.nodes)
    with pytest.raises(ProposalError, match="latest plan"):
        widget._run_workflow_assistant_plan(old_ids)
    assert calls == []
    widget._run_workflow_assistant_plan(latest_ids)
    assert len(calls) == 1


def test_changed_graph_cannot_run_from_old_add(editor):
    widget, _data, calls = editor
    ids = add_plan(widget)
    blur = next(
        node
        for node in widget.pipeline.nodes.values()
        if node.operation_id == "gaussian_blur"
    )
    blur.params["sigma"] = 3.0
    changed = GraphSnapshot.from_pipeline(widget.pipeline)
    with pytest.raises(ProposalError, match="changed"):
        widget._run_workflow_assistant_plan(ids)
    assert GraphSnapshot.from_pipeline(widget.pipeline) == changed
    assert calls == []


@pytest.mark.parametrize("change", ["pixels", "calibration"])
def test_changed_source_cannot_run_from_old_add(editor, change):
    widget, data, calls = editor
    ids = add_plan(widget)
    if change == "pixels":
        changed = data.copy()
        changed[0, 0] = 0.5
        changed.flags.writeable = False
        widget.pipeline.outputs["input"] = changed
        widget.pipeline.node_outputs["input"] = [changed]
    else:
        changed = data
        state = image_state_from_array(
            data,
            axes=(
                AxisMetadata("y", "space", "um", scale=2.0),
                AxisMetadata("x", "space", "um", scale=1.0),
            ),
        )
        widget.pipeline.output_states["input"] = state
        widget.pipeline.node_output_states["input"] = [state]
    with pytest.raises(ProposalError, match="source changed"):
        widget._run_workflow_assistant_plan(ids)
    assert calls == []
    assert widget.pipeline.outputs["input"] is changed


def test_changed_workflow_session_cannot_run_from_old_add(editor, monkeypatch):
    widget, _data, calls = editor
    ids = add_plan(widget)
    monkeypatch.setattr(widget, "_current_workflow_session_id", lambda: "other-session")
    with pytest.raises(ProposalError, match="tab changed"):
        widget._run_workflow_assistant_plan(ids)
    assert calls == []


def test_changed_compute_intent_cannot_run_from_old_add(editor):
    widget, _data, calls = editor
    ids = add_plan(widget)
    widget._compute_mode = ComputeMode.AUTO
    with pytest.raises(ProposalError, match="settings changed"):
        widget._run_workflow_assistant_plan(ids)
    assert calls == []


@pytest.mark.parametrize("busy", ["debounce", "active", "crop", "tuning", "source"])
def test_pending_or_active_work_blocks_assistant_run(editor, busy):
    widget, _data, calls = editor
    ids = add_plan(widget)
    if busy == "debounce":
        widget._debounce_timer.start(30_000)
    elif busy == "active":
        widget._active_pipeline_run_id = 999
    elif busy == "crop":
        widget._crop_draft_node_id = "input"
    elif busy == "tuning":
        widget._isolated_tuning_node_id = "input"
    else:
        widget._source_load_pending = True
    try:
        with pytest.raises(ProposalError):
            widget._run_workflow_assistant_plan(ids)
        assert calls == []
    finally:
        widget._debounce_timer.stop()
        widget._active_pipeline_run_id = None
        widget._crop_draft_node_id = None
        widget._isolated_tuning_node_id = None
        widget._source_load_pending = False


def test_closing_editor_cannot_start_assistant_run(editor):
    widget, _data, calls = editor
    ids = add_plan(widget)
    widget._closing = True
    try:
        with pytest.raises(ProposalError, match="closing"):
            widget._run_workflow_assistant_plan(ids)
        assert calls == []
    finally:
        widget._closing = False


@pytest.mark.parametrize("prior_added", [False, True])
def test_post_add_guard_failure_rolls_back_without_granting_run(
    editor, monkeypatch, prior_added
):
    widget, data, calls = editor
    old_ids = add_plan(widget) if prior_added else None
    previous_run = widget._workflow_assistant_run_revision
    previous_undo = list(widget._history.undo_stack)
    context, owner = widget._capture_workflow_proposal_context("input")
    prepared = validate_proposal(reference_proposal(context), context)
    before = GraphSnapshot.from_pipeline(widget.pipeline)
    monkeypatch.setattr(
        _widget,
        "capture_proposal_context",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ProposalError("guard failed")),
    )
    with pytest.raises(ProposalError, match="rolled back"):
        widget._apply_workflow_proposal(prepared, owner)
    assert GraphSnapshot.from_pipeline(widget.pipeline) == before
    assert widget._workflow_assistant_run_revision is previous_run
    assert widget.pipeline.outputs["input"] is data
    assert widget._history.undo_stack == previous_undo
    assert calls == []
    if prior_added:
        widget._run_workflow_assistant_plan(old_ids)
        assert len(calls) == 1
