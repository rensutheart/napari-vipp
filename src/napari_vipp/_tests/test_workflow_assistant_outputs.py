"""Local output choices are private policy, never provider or writer execution."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from qtpy.QtWidgets import QLabel, QLineEdit, QPlainTextEdit

from napari_vipp._tests.test_workflow_assistant_stages import assistant as assistant
from napari_vipp.core.pipeline import NODE_LIBRARY_BY_ID
from napari_vipp.core.workflow_proposal_outputs import (
    OUTPUT_DIRECTORY_PLACEHOLDER,
    OutputDestinationError,
)
from napari_vipp.core.workflow_proposals import ProposalDialogue, reference_proposal
from napari_vipp.ui import workflow_assistant


@pytest.fixture
def local_outputs(assistant, monkeypatch):
    """Use held workers and metadata-only physical source fixtures, not images."""
    dialog = assistant.dialog
    state = SimpleNamespace(paths=(), source_calls=[], captures=[], queued=[])
    legacy_capture = dialog._capture

    def source_paths(source_id):
        state.source_calls.append(source_id)
        return state.paths

    def capture(source_id, **options):
        state.captures.append((source_id, dict(options)))
        context, owner = legacy_capture(
            source_id, include_images=options["include_images"]
        )
        return replace(
            context, output_destination=options.get("output_destination")
        ), owner

    dialog._output_sources = source_paths
    dialog._capture = capture
    monkeypatch.setattr(
        workflow_assistant,
        "QThreadPool",
        SimpleNamespace(
            globalInstance=lambda: SimpleNamespace(start=state.queued.append)
        ),
    )
    state.assistant, state.dialog = assistant, dialog
    return state


def _choose(state, monkeypatch, folder):
    monkeypatch.setattr(
        workflow_assistant.QFileDialog,
        "getExistingDirectory",
        lambda *args: str(folder),
    )
    state.dialog.output_folder_button.click()


def _local(dialog):
    dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("local"))
    dialog.model_combo.setCurrentText("held-offline-model")


def _file_source(state, tmp_path):
    source = tmp_path / "original.tif"
    source.touch()  # A physical binding only: no image is loaded or decoded.
    state.paths = (str(source),)
    return source


def _save_proposal(context):
    params = {
        parameter.name: parameter.default
        for parameter in NODE_LIBRARY_BY_ID["save_output"].parameters
    }
    params.update(
        path=OUTPUT_DIRECTORY_PLACEHOLDER, enabled="on", overwrite="no", format="npy"
    )
    return {
        "kind": "napari-vipp-workflow-proposal",
        "version": 1,
        "context_id": context.context_id,
        "summary": "Save the selected image after explicit Run.",
        "assumptions": ["Use the folder selected locally; do not overwrite files."],
        "questions": [],
        "nodes": [
            {
                "key": "result",
                "operation_id": "save_output",
                "params": params,
                "reason": "The user requested an image output.",
            }
        ],
        "connections": [
            {"source": "input", "source_port": 0, "target": "result", "target_port": 0}
        ],
    }


def test_default_is_unbound_and_preserves_legacy_capture_signature(assistant):
    dialog = assistant.dialog
    assert dialog.output_folder_combo.currentText() == "Ask me when saving"
    assert dialog._output_destination is None
    assert "never save" in dialog.output_folder_status.text()
    request = dialog._ensure_context()
    assert assistant.captures == [("input", False)]
    assert dialog._context.output_destination is None
    assert request["output_destination"]["selected"] is False
    assert dialog.output_saving_label.isHidden()
    assert assistant.applies == assistant.runs == []


def test_choose_cancel_is_passive_and_dialog_cannot_create_folders(
    local_outputs, monkeypatch
):
    state, calls = local_outputs, []
    dialog = state.dialog

    def chooser(*args):
        calls.append(args)
        return ""

    monkeypatch.setattr(workflow_assistant.QFileDialog, "getExistingDirectory", chooser)
    dialog.output_folder_combo.setCurrentIndex(
        dialog.output_folder_combo.findData("directory")
    )
    assert len(calls) == 1
    assert calls[0][3] & workflow_assistant.QFileDialog.ReadOnly
    assert dialog.output_folder_combo.currentData() == "ask"
    assert dialog._output_destination is None
    assert state.captures == state.source_calls == state.queued == []


def test_manual_folder_is_local_and_captured_only_after_explicit_request(
    local_outputs, monkeypatch, tmp_path, qtbot
):
    state, folder = local_outputs, tmp_path / "private-results"
    folder.mkdir()
    _choose(state, monkeypatch, folder)
    dialog = state.dialog
    assert dialog._output_destination.directory == str(folder)
    assert str(folder) in dialog.output_folder_status.text()
    assert state.captures == state.queued == []
    assert list(folder.iterdir()) == []
    request = dialog._ensure_context()
    assert state.captures[0][1]["output_destination"] is dialog._output_destination
    assert request["output_destination"]["selected"] is True
    assert str(folder) not in json.dumps(request)
    preview = dialog._sharing_preview_dialog(request)
    qtbot.addWidget(preview)
    text = "\n".join(widget.text() for widget in preview.findChildren(QLabel))
    text += "\n".join(
        widget.toPlainText() for widget in preview.findChildren(QPlainTextEdit)
    )
    assert str(folder) not in text
    assert state.assistant.applies == state.assistant.runs == []
    assert list(folder.iterdir()) == []


@pytest.mark.parametrize("kind", ["tif", "zarr"])
def test_same_input_folder_uses_only_verified_physical_binding(
    local_outputs, tmp_path, kind
):
    state = local_outputs
    if kind == "tif":
        source = _file_source(state, tmp_path)
    else:
        source = tmp_path / "original.ome.zarr"
        source.mkdir()
        (source / ".zgroup").touch()
        state.paths = (str(source),)
    dialog = state.dialog
    dialog.output_folder_combo.setCurrentIndex(
        dialog.output_folder_combo.findData("same")
    )
    assert dialog._output_destination.mode == "same_input_folder"
    assert dialog._output_destination.directory == str(source.parent)
    assert state.captures == state.queued == []
    dialog._ensure_context()
    assert dialog._context.output_destination.source_paths == (str(source),)


@pytest.mark.parametrize("source_label", ["Bundled sample", "Live napari image"])
def test_sample_and_live_inputs_need_explicit_folder_not_guessed_name(
    local_outputs, source_label
):
    state, dialog = local_outputs, local_outputs.dialog
    dialog.source_combo.setItemText(0, source_label)
    dialog.output_folder_combo.setCurrentIndex(
        dialog.output_folder_combo.findData("same")
    )
    assert dialog._output_destination is None
    assert "no verified file folder" in dialog.output_folder_status.text()
    dialog.prepare()
    assert "Choose an output folder yourself" in dialog.status_label.text()
    assert state.captures == state.queued == []
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()


def test_multiple_physical_sources_do_not_guess_one_folder(local_outputs, tmp_path):
    state, dialog = local_outputs, local_outputs.dialog
    first = _file_source(state, tmp_path)
    other = tmp_path / "other.tif"
    other.touch()
    state.paths = (str(first), str(other))
    dialog.output_folder_combo.setCurrentIndex(
        dialog.output_folder_combo.findData("same")
    )
    assert dialog._output_destination is None
    assert "not unambiguous" in dialog.output_folder_status.text()
    assert state.captures == state.queued == []


def test_natural_folder_answer_redacts_only_detached_request_not_literal_history(
    local_outputs, tmp_path
):
    state, dialog = local_outputs, local_outputs.dialog
    folder = tmp_path / "private-selected-folder"
    folder.mkdir()
    goal = f"  Segment the image. Save into {folder}.  "
    answer = f"  {folder}  "
    question = "Where should the output be saved?"
    dialog.goal_edit.setPlainText(goal)
    objective = ProposalDialogue(goal, ((question, answer),))
    request = dialog._ensure_context(objective)
    dialog._append_request_chat(objective)
    assert objective.objective == goal and objective.clarifications[0][1] == answer
    assert dialog.goal_edit.toPlainText() == goal
    assert ("You · Goal", goal) in dialog._chat
    assert dialog._chat_replies[len(dialog._chat) - 1].answer == answer
    assert str(folder) not in json.dumps(request)
    assert "the output folder selected locally" in request["objective"]
    assert request["clarifications"][0]["answer"].strip() == (
        "the output folder selected locally"
    )
    assert str(folder) not in dialog.request_edit.toPlainText()
    assert state.queued == []


def test_exact_same_folder_message_binds_and_sends_once_with_this_clicks_approval(
    local_outputs, tmp_path
):
    state, dialog = local_outputs, local_outputs.dialog
    source = _file_source(state, tmp_path)
    _local(dialog)
    message = "  same folder as input  "
    dialog.discussion_edit.setPlainText(message)
    dialog._send_message_clicked()
    assert dialog._output_destination.directory == str(source.parent)
    assert dialog.consent_checkbox.isChecked()
    assert dialog._busy and len(state.queued) == 1
    assert not dialog.output_folder_combo.isEnabled()
    assert not dialog.output_folder_button.isEnabled()
    assert dialog._pending_clarifications[-1][1] == message
    assert dialog._chat_replies[len(dialog._chat) - 1].answer == message
    assert state.assistant.applies == state.assistant.runs == []


def test_folder_choice_keeps_discussion_but_revokes_old_add_and_run(
    local_outputs, monkeypatch, tmp_path
):
    state, dialog = local_outputs, local_outputs.dialog
    dialog._ensure_context()
    dialog._accept_proposal(reference_proposal(dialog._context))
    assert dialog.apply_button.isEnabled()
    dialog._clarifications = [
        ("Keep this exact question?", "  Yes, keep this answer.  ")
    ]
    field = QLineEdit("  Still unsent.  ", dialog)
    dialog._question_fields = [("Where?", field)]
    dialog._append_chat_reply("Keep this exact question?", "  Yes, keep this answer.  ")
    before = (
        dialog.goal_edit.toPlainText(),
        tuple(dialog._clarifications),
        tuple(dialog._chat),
        tuple(dialog._drafts),
    )
    folder = tmp_path / "chosen"
    folder.mkdir()
    _choose(state, monkeypatch, folder)
    assert before == (
        dialog.goal_edit.toPlainText(),
        tuple(dialog._clarifications),
        tuple(dialog._chat),
        tuple(dialog._drafts),
    )
    assert field.text() == "  Still unsent.  "
    assert dialog._prepared is None and not dialog._draft_authorized
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    assert dialog._recovery_kind == "update"
    assert state.queued == state.assistant.applies == state.assistant.runs == []


def test_old_folder_answer_cannot_undo_new_manual_ask_choice(local_outputs, tmp_path):
    dialog = local_outputs.dialog
    folder = tmp_path / "private-old-choice"
    folder.mkdir()
    objective = ProposalDialogue("Save the image.", (("Where?", str(folder)),))
    dialog._ensure_context(objective)
    dialog.output_folder_combo.setCurrentIndex(0)
    assert dialog._output_destination is None
    request = dialog._ensure_context(objective)
    assert dialog._output_destination is None
    assert request["output_destination"]["selected"] is False
    assert str(folder) not in json.dumps(request)


def test_replaced_folder_is_not_renewed_silently_by_old_answer(local_outputs, tmp_path):
    state, dialog = local_outputs, local_outputs.dialog
    folder = tmp_path / "chosen"
    folder.mkdir()
    objective = ProposalDialogue("Save the image.", (("Where?", str(folder)),))
    dialog._ensure_context(objective)
    binding = dialog._output_destination
    folder.rename(tmp_path / "moved-choice")
    folder.mkdir()
    with pytest.raises(OutputDestinationError, match="folder changed"):
        dialog._ensure_context(objective)
    assert dialog._output_destination is binding
    assert len(state.captures) == 1 and state.queued == []


def test_explicit_new_folder_can_repair_old_missing_folder_without_losing_answers(
    local_outputs, monkeypatch, tmp_path
):
    state, dialog = local_outputs, local_outputs.dialog
    old_folder, new_folder = tmp_path / "old-choice", tmp_path / "new-choice"
    old_folder.mkdir()
    new_folder.mkdir()
    _choose(state, monkeypatch, old_folder)
    dialog._ensure_context()
    dialog._accept_proposal(reference_proposal(dialog._context))
    dialog._clarifications = [("Keep these choices?", "  Yes, keep them.  ")]
    dialog._append_chat_reply("Keep these choices?", "  Yes, keep them.  ")
    before = (
        dialog.goal_edit.toPlainText(),
        tuple(dialog._clarifications),
        tuple(dialog._chat),
        tuple(dialog._drafts),
    )
    old_folder.rename(tmp_path / "moved-old-choice")
    _choose(state, monkeypatch, new_folder)
    assert dialog._output_destination.directory == str(new_folder)
    assert before == (
        dialog.goal_edit.toPlainText(),
        tuple(dialog._clarifications),
        tuple(dialog._chat),
        tuple(dialog._drafts),
    )
    assert state.assistant.assertions[-1][0].output_destination is None
    assert dialog._image_draft_guard[0].output_destination is dialog._output_destination
    assert dialog._prepared is None and not dialog._draft_authorized
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    dialog._ensure_context()
    assert dialog._context.output_destination.directory == str(new_folder)
    assert state.queued == state.assistant.applies == state.assistant.runs == []


def test_missing_absolute_folder_message_stops_before_capture_or_provider(
    local_outputs, tmp_path
):
    state, dialog = local_outputs, local_outputs.dialog
    _local(dialog)
    message = str(tmp_path / "not-created")
    dialog.discussion_edit.setPlainText(message)
    dialog._send_message_clicked()
    assert state.captures == state.queued == []
    assert dialog.discussion_edit.toPlainText() == message
    assert "no longer available" in dialog.status_label.text()
    assert dialog._output_destination is None
    assert not (tmp_path / "not-created").exists()


def test_new_folder_answer_can_replace_missing_guard_destination_locally(
    local_outputs, monkeypatch, tmp_path
):
    state, dialog = local_outputs, local_outputs.dialog
    old_folder, new_folder = tmp_path / "old-choice", tmp_path / "new-choice"
    old_folder.mkdir()
    new_folder.mkdir()
    _choose(state, monkeypatch, old_folder)
    dialog._ensure_context()
    dialog._clarifications = [("Keep my choices?", "  Yes.  ")]
    dialog._retain_for_revision("Sharing choice changed.", "A new request is needed.")
    assert dialog._context is None and dialog._image_draft_guard is not None
    old_folder.rename(tmp_path / "moved-old-choice")
    objective = ProposalDialogue(
        dialog.goal_edit.toPlainText(),
        (*dialog._clarifications, ("Where should it save?", str(new_folder))),
    )
    request = dialog._ensure_context(objective)
    assert dialog._context.output_destination.directory == str(new_folder)
    assert dialog._clarifications == [("Keep my choices?", "  Yes.  ")]
    assert str(new_folder) not in request["clarifications"][-1]["answer"]
    assert state.queued == state.assistant.applies == state.assistant.runs == []


def test_model_reply_cannot_select_folder(local_outputs, tmp_path):
    state, dialog = local_outputs, local_outputs.dialog
    dialog._ensure_context()
    raw = reference_proposal(dialog._context)
    raw["summary"] = f"Model suggests {tmp_path}."
    dialog._accept_proposal(raw)
    assert dialog._output_destination is None
    assert state.source_calls == state.queued == []


def test_unbound_save_cannot_be_added_and_does_not_open_chooser(
    local_outputs, monkeypatch
):
    state, dialog = local_outputs, local_outputs.dialog
    monkeypatch.setattr(
        workflow_assistant.QFileDialog,
        "getExistingDirectory",
        lambda *args: pytest.fail("An AI proposal must not open a folder chooser."),
    )
    dialog._ensure_context()
    dialog._accept_proposal(_save_proposal(dialog._context))
    assert dialog._prepared is None and not dialog.apply_button.isEnabled()
    assert "folder" in dialog.status_label.text().lower()
    assert dialog.output_folder_combo.currentData() == "ask"
    assert state.source_calls == state.queued == []
    assert state.assistant.applies == state.assistant.runs == []


def test_source_change_clears_old_output_authority(
    local_outputs, monkeypatch, tmp_path
):
    state, dialog = local_outputs, local_outputs.dialog
    folder = tmp_path / "chosen"
    folder.mkdir()
    _choose(state, monkeypatch, folder)
    dialog._ensure_context()
    dialog.source_combo.addItem("Another input", "another")
    dialog.source_combo.setCurrentIndex(1)
    assert dialog._output_destination is None
    assert dialog.output_folder_combo.currentData() == "ask"
    assert dialog._context is None and dialog._prepared is None
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    assert state.queued == []


def test_changed_graph_cannot_reuse_old_plan_on_folder_change(
    local_outputs, monkeypatch, tmp_path
):
    state, dialog = local_outputs, local_outputs.dialog
    dialog._ensure_context()
    dialog._accept_proposal(reference_proposal(dialog._context))
    state.assistant.pipeline.add_node("gaussian_blur")
    folder = tmp_path / "chosen"
    folder.mkdir()
    _choose(state, monkeypatch, folder)
    assert dialog._output_destination is None
    assert dialog._prepared is None and not dialog._draft_authorized
    assert "input or workflow changed" in dialog.status_label.text().lower()
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    assert state.queued == state.assistant.applies == state.assistant.runs == []


def test_existing_output_name_blocks_add_without_overwrite(
    local_outputs, monkeypatch, tmp_path
):
    state, dialog = local_outputs, local_outputs.dialog
    folder = tmp_path / "chosen"
    folder.mkdir()
    target = folder / "vipp-result.npy"
    target.touch()
    _choose(state, monkeypatch, folder)
    dialog._ensure_context()
    dialog._accept_proposal(_save_proposal(dialog._context))
    assert dialog._prepared is None and not dialog.apply_button.isEnabled()
    assert "already exists" in dialog.status_label.text()
    assert target.is_file() and target.stat().st_size == 0
    assert state.queued == state.assistant.applies == state.assistant.runs == []


def test_many_save_paths_have_compact_footer_and_complete_local_tooltip(
    local_outputs, monkeypatch, tmp_path
):
    state, dialog = local_outputs, local_outputs.dialog
    folder = tmp_path / "chosen"
    folder.mkdir()
    _choose(state, monkeypatch, folder)
    dialog._ensure_context()
    raw = _save_proposal(dialog._context)
    template = raw["nodes"][0]
    raw["nodes"] = [{**template, "key": f"result_{index:02}"} for index in range(64)]
    raw["connections"] = [
        {"source": "input", "source_port": 0, "target": node["key"], "target_port": 0}
        for node in raw["nodes"]
    ]
    dialog._accept_proposal(raw)
    assert dialog._prepared is not None, dialog.status_label.text()
    assert len(dialog._output_plan_paths) == 64
    assert len(dialog.output_saving_label.text()) < 200
    assert "+62 more" in dialog.output_saving_label.text()
    for index in range(64):
        assert str(folder / f"vipp-result_{index:02}.npy") in (
            dialog.output_saving_label.toolTip()
        )
    assert list(folder.iterdir()) == []


def test_local_saving_summary_does_not_write_or_put_paths_in_provider_details(
    local_outputs, monkeypatch, tmp_path
):
    state, dialog = local_outputs, local_outputs.dialog
    folder = tmp_path / "chosen"
    folder.mkdir()
    _choose(state, monkeypatch, folder)
    dialog._ensure_context()
    dialog._accept_proposal(_save_proposal(dialog._context))
    assert dialog._prepared is not None, dialog.status_label.text()
    target = folder / "vipp-result.npy"
    assert dialog.output_saving_label.isVisible()
    assert target.name in dialog.output_saving_label.text()
    assert str(target) in dialog.output_saving_label.toolTip()
    assert str(target) in dialog.output_saving_label.accessibleDescription()
    assert "explicit execution" in dialog.output_saving_label.text()
    assert str(folder) not in dialog.json_edit.toPlainText()
    assert str(folder) not in dialog.request_edit.toPlainText()
    assert OUTPUT_DIRECTORY_PLACEHOLDER in dialog.json_edit.toPlainText()
    assert not target.exists()
    dialog.apply_button.click()
    assert dialog.run_button.isEnabled()
    assert str(target) in dialog.output_saving_label.toolTip()
    assert not target.exists()
    dialog.run_button.click()
    assert state.assistant.applies and state.assistant.runs
    assert not target.exists() and list(folder.iterdir()) == []
    assert state.queued == []
