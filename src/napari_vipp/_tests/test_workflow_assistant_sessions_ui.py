"""Local session UI never imports sharing/execution authority from an archive."""

import json

import pytest

from napari_vipp._tests import test_workflow_assistant as assistant_tests
from napari_vipp.core.workflow_assistant_session import (
    make_checkpoint,
    save_assistant_session,
)
from napari_vipp.core.workflow_investigation import InvestigationSession

pytest_plugins = (assistant_tests.__name__,)


def test_plain_conversation_and_plan_reopen_passively(editor, tmp_path):
    widget, calls = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    dialog.provider_combo.setCurrentIndex(2)
    dialog.goal_edit.setPlainText(
        "Separate the wavy band from the central bright structure."
    )
    dialog.prepare()
    dialog._append_chat_reply("Keep the objects separate?", "  Yes, two objects.  ")
    dialog._clarifications = [("Keep the objects separate?", "  Yes, two objects.  ")]
    dialog.discussion_edit.setPlainText("Unsent note, kept literally.")
    before = dialog._sessions._ui_state()
    path = tmp_path / "conversation.vipp-session"
    receipt = dialog._sessions.save_path(path)
    assert receipt.path == path and receipt.byte_count > 0
    dialog.images_checkbox.setChecked(True)
    dialog.output_assessment_checkbox.setChecked(True)
    dialog._sessions.load_path(path)
    after = dialog._sessions._ui_state()
    assert after == before
    assert (
        not dialog._prepared and not dialog._draft_authorized and not dialog._added_ids
    )
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    assert not dialog._patching.button.isEnabled()
    assert not dialog.images_checkbox.isChecked()
    assert not dialog.output_assessment_checkbox.isChecked()
    assert not dialog.consent_checkbox.isChecked()
    assert dialog._context is None and dialog._owner is None
    assert dialog._investigation.session is None
    assert calls == []
    dialog.close()


def test_invalid_ui_fields_leave_current_conversation_untouched(editor, tmp_path):
    widget, calls = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    dialog._append_chat("You", "Keep this conversation.")
    state = dialog._sessions._ui_state()
    state["provider_token"] = "never-an-allowed-presentation-field"
    checkpoint = make_checkpoint({"ui_state": state}, {})
    path = tmp_path / "bad.vipp-session"
    save_assistant_session(path, checkpoint)
    before = dialog._sessions._ui_state()
    with pytest.raises(ValueError, match="Unsupported saved conversation"):
        dialog._sessions.load_path(path)
    assert dialog._sessions._ui_state() == before and calls == []
    dialog.close()


def test_session_cannot_overwrite_bound_original_or_workflow(editor, tmp_path):
    widget, calls = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    path = tmp_path / "original.vipp-session"
    original = b"original scientific data"
    path.write_bytes(original)
    widget.pipeline.nodes["input"].params["file_path"] = str(path)
    with pytest.raises(ValueError, match="bound input"):
        dialog._sessions.save_path(path, overwrite=True)
    assert path.read_bytes() == original and calls == []
    workflow = tmp_path / "workflow.vipp-session"
    workflow.write_text(json.dumps({"original": True}), encoding="utf-8")
    widget._workflow_tabs.current.mark_saved(workflow)
    with pytest.raises(ValueError, match="original workflow"):
        dialog._sessions.save_path(workflow, overwrite=True)
    assert json.loads(workflow.read_text()) == {"original": True}
    dialog.close()


def test_session_save_while_worker_active_requires_pause(editor, tmp_path):
    widget, _ = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    dialog._investigation._worker_kind = "trial"
    with pytest.raises(ValueError, match="Pause"):
        dialog._sessions.save_path(tmp_path / "unsafe.vipp-session")
    dialog._investigation._worker_kind = None
    dialog.close()


def test_verified_continuation_is_paused_until_explicit_continue(
    editor, tmp_path, monkeypatch
):
    widget, calculations = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    goal = "Keep the two structures separate."
    dialog.goal_edit.setPlainText(goal)
    dialog._investigation.session = InvestigationSession(
        widget.pipeline,
        "input",
        goal,
        compute_request=widget._current_compute_request(),
        include_output_images=True,
        continuation_enabled=True,
        assert_current=lambda: None,
    )
    dialog._investigation.session.request_pause()
    path = tmp_path / "paused.vipp-session"
    dialog._sessions.save_path(path)
    dialog._sessions.load_path(path)
    loaded = dialog._sessions.loaded
    copy = tmp_path / "paused-copy.vipp-session"
    dialog._sessions.save_path(copy)
    from napari_vipp.core.workflow_assistant_session import load_assistant_session

    copied = load_assistant_session(copy)
    assert copied.blob_names == loaded.blob_names
    assert copied.document["session_id"] == loaded.document["session_id"]
    assert copied.document["scientific"] == loaded.document["scientific"]
    dialog.provider_combo.setCurrentIndex(1)
    dialog.model_combo.setCurrentText("synthetic-local-model")
    dialog.output_assessment_checkbox.setChecked(True)
    result = dialog._sessions.prepare_resume()
    controller = dialog._investigation
    assert result.resumable and controller.session.paused
    assert controller.active and controller._paused
    assert calculations == []
    assert controller.session.actions_used == 0
    steps = []
    monkeypatch.setattr(
        controller, "_next_guarded", lambda: steps.append("explicit continuation")
    )
    controller._resume()
    assert steps == ["explicit continuation"] and calculations == []
    assert not controller.session.paused
    dialog.close()


def test_restored_finish_uses_review_not_continue(editor, tmp_path):
    from napari_vipp._tests.test_workflow_investigation_finish_review import nominate
    from napari_vipp._tests.test_workflow_investigation_warmstart import (
        accepted,
        update,
    )

    widget, calculations = editor
    pipeline, _, bundle = accepted(mesh=True)
    session = InvestigationSession.from_completed(
        pipeline,
        bundle,
        "Keep the structures separate.",
        compute_request=widget._current_compute_request(),
        include_output_images=True,
        continuation_enabled=True,
        review_before_finish=True,
        assert_current=lambda: None,
    )
    assert update(session, 80)["status"] == "ok"
    nominate(session, "candidate1")
    widget.pipeline = pipeline
    widget._workflow_tabs.current.pipeline = pipeline
    widget._build_graph_from_pipeline()
    widget._sync_current_workflow_tab_state()
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    dialog.goal_edit.setPlainText(session.dialogue.objective)
    dialog._investigation.session = session
    path = tmp_path / "finished.vipp-session"
    dialog._sessions.save_path(path)
    dialog._sessions.load_path(path)
    dialog.provider_combo.setCurrentIndex(1)
    dialog.model_combo.setCurrentText("synthetic-local-model")
    dialog.output_assessment_checkbox.setChecked(True)
    dialog._sessions.prepare_resume()
    controller = dialog._investigation
    assert controller._review_pending and not controller._paused
    assert not controller.review_card.isHidden()
    assert controller.pause_card.isHidden()
    assert not dialog.apply_button.isEnabled() and calculations == []
    controller._accept_finish_review()
    assert controller.session.finished and not controller.active
    assert dialog._prepared is not None and dialog._prepared.can_apply
    assert calculations == []
    dialog.close()


def test_saved_question_keeps_its_unsent_answer_without_duplicate_chat(
    editor, tmp_path
):
    from napari_vipp._tests.test_workflow_investigation import act

    widget, calculations = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    goal, question, answer = (
        "Separate the objects.",
        "Keep them separate?",
        "  Yes, two.  ",
    )
    dialog.goal_edit.setPlainText(goal)
    session = InvestigationSession(
        widget.pipeline,
        "input",
        goal,
        compute_request=widget._current_compute_request(),
        include_output_images=True,
        continuation_enabled=True,
        assert_current=lambda: None,
    )
    assert act(session, "question", question=question).public_result()["status"] == "ok"
    dialog._investigation.session = session
    dialog._append_chat("Assistant", question)
    dialog._investigation.pause_answer.setText(answer)
    path = tmp_path / "question.vipp-session"
    dialog._sessions.save_path(path)
    dialog._sessions.load_path(path)
    dialog.provider_combo.setCurrentIndex(1)
    dialog.model_combo.setCurrentText("synthetic-local-model")
    dialog.output_assessment_checkbox.setChecked(True)
    before = list(dialog._chat)
    dialog._sessions.prepare_resume()
    assert dialog._investigation._pending_question == question
    assert dialog._investigation.pause_answer.text() == answer
    assert dialog._chat == before and calculations == []
    dialog.close()


def test_normalized_session_filename_must_confirm_exact_replacement(
    editor, tmp_path, monkeypatch
):
    from qtpy.QtWidgets import QMessageBox

    import napari_vipp.ui.workflow_assistant_sessions as ui

    widget, _ = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    path = tmp_path / "existing.vipp-session"
    path.write_bytes(b"keep this original archive")
    monkeypatch.setattr(
        ui.QFileDialog,
        "getSaveFileName",
        lambda *a, **k: (str(path.with_suffix("")), ""),
    )
    prompts = []

    def reject(parent, title, text, *args):
        prompts.append((title, text))
        return QMessageBox.No

    monkeypatch.setattr(ui.QMessageBox, "question", reject)
    dialog._sessions.save()
    assert len(prompts) == 1 and str(path) in prompts[0][1]
    assert path.read_bytes() == b"keep this original archive"
    dialog.close()


def test_new_target_during_content_confirmation_is_not_overwritten(
    editor, tmp_path, monkeypatch
):
    from qtpy.QtWidgets import QMessageBox

    import napari_vipp.ui.workflow_assistant_sessions as ui

    widget, _ = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    path = tmp_path / "appeared.vipp-session"
    monkeypatch.setattr(
        ui.QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), "")
    )

    def appear(parent, title, text, *args):
        assert title == "Save local session data?"
        path.write_bytes(b"created by another writer while the notice was open")
        return QMessageBox.Yes

    monkeypatch.setattr(ui.QMessageBox, "question", appear)
    dialog._sessions.save()
    assert path.read_bytes() == b"created by another writer while the notice was open"
    assert "not saved" in dialog.status_label.text()
    dialog.close()
