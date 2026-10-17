"""Local session files: passive reopening, then explicitly approved continuation."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from pathlib import Path

from qtpy.QtWidgets import QFileDialog, QMenu, QMessageBox

from napari_vipp.ui.workflow_assistant_style import AssistantButton


class AssistantSessionController:
    def __init__(self, dialog, assert_destination=None):
        self.dialog = dialog
        self.assert_destination = assert_destination
        self.loaded = None
        self.last_receipt = None
        self.button = AssistantButton("Session…")
        self.button.setAutoDefault(False)
        self.button.setToolTip(
            "Save or open a local assistant session. Files can contain image "
            "and trial-result data; opening never sends requests or calculates."
        )
        self.menu = QMenu(self.button)
        self.save_action = self.menu.addAction("Save session…", self.save)
        self.open_action = self.menu.addAction("Open session…", self.open)
        self.menu.addSeparator()
        self.resume_action = self.menu.addAction(
            "Resume saved investigation…", self.resume
        )
        self.button.setMenu(self.menu)
        self.sync()

    def sync(self):
        dialog = self.dialog
        controller = dialog._investigation
        quiet = not dialog._busy or (
            controller.active
            and controller._worker_kind is None
            and (controller._paused or controller._review_pending)
        )
        self.save_action.setEnabled(quiet)
        self.open_action.setEnabled(not dialog._busy)
        self.resume_action.setEnabled(self.loaded is not None and not dialog._busy)

    def _ui_state(self):
        dialog = self.dialog
        state = {
            "version": 1,
            "goal": dialog.goal_edit.toPlainText(),
            "chat": [list(row) for row in dialog._chat],
            "replies": [
                dict(index=index, **asdict(reply))
                for index, reply in dialog._chat_replies.items()
            ],
            "clarifications": [list(row) for row in dialog._clarifications],
            "drafts": [asdict(draft) for draft in dialog._drafts],
            "draft_index": dialog.draft_combo.currentIndex(),
            "discussion_draft": dialog.discussion_edit.toPlainText(),
            "pause_answer_draft": dialog._investigation.pause_answer.text(),
            "initiative": dialog.initiative_slider.value(),
            "source_id": dialog.source_combo.currentData(),
            "stage": dialog.stage_tabs.currentIndex(),
        }
        return json.loads(json.dumps(state, allow_nan=False))

    def save(self):
        if not self.save_action.isEnabled():
            return
        path, _ = QFileDialog.getSaveFileName(
            self.dialog,
            "Save assistant session locally",
            "assistant-session.vipp-session",
            "VIPP assistant session (*.vipp-session)",
            options=QFileDialog.DontConfirmOverwrite,
        )
        if not path:
            return
        if not path.endswith(".vipp-session"):
            path += ".vipp-session"
        overwrite_authorized = False
        if Path(path).exists():
            if (
                QMessageBox.question(
                    self.dialog,
                    "Replace this saved session?",
                    "Replace this exact local session file?\n\n"
                    + str(Path(path).absolute()),
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                != QMessageBox.Yes
            ):
                return
            overwrite_authorized = True
        description = (
            "This local file keeps the goal, conversation, drafts and any retained "
            "investigation evidence, including image/result arrays and previews. "
            "It does not contain sign-in credentials. Nothing is uploaded. "
            "Store it where you would store the source images. Save this session?"
        )
        if (
            QMessageBox.question(
                self.dialog,
                "Save local session data?",
                description,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            != QMessageBox.Yes
        ):
            return
        try:
            self.save_path(Path(path).absolute(), overwrite=overwrite_authorized)
        except (ValueError, RuntimeError, OSError) as error:
            self.dialog._set_status("error", f"Session was not saved: {error}")

    def save_path(self, path, *, overwrite=False):
        from napari_vipp.core.workflow_assistant_session import (
            make_checkpoint,
            save_assistant_session,
        )
        from napari_vipp.core.workflow_investigation_persistence import (
            checkpoint_investigation,
        )

        controller = self.dialog._investigation
        path = Path(path)
        if not path.is_absolute() or path.suffix != ".vipp-session":
            raise ValueError("Choose an absolute .vipp-session destination.")
        if self.assert_destination is not None:
            self.assert_destination(path)
        if not self.save_action.isEnabled() or controller._worker_kind is not None:
            raise ValueError(
                "Pause the assistant and wait for its current step before saving."
            )
        if controller.session is None and self.loaded is not None:
            document = self.loaded.document
            document["ui_state"] = self._ui_state()
            checkpoint = make_checkpoint(
                document,
                {name: self.loaded.blob_bytes(name) for name in self.loaded.blob_names},
            )
        else:
            checkpoint = checkpoint_investigation(
                controller.session, ui_state=self._ui_state()
            )
        self.last_receipt = save_assistant_session(
            path, checkpoint, overwrite=overwrite
        )
        resume = checkpoint.document.get("resume", {})
        if controller.session is not None and resume.get("supported") is False:
            self.dialog._set_status(
                "warning",
                "Session saved locally as history only. "
                + str(resume.get("reason", "Scientific continuation is unavailable.")),
            )
            return self.last_receipt
        self.dialog._set_status(
            "success",
            "Assistant session saved locally. Credentials and execution "
            "permissions are not saved; reopening sends nothing.",
        )
        return self.last_receipt

    def open(self):
        if not self.open_action.isEnabled():
            return
        path, _ = QFileDialog.getOpenFileName(
            self.dialog,
            "Open saved assistant session",
            "",
            "VIPP assistant session (*.vipp-session)",
        )
        if not path:
            return
        if (
            self.dialog._chat
            and QMessageBox.question(
                self.dialog,
                "Replace this assistant conversation?",
                "Open replaces the conversation shown here, not your workflow. "
                "Save this conversation first if you need it. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            != QMessageBox.Yes
        ):
            return
        try:
            self.load_path(Path(path).absolute())
        except (ValueError, RuntimeError, OSError) as error:
            self.dialog._set_status(
                "error", f"Saved session could not be opened: {error}"
            )

    def load_path(self, path):
        from napari_vipp.core.workflow_assistant_session import load_assistant_session
        from napari_vipp.ui.workflow_assistant import _ChatReply, _Draft

        dialog = self.dialog
        if dialog._busy:
            raise ValueError(
                "Stop or finish current work before opening another session."
            )
        checkpoint = load_assistant_session(path)
        document = checkpoint.document
        state = _validated_ui(document.get("ui_state", {}))
        # Validate all presentation fields before changing the current window.
        drafts = [
            _Draft(
                row["summary"],
                tuple(tuple(step) for step in row["steps"]),
                tuple(row["assumptions"]),
                tuple(row["questions"]),
                row["proposal_text"],
                row["plan_text"],
            )
            for row in state["drafts"]
        ]
        replies = {
            row["index"]: _ChatReply(row["question"], row["answer"], row["is_question"])
            for row in state["replies"]
        }
        dialog.cancel_request(announce=False)
        dialog._invalidate_request()
        dialog._investigation.session = None
        dialog._investigation.agent_loop = None
        dialog._investigation._existing_bundle = None
        dialog._investigation._warm_bundle = None
        dialog._investigation._warm_snapshot = None
        # An archive never re-enables image sharing or restores a provider/MCP lease.
        dialog.images_checkbox.setChecked(False)
        dialog.output_assessment_checkbox.setChecked(False)
        dialog.consent_checkbox.setChecked(False)
        dialog.goal_edit.setPlainText(state["goal"])
        dialog.initiative_slider.setValue(state["initiative"])
        index = dialog.source_combo.findData(state["source_id"])
        if index >= 0:
            dialog.source_combo.setCurrentIndex(index)
        dialog.discussion_edit.setPlainText(state["discussion_draft"])
        dialog._clarifications = [tuple(row) for row in state["clarifications"]]
        dialog._chat = [tuple(row) for row in state["chat"]]
        dialog._chat_replies = replies
        dialog._drafts = drafts
        dialog._current_draft_index = -1
        dialog._draft_authorized = False
        dialog._context = dialog._owner = None
        dialog._image_draft_guard = None
        dialog._invalidate_proposal()
        dialog.draft_combo.blockSignals(True)
        dialog.draft_combo.clear()
        for number in range(len(drafts)):
            dialog.draft_combo.addItem(f"Draft {number + 1} · Saved history")
        dialog.draft_combo.blockSignals(False)
        if drafts:
            dialog.draft_combo.setCurrentIndex(
                min(max(state["draft_index"], 0), len(drafts) - 1)
            )
            dialog._draft_selected(dialog.draft_combo.currentIndex())
        else:
            dialog._clear_review()
        dialog._update_chat()
        dialog._investigation.pause_answer.setText(state["pause_answer_draft"])
        dialog.stage_tabs.setCurrentIndex(state["stage"])
        self.loaded = checkpoint
        dialog._set_status(
            "info",
            "Saved history opened locally. Nothing was run or sent. "
            "To continue an investigation, confirm the input and sharing choices, "
            "then choose Session → Resume saved investigation.",
        )
        dialog._sync_buttons()
        return checkpoint

    def resume(self):
        dialog = self.dialog
        if self.loaded is None or dialog._busy:
            return
        if dialog._provider() == "example" or (
            dialog._provider() == "codex" and not dialog._codex_authenticated
        ):
            dialog._set_status(
                "warning", "Choose a connected assistant in Settings first."
            )
            return
        if not dialog.output_assessment_checkbox.isChecked():
            dialog._set_status(
                "warning",
                "Choose Assess workflow outputs in Goal before continuing. "
                "Input previews remain a separate choice. Nothing was sent.",
            )
            dialog.stage_tabs.setCurrentIndex(0)
            return
        if (
            QMessageBox.question(
                dialog,
                "Continue this saved investigation?",
                "This uses your current assistant and sharing choices. The saved "
                "evidence, goal and remaining allowance will be verified against the "
                "current input/workflow first. Nothing is calculated or sent until "
                "you then choose Continue. Prepare this continuation?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            != QMessageBox.Yes
        ):
            return
        try:
            self.prepare_resume()
        except (ValueError, RuntimeError, OSError) as error:
            dialog._set_status(
                "warning", f"History is kept, but cannot continue: {error}"
            )

    def prepare_resume(self):
        from napari_vipp.core.workflow_agent_loop import WorkflowAgentLoop
        from napari_vipp.core.workflow_investigation_persistence import (
            restore_investigation,
        )

        dialog = self.dialog
        controller = dialog._investigation
        if self.loaded is None or dialog._busy:
            raise ValueError("Open a saved session while no work is running first.")
        saved_goal = self.loaded.document.get("ui_state", {}).get("goal")
        if saved_goal is not None and saved_goal != dialog.goal_edit.toPlainText():
            raise ValueError(
                "The goal changed. Start a new investigation for the new goal."
            )
        original_goal = self.loaded.document.get("dialogue", {}).get("objective")
        if (
            original_goal is not None
            and original_goal != dialog.goal_edit.toPlainText()
        ):
            raise ValueError(
                "These trials belong to a different goal. Start a new investigation."
            )
        context, owner = dialog._capture(
            dialog.source_combo.currentData(),
            include_images=dialog.images_checkbox.isChecked(),
        )
        pipeline, compute = controller.capture(context, owner)
        cancelled = threading.Event()
        result = restore_investigation(
            self.loaded,
            pipeline,
            context.source_id,
            compute_request=compute,
            assert_current=lambda: _not_cancelled(cancelled),
            include_input_images=dialog.images_checkbox.isChecked(),
            include_output_images=dialog.output_assessment_checkbox.isChecked(),
            cancelled=cancelled.is_set,
        )
        if not result.resumable:
            raise ValueError(result.reason)
        settings = controller._provider_settings()
        controller.session = result.session
        controller.agent_loop = WorkflowAgentLoop(result.session)
        controller._cancelled = cancelled
        controller._context, controller._owner = context, owner
        controller._compute = compute
        controller._settings = settings
        controller._external = False
        controller._existing_bundle = controller._warm_bundle = None
        controller._warm_snapshot = controller._warm_owner = (
            controller._warm_binding
        ) = None
        controller._worker_kind = None
        controller._review_pending = False
        controller._pause_after_model = False
        controller._epoch += 1
        dialog._context, dialog._owner = context, owner
        dialog._request_objective = controller.session.dialogue
        dialog.consent_checkbox.setChecked(True)
        controller._binding_value = controller._binding()
        controller.active, dialog._busy = True, True
        if controller.session.pending_finish_review is not None:
            controller._show_finish_review()
        else:
            # This question was verified from the archived scientific journal.
            # Its saved conversation and unsent answer are already on screen.
            # Do not append it again or clear its draft as if it were a new ask.
            controller._pending_question = controller.session.continuation_state.get(
                "question", ""
            )
            controller._show_pause()
        controller.timer.start()
        dialog._sync_buttons()
        return result


def _not_cancelled(event):
    if event.is_set():
        raise ValueError("Saved continuation was cancelled.")


def _validated_ui(state):
    """An archive can supply literal history, never objects/configuration/code."""
    defaults = dict(
        version=1,
        goal="",
        chat=[],
        replies=[],
        clarifications=[],
        drafts=[],
        draft_index=-1,
        discussion_draft="",
        pause_answer_draft="",
        initiative=2,
        source_id=None,
        stage=1,
    )
    if type(state) is not dict or set(state) - set(defaults):
        raise ValueError("Unsupported saved conversation fields.")
    state = {**defaults, **state}

    def text(value, maximum=262144):
        if type(value) is not str or len(value) > maximum:
            raise ValueError("Invalid or oversized saved conversation text.")

    text(state["goal"], 4000)
    text(state["discussion_draft"])
    text(state["pause_answer_draft"])
    if (
        state["version"] != 1
        or type(state["stage"]) is not int
        or not 0 <= state["stage"] <= 2
    ):
        raise ValueError("Unsupported saved conversation version/stage.")
    if type(state["initiative"]) is not int or not 0 <= state["initiative"] <= 2:
        raise ValueError("Invalid saved judgment choice.")
    if state["source_id"] is not None:
        text(state["source_id"], 128)
    for name in ("chat", "clarifications"):
        rows = state[name]
        if type(rows) is not list or len(rows) > 2000:
            raise ValueError("Saved conversation exceeds its limit.")
        for row in rows:
            if type(row) is not list or len(row) != 2:
                raise ValueError("Invalid saved conversation row.")
            for value in row:
                text(value)
    if type(state["replies"]) is not list or len(state["replies"]) > len(state["chat"]):
        raise ValueError("Invalid saved reply list.")
    seen = set()
    for row in state["replies"]:
        if type(row) is not dict or set(row) != {
            "index",
            "question",
            "answer",
            "is_question",
        }:
            raise ValueError("Invalid saved reply attribution.")
        index = row["index"]
        if (
            type(index) is not int
            or not 0 <= index < len(state["chat"])
            or index in seen
        ):
            raise ValueError("Invalid saved reply index.")
        seen.add(index)
        text(row["question"])
        text(row["answer"])
        if type(row["is_question"]) is not bool:
            raise ValueError("Invalid saved reply kind.")
    if type(state["drafts"]) is not list or len(state["drafts"]) > 100:
        raise ValueError("Too many saved plan drafts.")
    for row in state["drafts"]:
        if type(row) is not dict or set(row) != {
            "summary",
            "steps",
            "assumptions",
            "questions",
            "proposal_text",
            "plan_text",
        }:
            raise ValueError("Invalid saved plan draft.")
        for key in ("summary", "proposal_text", "plan_text"):
            text(row[key])
        for key in ("assumptions", "questions", "steps"):
            if type(row[key]) is not list or len(row[key]) > 128:
                raise ValueError("Invalid saved draft content.")
        for key in ("assumptions", "questions"):
            for value in row[key]:
                text(value)
        for step in row["steps"]:
            if type(step) is not list or len(step) != 2:
                raise ValueError("Invalid saved plan step.")
            for value in step:
                text(value)
    if type(state["draft_index"]) is not int:
        raise ValueError("Invalid saved draft index.")
    return state
