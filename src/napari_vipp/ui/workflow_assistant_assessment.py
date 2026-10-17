"""Explicit, run-bound output review for the Workflow Assistant window."""

from __future__ import annotations

import html
import json
import threading
from dataclasses import dataclass, replace

from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from napari_vipp.ui.toolbar_controls import toolbar_icon
from napari_vipp.ui.workflow_assistant_details import AssistantDetailsEditor
from napari_vipp.ui.workflow_assistant_style import (
    AssistantButton,
    AssistantSendButton,
    SharingPreviewDialog,
)


def _label(text=""):
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(Qt.PlainText)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return label


@dataclass(frozen=True)
class _ReviewReply:
    context: object
    owner: object
    objective: object
    choices: tuple
    result: object = None
    previous_record: object = None


class AssistantAssessmentController:
    """Keep output evidence separate from plan/apply authority and chat text."""

    def __init__(self, dialog, capture, assert_current):
        self.dialog = dialog
        self.capture = capture
        self.assert_current = assert_current
        self.candidate_ids = ()
        self.source_id = None
        self.run_id = None
        self.reply = None
        self.previous_record = None
        self.comparison_record = None
        self.revision_requested = False
        self.card = QFrame()
        self.card.setObjectName("AssistantAssessmentCard")
        layout = QVBoxLayout(self.card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        title = _label("Review the actual results")
        title.setObjectName("AssistantSectionTitle")
        layout.addWidget(title)
        self.note = _label()
        layout.addWidget(self.note)
        self.report = QLabel()
        self.report.setWordWrap(True)
        self.report.setTextFormat(Qt.RichText)
        self.report.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.report)
        actions = QHBoxLayout()
        self.preview_button = AssistantButton("Preview results shared")
        self.assess_button = AssistantSendButton("Assess results")
        self.revise_button = AssistantSendButton("Revise using results")
        self.preview_button.setProperty("assistantAction", "secondary")
        self.assess_button.setProperty("assistantAction", "primary")
        self.revise_button.setProperty("assistantAction", "secondary")
        for button in (self.preview_button, self.assess_button):
            actions.addWidget(button)
        actions.addStretch(1)
        layout.addLayout(actions)
        revision_actions = QHBoxLayout()
        revision_actions.addWidget(self.revise_button)
        revision_actions.addStretch(1)
        layout.addLayout(revision_actions)
        for button in (self.assess_button, self.revise_button):
            button.set_idle_icon(toolbar_icon("send", dialog.palette()))
        layout.addWidget(
            _label(
                "Assess sends this run's measurements and selected previews to your "
                "assistant. Revise drafts a new plan; it does not add or run it."
            )
        )
        self.preview_button.clicked.connect(lambda: self.start(preview=True))
        self.assess_button.clicked.connect(lambda: self.start(preview=False))
        self.revise_button.clicked.connect(self.revise)

    def _choices(self):
        d = self.dialog
        return (
            d.source_combo.currentData(),
            d.goal_edit.toPlainText(),
            d._provider(),
            d.images_checkbox.isChecked(),
            d.output_assessment_checkbox.isChecked(),
            tuple(d._clarifications),
            tuple((question, field.text()) for question, field in d._question_fields),
            d.discussion_edit.toPlainText(),
            d.initiative_slider.value(),
        )

    def invalidate(self, message="The results changed. Run the current plan again."):
        d = self.dialog
        if d._busy and d._job_kind.startswith("assessment"):
            d.cancel_request(announce=False)
        self.run_id = None
        self.reply = None
        self.revision_requested = False
        self.report.clear()
        self.note.setText(message)

    def permissions_changed(self, *_):
        if hasattr(self.dialog, "_refinement"):
            self.dialog._refinement.context_changed()
        # Withdraw output-backed revision authority as well as pending review.
        context = self.dialog._context
        if self.revision_requested or (
            context is not None and context._result_feedback is not None
        ):
            self.dialog.cancel_request(announce=False)
            self.dialog._context = self.dialog._owner = None
            self.dialog._invalidate_proposal()
            self.dialog._draft_authorized = False
        self.invalidate(
            "Output assessment is off. No output data will be sent."
            if not self.dialog.output_assessment_checkbox.isChecked()
            else "Add and run a fresh plan with output assessment on, then assess it."
        )
        self.dialog._sync_buttons()

    def input_permissions_changed(self):
        if hasattr(self.dialog, "_refinement"):
            self.dialog._refinement.context_changed()
        if self.dialog._busy and self.dialog._job_kind.startswith("assessment"):
            self.dialog.cancel_request(announce=False)
        self.reply = None
        self.revision_requested = False
        self.report.clear()
        if self.run_id:
            self.note.setText(
                "Preview sharing changed. Assess the results with these choices."
            )

    def note_run(self, candidate_ids, source_id):
        self.invalidate(
            "Calculating the plan. Results can be assessed after it finishes."
        )
        self.candidate_ids, self.source_id = tuple(candidate_ids), source_id

    def results_ready(self, candidate_ids, run_id):
        if (
            tuple(candidate_ids) != self.candidate_ids
            or not self.dialog.output_assessment_checkbox.isChecked()
        ):
            return
        self.run_id = run_id
        self.note.setText("Results are ready. Assess how well this run met your goal.")
        self.dialog._sync_buttons()

    def sync(self):
        d = self.dialog
        enabled = d.output_assessment_checkbox.isChecked()
        self.card.setVisible(enabled)
        connected = d._provider() != "codex" or (
            d._codex_authenticated and not d._connection_busy
        )
        supported = d._provider() not in {"example", "openai"}
        ready = (
            enabled
            and bool(self.run_id)
            and callable(self.capture)
            and callable(self.assert_current)
            and not d._busy
            and supported
            and connected
            and d.source_combo.currentData() == self.source_id
        )
        self.preview_button.setEnabled(bool(ready))
        self.assess_button.setEnabled(bool(ready))
        self.revise_button.setEnabled(bool(ready and self.reply is not None))
        api_hint = (
            "Use Start refinement to review and improve these results with "
            "OpenAI API's shared investigation tools."
            if d._provider() == "openai"
            else ""
        )
        for button in (self.preview_button, self.assess_button, self.revise_button):
            button.setToolTip(api_hint)
        if enabled and api_hint:
            self.note.setText(api_hint)
        sending = d._busy and d._job_kind == "assessment"
        self.assess_button.set_sending(sending)
        self.assess_button.setText(
            "Assessing results…" if sending else "Assess results"
        )
        if enabled and not self.run_id and not self.note.text():
            self.note.setText(
                "Add and run a plan, then assess its actual outputs here."
            )

    def start(self, *, preview):
        d = self.dialog
        if d._busy or not d.output_assessment_checkbox.isChecked() or not self.run_id:
            return
        try:
            if not callable(self.capture) or not callable(self.assert_current):
                raise ValueError("Output review is unavailable in this window.")
            snapshot, owner = self.capture(self.candidate_ids, self.source_id)
            self.assert_current(snapshot, owner)
            preservation, preservation_result = d._refinement.assessment_preservation(
                snapshot
            )
            if snapshot.run_id != self.run_id:
                raise ValueError(
                    "The completed run changed. Run the current plan again."
                )
            objective = d._public_output_objective(d._answered_objective())
            choices = self._choices()
            provider = d._provider()
            if provider == "codex":
                if not d._codex_authenticated or d._connection_busy:
                    raise ValueError("Connect to ChatGPT before assessing results.")
                from napari_vipp.core.workflow_proposal_codex import CodexSettings

                settings = CodexSettings(
                    model=d.codex_model_edit.text().strip(),
                    reasoning_effort=d.reasoning_combo.currentText(),
                    max_attempts=d.attempts_spin.value(),
                    timeout_seconds=d._refinement.timeout_seconds(
                        d.timeout_spin.value(), d.attempts_spin.value()
                    ),
                )
            elif provider == "local":
                from napari_vipp.core.workflow_proposal_model import ModelSettings

                settings = ModelSettings(
                    kind=d._kind(),
                    base_url=d.address_edit.text().strip(),
                    model=d.model_combo.currentText().strip(),
                    max_attempts=d.attempts_spin.value(),
                    timeout_seconds=d._refinement.timeout_seconds(
                        d.timeout_spin.value(), d.attempts_spin.value()
                    ),
                )
            else:
                raise ValueError(
                    "Choose ChatGPT or a local model for output assessment."
                )
        except (ValueError, RuntimeError) as exc:
            self.invalidate(str(exc))
            d._set_status("warning", str(exc))
            d._sync_buttons()
            return
        d._cancelled = threading.Event()
        cancelled = d._cancelled.is_set
        generation, signals = d._generation + 1, d._signals
        previous_record = self.previous_record
        if previous_record is not None and previous_record.run_id == snapshot.run_id:
            previous_record = self.comparison_record

        def work():
            from napari_vipp.core.workflow_assessment import build_assessment_context
            from napari_vipp.core.workflow_assessment_model import (
                generate_assessment,
                generate_codex_assessment,
            )

            context = build_assessment_context(
                snapshot,
                include_input_images=choices[3],
                include_output_images=provider == "codex",
                cancelled=cancelled,
            )
            if preservation is not None:
                context = preservation.with_assessment(
                    context,
                    include_output_images=provider == "codex",
                    result=preservation_result,
                )
            if previous_record is not None:
                from napari_vipp.core.workflow_assessment_history import (
                    with_previous_assessment,
                )

                context = with_previous_assessment(context, objective, previous_record)
            if cancelled():
                raise ValueError("Output review was cancelled.")
            result = None
            if not preview:
                generate = (
                    generate_codex_assessment
                    if provider == "codex"
                    else generate_assessment
                )
                result = generate(
                    context,
                    objective,
                    settings,
                    cancelled=cancelled,
                    progress=lambda message: signals.progress.emit(
                        generation, str(message)
                    ),
                )
            return _ReviewReply(
                context, owner, objective, choices, result, previous_record
            )

        d._start("assessment_preview" if preview else "assessment", work)

    def ready(self, reply):
        d = self.dialog
        try:
            if reply.choices != self._choices() or not reply.choices[4]:
                raise ValueError(
                    "Sharing choices changed. The late review was not used."
                )
            self.assert_current(reply.context, reply.owner)
            if reply.context.run_id != self.run_id:
                raise ValueError("This review belongs to an older run.")
        except (ValueError, RuntimeError) as exc:
            self.invalidate(str(exc))
            d._set_status("warning", str(exc))
            return
        if reply.result is None:
            self.preview_dialog(reply).exec()
            d._set_status("info", "Result preview prepared locally. Nothing was sent.")
            return
        self.reply = reply
        self.comparison_record = reply.previous_record
        report = reply.result.assessment
        from napari_vipp.core.workflow_assessment_history import (
            make_previous_assessment,
        )

        try:
            self.previous_record = make_previous_assessment(
                reply.context, reply.objective, report
            )
        except (ValueError, RuntimeError):
            # Even a faulty adapter must not turn unvalidated display text into
            # feedback authority. It can still be displayed safely as escaped text.
            self.previous_record = None
            self.comparison_record = None
            self.reply = None
        d._refinement.record_report(report)
        d.request_edit.setPlainText(
            json.dumps(
                reply.context.public_request(reply.objective),
                indent=2,
                ensure_ascii=False,
            )
        )
        names = {
            "achieved": "Goal appears met",
            "partly_achieved": "Goal partly met",
            "not_achieved": "Goal not met",
            "cannot_assess": "More evidence needed",
        }
        checks = {
            "met": "Met",
            "partly_met": "Partly met",
            "not_met": "Not met",
            "cannot_assess": "Cannot assess",
        }
        esc = html.escape
        parts = [f"<h3>{names[report.overall]}</h3><p>{esc(report.summary)}</p>"]
        comparison = getattr(report, "comparison", None)
        if comparison is not None:
            parts.append(
                "<p><b>Change since the previous run</b><br>"
                + esc(comparison.status.replace("_", " "))
                + " — "
                + esc(comparison.explanation)
                + "</p>"
            )
        for check in report.goal_checks:
            parts.append(
                f"<p><b>{esc(check.criterion)} — {checks[check.status]}</b><br>"
                f"{esc(check.evidence)}</p>"
            )
        for heading, values in (
            ("Observed results", report.findings),
            ("What to try next", report.suggested_changes),
            ("Limits of this assessment", report.limitations),
        ):
            if values:
                parts.append(
                    f"<p><b>{heading}</b></p><ul>"
                    + "".join(f"<li>{esc(value)}</li>" for value in values)
                    + "</ul>"
                )
        self.report.setText("".join(parts))
        self.note.setText(
            "Assessment of the latest completed run · assistant interpretation"
        )
        d._append_chat(
            "Assistant",
            "Output assessment: "
            + report.summary
            + "\n"
            + "\n".join(
                check.criterion + ": " + check.evidence for check in report.goal_checks
            ),
        )
        d.trace_edit.setPlainText(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
        )
        d._set_status(
            "success" if report.overall == "achieved" else "warning",
            "Results assessed. Review the evidence, or revise using results.",
        )

    def failed(self, error):
        d = self.dialog
        self.reply = None
        self.revision_requested = False
        self.report.clear()
        self.note.setText("The results were not assessed. Your workflow is unchanged.")
        d._error_diagnostic = d._safe_diagnostic(error)
        d.copy_error_button.setEnabled(d._error_diagnostic is not None)
        d.trace_edit.setPlainText(d._attempt_trace(error))
        d._set_status(
            "error",
            "Could not assess these results. Your workflow is unchanged. "
            "Check the connection or safe error details, then Assess results again.",
        )

    def preview_dialog(self, reply):
        d = self.dialog
        preview = SharingPreviewDialog(d, d._theme_probe.palette())
        panel = QWidget()
        content = QVBoxLayout(panel)
        content.setContentsMargins(14, 12, 14, 12)
        content.addWidget(
            _label(
                "This run's bounded measurements, goal and discussion are shared. "
                "Only the previews shown below are included. "
                "No raw volume or source path is sent."
            )
        )
        for item in reply.context.image_previews:
            content.addWidget(_label(json.dumps(item.manifest(), ensure_ascii=False)))
            pixels = QPixmap()
            pixels.loadFromData(item.png_bytes, "PNG")
            image = QLabel()
            image.setPixmap(pixels)
            image.setAlignment(Qt.AlignCenter)
            content.addWidget(image)
        details = AssistantDetailsEditor(json_syntax=True)
        details.setReadOnly(True)
        details.setPlainText(
            json.dumps(
                reply.context.public_request(reply.objective),
                indent=2,
                ensure_ascii=False,
            )
        )
        details.setMinimumHeight(220)
        content.addWidget(details)
        preview.set_content(panel)
        return preview

    def revise(self):
        d = self.dialog
        if (
            d._busy
            or self.reply is None
            or not d.output_assessment_checkbox.isChecked()
        ):
            return
        try:
            self.assert_current(self.reply.context, self.reply.owner)
            if self.reply.choices != self._choices():
                raise ValueError(
                    "The goal or sharing choices changed. Assess the current run again."
                )
        except (ValueError, RuntimeError) as exc:
            self.invalidate(str(exc))
            d._set_status("warning", str(exc))
            d._sync_buttons()
            return
        self.revision_requested = True
        d._context = d._owner = None
        d._image_draft_guard = None
        # The feedback is machine-provided evidence, not another user answer.
        # Keep the literal objective/answers stable across comparison rounds.
        try:
            objective = d._answered_objective()
            d.consent_checkbox.setChecked(True)
            d._sending_action = "message"
            d._submitted_message = ""
            d._ensure_context(objective, approved_request=True)
            from napari_vipp.core.workflow_proposals import ProposalDialogue

            d._pending_clarifications = (
                list(objective.clarifications)
                if isinstance(objective, ProposalDialogue)
                else []
            )
            d._append_chat(
                "VIPP",
                "Requesting a revised plan using the actual results and assessment.",
            )
            d._send()
        except (ValueError, RuntimeError) as exc:
            d._set_status("warning", str(exc))
        d._sync_buttons()

    def inject_feedback(self, context):
        if not self.revision_requested or self.reply is None:
            # A cached proposal context must never resurrect a previous output
            # permission, including after another assessment request failed.
            if context._result_feedback is not None:
                return replace(
                    context,
                    image_previews=(),
                    _result_feedback=None,
                    _result_feedback_guard=None,
                )
            return context
        if not self.dialog.output_assessment_checkbox.isChecked():
            raise ValueError("Output sharing is off; result feedback was not sent.")
        self.assert_current(self.reply.context, self.reply.owner)
        return replace(
            context,
            image_previews=self.reply.context.image_previews,
            _result_feedback=self.reply.context.revision_feedback(
                self.reply.result.assessment.to_dict()
            ),
            _result_feedback_guard=self.reply.context.snapshot,
        )
