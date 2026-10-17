"""Explicit Goal-started investigation; workers never touch live Qt/graph state."""

from __future__ import annotations

import hashlib
import html
import json
import threading
from dataclasses import replace

from qtpy.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
)

from napari_vipp.core.workflow_agent_loop import WorkflowAgentLoop
from napari_vipp.core.workflow_investigation import (
    InvestigationSession,
)
from napari_vipp.core.workflow_proposal_codex import CodexSettings
from napari_vipp.core.workflow_proposal_images import (
    MAX_IMAGE_PREVIEWS,
    MAX_PREVIEW_TOTAL_BYTES,
)
from napari_vipp.core.workflow_proposals import ProposalDialogue
from napari_vipp.ui.workflow_assistant_effort import AssistantEffortControls
from napari_vipp.ui.workflow_assistant_style import (
    AssistantButton,
    AssistantDisclosureButton,
)


def _label(text=""):
    widget = QLabel(text)
    widget.setWordWrap(True)
    widget.setTextFormat(Qt.PlainText)
    return widget


class _Signals(QObject):
    ready = Signal(int, str, object)
    failed = Signal(int, object)
    progress = Signal(int, str)


class _Work(QRunnable):
    def __init__(self, epoch, kind, call, signals):
        super().__init__()
        self.epoch, self.kind, self.call, self.signals = epoch, kind, call, signals

    def run(self):
        try:
            result = self.call()
        except Exception as error:
            self.signals.failed.emit(self.epoch, error)
        else:
            self.signals.ready.emit(self.epoch, self.kind, result)


class AssistantInvestigationController:
    """One local authority check surrounds every detached model/action step."""

    def __init__(self, dialog, capture=None, current_compute=None):
        self.dialog, self.capture, self.current_compute = (
            dialog,
            capture,
            current_compute,
        )
        self.session = None
        self.agent_loop = None
        self.active = False
        self._epoch = 0
        self._cancelled = threading.Event()
        self._context = self._owner = self._compute = self._binding_value = None
        self._pending_question = ""
        self._events = []
        self.model_records = []
        self._settings = None
        self._warm_snapshot = self._warm_owner = self._warm_binding = None
        self._warm_bundle = None
        self._selected_draft = None
        self._review_pending = False
        self._paused = False
        self._external = False
        self._existing_bundle = None
        self._worker_kind = None
        self._pause_after_model = False
        self.signals = _Signals()
        self.signals.ready.connect(self._ready)
        self.signals.failed.connect(self._failed)
        self.signals.progress.connect(self._progress)
        self.timer = QTimer(dialog)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self._tick)

        self.options = QFrame()
        self.options.setObjectName("AssistantInvestigationOptions")
        layout = QVBoxLayout(self.options)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)
        row = QHBoxLayout()
        self.checkbox = QCheckBox("Investigate with trial runs")
        self.checkbox.setAccessibleDescription(
            "Optional. Start investigation lets the assistant inspect and calculate "
            "detached candidates within the displayed limits. It changes no accepted "
            "workflow and saves no files. Uses the independent sharing choices above."
        )
        row.addWidget(self.checkbox, 1)
        row.addWidget(_label("Up to"))
        self.max_attempts = QSpinBox()
        self.max_attempts.setRange(1, 5)
        self.max_attempts.setValue(3)
        self.max_attempts.setAccessibleName("Maximum investigation trial attempts")
        row.addWidget(self.max_attempts)
        row.addWidget(_label("trial runs"))
        layout.addLayout(row)
        self.description = _label(
            "Inspect → test an idea → compare → select. Start allows up to 32 actions, "
            "40 model calls and 30 minutes; failed trials count. Up to 6 small "
            "diagnostic tests can compare ideas before a full trial (at most "
            "262,144 input voxels and 8 nodes each), within those same limits. "
            "Your workflow "
            "changes only with Add or Apply changes. Trial runs save no outputs."
        )
        self.description.setObjectName("AssistantMuted")
        layout.addWidget(self.description)
        self.review_checkbox = QCheckBox("Review before finishing")
        self.review_checkbox.setChecked(True)
        self.review_checkbox.setAccessibleDescription(
            "Pause before a proposed final choice so you can keep improving within "
            "the original limits, accept that exact result for plan review, or stop. "
            "Changing this option starts nothing."
        )
        layout.addWidget(self.review_checkbox)
        self.existing_checkbox = QCheckBox("Start from the current completed workflow")
        self.existing_checkbox.setChecked(True)
        self.existing_checkbox.setToolTip(
            "Use ready cached results downstream of the chosen input as the starting "
            "candidate. Uncalculated, multi-source or unsupported branches must be "
            "resolved first. Trials stay detached; Add creates a separate branch."
        )
        layout.addWidget(self.existing_checkbox)
        self.existing_checkbox.toggled.connect(self._choice_changed)
        self.effort = AssistantEffortControls()
        layout.addWidget(self.effort)
        for control in self.effort.controls:
            control.valueChanged.connect(self._choice_changed)
        self.checkbox.toggled.connect(self._choice_changed)
        self.max_attempts.valueChanged.connect(self._choice_changed)
        self.review_checkbox.toggled.connect(self._choice_changed)
        for control in (
            dialog.consent_checkbox,
            dialog.images_checkbox,
            dialog.output_assessment_checkbox,
        ):
            control.toggled.connect(self._permissions_changed)

        self.card = QFrame()
        self.card.setObjectName("AssistantInvestigationCard")
        body = QVBoxLayout(self.card)
        body.setContentsMargins(14, 10, 14, 10)
        body.setSpacing(8)
        heading = _label("Investigation")
        heading.setObjectName("AssistantSectionTitle")
        body.addWidget(heading)
        self.note = _label("No investigation started.")
        body.addWidget(self.note)
        self.pause_button = AssistantButton("Pause after this step")
        self.pause_button.setAutoDefault(False)
        self.pause_button.clicked.connect(self._pause)
        body.addWidget(self.pause_button)
        self.pause_card = QFrame()
        self.pause_card.setObjectName("AssistantQuestions")
        pause_layout = QVBoxLayout(self.pause_card)
        pause_layout.setContentsMargins(14, 12, 14, 12)
        pause_heading = _label("Continue your investigation")
        pause_heading.setObjectName("AssistantSectionTitle")
        pause_layout.addWidget(pause_heading)
        self.pause_note = _label()
        pause_layout.addWidget(self.pause_note)
        self.pause_answer = QLineEdit()
        self.pause_answer.setMaxLength(1000)
        self.pause_answer.setPlaceholderText(
            "Your answer, when clarification is needed…"
        )
        self.pause_answer.setAccessibleName("Answer the paused investigation question")
        pause_layout.addWidget(self.pause_answer)
        pause_actions = QHBoxLayout()
        self.resume_button = AssistantButton("Continue")
        self.resume_button.clicked.connect(self._resume)
        pause_actions.addWidget(self.resume_button)
        self.allowance_button = AssistantButton("Adjust allowance…")
        self.allowance_button.clicked.connect(self._adjust_allowance)
        pause_actions.addWidget(self.allowance_button)
        pause_layout.addLayout(pause_actions)
        self.pause_card.hide()
        body.addWidget(self.pause_card)
        self.review_card = QFrame()
        self.review_card.setObjectName("AssistantFinishReview")
        review = QVBoxLayout(self.review_card)
        review.setContentsMargins(0, 4, 0, 4)
        review.setSpacing(8)
        self.review_title = _label()
        self.review_title.setObjectName("AssistantSectionTitle")
        review.addWidget(self.review_title)
        self.review_body = QFrame()
        review_body = QVBoxLayout(self.review_body)
        review_body.setContentsMargins(2, 2, 2, 2)
        review_body.setSpacing(8)
        self.review_summary = _label()
        review_body.addWidget(self.review_summary)
        self.review_limitations = _label()
        review_body.addWidget(self.review_limitations)
        self.review_gallery_toggle = AssistantDisclosureButton("Trial images")
        review_body.addWidget(self.review_gallery_toggle)
        self.review_gallery = QFrame()
        self.review_gallery_layout = QVBoxLayout(self.review_gallery)
        self.review_gallery_layout.setContentsMargins(0, 0, 0, 0)
        self.review_gallery_layout.setSpacing(8)
        self.review_gallery_toggle.toggled.connect(self.review_gallery.setVisible)
        self.review_gallery.hide()
        self.review_gallery_images = []
        review_body.addWidget(self.review_gallery)
        self.review_body_area = QScrollArea()
        self.review_body_area.setWidgetResizable(True)
        self.review_body_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.review_body_area.setMinimumHeight(90)
        self.review_body_area.setMaximumHeight(220)
        self.review_body_area.setWidget(self.review_body)
        review.addWidget(self.review_body_area)
        self.review_budget = _label()
        review.addWidget(self.review_budget)
        self.review_allowance_button = AssistantButton("Adjust working allowance…")
        self.review_allowance_button.clicked.connect(self._adjust_allowance)
        review.addWidget(self.review_allowance_button)
        review.addWidget(
            _label("Optional feedback for the next request (up to 1,000 characters)")
        )
        self.review_feedback = QLineEdit()
        self.review_feedback.setMaxLength(1000)
        self.review_feedback.setAccessibleName(
            "Optional feedback for the next improvement request"
        )
        self.review_feedback.setPlaceholderText("What still needs improving?")
        review.addWidget(self.review_feedback)
        self.review_reason = _label()
        review.addWidget(self.review_reason)
        self.review_continue_button = AssistantButton("Keep improving")
        self.review_accept_button = AssistantButton("Good enough — review plan")
        self.review_stop_button = AssistantButton("Stop — keep accepted workflow")
        for button, action in (
            (self.review_continue_button, "primary"),
            (self.review_accept_button, "secondary"),
            (self.review_stop_button, "secondary"),
        ):
            button.setProperty("assistantAction", action)
            button.setMinimumHeight(34)
            button.setAutoDefault(False)
        review_actions = QGridLayout()
        review_actions.addWidget(self.review_continue_button, 0, 0)
        review_actions.addWidget(self.review_accept_button, 0, 1)
        review_actions.addWidget(self.review_stop_button, 1, 0, 1, 2)
        review.addLayout(review_actions)
        self.review_continue_button.clicked.connect(self._continue_finish_review)
        self.review_accept_button.clicked.connect(self._accept_finish_review)
        self.review_stop_button.clicked.connect(
            lambda: self.cancel(
                "Stopped by you. The accepted workflow is kept.", user_decision=True
            )
        )
        self.review_continue_button.setToolTip(
            "Send your optional feedback and continue this same session within its "
            "original limits and sharing choices. No allowance is restarted."
        )
        self.review_accept_button.setToolTip(
            "Accept this exact completed choice, including its limitations, for "
            "plan review. Add and Run remain separate; this is not scientific "
            "validation."
        )
        self.review_card.hide()
        body.addWidget(self.review_card)
        self.history_toggle = AssistantDisclosureButton(
            "What the assistant inspected and tried"
        )
        body.addWidget(self.history_toggle)
        self.history = _label()
        self.history.setTextFormat(Qt.RichText)
        self.history.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.history_area = QScrollArea()
        self.history_area.setWidgetResizable(True)
        self.history_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.history_area.setMaximumHeight(190)
        self.history_area.setWidget(self.history)
        body.addWidget(self.history_area)
        self.history_area.hide()
        self.history_toggle.toggled.connect(self.history_area.setVisible)
        self.card.hide()

    def _binding(self):
        d = self.dialog
        return (
            d.source_combo.currentData(),
            d.goal_edit.toPlainText(),
            d._provider(),
            d.images_checkbox.isChecked(),
            d.output_assessment_checkbox.isChecked(),
            d.initiative_slider.value(),
            d.codex_model_edit.text(),
            d.reasoning_combo.currentText(),
            d.timeout_spin.value(),
            d.attempts_spin.value(),
            self.max_attempts.value(),
            self.checkbox.isChecked(),
            self.review_checkbox.isChecked(),
            self.existing_checkbox.isChecked(),
            self.effort.binding(),
            d.address_edit.text(),
            d.model_combo.currentText(),
            d.local_context_spin.value(),
            d.kind_combo.currentIndex(),
            d.api_model_edit.text(),
            d.api_key_env_edit.text(),
            repr(d._output_destination),
        )

    def _choice_changed(self, *_):
        if self.active:
            self.cancel("Investigation settings changed.")
        self.dialog._sync_buttons()

    def _permissions_changed(self, *_):
        if self.active:
            self.cancel("Sharing choices changed.")

    def sync(self):
        d = self.dialog
        chosen = self.checkbox.isChecked()
        self.options.setVisible(d._provider() != "example")
        self.checkbox.setEnabled(
            not self.active and not d._busy and d._provider() != "openai"
        )
        self.checkbox.setToolTip(
            "OpenAI API uses the shared investigation tools in this version."
            if d._provider() == "openai"
            else "Let the assistant inspect and test ideas before proposing a plan."
        )
        self.max_attempts.setEnabled(chosen and not self.active and not d._busy)
        self.effort.setEnabled(not self.active and not d._busy)
        self.existing_checkbox.setEnabled(not self.active and not d._busy)
        self.review_checkbox.setVisible(
            d._provider() != "example" and (chosen or d._refinement.slider.value() > 0)
        )
        self.review_checkbox.setEnabled(not self.active and not d._busy)
        self.description.setVisible(chosen)
        if chosen and not self.active and not d._busy:
            d.prepare_button.setText("Start investigation")
            if d._provider() == "example" or not callable(self.capture):
                d.prepare_button.setEnabled(False)
                self.description.setText(
                    "Investigation needs an assistant connection and a "
                    "loaded VIPP input. Ordinary planning remains available when "
                    "this option is off."
                )
            elif not d.output_assessment_checkbox.isChecked():
                d.prepare_button.setEnabled(False)
                self.description.setText(
                    "Choose Assess workflow outputs above to let the assistant "
                    "receive trial measurements and result previews. Input previews "
                    "remain a separate choice. Nothing is sent by checking a box."
                )
            else:
                self.description.setText(
                    "Inspect → test an idea → compare → select. "
                    f"Start allows up to {self.effort.actions.value()} actions, "
                    f"{self.effort.calls.value()} model requests and "
                    f"{self.effort.minutes.value()} working minutes; "
                    "failed trials count. Up to 6 "
                    "small diagnostic tests can compare ideas before a full trial "
                    "(at most 262,144 input voxels and 8 nodes each), within those "
                    "same limits. Your workflow "
                    "changes only with Add or Apply changes. "
                    "Trial runs save no outputs."
                )
        elif not self.active and not d._busy and d._provider() != "example":
            d.prepare_button.setText("Send goal")
        if not self.active:
            d.source_combo.setEnabled(not d._busy)
            if not d._refinement.active:
                d.goal_edit.setReadOnly(False)
                d.discussion_edit.setReadOnly(False)
        if self.active:
            d.prepare_button.set_sending(
                not self._review_pending and not self._paused and not self._external
            )
            d.prepare_button.setText(
                "Review before finishing"
                if self._review_pending
                else "Paused"
                if self._paused
                else "External agent connected"
                if self._external
                else "Investigating…"
            )
            for control in (
                d.prepare_button,
                d.send_button,
                d.revise_button,
                d.apply_button,
                d.run_button,
                d.preview_button,
                d.discussion_preview_button,
                d.settings_button,
                d.draft_combo,
                d.source_combo,
                d.initiative_slider,
                d.output_folder_combo,
                d.output_folder_button,
            ):
                control.setEnabled(False)
            d.goal_edit.setReadOnly(True)
            d.discussion_edit.setReadOnly(True)
            d.cancel_button.setEnabled(True)
            if self._review_pending or self._paused or self._external:
                d.request_progress.hide()
                d.request_activity.hide()
                d.request_elapsed_label.hide()
        self.pause_button.setVisible(
            self.active and not self._review_pending and not self._paused
        )

    def _guard(self, *, check_budget=True):
        d = self.dialog
        if not self.active or d._closed or self._cancelled.is_set():
            raise ValueError("Investigation is no longer active.")
        if self._binding() != self._binding_value:
            raise ValueError("Input, goal, image permissions or settings changed.")
        d._assert_current(self._context, self._owner)
        if self._existing_bundle is not None:
            self.capture(
                self._context, self._owner, assert_existing=self._existing_bundle
            )
        if self._warm_snapshot is not None:
            d._assessment.assert_current(self._warm_snapshot, self._warm_owner)
            if d._refinement._binding() != self._warm_binding:
                raise ValueError("The accepted-result refinement choices changed.")
        if callable(self.current_compute) and self.current_compute() != self._compute:
            raise ValueError("The computation settings changed.")
        pending = self.session.pending_finish_review
        # The core owns active-time accounting and resumable budget stops.
        # GUI authority checks must also work while the owner is reviewing.
        if check_budget and pending is None and not self._paused:
            self.session._guard()
        return pending

    def context_changed(self):
        """A question is meaningful only for the goal/input that produced it."""
        self.cancel("The investigation context changed.")
        self._pending_question = ""
        self._selected_draft = None

    def start(self, *, external=False):
        d = self.dialog
        if self.active or d._busy:
            return
        if (d._provider() == "example" and not external) or not callable(self.capture):
            d._set_status(
                "warning", "Choose an assistant and a loaded input to investigate."
            )
            return
        if (
            not external
            and d._provider() == "codex"
            and (not d._codex_authenticated or d._connection_busy)
        ):
            d._set_status("warning", "Confirm the ChatGPT connection before starting.")
            return
        if not d.consent_checkbox.isChecked():
            d._set_status(
                "warning", "Start approves sharing with the selected assistant."
            )
            return
        if not d.output_assessment_checkbox.isChecked():
            d._set_status(
                "warning",
                "Choose Assess workflow outputs before investigating; "
                "trial measurements and previews are not shared without that choice.",
            )
            return
        d.cancel_request(announce=False)
        self._external = external
        self._paused = False
        self.pause_card.hide()
        d._invalidate_proposal()
        self._warm_snapshot = self._warm_owner = self._warm_binding = None
        self._warm_bundle = None
        self._existing_bundle = None
        self._selected_draft = None
        try:
            if self._pending_question:
                try:
                    d._assert_current(self._context, self._owner)
                except (ValueError, RuntimeError):
                    self._pending_question = ""
                    raise
            answer = d._extra_message if self._pending_question else ""
            if answer.strip():
                # This message answers the pending question; do not also add it
                # as an unrelated instruction or trim the user's literal text.
                d._extra_message = ""
            try:
                objective = d._answered_objective()
            finally:
                if answer.strip():
                    d._extra_message = answer
            if answer.strip():
                original = (
                    objective
                    if isinstance(objective, ProposalDialogue)
                    else ProposalDialogue(objective)
                )
                objective = replace(
                    original,
                    clarifications=tuple(
                        {
                            **dict(original.clarifications),
                            self._pending_question: answer,
                        }.items()
                    ),
                )
            d._context = None
            d._ensure_context(objective, approved_request=True)
            self._context, self._owner = d._context, d._owner
            if self.existing_checkbox.isChecked():
                pipeline, self._compute, self._existing_bundle = self.capture(
                    self._context, self._owner, existing_workflow=True
                )
            else:
                pipeline, self._compute = self.capture(self._context, self._owner)
            self._cancelled = threading.Event()
            cancelled = self._cancelled.is_set
            create_session = (
                InvestigationSession
                if self._existing_bundle is None
                else InvestigationSession.from_completed
            )
            starting_binding = (
                self._context.source_id
                if self._existing_bundle is None
                else self._existing_bundle
            )
            self.session = create_session(
                pipeline,
                starting_binding,
                d._request_objective,
                compute_request=self._compute,
                limits=self.effort.limits(self.max_attempts.value()),
                continuation_enabled=True,
                include_input_images=d.images_checkbox.isChecked(),
                include_output_images=d.output_assessment_checkbox.isChecked(),
                require_experiment_plan=True,
                review_before_finish=external or self.review_checkbox.isChecked(),
                assert_current=lambda: _assert_not_cancelled(cancelled),
                cancelled=cancelled,
            )
            self.agent_loop = WorkflowAgentLoop(self.session)
            self._settings = None if external else self._provider_settings()
            self._binding_value = self._binding()
            self._epoch += 1
            self.active, d._busy, d._job_kind = True, True, "investigation"
            d._generation += 1
            self._events = []
            self.model_records = []
            self.card.show()
            d._append_request_chat(objective)
            # Keep approved answers even if the next investigation asks another
            # essential question or its provider fails. Never make the original
            # goal absorb the growing dialogue.
            d._clarifications = (
                list(objective.clarifications)
                if isinstance(objective, ProposalDialogue)
                else []
            )
            self._pending_question = ""
            d._start_request_clock()
            self.timer.start()
            self._event(
                "Starting investigation",
                "Trials are detached; the accepted workflow is unchanged.",
            )
            if not external:
                self._next_model()
        except (ValueError, RuntimeError, TypeError) as error:
            self._failed(self._epoch, error)
        d._sync_buttons()

    def start_refinement(
        self,
        snapshot,
        owner,
        *,
        output_roles=None,
        preservation_constraint=None,
        protected_role=None,
        max_attempts=None,
    ):
        """Explicitly approved, GUI-owned warm start; no accepted graph mutation."""
        d = self.dialog
        if self.active or d._busy:
            return False
        if (
            d._provider() == "example"
            or not callable(self.capture)
            or (
                d._provider() == "codex"
                and (not d._codex_authenticated or d._connection_busy)
            )
            or not d.consent_checkbox.isChecked()
            or not d.output_assessment_checkbox.isChecked()
        ):
            raise ValueError(
                "Confirm the assistant and output sharing before refinement."
            )
        d._assessment.assert_current(snapshot, owner)
        self._selected_draft = None
        d.cancel_request(announce=False)
        self._external = self._paused = False
        self._existing_bundle = None
        self.pause_card.hide()
        try:
            objective = d._answered_objective()
            d._context = None
            d._ensure_context(objective, approved_request=True)
            self._context, self._owner = d._context, d._owner
            pipeline, self._compute, bundle = self.capture(
                self._context,
                self._owner,
                completed_snapshot=snapshot,
                completed_owner=owner,
                output_roles=output_roles,
                preservation_constraint=preservation_constraint,
                protected_role=protected_role,
            )
            self._cancelled = threading.Event()
            cancelled = self._cancelled.is_set
            self.session = InvestigationSession.from_completed(
                pipeline,
                bundle,
                d._request_objective,
                compute_request=self._compute,
                limits=self.effort.limits(
                    max_attempts
                    if max_attempts is not None
                    else d._refinement.slider.value()
                ),
                continuation_enabled=True,
                include_input_images=d.images_checkbox.isChecked(),
                include_output_images=True,
                require_experiment_plan=True,
                review_before_finish=self.review_checkbox.isChecked(),
                assert_current=lambda: _assert_not_cancelled(cancelled),
                cancelled=cancelled,
            )
            self.agent_loop = WorkflowAgentLoop(self.session)
            self._warm_snapshot, self._warm_owner = snapshot, owner
            self._warm_bundle = bundle
            self._warm_binding = d._refinement._binding()
            self._settings = self._provider_settings()
            self._binding_value = self._binding()
            self._epoch += 1
            self.active, d._busy, d._job_kind = True, True, "investigation"
            d._generation += 1
            self._events, self.model_records = [], []
            self.card.show()
            self._pending_question = ""
            d._start_request_clock()
            self.timer.start()
            self._event(
                "Refining the accepted workflow",
                "Inspecting, diagnostic tests and workflow alternatives share one "
                "investigation. The accepted result is kept.",
            )
            self._next_model()
            return True
        except (ValueError, RuntimeError, TypeError) as error:
            self._failed(self._epoch, error)
            return False

    def _submit(self, kind, call):
        self._guard(check_budget=False)
        self._worker_kind = kind
        QThreadPool.globalInstance().start(_Work(self._epoch, kind, call, self.signals))

    def _next_model(self):
        self._guard()
        if self.session.pending_finish_review is not None:
            self._show_finish_review()
            return
        session, epoch, settings = self.session, self._epoch, self._settings
        agent = self.agent_loop
        if agent is None or agent.session is not session:
            raise ValueError("The investigation agent no longer owns this session.")
        turn = agent.prepare_turn()
        request = turn.request
        self.dialog.request_edit.setPlainText(
            json.dumps(request, indent=2, ensure_ascii=False)
        )
        self.note.setText(
            "Considering the next inspection or experiment · "
            f"{session.attempted_trials}/{session.limits.max_attempts} "
            "trial attempts · "
            f"{session.actions_used} actions"
            + (
                f" · {request['diagnostic_probes']['attempts_remaining']} small "
                "diagnostic tests remaining"
                if "diagnostic_probes" in request
                else ""
            )
        )
        self._submit(
            "model",
            lambda: agent.choose_action(
                turn,
                settings,
                cancelled=self._cancelled.is_set,
                progress=lambda text: self.signals.progress.emit(epoch, str(text)),
            ),
        )

    def _ready(self, epoch, kind, result):
        if epoch != self._epoch or not self.active:
            return
        if self._review_pending:
            return  # No late/duplicate worker result can bypass the user's decision.
        self._worker_kind = None
        try:
            self._guard(check_budget=False)
            if kind == "model" and self.session.pending_finish_review is not None:
                return
            if kind == "model":
                self.model_records.append(
                    {
                        "provider": result.provider,
                        "model": result.model,
                        "model_provider": getattr(result, "model_provider", ""),
                        "requested_model": getattr(result, "requested_model", ""),
                        "response_normalizations": list(
                            getattr(result, "response_normalizations", ())
                        ),
                        "request_sha256": result.request_sha256,
                        "attempts": self.dialog._attempt_trace(result),
                        "action": result.action.to_dict(),
                    }
                )
                if self._pause_after_model:
                    self._pause_after_model = False
                    self.session.request_pause()
                    self.agent_loop.discard_pending_turn()
                    self._show_pause()
                    return
                action = result.action
                self._event(_ACTION_NAMES.get(action.action, "Inspecting results"), "")
                agent = self.agent_loop
                self._submit("action", lambda: agent.execute_action(action))
                return
            public = result.public_result()
            self.dialog.trace_edit.setPlainText(
                json.dumps(public, indent=2, ensure_ascii=False)
            )
            self._event(
                _ACTION_NAMES.get(public["action"], "Investigation step"),
                _action_detail(public),
            )
            if self.session.pending_finish_review is not None:
                self._show_finish_review()
                return
            if self.session.finished:
                self._publish_finished(public.get("result", {}))
                return
            if self.session.paused:
                self._show_pause()
                return
            if self.session.stop_reason:
                question = public.get("result", {}).get("question", "")
                self._pending_question = question
                self._complete()
                if question:
                    self.dialog._append_chat("Assistant", question)
                    self.dialog.stage_tabs.setCurrentIndex(1)
                self.dialog._set_status("warning", question or self.session.stop_reason)
                return
            QTimer.singleShot(0, self._next_guarded)
        except (ValueError, RuntimeError, TypeError) as error:
            self._failed(epoch, error)

    def external_result(self, public):
        """Display only a bridge receipt whose live authority was rechecked."""
        if not self.active or not self._external:
            return
        self._guard(check_budget=False)
        self.dialog.trace_edit.setPlainText(
            json.dumps(public, indent=2, ensure_ascii=False)
        )
        self._event(
            _ACTION_NAMES.get(public.get("action"), "External agent step"),
            _action_detail(public),
        )
        if self.session.pending_finish_review is not None:
            self._show_finish_review()
        elif self.session.paused:
            self._show_pause()
        elif self.session.finished:
            # With an external owner, Finish is a nomination and requires the
            # local review path. No external request publishes an Add capability.
            self.dialog._set_status(
                "info", "External work finished. Review its evidence in VIPP."
            )

    def _clear_finish_review(self):
        self._review_pending = False
        self.review_card.hide()
        self.review_feedback.clear()
        self.review_gallery_toggle.setChecked(False)
        self.review_gallery_images.clear()
        while self.review_gallery_layout.count():
            widget = self.review_gallery_layout.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()

    def _build_finish_review_gallery(self):
        """Decode only current approved cached PNGs once, never render or send."""
        self.review_gallery_toggle.setChecked(False)
        images = (
            tuple(self.session.preview_images())
            if self.dialog.output_assessment_checkbox.isChecked()
            else ()
        )
        self.review_gallery_toggle.setVisible(bool(images))
        if (
            not images
            or len(images) > MAX_IMAGE_PREVIEWS
            or sum(len(image.png_bytes) for image in images) > MAX_PREVIEW_TOTAL_BYTES
        ):
            return
        try:
            metadata = json.loads(self.dialog.request_edit.toPlainText())
            pairs = metadata.get("comparison_previews", [])
        except (ValueError, TypeError):
            pairs = []
        manifests = [image.manifest() for image in images]
        descriptions = [
            hashlib.sha256(image.description.encode("utf-8")).hexdigest()
            for image in images
        ]
        ordered, selections = [], {}
        for pair in pairs:
            if pair.get("status") != "supplied":
                continue
            for view in pair.get("views", []):
                for index, image in enumerate(images):
                    if (
                        manifests[index]["sha256"] == view.get("preview_sha256")
                        and image.label.startswith(
                            f"{view.get('candidate_id')}/{pair.get('role')} "
                        )
                        and descriptions[index]
                        == pair.get("selection", {})
                        .get("display_policy", {})
                        .get("render_description_sha256")
                    ):
                        if index not in ordered:
                            ordered.append(index)
                        selections[index] = pair["selection"]
        ordered.extend(index for index in range(len(images)) if index not in ordered)
        self.review_gallery_layout.addWidget(
            _label(
                "Existing cached views only; check each caption. Views may belong "
                "to other trials. Missing selected-result views are not rendered here."
            )
        )
        for index in ordered:
            image = images[index]
            pixmap = QPixmap()
            if not pixmap.loadFromData(image.png_bytes, "PNG"):
                continue
            caption = _label(image.label)
            tooltip = image.description
            if index in selections:
                tooltip += "\n" + json.dumps(selections[index], ensure_ascii=False)
            caption.setToolTip(tooltip)
            picture = QLabel()
            picture.setAlignment(Qt.AlignCenter)
            picture.setMinimumWidth(1)
            picture.setPixmap(
                pixmap.scaled(420, 200, Qt.KeepAspectRatio, Qt.FastTransformation)
            )
            picture.setToolTip(tooltip)
            self.review_gallery_layout.addWidget(caption)
            self.review_gallery_layout.addWidget(picture)
            self.review_gallery_images.append((caption, picture))

    def _sync_finish_review(self, pending=None):
        if pending is None and self.session is not None:
            pending = self.session.pending_finish_review
        if not self.active or pending is None:
            self.review_card.hide()
            return
        candidate_id = pending["candidate_id"]
        name = (
            "Accepted starting result"
            if candidate_id == "candidate0"
            else f"Trial {candidate_id.removeprefix('candidate')}"
        )
        self.review_title.setText(name + " — your decision")
        self.review_title.setToolTip(f"Candidate ID: {candidate_id}")
        self.review_summary.setText(pending["summary"])
        self.review_limitations.setText(
            "Remaining limitations:\n"
            + "\n".join("• " + text for text in pending["limitations"])
            if pending["limitations"]
            else "No limitations were recorded. This is not scientific validation."
        )
        budgets = pending["remaining_budgets"]
        self.review_budget.setText(
            "Working allowance remaining: "
            f"{budgets['attempts_remaining']} trial attempts · "
            f"{budgets['actions_remaining']} actions · "
            f"{budgets['model_calls_remaining']} model calls · "
            f"{budgets['seconds_remaining']:.0f} seconds. "
            "Review time is paused; used work is never reset."
        )
        allowed = pending["continuation_allowed"]
        self.review_continue_button.setEnabled(allowed)
        self.review_feedback.setEnabled(allowed)
        self.review_accept_button.setEnabled(True)
        self.review_stop_button.setEnabled(True)
        self.review_reason.setText(
            "Keep improving sends your optional feedback using the current sharing "
            "choices, within the current allowance."
            if allowed
            else pending["continuation_reason"]
        )
        self.review_card.show()

    def _show_finish_review(self):
        first_pause = not self._review_pending
        self._review_pending = True
        self.dialog._job_kind = "investigation_review"
        self.dialog._stop_request_clock()
        self.card.show()
        self.note.setText(
            "Paused for your decision. No request is being sent. "
            "A proposed choice is not proof that your goal is met."
        )
        self.dialog.progress_label.setText(
            "Waiting for your decision; original limits still apply."
        )
        if first_pause:
            self._build_finish_review_gallery()
        self._sync_finish_review()
        self.dialog._set_status(
            "info", "Review the proposed result and its remaining limitations."
        )
        self.dialog._sync_buttons()
        self.dialog.stage_tabs.setCurrentIndex(2)
        epoch = self._epoch
        QTimer.singleShot(0, lambda: self._reveal_finish_review(epoch))

    def _reveal_finish_review(self, epoch):
        if self.active and self._review_pending and epoch == self._epoch:
            self.dialog.review_scroll.ensureWidgetVisible(self.review_card, 16, 12)

    def _continue_finish_review(self):
        if not self.active or not self._review_pending:
            return
        try:
            self._guard()
        except (ValueError, RuntimeError, TypeError) as error:
            self._failed(self._epoch, error)
            return
        try:
            self.agent_loop.continue_finish_review(feedback=self.review_feedback.text())
        except (ValueError, RuntimeError, TypeError) as error:
            # A depleted budget is not authority to cancel or promote the choice.
            self._sync_finish_review()
            self.dialog._set_status("warning", _safe_error(error))
            return
        self._clear_finish_review()
        self.dialog._job_kind = "investigation"
        self.dialog._start_request_clock()
        self.dialog._sync_buttons()
        self._next_guarded()

    def _accept_finish_review(self):
        if not self.active or not self._review_pending:
            return
        try:
            self._guard()
            finish = self.agent_loop.accept_finish_review()
            self._publish_finished(finish, user_accepted=True)
        except (ValueError, RuntimeError, TypeError) as error:
            self._failed(self._epoch, error)

    def _publish_finished(self, finish, *, user_accepted=False):
        """Publish one exact finished choice through the checked draft path."""
        selected = self.session.selected_candidate
        if (
            selected is None
            or selected.status != "completed"
            or not self.session.finished
        ):
            raise ValueError(
                "Only an exact completed selection can be published for review."
            )
        if (
            self._existing_bundle is not None
            or self.session._accepted_bundle is not None
        ) and selected.candidate_id == "candidate0":
            self._complete()
            self.note.setText(
                "The current workflow was kept. No change was selected or added."
            )
            return
        if self._warm_snapshot is not None and selected.candidate_id == "candidate0":
            self._complete()
            self.note.setText(
                "You accepted the starting result. The original workflow and "
                "protected part remain."
                if user_accepted
                else "The assistant kept the accepted workflow. No tested change "
                "was selected; the original result and protected part remain."
            )
            self.dialog._append_chat(
                "Assistant",
                finish.get("model_interpretation", "")
                + "\n\n"
                + "\n".join(finish.get("limitations", ())),
            )
            self.dialog._set_status("info", self.note.text())
            self.dialog._refinement.note.setText(self.note.text())
            return
        proposal = selected.proposal
        proposal["context_id"] = self._context.context_id
        self._complete()
        self.dialog._context, self.dialog._owner = self._context, self._owner
        self.dialog._accept_proposal(proposal)
        if self.dialog._prepared is None or not self.dialog._prepared.can_apply:
            self.note.setText(
                "The selected trial needs attention before Add. "
                "The final workflow check did not pass."
            )
            return
        if self._warm_snapshot is not None and self.session._protected_role:
            self.dialog._refinement._pending_investigative_promotion = (
                f"proposal_{self._context.context_id[:12]}_protected_structure",
                0,
            )
        if self._warm_snapshot is not None:
            self._selected_draft = (
                self._context.context_id,
                self.dialog._prepared.fingerprint,
            )
        explanation = finish.get("model_interpretation", "")
        limitations = finish.get("limitations", ())
        if explanation or limitations:
            self.dialog._append_chat(
                "Assistant",
                explanation + "\n\n" + "\n".join("• " + text for text in limitations),
            )
        self.dialog._set_status(
            "info" if user_accepted else "success",
            "Accepted for plan review. Add and Run are still separate actions."
            if user_accepted
            else "Investigation complete. Review the selected trial and assumptions; "
            "Add and Run are still separate actions.",
        )
        self.note.setText(
            "You accepted this completed trial, including its limitations. "
            "No workflow was changed."
            if user_accepted
            else "A completed trial was selected. Its saved evidence is retained; "
            "no workflow was changed."
        )
        self.dialog.stage_tabs.setCurrentIndex(2)
        return

    def _next_guarded(self):
        if self.active and not self._external and not self._paused:
            try:
                self._next_model()
            except (ValueError, RuntimeError, TypeError) as error:
                self._failed(self._epoch, error)

    def assert_selected_draft(self, prepared):
        """Never promote edited JSON under a tested candidate's evidence badge."""
        if self._selected_draft is None:
            return
        context_id, fingerprint = self._selected_draft
        if prepared._context.context_id == context_id:
            self.dialog._assessment.assert_current(
                self._warm_snapshot, self._warm_owner
            )
        if (
            prepared._context.context_id == context_id
            and prepared.fingerprint != fingerprint
        ):
            raise ValueError(
                "This draft differs from the tested selection. Ask for a new "
                "investigation before adding it; the accepted workflow is kept."
            )

    def _event(self, heading, detail):
        self._events.append((heading, detail))
        self.history.setText(
            "<ul>"
            + "".join(
                f"<li>{html.escape(title)}"
                f"{': ' + html.escape(text) if text else ''}</li>"
                for title, text in self._events[-64:]
            )
            + "</ul>"
        )
        self.dialog._set_status("info", heading + ". You can cancel.")

    def _progress(self, epoch, message):
        if epoch == self._epoch and self.active and not self._review_pending:
            self.dialog.progress_label.setText(message)

    def _tick(self):
        try:
            pending = self._guard(check_budget=False)
            if self._review_pending:
                self._sync_finish_review(pending)
        except (ValueError, RuntimeError) as error:
            self.cancel(str(error))

    def _complete(self):
        self.active = False
        self._pause_after_model = False
        self._worker_kind = None
        live = getattr(self.dialog, "_live_agent", None)
        if live is not None:
            live.revoke("Investigation ended.", pause=False)
        self._paused = False
        self.pause_card.hide()
        self._clear_finish_review()
        self.timer.stop()
        self.dialog._busy = False
        self.dialog._stop_request_clock()
        self.dialog._sync_buttons()

    def _failed(self, epoch, error):
        if epoch != self._epoch:
            return
        self._worker_kind = None
        if self._pause_after_model and self.session is not None:
            self._pause_after_model = False
            self.session.request_pause()
        if self.session is not None and self.session.paused:
            try:
                self._guard(check_budget=False)
            except (ValueError, RuntimeError, TypeError) as revoked:
                # A resumable core checkpoint is not a new grant of GUI/source
                # authority. Never turn a stale-owner failure into Continue.
                self.session.cancel()
                error = revoked
            else:
                self._show_pause()
                return
        self._cancelled.set()
        self._complete()
        self.card.show()
        self.note.setText(
            "Investigation stopped. Completed trial evidence is retained; "
            "your workflow is unchanged."
        )
        self.dialog._set_status(
            "error", "Investigation stopped before a final draft. " + _safe_error(error)
        )
        diagnostic = self.dialog._safe_diagnostic(error)
        self.dialog._error_diagnostic = diagnostic
        self.dialog.copy_error_button.setEnabled(diagnostic is not None)
        details = self.dialog._attempt_trace(error)
        if diagnostic is not None:
            details += "\nSafe provider error details:\n" + json.dumps(
                diagnostic.to_dict(), indent=2, ensure_ascii=False
            )
        self.dialog.trace_edit.setPlainText(details)

    def _provider_settings(self):
        d = self.dialog
        common = dict(
            max_attempts=d.attempts_spin.value(), timeout_seconds=d.timeout_spin.value()
        )
        if d._provider() == "codex":
            return CodexSettings(
                model=d.codex_model_edit.text().strip(),
                reasoning_effort=d.reasoning_combo.currentText(),
                **common,
            )
        if d._provider() == "openai":
            from napari_vipp.core.workflow_agent_provider import OpenAISettings

            return OpenAISettings(
                model=d.api_model_edit.text().strip(),
                api_key_env=d.api_key_env_edit.text().strip(),
                reasoning_effort=d.reasoning_combo.currentText(),
                **common,
            )
        from napari_vipp.core.workflow_proposal_model import ModelSettings

        return ModelSettings(
            kind=d._kind(),
            base_url=d.address_edit.text().strip(),
            model=d.model_combo.currentText().strip(),
            context_tokens=d.local_context_spin.value(),
            **common,
        )

    def _pause(self):
        if self.active and self.session is not None and not self._review_pending:
            if self._worker_kind == "model":
                # A paid request remains active work until it returns. Do not
                # count its outstanding network/inference time as a human pause.
                self._pause_after_model = True
            else:
                self.session.request_pause()
            self.note.setText(
                "Pausing after the current step. Completed evidence and the "
                "accepted workflow are kept."
            )
            self.pause_button.setEnabled(False)
            if self._external:
                # No built-in worker is outstanding. The external bridge settles
                # its own request boundary, or the next shared-tool call pauses.
                live = getattr(self.dialog, "_live_agent", None)
                if live is not None:
                    live.revoke("Paused by you.")

    def _show_pause(self):
        self._paused = True
        self.dialog._stop_request_clock()
        self.dialog._job_kind = "investigation_pause"
        self.pause_button.setEnabled(True)
        self.pause_note.setText(
            "Paused — your history, trial results and working allowance are kept.\n"
            + (self.session.stop_reason or "Continue when you are ready.")
        )
        state = self.session.continuation_state
        self.pause_note.setText(
            "Paused — your history, trial results and working allowance are kept.\n"
            + (state.get("reason") or "Continue when you are ready.")
        )
        question = state.get("question", "")
        previous_question = self._pending_question
        self._pending_question = question
        if question != previous_question or not question:
            self.pause_answer.clear()
        self.pause_answer.setVisible(bool(question))
        if question:
            self.pause_note.setText(self.pause_note.text() + "\n\n" + question)
            if question != previous_question:
                self.dialog._append_chat("Assistant", question)
        self.pause_card.show()
        self.note.setText("Investigation paused, not failed. No request is being sent.")
        self.dialog._set_status(
            "info",
            "Continue the same investigation, or adjust its remaining allowance.",
        )
        self.dialog.stage_tabs.setCurrentIndex(1)
        self.dialog._sync_buttons()
        epoch = self._epoch
        QTimer.singleShot(0, lambda: self._reveal_pause(epoch))

    def _reveal_pause(self, epoch):
        if self.active and self._paused and epoch == self._epoch:
            self.dialog.discussion_scroll.ensureWidgetVisible(self.pause_card, 16, 12)

    def _resume(self):
        if not self.active or not self._paused:
            return
        try:
            self._guard(check_budget=False)
            raw_answer = self.pause_answer.text()
            answer = raw_answer if raw_answer.strip() else None
            with self.agent_loop.tools.exclusive():
                self.agent_loop.discard_pending_turn()
                self.session.resume(answer=answer)
            self.dialog._clarifications = list(self.session.dialogue.clarifications)
            if answer:
                self.dialog._append_chat("You", answer)
            self.pause_answer.clear()
            self._pending_question = ""
            self._paused = False
            self.pause_card.hide()
            self.dialog._job_kind = "investigation"
            if not self._external:
                self.dialog._start_request_clock()
            self.dialog._sync_buttons()
            self._next_guarded()
        except (ValueError, RuntimeError, TypeError) as error:
            self.dialog._set_status("warning", _safe_error(error))

    def _adjust_allowance(self):
        if not self.active or not (self._paused or self._review_pending):
            return
        editor = QDialog(self.dialog)
        editor.setWindowTitle("Working allowance — same investigation")
        layout = QVBoxLayout(editor)
        layout.addWidget(
            _label(
                "Raise total allowances within the safety ceilings. "
                "Used time, calls and trials are not reset."
            )
        )
        effort = AssistantEffortControls(editor)
        effort.toggle.setChecked(True)
        limits = self.session.limits
        for control, value in zip(
            effort.controls,
            (
                max(1, (int(limits.max_seconds) + 59) // 60),
                limits.max_actions,
                limits.max_model_calls,
                max(1, (limits.max_image_bytes + 1048575) // 1048576),
            ),
            strict=True,
        ):
            control.setMinimum(value)
            control.setValue(value)
        layout.addWidget(effort)
        trials = QSpinBox()
        trials.setRange(limits.max_attempts, 5)
        trials.setValue(limits.max_attempts)
        trials.setPrefix("Total trial attempts: ")
        layout.addWidget(trials)
        row = QHBoxLayout()
        apply = AssistantButton("Grant allowance")
        apply.clicked.connect(editor.accept)
        close = AssistantButton("Cancel")
        close.clicked.connect(editor.reject)
        row.addWidget(apply)
        row.addWidget(close)
        layout.addLayout(row)
        if editor.exec() != QDialog.Accepted:
            return
        try:
            self._guard(check_budget=False)
            with self.agent_loop.tools.exclusive():
                self.session.grant_allowance(effort.limits(trials.value()))
            self._event(
                "Working allowance updated", "Usage and trial history were not reset."
            )
            if self._review_pending:
                self._sync_finish_review()
            else:
                self._show_pause()
        except (ValueError, RuntimeError, TypeError) as error:
            self.dialog._set_status("warning", _safe_error(error))

    def cancel(self, reason="Investigation cancelled.", *, user_decision=False):
        if not self.active:
            return
        live = getattr(self.dialog, "_live_agent", None)
        if live is not None:
            live.revoke(reason, pause=False)
        self._cancelled.set()
        if self.agent_loop is not None:
            self.agent_loop.cancel(user_decision=user_decision)
        elif self.session is not None:
            self.session.cancel(user_decision=user_decision)
        self._epoch += 1
        self._complete()
        self.note.setText(reason + " No accepted workflow was changed.")
        self.dialog._set_status("warning", self.note.text())


def _assert_not_cancelled(cancelled):
    if cancelled():
        raise ValueError("Investigation was cancelled.")


def _safe_error(error):
    # Provider diagnostics are kept in Advanced by their normal transport. Do
    # not expose arbitrary filesystem/runtime exception text in this new path.
    if type(error).__module__.startswith("napari_vipp.core.workflow_investigation"):
        return str(error)
    return "Check the connection, selected input and limits before starting again."


def _action_detail(public):
    """Plain-language journal notes; all rendering escapes the returned text."""
    if public.get("status") != "ok":
        return "This step needs attention; the assistant can review the feedback."
    result = public.get("result")
    if not isinstance(result, dict):
        return ""
    if public.get("action") == "plan_experiment":
        plan = result.get("plan", {})
        return plan.get("hypothesis", "")
    if public.get("action") == "assess_experiment":
        assessment = result.get("assessment") or {}
        checks = assessment.get("check_results", [])
        labels = {
            "supports": "supported",
            "contradicts": "not supported",
            "unresolved": "still uncertain",
        }
        counts = {
            outcome: sum(check.get("outcome") == outcome for check in checks)
            for outcome in labels
        }
        return (
            "; ".join(
                f"{count} check{'s' if count != 1 else ''} {labels[outcome]}"
                for outcome, count in counts.items()
                if count
            )
            + ". These are the assistant's assessments, not proof of accuracy."
        )
    return ""


_ACTION_NAMES = {
    "diagnostic_probe": "Testing an idea on a small image region",
    "inspect_probe": "Inspecting a small diagnostic result",
    "read_notebook": "Recalling an earlier investigation result",
    "pin_notebook": "Keeping an investigation note in view",
    "pin_previews": "Keeping selected image views together",
    "find_nodes": "Finding suitable nodes",
    "describe_node": "Checking how a node works",
    "describe_nodes": "Checking how the selected nodes work",
    "describe_candidate": "Reviewing a tested workflow",
    "inspect_source": "Inspecting the input image",
    "trial_candidate": "Calculating a trial workflow",
    "trial_edit": "Testing small workflow edits",
    "inspect_candidate": "Inspecting trial results",
    "compare_candidates": "Comparing trial results",
    "retain_candidate": "Keeping a preferred trial",
    "finish": "Selecting a workflow for review",
    "question": "An important choice needs your answer",
    "record_observation": "Recording an observation and a possible explanation",
    "plan_experiment": "Choosing an idea and how to test it",
    "assess_experiment": "Checking what the experiment actually changed",
}
