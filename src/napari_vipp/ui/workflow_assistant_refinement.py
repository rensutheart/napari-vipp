"""Explicit, bounded assess/revise/run orchestration on the ordinary UI paths."""

from __future__ import annotations

import html
from dataclasses import replace
from time import monotonic

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
)

from napari_vipp.core.workflow_refinement import (
    PRESERVED_OUTPUT_KEY,
    RefinementLoop,
    assert_refinement_graph_safe,
    candidate_signature,
    capture_preservation_constraint,
    preservation_mesh_creators,
    preservation_outputs,
)
from napari_vipp.ui.workflow_assistant_style import (
    AssistantButton,
    AssistantDisclosureButton,
)


def _label(text=""):
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(Qt.PlainText)
    return label


def _output_ref(value):
    # Qt's QVariant converts tuple item data to a list on some bindings.
    return tuple(value) if type(value) in {tuple, list} else value


class AssistantRefinementController:
    """No activity from a slider change; only Start grants this bounded session."""

    def __init__(self, dialog, cancel_run=None, repair_run=None):
        self.dialog = dialog
        self.cancel_run = cancel_run
        self.repair_run = repair_run
        self._pending_failure = None
        self._failure_feedback = None
        self.loop = None
        self._own_mutation = False
        self._advancing = False
        self._history = []
        self._history_events = []
        self._epoch = 0
        self._session_open = False
        self.preservation_constraint = None
        self.preservation_result = None
        self._preserved_candidate_ref = None
        self._investigative = False
        self._investigative_output_roles = None
        self._investigative_protected_role = None
        self._pending_investigative_promotion = None
        self.options = QFrame()
        self.options.setObjectName("AssistantRefinementOptions")
        options = QVBoxLayout(self.options)
        options.setContentsMargins(14, 10, 14, 10)
        options.setSpacing(6)
        title = _label("Refine using the results")
        title.setObjectName("AssistantSectionTitle")
        options.addWidget(title)
        controls = QHBoxLayout()
        controls.addWidget(_label("Manual"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 5)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.slider.setAccessibleName("Maximum additional refinement run attempts")
        self.slider.setMinimumHeight(24)
        controls.addWidget(self.slider, 1)
        controls.addWidget(_label("5 attempts"))
        options.addLayout(controls)
        self.description = _label()
        options.addWidget(self.description)
        self.slider.valueChanged.connect(self._limit_changed)
        self.card = QFrame()
        self.card.setObjectName("AssistantRefinementCard")
        layout = QVBoxLayout(self.card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        heading = _label("Improve, run and review")
        heading.setObjectName("AssistantSectionTitle")
        layout.addWidget(heading)
        self.note = _label()
        layout.addWidget(self.note)
        self.authority = _label(
            "Start allows the selected assistant to assess results, draft changes, "
            "add each checked candidate and run it, up to your limit (30 minutes "
            "total). Uses your sharing choices. Files are not saved automatically. "
            "Failed calculations count toward the selected run limit. Up to two "
            "recognized mesh-input failures may prompt a replacement within the "
            "same request/time limits; only the exact failed addition is removed, "
            "as an undoable edit, and its draft stays in history. Stops for other "
            "errors, questions, cancellation or an AI judgment that the "
            "goal is met. That judgment is not scientific validation."
        )
        self._legacy_authority = self.authority.text()
        layout.addWidget(self.authority)
        self.preservation_checkbox = QCheckBox("Keep an accepted part of a mask")
        self.preservation_checkbox.setToolTip(
            "Optional: preserve all existing foreground in a selected region. "
            "This checks retention, not anatomical accuracy; it never repairs pixels."
        )
        layout.addWidget(self.preservation_checkbox)
        self.preservation_controls = QFrame()
        preserve_layout = QVBoxLayout(self.preservation_controls)
        preserve_layout.setContentsMargins(0, 0, 0, 0)
        self.preservation_primary = QComboBox()
        self.preservation_region = QComboBox()
        for caption, combo in (
            ("Mask to preserve", self.preservation_primary),
            ("Only within this region", self.preservation_region),
        ):
            row = QHBoxLayout()
            row.addWidget(_label(caption))
            row.addWidget(combo, 1)
            preserve_layout.addLayout(row)
        self.preservation_note = _label(
            "Run a plan first. Positive labels count as foreground. "
            "No output is changed."
        )
        preserve_layout.addWidget(self.preservation_note)
        layout.addWidget(self.preservation_controls)
        self.preservation_controls.hide()
        self.preservation_checkbox.toggled.connect(self._preservation_changed)
        self.preservation_primary.currentIndexChanged.connect(
            self._preservation_selection_changed
        )
        self.preservation_region.currentIndexChanged.connect(
            self._preservation_selection_changed
        )
        actions = QHBoxLayout()
        self.start_button = AssistantButton("Start refinement")
        self.start_button.setProperty("assistantAction", "primary")
        self.stop_button = AssistantButton("Stop refinement")
        self.stop_button.setProperty("assistantAction", "secondary")
        actions.addWidget(self.start_button)
        actions.addWidget(self.stop_button)
        actions.addStretch(1)
        layout.addLayout(actions)
        self.history_toggle = AssistantDisclosureButton(
            "Recent results and calculation failures"
        )
        layout.addWidget(self.history_toggle)
        self.history = QLabel()
        self.history.setWordWrap(True)
        self.history.setTextFormat(Qt.RichText)
        self.history.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.history)
        self.history.hide()
        self.history_toggle.toggled.connect(self.history.setVisible)
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(
            lambda: self.stop("Refinement cancelled.", cancel_work=True)
        )
        self.timer = QTimer(dialog)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._tick)
        self._limit_changed()

    @property
    def active(self):
        investigation = getattr(self.dialog, "_investigation", None)
        return bool(
            self._investigative and investigation is not None and investigation.active
        ) or (self.loop is not None and self.loop.active)

    def configure_investigative_outputs(self, output_roles, protected_role=None):
        """Passive explicit role selection; starts no model call or calculation."""
        from copy import deepcopy

        if self.active or self.dialog._busy:
            raise ValueError("Choose output roles before starting refinement.")
        self._investigative_output_roles = deepcopy(output_roles)
        self._investigative_protected_role = protected_role

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
            d.kind_combo.currentText(),
            d.address_edit.text(),
            d.model_combo.currentText(),
            d.attempts_spin.value(),
            d.timeout_spin.value(),
            repr(d._output_destination),
            self.slider.value(),
            self.preservation_checkbox.isChecked(),
            self.preservation_primary.currentData(),
            self.preservation_region.currentData(),
            self.preservation_constraint.constraint_id
            if self.preservation_constraint
            else None,
        )

    def _preservation_changed(self, checked):
        if self.active:
            self.stop("The preservation choice changed.", cancel_work=True)
        self._withdraw_preservation_draft_authority()
        self.preservation_controls.setVisible(checked)
        if not checked:
            d = self.dialog
            if d._busy:
                d.cancel_request(announce=False)
            d._assessment.reply = None
            d._assessment.revision_requested = False
            d._draft_authorized = False
            self.preservation_constraint = self.preservation_result = None
            self._preserved_candidate_ref = None
            return
        try:
            self._load_preservation_outputs()
            self.configure_preservation(
                _output_ref(self.preservation_primary.currentData()),
                _output_ref(self.preservation_region.currentData()),
            )
        except (ValueError, RuntimeError) as exc:
            self.preservation_note.setText(str(exc))

    def _load_preservation_outputs(self):
        d = self.dialog
        snapshot, owner = d._assessment.capture(
            d._assessment.candidate_ids, d._assessment.source_id
        )
        d._assessment.assert_current(snapshot, owner)
        refs = preservation_outputs(snapshot)
        if not refs:
            raise ValueError(
                "No calculated mask/label output is available in this run's evidence."
            )
        for combo in (self.preservation_primary, self.preservation_region):
            combo.blockSignals(True)
            combo.clear()
        self.preservation_region.addItem("Entire mask (no region limit)", None)
        for reference in refs:
            alias = snapshot.candidate_node_ids.index(reference[0]) + 1
            operation = next(
                node.operation_id
                for node in snapshot.baseline.nodes
                if node.id == reference[0]
            )
            text = (
                f"{operation.replace('_', ' ')} · result {alias}, port {reference[1]}"
            )
            self.preservation_primary.addItem(text, reference)
            self.preservation_region.addItem(text, reference)
        for combo in (self.preservation_primary, self.preservation_region):
            combo.blockSignals(False)
        return snapshot, owner

    def _preservation_selection_changed(self, *_):
        if not self.preservation_checkbox.isChecked():
            return
        if self.active:
            self.stop("The preservation selection changed.", cancel_work=True)
        self._withdraw_preservation_draft_authority()
        try:
            self.configure_preservation(
                _output_ref(self.preservation_primary.currentData()),
                _output_ref(self.preservation_region.currentData()),
            )
        except (ValueError, RuntimeError) as exc:
            self.preservation_constraint = None
            self.preservation_note.setText(str(exc))
        finally:
            self.dialog._sync_buttons()

    def _withdraw_preservation_draft_authority(self):
        """Retain display/history, never reuse a report for a changed constraint."""
        if self.dialog._busy:
            self.dialog.cancel_request(announce=False)
        self.dialog._assessment.reply = None
        self.dialog._assessment.revision_requested = False
        self.dialog._draft_authorized = False
        self.preservation_result = None
        self._preserved_candidate_ref = None
        self._pending_investigative_promotion = None

    def configure_preservation(self, primary_ref, region_ref=None):
        """Normal explicit GUI-thread API; no coordinates or inferred mask choice."""
        d = self.dialog
        if self.active or d._busy or not d.output_assessment_checkbox.isChecked():
            raise ValueError(
                "Choose preservation after a completed run with output "
                "assessment on, before Start."
            )
        self._withdraw_preservation_draft_authority()
        self.preservation_constraint = None
        snapshot, owner = self._load_preservation_outputs()
        d._assessment.assert_current(snapshot, owner)
        constraint = capture_preservation_constraint(
            snapshot, primary_ref, region_ref, output_sharing=True
        )
        for combo, reference in (
            (self.preservation_primary, primary_ref),
            (self.preservation_region, region_ref),
        ):
            index = next(
                (
                    index
                    for index in range(combo.count())
                    if _output_ref(combo.itemData(index)) == reference
                ),
                -1,
            )
            if index < 0:
                raise ValueError("The explicitly selected mask is not available.")
            combo.blockSignals(True)
            combo.setCurrentIndex(index)
            combo.blockSignals(False)
        self.preservation_checkbox.blockSignals(True)
        self.preservation_checkbox.setChecked(True)
        self.preservation_checkbox.blockSignals(False)
        self.preservation_controls.show()
        self.preservation_constraint = constraint
        self.preservation_result = None
        self._preserved_candidate_ref = None
        # A report without this explicit constraint is historical, not reusable.
        d._assessment.reply = None
        d._assessment.revision_requested = False
        self.preservation_note.setText(
            f"Keep all {constraint.protected_count:,} foreground pixels/voxels "
            "in the selected region. Changes elsewhere are allowed. "
            "This accepted starting inference is not anatomical truth."
        )
        return constraint

    def assessment_preservation(self, snapshot):
        """Capture only frozen worker inputs after the ordinary GUI ownership gate."""
        if not self.preservation_checkbox.isChecked():
            return None, None
        if (
            not self.dialog.output_assessment_checkbox.isChecked()
            or self.preservation_constraint is None
        ):
            raise ValueError(
                "Choose a current protected mask with output assessment enabled."
            )
        self.preservation_constraint.assert_snapshot(snapshot)
        result = self.preservation_result
        if (
            snapshot.candidate_node_ids
            != self.preservation_constraint._snapshot.candidate_node_ids
            and result is None
        ):
            raise ValueError(
                "This revised run has no preservation check. "
                "Use Start refinement for a protected run."
            )
        if result is not None and result.run_id != snapshot.run_id:
            raise ValueError("The preservation check belongs to another run.")
        return self.preservation_constraint, result

    def _limit_changed(self, *_):
        if self.active:
            self.stop("The refinement limit changed.", cancel_work=True)
        count = self.slider.value()
        self.description.setText(
            "Manual — assess and revise one step at a time."
            if not count
            else f"Up to {count} additional run attempt{'s' if count != 1 else ''}: "
            "consider outputs → improve → run → consider outputs. "
            "Failed runs count toward this limit. "
            "Start separately in Plans after the first run."
        )
        if hasattr(self.dialog, "apply_button"):
            self.dialog._sync_buttons()

    def sync(self):
        d = self.dialog
        investigation = getattr(d, "_investigation", None)
        warm_available = (
            d._provider() != "example" and investigation is not None
            and callable(investigation.capture)
        )
        self.authority.setText(
            "Start allows detached inspections and up to your selected trial "
            "limit and the working allowance in Goal. The accepted "
            "workflow stays unchanged. Review the selected draft before Add "
            "and Run; files are not saved. Failed trials count."
            if warm_available else self._legacy_authority
        )
        enabled = d.output_assessment_checkbox.isChecked()
        self.card.setVisible(enabled or bool(self._history))
        self.slider.setEnabled(not self.active and not d._busy)
        self.preservation_checkbox.setEnabled(
            enabled and not self.active and not d._busy
        )
        self.preservation_controls.setEnabled(not self.active and not d._busy)
        connected = d._provider() != "codex" or (
            d._codex_authenticated and not d._connection_busy
        )
        ready = (
            enabled
            and bool(d._assessment.run_id)
            and not d._busy
            and not self.active
            and connected
            and d._provider() != "example"
            and self.slider.value() > 0
            and callable(d._run)
            and (
                not self.preservation_checkbox.isChecked()
                or self.preservation_constraint is not None
            )
        )
        self.start_button.setEnabled(bool(ready))
        self.start_button.setText(
            f"Start up to {self.slider.value()} refinement attempts"
            if self.slider.value()
            else "Choose a refinement limit in Goal"
        )
        self.stop_button.setVisible(self.active)
        if self.active:
            # Keep navigation and Stop/Close usable, but avoid concurrent edits
            # or unrelated Send/Add/Run actions during explicit delegated work.
            for widget in (
                d.prepare_button,
                d.send_button,
                d.revise_button,
                d.apply_button,
                d.run_button,
                d.preview_button,
                d.discussion_preview_button,
                d.settings_button,
                d.initiative_slider,
                d.output_folder_combo,
                d.output_folder_button,
                d.draft_combo,
                d._assessment.assess_button,
                d._assessment.revise_button,
                d._assessment.preview_button,
            ):
                widget.setEnabled(False)
            d.cancel_button.setEnabled(True)
        else:
            d.draft_combo.setEnabled(True)
        d.goal_edit.setReadOnly(self.active)
        d.discussion_edit.setReadOnly(self.active)
        for _, field in d._question_fields:
            field.setReadOnly(self.active)
        self.history_toggle.setVisible(bool(self._history_events))
        if self.loop is None and not self._investigative:
            self.note.setText(
                "This run is ready. Start assesses it, then refines up to your limit. "
                "An already current assessment is reused."
                if d._assessment.run_id
                else "Run your first plan with output assessment on, then Start here. "
                "The slider only sets a limit; it starts nothing."
            )

    def _guard(self):
        if not self.active:
            return False
        self.loop.assert_current(self._binding())
        if (
            self.dialog._closed
            or not self.dialog.output_assessment_checkbox.isChecked()
        ):
            raise ValueError("Output sharing is off or this window was closed.")
        return True

    def start(self):
        d = self.dialog
        if not self.start_button.isEnabled():
            return
        # This explicit Start approves this bounded request, just like Send.
        # Image sharing remains separately opt-in and is never enabled here.
        d.consent_checkbox.setChecked(True)
        try:
            snapshot, owner = d._assessment.capture(
                d._assessment.candidate_ids, d._assessment.source_id
            )
            d._assessment.assert_current(snapshot, owner)
            assert_refinement_graph_safe(snapshot.baseline, snapshot.source_id)
            signature = candidate_signature(
                snapshot.baseline, snapshot.candidate_node_ids, snapshot.source_id
            )
            if self.preservation_checkbox.isChecked():
                if self.preservation_constraint is None:
                    raise ValueError(
                        "Choose the completed mask and region to preserve before Start."
                    )
                self.preservation_constraint.assert_snapshot(snapshot)
            if d._provider() != "example" and callable(d._investigation.capture):
                self._pending_investigative_promotion = None
                self._investigative = True
                started = d._investigation.start_refinement(
                    snapshot,
                    owner,
                    output_roles=self._investigative_output_roles,
                    preservation_constraint=self.preservation_constraint,
                    protected_role=self._investigative_protected_role,
                    max_attempts=self.slider.value(),
                )
                self.note.setText(
                    "Investigating small edits while keeping the accepted result."
                    if started
                    else "Refinement could not start; the accepted result is kept."
                )
                return
            self.preservation_result = None
            self._preserved_candidate_ref = None
            self.loop = RefinementLoop.start(
                self.slider.value(), self._binding(), initial_signature=signature
            )
            self._pending_failure = self._failure_feedback = None
            self._epoch += 1
            self._session_open = True
            self.timer.start()
            self.note.setText("Starting bounded refinement from this completed run…")
            d._append_chat(
                "VIPP", self.start_button.text() + ". " + self.authority.text()
            )
            if d._assessment.reply is not None:
                self._assessment_finished()
            else:
                self._request_assessment()
        except (ValueError, RuntimeError) as exc:
            self.stop(str(exc), cancel_work=True)
            d._set_status("warning", str(exc))
        d._sync_buttons()

    def before_request(self, kind):
        if not self.active:
            return True
        try:
            self._guard()
            if kind not in {"proposal", "assessment"}:
                raise ValueError("An unrelated request interrupted refinement.")
            self.loop.begin_request(
                "revision" if kind == "proposal" else "assessment", self._binding()
            )
            self.note.setText(
                f"Attempt {self.loop.attempted_runs + (kind == 'proposal')} "
                f"of up to {self.slider.value()} · "
                + (
                    "drafting a replacement after calculation failure…"
                    if self._failure_feedback is not None
                    else "drafting improvements…"
                    if kind == "proposal"
                    else "assessing outputs…"
                )
                + f" · {self.loop.completed_rounds} assessed"
            )
            return True
        except (ValueError, RuntimeError) as exc:
            self.stop(str(exc), cancel_work=True)
            return False

    def timeout_seconds(self, configured, attempts):
        """Fit the connector's existing repair allowance inside this deadline."""
        if not self.active:
            return configured
        self._guard()
        remaining_per_attempt = int((self.loop.deadline - monotonic()) / attempts)
        if remaining_per_attempt < 5:
            raise ValueError("Too little refinement time remains for another request.")
        return min(configured, remaining_per_attempt)

    def _queue(self, call):
        epoch = self._epoch

        def guarded():
            if not self.active or epoch != self._epoch:
                return
            try:
                if self._guard():
                    call()
            except (ValueError, RuntimeError) as exc:
                self.stop(str(exc), cancel_work=True)

        QTimer.singleShot(0, guarded)

    def job_ready(self, kind):
        if self.active:
            self._queue(
                self._assessment_finished
                if kind == "assessment"
                else self._proposal_finished
            )

    def _assessment_finished(self):
        d = self.dialog
        reply = d._assessment.reply
        if reply is None or reply.context.run_id != d._assessment.run_id:
            raise ValueError("No current assessment was received. Refinement stopped.")
        d._assessment.assert_current(reply.context, reply.owner)
        if reply.choices != d._assessment._choices():
            raise ValueError("The goal, answers or sharing choices changed.")
        report = reply.result.assessment
        proceed = self.loop.record_assessment(
            report.assessment_id, report.run_id, report.overall, summary=report.summary
        )
        if not proceed:
            self.stop(self.loop.stop_reason)
            return
        if report.overall == "cannot_assess":
            self.stop(
                "The assistant needs more evidence. Review this run before continuing."
            )
            return
        d._assessment.revise()
        if not d._busy:
            self.stop("A revision could not be started. Review the current results.")

    def _proposal_finished(self):
        d = self.dialog
        prepared = d._prepared
        if prepared is None or not prepared.can_apply:
            self.stop(
                "The draft needs your answers or did not pass checks. "
                "Review Discussion; nothing from this draft was run."
            )
            return
        assert_refinement_graph_safe(prepared.candidate, d.source_combo.currentData())
        signature = candidate_signature(
            prepared.candidate, prepared.added_node_ids, d.source_combo.currentData()
        )
        if self.preservation_constraint is not None:
            try:
                prefix = prepared._context.context_id[:12]
                target = f"proposal_{prefix}_{PRESERVED_OUTPUT_KEY}"
                if target not in prepared.added_node_ids:
                    raise ValueError(
                        "The revised plan did not nominate protected_structure; "
                        "nothing was added or run."
                    )
                self._preserved_candidate_ref = (target, 0)
                preservation_mesh_creators(
                    prepared.candidate,
                    prepared.added_node_ids,
                    self._preserved_candidate_ref,
                )
            except (ValueError, RuntimeError):
                self._withdraw_preservation_draft_authority()
                raise
        self.loop.accept_revision(signature, questions=prepared.questions)
        if not self.active:
            self.stop(self.loop.stop_reason)
            return
        self._own_mutation = True
        try:
            d.apply_proposal()
            if not d._added_ids:
                raise ValueError(
                    "The revised plan could not be added. Refinement stopped."
                )
            self.loop.run_started()
            self._failure_feedback = None
            self.note.setText(
                f"Running attempt {self.loop.attempted_runs} "
                f"of up to {self.slider.value()} · "
                f"{self.loop.completed_rounds} assessed…"
            )
            accepted = d._run_added_plan()
            if not self.active:
                return  # The precise synchronous terminal error was already shown.
            if not accepted:
                raise ValueError(
                    "Calculation was denied or provided no current assessable results. "
                    "Refinement stopped; check VIPP's calculation status."
                )
        finally:
            self._own_mutation = False
        if self.active and self.loop.phase == "running" and d._assessment.run_id:
            self.results_ready()

    def results_ready(self):
        if self._pending_investigative_promotion is not None:
            try:
                d = self.dialog
                snapshot, owner = d._assessment.capture(
                    d._assessment.candidate_ids, d._assessment.source_id
                )
                d._assessment.assert_current(snapshot, owner)
                reference = self._pending_investigative_promotion
                if reference[0] not in snapshot.candidate_node_ids:
                    raise ValueError("The exact nominated protected output is absent.")
                self.preservation_result = self.preservation_constraint.check(
                    snapshot, reference
                )
                if not self.preservation_result.passed:
                    raise ValueError("The promoted run lost protected foreground.")
                self._preserved_candidate_ref = reference
            except (ValueError, RuntimeError, StopIteration) as exc:
                self.dialog._draft_authorized = False
                self.dialog._assessment.invalidate(
                    "The final run needs a preservation review: " + str(exc)
                )
            finally:
                self._pending_investigative_promotion = None
        if self._investigative:
            return
        if not self.active or self.loop.phase != "running":
            return
        try:
            self._guard()
            if self.preservation_constraint is not None:
                d = self.dialog
                snapshot, owner = d._assessment.capture(
                    d._assessment.candidate_ids, d._assessment.source_id
                )
                d._assessment.assert_current(snapshot, owner)
                self.preservation_result = self.preservation_constraint.check(
                    snapshot, self._preserved_candidate_ref
                )
                if not self.preservation_result.passed:
                    missing = (
                        self.preservation_result.protected_count
                        - self.preservation_result.retained_count
                    )
                    message = (
                        f"This candidate was rejected: it lost {missing:,} protected "
                        "foreground pixels/voxels. No pixels were repaired. "
                        "The earlier branch and its historical evidence remain; "
                        "it is not automatically current."
                    )
                    d._draft_authorized = False
                    d._assessment.invalidate(message)
                    self.stop(message)
                    d._set_status("warning", message)
                    return
            self.loop.results_ready()
            self._queue(self._request_assessment)
        except (ValueError, RuntimeError) as exc:
            if self.preservation_constraint is not None:
                self.dialog._draft_authorized = False
                self.dialog._assessment.invalidate(
                    "This candidate was not accepted: " + str(exc)
                )
            self.stop(str(exc), cancel_work=True)

    def _request_assessment(self):
        self.dialog._assessment.start(preview=False)
        if not self.dialog._busy:
            raise ValueError(
                "The completed results could not be assessed. Refinement stopped."
            )

    def results_unavailable(self, message):
        if self.active and not self._own_mutation:
            self.stop(str(message), cancel_work=True)

    def calculation_failed(self, failed_run, message):
        """Queue a repair only for a positively qualified current owned failure."""
        if not self.active:
            return
        if self.preservation_constraint is not None:
            self.stop(
                str(message)
                + " The protected run failed; review it before another attempt."
            )
            return
        try:
            self._guard()
            if failed_run is None or not callable(self.repair_run):
                self._history_events.append(
                    ("failure", self.loop.attempted_runs, message)
                )
                self._render_history()
                self.stop(
                    message + " Automatic repair is unavailable; review this error."
                )
                return
            self._history_events.append(("failure", self.loop.attempted_runs, message))
            self._render_history()
            if not self.loop.record_failure(failed_run.run_id, message):
                self.stop(message + " " + self.loop.stop_reason)
                return
            self._pending_failure = failed_run
            self.note.setText(
                f"Attempt {self.loop.attempted_runs} failed · "
                f"{self.loop.completed_rounds} assessed. "
                f"Preparing repair {self.loop.calculation_repairs} of at most 2…"
            )
            self._queue(lambda: self._repair_failure(failed_run))
        except (ValueError, RuntimeError) as exc:
            self.stop(str(exc), cancel_work=True)

    def _repair_failure(self, failed_run):
        if self._pending_failure is not failed_run:
            raise ValueError("The failed calculation is no longer current.")
        d = self.dialog
        self._own_mutation = True
        try:
            context, owner, feedback = self.repair_run(
                failed_run, include_images=d.images_checkbox.isChecked()
            )
        finally:
            self._own_mutation = False
        self._pending_failure = None
        self._failure_feedback = feedback
        d._assessment.revision_requested = False
        d._context, d._owner = context, owner
        d._image_draft_guard = None
        d.consent_checkbox.setChecked(True)
        d._sending_action, d._submitted_message = "message", ""
        objective = d._answered_objective()
        d._ensure_context(objective, approved_request=True)
        from napari_vipp.core.workflow_proposals import ProposalDialogue

        d._pending_clarifications = (
            list(objective.clarifications)
            if isinstance(objective, ProposalDialogue)
            else []
        )
        d._append_chat(
            "VIPP",
            "The failed addition was removed as an undoable edit; its draft is "
            "kept in history. Earlier completed branches remain. Requesting a "
            "replacement using the safe calculation error—not partial outputs.",
        )
        d._send()
        if not d._busy:
            raise ValueError("The calculation repair could not be requested.")

    def inject_feedback(self, context):
        feedback = self._failure_feedback
        if feedback is None:
            if context._failure_feedback is not None:
                context = replace(
                    context, _failure_feedback=None, _failure_feedback_guard=None
                )
            if (
                self.preservation_constraint is not None
                and context._result_feedback is not None
            ):
                if not self.dialog.output_assessment_checkbox.isChecked():
                    raise ValueError(
                        "Output sharing is off; preservation evidence was not sent."
                    )
                self.preservation_constraint.assert_snapshot(
                    context._result_feedback_guard
                )
                result = dict(context._result_feedback)
                result["preservation_constraint"] = (
                    self.preservation_constraint.public_contract()
                )
                previews, omissions = self.preservation_constraint.with_previews(
                    context.image_previews,
                    include_output_images=self.dialog._provider() == "codex",
                )
                result["image_previews"] = [preview.manifest() for preview in previews]
                result["preservation_preview_omissions"] = omissions
                context = replace(
                    context, _result_feedback=result, image_previews=previews
                )
            return context
        self._guard()
        return replace(
            context,
            _result_feedback=None,
            _result_feedback_guard=None,
            _failure_feedback=feedback.public_feedback(),
            _failure_feedback_guard=feedback,
        )

    def context_changed(self):
        if self._investigative:
            self.dialog._investigation.cancel("The refinement context changed.")
        self._investigative = False
        self._investigative_output_roles = None
        self._investigative_protected_role = None
        self._pending_investigative_promotion = None
        self._pending_failure = self._failure_feedback = None
        self.preservation_constraint = self.preservation_result = None
        self._preserved_candidate_ref = None
        self.preservation_checkbox.blockSignals(True)
        self.preservation_checkbox.setChecked(False)
        self.preservation_checkbox.blockSignals(False)
        self.preservation_controls.hide()
        if self.active:
            self.stop(
                "The input, goal or settings changed. Refinement stopped.",
                cancel_work=True,
            )
        if hasattr(self.dialog, "_assessment"):
            self.dialog._assessment.previous_record = None
            self.dialog._assessment.comparison_record = None
        self._history.clear()
        self._history_events.clear()
        self.history.clear()

    def record_report(self, report):
        self._history.append(report)
        self._history = self._history[-6:]
        self._history_events.append(("report", 0, report))
        self._render_history()

    def _render_history(self):
        # Display only; neither an old report nor a failed attempt grants Run
        # authority. Retain up to six completed reports plus bounded failures.
        self._history_events = self._history_events[-8:]
        while sum(kind == "report" for kind, *_ in self._history_events) > 6:
            index = next(
                i
                for i, event in enumerate(self._history_events)
                if event[0] == "report"
            )
            del self._history_events[index]
        esc = html.escape
        blocks = []
        for kind, attempt, item in self._history_events:
            if kind == "failure":
                blocks.append(
                    f"<p><b>Attempt {attempt} — calculation failed</b><br>"
                    + esc(item)
                    + "<br>No completed output assessment was made.</p>"
                )
                continue
            comparison = getattr(item, "comparison", None)
            blocks.append(
                f"<p><b>Run {esc(str(item.run_id))}</b> · "
                + esc(item.overall.replace("_", " "))
                + "<br>"
                + esc(item.summary)
                + (
                    "<br><b>Change since the previous run:</b> "
                    + esc(comparison.explanation)
                    if comparison is not None
                    else ""
                )
                + "</p>"
            )
        self.history.setText("".join(blocks))

    def _tick(self):
        if not self.active:
            self.timer.stop()
            return
        try:
            self._guard()
        except (ValueError, RuntimeError) as exc:
            self.stop(str(exc), cancel_work=True)

    def stop(self, reason, *, cancel_work=False):
        if self._investigative:
            self.dialog._investigation.cancel(str(reason))
            return
        if self.loop is None:
            return
        if not self.active and not self._session_open:
            return  # First terminal reason wins; do not replace it with a denial.
        was_active = self.active
        owned_work = self._session_open
        self._session_open = False
        self._pending_failure = self._failure_feedback = None
        if was_active:
            self.loop.stop(str(reason))
        self._epoch += 1
        self.timer.stop()
        self.note.setText(
            f"Stopped after {self.loop.attempted_runs} run attempt(s); "
            f"{self.loop.completed_rounds} refinement result(s) assessed. "
            + str(reason)
        )
        if owned_work and cancel_work:
            if callable(self.cancel_run):
                self.cancel_run()
            self.dialog.cancel_request(announce=False)
        if owned_work or was_active or not cancel_work:
            self.dialog._append_chat("VIPP", self.note.text())
            self.dialog._set_status("info", self.note.text())
        self.dialog._sync_buttons()
