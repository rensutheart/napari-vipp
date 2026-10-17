"""Staged, consent-bound planning workspace with explicit guarded Add/Run."""

from __future__ import annotations

import html
import json
import re
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from qtpy.QtCore import (
    QElapsedTimer,
    QEvent,
    QObject,
    QRunnable,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
    Slot,
)
from qtpy.QtGui import QPainter, QPen, QPixmap
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.core.workflow_proposal_model import (
    DEFAULT_PROPOSAL_TIMEOUT_SECONDS,
    MAX_PROPOSAL_TIMEOUT_SECONDS,
)
from napari_vipp.core.workflow_proposal_outputs import (
    OutputDestination,
    OutputDestinationError,
    is_same_input_folder_answer,
    resolve_output_destination,
)
from napari_vipp.core.workflow_proposals import (
    MAX_PROPOSAL_BYTES,
    PlanningPreferences,
    PreparedProposal,
    ProposalContext,
    ProposalDialogue,
    ProposalError,
    reference_proposal,
    validate_proposal,
)
from napari_vipp.ui.dialog_buttons import add_dialog_buttons
from napari_vipp.ui.palette_roles import theme_colors
from napari_vipp.ui.toolbar_controls import toolbar_icon
from napari_vipp.ui.workflow_assistant_details import AssistantDetailsEditor
from napari_vipp.ui.workflow_assistant_style import (
    AssistantButton,
    AssistantDisclosureButton,
    AssistantSendButton,
    AssistantStageTabs,
    DiscussionTranscript,
    ResponsiveAssistantEditor,
    SharingPreviewDialog,
    assistant_action_icon,
    assistant_chat_surfaces,
    assistant_connection_pixmap,
    assistant_settings_stylesheet,
    assistant_stylesheet,
    assistant_surfaces,
)
from napari_vipp.ui.workspace_window import WorkspaceWindowController

_LOGIN_CLEANUP_LOCK = threading.Lock()

_INITIATIVE_CHOICES = (
    ("ask_first", "Ask first", "Ask up to 3 focused questions when choices matter."),
    (
        "balanced",
        "Balanced",
        "Choose routine defaults; ask up to 2 questions about important choices.",
    ),
    (
        "use_best_judgment",
        "Use best judgment",
        "Choose reasonable draft settings and show the assumptions. Ask only "
        "about a genuine blocker (at most 1 question).",
    ),
)

_PLAN_REVIEW_CHECKS = (
    "Compare the result with the original image. Are the parts you wanted "
    "included, without missing pieces or unwanted background?",
    "If you are making separate masks, check that the structures have not "
    "joined together or broken into pieces. For a z-stack, look through the "
    "slices, not just one preview.",
    "If you are measuring size, check the image scale and units. Structures "
    "cut off at the image edge are incomplete; do not treat their size as a "
    "complete-object measurement.",
)


def _codex_profile_issue(detail):
    """Recognise specific setup guards for guidance, never bypass or repair them."""
    profile_errors = (
        "The VIPP Codex profile was modified",
        "The VIPP Codex profile changed",
        "The VIPP Codex profile is incomplete",
        "Redirected Codex profile paths are not supported",
        "Extra rules, hooks or skills are unsupported in the VIPP Codex profile",
        "Extra skills are unsupported in the VIPP profile",
        "The embedded Codex skill tree is invalid",
        "An unrecognised Codex skill was found",
        "Redirected Codex skills are unsupported",
        "A previous VIPP connection repair did not finish",
        "This VIPP connection setup cannot be repaired automatically",
    )
    return (
        "profile" if any(detail.startswith(prefix) for prefix in profile_errors) else ""
    )


@dataclass(frozen=True)
class _Draft:
    """Presentation only: no source, pixels, owner or apply capability."""

    summary: str
    steps: tuple[tuple[str, str], ...]
    assumptions: tuple[str, ...]
    questions: tuple[str, ...]
    proposal_text: str
    plan_text: str


@dataclass(frozen=True, slots=True)
class _ChatReply:
    """Display-only attribution; raw history and provider dialogue stay literal."""

    question: str
    answer: str
    is_question: bool = True


class _AssistantSettingsDialog(QDialog):
    """Edit a copy: dismissing this dialog cannot invalidate a live plan."""

    def __init__(self, assistant):
        super().__init__(assistant)
        self.assistant = assistant
        self.setObjectName("WorkflowAssistantSettings")
        self.setWindowTitle("Workflow assistant settings")
        self.resize(600, 460)
        self.setMinimumWidth(420)
        self._closed = False
        self._generation = 0
        self._model_busy = False
        self._signals = _CallSignals()
        self._signals.ready.connect(self._models_ready)
        self._signals.failed.connect(self._models_failed)
        self._fit_timer = QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.timeout.connect(self._fit_content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        header = QFrame()
        header.setObjectName("AssistantSettingsHeader")
        heading = QVBoxLayout(header)
        heading.setContentsMargins(20, 18, 20, 16)
        title_row = QHBoxLayout()
        self.header_icon = QLabel()
        self.header_icon.setFixedSize(24, 24)
        title_row.addWidget(self.header_icon)
        title = _label("Assistant settings")
        title.setObjectName("SettingsHeading")
        title_row.addWidget(title, 1)
        heading.addLayout(title_row)
        subtitle = _label("Choose how VIPP connects. No goal or image is sent here.")
        subtitle.setObjectName("SettingsMuted")
        heading.addWidget(subtitle)
        outer.addWidget(header)
        self.scroll = QScrollArea()
        self.scroll.setObjectName("AssistantSettingsScroll")
        self.scroll.viewport().setObjectName("AssistantSettingsViewport")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(self.scroll, 1)
        self.body = QWidget()
        self.body.setObjectName("AssistantSettingsBody")
        body = QVBoxLayout(self.body)
        body.setContentsMargins(20, 16, 20, 16)
        body.setSpacing(14)
        body.setAlignment(Qt.AlignTop)
        self.scroll.setWidget(self.body)
        provider_card = QFrame()
        provider_card.setObjectName("AssistantSettingsCard")
        form = QVBoxLayout(provider_card)
        form.setContentsMargins(14, 12, 14, 14)
        provider_label = _label("Choose your assistant")
        provider_label.setObjectName("SettingsCardHeading")
        form.addWidget(provider_label)
        self.provider_combo = QComboBox()
        provider_labels = {
            "codex": "ChatGPT / Codex",
            "local": "Local model",
            "openai": "OpenAI API — separately billed",
            "example": "Example — no AI",
        }
        for index in range(assistant.provider_combo.count()):
            provider = assistant.provider_combo.itemData(index)
            self.provider_combo.addItem(
                provider_labels.get(provider, assistant.provider_combo.itemText(index)),
                provider,
            )
        self.provider_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.provider_combo.setMinimumContentsLength(12)
        self.provider_combo.setCurrentIndex(assistant.provider_combo.currentIndex())
        self.provider_combo.setAccessibleName("Assistant provider")
        form.addWidget(self.provider_combo)
        body.addWidget(provider_card)
        self.codex_group = QFrame()
        self.codex_group.setObjectName("AssistantConnectionCard")
        codex = QVBoxLayout(self.codex_group)
        codex.setContentsMargins(14, 14, 14, 14)
        codex.setSpacing(10)
        connection_heading = QHBoxLayout()
        connection_heading.setSpacing(10)
        self.connection_icon = QLabel()
        self.connection_icon.setFixedSize(24, 24)
        connection_heading.addWidget(self.connection_icon)
        self.connection_title = _label()
        self.connection_title.setObjectName("SettingsCardHeading")
        connection_heading.addWidget(self.connection_title, 1)
        codex.addLayout(connection_heading)
        self.connection_hint = _label()
        codex.addWidget(self.connection_hint)
        self.connection_details_button = AssistantButton("View connection details")
        self.connection_details_button.setProperty("assistantAction", "link")
        self.connection_details_button.setAutoDefault(False)
        self.connection_details_button.clicked.connect(self._show_connection_details)
        codex.addWidget(self.connection_details_button, 0, Qt.AlignLeft)
        self.connection_label = _label(assistant.connection_label.text())
        row = QHBoxLayout()
        row.setSpacing(10)
        self.connection_actions = row
        self.connection_repair_button = AssistantSendButton("Repair connection setup")
        self.connection_repair_button.setProperty("assistantAction", "primary")
        self.connection_repair_button.setAutoDefault(False)
        self.connection_repair_button.setToolTip(
            "Update only VIPP's unchanged managed setup for this Codex version. "
            "Keeps your sign-in; does not send a goal or image."
        )
        self.connection_repair_button.clicked.connect(
            assistant._repair_codex_connection
        )
        row.addWidget(self.connection_repair_button)
        self.signin_button = AssistantSendButton("Sign in with ChatGPT…")
        self.signin_button.setProperty("assistantAction", "primary")
        self.signin_button.setAutoDefault(False)
        self.signin_button.setToolTip(
            "Open the official ChatGPT sign-in page in your browser."
        )
        self.signin_button.clicked.connect(assistant._begin_codex_signin)
        row.addWidget(self.signin_button)
        self.connection_refresh_button = AssistantSendButton("Check connection")
        self.connection_refresh_button.setProperty("assistantAction", "secondary")
        self.connection_refresh_button.setAutoDefault(False)
        self.connection_refresh_button.setToolTip(
            "Check the existing sign-in. Does not open a browser "
            "or send a planning request."
        )
        self.connection_refresh_button.clicked.connect(
            assistant._check_codex_connection
        )
        row.addWidget(self.connection_refresh_button)
        codex.addLayout(row)
        self.browser_hint = _label(
            "Sign in in your browser, then return here. "
            "VIPP updates the status automatically."
        )
        self.browser_hint.setObjectName("SettingsMuted")
        codex.addWidget(self.browser_hint)
        body.addWidget(self.codex_group)
        self.local_group = QFrame()
        self.local_group.setObjectName("AssistantSettingsCard")
        local_outer = QVBoxLayout(self.local_group)
        local_outer.setContentsMargins(14, 14, 14, 14)
        local_title = _label("Connect a local model")
        local_title.setObjectName("SettingsCardHeading")
        local_outer.addWidget(local_title)
        local = QFormLayout()
        local.setSpacing(10)
        local.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        local.setRowWrapPolicy(QFormLayout.WrapLongRows)
        local_outer.addLayout(local)
        self.kind_combo = QComboBox()
        self.kind_combo.addItems(["Ollama", "Compatible local server"])
        self.kind_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.kind_combo.setMinimumContentsLength(10)
        self.kind_combo.setCurrentIndex(assistant.kind_combo.currentIndex())
        local.addRow("Server type", self.kind_combo)
        self.address_edit = QLineEdit(assistant.address_edit.text())
        local.addRow("Local address", self.address_edit)
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.model_combo.setMinimumContentsLength(8)
        self.model_combo.addItems(
            [
                assistant.model_combo.itemText(i)
                for i in range(assistant.model_combo.count())
            ]
        )
        self.model_combo.setCurrentText(assistant.model_combo.currentText())
        row = QHBoxLayout()
        self.local_model_actions = row
        row.addWidget(self.model_combo, 1)
        self.refresh_button = AssistantButton("Refresh models")
        self.refresh_button.setProperty("assistantAction", "secondary")
        self.refresh_button.setAutoDefault(False)
        self.refresh_button.clicked.connect(self._refresh_models)
        row.addWidget(self.refresh_button)
        local.addRow("Model", row)
        self.local_context_spin = QSpinBox()
        self.local_context_spin.setRange(4096, 65536)
        self.local_context_spin.setSingleStep(4096)
        self.local_context_spin.setValue(assistant.local_context_spin.value())
        self.local_context_spin.setToolTip(
            "Ollama context allocation for investigation. Larger values can use "
            "substantially more memory. No model is loaded by changing this setting."
        )
        local.addRow("Context tokens (Ollama)", self.local_context_spin)
        self.local_status = _label(
            "Only trusted loopback servers are supported. "
            "A local address cannot prove the server never forwards data."
        )
        local.addRow(self.local_status)
        body.addWidget(self.local_group)
        self.api_group = QFrame()
        self.api_group.setObjectName("AssistantSettingsCard")
        api = QFormLayout(self.api_group)
        api.setContentsMargins(14, 14, 14, 14)
        api.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.api_model_edit = QLineEdit(assistant.api_model_edit.text())
        self.api_model_edit.setPlaceholderText("Model available to your API project")
        api.addRow("API model", self.api_model_edit)
        self.api_key_env_edit = QLineEdit(assistant.api_key_env_edit.text())
        api.addRow("API key environment variable", self.api_key_env_edit)
        api_note = _label(
            "Enter the variable NAME, not your key. VIPP reads the key only when "
            "you send a request, and never stores it in a workflow. Official "
            "OpenAI HTTPS only. API usage is billed separately from ChatGPT. "
            "API mode uses the investigation harness."
        )
        api.addRow(api_note)
        body.addWidget(self.api_group)
        self.example_note = _label(
            "The example is a fixed recipe, not AI. "
            "It does not interpret your goal or inspect your image."
        )
        body.addWidget(self.example_note)
        advanced_card = QFrame()
        advanced_card.setObjectName("AssistantSettingsCard")
        advanced = QVBoxLayout(advanced_card)
        advanced.setContentsMargins(14, 4, 14, 4)
        advanced.setSpacing(8)
        self.advanced_toggle = AssistantDisclosureButton("Advanced")
        advanced.addWidget(self.advanced_toggle)
        self.advanced_panel = QWidget()
        self.advanced_panel.setObjectName("AssistantSettingsAdvanced")
        advanced_body = QVBoxLayout(self.advanced_panel)
        advanced_body.setContentsMargins(0, 4, 0, 10)
        advanced_body.setSpacing(12)
        self.codex_options = QWidget()
        self.codex_options.setObjectName("AssistantSettingsCodexOptions")
        codex_options = QFormLayout(self.codex_options)
        codex_options.setContentsMargins(0, 0, 0, 0)
        codex_options.setSpacing(10)
        codex_options.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.codex_model_edit = QLineEdit(assistant.codex_model_edit.text())
        self.codex_model_edit.setPlaceholderText("Use the default model")
        codex_options.addRow("Model (optional)", self.codex_model_edit)
        self.reasoning_combo = QComboBox()
        self.reasoning_combo.addItems(["medium", "high", "xhigh"])
        self.reasoning_combo.setCurrentText(assistant.reasoning_combo.currentText())
        codex_options.addRow("Thinking effort", self.reasoning_combo)
        advanced_body.addWidget(self.codex_options)
        limits = QFormLayout()
        limits.setSpacing(10)
        limits.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.attempts_spin = QSpinBox()
        self.attempts_spin.setRange(1, 2)
        self.attempts_spin.setValue(assistant.attempts_spin.value())
        limits.addRow("Plan attempts (includes repair)", self.attempts_spin)
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(5, MAX_PROPOSAL_TIMEOUT_SECONDS)
        self.timeout_spin.setValue(assistant.timeout_spin.value())
        self.timeout_spin.setSuffix(" s")
        self.timeout_spin.setToolTip(assistant.timeout_spin.toolTip())
        limits.addRow("Time limit per attempt", self.timeout_spin)
        advanced_body.addLayout(limits)
        advanced_body.addWidget(
            _label(
                "Attempts include automatic plan repair. These limits do not "
                "run or change your workflow."
            )
        )
        self.connection_details = QWidget()
        self.connection_details.setObjectName("AssistantSettingsConnectionDetails")
        details = QVBoxLayout(self.connection_details)
        details.setContentsMargins(0, 0, 0, 0)
        details_title = _label("Connection details")
        details_title.setObjectName("SettingsCardHeading")
        details.addWidget(details_title)
        details.addWidget(self.connection_label)
        details.addWidget(
            _label(
                "Sign-in uses VIPP's separate Codex profile. "
                "No credentials or chat history "
                "are copied from this app. Cancel does not remove a completed sign-in."
            )
        )
        advanced_body.addWidget(self.connection_details)
        advanced.addWidget(self.advanced_panel)
        self.advanced_panel.hide()
        self.advanced_toggle.toggled.connect(self._advanced_changed)
        body.addWidget(advanced_card)
        footer = QFrame()
        footer.setObjectName("AssistantSettingsFooter")
        self.accept_button = AssistantButton("Apply settings")
        self.accept_button.setProperty("assistantAction", "primary")
        self.accept_button.clicked.connect(self.accept)
        self.cancel_button = AssistantButton("Cancel")
        self.cancel_button.setProperty("assistantAction", "secondary")
        self.cancel_button.clicked.connect(self.reject)
        actions = QHBoxLayout(footer)
        actions.setContentsMargins(20, 12, 20, 12)
        actions.setSpacing(10)
        actions.addStretch()
        add_dialog_buttons(
            actions, actions=(self.accept_button,), dismiss=self.cancel_button
        )
        outer.addWidget(footer)
        self._footer = footer
        self._header = header
        self.provider_combo.currentIndexChanged.connect(self._provider_changed)
        for signal in (
            self.address_edit.textChanged,
            self.kind_combo.currentIndexChanged,
        ):
            signal.connect(self._cancel_discovery)
        self._provider_changed()
        self._refresh_appearance()
        self._refresh_connection_state()

    def _advanced_changed(self, expanded):
        self.advanced_panel.setVisible(expanded)
        self._fit_timer.start(0)

    def _show_connection_details(self):
        self.advanced_toggle.setChecked(True)
        self._fit_timer.start(0)
        QTimer.singleShot(0, self._reveal_connection_details)

    def _reveal_connection_details(self):
        if not self._closed:
            self.scroll.ensureWidgetVisible(self.connection_details)

    def _refresh_appearance(self):
        palette = self.assistant._theme_probe.palette()
        stylesheet = assistant_settings_stylesheet(palette)
        if self.styleSheet() != stylesheet:
            self.setStyleSheet(stylesheet)
        self.header_icon.setPixmap(toolbar_icon("assistant", palette).pixmap(24, 24))
        self.signin_button.set_idle_icon(
            assistant_action_icon("next", palette, primary=True)
        )
        self.connection_repair_button.set_idle_icon(
            assistant_action_icon("redo", palette, primary=True)
        )
        self.connection_refresh_button.set_idle_icon(
            assistant_action_icon("redo", palette)
        )
        self.advanced_toggle.refresh_disclosure()

    def _refresh_connection_state(self):
        if self._closed:
            return
        assistant = self.assistant
        busy = assistant._connection_busy
        authenticated = assistant._codex_authenticated
        blocked = assistant._codex_connection_issue == "profile"
        repairable = assistant._codex_profile_repairable
        repairing = busy and assistant._connection_kind == "repair"
        tone = (
            "info"
            if busy
            else "error"
            if blocked
            else "success"
            if authenticated
            else "warning"
        )
        if self.codex_group.property("connectionTone") != tone:
            self.codex_group.setProperty("connectionTone", tone)
            self.codex_group.style().unpolish(self.codex_group)
            self.codex_group.style().polish(self.codex_group)
        if busy:
            signing_in = assistant._connection_kind == "signin"
            title = (
                "Repairing connection setup…"
                if repairing
                else "Finish signing in in your browser"
                if signing_in
                else "Checking connection…"
            )
            hint = (
                "Checking and updating VIPP's managed settings. Your sign-in "
                "is kept; no goal or image is sent."
                if repairing
                else "Return here when you're done. VIPP will confirm the sign-in."
                if signing_in
                else "Checking the existing sign-in. No goal or image is sent."
            )
        elif blocked:
            if repairable:
                title = "Connection setup needs an update"
                hint = (
                    "Codex has changed since VIPP's connection was set up. "
                    "Choose Repair connection setup to update VIPP's own settings "
                    "and check your saved sign-in. Your credentials and workflow "
                    "are kept."
                )
            elif assistant._connection_kind == "repair":
                title = "Couldn't finish the connection repair"
                hint = (
                    "The connection is still blocked. Choose Check connection "
                    "to see its current state, or view the details for help. "
                    "No goal or image was sent."
                )
            else:
                title = "Connection setup needs attention"
                hint = (
                    "This setup contains changes VIPP cannot safely repair "
                    "automatically. The connection stays blocked. View connection "
                    "details for help; signing in again will not fix this setup."
                )
        elif authenticated:
            title = "ChatGPT connected"
            hint = (
                "You're signed in. Return to the assistant to send a goal "
                "when you're ready."
            )
        else:
            title = "Sign in to ChatGPT"
            hint = (
                "Use your ChatGPT account to plan with Codex. "
                "Sign in below, or check an existing sign-in."
            )
            if assistant.connection_label.property("tone") == "error":
                title = "Couldn't confirm the connection"
                hint = (
                    "Check the connection or sign in again. "
                    "Advanced contains the connection details."
                )
            elif assistant._connection_kind == "repair":
                hint = (
                    "Connection setup updated. Sign in with ChatGPT below "
                    "to finish connecting."
                )
        self.connection_title.setText(title)
        self.connection_hint.setText(hint)
        self.connection_label.setText(assistant.connection_label.text())
        self.connection_icon.setPixmap(
            assistant_connection_pixmap(tone, assistant._theme_probe.palette())
        )
        self.browser_hint.setVisible(not blocked and not repairing)
        self.connection_details_button.setVisible(blocked)
        self.signin_button.setVisible(not blocked and not repairing)
        self.connection_repair_button.setVisible(blocked or repairing)
        self.connection_repair_button.setEnabled(not busy and repairable)
        self.connection_repair_button.setText(
            "Repairing setup…" if repairing else "Repair connection setup"
        )
        self.connection_repair_button.set_sending(repairing)
        self.signin_button.setText(
            "Signing in…"
            if busy and assistant._connection_kind == "signin"
            else "Sign in again…"
            if authenticated or blocked
            else "Sign in with ChatGPT…"
        )
        self.signin_button.set_sending(busy and assistant._connection_kind == "signin")
        self.connection_refresh_button.set_sending(
            busy and assistant._connection_kind == "status"
        )
        self.connection_refresh_button.setText(
            "Checking…"
            if busy and assistant._connection_kind == "status"
            else "Check connection"
        )
        self.signin_button.setEnabled(not busy and not blocked)
        self.connection_refresh_button.setEnabled(not busy)
        self.accept_button.setEnabled(not busy)
        self.provider_combo.setEnabled(not busy)
        self._fit_timer.start(0)

    def _fit_content(self):
        if self._closed:
            return
        self.ensurePolished()
        self.layout().activate()
        width = max(1, self.scroll.viewport().width())
        primary_action = (
            self.connection_repair_button
            if not self.connection_repair_button.isHidden()
            else self.signin_button
        )
        action_width = (
            primary_action.sizeHint().width()
            + self.connection_refresh_button.sizeHint().width()
            + self.connection_actions.spacing()
        )
        self.connection_actions.setDirection(
            QHBoxLayout.TopToBottom
            if width - 68 < action_width
            else QHBoxLayout.LeftToRight
        )
        self.local_model_actions.setDirection(
            QHBoxLayout.TopToBottom
            if width - 68
            < self.model_combo.minimumSizeHint().width()
            + self.refresh_button.sizeHint().width()
            + self.local_model_actions.spacing()
            else QHBoxLayout.LeftToRight
        )
        self.body.layout().activate()
        height = self.body.heightForWidth(width)
        if height < 0:
            height = self.body.sizeHint().height()
        chrome = self._header.sizeHint().height() + self._footer.sizeHint().height() + 4
        screen = self.screen()
        cap = min(760, screen.availableGeometry().height() - 80) if screen else 760
        target = max(300, min(height + chrome, max(300, cap)))
        if self.height() != target:
            self.resize(self.width(), target)

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        self._fit_timer.start(0)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if event.oldSize().width() != event.size().width():
            self._fit_timer.start(0)

    def _provider_changed(self, *_):
        self._cancel_discovery()
        provider = self.provider_combo.currentData()
        self.codex_group.setVisible(provider == "codex")
        self.local_group.setVisible(provider == "local")
        self.api_group.setVisible(provider == "openai")
        self.example_note.setVisible(provider == "example")
        self.codex_options.setVisible(provider == "codex")
        self.connection_details.setVisible(provider == "codex")
        self.timeout_spin.setMaximum(MAX_PROPOSAL_TIMEOUT_SECONDS)
        self._fit_timer.start(0)

    def _cancel_discovery(self, *_):
        self._generation += 1
        self._model_busy = False
        self.refresh_button.setEnabled(True)

    def _refresh_models(self):
        from napari_vipp.core.workflow_proposal_model import discover_models

        if (
            self._closed
            or self._model_busy
            or self.provider_combo.currentData() != "local"
        ):
            return
        address = self.address_edit.text().strip()
        kind = "ollama" if self.kind_combo.currentIndex() == 0 else "openai-compatible"
        self._generation += 1
        self._model_busy = True
        self.refresh_button.setEnabled(False)
        self.local_status.setText("Looking for models… No goal or image is sent.")
        QThreadPool.globalInstance().start(
            _Call(
                self._generation, lambda: discover_models(address, kind), self._signals
            )
        )

    @Slot(int, object)
    def _models_ready(self, generation, models):
        if self._closed or generation != self._generation:
            return
        previous = self.model_combo.currentText()
        self.model_combo.clear()
        self.model_combo.addItems(list(models))
        if previous:
            self.model_combo.setCurrentText(previous)
        self._model_busy = False
        self.refresh_button.setEnabled(True)
        self.local_status.setText(
            f"Found {len(models)} models. Choose one, then apply settings."
        )

    @Slot(int, object)
    def _models_failed(self, generation, _error):
        if self._closed or generation != self._generation:
            return
        self._model_busy = False
        self.refresh_button.setEnabled(True)
        self.local_status.setText(
            "Could not list models. Check the address and that the server is running."
        )

    def done(self, result):
        self._closed = True
        self._cancel_discovery()
        # Cancels only this dialog's intentional sign-in/status work, not credentials.
        if self.assistant._connection_busy:
            self.assistant._cancel_connection()
        super().done(result)


class _CallSignals(QObject):
    ready = Signal(int, object)
    failed = Signal(int, object)
    progress = Signal(int, str)
    calculation_failed = Signal(object, str)


class _Call(QRunnable):
    """Worker owns detached inputs, never a widget or live pipeline."""

    def __init__(self, generation: int, call: Callable, signals: _CallSignals):
        super().__init__()
        self.generation, self.call, self.signals = generation, call, signals

    def run(self):
        try:
            result = self.call()
        except Exception as exc:
            self.signals.failed.emit(self.generation, exc)
        else:
            self.signals.ready.emit(self.generation, result)


def _label(text="", parent=None):
    label = QLabel(text, parent)
    label.setWordWrap(True)
    label.setTextFormat(Qt.PlainText)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return label


def _review_bullets_html(items):
    """Native hanging bullets; model text is literal, never interpreted markup."""
    return (
        '<ul style="margin-top: 0px; margin-bottom: 0px;">'
        + "".join(
            '<li style="margin-bottom: 8px;">'
            + html.escape(item).replace("\n", "<br>")
            + "</li>"
            for item in items
        )
        + "</ul>"
    )


def _review_section(heading, name):
    """Compact passive disclosure; only the body takes space when expanded."""
    frame = QFrame()
    frame.setObjectName(name)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(14, 0, 14, 0)
    layout.setSpacing(0)
    toggle = AssistantDisclosureButton(heading)
    toggle.setAccessibleDescription(
        "Show or hide these review notes without changing the plan or sending "
        "a request."
    )
    layout.addWidget(toggle)
    content = QWidget()
    content.setObjectName("AssistantReviewDisclosureContent")
    body = QVBoxLayout(content)
    body.setContentsMargins(0, 2, 0, 10)
    body.setSpacing(8)
    layout.addWidget(content)
    toggle.toggled.connect(content.setVisible)
    content.hide()
    frame.hide()
    return frame, toggle, content


class WorkflowAssistantDialog(QDialog):
    """Automatically checks/repairs plans; Apply remains explicit and undoable."""

    def __init__(
        self,
        parent,
        *,
        sources,
        capture,
        assert_current,
        apply,
        run=None,
        output_sources=None,
        capture_results=None,
        assert_results_current=None,
        cancel_run=None,
        repair_run=None,
        capture_investigation=None,
        current_compute=None,
        prepare_patch=None,
        apply_patch=None,
        assert_session_destination=None,
    ):
        super().__init__(parent)
        self.setObjectName("WorkflowAssistantWindow")
        self.setWindowTitle("VIPP · Workflow assistant")
        self.resize(1040, 820)
        self.setMinimumSize(760, 600)
        self._window_controller = WorkspaceWindowController(self)
        self._sources, self._capture = sources, capture
        self._output_sources = output_sources or (lambda source_id: ())
        self._output_destination: OutputDestination | None = None
        self._output_redactions: set[str] = set()
        self._output_resolved_choices: set[tuple[str, str]] = set()
        self._output_folder_error = ""
        self._output_plan_paths: tuple[str, ...] = ()
        # Never derive the next theme from our own explicit QSS colors.
        # This hidden probe inherits only the surrounding application theme.
        self._theme_probe = QWidget(self.parentWidget())
        self._theme_probe.setAttribute(Qt.WA_DontShowOnScreen)
        self._theme_probe.installEventFilter(self)
        self.destroyed.connect(self._theme_probe.deleteLater)
        self._appearance_timer = QTimer(self)
        self._appearance_timer.setSingleShot(True)
        self._appearance_timer.timeout.connect(self._refresh_appearance)
        self._assert_current, self._apply = assert_current, apply
        self._prepare_patch, self._apply_patch = prepare_patch, apply_patch
        self._run = run
        self._added_ids: tuple[str, ...] = ()
        self._drafts: list[_Draft] = []
        self._review_draft_key: tuple[int, str] | None = None
        self._current_draft_index = -1
        self._draft_authorized = False
        self._live_plan_text = ""
        self._live_proposal_text = ""
        self._chat: list[tuple[str, str]] = []
        self._chat_replies: dict[int, _ChatReply] = {}
        self._outgoing_message_pair: tuple[str, str] | None = None
        self._settings_dialog = None
        self._extra_message = ""
        self._submitted_message = ""
        self._sending_action = "goal"
        self._recovery_kind = None
        self._retry_action = "goal"
        self._recovery_submit_action = "goal"
        self._recovering = False
        self._context: ProposalContext | None = None
        self._owner = None
        self._image_draft_guard: tuple[ProposalContext, object] | None = None
        self._prepared: PreparedProposal | None = None
        self._generation = 0
        self._cancelled = threading.Event()
        self._busy = self._closed = False
        self._job_kind = ""
        self._request_elapsed = QElapsedTimer()
        self._request_clock_generation: int | None = None
        self._request_elapsed_timer = QTimer(self)
        self._request_elapsed_timer.setInterval(1000)
        self._request_elapsed_timer.timeout.connect(self._update_request_elapsed)
        self._request_objective: str | ProposalDialogue = ""
        self._clarifications: list[tuple[str, str]] = []
        self._pending_clarifications: list[tuple[str, str]] | None = None
        self._question_fields: list[tuple[str, QLineEdit]] = []
        self._step_widgets: list[QWidget] = []
        self._progress_messages: list[str] = []
        self._error_diagnostic = None
        self._signals = _CallSignals()
        self._signals.ready.connect(self._ready)
        self._signals.failed.connect(self._failed)
        self._signals.progress.connect(self._progress)
        self._connection_signals = _CallSignals()
        self._connection_signals.ready.connect(self._connection_ready)
        self._connection_signals.failed.connect(self._connection_failed)
        self._connection_generation = 0
        self._connection_busy = False
        self._connection_kind = ""
        self._connection_cancelled = threading.Event()
        self._codex_authenticated = False
        self._codex_connection_issue = ""
        self._codex_profile_repairable = False
        self._signin_process = None
        self._login_holder: dict[str, object] = {}
        self._signin_elapsed = QElapsedTimer()
        self._signin_timeout_ms = 300_000
        self._signin_timer = QTimer(self)
        self._signin_timer.setInterval(500)
        self._signin_timer.timeout.connect(self._poll_signin)
        self._validation_timer = QTimer(self)
        self._validation_timer.setSingleShot(True)
        self._validation_timer.setInterval(350)
        self._validation_timer.timeout.connect(self.validate)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.header_frame = QFrame()
        self.header_frame.setObjectName("AssistantHeader")
        heading = QHBoxLayout(self.header_frame)
        heading.setContentsMargins(16, 12, 16, 12)
        self.header_icon = QLabel()
        self.header_icon.setFixedSize(22, 22)
        heading.addWidget(self.header_icon)
        title = _label("Workflow assistant")
        title.setObjectName("AssistantHeaderTitle")
        heading.addWidget(title, 1)
        self.provider_status_icon = QLabel()
        self.provider_status_icon.setFixedSize(18, 18)
        heading.addWidget(self.provider_status_icon)
        self.provider_status_label = _label()
        self.provider_status_label.setWordWrap(False)
        heading.addWidget(self.provider_status_label)
        self.settings_button = AssistantButton("Settings…")
        self.settings_button.setObjectName("AssistantSettings")
        self.settings_button.setIcon(toolbar_icon("settings", self.palette()))
        self.settings_button.clicked.connect(self._open_settings)
        heading.addWidget(self.settings_button)
        self.close_button = AssistantButton("Close")
        self.close_button.setProperty("assistantAction", "secondary")
        self.close_button.setAutoDefault(False)
        self.close_button.clicked.connect(self.close)
        layout.addWidget(self.header_frame)

        # These controls remain the single active configuration. Settings edits a
        # copy, so Cancel/Escape cannot revoke a plan or change its provider.
        self._configuration = QWidget(self)
        self.provider_combo = QComboBox(self._configuration)
        self.provider_combo.addItem("Codex (ChatGPT sign-in)", "codex")
        self.provider_combo.addItem("Local model (Ollama / compatible server)", "local")
        self.provider_combo.addItem("Example pipeline — no AI", "example")
        self.provider_combo.addItem("OpenAI API — separately billed", "openai")
        self.codex_group = QGroupBox(self._configuration)
        self.codex_settings_widget = QWidget(self.codex_group)
        self.codex_model_edit = QLineEdit(self.codex_settings_widget)
        self.reasoning_combo = QComboBox(self.codex_settings_widget)
        self.reasoning_combo.addItems(["medium", "high", "xhigh"])
        self.reasoning_combo.setCurrentText("high")
        self.signin_button = AssistantButton(self.codex_group)
        self.signin_button.clicked.connect(self._signin_codex)
        self.connection_refresh_button = AssistantButton(
            "Refresh connection", self.codex_group
        )
        self.connection_refresh_button.clicked.connect(self._refresh_codex_connection)
        self.connection_label = _label("Checking Codex connection…", self.codex_group)
        self.local_group = QGroupBox(self._configuration)
        self.kind_combo = QComboBox(self.local_group)
        self.kind_combo.addItems(["Ollama", "OpenAI-compatible local server"])
        self.address_edit = QLineEdit("http://127.0.0.1:11434", self.local_group)
        self.model_combo = QComboBox(self.local_group)
        self.model_combo.setEditable(True)
        self.local_context_spin = QSpinBox(self._configuration)
        self.local_context_spin.setRange(4096, 65536)
        self.local_context_spin.setValue(16384)
        self.api_model_edit = QLineEdit(self._configuration)
        self.api_key_env_edit = QLineEdit("OPENAI_API_KEY", self._configuration)
        self.refresh_button = AssistantButton(self.local_group)
        self.refresh_button.clicked.connect(self._refresh_models)
        self.attempts_spin = QSpinBox(self._configuration)
        self.attempts_spin.setRange(1, 2)
        self.attempts_spin.setValue(2)
        self.timeout_spin = QSpinBox(self._configuration)
        self.timeout_spin.setRange(5, MAX_PROPOSAL_TIMEOUT_SECONDS)
        self.timeout_spin.setValue(DEFAULT_PROPOSAL_TIMEOUT_SECONDS)
        self.timeout_spin.setToolTip(
            "Maximum time for each model attempt: 600 seconds is 10 minutes. "
            "If enabled, one automatic plan repair is a separate attempt with "
            "the same limit. Cancel request stops VIPP waiting sooner."
        )
        self.consent_checkbox = QCheckBox(self._configuration)
        self._configuration.hide()

        self.stage_tabs = AssistantStageTabs()
        self.stage_tabs.setObjectName("WorkflowAssistantStages")
        self.goal_page = QWidget()
        self.discussion_page = QWidget()
        self.plans_page = QWidget()
        self.goal_page.setObjectName("AssistantGoalPage")
        self.discussion_page.setObjectName("AssistantDiscussionPage")
        self.plans_page.setObjectName("AssistantPlansPage")
        for page, caption in (
            (self.goal_page, "1 · Goal"),
            (self.discussion_page, "2 · Discussion"),
            (self.plans_page, "3 · Plans"),
        ):
            self.stage_tabs.addTab(page, caption)
        layout.addWidget(self.stage_tabs, 1)

        goal_container = QVBoxLayout(self.goal_page)
        goal_container.setContentsMargins(0, 0, 0, 0)
        goal_container.setSpacing(0)
        self.goal_scroll = QScrollArea()
        self.goal_scroll.setWidgetResizable(True)
        self.goal_scroll.setFrameShape(QFrame.NoFrame)
        goal_body = QWidget()
        goal_body.setObjectName("AssistantGoalBody")
        self.goal_scroll.setWidget(goal_body)
        goal_container.addWidget(self.goal_scroll, 1)
        goal = QVBoxLayout(goal_body)
        goal.setContentsMargins(16, 18, 16, 14)
        goal.setSpacing(10)
        goal_heading = _label("What would you like to achieve?")
        goal_heading.setObjectName("AssistantSectionHeading")
        goal.addWidget(goal_heading)
        description = _label(
            "Choose the input and describe the result you want, then send your "
            "goal. The assistant will ask questions or take you straight to a plan."
        )
        description.setObjectName("AssistantMuted")
        goal.addWidget(description)
        self.source_card = QFrame()
        self.source_card.setObjectName("AssistantSourceCard")
        source_layout = QVBoxLayout(self.source_card)
        source_layout.setContentsMargins(12, 12, 12, 12)
        source_layout.setSpacing(8)
        source_row = QHBoxLayout()
        source_layout.addLayout(source_row)
        self.source_icon = QLabel()
        self.source_icon.setFixedSize(22, 22)
        source_row.addWidget(self.source_icon)
        source_row.addWidget(_label("Input image"))
        self.source_combo = QComboBox()
        source_row.addWidget(self.source_combo, 1)
        output_row = QHBoxLayout()
        output_row.addWidget(_label("Save outputs"))
        self.output_folder_combo = QComboBox()
        self.output_folder_combo.setAccessibleName("Output folder choice")
        self.output_folder_combo.addItem("Ask me when saving", "ask")
        self.output_folder_combo.addItem("Same folder as input", "same")
        self.output_folder_combo.addItem("Choose folder…", "directory")
        output_row.addWidget(self.output_folder_combo, 1)
        self.output_folder_button = AssistantButton("Choose folder…")
        self.output_folder_button.setIcon(toolbar_icon("open", self.palette()))
        self.output_folder_button.setProperty("assistantAction", "secondary")
        self.output_folder_button.setAutoDefault(False)
        self.output_folder_button.clicked.connect(self._choose_output_folder)
        output_row.addWidget(self.output_folder_button)
        source_layout.addLayout(output_row)
        self.output_folder_status = _label()
        self.output_folder_status.setObjectName("AssistantMuted")
        self.output_folder_status.setAccessibleName("Local output folder")
        self.output_folder_status.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred
        )
        source_layout.addWidget(self.output_folder_status)
        self._update_output_folder_status()
        goal.addWidget(self.source_card)
        goal.addWidget(_label("Your goal"))
        self.goal_edit = ResponsiveAssistantEditor(
            minimum=84, maximum=180, lines=3, chrome=24, ratio=0.28
        )
        self.goal_edit.setObjectName("AssistantGoalInput")
        self.goal_edit.bind_height_to(self.goal_scroll.viewport())
        self.goal_edit.setPlainText(
            "Identify bright objects, measure each object and review the segmentation."
        )
        self.goal_edit.setPlaceholderText(
            "Describe the structures you want to identify and the result you need."
        )
        goal.addWidget(self.goal_edit)
        self.initiative_card = QFrame()
        self.initiative_card.setObjectName("AssistantInitiative")
        initiative = QVBoxLayout(self.initiative_card)
        initiative.setContentsMargins(14, 10, 14, 10)
        initiative.setSpacing(4)
        initiative_header = QHBoxLayout()
        initiative_heading = _label("Assistant initiative")
        initiative_heading.setObjectName("AssistantSectionHeading")
        initiative_header.addWidget(initiative_heading)
        initiative_header.addStretch()
        initiative.addLayout(initiative_header)
        self.initiative_slider = QSlider(Qt.Horizontal)
        self.initiative_slider.setObjectName("AssistantInitiativeSlider")
        self.initiative_slider.setRange(0, 2)
        self.initiative_slider.setSingleStep(1)
        self.initiative_slider.setPageStep(1)
        self.initiative_slider.setTickPosition(QSlider.TicksBelow)
        self.initiative_slider.setTickInterval(1)
        self.initiative_slider.setValue(2)
        self.initiative_slider.setAccessibleName("Assistant initiative")
        self.initiative_slider.setMinimumHeight(24)
        initiative_controls = QWidget()
        initiative_controls.setObjectName("AssistantInitiativeControls")
        initiative_controls_layout = QVBoxLayout(initiative_controls)
        initiative_controls_layout.setContentsMargins(0, 0, 0, 0)
        initiative_controls_layout.setSpacing(0)
        initiative_controls_layout.addWidget(self.initiative_slider)
        endpoints = QHBoxLayout()
        self.initiative_ask_label = _label("Ask me more")
        self.initiative_best_label = _label("Use best judgment")
        self.initiative_ask_label.setObjectName("AssistantInitiativeEndpoint")
        self.initiative_best_label.setObjectName("AssistantInitiativeEndpoint")
        endpoints.addWidget(self.initiative_ask_label)
        endpoints.addStretch()
        endpoints.addWidget(self.initiative_best_label)
        initiative_controls_layout.addLayout(endpoints)
        initiative.addWidget(initiative_controls)
        initiative.addSpacing(6)
        self.initiative_explanation = _label()
        self.initiative_explanation.setObjectName("AssistantInitiativeDescription")
        self.initiative_explanation.setTextFormat(Qt.RichText)
        initiative.addWidget(self.initiative_explanation)
        goal.addWidget(self.initiative_card)
        self._update_initiative_text()
        self.images_checkbox = QCheckBox("Include input image previews")
        self.images_checkbox.setToolTip(
            "Optional bounded display slices, not the full volume. "
            "A preview is not ground truth or proof that a segmentation works."
        )
        self.output_assessment_checkbox = QCheckBox("Assess workflow outputs")
        self.output_assessment_checkbox.setToolTip(
            "Turn on before Run to retain a bounded result snapshot locally. "
            "Assess results sends it. Repeated refinement needs a limit and "
            "a separate Start in Plans."
        )
        self.evidence_card = QFrame()
        self.evidence_card.setObjectName("AssistantEvidenceCard")
        evidence = QGridLayout(self.evidence_card)
        evidence.setContentsMargins(16, 12, 16, 12)
        evidence.setSpacing(10)
        evidence_title = _label("Give the assistant a view of your images")
        evidence_title.setObjectName("AssistantSectionTitle")
        evidence.addWidget(evidence_title, 0, 0, 1, 2)
        self.output_assessment_description = _label()
        for column, (checkbox, description) in enumerate(
            (
                (
                    self.images_checkbox,
                    _label(
                        "See representative input slices when choosing a starting plan."
                    ),
                ),
                (self.output_assessment_checkbox, self.output_assessment_description),
            )
        ):
            checkbox.setObjectName("AssistantEvidenceOption")
            evidence.addWidget(checkbox, 1, column)
            description.setObjectName("AssistantMuted")
            evidence.addWidget(description, 2, column, alignment=Qt.AlignTop)
        evidence.addWidget(
            _label("Both are off until you choose. Nothing is sent by checking a box."),
            3,
            0,
            1,
            2,
        )
        evidence.setColumnStretch(0, 1)
        evidence.setColumnStretch(1, 1)
        goal.insertWidget(goal.indexOf(self.initiative_card), self.evidence_card)
        self.reference_note = _label(
            "This is a fixed bright-object example, not AI. "
            "It does not interpret your goal or inspect the image."
        )
        goal.addWidget(self.reference_note)
        self.sharing_note = _label()
        self.sharing_note.setObjectName("AssistantMuted")
        goal.addWidget(self.sharing_note)
        self.goal_actions_frame = QFrame()
        self.goal_actions_frame.setObjectName("AssistantPageActions")
        goal_actions = QHBoxLayout(self.goal_actions_frame)
        goal_actions.setContentsMargins(16, 10, 16, 10)
        self.preview_button = AssistantButton("Preview what is shared")
        self.preview_button.setIcon(toolbar_icon("image", self.palette()))
        self.preview_button.clicked.connect(self.preview_sharing)
        goal_actions.addWidget(self.preview_button)
        goal_actions.addStretch()
        self.prepare_button = AssistantSendButton("Send goal")
        self.prepare_button.setProperty("assistantAction", "primary")
        self.prepare_button.set_idle_icon(toolbar_icon("send", self.palette()))
        self.prepare_button.clicked.connect(self._send_goal_clicked)
        goal_actions.addWidget(self.prepare_button)
        goal.addStretch(1)
        goal_container.addWidget(self.goal_actions_frame)

        discussion_container = QVBoxLayout(self.discussion_page)
        discussion_container.setContentsMargins(0, 0, 0, 0)
        discussion_container.setSpacing(0)
        self.discussion_scroll = QScrollArea()
        self.discussion_scroll.setObjectName("AssistantDiscussionScroll")
        self.discussion_scroll.viewport().setObjectName("AssistantDiscussionViewport")
        self.discussion_scroll.setWidgetResizable(True)
        self.discussion_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.discussion_scroll.setFrameShape(QFrame.NoFrame)
        discussion_body = QWidget()
        discussion_body.setObjectName("AssistantDiscussionBody")
        self.discussion_scroll.setWidget(discussion_body)
        discussion_container.addWidget(self.discussion_scroll, 1)
        discussion = QVBoxLayout(discussion_body)
        discussion.setContentsMargins(16, 18, 16, 14)
        discussion.setSpacing(18)
        discussion.setSizeConstraint(QLayout.SetMinimumSize)
        discussion_heading = _label("Clarify and refine together")
        discussion_heading.setObjectName("AssistantSectionHeading")
        discussion.addWidget(discussion_heading)
        discussion.addWidget(
            _label(
                "Your conversation stays here. Your original goal is kept "
                "with every revision."
            )
        )
        self.original_goal_frame = QFrame()
        self.original_goal_frame.setObjectName("AssistantOriginalGoal")
        original_goal = QVBoxLayout(self.original_goal_frame)
        original_goal.setContentsMargins(14, 0, 14, 0)
        original_goal.setSpacing(12)
        self.original_goal_toggle = AssistantDisclosureButton("Original goal")
        self.original_goal_toggle.setAccessibleDescription(
            "Read the full original goal without changing it or sending a request."
        )
        self.original_goal_toggle.toggled.connect(self._toggle_original_goal)
        original_goal.addWidget(self.original_goal_toggle)
        self.original_goal_label = _label()
        self.original_goal_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        original_goal.addWidget(self.original_goal_label)
        self.original_goal_label.hide()
        discussion.addWidget(self.original_goal_frame)

        self.conversation_frame = QFrame()
        self.conversation_frame.setObjectName("AssistantConversation")
        conversation = QVBoxLayout(self.conversation_frame)
        conversation.setContentsMargins(14, 0, 14, 0)
        conversation.setSpacing(12)
        self.conversation_toggle = AssistantDisclosureButton("Conversation history")
        self.conversation_toggle.setAccessibleDescription(
            "Read earlier messages without changing the plan or sending a request. "
            "Current questions are shown in the answer section."
        )
        self.conversation_toggle.toggled.connect(self._toggle_conversation)
        conversation.addWidget(self.conversation_toggle)
        self.transcript_edit = DiscussionTranscript()
        self.transcript_edit.setObjectName("WorkflowAssistantConversation")
        conversation.addWidget(self.transcript_edit)
        self.transcript_edit.hide()
        discussion.addWidget(self.conversation_frame)

        self.questions_group = QFrame()
        self.questions_group.setObjectName("AssistantQuestions")
        questions = QVBoxLayout(self.questions_group)
        questions.setContentsMargins(14, 14, 14, 14)
        questions.setSpacing(14)
        questions_heading = _label("A little more information")
        questions_heading.setObjectName("AssistantSectionHeading")
        questions.addWidget(questions_heading)
        self.questions_note = _label(
            "Answer the questions below, then send your answers. "
            "You can also add a message in the next section."
        )
        self.questions_note.setObjectName("AssistantMuted")
        questions.addWidget(self.questions_note)
        self.questions_layout = QFormLayout()
        self.questions_layout.setContentsMargins(0, 0, 0, 0)
        self.questions_layout.setVerticalSpacing(14)
        self.questions_layout.setRowWrapPolicy(QFormLayout.WrapAllRows)
        questions.addLayout(self.questions_layout)
        self.revise_button = AssistantSendButton("Send answers")
        self.revise_button.set_idle_icon(toolbar_icon("send", self.palette()))
        self.revise_button.clicked.connect(self._send_answers_clicked)
        discussion.addWidget(self.questions_group)
        self.composer_frame = QFrame()
        self.composer_frame.setObjectName("AssistantComposer")
        composer = QVBoxLayout(self.composer_frame)
        composer.setContentsMargins(14, 14, 14, 14)
        composer.setSpacing(14)
        composer_heading = _label("Message the assistant")
        composer_heading.setObjectName("AssistantSectionHeading")
        composer.addWidget(composer_heading)
        self.discussion_edit = ResponsiveAssistantEditor(
            minimum=72, maximum=140, lines=2, chrome=16, ratio=0.20
        )
        self.discussion_edit.setObjectName("AssistantMessageInput")
        self.discussion_edit.bind_height_to(self.discussion_scroll.viewport())
        self.discussion_edit.setPlaceholderText(
            "Ask for a change or explain a preference… "
            "If there are questions above, answer them too."
        )
        composer.addWidget(self.discussion_edit)
        self.discussion_sharing_note = _label()
        self.discussion_sharing_note.setObjectName("AssistantMuted")
        self.discussion_actions_frame = QFrame()
        self.discussion_actions_frame.setObjectName("AssistantPageActions")
        composer_actions = QHBoxLayout(self.discussion_actions_frame)
        composer_actions.setContentsMargins(16, 10, 16, 10)
        self.discussion_preview_button = AssistantButton("Preview what is shared")
        self.discussion_preview_button.setIcon(toolbar_icon("image", self.palette()))
        self.discussion_preview_button.clicked.connect(self.preview_sharing)
        composer_actions.addWidget(self.discussion_preview_button)
        composer_actions.addStretch()
        composer_actions.addWidget(self.revise_button)
        self.send_button = AssistantSendButton("Send message")
        self.send_button.setProperty("assistantAction", "primary")
        self.send_button.set_idle_icon(toolbar_icon("send", self.palette()))
        self.send_button.clicked.connect(self._send_message_clicked)
        composer_actions.addWidget(self.send_button)
        composer.addWidget(self.discussion_sharing_note)
        discussion.addWidget(self.composer_frame)
        discussion.addStretch(1)
        discussion_container.addWidget(self.discussion_actions_frame)

        plans = QVBoxLayout(self.plans_page)
        plans.setContentsMargins(0, 0, 0, 0)
        plans.setSpacing(0)
        self.review_scroll = QScrollArea()
        self.review_scroll.setWidgetResizable(True)
        self.review_scroll.setFrameShape(QFrame.NoFrame)
        self.review_widget = QWidget()
        review = QVBoxLayout(self.review_widget)
        review.setContentsMargins(16, 18, 16, 14)
        review.setSpacing(10)
        review.setSizeConstraint(QLayout.SetMinimumSize)
        self.review_scroll.setWidget(self.review_widget)
        plans.addWidget(self.review_scroll, 1)
        plans_heading = _label("Review your plans")
        plans_heading.setObjectName("AssistantSectionHeading")
        review.addWidget(plans_heading)
        plans_description = _label(
            "Compare earlier drafts, then choose the latest usable plan. "
            "Add it to your workflow before running."
        )
        plans_description.setObjectName("AssistantMuted")
        review.addWidget(plans_description)
        draft_row = QHBoxLayout()
        draft_row.addWidget(QLabel("Draft history"))
        draft_row.addStretch()
        self.draft_combo = QComboBox()
        self.draft_combo.currentIndexChanged.connect(self._draft_selected)
        draft_row.addWidget(self.draft_combo)
        review.addLayout(draft_row)
        self.draft_note = _label("No draft yet. Send your goal to start.")
        self.draft_note.setObjectName("AssistantMuted")
        review.addWidget(self.draft_note)
        from napari_vipp.ui.workflow_assistant_assessment import (
            AssistantAssessmentController,
        )

        self._assessment = AssistantAssessmentController(
            self, capture_results, assert_results_current
        )
        review.addWidget(self._assessment.card)
        from napari_vipp.ui.workflow_assistant_refinement import (
            AssistantRefinementController,
        )

        self._refinement = AssistantRefinementController(self, cancel_run, repair_run)
        self._calculation_failure_message = ""
        goal.insertWidget(
            goal.indexOf(self.initiative_card) + 1, self._refinement.options
        )
        review.addWidget(self._refinement.card)
        from napari_vipp.ui.workflow_assistant_investigation import (
            AssistantInvestigationController,
        )

        self._investigation = AssistantInvestigationController(
            self, capture_investigation, current_compute
        )
        discussion.insertWidget(
            discussion.indexOf(self.questions_group), self._investigation.pause_card
        )
        from napari_vipp.ui.workflow_assistant_live import AssistantLiveAgentController

        self._live_agent = AssistantLiveAgentController(self, self._investigation)
        goal.insertWidget(
            goal.indexOf(self.initiative_card), self._investigation.options
        )
        goal.insertWidget(goal.indexOf(self.initiative_card), self._live_agent.card)
        review.addWidget(self._investigation.card)
        self.summary_label = _label("Your proposed steps will appear here.")
        self.summary_label.setStyleSheet("font-weight: bold;")
        self.summary_card = QFrame()
        self.summary_card.setObjectName("AssistantPlanSummary")
        summary_layout = QVBoxLayout(self.summary_card)
        summary_layout.setContentsMargins(12, 10, 12, 10)
        summary_layout.addWidget(self.summary_label)
        review.addWidget(self.summary_card)
        self.steps_widget = QWidget()
        self.steps_layout = QVBoxLayout(self.steps_widget)
        self.steps_layout.setContentsMargins(0, 0, 0, 0)
        self.steps_layout.setSpacing(0)
        review.addWidget(self.steps_widget)
        (
            self.assumptions_frame,
            self.assumptions_toggle,
            self.assumptions_content,
        ) = _review_section("Assumptions to check", "AssistantAssumptionsSection")
        self.assumptions_label = _label()
        self.assumptions_label.setObjectName("AssistantPlanAssumptions")
        self.assumptions_label.setTextFormat(Qt.RichText)
        self.assumptions_content.layout().addWidget(self.assumptions_label)
        review.addWidget(self.assumptions_frame)
        (
            self.review_checks_frame,
            self.review_checks_toggle,
            self.review_checks_content,
        ) = _review_section("What to check after running", "AssistantChecksSection")
        self.review_checks_note = _label(
            "These are general checks, not a complete assessment. VIPP has not "
            "automatically assessed the result. The plan's assumptions are its "
            "starting choices, not confirmed facts."
        )
        self.review_checks_note.setObjectName("AssistantMuted")
        self.review_checks_content.layout().addWidget(self.review_checks_note)
        self.review_checks_label = _label()
        self.review_checks_label.setObjectName("AssistantReviewChecks")
        self.review_checks_label.setTextFormat(Qt.RichText)
        self.review_checks_content.layout().addWidget(self.review_checks_label)
        review.addWidget(self.review_checks_frame)
        self.advanced_checkbox = QCheckBox("Advanced details")
        review.addWidget(self.advanced_checkbox)
        self.advanced_group = QWidget()
        advanced = QVBoxLayout(self.advanced_group)
        advanced.setContentsMargins(0, 0, 0, 0)
        # Detailed activity belongs in Advanced, not between the page and footer.
        # The current status and cancellation remain visible in the global row.
        self.progress_label = _label()
        self.progress_label.setObjectName("AssistantMuted")
        advanced.addWidget(self.progress_label)
        # Expert tabs are distinct from task stages, retaining their copy/error tools.
        self.tabs = QTabWidget()
        self.preview_edit = QPlainTextEdit()
        self.preview_edit.setReadOnly(True)
        self.tabs.addTab(self.preview_edit, "Plan text")
        self.json_edit = AssistantDetailsEditor(json_syntax=True)
        self.json_edit.setPlaceholderText(
            "Plans are checked automatically. You do not need to edit JSON."
        )
        self.tabs.addTab(self.json_edit, "Proposal JSON")
        self.request_edit = AssistantDetailsEditor(json_syntax=True)
        self.request_edit.setReadOnly(True)
        self.tabs.addTab(self.request_edit, "Request")
        self.trace_edit = AssistantDetailsEditor()
        self.trace_edit.setReadOnly(True)
        self.tabs.addTab(self.trace_edit, "Attempts")
        self.tabs.setMaximumHeight(210)
        advanced.addWidget(self.tabs)
        copy_row = QHBoxLayout()
        self.copy_button = AssistantButton("Copy current detail")
        self.copy_button.clicked.connect(self._copy_tab)
        copy_row.addWidget(self.copy_button)
        self.copy_error_button = AssistantButton("Copy error details")
        self.copy_error_button.setToolTip(
            "Copies only safe error categories and codes, not your goal, "
            "images or credentials."
        )
        self.copy_error_button.clicked.connect(self._copy_error_details)
        self.copy_error_button.setEnabled(False)
        copy_row.addWidget(self.copy_error_button)
        advanced.addLayout(copy_row)
        review.addWidget(self.advanced_group)
        plan_caveat = _label(
            "A starting plan, not proof of scientific accuracy. "
            "Review parameters and results."
        )
        plan_caveat.setObjectName("AssistantMuted")
        plan_caveat.setToolTip(
            "Add changes the graph; Run starts normal workflow calculation. "
            "Output assessment and bounded refinement require separate opt-in "
            "and an explicit start. AI assessment is not scientific validation."
        )
        plan_caveat.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        review.addWidget(plan_caveat)
        self.plans_actions_frame = QFrame()
        self.plans_actions_frame.setObjectName("AssistantPageActions")
        plan_actions = QHBoxLayout(self.plans_actions_frame)
        plan_actions.setContentsMargins(16, 10, 16, 10)
        self.apply_button = AssistantButton("Add to workflow")
        self.apply_button.setProperty("assistantAction", "primary")
        self.apply_button.setIcon(toolbar_icon("checklist", self.palette()))
        self.apply_button.clicked.connect(self.apply_proposal)
        self.run_button = AssistantButton("Run workflow")
        self.run_button.setIcon(toolbar_icon("next", self.palette()))
        self.run_button.clicked.connect(self._run_added_plan)
        plan_actions.addWidget(self.apply_button)
        from napari_vipp.ui.workflow_assistant_patching import AssistantPatchController

        self._patching = AssistantPatchController(self, prepare_patch, apply_patch)
        plan_actions.addWidget(self._patching.button)
        plan_actions.addWidget(self.run_button)
        self.output_saving_label = _label()
        self.output_saving_label.setObjectName("AssistantMuted")
        self.output_saving_label.setAccessibleName("Local saving destinations")
        self.output_saving_label.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred
        )
        self.output_saving_label.hide()
        plan_actions.addWidget(self.output_saving_label, 1)
        plan_actions.addStretch()
        plans.addWidget(self.plans_actions_frame)
        self.refine_button = AssistantButton("Ask for a change in Discussion")
        self.refine_button.setProperty("assistantAction", "link")
        self.refine_button.setAutoDefault(False)
        self.refine_button.clicked.connect(lambda: self.stage_tabs.setCurrentIndex(1))
        review.addWidget(self.refine_button)
        review.addStretch()

        self.status_frame = QFrame()
        status_row = QHBoxLayout(self.status_frame)
        status_row.setContentsMargins(16, 8, 16, 8)
        self.status_icon = QLabel()
        self.status_icon.setFixedSize(22, 22)
        status_row.addWidget(self.status_icon)
        self.status_label = _label()
        status_row.addWidget(self.status_label, 1)
        self.request_progress = QProgressBar()
        self.request_progress.setObjectName("AssistantRequestProgress")
        self.request_progress.setAccessibleName("Assistant request in progress")
        self.request_progress.setRange(0, 1)
        self.request_progress.setTextVisible(False)
        self.request_progress.setFixedHeight(8)
        self.request_progress.setFixedWidth(120)
        self.request_progress.hide()
        self.request_activity = QWidget()
        self.request_activity.setObjectName("AssistantRequestActivity")
        request_activity = QVBoxLayout(self.request_activity)
        request_activity.setContentsMargins(0, 0, 0, 0)
        request_activity.setSpacing(5)
        request_activity.addWidget(self.request_progress, 0, Qt.AlignHCenter)
        self.request_elapsed_label = _label()
        self.request_elapsed_label.setObjectName("AssistantMuted")
        self.request_elapsed_label.setWordWrap(False)
        self.request_elapsed_label.setAlignment(Qt.AlignCenter)
        self.request_elapsed_label.setAccessibleName("Elapsed planning request time")
        self.request_elapsed_label.setToolTip(
            "Time since this request was sent, including plan checks and any "
            "automatic repair. Measured on this computer; it does not estimate "
            "completion or confirm that the service is responding."
        )
        self.request_elapsed_label.hide()
        request_activity.addWidget(self.request_elapsed_label)
        self.request_activity.hide()
        status_row.addWidget(self.request_activity)
        self.recovery_button = AssistantSendButton("Update plan")
        self.recovery_button.setObjectName("AssistantRecoveryAction")
        self.recovery_button.setAutoDefault(False)
        self.recovery_button.set_idle_icon(toolbar_icon("send", self.palette()))
        self.recovery_button.clicked.connect(self._recover_plan_clicked)
        self.recovery_button.hide()
        status_row.addWidget(self.recovery_button)
        self.cancel_button = AssistantButton("Cancel request")
        self.cancel_button.clicked.connect(self.cancel_request)
        status_row.addWidget(self.cancel_button)
        layout.addWidget(self.status_frame)
        self.footer_frame = QFrame()
        self.footer_frame.setObjectName("AssistantFooter")
        self._navigation = QGridLayout(self.footer_frame)
        self._navigation.setContentsMargins(16, 12, 16, 12)
        self._navigation.setColumnStretch(0, 1)
        self._navigation.setColumnStretch(1, 2)
        self._navigation.setColumnStretch(2, 1)
        self.back_button = AssistantButton("Back")
        self.back_button.setIcon(toolbar_icon("previous", self.palette()))
        self.back_button.clicked.connect(
            lambda: self.stage_tabs.setCurrentIndex(self.stage_tabs.currentIndex() - 1)
        )
        self._navigation.addWidget(self.back_button, 0, 0, Qt.AlignLeft)
        self.stage_label = _label()
        self.stage_label.setObjectName("AssistantStageCaption")
        self.stage_label.setAlignment(Qt.AlignCenter)
        self._navigation.addWidget(self.stage_label, 0, 1)
        self.next_button = AssistantButton("View plan")
        self.next_button.setIcon(toolbar_icon("next", self.palette()))
        self.next_button.clicked.connect(self._next_stage_clicked)
        self.next_button.setLayoutDirection(Qt.RightToLeft)
        self.next_button.setProperty("assistantAction", "secondary")
        self.footer_actions = QWidget()
        footer_actions = QHBoxLayout(self.footer_actions)
        footer_actions.setContentsMargins(0, 0, 0, 0)
        footer_actions.addWidget(self.next_button)
        self._navigation.addWidget(
            self.footer_actions, 0, 2, Qt.AlignRight | Qt.AlignAbsolute
        )
        layout.addWidget(self.footer_frame)
        self.bottom_frame = QFrame()
        self.bottom_frame.setObjectName("AssistantBottomNote")
        bottom_layout = QHBoxLayout(self.bottom_frame)
        bottom_layout.setContentsMargins(16, 8, 16, 8)
        note = _label("Planning only · Add and Run are separate actions.")
        note.setObjectName("AssistantMuted")
        bottom_layout.addWidget(note, 1)
        add_dialog_buttons(bottom_layout, actions=(), dismiss=self.close_button)
        layout.addWidget(self.bottom_frame)
        for button in (
            self.back_button,
            self.run_button,
            self.revise_button,
            self.preview_button,
            self.discussion_preview_button,
            self.cancel_button,
            self.recovery_button,
        ):
            button.setProperty("assistantAction", "secondary")
        for button in (
            self.settings_button,
            self.prepare_button,
            self.revise_button,
            self.send_button,
            self.apply_button,
            self.run_button,
            self.back_button,
            self.next_button,
            self.preview_button,
            self.discussion_preview_button,
            self.cancel_button,
        ):
            button.setAutoDefault(False)
        self.stage_tabs.currentChanged.connect(self._stage_changed)
        self.discussion_edit.textChanged.connect(self._sync_buttons)
        self._stage_changed(0)
        self._appearance_ready = True
        self._appearance_updating = False
        self._refresh_appearance()
        self.refresh_sources()
        self.provider_combo.currentIndexChanged.connect(self._provider_changed)
        self.source_combo.currentIndexChanged.connect(self._source_selection_changed)
        self.output_folder_combo.currentIndexChanged.connect(
            self._output_folder_choice_changed
        )
        for signal in (
            self.goal_edit.textChanged,
            self.address_edit.textChanged,
            self.kind_combo.currentIndexChanged,
            self.model_combo.currentTextChanged,
            self.local_context_spin.valueChanged,
            self.codex_model_edit.textChanged,
            self.api_model_edit.textChanged,
            self.api_key_env_edit.textChanged,
            self.reasoning_combo.currentTextChanged,
            self.attempts_spin.valueChanged,
            self.timeout_spin.valueChanged,
        ):
            signal.connect(self._invalidate_request)
        self.images_checkbox.toggled.connect(self._image_sharing_changed)
        self.output_assessment_checkbox.toggled.connect(
            self._assessment.permissions_changed
        )
        self.initiative_slider.valueChanged.connect(self._initiative_changed)
        self.json_edit.textChanged.connect(self._edited_proposal)
        self.consent_checkbox.toggled.connect(self._consent_changed)
        self.advanced_checkbox.toggled.connect(self.advanced_group.setVisible)
        self.advanced_group.hide()
        from napari_vipp.ui.workflow_assistant_sessions import (
            AssistantSessionController,
        )

        self._sessions = AssistantSessionController(self, assert_session_destination)
        heading.insertWidget(
            heading.indexOf(self.settings_button), self._sessions.button
        )
        self._provider_changed()
        self._update_chat()
        # Read-only status once; never opens login, captures images or starts AI.
        self._refresh_codex_connection()

    def _provider(self):
        return self.provider_combo.currentData()

    def _local_output_sources(self):
        """Private physical bindings only; display names are never filesystem paths."""
        try:
            paths = self._output_sources(self.source_combo.currentData())
            if not isinstance(paths, tuple):
                raise TypeError("Expected verified source paths.")
            return paths
        except (OSError, ValueError, TypeError) as exc:
            raise OutputDestinationError(
                "The input file location could not be verified. "
                "Choose an output folder yourself."
            ) from exc

    def _resolve_local_output(self, answer):
        paths = self._local_output_sources()
        if is_same_input_folder_answer(answer) and not paths:
            raise OutputDestinationError(
                "This input has no verified file folder, as with a sample or live "
                "image. Choose an output folder yourself."
            )
        return resolve_output_destination(answer, source_paths=paths)

    def _update_output_folder_status(self):
        destination = self._output_destination
        self.output_folder_status.setText(
            self._output_folder_error
            or (
                "Local output folder: "
                + destination.directory
                + "\nSend and Add never save files."
                if destination is not None
                else "Choose a folder before adding a Save step. "
                "Folder paths stay on this computer; Send and Add never save files."
            )
        )

    def _set_output_destination(self, destination, *, authored_text=""):
        """A local user choice revokes old plans; it never sends or writes."""
        if destination != self._output_destination:
            if not self._retain_for_revision(
                "The output folder choice changed.",
                "The output folder is selected locally. Nothing is sent or saved.",
                output_choice_changed=destination is not None,
            ):
                raise OutputDestinationError(
                    "The input or workflow changed. Review the input before "
                    "choosing the output folder again."
                )
            self._output_destination = destination
            if self._image_draft_guard is not None:
                context, owner = self._image_draft_guard
                self._image_draft_guard = (
                    replace(context, output_destination=destination),
                    owner,
                )
        if destination is not None:
            for text in (destination.directory, authored_text.strip()):
                if text and not is_same_input_folder_answer(text):
                    self._output_redactions.update(
                        (text, text.replace("\\", "/"), text.replace("/", "\\"))
                    )
        mode = (
            "ask"
            if destination is None
            else "same"
            if destination.mode == "same_input_folder"
            else "directory"
        )
        self.output_folder_combo.blockSignals(True)
        self.output_folder_combo.setCurrentIndex(
            self.output_folder_combo.findData(mode)
        )
        self.output_folder_combo.blockSignals(False)
        self._output_folder_error = ""
        self._update_output_folder_status()
        self._sync_buttons()

    def _output_folder_choice_changed(self, *_):
        if self._busy:
            return
        mode = self.output_folder_combo.currentData()
        if mode == "directory":
            self._choose_output_folder()
            return
        try:
            destination = (
                self._resolve_local_output("same folder as input")
                if mode == "same"
                else None
            )
            self._set_output_destination(destination)
        except (ValueError, RuntimeError) as exc:
            self._retain_for_revision(
                "The output folder needs attention.",
                "Choose an available output folder before adding a Save step.",
            )
            self._output_destination = None
            self._output_folder_error = str(exc)
            self._update_output_folder_status()
            self._set_status("warning", f"Choose an output folder: {exc}")
            self._sync_buttons()

    def _choose_output_folder(self):
        if self._busy:
            return
        destination = self._output_destination
        folder = QFileDialog.getExistingDirectory(
            self,
            "Choose workflow output folder",
            destination.directory if destination else "",
            QFileDialog.ShowDirsOnly
            | QFileDialog.DontResolveSymlinks
            | QFileDialog.ReadOnly,
        )
        if not folder:
            mode = (
                "same"
                if destination is not None and destination.mode == "same_input_folder"
                else "directory"
                if destination is not None
                else "ask"
            )
            self.output_folder_combo.blockSignals(True)
            self.output_folder_combo.setCurrentIndex(
                self.output_folder_combo.findData(mode)
            )
            self.output_folder_combo.blockSignals(False)
            return
        try:
            self._set_output_destination(
                self._resolve_local_output(folder), authored_text=folder
            )
        except (ValueError, RuntimeError) as exc:
            self._output_folder_error = str(exc)
            self._update_output_folder_status()
            self._set_status("warning", f"Cannot use this output folder: {exc}")

    def _source_selection_changed(self, *_):
        self._output_destination = None
        self._output_resolved_choices.clear()
        self._output_folder_error = ""
        self.output_folder_combo.blockSignals(True)
        self.output_folder_combo.setCurrentIndex(0)
        self.output_folder_combo.blockSignals(False)
        self._update_output_folder_status()
        self._invalidate_request()

    def _resolve_user_output_choice(self, objective):
        """Only exact user-authored answers/messages may bind a local folder."""
        choices = [
            (
                "goal",
                objective.objective
                if isinstance(objective, ProposalDialogue)
                else objective,
            )
        ]
        if isinstance(objective, ProposalDialogue):
            choices.extend(objective.clarifications)
        candidates = [
            (question, text)
            for question, text in choices
            if is_same_input_folder_answer(text) or Path(text.strip()).is_absolute()
        ]
        if candidates and candidates[-1] not in self._output_resolved_choices:
            _, answer = candidates[-1]
            self._set_output_destination(
                self._resolve_local_output(answer), authored_text=answer
            )
            # Do not replay an older answer after a newer local selection.
            self._output_resolved_choices.update(candidates)
        elif (
            self.output_folder_combo.currentData() == "same"
            and self._output_destination is None
        ):
            self._set_output_destination(
                self._resolve_local_output("same folder as input")
            )

    def _public_output_objective(self, objective):
        """Redact approved local folders in a detached request, not in chat/history."""

        def redact(text):
            for path in sorted(self._output_redactions, key=len, reverse=True):
                text = re.sub(
                    re.escape(path),
                    "the output folder selected locally",
                    text,
                    flags=re.IGNORECASE,
                )
            return text

        if isinstance(objective, ProposalDialogue):
            return ProposalDialogue(
                redact(objective.objective),
                tuple(
                    (redact(question), redact(answer))
                    for question, answer in objective.clarifications
                ),
                tuple(redact(question) for question in objective.deferred_questions),
            )
        return redact(objective)

    def _update_output_saving_summary(self):
        if self._prepared is not None:
            added = set(self._prepared.added_node_ids)
            self._output_plan_paths = tuple(
                node.params["path"]
                for node in self._prepared.candidate.nodes
                if node.id in added
                and node.operation_id == "save_output"
                and isinstance(node.params.get("path"), str)
            )
        current = self._draft_authorized and self._latest_selected()
        visible = (
            bool(self._output_plan_paths)
            and current
            and (self._prepared is not None or bool(self._added_ids))
        )
        names = ", ".join(Path(path).name for path in self._output_plan_paths[:2])
        if len(self._output_plan_paths) > 2:
            names += f" (+{len(self._output_plan_paths) - 2} more)"
        self.output_saving_label.setText(
            "Save on explicit execution: "
            + names
            + "\nLocal paths on hover · no overwriting"
            if visible
            else ""
        )
        self.output_saving_label.setToolTip(
            "Send and Add never save files. Normal manual-node checks still apply.\n"
            + "\n".join(self._output_plan_paths)
            if visible
            else ""
        )
        self.output_saving_label.setAccessibleDescription(
            "\n".join(self._output_plan_paths) if visible else ""
        )
        self.output_saving_label.setVisible(visible)

    def _planning_preferences(self):
        return PlanningPreferences(
            mode=_INITIATIVE_CHOICES[self.initiative_slider.value()][0]
        )

    def _uses_best_judgment(self):
        return self._planning_preferences().mode == "use_best_judgment"

    def _initiative_copy(self):
        # Keep disclosure/accessibility plain; formatting is presentation only.
        return _INITIATIVE_CHOICES[self.initiative_slider.value()][1:]

    def _update_initiative_text(self):
        label, explanation = self._initiative_copy()
        self.initiative_explanation.setText(
            f"<b>{html.escape(label)}</b> — {html.escape(explanation)}"
        )
        self.initiative_slider.setAccessibleDescription(f"{label}. {explanation}")
        self.initiative_slider.setToolTip(explanation)

    def _initiative_changed(self, *_):
        self._update_initiative_text()
        label, _ = self._initiative_copy()
        self._retain_for_revision(
            f"Assistant initiative changed to {label}.",
            f"{label} selected. Send your goal when you're ready.",
        )

    def _refresh_appearance(self):
        if not getattr(self, "_appearance_ready", False) or self._appearance_updating:
            return
        self._appearance_updating = True
        try:
            self._theme_probe.ensurePolished()
            palette = self._theme_probe.palette()
            stylesheet = assistant_stylesheet(palette)
            if stylesheet != self.styleSheet():
                self.setStyleSheet(stylesheet)
            for editor in (self.json_edit, self.request_edit, self.trace_edit):
                editor.refresh_highlighting()
            self.settings_button.setIcon(toolbar_icon("settings", palette))
            self.header_icon.setPixmap(
                toolbar_icon("assistant", palette).pixmap(22, 22)
            )
            self.source_icon.setPixmap(toolbar_icon("image", palette).pixmap(22, 22))
            for button, kind in (
                (self.prepare_button, "send"),
                (self.send_button, "send"),
                (self.revise_button, "send"),
                (self.recovery_button, "send"),
                (self.next_button, "next"),
                (self.back_button, "previous"),
                (self.apply_button, "checklist"),
                (self.run_button, "next"),
                (self.preview_button, "image"),
                (self.discussion_preview_button, "image"),
                (self.output_folder_button, "open"),
                (self._assessment.preview_button, "image"),
                (self._assessment.assess_button, "send"),
                (self._assessment.revise_button, "send"),
            ):
                icon = assistant_action_icon(
                    kind,
                    palette,
                    primary=button.property("assistantAction") == "primary",
                )
                if isinstance(button, AssistantSendButton):
                    button.set_idle_icon(icon)
                else:
                    button.setIcon(icon)
            self._update_provider_status()
            self.original_goal_toggle.refresh_disclosure()
            self.conversation_toggle.refresh_disclosure()
            self.assumptions_toggle.refresh_disclosure()
            self.review_checks_toggle.refresh_disclosure()
            self._update_chat()
            if self._settings_dialog is not None:
                self._settings_dialog._refresh_appearance()
                self._settings_dialog._refresh_connection_state()
        finally:
            self._appearance_updating = False

    def changeEvent(self, event):  # noqa: N802
        super().changeEvent(event)
        if event.type() in {
            QEvent.PaletteChange,
            QEvent.ApplicationPaletteChange,
            QEvent.StyleChange,
        } and getattr(self, "_appearance_ready", False):
            # A global stylesheet switch is still repolishing widgets here.
            # Defer our own stylesheet update to avoid nesting that Qt cascade.
            self._appearance_timer.start(0)

    def eventFilter(self, watched, event):  # noqa: N802
        if (
            watched is getattr(self, "_theme_probe", None)
            and event.type()
            in {
                QEvent.PaletteChange,
                QEvent.ApplicationPaletteChange,
                QEvent.StyleChange,
            }
            and getattr(self, "_appearance_ready", False)
        ):
            self._appearance_timer.start(0)
        return super().eventFilter(watched, event)

    def _stage_changed(self, index):
        self._sync_navigation()

    def _next_stage_clicked(self):
        # Returning to an existing conversation must never resubmit a goal.
        if self._busy or not self._drafts:
            return
        index = self.stage_tabs.currentIndex()
        if index in (0, 1):
            self.stage_tabs.setCurrentIndex(index + 1)

    def _sync_navigation(self):
        # Later stages are for returned drafts/history, not another send action.
        # Browsing an old draft never restores its Add/Run authority.
        browsable = bool(self._drafts)
        index = self.stage_tabs.currentIndex()
        self.back_button.setEnabled(index > 0)
        self.next_button.setText("Next: Discussion" if index == 0 else "View plan")
        self.next_button.setEnabled(index in (0, 1) and browsable and not self._busy)
        self.next_button.setVisible(index in (0, 1) and browsable)
        self.stage_label.setText(
            f"Step {index + 1} of 3 · "
            + ("Set your goal", "Discuss and refine", "Review drafts and use a plan")[
                index
            ]
        )
        self.back_button.setText(("Back", "Back: Goal", "Back: Discussion")[index])
        zone_width = max(
            self.back_button.sizeHint().width(), self.footer_actions.sizeHint().width()
        )
        self._navigation.setColumnMinimumWidth(0, zone_width)
        self._navigation.setColumnMinimumWidth(2, zone_width)

    def _open_settings(self):
        if self._busy:
            return
        dialog = _AssistantSettingsDialog(self)
        self._settings_dialog = dialog
        self._sync_buttons()
        try:
            if dialog.exec() != QDialog.Accepted:
                return
            controls = (
                (self.provider_combo, dialog.provider_combo, "index"),
                (self.kind_combo, dialog.kind_combo, "index"),
                (self.address_edit, dialog.address_edit, "text"),
                (self.model_combo, dialog.model_combo, "combo"),
                (self.local_context_spin, dialog.local_context_spin, "value"),
                (self.codex_model_edit, dialog.codex_model_edit, "text"),
                (self.api_model_edit, dialog.api_model_edit, "text"),
                (self.api_key_env_edit, dialog.api_key_env_edit, "text"),
                (self.reasoning_combo, dialog.reasoning_combo, "combo"),
                (self.attempts_spin, dialog.attempts_spin, "value"),
                (self.timeout_spin, dialog.timeout_spin, "value"),
            )
            changed = False
            for active, draft, kind in controls:
                getter, setter = {
                    "index": ("currentIndex", "setCurrentIndex"),
                    "text": ("text", "setText"),
                    "combo": ("currentText", "setCurrentText"),
                    "value": ("value", "setValue"),
                }[kind]
                value = getattr(draft, getter)()
                if value != getattr(active, getter)():
                    changed = True
                active.blockSignals(True)
                try:
                    getattr(active, setter)(value)
                finally:
                    active.blockSignals(False)
            if changed:
                self._provider_changed()
                self._append_chat(
                    "VIPP",
                    "Settings changed. Previous drafts are kept for reference; "
                    "send a fresh request before adding a plan.",
                )
            self._update_provider_status()
        finally:
            self._settings_dialog = None
            dialog.deleteLater()
            self._sync_buttons()

    def _update_provider_status(self):
        provider = self._provider()
        if provider == "codex":
            status = (
                "Checking connection…"
                if self._connection_busy
                else "Setup needs attention"
                if self._codex_connection_issue == "profile"
                else "Connected"
                if self._codex_authenticated
                else "Not connected"
            )
            text = f"ChatGPT · {status}"
            state = (
                "checking"
                if self._connection_busy
                else "connected"
                if self._codex_authenticated
                else "disconnected"
            )
        elif provider == "local":
            text = "Local model · " + (
                self.model_combo.currentText() or "Choose in Settings"
            )
            state = "local"
        elif provider == "openai":
            text = "OpenAI API · " + (
                self.api_model_edit.text() or "Choose in Settings"
            )
            state = "local"
        else:
            text = "Example · No AI"
            state = "example"
        self.provider_status_label.setText(text)
        self.provider_status_label.setToolTip(
            self.connection_label.text() if provider == "codex" else text
        )
        self.provider_status_icon.setProperty("status_state", state)
        self.provider_status_icon.setAccessibleName(text)
        self._theme_probe.ensurePolished()
        colors = theme_colors(self._theme_probe.palette())
        pixmap = QPixmap(18, 18)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(
            QPen(
                colors.success.foreground
                if state == "connected"
                else colors.muted_text,
                1.4,
            )
        )
        painter.drawEllipse(2, 2, 14, 14)
        if state == "connected":
            painter.drawLine(5, 9, 8, 12)
            painter.drawLine(8, 12, 13, 6)
        elif state in {"disconnected", "checking"}:
            painter.drawLine(9, 5, 9, 10)
            painter.drawPoint(9, 13)
        else:
            painter.drawPoint(9, 9)
        painter.end()
        self.provider_status_icon.setPixmap(pixmap)

    def _append_chat(self, speaker, text):
        # A rebuilt/cleared synthetic history must not inherit an old index's
        # attribution. Only the typed reply helper supplies it for a new entry.
        self._chat_replies.pop(len(self._chat), None)
        self._chat.append((speaker, str(text)))
        self._update_chat()

    def _append_chat_reply(self, question, answer, *, is_question=True):
        index = len(self._chat)
        self._chat.append(("You", f"{question}\n{answer}"))
        self._chat_replies[index] = _ChatReply(question, answer, is_question)
        self._update_chat()

    def _toggle_original_goal(self, expanded):
        self.original_goal_frame.layout().setContentsMargins(
            14, 10 if expanded else 0, 14, 12 if expanded else 0
        )
        self.original_goal_label.setVisible(expanded)

    def _toggle_conversation(self, expanded):
        self.conversation_frame.layout().setContentsMargins(
            14, 14 if expanded else 0, 14, 14 if expanded else 0
        )
        self.transcript_edit.setVisible(expanded)
        if expanded:
            self.transcript_edit.queue_fit()

    def _update_chat(self):
        self._theme_probe.ensurePolished()
        colors, _panel, _raised, _input_surface, user_surface = assistant_surfaces(
            self._theme_probe.palette()
        )
        assistant_surface, assistant_border, quote_surface, quote_border = (
            assistant_chat_surfaces(self._theme_probe.palette())
        )
        blocks = []
        last_goal_index = next(
            (
                index
                for index in range(len(self._chat) - 1, -1, -1)
                if self._chat[index][0] == "You · Goal"
            ),
            -1,
        )
        goal = self._chat[last_goal_index][1] if last_goal_index >= 0 else ""
        if goal != self.original_goal_label.text():
            self.original_goal_label.setText(goal)
            self.original_goal_toggle.setChecked(False)
        self.original_goal_frame.setVisible(bool(goal))
        question_message_index = -1
        question_summary = ""
        if (
            self._question_fields
            and not self.questions_group.isHidden()
            and 0 <= self._current_draft_index < len(self._drafts)
        ):
            draft = self._drafts[self._current_draft_index]
            if draft.questions == tuple(
                question for question, _ in self._question_fields
            ):
                question_summary = draft.summary
                question_message = draft.summary + "\n\n" + "\n".join(draft.questions)
                question_message_index = next(
                    (
                        index
                        for index in range(len(self._chat) - 1, -1, -1)
                        if self._chat[index] == ("Assistant", question_message)
                    ),
                    -1,
                )
        for index, (speaker, text) in enumerate(self._chat):
            # Keep the literal goal in _chat and the request dialogue; its
            # expandable reference should not occupy the conversation twice.
            if index == last_goal_index:
                continue
            if index == question_message_index:
                # Keep the complete literal message in history/drafts. Only the
                # current form's question appendix is duplicated presentation.
                text = question_summary + (
                    "\n\nAnswer the questions in “A little more information”."
                )
            user = speaker.startswith("You")
            background = user_surface if user else assistant_surface
            reply = self._chat_replies.get(index)
            if (
                speaker != "You"
                or reply is None
                or text != f"{reply.question}\n{reply.answer}"
            ):
                # Never guess question/answer boundaries by splitting text.
                reply = None
            body = (
                '<p style="margin: 0px;">'
                + html.escape(text).replace("\n", "<br>")
                + "</p>"
            )
            if reply is not None:
                body = (
                    '<p style="margin: 0px;">'
                    + html.escape(reply.answer).replace("\n", "<br>")
                    + "</p>"
                )
                if reply.is_question:
                    body = (
                        '<table width="100%" cellspacing="0" cellpadding="10" '
                        + f'style="border: 1px solid {quote_border.name()};">'
                        + '<tr><td bgcolor="'
                        + quote_surface.name()
                        + '" style="color: '
                        + colors.text.name()
                        + ';"><p style="margin-top: 0px; margin-bottom: 5px;">'
                        + '<b>Assistant’s question</b></p><p style="margin: 0px;">'
                        + html.escape(reply.question).replace("\n", "<br>")
                        + '</p></td></tr></table><p style="margin-top: 10px; '
                        + 'margin-bottom: 5px;"><b>Your answer</b></p>'
                        + body
                    )
            outline = (
                "" if user else f' style="border: 1px solid {assistant_border.name()};"'
            )
            blocks.append(
                f'<table width="{74 if user else 82}%" '
                + f'align="{"right" if user else "left"}" '
                + 'cellspacing="0" cellpadding="12"'
                + outline
                + '><tr><td bgcolor="'
                + background.name()
                + '"><p style="margin-top: 0px; margin-bottom: 8px; color: '
                + colors.text.name()
                + ';"><b>'
                + html.escape(speaker)
                + "</b></p>"
                + body
                # Qt collapses a whitespace-only paragraph, letting the next
                # table touch this outline. Keep a small, real spacer instead.
                + '</td></tr></table><p style="margin-top: 8px; '
                + 'margin-bottom: 8px; font-size: 1px;">&#160;</p>'
            )
        self.transcript_edit.setHtml(
            "".join(blocks)
            or "<p>Your conversation will appear here after you send a goal.</p>"
        )
        self.transcript_edit.queue_fit()

    def _send_goal_clicked(self):
        if self._busy or not self.prepare_button.isEnabled():
            return
        self.consent_checkbox.setChecked(self._provider() != "example")
        self.prepare()

    def _send_answers_clicked(self):
        if self._busy or not self.revise_button.isEnabled():
            return
        self.consent_checkbox.setChecked(True)
        self.revise_plan()

    def _send_message_clicked(self):
        if self._busy or not self.send_button.isEnabled():
            return
        # The original goal remains a separate, bounded field in the dialogue.
        self._extra_message = self.discussion_edit.toPlainText()
        self.consent_checkbox.setChecked(True)
        self.prepare()
        self._extra_message = ""

    def _append_request_chat(self, objective):
        goal = (
            objective.objective
            if isinstance(objective, ProposalDialogue)
            else objective
        )
        previous_goal = next(
            (text for speaker, text in reversed(self._chat) if speaker == "You · Goal"),
            None,
        )
        # A return to an earlier goal is a new conversation event. Deduplicate
        # only consecutive goals, otherwise the expandable reference is stale.
        if previous_goal != goal:
            self._append_chat("You · Goal", goal)
        if isinstance(objective, ProposalDialogue):
            previous = dict(self._clarifications)
            for question, answer in objective.clarifications:
                if previous.get(question) != answer:
                    self._append_chat_reply(
                        question,
                        answer,
                        is_question=(question, answer) != self._outgoing_message_pair,
                    )
            if objective.deferred_questions:
                self._append_chat(
                    "You · Preference",
                    "Use best judgment for these unanswered draft choices:\n"
                    + "\n".join(objective.deferred_questions),
                )

    def _latest_selected(self):
        return self.draft_combo.currentIndex() == self._current_draft_index

    def _record_draft(self, prepared):
        self._recovery_kind = None
        self._recovering = False
        pipeline = prepared.candidate.to_pipeline()
        draft = _Draft(
            prepared.summary,
            tuple(
                (pipeline.nodes[key].title, dict(prepared.node_reasons)[key])
                for key in prepared.added_node_ids
            ),
            prepared.assumptions,
            prepared.questions,
            self.json_edit.toPlainText(),
            self.preview_edit.toPlainText(),
        )
        if not self._drafts or draft != self._drafts[-1] or not self._draft_authorized:
            self._drafts.append(draft)
            self.draft_combo.blockSignals(True)
            self.draft_combo.addItem(
                f"Draft {len(self._drafts)} · "
                + ("Questions to discuss" if draft.questions else "Plan checked")
            )
            self.draft_combo.blockSignals(False)
            self._append_chat(
                "Assistant",
                draft.summary
                + (
                    "\n\n" + "\n".join(draft.questions)
                    if draft.questions
                    else "\n\nA checked draft is available in Plans."
                ),
            )
        self._current_draft_index = len(self._drafts) - 1
        self._draft_authorized = True
        self._live_proposal_text = draft.proposal_text
        self._live_plan_text = draft.plan_text
        self.draft_combo.setCurrentIndex(self._current_draft_index)
        self.json_edit.setReadOnly(False)
        self.draft_note.setText(
            "Latest draft · Answer questions in Discussion."
            if draft.questions
            else "Latest draft · Checked and ready for review."
        )
        self._sync_navigation()
        self.stage_tabs.setCurrentIndex(1 if prepared.questions else 2)
        self._update_chat()
        if prepared.questions:
            self.conversation_toggle.setChecked(False)
            QTimer.singleShot(0, self._show_current_questions)

    def _show_current_questions(self):
        if (
            not self._closed
            and self.stage_tabs.currentIndex() == 1
            and self._question_fields
        ):
            # The text browsers and wrapped labels finish fitting first. Bring
            # the new answer section into view without focusing or sending it.
            self.discussion_scroll.ensureWidgetVisible(
                self.questions_layout.itemAt(0).widget(), 0, 24
            )

    def _draft_selected(self, index):
        if not 0 <= index < len(self._drafts):
            return
        draft = self._drafts[index]
        self._clear_steps()
        self.summary_label.setText(draft.summary)
        self._render_steps(draft.steps)
        self._render_assumptions(
            draft.assumptions, draft_key=(index, draft.proposal_text)
        )
        current = self._latest_selected() and self._draft_authorized
        self.json_edit.blockSignals(True)
        self.json_edit.setPlainText(
            self._live_proposal_text if current else draft.proposal_text
        )
        self.json_edit.blockSignals(False)
        self.preview_edit.setPlainText(
            self._live_plan_text if current else draft.plan_text
        )
        self.json_edit.setReadOnly(not current or bool(self._added_ids))
        self.draft_note.setText(
            "Added to workflow · Run is a separate action."
            if current and self._added_ids
            else "Latest draft · Questions remain in Discussion."
            if current and draft.questions
            else "Latest draft · Review the steps before adding."
            if current
            else "Read-only draft · Send a fresh request to create a current plan."
        )
        self._sync_buttons()

    def _run_added_plan(self):
        if (
            self._busy
            or not self._run
            or not self._added_ids
            or not (self._draft_authorized and self._latest_selected())
        ):
            return False
        self._assessment.note_run(self._added_ids, self.source_combo.currentData())
        self._calculation_failure_message = ""
        try:
            accepted = self._run(self._added_ids)
        except (ValueError, RuntimeError) as exc:
            self._added_ids = ()
            self._assessment.invalidate(
                "The workflow did not start. No results to assess."
            )
            self._set_status("error", f"Workflow was not started: {exc}")
            return False
        else:
            self._added_ids = ()
            if self._calculation_failure_message:
                # A synchronous terminal callback already displayed the actual
                # failure and may have queued a bounded replacement request.
                self._sync_buttons()
                return accepted is not False
            self._append_chat(
                "VIPP",
                "Workflow calculation requested. Follow progress and review results "
                "in VIPP. Use Assess results after it finishes to review this run."
                if self.output_assessment_checkbox.isChecked()
                else "Workflow calculation requested. Output assessment is off; "
                "review the results in VIPP.",
            )
            self._set_status(
                "info", "Calculation requested using VIPP's normal workflow controls."
            )
            if accepted is False:
                self._assessment.invalidate(
                    "No current assessable results were accepted. "
                    "Check VIPP's calculation status."
                )
        self._sync_buttons()
        return accepted is not False

    def assessment_results_ready(self, candidate_ids, run_id):
        """Notification only; capture still requires the exact completed-run owner."""
        self._assessment.results_ready(candidate_ids, run_id)
        self._refinement.results_ready()

    def assessment_calculation_failed(self, failed_run, message):
        """A calculation error is not a result assessment or a changed goal."""
        self._calculation_failure_message = str(message)
        self._assessment.invalidate(str(message))
        self._append_chat("VIPP", str(message))
        self._set_status("error", str(message))
        self._refinement.calculation_failed(failed_run, str(message))
        self._signals.calculation_failed.emit(failed_run, str(message))
        self._sync_buttons()

    def assessment_results_unavailable(self, message):
        if hasattr(self, "_refinement"):
            self._refinement.results_unavailable(message)
        self._assessment.invalidate(message)
        self._sync_buttons()

    def assessment_session_changed(self):
        """A tab switch cannot inherit another session's report or authority."""
        self._invalidate_request()

    def refresh_sources(self):
        previous = self.source_combo.currentData()
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        for source_id, label in self._sources():
            self.source_combo.addItem(label, source_id)
        index = self.source_combo.findData(previous)
        if index >= 0:
            self.source_combo.setCurrentIndex(index)
        self.source_combo.blockSignals(False)
        if previous != self.source_combo.currentData() and hasattr(
            self, "apply_button"
        ):
            self._source_selection_changed()

    def _provider_changed(self, *_):
        self._cancel_connection(announce=False)
        provider = self._provider()
        self.local_group.setVisible(provider == "local")
        self.codex_group.setVisible(provider == "codex")
        self.codex_settings_widget.setVisible(provider == "codex")
        self.reference_note.setVisible(provider == "example")
        self.timeout_spin.setRange(5, MAX_PROPOSAL_TIMEOUT_SECONDS)
        if provider == "example":
            self.images_checkbox.setChecked(False)
        self.images_checkbox.setEnabled(provider != "example")
        if provider == "openai" and hasattr(self, "_investigation"):
            self._investigation.checkbox.setChecked(True)
        if provider == "example":
            self.output_assessment_checkbox.setChecked(False)
        self.output_assessment_checkbox.setEnabled(provider != "example")
        self.output_assessment_description.setText(
            "After Run, assess actual masks, meshes and measurements against your "
            "goal. Assess results or Start refinement shares selected output previews."
            if provider == "codex"
            else "During investigation or Start refinement, assess actual masks, "
            "meshes and measurements with output previews. OpenAI API uses "
            "these shared investigation tools, not the separate Assess results action."
            if provider == "openai"
            else "During investigation, assess actual masks, meshes and measurements "
            "with output previews. A local model must support images when "
            "previews are included. Ordinary local assessment is measurements only."
        )
        self.consent_checkbox.hide()
        self.preview_button.setVisible(provider != "example")
        self.prepare_button.setText(
            "Show example plan" if provider == "example" else "Send goal"
        )
        self.consent_checkbox.setText(
            "Allow this request to be sent to Codex / OpenAI."
            if provider == "codex"
            else "Allow this request to OpenAI API (separate billing)."
            if provider == "openai"
            else "Allow this request to my selected trusted local server."
        )
        self.sharing_note.setText(
            "Clicking Send allows this request to Codex / OpenAI. Your goal and "
            "image metadata are shared; image previews "
            "are included only if you opt in. Your ChatGPT plan's data policies "
            "apply: no-training and no-retention are different. No raw volume, "
            "source paths or source names are sent. Repairs may also share the "
            "previous response and check feedback."
            if provider == "codex"
            else "Clicking Send allows this request to your trusted local server. "
            "Requests include your goal and image metadata. Investigation shares "
            "input/output previews only when selected; it requires an image-capable "
            "model for those previews. No raw volume, paths or source names are sent. "
            "A local server may itself forward data. Repairs also include the "
            "previous response "
            "and check feedback."
            if provider == "local"
            else "Clicking Start allows the investigation to use OpenAI API, with "
            "separate billing. Your goal and metadata are shared; input and output "
            "previews are independent choices. No raw volume, source paths or "
            "source names are sent. Repairs may share the prior response and feedback."
            if provider == "openai"
            else "No request is sent anywhere."
        )
        self.discussion_sharing_note.setText(
            "Send shares your goal, answers and image metadata with Codex / OpenAI. "
            "Image previews are included only if selected in Goal."
            if provider == "codex"
            else "Send shares your goal, answers and image metadata with your trusted "
            "local server. Investigation previews follow your choices in Goal."
            if provider == "local"
            else "Start shares your goal, answers and metadata with OpenAI API; "
            "investigation previews follow your choices in Goal. "
            "Separate billing applies."
            if provider == "openai"
            else "The example does not use AI or send requests."
        )
        self.discussion_sharing_note.setToolTip(self.sharing_note.text())
        self._update_provider_status()
        self._invalidate_request()

    def _set_status(self, tone, text):
        self.status_label.setText(text)
        self.status_icon.setProperty("tone", tone)
        colors = theme_colors(self.palette())
        color = getattr(colors, tone, colors.info).accent
        pixmap = QPixmap(22, 22)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(color, 2.3))
        painter.drawEllipse(2, 2, 18, 18)
        if tone == "success":
            painter.drawLine(6, 11, 9, 14)
            painter.drawLine(9, 14, 16, 7)
        elif tone == "error":
            painter.drawLine(7, 7, 15, 15)
            painter.drawLine(7, 15, 15, 7)
        else:
            painter.drawLine(11, 6, 11, 12)
            painter.drawPoint(11, 16)
        painter.end()
        self.status_icon.setPixmap(pixmap)
        self.status_icon.setAccessibleName(
            {
                "success": "Ready to add",
                "error": "Plan not ready",
                "warning": "Needs attention",
            }.get(tone, "Information")
        )

    def _invalidate_proposal(self):
        self._validation_timer.stop()
        self._prepared = None
        self._added_ids = ()
        self.apply_button.setEnabled(False)
        self.run_button.setEnabled(False)

    def _clear_steps(self):
        while self.steps_layout.count():
            item = self.steps_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._step_widgets.clear()

    def _clear_review(self, *, reset_disclosures=True):
        self._clear_steps()
        self._output_plan_paths = ()
        self.output_saving_label.clear()
        self.output_saving_label.hide()
        while self.questions_layout.rowCount():
            self.questions_layout.removeRow(0)
        self._question_fields.clear()
        self.questions_group.hide()
        self.revise_button.hide()
        self.assumptions_label.clear()
        self.assumptions_label.setAccessibleDescription("")
        self.review_checks_label.clear()
        self.review_checks_label.setAccessibleDescription("")
        for frame in (self.assumptions_frame, self.review_checks_frame):
            frame.hide()
        if reset_disclosures:
            self._review_draft_key = None
            self.assumptions_toggle.setChecked(False)
            self.review_checks_toggle.setChecked(False)
        self.summary_label.setText("Your proposed steps will appear here.")
        self.preview_edit.clear()
        self._update_chat()

    def _invalidate_request(self, *_):
        if hasattr(self, "_investigation"):
            self._investigation.context_changed()
        if hasattr(self, "_refinement"):
            self._refinement.context_changed()
        if hasattr(self, "_assessment"):
            self._assessment.invalidate(
                "The goal, input or connection changed. "
                "Add and run a fresh plan to assess it."
            )
        self.cancel_request(announce=False)
        self._recovery_kind = None
        self._recovering = False
        self._context = self._owner = None
        self._image_draft_guard = None
        self._clarifications.clear()
        self._pending_clarifications = None
        self._invalidate_proposal()
        self._draft_authorized = False
        self.consent_checkbox.setChecked(False)
        self.request_edit.clear()
        self.trace_edit.clear()
        self._clear_error_details()
        self._clear_review()
        if self.draft_combo.currentIndex() >= 0:
            self._draft_selected(self.draft_combo.currentIndex())
        self._progress_messages.clear()
        self.progress_label.clear()
        self._set_status(
            "info", "Choose your input and objective, then inspect and plan."
        )
        self._sync_buttons()

    def _image_sharing_changed(self, *_):
        """Retain answers only for a still-current draft; never reuse old pixels."""
        self._assessment.input_permissions_changed()
        choice = "included" if self.images_checkbox.isChecked() else "not included"
        self._retain_for_revision(
            f"Input image previews are now {choice}.",
            f"Input image previews will be {choice}. Nothing is sent until you send.",
        )

    def _retain_for_revision(
        self, reason, initial_status, *, output_choice_changed=False
    ):
        """Policy-only changes retain literal discussion, not stale authority."""
        self._refinement.context_changed()
        had_plan = bool(self._drafts)
        was_planning = self._busy and self._job_kind == "proposal"
        guard = (
            (self._context, self._owner)
            if self._context is not None
            else self._image_draft_guard
        )
        try:
            if (
                guard is not None
                and guard[0].source_id != self.source_combo.currentData()
            ):
                raise ProposalError("The previous draft has no current input.")
            if guard is not None:
                # A newly verified local folder replaces the old output policy,
                # not the scientific input. Check the same owner/source/graph
                # without requiring an obsolete destination to still exist.
                context = (
                    replace(guard[0], output_destination=None)
                    if output_choice_changed
                    else guard[0]
                )
                self._assert_current(context, guard[1])
        except (ValueError, RuntimeError):
            self._invalidate_request()
            self._recovery_kind = "goal"
            self._set_status(
                "warning",
                "The input or workflow changed. Review your input in Goal before "
                "sending a new request; the old plan's assumptions were not reused.",
            )
            self._sync_buttons()
            return False
        self.cancel_request(announce=False)
        self._recovering = False
        self._image_draft_guard = (
            (replace(guard[0], image_previews=()), guard[1]) if guard else None
        )
        self._context = self._owner = None
        self._request_objective = ""
        self._invalidate_proposal()
        self._draft_authorized = False
        self.consent_checkbox.setChecked(False)
        self.json_edit.blockSignals(True)
        self.json_edit.clear()
        self.json_edit.blockSignals(False)
        self.request_edit.clear()
        self.trace_edit.clear()
        self.preview_edit.clear()
        self._clear_error_details()
        self._progress_messages.clear()
        self.progress_label.clear()
        self._recovery_kind = "update" if had_plan or was_planning else None
        if self._recovery_kind:
            self.summary_label.setText(reason + " A new draft is needed.")
            self._set_status(
                "warning",
                reason
                + (
                    " The draft needs updating for this choice. "
                    if had_plan
                    else " The previous request was cancelled. "
                )
                + "Choose Update plan to use this choice. Your goal and answers "
                "are kept. Changing this choice does not send, add or run anything.",
            )
        else:
            self._set_status("info", initial_status)
        self._sync_buttons()
        return True

    def _recover_plan_clicked(self):
        """One explicit recovery click; use the normal guarded submission paths."""
        kind = self._recovery_kind
        if self._closed or self._busy or kind is None:
            return
        if kind == "settings":
            self._open_settings()
            return
        if kind == "details":
            self.stage_tabs.setCurrentIndex(2)
            self.advanced_checkbox.setChecked(True)
            self.tabs.setCurrentIndex(3)
            self.review_scroll.ensureWidgetVisible(self.advanced_group)
            return
        if kind == "goal":
            self.stage_tabs.setCurrentIndex(0)
            return
        if not self._answers_ready():
            self.stage_tabs.setCurrentIndex(1)
            self._show_current_questions()
            self._set_status(
                "warning",
                "Answer the questions in Discussion, or choose Use best judgment "
                "in Goal to leave those choices to the assistant. Nothing was sent.",
            )
            return
        action = self._retry_action if kind == "retry" else "goal"
        button = {
            "goal": self.prepare_button,
            "answers": self.revise_button,
            "message": self.send_button,
        }[action]
        if not button.isEnabled():
            return
        self._recovering = True
        self._recovery_submit_action = action
        {
            "goal": self._send_goal_clicked,
            "answers": self._send_answers_clicked,
            "message": self._send_message_clicked,
        }[action]()
        if self._busy and self._job_kind == "proposal":
            self._sending_action = "recovery"
        else:
            self._recovering = False
        self._sync_buttons()

    def _sync_recovery_button(self, available, connection_ready):
        kind = self._recovery_kind
        sending = self._busy and self._job_kind == "proposal" and self._recovering
        self.recovery_button.setVisible(kind is not None or sending)
        self.recovery_button.set_sending(sending)
        enabled = not self._busy
        tooltip = ""
        if sending:
            label = "Updating plan…" if kind == "update" else "Retrying…"
            enabled = False
        elif kind in {"update", "retry"}:
            if not self._answers_ready():
                label = "Answer questions"
                tooltip = "Open Discussion. This does not send a request."
            else:
                action = self._retry_action if kind == "retry" else "goal"
                label = (
                    "Refresh example"
                    if self._provider() == "example"
                    else "Update plan"
                    if kind == "update"
                    else "Retry message"
                    if action == "message"
                    and self.discussion_edit.toPlainText() == self._submitted_message
                    else "Send edited message"
                    if action == "message"
                    else "Retry answers"
                    if action == "answers"
                    else "Retry plan"
                )
                enabled = available and connection_ready
                if action == "message":
                    enabled = enabled and self.send_button.isEnabled()
                elif action == "answers":
                    enabled = enabled and self.revise_button.isEnabled()
                tooltip = (
                    "Send your current goal and answers using the current image "
                    "preview choice. A fresh plan is checked automatically; "
                    "nothing is added or run. "
                    + (
                        "Your visible message is included."
                        if action == "message"
                        else "An unsent message in Discussion is not included."
                    )
                )
        else:
            label = {
                "settings": "Check connection",
                "details": "Show details",
                "goal": "Review goal",
            }.get(kind, "Update plan")
            tooltip = "Open the relevant settings or details. No request is sent."
        self.recovery_button.setText(label)
        self.recovery_button.setEnabled(enabled)
        self.recovery_button.setToolTip(tooltip)

    def _check_image_draft_guard(self):
        if self._image_draft_guard is None:
            return
        try:
            context, owner = self._image_draft_guard
            if context.source_id != self.source_combo.currentData():
                raise ProposalError("The selected input changed.")
            self._assert_current(context, owner)
        except (ValueError, RuntimeError) as exc:
            self._invalidate_request()
            self._recovery_kind = "goal"
            raise ProposalError(
                "The input or workflow changed. Inspect it again before a new plan; "
                "the previous draft was not reused."
            ) from exc

    def _sync_buttons(self, *_):
        if hasattr(self, "_sessions"):
            self._sessions.sync()
        if hasattr(self, "_patching"):
            self._patching.sync()
        available = (
            not self._busy
            and self.source_combo.count() > 0
            and bool(self.goal_edit.toPlainText().strip())
        )
        connection_ready = self._provider() != "codex" or (
            self._codex_authenticated and not self._connection_busy
        )
        self.prepare_button.setEnabled(available and connection_ready)
        self.preview_button.setEnabled(available)
        self.discussion_preview_button.setEnabled(available)
        self.refresh_button.setEnabled(not self._busy)
        self.signin_button.setEnabled(
            not self._busy
            and not self._connection_busy
            and not self._codex_connection_issue
        )
        self.signin_button.setText(
            "Sign in again…"
            if self._codex_authenticated
            else "Signing in…"
            if self._connection_kind == "signin" and self._connection_busy
            else "Sign in to Codex for VIPP"
        )
        self.connection_refresh_button.setEnabled(
            not self._busy and not self._connection_busy
        )
        self.cancel_button.setEnabled(self._busy or self._connection_busy)
        for _, field in self._question_fields:
            field.setEnabled(not self._busy)
        self.apply_button.setEnabled(
            not self._busy
            and self._prepared is not None
            and self._prepared.can_apply
            and self._draft_authorized
            and self._latest_selected()
        )
        self.run_button.setEnabled(
            not self._busy
            and self._run is not None
            and bool(self._added_ids)
            and self._draft_authorized
            and self._latest_selected()
        )
        self.revise_button.setEnabled(
            not self._busy
            and bool(self._question_fields)
            and self._answers_ready()
            and self._provider() != "example"
            and connection_ready
        )
        self.send_button.setEnabled(
            available
            and connection_ready
            and self._provider() != "example"
            and bool(self.discussion_edit.toPlainText().strip())
            and self._answers_ready()
        )
        self.initiative_slider.setEnabled(not self._busy)
        self.output_folder_combo.setEnabled(not self._busy)
        self.output_folder_button.setEnabled(not self._busy)
        delegated = self._uses_best_judgment()
        self.questions_note.setText(
            "Answer what matters to you. Leave a choice blank to use the "
            "assistant's best judgment, then send for a fresh draft."
            if delegated
            else "Answer the questions below, then send your answers. "
            "You can also add a message in the next section."
        )
        for _, field in self._question_fields:
            field.setPlaceholderText(
                "Your answer, or leave blank to use best judgment…"
                if delegated
                else "Your answer…"
            )
        self.discussion_edit.setPlaceholderText(
            "Ask for a change or explain a preference… Unanswered draft choices "
            "will use the assistant's best judgment."
            if delegated
            else "Ask for a change or explain a preference… "
            "If there are questions above, answer them too."
        )
        self.settings_button.setEnabled(not self._busy)
        working = self._busy and self._job_kind not in {
            "investigation_pause",
            "investigation_review",
        }
        self.request_progress.setRange(0, 0 if working else 1)
        self.request_progress.setVisible(working)
        self.request_activity.setVisible(working)
        planning = self._busy and self._job_kind in {"proposal", "investigation"}
        self.request_elapsed_label.setVisible(
            (planning or (self._busy and self._job_kind == "assessment"))
            and self._request_elapsed.isValid()
            and self._request_clock_generation == self._generation
        )
        for button, action, label in (
            (self.prepare_button, "goal", "goal"),
            (self.send_button, "message", "message"),
            (self.revise_button, "answers", "answers"),
        ):
            sending = planning and self._sending_action == action
            button.set_sending(sending)
            button.setText(
                f"Sending {label}…"
                if sending
                else "Show example plan"
                if action == "goal" and self._provider() == "example"
                else f"Send {label}"
            )
        self._sync_navigation()
        self._sync_recovery_button(available, connection_ready)
        self._update_provider_status()
        self._update_output_saving_summary()
        self._assessment.sync()
        self._refinement.sync()
        if hasattr(self, "_investigation"):
            self._investigation.sync()
        if hasattr(self, "_live_agent"):
            self._live_agent.sync()
        if self._settings_dialog is not None:
            self._settings_dialog._refresh_connection_state()

    def _ensure_context(self, objective=None, *, approved_request=False):
        objective = (
            self.goal_edit.toPlainText().strip() if objective is None else objective
        )
        was_approved = approved_request and self.consent_checkbox.isChecked()
        self._resolve_user_output_choice(objective)
        self._check_image_draft_guard()
        # Resolving this click's literal answer changes local policy and revokes
        # old authority. Preserve only the approval from this explicit Send,
        # never an old approval during Preview or an option change.
        if was_approved:
            self.consent_checkbox.setChecked(True)
        if self._context is None:
            capture_options = {"include_images": self.images_checkbox.isChecked()}
            if self._output_destination is not None:
                capture_options["output_destination"] = self._output_destination
            self._context, self._owner = self._capture(
                self.source_combo.currentData(),
                **capture_options,
            )
        self._check_image_draft_guard()
        self._assert_current(self._context, self._owner)
        self._context = replace(
            self._context,
            planning_preferences=self._planning_preferences(),
            output_destination=self._output_destination,
        )
        self._context = self._assessment.inject_feedback(self._context)
        self._context = self._refinement.inject_feedback(self._context)
        self._request_objective = self._public_output_objective(objective)
        request = self._context.public_request(self._request_objective)
        self.request_edit.setPlainText(
            json.dumps(request, indent=2, ensure_ascii=False)
        )
        self._image_draft_guard = None
        return request

    def _consent_changed(self, checked):
        if not checked and self._busy:
            self.cancel_request(announce=False)
            self._invalidate_proposal()
            self._draft_authorized = False
            self._set_status(
                "warning",
                "Permission withdrawn. The request was "
                "cancelled locally and late answers will be ignored.",
            )
        self._sync_buttons()

    def preview_sharing(self):
        """Inspect an immutable request locally; no provider is called."""
        previous_message = self._extra_message
        try:
            if self.stage_tabs.currentIndex() == 1:
                self._extra_message = self.discussion_edit.toPlainText()
            request = self._ensure_context(self._answered_objective())
        except (ValueError, RuntimeError) as exc:
            self._set_status("error", f"Cannot prepare this input: {exc}")
            return
        finally:
            self._extra_message = previous_message
        dialog = self._sharing_preview_dialog(request)
        dialog.exec()

    def _sharing_preview_dialog(self, request):
        """Build a local read-only preview; construction does not submit data."""
        dialog = SharingPreviewDialog(self, self._theme_probe.palette())
        panel = QWidget()
        content = QVBoxLayout(panel)
        content.setContentsMargins(14, 12, 14, 12)
        content.setSpacing(10)
        content.setAlignment(Qt.AlignTop)
        content.addWidget(_label(self.sharing_note.text()))
        investigating = self._investigation.checkbox.isChecked() or (
            self._provider() == "openai"
        )
        if investigating:
            scope = _label(
                "Input-sharing preview only\n"
                "This window shows your input request, not the whole investigation. "
                "Starting also shares the tool catalogue and trial history. With "
                "Assess workflow outputs selected, it can share workflow node types "
                "and parameters, measurements and output previews. Start from the "
                "current completed workflow also includes eligible cached results. "
                "Those output images are not shown here; they are separate from "
                "the input previews below. No raw volume or source paths are sent."
            )
            scope.setObjectName("SharedInvestigationScope")
            content.addWidget(scope)
        original = _label(f"Original objective\n{request['objective']}")
        original.setObjectName("SharedOriginalObjective")
        content.addWidget(original)
        label, explanation = self._initiative_copy()
        policy = _label("Assistant initiative\n" + label + "\n" + explanation)
        policy.setObjectName("SharedPlanningPreferences")
        content.addWidget(policy)
        for clarification in request.get("clarifications", ()):
            answer = _label(
                f"Question\n{clarification['question']}\n\n"
                f"Your answer\n{clarification['answer']}"
            )
            answer.setObjectName("SharedClarification")
            content.addWidget(answer)
        for question in request.get("deferred_questions", ()):
            deferred = _label(f"Left to the assistant's best judgment\n{question}")
            deferred.setObjectName("SharedDeferredQuestion")
            content.addWidget(deferred)
        summary = self._context.source_summary
        dimensions = " × ".join(str(size) for size in summary["shape"])
        image_range = summary["range"]
        calibration = "; ".join(
            f"{axis['axis']}: scale {axis['scale']} {axis['unit']}, "
            f"origin {axis['origin']}"
            for axis in summary["calibration"]
        )
        content.addWidget(
            _label(
                f"Input: {dimensions} · {summary['axes']} · {summary['dtype']}\n"
                f"Intensity range: {image_range['minimum']} "
                f"to {image_range['maximum']}\n"
                f"Declared calibration: {calibration}"
            )
        )
        previews = getattr(self._context, "image_previews", ())
        if not previews:
            content.addWidget(
                _label(
                    "No input image pixels are included in this preview. "
                    "Output sharing follows its separate choice in Goal."
                    if investigating
                    else "No image pixels are included in this request."
                )
            )
        else:
            content.addWidget(
                _label(
                    f"{len(previews)} approved image previews\n"
                    "Representative display slices, not the full stack. Their selected "
                    "positions and display transformations are listed beneath each "
                    "image. Scientific pixels remain unchanged."
                )
            )
        for preview in previews:
            title = _label(preview.label)
            title.setStyleSheet("font-weight: bold;")
            content.addWidget(title)
            pixmap = QPixmap()
            if pixmap.loadFromData(preview.png_bytes, "PNG"):
                image_label = QLabel()
                image_label.setPixmap(
                    pixmap.scaled(600, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                content.addWidget(image_label)
            content.addWidget(_label(preview.description))
        advanced_checkbox = QCheckBox("Advanced shared-data details")
        advanced_checkbox.setObjectName("SharedRequestAdvancedToggle")
        content.addWidget(advanced_checkbox)
        advanced = QWidget()
        advanced.setObjectName("SharedRequestAdvanced")
        advanced_layout = QVBoxLayout(advanced)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        details = AssistantDetailsEditor(
            self.request_edit.toPlainText(), json_syntax=True
        )
        details.setObjectName("SharedRequestJSON")
        details.setReadOnly(True)
        details.setMaximumHeight(200)
        advanced_layout.addWidget(
            _label(
                (
                    "Input request data only — additional investigation and output "
                    "evidence is not previewed here. "
                    if investigating
                    else "Complete request data "
                )
                + "(trusted contract instructions are also added "
                "by the connector; optional repair includes the AI response and "
                "check feedback)"
            )
        )
        advanced_layout.addWidget(details)
        content.addWidget(advanced)
        advanced_checkbox.toggled.connect(advanced.setVisible)
        advanced_checkbox.toggled.connect(dialog.queue_fit)
        advanced.hide()
        dialog.set_content(panel)
        return dialog

    def prepare(self):
        """One action captures and submits, after explicit provider consent."""
        if self._investigation.checkbox.isChecked() or self._provider() == "openai":
            self._investigation.start()
            return
        if self._busy:
            return
        if self._provider() == "codex" and (
            not self._codex_authenticated or self._connection_busy
        ):
            self._set_status(
                "warning",
                "Wait for the Codex connection check, or sign in first. "
                "No planning request has been sent.",
            )
            return
        if self._provider() != "example" and not self.consent_checkbox.isChecked():
            self._set_status(
                "warning",
                "Please approve sending this request to the selected "
                "assistant. You can preview exactly what is shared first.",
            )
            return
        self.cancel_request(announce=False)
        self._invalidate_proposal()
        self._submitted_message = ""
        self._sending_action = "message" if self._extra_message else "goal"
        try:
            objective = self._answered_objective()
            self._ensure_context(objective, approved_request=True)
            self._submitted_message = self._extra_message
            self._pending_clarifications = (
                list(objective.clarifications)
                if isinstance(objective, ProposalDialogue)
                else []
            )
            self._append_request_chat(objective)
            self.trace_edit.clear()
            self._progress_messages.clear()
            self.progress_label.clear()
            if not self._question_fields:
                self._clear_review()
            self.json_edit.blockSignals(True)
            self.json_edit.clear()
            self.json_edit.blockSignals(False)
            if self._provider() == "example":
                self._accept_proposal(reference_proposal(self._context))
            else:
                self._send()
        except (ValueError, RuntimeError) as exc:
            self._context = self._owner = None
            self._set_status("error", f"Cannot plan from this input: {exc}")
        self._sync_buttons()

    def _answered_objective(self):
        objective = self.goal_edit.toPlainText()
        self._outgoing_message_pair = None
        current = [
            (question, field.text())
            for question, field in self._question_fields
            if field.text().strip()
        ]
        # Preserve exact content while the latest answer wins for each question.
        pairs = {}
        for question, answer in [*self._clarifications, *current]:
            pairs[question.strip()] = (question, answer)
        if self._extra_message:
            number = 1 + sum(q.startswith("Additional instructions ") for q in pairs)
            self._outgoing_message_pair = (
                f"Additional instructions {number}",
                self._extra_message,
            )
            pairs[f"Additional instructions {number}"] = self._outgoing_message_pair
        deferred = tuple(
            question
            for question, field in self._question_fields
            if self._uses_best_judgment() and not field.text().strip()
        )
        return (
            ProposalDialogue(objective, tuple(pairs.values()), deferred)
            if pairs or deferred
            else objective.strip()
        )

    def _answers_ready(self):
        return self._uses_best_judgment() or all(
            field.text().strip() for _, field in self._question_fields
        )

    def _answers_changed(self, *_):
        self.consent_checkbox.setChecked(False)
        self._sync_buttons()

    def revise_plan(self):
        if self._busy or not self._question_fields:
            return
        if self._provider() == "codex" and (
            not self._codex_authenticated or self._connection_busy
        ):
            self._set_status("warning", "Confirm the Codex connection before revising.")
            return
        if not self._answers_ready():
            return
        if not self.consent_checkbox.isChecked():
            self._set_status(
                "warning", "Approve sharing your answers, then revise the plan."
            )
            return
        self._submitted_message = ""
        self._sending_action = "answers"
        try:
            objective = self._answered_objective()
            self._ensure_context(objective, approved_request=True)
            self._pending_clarifications = (
                list(objective.clarifications)
                if isinstance(objective, ProposalDialogue)
                else []
            )
            self._append_request_chat(objective)
            self._send()
        except (ValueError, RuntimeError) as exc:
            self._set_status("error", f"Cannot revise this plan: {exc}")

    def _edited_proposal(self):
        if not self._latest_selected() or not self._draft_authorized:
            self._invalidate_proposal()
            return
        self._live_proposal_text = self.json_edit.toPlainText()
        self._clear_error_details()
        if self._busy:
            self.cancel_request(announce=False)
        self._invalidate_proposal()
        if self._context is not None and not self._busy:
            self._set_status("info", "Checking the edited plan automatically…")
            self._validation_timer.start()

    def _accept_proposal(self, proposal, *, preserve_answers_on_failure=False):
        # A fresh response belongs to the current request, not the history page
        # the user happened to be reading while it completed.
        self.draft_combo.blockSignals(True)
        self.draft_combo.setCurrentIndex(self._current_draft_index)
        self.draft_combo.blockSignals(False)
        self.json_edit.blockSignals(True)
        self.json_edit.setPlainText(json.dumps(proposal, indent=2, ensure_ascii=False))
        self.json_edit.blockSignals(False)
        self.validate(preserve_answers_on_failure=preserve_answers_on_failure)

    def validate(self, *, preserve_answers_on_failure=False):
        """Automatic local recheck, including live ownership; no network access."""
        self._invalidate_proposal()
        try:
            if self._current_draft_index >= 0 and not self._latest_selected():
                raise ProposalError("Historical drafts are read-only.")
            if self._context is None:
                raise ProposalError("Choose an input and request a plan first.")
            text = self.json_edit.toPlainText()
            if len(text.encode("utf-8")) > MAX_PROPOSAL_BYTES:
                raise ProposalError("The plan is too large.")
            self._assert_current(self._context, self._owner)
            prepared = validate_proposal(text, self._context)
            self._prepared = prepared
            self._render_plan(prepared)
            self._record_draft(prepared)
            self._set_status(
                "warning" if prepared.questions else "success",
                (
                    "Answer what you can in Discussion, or leave draft choices "
                    "to the assistant and send for a revision. "
                    "Nothing can be added yet."
                    if self._uses_best_judgment()
                    else "Answer the questions in Discussion so the AI can "
                    "revise the plan. "
                    "Nothing can be added yet."
                )
                if prepared.questions
                else "Plan checked and ready to add. Adding is one undoable edit; no "
                "calculation or scientific assessment has been performed.",
            )
        except (ValueError, RuntimeError) as exc:
            if not preserve_answers_on_failure or not self._question_fields:
                self._clear_review()
            self._set_status(
                "error",
                f"The plan cannot be added: {exc}. You do not need to "
                "edit JSON; ask the assistant for a new plan.",
            )
        self._sync_buttons()

    def _render_steps(self, steps):
        for index, (step_title, reason) in enumerate(steps, 1):
            card = QFrame()
            card.setObjectName("WorkflowPlanStep")
            card.setAccessibleName(f"Proposed step {index}: {step_title}")
            row = QHBoxLayout(card)
            row.setContentsMargins(0, 6, 0, 6)
            badge = _label(str(index))
            badge.setObjectName("AssistantStepBadge")
            badge.setAlignment(Qt.AlignCenter)
            badge.setFixedSize(24, 24)
            row.addWidget(badge, 0, Qt.AlignTop)
            panel = QWidget()
            panel.setObjectName("WorkflowPlanStepBody")
            panel.setStyleSheet(
                "QWidget#WorkflowPlanStepBody, QLabel { background: transparent; }"
            )
            body = QVBoxLayout(panel)
            body.setContentsMargins(0, 0, 0, 0)
            body.setSpacing(3)
            title = _label(step_title)
            title.setObjectName("InspectorHistoryTitle")
            title.setStyleSheet("font-weight: bold;")
            body.addWidget(title)
            detail = _label(reason)
            detail.setObjectName("InspectorHistoryDetail")
            body.addWidget(detail)
            row.addWidget(panel, 1)
            self.steps_layout.addWidget(card)
            self._step_widgets.append(card)

    def _render_assumptions(self, assumptions, *, draft_key):
        if draft_key != self._review_draft_key:
            self.assumptions_toggle.setChecked(False)
            self.review_checks_toggle.setChecked(False)
            self._review_draft_key = draft_key
        self.review_checks_label.setText(_review_bullets_html(_PLAN_REVIEW_CHECKS))
        self.review_checks_label.setAccessibleName("What to check after running")
        self.review_checks_label.setAccessibleDescription(
            self.review_checks_note.text() + "\n" + "\n".join(_PLAN_REVIEW_CHECKS)
        )
        self.review_checks_content.setVisible(self.review_checks_toggle.isChecked())
        self.review_checks_frame.show()
        self.assumptions_label.setText(
            _review_bullets_html(assumptions) if assumptions else ""
        )
        self.assumptions_label.setAccessibleName("Assumptions to check")
        self.assumptions_label.setAccessibleDescription("\n".join(assumptions))
        self.assumptions_toggle.setChecked(
            bool(assumptions) and self.assumptions_toggle.isChecked()
        )
        self.assumptions_content.setVisible(self.assumptions_toggle.isChecked())
        self.assumptions_frame.setVisible(bool(assumptions))

    def _render_plan(self, prepared):
        self._clear_review(reset_disclosures=False)
        self.summary_label.setText(prepared.summary)
        staged = prepared.candidate.to_pipeline()
        reasons = prepared.node_reasons
        steps = tuple(
            (staged.nodes[key].title, reasons[key]) for key in prepared.added_node_ids
        )
        self._render_steps(steps)
        self._render_assumptions(
            prepared.assumptions,
            # The first addItem selects index 0 with signals blocked, so its
            # initial render already belongs to 0 rather than the empty -1.
            draft_key=(
                max(0, self.draft_combo.currentIndex()),
                self.json_edit.toPlainText(),
            ),
        )
        text = [prepared.summary]
        text.extend(
            f"{index}. {title}: {reason}"
            for index, (title, reason) in enumerate(steps, 1)
        )
        if prepared.assumptions:
            assumptions = "Assumptions to review\n" + "\n".join(
                f"• {item}" for item in prepared.assumptions
            )
            text.append(assumptions)
        for index, question in enumerate(prepared.questions, 1):
            card = QFrame()
            card.setObjectName("AssistantQuestion")
            question_body = QVBoxLayout(card)
            question_body.setContentsMargins(12, 12, 12, 12)
            question_body.setSpacing(10)
            number = _label(f"Question {index}")
            number.setObjectName("AssistantQuestionNumber")
            question_body.addWidget(number)
            question_label = _label(question)
            question_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            question_body.addWidget(question_label)
            field = QLineEdit()
            field.setObjectName("AssistantAnswerInput")
            field.setMinimumHeight(max(38, field.fontMetrics().height() + 20))
            field.setMaxLength(2000)
            field.setPlaceholderText("Your answer…")
            field.textChanged.connect(self._answers_changed)
            question_body.addWidget(field)
            self.questions_layout.addRow(card)
            self._question_fields.append((question, field))
        self.questions_group.setVisible(bool(prepared.questions))
        self.revise_button.setVisible(bool(prepared.questions))
        text.extend(prepared.questions)
        self.preview_edit.setPlainText("\n\n".join(text))
        self.review_scroll.verticalScrollBar().setValue(0)

    def apply_proposal(self):
        if (
            self._busy
            or self._prepared is None
            or not self._prepared.can_apply
            or not (self._draft_authorized and self._latest_selected())
        ):
            return
        try:
            self._assert_current(self._context, self._owner)
            self._investigation.assert_selected_draft(self._prepared)
            ids = self._apply(self._prepared, self._owner)
        except (ValueError, RuntimeError) as exc:
            self._invalidate_proposal()
            self._set_status("error", f"Nothing was added: {exc}")
        else:
            self._context = self._owner = None
            self._invalidate_proposal()
            self.consent_checkbox.setChecked(False)
            self._added_ids = tuple(ids)
            self.json_edit.setReadOnly(True)
            self._append_chat(
                "VIPP",
                f"Added {len(ids)} nodes as one undoable edit. "
                "Nothing was calculated. You can now choose Run workflow.",
            )
            self.draft_note.setText("Added to workflow · Run is a separate action.")
            self._set_status(
                "success",
                f"Added {len(ids)} nodes as one undoable edit. Nothing "
                "was calculated. Review the workflow, then calculate when ready.",
            )
        self._sync_buttons()

    def _kind(self):
        return "ollama" if self.kind_combo.currentIndex() == 0 else "openai-compatible"

    def _refresh_models(self):
        from napari_vipp.core.workflow_proposal_model import discover_models

        if self._provider() != "local" or self._busy:
            return
        address, kind = self.address_edit.text().strip(), self._kind()
        self._start("models", lambda: discover_models(address, kind))

    def _signin_codex(self):
        if self._provider() == "codex":
            self._begin_codex_signin()

    def _begin_codex_signin(self):
        from napari_vipp.core.workflow_proposal_codex import start_codex_login

        if self._busy or self._connection_busy or self._closed:
            return
        cancelled = threading.Event()
        holder: dict[str, object] = {}
        self._login_holder = holder

        def login():
            if cancelled.is_set():
                raise RuntimeError("Sign-in cancelled before it started.")
            process = start_codex_login(cancelled=cancelled.is_set)
            holder["process"] = process
            if cancelled.is_set():
                WorkflowAssistantDialog._stop_owned_login(process)
                raise RuntimeError("Sign-in cancelled.")
            return process

        self._start_connection("signin", login, cancelled)

    def _refresh_codex_connection(self, *_):
        if self._provider() == "codex":
            self._check_codex_connection()

    def _check_codex_connection(self, *_):
        from napari_vipp.core.workflow_proposal_codex import discover_codex_runtime

        if not self._busy and not self._connection_busy and not self._closed:
            self._start_connection("status", discover_codex_runtime)

    def _repair_codex_connection(self):
        """Explicit settings action only; never inferred from opening or sending."""
        from napari_vipp.core.workflow_proposal_codex import repair_codex_profile

        if (
            self._busy
            or self._connection_busy
            or self._closed
            or not self._codex_profile_repairable
        ):
            return
        cancelled = threading.Event()
        self._start_connection(
            "repair",
            lambda: repair_codex_profile(cancelled=cancelled.is_set),
            cancelled,
        )

    def _set_connection_status(self, tone, text):
        self.connection_label.setText(text)
        self.connection_label.setProperty("tone", tone)
        colors = theme_colors(self.palette())
        self.connection_label.setStyleSheet(
            f"color: {getattr(colors, tone, colors.info).foreground.name()};"
        )

    def _start_connection(self, kind, call, cancelled=None):
        self._connection_generation += 1
        self._connection_kind = kind
        self._connection_busy = True
        self._codex_connection_issue = ""
        self._codex_profile_repairable = False
        if kind in {"signin", "repair"}:
            self._codex_authenticated = False
        self._connection_cancelled = cancelled or threading.Event()
        self._set_connection_status(
            "info",
            "Repairing VIPP's managed connection setup. Your sign-in is kept; "
            "no planning request is sent."
            if kind == "repair"
            else "Opening the official sign-in flow…"
            if kind == "signin"
            else "Checking Codex connection… Nothing is sent to a model.",
        )
        self._sync_buttons()
        QThreadPool.globalInstance().start(
            _Call(self._connection_generation, call, self._connection_signals)
        )

    @staticmethod
    def _stop_owned_login(process):
        """Only a Popen returned by our login worker, never a PID or shared agent."""
        with _LOGIN_CLEANUP_LOCK:
            if getattr(process, "_vipp_cleanup_started", False):
                return
            process._vipp_cleanup_started = True

        def cleanup():
            try:
                if process.poll() is None:
                    try:
                        process.terminate()
                    except OSError:
                        pass  # It may have exited between poll and terminate.
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                    except OSError:
                        pass
                    process.wait(timeout=0.5)
            except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
                pass

        # A stubborn login process must never block the Qt event loop on Cancel.
        threading.Thread(
            target=cleanup, name="vipp-login-cleanup", daemon=False
        ).start()

    def _cancel_connection(self, *, announce=True):
        self._connection_cancelled.set()
        self._connection_generation += 1
        self._signin_timer.stop()
        process = self._signin_process or self._login_holder.get("process")
        self._signin_process = None
        if process is not None:
            self._stop_owned_login(process)
        self._login_holder = {}
        was_busy = self._connection_busy
        was_repair = self._connection_kind == "repair"
        self._connection_busy = False
        self._connection_kind = ""
        if was_busy:
            self._codex_authenticated = False
            if was_repair:
                self._codex_connection_issue = "profile"
                self._codex_profile_repairable = False
            self._set_connection_status(
                "warning",
                "Connection action cancelled. Check connection "
                "to check again; existing sign-in credentials were not removed.",
            )
        if announce and hasattr(self, "apply_button"):
            self._sync_buttons()

    @Slot(int, object)
    def _connection_ready(self, generation, result):
        if self._closed or generation != self._connection_generation:
            if hasattr(result, "poll") and hasattr(result, "terminate"):
                self._stop_owned_login(result)
            return
        if self._connection_kind == "signin":
            self._signin_process = result
            self._signin_elapsed.start()
            self._signin_timer.start()
            self._set_connection_status(
                "info",
                "Complete sign-in in your browser. VIPP will confirm when "
                "it finishes; no image or objective is being sent.",
            )
            self._poll_signin()
        else:
            self._connection_busy = False
            self._codex_authenticated = bool(result.supported and result.authenticated)
            self._codex_connection_issue = (
                ""
                if self._codex_authenticated
                else _codex_profile_issue(str(result.detail))
            )
            if self._connection_kind == "repair" and not result.supported:
                self._codex_connection_issue = "profile"
            self._codex_profile_repairable = bool(
                self._codex_connection_issue == "profile"
                and getattr(result, "repairable", False)
            )
            self._set_connection_status(
                "success"
                if self._codex_authenticated
                else "warning"
                if result.supported
                else "error",
                "Signed in to Codex. Ready to plan when you approve a request."
                if self._codex_authenticated
                else str(result.detail)[:500],
            )
        self._sync_buttons()

    def _poll_signin(self):
        """Poll only our child process; one expensive status check after exit."""
        process = self._signin_process
        if process is None or self._closed:
            return
        if self._signin_elapsed.elapsed() >= self._signin_timeout_ms:
            self._cancel_connection(announce=False)
            self._set_connection_status(
                "warning",
                "Sign-in timed out after five minutes. Try again or "
                "Refresh connection if your browser already confirmed success.",
            )
            self._sync_buttons()
            return
        exit_code = process.poll()
        if exit_code is None:
            return
        self._signin_timer.stop()
        self._signin_process = None
        self._login_holder = {}
        self._connection_busy = False
        if exit_code != 0:
            self._set_connection_status(
                "error",
                "Codex sign-in did not finish successfully. Try signing "
                "in again, or Refresh connection to check an existing sign-in.",
            )
        else:
            self._check_codex_connection()
        self._sync_buttons()

    @Slot(int, object)
    def _connection_failed(self, generation, error):
        if self._closed or generation != self._connection_generation:
            return
        self._connection_busy = False
        self._codex_authenticated = False
        detail = str(error)[:500].rstrip(". ")
        # Presentation classification only: the core guard and profile stay intact.
        self._codex_connection_issue = (
            "profile"
            if self._connection_kind == "repair"
            else _codex_profile_issue(detail)
        )
        self._codex_profile_repairable = False
        self._set_connection_status(
            "error",
            f"Connection setup needs attention: {detail}. "
            "The protected setup must be checked before sign-in can continue. "
            "No planning request was sent."
            if self._codex_connection_issue
            else f"Cannot confirm the Codex connection: {detail}. "
            "Try Check connection or sign in again.",
        )
        self._sync_buttons()

    def _send(self):
        if self._context is None or not self.consent_checkbox.isChecked():
            return
        self._assert_current(self._context, self._owner)
        if self._provider() == "codex":
            if not self._codex_authenticated or self._connection_busy:
                raise ProposalError(
                    "Confirm the Codex connection before sending a plan."
                )
            from napari_vipp.core.workflow_proposal_codex import (
                CodexSettings,
                generate_codex_proposal,
            )

            settings = CodexSettings(
                model=self.codex_model_edit.text().strip(),
                reasoning_effort=self.reasoning_combo.currentText(),
                max_attempts=self.attempts_spin.value(),
                timeout_seconds=self._refinement.timeout_seconds(
                    self.timeout_spin.value(), self.attempts_spin.value()
                ),
            )
            generate = generate_codex_proposal
        else:
            from napari_vipp.core.workflow_proposal_model import (
                ModelSettings,
                generate_proposal,
            )

            if getattr(self._context, "image_previews", ()):
                raise ProposalError(
                    "The local connector cannot receive image previews."
                )
            settings = ModelSettings(
                kind=self._kind(),
                base_url=self.address_edit.text().strip(),
                model=self.model_combo.currentText().strip(),
                max_attempts=self.attempts_spin.value(),
                timeout_seconds=self._refinement.timeout_seconds(
                    self.timeout_spin.value(), self.attempts_spin.value()
                ),
            )
            generate = generate_proposal
        context, objective = self._context, self._request_objective
        self._cancelled = threading.Event()
        cancelled = self._cancelled.is_set
        signals, generation = self._signals, self._generation + 1
        self._start(
            "proposal",
            lambda: generate(
                context,
                objective,
                settings,
                cancelled=cancelled,
                progress=lambda message: signals.progress.emit(
                    generation, str(message)
                ),
            ),
        )

    def _start(self, kind, call):
        if hasattr(self, "_refinement") and not self._refinement.before_request(kind):
            return
        if kind == "proposal":
            self._investigation._selected_draft = None
            self._refinement._pending_investigative_promotion = None
        if not self._recovering:
            self._recovery_kind = None
        self._generation += 1
        self._busy = True
        self._job_kind = kind
        if kind in {"proposal", "assessment"}:
            self._start_request_clock()
        else:
            self._stop_request_clock()
        if not kind.startswith("assessment"):
            self._invalidate_proposal()
            self._draft_authorized = False
        # Keep the submitting stage visible while loading. Only a checked reply
        # decides whether the next useful stage is Discussion or Plans.
        self._clear_error_details()
        self.trace_edit.clear()
        self._set_status(
            "info",
            "Preparing a local preview of the completed results… Nothing is sent."
            if kind == "assessment_preview"
            else "Assessing actual results against your goal… You can cancel."
            if kind == "assessment"
            else "Looking for local models…"
            if kind == "models"
            else "Requesting a plan… Any returned plan is checked automatically. "
            + (
                "One automatic repair is allowed before it is shown."
                if self.attempts_spin.value() > 1
                else "Automatic repair is turned off."
            ),
        )
        self._sync_buttons()
        QThreadPool.globalInstance().start(_Call(self._generation, call, self._signals))

    def _start_request_clock(self):
        """Local duration only; no remote progress or scientific authority."""
        self._request_clock_generation = self._generation
        self._request_elapsed.start()
        self._request_elapsed_timer.start()
        self._update_request_elapsed()

    def _request_elapsed_seconds(self):
        return (
            max(0, self._request_elapsed.elapsed() // 1000)
            if self._request_elapsed.isValid()
            else 0
        )

    @Slot()
    def _update_request_elapsed(self):
        if (
            self._closed
            or not self._busy
            or self._job_kind not in {"proposal", "assessment", "investigation"}
            or not self._request_elapsed.isValid()
            or self._request_clock_generation != self._generation
        ):
            self._stop_request_clock()
            return
        minutes, seconds = divmod(self._request_elapsed_seconds(), 60)
        self.request_elapsed_label.setText(f"Elapsed {minutes}:{seconds:02d}")
        self.request_elapsed_label.show()

    def _stop_request_clock(self):
        self._request_elapsed_timer.stop()
        self._request_elapsed.invalidate()
        self._request_clock_generation = None
        self.request_elapsed_label.hide()

    @Slot(int, object)
    def _ready(self, generation, result):
        if self._closed or generation != self._generation:
            return
        self._stop_request_clock()
        self._busy = False
        self._recovering = False
        if self._job_kind.startswith("assessment"):
            self._assessment.ready(result)
        elif self._job_kind == "models":
            self.model_combo.clear()
            self.model_combo.addItems(list(result))
            self._set_status(
                "info", f"Found {len(result)} models. Choose one to plan with."
            )
        else:
            self.trace_edit.setPlainText(self._attempt_trace(result))
            self._summarize_attempts(result)
            pending = self._pending_clarifications
            self._accept_proposal(result.proposal, preserve_answers_on_failure=True)
            if self._prepared is not None and pending is not None:
                self._clarifications = pending
                if self.discussion_edit.toPlainText() == self._submitted_message:
                    self.discussion_edit.clear()
                self._submitted_message = ""
            self._pending_clarifications = None
        self._sync_buttons()
        self._refinement.job_ready(self._job_kind)

    @Slot(int, object)
    def _failed(self, generation, error):
        if self._closed or generation != self._generation:
            return
        self._stop_request_clock()
        self._busy = False
        self._refinement.stop(
            "The assistant request failed. Review the details before continuing."
        )
        if self._job_kind.startswith("assessment"):
            self._assessment.failed(error)
            self._sync_buttons()
            return
        self._retry_action = (
            self._recovery_submit_action
            if self._sending_action == "recovery"
            else self._sending_action
        )
        self._recovering = False
        self._pending_clarifications = None
        self._invalidate_proposal()
        self._error_diagnostic = self._safe_diagnostic(error)
        self.copy_error_button.setEnabled(self._error_diagnostic is not None)
        self.trace_edit.setPlainText(self._attempt_trace(error))
        self._summarize_attempts(error)
        if self._job_kind == "models":
            self._recovery_kind = None
            explanation = "Could not list local models. Check the server address "
            explanation += "and whether the server is running."
        else:
            explanation = self._failure_explanation(error, self.attempts_spin.value())
            diagnostic = self._error_diagnostic
            self._recovery_kind = (
                "settings"
                if diagnostic is not None
                and (
                    diagnostic.category == "unauthorized"
                    or diagnostic.reason in {"unsupported_model", "unsupported_image"}
                )
                else "details"
                if diagnostic is not None
                and diagnostic.category
                in {"capability", "configuration", "protocol", "bad_request"}
                else "retry"
            )
            if self._recovery_kind == "retry":
                explanation += (
                    " Use the retry button to send a fresh request with your goal "
                    "and answers. Any returned plan is checked automatically."
                )
        self._set_status("error", "Nothing was changed. " + explanation)
        self._append_chat("VIPP", "Nothing was changed. " + explanation)
        self._sync_buttons()

    @staticmethod
    def _safe_diagnostic(result):
        from napari_vipp.core.workflow_proposal_model import SafeModelDiagnostic

        diagnostic = getattr(result, "diagnostic", None)
        return diagnostic if type(diagnostic) is SafeModelDiagnostic else None

    def _clear_error_details(self):
        self._error_diagnostic = None
        self.copy_error_button.setEnabled(False)

    def _copy_error_details(self):
        if self._error_diagnostic is not None:
            QApplication.clipboard().setText(
                json.dumps(
                    self._error_diagnostic.to_dict(), indent=2, ensure_ascii=False
                )
            )

    @staticmethod
    def _request_failure(result):
        attempts = getattr(result, "attempts", ())
        return not attempts or getattr(attempts[-1], "phase", "proposal") == "request"

    @staticmethod
    def _failure_explanation(result, selected_limit):
        if WorkflowAssistantDialog._request_failure(result):
            diagnostic = WorkflowAssistantDialog._safe_diagnostic(result)
            cause = "The assistant could not complete the request. Try again; "
            cause += "if it continues, the connection needs checking."
            if diagnostic is not None:
                if (
                    diagnostic.category == "bad_request"
                    and diagnostic.reason == "invalid_schema"
                ):
                    cause = (
                        "The service rejected VIPP's request format. This needs an "
                        "integration fix; changing your objective will not fix it."
                    )
                elif (
                    diagnostic.category == "bad_request"
                    and diagnostic.reason == "unsupported_model"
                ):
                    cause = (
                        "The selected model is unavailable for this connection. "
                        "Choose a supported model in Advanced or use the default."
                    )
                elif (
                    diagnostic.category == "bad_request"
                    and diagnostic.reason == "unsupported_image"
                ):
                    cause = (
                        "The selected model could not accept image previews. Choose "
                        "an image-capable model, or explicitly turn off previews."
                    )
                else:
                    cause = {
                        "bad_request": "The service could not accept the request. "
                        "The connection settings or VIPP integration need checking.",
                        "unauthorized": "The request was not authorized. Use Refresh "
                        "connection or Sign in again, then retry.",
                        "quota": "The service reported a usage or rate limit. Check "
                        "your account limits or wait before retrying.",
                        "network": "The connection to the assistant failed. Check "
                        "your connection and try again.",
                        "timeout": "The request timed out. Try again, or adjust the "
                        "time limit in Advanced.",
                        "capability": "VIPP stopped an attempt to use a capability "
                        "this assistant does not allow. Do not bypass the safeguards.",
                        "provider": "The service did not complete the request. "
                        "Try again later.",
                        "protocol": "VIPP and the assistant could not exchange a "
                        "usable response. The integration needs checking.",
                        "configuration": "This connection configuration is not "
                        "supported by VIPP's restricted assistant.",
                    }[diagnostic.category]
            explanation = cause + (
                " No usable plan was received; your objective has not been assessed."
            )
            if diagnostic is not None:
                explanation += " Safe error details are in Advanced."
            return explanation
        attempts = getattr(result, "attempts", ())
        proposals = [
            a for a in attempts if getattr(a, "phase", "proposal") == "proposal"
        ]
        if len(proposals) > 1:
            return (
                "Neither the initial plan nor its automatic repair passed VIPP's "
                "workflow checks. The selected limit was two attempts (one repair). "
                "Try again; technical check details are in Advanced. You do not "
                "need to edit JSON."
            )
        return (
            "The returned plan did not pass VIPP's workflow checks. "
            + (
                "The selected limit was one attempt, so automatic repair was not "
                "attempted. "
                if selected_limit == 1
                else "Only one proposal was checked; no repaired proposal "
                "was returned. "
            )
            + "Try again; technical check details are in Advanced. You do not "
            "need to edit JSON."
        )

    def _summarize_attempts(self, result):
        for attempt in getattr(result, "attempts", ()):
            message = (
                f"Attempt {attempt.number}: request failed before plan checks."
                if getattr(attempt, "phase", "proposal") == "request"
                else f"Attempt {attempt.number}: plan checks passed."
                if attempt.valid
                else f"Attempt {attempt.number}: plan did not pass checks."
            )
            if message not in self._progress_messages:
                self._progress_messages.append(message)
        self.progress_label.setText("\n".join(self._progress_messages[-8:]))

    @staticmethod
    def _attempt_trace(result):
        """Plan details stay Advanced; request errors use a closed safe capsule."""
        lines = []
        if hasattr(result, "model"):
            lines.extend(
                [
                    f"Model: {result.model} ({result.provider})",
                    f"Request SHA-256: {result.request_sha256}",
                ]
            )
        for attempt in getattr(result, "attempts", ()):
            request_phase = getattr(attempt, "phase", "proposal") == "request"
            lines.extend(
                [
                    f"Attempt {attempt.number}: {attempt.elapsed_seconds:.2f} seconds",
                    f"Phase: {'request' if request_phase else 'proposal'}",
                    "Plan checks: not run"
                    if request_phase
                    else f"Plan checks passed: {attempt.valid}",
                    "No proposal was received for this attempt."
                    if request_phase
                    else f"Check feedback: {attempt.validation_error or 'None'}",
                    "Input/output tokens: "
                    f"{attempt.input_tokens}/{attempt.output_tokens}",
                    "",
                ]
            )
            if not request_phase:
                lines.extend(
                    ["Response:", attempt.response_text[:MAX_PROPOSAL_BYTES], ""]
                )
        diagnostic = WorkflowAssistantDialog._safe_diagnostic(result)
        if diagnostic is not None:
            lines.extend(
                ["Safe error details:", json.dumps(diagnostic.to_dict(), indent=2)]
            )
        return "\n".join(lines) or "No model response was recorded."

    @Slot(int, str)
    def _progress(self, generation, message):
        if self._closed or not self._busy or generation != self._generation:
            return
        if self._job_kind == "assessment":
            self._set_status("info", "Assessing actual results against your goal…")
            return
        lower = message.lower()
        concise = (
            "The connection is recovering for the same request, within the "
            "existing time limit. You can cancel. Recovery does not start "
            "another plan attempt."
            if message == "Codex is recovering the connection for this same request…"
            else "The AI is repairing its plan using automatic check feedback."
            if "repair" in lower or "feedback" in lower
            else "Checking the AI's plan…"
            if "validat" in lower or "check" in lower
            else "Requesting a plan from the assistant…"
        )
        if concise not in self._progress_messages:
            self._progress_messages.append(concise)
        self.progress_label.setText("\n".join(self._progress_messages[-8:]))
        self._set_status("info", concise)

    def cancel_request(self, *, announce=True):
        if hasattr(self, "_investigation") and self._investigation.active:
            self._investigation.cancel(
                "Investigation cancelled or its input/settings changed."
            )
        if announce:
            if hasattr(self, "_refinement"):
                self._refinement.stop("Refinement cancelled.", cancel_work=True)
            self._cancel_connection()
            self._recovering = False
        self._cancelled.set()
        self._pending_clarifications = None
        self._submitted_message = ""
        self._generation += 1
        was_busy = self._busy
        self._stop_request_clock()
        self._busy = False
        if announce and was_busy:
            self._set_status(
                "warning",
                "Cancelled. Late answers are ignored; a provider may "
                "finish a request it has already received.",
            )
        if hasattr(self, "apply_button"):
            self._sync_buttons()

    def _copy_tab(self):
        editor = self.tabs.currentWidget()
        if isinstance(editor, QPlainTextEdit):
            QApplication.clipboard().setText(editor.toPlainText())

    def _deactivate(self):
        """All dismissal paths revoke authority and ignore late worker replies."""
        if self._closed:
            return
        self._closed = True
        if hasattr(self, "_live_agent"):
            self._live_agent.close()
        self._refinement.stop("The assistant window was closed.", cancel_work=True)
        self._recovery_kind = None
        self._recovering = False
        if self._settings_dialog is not None:
            self._settings_dialog.reject()
        self._cancel_connection(announce=False)
        self.cancel_request(announce=False)
        self._assessment.invalidate("This review window was closed.")
        self._invalidate_proposal()
        self._draft_authorized = False
        self._context = self._owner = None
        self._image_draft_guard = None
        self._request_objective = ""
        self._clarifications.clear()
        self.consent_checkbox.setChecked(False)
        self._clear_review()
        self.request_edit.clear()
        self.trace_edit.clear()
        if self.draft_combo.currentIndex() >= 0:
            self._draft_selected(self.draft_combo.currentIndex())

    def done(self, result):
        self._deactivate()
        super().done(result)

    def reject(self):
        self._deactivate()
        super().reject()

    def closeEvent(self, event):  # noqa: N802
        self._deactivate()
        super().closeEvent(event)

    def showEvent(self, event):  # noqa: N802
        self._closed = False
        super().showEvent(event)
