"""Human-facing finite allowances, separate from model/provider configuration."""

from qtpy.QtWidgets import QFormLayout, QFrame, QLabel, QSpinBox, QVBoxLayout

from napari_vipp.core.workflow_investigation import InvestigationLimits
from napari_vipp.ui.workflow_assistant_style import AssistantDisclosureButton


class AssistantEffortControls(QFrame):
    """A compact, explicit ceiling; editing it never grants a running session."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("AssistantInvestigationOptions")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 8, 14, 8)
        self.toggle = AssistantDisclosureButton("Working allowance")
        layout.addWidget(self.toggle)
        self.panel = QFrame()
        form = QFormLayout(self.panel)
        form.setContentsMargins(0, 4, 0, 4)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.minutes = self._spin(5, 30, 30, "Maximum active investigation time")
        self.minutes.setSuffix(" min")
        form.addRow("Working time", self.minutes)
        self.actions = self._spin(1, 64, 64, "Maximum investigation actions")
        form.addRow("Tool actions", self.actions)
        self.calls = self._spin(1, 80, 80, "Maximum model requests or MCP contexts")
        form.addRow("Model requests / external contexts", self.calls)
        self.image_mib = self._spin(1, 20, 10, "Maximum shared preview data")
        self.image_mib.setSuffix(" MiB")
        form.addRow("Shared previews", self.image_mib)
        note = QLabel(
            "These are upper limits, not a target. Work can finish earlier. "
            "At an allowance limit, pause and explicitly grant more capacity "
            "within the safety ceilings. Paused review time is not working time."
        )
        note.setWordWrap(True)
        note.setObjectName("AssistantMuted")
        form.addRow(note)
        layout.addWidget(self.panel)
        self.panel.hide()
        self.toggle.toggled.connect(self.panel.setVisible)

    @staticmethod
    def _spin(low, high, value, name):
        control = QSpinBox()
        control.setRange(low, high)
        control.setValue(value)
        control.setAccessibleName(name)
        return control

    @property
    def controls(self):
        return self.minutes, self.actions, self.calls, self.image_mib

    def binding(self):
        return tuple(control.value() for control in self.controls)

    def limits(self, attempts):
        return InvestigationLimits(
            max_attempts=attempts,
            max_actions=self.actions.value(),
            max_model_calls=self.calls.value(),
            max_seconds=self.minutes.value() * 60,
            max_image_bytes=self.image_mib.value() * 1024 * 1024,
        )
