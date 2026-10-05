"""Compact, presentation-only node name editor for the inspector."""

from html import escape

from qtpy.QtCore import QEvent, QSignalBlocker, QSize, Qt, Signal
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.core.node_names import MAX_NODE_NAME_LENGTH
from napari_vipp.ui.node_labels import NodePresentation


class _SettingsSummary(QLabel):
    """Wrap live settings without shrinking the same node's control positions."""

    def __init__(self):
        super().__init__()
        self._height_envelope: dict[int, int] = {}
        self._height_context = None
        self.setTextFormat(Qt.PlainText)
        self.setWordWrap(True)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

    def set_summary(self, text: str):
        if self.text() != text:
            self.setText(text)
        self.setToolTip(f"<qt>{escape(text)}</qt>" if text else "")
        self.setAccessibleDescription(text)
        self.setVisible(bool(text))

    def reset_height_envelope(self):
        """Release presentation-only space when its node or layout changes."""
        self._height_envelope.clear()
        self._height_context = None
        self.updateGeometry()

    def _sync_height_context(self):
        margins = self.contentsMargins()
        context = (
            self.font().toString(),
            self.fontMetrics().height(),
            self.style().objectName(),
            self.styleSheet(),
            self.margin(),
            self.indent(),
            margins.left(),
            margins.top(),
            margins.right(),
            margins.bottom(),
        )
        if context != self._height_context:
            self._height_envelope.clear()
            self._height_context = context

    def heightForWidth(self, width):  # noqa: N802
        self._sync_height_context()
        required = int(super().heightForWidth(width))
        if required < 0:
            return required
        # Changing 1000 to 1 can remove a wrapped line on some system fonts.
        # Keep that readable space while tuning this node, but still let a
        # genuinely longer summary grow. Widths are cached separately because
        # Qt can measure several candidate widths during one layout pass.
        reserved = max(required, self._height_envelope.get(int(width), 0))
        self._height_envelope[int(width)] = reserved
        return reserved

    def minimumSizeHint(self):  # noqa: N802
        hint = super().minimumSizeHint()
        # QLabel's one-line minimum must stay independent of its current
        # width. Promoting a narrow-width reservation into this global minimum
        # can clamp an owner's next resize before it gets the wider geometry.
        # The readable, width-specific reservation belongs in heightForWidth.
        return QSize(0, hint.height())

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if event.size().width() != event.oldSize().width():
            self.reset_height_envelope()

    def changeEvent(self, event):  # noqa: N802
        super().changeEvent(event)
        if hasattr(self, "_height_envelope") and event.type() in {
            QEvent.FontChange,
            QEvent.ApplicationFontChange,
            QEvent.StyleChange,
        }:
            self.reset_height_envelope()


class NodeNameEditor(QWidget):
    """Keep a user-authored name separate from the live operation summary.

    The owner must call :meth:`commit_pending` before changing the selected
    node or workflow session, while the old node still owns its edit. Passive
    presentation updates never commit an in-progress name. ``clear`` and a
    changed node in ``set_node`` deliberately discard any remaining draft so
    it cannot accidentally rename another workflow's node with the same ID.
    """

    name_committed = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._node_id = ""
        self._editing_node_id = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        row = QHBoxLayout()
        row.addWidget(QLabel("Name"))
        self.name_edit = QLineEdit()
        self.name_edit.setMaxLength(MAX_NODE_NAME_LENGTH)
        self.name_edit.setAccessibleName("Node name")
        self._name_help = (
            "Optional name for this node. Leave blank to use the automatic name. "
            "This does not change its settings or a plot's figure title."
        )
        row.addStretch(1)
        self.reset_button = QPushButton("Reset")
        self.reset_button.setToolTip("Reset to automatic name")
        row.addWidget(self.reset_button)
        layout.addLayout(row)
        layout.addWidget(self.name_edit)
        self.operation_label = QLabel()
        self.operation_label.setTextFormat(Qt.PlainText)
        self.operation_label.setWordWrap(True)
        self.operation_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.operation_label.setMinimumWidth(0)
        layout.addWidget(self.operation_label)
        self.summary_label = _SettingsSummary()
        layout.addWidget(self.summary_label)
        self.name_edit.textEdited.connect(self._edited)
        self.name_edit.editingFinished.connect(self._commit)
        self.reset_button.clicked.connect(self._reset)

    def _edited(self, _text):
        self._editing_node_id = self._node_id
        self.reset_button.setEnabled(bool(self.name_edit.text()))

    def commit_pending(self) -> None:
        """Commit a draft before its owning node or workflow changes."""
        self._commit()

    def _commit(self):
        if self._editing_node_id and self.name_edit.isModified():
            node_id = self._editing_node_id
            self._editing_node_id = ""
            self.name_edit.setModified(False)
            self.name_committed.emit(node_id, self.name_edit.text())

    def _reset(self):
        if self._node_id:
            self._editing_node_id = ""
            self.name_edit.setModified(False)
            self.name_committed.emit(self._node_id, "")

    def set_node(self, node_id: str, custom_name: str, presentation: NodePresentation):
        changed_node = self._node_id != node_id
        if changed_node:
            self.summary_label.reset_height_envelope()
        self._node_id = node_id
        # Background result refreshes must not erase a name being typed.
        if changed_node or not (
            self.name_edit.hasFocus() and self.name_edit.isModified()
        ):
            with QSignalBlocker(self.name_edit):
                if changed_node or self.name_edit.text() != custom_name:
                    self.name_edit.setText(custom_name)
                    self.name_edit.setCursorPosition(0)
                self.name_edit.setModified(False)
            self._editing_node_id = ""
        self.name_edit.setPlaceholderText(presentation.automatic_name)
        self.name_edit.setToolTip(
            f"<qt>{escape(presentation.name)}<br><br>{escape(self._name_help)}</qt>"
        )
        self.reset_button.setEnabled(bool(custom_name or self.name_edit.text()))
        self.operation_label.setText(f"Operation: {presentation.operation}")
        self.summary_label.set_summary(presentation.summary)
        self.operation_label.setToolTip(f"Node ID: {node_id}")
        self.show()

    def clear(self):
        self._node_id = self._editing_node_id = ""
        self.summary_label.reset_height_envelope()
        self.name_edit.clear()
        self.name_edit.setModified(False)
        self.hide()
