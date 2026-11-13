"""Read-only, on-demand view of a node's configuration, never its execution."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from qtpy.QtCore import Qt
from qtpy.QtGui import (
    QFont,
    QTextCharFormat,
    QTextCursor,
    QTextLength,
    QTextOption,
    QTextTableFormat,
)
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
)

from napari_vipp.core.pipeline import OperationSpec, optional_persisted_parameter_spec
from napari_vipp.ui.design import (
    SPACE_LG,
    SPACE_SM,
    SPACE_XS,
    apply_text_role,
    install_window_policy,
)
from napari_vipp.ui.dialog_buttons import add_dialog_buttons

_MAX_JSON_LENGTH = 1_000_000
_MAX_INLINE_VALUE_LENGTH = 1000
_PSF_METADATA_FIELDS = {
    "wavelength_nm",
    "numerical_aperture",
    "refractive_index",
    "pixel_size_xy_um",
    "z_step_um",
}


def _settings_snapshot(
    operation: OperationSpec, parameters: Mapping[str, Any]
) -> dict[str, Any]:
    values = {
        spec.name: spec.default
        for spec in operation.parameters
        if not spec.name.startswith("_")
    }
    values.update(
        (name, value) for name, value in parameters.items() if not name.startswith("_")
    )
    return values


def _validate_json_value(value: Any, ancestors: set[int]) -> None:
    """Do not let the JSON encoder silently change non-JSON values."""
    if value is None or type(value) in (bool, str, int, float):
        return
    if type(value) not in (dict, list):
        raise TypeError("Unsupported configuration value.")
    identity = id(value)
    if identity in ancestors:
        raise ValueError("Circular configuration value.")
    ancestors.add(identity)
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Configuration object keys must be strings.")
        children = value.values()
    else:
        children = value
    for child in children:
        _validate_json_value(child, ancestors)
    ancestors.remove(identity)


def _json_preview(operation_id: str, values: Mapping[str, Any]) -> tuple[str, bool]:
    """Keep exact JSON values or explicitly decline the preview, never coerce them."""
    document = {"operation_id": operation_id, "parameters": values}
    try:
        _validate_json_value(document, set())
        # ASCII escapes also prevent Qt's plain-text document from converting
        # Unicode paragraph separators or non-breaking spaces in JSON strings.
        chunks = []
        length = 0
        for chunk in json.JSONEncoder(indent=2, allow_nan=False).iterencode(document):
            length += len(chunk)
            if length > _MAX_JSON_LENGTH:
                return "This configuration is too large to display as JSON here.", False
            chunks.append(chunk)
        return "".join(chunks), True
    except (TypeError, ValueError, RecursionError):
        return (
            "JSON is unavailable because a configuration value cannot be "
            "represented faithfully as JSON.",
            False,
        )


def _settings_text(value: Any) -> str:
    if value is None:
        return "Not set"
    if isinstance(value, bool):
        return "On" if value else "Off"
    if isinstance(value, str):
        if len(value) > _MAX_INLINE_VALUE_LENGTH:
            return f"{value[:_MAX_INLINE_VALUE_LENGTH]}… (see JSON if available)"
        return value if value else "Empty"
    if isinstance(value, (int, float)):
        try:
            return str(value)
        except ValueError:
            return "Number too large to display"
    if isinstance(value, (dict, list, tuple)):
        count = len(value)
        return f"{count} {'item' if count == 1 else 'items'} (see JSON)"
    return "Not available as text"


def _settings_rows(
    node_name: str, operation: OperationSpec, values: Mapping[str, Any]
) -> list[tuple[str, str]]:
    specs = {spec.name: spec for spec in operation.parameters}
    rows = [("Operation", operation.title)]
    if node_name and node_name != operation.title:
        rows.insert(0, ("Node name", _settings_text(node_name)))
    if not values:
        rows.append(("Parameters", "No configurable parameters"))
    psf_auto = (
        operation.id == "born_wolf_psf" and values.get("auto_parameters") is True
    )
    for name, value in values.items():
        spec = specs.get(name) or optional_persisted_parameter_spec(operation, name)
        label = spec.label if spec is not None else name.replace("_", " ").capitalize()
        if spec is not None and isinstance(value, str) and spec.choice_labels:
            labels = dict(zip(spec.choices, spec.choice_labels, strict=False))
            value = labels.get(value, value)
        if (
            psf_auto
            and name in _PSF_METADATA_FIELDS
            and type(value) in (int, float)
            and value == 0
        ):
            # Zero requests no manual override. This says nothing about whether
            # metadata is present or what will be resolved for an actual run.
            display = "Not set manually"
        else:
            display = _settings_text(value)
        rows.append((label, display))
    return rows


def _populate_settings(editor: QTextEdit, rows: list[tuple[str, str]]) -> None:
    heading = QTextCharFormat()
    heading.setFontWeight(QFont.Bold)
    cursor = editor.textCursor()
    table_format = QTextTableFormat()
    table_format.setBorder(0)
    table_format.setCellPadding(SPACE_XS)
    table_format.setCellSpacing(0)
    table_format.setWidth(QTextLength(QTextLength.PercentageLength, 100))
    table_format.setColumnWidthConstraints(
        [QTextLength(QTextLength.PercentageLength, width) for width in (45, 55)]
    )
    table = cursor.insertTable(len(rows), 2, table_format)
    for index, (label, value) in enumerate(rows):
        table.cellAt(index, 0).firstCursorPosition().insertText(label, heading)
        table.cellAt(index, 1).firstCursorPosition().insertText(value)
    cursor.movePosition(QTextCursor.Start)
    editor.setTextCursor(cursor)


class NodeDetailsDialog(QDialog):
    """A local snapshot with a readable view and exact, copyable JSON."""

    def __init__(
        self,
        *,
        node_name: str,
        operation: OperationSpec,
        parameters: Mapping[str, Any],
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Node details")
        install_window_policy(self, profile="compact")
        values = _settings_snapshot(operation, parameters)
        self._json_text, self._has_json = _json_preview(operation.id, values)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACE_LG, SPACE_LG, SPACE_LG, SPACE_LG)
        layout.setSpacing(SPACE_SM)
        heading = QLabel("Node details")
        apply_text_role(heading, "title")
        layout.addWidget(heading)
        description = QLabel(
            "A settings snapshot, not a run result. Edit settings and check "
            "warnings in the inspector. Defaults and inactive settings may appear."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        self.tabs = QTabWidget()
        self.tabs.setAccessibleName("Node details views")
        self.settings = QTextEdit()
        self.settings.setReadOnly(True)
        self.settings.setAcceptRichText(False)
        self.settings.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.settings.setAccessibleName("Current node settings")
        _populate_settings(self.settings, _settings_rows(node_name, operation, values))
        self.tabs.addTab(self.settings, "Settings")

        self.json_view = QPlainTextEdit()
        self.json_view.setReadOnly(True)
        self.json_view.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.json_view.setAccessibleName("Node configuration JSON")
        json_scope = (
            "Configured settings and defaults; internal node metadata is omitted."
        )
        self.json_view.setAccessibleDescription(json_scope)
        self.json_view.setPlainText(self._json_text)
        apply_text_role(self.json_view, "code")
        json_index = self.tabs.addTab(self.json_view, "JSON")
        self.tabs.setTabToolTip(json_index, json_scope)
        layout.addWidget(self.tabs, 1)

        footer = QHBoxLayout()
        footer.addStretch()
        self.copy_button = QPushButton("Copy JSON")
        self.copy_button.setEnabled(self._has_json)
        self.copy_button.clicked.connect(self._copy_json)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.reject)
        add_dialog_buttons(
            footer, actions=(self.copy_button,), dismiss=self.close_button
        )
        layout.addLayout(footer)
        self.tabs.currentChanged.connect(self._show_tab_actions)
        self._show_tab_actions(self.tabs.currentIndex())
        self.settings.setFocus(Qt.OtherFocusReason)

    def _show_tab_actions(self, index: int) -> None:
        self.copy_button.setVisible(self.tabs.widget(index) is self.json_view)

    def _copy_json(self) -> None:
        if self._has_json:
            QApplication.clipboard().setText(self._json_text)
