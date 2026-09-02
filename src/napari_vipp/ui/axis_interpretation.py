"""Shared novice-facing controls for reviewed source-axis declarations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import QEvent, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.core.metadata import AxisDeclaration
from napari_vipp.ui.palette_roles import theme_colors

if TYPE_CHECKING:
    from napari_vipp.core.batch import BatchAxisSuggestion


class AxisInterpretationControl(QWidget):
    """Novice-facing axis choice with a compatible declaration value."""

    textChanged = Signal(str)

    AUTOMATIC = "automatic"
    FILE_METADATA = "file"
    Z_STACK = "z_stack"
    CUSTOM = "custom"
    Z_STACK_DECLARATION = "QYX -> ZYX"

    def __init__(
        self,
        parent=None,
        *,
        allow_automatic: bool = True,
        save_target: str = "batch",
    ) -> None:
        super().__init__(parent)
        self._applying_theme_style = False
        self._updating = False
        self._committed_text = ""
        self._last_text = ""
        self._suggestion_seen = False
        self._suggestion_declined = False
        self._auto_suggestion_active = False
        self._allow_automatic = bool(allow_automatic)
        self._save_target = (
            "workflow" if str(save_target).strip().casefold() == "workflow" else "batch"
        )

        self.mode_combo = QComboBox()
        if self._allow_automatic:
            self.mode_combo.addItem(
                "Automatic (recommended)",
                self.AUTOMATIC,
            )
        self.mode_combo.addItem(
            "Use the file's labels unchanged",
            self.FILE_METADATA,
        )
        self.mode_combo.addItem(
            "Stack planes are depth slices (Z stack)",
            self.Z_STACK,
        )
        self.mode_combo.addItem("Something else (advanced)...", self.CUSTOM)
        self.mode_combo.setAccessibleName("How VIPP should interpret the image stack")

        self.advanced_edit = QLineEdit()
        self.advanced_edit.setPlaceholderText("Advanced: source axes -> intended axes")
        self.advanced_edit.hide()

        self.notice_label = QLabel("")
        self.notice_label.setWordWrap(True)
        self.notice_label.setMinimumWidth(0)
        self.notice_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._apply_theme_style()
        self.notice_label.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.mode_combo)
        layout.addWidget(self.advanced_edit)
        layout.addWidget(self.notice_label)

        guidance = (
            "VIPP normally trusts the file. If the leading stack dimension "
            "really represents depth, choose Z. This changes axis labels only; "
            "it does not transpose pixels."
        )
        self.setToolTip(guidance)
        self.setAccessibleDescription(guidance)
        self.mode_combo.setToolTip(guidance)
        self.advanced_edit.setToolTip(
            "Advanced compatibility option for a reviewed source-to-result "
            "axis declaration."
        )

        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        self.advanced_edit.textChanged.connect(self._advanced_changed)
        self.advanced_edit.editingFinished.connect(self._advanced_edit_finished)

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        if not self._applying_theme_style and event.type() in (
            QEvent.PaletteChange,
            QEvent.StyleChange,
        ):
            self._apply_theme_style()

    def _apply_theme_style(self) -> None:
        if self._applying_theme_style:
            return
        self._applying_theme_style = True
        try:
            parent = self.parentWidget()
            palette = (
                QWidget.palette(parent)
                if parent is not None
                else QWidget.palette(self)
            )
            warning = theme_colors(palette).warning
            self.notice_label.setStyleSheet(
                "QLabel {"
                f" color: {warning.foreground.name()};"
                f" background: {warning.surface.name()};"
                f" border: 1px solid {warning.border.name()};"
                " border-radius: 4px; padding: 5px;"
                " }"
            )
        finally:
            self._applying_theme_style = False

    def text(self) -> str:
        """Return only the last complete declaration accepted for persistence."""
        return self._committed_text

    def setText(self, value: str) -> None:  # noqa: N802 - QLineEdit compatibility
        old_text = self.text()
        raw_value = str(value or "").strip()
        try:
            declaration = AxisDeclaration.from_value(raw_value)
        except ValueError:
            declaration = None
        is_z_stack = (
            declaration is not None
            and declaration.source_axes == "QYX"
            and declaration.effective_axes == "ZYX"
        )

        self._updating = True
        try:
            if not raw_value:
                self._committed_text = ""
                self.mode_combo.setCurrentIndex(
                    self.mode_combo.findData(self.FILE_METADATA)
                )
                self.advanced_edit.clear()
            elif is_z_stack:
                self._committed_text = self.Z_STACK_DECLARATION
                self.mode_combo.setCurrentIndex(self.mode_combo.findData(self.Z_STACK))
                self.advanced_edit.clear()
            elif declaration is not None:
                self._committed_text = declaration.display_text
                self.mode_combo.setCurrentIndex(self.mode_combo.findData(self.CUSTOM))
                self.advanced_edit.setText(declaration.display_text)
            else:
                self.mode_combo.setCurrentIndex(self.mode_combo.findData(self.CUSTOM))
                self.advanced_edit.setText(raw_value)
            self.advanced_edit.setVisible(self.mode_combo.currentData() == self.CUSTOM)
        finally:
            self._updating = False

        self._suggestion_seen = False
        self._suggestion_declined = False
        self._auto_suggestion_active = False
        if is_z_stack:
            self._show_notice(
                "Saved choice: treat the leading stack dimension as depth (Z). "
                "Pixel order is unchanged, and this choice is saved with the "
                f"{self._save_target}."
            )
        elif declaration is not None:
            self._show_notice(
                "Using a saved advanced axis interpretation. Pixel order is unchanged."
            )
        elif raw_value:
            self._show_notice(
                "This saved axis text is incomplete or invalid. It has not been "
                "applied; enter a complete mapping such as QYX -> ZYX."
            )
        else:
            self._hide_notice()
        self._last_text = old_text
        self._emit_if_changed()

    def apply_z_stack_suggestion(self, suggestion: BatchAxisSuggestion) -> bool:
        """Apply only VIPP's exact, guarded QYX-to-ZYX recommendation once."""
        declaration = suggestion.declaration
        if (
            declaration.source_axes != "QYX"
            or declaration.effective_axes != "ZYX"
            or self.mode_combo.currentData() != self.AUTOMATIC
            or self._suggestion_declined
        ):
            return False
        self._suggestion_seen = True
        self._auto_suggestion_active = True
        self._updating = True
        try:
            self._committed_text = self.Z_STACK_DECLARATION
            self.mode_combo.setCurrentIndex(self.mode_combo.findData(self.Z_STACK))
            self.advanced_edit.hide()
        finally:
            self._updating = False
        self._show_notice(
            "Automatic choice: QYX -> ZYX because this workflow requires 3D "
            "spatial processing. Pixel order is unchanged; verify that the "
            f"leading dimension is depth. This is saved with the {self._save_target}."
        )
        self.notice_label.setToolTip(
            "File labels: QYX; VIPP labels for this batch: ZYX. Verify the Z "
            "spacing separately."
        )
        self._emit_if_changed()
        return True

    @property
    def suggestion_declined(self) -> bool:
        return self._suggestion_declined

    def source_binding_changed(self) -> None:
        """Discard only an inference made for the previous file collection."""
        if not self._auto_suggestion_active:
            return
        self._updating = True
        try:
            self._committed_text = ""
            self.mode_combo.setCurrentIndex(self.mode_combo.findData(self.AUTOMATIC))
            self.advanced_edit.hide()
        finally:
            self._updating = False
        self._auto_suggestion_active = False
        self._suggestion_seen = False
        self._suggestion_declined = False
        self._hide_notice()
        self._emit_if_changed()

    def _mode_changed(self, _index: int) -> None:
        if self._updating:
            return
        mode = self.mode_combo.currentData()
        self._auto_suggestion_active = False
        self.advanced_edit.setVisible(mode == self.CUSTOM)
        if mode == self.AUTOMATIC:
            self._committed_text = ""
            self._suggestion_declined = False
            self._show_notice(
                "VIPP will change this only if the workflow proves that it "
                "needs a Z stack."
            )
        elif mode == self.FILE_METADATA:
            self._committed_text = ""
            if self._suggestion_seen:
                self._suggestion_declined = True
                self._show_notice(
                    "Using the file's labels unchanged. VIPP will not "
                    "choose Z stack again for this source."
                )
            else:
                self._hide_notice()
        elif mode == self.Z_STACK:
            self._committed_text = self.Z_STACK_DECLARATION
            self._suggestion_declined = False
            self._show_notice(
                "Treating the leading stack dimension as depth (Z). Pixel order is "
                "unchanged, and this choice will be saved with the "
                f"{self._save_target}."
            )
        else:
            self._show_notice(
                "Enter a complete source-to-result mapping. The previous choice "
                "remains active until you press Enter or leave this field."
            )
        self._emit_if_changed()

    def _advanced_changed(self, text: str) -> None:
        if self._updating or self.mode_combo.currentData() != self.CUSTOM:
            return
        raw_value = str(text).strip()
        if not raw_value:
            self._show_notice(
                "No mapping entered. Press Enter or leave the field to use the "
                "file's labels unchanged."
            )
            return
        try:
            AxisDeclaration.from_value(raw_value)
        except ValueError:
            self._show_notice(
                "Incomplete axis mapping; nothing has been applied. Complete a "
                "mapping such as QYX -> ZYX, then press Enter or leave the field."
            )
            return
        self._show_notice(
            "Valid axis mapping ready to apply. Press Enter or leave the field; "
            "pixel order will remain unchanged."
        )

    def _advanced_edit_finished(self) -> None:
        if self._updating or self.mode_combo.currentData() != self.CUSTOM:
            return
        raw_value = self.advanced_edit.text().strip()
        if not raw_value:
            self._committed_text = ""
            self._show_notice(
                "Using the file's labels unchanged; no advanced mapping is saved."
            )
            self._emit_if_changed()
            return
        try:
            declaration = AxisDeclaration.from_value(raw_value)
        except ValueError:
            self._show_notice(
                "Axis mapping not applied. Complete a mapping such as "
                "QYX -> ZYX."
            )
            return
        self._committed_text = declaration.display_text
        if self.advanced_edit.text() != declaration.display_text:
            self._updating = True
            try:
                self.advanced_edit.setText(declaration.display_text)
            finally:
                self._updating = False
        self._show_notice(
            "Using this reviewed axis interpretation. Pixel order is unchanged."
        )
        self._emit_if_changed()

    def _emit_if_changed(self) -> None:
        value = self.text()
        if value == self._last_text:
            return
        self._last_text = value
        self.textChanged.emit(value)

    def _show_notice(self, text: str) -> None:
        self.notice_label.setText(text)
        self.notice_label.show()

    def _hide_notice(self) -> None:
        self.notice_label.clear()
        self.notice_label.setToolTip("")
        self.notice_label.hide()


__all__ = ["AxisInterpretationControl"]
