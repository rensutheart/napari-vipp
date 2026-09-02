"""Shared side-panel toggle used throughout VIPP's Qt interface."""

from __future__ import annotations

from qtpy.QtCore import QEvent, QSize, Qt
from qtpy.QtWidgets import QToolButton, QWidget

from napari_vipp.ui.iconography import interface_icon


class SidePanelToggleButton(QToolButton):
    """Compact, unified glyph button for showing or hiding a side panel."""

    def __init__(self, side: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._side = side
        self._expanded = True
        self.setAutoRaise(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.setIconSize(QSize(20, 20))
        self.setFixedSize(30, 26)
        self._refresh_icon()

    def set_expanded(self, expanded: bool) -> None:
        """Show the direction of the action for the current panel state."""

        self._expanded = bool(expanded)
        self._refresh_icon()

    def _direction(self) -> int:
        if self._side == "left":
            return -1 if self._expanded else 1
        return 1 if self._expanded else -1

    def _refresh_icon(self) -> None:
        arrow = "left" if self._direction() < 0 else "right"
        self.setIcon(
            interface_icon(
                f"panel-{self._side}-{arrow}",
                self.palette(),
                20,
            )
        )

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() in (QEvent.PaletteChange, QEvent.StyleChange):
            self._refresh_icon()


__all__ = ["SidePanelToggleButton"]
