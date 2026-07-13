"""Responsive semantic dimension controls synchronized by the host widget."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qtpy.QtCore import QSignalBlocker, Qt, Signal
from qtpy.QtGui import QAction
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from napari_vipp.ui.axis_controls import _control_signal_blockers


@dataclass(frozen=True)
class ViewDimAxis:
    """One semantic non-XY axis exposed in the VIPP-local view controls."""

    name: str
    label: str
    step_axis: int
    size: int
    value: int


class ViewDimAxisControl(QWidget):
    """Responsive slider/spin control for one semantic viewer dimension."""

    value_changed = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._axis: ViewDimAxis | None = None
        self._syncing = False
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.label = QLabel("")
        self.label.setMinimumWidth(18)
        self.label.setStyleSheet("font-weight: 650; color: #cbd5e1;")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimumWidth(120)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.spin = QSpinBox()
        self.spin.setMinimumWidth(54)
        self.spin.setButtonSymbols(QSpinBox.NoButtons)
        self.range_label = QLabel("/0")
        self.range_label.setStyleSheet("color: #94a3b8;")

        layout.addWidget(self.label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)
        layout.addWidget(self.range_label)

        self.slider.valueChanged.connect(self._on_slider_value_changed)
        self.spin.valueChanged.connect(self._on_spin_value_changed)

    def set_axis(self, axis: ViewDimAxis) -> None:
        self._axis = axis
        maximum = max(int(axis.size) - 1, 0)
        with _control_signal_blockers((self.slider, self.spin)):
            self.label.setText(axis.label)
            self.label.setToolTip(f"{axis.name} axis")
            self.slider.setRange(0, maximum)
            self.slider.setValue(int(np.clip(axis.value, 0, maximum)))
            self.spin.setRange(0, maximum)
            self.spin.setValue(int(np.clip(axis.value, 0, maximum)))
            self.range_label.setText(f"/{maximum}")
        self.setToolTip(f"{axis.name} axis, {int(axis.size)} positions")

    def set_display_mode(self, mode: str) -> None:
        compact = mode == "compact"
        self.slider.setVisible(not compact)
        self.range_label.setVisible(True)

    def _on_slider_value_changed(self, value: int) -> None:
        if self._syncing or self._axis is None:
            return
        with QSignalBlocker(self.spin):
            self.spin.setValue(int(value))
        self.value_changed.emit(int(self._axis.step_axis), int(value))

    def _on_spin_value_changed(self, value: int) -> None:
        if self._syncing or self._axis is None:
            return
        with QSignalBlocker(self.slider):
            self.slider.setValue(int(value))
        self.value_changed.emit(int(self._axis.step_axis), int(value))


class ViewDimsBar(QWidget):
    """VIPP-local T/Z/C-style navigation controls synchronized to napari dims."""

    value_changed = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._axes: tuple[ViewDimAxis, ...] = ()
        self._controls: list[ViewDimAxisControl] = []
        self._responsive_mode: str | None = None

        self.setObjectName("ViewDimsBar")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet(
            "#ViewDimsBar {"
            " background: #20262f;"
            " border: 1px solid #374151;"
            " border-radius: 5px;"
            " padding: 2px;"
            "}"
        )

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(6, 3, 6, 3)
        self._layout.setSpacing(6)

        self.title_label = QLabel("View dims")
        self.title_label.setStyleSheet("font-weight: 650; color: #e5e7eb;")
        self._layout.addWidget(self.title_label)

        self.menu_button = QToolButton()
        self.menu_button.setText("View dims...")
        self.menu_button.setPopupMode(QToolButton.InstantPopup)
        self.menu_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.menu_button.setToolTip("Open full T/Z/C view sliders.")
        self.menu = QMenu(self.menu_button)
        self.menu.aboutToShow.connect(self._populate_menu)
        self.menu_button.setMenu(self.menu)
        self._layout.addStretch(1)
        self._layout.addWidget(self.menu_button)
        self.setHidden(True)

    def set_axes(self, axes: list[ViewDimAxis] | tuple[ViewDimAxis, ...]) -> None:
        self._axes = tuple(axes)
        self.setVisible(bool(self._axes))
        self._ensure_control_count(len(self._axes))
        for control, axis in zip(self._controls, self._axes, strict=False):
            control.set_axis(axis)
            control.setVisible(True)
        for control in self._controls[len(self._axes) :]:
            control.setVisible(False)
        self.sync_responsive_mode()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self.sync_responsive_mode()

    def sync_responsive_mode(self) -> None:
        if not self._axes:
            return
        width = max(int(self.width()), 1)
        full_width = 190 + 220 * len(self._axes)
        compact_width = 160 + 86 * len(self._axes)
        if width >= full_width:
            mode = "full"
        elif width >= compact_width:
            mode = "compact"
        else:
            mode = "menu"
        if mode == self._responsive_mode:
            return
        self._responsive_mode = mode
        self.title_label.setVisible(mode != "menu")
        active_controls = self._controls[: len(self._axes)]
        for control in self._controls:
            control.set_display_mode("compact" if mode == "compact" else "full")
            control.setVisible(mode != "menu" and control in active_controls)
        self.menu_button.setVisible(mode != "full")
        self.menu_button.setText("Sliders..." if mode == "compact" else "View dims...")

    def _ensure_control_count(self, count: int) -> None:
        while len(self._controls) < count:
            control = ViewDimAxisControl(self)
            control.value_changed.connect(self.value_changed.emit)
            self._controls.append(control)
            self._layout.insertWidget(max(self._layout.count() - 2, 1), control, 1)

    def _populate_menu(self) -> None:
        self.menu.clear()
        if not self._axes:
            action = QAction("No view dimensions", self.menu)
            action.setEnabled(False)
            self.menu.addAction(action)
            return
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        for axis in self._axes:
            control = ViewDimAxisControl(container)
            control.set_axis(axis)
            control.set_display_mode("full")
            control.value_changed.connect(self.value_changed.emit)
            layout.addWidget(control)
        action = QWidgetAction(self.menu)
        action.setDefaultWidget(container)
        self.menu.addAction(action)
