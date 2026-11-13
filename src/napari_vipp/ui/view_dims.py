"""Responsive semantic dimension controls synchronized by the host widget."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qtpy.QtCore import QEvent, QRect, QSignalBlocker, Qt, QTimer, Signal
from qtpy.QtGui import QAction, QIcon, QPainter, QPalette
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QSizePolicy,
    QSpacerItem,
    QSpinBox,
    QStyle,
    QStyleOptionSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from napari_vipp.ui.axis_controls import _control_signal_blockers
from napari_vipp.ui.design import SPACE_SM, SPACE_XS, type_style
from napari_vipp.ui.iconography import interface_icon
from napari_vipp.ui.palette_roles import custom_paint_colors
from napari_vipp.ui.sliders import VippSlider

_VIEW_DIMENSIONS = "View dimensions"
_SHOW_SLIDERS = "Show sliders"
_SLIDER_MINIMUM_WIDTH = 120
_VIEW_DIMENSIONS_HELP = (
    "Browse Z slices, time points, or channels for the current image. "
    "This changes the view, not node settings."
)


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
        self._applying_palette_styles = False
        self._slider_padding = 0
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE_XS)

        self.label = QLabel("")
        self.label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.slider = VippSlider(Qt.Horizontal)
        self.slider.setMinimumWidth(_SLIDER_MINIMUM_WIDTH)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.spin = QSpinBox()
        self.spin.setButtonSymbols(QSpinBox.NoButtons)
        self.spin.setAlignment(Qt.AlignCenter)
        self.spin.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self._metric_refresh = QTimer(self)
        self._metric_refresh.setSingleShot(True)
        self._metric_refresh.timeout.connect(self._refresh_spin_geometry)
        self.spin.installEventFilter(self)
        self.range_label = QLabel("/0")
        self.range_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self._trailing_space = QSpacerItem(0, 0, QSizePolicy.Fixed, QSizePolicy.Fixed)

        layout.addWidget(self.label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)
        layout.addWidget(self.range_label)
        layout.addItem(self._trailing_space)

        self.slider.valueChanged.connect(self._on_slider_value_changed)
        self.spin.valueChanged.connect(self._on_spin_value_changed)
        self._apply_palette_styles()

    def changeEvent(self, event):  # noqa: N802
        super().changeEvent(event)
        if event.type() in (
            QEvent.PaletteChange,
            QEvent.StyleChange,
            QEvent.FontChange,
            QEvent.ApplicationFontChange,
        ) and hasattr(self, "range_label"):
            self._apply_palette_styles()
            self._fit_axis_metrics()
            self._metric_refresh.start(0)
            parent = self.parentWidget()
            if isinstance(parent, ViewDimsBar):
                parent.sync_responsive_mode()

    def eventFilter(self, watched, event):  # noqa: N802
        if watched is self.spin and event.type() in (
            QEvent.Polish, QEvent.Show, QEvent.StyleChange, QEvent.FontChange,
        ):
            self._metric_refresh.start(0)
        return super().eventFilter(watched, event)

    def _refresh_spin_geometry(self) -> None:
        # Native/style-sheet polish can overwrite the explicit minimum width.
        # Reapply the measured bounds once after Qt has polished the editor.
        self._fit_axis_metrics()
        parent = self.parentWidget()
        if isinstance(parent, ViewDimsBar):
            parent.sync_responsive_mode()

    def _apply_palette_styles(self) -> None:
        if self._applying_palette_styles:
            return
        self._applying_palette_styles = True
        colors = custom_paint_colors(self.palette())
        try:
            # Define the frame as well as padding. Leaving a partly styled
            # spin box's frame to the native style can retain a large minimum
            # edit-field inset (including a one-pixel editor on macOS).
            self.spin.setStyleSheet(
                "QSpinBox { min-width: 0px; padding-left: 4px; padding-right: 4px;"
                f" border: 1px solid {colors.border.name()}; border-radius: 3px; }}"
            )
            self.label.setStyleSheet(
                f"{type_style(self.label, 'section')} color: {colors.text.name()};"
            )
            self.range_label.setStyleSheet(
                f"{type_style(self.range_label, 'secondary')}"
                f" color: {colors.muted_text.name()};"
            )
        finally:
            self._applying_palette_styles = False

    def _fit_axis_metrics(self) -> None:
        if self._axis is None:
            return
        # Use the editor's actual style chrome, without the generic themed
        # input minimum or Qt's extra range-based size-hint allowance.
        spin = self.spin
        spin.ensurePolished()
        option = QStyleOptionSpinBox()
        spin.initStyleOption(option)
        option.rect = QRect(0, 0, 512, spin.sizeHint().height())
        field = spin.style().subControlRect(
            QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxEditField, spin
        )
        editor = spin.lineEdit()
        margins = editor.textMargins()
        digit = max(
            editor.fontMetrics().horizontalAdvance(spin.textFromValue(value))
            for value in range(10)
        )
        chrome = max(0, option.rect.width() - field.width())
        target = (
            len(spin.textFromValue(spin.maximum())) * digit
            + chrome
            + margins.left()
            + margins.right()
            + SPACE_SM
            + SPACE_XS
        )
        if spin.minimumWidth() != target or spin.maximumWidth() != target:
            spin.setFixedWidth(target)

    def inline_width(self) -> int:
        """Width of the label, position and limit, excluding the slider."""
        return (
            self.label.sizeHint().width()
            + self.spin.maximumWidth()
            + self.range_label.sizeHint().width()
            + 2 * self.layout().spacing()
        )

    def match_slider_chrome(self, target_width: int) -> None:
        """Equalize group widths outside the slider, not inside the fields."""
        padding = max(0, target_width - self.inline_width())
        if padding != self._slider_padding:
            self._slider_padding = padding
            self._trailing_space.changeSize(
                padding, 0, QSizePolicy.Fixed, QSizePolicy.Fixed
            )
            self.layout().invalidate()

    def set_axis(self, axis: ViewDimAxis) -> None:
        self._axis = axis
        maximum = max(int(axis.size) - 1, 0)
        with _control_signal_blockers((self.slider, self.spin)):
            self.label.setText(axis.label)
            self.label.setToolTip(f"{axis.name} axis")
            self.slider.setAccessibleName(f"{axis.name} view slider")
            self.spin.setAccessibleName(f"{axis.name} view position")
            self.slider.setRange(0, maximum)
            self.slider.setValue(int(np.clip(axis.value, 0, maximum)))
            self.spin.setRange(0, maximum)
            self.spin.setValue(int(np.clip(axis.value, 0, maximum)))
            self.range_label.setText(f"/{maximum}")
        self._fit_axis_metrics()
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


class _ViewDimensionsButton(QToolButton):
    """Keep a font-independent menu chevron visible under every Qt style."""

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        size = 18
        glyph = QStyle.visualRect(
            self.layoutDirection(),
            self.rect(),
            QRect(
                self.width() - size - SPACE_XS, (self.height() - size) // 2, size, size
            ),
        )
        palette = self.palette()
        colors = custom_paint_colors(palette)
        palette.setColor(QPalette.ButtonText, colors.text)
        palette.setColor(QPalette.Button, colors.surface)
        painter = QPainter(self)
        interface_icon("chevron-down", palette, size).paint(
            painter,
            glyph,
            Qt.AlignCenter,
            QIcon.Normal if self.isEnabled() else QIcon.Disabled,
        )
        painter.end()


class ViewDimsBar(QWidget):
    """VIPP-local T/Z/C-style navigation controls synchronized to napari dims."""

    value_changed = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._axes: tuple[ViewDimAxis, ...] = ()
        self._controls: list[ViewDimAxisControl] = []
        self._responsive_mode: str | None = None
        self._responsive_in_progress = False
        self._applying_palette_styles = False

        self.setObjectName("ViewDimsBar")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(SPACE_SM, SPACE_XS, SPACE_SM, SPACE_XS)
        self._layout.setSpacing(SPACE_SM)

        self.title_label = QLabel(_VIEW_DIMENSIONS)
        self.title_label.setToolTip(_VIEW_DIMENSIONS_HELP)
        self._layout.addWidget(self.title_label)

        self.menu_button = _ViewDimensionsButton()
        self.menu_button.setObjectName("ViewDimensionsMenuButton")
        self.menu_button.setText(_VIEW_DIMENSIONS)
        self.menu_button.setAccessibleName(_VIEW_DIMENSIONS)
        self.menu_button.setAccessibleDescription(_VIEW_DIMENSIONS_HELP)
        self.menu_button.setPopupMode(QToolButton.InstantPopup)
        self.menu_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.menu_button.setFocusPolicy(Qt.StrongFocus)
        self.menu_button.setToolTip(f"Open view sliders. {_VIEW_DIMENSIONS_HELP}")
        self.menu = QMenu(self.menu_button)
        self.menu.setAccessibleName(_VIEW_DIMENSIONS)
        self.menu.aboutToShow.connect(self._populate_menu)
        self.menu_button.setMenu(self.menu)
        self._layout.addWidget(self.menu_button)
        self._layout.addStretch()
        self._apply_palette_styles()
        self.setHidden(True)

    def changeEvent(self, event):  # noqa: N802
        super().changeEvent(event)
        if event.type() in (
            QEvent.PaletteChange,
            QEvent.StyleChange,
            QEvent.FontChange,
            QEvent.ApplicationFontChange,
            QEvent.LayoutDirectionChange,
        ) and hasattr(self, "title_label"):
            self._apply_palette_styles()
            self.sync_responsive_mode()

    def _apply_palette_styles(self) -> None:
        if self._applying_palette_styles:
            return
        self._applying_palette_styles = True
        colors = custom_paint_colors(self.palette())
        try:
            self.setStyleSheet(
                "#ViewDimsBar {"
                f" background: {colors.alternate_surface.name()};"
                f" border: 1px solid {colors.border.name()};"
                " border-radius: 4px;"
                " padding: 2px;"
                "}"
            )
            # Reserve room for the painted indicator on either trailing edge.
            # Suppress the native glyph rather than showing two indicators.
            button = "QToolButton#ViewDimensionsMenuButton"
            right, left = (8, 24) if self.isRightToLeft() else (24, 8)
            self.menu_button.setStyleSheet(
                f"{button} {{ {type_style(self.menu_button, 'body')}"
                f" color: {colors.text.name()};"
                f" background: {colors.surface.name()};"
                f" border: 1px solid {colors.border.name()}; border-radius: 4px;"
                f" padding: 2px {right}px 2px {left}px; }}"
                f"{button}:hover, {button}:pressed {{"
                f" background: {colors.alternate_surface.name()}; }}"
                f"{button}:focus {{"
                f" border-color: {self.palette().color(QPalette.Highlight).name()}; }}"
                f"{button}::menu-indicator {{ image: none; width: 0px; }}"
            )
            if hasattr(self, "title_label"):
                self.title_label.setStyleSheet(
                    f"{type_style(self.title_label, 'section')}"
                    f" color: {colors.text.name()};"
                )
        finally:
            self._applying_palette_styles = False

    def set_axes(self, axes: list[ViewDimAxis] | tuple[ViewDimAxis, ...]) -> None:
        # Creating controls and showing the bar can synchronously deliver
        # Qt style/resize events. Publish the new axes only after they exist.
        self._ensure_control_count(len(axes))
        for control, axis in zip(self._controls, axes, strict=False):
            control.set_axis(axis)
        self._axes = tuple(axes)
        self.sync_responsive_mode()
        self.setVisible(bool(self._axes))

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self.sync_responsive_mode()

    def sync_responsive_mode(self) -> None:
        if not self._axes or self._responsive_in_progress:
            return
        self._responsive_in_progress = True
        try:
            self._update_responsive_mode()
        finally:
            self._responsive_in_progress = False

    def _update_responsive_mode(self) -> None:
        width = max(int(self.width()), 1)
        parent = self.parentWidget()
        if parent is not None:
            width = min(width, max(int(parent.contentsRect().width()) - 12, 1))

        def minimum_widget_width(widget: QWidget) -> int:
            return max(
                int(widget.minimumWidth()),
                int(widget.minimumSizeHint().width()),
                int(widget.sizeHint().width()),
                0,
            )

        margins = self._layout.contentsMargins()
        outer_padding = margins.left() + margins.right() + 12
        active_controls = self._controls[: len(self._axes)]
        count = len(active_controls)
        chrome = [control.inline_width() for control in active_controls]
        widest_chrome = max(chrome)
        gaps = count * self._layout.spacing()
        full_width = (
            outer_padding
            + minimum_widget_width(self.title_label)
            + count * (widest_chrome + _SLIDER_MINIMUM_WIDTH + SPACE_XS)
            + gaps
        )
        # Measure the compact caption regardless of the currently shown
        # caption, so changing modes cannot move its own breakpoint.
        metrics = self.menu_button.fontMetrics()
        compact_button_width = (
            minimum_widget_width(self.menu_button)
            - metrics.horizontalAdvance(self.menu_button.text())
            + metrics.horizontalAdvance(_SHOW_SLIDERS)
        )
        compact_width = outer_padding + compact_button_width + sum(chrome) + gaps
        if width >= full_width:
            mode = "full"
        elif width >= compact_width:
            mode = "compact"
        else:
            mode = "menu"
        # Reapply the complete visibility state even if the mode name stayed
        # the same: a refreshed axis or font must not revive hidden sliders.
        self._responsive_mode = mode
        self.title_label.setVisible(mode == "full")
        caption = _SHOW_SLIDERS if mode == "compact" else _VIEW_DIMENSIONS
        self.menu_button.setText(caption)
        self.menu_button.setAccessibleName(caption)
        self.menu_button.setToolTip(
            "Open sliders for the same dimensions shown here."
            if mode == "compact"
            else f"Open view sliders. {_VIEW_DIMENSIONS_HELP}"
        )
        for index, control in enumerate(self._controls):
            control.set_display_mode("compact" if mode == "compact" else "full")
            control.match_slider_chrome(widest_chrome if mode == "full" else 0)
            control.setVisible(mode != "menu" and control in active_controls)
            self._layout.setStretch(index + 2, int(mode == "full"))
        self.menu_button.setVisible(mode != "full")
        self._layout.setStretch(self._layout.count() - 1, int(mode != "full"))

    def _ensure_control_count(self, count: int) -> None:
        while len(self._controls) < count:
            control = ViewDimAxisControl(self)
            control.value_changed.connect(self.value_changed.emit)
            self._controls.append(control)
            self._layout.insertWidget(self._layout.count() - 1, control, 1)

    def _populate_menu(self) -> None:
        self.menu.clear()
        if not self._axes:
            action = QAction("No view dimensions", self.menu)
            action.setEnabled(False)
            self.menu.addAction(action)
            return
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(SPACE_SM, SPACE_SM, SPACE_SM, SPACE_SM)
        layout.setSpacing(SPACE_SM)
        title = QLabel(_VIEW_DIMENSIONS, container)
        title.setStyleSheet(type_style(title, "section"))
        layout.addWidget(title)
        help_label = QLabel(_VIEW_DIMENSIONS_HELP, container)
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        controls = []
        for axis in self._axes:
            control = ViewDimAxisControl(container)
            control.set_axis(axis)
            control.set_display_mode("full")
            control.value_changed.connect(self.value_changed.emit)
            layout.addWidget(control)
            controls.append(control)
        widest_chrome = max(control.inline_width() for control in controls)
        for control in controls:
            control.match_slider_chrome(widest_chrome)
        action = QWidgetAction(self.menu)
        action.setDefaultWidget(container)
        self.menu.addAction(action)
