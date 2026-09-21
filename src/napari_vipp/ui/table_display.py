"""View-only floating-point formatting shared by measurement table surfaces."""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import QEvent, QSize, Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QToolButton, QWidget

from napari_vipp.ui.iconography import interface_icon

DEFAULT_DECIMAL_PLACES = 3
MAX_DECIMAL_PLACES = 15


def validate_decimal_places(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("Display decimal places must be an integer from 0 to 15.")
    if not 0 <= value <= MAX_DECIMAL_PLACES:
        raise ValueError("Display decimal places must be an integer from 0 to 15.")
    return int(value)


def format_table_value(
    value: object, decimal_places: int = DEFAULT_DECIMAL_PLACES
) -> str:
    """Round only floating scalars for display; never coerce text or integers."""
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        if isinstance(value, np.floating):
            # NumPy's __format__ can narrow extended-precision scalars to a
            # Python float. Preserve their range and precision even in a view.
            text = np.format_float_positional(
                value, precision=decimal_places, unique=False, fractional=True, trim="k"
            ).removesuffix(".")
        else:
            text = format(value, f".{decimal_places}f")
        # A negative rounded zero is distracting; the exact sign/value remains
        # available in the tooltip and the original scientific table.
        if text.startswith("-") and not text[1:].strip("0."):
            text = text[1:]
        return text
    return str(value)


class TableDecimalControls(QWidget):
    """Compact, accessible spreadsheet-style precision buttons for one view."""

    decimals_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._decimal_places = DEFAULT_DECIMAL_PLACES
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.setAccessibleName("Table display decimal places")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.label = QLabel(self)
        self.label.setToolTip(
            "Decimal places shown for floating-point cells in this view only. "
            "Integers and original values are unchanged. Hover over a cell to "
            "see its original value; sorting, search and exports use full precision."
        )
        layout.addWidget(self.label)
        self.increase_button = QToolButton(self)
        self.decrease_button = QToolButton(self)
        for button, title, description, delta in (
            (self.increase_button, "Increase Decimal", "Show more decimal places", 1),
            (self.decrease_button, "Decrease Decimal", "Show fewer decimal places", -1),
        ):
            button.setText(title)
            button.setAccessibleName(title)
            button.setIconSize(QSize(24, 24))
            button.setToolTip(
                f"{title} — {description} for floating-point cells (0–15). "
                "Display only; integers, calculations and exports are unchanged."
            )
            button.clicked.connect(
                lambda _checked=False, step=delta: self.set_decimal_places(
                    max(0, min(MAX_DECIMAL_PLACES, self._decimal_places + step))
                )
            )
            layout.addWidget(button)
        self._refresh()

    @property
    def decimal_places(self) -> int:
        return self._decimal_places

    def set_decimal_places(self, value: int) -> None:
        value = validate_decimal_places(value)
        if value == self._decimal_places:
            return
        self._decimal_places = value
        self._refresh()
        self.decimals_changed.emit(value)

    def _refresh(self) -> None:
        self.label.setText(f"Decimals: {self._decimal_places}")
        self.increase_button.setEnabled(self._decimal_places < MAX_DECIMAL_PLACES)
        self.decrease_button.setEnabled(self._decimal_places > 0)
        self._refresh_icons()

    def _refresh_icons(self) -> None:
        for button, kind in (
            (self.increase_button, "increase-decimals"),
            (self.decrease_button, "decrease-decimals"),
        ):
            button.setIcon(interface_icon(kind, self.palette(), 24))

    def changeEvent(self, event):  # noqa: N802
        super().changeEvent(event)
        if hasattr(self, "decrease_button") and event.type() in {
            QEvent.PaletteChange,
            QEvent.ApplicationPaletteChange,
            QEvent.StyleChange,
        }:
            self._refresh_icons()
