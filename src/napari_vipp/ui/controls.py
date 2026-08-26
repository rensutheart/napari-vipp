"""Reusable parameter and image-source controls for the VIPP UI."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

import numpy as np
from qtpy.QtCore import QLocale, QSignalBlocker, Qt, Signal
from qtpy.QtGui import QAction, QValidator
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QWidget,
)

from napari_vipp.core.io import MICROSCOPE_FILE_FILTER
from napari_vipp.ui import recent_paths
from napari_vipp.ui.axis_interpretation import AxisInterpretationControl
from napari_vipp.ui.file_sources import (
    SourceLoadPhase,
    SourceLoadProgress,
    SourceLoadProgressUnit,
)


@dataclass(frozen=True)
class ParameterBounds:
    minimum: float | int
    maximum: float | int
    step: float | int
    decimals: int
    expandable: bool = False
    logarithmic: bool = False
    entry_minimum: float | int | None = None
    entry_maximum: float | int | None = None


@dataclass(frozen=True)
class ImageSourceResolutionPresentation:
    """Read-only pyramid state that never enters Image Source parameters."""

    analysis_axes: str = ""
    analysis_shape: tuple[int, ...] = ()
    level_shapes: tuple[tuple[int, ...], ...] = ()
    preview_state: str = "idle"
    preview_level: int | None = None
    preview_shape: tuple[int, ...] = ()
    preview_detail: str = ""
    viewer_choice: str = "analysis"
    can_select_preview: bool = False
    can_retry: bool = False

    @property
    def visible(self) -> bool:
        return bool(self.analysis_shape) and len(self.level_shapes) > 1


def _slider_safe_bounds(
    minimum: float,
    maximum: float,
    step: float | int,
    decimals: int,
    expandable: bool = False,
    logarithmic: bool = False,
    entry_minimum: float | int | None = None,
    entry_maximum: float | int | None = None,
) -> ParameterBounds:
    maximum_slider_units = 1_000_000_000
    decimals = int(decimals)
    numeric_minimum = float(minimum)
    numeric_maximum = float(maximum)
    extent = max(
        abs(numeric_minimum),
        abs(numeric_maximum),
        abs(numeric_maximum - numeric_minimum),
        np.finfo(float).tiny,
    )
    if not logarithmic:
        while decimals > 0 and extent * (10**decimals) > maximum_slider_units:
            decimals -= 1
        if extent > maximum_slider_units:
            # Absolute numeric levels outside QSlider's signed-int storage
            # cannot use the ordinary scaled mapping. The existing logarithmic
            # mapping uses a fixed 0..1000 slider while the spin box retains the
            # exact wider entry range supplied by the caller.
            logarithmic = True

    smallest_step = 1.0 if decimals == 0 else 10 ** (-decimals)
    safe_step = max(float(step), smallest_step)
    if not logarithmic:
        safe_step = min(safe_step, float(maximum_slider_units))
    return ParameterBounds(
        float(minimum),
        float(maximum),
        safe_step,
        decimals,
        expandable,
        logarithmic,
        entry_minimum,
        entry_maximum,
    )


class _ResettableSpinBoxMixin:
    """Add a parameter-default action without replacing standard edit actions."""

    _default_value: float | int | None = None

    def setDefaultValue(self, value: float | int) -> None:  # noqa: N802
        self._default_value = value

    def resetToDefault(self) -> None:  # noqa: N802
        if self._default_value is not None:
            self.setValue(self._default_value)

    def _create_context_menu(self) -> tuple[QMenu, QAction]:
        menu = self.lineEdit().createStandardContextMenu()
        menu.setParent(self)
        menu.addSeparator()
        reset_action = menu.addAction("Reset to default")
        reset_action.setEnabled(
            self._default_value is not None and self.value() != self._default_value
        )
        reset_action.triggered.connect(self.resetToDefault)
        return menu, reset_action

    def contextMenuEvent(self, event):  # noqa: N802
        menu, _reset_action = self._create_context_menu()
        try:
            menu.exec(event.globalPos())
        finally:
            menu.deleteLater()
        event.accept()


class ResettableSpinBox(_ResettableSpinBoxMixin, QSpinBox):
    """Integer parameter entry with a default-reset context action."""


class FlexibleDoubleSpinBox(_ResettableSpinBoxMixin, QDoubleSpinBox):
    """Locale-independent float entry accepting decimal points and commas."""

    SCIENTIFIC_THRESHOLD = 1e-3

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLocale(QLocale.c())
        self.setKeyboardTracking(False)

    @staticmethod
    def _normalized_text(text: str) -> str:
        return str(text).replace(",", ".")

    def validate(self, text: str, position: int):
        normalized = self._normalized_text(text)
        if re.fullmatch(
            r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?",
            normalized.strip(),
        ):
            return QValidator.Intermediate, text, position
        state, _normalized, validated_position = super().validate(
            normalized,
            position,
        )
        return state, text, validated_position

    def valueFromText(self, text: str) -> float:
        return super().valueFromText(self._normalized_text(text))

    def textFromValue(self, value: float) -> str:  # noqa: N802
        decimals = max(int(self.decimals()), 0)
        if 0.0 < abs(float(value)) < self.SCIENTIFIC_THRESHOLD:
            mantissa, exponent = f"{float(value):.{decimals}e}".split("e")
            mantissa = mantissa.rstrip("0").rstrip(".")
            return f"{mantissa}e{int(exponent)}"
        text = f"{float(value):.{decimals}f}" if decimals else f"{float(value):.0f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return "0" if text == "-0" else text


def _configure_numeric_spin_box(box: QSpinBox | QDoubleSpinBox) -> None:
    box.setKeyboardTracking(False)
    editor = box.lineEdit()
    editor.setAlignment(Qt.AlignCenter)
    editor.setTextMargins(0, 0, 0, 0)


class ParameterControl(QWidget):
    """Slider with numeric entry for a single node parameter."""

    valueChanged = Signal(object)

    def __init__(self, spec, value, bounds: ParameterBounds, parent=None):
        super().__init__(parent)
        self.spec = spec
        self._is_integer = spec.kind == "int"
        self._bounds = bounds
        self._entry_minimum = bounds.minimum
        self._entry_maximum = bounds.maximum
        self._scale = self._scale_for(bounds)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimumWidth(120)
        if self._is_integer:
            self.value_box = ResettableSpinBox()
        else:
            self.value_box = FlexibleDoubleSpinBox()
            self.value_box.setDecimals(bounds.decimals)
        self.value_box.setDefaultValue(spec.default)
        _configure_numeric_spin_box(self.value_box)
        self.value_box.setMinimumWidth(74)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_box)

        self.set_bounds(bounds, value, emit=False)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.value_box.valueChanged.connect(self._on_box_changed)

    def value(self):
        return self.value_box.value()

    def set_bounds(
        self,
        bounds: ParameterBounds,
        value=None,
        emit: bool = False,
    ) -> None:
        entry_minimum, entry_maximum = self._entry_bounds_for(bounds)
        current = self.value() if value is None else value
        current = self._clamped_value(current, entry_minimum, entry_maximum)
        bounds = self._expanded_bounds_for_value(bounds, current)
        bounds = _slider_safe_bounds(
            bounds.minimum,
            bounds.maximum,
            bounds.step,
            bounds.decimals,
            expandable=bounds.expandable,
            logarithmic=bounds.logarithmic,
            entry_minimum=bounds.entry_minimum,
            entry_maximum=bounds.entry_maximum,
        )
        self._bounds = bounds
        self._entry_minimum = entry_minimum
        self._entry_maximum = entry_maximum
        self._scale = self._scale_for(bounds)

        with QSignalBlocker(self.slider), QSignalBlocker(self.value_box):
            if self._is_integer:
                self.value_box.setRange(int(entry_minimum), int(entry_maximum))
                self.value_box.setSingleStep(max(int(bounds.step), 1))
                if bounds.logarithmic:
                    self.slider.setRange(0, 1000)
                    self.slider.setSingleStep(1)
                else:
                    self.slider.setRange(int(bounds.minimum), int(bounds.maximum))
                    self.slider.setSingleStep(max(int(bounds.step), 1))
                self.slider.setValue(self._bounded_slider_value(current))
                self.value_box.setValue(int(current))
            else:
                self.value_box.setDecimals(bounds.decimals)
                self.value_box.setRange(float(entry_minimum), float(entry_maximum))
                self.value_box.setSingleStep(float(bounds.step))
                if bounds.logarithmic:
                    self.slider.setRange(0, 1000)
                    self.slider.setSingleStep(1)
                else:
                    self.slider.setRange(
                        self._to_slider(bounds.minimum),
                        self._to_slider(bounds.maximum),
                    )
                    self.slider.setSingleStep(max(self._to_slider(bounds.step), 1))
                self.slider.setValue(self._bounded_slider_value(current))
                self.value_box.setValue(float(current))

        if emit:
            self.valueChanged.emit(self.value())

    def _on_slider_changed(self, value: int) -> None:
        mapped = self._from_slider(value)
        with QSignalBlocker(self.value_box):
            self.value_box.setValue(mapped)
        self.valueChanged.emit(self.value())

    def _on_box_changed(self, value) -> None:
        if self._bounds.expandable and not self._value_in_slider_bounds(value):
            self.set_bounds(self._bounds, value, emit=False)
            self.valueChanged.emit(self.value())
            return
        with QSignalBlocker(self.slider):
            self.slider.setValue(self._bounded_slider_value(value))
        self.valueChanged.emit(self.value())

    def _bounded_slider_value(self, value) -> int:
        slider_value = self._to_slider(value)
        return min(max(slider_value, self.slider.minimum()), self.slider.maximum())

    def _scale_for(self, bounds: ParameterBounds) -> int:
        if self._is_integer:
            return 1
        if bounds.decimals > 0:
            return 10**bounds.decimals
        step = float(bounds.step)
        return max(int(round(1 / step)), 1) if step > 0 else 100

    def _to_slider(self, value) -> int:
        if self._bounds.logarithmic:
            minimum = float(self._bounds.minimum)
            maximum = float(self._bounds.maximum)
            span = maximum - minimum
            if span <= 0:
                return 0
            if minimum > 0.0:
                clipped = float(np.clip(float(value), minimum, maximum))
                fraction = np.log(clipped / minimum) / np.log(maximum / minimum)
                return int(round(fraction * 1000))
            offset = float(np.clip(float(value) - minimum, 0.0, span))
            fraction = np.log1p(offset) / np.log1p(span)
            return int(round(fraction * 1000))
        if self._is_integer:
            return int(round(float(value)))
        return int(round(float(value) * self._scale))

    def _from_slider(self, value: int):
        if self._bounds.logarithmic:
            minimum = float(self._bounds.minimum)
            maximum = float(self._bounds.maximum)
            span = maximum - minimum
            fraction = float(np.clip(value, 0, 1000)) / 1000.0
            if minimum > 0.0 and span > 0.0:
                mapped = minimum * np.exp(
                    fraction * np.log(maximum / minimum)
                )
            else:
                mapped = minimum + np.expm1(
                    fraction * np.log1p(max(span, 0.0))
                )
            return int(round(mapped)) if self._is_integer else mapped
        return int(value) if self._is_integer else value / self._scale

    def _clamped_value(self, value, minimum, maximum):
        if value is None:
            value = minimum
        return min(max(value, minimum), maximum)

    def _entry_bounds_for(
        self,
        bounds: ParameterBounds,
    ) -> tuple[float | int, float | int]:
        if bounds.entry_minimum is not None or bounds.entry_maximum is not None:
            minimum = (
                bounds.minimum if bounds.entry_minimum is None else bounds.entry_minimum
            )
            maximum = (
                bounds.maximum if bounds.entry_maximum is None else bounds.entry_maximum
            )
            return minimum, maximum
        if not bounds.expandable:
            return bounds.minimum, bounds.maximum
        minimum = bounds.minimum
        maximum = bounds.maximum
        if float(minimum) < 0:
            minimum = min(float(minimum), -1_000_000.0)
        maximum = max(float(maximum), 1_000_000.0)
        if self._is_integer:
            return int(round(minimum)), int(round(maximum))
        return float(minimum), float(maximum)

    def _expanded_bounds_for_value(
        self,
        bounds: ParameterBounds,
        value,
    ) -> ParameterBounds:
        if not bounds.expandable:
            return bounds
        minimum = float(bounds.minimum)
        maximum = float(bounds.maximum)
        value = float(value)
        span = max(
            maximum - minimum,
            abs(maximum),
            abs(minimum),
            float(bounds.step),
            1.0,
        )
        if value > maximum:
            maximum = value + max(span * 0.25, abs(value) * 0.25, float(bounds.step))
        if value < minimum and minimum < 0:
            minimum = value - max(span * 0.25, abs(value) * 0.25, float(bounds.step))
        if self._is_integer:
            return ParameterBounds(
                int(np.floor(minimum)),
                int(np.ceil(maximum)),
                bounds.step,
                bounds.decimals,
                bounds.expandable,
                bounds.logarithmic,
                bounds.entry_minimum,
                bounds.entry_maximum,
            )
        return ParameterBounds(
            minimum,
            maximum,
            bounds.step,
            bounds.decimals,
            bounds.expandable,
            bounds.logarithmic,
            bounds.entry_minimum,
            bounds.entry_maximum,
        )

    def _value_in_slider_bounds(self, value) -> bool:
        if self._bounds.logarithmic:
            return (
                float(self._bounds.minimum)
                <= float(value)
                <= float(self._bounds.maximum)
            )
        if self._is_integer:
            return self.slider.minimum() <= int(value) <= self.slider.maximum()
        slider_value = self._to_slider(value)
        return self.slider.minimum() <= slider_value <= self.slider.maximum()


class NumericEntryControl(QWidget):
    """Numeric entry without a slider for parameters where sliders are misleading."""

    valueChanged = Signal(object)

    def __init__(self, spec, value, bounds: ParameterBounds, parent=None):
        super().__init__(parent)
        self.spec = spec
        self._is_integer = spec.kind == "int"
        if self._is_integer:
            self.value_box = ResettableSpinBox()
        else:
            self.value_box = FlexibleDoubleSpinBox()
            self.value_box.setDecimals(bounds.decimals)
        self.value_box.setDefaultValue(spec.default)
        _configure_numeric_spin_box(self.value_box)
        self.value_box.setMinimumWidth(100)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.value_box, 1)

        self.set_bounds(bounds, value, emit=False)
        self.value_box.valueChanged.connect(self.valueChanged.emit)

    def value(self):
        return self.value_box.value()

    def set_bounds(
        self,
        bounds: ParameterBounds,
        value=None,
        emit: bool = False,
    ) -> None:
        minimum = (
            bounds.minimum if bounds.entry_minimum is None else bounds.entry_minimum
        )
        maximum = (
            bounds.maximum if bounds.entry_maximum is None else bounds.entry_maximum
        )
        current = minimum if value is None else value
        if self._is_integer:
            current = int(np.clip(int(current), int(minimum), int(maximum)))
            with QSignalBlocker(self.value_box):
                self.value_box.setRange(int(minimum), int(maximum))
                self.value_box.setSingleStep(max(int(bounds.step), 1))
                self.value_box.setValue(current)
        else:
            current = float(np.clip(float(current), float(minimum), float(maximum)))
            with QSignalBlocker(self.value_box):
                self.value_box.setDecimals(bounds.decimals)
                self.value_box.setRange(float(minimum), float(maximum))
                self.value_box.setSingleStep(float(bounds.step))
                self.value_box.setValue(current)
        if emit:
            self.valueChanged.emit(self.value())


class ChoiceControl(QWidget):
    """Dropdown control for categorical node parameters."""

    valueChanged = Signal(object)

    def __init__(self, spec, value, bounds: ParameterBounds, parent=None):
        super().__init__(parent)
        self.spec = spec
        self._bounds = bounds
        self.combo = QComboBox()
        self._set_combo_items(spec.choices, spec.choice_labels)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.combo, 1)

        self.set_bounds(bounds, value, emit=False)
        self.combo.currentIndexChanged.connect(self._emit_current_value)

    def value(self):
        value = self.combo.currentData()
        return self.combo.currentText() if value is None else value

    def _emit_current_value(self, _index: int) -> None:
        self.valueChanged.emit(self.value())

    def _set_combo_items(
        self,
        choices: tuple[str, ...],
        choice_labels: tuple[str, ...] = (),
    ) -> None:
        labels = self._choice_labels(choices, choice_labels)
        self.combo.clear()
        for value, label in zip(choices, labels, strict=True):
            self.combo.addItem(label, value)

    @staticmethod
    def _choice_labels(
        choices: tuple[str, ...],
        choice_labels: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        if len(choice_labels) == len(choices):
            return tuple(str(label) for label in choice_labels)
        return tuple(str(choice) for choice in choices)

    def set_choices(
        self,
        choices: tuple[str, ...],
        value=None,
        emit: bool = False,
        choice_labels: tuple[str, ...] = (),
    ) -> None:
        self.spec = replace(
            self.spec,
            choices=tuple(choices),
            choice_labels=tuple(choice_labels),
        )
        current = self.spec.default if value is None else str(value)
        with QSignalBlocker(self.combo):
            self._set_combo_items(self.spec.choices, self.spec.choice_labels)
            index = self.combo.findData(current)
            self.combo.setCurrentIndex(max(index, 0))
        if emit:
            self.valueChanged.emit(self.value())

    def set_bounds(
        self,
        bounds: ParameterBounds,
        value=None,
        emit: bool = False,
    ) -> None:
        self._bounds = bounds
        current = self.spec.default if value is None else value
        current = str(current)
        with QSignalBlocker(self.combo):
            index = self.combo.findData(current)
            if index < 0:
                index = 0
            self.combo.setCurrentIndex(index)
        if emit:
            self.valueChanged.emit(self.value())


class TextControl(QWidget):
    """Single-line text control for path-like and free text parameters."""

    valueChanged = Signal(object)

    def __init__(self, spec, value, _bounds: ParameterBounds, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.edit = QLineEdit()
        self.edit.setText("" if value is None else str(value))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1)
        self.edit.textChanged.connect(self.valueChanged.emit)

    def value(self):
        return self.edit.text()

    def set_bounds(
        self,
        _bounds: ParameterBounds,
        value=None,
        emit: bool = False,
    ) -> None:
        current = "" if value is None else str(value)
        with QSignalBlocker(self.edit):
            self.edit.setText(current)
        if emit:
            self.valueChanged.emit(self.value())


class BoolControl(QWidget):
    """Checkbox control for boolean node parameters."""

    valueChanged = Signal(object)

    def __init__(self, spec, value, _bounds: ParameterBounds, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.checkbox = QCheckBox()
        self.checkbox.setChecked(bool(spec.default if value is None else value))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.checkbox)
        layout.addStretch(1)
        self.checkbox.toggled.connect(self.valueChanged.emit)

    def value(self):
        return self.checkbox.isChecked()

    def set_bounds(
        self,
        _bounds: ParameterBounds,
        value=None,
        emit: bool = False,
    ) -> None:
        current = self.spec.default if value is None else value
        with QSignalBlocker(self.checkbox):
            self.checkbox.setChecked(bool(current))
        if emit:
            self.valueChanged.emit(self.value())


class SourceLoadStatusControl(QWidget):
    """Generation-gated source progress with one-shot cancellation."""

    cancelRequested = Signal(int)

    _PHASE_LABELS = {
        SourceLoadPhase.INSPECT: "Inspecting source",
        SourceLoadPhase.DOWNLOAD: "Downloading source",
        SourceLoadPhase.READ: "Reading image item",
        SourceLoadPhase.DECODE: "Decoding image data",
        SourceLoadPhase.NORMALIZE: "Normalizing metadata",
        SourceLoadPhase.PREVIEW: "Loading presentation preview",
        SourceLoadPhase.VERIFY: "Verifying source revision",
    }
    _QT_PROGRESS_MAXIMUM = 1_000

    def __init__(self, parent=None):
        super().__init__(parent)
        self._active_run_id: int | None = None
        self._latest_run_id = -1
        self._cancel_requested_for: int | None = None
        self._has_status = False

        self.label = QLabel()
        self.label.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.cancel_button = QPushButton("Cancel load")
        self.cancel_button.setMaximumWidth(104)

        progress_row = QHBoxLayout()
        progress_row.setContentsMargins(0, 0, 0, 0)
        progress_row.addWidget(self.progress_bar, 1)
        progress_row.addWidget(self.cancel_button)
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addRow(self.label)
        layout.addRow(progress_row)

        self.cancel_button.clicked.connect(self._request_cancel)
        self.hide()

    @property
    def active_run_id(self) -> int | None:
        return self._active_run_id

    @property
    def has_status(self) -> bool:
        return self._has_status

    def begin(self, run_id: int, message: str = "Inspecting source.") -> bool:
        """Activate a newer load generation and reject stale starts."""

        run_id = int(run_id)
        if run_id <= self._latest_run_id:
            return False
        self._latest_run_id = run_id
        self._active_run_id = run_id
        self._cancel_requested_for = None
        self._has_status = True
        self.label.setText(str(message))
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("Inspecting")
        self.cancel_button.setText("Cancel load")
        self.cancel_button.setEnabled(True)
        self.show()
        return True

    def update_progress(self, progress: SourceLoadProgress) -> bool:
        """Apply only an update owned by the currently active generation."""

        if progress.run_id != self._active_run_id:
            return False
        if self._cancel_requested_for == progress.run_id:
            return False
        phase_label = self._PHASE_LABELS[progress.phase]
        message = str(progress.message).strip()
        item_suffix = (
            f" ({progress.item_index}/{progress.item_total})"
            if progress.item_total > 1
            else ""
        )
        self.label.setText(f"{phase_label}{item_suffix}. {message}".strip())
        self._set_progress_bar(progress, phase_label)
        self._has_status = True
        self.show()
        return True

    def finish(
        self,
        run_id: int,
        *,
        cancelled: bool = False,
        error: str = "",
    ) -> bool:
        """Finish the active generation; stale terminal signals are ignored."""

        run_id = int(run_id)
        if run_id != self._active_run_id:
            return False
        self._active_run_id = None
        self.cancel_button.setEnabled(False)
        if cancelled:
            self.label.setText("Source load cancelled at a safe checkpoint.")
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Cancelled")
        elif error:
            self.label.setText(f"Source load failed: {error}")
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Failed")
        else:
            self.label.setText("Source loaded and verified.")
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(1)
            self.progress_bar.setFormat("Complete")
        self._has_status = True
        self.show()
        return True

    def clear(self) -> None:
        """Hide presentation state without making an old generation current."""

        self._active_run_id = None
        self._cancel_requested_for = None
        self._has_status = False
        self.label.clear()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.cancel_button.setEnabled(False)
        self.hide()

    def _request_cancel(self) -> None:
        run_id = self._active_run_id
        if run_id is None or self._cancel_requested_for == run_id:
            return
        self._cancel_requested_for = run_id
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Cancelling…")
        self.label.setText(
            "Cancelling source load at the next safe I/O checkpoint…"
        )
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("Cancelling")
        self.cancelRequested.emit(run_id)

    def _set_progress_bar(
        self,
        progress: SourceLoadProgress,
        phase_label: str,
    ) -> None:
        if not progress.determinate:
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat(phase_label)
            return
        total = max(int(progress.total), 1)
        current = min(max(int(progress.current), 0), total)
        maximum = min(total, self._QT_PROGRESS_MAXIMUM)
        scaled = int(round((current / total) * maximum))
        self.progress_bar.setRange(0, maximum)
        self.progress_bar.setValue(scaled)
        if progress.unit is SourceLoadProgressUnit.BYTES:
            self.progress_bar.setFormat(
                f"{_human_bytes(current)} / {_human_bytes(total)}"
            )
        else:
            self.progress_bar.setFormat(f"{current} / {total}")


class ImageSourceControl(QWidget):
    """Source selector for explicit graph input nodes."""

    valueChanged = Signal(object)
    sourceLoadCancelRequested = Signal(int)
    viewerDisplayChanged = Signal(str)
    previewReloadRequested = Signal()

    def __init__(
        self,
        value: dict | None,
        *,
        layer_names: list[str],
        sample_names: list[str],
        series_options: list[tuple[int, str]] | None = None,
        source_summary: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["napari layer", "file path", "sample"])
        self.layer_combo = QComboBox()
        self.sample_combo = QComboBox()
        self.path_edit = QLineEdit()
        self.path_button = QPushButton("File...")
        self.path_button.setMaximumWidth(64)
        self.zarr_button = QPushButton("Zarr...")
        self.zarr_button.setMaximumWidth(64)
        self.series_combo = QComboBox()
        self.binding_combo = QComboBox()
        self.binding_combo.addItems(["single item", "collection"])
        self.axis_control = AxisInterpretationControl(
            allow_automatic=False,
            save_target="workflow",
        )
        self.source_summary = QLabel()
        self.source_summary.setWordWrap(True)
        self.source_summary.setStyleSheet("color: #94a3b8;")
        self.source_load_status = SourceLoadStatusControl()
        self.source_load_status.cancelRequested.connect(
            self.sourceLoadCancelRequested.emit
        )
        self._resolution_presentation = ImageSourceResolutionPresentation()
        self.resolution_panel = QWidget()
        resolution_layout = QFormLayout(self.resolution_panel)
        resolution_layout.setContentsMargins(0, 0, 0, 0)
        resolution_layout.setSpacing(4)
        self.analysis_resolution_label = QLabel()
        self.analysis_resolution_label.setWordWrap(True)
        self.pyramid_levels_label = QLabel()
        self.pyramid_levels_label.setWordWrap(True)
        self.pyramid_levels_label.setStyleSheet("color: #94a3b8;")
        self.preview_resolution_label = QLabel()
        self.preview_resolution_label.setWordWrap(True)
        self.viewer_display_combo = QComboBox()
        self.viewer_display_combo.setToolTip(
            "Choose what napari displays. Every preview choice is presentation "
            "only; processing and export always use analysis level 0."
        )
        self.preview_reload_button = QPushButton("Try loading preview again")
        self.preview_reload_button.setToolTip(
            "Retry the lower-resolution presentation layer. Scientific "
            "analysis remains unchanged."
        )
        resolution_layout.addRow("Analysis", self.analysis_resolution_label)
        resolution_layout.addRow("Pyramid", self.pyramid_levels_label)
        resolution_layout.addRow("Preview", self.preview_resolution_label)
        resolution_layout.addRow("Show in napari", self.viewer_display_combo)
        resolution_layout.addRow(self.preview_reload_button)
        self.resolution_panel.setToolTip(
            "The scientific graph always reads level 0. A lower level may be "
            "shown in napari for presentation only."
        )

        self.layer_row = QWidget()
        layer_layout = QHBoxLayout(self.layer_row)
        layer_layout.setContentsMargins(0, 0, 0, 0)
        layer_layout.addWidget(self.layer_combo, 1)

        self.file_row = QWidget()
        file_layout = QHBoxLayout(self.file_row)
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.addWidget(self.path_edit, 1)
        file_layout.addWidget(self.path_button)
        file_layout.addWidget(self.zarr_button)

        self.sample_row = QWidget()
        sample_layout = QHBoxLayout(self.sample_row)
        sample_layout.setContentsMargins(0, 0, 0, 0)
        sample_layout.addWidget(self.sample_combo, 1)

        self.series_row = QWidget()
        series_layout = QHBoxLayout(self.series_row)
        series_layout.setContentsMargins(0, 0, 0, 0)
        series_layout.addWidget(self.series_combo, 1)

        self.form_layout = QFormLayout(self)
        self.form_layout.setContentsMargins(0, 0, 0, 0)
        self.form_layout.addRow("Source", self.mode_combo)
        self.form_layout.addRow("Layer", self.layer_row)
        self.form_layout.addRow("File", self.file_row)
        self.form_layout.addRow("Series / image", self.series_row)
        self.form_layout.addRow("Binding", self.binding_combo)
        self.form_layout.addRow("Image stack", self.axis_control)
        self.form_layout.addRow("Sample", self.sample_row)
        self.form_layout.addRow(self.source_summary)
        self.form_layout.addRow("Resolution", self.resolution_panel)
        self.form_layout.addRow(self.source_load_status)

        self.set_options(
            layer_names,
            sample_names,
            series_options=series_options,
            source_summary=source_summary,
            value=value,
            emit=False,
        )

        self.mode_combo.currentTextChanged.connect(self._on_changed)
        self.layer_combo.currentTextChanged.connect(self._on_changed)
        self.sample_combo.currentTextChanged.connect(self._on_changed)
        self.path_edit.textChanged.connect(self._on_changed)
        self.series_combo.currentIndexChanged.connect(self._on_changed)
        self.binding_combo.currentTextChanged.connect(self._on_changed)
        self.axis_control.textChanged.connect(self._on_changed)
        self.path_button.clicked.connect(self._browse_path)
        self.zarr_button.clicked.connect(self._browse_zarr_path)
        self.viewer_display_combo.currentIndexChanged.connect(
            self._on_viewer_display_changed
        )
        self.preview_reload_button.clicked.connect(
            self.previewReloadRequested.emit
        )

    def value(self) -> dict[str, object]:
        return {
            "source_mode": self.mode_combo.currentText(),
            "layer_name": self.layer_combo.currentText(),
            "file_path": self.path_edit.text(),
            "sample_name": self.sample_combo.currentText(),
            "series_index": int(self.series_combo.currentData() or 0),
            "binding_mode": self.binding_combo.currentText(),
            "axis_declaration": self.axis_control.text(),
        }

    def set_options(
        self,
        layer_names: list[str],
        sample_names: list[str],
        *,
        series_options: list[tuple[int, str]] | None = None,
        source_summary: str = "",
        value: dict | None = None,
        emit: bool = False,
    ) -> None:
        current = value or self.value()
        self._set_combo_items(
            self.layer_combo,
            layer_names,
            str(current.get("layer_name", "")),
        )
        self._set_combo_items(
            self.sample_combo,
            sample_names,
            str(current.get("sample_name", "")),
        )
        mode = str(current.get("source_mode", "napari layer"))
        if self.mode_combo.findText(mode) < 0:
            mode = "napari layer"
        with QSignalBlocker(self.mode_combo), QSignalBlocker(self.path_edit):
            self.mode_combo.setCurrentText(mode)
            self.path_edit.setText(str(current.get("file_path", "")))
        self._set_series_items(
            series_options or [],
            int(current.get("series_index", 0) or 0),
        )
        binding = str(current.get("binding_mode", "single item"))
        if self.binding_combo.findText(binding) < 0:
            binding = "single item"
        with QSignalBlocker(self.binding_combo):
            self.binding_combo.setCurrentText(binding)
        with QSignalBlocker(self.axis_control):
            self.axis_control.setText(str(current.get("axis_declaration", "")))
        self.source_summary.setText(source_summary)
        self._sync_rows()
        if emit:
            self.valueChanged.emit(self.value())

    def _set_combo_items(
        self,
        combo: QComboBox,
        values: list[str],
        current: str,
    ) -> None:
        with QSignalBlocker(combo):
            combo.clear()
            combo.addItem("")
            for value in values:
                combo.addItem(value)
            if current:
                index = combo.findText(current)
                if index < 0:
                    combo.addItem(current)
                    index = combo.findText(current)
                combo.setCurrentIndex(index)

    def _set_series_items(
        self,
        values: list[tuple[int, str]],
        current: int,
    ) -> None:
        with QSignalBlocker(self.series_combo):
            self.series_combo.clear()
            if not values:
                self.series_combo.addItem("Series 1", 0)
            else:
                for index, label in values:
                    self.series_combo.addItem(label, index)
            selected = self.series_combo.findData(current)
            self.series_combo.setCurrentIndex(max(selected, 0))

    def _browse_path(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Select image source",
            recent_paths.initial_directory(
                recent_paths.INPUT_DIRECTORY,
                self.path_edit.text(),
            ),
            "Images and arrays (*.ome.tif *.ome.tiff *.tif *.tiff *.png *.jpg "
            "*.jpeg *.jpe *.jfif *.bmp *.dib *.gif *.webp *.tga *.pbm *.pgm "
            "*.ppm *.pnm *.npy *.npz *.nd2 *.czi *.ims *.lsm *.lif *.lof *.xlif "
            "*.oir *.oib *.oif *.vsi);;"
            f"{MICROSCOPE_FILE_FILTER};;"
            "All files (*.*)",
        )
        if path:
            recent_paths.remember_file_directory(
                recent_paths.INPUT_DIRECTORY,
                path,
            )
            self.path_edit.setText(path)

    def _browse_zarr_path(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select OME-Zarr source",
            recent_paths.initial_directory(
                recent_paths.INPUT_DIRECTORY,
                self.path_edit.text(),
            ),
        )
        if path:
            recent_paths.remember_directory(
                recent_paths.INPUT_DIRECTORY,
                path,
            )
            self.path_edit.setText(path)

    def _on_changed(self, *_args) -> None:
        self._sync_rows()
        self.valueChanged.emit(self.value())

    def begin_source_load(
        self,
        run_id: int,
        message: str = "Inspecting source.",
    ) -> bool:
        """Begin progress for a new source-load generation."""

        accepted = self.source_load_status.begin(run_id, message)
        self._sync_rows()
        return accepted

    def update_source_load_progress(self, progress: SourceLoadProgress) -> bool:
        """Apply a typed progress event if it belongs to the active load."""

        accepted = self.source_load_status.update_progress(progress)
        self._sync_rows()
        return accepted

    def finish_source_load(
        self,
        run_id: int,
        *,
        cancelled: bool = False,
        error: str = "",
    ) -> bool:
        """Apply a terminal state if it belongs to the active load."""

        accepted = self.source_load_status.finish(
            run_id,
            cancelled=cancelled,
            error=error,
        )
        self._sync_rows()
        return accepted

    def clear_source_load_status(self) -> None:
        self.source_load_status.clear()
        self._sync_rows()

    def set_resolution_presentation(
        self,
        presentation: ImageSourceResolutionPresentation,
    ) -> None:
        """Show derived resolution facts without changing ``value()``."""

        if not isinstance(presentation, ImageSourceResolutionPresentation):
            raise TypeError(
                "presentation must be an ImageSourceResolutionPresentation"
            )
        self._resolution_presentation = presentation
        shape = _shape_text(presentation.analysis_shape)
        axes = str(presentation.analysis_axes).strip()
        axis_prefix = f"{axes} " if axes else ""
        self.analysis_resolution_label.setText(
            f"Level 0 · {axis_prefix}{shape} · fixed scientific analysis"
        )
        self.pyramid_levels_label.setText(
            "; ".join(
                f"L{index} {_shape_text(level_shape)}"
                for index, level_shape in enumerate(presentation.level_shapes)
            )
        )
        viewer_choice = str(presentation.viewer_choice).strip().casefold()
        requested_level = (
            viewer_choice.partition(":")[2]
            if viewer_choice.startswith("preview:")
            and viewer_choice != "preview:auto"
            else ""
        )
        detail = str(presentation.preview_detail).strip()
        if presentation.preview_state == "ready":
            preview = (
                f"Level {int(presentation.preview_level or 0)} · "
                f"{_shape_text(presentation.preview_shape)} · presentation only"
            )
        elif presentation.preview_state == "loading":
            preview = (
                f"Loading requested level {requested_level} preview…"
                if requested_level
                else "Loading automatic lower-resolution preview…"
            )
        elif presentation.preview_state == "failed":
            preview = "Preview unavailable; level-0 analysis is unaffected."
        else:
            preview = (
                f"Level {requested_level} not loaded yet · presentation only"
                if requested_level
                else "Not loaded yet · automatic and presentation only"
            )
        if detail:
            preview = f"{preview} {detail}"
        self.preview_resolution_label.setText(preview)
        with QSignalBlocker(self.viewer_display_combo):
            self.viewer_display_combo.clear()
            self.viewer_display_combo.addItem(
                f"Analysis output — L0 · {_shape_text(presentation.analysis_shape)}",
                "analysis",
            )
            if presentation.can_select_preview:
                self.viewer_display_combo.addItem(
                    "Presentation preview — Auto (best fit)",
                    "preview:auto",
                )
                for level, level_shape in enumerate(
                    presentation.level_shapes[1:],
                    start=1,
                ):
                    self.viewer_display_combo.addItem(
                        f"Presentation preview — L{level} · "
                        f"{_shape_text(level_shape)}",
                        f"preview:{level}",
                    )
            selected = self.viewer_display_combo.findData(viewer_choice)
            self.viewer_display_combo.setCurrentIndex(max(selected, 0))
        self.preview_reload_button.setEnabled(bool(presentation.can_retry))
        self.preview_reload_button.setVisible(
            presentation.preview_state == "failed" and presentation.can_retry
        )
        self._sync_rows()

    def _on_viewer_display_changed(self, _index: int) -> None:
        choice = str(self.viewer_display_combo.currentData() or "analysis")
        self.viewerDisplayChanged.emit(choice)

    def _sync_rows(self) -> None:
        mode = self.mode_combo.currentText()
        file_mode = mode == "file path"
        self._set_form_row_visible(self.layer_row, mode == "napari layer")
        self._set_form_row_visible(self.file_row, file_mode)
        self._set_form_row_visible(
            self.series_row,
            file_mode and self.series_combo.count() > 1,
        )
        self._set_form_row_visible(self.binding_combo, file_mode)
        self._set_form_row_visible(
            self.source_summary,
            file_mode and bool(self.source_summary.text()),
        )
        self._set_form_row_visible(
            self.resolution_panel,
            file_mode and self._resolution_presentation.visible
        )
        self._set_form_row_visible(
            self.source_load_status,
            file_mode and self.source_load_status.has_status,
        )
        self._set_form_row_visible(self.sample_row, mode == "sample")

    def _set_form_row_visible(self, field: QWidget, visible: bool) -> None:
        """Hide both a dynamic field and its QFormLayout label."""

        if hasattr(self.form_layout, "setRowVisible"):
            self.form_layout.setRowVisible(field, bool(visible))
            return
        field.setVisible(bool(visible))
        label = self.form_layout.labelForField(field)
        if label is not None:
            label.setVisible(bool(visible))


def _human_bytes(value: int) -> str:
    amount = float(max(int(value), 0))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(amount)} {unit}"
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def _shape_text(shape: tuple[int, ...]) -> str:
    return " × ".join(str(int(size)) for size in shape)


__all__ = [
    "BoolControl",
    "ChoiceControl",
    "FlexibleDoubleSpinBox",
    "ImageSourceControl",
    "ImageSourceResolutionPresentation",
    "NumericEntryControl",
    "ParameterBounds",
    "ParameterControl",
    "ResettableSpinBox",
    "SourceLoadStatusControl",
    "TextControl",
    "_configure_numeric_spin_box",
    "_slider_safe_bounds",
]
