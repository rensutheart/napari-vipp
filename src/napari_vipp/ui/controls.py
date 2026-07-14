"""Reusable parameter and image-source controls for the VIPP UI."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from qtpy.QtCore import QLocale, QSignalBlocker, Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QWidget,
)

from napari_vipp.core.io import MICROSCOPE_FILE_FILTER


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
    extent = max(abs(float(minimum)), abs(float(maximum)), 1.0)
    if not logarithmic:
        while decimals > 0 and extent * (10**decimals) > maximum_slider_units:
            decimals -= 1
        if extent > maximum_slider_units:
            minimum = max(float(minimum), -maximum_slider_units)
            maximum = min(float(maximum), maximum_slider_units)

    smallest_step = 1.0 if decimals == 0 else 10 ** (-decimals)
    return ParameterBounds(
        float(minimum),
        float(maximum),
        max(float(step), smallest_step),
        decimals,
        expandable,
        logarithmic,
        entry_minimum,
        entry_maximum,
    )


class FlexibleDoubleSpinBox(QDoubleSpinBox):
    """Locale-independent float entry accepting decimal points and commas."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLocale(QLocale.c())
        self.setKeyboardTracking(False)

    @staticmethod
    def _normalized_text(text: str) -> str:
        return str(text).replace(",", ".")

    def validate(self, text: str, position: int):
        state, _normalized, validated_position = super().validate(
            self._normalized_text(text),
            position,
        )
        return state, text, validated_position

    def valueFromText(self, text: str) -> float:
        return super().valueFromText(self._normalized_text(text))

    def textFromValue(self, value: float) -> str:  # noqa: N802
        decimals = max(int(self.decimals()), 0)
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
            self.value_box = QSpinBox()
        else:
            self.value_box = FlexibleDoubleSpinBox()
            self.value_box.setDecimals(bounds.decimals)
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
                self.slider.setValue(self._to_slider(current))
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
                self.slider.setValue(self._to_slider(current))
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
            self.slider.setValue(self._to_slider(value))
        self.valueChanged.emit(self.value())

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
            self.value_box = QSpinBox()
        else:
            self.value_box = FlexibleDoubleSpinBox()
            self.value_box.setDecimals(bounds.decimals)
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


class ImageSourceControl(QWidget):
    """Source selector for explicit graph input nodes."""

    valueChanged = Signal(object)

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
        self.source_summary = QLabel()
        self.source_summary.setWordWrap(True)
        self.source_summary.setStyleSheet("color: #94a3b8;")

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

        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addRow("Source", self.mode_combo)
        layout.addRow("Layer", self.layer_row)
        layout.addRow("File", self.file_row)
        layout.addRow("Series / image", self.series_row)
        layout.addRow("Binding", self.binding_combo)
        layout.addRow("Sample", self.sample_row)
        layout.addRow(self.source_summary)

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
        self.path_button.clicked.connect(self._browse_path)
        self.zarr_button.clicked.connect(self._browse_zarr_path)

    def value(self) -> dict[str, object]:
        return {
            "source_mode": self.mode_combo.currentText(),
            "layer_name": self.layer_combo.currentText(),
            "file_path": self.path_edit.text(),
            "sample_name": self.sample_combo.currentText(),
            "series_index": int(self.series_combo.currentData() or 0),
            "binding_mode": self.binding_combo.currentText(),
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
            self.path_edit.text(),
            "Images and arrays (*.ome.tif *.ome.tiff *.tif *.tiff *.png *.jpg "
            "*.jpeg *.jpe *.jfif *.bmp *.dib *.gif *.webp *.tga *.pbm *.pgm "
            "*.ppm *.pnm *.npy *.npz *.nd2 *.czi *.lsm *.lif *.lof *.xlif "
            "*.oir *.oib *.oif *.vsi);;"
            f"{MICROSCOPE_FILE_FILTER};;"
            "All files (*.*)",
        )
        if path:
            self.path_edit.setText(path)

    def _browse_zarr_path(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select OME-Zarr source",
            self.path_edit.text(),
        )
        if path:
            self.path_edit.setText(path)

    def _on_changed(self, *_args) -> None:
        self._sync_rows()
        self.valueChanged.emit(self.value())

    def _sync_rows(self) -> None:
        mode = self.mode_combo.currentText()
        self.layer_row.setVisible(mode == "napari layer")
        self.file_row.setVisible(mode == "file path")
        file_mode = mode == "file path"
        self.series_row.setVisible(file_mode and self.series_combo.count() > 1)
        self.binding_combo.setVisible(file_mode)
        self.source_summary.setVisible(file_mode and bool(self.source_summary.text()))
        self.sample_row.setVisible(mode == "sample")


__all__ = [
    "BoolControl",
    "ChoiceControl",
    "FlexibleDoubleSpinBox",
    "ImageSourceControl",
    "NumericEntryControl",
    "ParameterBounds",
    "ParameterControl",
    "TextControl",
    "_configure_numeric_spin_box",
    "_slider_safe_bounds",
]
