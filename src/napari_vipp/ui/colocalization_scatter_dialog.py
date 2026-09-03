"""Resizable colocalization scatter inspection and image export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from qtpy.QtCore import QPoint, QSignalBlocker, Qt, QTimer, Signal
from qtpy.QtGui import QImage, QPainter
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.ui.controls import _configure_numeric_spin_box
from napari_vipp.ui.palette_roles import theme_colors
from napari_vipp.ui.plots import (
    COLOCALIZATION_SCATTER_COLORMAPS,
    COLOCALIZATION_SCATTER_DISPLAY_MAX_BINS,
    COLOCALIZATION_SCATTER_INSPECTOR_MAX_BINS,
    ColocalizationScatterPlot,
    cap_colocalization_scatter_density_for_display,
)


class ColocalizationScatterDialog(QDialog):
    """Large interactive view of a prepared scatter-density histogram.

    The dialog deliberately owns presentation state, not a workflow or source
    arrays.  Threshold changes are emitted to the caller so the same
    asynchronous exact-count path used by the inspector can remain
    authoritative.  While that result is pending, a histogram-derived count
    is shown immediately and is clearly labelled as an estimate.
    """

    thresholdChanged = Signal(int, float)
    colormapChanged = Signal(str)
    densitySettingsChanged = Signal(int, float)
    logDensityChanged = Signal(bool)
    exportCompleted = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Colocalization Scatter Plot")
        self.setSizeGripEnabled(True)
        self.resize(980, 900)

        self.plot = ColocalizationScatterPlot(
            self,
            max_density_bins=COLOCALIZATION_SCATTER_INSPECTOR_MAX_BINS,
        )
        self.plot.setMinimumSize(360, 360)
        self.colormap_combo = QComboBox(self)
        self.colormap_combo.setAccessibleName("Scatter colormap")
        self.colormap_combo.addItems(COLOCALIZATION_SCATTER_COLORMAPS)
        self.density_bins_spin = QSpinBox(self)
        self.density_bins_spin.setAccessibleName("Density bins per axis")
        self.density_bins_spin.setRange(
            32,
            COLOCALIZATION_SCATTER_INSPECTOR_MAX_BINS,
        )
        self.density_bins_spin.setSingleStep(16)
        self.density_bins_spin.setKeyboardTracking(False)
        self.density_bins_spin.setValue(255)
        self.density_bins_spin.setToolTip(
            "Number of bins along each density axis. This re-bins the pop-out "
            "in the background without recalculating or invalidating the workflow. "
            f"The interactive view supports up to "
            f"{COLOCALIZATION_SCATTER_INSPECTOR_MAX_BINS:,} bins per axis."
        )
        self.populated_range_spin = QDoubleSpinBox(self)
        self.populated_range_spin.setAccessibleName("Populated range percentile")
        self.populated_range_spin.setRange(50.0, 100.0)
        self.populated_range_spin.setSingleStep(0.1)
        self.populated_range_spin.setDecimals(1)
        self.populated_range_spin.setSuffix(" %")
        self.populated_range_spin.setKeyboardTracking(False)
        self.populated_range_spin.setValue(100.0)
        self.populated_range_spin.setToolTip(
            "Defines the central population range used when zooming. 100% uses "
            "the exact ROI minimum and maximum; lower values symmetrically omit "
            "sparse tails from the displayed density only. Exact metrics and "
            "threshold counts always use the full ROI."
        )
        self.log_density_checkbox = QCheckBox("Log density", self)
        self.log_density_checkbox.setChecked(True)
        self.log_density_checkbox.setToolTip(
            "Use logarithmic color scaling so sparse and dense populations can "
            "remain visible together. This only changes the rendering."
        )
        self.export_size_spin = QSpinBox(self)
        self.export_size_spin.setAccessibleName("Scatter export size")
        self.export_size_spin.setRange(64, 4_096)
        self.export_size_spin.setSingleStep(64)
        self.export_size_spin.setSuffix(" px")
        self.export_size_spin.setKeyboardTracking(False)
        self.export_size_spin.setValue(1_024)
        self.export_size_spin.setToolTip(
            "Width and height of the exported square PNG or TIFF. This does not "
            "resize the live window."
        )
        for spinbox in (
            self.density_bins_spin,
            self.populated_range_spin,
            self.export_size_spin,
        ):
            _configure_numeric_spin_box(spinbox)
        self.zoom_to_data_checkbox = QCheckBox("Zoom to populated data", self)
        self.zoom_to_data_checkbox.setChecked(False)
        self.zoom_to_data_checkbox.setToolTip(
            "Switch from the full native ROI extent to the central range selected "
            "by Populated range percentile. At 100%, the two ranges can coincide. "
            "Exact metrics and counts are unaffected."
        )
        self.equal_axes_checkbox = QCheckBox("Equal axis scales", self)
        self.equal_axes_checkbox.setChecked(True)
        self.equal_axes_checkbox.setToolTip(
            "Use the same numeric span on both square axes so equal intensity "
            "differences have the same visual length and slopes are not distorted. "
            "When zoomed, each axis may keep different bounds around its population."
        )
        settings_group = QGroupBox("Scatter display", self)
        settings_layout = QGridLayout(settings_group)
        settings_layout.addWidget(QLabel("Colormap", self), 0, 0)
        settings_layout.addWidget(self.colormap_combo, 0, 1)
        settings_layout.addWidget(QLabel("Density bins per axis", self), 0, 2)
        settings_layout.addWidget(self.density_bins_spin, 0, 3)
        settings_layout.addWidget(self.log_density_checkbox, 0, 4)
        settings_layout.addWidget(
            QLabel("Populated range percentile", self),
            1,
            0,
        )
        settings_layout.addWidget(self.populated_range_spin, 1, 1)
        settings_layout.addWidget(self.zoom_to_data_checkbox, 1, 2)
        settings_layout.addWidget(self.equal_axes_checkbox, 1, 3)
        settings_layout.addWidget(QLabel("Export size", self), 2, 0)
        settings_layout.addWidget(self.export_size_spin, 2, 1)
        self.axis_range_label = QLabel("View ranges unavailable.", self)
        self.axis_range_label.setWordWrap(True)
        self.axis_range_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        settings_layout.addWidget(self.axis_range_label, 2, 2, 1, 3)
        settings_layout.setColumnStretch(4, 1)
        self.summary_label = QLabel("No scatter density is available.", self)
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.export_hint = QLabel(
            "Export uses the selected square output size, independently of the "
            "live window size.",
            self,
        )
        self.export_hint.setWordWrap(True)

        self.export_button = QPushButton("Export PNG/TIFF...", self)
        self.close_button = QPushButton("Close", self)
        for button in (self.export_button, self.close_button):
            # Enter belongs to an active numeric editor. Neither action should
            # silently become QDialog's default while a value is being typed.
            button.setAutoDefault(False)
            button.setDefault(False)
        button_row = QHBoxLayout()
        button_row.addWidget(self.export_hint, 1)
        button_row.addWidget(self.export_button)
        button_row.addWidget(self.close_button)

        layout = QVBoxLayout(self)
        layout.addWidget(settings_group)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.summary_label)
        layout.addLayout(button_row)

        self._density_counts: np.ndarray | None = None
        self._estimate_density_counts: np.ndarray | None = None
        self._density_suffix_counts: np.ndarray | None = None
        self._bin_centers_1: np.ndarray | None = None
        self._bin_centers_2: np.ndarray | None = None
        self._visible_density_voxel_count = 0
        self._threshold_1 = 25.0
        self._threshold_2 = 25.0
        self._intensity_min = 0.0
        self._intensity_max = 255.0
        self._channel_1_min = 0.0
        self._channel_1_max = 255.0
        self._channel_2_min = 0.0
        self._channel_2_max = 255.0
        self._roi_voxels = 0
        self._colocalized_voxels = 0
        self._range_percentile = 100.0
        self._display_note = ""
        self._threshold_layout_lock: tuple[
            tuple[QWidget, int, int], ...
        ] = ()
        self._threshold_layout_was_enabled: bool | None = None
        self._density_settings_timer = QTimer(self)
        self._density_settings_timer.setSingleShot(True)
        self._density_settings_timer.setInterval(275)

        self.plot.thresholdChanged.connect(self._on_threshold_previewed)
        self.plot.thresholdCommitted.connect(self._on_threshold_changed)
        self.plot.gestureStarted.connect(self._begin_threshold_gesture)
        self.plot.gestureFinished.connect(self._end_threshold_gesture)
        self.colormap_combo.currentTextChanged.connect(
            self.colormapChanged.emit
        )
        self.density_bins_spin.valueChanged.connect(
            self._queue_density_settings_change
        )
        self.populated_range_spin.valueChanged.connect(
            self._queue_density_settings_change
        )
        self._density_settings_timer.timeout.connect(
            self._emit_density_settings_change
        )
        self.log_density_checkbox.toggled.connect(self._on_log_density_changed)
        self.zoom_to_data_checkbox.toggled.connect(self._on_zoom_changed)
        self.equal_axes_checkbox.toggled.connect(self._on_equal_axes_changed)
        self.export_button.clicked.connect(self.request_export)
        self.close_button.clicked.connect(self.close)

    @property
    def density_bins(self) -> int:
        return int(self.density_bins_spin.value())

    @property
    def populated_range_percentile(self) -> float:
        return float(self.populated_range_spin.value())

    @property
    def log_density(self) -> bool:
        return bool(self.log_density_checkbox.isChecked())

    @property
    def export_size(self) -> int:
        return int(self.export_size_spin.value())

    def configure_visualization(
        self,
        *,
        density_bins: int,
        range_percentile: float,
        log_counts: bool,
        export_size: int,
    ) -> None:
        """Initialize pop-out-only settings without launching calculations."""

        self._density_settings_timer.stop()
        controls = (
            self.density_bins_spin,
            self.populated_range_spin,
            self.log_density_checkbox,
            self.export_size_spin,
        )
        blockers = [QSignalBlocker(control) for control in controls]
        try:
            self.density_bins_spin.setValue(int(density_bins))
            self.populated_range_spin.setValue(float(range_percentile))
            self.log_density_checkbox.setChecked(bool(log_counts))
            self.export_size_spin.setValue(int(export_size))
        finally:
            del blockers
        self.plot.set_log_counts(bool(log_counts))
        self._update_axis_range_label()

    def set_density_pending(self) -> None:
        """Keep the current plot interactive while a new density is prepared."""

        self.summary_label.setText(
            f"Re-binning the pop-out at {self.density_bins:,} × "
            f"{self.density_bins:,} bins over the central "
            f"{self.populated_range_percentile:g}% range..."
        )

    def _queue_density_settings_change(self, _value) -> None:
        self._density_settings_timer.start()

    def _emit_density_settings_change(self) -> None:
        self.densitySettingsChanged.emit(
            self.density_bins,
            self.populated_range_percentile,
        )

    def _on_log_density_changed(self, enabled: bool) -> None:
        self.plot.set_log_counts(bool(enabled))
        self.logDensityChanged.emit(bool(enabled))

    def _on_zoom_changed(self, enabled: bool) -> None:
        self.plot.set_zoom_to_data(bool(enabled))
        self._update_axis_range_label()

    def _on_equal_axes_changed(self, enabled: bool) -> None:
        self.plot.set_equal_axes(bool(enabled))
        self._update_axis_range_label()

    def _update_axis_range_label(self) -> None:
        plot = self.plot
        ranges = (
            plot._display_channel_1_min,
            plot._display_channel_1_max,
            plot._display_channel_2_min,
            plot._display_channel_2_max,
        )
        if not all(np.isfinite(value) for value in ranges):
            self.axis_range_label.setText("View ranges unavailable.")
            return
        mode = (
            "Populated view"
            if self.zoom_to_data_checkbox.isChecked()
            else "Full view"
        )
        self.axis_range_label.setText(
            f"{mode}: Ch 1 {_format_range(ranges[0], ranges[1])} · "
            f"Ch 2 {_format_range(ranges[2], ranges[3])}"
        )

    def set_density(
        self,
        density_counts: np.ndarray,
        *,
        estimate_density_counts: np.ndarray | None = None,
        threshold_1: float,
        threshold_2: float,
        intensity_min: float,
        intensity_max: float,
        channel_1_range: tuple[float, float] | None = None,
        channel_2_range: tuple[float, float] | None = None,
        full_channel_1_range: tuple[float, float] | None = None,
        full_channel_2_range: tuple[float, float] | None = None,
        roi_voxels: int,
        colocalized_voxels: int,
        range_percentile: float = 100.0,
        channel_1_color: object = "Red",
        channel_2_color: object = "Green",
        colormap: str = "Viridis",
        log_counts: bool = True,
        display_note: str = "",
    ) -> None:
        """Replace the density and authoritative counts displayed in the dialog."""
        density = np.asarray(density_counts)
        if density.ndim != 2 or density.size == 0:
            raise ValueError("Scatter density must be a non-empty 2-D array.")
        if not np.isfinite(density).all() or np.any(density < 0):
            raise ValueError("Scatter density must contain finite non-negative counts.")

        self._density_counts = density.view()
        self._density_counts.flags.writeable = False
        estimate_density = (
            cap_colocalization_scatter_density_for_display(density)
            if estimate_density_counts is None
            else np.asarray(estimate_density_counts)
        )
        if estimate_density.ndim != 2 or estimate_density.size == 0:
            raise ValueError("Scatter estimate density must be a non-empty 2-D array.")
        if (
            estimate_density.shape[0] > COLOCALIZATION_SCATTER_DISPLAY_MAX_BINS
            or estimate_density.shape[1] > COLOCALIZATION_SCATTER_DISPLAY_MAX_BINS
        ):
            estimate_density = cap_colocalization_scatter_density_for_display(
                estimate_density
            )
        self._estimate_density_counts = estimate_density.view()
        self._estimate_density_counts.flags.writeable = False
        with QSignalBlocker(self.density_bins_spin):
            self.density_bins_spin.setValue(
                min(max(int(density.shape[0]), 32), self.density_bins_spin.maximum())
            )
        self._threshold_1 = float(threshold_1)
        self._threshold_2 = float(threshold_2)
        shared_range = (float(intensity_min), float(intensity_max))
        self._channel_1_min, self._channel_1_max = _validated_axis_range(
            channel_1_range or shared_range,
            label="Channel 1",
        )
        self._channel_2_min, self._channel_2_max = _validated_axis_range(
            channel_2_range or shared_range,
            label="Channel 2",
        )
        full_channel_1_range = full_channel_1_range or (
            self._channel_1_min,
            self._channel_1_max,
        )
        full_channel_2_range = full_channel_2_range or (
            self._channel_2_min,
            self._channel_2_max,
        )
        self._intensity_min = min(self._channel_1_min, self._channel_2_min)
        self._intensity_max = max(self._channel_1_max, self._channel_2_max)
        self._prepare_density_estimate_cache()
        self._roi_voxels = max(int(roi_voxels), 0)
        self._colocalized_voxels = max(int(colocalized_voxels), 0)
        self._range_percentile = float(range_percentile)
        self._display_note = str(display_note).strip()
        if not np.isfinite(self._range_percentile) or not (
            0.0 < self._range_percentile <= 100.0
        ):
            raise ValueError("Scatter range percentile must satisfy 0 < p <= 100.")
        resolved_colormap = _resolved_colormap(colormap)
        with QSignalBlocker(self.colormap_combo):
            self.colormap_combo.setCurrentText(resolved_colormap)
        with QSignalBlocker(self.density_bins_spin):
            self.density_bins_spin.setValue(int(density.shape[0]))
        with QSignalBlocker(self.populated_range_spin):
            self.populated_range_spin.setValue(self._range_percentile)
        with QSignalBlocker(self.log_density_checkbox):
            self.log_density_checkbox.setChecked(bool(log_counts))
        summary = self._summary_with_note(self._exact_summary())
        self.plot.set_density(
            self._density_counts,
            threshold_1=self._threshold_1,
            threshold_2=self._threshold_2,
            intensity_min=self._intensity_min,
            intensity_max=self._intensity_max,
            channel_1_range=(self._channel_1_min, self._channel_1_max),
            channel_2_range=(self._channel_2_min, self._channel_2_max),
            full_channel_1_range=full_channel_1_range,
            full_channel_2_range=full_channel_2_range,
            channel_1_color=channel_1_color,
            channel_2_color=channel_2_color,
            colormap=resolved_colormap,
            log_counts=log_counts,
            summary=summary,
        )
        self.summary_label.setText(summary)
        self.plot.setToolTip(self._display_note)
        self._update_axis_range_label()

    def set_exact_counts(
        self,
        *,
        threshold_1: float,
        threshold_2: float,
        roi_voxels: int,
        colocalized_voxels: int,
    ) -> None:
        """Apply an asynchronous exact-count result without replacing density."""
        self._threshold_1 = float(threshold_1)
        self._threshold_2 = float(threshold_2)
        self._roi_voxels = max(int(roi_voxels), 0)
        self._colocalized_voxels = max(int(colocalized_voxels), 0)
        summary = self._summary_with_note(self._exact_summary())
        self.plot.set_pending_thresholds(
            threshold_1=self._threshold_1,
            threshold_2=self._threshold_2,
            preserve_density=self._density_counts is not None,
            summary=summary,
        )
        self.summary_label.setText(summary)

    def set_colormap(self, colormap: str) -> None:
        """Recolor the current density while keeping scientific state intact."""
        resolved_colormap = _resolved_colormap(colormap)
        with QSignalBlocker(self.colormap_combo):
            self.colormap_combo.setCurrentText(resolved_colormap)
        self.plot.set_colormap(resolved_colormap)

    def set_log_density(self, enabled: bool) -> None:
        """Set the presentation transfer function without emitting a user edit."""

        with QSignalBlocker(self.log_density_checkbox):
            self.log_density_checkbox.setChecked(bool(enabled))
        self.plot.set_log_counts(bool(enabled))

    def set_pending_thresholds(
        self,
        *,
        threshold_1: float,
        threshold_2: float,
    ) -> None:
        """Move both guides and show a live density-derived count estimate."""
        self._threshold_1 = float(threshold_1)
        self._threshold_2 = float(threshold_2)
        summary = self._histogram_estimate_summary()
        self.plot.set_pending_thresholds(
            threshold_1=self._threshold_1,
            threshold_2=self._threshold_2,
            preserve_density=self._density_counts is not None,
            summary=summary,
        )
        visible = self._visible_density_voxels()
        dropped = max(self._roi_voxels - visible, 0)
        if self._range_percentile < 100.0 or dropped:
            scope = (
                f" Visible density includes {visible:,}/{self._roi_voxels:,} ROI "
                f"voxels; {dropped:,} tails are hidden by the "
                f"{self._range_percentile:g}% display range."
            )
        else:
            scope = ""
        pending_summary = (
            f"{summary}.{scope} Exact full-ROI counts will update asynchronously; "
            "display clipping does not change metrics."
        )
        self.summary_label.setText(self._summary_with_note(pending_summary))

    def request_export(self) -> None:
        """Prompt for PNG/TIFF output and export the current rendered plot."""
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export colocalization scatter plot",
            "colocalization-scatter.png",
            "PNG image (*.png);;TIFF image (*.tif *.tiff)",
        )
        if not path:
            return
        requested = Path(path)
        if not requested.suffix:
            suffix = ".tif" if "TIFF" in selected_filter else ".png"
            requested = requested.with_suffix(suffix)
        try:
            exported = self.export_image(requested)
        except Exception as exc:
            QMessageBox.critical(self, "Scatter export failed", str(exc))
            return
        self.exportCompleted.emit(str(exported))

    def export_image(self, path: str | Path) -> Path:
        """Export the visible plot at its current display resolution."""
        target = Path(path)
        suffix = target.suffix.casefold()
        if suffix not in {".png", ".tif", ".tiff"}:
            raise ValueError("Scatter export supports PNG, TIF, and TIFF files.")
        target.parent.mkdir(parents=True, exist_ok=True)

        image = render_widget_image(
            self.plot,
            width=self.export_size,
            height=self.export_size,
        )
        if suffix == ".png":
            if not image.save(str(target), "PNG"):
                raise OSError(f"Qt could not write PNG output to {target}.")
        else:
            tifffile.imwrite(
                target,
                qimage_rgb_array(image),
                photometric="rgb",
            )
        return target

    def _on_threshold_changed(self, axis: int, value: float) -> None:
        """Commit one completed guide drag to the owning workflow."""

        self._apply_threshold_preview(axis, value)
        self.thresholdChanged.emit(int(axis), float(value))

    def _on_threshold_previewed(self, axis: int, value: float) -> None:
        """Update the retained plot and estimate without touching the workflow."""

        self._apply_threshold_preview(axis, value)

    def _apply_threshold_preview(self, axis: int, value: float) -> None:
        if int(axis) == 1:
            self._threshold_1 = float(value)
        elif int(axis) == 2:
            self._threshold_2 = float(value)
        else:
            return
        self.set_pending_thresholds(
            threshold_1=self._threshold_1,
            threshold_2=self._threshold_2,
        )

    def _begin_threshold_gesture(self) -> None:
        """Hold the active decision surface still while its summary changes."""

        if self._threshold_layout_lock:
            return
        widgets = (self.plot, self.summary_label)
        self._threshold_layout_lock = tuple(
            (widget, widget.minimumHeight(), widget.maximumHeight())
            for widget in widgets
        )
        layout = self.layout()
        self._threshold_layout_was_enabled = layout.isEnabled()
        layout.setEnabled(False)
        for widget, _minimum, _maximum in self._threshold_layout_lock:
            widget.setFixedHeight(widget.height())

    def _end_threshold_gesture(self) -> None:
        snapshots = self._threshold_layout_lock
        if not snapshots:
            return
        self._threshold_layout_lock = ()
        for widget, minimum, maximum in reversed(snapshots):
            widget.setMaximumHeight(maximum)
            widget.setMinimumHeight(minimum)
            widget.updateGeometry()
        layout = self.layout()
        layout_was_enabled = self._threshold_layout_was_enabled
        self._threshold_layout_was_enabled = None
        if layout_was_enabled is not None:
            layout.setEnabled(layout_was_enabled)

    def _histogram_estimate_summary(self) -> str:
        suffix_counts = self._density_suffix_counts
        centers_1 = self._bin_centers_1
        centers_2 = self._bin_centers_2
        if suffix_counts is None or centers_1 is None or centers_2 is None:
            return "Count pending"
        index_1 = int(np.searchsorted(centers_1, self._threshold_1, side="left"))
        index_2 = int(np.searchsorted(centers_2, self._threshold_2, side="left"))
        if index_1 >= suffix_counts.shape[0] or index_2 >= suffix_counts.shape[1]:
            estimated = 0
        else:
            estimated = int(np.rint(float(suffix_counts[index_1, index_2])))
        return _count_summary(
            "Visible-density estimate",
            estimated,
            self._visible_density_voxel_count,
        )

    def _prepare_density_estimate_cache(self) -> None:
        """Build constant-time threshold-count lookups for live guide previews."""

        density = self._estimate_density_counts
        if density is None:
            self._density_suffix_counts = None
            self._bin_centers_1 = None
            self._bin_centers_2 = None
            self._visible_density_voxel_count = 0
            return

        bins_1, bins_2 = density.shape
        span_1 = self._channel_1_max - self._channel_1_min
        span_2 = self._channel_2_max - self._channel_2_min
        self._bin_centers_1 = self._channel_1_min + (
            (np.arange(bins_1, dtype=np.float64) + 0.5) / bins_1 * span_1
        )
        self._bin_centers_2 = self._channel_2_min + (
            (np.arange(bins_2, dtype=np.float64) + 0.5) / bins_2 * span_2
        )
        reversed_density = density[::-1, ::-1]
        suffix_counts = np.cumsum(reversed_density, axis=0)
        np.cumsum(suffix_counts, axis=1, out=suffix_counts)
        self._density_suffix_counts = suffix_counts[::-1, ::-1]
        self._visible_density_voxel_count = int(
            np.rint(float(self._density_suffix_counts[0, 0]))
        )

    def _exact_summary(self) -> str:
        return _count_summary(
            "Exact",
            self._colocalized_voxels,
            self._roi_voxels,
        )

    def _summary_with_note(self, summary: str) -> str:
        if not self._display_note:
            return summary
        return f"{summary} {self._display_note}"

    def _visible_density_voxels(self) -> int:
        return self._visible_density_voxel_count


def _count_summary(label: str, selected: int, total: int) -> str:
    percentage = f"{100.0 * selected / total:.1f}%" if total else "n/a"
    return f"{label}: {selected:,}/{total:,} ({percentage})"


def _format_range(minimum: float, maximum: float) -> str:
    """Format one visible native-intensity range compactly."""

    return f"{minimum:,.5g}–{maximum:,.5g}"


def _validated_axis_range(
    values: tuple[float, float],
    *,
    label: str,
) -> tuple[float, float]:
    minimum, maximum = (float(value) for value in values)
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError(f"{label} scatter range must be finite.")
    if maximum <= minimum:
        raise ValueError(f"{label} scatter maximum must exceed its minimum.")
    return minimum, maximum


def _resolved_colormap(value: object) -> str:
    requested = str(value or "").strip().casefold()
    return next(
        (
            option
            for option in COLOCALIZATION_SCATTER_COLORMAPS
            if option.casefold() == requested
        ),
        COLOCALIZATION_SCATTER_COLORMAPS[0],
    )


def render_widget_image(
    widget,
    *,
    width: int | None = None,
    height: int | None = None,
) -> QImage:
    """Render one widget to RGB, optionally at a temporary native size."""

    requested_width = max(int(widget.width() if width is None else width), 1)
    requested_height = max(int(widget.height() if height is None else height), 1)
    original_size = widget.size()
    original_minimum = widget.minimumSize()
    original_maximum = widget.maximumSize()
    resize_for_render = (
        requested_width != int(widget.width())
        or requested_height != int(widget.height())
    )
    if resize_for_render:
        widget.setMinimumSize(0, 0)
        widget.setMaximumSize(16_777_215, 16_777_215)
        widget.resize(requested_width, requested_height)
    try:
        image = QImage(requested_width, requested_height, QImage.Format_RGB888)
        image.fill(theme_colors(QWidget.palette(widget)).surface)
        painter = QPainter(image)
        try:
            # PySide6 does not expose the one-argument QPainter overload that PyQt6
            # accepts. Supplying the default offset selects the shared Qt API.
            widget.render(painter, QPoint())
        finally:
            painter.end()
    finally:
        if resize_for_render:
            widget.resize(original_size)
            widget.setMinimumSize(original_minimum)
            widget.setMaximumSize(original_maximum)
    return image


def qimage_rgb_array(image: QImage) -> np.ndarray:
    """Copy a QImage into a binding-independent contiguous RGB array."""
    rgb = image.convertToFormat(QImage.Format_RGB888)
    pointer = rgb.bits()
    byte_count = int(rgb.sizeInBytes())
    if hasattr(pointer, "setsize"):
        pointer.setsize(byte_count)
    raw = np.frombuffer(pointer, dtype=np.uint8, count=byte_count)
    rows = raw.reshape(int(rgb.height()), int(rgb.bytesPerLine()))
    return rows[:, : int(rgb.width()) * 3].reshape(
        int(rgb.height()),
        int(rgb.width()),
        3,
    ).copy()
