"""Resizable colocalization scatter inspection and image export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from qtpy.QtCore import QSignalBlocker, Qt, Signal
from qtpy.QtGui import QColor, QImage, QPainter
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from napari_vipp.ui.plots import (
    COLOCALIZATION_SCATTER_COLORMAPS,
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
    exportCompleted = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Colocalization Scatter Plot")
        self.setSizeGripEnabled(True)
        self.resize(900, 900)

        self.plot = ColocalizationScatterPlot(self)
        self.plot.setMinimumSize(360, 360)
        self.colormap_combo = QComboBox(self)
        self.colormap_combo.setAccessibleName("Scatter colormap")
        self.colormap_combo.addItems(COLOCALIZATION_SCATTER_COLORMAPS)
        colormap_row = QHBoxLayout()
        colormap_row.addWidget(QLabel("Colormap", self))
        colormap_row.addWidget(self.colormap_combo)
        colormap_row.addStretch(1)
        self.summary_label = QLabel("No scatter density is available.", self)
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.export_hint = QLabel(
            "Export uses the plot's current on-screen pixel dimensions.",
            self,
        )
        self.export_hint.setWordWrap(True)

        self.export_button = QPushButton("Export PNG/TIFF...", self)
        self.close_button = QPushButton("Close", self)
        button_row = QHBoxLayout()
        button_row.addWidget(self.export_hint, 1)
        button_row.addWidget(self.export_button)
        button_row.addWidget(self.close_button)

        layout = QVBoxLayout(self)
        layout.addLayout(colormap_row)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.summary_label)
        layout.addLayout(button_row)

        self._density_counts: np.ndarray | None = None
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

        self.plot.thresholdChanged.connect(self._on_threshold_changed)
        self.colormap_combo.currentTextChanged.connect(
            self.colormapChanged.emit
        )
        self.export_button.clicked.connect(self.request_export)
        self.close_button.clicked.connect(self.close)

    def set_density(
        self,
        density_counts: np.ndarray,
        *,
        threshold_1: float,
        threshold_2: float,
        intensity_min: float,
        intensity_max: float,
        channel_1_range: tuple[float, float] | None = None,
        channel_2_range: tuple[float, float] | None = None,
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
        density = cap_colocalization_scatter_density_for_display(
            np.asarray(density_counts)
        )
        if density.ndim != 2 or density.size == 0:
            raise ValueError("Scatter density must be a non-empty 2-D array.")
        if not np.isfinite(density).all() or np.any(density < 0):
            raise ValueError("Scatter density must contain finite non-negative counts.")

        self._density_counts = density.copy()
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
        self._intensity_min = min(self._channel_1_min, self._channel_2_min)
        self._intensity_max = max(self._channel_1_max, self._channel_2_max)
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
        summary = self._summary_with_note(self._exact_summary())
        self.plot.set_density(
            self._density_counts,
            threshold_1=self._threshold_1,
            threshold_2=self._threshold_2,
            intensity_min=self._intensity_min,
            intensity_max=self._intensity_max,
            channel_1_range=(self._channel_1_min, self._channel_1_max),
            channel_2_range=(self._channel_2_min, self._channel_2_max),
            channel_1_color=channel_1_color,
            channel_2_color=channel_2_color,
            colormap=resolved_colormap,
            log_counts=log_counts,
            summary=summary,
        )
        self.summary_label.setText(summary)
        self.plot.setToolTip(self._display_note)

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

        image = render_widget_image(self.plot)
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
        self.thresholdChanged.emit(int(axis), float(value))

    def _histogram_estimate_summary(self) -> str:
        if self._density_counts is None:
            return "Count pending"
        bins_1, bins_2 = self._density_counts.shape
        span_1 = self._channel_1_max - self._channel_1_min
        span_2 = self._channel_2_max - self._channel_2_min
        centers_1 = self._channel_1_min + (
            (np.arange(bins_1, dtype=np.float64) + 0.5) / bins_1 * span_1
        )
        centers_2 = self._channel_2_min + (
            (np.arange(bins_2, dtype=np.float64) + 0.5) / bins_2 * span_2
        )
        selected = self._density_counts[np.ix_(
            centers_1 >= self._threshold_1,
            centers_2 >= self._threshold_2,
        )]
        estimated = int(np.rint(float(np.sum(selected))))
        total = int(np.rint(float(np.sum(self._density_counts))))
        return _count_summary("Visible-density estimate", estimated, total)

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
        if self._density_counts is None:
            return 0
        return int(np.rint(float(np.sum(self._density_counts))))


def _count_summary(label: str, selected: int, total: int) -> str:
    percentage = f"{100.0 * selected / total:.1f}%" if total else "n/a"
    return f"{label}: {selected:,}/{total:,} ({percentage})"


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


def render_widget_image(widget) -> QImage:
    """Render one widget to an RGB image at its current pixel dimensions."""
    width = max(int(widget.width()), 1)
    height = max(int(widget.height()), 1)
    image = QImage(width, height, QImage.Format_RGB888)
    image.fill(QColor("#111827"))
    painter = QPainter(image)
    widget.render(painter)
    painter.end()
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
