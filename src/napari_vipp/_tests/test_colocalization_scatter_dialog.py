from __future__ import annotations

import numpy as np
import pytest
import tifffile
from qtpy.QtCore import QPoint, Qt
from qtpy.QtGui import QColor, QFont, QFontMetrics, QImage, QPalette
from qtpy.QtWidgets import QWidget

from napari_vipp.ui.colocalization_scatter_dialog import (
    ColocalizationScatterDialog,
    qimage_rgb_array,
    render_widget_image,
)
from napari_vipp.ui.palette_roles import theme_colors
from napari_vipp.ui.plots import (
    COLOCALIZATION_SCATTER_COLORMAPS,
    DetailedHistogramPlot,
)


class _TextPaintRecorder:
    """Minimal painter spy that retains the font used for each text call."""

    def __init__(self, font: QFont):
        self._font = QFont(font)
        self._saved_fonts: list[QFont] = []
        self.text_calls: list[tuple[str, QFont]] = []

    def font(self) -> QFont:
        return QFont(self._font)

    def setFont(self, font: QFont) -> None:  # noqa: N802
        self._font = QFont(font)

    def fontMetrics(self) -> QFontMetrics:  # noqa: N802
        return QFontMetrics(self._font)

    def setPen(self, _pen) -> None:  # noqa: N802
        return None

    def drawText(self, *_args) -> None:  # noqa: N802
        text = next(
            (argument for argument in reversed(_args) if isinstance(argument, str)),
            "",
        )
        self.text_calls.append((text, QFont(self._font)))

    def save(self) -> None:
        self._saved_fonts.append(QFont(self._font))

    def restore(self) -> None:
        self._font = self._saved_fonts.pop()

    def translate(self, _x: int, _y: int) -> None:
        return None

    def rotate(self, _angle: int) -> None:
        return None


def _populate_dialog(dialog: ColocalizationScatterDialog) -> None:
    dialog.set_density(
        np.ones((64, 64), dtype=np.float64),
        threshold_1=25.0,
        threshold_2=25.0,
        intensity_min=0.0,
        intensity_max=100.0,
        roi_voxels=4_096,
        colocalized_voxels=2_304,
        channel_1_color="Magenta",
        channel_2_color="Green",
        colormap="Magma",
    )


def _populate_asymmetric_dialog(dialog: ColocalizationScatterDialog) -> np.ndarray:
    density = np.ones((32, 32), dtype=np.float64)
    dialog.set_density(
        density,
        threshold_1=42.0,
        threshold_2=105.0,
        intensity_min=0.0,
        intensity_max=160.0,
        channel_1_range=(40.0, 60.0),
        channel_2_range=(100.0, 160.0),
        roi_voxels=int(density.sum()),
        colocalized_voxels=128,
        colormap="Gray",
    )
    return density


def _display_ranges(dialog: ColocalizationScatterDialog):
    plot = dialog.plot
    return (
        (plot._display_channel_1_min, plot._display_channel_1_max),
        (plot._display_channel_2_min, plot._display_channel_2_max),
    )


def test_scatter_axis_titles_match_histogram_axis_emphasis(qtbot):
    dialog = ColocalizationScatterDialog()
    histogram = DetailedHistogramPlot()
    qtbot.addWidget(dialog)
    qtbot.addWidget(histogram)
    _populate_dialog(dialog)
    plot = dialog.plot
    recorder = _TextPaintRecorder(plot.font())

    plot._draw_labels(
        recorder,
        plot.rect().adjusted(8, 8, -8, -8),
        plot._plot_rect(),
    )

    axis_fonts = {
        text: font
        for text, font in recorder.text_calls
        if text in {"Ch 1 intensity", "Ch 2 intensity"}
    }
    histogram_axis_font = histogram._axis_label_font()
    assert set(axis_fonts) == {"Ch 1 intensity", "Ch 2 intensity"}
    assert {font.weight() for font in axis_fonts.values()} == {
        histogram_axis_font.weight()
    }
    assert {font.pointSizeF() for font in axis_fonts.values()} == {
        histogram_axis_font.pointSizeF()
    }
    assert histogram_axis_font.weight() == QFont.DemiBold

    threshold_fonts = [
        font for text, font in recorder.text_calls if text.startswith(("T1 ", "T2 "))
    ]
    assert threshold_fonts
    assert {font.weight() for font in threshold_fonts} == {plot.font().weight()}


def test_scatter_dialog_defaults_to_zero_based_equal_intensity_axes(qtbot):
    dialog = ColocalizationScatterDialog()
    qtbot.addWidget(dialog)
    density = _populate_asymmetric_dialog(dialog)

    assert not dialog.zoom_to_data_checkbox.isChecked()
    assert dialog.equal_axes_checkbox.isChecked()
    assert _display_ranges(dialog) == ((0.0, 160.0), (0.0, 160.0))
    assert (
        dialog.plot._channel_1_min,
        dialog.plot._channel_1_max,
        dialog.plot._channel_2_min,
        dialog.plot._channel_2_max,
    ) == (40.0, 60.0, 100.0, 160.0)
    np.testing.assert_array_equal(dialog.plot._density_counts, density)

    plot_rect = dialog.plot._plot_rect()
    x_units_per_pixel = 160.0 / plot_rect.width()
    y_units_per_pixel = 160.0 / plot_rect.height()
    assert plot_rect.width() == plot_rect.height()
    assert x_units_per_pixel == y_units_per_pixel


def test_scatter_dialog_zoom_and_equal_axes_are_independent(qtbot):
    dialog = ColocalizationScatterDialog()
    qtbot.addWidget(dialog)
    _populate_asymmetric_dialog(dialog)
    retained_density = dialog.plot._density_counts

    dialog.zoom_to_data_checkbox.setChecked(True)
    assert _display_ranges(dialog) == ((40.0, 160.0), (40.0, 160.0))

    dialog.equal_axes_checkbox.setChecked(False)
    assert _display_ranges(dialog) == ((40.0, 60.0), (100.0, 160.0))
    assert dialog.plot._density_counts is retained_density

    _populate_asymmetric_dialog(dialog)
    assert dialog.zoom_to_data_checkbox.isChecked()
    assert not dialog.equal_axes_checkbox.isChecked()
    assert _display_ranges(dialog) == ((40.0, 60.0), (100.0, 160.0))
    retained_density = dialog.plot._density_counts

    dialog.zoom_to_data_checkbox.setChecked(False)
    assert _display_ranges(dialog) == ((0.0, 60.0), (0.0, 160.0))

    dialog.equal_axes_checkbox.setChecked(True)
    assert _display_ranges(dialog) == ((0.0, 160.0), (0.0, 160.0))
    assert dialog.plot._density_counts is retained_density
    assert (
        dialog.plot._channel_1_min,
        dialog.plot._channel_1_max,
        dialog.plot._channel_2_min,
        dialog.plot._channel_2_max,
    ) == (40.0, 60.0, 100.0, 160.0)


def test_zero_based_scatter_maps_density_into_its_true_source_bounds(qtbot):
    dialog = ColocalizationScatterDialog()
    qtbot.addWidget(dialog)
    _populate_asymmetric_dialog(dialog)
    plot = dialog.plot
    plot.resize(520, 520)
    palette = QPalette(plot.palette())
    palette.setColor(QPalette.Base, QColor("#111827"))
    palette.setColor(QPalette.Text, QColor("#f8fafc"))
    plot.setPalette(palette)

    # Rendering a child of the still-hidden dialog activates its parent layout
    # and may resize the plot.  Settle that layout before deriving probe points.
    render_widget_image(plot)
    plot_rect = plot._plot_rect()
    rendered = render_widget_image(plot)
    populated = rendered.pixelColor(
        plot._x_from_value(50.0, plot_rect),
        plot._y_from_value(130.0, plot_rect),
    )
    empty = rendered.pixelColor(
        plot._x_from_value(20.0, plot_rect),
        plot._y_from_value(80.0, plot_rect),
    )

    assert populated != empty, (
        populated.getRgb(),
        empty.getRgb(),
        plot_rect.getRect(),
        plot._density_target_rect(plot_rect).getRect(),
        (plot.width(), plot.height()),
    )
    assert populated.value() > empty.value()


def test_scatter_dialog_colormap_control_is_shared_and_programmatic_sync_is_quiet(
    qtbot,
):
    dialog = ColocalizationScatterDialog()
    qtbot.addWidget(dialog)
    emitted: list[str] = []
    dialog.colormapChanged.connect(emitted.append)

    assert [
        dialog.colormap_combo.itemText(index)
        for index in range(dialog.colormap_combo.count())
    ] == list(COLOCALIZATION_SCATTER_COLORMAPS)
    assert dialog.colormap_combo.currentText() == "Viridis"

    _populate_dialog(dialog)

    assert emitted == []
    assert dialog.colormap_combo.currentText() == "Magma"
    assert dialog.plot._colormap == "Magma"

    with qtbot.waitSignal(dialog.colormapChanged) as changed:
        dialog.colormap_combo.setCurrentText("Inferno")

    assert changed.args == ["Inferno"]
    assert emitted == ["Inferno"]

    dialog.set_colormap("Plasma")

    assert emitted == ["Inferno"]
    assert dialog.colormap_combo.currentText() == "Plasma"
    assert dialog.plot._colormap == "Plasma"


def test_scatter_dialog_normalizes_unsupported_colormap(qtbot):
    dialog = ColocalizationScatterDialog()
    qtbot.addWidget(dialog)
    _populate_dialog(dialog)

    dialog.set_density(
        np.ones((4, 4), dtype=np.float64),
        threshold_1=1.0,
        threshold_2=2.0,
        intensity_min=0.0,
        intensity_max=10.0,
        roi_voxels=16,
        colocalized_voxels=4,
        colormap="not-a-colormap",
    )

    assert dialog.colormap_combo.currentText() == "Viridis"
    assert dialog.plot._colormap == "Viridis"


def test_scatter_dialog_updates_estimate_immediately_then_exact_count(qtbot):
    dialog = ColocalizationScatterDialog()
    qtbot.addWidget(dialog)
    _populate_dialog(dialog)

    assert "Exact: 2,304/4,096 (56.2%)" in dialog.summary_label.text()
    with qtbot.waitSignal(dialog.thresholdChanged) as emitted:
        dialog._on_threshold_changed(1, 50.0)

    assert emitted.args == [1, 50.0]
    assert dialog.plot._image is not None
    assert "Visible-density estimate" in dialog.summary_label.text()
    assert "asynchronously" in dialog.summary_label.text()

    dialog.set_exact_counts(
        threshold_1=50.0,
        threshold_2=25.0,
        roi_voxels=4_096,
        colocalized_voxels=1_536,
    )

    assert dialog.summary_label.text() == "Exact: 1,536/4,096 (37.5%)"
    assert dialog.plot._summary == "Exact: 1,536/4,096 (37.5%)"


def test_scatter_dialog_threshold_hover_uses_directional_cursors(qtbot):
    dialog = ColocalizationScatterDialog()
    qtbot.addWidget(dialog)
    _populate_dialog(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)
    plot = dialog.plot
    rect = plot._plot_rect()
    vertical = QPoint(
        plot._x_from_value(plot._threshold_1, rect),
        rect.top() + 20,
    )
    horizontal = QPoint(
        rect.right() - 20,
        plot._y_from_value(plot._threshold_2, rect),
    )

    qtbot.mouseMove(plot, pos=vertical)
    assert plot.cursor().shape() == Qt.SizeHorCursor
    qtbot.mouseMove(plot, pos=horizontal)
    assert plot.cursor().shape() == Qt.SizeVerCursor


def test_scatter_dialog_threshold_scrub_keeps_plot_allocation_stable(qtbot):
    dialog = ColocalizationScatterDialog()
    qtbot.addWidget(dialog)
    _populate_dialog(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)
    plot = dialog.plot
    original_constraints = {
        id(widget): (widget.minimumHeight(), widget.maximumHeight())
        for widget in (plot, dialog.summary_label)
    }
    original_geometry = (plot.geometry(), dialog.summary_label.geometry())
    rect = plot._plot_rect()
    vertical = QPoint(
        plot._x_from_value(plot._threshold_1, rect),
        rect.center().y(),
    )

    qtbot.mousePress(plot, Qt.LeftButton, pos=vertical)
    qtbot.mouseMove(plot, pos=QPoint(vertical.x() + 40, vertical.y()))
    qtbot.wait(10)

    assert dialog._threshold_layout_lock
    assert not dialog.layout().isEnabled()
    assert "Visible-density estimate" in dialog.summary_label.text()
    assert (plot.geometry(), dialog.summary_label.geometry()) == original_geometry

    dialog.set_exact_counts(
        threshold_1=plot._threshold_1,
        threshold_2=plot._threshold_2,
        roi_voxels=4_096,
        colocalized_voxels=1_536,
    )
    qtbot.wait(10)

    assert (plot.geometry(), dialog.summary_label.geometry()) == original_geometry

    qtbot.mouseRelease(
        plot,
        Qt.LeftButton,
        pos=QPoint(vertical.x() + 40, vertical.y()),
    )

    assert dialog._threshold_layout_lock == ()
    assert dialog.layout().isEnabled()
    for widget in (plot, dialog.summary_label):
        assert (
            widget.minimumHeight(),
            widget.maximumHeight(),
        ) == original_constraints[id(widget)]


def test_scatter_dialog_estimate_uses_independent_axis_ranges(qtbot):
    dialog = ColocalizationScatterDialog()
    qtbot.addWidget(dialog)
    dialog.set_density(
        np.ones((10, 10), dtype=np.float64),
        threshold_1=5.0,
        threshold_2=550.0,
        intensity_min=0.0,
        intensity_max=600.0,
        channel_1_range=(0.0, 10.0),
        channel_2_range=(500.0, 600.0),
        roi_voxels=100,
        colocalized_voxels=25,
    )

    dialog.set_pending_thresholds(threshold_1=5.0, threshold_2=571.0)

    assert "Visible-density estimate: 15/100 (15.0%)" in (
        dialog.summary_label.text()
    )
    assert (dialog.plot._channel_1_min, dialog.plot._channel_1_max) == (0.0, 10.0)
    assert (dialog.plot._channel_2_min, dialog.plot._channel_2_max) == (
        500.0,
        600.0,
    )


def test_scatter_dialog_distinguishes_clipped_density_from_full_roi(qtbot):
    dialog = ColocalizationScatterDialog()
    qtbot.addWidget(dialog)
    density = np.zeros((10, 10), dtype=np.float64)
    density.ravel()[:80] = 1.0
    dialog.set_density(
        density,
        threshold_1=5.0,
        threshold_2=5.0,
        intensity_min=0.0,
        intensity_max=10.0,
        roi_voxels=100,
        colocalized_voxels=30,
        range_percentile=90.0,
    )

    dialog.set_pending_thresholds(threshold_1=6.0, threshold_2=6.0)

    summary = dialog.summary_label.text()
    assert "Visible density includes 80/100 ROI voxels" in summary
    assert "20 tails are hidden by the 90% display range" in summary
    assert "Exact full-ROI counts" in summary
    assert "display clipping does not change metrics" in summary


@pytest.mark.parametrize("suffix", [".png", ".tif", ".tiff"])
def test_scatter_dialog_exports_current_plot_resolution(qtbot, tmp_path, suffix):
    dialog = ColocalizationScatterDialog()
    qtbot.addWidget(dialog)
    _populate_dialog(dialog)
    dialog.plot.resize(420, 380)
    expected_shape = (dialog.plot.height(), dialog.plot.width())
    target = tmp_path / f"scatter{suffix}"

    exported = dialog.export_image(target)

    assert exported == target
    assert target.exists()
    if suffix == ".png":
        saved = QImage(str(target))
        assert not saved.isNull()
        assert (saved.height(), saved.width()) == expected_shape
    else:
        saved = tifffile.imread(target)
        assert saved.shape == expected_shape + (3,)
        assert saved.dtype == np.uint8


def test_scatter_dialog_rejects_unknown_export_format(qtbot, tmp_path):
    dialog = ColocalizationScatterDialog()
    qtbot.addWidget(dialog)

    with pytest.raises(ValueError, match="PNG, TIF, and TIFF"):
        dialog.export_image(tmp_path / "scatter.jpg")


def test_scatter_dialog_keeps_inspector_resolution_cap_visible(qtbot):
    dialog = ColocalizationScatterDialog()
    qtbot.addWidget(dialog)
    note = (
        "Inspector/popout density is capped at 1,024 x 1,024 bins; "
        "the graph operation keeps its requested 4,096-bin histogram."
    )

    dialog.set_density(
        np.ones((32, 32), dtype=np.float64),
        threshold_1=25.0,
        threshold_2=25.0,
        intensity_min=0.0,
        intensity_max=255.0,
        roi_voxels=1_024,
        colocalized_voxels=512,
        display_note=note,
    )

    assert note in dialog.summary_label.text()
    assert dialog.plot.toolTip() == note


def test_qimage_rgb_array_preserves_channel_values():
    image = QImage(3, 2, QImage.Format_RGB888)
    image.fill(QColor(18, 52, 86))

    array = qimage_rgb_array(image)

    assert array.shape == (2, 3, 3)
    assert np.all(array == np.asarray([18, 52, 86], dtype=np.uint8))


@pytest.mark.parametrize(
    ("base", "text"),
    [("#111827", "#f8fafc"), ("#ffffff", "#111827")],
)
def test_widget_export_background_uses_the_active_palette(qtbot, base, text):
    widget = QWidget()
    qtbot.addWidget(widget)
    widget.resize(12, 10)
    widget.setAttribute(Qt.WA_TranslucentBackground, True)
    palette = QPalette()
    palette.setColor(QPalette.Base, QColor(base))
    palette.setColor(QPalette.Window, QColor(base))
    palette.setColor(QPalette.Text, QColor(text))
    palette.setColor(QPalette.WindowText, QColor(text))
    widget.setPalette(palette)

    pixels = qimage_rgb_array(render_widget_image(widget))
    expected = theme_colors(palette).surface

    assert np.all(
        pixels[0, 0]
        == np.asarray(
            [expected.red(), expected.green(), expected.blue()],
            dtype=np.uint8,
        )
    )
