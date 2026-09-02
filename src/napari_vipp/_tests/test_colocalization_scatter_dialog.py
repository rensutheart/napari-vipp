from __future__ import annotations

import numpy as np
import pytest
import tifffile
from qtpy.QtCore import QPoint, Qt
from qtpy.QtGui import QColor, QImage, QPalette
from qtpy.QtWidgets import QWidget

from napari_vipp.ui.colocalization_scatter_dialog import (
    ColocalizationScatterDialog,
    qimage_rgb_array,
    render_widget_image,
)
from napari_vipp.ui.palette_roles import theme_colors
from napari_vipp.ui.plots import COLOCALIZATION_SCATTER_COLORMAPS


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
