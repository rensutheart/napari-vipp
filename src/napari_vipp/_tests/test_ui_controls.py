from __future__ import annotations

import pytest
from qtpy.QtCore import QPoint, Qt
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from napari_vipp.core.pipeline import ParameterSpec
from napari_vipp.ui import recent_paths
from napari_vipp.ui.axis_controls import (
    AxisSliceControl,
    AxisSliceOption,
    ReorderAxesControl,
    SelectTableColumnsControl,
)
from napari_vipp.ui.controls import (
    ImageSourceControl,
    ImageSourceMemoryRepairPresentation,
    ImageSourceResolutionPresentation,
)
from napari_vipp.ui.palette_roles import theme_colors


def _image_source_control():
    return ImageSourceControl(
        None,
        layer_names=[],
        sample_names=[],
    )


def test_reorder_axes_guidance_distinguishes_transpose_from_declaration(qtbot):
    control = ReorderAxesControl(
        [
            AxisSliceOption(0, "q", "unknown", 3),
            AxisSliceOption(1, "y", "space", 8),
            AxisSliceOption(2, "x", "space", 9),
        ]
    )
    qtbot.addWidget(control)

    guidance = " ".join(label.text() for label in control.findChildren(QLabel))
    assert "does not rename or reinterpret axes" in guidance
    assert "Declare axes" in guidance


def test_axis_slice_drag_and_hide_each_finish_their_gesture_once(qtbot):
    control = AxisSliceControl([AxisSliceOption(0, "z", "space", 10)])
    qtbot.addWidget(control)
    control.resize(420, 180)
    control.show()
    qtbot.waitExposed(control)
    events = []
    control.gestureStarted.connect(lambda: events.append("started"))
    control.gestureFinished.connect(lambda: events.append("finished"))

    slider = control._rows[0].range_slider
    start = QPoint(slider._x_for_value(0), slider.rect().center().y())
    middle = QPoint(slider._x_for_value(2), slider.rect().center().y())
    end = QPoint(slider._x_for_value(4), slider.rect().center().y())
    qtbot.mousePress(slider, Qt.LeftButton, pos=start)
    qtbot.mouseMove(slider, pos=middle)
    qtbot.mouseMove(slider, pos=end)

    assert events == ["started"]

    qtbot.mouseRelease(slider, Qt.LeftButton, pos=end)
    qtbot.mouseRelease(slider, Qt.LeftButton, pos=end)

    assert events == ["started", "finished"]

    start = QPoint(slider._x_for_value(4), slider.rect().center().y())
    end = QPoint(slider._x_for_value(6), slider.rect().center().y())
    qtbot.mousePress(slider, Qt.LeftButton, pos=start)
    qtbot.mouseMove(slider, pos=end)
    assert events == ["started", "finished", "started"]

    control.hide()
    control.hide()

    assert events == ["started", "finished", "started", "finished"]


@pytest.mark.parametrize("control_kind", ["axes", "table"])
def test_reorder_lists_forward_one_gesture_pair(control_kind, qtbot):
    if control_kind == "axes":
        control = ReorderAxesControl(
            [
                AxisSliceOption(0, "z", "space", 3),
                AxisSliceOption(1, "y", "space", 8),
                AxisSliceOption(2, "x", "space", 9),
            ]
        )
    else:
        control = SelectTableColumnsControl(["label", "area", "mean"])
    qtbot.addWidget(control)
    control.resize(420, 360)
    control.show()
    qtbot.waitExposed(control)
    events = []
    control.gestureStarted.connect(lambda: events.append("started"))
    control.gestureFinished.connect(lambda: events.append("finished"))

    axis_list = control.list_widget
    first = axis_list.visualItemRect(axis_list.item(0)).center()
    third = axis_list.visualItemRect(axis_list.item(2)).center()
    qtbot.mousePress(axis_list.viewport(), Qt.LeftButton, pos=first)
    qtbot.mouseMove(axis_list.viewport(), pos=third)

    assert events == ["started"]

    qtbot.mouseRelease(axis_list.viewport(), Qt.LeftButton, pos=third)
    qtbot.mouseRelease(axis_list.viewport(), Qt.LeftButton, pos=third)

    assert events == ["started", "finished"]


def test_image_source_choosers_share_recent_input_directory(
    qtbot,
    monkeypatch,
    tmp_path,
):
    selected_dir = tmp_path / "images"
    selected_dir.mkdir()
    selected_file = selected_dir / "image.npy"
    starts = []

    def choose_file(_parent, _title, start, _filters):
        starts.append(start)
        return str(selected_file), ""

    monkeypatch.setattr(
        "napari_vipp.ui.controls.QFileDialog.getOpenFileName",
        choose_file,
    )
    first = _image_source_control()
    qtbot.addWidget(first)
    first._browse_path()

    directory_starts = []

    def cancel_directory(_parent, _title, start):
        directory_starts.append(start)
        return ""

    monkeypatch.setattr(
        "napari_vipp.ui.controls.QFileDialog.getExistingDirectory",
        cancel_directory,
    )
    second = _image_source_control()
    qtbot.addWidget(second)
    second._browse_zarr_path()

    assert starts == [""]
    assert directory_starts == [str(selected_dir.resolve())]
    assert recent_paths.recent_directory(recent_paths.INPUT_DIRECTORY) == str(
        selected_dir.resolve()
    )


def test_image_source_path_is_published_only_after_editing_finishes(qtbot):
    control = _image_source_control()
    qtbot.addWidget(control)
    changes = []
    control.pathCommitted.connect(changes.append)

    control.path_edit.setText("C:/images/example.ome.tif")

    assert changes == []

    control.path_edit.editingFinished.emit()

    assert len(changes) == 1
    assert changes[0]["file_path"] == "C:/images/example.ome.tif"


def test_image_source_resolution_panel_is_read_only_and_actionable(qtbot):
    control = ImageSourceControl(
        {
            "source_mode": "file path",
            "file_path": "sample.ome.zarr",
            "series_index": 0,
        },
        layer_names=[],
        sample_names=[],
    )
    qtbot.addWidget(control)
    original_value = control.value()
    viewer_choices = []
    retried = []
    control.viewerDisplayChanged.connect(viewer_choices.append)
    control.previewReloadRequested.connect(lambda: retried.append(True))

    control.set_resolution_presentation(
        ImageSourceResolutionPresentation(
            analysis_axes="ZYX",
            analysis_shape=(12, 128, 160),
            level_shapes=((12, 128, 160), (6, 64, 80), (3, 32, 40)),
            preview_state="ready",
            preview_level=2,
            preview_shape=(32, 40),
            viewer_choice="preview:auto",
            can_select_preview=True,
        )
    )

    assert control.resolution_panel.isVisibleTo(control)
    assert "Level 0" in control.analysis_resolution_label.text()
    assert "ZYX 12 × 128 × 160" in control.analysis_resolution_label.text()
    assert "Source · Level 0" in control.analysis_resolution_label.text()
    assert "fixed processing data" in control.analysis_resolution_label.text()
    assert "L2 3 × 32 × 40" in control.pyramid_levels_label.text()
    assert "Level 2" in control.preview_resolution_label.text()
    assert "presentation only" in control.preview_resolution_label.text()
    assert [
        control.viewer_display_combo.itemData(index)
        for index in range(control.viewer_display_combo.count())
    ] == ["analysis", "preview:auto", "preview:1", "preview:2"]
    assert control.viewer_display_combo.itemText(0).startswith(
        "Full-resolution source"
    )
    assert control.viewer_display_combo.currentData() == "preview:auto"
    assert "L1" in control.viewer_display_combo.itemText(2)
    assert "6 × 64 × 80" in control.viewer_display_combo.itemText(2)
    control.viewer_display_combo.setCurrentIndex(
        control.viewer_display_combo.findData("analysis")
    )
    control.viewer_display_combo.setCurrentIndex(
        control.viewer_display_combo.findData("preview:1")
    )
    control.viewer_display_combo.setCurrentIndex(
        control.viewer_display_combo.findData("preview:auto")
    )

    control.set_resolution_presentation(
        ImageSourceResolutionPresentation(
            analysis_axes="ZYX",
            analysis_shape=(12, 128, 160),
            level_shapes=((12, 128, 160), (6, 64, 80), (3, 32, 40)),
            preview_state="failed",
            preview_detail="Temporary read failure.",
            can_retry=True,
        )
    )
    control.preview_reload_button.click()

    assert viewer_choices == ["analysis", "preview:1", "preview:auto"]
    assert retried == [True]
    assert control.value() == original_value


def test_image_source_resolution_panel_disables_unsupported_preview(qtbot):
    control = ImageSourceControl(
        {
            "source_mode": "file path",
            "file_path": "unsupported-pyramid.zarr",
        },
        layer_names=[],
        sample_names=[],
    )
    qtbot.addWidget(control)
    viewer_choices = []
    control.viewerDisplayChanged.connect(viewer_choices.append)

    control.set_resolution_presentation(
        ImageSourceResolutionPresentation(
            analysis_axes="ZYX",
            analysis_shape=(12, 128, 160),
            level_shapes=((12, 128, 160), (6, 64, 80)),
            viewer_choice="preview:auto",
            can_select_preview=False,
        )
    )

    assert control.viewer_display_combo.count() == 1
    assert control.viewer_display_combo.currentData() == "analysis"
    assert viewer_choices == []


def test_image_source_resolution_panel_labels_loaded_exact_crop_window(qtbot):
    control = ImageSourceControl(
        {
            "source_mode": "file path",
            "file_path": "large.ome.zarr",
        },
        layer_names=[],
        sample_names=[],
    )
    qtbot.addWidget(control)

    control.set_resolution_presentation(
        ImageSourceResolutionPresentation(
            analysis_axes="ZYX",
            analysis_shape=(512, 8192, 8192),
            level_shapes=((512, 8192, 8192),),
            analysis_window_bounds=((224, 288), (3584, 4608), (3584, 4608)),
            analysis_window_shape=(64, 1024, 1024),
        )
    )

    assert control.resolution_panel.isVisibleTo(control)
    assert "Loaded Crop Stack window" in control.analysis_resolution_label.text()
    assert "Z 224:288" in control.analysis_resolution_label.text()
    assert "Y 3584:4608" in control.analysis_resolution_label.text()
    assert "64 × 1024 × 1024" in control.analysis_resolution_label.text()
    assert "Full level 0 was not materialized" in (
        control.analysis_resolution_label.text()
    )
    assert control.viewer_display_combo.currentData() == "analysis"
    assert "Loaded Crop Stack window" in control.viewer_display_combo.currentText()


def test_image_source_resolution_panel_wraps_inside_narrow_inspector(qtbot):
    control = ImageSourceControl(
        {
            "source_mode": "file path",
            "file_path": "oversized-sparse.ome.zarr",
        },
        layer_names=[],
        sample_names=[],
    )
    qtbot.addWidget(control)
    control.set_resolution_presentation(
        ImageSourceResolutionPresentation(
            analysis_axes="ZYX",
            analysis_shape=(512, 8192, 8192),
            level_shapes=((512, 8192, 8192), (17, 513, 513)),
            preview_state="ready",
            preview_level=1,
            preview_shape=(17, 513, 513),
            viewer_choice="preview:auto",
            can_select_preview=True,
            analysis_window_bounds=(
                (229, 282),
                (2852, 4526),
                (3665, 4526),
            ),
            analysis_window_shape=(53, 1674, 861),
        )
    )

    control.resize(340, 900)
    control.show()
    qtbot.waitExposed(control)

    assert control.minimumSizeHint().width() <= 340
    assert control.resolution_panel.width() == control.width()
    for label in (
        control.analysis_resolution_label,
        control.pyramid_levels_label,
        control.preview_resolution_label,
    ):
        assert label.sizePolicy().horizontalPolicy() == QSizePolicy.Ignored
        assert label.width() <= control.resolution_panel.width()
        assert label.height() >= label.heightForWidth(label.width())
    assert control.viewer_display_combo.width() <= control.resolution_panel.width()


def test_image_source_resolution_panel_can_be_hosted_and_restored(qtbot):
    container = QWidget()
    qtbot.addWidget(container)
    container_layout = QVBoxLayout(container)
    control = ImageSourceControl(
        {
            "source_mode": "file path",
            "file_path": "sample.ome.zarr",
        },
        layer_names=[],
        sample_names=[],
    )
    host = QWidget()
    host_layout = QVBoxLayout(host)
    container_layout.addWidget(control)
    container_layout.addWidget(host)
    control.set_resolution_presentation(
        ImageSourceResolutionPresentation(
            analysis_axes="ZYX",
            analysis_shape=(12, 128, 160),
            level_shapes=((12, 128, 160), (6, 64, 80)),
            preview_state="ready",
            preview_level=1,
            preview_shape=(6, 64, 80),
            can_select_preview=True,
        )
    )
    container.show()
    qtbot.waitExposed(container)

    control.set_source_representation_host(host)

    assert control.source_representation_host is host
    assert control.resolution_panel.parentWidget() is host
    assert host_layout.indexOf(control.resolution_panel) >= 0
    assert control._source_representation_home.isHidden()
    assert control.resolution_panel.isVisibleTo(host)

    control.mode_combo.setCurrentText("sample")
    assert control.resolution_panel.isHidden()
    control.mode_combo.setCurrentText("file path")
    assert control.resolution_panel.isVisibleTo(host)

    control.set_resolution_presentation(ImageSourceResolutionPresentation())
    assert control.resolution_panel.isHidden()
    control.set_resolution_presentation(
        ImageSourceResolutionPresentation(
            analysis_shape=(12, 128, 160),
            level_shapes=((12, 128, 160), (6, 64, 80)),
        )
    )
    assert control.resolution_panel.isVisibleTo(host)

    control.restore_source_representation_panel()
    control.restore_source_representation_panel()

    assert control.source_representation_host is None
    assert (
        control.resolution_panel.parentWidget()
        is control._source_representation_home
    )
    assert host_layout.indexOf(control.resolution_panel) == -1
    assert control._source_representation_home.isVisibleTo(control)
    assert control.resolution_panel.isVisibleTo(control)


def test_image_source_resolution_host_requires_an_existing_layout(qtbot):
    control = _image_source_control()
    host = QWidget()
    qtbot.addWidget(control)
    qtbot.addWidget(host)

    with pytest.raises(ValueError, match="must already have a layout"):
        control.set_source_representation_host(host)

    assert control.source_representation_host is None
    assert (
        control.resolution_panel.parentWidget()
        is control._source_representation_home
    )


def test_hosted_image_source_resolution_panel_follows_host_palette(qtbot):
    control = ImageSourceControl(
        {"source_mode": "file path", "file_path": "sample.ome.zarr"},
        layer_names=[],
        sample_names=[],
    )
    host = QWidget()
    QVBoxLayout(host)
    qtbot.addWidget(control)
    qtbot.addWidget(host)
    control.set_source_representation_host(host)

    palette = QPalette(host.palette())
    surface = QColor("#f8fafc")
    foreground = QColor("#172033")
    palette.setColor(QPalette.Base, surface)
    palette.setColor(QPalette.Window, surface)
    palette.setColor(QPalette.AlternateBase, QColor("#eef2f7"))
    palette.setColor(QPalette.Text, foreground)
    palette.setColor(QPalette.WindowText, foreground)
    host.setPalette(palette)

    expected = theme_colors(palette).muted_text.name()
    qtbot.waitUntil(
        lambda: expected in control.pyramid_levels_label.styleSheet()
    )

    control.restore_source_representation_panel()


def test_image_source_memory_repair_is_explicit_transient_and_read_only(qtbot):
    control = ImageSourceControl(
        {"source_mode": "file path", "file_path": "large.ome.zarr"},
        layer_names=[],
        sample_names=[],
    )
    qtbot.addWidget(control)
    original = control.value()
    requested = []
    dismissed = []
    control.sourceCropRepairRequested.connect(lambda: requested.append(True))
    control.sourceCropRepairDismissed.connect(lambda: dismissed.append(True))

    control.set_memory_repair_presentation(
        ImageSourceMemoryRepairPresentation(
            visible=True,
            message="The full source does not fit safe RAM.",
            action_label="Add fitted Crop Stack",
            enabled=True,
            tooltip="Author one visible, undoable crop.",
        )
    )

    assert control.memory_repair_panel.isVisibleTo(control)
    assert "does not fit" in control.memory_repair_label.text()
    control.memory_repair_button.click()
    assert requested == [True]
    assert control.value() == original

    control.memory_repair_dismiss_button.click()
    assert control.memory_repair_panel.isHidden()
    assert dismissed == [True]
    assert control.value() == original


def test_image_source_mode_hides_dynamic_fields_and_their_labels(qtbot):
    control = _image_source_control()
    qtbot.addWidget(control)

    def row_hidden(field) -> bool:
        label = control.form_layout.labelForField(field)
        return field.isHidden() and (label is None or label.isHidden())

    control.mode_combo.setCurrentText("file path")
    assert not control.file_row.isHidden()
    assert not control.form_layout.labelForField(control.file_row).isHidden()
    assert row_hidden(control.layer_row)
    assert row_hidden(control.sample_row)

    control.mode_combo.setCurrentText("sample")
    assert not control.sample_row.isHidden()
    assert not control.form_layout.labelForField(control.sample_row).isHidden()
    assert row_hidden(control.layer_row)
    assert row_hidden(control.file_row)
    assert row_hidden(control.binding_combo)


def test_image_source_compact_form_keeps_visible_labels_readable(qtbot):
    control = ImageSourceControl(
        {"source_mode": "sample", "sample_name": "VIPP synthetic volume"},
        layer_names=[],
        sample_names=["VIPP synthetic volume"],
    )
    qtbot.addWidget(control)
    control.resize(280, 360)
    control.set_compact_form_mode(True)
    control.show()

    labels = [
        control.form_layout.labelForField(field)
        for field in (
            control.mode_combo,
            control.axis_control,
            control.sample_row,
        )
    ]
    qtbot.waitUntil(lambda: all(label.width() > 0 for label in labels))

    assert all(not label.isHidden() for label in labels)
    assert [label.text() for label in labels] == [
        "Source",
        "Image stack",
        "Sample",
    ]
    assert all(
        label.sizePolicy().horizontalPolicy() == QSizePolicy.Preferred
        for label in labels
    )


def test_parameter_spec_accepts_narrower_declarative_slider_window():
    spec = ParameterSpec(
        "value",
        "Value",
        "float",
        2.0,
        0.0,
        1_000_000.0,
        0.1,
        3,
        slider_minimum=0.0,
        slider_maximum=10.0,
    )

    assert spec.minimum == 0.0
    assert spec.maximum == 1_000_000.0
    assert spec.slider_minimum == 0.0
    assert spec.slider_maximum == 10.0


@pytest.mark.parametrize(
    "slider_bounds",
    [
        (11.0, 10.0),
        (-1.0, 10.0),
        (0.0, 1_000_001.0),
        (0.0, float("inf")),
    ],
)
def test_parameter_spec_rejects_invalid_slider_window(slider_bounds):
    with pytest.raises(ValueError, match="slider"):
        ParameterSpec(
            "value",
            "Value",
            "float",
            2.0,
            0.0,
            1_000_000.0,
            0.1,
            3,
            slider_minimum=slider_bounds[0],
            slider_maximum=slider_bounds[1],
        )


def test_parameter_spec_rejects_fractional_integer_slider_window():
    with pytest.raises(ValueError, match="whole numbers"):
        ParameterSpec(
            "value",
            "Value",
            "int",
            2,
            0,
            100,
            1,
            slider_minimum=0.5,
            slider_maximum=10,
        )
