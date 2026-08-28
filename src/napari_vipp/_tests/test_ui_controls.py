from __future__ import annotations

import pytest
from qtpy.QtCore import QPoint, Qt
from qtpy.QtWidgets import QLabel

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
    ImageSourceResolutionPresentation,
)


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
    assert "fixed scientific analysis" in control.analysis_resolution_label.text()
    assert "L2 3 × 32 × 40" in control.pyramid_levels_label.text()
    assert "Level 2" in control.preview_resolution_label.text()
    assert "presentation only" in control.preview_resolution_label.text()
    assert [
        control.viewer_display_combo.itemData(index)
        for index in range(control.viewer_display_combo.count())
    ] == ["analysis", "preview:auto", "preview:1", "preview:2"]
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
