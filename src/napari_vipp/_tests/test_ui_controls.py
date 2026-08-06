from __future__ import annotations

from qtpy.QtWidgets import QLabel

from napari_vipp.ui import recent_paths
from napari_vipp.ui.axis_controls import AxisSliceOption, ReorderAxesControl
from napari_vipp.ui.controls import ImageSourceControl


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
