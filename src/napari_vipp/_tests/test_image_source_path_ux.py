from __future__ import annotations

from copy import deepcopy

import numpy as np

from napari_vipp._graph import ImageSourceMimePayload
from napari_vipp._tests.test_widget import _Viewer
from napari_vipp._widget import VippWidget


def test_committed_file_uri_is_canonicalized_before_binding(qtbot, tmp_path):
    source = tmp_path / "source with spaces.npy"
    np.save(source, np.arange(12, dtype=np.uint16).reshape(3, 4))
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget.graph_view.select_node("input")
    control = widget._parameter_widgets["image_source"]
    control.mode_combo.setCurrentText("file path")
    widget._debounce_timer.stop()

    control.path_edit.setText(f'  "{source.as_uri()}"  ')

    assert widget.pipeline.nodes["input"].params["file_path"] == ""

    control.path_edit.editingFinished.emit()
    widget._debounce_timer.stop()

    assert widget.pipeline.nodes["input"].params["file_path"] == str(
        source.resolve()
    )
    assert control.path_edit.text() == str(source.resolve())


def test_invalid_directory_commit_preserves_previous_source_binding(
    qtbot,
    monkeypatch,
    tmp_path,
):
    source = tmp_path / "valid.npy"
    np.save(source, np.ones((2, 2), dtype=np.uint8))
    ordinary_directory = tmp_path / "ordinary folder"
    ordinary_directory.mkdir()
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget.graph_view.select_node("input")
    node = widget.pipeline.nodes["input"]
    node.params.update(source_mode="file path", file_path=str(source.resolve()))
    control = widget._parameter_widgets["image_source"]
    control.set_options(
        widget._available_layer_names(),
        widget._sample_names(),
        value=widget._image_source_value(node),
        emit=False,
    )
    before = deepcopy(node.params)
    undo_count = len(widget._undo_stack)
    refreshes = []
    workers = []
    monkeypatch.setattr(
        widget,
        "_refresh_image_source_options",
        lambda: refreshes.append(True),
    )
    monkeypatch.setattr(
        widget,
        "_start_source_inspection",
        lambda *_args, **_kwargs: workers.append("inspection"),
    )
    monkeypatch.setattr(
        widget,
        "_start_source_file_load",
        lambda *_args, **_kwargs: workers.append("load"),
    )

    control.path_edit.setText(str(ordinary_directory))
    control.path_edit.editingFinished.emit()
    qtbot.wait(1)

    assert node.params == before
    assert len(widget._undo_stack) == undo_count
    assert refreshes == []
    assert workers == []
    assert ".zarr" in widget.status_label.text()


def test_dropped_ordinary_directory_is_rejected_without_source_edit(qtbot, tmp_path):
    ordinary_directory = tmp_path / "images"
    ordinary_directory.mkdir()
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    node = widget.pipeline.nodes["input"]
    before = deepcopy(node.params)

    opened = widget._open_image_on_source_node(
        "input",
        ImageSourceMimePayload(local_path=str(ordinary_directory)),
    )

    assert opened is False
    assert node.params == before
    assert ".zarr" in widget.status_label.text()


def test_zarr_browser_rejects_ordinary_directory_without_rebinding(
    qtbot,
    monkeypatch,
    tmp_path,
):
    ordinary_directory = tmp_path / "not-a-store"
    ordinary_directory.mkdir()
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget.graph_view.select_node("input")
    node = widget.pipeline.nodes["input"]
    control = widget._parameter_widgets["image_source"]
    control.mode_combo.setCurrentText("file path")
    widget._debounce_timer.stop()
    before = deepcopy(node.params)
    monkeypatch.setattr(
        "napari_vipp.ui.controls.QFileDialog.getExistingDirectory",
        lambda *_args, **_kwargs: str(ordinary_directory),
    )

    control._browse_zarr_path()

    assert node.params == before
    assert ".zarr" in widget.status_label.text()
