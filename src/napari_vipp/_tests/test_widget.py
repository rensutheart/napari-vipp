from __future__ import annotations

import time
from dataclasses import replace

import imageio.v3 as iio
import numpy as np
import tifffile
from qtpy.QtCore import QEvent, QPointF, QSignalBlocker, Qt
from qtpy.QtGui import QKeySequence, QMouseEvent
from qtpy.QtWidgets import (
    QApplication,
    QDockWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QScrollArea,
    QWidget,
)

from napari_vipp._theme import category_color, category_tint
from napari_vipp._widget import VippWidget
from napari_vipp import __version__ as VIPP_VERSION
from napari_vipp.core.io import inspect_image_source, read_image
from napari_vipp.core.pipeline import (
    NODE_LIBRARY_BY_ID,
    PALETTE_NODE_LIBRARY,
    SourcePayload,
)
from napari_vipp.core.preview import make_preview


class _Event:
    def __init__(self):
        self.callback = None

    def connect(self, callback):
        self.callback = callback

    def emit(self):
        if self.callback is not None:
            self.callback()


class _LayerEvents:
    def __init__(self):
        self.inserted = _Event()
        self.removed = _Event()


class _DimsEvents:
    def __init__(self):
        self.current_step = _Event()
        self.point = _Event()


class _Dims:
    def __init__(self):
        self.current_step = (0, 0, 0)
        self.events = _DimsEvents()


class _Layer:
    def __init__(self, data, name, metadata=None, layer_type="image"):
        self.data = data
        self.name = name
        self.metadata = metadata or {}
        self.layer_type = layer_type
        self.blending = None
        self.colormap = None
        self.contrast_limits = None
        self.visible = True
        self.rgb = False


class _LayerList(list):
    def __init__(self, layers):
        super().__init__(layers)
        self.events = _LayerEvents()

    def __getitem__(self, item):
        if isinstance(item, str):
            for layer in self:
                if layer.name == item:
                    return layer
            raise KeyError(item)
        return super().__getitem__(item)

    def move(self, source, target):
        if source < target:
            target -= 1
        if source == target:
            return False
        layer = self.pop(source)
        self.insert(target, layer)
        return True


class _Viewer:
    def __init__(self, data=None, metadata=None):
        if data is None:
            data = np.zeros((4, 16, 18), dtype=np.float32)
        self.layers = _LayerList([_Layer(data, "input volume", metadata=metadata)])
        self.dims = _Dims()

    def add_image(self, data, **kwargs):
        layer = _Layer(
            data,
            kwargs["name"],
            metadata=kwargs.get("metadata"),
            layer_type="image",
        )
        layer.blending = kwargs.get("blending")
        layer.colormap = kwargs.get("colormap")
        layer.contrast_limits = kwargs.get("contrast_limits")
        layer.rgb = bool(kwargs.get("rgb", False))
        self.layers.append(layer)
        return layer

    def add_labels(self, data, **kwargs):
        layer = _Layer(
            data,
            kwargs["name"],
            metadata=kwargs.get("metadata"),
            layer_type="labels",
        )
        self.layers.append(layer)
        return layer


def _palette_item(widget, operation_id):
    def find_child(item):
        for child_index in range(item.childCount()):
            child = item.child(child_index)
            if child.data(0, Qt.UserRole) == operation_id:
                return child
            found = find_child(child)
            if found is not None:
                return found
        return None

    for category_index in range(widget.palette.topLevelItemCount()):
        category = widget.palette.topLevelItem(category_index)
        found = find_child(category)
        if found is not None:
            return found
    raise AssertionError(f"Palette item not found: {operation_id}")


def _palette_category(widget, category_name):
    for category_index in range(widget.palette.topLevelItemCount()):
        item = widget.palette.topLevelItem(category_index)
        if item.text(0) == category_name:
            return item
    raise AssertionError(f"Palette category not found: {category_name}")


def _palette_child_by_text(parent, text):
    for child_index in range(parent.childCount()):
        child = parent.child(child_index)
        if child.text(0) == text:
            return child
    raise AssertionError(f"Palette child not found: {text}")


def _metadata_value(widget, label):
    for row in range(widget.metadata_table.rowCount()):
        label_item = widget.metadata_table.item(row, 0)
        value_item = widget.metadata_table.item(row, 1)
        if label_item is not None and label_item.text() == label:
            return value_item.text() if value_item is not None else ""
    raise AssertionError(f"Metadata row not found: {label}")


def _graph_view_center(view):
    return view.mapToScene(view.viewport().rect().center())


def test_widget_builds_graph_and_inspects_node(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.layer_combo.count() == 1
    assert "gaussian" in widget.pipeline.outputs
    assert widget.version_label.text() == f"VIPP {VIPP_VERSION}"
    assert widget.background_all_checkbox.text() == "Run all in BG"

    inspect_layer = viewer.layers["VIPP Inspect"]
    assert inspect_layer.metadata["node_id"] == "gaussian"
    assert inspect_layer.data.shape == viewer.layers["input volume"].data.shape
    assert not viewer.layers["input volume"].visible


def test_delete_selected_node_removes_pipeline_node_and_connections(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    widget.graph_view.select_node("gaussian")

    qtbot.keyClick(widget.graph_view, Qt.Key_Delete)

    assert "gaussian" not in widget.pipeline.nodes
    assert "gaussian" not in widget.graph_view._cards
    assert all(
        connection.source_id != "gaussian" and connection.target_id != "gaussian"
        for connection in widget.pipeline.connections
    )
    assert widget._selected_node_id in widget.pipeline.nodes


def test_deleting_all_nodes_leaves_empty_inspector_without_error(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    for node_id in list(widget.pipeline.nodes):
        widget._delete_node(node_id)

    assert widget.pipeline.nodes == {}
    assert widget._selected_node_id == ""
    assert widget.selected_title.text() == "No node selected"
    assert widget.parameter_group.isHidden()
    assert widget.metadata_table.rowCount() == 0
    assert widget.history_label.text() == "No history yet."


def test_duplicate_node_copies_parameters_without_connections(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    widget.pipeline.set_param("gaussian", "sigma", 3.5)
    before_ids = set(widget.pipeline.nodes)

    widget._duplicate_node("gaussian")

    new_ids = set(widget.pipeline.nodes) - before_ids
    assert len(new_ids) == 1
    clone_id = new_ids.pop()
    clone = widget.pipeline.nodes[clone_id]
    assert clone.operation_id == "gaussian_blur"
    assert clone.params["sigma"] == 3.5
    assert not any(
        connection.source_id == clone_id or connection.target_id == clone_id
        for connection in widget.pipeline.connections
    )
    assert widget._selected_node_id == clone_id


def test_node_code_text_includes_call_and_source(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    code = widget._node_code_text("gaussian")

    assert "from napari_vipp.core.operations import gaussian_blur" in code
    assert "output = gaussian_blur(input_output" in code
    assert "def gaussian_blur" in code

    widget._inspect_node_code("gaussian")
    dialog = widget._code_dialogs[-1]
    editor = dialog.findChild(QPlainTextEdit)
    assert editor is not None
    assert hasattr(editor, "_vipp_python_highlighter")
    dialog.close()


def test_undo_redo_restores_deleted_node_and_connections(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    widget.graph_view.select_node("gaussian")
    widget.graph_view.resize(800, 420)
    widget.graph_view.set_zoom_percent(150)
    widget.graph_view.centerOn(QPointF(240, 90))
    zoom_before = widget.graph_view.zoom_percent
    transform_before = widget.graph_view.transform()
    center_before = _graph_view_center(widget.graph_view)

    assert (
        widget.undo_action.shortcut().matches(QKeySequence(QKeySequence.Undo))
        == QKeySequence.ExactMatch
    )
    assert (
        widget.redo_action.shortcut().matches(QKeySequence(QKeySequence.Redo))
        == QKeySequence.ExactMatch
    )
    assert not widget.undo_action.isEnabled()

    qtbot.keyClick(widget.graph_view, Qt.Key_Delete)

    assert "gaussian" not in widget.pipeline.nodes
    assert widget.undo_action.isEnabled()
    widget.undo()

    center_after = _graph_view_center(widget.graph_view)
    assert widget.graph_view.zoom_percent == zoom_before
    assert widget.graph_view.transform() == transform_before
    assert abs(center_after.x() - center_before.x()) <= 1.0
    assert abs(center_after.y() - center_before.y()) <= 1.0
    assert "gaussian" in widget.pipeline.nodes
    assert "gaussian" in widget.graph_view._cards
    assert ("input", "gaussian") in {
        (connection.source_id, connection.target_id)
        for connection in widget.pipeline.connections
    }
    assert ("gaussian", "threshold") in {
        (connection.source_id, connection.target_id)
        for connection in widget.pipeline.connections
    }
    assert widget._selected_node_id == "gaussian"
    assert widget.redo_action.isEnabled()

    widget.redo()

    assert "gaussian" not in widget.pipeline.nodes
    assert "gaussian" not in widget.graph_view._cards


def test_undo_restores_moved_node_position(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    proxy = widget.graph_view._proxies["gaussian"]
    old_pos = QPointF(proxy.pos())
    new_pos = old_pos + QPointF(120, 45)

    proxy.setPos(new_pos)
    widget._on_node_moved("gaussian", old_pos, new_pos)

    assert proxy.pos() == new_pos
    assert widget.undo_action.isEnabled()
    widget.undo()

    restored = widget.graph_view._proxies["gaussian"].pos()
    assert restored == old_pos


def test_widget_restores_hidden_source_layer_on_close(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    assert not viewer.layers["input volume"].visible

    widget.close()

    assert viewer.layers["input volume"].visible


def test_switching_input_layers_restores_previous_source(qtbot):
    viewer = _Viewer()
    viewer.layers.append(_Layer(np.ones((4, 16, 18), dtype=np.float32), "second"))
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    first = viewer.layers["input volume"]
    second = viewer.layers["second"]

    assert not first.visible
    assert second.visible

    widget.layer_combo.setCurrentText("second")

    assert first.visible
    assert not second.visible


def test_widget_prefers_time_lapse_multichannel_sample_input(qtbot):
    viewer = _Viewer(
        np.zeros((12, 16, 18), dtype=np.uint8),
        metadata={"napari_vipp_sample": True, "vipp_axis_order": "ZYX"},
    )
    viewer.layers[0].name = "VIPP synthetic volume"
    rich_data = np.zeros((5, 3, 4, 16, 18), dtype=np.uint16)
    viewer.layers.append(
        _Layer(
            rich_data,
            "VIPP synthetic time-lapse multichannel",
            metadata={
                "napari_vipp_sample": True,
                "napari_vipp_preferred_input": True,
                "vipp_axis_order": "TCZYX",
            },
        )
    )

    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.layer_combo.currentText() == "VIPP synthetic time-lapse multichannel"
    assert widget.pipeline.outputs["input"].shape == rich_data.shape

    widget.graph_view.select_node("input")

    assert _metadata_value(widget, "Dimensions") == "t=5, c=3, z=4, y=16, x=18"


def test_image_source_node_can_select_napari_layer(qtbot):
    viewer = _Viewer(np.zeros((2, 4, 5), dtype=np.uint8))
    second = np.ones((3, 6, 7), dtype=np.uint16)
    viewer.layers.append(_Layer(second, "second layer", metadata={"axes": "ZYX"}))
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.graph_view.select_node("input")
    control = widget._parameter_widgets["image_source"]
    control.layer_combo.setCurrentText("second layer")
    widget.run_pipeline()

    assert widget.pipeline.nodes["input"].params["layer_name"] == "second layer"
    assert widget.pipeline.outputs["input"].shape == second.shape
    assert _metadata_value(widget, "Dimensions") == "z=3, y=6, x=7"


def test_image_source_node_can_use_sample_mode(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.graph_view.select_node("input")
    control = widget._parameter_widgets["image_source"]
    control.mode_combo.setCurrentText("sample")
    control.sample_combo.setCurrentText("VIPP synthetic volume")
    widget.run_pipeline()

    assert widget.pipeline.nodes["input"].params["source_mode"] == "sample"
    assert widget.pipeline.outputs["input"].shape == (12, 96, 128)


def test_image_source_node_inspects_and_selects_tiff_series(qtbot, tmp_path):
    first = np.zeros((5, 6), dtype=np.uint8)
    second = np.ones((7, 8), dtype=np.uint16)
    path = tmp_path / "two-series.tif"
    with tifffile.TiffWriter(path) as tif:
        tif.write(first, metadata={"axes": "YX"})
        tif.write(second, metadata={"axes": "YX"})

    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    widget.graph_view.select_node("input")
    control = widget._parameter_widgets["image_source"]
    control.mode_combo.setCurrentText("file path")
    control.path_edit.setText(str(path))
    widget._refresh_image_source_options()

    assert control.series_combo.count() == 2
    control.series_combo.setCurrentIndex(1)
    widget.run_pipeline()

    assert widget.pipeline.nodes["input"].params["series_index"] == 1
    assert widget.pipeline.outputs["input"].shape == second.shape
    assert widget.pipeline.output_states["input"].source.format == "tiff"


def test_image_source_node_loads_common_raster_file(qtbot, tmp_path):
    data = np.zeros((5, 6, 3), dtype=np.uint8)
    data[..., 0] = 255
    path = tmp_path / "source.png"
    iio.imwrite(path, data)

    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    widget.graph_view.select_node("input")
    control = widget._parameter_widgets["image_source"]
    control.mode_combo.setCurrentText("file path")
    control.path_edit.setText(str(path))
    widget._refresh_image_source_options()

    assert control.series_combo.count() == 1
    assert "png" in control.source_summary.text()
    widget.run_pipeline()

    assert widget.pipeline.outputs["input"].shape == data.shape
    assert widget.pipeline.output_states["input"].kind == "RGB image"
    assert widget.pipeline.output_states["input"].source.format == "png"


def test_current_view_metadata_follows_napari_dims(qtbot):
    viewer = _Viewer(
        np.zeros((5, 3, 4, 16, 18), dtype=np.uint16),
        metadata={"axes": "TCZYX"},
    )
    viewer.dims.current_step = (2, 1, 3, 0, 0)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.graph_view.select_node("input")

    assert _metadata_value(widget, "Current view") == "t=2/4, c=1/2, z=3/3"

    viewer.dims.current_step = (4, 0, 1, 0, 0)
    viewer.dims.events.current_step.emit()

    assert _metadata_value(widget, "Current view") == "t=4/4, c=0/2, z=1/3"


def test_napari_layer_source_axes_are_right_aligned_to_viewer_dims(qtbot):
    viewer = _Viewer(
        np.zeros((3, 4, 16, 18), dtype=np.uint16),
        metadata={"axes": "CZYX"},
    )
    viewer.dims.current_step = (0, 0, 2, 0, 0)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    state = widget.pipeline.output_states["input"]

    assert state.axis_order == "CZYX"
    assert [axis.source_axis for axis in state.axes] == [1, 2, 3, 4]
    assert _metadata_value(widget, "Current view") == "c=0/2, z=2/3"


def test_sample_source_axes_are_right_aligned_to_viewer_dims(qtbot):
    viewer = _Viewer(
        np.zeros((5, 3, 4, 8, 9), dtype=np.uint16),
        metadata={"axes": "TCZYX"},
    )
    viewer.dims.current_step = (4, 2, 0, 0, 0)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    sample = np.zeros((4, 8, 9), dtype=np.uint8)
    sample[0, 2, 3] = 100
    sample[3, 5, 6] = 200
    widget._sample_payload_cache = {
        "tiny zyx": SourcePayload(sample, {"axes": "ZYX"}, "tiny zyx")
    }
    widget.pipeline.nodes["input"].params.update(
        {"source_mode": "sample", "sample_name": "tiny zyx"}
    )
    widget.run_pipeline()
    widget.graph_view.select_node("input")

    state = widget.pipeline.output_states["input"]
    assert [axis.source_axis for axis in state.axes] == [2, 3, 4]
    assert _metadata_value(widget, "Current view") == "z=0/3"

    first = make_preview(
        sample,
        mode="slice",
        current_step=(4, 2, 0, 0, 0),
        state=state,
    )
    second = make_preview(
        sample,
        mode="slice",
        current_step=(0, 0, 3, 0, 0),
        state=state,
    )

    assert first[2, 3] > 0
    assert first[5, 6] == 0
    assert second[5, 6] > 0
    assert second[2, 3] == 0


def test_dims_point_event_refreshes_thumbnails(qtbot, monkeypatch):
    viewer = _Viewer(
        np.zeros((5, 3, 4, 16, 18), dtype=np.uint16),
        metadata={"axes": "TCZYX"},
    )
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    calls = []

    def fake_make_preview(
        data,
        mode,
        current_step,
        state=None,
        channel_colors=None,
        contrast_mode="Percentile",
    ):
        calls.append(tuple(current_step))
        return np.zeros((16, 18), dtype=np.uint8)

    monkeypatch.setattr("napari_vipp._widget.make_preview", fake_make_preview)
    viewer.dims.current_step = (3, 1, 2, 0, 0)
    viewer.dims.events.point.emit()

    assert calls
    assert calls[-1] == (3, 1, 2, 0, 0)


def test_image_source_hides_channel_colours_for_mono_data(qtbot):
    viewer = _Viewer(
        np.zeros((12, 16, 18), dtype=np.uint8),
        metadata={"axes": "ZYX"},
    )
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.graph_view.select_node("input")

    assert widget._channel_color_control_count("input") == 0
    assert not any(
        name.startswith("channel_color_") for name in widget._parameter_widgets
    )


def test_image_source_shows_only_detected_multichannel_colours(qtbot):
    viewer = _Viewer(
        np.zeros((3, 12, 16, 18), dtype=np.uint16),
        metadata={"axes": "CZYX"},
    )
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.graph_view.select_node("input")

    controls = [
        name for name in widget._parameter_widgets if name.startswith("channel_color_")
    ]
    assert controls == ["channel_color_0", "channel_color_1", "channel_color_2"]


def test_image_source_hides_singleton_channel_axis(qtbot):
    viewer = _Viewer(
        np.zeros((1, 12, 16, 18), dtype=np.uint16),
        metadata={"axes": "CZYX"},
    )
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.graph_view.select_node("input")

    assert widget._channel_color_control_count("input") == 0
    assert not any(
        name.startswith("channel_color_") for name in widget._parameter_widgets
    )


def test_selecting_node_updates_inspection_layer(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.graph_view.select_node("input")

    inspect_layer = viewer.layers["VIPP Inspect"]
    assert inspect_layer.metadata["node_id"] == "input"
    assert inspect_layer.data.shape == viewer.layers["input volume"].data.shape
    assert not widget.pin_button.isHidden()
    assert widget.pin_button.text() == "Pin selected"


def test_widget_pins_threshold_as_labels(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.pin_node("threshold")

    pinned = viewer.layers["VIPP Pinned: Otsu Threshold"]
    assert pinned.metadata["napari_vipp_kind"] == "pinned"
    assert pinned.data.dtype == np.uint8
    assert widget.graph_view._cards["threshold"]._pinned
    assert (
        "border: 4px solid #facc15"
        in widget.graph_view._cards["threshold"].styleSheet()
    )
    assert widget.graph_view._cards["threshold"].pin_button.isHidden()
    assert not widget.pin_button.isHidden()


def test_label_pipeline_inspects_and_pins_integer_labels(qtbot):
    data = np.zeros((3, 12, 12), dtype=np.float32)
    data[:, 1:5, 1:5] = 10
    data[:, 7:11, 7:11] = 10
    viewer = _Viewer(data, metadata={"axes": "ZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    labels = widget.add_node_from_palette("label_connected_components")
    filtered = widget.add_node_from_palette("filter_labels_by_volume")
    widget.pipeline.set_param(filtered.id, "min_volume", 10)
    widget._connect_nodes("threshold", labels.id)
    widget._connect_nodes(labels.id, filtered.id)

    widget.inspect_node(filtered.id)

    inspect = viewer.layers["VIPP Inspect"]
    assert inspect.layer_type == "labels"
    assert inspect.data.dtype == np.int32
    assert inspect.metadata["data_kind"] == "labels"
    assert inspect.metadata["display_kind"] == "labels"
    assert widget.pipeline.output_states[filtered.id].kind == "label image"

    widget.pin_node(filtered.id)

    pinned = viewer.layers["VIPP Pinned: Filter Labels By Volume"]
    assert pinned.layer_type == "labels"
    assert pinned.data.dtype == np.int32
    assert pinned.metadata["data_kind"] == "labels"


def test_clear_border_node_preserves_label_display_type(qtbot):
    data = np.zeros((3, 12, 12), dtype=np.float32)
    data[:, 0:4, 0:4] = 10
    data[:, 6:10, 6:10] = 10
    viewer = _Viewer(data, metadata={"axes": "ZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    labels = widget.add_node_from_palette("label_connected_components")
    cleared = widget.add_node_from_palette("clear_border_objects")
    widget._connect_nodes("threshold", labels.id)
    widget._connect_nodes(labels.id, cleared.id)

    widget.inspect_node(cleared.id)

    inspect = viewer.layers["VIPP Inspect"]
    assert widget.pipeline.output_ports(cleared.id)[0].output_type == "labels"
    assert widget.pipeline.output_states[cleared.id].kind == "label image"
    assert inspect.layer_type == "labels"
    assert inspect.metadata["data_kind"] == "labels"


def test_clear_border_hides_lateral_choice_for_true_2d_input(qtbot):
    data = np.zeros((12, 12), dtype=np.float32)
    data[0:4, 0:4] = 10
    data[6:10, 6:10] = 10
    viewer = _Viewer(data, metadata={"axes": "YX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    cleared = widget.add_node_from_palette("clear_border_objects")
    widget._connect_nodes("threshold", cleared.id)

    control = widget._parameter_widgets["boundary_mode"]
    choices = [control.combo.itemText(index) for index in range(control.combo.count())]

    assert choices == ["All spatial borders"]
    assert widget.pipeline.nodes[cleared.id].params["boundary_mode"] in choices
    assert widget._parameter_widgets["border_buffer"]._bounds.maximum == 11


def test_clear_border_offers_all_or_lateral_boundaries_for_z_stack(qtbot):
    data = np.zeros((3, 12, 12), dtype=np.float32)
    data[:, 0:4, 0:4] = 10
    viewer = _Viewer(data, metadata={"axes": "ZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    cleared = widget.add_node_from_palette("clear_border_objects")
    widget._connect_nodes("threshold", cleared.id)

    control = widget._parameter_widgets["boundary_mode"]
    choices = [control.combo.itemText(index) for index in range(control.combo.count())]

    assert choices == [
        "All spatial borders",
        "Lateral borders only (YX)",
    ]
    assert widget._parameter_widgets["border_buffer"]._bounds.maximum == 2

    control.combo.setCurrentText("Lateral borders only (YX)")
    buffer_control = widget._parameter_widgets["border_buffer"]
    assert buffer_control._bounds.maximum == 11
    buffer_control.value_box.setValue(10)

    control.combo.setCurrentText("All spatial borders")

    assert buffer_control._bounds.maximum == 2
    assert widget.pipeline.nodes[cleared.id].params["border_buffer"] == 2


def test_fill_holes_uses_contextual_2d_and_3d_controls(qtbot):
    data = np.zeros((3, 12, 12), dtype=np.float32)
    data[:, 2:10, 2:10] = 10
    data[1, 5, 5] = 0
    viewer = _Viewer(data, metadata={"axes": "ZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    filled = widget.add_node_from_palette("fill_holes")
    widget._connect_nodes("threshold", filled.id)

    mode_control = widget._parameter_widgets["spatial_mode"]
    size_control = widget._parameter_widgets["max_hole_size"]
    choices = [
        mode_control.combo.itemText(index)
        for index in range(mode_control.combo.count())
    ]
    size_label = widget.parameter_form.labelForField(size_control)
    note = widget._parameter_widgets["fill_holes_scope_note"]

    assert choices == [
        "Auto from axes",
        "2D per XY slice (advanced)",
        "3D ZYX volume",
    ]
    assert size_control._bounds.maximum == 3 * 12 * 12
    assert "volume (voxels)" in size_label.text()
    assert "Recommended for z-stacks" in note.text()

    mode_control.combo.setCurrentText("2D per XY slice (advanced)")

    assert size_control._bounds.maximum == 12 * 12
    assert "area (pixels)" in size_label.text()
    assert "Advanced mode" in note.text()
    assert "open to background along Z" in note.text()


def test_fill_holes_hides_3d_mode_for_true_2d_input(qtbot):
    data = np.zeros((12, 12), dtype=np.float32)
    data[2:10, 2:10] = 10
    viewer = _Viewer(data, metadata={"axes": "YX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    filled = widget.add_node_from_palette("fill_holes")
    widget._connect_nodes("threshold", filled.id)

    control = widget._parameter_widgets["spatial_mode"]
    choices = [control.combo.itemText(index) for index in range(control.combo.count())]

    assert choices == [
        "Auto from axes",
        "2D per XY slice (advanced)",
    ]
    assert (
        "connected YX image"
        in widget._parameter_widgets["fill_holes_scope_note"].text()
    )


def test_remove_small_objects_uses_observed_sizes_and_contextual_units(qtbot):
    data = np.zeros((3, 12, 12), dtype=np.float32)
    data[:, 1:5, 1:5] = 10
    data[1, 8, 8] = 10
    viewer = _Viewer(data, metadata={"axes": "ZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    filtered = widget.add_node_from_palette("remove_small_objects")
    widget._connect_nodes("threshold", filtered.id)

    size_control = widget._parameter_widgets["min_size"]
    size_label = widget.parameter_form.labelForField(size_control)
    incoming_mask = widget.pipeline.input_data_for_node(filtered.id)
    largest_3d = widget._largest_object_size(
        incoming_mask,
        3,
        "Face connected",
    )

    assert size_control._bounds.maximum == largest_3d
    assert size_control._bounds.logarithmic
    assert "volume (voxels)" in size_label.text()
    assert size_control.value_box.maximum() == 1_000_000_000

    mode_control = widget._parameter_widgets["spatial_mode"]
    mode_control.combo.setCurrentText("2D YX")

    largest_2d = widget._largest_object_size(
        incoming_mask,
        2,
        "Face connected",
    )
    assert size_control._bounds.maximum == largest_2d
    assert largest_2d < largest_3d
    assert "area (pixels)" in size_label.text()


def test_label_volume_controls_use_observed_object_sizes(qtbot):
    data = np.zeros((3, 12, 12), dtype=np.float32)
    data[:, 1:5, 1:5] = 10
    data[:, 7:11, 7:11] = 10
    viewer = _Viewer(data, metadata={"axes": "ZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    labels = widget.add_node_from_palette("label_connected_components")
    filtered = widget.add_node_from_palette("filter_labels_by_volume")
    widget._connect_nodes("threshold", labels.id)
    widget._connect_nodes(labels.id, filtered.id)

    minimum_control = widget._parameter_widgets["min_volume"]
    maximum_control = widget._parameter_widgets["max_volume"]
    incoming_labels = widget.pipeline.input_data_for_node(filtered.id)
    largest_3d = widget._largest_label_volume(incoming_labels, 3)

    assert largest_3d > 0
    assert minimum_control._bounds.maximum == largest_3d
    assert maximum_control._bounds.maximum == largest_3d
    assert minimum_control.slider.minimum() == 0
    assert minimum_control.slider.maximum() == 1000
    assert minimum_control.value_box.maximum() == 1_000_000_000

    minimum_control.slider.setValue(500)

    assert 1 <= widget.pipeline.nodes[filtered.id].params["min_volume"] <= 10

    widget.pipeline.set_param(filtered.id, "spatial_mode", "2D YX")
    widget.run_pipeline()
    largest_2d = widget._largest_label_volume(incoming_labels, 2)

    assert largest_2d < largest_3d
    assert minimum_control._bounds.maximum == largest_2d
    assert maximum_control._bounds.maximum == largest_2d

    minimum_control.value_box.setValue(1_000_000)

    assert minimum_control.value() == 1_000_000
    assert minimum_control.slider.maximum() == 1000
    assert minimum_control.slider.value() == 1000


def test_label_volume_histogram_tracks_filter_thresholds(qtbot):
    data = np.zeros((3, 12, 12), dtype=np.float32)
    data[:, 1:5, 1:5] = 10
    data[:, 7:11, 7:11] = 10
    viewer = _Viewer(data, metadata={"axes": "ZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    labels = widget.add_node_from_palette("label_connected_components")
    filtered = widget.add_node_from_palette("filter_labels_by_volume")
    widget._connect_nodes("threshold", labels.id)
    widget._connect_nodes(labels.id, filtered.id)

    incoming_labels = widget.pipeline.input_data_for_node(filtered.id)
    volumes = widget._label_volumes(incoming_labels, 3)

    assert not widget.label_volume_group.isHidden()
    assert widget.label_volume_log_checkbox.isChecked()
    assert widget.label_volume_log_checkbox.text() == "Log volume axis"
    assert widget.label_volume_plot._counts.sum() == volumes.size
    assert widget.label_volume_plot._x_scale == "log"
    assert widget.label_volume_plot._x_range == (0.0, float(volumes.max()))
    assert [
        (label, value) for label, value, _color in widget.label_volume_plot._markers
    ] == [("min", 10.0)]
    assert "objects" in widget.label_volume_summary.text()
    assert "voxels" in widget.label_volume_summary.text()

    widget.label_volume_log_checkbox.setChecked(False)

    assert widget.label_volume_plot._counts.sum() == volumes.size
    assert widget.label_volume_plot._x_scale == "linear"
    assert widget.label_volume_plot._x_range == (0.0, float(volumes.max()))

    widget._parameter_widgets["min_volume"].value_box.setValue(20)
    widget._parameter_widgets["max_volume"].value_box.setValue(50)

    assert [
        (label, value) for label, value, _color in widget.label_volume_plot._markers
    ] == [("min", 20.0), ("max", 50.0)]

    widget._parameter_widgets["max_volume"].value_box.setValue(0)

    assert [
        (label, value) for label, value, _color in widget.label_volume_plot._markers
    ] == [("min", 20.0)]

    widget.graph_view.select_node(labels.id)

    assert widget.label_volume_group.isHidden()


def test_pin_toggles_active_node_layer(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.pin_node("threshold")
    widget.pin_node("threshold")

    pinned_layers = [
        layer
        for layer in viewer.layers
        if layer.metadata.get("napari_vipp_kind") == "pinned"
    ]
    assert pinned_layers == []
    assert widget._active_pinned_node_id is None
    assert len(viewer.layers) == 2
    assert not widget.graph_view._cards["threshold"]._pinned
    assert widget.graph_view._cards["threshold"].pin_button.isHidden()
    assert widget.status_label.text() == "Unpinned 'Otsu Threshold'."


def test_nodes_without_parameters_hide_parameter_group(qtbot):
    viewer = _Viewer(np.zeros((16, 18), dtype=np.float32), metadata={"axes": "YX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("sobel_filter")
    widget._select_node(node.id)

    assert widget.parameter_group.isHidden()


def test_slice_wise_stack_node_shows_axis_notice(qtbot):
    viewer = _Viewer(np.zeros((4, 16, 18), dtype=np.float32), metadata={"axes": "ZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("canny_edges")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    notice = widget._parameter_widgets["operation_notice"]

    assert not widget.parameter_group.isHidden()
    assert "processes each YX slice independently" in notice.text()
    assert "Reorder Axes" in notice.text()


def test_slice_wise_stack_notice_can_be_only_parameter_content(qtbot):
    viewer = _Viewer(np.zeros((4, 16, 18), dtype=np.float32), metadata={"axes": "ZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("sobel_filter")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    assert not widget.parameter_group.isHidden()
    assert set(widget._parameter_widgets) == {"operation_notice"}


def test_slice_wise_notice_hides_for_2d_input(qtbot):
    viewer = _Viewer(np.zeros((16, 18), dtype=np.float32), metadata={"axes": "YX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("canny_edges")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    assert "operation_notice" not in widget._parameter_widgets
    assert not widget.parameter_group.isHidden()


def test_palette_has_bottom_scroll_slack(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    spacer = widget.palette.topLevelItem(widget.palette.topLevelItemCount() - 1)

    assert widget.palette.verticalScrollBarPolicy() == Qt.ScrollBarAlwaysOn
    assert spacer.text(0) == ""
    assert spacer.sizeHint(0).height() >= 36


def test_palette_uses_category_colors(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    image_data = _palette_category(widget, "Image Data")
    image_source = _palette_item(widget, "input")
    assert image_data.foreground(0).color().name() == category_color("Image Data")
    assert image_data.background(0).color().name() == category_tint("Image Data")
    assert image_source.foreground(0).color().name() == category_color("Image Data")

    filtering = _palette_category(widget, "Filtering")
    gaussian = _palette_item(widget, "gaussian_blur")

    assert filtering.foreground(0).color().name() == category_color("Filtering")
    assert filtering.background(0).color().name() == category_tint("Filtering")
    assert gaussian.foreground(0).color().name() == category_color("Filtering")

    label_operations = _palette_category(widget, "Label Operations")
    label_node = _palette_item(widget, "label_connected_components")
    assert label_operations.foreground(0).color().name() == category_color(
        "Label Operations"
    )
    assert label_node.foreground(0).color().name() == category_color("Label Operations")


def test_image_data_category_groups_source_axis_and_channel_nodes(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    image_data = _palette_category(widget, "Image Data")
    subgroup_names = {
        image_data.child(index).text(0) for index in range(image_data.childCount())
    }

    assert {
        "Source & Output",
        "Axes & Regions",
        "Channels & Composites",
        "Utilities",
        "Math & Logic",
    } <= subgroup_names

    source_output = _palette_child_by_text(image_data, "Source & Output")
    axes_regions = _palette_child_by_text(image_data, "Axes & Regions")
    channels = _palette_child_by_text(image_data, "Channels & Composites")
    utilities = _palette_child_by_text(image_data, "Utilities")
    math_logic = _palette_child_by_text(image_data, "Math & Logic")
    intensity = _palette_category(widget, "Intensity & Contrast")

    assert _palette_child_by_text(source_output, "Image Source")
    assert _palette_child_by_text(source_output, "Save Image")
    assert _palette_child_by_text(axes_regions, "Crop Stack")
    assert _palette_child_by_text(axes_regions, "Select Axis Slice")
    assert _palette_child_by_text(axes_regions, "Reorder Axes")
    assert _palette_child_by_text(axes_regions, "Set Pixel Size / Units")
    assert _palette_child_by_text(axes_regions, "Rescale Axes")
    assert _palette_child_by_text(channels, "Extract Channel")
    assert _palette_child_by_text(channels, "Combine Channels")
    assert _palette_child_by_text(channels, "Split Channels")
    assert _palette_child_by_text(channels, "Composite \u2192 RGB")
    assert _palette_child_by_text(utilities, "Convert Dtype")
    assert _palette_child_by_text(intensity, "Rescale Intensity")
    assert _palette_child_by_text(intensity, "Normalize")
    assert _palette_child_by_text(intensity, "Clip")
    assert _palette_child_by_text(intensity, "Linear Scale + Offset")
    assert _palette_child_by_text(intensity, "Gamma Correction")
    assert _palette_child_by_text(math_logic, "Calculate New Image")
    assert _palette_child_by_text(math_logic, "Add")
    assert _palette_child_by_text(math_logic, "Logical XOR")


def test_set_pixel_size_uses_numeric_entries_without_sliders(qtbot):
    viewer = _Viewer(np.zeros((3, 16, 18), dtype=np.float32))
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("set_pixel_size")
    widget._connect_nodes("input", node.id)

    x_control = widget._parameter_widgets["x_size"]
    y_control = widget._parameter_widgets["y_size"]
    z_control = widget._parameter_widgets["z_size"]

    assert not hasattr(x_control, "slider")
    assert not hasattr(y_control, "slider")
    assert not hasattr(z_control, "slider")

    x_control.value_box.setValue(0.25)

    assert widget.pipeline.nodes[node.id].params["x_size"] == 0.25


def test_rescale_axes_can_lock_xy_scale(qtbot):
    viewer = _Viewer(np.zeros((3, 16, 18), dtype=np.float32))
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("rescale_axes")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    lock = widget._parameter_widgets["lock_xy"]
    x_scale = widget._parameter_widgets["x_scale"]
    y_scale = widget._parameter_widgets["y_scale"]
    z_scale = widget._parameter_widgets["z_scale"]

    assert lock.checkbox.isChecked()
    assert z_scale is not None

    x_scale.value_box.setValue(2.5)

    assert widget.pipeline.nodes[node.id].params["x_scale"] == 2.5
    assert widget.pipeline.nodes[node.id].params["y_scale"] == 2.5
    assert y_scale.value() == 2.5

    y_scale.value_box.setValue(0.75)

    assert widget.pipeline.nodes[node.id].params["y_scale"] == 0.75
    assert widget.pipeline.nodes[node.id].params["x_scale"] == 0.75
    assert x_scale.value() == 0.75

    lock.checkbox.setChecked(False)
    x_scale.value_box.setValue(1.25)

    assert widget.pipeline.nodes[node.id].params["lock_xy"] is False
    assert widget.pipeline.nodes[node.id].params["x_scale"] == 1.25
    assert widget.pipeline.nodes[node.id].params["y_scale"] == 0.75


def test_rescale_axes_uses_numeric_entry_without_sliders(qtbot):
    viewer = _Viewer(np.zeros((3, 16, 18), dtype=np.float32))
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("rescale_axes")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    x_scale = widget._parameter_widgets["x_scale"]
    assert not hasattr(x_scale, "slider")
    assert x_scale.layout().spacing() == 3
    assert x_scale.value_box.minimumWidth() == 112
    assert x_scale.value_box.maximumWidth() == 122
    assert x_scale.value_box.lineEdit().alignment() == Qt.AlignCenter
    assert x_scale.value_box.lineEdit().textMargins().left() == 0
    assert widget._parameter_widgets["x_scale_reset"].width() == 20

    x_scale.value_box.setValue(20.25)

    assert widget.pipeline.nodes[node.id].params["x_scale"] == 20.25


def test_float_spinners_accept_decimal_point_or_comma(qtbot):
    viewer = _Viewer(np.zeros((16, 18), dtype=np.float32))
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("rescale_axes")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)
    value_box = widget._parameter_widgets["x_scale"].value_box

    value_box.lineEdit().setText("1,23")
    value_box.interpretText()
    assert np.isclose(value_box.value(), 1.23)

    value_box.lineEdit().setText("2.34")
    value_box.interpretText()
    assert np.isclose(value_box.value(), 2.34)


def test_rescale_axes_labels_show_mapped_axis_sizes(qtbot):
    viewer = _Viewer(
        np.zeros((3, 12, 96, 128), dtype=np.float32),
        metadata={"axes": "CZYX"},
    )
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("rescale_axes")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    z_control = widget._parameter_widgets["z_scale"]
    z_control.value_box.setValue(2.0)
    widget._debounce_timer.stop()
    widget.run_pipeline()
    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and not widget._pipeline_run_pending
            and widget.pipeline.outputs.get(node.id) is not None
        ),
        timeout=30_000,
    )

    label = widget.parameter_form.labelForField(z_control)
    assert label.text() == "Z scale factor (12 -> 24)"
    assert widget.pipeline.outputs[node.id].shape == (3, 24, 96, 128)
    assert _metadata_value(widget, "Dimensions") == "c=3, z=24, y=96, x=128"


def test_rescale_axes_labels_follow_reordered_spatial_semantics(qtbot):
    viewer = _Viewer(
        np.zeros((3, 12, 96, 128), dtype=np.float32),
        metadata={"axes": "CZYX"},
    )
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    reorder = widget.add_node_from_palette("reorder_axes")
    rescale = widget.add_node_from_palette("rescale_axes")
    widget.pipeline.set_param(reorder.id, "order", "CYZX")
    widget._connect_nodes("input", reorder.id)
    widget._connect_nodes(reorder.id, rescale.id)
    widget.graph_view.select_node(rescale.id)

    z_control = widget._parameter_widgets["z_scale"]
    z_control.value_box.setValue(2.0)
    widget._debounce_timer.stop()
    widget.run_pipeline()
    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and not widget._pipeline_run_pending
            and widget.pipeline.outputs.get(rescale.id) is not None
        ),
        timeout=30_000,
    )

    label = widget.parameter_form.labelForField(z_control)
    assert label.text() == "Z scale factor (96 -> 192)"
    assert widget.pipeline.outputs[rescale.id].shape == (3, 192, 12, 128)
    assert _metadata_value(widget, "Dimensions") == "c=3, z=192, y=12, x=128"


def test_rescale_axes_supports_output_size_mode_and_axis_reset(qtbot):
    viewer = _Viewer(
        np.zeros((12, 96, 128), dtype=np.float32),
        metadata={"axes": "ZYX"},
    )
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("rescale_axes")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    mode = widget._parameter_widgets["resize_mode"]
    mode.combo.setCurrentIndex(mode.combo.findData("Output size"))
    x_size = widget._parameter_widgets["x_size"]
    y_size = widget._parameter_widgets["y_size"]
    assert x_size.value() == 128
    assert y_size.value() == 96

    x_size.value_box.setValue(256)
    assert widget.pipeline.nodes[node.id].params["x_size"] == 256
    assert widget.pipeline.nodes[node.id].params["y_size"] == 192

    widget._parameter_widgets["x_size_reset"].click()
    assert widget.pipeline.nodes[node.id].params["x_size"] == 128
    assert widget.pipeline.nodes[node.id].params["y_size"] == 96


def test_rescale_axes_auto_interpolation_names_resolved_method(qtbot):
    viewer = _Viewer(np.zeros((16, 18), dtype=np.float32))
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("rescale_axes")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    interpolation = widget._parameter_widgets["interpolation"]
    assert interpolation.combo.currentData() == "Auto"
    assert interpolation.combo.currentText() == "Auto - Linear"


def test_project_image_uses_contextual_axis_dropdown(qtbot):
    data = np.zeros((2, 3, 4, 16, 18), dtype=np.float32)
    viewer = _Viewer(data, metadata={"axes": "TCZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("project_image")
    widget._connect_nodes("input", node.id)
    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and not widget._pipeline_run_pending
            and widget.pipeline.outputs.get(node.id) is not None
        ),
        timeout=30_000,
    )

    control = widget._parameter_widgets["axes"]
    choices = [control.combo.itemText(index) for index in range(control.combo.count())]
    values = [control.combo.itemData(index) for index in range(control.combo.count())]

    assert choices[0] == "Auto (Z if present)"
    assert "T axis (time, size 2)" in choices
    assert "C axis (channel, size 3)" in choices
    assert "Z axis (space, size 4)" in choices
    assert "Y axis (space, size 16)" in choices
    assert "X axis (space, size 18)" in choices
    assert "All non-YX spatial axes" in choices
    assert values == [
        "auto",
        "axis:0",
        "axis:1",
        "axis:2",
        "axis:3",
        "axis:4",
        "non_yx_spatial",
    ]

    control.combo.setCurrentText("Z axis (space, size 4)")

    assert widget.pipeline.nodes[node.id].params["axes"] == "axis:2"


def test_filtering_and_segmentation_categories_are_grouped(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    filtering = _palette_category(widget, "Filtering")
    filtering_subgroups = {
        filtering.child(index).text(0) for index in range(filtering.childCount())
    }
    assert {
        "Smoothing & Denoising",
        "Edge & Detail",
        "Background Correction",
    } <= filtering_subgroups

    smoothing = _palette_child_by_text(filtering, "Smoothing & Denoising")
    edge_detail = _palette_child_by_text(filtering, "Edge & Detail")
    background = _palette_child_by_text(filtering, "Background Correction")
    assert _palette_child_by_text(smoothing, "Gaussian Blur")
    assert _palette_child_by_text(smoothing, "Non-Local Means")
    assert _palette_child_by_text(edge_detail, "Difference of Gaussians")
    assert _palette_child_by_text(edge_detail, "Sobel Edges")
    assert _palette_child_by_text(edge_detail, "Canny Edges")
    assert _palette_child_by_text(background, "Rolling-Ball Background")
    assert _palette_child_by_text(background, "Subtract Background")

    projection = _palette_category(widget, "Projection")
    assert _palette_child_by_text(projection, "Maximum Projection")
    assert _palette_child_by_text(projection, "Project Image")
    assert _palette_child_by_text(projection, "Orthogonal Projection")

    segmentation = _palette_category(widget, "Segmentation")
    segmentation_subgroups = {
        segmentation.child(index).text(0) for index in range(segmentation.childCount())
    }
    assert {
        "Global Thresholds",
        "Local Thresholds",
        "Object Separation",
    } <= segmentation_subgroups
    assert "Edge-Based" not in segmentation_subgroups

    global_thresholds = _palette_child_by_text(segmentation, "Global Thresholds")
    local_thresholds = _palette_child_by_text(segmentation, "Local Thresholds")
    object_separation = _palette_child_by_text(segmentation, "Object Separation")
    assert _palette_child_by_text(global_thresholds, "Otsu Threshold")
    assert _palette_child_by_text(global_thresholds, "Li Threshold")
    assert _palette_child_by_text(global_thresholds, "Hysteresis Threshold")
    assert _palette_child_by_text(local_thresholds, "Adaptive Gaussian Threshold")
    assert _palette_child_by_text(local_thresholds, "Sauvola Threshold")
    assert _palette_child_by_text(object_separation, "Auto Watershed From Mask")
    assert _palette_child_by_text(object_separation, "Euclidean Distance Transform")
    assert _palette_child_by_text(object_separation, "H-Maxima Markers")
    assert _palette_child_by_text(object_separation, "Marker-Controlled Watershed")
    assert _palette_child_by_text(object_separation, "Expand Labels")
    assert object_separation.child(0).text(0) == "Auto Watershed From Mask"


def test_global_threshold_scope_control_hides_for_2d_input(qtbot):
    viewer = _Viewer(np.zeros((16, 18), dtype=np.float32))
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("li_threshold")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    assert "threshold_scope" not in widget._parameter_widgets
    assert widget.parameter_group.isHidden()
    assert not widget.rescale_input_histogram_group.isHidden()
    assert widget.rescale_input_histogram_scope_row.isHidden()
    assert widget.rescale_input_histogram_group.title() == "Input Histogram"


def test_global_threshold_scope_control_shows_for_stack_input(qtbot):
    viewer = _Viewer(np.zeros((3, 16, 18), dtype=np.float32), metadata={"axes": "ZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("li_threshold")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    assert "threshold_scope" in widget._parameter_widgets
    assert not widget.parameter_group.isHidden()
    control = widget._parameter_widgets["threshold_scope"]
    label = widget.parameter_form.labelForField(control)
    assert label.text() == "Threshold uses"
    assert control.combo.itemText(0) == "Stack histogram"
    assert control.combo.itemText(1) == "Slice histogram"
    assert widget.pipeline.nodes[node.id].params["threshold_scope"] == (
        "Stack histogram"
    )


def test_auto_watershed_hides_spatial_mode_for_2d_input(qtbot):
    viewer = _Viewer(np.zeros((16, 18), dtype=np.float32), metadata={"axes": "YX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("auto_watershed_from_mask")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    assert "spatial_mode" not in widget._parameter_widgets


def test_auto_watershed_shows_spatial_mode_for_z_stack_input(qtbot):
    viewer = _Viewer(np.zeros((3, 16, 18), dtype=np.float32), metadata={"axes": "ZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("auto_watershed_from_mask")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    assert "spatial_mode" in widget._parameter_widgets
    control = widget._parameter_widgets["spatial_mode"]
    assert control.combo.itemText(0) == "Auto from axes - using 3D ZYX"


def test_watershed_h_parameter_has_sane_upper_bound(qtbot):
    viewer = _Viewer(np.zeros((3, 16, 18), dtype=np.float32), metadata={"axes": "ZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    for operation_id in ("h_maxima_markers", "auto_watershed_from_mask"):
        node = widget.add_node_from_palette(operation_id)
        widget._connect_nodes("input", node.id)
        widget.graph_view.select_node(node.id)
        h_widget = widget._parameter_widgets["h"]
        assert np.isclose(h_widget._from_slider(h_widget.slider.maximum()), 5.0)
        assert float(h_widget.value_box.maximum()) >= 1_000_000.0
        h_widget.value_box.setValue(7.5)
        assert np.isclose(float(h_widget.value_box.value()), 7.5)
        label = widget.parameter_form.labelForField(h_widget)
        assert label.text() == "H / prominence in px/voxels (0 = local maxima)"


def test_watershed_h_parameter_shows_units_and_tuning_note(qtbot):
    viewer = _Viewer(np.zeros((3, 16, 18), dtype=np.float32), metadata={"axes": "ZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    for operation_id in ("h_maxima_markers", "auto_watershed_from_mask"):
        node = widget.add_node_from_palette(operation_id)
        widget._connect_nodes("input", node.id)
        widget.graph_view.select_node(node.id)

        note = widget._parameter_widgets.get("operation_notice")
        assert note is not None
        text = note.text().lower()
        assert "pixels/voxels" in text
        assert "0 to 2" in text
        assert "local maxima" in text


def test_marker_controlled_watershed_shows_input_guide_note(qtbot):
    viewer = _Viewer(np.zeros((3, 16, 18), dtype=np.float32), metadata={"axes": "ZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("marker_controlled_watershed")
    widget.graph_view.select_node(node.id)

    note = widget._parameter_widgets.get("operation_notice")
    assert note is not None
    text = note.text().lower()
    assert "image / distance" in text
    assert "markers" in text
    assert "mask" in text


def test_global_threshold_input_histogram_shows_chosen_threshold(qtbot):
    data = np.zeros((2, 10, 10), dtype=np.float32)
    data[0, :, 5:] = 10.0
    data[1, :, :5] = 100.0
    data[1, :, 5:] = 110.0
    viewer = _Viewer(data, metadata={"axes": "ZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("otsu_threshold")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    assert not widget.rescale_input_histogram_group.isHidden()
    assert widget.rescale_input_histogram_scope_row.isHidden()
    assert widget.rescale_input_histogram_group.title() == (
        "Input Histogram (Stack histogram)"
    )
    stack_markers = {
        label: value
        for label, value, _color in widget.rescale_input_histogram_plot._markers
    }
    assert "threshold" in stack_markers

    widget._parameter_widgets["threshold_scope"].combo.setCurrentText("Slice histogram")

    assert widget.rescale_input_histogram_group.title() == (
        "Input Histogram (Slice histogram)"
    )
    slice_markers = {
        label: value
        for label, value, _color in widget.rescale_input_histogram_plot._markers
    }
    assert "threshold" in slice_markers
    assert not np.isclose(stack_markers["threshold"], slice_markers["threshold"])


def test_palette_search_filters_nodes_fuzzily(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.palette_search.setText("gblr")

    assert not _palette_item(widget, "gaussian_blur").isHidden()
    assert _palette_item(widget, "median_filter").isHidden()
    assert widget.palette._no_results_item.isHidden()

    widget.palette_search.clear()

    assert not _palette_item(widget, "median_filter").isHidden()


def test_palette_search_shows_no_result_message(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.palette_search.setText("zzzz")

    assert not widget.palette._no_results_item.isHidden()
    assert _palette_item(widget, "gaussian_blur").isHidden()


def test_dock_widget_can_shrink_vertically(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.minimumSizeHint().height() <= 120
    assert widget.sizeHint().height() >= 560
    assert widget.graph_view.minimumHeight() <= 80
    assert widget.histogram_plot.minimumHeight() >= 120
    assert widget.splitter.minimumHeight() == 0
    assert isinstance(widget.inspector_panel, QScrollArea)
    assert widget.inspector_panel.minimumHeight() == 0


def test_inspector_shows_histogram_before_metadata(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    layout = widget.inspector_content.layout()

    assert layout.indexOf(widget.label_volume_group) < layout.indexOf(
        widget.histogram_group
    )
    assert layout.indexOf(widget.histogram_group) < layout.indexOf(
        widget.metadata_group
    )


def test_side_panels_can_be_collapsed_and_restored(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    assert not widget.palette_panel.isHidden()
    assert not widget.inspector_panel.isHidden()
    assert widget.left_panel_toggle._expanded
    assert widget.right_panel_toggle._expanded
    assert widget.left_panel_toggle._direction() == -1
    assert widget.right_panel_toggle._direction() == 1

    widget.left_panel_toggle.click()
    widget.right_panel_toggle.click()

    assert widget.palette_panel.isHidden()
    assert widget.inspector_panel.isHidden()
    assert not widget.left_panel_toggle._expanded
    assert not widget.right_panel_toggle._expanded
    assert widget.left_panel_toggle.toolTip() == "Show node library"
    assert widget.right_panel_toggle.toolTip() == "Show inspector"

    widget.left_panel_toggle.click()
    widget.right_panel_toggle.click()

    assert not widget.palette_panel.isHidden()
    assert not widget.inspector_panel.isHidden()
    assert widget.left_panel_toggle._expanded
    assert widget.right_panel_toggle._expanded
    assert widget.left_panel_toggle.toolTip() == "Hide node library"
    assert widget.right_panel_toggle.toolTip() == "Hide inspector"


def test_dock_widget_chrome_is_restored_when_hosted(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    window = QMainWindow()
    dock = QDockWidget()
    qtbot.addWidget(window)
    qtbot.addWidget(dock)
    dock.setTitleBarWidget(QWidget())
    dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
    dock.setWidget(widget)
    window.addDockWidget(Qt.BottomDockWidgetArea, dock)

    widget._ensure_dock_widget_chrome()

    assert dock.titleBarWidget() is None
    assert dock.windowTitle() == "VIPP Workflow"
    assert dock.features() & QDockWidget.DockWidgetMovable
    assert dock.features() & QDockWidget.DockWidgetFloatable
    assert dock.features() & QDockWidget.DockWidgetClosable
    assert widget._dock_chrome_configured


def test_floating_dock_window_has_standard_maximize_controls(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    window = QMainWindow()
    dock = QDockWidget()
    title_bar = QWidget()
    qtbot.addWidget(window)
    dock.setTitleBarWidget(title_bar)
    dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
    dock.setWidget(widget)
    window.addDockWidget(Qt.BottomDockWidgetArea, dock)
    window.show()

    def reset_floating_flags_like_napari(visible):
        if visible and dock.isFloating():
            with QSignalBlocker(dock):
                dock.setTitleBarWidget(None)

    dock.visibilityChanged.connect(reset_floating_flags_like_napari)
    QApplication.processEvents()
    widget._ensure_dock_widget_chrome()
    dock.setFloating(True)
    QApplication.processEvents()
    widget._configure_floating_dock_window()

    assert dock.titleBarWidget() is None
    assert dock.features() & QDockWidget.DockWidgetMovable
    assert dock.features() & QDockWidget.DockWidgetFloatable
    assert dock.features() & QDockWidget.DockWidgetClosable
    assert dock.windowFlags() & Qt.WindowMaximizeButtonHint
    assert dock.windowFlags() & Qt.WindowMinimizeButtonHint
    assert dock.windowFlags() & Qt.WindowCloseButtonHint
    assert dock.windowFlags() & Qt.WindowType_Mask == Qt.Window
    assert widget._dock_chrome_configured
    assert widget._dock_window_behavior_configured

    dock.hide()
    dock.show()
    qtbot.waitUntil(lambda: bool(dock.windowFlags() & Qt.WindowMaximizeButtonHint))

    assert dock.windowFlags() & Qt.WindowType_Mask == Qt.Window


def test_floating_dock_title_double_click_toggles_maximized(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    window = QMainWindow()
    dock = QDockWidget()
    qtbot.addWidget(window)
    dock.setWidget(widget)
    window.addDockWidget(Qt.BottomDockWidgetArea, dock)
    window.show()
    widget._ensure_dock_widget_chrome()
    dock.setFloating(True)
    widget._configure_floating_dock_window()

    def double_click_title_bar():
        event = QMouseEvent(
            QEvent.NonClientAreaMouseButtonDblClick,
            QPointF(4, 4),
            QPointF(4, 4),
            QPointF(4, 4),
            Qt.LeftButton,
            Qt.LeftButton,
            Qt.NoModifier,
        )
        QApplication.sendEvent(dock, event)

    double_click_title_bar()

    assert dock.isFloating()
    assert dock.isMaximized()

    double_click_title_bar()

    assert dock.isFloating()
    assert not dock.isMaximized()


def test_dock_widget_chrome_is_not_rewritten_after_configured(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    window = QMainWindow()
    dock = QDockWidget()
    qtbot.addWidget(window)
    qtbot.addWidget(dock)
    dock.setWidget(widget)
    window.addDockWidget(Qt.BottomDockWidgetArea, dock)

    widget._ensure_dock_widget_chrome()
    dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
    widget._ensure_dock_widget_chrome()

    assert dock.features() == QDockWidget.NoDockWidgetFeatures


def test_initial_bottom_dock_size_is_applied_once(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    window = QMainWindow()
    dock = QDockWidget()
    window.resize(1200, 900)
    qtbot.addWidget(window)
    qtbot.addWidget(dock)
    dock.setWidget(widget)
    window.addDockWidget(Qt.BottomDockWidgetArea, dock)

    widget._apply_initial_dock_size()
    first_height = dock.height()
    widget._initial_dock_size_applied = True
    dock.resize(dock.width(), 120)
    widget._apply_initial_dock_size()

    assert widget._initial_dock_size_applied
    assert first_height >= 300
    assert dock.height() == 120


def test_initial_dock_size_is_not_applied_while_floating(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    dock = QDockWidget()
    qtbot.addWidget(dock)
    dock.setWidget(widget)
    dock.setFloating(True)

    widget._apply_initial_dock_size()

    assert not widget._initial_dock_size_applied


def test_histogram_updates_for_selected_node(qtbot):
    data = np.arange(4 * 16 * 18, dtype=np.uint8).reshape(4, 16, 18)
    viewer = _Viewer(data)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.histogram_group.title() == "Output Histogram"
    assert not widget.histogram_scope_row.isHidden()
    assert widget.histogram_scope_combo.currentText() == "Slice"
    assert not widget.histogram_log_checkbox.isChecked()
    assert widget.histogram_plot._counts.size > 0

    widget.histogram_log_checkbox.setChecked(True)
    assert widget.histogram_plot._log_scale

    widget.graph_view.select_node("threshold")

    assert widget.histogram_plot._counts.size == 2


def test_histogram_scope_is_hidden_for_2d_outputs(qtbot):
    data = np.arange(16 * 18, dtype=np.uint8).reshape(16, 18)
    viewer = _Viewer(data, metadata={"axes": "YX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.graph_view.select_node("input")

    assert widget.histogram_group.title() == "Output Histogram"
    assert widget.histogram_scope_row.isHidden()
    assert widget.histogram_plot._counts.sum() == data.size


def test_histogram_can_switch_between_slice_and_stack(qtbot):
    data = np.zeros((2, 5, 6), dtype=np.uint8)
    data[1] = 200
    viewer = _Viewer(data)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.graph_view.select_node("input")

    assert not widget.histogram_scope_row.isHidden()
    assert widget.histogram_plot._counts.tolist() == [30.0]
    assert widget.histogram_plot._x_min_label == "0"
    assert widget.histogram_plot._x_max_label == "0"

    widget.histogram_scope_combo.setCurrentText("Stack")

    assert widget.histogram_plot._counts.size == 256
    assert widget.histogram_plot._counts.sum() == 60
    assert widget.histogram_plot._x_min_label == "0"
    assert widget.histogram_plot._x_max_label == "255"


def test_histogram_separates_multichannel_series(qtbot):
    data = np.zeros((2, 3, 4, 4, 5), dtype=np.uint8)
    data[:, 0] = 20
    data[:, 1] = 100
    data[:, 2] = 220
    viewer = _Viewer(data, metadata={"axes": "TCZYX"})
    viewer.dims.current_step = (1, 0, 2, 0, 0)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.graph_view.select_node("input")

    series = widget.histogram_plot._series_counts
    colors = widget.histogram_plot._series_colors

    assert series.shape == (3, 256)
    assert series.sum(axis=1).tolist() == [20.0, 20.0, 20.0]
    assert colors[0].blueF() > colors[0].redF()
    assert colors[1].greenF() > colors[1].redF()
    assert colors[2].redF() > colors[2].blueF()


def test_rescale_intensity_shows_input_and_output_histograms(qtbot):
    data = np.arange(256, dtype=np.uint8).reshape(1, 16, 16)
    viewer = _Viewer(data)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("rescale_intensity")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    assert widget.pipeline.outputs[node.id].dtype == np.uint8
    assert widget.pipeline.nodes[node.id].params["out_min"] == 0.0
    assert widget.pipeline.nodes[node.id].params["out_max"] == 255.0
    assert widget.pipeline.nodes[node.id].params["in_low_value"] == 0.0
    assert widget.pipeline.nodes[node.id].params["in_high_value"] == 255.0
    assert list(widget._parameter_widgets)[:4] == [
        "in_low_value",
        "in_high_value",
        "in_low_percentile",
        "in_high_percentile",
    ]
    assert not widget.rescale_input_histogram_group.isHidden()
    assert widget.rescale_input_histogram_scope_row.isHidden()
    assert widget.histogram_group.title() == "Output Histogram"
    assert widget.rescale_input_histogram_plot._counts.size == 256
    assert widget.histogram_plot._counts.size == 256
    assert [
        (label, value)
        for label, value, _color in widget.rescale_input_histogram_plot._markers
    ] == [("low", 0.0), ("high", 255.0)]

    widget.rescale_input_histogram_log_checkbox.setChecked(True)

    assert widget.rescale_input_histogram_plot._log_scale

    widget.graph_view.select_node("input")

    assert widget.rescale_input_histogram_group.isHidden()
    assert widget.histogram_group.title() == "Output Histogram"


def test_input_histogram_scope_switches_between_slice_and_stack(qtbot):
    data = np.zeros((2, 10, 10), dtype=np.uint8)
    data[1] = 255
    viewer = _Viewer(data, metadata={"axes": "ZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("binary_threshold")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    assert not widget.rescale_input_histogram_group.isHidden()
    assert not widget.rescale_input_histogram_scope_row.isHidden()
    assert widget.rescale_input_histogram_scope_combo.currentText() == (
        "Slice histogram"
    )
    assert widget.histogram_scope_combo.currentText() == "Slice"
    assert widget.rescale_input_histogram_plot._counts.sum() == 100.0
    assert widget.histogram_plot._counts.sum() == 100.0

    widget.rescale_input_histogram_scope_combo.setCurrentText("Stack histogram")

    assert widget.rescale_input_histogram_plot._counts.sum() == 200.0
    assert widget.histogram_scope_combo.currentText() == "Slice"
    assert widget.histogram_plot._counts.sum() == 100.0


def test_rescale_cutoff_values_and_percentiles_stay_in_sync(qtbot):
    data = np.arange(256, dtype=np.uint8).reshape(1, 16, 16)
    viewer = _Viewer(data)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("rescale_intensity")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    widget._parameter_widgets["in_low_percentile"].value_box.setValue(25.0)

    assert np.isclose(
        widget.pipeline.nodes[node.id].params["in_low_value"],
        63.75,
    )
    assert np.isclose(
        widget._parameter_widgets["in_low_value"].value(),
        63.75,
    )
    assert [
        (label, value)
        for label, value, _color in widget.rescale_input_histogram_plot._markers
    ][0] == ("low", 63.75)

    widget._parameter_widgets["in_high_value"].value_box.setValue(127.5)

    assert np.isclose(
        widget.pipeline.nodes[node.id].params["in_high_percentile"],
        50.19607843137255,
    )
    assert np.isclose(
        widget._parameter_widgets["in_high_percentile"].value(),
        50.2,
    )
    assert [
        (label, value)
        for label, value, _color in widget.rescale_input_histogram_plot._markers
    ][1] == ("high", 127.5)


def test_clip_intensity_shows_input_and_output_histograms_with_live_markers(qtbot):
    data = np.arange(100, dtype=np.uint16).reshape(1, 10, 10)
    viewer = _Viewer(data)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("clip_intensity")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    assert widget.pipeline.outputs[node.id].dtype == np.uint16
    assert widget.pipeline.nodes[node.id].params["minimum"] == 0.0
    assert widget.pipeline.nodes[node.id].params["maximum"] == 99.0
    assert not widget.rescale_input_histogram_group.isHidden()
    assert widget.histogram_group.title() == "Output Histogram"
    assert widget.rescale_input_histogram_plot._counts.sum() == 100.0
    assert widget.histogram_plot._counts.sum() == 100.0
    assert [
        (label, value)
        for label, value, _color in widget.rescale_input_histogram_plot._markers
    ] == [("min", 0.0), ("max", 99.0)]
    assert np.isclose(widget._parameter_widgets["minimum"]._bounds.minimum, 0.0)
    assert np.isclose(widget._parameter_widgets["maximum"]._bounds.maximum, 99.0)

    widget._parameter_widgets["minimum"].value_box.setValue(25.0)

    assert widget.pipeline.nodes[node.id].params["minimum"] == 25.0
    assert [
        (label, value)
        for label, value, _color in widget.rescale_input_histogram_plot._markers
    ][0] == ("min", 25.0)


def test_binary_threshold_shows_input_histogram_marker(qtbot):
    data = np.arange(256, dtype=np.uint8).reshape(1, 16, 16)
    viewer = _Viewer(data)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("binary_threshold")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    assert not widget.rescale_input_histogram_group.isHidden()
    assert widget.histogram_group.title() == "Output Histogram"

    widget._parameter_widgets["threshold"].value_box.setValue(128.0)

    assert [
        (label, value)
        for label, value, _color in widget.rescale_input_histogram_plot._markers
    ] == [("threshold", 128.0)]


def test_hysteresis_threshold_shows_input_histogram_markers(qtbot):
    viewer = _Viewer(np.arange(256, dtype=np.uint8).reshape(16, 16))
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("hysteresis_threshold")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    widget._parameter_widgets["low_threshold"].value_box.setValue(64.0)
    widget._parameter_widgets["high_threshold"].value_box.setValue(192.0)

    assert not widget.rescale_input_histogram_group.isHidden()
    assert [
        (label, value)
        for label, value, _color in widget.rescale_input_histogram_plot._markers
    ] == [("low", 64.0), ("high", 192.0)]


def test_selected_node_shows_output_metadata(qtbot):
    data = np.arange(4 * 16 * 18, dtype=np.uint16).reshape(4, 16, 18)
    viewer = _Viewer(data)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.graph_view.select_node("input")

    card_text = widget.graph_view._cards["input"].metadata_label.text()

    assert _metadata_value(widget, "Shape") == "4 x 16 x 18"
    assert _metadata_value(widget, "Axes") == "z(space), y(space), x(space)"
    assert _metadata_value(widget, "Dimensions") == "z=4, y=16, x=18"
    assert _metadata_value(widget, "Z slices") == "4"
    assert _metadata_value(widget, "Dtype") == "uint16"
    assert _metadata_value(widget, "Bit depth") == "16-bit integer"
    assert _metadata_value(widget, "Metadata source") == "inferred from array shape"
    assert "ZYX: 4 x 16 x 18 | uint16" in card_text
    assert "range" not in card_text


def test_ome_ngff_axes_metadata_is_displayed_without_guessing(qtbot):
    data = np.zeros((2, 3, 4, 5, 6), dtype=np.uint16)
    metadata = {
        "ome": {
            "version": "0.5",
            "multiscales": [
                {
                    "axes": [
                        {"name": "t", "type": "time", "unit": "second"},
                        {"name": "c", "type": "channel"},
                        {"name": "z", "type": "space", "unit": "micrometer"},
                        {"name": "y", "type": "space", "unit": "micrometer"},
                        {"name": "x", "type": "space", "unit": "micrometer"},
                    ],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [1, 1, 0.5, 0.2, 0.2]},
                                {
                                    "type": "translation",
                                    "translation": [0, 0, 10, 0, 0],
                                },
                            ],
                        }
                    ],
                }
            ],
        }
    }
    viewer = _Viewer(data, metadata=metadata)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.graph_view.select_node("input")

    assert (
        _metadata_value(widget, "Axes")
        == "t(time), c(channel), z(space), y(space), x(space)"
    )
    assert _metadata_value(widget, "Dimensions") == "t=2, c=3, z=4, y=5, x=6"
    assert "t=1 second, z=0.5 micrometer" in _metadata_value(
        widget,
        "Physical scale",
    )
    assert "z=10 micrometer" in _metadata_value(widget, "Origin")
    assert _metadata_value(widget, "Channels") == "3"
    assert _metadata_value(widget, "Timepoints") == "2"
    assert _metadata_value(widget, "Metadata source") == "OME-NGFF multiscales"


def test_composite_to_rgb_maps_channel_axis(qtbot):
    data = np.zeros((2, 3, 4, 5, 6), dtype=np.uint16)
    data[:, 0] = 1000
    data[:, 1] = 2000
    data[:, 2] = 3000
    viewer = _Viewer(data, metadata={"axes": "TCZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("composite_to_rgb")
    widget.pipeline.set_param(node.id, "channel_axis", 1)
    widget._connect_nodes("input", node.id)
    widget.run_pipeline()
    widget.graph_view.select_node(node.id)

    assert widget.pipeline.nodes[node.id].params["channel_axis"] == 1
    assert widget.pipeline.outputs[node.id].shape == (2, 4, 5, 6, 3)

    assert _metadata_value(widget, "Kind") == "RGB image"
    assert _metadata_value(widget, "Dimensions") == "t=2, z=4, y=5, x=6, rgb=3"
    inspect = viewer.layers["VIPP Inspect"]
    assert inspect.rgb
    assert inspect.metadata["display_rgb"] is True
    assert inspect.data.shape == (2, 4, 5, 6, 3)
    assert (
        "1. Composite \u2192 RGB: mapped channels to RGB" in widget.history_label.text()
    )


def test_composite_to_rgb_auto_channel_axis_remains_selectable(qtbot):
    data = np.zeros((3, 12, 16, 18), dtype=np.uint16)
    data[0] = 1000
    data[1] = 2000
    data[2] = 3000
    viewer = _Viewer(data, metadata={"axes": "CZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("composite_to_rgb")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    channel_axis = widget._parameter_widgets["channel_axis"]
    red_channel = widget._parameter_widgets["red_channel"]
    assert channel_axis.slider.minimum() == -1
    assert channel_axis.value_box.minimum() == -1
    assert red_channel.slider.minimum() == -1
    assert red_channel.value_box.minimum() == -1
    assert widget.pipeline.nodes[node.id].params["channel_axis"] == -1
    assert widget.pipeline.outputs[node.id].shape == (12, 16, 18, 3)
    assert widget.pipeline.outputs[node.id].max() == 1.0
    assert _metadata_value(widget, "Dimensions") == "z=12, y=16, x=18, rgb=3"
    inspect = viewer.layers["VIPP Inspect"]
    assert inspect.rgb
    assert inspect.metadata["display_rgb"] is True

    channel_axis.value_box.setValue(0)
    red_channel.value_box.setValue(0)
    widget._debounce_timer.stop()
    widget.run_pipeline()

    assert widget.pipeline.nodes[node.id].params["channel_axis"] == 0
    assert widget.pipeline.nodes[node.id].params["red_channel"] == 0
    assert widget.pipeline.outputs[node.id].max() == 1.0


def test_composite_to_rgb_and_input_share_z_slider_mapping(qtbot):
    data = np.zeros((3, 12, 16, 18), dtype=np.uint16)
    for z_index in range(data.shape[1]):
        data[:, z_index, z_index % data.shape[2], z_index % data.shape[3]] = 1000
    viewer = _Viewer(data, metadata={"axes": "CZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("composite_to_rgb")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    viewer.dims.current_step = (5, 0, 0)
    current_step = widget._current_step()
    assert current_step == (0, 5, 0, 0)
    input_first = make_preview(
        widget.pipeline.outputs["input"],
        mode="slice",
        current_step=current_step,
        state=widget.pipeline.output_states["input"],
    )
    rgb_first = make_preview(
        widget.pipeline.outputs[node.id],
        mode="slice",
        current_step=current_step,
        state=widget.pipeline.output_states[node.id],
    )

    viewer.dims.current_step = (9, 0, 0)
    current_step = widget._current_step()
    assert current_step == (0, 9, 0, 0)
    input_second = make_preview(
        widget.pipeline.outputs["input"],
        mode="slice",
        current_step=current_step,
        state=widget.pipeline.output_states["input"],
    )
    rgb_second = make_preview(
        widget.pipeline.outputs[node.id],
        mode="slice",
        current_step=current_step,
        state=widget.pipeline.output_states[node.id],
    )

    assert not np.array_equal(input_first, input_second)
    assert not np.array_equal(rgb_first, rgb_second)

    widget.graph_view.select_node("input")
    viewer.dims.current_step = (0, 7, 0, 0)
    assert widget._current_step() == (0, 7, 0, 0)


def test_composite_to_rgb_and_input_share_time_and_z_slider_mapping(qtbot):
    data = np.zeros((5, 3, 12, 16, 18), dtype=np.uint16)
    for time_index in range(data.shape[0]):
        for z_index in range(data.shape[2]):
            y_index = (time_index + z_index) % data.shape[3]
            x_index = (2 * time_index + z_index) % data.shape[4]
            data[time_index, :, z_index, y_index, x_index] = 1000
    viewer = _Viewer(data, metadata={"axes": "TCZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("composite_to_rgb")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    viewer.dims.current_step = (3, 5, 0, 0)
    current_step = widget._current_step()
    assert current_step == (3, 0, 5, 0, 0)
    input_first = make_preview(
        widget.pipeline.outputs["input"],
        mode="slice",
        current_step=current_step,
        state=widget.pipeline.output_states["input"],
    )
    rgb_first = make_preview(
        widget.pipeline.outputs[node.id],
        mode="slice",
        current_step=current_step,
        state=widget.pipeline.output_states[node.id],
    )

    viewer.dims.current_step = (1, 9, 0, 0)
    current_step = widget._current_step()
    assert current_step == (1, 0, 9, 0, 0)
    input_second = make_preview(
        widget.pipeline.outputs["input"],
        mode="slice",
        current_step=current_step,
        state=widget.pipeline.output_states["input"],
    )
    rgb_second = make_preview(
        widget.pipeline.outputs[node.id],
        mode="slice",
        current_step=current_step,
        state=widget.pipeline.output_states[node.id],
    )

    assert not np.array_equal(input_first, input_second)
    assert not np.array_equal(rgb_first, rgb_second)

    widget.graph_view.select_node("input")
    viewer.dims.current_step = (4, 0, 7, 0, 0)
    assert widget._current_step() == (4, 0, 7, 0, 0)


def test_split_channels_thumbnail_channel_selector(qtbot, monkeypatch):
    data = np.zeros((3, 2, 4, 5), dtype=np.uint16)
    data[0] = 10
    data[1] = 20
    data[2] = 30
    viewer = _Viewer(data, metadata={"axes": "CZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    split = widget.add_node_from_palette("split_channels")
    widget._connect_nodes("input", split.id)
    widget.run_pipeline()
    widget.graph_view.select_node(split.id)

    control = widget._parameter_widgets["preview_channel"]
    assert control.slider.minimum() == 0
    assert control.slider.maximum() == 2

    calls = []

    def fake_make_preview(
        data,
        mode,
        current_step,
        state=None,
        channel_colors=None,
        contrast_mode="Percentile",
    ):
        arr = np.asarray(data)
        calls.append((tuple(arr.shape), int(arr.max())))
        return None

    monkeypatch.setattr("napari_vipp._widget.make_preview", fake_make_preview)
    widget.pipeline.set_param(split.id, "preview_channel", 2)
    widget._update_thumbnails()

    assert ((2, 4, 5), 30) in calls


def test_split_threshold_channel_drag_connects_to_label_node(qtbot):
    data = np.zeros((3, 4, 16, 18), dtype=np.uint16)
    data[0, :, 3:12, 4:14] = 5000
    viewer = _Viewer(data, metadata={"axes": "CZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    split = widget.add_node_from_palette("split_channels")
    labels = widget.add_node_from_palette("label_connected_components")
    widget._connect_nodes("threshold", split.id)

    source_port = widget.graph_view._proxies[split.id].output_port_at(0)
    target_port = widget.graph_view._proxies[labels.id].input_port_at(0)

    assert source_port is not None
    assert target_port is not None
    assert source_port.data_type == "mask"

    widget.graph_view.begin_connection(
        source_port,
        source_port.mapToScene(QPointF(0, 0)),
    )
    widget.graph_view.complete_connection(target_port)

    assert any(
        connection.source_id == split.id
        and connection.source_port == 0
        and connection.target_id == labels.id
        for connection in widget.pipeline.connections
    )
    assert widget.pipeline.outputs[labels.id] is not None
    assert widget.pipeline.outputs[labels.id].dtype == np.int32


def test_combine_channels_accepts_multiple_connected_inputs(qtbot):
    data = np.zeros((2, 3, 4, 5, 6), dtype=np.uint16)
    data[:, 0] = 1000
    data[:, 1] = 2000
    viewer = _Viewer(data, metadata={"axes": "TCZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    first = widget.add_node_from_palette("extract_channel")
    second = widget.add_node_from_palette("extract_channel")
    composite = widget.add_node_from_palette("combine_channels")
    widget._connect_nodes("input", first.id)
    widget._connect_nodes("input", second.id)
    widget.pipeline.set_param(first.id, "channel", 0)
    widget.pipeline.set_param(second.id, "channel", 1)
    widget._connect_nodes(first.id, composite.id)
    widget._connect_nodes(second.id, composite.id)

    composite_ports = widget.graph_view._proxies[composite.id].input_ports
    assert len(composite_ports) == 2
    assert composite_ports[0].label == "Channel 1: Red"
    assert composite_ports[1].label == "Channel 2: Green"
    assert [
        connection.target_port
        for connection in widget.pipeline.connections
        if connection.target_id == composite.id
    ] == [0, 1]

    widget.run_pipeline()
    widget.graph_view.select_node(composite.id)

    assert widget.pipeline.outputs[composite.id].shape == (2, 2, 4, 5, 6)
    assert (
        len(
            [
                connection
                for connection in widget.pipeline.connections
                if connection.target_id == composite.id
            ]
        )
        == 2
    )
    assert _metadata_value(widget, "Kind") == "multi-channel image"
    assert _metadata_value(widget, "Dimensions") == "t=2, c=2, z=4, y=5, x=6"
    assert (
        "1. Extract Channel: selected channel 0\n"
        "2. Combine Channels: combined 2 inputs as channels"
        in widget.history_label.text()
    )


def test_combine_channels_input_count_and_colours_update_ports(qtbot):
    data = np.zeros((2, 3, 4, 5, 6), dtype=np.uint16)
    viewer = _Viewer(data, metadata={"axes": "TCZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    composite = widget.add_node_from_palette("combine_channels")
    widget.graph_view.select_node(composite.id)

    widget._on_combine_channels_input_count_changed(4)
    widget._on_channel_color_changed(2, "Yellow")

    node = widget.pipeline.nodes[composite.id]
    ports = widget.graph_view._proxies[composite.id].input_ports

    assert node.params["input_count"] == 4
    assert node.params["channel_colors"] == "Red,Green,Yellow,Magenta"
    assert len(ports) == 4
    assert ports[2].label == "Channel 3: Yellow"
    assert ports[2].accent_color == "#eab308"


def test_combine_channels_colour_change_refreshes_thumbnail_palette(
    qtbot,
    monkeypatch,
):
    data = np.zeros((2, 3, 4, 5, 6), dtype=np.uint16)
    data[:, 0] = 1000
    data[:, 1] = 2000
    viewer = _Viewer(data, metadata={"axes": "TCZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    first = widget.add_node_from_palette("extract_channel")
    second = widget.add_node_from_palette("extract_channel")
    composite = widget.add_node_from_palette("combine_channels")
    widget._connect_nodes("input", first.id)
    widget._connect_nodes("input", second.id)
    widget.pipeline.set_param(first.id, "channel", 0)
    widget.pipeline.set_param(second.id, "channel", 1)
    widget._connect_nodes(first.id, composite.id)
    widget._connect_nodes(second.id, composite.id)
    widget.run_pipeline()
    widget.graph_view.select_node(composite.id)

    calls = []

    def fake_make_preview(
        data,
        mode,
        current_step,
        state=None,
        channel_colors=None,
        contrast_mode="Percentile",
    ):
        if channel_colors is not None:
            calls.append(list(channel_colors))
        return None

    monkeypatch.setattr("napari_vipp._widget.make_preview", fake_make_preview)

    widget._on_channel_color_changed(1, "Cyan")

    assert ["Red", "Cyan"] in calls


def test_select_axis_slice_updates_metadata_axes(qtbot):
    data = np.zeros((2, 3, 4, 5, 6), dtype=np.uint16)
    viewer = _Viewer(data, metadata={"axes": "TCZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("select_axis_slice")
    widget._connect_nodes("input", node.id)
    control = widget._parameter_widgets["axis_slice"]

    assert widget.pipeline.outputs[node.id].shape == data.shape
    assert control.value()["ranges"] == ""

    control.set_ranges({1: (2, 2)})
    widget.run_pipeline()

    widget.graph_view.select_node(node.id)

    assert widget.pipeline.outputs[node.id].shape == (2, 1, 4, 5, 6)
    assert _metadata_value(widget, "Dimensions") == "t=2, c=1, z=4, y=5, x=6"
    assert _metadata_value(widget, "Channels") == "1"
    assert "1. Select Axis Slice: kept c axis (1)[2..2]" in widget.history_label.text()


def test_select_axis_slice_can_slice_multiple_metadata_axes(qtbot):
    data = np.zeros((2, 3, 4, 5, 6), dtype=np.uint16)
    viewer = _Viewer(data, metadata={"axes": "TCZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("select_axis_slice")
    widget._connect_nodes("input", node.id)
    control = widget._parameter_widgets["axis_slice"]
    control.set_ranges({0: (1, 1), 1: (2, 2)})
    widget.run_pipeline()
    widget.graph_view.select_node(node.id)

    assert widget.pipeline.nodes[node.id].params["axes"] == ""
    assert widget.pipeline.nodes[node.id].params["indices"] == ""
    assert widget.pipeline.nodes[node.id].params["ranges"] == "0:1:1;1:2:2"
    assert widget.pipeline.nodes[node.id].params["range_mode"] is True
    assert widget.pipeline.outputs[node.id].shape == (1, 1, 4, 5, 6)
    assert _metadata_value(widget, "Dimensions") == "t=1, c=1, z=4, y=5, x=6"
    assert (
        "1. Select Axis Slice: kept t axis (0)[1..1], c axis (1)[2..2]"
        in widget.history_label.text()
    )


def test_select_axis_slice_can_remove_metadata_axis(qtbot):
    data = np.zeros((2, 3, 4, 5, 6), dtype=np.uint16)
    viewer = _Viewer(data, metadata={"axes": "TCZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("select_axis_slice")
    widget._connect_nodes("input", node.id)
    control = widget._parameter_widgets["axis_slice"]
    control.set_removed_axes({1: 2})
    widget.run_pipeline()
    widget.graph_view.select_node(node.id)

    assert widget.pipeline.nodes[node.id].params["remove_axes"] == "1"
    assert widget.pipeline.nodes[node.id].params["remove_indices"] == "2"
    assert widget.pipeline.outputs[node.id].shape == (2, 4, 5, 6)
    assert _metadata_value(widget, "Dimensions") == "t=2, z=4, y=5, x=6"
    assert _metadata_value(widget, "Channels") == "none"
    assert "1. Select Axis Slice: removed c axis (1)[2]" in widget.history_label.text()


def test_select_axis_slice_can_mix_ranges_and_removed_axes(qtbot):
    data = np.zeros((2, 3, 4, 5, 6), dtype=np.uint16)
    viewer = _Viewer(data, metadata={"axes": "TCZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("select_axis_slice")
    widget._connect_nodes("input", node.id)
    control = widget._parameter_widgets["axis_slice"]
    control.set_ranges({0: (1, 1)}, emit=False)
    control.set_removed_axes({1: 2})
    widget.run_pipeline()
    widget.graph_view.select_node(node.id)

    assert widget.pipeline.nodes[node.id].params["ranges"] == "0:1:1"
    assert widget.pipeline.nodes[node.id].params["remove_axes"] == "1"
    assert widget.pipeline.nodes[node.id].params["remove_indices"] == "2"
    assert widget.pipeline.outputs[node.id].shape == (1, 4, 5, 6)
    assert _metadata_value(widget, "Dimensions") == "t=1, z=4, y=5, x=6"
    assert (
        "1. Select Axis Slice: kept t axis (0)[1..1]; removed c axis (1)[2]"
        in widget.history_label.text()
    )


def test_reorder_axes_updates_metadata_axes(qtbot):
    data = np.zeros((2, 3, 4, 5, 6), dtype=np.uint16)
    viewer = _Viewer(data, metadata={"axes": "TCZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("reorder_axes")
    widget._connect_nodes("input", node.id)
    control = widget._parameter_widgets["order"]
    assert [
        control.list_widget.item(row).data(Qt.UserRole)
        for row in range(control.list_widget.count())
    ] == [0, 1, 2, 3, 4]

    control.set_order("TZYXC")
    widget.run_pipeline()
    widget.graph_view.select_node(node.id)

    assert widget.pipeline.nodes[node.id].params["order"] == "TZYXC"
    assert widget.pipeline.outputs[node.id].shape == (2, 4, 5, 6, 3)
    assert _metadata_value(widget, "Axes") == (
        "t(time), z(space), y(space), x(space), c(channel)"
    )
    assert _metadata_value(widget, "Dimensions") == "t=2, z=4, y=5, x=6, c=3"
    assert "1. Reorder Axes: reordered axes to TZYXC" in widget.history_label.text()

    control.reset_order()
    widget.run_pipeline()

    assert widget.pipeline.nodes[node.id].params["order"] == ""
    assert widget.pipeline.outputs[node.id].shape == data.shape


def test_reorder_axes_list_drag_changes_order(qtbot):
    data = np.zeros((2, 3, 4, 5, 6), dtype=np.uint16)
    viewer = _Viewer(data, metadata={"axes": "TCZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("reorder_axes")
    widget._connect_nodes("input", node.id)
    control = widget._parameter_widgets["order"]
    control.show()
    qtbot.waitExposed(control)
    axis_list = control.list_widget
    first = axis_list.visualItemRect(axis_list.item(0)).center()
    third = axis_list.visualItemRect(axis_list.item(2)).center()

    qtbot.mousePress(axis_list.viewport(), Qt.LeftButton, pos=first)
    qtbot.mouseMove(axis_list.viewport(), pos=third)
    qtbot.mouseRelease(axis_list.viewport(), Qt.LeftButton, pos=third)

    order = [axis_list.item(row).data(Qt.UserRole) for row in range(axis_list.count())]
    assert order == [1, 2, 0, 3, 4]
    assert widget.pipeline.nodes[node.id].params["order"] == "CZTYX"


def test_reorder_axes_reinterprets_spatial_axes_downstream(qtbot):
    data = np.zeros((3, 12, 96, 128), dtype=np.uint16)
    for y_index in range(data.shape[2]):
        data[:, :, y_index, y_index % data.shape[3]] = 100
    viewer = _Viewer(data, metadata={"axes": "CZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    reorder = widget.add_node_from_palette("reorder_axes")
    crop = widget.add_node_from_palette("crop_stack")
    widget._connect_nodes("input", reorder.id)
    widget._connect_nodes(reorder.id, crop.id)
    widget.pipeline.set_param(reorder.id, "order", "CYZX")
    widget.pipeline.set_param(crop.id, "top", 1)
    widget.pipeline.set_param(crop.id, "bottom", 2)
    widget.pipeline.set_param(crop.id, "left", 3)
    widget.pipeline.set_param(crop.id, "right", 4)
    widget.run_pipeline()

    reorder_state = widget.pipeline.output_states[reorder.id]
    crop_state = widget.pipeline.output_states[crop.id]

    assert widget.pipeline.outputs[reorder.id].shape == (3, 96, 12, 128)
    assert reorder_state.axis_order == "CZYX"
    assert [axis.source_axis for axis in reorder_state.axes] == [0, 1, 2, 3]
    assert "effective axes CZYX" in reorder_state.history[-1]
    assert widget.pipeline.outputs[crop.id].shape == (3, 96, 9, 121)
    assert crop_state.axis_order == "CZYX"

    first = make_preview(
        widget.pipeline.outputs[reorder.id],
        mode="slice",
        current_step=(0, 0, 0, 0),
        state=reorder_state,
    )
    second = make_preview(
        widget.pipeline.outputs[reorder.id],
        mode="slice",
        current_step=(0, 10, 0, 0),
        state=reorder_state,
    )
    assert not np.array_equal(first, second)


def test_reorder_axes_thumbnail_uses_reoriented_state(qtbot, monkeypatch):
    data = np.zeros((3, 12, 96, 128), dtype=np.uint16)
    viewer = _Viewer(data, metadata={"axes": "CZYX"})
    viewer.dims.current_step = (0, 7, 4, 0)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("reorder_axes")
    widget._connect_nodes("input", node.id)
    widget.pipeline.set_param(node.id, "order", "CYZX")
    widget.run_pipeline()
    calls = []

    def fake_make_preview(
        data,
        mode,
        current_step,
        state=None,
        channel_colors=None,
        contrast_mode="Percentile",
    ):
        calls.append((tuple(data.shape), current_step, state))
        return np.zeros((5, 6), dtype=np.uint8)

    monkeypatch.setattr("napari_vipp._widget.make_preview", fake_make_preview)
    widget._update_thumbnails()

    reorder_shape = tuple(widget.pipeline.outputs[node.id].shape)
    reorder_calls = [call for call in calls if call[0] == reorder_shape]
    assert reorder_calls
    assert reorder_calls[-1][1] == (0, 7, 4, 0)
    assert reorder_calls[-1][2].axis_order == "CZYX"


def test_converter_node_uses_choice_controls_and_updates_dtype(qtbot):
    data = np.arange(4 * 16 * 18, dtype=np.uint16).reshape(4, 16, 18)
    viewer = _Viewer(data)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("convert_dtype")
    widget._connect_nodes("input", node.id)

    assert widget.pipeline.outputs[node.id].dtype == np.uint8

    dtype_control = widget._parameter_widgets["output_dtype"]
    scaling_control = widget._parameter_widgets["scaling"]

    assert dtype_control.combo.currentText() == "uint8"
    assert scaling_control.combo.currentText() == "rescale"

    dtype_control.combo.setCurrentText("float32")
    widget.run_pipeline()

    assert widget.pipeline.nodes[node.id].params["output_dtype"] == "float32"
    assert widget.pipeline.outputs[node.id].dtype == np.float32
    assert _metadata_value(widget, "Dtype") == "float32"


def test_selected_node_preview_can_be_disabled(qtbot, monkeypatch):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.graph_view.select_node("gaussian")
    gaussian_card = widget.graph_view._cards["gaussian"]
    threshold_card = widget.graph_view._cards["threshold"]

    assert widget.thumbnail_checkbox.isChecked()
    assert not gaussian_card.preview.isHidden()
    assert not threshold_card.preview.isHidden()

    widget.thumbnail_checkbox.setChecked(False)

    assert "gaussian" in widget._preview_disabled_node_ids
    assert gaussian_card.preview.isHidden()
    assert not threshold_card.preview.isHidden()

    calls = []

    def fake_make_preview(
        data,
        mode,
        current_step,
        state=None,
        channel_colors=None,
        contrast_mode="Percentile",
    ):
        calls.append(data)
        return None

    monkeypatch.setattr("napari_vipp._widget.make_preview", fake_make_preview)
    widget._update_thumbnails()

    assert len(calls) == 2


def test_node_preview_toggle_is_restored_when_reenabled(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.graph_view.select_node("gaussian")
    card = widget.graph_view._cards["gaussian"]

    widget.thumbnail_checkbox.setChecked(False)
    widget.thumbnail_checkbox.setChecked(True)

    assert "gaussian" not in widget._preview_disabled_node_ids
    assert not card.preview.isHidden()


def test_graph_zoom_slider_controls_view_and_shows_default(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.graph_zoom_slider.minimum() == 40
    assert widget.graph_zoom_slider.maximum() == 250
    assert widget.graph_zoom_slider.value() == 100
    assert widget.graph_zoom_label.text() == "100%"
    assert np.isclose(widget.graph_view.transform().m11(), 1.0)
    assert widget.graph_zoom_reset_button.isEnabled()
    assert not widget.graph_zoom_reset_button.icon().isNull()

    widget.graph_zoom_slider.setValue(150)

    assert widget.graph_view.zoom_percent == 150
    assert widget.graph_zoom_label.text() == "150%"
    assert np.isclose(widget.graph_view.transform().m11(), 1.5)

    qtbot.mouseClick(widget.graph_zoom_reset_button, Qt.LeftButton)

    assert widget.graph_view.zoom_percent == 100
    assert widget.graph_zoom_slider.value() == 100
    assert widget.graph_zoom_label.text() == "100%"
    assert np.isclose(widget.graph_view.transform().m11(), 1.0)
    assert widget.graph_zoom_reset_button.isEnabled()


def test_graph_wheel_zoom_can_report_beyond_slider_range(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.graph_view.set_zoom_percent(400)

    assert widget.graph_view.zoom_percent == 400
    assert widget.graph_zoom_label.text() == "400%"
    assert widget.graph_zoom_slider.value() == 250


def test_gaussian_blur_3d_can_lock_xy_sigma(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("gaussian_blur_3d")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    lock = widget._parameter_widgets["lock_xy"]
    sigma_y = widget._parameter_widgets["sigma_y"]
    sigma_x = widget._parameter_widgets["sigma_x"]

    assert lock.checkbox.isChecked()

    sigma_y.value_box.setValue(3.4)

    assert widget.pipeline.nodes[node.id].params["sigma_y"] == 3.4
    assert widget.pipeline.nodes[node.id].params["sigma_x"] == 3.4
    assert sigma_x.value() == 3.4

    sigma_x.value_box.setValue(1.6)

    assert widget.pipeline.nodes[node.id].params["sigma_x"] == 1.6
    assert widget.pipeline.nodes[node.id].params["sigma_y"] == 1.6
    assert sigma_y.value() == 1.6

    lock.checkbox.setChecked(False)
    sigma_y.value_box.setValue(4.2)

    assert widget.pipeline.nodes[node.id].params["lock_xy"] is False
    assert widget.pipeline.nodes[node.id].params["sigma_y"] == 4.2
    assert widget.pipeline.nodes[node.id].params["sigma_x"] == 1.6


def test_slow_pipeline_run_shows_busy_indicator(qtbot):
    viewer = _Viewer(np.zeros((3, 12, 12), dtype=np.float32))
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("gaussian_blur_3d")
    widget._connect_nodes("input", node.id)

    assert widget._active_pipeline_run_id is not None
    assert not widget.pipeline_busy_bar.isHidden()
    assert widget.graph_view._cards[node.id].is_processing()

    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and widget.pipeline.outputs.get(node.id) is not None
        ),
        timeout=30_000,
    )

    assert widget.pipeline_busy_bar.isHidden()
    assert not widget.graph_view._cards[node.id].is_processing()
    assert widget.pipeline.outputs[node.id].shape == (3, 12, 12)


def test_downstream_parameter_change_reuses_cached_upstream_slow_node(
    qtbot,
    monkeypatch,
):
    calls = {"subtract": 0}
    original = NODE_LIBRARY_BY_ID["subtract_background"]

    def fake_subtract_background(image, **_kwargs):
        calls["subtract"] += 1
        return np.asarray(image)

    monkeypatch.setitem(
        NODE_LIBRARY_BY_ID,
        "subtract_background",
        replace(original, function=fake_subtract_background),
    )

    viewer = _Viewer(np.ones((8, 8), dtype=np.uint8) * 20)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    background = widget.add_node_from_palette("subtract_background")
    widget._connect_nodes("input", background.id)
    gamma = widget.add_node_from_palette("gamma_correction")
    widget._connect_nodes(background.id, gamma.id)
    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and not widget._pipeline_run_pending
            and widget.pipeline.outputs.get(gamma.id) is not None
        ),
        timeout=30_000,
    )
    calls_before = calls["subtract"]

    widget.graph_view.select_node(gamma.id)
    widget._parameter_widgets["gamma"].value_box.setValue(0.8)
    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and not widget._pending_dirty_node_ids
            and widget.pipeline.nodes[gamma.id].params["gamma"] == 0.8
        ),
        timeout=30_000,
    )

    assert calls["subtract"] == calls_before


def test_reedit_while_run_in_flight_stays_incremental(qtbot):
    viewer = _Viewer(np.ones((8, 8), dtype=np.uint8) * 20)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    background = widget.add_node_from_palette("subtract_background")
    widget._connect_nodes("input", background.id)
    gamma = widget.add_node_from_palette("gamma_correction")
    widget._connect_nodes(background.id, gamma.id)
    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and not widget._pipeline_run_pending
            and widget.pipeline.outputs.get(gamma.id) is not None
        ),
        timeout=30_000,
    )
    signature = widget._last_pipeline_source_signature
    assert signature is not None

    # A downstream-only run is dispatched for the gamma node.
    widget._mark_pipeline_dirty(gamma.id)
    dirty = {gamma.id}
    widget._begin_pipeline_dispatch(dirty)
    # The dispatched node is cleared from the pending set so that a re-edit of
    # the same node while the run is in flight is preserved as new work.
    assert gamma.id not in widget._pending_dirty_node_ids

    # The user edits gamma again before the in-flight run finishes.
    widget._mark_pipeline_dirty(gamma.id)
    assert gamma.id in widget._pending_dirty_node_ids

    # Completing the in-flight run must not discard the re-queued edit, so the
    # follow-up run stays incremental (gamma only) instead of recomputing the
    # whole pipeline from the source.
    widget._complete_pipeline_run(signature, dirty)
    assert gamma.id in widget._pending_dirty_node_ids
    assert widget._dirty_nodes_for_run(signature) == {gamma.id}


def test_discarded_inflight_run_requeues_dirty_nodes(qtbot):
    viewer = _Viewer(np.ones((8, 8), dtype=np.uint8) * 20)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    background = widget.add_node_from_palette("subtract_background")
    widget._connect_nodes("input", background.id)
    gamma = widget.add_node_from_palette("gamma_correction")
    widget._connect_nodes(background.id, gamma.id)
    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and not widget._pipeline_run_pending
            and widget.pipeline.outputs.get(gamma.id) is not None
        ),
        timeout=30_000,
    )

    widget._mark_pipeline_dirty(gamma.id)
    dirty = {gamma.id}
    widget._begin_pipeline_dispatch(dirty)
    assert gamma.id not in widget._pending_dirty_node_ids

    # A discarded/restarted run must return its in-flight dirty nodes to the
    # pending set so the rerun still covers them.
    widget._requeue_inflight_dirty_nodes()
    assert gamma.id in widget._pending_dirty_node_ids
    assert widget._inflight_dirty_node_ids is None


def test_autodefault_rerun_starts_at_changed_node_not_original_dirty(
    qtbot, monkeypatch
):
    # After an incremental background run, an auto-tracking node downstream of
    # the edit can shift its range and request a follow-up run. That follow-up
    # must start at the changed node (reusing the cached upstream output of the
    # edited node), not recompute the original dirty subtree from its source.
    calls = {"gamma": 0}
    original_gamma = NODE_LIBRARY_BY_ID["gamma_correction"]

    def fake_gamma(image, **_kwargs):
        calls["gamma"] += 1
        return np.asarray(image)

    monkeypatch.setitem(
        NODE_LIBRARY_BY_ID,
        "gamma_correction",
        replace(original_gamma, function=fake_gamma),
    )

    viewer = _Viewer(np.ones((8, 8), dtype=np.uint8) * 20)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    widget.background_all_checkbox.setChecked(True)

    gamma = widget.add_node_from_palette("gamma_correction")
    widget._connect_nodes("input", gamma.id)
    rescale = widget.add_node_from_palette("rescale_intensity")
    widget._connect_nodes(gamma.id, rescale.id)
    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and not widget._pipeline_run_pending
            and widget.pipeline.outputs.get(rescale.id) is not None
        ),
        timeout=30_000,
    )

    # Pretend the downstream rescale node's auto-tracked range shifts exactly
    # once, right after the next incremental run completes. Neutralize the
    # selected-control refresh so only the auto-default path drives the rerun.
    state = {"armed": False, "fired": False}
    monkeypatch.setattr(widget, "_refresh_selected_parameter_controls", lambda: False)

    def fake_resync():
        if state["armed"] and not state["fired"]:
            state["fired"] = True
            return {rescale.id}
        return set()

    monkeypatch.setattr(widget, "_resync_autodefault_nodes", fake_resync)

    widget.graph_view.select_node(gamma.id)
    calls_at_edit = calls["gamma"]
    state["armed"] = True
    widget._parameter_widgets["gamma"].value_box.setValue(0.8)
    qtbot.waitUntil(
        lambda: (
            state["fired"]
            and widget._active_pipeline_run_id is None
            and not widget._pipeline_run_pending
            and not widget._pending_dirty_node_ids
            and widget.pipeline.nodes[gamma.id].params["gamma"] == 0.8
        ),
        timeout=30_000,
    )

    # The edit recomputed gamma exactly once. The auto-default follow-up run
    # started at the changed rescale node and reused the cached gamma output,
    # so gamma was NOT recomputed a second time.
    assert calls["gamma"] == calls_at_edit + 1


def test_refresh_controls_ignores_float_spinner_noise(qtbot):
    # Reproduces the spurious-recompute bug: a sigma value produced by the
    # spinner (1.74 + 0.1) is stored with floating-point noise
    # (1.8399999999999999) while the spin box's value() rounds to 1.84.
    # Refreshing the selected node's controls must NOT report that as a change,
    # otherwise it forces an unnecessary follow-up pipeline run.
    viewer = _Viewer(np.ones((8, 8), dtype=np.uint8) * 20)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    gaussian = widget.add_node_from_palette("gaussian_blur")
    widget._connect_nodes("input", gaussian.id)
    widget.graph_view.select_node(gaussian.id)

    noisy_sigma = float(np.nextafter(1.84, 0.0))  # 1.8399999999999999
    assert noisy_sigma != 1.84
    assert round(noisy_sigma, 2) == 1.84
    gaussian.params["sigma"] = noisy_sigma
    widget._parameter_widgets["sigma"].value_box.setValue(1.84)

    changed = widget._refresh_selected_parameter_controls()

    # No meaningful change: the follow-up rerun must not be triggered, and the
    # stored parameter is normalized to the clean spin-box value.
    assert changed is False
    assert gaussian.params["sigma"] == widget._parameter_widgets["sigma"].value()


def test_rescale_axes_dirty_run_starts_at_rescale_and_reuses_upstream_cache(
    qtbot,
    monkeypatch,
):
    calls = {"subtract": 0, "rescale": 0}
    original_subtract = NODE_LIBRARY_BY_ID["subtract_background"]
    original_rescale = NODE_LIBRARY_BY_ID["rescale_axes"]

    def fake_subtract_background(image, **_kwargs):
        calls["subtract"] += 1
        return np.asarray(image)

    def fake_rescale_axes(image, **_kwargs):
        calls["rescale"] += 1
        time.sleep(0.1)
        return np.asarray(image)

    monkeypatch.setitem(
        NODE_LIBRARY_BY_ID,
        "subtract_background",
        replace(original_subtract, function=fake_subtract_background),
    )
    monkeypatch.setitem(
        NODE_LIBRARY_BY_ID,
        "rescale_axes",
        replace(original_rescale, function=fake_rescale_axes),
    )

    viewer = _Viewer(np.ones((8, 8), dtype=np.uint8) * 20)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    background = widget.add_node_from_palette("subtract_background")
    widget._connect_nodes("input", background.id)
    rescale = widget.add_node_from_palette("rescale_axes")
    widget._connect_nodes(background.id, rescale.id)
    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and not widget._pipeline_run_pending
            and widget.pipeline.outputs.get(rescale.id) is not None
        ),
        timeout=30_000,
    )
    subtract_calls_before = calls["subtract"]

    widget.graph_view.select_node(rescale.id)
    widget._parameter_widgets["x_scale"].value_box.setValue(1.25)
    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is not None
            and widget._active_pipeline_node_id == rescale.id
        ),
        timeout=30_000,
    )
    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and not widget._pending_dirty_node_ids
            and widget.pipeline.nodes[rescale.id].params["x_scale"] == 1.25
        ),
        timeout=30_000,
    )

    assert calls["subtract"] == subtract_calls_before
    assert calls["rescale"] >= 1


def test_global_preview_off_skips_thumbnail_generation(qtbot, monkeypatch):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    calls = []

    def fake_make_preview(
        data,
        mode,
        current_step,
        state=None,
        channel_colors=None,
        contrast_mode="Percentile",
    ):
        calls.append(data)
        return None

    monkeypatch.setattr("napari_vipp._widget.make_preview", fake_make_preview)
    widget.preview_mode_combo.setCurrentText("Off")

    assert calls == []
    assert widget.graph_view._cards["input"].preview.isHidden()
    assert widget.graph_view._cards["gaussian"].preview.isHidden()
    assert widget.graph_view._cards["threshold"].preview.isHidden()


def test_global_thumbnail_checkbox_skips_thumbnail_generation(qtbot, monkeypatch):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    calls = []

    def fake_make_preview(
        data,
        mode,
        current_step,
        state=None,
        channel_colors=None,
        contrast_mode="Percentile",
    ):
        calls.append(data)
        return None

    monkeypatch.setattr("napari_vipp._widget.make_preview", fake_make_preview)
    widget.global_thumbnail_checkbox.setChecked(False)

    assert calls == []
    assert widget.graph_view._cards["input"].preview.isHidden()
    assert widget.graph_view._cards["gaussian"].preview.isHidden()
    assert widget.graph_view._cards["threshold"].preview.isHidden()


def test_thumbnail_contrast_mode_is_passed_to_preview(qtbot, monkeypatch):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    calls = []

    def fake_make_preview(
        data,
        mode,
        current_step,
        state=None,
        channel_colors=None,
        contrast_mode="Percentile",
    ):
        calls.append(contrast_mode)
        return None

    monkeypatch.setattr("napari_vipp._widget.make_preview", fake_make_preview)
    widget.thumbnail_contrast_combo.setCurrentText("Raw")

    assert calls
    assert set(calls) == {"Raw"}


def test_label_thumbnail_output_type_is_passed_to_normalizer(qtbot, monkeypatch):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    labels = widget.add_node_from_palette("label_connected_components")
    widget._connect_nodes("threshold", labels.id)

    calls = []

    def fake_make_preview(
        data,
        mode,
        current_step,
        state=None,
        channel_colors=None,
        contrast_mode="Percentile",
    ):
        return np.zeros((4, 4), dtype=np.uint8)

    def fake_normalize_thumbnail(
        data,
        size=(180, 110),
        *,
        colormap="Gray",
        contrast_mode="Percentile",
        data_kind="image",
    ):
        calls.append(data_kind)
        return None

    monkeypatch.setattr("napari_vipp._widget.make_preview", fake_make_preview)
    monkeypatch.setattr(
        "napari_vipp._widget.normalize_thumbnail_with_colormap",
        fake_normalize_thumbnail,
    )

    widget.thumbnail_contrast_combo.setCurrentText("Raw")

    assert "labels" in calls


def test_palette_adds_node_and_connects_branch(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("median_filter")
    widget._connect_nodes("input", node.id)

    assert node.id in widget.pipeline.nodes
    assert (("input", node.id)) in {
        (connection.source_id, connection.target_id)
        for connection in widget.pipeline.connections
    }
    assert widget.pipeline.outputs[node.id] is not None


def test_palette_image_operations_can_run(qtbot):
    data = np.zeros((3, 18, 20), dtype=np.uint8)
    data[:, 5:14, 6:16] = 180
    data[1, 8, 10] = 255
    viewer = _Viewer(data)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    label_source_id = None
    table_source_id = None

    def ensure_label_source() -> str:
        nonlocal label_source_id
        if label_source_id is not None:
            return label_source_id
        labels = widget.add_node_from_palette("label_connected_components")
        widget._connect_nodes("threshold", labels.id)
        qtbot.waitUntil(
            lambda: widget.pipeline.outputs.get(labels.id) is not None,
            timeout=30_000,
        )
        assert widget.pipeline.outputs[labels.id] is not None
        label_source_id = labels.id
        return label_source_id

    def ensure_table_source() -> str:
        nonlocal table_source_id
        if table_source_id is not None:
            return table_source_id
        measurements = widget.add_node_from_palette("measure_objects")
        widget._connect_nodes(ensure_label_source(), measurements.id)
        qtbot.waitUntil(
            lambda: widget.pipeline.outputs.get(measurements.id) is not None,
            timeout=30_000,
        )
        assert widget.pipeline.outputs[measurements.id] is not None
        table_source_id = measurements.id
        return table_source_id

    def source_for_type(input_type: str | None) -> str:
        if input_type in {"mask", "mask_or_labels"}:
            return "threshold"
        if input_type == "labels":
            return ensure_label_source()
        if input_type == "table":
            return ensure_table_source()
        return "input"

    for spec in PALETTE_NODE_LIBRARY:
        if not spec.has_input:
            continue
        node = widget.add_node_from_palette(spec.id)
        if spec.inputs:
            for port_index, input_spec in enumerate(spec.inputs):
                widget._connect_nodes(
                    source_for_type(input_spec.input_type),
                    node.id,
                    target_port=port_index,
                )
        else:
            source_id = source_for_type(spec.input_type)
            widget._connect_nodes(source_id, node.id)
        if not spec.inputs and (spec.max_inputs is None or spec.max_inputs != 1):
            if spec.input_type == "table":
                widget._connect_nodes(table_source_id, node.id)
            else:
                second_input = widget.add_node_from_palette("input")
                widget._connect_nodes(second_input.id, node.id)

        qtbot.waitUntil(
            lambda node_id=node.id: (
                widget._active_pipeline_run_id is None
                and not widget._pipeline_run_pending
                and widget.pipeline.outputs.get(node_id) is not None
            ),
            timeout=30_000,
        )
        assert widget.pipeline.outputs[node.id] is not None, spec.id
        if spec.output_type == "labels":
            label_source_id = node.id
        if spec.output_type == "table":
            table_source_id = node.id


def test_save_selected_output_writes_npy(qtbot, tmp_path):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    path = tmp_path / "selected-output.npy"

    saved = widget._save_node_output("gaussian", str(path))

    assert saved == path
    assert path.exists()
    np.testing.assert_array_equal(np.load(path), widget.pipeline.outputs["gaussian"])


def test_measure_objects_shows_table_preview_and_saves_csv(qtbot, tmp_path):
    image = np.zeros((3, 9, 9), dtype=np.float32)
    image[:, 1:4, 1:4] = 10
    image[1, 7, 7] = 10
    viewer = _Viewer(image, metadata={"axes": "ZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    threshold = widget.add_node_from_palette("binary_threshold")
    labels = widget.add_node_from_palette("label_connected_components")
    measurements = widget.add_node_from_palette("measure_objects")
    widget.pipeline.set_param(threshold.id, "threshold", 5)
    widget._connect_nodes("input", threshold.id)
    widget._connect_nodes(threshold.id, labels.id)
    widget._connect_nodes(labels.id, measurements.id)
    widget.graph_view.select_node(measurements.id)
    path = tmp_path / "measurements.csv"

    saved = widget._save_node_output(measurements.id, str(path), format="csv")

    assert saved == path
    assert path.exists()
    assert widget.table_group.isHidden() is False
    assert widget.table_preview.rowCount() == 2
    assert widget.table_preview.columnCount() > 0
    assert widget.histogram_group.isHidden()
    assert widget.thumbnail_checkbox.isHidden()
    assert "include_shape_descriptors" in widget._parameter_widgets
    assert "include_axis_descriptors" in widget._parameter_widgets
    assert "include_2d_boundary_descriptors" not in widget._parameter_widgets
    widget.graph_view.select_node(labels.id)
    assert not widget.thumbnail_checkbox.isHidden()
    assert widget.thumbnail_checkbox.isEnabled()
    assert "label_id" in path.read_text(encoding="utf-8")


def test_save_selected_output_dialog_defaults_to_ome_tiff(qtbot, monkeypatch, tmp_path):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    captured = {}
    path = tmp_path / "selected-output.tif"

    def fake_get_save_file_name(_parent, title, default_name, filters):
        captured["title"] = title
        captured["default_name"] = default_name
        captured["filters"] = filters
        return str(path), "OME-TIFF (*.ome.tif *.ome.tiff)"

    monkeypatch.setattr(
        "napari_vipp._widget.QFileDialog.getSaveFileName",
        fake_get_save_file_name,
    )

    widget._save_selected_output_dialog()

    assert captured["title"] == "Save selected node output"
    assert captured["default_name"].endswith(".ome.tif")
    assert captured["filters"].startswith("OME-TIFF")
    assert "PNG image" not in captured["filters"]
    assert path.exists()


def test_save_selected_output_dialog_allows_raster_for_2d_output(
    qtbot,
    monkeypatch,
    tmp_path,
):
    viewer = _Viewer(np.arange(6 * 7, dtype=np.uint8).reshape(6, 7))
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    captured = {}
    path = tmp_path / "selected-output.png"

    def fake_get_save_file_name(_parent, title, default_name, filters):
        captured["title"] = title
        captured["default_name"] = default_name
        captured["filters"] = filters
        return str(path), "PNG image (*.png)"

    monkeypatch.setattr(
        "napari_vipp._widget.QFileDialog.getSaveFileName",
        fake_get_save_file_name,
    )

    widget._save_selected_output_dialog()

    assert captured["title"] == "Save selected node output"
    assert "PNG image" in captured["filters"]
    assert path.exists()
    assert iio.imread(path).ndim == 2


def test_save_image_node_writes_when_enabled(qtbot, tmp_path):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("save_output")
    widget._connect_nodes("gaussian", node.id)
    path = tmp_path / "graph-output.npy"

    widget.pipeline.set_param(node.id, "enabled", "on")
    widget.pipeline.set_param(node.id, "path", str(path))
    widget.pipeline.set_param(node.id, "format", "npy")
    widget.pipeline.set_param(node.id, "overwrite", "yes")
    widget.run_pipeline()

    assert path.exists()
    np.testing.assert_array_equal(np.load(path), widget.pipeline.outputs["gaussian"])


def test_save_image_node_writes_imagej_tiff_with_metadata(qtbot, tmp_path):
    data = np.zeros((2, 3, 4, 5, 6), dtype=bool)
    data[:, 1, 2, 1:4, 2:5] = True
    viewer = _Viewer(data, metadata={"axes": "TCZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("save_output")
    widget._connect_nodes("input", node.id)
    path = tmp_path / "graph-output.tif"

    widget.pipeline.set_param(node.id, "enabled", "on")
    widget.pipeline.set_param(node.id, "path", str(path))
    widget.pipeline.set_param(node.id, "format", "imagej-tiff")
    widget.pipeline.set_param(node.id, "overwrite", "yes")
    widget.run_pipeline()

    with tifffile.TiffFile(path) as tif:
        metadata = tif.imagej_metadata
        series = tif.series[0]
        saved = series.asarray()

    assert metadata["frames"] == 2
    assert metadata["slices"] == 4
    assert metadata["channels"] == 3
    assert series.axes == "TZCYX"
    assert set(np.unique(saved)) == {0, 255}


def test_save_image_node_writes_png_for_2d_output(qtbot, tmp_path):
    viewer = _Viewer(np.arange(6 * 7, dtype=np.uint8).reshape(6, 7))
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("save_output")
    widget._connect_nodes("input", node.id)
    path = tmp_path / "graph-output.png"

    widget.pipeline.set_param(node.id, "enabled", "on")
    widget.pipeline.set_param(node.id, "path", str(path))
    widget.pipeline.set_param(node.id, "format", "png")
    widget.pipeline.set_param(node.id, "overwrite", "yes")
    widget.run_pipeline()

    assert path.exists()
    assert iio.imread(path).shape == (6, 7)


def test_new_workflow_prompts_and_creates_empty_source_graph(
    qtbot,
    monkeypatch,
):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("binary_threshold")

    monkeypatch.setattr(
        "napari_vipp._widget.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.No,
    )
    widget._new_workflow_dialog()
    assert node.id in widget.pipeline.nodes

    monkeypatch.setattr(
        "napari_vipp._widget.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.Yes,
    )
    widget._new_workflow_dialog()

    assert list(widget.pipeline.nodes) == ["input"]
    assert widget.pipeline.connections == []
    assert widget.pipeline.nodes["input"].params["source_mode"] == "file path"
    assert widget.pipeline.nodes["input"].params["file_path"] == ""
    assert widget.pipeline.outputs["input"] is None
    assert widget.status_label.text() == "New empty workflow created."


def test_export_ome_dataset_dialog_writes_reference_and_labels(
    qtbot,
    monkeypatch,
    tmp_path,
):
    image = np.zeros((4, 8, 9), dtype=np.float32)
    image[:, 2:6, 3:7] = 10
    viewer = _Viewer(image, metadata={"axes": "ZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    threshold = widget.add_node_from_palette("binary_threshold")
    labels = widget.add_node_from_palette("label_connected_components")
    widget.pipeline.set_param(threshold.id, "threshold", 5)
    widget._connect_nodes("input", threshold.id)
    widget._connect_nodes(threshold.id, labels.id)
    widget.run_pipeline()
    path = tmp_path / "analysis.ome.zarr"

    def fake_get_save_file_name(_parent, title, default_name, filters):
        assert title == "Export OME analysis dataset"
        assert default_name.endswith(".ome.zarr")
        assert "OME-Zarr 0.4" in filters
        return str(path), "OME-Zarr 0.4 (*.ome.zarr)"

    monkeypatch.setattr(
        "napari_vipp._widget.QFileDialog.getSaveFileName",
        fake_get_save_file_name,
    )

    widget._export_ome_dataset_dialog()
    inspection = inspect_image_source(path)
    loaded_labels = read_image(path, series_index=1)

    assert path.exists()
    assert [series.kind for series in inspection.series] == ["image", "labels"]
    assert loaded_labels.image_state.kind == "label image"
    assert int(loaded_labels.data.compute().max()) == 1
    assert "1 label output" in widget.status_label.text()


def test_mask_output_can_feed_gaussian_blur(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("gaussian_blur")
    widget._connect_nodes("threshold", node.id)

    assert ("threshold", node.id) in {
        (connection.source_id, connection.target_id)
        for connection in widget.pipeline.connections
    }
    assert widget.pipeline.outputs[node.id] is not None
    assert widget.pipeline.outputs[node.id].dtype != bool
    assert widget.graph_view._cards[node.id].pin_button.isHidden()


def test_mask_output_can_feed_projection_and_remain_pinnable(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("mip")
    widget._connect_nodes("threshold", node.id)

    assert ("threshold", node.id) in {
        (connection.source_id, connection.target_id)
        for connection in widget.pipeline.connections
    }
    assert widget.pipeline.outputs[node.id].dtype == bool
    assert widget.graph_view._proxies[node.id].output_type == "mask"
    assert widget.graph_view._cards[node.id]._can_pin
    assert widget.graph_view._cards[node.id].pin_button.isHidden()

    widget.pin_node(node.id)

    pinned = viewer.layers["VIPP Pinned: Maximum Projection"]
    assert pinned.metadata["node_id"] == node.id
    assert pinned.layer_type == "labels"


def test_image_nodes_can_be_pinned(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.pin_node("gaussian")

    pinned_layers = [
        layer
        for layer in viewer.layers
        if layer.metadata.get("napari_vipp_kind") == "pinned"
    ]
    assert len(pinned_layers) == 1
    assert pinned_layers[0].metadata["node_id"] == "gaussian"
    assert pinned_layers[0].layer_type == "image"
    assert pinned_layers[0].metadata["display_kind"] == "image"
    assert widget._active_pinned_node_id == "gaussian"
    assert widget.graph_view._cards["gaussian"]._pinned
    assert widget.graph_view._cards["gaussian"].pin_button.isHidden()


def test_table_nodes_cannot_be_pinned(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    labels = widget.add_node_from_palette("label_connected_components")
    measurements = widget.add_node_from_palette("measure_objects")
    widget._connect_nodes("threshold", labels.id)
    widget._connect_nodes(labels.id, measurements.id)

    widget.pin_node(measurements.id)

    pinned_layers = [
        layer
        for layer in viewer.layers
        if layer.metadata.get("napari_vipp_kind") == "pinned"
    ]
    assert pinned_layers == []
    assert widget._active_pinned_node_id is None
    assert (
        "'Measure Objects' does not produce a displayable image output."
        in widget.status_label.text()
    )


def test_pin_button_visible_for_selected_image_node(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    labels = widget.add_node_from_palette("label_connected_components")
    measurements = widget.add_node_from_palette("measure_objects")
    widget._connect_nodes("threshold", labels.id)
    widget._connect_nodes(labels.id, measurements.id)

    widget.graph_view.select_node("gaussian")
    assert not widget.pin_button.isHidden()
    assert widget.pin_button.text() == "Pin selected"

    widget.graph_view.select_node(measurements.id)
    assert widget.pin_button.isHidden()


def test_only_one_image_node_is_actively_pinned(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    binary = widget.add_node_from_palette("binary_threshold")
    widget._connect_nodes("gaussian", binary.id)
    widget.pin_node(binary.id)
    widget.pin_node("gaussian")

    pinned_layers = [
        layer
        for layer in viewer.layers
        if layer.metadata.get("napari_vipp_kind") == "pinned"
    ]
    assert len(pinned_layers) == 1
    assert pinned_layers[0].metadata["node_id"] == "gaussian"
    assert widget._active_pinned_node_id == "gaussian"
    assert widget.graph_view._cards["gaussian"]._pinned
    assert not widget.graph_view._cards[binary.id]._pinned
    assert viewer.layers[-1] is pinned_layers[0]


def test_selecting_another_node_does_not_clear_pin(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.graph_view.select_node("threshold")
    widget.pin_node("threshold")
    widget.graph_view.select_node("gaussian")

    assert widget._active_pinned_node_id == "threshold"
    assert widget.graph_view._cards["threshold"]._pinned
    assert widget.graph_view._cards["threshold"].pin_button.isHidden()
    assert widget.graph_view._cards["gaussian"].pin_button.isHidden()
    assert not widget.pin_button.isHidden()
    assert widget.pin_button.text() == "Pin selected"


def test_selected_pinned_node_shows_unpin_in_inspector(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.graph_view.select_node("threshold")
    widget.pin_node("threshold")

    assert widget.pin_button.text() == "Unpin selected"
    assert not widget.pin_button.isHidden()


def test_active_pin_stays_on_top_after_inspect(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.pin_node("threshold")
    widget.inspect_node("gaussian")

    assert viewer.layers[-1].metadata["napari_vipp_kind"] == "pinned"
    assert viewer.layers[-1].metadata["node_id"] == "threshold"
    assert viewer.layers[-2].metadata["napari_vipp_kind"] == "inspect"


def test_pinned_image_stays_visible_while_editing_other_node(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.pin_node("gaussian")
    widget.graph_view.select_node("threshold")
    widget.pipeline.set_param("threshold", "threshold_scope", "Slice histogram")
    widget.run_pipeline()

    pinned = viewer.layers["VIPP Pinned: Gaussian Blur"]

    assert pinned.layer_type == "image"
    assert pinned.metadata["node_id"] == "gaussian"
    assert widget._active_pinned_node_id == "gaussian"
    assert widget.graph_view._cards["gaussian"]._pinned
    assert viewer.layers[-1] is pinned
    assert viewer.layers[-2].metadata["node_id"] == "threshold"


def test_inspect_shows_mask_as_standalone_image(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.inspect_node("gaussian")
    first_inspect = viewer.layers["VIPP Inspect"]
    assert first_inspect.layer_type == "image"
    assert first_inspect.metadata["display_kind"] == "image"

    widget.inspect_node("threshold")
    second_inspect = viewer.layers["VIPP Inspect"]

    assert second_inspect is not first_inspect
    assert second_inspect.layer_type == "image"
    assert second_inspect.metadata["display_kind"] == "image"
    assert second_inspect.metadata["data_kind"] == "mask"
    assert second_inspect.metadata["node_id"] == "threshold"
    assert second_inspect.contrast_limits == (0, 1)
    assert second_inspect.blending == "opaque"


def test_inspecting_active_mask_pin_keeps_pin_overlay_on_mask_image(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.graph_view.select_node("threshold")
    widget.pin_node("threshold")
    widget.inspect_node("threshold")

    inspect = viewer.layers["VIPP Inspect"]
    pinned = viewer.layers["VIPP Pinned: Otsu Threshold"]

    assert inspect.layer_type == "image"
    assert inspect.metadata["data_kind"] == "mask"
    assert pinned.layer_type == "labels"
    assert pinned.metadata["display_kind"] == "labels"
    assert viewer.layers[-2] is inspect
    assert viewer.layers[-1] is pinned


def test_signed_image_inspect_anchors_contrast_at_zero(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    # Signed difference image (e.g. Subtract of masks) with values in {-1, 0, 1}.
    signed = np.zeros((4, 16, 18), dtype=np.float32)
    signed[:, 4:8, :] = 1.0
    signed[:, 0:3, 4:8] = -1.0
    metadata = {
        "napari_vipp_kind": "inspect",
        "node_id": "subtract",
        "data_kind": widget._data_kind(signed),
        "display_kind": "image",
        "display_ndim": signed.ndim,
    }

    layer = widget._add_image_or_labels("VIPP Inspect", signed, metadata=metadata)

    assert layer.metadata["data_kind"] == "image"
    assert layer.contrast_limits == (0.0, 1.0)


def test_non_negative_image_keeps_default_contrast(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    positive = np.linspace(0, 200, 4 * 16 * 18, dtype=np.float32).reshape(4, 16, 18)

    assert widget._signed_image_contrast_limits(positive) is None


def test_inspecting_input_after_mask_resets_inspect_display(qtbot):
    data = np.arange(4 * 16 * 18, dtype=np.uint8).reshape(4, 16, 18)
    viewer = _Viewer(data)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.pin_node("threshold")
    widget.inspect_node("threshold")
    mask_inspect = viewer.layers["VIPP Inspect"]
    assert mask_inspect.metadata["data_kind"] == "mask"
    assert mask_inspect.contrast_limits == (0, 1)

    widget.inspect_node("input")
    input_inspect = viewer.layers["VIPP Inspect"]
    pinned = viewer.layers["VIPP Pinned: Otsu Threshold"]

    assert input_inspect is not mask_inspect
    assert input_inspect.layer_type == "image"
    assert input_inspect.metadata["data_kind"] == "image"
    assert input_inspect.metadata["node_id"] == "input"
    assert input_inspect.contrast_limits is None
    assert input_inspect.blending is None
    assert viewer.layers[-2] is input_inspect
    assert viewer.layers[-1] is pinned


def test_inspection_layer_is_replaced_when_dimensionality_changes(qtbot):
    viewer = _Viewer(np.zeros((4, 16, 18), dtype=np.float32))
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("mip")
    widget._connect_nodes("gaussian", node.id)
    projected_inspect = viewer.layers["VIPP Inspect"]
    assert projected_inspect.data.ndim == 2

    widget.inspect_node("gaussian")
    stack_inspect = viewer.layers["VIPP Inspect"]

    assert stack_inspect is not projected_inspect
    assert stack_inspect.data.ndim == 3
    assert stack_inspect.metadata["display_ndim"] == 3


def test_inspection_layer_is_replaced_when_shape_changes(qtbot):
    viewer = _Viewer(np.zeros((4, 16, 18), dtype=np.float32))
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget._set_or_add_generated_layer(
        "VIPP Inspect",
        np.zeros((4, 16, 18), dtype=np.float32),
        metadata={"napari_vipp_kind": "inspect", "node_id": "manual"},
        role="inspect",
    )
    first_inspect = viewer.layers["VIPP Inspect"]

    widget._set_or_add_generated_layer(
        "VIPP Inspect",
        np.zeros((8, 16, 18), dtype=np.float32),
        metadata={"napari_vipp_kind": "inspect", "node_id": "manual"},
        role="inspect",
    )
    second_inspect = viewer.layers["VIPP Inspect"]

    assert second_inspect is not first_inspect
    assert second_inspect.data.shape == (8, 16, 18)
    assert second_inspect.metadata["display_shape"] == (8, 16, 18)


def test_one_dimensional_inspection_data_is_displayed_as_row_image(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget._set_or_add_generated_layer(
        "VIPP Inspect",
        np.arange(12, dtype=np.uint16),
        metadata={"napari_vipp_kind": "inspect", "node_id": "manual"},
        role="inspect",
    )

    inspect = viewer.layers["VIPP Inspect"]
    assert inspect.data.shape == (1, 12)
    assert inspect.metadata["display_ndim"] == 2


def test_binary_threshold_uses_uint8_slider_range(qtbot):
    viewer = _Viewer(np.arange(16, dtype=np.uint8).reshape(4, 4))
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("binary_threshold")
    widget._connect_nodes("input", node.id)

    control = widget._parameter_widgets["threshold"]
    assert control.slider.minimum() == 0
    assert control.slider.maximum() == 255
    assert control.value_box.minimum() == 0
    assert control.value_box.maximum() == 255

    control.slider.setValue(128)

    assert widget.pipeline.nodes[node.id].params["threshold"] == 128


def test_binary_threshold_uses_unit_float_slider_range(qtbot):
    data = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    viewer = _Viewer(data)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("binary_threshold")
    widget._connect_nodes("input", node.id)

    control = widget._parameter_widgets["threshold"]
    assert control.value_box.minimum() == 0.0
    assert control.value_box.maximum() == 1.0
    assert control.slider.maximum() == 1000


def test_projection_axis_slider_uses_input_dimensionality(qtbot):
    viewer = _Viewer(np.zeros((2, 3, 4, 5), dtype=np.float32))
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("mip")
    widget._connect_nodes("input", node.id)

    control = widget._parameter_widgets["axis"]
    assert control.slider.minimum() == 0
    assert control.slider.maximum() == 3
    assert control.value_box.maximum() == 3


def test_soft_parameter_text_entry_expands_slider_range(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    control = widget._parameter_widgets["sigma"]

    assert control.value_box.maximum() > control.slider.maximum() / 100
    assert control.slider.maximum() == 1200

    control.value_box.setValue(20.0)

    assert control.slider.maximum() >= 2000
    assert control.slider.value() == 2000
    assert widget.pipeline.nodes["gaussian"].params["sigma"] == 20.0


def test_subtract_background_radius_slider_is_capped_but_entry_allows_more(qtbot):
    viewer = _Viewer(np.zeros((8, 8), dtype=np.uint8))
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("subtract_background")
    widget._connect_nodes("input", node.id)

    control = widget._parameter_widgets["radius"]
    control.slider.setValue(control.slider.maximum())

    assert control.value() == 100.0
    assert control.value_box.maximum() == 500.0

    control.value_box.setValue(250.0)

    assert control.slider.maximum() == 1000
    assert control.slider.value() == 1000
    assert control.value() == 250.0
    assert widget.pipeline.nodes[node.id].params["radius"] == 250.0


def test_crop_parameter_text_entry_stays_image_limited(qtbot):
    viewer = _Viewer(np.zeros((4, 16, 18), dtype=np.float32))
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("crop_stack")
    widget._connect_nodes("input", node.id)

    control = widget._parameter_widgets["top"]

    assert control.slider.maximum() == 15
    assert control.value_box.maximum() == 15

    control.value_box.setValue(99)

    assert control.value() == 15
    assert control.slider.maximum() == 15
    assert widget.pipeline.nodes[node.id].params["top"] == 15


def test_auto_contrast_button_updates_scale_and_offset(qtbot):
    data = np.arange(100, dtype=np.uint8).reshape(10, 10)
    viewer = _Viewer(data)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.auto_contrast_group.isHidden()

    node = widget.add_node_from_palette("linear_scale_offset")
    widget._connect_nodes("input", node.id)

    assert not widget.auto_contrast_group.isHidden()

    widget.auto_saturation_control.value_box.setValue(0.0)
    widget.auto_contrast_button.click()

    expected_alpha = 255.0 / 99.0
    params = widget.pipeline.nodes[node.id].params
    output = widget.pipeline.outputs[node.id]

    np.testing.assert_allclose(params["alpha"], expected_alpha, atol=0.0001)
    np.testing.assert_allclose(params["beta"], 0.0, atol=0.0001)
    np.testing.assert_allclose(
        widget._parameter_widgets["alpha"].value(),
        expected_alpha,
        atol=0.0001,
    )
    assert output.min() == 0
    assert output.max() == 255
