from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
import tifffile
from qtpy.QtCore import QEvent, QPoint, QPointF, QSignalBlocker, Qt
from qtpy.QtGui import QColor, QKeySequence, QMouseEvent
from qtpy.QtWidgets import (
    QApplication,
    QDockWidget,
    QGraphicsItem,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QWidget,
)

from napari_vipp import __version__ as VIPP_VERSION
from napari_vipp._theme import category_color, category_tint
from napari_vipp._widget import (
    CACHE_MODE_KEEP_ALL,
    CACHE_MODE_LOW_MEMORY,
    CACHE_MODE_SMART,
    EXAMPLE_WORKFLOWS,
    CollectionBatchDialog,
    ColocalizationScatterRequest,
    ColocalizationScatterResult,
    ConnectionInsertDialog,
    ConnectionInsertMappingDialog,
    ConnectionInsertPortMapping,
    ExampleWorkflowDialog,
    FlexibleDoubleSpinBox,
    GeneratedLayerContrastResult,
    HistogramPlot,
    InputHistogramResult,
    PipelineRunResult,
    SelectTableColumnsControl,
    VippWidget,
    _auto_contrast_scale_offset,
    _exact_finite_percentiles,
    _exact_generated_layer_contrast_limits,
    _example_workflow_path,
    _histogram_summary,
    _input_histogram_markers,
    _prepare_colocalization_scatter_density,
    _rescale_dtype_output_range,
)
from napari_vipp.core.batch import (
    BATCH_CONFIG_FILENAME,
    BATCH_CONFIG_TYPE,
    BATCH_MANIFEST_FILENAME,
    BATCH_MANIFEST_TYPE,
    BATCH_SCRIPT_FILENAME,
    BATCH_WORKFLOW_FILENAME,
    BatchStatus,
    ExistingFilePolicy,
    load_batch_config,
    plan_batch,
    scientific_workflow_hash,
)
from napari_vipp.core.batch_demo import (
    SYNTHETIC_BATCH_DEMO_DIRNAME,
    SyntheticBatchDemo,
    validate_synthetic_batch_demo,
)
from napari_vipp.core.graph_search import find_graph_matches
from napari_vipp.core.io import (
    ImageDataset,
    ImageSeriesInfo,
    OptionalMicroscopeReaderError,
    SourceInspection,
    inspect_image_source,
    read_image,
)
from napari_vipp.core.metadata import (
    AcquisitionMetadata,
    AxisMetadata,
    ChannelMetadata,
    image_state_from_array,
)
from napari_vipp.core.operations import (
    NO_TABLE_COLUMNS_VALUE,
    automatic_threshold_value,
)
from napari_vipp.core.pipeline import (
    EXECUTION_NOT_CALCULATED,
    EXECUTION_READY,
    EXECUTION_STALE,
    NODE_LIBRARY_BY_ID,
    PALETTE_NODE_LIBRARY,
    GraphNode,
    OutputTunnel,
    PrototypePipeline,
    SourcePayload,
)
from napari_vipp.core.preview import make_preview
from napari_vipp.core.progress import OperationCancelled, ProgressContext
from napari_vipp.core.workflow import save_workflow


class _Event:
    def __init__(self):
        self.callback = None

    def connect(self, callback):
        self.callback = callback

    def emit(self):
        if self.callback is not None:
            self.callback()


class _QueuedThreadPool:
    def __init__(self):
        self.workers = []

    def start(self, worker, _priority=0):
        self.workers.append(worker)


class _LayerEvents:
    def __init__(self):
        self.inserted = _Event()
        self.removed = _Event()


class _DimsEvents:
    def __init__(self):
        self.current_step = _Event()
        self.point = _Event()


class _Dims:
    def __init__(self, shape=(4, 16, 18)):
        self.nsteps = tuple(int(size) for size in shape)
        self.current_step = tuple(0 for _ in self.nsteps)
        self.events = _DimsEvents()

    def set_current_step(self, axis, value):
        current = list(self.current_step)
        while len(current) <= int(axis):
            current.append(0)
        upper = (
            max(int(self.nsteps[int(axis)]) - 1, 0)
            if int(axis) < len(self.nsteps)
            else int(value)
        )
        new_value = int(np.clip(value, 0, upper))
        if current[int(axis)] == new_value:
            return
        current[int(axis)] = new_value
        self.current_step = tuple(current)
        self.events.current_step.emit()


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
        self.scale = None


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
        self.dims = _Dims(np.asarray(data).shape)

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
        layer.scale = kwargs.get("scale")
        self.layers.append(layer)
        return layer

    def add_labels(self, data, **kwargs):
        layer = _Layer(
            data,
            kwargs["name"],
            metadata=kwargs.get("metadata"),
            layer_type="labels",
        )
        layer.scale = kwargs.get("scale")
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


def _assert_rgb_channel_layers(viewer, base_name: str, expected_shape: tuple[int, ...]):
    channels = (
        (base_name, "Red", "red", 0),
        (f"{base_name} Green", "Green", "green", 1),
        (f"{base_name} Blue", "Blue", "blue", 2),
    )
    for layer_name, channel_name, colormap, channel_index in channels:
        layer = viewer.layers[layer_name]
        assert layer.data.shape == expected_shape
        assert not layer.rgb
        assert layer.colormap == colormap
        assert layer.blending == "additive"
        assert layer.metadata["display_rgb"] is True
        assert layer.metadata["display_rgb_as_channels"] is True
        assert layer.metadata["display_rgb_channel"] == channel_name
        assert layer.metadata["display_rgb_channel_index"] == channel_index


def _metadata_value(widget, label):
    for row in range(widget.metadata_table.rowCount()):
        label_item = widget.metadata_table.item(row, 0)
        value_item = widget.metadata_table.item(row, 1)
        if label_item is not None and label_item.text() == label:
            return value_item.text() if value_item is not None else ""
    raise AssertionError(f"Metadata row not found: {label}")


def _view_dim_control(widget, label):
    for control in widget.view_dims_bar._controls:
        if control.label.text() == label:
            return control
    raise AssertionError(f"View dim control not found: {label}")


def _graph_view_center(view):
    return view.mapToScene(view.viewport().rect().center())


def test_flexible_double_spinbox_allows_decimal_typing_without_padding(qtbot):
    box = FlexibleDoubleSpinBox()
    qtbot.addWidget(box)
    box.setRange(-100.0, 100.0)
    box.setDecimals(3)
    box.setValue(5.0)
    box.show()
    box.lineEdit().setFocus()
    box.selectAll()
    emitted = []
    box.valueChanged.connect(emitted.append)

    qtbot.keyClicks(box.lineEdit(), "1.")

    assert box.lineEdit().text() == "1."
    assert emitted == []

    qtbot.keyClicks(box.lineEdit(), "2")

    assert box.lineEdit().text() == "1.2"
    assert emitted == []

    qtbot.keyClick(box.lineEdit(), Qt.Key_Return)

    assert abs(box.value() - 1.2) < 1e-9
    assert box.text() == "1.2"
    assert emitted == [1.2]


def test_flexible_double_spinbox_shows_compact_float_text(qtbot):
    box = FlexibleDoubleSpinBox()
    qtbot.addWidget(box)
    box.setRange(-100.0, 100.0)
    box.setDecimals(3)

    box.setValue(1.2)
    assert box.text() == "1.2"

    box.setValue(1.0)
    assert box.text() == "1"

    box.setValue(0.0)
    assert box.text() == "0"


def test_histogram_plot_dragging_marker_emits_histogram_value(qtbot):
    plot = HistogramPlot()
    qtbot.addWidget(plot)
    plot.resize(240, 160)
    plot.set_histogram(
        np.ones(32, dtype=np.float32),
        log_scale=False,
        x_range=(0.0, 100.0),
        markers=[("threshold", 50.0, QColor("#f59e0b"))],
        draggable_markers={"threshold"},
    )
    plot.show()
    captured = []
    plot.markerChanged.connect(
        lambda label, value: captured.append((str(label), float(value)))
    )

    rect = plot._plot_rect()
    y = rect.center().y()
    start = QPoint(
        rect.left() + int(round(plot._x_fraction(50.0) * rect.width())),
        y,
    )
    end = QPoint(rect.left() + int(round(0.75 * rect.width())), y)

    qtbot.mousePress(plot, Qt.LeftButton, pos=start)
    qtbot.mouseMove(plot, pos=end)
    qtbot.mouseRelease(plot, Qt.LeftButton, pos=end)

    assert captured
    assert captured[-1][0] == "threshold"
    assert np.isclose(captured[-1][1], 75.0, atol=1.0)


def test_widget_builds_graph_and_inspects_node(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    assert not hasattr(widget, "layer_combo")
    assert "gaussian" in widget.pipeline.outputs
    assert widget.version_label.text() == f"VIPP {VIPP_VERSION}"
    assert widget.background_all_checkbox.text() == "Run all in BG"
    assert widget.open_example_button.text() == "Open example..."

    inspect_layer = viewer.layers["VIPP Inspect"]
    assert inspect_layer.metadata["node_id"] == "gaussian"
    assert inspect_layer.data.shape == viewer.layers["input volume"].data.shape
    assert not viewer.layers["input volume"].visible


def test_example_workflow_dialog_groups_and_filters_examples(qtbot):
    dialog = ExampleWorkflowDialog()
    qtbot.addWidget(dialog)

    assert len(EXAMPLE_WORKFLOWS) >= 10
    assert dialog.tree.topLevelItemCount() >= 4

    dialog.filter_edit.setText("RACC")

    titles = []
    for index in range(dialog.tree.topLevelItemCount()):
        category = dialog.tree.topLevelItem(index)
        for child_index in range(category.childCount()):
            titles.append(category.child(child_index).text(0))
    assert titles == ["RACC Colocalization"]

    dialog.select_example("racc-colocalization")
    assert dialog.selected_example().id == "racc-colocalization"
    assert dialog.open_button.isEnabled()

    dialog.filter_edit.clear()
    dialog.select_example("batch-provenance")
    assert dialog.open_button.text() == "Open batch demo..."
    assert "ready-to-run" in dialog.details_label.text().lower()
    assert "Demo data" in dialog.details_label.text()
    assert "working copy" in dialog.details_label.text()
    assert "Run demo batch" in dialog.details_label.text()


def test_example_workflow_files_are_packaged():
    repo_examples = Path(__file__).resolve().parents[3] / "examples"
    for spec in EXAMPLE_WORKFLOWS:
        packaged = _example_workflow_path(spec)
        assert packaged.exists(), spec.filename
        repo_copy = repo_examples / spec.filename
        if repo_copy.exists():
            assert packaged.read_text(encoding="utf-8") == repo_copy.read_text(
                encoding="utf-8"
            )


def test_example_launcher_resolves_registry_and_rejects_unknown_aliases():
    from scripts.launch_vipp_intensity_workflow import _workflow_args

    for spec in EXAMPLE_WORKFLOWS:
        path, selected_node = _workflow_args([spec.id])
        assert path.name == spec.filename
        assert selected_node is None

    with pytest.raises(ValueError, match="Unknown example workflow"):
        _workflow_args(["typo-that-used-to-fall-back-silently"])


def test_open_example_workflow_loads_bundled_template(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    loaded = widget.load_example_workflow("label-cleanup")

    assert loaded.name == "otsu-red-channel-labels.json"
    assert widget.pipeline.nodes["input"].params["source_mode"] == "sample"
    assert (
        widget.pipeline.nodes["input"].params["sample_name"]
        == "VIPP synthetic multichannel volume"
    )
    assert "filter_labels_by_volume_1" in widget.pipeline.nodes
    assert widget.status_label.text().startswith("Opened example workflow")


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


def test_image_source_layer_selection_restores_previous_source(qtbot):
    viewer = _Viewer()
    viewer.layers.append(_Layer(np.ones((4, 16, 18), dtype=np.float32), "second"))
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    first = viewer.layers["input volume"]
    second = viewer.layers["second"]

    assert not first.visible
    assert second.visible

    widget.graph_view.select_node("input")
    control = widget._parameter_widgets["image_source"]
    assert control.layer_combo.currentText() == "input volume"

    control.layer_combo.setCurrentText("second")
    widget.run_pipeline()

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

    assert (
        widget.pipeline.nodes["input"].params["layer_name"]
        == "VIPP synthetic time-lapse multichannel"
    )
    assert widget.pipeline.outputs["input"].shape == rich_data.shape

    widget.graph_view.select_node("input")
    control = widget._parameter_widgets["image_source"]
    assert control.layer_combo.currentText() == "VIPP synthetic time-lapse multichannel"

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


def test_image_source_mode_change_autoselects_napari_layer(qtbot):
    viewer = _Viewer(np.zeros((2, 4, 5), dtype=np.uint8))
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget._new_workflow()
    control = widget._parameter_widgets["image_source"]
    assert control.mode_combo.currentText() == "file path"

    control.mode_combo.setCurrentText("napari layer")
    widget.run_pipeline()

    assert widget.pipeline.nodes["input"].params["layer_name"] == "input volume"
    assert control.layer_combo.currentText() == "input volume"
    assert widget.pipeline.outputs["input"].shape == (2, 4, 5)


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


def test_microscope_file_source_loads_in_background(qtbot, tmp_path, monkeypatch):
    path = tmp_path / "slow-source.czi"
    path.write_bytes(b"fake czi")
    started = threading.Event()
    release = threading.Event()

    def fake_read_image(path_arg, *, series_index=0):
        started.set()
        assert release.wait(5)
        data = np.ones((4, 5), dtype=np.uint8)
        state = image_state_from_array(
            data,
            layer_metadata={"axes": "YX"},
            source_name="Loaded CZI",
        )
        series = ImageSeriesInfo(0, "0", "Loaded CZI", data.shape, "uint8", "YX")
        inspection = SourceInspection(str(path_arg), "zeiss-czi", (series,))
        return ImageDataset(data, state, inspection, series)

    monkeypatch.setattr("napari_vipp._widget.read_image", fake_read_image)

    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    node = widget.pipeline.nodes["input"]
    node.params["source_mode"] = "file path"
    node.params["file_path"] = str(path)
    node.params["series_index"] = 0
    widget._mark_pipeline_dirty("input")

    widget.run_pipeline()

    assert widget._active_source_load_id is not None
    assert not widget.pipeline_busy_bar.isHidden()
    assert widget.graph_view._cards["input"].is_processing()
    qtbot.waitUntil(started.is_set, timeout=5_000)

    release.set()
    qtbot.waitUntil(
        lambda: (
            widget._active_source_load_id is None
            and widget.pipeline.outputs["input"].shape == (4, 5)
        ),
        timeout=10_000,
    )
    qtbot.waitUntil(
        lambda: (
            widget._active_thumbnail_contrast_run_id is None
            and not widget._pending_thumbnail_contrast_limit_keys
        ),
        timeout=5_000,
    )

    assert widget.pipeline_busy_bar.isHidden()
    assert not widget.graph_view._cards["input"].is_processing()
    assert widget.pipeline.output_states["input"].axis_order == "YX"


def test_current_view_metadata_follows_napari_dims(qtbot):
    viewer = _Viewer(
        np.zeros((5, 3, 4, 16, 18), dtype=np.uint16),
        metadata={"axes": "TCZYX"},
    )
    viewer.dims.current_step = (2, 1, 3, 0, 0)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    widget.follow_dims_checkbox.setChecked(True)

    widget.graph_view.select_node("input")

    assert _metadata_value(widget, "Current view") == "t=2/4, c=1/2, z=3/3"

    viewer.dims.current_step = (4, 0, 1, 0, 0)
    viewer.dims.events.current_step.emit()

    assert _metadata_value(widget, "Current view") == "t=4/4, c=0/2, z=1/3"


def test_unlinked_napari_dims_do_not_refresh_vipp_thumbnails(qtbot, monkeypatch):
    viewer = _Viewer(
        np.zeros((5, 3, 4, 16, 18), dtype=np.uint16),
        metadata={"axes": "TCZYX"},
    )
    viewer.dims.current_step = (1, 2, 3, 0, 0)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    widget.graph_view.select_node("input")
    widget.follow_dims_checkbox.setChecked(False)
    assert not widget.follow_dims_checkbox.isChecked()

    calls = []
    monkeypatch.setattr(widget, "_update_thumbnails", lambda: calls.append("refresh"))

    viewer.dims.set_current_step(0, 4)

    assert calls == []
    assert viewer.dims.current_step == (4, 2, 3, 0, 0)
    assert _view_dim_control(widget, "T").spin.value() == 1
    assert _metadata_value(widget, "Current view") == "t=1/4, c=2/2, z=3/3"


def test_unlinked_vipp_dims_refresh_thumbnails_without_moving_napari(
    qtbot,
    monkeypatch,
):
    viewer = _Viewer(
        np.zeros((5, 3, 4, 16, 18), dtype=np.uint16),
        metadata={"axes": "TCZYX"},
    )
    viewer.dims.current_step = (1, 2, 3, 0, 0)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    widget.graph_view.select_node("input")
    widget.follow_dims_checkbox.setChecked(False)
    assert not widget.follow_dims_checkbox.isChecked()

    calls = []
    monkeypatch.setattr(
        widget,
        "_update_thumbnails",
        lambda: calls.append(widget._current_step()),
    )

    _view_dim_control(widget, "Z").spin.setValue(1)

    assert viewer.dims.current_step == (1, 2, 3, 0, 0)
    assert calls[-1] == (1, 2, 1, 0, 0)
    assert _view_dim_control(widget, "Z").spin.value() == 1
    assert _metadata_value(widget, "Current view") == "t=1/4, c=2/2, z=1/3"


def test_view_dims_bar_exposes_semantic_axes_and_syncs_napari(qtbot):
    viewer = _Viewer(
        np.zeros((5, 3, 4, 16, 18), dtype=np.uint16),
        metadata={"axes": "TCZYX"},
    )
    viewer.dims.current_step = (1, 2, 3, 0, 0)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    widget.follow_dims_checkbox.setChecked(True)

    widget.graph_view.select_node("input")

    assert not widget.view_dims_bar.isHidden()
    view_axes = [
        (axis.label, axis.size, axis.value) for axis in widget.view_dims_bar._axes
    ]
    assert view_axes == [
        ("T", 5, 1),
        ("C", 3, 2),
        ("Z", 4, 3),
    ]

    z_control = _view_dim_control(widget, "Z")
    z_control.spin.setValue(1)

    assert viewer.dims.current_step == (1, 2, 1, 0, 0)
    assert _metadata_value(widget, "Current view") == "t=1/4, c=2/2, z=1/3"

    viewer.dims.set_current_step(0, 4)

    assert _view_dim_control(widget, "T").spin.value() == 4
    assert _metadata_value(widget, "Current view") == "t=4/4, c=2/2, z=1/3"


def test_view_dims_bar_syncs_after_selecting_axis_dropped_node(qtbot):
    viewer = _Viewer(
        np.zeros((3, 12, 16, 18), dtype=np.uint16),
        metadata={"axes": "CZYX"},
    )
    viewer.dims.current_step = (1, 4, 0, 0)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    widget.follow_dims_checkbox.setChecked(True)

    split = widget.add_node_from_palette("split_channels")
    blur = widget.add_node_from_palette("gaussian_blur")
    widget._connect_nodes("input", split.id)
    widget._connect_nodes(split.id, blur.id)
    widget.run_pipeline(force_sync=True)
    widget.graph_view.select_node(blur.id)

    view_axes = [
        (axis.label, axis.size, axis.value) for axis in widget.view_dims_bar._axes
    ]
    assert view_axes == [
        ("Z", 12, 4),
    ]

    _view_dim_control(widget, "Z").spin.setValue(7)

    assert viewer.dims.current_step == (1, 7, 0, 0)
    assert _metadata_value(widget, "Current view") == "z=7/11"


def test_view_dims_bar_hides_for_plain_2d_images(qtbot):
    viewer = _Viewer(np.zeros((16, 18), dtype=np.uint16), metadata={"axes": "YX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.graph_view.select_node("input")

    assert widget.view_dims_bar.isHidden()
    assert widget.view_dims_bar._axes == ()


def test_view_dims_bar_responsive_modes(qtbot):
    viewer = _Viewer(
        np.zeros((5, 3, 4, 16, 18), dtype=np.uint16),
        metadata={"axes": "TCZYX"},
    )
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.view_dims_bar.resize(1000, 32)
    widget.view_dims_bar.sync_responsive_mode()
    assert not widget.view_dims_bar.menu_button.isVisible()
    assert not _view_dim_control(widget, "Z").slider.isHidden()

    widget.view_dims_bar.resize(450, 32)
    widget.view_dims_bar.sync_responsive_mode()
    assert not widget.view_dims_bar.menu_button.isHidden()
    assert _view_dim_control(widget, "Z").slider.isHidden()
    assert not _view_dim_control(widget, "Z").spin.isHidden()

    widget.view_dims_bar.resize(260, 32)
    widget.view_dims_bar.sync_responsive_mode()
    assert not widget.view_dims_bar.menu_button.isHidden()
    assert _view_dim_control(widget, "Z").isHidden()


def test_view_dims_bar_maps_rescaled_axis_values_to_viewer_steps(qtbot):
    viewer = _Viewer(np.zeros((12, 16, 18), dtype=np.uint16), metadata={"axes": "ZYX"})
    viewer.dims.current_step = (11, 0, 0)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    widget.follow_dims_checkbox.setChecked(True)

    node = widget.add_node_from_palette("rescale_axes")
    widget.pipeline.set_param(node.id, "z_scale", 0.5)
    widget._connect_nodes("input", node.id)
    widget.run_pipeline(force_sync=True)
    widget.graph_view.select_node(node.id)

    assert widget.pipeline.outputs[node.id].shape[0] == 6
    view_axes = [
        (axis.label, axis.size, axis.value) for axis in widget.view_dims_bar._axes
    ]
    assert view_axes == [
        ("Z", 6, 5),
    ]

    _view_dim_control(widget, "Z").spin.setValue(3)

    assert viewer.dims.current_step[0] == 7


def test_view_dims_bar_uses_pinned_image_context_first(qtbot):
    viewer = _Viewer(
        np.zeros((5, 3, 4, 16, 18), dtype=np.uint16),
        metadata={"axes": "TCZYX"},
    )
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    projection = widget.add_node_from_palette("mip")
    widget._connect_nodes("input", projection.id)
    widget.run_pipeline(force_sync=True)
    widget.graph_view.select_node("input")

    assert [axis.label for axis in widget.view_dims_bar._axes] == ["T", "C", "Z"]

    widget.pin_node(projection.id)

    assert [axis.label for axis in widget.view_dims_bar._axes] == ["C", "Z"]

    widget.pin_node(projection.id)

    assert [axis.label for axis in widget.view_dims_bar._axes] == ["T", "C", "Z"]


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
        current_step_nsteps=None,
        state=None,
        channel_colors=None,
        contrast_mode="Percentile",
        contrast_scope="Slice",
        contrast_limits=None,
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
        "Auto from axes - using 3D ZYX",
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
        "Auto from axes - using 2D YX",
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

    widget._on_label_volume_marker_changed("min", 24.3)

    assert widget.pipeline.nodes[filtered.id].params["min_volume"] == 24
    assert widget._parameter_widgets["min_volume"].value() == 24
    assert [
        (label, value) for label, value, _color in widget.label_volume_plot._markers
    ][0] == ("min", 24.0)

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


def test_label_volume_histogram_reuses_input_distribution(
    qtbot,
    monkeypatch,
):
    data = np.zeros((3, 12, 12), dtype=np.float32)
    data[:, 1:5, 1:5] = 10
    data[:, 7:11, 7:11] = 10
    widget = VippWidget(_Viewer(data, metadata={"axes": "ZYX"}))
    qtbot.addWidget(widget)
    labels = widget.add_node_from_palette("label_connected_components")
    filtered = widget.add_node_from_palette("filter_labels_by_volume")
    widget._connect_nodes("threshold", labels.id)
    widget._connect_nodes(labels.id, filtered.id)

    calls: list[int] = []
    original = VippWidget._label_volumes

    def counted_label_volumes(values, spatial_ndim):
        calls.append(int(spatial_ndim))
        return original(values, spatial_ndim)

    monkeypatch.setattr(
        VippWidget,
        "_label_volumes",
        staticmethod(counted_label_volumes),
    )
    widget._label_volume_cache.clear()
    widget._update_label_volume_histogram()

    widget._on_label_volume_marker_changed("min", 12.0)
    widget._debounce_timer.stop()
    widget._on_label_volume_marker_changed("max", 40.0)
    widget._debounce_timer.stop()

    assert calls == [3]
    assert widget.pipeline.nodes[filtered.id].params["min_volume"] == 12
    assert widget.pipeline.nodes[filtered.id].params["max_volume"] == 40

    widget.pipeline.set_param(filtered.id, "spatial_mode", "2D YX")
    widget._update_label_volume_histogram()
    assert calls == [3, 2]

    replacement = widget.pipeline.input_data_for_node(filtered.id).copy()
    widget.pipeline.outputs[labels.id] = replacement
    widget.pipeline.node_outputs[labels.id] = [replacement]
    widget._update_label_volume_histogram()

    assert calls == [3, 2, 2]


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
    assert _palette_child_by_text(source_output, "Batch Output")
    assert _palette_child_by_text(axes_regions, "Crop Stack")
    assert _palette_child_by_text(axes_regions, "Select Axis Slice")
    assert _palette_child_by_text(axes_regions, "Split Axis")
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


def test_set_pixel_size_inspection_applies_napari_layer_scale(qtbot):
    viewer = _Viewer(np.zeros((3, 16, 18), dtype=np.float32))
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("set_pixel_size")
    widget._connect_nodes("input", node.id)
    widget.pipeline.set_param(node.id, "x_size", 1.0)
    widget.pipeline.set_param(node.id, "y_size", 1.0)
    widget.pipeline.set_param(node.id, "z_size", 1.0)
    widget.run_pipeline()
    widget.inspect_node(node.id)

    inspect = viewer.layers["VIPP Inspect"]
    assert inspect.scale == (1.0, 1.0, 1.0)
    widget.pin_node(node.id)
    pinned = viewer.layers["VIPP Pinned: Set Pixel Size / Units"]
    assert pinned.scale == (1.0, 1.0, 1.0)

    widget.pipeline.set_param(node.id, "z_size", 10.0)
    widget.run_pipeline()

    refreshed = viewer.layers["VIPP Inspect"]
    assert refreshed is inspect
    assert refreshed.scale == (10.0, 1.0, 1.0)
    refreshed_pin = viewer.layers["VIPP Pinned: Set Pixel Size / Units"]
    assert refreshed_pin is pinned
    assert refreshed_pin.scale == (10.0, 1.0, 1.0)
    assert [
        axis["scale"] for axis in refreshed.metadata["vipp_image_state"]["axes"]
    ] == [10.0, 1.0, 1.0]


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


def test_born_wolf_psf_auto_shows_resolved_values_without_sliders(qtbot):
    data = np.zeros((3, 16, 18), dtype=np.float32)
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("z", "space", unit="micrometer", scale=0.4),
            AxisMetadata("y", "space", unit="micrometer", scale=0.08),
            AxisMetadata("x", "space", unit="micrometer", scale=0.08),
        ),
        channels=(
            ChannelMetadata(
                name="green",
                emission_wavelength=520.0,
                emission_wavelength_unit="nanometer",
            ),
        ),
        acquisition=AcquisitionMetadata(
            objective_na=1.2,
            refractive_index=1.33,
        ),
    )
    viewer = _Viewer(data, metadata={"vipp_image_state": state.to_dict()})
    widget = VippWidget(viewer)
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("born_wolf_psf")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    auto = widget._parameter_widgets["auto_parameters"]
    wavelength = widget._parameter_widgets["wavelength_nm"]
    xy_size = widget._parameter_widgets["xy_size"]
    status = widget._parameter_widgets["wavelength_nm_status"]

    assert auto.checkbox.isChecked()
    assert not hasattr(wavelength, "slider")
    assert not hasattr(xy_size, "slider")
    assert not wavelength.isEnabled()
    assert wavelength.value() == 520.0
    assert "metadata" in status.text()


def test_born_wolf_psf_auto_marks_unresolved_metadata_red(qtbot):
    viewer = _Viewer(np.zeros((3, 16, 18), dtype=np.float32), metadata={"axes": "ZYX"})
    widget = VippWidget(viewer)
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("born_wolf_psf")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    wavelength = widget._parameter_widgets["wavelength_nm"]
    status = widget._parameter_widgets["wavelength_nm_status"]
    label = widget._parameter_widgets["wavelength_nm_label"]

    assert not wavelength.isEnabled()
    assert status.text() == "Unresolved"
    assert "#f87171" in status.styleSheet()
    assert "#f87171" in label.styleSheet()

    widget._parameter_widgets["auto_parameters"].checkbox.setChecked(False)

    assert widget.pipeline.nodes[node.id].params["auto_parameters"] is False
    assert widget._parameter_widgets["wavelength_nm"].isEnabled()
    assert widget.pipeline.nodes[node.id].params["wavelength_nm"] > 0
    assert widget.pipeline.nodes[node.id].params["pixel_size_xy_um"] > 0


def test_born_wolf_psf_auto_refreshes_inspector_and_outputs_all_channels(qtbot):
    data = np.zeros((2, 3, 16, 18), dtype=np.float32)
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("c", "channel"),
            AxisMetadata("z", "space", unit="micrometer", scale=0.4),
            AxisMetadata("y", "space", unit="micrometer", scale=0.08),
            AxisMetadata("x", "space", unit="micrometer", scale=0.08),
        ),
        channels=(
            ChannelMetadata(
                name="green",
                emission_wavelength=520.0,
                emission_wavelength_unit="nanometer",
            ),
            ChannelMetadata(
                name="red",
                emission_wavelength=620.0,
                emission_wavelength_unit="nanometer",
            ),
        ),
        acquisition=AcquisitionMetadata(
            objective_na=1.2,
            refractive_index=1.33,
        ),
    )
    viewer = _Viewer(data, metadata={"vipp_image_state": state.to_dict()})
    widget = VippWidget(viewer)
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("born_wolf_psf")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    assert len(widget.pipeline.output_ports(node.id)) == 2
    assert widget._parameter_widgets["channel"].value() == -1
    assert widget._parameter_widgets["channel_status"].text() == "all channels (2)"

    widget._parameter_widgets["channel_status"].setText("stale")
    widget.run_pipeline(force_sync=True)

    assert widget._parameter_widgets["channel_status"].text() == "all channels (2)"
    assert [port.label for port in widget.pipeline.output_ports(node.id)] == [
        "green PSF",
        "red PSF",
    ]
    assert len(widget.pipeline.node_outputs[node.id]) == 2


def test_born_wolf_channel_psfs_subtract_as_zyx_without_time_channel_dims(qtbot):
    data = np.zeros((2, 24, 10, 16, 18), dtype=np.uint16)
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("c", "channel"),
            AxisMetadata("t", "time"),
            AxisMetadata("z", "space", unit="micrometer", scale=0.4),
            AxisMetadata("y", "space", unit="micrometer", scale=0.08),
            AxisMetadata("x", "space", unit="micrometer", scale=0.08),
        ),
        channels=(
            ChannelMetadata(
                name="green",
                emission_wavelength=520.0,
                emission_wavelength_unit="nanometer",
            ),
            ChannelMetadata(
                name="red",
                emission_wavelength=620.0,
                emission_wavelength_unit="nanometer",
            ),
        ),
        acquisition=AcquisitionMetadata(
            objective_na=1.2,
            refractive_index=1.33,
        ),
    )
    viewer = _Viewer(data, metadata={"vipp_image_state": state.to_dict()})
    viewer.dims.current_step = (1, 9, 3, 0, 0)
    widget = VippWidget(viewer)
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    qtbot.addWidget(widget)

    psf = widget.add_node_from_palette("born_wolf_psf")
    widget.pipeline.set_param(psf.id, "xy_size", 15)
    widget.pipeline.set_param(psf.id, "z_size", 5)
    widget.pipeline.set_param(psf.id, "pupil_samples", 48)
    widget._connect_nodes("input", psf.id)
    widget.run_pipeline(force_sync=True)

    subtract = widget.add_node_from_palette("subtract_images")
    widget._connect_nodes(psf.id, subtract.id, source_port=0, target_port=0)
    widget._connect_nodes(psf.id, subtract.id, source_port=1, target_port=1)
    widget.run_pipeline(force_sync=True)

    output = widget.pipeline.outputs[subtract.id]
    output_state = widget.pipeline.output_states[subtract.id]
    assert output.shape == (5, 15, 15)
    assert output_state.axis_order == "ZYX"
    assert [axis.source_axis for axis in output_state.axes] == [2, 3, 4]

    preview = make_preview(
        output,
        mode="slice",
        current_step=viewer.dims.current_step,
        current_step_nsteps=viewer.dims.nsteps,
        state=output_state,
    )
    assert preview.shape == (15, 15)

    widget.graph_view.select_node(subtract.id)
    view_axes = [
        (axis.label, axis.size, axis.value) for axis in widget.view_dims_bar._axes
    ]
    assert view_axes == [
        ("Z", 5, 1),
    ]


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
        "Restoration & PSF",
    } <= filtering_subgroups

    smoothing = _palette_child_by_text(filtering, "Smoothing & Denoising")
    edge_detail = _palette_child_by_text(filtering, "Edge & Detail")
    background = _palette_child_by_text(filtering, "Background Correction")
    restoration = _palette_child_by_text(filtering, "Restoration & PSF")
    assert _palette_child_by_text(smoothing, "Gaussian Blur")
    assert _palette_child_by_text(smoothing, "Non-Local Means")
    assert _palette_child_by_text(edge_detail, "Difference of Gaussians")
    assert _palette_child_by_text(edge_detail, "Sobel Edges")
    assert _palette_child_by_text(edge_detail, "Canny Edges")
    assert _palette_child_by_text(background, "Rolling-Ball Background")
    assert _palette_child_by_text(background, "Subtract Background")
    assert _palette_child_by_text(restoration, "Born-Wolf PSF")
    assert _palette_child_by_text(restoration, "Prepare / Validate PSF")
    assert _palette_child_by_text(restoration, "Richardson-Lucy Deconvolution")
    assert _palette_child_by_text(restoration, "Richardson-Lucy TV Deconvolution")

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

    morphology = _palette_category(widget, "Morphology")
    skeleton_qc = _palette_child_by_text(morphology, "Skeleton / Network QC")
    assert _palette_child_by_text(skeleton_qc, "Skeletonize")
    assert _palette_child_by_text(skeleton_qc, "Skeleton Keypoints")
    assert _palette_child_by_text(skeleton_qc, "Skeleton Graph Overlay")
    assert _palette_child_by_text(skeleton_qc, "Prune Skeleton Branches")

    labels = _palette_category(widget, "Label Operations")
    label_skeleton_qc = _palette_child_by_text(labels, "Skeleton / Network QC")
    assert _palette_child_by_text(label_skeleton_qc, "Label Skeleton Components")
    assert _palette_child_by_text(label_skeleton_qc, "Label Skeleton Branches")

    measurements = _palette_category(widget, "Measurements")
    measurement_skeleton_qc = _palette_child_by_text(
        measurements,
        "Skeleton / Network QC",
    )
    assert _palette_child_by_text(measurement_skeleton_qc, "Analyze Skeleton")
    assert _palette_child_by_text(
        measurement_skeleton_qc,
        "Measure Skeleton Branches",
    )
    assert _palette_child_by_text(
        measurement_skeleton_qc,
        "Summarize Skeleton Branches",
    )
    assert _palette_child_by_text(measurement_skeleton_qc, "Skeleton Graph Tables")
    assert _palette_child_by_text(
        measurement_skeleton_qc,
        "Measure Overall Skeleton Network",
    )


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


def test_large_threshold_histogram_is_backgrounded_and_cached(qtbot, monkeypatch):
    data = np.arange(200, dtype=np.float32).reshape(2, 10, 10)
    viewer = _Viewer(data, metadata={"axes": "ZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    widget.graph_view.select_node("threshold")

    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_BYTES", 100)
    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_ELEMENTS", 100)
    started = threading.Event()
    release = threading.Event()
    calls = {"count": 0}

    def blocking_threshold(
        values,
        operation_id,
        histogram_bins=256,
        max_iterations=10_000,
        progress=None,
    ):
        calls["count"] += 1
        started.set()
        assert release.wait(5)
        return automatic_threshold_value(
            values,
            operation_id,
            histogram_bins=histogram_bins,
            max_iterations=max_iterations,
            progress=progress,
        )

    monkeypatch.setattr(
        "napari_vipp._widget.automatic_threshold_value",
        blocking_threshold,
    )
    widget._input_histogram_cache.clear()

    widget._update_histogram()

    qtbot.waitUntil(started.is_set, timeout=5_000)
    assert "calculating" in widget.rescale_input_histogram_group.title()
    release.set()
    qtbot.waitUntil(
        lambda: widget._active_input_histogram_run_id is None,
        timeout=5_000,
    )
    assert any(
        label == "threshold"
        for label, _value, _color in widget.rescale_input_histogram_plot._markers
    )

    widget._update_histogram()
    widget._update_histogram()
    viewer.dims.set_current_step(0, 1)

    assert calls["count"] == 1


@pytest.mark.parametrize(
    (
        "operation_id",
        "initial_params",
        "marker_label",
        "parameter_name",
        "new_value",
    ),
    [
        ("binary_threshold", {}, "threshold", "threshold", 64.0),
        (
            "hysteresis_threshold",
            {"low_threshold": 20.0, "high_threshold": 180.0},
            "low",
            "low_threshold",
            64.0,
        ),
        (
            "rescale_intensity",
            {
                "cutoff_mode": "Values",
                "in_low_value": 10.0,
                "in_high_value": 180.0,
            },
            "low",
            "in_low_value",
            64.0,
        ),
        (
            "clip_intensity",
            {"cutoff_mode": "Values", "minimum": 10.0, "maximum": 180.0},
            "min",
            "minimum",
            64.0,
        ),
    ],
)
def test_large_input_histogram_reuses_distribution_for_marker_drag(
    qtbot,
    monkeypatch,
    operation_id,
    initial_params,
    marker_label,
    parameter_name,
    new_value,
):
    data = np.arange(200, dtype=np.uint8).reshape(2, 10, 10)
    widget = VippWidget(_Viewer(data, metadata={"axes": "ZYX"}))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette(operation_id)
    widget._connect_nodes("input", node.id)
    for name, value in initial_params.items():
        widget.pipeline.set_param(node.id, name, value)
    widget.graph_view.select_node(node.id)

    calls = {"count": 0}
    original = _histogram_summary

    def counted_histogram(*args, **kwargs):
        if args and args[0] is data:
            calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr("napari_vipp._widget._histogram_summary", counted_histogram)
    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_BYTES", 1)
    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_ELEMENTS", 1)
    widget._clear_input_histogram_cache()
    widget._update_rescale_input_histogram(node.id, widget._current_step())
    qtbot.waitUntil(
        lambda: widget._active_input_histogram_run_id is None,
        timeout=5_000,
    )

    assert calls["count"] == 1
    assert len(widget._input_histogram_distribution_cache) == 1

    widget._on_input_histogram_marker_changed(marker_label, new_value)
    widget._debounce_timer.stop()

    markers = {
        label: value
        for label, value, _color in widget.rescale_input_histogram_plot._markers
    }
    assert calls["count"] == 1
    assert widget._active_input_histogram_run_id is None
    assert widget.pipeline.nodes[node.id].params[parameter_name] == new_value
    assert markers[marker_label] == new_value

    widget.run_pipeline(force_sync=True)

    assert widget.pipeline.input_data_for_node(node.id) is data
    assert calls["count"] == 1
    assert {
        label: value
        for label, value, _color in widget.rescale_input_histogram_plot._markers
    }[marker_label] == new_value


def test_input_histogram_distribution_invalidates_for_data_scope_and_slice(
    qtbot,
    monkeypatch,
):
    data = np.arange(200, dtype=np.float32).reshape(2, 10, 10)
    widget = VippWidget(_Viewer(data, metadata={"axes": "ZYX"}))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("binary_threshold")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    calls = {"count": 0}
    original = _histogram_summary

    def counted_histogram(*args, **kwargs):
        if args and args[0] is widget.pipeline.input_data_for_node(node.id):
            calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr("napari_vipp._widget._histogram_summary", counted_histogram)
    widget._clear_input_histogram_cache()

    widget._update_rescale_input_histogram(node.id, (0, 0, 0))
    widget._update_rescale_input_histogram(node.id, (0, 0, 0))
    assert calls["count"] == 1

    widget.rescale_input_histogram_log_checkbox.setChecked(True)
    assert calls["count"] == 1

    widget._update_rescale_input_histogram(node.id, (1, 0, 0))
    assert calls["count"] == 2

    with QSignalBlocker(widget.rescale_input_histogram_scope_combo):
        widget.rescale_input_histogram_scope_combo.setCurrentText("Stack histogram")
    widget._update_rescale_input_histogram(node.id, (1, 0, 0))
    assert calls["count"] == 3

    replacement = data.copy()
    widget.pipeline.outputs["input"] = replacement
    widget.pipeline.node_outputs["input"] = [replacement]
    widget._update_rescale_input_histogram(node.id, (1, 0, 0))
    assert calls["count"] == 4


def test_otsu_histogram_bins_refresh_marker_but_reuse_distribution(
    qtbot,
    monkeypatch,
):
    data = np.linspace(0.0, 1.0, 200, dtype=np.float32).reshape(2, 10, 10)
    widget = VippWidget(_Viewer(data, metadata={"axes": "ZYX"}))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("otsu_threshold")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    histogram_calls = {"count": 0}
    threshold_calls: list[tuple[int, float]] = []
    original_histogram = _histogram_summary
    original_threshold = automatic_threshold_value

    def counted_histogram(*args, **kwargs):
        histogram_calls["count"] += 1
        return original_histogram(*args, **kwargs)

    def counted_threshold(
        values,
        operation_id,
        histogram_bins=256,
        max_iterations=10_000,
        progress=None,
    ):
        result = original_threshold(
            values,
            operation_id,
            histogram_bins=histogram_bins,
            max_iterations=max_iterations,
            progress=progress,
        )
        threshold_calls.append((int(histogram_bins), float(result)))
        return result

    monkeypatch.setattr("napari_vipp._widget._histogram_summary", counted_histogram)
    monkeypatch.setattr(
        "napari_vipp._widget.automatic_threshold_value",
        counted_threshold,
    )
    widget._clear_input_histogram_cache()
    widget._update_rescale_input_histogram(node.id, widget._current_step())

    widget._on_param_changed("histogram_bins", 512)
    widget._debounce_timer.stop()

    markers = {
        label: value
        for label, value, _color in widget.rescale_input_histogram_plot._markers
    }
    assert histogram_calls["count"] == 1
    assert [bins for bins, _value in threshold_calls] == [256, 512]
    assert np.isclose(markers["threshold"], threshold_calls[-1][1])


def test_histogram_cache_invalidation_rejects_an_inflight_result(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    key = ("old-input",)
    widget._active_input_histogram_run_id = 7
    widget._active_input_histogram_key = key
    widget._current_input_histogram_key = key

    widget._clear_input_histogram_cache()
    widget._on_input_histogram_finished(
        InputHistogramResult(
            7,
            key,
            "threshold",
            counts=np.array([1]),
        )
    )

    assert widget._active_input_histogram_run_id is None
    assert widget._current_input_histogram_key is None
    assert widget._input_histogram_cache == {}


def test_colocalization_inspector_scatter_syncs_thresholds(qtbot):
    data = np.zeros((2, 16, 16), dtype=np.uint16)
    data[0, 3:11, 3:11] = 6000
    data[1, 6:14, 6:14] = 6500
    data[0, 1:4, 11:14] = 3500
    data[1, 11:14, 1:4] = 3500
    viewer = _Viewer(data, metadata={"axes": "CYX"})
    widget = VippWidget(viewer)
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    qtbot.addWidget(widget)

    split = widget.add_node_from_palette("split_channels")
    coloc = widget.add_node_from_palette("colocalized_voxels")
    widget.pipeline.set_param(coloc.id, "threshold_mode", "Costes auto")
    widget.pipeline.set_param(coloc.id, "channel_1_color", "Blue")
    widget.pipeline.set_param(coloc.id, "channel_2_color", "Yellow")
    widget._connect_nodes("input", split.id)
    widget._connect_nodes(split.id, coloc.id, source_port=0, target_port=0)
    widget._connect_nodes(split.id, coloc.id, source_port=1, target_port=1)

    widget.run_pipeline(force_sync=True)
    widget.graph_view.select_node(coloc.id)

    assert not widget.colocalization_scatter_group.isHidden()
    assert widget.colocalization_scatter_plot._image is not None
    assert widget.colocalization_scatter_plot._channel_1_color.name() == "#0000ff"
    assert widget.colocalization_scatter_plot._channel_2_color.name() == "#ffff00"
    assert widget.colocalization_scatter_plot.minimumHeight() == 300
    assert widget.colocalization_scatter_summary.minimumHeight() == 42
    assert widget.colocalization_scatter_summary.maximumHeight() > 42
    assert widget.colocalization_scatter_summary.wordWrap()
    wrapped_height = widget.colocalization_scatter_summary.heightForWidth(240)
    assert wrapped_height > 42
    assert widget.colocalization_scatter_summary.maximumHeight() >= wrapped_height
    assert widget.colocalization_scatter_colormap_combo.currentText() == "Viridis"
    widget.colocalization_scatter_plot.resize(620, 260)
    plot_rect = widget.colocalization_scatter_plot._plot_rect()
    assert plot_rect.width() == plot_rect.height()
    assert widget.pipeline.nodes[coloc.id].params["threshold_mode"] == "Costes auto"
    assert 0 <= widget.pipeline.nodes[coloc.id].params["channel_1_threshold"] <= 255
    assert 0 <= widget.pipeline.nodes[coloc.id].params["channel_2_threshold"] <= 255
    assert not np.isclose(
        widget.pipeline.nodes[coloc.id].params["channel_1_threshold"],
        25.0,
    )

    widget._on_colocalization_scatter_threshold_changed(1, 12.5)
    widget._debounce_timer.stop()

    assert widget.pipeline.nodes[coloc.id].params["threshold_mode"] == "Manual"
    assert np.isclose(
        widget.pipeline.nodes[coloc.id].params["channel_1_threshold"],
        12.5,
    )
    widget.colocalization_scatter_colormap_combo.setCurrentText("Magma")

    assert widget.colocalization_scatter_plot._colormap == "Magma"


def test_colocalization_scatter_density_and_counts_are_exact_beyond_old_cap():
    size = 600_123
    indices = np.arange(size, dtype=np.uint32)
    channel_1 = (indices % 256).astype(np.float32)
    channel_2 = ((indices * 37 + 11) % 256).astype(np.float32)
    roi = indices % 11 != 0

    density, roi_voxels, colocalized_voxels = _prepare_colocalization_scatter_density(
        channel_1,
        channel_2,
        threshold_1=125.0,
        threshold_2=140.0,
        roi_mask=roi,
        intensity_max=255.0,
        bins=64,
    )

    expected_density = np.histogram2d(
        channel_1[roi],
        channel_2[roi],
        bins=64,
        range=((0.0, 255.0), (0.0, 255.0)),
    )[0]
    expected_colocalized = np.count_nonzero(
        (channel_1 >= 125.0) & (channel_2 >= 140.0) & roi
    )
    assert roi_voxels > 500_000
    assert roi_voxels == int(np.count_nonzero(roi))
    assert colocalized_voxels == int(expected_colocalized)
    np.testing.assert_array_equal(density, expected_density)
    assert int(density.sum()) == roi_voxels


def test_colocalization_scatter_density_is_cooperatively_cancellable(monkeypatch):
    channel_1 = np.arange(100, dtype=np.float32)
    channel_2 = channel_1[::-1].copy()
    checks = 0

    def cancelled():
        nonlocal checks
        checks += 1
        return checks >= 3

    monkeypatch.setattr(
        "napari_vipp._widget.INSPECTOR_STATISTICS_CHUNK_ELEMENTS",
        10,
    )
    with pytest.raises(OperationCancelled):
        _prepare_colocalization_scatter_density(
            channel_1,
            channel_2,
            threshold_1=25.0,
            threshold_2=25.0,
            roi_mask=None,
            intensity_max=255.0,
            bins=32,
            progress=ProgressContext(cancelled=cancelled),
        )


def test_large_colocalization_scatter_returns_immediately_and_is_exact(
    qtbot,
    monkeypatch,
):
    data = np.zeros((100, 100), dtype=np.uint8)
    data.ravel()[::2] = 200
    viewer = _Viewer(data, metadata={"axes": "YX"})
    widget = VippWidget(viewer)
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    qtbot.addWidget(widget)
    coloc = widget.add_node_from_palette("colocalized_voxels")
    widget.pipeline.set_param(coloc.id, "threshold_mode", "Manual")
    widget.pipeline.set_param(coloc.id, "channel_1_threshold", 100.0)
    widget.pipeline.set_param(coloc.id, "channel_2_threshold", 100.0)
    widget._connect_nodes("input", coloc.id, target_port=0)
    widget._connect_nodes("input", coloc.id, target_port=1)
    widget.run_pipeline(force_sync=True)
    widget.graph_view.select_node(coloc.id)

    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_BYTES", 100)
    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_ELEMENTS", 100)
    import napari_vipp._widget as widget_module

    real_normalize = widget_module.colocalization_normalized_inputs
    started = threading.Event()
    release = threading.Event()

    def blocking_normalize(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return real_normalize(*args, **kwargs)

    monkeypatch.setattr(
        "napari_vipp._widget.colocalization_normalized_inputs",
        blocking_normalize,
    )
    widget._clear_colocalization_scatter_cache()

    before = time.perf_counter()
    widget._update_colocalization_scatter()
    elapsed = time.perf_counter() - before
    try:
        qtbot.waitUntil(started.is_set, timeout=5_000)
        assert elapsed < 0.2
        assert "Calculating the exact" in widget.colocalization_scatter_summary.text()
    finally:
        release.set()

    qtbot.waitUntil(
        lambda: widget._active_colocalization_scatter_run_id is None,
        timeout=5_000,
    )
    summary = widget.colocalization_scatter_summary.text()
    tooltip = widget.colocalization_scatter_summary.toolTip()
    assert "Exact colocalized count: 5,000/10,000 (50.0%)" in summary
    assert "Exact scatter density from all 10,000 ROI voxels" in summary
    assert "Every ROI voxel contributes" in tooltip
    assert "5,000/10,000 (50.0%) meet both thresholds" in tooltip
    assert "Exact: 5,000/10,000 (50.0%)" == widget.colocalization_scatter_plot._summary
    result = widget._colocalization_scatter_cache[
        widget._current_colocalization_scatter_key
    ]
    assert np.asarray(result.density_counts).shape == (255, 255)
    assert int(np.asarray(result.density_counts).sum()) == 10_000


def test_colocalization_scatter_zero_roi_reports_percentage_unavailable(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    coloc = widget.add_node_from_palette("colocalized_voxels")
    widget.graph_view.select_node(coloc.id)
    key = ("empty-roi",)
    widget._current_colocalization_scatter_key = key

    widget._apply_colocalization_scatter_result(
        ColocalizationScatterResult(
            0,
            key,
            coloc.id,
            "Manual",
            25.0,
            25.0,
            density_counts=np.zeros((32, 32), dtype=np.float64),
            roi_voxels=0,
            colocalized_voxels=0,
        )
    )

    assert "Exact colocalized count: 0/0 (n/a)" in (
        widget.colocalization_scatter_summary.text()
    )
    assert widget.colocalization_scatter_plot._summary == "Exact: 0/0 (n/a)"
    assert "0/0 (n/a) meet both thresholds" in (
        widget.colocalization_scatter_summary.toolTip()
    )


def test_colocalization_scatter_rejects_result_for_superseded_key(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    coloc = widget.add_node_from_palette("colocalized_voxels")
    old_key = ("old-scatter",)
    new_key = ("new-scatter",)
    widget._active_colocalization_scatter_run_id = 7
    widget._active_colocalization_scatter_key = old_key
    widget._current_colocalization_scatter_key = new_key
    widget.colocalization_scatter_summary.setText("New request is pending")

    widget._on_colocalization_scatter_finished(
        ColocalizationScatterResult(
            7,
            old_key,
            coloc.id,
            "Manual",
            25.0,
            25.0,
            density_counts=np.ones((32, 32), dtype=np.float64),
            roi_voxels=100,
            colocalized_voxels=50,
        )
    )

    assert widget.colocalization_scatter_summary.text() == "New request is pending"
    assert widget.colocalization_scatter_plot._image is None


def test_colocalization_scatter_a_b_a_requeues_cancelled_request(qtbot, monkeypatch):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    started = []

    def fake_start(request):
        started.append(request)
        widget._active_colocalization_scatter_run_id = 1
        widget._active_colocalization_scatter_key = request.key
        widget._active_colocalization_scatter_cancel_event = threading.Event()

    monkeypatch.setattr(widget, "_start_colocalization_scatter_request", fake_start)
    common = {
        "run_id": 0,
        "node_id": "coloc",
        "inputs": (np.zeros((2, 2)), np.zeros((2, 2))),
        "threshold_mode": "Manual",
        "threshold_1": 25.0,
        "threshold_2": 25.0,
    }
    first = ColocalizationScatterRequest(key=("a",), **common)
    second = ColocalizationScatterRequest(key=("b",), **common)

    widget._queue_colocalization_scatter(first)
    widget._queue_colocalization_scatter(second)
    assert widget._active_colocalization_scatter_cancel_event.is_set()
    widget._queue_colocalization_scatter(first)

    assert len(started) == 1
    assert widget._pending_colocalization_scatter_request is not None
    assert widget._pending_colocalization_scatter_request.key == first.key


def test_pipeline_edit_invalidates_colocalization_scatter_cache(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    key = ("cached-scatter",)
    result = ColocalizationScatterResult(
        4,
        key,
        "input",
        "Manual",
        25.0,
        25.0,
        density_counts=np.ones((32, 32), dtype=np.float64),
    )
    widget._colocalization_scatter_cache[key] = result
    widget._current_colocalization_scatter_key = key
    widget._active_colocalization_scatter_run_id = 4
    cancel_event = threading.Event()
    widget._active_colocalization_scatter_cancel_event = cancel_event
    old_serial = widget._colocalization_scatter_serial

    assert widget._mark_pipeline_dirty("input")

    assert widget._colocalization_scatter_cache == {}
    assert widget._current_colocalization_scatter_key is None
    assert widget._active_colocalization_scatter_run_id is None
    assert cancel_event.is_set()
    assert widget._colocalization_scatter_serial == old_serial + 1


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


def test_exact_histogram_and_percentile_markers_retain_rare_extrema():
    values = np.zeros(600_123, dtype=np.float32)
    values[500_001] = -17.0
    values[-1] = 1_000.0
    values[511_111] = np.nan

    counts, x_range, _colors = _histogram_summary(values, scope="Stack")
    percentiles = _exact_finite_percentiles(values, (0.0, 50.0, 100.0))

    assert counts is not None
    assert int(counts.sum()) == values.size - 1
    assert x_range == (-17.0, 1_000.0)
    assert counts[0] >= 1
    assert counts[-1] >= 1
    assert percentiles == (-17.0, 0.0, 1_000.0)


def test_multichannel_histogram_preserves_all_nonfinite_channel_position():
    data = np.empty((4, 5, 3), dtype=np.float32)
    data[..., 0] = np.nan
    data[..., 1] = np.arange(20, dtype=np.float32).reshape(4, 5)
    data[..., 2] = np.arange(20, 40, dtype=np.float32).reshape(4, 5)

    counts, _x_range, colors = _histogram_summary(data, scope="Stack")

    assert counts is not None
    assert counts.shape[0] == 3
    np.testing.assert_array_equal(counts.sum(axis=1), [0, 20, 20])
    assert colors is not None
    assert [color.name() for color in colors] == ["#ef4444", "#22c55e", "#60a5fa"]


@pytest.mark.parametrize(
    ("dtype", "base"),
    [
        (np.int64, 2**60),
        (np.uint64, np.iinfo(np.uint64).max - 3),
    ],
)
def test_integer_histogram_and_marker_preserve_wide_native_levels(dtype, base):
    data = np.fromiter(
        [int(base), int(base) + 1, int(base) + 1, int(base) + 3],
        dtype=dtype,
        count=4,
    )

    counts, x_range, _colors = _histogram_summary(data, scope="Stack")
    markers = _input_histogram_markers(
        "otsu_threshold",
        data,
        scope="Stack histogram",
        params={"histogram_bins": 256},
    )

    assert x_range == (int(base), int(base) + 3)
    np.testing.assert_array_equal(counts, [1, 2, 0, 1])
    assert markers and isinstance(markers[0][1], int)
    plot = HistogramPlot()
    plot.set_histogram(counts, False, x_range=x_range, markers=markers)
    assert plot._x_min_label == str(int(base))
    assert plot._x_max_label == str(int(base) + 3)
    expected_fraction = (markers[0][1] - int(base)) / 3
    assert plot._x_fraction(markers[0][1]) == expected_fraction


def test_wide_integer_display_span_is_grouped_without_float_collapse():
    data = np.array([0, 2**60], dtype=np.int64)

    counts, x_range, _colors = _histogram_summary(data, scope="Stack")

    assert x_range == (0, 2**60)
    assert counts is not None and counts.size == 128
    assert int(counts.sum()) == 2
    assert counts[0] == 1
    assert counts[-1] == 1


def test_wide_integer_rescale_percentile_markers_keep_exact_fractional_levels(
    qtbot,
):
    base = 2**60
    data = np.asarray([base + offset for offset in range(4)], dtype=np.int64)

    cutoffs = _exact_finite_percentiles(data, (25.0, 75.0))
    markers = _input_histogram_markers(
        "rescale_intensity",
        data,
        scope="Stack",
        params={
            "cutoff_mode": "Percentiles",
            "in_low_percentile": 25.0,
            "in_high_percentile": 75.0,
        },
    )

    assert cutoffs == (
        Fraction(4 * base + 3, 4),
        Fraction(4 * base + 9, 4),
    )
    plot = HistogramPlot()
    qtbot.addWidget(plot)
    plot.set_histogram(
        np.ones(4),
        False,
        x_range=(base, base + 3),
        markers=markers,
    )
    assert plot._x_fraction(markers[0][1]) == 0.25
    assert plot._x_fraction(markers[1][1]) == 0.75


def test_wide_integer_rescale_defaults_do_not_round_past_dtype_limits():
    assert _rescale_dtype_output_range(np.dtype(np.int64)) == (0.0, 1.0, 1.0, 0)
    assert _rescale_dtype_output_range(np.dtype(np.uint64)) == (0.0, 1.0, 1.0, 0)
    assert _rescale_dtype_output_range(np.dtype(np.uint16)) == (
        0.0,
        65_535.0,
        1.0,
        0,
    )


@pytest.mark.parametrize(
    ("operation_id", "params", "expected"),
    [
        (
            "rescale_intensity",
            {
                "cutoff_mode": "Percentiles",
                "in_low_percentile": 0.0,
                "in_high_percentile": 100.0,
            },
            [("low", 0.0), ("high", 100.0)],
        ),
        (
            "clip_intensity",
            {"cutoff_mode": "Data range"},
            [("min", 0.0), ("max", 100.0)],
        ),
    ],
)
def test_full_input_cutoff_markers_do_not_follow_displayed_slice(
    operation_id,
    params,
    expected,
):
    data = np.zeros((2, 10, 10), dtype=np.float32)
    data[1] = 100.0
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("z", "space"),
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
        ),
    )

    markers = _input_histogram_markers(
        operation_id,
        data,
        state=state,
        scope="Slice histogram",
        current_step=(0, 0, 0),
        current_step_nsteps=data.shape,
        params=params,
    )

    assert [(label, value) for label, value, _color in markers] == expected


def test_large_full_input_marker_uses_worker_when_displayed_slice_is_small(
    qtbot,
    monkeypatch,
):
    data = np.zeros((2, 10, 10), dtype=np.float32)
    data[1] = 100.0
    viewer = _Viewer(data, metadata={"axes": "ZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("rescale_intensity")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_BYTES", 10**9)
    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_ELEMENTS", 150)
    widget._input_histogram_cache.clear()
    widget._update_rescale_input_histogram(node.id, widget._current_step())

    assert widget._active_input_histogram_run_id is not None
    qtbot.waitUntil(
        lambda: widget._active_input_histogram_run_id is None,
        timeout=5_000,
    )
    assert [
        (label, value)
        for label, value, _color in widget.rescale_input_histogram_plot._markers
    ] == [("low", 0.0), ("high", 100.0)]


def test_histogram_a_b_a_race_requeues_canceled_original_request(qtbot, monkeypatch):
    data = np.arange(200, dtype=np.float32).reshape(2, 10, 10)
    widget = VippWidget(_Viewer(data, metadata={"axes": "ZYX"}))
    qtbot.addWidget(widget)
    started = []

    def fake_start(request):
        started.append(request)
        widget._active_input_histogram_run_id = 1
        widget._active_input_histogram_key = request.key
        widget._active_input_histogram_cancel_event = threading.Event()

    monkeypatch.setattr(widget, "_start_input_histogram_request", fake_start)
    common = {
        "node_id": "input",
        "operation_id": "binary_threshold",
        "data": data,
        "state": widget.pipeline.output_states.get("input"),
        "current_step_nsteps": data.shape,
        "params": {"threshold": 0.5},
        "title": "Input Histogram",
    }
    widget._queue_input_histogram(
        **common,
        scope="Slice",
        current_step=(0, 0, 0),
    )
    first_key = widget._active_input_histogram_key
    widget._queue_input_histogram(
        **common,
        scope="Slice",
        current_step=(1, 0, 0),
    )
    assert widget._active_input_histogram_cancel_event.is_set()
    widget._queue_input_histogram(
        **common,
        scope="Slice",
        current_step=(0, 0, 0),
    )

    assert len(started) == 1
    assert widget._pending_input_histogram_request is not None
    assert widget._pending_input_histogram_request.key == first_key


def test_minimum_marker_failure_keeps_exact_histogram_visible(qtbot):
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    widget = VippWidget(_Viewer(data))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("minimum_threshold")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and widget._active_input_histogram_run_id is None
        ),
        timeout=10_000,
    )

    assert widget.rescale_input_histogram_plot._counts.sum() == data.size
    assert widget.rescale_input_histogram_plot._markers == []
    assert "marker unavailable" in widget.rescale_input_histogram_group.title()
    assert "Unable to find two maxima" in (
        widget.rescale_input_histogram_group.toolTip()
    )


def test_large_output_histogram_is_calculated_in_background(qtbot, monkeypatch):
    data = np.arange(200, dtype=np.float32).reshape(2, 10, 10)
    viewer = _Viewer(data, metadata={"axes": "ZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    widget.graph_view.select_node("input")

    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_BYTES", 100)
    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_ELEMENTS", 100)
    widget._output_histogram_cache.clear()
    widget._update_histogram()

    qtbot.waitUntil(
        lambda: widget._active_output_histogram_run_id is None,
        timeout=5_000,
    )
    assert widget.histogram_plot._counts.sum() == 100
    assert "all 100 finite pixels" in widget.histogram_group.toolTip()


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
    assert widget.pipeline.nodes[node.id].params["cutoff_mode"] == "Percentiles"
    assert list(widget._parameter_widgets)[:4] == [
        "cutoff_mode",
        "in_low_percentile",
        "in_high_percentile",
        "out_min",
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


def test_rescale_cutoff_modes_keep_inactive_parameters_from_driving_output(qtbot):
    data = np.arange(256, dtype=np.uint8).reshape(1, 16, 16)
    viewer = _Viewer(data)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("rescale_intensity")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    widget._parameter_widgets["in_low_percentile"].value_box.setValue(25.0)

    assert widget.pipeline.nodes[node.id].params["in_low_value"] == 0.0
    assert [
        (label, value)
        for label, value, _color in widget.rescale_input_histogram_plot._markers
    ][0] == ("low", 63.75)

    mode_combo = widget._parameter_widgets["cutoff_mode"].combo
    mode_combo.setCurrentIndex(mode_combo.findData("Values"))
    assert "in_low_value" in widget._parameter_widgets
    assert "in_low_percentile" not in widget._parameter_widgets
    widget._parameter_widgets["in_high_value"].value_box.setValue(127.5)

    assert widget.pipeline.nodes[node.id].params["in_high_percentile"] == 100.0
    assert [
        (label, value)
        for label, value, _color in widget.rescale_input_histogram_plot._markers
    ][1] == ("high", 127.5)

    widget._on_input_histogram_marker_changed("low", 32.0)

    assert widget.pipeline.nodes[node.id].params["in_low_value"] == 32.0
    assert widget._parameter_widgets["in_low_value"].value() == 32.0
    assert widget.pipeline.nodes[node.id].params["in_low_percentile"] == 25.0
    assert [
        (label, value)
        for label, value, _color in widget.rescale_input_histogram_plot._markers
    ][0] == ("low", 32.0)


@pytest.mark.parametrize(
    ("operation_id", "params", "message"),
    [
        (
            "rescale_intensity",
            {
                "cutoff_mode": "Percentiles",
                "in_low_percentile": 90.0,
                "in_high_percentile": 10.0,
            },
            "low percentile must not exceed",
        ),
        (
            "rescale_intensity",
            {
                "cutoff_mode": "Values",
                "in_low_value": 90.0,
                "in_high_value": 10.0,
            },
            "low input value must not exceed",
        ),
        (
            "clip_intensity",
            {"cutoff_mode": "Values", "minimum": 90.0, "maximum": 10.0},
            "Clip minimum must not exceed",
        ),
    ],
)
def test_crossed_cutoffs_show_marker_error_instead_of_silent_reordering(
    qtbot,
    operation_id,
    params,
    message,
):
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    widget = VippWidget(_Viewer(data))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette(operation_id)
    widget._connect_nodes("input", node.id)
    for name, value in params.items():
        widget.pipeline.set_param(node.id, name, value)

    widget.graph_view.select_node(node.id)
    widget._update_rescale_input_histogram(node.id, widget._current_step())

    assert "marker unavailable" in widget.rescale_input_histogram_group.title()
    assert message in widget.rescale_input_histogram_group.toolTip()
    assert widget.rescale_input_histogram_plot._markers == []


def test_clip_intensity_shows_input_and_output_histograms_with_live_markers(qtbot):
    data = np.arange(100, dtype=np.uint16).reshape(1, 10, 10)
    viewer = _Viewer(data)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("clip_intensity")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    assert widget.pipeline.outputs[node.id].dtype == np.uint16
    assert widget.pipeline.nodes[node.id].params["cutoff_mode"] == "Data range"
    assert widget.pipeline.nodes[node.id].params["minimum"] == 0.0
    assert widget.pipeline.nodes[node.id].params["maximum"] == 255.0
    assert not widget.rescale_input_histogram_group.isHidden()
    assert widget.histogram_group.title() == "Output Histogram"
    assert widget.rescale_input_histogram_plot._counts.sum() == 100.0
    assert widget.histogram_plot._counts.sum() == 100.0
    assert [
        (label, value)
        for label, value, _color in widget.rescale_input_histogram_plot._markers
    ] == [("min", 0.0), ("max", 99.0)]

    mode_combo = widget._parameter_widgets["cutoff_mode"].combo
    mode_combo.setCurrentIndex(mode_combo.findData("Values"))
    assert "minimum" in widget._parameter_widgets
    assert [
        (label, value)
        for label, value, _color in widget.rescale_input_histogram_plot._markers
    ] == [("min", 0.0), ("max", 255.0)]
    assert np.isclose(widget._parameter_widgets["minimum"]._bounds.minimum, 0.0)
    assert widget._parameter_widgets["maximum"]._bounds.maximum >= 255.0

    widget._parameter_widgets["minimum"].value_box.setValue(25.0)

    assert widget.pipeline.nodes[node.id].params["minimum"] == 25.0
    assert [
        (label, value)
        for label, value, _color in widget.rescale_input_histogram_plot._markers
    ][0] == ("min", 25.0)

    widget._on_input_histogram_marker_changed("max", 75.0)

    assert widget.pipeline.nodes[node.id].params["maximum"] == 75.0
    assert widget._parameter_widgets["maximum"].value() == 75.0
    assert [
        (label, value)
        for label, value, _color in widget.rescale_input_histogram_plot._markers
    ][1] == ("max", 75.0)


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

    widget._on_input_histogram_marker_changed("threshold", 64.0)

    assert widget.pipeline.nodes[node.id].params["threshold"] == 64.0
    assert widget._parameter_widgets["threshold"].value() == 64.0
    assert [
        (label, value)
        for label, value, _color in widget.rescale_input_histogram_plot._markers
    ] == [("threshold", 64.0)]


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

    widget._on_input_histogram_marker_changed("low", 96.0)

    assert widget.pipeline.nodes[node.id].params["low_threshold"] == 96.0
    assert widget._parameter_widgets["low_threshold"].value() == 96.0
    assert [
        (label, value)
        for label, value, _color in widget.rescale_input_histogram_plot._markers
    ] == [("low", 96.0), ("high", 192.0)]


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
    _assert_rgb_channel_layers(viewer, "VIPP Inspect", (2, 4, 5, 6))
    assert (
        "1. Composite \u2192 RGB: mapped channels to RGB" in widget.history_label.text()
    )


def test_skeleton_graph_overlay_inspects_as_rgb_layer(qtbot):
    data = np.zeros((3, 16, 18), dtype=np.float32)
    data[:, 4:12, 8] = 1
    data[:, 8, 4:14] = 1
    viewer = _Viewer(data, metadata={"axes": "ZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("skeleton_graph_overlay")
    widget._connect_nodes("threshold", node.id)
    widget.run_pipeline()
    widget.inspect_node(node.id)

    assert widget.pipeline.outputs[node.id].shape == (3, 16, 18, 3)
    assert _metadata_value(widget, "Kind") == "RGB image"
    assert _metadata_value(widget, "Dimensions") == "z=3, y=16, x=18, rgb=3"
    _assert_rgb_channel_layers(viewer, "VIPP Inspect", (3, 16, 18))


def test_skeleton_graph_overlay_2d_inspects_as_single_rgb_layer(qtbot):
    data = np.zeros((16, 18), dtype=np.float32)
    data[4:12, 8] = 1
    data[8, 4:14] = 1
    viewer = _Viewer(data, metadata={"axes": "YX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("skeleton_graph_overlay")
    widget._connect_nodes("threshold", node.id)
    widget.run_pipeline()
    widget.inspect_node(node.id)

    assert widget.pipeline.outputs[node.id].shape == (16, 18, 3)
    inspect = viewer.layers["VIPP Inspect"]
    assert inspect.rgb
    assert inspect.metadata["display_rgb"] is True
    assert inspect.data.shape == (16, 18, 3)


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
    _assert_rgb_channel_layers(viewer, "VIPP Inspect", (12, 16, 18))

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
    widget.follow_dims_checkbox.setChecked(True)

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
    widget.follow_dims_checkbox.setChecked(True)

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
        current_step_nsteps=None,
        state=None,
        channel_colors=None,
        contrast_mode="Percentile",
        contrast_scope="Slice",
        contrast_limits=None,
    ):
        arr = np.asarray(data)
        calls.append((tuple(arr.shape), int(arr.max())))
        return None

    monkeypatch.setattr("napari_vipp._widget.make_preview", fake_make_preview)
    widget.pipeline.set_param(split.id, "preview_channel", 2)
    widget._update_thumbnails()

    assert ((2, 4, 5), 30) in calls


def test_split_channels_thumbnail_uses_single_retained_output(qtbot):
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
    node = widget.pipeline.nodes[split.id]
    node.params["preview_channel"] = 0
    widget.pipeline.node_outputs[split.id] = [
        None,
        widget.pipeline.node_outputs[split.id][1],
        None,
    ]
    widget.pipeline.node_output_states[split.id] = [
        None,
        widget.pipeline.node_output_states[split.id][1],
        None,
    ]

    preview_data, preview_state = widget._thumbnail_payload_for_node(
        split.id,
        widget.pipeline.outputs[split.id],
    )

    assert preview_state is widget.pipeline.node_output_states[split.id][1]
    assert preview_data.shape == (2, 4, 5)
    assert int(np.max(preview_data)) == 20


def _assert_split_channel_presentation(
    widget,
    viewer,
    node_id: str,
    output_port: int,
    saved_preview_channel: int,
):
    outputs = widget.pipeline.node_outputs[node_id]
    states = widget.pipeline.node_output_states[node_id]
    expected_value = (output_port + 1) * 10
    preview_data, preview_state = widget._thumbnail_payload_for_node(
        node_id,
        widget.pipeline.outputs[node_id],
    )

    assert preview_data is outputs[output_port]
    assert preview_state is states[output_port]
    assert int(np.max(preview_data)) == expected_value

    inspect_layer = viewer.layers["VIPP Inspect"]
    assert inspect_layer.metadata["node_id"] == node_id
    assert inspect_layer.metadata["output_port"] == output_port
    assert inspect_layer.metadata["vipp_image_state"] == states[output_port].to_dict()
    assert int(np.max(inspect_layer.data)) == expected_value

    port = widget.pipeline.output_ports(node_id)[output_port]
    assert widget.histogram_group.title().endswith(port.label)
    assert widget.histogram_plot._x_min_label == str(expected_value)
    assert widget.histogram_plot._x_max_label == str(expected_value)
    assert _metadata_value(widget, "Value range") == (
        f"{expected_value} to {expected_value}"
    )

    control = widget._parameter_widgets["preview_channel"]
    assert control.value() == output_port
    used_ports = widget._used_split_channel_ports(node_id)
    expected_bounds = (
        (output_port, output_port) if len(used_ports) == 1 else (0, len(outputs) - 1)
    )
    assert (control.slider.minimum(), control.slider.maximum()) == expected_bounds
    assert widget.pipeline.nodes[node_id].params["preview_channel"] == (
        saved_preview_channel
    )


def test_split_channels_presentation_follows_distinct_used_output_ports(qtbot):
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
    first_consumer = widget.add_node_from_palette("gaussian_blur")
    second_consumer = widget.add_node_from_palette("gaussian_blur")
    widget.graph_view.select_node(split.id)

    widget._connect_nodes(split.id, first_consumer.id, source_port=2)
    assert widget._used_split_channel_ports(split.id) == (2,)
    _assert_split_channel_presentation(widget, viewer, split.id, 2, 0)

    # Replacing the sole connection must immediately switch every display surface,
    # while leaving the workflow's saved fallback selection untouched.
    widget._connect_nodes(
        split.id,
        first_consumer.id,
        target_port=0,
        source_port=1,
    )
    assert widget._used_split_channel_ports(split.id) == (1,)
    _assert_split_channel_presentation(widget, viewer, split.id, 1, 0)

    widget.pipeline.set_param(split.id, "preview_channel", 2)
    widget._connect_nodes(split.id, second_consumer.id, source_port=1)
    assert widget._used_split_channel_ports(split.id) == (1,)
    _assert_split_channel_presentation(widget, viewer, split.id, 1, 2)

    # Two distinct used ports are ambiguous, so presentation falls back to the
    # saved selector even though two consumers of one port were not ambiguous.
    widget._connect_nodes(
        split.id,
        second_consumer.id,
        target_port=0,
        source_port=0,
    )
    assert widget._used_split_channel_ports(split.id) == (0, 1)
    _assert_split_channel_presentation(widget, viewer, split.id, 2, 2)

    widget._delete_node(second_consumer.id)
    assert widget._used_split_channel_ports(split.id) == (1,)
    _assert_split_channel_presentation(widget, viewer, split.id, 1, 2)

    # Deleting the sole remaining consumer must refresh back to the saved selector.
    widget._delete_node(first_consumer.id)
    assert widget._used_split_channel_ports(split.id) == ()
    _assert_split_channel_presentation(widget, viewer, split.id, 2, 2)


def test_extract_channel_thumbnail_uses_selected_semantic_channel(qtbot):
    data = np.zeros((2, 3, 5, 6), dtype=np.uint16)
    data[:, 2] = 42
    viewer = _Viewer(data, metadata={"axes": "ZCYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    extract = widget.add_node_from_palette("extract_channel")
    widget._connect_nodes("input", extract.id)
    widget.pipeline.set_param(extract.id, "channel", 2)
    widget.run_pipeline()

    output = widget.pipeline.outputs[extract.id]
    state = widget.pipeline.output_states[extract.id]
    viewer.dims.current_step = (1, 0, 0, 0)
    preview = make_preview(
        output,
        mode="slice",
        current_step=viewer.dims.current_step,
        current_step_nsteps=viewer.dims.nsteps,
        state=state,
    )

    assert output.shape == (2, 5, 6)
    assert state.axis_order == "ZYX"
    assert [axis.source_axis for axis in state.axes] == [0, 2, 3]
    assert preview.shape == (5, 6)
    assert int(np.max(preview)) == 42


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
        current_step_nsteps=None,
        state=None,
        channel_colors=None,
        contrast_mode="Percentile",
        contrast_scope="Slice",
        contrast_limits=None,
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
        current_step_nsteps=None,
        state=None,
        channel_colors=None,
        contrast_mode="Percentile",
        contrast_scope="Slice",
        contrast_limits=None,
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


def test_graph_search_matches_title_operation_tunnel_and_output_tag():
    nodes = (
        GraphNode(
            "counter",
            "quantify_cells",
            "Cell Counter",
            "Measurements",
            "labels",
            "table",
        ),
        GraphNode(
            "batch",
            "batch_output",
            "Batch Output",
            "Image Data",
            "any",
            "any",
            {"tag": "QC mask"},
        ),
    )
    tunnels = (OutputTunnel("Raw Reference", "counter", 0),)

    title_matches = find_graph_matches("cell counter", nodes, tunnels)
    operation_matches = find_graph_matches("quantify_cells", nodes, tunnels)
    tag_matches = find_graph_matches("qc mask", nodes, tunnels)
    tunnel_matches = find_graph_matches("raw reference", nodes, tunnels)

    assert [(match.kind, match.node_id) for match in title_matches] == [
        ("node", "counter")
    ]
    assert title_matches[0].matched_fields == ("title",)
    assert [(match.kind, match.node_id) for match in operation_matches] == [
        ("node", "counter")
    ]
    assert operation_matches[0].matched_fields == ("operation id",)
    assert [(match.kind, match.node_id) for match in tag_matches] == [("node", "batch")]
    assert tag_matches[0].matched_fields == ("output tag",)
    assert [(match.kind, match.tunnel_name) for match in tunnel_matches] == [
        ("tunnel", "Raw Reference")
    ]


def test_graph_search_highlights_and_focuses_node_matches(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.graph_search_edit.setText("otsu")

    assert widget.graph_search_status.text() == "1 match"
    assert widget.graph_search_focus_button.isEnabled()
    assert widget._graph_search_matches[0].node_id == "threshold"
    assert widget.graph_view._cards["threshold"]._search_highlight
    assert not widget.graph_view._cards["input"]._search_highlight
    assert widget.graph_view._proxies["threshold"].opacity() == 1.0
    assert widget.graph_view._proxies["input"].opacity() < 1.0
    assert widget._selected_node_id == "gaussian"

    widget.graph_search_edit.setFocus()
    qtbot.keyClick(widget.graph_search_edit, Qt.Key_Return)

    assert widget._selected_node_id == "threshold"
    assert widget.graph_view._cards["threshold"]._selected
    assert "Focused 'Otsu Threshold'" in widget.status_label.text()


def test_graph_search_focuses_output_tag_and_tunnel_matches(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    batch_output = widget.add_node_from_palette("batch_output")
    widget._connect_nodes("threshold", batch_output.id)
    widget.pipeline.set_param(batch_output.id, "tag", "segmented masks")

    widget.graph_search_edit.setText("segmented masks")
    assert widget._graph_search_matches[0].node_id == batch_output.id
    assert widget.graph_view._cards[batch_output.id]._search_highlight

    widget.graph_search_edit.setFocus()
    qtbot.keyClick(widget.graph_search_edit, Qt.Key_Return)

    assert widget._selected_node_id == batch_output.id
    assert "output tag" in widget.status_label.text()

    widget.pipeline.add_output_tunnel("Raw Reference", "input", 0)
    widget._sync_port_tunnels()
    widget.graph_search_edit.setText("raw reference")

    assert widget._graph_search_matches[0].kind == "tunnel"

    qtbot.keyClick(widget.graph_search_edit, Qt.Key_Return)

    assert widget.graph_view._active_tunnel_name == "Raw Reference"
    assert (
        widget.graph_view._proxies["input"].output_port_at(0)._tunnel_highlight_role
        == "source"
    )
    assert widget.graph_view._proxies["input"].opacity() == 1.0
    assert widget.graph_view._proxies["threshold"].opacity() < 1.0
    assert "Raw Reference" in widget.status_label.text()


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
        current_step_nsteps=None,
        state=None,
        channel_colors=None,
        contrast_mode="Percentile",
        contrast_scope="Slice",
        contrast_limits=None,
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


def test_workflow_roundtrip_restores_inspector_and_optional_thumbnails(
    qtbot,
    tmp_path,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget.graph_view.select_node("gaussian")
    widget.thumbnail_checkbox.setChecked(False)
    widget.graph_view.select_node("threshold")
    widget._set_right_panel_visible(False)

    path = tmp_path / "workflow.json"
    save_workflow(
        path,
        widget.pipeline,
        widget.graph_view.node_positions(),
        widget._graph_note_documents(),
        widget._workflow_metadata(),
    )
    document = json.loads(path.read_text(encoding="utf-8"))

    vipp_metadata = document["metadata"]["vipp"]
    assert vipp_metadata["inspector"] == {
        "right_panel_visible": False,
        "selected_node_id": "threshold",
    }
    assert "thumbnails" not in vipp_metadata

    widget.save_thumbnail_visibility_checkbox.setChecked(True)
    path_with_thumbnails = tmp_path / "workflow-with-thumbnails.json"
    save_workflow(
        path_with_thumbnails,
        widget.pipeline,
        widget.graph_view.node_positions(),
        widget._graph_note_documents(),
        widget._workflow_metadata(),
    )
    document = json.loads(path_with_thumbnails.read_text(encoding="utf-8"))
    assert document["metadata"]["vipp"]["thumbnails"] == {
        "disabled_node_ids": ["gaussian"]
    }

    restored = VippWidget(_Viewer())
    qtbot.addWidget(restored)
    restored._preview_disabled_node_ids.add("threshold")
    restored.load_workflow_file(path_with_thumbnails)

    assert restored._selected_node_id == "threshold"
    assert restored.inspector_panel.isHidden()
    assert restored._preview_disabled_node_ids == {"gaussian"}
    assert restored.thumbnail_checkbox.isChecked()


def test_workflow_load_without_thumbnail_metadata_clears_preview_state(
    qtbot,
    tmp_path,
    monkeypatch,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    path = tmp_path / "workflow-without-thumbnails.json"
    save_workflow(
        path,
        widget.pipeline,
        widget.graph_view.node_positions(),
        widget._graph_note_documents(),
        {
            "vipp": {
                "inspector": {
                    "selected_node_id": "gaussian",
                    "right_panel_visible": True,
                }
            }
        },
    )

    restored = VippWidget(_Viewer())
    qtbot.addWidget(restored)
    restored._preview_disabled_node_ids.add("threshold")
    restored._set_right_panel_visible(False)
    run_calls = []
    original_run_pipeline = restored.run_pipeline

    def recorded_run_pipeline(*args, **kwargs):
        run_calls.append((args, kwargs))
        return original_run_pipeline(*args, **kwargs)

    monkeypatch.setattr(restored, "run_pipeline", recorded_run_pipeline)
    restored.load_workflow_file(path)

    assert restored._selected_node_id == "gaussian"
    assert not restored.inspector_panel.isHidden()
    assert restored._preview_disabled_node_ids == set()
    assert run_calls == [((), {})]


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


def test_large_inputs_automatically_use_background_processing(qtbot, monkeypatch):
    viewer = _Viewer(np.zeros((2, 8, 8), dtype=np.float32))
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    assert widget._background_processing_node_id({"threshold"}) is None
    watershed = widget.pipeline.add_node("auto_watershed_from_mask")
    assert widget._background_processing_node_id({watershed.id}) == watershed.id
    minimum = widget.pipeline.add_node("minimum_threshold")
    assert widget._background_processing_node_id({minimum.id}) == minimum.id

    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_BYTES", 1_000)
    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_ELEMENTS", 1_000)
    large = np.zeros(1_001, dtype=np.uint8)
    widget.pipeline.outputs["gaussian"] = large
    widget.pipeline.node_outputs["gaussian"] = [large]

    assert widget._background_processing_node_id({"threshold"}) == "threshold"

    small = np.zeros(100, dtype=np.uint8)
    widget.pipeline.outputs["gaussian"] = small
    widget.pipeline.node_outputs["gaussian"] = [small]
    assert widget._background_processing_node_id({"threshold"}) is None
    assert (
        widget._background_processing_node_id(
            {"input"},
            source_payloads={"input": SourcePayload(large)},
        )
        == "input"
    )

    widget.background_all_checkbox.setChecked(True)
    assert widget._background_processing_node_id({"threshold"}) == "threshold"


def test_large_viewer_source_defers_exact_metadata_until_background(qtbot, monkeypatch):
    widget = VippWidget(_Viewer(np.zeros((4, 4), dtype=np.float32)))
    qtbot.addWidget(widget)
    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_BYTES", 10**9)
    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_ELEMENTS", 100)
    data = np.arange(101, dtype=np.float32)

    state = widget._viewer_aligned_image_state(data, {}, "large")

    assert state.value_range == "pending exact background calculation"
    assert state.value_pattern == ""


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


def test_parallel_branch_queues_behind_active_deconvolution(qtbot):
    viewer = _Viewer(np.ones((8, 8), dtype=np.float32))
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: widget._active_pipeline_run_id is None,
        timeout=30_000,
    )

    deconvolution = widget.add_node_from_palette("richardson_lucy_deconvolution")
    widget.pipeline.set_param(deconvolution.id, "iterations", 1)
    widget._connect_nodes("input", deconvolution.id, target_port=0)
    widget._connect_nodes("input", deconvolution.id, target_port=1)

    pool = _QueuedThreadPool()
    widget._pipeline_thread_pool = pool
    widget._calculate_node(deconvolution.id)

    run_id = widget._active_pipeline_run_id
    assert run_id is not None
    assert len(pool.workers) == 1
    cancel_event = widget._pipeline_cancel_events[run_id]
    inflight_dirty = set(widget._inflight_dirty_node_ids or set())
    widget._on_background_pipeline_progress(
        (
            run_id,
            deconvolution.id,
            2,
            5,
            "Richardson-Lucy deconvolution",
        )
    )

    parallel = widget.add_node_from_palette("binary_threshold")

    assert widget._active_pipeline_run_id == run_id
    assert widget._pipeline_run_pending is True
    assert parallel.id in widget._pending_dirty_node_ids
    assert not cancel_event.is_set()
    assert not widget.pipeline_busy_bar.isHidden()
    assert widget.pipeline_busy_bar.maximum() == 5
    assert widget.pipeline_busy_bar.value() == 2
    assert widget.graph_view._cards[deconvolution.id].is_processing()
    assert (
        widget.graph_view._cards[deconvolution.id]._execution_summary()
        == "Calculating..."
    )

    widget._connect_nodes("input", parallel.id)

    assert widget._active_pipeline_run_id == run_id
    assert widget._inflight_dirty_node_ids == inflight_dirty
    assert parallel.id in widget._pending_dirty_node_ids
    assert not cancel_event.is_set()
    assert not widget.pipeline_busy_bar.isHidden()
    assert widget.pipeline_busy_bar.maximum() == 5
    assert widget.pipeline_busy_bar.value() == 2
    assert widget.graph_view._cards[deconvolution.id].is_processing()

    # Complete the old workflow snapshot. The unrelated graph addition must not
    # invalidate its deconvolution result; the cheap parallel branch runs next.
    pool.workers[0].run()
    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and not widget._pipeline_run_pending
            and not widget._pending_dirty_node_ids
            and widget.pipeline.outputs.get(deconvolution.id) is not None
            and widget.pipeline.outputs.get(parallel.id) is not None
        ),
        timeout=30_000,
    )

    assert not cancel_event.is_set()
    assert widget.pipeline.outputs[deconvolution.id].shape == (8, 8)


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


def test_cancel_background_run_requeues_dirty_nodes(qtbot):
    viewer = _Viewer(np.ones((8, 8), dtype=np.uint8) * 20)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget._active_pipeline_run_id = 123
    widget._pipeline_run_pending = True
    widget._pipeline_run_context[123] = (None, "input volume", "gaussian", None, None)
    widget._inflight_dirty_node_ids = {"gaussian"}
    widget._set_pipeline_busy(True, "gaussian", queued=True)

    assert not widget.pipeline_cancel_button.isHidden()
    assert widget.graph_view._cards["gaussian"].is_processing()

    widget._cancel_background_pipeline_run()

    assert widget._active_pipeline_run_id is None
    assert widget._pipeline_run_pending is False
    assert 123 not in widget._pipeline_run_context
    assert "gaussian" in widget._pending_dirty_node_ids
    assert widget._inflight_dirty_node_ids is None
    assert widget.pipeline_cancel_button.isHidden()
    assert not widget.graph_view._cards["gaussian"].is_processing()
    assert "result will be ignored" in widget.status_label.text()


def test_affecting_background_request_cancels_active_run_and_remembers_manual(
    qtbot,
):
    viewer = _Viewer(np.ones((8, 8), dtype=np.uint8) * 20)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    measurements = widget.add_node_from_palette("measure_objects")
    cancel_event = threading.Event()

    widget._active_pipeline_run_id = 123
    widget._pipeline_cancel_events[123] = cancel_event
    widget._pipeline_run_context[123] = (None, "input volume", "gaussian", None, None)
    widget._active_pipeline_node_id = "gaussian"

    widget._start_background_pipeline_run(
        None,
        None,
        "",
        {},
        None,
        "input volume",
        ("sources", ()),
        {"input"},
        {measurements.id},
    )

    assert cancel_event.is_set()
    assert widget._pipeline_run_pending is True
    assert "input" in widget._pending_dirty_node_ids
    assert measurements.id in widget._pending_manual_node_ids
    assert "Canceling" in widget.status_label.text()


def test_cancelled_run_with_independent_pending_work_restarts(qtbot, monkeypatch):
    viewer = _Viewer(np.ones((8, 8), dtype=np.uint8) * 20)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    manual = widget.add_node_from_palette("measure_objects")
    parallel = widget.add_node_from_palette("gamma_correction")
    reruns = []
    monkeypatch.setattr(widget, "run_pipeline", lambda: reruns.append("run"))

    widget._active_pipeline_run_id = 123
    widget._active_pipeline_node_id = manual.id
    widget._pipeline_run_pending = True
    widget._pipeline_run_context[123] = (
        None,
        "input volume",
        manual.id,
        widget._last_pipeline_source_signature,
        {manual.id},
    )
    widget._pipeline_run_manual_node_ids[123] = frozenset({manual.id})
    widget._inflight_dirty_node_ids = {manual.id}
    widget._pending_dirty_node_ids = {parallel.id}
    widget.pipeline.node_execution_states[manual.id] = "running"
    widget._set_pipeline_busy(True, manual.id, queued=True)

    widget._on_background_pipeline_finished(PipelineRunResult(123, {}, cancelled=True))

    assert widget._active_pipeline_run_id is None
    assert widget._pipeline_run_pending is False
    assert {manual.id, parallel.id} <= widget._pending_dirty_node_ids
    assert manual.id in widget._pending_manual_node_ids
    assert not widget.pipeline_busy_bar.isHidden()
    assert widget.graph_view._cards[manual.id].is_processing()
    qtbot.waitUntil(lambda: reruns == ["run"], timeout=5_000)


def test_force_sync_supersedes_full_background_scope(qtbot, monkeypatch):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.uint8) * 20))
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: widget._active_pipeline_run_id is None,
        timeout=30_000,
    )
    manual = widget.add_node_from_palette("measure_objects")
    pending = widget.add_node_from_palette("gamma_correction")
    cancel_event = threading.Event()
    captured = []

    widget._active_pipeline_run_id = 123
    widget._active_pipeline_node_id = "gaussian"
    widget._pipeline_cancel_events[123] = cancel_event
    widget._pipeline_run_context[123] = (
        None,
        "input volume",
        "gaussian",
        widget._last_pipeline_source_signature,
        None,
    )
    widget._pipeline_run_manual_node_ids[123] = frozenset({manual.id})
    widget._inflight_dirty_node_ids = None
    widget._pending_dirty_node_ids = {pending.id}
    widget._set_pipeline_busy(True, "gaussian")
    monkeypatch.setattr(
        widget,
        "_run_pipeline_synchronously",
        lambda *args: captured.append(args),
    )

    widget.run_pipeline(force_sync=True)

    assert len(captured) == 1
    assert captured[0][-2] is None
    assert manual.id in captured[0][-1]
    assert cancel_event.is_set()
    assert widget._active_pipeline_run_id is None
    assert widget._pending_dirty_node_ids == set()
    assert 123 not in widget._pipeline_run_context
    assert 123 not in widget._pipeline_run_manual_node_ids
    assert widget.pipeline_busy_bar.isHidden()


def test_background_progress_updates_busy_bar(qtbot):
    viewer = _Viewer(np.ones((8, 8), dtype=np.uint8) * 20)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    rolling = widget.add_node_from_palette("rolling_ball_background")

    widget._active_pipeline_run_id = 321
    widget._set_pipeline_busy(True, "gaussian")

    widget._on_background_pipeline_progress(
        (321, "gaussian", 2, 5, "Rolling-ball background")
    )

    assert widget.pipeline_busy_bar.minimum() == 0
    assert widget.pipeline_busy_bar.maximum() == 5
    assert widget.pipeline_busy_bar.value() == 2
    assert widget.pipeline_busy_bar.isTextVisible()
    assert "Rolling-ball background" in widget.pipeline_busy_label.text()

    widget._set_pipeline_busy(True, rolling.id)
    widget._on_background_pipeline_progress(
        (321, rolling.id, 3, 5, "Rolling-ball background")
    )

    assert widget.pipeline_busy_label.text() == "Processing: Rolling-Ball Background"

    widget._set_pipeline_busy(False)


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
    rescale_calls_before = calls["rescale"]

    widget.graph_view.select_node(rescale.id)
    widget._parameter_widgets["x_scale"].value_box.setValue(1.25)
    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and not widget._pending_dirty_node_ids
            and calls["rescale"] > rescale_calls_before
            and widget.pipeline.nodes[rescale.id].params["x_scale"] == 1.25
        ),
        timeout=30_000,
    )

    assert calls["subtract"] == subtract_calls_before
    assert calls["rescale"] > rescale_calls_before


def test_global_preview_off_skips_thumbnail_generation(qtbot, monkeypatch):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    calls = []

    def fake_make_preview(
        data,
        mode,
        current_step,
        current_step_nsteps=None,
        state=None,
        channel_colors=None,
        contrast_mode="Percentile",
        contrast_scope="Slice",
        contrast_limits=None,
    ):
        calls.append(data)
        return None

    monkeypatch.setattr("napari_vipp._widget.make_preview", fake_make_preview)
    widget.preview_mode_combo.setCurrentText("Off")

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
        current_step_nsteps=None,
        state=None,
        channel_colors=None,
        contrast_mode="Percentile",
        contrast_scope="Slice",
        contrast_limits=None,
    ):
        calls.append((contrast_mode, contrast_scope))
        return None

    monkeypatch.setattr("napari_vipp._widget.make_preview", fake_make_preview)
    widget.thumbnail_contrast_combo.setCurrentText("Raw")
    widget.thumbnail_scope_combo.setCurrentText("Slice")

    assert calls
    assert ("Raw", "Slice") in calls


def test_stack_thumbnail_contrast_limits_are_cached(qtbot, monkeypatch):
    data = np.zeros((5, 16, 18), dtype=np.float32)
    data[4, 8, 9] = 10.0
    viewer = _Viewer(data)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    calls = []
    started = threading.Event()
    release = threading.Event()

    def fake_thumbnail_contrast_limits(
        data,
        *,
        contrast_mode="Percentile",
        data_kind="image",
    ):
        calls.append((id(data), contrast_mode, data_kind))
        if len(calls) == 1:
            started.set()
            assert release.wait(5)
        return (0.0, 10.0)

    monkeypatch.setattr(
        "napari_vipp._widget.thumbnail_contrast_limits",
        fake_thumbnail_contrast_limits,
    )
    widget._clear_thumbnail_contrast_limit_state()
    widget.thumbnail_scope_combo.setCurrentText("Stack")
    widget.thumbnail_contrast_combo.setCurrentText("Min-max")
    widget._update_thumbnails()

    qtbot.waitUntil(started.is_set, timeout=5_000)
    assert not widget.pipeline_busy_bar.isHidden()
    assert "thumbnail contrast" in widget.pipeline_busy_label.text().lower()

    release.set()
    qtbot.waitUntil(
        lambda: (
            widget._active_thumbnail_contrast_run_id is None
            and not widget._pending_thumbnail_contrast_limit_keys
            and len(calls) > 0
        ),
        timeout=5_000,
    )
    first_count = len(calls)

    widget._update_thumbnails()
    viewer.dims.set_current_step(0, 3)
    widget._finish_pipeline_update(None, "input volume")

    assert first_count > 0
    assert len(calls) == first_count


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
        current_step_nsteps=None,
        state=None,
        channel_colors=None,
        contrast_mode="Percentile",
        contrast_scope="Slice",
        contrast_limits=None,
    ):
        return np.zeros((4, 4), dtype=np.uint8)

    def fake_normalize_thumbnail(
        data,
        size=(180, 110),
        *,
        colormap="Gray",
        contrast_mode="Percentile",
        contrast_reference=None,
        contrast_limits=None,
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


def test_adding_unconnected_node_does_not_rerun_cached_pipeline(qtbot, monkeypatch):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    qtbot.addWidget(widget)
    calls = []
    original_run_node = widget.pipeline._run_node

    def counted_run_node(node_id, *args, **kwargs):
        calls.append(node_id)
        return original_run_node(node_id, *args, **kwargs)

    monkeypatch.setattr(widget.pipeline, "_run_node", counted_run_node)

    node = widget.add_node_from_palette("binary_threshold")

    assert node.id in widget.pipeline.nodes
    assert calls == []
    assert widget.pipeline.outputs["gaussian"] is not None
    assert widget.pipeline.outputs["threshold"] is not None


def test_connecting_new_branch_reuses_cached_upstream(qtbot, monkeypatch):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("binary_threshold")
    calls = []
    original_run_node = widget.pipeline._run_node

    def counted_run_node(node_id, *args, **kwargs):
        calls.append(node_id)
        return original_run_node(node_id, *args, **kwargs)

    monkeypatch.setattr(widget.pipeline, "_run_node", counted_run_node)

    widget._connect_nodes("gaussian", node.id)

    assert node.id in calls
    assert "input" not in calls
    assert "gaussian" not in calls
    assert widget.pipeline.outputs[node.id] is not None


def test_inserting_node_on_wire_reuses_cached_source_side(qtbot, monkeypatch):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    qtbot.addWidget(widget)
    calls = []
    original_run_node = widget.pipeline._run_node

    def counted_run_node(node_id, *args, **kwargs):
        calls.append(node_id)
        return original_run_node(node_id, *args, **kwargs)

    monkeypatch.setattr(widget.pipeline, "_run_node", counted_run_node)

    node = widget._insert_node_on_connection(
        "median_filter",
        ("gaussian", "threshold", 0, 0),
        QPointF(250, 100),
    )

    assert node is not None
    assert node.id in calls
    assert "threshold" in calls
    assert "input" not in calls
    assert "gaussian" not in calls


def test_cache_mode_defaults_to_keep_all_and_reports_memory(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.cache_mode_combo.currentText() == CACHE_MODE_KEEP_ALL
    assert widget.memory_limit_spin.value() == 90
    assert not widget._cache_pruning_enabled()
    assert widget.pipeline.outputs["input"] is not None
    assert widget.pipeline.outputs["gaussian"] is not None
    assert widget.pipeline.outputs["threshold"] is not None
    assert widget.cache_status_label.text().startswith("Cache ")
    assert "(Keep all)" in widget.cache_status_label.text()
    assert CACHE_MODE_KEEP_ALL in widget.cache_status_label.toolTip()


def test_smart_cache_prunes_expendable_linear_outputs(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.graph_view.select_node("gaussian")
    widget.cache_mode_combo.setCurrentText(CACHE_MODE_SMART)

    assert widget.pipeline.outputs["input"] is not None
    assert widget.pipeline.outputs["gaussian"] is not None
    assert widget.pipeline.outputs["threshold"] is None
    assert "(Smart interactive)" in widget.cache_status_label.text()


def test_smart_cache_selection_restores_pruned_selected_output(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False

    widget.graph_view.select_node("input")
    widget.cache_mode_combo.setCurrentText(CACHE_MODE_LOW_MEMORY)
    assert widget.pipeline.outputs["threshold"] is None

    widget.cache_mode_combo.setCurrentText(CACHE_MODE_SMART)
    assert widget.pipeline.outputs["threshold"] is None

    widget.graph_view.select_node("threshold")

    assert widget.pipeline.outputs["gaussian"] is not None
    assert widget.pipeline.outputs["threshold"] is not None
    assert widget.graph_view._cards["threshold"].preview.isHidden() is False


def test_large_pruned_threshold_restores_in_background(qtbot, monkeypatch):
    data = np.arange(200, dtype=np.float32).reshape(2, 10, 10)
    widget = VippWidget(_Viewer(data, metadata={"axes": "ZYX"}))
    qtbot.addWidget(widget)
    widget.graph_view.select_node("input")
    widget.cache_mode_combo.setCurrentText(CACHE_MODE_LOW_MEMORY)
    widget.cache_mode_combo.setCurrentText(CACHE_MODE_SMART)
    assert widget.pipeline.outputs["threshold"] is None

    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_BYTES", 100)
    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_ELEMENTS", 100)
    dispatches = []
    monkeypatch.setattr(
        widget,
        "_start_background_pipeline_run",
        lambda *_args, **_kwargs: dispatches.append("pipeline"),
    )

    widget.graph_view.select_node("threshold")

    assert dispatches == ["pipeline"]


def test_low_memory_cache_keeps_working_node_input_and_explicit_outputs(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    batch_output = widget.add_node_from_palette("batch_output")
    widget._connect_nodes("threshold", batch_output.id)
    widget.graph_view.select_node("threshold")
    widget.cache_mode_combo.setCurrentText(CACHE_MODE_LOW_MEMORY)

    assert widget.pipeline.outputs["input"] is None
    assert widget.pipeline.outputs["gaussian"] is not None
    assert widget.pipeline.outputs["threshold"] is not None
    assert widget.pipeline.outputs[batch_output.id] is not None


def test_low_memory_dirty_run_reuses_retained_working_input(qtbot, monkeypatch):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.graph_view.select_node("threshold")
    widget.cache_mode_combo.setCurrentText(CACHE_MODE_LOW_MEMORY)
    assert widget.pipeline.outputs["input"] is None
    assert widget.pipeline.outputs["gaussian"] is not None

    calls = []
    original_run_node = widget.pipeline._run_node

    def counted_run_node(node_id, *args, **kwargs):
        calls.append(node_id)
        return original_run_node(node_id, *args, **kwargs)

    monkeypatch.setattr(widget.pipeline, "_run_node", counted_run_node)

    widget._mark_pipeline_dirty("threshold")
    widget.run_pipeline(force_sync=True)

    assert calls == ["threshold"]
    assert widget.pipeline.outputs["gaussian"] is not None
    assert widget.pipeline.outputs["threshold"] is not None


def test_keep_cached_node_survives_low_memory_pruning(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.graph_view.select_node("gaussian")
    widget.keep_cached_checkbox.setChecked(True)
    assert widget.pipeline.nodes["gaussian"].params["_vipp_keep_cached"] is True

    widget.graph_view.select_node("input")
    widget.cache_mode_combo.setCurrentText(CACHE_MODE_LOW_MEMORY)

    assert widget.pipeline.outputs["input"] is not None
    assert widget.pipeline.outputs["gaussian"] is not None
    assert widget.pipeline.outputs["threshold"] is None


def test_memory_guard_switches_keep_all_to_smart_and_prunes(
    qtbot,
    monkeypatch,
):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    widget.graph_view.select_node("gaussian")
    warnings = []
    monkeypatch.setattr("napari_vipp._widget._pipeline_cache_nbytes", lambda _p: 950)
    monkeypatch.setattr(
        "napari_vipp._widget._system_memory_bytes",
        lambda: (50, 1000),
    )
    monkeypatch.setattr(
        "napari_vipp._widget.QMessageBox.warning",
        lambda *_args: warnings.append(_args),
    )

    message = widget._enforce_memory_guard()

    assert "Memory guard switched cache mode" in message
    assert "reclaimable-memory" in message
    assert widget.cache_mode_combo.currentText() == CACHE_MODE_SMART
    assert widget.pipeline.outputs["input"] is not None
    assert widget.pipeline.outputs["gaussian"] is not None
    assert widget.pipeline.outputs["threshold"] is None
    assert warnings


def test_memory_guard_uses_free_plus_cache_budget(qtbot, monkeypatch):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    monkeypatch.setattr("napari_vipp._widget._pipeline_cache_nbytes", lambda _p: 800)
    monkeypatch.setattr(
        "napari_vipp._widget._system_memory_bytes",
        lambda: (100, 1000),
    )

    assert widget._enforce_memory_guard() == ""
    assert widget.cache_mode_combo.currentText() == CACHE_MODE_KEEP_ALL


def test_memory_guard_can_be_disabled(qtbot, monkeypatch):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    widget.memory_guard_checkbox.setChecked(False)
    monkeypatch.setattr("napari_vipp._widget._pipeline_cache_nbytes", lambda _p: 800)
    monkeypatch.setattr(
        "napari_vipp._widget._system_memory_bytes",
        lambda: (100, 1000),
    )

    assert widget._enforce_memory_guard() == ""
    assert widget.cache_mode_combo.currentText() == CACHE_MODE_KEEP_ALL


def test_named_tunnel_replaces_visible_wire_and_is_undoable(qtbot, monkeypatch):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    monkeypatch.setattr(widget, "_prompt_tunnel_name", lambda *_args: "Raw")
    widget._create_output_tunnel("input", 0)

    assert widget.pipeline.output_tunnel("Raw") is not None
    assert widget.graph_view._proxies["input"].output_port_at(0)._tunnel_label == "Raw"
    output_badge = widget.graph_view._proxies["input"].output_port_at(0)._tunnel_badge
    assert output_badge.kind == "output"
    assert output_badge._label == "Raw"
    assert output_badge.pos().x() > 0
    assert not output_badge.flags() & QGraphicsItem.ItemIgnoresTransformations

    widget._connect_input_to_tunnel("Raw", "gaussian", 0)

    connection = widget.pipeline.tunnel_connection_for_input("gaussian", 0)
    assert connection is not None
    assert connection.source_id == "input"
    assert connection.tunnel_name == "Raw"
    assert not any(
        item.source_id == "input" and item.target_id == "gaussian"
        for item in widget.graph_view._connections
    )
    assert (
        widget.graph_view._proxies["gaussian"].input_port_at(0)._tunnel_label == "Raw"
    )
    input_badge = widget.graph_view._proxies["gaussian"].input_port_at(0)._tunnel_badge
    assert input_badge.kind == "input"
    assert input_badge._label == "Raw"
    assert input_badge.pos().x() < 0
    assert not input_badge.flags() & QGraphicsItem.ItemIgnoresTransformations

    widget.undo()

    assert widget.pipeline.output_tunnel("Raw") is not None
    assert widget.pipeline.tunnel_connection_for_input("gaussian", 0) is None
    assert any(
        item.source_id == "input" and item.target_id == "gaussian"
        for item in widget.graph_view._connections
    )

    widget.undo()

    assert widget.pipeline.output_tunnel("Raw") is None


def test_tunnel_management_summary_highlight_and_rename(qtbot, monkeypatch):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    monkeypatch.setattr(widget, "_prompt_tunnel_name", lambda *_args: "Raw")
    widget._create_output_tunnel("input", 0)
    widget._connect_input_to_tunnel("Raw", "gaussian", 0)

    summaries = widget._tunnel_summaries()
    assert len(summaries) == 1
    assert summaries[0].name == "Raw"
    assert summaries[0].source_id == "input"
    assert summaries[0].source_title == "Image Source"
    assert summaries[0].subscriber_count == 1
    assert summaries[0].subscribers == (("gaussian", "Gaussian Blur", 0),)

    widget._show_tunnel_manager()
    dialog = widget._tunnel_manager_dialog
    assert dialog is not None
    assert dialog.table.rowCount() == 1
    assert dialog.table.item(0, 0).text() == "Raw"
    dialog.filter_edit.setText("gauss")
    assert dialog.table.rowCount() == 1
    dialog.filter_edit.setText("missing")
    assert dialog.table.rowCount() == 0
    assert dialog.selected_tunnel_name() == ""
    dialog.filter_edit.clear()
    assert dialog.table.rowCount() == 1

    widget.pipeline.add_output_tunnel("Mask", "threshold", 0)
    widget._sync_port_tunnels()
    widget._highlight_output_tunnel("Raw")
    assert widget.graph_view._active_tunnel_name == "Raw"
    assert (
        widget.graph_view._proxies["input"].output_port_at(0)._tunnel_highlight_role
        == "source"
    )
    assert (
        widget.graph_view._proxies["gaussian"].input_port_at(0)._tunnel_highlight_role
        == "subscriber"
    )
    assert (
        widget.graph_view._proxies["threshold"].output_port_at(0)._tunnel_highlight_role
        == "dimmed"
    )
    assert widget.graph_view._proxies["threshold"].opacity() < 1.0
    widget.pipeline.remove_output_tunnel("Mask")
    widget._sync_port_tunnels()

    assert widget._rename_output_tunnel_to("Raw", "Reference")
    assert widget.pipeline.output_tunnel("Raw") is None
    assert widget.pipeline.output_tunnel("Reference") is not None
    connection = widget.pipeline.tunnel_connection_for_input("gaussian", 0)
    assert connection is not None
    assert connection.tunnel_name == "Reference"
    assert (
        widget.graph_view._proxies["input"].output_port_at(0)._tunnel_label
        == "Reference"
    )
    assert (
        widget.graph_view._proxies["gaussian"].input_port_at(0)._tunnel_label
        == "Reference"
    )
    assert widget._tunnel_summaries()[0].name == "Reference"
    assert dialog.table.item(0, 0).text() == "Reference"

    widget._remove_output_tunnel("Reference")
    assert widget.pipeline.output_tunnel("Reference") is None
    assert widget.pipeline.tunnel_connection_for_input("gaussian", 0) is None
    assert widget.graph_view._active_tunnel_name == ""
    assert widget.graph_view._proxies["input"].output_port_at(0)._tunnel_label == ""
    assert dialog.table.rowCount() == 0


def test_graph_notes_are_undoable_and_restored(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    note_id = widget._add_graph_note(
        "Review threshold",
        QPointF(25, 35),
        attached_node="gaussian",
    )

    assert note_id in widget._graph_notes
    assert note_id in widget.graph_view._notes
    assert widget.graph_view._notes[note_id].toPlainText() == "Review threshold"
    assert widget._graph_notes[note_id].attached_node == "gaussian"
    assert widget.graph_view._notes[note_id].attached_node == "gaussian"

    widget._set_graph_note_text(note_id, "Review threshold and mask")

    assert widget._graph_notes[note_id].text == "Review threshold and mask"
    assert (
        widget.graph_view._notes[note_id].toPlainText() == "Review threshold and mask"
    )

    widget.undo()

    assert widget._graph_notes[note_id].text == "Review threshold"
    assert widget.graph_view._notes[note_id].toPlainText() == "Review threshold"

    old_pos = QPointF(widget.graph_view._notes[note_id].pos())
    new_pos = QPointF(140, 155)
    widget.graph_view._notes[note_id].setPos(new_pos)
    widget._on_graph_note_moved(note_id, old_pos, new_pos)

    assert widget._graph_notes[note_id].position == (140.0, 155.0)

    widget.undo()

    assert widget._graph_notes[note_id].position == (25.0, 35.0)
    assert widget.graph_view._notes[note_id].pos() == QPointF(25, 35)

    old_node_pos = QPointF(widget.graph_view.node_position("gaussian"))
    old_note_pos = QPointF(widget.graph_view._notes[note_id].pos())
    new_node_pos = old_node_pos + QPointF(80, 30)
    widget.graph_view.apply_node_positions({"gaussian": new_node_pos})
    widget._on_node_moved("gaussian", old_node_pos, new_node_pos)

    assert widget.graph_view.node_position("gaussian") == new_node_pos
    assert widget.graph_view._notes[note_id].pos() == old_note_pos + QPointF(80, 30)
    assert widget._graph_notes[note_id].position == (
        old_note_pos.x() + 80.0,
        old_note_pos.y() + 30.0,
    )

    widget.undo()

    assert widget.graph_view.node_position("gaussian") == old_node_pos
    assert widget.graph_view._notes[note_id].pos() == old_note_pos
    assert widget._graph_notes[note_id].position == (
        old_note_pos.x(),
        old_note_pos.y(),
    )

    widget._delete_graph_note(note_id)

    assert note_id not in widget._graph_notes
    assert note_id not in widget.graph_view._notes

    widget.undo()

    assert note_id in widget._graph_notes
    assert note_id in widget.graph_view._notes


def test_delete_node_removes_attached_graph_notes_with_undo(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    note_id = widget._add_graph_note(
        "Tune blur radius",
        QPointF(220, 40),
        attached_node="gaussian",
    )

    widget._delete_node("gaussian")

    assert "gaussian" not in widget.pipeline.nodes
    assert note_id not in widget._graph_notes
    assert note_id not in widget.graph_view._notes

    widget.undo()

    assert "gaussian" in widget.pipeline.nodes
    assert note_id in widget._graph_notes
    assert widget._graph_notes[note_id].attached_node == "gaussian"
    assert note_id in widget.graph_view._notes


def test_insert_node_on_connection_full_splice_moves_downstream(qtbot, monkeypatch):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    source_before = QPointF(widget.graph_view.node_position("input"))
    target_before = QPointF(widget.graph_view.node_position("gaussian"))

    def fail_mapping_prompt(*_args, **_kwargs):
        raise AssertionError("unambiguous insert should not prompt for ports")

    monkeypatch.setattr(
        widget,
        "_choose_connection_insert_mapping",
        fail_mapping_prompt,
    )

    node = widget._insert_node_on_connection(
        "median_filter",
        ("input", "gaussian", 0, 0),
        QPointF(180, 100),
    )

    assert node is not None
    assert node.operation_id == "median_filter"
    assert ("input", node.id, 0, 0) in {
        (
            connection.source_id,
            connection.target_id,
            connection.target_port,
            connection.source_port,
        )
        for connection in widget.pipeline.connections
    }
    assert (node.id, "gaussian", 0, 0) in {
        (
            connection.source_id,
            connection.target_id,
            connection.target_port,
            connection.source_port,
        )
        for connection in widget.pipeline.connections
    }
    assert ("input", "gaussian") not in {
        (connection.source_id, connection.target_id)
        for connection in widget.pipeline.connections
    }
    assert widget.graph_view.node_position("input") == source_before
    assert widget.graph_view.node_position("gaussian").x() > target_before.x()
    source_rect = widget.graph_view.node_scene_rect("input")
    inserted_rect = widget.graph_view.node_scene_rect(node.id)
    target_rect = widget.graph_view.node_scene_rect("gaussian")
    assert source_rect is not None
    assert inserted_rect is not None
    assert target_rect is not None
    left_gap = inserted_rect.left() - source_rect.right()
    right_gap = target_rect.left() - inserted_rect.right()
    assert left_gap >= widget.INSERT_GAP_PADDING_X
    assert right_gap >= widget.INSERT_GAP_PADDING_X

    widget.undo()

    assert node.id not in widget.pipeline.nodes
    assert ("input", "gaussian") in {
        (connection.source_id, connection.target_id)
        for connection in widget.pipeline.connections
    }
    assert widget.graph_view.node_position("gaussian") == target_before


def test_insert_node_on_connection_does_not_shift_when_gap_is_sufficient(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.graph_view.apply_node_positions(
        {
            "input": QPointF(0, 20),
            "gaussian": QPointF(900, 20),
            "threshold": QPointF(1230, 20),
        }
    )
    source_rect = widget.graph_view.node_scene_rect("input")
    target_rect = widget.graph_view.node_scene_rect("gaussian")
    assert source_rect is not None
    assert target_rect is not None
    generous_gap = 10.0 * max(source_rect.width(), target_rect.width())
    target_delta_x = source_rect.right() + generous_gap - target_rect.left()
    widget.graph_view.apply_node_positions(
        {
            "gaussian": widget.graph_view.node_position("gaussian")
            + QPointF(target_delta_x, 0),
            "threshold": widget.graph_view.node_position("threshold")
            + QPointF(target_delta_x, 0),
        }
    )
    target_before = QPointF(widget.graph_view.node_position("gaussian"))
    downstream_before = QPointF(widget.graph_view.node_position("threshold"))
    source_rect = widget.graph_view.node_scene_rect("input")
    target_rect = widget.graph_view.node_scene_rect("gaussian")
    assert source_rect is not None
    assert target_rect is not None
    insertion_point = QPointF(
        (source_rect.right() + target_rect.left()) / 2.0,
        (source_rect.center().y() + target_rect.center().y()) / 2.0,
    )

    node = widget._insert_node_on_connection(
        "median_filter",
        ("input", "gaussian", 0, 0),
        insertion_point,
    )

    assert node is not None
    assert widget.graph_view.node_position("gaussian") == target_before
    assert widget.graph_view.node_position("threshold") == downstream_before
    source_rect = widget.graph_view.node_scene_rect("input")
    inserted_rect = widget.graph_view.node_scene_rect(node.id)
    target_rect = widget.graph_view.node_scene_rect("gaussian")
    assert source_rect is not None
    assert inserted_rect is not None
    assert target_rect is not None
    assert inserted_rect.left() - source_rect.right() >= widget.INSERT_GAP_PADDING_X
    assert target_rect.left() - inserted_rect.right() >= widget.INSERT_GAP_PADDING_X


def test_insert_node_on_connection_ambiguous_chooser_options(qtbot):
    viewer = _Viewer(np.zeros((12, 8, 9), dtype=np.uint8), metadata={"axes": "ZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    connection_key = ("input", "gaussian", 0, 0)

    channel_mode, channel_reason = widget._connection_insert_mode(
        "split_channels",
        connection_key,
    )
    axis_mode, _reason = widget._connection_insert_mode("split_axis", connection_key)
    preview_mode, preview_text = widget._connection_insert_preview_state(
        "split_axis",
        connection_key,
    )
    axis_options = widget._connection_insert_mapping_options(
        "split_axis",
        connection_key,
        params_override={"axis": "axis:0"},
    )
    add_options = widget._connection_insert_mapping_options(
        "add_images",
        connection_key,
    )

    assert channel_mode == "incompatible"
    assert "Split Channels" in channel_reason
    assert axis_mode == "choose"
    assert preview_mode == "partial"
    assert "choose ports" in preview_text
    assert [(mapping.input_port, mapping.output_port) for mapping in axis_options] == [
        (0, index) for index in range(12)
    ]
    assert axis_options[10].output_label == "output 11: Z 11"
    assert axis_options[10].params == (("axis", "axis:0"),)
    assert [(mapping.input_port, mapping.output_port) for mapping in add_options] == [
        (0, 0),
        (1, 0),
    ]
    assert add_options[1].input_label == "input 2: Input 2"


def test_split_axis_insert_mapping_dialog_switches_axis_options(qtbot):
    z_mapping = ConnectionInsertPortMapping(
        0,
        10,
        "input 1: Input 1",
        "output 11: Z 11",
        "Image Source -> Input 1; Z 11 -> Gaussian Blur input 1",
        params=(("axis", "axis:0"),),
    )
    t_mapping = ConnectionInsertPortMapping(
        0,
        2,
        "input 1: Input 1",
        "output 3: T 3",
        "Image Source -> Input 1; T 3 -> Gaussian Blur input 1",
        params=(("axis", "axis:1"),),
    )

    dialog = ConnectionInsertMappingDialog(
        [z_mapping],
        "Split Axis",
        "Image Source",
        "Gaussian Blur",
        axis_choices=[
            ("axis:0", "Z axis (space, size 12)"),
            ("axis:1", "T axis (time, size 3)"),
        ],
        mappings_by_axis={
            "axis:0": [z_mapping],
            "axis:1": [t_mapping],
        },
    )
    qtbot.addWidget(dialog)

    assert dialog.selected_mapping() == z_mapping

    dialog.axis_combo.setCurrentIndex(1)

    assert dialog.selected_mapping() == t_mapping


def test_split_axis_inspector_offers_semantic_axis_choices(qtbot):
    viewer = _Viewer(
        np.zeros((2, 3, 4, 16, 18), dtype=np.uint8),
        metadata={"axes": "TCZYX"},
    )
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("split_axis")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    control = widget._parameter_widgets["axis"]
    labels = [control.combo.itemText(index) for index in range(control.combo.count())]

    assert labels == [
        "T axis (time, size 2)",
        "C axis (channel, size 3)",
        "Z axis (space, size 4)",
    ]

    control.combo.setCurrentIndex(labels.index("Z axis (space, size 4)"))

    assert node.params["axis"] == "axis:2"
    assert len(widget.pipeline.output_ports(node.id)) == 4
    assert widget.pipeline.output_ports(node.id)[3].label == "Z 4"


def test_insert_node_on_connection_applies_selected_mapping(qtbot, monkeypatch):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    connection_key = ("input", "gaussian", 0, 0)
    selected = widget._connection_insert_mapping_options(
        "add_images",
        connection_key,
    )[1]

    monkeypatch.setattr(
        widget,
        "_choose_connection_insert_mapping",
        lambda *_args, **_kwargs: selected,
    )

    node = widget._insert_node_on_connection(
        "add_images",
        connection_key,
        QPointF(180, 100),
    )

    assert node is not None
    assert node.operation_id == "add_images"
    assert ("input", node.id, 1, 0) in {
        (
            connection.source_id,
            connection.target_id,
            connection.target_port,
            connection.source_port,
        )
        for connection in widget.pipeline.connections
    }
    assert (node.id, "gaussian", 0, 0) in {
        (
            connection.source_id,
            connection.target_id,
            connection.target_port,
            connection.source_port,
        )
        for connection in widget.pipeline.connections
    }
    assert ("input", "gaussian") not in {
        (connection.source_id, connection.target_id)
        for connection in widget.pipeline.connections
    }
    assert "using input 2: Input 2 and output 1: out" in widget.status_label.text()


def test_insert_split_axis_on_connection_applies_inferred_high_output(
    qtbot,
    monkeypatch,
):
    viewer = _Viewer(np.zeros((12, 8, 9), dtype=np.uint8), metadata={"axes": "ZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    connection_key = ("input", "gaussian", 0, 0)
    selected = widget._connection_insert_mapping_options(
        "split_axis",
        connection_key,
        params_override={"axis": "axis:0"},
    )[10]

    monkeypatch.setattr(
        widget,
        "_choose_connection_insert_mapping",
        lambda *_args, **_kwargs: selected,
    )

    node = widget._insert_node_on_connection(
        "split_axis",
        connection_key,
        QPointF(180, 100),
    )

    assert node is not None
    assert node.operation_id == "split_axis"
    assert node.params["axis"] == "axis:0"
    assert (node.id, "gaussian", 0, 10) in {
        (
            connection.source_id,
            connection.target_id,
            connection.target_port,
            connection.source_port,
        )
        for connection in widget.pipeline.connections
    }
    assert len(widget.pipeline.output_ports(node.id)) == 12
    assert widget.pipeline.output_ports(node.id)[10].label == "Z 11"
    assert "using input 1: Input 1 and output 11: Z 11" in widget.status_label.text()


def test_insert_existing_loose_node_on_connection_full_splice_is_undoable(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("median_filter")
    node_count = len(widget.pipeline.nodes)
    old_pos = QPointF(widget.graph_view.node_position(node.id))
    target_before = QPointF(widget.graph_view.node_position("gaussian"))
    widget.graph_view.center_node_on(node.id, QPointF(180, 100))

    result = widget._insert_existing_node_on_connection(
        node.id,
        ("input", "gaussian", 0, 0),
        old_pos,
        widget.graph_view.node_position(node.id),
    )

    assert result is node
    assert len(widget.pipeline.nodes) == node_count
    assert ("input", node.id, 0, 0) in {
        (
            connection.source_id,
            connection.target_id,
            connection.target_port,
            connection.source_port,
        )
        for connection in widget.pipeline.connections
    }
    assert (node.id, "gaussian", 0, 0) in {
        (
            connection.source_id,
            connection.target_id,
            connection.target_port,
            connection.source_port,
        )
        for connection in widget.pipeline.connections
    }
    assert ("input", "gaussian") not in {
        (connection.source_id, connection.target_id)
        for connection in widget.pipeline.connections
    }
    assert widget.graph_view.node_position("gaussian").x() > target_before.x()
    assert "Inserted existing" in widget.status_label.text()

    widget.undo()

    assert node.id in widget.pipeline.nodes
    assert widget.graph_view.node_position(node.id) == old_pos
    assert widget.graph_view.node_position("gaussian") == target_before
    assert ("input", "gaussian") in {
        (connection.source_id, connection.target_id)
        for connection in widget.pipeline.connections
    }
    assert not any(
        connection.source_id == node.id or connection.target_id == node.id
        for connection in widget.pipeline.connections
    )


def test_insert_existing_split_axis_on_connection_applies_selected_mapping(
    qtbot,
    monkeypatch,
):
    viewer = _Viewer(np.zeros((12, 8, 9), dtype=np.uint8), metadata={"axes": "ZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("split_axis")
    old_pos = QPointF(widget.graph_view.node_position(node.id))
    widget.graph_view.center_node_on(node.id, QPointF(180, 100))
    connection_key = ("input", "gaussian", 0, 0)
    selected = widget._connection_insert_mapping_options(
        "split_axis",
        connection_key,
        inserted_node_id=node.id,
        params_override={"axis": "axis:0"},
    )[10]

    monkeypatch.setattr(
        widget,
        "_choose_connection_insert_mapping",
        lambda *_args, **_kwargs: selected,
    )

    result = widget._insert_existing_node_on_connection(
        node.id,
        connection_key,
        old_pos,
        widget.graph_view.node_position(node.id),
    )

    assert result is node
    assert node.params["axis"] == "axis:0"
    assert ("input", node.id, 0, 0) in {
        (
            connection.source_id,
            connection.target_id,
            connection.target_port,
            connection.source_port,
        )
        for connection in widget.pipeline.connections
    }
    assert (node.id, "gaussian", 0, 10) in {
        (
            connection.source_id,
            connection.target_id,
            connection.target_port,
            connection.source_port,
        )
        for connection in widget.pipeline.connections
    }
    assert ("input", "gaussian") not in {
        (connection.source_id, connection.target_id)
        for connection in widget.pipeline.connections
    }
    assert len(widget.pipeline.output_ports(node.id)) == 12
    assert widget.pipeline.output_ports(node.id)[10].label == "Z 11"
    assert "using input 1: Input 1 and output 11: Z 11" in widget.status_label.text()


def test_insert_existing_connected_node_is_rejected_without_rewiring(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    before_connections = {
        (
            connection.source_id,
            connection.target_id,
            connection.target_port,
            connection.source_port,
        )
        for connection in widget.pipeline.connections
    }
    old_pos = QPointF(widget.graph_view.node_position("gaussian"))
    widget.graph_view.center_node_on("gaussian", QPointF(180, 100))

    result = widget._insert_existing_node_on_connection(
        "gaussian",
        ("input", "gaussian", 0, 0),
        old_pos,
        widget.graph_view.node_position("gaussian"),
    )

    after_connections = {
        (
            connection.source_id,
            connection.target_id,
            connection.target_port,
            connection.source_port,
        )
        for connection in widget.pipeline.connections
    }
    assert result is None
    assert after_connections == before_connections
    assert "Disconnect 'Gaussian Blur' before inserting it on a wire." in (
        widget.status_label.text()
    )

    widget.undo()

    assert widget.graph_view.node_position("gaussian") == old_pos


def test_connection_insert_candidates_show_modes(qtbot):
    viewer = _Viewer(np.zeros((3, 16, 18), dtype=np.uint8), metadata={"axes": "CYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    candidates = {
        candidate.operation_id: candidate
        for candidate in widget._connection_insert_candidates(
            ("input", "gaussian", 0, 0)
        )
    }

    assert candidates["median_filter"].mode == "full"
    assert candidates["split_channels"].mode == "choose"
    assert candidates["split_axis"].mode == "choose"
    assert candidates["add_images"].mode == "choose"
    assert "Full splice" in candidates["median_filter"].detail
    assert "Choose ports" in candidates["split_channels"].detail
    assert "measure_objects" not in candidates


def test_connection_insert_dialog_filters_candidates(qtbot):
    viewer = _Viewer(np.zeros((12, 8, 9), dtype=np.uint8), metadata={"axes": "ZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    candidates = widget._connection_insert_candidates(("input", "gaussian", 0, 0))

    dialog = ConnectionInsertDialog(candidates, widget)
    qtbot.addWidget(dialog)

    dialog.search.setText("split")

    assert dialog.selected_operation_id() == "split_axis"
    assert dialog.ok_button.isEnabled()


def test_connection_menu_insert_uses_selected_candidate(qtbot, monkeypatch):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    monkeypatch.setattr(
        widget,
        "_choose_connection_insert_operation",
        lambda _connection_key: "median_filter",
    )

    widget._insert_node_from_connection_menu(
        ("input", "gaussian", 0, 0),
        QPointF(180, 100),
    )

    inserted_nodes = [
        node
        for node in widget.pipeline.nodes.values()
        if node.operation_id == "median_filter"
    ]
    assert len(inserted_nodes) == 1
    inserted = inserted_nodes[0]
    assert ("input", inserted.id) in {
        (connection.source_id, connection.target_id)
        for connection in widget.pipeline.connections
    }
    assert (inserted.id, "gaussian") in {
        (connection.source_id, connection.target_id)
        for connection in widget.pipeline.connections
    }


def test_auto_structure_graph_is_undoable_position_only_edit(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    messy_positions = {
        "input": (520, 280),
        "gaussian": (40, 460),
        "threshold": (180, 30),
    }
    widget.graph_view.apply_node_positions(messy_positions)
    note_id = widget._add_graph_note(
        "Check blur",
        QPointF(90, 500),
        attached_node="gaussian",
    )
    messy_note_pos = QPointF(widget.graph_view._notes[note_id].pos())
    before_connections = {
        (
            connection.source_id,
            connection.target_id,
            connection.target_port,
            connection.source_port,
        )
        for connection in widget.pipeline.connections
    }

    widget._auto_structure_graph()

    after_connections = {
        (
            connection.source_id,
            connection.target_id,
            connection.target_port,
            connection.source_port,
        )
        for connection in widget.pipeline.connections
    }
    assert after_connections == before_connections
    input_pos = widget.graph_view.node_position("input")
    gaussian_pos = widget.graph_view.node_position("gaussian")
    threshold_pos = widget.graph_view.node_position("threshold")
    assert input_pos is not None
    assert gaussian_pos is not None
    assert threshold_pos is not None
    assert input_pos.x() < gaussian_pos.x()
    assert gaussian_pos.x() < threshold_pos.x()
    structured_note_pos = QPointF(widget.graph_view._notes[note_id].pos())
    assert structured_note_pos != messy_note_pos
    assert widget._graph_notes[note_id].position == (
        structured_note_pos.x(),
        structured_note_pos.y(),
    )
    assert "Auto-structured graph layout" in widget.status_label.text()

    widget.undo()

    assert widget.graph_view.node_positions() == {
        node_id: (float(x), float(y)) for node_id, (x, y) in messy_positions.items()
    }
    assert widget.graph_view._notes[note_id].pos() == messy_note_pos


def test_toolbar_compacts_in_stages_when_space_runs_out(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.resize(1600, 600)
    widget._sync_toolbar_responsive_mode()

    assert widget.settings_menu_button.isHidden() is False
    assert widget.settings_menu_button.text() == "Settings"
    assert widget.settings_menu_button.minimumWidth() >= 96
    assert widget.background_all_checkbox.isHidden()
    assert widget.follow_dims_checkbox.isHidden()
    assert widget.preview_mode_combo.isHidden() is False
    assert widget.thumbnail_contrast_combo.isHidden() is False
    assert widget.thumbnail_scope_combo.isHidden() is False
    assert widget.graph_zoom_slider.isHidden() is False
    assert widget.save_workflow_button.isHidden() is False
    assert widget.export_button.isHidden() is False
    assert widget.auto_structure_button.text() == "Auto structure graph"

    widget.resize(1400, 600)
    widget._sync_toolbar_responsive_mode()

    assert widget.preview_mode_combo.isHidden()
    assert widget.thumbnail_contrast_combo.isHidden()
    assert widget.thumbnail_scope_combo.isHidden()
    assert widget.thumbnail_colormap_combo.isHidden()
    assert widget.graph_zoom_slider.isHidden() is False

    widget.resize(1200, 600)
    widget._sync_toolbar_responsive_mode()

    assert widget.graph_zoom_slider.isHidden() is False
    assert widget.graph_zoom_reset_button.isHidden() is False
    assert widget.graph_zoom_label.isHidden() is False

    widget.resize(1000, 600)
    widget._sync_toolbar_responsive_mode()

    assert widget.graph_zoom_slider.isHidden()
    assert widget.graph_zoom_reset_button.isHidden()
    assert widget.graph_zoom_label.isHidden()

    widget.resize(widget.TOOLBAR_HIDE_CHECKBOXES_WIDTH + 100, 600)
    widget._sync_toolbar_responsive_mode()

    assert widget.settings_menu_button.isHidden() is False
    assert widget.background_all_checkbox.isHidden()
    assert widget.follow_dims_checkbox.isHidden()
    assert widget.preview_mode_combo.isHidden() is False
    assert widget.thumbnail_contrast_combo.isHidden() is False
    assert widget.thumbnail_scope_combo.isHidden() is False
    assert widget.graph_zoom_slider.isHidden() is False
    assert widget.save_workflow_button.isHidden() is False
    assert widget.export_button.isHidden() is False
    assert widget.auto_structure_button.text() == "Auto structure graph"


def test_settings_menu_shows_controls_hidden_at_current_stage(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.resize(1600, 600)
    widget._sync_toolbar_responsive_mode()
    widget._populate_settings_toolbar_menu()

    labels = [
        action.text()
        for action in widget.settings_menu.actions()
        if not action.isSeparator() and action.text()
    ]
    assert "Show thumbnails" not in labels
    assert "Save thumbnail visibility in workflows" in labels
    assert "Run all in background" in labels
    assert "Link napari/VIPP sliders" in labels
    assert "Cache mode" in labels
    assert "Auto memory guard" in labels
    assert "Preview mode" not in labels
    assert "Contrast range" not in labels
    actions = widget.settings_menu.actions()
    link_action = next(
        action for action in actions if action.text() == "Link napari/VIPP sliders"
    )
    cache_mode_action = next(
        action for action in actions if action.text() == "Cache mode"
    )
    assert widget.follow_dims_checkbox.isChecked()
    assert link_action.isChecked()
    assert any(
        action.isSeparator()
        for action in actions[
            actions.index(link_action) + 1 : actions.index(cache_mode_action)
        ]
    )

    cache_limit_widget = None
    for action in actions:
        default_widget = getattr(action, "defaultWidget", lambda: None)()
        if default_widget is None:
            continue
        label_widget = default_widget.findChild(QLabel)
        if label_widget is not None and label_widget.text() == "Cache limit":
            cache_limit_widget = default_widget
            break
    assert cache_limit_widget is not None
    assert cache_limit_widget.findChild(QLabel).font() == widget.settings_menu.font()
    assert cache_limit_widget.findChild(QSpinBox).font() == widget.settings_menu.font()

    widget.resize(1200, 600)
    widget._sync_toolbar_responsive_mode()
    widget._populate_settings_toolbar_menu()
    labels = [
        action.text()
        for action in widget.settings_menu.actions()
        if not action.isSeparator() and action.text()
    ]
    assert "Preview mode" in labels
    assert "Thumbnail contrast" in labels
    assert "Contrast range" in labels
    assert "Monochrome colormap" in labels

    save_thumbnail_action = next(
        action
        for action in widget.settings_menu.actions()
        if action.text() == "Save thumbnail visibility in workflows"
    )
    assert not widget.save_thumbnail_visibility_checkbox.isChecked()
    save_thumbnail_action.trigger()
    assert widget.save_thumbnail_visibility_checkbox.isChecked()


def test_palette_registry_nodes_are_constructible():
    pipeline = PrototypePipeline()
    palette_ids = [spec.id for spec in PALETTE_NODE_LIBRARY]

    assert len(palette_ids) == len(set(palette_ids))
    for spec in PALETTE_NODE_LIBRARY:
        node = pipeline.add_node(spec.id)
        assert node.operation_id == spec.id
        assert node.params == {param.name: param.default for param in spec.parameters}
        assert pipeline.node_parameter_specs(node.id) == spec.parameters
        assert pipeline.remove_node(node.id)


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

    assert widget.execution_group.isHidden() is False
    assert (
        widget.pipeline.node_execution_states[measurements.id]
        == EXECUTION_NOT_CALCULATED
    )
    assert widget.table_group.isHidden()

    widget.run_pipeline(force_sync=True, manual_node_ids={measurements.id})

    assert widget.pipeline.node_execution_states[measurements.id] == EXECUTION_READY
    widget.graph_view.select_node(measurements.id)
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
    assert "include_derived_shape_ratios" in widget._parameter_widgets
    assert "include_2d_boundary_descriptors" not in widget._parameter_widgets
    assert "include_2d_shape_moments" not in widget._parameter_widgets
    widget.graph_view.select_node(labels.id)
    assert not widget.thumbnail_checkbox.isHidden()
    assert widget.thumbnail_checkbox.isEnabled()
    assert "label_id" in path.read_text(encoding="utf-8")


def test_select_table_columns_uses_detected_column_checklist(qtbot):
    image = np.zeros((9, 9), dtype=np.float32)
    image[1:4, 1:4] = 10
    image[6, 6] = 10
    viewer = _Viewer(image, metadata={"axes": "YX"})
    widget = VippWidget(viewer)
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    qtbot.addWidget(widget)
    threshold = widget.add_node_from_palette("binary_threshold")
    labels = widget.add_node_from_palette("label_connected_components")
    measurements = widget.add_node_from_palette("measure_objects")
    selected = widget.add_node_from_palette("select_table_columns")
    widget.pipeline.set_param(threshold.id, "threshold", 5)
    widget._connect_nodes("input", threshold.id)
    widget._connect_nodes(threshold.id, labels.id)
    widget._connect_nodes(labels.id, measurements.id)
    widget._connect_nodes(measurements.id, selected.id)
    widget.run_pipeline(force_sync=True, manual_node_ids={measurements.id})

    widget.graph_view.select_node(selected.id)
    control = widget._parameter_widgets["columns"]

    assert isinstance(control, SelectTableColumnsControl)
    assert "selection_mode" not in widget.pipeline.nodes[selected.id].params
    assert "append_unlisted" not in widget.pipeline.nodes[selected.id].params
    assert control.list_widget.count() > 0
    assert control.list_widget.item(0).text() == "label_id"
    assert "area_pixels" in [
        control.list_widget.item(row).text()
        for row in range(control.list_widget.count())
    ]

    control.deselect_all_button.click()
    assert (
        widget.pipeline.nodes[selected.id].params["columns"] == NO_TABLE_COLUMNS_VALUE
    )
    widget.run_pipeline(force_sync=True)
    assert widget.pipeline.outputs[selected.id].columns == ()
    assert widget.pipeline.outputs[selected.id].row_count == 2

    control.select_all_button.click()
    control.reset_button.click()
    assert widget.pipeline.nodes[selected.id].params["columns"] == "auto"


def test_select_table_columns_preserves_saved_selection_until_input_is_ready(qtbot):
    image = np.zeros((9, 9), dtype=np.float32)
    image[1:4, 1:4] = 10
    viewer = _Viewer(image, metadata={"axes": "YX"})
    widget = VippWidget(viewer)
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    qtbot.addWidget(widget)
    threshold = widget.add_node_from_palette("binary_threshold")
    labels = widget.add_node_from_palette("label_connected_components")
    measurements = widget.add_node_from_palette("measure_objects")
    selected = widget.add_node_from_palette("select_table_columns")
    widget.pipeline.set_param(threshold.id, "threshold", 5)
    widget.pipeline.set_param(
        selected.id,
        "columns",
        "label_id,area_pixels",
    )
    widget._connect_nodes("input", threshold.id)
    widget._connect_nodes(threshold.id, labels.id)
    widget._connect_nodes(labels.id, measurements.id)
    widget._connect_nodes(measurements.id, selected.id)

    widget.graph_view.select_node(selected.id)

    assert widget.pipeline.nodes[selected.id].params["columns"] == (
        "label_id,area_pixels"
    )

    widget.run_pipeline(force_sync=True, manual_node_ids={measurements.id})

    assert widget.pipeline.nodes[selected.id].params["columns"] == (
        "label_id,area_pixels"
    )
    assert widget.pipeline.outputs[selected.id].columns == (
        "label_id",
        "area_pixels",
    )


def test_manual_node_auto_recalculate_updates_and_hides_button(qtbot):
    image = np.zeros((9, 9), dtype=np.float32)
    image[1:4, 1:4] = 10
    image[6, 6] = 10
    viewer = _Viewer(image, metadata={"axes": "YX"})
    widget = VippWidget(viewer)
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    qtbot.addWidget(widget)
    threshold = widget.add_node_from_palette("binary_threshold")
    labels = widget.add_node_from_palette("label_connected_components")
    measurements = widget.add_node_from_palette("measure_objects")
    widget.pipeline.set_param(threshold.id, "threshold", 5)
    widget._connect_nodes("input", threshold.id)
    widget._connect_nodes(threshold.id, labels.id)
    widget._connect_nodes(labels.id, measurements.id)

    widget.graph_view.select_node(measurements.id)

    assert not widget.auto_recalculate_checkbox.isChecked()
    assert not widget.calculate_button.isHidden()
    assert widget.graph_view._cards[measurements.id].calculate_button.isVisible()

    widget.auto_recalculate_checkbox.setChecked(True)

    assert widget.pipeline.node_auto_recalculate(measurements.id)
    assert widget.calculate_button.isHidden()
    assert not widget.graph_view._cards[measurements.id].calculate_button.isVisible()
    assert widget.pipeline.node_execution_states[measurements.id] == EXECUTION_READY
    assert widget.pipeline.outputs[measurements.id].row_count == 2

    widget.pipeline.set_param(threshold.id, "threshold", 20)
    widget._mark_pipeline_dirty(threshold.id)
    widget.run_pipeline(force_sync=True)

    assert widget.pipeline.node_execution_states[measurements.id] == EXECUTION_READY
    assert widget.pipeline.outputs[measurements.id].row_count == 0

    widget.auto_recalculate_checkbox.setChecked(False)
    assert not widget.pipeline.node_auto_recalculate(measurements.id)
    assert not widget.calculate_button.isHidden()
    assert widget.graph_view._cards[measurements.id].calculate_button.isVisible()

    widget.pipeline.set_param(threshold.id, "threshold", 5)
    widget._mark_pipeline_dirty(threshold.id)
    widget.run_pipeline(force_sync=True)

    assert widget.pipeline.node_execution_states[measurements.id] == EXECUTION_STALE
    assert widget.pipeline.outputs[measurements.id].row_count == 0


def test_calculate_all_button_runs_all_manual_nodes_needing_work(qtbot):
    image = np.zeros((9, 9), dtype=np.float32)
    image[1:4, 1:4] = 10
    image[6, 6] = 10
    viewer = _Viewer(image, metadata={"axes": "YX"})
    widget = VippWidget(viewer)
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    qtbot.addWidget(widget)
    threshold = widget.add_node_from_palette("binary_threshold")
    labels = widget.add_node_from_palette("label_connected_components")
    measurements = widget.add_node_from_palette("measure_objects")
    intensity = widget.add_node_from_palette("measure_objects_intensity")
    widget.pipeline.set_param(threshold.id, "threshold", 5)
    widget._connect_nodes("input", threshold.id)
    widget._connect_nodes(threshold.id, labels.id)
    widget._connect_nodes(labels.id, measurements.id)
    widget._connect_nodes(labels.id, intensity.id, target_port=0)
    widget._connect_nodes("input", intensity.id, target_port=1)

    widget.run_pipeline(force_sync=True)

    assert widget.calculate_all_button.text() == "Calculate all"
    assert widget.pipeline.node_execution_states[measurements.id] == (
        EXECUTION_NOT_CALCULATED
    )
    assert widget.pipeline.node_execution_states[intensity.id] == (
        EXECUTION_NOT_CALCULATED
    )
    assert widget._manual_node_ids_needing_calculation() == {
        measurements.id,
        intensity.id,
    }

    widget.calculate_all_button.click()

    assert widget.pipeline.node_execution_states[measurements.id] == EXECUTION_READY
    assert widget.pipeline.node_execution_states[intensity.id] == EXECUTION_READY
    assert widget.pipeline.outputs[measurements.id].row_count == 2
    assert widget.pipeline.outputs[intensity.id].row_count == 2
    assert widget._manual_node_ids_needing_calculation() == set()

    widget.pipeline.set_param(threshold.id, "threshold", 20)
    widget._mark_pipeline_dirty(threshold.id)
    widget.run_pipeline(force_sync=True)

    assert widget.pipeline.node_execution_states[measurements.id] == EXECUTION_STALE
    assert widget.pipeline.node_execution_states[intensity.id] == EXECUTION_STALE
    assert widget._manual_node_ids_needing_calculation() == {
        measurements.id,
        intensity.id,
    }

    widget.calculate_all_button.click()

    assert widget.pipeline.node_execution_states[measurements.id] == EXECUTION_READY
    assert widget.pipeline.node_execution_states[intensity.id] == EXECUTION_READY
    assert widget.pipeline.outputs[measurements.id].row_count == 0
    assert widget.pipeline.outputs[intensity.id].row_count == 0


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


def test_collection_batch_dialog_defaults(qtbot):
    dialog = CollectionBatchDialog()
    qtbot.addWidget(dialog)

    values = dialog.values()

    assert "*.ome.tif" in values["pattern"]
    assert values["source_bindings"][0]["node_id"] == "input"
    assert "*.ome.tif" in values["source_bindings"][0]["pattern"]
    assert values["image_format"] == "ome-tiff"
    assert values["existing_file_policy"] == ExistingFilePolicy.ERROR.value
    assert values["save_workflow_snapshot"] is True
    assert not dialog.workflow_checkbox.isEnabled()
    assert values["save_python_script"] is True
    assert values["continue_on_error"] is True
    assert dialog.load_config_button.text() == "Load config..."
    assert dialog.load_config_button.isEnabled()
    assert dialog.save_config_button.text() == "Save config..."
    assert dialog.save_config_button.isEnabled()
    assert dialog.demo_config_button.text() == "Open batch demo..."
    assert dialog.demo_config_button.isEnabled()


def test_collection_batch_demo_auto_loads_first_pair_without_rebinding(
    qtbot,
    tmp_path,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)

    demo = widget._create_collection_batch_demo(tmp_path / "demo")
    primary_path = demo.root / "inputs" / "primary" / "01_shifted.npy"
    reference_path = demo.root / "inputs" / "reference" / "alpha_reference.npy"
    primary = np.load(primary_path)
    reference = np.load(reference_path)

    assert np.array_equal(widget.pipeline.outputs["input"], primary)
    assert np.array_equal(widget.pipeline.outputs["input_2"], reference)
    assert np.array_equal(
        widget.pipeline.outputs["batch_output_1"],
        primary.astype(np.float32) + reference.astype(np.float32),
    )
    labels = widget.pipeline.outputs["batch_output_2"]
    measurements = widget.pipeline.outputs["batch_output_3"]
    assert np.count_nonzero(labels) == 4
    assert set(np.unique(labels)) == {0, 1}
    assert measurements.rows[0][measurements.columns.index("area_pixels")] == 4
    for node_id in ("input", "input_2"):
        node = widget.pipeline.nodes[node_id]
        card = widget.graph_view._cards[node_id]
        assert node.params["file_path"] == ""
        assert node.params["binding_mode"] == "collection"
        assert card.metadata_label.text() != "No output"
        assert card.preview.pixmap() is not None
        assert not card.preview.pixmap().isNull()

    config = load_batch_config(demo.config_path)
    workflow = widget._batch_workflow_document()
    assert scientific_workflow_hash(workflow) == config.workflow_sha256
    plan = plan_batch(workflow, config, workflow_path=demo.workflow_path)
    assert len(plan.items) == 3
    assert plan.output_count == 9
    assert [
        tuple(path.name for path in item.source_paths.values())
        for item in plan.items
    ] == [
        ("01_shifted.npy", "alpha_reference.npy"),
        ("02_two_objects.npy", "beta_reference.npy"),
        ("03_disjoint.npy", "gamma_reference.npy"),
    ]

    widget._invalidate_pipeline_cache()
    widget.run_pipeline(
        force_sync=True,
        manual_node_ids=widget.pipeline.manual_node_ids(),
    )

    assert np.array_equal(widget.pipeline.outputs["input"], primary)
    assert np.array_equal(widget.pipeline.outputs["input_2"], reference)
    assert scientific_workflow_hash(widget._batch_workflow_document()) == (
        config.workflow_sha256
    )


def test_collection_batch_demo_button_creates_loads_and_previews_bundle(
    qtbot,
    monkeypatch,
    tmp_path,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    dialog = CollectionBatchDialog(widget, source_nodes=widget._batch_source_rows())
    qtbot.addWidget(dialog)
    monkeypatch.setattr(
        "napari_vipp._widget.QFileDialog.getExistingDirectory",
        lambda *_args, **_kwargs: str(tmp_path),
    )
    monkeypatch.setattr(
        "napari_vipp._widget.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.Yes,
    )

    qtbot.mouseClick(dialog.demo_config_button, Qt.LeftButton)

    demo_root = tmp_path / SYNTHETIC_BATCH_DEMO_DIRNAME
    assert (demo_root / BATCH_WORKFLOW_FILENAME).is_file()
    assert (demo_root / BATCH_CONFIG_FILENAME).is_file()
    assert (demo_root / BATCH_SCRIPT_FILENAME).is_file()
    assert "batch_output_3" in widget.pipeline.nodes
    assert [str(row["node_id"]) for row in dialog._source_rows] == [
        "input",
        "input_2",
    ]
    assert [str(row["title"]) for row in dialog._source_rows] == [
        "Primary signal",
        "Secondary reference",
    ]
    assert dialog._source_rows[0]["folder"].text() == str(
        demo_root / "inputs" / "primary"
    )
    assert dialog._source_rows[1]["folder"].text() == str(
        demo_root / "inputs" / "reference"
    )
    assert dialog.output_edit.text() == str(demo_root / "results")
    assert dialog.preview_table.rowCount() == 3
    assert dialog.preview_table.item(0, 2).text().count("\n") == 2
    assert dialog.preview_table.item(0, 3).text().splitlines() == [
        "new",
        "new",
        "new",
    ]
    assert not dialog.demo_guide_label.isHidden()
    assert "Ready-to-run batch demo" in dialog.demo_guide_label.text()
    assert str(demo_root) in dialog.demo_guide_label.text()
    assert dialog.run_button.text() == "Run demo batch"
    assert "Demo ready" in dialog.preview_status.text()
    assert "3 paired items" in dialog.preview_status.text()
    assert "9 outputs" in dialog.preview_status.text()

    result = widget._run_collection_batch(**dialog.values())
    validation = validate_synthetic_batch_demo(
        SyntheticBatchDemo.from_root(demo_root),
        result=result,
    )

    assert validation.ok
    assert result.summary == {
        "completed": 3,
        "partial": 0,
        "skipped": 0,
        "failed": 0,
    }
    assert {path.name for path in result.artifact_paths} == {
        BATCH_WORKFLOW_FILENAME,
        BATCH_SCRIPT_FILENAME,
    }
    validation_text = widget._validate_collection_batch_demo_result(
        demo_root / BATCH_CONFIG_FILENAME,
        result,
    )
    assert validation_text == "Synthetic ground truth passed (5 checks)."


def test_batch_provenance_example_is_registered_as_generated_demo():
    spec = next(item for item in EXAMPLE_WORKFLOWS if item.id == "batch-provenance")

    assert spec.generated_batch_demo
    assert spec.filename == "synthetic-batch-provenance.json"
    assert _example_workflow_path(spec).is_file()


def test_generated_batch_example_uses_demo_creation_branch(
    qtbot,
    monkeypatch,
    tmp_path,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    spec = next(item for item in EXAMPLE_WORKFLOWS if item.id == "batch-provenance")
    demo = SyntheticBatchDemo.from_root(tmp_path / "demo")

    class AcceptedExampleDialog:
        def __init__(self, _parent):
            pass

        def exec(self):
            return 1

        def selected_example(self):
            return spec

    opened = []
    monkeypatch.setattr(
        "napari_vipp._widget.ExampleWorkflowDialog",
        AcceptedExampleDialog,
    )
    monkeypatch.setattr(
        widget,
        "load_example_workflow",
        lambda *_args: pytest.fail("Generated demo used ordinary example loading"),
    )
    monkeypatch.setattr(
        widget,
        "_choose_collection_batch_demo",
        lambda: demo,
    )
    monkeypatch.setattr(
        widget,
        "_batch_collection_dialog",
        lambda **kwargs: opened.append(kwargs),
    )

    widget._open_example_workflow_dialog()

    assert opened == [{"config_path": demo.config_path}]

    monkeypatch.setattr(widget, "_choose_collection_batch_demo", lambda: None)
    widget._open_example_workflow_dialog()
    assert opened == [{"config_path": demo.config_path}]


def test_open_batch_example_builds_a_ready_to_run_workspace(
    qtbot,
    monkeypatch,
    tmp_path,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    spec = next(item for item in EXAMPLE_WORKFLOWS if item.id == "batch-provenance")

    class AcceptedExampleDialog:
        def __init__(self, _parent):
            pass

        def exec(self):
            return 1

        def selected_example(self):
            return spec

    opened_dialogs = []
    monkeypatch.setattr(
        "napari_vipp._widget.ExampleWorkflowDialog",
        AcceptedExampleDialog,
    )
    monkeypatch.setattr(
        "napari_vipp._widget.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.Yes,
    )
    monkeypatch.setattr(
        "napari_vipp._widget.QFileDialog.getExistingDirectory",
        lambda *_args, **_kwargs: str(tmp_path),
    )

    def inspect_then_cancel(dialog):
        opened_dialogs.append(dialog)
        return 0

    monkeypatch.setattr(CollectionBatchDialog, "exec", inspect_then_cancel)

    widget._open_example_workflow_dialog()

    demo_root = tmp_path / SYNTHETIC_BATCH_DEMO_DIRNAME
    assert (demo_root / BATCH_CONFIG_FILENAME).is_file()
    assert "batch_output_3" in widget.pipeline.nodes
    assert len(opened_dialogs) == 1
    dialog = opened_dialogs[0]
    assert dialog._demo == SyntheticBatchDemo.from_root(demo_root)
    assert dialog.preview_table.rowCount() == 3
    assert dialog.preview_table.item(0, 3).text().splitlines() == [
        "new",
        "new",
        "new",
    ]
    assert dialog.run_button.text() == "Run demo batch"
    assert "Demo ready" in dialog.preview_status.text()


def test_collection_batch_demo_confirmation_uses_active_dialog_parent(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    dialog = CollectionBatchDialog(widget, source_nodes=widget._batch_source_rows())
    qtbot.addWidget(dialog)
    captured = []

    def decline(owner, *_args, **_kwargs):
        captured.append(owner)
        return QMessageBox.No

    monkeypatch.setattr(
        "napari_vipp._widget.QMessageBox.question",
        decline,
    )
    monkeypatch.setattr(
        "napari_vipp._widget.QFileDialog.getExistingDirectory",
        lambda *_args, **_kwargs: pytest.fail(
            "Folder chooser opened after replacement was declined"
        ),
    )
    original_nodes = tuple(widget.pipeline.nodes)

    result = widget._choose_collection_batch_demo(dialog_parent=dialog)

    assert result is None
    assert captured == [dialog]
    assert tuple(widget.pipeline.nodes) == original_nodes


def test_run_collection_batch_writes_terminal_outputs(qtbot, tmp_path):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    first = np.zeros((3, 8, 9), dtype=np.uint8)
    second = np.zeros((3, 8, 9), dtype=np.uint8)
    first[:, 2:6, 3:7] = 200
    second[:, 1:4, 1:5] = 220
    tifffile.imwrite(input_dir / "field_a.ome.tif", first, photometric="minisblack")
    tifffile.imwrite(input_dir / "field_b.tif", second, photometric="minisblack")
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    saved = widget._run_collection_batch(
        input_dir,
        output_dir,
        "*.ome.tif;*.tif",
        image_format="npy",
        save_workflow_snapshot=True,
        save_python_script=True,
    )

    saved_names = {path.name for path in saved}
    assert "vipp_batch_workflow.json" in saved_names
    assert "vipp_batch_pipeline.py" in saved_names
    assert "field_a__Otsu_Threshold-threshold.npy" in saved_names
    assert "field_b__Otsu_Threshold-threshold.npy" in saved_names
    assert np.load(output_dir / "field_a__Otsu_Threshold-threshold.npy").dtype == bool
    assert "Batch 2/2" in widget.status_label.text()


def test_run_collection_batch_prefers_explicit_batch_outputs(qtbot, tmp_path):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    image = np.zeros((3, 8, 9), dtype=np.uint8)
    image[:, 2:6, 3:7] = 200
    tifffile.imwrite(input_dir / "field_a.tif", image, photometric="minisblack")
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    batch_output = widget.add_node_from_palette("batch_output")
    widget._connect_nodes("gaussian", batch_output.id)
    widget.pipeline.set_param(batch_output.id, "tag", "blurred")
    widget.pipeline.set_param(batch_output.id, "format", "npy")
    widget.pipeline.set_param(batch_output.id, "subfolder", "images/blurred")
    widget.pipeline.set_param(
        batch_output.id,
        "filename_template",
        "{source_stem}_{tag}",
    )

    saved = widget._run_collection_batch(
        input_dir,
        output_dir,
        "*.tif",
        image_format="ome-tiff",
        save_workflow_snapshot=True,
        save_python_script=False,
    )

    saved_names = {path.name for path in saved.saved_paths}
    assert saved_names == {"field_a_blurred.npy"}
    explicit_path = output_dir / "images" / "blurred" / "field_a_blurred.npy"
    terminal_path = output_dir / "field_a__Otsu_Threshold-threshold.ome.tif"
    assert explicit_path.exists()
    assert not terminal_path.exists()
    saved_array = np.load(explicit_path)
    assert saved_array.shape == image.shape
    assert saved_array.dtype == image.dtype


def test_run_collection_batch_uses_low_memory_retention(
    qtbot,
    monkeypatch,
    tmp_path,
):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    image = np.zeros((3, 8, 9), dtype=np.uint8)
    image[:, 2:6, 3:7] = 200
    tifffile.imwrite(input_dir / "field_a.tif", image, photometric="minisblack")
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    batch_output = widget.add_node_from_palette("batch_output")
    widget._connect_nodes("gaussian", batch_output.id)
    widget.pipeline.set_param(batch_output.id, "format", "npy")

    calls = []
    original_run = PrototypePipeline.run

    def captured_run(pipeline, *args, **kwargs):
        calls.append(
            {
                "prune_unretained": kwargs.get("prune_unretained"),
                "retain_node_ids": tuple(kwargs.get("retain_node_ids") or ()),
            }
        )
        return original_run(pipeline, *args, **kwargs)

    monkeypatch.setattr(PrototypePipeline, "run", captured_run)

    saved = widget._run_collection_batch(
        input_dir,
        output_dir,
        "*.tif",
        image_format="ome-tiff",
        save_workflow_snapshot=True,
        save_python_script=False,
    )

    batch_calls = [call for call in calls if call["prune_unretained"]]
    assert batch_calls
    assert {path.name for path in saved.saved_paths} == {"field_a__output.npy"}
    assert all(call["retain_node_ids"] == (batch_output.id,) for call in batch_calls)


def test_run_collection_batch_supports_independent_source_bindings(qtbot, tmp_path):
    primary_dir = tmp_path / "primary"
    mask_dir = tmp_path / "masks"
    output_dir = tmp_path / "outputs"
    primary_dir.mkdir()
    mask_dir.mkdir()
    first = np.zeros((3, 8, 9), dtype=np.uint8)
    second = np.zeros((3, 8, 9), dtype=np.uint8)
    first[:, 2:6, 3:7] = 120
    second[:, 1:4, 1:5] = 220
    first_mask = np.zeros((3, 8, 9), dtype=np.uint8)
    second_mask = np.zeros((3, 8, 9), dtype=np.uint8)
    first_mask[:, 0:2, 0:3] = 33
    second_mask[:, 5:7, 6:9] = 77
    tifffile.imwrite(primary_dir / "field_a.tif", first, photometric="minisblack")
    tifffile.imwrite(primary_dir / "field_b.tif", second, photometric="minisblack")
    tifffile.imwrite(mask_dir / "mask_a.tif", first_mask, photometric="minisblack")
    tifffile.imwrite(mask_dir / "mask_b.tif", second_mask, photometric="minisblack")

    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    secondary = widget.add_node_from_palette("input")
    output = widget.add_node_from_palette("batch_output")
    widget._connect_nodes(secondary.id, output.id)
    widget.pipeline.set_param(output.id, "tag", "secondary")
    widget.pipeline.set_param(output.id, "format", "npy")
    widget.pipeline.set_param(
        output.id,
        "filename_template",
        "{batch_index}_{source_stem}_{tag}",
    )
    source_bindings = [
        {
            "node_id": "input",
            "input_dir": str(primary_dir),
            "pattern": "*.tif",
        },
        {
            "node_id": secondary.id,
            "input_dir": str(mask_dir),
            "pattern": "*.tif",
        },
    ]

    preview = widget._preview_collection_batch(
        "",
        output_dir,
        image_format="ome-tiff",
        source_bindings=source_bindings,
    )

    assert [row.batch_id for row in preview] == ["0001_field_a", "0002_field_b"]
    assert preview[0].sources["input"].name == "field_a.tif"
    assert preview[0].sources[secondary.id].name == "mask_a.tif"
    assert preview[0].outputs[0].name == "0001_field_a_secondary.npy"

    saved = widget._run_collection_batch(
        "",
        output_dir,
        image_format="ome-tiff",
        save_workflow_snapshot=True,
        save_python_script=False,
        source_bindings=source_bindings,
    )

    assert {path.name for path in saved.saved_paths} == {
        "0001_field_a_secondary.npy",
        "0002_field_b_secondary.npy",
    }
    np.testing.assert_array_equal(
        np.load(output_dir / "0001_field_a_secondary.npy"),
        first_mask,
    )
    np.testing.assert_array_equal(
        np.load(output_dir / "0002_field_b_secondary.npy"),
        second_mask,
    )


def test_collection_batch_rejects_mismatched_source_binding_counts(qtbot, tmp_path):
    primary_dir = tmp_path / "primary"
    mask_dir = tmp_path / "masks"
    output_dir = tmp_path / "outputs"
    primary_dir.mkdir()
    mask_dir.mkdir()
    image = np.zeros((3, 8, 9), dtype=np.uint8)
    tifffile.imwrite(primary_dir / "field_a.tif", image, photometric="minisblack")
    tifffile.imwrite(primary_dir / "field_b.tif", image, photometric="minisblack")
    tifffile.imwrite(mask_dir / "mask_a.tif", image, photometric="minisblack")

    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    secondary = widget.add_node_from_palette("input")

    try:
        widget._preview_collection_batch(
            "",
            output_dir,
            source_bindings=[
                {
                    "node_id": "input",
                    "input_dir": str(primary_dir),
                    "pattern": "*.tif",
                },
                {
                    "node_id": secondary.id,
                    "input_dir": str(mask_dir),
                    "pattern": "*.tif",
                },
            ],
        )
    except ValueError as exc:
        assert "same number" in str(exc)
    else:
        raise AssertionError("Expected mismatched batch source counts to fail.")


def test_collection_batch_config_roundtrip_maps_sources_by_node_id(
    qtbot,
    tmp_path,
):
    primary_dir = tmp_path / "primary"
    secondary_dir = tmp_path / "secondary"
    output_dir = tmp_path / "outputs"
    primary_dir.mkdir()
    secondary_dir.mkdir()
    (primary_dir / "primary-a.tif").write_bytes(b"primary")
    (secondary_dir / "secondary-a.tif").write_bytes(b"secondary")
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    secondary = widget.add_node_from_palette("input")
    config_path = tmp_path / "saved_batch.json"
    source_bindings = [
        {
            "node_id": "input",
            "title": "Primary",
            "input_dir": str(primary_dir),
            "pattern": "primary-*.tif",
        },
        {
            "node_id": secondary.id,
            "title": "Secondary",
            "input_dir": str(secondary_dir),
            "pattern": "secondary-*.tif",
        },
    ]

    saved_config, saved_workflow = widget._save_collection_batch_config(
        config_path,
        input_dir=primary_dir,
        output_dir=output_dir,
        pattern="*.tif",
        image_format="npy",
        existing_file_policy=ExistingFilePolicy.SKIP.value,
        save_workflow_snapshot=True,
        save_python_script=False,
        source_bindings=source_bindings,
        continue_on_error=True,
    )
    loaded = widget._load_collection_batch_config(saved_config)

    assert saved_workflow.name == BATCH_WORKFLOW_FILENAME
    assert [source.node_id for source in loaded.sources] == ["input", secondary.id]
    dialog = CollectionBatchDialog(
        widget,
        source_nodes=list(reversed(widget._batch_source_rows())),
    )
    qtbot.addWidget(dialog)
    dialog._apply_config(loaded)
    rows = {str(row["node_id"]): row for row in dialog._source_rows}
    assert [str(row["node_id"]) for row in dialog._source_rows[:2]] == [
        "input",
        secondary.id,
    ]
    assert rows["input"]["folder"].text() == str(primary_dir)
    assert rows["input"]["pattern"].text() == "primary-*.tif"
    assert rows[secondary.id]["folder"].text() == str(secondary_dir)
    assert rows[secondary.id]["pattern"].text() == "secondary-*.tif"
    assert [
        binding["title"] for binding in dialog.values()["source_bindings"][:2]
    ] == ["Primary", "Secondary"]
    assert dialog.values()["existing_file_policy"] == "skip"
    preview = widget._preview_collection_batch(**dialog.values())
    assert preview[0].batch_id == "0001_primary-a"

    widget.pipeline.set_param("gaussian", "sigma", 3.5)
    with pytest.raises(ValueError, match="different workflow"):
        widget._load_collection_batch_config(saved_config)


def test_collection_batch_preview_reports_new_collision_and_terminal_fallback(
    qtbot,
    tmp_path,
):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    np.save(input_dir / "field_a.npy", np.arange(20, dtype=np.uint8).reshape(4, 5))
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)

    preview = widget._preview_collection_batch(
        input_dir,
        output_dir,
        "*.npy",
        image_format="npy",
    )

    assert preview[0].explicit_outputs is False
    assert set(preview[0].output_statuses) == {"new"}
    existing_path = preview[0].outputs[0]
    existing_path.parent.mkdir(parents=True)
    existing_path.write_bytes(b"already here")

    dialog = CollectionBatchDialog(widget, source_nodes=widget._batch_source_rows())
    qtbot.addWidget(dialog)
    dialog.input_edit.setText(str(input_dir))
    dialog.pattern_edit.setText("*.npy")
    dialog.output_edit.setText(str(output_dir))
    dialog.format_combo.setCurrentText("npy")
    dialog._preview_batch()

    assert dialog.preview_table.rowCount() == 1
    assert "exists; collision" in dialog.preview_table.item(0, 3).text()
    assert "collision" in dialog.preview_status.text().lower()
    assert "Compatibility fallback" in dialog.preview_status.text()


def test_collection_batch_config_clears_omitted_fixed_source_binding(
    qtbot,
    tmp_path,
):
    primary_dir = tmp_path / "primary"
    primary_dir.mkdir()
    fixed_path = tmp_path / "fixed.npy"
    np.save(fixed_path, np.ones((2, 3), dtype=np.uint8))
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    fixed = widget.add_node_from_palette("input")
    widget.pipeline.set_param(fixed.id, "source_mode", "file path")
    widget.pipeline.set_param(fixed.id, "file_path", str(fixed_path))
    workflow = widget._batch_workflow_document()
    config = widget._collection_batch_config(
        input_dir=primary_dir,
        output_dir=tmp_path / "outputs",
        source_bindings=[
            {
                "node_id": "input",
                "input_dir": str(primary_dir),
                "pattern": "*.npy",
            }
        ],
        workflow=workflow,
    )
    dialog = CollectionBatchDialog(widget, source_nodes=widget._batch_source_rows())
    qtbot.addWidget(dialog)
    rows = {str(row["node_id"]): row for row in dialog._source_rows}
    rows[fixed.id]["folder"].setText(str(tmp_path / "stale"))

    dialog._apply_config(config)

    assert rows[fixed.id]["folder"].text() == ""


def test_collection_batch_config_rejects_reserved_companion_filename(
    qtbot,
    tmp_path,
):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)

    with pytest.raises(ValueError, match="reserved"):
        widget._save_collection_batch_config(
            tmp_path / BATCH_WORKFLOW_FILENAME,
            input_dir=input_dir,
            output_dir=tmp_path / "outputs",
            save_workflow_snapshot=True,
            save_python_script=False,
        )


def test_collection_batch_preview_resolves_table_default_to_csv(qtbot, tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    np.save(input_dir / "field_a.npy", np.arange(20, dtype=np.uint8).reshape(4, 5))
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    labels = widget.add_node_from_palette("label_connected_components")
    widget._connect_nodes("threshold", labels.id)
    measurements = widget.add_node_from_palette("measure_objects")
    widget._connect_nodes(labels.id, measurements.id)
    output = widget.add_node_from_palette("batch_output")
    widget._connect_nodes(measurements.id, output.id)
    widget.pipeline.set_param(output.id, "tag", "measurements")

    preview = widget._preview_collection_batch(
        input_dir,
        tmp_path / "outputs",
        "*.npy",
        image_format="npy",
    )

    assert preview[0].outputs[0].name == "field_a__measurements.csv"
    assert preview[0].output_statuses == ("new",)


def test_collection_batch_preview_reports_collisions_beyond_display_limit(
    qtbot,
    tmp_path,
):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    for index in range(26):
        np.save(
            input_dir / f"field_{index:02d}.npy",
            np.full((2, 3), index, dtype=np.uint8),
        )
    output_dir.mkdir()
    np.save(
        output_dir / "field_25__output.npy",
        np.zeros((2, 3), dtype=np.uint8),
    )
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    output = widget.add_node_from_palette("batch_output")
    widget._connect_nodes("input", output.id)
    widget.pipeline.set_param(output.id, "format", "npy")

    preview = widget._preview_collection_batch(
        input_dir,
        output_dir,
        "*.npy",
        image_format="npy",
        preview_limit=25,
    )

    assert len(preview) == 25
    assert preview.total_items == 26
    assert preview.collision_count == 1


def test_run_collection_batch_writes_reproducibility_artifacts_and_manifest(
    qtbot,
    tmp_path,
):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    source = np.arange(20, dtype=np.uint16).reshape(4, 5)
    np.save(input_dir / "field_a.npy", source)
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    batch_output = widget.add_node_from_palette("batch_output")
    widget._connect_nodes("input", batch_output.id)
    widget.pipeline.set_param(batch_output.id, "tag", "image")
    widget.pipeline.set_param(batch_output.id, "format", "npy")

    result = widget._run_collection_batch(
        input_dir,
        output_dir,
        "*.npy",
        image_format="npy",
        save_workflow_snapshot=True,
        save_python_script=True,
    )

    config_path = output_dir / BATCH_CONFIG_FILENAME
    manifest_path = output_dir / BATCH_MANIFEST_FILENAME
    runner_path = output_dir / BATCH_SCRIPT_FILENAME
    workflow_path = output_dir / BATCH_WORKFLOW_FILENAME
    assert config_path.is_file()
    assert manifest_path.is_file()
    assert runner_path.is_file()
    assert workflow_path.is_file()
    assert result.manifest_archive_path is not None
    assert result.manifest_archive_path.is_file()
    assert {path.name for path in result.artifact_paths} == {
        BATCH_SCRIPT_FILENAME,
        BATCH_WORKFLOW_FILENAME,
    }
    runner = runner_path.read_text(encoding="utf-8")
    assert "from napari_vipp.core.batch import run_batch_from_files" in runner
    assert "PrototypePipeline" not in runner

    config_document = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert config_document["type"] == BATCH_CONFIG_TYPE
    assert manifest["type"] == BATCH_MANIFEST_TYPE
    assert manifest["workflow"]["sha256"] == config_document["workflow"]["sha256"]
    assert manifest["summary"] == {
        "completed": 1,
        "partial": 0,
        "skipped": 0,
        "failed": 0,
    }
    item = manifest["items"][0]
    assert item["status"] == BatchStatus.COMPLETED.value
    assert item["sources"][0]["node_id"] == "input"
    assert item["sources"][0]["identity"]["size_bytes"] > source.nbytes
    assert item["sources"][0]["series"]["shape"] == [4, 5]
    assert item["outputs"][0]["node_id"] == batch_output.id
    assert item["outputs"][0]["status"] == BatchStatus.COMPLETED.value
    assert item["outputs"][0]["size_bytes"] > 0
    assert manifest["finished_at"]
    assert manifest["runtime"]["packages"]["napari-vipp"] == VIPP_VERSION


def test_run_collection_batch_continues_after_middle_read_failure(qtbot, tmp_path):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    first = np.full((3, 4), 11, dtype=np.uint8)
    last = np.full((3, 4), 33, dtype=np.uint8)
    np.save(input_dir / "01_first.npy", first)
    (input_dir / "02_broken.npy").write_bytes(b"not a NumPy file")
    np.save(input_dir / "03_last.npy", last)
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    batch_output = widget.add_node_from_palette("batch_output")
    widget._connect_nodes("input", batch_output.id)
    widget.pipeline.set_param(batch_output.id, "tag", "result")
    widget.pipeline.set_param(batch_output.id, "format", "npy")

    result = widget._run_collection_batch(
        input_dir,
        output_dir,
        "*.npy",
        image_format="npy",
        save_workflow_snapshot=True,
        save_python_script=False,
        continue_on_error=True,
    )

    assert result.summary == {
        "completed": 2,
        "partial": 0,
        "skipped": 0,
        "failed": 1,
    }
    assert [item.status for item in result.manifest.items] == [
        BatchStatus.COMPLETED,
        BatchStatus.FAILED,
        BatchStatus.COMPLETED,
    ]
    assert {path.name for path in result.saved_paths} == {
        "01_first__result.npy",
        "03_last__result.npy",
    }
    np.testing.assert_array_equal(np.load(output_dir / "01_first__result.npy"), first)
    np.testing.assert_array_equal(np.load(output_dir / "03_last__result.npy"), last)
    assert "Batch 3/3" in widget.status_label.text()


def test_collection_batch_collision_does_not_replace_prior_artifacts(
    qtbot,
    tmp_path,
):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    np.save(input_dir / "field.npy", np.ones((2, 3), dtype=np.uint8))
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    output = widget.add_node_from_palette("batch_output")
    widget._connect_nodes("input", output.id)
    widget.pipeline.set_param(output.id, "format", "npy")
    widget._run_collection_batch(
        input_dir,
        output_dir,
        "*.npy",
        image_format="npy",
    )
    artifact_paths = (
        output_dir / BATCH_CONFIG_FILENAME,
        output_dir / BATCH_WORKFLOW_FILENAME,
        output_dir / BATCH_SCRIPT_FILENAME,
        output_dir / BATCH_MANIFEST_FILENAME,
    )
    before = {path: path.read_bytes() for path in artifact_paths}

    with pytest.raises(FileExistsError, match="preflight found output collisions"):
        widget._run_collection_batch(
            input_dir,
            output_dir,
            "*.npy",
            image_format="npy",
        )

    assert {path: path.read_bytes() for path in artifact_paths} == before


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


def test_optional_reader_error_uses_reader_dialog(qtbot, monkeypatch):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    error = OptionalMicroscopeReaderError(
        "CZI support is missing",
        suffix=".czi",
        format_name="zeiss-czi",
        module_name="bioio",
        install_command='pip install "napari-vipp[czi]"',
    )
    calls = []

    def raise_error():
        raise error

    monkeypatch.setattr(widget, "_source_payloads_for_pipeline", raise_error)
    monkeypatch.setattr(
        widget,
        "_show_optional_reader_error",
        lambda exc: calls.append(exc),
    )

    widget.run_pipeline()

    assert calls == [error]


def test_optional_reader_dialog_copies_install_command(qtbot, monkeypatch):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    command = 'pip install "napari-vipp[czi]"'
    error = OptionalMicroscopeReaderError(
        "CZI support is missing",
        suffix=".czi",
        format_name="zeiss-czi",
        module_name="bioio",
        install_command=command,
        fallback_install_command='pip install "napari-vipp[bioformats]"',
    )

    class FakeMessageBox:
        Warning = QMessageBox.Warning
        ActionRole = QMessageBox.ActionRole
        Close = QMessageBox.Close
        instances = []

        def __init__(self, parent=None):
            self.parent = parent
            self.icon = None
            self.window_title = ""
            self.text = ""
            self.informative_text = ""
            self.detailed_text = ""
            self.copy_button = object()
            self.clicked = None
            self.buttons = []
            self.instances.append(self)

        def setIcon(self, icon):
            self.icon = icon

        def setWindowTitle(self, title):
            self.window_title = title

        def setText(self, text):
            self.text = text

        def setInformativeText(self, text):
            self.informative_text = text

        def setDetailedText(self, text):
            self.detailed_text = text

        def addButton(self, button, role=None):
            if button == "Copy install command":
                self.buttons.append((button, role, self.copy_button))
                return self.copy_button
            close_button = object()
            self.buttons.append((button, role, close_button))
            return close_button

        def exec(self):
            self.clicked = self.copy_button
            return 0

        def clickedButton(self):
            return self.clicked

    class FakeClipboard:
        def __init__(self):
            self.value = ""

        def setText(self, value):
            self.value = str(value)

        def text(self):
            return self.value

    clipboard = FakeClipboard()
    monkeypatch.setattr("napari_vipp._widget.QMessageBox", FakeMessageBox)
    monkeypatch.setattr(
        "napari_vipp._widget.QApplication.clipboard",
        lambda: clipboard,
    )

    widget._show_optional_reader_error(error)

    box = FakeMessageBox.instances[0]
    assert box.window_title == "Optional Image Reader Missing"
    assert "CZI reader is not installed" in box.text
    assert command in box.informative_text
    assert "restart napari" in box.informative_text
    assert clipboard.text() == command
    assert widget.status_label.text() == f"Copied reader install command: {command}"


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


def test_rgb_volume_pin_uses_additive_channel_layers(qtbot):
    data = np.zeros((3, 12, 16, 18), dtype=np.uint16)
    data[0] = 1000
    data[1] = 2000
    data[2] = 3000
    viewer = _Viewer(data, metadata={"axes": "CZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("composite_to_rgb")
    widget._connect_nodes("input", node.id)
    widget.pin_node(node.id)

    base_name = "VIPP Pinned: Composite \u2192 RGB"
    _assert_rgb_channel_layers(viewer, base_name, (12, 16, 18))
    pinned_layers = [
        layer
        for layer in viewer.layers
        if layer.metadata.get("napari_vipp_kind") == "pinned"
    ]
    assert len(pinned_layers) == 3
    assert {layer.metadata["display_rgb_channel"] for layer in pinned_layers} == {
        "Red",
        "Green",
        "Blue",
    }
    assert widget._active_pinned_node_id == node.id

    widget.pin_node(node.id)

    pinned_layers = [
        layer
        for layer in viewer.layers
        if layer.metadata.get("napari_vipp_kind") == "pinned"
    ]
    assert pinned_layers == []
    assert widget._active_pinned_node_id is None


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
    assert second_inspect.data.dtype == bool
    assert np.shares_memory(
        second_inspect.data,
        widget.pipeline.outputs["threshold"],
    )


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


def test_signed_image_inspect_contrast_includes_negative_values(qtbot):
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
    assert layer.contrast_limits == (-1.0, 1.0)
    assert layer.metadata["vipp_display_contrast_basis"] == (
        "Exact full finite data range (display only)"
    )
    assert layer.metadata["vipp_display_contrast_adjustable"] is True


def test_generated_image_contrast_uses_exact_full_finite_range():
    positive = np.linspace(0, 200, 4 * 16 * 18, dtype=np.float32).reshape(4, 16, 18)
    non_finite = np.array([-np.inf, -2.0, np.nan, 5.0, np.inf], dtype=np.float32)

    assert _exact_generated_layer_contrast_limits(positive) == (0.0, 200.0)
    assert _exact_generated_layer_contrast_limits(non_finite) == (-2.0, 5.0)


def test_large_float_inspect_contrast_is_calculated_in_background(
    qtbot,
    monkeypatch,
):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    data = np.linspace(-7.0, 42.0, 200, dtype=np.float32).reshape(2, 10, 10)
    widget.pipeline.outputs["gaussian"] = data
    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_BYTES", 100)
    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_ELEMENTS", 100)
    started = threading.Event()
    release = threading.Event()

    def blocking_exact_limits(values):
        assert values is data
        started.set()
        assert release.wait(5)
        return (-7.0, 42.0)

    monkeypatch.setattr(
        "napari_vipp._widget._exact_generated_layer_contrast_limits",
        blocking_exact_limits,
    )

    before = time.perf_counter()
    widget.inspect_node("gaussian")
    elapsed = time.perf_counter() - before

    inspect = viewer.layers["VIPP Inspect"]
    assert elapsed < 0.25
    assert inspect.contrast_limits == (0.0, 1.0)
    assert inspect.metadata["vipp_display_contrast_pending"] is True
    qtbot.waitUntil(started.is_set, timeout=5_000)

    release.set()
    qtbot.waitUntil(
        lambda: not widget._generated_layer_contrast_pending,
        timeout=5_000,
    )

    assert inspect.contrast_limits == (-7.0, 42.0)
    assert inspect.metadata["vipp_display_contrast_pending"] is False
    assert inspect.metadata["vipp_display_contrast_basis"] == (
        "Exact full finite data range (display only)"
    )


def test_generated_layer_contrast_rejects_stale_worker_result(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    layer = viewer.layers[0]
    layer.name = "VIPP Inspect"
    generation = widget._generated_layer_contrast_generation
    old_key = (generation, "old")
    new_key = (generation, "new")
    layer.metadata["_vipp_display_contrast_key"] = new_key
    layer.contrast_limits = (-2.0, 3.0)
    widget._generated_layer_contrast_keys[layer.name] = new_key

    widget._on_generated_layer_contrast_finished(
        GeneratedLayerContrastResult(
            old_key,
            layer.name,
            limits=(-100.0, 100.0),
        )
    )

    assert layer.contrast_limits == (-2.0, 3.0)


def test_generated_layer_contrast_preserves_user_adjustment(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    layer = viewer.layers[0]
    layer.name = "VIPP Inspect"
    key = (widget._generated_layer_contrast_generation, "current")
    layer.metadata.update(
        {
            "_vipp_display_contrast_key": key,
            "_vipp_display_contrast_initial_limits": (0.0, 1.0),
        }
    )
    layer.contrast_limits = (0.2, 0.8)
    widget._generated_layer_contrast_keys[layer.name] = key

    widget._on_generated_layer_contrast_finished(
        GeneratedLayerContrastResult(
            key,
            layer.name,
            limits=(-7.0, 42.0),
        )
    )

    assert layer.contrast_limits == (0.2, 0.8)
    assert layer.metadata["vipp_exact_finite_data_range"] == (-7.0, 42.0)
    assert "User-adjusted" in layer.metadata["vipp_display_contrast_basis"]


def test_large_rgb_channels_receive_exact_background_contrast(qtbot, monkeypatch):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    data = np.zeros((2, 4, 5, 6, 3), dtype=np.float32)
    data[..., 0] = np.linspace(-2.0, 5.0, data[..., 0].size).reshape(data[..., 0].shape)
    data[..., 1] = np.linspace(0.25, 2.0, data[..., 1].size).reshape(data[..., 1].shape)
    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_BYTES", 100)
    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_ELEMENTS", 100)
    started = threading.Event()
    release = threading.Event()
    original = _exact_generated_layer_contrast_limits

    def blocking_exact_limits(values):
        started.set()
        assert release.wait(5)
        return original(values)

    monkeypatch.setattr(
        "napari_vipp._widget._exact_generated_layer_contrast_limits",
        blocking_exact_limits,
    )
    metadata = {
        "napari_vipp_kind": "inspect",
        "node_id": "manual-rgb",
        "data_kind": "image",
        "display_kind": "image",
        "display_rgb": True,
        "display_ndim": data.ndim,
        "display_shape": data.shape,
    }

    widget._set_or_add_rgb_channel_layers("Scientific RGB", data, metadata)

    qtbot.waitUntil(started.is_set, timeout=5_000)
    for layer in widget._rgb_channel_layers("Scientific RGB"):
        assert layer.contrast_limits == (0.0, 1.0)
        assert layer.metadata["vipp_display_contrast_pending"] is True

    release.set()
    qtbot.waitUntil(
        lambda: not widget._generated_layer_contrast_pending,
        timeout=5_000,
    )

    assert viewer.layers["Scientific RGB"].contrast_limits == (-2.0, 5.0)
    assert viewer.layers["Scientific RGB Green"].contrast_limits == (0.0, 2.0)
    assert viewer.layers["Scientific RGB Blue"].contrast_limits == (0.0, 1.0)
    assert all(
        layer.metadata["vipp_display_contrast_pending"] is False
        for layer in widget._rgb_channel_layers("Scientific RGB")
    )

    widget._set_or_add_rgb_channel_layers("Scientific RGB", data, metadata)

    assert not widget._generated_layer_contrast_pending


def test_rescaled_float_inspect_refreshes_reused_contrast_limits(qtbot):
    data = np.linspace(0.0, 187.0, 4 * 16 * 18, dtype=np.float32).reshape(4, 16, 18)
    viewer = _Viewer(data)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("rescale_intensity")
    widget._connect_nodes("input", node.id)
    widget.pipeline.set_param(node.id, "in_low_value", 0.0)
    widget.pipeline.set_param(node.id, "in_high_value", 187.0)
    widget.pipeline.set_param(node.id, "out_min", 0.0)
    widget.pipeline.set_param(node.id, "out_max", 1.0)
    widget.run_pipeline()

    widget.inspect_node("input")
    inspect = viewer.layers["VIPP Inspect"]
    inspect.contrast_limits = (0.0, 187.0)

    widget.inspect_node(node.id)

    assert viewer.layers["VIPP Inspect"] is inspect
    assert np.isclose(float(np.max(inspect.data)), 1.0)
    assert inspect.contrast_limits == (0.0, 1.0)


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
    assert input_inspect.contrast_limits == (0.0, 255.0)
    assert input_inspect.metadata["vipp_display_contrast_basis"] == (
        "Exact full finite data range (display only)"
    )
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


def test_auto_contrast_uses_rare_values_beyond_old_sampling_limit():
    values = np.zeros(1_000_003, dtype=np.float32)
    values[777_777] = -17.0
    values[999_999] = 1_000.0

    result = _auto_contrast_scale_offset(values, 0.0)

    assert result is not None
    _alpha, _beta, lower, upper = result
    assert lower == -17.0
    assert upper == 1_000.0


def test_auto_contrast_uses_all_values_in_large_periodic_input():
    values = np.empty(1_000_002, dtype=np.uint16)
    values[::2] = 0
    values[1::2] = 4_096

    result = _auto_contrast_scale_offset(values, 0.35)

    assert result is not None
    _alpha, _beta, lower, upper = result
    assert lower == 0.0
    assert upper == 4_096.0


def test_auto_contrast_rgb_uses_weighted_luminance_reference():
    rgb = np.array(
        [[[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]]],
        dtype=np.uint8,
    )

    result = _auto_contrast_scale_offset(rgb, 0.0)

    assert result is not None
    _alpha, _beta, lower, upper = result
    assert lower == 0.0
    np.testing.assert_allclose(upper, 58.7, rtol=0.0, atol=1e-5)


def test_auto_contrast_rgba_ignores_alpha_channel():
    rgb = np.array(
        [[[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]]],
        dtype=np.uint8,
    )
    alpha = np.array([[[255], [0], [17], [240]]], dtype=np.uint8)
    rgba = np.concatenate((rgb, alpha), axis=-1)

    rgb_result = _auto_contrast_scale_offset(rgb, 0.0)
    rgba_result = _auto_contrast_scale_offset(rgba, 0.0)

    assert rgb_result is not None
    assert rgba_result is not None
    np.testing.assert_allclose(rgba_result, rgb_result, rtol=0.0, atol=1e-12)


def test_large_auto_contrast_dispatches_exact_work_without_blocking(qtbot):
    data = np.empty(1_000_002, dtype=np.uint8)
    data[::2] = 0
    data[1::2] = 100
    viewer = _Viewer(data)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("linear_scale_offset")
    widget._connect_nodes("input", node.id)
    widget.auto_saturation_control.value_box.setValue(0.0)

    pool = _QueuedThreadPool()
    widget._pipeline_thread_pool = pool
    undo_count = len(widget._undo_stack)

    started = time.perf_counter()
    widget.auto_contrast_button.click()
    elapsed = time.perf_counter() - started

    assert elapsed < 0.5
    assert len(pool.workers) == 1
    assert widget._active_auto_contrast_run_id is not None
    assert not widget.auto_contrast_button.isEnabled()
    assert not widget.pipeline_busy_label.isHidden()
    assert "exact full-input" in widget.status_label.text()

    pool.workers[0].run()
    qtbot.waitUntil(lambda: widget._active_auto_contrast_run_id is None)

    params = widget.pipeline.nodes[node.id].params
    np.testing.assert_allclose(params["alpha"], 2.55, atol=0.0001)
    np.testing.assert_allclose(params["beta"], 0.0, atol=0.0001)
    assert len(widget._undo_stack) == undo_count + 1
    assert widget.auto_contrast_button.isEnabled()
    assert widget.pipeline_busy_label.isHidden()
    assert widget.pipeline.outputs[node.id].min() == 0
    assert widget.pipeline.outputs[node.id].max() == 255


def test_large_auto_contrast_ignores_stale_setting_result(qtbot):
    data = np.arange(1_000_002, dtype=np.float32)
    viewer = _Viewer(data)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("linear_scale_offset")
    widget._connect_nodes("input", node.id)

    pool = _QueuedThreadPool()
    widget._pipeline_thread_pool = pool
    undo_count = len(widget._undo_stack)

    widget.auto_contrast_button.click()
    widget.auto_saturation_control.value_box.setValue(5.0)
    pool.workers[0].run()
    qtbot.waitUntil(lambda: widget._active_auto_contrast_run_id is None)

    params = widget.pipeline.nodes[node.id].params
    assert params["alpha"] == 3.0
    assert params["beta"] == 1.0
    assert len(widget._undo_stack) == undo_count
    assert "stale result was ignored" in widget.status_label.text()
