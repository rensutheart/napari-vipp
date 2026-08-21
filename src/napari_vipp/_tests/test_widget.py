from __future__ import annotations

import json
import os
import threading
import time
import weakref
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import imageio.v3 as iio
import numpy as np
import pytest
import tifffile
from napari.components import ViewerModel
from qtpy.QtCore import QEvent, QPoint, QPointF, QSignalBlocker, Qt, QTimer
from qtpy.QtGui import QCloseEvent, QColor, QKeySequence, QMouseEvent
from qtpy.QtWidgets import (
    QApplication,
    QDockWidget,
    QFormLayout,
    QFrame,
    QGraphicsItem,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QWidget,
)

from napari_vipp import __version__ as VIPP_VERSION
from napari_vipp._graph import (
    BLOCKED_EXECUTION_ACCENT,
    STALE_EXECUTION_ACCENT,
    PortLabelMode,
    ThumbnailStatsBadgeKind,
)
from napari_vipp._theme import category_color, category_tint
from napari_vipp._widget import (
    CACHE_KEEP_NODE_PARAM,
    CACHE_MODE_KEEP_ALL,
    CACHE_MODE_LOW_MEMORY,
    CACHE_MODE_SMART,
    EXAMPLE_WORKFLOWS,
    INTENSITY_CONTRAST_HISTOGRAM_OPERATIONS,
    AutoContrastResult,
    CollectionBatchDialog,
    ColocalizationScatterRequest,
    ColocalizationScatterResult,
    ConnectionInsertCandidate,
    ConnectionInsertDialog,
    ConnectionInsertMappingDialog,
    ConnectionInsertPortMapping,
    ExampleWorkflowDialog,
    FlexibleDoubleSpinBox,
    GeneratedLayerContrastResult,
    GraphNoteState,
    HistogramPlot,
    InputHistogramResult,
    PipelineNodeResult,
    PipelineRunResult,
    SelectTableColumnsControl,
    SourceFileLoadResult,
    ThumbnailContrastLimitResult,
    VippWidget,
    _auto_contrast_scale_offset,
    _exact_finite_percentiles,
    _exact_generated_layer_contrast_limits,
    _example_workflow_path,
    _histogram_summary,
    _input_histogram_marker_key,
    _input_histogram_markers,
    _macos_memory_bytes,
    _prepare_colocalization_scatter_density,
    _rescale_dtype_output_range,
    _system_memory_bytes,
    _windows_memory_bytes,
)
from napari_vipp.core.batch import (
    BATCH_CONFIG_FILENAME,
    BATCH_CONFIG_TYPE,
    BATCH_MANIFEST_FILENAME,
    BATCH_MANIFEST_TYPE,
    BATCH_SCRIPT_FILENAME,
    BATCH_WORKFLOW_FILENAME,
    BatchAxisSuggestion,
    BatchConfig,
    BatchExecutionProgress,
    BatchOutputConfig,
    BatchScientificPreflightError,
    BatchSourceConfig,
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
from napari_vipp.core.compute import (
    ComputeEnvironment,
    ComputeMode,
    ComputeRepairAction,
    ComputeRepairCandidate,
    ComputeRepairSuggestion,
    ComputeRequest,
    DecisionKind,
    DecisionReason,
    ExactWorkloadCandidateQualification,
    ExecutionPlan,
    ExecutionReport,
    FallbackPolicy,
    FallbackReason,
    NodeComputePreference,
    NodeExecutionDecision,
    NodePreferenceKind,
    canonical_digest,
)
from napari_vipp.core.compute_pipeline_optimizer import (
    PipelineOptimizationIdentity,
    PipelineOptimizationProposal,
    PipelineOptimizationRow,
    PipelineOptimizationSelectionBasis,
    PipelineParityDeviation,
    PipelineParityReviewMetric,
    PipelineValidationWinner,
)
from napari_vipp.core.execution import (
    PipelineExecutionFailure,
    ResidentThumbnailStatisticsObservation,
)
from napari_vipp.core.export import export_pipeline_to_python
from napari_vipp.core.graph_fragments import (
    GraphFragment,
    GraphFragmentNode,
    GraphFragmentNote,
    capture_graph_fragment,
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
    AxisDeclaration,
    AxisMetadata,
    ChannelMetadata,
    image_state_from_array,
)
from napari_vipp.core.operations import (
    NO_TABLE_COLUMNS_VALUE,
    automatic_threshold_value,
)
from napari_vipp.core.pipeline import (
    EXECUTION_BLOCKED,
    EXECUTION_ERROR,
    EXECUTION_NOT_CALCULATED,
    EXECUTION_READY,
    EXECUTION_RUNNING,
    EXECUTION_STALE,
    NODE_LIBRARY_BY_ID,
    PALETTE_NODE_LIBRARY,
    GraphConnection,
    GraphNode,
    OutputTunnel,
    PrototypePipeline,
    SourcePayload,
)
from napari_vipp.core.preview import make_preview
from napari_vipp.core.progress import OperationCancelled, ProgressContext
from napari_vipp.core.source_identity import (
    BundledSampleRevisionToken,
    LocalSourceIdentity,
)
from napari_vipp.core.tables import TableData, TableState
from napari_vipp.core.thumbnail_statistics import (
    ThumbnailStatisticsBackend,
    ThumbnailStatisticsDecision,
    ThumbnailStatisticsResult,
)
from napari_vipp.core.workflow import (
    deserialize_workflow,
    save_workflow,
    serialize_workflow,
)
from napari_vipp.ui import recent_paths
from napari_vipp.ui.batch_workers import CollectionBatchOperationProgress
from napari_vipp.ui.compute_benchmark_dialog import NodeBenchmarkWorkerOutcome
from napari_vipp.ui.compute_pipeline_optimizer_dialog import (
    PipelineOptimizerApplyRequest,
    PipelineOptimizerWorkerOutcome,
)
from napari_vipp.ui.compute_setup import ComputeDeviceOption
from napari_vipp.ui.diagnostic_workers import ThumbnailContrastProgress
from napari_vipp.ui.file_sources import SourceFileLoadSpec
from napari_vipp.ui.presentation_settings import ThumbnailStatisticsPolicy


class _Event:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def disconnect(self, callback):
        self.callbacks.remove(callback)

    def emit(self):
        for callback in tuple(self.callbacks):
            callback()


class _QueuedThreadPool:
    def __init__(self):
        self.workers = []

    def start(self, worker, _priority=0):
        self.workers.append(worker)


def _thumbnail_statistics_result(
    data,
    *,
    limits=(0.0, 10.0),
    intended_backend=ThumbnailStatisticsBackend.CPU_NUMPY,
    actual_backend=ThumbnailStatisticsBackend.CPU_NUMPY,
    decision_reason_code="test-selection",
    decision_reason="Test backend selection.",
    fallback_reason_code="",
    fallback_message="",
    requested_compute_mode=ComputeMode.AUTO,
    input_path="",
    logical_input_host_to_device_bytes=0,
    auxiliary_host_to_device_bytes=0,
    device_to_host_bytes=0,
    device_to_host_values=0,
):
    arr = np.asarray(data)
    decision = ThumbnailStatisticsDecision(
        intended_backend,
        decision_reason_code,
        decision_reason,
        int(arr.size),
        int(arr.nbytes),
        32 * 1024**2,
        True,
    )
    return ThumbnailStatisticsResult(
        limits,
        decision,
        actual_backend,
        "exact-native-uint-histogram-test-v1",
        0.125,
        runtime_id=(
            "cuda-cupy"
            if intended_backend is ThumbnailStatisticsBackend.GPU_CUPY
            else ""
        ),
        device_id=(
            "test-gpu" if actual_backend is ThumbnailStatisticsBackend.GPU_CUPY else ""
        ),
        fallback_reason_code=fallback_reason_code,
        fallback_message=fallback_message,
        requested_compute_mode=requested_compute_mode,
        input_path=input_path,
        logical_input_host_to_device_bytes=logical_input_host_to_device_bytes,
        auxiliary_host_to_device_bytes=auxiliary_host_to_device_bytes,
        device_to_host_bytes=device_to_host_bytes,
        device_to_host_values=device_to_host_values,
    )


class _LayerEvents:
    def __init__(self):
        self.inserted = _Event()
        self.removed = _Event()


class _DimsEvents:
    def __init__(self):
        self.current_step = _Event()
        self.point = _Event()


class _SourceEvents:
    def __init__(self):
        for name in (
            "data",
            "set_data",
            "metadata",
            "name",
            "scale",
            "translate",
            "rotate",
            "shear",
            "affine",
            "units",
            "axis_labels",
            "labels_update",
        ):
            setattr(self, name, _Event())


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
        self.translate = None
        self.rotate = None
        self.shear = None
        self.affine = None
        self.units = None
        self.axis_labels = None
        self.editable = True
        self.events = _SourceEvents()


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


def _inspect_layer_node_id(viewer) -> str | None:
    try:
        layer = viewer.layers["VIPP Inspect"]
    except (KeyError, IndexError):
        return None
    metadata = getattr(layer, "metadata", {})
    return metadata.get("node_id") if isinstance(metadata, dict) else None


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


def test_flexible_double_spinbox_uses_scientific_text_below_one_thousandth(qtbot):
    box = FlexibleDoubleSpinBox()
    qtbot.addWidget(box)
    box.setRange(-1.0, 1.0)
    box.setDecimals(12)

    for value, expected in (
        (0.002, "0.002"),
        (0.001, "0.001"),
        (1e-4, "1e-4"),
        (1e-6, "1e-6"),
        (-1e-4, "-1e-4"),
        (0.0, "0"),
    ):
        box.setValue(value)
        assert box.text() == expected

    box.show()
    box.lineEdit().setFocus()
    box.selectAll()
    qtbot.keyClicks(box.lineEdit(), "1e-4")
    assert box.lineEdit().text() == "1e-4"
    qtbot.keyClick(box.lineEdit(), Qt.Key_Return)
    assert box.value() == pytest.approx(1e-4)


def test_numeric_parameter_context_menu_resets_to_default(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("richardson_lucy_tv_deconvolution")
    widget.graph_view.select_node(node.id)

    for name, changed_value in (("iterations", 40), ("denominator_floor", 0.2)):
        control = widget._parameter_widgets[name]
        control.value_box.setValue(changed_value)
        menu, reset_action = control.value_box._create_context_menu()

        assert reset_action.text() == "Reset to default"
        assert reset_action.isEnabled()
        assert len(menu.actions()) > 2

        reset_action.trigger()

        spec = next(
            spec
            for spec in widget.pipeline.node_parameter_specs(node.id)
            if spec.name == name
        )
        assert control.value() == pytest.approx(spec.default)
        assert widget.pipeline.nodes[node.id].params[name] == pytest.approx(
            spec.default
        )
        menu.deleteLater()


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


def test_histogram_plot_clicking_marker_does_not_change_it(qtbot):
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
    marker = QPoint(
        rect.left() + int(round(plot._x_fraction(50.0) * rect.width())),
        rect.center().y(),
    )
    qtbot.mouseClick(plot, Qt.LeftButton, pos=marker)

    assert captured == []
    assert plot.marker_values()["threshold"] == 50.0


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


def test_widget_can_defer_initial_pipeline_and_run_it_exactly_once(
    qtbot,
    monkeypatch,
):
    runs = []
    monkeypatch.setattr(
        VippWidget,
        "run_pipeline",
        lambda self, *args, **kwargs: runs.append((args, kwargs)),
    )

    widget = VippWidget(
        _Viewer(),
        defer_initial_run=True,
        initial_compute_mode="prefer_gpu",
    )
    qtbot.addWidget(widget)

    assert runs == []
    assert widget.compute_mode_combo.currentData() == "prefer_gpu"
    assert widget.run_initial_pipeline_once()
    assert len(runs) == 1
    assert not widget.run_initial_pipeline_once()
    assert len(runs) == 1


def test_widget_direct_construction_still_runs_initial_pipeline(qtbot, monkeypatch):
    runs = []
    monkeypatch.setattr(
        VippWidget,
        "run_pipeline",
        lambda self, *args, **kwargs: runs.append((args, kwargs)),
    )

    widget = VippWidget(_Viewer(), initial_compute_mode=ComputeMode.CPU)
    qtbot.addWidget(widget)

    assert len(runs) == 1
    assert widget.compute_mode_combo.currentData() == "cpu"
    assert not widget.run_initial_pipeline_once()


def test_compute_toolbar_defaults_to_auto_and_shows_actual_compute_badges(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)

    assert [
        widget.compute_mode_combo.itemData(index)
        for index in range(widget.compute_mode_combo.count())
    ] == ["auto", "cpu", "prefer_gpu", "custom"]
    assert [
        widget.compute_mode_combo.itemText(index)
        for index in range(widget.compute_mode_combo.count())
    ] == ["Auto", "CPU", "Prefer GPU", "Custom"]
    prefer_gpu_index = widget.compute_mode_combo.findData("prefer_gpu")
    assert "scientifically eligible" in str(
        widget.compute_mode_combo.itemData(prefer_gpu_index, Qt.ToolTipRole)
    )
    assert "without requiring it to beat CPU" in str(
        widget.compute_mode_combo.itemData(prefer_gpu_index, Qt.ToolTipRole)
    )
    assert "data-selection step on CPU" in str(
        widget.compute_mode_combo.itemData(prefer_gpu_index, Qt.ToolTipRole)
    )
    assert "Prefer GPU" in widget.compute_mode_combo.toolTip()
    assert widget.compute_mode_combo.currentData() == "auto"
    assert widget.graph_view._cards["input"].compute_badge.isHidden()
    badges = (
        widget.graph_view._cards["gaussian"].compute_badge.text(),
        widget.graph_view._cards["threshold"].compute_badge.text(),
    )
    assert all(text == "CPU" or text.startswith("GPU ·") for text in badges)
    assert widget.compute_status_label.text().startswith("Auto ·")
    assert widget.compute_group.isHidden()


def test_entering_custom_preserves_actual_results_without_recalculation(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    output = widget.pipeline.outputs["gaussian"]
    previous_request = widget._last_execution_report.request
    previous_badge = widget.graph_view._cards["gaussian"].compute_badge.text()
    monkeypatch.setattr(
        widget,
        "_invalidate_compute_policy_results",
        lambda *_args, **_kwargs: pytest.fail("Custom must not invalidate results"),
    )
    monkeypatch.setattr(
        widget,
        "run_pipeline",
        lambda: pytest.fail("Custom must not recalculate on entry"),
    )

    widget.compute_mode_combo.setCurrentIndex(
        widget.compute_mode_combo.findData("custom")
    )

    assert widget._compute_mode is ComputeMode.CUSTOM
    assert widget.pipeline.outputs["gaussian"] is output
    assert widget._last_execution_report.request is previous_request
    assert widget.graph_view._cards["gaussian"].compute_badge.text() == previous_badge
    assert widget.compute_status_label.text().startswith("Auto ·")
    assert "last actual Auto result" in widget.status_label.text()


def test_entering_custom_marks_mismatched_dormant_choice_as_previous(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    widget._compute_node_preferences["gaussian"] = NodeComputePreference(
        "implementation",
        "unavailable-test-gpu",
    )

    widget.compute_mode_combo.setCurrentIndex(
        widget.compute_mode_combo.findData("custom")
    )

    badge = widget.graph_view._cards["gaussian"].compute_badge
    assert widget.pipeline.outputs["gaussian"] is not None
    assert badge.text() in {"CPU", "GPU · CuPy", "GPU · cuCIM"}
    assert "Previous result (stale)" in badge.toolTip()
    assert "Current intent: exact implementation unavailable-test-gpu" in (
        badge.toolTip()
    )
    assert widget.compute_status_label.text().endswith("· previous")


def test_compute_mode_cannot_change_until_active_calculation_is_cancelled(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.uint8)))
    qtbot.addWidget(widget)
    widget._active_pipeline_run_id = 123
    widget._pipeline_cancel_events[123] = threading.Event()
    widget._pipeline_run_context[123] = (
        None,
        "input volume",
        "gaussian",
        widget._last_pipeline_source_signature,
        {"gaussian"},
    )
    widget._inflight_dirty_node_ids = {"gaussian"}
    widget._set_pipeline_busy(True, "gaussian")

    assert not widget.compute_mode_combo.isEnabled()
    widget._on_compute_mode_changed(widget.compute_mode_combo.findData("cpu"))

    assert widget._compute_mode is ComputeMode.AUTO
    assert widget.compute_mode_combo.currentData() == "auto"
    assert "Cancel the current calculation" in widget.status_label.text()


def test_widget_uses_one_severity_aware_message_strip(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)

    widget._set_status("Fix the source path.", severity="error", actionable=True)

    assert widget.status_label.property("fullWidthAlert") is True
    assert widget.status_label.property("messageSeverity") == "error"

    widget.graph_view.status_message.emit("Focused workflow node.")

    assert widget.status_label.text() == "Focused workflow node."
    assert widget.status_label.property("fullWidthAlert") is False
    assert widget.status_label.property("messageSeverity") == "neutral"


def test_diagnostic_failure_callbacks_classify_nonactionable_status(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget._active_thumbnail_contrast_run_id = 11
    widget._on_thumbnail_contrast_limit_finished(
        ThumbnailContrastLimitResult(11, frozenset(), {}, error="thumbnail boom")
    )
    assert widget.status_label.property("messageSeverity") == "warning"
    assert widget.status_label.property("fullWidthAlert") is False

    widget._active_auto_contrast_run_id = 12
    widget._active_auto_contrast_key = ("auto",)
    widget._on_auto_contrast_finished(
        AutoContrastResult(
            12,
            ("auto",),
            "gaussian",
            0.1,
            error="auto boom",
        )
    )
    assert widget.status_label.property("messageSeverity") == "error"
    assert widget.status_label.property("fullWidthAlert") is False

    layer = viewer.layers[0]
    layer.name = "VIPP Inspect"
    key = (widget._generated_layer_contrast_generation, "display")
    layer.metadata["_vipp_display_contrast_key"] = key
    widget._generated_layer_contrast_keys[layer.name] = key
    widget._on_generated_layer_contrast_finished(
        GeneratedLayerContrastResult(
            key,
            layer.name,
            error="display boom",
        )
    )
    assert widget.status_label.property("messageSeverity") == "warning"
    assert widget.status_label.property("fullWidthAlert") is False


@pytest.mark.parametrize("failure_kind", ("worker", "cache"))
def test_obsolete_source_failure_is_not_actionable_while_retrying_latest_edit(
    qtbot,
    monkeypatch,
    failure_kind,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    queued_runs: list[None] = []
    widget.run_pipeline = lambda *args, **kwargs: queued_runs.append(None)
    widget._active_source_load_id = 73
    widget._source_load_pending = True
    if failure_kind == "worker":
        result = SourceFileLoadResult(
            73,
            {},
            error="obsolete source failed",
            node_id="gaussian",
        )
    else:
        result = SourceFileLoadResult(73, {("source",): object()})

        def fail_to_cache(*_args, **_kwargs):
            raise RuntimeError("obsolete cache failed")

        monkeypatch.setattr(widget, "_cache_file_source_snapshot", fail_to_cache)

    widget._on_source_file_load_finished(result)

    assert widget._active_source_load_id is None
    assert not widget._source_load_pending
    assert widget.status_label.property("messageSeverity") == "error"
    assert widget.status_label.property("messageActionable") is False
    assert widget.status_label.property("fullWidthAlert") is False
    assert "retrying latest source edit" in widget.status_label.text()
    qtbot.waitUntil(lambda: bool(queued_runs))


def test_custom_gpu_choice_is_captured_by_background_request(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    pool = _QueuedThreadPool()
    widget._pipeline_thread_pool = pool

    widget.compute_mode_combo.setCurrentIndex(
        widget.compute_mode_combo.findData("custom")
    )
    widget.graph_view.select_node("gaussian")

    assert not widget.compute_group.isHidden()
    assert widget.compute_group.title() == "Compute"
    library_index = widget.node_compute_preference_combo.findData("library:cupy")
    assert library_index >= 0
    assert (
        "experimental"
        not in widget.node_compute_preference_combo.itemText(library_index).casefold()
    )

    widget.node_compute_preference_combo.setCurrentIndex(library_index)

    assert pool.workers
    request = pool.workers[-1].request.compute_request
    assert request.mode.value == "custom"
    assert not request.allow_experimental
    assert request.preference_for("gaussian") == NodeComputePreference(
        "library",
        "cupy",
    )


def test_node_benchmark_apply_is_atomic_and_undoable(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    widget._abandon_background_pipeline_run()
    widget.run_pipeline = lambda *args, **kwargs: None
    with QSignalBlocker(widget.compute_mode_combo):
        widget.compute_mode_combo.setCurrentIndex(
            widget.compute_mode_combo.findData("custom")
        )
    widget._compute_mode = ComputeMode.CUSTOM
    widget.graph_view.select_node("gaussian")
    widget._node_benchmark_baseline = widget._current_history_snapshot()
    result = SimpleNamespace(
        plan=SimpleNamespace(node_id="gaussian"),
        record=SimpleNamespace(accepted_implementation_id="cupyx-gaussian-filter-v1"),
        winner_preference=NodeComputePreference("library", "cupyx"),
    )
    undo_count = len(widget._undo_stack)

    widget._apply_node_benchmark_result(result)

    assert widget._compute_node_preferences["gaussian"] == NodeComputePreference(
        "library",
        "cupyx",
    )
    widget.undo()
    assert "gaussian" not in widget._compute_node_preferences
    widget.redo()
    assert widget._compute_node_preferences["gaussian"] == NodeComputePreference(
        "library",
        "cupyx",
    )
    assert len(widget._undo_stack) == undo_count + 1

    widget.undo()

    assert "gaussian" not in widget._compute_node_preferences


def test_node_benchmark_control_accepts_resolved_ordered_multi_input_node(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    widget._abandon_background_pipeline_run()
    widget._active_source_load_id = None
    node = widget.pipeline.add_node("richardson_lucy_tv_deconvolution")
    widget._selected_node_id = node.id
    widget._compute_mode = ComputeMode.CUSTOM
    image = np.ones((8, 8), dtype=np.float32)
    psf = np.ones((3, 3), dtype=np.float32) / np.float32(9)
    monkeypatch.setattr(
        widget.pipeline,
        "input_data_by_port_for_node",
        lambda _node_id: {0: image, 1: psf},
    )

    ready, reason = widget._can_benchmark_selected_node()

    assert ready
    assert "exact current input" in reason

    monkeypatch.setattr(
        widget.pipeline,
        "input_data_by_port_for_node",
        lambda _node_id: {0: image},
    )

    ready, reason = widget._can_benchmark_selected_node()

    assert not ready
    assert "every ordered input" in reason


def test_pipeline_optimizer_action_is_custom_only(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    widget.run_pipeline = lambda *args, **kwargs: None

    assert widget.optimize_pipeline_button.isHidden()
    widget._populate_settings_toolbar_menu()
    assert "Find fastest pipeline…" not in {
        action.text() for action in widget.settings_menu.actions()
    }

    with QSignalBlocker(widget.compute_mode_combo):
        widget.compute_mode_combo.setCurrentIndex(
            widget.compute_mode_combo.findData("prefer_gpu")
        )
    widget._compute_mode = ComputeMode.PREFER_GPU
    widget.graph_view.select_node("gaussian")
    widget._sync_node_compute_control()
    widget._sync_compute_toolbar_summary()
    widget._populate_settings_toolbar_menu()

    assert widget.compute_group.isHidden()
    assert widget.optimize_pipeline_button.isHidden()
    assert "Find fastest pipeline…" not in {
        action.text() for action in widget.settings_menu.actions()
    }

    with QSignalBlocker(widget.compute_mode_combo):
        widget.compute_mode_combo.setCurrentIndex(
            widget.compute_mode_combo.findData("custom")
        )
    widget._compute_mode = ComputeMode.CUSTOM
    widget._sync_compute_toolbar_summary()
    widget._populate_settings_toolbar_menu()

    assert not widget.optimize_pipeline_button.isHidden()
    assert "Find fastest pipeline…" in {
        action.text() for action in widget.settings_menu.actions()
    }


def test_optimizer_captures_exact_retained_mixed_assignment(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    widget._abandon_background_pipeline_run()
    widget._compute_mode = ComputeMode.CUSTOM
    authored = widget._current_compute_request()

    baseline = widget._retained_optimizer_baseline_request(authored)

    assert baseline is not None
    assert baseline.mode is ComputeMode.CUSTOM
    assert baseline.fallback_policy is FallbackPolicy.STRICT
    assert set(baseline.node_preferences) == set(widget._accepted_compute_decisions)
    for node_id, decision in widget._accepted_compute_decisions.items():
        preference = baseline.preference_for(node_id)
        if decision.runtime_id == "cpu-numpy":
            assert preference.kind is NodePreferenceKind.CPU
        else:
            assert preference == NodeComputePreference(
                NodePreferenceKind.IMPLEMENTATION,
                decision.implementation_id,
            )


def test_optimizer_uses_fresh_private_baseline_after_scientific_edit(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    widget._abandon_background_pipeline_run()
    widget._compute_mode = ComputeMode.CUSTOM
    authored = widget._current_compute_request()
    widget._pending_dirty_node_ids.add("gaussian")

    assert widget._retained_optimizer_baseline_request(authored) is None


def test_optimizer_lock_is_separate_undoable_and_does_not_recalculate(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    widget._abandon_background_pipeline_run()
    with QSignalBlocker(widget.compute_mode_combo):
        widget.compute_mode_combo.setCurrentIndex(
            widget.compute_mode_combo.findData("custom")
        )
    widget._compute_mode = ComputeMode.CUSTOM
    widget._compute_node_preferences["gaussian"] = NodeComputePreference(
        "library",
        "cupyx",
    )
    widget.graph_view.select_node("gaussian")
    runs = []
    widget.run_pipeline = lambda *args, **kwargs: runs.append(None)
    widget._sync_node_compute_control()

    widget.node_compute_optimizer_lock_checkbox.setChecked(True)

    assert widget._compute_optimizer_locked_node_ids == {"gaussian"}
    assert runs == []
    assert "will preserve" in widget.node_compute_note.text()

    widget.undo()
    assert widget._compute_optimizer_locked_node_ids == set()
    assert widget._compute_node_preferences["gaussian"] == NodeComputePreference(
        "library",
        "cupyx",
    )
    widget.redo()
    assert widget._compute_optimizer_locked_node_ids == {"gaussian"}

    auto_index = widget.node_compute_preference_combo.findData("auto")
    widget.node_compute_preference_combo.setCurrentIndex(auto_index)
    assert "gaussian" not in widget._compute_optimizer_locked_node_ids


def test_integer_gaussian_compute_note_does_not_duplicate_repair_advice(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.uint16)))
    qtbot.addWidget(widget)
    widget.run_pipeline = lambda *args, **kwargs: None
    widget.compute_mode_combo.setCurrentIndex(
        widget.compute_mode_combo.findData("custom")
    )
    widget.graph_view.select_node("gaussian")

    assert "Convert Dtype" not in widget.node_compute_note.text()


def test_pipeline_optimizer_apply_is_atomic_undoable_and_review_only(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    widget._abandon_background_pipeline_run()
    widget.run_pipeline = lambda *args, **kwargs: None
    with QSignalBlocker(widget.compute_mode_combo):
        widget.compute_mode_combo.setCurrentIndex(
            widget.compute_mode_combo.findData("custom")
        )
    widget._compute_mode = ComputeMode.CUSTOM
    widget._pipeline_optimizer_baseline = widget._current_history_snapshot()
    payloads, _layers = widget._source_payloads_for_pipeline()
    widget._pipeline_optimizer_source_signature = widget._pipeline_source_signature(
        None, None, "", payloads
    )
    row = SimpleNamespace(
        node_id="gaussian",
        current_preference=NodeComputePreference(),
        proposed_preference=NodeComputePreference("library", "cupyx"),
    )

    class _Proposal:
        rows = (row,)

        def is_current(self, identity, request, assignments):
            return bool(identity and request.mode is ComputeMode.CUSTOM and assignments)

        def updated_request(self, request):
            return replace(
                request,
                node_preferences={
                    **dict(request.node_preferences),
                    "gaussian": NodeComputePreference("library", "cupyx"),
                },
            )

    monkeypatch.setattr(
        "napari_vipp.core.compute_pipeline_optimizer_coordinator."
        "probe_pipeline_optimizer_environment",
        lambda *_args: SimpleNamespace(fingerprint="environment-a"),
    )
    monkeypatch.setattr(
        "napari_vipp.core.compute_pipeline_optimizer_coordinator."
        "fingerprint_pipeline_optimizer_sources",
        lambda *_args: "source-a",
    )
    monkeypatch.setattr(
        "napari_vipp.core.compute_benchmark_coordinator."
        "benchmark_environment_fingerprint",
        lambda *_args: "benchmark-a",
    )
    retention_fingerprint = canonical_digest(sorted(widget._cache_retention_node_ids()))
    qualification = ExactWorkloadCandidateQualification(
        node_id="gaussian",
        operation_id="gaussian_blur",
        implementation_id="implementation-a",
        implementation_version="1",
        workload_identity_digest="workload-a",
        benchmark_workload_fingerprint="benchmark-workload-a",
        compute_environment_fingerprint="environment-a",
        benchmark_environment_fingerprint="benchmark-a",
        parity_policy_id="parity-a",
        benchmark_record_digest="record-a",
        qualification_scope_digest="scope-a",
    )
    result = SimpleNamespace(
        proposal=_Proposal(),
        identity=SimpleNamespace(
            digest="scope-a",
            environment_fingerprint="environment-a",
            benchmark_environment_fingerprint="benchmark-a",
            cache_retention_fingerprint=retention_fingerprint,
            source_fingerprint="source-a",
        ),
        exact_workload_qualifications=frozenset({qualification}),
    )
    undo_count = len(widget._undo_stack)

    # Merely constructing/reviewing a result cannot mutate authored intent.
    assert "gaussian" not in widget._compute_node_preferences
    widget._apply_pipeline_optimizer_result(result)

    assert widget._compute_node_preferences["gaussian"] == NodeComputePreference(
        "library",
        "cupyx",
    )
    assert len(widget._undo_stack) == undo_count + 1
    assert widget._exact_workload_qualifications == {qualification}
    assert widget._exact_workload_qualification_scope_digest == "scope-a"
    qualified_workflow = deepcopy(
        serialize_workflow(
            widget.pipeline,
            compute_request=widget._current_compute_request(),
        )
    )
    assert widget._validated_exact_workload_qualifications(
        qualified_workflow,
        widget._pipeline_optimizer_source_signature,
    ) == (frozenset({qualification}), "scope-a")

    widget.undo()

    assert "gaussian" not in widget._compute_node_preferences
    assert not widget._exact_workload_qualifications

    widget._apply_pipeline_optimizer_result(
        SimpleNamespace(
            proposal=_Proposal(),
            identity=SimpleNamespace(
                environment_fingerprint="environment-b",
                benchmark_environment_fingerprint="benchmark-a",
                cache_retention_fingerprint=retention_fingerprint,
                source_fingerprint="source-a",
            ),
        )
    )

    assert "gaussian" not in widget._compute_node_preferences
    assert "environment changed" in widget.status_label.text()

    widget._apply_pipeline_optimizer_result(
        SimpleNamespace(
            proposal=_Proposal(),
            identity=SimpleNamespace(
                environment_fingerprint="environment-a",
                benchmark_environment_fingerprint="benchmark-a",
                cache_retention_fingerprint=retention_fingerprint,
                source_fingerprint="source-b",
            ),
        )
    )

    assert "gaussian" not in widget._compute_node_preferences
    assert "source bytes" in widget.status_label.text()

    widget._apply_pipeline_optimizer_result(
        SimpleNamespace(
            proposal=_Proposal(),
            identity=SimpleNamespace(
                environment_fingerprint="environment-a",
                benchmark_environment_fingerprint="benchmark-b",
                cache_retention_fingerprint=retention_fingerprint,
                source_fingerprint="source-a",
            ),
        )
    )

    assert "gaussian" not in widget._compute_node_preferences
    assert "benchmark environment changed" in widget.status_label.text()

    widget._apply_pipeline_optimizer_result(
        SimpleNamespace(
            proposal=_Proposal(),
            identity=SimpleNamespace(
                environment_fingerprint="environment-a",
                benchmark_environment_fingerprint="benchmark-a",
                cache_retention_fingerprint="stale-retention",
                source_fingerprint="source-a",
            ),
        )
    )

    assert "gaussian" not in widget._compute_node_preferences
    assert "Analyze the pipeline again" in widget.status_label.text()

    class FailingProbeCleanupRegistry:
        def close(self):
            raise RuntimeError("probe cleanup failed")

    monkeypatch.setattr(
        "napari_vipp.core.compute_registry.ComputeRegistry",
        FailingProbeCleanupRegistry,
    )
    widget._apply_pipeline_optimizer_result(result)

    assert "gaussian" not in widget._compute_node_preferences
    assert "did not clean up safely" in widget.status_label.text()

    monkeypatch.undo()
    monkeypatch.setattr(
        "napari_vipp.core.compute_pipeline_optimizer_coordinator."
        "probe_pipeline_optimizer_environment",
        lambda *_args: SimpleNamespace(fingerprint="environment-a"),
    )
    monkeypatch.setattr(
        "napari_vipp.core.compute_pipeline_optimizer_coordinator."
        "fingerprint_pipeline_optimizer_sources",
        lambda *_args: "source-a",
    )
    monkeypatch.setattr(
        "napari_vipp.core.compute_benchmark_coordinator."
        "benchmark_environment_fingerprint",
        lambda *_args: "benchmark-a",
    )

    def mutate_graph_during_source_verification(*_args):
        widget.pipeline.nodes["gaussian"].params["sigma"] = 3.25
        return "source-a"

    monkeypatch.setattr(
        "napari_vipp.core.compute_pipeline_optimizer_coordinator."
        "fingerprint_pipeline_optimizer_sources",
        mutate_graph_during_source_verification,
    )
    widget._apply_pipeline_optimizer_result(result)

    assert "gaussian" not in widget._compute_node_preferences
    assert "changed during verification" in widget.status_label.text()


def test_pipeline_optimizer_rejects_stale_review_result(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    widget._abandon_background_pipeline_run()
    widget.run_pipeline = lambda *args, **kwargs: None
    widget._compute_mode = ComputeMode.CUSTOM
    widget._pipeline_optimizer_baseline = widget._current_history_snapshot()
    payloads, _layers = widget._source_payloads_for_pipeline()
    widget._pipeline_optimizer_source_signature = widget._pipeline_source_signature(
        None, None, "", payloads
    )
    widget.pipeline.nodes["gaussian"].params["sigma"] = 3.25
    proposal = SimpleNamespace(
        rows=(),
        is_current=lambda *_args: True,
        updated_request=lambda request: request,
    )

    widget._apply_pipeline_optimizer_result(
        SimpleNamespace(proposal=proposal, identity=object())
    )

    assert widget._compute_node_preferences == {}
    assert "Analyze the pipeline again" in widget.status_label.text()


def test_pipeline_optimizer_requires_digest_bound_near_parity_acceptance(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    widget._abandon_background_pipeline_run()
    widget.run_pipeline = lambda *args, **kwargs: None
    widget._compute_mode = ComputeMode.CUSTOM
    current_preference = NodeComputePreference()
    proposed_preference = NodeComputePreference("library", "cupy")
    deviation = PipelineParityDeviation(
        "gaussian",
        "gaussian_blur",
        0,
        PipelineParityReviewMetric.NORMALIZED_RMSE,
        0.0001,
        0.001,
        64,
        64,
        1.0,
        0.0002,
        0.0002,
        0.0001,
        0.0001,
        "Small measured floating-point difference.",
    )
    proposal = PipelineOptimizationProposal(
        "identity",
        "request",
        (("gaussian", "cpu-gaussian"),),
        (
            PipelineOptimizationRow(
                "gaussian",
                "cpu-gaussian",
                "cupy-gaussian",
                current_preference,
                proposed_preference,
                True,
                True,
            ),
        ),
        {"gaussian": proposed_preference},
        1.0,
        0.5,
        1.0,
        0.5,
        1.5,
        True,
        5,
        0.5,
        PipelineValidationWinner.PROPOSED,
        (("gaussian", "cupy-gaussian"),),
        PipelineOptimizationSelectionBasis.PAIRED_VALIDATED_ALTERNATIVE,
        (deviation,),
    )
    result = SimpleNamespace(proposal=proposal, identity=object())

    widget._apply_pipeline_optimizer_result(result)

    assert "not been explicitly accepted" in widget.status_label.text()
    assert widget._compute_node_preferences == {}

    widget._apply_pipeline_optimizer_result(
        PipelineOptimizerApplyRequest(result, proposal.parity_review_digest)
    )

    assert "not been explicitly accepted" not in widget.status_label.text()
    assert "Analyze the pipeline again" in widget.status_label.text()
    assert widget._compute_node_preferences == {}


def test_background_run_forwards_only_source_current_exact_qualification(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    qtbot.waitUntil(lambda: widget._active_pipeline_run_id is None, timeout=30_000)
    pool = _QueuedThreadPool()
    widget._pipeline_thread_pool = pool
    payloads, _layers = widget._source_payloads_for_pipeline()
    workflow = deepcopy(
        serialize_workflow(
            widget.pipeline,
            compute_request=widget._current_compute_request(),
        )
    )
    qualification = ExactWorkloadCandidateQualification(
        node_id="gaussian",
        operation_id="gaussian_blur",
        implementation_id="implementation-a",
        implementation_version="1",
        workload_identity_digest="workload-a",
        benchmark_workload_fingerprint="benchmark-workload-a",
        compute_environment_fingerprint="environment-a",
        benchmark_environment_fingerprint="benchmark-environment-a",
        parity_policy_id="parity-a",
        benchmark_record_digest="record-a",
        qualification_scope_digest="scope-a",
    )
    widget._exact_workload_qualifications = frozenset({qualification})
    widget._exact_workload_qualification_scope_digest = "scope-a"
    widget._exact_workload_qualification_workflow_fingerprint = canonical_digest(
        workflow
    )
    source_signature = widget._pipeline_source_signature(
        None,
        None,
        "",
        payloads,
    )
    widget._exact_workload_qualification_source_signature = source_signature

    widget._start_background_pipeline_run(
        None,
        None,
        "",
        dict(payloads),
        None,
        "input volume",
        source_signature,
        {"gaussian"},
    )

    assert len(pool.workers) == 1
    request = pool.workers[0].request
    assert request.exact_workload_qualifications == {qualification}
    assert request.exact_workload_qualification_scope_digest == "scope-a"
    widget._on_background_pipeline_finished(
        PipelineRunResult(request.run_id, {}, cancelled=True)
    )

    # A changed source revision invalidates the private proof before dispatch.
    changed_payloads = dict(payloads)
    source_id, original = next(iter(changed_payloads.items()))
    changed_data = np.array(original.data, copy=True)
    changed_data.flat[0] += 1
    changed_payloads[source_id] = replace(
        original,
        data=changed_data,
        revision_token=replace(
            original.revision_token,
            revision=original.revision_token.revision + 1,
        ),
    )
    pool.workers.clear()
    widget._exact_workload_qualifications = frozenset({qualification})
    widget._exact_workload_qualification_scope_digest = "scope-a"
    widget._exact_workload_qualification_source_signature = source_signature
    widget._exact_workload_qualification_workflow_fingerprint = canonical_digest(
        workflow
    )
    changed_source_signature = widget._pipeline_source_signature(
        None,
        None,
        "",
        changed_payloads,
    )
    widget._start_background_pipeline_run(
        None,
        None,
        "",
        changed_payloads,
        None,
        "input volume",
        changed_source_signature,
        {"gaussian"},
    )

    assert len(pool.workers) == 1
    stale_request = pool.workers[0].request
    assert not stale_request.exact_workload_qualifications
    assert stale_request.exact_workload_qualification_scope_digest == ""
    assert not widget._exact_workload_qualifications
    widget._on_background_pipeline_finished(
        PipelineRunResult(stale_request.run_id, {}, cancelled=True)
    )


def test_pipeline_optimizer_dispatch_failure_clears_review_baseline(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    widget._abandon_background_pipeline_run()
    widget.run_pipeline = lambda *args, **kwargs: None
    widget._compute_mode = ComputeMode.CUSTOM
    monkeypatch.setattr(widget, "_can_optimize_pipeline", lambda: (True, ""))

    class FailingDialog:
        running = False

        @staticmethod
        def start(_worker, _pool):
            raise RuntimeError("thread pool unavailable")

    widget._pipeline_optimizer_dialog = FailingDialog()

    widget._start_pipeline_optimizer_analysis()

    assert widget._pipeline_optimizer_baseline is None
    assert widget._pipeline_optimizer_source_signature is None
    assert "could not start" in widget.status_label.text()


def test_pipeline_optimizer_uses_dialog_time_limit(qtbot, monkeypatch):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    widget._abandon_background_pipeline_run()
    widget.run_pipeline = lambda *args, **kwargs: None
    widget._compute_mode = ComputeMode.CUSTOM
    monkeypatch.setattr(widget, "_can_optimize_pipeline", lambda: (True, ""))
    captured = {}

    class Registry:
        @staticmethod
        def close():
            return None

    class Coordinator:
        def __init__(self, _registry, _store_path):
            pass

        @staticmethod
        def optimize(*_args, **kwargs):
            captured.update(kwargs)
            return object()

    class SynchronousDialog:
        running = False
        time_budget_seconds = 1_800.0

        @staticmethod
        def start(worker, _pool):
            worker.run()

    monkeypatch.setattr(
        "napari_vipp.core.compute_registry.ComputeRegistry",
        Registry,
    )
    monkeypatch.setattr(
        "napari_vipp.core.compute_pipeline_optimizer_coordinator."
        "ApplicationPipelineOptimizerCoordinator",
        Coordinator,
    )
    widget._pipeline_optimizer_dialog = SynchronousDialog()

    widget._start_pipeline_optimizer_analysis()

    assert captured["time_budget_seconds"] == pytest.approx(1_800.0)
    baseline = captured["baseline_compute_request"]
    assert baseline is not None
    assert baseline.mode is ComputeMode.CUSTOM
    assert baseline.fallback_policy is FallbackPolicy.STRICT


def test_pipeline_optimizer_apply_requires_active_calculation_to_be_cancelled(
    qtbot,
):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    widget._compute_mode = ComputeMode.CUSTOM
    widget._active_pipeline_run_id = 123

    widget._apply_pipeline_optimizer_result(
        SimpleNamespace(proposal=SimpleNamespace(rows=()), identity=object())
    )

    assert "Cancel the current calculation" in widget.status_label.text()


def test_stale_retained_assignments_use_optimizer_private_baseline(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    rows = tuple(
        SimpleNamespace(node_id=node_id)
        for node_id in widget._accepted_compute_decisions
    )
    baseline = tuple(
        (node_id, decision.implementation_id)
        for node_id, decision in widget._accepted_compute_decisions.items()
    )
    proposal = SimpleNamespace(rows=rows, baseline_assignment=baseline)
    widget._stale_compute_badge_node_ids.update(widget._accepted_compute_decisions)

    assert widget._current_pipeline_optimizer_assignments(proposal) == dict(baseline)


def test_clean_optimizer_assignments_reconstruct_source_from_authoritative_cpu_spec(
    qtbot,
):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    widget._abandon_background_pipeline_run()
    widget._compute_mode = ComputeMode.CUSTOM
    widget._pending_dirty_node_ids.clear()
    widget._stale_compute_badge_node_ids.clear()
    gaussian_decision = widget._accepted_compute_decisions["gaussian"]
    request = widget._current_compute_request()
    source_preference = request.preference_for("input")
    gaussian_preference = request.preference_for("gaussian")
    baseline = (
        ("input", "cpu-input-v1"),
        ("gaussian", gaussian_decision.implementation_id),
    )
    identity = PipelineOptimizationIdentity(
        pipeline_fingerprint="pipeline-a",
        source_fingerprint="source-a",
        topology_fingerprint="topology-a",
        cache_retention_fingerprint="retention-a",
        environment_fingerprint="environment-a",
        workload_fingerprints={
            "input": "source-workload-a",
            "gaussian": "gaussian-workload-a",
        },
    )
    proposal = PipelineOptimizationProposal(
        identity_digest=identity.digest,
        request_fingerprint=request.fingerprint,
        baseline_assignment=baseline,
        rows=(
            PipelineOptimizationRow(
                "input",
                "cpu-input-v1",
                "cpu-input-v1",
                source_preference,
                source_preference,
                False,
                False,
            ),
            PipelineOptimizationRow(
                "gaussian",
                gaussian_decision.implementation_id,
                gaussian_decision.implementation_id,
                gaussian_preference,
                gaussian_preference,
                False,
                True,
            ),
        ),
        preference_mapping={
            "input": source_preference,
            "gaussian": gaussian_preference,
        },
        estimated_current_seconds=1.0,
        estimated_proposed_seconds=1.0,
        validated_current_seconds=0.0,
        validated_proposed_seconds=0.0,
        validated_speedup_lower_confidence_bound=0.0,
        pipeline_validation_performed=False,
        validation_winner=PipelineValidationWinner.CURRENT,
        tested_assignment=baseline,
        selection_basis=(
            PipelineOptimizationSelectionBasis.EXACT_MODEL_RETAINED_CURRENT
        ),
    )

    assert not widget.pipeline.operation_spec("input").has_input
    assert widget.pipeline.operation_spec("gaussian_blur").has_input
    assignments = widget._current_pipeline_optimizer_assignments(proposal)
    assert assignments == dict(baseline)
    assert proposal.is_current(identity, request, assignments)

    mismatched_source = replace(
        proposal,
        baseline_assignment=(
            ("input", "forged-source-implementation"),
            ("gaussian", gaussian_decision.implementation_id),
        ),
    )
    mismatched_assignments = widget._current_pipeline_optimizer_assignments(
        mismatched_source
    )
    assert mismatched_assignments["input"] == "cpu-input-v1"
    assert not mismatched_source.is_current(identity, request, mismatched_assignments)

    widget._accepted_compute_decisions.pop("gaussian")

    # Only a source row may use its authoritative declared identity. Missing
    # provenance for an executable operation must continue to stale the result.
    assert widget._current_pipeline_optimizer_assignments(proposal) == {
        "input": "cpu-input-v1",
        "gaussian": "<no-current-actual-assignment>",
    }


def test_pipeline_optimizer_clean_apply_accepts_real_source_row(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    widget._abandon_background_pipeline_run()
    widget.run_pipeline = lambda *args, **kwargs: None
    with QSignalBlocker(widget.compute_mode_combo):
        widget.compute_mode_combo.setCurrentIndex(
            widget.compute_mode_combo.findData("custom")
        )
    widget._compute_mode = ComputeMode.CUSTOM
    widget._pending_dirty_node_ids.clear()
    widget._stale_compute_badge_node_ids.clear()
    widget._pipeline_optimizer_baseline = widget._current_history_snapshot()
    payloads, _layers = widget._source_payloads_for_pipeline()
    widget._pipeline_optimizer_source_signature = widget._pipeline_source_signature(
        None,
        None,
        "",
        payloads,
    )
    request = widget._current_compute_request()
    gaussian_decision = widget._accepted_compute_decisions["gaussian"]
    retention_fingerprint = canonical_digest(sorted(widget._cache_retention_node_ids()))
    identity = PipelineOptimizationIdentity(
        pipeline_fingerprint="pipeline-a",
        source_fingerprint="source-a",
        topology_fingerprint="topology-a",
        cache_retention_fingerprint=retention_fingerprint,
        environment_fingerprint="environment-a",
        benchmark_environment_fingerprint="benchmark-a",
        workload_fingerprints={
            "input": "source-workload-a",
            "gaussian": "gaussian-workload-a",
        },
    )
    source_preference = request.preference_for("input")
    gaussian_preference = request.preference_for("gaussian")
    proposed_preference = NodeComputePreference("library", "cupy")
    baseline = (
        ("input", "cpu-input-v1"),
        ("gaussian", gaussian_decision.implementation_id),
    )
    tested = (
        ("input", "cpu-input-v1"),
        ("gaussian", "cupy-gaussian-blur-v1"),
    )
    proposal = PipelineOptimizationProposal(
        identity_digest=identity.digest,
        request_fingerprint=request.fingerprint,
        baseline_assignment=baseline,
        rows=(
            PipelineOptimizationRow(
                "input",
                "cpu-input-v1",
                "cpu-input-v1",
                source_preference,
                source_preference,
                False,
                False,
            ),
            PipelineOptimizationRow(
                "gaussian",
                gaussian_decision.implementation_id,
                "cupy-gaussian-blur-v1",
                gaussian_preference,
                proposed_preference,
                True,
                True,
            ),
        ),
        preference_mapping={
            "input": source_preference,
            "gaussian": proposed_preference,
        },
        estimated_current_seconds=1.0,
        estimated_proposed_seconds=0.5,
        validated_current_seconds=1.0,
        validated_proposed_seconds=0.5,
        validated_speedup_lower_confidence_bound=1.5,
        pipeline_validation_performed=True,
        validation_measurement_rounds=5,
        validated_current_speedup_lower_confidence_bound=0.5,
        validation_winner=PipelineValidationWinner.PROPOSED,
        tested_assignment=tested,
        selection_basis=(
            PipelineOptimizationSelectionBasis.PAIRED_VALIDATED_ALTERNATIVE
        ),
    )

    class Registry:
        @staticmethod
        def close():
            return None

    monkeypatch.setattr(
        "napari_vipp.core.compute_registry.ComputeRegistry",
        Registry,
    )
    monkeypatch.setattr(
        "napari_vipp.core.compute_pipeline_optimizer_coordinator."
        "probe_pipeline_optimizer_environment",
        lambda *_args: SimpleNamespace(fingerprint="environment-a"),
    )
    monkeypatch.setattr(
        "napari_vipp.core.compute_pipeline_optimizer_coordinator."
        "fingerprint_pipeline_optimizer_sources",
        lambda *_args: "source-a",
    )
    monkeypatch.setattr(
        "napari_vipp.core.compute_benchmark_coordinator."
        "benchmark_environment_fingerprint",
        lambda *_args: "benchmark-a",
    )

    widget._apply_pipeline_optimizer_result(
        SimpleNamespace(proposal=proposal, identity=identity)
    )

    assert widget._compute_node_preferences["gaussian"] == proposed_preference
    assert "Applied 1 measured pipeline compute choice" in widget.status_label.text()


def test_pipeline_optimizer_cleanup_failure_cannot_publish_result(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    widget._abandon_background_pipeline_run()
    widget.run_pipeline = lambda *args, **kwargs: None
    widget._compute_mode = ComputeMode.CUSTOM
    monkeypatch.setattr(widget, "_can_optimize_pipeline", lambda: (True, ""))
    outcomes = []
    existing_key = SimpleNamespace(digest="existing-exact-benchmark")
    existing_record = SimpleNamespace(key=existing_key, marker="previous")
    replacement_record = SimpleNamespace(key=existing_key, marker="replacement")
    new_key = SimpleNamespace(digest="new-exact-benchmark")
    new_record = SimpleNamespace(key=new_key, marker="new")

    class TrackingStore:
        def __init__(self):
            self.records = {existing_key.digest: existing_record}
            self.discarded = []

        def get(self, key):
            return self.records.get(key.digest)

        def put(self, record):
            self.records[record.key.digest] = record

        def discard(self, key):
            self.discarded.append(key)
            self.records.pop(key.digest, None)

    store = TrackingStore()

    class FailingCleanupRegistry:
        def close(self):
            raise RuntimeError("GPU cleanup failed")

    class SuccessfulCoordinator:
        def __init__(self, _registry, _store_path):
            self.node_benchmarker = SimpleNamespace(store=store)

        def optimize(self, *_args, **_kwargs):
            self.node_benchmarker.store.put(replacement_record)
            self.node_benchmarker.store.put(new_record)
            return object()

    class SynchronousDialog:
        running = False

        @staticmethod
        def start(worker, _pool):
            worker.signals.finished.connect(outcomes.append)
            worker.run()

    monkeypatch.setattr(
        "napari_vipp.core.compute_registry.ComputeRegistry",
        FailingCleanupRegistry,
    )
    monkeypatch.setattr(
        "napari_vipp.core.compute_pipeline_optimizer_coordinator."
        "ApplicationPipelineOptimizerCoordinator",
        SuccessfulCoordinator,
    )
    widget._pipeline_optimizer_dialog = SynchronousDialog()

    widget._start_pipeline_optimizer_analysis()

    assert len(outcomes) == 1
    assert outcomes[0].result is None
    assert outcomes[0].reason_code == "cleanup_failed"
    assert "cleanup failed" in outcomes[0].error
    assert store.records == {existing_key.digest: existing_record}
    assert store.discarded == [new_key]


def test_pipeline_optimizer_rollback_failure_quarantines_entire_store(
    qtbot,
    monkeypatch,
    tmp_path,
):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    widget._abandon_background_pipeline_run()
    widget.run_pipeline = lambda *args, **kwargs: None
    widget._compute_mode = ComputeMode.CUSTOM
    monkeypatch.setattr(widget, "_can_optimize_pipeline", lambda: (True, ""))
    benchmark_path = tmp_path / "compute-benchmarks-v1.json"
    benchmark_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "napari_vipp._widget._default_compute_benchmark_store_path",
        lambda: benchmark_path,
    )
    outcomes = []
    key = SimpleNamespace(digest="existing-exact-benchmark")
    existing_record = SimpleNamespace(key=key, marker="previous")
    replacement_record = SimpleNamespace(key=key, marker="replacement")

    class FailingRestoreStore:
        def __init__(self):
            self.current = existing_record

        def get(self, _key):
            return self.current

        def put(self, record):
            if record is existing_record:
                raise OSError("synthetic restore failure")
            self.current = record

        def discard(self, _key):
            self.current = None

    store = FailingRestoreStore()

    class FailingCleanupRegistry:
        def close(self):
            raise RuntimeError("GPU cleanup failed")

    class SuccessfulCoordinator:
        def __init__(self, _registry, _store_path):
            self.node_benchmarker = SimpleNamespace(store=store)

        def optimize(self, *_args, **_kwargs):
            self.node_benchmarker.store.put(replacement_record)
            return object()

    class SynchronousDialog:
        running = False

        @staticmethod
        def start(worker, _pool):
            worker.signals.finished.connect(outcomes.append)
            worker.run()

    monkeypatch.setattr(
        "napari_vipp.core.compute_registry.ComputeRegistry",
        FailingCleanupRegistry,
    )
    monkeypatch.setattr(
        "napari_vipp.core.compute_pipeline_optimizer_coordinator."
        "ApplicationPipelineOptimizerCoordinator",
        SuccessfulCoordinator,
    )
    widget._pipeline_optimizer_dialog = SynchronousDialog()

    widget._start_pipeline_optimizer_analysis()

    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.reason_code == "cleanup_failed"
    assert "could not be fully rolled back" in outcome.error
    assert "will not be reused" in outcome.error
    assert not benchmark_path.exists()
    quarantined = tuple(tmp_path.glob("compute-benchmarks-v1.json.unsafe-*"))
    assert len(quarantined) == 1

    widget._on_pipeline_optimizer_finished(outcome)

    assert "No proposed assignment was accepted" in widget.status_label.text()
    assert "will not be reused" in widget.status_label.text()
    assert widget.status_label.property("messageSeverity") == "error"


def test_benchmark_cleanup_failure_quarantines_all_compute(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    widget._pipeline_run_pending = True

    widget._on_node_benchmark_finished(
        NodeBenchmarkWorkerOutcome(
            "input",
            error="Benchmark cleanup failed: provider did not close",
            reason_code="cleanup_failed",
        )
    )

    assert not widget._pipeline_run_pending
    assert "Restart VIPP" in widget._compute_runtime_quarantined_reason
    assert not widget.compute_mode_combo.isEnabled()
    assert "No benchmark preference was accepted" in widget.status_label.text()
    assert "Benchmark cleanup failed" in widget.status_label.text()


def test_optimizer_cleanup_failure_quarantines_all_compute(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    widget._pipeline_run_pending = True

    widget._on_pipeline_optimizer_finished(
        PipelineOptimizerWorkerOutcome(
            error="Find fastest cleanup failed",
            reason_code="cleanup_failed",
        )
    )

    assert not widget._pipeline_run_pending
    assert "Restart VIPP" in widget._compute_runtime_quarantined_reason
    assert not widget.compute_mode_combo.isEnabled()
    assert "proposed assignment" in widget.status_label.text()


def test_inconclusive_optimizer_result_keeps_measurements_reviewable(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    widget._pipeline_run_pending = False
    proposal = SimpleNamespace(
        rows=(),
        pipeline_validation_performed=True,
        validation_winner=SimpleNamespace(value="inconclusive"),
    )

    widget._on_pipeline_optimizer_finished(
        PipelineOptimizerWorkerOutcome(
            result=SimpleNamespace(proposal=proposal),
        )
    )

    assert "neither pipeline was clearly faster" in widget.status_label.text()
    assert "complete measurements remain available" in widget.status_label.text()
    assert widget.status_label.property("messageSeverity") == "info"


def test_inconclusive_optimizer_result_cannot_be_applied_programmatically(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    before = dict(widget._compute_node_preferences)
    proposal = SimpleNamespace(
        validation_winner=SimpleNamespace(value="inconclusive"),
    )

    widget._apply_pipeline_optimizer_result(
        SimpleNamespace(proposal=proposal),
    )

    assert widget._compute_node_preferences == before
    assert "no clear speed winner" in widget.status_label.text().lower()
    assert "current settings remain unchanged" in widget.status_label.text().lower()
    assert widget.status_label.property("messageSeverity") == "info"


def test_pending_graph_work_reports_active_optimizer_owner(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    widget._pipeline_run_pending = True
    widget._pipeline_optimizer_dialog = SimpleNamespace(running=True)

    reason = widget._compute_policy_edit_block_reason()

    assert "Find fastest" in reason
    assert "current calculation" not in reason


def test_normal_pipeline_run_waits_for_optimizer_evidence_window(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    widget._abandon_background_pipeline_run()
    widget._pipeline_optimizer_dialog = SimpleNamespace(running=True)

    widget.run_pipeline()

    assert widget._pipeline_run_pending
    assert "exclusive GPU evidence window" in widget.status_label.text()

    runs = []
    widget._pipeline_optimizer_dialog.running = False
    widget.run_pipeline = lambda *args, **kwargs: runs.append(None)
    monkeypatch.setattr(QTimer, "singleShot", lambda _delay, callback: callback())
    widget._resume_pipeline_after_optimizer_if_pending()

    assert not widget._pipeline_run_pending
    assert runs == [None]


def test_scoped_compute_invalidation_preserves_clean_upstream_cache(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    widget._abandon_background_pipeline_run()
    upstream = widget.pipeline.outputs["gaussian"]
    completed = set(widget.pipeline.completed_node_ids)

    widget._invalidate_compute_policy_results({"threshold"})

    assert widget.pipeline.outputs["gaussian"] is upstream
    assert "gaussian" in widget.pipeline.completed_node_ids
    assert completed - {"threshold"} <= widget.pipeline.completed_node_ids
    assert widget._pending_dirty_node_ids == {"threshold"}


@pytest.mark.parametrize(
    ("mode", "expected_background"),
    (
        (ComputeMode.AUTO, False),
        (ComputeMode.PREFER_GPU, True),
        (ComputeMode.CUSTOM, True),
    ),
)
def test_non_cpu_compute_modes_use_detached_compute_service(
    qtbot,
    mode,
    expected_background,
):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    widget._compute_mode = mode

    assert widget._background_processing_node_id({"gaussian"}) is None
    assert (
        widget._should_run_pipeline_in_background({"gaussian"}) is expected_background
    )


@pytest.mark.parametrize(
    "mode",
    (ComputeMode.AUTO, ComputeMode.PREFER_GPU, ComputeMode.CUSTOM),
)
def test_force_sync_required_gpu_intent_still_uses_detached_compute_service(
    qtbot,
    monkeypatch,
    mode,
):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    widget._abandon_background_pipeline_run()
    widget._compute_mode = mode
    captured: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        widget,
        "_run_pipeline_synchronously",
        lambda *_args, **_kwargs: pytest.fail(
            "Required GPU intent must not bypass the detached compute service"
        ),
    )
    monkeypatch.setattr(
        widget,
        "_start_background_pipeline_run",
        lambda *args, **kwargs: captured.append((args, kwargs)),
    )

    widget.run_pipeline(force_sync=True)

    assert len(captured) == 1
    assert captured[0][1]["execute_synchronously"] is True
    assert widget._current_compute_request().mode is mode


def test_background_auto_run_is_detached_and_cancel_button_remains_usable(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    pool = _QueuedThreadPool()
    widget._pipeline_thread_pool = pool
    widget.background_all_checkbox.setChecked(True)
    widget._invalidate_pipeline_cache()

    widget.run_pipeline()

    assert len(pool.workers) == 1
    run_id = widget._active_pipeline_run_id
    assert run_id is not None
    assert not widget.pipeline_cancel_button.isHidden()
    assert widget.pipeline_cancel_button.isEnabled()
    assert not widget.compute_mode_combo.isEnabled()

    widget._cancel_background_pipeline_run()

    assert widget._pipeline_cancel_events[run_id].is_set()
    assert widget.pipeline_cancel_button.isHidden()
    assert not widget.compute_mode_combo.isEnabled()
    pool.workers[0].run()
    qtbot.waitUntil(lambda: widget._active_pipeline_run_id is None)
    assert widget.compute_mode_combo.isEnabled()


def test_small_auto_wait_keeps_qt_cancel_action_responsive(qtbot, monkeypatch):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    cancel_clicked = []

    def wait_for_cancel(request, **_kwargs):
        assert request.cancel_event.wait(timeout=5)
        return PipelineRunResult(
            request.run_id,
            request.workflow,
            cancelled=True,
            failure=PipelineExecutionFailure(
                kind="cancelled",
                error_type="OperationCancelled",
                message="cancelled by test",
                cleanup_succeeded=True,
            ),
        )

    monkeypatch.setattr(
        "napari_vipp.ui.workers.execute_pipeline_request",
        wait_for_cancel,
    )
    widget._invalidate_pipeline_cache()

    def click_cancel() -> None:
        assert widget._active_pipeline_run_id is not None
        assert not widget.pipeline_cancel_button.isHidden()
        cancel_clicked.append(True)
        widget.pipeline_cancel_button.click()

    QTimer.singleShot(0, click_cancel)
    widget.run_pipeline()

    assert cancel_clicked == [True]
    assert widget._active_pipeline_run_id is None
    assert widget.compute_mode_combo.isEnabled()
    assert "cleanup completed" in widget.status_label.text()


def test_small_cpu_run_keeps_existing_responsiveness_heuristic(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    widget._compute_mode = ComputeMode.CPU

    assert widget._background_processing_node_id({"gaussian"}) is None
    assert widget._should_run_pipeline_in_background({"gaussian"}) is False


def test_custom_exact_pin_is_honest_until_user_replaces_it(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    widget.run_pipeline = lambda *args, **kwargs: None
    widget.compute_mode_combo.setCurrentIndex(
        widget.compute_mode_combo.findData("custom")
    )
    widget.graph_view.select_node("gaussian")
    exact_value = "implementation:cupy-gaussian-blur-v1"
    widget._compute_node_preferences["gaussian"] = NodeComputePreference.parse(
        exact_value
    )

    widget._sync_node_compute_control()

    assert widget.node_compute_preference_combo.currentData() == exact_value
    assert widget.node_compute_preference_combo.currentText().startswith(
        "Advanced pin · CuPy"
    )

    follow_index = widget.node_compute_preference_combo.findData("auto")
    widget.node_compute_preference_combo.setCurrentIndex(follow_index)

    assert "gaussian" not in widget._compute_node_preferences
    assert widget.node_compute_preference_combo.currentText() == "Auto for this node"
    assert all(
        not str(widget.node_compute_preference_combo.itemData(index)).startswith(
            "implementation:"
        )
        for index in range(widget.node_compute_preference_combo.count())
    )

    widget.undo()

    assert widget._compute_node_preferences["gaussian"] == (
        NodeComputePreference.parse(exact_value)
    )
    assert widget.node_compute_preference_combo.currentData() == exact_value
    assert widget.node_compute_preference_combo.currentText().startswith(
        "Advanced pin · CuPy"
    )

    widget.redo()

    assert "gaussian" not in widget._compute_node_preferences
    assert widget.node_compute_preference_combo.currentData() == "auto"
    assert widget.node_compute_preference_combo.currentText() == "Auto for this node"
    assert all(
        not str(widget.node_compute_preference_combo.itemData(index)).startswith(
            "implementation:"
        )
        for index in range(widget.node_compute_preference_combo.count())
    )


def test_accepted_gpu_report_updates_node_badge_and_toolbar_summary(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget._reset_compute_decisions()
    request = ComputeRequest(mode="auto")
    decision = NodeExecutionDecision(
        "gaussian",
        "gaussian_blur",
        NodeComputePreference(),
        "cuda-cupy",
        "cupy",
        "cupy-gaussian-blur-v1",
        DecisionKind.SELECTED,
        DecisionReason.SELECTED_IMPLEMENTATION,
        "Validated CuPy implementation selected.",
    )
    cpu_decision = NodeExecutionDecision(
        "threshold",
        widget.pipeline.nodes["threshold"].operation_id,
        NodeComputePreference(),
        "cpu-numpy",
        "cpu",
        f"cpu-{widget.pipeline.nodes['threshold'].operation_id}-v1",
        DecisionKind.POLICY_CPU,
        DecisionReason.AUTO_CPU,
        "CPU selected for this node.",
    )
    gpu_environment = ComputeEnvironment(
        runtime_ids=("cpu-numpy", "cuda-cupy"),
        implementation_libraries=("cpu", "cupy"),
        device_id="cuda:0",
        device_name="Test GPU",
        device_class="nvidia-cuda",
        memory_topology="discrete",
    )
    plan = ExecutionPlan(
        request.fingerprint,
        gpu_environment.fingerprint,
        (),
        (decision, cpu_decision),
    )
    report = ExecutionReport(
        request,
        gpu_environment,
        plan=plan,
        actual_decisions=(decision, cpu_decision),
    )

    widget._accept_execution_report(report)

    badge = widget.graph_view._cards["gaussian"].compute_badge
    assert badge.text() == "GPU · CuPy"
    assert "Test GPU" in badge.toolTip()
    assert widget.compute_status_label.text() == "Auto · 1 GPU / 1 CPU"
    assert "Test GPU" in widget.compute_status_label.toolTip()

    widget._record_synchronous_cpu_decisions({"threshold"}, request)

    assert widget._last_execution_report is not None
    assert widget._last_execution_report.environment.device_name == "Host CPU"
    assert widget._last_execution_report.plan is None
    assert widget._last_execution_report.actual_decisions == (
        widget._accepted_compute_decisions["threshold"],
    )
    assert "Test GPU" in widget.graph_view._cards["gaussian"].compute_badge.toolTip()

    cpu_decision = widget._accepted_compute_decisions["threshold"]
    cpu_environment = ComputeEnvironment(device_name="Host CPU")
    cpu_plan = ExecutionPlan(
        request.fingerprint,
        cpu_environment.fingerprint,
        (),
        (cpu_decision,),
    )
    cpu_report = ExecutionReport(
        request,
        cpu_environment,
        plan=cpu_plan,
        actual_decisions=(cpu_decision,),
    )
    widget._accept_execution_report(cpu_report)

    assert widget._last_execution_report is cpu_report
    assert widget._last_execution_report.environment.device_name == "Host CPU"
    assert (
        widget._last_execution_report.environment.fingerprint
        == widget._last_execution_report.plan.environment_fingerprint
    )
    gaussian_tooltip = widget.graph_view._cards["gaussian"].compute_badge.toolTip()
    assert "Host CPU" not in gaussian_tooltip


def test_dtype_repair_report_offers_one_click_atomic_visible_conversion(qtbot):
    widget = VippWidget(_Viewer(np.arange(64, dtype=np.uint16).reshape(8, 8)))
    qtbot.addWidget(widget)
    widget.run_pipeline = lambda *args, **kwargs: None
    widget._reset_compute_decisions()
    request = ComputeRequest(mode="prefer_gpu")
    cpu_decision = NodeExecutionDecision(
        "gaussian",
        "gaussian_blur",
        NodeComputePreference(),
        "cpu-numpy",
        "cpu",
        "cpu-gaussian-blur-v1",
        DecisionKind.POLICY_CPU,
        DecisionReason.WORKLOAD_UNSUPPORTED,
        "The integer input is outside the admitted GPU region.",
    )
    suggestion = ComputeRepairSuggestion(
        ComputeRepairAction.INSERT_CONVERT_DTYPE,
        "gaussian",
        "gaussian_blur",
        0,
        "image",
        "uint16",
        "float32",
        "preserve",
        True,
        (
            "This node could use GPU if its uint16 input is converted to float32. "
            "Pixel values will be preserved exactly; this image will use twice "
            "the memory."
        ),
        ComputeRepairCandidate(
            "cupy-gaussian-blur-v1",
            "1",
            "cuda-cupy",
            "cupy",
        ),
    )
    environment = ComputeEnvironment(
        runtime_ids=("cpu-numpy", "cuda-cupy"),
        implementation_libraries=("cpu", "cupy"),
        device_id="cuda:0",
        device_name="Test GPU",
        device_class="nvidia-cuda",
    )
    report = ExecutionReport(
        request,
        environment,
        plan=ExecutionPlan(
            request.fingerprint,
            environment.fingerprint,
            (),
            (cpu_decision,),
            repair_suggestions=(suggestion,),
        ),
        actual_decisions=(cpu_decision,),
    )

    widget._accept_execution_report(report)
    widget.graph_view.select_node("gaussian")

    assert not widget.compute_repair_panel.isHidden()
    assert widget.add_compute_conversion_button.text() == "Add conversion"
    assert "twice the memory" in widget.compute_repair_label.text()
    assert not widget.graph_view._cards["gaussian"].optimization_badge.isHidden()
    before_nodes = set(widget.pipeline.nodes)
    before_connections = tuple(widget.pipeline.connections)

    widget.add_compute_conversion_button.click()

    added = set(widget.pipeline.nodes) - before_nodes
    assert len(added) == 1
    conversion_id = added.pop()
    conversion = widget.pipeline.nodes[conversion_id]
    assert conversion.operation_id == "convert_dtype"
    assert conversion.params["output_dtype"] == "float32"
    assert conversion.params["scaling"] == "preserve"
    assert any(
        connection.source_id == "input" and connection.target_id == conversion_id
        for connection in widget.pipeline.connections
    )
    assert any(
        connection.source_id == conversion_id
        and connection.target_id == "gaussian"
        and connection.target_port == 0
        for connection in widget.pipeline.connections
    )
    assert not any(
        connection.source_id == "input" and connection.target_id == "gaussian"
        for connection in widget.pipeline.connections
    )
    assert "ready for Find fastest" in widget.status_label.text()

    widget.undo()

    assert set(widget.pipeline.nodes) == before_nodes
    assert tuple(widget.pipeline.connections) == before_connections


def test_prefer_gpu_manual_rl_repair_tips_survive_dirty_and_cpu_report(qtbot):
    image = np.ones((9, 48, 56), dtype=np.uint16)
    psf = np.ones((3, 5, 5), dtype=np.float32)
    psf /= np.sum(psf, dtype=np.float32)
    widget = VippWidget(
        _Viewer(image),
        defer_initial_run=True,
        initial_compute_mode="prefer_gpu",
    )
    qtbot.addWidget(widget)
    widget.run_pipeline = lambda *args, **kwargs: None

    def seed_output(node_id, value):
        state = image_state_from_array(value)
        widget.pipeline.outputs[node_id] = value
        widget.pipeline.output_states[node_id] = state
        widget.pipeline.node_outputs[node_id] = [value]
        widget.pipeline.node_output_states[node_id] = [state]
        widget.pipeline.completed_node_ids.add(node_id)

    seed_output("input", image)
    psf_source = widget.pipeline.add_node("input")
    seed_output(psf_source.id, psf)
    rl = widget.pipeline.add_node("richardson_lucy_deconvolution")
    rl_tv = widget.pipeline.add_node("richardson_lucy_tv_deconvolution")
    deconvolution_nodes = (rl, rl_tv)
    for node in deconvolution_nodes:
        widget.pipeline.set_param(node.id, "spatial_mode", "3D ZYX")
        widget.pipeline.set_param(node.id, "iterations", 25)
        widget.pipeline.set_param(node.id, "filter_epsilon", 1e-12)
        assert widget.pipeline.connect("input", node.id, target_port=0).success
        assert widget.pipeline.connect(psf_source.id, node.id, target_port=1).success
    widget.graph_view.build_graph(
        widget.pipeline.nodes.values(),
        widget.pipeline.connections,
    )
    widget.graph_view.select_node(rl.id)

    def assert_repair_tips_visible():
        for node in deconvolution_nodes:
            badge = widget.graph_view._cards[node.id].optimization_badge
            assert not badge.isHidden()
            assert badge.text() == "GPU tip"
            assert "available but has not been applied" in badge.toolTip()
        assert not widget.compute_repair_panel.isHidden()
        assert widget.add_compute_conversion_button.isEnabled()

    widget._sync_all_compute_repair_hints()
    widget._sync_selected_compute_repair()
    assert_repair_tips_visible()

    # Graph and compute-policy invalidation used to clear both card tips before
    # the selected-node inspector lazily rediscovered the exact same repairs.
    widget._mark_pipeline_branches_dirty({rl.id, rl_tv.id})
    assert_repair_tips_visible()

    request = ComputeRequest(mode="prefer_gpu")
    environment = ComputeEnvironment(
        runtime_ids=("cpu-numpy", "cuda-cupy"),
        implementation_libraries=("cpu", "cupy"),
        device_id="cuda:0",
        device_name="Test GPU",
        device_class="nvidia-cuda",
    )
    cpu_decisions = tuple(
        NodeExecutionDecision(
            node.id,
            node.operation_id,
            NodeComputePreference(),
            "cpu-numpy",
            "cpu",
            f"cpu-{node.operation_id}-v1",
            DecisionKind.FALLBACK_CPU,
            DecisionReason.VISIBLE_FALLBACK,
            "The integer input used visible CPU fallback.",
            fallback_reason=FallbackReason.WORKLOAD_UNSUPPORTED,
        )
        for node in deconvolution_nodes
    )
    widget._accept_execution_report(
        ExecutionReport(
            request,
            environment,
            plan=ExecutionPlan(
                request.fingerprint,
                environment.fingerprint,
                (),
                cpu_decisions,
                repair_suggestions=(),
            ),
            actual_decisions=cpu_decisions,
        )
    )

    # A CPU result and unapplied GPU optimization advice describe different
    # facts, so both must remain visible after Calculate.
    for node in deconvolution_nodes:
        assert "CPU" in widget.graph_view._cards[node.id].compute_badge.text()
    assert_repair_tips_visible()

    conversion = widget.pipeline.add_node("convert_dtype")
    widget.pipeline.set_param(conversion.id, "output_dtype", "float32")
    widget.pipeline.set_param(conversion.id, "scaling", "preserve")
    converted = image.astype(np.float32)
    seed_output(conversion.id, converted)
    assert widget.pipeline.connect("input", conversion.id).success
    for node in deconvolution_nodes:
        assert widget.pipeline.disconnect("input", node.id, 0)
        assert widget.pipeline.connect(conversion.id, node.id, target_port=0).success
    widget.graph_view.build_graph(
        widget.pipeline.nodes.values(),
        widget.pipeline.connections,
    )
    widget.graph_view.select_node(rl.id)
    widget._sync_all_compute_repair_hints()
    widget._sync_selected_compute_repair()

    assert all(
        widget.graph_view._cards[node.id].optimization_badge.isHidden()
        for node in deconvolution_nodes
    )
    assert widget.compute_repair_panel.isHidden()

    # Re-expose the uint16 input, then prove an explicit Custom CPU choice also
    # makes the GPU advice inapplicable rather than merely hiding stale paint.
    for node in deconvolution_nodes:
        assert widget.pipeline.disconnect(conversion.id, node.id, 0)
        assert widget.pipeline.connect("input", node.id, target_port=0).success
    widget._compute_mode = ComputeMode.PREFER_GPU
    widget._sync_all_compute_repair_hints()
    widget._sync_selected_compute_repair()
    assert_repair_tips_visible()
    widget._compute_mode = ComputeMode.CUSTOM
    widget._compute_node_preferences.update(
        {node.id: NodeComputePreference("cpu") for node in deconvolution_nodes}
    )
    widget._sync_all_compute_repair_hints()
    widget._sync_selected_compute_repair()

    assert all(
        widget.graph_view._cards[node.id].optimization_badge.isHidden()
        for node in deconvolution_nodes
    )
    assert widget.compute_repair_panel.isHidden()


def test_dtype_repair_inserts_one_shared_conversion_for_sibling_branches(qtbot):
    widget = VippWidget(_Viewer(np.arange(64, dtype=np.uint16).reshape(8, 8)))
    qtbot.addWidget(widget)
    widget.run_pipeline = lambda *args, **kwargs: None
    sibling = widget.add_node_from_palette("gaussian_blur")
    widget._connect_nodes("input", sibling.id)
    widget.graph_view.select_node("gaussian")
    authored = {
        node_id: dict(widget.pipeline.nodes[node_id].params)
        for node_id in ("gaussian", sibling.id)
    }

    def suggestion(node_id: str) -> ComputeRepairSuggestion:
        return ComputeRepairSuggestion(
            "insert_convert_dtype",
            node_id,
            "gaussian_blur",
            0,
            "image",
            "uint16",
            "float32",
            "preserve",
            True,
            "One exact conversion can make this node checkable on GPU.",
            ComputeRepairCandidate(
                "cupy-gaussian-blur-v1",
                "1",
                "cuda-cupy",
                "cupy",
            ),
        )

    widget._compute_repair_suggestions.update(
        {
            "gaussian": suggestion("gaussian"),
            sibling.id: suggestion(sibling.id),
        }
    )
    widget._sync_all_compute_repair_hints()
    widget._sync_selected_compute_repair()
    before_nodes = set(widget.pipeline.nodes)
    before_connections = tuple(widget.pipeline.connections)

    assert "One visible Convert Dtype" in widget.compute_repair_label.text()
    assert "both branches" in widget.compute_repair_label.text()
    widget.add_compute_conversion_button.click()

    added = set(widget.pipeline.nodes) - before_nodes
    assert len(added) == 1
    conversion_id = added.pop()
    assert widget.pipeline.nodes[conversion_id].operation_id == "convert_dtype"
    assert widget.pipeline.nodes[conversion_id].params == {
        "output_dtype": "float32",
        "scaling": "preserve",
    }
    assert (
        sum(
            connection.source_id == "input" and connection.target_id == conversion_id
            for connection in widget.pipeline.connections
        )
        == 1
    )
    assert {
        connection.target_id
        for connection in widget.pipeline.connections
        if connection.source_id == conversion_id
    } == {"gaussian", sibling.id}
    assert not any(
        connection.source_id == "input"
        and connection.target_id in {"gaussian", sibling.id}
        for connection in widget.pipeline.connections
    )
    assert {
        node_id: dict(widget.pipeline.nodes[node_id].params)
        for node_id in ("gaussian", sibling.id)
    } == authored
    assert "2 compatible branches" in widget.status_label.text()
    assert "one step" in widget.status_label.text()

    widget.undo()

    assert set(widget.pipeline.nodes) == before_nodes
    assert tuple(widget.pipeline.connections) == before_connections


def test_optimizer_dtype_refusal_keeps_one_click_repair_actionable(qtbot):
    widget = VippWidget(_Viewer(np.arange(64, dtype=np.uint16).reshape(8, 8)))
    qtbot.addWidget(widget)
    widget.run_pipeline = lambda *args, **kwargs: None
    widget._pipeline_run_pending = False
    widget._compute_repair_suggestions["gaussian"] = ComputeRepairSuggestion(
        "insert_convert_dtype",
        "gaussian",
        "gaussian_blur",
        0,
        "image",
        "uint16",
        "float32",
        "preserve",
        True,
        "A safe exact conversion is available.",
        ComputeRepairCandidate(
            "cupy-gaussian-blur-v1",
            "1",
            "cuda-cupy",
            "cupy",
        ),
    )
    widget.graph_view.select_node("gaussian")

    widget._on_pipeline_optimizer_finished(
        PipelineOptimizerWorkerOutcome(
            error="generic candidate wall that should not replace the action",
            reason_code="evidence_incomplete",
        )
    )

    assert "one fixable input issue" in widget.status_label.text()
    assert "Add conversion" in widget.status_label.text()
    assert "epsilon" in widget.status_label.text()
    assert "generic candidate wall" not in widget.status_label.text()
    assert not widget.compute_repair_panel.isHidden()
    assert widget.add_compute_conversion_button.isEnabled()


def test_optimizer_apply_without_result_is_a_safe_noop(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    before = dict(widget._compute_node_preferences)

    widget._apply_pipeline_optimizer_result(None)

    assert widget._compute_node_preferences == before
    assert "no completed Find fastest result" in widget.status_label.text()


def test_dtype_repair_on_tunnel_input_changes_only_that_subscriber(qtbot):
    widget = VippWidget(_Viewer(np.arange(64, dtype=np.uint16).reshape(8, 8)))
    qtbot.addWidget(widget)
    widget.run_pipeline = lambda *args, **kwargs: None
    assert widget.pipeline.add_output_tunnel("Raw", "input", 0).name == "Raw"
    gaussian_connection = next(
        connection
        for connection in widget.pipeline.connections
        if connection.target_id == "gaussian"
    )
    assert widget.pipeline.disconnect(
        gaussian_connection.source_id,
        gaussian_connection.target_id,
        gaussian_connection.target_port,
    )
    assert widget.pipeline.connect_to_tunnel("Raw", "gaussian", 0).success
    other = widget.pipeline.add_node("invert")
    assert widget.pipeline.connect_to_tunnel("Raw", other.id, 0).success
    widget.graph_view.build_graph(
        widget.pipeline.nodes.values(),
        widget.pipeline.connections,
        output_tunnels=widget.pipeline.output_tunnel_list(),
    )
    widget._sync_all_input_ports()
    widget._sync_all_output_ports()
    target_before = QPointF(widget.graph_view.node_position("gaussian"))
    other_before = QPointF(widget.graph_view.node_position(other.id))
    before_connections = tuple(widget.pipeline.connections)
    widget._compute_repair_suggestions["gaussian"] = ComputeRepairSuggestion(
        "insert_convert_dtype",
        "gaussian",
        "gaussian_blur",
        0,
        "image",
        "uint16",
        "float32",
        "preserve",
        True,
        "Convert only this tunnel subscriber while preserving exact values.",
        ComputeRepairCandidate(
            "cupy-gaussian-blur-v1",
            "1",
            "cuda-cupy",
            "cupy",
        ),
    )
    widget.graph_view.select_node("gaussian")

    widget.add_compute_conversion_button.click()

    conversion_id = next(
        node_id
        for node_id, node in widget.pipeline.nodes.items()
        if node.operation_id == "convert_dtype"
    )
    assert widget.pipeline.tunnel_connection_for_input(conversion_id, 0) is not None
    assert widget.pipeline.tunnel_connection_for_input("gaussian", 0) is None
    other_tunnel = widget.pipeline.tunnel_connection_for_input(other.id, 0)
    assert other_tunnel is not None and other_tunnel.tunnel_name == "Raw"
    assert widget.pipeline.output_tunnel("Raw").source_id == "input"
    assert widget.graph_view.node_position("gaussian") == target_before
    assert widget.graph_view.node_position(other.id) == other_before
    conversion_rect = widget.graph_view.node_scene_rect(conversion_id)
    target_rect = widget.graph_view.node_scene_rect("gaussian")
    assert conversion_rect is not None and target_rect is not None
    assert target_rect.left() - conversion_rect.right() >= widget.INSERT_GAP_PADDING_X

    widget.undo()

    assert tuple(widget.pipeline.connections) == before_connections
    assert widget.pipeline.output_tunnel("Raw").source_id == "input"
    assert widget.graph_view.node_position("gaussian") == target_before
    assert widget.graph_view.node_position(other.id) == other_before


def test_dtype_repair_is_hidden_for_custom_cpu_intent(qtbot):
    widget = VippWidget(_Viewer(np.arange(64, dtype=np.uint16).reshape(8, 8)))
    qtbot.addWidget(widget)
    widget.run_pipeline = lambda *args, **kwargs: None
    widget._compute_repair_suggestions["gaussian"] = ComputeRepairSuggestion(
        "insert_convert_dtype",
        "gaussian",
        "gaussian_blur",
        0,
        "image",
        "uint16",
        "float32",
        "preserve",
        True,
        "A safe GPU eligibility improvement is available.",
        ComputeRepairCandidate(
            "cupy-gaussian-blur-v1",
            "1",
            "cuda-cupy",
            "cupy",
        ),
    )
    widget._compute_node_preferences["gaussian"] = NodeComputePreference("cpu")
    widget.graph_view.select_node("gaussian")
    widget._sync_all_compute_repair_hints()
    assert not widget.compute_repair_panel.isHidden()

    widget.compute_mode_combo.setCurrentIndex(
        widget.compute_mode_combo.findData("custom")
    )

    assert widget.compute_repair_panel.isHidden()
    assert widget.graph_view._cards["gaussian"].optimization_badge.isHidden()


def test_dtype_repair_ui_rejects_unknown_candidate_identity(qtbot, monkeypatch):
    widget = VippWidget(_Viewer(np.arange(64, dtype=np.uint16).reshape(8, 8)))
    qtbot.addWidget(widget)
    widget.run_pipeline = lambda *args, **kwargs: None
    monkeypatch.setattr(widget, "_refresh_proactive_compute_repairs", lambda: ())
    widget._compute_repair_suggestions["gaussian"] = ComputeRepairSuggestion(
        "insert_convert_dtype",
        "gaussian",
        "gaussian_blur",
        0,
        "image",
        "uint16",
        "float32",
        "preserve",
        True,
        "A forged candidate should never be applied.",
        ComputeRepairCandidate("unknown-provider", "1", "cuda-cupy", "cupyx"),
    )
    widget.graph_view.select_node("gaussian")
    widget._sync_all_compute_repair_hints()

    assert widget.compute_repair_panel.isHidden()
    assert not widget.add_compute_conversion_button.isEnabled()
    assert widget.graph_view._cards["gaussian"].optimization_badge.isHidden()


def test_stale_compute_summary_distinguishes_active_update_from_previous_result(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)

    widget._mark_compute_badges_stale({"gaussian"})

    assert widget.compute_status_label.text().endswith("· previous")

    widget._active_pipeline_run_id = 99
    widget._sync_compute_toolbar_summary()

    assert widget.compute_status_label.text() == "Auto · updating"
    widget._active_pipeline_run_id = None


def test_loading_legacy_workflow_stays_cpu_until_user_opts_in(qtbot, tmp_path):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    path = tmp_path / "legacy-v3-workflow.json"
    legacy_document = serialize_workflow(widget.pipeline, {})
    legacy_document["version"] = 3
    legacy_document.pop("execution")
    path.write_text(json.dumps(legacy_document), encoding="utf-8")

    assert widget.compute_mode_combo.currentData() == "auto"
    widget._compute_mode = ComputeMode.CUSTOM
    widget._compute_fallback_policy = FallbackPolicy.STRICT
    widget._compute_node_preferences["gaussian"] = NodeComputePreference(
        "library",
        "cupyx",
    )
    widget.compute_mode_combo.blockSignals(True)
    widget.compute_mode_combo.setCurrentIndex(
        widget.compute_mode_combo.findData("custom")
    )
    widget.compute_mode_combo.blockSignals(False)
    widget.strict_compute_checkbox.blockSignals(True)
    widget.strict_compute_checkbox.setChecked(True)
    widget.strict_compute_checkbox.blockSignals(False)
    original_session = widget._workflow_tabs.current
    assert original_session is not None

    widget.load_workflow_file(path)

    assert widget.compute_mode_combo.currentData() == "cpu"
    assert widget._compute_mode is ComputeMode.CPU
    assert widget.compute_status_label.text().startswith("CPU")
    assert not widget._history.can_undo

    widget.undo()

    assert widget.compute_mode_combo.currentData() == "cpu"
    widget._activate_workflow_tab(
        widget._workflow_tabs.index_of(original_session.session_id)
    )

    assert widget.compute_mode_combo.currentData() == "custom"
    assert widget._compute_fallback_policy is FallbackPolicy.STRICT
    assert widget.strict_compute_checkbox.isChecked()
    assert widget._compute_node_preferences["gaussian"] == NodeComputePreference(
        "library",
        "cupyx",
    )


def test_loading_v4_workflow_restores_portable_compute_intent(qtbot, tmp_path):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget.run_pipeline = lambda *args, **kwargs: None
    path = tmp_path / "custom-v4-workflow.json"
    preference = NodeComputePreference(
        "implementation",
        "future-provider.gaussian-v2",
    )
    save_workflow(
        path,
        widget.pipeline,
        {},
        metadata={"vipp": {"compute_optimizer": {"locked_node_ids": ["gaussian"]}}},
        compute_request=ComputeRequest(
            mode="custom",
            fallback_policy="strict",
            node_preferences={"gaussian": preference},
            runtime_id="machine-local-runtime",
            device_id="machine-local-device",
            precision_policy_id="verified-float32-v2",
            workload_policy_id="interactive-volume-v2",
            allow_experimental=True,
        ),
    )

    widget.load_workflow_file(path)

    assert widget.compute_mode_combo.currentData() == "custom"
    assert widget.strict_compute_checkbox.isChecked()
    assert widget._compute_node_preferences == {"gaussian": preference}
    assert widget._compute_optimizer_locked_node_ids == {"gaussian"}
    restored_request = widget._current_compute_request()
    assert restored_request.precision_policy_id == "verified-float32-v2"
    assert restored_request.workload_policy_id == "interactive-volume-v2"
    assert restored_request.runtime_id == ""
    assert restored_request.device_id == ""
    assert restored_request.allow_experimental is False


def test_loading_v4_workflow_restores_prefer_gpu_as_global_policy(
    qtbot,
    tmp_path,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget.run_pipeline = lambda *args, **kwargs: None
    path = tmp_path / "prefer-gpu-v4-workflow.json"
    save_workflow(
        path,
        widget.pipeline,
        {},
        compute_request=ComputeRequest(mode="prefer_gpu"),
    )

    widget.load_workflow_file(path)

    assert widget.compute_mode_combo.currentData() == "prefer_gpu"
    assert widget._compute_mode is ComputeMode.PREFER_GPU
    assert widget._current_compute_request() == ComputeRequest(mode="prefer_gpu")
    assert widget.compute_status_label.text().startswith("Prefer GPU")
    assert widget.compute_group.isHidden()
    assert widget.optimize_pipeline_button.isHidden()


def test_explicit_compute_device_is_session_only_and_dormant_under_cpu(qtbot):
    widget = VippWidget(
        _Viewer(),
        defer_initial_run=True,
        initial_compute_mode=ComputeMode.PREFER_GPU,
        initial_compute_runtime_id="cuda-cupy",
        initial_compute_device_id="cuda:1",
        initial_compute_device_display_name="Second RTX",
    )
    qtbot.addWidget(widget)

    request = widget._current_compute_request()
    assert request.runtime_id == "cuda-cupy"
    assert request.device_id == "cuda:1"
    document = serialize_workflow(widget.pipeline, compute_request=request)
    assert "runtime_id" not in document["execution"]["compute"]
    assert "device_id" not in document["execution"]["compute"]

    widget._compute_mode = ComputeMode.CPU
    cpu_request = widget._current_compute_request()
    assert cpu_request.runtime_id == ""
    assert cpu_request.device_id == ""
    assert widget._compute_runtime_id == "cuda-cupy"
    assert widget._compute_device_id == "cuda:1"
    assert widget._compute_device_display_name == "Second RTX"


def test_compute_device_change_recalculates_without_dirtying_workflow(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(
        _Viewer(),
        defer_initial_run=True,
        initial_compute_mode=ComputeMode.PREFER_GPU,
    )
    qtbot.addWidget(widget)
    session = widget._workflow_tabs.current
    assert session is not None
    session.mark_clean(
        widget._current_history_snapshot(),
        persistence_token=widget._workflow_tab_persistence_token(),
    )
    undo_before = widget._history.can_undo
    calls: list[str] = []
    monkeypatch.setattr(
        widget._thumbnail_statistics_engine,
        "reset_accelerator_capability",
        lambda: calls.append("reset-thumbnail-provider"),
    )
    monkeypatch.setattr(
        widget,
        "_clear_thumbnail_contrast_limit_state",
        lambda: calls.append("clear-thumbnail-state"),
    )
    monkeypatch.setattr(
        widget,
        "_invalidate_compute_policy_results",
        lambda: calls.append("invalidate-compute"),
    )
    monkeypatch.setattr(widget, "run_pipeline", lambda: calls.append("run"))

    widget._on_compute_device_changed(
        ComputeDeviceOption("cuda-cupy", "cuda:1", "Second RTX")
    )

    assert widget._current_compute_request().device_id == "cuda:1"
    assert calls == [
        "reset-thumbnail-provider",
        "clear-thumbnail-state",
        "invalidate-compute",
        "run",
    ]
    assert widget._history.can_undo is undo_before
    assert not session.dirty


def test_compute_device_setter_can_seed_only_an_uncalculated_session(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(
        _Viewer(),
        defer_initial_run=True,
        initial_compute_mode=ComputeMode.PREFER_GPU,
    )
    qtbot.addWidget(widget)
    runs = []
    monkeypatch.setattr(widget, "run_pipeline", lambda: runs.append(True))

    assert widget.set_compute_device_selection(
        "cuda-cupy",
        "cuda:1",
        "Second RTX",
        recalculate=False,
    )
    assert widget._current_compute_request().device_id == "cuda:1"
    assert runs == []

    widget.pipeline.completed_node_ids.add("input")
    with pytest.raises(RuntimeError, match="cannot change GPU selection"):
        widget.set_compute_device_selection(
            "cuda-cupy",
            "cuda:0",
            "First RTX",
            recalculate=False,
        )
    assert widget._current_compute_request().device_id == "cuda:1"


def test_compute_device_choice_is_retained_without_recalculation_under_cpu(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(
        _Viewer(),
        defer_initial_run=True,
        initial_compute_mode=ComputeMode.CPU,
    )
    qtbot.addWidget(widget)
    calls: list[str] = []
    monkeypatch.setattr(
        widget._thumbnail_statistics_engine,
        "reset_accelerator_capability",
        lambda: calls.append("reset-thumbnail-provider"),
    )
    monkeypatch.setattr(
        widget,
        "_clear_thumbnail_contrast_limit_state",
        lambda: calls.append("clear-thumbnail-state"),
    )
    monkeypatch.setattr(
        widget,
        "_invalidate_compute_policy_results",
        lambda: calls.append("invalidate-compute"),
    )
    monkeypatch.setattr(widget, "run_pipeline", lambda: calls.append("run"))

    widget._on_compute_device_changed(
        ComputeDeviceOption("cuda-cupy", "cuda:1", "Second RTX")
    )

    assert calls == ["reset-thumbnail-provider", "clear-thumbnail-state"]
    assert widget._compute_device_id == "cuda:1"
    assert widget._current_compute_request().device_id == ""
    assert "CPU remains active" in widget.status_label.text()


@pytest.mark.parametrize("blocked_kind", ("busy", "unavailable"))
def test_compute_device_change_rejects_unusable_choice_and_restores_dialog(
    qtbot,
    blocked_kind,
):
    widget = VippWidget(
        _Viewer(),
        defer_initial_run=True,
        initial_compute_mode=ComputeMode.PREFER_GPU,
        initial_compute_runtime_id="cuda-cupy",
        initial_compute_device_id="cuda:0",
        initial_compute_device_display_name="First RTX",
    )
    qtbot.addWidget(widget)
    restored = []
    widget._compute_setup_dialog = SimpleNamespace(
        set_device_selection=lambda *values: restored.append(values),
        set_device_selection_editable=lambda _enabled: None,
    )
    if blocked_kind == "busy":
        widget._active_pipeline_run_id = 71
        option = ComputeDeviceOption("cuda-cupy", "cuda:1", "Second RTX")
    else:
        option = ComputeDeviceOption(
            "cuda-cupy",
            "cuda:1",
            "Missing RTX",
            available=False,
        )

    widget._on_compute_device_changed(option)

    assert widget._compute_device_id == "cuda:0"
    assert restored == [("cuda-cupy", "cuda:0", "First RTX")]
    assert widget._current_compute_request().device_id == "cuda:0"
    widget._active_pipeline_run_id = None
    widget._compute_setup_dialog = None


def test_compute_device_choice_is_independent_per_workflow_tab(qtbot):
    widget = VippWidget(
        _Viewer(),
        defer_initial_run=True,
        initial_compute_mode=ComputeMode.PREFER_GPU,
        initial_compute_runtime_id="cuda-cupy",
        initial_compute_device_id="cuda:0",
        initial_compute_device_display_name="First RTX",
    )
    qtbot.addWidget(widget)
    widget.run_pipeline = lambda *args, **kwargs: None
    first = widget._workflow_tabs.current
    assert first is not None

    widget._new_workflow()
    second = widget._workflow_tabs.current
    assert second is not None and second is not first
    assert widget._compute_device_id == ""
    widget._on_compute_device_changed(
        ComputeDeviceOption("cuda-cupy", "cuda:1", "Second RTX")
    )
    assert widget._compute_device_id == "cuda:1"

    first_index = widget._workflow_tabs.index_of(first.session_id)
    assert widget._activate_workflow_tab(first_index, check_safety=False)
    assert widget._compute_runtime_id == "cuda-cupy"
    assert widget._compute_device_id == "cuda:0"
    assert widget._compute_device_display_name == "First RTX"

    second_index = widget._workflow_tabs.index_of(second.session_id)
    assert widget._activate_workflow_tab(second_index, check_safety=False)
    assert widget._compute_runtime_id == "cuda-cupy"
    assert widget._compute_device_id == "cuda:1"
    assert widget._compute_device_display_name == "Second RTX"


def test_compute_policy_edits_are_directly_undoable_and_redoable(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget.run_pipeline = lambda *args, **kwargs: None

    widget.compute_mode_combo.setCurrentIndex(
        widget.compute_mode_combo.findData("custom")
    )
    assert widget._compute_mode is ComputeMode.CUSTOM
    widget.undo()
    assert widget._compute_mode is ComputeMode.AUTO
    assert widget.compute_mode_combo.currentData() == "auto"
    widget.redo()
    assert widget._compute_mode is ComputeMode.CUSTOM

    widget.strict_compute_checkbox.setChecked(True)
    assert widget._compute_fallback_policy is FallbackPolicy.STRICT
    widget.undo()
    assert widget._compute_fallback_policy is FallbackPolicy.VISIBLE
    assert not widget.strict_compute_checkbox.isChecked()
    widget.redo()
    assert widget._compute_fallback_policy is FallbackPolicy.STRICT

    widget.graph_view.select_node("gaussian")
    cupy_index = widget.node_compute_preference_combo.findData("library:cupy")
    assert cupy_index >= 0
    widget.node_compute_preference_combo.setCurrentIndex(cupy_index)
    assert widget._compute_node_preferences["gaussian"] == NodeComputePreference(
        "library",
        "cupy",
    )


def test_undo_cannot_change_compute_policy_during_active_calculation(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget.run_pipeline = lambda *args, **kwargs: None
    widget.compute_mode_combo.setCurrentIndex(
        widget.compute_mode_combo.findData("custom")
    )
    assert widget._history.can_undo
    widget._active_pipeline_run_id = 123
    widget._sync_compute_policy_editability()

    widget.undo()

    assert widget._compute_mode is ComputeMode.CUSTOM
    assert widget._history.can_undo
    assert not widget.undo_action.isEnabled()
    assert "Cancel the current calculation" in widget.status_label.text()

    widget._active_pipeline_run_id = None
    widget._sync_compute_policy_editability()
    widget.undo()

    assert widget._compute_mode is ComputeMode.AUTO
    assert widget.compute_mode_combo.currentData() == "auto"


def test_undo_cannot_change_compute_policy_after_runtime_quarantine(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget.run_pipeline = lambda *args, **kwargs: None
    widget.compute_mode_combo.setCurrentIndex(
        widget.compute_mode_combo.findData("custom")
    )
    assert widget._history.can_undo
    widget._compute_runtime_quarantined_reason = (
        "CPU/GPU cleanup failed. Restart VIPP before calculating again."
    )
    widget._sync_compute_policy_editability()

    widget.undo()

    assert widget._compute_mode is ComputeMode.CUSTOM
    assert widget._history.can_undo
    assert not widget.undo_action.isEnabled()
    assert "Restart VIPP" in widget.status_label.text()

    widget._compute_runtime_quarantined_reason = ""
    widget._sync_compute_policy_editability()
    assert widget.undo_action.isEnabled()
    widget.undo()
    assert widget._compute_mode is ComputeMode.AUTO


def test_prefer_gpu_policy_is_undoable_and_local_to_each_workflow_tab(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget.run_pipeline = lambda *args, **kwargs: None

    widget.compute_mode_combo.setCurrentIndex(
        widget.compute_mode_combo.findData("prefer_gpu")
    )
    assert widget._compute_mode is ComputeMode.PREFER_GPU

    widget.undo()
    assert widget._compute_mode is ComputeMode.AUTO
    assert widget.compute_mode_combo.currentData() == "auto"
    widget.redo()
    assert widget._compute_mode is ComputeMode.PREFER_GPU
    assert widget.compute_mode_combo.currentData() == "prefer_gpu"

    first = widget._workflow_tabs.current
    assert first is not None
    widget._new_workflow()
    assert widget._workflow_tabs.current is not first
    assert widget._compute_mode is ComputeMode.AUTO
    assert widget.compute_mode_combo.currentData() == "auto"

    first_index = widget._workflow_tabs.index_of(first.session_id)
    widget.workflow_tab_bar.setCurrentIndex(first_index)

    assert widget._workflow_tabs.current is first
    assert widget._compute_mode is ComputeMode.PREFER_GPU
    assert widget.compute_mode_combo.currentData() == "prefer_gpu"


def test_strict_custom_setting_does_not_leak_into_other_modes(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget.run_pipeline = lambda *args, **kwargs: None

    widget.strict_compute_checkbox.setChecked(True)

    assert widget._compute_fallback_policy is FallbackPolicy.STRICT
    assert widget._current_compute_request().fallback_policy is FallbackPolicy.VISIBLE

    widget.compute_mode_combo.setCurrentIndex(
        widget.compute_mode_combo.findData("custom")
    )
    assert widget._current_compute_request().fallback_policy is FallbackPolicy.STRICT

    widget.compute_mode_combo.setCurrentIndex(
        widget.compute_mode_combo.findData("prefer_gpu")
    )
    assert widget._current_compute_request().fallback_policy is FallbackPolicy.VISIBLE

    widget.compute_mode_combo.setCurrentIndex(widget.compute_mode_combo.findData("cpu"))
    assert widget._current_compute_request().fallback_policy is FallbackPolicy.VISIBLE


@pytest.mark.parametrize("mode", (ComputeMode.AUTO, ComputeMode.PREFER_GPU))
def test_synchronous_global_policy_decision_ignores_dormant_preference(
    qtbot,
    mode,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget._abandon_background_pipeline_run()
    widget._compute_mode = mode
    widget._compute_node_preferences["gaussian"] = NodeComputePreference("cpu")
    request = widget._current_compute_request()

    widget._record_synchronous_cpu_decisions({"gaussian"}, request)

    decision = widget._accepted_compute_decisions["gaussian"]
    assert request.preference_for("gaussian").kind.value == "cpu"
    assert decision.requested_preference == NodeComputePreference()
    assert decision.reason is DecisionReason.AUTO_CPU


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
    assert widget._selected_node_id == "input"
    assert widget.graph_view._cards["input"]._selected
    assert widget.status_label.text().startswith("Opened example workflow")


def test_example_workflow_selects_source_before_background_thumbnail_run(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    pool = _QueuedThreadPool()
    widget._pipeline_thread_pool = pool
    widget.cache_mode_combo.setCurrentText(CACHE_MODE_KEEP_ALL)
    widget.background_all_checkbox.setChecked(True)

    widget.load_example_workflow("label-cleanup")

    assert widget.cache_mode_combo.currentText() == CACHE_MODE_KEEP_ALL
    assert widget._selected_node_id == "input"
    assert widget.graph_view._cards["input"]._selected
    assert widget._active_pipeline_run_id is not None
    assert "Processing" in widget.status_label.text()
    assert all(output is None for output in widget.pipeline.outputs.values())

    pool.workers[0].run()
    qtbot.waitUntil(lambda: widget._active_pipeline_run_id is None)
    qtbot.waitUntil(lambda: len(pool.workers) > 1, timeout=5_000)
    pool.workers[1].run()
    qtbot.waitUntil(
        lambda: all(
            widget.graph_view.node_has_thumbnail(node_id)
            for node_id in widget.pipeline.outputs
        ),
        timeout=5_000,
    )

    for node_id, output in widget.pipeline.outputs.items():
        assert output is not None, node_id
        preview = widget.graph_view._cards[node_id].preview
        pixmap = preview.source_pixmap()
        assert preview.has_source_pixmap(), node_id
        assert not pixmap.isNull(), node_id


def test_delete_selected_node_removes_pipeline_node_and_connections(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    widget._compute_node_preferences["gaussian"] = NodeComputePreference(
        "library", "cupyx"
    )
    widget._compute_optimizer_locked_node_ids.add("gaussian")
    widget.graph_view.select_node("gaussian")

    qtbot.keyClick(widget.graph_view, Qt.Key_Delete)

    assert "gaussian" not in widget.pipeline.nodes
    assert "gaussian" not in widget.graph_view._cards
    assert all(
        connection.source_id != "gaussian" and connection.target_id != "gaussian"
        for connection in widget.pipeline.connections
    )
    assert widget._selected_node_id in widget.pipeline.nodes
    assert "gaussian" not in widget._compute_optimizer_locked_node_ids


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
    widget._compute_node_preferences["gaussian"] = NodeComputePreference(
        "library", "cupyx"
    )
    widget._compute_optimizer_locked_node_ids.add("gaussian")
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
    assert widget._compute_node_preferences[clone_id] == NodeComputePreference(
        "library", "cupyx"
    )
    assert clone_id in widget._compute_optimizer_locked_node_ids


def test_graph_fragment_copy_paste_is_atomic_and_keeps_only_internal_edges(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    monkeypatch.setattr(widget, "run_pipeline", lambda *args, **kwargs: None)
    QApplication.clipboard().clear()
    widget._compute_node_preferences["gaussian"] = NodeComputePreference(
        "library", "cupyx"
    )
    widget._compute_optimizer_locked_node_ids.add("gaussian")
    widget._add_graph_note("Copied rationale", attached_node="gaussian")
    widget._history.clear()
    before_ids = set(widget.pipeline.nodes)

    widget._copy_graph_nodes(("gaussian", "threshold"))
    pasted_ids = widget._paste_graph_fragment(QPointF(900.0, 500.0))

    assert len(pasted_ids) == 2
    assert set(widget.pipeline.nodes) - before_ids == set(pasted_ids)
    pasted_gaussian = next(
        node_id
        for node_id in pasted_ids
        if widget.pipeline.nodes[node_id].operation_id == "gaussian_blur"
    )
    pasted_threshold = next(
        node_id
        for node_id in pasted_ids
        if widget.pipeline.nodes[node_id].operation_id == "otsu_threshold"
    )
    assert (
        GraphConnection(pasted_gaussian, pasted_threshold)
        in widget.pipeline.connections
    )
    assert not any(
        connection.source_id == "input" and connection.target_id == pasted_gaussian
        for connection in widget.pipeline.connections
    )
    assert widget._compute_node_preferences[pasted_gaussian] == NodeComputePreference(
        "library", "cupyx"
    )
    assert pasted_gaussian in widget._compute_optimizer_locked_node_ids
    assert any(
        note.attached_node == pasted_gaussian and note.text == "Copied rationale"
        for note in widget._graph_notes.values()
    )
    assert widget.graph_view.selected_node_ids() == pasted_ids
    assert len(widget._undo_stack) == 1

    widget.undo()

    assert set(widget.pipeline.nodes) == before_ids
    assert not any(
        note.attached_node in set(pasted_ids) for note in widget._graph_notes.values()
    )


def test_paste_values_requires_same_operation_and_keeps_target_execution_intent(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    monkeypatch.setattr(widget, "run_pipeline", lambda *args, **kwargs: None)
    QApplication.clipboard().clear()
    target = widget._add_node_at("gaussian_blur", QPointF(800.0, 400.0))
    widget.pipeline.set_param("gaussian", "sigma", 4.0)
    widget.pipeline.set_param(target.id, "sigma", 0.6)
    preference = NodeComputePreference("library", "numpy")
    widget._compute_node_preferences[target.id] = preference
    connections_before = tuple(widget.pipeline.connections)
    position_before = QPointF(widget.graph_view.node_position(target.id))
    widget._history.clear()

    widget._copy_graph_nodes(("gaussian",))
    assert widget._paste_graph_node_values(target.id)

    assert widget.pipeline.nodes[target.id].params["sigma"] == 4.0
    assert widget._compute_node_preferences[target.id] == preference
    assert tuple(widget.pipeline.connections) == connections_before
    assert widget.graph_view.node_position(target.id) == position_before
    assert len(widget._undo_stack) == 1
    widget._debounce_timer.stop()

    widget.undo()

    assert widget.pipeline.nodes[target.id].params["sigma"] == 0.6
    assert widget._compute_node_preferences[target.id] == preference


def test_paste_values_preserves_rescale_representations_and_refreshes_ui(
    qtbot,
    monkeypatch,
):
    viewer = _Viewer(
        np.zeros((12, 96, 128), dtype=np.float32),
        metadata={"axes": "ZYX"},
    )
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    monkeypatch.setattr(widget, "run_pipeline", lambda *args, **kwargs: None)
    QApplication.clipboard().clear()
    source = widget._add_node_at("rescale_axes", QPointF(700.0, 300.0))
    target = widget._add_node_at("rescale_axes", QPointF(1000.0, 300.0))
    widget._connect_nodes("input", target.id)
    copied_values = {
        "resize_mode": "Output size",
        "x_scale": 2.01,
        "y_scale": 2.02,
        "z_scale": 1.31,
        "x_size": 257,
        "y_size": 193,
        "z_size": 17,
        "lock_xy": True,
    }
    for name, value in copied_values.items():
        widget.pipeline.set_param(source.id, name, value)
    widget._copy_graph_nodes((source.id,))
    widget.graph_view.select_node(target.id)
    connections_before = tuple(widget.pipeline.connections)
    widget._history.clear()

    assert widget._paste_graph_node_values(target.id)

    for name, value in copied_values.items():
        assert widget.pipeline.nodes[target.id].params[name] == value
    assert tuple(widget.pipeline.connections) == connections_before
    assert widget._parameter_widgets["resize_mode"].combo.currentData() == (
        "Output size"
    )
    assert widget._parameter_widgets["x_size"].value() == 257
    assert widget._parameter_widgets["y_size"].value() == 193
    assert widget._parameter_widgets["z_size"].value() == 17
    assert len(widget._undo_stack) == 1
    widget._debounce_timer.stop()


def test_paste_values_refreshes_input_histogram_and_schedules_once(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    monkeypatch.setattr(widget, "run_pipeline", lambda *args, **kwargs: None)
    QApplication.clipboard().clear()
    source = widget._add_node_at("binary_threshold", QPointF(700.0, 300.0))
    target = widget._add_node_at("binary_threshold", QPointF(1000.0, 300.0))
    widget.pipeline.set_param(source.id, "threshold", 0.77)
    widget._copy_graph_nodes((source.id,))
    widget.graph_view.select_node(target.id)
    histogram_calls = []
    monkeypatch.setattr(
        widget,
        "_update_rescale_input_histogram",
        lambda *args: histogram_calls.append(args),
    )

    class _TimerSpy:
        def __init__(self, timer):
            self._timer = timer
            self.starts = 0

        def __getattr__(self, name):
            return getattr(self._timer, name)

        def start(self, *args):
            self.starts += 1
            return self._timer.start(*args)

    timer = _TimerSpy(widget._debounce_timer)
    widget._debounce_timer = timer
    widget._history.clear()

    assert widget._paste_graph_node_values(target.id)

    assert widget.pipeline.nodes[target.id].params["threshold"] == 0.77
    assert len(histogram_calls) == 1
    assert timer.starts == 1
    assert len(widget._undo_stack) == 1
    timer.stop()


def test_paste_values_rolls_back_if_live_ui_commit_fails(qtbot, monkeypatch):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    monkeypatch.setattr(widget, "run_pipeline", lambda *args, **kwargs: None)
    QApplication.clipboard().clear()
    target = widget._add_node_at("gaussian_blur", QPointF(800.0, 400.0))
    widget.pipeline.set_param("gaussian", "sigma", 4.0)
    widget.pipeline.set_param(target.id, "sigma", 0.6)
    widget._copy_graph_nodes(("gaussian",))
    before_params = deepcopy(widget.pipeline.nodes[target.id].params)
    before_counters = deepcopy(widget.pipeline._counters)
    widget._history.clear()

    sync_count = 0

    def fail_first_sync(_node_id):
        nonlocal sync_count
        sync_count += 1
        if sync_count == 1:
            raise RuntimeError("injected output-port presentation failure")

    monkeypatch.setattr(widget, "_sync_node_output_ports", fail_first_sync)

    assert not widget._paste_graph_node_values(target.id)

    assert widget.pipeline.nodes[target.id].params == before_params
    assert widget.pipeline._counters == before_counters
    assert len(widget._undo_stack) == 0


def test_failed_graph_paste_restores_monotonic_node_counters(qtbot, monkeypatch):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    QApplication.clipboard().clear()
    widget._copy_graph_nodes(("gaussian",))
    widget.pipeline._counters["gaussian_blur"] = 9
    before_ids = set(widget.pipeline.nodes)
    before_counters = deepcopy(widget.pipeline._counters)
    run_count = 0

    def fail_first_run(*_args, **_kwargs):
        nonlocal run_count
        run_count += 1
        if run_count == 1:
            raise RuntimeError("injected calculation handoff failure")

    monkeypatch.setattr(widget, "run_pipeline", fail_first_run)

    assert widget._paste_graph_fragment(QPointF(900.0, 500.0)) == ()

    assert set(widget.pipeline.nodes) == before_ids
    assert widget.pipeline._counters == before_counters


def test_fragment_paste_note_ids_are_case_insensitively_unique(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    widget._graph_notes = {"NOTE_1": GraphNoteState("NOTE_1", "Existing", (0.0, 0.0))}
    fragment = GraphFragment(
        (
            GraphFragmentNode(
                "n0",
                "gaussian_blur",
                {"sigma": 1.2, "channel_axis": -1},
            ),
        ),
        notes=(
            GraphFragmentNote(
                "note0",
                "Copied rationale",
                (20.0, 30.0),
                240.0,
                "n0",
            ),
        ),
    )

    plan = widget._prepare_graph_fragment_paste(fragment, QPointF(0.0, 0.0))

    assert plan[5][0].id == "note_2"


def test_fragment_paste_remaps_colliding_tunnel_name_and_fresh_node_ids(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    widget.pipeline.add_output_tunnel("Shared", "input")
    source = PrototypePipeline()
    source.reset_empty_graph()
    blur = source.add_node("gaussian_blur")
    threshold = source.add_node("otsu_threshold")
    source.add_output_tunnel("Shared", blur.id)
    assert source.connect_to_tunnel("Shared", threshold.id).success
    fragment = capture_graph_fragment(
        source,
        (blur.id, threshold.id),
        positions={blur.id: (0.0, 0.0), threshold.id: (300.0, 0.0)},
    )

    plan = widget._prepare_graph_fragment_paste(fragment, QPointF(500.0, 300.0))
    staged, node_ids, _positions, connections, tunnels = plan[:5]

    assert set(node_ids).isdisjoint(widget.pipeline.nodes)
    assert [tunnel.name for tunnel in tunnels] == ["Shared copy"]
    assert connections[0].tunnel_name == "Shared copy"
    assert staged.output_tunnel("Shared") is not None
    assert staged.output_tunnel("Shared copy") is not None


def test_insert_before_tunnel_keeps_subscribers_and_direct_wires_in_one_undo(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    assert widget.load_example_workflow("graph-authoring") is not None
    monkeypatch.setattr(widget, "run_pipeline", lambda *args, **kwargs: None)
    widget._history.clear()
    before_connections = tuple(widget.pipeline.connections)
    before_tunnels = widget.pipeline.output_tunnel_list()
    before_ids = set(widget.pipeline.nodes)

    inserted = widget._insert_new_node_before_tunnel(
        "invert",
        "Shared processed image",
        QPointF(620.0, 180.0),
    )

    assert inserted is not None
    tunnel = widget.pipeline.output_tunnel("Shared processed image")
    assert tunnel is not None
    assert tunnel.source_id == inserted.id
    subscribers = [
        connection
        for connection in widget.pipeline.connections
        if connection.tunnel_name == tunnel.name
    ]
    assert {connection.target_id for connection in subscribers} == {
        "threshold_low",
        "threshold_high",
    }
    assert GraphConnection("gaussian_main", inserted.id) in widget.pipeline.connections
    assert (
        GraphConnection("gaussian_main", "rescale_direct")
        in widget.pipeline.connections
    )
    assert len(widget._undo_stack) == 1

    widget.undo()

    assert set(widget.pipeline.nodes) == before_ids
    assert tuple(widget.pipeline.connections) == before_connections
    assert widget.pipeline.output_tunnel_list() == before_tunnels


def test_tunnel_palette_preview_rejects_node_that_cannot_preserve_users(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    assert widget.load_example_workflow("graph-authoring") is not None

    compatible, _message = widget._tunnel_insert_preview_state(
        "invert",
        "Shared processed image",
    )
    incompatible, message = widget._tunnel_insert_preview_state(
        "input",
        "Shared processed image",
    )

    assert compatible == "compatible"
    assert incompatible == "incompatible"
    assert "cannot keep every user" in message


def test_canceling_tunnel_port_choice_is_reported_as_cancel_not_incompatible(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    assert widget.load_example_workflow("graph-authoring") is not None
    mapping = ConnectionInsertPortMapping(0, 0, "input", "output", "")
    monkeypatch.setattr(
        widget,
        "_tunnel_insert_mapping_options",
        lambda *_args, **_kwargs: [mapping, mapping],
    )
    monkeypatch.setattr(
        widget,
        "_choose_tunnel_insert_mapping",
        lambda *_args, **_kwargs: None,
    )
    before_ids = set(widget.pipeline.nodes)

    inserted = widget._insert_new_node_before_tunnel(
        "invert",
        "Shared processed image",
        QPointF(620.0, 180.0),
    )

    assert inserted is None
    assert set(widget.pipeline.nodes) == before_ids
    assert widget.status_label.text() == "Insert before tunnel canceled."


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


def test_node_code_translates_scalar_channel_axis_sentinel(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("bilateral_filter")
    widget._connect_nodes("input", node.id)

    code = widget._node_code_text(node.id)
    call_line = next(line for line in code.splitlines() if line.startswith("output ="))

    assert "'channel_axis': None" in code
    assert "channel_axis=None" in call_line
    assert "channel_axis=-1" not in call_line


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


def test_widget_close_terminates_queued_optimizer_dialog(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    shutdown_calls = []

    class QueuedDialog:
        running = True

        @staticmethod
        def shutdown():
            shutdown_calls.append(None)

    widget._pipeline_optimizer_dialog = QueuedDialog()
    widget._pipeline_optimizer_baseline = widget._current_history_snapshot()
    widget._pipeline_optimizer_source_signature = ("captured",)

    widget.close()

    assert widget._pipeline_optimizer_dialog is None
    assert widget._pipeline_optimizer_baseline is None
    assert widget._pipeline_optimizer_source_signature is None
    assert shutdown_calls == [None]
    assert widget._compute_node_preferences == {}


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


def test_input_node_tile_binding_subtitle_updates_live(qtbot, tmp_path):
    viewer = _Viewer()
    viewer.layers.append(_Layer(np.ones((4, 5), dtype=np.uint8), "second layer"))
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    widget.graph_view.select_node("input")
    card = widget.graph_view._cards["input"]

    assert card.subtitle_label._full_text == "Layer · input volume"
    control = widget._parameter_widgets["image_source"]
    control.layer_combo.setCurrentText("second layer")
    assert card.subtitle_label._full_text == "Layer · second layer"
    assert card.subtitle_label.toolTip() == "Napari layer: second layer"

    file_path = tmp_path / "P-tau_MAGENTA.tif"
    value = widget._image_source_value(widget.pipeline.nodes["input"])
    value.update(
        source_mode="file path",
        file_path=str(file_path),
        binding_mode="single item",
    )
    widget._on_image_source_changed(value)
    assert card.subtitle_label._full_text == "File · P-tau_MAGENTA"
    assert card.subtitle_label.toolTip() == f"File source: {file_path}"

    value.update(
        source_mode="sample",
        sample_name="VIPP synthetic colocalization",
    )
    widget._on_image_source_changed(value)
    assert card.subtitle_label._full_text == ("Sample · VIPP synthetic colocalization")
    assert card.subtitle_label.toolTip() == (
        "Bundled sample: VIPP synthetic colocalization"
    )


def test_live_napari_source_uses_one_owned_read_only_revision(qtbot):
    data = np.arange(20, dtype=np.uint16).reshape(4, 5)
    viewer = _Viewer(data, metadata={"axes": "YX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    source_output = widget.pipeline.outputs["input"]
    first_payload, _layer = widget._resolve_source_payload(
        widget.pipeline.nodes["input"]
    )
    second_payload, _layer = widget._resolve_source_payload(
        widget.pipeline.nodes["input"]
    )

    assert first_payload is not None
    assert second_payload is not None
    assert first_payload.data is second_payload.data
    assert source_output is first_payload.data
    assert not np.shares_memory(source_output, data)
    assert not source_output.flags.writeable
    expected = source_output.copy()
    data[:] = 0
    np.testing.assert_array_equal(source_output, expected)


def test_live_source_data_event_advances_revision_and_recalculates(qtbot):
    original = np.arange(20, dtype=np.uint16).reshape(4, 5)
    viewer = _Viewer(original, metadata={"axes": "YX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    layer = viewer.layers["input volume"]
    old_output = widget.pipeline.outputs["input"]
    old_signature = widget._last_pipeline_source_signature
    qtbot.waitUntil(
        lambda: widget._active_thumbnail_contrast_run_id is None,
        timeout=5_000,
    )
    widget.preview_mode_combo.setCurrentText("Off")
    sentinel_key = ("thumbnail-sentinel",)
    widget._thumbnail_contrast_limit_cache[sentinel_key] = (0.0, 1.0)
    widget._thumbnail_contrast_statistics_cache[sentinel_key] = object()
    widget._thumbnail_contrast_failure_cache[sentinel_key] = "old failure"
    widget._thumbnail_contrast_identity_refs[sentinel_key] = weakref.ref(old_output)

    replacement = np.full((4, 5), 17, dtype=np.uint16)
    layer.data = replacement
    layer.events.data.emit()
    widget._debounce_timer.stop()
    widget.run_pipeline(force_sync=True)

    updated = widget.pipeline.outputs["input"]
    assert updated is not old_output
    assert widget._last_pipeline_source_signature != old_signature
    assert not np.shares_memory(updated, replacement)
    assert not updated.flags.writeable
    np.testing.assert_array_equal(updated, replacement)
    assert widget._thumbnail_contrast_limit_cache == {}
    assert widget._thumbnail_contrast_statistics_cache == {}
    assert widget._thumbnail_contrast_failure_cache == {}
    assert widget._thumbnail_contrast_identity_refs == {}


def test_explicit_refresh_captures_direct_live_array_mutation(qtbot):
    live_data = np.zeros((4, 5), dtype=np.uint8)
    viewer = _Viewer(live_data, metadata={"axes": "YX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    old_output = widget.pipeline.outputs["input"]

    live_data[:] = 31
    widget._refresh_and_run()

    updated = widget.pipeline.outputs["input"]
    assert updated is not old_output
    assert not np.shares_memory(updated, live_data)
    np.testing.assert_array_equal(updated, live_data)


def test_live_napari_scale_translation_and_units_enter_image_state(qtbot):
    viewer = _Viewer(
        np.zeros((4, 5), dtype=np.float32),
        metadata={"axes": "YX"},
    )
    layer = viewer.layers["input volume"]
    layer.scale = (0.5, 0.25)
    layer.translate = (10.0, 20.0)
    layer.units = ("micrometer", "micrometer")
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    state = widget.pipeline.output_states["input"]

    assert tuple(axis.scale for axis in state.axes) == (0.5, 0.25)
    assert tuple(axis.translation for axis in state.axes) == (10.0, 20.0)
    assert tuple(axis.unit for axis in state.axes) == (
        "micrometer",
        "micrometer",
    )
    assert "napari layer transform" in state.metadata_source


def test_live_napari_rotation_is_rejected_instead_of_discarded(qtbot):
    viewer = _Viewer(
        np.zeros((4, 5), dtype=np.float32),
        metadata={"axes": "YX"},
    )
    viewer.layers["input volume"].rotate = (
        (0.0, -1.0),
        (1.0, 0.0),
    )

    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    assert "has rotation" in widget.status_label.text()
    assert "does not silently discard" in widget.status_label.text()
    assert widget.pipeline.outputs.get("input") is None


def test_replacing_bound_layer_object_recalculates_same_named_source(qtbot):
    viewer = _Viewer(np.zeros((4, 5), dtype=np.uint8), metadata={"axes": "YX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    old_layer = viewer.layers["input volume"]
    replacement_data = np.full((4, 5), 9, dtype=np.uint8)
    replacement = _Layer(
        replacement_data,
        "input volume",
        metadata={"axes": "YX"},
    )

    viewer.layers.remove(old_layer)
    viewer.layers.append(replacement)
    viewer.layers.events.removed.emit()

    output = widget.pipeline.outputs["input"]
    assert not np.shares_memory(output, replacement_data)
    np.testing.assert_array_equal(output, replacement_data)


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


def test_file_source_is_one_owned_read_only_snapshot_until_refresh(
    qtbot,
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "source.npy"
    path.write_bytes(b"stable source identity")
    backing = np.arange(20, dtype=np.uint16).reshape(4, 5)
    read_calls = []

    def fake_read_image(path_arg, *, series_index=0):
        read_calls.append((Path(path_arg), series_index))
        state = image_state_from_array(
            backing,
            layer_metadata={"axes": "YX"},
            source_name="Frozen source",
        )
        series = ImageSeriesInfo(
            0,
            "0",
            "Frozen source",
            backing.shape,
            "uint16",
            "YX",
        )
        inspection = SourceInspection(str(path_arg), "numpy-npy", (series,))
        return ImageDataset(backing, state, inspection, series)

    monkeypatch.setattr("napari_vipp._widget.read_image", fake_read_image)

    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    node = widget.pipeline.nodes["input"]
    node.params.update(
        source_mode="file path",
        file_path=str(path),
        series_index=0,
    )
    widget._mark_pipeline_dirty("input")
    widget.run_pipeline(force_sync=True)

    frozen = widget.pipeline.outputs["input"]
    expected = backing.copy()
    snapshot = next(iter(widget._file_source_payload_cache.values()))
    assert isinstance(frozen, np.ndarray)
    assert frozen.flags.owndata
    assert not frozen.flags.writeable
    assert not np.shares_memory(frozen, backing)
    assert snapshot.payload.metadata["vipp_source_snapshot_policy"] == (
        "pinned until Refresh"
    )
    assert "pinned until Refresh" in widget.status_label.text()
    with pytest.raises(ValueError, match="read-only"):
        frozen[0, 0] = 99

    backing[:] = 0
    widget.run_pipeline(force_sync=True)

    assert len(read_calls) == 1
    assert widget.pipeline.outputs["input"] is frozen
    np.testing.assert_array_equal(frozen, expected)


def test_interactive_batch_payload_uses_same_explicit_axis_declaration(qtbot):
    data = np.zeros((3, 8, 9), dtype=np.uint16)
    raw_state = image_state_from_array(
        data,
        layer_metadata={"axes": "QYX"},
        source_name="generic stack",
    )
    assert raw_state is not None
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget._interactive_collection_batch_config = BatchConfig(
        workflow_file=Path("workflow.json"),
        workflow_sha256="a" * 64,
        output_dir=Path("outputs"),
        sources=(
            BatchSourceConfig(
                "input",
                "Image Source",
                Path("inputs"),
                "*.tif",
                AxisDeclaration("QYX", "ZYX"),
            ),
        ),
        outputs=(
            BatchOutputConfig(
                "output",
                "Batch Output",
                "result",
                "image",
                "npy",
                "",
                "{source_stem}__{tag}",
            ),
        ),
        save_python_script=False,
    )
    raw_payload = SourcePayload(data, {}, "generic stack", raw_state)

    effective = widget._declared_interactive_batch_payload("input", raw_payload)

    assert raw_payload.image_state.axis_order == "QYX"
    assert effective.image_state.axis_order == "ZYX"
    assert effective.metadata["vipp_axis_semantics"] == {
        "raw_axes": "QYX",
        "effective_axes": "ZYX",
        "declaration": {
            "source_axes": "QYX",
            "effective_axes": "ZYX",
            "source": "batch config",
            "applied": True,
            "data_order_changed": False,
        },
    }
    np.testing.assert_array_equal(effective.data, data)


def test_interactive_batch_recalculates_when_axis_declaration_changes(
    qtbot,
    tmp_path,
    monkeypatch,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    demo = widget._create_collection_batch_demo(tmp_path / "demo")
    widget._batch_collection_dialog(config_path=demo.config_path)
    qtbot.waitUntil(
        lambda: (
            widget._interactive_collection_batch_requested_index == -1
            and widget._active_pipeline_run_id is None
            and widget._active_source_load_id is None
        ),
        timeout=5_000,
    )
    config = widget._interactive_collection_batch_config
    assert config is not None
    changed = replace(
        config,
        sources=(
            replace(
                config.sources[0],
                axis_declaration=AxisDeclaration("YX", "YX"),
            ),
            *config.sources[1:],
        ),
    )
    recalculated: list[int] = []

    def record_preview(index, *, force_sync=False):
        del force_sync
        recalculated.append(index)
        return True

    monkeypatch.setattr(
        widget,
        "_preview_interactive_collection_batch_item",
        record_preview,
    )

    widget._activate_interactive_collection_batch(
        widget._interactive_collection_batch_items,
        changed,
        initial_index=0,
        force_sync=True,
    )

    assert recalculated == [0]
    assert widget._interactive_collection_batch_config is changed


def test_zarr_file_source_materializes_in_background(
    qtbot,
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "source.zarr"
    path.mkdir()
    (path / "chunk.bin").write_bytes(b"stable")
    started = threading.Event()
    release = threading.Event()
    backing = np.arange(20, dtype=np.uint8).reshape(4, 5)

    def fake_read_image(path_arg, *, series_index=0):
        started.set()
        assert release.wait(5)
        state = image_state_from_array(
            backing,
            layer_metadata={"axes": "YX"},
            source_name="Loaded Zarr",
        )
        series = ImageSeriesInfo(
            0,
            "0",
            "Loaded Zarr",
            backing.shape,
            "uint8",
            "YX",
        )
        inspection = SourceInspection(str(path_arg), "ome-zarr-0.4", (series,))
        return ImageDataset(backing, state, inspection, series)

    monkeypatch.setattr("napari_vipp._widget.read_image", fake_read_image)

    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    node = widget.pipeline.nodes["input"]
    node.params.update(
        source_mode="file path",
        file_path=str(path),
        series_index=0,
    )
    widget._mark_pipeline_dirty("input")

    widget.run_pipeline()

    assert widget._active_source_load_id is not None
    qtbot.waitUntil(started.is_set, timeout=5_000)
    assert "Loading image source" in widget.status_label.text()

    release.set()
    qtbot.waitUntil(
        lambda: (
            widget._active_source_load_id is None
            and np.array_equal(widget.pipeline.outputs.get("input"), backing)
        ),
        timeout=10_000,
    )

    frozen = widget.pipeline.outputs["input"]
    assert frozen.flags.owndata
    assert not frozen.flags.writeable
    assert not np.shares_memory(frozen, backing)


def test_changed_zarr_stays_pinned_until_explicit_refresh(
    qtbot,
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "changing.zarr"
    path.mkdir()
    chunk_path = path / "chunk.bin"
    chunk_path.write_bytes(b"A")
    root_stat = path.stat()
    read_values = []

    def fake_read_image(path_arg, *, series_index=0):
        value = (Path(path_arg) / "chunk.bin").read_bytes()[0]
        read_values.append(value)
        data = np.full((3, 4), value, dtype=np.uint8)
        state = image_state_from_array(
            data,
            layer_metadata={"axes": "YX"},
            source_name="Changing Zarr",
        )
        series = ImageSeriesInfo(
            0,
            "0",
            "Changing Zarr",
            data.shape,
            "uint8",
            "YX",
        )
        inspection = SourceInspection(str(path_arg), "ome-zarr-0.4", (series,))
        return ImageDataset(data, state, inspection, series)

    monkeypatch.setattr("napari_vipp._widget.read_image", fake_read_image)

    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    node = widget.pipeline.nodes["input"]
    node.params.update(
        source_mode="file path",
        file_path=str(path),
        series_index=0,
    )
    widget._mark_pipeline_dirty("input")
    widget.run_pipeline()
    qtbot.waitUntil(
        lambda: (
            widget._active_source_load_id is None
            and np.all(widget.pipeline.outputs.get("input") == ord("A"))
        ),
        timeout=10_000,
    )
    first_snapshot = widget.pipeline.outputs["input"]

    chunk_path.write_bytes(b"B")
    os.utime(
        path,
        ns=(root_stat.st_atime_ns, root_stat.st_mtime_ns),
    )
    assert path.stat().st_mtime_ns == root_stat.st_mtime_ns
    assert path.stat().st_size == root_stat.st_size

    widget.run_pipeline(force_sync=True)

    assert read_values == [ord("A")]
    assert widget.pipeline.outputs["input"] is first_snapshot
    assert np.all(first_snapshot == ord("A"))

    widget._refresh_and_run()
    qtbot.waitUntil(
        lambda: (
            widget._active_source_load_id is None
            and np.all(widget.pipeline.outputs.get("input") == ord("B"))
        ),
        timeout=10_000,
    )

    assert read_values == [ord("A"), ord("B")]
    assert widget.pipeline.outputs["input"] is not first_snapshot


def test_refresh_rejects_stale_inflight_file_load(
    qtbot,
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "slow.zarr"
    path.mkdir()
    (path / "chunk.bin").write_bytes(b"stable")
    first_started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    calls = 0

    def fake_read_image(path_arg, *, series_index=0):
        nonlocal calls
        calls += 1
        call = calls
        if call == 1:
            first_started.set()
            assert release_first.wait(5)
        else:
            second_started.set()
        data = np.full((3, 4), call, dtype=np.uint8)
        state = image_state_from_array(
            data,
            layer_metadata={"axes": "YX"},
            source_name=f"Load {call}",
        )
        series = ImageSeriesInfo(
            0,
            "0",
            f"Load {call}",
            data.shape,
            "uint8",
            "YX",
        )
        inspection = SourceInspection(str(path_arg), "ome-zarr-0.4", (series,))
        return ImageDataset(data, state, inspection, series)

    monkeypatch.setattr("napari_vipp._widget.read_image", fake_read_image)

    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    node = widget.pipeline.nodes["input"]
    node.params.update(
        source_mode="file path",
        file_path=str(path),
        series_index=0,
    )
    widget._mark_pipeline_dirty("input")
    widget.run_pipeline()
    qtbot.waitUntil(first_started.is_set, timeout=5_000)
    first_run_id = widget._active_source_load_id

    widget._refresh_and_run()

    second_run_id = widget._active_source_load_id
    assert second_run_id is not None
    assert second_run_id != first_run_id
    release_first.set()
    qtbot.waitUntil(second_started.is_set, timeout=5_000)
    qtbot.waitUntil(
        lambda: (
            widget._active_source_load_id is None
            and np.all(widget.pipeline.outputs.get("input") == 2)
        ),
        timeout=10_000,
    )

    assert calls == 2
    assert np.all(widget.pipeline.outputs["input"] == 2)
    snapshots = list(widget._file_source_payload_cache.values())
    assert len(snapshots) == 1
    assert np.all(snapshots[0].payload.data == 2)


def test_verified_file_inspection_pins_revision_until_refresh(qtbot, tmp_path):
    path = tmp_path / "inspected.npy"
    first = np.full((3, 4), 7, dtype=np.uint8)
    second = np.full((3, 4), 9, dtype=np.uint8)
    np.save(path, first)

    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    node = widget.pipeline.nodes["input"]
    node.params.update(
        source_mode="file path",
        file_path=str(path),
        series_index=0,
    )

    inspection = widget._inspect_source_file(str(path))
    assert inspection is not None
    np.save(path, second)
    widget._mark_pipeline_dirty("input")
    widget.run_pipeline(force_sync=True)

    assert "Press Refresh" in widget.status_label.text()

    widget._refresh_and_run()

    np.testing.assert_array_equal(widget.pipeline.outputs["input"], second)


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
            and not widget._queued_thumbnail_contrast_limit_requests
            and widget.pipeline_busy_bar.isHidden()
        ),
        timeout=5_000,
    )

    assert widget.pipeline_busy_bar.isHidden()
    assert not widget.graph_view._cards["input"].is_processing()
    assert widget.pipeline.output_states["input"].axis_order == "YX"


def test_uncached_source_waits_for_active_pipeline_cleanup(qtbot, monkeypatch):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    cancel_event = threading.Event()
    widget._active_pipeline_run_id = 41
    widget._pipeline_cancel_events[41] = cancel_event
    spec = SourceFileLoadSpec(
        node_id="input",
        path="C:/data/new-source.nd2",
        series_index=0,
        cache_key=("C:/data/new-source.nd2", 0),
    )
    monkeypatch.setattr(
        widget,
        "_uncached_async_file_source_specs",
        lambda: (spec,),
    )
    started = []
    monkeypatch.setattr(
        widget,
        "_start_source_file_load",
        lambda specs: started.append(specs),
    )

    widget.run_pipeline()

    assert started == []
    assert cancel_event.is_set()
    assert widget._pipeline_run_pending
    assert widget._active_source_load_id is None
    assert "waiting for CPU/GPU cleanup" in widget.status_label.text()

    widget._active_pipeline_run_id = None
    widget._pipeline_run_pending = False
    widget.run_pipeline()

    assert started == [(spec,)]


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
    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and widget.pipeline.outputs[node.id] is not None
        ),
        timeout=30_000,
    )
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
    qtbot.waitUntil(
        lambda: widget.graph_view.node_has_thumbnail("input"),
        timeout=5_000,
    )

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
        preview_size=None,
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
    scientific_output = widget.pipeline.outputs["input"]
    assert np.shares_memory(inspect_layer.data, scientific_output)
    assert not np.shares_memory(
        inspect_layer.data,
        viewer.layers["input volume"].data,
    )
    expected_source = viewer.layers["input volume"].data.copy()
    assert not inspect_layer.data.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        inspect_layer.data.flat[0] = 1
    np.testing.assert_array_equal(
        viewer.layers["input volume"].data,
        expected_source,
    )
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
    assert pinned.editable is False
    assert not np.shares_memory(pinned.data, widget.pipeline.outputs["threshold"])
    assert not pinned.data.flags.writeable
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
    assert inspect.editable is False
    assert inspect.metadata["data_kind"] == "labels"
    assert inspect.metadata["display_kind"] == "labels"
    assert widget.pipeline.output_states[filtered.id].kind == "label image"
    labels_output = widget.pipeline.outputs[filtered.id]
    expected_labels = labels_output.copy()
    assert np.shares_memory(inspect.data, labels_output)
    assert not inspect.data.flags.writeable

    widget.pin_node(filtered.id)

    pinned = viewer.layers["VIPP Pinned: Filter Labels By Volume"]
    assert pinned.layer_type == "labels"
    assert pinned.data.dtype == np.int32
    assert pinned.editable is False
    assert pinned.metadata["data_kind"] == "labels"
    assert np.shares_memory(pinned.data, labels_output)
    assert np.shares_memory(pinned.data, inspect.data)
    assert not pinned.data.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        inspect.data.flat[0] = 101
    with pytest.raises(ValueError, match="read-only"):
        pinned.data.flat[-1] = 202
    np.testing.assert_array_equal(labels_output, expected_labels)


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


def test_clear_border_hides_equivalent_boundary_control_for_true_2d_input(qtbot):
    data = np.zeros((12, 12), dtype=np.float32)
    data[0:4, 0:4] = 10
    data[6:10, 6:10] = 10
    viewer = _Viewer(data, metadata={"axes": "YX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    cleared = widget.add_node_from_palette("clear_border_objects")
    widget._connect_nodes("threshold", cleared.id)

    assert "boundary_mode" not in widget._parameter_widgets
    assert widget.pipeline.nodes[cleared.id].params["boundary_mode"] == (
        "All spatial borders"
    )
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

    assert "spatial_mode" not in widget._parameter_widgets
    assert (
        "connected YX image"
        in widget._parameter_widgets["fill_holes_scope_note"].text()
    )


@pytest.mark.parametrize("trailing_size", [3, 4])
def test_inferred_yxc_auto_is_unavailable_but_explicit_spatial_modes_remain(
    qtbot,
    trailing_size,
):
    data = np.zeros((7, 9, trailing_size), dtype=np.float32)
    viewer = _Viewer(data)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    filtered = widget.add_node_from_palette("remove_small_objects")
    widget._connect_nodes("threshold", filtered.id)

    control = widget._parameter_widgets["spatial_mode"]
    choices = [control.combo.itemText(index) for index in range(control.combo.count())]

    assert choices == [
        "Auto from axes - unavailable (axes are inferred or missing)",
        "2D YX",
        "3D ZYX",
    ]
    assert widget.pipeline.input_state_for_node(filtered.id).axis_order == "YXC"
    assert (
        widget.pipeline.input_state_for_node(filtered.id).axis_confidence
        == "shape-inferred"
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
    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and not widget._pipeline_run_pending
            and widget.pipeline.outputs.get(filtered.id) is not None
        ),
        timeout=5_000,
    )

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
    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and not widget._pipeline_run_pending
            and minimum_control._bounds.maximum == largest_2d
            and maximum_control._bounds.maximum == largest_2d
        ),
        timeout=5_000,
    )

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

    node = widget.add_node_from_palette("invert")
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


def test_slice_wise_stack_notice_hides_irrelevant_rgb_axis_parameter(qtbot):
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
    assert _palette_child_by_text(axes_regions, "Set Microscope Metadata")
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


def test_set_microscope_metadata_uses_precise_numeric_entries(qtbot):
    viewer = _Viewer(np.zeros((2, 16, 18), dtype=np.float32))
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("set_microscope_metadata")
    widget._connect_nodes("input", node.id)

    wavelength = widget._parameter_widgets["channel_1_wavelength_nm"]
    numerical_aperture = widget._parameter_widgets["numerical_aperture"]

    assert not hasattr(wavelength, "slider")
    assert not hasattr(numerical_aperture, "slider")
    wavelength.value_box.setValue(461.0)
    numerical_aperture.value_box.setValue(1.4)

    assert node.params["channel_1_wavelength_nm"] == 461.0
    assert node.params["numerical_aperture"] == 1.4


def test_set_pixel_size_inspection_applies_napari_layer_scale(qtbot):
    viewer = _Viewer(
        np.zeros((3, 16, 18), dtype=np.float32),
        metadata={"axes": "ZYX"},
    )
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("set_pixel_size")
    widget._connect_nodes("input", node.id)
    widget.pipeline.set_param(node.id, "x_size", 1.0)
    widget.pipeline.set_param(node.id, "y_size", 1.0)
    widget.pipeline.set_param(node.id, "z_size", 1.0)
    widget.run_pipeline()
    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and not widget._pipeline_run_pending
            and widget.pipeline.outputs.get(node.id) is not None
        ),
        timeout=5_000,
    )
    widget.inspect_node(node.id)

    inspect = viewer.layers["VIPP Inspect"]
    assert inspect.scale == (1.0, 1.0, 1.0)
    widget.pin_node(node.id)
    pinned = viewer.layers["VIPP Pinned: Set Pixel Size / Units"]
    assert pinned.scale == (1.0, 1.0, 1.0)

    widget.pipeline.set_param(node.id, "z_size", 10.0)
    widget.run_pipeline()
    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and not widget._pipeline_run_pending
            and viewer.layers["VIPP Inspect"].scale == (10.0, 1.0, 1.0)
            and viewer.layers["VIPP Pinned: Set Pixel Size / Units"].scale
            == (10.0, 1.0, 1.0)
        ),
        timeout=5_000,
    )

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
    assert label.text() == "Z scale factor (12 -> 24)"
    assert widget.pipeline.outputs[rescale.id].shape == (3, 96, 24, 128)
    assert _metadata_value(widget, "Dimensions") == "c=3, y=96, z=24, x=128"


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
    assert "finite support window" in xy_size.toolTip()
    assert "finite support window" in widget._parameter_widgets["z_size"].toolTip()
    assert "Quadrature samples" in widget._parameter_widgets["pupil_samples"].toolTip()
    assert widget._parameter_widgets["xy_size_label"].toolTip() == xy_size.toolTip()
    assert "user set;" in widget._parameter_widgets["xy_size_status"].text()
    guidance = widget._parameter_widgets["operation_notice"]
    assert guidance.property("preflightStatus") == "warning"
    assert "WIDEFIELD NYQUIST ESTIMATE NOT MET" in guidance.text()
    assert "Requested PSF: 33 x 65 x 65 samples" in guidance.text()
    assert "Use the tail check after calculation" in guidance.text()
    assert "Z: PSF 33, image 3" in guidance.text()
    assert "Born-Wolf support guide" in guidance.text()
    assert guidance.openExternalLinks()


def test_born_wolf_psf_support_shows_attached_metadata_scale_without_calculation(
    qtbot,
):
    data = np.zeros((11, 35, 37), dtype=np.float32)
    axes = (
        AxisMetadata("z", "space", unit="micrometer", scale=0.101),
        AxisMetadata("y", "space", unit="micrometer", scale=0.025),
        AxisMetadata("x", "space", unit="micrometer", scale=0.025),
    )
    state = image_state_from_array(
        data,
        axes=axes,
        channels=(
            ChannelMetadata(
                name="561 nm",
                emission_wavelength=561.0,
                emission_wavelength_unit="nanometer",
            ),
        ),
        acquisition=AcquisitionMetadata(
            objective_na=1.46,
            refractive_index=1.518,
        ),
    )
    widget = VippWidget(_Viewer(data, metadata={"vipp_image_state": state.to_dict()}))
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("born_wolf_psf")
    assert widget.pipeline.connect("input", node.id).success
    widget._render_parameters(node.id)

    assert widget._parameter_widgets["xy_size_status"].text() == (
        "user set; 1.6 um span"
    )
    assert widget._parameter_widgets["z_size_status"].text() == (
        "user set; 3.23 um span"
    )
    guidance = widget._parameter_widgets["operation_notice"].text()
    assert "Physical span between outer sample centers" in guidance
    assert "Z 3.23 um; YX 1.6 um" in guidance
    assert "TAIL CHECK PENDING" in guidance
    assert "Born-Wolf support guide" in guidance


def test_born_wolf_psf_support_reports_generated_tail_containment(qtbot):
    data = np.zeros((35, 70, 72), dtype=np.float32)
    axes = (
        AxisMetadata("z", "space", unit="micrometer", scale=0.101),
        AxisMetadata("y", "space", unit="micrometer", scale=0.025),
        AxisMetadata("x", "space", unit="micrometer", scale=0.025),
    )
    state = image_state_from_array(
        data,
        axes=axes,
        channels=(
            ChannelMetadata(
                name="561 nm",
                emission_wavelength=561.0,
                emission_wavelength_unit="nanometer",
            ),
        ),
        acquisition=AcquisitionMetadata(
            objective_na=1.46,
            refractive_index=1.518,
        ),
    )
    widget = VippWidget(_Viewer(data, metadata={"vipp_image_state": state.to_dict()}))
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("born_wolf_psf")
    assert widget.pipeline.connect("input", node.id).success
    psf = np.zeros((33, 65, 65), dtype=np.float32)
    psf[16, 32, 32] = 1.0
    widget.pipeline.outputs[node.id] = psf
    widget.pipeline.output_states[node.id] = image_state_from_array(psf, axes=axes)
    widget.pipeline.node_execution_states[node.id] = "ready"
    widget._pending_dirty_node_ids.discard(node.id)

    widget._render_parameters(node.id)

    guidance = widget._parameter_widgets["operation_notice"]
    assert "TAIL CONTAINMENT CHECK PASSED" in guidance.text()
    assert "outermost samples contain 0.0%" in guidance.text()

    psf[0, 32, 32] = 0.02
    psf[16, 32, 32] = 0.98
    widget._render_parameters(node.id)

    guidance = widget._parameter_widgets["operation_notice"]
    assert guidance.property("preflightStatus") == "warning"
    assert "TAIL REACHES THE WINDOW EDGE" in guidance.text()
    assert "outermost samples contain 2.0%" in guidance.text()


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
    viewer = _Viewer(
        np.zeros((16, 18), dtype=np.float32),
        metadata={"axes": "YX"},
    )
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("li_threshold")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    assert "threshold_scope" not in widget._parameter_widgets
    assert not widget._parameter_widgets
    assert not widget.parameter_group.isHidden()
    assert not widget.rescale_input_histogram_group.isHidden()
    assert widget.rescale_input_histogram_scope_row.isHidden()
    assert widget.rescale_input_histogram_group.title() == "Input Histogram"


def test_global_threshold_scope_remains_visible_for_shape_only_2d_input(qtbot):
    widget = VippWidget(_Viewer(np.zeros((16, 18), dtype=np.float32)))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("li_threshold")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    assert "threshold_scope" in widget._parameter_widgets
    notes = []
    for row in range(widget.parameter_form.rowCount()):
        item = widget.parameter_form.itemAt(row, QFormLayout.SpanningRole)
        if item is not None:
            notes.append(item.widget())
    assert any(
        isinstance(note, QLabel) and "unresolved" in note.text().casefold()
        for note in notes
    )


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


@pytest.mark.parametrize(
    "operation_id",
    (
        "otsu_threshold",
        "triangle_threshold",
        "yen_threshold",
        "isodata_threshold",
        "minimum_threshold",
    ),
)
@pytest.mark.parametrize(
    ("dtype", "expected_visible"),
    ((np.float32, True), (np.uint16, False)),
)
def test_global_threshold_float_bins_follow_input_dtype(
    qtbot,
    operation_id,
    dtype,
    expected_visible,
):
    widget = VippWidget(
        _Viewer(np.zeros((3, 16, 18), dtype=dtype), metadata={"axes": "ZYX"})
    )
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette(operation_id)
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    assert ("histogram_bins" in widget._parameter_widgets) is expected_visible


def test_input_aware_controls_refresh_after_upstream_dtype_change(qtbot):
    data = np.zeros((3, 8, 9), dtype=np.float32)
    widget = VippWidget(_Viewer(data, metadata={"axes": "ZYX"}))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("otsu_threshold")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)
    assert "histogram_bins" in widget._parameter_widgets

    integer = np.zeros_like(data, dtype=np.int16)
    widget.pipeline.outputs["input"] = integer
    widget.pipeline.node_outputs["input"] = [integer]
    widget.pipeline.output_states["input"] = image_state_from_array(
        integer,
        layer_metadata={"axes": "ZYX"},
    )
    widget.pipeline.node_output_states["input"] = [
        widget.pipeline.output_states["input"]
    ]
    widget._refresh_selected_parameter_controls()
    assert "histogram_bins" not in widget._parameter_widgets

    floating = np.zeros_like(data, dtype=np.float64)
    widget.pipeline.outputs["input"] = floating
    widget.pipeline.node_outputs["input"] = [floating]
    widget.pipeline.output_states["input"] = image_state_from_array(
        floating,
        layer_metadata={"axes": "ZYX"},
    )
    widget.pipeline.node_output_states["input"] = [
        widget.pipeline.output_states["input"]
    ]
    widget._refresh_selected_parameter_controls()
    assert "histogram_bins" in widget._parameter_widgets


def test_visibility_refreshes_after_actual_upstream_dtype_conversion(qtbot):
    data = np.linspace(0, 1, 3 * 8 * 9, dtype=np.float32).reshape(3, 8, 9)
    widget = VippWidget(_Viewer(data, metadata={"axes": "ZYX"}))
    qtbot.addWidget(widget)
    convert = widget.add_node_from_palette("convert_dtype")
    threshold = widget.add_node_from_palette("otsu_threshold")
    widget._connect_nodes("input", convert.id)
    widget._connect_nodes(convert.id, threshold.id)

    widget.pipeline.run(data, input_metadata={"axes": "ZYX"})
    widget.graph_view.select_node(threshold.id)
    assert widget.pipeline.outputs[convert.id].dtype == np.uint8
    assert "histogram_bins" not in widget._parameter_widgets

    widget.pipeline.set_param(convert.id, "output_dtype", "float32")
    widget.pipeline.run(
        data,
        input_metadata={"axes": "ZYX"},
        dirty_node_ids={convert.id},
    )
    widget._refresh_selected_parameter_controls()

    assert widget.pipeline.outputs[convert.id].dtype == np.float32
    assert "histogram_bins" in widget._parameter_widgets


def test_unresolved_controls_refresh_when_connection_changes(qtbot, monkeypatch):
    data = np.zeros((3, 8, 9), dtype=np.uint8)
    widget = VippWidget(_Viewer(data, metadata={"axes": "ZYX"}))
    qtbot.addWidget(widget)
    monkeypatch.setattr(widget, "run_pipeline", lambda: None)
    node = widget.add_node_from_palette("otsu_threshold")
    widget.graph_view.select_node(node.id)

    assert "histogram_bins" in widget._parameter_widgets
    assert "channel_axis" in widget._parameter_widgets

    widget._connect_nodes("input", node.id)
    assert "histogram_bins" not in widget._parameter_widgets
    assert "channel_axis" not in widget._parameter_widgets

    widget._disconnect_nodes("input", node.id)
    assert "histogram_bins" in widget._parameter_widgets
    assert "channel_axis" in widget._parameter_widgets


def test_hidden_histogram_bins_preserve_value_and_generated_code(qtbot):
    data = np.zeros((3, 8, 9), dtype=np.uint16)
    widget = VippWidget(_Viewer(data, metadata={"axes": "ZYX"}))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("otsu_threshold")
    widget.pipeline.set_param(node.id, "histogram_bins", 4_096)
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    assert "histogram_bins" not in widget._parameter_widgets
    assert node.params["histogram_bins"] == 4_096
    assert "histogram_bins=4096" in widget._node_code_text(node.id)


@pytest.mark.parametrize(
    ("axes", "expected_visible"),
    (("Y,X,rgb", True), ("Y,X,rgba", True), ("ZYX", False)),
)
def test_rgb_axis_control_follows_explicit_input_semantics(
    qtbot,
    axes,
    expected_visible,
):
    shape = (
        (8, 9, 3)
        if axes == "Y,X,rgb"
        else (8, 9, 4)
        if axes == "Y,X,rgba"
        else (3, 8, 9)
    )
    widget = VippWidget(
        _Viewer(np.zeros(shape, dtype=np.float32), metadata={"axes": axes})
    )
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("sobel_filter")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    assert ("channel_axis" in widget._parameter_widgets) is expected_visible
    if expected_visible:
        tooltip = widget._parameter_widgets["channel_axis"].toolTip().lower()
        assert "rgb or rgba" in tooltip
        assert "-1 for scalar" in tooltip


def test_rgb_axis_control_does_not_treat_shape_as_explicit_semantics(qtbot):
    data = np.zeros((8, 9, 3), dtype=np.float32)
    widget = VippWidget(_Viewer(data))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("sobel_filter")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    assert "channel_axis" in widget._parameter_widgets


def test_parameter_dependent_visibility_refreshes_and_preserves_values(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("richardson_lucy_tv_deconvolution")
    widget.pipeline.set_param(node.id, "tv_epsilon", 2.5e-6)
    widget.pipeline.set_param(node.id, "denominator_floor", 0.125)
    widget.graph_view.select_node(node.id)

    assert "tv_epsilon" in widget._parameter_widgets
    assert "denominator_floor" in widget._parameter_widgets
    widget._on_param_changed("tv_regularization", 0.0)

    assert "tv_epsilon" not in widget._parameter_widgets
    assert "denominator_floor" not in widget._parameter_widgets
    assert node.params["tv_epsilon"] == 2.5e-6
    assert node.params["denominator_floor"] == 0.125
    code = widget._node_code_text(node.id)
    assert "tv_epsilon=2.5e-06" in code
    assert "denominator_floor=0.125" in code

    widget._on_param_changed("tv_regularization", 0.002)
    assert "tv_epsilon" in widget._parameter_widgets
    assert "denominator_floor" in widget._parameter_widgets


def test_costes_auto_keeps_resolved_threshold_rows_as_manual_start(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("object_colocalization_metrics")
    widget.pipeline.set_param(node.id, "channel_1_threshold", 41.0)
    widget.pipeline.set_param(node.id, "channel_2_threshold", 87.0)
    widget.graph_view.select_node(node.id)

    threshold_1 = widget._parameter_widgets["channel_1_threshold"]
    threshold_2 = widget._parameter_widgets["channel_2_threshold"]
    assert threshold_1.isEnabled()
    assert threshold_2.isEnabled()

    widget._on_param_changed("threshold_mode", "Costes auto")

    assert widget._parameter_widgets["channel_1_threshold"] is threshold_1
    assert widget._parameter_widgets["channel_2_threshold"] is threshold_2
    assert not threshold_1.isEnabled()
    assert not threshold_2.isEnabled()
    assert node.params["channel_1_threshold"] == 41.0
    assert node.params["channel_2_threshold"] == 87.0

    # The completed Costes calculation writes its derived values to the node;
    # the inspector must display them even though they remain read-only.
    node.params["channel_1_threshold"] = 53.0
    node.params["channel_2_threshold"] = 97.0
    widget._refresh_selected_parameter_controls()

    assert threshold_1.value() == 53.0
    assert threshold_2.value() == 97.0
    assert not threshold_1.isEnabled()
    assert not threshold_2.isEnabled()

    widget._on_param_changed("threshold_mode", "Manual")

    assert threshold_1.isEnabled()
    assert threshold_2.isEnabled()
    assert threshold_1.value() == 53.0
    assert threshold_2.value() == 97.0
    assert node.params["channel_1_threshold"] == 53.0
    assert node.params["channel_2_threshold"] == 97.0


def test_stored_3d_mode_remains_visible_and_truthful_after_yx_replacement(qtbot):
    data = np.zeros((3, 8, 9), dtype=bool)
    widget = VippWidget(_Viewer(data, metadata={"axes": "ZYX"}))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("auto_watershed_from_mask")
    widget._connect_nodes("input", node.id)
    widget.pipeline.set_param(node.id, "spatial_mode", "3D ZYX")
    widget.graph_view.select_node(node.id)

    replacement = np.zeros((8, 9), dtype=bool)
    replacement_state = image_state_from_array(
        replacement,
        layer_metadata={"axes": "YX"},
    )
    widget.pipeline.outputs["input"] = replacement
    widget.pipeline.node_outputs["input"] = [replacement]
    widget.pipeline.output_states["input"] = replacement_state
    widget.pipeline.node_output_states["input"] = [replacement_state]
    widget._refresh_selected_parameter_controls()

    control = widget._parameter_widgets["spatial_mode"]
    assert control.value() == "3D ZYX"
    assert node.params["spatial_mode"] == "3D ZYX"
    assert "unavailable" in control.combo.currentText().casefold()


def test_visibility_only_refresh_preserves_workflow_history_and_cache(qtbot):
    data = np.zeros((3, 8, 9), dtype=np.float32)
    widget = VippWidget(_Viewer(data, metadata={"axes": "ZYX"}))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("otsu_threshold")
    widget.pipeline.set_param(node.id, "histogram_bins", 4_096)
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)
    assert "histogram_bins" in widget._parameter_widgets

    params_before = dict(node.params)
    workflow_before = serialize_workflow(widget.pipeline)
    hash_before = scientific_workflow_hash(workflow_before)
    history_before = (len(widget._undo_stack), len(widget._redo_stack))
    dirty_before = set(widget._pending_dirty_node_ids)
    completed_before = set(widget.pipeline.completed_node_ids)
    cached_output = widget.pipeline.outputs.get(node.id)

    integer = np.zeros_like(data, dtype=np.uint16)
    integer_state = image_state_from_array(
        integer,
        layer_metadata={"axes": "ZYX"},
    )
    widget.pipeline.outputs["input"] = integer
    widget.pipeline.node_outputs["input"] = [integer]
    widget.pipeline.output_states["input"] = integer_state
    widget.pipeline.node_output_states["input"] = [integer_state]
    assert widget._refresh_selected_parameter_controls() is False

    assert "histogram_bins" not in widget._parameter_widgets
    assert node.params == params_before
    assert serialize_workflow(widget.pipeline) == workflow_before
    assert scientific_workflow_hash(serialize_workflow(widget.pipeline)) == hash_before
    assert (len(widget._undo_stack), len(widget._redo_stack)) == history_before
    assert widget._pending_dirty_node_ids == dirty_before
    assert widget.pipeline.completed_node_ids == completed_before
    assert widget.pipeline.outputs.get(node.id) is cached_output


def test_rescale_z_visibility_refresh_preserves_specialized_params(qtbot):
    data = np.zeros((3, 8, 9), dtype=np.float32)
    widget = VippWidget(_Viewer(data, metadata={"axes": "ZYX"}))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("rescale_axes")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)
    widget._on_param_changed("resize_mode", "Output size")
    widget._on_param_changed("z_size", 5)
    assert "z_size" in widget._parameter_widgets
    params_before = deepcopy(node.params)

    replacement = np.zeros((8, 9), dtype=np.float32)
    replacement_state = image_state_from_array(
        replacement,
        layer_metadata={"axes": "YX"},
    )
    widget.pipeline.outputs["input"] = replacement
    widget.pipeline.node_outputs["input"] = [replacement]
    widget.pipeline.output_states["input"] = replacement_state
    widget.pipeline.node_output_states["input"] = [replacement_state]
    assert widget._refresh_selected_parameter_controls() is False

    assert "z_size" not in widget._parameter_widgets
    assert node.params == params_before


def test_hidden_parameter_round_trips_through_current_workflow_schema(qtbot):
    data = np.zeros((8, 9), dtype=np.uint16)
    widget = VippWidget(_Viewer(data, metadata={"axes": "YX"}))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("otsu_threshold")
    widget.pipeline.set_param(node.id, "histogram_bins", 8_192)
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)
    assert "histogram_bins" not in widget._parameter_widgets

    document = serialize_workflow(widget.pipeline)
    restored_graph = deserialize_workflow(document)
    restored = PrototypePipeline()
    restored.restore_graph(
        restored_graph["nodes"],
        restored_graph["connections"],
        restored_graph.get("output_tunnels", ()),
    )

    assert document["version"] == 4
    assert restored.nodes[node.id].params["histogram_bins"] == 8_192
    assert "histogram_bins=8192" in widget._node_code_text(node.id)
    exported = export_pipeline_to_python(widget.pipeline)
    assert '"histogram_bins":8192' in exported


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
        channel_axis=None,
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
            channel_axis=channel_axis,
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
    scientific_input = widget.pipeline.outputs["input"]

    def counted_histogram(*args, **kwargs):
        if args and args[0] is scientific_input:
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

    assert widget.pipeline.input_data_for_node(node.id) is scientific_input
    assert not np.shares_memory(scientific_input, data)
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
        channel_axis=None,
    ):
        result = original_threshold(
            values,
            operation_id,
            histogram_bins=histogram_bins,
            max_iterations=max_iterations,
            progress=progress,
            channel_axis=channel_axis,
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


def test_node_selection_rejects_a_stale_output_histogram_result(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    old_key = ("gaussian-stack",)
    new_key = ("threshold-stack",)
    widget._active_output_histogram_run_id = 7
    widget._active_output_histogram_key = old_key
    widget._current_output_histogram_key = new_key
    widget._selected_node_id = "threshold"
    applied = []
    monkeypatch.setattr(
        widget,
        "_apply_output_histogram_result",
        lambda result: applied.append(result.key),
    )

    widget._on_output_histogram_finished(
        InputHistogramResult(
            7,
            old_key,
            "gaussian",
            counts=np.array([1]),
        )
    )

    assert applied == []
    assert widget._current_output_histogram_key == new_key


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
    assert wrapped_height >= widget.colocalization_scatter_summary.minimumHeight()
    assert widget.colocalization_scatter_summary.maximumHeight() >= wrapped_height
    assert widget.colocalization_scatter_colormap_combo.currentText() == "Viridis"
    widget.colocalization_scatter_plot.resize(620, 260)
    plot_rect = widget.colocalization_scatter_plot._plot_rect()
    assert plot_rect.width() == plot_rect.height()
    assert widget.pipeline.nodes[coloc.id].params["threshold_mode"] == "Costes auto"
    assert (
        0
        <= widget.pipeline.nodes[coloc.id].params["channel_1_threshold"]
        <= float(data[0].max())
    )
    assert (
        0
        <= widget.pipeline.nodes[coloc.id].params["channel_2_threshold"]
        <= float(data[1].max())
    )
    assert not np.isclose(
        widget.pipeline.nodes[coloc.id].params["channel_1_threshold"],
        25.0,
    )
    threshold_1_control = widget._parameter_widgets["channel_1_threshold"]
    threshold_2_control = widget._parameter_widgets["channel_2_threshold"]
    assert not threshold_1_control.isEnabled()
    assert not threshold_2_control.isEnabled()
    assert np.isclose(
        threshold_1_control.value(),
        widget.pipeline.nodes[coloc.id].params["channel_1_threshold"],
    )
    assert np.isclose(
        threshold_2_control.value(),
        widget.pipeline.nodes[coloc.id].params["channel_2_threshold"],
    )

    widget._on_colocalization_scatter_threshold_changed(1, 12.5)
    widget._debounce_timer.stop()

    assert widget.pipeline.nodes[coloc.id].params["threshold_mode"] == "Manual"
    assert threshold_1_control.isEnabled()
    assert threshold_2_control.isEnabled()
    assert np.isclose(
        widget.pipeline.nodes[coloc.id].params["channel_1_threshold"],
        12.5,
    )
    widget.colocalization_scatter_colormap_combo.setCurrentText("Magma")

    assert widget.colocalization_scatter_plot._colormap == "Magma"


def test_scatter_node_popout_uses_independent_native_ranges(qtbot):
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    widget = VippWidget(_Viewer(data, metadata={"axes": "YX"}))
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    widget._compute_mode = ComputeMode.CPU
    qtbot.addWidget(widget)

    offset = widget.add_node_from_palette("linear_scale_offset")
    scatter = widget.add_node_from_palette("colocalization_scatter_plot")
    widget.pipeline.set_param(offset.id, "alpha", 1.0)
    widget.pipeline.set_param(offset.id, "beta", 500.0)
    widget.pipeline.set_param(scatter.id, "bins", 64)
    widget.pipeline.set_param(scatter.id, "range_percentile", 100.0)
    widget._connect_nodes("input", offset.id)
    widget._connect_nodes("input", scatter.id, target_port=0)
    widget._connect_nodes(offset.id, scatter.id, target_port=1)
    widget.run_pipeline(force_sync=True)
    widget.graph_view.select_node(scatter.id)

    plot = widget.colocalization_scatter_plot
    assert (plot._channel_1_min, plot._channel_1_max) == (0.0, 15.0)
    assert (plot._channel_2_min, plot._channel_2_max) == (500.0, 515.0)
    result = widget._colocalization_scatter_cache[
        widget._current_colocalization_scatter_key
    ]
    assert np.asarray(result.density_counts).shape == (64, 64)
    assert widget.colocalization_scatter_popout_button.isEnabled()

    qtbot.mouseClick(widget.colocalization_scatter_popout_button, Qt.LeftButton)

    dialog = widget._colocalization_scatter_dialog
    assert dialog is not None
    assert dialog.isVisible()
    assert (dialog.plot._channel_1_min, dialog.plot._channel_1_max) == (0.0, 15.0)
    assert (dialog.plot._channel_2_min, dialog.plot._channel_2_max) == (
        500.0,
        515.0,
    )
    plot_rect = dialog.plot._plot_rect()
    assert abs(dialog.plot._x_from_value(7.5, plot_rect) - plot_rect.center().x()) <= 1
    assert (
        abs(dialog.plot._y_from_value(507.5, plot_rect) - plot_rect.center().y()) <= 1
    )


def test_scatter_popout_colormap_is_linked_without_recomputing_density(
    qtbot,
    monkeypatch,
):
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    widget = VippWidget(_Viewer(data, metadata={"axes": "YX"}))
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    qtbot.addWidget(widget)

    offset = widget.add_node_from_palette("linear_scale_offset")
    scatter = widget.add_node_from_palette("colocalization_scatter_plot")
    widget.pipeline.set_param(offset.id, "alpha", 1.0)
    widget.pipeline.set_param(offset.id, "beta", 500.0)
    widget.pipeline.set_param(scatter.id, "bins", 64)
    widget.pipeline.set_param(scatter.id, "range_percentile", 100.0)
    widget._connect_nodes("input", offset.id)
    widget._connect_nodes("input", scatter.id, target_port=0)
    widget._connect_nodes(offset.id, scatter.id, target_port=1)
    widget.run_pipeline(force_sync=True)
    widget.graph_view.select_node(scatter.id)
    qtbot.mouseClick(widget.colocalization_scatter_popout_button, Qt.LeftButton)

    dialog = widget._colocalization_scatter_dialog
    assert dialog is not None
    assert dialog.isVisible()
    original_key = widget._current_colocalization_scatter_key
    original_result = widget._colocalization_scatter_cache[original_key]
    original_density = original_result.density_counts
    original_density_key = original_result.density_key
    session = widget._workflow_tabs.current
    assert session is not None
    session.mark_clean(
        widget._current_history_snapshot(),
        persistence_token=widget._workflow_tab_persistence_token(),
    )

    def fail_if_density_is_recomputed(*_args, **_kwargs):
        raise AssertionError("A colormap change must reuse the cached density.")

    monkeypatch.setattr(
        "napari_vipp._widget._prepare_colocalization_scatter_density",
        fail_if_density_is_recomputed,
    )

    def fail_if_scientific_state_is_refreshed():
        raise AssertionError("A colormap change must not refresh scientific state.")

    monkeypatch.setattr(
        widget,
        "_update_colocalization_scatter",
        fail_if_scientific_state_is_refreshed,
    )
    inspector_image = widget.colocalization_scatter_plot._image
    dialog_image = dialog.plot._image

    widget.colocalization_scatter_colormap_combo.setCurrentText("Magma")

    assert dialog.colormap_combo.currentText() == "Magma"
    assert widget.colocalization_scatter_plot._colormap == "Magma"
    assert dialog.plot._colormap == "Magma"
    assert widget.colocalization_scatter_plot._image is not inspector_image
    assert dialog.plot._image is not dialog_image

    inspector_image = widget.colocalization_scatter_plot._image
    dialog_image = dialog.plot._image
    dialog.colormap_combo.setCurrentText("Cividis")

    assert widget.colocalization_scatter_colormap_combo.currentText() == "Cividis"
    assert widget.colocalization_scatter_plot._colormap == "Cividis"
    assert dialog.plot._colormap == "Cividis"
    assert widget.colocalization_scatter_plot._image is not inspector_image
    assert dialog.plot._image is not dialog_image
    assert widget._current_colocalization_scatter_key == original_key
    assert widget._colocalization_scatter_cache[original_key] is original_result
    assert (
        widget._colocalization_scatter_density_cache[
            original_density_key
        ].density_counts
        is original_density
    )
    assert not session.dirty

    dialog.close()
    assert not dialog.isVisible()
    widget.colocalization_scatter_colormap_combo.setCurrentText("Gray")
    qtbot.mouseClick(widget.colocalization_scatter_popout_button, Qt.LeftButton)

    assert widget._colocalization_scatter_dialog is dialog
    assert dialog.isVisible()
    assert dialog.colormap_combo.currentText() == "Gray"
    assert widget.colocalization_scatter_plot._colormap == "Gray"
    assert dialog.plot._colormap == "Gray"
    assert not session.dirty


def test_colocalization_scatter_density_and_counts_are_exact_beyond_old_cap():
    size = 600_123
    indices = np.arange(size, dtype=np.uint32)
    channel_1 = (indices % 256).astype(np.float32)
    channel_2 = ((indices * 37 + 11) % 256).astype(np.float32)
    roi = indices % 11 != 0

    density, roi_voxels, colocalized_voxels, x_min, x_max, y_min, y_max = (
        _prepare_colocalization_scatter_density(
            channel_1,
            channel_2,
            threshold_1=125.0,
            threshold_2=140.0,
            roi_mask=roi,
            intensity_max=255.0,
            bins=64,
        )
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
    assert (x_min, x_max) == (0.0, 255.0)
    assert (y_min, y_max) == (0.0, 255.0)
    np.testing.assert_array_equal(density, expected_density)
    assert int(density.sum()) == roi_voxels


def test_colocalization_scatter_expands_to_native_intensity_range():
    channel_1 = np.array([-20.0, 255.0, 2_000.0], dtype=np.float32)
    channel_2 = np.array([4_000.0, 300.0, -10.0], dtype=np.float32)

    density, roi_voxels, _colocalized_voxels, x_min, x_max, y_min, y_max = (
        _prepare_colocalization_scatter_density(
            channel_1,
            channel_2,
            threshold_1=4.0,
            threshold_2=3.0,
            roi_mask=None,
            intensity_max=255.0,
            bins=32,
        )
    )

    assert (x_min, x_max) == (-20.0, 2_000.0)
    assert (y_min, y_max) == (-10.0, 4_000.0)
    assert roi_voxels == 3
    assert int(density.sum()) == roi_voxels


def test_colocalization_scatter_uses_unit_extent_for_normalized_floats():
    channel_1 = np.array([0.0, 0.25, 1.0], dtype=np.float32)
    channel_2 = np.array([0.1, 0.5, 0.75], dtype=np.float32)

    density, roi_voxels, _colocalized_voxels, x_min, x_max, y_min, y_max = (
        _prepare_colocalization_scatter_density(
            channel_1,
            channel_2,
            threshold_1=0.4,
            threshold_2=0.3,
            roi_mask=None,
            intensity_max=255.0,
            bins=32,
        )
    )

    assert (x_min, x_max) == (0.0, 1.0)
    assert np.allclose((y_min, y_max), (0.1, 0.75))
    assert int(density.sum()) == roi_voxels == 3


def test_colocalization_scatter_percentile_clips_density_not_exact_counts():
    channel_1 = np.asarray([0.0, 1.0, 2.0, 3.0, 10_000.0])
    channel_2 = np.asarray([-10_000.0, 100.0, 101.0, 102.0, 103.0])

    density, roi_voxels, colocalized_voxels, *_ranges = (
        _prepare_colocalization_scatter_density(
            channel_1,
            channel_2,
            threshold_1=0.0,
            threshold_2=0.0,
            roi_mask=None,
            intensity_max=255.0,
            bins=32,
            range_percentile=80.0,
        )
    )

    assert roi_voxels == 5
    assert colocalized_voxels == 4
    assert int(density.sum()) < roi_voxels


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


def test_colocalization_scatter_keeps_density_during_threshold_scrubbing(
    qtbot,
    monkeypatch,
):
    data = np.zeros((100, 100), dtype=np.uint8)
    data.ravel()[::2] = 200
    viewer = _Viewer(data, metadata={"axes": "YX"})
    widget = VippWidget(viewer)
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    qtbot.addWidget(widget)
    coloc = widget.add_node_from_palette("racc_index")
    widget.pipeline.set_param(coloc.id, "threshold_mode", "Manual")
    widget.pipeline.set_param(coloc.id, "channel_1_threshold", 100.0)
    widget.pipeline.set_param(coloc.id, "channel_2_threshold", 100.0)
    widget._connect_nodes("input", coloc.id, target_port=0)
    widget._connect_nodes("input", coloc.id, target_port=1)
    widget.run_pipeline(force_sync=True)
    widget.graph_view.select_node(coloc.id)
    qtbot.mouseClick(widget.colocalization_scatter_popout_button, Qt.LeftButton)

    density_image = widget.colocalization_scatter_plot._image
    density_counts = widget.colocalization_scatter_plot._density_counts
    density_key = widget._displayed_colocalization_scatter_density_key
    dialog = widget._colocalization_scatter_dialog
    assert dialog is not None and dialog.isVisible()
    dialog_density_counts = dialog._density_counts
    assert density_image is not None
    assert density_counts is not None
    assert dialog_density_counts is not None
    assert density_key is not None
    assert widget.colocalization_scatter_plot._summary == (
        "Exact: 5,000/10,000 (50.0%)"
    )

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

    widget._on_colocalization_scatter_threshold_changed(1, 150.0)
    widget._debounce_timer.stop()
    qtbot.waitUntil(started.is_set, timeout=5_000)
    active_run_id = widget._active_colocalization_scatter_run_id

    widget._on_colocalization_scatter_threshold_changed(1, 210.0)
    widget._debounce_timer.stop()

    try:
        assert widget._active_colocalization_scatter_run_id == active_run_id
        assert widget._active_colocalization_scatter_cancel_event.is_set()
        assert widget._pending_colocalization_scatter_request is not None
        assert widget._pending_colocalization_scatter_request.threshold_1 == 210.0
        assert widget.colocalization_scatter_plot._image is density_image
        assert widget._displayed_colocalization_scatter_density_key == density_key
        assert widget.colocalization_scatter_plot._threshold_1 == 210.0
        assert (
            widget.colocalization_scatter_plot._summary == "Calculating exact count..."
        )
        assert "Calculating the exact" in (widget.colocalization_scatter_summary.text())

        pending_request = widget._pending_colocalization_scatter_request
        pending_cancel_event = widget._active_colocalization_scatter_cancel_event
        inspector_image = widget.colocalization_scatter_plot._image
        dialog_image = dialog.plot._image
        dialog.colormap_combo.setCurrentText("Gray")

        assert widget.colocalization_scatter_colormap_combo.currentText() == "Gray"
        assert widget.colocalization_scatter_plot._colormap == "Gray"
        assert dialog.plot._colormap == "Gray"
        assert widget.colocalization_scatter_plot._image is not inspector_image
        assert widget.colocalization_scatter_plot._image != inspector_image
        assert dialog.plot._image is not dialog_image
        assert dialog.plot._image != dialog_image
        assert widget.colocalization_scatter_plot._density_counts is density_counts
        assert dialog._density_counts is dialog_density_counts
        assert widget._active_colocalization_scatter_run_id == active_run_id
        assert (
            widget._active_colocalization_scatter_cancel_event is pending_cancel_event
        )
        assert widget._pending_colocalization_scatter_request is pending_request
        assert widget._displayed_colocalization_scatter_density_key == density_key
        assert (
            widget.colocalization_scatter_plot._summary == "Calculating exact count..."
        )
    finally:
        release.set()

    qtbot.waitUntil(
        lambda: widget._active_colocalization_scatter_run_id is None,
        timeout=5_000,
    )
    assert widget._pending_colocalization_scatter_request is None
    assert widget.colocalization_scatter_plot._image is not None
    assert widget.colocalization_scatter_plot._summary == ("Exact: 0/10,000 (0.0%)")
    assert "Exact colocalized count: 0/10,000 (0.0%)" in (
        widget.colocalization_scatter_summary.text()
    )


def test_scatter_popout_survives_real_debounced_threshold_pipeline_run(
    qtbot,
    monkeypatch,
):
    data = np.zeros((100, 100), dtype=np.uint8)
    data.ravel()[::2] = 200
    widget = VippWidget(_Viewer(data, metadata={"axes": "YX"}))
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
    qtbot.mouseClick(widget.colocalization_scatter_popout_button, Qt.LeftButton)

    dialog = widget._colocalization_scatter_dialog
    density_key = widget._displayed_colocalization_scatter_density_key
    assert dialog is not None and dialog.isVisible()
    assert density_key is not None
    density = widget._colocalization_scatter_density_cache[density_key].density_counts
    assert dialog._colocalized_voxels == 5_000

    finish_calls = []
    real_finish = widget._finish_pipeline_update

    def tracked_finish(primary_layer, source_label):
        real_finish(primary_layer, source_label)
        finish_calls.append(True)

    density_recomputations = 0

    def reject_density_recomputation(*_args, **_kwargs):
        nonlocal density_recomputations
        density_recomputations += 1
        raise AssertionError("compatible threshold edits must reuse density")

    monkeypatch.setattr(widget, "_finish_pipeline_update", tracked_finish)
    monkeypatch.setattr(
        "napari_vipp._widget._prepare_colocalization_scatter_density",
        reject_density_recomputation,
    )

    dialog._on_threshold_changed(1, 210.0)

    assert widget._debounce_timer.isActive()
    qtbot.waitUntil(
        lambda: (
            bool(finish_calls)
            and not widget._debounce_timer.isActive()
            and widget._active_pipeline_run_id is None
        ),
        timeout=5_000,
    )

    assert density_recomputations == 0
    assert dialog.isVisible()
    assert widget._colocalization_scatter_dialog is dialog
    assert widget._displayed_colocalization_scatter_density_key == density_key
    assert (
        widget._colocalization_scatter_density_cache[density_key].density_counts
        is density
    )
    result = widget._colocalization_scatter_cache[
        widget._current_colocalization_scatter_key
    ]
    assert result.colocalized_voxels == 0
    assert result.roi_voxels == 10_000
    assert dialog._colocalized_voxels == 0
    assert "Exact: 0/10,000 (0.0%)" in dialog.summary_label.text()


def test_colocalization_scatter_clears_density_for_different_inputs(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget.colocalization_scatter_plot.set_density(
        np.ones((16, 16), dtype=np.float64),
        threshold_1=25.0,
        threshold_2=25.0,
    )
    widget._displayed_colocalization_scatter_density_key = ("old-inputs",)
    monkeypatch.setattr(
        widget,
        "_start_colocalization_scatter_request",
        lambda request: None,
    )

    widget._queue_colocalization_scatter(
        ColocalizationScatterRequest(
            0,
            ("new-result",),
            "coloc",
            (np.zeros((2, 2)), np.zeros((2, 2))),
            "Manual",
            40.0,
            50.0,
            density_key=("new-inputs",),
        )
    )

    assert widget.colocalization_scatter_plot._image is None
    assert widget._displayed_colocalization_scatter_density_key is None
    assert widget.colocalization_scatter_plot._summary == ("Calculating exact count...")


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


def test_costes_scatter_explains_too_small_racc_population(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    coloc = widget.add_node_from_palette("masked_racc_index")
    widget.graph_view.select_node(coloc.id)
    key = ("degenerate-costes",)
    widget._current_colocalization_scatter_key = key

    widget._apply_colocalization_scatter_result(
        ColocalizationScatterResult(
            0,
            key,
            coloc.id,
            "Costes auto",
            48_409.0,
            61_092.0,
            density_counts=np.ones((32, 32), dtype=np.float64),
            roi_voxels=2_311,
            colocalized_voxels=1,
        )
    )

    summary = widget.colocalization_scatter_summary.text()
    tooltip = widget.colocalization_scatter_summary.toolTip()
    assert "Costes diagnostic" in summary
    assert "usable jointly threshold-positive population" in summary
    assert "RACC is unavailable" in summary
    assert "spatial overlap or co-occurrence" in summary
    assert "Review the scatter and resolved thresholds" in summary
    assert "switch to Manual" in summary
    assert "RACC is unavailable" in tooltip
    assert "channel-neutral ROI" not in summary
    assert "weakly or negatively correlated" not in summary


def test_clipped_scatter_summary_separates_density_from_exact_counts(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    coloc = widget.add_node_from_palette("colocalized_voxels")
    widget.graph_view.select_node(coloc.id)
    key = ("clipped-density",)
    widget._current_colocalization_scatter_key = key
    density = np.zeros((10, 10), dtype=np.float64)
    density.ravel()[:80] = 1.0

    widget._apply_colocalization_scatter_result(
        ColocalizationScatterResult(
            0,
            key,
            coloc.id,
            "Manual",
            25.0,
            25.0,
            density_counts=density,
            roi_voxels=100,
            colocalized_voxels=30,
            range_percentile=90.0,
        )
    )

    summary = widget.colocalization_scatter_summary.text()
    assert "Exact colocalized count: 30/100 (30.0%)" in summary
    assert "Visible scatter density contains 80/100 ROI voxels" in summary
    assert "20 tail voxels are hidden by the 90% display range" in summary
    assert "display clip does not change exact counts or metrics" in summary


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


def test_colocalization_scatter_reuses_ready_sibling_analysis(qtbot, monkeypatch):
    data = np.arange(256, dtype=np.float32).reshape(16, 16)
    widget = VippWidget(_Viewer(data, metadata={"axes": "YX"}))
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    qtbot.addWidget(widget)
    first = widget.add_node_from_palette("colocalized_voxels")
    second = widget.add_node_from_palette("colocalized_voxels")
    for node in (first, second):
        widget._connect_nodes("input", node.id, target_port=0)
        widget._connect_nodes("input", node.id, target_port=1)
        widget.pipeline.set_param(node.id, "threshold_mode", "Costes auto")
    widget.pipeline.set_param(first.id, "channel_1_threshold", 40.0)
    widget.pipeline.set_param(first.id, "channel_2_threshold", 50.0)
    widget.pipeline.node_execution_states[first.id] = EXECUTION_READY
    widget.pipeline.node_execution_states[second.id] = EXECUTION_STALE
    widget.pipeline.set_param(second.id, "channel_1_threshold", -101.0)
    widget.pipeline.set_param(second.id, "channel_2_threshold", -202.0)
    widget.pipeline.set_param(second.id, "channel_1_color", "Blue")
    widget.pipeline.set_param(second.id, "channel_2_color", "Yellow")
    widget.graph_view.select_node("input")
    widget._clear_colocalization_scatter_cache()

    import napari_vipp._widget as widget_module

    density_calls = 0
    real_density = widget_module._prepare_colocalization_scatter_density

    def tracked_density(*args, **kwargs):
        nonlocal density_calls
        density_calls += 1
        return real_density(*args, **kwargs)

    monkeypatch.setattr(
        widget_module,
        "_prepare_colocalization_scatter_density",
        tracked_density,
    )
    monkeypatch.setattr(
        widget_module,
        "colocalization_threshold_values",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("READY Costes thresholds must be reused")
        ),
    )

    widget.graph_view.select_node(first.id)
    shared_key = widget._current_colocalization_scatter_key
    cached = widget._colocalization_scatter_cache[shared_key]
    widget.graph_view.select_node(second.id)

    assert widget._current_colocalization_scatter_key == shared_key
    assert widget._colocalization_scatter_cache[shared_key] is cached
    assert density_calls == 1
    assert widget.pipeline.nodes[second.id].params["channel_1_threshold"] == 40.0
    assert widget.pipeline.nodes[second.id].params["channel_2_threshold"] == 50.0
    assert widget.colocalization_scatter_plot._channel_1_color.name() == "#0000ff"
    assert widget.colocalization_scatter_plot._channel_2_color.name() == "#ffff00"


def test_unresolved_costes_scatter_queues_once_and_hands_off(qtbot, monkeypatch):
    data = np.arange(256, dtype=np.float32).reshape(16, 16)
    widget = VippWidget(_Viewer(data, metadata={"axes": "YX"}))
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    qtbot.addWidget(widget)
    first = widget.add_node_from_palette("colocalized_voxels")
    second = widget.add_node_from_palette("colocalized_voxels")
    for node in (first, second):
        widget._connect_nodes("input", node.id, target_port=0)
        widget._connect_nodes("input", node.id, target_port=1)
        widget.pipeline.set_param(node.id, "threshold_mode", "Costes auto")
        widget.pipeline.node_execution_states[node.id] = EXECUTION_STALE
    widget._clear_colocalization_scatter_cache()
    pool = _QueuedThreadPool()
    widget._pipeline_thread_pool = pool
    progress_values = []

    def resolved_in_worker(*_args, progress=None, **_kwargs):
        progress_values.append(progress)
        return 41.0, 51.0

    monkeypatch.setattr(
        "napari_vipp._widget.colocalization_threshold_values",
        resolved_in_worker,
    )
    widget._selected_node_id = first.id
    widget._update_colocalization_scatter()
    shared_key = widget._current_colocalization_scatter_key
    run_id = widget._active_colocalization_scatter_run_id
    assert len(pool.workers) == 1
    assert not pool.workers[0].request.thresholds_resolved

    widget._selected_node_id = second.id
    widget._update_colocalization_scatter()

    assert widget._current_colocalization_scatter_key == shared_key
    assert widget._active_colocalization_scatter_run_id == run_id
    assert widget._pending_colocalization_scatter_request is None
    assert len(pool.workers) == 1

    pool.workers[0].run()

    assert progress_values and progress_values[0] is not None
    assert widget._active_colocalization_scatter_run_id is None
    assert widget.pipeline.nodes[second.id].params["channel_1_threshold"] == 41.0
    assert widget.pipeline.nodes[second.id].params["channel_2_threshold"] == 51.0
    assert widget.colocalization_scatter_plot._image is not None


def test_ready_costes_sibling_marks_background_request_resolved(qtbot, monkeypatch):
    widget = VippWidget(_Viewer(np.arange(16).reshape(4, 4)))
    qtbot.addWidget(widget)
    ready = widget.add_node_from_palette("colocalized_voxels")
    selected = widget.add_node_from_palette("racc_index")
    for node in (ready, selected):
        widget._connect_nodes("input", node.id, target_port=0)
        widget._connect_nodes("input", node.id, target_port=1)
        widget.pipeline.set_param(node.id, "threshold_mode", "Costes auto")
    widget.pipeline.set_param(ready.id, "channel_1_threshold", 61.0)
    widget.pipeline.set_param(ready.id, "channel_2_threshold", 71.0)
    widget.pipeline.node_execution_states[ready.id] = EXECUTION_READY
    widget.pipeline.node_execution_states[selected.id] = EXECUTION_NOT_CALCULATED
    widget._clear_colocalization_scatter_cache()
    pool = _QueuedThreadPool()
    widget._pipeline_thread_pool = pool
    monkeypatch.setattr(
        "napari_vipp._widget.colocalization_scatter_requires_background",
        lambda _bins: True,
    )
    monkeypatch.setattr(
        "napari_vipp._widget.colocalization_threshold_values",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("compatible READY sibling must skip Costes")
        ),
    )

    widget._selected_node_id = selected.id
    widget._update_colocalization_scatter()

    assert len(pool.workers) == 1
    request = pool.workers[0].request
    assert request.thresholds_resolved
    assert (request.threshold_1, request.threshold_2) == (61.0, 71.0)

    pool.workers[0].run()

    result = widget._colocalization_scatter_cache[
        widget._current_colocalization_scatter_key
    ]
    assert (result.threshold_1, result.threshold_2) == (61.0, 71.0)


def test_active_pipeline_defers_unresolved_costes_and_blocks_writeback(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer(np.arange(16).reshape(4, 4)))
    qtbot.addWidget(widget)
    coloc = widget.add_node_from_palette("colocalized_voxels")
    widget._connect_nodes("input", coloc.id, target_port=0)
    widget._connect_nodes("input", coloc.id, target_port=1)
    widget.pipeline.set_param(coloc.id, "threshold_mode", "Costes auto")
    widget.pipeline.set_param(coloc.id, "channel_1_threshold", 12.0)
    widget.pipeline.set_param(coloc.id, "channel_2_threshold", 13.0)
    widget.pipeline.node_execution_states[coloc.id] = EXECUTION_RUNNING
    widget._clear_colocalization_scatter_cache()
    widget._selected_node_id = coloc.id
    pool = _QueuedThreadPool()
    widget._pipeline_thread_pool = pool
    control_updates = []
    monkeypatch.setattr(
        widget,
        "_set_parameter_control_value",
        lambda *args: control_updates.append(args),
    )
    workflow = serialize_workflow(
        widget.pipeline,
        compute_request=widget._current_compute_request(),
    )

    widget._update_colocalization_scatter()

    assert len(pool.workers) == 1
    cancel_event = pool.workers[0].request.cancel_event
    assert cancel_event is not None and not cancel_event.is_set()

    widget._active_pipeline_run_id = 99
    widget._update_colocalization_scatter()

    assert cancel_event.is_set()
    assert widget._active_colocalization_scatter_run_id is None
    assert "Waiting for the active pipeline" in (
        widget.colocalization_scatter_summary.text()
    )
    key = widget._current_colocalization_scatter_key
    widget._apply_colocalization_scatter_result(
        ColocalizationScatterResult(
            0,
            key,
            coloc.id,
            "Costes auto",
            80.0,
            90.0,
            density_counts=np.ones((32, 32), dtype=np.float64),
            roi_voxels=16,
            colocalized_voxels=4,
        )
    )

    assert widget.pipeline.nodes[coloc.id].params["channel_1_threshold"] == 12.0
    assert widget.pipeline.nodes[coloc.id].params["channel_2_threshold"] == 13.0
    assert control_updates == []
    assert widget._workflow_matches_current_pipeline(workflow)
    assert widget.colocalization_scatter_plot._threshold_1 == 80.0
    assert widget.colocalization_scatter_plot._threshold_2 == 90.0


def test_colocalization_scatter_keys_separate_masked_domain(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    channel_1 = np.zeros((4, 4), dtype=np.float32)
    channel_2 = np.ones((4, 4), dtype=np.float32)
    roi_mask = np.eye(4, dtype=bool)
    common = {
        "threshold_mode": "Costes auto",
        "threshold_1": 25.0,
        "threshold_2": 25.0,
        "intensity_max": 255.0,
        "bins": 64,
        "range_percentile": 100.0,
    }
    unmasked = widget._colocalization_scatter_key(
        [channel_1, channel_2],
        **common,
    )
    masked = widget._colocalization_scatter_key(
        [channel_1, channel_2, roi_mask],
        **common,
    )
    unmasked_density = widget._colocalization_scatter_density_key(
        [channel_1, channel_2],
        intensity_max=255.0,
        bins=64,
        range_percentile=100.0,
    )
    masked_density = widget._colocalization_scatter_density_key(
        [channel_1, channel_2, roi_mask],
        intensity_max=255.0,
        bins=64,
        range_percentile=100.0,
    )

    assert masked != unmasked
    assert masked_density != unmasked_density


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


def test_4096_bin_node_queues_capped_inspector_density_with_visible_notice(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    scatter = widget.add_node_from_palette("colocalization_scatter_plot")
    widget.pipeline.set_param(scatter.id, "bins", 4_096)
    widget.graph_view.select_node(scatter.id)
    widget._clear_colocalization_scatter_cache()
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    monkeypatch.setattr(
        widget,
        "_colocalization_inputs_for_node",
        lambda _node_id: [data, data],
    )
    monkeypatch.setattr(
        widget,
        "_start_colocalization_scatter_request",
        lambda _request: None,
    )
    queued = []
    original_queue = widget._queue_colocalization_scatter

    def record_queue(request):
        queued.append(request)
        original_queue(request)

    monkeypatch.setattr(widget, "_queue_colocalization_scatter", record_queue)

    widget._update_colocalization_scatter()

    assert len(queued) == 1
    assert queued[0].bins == 1_024
    assert scatter.params["bins"] == 4_096
    assert "capped at 1,024 x 1,024 bins" in (
        widget.colocalization_scatter_summary.text()
    )
    assert "graph operation keeps its requested 4,096-bin histogram" in (
        widget.colocalization_scatter_summary.toolTip()
    )


def test_scatter_cache_shares_density_across_threshold_results(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    density_key = ("same-inputs", 64)
    cached_results = []

    for index in range(16):
        density = np.full((64, 64), index + 1.0, dtype=np.float64)
        cached_results.append(
            widget._cache_colocalization_scatter_result(
                ColocalizationScatterResult(
                    run_id=index,
                    key=("threshold", index),
                    node_id="scatter",
                    threshold_mode="Manual",
                    threshold_1=float(index),
                    threshold_2=25.0,
                    density_counts=density,
                    roi_voxels=100,
                    colocalized_voxels=50,
                    density_key=density_key,
                    channel_1_min=0.0,
                    channel_1_max=255.0,
                    channel_2_min=0.0,
                    channel_2_max=255.0,
                )
            )
        )

    retained = {
        id(result.density_counts)
        for result in widget._colocalization_scatter_cache.values()
    }
    assert len(widget._colocalization_scatter_density_cache) == 1
    assert len(retained) == 1
    assert all(
        result.density_counts is cached_results[0].density_counts
        for result in cached_results
    )
    assert widget._colocalization_scatter_density_cache_bytes() == 64 * 64 * 8


def test_scatter_density_cache_enforces_explicit_byte_budget(qtbot, monkeypatch):
    import napari_vipp._widget as widget_module

    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    monkeypatch.setattr(
        widget_module,
        "COLOCALIZATION_SCATTER_CACHE_BUDGET_BYTES",
        600,
    )

    for index in range(2):
        widget._cache_colocalization_scatter_result(
            ColocalizationScatterResult(
                run_id=index,
                key=("result", index),
                node_id="scatter",
                threshold_mode="Manual",
                threshold_1=25.0,
                threshold_2=25.0,
                density_counts=np.ones((8, 8), dtype=np.float64),
                density_key=("density", index),
                channel_1_min=0.0,
                channel_1_max=255.0,
                channel_2_min=0.0,
                channel_2_max=255.0,
            )
        )

    assert list(widget._colocalization_scatter_density_cache) == [("density", 1)]
    assert list(widget._colocalization_scatter_cache) == [("result", 1)]
    assert widget._colocalization_scatter_density_cache_bytes() == 512


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
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
            AxisMetadata("rgb", "channel"),
        ),
    )

    counts, _x_range, colors = _histogram_summary(
        data,
        state=state,
        scope="Stack",
    )

    assert counts is not None
    assert counts.shape[0] == 3
    np.testing.assert_array_equal(counts.sum(axis=1), [0, 20, 20])
    assert colors is not None
    assert [color.name() for color in colors] == ["#ef4444", "#22c55e", "#60a5fa"]


def test_histogram_does_not_infer_channels_from_trailing_axis_size():
    data = np.arange(60, dtype=np.float32).reshape(4, 5, 3)

    counts, _x_range, colors = _histogram_summary(data, scope="Stack")

    assert counts is not None and counts.ndim == 1
    assert int(counts.sum()) == data.size
    assert colors is not None and len(colors) == 1


@pytest.mark.parametrize("x_size", [3, 4])
def test_histogram_does_not_treat_explicit_zyx_x_as_channels(x_size):
    data = np.arange(5 * 7 * x_size, dtype=np.float32).reshape(5, 7, x_size)
    state = image_state_from_array(data, layer_metadata={"axes": "ZYX"})

    counts, _x_range, colors = _histogram_summary(
        data,
        state=state,
        scope="Stack",
    )

    assert state is not None and state.axis_confidence == "explicit"
    assert counts is not None and counts.ndim == 1
    assert int(counts.sum()) == data.size
    assert colors is not None and len(colors) == 1


@pytest.mark.parametrize("channel_count", [3, 4])
def test_histogram_uses_explicit_yxc_channel_axis(channel_count):
    data = np.arange(7 * 9 * channel_count, dtype=np.float32).reshape(
        7,
        9,
        channel_count,
    )
    state = image_state_from_array(data, layer_metadata={"axes": "YXC"})

    counts, _x_range, colors = _histogram_summary(
        data,
        state=state,
        scope="Stack",
    )

    assert state is not None and state.axis_confidence == "explicit"
    assert counts is not None and counts.shape[0] == channel_count
    assert counts.sum(axis=1).tolist() == [63] * channel_count
    assert colors is not None and len(colors) == channel_count


@pytest.mark.parametrize("metadata", [None, {"axes": "ZYX"}, {"axes": "YXC"}])
def test_automatic_threshold_marker_defaults_to_scalar(
    monkeypatch,
    metadata,
):
    data = np.arange(7 * 9 * 3, dtype=np.float32).reshape(7, 9, 3)
    state = image_state_from_array(data, layer_metadata=metadata)
    observed = []

    def recorded_threshold(
        values,
        operation_id,
        histogram_bins=256,
        max_iterations=10_000,
        progress=None,
        channel_axis=None,
    ):
        observed.append(channel_axis)
        return 1.0

    monkeypatch.setattr(
        "napari_vipp._widget.automatic_threshold_value",
        recorded_threshold,
    )

    markers = _input_histogram_markers(
        "otsu_threshold",
        data,
        state=state,
        scope="Stack histogram",
        params={"histogram_bins": 256},
    )

    assert observed == [None]
    assert markers[0][1] == 1.0


def test_automatic_threshold_scalar_slice_uses_operation_trailing_plane(
    monkeypatch,
):
    data = np.arange(7 * 9 * 3, dtype=np.float32).reshape(7, 9, 3)
    state = image_state_from_array(data, layer_metadata={"axes": "YXC"})
    observed = []

    def recorded_threshold(
        values,
        operation_id,
        histogram_bins=256,
        max_iterations=10_000,
        progress=None,
        channel_axis=None,
    ):
        observed.append((np.asarray(values).shape, channel_axis))
        return 1.0

    monkeypatch.setattr(
        "napari_vipp._widget.automatic_threshold_value",
        recorded_threshold,
    )

    _input_histogram_markers(
        "otsu_threshold",
        data,
        state=state,
        scope="Slice histogram",
        current_step=(4, 0, 0),
        current_step_nsteps=data.shape,
        params={"histogram_bins": 256, "channel_axis": -1},
    )

    assert observed == [((9, 3), None)]


@pytest.mark.parametrize(
    ("scope", "expected_shape", "expected_channel_axis"),
    [
        ("Stack histogram", (2, 7, 9, 3), 3),
        ("Slice histogram", (7, 9, 3), 2),
    ],
)
def test_automatic_threshold_marker_remaps_persisted_numeric_channel_axis(
    monkeypatch,
    scope,
    expected_shape,
    expected_channel_axis,
):
    data = np.arange(2 * 7 * 9 * 3, dtype=np.float32).reshape(2, 7, 9, 3)
    state = image_state_from_array(data)
    observed = []

    def recorded_threshold(
        values,
        operation_id,
        histogram_bins=256,
        max_iterations=10_000,
        progress=None,
        channel_axis=None,
    ):
        observed.append((np.asarray(values).shape, channel_axis))
        return 1.0

    monkeypatch.setattr(
        "napari_vipp._widget.automatic_threshold_value",
        recorded_threshold,
    )

    _input_histogram_markers(
        "otsu_threshold",
        data,
        state=state,
        scope=scope,
        current_step=(1, 0, 0, 0),
        current_step_nsteps=data.shape,
        params={"histogram_bins": 256, "channel_axis": 3},
    )

    assert state.axis_order == "ZYXC"
    assert state.axis_confidence == "shape-inferred"
    assert observed == [(expected_shape, expected_channel_axis)]


def test_global_threshold_marker_cache_key_includes_channel_axis():
    scalar_key = _input_histogram_marker_key(
        "otsu_threshold",
        {"histogram_bins": 256, "channel_axis": -1},
    )
    channel_key = _input_histogram_marker_key(
        "otsu_threshold",
        {"histogram_bins": 256, "channel_axis": 2},
    )

    assert scalar_key != channel_key


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
    assert widget.rescale_input_histogram_plot._draggable_markers == {
        "low",
        "high",
    }

    widget.rescale_input_histogram_log_checkbox.setChecked(True)

    assert widget.rescale_input_histogram_plot._log_scale

    widget.graph_view.select_node("input")

    assert widget.rescale_input_histogram_group.isHidden()
    assert widget.histogram_group.title() == "Output Histogram"


@pytest.mark.parametrize(
    "operation_id",
    [
        "linear_scale_offset",
        "gamma_correction",
        "normalize_image",
    ],
)
def test_intensity_contrast_nodes_show_input_and_output_histograms(
    qtbot,
    operation_id,
):
    data = np.arange(2 * 10 * 10, dtype=np.uint8).reshape(2, 10, 10)
    widget = VippWidget(_Viewer(data, metadata={"axes": "ZYX"}))
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette(operation_id)
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    assert not widget.rescale_input_histogram_group.isHidden()
    assert not widget.histogram_group.isHidden()
    assert not widget.rescale_input_histogram_scope_row.isHidden()
    assert widget.rescale_input_histogram_plot._counts.sum() == 100.0
    assert widget.histogram_plot._counts.sum() == 100.0
    assert widget.rescale_input_histogram_plot._markers == []
    assert widget.rescale_input_histogram_plot._draggable_markers == set()

    widget.rescale_input_histogram_scope_combo.setCurrentText("Stack histogram")

    assert widget.rescale_input_histogram_plot._counts.sum() == 200.0
    assert widget.histogram_plot._counts.sum() == 100.0


def test_intensity_contrast_histogram_membership_follows_palette_category():
    expected = {
        spec.id
        for spec in PALETTE_NODE_LIBRARY
        if spec.category == "Intensity & Contrast"
        and spec.input_type == "array"
        and spec.output_type == "image"
    }

    assert INTENSITY_CONTRAST_HISTOGRAM_OPERATIONS == expected
    assert expected == {
        "clip_intensity",
        "gamma_correction",
        "linear_scale_offset",
        "normalize_image",
        "rescale_intensity",
    }


def test_intensity_contrast_input_histogram_reuses_cached_distribution(
    qtbot,
    monkeypatch,
):
    data = np.arange(100, dtype=np.uint16).reshape(10, 10)
    widget = VippWidget(_Viewer(data, metadata={"axes": "YX"}))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("linear_scale_offset")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    assert widget._input_histogram_distribution_cache

    def unexpected_rescan(*_args, **_kwargs):
        raise AssertionError("the unchanged input distribution was rescanned")

    monkeypatch.setattr(
        widget,
        "_calculate_input_histogram_distribution",
        unexpected_rescan,
    )
    widget._update_rescale_input_histogram(node.id, widget._current_step())

    assert widget.rescale_input_histogram_plot._counts.sum() == data.size


def test_generic_intensity_histogram_background_request_is_cancellable(
    qtbot,
    monkeypatch,
):
    data = np.arange(200, dtype=np.float32).reshape(2, 10, 10)
    widget = VippWidget(_Viewer(data, metadata={"axes": "ZYX"}))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("normalize_image")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_BYTES", 1)
    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_ELEMENTS", 1)
    widget._clear_input_histogram_cache()
    widget._update_rescale_input_histogram(node.id, widget._current_step())

    assert widget._active_input_histogram_run_id is not None
    cancel_event = widget._active_input_histogram_cancel_event
    assert cancel_event is not None

    widget._clear_input_histogram_cache()

    assert cancel_event.is_set()
    assert widget._active_input_histogram_run_id is None
    assert widget._current_input_histogram_key is None


def test_rescale_percentile_histogram_drag_switches_to_persisted_values(qtbot):
    data = np.arange(256, dtype=np.uint8).reshape(1, 16, 16)
    widget = VippWidget(_Viewer(data))
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("rescale_intensity")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)
    plot = widget.rescale_input_histogram_plot
    plot.resize(320, 160)
    plot.show()

    rect = plot._plot_rect()
    start = QPoint(rect.left(), rect.center().y())
    end = QPoint(rect.left() + int(round(0.25 * rect.width())), rect.center().y())
    qtbot.mousePress(plot, Qt.LeftButton, pos=start)
    qtbot.mouseMove(plot, pos=end)
    qtbot.mouseRelease(plot, Qt.LeftButton, pos=end)

    assert node.params["cutoff_mode"] == "Values"
    assert node.params["in_low_value"] == pytest.approx(64.0, abs=1.0)
    assert node.params["in_high_value"] == 255.0
    assert "in_low_value" in widget._parameter_widgets
    assert "in_low_percentile" not in widget._parameter_widgets
    qtbot.waitUntil(
        lambda: (
            widget.pipeline.outputs[node.id] is not None
            and int(np.asarray(widget.pipeline.outputs[node.id]).reshape(-1)[32]) == 0
        ),
        timeout=5_000,
    )
    widget.graph_view.select_node("input")
    widget.graph_view.select_node(node.id)
    assert node.params["cutoff_mode"] == "Values"
    assert widget._parameter_widgets["in_low_value"].value() == pytest.approx(
        64.0,
        abs=1.0,
    )


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
        (
            "hysteresis_threshold",
            {"low_threshold": 90.0, "high_threshold": 10.0},
            "Hysteresis low threshold must not exceed",
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

    widget._parameter_widgets["minimum"].value_box.setValue(25)

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


def test_binary_threshold_histogram_drag_persists_parameter(qtbot):
    data = np.arange(256, dtype=np.uint8).reshape(1, 16, 16)
    widget = VippWidget(_Viewer(data))
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("binary_threshold")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)
    plot = widget.rescale_input_histogram_plot
    plot.resize(320, 160)
    plot.show()

    rect = plot._plot_rect()
    start = QPoint(
        rect.left()
        + int(round(plot._x_fraction(node.params["threshold"]) * rect.width())),
        rect.center().y(),
    )
    end = QPoint(rect.left() + int(round(0.75 * rect.width())), rect.center().y())
    qtbot.mousePress(plot, Qt.LeftButton, pos=start)
    qtbot.mouseMove(plot, pos=end)
    qtbot.mouseRelease(plot, Qt.LeftButton, pos=end)

    assert node.params["threshold"] == pytest.approx(191.0, abs=1.0)
    widget.graph_view.select_node("input")
    widget.graph_view.select_node(node.id)
    assert node.params["threshold"] == pytest.approx(191.0, abs=1.0)
    assert widget._parameter_widgets["threshold"].value() == pytest.approx(
        191.0,
        abs=1.0,
    )


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


@pytest.mark.parametrize(
    ("operation_id", "params", "message"),
    [
        ("binary_threshold", {"threshold": np.nan}, "finite number"),
        (
            "hysteresis_threshold",
            {"low_threshold": 10.0, "high_threshold": np.inf},
            "finite number",
        ),
    ],
)
def test_threshold_markers_reject_nonfinite_values(operation_id, params, message):
    with pytest.raises(ValueError, match=message):
        _input_histogram_markers(
            operation_id,
            np.arange(16, dtype=np.float32).reshape(4, 4),
            params=params,
        )


def test_threshold_bounds_do_not_infer_rgb_from_trailing_axis_size(qtbot):
    data = np.array(
        [[[0.0, 0.0, 100.0], [0.0, 100.0, 0.0]]],
        dtype=np.float32,
    )
    widget = VippWidget(_Viewer(data))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("binary_threshold")
    widget._connect_nodes("input", node.id)
    threshold_spec = next(
        spec
        for spec in widget.pipeline.node_parameter_specs(node.id)
        if spec.name == "threshold"
    )

    scalar_bounds = widget._threshold_bounds(node.id, threshold_spec)
    widget.pipeline.set_param(node.id, "channel_axis", 2)
    color_bounds = widget._threshold_bounds(node.id, threshold_spec)

    assert scalar_bounds.maximum == 100.0
    np.testing.assert_allclose(color_bounds.maximum, 58.7, atol=1e-5)


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
    history = widget.history_label.text()
    assert "1. Composite \u2192 RGB: c axis (1)" in history
    assert "native intensity scale retained" in history
    assert "no normalization or clipping" in history


@pytest.mark.parametrize(
    ("metadata", "expected_axis", "expected_rgb"),
    [
        (None, None, False),
        ({"axes": "ZYX"}, None, False),
        ({"axes": "YXC"}, 2, False),
    ],
)
def test_channel_axis_and_rgb_presentation_require_explicit_semantics(
    qtbot,
    metadata,
    expected_axis,
    expected_rgb,
):
    data = np.zeros((7, 9, 3), dtype=np.uint16)
    viewer = _Viewer(data, metadata=metadata)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    extract = widget.add_node_from_palette("extract_channel")
    widget._connect_nodes("input", extract.id)

    assert widget._preferred_channel_axis(extract.id) == expected_axis
    assert widget._selected_channel_axis(extract.id, data) == expected_axis
    assert widget._display_rgb(data, "input") is expected_rgb


def test_explicit_numeric_channel_axis_is_used_without_shape_fallback(qtbot):
    data = np.zeros((7, 9, 3), dtype=np.uint16)
    widget = VippWidget(_Viewer(data))
    qtbot.addWidget(widget)
    composite = widget.add_node_from_palette("composite_to_rgb")

    widget.pipeline.set_param(composite.id, "channel_axis_mode", "Manual")
    widget.pipeline.set_param(composite.id, "channel_axis", 0)
    widget._connect_nodes("input", composite.id)

    assert widget._selected_channel_axis(composite.id, data) == 0

    widget.pipeline.set_param(composite.id, "channel_axis", 99)

    assert widget._selected_channel_axis(composite.id, data) is None


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


def test_composite_to_rgb_inspector_exposes_auto_manual_channel_mapping(qtbot):
    data = np.zeros((3, 12, 16, 18), dtype=np.uint16)
    data[0] = 1000
    data[1] = 2000
    data[2] = 3000
    viewer = _Viewer(data, metadata={"axes": "CZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("composite_to_rgb")
    saved_auto_params = dict(node.params)
    widget._connect_nodes("input", node.id)
    widget.run_pipeline(force_sync=True)
    qtbot.waitUntil(
        lambda: (
            widget.pipeline.outputs[node.id] is not None
            and float(np.asarray(widget.pipeline.outputs[node.id]).max()) == 3000.0
        ),
        timeout=30_000,
    )
    widget.graph_view.select_node(node.id)

    axis_mode = widget._parameter_widgets["channel_axis_mode"]
    channel_axis = widget._parameter_widgets["channel_axis"]
    mapping_mode = widget._parameter_widgets["mapping_mode"]
    assignments = [
        widget._parameter_widgets[f"channel_color_{index}"] for index in range(3)
    ]

    assert axis_mode.value() == "Auto"
    assert mapping_mode.value() == "Auto"
    assert channel_axis.value() == "0"
    assert not channel_axis.isEnabled()
    assert [control.value() for control in assignments] == [
        "Blue",
        "Green",
        "Red",
    ]
    assert all(not control.isEnabled() for control in assignments)
    assert (
        "Auto resolved: 0: c (3, channel)."
        in widget._parameter_widgets["channel_axis_status"].text()
    )
    assert "Channel 1 → Blue" in widget._parameter_widgets["mapping_status"].text()
    assert "red_channel" not in widget._parameter_widgets
    assert "green_channel" not in widget._parameter_widgets
    assert "blue_channel" not in widget._parameter_widgets

    for name in (
        "channel_axis_mode",
        "channel_axis",
        "mapping_mode",
        "channel_color_0",
        "channel_color_1",
        "channel_color_2",
        "intensity_mapping",
    ):
        control = widget._parameter_widgets[name]
        label = widget.parameter_form.labelForField(control)
        assert control.toolTip()
        assert label.toolTip() == control.toolTip()

    assert widget.pipeline.nodes[node.id].params == saved_auto_params
    assert widget.pipeline.nodes[node.id].params["channel_axis"] == -1
    assert widget.pipeline.nodes[node.id].params["channel_axis_mode"] == "Auto"
    assert widget.pipeline.nodes[node.id].params["mapping_mode"] == "Auto"
    # Auto remains metadata-driven; explicit assignments are persisted only
    # when Manual mapping is chosen.
    assert widget.pipeline.nodes[node.id].params.get("channel_colors", "") == ""
    assert widget.pipeline.outputs[node.id].shape == (12, 16, 18, 3)
    assert widget.pipeline.outputs[node.id].max() == 3000.0
    assert _metadata_value(widget, "Dimensions") == "z=12, y=16, x=18, rgb=3"
    _assert_rgb_channel_layers(viewer, "VIPP Inspect", (12, 16, 18))

    axis_mode.combo.setCurrentText("Manual")
    mapping_mode = widget._parameter_widgets["mapping_mode"]
    mapping_mode.combo.setCurrentText("Manual")
    assert widget._parameter_widgets["channel_axis"].isEnabled()
    assert all(
        widget._parameter_widgets[f"channel_color_{index}"].isEnabled()
        for index in range(3)
    )

    widget._parameter_widgets["channel_color_2"].combo.setCurrentText("Unassigned")
    widget._debounce_timer.stop()
    widget.run_pipeline(force_sync=True)
    qtbot.waitUntil(
        lambda: (
            widget.pipeline.outputs[node.id] is not None
            and float(np.asarray(widget.pipeline.outputs[node.id]).max()) == 2000.0
        ),
        timeout=30_000,
    )

    assert widget.pipeline.nodes[node.id].params["channel_axis"] == 0
    assert widget.pipeline.nodes[node.id].params["channel_axis_mode"] == "Manual"
    assert widget.pipeline.nodes[node.id].params["mapping_mode"] == "Manual"
    assert widget.pipeline.nodes[node.id].params["channel_colors"] == (
        "Blue,Green,Unassigned"
    )
    assert widget.pipeline.outputs[node.id].max() == 2000.0


def test_composite_to_rgb_auto_mapping_shows_exact_metadata_colour(qtbot):
    data = np.zeros((2, 8, 9), dtype=np.uint16)
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("c", "channel"),
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
        ),
        channels=(
            ChannelMetadata(name="DNA", color=0x123456),
            ChannelMetadata(name="Actin", color=0x00FF00),
        ),
    )
    widget = VippWidget(_Viewer(data, metadata={"vipp_image_state": state.to_dict()}))
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("composite_to_rgb")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    assert widget._parameter_widgets["channel_color_0"].value() == "#123456"
    assert widget._parameter_widgets["channel_color_1"].value() == "Green"
    status = widget._parameter_widgets["mapping_status"].text()
    assert "DNA → #123456" in status
    assert "Actin → Green" in status


def test_composite_to_rgb_auto_mapping_excludes_encoded_rgba_alpha(qtbot):
    data = np.zeros((8, 9, 4), dtype=np.uint8)
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
            AxisMetadata("rgba", "channel"),
        ),
    )
    widget = VippWidget(_Viewer(data, metadata={"vipp_image_state": state.to_dict()}))
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("composite_to_rgb")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    assert [
        widget._parameter_widgets[f"channel_color_{index}"].value()
        for index in range(4)
    ] == ["Red", "Green", "Blue", "Unassigned"]
    assert (
        "Channel 4 → Unassigned" in widget._parameter_widgets["mapping_status"].text()
    )


def test_composite_to_rgb_auto_axis_reports_ambiguous_metadata(qtbot):
    data = np.zeros((2, 3, 8, 9), dtype=np.uint16)
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("c", "channel"),
            AxisMetadata("channel", "channel"),
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
        ),
    )
    widget = VippWidget(_Viewer(data, metadata={"vipp_image_state": state.to_dict()}))
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("composite_to_rgb")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    assert widget._parameter_widgets["channel_axis_mode"].value() == "Auto"
    assert widget._parameter_widgets["channel_axis"].value() == "-1"
    assert not widget._parameter_widgets["channel_axis"].isEnabled()
    assert (
        "more than one explicit channel-like axis"
        in widget._parameter_widgets["channel_axis_status"].text()
    )
    assert not any(
        name.startswith("channel_color_") for name in widget._parameter_widgets
    )


def test_composite_to_rgb_manual_axis_uses_selected_dimension_labels(qtbot):
    data = np.zeros((3, 4, 8, 9), dtype=np.uint16)
    widget = VippWidget(_Viewer(data, metadata={"axes": "CZYX"}))
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("composite_to_rgb")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    widget._parameter_widgets["channel_axis_mode"].combo.setCurrentText("Manual")
    axis = widget._parameter_widgets["channel_axis"]
    assert [axis.combo.itemData(index) for index in range(axis.combo.count())] == [
        "0",
        "1",
        "2",
        "3",
    ]
    axis.combo.setCurrentIndex(axis.combo.findData("1"))
    widget._debounce_timer.stop()

    assert widget.pipeline.nodes[node.id].params["channel_axis"] == 1
    assert "channel_color_3" in widget._parameter_widgets
    label = widget.parameter_form.labelForField(
        widget._parameter_widgets["channel_color_0"]
    )
    assert label.text() == "Z 1 assignment"
    assert "Channel 1 assignment" not in label.text()


def test_composite_to_rgb_legacy_manual_mapping_renders_unassigned_slots(qtbot):
    data = np.zeros((3, 8, 9), dtype=np.uint16)
    widget = VippWidget(_Viewer(data, metadata={"axes": "CYX"}))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("composite_to_rgb")
    node.params.update(
        {
            "channel_axis": 0,
            "red_channel": 0,
            "green_channel": 1,
            "blue_channel": -1,
        }
    )
    for name in ("channel_axis_mode", "mapping_mode", "channel_colors"):
        node.params.pop(name, None)
    saved_params = dict(node.params)

    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    assert widget._parameter_widgets["channel_axis_mode"].value() == "Manual"
    assert widget._parameter_widgets["mapping_mode"].value() == "Manual"
    assert [
        widget._parameter_widgets[f"channel_color_{index}"].value()
        for index in range(3)
    ] == ["Red", "Green", "Unassigned"]
    assert node.params == saved_params


def test_composite_to_rgb_manual_empty_render_and_rerun_remain_black(qtbot):
    data = np.zeros((3, 8, 9), dtype=np.uint16)
    data[0] = 100
    data[1] = 200
    data[2] = 300
    widget = VippWidget(_Viewer(data, metadata={"axes": "CYX"}))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("composite_to_rgb")
    node.params.update(
        {
            "channel_axis_mode": "Auto",
            "mapping_mode": "Manual",
            "channel_colors": "",
            "red_channel": -1,
            "green_channel": -1,
            "blue_channel": -1,
        }
    )
    widget._connect_nodes("input", node.id)
    widget.run_pipeline(force_sync=True)

    assert not np.any(widget.pipeline.outputs[node.id])
    saved_params = dict(node.params)

    widget.inspect_node(node.id)
    widget._render_parameters(node.id)

    assert node.params == saved_params
    assert [
        widget._parameter_widgets[f"channel_color_{index}"].value()
        for index in range(3)
    ] == ["Unassigned", "Unassigned", "Unassigned"]
    assert "Manual mapping" in widget._parameter_widgets["mapping_status"].text()

    widget.run_pipeline(force_sync=True)

    assert node.params == saved_params
    assert not np.any(widget.pipeline.outputs[node.id])


def test_composite_to_rgb_invalid_manual_mapping_is_preserved_and_warned(qtbot):
    data = np.zeros((3, 8, 9), dtype=np.uint16)
    widget = VippWidget(_Viewer(data, metadata={"axes": "CYX"}))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("composite_to_rgb")
    widget._connect_nodes("input", node.id)
    node.params.update(
        {
            "channel_axis_mode": "Auto",
            "mapping_mode": "Manual",
            "channel_colors": "Red,not-a-colour",
        }
    )
    saved_params = dict(node.params)

    widget._render_parameters(node.id)

    assert node.params == saved_params
    assert [
        widget._parameter_widgets[f"channel_color_{index}"].value()
        for index in range(3)
    ] == ["Red", "not-a-colour", "Unassigned"]
    status = widget._parameter_widgets["mapping_status"]
    assert "Saved Manual mapping 'Red,not-a-colour' is invalid" in status.text()
    assert "2 entries" in status.text()
    assert "selected axis has 3" in status.text()
    assert "'not-a-colour'" in status.text()
    assert "#f59e0b" in status.styleSheet()


def test_composite_to_rgb_assignment_edit_updates_status_immediately(qtbot):
    data = np.zeros((3, 8, 9), dtype=np.uint16)
    widget = VippWidget(_Viewer(data, metadata={"axes": "CYX"}))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("composite_to_rgb")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)

    widget._parameter_widgets["mapping_mode"].combo.setCurrentText("Manual")
    assert node.params["channel_colors"] == "Blue,Green,Red"

    widget._parameter_widgets["channel_color_0"].combo.setCurrentText("Unassigned")

    assert node.params["mapping_mode"] == "Manual"
    assert node.params["channel_colors"] == "Unassigned,Green,Red"
    assert (
        "Channel 1 → Unassigned" in widget._parameter_widgets["mapping_status"].text()
    )
    widget._debounce_timer.stop()


def test_composite_to_rgb_invalid_manual_axis_stays_unresolved_until_edited(qtbot):
    data = np.zeros((3, 8, 9), dtype=np.uint16)
    widget = VippWidget(_Viewer(data, metadata={"axes": "CYX"}))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("composite_to_rgb")
    node.params.update({"channel_axis_mode": "Manual", "channel_axis": 99})
    widget._connect_nodes("input", node.id)
    saved_params = dict(node.params)

    widget._render_parameters(node.id)

    axis = widget._parameter_widgets["channel_axis"]
    assert axis.value() == "-1"
    assert axis.isEnabled()
    assert [axis.combo.itemData(index) for index in range(axis.combo.count())] == [
        "-1",
        "0",
        "1",
        "2",
    ]
    assert (
        "missing or invalid" in widget._parameter_widgets["channel_axis_status"].text()
    )
    assert node.params == saved_params


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
    qtbot.waitUntil(lambda: _inspect_layer_node_id(viewer) == node.id)

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
    qtbot.waitUntil(lambda: _inspect_layer_node_id(viewer) == "input")
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
    qtbot.waitUntil(lambda: _inspect_layer_node_id(viewer) == node.id)

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
    qtbot.waitUntil(lambda: _inspect_layer_node_id(viewer) == "input")
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
    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and len(widget.pipeline.node_outputs.get(split.id) or ()) == 3
            and all(
                output is not None
                for output in (widget.pipeline.node_outputs.get(split.id) or ())
            )
            and widget.graph_view.node_has_thumbnail(split.id)
        ),
        timeout=5_000,
    )

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
        preview_size=None,
    ):
        arr = np.asarray(data)
        calls.append((tuple(arr.shape), int(arr.max())))
        return None

    monkeypatch.setattr("napari_vipp._widget.make_preview", fake_make_preview)
    widget.pipeline.set_param(split.id, "preview_channel", 2)
    widget._update_thumbnails()

    qtbot.waitUntil(lambda: ((2, 4, 5), 30) in calls, timeout=5_000)


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

    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and not widget._pipeline_run_pending
            and widget.pipeline.outputs.get(extract.id) is not None
            and int(np.max(widget.pipeline.outputs[extract.id])) == 42
        ),
        timeout=30_000,
    )

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
    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and not widget._pipeline_run_pending
            and widget.pipeline.outputs.get(labels.id) is not None
        ),
        timeout=30_000,
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
    qtbot.waitUntil(
        lambda: widget.graph_view.node_has_thumbnail(composite.id),
        timeout=5_000,
    )

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
    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and widget.pipeline.outputs.get(composite.id) is not None
            and widget.graph_view.node_has_thumbnail(composite.id)
        ),
        timeout=5_000,
    )

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
        preview_size=None,
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
    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and not widget._pipeline_run_pending
            and widget.pipeline.outputs.get(node.id) is not None
            and widget.pipeline.outputs[node.id].shape == data.shape
        ),
        timeout=5_000,
    )

    assert widget.pipeline.outputs[node.id].shape == data.shape
    assert control.value()["ranges"] == ""

    control.set_ranges({1: (2, 2)})
    widget.run_pipeline()
    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and not widget._pipeline_run_pending
            and widget.pipeline.outputs.get(node.id) is not None
            and widget.pipeline.outputs[node.id].shape == (2, 1, 4, 5, 6)
        ),
        timeout=5_000,
    )

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
    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and not widget._pipeline_run_pending
            and widget.pipeline.outputs.get(node.id) is not None
            and widget.pipeline.outputs[node.id].shape == (1, 1, 4, 5, 6)
        ),
        timeout=5_000,
    )
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
    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and not widget._pipeline_run_pending
            and widget.pipeline.outputs.get(node.id) is not None
            and widget.pipeline.outputs[node.id].shape == (2, 4, 5, 6)
        ),
        timeout=5_000,
    )
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
    assert widget._debounce_timer.isActive()
    widget.run_pipeline()
    assert not widget._debounce_timer.isActive()
    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and not widget._pipeline_run_pending
            and widget.pipeline.outputs.get(node.id) is not None
            and widget.pipeline.outputs[node.id].shape == (1, 4, 5, 6)
        ),
        timeout=5_000,
    )
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
    qtbot.waitUntil(
        lambda: (
            widget.pipeline.outputs.get(node.id) is not None
            and widget.pipeline.outputs[node.id].shape == (2, 4, 5, 6, 3)
        ),
        timeout=5_000,
    )
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
    qtbot.waitUntil(
        lambda: (
            widget.pipeline.outputs.get(node.id) is not None
            and widget.pipeline.outputs[node.id].shape == data.shape
        ),
        timeout=5_000,
    )

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


def test_reorder_axes_moves_spatial_semantics_with_pixels_downstream(qtbot):
    data = np.zeros((3, 12, 96, 128), dtype=np.uint16)
    for z_index in range(data.shape[1]):
        data[:, z_index, :, z_index] = 100
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
    assert reorder_state.axis_order == "CYZX"
    assert [axis.source_axis for axis in reorder_state.axes] == [0, 2, 1, 3]
    assert "reordered axes to CYZX" in reorder_state.history[-1]
    assert widget.pipeline.outputs[crop.id].shape == (3, 93, 12, 121)
    assert crop_state.axis_order == "CYZX"
    assert [axis.translation for axis in crop_state.axes] == [0.0, 1.0, 0.0, 3.0]

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
    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and not widget._pipeline_run_pending
            and widget.pipeline.outputs.get(node.id) is not None
            and tuple(widget.pipeline.outputs[node.id].shape) == (3, 96, 12, 128)
            and widget.graph_view.node_has_thumbnail(node.id)
        ),
        timeout=30_000,
    )
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
        preview_size=None,
    ):
        calls.append((tuple(data.shape), current_step, state))
        return np.zeros((5, 6), dtype=np.uint8)

    monkeypatch.setattr("napari_vipp._widget.make_preview", fake_make_preview)
    widget._update_thumbnails()

    reorder_shape = tuple(widget.pipeline.outputs[node.id].shape)
    reorder_calls = [call for call in calls if call[0] == reorder_shape]
    assert reorder_calls
    assert reorder_calls[-1][1] == (0, 4, 7, 0)
    assert reorder_calls[-1][2].axis_order == "CYZX"


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

    # Preview visibility is the behavior under test. Slice contrast renders
    # immediately, while Stack contrast intentionally waits for exact
    # statistics and retains the previous complete thumbnail (#28).
    widget.thumbnail_scope_combo.setCurrentText("Slice")
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
        preview_size=None,
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
    inspector_metadata = vipp_metadata["inspector"]
    assert inspector_metadata["right_panel_visible"] is False
    assert inspector_metadata["selected_node_id"] == "threshold"
    assert {
        profile["node_id"] for profile in inspector_metadata["display_profiles"]
    } == {"gaussian", "threshold"}
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
    restored.viewer.layers["VIPP Inspect"].colormap = "magma"
    restored._workflow_metadata()
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
    assert all(key[0] == "gaussian" for key in restored._inspect_display_profiles)
    assert all(
        profile["settings"].get("colormap") == "gray"
        for profile in restored._inspect_display_profiles.values()
    )
    assert run_calls == [((), {})]


def test_loaded_workflow_highlights_uncalculated_manual_frontier(qtbot, tmp_path):
    data = np.zeros((9, 9), dtype=np.float32)
    data[1:4, 1:4] = 10
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    labels = pipeline.add_node("label_connected_components")
    measurements = pipeline.add_node("measure_objects")
    selected = pipeline.add_node("select_table_columns")
    pipeline.set_param(threshold.id, "threshold", 5)
    assert pipeline.connect("input", threshold.id).success
    assert pipeline.connect(threshold.id, labels.id).success
    assert pipeline.connect(labels.id, measurements.id).success
    assert pipeline.connect(measurements.id, selected.id).success
    path = tmp_path / "manual-workflow.json"
    save_workflow(path, pipeline, {})

    restored = VippWidget(_Viewer(data, metadata={"axes": "YX"}))
    restored._should_run_pipeline_in_background = lambda *args, **kwargs: False
    qtbot.addWidget(restored)

    restored.load_workflow_file(path)

    assert restored.pipeline.node_execution_states[measurements.id] == (
        EXECUTION_NOT_CALCULATED
    )
    assert restored.pipeline.node_execution_states[selected.id] == EXECUTION_BLOCKED
    assert (
        STALE_EXECUTION_ACCENT
        in restored.graph_view._cards[measurements.id].styleSheet()
    )
    assert (
        BLOCKED_EXECUTION_ACCENT in restored.graph_view._cards[selected.id].styleSheet()
    )
    assert restored.calculate_all_button.property("attentionRequired") is True
    assert STALE_EXECUTION_ACCENT in restored.calculate_all_button.styleSheet()


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


def test_graph_focus_button_recenters_canvas_and_preserves_zoom(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget.graph_view.resize(800, 420)
    widget.graph_view.set_zoom_percent(150)
    transform_before = widget.graph_view.transform()
    graph_center = widget.graph_view._graph_content_rect().center()

    widget.graph_view.centerOn(widget.graph_view.sceneRect().bottomRight())
    qtbot.mouseClick(widget.graph_focus_button, Qt.LeftButton)

    focused_center = _graph_view_center(widget.graph_view)
    assert widget.graph_focus_button.text() == "Focus"
    assert "without changing zoom" in widget.graph_focus_button.toolTip()
    assert abs(focused_center.x() - graph_center.x()) <= 1.0
    assert abs(focused_center.y() - graph_center.y()) <= 1.0
    assert widget.graph_view.zoom_percent == 150
    assert widget.graph_view.transform() == transform_before
    assert widget.status_label.text() == "Centered workflow graph."


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
    assert widget.pipeline.connect("threshold", watershed.id).success
    assert widget._background_processing_node_id({watershed.id}) == watershed.id
    assert widget.pipeline.disconnect("threshold", watershed.id)
    minimum = widget.pipeline.add_node("minimum_threshold")
    assert widget.pipeline.connect("input", minimum.id).success
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
    payload = widget._viewer_aligned_source_payload(SourcePayload(data, {}, "large"))

    assert state.value_range == "pending exact background calculation"
    assert state.value_pattern == ""
    assert payload.image_state is not None
    assert payload.image_state.value_range == "pending exact background calculation"
    assert payload.image_state.value_pattern == ""


def test_unchanged_owned_sample_reuses_exact_source_statistics(qtbot, monkeypatch):
    widget = VippWidget(_Viewer(np.zeros((4, 4), dtype=np.float32)))
    qtbot.addWidget(widget)
    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_BYTES", 10**9)
    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_ELEMENTS", 100)
    data = np.arange(101, dtype=np.float32)
    data.setflags(write=False)
    metadata: dict[str, object] = {}
    widget._sample_payload_cache = {
        "large": SourcePayload(
            data,
            metadata,
            "large",
            revision_token=BundledSampleRevisionToken("large"),
        )
    }
    source = widget.pipeline.nodes["input"]
    source.params["source_mode"] = "sample"
    source.params["sample_name"] = "large"

    initial_payloads, _layers = widget._source_payloads_for_pipeline()
    initial = initial_payloads["input"]
    assert initial.image_state is not None
    assert initial.image_state.value_range == "pending exact background calculation"
    exact = image_state_from_array(data, source_name="large")
    widget.pipeline.outputs["input"] = data
    widget.pipeline.output_states["input"] = exact
    widget.pipeline.node_outputs["input"] = [data]
    widget.pipeline.node_output_states["input"] = [exact]
    widget.pipeline.completed_node_ids.add("input")
    widget._last_pipeline_source_signature = widget._pipeline_source_signature(
        None,
        None,
        "",
        initial_payloads,
    )

    reused_payloads, _layers = widget._source_payloads_for_pipeline()
    reused = reused_payloads["input"]

    assert reused.image_state is not None
    assert reused.image_state.value_range == exact.value_range
    assert reused.image_state.value_pattern == exact.value_pattern


def test_bundled_samples_have_stable_revision_tokens_and_read_only_arrays(qtbot):
    widget = VippWidget(_Viewer(np.zeros((4, 4), dtype=np.float32)))
    qtbot.addWidget(widget)

    payloads = widget._sample_payloads()

    assert payloads
    for name, payload in payloads.items():
        assert payload.revision_token == BundledSampleRevisionToken(name)
        assert isinstance(payload.data, np.ndarray)
        assert not payload.data.flags.writeable


def test_bundled_sample_signature_distinguishes_regenerated_array(qtbot):
    widget = VippWidget(_Viewer(np.zeros((4, 4), dtype=np.float32)))
    qtbot.addWidget(widget)
    first = np.zeros((8, 8), dtype=np.uint16)
    second = np.zeros((8, 8), dtype=np.uint16)
    first.setflags(write=False)
    second.setflags(write=False)
    token = BundledSampleRevisionToken("same sample")

    first_signature = widget._pipeline_source_signature(
        None,
        None,
        "",
        {"input": SourcePayload(first, {}, "same sample", revision_token=token)},
    )
    second_signature = widget._pipeline_source_signature(
        None,
        None,
        "",
        {"input": SourcePayload(second, {}, "same sample", revision_token=token)},
    )

    assert first_signature != second_signature


def test_source_statistics_reuse_rejects_unrecognized_revision_tokens(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer(np.zeros((4, 4), dtype=np.float32)))
    qtbot.addWidget(widget)
    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_BYTES", 10**9)
    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_ELEMENTS", 100)
    data = np.arange(101, dtype=np.float32)
    payload = widget._viewer_aligned_source_payload(
        SourcePayload(data, {}, "mutable", revision_token=object())
    )
    exact = image_state_from_array(data, source_name="mutable")
    widget.pipeline.output_states["input"] = exact
    widget.pipeline.completed_node_ids.add("input")
    payloads = {"input": payload}
    widget._last_pipeline_source_signature = widget._pipeline_source_signature(
        None,
        None,
        "",
        payloads,
    )

    widget._reuse_cached_source_statistics(payloads)

    assert payloads["input"].image_state is not None
    assert (
        payloads["input"].image_state.value_range
        == "pending exact background calculation"
    )


def test_file_source_signature_distinguishes_pinned_series_arrays(qtbot):
    widget = VippWidget(_Viewer(np.zeros((4, 4), dtype=np.float32)))
    qtbot.addWidget(widget)
    identity = LocalSourceIdentity("file", "a" * 64, 1, 4096)
    first = SourcePayload(
        np.zeros((8, 8), dtype=np.uint16),
        {},
        "Same series name",
        revision_token=identity,
    )
    second = SourcePayload(
        np.full((8, 8), 900, dtype=np.uint16),
        {},
        "Same series name",
        revision_token=identity,
    )

    first_signature = widget._pipeline_source_signature(
        None,
        None,
        "",
        {"input": first},
    )
    second_signature = widget._pipeline_source_signature(
        None,
        None,
        "",
        {"input": second},
    )

    assert first_signature != second_signature


def test_background_result_from_old_live_source_revision_is_rejected(
    qtbot,
    monkeypatch,
):
    viewer = _Viewer(np.ones((8, 8), dtype=np.uint8))
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    pool = _QueuedThreadPool()
    widget._pipeline_thread_pool = pool
    widget.background_all_checkbox.setChecked(True)
    widget._mark_pipeline_dirty("input")
    widget.run_pipeline()

    run_id = widget._active_pipeline_run_id
    assert run_id is not None
    request = pool.workers[-1].request
    assert request.source_revisions
    applied = []
    reruns = []
    monkeypatch.setattr(
        widget,
        "_apply_pipeline_run_result",
        lambda *_args, **_kwargs: applied.append("applied"),
    )
    monkeypatch.setattr(widget, "run_pipeline", lambda: reruns.append("run"))

    source_layer = viewer.layers["input volume"]
    source_layer.data = np.full((8, 8), 23, dtype=np.uint8)
    source_layer.events.data.emit()
    widget._debounce_timer.stop()

    assert widget._pipeline_cancel_events[run_id].is_set()
    widget._on_background_pipeline_finished(
        PipelineRunResult(
            run_id,
            request.workflow,
            widget.pipeline,
            source_revisions=request.source_revisions,
        )
    )

    assert applied == []
    assert "old live-source revision" in widget.status_label.text()
    qtbot.waitUntil(lambda: reruns == ["run"], timeout=5_000)


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
            and widget.pipeline_busy_bar.isHidden()
        ),
        timeout=30_000,
    )

    assert widget.pipeline_busy_bar.isHidden()
    assert not widget.graph_view._cards[node.id].is_processing()
    assert widget.pipeline.outputs[node.id].shape == (3, 12, 12)


def test_richardson_lucy_tv_controls_explain_effects_and_separate_ranges(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("richardson_lucy_tv_deconvolution")
    widget.graph_view.select_node(node.id)

    parameter_names = {
        "spatial_mode",
        "iterations",
        "tv_regularization",
        "tv_epsilon",
        "normalize_psf",
        "clip_negative_input",
        "clip_output_negative",
        "preserve_input_scale",
        "filter_epsilon",
        "denominator_floor",
    }
    for name in parameter_names:
        control = widget._parameter_widgets[name]
        label = widget.parameter_form.labelForField(control)
        assert control.toolTip()
        assert label.toolTip() == control.toolTip()
        for child in control.findChildren(QWidget):
            assert child.toolTip() == control.toolTip()

    iterations = widget._parameter_widgets["iterations"]
    assert (iterations._bounds.minimum, iterations._bounds.maximum) == (1, 100)
    assert not iterations._bounds.logarithmic
    assert iterations.value_box.maximum() == 2_147_483_647
    iterations.value_box.setValue(250)
    assert iterations.value() == 250
    assert iterations.slider.maximum() == 100
    assert iterations.slider.value() == 100
    assert node.params["iterations"] == 250

    expected_log_bounds = {
        "tv_regularization": (1e-6, 1e-1),
        "tv_epsilon": (1e-12, 1e-2),
        "filter_epsilon": (1e-15, 1e-3),
        "denominator_floor": (1e-3, 1.0),
    }
    for name, expected in expected_log_bounds.items():
        control = widget._parameter_widgets[name]
        assert control._bounds.logarithmic
        assert (control._bounds.minimum, control._bounds.maximum) == expected
        assert (control.slider.minimum(), control.slider.maximum()) == (0, 1000)
        assert control.value_box.maximum() == 1_000_000.0

    filter_epsilon = widget._parameter_widgets["filter_epsilon"]
    assert filter_epsilon.value_box.decimals() == 15
    assert filter_epsilon.value() == pytest.approx(1e-12)
    assert filter_epsilon.value_box.text() == "1e-12"
    filter_epsilon.value_box.setValue(0.0)
    assert filter_epsilon.value() == 0.0
    assert filter_epsilon.slider.value() == 0
    assert filter_epsilon._bounds.minimum == 1e-15
    assert node.params["filter_epsilon"] == 0.0

    tv_epsilon = widget._parameter_widgets["tv_epsilon"]
    tv_epsilon.slider.setValue(500)
    assert tv_epsilon.value() == pytest.approx(1e-7, rel=1e-5)

    tv_regularization = widget._parameter_widgets["tv_regularization"]
    tv_regularization.value_box.setValue(0.25)
    assert tv_regularization.value() == 0.25
    assert tv_regularization._bounds.maximum == 0.1
    assert tv_regularization.slider.value() == 1000
    assert node.params["tv_regularization"] == 0.25

    assert "under-converged" in widget._parameter_widgets["iterations"].toolTip()
    assert "0.002" in widget._parameter_widgets["tv_regularization"].toolTip()
    assert "0.008-0.012" in widget._parameter_widgets["tv_regularization"].toolTip()
    assert "first response" in widget._parameter_widgets["filter_epsilon"].toolTip()
    assert "first response" in widget._parameter_widgets["denominator_floor"].toolTip()


def test_deconvolution_psf_status_renders_once_without_mutating_parameters(
    qtbot,
    monkeypatch,
):
    image = np.zeros((32, 32), dtype=np.float32)
    psf = np.zeros((5, 5), dtype=np.float32)
    psf[2, 2] = 1.0
    widget = VippWidget(_Viewer(image, metadata={"axes": "YX"}))
    qtbot.addWidget(widget)

    psf_source = widget.pipeline.add_node("input")
    node = widget.add_node_from_palette("richardson_lucy_tv_deconvolution")
    widget.pipeline.set_param(node.id, "spatial_mode", "2D YX")
    assert widget.pipeline.connect("input", node.id, target_port=0).success
    assert widget.pipeline.connect(psf_source.id, node.id, target_port=1).success
    axes = (
        AxisMetadata("y", "space", unit="micrometer", scale=0.1),
        AxisMetadata("x", "space", unit="micrometer", scale=0.1),
    )
    image_state = image_state_from_array(image, axes=axes)
    psf_state = image_state_from_array(
        psf,
        axes=axes,
    )
    widget.pipeline.outputs["input"] = image
    widget.pipeline.output_states["input"] = image_state
    widget.pipeline.node_outputs["input"] = [image]
    widget.pipeline.node_output_states["input"] = [image_state]
    widget.pipeline.outputs[psf_source.id] = psf
    widget.pipeline.output_states[psf_source.id] = psf_state
    widget.pipeline.node_outputs[psf_source.id] = [psf]
    widget.pipeline.node_output_states[psf_source.id] = [psf_state]

    import napari_vipp._widget as widget_module

    calls = []
    original = widget_module.psf_preflight

    def counted_preflight(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(widget_module, "psf_preflight", counted_preflight)
    widget._selected_node_id = node.id
    widget._psf_preflight_cache.clear()
    params_before = dict(node.params)
    execution_before = (
        widget.pipeline.node_execution_states[node.id],
        widget.pipeline.node_execution_messages[node.id],
    )

    widget._render_parameters(node.id)
    widget._render_parameters(node.id)

    note = widget._parameter_widgets["operation_notice"]
    assert note.property("preflightStatus") == "ready"
    assert "PSF preflight: Ready" in note.text()
    assert "Excessive TV regularization" in note.text()
    assert "under-converged" in note.text()
    assert "Validate PSF sampling and centering" in note.text()
    assert calls == [1]
    assert node.params == params_before
    assert (
        widget.pipeline.node_execution_states[node.id],
        widget.pipeline.node_execution_messages[node.id],
    ) == execution_before

    shifted_psf = np.zeros_like(psf)
    shifted_psf[2, 1] = 1.0
    shifted_state = image_state_from_array(shifted_psf, axes=axes)
    widget.pipeline.outputs[psf_source.id] = shifted_psf
    widget.pipeline.output_states[psf_source.id] = shifted_state
    widget.pipeline.node_outputs[psf_source.id] = [shifted_psf]
    widget.pipeline.node_output_states[psf_source.id] = [shifted_state]

    widget._refresh_selected_parameter_controls()

    note = widget._parameter_widgets["operation_notice"]
    assert note.property("preflightStatus") == "warning"
    assert "PSF preflight: Warning" in note.text()
    assert "peak is 1 voxels" in note.text()
    assert calls == [1, 1]
    assert node.params == params_before
    assert (
        widget.pipeline.node_execution_states[node.id],
        widget.pipeline.node_execution_messages[node.id],
    ) == execution_before


def test_deconvolution_psf_status_warns_when_calibration_is_missing(qtbot):
    image = np.zeros((32, 32), dtype=np.float32)
    psf = np.zeros((5, 5), dtype=np.float32)
    psf[2, 2] = 1.0
    widget = VippWidget(_Viewer(image))
    qtbot.addWidget(widget)

    psf_source = widget.pipeline.add_node("input")
    node = widget.add_node_from_palette("richardson_lucy_deconvolution")
    widget.pipeline.set_param(node.id, "spatial_mode", "2D YX")
    assert widget.pipeline.connect("input", node.id, target_port=0).success
    assert widget.pipeline.connect(psf_source.id, node.id, target_port=1).success
    widget.pipeline.outputs["input"] = image
    widget.pipeline.output_states["input"] = image_state_from_array(image)
    widget.pipeline.outputs[psf_source.id] = psf
    widget.pipeline.output_states[psf_source.id] = image_state_from_array(psf)
    widget._selected_node_id = node.id
    widget._psf_preflight_cache.clear()

    widget._render_parameters(node.id)

    note = widget._parameter_widgets["operation_notice"]
    assert note.property("preflightStatus") == "warning"
    assert "PSF preflight: Warning" in note.text()
    assert "calibration is missing" in note.text()


def test_deconvolution_psf_note_separates_passes_and_actionable_size_warning(qtbot):
    image = np.zeros((11, 64, 64), dtype=np.float32)
    psf = np.zeros((33, 5, 5), dtype=np.float32)
    psf[16, 2, 2] = 0.96
    psf[0, 2, 2] = 0.02
    psf[-1, 2, 2] = 0.02
    axes = (
        AxisMetadata("z", "space", unit="micrometer", scale=0.101),
        AxisMetadata("y", "space", unit="micrometer", scale=0.025),
        AxisMetadata("x", "space", unit="micrometer", scale=0.025),
    )
    widget = VippWidget(_Viewer(image))
    qtbot.addWidget(widget)

    born_wolf = widget.pipeline.add_node("born_wolf_psf")
    prepared = widget.pipeline.add_node("prepare_validate_psf")
    node = widget.add_node_from_palette("richardson_lucy_tv_deconvolution")
    for name, value in {
        "auto_parameters": False,
        "wavelength_nm": 561.0,
        "numerical_aperture": 1.46,
        "refractive_index": 1.518,
        "pixel_size_xy_um": 0.025,
        "z_step_um": 0.101,
    }.items():
        widget.pipeline.set_param(born_wolf.id, name, value)
    widget.pipeline.set_param(node.id, "spatial_mode", "3D ZYX")
    assert widget.pipeline.connect("input", node.id, target_port=0).success
    assert widget.pipeline.connect("input", born_wolf.id).success
    assert widget.pipeline.connect(born_wolf.id, prepared.id).success
    assert widget.pipeline.connect(prepared.id, node.id, target_port=1).success
    image_state = image_state_from_array(image, axes=axes)
    psf_state = image_state_from_array(psf, axes=axes)
    widget.pipeline.outputs["input"] = image
    widget.pipeline.output_states["input"] = image_state
    widget.pipeline.outputs[prepared.id] = psf
    widget.pipeline.output_states[prepared.id] = psf_state
    widget._selected_node_id = node.id
    widget._psf_preflight_cache.clear()

    widget._render_parameters(node.id)

    note = widget._parameter_widgets["operation_notice"]
    text = note.text()
    assert note.property("preflightStatus") == "warning"
    assert "#cbd5e1" in note.styleSheet()
    assert "PSF preflight: Warning" in text
    assert "CHECKS PASSED" in text
    assert "conventional-widefield Nyquist estimate is met" in text
    assert "XY 0.025 um &lt;= 0.09606 um" in text
    assert "Z 0.101 um &lt;= 0.2544 um" in text
    assert "normalized (sum = 1)" in text
    assert "centered (peak offset 0; centroid offset 0 voxel)" in text
    assert "NEEDS ATTENTION" in text
    assert "PSF support reaches or exceeds the image extent" in text
    assert "Z support is larger than the image" in text
    assert "33 PSF samples versus 11 image samples" in text
    assert "Cropping Z from 33 to 11 centered samples" in text
    assert "retain 96.0% and discard 4.0%" in text
    assert "not intensity outside the array" in text
    assert "WHAT TO DO NEXT" in text
    assert "set Spatial processing to 2D YX on both Born-Wolf PSF" in text
    assert "Prepare / Validate PSF is working as intended" in text
    assert "support/image" not in text


def test_richardson_lucy_baseline_controls_include_safety_guidance(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("richardson_lucy_deconvolution")
    widget.graph_view.select_node(node.id)

    for name in ("spatial_mode", "iterations", "normalize_psf", "filter_epsilon"):
        control = widget._parameter_widgets[name]
        label = widget.parameter_form.labelForField(control)
        assert control.toolTip()
        assert label.toolTip() == control.toolTip()
    assert "under-converged" in widget._parameter_widgets["iterations"].toolTip()
    assert "first response" in widget._parameter_widgets["filter_epsilon"].toolTip()


@pytest.mark.parametrize(
    "operation_id",
    [
        "richardson_lucy_deconvolution",
        "richardson_lucy_tv_deconvolution",
    ],
)
def test_richardson_lucy_inspector_controls_can_shrink(qtbot, operation_id):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette(operation_id)
    widget.graph_view.select_node(node.id)

    assert widget.parameter_form.rowWrapPolicy() == QFormLayout.WrapLongRows
    assert widget.selected_title.wordWrap()
    assert widget.selected_title.sizePolicy().horizontalPolicy() == QSizePolicy.Ignored
    assert (
        widget.auto_recalculate_notice.sizePolicy().horizontalPolicy()
        == QSizePolicy.Ignored
    )

    spatial_mode = widget._parameter_widgets["spatial_mode"]
    assert spatial_mode.sizePolicy().horizontalPolicy() == QSizePolicy.Ignored
    assert spatial_mode.combo.sizePolicy().horizontalPolicy() == QSizePolicy.Ignored

    slider_controls = [
        control
        for control in widget._parameter_widgets.values()
        if hasattr(control, "slider")
    ]
    assert slider_controls
    assert all(
        control.sizePolicy().horizontalPolicy() == QSizePolicy.Ignored
        for control in slider_controls
    )
    assert all(control.slider.minimumWidth() == 72 for control in slider_controls)

    gaussian = widget.add_node_from_palette("gaussian_blur")
    widget.graph_view.select_node(gaussian.id)
    assert widget.parameter_form.rowWrapPolicy() == QFormLayout.DontWrapRows


def test_richardson_lucy_note_reserves_wrapped_height_without_group_stretch(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("richardson_lucy_tv_deconvolution")
    widget.graph_view.select_node(node.id)
    note = widget._parameter_widgets["operation_notice"]

    widget.show()
    qtbot.waitUntil(
        lambda: note.minimumHeight() >= note.heightForWidth(note.contentsRect().width())
    )

    required_height = note.heightForWidth(note.contentsRect().width())
    assert required_height > 0
    assert note.minimumHeight() >= required_height
    assert note.alignment() & Qt.AlignTop
    assert widget.parameter_group.sizePolicy().verticalPolicy() == QSizePolicy.Maximum
    session = widget._workflow_tabs.current
    assert session is not None
    session.mark_clean(
        widget._current_history_snapshot(),
        persistence_token=widget._workflow_tab_persistence_token(),
    )
    widget.hide()


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
    # invalidate its deconvolution result; the cheap parallel branch is then
    # dispatched through the same detached Auto service.
    pool.workers[0].run()
    qtbot.waitUntil(lambda: len(pool.workers) == 2, timeout=5_000)
    pool.workers[1].run()
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


def test_discarded_full_graph_run_requeues_every_node(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.uint8)))
    qtbot.addWidget(widget)

    widget._begin_pipeline_dispatch(None)
    assert widget._inflight_full_graph is True

    widget._requeue_inflight_dirty_nodes()

    assert widget._inflight_full_graph is False
    assert widget._inflight_dirty_node_ids is None
    assert set(widget.pipeline.nodes) <= widget._pending_dirty_node_ids


def test_failed_partial_clone_preserves_valid_outputs_thumbnails_and_badges(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: widget.graph_view.node_has_thumbnail("gaussian"),
        timeout=5_000,
    )
    old_output = widget.pipeline.outputs["gaussian"]
    card = widget.graph_view._cards["gaussian"]
    old_badge = card.compute_badge.text()
    assert card.preview.has_source_pixmap()
    assert not card.preview.source_pixmap().isNull()
    partial = deepcopy(widget.pipeline)
    unreported_output = np.full_like(old_output, 77)
    partial.outputs["gaussian"] = unreported_output
    partial.node_outputs["gaussian"] = [unreported_output]
    partial.completed_node_ids.add("gaussian")
    run_id = 123
    widget._active_pipeline_run_id = run_id
    widget._pipeline_cancel_events[run_id] = threading.Event()
    widget._pipeline_run_context[run_id] = (
        None,
        "input volume",
        "gaussian",
        widget._last_pipeline_source_signature,
        {"gaussian"},
        widget._current_compute_request(),
        frozenset({"gaussian"}),
    )
    widget._begin_pipeline_dispatch({"gaussian"})
    widget._set_pipeline_busy(True, "gaussian")

    widget._on_background_pipeline_finished(
        PipelineRunResult(
            run_id,
            serialize_workflow(
                widget.pipeline,
                compute_request=widget._current_compute_request(),
            ),
            pipeline=partial,
            error="synthetic allocation failure",
        )
    )

    assert widget.pipeline.outputs["gaussian"] is old_output
    assert card.preview.has_source_pixmap()
    assert not card.preview.source_pixmap().isNull()
    assert card.compute_badge.text() == old_badge
    assert "Previous result (stale)" in card.compute_badge.toolTip()
    assert "gaussian" in widget._pending_dirty_node_ids
    assert "synthetic allocation failure" in widget.status_label.text()


def test_cpu_partial_failure_updates_completed_metrics_and_errors_actual_node(
    qtbot,
):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    metrics = widget.add_node_from_palette("masked_colocalization_metrics")
    racc = widget.add_node_from_palette("masked_racc_index")

    old_table = TableData(("value",), ((26067,),), name="old metrics")
    old_state = TableState(1, 1, ("value",), source_name="old metrics")
    new_table = TableData(("value",), ((48403,),), name="new metrics")
    new_state = TableState(1, 1, ("value",), source_name="new metrics")
    widget.pipeline.outputs[metrics.id] = old_table
    widget.pipeline.output_states[metrics.id] = old_state
    widget.pipeline.node_outputs[metrics.id] = [old_table]
    widget.pipeline.node_output_states[metrics.id] = [old_state]
    widget.pipeline.completed_node_ids.add(metrics.id)
    widget.pipeline.node_execution_states[metrics.id] = EXECUTION_READY
    widget.pipeline.nodes[metrics.id].params["threshold_mode"] = "Costes auto"
    widget.pipeline.nodes[metrics.id].params["channel_1_threshold"] = 26067.0
    widget.pipeline.nodes[racc.id].params["threshold_mode"] = "Costes auto"
    widget.pipeline.nodes[racc.id].params["channel_1_threshold"] = 26067.0
    widget.pipeline.nodes[racc.id].params["channel_2_threshold"] = 3071.0
    widget.graph_view.select_node(racc.id)
    threshold_1_control = widget._parameter_widgets["channel_1_threshold"]
    threshold_2_control = widget._parameter_widgets["channel_2_threshold"]

    partial = deepcopy(widget.pipeline)
    partial.outputs[metrics.id] = new_table
    partial.output_states[metrics.id] = new_state
    partial.node_outputs[metrics.id] = [new_table]
    partial.node_output_states[metrics.id] = [new_state]
    partial.completed_node_ids.add(metrics.id)
    partial.node_execution_states[metrics.id] = EXECUTION_READY
    partial.nodes[metrics.id].params["channel_1_threshold"] = 48403.0
    partial.completed_node_ids.discard(racc.id)
    partial.node_execution_states[racc.id] = EXECUTION_ERROR
    partial.node_execution_messages[racc.id] = "masked RACC sentinel"
    partial.nodes[racc.id].params["channel_1_threshold"] = 48403.0
    partial.nodes[racc.id].params["channel_2_threshold"] = 61092.0

    decision = NodeExecutionDecision(
        metrics.id,
        metrics.operation_id,
        NodeComputePreference(NodePreferenceKind.CPU),
        "cpu-numpy",
        "cpu",
        f"cpu-{metrics.operation_id}-v1",
        DecisionKind.POLICY_CPU,
        DecisionReason.EXPLICIT_CPU,
        "CPU test decision.",
        implementation_version="1",
    )
    request = ComputeRequest(mode=ComputeMode.CPU)
    run_id = 127
    dirty = {metrics.id, racc.id}
    widget._active_pipeline_run_id = run_id
    widget._pipeline_cancel_events[run_id] = threading.Event()
    widget._pipeline_run_context[run_id] = (
        None,
        "input volume",
        metrics.id,
        widget._last_pipeline_source_signature,
        dirty,
        request,
        frozenset(dirty),
    )
    widget._pipeline_run_manual_node_ids[run_id] = frozenset(dirty)
    widget._begin_pipeline_dispatch(dirty)
    widget._set_pipeline_busy(True, metrics.id)
    widget._on_background_pipeline_node_started((run_id, racc.id))

    widget._on_background_pipeline_finished(
        PipelineRunResult(
            run_id,
            serialize_workflow(widget.pipeline, compute_request=request),
            pipeline=partial,
            error="masked RACC sentinel",
            execution_report=ExecutionReport(
                request,
                ComputeEnvironment(),
                actual_decisions=(decision,),
            ),
        )
    )

    assert widget.pipeline.outputs[metrics.id] is new_table
    assert widget.pipeline.node_execution_states[metrics.id] == EXECUTION_READY
    assert widget.pipeline.node_execution_messages[metrics.id] == ""
    assert widget.pipeline.nodes[metrics.id].params["channel_1_threshold"] == 48403.0
    assert widget.pipeline.node_execution_states[racc.id] == EXECUTION_ERROR
    assert widget.pipeline.node_execution_messages[racc.id] == ("masked RACC sentinel")
    assert widget.pipeline.nodes[racc.id].params["channel_1_threshold"] == 48403.0
    assert widget.pipeline.nodes[racc.id].params["channel_2_threshold"] == 61092.0
    assert threshold_1_control.value() == 48403.0
    assert threshold_2_control.value() == 61092.0
    assert not threshold_1_control.isEnabled()
    assert not threshold_2_control.isEnabled()

    widget._on_param_changed("threshold_mode", "Manual")
    widget._debounce_timer.stop()

    assert threshold_1_control.value() == 48403.0
    assert threshold_2_control.value() == 61092.0
    assert threshold_1_control.isEnabled()
    assert threshold_2_control.isEnabled()
    assert widget.pipeline.nodes[racc.id].params["channel_1_threshold"] == 48403.0
    assert widget.pipeline.nodes[racc.id].params["channel_2_threshold"] == 61092.0
    assert racc.id in widget._pending_dirty_node_ids
    assert racc.id not in widget._pending_manual_node_ids


def test_cancel_background_run_requeues_dirty_nodes(qtbot):
    viewer = _Viewer(np.ones((8, 8), dtype=np.uint8) * 20)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget._active_pipeline_run_id = 123
    cancel_event = threading.Event()
    widget._pipeline_cancel_events[123] = cancel_event
    widget._pipeline_run_pending = True
    widget._pipeline_run_context[123] = (None, "input volume", "gaussian", None, None)
    widget._inflight_dirty_node_ids = {"gaussian"}
    widget._set_pipeline_busy(True, "gaussian", queued=True)

    assert not widget.pipeline_cancel_button.isHidden()
    assert widget.graph_view._cards["gaussian"].is_processing()

    widget._cancel_background_pipeline_run()

    assert cancel_event.is_set()
    assert widget._active_pipeline_run_id == 123
    assert widget._pipeline_run_pending is False
    assert 123 in widget._pipeline_run_context
    assert widget._inflight_dirty_node_ids == {"gaussian"}
    assert widget.pipeline_cancel_button.isHidden()
    assert widget.graph_view._cards["gaussian"].is_processing()
    assert not widget.compute_mode_combo.isEnabled()
    assert "waiting" in widget.status_label.text().lower() or (
        "remain locked" in widget.status_label.text().lower()
    )

    widget._on_background_pipeline_finished(PipelineRunResult(123, {}, cancelled=True))

    assert widget._active_pipeline_run_id is None
    assert 123 not in widget._pipeline_run_context
    assert "gaussian" in widget._pending_dirty_node_ids
    assert widget._inflight_dirty_node_ids is None
    assert not widget.graph_view._cards["gaussian"].is_processing()
    assert widget.compute_mode_combo.isEnabled()
    assert "cleanup completed" in widget.status_label.text()


def test_cancel_cleanup_failure_quarantines_compute_until_restart(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.uint8) * 20))
    qtbot.addWidget(widget)
    previous_output = widget.pipeline.outputs["gaussian"]
    run_id = 124
    widget._active_pipeline_run_id = run_id
    widget._pipeline_cancel_events[run_id] = threading.Event()
    widget._pipeline_run_context[run_id] = (
        None,
        "input volume",
        "gaussian",
        widget._last_pipeline_source_signature,
        {"gaussian"},
        widget._current_compute_request(),
        frozenset({"gaussian"}),
    )
    widget._inflight_dirty_node_ids = {"gaussian"}
    widget._set_pipeline_busy(True, "gaussian")

    widget._cancel_background_pipeline_run()
    widget._on_background_pipeline_finished(
        PipelineRunResult(
            run_id,
            {},
            cancelled=True,
            failure=PipelineExecutionFailure(
                kind="cancelled",
                error_type="RuntimeCleanupError",
                message="CUDA cleanup failed",
                cleanup_succeeded=False,
            ),
        )
    )

    assert widget._compute_runtime_quarantined_reason
    assert "Restart VIPP" in widget._compute_runtime_quarantined_reason
    assert widget.pipeline.outputs["gaussian"] is previous_output
    assert not widget.compute_mode_combo.isEnabled()
    assert widget.status_label.property("messageSeverity") == "error"
    assert widget.status_label.property("messageActionable") is True

    pool = _QueuedThreadPool()
    widget._pipeline_thread_pool = pool
    widget.run_pipeline()
    assert pool.workers == []
    assert "Restart VIPP" in widget.status_label.text()


def test_internal_abandon_retains_worker_ownership_until_cleanup(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.uint8) * 20))
    qtbot.addWidget(widget)
    run_id = 125
    cancel_event = threading.Event()
    context = (None, "input volume", "gaussian", None, {"gaussian"})
    widget._active_pipeline_run_id = run_id
    widget._active_pipeline_node_id = "gaussian"
    widget._pipeline_cancel_events[run_id] = cancel_event
    widget._pipeline_run_context[run_id] = context
    widget._set_pipeline_busy(True, "gaussian")

    widget._abandon_background_pipeline_run()

    assert cancel_event.is_set()
    assert widget._active_pipeline_run_id == run_id
    assert widget._pipeline_run_context[run_id] is context
    assert not widget.pipeline_busy_bar.isHidden()
    assert widget.pipeline_cancel_button.isHidden()
    assert not widget.compute_mode_combo.isEnabled()
    assert "cleaning up" in widget.pipeline_busy_label.text()


def test_internal_cleanup_failure_also_quarantines_compute(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.uint8) * 20))
    qtbot.addWidget(widget)
    run_id = 126
    widget._active_pipeline_run_id = run_id
    widget._pipeline_cancel_events[run_id] = threading.Event()
    widget._pipeline_run_context[run_id] = (
        None,
        "input volume",
        "gaussian",
        widget._last_pipeline_source_signature,
        {"gaussian"},
    )
    widget._inflight_dirty_node_ids = {"gaussian"}
    widget._set_pipeline_busy(True, "gaussian")

    widget._on_background_pipeline_finished(
        PipelineRunResult(
            run_id,
            {},
            cancelled=True,
            failure=PipelineExecutionFailure(
                kind="cancelled",
                error_type="RuntimeCleanupError",
                message="CUDA cleanup failed",
                cleanup_succeeded=False,
            ),
        )
    )

    assert widget._compute_runtime_quarantined_reason
    assert not widget.compute_mode_combo.isEnabled()
    assert "Restart VIPP" in widget.status_label.text()


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


def test_force_sync_does_not_detach_active_background_worker(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.uint8) * 20))
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: widget._active_pipeline_run_id is None,
        timeout=30_000,
    )
    manual = widget.add_node_from_palette("measure_objects")
    pending = widget.add_node_from_palette("gamma_correction")
    cancel_event = threading.Event()
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

    widget.run_pipeline(force_sync=True)

    # The new work is structurally independent, so the active worker may finish
    # normally; either way, force_sync must not detach its cleanup ownership.
    assert not cancel_event.is_set()
    assert widget._active_pipeline_run_id == 123
    assert widget._pipeline_run_pending
    assert widget._pipeline_run_manual_node_ids[123] == {manual.id}
    assert pending.id in widget._pending_dirty_node_ids
    assert 123 in widget._pipeline_run_context
    assert 123 in widget._pipeline_run_manual_node_ids
    assert not widget.pipeline_busy_bar.isHidden()
    assert not widget.compute_mode_combo.isEnabled()


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


def test_completed_background_node_withholds_stack_thumbnail_until_final_limits(
    qtbot,
    monkeypatch,
):
    viewer = _Viewer(np.zeros((5, 18, 20), dtype=np.float32))
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: widget.graph_view.node_has_thumbnail("gaussian"),
        timeout=30_000,
    )
    widget.thumbnail_scope_combo.setCurrentText("Stack")
    otsu = widget.add_node_from_palette("otsu_threshold")
    old_output = widget.pipeline.outputs["gaussian"]
    state = widget.pipeline.output_states["gaussian"]
    completed_output = np.arange(5 * 18 * 20, dtype=np.float32).reshape(5, 18, 20)
    thumbnails = []
    monkeypatch.setattr(
        widget.graph_view,
        "set_thumbnail",
        lambda node_id, thumbnail: thumbnails.append((node_id, thumbnail)),
    )
    widget._active_pipeline_run_id = 412
    widget.pipeline.node_execution_states["gaussian"] = EXECUTION_BLOCKED
    widget._sync_execution_ui()
    gaussian_card = widget.graph_view._cards["gaussian"]
    committed_thumbnail = gaussian_card.preview.source_pixmap().toImage()
    widget.inspect_node("gaussian")
    inspection_layer = viewer.layers["VIPP Inspect"]
    assert np.shares_memory(inspection_layer.data, old_output)
    assert BLOCKED_EXECUTION_ACCENT in gaussian_card.styleSheet()
    widget.graph_view.set_node_processing("gaussian", True)

    widget._on_background_pipeline_node_finished(
        PipelineNodeResult(
            run_id=412,
            node_id="gaussian",
            operation_id=widget.pipeline.nodes["gaussian"].operation_id,
            output=completed_output,
            output_state=state,
            node_outputs=(completed_output,),
            node_output_states=(state,),
            execution_state=EXECUTION_READY,
        )
    )

    assert widget.pipeline.outputs["gaussian"] is old_output
    assert widget.pipeline.node_execution_states["gaussian"] == EXECUTION_BLOCKED
    assert widget._node_display_payload("gaussian")[0] is completed_output
    assert np.shares_memory(inspection_layer.data, completed_output)
    assert thumbnails == []
    assert gaussian_card.preview.has_source_pixmap()
    assert gaussian_card.preview.source_pixmap().toImage() == committed_thumbnail
    assert not gaussian_card.is_processing()
    assert gaussian_card._execution_state == EXECUTION_READY
    assert BLOCKED_EXECUTION_ACCENT not in gaussian_card.styleSheet()

    widget._on_background_pipeline_node_started((412, otsu.id))

    assert widget.graph_view._cards[otsu.id].is_processing()
    assert gaussian_card._execution_state == EXECUTION_READY
    assert BLOCKED_EXECUTION_ACCENT not in gaussian_card.styleSheet()

    widget._on_background_pipeline_finished(PipelineRunResult(412, {}, cancelled=True))

    assert widget.pipeline.node_execution_states["gaussian"] == EXECUTION_BLOCKED
    assert widget._node_display_payload("gaussian")[0] is old_output
    assert np.shares_memory(inspection_layer.data, old_output)
    assert gaussian_card._execution_state == EXECUTION_BLOCKED
    assert BLOCKED_EXECUTION_ACCENT in gaussian_card.styleSheet()


def test_completed_background_node_display_state_is_invalidated_by_new_edit(qtbot):
    widget = VippWidget(_Viewer(np.zeros((18, 20), dtype=np.float32)))
    qtbot.addWidget(widget)
    qtbot.waitUntil(lambda: widget._active_pipeline_run_id is None, timeout=30_000)
    old_output = widget.pipeline.outputs["gaussian"]
    state = widget.pipeline.output_states["gaussian"]
    completed_output = np.ones((18, 20), dtype=np.float32)
    widget._active_pipeline_run_id = 413
    widget.pipeline.node_execution_states["gaussian"] = EXECUTION_BLOCKED
    widget._sync_execution_ui()

    widget._on_background_pipeline_node_finished(
        PipelineNodeResult(
            run_id=413,
            node_id="gaussian",
            operation_id=widget.pipeline.nodes["gaussian"].operation_id,
            output=completed_output,
            output_state=state,
            node_outputs=(completed_output,),
            node_output_states=(state,),
            execution_state=EXECUTION_READY,
        )
    )

    card = widget.graph_view._cards["gaussian"]
    assert card._execution_state == EXECUTION_READY
    assert widget.pipeline.outputs["gaussian"] is old_output
    assert widget._node_display_payload("gaussian")[0] is completed_output

    widget._mark_pipeline_dirty("input")

    assert card._execution_state == EXECUTION_BLOCKED
    assert BLOCKED_EXECUTION_ACCENT in card.styleSheet()
    assert widget.pipeline.outputs["gaussian"] is old_output
    assert widget._node_display_payload("gaussian")[0] is old_output

    widget._on_background_pipeline_finished(PipelineRunResult(413, {}, cancelled=True))


def test_predebounce_dirty_generation_suppresses_affected_progressive_result(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer(np.zeros((18, 20), dtype=np.float32)))
    qtbot.addWidget(widget)
    qtbot.waitUntil(lambda: widget._active_pipeline_run_id is None, timeout=30_000)
    independent = widget.add_node_from_palette("gamma_correction")
    for connection in tuple(widget.pipeline._input_connections(independent.id)):
        assert widget.pipeline.disconnect(
            connection.source_id,
            independent.id,
            connection.target_port,
        )
    state = widget.pipeline.output_states["gaussian"]
    completed_output = np.ones((18, 20), dtype=np.float32)
    widget._active_pipeline_run_id = 420
    widget._pipeline_run_pending = False
    widget._pending_dirty_node_ids = {"input"}
    published = []
    monkeypatch.setattr(
        widget,
        "_update_node_thumbnail",
        lambda node_id, *_args, **_kwargs: published.append(node_id),
    )

    widget._on_background_pipeline_node_finished(
        PipelineNodeResult(
            run_id=420,
            node_id="gaussian",
            operation_id=widget.pipeline.nodes["gaussian"].operation_id,
            output=completed_output,
            output_state=state,
            node_outputs=(completed_output,),
            node_output_states=(state,),
            execution_state=EXECUTION_READY,
        )
    )

    assert published == []
    assert "gaussian" not in widget._background_node_result_overrides

    widget._on_background_pipeline_node_finished(
        PipelineNodeResult(
            run_id=420,
            node_id=independent.id,
            operation_id=independent.operation_id,
            output=completed_output,
            output_state=state,
            node_outputs=(completed_output,),
            node_output_states=(state,),
            execution_state=EXECUTION_READY,
        )
    )

    assert published == [independent.id]
    widget._on_background_pipeline_finished(PipelineRunResult(420, {}, cancelled=True))


def test_completed_background_table_uses_and_rolls_back_run_local_preview(qtbot):
    widget = VippWidget(_Viewer(np.zeros((18, 20), dtype=np.float32)))
    qtbot.addWidget(widget)
    qtbot.waitUntil(lambda: widget._active_pipeline_run_id is None, timeout=30_000)
    table_node = widget.add_node_from_palette("measure_objects")
    old_table = TableData(("value",), ((1,),), name="old")
    old_state = TableState(1, 1, ("value",), source_name="old")
    new_table = TableData(("value",), ((2,), (3,)), name="new")
    new_state = TableState(2, 1, ("value",), source_name="new")
    widget.pipeline.outputs[table_node.id] = old_table
    widget.pipeline.output_states[table_node.id] = old_state
    widget.pipeline.node_outputs[table_node.id] = [old_table]
    widget.pipeline.node_output_states[table_node.id] = [old_state]
    widget.pipeline.node_execution_states[table_node.id] = EXECUTION_BLOCKED
    widget._active_pipeline_run_id = 414
    widget._sync_execution_ui()
    widget._update_table_preview()

    assert widget.table_preview.item(0, 0).text() == "1"

    widget._on_background_pipeline_node_finished(
        PipelineNodeResult(
            run_id=414,
            node_id=table_node.id,
            operation_id=table_node.operation_id,
            output=new_table,
            output_state=new_state,
            node_outputs=(new_table,),
            node_output_states=(new_state,),
            execution_state=EXECUTION_READY,
        )
    )

    assert widget.pipeline.outputs[table_node.id] is old_table
    assert widget._node_display_payload(table_node.id)[0] is new_table
    assert widget.table_preview.rowCount() == 2
    assert widget.table_preview.item(0, 0).text() == "2"

    widget._on_background_pipeline_finished(PipelineRunResult(414, {}, cancelled=True))

    assert widget._node_display_payload(table_node.id)[0] is old_table
    assert widget.table_preview.rowCount() == 1
    assert widget.table_preview.item(0, 0).text() == "1"


def test_background_state_overlay_does_not_retain_unretained_payload(qtbot):
    widget = VippWidget(_Viewer(np.zeros((18, 20), dtype=np.float32)))
    qtbot.addWidget(widget)
    qtbot.waitUntil(lambda: widget._active_pipeline_run_id is None, timeout=30_000)
    node = widget.add_node_from_palette("gamma_correction")
    widget.graph_view.select_node("input")
    widget.cache_mode_combo.setCurrentText(CACHE_MODE_LOW_MEMORY)
    assert node.id not in widget._cache_retention_node_ids()
    state = widget.pipeline.output_states["input"]
    completed_output = np.ones((18, 20), dtype=np.float32)
    widget._active_pipeline_run_id = 415

    widget._on_background_pipeline_node_finished(
        PipelineNodeResult(
            run_id=415,
            node_id=node.id,
            operation_id=node.operation_id,
            output=completed_output,
            output_state=state,
            node_outputs=(completed_output,),
            node_output_states=(state,),
            execution_state=EXECUTION_READY,
        )
    )

    assert node.id in widget._background_execution_state_overrides
    assert node.id not in widget._background_node_result_overrides
    assert widget.graph_view._cards[node.id]._execution_state == EXECUTION_READY

    widget._on_background_pipeline_finished(PipelineRunResult(415, {}, cancelled=True))

    assert widget._background_execution_state_overrides == {}
    assert widget._background_node_result_overrides == {}


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

    viewer = _Viewer(
        np.ones((8, 8), dtype=np.uint8) * 20,
        metadata={"axes": "YX"},
    )
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
        preview_size=None,
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
        preview_size=None,
    ):
        calls.append((contrast_mode, contrast_scope))
        return None

    monkeypatch.setattr("napari_vipp._widget.make_preview", fake_make_preview)
    widget.thumbnail_contrast_combo.setCurrentText("Raw")
    widget.thumbnail_scope_combo.setCurrentText("Slice")

    assert calls
    assert ("Raw", "Slice") in calls


def test_thumbnail_controls_default_and_persist_between_widgets(qtbot):
    first = VippWidget(_Viewer(np.ones((4, 8, 8), dtype=np.uint16)))
    qtbot.addWidget(first)

    assert first.thumbnail_resolution_combo.currentData() == "standard"
    assert first._thumbnail_render_size() == (180, 110)
    assert first.thumbnail_statistics_policy_combo.currentData() == "auto"
    assert first._thumbnail_statistics_policy is ThumbnailStatisticsPolicy.AUTO

    qtbot.waitUntil(
        lambda: first._active_thumbnail_contrast_run_id is None,
        timeout=5_000,
    )
    first.preview_mode_combo.setCurrentText("Off")
    first.thumbnail_resolution_combo.setCurrentIndex(
        first.thumbnail_resolution_combo.findData("very_high")
    )
    first.thumbnail_statistics_policy_combo.setCurrentIndex(
        first.thumbnail_statistics_policy_combo.findData("prefer_gpu")
    )

    second = VippWidget(_Viewer(np.ones((4, 8, 8), dtype=np.uint16)))
    qtbot.addWidget(second)

    assert second.thumbnail_resolution_combo.currentData() == "very_high"
    assert second._thumbnail_render_size() == (720, 440)
    assert second.thumbnail_statistics_policy_combo.currentData() == "prefer_gpu"
    assert second._thumbnail_statistics_policy is ThumbnailStatisticsPolicy.PREFER_GPU


def test_thumbnail_detail_changes_render_size_without_recalculation_or_rescan(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer(np.ones((4, 8, 8), dtype=np.uint16)))
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: (
            widget._active_thumbnail_contrast_run_id is None
            and not widget._queued_thumbnail_contrast_limit_requests
            and not widget._pending_thumbnail_contrast_limit_keys
        ),
        timeout=5_000,
    )
    widget.thumbnail_scope_combo.setCurrentText("Slice")
    preview_sizes = []
    normalize_sizes = []

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
        preview_size=None,
    ):
        preview_sizes.append(preview_size)
        return np.zeros((4, 4), dtype=np.uint8)

    def fake_normalize(
        data,
        size=(180, 110),
        *,
        colormap="Gray",
        contrast_mode="Percentile",
        contrast_reference=None,
        contrast_limits=None,
        data_kind="image",
    ):
        normalize_sizes.append(size)
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)

    monkeypatch.setattr("napari_vipp._widget.make_preview", fake_make_preview)
    monkeypatch.setattr(
        "napari_vipp._widget.normalize_thumbnail_with_colormap",
        fake_normalize,
    )
    monkeypatch.setattr(
        widget,
        "run_pipeline",
        lambda: pytest.fail("Thumbnail detail must not recalculate the pipeline"),
    )
    exact_limits = {("limits",): (0.0, 1.0)}
    exact_statistics = {("statistics",): object()}
    # The initial preview may have populated legitimate exact Stack entries
    # before this test switches to Slice. Establish the cache baseline here.
    widget._thumbnail_contrast_limit_cache.clear()
    widget._thumbnail_contrast_statistics_cache.clear()
    widget._thumbnail_contrast_limit_cache.update(exact_limits)
    widget._thumbnail_contrast_statistics_cache.update(exact_statistics)

    widget.thumbnail_resolution_combo.setCurrentIndex(
        widget.thumbnail_resolution_combo.findData("very_high")
    )

    assert preview_sizes
    assert normalize_sizes
    assert set(preview_sizes) == {(720, 440)}
    assert set(normalize_sizes) == {(720, 440)}
    assert widget._thumbnail_contrast_limit_cache == exact_limits
    assert widget._thumbnail_contrast_statistics_cache == exact_statistics
    assert "cached full-stack contrast statistics were retained" in (
        widget.status_label.text()
    )


def test_thumbnail_statistics_effective_policy_honors_global_compute_intent(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.uint16)))
    qtbot.addWidget(widget)

    cases = (
        (ComputeMode.AUTO, ThumbnailStatisticsPolicy.AUTO, ComputeMode.AUTO),
        (
            ComputeMode.PREFER_GPU,
            ThumbnailStatisticsPolicy.AUTO,
            ComputeMode.PREFER_GPU,
        ),
        (
            ComputeMode.CUSTOM,
            ThumbnailStatisticsPolicy.AUTO,
            ComputeMode.AUTO,
        ),
        (
            ComputeMode.AUTO,
            ThumbnailStatisticsPolicy.PREFER_GPU,
            ComputeMode.PREFER_GPU,
        ),
        (ComputeMode.CPU, ThumbnailStatisticsPolicy.PREFER_GPU, ComputeMode.CPU),
        (ComputeMode.PREFER_GPU, ThumbnailStatisticsPolicy.CPU, ComputeMode.CPU),
    )
    for compute_mode, statistics_policy, expected in cases:
        widget._compute_mode = compute_mode
        widget._thumbnail_statistics_policy = statistics_policy
        widget._compute_runtime_quarantined_reason = ""
        assert widget._effective_thumbnail_statistics_compute_mode() is expected

    widget._compute_mode = ComputeMode.PREFER_GPU
    widget._thumbnail_statistics_policy = ThumbnailStatisticsPolicy.PREFER_GPU
    widget._compute_runtime_quarantined_reason = "cleanup failed"
    assert widget._effective_thumbnail_statistics_compute_mode() is ComputeMode.CPU

    widget._compute_runtime_quarantined_reason = ""
    widget._compute_runtime_id = "future-non-cuda-runtime"
    assert widget._effective_thumbnail_statistics_compute_mode() is ComputeMode.CPU


def test_tiny_cpu_thumbnail_statistics_finish_inline_without_busy_ownership(
    qtbot,
):
    data = np.arange(64, dtype=np.uint16).reshape(8, 8)
    widget = VippWidget(_Viewer(data))
    qtbot.addWidget(widget)
    widget.preview_mode_combo.setCurrentText("Off")
    pool = _QueuedThreadPool()
    widget._pipeline_thread_pool = pool
    request = widget._thumbnail_contrast_limit_request(
        "input",
        data,
        widget.pipeline.output_states["input"],
        "Percentile",
        "Stack",
        "image",
    )
    assert request is not None
    widget._queued_thumbnail_contrast_limit_requests[request.key] = request

    widget._start_thumbnail_contrast_limit_run()

    assert pool.workers == []
    assert widget._active_thumbnail_contrast_run_id is None
    assert not widget._thumbnail_contrast_busy_visible
    assert widget.pipeline_busy_label.isHidden()
    result = widget._thumbnail_contrast_statistics_cache[request.key]
    assert result.actual_backend is ThumbnailStatisticsBackend.CPU_NUMPY
    assert request.key in widget._thumbnail_contrast_limit_cache
    assert widget._compute_policy_edit_block_reason() == ""
    assert widget._workflow_tab_switch_block_reason() == ""


def test_tiny_inline_thumbnail_failure_releases_state_and_records_error(
    qtbot,
    monkeypatch,
):
    data = np.arange(64, dtype=np.uint16).reshape(8, 8)
    widget = VippWidget(_Viewer(data))
    qtbot.addWidget(widget)
    widget.preview_mode_combo.setCurrentText("Off")
    pool = _QueuedThreadPool()
    widget._pipeline_thread_pool = pool
    request = widget._thumbnail_contrast_limit_request(
        "input",
        data,
        widget.pipeline.output_states["input"],
        "Percentile",
        "Stack",
        "image",
    )
    assert request is not None
    widget._queued_thumbnail_contrast_limit_requests[request.key] = request

    def fail_calculation(*_args, **_kwargs):
        raise RuntimeError("synthetic inline statistics failure")

    monkeypatch.setattr(
        widget._thumbnail_statistics_engine,
        "calculate",
        fail_calculation,
    )

    widget._start_thumbnail_contrast_limit_run()

    assert pool.workers == []
    assert widget._active_thumbnail_contrast_run_id is None
    assert not widget._pending_thumbnail_contrast_limit_keys
    assert (
        "synthetic inline statistics failure"
        in (widget._thumbnail_contrast_failure_cache[request.key])
    )
    assert widget._compute_policy_edit_block_reason() == ""


@pytest.mark.parametrize("boundary", ("gpu", "large", "many_channels"))
def test_nontrivial_thumbnail_statistics_keep_background_ownership(
    qtbot,
    boundary,
):
    if boundary == "large":
        data = np.zeros(1 * 1024 * 1024 + 1, dtype=np.uint8)
        state = image_state_from_array(data)
    elif boundary == "many_channels":
        data = np.zeros((9, 8, 8), dtype=np.uint16)
        state = image_state_from_array(
            data,
            layer_metadata={
                "axes": [
                    {"name": "c", "type": "channel"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ]
            },
        )
    else:
        data = np.arange(64, dtype=np.uint16).reshape(8, 8)
        state = image_state_from_array(data)

    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.uint16)))
    qtbot.addWidget(widget)
    widget.preview_mode_combo.setCurrentText("Off")
    widget._thumbnail_statistics_policy = (
        ThumbnailStatisticsPolicy.PREFER_GPU
        if boundary == "gpu"
        else ThumbnailStatisticsPolicy.CPU
    )
    if boundary == "gpu":
        widget._compute_runtime_id = "cuda-cupy"
        widget._compute_device_id = "cuda:1"
        widget._compute_device_display_name = "Second RTX"
    pool = _QueuedThreadPool()
    widget._pipeline_thread_pool = pool
    request = widget._thumbnail_contrast_limit_request(
        "input",
        data,
        state,
        "Percentile",
        "Stack",
        "image",
    )
    assert request is not None
    widget._queued_thumbnail_contrast_limit_requests[request.key] = request

    widget._start_thumbnail_contrast_limit_run()

    assert len(pool.workers) == 1
    assert pool.workers[0]._device_id == ("cuda:1" if boundary == "gpu" else "")
    run_id = widget._active_thumbnail_contrast_run_id
    assert run_id is not None
    assert widget._thumbnail_contrast_busy_visible
    widget._on_thumbnail_contrast_limit_finished(
        ThumbnailContrastLimitResult(
            run_id,
            frozenset({request.key}),
            {},
            cancelled=True,
        )
    )
    assert widget._active_thumbnail_contrast_run_id is None


def test_thumbnail_statistics_policy_retries_only_failures_and_keeps_exact_cache(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.uint16)))
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: widget._active_thumbnail_contrast_run_id is None,
        timeout=5_000,
    )
    exact_limits = {("ready",): (1.0, 2.0)}
    exact_statistics = {("ready",): object()}
    widget._thumbnail_contrast_limit_cache.update(exact_limits)
    widget._thumbnail_contrast_statistics_cache.update(exact_statistics)
    widget._thumbnail_contrast_failure_cache[("failed",)] = "old failure"
    refreshes = []
    monkeypatch.setattr(widget, "_update_thumbnails", lambda: refreshes.append(True))

    widget.thumbnail_statistics_policy_combo.setCurrentIndex(
        widget.thumbnail_statistics_policy_combo.findData("cpu")
    )

    assert widget._thumbnail_statistics_policy is ThumbnailStatisticsPolicy.CPU
    assert widget._thumbnail_contrast_limit_cache == exact_limits
    assert widget._thumbnail_contrast_statistics_cache == exact_statistics
    assert widget._thumbnail_contrast_failure_cache == {}
    assert refreshes == [True]


def test_thumbnail_statistics_cache_rejects_reused_object_identity(qtbot):
    data = np.arange(64, dtype=np.uint16).reshape(8, 8)
    widget = VippWidget(_Viewer(data))
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: widget.graph_view.node_has_thumbnail("input"),
        timeout=5_000,
    )
    state = widget.pipeline.output_states["input"]
    request = widget._thumbnail_contrast_limit_request(
        "input",
        data,
        state,
        "Percentile",
        "Stack",
        "image",
    )
    assert request is not None
    impostor = np.zeros_like(data)
    widget._thumbnail_contrast_limit_cache[request.key] = (12.0, 34.0)
    widget._thumbnail_contrast_statistics_cache[request.key] = object()
    widget._thumbnail_contrast_failure_cache[request.key] = "stale"
    widget._thumbnail_contrast_identity_refs[request.key] = weakref.ref(impostor)

    assert widget._register_thumbnail_contrast_identity(request)

    assert request.key not in widget._thumbnail_contrast_limit_cache
    assert request.key not in widget._thumbnail_contrast_statistics_cache
    assert request.key not in widget._thumbnail_contrast_failure_cache
    assert widget._thumbnail_contrast_identity_refs[request.key]() is request.data


def test_encoded_uint8_rgb_thumbnail_is_scan_free(qtbot):
    data = np.full((32, 48, 3), (200, 100, 50), dtype=np.uint8)
    state = image_state_from_array(
        data,
        layer_metadata={
            "axes": [
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
                {"name": "rgb", "type": "channel"},
            ]
        },
    )
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)

    request = widget._thumbnail_contrast_limit_request(
        "input",
        data,
        state,
        "Percentile",
        "Stack",
        "image",
    )

    assert request is None


def test_thumbnail_statistics_progress_and_cancel_use_shared_toolbar(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.uint16)))
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: widget.graph_view.node_has_thumbnail("input"),
        timeout=5_000,
    )
    run_id = 407
    cancel_event = threading.Event()
    widget._active_thumbnail_contrast_run_id = run_id
    widget._active_thumbnail_contrast_cancel_event = cancel_event
    widget._show_thumbnail_contrast_busy(2, 100)
    widget.graph_view.select_node("input")

    assert widget.pipeline_cancel_button.text() == "Cancel thumbnails"
    assert widget.pipeline_cancel_button.isEnabled()
    widget._on_thumbnail_contrast_limit_progress(
        ThumbnailContrastProgress(
            run_id,
            "input",
            1,
            2,
            40,
            50,
            40,
            100,
            "GPU",
            "Exact uint16 histogram",
        )
    )

    assert widget.pipeline_busy_bar.value() == 400
    assert widget.pipeline_busy_bar.format() == "Overall 40%"
    assert "Image Source" in widget.pipeline_busy_label.text()
    assert "GPU" in widget.pipeline_busy_label.text()
    presentation = widget._thumbnail_statistics_presentations["input"]
    assert presentation.kind is ThumbnailStatsBadgeKind.PENDING
    assert not widget.thumbnail_contrast_status_panel.isHidden()
    assert widget.thumbnail_contrast_status_value.text() == "Calculating…"
    assert "in progress" in widget.thumbnail_contrast_status_panel.toolTip()
    queued_data = np.arange(64, dtype=np.uint16).reshape(8, 8)
    queued_request = widget._thumbnail_contrast_limit_request(
        "input",
        queued_data,
        widget.pipeline.output_states["input"],
        "Min-max",
        "Stack",
        "image",
    )
    assert queued_request is not None
    widget._queued_thumbnail_contrast_limit_requests[queued_request.key] = (
        queued_request
    )
    assert queued_request.key in widget._thumbnail_contrast_identity_refs

    widget._cancel_thumbnail_contrast_run()

    assert cancel_event.is_set()
    assert run_id in widget._thumbnail_contrast_discarded_run_ids
    assert queued_request.key not in widget._thumbnail_contrast_identity_refs
    assert widget.pipeline_cancel_button.isHidden()
    assert widget._active_thumbnail_contrast_run_id == run_id

    # A worker may have emitted progress immediately before it observed the
    # cancellation flag. Discarded runs must not resurrect inspector status.
    widget._on_thumbnail_contrast_limit_progress(
        ThumbnailContrastProgress(
            run_id,
            "input",
            1,
            2,
            41,
            50,
            41,
            100,
            "GPU",
            "Synchronizing a discarded result",
        )
    )
    assert "input" not in widget._thumbnail_statistics_presentations
    assert widget.thumbnail_contrast_status_panel.isHidden()

    widget._on_thumbnail_contrast_limit_finished(
        ThumbnailContrastLimitResult(
            run_id,
            frozenset(),
            {},
            cancelled=True,
        )
    )

    assert widget._active_thumbnail_contrast_run_id is None
    assert widget.pipeline_busy_label.isHidden()
    assert "cancelled" in widget.status_label.text().lower()


def test_scientific_edit_preempts_thumbnail_statistics_before_debounce(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.uint16)))
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and widget._active_thumbnail_contrast_run_id is None
        ),
        timeout=5_000,
    )
    widget.preview_mode_combo.setCurrentText("Off")
    widget._background_node_result_overrides.clear()
    widget._background_execution_state_overrides.clear()
    run_id = 518
    cancel_event = threading.Event()
    widget._active_thumbnail_contrast_run_id = run_id
    widget._active_thumbnail_contrast_cancel_event = cancel_event
    widget._pipeline_run_pending = False
    request_key = ("scalar", "input", 518)
    widget._pending_thumbnail_contrast_limit_keys.add(request_key)
    discard_calls = []
    original_discard = widget._discard_pending_thumbnail_contrast_limit_requests

    def record_discard():
        discard_calls.append(True)
        original_discard()

    monkeypatch.setattr(
        widget,
        "_discard_pending_thumbnail_contrast_limit_requests",
        record_discard,
    )

    assert widget._mark_pipeline_dirty("input")
    assert widget._mark_pipeline_dirty("input")

    assert cancel_event.is_set()
    assert run_id in widget._thumbnail_contrast_discarded_run_ids
    assert widget._active_thumbnail_contrast_run_id == run_id
    assert discard_calls == [True]
    # Dirtying scientific intent must not bypass the edit debounce by queuing
    # a run from the thumbnail worker's cleanup callback.
    assert not widget._pipeline_run_pending
    widget._on_thumbnail_contrast_limit_finished(
        ThumbnailContrastLimitResult(
            run_id,
            frozenset((request_key,)),
            {},
            cancelled=True,
        )
    )
    QApplication.processEvents()

    assert widget._active_thumbnail_contrast_run_id is None
    assert not widget._pipeline_run_pending


def test_full_batch_preempts_thumbnail_statistics_then_resumes_once(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.uint16)))
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: widget._active_thumbnail_contrast_run_id is None,
        timeout=5_000,
    )
    widget.preview_mode_combo.setCurrentText("Off")
    dialog = widget._batch_collection_dialog(preview_config=False)
    assert dialog is not None
    run_id = 510
    cancel_event = threading.Event()
    widget._active_thumbnail_contrast_run_id = run_id
    widget._active_thumbnail_contrast_cancel_event = cancel_event
    widget._set_node_thumbnail_statistics_presentation(
        "input",
        ThumbnailStatsBadgeKind.PENDING,
    )
    request_key = ("scalar", "input", 510)
    widget._pending_thumbnail_contrast_limit_keys.add(request_key)
    values = {"collection": "frozen"}

    widget._run_collection_batch_from_workspace(dialog, values)

    assert widget._pending_collection_batch_start == (dialog, values)
    assert cancel_event.is_set()
    assert run_id in widget._thumbnail_contrast_discarded_run_ids
    assert widget.pipeline_cancel_button.text() == "Cancel queued batch"
    assert widget.pipeline_cancel_button.isEnabled()
    assert "input" not in widget._thumbnail_statistics_presentations

    latest_values = {"collection": "edited while queued"}
    monkeypatch.setattr(dialog, "values", lambda: latest_values)
    resumed = []
    monkeypatch.setattr(
        widget,
        "_run_collection_batch_from_workspace",
        lambda queued_dialog, queued_values: resumed.append(
            (queued_dialog, queued_values)
        ),
    )
    widget._on_thumbnail_contrast_limit_finished(
        ThumbnailContrastLimitResult(
            run_id,
            frozenset((request_key,)),
            {},
            cancelled=True,
        )
    )

    qtbot.waitUntil(lambda: bool(resumed), timeout=5_000)
    assert resumed == [(dialog, latest_values)]
    assert widget._pending_collection_batch_start is None


def test_cancel_queued_batch_does_not_resurrect_after_thumbnail_cleanup(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.uint16)))
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: widget._active_thumbnail_contrast_run_id is None,
        timeout=5_000,
    )
    widget.preview_mode_combo.setCurrentText("Off")
    dialog = widget._batch_collection_dialog(preview_config=False)
    assert dialog is not None
    run_id = 511
    cancel_event = threading.Event()
    widget._active_thumbnail_contrast_run_id = run_id
    widget._active_thumbnail_contrast_cancel_event = cancel_event
    widget._run_collection_batch_from_workspace(dialog, {"collection": "frozen"})
    resumed = []
    monkeypatch.setattr(
        widget,
        "_run_collection_batch_from_workspace",
        lambda *_args, **_kwargs: resumed.append(True),
    )

    widget.pipeline_cancel_button.click()

    assert widget._pending_collection_batch_start is None
    assert widget._active_thumbnail_contrast_run_id == run_id
    assert widget.pipeline_cancel_button.isHidden()
    assert "queued full batch cancelled" in widget.status_label.text().lower()
    widget._on_thumbnail_contrast_limit_finished(
        ThumbnailContrastLimitResult(
            run_id,
            frozenset(),
            {},
            cancelled=True,
        )
    )
    QApplication.processEvents()

    assert resumed == []
    assert widget._active_thumbnail_contrast_run_id is None


def test_discarding_queued_batch_dialog_clears_deferred_start(qtbot, monkeypatch):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.uint16)))
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: widget._active_thumbnail_contrast_run_id is None,
        timeout=5_000,
    )
    widget.preview_mode_combo.setCurrentText("Off")
    dialog = widget._batch_collection_dialog(preview_config=False)
    assert dialog is not None
    run_id = 516
    widget._active_thumbnail_contrast_run_id = run_id
    widget._active_thumbnail_contrast_cancel_event = threading.Event()
    widget._run_collection_batch_from_workspace(dialog, {"collection": "frozen"})
    resumed = []
    monkeypatch.setattr(
        widget,
        "_run_collection_batch_from_workspace",
        lambda *_args, **_kwargs: resumed.append(True),
    )

    widget._discard_collection_batch_dialog(dialog)
    widget._on_thumbnail_contrast_limit_finished(
        ThumbnailContrastLimitResult(
            run_id,
            frozenset(),
            {},
            cancelled=True,
        )
    )
    QApplication.processEvents()

    assert widget._pending_collection_batch_start is None
    assert widget._active_collection_batch_dialog is None
    assert resumed == []


def test_pending_scientific_run_wins_without_consuming_thumbnail_worker(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.uint16)))
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: widget._active_thumbnail_contrast_run_id is None,
        timeout=5_000,
    )
    widget.preview_mode_combo.setCurrentText("Off")
    dialog = widget._batch_collection_dialog(preview_config=False)
    assert dialog is not None
    run_id = 512
    cancel_event = threading.Event()
    widget._active_thumbnail_contrast_run_id = run_id
    widget._active_thumbnail_contrast_cancel_event = cancel_event
    widget._pipeline_run_pending = True

    widget._run_collection_batch_from_workspace(dialog, {"collection": "stale"})

    assert widget._pending_collection_batch_start is None
    assert not cancel_event.is_set()
    assert widget._active_thumbnail_contrast_run_id == run_id
    scientific_runs = []
    monkeypatch.setattr(
        widget,
        "run_pipeline",
        lambda: scientific_runs.append(widget._pipeline_run_pending),
    )
    widget._on_thumbnail_contrast_limit_finished(
        ThumbnailContrastLimitResult(run_id, frozenset(), {})
    )

    qtbot.waitUntil(lambda: bool(scientific_runs), timeout=5_000)
    assert scientific_runs == [False]
    assert not widget._pipeline_run_pending
    assert widget._pending_collection_batch_start is None


def test_newer_debounce_owns_resume_after_thumbnail_cleanup(qtbot, monkeypatch):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.uint16)))
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and widget._active_thumbnail_contrast_run_id is None
        ),
        timeout=5_000,
    )
    widget.preview_mode_combo.setCurrentText("Off")
    run_id = 519
    widget._active_thumbnail_contrast_run_id = run_id
    widget._active_thumbnail_contrast_cancel_event = threading.Event()
    widget._pipeline_run_pending = True
    widget._debounce_timer.start()
    scheduled = []
    monkeypatch.setattr(
        "napari_vipp._widget.QTimer.singleShot",
        lambda _delay, callback: scheduled.append(callback),
    )

    widget._on_thumbnail_contrast_limit_finished(
        ThumbnailContrastLimitResult(run_id, frozenset(), {}, cancelled=True)
    )

    assert not widget._pipeline_run_pending
    assert widget._debounce_timer.isActive()
    assert scheduled == []
    widget._debounce_timer.stop()


@pytest.mark.parametrize(
    "owner",
    (
        "pipeline",
        "debounce",
        "source",
        "benchmark",
        "optimizer",
        "batch",
        "closing",
    ),
)
def test_thumbnail_timer_discards_requests_when_exclusive_owner_arrives(
    qtbot,
    owner,
):
    data = np.arange(64, dtype=np.uint16).reshape(8, 8)
    widget = VippWidget(_Viewer(data))
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: widget.graph_view.node_has_thumbnail("input"),
        timeout=5_000,
    )
    widget.preview_mode_combo.setCurrentText("Off")
    request = widget._thumbnail_contrast_limit_request(
        "input",
        data,
        widget.pipeline.output_states["input"],
        "Percentile",
        "Stack",
        "image",
    )
    assert request is not None
    widget._queued_thumbnail_contrast_limit_requests[request.key] = request
    if owner == "pipeline":
        widget._pipeline_run_pending = True
    elif owner == "debounce":
        widget._debounce_timer.start()
    elif owner == "source":
        widget._source_load_pending = True
    elif owner == "benchmark":
        widget._node_benchmark_dialog = SimpleNamespace(running=True)
    elif owner == "optimizer":
        widget._pipeline_optimizer_dialog = SimpleNamespace(running=True)
    elif owner == "batch":
        widget._collection_batch_running = True
    else:
        widget._closing = True

    widget._start_thumbnail_contrast_limit_run()

    assert widget._active_thumbnail_contrast_run_id is None
    assert widget._queued_thumbnail_contrast_limit_requests == {}
    widget._pipeline_run_pending = False
    widget._debounce_timer.stop()
    widget._source_load_pending = False
    widget._node_benchmark_dialog = None
    widget._pipeline_optimizer_dialog = None
    widget._collection_batch_running = False
    widget._closing = False


def test_active_thumbnail_statistics_clearly_block_timing_actions(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.uint16)))
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: widget.graph_view.node_has_thumbnail("input"),
        timeout=5_000,
    )
    widget.preview_mode_combo.setCurrentText("Off")
    widget._compute_mode = ComputeMode.CUSTOM
    widget._active_thumbnail_contrast_run_id = 517

    benchmark_ready, benchmark_reason = widget._can_benchmark_selected_node()
    optimizer_ready, optimizer_reason = widget._can_optimize_pipeline()
    widget._sync_compute_policy_editability()

    assert not benchmark_ready
    assert "thumbnail statistics" in benchmark_reason.lower()
    assert not optimizer_ready
    assert "thumbnail statistics" in optimizer_reason.lower()
    assert not widget.node_benchmark_button.isEnabled()


@pytest.mark.parametrize(
    ("handler_name", "value"),
    (
        ("_on_thumbnail_preview_mode_changed", "Off"),
        ("_on_thumbnail_contrast_mode_changed", "Min-max"),
        ("_on_thumbnail_contrast_scope_changed", "Slice"),
    ),
)
def test_obsolete_thumbnail_presentation_change_cancels_active_statistics(
    qtbot,
    handler_name,
    value,
):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.uint16)))
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: widget._active_thumbnail_contrast_run_id is None,
        timeout=5_000,
    )
    widget.preview_mode_combo.setCurrentText("Off")
    run_id = 513
    cancel_event = threading.Event()
    widget._active_thumbnail_contrast_run_id = run_id
    widget._active_thumbnail_contrast_cancel_event = cancel_event

    getattr(widget, handler_name)(value)

    assert cancel_event.is_set()
    assert run_id in widget._thumbnail_contrast_discarded_run_ids
    assert widget._active_thumbnail_contrast_run_id == run_id


def test_thumbnail_resolution_change_retains_active_statistics(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.uint16)))
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: widget._active_thumbnail_contrast_run_id is None,
        timeout=5_000,
    )
    widget.preview_mode_combo.setCurrentText("Off")
    run_id = 514
    cancel_event = threading.Event()
    widget._active_thumbnail_contrast_run_id = run_id
    widget._active_thumbnail_contrast_cancel_event = cancel_event
    next_index = (
        widget.thumbnail_resolution_combo.currentIndex() + 1
    ) % widget.thumbnail_resolution_combo.count()

    widget._on_thumbnail_resolution_changed(next_index)
    widget._update_thumbnails()

    assert not cancel_event.is_set()
    assert run_id not in widget._thumbnail_contrast_discarded_run_ids
    assert widget._active_thumbnail_contrast_run_id == run_id


def test_batch_completion_requeues_thumbnail_statistics_only_when_safe(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    calls = []
    scheduled = []
    monkeypatch.setattr(widget, "_update_thumbnails", lambda: calls.append(True))
    monkeypatch.setattr(
        "napari_vipp._widget.QTimer.singleShot",
        lambda _delay, callback: scheduled.append(callback),
    )

    widget._resume_thumbnail_statistics_after_batch(
        origin_active=True,
        graph_refresh_pending=False,
    )
    qtbot.waitUntil(lambda: bool(calls), timeout=5_000)
    widget._resume_thumbnail_statistics_after_batch(
        origin_active=True,
        graph_refresh_pending=True,
    )
    widget._resume_thumbnail_statistics_after_batch(
        origin_active=False,
        graph_refresh_pending=False,
    )
    assert len(scheduled) == 1
    calls.clear()
    scheduled[0]()

    assert calls == [True]


@pytest.mark.parametrize(
    ("actual_backend", "fallback_code", "fallback_message", "badge_kind"),
    (
        (
            ThumbnailStatisticsBackend.GPU_CUPY,
            "",
            "",
            ThumbnailStatsBadgeKind.GPU,
        ),
        (
            ThumbnailStatisticsBackend.CPU_NUMPY,
            "gpu-runtime-unavailable",
            "CuPy could not initialize; exact CPU statistics were used.",
            ThumbnailStatsBadgeKind.CPU_FALLBACK,
        ),
    ),
)
def test_thumbnail_statistics_inspector_reports_actual_backend_and_provenance(
    qtbot,
    actual_backend,
    fallback_code,
    fallback_message,
    badge_kind,
):
    data = np.arange(64, dtype=np.uint16).reshape(8, 8)
    widget = VippWidget(_Viewer(data))
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: widget.graph_view.node_has_thumbnail("input"),
        timeout=5_000,
    )
    widget.graph_view.select_node("input")
    state = widget.pipeline.output_states["input"]
    request = widget._thumbnail_contrast_limit_request(
        "input",
        data,
        state,
        "Percentile",
        "Stack",
        "image",
    )
    assert request is not None
    result = _thumbnail_statistics_result(
        data,
        intended_backend=ThumbnailStatisticsBackend.GPU_CUPY,
        actual_backend=actual_backend,
        fallback_reason_code=fallback_code,
        fallback_message=fallback_message,
        input_path=(
            "host_upload"
            if actual_backend is ThumbnailStatisticsBackend.GPU_CUPY
            else ""
        ),
        logical_input_host_to_device_bytes=(
            data.nbytes if actual_backend is ThumbnailStatisticsBackend.GPU_CUPY else 0
        ),
        auxiliary_host_to_device_bytes=(
            40 if actual_backend is ThumbnailStatisticsBackend.GPU_CUPY else 0
        ),
        device_to_host_bytes=(
            24 if actual_backend is ThumbnailStatisticsBackend.GPU_CUPY else 0
        ),
        device_to_host_values=(
            6 if actual_backend is ThumbnailStatisticsBackend.GPU_CUPY else 0
        ),
    )
    widget._thumbnail_contrast_statistics_cache[request.key] = result
    before_compute_badge = widget.graph_view._cards["input"].compute_badge.text()

    widget._sync_node_thumbnail_statistics_presentation(
        "input",
        data,
        state,
        "Percentile",
        "Stack",
        "image",
    )

    card = widget.graph_view._cards["input"]
    presentation = widget._thumbnail_statistics_presentations["input"]
    detail = widget.thumbnail_contrast_status_panel.toolTip()
    summary = widget.thumbnail_contrast_status_panel.statusTip()
    assert presentation.kind is badge_kind
    assert not widget.thumbnail_contrast_status_panel.isHidden()
    assert "Processed 128 B in 0.125 s" in detail
    assert "exact limits cached" in detail
    assert "scientific compute provenance" in detail
    accessible = widget.thumbnail_contrast_status_value.accessibleDescription()
    assert "Processed 128 B in 0.125 s" in accessible
    assert "scientific compute provenance" in accessible
    assert summary == widget.thumbnail_contrast_status_value.statusTip()
    assert summary.startswith("Cached thumbnail contrast")
    assert "\n" not in summary
    assert (
        widget.thumbnail_contrast_status_panel.fontMetrics().horizontalAdvance(summary)
        <= 640
    )
    assert "histogram" not in summary.casefold()
    assert "cuda-cupy" not in summary
    assert "test-gpu" not in summary
    assert "128 B" not in summary
    assert "Auto GPU crossover" not in detail
    assert "Image Source" in (widget.thumbnail_contrast_status_value.accessibleName())
    assert card.compute_badge.text() == before_compute_badge
    assert not hasattr(card, "thumbnail_stats_badge")
    if fallback_message:
        assert widget.thumbnail_contrast_status_value.text() == "CPU fallback"
        assert fallback_message in detail
        assert "used CPU" in summary
        assert "Attempted runtime: cuda-cupy" in detail
    else:
        assert widget.thumbnail_contrast_status_value.text() == "GPU · CuPy"
        assert "GPU · CuPy" in detail
        assert "used GPU" in summary
        assert "Input path: host upload; logical H2D 128 B" in detail
        assert "auxiliary H2D 40 B; D2H 24 B across 6 values" in detail
        assert "host upload" not in summary


def test_thumbnail_status_deduplicates_fallback_and_marks_cached_policy(
    qtbot,
    monkeypatch,
):
    source_data = np.arange(64, dtype=np.uint16).reshape(8, 8)
    widget = VippWidget(_Viewer(source_data))
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: widget.graph_view.node_has_thumbnail("input"),
        timeout=5_000,
    )
    data = widget.pipeline.outputs["input"]
    widget.graph_view.select_node("input")
    state = widget.pipeline.output_states["input"]
    request = widget._thumbnail_contrast_limit_request(
        "input",
        data,
        state,
        "Percentile",
        "Stack",
        "image",
    )
    assert request is not None
    repeated_reason = (
        "Exact GPU thumbnail percentiles currently support native uint8 and "
        "uint16 image data."
    )
    widget._thumbnail_contrast_statistics_cache[request.key] = (
        _thumbnail_statistics_result(
            data,
            intended_backend=ThumbnailStatisticsBackend.GPU_CUPY,
            actual_backend=ThumbnailStatisticsBackend.CPU_NUMPY,
            decision_reason_code="gpu_ineligible",
            decision_reason=repeated_reason,
            fallback_reason_code="gpu_ineligible",
            fallback_message=f"  {repeated_reason.rstrip('.')}  ",
            requested_compute_mode=ComputeMode.PREFER_GPU,
        )
    )
    widget._compute_mode = ComputeMode.PREFER_GPU

    widget._sync_node_thumbnail_statistics_presentation(
        "input",
        data,
        state,
        "Percentile",
        "Stack",
        "image",
    )

    detail = widget.thumbnail_contrast_status_panel.toolTip()
    assert detail.count("Exact GPU thumbnail percentiles") == 1
    assert "Fallback:" in detail
    assert "Selection:" not in detail
    assert widget.thumbnail_contrast_status_value.text() == "CPU fallback"

    widget._compute_mode = ComputeMode.AUTO
    widget._thumbnail_contrast_statistics_cache[request.key] = (
        _thumbnail_statistics_result(
            data,
            requested_compute_mode=ComputeMode.CPU,
        )
    )
    monkeypatch.setattr(widget, "run_pipeline", lambda: None)
    widget.compute_mode_combo.setCurrentIndex(
        widget.compute_mode_combo.findData(ComputeMode.PREFER_GPU.value)
    )

    presentation = widget._thumbnail_statistics_presentations["input"]
    assert widget._compute_mode is ComputeMode.PREFER_GPU
    assert presentation.kind is ThumbnailStatsBadgeKind.CPU
    assert presentation.summary.startswith("Cached thumbnail contrast used CPU")
    assert "Cached producer policy: CPU; current policy: Prefer GPU" in (
        presentation.accessible_description
    )


def test_thumbnail_technical_detail_wraps_without_truncating_accessibility(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.uint16)))
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: widget.graph_view.node_has_thumbnail("input"),
        timeout=5_000,
    )
    widget.graph_view.select_node("input")
    long_reason = " ".join(["technical-provenance-token"] * 30)

    widget._set_node_thumbnail_statistics_presentation(
        "input",
        ThumbnailStatsBadgeKind.CPU_FALLBACK,
        summary="Thumbnail contrast used CPU because the GPU path was unavailable.",
        detail=f"Fallback: {long_reason}",
    )

    tooltip = widget.thumbnail_contrast_status_panel.toolTip()
    accessible = widget.thumbnail_contrast_status_value.accessibleDescription()
    assert all(len(line) <= 88 for line in tooltip.splitlines())
    assert long_reason in accessible
    assert long_reason not in tooltip


def test_thumbnail_detail_shows_crossover_only_for_decisive_auto_choice(qtbot):
    data = np.arange(64, dtype=np.uint16).reshape(8, 8)
    widget = VippWidget(_Viewer(data))
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: widget.graph_view.node_has_thumbnail("input"),
        timeout=5_000,
    )
    widget.graph_view.select_node("input")
    state = widget.pipeline.output_states["input"]
    request = widget._thumbnail_contrast_limit_request(
        "input",
        data,
        state,
        "Percentile",
        "Stack",
        "image",
    )
    assert request is not None
    widget._thumbnail_contrast_statistics_cache[request.key] = (
        _thumbnail_statistics_result(
            data,
            intended_backend=ThumbnailStatisticsBackend.GPU_CUPY,
            actual_backend=ThumbnailStatisticsBackend.GPU_CUPY,
            decision_reason_code="auto_gpu_threshold_met",
            decision_reason="The exact workload met the measured crossover.",
            requested_compute_mode=ComputeMode.AUTO,
        )
    )

    widget._sync_node_thumbnail_statistics_presentation(
        "input",
        data,
        state,
        "Percentile",
        "Stack",
        "image",
    )

    assert "Auto GPU crossover" in widget.thumbnail_contrast_status_panel.toolTip()
    assert "crossover" not in widget.thumbnail_contrast_status_panel.statusTip()


def test_resident_thumbnail_request_is_warm_large_active_image_only(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: widget._active_pipeline_run_id is None,
        timeout=5_000,
    )
    widget.graph_view.select_node("gaussian")
    widget._compute_mode = ComputeMode.PREFER_GPU
    widget.thumbnail_scope_combo.setCurrentText("Stack")
    widget.thumbnail_contrast_combo.setCurrentText("Percentile")

    monkeypatch.setattr(
        widget._thumbnail_statistics_engine,
        "gpu_contract_is_warm",
        lambda _dtype, _mode: False,
    )
    assert (
        widget._resident_thumbnail_statistics_request({"gaussian"}, "gaussian") is None
    )

    monkeypatch.setattr(
        widget._thumbnail_statistics_engine,
        "gpu_contract_is_warm",
        lambda dtype, mode: (
            np.dtype(dtype) == np.dtype(np.float32) and mode == "Percentile"
        ),
    )
    request = widget._resident_thumbnail_statistics_request(
        {"gaussian"},
        "gaussian",
    )

    assert request is not None
    assert request.node_id == "gaussian"
    assert request.output_port == 0
    assert request.contrast_mode == "Percentile"
    assert request.minimum_scanned_bytes == 128 * 1024 * 1024
    assert request.gpu_contract_warm

    monkeypatch.setattr(widget, "run_pipeline", lambda *_args, **_kwargs: None)
    conversion = widget.add_node_from_palette("convert_dtype")
    widget.graph_view.select_node(conversion.id)
    conversion_request = widget._resident_thumbnail_statistics_request(
        {conversion.id},
        conversion.id,
    )
    assert conversion_request is not None
    assert conversion_request.node_id == conversion.id

    widget.thumbnail_scope_combo.setCurrentText("Slice")
    assert (
        widget._resident_thumbnail_statistics_request({"gaussian"}, "gaussian") is None
    )


def test_resident_thumbnail_sidecar_binds_to_host_identity_without_reupload(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: widget._active_pipeline_run_id is None,
        timeout=5_000,
    )
    widget.graph_view.select_node("gaussian")
    widget.thumbnail_scope_combo.setCurrentText("Stack")
    widget.thumbnail_contrast_combo.setCurrentText("Percentile")
    data = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    state = image_state_from_array(data)
    result = _thumbnail_statistics_result(
        data,
        limits=(0.0, 1.0),
        intended_backend=ThumbnailStatisticsBackend.GPU_CUPY,
        actual_backend=ThumbnailStatisticsBackend.GPU_CUPY,
        requested_compute_mode=ComputeMode.PREFER_GPU,
        input_path="resident_borrow",
        logical_input_host_to_device_bytes=0,
        auxiliary_host_to_device_bytes=40,
        device_to_host_bytes=24,
        device_to_host_values=6,
    )
    observation = ResidentThumbnailStatisticsObservation(
        node_id="gaussian",
        output_port=0,
        contrast_mode="Percentile",
        result=result,
    )
    recorded = []
    monkeypatch.setattr(
        widget._thumbnail_statistics_engine,
        "record_resident_gpu_success",
        lambda dtype, mode: recorded.append((np.dtype(dtype), mode)),
    )

    assert widget._cache_resident_thumbnail_statistics(
        observation,
        data,
        state,
        0,
    )
    request = widget._thumbnail_contrast_limit_request(
        "gaussian",
        data,
        state,
        "Percentile",
        "Stack",
        "image",
    )
    assert request is not None
    assert widget._thumbnail_contrast_limit_cache[request.key] == (0.0, 1.0)
    assert widget._thumbnail_contrast_statistics_cache[request.key] == result
    assert recorded == [(np.dtype(np.float32), "Percentile")]
    assert request.key not in widget._queued_thumbnail_contrast_limit_requests

    widget.graph_view.set_thumbnail(
        "gaussian",
        np.full((110, 180, 3), 47, dtype=np.uint8),
    )
    widget._sync_node_thumbnail_statistics_presentation(
        "gaussian",
        data,
        state,
        "Percentile",
        "Stack",
        "image",
    )
    detail = widget.thumbnail_contrast_status_panel.toolTip()
    assert "borrowed resident GPU output" in detail
    assert "no logical input upload" in detail
    assert widget.thumbnail_contrast_status_panel.statusTip().startswith(
        "Cached thumbnail contrast used GPU"
    )

    widget.thumbnail_contrast_combo.setCurrentText("Min-max")
    assert not widget._cache_resident_thumbnail_statistics(
        observation,
        data,
        state,
        0,
    )


def test_resident_thumbnail_handler_rejects_stale_and_dirty_generations(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: widget._active_pipeline_run_id is None,
        timeout=5_000,
    )
    widget.graph_view.select_node("gaussian")
    widget.thumbnail_scope_combo.setCurrentText("Stack")
    widget.thumbnail_contrast_combo.setCurrentText("Percentile")
    data = np.linspace(0.0, 1.0, 80, dtype=np.float32).reshape(8, 10)
    state = image_state_from_array(data)
    observation = ResidentThumbnailStatisticsObservation(
        node_id="gaussian",
        output_port=0,
        contrast_mode="Percentile",
        result=_thumbnail_statistics_result(
            data,
            limits=(0.0, 1.0),
            intended_backend=ThumbnailStatisticsBackend.GPU_CUPY,
            actual_backend=ThumbnailStatisticsBackend.GPU_CUPY,
            requested_compute_mode=ComputeMode.PREFER_GPU,
            input_path="resident_borrow",
        ),
    )
    node_result = PipelineNodeResult(
        run_id=700,
        node_id="gaussian",
        operation_id="gaussian_blur",
        output=data,
        output_state=state,
        node_outputs=(data,),
        node_output_states=(state,),
        execution_state=EXECUTION_READY,
        resident_thumbnail_statistics=(observation,),
    )
    request = widget._thumbnail_contrast_limit_request(
        "gaussian",
        data,
        state,
        "Percentile",
        "Stack",
        "image",
    )
    assert request is not None
    widget._drop_thumbnail_contrast_cache_key(request.key)

    widget._active_pipeline_run_id = 701
    widget._on_background_pipeline_node_finished(node_result)
    assert request.key not in widget._thumbnail_contrast_statistics_cache

    widget._active_pipeline_run_id = 700
    widget._pending_dirty_node_ids.add("gaussian")
    widget._on_background_pipeline_node_finished(node_result)
    assert request.key not in widget._thumbnail_contrast_statistics_cache

    widget._pending_dirty_node_ids.clear()
    widget._on_background_pipeline_node_finished(node_result)
    assert request.key in widget._thumbnail_contrast_statistics_cache


def test_resident_thumbnail_rejects_wrong_channel_limit_count(qtbot):
    widget = VippWidget(_Viewer(np.ones((2, 4, 5), dtype=np.float32)))
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: widget._active_pipeline_run_id is None,
        timeout=5_000,
    )
    widget.thumbnail_scope_combo.setCurrentText("Stack")
    widget.thumbnail_contrast_combo.setCurrentText("Percentile")
    data = np.arange(40, dtype=np.float32).reshape(2, 4, 5)
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("c", "channel"),
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
        ),
    )
    observation = ResidentThumbnailStatisticsObservation(
        node_id="gaussian",
        output_port=0,
        contrast_mode="Percentile",
        result=_thumbnail_statistics_result(
            data,
            limits=(0.0, 39.0),
            intended_backend=ThumbnailStatisticsBackend.GPU_CUPY,
            actual_backend=ThumbnailStatisticsBackend.GPU_CUPY,
            requested_compute_mode=ComputeMode.PREFER_GPU,
            input_path="resident_borrow",
        ),
    )

    assert not widget._cache_resident_thumbnail_statistics(
        observation,
        data,
        state,
        0,
    )


def test_thumbnail_statistics_inspector_follows_selection_without_cross_talk(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.uint16)))
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: widget.graph_view.node_has_thumbnail("input"),
        timeout=5_000,
    )
    widget._clear_thumbnail_statistics_presentations()
    widget.graph_view.select_node("input")
    input_detail = "Thumbnail presentation only. Input used GPU statistics."
    gaussian_detail = "Thumbnail presentation only. Gaussian used CPU fallback."

    widget._set_node_thumbnail_statistics_presentation(
        "input",
        ThumbnailStatsBadgeKind.GPU,
        detail=input_detail,
    )
    assert widget.thumbnail_contrast_status_value.text() == "GPU · CuPy"
    assert widget.thumbnail_contrast_status_panel.toolTip() == input_detail
    assert widget.thumbnail_contrast_status_panel.accessibleName() == (
        "Thumbnail contrast backend for Image Source: GPU · CuPy"
    )

    # A background completion for another node must not overwrite the selected
    # node's inspector. Its status appears only when that node is selected.
    widget._set_node_thumbnail_statistics_presentation(
        "gaussian",
        ThumbnailStatsBadgeKind.CPU_FALLBACK,
        detail=gaussian_detail,
    )
    assert widget.thumbnail_contrast_status_value.text() == "GPU · CuPy"
    assert widget.thumbnail_contrast_status_panel.toolTip() == input_detail

    widget.graph_view.select_node("gaussian")
    assert widget.thumbnail_contrast_status_value.text() == "CPU fallback"
    assert widget.thumbnail_contrast_status_panel.toolTip() == gaussian_detail

    widget._clear_node_thumbnail_statistics_presentation("gaussian")
    assert widget.thumbnail_contrast_status_panel.isHidden()
    assert widget.thumbnail_contrast_status_value.text() == ""
    assert widget.thumbnail_contrast_status_panel.toolTip() == ""
    assert widget.thumbnail_contrast_status_panel.accessibleDescription() == ""
    assert widget.thumbnail_contrast_status_panel.accessibleName() == (
        "Thumbnail contrast status for selected node"
    )

    widget.graph_view.select_node("input")
    assert widget.thumbnail_contrast_status_value.text() == "GPU · CuPy"
    assert widget.thumbnail_contrast_status_panel.toolTip() == input_detail


def test_thumbnail_statistics_inspector_requires_a_rendered_thumbnail(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.uint16)))
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: widget._active_thumbnail_contrast_run_id is None,
        timeout=5_000,
    )
    widget.graph_view.select_node("input")
    widget.pipeline.outputs["input"] = None
    widget.pipeline.output_states["input"] = None

    widget._update_node_thumbnail(
        "input",
        None,
        None,
        0,
        queue_stack_contrast=False,
    )

    assert not widget.graph_view.node_has_thumbnail("input")
    assert "input" not in widget._thumbnail_statistics_presentations
    assert widget.thumbnail_contrast_status_panel.isHidden()
    assert widget.graph_view._cards["input"].preview.toolTip() == ""

    # Even a stale presentation record cannot become visible when global
    # previews are Off or the selected card has no rendered image.
    widget.preview_mode_combo.setCurrentText("Off")
    widget._set_node_thumbnail_statistics_presentation(
        "input",
        ThumbnailStatsBadgeKind.GPU,
        detail="Stale presentation detail.",
    )
    assert widget.thumbnail_contrast_status_panel.isHidden()
    assert widget.graph_view._cards["input"].preview.toolTip() == ""


def test_thumbnail_statistics_failure_keeps_partial_results_and_explains_preview(
    qtbot,
):
    data = np.arange(64, dtype=np.uint16).reshape(8, 8)
    widget = VippWidget(_Viewer(data))
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: widget._active_thumbnail_contrast_run_id is None,
        timeout=5_000,
    )
    widget.preview_mode_combo.setCurrentText("Off")
    successful_key = ("scalar", "input", 408, "successful")
    failed_key = ("scalar", "input", 408, "failed")
    statistics = _thumbnail_statistics_result(data)
    widget._thumbnail_contrast_identity_refs.update(
        {
            successful_key: weakref.ref(data),
            failed_key: weakref.ref(data),
        }
    )
    widget._active_thumbnail_contrast_run_id = 408

    widget._on_thumbnail_contrast_limit_finished(
        ThumbnailContrastLimitResult(
            408,
            frozenset((successful_key, failed_key)),
            {successful_key: statistics.limits},
            statistics={successful_key: statistics},
            errors={failed_key: "GPU and CPU statistics failed"},
        )
    )

    assert widget._thumbnail_contrast_limit_cache[successful_key] == (statistics.limits)
    assert widget._thumbnail_contrast_statistics_cache[successful_key] is statistics
    assert widget._thumbnail_contrast_failure_cache[failed_key] == (
        "GPU and CPU statistics failed"
    )
    assert widget.status_label.property("messageSeverity") == "warning"
    assert "previous complete previews were retained" in widget.status_label.text()
    assert "Affected nodes: Image Source" in widget.status_label.text()

    widget.preview_mode_combo.setCurrentText("Slice")
    widget.graph_view.set_thumbnail(
        "input",
        np.full((110, 180, 3), 51, dtype=np.uint8),
    )
    state = widget.pipeline.output_states["input"]
    request = widget._thumbnail_contrast_limit_request(
        "input",
        data,
        state,
        "Percentile",
        "Stack",
        "image",
    )
    assert request is not None
    widget._thumbnail_contrast_failure_cache[request.key] = "exact scan failed"
    widget.graph_view.select_node("input")
    widget._sync_node_thumbnail_statistics_presentation(
        "input",
        data,
        state,
        "Percentile",
        "Stack",
        "image",
    )
    presentation = widget._thumbnail_statistics_presentations["input"]
    assert presentation.kind is ThumbnailStatsBadgeKind.ERROR
    assert widget.thumbnail_contrast_status_value.text() == "Error"
    assert "exact scan failed" in widget.thumbnail_contrast_status_panel.toolTip()
    assert "Slice contrast" in widget.thumbnail_contrast_status_panel.toolTip()


def test_thumbnail_statistics_cleanup_failure_quarantines_accelerator(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.uint16)))
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: widget._active_thumbnail_contrast_run_id is None,
        timeout=5_000,
    )
    widget.preview_mode_combo.setCurrentText("Off")
    widget._active_thumbnail_contrast_run_id = 409
    widget._thumbnail_contrast_busy_visible = True
    widget._show_thumbnail_contrast_busy(1, 64)

    widget._on_thumbnail_contrast_limit_finished(
        ThumbnailContrastLimitResult(
            409,
            frozenset(),
            {},
            error="CUDA cleanup timed out",
            cleanup_failed=True,
        )
    )

    assert "cleanup failed during thumbnail statistics" in (
        widget._compute_runtime_quarantined_reason
    )
    assert widget._effective_thumbnail_statistics_compute_mode() is ComputeMode.CPU
    assert widget.status_label.property("messageSeverity") == "error"
    assert widget.status_label.property("fullWidthAlert") is True
    assert "Restart VIPP" in widget.status_label.text()


def test_stack_thumbnail_contrast_limits_are_cached(qtbot, monkeypatch):
    data = np.zeros((5, 16, 18), dtype=np.float32)
    data[4, 8, 9] = 10.0
    viewer = _Viewer(data)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    calls = []
    started = threading.Event()
    release = threading.Event()

    class BlockingStatisticsEngine:
        def select(self, request):
            # Exercise background progress rather than the tiny CPU fast path.
            return replace(
                _thumbnail_statistics_result(request.data).decision,
                scanned_bytes=2 * 1024 * 1024,
            )

        def calculate(self, request, *, progress=None):
            calls.append((id(request.data), request.contrast_mode, request.data_kind))
            if len(calls) == 1:
                started.set()
                assert release.wait(5)
            return _thumbnail_statistics_result(request.data)

    widget._thumbnail_statistics_engine = BlockingStatisticsEngine()
    widget._clear_thumbnail_contrast_limit_state()
    widget.thumbnail_scope_combo.setCurrentText("Stack")
    widget.thumbnail_contrast_combo.setCurrentText("Min-max")
    widget._update_thumbnails()

    qtbot.waitUntil(started.is_set, timeout=5_000)
    assert not widget.pipeline_busy_bar.isHidden()
    assert "thumbnail" in widget.pipeline_busy_label.text().lower()

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


@pytest.mark.parametrize("compute_mode", (ComputeMode.CPU, ComputeMode.PREFER_GPU))
def test_pending_stack_thumbnail_keeps_previous_complete_until_exact_limits(
    qtbot,
    monkeypatch,
    compute_mode,
):
    viewer = _Viewer(np.zeros((5, 16, 18), dtype=np.float32))
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: widget.graph_view.node_has_thumbnail("input"),
        timeout=5_000,
    )
    widget._compute_mode = compute_mode
    started = threading.Event()
    release = threading.Event()
    preview_calls = []

    class BlockingStatisticsEngine:
        def select(self, request):
            # Keep the no-intermediate-publication assertion asynchronous.
            return replace(
                _thumbnail_statistics_result(request.data).decision,
                scanned_bytes=2 * 1024 * 1024,
            )

        def calculate(self, request, *, progress=None):
            started.set()
            assert release.wait(5)
            return _thumbnail_statistics_result(request.data)

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
        preview_size=None,
    ):
        preview_calls.append((id(data), contrast_scope, contrast_limits, preview_size))
        return np.zeros((4, 4), dtype=np.uint8)

    widget._thumbnail_statistics_engine = BlockingStatisticsEngine()
    monkeypatch.setattr("napari_vipp._widget.make_preview", fake_make_preview)
    widget._clear_thumbnail_contrast_limit_state()
    widget.thumbnail_scope_combo.setCurrentText("Stack")
    sentinel = np.full((110, 180, 3), 37, dtype=np.uint8)
    widget.graph_view.set_thumbnail("input", sentinel)
    card = widget.graph_view._cards["input"]
    old_pixmap_key = card.preview.source_pixmap().cacheKey()
    target_data_id = id(widget.pipeline.outputs["input"])
    preview_calls.clear()
    widget._finish_pipeline_update(None, "input volume")

    assert not widget.pipeline_busy_bar.isHidden()
    assert "thumbnail" in widget.pipeline_busy_label.text().lower()
    qtbot.waitUntil(started.is_set, timeout=5_000)
    assert [call for call in preview_calls if call[0] == target_data_id] == []
    assert card.preview.source_pixmap().cacheKey() == old_pixmap_key

    release.set()
    qtbot.waitUntil(
        lambda: (
            widget._active_thumbnail_contrast_run_id is None
            and any(call[0] == target_data_id for call in preview_calls)
        ),
        timeout=5_000,
    )
    target_calls = [call for call in preview_calls if call[0] == target_data_id]
    assert target_calls
    assert all(scope == "Stack" for _id, scope, _limits, _size in target_calls)
    assert all(limits == (0.0, 10.0) for _id, _scope, limits, _size in target_calls)
    assert all(size == (180, 110) for _id, _scope, _limits, size in target_calls)
    assert card.preview.source_pixmap().cacheKey() != old_pixmap_key


def test_thumbnail_failure_and_cancellation_preserve_previous_complete_pixels(qtbot):
    source_data = np.arange(64, dtype=np.uint16).reshape(8, 8)
    widget = VippWidget(_Viewer(source_data))
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: widget.graph_view.node_has_thumbnail("input"),
        timeout=5_000,
    )
    data = widget.pipeline.outputs["input"]
    sentinel = np.full((110, 180, 3), 91, dtype=np.uint8)
    widget.graph_view.set_thumbnail("input", sentinel)
    card = widget.graph_view._cards["input"]
    committed_key = card.preview.source_pixmap().cacheKey()
    request = widget._thumbnail_contrast_limit_request(
        "input",
        data,
        widget.pipeline.output_states["input"],
        "Percentile",
        "Stack",
        "image",
    )
    assert request is not None
    widget._thumbnail_contrast_limit_cache.pop(request.key, None)
    widget._thumbnail_contrast_statistics_cache.pop(request.key, None)

    widget._active_thumbnail_contrast_run_id = 501
    widget._on_thumbnail_contrast_limit_finished(
        ThumbnailContrastLimitResult(
            501,
            frozenset({request.key}),
            {},
            errors={request.key: "exact scan failed"},
        )
    )

    assert card.preview.source_pixmap().cacheKey() == committed_key
    assert widget._thumbnail_statistics_presentations["input"].kind is (
        ThumbnailStatsBadgeKind.ERROR
    )

    widget._thumbnail_contrast_failure_cache.clear()
    widget._active_thumbnail_contrast_run_id = 502
    widget._thumbnail_user_cancel_requested_run_id = 502
    widget._on_thumbnail_contrast_limit_finished(
        ThumbnailContrastLimitResult(
            502,
            frozenset({request.key}),
            {},
            cancelled=True,
        )
    )

    assert card.preview.source_pixmap().cacheKey() == committed_key
    assert "Previous complete thumbnails" in widget.status_label.text()

    widget.graph_view.set_thumbnail("input", None)
    current_request = widget._thumbnail_contrast_limit_request(
        "input",
        data,
        widget.pipeline.output_states["input"],
        widget.thumbnail_contrast_combo.currentText(),
        widget.thumbnail_scope_combo.currentText(),
        "image",
    )
    assert current_request is not None
    widget._thumbnail_contrast_failure_cache[current_request.key] = "exact scan failed"
    widget._update_node_thumbnail(
        "input",
        data,
        widget.pipeline.output_states["input"],
        0,
        queue_stack_contrast=False,
    )

    assert not widget.graph_view.node_has_thumbnail("input")
    assert card.preview.text() == "Preview unavailable"
    assert "exact scan failed" in card.preview.accessibleDescription()
    assert "waiting" not in card.preview.accessibleDescription().casefold()


def test_superseded_thumbnail_statistics_cannot_publish_old_generation(qtbot):
    data_a = np.arange(64, dtype=np.uint16).reshape(8, 8)
    data_b = np.arange(64, dtype=np.uint16).reshape(8, 8) * 2
    widget = VippWidget(_Viewer(data_a))
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: widget.graph_view.node_has_thumbnail("input"),
        timeout=5_000,
    )
    sentinel = np.full((110, 180, 3), 123, dtype=np.uint8)
    widget.graph_view.set_thumbnail("input", sentinel)
    card = widget.graph_view._cards["input"]
    committed_key = card.preview.source_pixmap().cacheKey()
    state = widget.pipeline.output_states["input"]
    request_a = widget._thumbnail_contrast_limit_request(
        "input",
        data_a,
        state,
        "Percentile",
        "Stack",
        "image",
    )
    request_b = widget._thumbnail_contrast_limit_request(
        "input",
        data_b,
        state,
        "Percentile",
        "Stack",
        "image",
    )
    assert request_a is not None and request_b is not None
    widget.pipeline.outputs["input"] = data_b
    widget._active_source_load_id = 999
    widget._active_thumbnail_contrast_run_id = 503

    old_statistics = _thumbnail_statistics_result(data_a, limits=(0.0, 63.0))
    widget._on_thumbnail_contrast_limit_finished(
        ThumbnailContrastLimitResult(
            503,
            frozenset({request_a.key}),
            {request_a.key: old_statistics.limits},
            statistics={request_a.key: old_statistics},
        )
    )

    assert request_a.key in widget._thumbnail_contrast_limit_cache
    assert request_b.key not in widget._thumbnail_contrast_limit_cache
    assert card.preview.source_pixmap().cacheKey() == committed_key

    widget._active_source_load_id = None
    new_statistics = _thumbnail_statistics_result(data_b, limits=(0.0, 126.0))
    widget._thumbnail_contrast_limit_cache[request_b.key] = new_statistics.limits
    widget._thumbnail_contrast_statistics_cache[request_b.key] = new_statistics
    widget._update_node_thumbnail(
        "input",
        data_b,
        state,
        0,
        queue_stack_contrast=True,
    )

    assert card.preview.source_pixmap().cacheKey() != committed_key


def test_finish_refreshes_selected_inspection_layer_only_once(qtbot, monkeypatch):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget.preview_mode_combo.setCurrentText("Off")
    calls = []
    monkeypatch.setattr(
        widget,
        "_refresh_inspection_layer_if_active",
        lambda: calls.append("refresh-active"),
    )
    monkeypatch.setattr(
        widget,
        "_inspect_selected_node",
        lambda: calls.append("inspect-selected"),
    )

    widget._finish_pipeline_update(None, "input volume")

    assert calls == ["inspect-selected"]


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
        preview_size=None,
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
    widget._compute_mode = ComputeMode.CPU
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
    widget._compute_mode = ComputeMode.CPU
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


def test_macos_memory_uses_native_vm_statistics(monkeypatch):
    class _Function:
        def __init__(self, callback):
            self.callback = callback
            self.restype = None

        def __call__(self, *args):
            return self.callback(*args)

    class _LibSystem:
        mach_host_self = _Function(lambda: 7)

        @staticmethod
        def _host_statistics(_host, _flavor, statistics, count):
            statistics._obj.free_count = 100
            statistics._obj.inactive_count = 300
            assert count._obj.value == 38
            return 0

        host_statistics64 = _Function(_host_statistics)

    values = {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 2000}
    monkeypatch.setattr(
        "napari_vipp._widget.os.sysconf",
        values.__getitem__,
        raising=False,
    )
    monkeypatch.setattr(
        "napari_vipp._widget.ctypes.CDLL",
        lambda _path: _LibSystem(),
    )

    assert _macos_memory_bytes() == (400 * 4096, 2000 * 4096)


def test_system_memory_uses_windows_backend_without_sysconf(monkeypatch):
    expected = (3_000_000, 8_000_000)

    def unexpected_sysconf(_name):
        raise AssertionError("Windows memory reporting must not use os.sysconf")

    monkeypatch.setattr("napari_vipp._widget.sys.platform", "win32")
    monkeypatch.setattr(
        "napari_vipp._widget.os.sysconf",
        unexpected_sysconf,
        raising=False,
    )
    monkeypatch.setattr("napari_vipp._widget._windows_memory_bytes", lambda: expected)

    assert _system_memory_bytes() == expected


def test_system_memory_uses_macos_backend(monkeypatch):
    expected = (5_000_000, 12_000_000)
    monkeypatch.setattr("napari_vipp._widget.sys.platform", "darwin")
    monkeypatch.setattr("napari_vipp._widget._macos_memory_bytes", lambda: expected)

    assert _system_memory_bytes() == expected


def test_system_memory_handles_missing_posix_sysconf(monkeypatch):
    monkeypatch.setattr("napari_vipp._widget.sys.platform", "linux")
    monkeypatch.delattr("napari_vipp._widget.os.sysconf", raising=False)

    assert _system_memory_bytes() == (None, None)


def test_macos_memory_handles_missing_sysconf(monkeypatch):
    monkeypatch.delattr("napari_vipp._widget.os.sysconf", raising=False)

    assert _macos_memory_bytes() == (None, None)


def test_windows_memory_uses_global_memory_status(monkeypatch):
    class _Kernel32:
        @staticmethod
        def GlobalMemoryStatusEx(status):
            status._obj.ullAvailPhys = 3_000_000
            status._obj.ullTotalPhys = 8_000_000
            return 1

    class _Windll:
        kernel32 = _Kernel32()

    monkeypatch.setattr("napari_vipp._widget.ctypes.windll", _Windll(), raising=False)

    assert _windows_memory_bytes() == (3_000_000, 8_000_000)


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
    widget._compute_mode = ComputeMode.CPU
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


def test_composite_edit_preserves_upstream_manual_deconvolution_cache(qtbot):
    data = np.arange(8 * 9, dtype=np.float32).reshape(8, 9)
    widget = VippWidget(_Viewer(data, metadata={"axes": "YX"}))
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    widget._compute_mode = ComputeMode.CPU
    qtbot.addWidget(widget)

    deconvolution = widget.add_node_from_palette("richardson_lucy_deconvolution")
    combined = widget.add_node_from_palette("combine_channels")
    composite = widget.add_node_from_palette("composite_to_rgb")
    widget._connect_nodes("input", deconvolution.id, target_port=0)
    widget._connect_nodes("input", deconvolution.id, target_port=1)
    widget._connect_nodes(deconvolution.id, combined.id, target_port=0)
    widget._connect_nodes(deconvolution.id, combined.id, target_port=1)
    widget._connect_nodes(combined.id, composite.id)

    # Seed the explicitly calculated/manual result so this regression exercises
    # downstream invalidation and retention without running deconvolution in the
    # test itself.
    manual_output = np.asarray(data).copy()
    manual_state = widget.pipeline.output_states["input"]
    widget.pipeline.outputs[deconvolution.id] = manual_output
    widget.pipeline.output_states[deconvolution.id] = manual_state
    widget.pipeline.node_outputs[deconvolution.id] = [manual_output]
    widget.pipeline.node_output_states[deconvolution.id] = [manual_state]
    widget.pipeline.completed_node_ids.add(deconvolution.id)
    widget.pipeline.node_execution_states[deconvolution.id] = EXECUTION_READY
    widget.pipeline.node_execution_messages[deconvolution.id] = ""

    widget._mark_pipeline_dirty(combined.id)
    widget.run_pipeline(force_sync=True)
    widget.graph_view.select_node(composite.id)
    widget.cache_mode_combo.setCurrentText(CACHE_MODE_LOW_MEMORY)

    assert widget.pipeline.outputs[deconvolution.id] is manual_output
    assert widget.pipeline.node_execution_states[deconvolution.id] == EXECUTION_READY

    widget._parameter_widgets["mapping_mode"].combo.setCurrentText("Manual")
    widget._debounce_timer.stop()
    widget.run_pipeline(force_sync=True)

    assert widget.pipeline.outputs[deconvolution.id] is manual_output
    assert widget.pipeline.node_outputs[deconvolution.id][0] is manual_output
    assert widget.pipeline.node_execution_states[deconvolution.id] == EXECUTION_READY
    assert widget.pipeline.outputs[composite.id] is not None


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


def test_rerouting_tunnel_source_is_undoable_and_updates_badges(qtbot, monkeypatch):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    monkeypatch.setattr(widget, "_prompt_tunnel_name", lambda *_args: "Raw")
    widget._create_output_tunnel("input", 0)
    widget._connect_input_to_tunnel("Raw", "threshold", 0)

    assert widget._reroute_output_tunnel("Raw", "gaussian", 0)

    tunnel = widget.pipeline.output_tunnel("Raw")
    assert tunnel == OutputTunnel("Raw", "gaussian", 0)
    connection = widget.pipeline.tunnel_connection_for_input("threshold", 0)
    assert connection is not None
    assert connection.source_id == "gaussian"
    assert widget.graph_view._proxies["input"].output_port_at(0)._tunnel_label == ""
    assert (
        widget.graph_view._proxies["gaussian"].output_port_at(0)._tunnel_label == "Raw"
    )

    widget.undo()

    tunnel = widget.pipeline.output_tunnel("Raw")
    assert tunnel == OutputTunnel("Raw", "input", 0)
    connection = widget.pipeline.tunnel_connection_for_input("threshold", 0)
    assert connection is not None
    assert connection.source_id == "input"

    widget.redo()

    assert widget.pipeline.output_tunnel("Raw") == OutputTunnel("Raw", "gaussian", 0)


def test_tunnel_reroute_preview_rejects_cycle(qtbot, monkeypatch):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    monkeypatch.setattr(widget, "_prompt_tunnel_name", lambda *_args: "Raw")
    widget._create_output_tunnel("input", 0)
    widget._connect_input_to_tunnel("Raw", "gaussian", 0)

    state, message = widget._tunnel_reroute_preview_state("Raw", "threshold", 0)

    assert state == "incompatible"
    assert "cycle" in message.casefold()


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
    assert dialog.tree.palette().base().color().name() == "#1f242c"
    assert dialog.tree.palette().alternateBase().color().name() == "#252b35"

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
    assert len(widget.pipeline.connections) == len(set(widget.pipeline.connections))
    assert widget.graph_view.selected_node_ids() == (node.id,)
    assert widget.graph_view.node_position("gaussian").x() > target_before.x()
    assert "Inserted existing" in widget.status_label.text()

    inserted_position = QPointF(widget.graph_view.node_position(node.id))
    target_after = QPointF(widget.graph_view.node_position("gaussian"))

    def assert_position_restored(node_id: str, expected: QPointF) -> None:
        actual = widget.graph_view.node_position(node_id)
        assert actual is not None
        # QGraphicsProxyWidget can settle onto a neighboring quarter-pixel on
        # macOS while preserving the same visible scene position.
        assert actual.x() == pytest.approx(expected.x(), abs=0.5)
        assert actual.y() == pytest.approx(expected.y(), abs=0.5)

    widget.undo()

    assert node.id in widget.pipeline.nodes
    assert_position_restored(node.id, old_pos)
    assert_position_restored("gaussian", target_before)
    assert ("input", "gaussian") in {
        (connection.source_id, connection.target_id)
        for connection in widget.pipeline.connections
    }
    assert not any(
        connection.source_id == node.id or connection.target_id == node.id
        for connection in widget.pipeline.connections
    )

    widget.redo()

    assert_position_restored(node.id, inserted_position)
    assert_position_restored("gaussian", target_after)
    assert GraphConnection("input", node.id) in widget.pipeline.connections
    assert GraphConnection(node.id, "gaussian") in widget.pipeline.connections
    assert GraphConnection("input", "gaussian") not in widget.pipeline.connections
    assert len(widget.pipeline.connections) == len(set(widget.pipeline.connections))


def test_pasted_loose_node_can_be_spliced_without_duplicate_connections(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    monkeypatch.setattr(widget, "run_pipeline", lambda *args, **kwargs: None)
    QApplication.clipboard().clear()
    widget._copy_graph_nodes(("gaussian",))
    (node_id,) = widget._paste_graph_fragment(QPointF(900.0, 500.0))
    assert not widget._node_has_connections(node_id)
    old_pos = QPointF(widget.graph_view.node_position(node_id))
    widget.graph_view.center_node_on(node_id, QPointF(180.0, 100.0))
    widget._history.clear()

    result = widget._insert_existing_node_on_connection(
        node_id,
        ("input", "gaussian", 0, 0),
        old_pos,
        widget.graph_view.node_position(node_id),
    )

    assert result is widget.pipeline.nodes[node_id]
    expected_splice = {
        GraphConnection("input", node_id),
        GraphConnection(node_id, "gaussian"),
    }
    assert expected_splice <= set(widget.pipeline.connections)
    assert GraphConnection("input", "gaussian") not in widget.pipeline.connections
    assert len(widget.pipeline.connections) == len(set(widget.pipeline.connections))
    assert widget.graph_view.selected_node_ids() == (node_id,)


@pytest.mark.parametrize("failed_connect_call", [1, 2])
def test_existing_node_splice_failure_restores_wire_and_layout(
    qtbot,
    monkeypatch,
    failed_connect_call,
):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    monkeypatch.setattr(widget, "run_pipeline", lambda *args, **kwargs: None)
    node = widget.add_node_from_palette("median_filter")
    old_pos = QPointF(widget.graph_view.node_position(node.id))
    original_positions = widget.graph_view.node_positions()
    original_connections = tuple(widget.pipeline.connections)
    widget.graph_view.center_node_on(node.id, old_pos + QPointF(160.0, 90.0))
    widget._history.clear()
    original_connect = widget.pipeline.connect
    connect_calls = 0

    def fail_selected_connect(*args, **kwargs):
        nonlocal connect_calls
        connect_calls += 1
        if connect_calls == failed_connect_call:
            return SimpleNamespace(
                success=False,
                message="injected splice connection failure",
                removed=(),
                connection=None,
            )
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(widget.pipeline, "connect", fail_selected_connect)

    result = widget._insert_existing_node_on_connection(
        node.id,
        ("input", "gaussian", 0, 0),
        old_pos,
        widget.graph_view.node_position(node.id),
    )

    assert result is None
    assert tuple(widget.pipeline.connections) == original_connections
    assert widget.graph_view.node_positions() == original_positions
    assert len(widget.graph_view._connections) == len(original_connections)
    assert len(widget._undo_stack) == 0
    assert "injected splice connection failure" in widget.status_label.text()


def test_incompatible_existing_node_drop_keeps_original_wire(qtbot, monkeypatch):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    monkeypatch.setattr(widget, "run_pipeline", lambda *args, **kwargs: None)
    node = widget.add_node_from_palette("fill_holes")
    old_pos = QPointF(widget.graph_view.node_position(node.id))
    original_connections = tuple(widget.pipeline.connections)
    widget.graph_view.center_node_on(node.id, old_pos + QPointF(80.0, 40.0))

    result = widget._insert_existing_node_on_connection(
        node.id,
        ("input", "gaussian", 0, 0),
        old_pos,
        widget.graph_view.node_position(node.id),
    )

    assert result is None
    assert tuple(widget.pipeline.connections) == original_connections
    assert not widget._node_has_connections(node.id)
    assert "Cannot feed image output" in widget.status_label.text()


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


def test_connection_insert_dialog_uses_subtle_alternating_rows(qtbot):
    candidate = ConnectionInsertCandidate(
        operation_id="gaussian_blur",
        title="Gaussian Blur",
        category="Filtering",
        subcategory="",
        mode="full",
        detail="Insert Gaussian Blur.",
        search_text="gaussian blur filtering",
    )
    dialog = ConnectionInsertDialog([candidate])
    qtbot.addWidget(dialog)

    base = dialog.tree.palette().base().color()
    alternate = dialog.tree.palette().alternateBase().color()

    assert base.name() == "#1f242c"
    assert alternate.name() == "#252b35"
    assert abs(base.lightness() - alternate.lightness()) <= 10


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

    expanded_width = widget._expanded_toolbar_required_width()
    widget.resize(expanded_width + 100, 600)
    widget._sync_toolbar_responsive_mode()

    assert widget.settings_menu_button.isHidden() is False
    assert widget.settings_menu_button.text() == "Settings"
    assert widget.settings_menu_button.minimumWidth() >= 96
    assert widget.background_all_checkbox.isHidden()
    assert widget.follow_dims_checkbox.isHidden()
    assert widget.thumbnail_toolbar_group.isHidden() is False
    assert widget.preview_mode_combo.isHidden() is False
    assert widget.thumbnail_contrast_combo.isHidden() is False
    assert widget.thumbnail_scope_combo.isHidden() is False
    assert widget.graph_zoom_slider.isHidden() is False
    assert widget.compute_toolbar_group.isHidden() is False
    assert widget.compute_status_label.isHidden() is False
    assert widget.save_workflow_button.isHidden() is False
    assert widget.export_button.isHidden() is False
    assert widget.auto_structure_button.text() == "Auto structure graph"

    widget.resize(1400, 600)
    widget._sync_toolbar_responsive_mode()

    assert widget.thumbnail_toolbar_group.isHidden()
    assert widget.preview_mode_combo.isHidden() is False
    assert widget.thumbnail_contrast_combo.isHidden() is False
    assert widget.thumbnail_scope_combo.isHidden() is False
    assert widget.thumbnail_colormap_combo.isHidden() is False
    assert widget.zoom_toolbar_field.isHidden() is False
    assert widget.graph_zoom_slider.isHidden() is False
    assert widget.compute_toolbar_group.isHidden() is False
    assert widget.compute_status_label.isHidden()

    widget.resize(1200, 600)
    widget._sync_toolbar_responsive_mode()

    assert widget.graph_zoom_slider.isHidden() is False
    assert widget.graph_zoom_reset_button.isHidden() is False
    assert widget.graph_zoom_label.isHidden() is False
    assert widget.compute_toolbar_group.isHidden()
    assert widget.compute_status_label.isHidden()

    widget.resize(1000, 600)
    widget._sync_toolbar_responsive_mode()

    assert widget.zoom_toolbar_field.isHidden()
    assert widget.graph_zoom_slider.isHidden() is False
    assert widget.graph_zoom_reset_button.isHidden() is False
    assert widget.graph_zoom_label.isHidden() is False
    assert widget.compute_toolbar_group.isHidden()
    assert widget.compute_status_label.isHidden()

    widget.resize(expanded_width + 100, 600)
    widget._sync_toolbar_responsive_mode()

    assert widget.settings_menu_button.isHidden() is False
    assert widget.background_all_checkbox.isHidden()
    assert widget.follow_dims_checkbox.isHidden()
    assert widget.thumbnail_toolbar_group.isHidden() is False
    assert widget.preview_mode_combo.isHidden() is False
    assert widget.thumbnail_contrast_combo.isHidden() is False
    assert widget.thumbnail_scope_combo.isHidden() is False
    assert widget.graph_zoom_slider.isHidden() is False
    assert widget.save_workflow_button.isHidden() is False
    assert widget.export_button.isHidden() is False
    assert widget.compute_status_label.isHidden() is False
    assert widget.auto_structure_button.text() == "Auto structure graph"


@pytest.mark.parametrize(
    ("width", "compute_visible", "zoom_visible"),
    (
        (1400, True, True),
        (1200, False, True),
        (1100, False, False),
        (1051, False, False),
        (1050, False, False),
        (1000, False, False),
    ),
)
def test_toolbar_compute_stages_preserve_primary_action_width(
    qtbot,
    width,
    compute_visible,
    zoom_visible,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget.setFixedWidth(width)
    widget.show()
    qtbot.waitExposed(widget)
    widget._sync_toolbar_responsive_mode()
    QApplication.processEvents()

    assert widget.compute_toolbar_group.isHidden() is (not compute_visible)
    assert widget.compute_status_label.isHidden()
    assert widget.zoom_toolbar_field.isHidden() is (not zoom_visible)
    assert (
        widget.calculate_all_button.width()
        >= widget.calculate_all_button.minimumSizeHint().width()
    )
    assert widget.settings_menu_button.geometry().right() < widget.width()


def test_long_compute_summary_collapses_before_compressing_primary_action(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget.setFixedWidth(1600)
    widget.show()
    qtbot.waitExposed(widget)

    def decision(node_id, *, gpu=False, fallback=False):
        return NodeExecutionDecision(
            node_id,
            "median_filter",
            NodeComputePreference(),
            "cuda-cupy" if gpu else "cpu-numpy",
            "cupyx" if gpu else "cpu",
            f"{'cupyx' if gpu else 'cpu'}-{node_id}-v1",
            DecisionKind.FALLBACK_CPU
            if fallback
            else (DecisionKind.SELECTED if gpu else DecisionKind.POLICY_CPU),
            DecisionReason.VISIBLE_FALLBACK
            if fallback
            else (
                DecisionReason.SELECTED_IMPLEMENTATION
                if gpu
                else DecisionReason.AUTO_CPU
            ),
            "Synthetic responsive-toolbar decision.",
            fallback_reason=(
                FallbackReason.DEPENDENCY_UNAVAILABLE
                if fallback
                else FallbackReason.NONE
            ),
        )

    decisions = {
        "gpu": decision("gpu", gpu=True),
        "cpu-1": decision("cpu-1"),
        "cpu-2": decision("cpu-2"),
        "cpu-3": decision("cpu-3"),
        "fallback": decision("fallback", fallback=True),
    }
    widget._accepted_compute_decisions = decisions
    widget._compute_decision_environments = {
        "gpu": ComputeEnvironment(
            runtime_ids=("cpu-numpy", "cuda-cupy"),
            implementation_libraries=("cpu", "cupyx"),
            device_id="cuda:0",
            device_name="Test GPU",
            device_class="nvidia-cuda",
        )
    }
    widget._sync_compute_toolbar_summary()
    QApplication.processEvents()

    assert widget.compute_status_label.text() == ("Auto · 1 GPU / 4 CPU · 1 fallback")
    assert widget.compute_toolbar_group.isHidden() is False
    assert widget.compute_status_label.isHidden()
    assert (
        widget.calculate_all_button.width()
        >= widget.calculate_all_button.minimumSizeHint().width()
    )

    widget._accepted_compute_decisions = {"cpu-1": decisions["cpu-1"]}
    widget._sync_compute_toolbar_summary()
    QApplication.processEvents()

    assert widget.compute_status_label.text() == "Auto · 1 CPU"
    assert widget.compute_status_label.isHidden() is False


def test_load_precedes_save_and_batch_is_separated_from_exports(
    qtbot,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    layout = widget.workflow_toolbar_layout

    load_index = layout.indexOf(widget.load_workflow_button)
    save_index = layout.indexOf(widget.save_workflow_button)
    left_separator_index = layout.indexOf(widget._batch_toolbar_left_separator)
    batch_index = layout.indexOf(widget.batch_button)
    leave_batch_index = layout.indexOf(widget.leave_batch_button)
    right_separator_index = layout.indexOf(widget._batch_toolbar_right_separator)
    export_index = layout.indexOf(widget.export_button)
    export_ome_index = layout.indexOf(widget.export_ome_button)

    assert (
        load_index
        < save_index
        < left_separator_index
        < batch_index
        < leave_batch_index
        < right_separator_index
        < export_index
        < export_ome_index
    )
    assert isinstance(widget._batch_toolbar_left_separator, QFrame)
    assert isinstance(widget._batch_toolbar_right_separator, QFrame)
    assert widget.leave_batch_button.isHidden()

    widget.batch_navigator.set_session(2, 0, "0001_a", ["a.npy"])
    batch_buttons = [
        button
        for button in widget.findChildren(QPushButton)
        if button.text() == "Batch workspace..."
    ]
    assert batch_buttons == [widget.batch_button]

    dialog = CollectionBatchDialog()
    qtbot.addWidget(dialog)
    config_layout = dialog.load_config_button.parentWidget().layout()
    assert config_layout.indexOf(dialog.load_config_button) < config_layout.indexOf(
        dialog.save_config_button
    )


def test_toolbar_field_pairs_stay_adjacent_and_responsive(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget.resize(widget._expanded_toolbar_required_width() + 100, 600)
    widget.show()
    qtbot.waitExposed(widget)
    widget._sync_toolbar_responsive_mode()
    QApplication.processEvents()

    fields = (
        (
            widget.preview_toolbar_label,
            widget.preview_mode_combo,
        ),
        (
            widget.contrast_toolbar_label,
            widget.thumbnail_contrast_combo,
        ),
        (
            widget.contrast_range_toolbar_label,
            widget.thumbnail_scope_combo,
        ),
        (
            widget.mono_toolbar_label,
            widget.thumbnail_colormap_combo,
        ),
        (
            widget.zoom_toolbar_label,
            widget.zoom_toolbar_controls,
        ),
        (
            widget.compute_toolbar_label,
            widget.compute_mode_combo,
        ),
    )
    controls = [control for _label, control in fields]

    def rect_in_toolbar(child):
        return child.rect().translated(child.mapTo(widget, QPoint(0, 0)))

    def horizontal_gap(first, second):
        if first.right() < second.left():
            return second.left() - first.right()
        if second.right() < first.left():
            return first.left() - second.right()
        return 0

    label_left_edges = []
    for label, control in fields:
        assert label.alignment() & Qt.AlignRight
        label_rect = rect_in_toolbar(label)
        control_rect = rect_in_toolbar(control)
        label_left_edges.append(label_rect.left())
        own_gap = horizontal_gap(label_rect, control_rect)
        assert 0 < own_gap <= 6
        other_gaps = [
            horizontal_gap(label_rect, rect_in_toolbar(other))
            for other in controls
            if other is not control
        ]
        assert own_gap < min(other_gaps)

    assert label_left_edges[0] == min(label_left_edges)

    widget.hide()
    widget.setFixedWidth(1400)
    widget._sync_toolbar_responsive_mode()
    assert widget.thumbnail_toolbar_group.isHidden()
    assert widget.zoom_toolbar_field.isHidden() is False
    assert widget.compute_toolbar_group.isHidden() is False
    assert widget.compute_status_label.isHidden()
    assert widget._toolbar_zoom_separator.isHidden()
    assert widget._toolbar_action_separator.isHidden() is False

    widget.setFixedWidth(1000)
    widget._sync_toolbar_responsive_mode()
    assert widget.thumbnail_toolbar_group.isHidden()
    assert widget.zoom_toolbar_field.isHidden()
    assert widget.compute_toolbar_group.isHidden()
    assert widget.compute_status_label.isHidden()
    assert widget._toolbar_zoom_separator.isHidden()
    assert widget._toolbar_action_separator.isHidden()

    widget.setFixedWidth(700)
    widget._sync_toolbar_responsive_mode()
    assert widget.compute_toolbar_group.isHidden()
    assert widget.compute_status_label.isHidden()
    assert widget._toolbar_compute_separator.isHidden()
    assert widget.settings_menu_button.isHidden() is False


def test_toolbar_fields_do_not_clip_with_larger_font(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    font = widget.font()
    font.setPointSizeF(max(font.pointSizeF() * 1.5, 13.0))
    widget.setFont(font)
    expanded_width = widget._expanded_toolbar_required_width()
    widget.setFixedWidth(expanded_width + 100)
    widget.resize(expanded_width + 100, 700)
    widget.show()
    qtbot.waitExposed(widget)
    widget._sync_toolbar_responsive_mode()
    QApplication.processEvents()

    labels = (
        widget.preview_toolbar_label,
        widget.contrast_toolbar_label,
        widget.contrast_range_toolbar_label,
        widget.mono_toolbar_label,
        widget.zoom_toolbar_label,
        widget.compute_toolbar_label,
    )
    combos = (
        widget.preview_mode_combo,
        widget.thumbnail_contrast_combo,
        widget.thumbnail_scope_combo,
        widget.thumbnail_colormap_combo,
    )
    for label in labels:
        assert label.width() >= label.sizeHint().width()
        assert label.height() >= label.sizeHint().height()
    for combo in combos:
        assert combo.width() >= combo.minimumSizeHint().width()
        assert combo.height() >= combo.minimumSizeHint().height()
    assert widget.settings_menu_button.geometry().right() < widget.width()


def test_settings_menu_shows_controls_hidden_at_current_stage(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.resize(widget._expanded_toolbar_required_width() + 100, 600)
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
    assert "Compute setup and memory…" in labels
    assert "Port labels" in labels
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


def test_settings_menu_controls_port_label_mode(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.port_label_mode_combo.currentText() == "Ambiguous only"
    assert widget.graph_view.port_label_mode == PortLabelMode.AMBIGUOUS_ONLY
    widget._populate_settings_toolbar_menu()
    submenu = next(
        action.menu()
        for action in widget.settings_menu.actions()
        if action.text() == "Port labels"
    )
    actions = {action.text(): action for action in submenu.actions()}
    assert list(actions) == ["Ambiguous only", "Show all", "Hide all"]

    actions["Show all"].trigger()

    assert widget.port_label_mode_combo.currentText() == "Show all"
    assert widget.graph_view.port_label_mode == PortLabelMode.SHOW_ALL
    assert widget.status_label.text().startswith("Port labels set to Show all")

    widget._populate_settings_toolbar_menu()
    submenu = next(
        action.menu()
        for action in widget.settings_menu.actions()
        if action.text() == "Port labels"
    )
    checked = [action.text() for action in submenu.actions() if action.isChecked()]
    assert checked == ["Show all"]


def test_palette_registry_nodes_are_constructible():
    pipeline = PrototypePipeline()
    palette_ids = [spec.id for spec in PALETTE_NODE_LIBRARY]

    assert len(palette_ids) == len(set(palette_ids))
    for spec in PALETTE_NODE_LIBRARY:
        node = pipeline.add_node(spec.id)
        assert node.operation_id == spec.id
        expected_params = {param.name: param.default for param in spec.parameters}
        if spec.id == "composite_to_rgb":
            expected_params.update(
                {
                    "channel_axis_mode": "Auto",
                    "mapping_mode": "Auto",
                }
            )
        assert node.params == expected_params
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


def test_tune_node_in_isolation_marks_and_holds_automatic_descendants(
    qtbot,
    monkeypatch,
):
    image = np.arange(100, dtype=np.float32).reshape(10, 10)
    widget = VippWidget(_Viewer(image, metadata={"axes": "YX"}))
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    widget._compute_mode = ComputeMode.CPU
    qtbot.addWidget(widget)
    widget.run_pipeline(force_sync=True)
    widget.graph_view.select_node("gaussian")
    cached_threshold = widget.pipeline.outputs["threshold"]
    calls: list[str] = []
    original_run_node = widget.pipeline._run_node

    def counted_run_node(node_id, *args, **kwargs):
        calls.append(node_id)
        return original_run_node(node_id, *args, **kwargs)

    monkeypatch.setattr(widget.pipeline, "_run_node", counted_run_node)

    inspector_layout = widget.inspector_content.layout()
    assert inspector_layout.indexOf(widget.isolated_tuning_checkbox) == (
        inspector_layout.indexOf(widget.keep_cached_checkbox) + 1
    )
    widget.isolated_tuning_checkbox.setChecked(True)
    widget.graph_view.select_node("input")
    assert not widget.isolated_tuning_panel.isHidden()
    assert not widget.isolated_tuning_checkbox.isChecked()
    assert not widget.isolated_tuning_checkbox.isEnabled()
    widget.graph_view.select_node("gaussian")
    assert widget.isolated_tuning_checkbox.isChecked()
    widget._on_param_changed("sigma", 0.0)
    widget._debounce_timer.stop()

    assert widget._isolated_tuning_node_id == "gaussian"
    assert not widget.isolated_tuning_panel.isHidden()
    assert widget.graph_view._cards["gaussian"]._isolated_tuning
    assert widget.pipeline.node_execution_states["threshold"] == EXECUTION_BLOCKED
    assert (
        BLOCKED_EXECUTION_ACCENT in widget.graph_view._cards["threshold"].styleSheet()
    )
    assert "propagation is paused" in widget.graph_view._cards["threshold"].toolTip()
    assert cached_threshold is widget.pipeline.outputs["threshold"]
    assert {"gaussian", "threshold"} <= widget._cache_retention_node_ids(
        CACHE_MODE_LOW_MEMORY
    )

    widget.run_pipeline(force_sync=True)

    assert calls == ["gaussian"]
    assert widget.pipeline.outputs["threshold"] is cached_threshold
    assert widget.pipeline.node_execution_states["gaussian"] == EXECUTION_READY
    assert widget.pipeline.node_execution_states["threshold"] == EXECUTION_BLOCKED

    widget.apply_isolated_tuning_button.click()

    assert calls == ["gaussian", "threshold"]
    assert widget._isolated_tuning_node_id is None
    assert widget.isolated_tuning_panel.isHidden()
    assert not widget.isolated_tuning_checkbox.isChecked()
    assert widget.pipeline.node_execution_states["threshold"] == EXECUTION_READY


def test_cancel_isolated_tuning_restores_parameters_and_cached_results(qtbot):
    image = np.arange(100, dtype=np.float32).reshape(10, 10)
    widget = VippWidget(_Viewer(image, metadata={"axes": "YX"}))
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    qtbot.addWidget(widget)
    widget.run_pipeline(force_sync=True)
    qtbot.waitUntil(
        lambda: widget._active_thumbnail_contrast_run_id is None,
        timeout=5_000,
    )
    widget.graph_view.select_node("gaussian")
    original_sigma = widget.pipeline.nodes["gaussian"].params["sigma"]
    original_gaussian = widget.pipeline.outputs["gaussian"]
    original_threshold = widget.pipeline.outputs["threshold"]
    original_compute_decision = widget._accepted_compute_decisions["gaussian"]
    original_compute_stale = set(widget._stale_compute_badge_node_ids)
    original_execution_report = widget._last_execution_report

    widget.isolated_tuning_checkbox.setChecked(True)
    widget._on_param_changed("sigma", 0.0)
    widget._debounce_timer.stop()
    widget.run_pipeline(force_sync=True)
    assert widget.pipeline.outputs["gaussian"] is not original_gaussian
    assert (
        widget._accepted_compute_decisions["gaussian"] is not original_compute_decision
    )

    widget.cancel_isolated_tuning_button.click()

    assert widget._isolated_tuning_node_id is None
    assert widget.pipeline.nodes["gaussian"].params["sigma"] == original_sigma
    assert widget.pipeline.outputs["gaussian"] is original_gaussian
    assert widget.pipeline.outputs["threshold"] is original_threshold
    assert widget.pipeline.node_execution_states["gaussian"] == EXECUTION_READY
    assert widget.pipeline.node_execution_states["threshold"] == EXECUTION_READY
    assert widget._accepted_compute_decisions["gaussian"] is original_compute_decision
    assert widget._stale_compute_badge_node_ids == original_compute_stale
    assert widget._last_execution_report is original_execution_report
    assert "updating" not in widget.compute_status_label.text()


def test_isolated_tuning_debounce_keeps_session_open(qtbot):
    image = np.arange(100, dtype=np.float32).reshape(10, 10)
    widget = VippWidget(_Viewer(image, metadata={"axes": "YX"}))
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    qtbot.addWidget(widget)
    widget.run_pipeline(force_sync=True)
    widget.graph_view.select_node("gaussian")
    widget.isolated_tuning_checkbox.setChecked(True)

    widget._on_param_changed("sigma", 0.0)
    qtbot.waitUntil(
        lambda: not widget._debounce_timer.isActive(),
        timeout=1000,
    )

    assert widget._isolated_tuning_node_id == "gaussian"
    assert widget.pipeline.node_execution_states["gaussian"] == EXECUTION_READY
    assert widget.pipeline.node_execution_states["threshold"] == EXECUTION_BLOCKED


def test_isolated_tuning_preserves_napari_camera_slice_and_inspection_layer(qtbot):
    image = np.arange(2 * 3 * 8 * 9, dtype=np.float32).reshape(2, 3, 8, 9)
    viewer = ViewerModel()
    viewer.add_image(
        image,
        name="input volume",
        metadata={"axes": "TZYX"},
    )
    widget = VippWidget(viewer)
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    qtbot.addWidget(widget)
    widget.run_pipeline(force_sync=True)
    widget.graph_view.select_node("gaussian")
    widget.isolated_tuning_checkbox.setChecked(True)

    viewer.dims.ndisplay = 3
    viewer.dims.set_current_step(0, 1)
    viewer.camera.center = (1.5, 4.0, 4.5)
    viewer.camera.zoom = 4.25
    viewer.camera.angles = (18.0, 27.0, 41.0)
    viewer.camera.perspective = 12.0
    camera_before = (
        viewer.camera.center,
        viewer.camera.zoom,
        viewer.camera.angles,
        viewer.camera.perspective,
    )
    step_before = tuple(viewer.dims.current_step)
    inspect_layer = viewer.layers[widget._inspect_layer_name]

    widget._on_param_changed("sigma", 0.0)
    widget._debounce_timer.stop()
    widget.run_pipeline(force_sync=True)

    assert (
        viewer.camera.center,
        viewer.camera.zoom,
        viewer.camera.angles,
        viewer.camera.perspective,
    ) == camera_before
    assert tuple(viewer.dims.current_step) == step_before
    assert viewer.layers[widget._inspect_layer_name] is inspect_layer


def test_isolated_tuning_recalculates_a_manual_root_without_auto_mode(
    qtbot,
    monkeypatch,
):
    image = np.zeros((9, 9), dtype=np.float32)
    image[1:4, 1:4] = 10
    widget = VippWidget(_Viewer(image, metadata={"axes": "YX"}))
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    widget._compute_mode = ComputeMode.CPU
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
    widget.graph_view.select_node(measurements.id)
    calls: list[str] = []
    original_run_node = widget.pipeline._run_node

    def counted_run_node(node_id, *args, **kwargs):
        calls.append(node_id)
        return original_run_node(node_id, *args, **kwargs)

    monkeypatch.setattr(widget.pipeline, "_run_node", counted_run_node)
    widget.isolated_tuning_checkbox.setChecked(True)
    current = bool(
        widget.pipeline.nodes[measurements.id].params["include_shape_descriptors"]
    )
    widget._on_param_changed("include_shape_descriptors", not current)
    widget._debounce_timer.stop()
    widget.run_pipeline(force_sync=True)

    assert calls == [measurements.id]
    assert not widget.pipeline.node_auto_recalculate(measurements.id)
    assert widget.pipeline.node_execution_states[measurements.id] == EXECUTION_READY
    assert widget.pipeline.node_execution_states[selected.id] == EXECUTION_BLOCKED

    widget.apply_isolated_tuning_button.click()

    assert calls == [measurements.id, selected.id]
    assert widget.pipeline.node_execution_states[selected.id] == EXECUTION_READY


def test_calculate_all_releases_isolation_in_all_automatic_graph(
    qtbot,
    monkeypatch,
):
    image = np.arange(100, dtype=np.float32).reshape(10, 10)
    widget = VippWidget(_Viewer(image, metadata={"axes": "YX"}))
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    widget._compute_mode = ComputeMode.CPU
    qtbot.addWidget(widget)
    widget.run_pipeline(force_sync=True)
    widget.graph_view.select_node("gaussian")
    calls: list[str] = []
    original_run_node = widget.pipeline._run_node

    def counted_run_node(node_id, *args, **kwargs):
        calls.append(node_id)
        return original_run_node(node_id, *args, **kwargs)

    monkeypatch.setattr(widget.pipeline, "_run_node", counted_run_node)
    widget.isolated_tuning_checkbox.setChecked(True)
    widget._on_param_changed("sigma", 0.0)
    widget._debounce_timer.stop()
    widget.run_pipeline(force_sync=True)
    assert calls == ["gaussian"]
    assert widget._manual_node_ids_needing_calculation() == set()

    widget.calculate_all_button.click()

    assert calls == ["gaussian", "threshold"]
    assert widget._isolated_tuning_node_id is None
    assert widget.pipeline.node_execution_states["threshold"] == EXECUTION_READY


def test_apply_before_local_result_is_ready_recalculates_the_tuned_node(
    qtbot,
    monkeypatch,
):
    image = np.arange(100, dtype=np.float32).reshape(10, 10)
    widget = VippWidget(_Viewer(image, metadata={"axes": "YX"}))
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    widget._compute_mode = ComputeMode.CPU
    qtbot.addWidget(widget)
    widget.run_pipeline(force_sync=True)
    widget.graph_view.select_node("gaussian")
    calls: list[str] = []
    original_run_node = widget.pipeline._run_node

    def counted_run_node(node_id, *args, **kwargs):
        calls.append(node_id)
        return original_run_node(node_id, *args, **kwargs)

    monkeypatch.setattr(widget.pipeline, "_run_node", counted_run_node)
    widget.isolated_tuning_checkbox.setChecked(True)
    widget._on_param_changed("sigma", 0.0)
    assert widget._debounce_timer.isActive()

    widget.apply_isolated_tuning_button.click()
    qtbot.wait(widget._debounce_timer.interval() + 50)

    assert calls == ["gaussian", "threshold"]
    assert not widget._debounce_timer.isActive()
    assert widget._isolated_tuning_node_id is None
    assert widget.pipeline.node_execution_states["gaussian"] == EXECUTION_READY
    assert widget.pipeline.node_execution_states["threshold"] == EXECUTION_READY


@pytest.mark.parametrize(
    "edit_kind",
    ["add", "duplicate", "delete", "disconnect", "note"],
)
def test_history_edit_commits_isolated_tuning_before_mutation(qtbot, edit_kind):
    image = np.arange(100, dtype=np.float32).reshape(10, 10)
    widget = VippWidget(_Viewer(image, metadata={"axes": "YX"}))
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    qtbot.addWidget(widget)
    widget.run_pipeline(force_sync=True)
    widget.graph_view.select_node("gaussian")
    widget.isolated_tuning_checkbox.setChecked(True)
    widget._on_param_changed("sigma", 0.0)
    assert widget._debounce_timer.isActive()

    if edit_kind == "add":
        widget.add_node_from_palette("invert")
    elif edit_kind == "duplicate":
        widget._duplicate_node("threshold")
    elif edit_kind == "delete":
        widget._delete_node("threshold")
    elif edit_kind == "disconnect":
        widget._disconnect_nodes("gaussian", "threshold")
    else:
        widget._add_graph_note("Review tuned result")

    qtbot.waitUntil(
        lambda: not (widget._pending_dirty_node_ids & set(widget.pipeline.nodes)),
        timeout=1000,
    )
    assert widget._isolated_tuning_node_id is None
    assert widget._isolated_tuning_snapshot is None
    assert not widget._debounce_timer.isActive()
    assert set(widget.pipeline.node_execution_states) == set(widget.pipeline.nodes)
    assert set(widget.pipeline.node_execution_messages) == set(widget.pipeline.nodes)
    assert not widget._cancel_isolated_tuning(announce=False)


def test_editing_other_node_setting_commits_isolated_tuning(qtbot):
    image = np.arange(100, dtype=np.float32).reshape(10, 10)
    widget = VippWidget(_Viewer(image, metadata={"axes": "YX"}))
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    qtbot.addWidget(widget)
    widget.run_pipeline(force_sync=True)
    widget.graph_view.select_node("gaussian")
    widget.isolated_tuning_checkbox.setChecked(True)
    widget._on_param_changed("sigma", 0.0)
    widget._debounce_timer.stop()
    widget.graph_view.select_node("threshold")

    widget.keep_cached_checkbox.setChecked(True)
    qtbot.waitUntil(
        lambda: not (widget._pending_dirty_node_ids & set(widget.pipeline.nodes)),
        timeout=1000,
    )

    assert widget._isolated_tuning_node_id is None
    assert widget.pipeline.nodes["threshold"].params[CACHE_KEEP_NODE_PARAM]
    assert not widget._cancel_isolated_tuning(announce=False)


def test_editing_other_node_parameter_after_isolation_runs_branch_once(
    qtbot,
    monkeypatch,
):
    image = np.arange(100, dtype=np.float32).reshape(10, 10)
    widget = VippWidget(_Viewer(image, metadata={"axes": "YX"}))
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    widget._compute_mode = ComputeMode.CPU
    qtbot.addWidget(widget)
    widget.run_pipeline(force_sync=True)
    calls: list[str] = []
    original_run_node = widget.pipeline._run_node

    def counted_run_node(node_id, *args, **kwargs):
        calls.append(node_id)
        return original_run_node(node_id, *args, **kwargs)

    monkeypatch.setattr(widget.pipeline, "_run_node", counted_run_node)
    widget.graph_view.select_node("gaussian")
    widget.isolated_tuning_checkbox.setChecked(True)
    widget._on_param_changed("sigma", 0.0)
    widget.graph_view.select_node("threshold")
    widget._on_param_changed("histogram_bins", 128)

    qtbot.wait(widget._debounce_timer.interval() + 100)

    assert widget._isolated_tuning_node_id is None
    assert calls == ["gaussian", "threshold"]
    assert not widget._pending_dirty_node_ids


def test_stale_deconvolution_holds_automatic_descendants_in_widget(qtbot, monkeypatch):
    data = np.zeros((9, 9), dtype=np.float32)
    data[2:7, 2:7] = 0.1
    data[4, 4] = 1.0
    widget = VippWidget(_Viewer(data, metadata={"axes": "YX"}))
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    widget._compute_mode = ComputeMode.CPU
    qtbot.addWidget(widget)
    psf = widget.add_node_from_palette("gaussian_blur")
    deconvolution = widget.add_node_from_palette("richardson_lucy_deconvolution")
    rescale = widget.add_node_from_palette("rescale_intensity")
    otsu = widget.add_node_from_palette("otsu_threshold")
    widget.pipeline.set_param(deconvolution.id, "spatial_mode", "2D YX")
    widget.pipeline.set_param(deconvolution.id, "iterations", 1)
    widget._connect_nodes("input", psf.id)
    widget._connect_nodes("input", deconvolution.id, target_port=0)
    widget._connect_nodes(psf.id, deconvolution.id, target_port=1)
    widget._connect_nodes(deconvolution.id, rescale.id)
    widget._connect_nodes(rescale.id, otsu.id)
    widget.run_pipeline(force_sync=True, manual_node_ids={deconvolution.id})
    cached_rescale = widget.pipeline.outputs[rescale.id]
    cached_otsu = widget.pipeline.outputs[otsu.id]
    calls = []
    original_run_node = widget.pipeline._run_node

    def counted_run_node(node_id, *args, **kwargs):
        calls.append(node_id)
        return original_run_node(node_id, *args, **kwargs)

    monkeypatch.setattr(widget.pipeline, "_run_node", counted_run_node)
    widget.pipeline.set_param(psf.id, "sigma", 2.0)
    widget._mark_pipeline_dirty(psf.id)

    deconvolution_card = widget.graph_view._cards[deconvolution.id]
    rescale_card = widget.graph_view._cards[rescale.id]
    otsu_card = widget.graph_view._cards[otsu.id]
    assert STALE_EXECUTION_ACCENT in deconvolution_card.styleSheet()
    assert BLOCKED_EXECUTION_ACCENT in rescale_card.styleSheet()
    assert BLOCKED_EXECUTION_ACCENT in otsu_card.styleSheet()
    assert (
        QColor(BLOCKED_EXECUTION_ACCENT).lightness()
        < QColor(STALE_EXECUTION_ACCENT).lightness()
    )
    assert "upstream manual node" in rescale_card.toolTip()
    widget.background_all_checkbox.setChecked(True)
    assert widget._background_processing_node_id({psf.id}) == psf.id
    assert widget._background_processing_node_id({deconvolution.id}) is None
    low_memory_retained = widget._cache_retention_node_ids(CACHE_MODE_LOW_MEMORY)
    assert {deconvolution.id, rescale.id, otsu.id} <= low_memory_retained

    widget.run_pipeline(force_sync=True)

    assert calls == [psf.id]
    assert widget.pipeline.node_execution_states[deconvolution.id] == EXECUTION_STALE
    assert widget.pipeline.node_execution_states[rescale.id] == EXECUTION_BLOCKED
    assert widget.pipeline.node_execution_states[otsu.id] == EXECUTION_BLOCKED
    assert widget.pipeline.outputs[rescale.id] is cached_rescale
    assert widget.pipeline.outputs[otsu.id] is cached_otsu
    assert not widget.graph_view._cards[rescale.id].is_processing()
    assert not widget.graph_view._cards[otsu.id].is_processing()

    calls.clear()
    widget.run_pipeline(force_sync=True, manual_node_ids={deconvolution.id})

    assert calls == [deconvolution.id, rescale.id, otsu.id]
    assert widget.pipeline.node_execution_states[deconvolution.id] == EXECUTION_READY


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
    assert widget._manual_node_ids_requiring_attention() == {
        measurements.id,
        intensity.id,
    }
    assert widget.calculate_all_button.property("attentionRequired") is True
    assert STALE_EXECUTION_ACCENT in widget.calculate_all_button.styleSheet()
    for node_id in (measurements.id, intensity.id):
        card = widget.graph_view._cards[node_id]
        assert STALE_EXECUTION_ACCENT in card.styleSheet()
        assert "Calculate" in card.toolTip()

    widget.calculate_all_button.click()

    assert widget.pipeline.node_execution_states[measurements.id] == EXECUTION_READY
    assert widget.pipeline.node_execution_states[intensity.id] == EXECUTION_READY
    assert widget.pipeline.outputs[measurements.id].row_count == 2
    assert widget.pipeline.outputs[intensity.id].row_count == 2
    assert widget._manual_node_ids_needing_calculation() == set()
    assert widget._manual_node_ids_requiring_attention() == set()
    assert widget.calculate_all_button.property("attentionRequired") is False
    assert widget.calculate_all_button.styleSheet() == ""

    widget.pipeline.set_param(threshold.id, "threshold", 20)
    widget._mark_pipeline_dirty(threshold.id)
    widget.run_pipeline(force_sync=True)

    assert widget.pipeline.node_execution_states[measurements.id] == EXECUTION_STALE
    assert widget.pipeline.node_execution_states[intensity.id] == EXECUTION_STALE
    assert widget._manual_node_ids_needing_calculation() == {
        measurements.id,
        intensity.id,
    }
    assert widget._manual_node_ids_requiring_attention() == {
        measurements.id,
        intensity.id,
    }
    assert widget.calculate_all_button.property("attentionRequired") is True
    assert STALE_EXECUTION_ACCENT in widget.calculate_all_button.styleSheet()

    widget.calculate_all_button.click()

    assert widget.pipeline.node_execution_states[measurements.id] == EXECUTION_READY
    assert widget.pipeline.node_execution_states[intensity.id] == EXECUTION_READY
    assert widget.pipeline.outputs[measurements.id].row_count == 0
    assert widget.pipeline.outputs[intensity.id].row_count == 0
    assert widget.calculate_all_button.property("attentionRequired") is False


def test_calculate_all_attention_tracks_actionable_manual_frontiers(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    manual = widget.add_node_from_palette("measure_objects")
    card = widget.graph_view._cards[manual.id]

    widget.pipeline.set_node_auto_recalculate(manual.id, True)
    widget.pipeline.node_execution_states[manual.id] = EXECUTION_NOT_CALCULATED
    widget.pipeline.node_execution_messages[manual.id] = ""
    widget._sync_execution_ui()

    assert STALE_EXECUTION_ACCENT in card.styleSheet()
    assert "auto recalculation is pending" in card.toolTip()
    assert widget.calculate_all_button.property("attentionRequired") is False

    widget.pipeline.set_node_auto_recalculate(manual.id, False)
    widget.pipeline.node_execution_states[manual.id] = EXECUTION_BLOCKED
    widget._sync_execution_ui()

    assert BLOCKED_EXECUTION_ACCENT in card.styleSheet()
    assert not card.calculate_button.isEnabled()
    assert widget.calculate_all_button.property("attentionRequired") is False

    widget.pipeline.node_execution_states[manual.id] = EXECUTION_RUNNING
    widget._sync_execution_ui()

    assert widget.calculate_all_button.property("attentionRequired") is False

    widget.pipeline.node_execution_states[manual.id] = EXECUTION_STALE
    widget._sync_execution_ui()

    assert STALE_EXECUTION_ACCENT in card.styleSheet()
    assert widget.calculate_all_button.property("attentionRequired") is True

    widget.graph_view.select_node("input")
    widget._delete_node(manual.id)

    assert widget._manual_node_ids_requiring_attention() == set()
    assert widget.calculate_all_button.property("attentionRequired") is False
    assert widget.calculate_all_button.styleSheet() == ""


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


def _configure_workflow_attached_batch(widget, tmp_path):
    input_dir = tmp_path / "batch-input"
    input_dir.mkdir(exist_ok=True)
    np.save(input_dir / "field.npy", np.arange(20, dtype=np.uint16).reshape(4, 5))
    output_dir = tmp_path / "batch-output"
    dialog = widget._batch_collection_dialog()
    assert dialog is not None
    dialog.input_edit.setText(str(input_dir))
    dialog.pattern_edit.setText("*.npy")
    dialog.output_edit.setText(str(output_dir))
    dialog.format_combo.setCurrentText("npy")
    dialog.existing_policy_combo.setCurrentIndex(
        dialog.existing_policy_combo.findData(ExistingFilePolicy.SKIP.value)
    )
    dialog.continue_checkbox.setChecked(False)
    return dialog, input_dir, output_dir


def test_save_workflow_can_include_active_batch_workspace(
    qtbot,
    monkeypatch,
    tmp_path,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    dialog, input_dir, output_dir = _configure_workflow_attached_batch(
        widget,
        tmp_path,
    )
    target = tmp_path / "workflow-with-batch.json"
    monkeypatch.setattr(
        "napari_vipp._widget.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.Yes,
    )
    monkeypatch.setattr(
        "napari_vipp._widget.QFileDialog.getSaveFileName",
        lambda *_args, **_kwargs: (str(target), ""),
    )

    widget._save_workflow_dialog()

    document = json.loads(target.read_text(encoding="utf-8"))
    attached = BatchConfig.from_dict(
        document["batch_config"],
        base_dir=target.parent,
    )
    assert attached.workflow_file == Path(target.name)
    assert attached.workflow_sha256 == scientific_workflow_hash(document)
    assert attached.resolve_path(attached.output_dir) == output_dir.resolve()
    assert attached.resolve_path(attached.sources[0].input_dir) == input_dir.resolve()
    assert attached.sources[0].pattern == "*.npy"
    assert attached.default_image_format == "npy"
    assert attached.existing_file_policy == ExistingFilePolicy.SKIP
    assert attached.continue_on_error is False
    assert dialog._preview_result is None
    assert not (tmp_path / BATCH_CONFIG_FILENAME).exists()
    assert "with its Batch workspace" in widget.status_label.text()


def test_save_workflow_without_batch_workspace_does_not_prompt(
    qtbot,
    monkeypatch,
    tmp_path,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    target = tmp_path / "ordinary-workflow.json"
    monkeypatch.setattr(
        "napari_vipp._widget.QMessageBox.question",
        lambda *_args, **_kwargs: pytest.fail(
            "An ordinary workflow save must not show the batch prompt."
        ),
    )
    monkeypatch.setattr(
        "napari_vipp._widget.QFileDialog.getSaveFileName",
        lambda *_args, **_kwargs: (str(target), ""),
    )

    widget._save_workflow_dialog()

    assert "batch_config" not in json.loads(target.read_text(encoding="utf-8"))


def test_workflow_save_directory_is_reused_by_load(qtbot, monkeypatch, tmp_path):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    selected_dir = tmp_path / "saved-workflows"
    selected_dir.mkdir()
    target = selected_dir / "remembered.json"
    starts = []

    def select_save(_parent, _title, start, _filters):
        starts.append(start)
        return str(target), ""

    monkeypatch.setattr(
        "napari_vipp._widget.QFileDialog.getSaveFileName",
        select_save,
    )
    widget._save_workflow_dialog()
    assert starts == ["vipp_workflow.json"]

    load_starts = []

    def cancel_load(_parent, _title, start, _filters):
        load_starts.append(start)
        return "", ""

    monkeypatch.setattr(
        "napari_vipp._widget.QFileDialog.getOpenFileName",
        cancel_load,
    )
    widget._load_workflow_dialog()

    assert load_starts == [str(selected_dir.resolve())]
    assert recent_paths.recent_directory(recent_paths.WORKFLOW_DIRECTORY) == str(
        selected_dir.resolve()
    )


def test_workflow_load_directory_is_reused_by_save(qtbot, monkeypatch, tmp_path):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    selected_dir = tmp_path / "loaded-workflows"
    selected_dir.mkdir()
    selected = selected_dir / "selected.json"
    load_starts = []

    def select_load(_parent, _title, start, _filters):
        load_starts.append(start)
        return str(selected), ""

    monkeypatch.setattr(
        "napari_vipp._widget.QFileDialog.getOpenFileName",
        select_load,
    )
    monkeypatch.setattr(widget, "load_workflow_file", lambda path: Path(path))
    widget._load_workflow_dialog()
    assert load_starts == [""]

    save_starts = []

    def cancel_save(_parent, _title, start, _filters):
        save_starts.append(start)
        return "", ""

    monkeypatch.setattr(
        "napari_vipp._widget.QFileDialog.getSaveFileName",
        cancel_save,
    )
    widget._save_workflow_dialog()

    assert save_starts == [str(selected_dir / "vipp_workflow.json")]
    assert recent_paths.recent_directory(recent_paths.WORKFLOW_DIRECTORY) == str(
        selected_dir.resolve()
    )


def test_closing_untouched_batch_workspace_leaves_single_image_mode(
    qtbot,
    monkeypatch,
    tmp_path,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    target = tmp_path / "ordinary-after-batch-cancel.json"

    qtbot.mouseClick(widget.batch_button, Qt.LeftButton)
    dialog = widget._active_collection_batch_dialog
    assert dialog is not None
    assert dialog.isVisible()
    assert widget.leave_batch_button.isHidden()

    qtbot.mouseClick(dialog.close_button, Qt.LeftButton)

    assert widget._active_collection_batch_dialog is None
    assert widget._interactive_collection_batch_items == ()
    assert widget._interactive_collection_source_paths == {}
    assert widget.batch_navigator.isHidden()
    assert widget.leave_batch_button.isHidden()
    monkeypatch.setattr(
        "napari_vipp._widget.QMessageBox.question",
        lambda *_args, **_kwargs: pytest.fail(
            "Cancelling an untouched Batch workspace must not affect saving."
        ),
    )
    monkeypatch.setattr(
        "napari_vipp._widget.QFileDialog.getSaveFileName",
        lambda *_args, **_kwargs: (str(target), ""),
    )

    widget._save_workflow_dialog()

    assert "batch_config" not in json.loads(target.read_text(encoding="utf-8"))


def test_leave_batch_mode_discards_configured_workspace(qtbot, tmp_path):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    dialog, _input_dir, _output_dir = _configure_workflow_attached_batch(
        widget,
        tmp_path,
    )

    assert not widget.leave_batch_button.isHidden()
    qtbot.mouseClick(dialog.close_button, Qt.LeftButton)
    assert widget._active_collection_batch_dialog is dialog
    assert dialog.isHidden()
    assert not widget.leave_batch_button.isHidden()

    qtbot.mouseClick(widget.leave_batch_button, Qt.LeftButton)

    assert widget._active_collection_batch_dialog is None
    assert widget._interactive_collection_batch_items == ()
    assert widget._interactive_collection_source_paths == {}
    assert widget.batch_navigator.isHidden()
    assert widget.leave_batch_button.isHidden()
    assert "single-image mode" in widget.status_label.text()


def test_leave_batch_mode_clears_representative_source_overrides(qtbot, tmp_path):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    demo = widget._create_collection_batch_demo(tmp_path / "demo")
    widget._batch_collection_dialog(config_path=demo.config_path)

    assert widget._interactive_collection_batch_items
    assert widget._interactive_collection_source_paths
    assert not widget.leave_batch_button.isHidden()

    qtbot.mouseClick(widget.leave_batch_button, Qt.LeftButton)

    assert widget._active_collection_batch_dialog is None
    assert widget._interactive_collection_batch_items == ()
    assert widget._interactive_collection_source_paths == {}
    assert widget._interactive_collection_batch_config is None
    assert widget.batch_navigator.isHidden()
    for node in widget.pipeline.nodes.values():
        if node.operation_id == "input":
            assert widget._file_source_path_for_node(node) is None


def test_save_workflow_can_exclude_active_batch_workspace(
    qtbot,
    monkeypatch,
    tmp_path,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    _configure_workflow_attached_batch(widget, tmp_path)
    target = tmp_path / "workflow-only.json"
    monkeypatch.setattr(
        "napari_vipp._widget.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.No,
    )
    monkeypatch.setattr(
        "napari_vipp._widget.QFileDialog.getSaveFileName",
        lambda *_args, **_kwargs: (str(target), ""),
    )

    widget._save_workflow_dialog()

    document = json.loads(target.read_text(encoding="utf-8"))
    assert "batch_config" not in document
    assert widget._active_collection_batch_dialog is not None


def test_cancel_batch_workspace_save_prompt_writes_nothing(
    qtbot,
    monkeypatch,
    tmp_path,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    _configure_workflow_attached_batch(widget, tmp_path)
    file_dialog_calls = []
    monkeypatch.setattr(
        "napari_vipp._widget.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.Cancel,
    )
    monkeypatch.setattr(
        "napari_vipp._widget.QFileDialog.getSaveFileName",
        lambda *_args, **_kwargs: file_dialog_calls.append(True),
    )

    widget._save_workflow_dialog()

    assert file_dialog_calls == []
    assert list(tmp_path.glob("*.json")) == []
    assert "cancelled" in widget.status_label.text().lower()


def test_invalid_included_batch_workspace_aborts_workflow_save(
    qtbot,
    monkeypatch,
    tmp_path,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    dialog = widget._batch_collection_dialog()
    assert dialog is not None
    target = tmp_path / "invalid-batch-workflow.json"
    monkeypatch.setattr(
        "napari_vipp._widget.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.Yes,
    )
    monkeypatch.setattr(
        "napari_vipp._widget.QFileDialog.getSaveFileName",
        lambda *_args, **_kwargs: (str(target), ""),
    )

    widget._save_workflow_dialog()

    assert not target.exists()
    assert widget.status_label.text().startswith("Save failed:")


def test_workflow_load_restores_attached_batch_without_previewing(
    qtbot,
    monkeypatch,
    tmp_path,
):
    source_widget = VippWidget(_Viewer())
    qtbot.addWidget(source_widget)
    _dialog, input_dir, output_dir = _configure_workflow_attached_batch(
        source_widget,
        tmp_path,
    )
    target = tmp_path / "restorable-batch-workflow.json"
    document = source_widget._workflow_document_with_batch_config(
        target,
        source_widget.graph_view.node_positions(),
    )
    target.write_text(json.dumps(document), encoding="utf-8")

    restored = VippWidget(_Viewer())
    qtbot.addWidget(restored)
    monkeypatch.setattr(
        restored._collection_batch_controller,
        "preview",
        lambda **_kwargs: pytest.fail(
            "Loading attached settings must not plan or preview the batch."
        ),
    )

    restored.load_workflow_file(target)

    dialog = restored._active_collection_batch_dialog
    assert dialog is not None
    assert dialog.isVisible()
    assert dialog._preview_result is None
    assert dialog._loaded_config_path is None
    assert restored._interactive_collection_batch_items == ()
    assert restored._interactive_collection_batch_config_path is None
    assert Path(dialog.input_edit.text()) == input_dir.resolve()
    assert dialog.pattern_edit.text() == "*.npy"
    assert Path(dialog.output_edit.text()) == output_dir.resolve()
    assert dialog.format_combo.currentText() == "npy"
    assert dialog.values()["existing_file_policy"] == ExistingFilePolicy.SKIP.value
    assert dialog.continue_checkbox.isChecked() is False
    assert "restored from this workflow" in dialog.preview_status.text()
    assert "settings were restored" in restored._last_workflow_load_detail


def test_invalid_attached_batch_config_does_not_block_workflow_load(
    qtbot,
    tmp_path,
):
    source = VippWidget(_Viewer())
    qtbot.addWidget(source)
    document = serialize_workflow(
        source.pipeline,
        source.graph_view.node_positions(),
    )
    document["batch_config"] = {
        "type": "unsupported-batch-config",
        "version": 999,
    }
    target = tmp_path / "workflow-with-invalid-batch.json"
    target.write_text(json.dumps(document), encoding="utf-8")
    restored = VippWidget(_Viewer())
    qtbot.addWidget(restored)

    restored.load_workflow_file(target)

    assert set(restored.pipeline.nodes) == set(source.pipeline.nodes)
    assert restored._active_collection_batch_dialog is None
    assert "could not be restored" in restored._last_workflow_load_detail


def test_attached_batch_hash_mismatch_does_not_block_workflow_load(
    qtbot,
    tmp_path,
):
    source = VippWidget(_Viewer())
    qtbot.addWidget(source)
    _configure_workflow_attached_batch(source, tmp_path)
    target = tmp_path / "workflow-with-mismatched-batch-hash.json"
    document = source._workflow_document_with_batch_config(
        target,
        source.graph_view.node_positions(),
    )
    document["batch_config"]["workflow"]["sha256"] = "0" * 64
    target.write_text(json.dumps(document), encoding="utf-8")
    restored = VippWidget(_Viewer())
    qtbot.addWidget(restored)

    restored.load_workflow_file(target)

    assert set(restored.pipeline.nodes) == set(source.pipeline.nodes)
    assert restored._active_collection_batch_dialog is None
    assert "could not be restored" in restored._last_workflow_load_detail
    assert "hash does not match" in restored._last_workflow_load_detail


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
    assert not dialog.load_config_button.isEnabled()
    assert dialog.save_config_button.text() == "Save config..."
    assert not dialog.save_config_button.isEnabled()
    assert dialog.demo_config_button.text() == "Open batch demo..."
    assert not dialog.demo_config_button.isEnabled()
    assert not dialog.preview_button.isEnabled()


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
    qtbot.waitUntil(
        lambda: all(
            widget.graph_view.node_has_thumbnail(node_id)
            for node_id in ("input", "input_2")
        ),
        timeout=30_000,
    )
    for node_id in ("input", "input_2"):
        node = widget.pipeline.nodes[node_id]
        card = widget.graph_view._cards[node_id]
        assert node.params["file_path"] == ""
        assert node.params["binding_mode"] == "collection"
        assert card.metadata_label.text() != "No output"
        assert card.preview.has_source_pixmap()
        assert not card.preview.source_pixmap().isNull()

    config = load_batch_config(demo.config_path)
    workflow = widget._batch_workflow_document()
    assert scientific_workflow_hash(workflow) == config.workflow_sha256
    plan = plan_batch(workflow, config, workflow_path=demo.workflow_path)
    assert len(plan.items) == 3
    assert plan.output_count == 9
    assert [
        tuple(path.name for path in item.source_paths.values()) for item in plan.items
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

    widget.load_workflow_file(demo.workflow_path)

    assert widget._interactive_collection_source_paths == {}
    assert widget._interactive_collection_batch_items == ()
    assert widget.batch_navigator.isHidden()
    assert widget.pipeline.outputs["input"] is None
    assert widget.pipeline.outputs["input_2"] is None


def test_collection_input_subtitles_track_active_bindings_and_representatives(
    qtbot,
    tmp_path,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    demo = widget._create_collection_batch_demo(tmp_path / "demo")
    config = load_batch_config(demo.config_path)

    for source in config.sources:
        directory = config.resolve_path(source.input_dir)
        label = widget.graph_view._cards[source.node_id].subtitle_label
        assert label._full_text == (f"Collection · {directory.name} · {source.pattern}")
        assert label.toolTip() == (
            f"Collection source: {directory}\nPattern: {source.pattern}"
        )

    assert widget._preview_interactive_collection_batch_item(1, force_sync=True)
    representatives = dict(widget._interactive_collection_source_paths)
    for source in config.sources:
        directory = config.resolve_path(source.input_dir)
        label = widget.graph_view._cards[source.node_id].subtitle_label
        assert label._full_text == (f"Collection · {directory.name} · {source.pattern}")

    widget._clear_interactive_collection_batch_session(close_workspace=False)
    for source in config.sources:
        assert widget.graph_view._cards[source.node_id].subtitle_label._full_text == ""

    widget._interactive_collection_source_paths = representatives
    widget._sync_input_node_subtitles(representatives)
    for node_id, path in representatives.items():
        label = widget.graph_view._cards[node_id].subtitle_label
        assert label._full_text == f"Collection · {path.name}"
        assert label.toolTip() == f"Representative collection item: {path}"


def test_collection_batch_navigator_switches_the_complete_representative_pair(
    qtbot,
    tmp_path,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    demo = widget._create_collection_batch_demo(tmp_path / "demo")
    config = load_batch_config(demo.config_path)

    assert widget.batch_navigator.item_count == 3
    assert widget.batch_navigator.current_index == 0
    assert widget.batch_navigator.item_label.text() == "Item 1 of 3"
    assert "01_shifted.npy" in widget.batch_navigator.sources_label.text()
    assert "alpha_reference.npy" in widget.batch_navigator.sources_label.text()

    widget.batch_navigator.next_button.click()
    primary = np.load(demo.root / "inputs" / "primary" / "02_two_objects.npy")
    reference = np.load(demo.root / "inputs" / "reference" / "beta_reference.npy")
    qtbot.waitUntil(
        lambda: (
            np.array_equal(widget.pipeline.outputs.get("input"), primary)
            and np.array_equal(widget.pipeline.outputs.get("input_2"), reference)
        ),
        timeout=5000,
    )

    assert widget.batch_navigator.current_index == 1
    assert widget.batch_navigator.item_label.text() == "Item 2 of 3"
    assert {
        node_id: path.name
        for node_id, path in widget._interactive_collection_source_paths.items()
    } == {
        "input": "02_two_objects.npy",
        "input_2": "beta_reference.npy",
    }
    np.testing.assert_array_equal(
        widget.pipeline.outputs["batch_output_1"],
        primary.astype(np.float32) + reference.astype(np.float32),
    )
    for node_id in ("input", "input_2"):
        assert widget.pipeline.nodes[node_id].params["file_path"] == ""
        card_text = widget.graph_view._cards[node_id].metadata_label.text()
        assert "Batch 2/3" in card_text
    assert "02_two_objects.npy" in (
        widget._source_summary(
            widget._source_inspection_for_node(widget.pipeline.nodes["input"]),
            widget.pipeline.nodes["input"],
        )
    )
    assert scientific_workflow_hash(widget._batch_workflow_document()) == (
        config.workflow_sha256
    )


def test_collection_batch_navigator_browses_series_inside_one_container(
    qtbot,
    tmp_path,
):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    first = np.full((4, 16, 18), 11, dtype=np.uint8)
    second = np.full((4, 16, 18), 22, dtype=np.uint8)
    np.savez(input_dir / "multi_series.npz", upper=first, lower=second)

    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    original_params = dict(widget.pipeline.nodes["input"].params)

    preview = widget._preview_collection_batch(
        input_dir="",
        output_dir=tmp_path / "outputs",
        pattern="*.npz",
        image_format="npy",
        source_bindings=[
            {
                "node_id": "input",
                "title": "Batch input",
                "input_dir": str(input_dir),
                "pattern": "*.npz",
            }
        ],
    )

    assert len(preview.items) == 2
    assert [item.source_series_indices["input"] for item in preview.items] == [
        0,
        1,
    ]
    np.testing.assert_array_equal(widget.pipeline.outputs["input"], first)
    assert "multi_series.npz" in widget.batch_navigator.sources_label.text()
    assert "upper" in widget.batch_navigator.sources_label.text()

    widget.batch_navigator.next_button.click()
    qtbot.waitUntil(
        lambda: np.array_equal(widget.pipeline.outputs.get("input"), second),
        timeout=5000,
    )

    assert widget.batch_navigator.current_index == 1
    assert widget._interactive_collection_source_series_indices == {"input": 1}
    assert "lower" in widget.batch_navigator.sources_label.text()
    assert widget.pipeline.nodes["input"].params == original_params


@pytest.mark.parametrize("source_mode", ["napari layer", "file path"])
def test_batch_representative_overrides_the_interactive_source_selection(
    qtbot,
    tmp_path,
    source_mode,
):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    first = np.full((4, 16, 18), 11, dtype=np.uint8)
    second = np.full((4, 16, 18), 22, dtype=np.uint8)
    fixed = np.full((4, 16, 18), 99, dtype=np.uint8)
    np.save(input_dir / "field_a.npy", first)
    np.save(input_dir / "field_b.npy", second)
    fixed_path = tmp_path / "fixed.npy"
    np.save(fixed_path, fixed)

    widget = VippWidget(_Viewer(fixed))
    qtbot.addWidget(widget)
    node = widget.pipeline.nodes["input"]
    node.params["source_mode"] = source_mode
    node.params["layer_name"] = "input volume"
    node.params["file_path"] = str(fixed_path) if source_mode == "file path" else ""
    original_source_params = dict(node.params)

    preview = widget._preview_collection_batch(
        input_dir="",
        output_dir=tmp_path / "outputs",
        pattern="*.npy",
        image_format="npy",
        source_bindings=[
            {
                "node_id": "input",
                "title": "Batch input",
                "input_dir": str(input_dir),
                "pattern": "*.npy",
            }
        ],
    )

    np.testing.assert_array_equal(widget.pipeline.outputs["input"], first)
    widget._preview_interactive_collection_batch_item(1, force_sync=True)
    np.testing.assert_array_equal(widget.pipeline.outputs["input"], second)
    assert node.params == original_source_params
    assert scientific_workflow_hash(widget._batch_workflow_document()) == (
        preview.config.workflow_sha256
    )


def test_batch_representative_does_not_commit_dtype_autodefaults(qtbot, tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    np.save(input_dir / "field.npy", np.full((4, 16, 18), 4096, dtype=np.uint16))

    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    rescale = widget.add_node_from_palette("rescale_intensity")
    widget._connect_nodes("input", rescale.id)
    widget.pipeline.set_param(rescale.id, "out_min", 0.0)
    widget.pipeline.set_param(rescale.id, "out_max", 1.0)
    widget._rescale_auto_output_ranges[rescale.id] = (0.0, 1.0)
    widget.graph_view.select_node(rescale.id)
    original_params = dict(widget.pipeline.nodes[rescale.id].params)
    original_auto_ranges = dict(widget._rescale_auto_output_ranges)

    preview = widget._preview_collection_batch(
        input_dir="",
        output_dir=tmp_path / "outputs",
        pattern="*.npy",
        image_format="npy",
        source_bindings=[
            {
                "node_id": "input",
                "title": "Batch input",
                "input_dir": str(input_dir),
                "pattern": "*.npy",
            }
        ],
    )

    assert widget.pipeline.nodes[rescale.id].params == original_params
    assert widget._rescale_auto_output_ranges == original_auto_ranges
    assert scientific_workflow_hash(widget._batch_workflow_document()) == (
        preview.config.workflow_sha256
    )


def test_batch_slider_bounds_materialized_source_cache_but_pins_identities(
    qtbot,
    tmp_path,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    demo = widget._create_collection_batch_demo(tmp_path / "demo")

    for index in (1, 2):
        assert widget._preview_interactive_collection_batch_item(
            index,
            force_sync=True,
        )
        active_paths = {
            str(path) for path in widget._interactive_collection_source_paths.values()
        }
        assert {str(key[0]) for key in widget._file_source_payload_cache} == (
            active_paths
        )

    visited_paths = {
        str(path.resolve())
        for folder in (
            demo.root / "inputs" / "primary",
            demo.root / "inputs" / "reference",
        )
        for path in folder.glob("*.npy")
    }
    assert visited_paths <= set(widget._file_source_path_identities)
    assert len(widget._file_source_payload_cache) == 2


def test_missing_batch_representative_is_not_committed_as_displayed(qtbot, tmp_path):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    demo = widget._create_collection_batch_demo(tmp_path / "demo")
    first_output = np.array(widget.pipeline.outputs["batch_output_1"], copy=True)
    (demo.root / "inputs" / "primary" / "02_two_objects.npy").unlink()

    widget.batch_navigator.slider.setValue(1)

    assert widget._interactive_collection_batch_index == 0
    assert widget.batch_navigator.current_index == 0
    assert "preview failed" in widget.batch_navigator.representative_label.text()
    np.testing.assert_array_equal(
        widget.pipeline.outputs["batch_output_1"],
        first_output,
    )


def test_async_batch_navigation_commits_only_the_latest_requested_item(
    qtbot,
    monkeypatch,
    tmp_path,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    demo = widget._create_collection_batch_demo(tmp_path / "demo")
    widget._batch_collection_dialog(config_path=demo.config_path)
    dialog = widget._active_collection_batch_dialog
    assert dialog is not None
    monkeypatch.setattr(
        widget,
        "_file_source_should_load_async",
        lambda node: widget._file_source_path_for_node(node) is not None,
    )

    widget.batch_navigator.slider.setValue(1)
    widget.batch_navigator.slider.setValue(2)

    assert widget._interactive_collection_batch_index == 0
    assert widget._interactive_collection_batch_requested_index == 2
    assert not dialog.run_button.isEnabled()
    assert "Loading and calculating" in (
        widget.batch_navigator.representative_label.text()
    )
    qtbot.waitUntil(
        lambda: widget._interactive_collection_batch_index == 2,
        timeout=5_000,
    )

    primary = np.load(demo.root / "inputs" / "primary" / "03_disjoint.npy")
    reference = np.load(demo.root / "inputs" / "reference" / "gamma_reference.npy")
    assert widget._interactive_collection_batch_requested_index == -1
    assert widget.batch_navigator.current_index == 2
    assert dialog.run_button.isEnabled()
    np.testing.assert_array_equal(
        widget.pipeline.outputs["batch_output_1"],
        primary.astype(np.float32) + reference.astype(np.float32),
    )


def test_partial_representative_failure_keeps_failed_item_selected(
    qtbot,
    monkeypatch,
    tmp_path,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    demo = widget._create_collection_batch_demo(tmp_path / "demo")
    widget._batch_collection_dialog(config_path=demo.config_path)
    dialog = widget._active_collection_batch_dialog
    assert dialog is not None
    failed_primary = np.load(demo.root / "inputs" / "primary" / "02_two_objects.npy")
    calls = 0

    def fail_after_source(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        widget.pipeline.outputs["input"] = failed_primary
        widget.pipeline.outputs["batch_output_1"] = None
        raise RuntimeError("downstream representative failure")

    monkeypatch.setattr(widget.pipeline, "run", fail_after_source)

    assert widget._preview_interactive_collection_batch_item(1, force_sync=True)

    assert widget._interactive_collection_batch_index == 1
    assert widget._interactive_collection_batch_failed_index == 1
    assert widget.batch_navigator.current_index == 1
    assert "02_two_objects.npy" in widget.batch_navigator.sources_label.text()
    assert "preview failed" in widget.batch_navigator.representative_label.text()
    assert not dialog.run_button.isEnabled()
    np.testing.assert_array_equal(widget.pipeline.outputs["input"], failed_primary)
    assert widget.pipeline.outputs["batch_output_1"] is None

    assert widget._preview_interactive_collection_batch_item(1, force_sync=True)
    assert calls == 2


def test_run_stops_when_reviewed_source_changes_in_place(qtbot, tmp_path):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    demo = widget._create_collection_batch_demo(tmp_path / "demo")
    widget._batch_collection_dialog(config_path=demo.config_path)
    dialog = widget._active_collection_batch_dialog
    assert dialog is not None
    reviewed = np.array(widget.pipeline.outputs["input"], copy=True)
    source_path = demo.root / "inputs" / "primary" / "01_shifted.npy"
    np.save(source_path, np.full(reviewed.shape, 65535, dtype=np.uint16))

    qtbot.mouseClick(dialog.run_button, Qt.LeftButton)

    assert dialog._preview_result is None
    assert "Press Refresh" in dialog.preview_status.text()
    assert not (demo.root / "results" / BATCH_MANIFEST_FILENAME).exists()
    np.testing.assert_array_equal(widget.pipeline.outputs["input"], reviewed)
    assert "pinned earlier revision" in (
        widget.batch_navigator.representative_label.text()
    )


def test_loaded_batch_config_runs_on_first_click_without_graph_preview(
    qtbot,
    monkeypatch,
    tmp_path,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    demo = widget._create_collection_batch_demo(tmp_path / "demo")
    widget._clear_interactive_collection_batch_session(close_workspace=False)
    widget._batch_collection_dialog()
    dialog = widget._active_collection_batch_dialog
    assert dialog is not None
    monkeypatch.setattr(
        "napari_vipp.ui.batch.QFileDialog.getOpenFileName",
        lambda *_args, **_kwargs: (str(demo.config_path), ""),
    )

    qtbot.mouseClick(dialog.load_config_button, Qt.LeftButton)

    assert dialog._preview_result is None
    assert "Loaded" in dialog.preview_status.text()
    plan_calls = []
    expected_plans = []
    original_preview = widget._collection_batch_controller.preview
    original_prepare = widget._prepare_collection_batch_run

    def tracked_preview(**kwargs):
        result = original_preview(**kwargs)
        plan_calls.append(result)
        return result

    def tracked_prepare(**kwargs):
        expected_plans.append(kwargs.get("expected_items"))
        return original_prepare(**kwargs)

    monkeypatch.setattr(
        widget._collection_batch_controller,
        "preview",
        tracked_preview,
    )
    monkeypatch.setattr(
        widget,
        "_prepare_collection_batch_run",
        tracked_prepare,
    )
    monkeypatch.setattr(
        dialog,
        "_preview_batch",
        lambda: pytest.fail("Run must not launch the graph-preview action."),
    )
    monkeypatch.setattr(
        widget,
        "run_pipeline",
        lambda *_args, **_kwargs: pytest.fail(
            "Run must not calculate a live graph representative."
        ),
    )

    qtbot.mouseClick(dialog.run_button, Qt.LeftButton)
    qtbot.waitUntil(lambda: not widget._collection_batch_running, timeout=10_000)

    assert len(plan_calls) == 1
    assert expected_plans == [plan_calls[0].items]
    assert (demo.root / "results" / BATCH_MANIFEST_FILENAME).is_file()
    assert "3 completed" in dialog.run_progress_label.text()


def test_direct_run_applies_qyx_z_stack_suggestion_and_retries_once(
    qtbot,
    monkeypatch,
    tmp_path,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    demo = widget._create_collection_batch_demo(tmp_path / "demo")
    config = widget._load_collection_batch_config(demo.config_path)
    dialog = widget._batch_collection_dialog()
    assert dialog is not None
    assert dialog._preview_result is None
    source_rows = {str(row["node_id"]): row for row in dialog._source_rows}
    for source in config.sources:
        row = source_rows[source.node_id]
        row["folder"].setText(str(config.resolve_path(source.input_dir)))
        row["pattern"].setText(source.pattern)
        assert row["axis_declaration"].mode_combo.currentData() == "automatic"
    dialog.output_edit.setText(str(config.resolve_path(config.output_dir)))
    dialog.format_combo.setCurrentText(config.default_image_format)
    policy_index = dialog.existing_policy_combo.findData(
        config.existing_file_policy.value
    )
    assert policy_index >= 0
    dialog.existing_policy_combo.setCurrentIndex(policy_index)

    initial_values = dialog.values()
    original_preview = widget._collection_batch_controller.preview
    successful_preview = original_preview(**initial_values, preview_limit=25)
    source_id = successful_preview.config.sources[0].node_id
    declaration = AxisDeclaration("QYX", "ZYX")
    successful_preview = replace(
        successful_preview,
        config=replace(
            successful_preview.config,
            sources=tuple(
                replace(source, axis_declaration=declaration)
                if source.node_id == source_id
                else source
                for source in successful_preview.config.sources
            ),
        ),
    )
    error = BatchScientificPreflightError(
        "Batch scientific preflight failed before item processing, output "
        "creation, or CPU/GPU device setup. Representative source axes: "
        "raw QYX, effective QYX. Subtract Background, Gaussian Blur 3D, and "
        "Reorder Axes cannot continue.",
        user_message=(
            "This TIFF looks like a Z stack, but its first dimension is "
            "labelled Q. VIPP can treat it as Z for this batch."
        ),
        axis_suggestion=BatchAxisSuggestion(
            source_node_id=source_id,
            source_title=successful_preview.config.sources[0].title,
            declaration=declaration,
        ),
    )
    preview_calls: list[dict[str, object]] = []

    def suggested_preview(**values):
        preview_calls.append(values)
        binding = next(
            item for item in values["source_bindings"] if item["node_id"] == source_id
        )
        if len(preview_calls) == 1:
            assert binding["axis_declaration"] == ""
            raise error
        assert binding["axis_declaration"] == "QYX -> ZYX"
        return successful_preview

    started: list[tuple[CollectionBatchDialog, dict[str, object]]] = []

    def record_start(active_dialog, **values):
        started.append((active_dialog, values))

    monkeypatch.setattr(
        widget._collection_batch_controller,
        "preview",
        suggested_preview,
    )
    monkeypatch.setattr(widget, "_start_collection_batch_worker", record_start)
    suggestion_applications = []
    original_apply_suggestion = dialog.apply_axis_suggestion

    def tracked_apply_suggestion(preflight_error):
        source_control = next(
            row["axis_declaration"]
            for row in dialog._source_rows
            if row["node_id"] == source_id
        )
        before = (
            source_control.mode_combo.currentData(),
            source_control.suggestion_declined,
            source_control.text(),
        )
        outcome = original_apply_suggestion(preflight_error)
        suggestion_applications.append((before, outcome))
        return outcome

    monkeypatch.setattr(
        dialog,
        "apply_axis_suggestion",
        tracked_apply_suggestion,
    )
    monkeypatch.setattr(
        dialog,
        "_preview_batch",
        lambda: pytest.fail("Direct Run must not invoke Preview batch."),
    )

    qtbot.mouseClick(dialog.run_button, Qt.LeftButton)

    assert suggestion_applications == [(("automatic", False, ""), True)]
    assert len(preview_calls) == 2
    control = next(
        row["axis_declaration"]
        for row in dialog._source_rows
        if row["node_id"] == source_id
    )
    assert control.mode_combo.currentData() == "z_stack"
    assert control.text() == "QYX -> ZYX"
    assert not control.notice_label.isHidden()
    assert "VIPP selected Z stack" in control.notice_label.text()
    assert len(started) == 1
    active_dialog, run_values = started[0]
    assert active_dialog is dialog
    run_binding = next(
        item for item in run_values["source_bindings"] if item["node_id"] == source_id
    )
    assert run_binding["axis_declaration"] == "QYX -> ZYX"
    visible_text = " ".join(
        (
            dialog.preview_status.text(),
            dialog.run_result_label.text(),
            widget.status_label.text(),
        )
    )
    assert "Representative source axes" not in visible_text
    assert "CPU/GPU device setup" not in visible_text
    assert "Subtract Background" not in visible_text
    assert "Gaussian Blur 3D" not in visible_text
    assert "Reorder Axes" not in visible_text


def test_batch_worker_nested_progress_and_safe_cancel_reach_retained_dialog(
    qtbot,
    monkeypatch,
    tmp_path,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    demo = widget._create_collection_batch_demo(tmp_path / "demo")
    widget._batch_collection_dialog(config_path=demo.config_path)
    dialog = widget._active_collection_batch_dialog
    assert dialog is not None
    started = threading.Event()
    from napari_vipp.ui import batch_workers

    original_run_batch = batch_workers.run_batch

    def wait_for_cancel(*args, **kwargs):
        kwargs["execution_progress_callback"](
            BatchExecutionProgress(
                item_index=1,
                item_total=3,
                batch_id="sample",
                node_id="gaussian-1",
                operation_id="gaussian_blur",
                current=2,
                total=5,
                message="GPU tile 2 of 5",
            )
        )
        started.set()
        assert kwargs["cancel_event"].wait(timeout=5)
        return original_run_batch(*args, **kwargs)

    monkeypatch.setattr(batch_workers, "run_batch", wait_for_cancel)

    qtbot.mouseClick(dialog.run_button, Qt.LeftButton)
    qtbot.waitUntil(started.is_set, timeout=5_000)
    qtbot.waitUntil(
        lambda: "GPU tile 2 of 5" in dialog.operation_progress_label.text(),
        timeout=5_000,
    )
    assert dialog.operation_progress_bar.maximum() == 5
    assert dialog.operation_progress_bar.value() == 2
    context = widget._active_collection_batch_job
    assert context is not None
    worker = widget._collection_batch_workers[context.job_id]

    stale_dialog = CollectionBatchDialog(widget)
    widget._cancel_collection_batch_worker(stale_dialog)
    assert not worker.cancellation_requested
    prior_text = dialog.operation_progress_label.text()
    widget._on_collection_batch_worker_operation_progress(
        CollectionBatchOperationProgress(
            job_id=context.job_id + 100,
            progress=BatchExecutionProgress(
                1,
                3,
                "stale",
                "stale-node",
                "stale-operation",
                9,
                10,
                "must be ignored",
            ),
        )
    )
    assert dialog.operation_progress_label.text() == prior_text

    qtbot.mouseClick(dialog.cancel_run_button, Qt.LeftButton)
    assert worker.cancellation_requested
    assert not dialog.cancel_run_button.isEnabled()
    qtbot.waitUntil(lambda: not widget._collection_batch_running, timeout=10_000)

    assert "Batch cancelled" in dialog.run_progress_label.text()
    assert "safe cancellation checkpoint" in dialog.operation_progress_label.text()
    assert not dialog.cancel_run_button.isVisible()
    manifest = json.loads(
        (demo.root / "results" / BATCH_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["summary"]["cancelled"] == 1
    assert manifest["compute"]["runtime_cleanup_succeeded"] is True


def test_loaded_batch_compute_request_wins_until_toolbar_changes(qtbot, tmp_path):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    demo = widget._create_collection_batch_demo(tmp_path / "demo")
    config = load_batch_config(demo.config_path)
    saved_request = ComputeRequest(
        mode=ComputeMode.AUTO,
        runtime_id="cuda-cupy",
        device_id="cuda:0",
        accelerator_memory_cap_bytes=3_000_000_000,
        accelerator_safety_reserve_bytes=300_000_000,
        allow_experimental=True,
    )
    config = replace(config, compute_request=saved_request)

    widget._batch_collection_dialog(config=config, preview_config=False)
    dialog = widget._active_collection_batch_dialog
    assert dialog is not None

    assert widget._compute_request_for_batch_dialog(dialog) == saved_request

    widget._compute_mode = ComputeMode.CUSTOM
    current = widget._current_compute_request()
    assert widget._compute_request_for_batch_dialog(dialog) == current
    assert dialog._loaded_compute_request is None


def test_edited_batch_settings_run_on_first_click_without_repreview(
    qtbot,
    monkeypatch,
    tmp_path,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    demo = widget._create_collection_batch_demo(tmp_path / "demo")
    widget._batch_collection_dialog(config_path=demo.config_path)
    dialog = widget._active_collection_batch_dialog
    assert dialog is not None
    qtbot.waitUntil(dialog.run_button.isEnabled, timeout=5_000)
    edited_output = tmp_path / "edited-results"

    dialog.output_edit.setText(str(edited_output))

    assert dialog._preview_result is None
    monkeypatch.setattr(
        dialog,
        "_preview_batch",
        lambda: pytest.fail("Run must not force an optional preview."),
    )

    qtbot.mouseClick(dialog.run_button, Qt.LeftButton)
    qtbot.waitUntil(lambda: not widget._collection_batch_running, timeout=10_000)

    assert (edited_output / BATCH_MANIFEST_FILENAME).is_file()
    assert "3 completed" in dialog.run_progress_label.text()


def test_batch_cleanup_failure_quarantines_interactive_compute(
    qtbot,
    monkeypatch,
    tmp_path,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    session = widget._workflow_tabs.current
    assert session is not None

    class Dialog:
        @staticmethod
        def finish_run(*_args, **_kwargs):
            return None

        @staticmethod
        def mark_plan_historical_after_run():
            return None

    context = SimpleNamespace(
        job_id=7,
        origin_session_id=session.session_id,
        dialog=Dialog(),
        validation_config_path=None,
        total=1,
        graph_refresh_pending=False,
    )
    widget._active_collection_batch_job = context
    widget._collection_batch_workers[7] = object()
    widget._collection_batch_running = True
    active_run_id = 91
    active_cancel_event = threading.Event()
    widget._active_pipeline_run_id = active_run_id
    widget._pipeline_cancel_events[active_run_id] = active_cancel_event
    widget._pipeline_run_pending = True
    cancelled_surfaces = []
    widget._node_benchmark_dialog = SimpleNamespace(
        running=True,
        cancel=lambda: cancelled_surfaces.append("benchmark"),
    )
    widget._pipeline_optimizer_dialog = SimpleNamespace(
        running=True,
        cancel=lambda: cancelled_surfaces.append("optimizer"),
    )
    monkeypatch.setattr(
        widget,
        "_validate_collection_batch_demo_result",
        lambda *_args, **_kwargs: "",
    )
    result = SimpleNamespace(
        manifest=SimpleNamespace(compute={"runtime_cleanup_succeeded": False}),
        manifest_path=tmp_path / "manifest.json",
        summary={
            "completed": 0,
            "partial": 0,
            "skipped": 0,
            "cancelled": 0,
            "failed": 1,
        },
        cancelled=False,
        saved_paths=(),
    )

    widget._on_collection_batch_worker_finished(
        SimpleNamespace(job_id=7, error="", result=result)
    )

    assert not widget._collection_batch_running
    assert "Restart VIPP" in widget._compute_runtime_quarantined_reason
    assert not widget.compute_mode_combo.isEnabled()
    assert "durable manifest" in widget.status_label.text()
    assert active_cancel_event.is_set()
    assert widget._active_pipeline_run_id == active_run_id
    assert widget._pipeline_user_cancel_requested_run_id == active_run_id
    assert not widget._pipeline_run_pending
    assert cancelled_surfaces == ["benchmark", "optimizer"]
    widget._interactive_collection_batch_requested_index = 0

    widget._on_background_pipeline_finished(
        PipelineRunResult(active_run_id, {}, cancelled=True)
    )

    assert widget._active_pipeline_run_id is None
    assert widget._pipeline_quarantine_cancel_requested_run_id is None
    assert "Restart VIPP" in widget.status_label.text()
    assert "Representative preview failed" in widget.status_label.text()
    assert widget.status_label.property("messageSeverity") == "error"
    assert widget.status_label.property("messageActionable") is True


def test_full_batch_locks_only_its_origin_workflow_compute_policy(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    session = widget._workflow_tabs.current
    assert session is not None
    widget._collection_batch_running = True
    widget._active_collection_batch_job = SimpleNamespace(
        origin_session_id=session.session_id,
    )

    widget._sync_compute_policy_editability()

    assert not widget.compute_mode_combo.isEnabled()
    assert "Cancel the active full batch" in widget.compute_mode_combo.toolTip()

    widget._active_collection_batch_job = SimpleNamespace(
        origin_session_id="another-workflow",
    )
    widget._sync_compute_policy_editability()

    assert widget.compute_mode_combo.isEnabled()


def test_run_stops_when_reviewed_fixed_batch_source_changes(qtbot, tmp_path):
    collection_dir = tmp_path / "collection"
    collection_dir.mkdir()
    collection = np.arange(20, dtype=np.uint16).reshape(4, 5)
    fixed = np.full((4, 5), 3, dtype=np.uint16)
    np.save(collection_dir / "field.npy", collection)
    fixed_path = tmp_path / "fixed.npy"
    np.save(fixed_path, fixed)

    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    fixed_source = widget.add_node_from_palette("input")
    add = widget.add_node_from_palette("add_images")
    output = widget.add_node_from_palette("batch_output")
    widget.pipeline.set_param(fixed_source.id, "source_mode", "file path")
    widget.pipeline.set_param(fixed_source.id, "file_path", str(fixed_path))
    widget.pipeline.set_param(output.id, "format", "npy")
    widget._connect_nodes("input", add.id)
    widget._connect_nodes(fixed_source.id, add.id)
    widget._connect_nodes(add.id, output.id)

    widget._batch_collection_dialog()
    dialog = widget._active_collection_batch_dialog
    assert dialog is not None
    source_rows = {str(row["node_id"]): row for row in dialog._source_rows}
    source_rows["input"]["folder"].setText(str(collection_dir))
    source_rows["input"]["pattern"].setText("*.npy")
    dialog.output_edit.setText(str(tmp_path / "outputs"))
    dialog.format_combo.setCurrentText("npy")
    assert dialog._preview_batch()
    qtbot.waitUntil(dialog.run_button.isEnabled, timeout=5_000)
    reviewed = np.array(widget.pipeline.outputs[output.id], copy=True)

    np.save(fixed_path, np.full((4, 5), 99, dtype=np.uint16))
    qtbot.mouseClick(dialog.run_button, Qt.LeftButton)
    qtbot.waitUntil(lambda: not widget._collection_batch_running, timeout=10_000)

    assert dialog._preview_result is None
    assert "Press Refresh" in dialog.preview_status.text()
    assert not (tmp_path / "outputs" / BATCH_MANIFEST_FILENAME).exists()
    np.testing.assert_array_equal(widget.pipeline.outputs[output.id], reviewed)


def test_source_refresh_blocks_batch_until_representative_is_recalculated(
    qtbot,
    monkeypatch,
    tmp_path,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    demo = widget._create_collection_batch_demo(tmp_path / "demo")
    widget._batch_collection_dialog(config_path=demo.config_path)
    dialog = widget._active_collection_batch_dialog
    assert dialog is not None
    previous = np.array(widget.pipeline.outputs["batch_output_1"], copy=True)
    source_path = demo.root / "inputs" / "primary" / "01_shifted.npy"
    np.save(source_path, np.full(previous.shape, 41, dtype=np.uint16))
    monkeypatch.setattr(
        widget,
        "_file_source_should_load_async",
        lambda node: widget._file_source_path_for_node(node) is not None,
    )

    qtbot.mouseClick(widget.refresh_button, Qt.LeftButton)

    assert widget._interactive_collection_batch_requested_index == 0
    assert widget._active_source_load_id is not None
    assert not dialog.run_button.isEnabled()
    widget._run_collection_batch_from_workspace(dialog, dialog.values())
    assert not (demo.root / "results" / BATCH_MANIFEST_FILENAME).exists()
    qtbot.waitUntil(
        lambda: (
            widget._interactive_collection_batch_requested_index == -1
            and dialog.run_button.isEnabled()
        ),
        timeout=5_000,
    )

    assert dialog._preview_result is None
    assert not np.array_equal(widget.pipeline.outputs["batch_output_1"], previous)
    qtbot.mouseClick(dialog.run_button, Qt.LeftButton)
    manifest_path = demo.root / "results" / BATCH_MANIFEST_FILENAME
    qtbot.waitUntil(
        lambda: (
            widget._pending_collection_batch_start is None
            and not widget._collection_batch_running
            and manifest_path.is_file()
        ),
        timeout=15_000,
    )
    assert manifest_path.is_file()
    assert "3 completed" in dialog.run_progress_label.text()


def test_batch_continues_in_origin_tab_while_new_workflow_is_edited(
    qtbot,
    monkeypatch,
    tmp_path,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    demo = widget._create_collection_batch_demo(tmp_path / "demo")
    widget._batch_collection_dialog(config_path=demo.config_path)
    dialog = widget._active_collection_batch_dialog
    assert dialog is not None
    origin = widget._workflow_tabs.current
    assert origin is not None
    started = threading.Event()
    release = threading.Event()
    from napari_vipp.ui import batch_workers

    original_run_batch = batch_workers.run_batch

    def gated_run_batch(*args, **kwargs):
        started.set()
        assert release.wait(timeout=10)
        return original_run_batch(*args, **kwargs)

    monkeypatch.setattr(batch_workers, "run_batch", gated_run_batch)

    qtbot.mouseClick(dialog.run_button, Qt.LeftButton)
    qtbot.waitUntil(started.is_set, timeout=5_000)
    qtbot.waitUntil(
        lambda: not widget._workflow_tab_switch_block_reason(),
        timeout=5_000,
    )
    widget._new_workflow()

    assert len(widget._workflow_tabs) == 2
    assert widget._workflow_tabs.current is not origin
    assert widget._active_collection_batch_dialog is None
    assert widget._collection_batch_running
    widget.pipeline.set_param("input", "file_path", "edited-in-workflow-b.tif")
    assert origin.pipeline.nodes["input"].params["file_path"] != (
        "edited-in-workflow-b.tif"
    )

    origin_index = widget._workflow_tabs.index_of(origin.session_id)
    widget.workflow_tab_bar.setCurrentIndex(origin_index)
    assert widget._workflow_tabs.current is origin
    assert not widget.batch_navigator._navigation_enabled
    assert not widget.batch_navigator.slider.isEnabled()

    other_index = next(
        index
        for index, session in enumerate(widget._workflow_tabs)
        if session is not origin
    )
    widget.workflow_tab_bar.setCurrentIndex(other_index)
    assert widget._workflow_tabs.current is not origin

    release.set()
    qtbot.waitUntil(lambda: not widget._collection_batch_running, timeout=10_000)
    assert dialog.run_progress_bar.value() == 3
    assert "3 completed" in dialog.run_progress_label.text()
    assert widget._interactive_collection_batch_items == ()
    assert set(widget.pipeline.nodes) == {"input"}
    assert widget.pipeline.nodes["input"].operation_id == "input"

    widget.workflow_tab_bar.setCurrentIndex(origin_index)
    assert widget._active_collection_batch_dialog is dialog
    assert "3 completed" in dialog.run_progress_label.text()
    assert "Batch finished: 3 completed" in widget.status_label.text()
    assert widget.batch_navigator.progress_bar.value() == 3
    assert "Batch finished: 3 completed" in (
        widget.batch_navigator.progress_label.text()
    )
    assert "_collection_batch_last_summary" not in origin.runtime_cache
    assert "_collection_batch_last_total" not in origin.runtime_cache


def test_inactive_batch_failure_is_presented_when_origin_tab_is_reactivated(
    qtbot,
    monkeypatch,
    tmp_path,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    demo = widget._create_collection_batch_demo(tmp_path / "demo")
    widget._batch_collection_dialog(config_path=demo.config_path)
    dialog = widget._active_collection_batch_dialog
    assert dialog is not None
    origin = widget._workflow_tabs.current
    assert origin is not None
    started = threading.Event()
    release = threading.Event()
    from napari_vipp.ui import batch_workers

    def failing_run_batch(*_args, **_kwargs):
        started.set()
        assert release.wait(timeout=10)
        raise RuntimeError("synthetic inactive failure")

    monkeypatch.setattr(batch_workers, "run_batch", failing_run_batch)

    qtbot.mouseClick(dialog.run_button, Qt.LeftButton)
    qtbot.waitUntil(started.is_set, timeout=5_000)
    qtbot.waitUntil(
        lambda: not widget._workflow_tab_switch_block_reason(),
        timeout=5_000,
    )
    widget._new_workflow()
    assert widget._workflow_tabs.current is not origin

    release.set()
    qtbot.waitUntil(lambda: not widget._collection_batch_running, timeout=10_000)
    assert "Batch failed" in dialog.run_progress_label.text()
    assert "synthetic inactive failure" in dialog.run_result_label.text()

    origin_index = widget._workflow_tabs.index_of(origin.session_id)
    widget.workflow_tab_bar.setCurrentIndex(origin_index)

    assert widget._active_collection_batch_dialog is dialog
    assert "Batch failed: synthetic inactive failure" in widget.status_label.text()
    assert widget.batch_navigator.progress_bar.format().startswith("Failed")
    assert "synthetic inactive failure" in (
        widget.batch_navigator.progress_label.text()
    )
    assert "_collection_batch_last_error" not in origin.runtime_cache
    assert "_collection_batch_last_total" not in origin.runtime_cache


def test_scientific_graph_edit_invalidates_plan_but_keeps_slider_useful(
    qtbot,
    tmp_path,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    demo = widget._create_collection_batch_demo(tmp_path / "demo")
    widget._batch_collection_dialog(config_path=demo.config_path)
    dialog = widget._active_collection_batch_dialog
    assert dialog is not None

    output = widget.pipeline.nodes["batch_output_1"]
    widget.pipeline.set_param(output.id, "tag", "edited-result")
    widget._mark_pipeline_dirty(output.id)

    assert dialog._preview_result is None
    assert widget._interactive_collection_batch_workflow_stale
    assert widget.batch_navigator.slider.isEnabled()
    assert "scientific workflow changed" in (
        widget.batch_navigator.representative_label.text().lower()
    )
    assert dialog.run_button.text() == "Run batch"

    assert widget._preview_interactive_collection_batch_item(1, force_sync=True)
    assert widget.batch_navigator.current_index == 1
    assert dialog._preview_batch()
    assert dialog._preview_result is not None
    assert not widget._interactive_collection_batch_plan_stale
    assert dialog._preview_result.config.workflow_sha256 == (
        scientific_workflow_hash(widget._batch_workflow_document())
    )


@pytest.mark.parametrize("change", ["add", "duplicate"])
def test_image_source_topology_change_rebuilds_batch_workspace(
    qtbot,
    tmp_path,
    change,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    demo = widget._create_collection_batch_demo(tmp_path / "demo")
    widget._batch_collection_dialog(config_path=demo.config_path)
    assert widget._active_collection_batch_dialog is not None

    if change == "add":
        widget.add_node_from_palette("input")
    else:
        widget._duplicate_node("input")

    assert widget._active_collection_batch_dialog is None
    assert widget._interactive_collection_batch_items == ()
    qtbot.mouseClick(widget.batch_button, Qt.LeftButton)
    rebuilt = widget._active_collection_batch_dialog
    assert rebuilt is not None
    graph_source_ids = {
        node_id
        for node_id, node in widget.pipeline.nodes.items()
        if node.operation_id == "input"
    }
    assert {str(row["node_id"]) for row in rebuilt._source_rows} == graph_source_ids


def test_run_refreshes_changed_filesystem_plan_and_requires_review(
    qtbot,
    monkeypatch,
    tmp_path,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    demo = widget._create_collection_batch_demo(tmp_path / "demo")
    widget._batch_collection_dialog(config_path=demo.config_path)
    dialog = widget._active_collection_batch_dialog
    assert dialog is not None
    assert dialog.preview_table.rowCount() == 3
    qtbot.waitUntil(dialog.run_button.isEnabled, timeout=5_000)

    np.save(
        demo.root / "inputs" / "primary" / "04_added.npy",
        np.zeros((8, 8), dtype=np.uint16),
    )
    np.save(
        demo.root / "inputs" / "reference" / "delta_reference.npy",
        np.zeros((8, 8), dtype=np.uint16),
    )
    preview_calls = 0
    original_preview = widget._collection_batch_controller.preview

    def tracked_preview(**kwargs):
        nonlocal preview_calls
        preview_calls += 1
        return original_preview(**kwargs)

    monkeypatch.setattr(
        widget._collection_batch_controller,
        "preview",
        tracked_preview,
    )
    monkeypatch.setattr(
        dialog,
        "_preview_batch",
        lambda: pytest.fail("Run-plan refresh must not calculate a representative."),
    )
    qtbot.mouseClick(dialog.run_button, Qt.LeftButton)

    assert preview_calls == 1
    assert dialog.preview_table.rowCount() == 4
    assert dialog._preview_result is not None
    assert "review it" in dialog.preview_status.text().lower()
    assert not (demo.root / "results" / BATCH_MANIFEST_FILENAME).exists()

    qtbot.mouseClick(dialog.run_button, Qt.LeftButton)
    manifest_path = demo.root / "results" / BATCH_MANIFEST_FILENAME
    qtbot.waitUntil(
        lambda: (
            widget._pending_collection_batch_start is None
            and not widget._collection_batch_running
            and manifest_path.is_file()
        ),
        timeout=15_000,
    )

    assert preview_calls == 2
    assert manifest_path.is_file()
    assert "4 completed" in dialog.run_progress_label.text()


def test_preflight_failure_does_not_fill_batch_progress(qtbot, tmp_path):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    demo = widget._create_collection_batch_demo(tmp_path / "demo")
    widget._batch_collection_dialog(config_path=demo.config_path)
    dialog = widget._active_collection_batch_dialog
    assert dialog is not None
    qtbot.waitUntil(dialog.run_button.isEnabled, timeout=5_000)
    collision_path = Path(dialog.preview_table.item(0, 2).toolTip().splitlines()[0])
    collision_path.parent.mkdir(parents=True, exist_ok=True)
    collision_path.write_bytes(b"collision")

    # First click refreshes the changed preflight for review; the second tries
    # the reviewed Error-policy plan and fails before item one.
    qtbot.mouseClick(dialog.run_button, Qt.LeftButton)
    qtbot.mouseClick(dialog.run_button, Qt.LeftButton)
    qtbot.waitUntil(lambda: not widget._collection_batch_running, timeout=10_000)

    assert widget.batch_navigator.progress_bar.value() == 0
    assert widget.batch_navigator.progress_bar.format().startswith("Failed")
    assert dialog.run_progress_bar.value() == 0
    assert dialog.run_progress_bar.format() == "Failed"


def test_batch_workspace_row_navigation_progress_and_reopen_are_persistent(
    qtbot,
    tmp_path,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    demo = widget._create_collection_batch_demo(tmp_path / "demo")

    widget._batch_collection_dialog(config_path=demo.config_path)
    dialog = widget._active_collection_batch_dialog
    assert dialog is not None
    assert dialog.isVisible()
    assert dialog.preview_table.rowCount() == 3

    assert dialog.select_preview_item(1)
    qtbot.mouseClick(dialog.preview_item_button, Qt.LeftButton)
    assert widget.batch_navigator.current_index == 1
    widget.batch_navigator.slider.setValue(2)
    assert widget.batch_navigator.current_index == 2
    assert dialog.preview_table.currentRow() == 2
    qtbot.waitUntil(
        lambda: (
            widget._interactive_collection_batch_requested_index == -1
            and dialog.run_button.isEnabled()
        ),
        timeout=5_000,
    )

    qtbot.mouseClick(dialog.run_button, Qt.LeftButton)
    manifest_path = demo.root / "results" / BATCH_MANIFEST_FILENAME
    qtbot.waitUntil(
        lambda: (
            widget._pending_collection_batch_start is None
            and not widget._collection_batch_running
            and manifest_path.is_file()
        ),
        timeout=15_000,
    )

    assert dialog.isVisible()
    assert dialog.run_progress_bar.value() == 3
    assert [dialog.preview_table.item(row, 4).text() for row in range(3)] == [
        "Completed",
        "Completed",
        "Completed",
    ]
    assert "3 completed" in dialog.run_progress_label.text()
    assert "Synthetic ground truth passed" in dialog.run_result_label.text()
    assert widget.batch_navigator.progress_bar.value() == 3
    assert manifest_path.is_file()
    assert dialog._preview_result is None
    assert "Historical preflight" in dialog.preview_status.text()
    qtbot.waitUntil(dialog.run_button.isEnabled, timeout=1_000)

    qtbot.mouseClick(dialog.close_button, Qt.LeftButton)
    assert dialog.isHidden()
    qtbot.mouseClick(widget.batch_button, Qt.LeftButton)
    assert widget._active_collection_batch_dialog is dialog
    assert dialog.isVisible()
    assert "3 completed" in dialog.run_progress_label.text()

    dialog.pattern_edit.setText("*.changed")
    assert dialog._preview_result is None
    assert len(widget._interactive_collection_batch_items) == 3
    assert not widget.batch_navigator.isHidden()
    assert widget.batch_navigator.slider.isEnabled()
    assert "previous plan" in widget.batch_navigator.representative_label.text()
    assert dialog.run_button.text() == "Run batch"
    assert dialog.demo_guide_label.isHidden()
    assert widget._active_collection_batch_dialog is dialog


def test_collection_batch_demo_button_creates_loads_and_previews_bundle(
    qtbot,
    monkeypatch,
    tmp_path,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    dialog = CollectionBatchDialog(
        widget,
        source_nodes=widget._batch_source_rows(),
        actions=widget._collection_batch_dialog_actions(),
    )
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
    assert dialog.demo_path_edit.text() == str(demo_root)
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
        "cancelled": 0,
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

    widget._open_example_workflow_dialog()

    demo_root = tmp_path / SYNTHETIC_BATCH_DEMO_DIRNAME
    assert (demo_root / BATCH_CONFIG_FILENAME).is_file()
    assert "batch_output_3" in widget.pipeline.nodes
    dialog = widget._active_collection_batch_dialog
    assert dialog is not None
    assert dialog.isVisible()
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
    dialog = CollectionBatchDialog(
        widget,
        source_nodes=widget._batch_source_rows(),
        actions=widget._collection_batch_dialog_actions(),
    )
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
    from napari_vipp.core import batch as batch_module

    original_execute = batch_module.execute_pipeline_request

    def captured_execute(request, *args, **kwargs):
        calls.append(
            {
                "prune_unretained": request.prune_unretained,
                "retain_node_ids": tuple(request.retain_node_ids),
            }
        )
        return original_execute(request, *args, **kwargs)

    monkeypatch.setattr(
        batch_module,
        "execute_pipeline_request",
        captured_execute,
    )

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
    assert [binding["title"] for binding in dialog.values()["source_bindings"][:2]] == [
        "Primary",
        "Secondary",
    ]
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

    dialog = CollectionBatchDialog(
        widget,
        source_nodes=widget._batch_source_rows(),
        actions=widget._collection_batch_dialog_actions(),
    )
    qtbot.addWidget(dialog)
    dialog.input_edit.setText(str(input_dir))
    dialog.pattern_edit.setText("*.npy")
    dialog.output_edit.setText(str(output_dir))
    dialog.format_combo.setCurrentText("npy")
    dialog._preview_batch()

    assert dialog.preview_table.rowCount() == 1
    assert "exists; collision" in dialog.preview_table.item(0, 3).text()
    assert "collision" in dialog.preview_status.text().lower()
    assert "save the final graph results" in dialog.preview_status.text()


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
    assert "from napari_vipp.core.batch import (" in runner
    assert "    run_batch," in runner
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
        "cancelled": 0,
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
        "cancelled": 0,
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

    qtbot.waitUntil(
        lambda: (
            path.exists()
            and widget._active_pipeline_run_id is None
            and not widget._pipeline_run_pending
        ),
        timeout=5_000,
    )
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

    qtbot.waitUntil(
        lambda: (
            path.exists()
            and widget._active_pipeline_run_id is None
            and not widget._pipeline_run_pending
        ),
        timeout=5_000,
    )
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

    qtbot.waitUntil(
        lambda: (
            path.exists()
            and widget._active_pipeline_run_id is None
            and not widget._pipeline_run_pending
        ),
        timeout=5_000,
    )
    assert path.exists()
    assert iio.imread(path).shape == (6, 7)


def test_new_workflow_action_creates_empty_source_graph_without_prompt(
    qtbot,
    monkeypatch,
):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("binary_threshold")

    monkeypatch.setattr(
        "napari_vipp._widget.QMessageBox.question",
        lambda *_args, **_kwargs: pytest.fail(
            "opening a non-destructive workflow tab must not prompt"
        ),
    )
    widget._new_workflow_dialog()

    assert node.id in widget._workflow_tabs[0].pipeline.nodes
    assert list(widget.pipeline.nodes) == ["input"]
    assert widget.pipeline.connections == []
    assert widget.pipeline.nodes["input"].params["source_mode"] == "file path"
    assert widget.pipeline.nodes["input"].params["file_path"] == ""
    assert widget.pipeline.outputs["input"] is None
    assert widget.status_label.text() == "New empty workflow created."


def test_workflow_tab_switch_shows_loading_feedback_during_install(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    target = widget._workflow_tabs.current
    assert target is not None
    widget._new_workflow()

    target_bar_index = next(
        index
        for index in range(widget.workflow_tab_bar.count())
        if widget.workflow_tab_bar.tabData(index) == target.session_id
    )
    original_install = widget._install_workflow_tab_session
    observed: list[str] = []

    def observe_install(session):
        observed.append(session.session_id)
        assert session is target
        assert widget.workflow_tab_bar.currentIndex() == target_bar_index
        assert widget.workflow_tab_bar.tabData(target_bar_index) == target.session_id
        assert widget._workflow_tab_loading_visible
        assert not widget.pipeline_busy_label.isHidden()
        assert not widget.pipeline_busy_bar.isHidden()
        assert widget.pipeline_busy_bar.minimum() == 0
        assert widget.pipeline_busy_bar.maximum() == 0
        assert not widget.pipeline_busy_bar.isTextVisible()
        assert widget.pipeline_busy_label.text() == (
            f"Switching workflow: {target.title}"
        )
        assert widget.pipeline_cancel_button.isHidden()
        return original_install(session)

    monkeypatch.setattr(widget, "_install_workflow_tab_session", observe_install)

    widget.workflow_tab_bar.setCurrentIndex(target_bar_index)

    assert observed == [target.session_id]
    assert widget._workflow_tabs.current is target
    assert not widget._workflow_tab_loading_visible
    assert widget.pipeline_busy_label.isHidden()
    assert widget.pipeline_busy_bar.isHidden()
    assert widget.pipeline_cancel_button.isHidden()
    assert widget.pipeline_busy_label.text() == "Processing"
    assert widget.pipeline_busy_label.toolTip() == ""
    assert widget.pipeline_busy_bar.toolTip() == ""


def test_blocked_workflow_tab_switch_restores_selection_without_loading_feedback(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    target = widget._workflow_tabs.current
    assert target is not None
    widget._new_workflow()
    current = widget._workflow_tabs.current
    assert current is not None and current is not target

    target_bar_index = next(
        index
        for index in range(widget.workflow_tab_bar.count())
        if widget.workflow_tab_bar.tabData(index) == target.session_id
    )
    current_bar_index = next(
        index
        for index in range(widget.workflow_tab_bar.count())
        if widget.workflow_tab_bar.tabData(index) == current.session_id
    )
    loading_calls: list[tuple[bool, str]] = []
    original_set_loading = widget._set_workflow_tab_loading

    def record_loading(loading, title=""):
        loading_calls.append((bool(loading), str(title)))
        original_set_loading(loading, title)

    monkeypatch.setattr(
        widget,
        "_workflow_tab_switch_block_reason",
        lambda: "the test operation finishes",
    )
    monkeypatch.setattr(widget, "_set_workflow_tab_loading", record_loading)
    busy_label_hidden_before = widget.pipeline_busy_label.isHidden()
    busy_bar_hidden_before = widget.pipeline_busy_bar.isHidden()

    widget.workflow_tab_bar.setCurrentIndex(target_bar_index)

    assert widget._workflow_tabs.current is current
    assert widget.workflow_tab_bar.currentIndex() == current_bar_index
    assert loading_calls == []
    assert not widget._workflow_tab_loading_visible
    assert widget.pipeline_busy_label.isHidden() is busy_label_hidden_before
    assert widget.pipeline_busy_bar.isHidden() is busy_bar_hidden_before
    assert widget.status_label.text() == (
        "Wait until the test operation finishes before switching workflow tabs."
    )


def test_workflow_tab_switch_rolls_back_and_clears_loading_when_install_raises(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    target = widget._workflow_tabs.current
    assert target is not None
    widget._new_workflow()
    current = widget._workflow_tabs.current
    assert current is not None and current is not target
    target_bar_index = next(
        index
        for index in range(widget.workflow_tab_bar.count())
        if widget.workflow_tab_bar.tabData(index) == target.session_id
    )

    original_install = widget._install_workflow_tab_session
    install_calls: list[str] = []

    def fail_target_install(session):
        install_calls.append(session.session_id)
        if session is target:
            assert widget._workflow_tab_loading_visible
            assert not widget.pipeline_busy_label.isHidden()
            assert not widget.pipeline_busy_bar.isHidden()
            raise RuntimeError("synthetic tab activation failure")
        return original_install(session)

    monkeypatch.setattr(
        widget,
        "_install_workflow_tab_session",
        fail_target_install,
    )

    widget.workflow_tab_bar.setCurrentIndex(target_bar_index)

    assert widget._workflow_tabs.current is current
    assert widget.pipeline is current.pipeline
    assert install_calls == [target.session_id, current.session_id]
    assert not widget._workflow_tab_loading_visible
    assert widget.pipeline_busy_label.isHidden()
    assert widget.pipeline_busy_bar.isHidden()
    assert widget.pipeline_cancel_button.isHidden()
    assert widget.pipeline_busy_label.text() == "Processing"
    assert widget.pipeline_busy_label.toolTip() == ""
    assert widget.pipeline_busy_bar.toolTip() == ""
    assert "Could not switch" in widget.status_label.text()
    assert "synthetic tab activation failure" in widget.status_label.text()


def test_tab_loader_is_not_retained_by_workers_that_do_not_own_busy_strip(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    target = widget._workflow_tabs.current
    assert target is not None
    widget._new_workflow()
    target_bar_index = next(
        index
        for index in range(widget.workflow_tab_bar.count())
        if widget.workflow_tab_bar.tabData(index) == target.session_id
    )
    original_install = widget._install_workflow_tab_session

    def install_with_plot_workers(session):
        result = original_install(session)
        widget._active_input_histogram_run_id = 101
        widget._active_output_histogram_run_id = 102
        widget._active_colocalization_scatter_run_id = 103
        widget._generated_layer_contrast_pending = {("synthetic",)}
        return result

    monkeypatch.setattr(
        widget,
        "_install_workflow_tab_session",
        install_with_plot_workers,
    )

    widget.workflow_tab_bar.setCurrentIndex(target_bar_index)

    assert not widget._workflow_tab_loading_visible
    assert widget.pipeline_busy_label.isHidden()
    assert widget.pipeline_busy_bar.isHidden()
    widget._active_input_histogram_run_id = None
    widget._active_output_histogram_run_id = None
    widget._active_colocalization_scatter_run_id = None
    widget._generated_layer_contrast_pending.clear()


def test_workflow_tabs_restore_pipeline_cache_and_history_without_recompute(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: (
            widget._active_thumbnail_contrast_run_id is None
            and not widget._queued_thumbnail_contrast_limit_requests
            and not widget._pending_thumbnail_contrast_limit_keys
        ),
        timeout=5_000,
    )
    widget.graph_view.select_node("input")
    qtbot.waitUntil(
        lambda: (
            widget._active_thumbnail_contrast_run_id is None
            and not widget._queued_thumbnail_contrast_limit_requests
            and not widget._pending_thumbnail_contrast_limit_keys
        ),
        timeout=5_000,
    )
    pipeline_a = widget.pipeline
    history_a = widget._history
    cached = np.arange(9, dtype=np.float32).reshape(3, 3)
    pipeline_a.outputs["gaussian"] = cached
    pipeline_a.node_outputs["gaussian"] = [cached]
    widget._set_right_panel_visible(False)
    profile_key = ("gaussian", 0, "image", "image", False, False, None, 2)
    widget._inspect_display_profiles[profile_key] = {
        "node_id": "gaussian",
        "output_port": 0,
        "data_kind": "image",
        "display_kind": "image",
        "display_rgb": False,
        "display_rgb_as_channels": False,
        "display_ndim": 2,
        "settings": {"opacity": 0.42},
    }
    # Isolate tab-runtime persistence from the asynchronous thumbnail renderer;
    # backend derivation itself is covered by the focused statistics tests.
    monkeypatch.setattr(
        widget,
        "_sync_node_thumbnail_statistics_presentation",
        lambda *args, **kwargs: widget._sync_thumbnail_statistics_inspector(),
    )
    widget._clear_thumbnail_statistics_presentations()
    widget._set_node_thumbnail_statistics_presentation(
        "input",
        ThumbnailStatsBadgeKind.GPU,
        detail="First workflow thumbnail statistics used GPU.",
    )
    before = widget._current_history_snapshot()
    pipeline_a.set_param("gaussian", "sigma", 3.25)
    widget._push_undo_snapshot(before)
    assert history_a.can_undo

    widget._new_workflow()
    pipeline_b = widget.pipeline
    history_b = widget._history
    assert pipeline_b is not pipeline_a
    assert history_b is not history_a
    assert not widget.inspector_panel.isHidden()
    input_data = pipeline_a.outputs["input"]
    input_state = pipeline_a.output_states["input"]
    pipeline_b.outputs["input"] = input_data
    pipeline_b.output_states["input"] = input_state
    pipeline_b.node_outputs["input"] = [input_data]
    pipeline_b.node_output_states["input"] = [input_state]
    widget.thumbnail_scope_combo.setCurrentText("Slice")
    widget._update_thumbnails()
    qtbot.waitUntil(
        lambda: (
            widget._active_thumbnail_contrast_run_id is None
            and not widget._queued_thumbnail_contrast_limit_requests
            and not widget._pending_thumbnail_contrast_limit_keys
        ),
        timeout=5_000,
    )
    widget._clear_thumbnail_statistics_presentations()
    widget._set_node_thumbnail_statistics_presentation(
        "input",
        ThumbnailStatsBadgeKind.CPU,
        detail="Second workflow thumbnail statistics used CPU.",
    )

    monkeypatch.setattr(
        widget,
        "_invalidate_pipeline_cache",
        lambda: pytest.fail("tab activation must not invalidate node caches"),
    )
    monkeypatch.setattr(
        widget,
        "run_pipeline",
        lambda *args, **kwargs: pytest.fail(
            "tab activation must not recompute the graph"
        ),
    )

    first_id = widget._workflow_tabs[0].session_id
    first_bar_index = next(
        index
        for index in range(widget.workflow_tab_bar.count())
        if widget.workflow_tab_bar.tabData(index) == first_id
    )
    widget.workflow_tab_bar.setCurrentIndex(first_bar_index)

    assert widget.pipeline is pipeline_a
    assert widget.pipeline.outputs["gaussian"] is cached
    assert widget.inspector_panel.isHidden()
    assert widget._inspect_display_profiles[profile_key]["settings"]["opacity"] == 0.42
    assert widget._history is history_a
    assert widget._undo_stack is history_a.undo_stack
    assert widget._redo_stack is history_a.redo_stack
    assert widget._history.can_undo
    assert widget._thumbnail_statistics_presentations["input"].kind is (
        ThumbnailStatsBadgeKind.GPU
    )
    assert widget.thumbnail_contrast_status_value.text() == "GPU · CuPy"
    assert "First workflow" in widget.thumbnail_contrast_status_panel.toolTip()

    second_id = next(
        session.session_id
        for session in widget._workflow_tabs
        if session.pipeline is pipeline_b
    )
    second_bar_index = next(
        index
        for index in range(widget.workflow_tab_bar.count())
        if widget.workflow_tab_bar.tabData(index) == second_id
    )
    widget.workflow_tab_bar.setCurrentIndex(second_bar_index)
    assert widget.pipeline is pipeline_b
    assert widget._history is history_b
    assert not widget.inspector_panel.isHidden()
    assert widget._thumbnail_statistics_presentations["input"].kind is (
        ThumbnailStatsBadgeKind.CPU
    )
    assert widget.thumbnail_contrast_status_value.text() == "CPU · NumPy"
    assert "Second workflow" in widget.thumbnail_contrast_status_panel.toolTip()


def test_loading_workflow_opens_new_tab_and_retains_previous_session(
    qtbot,
    tmp_path,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    old_pipeline = widget.pipeline
    old_pipeline.outputs["input"] = "retained cache"
    path = save_workflow(
        tmp_path / "loaded-variant.json",
        old_pipeline,
        widget.graph_view.node_positions(),
    )

    widget.run_pipeline = lambda *args, **kwargs: None
    loaded = widget.load_workflow_file(path)

    assert loaded == path
    assert len(widget._workflow_tabs) == 2
    assert widget._workflow_tabs[0].pipeline is old_pipeline
    assert old_pipeline.outputs["input"] == "retained cache"
    assert widget.pipeline is not old_pipeline
    assert widget._workflow_tabs.current.path == path.resolve()
    assert widget._workflow_tabs.current.title == "loaded-variant"
    assert not widget._workflow_tabs.current.dirty
    assert not widget._history.can_undo
    loaded_pipeline = widget.pipeline
    widget.undo()
    assert widget.pipeline is loaded_pipeline
    assert widget.pipeline is not old_pipeline


def test_parameter_handler_immediately_marks_workflow_tab_dirty(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("gaussian_blur")
    widget.graph_view.select_node(node.id)
    session = widget._workflow_tabs.current
    assert session is not None
    session.mark_clean(
        widget._current_history_snapshot(),
        persistence_token=widget._workflow_tab_persistence_token(),
    )
    widget.workflow_tab_bar.sync_from_model(widget._workflow_tabs)

    widget._on_param_changed("sigma", float(node.params["sigma"]) + 0.5)
    widget._debounce_timer.stop()

    assert session.dirty
    assert widget.workflow_tab_bar.tabText(
        widget.workflow_tab_bar.currentIndex()
    ).endswith(" *")


def test_split_preview_channel_handler_immediately_marks_tab_dirty(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer(np.ones((2, 8, 8)), metadata={"axes": "CYX"}))
    qtbot.addWidget(widget)
    split = widget.add_node_from_palette("split_channels")
    widget.graph_view.select_node(split.id)
    monkeypatch.setattr(
        widget,
        "_refresh_split_channel_display_surfaces",
        lambda *_args, **_kwargs: None,
    )
    session = widget._workflow_tabs.current
    assert session is not None
    session.mark_clean(
        widget._current_history_snapshot(),
        persistence_token=widget._workflow_tab_persistence_token(),
    )
    widget.workflow_tab_bar.sync_from_model(widget._workflow_tabs)
    old_value = int(split.params["preview_channel"])

    widget._on_param_changed("preview_channel", 1 if old_value == 0 else 0)

    assert session.dirty
    assert widget.workflow_tab_bar.tabText(
        widget.workflow_tab_bar.currentIndex()
    ).endswith(" *")


def test_scatter_threshold_guide_immediately_marks_workflow_tab_dirty(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer(np.ones((8, 8)), metadata={"axes": "YX"}))
    qtbot.addWidget(widget)
    monkeypatch.setattr(
        widget,
        "_update_colocalization_scatter",
        lambda *_args, **_kwargs: None,
    )
    coloc = widget.add_node_from_palette("racc_index")
    widget.graph_view.select_node(coloc.id)
    session = widget._workflow_tabs.current
    assert session is not None
    session.mark_clean(
        widget._current_history_snapshot(),
        persistence_token=widget._workflow_tab_persistence_token(),
    )
    widget.workflow_tab_bar.sync_from_model(widget._workflow_tabs)

    widget._on_colocalization_scatter_threshold_changed(1, 12.5)
    widget._debounce_timer.stop()

    assert coloc.params["threshold_mode"] == "Manual"
    assert session.dirty
    assert widget.workflow_tab_bar.tabText(
        widget.workflow_tab_bar.currentIndex()
    ).endswith(" *")


def test_deferred_pipeline_refresh_remains_owned_by_origin_workflow_tab(
    qtbot,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    origin = widget._workflow_tabs.current
    assert origin is not None
    calls: list[str] = []
    widget.run_pipeline = lambda *_args, **_kwargs: calls.append(
        widget._workflow_tabs.current.session_id
    )
    widget._collection_batch_graph_refresh_pending = True
    origin.runtime_cache["_collection_batch_graph_refresh_pending"] = True

    widget._install_workflow_tab_session(origin)
    other = widget._workflow_tabs.create_blank(make_current=False)
    other.pipeline.outputs["input"] = None
    other.pipeline.output_states["input"] = None
    other_index = widget._workflow_tabs.index_of(other.session_id)
    assert widget._activate_workflow_tab(other_index, check_safety=False)

    QApplication.processEvents()

    assert calls == []
    assert origin.runtime_cache["_collection_batch_graph_refresh_pending"]

    origin_index = widget._workflow_tabs.index_of(origin.session_id)
    assert widget._activate_workflow_tab(origin_index, check_safety=False)
    QApplication.processEvents()

    assert calls == [origin.session_id]
    assert not widget._collection_batch_graph_refresh_pending
    assert not origin.runtime_cache["_collection_batch_graph_refresh_pending"]


def test_terminal_close_checks_active_batch_before_dirty_tabs(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget.pipeline.set_param("input", "file_path", "unsaved-source.tif")
    widget._sync_current_workflow_tab_state()
    widget.show()
    widget._collection_batch_running = True
    monkeypatch.setattr(
        "napari_vipp._widget.QMessageBox.question",
        lambda *_args, **_kwargs: pytest.fail(
            "dirty-tab prompts must not run while a batch owns the widget"
        ),
    )
    event = QCloseEvent()

    widget.closeEvent(event)

    assert not event.isAccepted()
    assert "batch to finish" in widget.status_label.text()
    widget._collection_batch_running = False
    widget.hide()
    session = widget._workflow_tabs.current
    assert session is not None
    session.mark_clean(
        widget._current_history_snapshot(),
        persistence_token=widget._workflow_tab_persistence_token(),
    )


def test_terminal_close_requires_explicit_queued_batch_cancellation(qtbot):
    widget = VippWidget(_Viewer(np.ones((8, 8), dtype=np.uint16)))
    qtbot.addWidget(widget)
    qtbot.waitUntil(
        lambda: widget._active_thumbnail_contrast_run_id is None,
        timeout=5_000,
    )
    widget.preview_mode_combo.setCurrentText("Off")
    dialog = widget._batch_collection_dialog(preview_config=False)
    assert dialog is not None
    run_id = 515
    cancel_event = threading.Event()
    widget._active_thumbnail_contrast_run_id = run_id
    widget._active_thumbnail_contrast_cancel_event = cancel_event
    widget._run_collection_batch_from_workspace(dialog, {"collection": "frozen"})
    assert widget._pending_collection_batch_start is not None
    first_event = QCloseEvent()

    widget.closeEvent(first_event)

    assert not first_event.isAccepted()
    assert not widget._closing
    assert widget._pending_collection_batch_start is not None
    assert cancel_event.is_set()
    assert "Cancel the queued full batch" in widget.status_label.text()

    widget.pipeline_cancel_button.click()

    assert widget._pending_collection_batch_start is None
    second_event = QCloseEvent()
    widget.closeEvent(second_event)

    assert second_event.isAccepted()
    assert widget._closing


def test_terminal_close_cancel_keeps_all_workflow_tabs_open(qtbot, monkeypatch):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget.pipeline.set_param("input", "file_path", "unsaved-source.tif")
    widget._sync_current_workflow_tab_state()
    widget.show()
    monkeypatch.setattr(
        "napari_vipp._widget.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.Cancel,
    )
    event = QCloseEvent()

    widget.closeEvent(event)

    assert not event.isAccepted()
    assert not widget._closing
    assert len(widget._workflow_tabs) == 1
    widget.hide()
    session = widget._workflow_tabs.current
    assert session is not None
    session.mark_clean(
        widget._current_history_snapshot(),
        persistence_token=widget._workflow_tab_persistence_token(),
    )


def test_terminal_close_save_dialog_failure_keeps_widget_open(qtbot, monkeypatch):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget.pipeline.set_param("input", "file_path", "unsaved-source.tif")
    widget._sync_current_workflow_tab_state()
    widget.show()
    monkeypatch.setattr(
        "napari_vipp._widget.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.Save,
    )
    monkeypatch.setattr(widget, "_save_workflow_dialog", lambda: False)
    event = QCloseEvent()

    widget.closeEvent(event)

    assert not event.isAccepted()
    assert not widget._closing
    assert "was not saved" in widget.status_label.text()
    widget.hide()
    session = widget._workflow_tabs.current
    assert session is not None
    session.mark_clean(
        widget._current_history_snapshot(),
        persistence_token=widget._workflow_tab_persistence_token(),
    )


def test_terminal_close_hidden_after_show_still_checks_dirty_tabs(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget.pipeline.set_param("input", "file_path", "unsaved-source.tif")
    widget._sync_current_workflow_tab_state()
    widget.show()
    QApplication.processEvents()
    widget.hide()
    session = widget._workflow_tabs.current
    assert session is not None
    assert widget._was_ever_visible
    assert session.dirty
    prompts: list[str] = []
    monkeypatch.setattr(
        "napari_vipp._widget.QMessageBox.question",
        lambda _parent, _title, message, *_args, **_kwargs: (
            prompts.append(str(message)) or QMessageBox.Cancel
        ),
    )
    event = QCloseEvent()

    widget.closeEvent(event)

    assert not event.isAccepted()
    assert len(prompts) == 1
    session.mark_clean(
        widget._current_history_snapshot(),
        persistence_token=widget._workflow_tab_persistence_token(),
    )


def test_terminal_close_resolves_each_dirty_tab_before_shutdown(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    first = widget._workflow_tabs.current
    assert first is not None
    widget.pipeline.set_param("input", "file_path", "first-unsaved.tif")
    widget._sync_current_workflow_tab_state()
    widget._new_workflow()
    second = widget._workflow_tabs.current
    assert second is not None and second is not first
    widget.pipeline.set_param("input", "file_path", "second-unsaved.tif")
    widget._sync_current_workflow_tab_state()
    assert first.dirty and second.dirty
    widget.show()
    responses = [QMessageBox.Save, QMessageBox.Discard]
    prompts: list[str] = []
    saved: list[str] = []

    def answer(_parent, _title, message, *_args, **_kwargs):
        prompts.append(str(message))
        return responses.pop(0)

    def save_current():
        session = widget._workflow_tabs.current
        assert session is not None
        saved.append(session.session_id)
        session.mark_clean(
            widget._current_history_snapshot(),
            persistence_token=widget._workflow_tab_persistence_token(),
        )
        return True

    monkeypatch.setattr("napari_vipp._widget.QMessageBox.question", answer)
    monkeypatch.setattr(widget, "_save_workflow_dialog", save_current)
    event = QCloseEvent()

    widget.closeEvent(event)

    assert event.isAccepted()
    assert widget._closing
    assert saved == [first.session_id]
    assert len(prompts) == 2
    assert first.title in prompts[0]
    assert second.title in prompts[1]
    assert responses == []
    repeated_event = QCloseEvent()
    widget.closeEvent(repeated_event)
    assert repeated_event.isAccepted()
    widget.hide()


def test_closing_inactive_dirty_tab_restores_active_tab_after_save_failure(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    closing = widget._workflow_tabs.current
    assert closing is not None
    widget.pipeline.set_param("input", "file_path", "unsaved-source.tif")
    widget._sync_current_workflow_tab_state()
    widget._new_workflow()
    original = widget._workflow_tabs.current
    assert original is not None and original is not closing
    monkeypatch.setattr(
        "napari_vipp._widget.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.Save,
    )
    monkeypatch.setattr(widget, "_save_workflow_dialog", lambda: False)

    closing_index = widget._workflow_tabs.index_of(closing.session_id)
    widget._close_workflow_tab(closing_index)

    assert len(widget._workflow_tabs) == 2
    assert widget._workflow_tabs.current is original
    assert widget.pipeline is original.pipeline


def test_closing_inactive_dirty_tab_restores_active_tab_after_save(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    closing = widget._workflow_tabs.current
    assert closing is not None
    widget.pipeline.set_param("input", "file_path", "unsaved-source.tif")
    widget._sync_current_workflow_tab_state()
    widget._new_workflow()
    original = widget._workflow_tabs.current
    assert original is not None and original is not closing
    monkeypatch.setattr(
        "napari_vipp._widget.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.Save,
    )

    def save_current():
        session = widget._workflow_tabs.current
        assert session is closing
        session.mark_clean(
            widget._current_history_snapshot(),
            persistence_token=widget._workflow_tab_persistence_token(),
        )
        return True

    monkeypatch.setattr(widget, "_save_workflow_dialog", save_current)

    closing_index = widget._workflow_tabs.index_of(closing.session_id)
    widget._close_workflow_tab(closing_index)

    assert len(widget._workflow_tabs) == 1
    assert widget._workflow_tabs.current is original
    assert widget.pipeline is original.pipeline


def test_dirty_workflow_tab_close_supports_cancel_then_discard(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget._new_workflow()
    before = widget._current_history_snapshot()
    widget.pipeline.set_param("input", "file_path", "unsaved-source.tif")
    widget._push_undo_snapshot(before)
    assert widget._workflow_tabs.current.dirty
    calls: list[str] = []

    monkeypatch.setattr(
        "napari_vipp._widget.QMessageBox.question",
        lambda _parent, title, *_args, **_kwargs: (
            calls.append(title) or QMessageBox.Cancel
        ),
    )
    widget._close_workflow_tab(widget.workflow_tab_bar.currentIndex())
    assert len(widget._workflow_tabs) == 2
    assert calls == ["Unsaved workflow"]

    monkeypatch.setattr(
        "napari_vipp._widget.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.Discard,
    )
    widget._close_workflow_tab(widget.workflow_tab_bar.currentIndex())
    assert len(widget._workflow_tabs) == 1
    assert widget.pipeline is widget._workflow_tabs.current.pipeline


def test_workflow_tab_rename_reorder_and_last_close_keep_valid_session(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget._new_workflow()
    widget._new_workflow()
    original_ids = tuple(session.session_id for session in widget._workflow_tabs)

    widget._rename_workflow_tab(1, "Reference")
    assert widget._workflow_tabs[1].title == "Reference"
    assert not widget._workflow_tabs[1].dirty

    widget.workflow_tab_bar.moveTab(0, 2)
    assert tuple(session.session_id for session in widget._workflow_tabs) == (
        original_ids[1],
        original_ids[2],
        original_ids[0],
    )
    assert tuple(
        widget.workflow_tab_bar.tabData(index)
        for index in range(widget.workflow_tab_bar.count())
    ) == tuple(session.session_id for session in widget._workflow_tabs)

    for session in widget._workflow_tabs:
        session.mark_clean()
    monkeypatch.setattr(
        "napari_vipp._widget.QMessageBox.question",
        lambda *_args, **_kwargs: pytest.fail(
            "clean workflow tabs do not need a close prompt"
        ),
    )
    while len(widget._workflow_tabs) > 1:
        widget._close_workflow_tab(widget.workflow_tab_bar.currentIndex())
    previous = widget._workflow_tabs.current
    widget._close_workflow_tab(widget.workflow_tab_bar.currentIndex())
    assert len(widget._workflow_tabs) == 1
    assert widget._workflow_tabs.current is not previous
    assert list(widget.pipeline.nodes) == ["input"]


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
    qtbot.waitUntil(
        lambda: (
            widget.pipeline.outputs.get(labels.id) is not None
            and widget._active_pipeline_run_id is None
            and not widget._pipeline_run_pending
        ),
        timeout=5_000,
    )
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
    rgb_output = widget.pipeline.outputs[node.id]
    expected_rgb = rgb_output.copy()
    red = viewer.layers[base_name]
    green = viewer.layers[f"{base_name} Green"]
    blue = viewer.layers[f"{base_name} Blue"]
    for layer in (red, green, blue):
        assert np.shares_memory(layer.data, rgb_output)
        assert not layer.data.flags.writeable
        assert layer.editable is False
    expected_green = green.data.copy()
    expected_blue = blue.data.copy()
    with pytest.raises(ValueError, match="read-only"):
        red.data.flat[0] = 0 if red.data.flat[0] != 0 else 1
    np.testing.assert_array_equal(rgb_output, expected_rgb)
    np.testing.assert_array_equal(green.data, expected_green)
    np.testing.assert_array_equal(blue.data, expected_blue)
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

    assert second_inspect is first_inspect
    assert second_inspect.layer_type == "image"
    assert second_inspect.metadata["display_kind"] == "image"
    assert second_inspect.metadata["data_kind"] == "mask"
    assert second_inspect.metadata["node_id"] == "threshold"
    assert second_inspect.contrast_limits == (0, 1)
    assert second_inspect.blending == "opaque"
    assert second_inspect.data.dtype == bool
    cached_mask = widget.pipeline.outputs["threshold"]
    expected_mask = cached_mask.copy()
    assert np.shares_memory(
        second_inspect.data,
        cached_mask,
    )
    assert not second_inspect.data.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        second_inspect.data.flat[0] = not second_inspect.data.flat[0]
    np.testing.assert_array_equal(cached_mask, expected_mask)


def test_otsu_and_rescale_reuse_zero_copy_image_layer(qtbot):
    data = np.arange(4 * 16 * 18, dtype=np.uint16).reshape(4, 16, 18)
    viewer = _Viewer(data, metadata={"axes": "ZYX"})
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    rescale = widget.add_node_from_palette("rescale_intensity")
    widget._connect_nodes("input", rescale.id)

    widget.inspect_node("threshold")
    inspect = viewer.layers["VIPP Inspect"]
    widget.inspect_node(rescale.id)

    rescaled_output = widget.pipeline.outputs[rescale.id]
    assert viewer.layers["VIPP Inspect"] is inspect
    assert inspect.metadata["node_id"] == rescale.id
    assert inspect.metadata["data_kind"] == "image"
    assert np.shares_memory(inspect.data, rescaled_output)
    assert not inspect.data.flags.writeable
    assert inspect.blending == "translucent"
    assert inspect.colormap == "gray"


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
    assert inspect.iso_threshold == 17.5
    assert inspect.metadata["vipp_display_contrast_pending"] is False
    assert inspect.metadata["vipp_display_contrast_basis"] == (
        "Exact full finite data range (display only)"
    )


def test_renamed_inspect_receives_background_contrast_despite_name_collision(
    qtbot,
    monkeypatch,
):
    viewer = _Viewer()
    source = viewer.layers[0]
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

    widget.inspect_node("gaussian")
    inspect = viewer.layers["VIPP Inspect"]
    qtbot.waitUntil(started.is_set, timeout=5_000)
    inspect.name = "Renamed Inspect"
    source.name = "VIPP Inspect"

    release.set()
    qtbot.waitUntil(
        lambda: not widget._generated_layer_contrast_pending,
        timeout=5_000,
    )

    assert inspect.contrast_limits == (-7.0, 42.0)
    assert inspect.iso_threshold == 17.5
    assert inspect.metadata["vipp_display_contrast_pending"] is False
    assert source.contrast_limits is None


def test_reused_mask_layer_ignores_stale_float_contrast(qtbot, monkeypatch):
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

    widget.inspect_node("gaussian")
    inspect = viewer.layers["VIPP Inspect"]
    qtbot.waitUntil(started.is_set, timeout=5_000)
    widget.inspect_node("threshold")

    assert viewer.layers["VIPP Inspect"] is inspect
    assert inspect.metadata["data_kind"] == "mask"
    assert inspect.contrast_limits == (0, 1)
    assert "_vipp_display_contrast_key" not in inspect.metadata

    release.set()
    qtbot.waitUntil(
        lambda: not widget._generated_layer_contrast_pending,
        timeout=5_000,
    )

    assert inspect.contrast_limits == (0, 1)
    assert "vipp_exact_finite_data_range" not in inspect.metadata


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


def test_racc_recalculation_preserves_user_inspect_display_settings(qtbot):
    channel_1 = np.zeros((5, 6), dtype=np.uint16)
    channel_2 = np.zeros((5, 6), dtype=np.uint16)
    channel_1[1:3, 1:3] = np.asarray(
        [[80, 100], [120, 140]],
        dtype=np.uint16,
    )
    channel_2[1:3, 1:3] = np.asarray(
        [[90, 110], [130, 150]],
        dtype=np.uint16,
    )
    channel_1[3, 1] = 150
    channel_2[3, 4] = 150
    viewer = _Viewer(
        np.stack((channel_1, channel_2)),
        metadata={"axes": "CYX"},
    )
    widget = VippWidget(viewer)
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    qtbot.addWidget(widget)
    split = widget.add_node_from_palette("split_channels")
    racc = widget.add_node_from_palette("racc_index")
    widget.pipeline.set_param(racc.id, "channel_1_threshold", 20.0)
    widget.pipeline.set_param(racc.id, "channel_2_threshold", 20.0)
    widget._connect_nodes("input", split.id)
    widget._connect_nodes(split.id, racc.id, source_port=0, target_port=0)
    widget._connect_nodes(split.id, racc.id, source_port=1, target_port=1)
    widget.run_pipeline(force_sync=True, manual_node_ids={racc.id})
    widget.graph_view.select_node(racc.id)

    inspect = viewer.layers["VIPP Inspect"]
    old_data = inspect.data
    inspect.colormap = "magma"
    inspect.visible = False
    inspect.blending = "additive"
    inspect.opacity = 0.4
    inspect.gamma = 1.7
    inspect.interpolation2d = "linear"
    inspect.contrast_limits = (0.15, 0.65)

    widget.pipeline.set_param(racc.id, "channel_1_threshold", 90.0)
    widget._mark_pipeline_dirty(racc.id)
    widget.run_pipeline(force_sync=True, manual_node_ids={racc.id})

    refreshed = viewer.layers["VIPP Inspect"]
    assert refreshed is inspect
    assert refreshed.data is not old_data
    assert np.shares_memory(refreshed.data, widget.pipeline.outputs[racc.id])
    assert refreshed.colormap == "magma"
    assert refreshed.visible is False
    assert refreshed.blending == "additive"
    assert refreshed.opacity == 0.4
    assert refreshed.gamma == 1.7
    assert refreshed.interpolation2d == "linear"
    assert refreshed.contrast_limits == (0.15, 0.65)
    assert "User-preserved" in refreshed.metadata["vipp_display_contrast_basis"]


def test_inspect_display_settings_are_remembered_per_node(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)

    widget.graph_view.select_node("gaussian")
    gaussian = widget.viewer.layers["VIPP Inspect"]
    gaussian.colormap = "magma"
    gaussian.opacity = 0.4
    gaussian.gamma = 1.7
    gaussian.interpolation2d = "linear"

    widget.graph_view.select_node("threshold")
    threshold = widget.viewer.layers["VIPP Inspect"]
    assert threshold.colormap == "gray"
    assert threshold.opacity == 1.0
    assert threshold.gamma == 1.0
    threshold.colormap = "viridis"
    threshold.opacity = 0.65

    widget.graph_view.select_node("gaussian")
    restored_gaussian = widget.viewer.layers["VIPP Inspect"]
    assert restored_gaussian.colormap == "magma"
    assert restored_gaussian.opacity == 0.4
    assert restored_gaussian.gamma == 1.7
    assert restored_gaussian.interpolation2d == "linear"

    widget.graph_view.select_node("threshold")
    restored_threshold = widget.viewer.layers["VIPP Inspect"]
    assert restored_threshold.colormap == "viridis"
    assert restored_threshold.opacity == 0.65


def test_workflow_roundtrip_restores_node_inspect_display_profile(
    qtbot,
    tmp_path,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget.graph_view.select_node("gaussian")
    inspect = widget.viewer.layers["VIPP Inspect"]
    inspect.colormap = "magma"
    inspect.visible = False
    inspect.opacity = 0.4
    inspect.gamma = 1.7
    inspect.contrast_limits = (0.2, 0.8)

    path = save_workflow(
        tmp_path / "inspect-display.json",
        widget.pipeline,
        widget.graph_view.node_positions(),
        widget._graph_note_documents(),
        widget._workflow_metadata(),
    )

    restored = VippWidget(_Viewer())
    restored._should_run_pipeline_in_background = lambda *args, **kwargs: False
    qtbot.addWidget(restored)
    restored.load_workflow_file(path)

    restored_inspect = restored.viewer.layers["VIPP Inspect"]
    assert restored._selected_node_id == "gaussian"
    assert restored_inspect.metadata["node_id"] == "gaussian"
    assert restored_inspect.colormap == "magma"
    assert restored_inspect.visible is False
    assert restored_inspect.opacity == 0.4
    assert restored_inspect.gamma == 1.7
    assert restored_inspect.contrast_limits == (0.2, 0.8)


def test_reset_selected_inspect_display_restores_defaults_without_rerun(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget.graph_view.select_node("gaussian")
    inspect = widget.viewer.layers["VIPP Inspect"]
    output = widget.pipeline.outputs["gaussian"]
    inspect.colormap = "magma"
    inspect.visible = False
    inspect.opacity = 0.4
    inspect.blending = "additive"
    inspect.gamma = 1.7
    inspect.contrast_limits = (0.2, 0.8)

    assert not widget.reset_inspect_display_button.isHidden()
    widget.reset_inspect_display_button.click()

    reset = widget.viewer.layers["VIPP Inspect"]
    assert reset is not inspect
    assert widget.pipeline.outputs["gaussian"] is output
    assert np.shares_memory(reset.data, output)
    assert not reset.data.flags.writeable
    assert reset.colormap == "gray"
    assert reset.visible is True
    assert reset.opacity == 1.0
    assert reset.blending == "translucent"
    assert reset.gamma == 1.0
    assert reset.contrast_limits == tuple(
        reset.metadata["vipp_exact_finite_data_range"]
    )
    assert all(key[0] != "gaussian" for key in widget._inspect_display_profiles)
    assert "Reset 'Gaussian Blur' Inspect display" in widget.status_label.text()


def test_history_restore_does_not_leak_profile_to_reused_node_id(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    before_add = widget._current_history_snapshot()
    added = widget.add_node_from_palette("mip")
    key = (
        added.id,
        0,
        "image",
        "image",
        False,
        False,
        None,
        2,
    )
    widget._inspect_display_profiles[key] = {
        **widget._inspect_display_profile_identity(key),
        "settings": {"colormap": "magma"},
    }

    widget._restore_history_snapshot(before_add)

    assert key not in widget._inspect_display_profiles
    readded = widget.add_node_from_palette("mip")
    assert readded.id == added.id
    assert all(
        profile_key[0] != readded.id for profile_key in widget._inspect_display_profiles
    )


def test_history_snapshots_restore_inspect_display_profiles(qtbot):
    widget = VippWidget(_Viewer())
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    qtbot.addWidget(widget)
    widget.graph_view.select_node("gaussian")
    inspect = widget.viewer.layers["VIPP Inspect"]
    inspect.colormap = "magma"
    inspect.opacity = 0.4
    magma_snapshot = widget._current_history_snapshot()

    inspect.colormap = "viridis"
    inspect.opacity = 0.7
    viridis_snapshot = widget._current_history_snapshot()

    widget._restore_history_snapshot(magma_snapshot)
    restored = widget.viewer.layers["VIPP Inspect"]
    assert restored.colormap == "magma"
    assert restored.opacity == 0.4

    widget._restore_history_snapshot(viridis_snapshot)
    restored = widget.viewer.layers["VIPP Inspect"]
    assert restored.colormap == "viridis"
    assert restored.opacity == 0.7


def test_manual_inspect_layer_removal_hides_reset_control(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    widget.graph_view.select_node("gaussian")
    inspect = viewer.layers["VIPP Inspect"]
    assert not widget.reset_inspect_display_button.isHidden()

    viewer.layers.remove(inspect)
    widget._on_viewer_layers_changed()

    assert widget.reset_inspect_display_button.isHidden()
    assert not widget.reset_inspect_display_button.isEnabled()


def test_renamed_inspect_layer_is_reused_and_reset_without_duplicate(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    widget.graph_view.select_node("gaussian")
    inspect = viewer.layers["VIPP Inspect"]
    inspect.name = "My Inspect View"
    inspect.colormap = "magma"

    widget.inspect_node("gaussian")

    inspect_layers = [
        layer
        for layer in viewer.layers
        if layer.metadata.get("napari_vipp_kind") == "inspect"
    ]
    assert inspect_layers == [inspect]
    assert inspect.name == "My Inspect View"
    assert inspect.colormap == "magma"

    widget.reset_inspect_display_button.click()

    assert all(layer is not inspect for layer in viewer.layers)
    assert viewer.layers["VIPP Inspect"].colormap == "gray"
    assert (
        sum(
            layer.metadata.get("napari_vipp_kind") == "inspect"
            for layer in viewer.layers
        )
        == 1
    )


def test_inspect_name_collision_does_not_mutate_unrelated_layer(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    source = viewer.layers[0]
    source_data = source.data
    source.name = "VIPP Inspect"

    widget.inspect_node("gaussian")

    inspect_layers = [
        layer
        for layer in viewer.layers
        if layer.metadata.get("napari_vipp_kind") == "inspect"
    ]
    assert len(inspect_layers) == 1
    generated = inspect_layers[0]
    assert generated is not source
    assert source in viewer.layers
    assert source.data is source_data
    assert source.metadata.get("napari_vipp_kind") is None

    widget.inspect_node("gaussian")

    assert source in viewer.layers
    assert [
        layer
        for layer in viewer.layers
        if layer.metadata.get("napari_vipp_kind") == "inspect"
    ] == [generated]


def test_rgb_inspect_name_collisions_do_not_mutate_unrelated_layers(qtbot):
    data = np.zeros((3, 4, 5, 6), dtype=np.uint16)
    viewer = _Viewer(data, metadata={"axes": "CZYX"})
    base_collision = viewer.layers[0]
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("composite_to_rgb")
    widget._connect_nodes("input", node.id)
    widget.run_pipeline()
    widget._discard_inspect_layers()
    base_data = base_collision.data
    base_collision.name = "VIPP Inspect"
    green_data = np.ones((3, 3), dtype=np.float32)
    green_collision = viewer.add_image(
        green_data,
        name="VIPP Inspect Green",
        metadata={},
    )

    widget.inspect_node(node.id)

    generated = widget._rgb_channel_layers("VIPP Inspect")
    assert len(generated) == 3
    assert base_collision in viewer.layers
    assert green_collision in viewer.layers
    assert base_collision.data is base_data
    assert green_collision.data is green_data
    assert {layer.metadata["display_rgb_channel_index"] for layer in generated} == {
        0,
        1,
        2,
    }

    widget.inspect_node(node.id)

    assert base_collision in viewer.layers
    assert green_collision in viewer.layers
    assert widget._rgb_channel_layers("VIPP Inspect") == generated


def test_pinned_refresh_does_not_reset_unmanaged_display_settings(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    widget.pin_node("gaussian")
    pinned = viewer.layers["VIPP Pinned: Gaussian Blur"]
    pinned.opacity = 0.4
    pinned.gamma = 1.7
    pinned.visible = False

    widget._refresh_pinned_layer_if_active()

    assert viewer.layers["VIPP Pinned: Gaussian Blur"] is pinned
    assert pinned.opacity == 0.4
    assert pinned.gamma == 1.7
    assert pinned.visible is True


def test_racc_dtype_change_resets_only_intensity_domain_display_settings(qtbot):
    channel_1 = np.asarray(
        [[0, 0, 0], [0, 80, 140], [0, 150, 0]],
        dtype=np.uint16,
    )
    channel_2 = np.asarray(
        [[0, 0, 0], [0, 90, 150], [0, 0, 150]],
        dtype=np.uint16,
    )
    viewer = _Viewer(
        np.stack((channel_1, channel_2)),
        metadata={"axes": "CYX"},
    )
    widget = VippWidget(viewer)
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    qtbot.addWidget(widget)
    split = widget.add_node_from_palette("split_channels")
    racc = widget.add_node_from_palette("racc_index")
    widget._connect_nodes("input", split.id)
    widget._connect_nodes(split.id, racc.id, source_port=0, target_port=0)
    widget._connect_nodes(split.id, racc.id, source_port=1, target_port=1)
    widget.run_pipeline(force_sync=True, manual_node_ids={racc.id})
    widget.graph_view.select_node(racc.id)

    inspect = viewer.layers["VIPP Inspect"]
    inspect.colormap = "magma"
    inspect.opacity = 0.4
    inspect.contrast_limits = (0.15, 0.65)
    inspect.iso_threshold = 0.3

    widget.pipeline.set_param(racc.id, "output_dtype", "uint8")
    widget._mark_pipeline_dirty(racc.id)
    widget.run_pipeline(force_sync=True, manual_node_ids={racc.id})

    refreshed = viewer.layers["VIPP Inspect"]
    assert refreshed is inspect
    assert refreshed.data.dtype == np.uint8
    assert refreshed.metadata["display_dtype"] == "uint8"
    assert refreshed.colormap == "magma"
    assert refreshed.opacity == 0.4
    assert refreshed.contrast_limits != (0.15, 0.65)
    assert refreshed.contrast_limits == tuple(
        refreshed.metadata["vipp_exact_finite_data_range"]
    )
    assert refreshed.iso_threshold == pytest.approx(
        sum(refreshed.contrast_limits) * 0.5
    )

    widget.pipeline.set_param(racc.id, "output_dtype", "float32")
    widget._mark_pipeline_dirty(racc.id)
    widget.run_pipeline(force_sync=True, manual_node_ids={racc.id})

    float_again = viewer.layers["VIPP Inspect"]
    assert float_again is inspect
    assert float_again.data.dtype == np.float32
    assert float_again.colormap == "magma"
    assert float_again.opacity == 0.4
    assert float_again.contrast_limits == (0.15, 0.65)
    assert float_again.iso_threshold == 0.3


def test_same_inspect_output_refreshes_unmodified_automatic_contrast(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    metadata = {"napari_vipp_kind": "inspect", "node_id": "manual"}
    widget._set_or_add_generated_layer(
        "VIPP Inspect",
        np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4),
        metadata=metadata,
        role="inspect",
    )
    inspect = widget.viewer.layers["VIPP Inspect"]
    assert inspect.contrast_limits == (0.0, 1.0)

    widget._set_or_add_generated_layer(
        "VIPP Inspect",
        np.linspace(-2.0, 5.0, 16, dtype=np.float32).reshape(4, 4),
        metadata=metadata,
        role="inspect",
    )

    assert widget.viewer.layers["VIPP Inspect"] is inspect
    assert inspect.contrast_limits == (-2.0, 5.0)


def test_same_inspect_refresh_keeps_user_contrast_after_exact_scan(
    qtbot,
    monkeypatch,
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    metadata = {"napari_vipp_kind": "inspect", "node_id": "manual"}
    widget._set_or_add_generated_layer(
        "VIPP Inspect",
        np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4),
        metadata=metadata,
        role="inspect",
    )
    inspect = widget.viewer.layers["VIPP Inspect"]
    inspect.contrast_limits = (0.2, 0.8)

    refreshed_data = np.linspace(-7.0, 42.0, 200, dtype=np.float32).reshape(
        20,
        10,
    )
    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_BYTES", 100)
    monkeypatch.setattr("napari_vipp._widget.AUTO_BACKGROUND_MIN_ELEMENTS", 100)
    started = threading.Event()
    release = threading.Event()

    def blocking_exact_limits(values):
        assert values is refreshed_data
        started.set()
        assert release.wait(5)
        return (-7.0, 42.0)

    monkeypatch.setattr(
        "napari_vipp._widget._exact_generated_layer_contrast_limits",
        blocking_exact_limits,
    )

    widget._set_or_add_generated_layer(
        "VIPP Inspect",
        refreshed_data,
        metadata=metadata,
        role="inspect",
    )

    try:
        assert widget.viewer.layers["VIPP Inspect"] is inspect
        assert inspect.contrast_limits == (0.2, 0.8)
        assert inspect.metadata["vipp_display_contrast_pending"] is True
        qtbot.waitUntil(started.is_set, timeout=5_000)
    finally:
        release.set()

    qtbot.waitUntil(
        lambda: not widget._generated_layer_contrast_pending,
        timeout=5_000,
    )
    assert inspect.contrast_limits == (0.2, 0.8)
    assert inspect.metadata["vipp_exact_finite_data_range"] == (-7.0, 42.0)
    assert "User-adjusted" in inspect.metadata["vipp_display_contrast_basis"]


def test_same_inspect_refresh_keeps_truthful_contrast_for_nonfinite_data(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    metadata = {"napari_vipp_kind": "inspect", "node_id": "manual"}
    widget._set_or_add_generated_layer(
        "VIPP Inspect",
        np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4),
        metadata=metadata,
        role="inspect",
    )
    inspect = widget.viewer.layers["VIPP Inspect"]
    inspect.contrast_limits = (0.2, 0.8)

    widget._set_or_add_generated_layer(
        "VIPP Inspect",
        np.full((4, 4), np.nan, dtype=np.float32),
        metadata=metadata,
        role="inspect",
    )

    assert widget.viewer.layers["VIPP Inspect"] is inspect
    assert inspect.contrast_limits == (0.2, 0.8)
    assert "vipp_exact_finite_data_range" not in inspect.metadata
    assert inspect.metadata["vipp_display_contrast_basis"] == (
        "User-preserved display limits; no finite values available"
    )


def test_generated_layer_contrast_failure_keeps_user_basis_truthful(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    layer = widget.viewer.layers[0]
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
            error="scan failed",
        )
    )

    assert layer.contrast_limits == (0.2, 0.8)
    assert layer.metadata["vipp_display_contrast_basis"] == (
        "User-adjusted display limits; exact full-data scan failed"
    )


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

    assert input_inspect is mask_inspect
    assert input_inspect.layer_type == "image"
    assert input_inspect.metadata["data_kind"] == "image"
    assert input_inspect.metadata["node_id"] == "input"
    assert input_inspect.contrast_limits == (0.0, 255.0)
    assert input_inspect.metadata["vipp_display_contrast_basis"] == (
        "Exact full finite data range (display only)"
    )
    assert input_inspect.blending == "translucent"
    assert input_inspect.colormap == "gray"
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


def test_inspection_layer_is_reused_when_shape_changes_with_same_rank(qtbot):
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

    assert second_inspect is first_inspect
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


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16])
def test_clip_integer_input_uses_whole_number_controls(qtbot, dtype):
    base = 100 if dtype is np.uint8 else 1_000
    data = np.arange(base, base + 16).reshape(4, 4).astype(dtype)
    widget = VippWidget(_Viewer(data))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("clip_intensity")
    widget.pipeline.set_param(node.id, "cutoff_mode", "Values")
    widget._connect_nodes("input", node.id)

    for name in ("minimum", "maximum"):
        control = widget._parameter_widgets[name]
        assert isinstance(control.value_box, QSpinBox)
        assert control.value_box.singleStep() == 1
        assert control.slider.singleStep() >= 1
        assert control.value() == node.params[name]
        control.value_box.setValue(1)
        control.value_box.resetToDefault()
        assert control.value() == int(control.spec.default)


@pytest.mark.parametrize(
    ("operation_id", "parameter"),
    [("clip_intensity", "minimum"), ("rescale_intensity", "out_min")],
)
def test_bool_intensity_bounds_retain_fractional_controls(
    qtbot,
    operation_id,
    parameter,
):
    data = (np.arange(16).reshape(4, 4) % 2).astype(bool)
    widget = VippWidget(_Viewer(data))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette(operation_id)
    if operation_id == "clip_intensity":
        widget.pipeline.set_param(node.id, "cutoff_mode", "Values")
    widget._connect_nodes("input", node.id)

    control = widget._parameter_widgets[parameter]
    assert isinstance(control.value_box, FlexibleDoubleSpinBox)
    assert control.value_box.decimals() > 0

    control.value_box.setValue(0.5)

    assert node.params[parameter] == pytest.approx(0.5)


def test_clip_float_input_retains_fractional_authoring(qtbot):
    data = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    widget = VippWidget(_Viewer(data))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("clip_intensity")
    widget.pipeline.set_param(node.id, "cutoff_mode", "Values")
    widget._connect_nodes("input", node.id)

    control = widget._parameter_widgets["minimum"]
    assert isinstance(control.value_box, FlexibleDoubleSpinBox)
    assert control.value_box.decimals() > 0

    control.value_box.setValue(0.125)

    assert widget.pipeline.nodes[node.id].params["minimum"] == pytest.approx(0.125)


def test_rescale_integer_output_is_whole_but_explicit_input_cutoffs_are_float(qtbot):
    data = np.arange(16, dtype=np.uint16).reshape(4, 4)
    widget = VippWidget(_Viewer(data))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("rescale_intensity")
    widget.pipeline.set_param(node.id, "cutoff_mode", "Values")
    widget._connect_nodes("input", node.id)

    for name in ("out_min", "out_max"):
        control = widget._parameter_widgets[name]
        assert isinstance(control.value_box, QSpinBox)
        assert control.value_box.singleStep() == 1
        control.value_box.resetToDefault()
    for name in ("in_low_value", "in_high_value"):
        control = widget._parameter_widgets[name]
        assert isinstance(control.value_box, FlexibleDoubleSpinBox)
        assert control.value_box.decimals() > 0

    widget._parameter_widgets["in_low_value"].value_box.setValue(0.5)

    assert widget.pipeline.nodes[node.id].params["in_low_value"] == pytest.approx(0.5)


def _replace_cached_input_array(widget, data) -> None:
    widget.pipeline.outputs["input"] = data
    widget.pipeline.node_outputs["input"] = [data]
    state = image_state_from_array(data)
    widget.pipeline.output_states["input"] = state
    widget.pipeline.node_output_states["input"] = [state]


class _DtypeOnlyLazyCarrier:
    def __init__(self, dtype):
        self.dtype = np.dtype(dtype)
        self.shape = (2_000, 2_000)
        self.size = 4_000_000
        self.nbytes = self.size * self.dtype.itemsize

    def __array__(self, *args, **kwargs):
        raise AssertionError("numeric control rendering materialized lazy data")


class _NoDtypeLazyCarrier:
    shape = (2_000, 2_000)
    size = 4_000_000
    nbytes = 8_000_000

    def __array__(self, *args, **kwargs):
        raise AssertionError("numeric control rendering materialized lazy data")


def test_clip_editor_rebuilds_across_dtype_changes_without_rewriting_values(qtbot):
    integer = np.arange(16, dtype=np.uint16).reshape(4, 4)
    widget = VippWidget(_Viewer(integer))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("clip_intensity")
    widget.pipeline.set_param(node.id, "cutoff_mode", "Values")
    widget.pipeline.set_param(node.id, "minimum", 1.0)
    widget.pipeline.set_param(node.id, "maximum", 12.0)
    widget._connect_nodes("input", node.id)

    integer_control = widget._parameter_widgets["minimum"]
    assert isinstance(integer_control.value_box, QSpinBox)

    floating = integer.astype(np.float32)
    _replace_cached_input_array(widget, floating)
    assert widget._refresh_selected_parameter_controls() is False
    float_control = widget._parameter_widgets["minimum"]
    assert float_control is not integer_control
    assert isinstance(float_control.value_box, FlexibleDoubleSpinBox)

    widget.pipeline.set_param(node.id, "minimum", 1.25)
    authored = dict(node.params)
    undo_count = len(widget._undo_stack)
    _replace_cached_input_array(widget, integer)
    assert widget._refresh_selected_parameter_controls() is False

    invalid_integer_control = widget._parameter_widgets["minimum"]
    assert invalid_integer_control is not float_control
    assert isinstance(invalid_integer_control.value_box, FlexibleDoubleSpinBox)
    assert invalid_integer_control.value_box.decimals() == 0
    label = widget.parameter_form.labelForField(invalid_integer_control)
    assert "whole number required" in label.text()
    assert "1.25" in label.text()
    assert node.params == authored
    assert len(widget._undo_stack) == undo_count

    _replace_cached_input_array(widget, floating)
    widget._refresh_selected_parameter_controls()
    restored_float_control = widget._parameter_widgets["minimum"]
    assert isinstance(restored_float_control.value_box, FlexibleDoubleSpinBox)
    assert restored_float_control.value() == pytest.approx(1.25)

    _replace_cached_input_array(widget, integer)
    widget._refresh_selected_parameter_controls()
    correction_control = widget._parameter_widgets["minimum"]
    assert isinstance(correction_control.value_box, FlexibleDoubleSpinBox)
    assert correction_control.value_box.decimals() == 0
    correction_control.value_box.setValue(1.75)

    assert isinstance(widget._parameter_widgets["minimum"].value_box, QSpinBox)
    assert node.params["minimum"] == 2
    authored["minimum"] = 2
    assert node.params == authored


@pytest.mark.parametrize(
    ("dtype", "base"),
    [
        (np.uint32, 3_000_000_000),
        (np.int64, 2**60),
        (np.uint64, int(np.iinfo(np.uint64).max) - 15),
    ],
)
def test_wide_integer_clip_uses_zero_decimal_fallback_without_qt_overflow(
    qtbot,
    dtype,
    base,
):
    data = np.asarray(
        [base + offset for offset in range(16)],
        dtype=dtype,
    ).reshape(4, 4)
    widget = VippWidget(_Viewer(data))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("clip_intensity")
    widget.pipeline.set_param(node.id, "cutoff_mode", "Values")
    widget._connect_nodes("input", node.id)

    for name in ("minimum", "maximum"):
        control = widget._parameter_widgets[name]
        assert isinstance(control.value_box, FlexibleDoubleSpinBox)
        assert control.value_box.decimals() == 0
        assert control.value_box.singleStep() >= 1.0
        assert -(2**31) <= control.slider.minimum() <= control.slider.maximum()
        assert control.slider.maximum() <= 2**31 - 1
        assert control.value() == node.params[name]


def test_high_uint32_rescale_output_uses_safe_slider_and_full_entry_range(qtbot):
    data = np.arange(3_000_000_000, 3_000_000_016, dtype=np.uint32).reshape(4, 4)
    widget = VippWidget(_Viewer(data))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("rescale_intensity")
    widget._connect_nodes("input", node.id)

    for name in ("out_min", "out_max"):
        control = widget._parameter_widgets[name]
        assert isinstance(control.value_box, FlexibleDoubleSpinBox)
        assert control.value_box.decimals() == 0
        assert 0 <= control.slider.minimum() <= control.slider.maximum()
        assert control.slider.maximum() <= 1_000_000_000
        assert control.value_box.maximum() == float(np.iinfo(np.uint32).max)
        assert control.value() == node.params[name]


@pytest.mark.parametrize(
    ("name", "invalid_value", "replacement"),
    [("out_min", -1.0, 0.0), ("out_max", 100_000.0, 65_535.0)],
)
def test_rescale_out_of_dtype_saved_value_stays_visible_until_corrected(
    qtbot,
    name,
    invalid_value,
    replacement,
):
    data = np.arange(16, dtype=np.uint16).reshape(4, 4)
    widget = VippWidget(_Viewer(data))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("rescale_intensity")
    widget.pipeline.set_param(node.id, name, invalid_value)
    widget.run_pipeline = lambda *args, **kwargs: None
    widget._connect_nodes("input", node.id)

    correction = widget._parameter_widgets[name]
    assert isinstance(correction.value_box, FlexibleDoubleSpinBox)
    assert correction.value_box.decimals() == 0
    assert correction.value() == invalid_value
    label = widget.parameter_form.labelForField(correction)
    assert repr(invalid_value) in label.text()
    assert "outside" in label.text()
    assert node.params[name] == invalid_value

    correction.value_box.setValue(replacement)

    assert node.params[name] == int(replacement)
    assert isinstance(widget._parameter_widgets[name].value_box, QSpinBox)


def test_state_only_uint16_input_still_uses_integer_clip_controls(qtbot):
    data = np.arange(16, dtype=np.uint16).reshape(4, 4)
    widget = VippWidget(_Viewer(data))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("clip_intensity")
    widget.pipeline.set_param(node.id, "cutoff_mode", "Values")
    widget._connect_nodes("input", node.id)
    state = widget.pipeline.node_output_states["input"][0]
    widget.pipeline.node_outputs["input"] = [None]
    widget.pipeline.outputs["input"] = None
    widget.pipeline.node_output_states["input"] = [state]
    widget.pipeline.output_states["input"] = state

    widget._render_parameters(node.id, preserve_authored_values=True)

    assert widget.pipeline.input_data_for_node(node.id) is None
    assert widget.pipeline.input_state_for_node(node.id).dtype == "uint16"
    assert isinstance(widget._parameter_widgets["minimum"].value_box, QSpinBox)
    assert isinstance(widget._parameter_widgets["maximum"].value_box, QSpinBox)


@pytest.mark.parametrize(
    "carrier",
    [_DtypeOnlyLazyCarrier(np.uint16), _NoDtypeLazyCarrier()],
    ids=["carrier-dtype", "state-dtype"],
)
def test_lazy_dtype_carrier_is_not_materialized_for_integer_controls(
    qtbot,
    carrier,
):
    data = np.arange(16, dtype=np.uint16).reshape(4, 4)
    widget = VippWidget(_Viewer(data))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("clip_intensity")
    widget.pipeline.set_param(node.id, "cutoff_mode", "Values")
    widget._connect_nodes("input", node.id)
    # This test substitutes a dtype-only stand-in for parameter presentation,
    # not for the independent asynchronous thumbnail renderer.
    widget._update_thumbnails = lambda: None
    widget.pipeline.node_outputs["input"] = [carrier]
    widget.pipeline.outputs["input"] = carrier

    widget._render_parameters(node.id, preserve_authored_values=True)

    assert isinstance(widget._parameter_widgets["minimum"].value_box, QSpinBox)
    assert isinstance(widget._parameter_widgets["maximum"].value_box, QSpinBox)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16])
def test_mask_image_integer_input_uses_native_whole_number_control(qtbot, dtype):
    data = np.arange(16).reshape(4, 4).astype(dtype)
    widget = VippWidget(_Viewer(data))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("mask_image")
    widget.run_pipeline = lambda *args, **kwargs: None
    widget._connect_nodes("input", node.id, target_port=0)
    widget._connect_nodes("threshold", node.id, target_port=1)

    control = widget._parameter_widgets["outside_value"]
    info = np.iinfo(dtype)
    assert isinstance(control.value_box, QSpinBox)
    assert control.value_box.minimum() == int(info.min)
    assert control.value_box.maximum() == int(info.max)
    assert control.value_box.singleStep() == 1
    assert control.value() == node.params["outside_value"]

    control.value_box.setValue(7)

    assert node.params["outside_value"] == 7


@pytest.mark.parametrize("dtype", [np.float32, bool])
def test_mask_image_float_and_bool_inputs_retain_fractional_control(qtbot, dtype):
    data = np.arange(16).reshape(4, 4).astype(dtype)
    widget = VippWidget(_Viewer(data))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("mask_image")
    widget.run_pipeline = lambda *args, **kwargs: None
    widget._connect_nodes("input", node.id, target_port=0)
    widget._connect_nodes("threshold", node.id, target_port=1)

    control = widget._parameter_widgets["outside_value"]
    assert isinstance(control.value_box, FlexibleDoubleSpinBox)
    assert control.value_box.decimals() > 0

    control.value_box.setValue(0.5)

    assert node.params["outside_value"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("invalid_value", "reason", "replacement"),
    [(0.5, "fractional", 2), (256.0, "outside", 255)],
)
def test_mask_image_invalid_saved_integer_fill_stays_visible_until_corrected(
    qtbot,
    invalid_value,
    reason,
    replacement,
):
    data = np.arange(16, dtype=np.uint8).reshape(4, 4)
    widget = VippWidget(_Viewer(data))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("mask_image")
    widget.pipeline.set_param(node.id, "outside_value", invalid_value)
    widget.run_pipeline = lambda *args, **kwargs: None
    widget._connect_nodes("input", node.id, target_port=0)
    widget._connect_nodes("threshold", node.id, target_port=1)

    correction = widget._parameter_widgets["outside_value"]
    assert isinstance(correction.value_box, FlexibleDoubleSpinBox)
    assert correction.value_box.decimals() == 0
    label = widget.parameter_form.labelForField(correction)
    assert repr(invalid_value) in label.text()
    assert reason in label.text()
    assert node.params["outside_value"] == invalid_value
    if reason == "outside":
        assert correction.value() == invalid_value

    correction.value_box.setValue(replacement)

    assert node.params["outside_value"] == replacement
    assert isinstance(widget._parameter_widgets["outside_value"].value_box, QSpinBox)


def test_state_only_uint16_input_uses_integer_mask_fill_control(qtbot):
    data = np.arange(16, dtype=np.uint16).reshape(4, 4)
    widget = VippWidget(_Viewer(data))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("mask_image")
    widget.run_pipeline = lambda *args, **kwargs: None
    widget._connect_nodes("input", node.id, target_port=0)
    state = widget.pipeline.node_output_states["input"][0]
    widget.pipeline.node_outputs["input"] = [None]
    widget.pipeline.outputs["input"] = None
    widget.pipeline.node_output_states["input"] = [state]
    widget.pipeline.output_states["input"] = state

    widget._render_parameters(node.id, preserve_authored_values=True)

    control = widget._parameter_widgets["outside_value"]
    assert widget.pipeline.input_data_for_node(node.id) is None
    assert widget.pipeline.input_state_for_node(node.id).dtype == "uint16"
    assert isinstance(control.value_box, QSpinBox)
    assert control.value_box.minimum() == 0
    assert control.value_box.maximum() == int(np.iinfo(np.uint16).max)


def test_sigma_filter_has_practical_slider_and_wide_numeric_entry(qtbot):
    widget = VippWidget(_Viewer(np.zeros((8, 8), dtype=np.uint8)))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("sigma_filter")
    widget._connect_nodes("input", node.id)

    control = widget._parameter_widgets["sigma_width"]
    assert control.slider.minimum() == 0
    assert control.slider.maximum() == 10_000
    assert control.value_box.minimum() == 0.0
    assert control.value_box.maximum() == 1_000_000.0

    control.value_box.setValue(250.0)

    assert control.slider.maximum() == 10_000
    assert control.slider.value() == 10_000
    assert control.value() == 250.0
    assert widget.pipeline.nodes[node.id].params["sigma_width"] == 250.0


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


@pytest.mark.parametrize(
    ("shape", "metadata", "expected_height", "expected_width"),
    [
        ((5, 7, 3), {"axes": "ZYX"}, 7, 3),
        ((5, 7, 4), {"axes": "ZYX"}, 7, 4),
        ((7, 9, 3), {"axes": "YXC"}, 7, 9),
        ((7, 9, 3), None, 9, 3),
    ],
)
def test_crop_and_block_bounds_use_explicit_xy_or_conservative_fallback(
    qtbot,
    shape,
    metadata,
    expected_height,
    expected_width,
):
    widget = VippWidget(_Viewer(np.zeros(shape), metadata=metadata))
    qtbot.addWidget(widget)
    crop = widget.add_node_from_palette("crop_stack")
    adaptive = widget.add_node_from_palette("adaptive_mean_threshold")
    widget._connect_nodes("input", crop.id)
    widget._connect_nodes("input", adaptive.id)

    crop_specs = {
        spec.name: spec for spec in NODE_LIBRARY_BY_ID["crop_stack"].parameters
    }
    block_spec = next(
        spec
        for spec in NODE_LIBRARY_BY_ID["adaptive_mean_threshold"].parameters
        if spec.name == "block_size"
    )
    block_maximum = max(min(expected_height, expected_width), 3)
    if block_maximum % 2 == 0:
        block_maximum -= 1

    assert widget._crop_bounds(crop.id, crop_specs["top"]).maximum == (
        expected_height - 1
    )
    assert widget._crop_bounds(crop.id, crop_specs["left"]).maximum == (
        expected_width - 1
    )
    assert widget._block_size_bounds(adaptive.id, block_spec).maximum == block_maximum


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


def test_auto_contrast_button_uses_connected_explicit_rgb_semantics(qtbot):
    data = np.array(
        [[[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]]],
        dtype=np.uint8,
    )
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
            AxisMetadata("rgb", "channel"),
        ),
    )
    widget = VippWidget(_Viewer(data, metadata={"vipp_image_state": state.to_dict()}))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("linear_scale_offset")
    widget._connect_nodes("input", node.id)

    widget.auto_saturation_control.value_box.setValue(0.0)
    widget.auto_contrast_button.click()

    params = widget.pipeline.nodes[node.id].params
    np.testing.assert_allclose(params["alpha"], 255.0 / 58.7, atol=0.0001)
    np.testing.assert_allclose(params["beta"], 0.0, atol=0.0001)


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
    state = image_state_from_array(
        rgb,
        axes=(
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
            AxisMetadata("rgb", "channel"),
        ),
    )

    result = _auto_contrast_scale_offset(rgb, 0.0, state=state)

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
    rgb_state = image_state_from_array(
        rgb,
        axes=(
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
            AxisMetadata("rgb", "channel"),
        ),
    )
    rgba_state = image_state_from_array(
        rgba,
        axes=(
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
            AxisMetadata("rgba", "channel"),
        ),
    )

    rgb_result = _auto_contrast_scale_offset(rgb, 0.0, state=rgb_state)
    rgba_result = _auto_contrast_scale_offset(rgba, 0.0, state=rgba_state)

    assert rgb_result is not None
    assert rgba_result is not None
    np.testing.assert_allclose(rgba_result, rgb_result, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("x_size", [3, 4])
def test_auto_contrast_does_not_infer_rgb_from_trailing_axis_size(x_size):
    values = np.arange(2 * 5 * x_size, dtype=np.float32).reshape(2, 5, x_size)

    result = _auto_contrast_scale_offset(values, 0.0)

    assert result is not None
    _alpha, _beta, lower, upper = result
    assert lower == 0.0
    assert upper == float(values.max())


def test_auto_contrast_supports_explicit_nontrailing_rgb_axis():
    trailing = np.array(
        [[[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]]],
        dtype=np.uint8,
    )
    channel_first = np.moveaxis(trailing, -1, 0)
    state = image_state_from_array(
        channel_first,
        axes=(
            AxisMetadata("rgb", "channel"),
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
        ),
    )

    result = _auto_contrast_scale_offset(channel_first, 0.0, state=state)

    assert result is not None
    _alpha, _beta, lower, upper = result
    assert lower == 0.0
    np.testing.assert_allclose(upper, 58.7, rtol=0.0, atol=1e-5)


@pytest.mark.parametrize("saturation", [-0.1, 100.1, np.nan, np.inf, "bad"])
def test_auto_contrast_rejects_invalid_saturation_instead_of_clamping(saturation):
    with pytest.raises(ValueError, match="Auto-contrast saturation"):
        _auto_contrast_scale_offset(np.arange(4), saturation)


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
    qtbot.waitUntil(lambda: len(pool.workers) == 2)
    pool.workers[1].run()
    qtbot.waitUntil(lambda: widget._active_pipeline_run_id is None)

    params = widget.pipeline.nodes[node.id].params
    np.testing.assert_allclose(params["alpha"], 2.55, atol=0.0001)
    np.testing.assert_allclose(params["beta"], 0.0, atol=0.0001)
    assert len(widget._undo_stack) == undo_count + 1
    assert widget.auto_contrast_button.isEnabled()
    assert widget.pipeline.outputs[node.id].min() == 0
    assert widget.pipeline.outputs[node.id].max() == 255

    # Publishing the new output intentionally keeps the shared progress area
    # active until its exact thumbnail contrast worker also completes.
    qtbot.waitUntil(lambda: len(pool.workers) > 2)
    pool.workers[2].run()
    qtbot.waitUntil(
        lambda: (
            widget._active_thumbnail_contrast_run_id is None
            and widget.pipeline_busy_label.isHidden()
        )
    )


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
