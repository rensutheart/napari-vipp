from __future__ import annotations

import threading

import numpy as np
from qtpy.QtWidgets import QApplication

from napari_vipp._widget import PipelineRunResult, VippWidget


class _Event:
    def __init__(self) -> None:
        self.callbacks: list[object] = []

    def connect(self, callback) -> None:
        self.callbacks.append(callback)

    def disconnect(self, callback) -> None:
        self.callbacks.remove(callback)

    def emit(self) -> None:
        for callback in tuple(self.callbacks):
            callback()


class _LayerEvents:
    def __init__(self) -> None:
        self.inserted = _Event()
        self.removed = _Event()


class _DimsEvents:
    def __init__(self) -> None:
        self.current_step = _Event()
        self.point = _Event()


class _Dims:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.nsteps = shape
        self.current_step = tuple(0 for _ in shape)
        self.events = _DimsEvents()


class _Layer:
    def __init__(self, data: np.ndarray, name: str) -> None:
        self.data = data
        self.name = name
        self.metadata: dict[str, object] = {}
        self.layer_type = "image"
        self.visible = True
        self.rgb = False


class _LayerList(list[_Layer]):
    def __init__(self, layers: list[_Layer]) -> None:
        super().__init__(layers)
        self.events = _LayerEvents()

    def __getitem__(self, item):
        if isinstance(item, str):
            for layer in self:
                if layer.name == item:
                    return layer
            raise KeyError(item)
        return super().__getitem__(item)


class _Viewer:
    def __init__(self) -> None:
        data = np.zeros((4, 16, 18), dtype=np.float32)
        self.layers = _LayerList([_Layer(data, "input volume")])
        self.dims = _Dims(data.shape)

    def add_image(self, data, **kwargs):
        layer = _Layer(np.asarray(data), kwargs["name"])
        layer.metadata = kwargs.get("metadata") or {}
        layer.blending = kwargs.get("blending")
        layer.colormap = kwargs.get("colormap")
        layer.contrast_limits = kwargs.get("contrast_limits")
        layer.rgb = bool(kwargs.get("rgb", False))
        layer.scale = kwargs.get("scale")
        self.layers.append(layer)
        return layer

    def add_labels(self, data, **kwargs):
        layer = _Layer(np.asarray(data), kwargs["name"])
        layer.layer_type = "labels"
        layer.metadata = kwargs.get("metadata") or {}
        layer.scale = kwargs.get("scale")
        self.layers.append(layer)
        return layer


class _RecordingThreadPool:
    def __init__(self) -> None:
        self.queued: list[object] = [object()]
        self.started: list[object] = []
        self.clear_calls = 0

    def start(self, worker, _priority: int = 0) -> None:
        self.started.append(worker)

    def clear(self) -> None:
        self.clear_calls += 1
        self.queued.clear()


def _make_widget(qtbot) -> tuple[VippWidget, _Viewer]:
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)
    return widget, viewer


def test_close_stops_and_invalidates_all_background_work(qtbot):
    widget, _viewer = _make_widget(qtbot)
    pool = _RecordingThreadPool()
    widget._pipeline_thread_pool = pool

    widget._debounce_timer.setInterval(60_000)
    widget._debounce_timer.start()

    pipeline_cancel = threading.Event()
    input_histogram_cancel = threading.Event()
    scatter_cancel = threading.Event()

    widget._active_pipeline_run_id = 11
    widget._active_pipeline_node_id = "gaussian"
    widget._pipeline_cancel_events = {11: pipeline_cancel}
    widget._pipeline_run_context = {11: (None, "input", "gaussian", (), None)}
    widget._pipeline_run_manual_node_ids = {11: frozenset()}
    widget._pipeline_run_pending = True
    widget._inflight_dirty_node_ids = {"gaussian"}

    widget._active_source_load_id = 12
    widget._source_load_pending = True

    widget._active_thumbnail_contrast_run_id = 13
    widget._queued_thumbnail_contrast_limit_requests = {("queued",): object()}
    widget._pending_thumbnail_contrast_limit_keys = {("pending",)}

    widget._active_input_histogram_run_id = 14
    widget._active_input_histogram_key = ("input",)
    widget._active_input_histogram_cancel_event = input_histogram_cancel
    widget._pending_input_histogram_request = object()

    widget._active_output_histogram_run_id = 15
    widget._active_output_histogram_key = ("output",)
    widget._pending_output_histogram_request = object()

    widget._active_colocalization_scatter_run_id = 16
    widget._active_colocalization_scatter_key = ("scatter",)
    widget._active_colocalization_scatter_cancel_event = scatter_cancel
    widget._pending_colocalization_scatter_request = object()

    widget._active_auto_contrast_run_id = 17
    widget._active_auto_contrast_key = ("auto",)

    contrast_generation = widget._generated_layer_contrast_generation
    widget._generated_layer_contrast_pending = {(contrast_generation, "pending")}

    assert widget._debounce_timer.isActive()

    widget.close()
    QApplication.processEvents()
    widget.run_pipeline()

    assert not widget._debounce_timer.isActive()
    assert pipeline_cancel.is_set()
    assert input_histogram_cancel.is_set()
    assert scatter_cancel.is_set()
    assert pool.clear_calls == 1
    assert pool.queued == []
    assert pool.started == []

    assert widget._active_pipeline_run_id is None
    assert widget._active_pipeline_node_id is None
    assert widget._pipeline_cancel_events == {}
    assert widget._pipeline_run_context == {}
    assert widget._pipeline_run_manual_node_ids == {}
    assert not widget._pipeline_run_pending
    assert widget._inflight_dirty_node_ids is None

    assert widget._active_source_load_id is None
    assert not widget._source_load_pending

    assert widget._active_thumbnail_contrast_run_id is None
    assert widget._queued_thumbnail_contrast_limit_requests == {}
    assert widget._pending_thumbnail_contrast_limit_keys == set()

    assert widget._active_input_histogram_run_id is None
    assert widget._active_input_histogram_key is None
    assert widget._active_input_histogram_cancel_event is None
    assert widget._pending_input_histogram_request is None

    assert widget._active_output_histogram_run_id is None
    assert widget._active_output_histogram_key is None
    assert widget._pending_output_histogram_request is None

    assert widget._active_colocalization_scatter_run_id is None
    assert widget._active_colocalization_scatter_key is None
    assert widget._active_colocalization_scatter_cancel_event is None
    assert widget._pending_colocalization_scatter_request is None

    assert widget._active_auto_contrast_run_id is None
    assert widget._active_auto_contrast_key is None

    assert widget._generated_layer_contrast_generation > contrast_generation
    assert widget._generated_layer_contrast_pending == set()


def test_close_disconnects_external_viewer_events(qtbot):
    widget, viewer = _make_widget(qtbot)
    layer_events = (viewer.layers.events.inserted, viewer.layers.events.removed)
    dims_events = (viewer.dims.events.current_step, viewer.dims.events.point)

    assert all(
        widget._on_viewer_layers_changed in event.callbacks
        for event in layer_events
    )
    assert all(widget._on_dims_changed in event.callbacks for event in dims_events)

    layer_notifications: list[None] = []
    dims_notifications: list[None] = []
    widget._autobind_default_image_sources = lambda: layer_notifications.append(None)
    widget._capture_vipp_dims_from_viewer = lambda: dims_notifications.append(None)

    widget.close()

    assert all(
        widget._on_viewer_layers_changed not in event.callbacks
        for event in layer_events
    )
    assert all(widget._on_dims_changed not in event.callbacks for event in dims_events)

    for event in (*layer_events, *dims_events):
        event.emit()

    assert layer_notifications == []
    assert dims_notifications == []


def test_late_pipeline_callbacks_after_close_are_ignored(qtbot):
    widget, _viewer = _make_widget(qtbot)
    run_id = 31
    cancel_event = threading.Event()
    widget._active_pipeline_run_id = run_id
    widget._active_pipeline_node_id = "gaussian"
    widget._pipeline_cancel_events = {run_id: cancel_event}
    widget._pipeline_run_context = {
        run_id: (None, "input", "gaussian", (), {"gaussian"})
    }
    widget._pipeline_run_manual_node_ids = {run_id: frozenset()}

    widget.close()

    status_after_close = widget.status_label.text()
    execution_states_after_close = dict(widget.pipeline.node_execution_states)
    execution_messages_after_close = dict(widget.pipeline.node_execution_messages)

    widget._on_background_pipeline_progress(
        (run_id, "gaussian", 1, 2, "late progress")
    )
    widget._on_background_pipeline_finished(
        PipelineRunResult(
            run_id=run_id,
            workflow={},
            error="late worker failure",
        )
    )

    assert widget.status_label.text() == status_after_close
    assert widget.pipeline.node_execution_states == execution_states_after_close
    assert widget.pipeline.node_execution_messages == execution_messages_after_close
