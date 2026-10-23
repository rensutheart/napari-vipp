"""Opening a recipe must present its source without running manual work."""

import numpy as np
import pytest
from qtpy.QtCore import QTimer

from napari_vipp import _widget as widget_module
from napari_vipp._tests.test_widget import _Viewer
from napari_vipp._widget import (
    CACHE_MODE_KEEP_ALL,
    CACHE_MODE_LOW_MEMORY,
    CACHE_MODE_SMART,
    VippWidget,
)
from napari_vipp.core.compute import ComputeMode, ComputeRequest
from napari_vipp.core.io import registry as image_registry
from napari_vipp.core.pipeline import (
    EXECUTION_BLOCKED,
    EXECUTION_NOT_CALCULATED,
    PrototypePipeline,
)
from napari_vipp.core.workflow import save_workflow


def _saved_recipe(tmp_path, source_kind, selected):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    source = pipeline.nodes["input"]
    source_file = None
    expected = None
    if source_kind == "sample":
        source.params.update(
            source_mode="sample", sample_name="VIPP synthetic volume"
        )
    else:
        expected = np.arange(16 * 18, dtype=np.float32).reshape(16, 18)
        source_file = tmp_path / "source.npy"
        np.save(source_file, expected)
        source.params.update(source_mode="file path", file_path=str(source_file))
    manual = pipeline.add_node("intensity_histogram")
    pipeline.set_node_auto_recalculate(manual.id, False)
    downstream = pipeline.add_node("select_table_columns")
    assert pipeline.connect("input", manual.id).success
    assert pipeline.connect(manual.id, downstream.id).success
    selected_id = {"source": "input", "manual": manual.id, "blocked": downstream.id}[
        selected
    ]
    path = save_workflow(
        tmp_path / "workflow.json",
        pipeline,
        metadata={"vipp": {"inspector": {"selected_node_id": selected_id}}},
        compute_request=ComputeRequest(mode=ComputeMode.CPU),
    )
    return path, selected_id, manual.id, downstream.id, source_file, expected


def _source_inspect_layer(widget):
    return next(
        (
            layer
            for layer in widget.viewer.layers
            if getattr(layer, "metadata", {}).get("napari_vipp_kind") == "inspect"
            and layer.metadata.get("node_id") == "input"
        ),
        None,
    )


@pytest.fixture
def opening_widget(qtbot):
    widget = VippWidget(
        _Viewer(), defer_initial_run=True, initial_compute_mode=ComputeMode.CPU
    )
    qtbot.addWidget(widget)
    yield widget
    widget._discard_incomplete_startup_on_close = True


@pytest.mark.parametrize("source_kind", ["sample", "npy", "npy-async"])
@pytest.mark.parametrize("selected", ["source", "manual", "blocked"])
@pytest.mark.parametrize(
    "cache_mode",
    [CACHE_MODE_KEEP_ALL, CACHE_MODE_SMART, CACHE_MODE_LOW_MEMORY],
    ids=["keep-all", "smart", "low-memory"],
)
def test_first_open_shows_source_without_running_saved_manual_selection(
    opening_widget, qtbot, monkeypatch, tmp_path, source_kind, selected, cache_mode
):
    widget = opening_widget
    widget.cache_mode_combo.setCurrentText(cache_mode)
    path, selected_id, manual_id, downstream_id, source_file, expected = _saved_recipe(
        tmp_path, source_kind, selected
    )
    workflow_bytes = path.read_bytes()
    source_bytes = None if source_file is None else source_file.read_bytes()
    created_loaders = []
    pixel_reads = []
    if source_file is not None:
        original_read = image_registry.read_numpy

        def record_pixels(path, series_index=0):
            pixel_reads.append((path, series_index))
            return original_read(path, series_index)

        monkeypatch.setattr(image_registry, "read_numpy", record_pixels)
    if source_kind == "npy-async":
        monkeypatch.setattr(
            widget, "_file_source_should_load_async", lambda _node: True
        )
        original_worker = widget_module.SourceFileLoadWorker

        def record_worker(*args, **kwargs):
            worker = original_worker(*args, **kwargs)
            created_loaders.append(worker)
            return worker

        monkeypatch.setattr(widget_module, "SourceFileLoadWorker", record_worker)

    widget.load_workflow_file(path)

    qtbot.waitUntil(
        lambda: widget._active_source_load_id is None
        and widget._active_pipeline_run_id is None
        and widget._last_pipeline_source_signature is not None,
        timeout=10_000,
    )
    qtbot.waitUntil(lambda: _source_inspect_layer(widget) is not None, timeout=5000)
    qtbot.waitUntil(lambda: widget.graph_view.node_has_thumbnail("input"), timeout=5000)
    assert widget._selected_node_id == selected_id
    assert widget.pipeline.node_execution_states[manual_id] == EXECUTION_NOT_CALCULATED
    assert widget.pipeline.node_execution_states[downstream_id] == EXECUTION_BLOCKED
    assert manual_id not in widget.pipeline.completed_node_ids
    assert downstream_id not in widget.pipeline.completed_node_ids
    assert widget.pipeline.outputs.get(manual_id) is None
    assert widget.pipeline.outputs.get(downstream_id) is None
    if expected is None:
        expected = widget._sample_payloads()["VIPP synthetic volume"].data
    source_output = widget.pipeline.outputs["input"]
    if not (cache_mode == CACHE_MODE_LOW_MEMORY and selected == "blocked"):
        np.testing.assert_array_equal(source_output, expected)
    np.testing.assert_array_equal(_source_inspect_layer(widget).data, expected)
    assert path.read_bytes() == workflow_bytes
    if source_file is not None:
        assert source_file.read_bytes() == source_bytes
        assert len(pixel_reads) == 1
        assert pixel_reads[0][0] == source_file
    if source_kind == "npy-async":
        assert len(created_loaders) == 1
        assert {spec.node_id for spec in created_loaders[0].specs} == {"input"}
    if cache_mode == CACHE_MODE_LOW_MEMORY and selected == "blocked":
        # The initial fallback owns a short-lived cache reservation, not a new
        # persistent Low-memory rule. Once shown, its detached viewer pixels
        # stay visible even when the ordinary depth-two cache pruning resumes.
        assert not widget._initial_viewer_source_fallback
        assert "input" not in widget._cache_retention_node_ids()
        widget._apply_cache_retention()
        assert widget.pipeline.outputs["input"] is None
        np.testing.assert_array_equal(_source_inspect_layer(widget).data, expected)


@pytest.mark.parametrize("source_kind", ["sample", "npy"])
def test_reproduction_open_does_not_start_graph_or_analysis_source_loading(
    opening_widget, qtbot, monkeypatch, tmp_path, source_kind
):
    widget = opening_widget
    path, selected_id, _manual, _downstream, _source_file, _expected = _saved_recipe(
        tmp_path, source_kind, "manual"
    )
    workflow_bytes = path.read_bytes()

    def unexpected_execution(*_args, **_kwargs):
        pytest.fail("Opening a reproduction package must not execute its recipe")

    monkeypatch.setattr(widget, "run_pipeline", unexpected_execution)
    monkeypatch.setattr(widget, "_start_source_file_load", unexpected_execution)

    widget._load_workflow_file_into_active_tab(path, reproduction_opened=True)

    event_turn = []
    QTimer.singleShot(0, lambda: event_turn.append(True))
    qtbot.waitUntil(lambda: bool(event_turn), timeout=5000)
    assert widget._selected_node_id == selected_id
    assert widget._active_pipeline_run_id is None
    assert widget._active_source_load_id is None
    assert not widget.pipeline.completed_node_ids
    assert all(output is None for output in widget.pipeline.outputs.values())
    assert _source_inspect_layer(widget) is None
    assert path.read_bytes() == workflow_bytes
