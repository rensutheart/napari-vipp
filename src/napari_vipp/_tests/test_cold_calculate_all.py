"""An explicit Calculate all must evaluate a cold automatic-only recipe."""

import numpy as np
import pytest

from napari_vipp._tests.test_widget import _Viewer
from napari_vipp._widget import VippWidget
from napari_vipp.core.compute import ComputeMode
from napari_vipp.core.pipeline import EXECUTION_READY


@pytest.fixture
def cold_widget(qtbot):
    widget = VippWidget(
        _Viewer(), defer_initial_run=True, initial_compute_mode=ComputeMode.CPU
    )
    qtbot.addWidget(widget)
    widget.pipeline.nodes["input"].params.update(
        source_mode="sample", sample_name="VIPP synthetic volume"
    )
    assert widget._last_pipeline_source_signature is None
    assert not widget._pending_dirty_node_ids
    assert not widget.pipeline.manual_node_ids()
    assert not widget.pipeline.completed_node_ids
    yield widget
    widget._discard_incomplete_startup_on_close = True


def test_explicit_calculate_all_evaluates_cold_bound_automatic_graph(
    cold_widget, qtbot
):
    widget = cold_widget
    source = widget._sample_payloads()["VIPP synthetic volume"].data
    before = source.copy()
    assert not source.flags.writeable

    widget._calculate_all_nodes()

    assert set(widget.pipeline.completed_node_ids) == set(widget.pipeline.nodes)
    assert all(
        state == EXECUTION_READY
        for state in widget.pipeline.node_execution_states.values()
    )
    np.testing.assert_array_equal(widget.pipeline.outputs["input"], before)
    np.testing.assert_array_equal(source, before)
    assert not source.flags.writeable
    assert widget._last_pipeline_source_signature is not None
    qtbot.waitUntil(lambda: widget.graph_view.node_has_thumbnail("input"), timeout=5000)
    assert widget.viewer.layers["VIPP Inspect"].metadata["node_id"] == "gaussian"


def test_calculate_all_remains_noop_after_clean_automatic_completion(
    cold_widget, monkeypatch
):
    widget = cold_widget
    widget.run_pipeline()
    outputs = dict(widget.pipeline.outputs)
    signature = widget._last_pipeline_source_signature
    assert signature is not None
    assert not widget._pending_dirty_node_ids

    def unexpected_run(**_kwargs):
        pytest.fail("A settled automatic graph does not need another calculation")

    monkeypatch.setattr(widget, "run_pipeline", unexpected_run)
    widget._calculate_all_nodes()

    assert widget._last_pipeline_source_signature == signature
    assert widget.pipeline.outputs.keys() == outputs.keys()
    assert all(widget.pipeline.outputs[key] is value for key, value in outputs.items())


def test_cold_calculate_all_reports_unavailable_binding_instead_of_ready(cold_widget):
    widget = cold_widget
    widget.pipeline.nodes["input"].params.update(
        source_mode="napari layer", layer_name="missing original source"
    )

    widget._calculate_all_nodes()

    assert "No Image Source node has a selected source" in widget.status_label.text()
    assert all(value is None for value in widget.pipeline.outputs.values())
    assert widget._last_pipeline_source_signature is None
    assert not widget.graph_view.node_has_thumbnail("input")
    assert not any(
        getattr(layer, "metadata", {}).get("napari_vipp_kind") == "inspect"
        for layer in widget.viewer.layers
    )


@pytest.mark.parametrize(
    "active_field", ["_active_pipeline_run_id", "_active_source_load_id"]
)
def test_cold_calculate_all_does_not_restart_owned_inflight_work(
    cold_widget, monkeypatch, active_field
):
    widget = cold_widget

    def unexpected_run(**_kwargs):
        pytest.fail("An explicit repeat must not supersede the cold run or source load")

    monkeypatch.setattr(widget, "run_pipeline", unexpected_run)
    setattr(widget, active_field, 41)
    try:
        widget._calculate_all_nodes()
        assert getattr(widget, active_field) == 41
        assert not widget._pending_dirty_node_ids
        assert not widget._pending_manual_node_ids
    finally:
        setattr(widget, active_field, None)


def test_pending_recovery_does_not_suppress_distinct_starter_sample(
    cold_widget, monkeypatch, qtbot
):
    widget = cold_widget
    widget._workflow_recovery.pending_recovery = True
    original_run = widget.run_pipeline
    calls = []

    def record_run(**kwargs):
        calls.append(kwargs)
        return original_run(**kwargs)

    monkeypatch.setattr(widget, "run_pipeline", record_run)

    assert widget.run_initial_pipeline_once()
    assert widget.pipeline.outputs["input"].shape == (12, 96, 128)
    qtbot.waitUntil(lambda: widget.graph_view.node_has_thumbnail("input"), timeout=5000)
    assert widget._workflow_recovery.pending_recovery
    assert not widget.run_initial_pipeline_once()
    assert len(calls) == 1
