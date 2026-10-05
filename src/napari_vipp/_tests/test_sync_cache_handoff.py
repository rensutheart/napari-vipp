"""Direct CPU completion must be safely reusable by the first detached run."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from napari.components import ViewerModel
from qtpy.QtCore import QPointF

from napari_vipp._tests.test_execution import _pipeline_cache_kwargs
from napari_vipp._tests.test_widget import _Viewer
from napari_vipp._widget import VippWidget
from napari_vipp.core.compute import ComputeMode, ComputeRequest
from napari_vipp.core.execution import (
    PipelineRunRequest,
    execute_pipeline_request,
    execute_synchronous_cpu_pipeline,
)
from napari_vipp.core.pipeline import (
    EXECUTION_BLOCKED,
    EXECUTION_READY,
    EXECUTION_STALE,
    PrototypePipeline,
    SourcePayload,
)
from napari_vipp.core.workflow import load_workflow, save_workflow, serialize_workflow

CPU = ComputeRequest(mode=ComputeMode.CPU)


def _request(pipeline, source, *, dirty=None, manual=None, policy=CPU):
    return PipelineRunRequest(
        run_id=1,
        workflow=serialize_workflow(pipeline, compute_request=policy),
        input_data=None,
        input_metadata=None,
        input_name="",
        source_payloads={"input": source},
        compute_request=policy,
        dirty_node_ids=None if dirty is None else frozenset(dirty),
        manual_node_ids=frozenset(manual or ()),
        **_pipeline_cache_kwargs(pipeline),
    )


def _generic_pipeline(*, manual_upstream):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    upstream = pipeline.add_node(
        "intensity_histogram" if manual_upstream else "gaussian_blur"
    )
    downstream = pipeline.add_node(
        "select_table_columns" if manual_upstream else "median_filter"
    )
    assert pipeline.connect("input", upstream.id).success
    assert pipeline.connect(upstream.id, downstream.id).success
    data = np.arange(80, dtype=np.float32).reshape(8, 10)
    data.setflags(write=False)
    source = SourcePayload(data, {"axes": "YX"}, "Known input")
    return pipeline, upstream.id, downstream.id, source


@pytest.mark.parametrize("manual_upstream", [False, True])
def test_first_detached_downstream_run_reuses_authentic_sync_upstream(manual_upstream):
    pipeline, upstream, downstream, source = _generic_pipeline(
        manual_upstream=manual_upstream
    )
    report = execute_synchronous_cpu_pipeline(
        pipeline, _request(pipeline, source, manual=set(pipeline.nodes))
    )
    cached = pipeline.outputs[upstream]
    provenance = pipeline.node_compute_provenance[upstream]
    assert set(pipeline.node_compute_provenance) == set(pipeline.nodes)
    assert {d.node_id for d in report.actual_decisions} == {upstream, downstream}
    pipeline.set_param(
        downstream,
        "columns" if manual_upstream else "size",
        "count" if manual_upstream else 3,
    )
    result = execute_pipeline_request(
        _request(pipeline, source, dirty={downstream}, manual={downstream}),
        raise_errors=True,
    )
    assert result.pipeline is not None
    assert result.pipeline.node_execution_states[upstream] == EXECUTION_READY
    assert result.pipeline.node_execution_states[downstream] == EXECUTION_READY
    assert result.pipeline.outputs[upstream] is cached
    assert result.pipeline.node_compute_provenance[upstream] == provenance
    np.testing.assert_array_equal(source.data, np.arange(80).reshape(8, 10))
    assert not source.data.flags.writeable


@pytest.mark.parametrize(
    "change", ["source_bytes", "source_metadata", "params", "policy"]
)
def test_sync_handoff_rejects_changed_manual_upstream_context(change):
    pipeline, upstream, downstream, source = _generic_pipeline(manual_upstream=True)
    execute_synchronous_cpu_pipeline(
        pipeline, _request(pipeline, source, manual=set(pipeline.nodes))
    )
    cached = pipeline.outputs[upstream]
    policy = CPU
    if change == "source_bytes":
        data = source.data.copy()
        data[0, 0] += 1
        data.setflags(write=False)
        source = replace(source, data=data)
    elif change == "source_metadata":
        source = replace(source, metadata={"axes": "YX", "physical_scale": 2})
    elif change == "params":
        pipeline.nodes[upstream].params["bin_count"] += 1
    else:
        policy = ComputeRequest(mode=ComputeMode.CUSTOM)
    result = execute_pipeline_request(
        _request(
            pipeline, source, dirty={downstream}, manual={downstream}, policy=policy
        ),
        raise_errors=True,
    )
    assert result.pipeline is not None
    assert upstream not in result.pipeline.completed_node_ids
    assert upstream not in result.pipeline.node_compute_provenance
    assert result.pipeline.node_execution_states[upstream] == EXECUTION_STALE
    assert result.pipeline.node_execution_states[downstream] == EXECUTION_BLOCKED
    assert result.pipeline.outputs[upstream] is cached


def _tracking_widget(qtbot, tmp_path):
    # Same sample, settings, histogram and plot as the user's fresh launch,
    # without an external evidence-folder dependency or a visible VIPP session.
    workflow = load_workflow(
        Path(__file__).parents[1] / "examples" / "synthetic-tracking-spots-2d.json"
    )
    pipeline = PrototypePipeline()
    pipeline.restore_graph(
        workflow["nodes"], workflow["connections"], workflow["output_tunnels"]
    )
    histogram = pipeline.add_node("intensity_histogram")
    plot = pipeline.add_node("plot_results")
    pipeline.set_param(histogram.id, "bin_count", 64)
    for name, value in (
        ("plot_type", "Scatter"),
        ("y_column", "y_index"),
        ("x_column", "t_index"),
        ("group_column", "track_id"),
        ("summary", "None"),
    ):
        pipeline.set_param(plot.id, name, value)
    assert pipeline.connect("channel", histogram.id).success
    assert pipeline.connect("tracks", plot.id).success
    path = tmp_path / "fresh-tracking.json"
    save_workflow(path, pipeline, compute_request=CPU)
    widget = VippWidget(
        ViewerModel(), defer_initial_run=True, initial_compute_mode=ComputeMode.CPU
    )
    qtbot.addWidget(widget)
    widget._load_workflow_file_into_active_tab(path, reproduction_opened=True)
    widget.run_pipeline(force_sync=True, manual_node_ids=set(widget.pipeline.nodes))
    widget.graph_view.select_node("tracks")
    assert len(widget.pipeline.nodes) == 7
    assert widget.pipeline.node_outputs["tracks"][0].row_count == 24
    assert widget.pipeline.node_outputs["tracks"][1].row_count == 4
    assert set(widget.pipeline.node_compute_provenance) == set(widget.pipeline.nodes)
    return widget


@pytest.mark.parametrize("after_debounce", [False, True])
def test_fresh_tracking_gap_zero_first_manual_recalculation(
    qtbot, tmp_path, after_debounce
):
    widget = _tracking_widget(qtbot, tmp_path)
    detections = widget.pipeline.outputs["detections"]
    provenance = widget.pipeline.node_compute_provenance["detections"]
    source = widget.pipeline.outputs["input"]
    before = source.copy()
    widget._on_param_changed("maximum_gap", 0)
    if after_debounce:
        qtbot.waitUntil(
            lambda: (
                not widget._debounce_timer.isActive()
                and widget._active_pipeline_run_id is None
            ),
            timeout=20000,
        )
    assert widget.pipeline.node_execution_states["detections"] == EXECUTION_READY
    widget._calculate_node("tracks")
    qtbot.waitUntil(
        lambda: (
            widget._active_pipeline_run_id is None
            and widget.pipeline.node_outputs["tracks"][1].row_count == 8
        ),
        timeout=20000,
    )
    assert widget.pipeline.node_execution_states["detections"] == EXECUTION_READY
    assert widget.pipeline.node_execution_states["tracks"] == EXECUTION_READY
    assert widget.pipeline.outputs["detections"] is detections
    assert widget.pipeline.node_compute_provenance["detections"] == provenance
    assert widget.pipeline.node_outputs["tracks"][0].row_count == 24
    assert widget.pipeline.nodes["tracks"].params["maximum_gap"] == 0
    np.testing.assert_array_equal(source, before)


def test_sync_shortcut_refuses_non_cpu_policy():
    pipeline, _upstream, _downstream, source = _generic_pipeline(manual_upstream=False)
    with pytest.raises(ValueError, match="explicit CPU policy"):
        execute_synchronous_cpu_pipeline(
            pipeline, _request(pipeline, source, policy=ComputeRequest(mode="auto"))
        )
    assert not pipeline.completed_node_ids
    assert not pipeline.node_compute_provenance


def test_sync_failure_publishes_only_completed_prefix_for_next_manual_retry():
    pipeline, upstream, downstream, source = _generic_pipeline(manual_upstream=True)
    pipeline.set_param(downstream, "columns", "missing_column")
    with pytest.raises(ValueError, match="missing_column"):
        execute_synchronous_cpu_pipeline(
            pipeline, _request(pipeline, source, manual=set(pipeline.nodes))
        )
    cached = pipeline.outputs[upstream]
    assert cached is not None
    assert set(pipeline.node_compute_provenance) == {"input", upstream}
    assert downstream not in pipeline.completed_node_ids
    pipeline.set_param(downstream, "columns", "count")
    result = execute_pipeline_request(
        _request(pipeline, source, dirty={downstream}, manual={downstream}),
        raise_errors=True,
    )
    assert result.pipeline is not None
    assert result.pipeline.outputs[upstream] is cached
    assert result.pipeline.node_execution_states[upstream] == EXECUTION_READY
    assert result.pipeline.node_execution_states[downstream] == EXECUTION_READY


@pytest.mark.parametrize("action", ["branch", "insert"])
def test_cpu_initialized_graph_edit_preserves_authentic_upstream(
    qtbot, monkeypatch, action
):
    widget = VippWidget(_Viewer(), initial_compute_mode=ComputeMode.CPU)
    qtbot.addWidget(widget)
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    cached = widget.pipeline.outputs["gaussian"]
    provenance = widget.pipeline.node_compute_provenance["gaussian"]
    calls = []
    original = widget.pipeline._run_node

    def counted(node_id, *args, **kwargs):
        calls.append(node_id)
        return original(node_id, *args, **kwargs)

    monkeypatch.setattr(widget.pipeline, "_run_node", counted)
    if action == "branch":
        node = widget.add_node_from_palette("binary_threshold")
        widget._connect_nodes("gaussian", node.id)
    else:
        node = widget._insert_node_on_connection(
            "median_filter", ("gaussian", "threshold", 0, 0), QPointF(250, 100)
        )
    assert node is not None and node.id in calls
    assert "input" not in calls
    assert "gaussian" not in calls
    assert widget.pipeline.outputs["gaussian"] is cached
    assert widget.pipeline.node_compute_provenance["gaussian"] == provenance


@pytest.mark.parametrize("detached", [False, True])
def test_ready_completion_without_provenance_does_not_authorize_manual_cache(detached):
    pipeline, upstream, downstream, source = _generic_pipeline(manual_upstream=True)
    execute_synchronous_cpu_pipeline(
        pipeline, _request(pipeline, source, manual=set(pipeline.nodes))
    )
    cached = pipeline.outputs[upstream]
    pipeline.node_compute_provenance.pop(upstream)
    pipeline.node_cache_lineage.pop(upstream)
    request = _request(pipeline, source, dirty={downstream}, manual={downstream})
    if detached:
        result = execute_pipeline_request(request, raise_errors=True)
        pipeline = result.pipeline
        assert pipeline is not None
    else:
        execute_synchronous_cpu_pipeline(pipeline, request)
    assert pipeline.outputs[upstream] is cached
    assert upstream not in pipeline.completed_node_ids
    assert upstream not in pipeline.node_compute_provenance
    assert pipeline.node_execution_states[upstream] == EXECUTION_STALE
    assert pipeline.node_execution_states[downstream] == EXECUTION_BLOCKED


def test_tiny_authorized_deconvolution_cache_survives_downstream_composite_edit(qtbot):
    data = np.arange(8 * 9, dtype=np.float32).reshape(8, 9)
    widget = VippWidget(
        _Viewer(data, metadata={"axes": "YX"}), initial_compute_mode=ComputeMode.CPU
    )
    qtbot.addWidget(widget)
    widget._should_run_pipeline_in_background = lambda *args, **kwargs: False
    deconvolution = widget.add_node_from_palette("richardson_lucy_deconvolution")
    combined = widget.add_node_from_palette("combine_channels")
    composite = widget.add_node_from_palette("composite_to_rgb")
    for source, target, target_port in (
        ("input", deconvolution.id, 0),
        ("input", deconvolution.id, 1),
        (deconvolution.id, combined.id, 0),
        (deconvolution.id, combined.id, 1),
        (combined.id, composite.id, 0),
    ):
        widget._connect_nodes(source, target, target_port=target_port)
    widget.pipeline.set_param(deconvolution.id, "iterations", 1)
    widget._mark_pipeline_dirty(deconvolution.id)
    widget.run_pipeline(force_sync=True, manual_node_ids={deconvolution.id})
    cached = widget.pipeline.outputs[deconvolution.id]
    provenance = widget.pipeline.node_compute_provenance[deconvolution.id]
    assert cached is not None
    assert widget.pipeline.node_execution_states[deconvolution.id] == EXECUTION_READY
    widget.graph_view.select_node(composite.id)
    widget._on_param_changed("mapping_mode", "Manual")
    widget._debounce_timer.stop()
    widget.run_pipeline(force_sync=True)
    assert widget.pipeline.outputs[deconvolution.id] is cached
    assert widget.pipeline.node_compute_provenance[deconvolution.id] == provenance
    assert widget.pipeline.node_execution_states[deconvolution.id] == EXECUTION_READY
    assert widget.pipeline.outputs[composite.id] is not None
