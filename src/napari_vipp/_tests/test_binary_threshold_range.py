"""Closed-interval threshold semantics across execution and inspector controls."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from napari_vipp._tests.test_binary_threshold_direction import _pipeline
from napari_vipp._tests.test_compute_policy_segmentation_bridge_gpu import (
    _binary_workload,
    _cuda_environment,
)
from napari_vipp._tests.test_gpu_binary_threshold_provider import (
    _FakeCupy,
    _FakeStream,
    _Progress,
    _real_cuda_or_skip,
)
from napari_vipp.core.compute_policy import evaluate_candidate_support
from napari_vipp.core.compute_specs import compute_specs_for
from napari_vipp.core.export import export_pipeline_to_python
from napari_vipp.core.gpu import cupy_binary_threshold as provider
from napari_vipp.core.operations import binary_threshold
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow

MODES = ("In range", "Outside range")


def _expected(data, mode, low=1.0, high=3.0):
    if mode == "In range":
        return (data >= low) & (data <= high)
    return (data < low) | (data > high)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "dtype", (bool, np.int16, np.uint16, np.int64, np.uint64, np.float32, np.float64)
)
@pytest.mark.parametrize("shape", ((3, 5), (2, 3, 5), (2, 2, 3, 4, 5)))
def test_range_is_elementwise_shape_preserving_and_input_is_immutable(
    mode, dtype, shape
):
    data = (np.arange(np.prod(shape)).reshape(shape) % 5).astype(dtype)[..., ::-1]
    before = data.copy()
    data.setflags(write=False)
    actual = binary_threshold(data, foreground=mode, low_threshold=1, high_threshold=3)
    np.testing.assert_array_equal(actual, _expected(data, mode), strict=True)
    np.testing.assert_array_equal(data, before, strict=True)
    assert not np.shares_memory(actual, data)
    assert not data.flags.writeable


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("low,high", ((-1, 1), (0, 0), (1.00000008, 5613.0001)))
def test_range_ieee_values_endpoints_and_gpu_fake_parity(monkeypatch, mode, low, high):
    data = np.array(
        [-np.inf, low - 1, low, -0.0, 0.0, high, high + 1, np.inf, np.nan], np.float32
    )
    stream = _FakeStream()
    monkeypatch.setattr(provider, "_cupy_module", lambda: _FakeCupy(stream))
    progress = _Progress()
    params = dict(
        foreground=mode, low_threshold=low, high_threshold=high, threshold="inactive"
    )
    expected = _expected(data, mode, low, high)
    np.testing.assert_array_equal(binary_threshold(data, **params), expected)
    np.testing.assert_array_equal(
        provider.binary_threshold(data, **params, progress=progress), expected
    )
    assert not expected[-1]  # Outside range is not logical inversion of In range.
    assert stream.synchronize_count == 1
    assert [entry[:2] for entry in progress.reports] == [(0, 1), (1, 1)]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "low,high", ((2, 1), (np.nan, 2), (1, np.inf), (-np.inf, 2), (None, 1), (0, "bad"))
)
def test_invalid_ranges_rejected_by_cpu_gpu_and_admission_before_upload(
    monkeypatch, mode, low, high
):
    cupy = _FakeCupy(_FakeStream())
    monkeypatch.setattr(provider, "_cupy_module", lambda: cupy)
    for operation in (binary_threshold, provider.binary_threshold):
        with pytest.raises(ValueError, match="Binary Threshold"):
            operation(
                np.zeros((3, 5), np.float32),
                foreground=mode,
                low_threshold=low,
                high_threshold=high,
            )
    assert not cupy.asarray_sizes
    workload = _binary_workload(foreground=mode)
    # JSON contracts reject NaN/Inf before admission. String spellings reach
    # the parameter validator and must be rejected there as well.
    low = str(low) if isinstance(low, float) and not np.isfinite(low) else low
    high = str(high) if isinstance(high, float) and not np.isfinite(high) else high
    workload = replace(
        workload,
        parameters=workload.parameters
        + (("low_threshold", low), ("high_threshold", high)),
    )
    decision = evaluate_candidate_support(
        compute_specs_for("binary_threshold", include_cpu=False)[0],
        workload,
        _cuda_environment(),
        allow_experimental=False,
    )
    assert not decision.supported
    assert not decision.fallback_allowed


@pytest.mark.parametrize("mode", MODES)
def test_range_rgb_requires_declared_luma_axis(mode):
    gray = np.array([[0, 1, 2], [3, 4, 5]], np.float32)
    rgb = np.repeat(gray[None], 3, axis=0)
    actual = binary_threshold(
        rgb, channel_axis=0, foreground=mode, low_threshold=0.5, high_threshold=3.5
    )
    np.testing.assert_array_equal(actual, _expected(gray, mode, 0.5, 3.5))
    assert binary_threshold(rgb, foreground=mode).shape == rgb.shape


@pytest.mark.parametrize("mode", MODES)
def test_range_roundtrip_export_metadata_and_cache_invalidation(mode):
    pipeline, node = _pipeline(mode)
    pipeline.set_param(node.id, "low_threshold", 1)
    pipeline.set_param(node.id, "high_threshold", 3)
    document = serialize_workflow(pipeline)
    payload = deserialize_workflow(json.loads(json.dumps(document)))
    restored = PrototypePipeline()
    restored.restore_graph(payload["nodes"], payload["connections"])
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    expected = _expected(data, mode)
    np.testing.assert_array_equal(
        restored.run(data, input_metadata={"axes": "ZYX"})[node.id], expected
    )
    assert restored.output_states[node.id].axes == restored.output_states["input"].axes
    assert restored.output_states[node.id].kind == "binary mask"
    assert mode in restored.output_states[node.id].history[-1]
    namespace = {"__name__": "exported_pipeline"}
    exec(compile(export_pipeline_to_python(pipeline), "<exported>", "exec"), namespace)
    np.testing.assert_array_equal(namespace["run_pipeline"](data)[node.id], expected)
    restored.set_param(node.id, "high_threshold", 4)
    np.testing.assert_array_equal(
        restored.run(data)[node.id], _expected(data, mode, 1, 4)
    )


@pytest.mark.parametrize("mode", MODES)
def test_range_admission_and_memory_include_the_second_comparison(mode):
    from napari_vipp.core.compute_policy import estimate_candidate_memory

    spec = compute_specs_for("binary_threshold", include_cpu=False)[0]
    single = _binary_workload()
    ranged = _binary_workload(foreground=mode, threshold="inactive")
    support = evaluate_candidate_support(
        spec,
        ranged,
        _cuda_environment(),
        allow_experimental=False,
    )
    assert support.supported
    assert not support.requires_complete_facts
    single_memory = estimate_candidate_memory(spec, single)
    range_memory = estimate_candidate_memory(spec, ranged)
    assert range_memory.total_device_peak_bytes >= (
        single_memory.total_device_peak_bytes + np.prod(ranged.input_shapes[0])
    )


@pytest.mark.parametrize("mode", MODES)
def test_real_cuda_range_is_bitwise_cpu_equivalent_and_resident(mode):
    cupy = _real_cuda_or_skip()
    for low, high in ((0.0, 0.0), (1.00000008, 5613.0001), (-2, 17906.348)):
        data = np.array(
            [np.nan, -np.inf, low - 1, low, 0, high, high + 1, np.inf], np.float32
        ).reshape(2, 4)
        device = cupy.asarray(data)[:, ::-1]
        before = device.copy()
        actual = provider.binary_threshold(
            device, foreground=mode, low_threshold=low, high_threshold=high
        )
        assert isinstance(actual, cupy.ndarray)
        assert actual.dtype == cupy.bool_
        cupy.testing.assert_array_equal(device, before)
        np.testing.assert_array_equal(
            cupy.asnumpy(actual), _expected(data[:, ::-1], mode, low, high), strict=True
        )


@pytest.mark.parametrize("mode", MODES)
def test_inspector_range_controls_markers_ordering_and_undo(qtbot, tmp_path, mode):
    from napari_vipp._tests.test_widget import _Viewer
    from napari_vipp._widget import VippWidget

    widget = VippWidget(_Viewer(np.arange(256, dtype=np.uint8).reshape(1, 16, 16)))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("binary_threshold")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)
    widget._parameter_widgets["foreground"].combo.setCurrentText(mode)
    assert "threshold" not in widget._parameter_widgets
    assert {"low_threshold", "high_threshold"} <= widget._parameter_widgets.keys()
    widget._parameter_widgets["high_threshold"].value_box.setValue(200)
    widget._parameter_widgets["low_threshold"].value_box.setValue(50)
    assert node.params["low_threshold"] == 50
    assert node.params["high_threshold"] == 200
    assert widget._parameter_widgets["low_threshold"].value_box.maximum() <= 200
    assert widget._parameter_widgets["high_threshold"].value_box.minimum() >= 50
    assert set(widget.rescale_input_histogram_plot.marker_values()) == {"low", "high"}
    panel = widget.inspector_panel
    panel.setParent(None)
    qtbot.addWidget(panel)
    panel.resize(480, 800)
    panel.show()
    qtbot.wait(50)
    assert widget.parameter_group.grab().save(str(tmp_path / "range-controls.png"))
    widget._on_input_histogram_marker_changed("low", 250)
    assert node.params["low_threshold"] == 200
    assert widget._parameter_widgets["high_threshold"].value_box.minimum() >= 200
    widget._on_input_histogram_marker_changed("high", 100)
    assert node.params["high_threshold"] == 200
    widget._parameter_widgets["foreground"].combo.setCurrentText("Above")
    assert "threshold" in widget._parameter_widgets
    assert "low_threshold" not in widget._parameter_widgets
    widget.undo()
    assert widget.pipeline.nodes[node.id].params["foreground"] == mode
    assert "low_threshold" in widget._parameter_widgets
    widget.redo()
    assert widget.pipeline.nodes[node.id].params["foreground"] == "Above"


@pytest.mark.parametrize("mode", MODES)
def test_real_cuda_range_through_the_shared_executor(mode):
    from napari_vipp._tests.test_gpu_execution_integration import _accelerated_request
    from napari_vipp.core.compute import ComputeRequest
    from napari_vipp.core.compute_registry import ComputeRegistry
    from napari_vipp.core.execution import execute_pipeline_request

    _real_cuda_or_skip()
    registry = ComputeRegistry()
    try:
        probe = registry.probe_runtime("cuda-cupy", refresh=True)
        if not probe.available:
            pytest.skip(probe.message)
        library = registry.probe_library("cupy", refresh=True)
        if not library.available:
            pytest.skip(library.message)
        pipeline, node = _pipeline(mode)
        pipeline.set_param(node.id, "low_threshold", 1)
        pipeline.set_param(node.id, "high_threshold", 3)
        data = np.array(
            [[0, 1, 2, 3, 4], [np.nan, -np.inf, np.inf, -0.0, 2]], np.float32
        )
        request = _accelerated_request(
            pipeline,
            data,
            ComputeRequest(
                mode="custom",
                runtime_id="cuda-cupy",
                device_id=probe.selected_device_id,
                node_preferences={
                    node.id: "implementation:cupy-binary-threshold-f32-exact-v1"
                },
                fallback_policy="strict",
            ),
        )
        result = execute_pipeline_request(request, compute_registry=registry)
        assert not result.error
        assert result.execution_report.cleanup_succeeded
        decisions = {
            item.node_id: item for item in result.execution_report.actual_decisions
        }
        assert decisions[node.id].implementation_id == (
            "cupy-binary-threshold-f32-exact-v1"
        )
        np.testing.assert_array_equal(
            result.pipeline.outputs[node.id], _expected(data, mode)
        )
    finally:
        registry.close()
