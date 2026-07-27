from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from napari_vipp.core.node_execution import (
    DEFAULT_CPU_NODE_EXECUTOR,
    NodeCallExecutor,
    PreparedNodeCall,
)
from napari_vipp.core.pipeline import PrototypePipeline


class _RecordingExecutor:
    def __init__(self, delegate: NodeCallExecutor = DEFAULT_CPU_NODE_EXECUTOR):
        self.delegate = delegate
        self.calls: list[PreparedNodeCall] = []

    def execute(self, call: PreparedNodeCall, /) -> Any:
        self.calls.append(call)
        return self.delegate.execute(call)


def test_single_input_executor_receives_canonical_kwargs_before_cpu_call():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    pipeline.set_param(gaussian.id, "sigma", 0.0)
    assert pipeline.connect("input", gaussian.id).success
    image = np.arange(30, dtype=np.uint16).reshape(5, 6)
    executor = _RecordingExecutor()

    outputs = pipeline.run(
        image,
        input_metadata={"axes": "YX"},
        node_executor=executor,
    )

    assert [call.operation_id for call in executor.calls] == ["gaussian_blur"]
    call = executor.calls[0]
    assert call.node_id == gaussian.id
    assert not call.multiple_inputs
    assert call.inputs == (image,)
    assert call.input_states[0].axis_order == "YX"
    assert call.kwargs["sigma"] == 0.0
    assert call.kwargs["channel_axis"] is None
    with pytest.raises(TypeError):
        call.kwargs["sigma"] = 2.0
    np.testing.assert_array_equal(outputs[gaussian.id], image)
    assert outputs[gaussian.id] is not image
    assert pipeline.output_states[gaussian.id].axis_order == "YX"


def test_multi_input_executor_preserves_target_port_order_and_list_contract():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    added = pipeline.add_node("add_images")
    assert pipeline.connect("input", added.id, target_port=1).success
    assert pipeline.connect("input", added.id, target_port=0).success
    image = np.arange(20, dtype=np.float32).reshape(4, 5)
    executor = _RecordingExecutor()

    outputs = pipeline.run(
        image,
        input_metadata={"axes": "YX"},
        node_executor=executor,
    )

    assert len(executor.calls) == 1
    call = executor.calls[0]
    assert call.operation_id == "add_images"
    assert call.multiple_inputs
    assert call.inputs == (image, image)
    assert isinstance(call.positional_input(), list)
    assert call.positional_input() == [image, image]
    assert call.kwargs["input_count"] == 2
    np.testing.assert_array_equal(outputs[added.id], image + image)


def test_resolved_spatial_kwargs_and_progress_are_prepared_before_delegation():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    background = pipeline.add_node("rolling_ball_background")
    pipeline.set_param(background.id, "radius", 1.0)
    pipeline.set_param(background.id, "spatial_mode", "2D YX")
    assert pipeline.connect("input", background.id).success
    executor = _RecordingExecutor()

    pipeline.run(
        np.arange(25, dtype=np.uint8).reshape(5, 5),
        input_metadata={"axes": "YX"},
        node_executor=executor,
    )

    call = executor.calls[0]
    assert call.kwargs["resolved_spatial_ndim"] == 2
    assert call.kwargs["channel_axis"] is None
    assert call.kwargs["progress"] is not None
    assert pipeline.nodes[background.id].params["resolved_spatial_ndim"] == 2


def _split_pipeline() -> tuple[PrototypePipeline, str]:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    split = pipeline.add_node("split_channels")
    assert pipeline.connect("input", split.id).success
    return pipeline, split.id


def test_dynamic_outputs_and_states_match_the_default_cpu_executor():
    image = np.stack(
        (
            np.full((4, 5), 7, dtype=np.uint8),
            np.full((4, 5), 19, dtype=np.uint8),
        )
    )
    expected_pipeline, expected_node_id = _split_pipeline()
    actual_pipeline, actual_node_id = _split_pipeline()
    expected_pipeline.run(image, input_metadata={"axes": "CYX"})
    executor = _RecordingExecutor()

    actual_pipeline.run(
        image,
        input_metadata={"axes": "CYX"},
        node_executor=executor,
    )

    call = executor.calls[0]
    assert call.operation_id == "split_channels"
    assert call.output_port_count == 2
    assert len(actual_pipeline.node_outputs[actual_node_id]) == 2
    assert actual_pipeline.nodes[actual_node_id].params == (
        expected_pipeline.nodes[expected_node_id].params
    )
    assert (
        actual_pipeline.node_output_states[actual_node_id]
        == (expected_pipeline.node_output_states[expected_node_id])
    )
    for expected, actual in zip(
        expected_pipeline.node_outputs[expected_node_id],
        actual_pipeline.node_outputs[actual_node_id],
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_prepared_call_rejects_mismatched_runtime_containers():
    with pytest.raises(ValueError, match="exactly one input"):
        PreparedNodeCall(
            node_id="node",
            operation_id="operation",
            cpu_function=lambda value: value,
            inputs=(1, 2),
        )
    with pytest.raises(ValueError, match="describe every prepared input"):
        PreparedNodeCall(
            node_id="node",
            operation_id="operation",
            cpu_function=lambda value: value,
            inputs=(1,),
            input_states=(None, None),
        )


def test_executor_exception_is_not_retyped_or_hidden():
    class _FailingExecutor:
        def execute(self, _call: PreparedNodeCall, /) -> Any:
            raise LookupError("executor sentinel")

    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    assert pipeline.connect("input", gaussian.id).success

    with pytest.raises(LookupError, match="executor sentinel"):
        pipeline.run(
            np.zeros((3, 3), dtype=np.float32),
            node_executor=_FailingExecutor(),
        )
