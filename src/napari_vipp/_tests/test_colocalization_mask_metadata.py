from dataclasses import replace

import numpy as np
import pytest

from napari_vipp.core import execution as execution_module
from napari_vipp.core.compute import (
    ComputeEnvironment,
    ComputeMode,
    ComputeRequest,
    DecisionReason,
    OutputPortKey,
)
from napari_vipp.core.compute_planning import plan_compute_decisions
from napari_vipp.core.compute_registry import ComputeRegistry
from napari_vipp.core.metadata import (
    AxisMetadata,
    ChannelMetadata,
    image_state_from_array,
    transform_multi_input_image_state,
)
from napari_vipp.core.pipeline import PrototypePipeline, SourcePayload


def _inputs(axis_order="TZYX"):
    shape = tuple({"T": 2, "Z": 3, "Y": 5, "X": 7}[name] for name in axis_order)
    first = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    second = np.flip(first, axis=-1).copy()
    first.flags.writeable = False
    second.flags.writeable = False
    axes = tuple(
        AxisMetadata(
            name.lower(),
            "time" if name == "T" else "space",
            unit="second" if name == "T" else "micrometer",
            scale=2.0 if name == "T" else 0.25 * (index + 1),
            translation=3.0 * index,
            source_axis=index,
        )
        for index, name in enumerate(axis_order)
    )
    states = tuple(
        image_state_from_array(
            array,
            axes=axes,
            source_name=f"channel {index + 1}",
            channels=(ChannelMetadata(name=f"Signal {index + 1}"),),
        )
        for index, array in enumerate((first, second))
    )
    return (first, second), states


def _pipeline():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    second = pipeline.add_node("input")
    mask = pipeline.add_node("colocalization_mask")
    assert pipeline.connect("input", mask.id, target_port=0).success
    assert pipeline.connect(second.id, mask.id, target_port=1).success
    pipeline.set_param(mask.id, "channel_1_threshold", 12.0)
    pipeline.set_param(mask.id, "channel_2_threshold", 18.0)
    return pipeline, second, mask


@pytest.mark.parametrize("axis_order", ["YX", "ZYX", "TZYX"])
def test_mask_pipeline_preserves_sampled_grid_without_rgb(axis_order):
    inputs, states = _inputs(axis_order)
    pipeline, second, mask = _pipeline()

    outputs = pipeline.run(
        inputs[0],
        source_payloads={
            "input": SourcePayload(inputs[0], image_state=states[0]),
            second.id: SourcePayload(inputs[1], image_state=states[1]),
        },
    )

    expected = (inputs[0] >= 12) & (inputs[1] >= 18)
    np.testing.assert_array_equal(outputs[mask.id], expected)
    result = pipeline.output_states[mask.id]
    assert outputs[mask.id].dtype == np.dtype(bool)
    assert result.shape == inputs[0].shape
    assert result.axes == states[0].axes
    assert result.kind == "binary mask"
    assert result.channels == ()
    assert "channel 1 >= 12, channel 2 >= 18" in result.history[-1]
    assert not inputs[0].flags.writeable and not inputs[1].flags.writeable


@pytest.mark.parametrize(
    "changed_axis",
    [
        {"scale": 3.0},
        {"translation": 9.0},
        {"unit": "pixel"},
        {"name": "z"},
    ],
)
def test_mask_rejects_equal_shape_but_incompatible_physical_grid(changed_axis):
    inputs, states = _inputs("YX")
    pipeline, _second, mask = _pipeline()
    mismatched = replace(
        states[1], axes=(replace(states[1].axes[0], **changed_axis), states[1].axes[1])
    )

    with pytest.raises(ValueError):
        pipeline.prepare_node_call(mask.id, inputs, (states[0], mismatched))


@pytest.mark.parametrize("input_index", [0, 1])
@pytest.mark.parametrize("channel_name", ["c", "rgb", "rgba"])
@pytest.mark.parametrize("preflight_only", [False, True])
def test_mask_rejects_multichannel_inputs_with_extraction_guidance(
    input_index, channel_name, preflight_only
):
    # Equal array shapes are not proof that each input is a scalar signal.
    # Only one source has the channel designation, so both ports must be checked.
    channel_count = 4 if channel_name == "rgba" else 3
    data = np.zeros((channel_count, 5, 7), dtype=np.uint16)
    scalar_axes = tuple(AxisMetadata(name, "space") for name in "zyx")
    scalar_state = image_state_from_array(data, axes=scalar_axes)
    channel_state = image_state_from_array(
        data,
        axes=(AxisMetadata(channel_name, "channel"), *scalar_axes[1:]),
    )
    states = [scalar_state, scalar_state]
    states[input_index] = channel_state
    pipeline, _second, mask = _pipeline()

    with pytest.raises(ValueError, match="Split Channels|Extract Channel"):
        pipeline.prepare_node_call(
            mask.id,
            (data, data),
            tuple(states),
            axis_contract_only=preflight_only,
        )


@pytest.mark.parametrize("channel_name", ["rgb", "rgba"])
def test_mask_rejects_rgb_designation_even_with_singleton_axis(channel_name):
    data = np.zeros((1, 5, 7), dtype=np.uint16)
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata(channel_name, "channel"),
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
        ),
    )
    pipeline, _second, mask = _pipeline()

    with pytest.raises(ValueError, match="Split Channels|Extract Channel"):
        pipeline.prepare_node_call(mask.id, (data, data), (state, state))


def test_mask_preserves_singleton_channel_axis_without_silent_squeeze():
    data = np.arange(35, dtype=np.uint16).reshape(1, 5, 7)
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("c", "channel"),
            AxisMetadata("y", "space", scale=0.5, unit="micrometer"),
            AxisMetadata("x", "space", scale=0.75, unit="micrometer"),
        ),
    )
    pipeline, second, mask = _pipeline()
    outputs = pipeline.run(
        data,
        source_payloads={
            "input": SourcePayload(data, image_state=state),
            second.id: SourcePayload(data, image_state=state),
        },
    )

    np.testing.assert_array_equal(outputs[mask.id], data >= 18)
    assert outputs[mask.id].shape == (1, 5, 7)
    assert pipeline.output_states[mask.id].axes == state.axes
    assert pipeline.output_states[mask.id].kind == "binary mask"


def test_mask_shape_and_boolean_preflight_matches_execution_without_kernel():
    inputs, states = _inputs()
    pipeline, _second, mask = _pipeline()
    call = pipeline.prepare_node_call(mask.id, inputs, states)
    assert call is not None

    def forbidden_kernel(*_args, **_kwargs):
        raise AssertionError("Planning must not execute the mask kernel.")

    projected = pipeline._axis_contract_preflight_results(
        replace(call, cpu_function=forbidden_kernel)
    )
    ((description, projected_state),) = projected
    expected = (inputs[0] >= 12) & (inputs[1] >= 18)
    actual_state = transform_multi_input_image_state(
        expected,
        list(states),
        operation_id="colocalization_mask",
        operation_title=mask.title,
        params=dict(call.kwargs),
    )
    assert description.shape == inputs[0].shape
    assert description.dtype == np.dtype(bool)
    assert projected_state.axes == actual_state.axes == states[0].axes
    assert projected_state.history == actual_state.history
    assert projected_state.dtype == actual_state.dtype == "bool"
    assert projected_state.kind == actual_state.kind == "binary mask"
    assert projected_state.channels == actual_state.channels == ()


def test_mask_projects_boolean_facts_for_cleanup_and_stays_cpu_only():
    inputs, states = _inputs("YX")
    pipeline, second, mask = _pipeline()
    cleanup = pipeline.add_node("remove_small_objects")
    assert pipeline.connect(mask.id, cleanup.id).success
    input_ports = (OutputPortKey("input", 0), OutputPortKey(second.id, 0))

    with ComputeRegistry() as registry:
        workloads, facts, _lineage = execution_module._assemble_workloads(
            pipeline,
            frozenset(pipeline.nodes),
            dict(zip(input_ports, inputs, strict=True)),
            dict(zip(input_ports, states, strict=True)),
            registry,
            False,
            seed_facts_by_port={},
        )
        by_node = {workload.node_id: workload for workload in workloads}
        result = plan_compute_decisions(
            ComputeRequest(mode=ComputeMode.AUTO),
            (by_node[mask.id],),
            registry=registry,
            environment=ComputeEnvironment(),
        )
        assert not registry.implementations_for_operation(
            "colocalization_mask", allow_experimental=True
        )

    assert result.decisions[0].reason == DecisionReason.NO_VALIDATED_IMPLEMENTATION
    assert by_node[cleanup.id].inputs_resolved
    assert by_node[cleanup.id].input_shapes == (inputs[0].shape,)
    assert by_node[cleanup.id].input_dtypes == ("bool",)
    assert facts[cleanup.id][0].all_finite is True
    assert {"nonnegative", "no-negative-zero"} <= set(facts[cleanup.id][0].guarantees)
    assert "integer-labels" not in facts[cleanup.id][0].guarantees
