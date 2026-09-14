from __future__ import annotations

import numpy as np
import pytest

from napari_vipp.core import operations
from napari_vipp.core.pipeline import (
    COLOCALIZATION_CATEGORY,
    PrototypePipeline,
    SourcePayload,
)
from napari_vipp.core.progress import OperationCancelled, ProgressContext
from napari_vipp.core.workflow import load_workflow, save_workflow


@pytest.mark.parametrize("shape", [(4, 6), (2, 4, 6), (2, 3, 4, 6)])
def test_mask_matches_white_region_without_rgb_axis_and_preserves_inputs(shape):
    channel_1 = (np.arange(np.prod(shape)) % 6).reshape(shape).astype(np.uint16)
    channel_2 = np.flip(channel_1, axis=-1).copy()
    before_1, before_2 = channel_1.copy(), channel_2.copy()
    channel_1.flags.writeable = False
    channel_2.flags.writeable = False

    result = operations.colocalization_mask(
        [channel_1, channel_2],
        channel_1_threshold=2,
        channel_2_threshold=2,
    )
    overlay = operations.colocalized_voxels(
        [channel_1, channel_2],
        channel_1_threshold=2,
        channel_2_threshold=2,
        display_mode="White on black",
    )

    assert result.shape == shape
    assert result.dtype == np.bool_
    np.testing.assert_array_equal(result, np.all(overlay == 1, axis=-1))
    np.testing.assert_array_equal(result, (channel_1 >= 2) & (channel_2 >= 2))
    np.testing.assert_array_equal(channel_1, before_1)
    np.testing.assert_array_equal(channel_2, before_2)
    assert not np.shares_memory(result, channel_1)
    assert not np.shares_memory(result, channel_2)


@pytest.mark.parametrize(
    ("values_1", "values_2", "dtype", "threshold_1", "threshold_2"),
    [
        ([30000, 43970, 43971, 65535], [65535, 65535, 48074, 48073], np.uint16,
         43970.51, 48073.03),
        ([-3, -1, 0, 2], [4, -2, 1, 0], np.int16, -1, 0),
        ([-0.75, -0.5, 0.25, 0.5], [0.5, 0.5, 0.25, -0.25], np.float32,
         -0.5, 0.25),
        ([False, True, True, False], [True, False, True, False], bool, 1, 1),
        ([2**32 - 1, 2**32, 2**32 + 1, 2**32 + 2],
         [2**32 + 2, 2**32, 2**32 - 1, 2**32 + 1], np.uint64, 2**32, 2**32),
    ],
)
def test_mask_uses_native_values_and_inclusive_thresholds(
    values_1, values_2, dtype, threshold_1, threshold_2,
):
    channel_1 = np.asarray(values_1, dtype=dtype).reshape(2, 2)
    channel_2 = np.asarray(values_2, dtype=dtype).reshape(2, 2)
    result = operations.colocalization_mask(
        [channel_1, channel_2],
        channel_1_threshold=threshold_1,
        channel_2_threshold=threshold_2,
    )
    expected = (channel_1 >= threshold_1) & (channel_2 >= threshold_2)
    np.testing.assert_array_equal(result, expected)


def test_mask_uint64_at_float_exact_integer_limit_matches_overlay():
    # Native integers up to and including 2**53 are exact under the shared
    # floating-point threshold comparison. The immediately smaller value must
    # not be confused with this inclusive threshold.
    threshold = 2**53
    channel_1 = np.array(
        [[threshold - 1, threshold], [threshold - 2, threshold]],
        dtype=np.uint64,
    )
    channel_2 = np.full((2, 2), threshold, dtype=np.uint64)
    result = operations.colocalization_mask(
        [channel_1, channel_2],
        channel_1_threshold=threshold,
        channel_2_threshold=threshold,
    )
    overlay = operations.colocalized_voxels(
        [channel_1, channel_2],
        channel_1_threshold=threshold,
        channel_2_threshold=threshold,
        display_mode="White on black",
    )
    np.testing.assert_array_equal(result, [[False, True], [False, True]])
    np.testing.assert_array_equal(result, np.all(overlay == 1, axis=-1))


@pytest.mark.parametrize(
    ("value", "dtype"),
    [(2**53 + 1, np.uint64), (2**54 - 1, np.uint64),
     (2**53 + 1, np.int64), (-(2**53) - 1, np.int64),
     (-(2**54) + 1, np.int64)],
)
@pytest.mark.parametrize("threshold_mode", ["Manual", "Costes auto"])
def test_mask_rejects_wide_integers_before_float_comparison_or_costes_fit(
    value, dtype, threshold_mode, monkeypatch,
):
    channel = np.full((2, 2), value, dtype=dtype)
    monkeypatch.setattr(
        operations,
        "_costes_thresholds",
        lambda *args, **kwargs: pytest.fail("Unsafe integer data reached Costes"),
    )
    with pytest.raises(ValueError, match="cannot compare integer values.*exactly"):
        operations.colocalization_mask(
            [channel, channel],
            channel_1_threshold=2**53,
            channel_2_threshold=2**53,
            threshold_mode=threshold_mode,
        )


def test_pipeline_rejects_wide_integers_before_costes_fit(monkeypatch):
    pipeline, channel_2, mask = _mask_pipeline()
    pipeline.set_param(mask.id, "threshold_mode", "Costes auto")
    channel = np.full((2, 2), 2**53 + 1, dtype=np.uint64)
    monkeypatch.setattr(
        operations,
        "_costes_thresholds",
        lambda *args, **kwargs: pytest.fail("Unsafe integer data reached Costes"),
    )
    with pytest.raises(ValueError, match="cannot compare integer values.*exactly"):
        pipeline.run(
            channel,
            input_metadata={"axes": "YX"},
            source_payloads={channel_2.id: SourcePayload(channel, {"axes": "YX"})},
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_mask_rejects_nonfinite_channel_values(value):
    channel = np.array([[0, 1], [2, value]])
    with pytest.raises(ValueError, match="only finite values"):
        operations.colocalization_mask([channel, np.ones((2, 2))])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_mask_rejects_nonfinite_manual_thresholds(value):
    with pytest.raises(ValueError, match="finite channel thresholds"):
        operations.colocalization_mask(
            [np.ones((2, 2)), np.ones((2, 2))],
            channel_1_threshold=value,
        )


@pytest.mark.parametrize("count", [0, 1, 3])
def test_mask_requires_exactly_two_channels(count):
    with pytest.raises(ValueError, match="exactly two image inputs"):
        operations.colocalization_mask([np.ones((2, 2))] * count)


@pytest.mark.parametrize("dtype", [complex, object, str])
def test_mask_rejects_nonscalar_numeric_images(dtype):
    with pytest.raises(ValueError, match="scalar numeric image data"):
        operations.colocalization_mask(
            [np.ones((2, 2), dtype=dtype), np.ones((2, 2))]
        )


def test_mask_rejects_shape_mismatch_and_1d_data():
    with pytest.raises(ValueError, match="Input shapes differ"):
        operations.colocalization_mask([np.ones((2, 2)), np.ones((2, 3))])
    with pytest.raises(ValueError, match="at least 2D"):
        operations.colocalization_mask([np.ones(4), np.ones(4)])


def test_mask_empty_and_no_overlap_are_valid_boolean_outputs():
    empty = operations.colocalization_mask(
        [np.empty((0, 3)), np.empty((0, 3))]
    )
    assert empty.shape == (0, 3)
    assert empty.dtype == bool
    no_overlap = operations.colocalization_mask(
        [np.array([[4, 0], [4, 0]]), np.array([[0, 4], [0, 4]])],
        channel_1_threshold=1,
        channel_2_threshold=1,
    )
    assert not no_overlap.any()


def test_mask_costes_matches_overlay_with_actual_fit():
    channel_1 = np.arange(64, dtype=np.uint16).reshape(8, 8)
    channel_2 = np.roll(channel_1, 2, axis=0)
    result = operations.colocalization_mask(
        [channel_1, channel_2], threshold_mode="Costes auto"
    )
    overlay = operations.colocalized_voxels(
        [channel_1, channel_2],
        threshold_mode="Costes auto",
        display_mode="White on black",
    )
    np.testing.assert_array_equal(result, np.all(overlay == 1, axis=-1))


def test_mask_costes_is_cancellable_even_with_prepared_thresholds():
    with pytest.raises(OperationCancelled):
        operations.colocalization_mask(
            [np.ones((2, 2)), np.ones((2, 2))],
            threshold_mode="Costes auto",
            _vipp_resolved_costes={"threshold_1": 1.0, "threshold_2": 1.0},
            progress=ProgressContext(cancelled=lambda: True),
        )


def _mask_pipeline():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    channel_2 = pipeline.add_node("input")
    mask = pipeline.add_node("colocalization_mask")
    assert pipeline.connect("input", mask.id, target_port=0).success
    assert pipeline.connect(channel_2.id, mask.id, target_port=1).success
    return pipeline, channel_2, mask


def test_mask_node_connects_to_cleanup_and_object_labeling():
    pipeline, channel_2, mask = _mask_pipeline()
    assert mask.category == COLOCALIZATION_CATEGORY
    assert mask.output_type == "mask"
    pipeline.set_param(mask.id, "channel_1_threshold", 100.0)
    pipeline.set_param(mask.id, "channel_2_threshold", 100.0)
    cleanup = pipeline.add_node("remove_small_objects")
    pipeline.set_param(cleanup.id, "min_size", 2)
    labels = pipeline.add_node("label_connected_components")
    measurements = pipeline.add_node("measure_objects")
    assert pipeline.connect(mask.id, cleanup.id).success
    assert pipeline.connect(cleanup.id, labels.id).success
    assert pipeline.connect(labels.id, measurements.id).success

    data_1 = np.zeros((8, 10), dtype=np.uint16)
    data_2 = np.zeros_like(data_1)
    data_1[1:3, 1:3] = data_2[1:3, 1:3] = 100
    data_1[4:6, 6:8] = data_2[4:6, 6:8] = 200
    data_1[7, 0] = data_2[7, 0] = 200
    data_1[0, 9] = 200  # Channel-1-only signal must not enter the mask.
    data_1.flags.writeable = data_2.flags.writeable = False
    outputs = pipeline.run(
        data_1,
        input_metadata={"axes": "YX"},
        source_payloads={channel_2.id: SourcePayload(data_2, {"axes": "YX"})},
    )

    assert outputs[mask.id].dtype == bool
    assert int(outputs[mask.id].sum()) == 9
    assert int(outputs[cleanup.id].sum()) == 8
    assert int(outputs[labels.id].max()) == 2
    assert outputs[measurements.id].row_count == 2
    assert [row["area_pixels"] for row in outputs[measurements.id].records()] == [4, 4]
    assert pipeline.output_states[mask.id].axis_order == "YX"


def test_mask_workflow_save_load_preserves_manual_thresholds_and_connections(tmp_path):
    pipeline, channel_2, mask = _mask_pipeline()
    pipeline.set_param(mask.id, "channel_1_threshold", 43970.51)
    pipeline.set_param(mask.id, "channel_2_threshold", 48073.03)
    saved = save_workflow(tmp_path / "colocalization-mask.json", pipeline)
    workflow = load_workflow(saved)
    restored = PrototypePipeline()
    restored.restore_graph(
        workflow["nodes"], workflow["connections"], workflow["output_tunnels"]
    )
    assert restored.nodes[mask.id].operation_id == "colocalization_mask"
    assert restored.nodes[mask.id].output_type == "mask"
    assert restored.nodes[mask.id].params["threshold_mode"] == "Manual"
    assert restored.nodes[mask.id].params["channel_1_threshold"] == 43970.51
    assert restored.nodes[mask.id].params["channel_2_threshold"] == 48073.03
    assert restored.connections == pipeline.connections
    data_1 = np.array([[43970, 43971], [43971, 65535]], dtype=np.uint16)
    data_2 = np.array([[65535, 48073], [48074, 65535]], dtype=np.uint16)
    result = restored.run(
        data_1,
        input_metadata={"axes": "YX"},
        source_payloads={channel_2.id: SourcePayload(data_2, {"axes": "YX"})},
    )[mask.id]
    np.testing.assert_array_equal(result, [[False, False], [True, True]])


def test_pipeline_costes_is_one_global_fit_shared_with_overlay(monkeypatch):
    pipeline, channel_2, mask = _mask_pipeline()
    overlay = pipeline.add_node("colocalized_voxels")
    for node in (mask, overlay):
        pipeline.set_param(node.id, "threshold_mode", "Costes auto")
    pipeline.set_param(overlay.id, "display_mode", "White on black")
    assert pipeline.connect("input", overlay.id, target_port=0).success
    assert pipeline.connect(channel_2.id, overlay.id, target_port=1).success
    data_1 = np.arange(120, dtype=np.uint16).reshape(2, 3, 4, 5)
    data_2 = np.flip(data_1, axis=-1).copy()
    calls = []

    def fit(first, second, *, progress=None):
        calls.append((first.shape, second.shape))
        return {"threshold_1": 42.0, "threshold_2": 55.0}

    monkeypatch.setattr(operations, "_costes_thresholds", fit)
    outputs = pipeline.run(
        data_1,
        input_metadata={"axes": "TZYX"},
        source_payloads={channel_2.id: SourcePayload(data_2, {"axes": "TZYX"})},
    )

    assert calls == [(data_1.shape, data_2.shape)]
    assert mask.params["channel_1_threshold"] == 42.0
    assert mask.params["channel_2_threshold"] == 55.0
    np.testing.assert_array_equal(outputs[mask.id], (data_1 >= 42) & (data_2 >= 55))
    np.testing.assert_array_equal(
        outputs[mask.id], np.all(outputs[overlay.id] == 1, axis=-1)
    )
    assert pipeline.output_states[mask.id].axis_order == "TZYX"
