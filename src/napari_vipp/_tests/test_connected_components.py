from __future__ import annotations

import numpy as np
import pytest

import napari_vipp.core.connected_components as connected_components
from napari_vipp.core.metadata import AmbiguousAxisError
from napari_vipp.core.operations import label_connected_components
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.progress import OperationCancelled, ProgressContext

ORDERING_MASK = np.array(
    [
        [0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [1, 0, 0, 0, 1],
    ],
    dtype=bool,
)


@pytest.mark.parametrize(
    ("connectivity", "expected"),
    (
        (
            "Face connected",
            np.array(
                [
                    [0, 0, 1, 0, 2],
                    [3, 0, 0, 0, 0],
                    [0, 0, 4, 4, 0],
                    [5, 0, 0, 0, 6],
                ],
                dtype=np.int32,
            ),
        ),
        (
            "Full connectivity",
            np.array(
                [
                    [0, 0, 1, 0, 2],
                    [3, 0, 0, 0, 0],
                    [0, 0, 4, 4, 0],
                    [5, 0, 0, 0, 4],
                ],
                dtype=np.int32,
            ),
        ),
    ),
)
def test_exact_2d_scipy_numbering_is_frozen(connectivity, expected):
    labels = label_connected_components(
        ORDERING_MASK,
        spatial_mode="2D YX",
        connectivity=connectivity,
    )

    np.testing.assert_array_equal(labels, expected)
    assert labels.dtype == np.int32
    assert labels.flags.c_contiguous


def test_equivalent_provisional_components_are_compacted_in_scan_order():
    mask = np.array(
        [
            [1, 0, 1, 0, 1],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1],
            [1, 1, 0, 1, 1],
        ],
        dtype=bool,
    )

    labels = label_connected_components(
        mask,
        spatial_mode="2D YX",
        connectivity="Face connected",
    )

    np.testing.assert_array_equal(
        labels,
        np.array(
            [
                [1, 0, 1, 0, 2],
                [1, 1, 1, 0, 0],
                [0, 0, 0, 0, 3],
                [4, 4, 0, 3, 3],
            ],
            dtype=np.int32,
        ),
    )


def test_exact_3d_face_and_full_connectivity_are_frozen():
    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[0, 0, 0] = True
    mask[1, 1, 1] = True
    mask[2, 0, 2] = True

    face = label_connected_components(
        mask,
        spatial_mode="3D ZYX",
        connectivity="Face connected",
    )
    full = label_connected_components(
        mask,
        spatial_mode="3D ZYX",
        connectivity="Full connectivity",
    )

    expected_face = np.zeros(mask.shape, dtype=np.int32)
    expected_face[0, 0, 0] = 1
    expected_face[1, 1, 1] = 2
    expected_face[2, 0, 2] = 3
    expected_full = mask.astype(np.int32)
    np.testing.assert_array_equal(face, expected_face)
    np.testing.assert_array_equal(full, expected_full)


def test_leading_blocks_are_independent_and_restart_ids_at_one():
    mask = np.zeros((2, 3, 4), dtype=bool)
    mask[0, 0, 0] = True
    mask[0, 2, 3] = True
    mask[1, 1, 1:3] = True

    labels = label_connected_components(
        mask,
        spatial_mode="2D YX",
        connectivity="Full connectivity",
    )

    expected = np.zeros(mask.shape, dtype=np.int32)
    expected[0, 0, 0] = 1
    expected[0, 2, 3] = 2
    expected[1, 1, 1:3] = 1
    np.testing.assert_array_equal(labels, expected)


@pytest.mark.parametrize(
    "dtype",
    (
        np.bool_,
        np.uint8,
        np.uint16,
        np.int16,
        np.int64,
        np.float32,
        np.float64,
        np.complex64,
    ),
)
def test_numeric_inputs_use_exact_nonzero_foreground_semantics(dtype):
    values = np.array([[0, 2, 0], [3, 0, 4]], dtype=dtype)
    expected = label_connected_components(
        values != 0,
        spatial_mode="2D YX",
        connectivity="Face connected",
    )

    actual = label_connected_components(
        values,
        spatial_mode="2D YX",
        connectivity="Face connected",
    )

    np.testing.assert_array_equal(actual, expected)


def test_nan_and_infinities_are_foreground_while_signed_zero_is_background():
    values = np.array(
        [[0.0, -0.0, np.nan, np.inf, -np.inf], [-2.0, 3.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    labels = label_connected_components(
        values,
        spatial_mode="2D YX",
        connectivity="Face connected",
    )

    np.testing.assert_array_equal(
        labels,
        np.array([[0, 0, 1, 1, 1], [2, 2, 0, 0, 1]], dtype=np.int32),
    )


def test_non_native_input_and_noncontiguous_readonly_layout_preserve_contract():
    logical = ORDERING_MASK.astype(">i2")
    storage = logical[::-1, ::-1].copy()
    view = storage[::-1, ::-1]
    view.setflags(write=False)
    snapshot = view.tobytes(order="A")

    labels = label_connected_components(
        view,
        spatial_mode="2D YX",
        connectivity="Face connected",
    )

    np.testing.assert_array_equal(
        labels,
        label_connected_components(
            ORDERING_MASK,
            spatial_mode="2D YX",
            connectivity="Face connected",
        ),
    )
    assert view.tobytes(order="A") == snapshot
    assert not view.flags.writeable
    assert labels.dtype == np.dtype(np.int32)
    assert labels.dtype.isnative
    assert labels.flags.c_contiguous
    assert not np.shares_memory(labels, view)


@pytest.mark.parametrize("shape", ((0,), (0, 5), (2, 0, 5), (0, 5, 5)))
def test_empty_inputs_return_same_shape_native_int32(shape):
    spatial_ndim = min(max(len(shape), 1), 2)
    labels = label_connected_components(
        np.zeros(shape, dtype=bool),
        resolved_spatial_ndim=spatial_ndim,
    )

    assert labels.shape == shape
    assert labels.dtype == np.int32
    assert labels.dtype.isnative


@pytest.mark.parametrize("resolved", (0, 4, 2.5, True, "2"))
def test_auto_mode_rejects_invalid_resolved_spatial_rank(resolved):
    with pytest.raises(ValueError, match="resolved_spatial_ndim"):
        label_connected_components(
            np.zeros((3, 5, 5), dtype=bool),
            resolved_spatial_ndim=resolved,
        )


def test_direct_spatial_errors_are_stable():
    volume = np.zeros((3, 5, 5), dtype=bool)

    with pytest.raises(ValueError, match="Spatial mode must be"):
        label_connected_components(volume, spatial_mode="guess")
    with pytest.raises(ValueError, match="Auto from axes requires explicit"):
        label_connected_components(volume)
    with pytest.raises(ValueError, match="3D spatial processing cannot be applied"):
        label_connected_components(np.zeros((5, 5), dtype=bool), spatial_mode="3D ZYX")


def test_scipy_is_required_to_produce_int32_and_overflow_errors_propagate(monkeypatch):
    def overflowing_label(data, *, structure, output):
        assert np.dtype(output) == np.dtype(np.int32)
        raise RuntimeError("insufficient bit-depth in requested output type")

    monkeypatch.setattr(connected_components.ndi, "label", overflowing_label)

    with pytest.raises(RuntimeError, match="insufficient bit-depth"):
        connected_components.label_connected_components(
            np.ones((3, 3), dtype=bool),
            spatial_mode="2D YX",
        )


def test_progress_reports_each_completed_leading_block():
    updates = []
    mask = np.zeros((2, 3, 3), dtype=bool)
    mask[:, 1, 1] = True

    label_connected_components(
        mask,
        spatial_mode="2D YX",
        progress=ProgressContext(reporter=updates.append),
    )

    assert [(update.current, update.total) for update in updates] == [
        (0, 2),
        (1, 2),
        (2, 2),
    ]
    assert {update.message for update in updates} == {"Connected-component blocks"}


def test_zero_leading_block_progress_completes_immediately():
    updates = []

    labels = label_connected_components(
        np.zeros((0, 3, 3), dtype=bool),
        spatial_mode="2D YX",
        progress=ProgressContext(reporter=updates.append),
    )

    assert labels.shape == (0, 3, 3)
    assert [(update.current, update.total) for update in updates] == [(0, 1), (1, 1)]


def test_cancellation_after_one_block_prevents_later_scipy_calls(monkeypatch):
    original_label = connected_components.ndi.label
    calls = 0
    cancelled = False

    def counted_label(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_label(*args, **kwargs)

    def record(update):
        nonlocal cancelled
        if update.current == 1:
            cancelled = True

    monkeypatch.setattr(connected_components.ndi, "label", counted_label)
    progress = ProgressContext(cancelled=lambda: cancelled, reporter=record)

    with pytest.raises(OperationCancelled, match="Operation cancelled"):
        label_connected_components(
            np.ones((3, 4, 4), dtype=bool),
            spatial_mode="2D YX",
            progress=progress,
        )

    assert calls == 1


def _label_pipeline() -> tuple[PrototypePipeline, str]:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    node = pipeline.add_node("label_connected_components")
    pipeline.set_param(threshold.id, "threshold", 0.5)
    assert pipeline.connect("input", threshold.id).success
    assert pipeline.connect(threshold.id, node.id).success
    return pipeline, node.id


def test_pipeline_auto_injects_zyx_rank_and_preserves_label_metadata():
    mask = np.zeros((2, 3, 5, 5), dtype=bool)
    mask[:, :, 1, 1] = True
    pipeline, node_id = _label_pipeline()
    pipeline.set_param(node_id, "resolved_spatial_ndim", 1)

    outputs = pipeline.run(
        mask,
        input_metadata={"axes": "TZYX", "scale": (2.0, 1.5, 0.4, 0.4)},
        input_name="mask stack",
    )

    state = pipeline.output_states[node_id]
    assert pipeline.nodes[node_id].params["resolved_spatial_ndim"] == 3
    assert outputs[node_id].dtype == np.int32
    assert state is not None
    assert state.kind == "label image"
    assert state.axis_order == "TZYX"
    assert state.shape == mask.shape
    assert np.dtype(state.dtype) == np.dtype(np.int32)


def test_pipeline_2d_mode_restarts_labels_per_z_slice():
    mask = np.zeros((2, 4, 4), dtype=bool)
    mask[0, 0, 0] = True
    mask[:, 3, 3] = True
    pipeline, node_id = _label_pipeline()
    pipeline.set_param(node_id, "spatial_mode", "2D YX")

    labels = pipeline.run(mask, input_metadata={"axes": "ZYX"})[node_id]

    assert labels[0, 0, 0] == 1
    assert labels[0, 3, 3] == 2
    assert labels[1, 3, 3] == 1
    assert pipeline.nodes[node_id].params["resolved_spatial_ndim"] == 2


def test_pipeline_auto_rejects_ambiguous_or_noncanonical_axes():
    ambiguous, _ambiguous_id = _label_pipeline()
    with pytest.raises(AmbiguousAxisError, match="Auto from axes"):
        ambiguous.run(np.zeros((3, 5, 5), dtype=bool))

    noncanonical, _noncanonical_id = _label_pipeline()
    with pytest.raises(AmbiguousAxisError, match="positional ZYX processing"):
        noncanonical.run(
            np.zeros((2, 4, 3, 5), dtype=bool),
            input_metadata={"axes": "CYZX"},
        )


def test_pipeline_label_node_rejects_an_upstream_label_image():
    pipeline = PrototypePipeline()
    first = pipeline.add_node("label_connected_components")
    second = pipeline.add_node("label_connected_components")

    result = pipeline.connect(first.id, second.id)

    assert not result.success
    assert "labels output" in result.message
    assert "mask input" in result.message
