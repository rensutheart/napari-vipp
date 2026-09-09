from __future__ import annotations

import numpy as np
import pytest
from skimage.segmentation import find_boundaries

import napari_vipp.core.label_boundaries as label_boundaries
from napari_vipp.core.operations import find_label_boundaries
from napari_vipp.core.progress import OperationCancelled, ProgressContext

PLACEMENTS = (
    ("Inside objects", "inner"),
    ("Outside objects", "outer"),
    ("Both sides", "thick"),
)
CONNECTIVITIES = ("Face connected", "Full connectivity")


@pytest.mark.parametrize(("placement", "mode"), PLACEMENTS)
@pytest.mark.parametrize("connectivity", CONNECTIVITIES)
@pytest.mark.parametrize("ndim", (2, 3))
@pytest.mark.parametrize("dtype", (bool, np.uint8, np.uint16, np.uint32, np.int32))
def test_2d_3d_modes_and_connectivity_match_reference(
    placement, mode, connectivity, ndim, dtype
):
    rng = np.random.default_rng(1984)
    labels = rng.integers(0, 4, size=(5,) * ndim).astype(dtype)
    rank = 1 if connectivity == "Face connected" else ndim

    result = find_label_boundaries(
        labels,
        boundary_placement=placement,
        connectivity=connectivity,
        resolved_spatial_ndim=ndim,
    )

    np.testing.assert_array_equal(
        result, find_boundaries(labels, connectivity=rank, mode=mode, background=0)
    )
    assert result.dtype == np.bool_
    assert result.shape == labels.shape
    assert not np.shares_memory(labels, result)


@pytest.mark.parametrize(("placement", "mode"), PLACEMENTS)
def test_touching_positive_labels_retain_their_interface(placement, mode):
    labels = np.array([[5, 5, 9, 9]] * 4, dtype=np.uint16)
    expected = np.array([[False, True, True, False]] * 4)

    result = find_label_boundaries(labels, boundary_placement=placement)
    binary = find_label_boundaries(labels != 0, boundary_placement=placement)

    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(result, find_boundaries(labels, mode=mode))
    assert not binary.any()


@pytest.mark.parametrize("ndim", (2, 3))
def test_face_and_full_connectivity_differ_at_diagonal_neighbors(ndim):
    labels = np.ones((3,) * ndim, dtype=np.uint8)
    labels[(0,) * ndim] = 0
    face = find_label_boundaries(labels, resolved_spatial_ndim=ndim)
    full = find_label_boundaries(
        labels, connectivity="Full connectivity", resolved_spatial_ndim=ndim
    )

    assert not face[(1,) * ndim]
    assert full[(1,) * ndim]


@pytest.mark.parametrize(("placement", "mode"), PLACEMENTS)
@pytest.mark.parametrize("connectivity", CONNECTIVITIES)
@pytest.mark.parametrize("ndim", (2, 3))
@pytest.mark.parametrize("dtype", (np.int64, np.uint64))
@pytest.mark.parametrize("with_background", (False, True))
def test_wide_integer_ids_match_exact_small_id_equivalent(
    placement, mode, connectivity, ndim, dtype, with_background
):
    # Neighboring IDs above 2**53 can alias through SciPy's morphology double
    # intermediates. Test equality against a known bijection, not that unsafe
    # wide-ID backend call. Include the exact dtype maximum as a valid label.
    small = np.array([[1, 2, 2, 3], [1, 2, 3, 3], [3, 3, 1, 1]], dtype=np.uint32)
    if with_background:
        small[0, 0] = 0
    if ndim == 3:
        small = np.stack((small, np.flip(small, axis=0), small))
    identifiers = np.array([0, 2**53, 2**53 + 1, np.iinfo(dtype).max], dtype=dtype)
    labels = identifiers[small]
    before = labels.copy()
    labels.flags.writeable = False
    rank = 1 if connectivity == "Face connected" else ndim

    result = find_label_boundaries(
        labels,
        boundary_placement=placement,
        connectivity=connectivity,
        resolved_spatial_ndim=ndim,
    )

    np.testing.assert_array_equal(
        result, find_boundaries(small, connectivity=rank, mode=mode, background=0)
    )
    np.testing.assert_array_equal(labels, before)
    assert not labels.flags.writeable


@pytest.mark.parametrize(("placement", "mode"), PLACEMENTS)
@pytest.mark.parametrize("value", (0, 1, 2**64 - 1))
@pytest.mark.parametrize("shape", ((1, 1), (1, 1, 1), (4, 5), (2, 3, 4)))
def test_constant_blocks_have_no_invented_exterior_boundary(
    placement, mode, value, shape
):
    labels = np.full(shape, value, dtype=np.uint64)
    result = find_label_boundaries(
        labels,
        boundary_placement=placement,
        resolved_spatial_ndim=len(shape),
    )

    assert result.shape == shape
    assert result.dtype == bool
    assert not result.any()


@pytest.mark.parametrize(("placement", "mode"), PLACEMENTS)
@pytest.mark.parametrize("shape", ((1, 5), (1, 1, 5), (1, 5, 1)))
def test_singleton_axes_preserve_observed_transitions(placement, mode, shape):
    labels = np.array([0, 1, 1, 2, 2], dtype=np.uint16).reshape(shape)
    result = find_label_boundaries(
        labels, boundary_placement=placement, resolved_spatial_ndim=len(shape)
    )

    np.testing.assert_array_equal(result, find_boundaries(labels, mode=mode))


@pytest.mark.parametrize("dtype", (bool, np.int16, np.uint64, ">u4"))
def test_readonly_noncontiguous_inputs_are_unchanged(dtype):
    original = np.arange(80).reshape(8, 10).astype(dtype)
    original.flags.writeable = False
    labels = original[::-2, ::2]
    before = labels.copy()

    result = find_label_boundaries(labels)

    np.testing.assert_array_equal(labels, before)
    assert not labels.flags.writeable
    assert not np.shares_memory(result, labels)
    assert result.flags.c_contiguous
    assert result.dtype == np.bool_


@pytest.mark.parametrize("spatial_ndim", (2, 3))
def test_leading_channel_and_time_blocks_are_independent(spatial_ndim):
    shape = (2, 3) + (4,) * spatial_ndim
    labels = np.zeros(shape, dtype=np.uint16)
    labels[0, 0] = 1  # No boundaries against a different leading C/T block.
    labels[0, 1][(1,) * spatial_ndim] = 3
    labels[1, 2][(slice(1, 3),) * spatial_ndim] = 9
    result = find_label_boundaries(
        labels, boundary_placement="Outside objects", resolved_spatial_ndim=spatial_ndim
    )

    for index in np.ndindex(shape[:2]):
        np.testing.assert_array_equal(
            result[index], find_boundaries(labels[index], mode="outer")
        )
    assert not result[0, 0].any()


@pytest.mark.parametrize("shape", ((0, 5), (5, 0), (0, 3, 4), (2, 0, 5), (2, 0, 3, 4)))
def test_empty_inputs_have_same_shape_without_backend_calls(monkeypatch, shape):
    def unexpected_call(*args, **kwargs):
        pytest.fail("Empty spatial blocks must not call scikit-image.")

    monkeypatch.setattr(
        label_boundaries.segmentation, "find_boundaries", unexpected_call
    )
    result = find_label_boundaries(
        np.zeros(shape, dtype=np.int64), spatial_mode="2D YX"
    )

    assert result.shape == shape
    assert result.dtype == bool
    assert result.size == 0


@pytest.mark.parametrize(
    "data",
    (
        np.array([[0.0, 1.0]]),
        np.array([[np.nan, 1.0]]),
        np.array([[np.inf, 1.0]]),
        np.array([[0, 1]], dtype=np.complex64),
        np.array([[0, 1]], dtype=object),
        np.array([["0", "1"]]),
        np.empty((0, 4), dtype=np.float32),
    ),
)
def test_invalid_dtypes_are_rejected_without_coercion(data):
    with pytest.raises(ValueError, match="Boolean mask or nonnegative integer"):
        find_label_boundaries(data)


@pytest.mark.parametrize("dtype", (np.int8, np.int16, np.int32, np.int64))
def test_negative_labels_are_rejected(dtype):
    labels = np.array([[0, 1], [-1, 2]], dtype=dtype)
    with pytest.raises(ValueError, match="nonnegative"):
        find_label_boundaries(labels)


@pytest.mark.parametrize("placement", ("inner", "subpixel", "", None, 1))
def test_invalid_boundary_placements_are_rejected(placement):
    with pytest.raises(ValueError, match="Boundary placement must be"):
        find_label_boundaries(np.ones((3, 3), dtype=bool), boundary_placement=placement)


@pytest.mark.parametrize("connectivity", ("face", "Full", "", None, 1))
def test_invalid_connectivity_is_rejected(connectivity):
    with pytest.raises(ValueError, match="Connectivity must be"):
        find_label_boundaries(np.ones((3, 3), dtype=bool), connectivity=connectivity)


@pytest.mark.parametrize("shape", ((), (0,), (5,)))
def test_arrays_below_2d_are_rejected(shape):
    with pytest.raises(ValueError, match="at least a 2D"):
        find_label_boundaries(np.zeros(shape, dtype=bool))


@pytest.mark.parametrize("resolved", (0, 4, 2.5, True, "2"))
def test_invalid_resolved_spatial_rank_is_rejected(resolved):
    with pytest.raises(ValueError, match="resolved_spatial_ndim"):
        find_label_boundaries(
            np.ones((3, 3, 3), dtype=bool), resolved_spatial_ndim=resolved
        )


def test_spatial_interpretations_fail_visibly():
    labels = np.ones((3, 3, 3), dtype=bool)
    with pytest.raises(ValueError, match="Spatial mode must be"):
        find_label_boundaries(labels, spatial_mode="guess")
    with pytest.raises(ValueError, match="Auto from axes requires explicit"):
        find_label_boundaries(labels)
    with pytest.raises(ValueError, match="requires 2D or 3D spatial"):
        find_label_boundaries(labels, resolved_spatial_ndim=1)
    with pytest.raises(ValueError, match="3D spatial processing cannot be applied"):
        find_label_boundaries(labels[0], spatial_mode="3D ZYX")


def test_progress_reports_each_complete_leading_block():
    updates = []
    result = find_label_boundaries(
        np.zeros((2, 3, 3), dtype=np.uint16),
        spatial_mode="2D YX",
        progress=ProgressContext(reporter=updates.append),
    )

    assert result.shape == (2, 3, 3)
    assert [(update.current, update.total) for update in updates] == [
        (0, 2),
        (1, 2),
        (2, 2),
    ]
    assert {update.message for update in updates} == {"Label-boundary blocks"}


def test_zero_leading_blocks_report_completion():
    updates = []
    find_label_boundaries(
        np.zeros((0, 3, 3), dtype=bool),
        spatial_mode="2D YX",
        progress=ProgressContext(reporter=updates.append),
    )
    assert [(update.current, update.total) for update in updates] == [(0, 1), (1, 1)]


def test_preexisting_cancellation_prevents_backend_call(monkeypatch):
    def unexpected_call(*args, **kwargs):
        pytest.fail("Cancelled work must not call scikit-image.")

    monkeypatch.setattr(
        label_boundaries.segmentation, "find_boundaries", unexpected_call
    )
    with pytest.raises(OperationCancelled):
        find_label_boundaries(
            np.ones((3, 3), dtype=bool),
            progress=ProgressContext(cancelled=lambda: True),
        )


def test_cancellation_during_final_backend_call_is_observed(monkeypatch):
    cancelled = False

    def cancelling_backend(*args, **kwargs):
        nonlocal cancelled
        result = find_boundaries(*args, **kwargs)
        cancelled = True
        return result

    monkeypatch.setattr(
        label_boundaries.segmentation, "find_boundaries", cancelling_backend
    )
    with pytest.raises(OperationCancelled):
        find_label_boundaries(
            np.eye(5, dtype=bool), progress=ProgressContext(cancelled=lambda: cancelled)
        )


@pytest.mark.parametrize("block_count", (1, 3))
def test_cancellation_after_block_report_never_returns_success(
    monkeypatch, block_count
):
    cancelled = False
    calls = 0

    def counted_backend(*args, **kwargs):
        nonlocal calls
        calls += 1
        return find_boundaries(*args, **kwargs)

    def record(update):
        nonlocal cancelled
        if update.current == 1:
            cancelled = True

    monkeypatch.setattr(
        label_boundaries.segmentation, "find_boundaries", counted_backend
    )
    with pytest.raises(OperationCancelled):
        find_label_boundaries(
            np.ones((block_count, 3, 3), dtype=bool),
            spatial_mode="2D YX",
            progress=ProgressContext(cancelled=lambda: cancelled, reporter=record),
        )
    assert calls == 1


def test_cancellation_after_empty_batch_report_is_observed():
    cancelled = False

    def record(update):
        nonlocal cancelled
        cancelled = update.current == 1

    with pytest.raises(OperationCancelled):
        find_label_boundaries(
            np.empty((0, 3, 3), dtype=bool),
            spatial_mode="2D YX",
            progress=ProgressContext(cancelled=lambda: cancelled, reporter=record),
        )
