from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import napari_vipp.core.object_filter_counts as counts_module
from napari_vipp.core.host_memory import HostMemorySnapshot
from napari_vipp.core.object_filter_counts import (
    ObjectFilterCounts,
    object_filter_counts,
)
from napari_vipp.core.progress import OperationCancelled


@pytest.fixture(autouse=True)
def stable_memory_headroom(monkeypatch):
    monkeypatch.setattr(
        counts_module,
        "capture_host_memory",
        lambda: HostMemorySnapshot(
            platform="linux",
            source="linux_proc_meminfo",
            physical_total_bytes=64 * 1024**3,
            physical_available_bytes=48 * 1024**3,
        ),
    )


def test_sparse_label_ids_are_counted_not_maximum():
    before = np.array([[0, 10, 10], [9999, 0, 50]], dtype=np.int32)
    after = np.where(before == 9999, 0, before)
    assert object_filter_counts(before, after, spatial_ndim=2) == ObjectFilterCounts(
        3, 2, 1, 1
    )


def test_same_ids_in_independent_blocks_are_distinct_objects():
    before = np.array([[[0, 10], [20, 20]], [[10, 10], [20, 0]]], dtype=np.int16)
    after = before.copy()
    after[0, after[0] == 10] = 0
    after[1] = 0
    assert object_filter_counts(before, after, spatial_ndim=2) == ObjectFilterCounts(
        4, 1, 3, 2
    )


def test_disconnected_pieces_of_same_integer_label_count_as_one_id():
    before = np.array([[7, 0, 7], [0, 0, 0]], dtype=np.int16)
    after = np.array([[0, 0, 7], [0, 0, 0]], dtype=np.int16)
    assert object_filter_counts(before, after, spatial_ndim=2) == ObjectFilterCounts(
        1, 1, 0, 1
    )


@pytest.mark.parametrize("ndim", [2, 3])
def test_boolean_connectivity_changes_object_definition(ndim):
    before = np.zeros((3,) * ndim, dtype=bool)
    before[(0,) * ndim] = True
    before[(1,) * ndim] = True
    assert object_filter_counts(
        before, before, spatial_ndim=ndim
    ) == ObjectFilterCounts(2, 2, 0, 1)
    assert object_filter_counts(
        before, before, spatial_ndim=ndim, connectivity="Full connectivity"
    ) == ObjectFilterCounts(1, 1, 0, 1)


def test_boolean_leading_blocks_are_not_connected_to_each_other():
    before = np.ones((2, 3, 4), dtype=bool)
    after = before.copy()
    after[0] = False
    assert object_filter_counts(before, after, spatial_ndim=2) == ObjectFilterCounts(
        2, 1, 1, 2
    )


def test_boolean_partial_removal_across_chunks_is_rejected(monkeypatch):
    monkeypatch.setattr(counts_module, "OBJECT_COUNT_CHUNK_ELEMENTS", 3)
    before = np.ones((2, 10), dtype=bool)
    after = before.copy()
    after[:, :5] = False
    with pytest.raises(ValueError, match="partially removes"):
        object_filter_counts(before, after, spatial_ndim=2)


@pytest.mark.parametrize("dtype", [bool, np.uint8])
def test_new_foreground_is_not_misreported_as_retained(dtype):
    before = np.zeros((3, 3), dtype=dtype)
    after = before.copy()
    after[1, 1] = 1
    with pytest.raises(ValueError, match="new foreground"):
        object_filter_counts(before, after, spatial_ndim=2)


def test_integer_relabelling_to_an_existing_id_is_rejected():
    before = np.array([[1, 2]], dtype=np.int32)
    after = np.array([[2, 2]], dtype=np.int32)
    with pytest.raises(ValueError, match="changed label IDs"):
        object_filter_counts(before, after, spatial_ndim=2)


def test_wide_uint64_ids_remain_exact_and_do_not_allocate_by_maximum():
    maximum = np.iinfo(np.uint64).max
    before = np.array([[0, maximum, maximum - 1, 2**53 + 1]], dtype=np.uint64)
    after = before.copy()
    after[0, 1] = 0
    assert object_filter_counts(before, after, spatial_ndim=2) == ObjectFilterCounts(
        3, 2, 1, 1
    )


def test_mixed_signed_unsigned_ids_do_not_compare_through_float64():
    before = np.array([[2**53, 2**53 + 1]], dtype=np.uint64)
    after = np.array([[2**53 + 1, 0]], dtype=np.int64)
    with pytest.raises(ValueError, match="changed label IDs"):
        object_filter_counts(before, after, spatial_ndim=2)
    after[0, 0] = 2**53
    assert object_filter_counts(before, after, spatial_ndim=2) == ObjectFilterCounts(
        2, 1, 1, 1
    )


@pytest.mark.parametrize("dtype", [np.int8, np.int64])
def test_negative_labels_fail_clearly(dtype):
    before = np.array([[0, -1]], dtype=dtype)
    with pytest.raises(ValueError, match="non-negative"):
        object_filter_counts(before, np.zeros_like(before), spatial_ndim=2)


@pytest.mark.parametrize("value", [0.0, 1.0, np.nan, np.inf, 1 + 2j])
def test_non_integer_data_is_not_silently_cast(value):
    data = np.full((2, 2), value)
    with pytest.raises(TypeError, match="Boolean or integer"):
        object_filter_counts(data, data, spatial_ndim=2)


def test_boolean_and_integer_pairing_is_not_silently_reinterpreted():
    with pytest.raises(TypeError, match="both be Boolean"):
        object_filter_counts(
            np.ones((2, 2), bool), np.ones((2, 2), np.uint8), spatial_ndim=2
        )


@pytest.mark.parametrize("spatial_ndim", [0, -1, 3, 1.5, "2", True, None])
def test_invalid_spatial_dimensionality_is_rejected(spatial_ndim):
    data = np.zeros((2, 2), np.uint8)
    with pytest.raises(ValueError, match="Spatial dimensionality"):
        object_filter_counts(data, data, spatial_ndim=spatial_ndim)


def test_shape_and_mask_connectivity_validation():
    with pytest.raises(ValueError, match="identical"):
        object_filter_counts(
            np.zeros((2, 2), bool), np.zeros((3, 2), bool), spatial_ndim=2
        )
    with pytest.raises(ValueError, match="Connectivity"):
        object_filter_counts(
            np.zeros((2, 2), bool),
            np.zeros((2, 2), bool),
            spatial_ndim=2,
            connectivity="guess",
        )


@pytest.mark.parametrize("dtype", [bool, np.uint64])
@pytest.mark.parametrize("shape,blocks", [((0, 3), 1), ((2, 0, 3), 2), ((0, 2, 3), 0)])
def test_empty_axes_preserve_logical_block_counts(dtype, shape, blocks):
    data = np.zeros(shape, dtype)
    assert object_filter_counts(data, data, spatial_ndim=2) == ObjectFilterCounts(
        0, 0, 0, blocks
    )


@pytest.mark.parametrize("dtype", [bool, np.uint64])
def test_readonly_noncontiguous_arrays_are_not_modified(dtype, monkeypatch):
    monkeypatch.setattr(counts_module, "OBJECT_COUNT_CHUNK_ELEMENTS", 3)
    before = np.array([[0, 1, 1, 0], [2, 0, 0, 3]], dtype=dtype).T[::-1]
    after = np.zeros_like(before)
    original_before, original_after = before.copy(), after.copy()
    before.flags.writeable = False
    after.flags.writeable = False
    counts = object_filter_counts(before, after, spatial_ndim=2)
    assert counts.kept_count == 0 and counts.input_count == counts.removed_count
    np.testing.assert_array_equal(before, original_before)
    np.testing.assert_array_equal(after, original_after)
    assert not before.flags.writeable and not after.flags.writeable


@pytest.mark.parametrize("dtype", [bool, np.uint64])
def test_cancellation_is_checked_inside_large_blocks(dtype, monkeypatch):
    monkeypatch.setattr(counts_module, "OBJECT_COUNT_CHUNK_ELEMENTS", 4)
    checks = 0

    def cancel():
        nonlocal checks
        checks += 1
        return checks >= 5

    before = np.ones((10, 10), dtype=dtype)
    with pytest.raises(OperationCancelled, match="counts cancelled"):
        object_filter_counts(before, before, spatial_ndim=2, cancel_callback=cancel)
    assert checks == 5


def test_low_memory_is_checked_before_boolean_component_labelling(monkeypatch):
    monkeypatch.setattr(
        counts_module,
        "capture_host_memory",
        lambda: HostMemorySnapshot.unavailable("windows", "test unavailable"),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("component labelling should not allocate")

    monkeypatch.setattr(counts_module.ndi, "label", forbidden)
    before = np.ones((10, 10), dtype=bool)
    with pytest.raises(MemoryError, match="exact object-filter counts"):
        object_filter_counts(before, before, spatial_ndim=2)


def test_result_is_immutable():
    result = ObjectFilterCounts(3, 2, 1, 1)
    with pytest.raises(FrozenInstanceError):
        result.kept_count = 3


def test_counts_match_existing_boolean_filter_operations():
    from napari_vipp.core.operations import clear_border_objects, remove_small_objects

    before = np.zeros((7, 7), dtype=bool)
    before[0, 0] = True
    before[1, 1] = True  # Connected to the border only under full connectivity.
    before[4:6, 3:5] = True
    small_output = remove_small_objects(before, min_size=2, resolved_spatial_ndim=2)
    assert object_filter_counts(
        before, small_output, spatial_ndim=2
    ) == ObjectFilterCounts(3, 1, 2, 1)
    border_output = clear_border_objects(before, resolved_spatial_ndim=2)
    assert object_filter_counts(
        before, border_output, spatial_ndim=2, connectivity="Full connectivity"
    ) == ObjectFilterCounts(2, 1, 1, 1)
