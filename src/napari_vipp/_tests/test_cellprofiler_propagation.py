from __future__ import annotations

import numpy as np
import pytest
from centrosome.propagate import propagate

import napari_vipp.core.cellprofiler_propagation as propagation
from napari_vipp.core.cellprofiler_propagation import cellprofiler_propagation
from napari_vipp.core.progress import OperationCancelled, ProgressContext


def _inputs(shape=(9, 12)):
    image = np.zeros(shape, dtype=np.float64)
    seeds = np.zeros(shape, dtype=np.uint32)
    if seeds.size:
        seeds.flat[0] = 7
        seeds.flat[-1] = 19
    return [image, seeds, np.ones(shape, dtype=bool)]


def test_flat_guidance_produces_expected_two_seed_partition():
    image, seeds, mask = _inputs((5, 12))
    seeds[:] = 0
    seeds[2, 2], seeds[2, 9] = 7, 19
    expected = np.full(image.shape, 19, dtype=np.int32)
    expected[:, :6] = 7

    actual = cellprofiler_propagation([image, seeds, mask], regularization=1.0)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("weight", (0.0, 0.05, 1.0, 10.0))
@pytest.mark.parametrize("shape", ((1, 1), (1, 13), (11, 1), (9, 12)))
def test_borders_ties_and_irregular_masks_match_centrosome(weight, shape):
    rng = np.random.default_rng(72)
    image, seeds, _mask = _inputs(shape)
    image[:] = rng.integers(0, 5, size=shape) / 4
    mask = rng.random(shape) > 0.2

    actual = cellprofiler_propagation([image, seeds, mask], regularization=weight)
    expected, _ = propagate(image, seeds.astype(np.int32), mask, weight)

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.int32


@pytest.mark.parametrize("weight", (0.0, 0.05, 1.0))
def test_equal_cost_ties_retain_reference_resolution(weight):
    image, seeds, mask = _inputs((7, 7))
    seeds[:] = 0
    seeds[0, 3], seeds[3, 0], seeds[3, 6], seeds[6, 3] = 31, 2, 19, 7

    actual = cellprofiler_propagation([image, seeds, mask], regularization=weight)
    expected, _ = propagate(image, seeds.astype(np.int32), mask, weight)

    np.testing.assert_array_equal(actual, expected)


def test_seed_outside_mask_is_retained_but_does_not_start_growth():
    image, seeds, mask = _inputs((5, 9))
    seeds[:] = 0
    seeds[2, 0], seeds[2, 8] = 7, 19
    mask[:, 0] = False
    mask[:, 4] = False
    expected = np.zeros(image.shape, dtype=np.int32)
    expected[2, 0] = 7
    expected[:, 5:] = 19

    actual = cellprofiler_propagation([image, seeds, mask])

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("no_seeds", (False, True))
@pytest.mark.parametrize("no_mask", (False, True))
def test_missing_seeds_and_all_false_mask_match_reference(no_seeds, no_mask):
    image, seeds, mask = _inputs()
    if no_seeds:
        seeds[:] = 0
    if no_mask:
        mask[:] = False

    actual = cellprofiler_propagation([image, seeds, mask])
    expected, _ = propagate(image, seeds.astype(np.int32), mask, 0.05)

    np.testing.assert_array_equal(actual, expected)
    if no_mask:
        np.testing.assert_array_equal(actual, seeds)


@pytest.mark.parametrize(
    "dtype",
    (
        bool,
        np.int8,
        np.uint16,
        np.int64,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        ">u2",
        ">f8",
    ),
)
def test_readonly_noncontiguous_inputs_and_dtype_conversion_are_exact(dtype):
    image = np.arange(48).reshape(6, 8).astype(dtype)
    seeds = np.zeros(image.shape, dtype=">i4")
    seeds[0, 0], seeds[4, 6] = 7, np.iinfo(np.int32).max
    mask = np.ones(image.shape, dtype=bool)
    originals = [array.copy() for array in (image, seeds, mask)]
    for array in (image, seeds, mask):
        array.flags.writeable = False
    views = [array[::-2, ::2] for array in (image, seeds, mask)]

    actual = cellprofiler_propagation(views)
    expected, _ = propagate(
        views[0].astype(np.float64), views[1].astype(np.int32), views[2], 0.05
    )

    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.c_contiguous
    assert actual.dtype == np.int32
    for original, array in zip(originals, (image, seeds, mask), strict=True):
        np.testing.assert_array_equal(array, original)
        assert not array.flags.writeable
        assert not np.shares_memory(actual, array)


@pytest.mark.parametrize(
    "dtype,value", ((np.int64, -(2**63)), (np.int64, 2**60), (np.uint64, 2**63))
)
def test_exactly_representable_wide_integer_guidance_is_supported(dtype, value):
    image, seeds, mask = _inputs()
    image = np.full(image.shape, value, dtype=dtype)

    actual = cellprofiler_propagation([image, seeds, mask])
    expected, _ = propagate(image.astype(np.float64), seeds, mask, 0.05)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "dtype,value",
    ((np.int64, 2**53 + 1), (np.int64, 2**63 - 1), (np.uint64, 2**64 - 1)),
)
def test_inexact_wide_integer_guidance_is_rejected(dtype, value):
    image, seeds, mask = _inputs()
    image = np.full(image.shape, value, dtype=dtype)

    with pytest.raises(ValueError, match="represented exactly in float64"):
        cellprofiler_propagation([image, seeds, mask])


@pytest.mark.parametrize("shape", ((0, 5), (5, 0), (0, 0)))
def test_empty_yx_arrays_skip_backend_and_return_detached_labels(monkeypatch, shape):
    def unexpected_call(*args):
        pytest.fail("Empty arrays must not enter the compiled reference kernel.")

    monkeypatch.setattr(propagation, "_propagate", unexpected_call)
    arrays = _inputs(shape)

    actual = cellprofiler_propagation(arrays)

    assert actual.shape == shape
    assert actual.dtype == np.int32
    assert not np.shares_memory(actual, arrays[1])


@pytest.mark.parametrize("inputs", (None, [], [np.zeros((2, 2))]))
def test_requires_three_inputs(inputs):
    with pytest.raises(ValueError, match="requires"):
        cellprofiler_propagation(inputs)


@pytest.mark.parametrize("shape", ((5,), (2, 3, 4), (2, 1, 3, 4)))
def test_non_2d_inputs_rejected(shape):
    with pytest.raises(ValueError, match="one 2D YX image only"):
        cellprofiler_propagation(_inputs(shape))


@pytest.mark.parametrize("index", (0, 1, 2))
def test_different_input_shape_rejected(index):
    arrays = _inputs()
    arrays[index] = arrays[index][:-1]
    with pytest.raises(ValueError, match="matching"):
        cellprofiler_propagation(arrays)


def test_axis_lengths_exceeding_backend_coordinates_rejected_without_allocation():
    arrays = [
        np.broadcast_to(array, (1, 2**31))
        for array in (
            np.zeros((1, 1)),
            np.zeros((1, 1), dtype=int),
            np.ones((1, 1), dtype=bool),
        )
    ]
    with pytest.raises(ValueError, match="int32 backend coordinates"):
        cellprofiler_propagation(arrays)


@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf))
def test_nonfinite_guidance_rejected_even_outside_mask(value):
    arrays = _inputs()
    arrays[0][3, 3] = value
    arrays[2][3, 3] = False
    with pytest.raises(ValueError, match="finite intensities"):
        cellprofiler_propagation(arrays)


@pytest.mark.parametrize("dtype", (complex, object, "U2"))
def test_invalid_guidance_dtype_rejected(dtype):
    arrays = _inputs()
    arrays[0] = arrays[0].astype(dtype)
    with pytest.raises(ValueError, match="real numeric"):
        cellprofiler_propagation(arrays)


@pytest.mark.parametrize("dtype", (bool, np.float32, complex, object))
def test_invalid_seed_dtype_rejected(dtype):
    arrays = _inputs()
    arrays[1] = arrays[1].astype(dtype)
    with pytest.raises(ValueError, match="integer label IDs"):
        cellprofiler_propagation(arrays)


@pytest.mark.parametrize("value", (-1, 2**31, 2**63 - 1))
def test_unsafe_seed_identifiers_rejected(value):
    arrays = _inputs()
    arrays[1] = arrays[1].astype(np.int64)
    arrays[1][3, 3] = value
    with pytest.raises(ValueError, match="must lie between"):
        cellprofiler_propagation(arrays)


@pytest.mark.parametrize("dtype", (np.uint8, np.float32, complex, object))
def test_nonboolean_masks_rejected_including_zero_one_data(dtype):
    arrays = _inputs()
    arrays[2] = arrays[2].astype(dtype)
    with pytest.raises(ValueError, match="Boolean dtype"):
        cellprofiler_propagation(arrays)


@pytest.mark.parametrize(
    "weight",
    (
        -1,
        np.nan,
        np.inf,
        -np.inf,
        True,
        np.bool_(0),
        "0.05",
        None,
        complex(0.05),
        10**400,
    ),
)
def test_invalid_regularization_rejected(weight):
    with pytest.raises(ValueError, match="finite nonnegative number"):
        cellprofiler_propagation(_inputs(), regularization=weight)


@pytest.mark.parametrize("unsafe_image", (False, True))
def test_overflow_prone_finite_costs_rejected_before_backend(monkeypatch, unsafe_image):
    def unexpected_call(*args):
        pytest.fail("Dangerous numerical ranges must not reach Centrosome.")

    monkeypatch.setattr(propagation, "_propagate", unexpected_call)
    arrays = _inputs()
    weight = 0.05 if unsafe_image else 1e308
    if unsafe_image:
        arrays[0][0, 0] = -1e308
        arrays[0][-1, -1] = 1e308
    with pytest.raises(ValueError, match="overflow"):
        cellprofiler_propagation(arrays, regularization=weight)


def test_guidance_and_regularization_reach_backend_without_normalization(monkeypatch):
    arrays = _inputs()
    arrays[0][:] = np.arange(arrays[0].size).reshape(arrays[0].shape) * 128 - 90
    actual_backend = propagation._propagate

    def inspect_buffers(image, seeds, mask, weight):
        np.testing.assert_array_equal(image, arrays[0])
        assert image.dtype == np.float64
        assert seeds.dtype == np.int32
        assert mask.dtype == bool
        assert weight == 0.125
        for buffer, original in zip((image, seeds, mask), arrays, strict=True):
            assert buffer.flags.c_contiguous
            assert not np.shares_memory(buffer, original)
        return actual_backend(image, seeds, mask, weight)

    monkeypatch.setattr(propagation, "_propagate", inspect_buffers)
    cellprofiler_propagation(arrays, regularization=0.125)


def test_progress_reports_one_block_with_reference_work_boundary():
    updates = []
    cellprofiler_propagation(
        _inputs(), progress=ProgressContext(reporter=updates.append)
    )
    assert [(update.current, update.total) for update in updates] == [(0, 1), (1, 1)]


def test_cancellation_before_backend_avoids_work(monkeypatch):
    def unexpected_call(*args):
        pytest.fail("A cancelled operation must not call the backend.")

    monkeypatch.setattr(propagation, "_propagate", unexpected_call)
    with pytest.raises(OperationCancelled):
        cellprofiler_propagation(
            _inputs(), progress=ProgressContext(cancelled=lambda: True)
        )


def test_cancellation_during_blocking_backend_discards_result(monkeypatch):
    cancelled = False
    updates = []
    actual_backend = propagation._propagate

    def finish_after_cancellation(*args):
        nonlocal cancelled
        result = actual_backend(*args)
        cancelled = True
        return result

    monkeypatch.setattr(propagation, "_propagate", finish_after_cancellation)
    with pytest.raises(OperationCancelled):
        cellprofiler_propagation(
            _inputs(),
            progress=ProgressContext(
                cancelled=lambda: cancelled, reporter=updates.append
            ),
        )
    assert [(update.current, update.total) for update in updates] == [(0, 1)]
