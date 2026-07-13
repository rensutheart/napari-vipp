from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from napari_vipp.core.diagnostics import (
    DISPLAY_HISTOGRAM_BINS,
    exact_finite_percentiles,
    exact_finite_stats,
    exact_generated_layer_contrast_limits,
    exact_histogram,
    label_volumes,
    largest_label_volume,
    largest_object_size,
    provisional_generated_layer_contrast_limits,
)


def test_exact_finite_stats_scans_every_value_and_excludes_only_nonfinite():
    values = np.zeros(600_123, dtype=np.float32)
    values[500_001] = -17.0
    values[-1] = 1_000.0
    values[511_111] = np.nan
    values[123_456] = np.inf

    stats = exact_finite_stats(values)

    assert stats.count == values.size - 2
    assert stats.minimum == -17.0
    assert stats.maximum == 1_000.0


def test_exact_finite_percentiles_retain_rare_extrema_and_constant_values():
    values = np.zeros(600_123, dtype=np.float32)
    values[500_001] = -17.0
    values[-1] = 1_000.0
    values[511_111] = np.nan

    assert exact_finite_percentiles(values, (0.0, 50.0, 100.0)) == (
        -17.0,
        0.0,
        1_000.0,
    )
    assert exact_finite_percentiles(np.full(7, 3.5), (10.0, 90.0)) == (
        3.5,
        3.5,
    )
    assert exact_finite_percentiles([np.nan, np.inf], (50.0,)) is None


@pytest.mark.parametrize(
    ("dtype", "base"),
    [
        (np.int64, 2**60),
        (np.uint64, np.iinfo(np.uint64).max - 3),
    ],
)
def test_exact_finite_percentiles_preserve_wide_integer_fractions(dtype, base):
    values = np.asarray([base + offset for offset in range(4)], dtype=dtype)

    result = exact_finite_percentiles(values, (25.0, 75.0))

    assert result == (
        Fraction(4 * int(base) + 3, 4),
        Fraction(4 * int(base) + 9, 4),
    )


@pytest.mark.parametrize("percentile", [-0.1, 100.1, np.nan, np.inf, "bad"])
def test_exact_finite_percentiles_reject_invalid_requests(percentile):
    with pytest.raises(ValueError, match="Percentiles"):
        exact_finite_percentiles(np.arange(5), (percentile,))


def test_boolean_interior_percentile_requires_explicit_numeric_conversion():
    with pytest.raises(ValueError, match="boolean"):
        exact_finite_percentiles(np.array([False, True]), (50.0,))


def test_exact_float_histogram_counts_every_finite_value_and_rare_extrema():
    values = np.zeros(600_123, dtype=np.float32)
    values[500_001] = -17.0
    values[-1] = 1_000.0
    values[511_111] = np.nan

    counts, value_range = exact_histogram(values)

    assert counts is not None
    assert counts.size == DISPLAY_HISTOGRAM_BINS
    assert int(counts.sum()) == values.size - 1
    assert counts[0] >= 1
    assert counts[-1] >= 1
    assert value_range == (-17.0, 1_000.0)


def test_exact_histogram_requires_explicit_channel_axis():
    data = np.zeros((2, 2, 3), dtype=np.float32)
    data[..., 1] = 10.0
    data[..., 2] = 20.0

    scalar_counts, scalar_range = exact_histogram(data)
    channel_counts, channel_range = exact_histogram(data, channel_axis=-1)

    assert scalar_counts is not None and scalar_counts.ndim == 1
    assert channel_counts is not None and channel_counts.shape[0] == 3
    assert int(scalar_counts.sum()) == data.size
    np.testing.assert_array_equal(channel_counts.sum(axis=1), [4, 4, 4])
    assert scalar_range == channel_range == (0.0, 20.0)


def test_multichannel_histogram_retains_all_nonfinite_channel_position():
    data = np.empty((4, 5, 3), dtype=np.float32)
    data[..., 0] = np.nan
    data[..., 1] = np.arange(20, dtype=np.float32).reshape(4, 5)
    data[..., 2] = np.arange(20, 40, dtype=np.float32).reshape(4, 5)

    counts, value_range = exact_histogram(data, channel_axis=2)

    assert counts is not None
    assert counts.shape == (3, DISPLAY_HISTOGRAM_BINS)
    np.testing.assert_array_equal(counts.sum(axis=1), [0, 20, 20])
    assert value_range == (0.0, 39.0)


@pytest.mark.parametrize(
    ("dtype", "base"),
    [
        (np.int64, 2**60),
        (np.uint64, np.iinfo(np.uint64).max - 3),
    ],
)
def test_integer_histogram_preserves_wide_native_levels(dtype, base):
    data = np.fromiter(
        [int(base), int(base) + 1, int(base) + 1, int(base) + 3],
        dtype=dtype,
        count=4,
    )

    counts, value_range = exact_histogram(data)

    np.testing.assert_array_equal(counts, [1, 2, 0, 1])
    assert value_range == (int(base), int(base) + 3)


def test_wide_integer_display_span_is_grouped_without_float_collapse():
    data = np.array([0, 2**60], dtype=np.int64)

    counts, value_range = exact_histogram(data)

    assert counts is not None and counts.size == DISPLAY_HISTOGRAM_BINS
    assert int(counts.sum()) == 2
    assert counts[0] == 1
    assert counts[-1] == 1
    assert value_range == (0, 2**60)


def test_boolean_and_constant_histograms_remain_exact():
    boolean_counts, boolean_range = exact_histogram(
        np.array([False, True, True]),
    )
    constant_counts, constant_range = exact_histogram(
        np.array([7, 7, 7], dtype=np.int16),
    )

    np.testing.assert_array_equal(boolean_counts, [1, 2])
    assert boolean_range == (0.0, 1.0)
    np.testing.assert_array_equal(constant_counts, [3])
    assert constant_range == (7, 7)


def test_exact_histogram_rejects_invalid_explicit_channel_axis():
    with pytest.raises(ValueError, match="outside"):
        exact_histogram(np.zeros((2, 3)), channel_axis=2)
    with pytest.raises(ValueError, match="scalar"):
        exact_histogram(np.asarray(5), channel_axis=0)


def test_generated_layer_contrast_limits_use_complete_finite_range_and_zero():
    nonfinite = np.array([-np.inf, -2.0, np.nan, 5.0, np.inf], dtype=np.float32)

    assert exact_generated_layer_contrast_limits(nonfinite) == (-2.0, 5.0)
    assert exact_generated_layer_contrast_limits(np.array([4.0, 8.0])) == (
        0.0,
        8.0,
    )
    assert exact_generated_layer_contrast_limits(np.zeros(4)) == (0.0, 1.0)
    assert exact_generated_layer_contrast_limits([np.nan, np.inf]) is None


def test_provisional_generated_layer_contrast_limits_are_scan_free_dtype_ranges():
    assert provisional_generated_layer_contrast_limits(
        np.zeros(2, dtype=bool)
    ) == (0.0, 1.0)
    assert provisional_generated_layer_contrast_limits(
        np.zeros(2, dtype=np.int16)
    ) == (-32_768.0, 32_767.0)
    assert provisional_generated_layer_contrast_limits(
        np.zeros(2, dtype=np.uint16)
    ) == (0.0, 65_535.0)
    assert provisional_generated_layer_contrast_limits(
        np.zeros(2, dtype=np.float32)
    ) == (0.0, 1.0)


def test_provisional_contrast_uses_dtype_without_materializing_data():
    class DtypeOnlyArray:
        dtype = np.dtype(np.uint16)

        def __array__(self):
            raise AssertionError("provisional contrast must not materialize data")

    assert provisional_generated_layer_contrast_limits(DtypeOnlyArray()) == (
        0.0,
        65_535.0,
    )


def test_label_volumes_ignore_nonpositive_ids_and_separate_leading_blocks():
    labels = np.array(
        [
            [[1, 1, 0], [2, 0, -1]],
            [[1, 1, 0], [2, 2, 0]],
        ],
        dtype=np.int16,
    )

    per_slice = label_volumes(labels, spatial_ndim=2)
    full_volume = label_volumes(labels, spatial_ndim=3)

    np.testing.assert_array_equal(per_slice, [2, 1, 2, 2])
    np.testing.assert_array_equal(full_volume, [4, 3])
    assert per_slice.dtype == np.int64
    assert largest_label_volume(labels, 2) == 2
    assert largest_label_volume(labels, 3) == 4


def test_label_volumes_return_empty_int64_for_no_positive_objects():
    volumes = label_volumes(np.zeros((0, 3), dtype=np.int32), spatial_ndim=2)

    assert volumes.dtype == np.int64
    assert volumes.size == 0
    assert largest_label_volume(np.zeros((2, 3), dtype=np.int32), 2) == 0


def test_largest_boolean_object_respects_connectivity_and_spatial_blocks():
    diagonal = np.eye(2, dtype=bool)
    z_connected = np.zeros((2, 2, 2), dtype=bool)
    z_connected[:, 0, 0] = True

    assert largest_object_size(diagonal, 2, "Face connected") == 1
    assert largest_object_size(diagonal, 2, "Fully connected") == 2
    assert largest_object_size(z_connected, 3, "Face connected") == 2
    assert largest_object_size(z_connected, 2, "Face connected") == 1


def test_largest_object_size_preserves_existing_label_ids():
    labels = np.array([[1, 1, 0], [2, 2, 2]], dtype=np.uint16)

    assert largest_object_size(labels, 2, "Face connected") == 3
    assert largest_object_size(labels, 2, "Fully connected") == 3


def test_label_diagnostics_reject_invalid_dimensionality_and_connectivity():
    labels = np.ones((2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="outside"):
        label_volumes(labels, spatial_ndim=3)
    with pytest.raises(ValueError, match="at least one"):
        label_volumes(np.asarray(1), spatial_ndim=1)
    with pytest.raises(ValueError, match="Connectivity"):
        largest_object_size(labels.astype(bool), 2, "diagonal-ish")
