"""Hysteresis connectivity and arithmetic without full-volume weighted counts."""

import numpy as np
import pytest
from skimage.filters import apply_hysteresis_threshold

from napari_vipp.core import operations
from napari_vipp.core.operations import hysteresis_threshold


def test_hysteresis_does_not_use_full_volume_weighted_bincount(monkeypatch):
    data = np.zeros((5, 7, 9), dtype=np.float32)
    data[0, 3, 4] = 0.9
    data[1:, 3, 4] = 0.5
    data[4, 1, 1] = 0.5
    expected = apply_hysteresis_threshold(data, 0.2, 0.7)
    data.setflags(write=False)
    original_bincount = np.bincount
    weighted_calls = []

    def observe_bincount(values, /, weights=None, minlength=0):
        if weights is not None:
            weighted_calls.append((weights.shape, weights.dtype))
        return original_bincount(values, weights=weights, minlength=minlength)

    monkeypatch.setattr(np, "bincount", observe_bincount)
    actual = hysteresis_threshold(data, 0.2, 0.7, spatial_mode="3D ZYX")
    np.testing.assert_array_equal(actual, expected)
    assert not weighted_calls


@pytest.mark.parametrize(
    ("values", "low", "high", "expected"),
    [
        # The legacy reference compares low against a NumPy scalar produced
        # by np.clip, but compares high against the original Python float.
        ([1.0, 2.0], 1.0 - 2**-26, 1.5, [True, True]),
        ([1.0, 0.0, 2.0], 0.5, 1.0 - 2**-26, [False, False, True]),
        (
            [1.0, 2.0, 0.0, 1.0],
            1.0 - 2**-26,
            1.0 - 2**-26,
            [True, True, False, False],
        ),
        ([0.5, 1.0, 2.0], 1.0, 1.0, [False, False, True]),
    ],
)
def test_hysteresis_preserves_reference_threshold_rounding(values, low, high, expected):
    data = np.array([values], dtype=np.float32)
    reference = apply_hysteresis_threshold(data, low, high)
    np.testing.assert_array_equal(reference, [expected])
    np.testing.assert_array_equal(hysteresis_threshold(data, low, high), reference)


@pytest.mark.parametrize("spatial_mode", ("2D YX", "3D ZYX"))
def test_hysteresis_preserves_strided_readonly_nonfinite_input(
    monkeypatch, spatial_mode
):
    monkeypatch.setattr(operations, "_THRESHOLD_CHUNK_SIZE", 7)
    data = np.zeros((2, 3, 4, 10), dtype=np.float64)[..., ::2]
    assert not data.flags.c_contiguous
    # The lower-only path spans Z in one block, plus disconnected and
    # nonfinite values. The other block has no qualifying seed.
    data[0, 0, 1, 2] = np.inf
    data[0, 1:, 1, 2] = 0.6
    data[0, 2, 0, 0] = 0.6
    data[0, 2, 2, 2] = np.nan
    data[0, 2, 1, 3] = -np.inf
    data[1, :, 1, 2] = 0.6
    original = data.copy()
    data.setflags(write=False)
    reference = np.zeros(data.shape, dtype=bool)
    spatial_ndim = 2 if spatial_mode == "2D YX" else 3
    for index in np.ndindex(data.shape[:-spatial_ndim]):
        reference[index] = apply_hysteresis_threshold(
            data[index].astype(np.float32), 0.3, 0.8
        )

    actual = hysteresis_threshold(data, 0.3, 0.8, spatial_mode=spatial_mode)
    assert actual.dtype == np.dtype(bool)
    np.testing.assert_array_equal(actual, reference)
    np.testing.assert_array_equal(data, original)
    assert not data.flags.writeable and not np.shares_memory(data, actual)
    assert actual[0, 0, 1, 2]
    assert bool(actual[0, 2, 1, 2]) == (spatial_ndim == 3)
    assert not actual[0, 2, 0, 0] and not actual[0, 2, 2, 2] and not actual[1].any()


def test_hysteresis_still_rejects_empty_image_data():
    data = np.empty((2, 0, 3), dtype=np.uint16)
    with pytest.raises(ValueError, match="Hysteresis requires non-empty image data"):
        hysteresis_threshold(data, 0.3, 0.8, spatial_mode="3D ZYX")
