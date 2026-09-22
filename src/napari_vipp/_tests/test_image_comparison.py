from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import minimum_filter
from skimage.metrics import structural_similarity

from napari_vipp.core import image_comparison
from napari_vipp.core.image_comparison import compare_images
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.progress import OperationCancelled, ProgressContext


def state(array, names=None):
    names = names or ("YX" if array.ndim == 2 else "ZYX")
    axes = tuple(
        AxisMetadata(
            name.lower(),
            "time" if name == "T" else "channel" if name == "C" else "space",
        )
        for name in names
    )
    return image_state_from_array(array, axes=axes, defer_statistics=True)


def compare(a, b, mask=None, **kwargs):
    inputs = [a, b] + ([mask] if mask is not None else [])
    return compare_images(
        inputs,
        input_states=[state(x) for x in inputs],
        use_mask=mask is not None,
        **kwargs,
    ).records()[0]


@pytest.mark.parametrize("shape", [(19, 23), (11, 13, 17)])
def test_matches_independent_metric_oracles_and_preserves_inputs(shape):
    rng = np.random.default_rng(71)
    a = rng.random(shape)
    b = a * 0.8 + rng.normal(0, 0.03, shape)
    copies = [x.copy() for x in (a, b)]
    a.setflags(write=False)
    b.setflags(write=False)
    result = compare(a, b, data_range=1.0)
    assert result["valid_count"] == a.size
    assert result["rmse"] == pytest.approx(np.sqrt(np.mean((a - b) ** 2)))
    assert result["pearson_r"] == pytest.approx(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    assert result["ssim"] == pytest.approx(structural_similarity(a, b, data_range=1.0))
    assert result["ssim_window_count"] == np.prod(np.asarray(shape) - 6)
    assert result["psnr_db"] == pytest.approx(20 * np.log10(1 / result["rmse"]))
    for source, original in zip((a, b), copies, strict=True):
        np.testing.assert_array_equal(source, original)


def test_mask_counts_entire_ssim_windows_and_ignores_excluded_nan():
    rng = np.random.default_rng(3)
    a = rng.random((25, 29))
    b = a * 0.9
    mask = np.ones(a.shape, bool)
    mask[10:13, 14:17] = False
    a[~mask] = np.nan
    b[~mask] = np.inf
    result = compare(a, b, mask, window_size=5)
    expected_centers = minimum_filter(
        mask.astype(np.uint8), size=5, mode="constant", cval=0
    ).astype(bool)
    _mean, expected_ssim = structural_similarity(
        np.where(mask, a, 0),
        np.where(mask, b, 0),
        data_range=1.0,
        win_size=5,
        full=True,
    )
    assert result["valid_count"] == mask.sum()
    assert result["ssim_window_count"] == expected_centers.sum()
    assert result["ssim"] == pytest.approx(expected_ssim[expected_centers].mean())
    assert result["rmse"] == pytest.approx(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def test_ssim_slabs_match_full_image_including_short_final_slab(monkeypatch):
    monkeypatch.setattr(image_comparison, "_SSIM_SLAB_VALUES", 63)
    a = np.random.default_rng(89).random((22, 9))
    b = a * 0.75
    result = compare(a, b)
    assert result["ssim"] == pytest.approx(structural_similarity(a, b, data_range=1.0))


def test_time_channels_are_reported_separately_with_interleaved_axes():
    rng = np.random.default_rng(8)
    a = rng.random((2, 9, 3, 11, 13))
    b = a.copy()
    b[1, :, 2] += 0.25
    states = [state(x, "TZCYX") for x in (a, b)]
    records = compare_images([a, b], input_states=states).records()
    assert len(records) == 6
    for row in records:
        assert row["valid_count"] == 9 * 11 * 13
        expected = 0.25 if (row["t_index"], row["c_index"]) == (1, 2) else 0.0
        assert row["rmse"] == pytest.approx(expected)


def test_constant_exact_psnr_and_undefined_correlation_are_json_safe():
    a = np.ones((11, 13))
    result = compare(a, a)
    assert result["rmse"] == 0
    assert result["pearson_r"] is None
    assert result["ssim"] == 1
    assert result["psnr_db"] is None
    assert result["psnr_status"] == "infinite (exact match)"
    assert result["correlation_status"] == "undefined (constant image)"


def test_insufficient_ssim_extent_does_not_claim_zero_score():
    a = np.arange(24.0).reshape(4, 6)
    result = compare(a, a)
    assert result["ssim"] is None and result["ssim_window_count"] == 0
    assert result["pearson_r"] == pytest.approx(1)


@pytest.mark.parametrize(
    "problem",
    ["origin", "scale", "dtype", "ambiguous", "duplicate", "semantics", "shape"],
)
def test_invalid_metadata_or_grid_rejected(problem):
    a = np.ones((11, 13))
    first, second = state(a), state(a)
    if problem == "origin":
        second = replace(
            second, axes=(replace(second.axes[0], translation=1), second.axes[1])
        )
    elif problem == "scale":
        second = replace(
            second, axes=(replace(second.axes[0], scale=2), second.axes[1])
        )
    elif problem == "dtype":
        second = replace(second, dtype="uint8")
    elif problem == "ambiguous":
        second = replace(
            second,
            axes=tuple(replace(x, confidence="shape-inferred") for x in second.axes),
        )
    elif problem == "duplicate":
        first = second = replace(first, axes=(first.axes[0], first.axes[0]))
    elif problem == "semantics":
        first = second = replace(
            first, axes=(replace(first.axes[0], name="t"), first.axes[1])
        )
    else:
        second = replace(second, shape=(13, 11))
    with pytest.raises(ValueError):
        compare_images([a, a], input_states=[first, second])


@pytest.mark.parametrize(
    "problem",
    ["nan", "inf", "complex", "wide integer", "empty mask", "nonboolean mask"],
)
def test_invalid_data_rejected(problem):
    a = np.ones((11, 13))
    b = a.copy()
    mask = None
    if problem == "nan":
        b[0, 0] = np.nan
    elif problem == "inf":
        b[0, 0] = np.inf
    elif problem == "complex":
        b = b.astype(complex)
    elif problem == "wide integer":
        b = np.full(a.shape, 2**64 - 1, dtype=np.uint64)
    elif problem == "empty mask":
        mask = np.zeros(a.shape, bool)
    else:
        mask = np.ones(a.shape, np.uint8)
    with pytest.raises(ValueError):
        compare(a, b, mask)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"data_range": 0},
        {"data_range": np.inf},
        {"data_range": True},
        {"window_size": 2},
        {"window_size": 4},
        {"window_size": True},
    ],
)
def test_invalid_parameters_rejected(kwargs):
    a = np.ones((11, 13))
    with pytest.raises(ValueError):
        compare(a, a, **kwargs)


def test_chunked_covariance_matches_oracle_for_large_transposed_array():
    rng = np.random.default_rng(18)
    a = (1e9 + rng.normal(size=(521, 519))).T
    b = 2 * a + rng.normal(size=a.shape)
    count, rmse, correlation = image_comparison._moments(
        a, b, np.ones(a.shape, bool), None
    )
    assert count == a.size
    assert correlation == pytest.approx(
        np.corrcoef(a.ravel(), b.ravel())[0, 1], abs=1e-9
    )
    assert rmse == pytest.approx(np.sqrt(np.mean((a - b) ** 2)))


def test_large_finite_variance_does_not_overflow_pearson_denominator():
    a = np.arange(99.0).reshape(9, 11) * 1e100
    count, rmse, correlation = image_comparison._moments(
        a, a, np.ones(a.shape, bool), None
    )
    assert count == 99 and rmse == 0 and correlation == pytest.approx(1)


def test_cancel_and_no_valid_ssim_windows():
    a = np.arange(143.0).reshape(11, 13)
    with pytest.raises(OperationCancelled):
        compare_images(
            [a, a],
            input_states=[state(a), state(a)],
            progress=ProgressContext(cancelled=lambda: True),
        )
    mask = np.zeros(a.shape, bool)
    mask[5, 6] = True
    result = compare(a, a, mask)
    assert result["valid_count"] == 1
    assert result["ssim_window_count"] == 0 and result["ssim"] is None
