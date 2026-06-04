from __future__ import annotations

import numpy as np

from napari_vipp.core.operations import (
    adaptive_gaussian_threshold,
    adaptive_mean_threshold,
    average_blur,
    bilateral_filter,
    binary_threshold,
    black_hat,
    closing,
    contrast_stretch,
    crop_stack,
    dilate,
    erode,
    extract_channel,
    fill_holes,
    gamma_correction,
    gaussian_blur,
    gaussian_blur_3d,
    median_filter,
    morphological_gradient,
    opening,
    otsu_threshold,
    top_hat,
    triangle_threshold,
    volume_filter,
)
from napari_vipp.core.pipeline import NODE_LIBRARY_BY_ID


def test_vipp_operation_nodes_are_registered():
    expected = {
        "crop_stack",
        "contrast_stretch",
        "gamma_correction",
        "average_blur",
        "gaussian_blur",
        "gaussian_blur_3d",
        "median_filter",
        "bilateral_filter",
        "binary_threshold",
        "adaptive_mean_threshold",
        "adaptive_gaussian_threshold",
        "otsu_threshold",
        "triangle_threshold",
        "dilate",
        "erode",
        "opening",
        "closing",
        "top_hat",
        "black_hat",
        "morphological_gradient",
        "fill_holes",
        "volume_filter",
        "extract_channel",
    }

    assert expected <= set(NODE_LIBRARY_BY_ID)


def test_slice_wise_filters_preserve_z_independence():
    data = np.zeros((3, 9, 9), dtype=np.float32)
    data[1, 4, 4] = 1.0

    blurred = gaussian_blur(data, sigma=1.0)
    averaged = average_blur(data, size=3)
    medianed = median_filter(data, size=3)

    assert blurred.shape == data.shape
    assert averaged.shape == data.shape
    assert medianed.shape == data.shape
    assert np.allclose(blurred[0], 0)
    assert np.allclose(averaged[0], 0)
    assert np.allclose(medianed[0], 0)


def test_gaussian_3d_spreads_across_z_axis():
    data = np.zeros((3, 9, 9), dtype=np.float32)
    data[1, 4, 4] = 1.0

    blurred = gaussian_blur_3d(data, sigma_z=1.0, sigma_y=0.0, sigma_x=0.0)

    assert blurred.shape == data.shape
    assert blurred[0, 4, 4] > 0
    assert blurred[2, 4, 4] > 0


def test_bilateral_filter_preserves_shape():
    data = np.zeros((2, 8, 8), dtype=np.uint8)
    data[:, 2:6, 2:6] = 200

    result = bilateral_filter(data, diameter=3, sigma_color=5, sigma_space=2)

    assert result.shape == data.shape
    assert result.dtype == np.float32


def test_contrast_gamma_crop_and_extract_channel():
    data = np.zeros((2, 6, 7, 3), dtype=np.uint8)
    data[..., 1] = 64
    data[:, 2:5, 1:6, 2] = 128

    stretched = contrast_stretch(data, alpha=2, beta=1)
    gamma = gamma_correction(data, gamma=0.5)
    cropped = crop_stack(data, top=1, bottom=2, left=1, right=3)
    channel = extract_channel(data, channel=2)

    assert stretched.dtype == np.uint8
    assert stretched[..., 1].max() == 129
    assert gamma.dtype == np.uint8
    assert cropped.shape == (2, 3, 3, 3)
    assert channel.shape == (2, 6, 7)
    assert channel.max() == 128


def test_thresholding_operations_return_masks():
    data = np.tile(np.arange(12, dtype=np.uint8), (12, 1))
    data = np.stack([data, data + 20])

    masks = [
        binary_threshold(data, threshold=6),
        adaptive_mean_threshold(data, block_size=5, c=0),
        adaptive_gaussian_threshold(data, block_size=5, c=0),
        otsu_threshold(data),
        triangle_threshold(data),
    ]

    for mask in masks:
        assert mask.dtype == bool
        assert mask.shape == data.shape
        assert mask.any()


def test_morphology_and_volume_operations_return_masks():
    mask = np.zeros((3, 9, 9), dtype=bool)
    mask[1, 4, 4] = True
    mask[1, 2:7, 2:7] = True
    mask[1, 4, 4] = False
    mask[0, 0, 0] = True

    closed_cavity = np.ones((3, 5, 5), dtype=bool)
    closed_cavity[1, 2, 2] = False
    filled = fill_holes(closed_cavity)
    filtered = volume_filter(mask, min_volume=5)

    assert dilate(mask, size=3, iterations=1).sum() > mask.sum()
    assert erode(filled, size=3, iterations=1).sum() < filled.sum()
    assert opening(mask, size=2).dtype == bool
    assert closing(mask, size=2).dtype == bool
    assert top_hat(mask, size=2).dtype == bool
    assert black_hat(mask, size=2).dtype == bool
    assert morphological_gradient(mask, size=2).dtype == bool
    assert filled[1, 2, 2]
    assert not filtered[0, 0, 0]
    assert filtered[1].any()
