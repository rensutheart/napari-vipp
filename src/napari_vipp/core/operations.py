"""Prototype image-processing operations."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def gaussian_blur(data, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian blur while preserving RGB/RGBA channel axes."""
    arr = np.asarray(data)
    sigma = max(float(sigma), 0.0)
    if sigma == 0:
        return arr.copy()

    if arr.ndim >= 3 and arr.shape[-1] in (3, 4):
        sigma_by_axis = [sigma] * arr.ndim
        sigma_by_axis[-1] = 0.0
        return ndi.gaussian_filter(arr, sigma=sigma_by_axis)
    return ndi.gaussian_filter(arr, sigma=sigma)


def otsu_threshold(data) -> np.ndarray:
    """Return a binary mask from an Otsu threshold."""
    arr = np.asarray(data)
    grayscale = _to_grayscale(arr)
    threshold = _otsu_value(grayscale)
    return grayscale > threshold


def binary_threshold(data, threshold: float = 0.5) -> np.ndarray:
    """Return a binary mask using a fixed intensity threshold."""
    arr = _to_grayscale(np.asarray(data))
    return arr > float(threshold)


def median_filter(data, size: int = 3) -> np.ndarray:
    """Apply a median filter while preserving RGB/RGBA channel axes."""
    arr = np.asarray(data)
    size = max(int(size), 1)
    if arr.ndim >= 3 and arr.shape[-1] in (3, 4):
        size_by_axis = [size] * arr.ndim
        size_by_axis[-1] = 1
        return ndi.median_filter(arr, size=size_by_axis)
    return ndi.median_filter(arr, size=size)


def invert(data) -> np.ndarray:
    """Invert boolean masks or intensity images."""
    arr = np.asarray(data)
    if arr.dtype == bool:
        return np.logical_not(arr)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr.copy()
    return finite.max() + finite.min() - arr


def max_intensity_projection(data, axis: int = 0) -> np.ndarray:
    """Project an image along one axis using maximum intensity."""
    arr = np.asarray(data)
    if arr.ndim <= 2:
        return arr.copy()
    axis = int(axis)
    if axis < 0 or axis >= arr.ndim:
        axis = 0
    return np.max(arr, axis=axis)


def _to_grayscale(arr: np.ndarray) -> np.ndarray:
    if arr.ndim >= 3 and arr.shape[-1] in (3, 4):
        rgb = arr[..., :3].astype(np.float32)
        return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    return arr.astype(np.float32, copy=False)


def _otsu_value(arr: np.ndarray) -> float:
    values = arr[np.isfinite(arr)].astype(np.float64, copy=False)
    if values.size == 0:
        return 0.0
    if values.min() == values.max():
        return float(values.min())

    hist, edges = np.histogram(values, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * centers) / np.maximum(weight1, 1)
    mean2 = (
        np.cumsum((hist * centers)[::-1]) / np.maximum(weight2[::-1], 1)
    )[::-1]
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    if variance12.size == 0:
        return float(values.mean())
    return float(centers[:-1][np.argmax(variance12)])
