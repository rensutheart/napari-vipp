"""Exact median qualification against the previous SciPy CPU implementation."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from napari_vipp.core import simpleitk_filters as adapters
from napari_vipp.core.operations import median_filter

_DTYPES = (
    np.uint8,
    np.uint16,
    np.uint32,
    np.int8,
    np.int16,
    np.int32,
    np.float32,
    np.float64,
)
_LAYOUTS = (
    ((1, 1), None),
    ((1, 9), None),
    ((11, 1), None),
    ((9, 11), None),
    ((2, 9, 11), None),
    ((9, 11, 3), None),  # no implicit RGB interpretation
    ((3, 9, 11), 0),
    ((9, 3, 11), 1),
    ((9, 11, 3), -1),
    ((2, 3, 9, 11), 1),
    ((2, 9, 3, 11), 2),
    ((2, 3, 4, 9, 11), 2),
)


def _fixture(dtype, shape):
    rng = np.random.default_rng(260922 + sum(shape))
    dtype = np.dtype(dtype)
    if dtype.kind in "iu":
        info = np.iinfo(dtype)
        values = rng.integers(info.min, info.max, size=shape, dtype=dtype)
        specials = (info.min, info.max, 0, 7, 7, 7, info.max - 1)
    else:
        values = rng.normal(size=shape).astype(dtype)
        info = np.finfo(dtype)
        tiny = np.nextafter(dtype.type(0), dtype.type(1), dtype=dtype)
        specials = (-info.max, info.max, 0, -tiny, tiny, 1.25, 1.25)
    values.flat[: min(values.size, len(specials))] = specials[: values.size]
    return values


def _axes(array, channel_axis):
    spatial = list(range(array.ndim))
    if channel_axis is not None:
        spatial.remove(channel_axis % array.ndim)
    return tuple(spatial[-2:])


def _assert_exact(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    # Numeric equality alone cannot detect floating-point signed-zero changes.
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("size", range(1, 52, 2))
def test_all_qualified_footprints_and_dtype_extremes_are_bitwise_exact(dtype, size):
    array = _fixture(dtype, (9, 11))
    expected = adapters.scipy_median_filter(array, size=size, xy_axes=(0, 1))
    actual = adapters._simpleitk_median_filter(array, size=size, xy_axes=(0, 1))
    _assert_exact(actual, expected)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape,channel_axis", _LAYOUTS)
def test_layouts_degenerate_axes_and_independent_planes(dtype, shape, channel_axis):
    array = _fixture(dtype, shape)
    axes = _axes(array, channel_axis)
    expected = adapters.scipy_median_filter(array, size=5, xy_axes=axes)
    actual = adapters._simpleitk_median_filter(array, size=5, xy_axes=axes)
    _assert_exact(actual, expected)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_read_only_strided_views_and_chunking_preserve_upstream(dtype, monkeypatch):
    original = _fixture(dtype, (2, 3, 13, 17, 3))
    array = original[:, :, ::-1, ::2, :]
    array.flags.writeable = False
    before = original.copy()
    monkeypatch.setattr(adapters, "_MAX_CHUNK_BYTES", 2500)
    expected = adapters.scipy_median_filter(array, size=7, xy_axes=(2, 3))
    actual = adapters._simpleitk_median_filter(array, size=7, xy_axes=(2, 3))
    _assert_exact(actual, expected)
    _assert_exact(original, before)
    assert not np.shares_memory(actual, array)
    assert actual.flags.c_contiguous


@pytest.mark.parametrize("dtype", _DTYPES)
def test_public_operation_uses_qualified_backend_and_preserves_axes(dtype, monkeypatch):
    array = _fixture(dtype, (2, 3, 11, 13))
    before = array.copy()
    array.flags.writeable = False
    monkeypatch.setattr(adapters, "MEDIAN_MIN_PLANE_PIXELS", 1)
    assert adapters.median_filter_backend(array, size=5, xy_axes=(2, 3)) == "simpleitk"
    expected = adapters.scipy_median_filter(array, size=5, xy_axes=(2, 3))
    actual = median_filter(array, size=5, channel_axis=1)
    _assert_exact(actual, expected)
    _assert_exact(array, before)


@pytest.mark.parametrize(
    "requested,canonical", [(-4, 1), (0, 1), (2, 3), (4, 5), (50, 51)]
)
def test_existing_size_canonicalization_is_unchanged(requested, canonical, monkeypatch):
    array = _fixture(np.uint16, (51, 53))
    monkeypatch.setattr(adapters, "MEDIAN_MIN_PLANE_PIXELS", 1)
    expected = adapters.scipy_median_filter(array, size=canonical, xy_axes=(0, 1))
    _assert_exact(median_filter(array, size=requested), expected)


@pytest.mark.parametrize("dtype", [np.bool_, np.int64, np.uint64, ">u2"])
def test_unqualified_dtype_retains_scipy(dtype, monkeypatch):
    array = np.arange(256 * 256).reshape(256, 256).astype(dtype)
    if array.dtype.kind in "iu" and array.dtype.itemsize == 8:
        array += 2**54  # retain SciPy's existing wide-integer behaviour
    monkeypatch.setattr(
        adapters, "_simpleitk_module", lambda: pytest.fail("ITK loaded")
    )
    assert adapters.median_filter_backend(array, size=5, xy_axes=(0, 1)) == "scipy"
    expected = adapters.scipy_median_filter(array, size=5, xy_axes=(0, 1))
    _assert_exact(median_filter(array), expected)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -0.0])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nonfinite_and_negative_zero_retain_scipy(value, dtype, monkeypatch):
    array = _fixture(dtype, (256, 256))
    array[80, 70] = value
    monkeypatch.setattr(
        adapters, "_simpleitk_module", lambda: pytest.fail("ITK loaded")
    )
    assert adapters.median_filter_backend(array, size=5, xy_axes=(0, 1)) == "scipy"
    expected = adapters.scipy_median_filter(array, size=5, xy_axes=(0, 1))
    _assert_exact(median_filter(array), expected)


@pytest.mark.parametrize(
    "shape,size",
    [
        ((17,), 3),
        ((128, 128), 5),
        ((1, 65536), 5),
        ((0, 256), 5),
        ((256, 256), 1),
        ((55, 55), 53),
    ],
)
def test_small_degenerate_and_unqualified_sizes_retain_scipy(shape, size, monkeypatch):
    array = _fixture(np.uint16, shape)
    axes = (0, 0) if array.ndim == 1 else (0, 1)
    monkeypatch.setattr(
        adapters, "_simpleitk_module", lambda: pytest.fail("ITK loaded")
    )
    assert adapters.median_filter_backend(array, size=size, xy_axes=axes) == "scipy"
    expected = adapters.scipy_median_filter(array, size=size, xy_axes=axes)
    _assert_exact(median_filter(array, size=size), expected)


@pytest.mark.parametrize("dtype", [np.float16, np.complex64, object])
def test_previously_unsupported_dtypes_do_not_gain_implicit_conversions(dtype):
    array = np.ones((9, 11), dtype=dtype)
    with pytest.raises((RuntimeError, TypeError)) as previous:
        adapters.scipy_median_filter(array, size=5, xy_axes=(0, 1))
    with pytest.raises(type(previous.value), match=str(previous.value)):
        median_filter(array)


@pytest.mark.parametrize("channel_axis", [True, 1.5, 3, -4])
def test_public_channel_validation_precedes_backend_loading(channel_axis, monkeypatch):
    monkeypatch.setattr(
        adapters, "_simpleitk_module", lambda: pytest.fail("ITK loaded")
    )
    with pytest.raises(ValueError, match="channel_axis"):
        median_filter(
            np.zeros((256, 256, 3), dtype=np.uint8), channel_axis=channel_axis
        )


def test_backend_import_failure_is_not_silently_downgraded(monkeypatch):
    def broken_import():
        raise ImportError("broken SimpleITK binary")

    monkeypatch.setattr(adapters, "_simpleitk_module", broken_import)
    with pytest.raises(ImportError, match="broken SimpleITK binary"):
        median_filter(np.zeros((256, 256), dtype=np.uint16))


def test_execution_failure_is_not_silently_downgraded(monkeypatch):
    def broken_filter(*args, **kwargs):
        raise RuntimeError("ITK failed during execution")

    monkeypatch.setattr(adapters, "_simpleitk_median_filter", broken_filter)
    with pytest.raises(RuntimeError, match="ITK failed during execution"):
        median_filter(np.zeros((256, 256), dtype=np.uint16))


def test_filter_does_not_change_global_itk_thread_settings(monkeypatch):
    sitk = adapters._simpleitk_module()
    before_threads = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    before_threader = sitk.ProcessObject.GetGlobalDefaultThreader()
    monkeypatch.setattr(adapters.os, "cpu_count", lambda: 512)
    array = _fixture(np.uint16, (9, 11))
    actual = adapters._simpleitk_median_filter(array, size=3, xy_axes=(0, 1))
    _assert_exact(actual, adapters.scipy_median_filter(array, size=3, xy_axes=(0, 1)))
    assert sitk.ProcessObject.GetGlobalDefaultNumberOfThreads() == before_threads
    assert sitk.ProcessObject.GetGlobalDefaultThreader() == before_threader


def test_importing_operations_does_not_load_simpleitk():
    script = """
import builtins
import importlib
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'SimpleITK' or name.startswith('SimpleITK.'):
        raise AssertionError('SimpleITK imported eagerly')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
importlib.import_module('napari_vipp.core.operations')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
