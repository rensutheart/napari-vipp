from __future__ import annotations

import importlib
import subprocess
import sys

import numpy as np
import pytest

from napari_vipp.core.gpu.cupy_median import median_filter as gpu_median_filter
from napari_vipp.core.operations import median_filter as cpu_median_filter


@pytest.fixture(scope="module")
def cupy_module():
    try:
        cupy = importlib.import_module("cupy")
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No CUDA device is available.")
        cupy.zeros(1, dtype=cupy.uint8).sum().item()
    except Exception as exc:
        pytest.skip(f"A working CuPy CUDA device is unavailable: {exc}")
    return cupy


def test_import_is_safe_when_cupy_imports_are_blocked():
    script = r"""
import builtins
import importlib
import sys

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "cupy" or name.startswith("cupy.") or name.startswith("cupyx"):
        raise AssertionError(f"optional CUDA import attempted: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
module = importlib.import_module("napari_vipp.core.gpu.cupy_median")
assert callable(module.median_filter)
assert "cupy" not in sys.modules
assert not any(name.startswith("cupyx") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


_VALIDATED_LAYOUTS = (
    ((17,), None),
    ((9, 11), None),
    ((2, 9, 11), None),
    ((3, 9, 11), 0),
    ((9, 3, 11), 1),
    ((9, 11, 3), None),
    ((9, 11, 3), -1),
    ((2, 3, 9, 11), 1),
    ((2, 9, 3, 11), 2),
    ((2, 3, 4, 9, 11), 2),
)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("shape,channel_axis", _VALIDATED_LAYOUTS)
def test_real_gpu_exact_parity_across_dtype_rank_and_channel_axes(
    cupy_module,
    dtype,
    shape,
    channel_axis,
):
    host = _finite_fixture(dtype, shape)
    device = cupy_module.asarray(host)

    expected = cpu_median_filter(host, size=5, channel_axis=channel_axis)
    output = gpu_median_filter(device, size=5, channel_axis=channel_axis)
    actual = cupy_module.asnumpy(output)

    assert isinstance(output, cupy_module.ndarray)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    _assert_bitwise_equal(expected, actual)


@pytest.mark.parametrize(
    "requested_size,canonical_size",
    [(-4, 1), (0, 1), (1, 1), (2, 3), (4, 5), (50, 51), (51, 51)],
)
def test_real_gpu_size_canonicalization_and_validated_boundaries(
    cupy_module,
    requested_size,
    canonical_size,
):
    # The active axes are at least as large as the largest admitted footprint.
    # Smaller axes remain outside the declaration evidence, even though many
    # such combinations happen to agree.
    host = _finite_fixture(np.uint8, (51, 53))
    device = cupy_module.asarray(host)

    expected = cpu_median_filter(host, size=requested_size)
    canonical = cpu_median_filter(host, size=canonical_size)
    output = gpu_median_filter(device, size=requested_size)
    actual = cupy_module.asnumpy(output)

    _assert_bitwise_equal(expected, canonical)
    _assert_bitwise_equal(expected, actual)


@pytest.mark.parametrize(
    "shape,channel_axis",
    [
        ((9, 11, 3), True),
        ((9, 11, 3), 1.5),
        ((9, 11), 0),
        ((9, 11, 3), 3),
        ((9, 11, 3), -4),
    ],
)
def test_channel_axis_validation_matches_cpu(cupy_module, shape, channel_axis):
    host = np.zeros(shape, dtype=np.uint8)
    device = cupy_module.asarray(host)
    with pytest.raises(ValueError) as cpu_error:
        cpu_median_filter(host, channel_axis=channel_axis)

    with pytest.raises(ValueError) as gpu_error:
        gpu_median_filter(device, channel_axis=channel_axis)

    assert str(gpu_error.value) == str(cpu_error.value)


def test_device_input_is_not_mutated_and_output_remains_device_resident(cupy_module):
    host = _finite_fixture(np.float32, (2, 9, 11)).swapaxes(1, 2)
    device = cupy_module.asarray(host).swapaxes(1, 2)
    expected_input = device.copy()

    output = gpu_median_filter(device, size=3)
    cupy_module.cuda.get_current_stream().synchronize()

    assert isinstance(output, cupy_module.ndarray)
    assert output.data.ptr != device.data.ptr
    cupy_module.testing.assert_array_equal(device, expected_input)
    expected = cpu_median_filter(cupy_module.asnumpy(device), size=3)
    _assert_bitwise_equal(expected, cupy_module.asnumpy(output))


def _finite_fixture(dtype, shape):
    seed = sum((index + 1) * extent for index, extent in enumerate(shape))
    seed += np.dtype(dtype).itemsize * 10_003
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        values = rng.integers(0, info.max + 1, size=shape, dtype=dtype)
        values.flat[:6] = (0, info.max, info.max // 2, 7, 7, 7)
        return values
    values = rng.normal(loc=0.25, scale=3.0, size=shape).astype(dtype)
    info = np.finfo(dtype)
    values.flat[:7] = (-info.max, info.max, 0.0, 1.25, 1.25, 1.25, -2.5)
    assert np.isfinite(values).all()
    assert not np.any((values == 0) & np.signbit(values))
    return values


def _assert_bitwise_equal(expected, actual):
    assert expected.dtype == actual.dtype
    assert expected.shape == actual.shape
    byte_dtype = np.dtype((np.void, expected.dtype.itemsize))
    np.testing.assert_array_equal(
        expected.view(byte_dtype),
        actual.view(byte_dtype),
    )
