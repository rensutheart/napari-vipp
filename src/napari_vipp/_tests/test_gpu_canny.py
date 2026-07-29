from __future__ import annotations

import importlib
import subprocess
import sys

import numpy as np
import pytest

from napari_vipp.core.gpu.cupy_canny import canny_edges as gpu_canny_edges
from napari_vipp.core.operations import canny_edges as cpu_canny_edges
from napari_vipp.core.progress import OperationCancelled, ProgressContext


@pytest.fixture(scope="module")
def cupy_module():
    try:
        cupy = importlib.import_module("cupy")
        importlib.import_module("cupyx.scipy.ndimage")
        if int(cupy.cuda.runtime.getDeviceCount()) < 1:
            pytest.skip("No CUDA device is available.")
        cupy.zeros(1, dtype=cupy.uint8).sum().item()
    except Exception as exc:
        pytest.skip(f"A working CuPy CUDA device is unavailable: {exc}")
    return cupy


def test_canny_adapter_import_does_not_import_optional_cuda_modules():
    script = r"""
import importlib.abc
import sys

class BlockCUDA(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if fullname == "cupy" or fullname.startswith("cupy."):
            raise ImportError("CuPy import was blocked")
        if fullname == "cupyx" or fullname.startswith("cupyx."):
            raise ImportError("CuPyX import was blocked")
        return None

sys.meta_path.insert(0, BlockCUDA())
import napari_vipp.core.gpu.cupy_canny as adapter
assert adapter.__all__ == ["canny_edges"]
assert not any(name == "cupy" or name.startswith("cupy.") for name in sys.modules)
assert not any(name == "cupyx" or name.startswith("cupyx.") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


_LAYOUTS = (
    ("yx", (37, 43), None),
    ("leading-planes", (3, 37, 43), None),
    ("channel-first", (3, 37, 43), 0),
    ("leading-channel", (2, 3, 37, 43), 1),
    ("rgba-last", (37, 43, 4), -1),
    ("scalar-trailing-three", (2, 37, 3), None),
)


@pytest.mark.parametrize("dtype", (np.bool_, np.uint8, np.uint16, np.float32))
@pytest.mark.parametrize("case_name,shape,channel_axis", _LAYOUTS)
def test_canny_real_gpu_exact_mask_parity_across_dtype_rank_and_channels(
    cupy_module,
    dtype,
    case_name,
    shape,
    channel_axis,
):
    host = _finite_fixture(dtype, shape, seed=1901 + len(shape) * 37)
    expected = cpu_canny_edges(
        host,
        sigma=1.3,
        low_quantile=0.1,
        high_quantile=0.25,
        channel_axis=channel_axis,
    )
    output = gpu_canny_edges(
        cupy_module.asarray(host),
        sigma=1.3,
        low_quantile=0.1,
        high_quantile=0.25,
        channel_axis=channel_axis,
    )
    cupy_module.cuda.get_current_stream().synchronize()

    assert isinstance(output, cupy_module.ndarray), case_name
    assert output.dtype == cupy_module.bool_
    actual = cupy_module.asnumpy(output)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected, strict=True)


@pytest.mark.parametrize("sigma", (-3.0, 0.0, 0.25, 1.0, 3.0, 12.0))
@pytest.mark.parametrize(
    "low_quantile,high_quantile",
    ((0.0, 0.0), (0.0, 1.0), (0.1, 0.2), (1.0, 1.0)),
)
def test_canny_real_gpu_exact_parameter_boundary_parity(
    cupy_module,
    sigma,
    low_quantile,
    high_quantile,
):
    host = _structured_fixture((31, 35))
    expected = cpu_canny_edges(
        host,
        sigma=sigma,
        low_quantile=low_quantile,
        high_quantile=high_quantile,
    )
    output = gpu_canny_edges(
        cupy_module.asarray(host),
        sigma=sigma,
        low_quantile=low_quantile,
        high_quantile=high_quantile,
    )

    np.testing.assert_array_equal(
        cupy_module.asnumpy(output),
        expected,
        strict=True,
    )


def test_canny_gpu_matches_scipy_symmetric_tap_order_under_extreme_cancellation(
    cupy_module,
):
    maximum = np.finfo(np.float32).max
    host = np.array(
        (
            (maximum / 2, 0.0, 0.0),
            (-1.0, -1.0, 1.0),
            (-maximum / 2, -1.0, -1.0),
        ),
        dtype=np.float32,
    )

    with np.errstate(over="ignore", invalid="ignore"):
        expected = cpu_canny_edges(
            host,
            sigma=0.0,
            low_quantile=0.0,
            high_quantile=0.0,
        )
        output = gpu_canny_edges(
            cupy_module.asarray(host),
            sigma=0.0,
            low_quantile=0.0,
            high_quantile=0.0,
        )

    actual = cupy_module.asnumpy(output)
    assert expected[1, 1]
    np.testing.assert_array_equal(actual, expected, strict=True)


@pytest.mark.parametrize("shape", ((1, 1), (1, 13), (13, 1), (2, 2), (3, 5)))
def test_canny_real_gpu_exact_narrow_plane_and_border_parity(cupy_module, shape):
    host = _structured_fixture(shape)
    expected = cpu_canny_edges(host, sigma=1.0)
    output = gpu_canny_edges(cupy_module.asarray(host), sigma=1.0)

    np.testing.assert_array_equal(
        cupy_module.asnumpy(output),
        expected,
        strict=True,
    )


@pytest.mark.parametrize(
    "low_quantile,high_quantile",
    (
        (np.nan, 0.2),
        (0.1, np.inf),
        (-0.1, 0.2),
        (0.1, 1.1),
        (0.8, 0.2),
    ),
)
def test_canny_gpu_threshold_validation_matches_cpu(
    cupy_module,
    low_quantile,
    high_quantile,
):
    host = np.zeros((8, 9), dtype=np.float32)
    with pytest.raises(ValueError) as cpu_error:
        cpu_canny_edges(
            host,
            low_quantile=low_quantile,
            high_quantile=high_quantile,
        )
    with pytest.raises(ValueError) as gpu_error:
        gpu_canny_edges(
            cupy_module.asarray(host),
            low_quantile=low_quantile,
            high_quantile=high_quantile,
        )

    assert str(gpu_error.value) == str(cpu_error.value)


@pytest.mark.parametrize(
    "shape,channel_axis",
    (
        ((8, 9, 3), True),
        ((8, 9, 3), 1.5),
        ((8, 9), 0),
        ((8, 9, 3), 3),
        ((8, 9, 3), -4),
        ((2, 5, 8, 9), 1),
    ),
)
def test_canny_gpu_channel_validation_matches_cpu(
    cupy_module,
    shape,
    channel_axis,
):
    host = np.zeros(shape, dtype=np.uint8)
    with pytest.raises(ValueError) as cpu_error:
        cpu_canny_edges(host, channel_axis=channel_axis)
    with pytest.raises(ValueError) as gpu_error:
        gpu_canny_edges(cupy_module.asarray(host), channel_axis=channel_axis)

    assert str(gpu_error.value) == str(cpu_error.value)


def test_canny_gpu_preserves_noncontiguous_input_and_output_residency(cupy_module):
    host_base = _finite_fixture(np.float32, (2, 43, 37), seed=2903)
    host = host_base.transpose(0, 2, 1)
    device_base = cupy_module.asarray(host_base)
    device = device_base.transpose(0, 2, 1)
    before = device.copy()

    expected = cpu_canny_edges(host, sigma=0.85, low_quantile=0.2)
    output = gpu_canny_edges(device, sigma=0.85, low_quantile=0.2)
    cupy_module.cuda.get_current_stream().synchronize()

    assert isinstance(output, cupy_module.ndarray)
    assert output.device.id == device.device.id
    assert output.data.ptr != device.data.ptr
    cupy_module.testing.assert_array_equal(device, before)
    np.testing.assert_array_equal(
        cupy_module.asnumpy(output),
        expected,
        strict=True,
    )


def test_canny_gpu_progress_matches_cpu_completed_plane_updates(cupy_module):
    host = _finite_fixture(np.uint16, (2, 3, 31, 35), seed=3907)
    cpu_updates = []
    gpu_updates = []

    cpu_canny_edges(
        host,
        sigma=1.2,
        progress=ProgressContext(reporter=cpu_updates.append),
    )
    gpu_canny_edges(
        cupy_module.asarray(host),
        sigma=1.2,
        progress=ProgressContext(reporter=gpu_updates.append),
    )

    assert gpu_updates == cpu_updates


def test_canny_gpu_checks_cancellation_before_and_between_planes(cupy_module):
    host = _finite_fixture(np.float32, (3, 31, 35), seed=4909)
    with pytest.raises(OperationCancelled):
        gpu_canny_edges(
            cupy_module.asarray(host),
            progress=ProgressContext(cancelled=lambda: True),
        )

    cancel = False
    updates = []

    def cancelled():
        return cancel

    def reported(update):
        nonlocal cancel
        updates.append(update)
        if update.current == 1:
            cancel = True

    with pytest.raises(OperationCancelled):
        gpu_canny_edges(
            cupy_module.asarray(host),
            progress=ProgressContext(cancelled=cancelled, reporter=reported),
        )

    assert [(update.current, update.total) for update in updates] == [(0, 3), (1, 3)]


def test_canny_gpu_empty_input_error_matches_cpu(cupy_module):
    host = np.empty((0, 9), dtype=np.float32)
    with pytest.raises(ValueError) as cpu_error:
        cpu_canny_edges(host)
    with pytest.raises(ValueError) as gpu_error:
        gpu_canny_edges(cupy_module.asarray(host))

    assert str(gpu_error.value) == str(cpu_error.value)


def _finite_fixture(dtype, shape, *, seed):
    rng = np.random.default_rng(seed + np.dtype(dtype).itemsize * 10_007)
    if np.dtype(dtype) == np.dtype(bool):
        values = rng.random(shape) > 0.58
    elif np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        values = rng.integers(0, info.max + 1, size=shape, dtype=dtype)
        values.flat[:4] = (0, info.max, info.max // 2, 0)
    else:
        values = rng.normal(0.25, 2.5, size=shape).astype(dtype)
        values.flat[:5] = (-8.5, 0.0, 0.0, 3.25, 12.0)
    return values


def _structured_fixture(shape):
    values = np.zeros(shape, dtype=np.float32)
    values[..., shape[-2] // 4 : 3 * shape[-2] // 4, shape[-1] // 5 :] = 0.6
    values[..., shape[-2] // 3 :, shape[-1] // 3 : 2 * shape[-1] // 3] = 1.0
    if values.size:
        values.flat[0] = 0.25
        values.flat[-1] = 0.75
    return values
