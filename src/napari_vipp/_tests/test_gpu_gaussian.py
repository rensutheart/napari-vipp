from __future__ import annotations

import importlib
import subprocess
import sys
from collections.abc import Callable

import numpy as np
import pytest

from napari_vipp.core.gpu import cupy_gaussian as cupy_gaussian_adapter
from napari_vipp.core.gpu.cupy_gaussian import (
    gaussian_blur as cupy_gaussian_blur,
)
from napari_vipp.core.gpu.cupy_gaussian import (
    gaussian_blur_3d as cupy_gaussian_blur_3d,
)
from napari_vipp.core.operations import gaussian_blur, gaussian_blur_3d

FLOAT32_NRMSE_LIMIT = 2e-6


@pytest.fixture(scope="module")
def real_cupy():
    try:
        cupy = importlib.import_module("cupy")
        importlib.import_module("cupyx.scipy.ndimage")
        if int(cupy.cuda.runtime.getDeviceCount()) < 1:
            pytest.skip("CuPy reports no CUDA device.")
        probe = cupy.arange(4, dtype=cupy.float32)
        cupy.cuda.get_current_stream().synchronize()
        del probe
    except Exception as exc:
        pytest.skip(f"A working CuPy CUDA runtime is unavailable: {exc}")
    return cupy


def test_gaussian_adapter_import_does_not_discover_or_import_cupy():
    script = r"""
import importlib.abc
import sys

class BlockCuPy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if fullname == "cupy" or fullname.startswith("cupy."):
            raise ImportError("CuPy import was blocked")
        if fullname == "cupyx" or fullname.startswith("cupyx."):
            raise ImportError("CuPyX import was blocked")
        return None

sys.meta_path.insert(0, BlockCuPy())
import napari_vipp.core.gpu.cupy_gaussian as adapter
assert adapter.__all__ == ["gaussian_blur", "gaussian_blur_3d"]
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


@pytest.mark.parametrize("radius", (1, 5, 17, 48))
def test_dynamic_gaussian_weights_are_symmetric_normalized_float32(radius):
    sigma = radius / 4.0

    weights = cupy_gaussian_adapter._gaussian_weights(sigma, radius)

    assert weights.dtype == np.dtype(np.float32)
    assert weights.shape == (2 * radius + 1,)
    np.testing.assert_array_equal(weights, weights[::-1])
    assert np.isclose(weights.sum(dtype=np.float64), 1.0, atol=1e-7)
    assert np.all(weights > 0)


def _image(dtype: np.dtype, shape: tuple[int, ...], *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        values = rng.integers(info.min, info.max + 1, size=shape, dtype=dtype)
        values.flat[0] = info.max
        values.flat[-1] = info.min
    else:
        values = rng.normal(0.0, 3.0, size=shape).astype(dtype)
        values.flat[0] = np.float32(31.25)
        values.flat[-1] = np.float32(-17.5)
    values.setflags(write=False)
    return values


def _assert_integer_exact(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    region: str,
) -> None:
    mismatch = reference != candidate
    count = int(np.count_nonzero(mismatch))
    if not count:
        return
    locations = np.argwhere(mismatch)
    sample = []
    for location in locations[:8]:
        index = tuple(int(value) for value in location)
        sample.append(
            f"{index}: cpu={int(reference[index])}, gpu={int(candidate[index])}"
        )
    difference = np.abs(reference.astype(np.int64) - candidate.astype(np.int64))
    raise AssertionError(
        f"Gaussian integer parity failed in {region}: {count}/{reference.size} "
        f"values differ, max_abs={int(difference.max())}; " + "; ".join(sample)
    )


def _assert_float32_gate(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    region: str,
) -> None:
    reference64 = reference.astype(np.float64)
    candidate64 = candidate.astype(np.float64)
    difference = candidate64 - reference64
    max_abs = float(np.max(np.abs(difference), initial=0.0))
    peak = float(np.max(np.abs(reference64), initial=0.0))
    max_abs_limit = 1e-6 + 5e-6 * peak
    denominator = max(
        float(np.linalg.norm(reference64.ravel())),
        float(np.sqrt(reference.size) * 1e-12),
    )
    nrmse = float(np.linalg.norm(difference.ravel()) / denominator)
    assert nrmse <= FLOAT32_NRMSE_LIMIT, (
        f"Gaussian float32 NRMSE failed in {region}: "
        f"{nrmse:.9g} > {FLOAT32_NRMSE_LIMIT:.9g}"
    )
    assert max_abs <= max_abs_limit, (
        f"Gaussian float32 local error failed in {region}: "
        f"{max_abs:.9g} > {max_abs_limit:.9g}"
    )


def _assert_device_parity(
    cupy,
    host: np.ndarray,
    *,
    cpu_operation: Callable,
    gpu_operation: Callable,
    region: str,
    kwargs: dict[str, object],
) -> None:
    reference = cpu_operation(host, **kwargs)
    device_input = cupy.asarray(host)
    before = cupy.asnumpy(device_input)
    candidate_device = gpu_operation(device_input, **kwargs)
    cupy.cuda.get_current_stream().synchronize()

    assert isinstance(candidate_device, cupy.ndarray)
    assert candidate_device.device.id == device_input.device.id
    assert candidate_device is not device_input
    np.testing.assert_array_equal(cupy.asnumpy(device_input), before)
    np.testing.assert_array_equal(host, before)

    candidate = cupy.asnumpy(candidate_device)
    assert candidate.shape == reference.shape == host.shape
    assert candidate.dtype == reference.dtype
    if np.issubdtype(reference.dtype, np.integer):
        _assert_integer_exact(reference, candidate, region=region)
    else:
        assert reference.dtype == np.float32
        _assert_float32_gate(reference, candidate, region=region)


GAUSSIAN_2D_CASES = (
    ("yx", (19, 23), None, 1.2),
    ("scalar-trailing-x3", (4, 17, 3), None, 0.8),
    ("scalar-trailing-x4", (4, 17, 4), None, 1.4),
    ("channel-middle", (2, 13, 3, 17), 2, 0.65),
)


@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.float32))
@pytest.mark.parametrize(
    ("case_name", "shape", "channel_axis", "sigma"),
    GAUSSIAN_2D_CASES,
)
def test_gaussian_2d_real_device_parity_across_dtype_rank_and_channel(
    real_cupy,
    dtype,
    case_name: str,
    shape: tuple[int, ...],
    channel_axis: int | None,
    sigma: float,
):
    host = _image(dtype, shape, seed=701 + len(shape) * 19 + int(sigma * 100))
    region = (
        f"2D/{case_name}/{np.dtype(dtype).name}/shape={shape}/"
        f"channel_axis={channel_axis}/sigma={sigma}"
    )
    _assert_device_parity(
        real_cupy,
        host,
        cpu_operation=gaussian_blur,
        gpu_operation=cupy_gaussian_blur,
        region=region,
        kwargs={"sigma": sigma, "channel_axis": channel_axis},
    )


GAUSSIAN_3D_CASES = (
    ("zyx", (7, 13, 15), None),
    ("leading-block", (2, 7, 13, 15), None),
    ("channel-in-tczyx", (2, 3, 7, 13, 15), 1),
    ("trailing-channel-yxc", (13, 15, 3), -1),
)


@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.float32))
@pytest.mark.parametrize(
    ("case_name", "shape", "channel_axis"),
    GAUSSIAN_3D_CASES,
)
def test_gaussian_3d_real_device_anisotropic_parity(
    real_cupy,
    dtype,
    case_name: str,
    shape: tuple[int, ...],
    channel_axis: int | None,
):
    host = _image(dtype, shape, seed=907 + len(shape) * 23)
    kwargs = {
        "sigma_z": 0.6,
        "sigma_y": 1.1,
        "sigma_x": 1.7,
        "lock_xy": False,
        "channel_axis": channel_axis,
    }
    region = (
        f"3D/{case_name}/{np.dtype(dtype).name}/shape={shape}/"
        f"channel_axis={channel_axis}/sigma=(0.6,1.1,1.7)"
    )
    _assert_device_parity(
        real_cupy,
        host,
        cpu_operation=gaussian_blur_3d,
        gpu_operation=cupy_gaussian_blur_3d,
        region=region,
        kwargs=kwargs,
    )


def test_gaussian_3d_real_device_sigma_z_controls_depth_spread(real_cupy):
    host = np.zeros((5, 9, 9), dtype=np.float32)
    host[2, 4, 4] = 1.0

    _assert_device_parity(
        real_cupy,
        host,
        cpu_operation=gaussian_blur_3d,
        gpu_operation=cupy_gaussian_blur_3d,
        region="3D/ZYX/z-only-positive",
        kwargs={
            "sigma_z": 1.0,
            "sigma_y": 0.0,
            "sigma_x": 0.0,
            "lock_xy": False,
        },
    )
    _assert_device_parity(
        real_cupy,
        host,
        cpu_operation=gaussian_blur_3d,
        gpu_operation=cupy_gaussian_blur_3d,
        region="3D/ZYX/z-only-zero",
        kwargs={
            "sigma_z": 0.0,
            "sigma_y": 0.0,
            "sigma_x": 0.0,
            "lock_xy": False,
        },
    )

    positive = gaussian_blur_3d(
        host,
        sigma_z=1.0,
        sigma_y=0.0,
        sigma_x=0.0,
        lock_xy=False,
    )
    zero = gaussian_blur_3d(
        host,
        sigma_z=0.0,
        sigma_y=0.0,
        sigma_x=0.0,
        lock_xy=False,
    )
    assert positive[1, 4, 4] > 0.0
    assert positive[3, 4, 4] > 0.0
    np.testing.assert_array_equal(zero, host)


@pytest.mark.parametrize("sigma", (0.0, 12.0))
def test_gaussian_2d_float32_public_sigma_boundaries(real_cupy, sigma):
    host = _image(np.float32, (31, 37), seed=1217 + int(sigma))

    _assert_device_parity(
        real_cupy,
        host,
        cpu_operation=gaussian_blur,
        gpu_operation=cupy_gaussian_blur,
        region=f"2D/public-boundary/float32/sigma={sigma}",
        kwargs={"sigma": sigma},
    )


def test_all_public_float32_radii_reuse_one_compiled_kernel_signature(real_cupy):
    device = real_cupy.arange(
        3 * 5 * 7,
        dtype=real_cupy.float32,
    ).reshape(3, 5, 7)
    kernel = cupy_gaussian_adapter._float32_gaussian_axis_kernel()

    assert isinstance(kernel, real_cupy.RawKernel)
    first = cupy_gaussian_blur(device, sigma=0.25)
    real_cupy.cuda.get_current_stream().synchronize()
    compiled_pointer = kernel.kernel.ptr
    last = first
    for radius in range(2, 49):
        last = cupy_gaussian_blur(device, sigma=radius / 4.0)
        real_cupy.cuda.get_current_stream().synchronize()
        assert cupy_gaussian_adapter._float32_gaussian_axis_kernel() is kernel
        assert kernel.kernel.ptr == compiled_pointer

    assert kernel.name == "vipp_dynamic_float32_gaussian_axis_v2"
    assert "const int radius" in kernel.code
    assert first.shape == last.shape == device.shape
    assert not real_cupy.array_equal(first, last)


def test_public_float32_execution_does_not_import_cupyx(real_cupy):
    del real_cupy
    script = r"""
import importlib.abc
import sys

import cupy
cupyx_modules_before = {
    name for name in sys.modules if name == "cupyx" or name.startswith("cupyx.")
}

class BlockCuPyX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if fullname == "cupyx" or fullname.startswith("cupyx."):
            raise ImportError("CuPyX import was blocked")
        return None

sys.meta_path.insert(0, BlockCuPyX())
from napari_vipp.core.gpu.cupy_gaussian import gaussian_blur

source = cupy.arange(35, dtype=cupy.float32).reshape(5, 7)
result = gaussian_blur(source, sigma=1.25)
cupy.cuda.get_current_stream().synchronize()
assert isinstance(result, cupy.ndarray)
assert result.shape == source.shape
cupyx_modules_after = {
    name for name in sys.modules if name == "cupyx" or name.startswith("cupyx.")
}
assert cupyx_modules_after == cupyx_modules_before
assert "cupyx.scipy.ndimage" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("shape", "channel_axis", "sigma"),
    (
        ((19, 23), None, 1.2),
        ((4, 17, 3), None, 0.8),
        ((2, 13, 3, 17), 2, 0.65),
        ((12, 96, 128), None, 8.0),
        ((12, 96, 128), None, 12.0),
    ),
)
def test_dynamic_float32_kernel_is_bitwise_equivalent_to_cupyx(
    real_cupy,
    shape,
    channel_axis,
    sigma,
):
    ndimage = importlib.import_module("cupyx.scipy.ndimage")
    host = _image(np.float32, shape, seed=1501 + len(shape) + int(sigma * 100))
    device = real_cupy.asarray(host)
    normalized_channel = None if channel_axis is None else channel_axis % len(shape)
    spatial_axes = [axis for axis in range(len(shape)) if axis != normalized_channel]
    sigma_by_axis = [0.0] * len(shape)
    for axis in spatial_axes[-2:]:
        sigma_by_axis[axis] = sigma

    reference = ndimage.gaussian_filter(
        device,
        sigma=tuple(sigma_by_axis),
        mode="reflect",
    )
    candidate = cupy_gaussian_blur(
        device,
        sigma=sigma,
        channel_axis=channel_axis,
    )
    real_cupy.cuda.get_current_stream().synchronize()

    np.testing.assert_array_equal(
        real_cupy.asnumpy(candidate),
        real_cupy.asnumpy(reference),
    )


def test_dynamic_float32_zero_radius_preserves_exact_input_bits(real_cupy):
    ndimage = importlib.import_module("cupyx.scipy.ndimage")
    input_bits = np.array(
        (
            0x80000000,  # negative zero
            0x00000000,  # positive zero
            0x7F800001,  # signalling NaN payload
            0xFF812345,  # negative NaN payload
        ),
        dtype=np.uint32,
    ).reshape(2, 2)
    host = input_bits.view(np.float32)
    device = real_cupy.asarray(host)

    reference = ndimage.gaussian_filter(device, sigma=0.1, mode="reflect")
    candidate = cupy_gaussian_blur(device, sigma=0.1)
    real_cupy.cuda.get_current_stream().synchronize()

    reference_bits = real_cupy.asnumpy(reference).view(np.uint32)
    candidate_bits = real_cupy.asnumpy(candidate).view(np.uint32)
    np.testing.assert_array_equal(reference_bits, input_bits)
    np.testing.assert_array_equal(candidate_bits, input_bits)
    assert candidate.data.ptr != device.data.ptr


@pytest.mark.parametrize(
    ("shape", "channel_axis"),
    (
        ((7, 13, 15), None),
        ((2, 3, 7, 13, 15), 1),
        ((13, 15, 3), -1),
    ),
)
def test_dynamic_float32_3d_kernel_is_bitwise_equivalent_to_cupyx(
    real_cupy,
    shape,
    channel_axis,
):
    ndimage = importlib.import_module("cupyx.scipy.ndimage")
    host = _image(np.float32, shape, seed=1601 + len(shape))
    device = real_cupy.asarray(host)
    normalized_channel = None if channel_axis is None else channel_axis % len(shape)
    spatial_axes = [axis for axis in range(len(shape)) if axis != normalized_channel]
    sigmas = (0.6, 1.1, 1.7)
    sigma_by_axis = [0.0] * len(shape)
    active_axes = spatial_axes[-3:]
    for axis, sigma in zip(
        active_axes,
        sigmas[-len(active_axes) :],
        strict=True,
    ):
        sigma_by_axis[axis] = sigma

    reference = ndimage.gaussian_filter(
        device,
        sigma=tuple(sigma_by_axis),
        mode="reflect",
    )
    candidate = cupy_gaussian_blur_3d(
        device,
        sigma_z=sigmas[0],
        sigma_y=sigmas[1],
        sigma_x=sigmas[2],
        lock_xy=False,
        channel_axis=channel_axis,
    )
    real_cupy.cuda.get_current_stream().synchronize()

    np.testing.assert_array_equal(
        real_cupy.asnumpy(candidate),
        real_cupy.asnumpy(reference),
    )


def test_dynamic_float32_kernel_handles_noncontiguous_tiny_spatial_axes(
    real_cupy,
):
    host_base = _image(np.float32, (4, 5, 7, 9), seed=1701).copy()
    host_base[0, 0, 0, 0] = np.nextafter(np.float32(0.0), np.float32(1.0))
    host_base[0, 0, 0, 1] = np.float32(-0.0)
    host_base[0, 0, 0, 2] = np.finfo(np.float32).max / np.float32(256.0)
    device_base = real_cupy.asarray(host_base)
    device_view = device_base[:, ::2, :2, ::-2]
    before = real_cupy.asnumpy(device_view)
    assert not device_view.flags.c_contiguous

    candidate_device = cupy_gaussian_blur(device_view, sigma=12.0)
    real_cupy.cuda.get_current_stream().synchronize()
    candidate = real_cupy.asnumpy(candidate_device)
    reference = gaussian_blur(before, sigma=12.0)

    assert candidate_device.flags.c_contiguous
    assert candidate_device.data.ptr != device_view.data.ptr
    np.testing.assert_array_equal(real_cupy.asnumpy(device_view), before)
    _assert_float32_gate(
        reference,
        candidate,
        region="dynamic/noncontiguous/tiny-spatial-axes/sigma=12",
    )


@pytest.mark.parametrize(
    ("shape", "sigmas"),
    (
        ((5, 17, 19), (12.0, 0.0, 12.0)),
        ((2, 5, 17, 19), (0.0, 12.0, 12.0)),
    ),
)
def test_gaussian_3d_float32_public_anisotropic_boundaries(
    real_cupy,
    shape,
    sigmas,
):
    host = _image(np.float32, shape, seed=1291 + len(shape))
    sigma_z, sigma_y, sigma_x = sigmas

    _assert_device_parity(
        real_cupy,
        host,
        cpu_operation=gaussian_blur_3d,
        gpu_operation=cupy_gaussian_blur_3d,
        region=f"3D/public-boundary/float32/shape={shape}/sigma={sigmas}",
        kwargs={
            "sigma_z": sigma_z,
            "sigma_y": sigma_y,
            "sigma_x": sigma_x,
            "lock_xy": False,
        },
    )


@pytest.mark.parametrize(
    ("cpu_operation", "gpu_operation", "kwargs"),
    (
        (gaussian_blur, cupy_gaussian_blur, {"sigma": 0.9}),
        (
            gaussian_blur_3d,
            cupy_gaussian_blur_3d,
            {"sigma_z": 0.4, "sigma_y": 0.8, "sigma_x": 1.2},
        ),
    ),
)
def test_gaussian_bool_input_becomes_resident_float32_with_cpu_parity(
    real_cupy,
    cpu_operation: Callable,
    gpu_operation: Callable,
    kwargs: dict[str, object],
):
    host = np.zeros((5, 9, 11), dtype=bool)
    host[:, 0, 0] = True
    host[2, 4, 5] = True
    host.setflags(write=False)
    _assert_device_parity(
        real_cupy,
        host,
        cpu_operation=cpu_operation,
        gpu_operation=gpu_operation,
        region=f"bool/{gpu_operation.__name__}",
        kwargs=kwargs,
    )


def test_zero_and_negative_sigma_return_distinct_device_copies(real_cupy):
    host = _image(np.uint16, (7, 9, 11), seed=1103)
    device = real_cupy.asarray(host)

    zero_2d = cupy_gaussian_blur(device, sigma=0.0)
    negative_2d = cupy_gaussian_blur(device, sigma=-2.0)
    tiny_2d = cupy_gaussian_blur(device, sigma=1e-16)
    zero_3d = cupy_gaussian_blur_3d(
        device,
        sigma_z=0.0,
        sigma_y=0.0,
        sigma_x=0.0,
    )

    for result in (zero_2d, negative_2d, tiny_2d, zero_3d):
        assert isinstance(result, real_cupy.ndarray)
        assert result is not device
        assert result.data.ptr != device.data.ptr
        np.testing.assert_array_equal(real_cupy.asnumpy(result), host)
    np.testing.assert_array_equal(real_cupy.asnumpy(device), host)


@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf, "not-a-number"))
def test_gaussian_rejects_nonfinite_sigma(real_cupy, value):
    device = real_cupy.zeros((5, 7, 9), dtype=real_cupy.float32)

    with pytest.raises(ValueError, match="finite number"):
        cupy_gaussian_blur(device, sigma=value)
    with pytest.raises(ValueError, match="finite number"):
        cupy_gaussian_blur_3d(device, sigma_z=value)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "CuPy 14.1.1 RTX integer parity gap: uint16 5x6 sigma=0.125 has "
        "a sparse one-unit result despite double-precision per-axis intermediates"
    ),
)
def test_report_known_uint16_narrow_sigma_parity_gap(real_cupy):
    host = np.random.default_rng(10).integers(
        0,
        65_536,
        size=(5, 6),
        dtype=np.uint16,
    )
    host.setflags(write=False)
    _assert_device_parity(
        real_cupy,
        host,
        cpu_operation=gaussian_blur,
        gpu_operation=cupy_gaussian_blur,
        region="known-gap/uint16/shape=(5,6)/sigma=0.125/seed=10",
        kwargs={"sigma": 0.125},
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "CuPy 14.1.1 RTX integer parity gap: uint8 2x3x4x5 anisotropic "
        "sigma=(5,0.125,1.1) has sparse one-unit disagreements"
    ),
)
def test_report_known_uint8_anisotropic_parity_gap(real_cupy):
    host = np.random.default_rng(7).integers(
        0,
        256,
        size=(2, 3, 4, 5),
        dtype=np.uint8,
    )
    host.setflags(write=False)
    _assert_device_parity(
        real_cupy,
        host,
        cpu_operation=gaussian_blur_3d,
        gpu_operation=cupy_gaussian_blur_3d,
        region=("known-gap/uint8/shape=(2,3,4,5)/sigma=(5,0.125,1.1)/seed=7"),
        kwargs={"sigma_z": 5.0, "sigma_y": 0.125, "sigma_x": 1.1},
    )
