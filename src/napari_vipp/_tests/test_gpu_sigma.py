from __future__ import annotations

import importlib
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from napari_vipp.core.gpu import cupy_sigma
from napari_vipp.core.operations import sigma_filter as cpu_sigma_filter
from napari_vipp.core.progress import OperationCancelled, ProgressContext


class _FakeStream:
    def __init__(self) -> None:
        self.synchronizations = 0

    def synchronize(self) -> None:
        self.synchronizations += 1


class _FakeRawKernel:
    def __init__(self, owner, name: str, options: tuple[str, ...]) -> None:
        self.owner = owner
        self.name = name
        self.options = tuple(options)

    def __call__(self, grid, block, arguments) -> None:
        self.owner.launches.append((self.name, grid, block))
        if self.name == "vipp_sigma_validate_float32":
            values, size, square_limit, status = arguments
            flattened = np.asarray(values).reshape(-1)[: int(size)]
            finite = np.isfinite(flattened)
            if not finite.all():
                status[0] |= np.uint32(1)
            if np.any(finite & (np.abs(flattened) > np.float32(square_limit))):
                status[0] |= np.uint32(2)
            return

        (
            source,
            rows,
            columns,
            row_start,
            row_stop,
            offsets,
            footprint_count,
            sigma_width,
            minimum_count,
            outlier_aware,
            output,
        ) = arguments
        source = np.asarray(source, dtype=np.float32)
        output = np.asarray(output)
        rows = int(rows)
        columns = int(columns)
        row_start = int(row_start)
        row_stop = int(row_stop)
        offsets = np.asarray(offsets, dtype=np.int32).reshape(-1, 2)
        footprint_count = int(footprint_count)
        sigma_width = float(sigma_width)
        minimum_count = int(minimum_count)
        outlier_aware = bool(outlier_aware)

        for y in range(row_start, row_stop):
            for x in range(columns):
                samples: list[np.float32] = []
                full_sum = 0.0
                full_sum_squared = 0.0
                for dy, dx in offsets[:footprint_count]:
                    yy = min(max(y + int(dy), 0), rows - 1)
                    xx = min(max(x + int(dx), 0), columns - 1)
                    sample = np.float32(source[yy, xx])
                    samples.append(sample)
                    full_sum += float(sample)
                    full_sum_squared += float(np.float32(sample * sample))
                mean = full_sum / footprint_count
                variance = full_sum_squared / footprint_count - mean * mean
                variance = max(variance, 0.0)
                spread = sigma_width * math.sqrt(variance)
                center = float(source[y, x])
                lower = center - spread
                upper = center + spread
                selected = [
                    float(sample)
                    for sample in samples
                    if lower <= float(sample) <= upper
                ]
                if len(selected) >= minimum_count:
                    filtered = sum(selected) / len(selected)
                elif outlier_aware:
                    filtered = (full_sum - center) / (footprint_count - 1)
                else:
                    filtered = mean
                if self.name == "vipp_sigma_filter_float32":
                    output[y, x] = np.float32(filtered)
                else:
                    maximum = 255 if self.name.endswith("uint8") else 65_535
                    fiji_value = np.float32(filtered)
                    rounded = np.floor(fiji_value + np.float32(0.5))
                    output[y, x] = np.clip(rounded, 0, maximum)


class _FakeCupy:
    float32 = np.float32
    uint32 = np.uint32

    def __init__(self) -> None:
        self.stream = _FakeStream()
        self.cuda = SimpleNamespace(get_current_stream=lambda: self.stream)
        self.kernels: list[_FakeRawKernel] = []
        self.launches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    asarray = staticmethod(np.asarray)
    ascontiguousarray = staticmethod(np.ascontiguousarray)
    transpose = staticmethod(np.transpose)
    empty = staticmethod(np.empty)
    zeros = staticmethod(np.zeros)

    def RawKernel(self, _source, name: str, *, options=()):
        kernel = _FakeRawKernel(self, name, tuple(options))
        self.kernels.append(kernel)
        return kernel


@pytest.fixture
def fake_cupy(monkeypatch):
    cupy = _FakeCupy()
    real_import = importlib.import_module

    def load(name: str):
        return cupy if name == "cupy" else real_import(name)

    _clear_provider_caches()
    monkeypatch.setattr(cupy_sigma.importlib, "import_module", load)
    yield cupy
    _clear_provider_caches()


@pytest.fixture(scope="module")
def real_cupy():
    try:
        cupy = importlib.import_module("cupy")
        if int(cupy.cuda.runtime.getDeviceCount()) < 1:
            pytest.skip("CuPy reports no CUDA device.")
        probe = cupy.arange(4, dtype=cupy.float32)
        cupy.cuda.get_current_stream().synchronize()
        del probe
    except Exception as exc:
        pytest.skip(f"A working CuPy CUDA runtime is unavailable: {exc}")
    return cupy


def _clear_provider_caches() -> None:
    cupy_sigma._cupy_module.cache_clear()
    cupy_sigma._validation_kernel.cache_clear()
    cupy_sigma._sigma_filter_kernel.cache_clear()


def test_import_is_safe_without_cupy_or_cupyx() -> None:
    source_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(source_root), environment.get("PYTHONPATH", "")))
    )
    code = r"""
import builtins
import importlib
import sys

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "cupy" or name.startswith("cupyx"):
        raise AssertionError(f"optional CUDA import attempted: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
module = importlib.import_module("napari_vipp.core.gpu.cupy_sigma")
assert callable(module.sigma_filter)
assert module.__all__ == ["sigma_filter"]
assert "cupy" not in sys.modules
assert not any(name.startswith("cupyx") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.float32))
@pytest.mark.parametrize(
    ("radius", "sigma_width", "minimum_fraction", "outlier_aware"),
    (
        (0.5, 0.0, 0.2, False),
        (1.5, 1.0, 0.8, True),
        (2.0, 2.0, 0.2, True),
        (2.5, 3.0, 1.0, False),
        (10.0, 2.0, 0.0, True),
    ),
)
def test_fake_provider_matches_cpu_across_numeric_contract(
    fake_cupy,
    dtype,
    radius,
    sigma_width,
    minimum_fraction,
    outlier_aware,
) -> None:
    source = np.asarray(
        [
            [0, 0, 1, 3, 8, 13, 21],
            [0, 2, 4, 6, 8, 10, 12],
            [1, 4, 9, 250, 9, 4, 1],
            [2, 6, 10, 14, 10, 6, 2],
            [50, 12, 8, 4, 0, 4, 8],
        ],
        dtype=dtype,
    )
    before = source.copy()
    kwargs = {
        "radius": radius,
        "sigma_width": sigma_width,
        "minimum_pixel_fraction": minimum_fraction,
        "outlier_aware": outlier_aware,
    }

    expected = cpu_sigma_filter(source, **kwargs)
    actual = cupy_sigma.sigma_filter(source, **kwargs)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, before)
    assert actual.dtype == source.dtype
    assert actual.flags.c_contiguous


@pytest.mark.parametrize("channel_axis", (0, 1, -1))
def test_fake_provider_preserves_arbitrary_channel_axes(
    fake_cupy,
    channel_axis,
) -> None:
    canonical = np.arange(3 * 7 * 9, dtype=np.uint16).reshape(3, 7, 9)
    source = np.moveaxis(canonical, 0, channel_axis)

    expected = cpu_sigma_filter(source, radius=1.75, channel_axis=channel_axis)
    actual = cupy_sigma.sigma_filter(source, radius=1.75, channel_axis=channel_axis)

    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == source.shape
    assert actual.flags.c_contiguous


def test_fake_provider_handles_noncontiguous_stacks_without_mutation(fake_cupy) -> None:
    owner = np.arange(2 * 3 * 10 * 12, dtype=np.float32).reshape(2, 3, 10, 12)
    source = owner[:, :, ::2, 1::2].transpose(2, 1, 0, 3)
    before = owner.copy()

    expected = cpu_sigma_filter(source, radius=1.0, channel_axis=1)
    actual = cupy_sigma.sigma_filter(source, radius=1.0, channel_axis=1)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(owner, before)
    assert not source.flags.c_contiguous
    assert actual.flags.c_contiguous


def test_fake_provider_uses_exact_cuda_arithmetic_options(fake_cupy) -> None:
    cupy_sigma.sigma_filter(
        np.arange(25, dtype=np.float32).reshape(5, 5),
        radius=0.5,
    )

    assert {kernel.name for kernel in fake_cupy.kernels} == {
        "vipp_sigma_validate_float32",
        "vipp_sigma_filter_float32",
    }
    for kernel in fake_cupy.kernels:
        assert "--fmad=false" in kernel.options
        assert "--ftz=false" in kernel.options
        assert "--prec-div=true" in kernel.options
        assert "--prec-sqrt=true" in kernel.options


def test_fake_progress_is_tiled_synchronized_and_truthful(fake_cupy) -> None:
    updates = []
    cupy_sigma.sigma_filter(
        np.ones((2, 130, 5), dtype=np.uint8),
        radius=0.5,
        progress=ProgressContext(reporter=updates.append),
    )

    assert [(update.current, update.total) for update in updates] == [
        (current, 6) for current in range(7)
    ]
    assert {update.message for update in updates} == {"Sigma Filter rows"}
    launches = [launch for launch in fake_cupy.launches if "filter_uint8" in launch[0]]
    assert len(launches) == 6
    # Every tile is synchronized before its progress milestone, and one final
    # synchronization covers restoration to the caller's contiguous axis order.
    assert fake_cupy.stream.synchronizations == 7


def test_fake_cancellation_stops_before_the_next_row_tile(fake_cupy) -> None:
    updates = []
    progress = ProgressContext(
        cancelled=lambda: len(updates) >= 2,
        reporter=updates.append,
    )

    with pytest.raises(OperationCancelled):
        cupy_sigma.sigma_filter(
            np.ones((130, 7), dtype=np.uint16),
            radius=10.0,
            progress=progress,
        )

    launches = [launch for launch in fake_cupy.launches if "filter_uint16" in launch[0]]
    assert len(launches) == 1
    assert [(update.current, update.total) for update in updates] == [(0, 3), (1, 3)]
    assert fake_cupy.stream.synchronizations == 1


@pytest.mark.parametrize(
    ("source", "kwargs", "match"),
    (
        (np.ones((3, 3), dtype=bool), {}, "uint8, uint16, and float32"),
        (np.ones((3, 3), dtype=np.float64), {}, "uint8, uint16, and float32"),
        (np.ones((3, 3), dtype=">u2"), {}, "received >u2"),
        (np.ones((3, 3), dtype=">f4"), {}, "received >f4"),
        (np.ones((3, 3), dtype=np.uint8), {"radius": 0.49}, "radius"),
        (np.ones((3, 3), dtype=np.uint8), {"sigma_width": -1}, "sigma_width"),
        (
            np.ones((3, 3), dtype=np.uint8),
            {"minimum_pixel_fraction": 1.01},
            "minimum_pixel_fraction",
        ),
        (np.ones((3, 3), dtype=np.uint8), {"outlier_aware": 1}, "outlier_aware"),
        (np.ones((3, 3), dtype=np.uint8), {"channel_axis": True}, "channel_axis"),
    ),
)
def test_fake_provider_rejects_invalid_contract(
    fake_cupy,
    source,
    kwargs,
    match,
) -> None:
    with pytest.raises(ValueError, match=match):
        cupy_sigma.sigma_filter(source, **kwargs)


@pytest.mark.parametrize("nonfinite", (math.nan, math.inf, -math.inf))
def test_fake_provider_rejects_nonfinite_float32(fake_cupy, nonfinite) -> None:
    source = np.ones((3, 3), dtype=np.float32)
    source[1, 1] = nonfinite

    with pytest.raises(ValueError, match="finite image intensities"):
        cupy_sigma.sigma_filter(source)


def test_fake_provider_rejects_float32_square_overflow(fake_cupy) -> None:
    safe_limit = np.float32(math.sqrt(float(np.finfo(np.float32).max)))
    source = np.full((3, 3), safe_limit, dtype=np.float32)
    np.testing.assert_array_equal(
        cupy_sigma.sigma_filter(source, radius=0.5),
        source,
    )
    source[1, 1] = np.nextafter(safe_limit, np.float32(np.inf))

    with pytest.raises(ValueError, match="square workspace"):
        cupy_sigma.sigma_filter(source)


@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.float32))
@pytest.mark.parametrize("radius", (0.5, 1.5, 2.0, 2.5, 5.0, 10.0))
def test_real_cuda_matches_cpu_exactly_on_structured_and_random_planes(
    real_cupy,
    dtype,
    radius,
) -> None:
    rng = np.random.default_rng(50_900 + int(radius * 100))
    if dtype is np.float32:
        source = rng.uniform(-100.0, 4096.0, size=(33, 37)).astype(dtype)
    elif dtype is np.uint16:
        source = rng.integers(0, 65_536, size=(33, 37), dtype=dtype)
    else:
        source = rng.integers(0, 256, size=(33, 37), dtype=dtype)
    source[0, 0] = 0
    source[0, 1] = np.iinfo(dtype).max if dtype is not np.float32 else -0.0
    kwargs = {
        "radius": radius,
        "sigma_width": 2.0,
        "minimum_pixel_fraction": 0.2,
        "outlier_aware": True,
    }

    expected = cpu_sigma_filter(source, **kwargs)
    device_source = real_cupy.asarray(source)
    before = device_source.copy()
    actual = cupy_sigma.sigma_filter(device_source, **kwargs)
    host_actual = real_cupy.asnumpy(actual)

    np.testing.assert_array_equal(host_actual, expected)
    assert isinstance(actual, real_cupy.ndarray)
    assert actual.dtype == source.dtype
    assert actual.flags.c_contiguous
    assert bool(real_cupy.array_equal(device_source, before).item())


def test_real_cuda_matches_branch_sensitive_fallback_and_rounding(real_cupy) -> None:
    values = np.asarray(
        [
            [0, 0, 1, 3, 8, 13, 21],
            [0, 2, 4, 6, 8, 10, 12],
            [1, 4, 9, 250, 9, 4, 1],
            [2, 6, 10, 14, 10, 6, 2],
            [50, 12, 8, 4, 0, 4, 8],
        ],
        dtype=np.uint16,
    )
    for sigma_width, minimum_fraction, outlier_aware in (
        (0.0, 0.2, False),
        (0.0, 1.0, True),
        (1.0, 0.8, True),
        (3.0, 1.0, False),
    ):
        kwargs = {
            "radius": 2.0,
            "sigma_width": sigma_width,
            "minimum_pixel_fraction": minimum_fraction,
            "outlier_aware": outlier_aware,
        }
        expected = cpu_sigma_filter(values, **kwargs)
        actual = cupy_sigma.sigma_filter(real_cupy.asarray(values), **kwargs)
        np.testing.assert_array_equal(real_cupy.asnumpy(actual), expected)

    half_up = np.zeros((3, 3), dtype=np.uint8)
    half_up[1, 1] = 10
    half_up[1, 0] = half_up[1, 2] = 1
    expected = cpu_sigma_filter(
        half_up,
        radius=0.5,
        sigma_width=0.0,
        minimum_pixel_fraction=1.0,
        outlier_aware=True,
    )
    actual = cupy_sigma.sigma_filter(
        real_cupy.asarray(half_up),
        radius=0.5,
        sigma_width=0.0,
        minimum_pixel_fraction=1.0,
        outlier_aware=True,
    )
    np.testing.assert_array_equal(real_cupy.asnumpy(actual), expected)
    assert int(actual[1, 1].item()) == 1


def test_real_cuda_preserves_subnormal_samples_and_float32_squares(
    real_cupy,
) -> None:
    smallest = np.nextafter(np.float32(0.0), np.float32(1.0))
    sources = (
        np.asarray(
            [
                [0.0, smallest, -smallest],
                [smallest * 2, -smallest * 2, 0.0],
                [smallest * 4, smallest, -smallest],
            ],
            dtype=np.float32,
        ),
        np.asarray(
            [
                [0.0, 1e-20, -1e-20],
                [2e-20, -2e-20, 0.0],
                [4e-20, 1e-20, -1e-20],
            ],
            dtype=np.float32,
        ),
    )
    for source in sources:
        for sigma_width, minimum_fraction, outlier_aware in (
            (0.0, 0.0, False),
            (0.0, 1.0, True),
            (1.0, 0.2, True),
            (3.0, 0.8, False),
        ):
            kwargs = {
                "radius": 2.0,
                "sigma_width": sigma_width,
                "minimum_pixel_fraction": minimum_fraction,
                "outlier_aware": outlier_aware,
            }
            expected = cpu_sigma_filter(source, **kwargs)
            actual = cupy_sigma.sigma_filter(real_cupy.asarray(source), **kwargs)
            host_actual = real_cupy.asnumpy(actual)
            np.testing.assert_array_equal(host_actual, expected)
            np.testing.assert_array_equal(
                np.signbit(host_actual),
                np.signbit(expected),
            )


def test_real_cuda_matches_cpu_for_noncontiguous_arbitrary_axis_stack(
    real_cupy,
) -> None:
    rng = np.random.default_rng(509)
    owner = rng.integers(0, 65_536, size=(2, 4, 38, 42), dtype=np.uint16)
    host_source = owner[:, :, ::2, 1::2].transpose(2, 1, 0, 3)
    device_owner = real_cupy.asarray(owner)
    device_source = device_owner[:, :, ::2, 1::2].transpose(2, 1, 0, 3)

    expected = cpu_sigma_filter(host_source, radius=2.85, channel_axis=1)
    actual = cupy_sigma.sigma_filter(
        device_source,
        radius=2.85,
        channel_axis=1,
    )

    np.testing.assert_array_equal(real_cupy.asnumpy(actual), expected)
    assert not device_source.flags.c_contiguous
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(real_cupy.asnumpy(device_owner), owner)


def test_real_cuda_progress_and_cancellation_are_row_tile_boundaries(real_cupy) -> None:
    updates = []
    source = real_cupy.ones((130, 17), dtype=real_cupy.uint16)
    result = cupy_sigma.sigma_filter(
        source,
        radius=0.5,
        progress=ProgressContext(reporter=updates.append),
    )

    assert [(update.current, update.total) for update in updates] == [
        (0, 3),
        (1, 3),
        (2, 3),
        (3, 3),
    ]
    np.testing.assert_array_equal(real_cupy.asnumpy(result), np.ones((130, 17)))

    cancelled_updates = []
    with pytest.raises(OperationCancelled):
        cupy_sigma.sigma_filter(
            source,
            radius=10.0,
            progress=ProgressContext(
                cancelled=lambda: len(cancelled_updates) >= 2,
                reporter=cancelled_updates.append,
            ),
        )
    assert [(item.current, item.total) for item in cancelled_updates] == [
        (0, 3),
        (1, 3),
    ]


def test_real_cuda_rejects_nonfinite_and_square_unsafe_float32(real_cupy) -> None:
    nonfinite = real_cupy.ones((5, 5), dtype=real_cupy.float32)
    nonfinite[2, 2] = real_cupy.nan
    with pytest.raises(ValueError, match="finite image intensities"):
        cupy_sigma.sigma_filter(nonfinite)

    safe_limit = np.float32(math.sqrt(float(np.finfo(np.float32).max)))
    safe = real_cupy.full((5, 5), safe_limit, dtype=real_cupy.float32)
    safe_result = cupy_sigma.sigma_filter(safe, radius=0.5)
    real_cupy.testing.assert_array_equal(safe_result, safe)

    unsafe = safe.copy()
    unsafe[2, 2] = np.nextafter(safe_limit, np.float32(np.inf))
    with pytest.raises(ValueError, match="square workspace"):
        cupy_sigma.sigma_filter(unsafe)
