from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from napari_vipp.core.gpu import cupy_binary_threshold as provider
from napari_vipp.core.operations import binary_threshold as cpu_binary_threshold

SOURCE_ROOT = Path(__file__).resolve().parents[2]


class _FakeStream:
    def __init__(self) -> None:
        self.synchronize_count = 0
        self.on_synchronize = None

    def synchronize(self) -> None:
        self.synchronize_count += 1
        if self.on_synchronize is not None:
            self.on_synchronize()


class _FakeCupy:
    def __init__(self, stream: _FakeStream) -> None:
        self.cuda = SimpleNamespace(get_current_stream=lambda: stream)
        self.asarray_sizes: list[int] = []

    def asarray(self, value):
        result = np.asarray(value)
        self.asarray_sizes.append(int(result.size))
        return result

    @staticmethod
    def asnumpy(_value):
        raise AssertionError("Binary Threshold must not transfer an image to host.")

    @staticmethod
    def isfinite(_value):
        raise AssertionError("Binary Threshold must not scan input finiteness.")


@dataclass
class _Progress:
    cancel_after_report: int | None = None
    cancelled: bool = False
    reports: list[tuple[int, int, str]] = field(default_factory=list)

    def report(self, current: int, total: int, message: str) -> None:
        self.reports.append((current, total, message))
        if current == self.cancel_after_report:
            self.cancelled = True

    def check_cancelled(self) -> None:
        if self.cancelled:
            raise RuntimeError("cancelled")


@pytest.fixture
def fake_cupy(monkeypatch):
    stream = _FakeStream()
    cupy = _FakeCupy(stream)
    monkeypatch.setattr(provider, "_cupy_module", lambda: cupy)
    return cupy, stream


def test_provider_module_import_is_gpu_lazy_in_a_fresh_process() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(SOURCE_ROOT), environment.get("PYTHONPATH", "")))
    )
    script = r"""
import builtins
import importlib
import sys

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "cupy" or name.startswith(("cupyx", "cucim")):
        raise AssertionError(f"optional CUDA import attempted: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
module = importlib.import_module("napari_vipp.core.gpu.cupy_binary_threshold")
assert callable(module.binary_threshold)
assert module.__all__ == ["binary_threshold"]
for name in ("cupy", "cupyx", "cucim"):
    assert name not in sys.modules, name
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "threshold",
    (
        -0.0,
        0.0,
        0.1,
        0.5,
        1.0,
        1.00000008,
        float(np.nextafter(np.float32(1.0), np.float32(2.0))),
    ),
)
def test_fake_provider_is_bitwise_exact_for_scalar_float32_region(
    fake_cupy,
    threshold,
) -> None:
    del fake_cupy
    values = np.asarray(
        [
            -np.inf,
            -1.0,
            -0.0,
            0.0,
            np.nextafter(np.float32(0.0), np.float32(1.0)),
            0.1,
            0.5,
            1.0,
            np.nextafter(np.float32(1.0), np.float32(2.0)),
            np.inf,
            np.nan,
        ],
        dtype=np.float32,
    )
    before_bits = values.view(np.uint32).copy()
    values.setflags(write=False)

    actual = provider.binary_threshold(values, threshold=threshold)
    expected = cpu_binary_threshold(values, threshold=threshold)

    np.testing.assert_array_equal(actual, expected, strict=True)
    np.testing.assert_array_equal(values.view(np.uint32), before_bits)
    assert not values.flags.writeable
    assert not np.shares_memory(actual, values)


def test_fake_provider_handles_noncontiguous_and_nonfinite_data(fake_cupy):
    del fake_cupy
    base = np.asarray(
        [
            [np.nan, -np.inf, -0.0, 0.0, np.inf, 1.0],
            [2.0, 0.25, 0.5, 0.75, -1.0, np.nan],
            [3.0, -3.0, np.inf, -np.inf, 0.0, -0.0],
        ],
        dtype=np.float32,
    )
    values = base[::-1, 1::2]
    before_bits = base.view(np.uint32).copy()

    actual = provider.binary_threshold(values, threshold=0.5)
    np.testing.assert_array_equal(actual, cpu_binary_threshold(values, 0.5))
    np.testing.assert_array_equal(base.view(np.uint32), before_bits)
    assert actual.dtype == bool


def test_fake_provider_accepts_zero_dimensional_float32_like_cpu(fake_cupy):
    del fake_cupy
    value = np.asarray(0.75, dtype=np.float32)

    actual = provider.binary_threshold(value, threshold=0.5)
    expected = cpu_binary_threshold(value, threshold=0.5)

    np.testing.assert_array_equal(actual, expected, strict=True)
    assert actual.shape == ()


def test_provider_rejects_non_native_float32_before_cupy_upload(fake_cupy):
    del fake_cupy
    values = np.asarray([0.25, 0.75], dtype=np.float32).astype(
        np.dtype(np.float32).newbyteorder("S")
    )

    with pytest.raises(ValueError, match="native-endian float32"):
        provider.binary_threshold(values, threshold=0.5)


@pytest.mark.parametrize("shape", ((0,), (0, 3), (3, 0, 4)))
def test_fake_provider_rejects_empty_data_like_cpu(fake_cupy, shape) -> None:
    del fake_cupy
    values = np.empty(shape, dtype=np.float32)

    with pytest.raises(ValueError, match="requires non-empty image data"):
        cpu_binary_threshold(values)
    with pytest.raises(ValueError, match="requires non-empty image data"):
        provider.binary_threshold(values)


@pytest.mark.parametrize("dtype", (bool, np.uint8, np.uint16, np.float64))
def test_fake_provider_rejects_every_non_float32_input(fake_cupy, dtype) -> None:
    del fake_cupy
    with pytest.raises(ValueError, match="requires float32 image data"):
        provider.binary_threshold(np.zeros((3, 4), dtype=dtype))


@pytest.mark.parametrize("threshold", (np.nan, np.inf, -np.inf, None, "banana"))
def test_fake_provider_rejects_nonfinite_or_nonnumeric_thresholds(
    fake_cupy,
    threshold,
) -> None:
    del fake_cupy
    with pytest.raises(ValueError, match="threshold must be a finite number"):
        provider.binary_threshold(
            np.zeros((3, 4), dtype=np.float32),
            threshold=threshold,
        )


@pytest.mark.parametrize("channel_axis", (0, -1, False))
def test_fake_provider_rejects_channel_axis_reduction(fake_cupy, channel_axis) -> None:
    del fake_cupy
    with pytest.raises(ValueError, match="channel_axis must be None"):
        provider.binary_threshold(
            np.zeros((3, 4, 3), dtype=np.float32),
            channel_axis=channel_axis,
        )


def test_fake_provider_has_no_input_scan_or_image_host_transfer(fake_cupy) -> None:
    cupy, stream = fake_cupy
    values = np.asarray([np.nan, -np.inf, -0.0, 0.0, np.inf], dtype=np.float32)

    result = provider.binary_threshold(values, threshold=0.0)

    np.testing.assert_array_equal(result, cpu_binary_threshold(values, 0.0))
    assert cupy.asarray_sizes == [values.size]
    assert stream.synchronize_count == 0


def test_progress_reports_only_synchronized_completion_and_honours_cancellation(
    fake_cupy,
) -> None:
    _cupy, stream = fake_cupy
    values = np.arange(8, dtype=np.float32)
    progress = _Progress()

    output = provider.binary_threshold(values, threshold=3.5, progress=progress)

    np.testing.assert_array_equal(output, values > 3.5)
    assert stream.synchronize_count == 1
    assert progress.reports == [
        (0, 1, "Applying binary threshold"),
        (1, 1, "Applying binary threshold"),
    ]

    cancelled_before = _Progress(cancel_after_report=0)
    with pytest.raises(RuntimeError, match="cancelled"):
        provider.binary_threshold(
            values,
            threshold=3.5,
            progress=cancelled_before,
        )
    assert stream.synchronize_count == 1

    cancelled_after = _Progress()
    stream.on_synchronize = lambda: setattr(cancelled_after, "cancelled", True)
    with pytest.raises(RuntimeError, match="cancelled"):
        provider.binary_threshold(
            values,
            threshold=3.5,
            progress=cancelled_after,
        )
    assert stream.synchronize_count == 2
    assert cancelled_after.reports == [(0, 1, "Applying binary threshold")]

    stream.on_synchronize = None
    recovered = provider.binary_threshold(values, threshold=3.5)
    np.testing.assert_array_equal(recovered, values > 3.5)


def _real_cuda_or_skip():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device is unavailable")
        probe = cupy.asarray([0.0, 1.0], dtype=cupy.float32) > 0.5
        cupy.cuda.get_current_stream().synchronize()
        del probe
    except Exception as exc:  # pragma: no cover - host-specific failure
        pytest.skip(f"CUDA runtime is unavailable: {exc}")
    return cupy


def test_real_cuda_output_is_resident_exact_and_input_is_immutable() -> None:
    cupy = _real_cuda_or_skip()
    host = np.asarray(
        [
            [np.nan, -np.inf, -0.0, 0.0, np.inf],
            [0.1, 0.5, 1.0, np.nextafter(np.float32(1), np.float32(2)), -1.0],
        ],
        dtype=np.float32,
    )
    device = cupy.asarray(host)[:, ::2]
    before = device.copy()

    actual = provider.binary_threshold(device, threshold=1.00000008)

    assert isinstance(actual, cupy.ndarray)
    assert actual.dtype == cupy.bool_
    assert actual.data.ptr != device.data.ptr
    cupy.testing.assert_array_equal(device, before)
    np.testing.assert_array_equal(
        cupy.asnumpy(actual),
        cpu_binary_threshold(host[:, ::2], threshold=1.00000008),
        strict=True,
    )


def test_real_cuda_reported_decimal_thresholds_match_cpu_bitwise() -> None:
    cupy = _real_cuda_or_skip()

    for threshold in (5613.0001, 5613.375, 5613.9999, 17906.348):
        center = np.float32(threshold)
        host = np.asarray(
            [
                np.nextafter(center, np.float32(-np.inf)),
                center,
                np.nextafter(center, np.float32(np.inf)),
            ],
            dtype=np.float32,
        )

        actual = provider.binary_threshold(
            cupy.asarray(host),
            threshold=threshold,
        )

        assert isinstance(actual, cupy.ndarray)
        np.testing.assert_array_equal(
            cupy.asnumpy(actual),
            cpu_binary_threshold(host, threshold=threshold),
            strict=True,
        )
