from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from napari_vipp.core.gpu import cupy_extract_channel as provider
from napari_vipp.core.operations import extract_channel as cpu_extract_channel

SOURCE_ROOT = Path(__file__).resolve().parents[2]


class _FakeCupy:
    @staticmethod
    def asarray(value):
        return np.asarray(value)


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
    cupy = _FakeCupy()
    monkeypatch.setattr(provider, "_cupy_module", lambda: cupy)
    return cupy


def test_provider_module_import_is_gpu_lazy_in_a_fresh_process() -> None:
    script = f"""
import sys
sys.path.insert(0, {str(SOURCE_ROOT)!r})
import napari_vipp.core.gpu.cupy_extract_channel as module
assert module.__all__ == ["extract_channel"]
for name in ("cupy", "cupyx", "cucim"):
    assert name not in sys.modules, name
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("shape", "axis_names", "axis_types", "channel"),
    (
        ((3, 5, 7), ("c", "y", "x"), (), 1),
        ((2, 3, 5, 7), ("t", "c", "y", "x"), (), -1),
        ((2, 3, 4, 5, 7), (), ("time", "channel", "space", "space", "space"), 2),
        ((4, 5, 7, 3), ("z", "y", "x", "rgba"), (), 0),
        ((2, 3, 5, 7), ("c", "z", "y", "x"), ("time", "channel"), 1),
    ),
)
@pytest.mark.parametrize("dtype", (bool, np.uint8, np.uint16, np.float32))
def test_fake_provider_matches_cpu_and_returns_an_allocation_sharing_view(
    fake_cupy,
    shape,
    axis_names,
    axis_types,
    channel,
    dtype,
) -> None:
    del fake_cupy
    values = np.arange(np.prod(shape), dtype=np.uint32).reshape(shape).astype(dtype)
    before = values.copy()

    expected = cpu_extract_channel(
        values,
        channel=channel,
        axis_names=axis_names,
        axis_types=axis_types,
    )
    actual = provider.extract_channel(
        values,
        channel=channel,
        axis_names=axis_names,
        axis_types=axis_types,
    )

    np.testing.assert_array_equal(actual, expected, strict=True)
    np.testing.assert_array_equal(values, before, strict=True)
    assert np.shares_memory(actual, values)


@pytest.mark.parametrize(
    ("axis_names", "axis_types", "channel", "message"),
    (
        (("z", "y", "x"), (), 0, "explicitly declared channel axis"),
        (("c", "y", "x"), (), True, "channel index must be an integer"),
        (("c", "y", "x"), (), 1.0, "channel index must be an integer"),
        (("c", "y", "x"), (), 3, "index 3 is out of range for 3 channels"),
        (("c", "y", "x"), (), -4, "index -1 is out of range for 3 channels"),
    ),
)
def test_fake_provider_matches_authoritative_validation(
    fake_cupy,
    axis_names,
    axis_types,
    channel,
    message,
) -> None:
    del fake_cupy
    values = np.arange(3 * 5 * 7, dtype=np.uint16).reshape(3, 5, 7)

    with pytest.raises(ValueError, match=message):
        provider.extract_channel(
            values,
            channel=channel,
            axis_names=axis_names,
            axis_types=axis_types,
        )
    with pytest.raises(ValueError, match=message):
        cpu_extract_channel(
            values,
            channel=channel,
            axis_names=axis_names,
            axis_types=axis_types,
        )


def test_fake_provider_handles_empty_spatial_axes_and_noncontiguous_input(
    fake_cupy,
) -> None:
    del fake_cupy
    base = np.arange(4 * 6 * 10, dtype=np.uint16).reshape(4, 6, 10)
    values = base[:, ::2, 1::2]
    empty = np.empty((3, 0, 5), dtype=np.float32)
    before = base.copy()

    actual = provider.extract_channel(
        values,
        channel=2,
        axis_names=("c", "y", "x"),
    )
    empty_actual = provider.extract_channel(
        empty,
        channel=1,
        axis_types=("channel", "space", "space"),
    )

    np.testing.assert_array_equal(
        actual,
        cpu_extract_channel(
            values,
            channel=2,
            axis_names=("c", "y", "x"),
        ),
        strict=True,
    )
    assert empty_actual.shape == (0, 5)
    assert empty_actual.dtype == np.float32
    assert np.shares_memory(actual, base)
    # NumPy defines zero-sized arrays as sharing no byte, even though indexing
    # still returns a view whose base retains the original allocation.
    assert empty_actual.base is empty
    np.testing.assert_array_equal(base, before, strict=True)


def test_provider_rejects_non_native_endian_before_cupy_changes_the_dtype(
    fake_cupy,
) -> None:
    del fake_cupy
    values = np.arange(3 * 5 * 7, dtype=np.uint16).reshape(3, 5, 7).astype(
        np.dtype(np.uint16).newbyteorder("S")
    )

    expected = cpu_extract_channel(
        values,
        channel=1,
        axis_names=("c", "y", "x"),
    )
    assert not expected.dtype.isnative
    with pytest.raises(ValueError, match="requires native-endian input"):
        provider.extract_channel(
            values,
            channel=1,
            axis_names=("c", "y", "x"),
        )


@pytest.mark.parametrize("dtype", (np.int16, np.int64, np.float64, np.complex64))
def test_provider_rejects_dtypes_outside_the_reviewed_region(
    fake_cupy,
    dtype,
) -> None:
    del fake_cupy
    values = np.zeros((3, 5, 7), dtype=dtype)

    with pytest.raises(ValueError, match="supports only bool, uint8, uint16"):
        provider.extract_channel(
            values,
            channel=1,
            axis_names=("c", "y", "x"),
        )


def test_progress_has_constant_time_boundaries_and_honours_cancellation(
    fake_cupy,
) -> None:
    del fake_cupy
    values = np.arange(3 * 5 * 7, dtype=np.uint16).reshape(3, 5, 7)
    progress = _Progress()

    actual = provider.extract_channel(
        values,
        channel=1,
        axis_names=("c", "y", "x"),
        progress=progress,
    )

    np.testing.assert_array_equal(actual, values[1], strict=True)
    assert progress.reports == [
        (0, 1, "Extracting channel"),
        (1, 1, "Extracting channel"),
    ]

    cancelled_before_view = _Progress(cancel_after_report=0)
    with pytest.raises(RuntimeError, match="cancelled"):
        provider.extract_channel(
            values,
            channel=1,
            axis_names=("c", "y", "x"),
            progress=cancelled_before_view,
        )
    assert cancelled_before_view.reports == [(0, 1, "Extracting channel")]

    cancelled_before_call = _Progress(cancelled=True)
    with pytest.raises(RuntimeError, match="cancelled"):
        provider.extract_channel(
            values,
            channel=1,
            axis_names=("c", "y", "x"),
            progress=cancelled_before_call,
        )
    assert cancelled_before_call.reports == []


def _real_cuda_or_skip():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device is unavailable")
        cupy.cuda.runtime.getDevice()
    except Exception as exc:  # pragma: no cover - host-specific failure
        pytest.skip(f"CUDA runtime is unavailable: {exc}")
    return cupy


@pytest.mark.parametrize(
    ("shape", "axis_names", "axis_types", "channel"),
    (
        ((3, 31, 37), ("c", "y", "x"), (), 1),
        ((2, 3, 7, 11, 13), (), ("time", "channel", "space", "space", "space"), -1),
        ((2, 7, 11, 13, 3), ("t", "z", "y", "x", "rgba"), (), 2),
    ),
)
def test_real_cuda_output_is_exact_resident_view_without_device_allocation(
    shape,
    axis_names,
    axis_types,
    channel,
) -> None:
    cupy = _real_cuda_or_skip()
    host = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    device = cupy.asarray(host)
    before = device.copy()
    pool = cupy.get_default_memory_pool()
    used_before = pool.used_bytes()

    actual = provider.extract_channel(
        device,
        channel=channel,
        axis_names=axis_names,
        axis_types=axis_types,
    )

    assert isinstance(actual, cupy.ndarray)
    assert actual.data.mem.ptr == device.data.mem.ptr
    assert pool.used_bytes() == used_before
    cupy.testing.assert_array_equal(device, before)
    np.testing.assert_array_equal(
        cupy.asnumpy(actual),
        cpu_extract_channel(
            host,
            channel=channel,
            axis_names=axis_names,
            axis_types=axis_types,
        ),
        strict=True,
    )
