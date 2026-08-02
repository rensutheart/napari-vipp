from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage as ndi

from napari_vipp.core.connected_components import (
    label_connected_components as cpu_label_connected_components,
)
from napari_vipp.core.gpu import cupy_connected_components as provider

SOURCE_ROOT = Path(__file__).resolve().parents[2]


class _FakeStream:
    def __init__(self) -> None:
        self.synchronize_count = 0

    def synchronize(self) -> None:
        self.synchronize_count += 1


class _FakeCupy:
    int32 = np.int32

    def __init__(self, stream: _FakeStream) -> None:
        self.cuda = SimpleNamespace(get_current_stream=lambda: stream)
        self.empty_calls: list[tuple[tuple[int, ...], object]] = []

    @staticmethod
    def asarray(value):
        return value if hasattr(value, "shape") else np.asarray(value)

    def empty(self, shape, dtype):
        normalized_shape = tuple(int(size) for size in shape)
        self.empty_calls.append((normalized_shape, dtype))
        return np.empty(normalized_shape, dtype=dtype)


class _FakeCuPyXNdimage:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def label(self, source, *, structure, output):
        labels, count = ndi.label(np.asarray(source), structure=structure)
        output[...] = labels.astype(np.int32, copy=False)
        self.calls.append(
            {
                "source": np.asarray(source).copy(),
                "structure": np.asarray(structure).copy(),
                "output": output,
                "count": int(count),
            }
        )
        return int(count)


@dataclass
class _Progress:
    cancel_on_check: int | None = None
    cancel_after_completed: int | None = None
    cancelled: bool = False
    checks: int = 0
    reports: list[tuple[int, int, str]] = field(default_factory=list)

    def report(self, current: int, total: int, message: str) -> None:
        self.reports.append((current, total, message))
        if current == self.cancel_after_completed:
            self.cancelled = True

    def check_cancelled(self) -> None:
        self.checks += 1
        if self.cancelled or self.checks == self.cancel_on_check:
            raise RuntimeError("cancelled")


@pytest.fixture
def fake_runtime(monkeypatch):
    stream = _FakeStream()
    cupy = _FakeCupy(stream)
    cupyx_ndimage = _FakeCuPyXNdimage()
    monkeypatch.setattr(provider, "_cupy_module", lambda: cupy)
    monkeypatch.setattr(
        provider,
        "_cupyx_ndimage_module",
        lambda: cupyx_ndimage,
    )
    return cupy, cupyx_ndimage, stream


def test_provider_module_import_is_gpu_lazy_in_a_fresh_process() -> None:
    script = f"""
import sys
sys.path.insert(0, {str(SOURCE_ROOT)!r})
import napari_vipp.core.gpu.cupy_connected_components as module
assert module.__all__ == ["label_connected_components"]
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


@pytest.mark.parametrize("connectivity", ["Face connected", "Full connectivity"])
def test_fake_provider_matches_cpu_and_converts_nonzero_on_device(
    fake_runtime,
    connectivity,
) -> None:
    _cupy, cupyx_ndimage, _stream = fake_runtime
    source = np.zeros((2, 7, 8), dtype=np.float32)
    source[0, 1, 1] = -3.0
    source[0, 2, 2] = np.nan
    source[0, 5:7, 5:7] = np.inf
    source[1, 1:4, 1:4] = 4.0
    source[1, 5, 6] = -1.0
    before = source.copy()

    expected = cpu_label_connected_components(
        source,
        spatial_mode="2D YX",
        connectivity=connectivity,
    )
    actual = provider.label_connected_components(
        source,
        spatial_mode="2D YX",
        connectivity=connectivity,
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, before)
    assert actual.dtype == np.int32
    assert actual.flags.c_contiguous
    assert not np.shares_memory(actual, source)
    assert len(cupyx_ndimage.calls) == 2
    assert all(call["source"].dtype == bool for call in cupyx_ndimage.calls)


def test_fake_provider_writes_leading_blocks_directly_and_restarts_ids(
    fake_runtime,
) -> None:
    _cupy, cupyx_ndimage, _stream = fake_runtime
    mask = np.zeros((2, 3, 6, 7), dtype=bool)
    mask[..., 1, 1] = True
    mask[..., 4, 5] = True
    mask[1, 2, 2:5, 2:5] = True

    actual = provider.label_connected_components(
        mask,
        spatial_mode="2D YX",
        connectivity="Face connected",
    )
    expected = cpu_label_connected_components(
        mask,
        spatial_mode="2D YX",
        connectivity="Face connected",
    )

    np.testing.assert_array_equal(actual, expected)
    assert len(cupyx_ndimage.calls) == 6
    assert all(
        np.shares_memory(call["output"], actual) for call in cupyx_ndimage.calls
    )
    assert all(
        set(np.unique(actual[index])) <= {0, 1, 2}
        for index in np.ndindex(mask.shape[:-2])
    )


def test_fake_progress_and_cancellation_are_complete_block_boundaries(
    fake_runtime,
) -> None:
    _cupy, cupyx_ndimage, stream = fake_runtime
    mask = np.zeros((3, 6, 7), dtype=bool)
    mask[:, 1:3, 1:3] = True
    progress = _Progress()

    provider.label_connected_components(
        mask,
        spatial_mode="2D YX",
        progress=progress,
    )

    assert progress.checks >= 2 * 3
    assert progress.reports == [
        (0, 3, "Connected-component blocks"),
        (1, 3, "Connected-component blocks"),
        (2, 3, "Connected-component blocks"),
        (3, 3, "Connected-component blocks"),
    ]
    assert stream.synchronize_count == 3

    calls_before_cancel = len(cupyx_ndimage.calls)
    syncs_before_cancel = stream.synchronize_count
    cancelled = _Progress(cancel_after_completed=1)
    with pytest.raises(RuntimeError, match="cancelled"):
        provider.label_connected_components(
            mask,
            spatial_mode="2D YX",
            progress=cancelled,
        )
    assert cancelled.reports == [
        (0, 3, "Connected-component blocks"),
        (1, 3, "Connected-component blocks"),
    ]
    assert len(cupyx_ndimage.calls) - calls_before_cancel == 1
    assert stream.synchronize_count - syncs_before_cancel == 1


def test_cancellation_requested_during_only_gpu_block_is_observed(
    fake_runtime,
) -> None:
    _cupy, cupyx_ndimage, stream = fake_runtime
    original_label = cupyx_ndimage.label
    progress = _Progress()

    def cancelling_label(*args, **kwargs):
        result = original_label(*args, **kwargs)
        progress.cancelled = True
        return result

    cupyx_ndimage.label = cancelling_label

    with pytest.raises(RuntimeError, match="cancelled"):
        provider.label_connected_components(
            np.eye(6, dtype=bool),
            spatial_mode="2D YX",
            progress=progress,
        )

    assert len(cupyx_ndimage.calls) == 1
    assert stream.synchronize_count == 1
    assert progress.reports == [(0, 1, "Connected-component blocks")]


def test_per_block_int32_safety_rejects_before_output_allocation(
    fake_runtime,
) -> None:
    cupy, cupyx_ndimage, _stream = fake_runtime

    class _HugeBooleanArray:
        dtype = np.dtype(bool)
        ndim = 2
        shape = (1, 2**31 - 2)

    with pytest.raises(ValueError, match="fewer than 2,147,483,646"):
        provider.label_connected_components(
            _HugeBooleanArray(),
            spatial_mode="2D YX",
        )

    assert cupy.empty_calls == []
    assert cupyx_ndimage.calls == []


def test_direct_provider_rejects_unpromoted_one_dimensional_region(
    fake_runtime,
) -> None:
    with pytest.raises(ValueError, match="resolved 2D or 3D"):
        provider.label_connected_components(
            np.array([False, True, False]),
            spatial_mode="Auto from axes",
        )


def _real_cuda_modules_or_skip():
    cupy = pytest.importorskip("cupy")
    pytest.importorskip("cupyx.scipy.ndimage")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device is unavailable")
        cupy.cuda.runtime.getDevice()
    except Exception as exc:  # pragma: no cover - host-specific failure
        pytest.skip(f"CUDA runtime is unavailable: {exc}")
    return cupy


@pytest.mark.parametrize(
    ("shape", "spatial_mode", "connectivity"),
    [
        ((2, 31, 37), "2D YX", "Face connected"),
        ((3, 29, 35), "2D YX", "Full connectivity"),
        ((2, 9, 17, 19), "3D ZYX", "Face connected"),
        ((2, 7, 15, 17), "3D ZYX", "Full connectivity"),
    ],
)
def test_real_cuda_is_resident_int32_and_bitwise_cpu_exact(
    shape,
    spatial_mode,
    connectivity,
) -> None:
    cupy = _real_cuda_modules_or_skip()
    rng = np.random.default_rng(sum(shape) + len(connectivity))
    mask = rng.random(shape) < 0.22
    expected = cpu_label_connected_components(
        mask,
        spatial_mode=spatial_mode,
        connectivity=connectivity,
    )
    device_source = cupy.asarray(mask)
    before = device_source.copy()

    actual = provider.label_connected_components(
        device_source,
        spatial_mode=spatial_mode,
        connectivity=connectivity,
    )

    assert isinstance(actual, cupy.ndarray)
    assert actual.dtype == cupy.int32
    assert actual.flags.c_contiguous
    assert actual.data.ptr != device_source.data.ptr
    cupy.testing.assert_array_equal(device_source, before)
    np.testing.assert_array_equal(cupy.asnumpy(actual), expected)


def test_real_cuda_noncontiguous_numeric_input_and_repeated_ids_are_exact() -> None:
    cupy = _real_cuda_modules_or_skip()
    rng = np.random.default_rng(20260802)
    host_base = rng.standard_normal((2, 13, 34), dtype=np.float32)
    host_base[np.abs(host_base) < 1.1] = 0.0
    device_base = cupy.asarray(host_base)
    device_view = device_base[:, :, ::2]
    expected = cpu_label_connected_components(
        host_base[:, :, ::2],
        spatial_mode="3D ZYX",
        connectivity="Full connectivity",
    )

    outputs = [
        provider.label_connected_components(
            device_view,
            spatial_mode="3D ZYX",
            connectivity="Full connectivity",
        )
        for _ in range(10)
    ]

    for output in outputs:
        np.testing.assert_array_equal(cupy.asnumpy(output), expected)
        cupy.testing.assert_array_equal(output, outputs[0])
