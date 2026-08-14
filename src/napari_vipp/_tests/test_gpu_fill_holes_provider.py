from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage as ndi

from napari_vipp.core.gpu import cupy_fill_holes as provider
from napari_vipp.core.operations import fill_holes as cpu_fill_holes

SOURCE_ROOT = Path(__file__).resolve().parents[2]


class _FakeStream:
    def __init__(self) -> None:
        self.synchronize_count = 0

    def synchronize(self) -> None:
        self.synchronize_count += 1


class _FakeCupy:
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

    def binary_fill_holes(self, source, *, structure, output):
        ndi.binary_fill_holes(
            np.asarray(source),
            structure=np.asarray(structure),
            output=output,
        )
        self.calls.append(
            {
                "source": np.asarray(source).copy(),
                "structure": np.asarray(structure).copy(),
                "output": output,
            }
        )
        return output


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
import napari_vipp.core.gpu.cupy_fill_holes as module
assert module.__all__ == ["fill_holes"]
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
@pytest.mark.parametrize(
    ("shape", "spatial_mode"),
    [
        ((2, 17, 19), "2D YX"),
        ((2, 7, 13, 15), "3D ZYX"),
    ],
)
def test_fake_provider_is_bitwise_cpu_exact_for_2d_and_3d_blocks(
    fake_runtime,
    shape,
    spatial_mode,
    connectivity,
) -> None:
    _cupy, cupyx_ndimage, _stream = fake_runtime
    rng = np.random.default_rng(sum(shape) + len(connectivity))
    mask = rng.random(shape) < 0.61
    before = mask.copy()

    expected = cpu_fill_holes(
        mask,
        max_hole_size=0,
        spatial_mode=spatial_mode,
        connectivity=connectivity,
    )
    actual = provider.fill_holes(
        mask,
        max_hole_size=0,
        spatial_mode=spatial_mode,
        connectivity=connectivity,
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(mask, before)
    assert actual.dtype == bool
    assert actual.flags.c_contiguous
    assert not np.shares_memory(actual, mask)
    expected_blocks = int(np.prod(shape[: -2 if spatial_mode == "2D YX" else -3]))
    assert len(cupyx_ndimage.calls) == expected_blocks


def test_leading_blocks_are_written_directly_and_independently(fake_runtime) -> None:
    _cupy, cupyx_ndimage, _stream = fake_runtime
    mask = np.ones((2, 3, 7, 9), dtype=bool)
    mask[0, 0, 3, 4] = False
    mask[0, 1, 0, 4] = False
    mask[1, 2, 2:4, 5:7] = False

    actual = provider.fill_holes(mask, spatial_mode="2D YX")
    expected = cpu_fill_holes(mask, spatial_mode="2D YX")

    np.testing.assert_array_equal(actual, expected)
    assert actual[0, 0, 3, 4]
    assert not actual[0, 1, 0, 4]
    assert len(cupyx_ndimage.calls) == 6
    assert all(np.shares_memory(call["output"], actual) for call in cupyx_ndimage.calls)


@pytest.mark.parametrize("connectivity", ["Face connected", "Full connectivity"])
def test_noncontiguous_adversarial_input_is_exact_and_independent(
    fake_runtime,
    connectivity,
) -> None:
    _cupy, _cupyx_ndimage, _stream = fake_runtime
    base = np.indices((2, 19, 42)).sum(axis=0) % 2 == 0
    base[:, 5:14, 10:30] = True
    base[:, 8:11, 16:22] = False
    view = base[:, :, ::2]
    assert not view.flags.c_contiguous
    before = base.copy()

    actual = provider.fill_holes(
        view,
        spatial_mode="2D YX",
        connectivity=connectivity,
    )
    expected = cpu_fill_holes(
        view,
        spatial_mode="2D YX",
        connectivity=connectivity,
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(base, before)
    assert actual.flags.c_contiguous
    assert not np.shares_memory(actual, view)


def test_empty_leading_batch_returns_an_independent_empty_mask(fake_runtime) -> None:
    cupy, cupyx_ndimage, _stream = fake_runtime
    mask = np.empty((0, 7, 9), dtype=bool)
    progress = _Progress()

    actual = provider.fill_holes(
        mask,
        spatial_mode="2D YX",
        progress=progress,
    )

    assert actual.shape == mask.shape
    assert actual.dtype == bool
    assert cupy.empty_calls == [(mask.shape, bool)]
    assert cupyx_ndimage.calls == []
    assert progress.reports == [
        (0, 1, "Fill-hole blocks"),
        (1, 1, "Fill-hole blocks"),
    ]


def test_empty_spatial_blocks_match_cpu(fake_runtime) -> None:
    _cupy, cupyx_ndimage, _stream = fake_runtime
    mask = np.empty((3, 0, 7), dtype=bool)

    actual = provider.fill_holes(mask, spatial_mode="2D YX")
    expected = cpu_fill_holes(mask, spatial_mode="2D YX")

    np.testing.assert_array_equal(actual, expected)
    assert len(cupyx_ndimage.calls) == 3


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32, np.float32])
def test_numeric_nonzero_inputs_fail_closed_before_output_allocation(
    fake_runtime,
    dtype,
) -> None:
    cupy, cupyx_ndimage, _stream = fake_runtime

    with pytest.raises(ValueError, match="requires a boolean mask"):
        provider.fill_holes(np.ones((5, 7), dtype=dtype), spatial_mode="2D YX")

    assert cupy.empty_calls == []
    assert cupyx_ndimage.calls == []


@pytest.mark.parametrize("maximum", [-1, 1, 12, False, 0.0, "0", None])
def test_noncanonical_or_size_limited_parameters_fail_closed(
    fake_runtime,
    maximum,
) -> None:
    cupy, cupyx_ndimage, _stream = fake_runtime

    with pytest.raises(ValueError, match="requires max_hole_size=0"):
        provider.fill_holes(
            np.ones((5, 7), dtype=bool),
            max_hole_size=maximum,
            spatial_mode="2D YX",
        )

    assert cupy.empty_calls == []
    assert cupyx_ndimage.calls == []


@pytest.mark.parametrize(
    ("shape", "spatial_mode", "resolved", "message"),
    [
        ((7,), "Auto from axes", None, "resolved 2D or 3D"),
        ((5, 7), "3D ZYX", None, "cannot be applied"),
        ((5, 7), "not a mode", None, "Spatial mode must be"),
        ((3, 5, 7), "Auto from axes", 4, "resolved_spatial_ndim"),
    ],
)
def test_invalid_spatial_scope_fails_before_output_allocation(
    fake_runtime,
    shape,
    spatial_mode,
    resolved,
    message,
) -> None:
    cupy, cupyx_ndimage, _stream = fake_runtime

    with pytest.raises(ValueError, match=message):
        provider.fill_holes(
            np.ones(shape, dtype=bool),
            spatial_mode=spatial_mode,
            resolved_spatial_ndim=resolved,
        )

    assert cupy.empty_calls == []
    assert cupyx_ndimage.calls == []


def test_invalid_connectivity_fails_before_output_allocation(fake_runtime) -> None:
    cupy, cupyx_ndimage, _stream = fake_runtime

    with pytest.raises(ValueError, match="Face connected.*Full connectivity"):
        provider.fill_holes(
            np.ones((5, 7), dtype=bool),
            spatial_mode="2D YX",
            connectivity="edge connected",
        )

    assert cupy.empty_calls == []
    assert cupyx_ndimage.calls == []


def test_padded_block_int32_limit_is_exclusive_and_preallocation_safe(
    fake_runtime,
    monkeypatch,
) -> None:
    cupy, cupyx_ndimage, _stream = fake_runtime
    monkeypatch.setattr(provider, "_MAXIMUM_PADDED_SPATIAL_BLOCK_ELEMENTS", 20)

    with pytest.raises(ValueError, match="including CuPyX's one-pixel.*fewer than 20"):
        provider.fill_holes(
            np.ones((2, 3), dtype=bool),
            spatial_mode="2D YX",
        )

    assert cupy.empty_calls == []
    assert cupyx_ndimage.calls == []


def test_progress_and_cancellation_are_complete_synchronized_blocks(
    fake_runtime,
) -> None:
    _cupy, cupyx_ndimage, stream = fake_runtime
    mask = np.ones((3, 7, 9), dtype=bool)
    mask[:, 3, 4] = False
    progress = _Progress()

    provider.fill_holes(mask, spatial_mode="2D YX", progress=progress)

    assert progress.checks >= 2 * 3
    assert progress.reports == [
        (0, 3, "Fill-hole blocks"),
        (1, 3, "Fill-hole blocks"),
        (2, 3, "Fill-hole blocks"),
        (3, 3, "Fill-hole blocks"),
    ]
    assert stream.synchronize_count == 3

    calls_before_cancel = len(cupyx_ndimage.calls)
    syncs_before_cancel = stream.synchronize_count
    cancelled = _Progress(cancel_after_completed=1)
    with pytest.raises(RuntimeError, match="cancelled"):
        provider.fill_holes(
            mask,
            spatial_mode="2D YX",
            progress=cancelled,
        )
    assert cancelled.reports == [
        (0, 3, "Fill-hole blocks"),
        (1, 3, "Fill-hole blocks"),
    ]
    assert len(cupyx_ndimage.calls) - calls_before_cancel == 1
    assert stream.synchronize_count - syncs_before_cancel == 1


def test_cancellation_requested_during_only_gpu_block_is_observed(
    fake_runtime,
) -> None:
    _cupy, cupyx_ndimage, stream = fake_runtime
    original_fill = cupyx_ndimage.binary_fill_holes
    progress = _Progress()

    def cancelling_fill(*args, **kwargs):
        result = original_fill(*args, **kwargs)
        progress.cancelled = True
        return result

    cupyx_ndimage.binary_fill_holes = cancelling_fill

    with pytest.raises(RuntimeError, match="cancelled"):
        provider.fill_holes(
            np.eye(7, dtype=bool),
            spatial_mode="2D YX",
            progress=progress,
        )

    assert len(cupyx_ndimage.calls) == 1
    assert stream.synchronize_count == 1
    assert progress.reports == [(0, 1, "Fill-hole blocks")]


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
        ((3, 31, 37), "2D YX", "Face connected"),
        ((2, 29, 35), "2D YX", "Full connectivity"),
        ((2, 9, 17, 19), "3D ZYX", "Face connected"),
        ((2, 7, 15, 17), "3D ZYX", "Full connectivity"),
    ],
)
def test_real_cuda_is_resident_bool_and_bitwise_cpu_exact(
    shape,
    spatial_mode,
    connectivity,
) -> None:
    cupy = _real_cuda_modules_or_skip()
    rng = np.random.default_rng(sum(shape) + len(connectivity))
    mask = rng.random(shape) < 0.62
    expected = cpu_fill_holes(
        mask,
        spatial_mode=spatial_mode,
        connectivity=connectivity,
    )
    device_source = cupy.asarray(mask)
    before = device_source.copy()

    actual = provider.fill_holes(
        device_source,
        spatial_mode=spatial_mode,
        connectivity=connectivity,
    )

    assert isinstance(actual, cupy.ndarray)
    assert actual.dtype == cupy.bool_
    assert actual.flags.c_contiguous
    assert actual.data.ptr != device_source.data.ptr
    cupy.testing.assert_array_equal(device_source, before)
    np.testing.assert_array_equal(cupy.asnumpy(actual), expected)


def test_real_cuda_noncontiguous_checkerboard_and_empty_batches_are_exact() -> None:
    cupy = _real_cuda_modules_or_skip()
    host_base = np.indices((2, 23, 46)).sum(axis=0) % 2 == 0
    host_base[:, 6:18, 12:36] = True
    host_base[:, 10:14, 20:28] = False
    device_base = cupy.asarray(host_base)
    device_view = device_base[:, :, ::2]
    expected = cpu_fill_holes(
        host_base[:, :, ::2],
        spatial_mode="2D YX",
        connectivity="Face connected",
    )

    actual = provider.fill_holes(
        device_view,
        spatial_mode="2D YX",
        connectivity="Face connected",
    )
    empty = provider.fill_holes(
        cupy.empty((0, 7, 9), dtype=bool),
        spatial_mode="2D YX",
    )

    np.testing.assert_array_equal(cupy.asnumpy(actual), expected)
    assert actual.flags.c_contiguous
    assert actual.data.ptr != device_view.data.ptr
    assert empty.shape == (0, 7, 9)
    assert empty.dtype == cupy.bool_
