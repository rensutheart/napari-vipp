from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage as ndi

from napari_vipp.core.gpu import cupy_remove_small_objects as provider
from napari_vipp.core.operations import remove_small_objects as cpu_remove_small_objects

SOURCE_ROOT = Path(__file__).resolve().parents[2]
MAXIMUM_SPATIAL_BLOCK_ELEMENTS = 2**31 - 2


class _FakeStream:
    def __init__(self) -> None:
        self.synchronize_count = 0

    def synchronize(self) -> None:
        self.synchronize_count += 1


class _FakeCupy:
    int32 = np.int32

    def __init__(self, stream: _FakeStream) -> None:
        self.cuda = SimpleNamespace(get_current_stream=lambda: stream)
        self.empty_calls: list[tuple[tuple[int, ...], np.dtype]] = []
        self.bincount_calls: list[np.ndarray] = []
        self.bincount_callback = None

    @staticmethod
    def asarray(value):
        return value if hasattr(value, "shape") else np.asarray(value)

    def empty(self, shape, dtype):
        normalized_shape = tuple(int(size) for size in shape)
        normalized_dtype = np.dtype(dtype)
        self.empty_calls.append((normalized_shape, normalized_dtype))
        return np.empty(normalized_shape, dtype=normalized_dtype)

    def bincount(self, values):
        copied = np.asarray(values).copy()
        self.bincount_calls.append(copied)
        result = np.bincount(copied)
        if self.bincount_callback is not None:
            self.bincount_callback()
        return result


class _FakeCuPyXNdimage:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.label_callback = None

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
        if self.label_callback is not None:
            self.label_callback()
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


def _component_mask() -> np.ndarray:
    mask = np.zeros((12, 13), dtype=bool)
    mask[1, 1] = True
    mask[1, 5:7] = True
    mask[6:9, 2] = True
    mask[8:10, 9:11] = True
    return mask


def test_provider_module_import_is_gpu_lazy_in_a_fresh_process() -> None:
    script = f"""
import sys
sys.path.insert(0, {str(SOURCE_ROOT)!r})
import napari_vipp.core.gpu.cupy_remove_small_objects as module
assert module.__all__ == ["remove_small_objects"]
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
@pytest.mark.parametrize("minimum", [-7, 0, 1, 2, 3, 4, 5])
def test_minimum_boundaries_are_bitwise_cpu_exact_and_preserve_input(
    fake_runtime,
    connectivity,
    minimum,
) -> None:
    cupy, cupyx_ndimage, _stream = fake_runtime
    source = _component_mask()
    before = source.copy()
    expected = cpu_remove_small_objects(
        source,
        min_size=minimum,
        spatial_mode="2D YX",
        connectivity=connectivity,
    )

    actual = provider.remove_small_objects(
        source,
        min_size=np.int64(minimum),
        spatial_mode="2D YX",
        connectivity=connectivity,
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, before)
    assert actual.dtype == bool
    assert actual.flags.c_contiguous
    assert not np.shares_memory(actual, source)
    assert cupy.empty_calls[0] == (source.shape, np.dtype(bool))
    if minimum <= 1:
        assert cupyx_ndimage.calls == []
        assert cupy.bincount_calls == []
        assert len(cupy.empty_calls) == 1
    else:
        assert len(cupyx_ndimage.calls) == 1
        assert len(cupy.bincount_calls) == 1
        assert cupy.empty_calls[1] == (source.shape, np.dtype(np.int32))


@pytest.mark.parametrize("minimum", [2**64, 2**100])
def test_arbitrarily_large_minimum_short_circuits_exactly_without_labeling(
    fake_runtime,
    minimum,
) -> None:
    cupy, cupyx_ndimage, _stream = fake_runtime
    source = _component_mask()
    before = source.copy()
    expected = cpu_remove_small_objects(
        source,
        min_size=minimum,
        spatial_mode="2D YX",
        connectivity="Face connected",
    )

    actual = provider.remove_small_objects(
        source,
        min_size=minimum,
        spatial_mode="2D YX",
        connectivity="Face connected",
    )

    np.testing.assert_array_equal(actual, expected)
    assert not actual.any()
    np.testing.assert_array_equal(source, before)
    assert not np.shares_memory(actual, source)
    assert cupy.empty_calls == [(source.shape, np.dtype(bool))]
    assert cupyx_ndimage.calls == []
    assert cupy.bincount_calls == []


def test_one_int32_workspace_is_reused_across_independent_leading_blocks(
    fake_runtime,
) -> None:
    cupy, cupyx_ndimage, _stream = fake_runtime
    mask = np.zeros((2, 3, 8, 9), dtype=bool)
    mask[..., 1:3, 1:3] = True
    mask[0, 0, 6, 7] = True
    mask[1, 2, 4:7, 5:8] = True
    expected = cpu_remove_small_objects(
        mask,
        min_size=3,
        spatial_mode="2D YX",
        connectivity="Face connected",
    )

    actual = provider.remove_small_objects(
        mask,
        min_size=3,
        spatial_mode="2D YX",
        connectivity="Face connected",
    )

    np.testing.assert_array_equal(actual, expected)
    assert len(cupyx_ndimage.calls) == 6
    assert len(cupy.bincount_calls) == 6
    assert cupy.empty_calls == [
        (mask.shape, np.dtype(bool)),
        (mask.shape[-2:], np.dtype(np.int32)),
    ]
    workspace_ids = {id(call["output"]) for call in cupyx_ndimage.calls}
    assert len(workspace_ids) == 1


@pytest.mark.parametrize("connectivity", ["Face connected", "Full connectivity"])
def test_true_3d_connectivity_is_cpu_exact(fake_runtime, connectivity) -> None:
    _cupy, cupyx_ndimage, _stream = fake_runtime
    mask = np.zeros((2, 5, 8, 9), dtype=bool)
    mask[:, 1:4, 1:4, 1:4] = True
    mask[0, 3, 6, 7] = True
    mask[1, 1, 5, 5] = True
    mask[1, 2, 6, 6] = True
    expected = cpu_remove_small_objects(
        mask,
        min_size=2,
        spatial_mode="3D ZYX",
        connectivity=connectivity,
    )

    actual = provider.remove_small_objects(
        mask,
        min_size=2,
        spatial_mode="3D ZYX",
        connectivity=connectivity,
    )

    np.testing.assert_array_equal(actual, expected)
    assert len(cupyx_ndimage.calls) == 2
    expected_rank = 1 if connectivity.startswith("Face") else 3
    np.testing.assert_array_equal(
        cupyx_ndimage.calls[0]["structure"],
        ndi.generate_binary_structure(3, expected_rank),
    )


@pytest.mark.parametrize(
    ("shape", "spatial_mode"),
    [
        ((0, 5, 6), "2D YX"),
        ((2, 0, 6), "2D YX"),
        ((0, 6), "2D YX"),
        ((0, 4, 5), "3D ZYX"),
    ],
)
def test_empty_arrays_are_independent_and_cpu_exact(
    fake_runtime,
    shape,
    spatial_mode,
) -> None:
    cupy, cupyx_ndimage, _stream = fake_runtime
    source = np.empty(shape, dtype=bool)
    expected = cpu_remove_small_objects(
        source,
        min_size=4,
        spatial_mode=spatial_mode,
    )

    actual = provider.remove_small_objects(
        source,
        min_size=4,
        spatial_mode=spatial_mode,
    )

    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == shape
    assert actual.dtype == bool
    assert not np.shares_memory(actual, source)
    assert cupyx_ndimage.calls == []
    assert cupy.bincount_calls == []


@pytest.mark.parametrize("connectivity", ["Face connected", "Full connectivity"])
def test_noncontiguous_adversarial_view_is_exact(fake_runtime, connectivity) -> None:
    _cupy, cupyx_ndimage, _stream = fake_runtime
    checkerboard = np.indices((3, 17, 38)).sum(axis=0) % 2 == 0
    source = checkerboard[:, :, ::2]
    assert not source.flags.c_contiguous
    expected = cpu_remove_small_objects(
        source,
        min_size=4,
        spatial_mode="2D YX",
        connectivity=connectivity,
    )

    actual = provider.remove_small_objects(
        source,
        min_size=4,
        spatial_mode="2D YX",
        connectivity=connectivity,
    )

    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.c_contiguous
    assert len(cupyx_ndimage.calls) == 3


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32, np.float32])
def test_direct_provider_rejects_non_boolean_input_before_allocation(
    fake_runtime,
    dtype,
) -> None:
    cupy, cupyx_ndimage, _stream = fake_runtime

    with pytest.raises(ValueError, match="requires a boolean mask"):
        provider.remove_small_objects(
            np.zeros((6, 7), dtype=dtype),
            spatial_mode="2D YX",
        )

    assert cupy.empty_calls == []
    assert cupyx_ndimage.calls == []


@pytest.mark.parametrize("minimum", [True, False, 2.0, "2", None])
def test_direct_provider_rejects_non_integral_or_boolean_minimum(
    fake_runtime,
    minimum,
) -> None:
    cupy, cupyx_ndimage, _stream = fake_runtime

    with pytest.raises(ValueError, match="must be an integer"):
        provider.remove_small_objects(
            np.zeros((6, 7), dtype=bool),
            min_size=minimum,
            spatial_mode="2D YX",
        )

    assert cupy.empty_calls == []
    assert cupyx_ndimage.calls == []


@pytest.mark.parametrize(
    ("source", "spatial_mode", "resolved_spatial_ndim", "connectivity", "message"),
    [
        (
            np.zeros((7,), dtype=bool),
            "Auto from axes",
            None,
            "Face connected",
            "resolved 2D or 3D",
        ),
        (
            np.zeros((6, 7), dtype=bool),
            "3D ZYX",
            None,
            "Face connected",
            "cannot be applied",
        ),
        (
            np.zeros((3, 6, 7), dtype=bool),
            "Auto from axes",
            2,
            "Edge connected",
            "Connectivity must be",
        ),
    ],
)
def test_direct_provider_fails_closed_for_invalid_scope(
    fake_runtime,
    source,
    spatial_mode,
    resolved_spatial_ndim,
    connectivity,
    message,
) -> None:
    cupy, cupyx_ndimage, _stream = fake_runtime

    with pytest.raises(ValueError, match=message):
        provider.remove_small_objects(
            source,
            min_size=2,
            spatial_mode=spatial_mode,
            resolved_spatial_ndim=resolved_spatial_ndim,
            connectivity=connectivity,
        )

    assert cupy.empty_calls == []
    assert cupyx_ndimage.calls == []


def test_spatial_block_limit_rejects_before_output_allocation(fake_runtime) -> None:
    cupy, cupyx_ndimage, _stream = fake_runtime

    class _HugeBooleanArray:
        dtype = np.dtype(bool)
        ndim = 2
        shape = (1, MAXIMUM_SPATIAL_BLOCK_ELEMENTS)

    with pytest.raises(ValueError, match="fewer than 2,147,483,646"):
        provider.remove_small_objects(
            _HugeBooleanArray(),
            min_size=2,
            spatial_mode="2D YX",
        )

    assert cupy.empty_calls == []
    assert cupyx_ndimage.calls == []


def test_progress_and_cancellation_are_complete_compound_block_boundaries(
    fake_runtime,
) -> None:
    _cupy, cupyx_ndimage, stream = fake_runtime
    mask = np.zeros((3, 6, 7), dtype=bool)
    mask[:, 1:4, 1:4] = True
    progress = _Progress()

    provider.remove_small_objects(
        mask,
        min_size=2,
        spatial_mode="2D YX",
        progress=progress,
    )

    assert progress.checks >= 2 * 3
    assert progress.reports == [
        (0, 3, "Small-object blocks"),
        (1, 3, "Small-object blocks"),
        (2, 3, "Small-object blocks"),
        (3, 3, "Small-object blocks"),
    ]
    assert stream.synchronize_count == 3

    calls_before_cancel = len(cupyx_ndimage.calls)
    syncs_before_cancel = stream.synchronize_count
    cancelled = _Progress(cancel_after_completed=1)
    with pytest.raises(RuntimeError, match="cancelled"):
        provider.remove_small_objects(
            mask,
            min_size=2,
            spatial_mode="2D YX",
            progress=cancelled,
        )
    assert cancelled.reports == [
        (0, 3, "Small-object blocks"),
        (1, 3, "Small-object blocks"),
    ]
    assert len(cupyx_ndimage.calls) - calls_before_cancel == 1
    assert stream.synchronize_count - syncs_before_cancel == 1


def test_cancellation_during_compound_gpu_block_is_observed_after_sync(
    fake_runtime,
) -> None:
    cupy, cupyx_ndimage, stream = fake_runtime
    progress = _Progress()
    cupy.bincount_callback = lambda: setattr(progress, "cancelled", True)

    with pytest.raises(RuntimeError, match="cancelled"):
        provider.remove_small_objects(
            _component_mask(),
            min_size=2,
            spatial_mode="2D YX",
            progress=progress,
        )

    assert len(cupyx_ndimage.calls) == 1
    assert len(cupy.bincount_calls) == 1
    assert stream.synchronize_count == 1
    assert progress.reports == [(0, 1, "Small-object blocks")]


def test_oversized_minimum_shortcut_retains_progress_and_cancellation_boundaries(
    fake_runtime,
) -> None:
    cupy, cupyx_ndimage, stream = fake_runtime
    mask = np.zeros((3, 6, 7), dtype=bool)
    mask[:, 1:4, 1:4] = True
    progress = _Progress()

    actual = provider.remove_small_objects(
        mask,
        min_size=2**100,
        spatial_mode="2D YX",
        progress=progress,
    )

    assert not actual.any()
    assert progress.reports == [
        (0, 3, "Small-object blocks"),
        (1, 3, "Small-object blocks"),
        (2, 3, "Small-object blocks"),
        (3, 3, "Small-object blocks"),
    ]
    assert stream.synchronize_count == 3
    assert cupyx_ndimage.calls == []
    assert cupy.bincount_calls == []

    syncs_before_cancel = stream.synchronize_count
    cancelled = _Progress(cancel_after_completed=1)
    with pytest.raises(RuntimeError, match="cancelled"):
        provider.remove_small_objects(
            mask,
            min_size=2**100,
            spatial_mode="2D YX",
            progress=cancelled,
        )
    assert cancelled.reports == [
        (0, 3, "Small-object blocks"),
        (1, 3, "Small-object blocks"),
    ]
    assert stream.synchronize_count - syncs_before_cancel == 1
    assert cupyx_ndimage.calls == []
    assert cupy.bincount_calls == []


def test_empty_leading_batch_has_truthful_terminal_progress(fake_runtime) -> None:
    _cupy, cupyx_ndimage, stream = fake_runtime
    progress = _Progress()

    actual = provider.remove_small_objects(
        np.empty((0, 5, 6), dtype=bool),
        min_size=2,
        spatial_mode="2D YX",
        progress=progress,
    )

    assert actual.shape == (0, 5, 6)
    assert cupyx_ndimage.calls == []
    assert stream.synchronize_count == 0
    assert _cupy.empty_calls == [
        ((0, 5, 6), np.dtype(bool)),
    ]
    assert progress.reports == [
        (0, 1, "Small-object blocks"),
        (1, 1, "Small-object blocks"),
    ]


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
    ("shape", "spatial_mode", "connectivity", "minimum"),
    [
        ((3, 31, 37), "2D YX", "Face connected", 1),
        ((2, 29, 35), "2D YX", "Full connectivity", 4),
        ((2, 7, 15, 17), "3D ZYX", "Face connected", 6),
        ((2, 7, 15, 17), "3D ZYX", "Full connectivity", 8),
    ],
)
def test_real_cuda_is_resident_boolean_and_bitwise_cpu_exact(
    shape,
    spatial_mode,
    connectivity,
    minimum,
) -> None:
    cupy = _real_cuda_modules_or_skip()
    rng = np.random.default_rng(sum(shape) + minimum + len(connectivity))
    mask = rng.random(shape) < 0.18
    expected = cpu_remove_small_objects(
        mask,
        min_size=minimum,
        spatial_mode=spatial_mode,
        connectivity=connectivity,
    )
    device_source = cupy.asarray(mask)
    before = device_source.copy()

    actual = provider.remove_small_objects(
        device_source,
        min_size=minimum,
        spatial_mode=spatial_mode,
        connectivity=connectivity,
    )

    assert isinstance(actual, cupy.ndarray)
    assert actual.dtype == cupy.bool_
    assert actual.flags.c_contiguous
    assert actual.data.ptr != device_source.data.ptr
    cupy.testing.assert_array_equal(device_source, before)
    np.testing.assert_array_equal(cupy.asnumpy(actual), expected)


@pytest.mark.parametrize("connectivity", ["Face connected", "Full connectivity"])
def test_real_cuda_noncontiguous_checkerboard_is_exact_and_deterministic(
    connectivity,
) -> None:
    cupy = _real_cuda_modules_or_skip()
    host_base = np.indices((3, 33, 70)).sum(axis=0) % 2 == 0
    device_base = cupy.asarray(host_base)
    device_view = device_base[:, :, ::2]
    expected = cpu_remove_small_objects(
        host_base[:, :, ::2],
        min_size=3,
        spatial_mode="2D YX",
        connectivity=connectivity,
    )

    outputs = [
        provider.remove_small_objects(
            device_view,
            min_size=3,
            spatial_mode="2D YX",
            connectivity=connectivity,
        )
        for _ in range(3)
    ]

    for output in outputs:
        np.testing.assert_array_equal(cupy.asnumpy(output), expected)
        cupy.testing.assert_array_equal(output, outputs[0])


@pytest.mark.parametrize("minimum", [2**64, 2**100])
def test_real_cuda_arbitrarily_large_minimum_is_exact_without_label_or_bincount(
    monkeypatch,
    minimum,
) -> None:
    cupy = _real_cuda_modules_or_skip()
    source = _component_mask()
    expected = cpu_remove_small_objects(
        source,
        min_size=minimum,
        spatial_mode="2D YX",
        connectivity="Full connectivity",
    )
    device_source = cupy.asarray(source)
    before = device_source.copy()

    def forbidden(*_args, **_kwargs):
        pytest.fail("oversized minimum must not label or bincount")

    monkeypatch.setattr(
        provider,
        "_cupyx_ndimage_module",
        lambda: SimpleNamespace(label=forbidden),
    )
    monkeypatch.setattr(cupy, "bincount", forbidden)

    actual = provider.remove_small_objects(
        device_source,
        min_size=minimum,
        spatial_mode="2D YX",
        connectivity="Full connectivity",
    )

    np.testing.assert_array_equal(cupy.asnumpy(actual), expected)
    assert not bool(cupy.any(actual).item())
    assert actual.dtype == cupy.bool_
    assert actual.data.ptr != device_source.data.ptr
    cupy.testing.assert_array_equal(device_source, before)
