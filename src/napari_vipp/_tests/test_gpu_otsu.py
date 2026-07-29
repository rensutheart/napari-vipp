from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from napari_vipp.core.gpu import cupy_otsu
from napari_vipp.core.operations import _otsu_value as cpu_otsu_value
from napari_vipp.core.operations import otsu_threshold as cpu_otsu
from napari_vipp.core.progress import OperationCancelled, ProgressContext


class _FakeStream:
    def __init__(self) -> None:
        self.synchronizations = 0

    def synchronize(self) -> None:
        self.synchronizations += 1


class _FakeCupy:
    bool_ = np.bool_
    uint64 = np.uint64
    int64 = np.int64
    float64 = np.float64

    def __init__(self) -> None:
        self.stream = _FakeStream()
        self.cuda = SimpleNamespace(get_current_stream=lambda: self.stream)
        self.host_transfer_sizes: list[int] = []

    asarray = staticmethod(np.asarray)
    moveaxis = staticmethod(np.moveaxis)
    sum = staticmethod(np.sum)
    isfinite = staticmethod(np.isfinite)
    min = staticmethod(np.min)
    max = staticmethod(np.max)
    histogram = staticmethod(np.histogram)
    searchsorted = staticmethod(np.searchsorted)
    minimum = staticmethod(np.minimum)
    empty = staticmethod(np.empty)
    zeros = staticmethod(np.zeros)
    count_nonzero = staticmethod(np.count_nonzero)

    @staticmethod
    def bincount(*_args, **_kwargs):
        raise AssertionError("Otsu must not dispatch through CuPy/CUB bincount.")

    @staticmethod
    def RawKernel(_source, _name, *, options=()):
        del options

        def launch(_grid, _block, arguments):
            values, size, bin_count, counts = arguments
            flattened = np.asarray(values).reshape(-1)[: int(size)]
            valid = (flattened >= 0) & (flattened < int(bin_count))
            np.add.at(counts, flattened[valid].astype(np.intp), np.uint64(1))

        return launch

    def asnumpy(self, value):
        result = np.asarray(value)
        self.host_transfer_sizes.append(int(result.size))
        return result


@pytest.fixture
def fake_cupy(monkeypatch):
    cupy = _FakeCupy()
    real_import = importlib.import_module

    def load(name: str):
        return cupy if name == "cupy" else real_import(name)

    cupy_otsu._cupy_module.cache_clear()
    cupy_otsu._atomic_bincount_kernel.cache_clear()
    monkeypatch.setattr(cupy_otsu.importlib, "import_module", load)
    yield cupy
    cupy_otsu._cupy_module.cache_clear()
    cupy_otsu._atomic_bincount_kernel.cache_clear()


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


def test_import_is_safe_without_cupy():
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
module = importlib.import_module("napari_vipp.core.gpu.cupy_otsu")
assert callable(module.otsu_threshold)
assert module.__all__ == ["otsu_threshold"]
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


def _bimodal_float(shape: tuple[int, ...], *, seed: int, dtype) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = int(np.prod(shape))
    values = np.concatenate(
        (
            rng.normal(-2.0, 0.45, count // 2),
            rng.normal(4.0, 0.8, count - count // 2),
        )
    ).astype(dtype)
    rng.shuffle(values)
    result = values.reshape(shape)
    if np.issubdtype(np.dtype(dtype), np.floating):
        result.flat[3] = np.nan
        result.flat[17] = np.inf
        result.flat[31] = -np.inf
    return result


@pytest.mark.parametrize("scope", ["Stack histogram", "Slice histogram"])
@pytest.mark.parametrize("histogram_bins", [2, 17, 256])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_fake_float_masks_and_thresholds_match_cpu_exactly(
    fake_cupy,
    scope,
    histogram_bins,
    dtype,
):
    data = _bimodal_float((3, 37, 41), seed=107, dtype=dtype)
    data.setflags(write=False)
    before = data.copy()

    expected = cpu_otsu(
        data,
        threshold_scope=scope,
        histogram_bins=histogram_bins,
    )
    actual = cupy_otsu.otsu_threshold(
        data,
        threshold_scope=scope,
        histogram_bins=histogram_bins,
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(data, before)
    assert not data.flags.writeable
    if scope == "Stack histogram":
        assert cupy_otsu._otsu_value(
            data,
            histogram_bins=histogram_bins,
            cupy=fake_cupy,
        ) == cpu_otsu_value(data, histogram_bins)


@pytest.mark.parametrize(
    "data",
    [
        np.arange(256, dtype=np.uint8).reshape(16, 16),
        np.array([-120, -120, -3, 7, 7, 99], dtype=np.int16),
        np.array([0, 12, 12, 65_535], dtype=np.uint16),
        np.array([2**60 + offset for offset in (2, 2, 7, 199)], dtype=np.int64),
        np.array(
            [2**63 + 123 + offset for offset in (0, 4, 4, 220)],
            dtype=np.uint64,
        ),
    ],
)
def test_fake_integer_native_levels_and_large_offsets_match_exactly(fake_cupy, data):
    expected = cpu_otsu(data, histogram_bins=2)
    actual = cupy_otsu.otsu_threshold(data, histogram_bins=2)

    np.testing.assert_array_equal(actual, expected)
    assert cupy_otsu._otsu_value(
        data,
        histogram_bins=2,
        cupy=fake_cupy,
    ) == cpu_otsu_value(data, 2)


@pytest.mark.parametrize("scope", ["Stack histogram", "Slice histogram"])
def test_fake_boolean_input_is_an_independent_identity_copy(fake_cupy, scope):
    data = np.array(
        [[[False, True], [True, False]], [[True, True], [False, False]]]
    )

    result = cupy_otsu.otsu_threshold(
        data,
        threshold_scope=scope,
        histogram_bins="ignored for boolean input",
    )

    np.testing.assert_array_equal(result, data)
    assert result is not data


@pytest.mark.parametrize("dtype", [np.uint16, np.float32, np.float64])
@pytest.mark.parametrize("channel_axis", [1, -1])
@pytest.mark.parametrize("scope", ["Stack histogram", "Slice histogram"])
def test_fake_explicit_rgba_luma_matches_cpu_exactly(
    fake_cupy,
    dtype,
    channel_axis,
    scope,
):
    rng = np.random.default_rng(509)
    canonical = rng.uniform(0.0, 2_048.0, size=(2, 19, 23, 4)).astype(dtype)
    data = canonical if channel_axis == -1 else np.moveaxis(canonical, -1, 1)

    expected = cpu_otsu(
        data,
        threshold_scope=scope,
        histogram_bins=31,
        channel_axis=channel_axis,
    )
    actual = cupy_otsu.otsu_threshold(
        data,
        threshold_scope=scope,
        histogram_bins=31,
        channel_axis=channel_axis,
    )

    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (2, 19, 23)
    assert actual.dtype == bool


def test_fake_constant_and_nonfinite_behavior_matches_cpu(fake_cupy):
    constant = np.full((7, 9), 3.25, dtype=np.float32)
    mixed = np.array([0.0, 1.0, np.nan, np.inf, -np.inf], dtype=np.float32)

    np.testing.assert_array_equal(
        cupy_otsu.otsu_threshold(constant),
        cpu_otsu(constant),
    )
    np.testing.assert_array_equal(
        cupy_otsu.otsu_threshold(mixed),
        cpu_otsu(mixed),
    )
    assert not cupy_otsu.otsu_threshold(mixed)[2:].any()


def test_fake_extreme_float32_range_uses_exact_comparison_bins(fake_cupy):
    maximum = np.finfo(np.float32).max
    data = np.asarray(
        [
            [-1.0, -1.0, maximum / 2.0],
            [maximum / 2.0, 0.0, 1.0],
            [maximum / 2.0, 1.0, -maximum / 2.0],
        ],
        dtype=np.float32,
    )

    expected = cpu_otsu(data, histogram_bins=3)
    actual = cupy_otsu.otsu_threshold(data, histogram_bins=3)

    np.testing.assert_array_equal(actual, expected)
    assert cupy_otsu._otsu_value(
        data,
        histogram_bins=3,
        cupy=fake_cupy,
    ) == cpu_otsu_value(data, 3)


@pytest.mark.parametrize(
    ("data", "kwargs", "message"),
    [
        (np.array([], dtype=np.float32), {}, "requires non-empty image data"),
        (
            np.array([np.nan, np.inf, -np.inf], dtype=np.float32),
            {},
            "at least one finite input value",
        ),
        (
            np.zeros((4, 5), dtype=np.float32),
            {"threshold_scope": "banana"},
            "must be 'Stack histogram' or 'Slice histogram'",
        ),
        (
            np.zeros((3, 5, 6), dtype=np.float32),
            {"channel_axis": 0, "histogram_bins": 1},
            "integer from 2 to 65,536",
        ),
        (
            np.zeros((2, 5, 6), dtype=np.float32),
            {"channel_axis": 0},
            "exactly 3 RGB or 4 RGBA channels",
        ),
        (
            np.array([0, 100_000], dtype=np.int32),
            {},
            "100,001 levels",
        ),
        (
            np.ones((3, 4), dtype=np.complex64),
            {},
            "require boolean, integer, or floating-point",
        ),
    ],
)
def test_fake_error_contract_matches_cpu(fake_cupy, data, kwargs, message):
    with pytest.raises(ValueError, match=message):
        cpu_otsu(data, **kwargs)
    with pytest.raises(ValueError, match=message):
        cupy_otsu.otsu_threshold(data, **kwargs)


def test_fake_slice_scope_reports_truthful_progress_and_synchronizes(fake_cupy):
    data = _bimodal_float((3, 11, 13), seed=613, dtype=np.float32)
    updates = []
    progress = ProgressContext(reporter=updates.append)

    actual = cupy_otsu.otsu_threshold(
        data,
        threshold_scope="Slice histogram",
        progress=progress,
    )

    np.testing.assert_array_equal(
        actual,
        cpu_otsu(data, threshold_scope="Slice histogram"),
    )
    assert [(update.current, update.total) for update in updates] == [
        (0, 3),
        (1, 3),
        (2, 3),
        (3, 3),
    ]
    assert fake_cupy.stream.synchronizations == 3


def test_fake_progress_is_cooperatively_cancellable_before_gpu_work(fake_cupy):
    progress = ProgressContext(cancelled=lambda: True)

    with pytest.raises(OperationCancelled):
        cupy_otsu.otsu_threshold(
            np.arange(64, dtype=np.float32).reshape(8, 8),
            progress=progress,
        )

    assert fake_cupy.stream.synchronizations == 0
    assert fake_cupy.host_transfer_sizes == []


def test_fake_slice_cancellation_stops_before_second_plane_and_is_reusable(
    fake_cupy,
    monkeypatch,
):
    data = _bimodal_float((3, 17, 19), seed=701, dtype=np.float32)
    updates = []
    cancelled = False
    histogram_calls = 0
    original_otsu_value = cupy_otsu._otsu_value

    def count_histogram(*args, **kwargs):
        nonlocal histogram_calls
        histogram_calls += 1
        return original_otsu_value(*args, **kwargs)

    def report(update):
        nonlocal cancelled
        updates.append(update)
        if update.current == 1:
            cancelled = True

    monkeypatch.setattr(cupy_otsu, "_otsu_value", count_histogram)
    progress = ProgressContext(
        reporter=report,
        cancelled=lambda: cancelled,
    )

    with pytest.raises(OperationCancelled):
        cupy_otsu.otsu_threshold(
            data,
            threshold_scope="Slice histogram",
            progress=progress,
        )

    assert [(update.current, update.total) for update in updates] == [(0, 3), (1, 3)]
    assert histogram_calls == 1
    assert fake_cupy.host_transfer_sizes == [256]
    assert fake_cupy.stream.synchronizations == 1

    # A cancelled partial output is never returned and does not poison the
    # provider: a fresh invocation can immediately process all three planes.
    recovered = cupy_otsu.otsu_threshold(
        data,
        threshold_scope="Slice histogram",
    )
    np.testing.assert_array_equal(
        recovered,
        cpu_otsu(data, threshold_scope="Slice histogram"),
    )
    assert histogram_calls == 4
    assert fake_cupy.host_transfer_sizes == [256, 256, 256, 256]


def test_fake_boolean_identity_progress_matches_cpu(fake_cupy):
    data = np.asarray([[False, True], [True, False]])
    cpu_updates = []
    gpu_updates = []

    expected = cpu_otsu(
        data,
        progress=ProgressContext(reporter=cpu_updates.append),
    )
    actual = cupy_otsu.otsu_threshold(
        data,
        progress=ProgressContext(reporter=gpu_updates.append),
    )

    np.testing.assert_array_equal(actual, expected)
    assert gpu_updates == cpu_updates
    assert fake_cupy.stream.synchronizations == 1


def test_fake_only_transfers_the_bounded_histogram_to_host(fake_cupy):
    data = _bimodal_float((4, 200, 300), seed=719, dtype=np.float32)

    cupy_otsu.otsu_threshold(data, histogram_bins=127)

    assert fake_cupy.host_transfer_sizes == [127]
    assert max(fake_cupy.host_transfer_sizes) < data.size


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_fake_exact_bincount_uses_one_bounded_uint64_histogram(fake_cupy, dtype):
    indices = np.asarray([0, 2, 2, 4, 4, 4], dtype=dtype)

    counts = cupy_otsu._exact_bincount(
        indices,
        bin_count=5,
        cupy=fake_cupy,
    )

    np.testing.assert_array_equal(counts, np.asarray([1, 0, 2, 0, 3], dtype=np.uint64))
    assert counts.dtype == np.uint64


@pytest.mark.parametrize("scope", ["Stack histogram", "Slice histogram"])
@pytest.mark.parametrize(
    "dtype",
    [np.uint8, np.uint16, np.int16, np.int64, np.uint64, np.float32, np.float64],
)
def test_real_cupy_final_masks_are_bitwise_equal(real_cupy, scope, dtype):
    rng = np.random.default_rng(823)
    if np.issubdtype(np.dtype(dtype), np.integer):
        data = rng.integers(0, 221, size=(3, 43, 47), dtype=np.uint16).astype(dtype)
        if dtype == np.int64:
            data += 2**60
        elif dtype == np.uint64:
            data += np.uint64(2**63 + 123)
    else:
        data = _bimodal_float((3, 43, 47), seed=827, dtype=dtype)

    expected = cpu_otsu(data, threshold_scope=scope, histogram_bins=37)
    resident = cupy_otsu.otsu_threshold(
        real_cupy.asarray(data),
        threshold_scope=scope,
        histogram_bins=37,
    )
    real_cupy.cuda.get_current_stream().synchronize()

    assert isinstance(resident, real_cupy.ndarray)
    assert resident.dtype == real_cupy.bool_
    np.testing.assert_array_equal(real_cupy.asnumpy(resident), expected)


def test_real_cupy_rgba_and_nonfinite_contract_is_exact(real_cupy):
    rng = np.random.default_rng(929)
    data = rng.normal(size=(2, 31, 37, 4)).astype(np.float32)
    data[0, 0, 0, 0] = np.nan
    data[1, 2, 3, 1] = np.inf

    expected = cpu_otsu(
        data,
        threshold_scope="Slice histogram",
        histogram_bins=53,
        channel_axis=-1,
    )
    resident = cupy_otsu.otsu_threshold(
        real_cupy.asarray(data),
        threshold_scope="Slice histogram",
        histogram_bins=53,
        channel_axis=-1,
    )

    np.testing.assert_array_equal(real_cupy.asnumpy(resident), expected)


def test_real_cupy_extreme_float32_histogram_is_bitwise_exact(real_cupy):
    maximum = np.finfo(np.float32).max
    data = np.asarray(
        [
            [-1.0, -1.0, maximum / 2.0],
            [maximum / 2.0, 0.0, 1.0],
            [maximum / 2.0, 1.0, -maximum / 2.0],
        ],
        dtype=np.float32,
    )

    expected = cpu_otsu(data, histogram_bins=3)
    resident = cupy_otsu.otsu_threshold(
        real_cupy.asarray(data),
        histogram_bins=3,
    )

    np.testing.assert_array_equal(real_cupy.asnumpy(resident), expected)


def test_real_cupy_wide_int64_rgb_luma_histogram_is_bitwise_exact(real_cupy):
    rng = np.random.default_rng(0)
    data = rng.integers(
        -(2**63),
        2**63 - 1,
        size=(7, 9, 3),
        dtype=np.int64,
    )

    expected = cpu_otsu(data, channel_axis=-1, histogram_bins=256)
    resident = cupy_otsu.otsu_threshold(
        real_cupy.asarray(data),
        channel_axis=-1,
        histogram_bins=256,
    )

    np.testing.assert_array_equal(real_cupy.asnumpy(resident), expected)
