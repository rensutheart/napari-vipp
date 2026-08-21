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

from napari_vipp.core.gpu import cupy_remove_binary_outliers as provider
from napari_vipp.core.progress import OperationCancelled, ProgressContext
from napari_vipp.core.remove_outliers import (
    BACKGROUND_OUTLIERS,
    FOREGROUND_OUTLIERS,
    imagej_remove_outliers_footprint,
)
from napari_vipp.core.remove_outliers import (
    remove_binary_outliers as cpu_remove_binary_outliers,
)


class _FakeStream:
    def __init__(self) -> None:
        self.synchronizations = 0

    def synchronize(self) -> None:
        self.synchronizations += 1


class _FakeRawKernel:
    def __init__(self, owner, source: str, name: str, options: tuple[str, ...]) -> None:
        self.owner = owner
        self.source = source
        self.name = name
        self.options = options

    def __call__(self, grid, block, arguments) -> None:
        self.owner.launches.append(
            (self.name, grid, block, arguments[4] - arguments[3])
        )
        (
            source,
            rows,
            columns,
            pixel_start,
            pixel_stop,
            row_half_widths,
            footprint_rows,
            majority_threshold,
            remove_foreground,
            output,
        ) = arguments
        rows = int(rows)
        columns = int(columns)
        pixel_start = int(pixel_start)
        pixel_stop = int(pixel_stop)
        half_widths = np.asarray(row_half_widths, dtype=np.int32).reshape(-1)
        footprint_rows = int(footprint_rows)
        majority_threshold = int(majority_threshold)
        remove_foreground = bool(remove_foreground)
        source_plane = np.asarray(source, dtype=bool).reshape(rows, columns)
        output_plane = np.asarray(output).reshape(rows, columns)
        center_row = footprint_rows // 2

        for output_index in range(pixel_start, pixel_stop):
            y, x = divmod(output_index, columns)
            count = 0
            for row_index, half_width in enumerate(half_widths[:footprint_rows]):
                yy = min(max(y + row_index - center_row, 0), rows - 1)
                for dx in range(-int(half_width), int(half_width) + 1):
                    xx = min(max(x + dx, 0), columns - 1)
                    count += int(source_plane[yy, xx])
            majority = count > majority_threshold
            center = bool(source_plane[y, x])
            output_plane[y, x] = (
                center and majority if remove_foreground else center or majority
            )


class _FakeCupy:
    def __init__(self) -> None:
        self.stream = _FakeStream()
        self.cuda = SimpleNamespace(get_current_stream=lambda: self.stream)
        self.kernels: list[_FakeRawKernel] = []
        self.launches: list[tuple[str, tuple[int, ...], tuple[int, ...], object]] = []

    asarray = staticmethod(np.asarray)
    ascontiguousarray = staticmethod(np.ascontiguousarray)
    empty = staticmethod(np.empty)

    def RawKernel(self, source, name: str, *, options=()):
        kernel = _FakeRawKernel(self, source, name, tuple(options))
        self.kernels.append(kernel)
        return kernel


def _clear_provider_caches() -> None:
    provider._cupy_module.cache_clear()
    provider._remove_binary_outliers_kernel.cache_clear()


@pytest.fixture
def fake_cupy(monkeypatch):
    cupy = _FakeCupy()
    real_import = importlib.import_module

    def load(name: str):
        return cupy if name == "cupy" else real_import(name)

    _clear_provider_caches()
    monkeypatch.setattr(provider.importlib, "import_module", load)
    yield cupy
    _clear_provider_caches()


@pytest.fixture(scope="module")
def real_cupy():
    try:
        cupy = importlib.import_module("cupy")
        if int(cupy.cuda.runtime.getDeviceCount()) < 1:
            pytest.skip("CuPy reports no CUDA device.")
        probe = cupy.arange(4, dtype=cupy.uint8)
        cupy.cuda.get_current_stream().synchronize()
        del probe
    except Exception as exc:
        pytest.skip(f"A working CuPy CUDA runtime is unavailable: {exc}")
    return cupy


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
module = importlib.import_module(
    "napari_vipp.core.gpu.cupy_remove_binary_outliers"
)
assert callable(module.remove_binary_outliers)
assert module.__all__ == ["remove_binary_outliers"]
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


@pytest.mark.parametrize(
    "radius",
    (0.5, 1.49, 1.5, 1.74, 1.75, 2.5, 2.84, 2.85, 3.0, 8.0, 25.0),
)
@pytest.mark.parametrize(
    "which_outliers",
    (FOREGROUND_OUTLIERS, BACKGROUND_OUTLIERS),
)
def test_fake_provider_matches_cpu_across_historical_footprints(
    fake_cupy,
    radius,
    which_outliers,
) -> None:
    rng = np.random.default_rng(50_900 + int(radius * 100))
    source = rng.random((7, 9)) > 0.56
    before = source.copy()

    expected = cpu_remove_binary_outliers(
        source,
        radius=radius,
        which_outliers=which_outliers,
    )
    actual = provider.remove_binary_outliers(
        source,
        radius=radius,
        which_outliers=which_outliers,
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, before)
    assert actual.dtype == np.dtype(bool)
    assert actual.flags.c_contiguous
    assert not np.shares_memory(actual, source)


@pytest.mark.parametrize("shape", ((1, 7), (7, 1), (2, 3), (2, 2, 3)))
@pytest.mark.parametrize(
    "which_outliers",
    (FOREGROUND_OUTLIERS, BACKGROUND_OUTLIERS),
)
def test_fake_provider_matches_nearest_edges_when_footprint_exceeds_image(
    fake_cupy,
    shape,
    which_outliers,
) -> None:
    source = np.indices(shape).sum(axis=0) % 3 == 0
    expected = cpu_remove_binary_outliers(
        source,
        radius=100.0,
        which_outliers=which_outliers,
    )
    actual = provider.remove_binary_outliers(
        source,
        radius=100.0,
        which_outliers=which_outliers,
    )
    np.testing.assert_array_equal(actual, expected)


def test_fake_provider_uses_original_plane_without_cascade(fake_cupy) -> None:
    source = np.zeros((1, 9), dtype=bool)
    source[0, 2:7] = True

    expected = cpu_remove_binary_outliers(source, radius=2.0)
    actual = provider.remove_binary_outliers(source, radius=2.0)

    np.testing.assert_array_equal(actual, expected)
    assert actual[0, 4]


def test_fake_provider_handles_noncontiguous_readonly_tzyx_without_mutation(
    fake_cupy,
) -> None:
    rng = np.random.default_rng(509)
    owner = rng.random((2, 3, 14, 18)) > 0.6
    source = owner[:, :, ::2, 1::2].transpose(1, 0, 2, 3)
    source.setflags(write=False)
    before = owner.copy()

    expected = cpu_remove_binary_outliers(
        source,
        radius=2.85,
        which_outliers=BACKGROUND_OUTLIERS,
    )
    actual = provider.remove_binary_outliers(
        source,
        radius=2.85,
        which_outliers=BACKGROUND_OUTLIERS,
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(owner, before)
    assert not source.flags.c_contiguous
    assert actual.flags.c_contiguous


def test_fake_parameter_sweep_reuses_one_fixed_kernel(fake_cupy) -> None:
    source_a = np.eye(7, dtype=bool)
    source_b = np.ones((2, 5, 9), dtype=bool)
    cases = (
        (source_a, 0.5, FOREGROUND_OUTLIERS),
        (source_b, 1.5, BACKGROUND_OUTLIERS),
        (source_a, 2.85, FOREGROUND_OUTLIERS),
        (source_b, 8.0, BACKGROUND_OUTLIERS),
        (source_a, 1.5, FOREGROUND_OUTLIERS),
        (source_b, 0.5, BACKGROUND_OUTLIERS),
    )
    for source, radius, which_outliers in cases:
        provider.remove_binary_outliers(
            source,
            radius=radius,
            which_outliers=which_outliers,
        )

    assert len(fake_cupy.kernels) == 1
    (kernel,) = fake_cupy.kernels
    assert kernel.name == "vipp_remove_binary_outliers_bool"
    assert "radius" not in kernel.source
    assert "which_outliers" not in kernel.source
    assert len(fake_cupy.launches) == sum(
        math.prod(source.shape[:-2]) if source.ndim > 2 else 1
        for source, _radius, _choice in cases
    )


def test_wide_radius_25_plane_is_split_by_sample_visit_budget() -> None:
    footprint_points = int(np.count_nonzero(imagej_remove_outliers_footprint(25.0)))
    columns = 4096
    tile_pixels = provider._pixel_tile_size(
        columns=columns,
        footprint_point_count=footprint_points,
    )

    assert tile_pixels < 64 * columns
    assert tile_pixels * footprint_points <= provider._TARGET_SAMPLE_VISITS_PER_TILE
    assert math.ceil((64 * columns) / tile_pixels) > 1


def test_fake_progress_is_synchronized_and_cancellable_per_pixel_tile(
    fake_cupy,
    monkeypatch,
) -> None:
    monkeypatch.setattr(provider, "_TARGET_SAMPLE_VISITS_PER_TILE", 63)
    updates = []
    source = np.ones((3, 15), dtype=bool)
    provider.remove_binary_outliers(
        source,
        radius=1.0,
        progress=ProgressContext(reporter=updates.append),
    )

    expected_tiles = math.ceil(source.size / 7)
    assert [(item.current, item.total) for item in updates] == [
        (current, expected_tiles) for current in range(expected_tiles + 1)
    ]
    assert fake_cupy.stream.synchronizations == expected_tiles

    _clear_provider_caches()
    fake_cupy.launches.clear()
    fake_cupy.kernels.clear()
    fake_cupy.stream.synchronizations = 0
    cancelled_updates = []
    with pytest.raises(OperationCancelled):
        provider.remove_binary_outliers(
            source,
            radius=1.0,
            progress=ProgressContext(
                cancelled=lambda: len(cancelled_updates) >= 2,
                reporter=cancelled_updates.append,
            ),
        )
    assert len(fake_cupy.launches) == 1
    assert fake_cupy.stream.synchronizations == 1


def test_fake_terminal_report_cancellation_prevents_result_return(fake_cupy) -> None:
    updates = []
    progress = ProgressContext(
        cancelled=lambda: len(updates) >= 2,
        reporter=updates.append,
    )

    with pytest.raises(OperationCancelled):
        provider.remove_binary_outliers(
            np.ones((3, 3), dtype=bool),
            radius=0.5,
            progress=progress,
        )

    assert len(fake_cupy.launches) == 1
    assert fake_cupy.stream.synchronizations == 1
    assert [(item.current, item.total) for item in updates] == [(0, 1), (1, 1)]


@pytest.mark.parametrize(
    ("source", "kwargs", "match"),
    (
        (np.ones((3, 3), dtype=np.uint8), {}, "boolean mask"),
        (np.empty((0, 3), dtype=bool), {}, "empty masks"),
        (np.ones((3,), dtype=bool), {}, "at least two dimensions"),
        (np.ones((3, 3), dtype=bool), {"radius": 0.49}, "radius"),
        (np.ones((3, 3), dtype=bool), {"radius": 100.1}, "radius"),
        (
            np.ones((3, 3), dtype=bool),
            {"which_outliers": "Both"},
            "outlier type",
        ),
    ),
)
def test_fake_provider_rejects_outside_direct_contract(
    fake_cupy,
    source,
    kwargs,
    match,
) -> None:
    with pytest.raises(ValueError, match=match):
        provider.remove_binary_outliers(source, **kwargs)


@pytest.mark.parametrize("radius", (0.5, 1.5, 1.75, 2.5, 2.85, 8.0, 25.0))
@pytest.mark.parametrize(
    "which_outliers",
    (FOREGROUND_OUTLIERS, BACKGROUND_OUTLIERS),
)
def test_real_cuda_matches_cpu_bitwise(
    real_cupy,
    radius,
    which_outliers,
) -> None:
    rng = np.random.default_rng(509_000 + int(radius * 100))
    source = rng.random((33, 37)) > 0.58
    expected = cpu_remove_binary_outliers(
        source,
        radius=radius,
        which_outliers=which_outliers,
    )
    device_source = real_cupy.asarray(source)
    before = device_source.copy()

    actual = provider.remove_binary_outliers(
        device_source,
        radius=radius,
        which_outliers=which_outliers,
    )

    np.testing.assert_array_equal(real_cupy.asnumpy(actual), expected)
    assert isinstance(actual, real_cupy.ndarray)
    assert actual.dtype == real_cupy.bool_
    assert actual.flags.c_contiguous
    assert bool(real_cupy.array_equal(device_source, before).item())


def test_real_cuda_matches_noncontiguous_tzyx_and_reuses_kernel(real_cupy) -> None:
    rng = np.random.default_rng(50_900)
    owner = rng.random((2, 3, 18, 22)) > 0.6
    host_source = owner[:, :, ::2, 1::2].transpose(1, 0, 2, 3)
    device_owner = real_cupy.asarray(owner)
    device_source = device_owner[:, :, ::2, 1::2].transpose(1, 0, 2, 3)
    before_kernel = provider._remove_binary_outliers_kernel(real_cupy)

    for radius, which_outliers in (
        (0.5, FOREGROUND_OUTLIERS),
        (2.85, BACKGROUND_OUTLIERS),
        (8.0, FOREGROUND_OUTLIERS),
        (0.5, BACKGROUND_OUTLIERS),
    ):
        expected = cpu_remove_binary_outliers(
            host_source,
            radius=radius,
            which_outliers=which_outliers,
        )
        actual = provider.remove_binary_outliers(
            device_source,
            radius=radius,
            which_outliers=which_outliers,
        )
        np.testing.assert_array_equal(real_cupy.asnumpy(actual), expected)

    assert provider._remove_binary_outliers_kernel(real_cupy) is before_kernel
    np.testing.assert_array_equal(real_cupy.asnumpy(device_owner), owner)


def test_real_cuda_progress_and_cancellation_use_synchronized_tiles(
    real_cupy,
    monkeypatch,
) -> None:
    monkeypatch.setattr(provider, "_TARGET_SAMPLE_VISITS_PER_TILE", 1024)
    source = real_cupy.ones((65, 129), dtype=real_cupy.bool_)
    updates = []
    result = provider.remove_binary_outliers(
        source,
        radius=2.0,
        progress=ProgressContext(reporter=updates.append),
    )
    assert updates[-1].current == updates[-1].total
    real_cupy.testing.assert_array_equal(result, source)

    cancelled_updates = []
    with pytest.raises(OperationCancelled):
        provider.remove_binary_outliers(
            source,
            radius=25.0,
            progress=ProgressContext(
                cancelled=lambda: len(cancelled_updates) >= 2,
                reporter=cancelled_updates.append,
            ),
        )
    assert [(item.current, item.total) for item in cancelled_updates[:2]] == [
        (0, cancelled_updates[0].total),
        (1, cancelled_updates[0].total),
    ]
