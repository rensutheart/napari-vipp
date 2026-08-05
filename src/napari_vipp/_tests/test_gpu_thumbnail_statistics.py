from __future__ import annotations

import numpy as np
import pytest

from napari_vipp.core.compute_registry import ComputeRegistry
from napari_vipp.core.gpu.cupy_thumbnail_statistics import (
    exact_uint_histogram_counts,
)
from napari_vipp.core.thumbnail_statistics import (
    ThumbnailStatisticsBackend,
    ThumbnailStatisticsEngine,
    ThumbnailStatisticsRequest,
)


def _working_cuda_registry():
    registry = ComputeRegistry()
    probe = registry.probe_runtime("cuda-cupy", refresh=True)
    if not probe.available:
        registry.close()
        pytest.skip(probe.message or "A working CuPy CUDA runtime is unavailable.")
    return registry, probe


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("channel_axis", [0, 1, 2])
def test_real_gpu_thumbnail_histogram_is_exact_multichannel_and_cleans_up(
    dtype,
    channel_axis,
):
    registry, probe = _working_cuda_registry()
    runtime = registry.runtime("cuda-cupy")
    rng = np.random.default_rng(537)
    maximum = np.iinfo(dtype).max + 1
    values = rng.integers(0, maximum, size=(7, 3, 11), dtype=dtype)
    level_count = maximum

    try:
        with runtime.execution_scope(
            device_id=probe.selected_device_id,
            memory_limit_bytes=512 * 1024**2,
            safety_reserve_bytes=256 * 1024**2,
        ):
            counts = exact_uint_histogram_counts(
                runtime,
                values,
                device_id=probe.selected_device_id,
                channel_axis=channel_axis,
            )

        expected = np.vstack(
            [
                np.bincount(
                    np.take(values, channel, axis=channel_axis)
                    .reshape(-1)
                    .astype(np.intp),
                    minlength=level_count,
                )
                for channel in range(values.shape[channel_axis])
            ]
        ).astype(np.uint64)
        np.testing.assert_array_equal(counts, expected)
        terminal = runtime.memory_snapshot(device_id=probe.selected_device_id)
        assert terminal.runtime_live_bytes == 0
        assert terminal.runtime_reserved_bytes == 0
    finally:
        registry.close()


def test_real_gpu_thumbnail_engine_returns_exact_limits_and_closes_runtime():
    registry, probe = _working_cuda_registry()
    registry.close()
    rng = np.random.default_rng(538)
    values = rng.integers(0, 65_536, size=(9, 31, 37), dtype=np.uint16)
    expected = np.percentile(values.astype(np.float32), (0.5, 99.9))
    engine = ThumbnailStatisticsEngine()

    result = engine.calculate(
        ThumbnailStatisticsRequest(
            values,
            compute_mode="prefer_gpu",
            device_id=probe.selected_device_id,
            accelerator_memory_cap_bytes=512 * 1024**2,
            accelerator_safety_reserve_bytes=256 * 1024**2,
        )
    )

    assert result.actual_backend is ThumbnailStatisticsBackend.GPU_CUPY
    assert not result.used_fallback
    assert engine.gpu_warm
    np.testing.assert_array_equal(
        np.asarray(result.limits).view(np.uint64),
        expected.view(np.uint64),
    )


@pytest.mark.parametrize("layout", ["fortran", "strided"])
def test_real_gpu_thumbnail_histogram_stages_noncontiguous_layout_exactly(layout):
    registry, probe = _working_cuda_registry()
    runtime = registry.runtime("cuda-cupy")
    rng = np.random.default_rng(539)
    base = rng.integers(0, 65_536, size=(17, 31, 37), dtype=np.uint16)
    values = np.asfortranarray(base) if layout == "fortran" else base[:, :, ::2]
    assert not values.flags.c_contiguous

    try:
        with runtime.execution_scope(
            device_id=probe.selected_device_id,
            memory_limit_bytes=512 * 1024**2,
            safety_reserve_bytes=256 * 1024**2,
        ):
            counts = exact_uint_histogram_counts(
                runtime,
                values,
                device_id=probe.selected_device_id,
            )

        expected = np.bincount(
            values.reshape(-1).astype(np.intp),
            minlength=65_536,
        ).astype(np.uint64)
        np.testing.assert_array_equal(counts, expected)
        terminal = runtime.memory_snapshot(device_id=probe.selected_device_id)
        assert terminal.runtime_live_bytes == 0
        assert terminal.runtime_reserved_bytes == 0
    finally:
        registry.close()
