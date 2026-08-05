from __future__ import annotations

from contextlib import AbstractContextManager
from types import SimpleNamespace

import numpy as np
import pytest

from napari_vipp.core.compute import ComputeMode
from napari_vipp.core.preview import (
    thumbnail_channel_contrast_limits,
    thumbnail_contrast_limits,
)
from napari_vipp.core.progress import (
    OperationCancelled,
    ProgressContext,
    ProgressUpdate,
)
from napari_vipp.core.thumbnail_statistics import (
    DEFAULT_COLD_GPU_THRESHOLD_BYTES,
    DEFAULT_COLD_UINT8_GPU_THRESHOLD_BYTES,
    DEFAULT_COLD_UINT16_GPU_THRESHOLD_BYTES,
    DEFAULT_WARM_GPU_THRESHOLD_BYTES,
    EXACT_NATIVE_MINMAX_ALGORITHM_ID,
    EXACT_UINT_HISTOGRAM_ALGORITHM_ID,
    ThumbnailStatisticsBackend,
    ThumbnailStatisticsCleanupError,
    ThumbnailStatisticsEngine,
    ThumbnailStatisticsMemoryError,
    ThumbnailStatisticsRequest,
    exact_uint_thumbnail_contrast_limits,
)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_exact_histogram_percentiles_are_bitwise_numpy_linear(dtype):
    rng = np.random.default_rng(535)
    maximum = np.iinfo(dtype).max + 1
    for size in (*range(1, 65), 257, 999, 1_000, 1_001, 65_535):
        values = rng.integers(0, maximum, size=size, dtype=dtype)
        expected = np.asarray(
            thumbnail_contrast_limits(values, contrast_mode="Percentile")
        )
        actual = np.asarray(
            exact_uint_thumbnail_contrast_limits(
                values,
                contrast_mode="Percentile",
            )
        )
        np.testing.assert_array_equal(
            actual.view(np.uint64),
            expected.view(np.uint64),
        )


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_exact_multichannel_histograms_match_existing_semantics(dtype, axis):
    rng = np.random.default_rng(536)
    maximum = np.iinfo(dtype).max + 1
    values = rng.integers(0, maximum, size=(5, 3, 7), dtype=dtype)

    actual = exact_uint_thumbnail_contrast_limits(values, channel_axis=axis)
    expected = thumbnail_channel_contrast_limits(values, channel_axis=axis)

    assert len(actual) == len(expected)
    for actual_limits, expected_limits in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(
            np.asarray(actual_limits).view(np.uint64),
            np.asarray(expected_limits).view(np.uint64),
        )


def test_exact_histogram_preserves_scan_free_and_collapsed_limit_contracts():
    values = np.zeros(10_000, dtype=np.uint16)
    values[-1] = 2

    assert exact_uint_thumbnail_contrast_limits(values) == (0.0, 2.0)
    assert exact_uint_thumbnail_contrast_limits(
        values,
        contrast_mode="Min-max",
    ) == (0.0, 2.0)
    assert (
        exact_uint_thumbnail_contrast_limits(
            values,
            contrast_mode="Raw",
        )
        is None
    )
    assert exact_uint_thumbnail_contrast_limits(values, data_kind="mask") == (
        0.0,
        1.0,
    )
    assert exact_uint_thumbnail_contrast_limits(
        np.zeros((2, 3), dtype=np.uint8),
        data_kind="mask",
        channel_axis=0,
    ) == ((0.0, 1.0), (0.0, 1.0))
    assert (
        exact_uint_thumbnail_contrast_limits(
            np.zeros((2, 3), dtype=np.uint8),
            channel_axis=8,
        )
        == ()
    )


def test_cpu_histogram_progress_is_chunked_monotonic_and_cancelable():
    values = np.arange(100, dtype=np.uint16)
    updates: list[ProgressUpdate] = []
    cancelled = False

    def report(update: ProgressUpdate) -> None:
        nonlocal cancelled
        updates.append(update)
        if update.current >= 20:
            cancelled = True

    progress = ProgressContext(
        cancelled=lambda: cancelled,
        reporter=report,
    )

    with pytest.raises(OperationCancelled):
        exact_uint_thumbnail_contrast_limits(
            values,
            progress=progress,
            chunk_elements=20,
        )

    assert [update.current for update in updates] == [0, 20]
    assert all(update.total == values.size for update in updates)


def test_cpu_mode_and_small_auto_requests_never_construct_gpu_registry():
    calls = 0

    def forbidden_registry():
        nonlocal calls
        calls += 1
        raise AssertionError("CPU selection must not construct or probe CUDA.")

    engine = ThumbnailStatisticsEngine(
        cold_gpu_threshold_bytes=1_000,
        warm_gpu_threshold_bytes=100,
        registry_factory=forbidden_registry,
    )
    values = np.arange(100, dtype=np.uint16)

    cpu = engine.calculate(
        ThumbnailStatisticsRequest(values, compute_mode=ComputeMode.CPU)
    )
    automatic = engine.calculate(
        ThumbnailStatisticsRequest(values, compute_mode=ComputeMode.AUTO)
    )

    assert calls == 0
    assert cpu.actual_backend is ThumbnailStatisticsBackend.CPU_NUMPY
    assert automatic.actual_backend is ThumbnailStatisticsBackend.CPU_NUMPY
    assert automatic.decision.reason_code == "auto_below_cold_gpu_threshold"


def test_selector_uses_byte_threshold_warm_state_and_prefer_gpu():
    engine = ThumbnailStatisticsEngine(
        cold_gpu_threshold_bytes=100,
        warm_gpu_threshold_bytes=20,
        registry_factory=lambda: _SuccessfulRegistry(),
    )
    values = np.arange(30, dtype=np.uint16)
    automatic_request = ThumbnailStatisticsRequest(
        values,
        compute_mode=ComputeMode.AUTO,
    )

    assert engine.select(automatic_request).backend is (
        ThumbnailStatisticsBackend.CPU_NUMPY
    )
    preferred = engine.calculate(
        ThumbnailStatisticsRequest(values, compute_mode=ComputeMode.PREFER_GPU)
    )
    assert preferred.actual_backend is ThumbnailStatisticsBackend.GPU_CUPY
    assert preferred.algorithm_id == EXACT_UINT_HISTOGRAM_ALGORITHM_ID
    assert engine.gpu_warm
    assert engine.select(automatic_request).backend is (
        ThumbnailStatisticsBackend.GPU_CUPY
    )
    assert engine.select(automatic_request).reason_code == "auto_gpu_threshold_met"


def test_prefer_gpu_ineligible_and_memory_cap_decisions_are_visible_fallbacks():
    engine = ThumbnailStatisticsEngine()
    float_result = engine.calculate(
        ThumbnailStatisticsRequest(
            np.linspace(0, 1, 20, dtype=np.float32),
            compute_mode=ComputeMode.PREFER_GPU,
        )
    )
    capped_result = engine.calculate(
        ThumbnailStatisticsRequest(
            np.arange(20, dtype=np.uint16),
            compute_mode=ComputeMode.PREFER_GPU,
            accelerator_memory_cap_bytes=1,
        )
    )

    assert float_result.fallback_reason_code == "gpu_ineligible"
    assert capped_result.fallback_reason_code == "gpu_memory_cap_insufficient"
    assert float_result.fallback_message
    assert capped_result.fallback_message


def test_unavailable_gpu_falls_back_with_typed_metadata_and_closes_registry():
    registry = _UnavailableRegistry()
    engine = ThumbnailStatisticsEngine(registry_factory=lambda: registry)

    result = engine.calculate(
        ThumbnailStatisticsRequest(
            np.arange(100, dtype=np.uint16),
            compute_mode=ComputeMode.PREFER_GPU,
        )
    )

    assert result.actual_backend is ThumbnailStatisticsBackend.CPU_NUMPY
    assert result.fallback_reason_code == "dependency_missing"
    assert "not installed" in result.fallback_message
    assert registry.closed
    assert not engine.gpu_warm


def test_gpu_failure_falls_back_only_after_successful_runtime_cleanup():
    registry = _ExecutionFailureRegistry(reason_code="cuda_out_of_memory")
    engine = ThumbnailStatisticsEngine(registry_factory=lambda: registry)

    result = engine.calculate(
        ThumbnailStatisticsRequest(
            np.arange(100, dtype=np.uint16),
            compute_mode=ComputeMode.PREFER_GPU,
        )
    )

    assert result.actual_backend is ThumbnailStatisticsBackend.CPU_NUMPY
    assert result.fallback_reason_code == "cuda_out_of_memory"
    assert registry.runtime_instance.scope_exited
    assert registry.closed


def test_gpu_to_cpu_fallback_progress_is_monotonic_and_names_the_phase(monkeypatch):
    registry = _ExecutionFailureRegistry(reason_code="cuda_out_of_memory")
    engine = ThumbnailStatisticsEngine(
        registry_factory=lambda: registry,
        cpu_chunk_elements=25,
    )
    updates = []

    def fail_after_gpu_progress(_runtime, _arr, *, progress=None, **_kwargs):
        assert progress is not None
        progress.report(3, 4, "Returning thumbnail histogram from GPU")
        raise _ExecutionFailure("GPU attempt failed")

    monkeypatch.setattr(
        "napari_vipp.core.thumbnail_statistics._calculate_cupy_counts",
        fail_after_gpu_progress,
    )

    result = engine.calculate(
        ThumbnailStatisticsRequest(
            np.arange(100, dtype=np.uint16),
            compute_mode=ComputeMode.PREFER_GPU,
        ),
        progress=ProgressContext(reporter=updates.append),
    )

    fractions = [
        update.current / update.total
        for update in updates
        if update.total > 0
    ]
    assert fractions == sorted(fractions)
    assert fractions[0] == pytest.approx(0.75)
    assert fractions[-1] == pytest.approx(1.0)
    assert any(update.message.startswith("CPU fallback") for update in updates)
    assert result.actual_backend is ThumbnailStatisticsBackend.CPU_NUMPY
    assert result.fallback_reason_code == "cuda_out_of_memory"


def test_gpu_cancellation_closes_runtime_and_never_falls_back():
    registry = _SuccessfulRegistry()
    engine = ThumbnailStatisticsEngine(registry_factory=lambda: registry)
    checks = 0

    def cancel_inside_runtime() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 6

    progress = ProgressContext(cancelled=cancel_inside_runtime)

    with pytest.raises(OperationCancelled):
        engine.calculate(
            ThumbnailStatisticsRequest(
                np.arange(100, dtype=np.uint16),
                compute_mode=ComputeMode.PREFER_GPU,
            ),
            progress=progress,
        )

    assert registry.runtime_instance.scope_exited
    assert registry.closed
    assert not engine.gpu_warm


@pytest.mark.parametrize("failure_site", ["scope", "registry"])
def test_gpu_cleanup_failure_is_never_masked_by_cpu_fallback(failure_site):
    registry = _CleanupFailureRegistry(failure_site)
    engine = ThumbnailStatisticsEngine(registry_factory=lambda: registry)

    with pytest.raises(ThumbnailStatisticsCleanupError, match="no CPU fallback"):
        engine.calculate(
            ThumbnailStatisticsRequest(
                np.arange(100, dtype=np.uint16),
                compute_mode=ComputeMode.PREFER_GPU,
            )
        )

    assert registry.closed
    assert not engine.gpu_warm


def test_float_cpu_path_matches_existing_thumbnail_limits():
    values = np.asarray([np.nan, -2.0, 0.0, 1.5, 9.0], dtype=np.float32)
    engine = ThumbnailStatisticsEngine()

    result = engine.calculate(
        ThumbnailStatisticsRequest(values, compute_mode=ComputeMode.CPU)
    )

    assert result.limits == thumbnail_contrast_limits(values)


@pytest.mark.parametrize(
    "values",
    [
        np.asarray([-np.inf, 1.0, 2.0], dtype=np.float32),
        np.asarray([-np.inf, -2.0, 1.0, np.inf, np.nan], dtype=np.float32),
    ],
)
def test_float_percentile_preserves_infinity_and_negative_clipping_contract(values):
    result = ThumbnailStatisticsEngine(cpu_chunk_elements=2).calculate(
        ThumbnailStatisticsRequest(values, compute_mode=ComputeMode.CPU)
    )

    np.testing.assert_array_equal(
        np.asarray(result.limits).view(np.uint64),
        np.asarray(thumbnail_contrast_limits(values)).view(np.uint64),
    )


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_chunked_native_minmax_is_exact_and_never_probes_gpu(dtype):
    calls = 0

    def forbidden_registry():
        nonlocal calls
        calls += 1
        raise AssertionError("Min-max must use the cheaper CPU reduction.")

    source = (
        [np.nan, -3.0, 0.0, 8.0]
        if np.issubdtype(dtype, np.floating)
        else [0, 3, 0, 8]
    )
    values = np.asarray(source, dtype=dtype)
    engine = ThumbnailStatisticsEngine(
        registry_factory=forbidden_registry,
        cpu_chunk_elements=2,
    )
    updates = []
    result = engine.calculate(
        ThumbnailStatisticsRequest(
            values,
            contrast_mode="Min-max",
            compute_mode=ComputeMode.PREFER_GPU,
        ),
        progress=ProgressContext(reporter=updates.append),
    )

    expected = thumbnail_contrast_limits(values, contrast_mode="Min-max")
    assert result.limits == expected
    assert result.algorithm_id == EXACT_NATIVE_MINMAX_ALGORITHM_ID
    assert result.actual_backend is ThumbnailStatisticsBackend.CPU_NUMPY
    assert result.fallback_reason_code == "gpu_ineligible"
    assert calls == 0
    assert [update.current for update in updates] == [0, 2, 4]


def test_multichannel_uint16_histograms_stream_one_counter_vector(monkeypatch):
    values = np.arange(512 * 2, dtype=np.uint16).reshape(512, 2)
    allocated_shapes = []
    real_zeros = np.zeros

    def recording_zeros(shape, *args, **kwargs):
        allocated_shapes.append(shape)
        return real_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(
        "napari_vipp.core.thumbnail_statistics.np.zeros",
        recording_zeros,
    )

    limits = exact_uint_thumbnail_contrast_limits(values, channel_axis=0)

    assert len(limits) == 512
    assert allocated_shapes
    assert all(shape == 65_536 for shape in allocated_shapes)


def test_large_spectral_counter_matrix_uses_streamed_cpu_fallback():
    calls = 0

    def forbidden_registry():
        nonlocal calls
        calls += 1
        raise AssertionError("Oversized GPU counter matrix must be rejected early.")

    values = np.arange(129, dtype=np.uint16).reshape(129, 1)
    engine = ThumbnailStatisticsEngine(registry_factory=forbidden_registry)

    result = engine.calculate(
        ThumbnailStatisticsRequest(
            values,
            channel_axis=0,
            compute_mode=ComputeMode.PREFER_GPU,
        )
    )

    assert len(result.limits) == 129
    assert result.actual_backend is ThumbnailStatisticsBackend.CPU_NUMPY
    assert result.fallback_reason_code == "gpu_counter_allocation_too_large"
    assert calls == 0


def test_integer_minmax_never_builds_a_float_finite_mask(monkeypatch):
    values = np.arange(100, dtype=np.uint16)

    def forbidden_isfinite(_values):
        raise AssertionError("Native integer min-max must not build a finite mask.")

    monkeypatch.setattr(
        "napari_vipp.core.thumbnail_statistics.np.isfinite",
        forbidden_isfinite,
    )

    result = ThumbnailStatisticsEngine(cpu_chunk_elements=13).calculate(
        ThumbnailStatisticsRequest(
            values,
            contrast_mode="Min-max",
            compute_mode=ComputeMode.CPU,
        )
    )

    assert result.limits == (0.0, 99.0)


def test_stable_unavailable_gpu_probe_is_cached_until_explicit_reset():
    calls = 0

    def registry_factory():
        nonlocal calls
        calls += 1
        return _UnavailableRegistry()

    engine = ThumbnailStatisticsEngine(registry_factory=registry_factory)
    request = ThumbnailStatisticsRequest(
        np.arange(100, dtype=np.uint16),
        compute_mode=ComputeMode.PREFER_GPU,
    )

    first = engine.calculate(request)
    second = engine.calculate(request)

    assert first.fallback_reason_code == "dependency_missing"
    assert second.fallback_reason_code == "gpu_session_unavailable"
    assert calls == 1

    engine.reset_accelerator_capability()
    third = engine.calculate(request)
    assert third.fallback_reason_code == "dependency_missing"
    assert calls == 2


def test_auto_cold_threshold_is_dtype_specific_and_warmth_is_per_dtype():
    engine = ThumbnailStatisticsEngine(registry_factory=lambda: _SuccessfulRegistry())
    uint8_request = ThumbnailStatisticsRequest(
        np.arange(20, dtype=np.uint8),
        compute_mode=ComputeMode.AUTO,
    )
    uint16_request = ThumbnailStatisticsRequest(
        np.arange(20, dtype=np.uint16),
        compute_mode=ComputeMode.AUTO,
    )

    assert engine.select(uint8_request).threshold_bytes == 384 * 1024**2
    assert engine.select(uint16_request).threshold_bytes == 512 * 1024**2

    engine.calculate(
        ThumbnailStatisticsRequest(
            np.arange(20, dtype=np.uint8),
            compute_mode=ComputeMode.PREFER_GPU,
        )
    )
    assert engine.select(uint8_request).threshold_bytes == 32 * 1024**2
    assert engine.select(uint16_request).threshold_bytes == 512 * 1024**2


def test_noncontiguous_auto_stays_cpu_and_exposes_staging_bytes():
    values = np.arange(48, dtype=np.uint16).reshape(6, 8).T
    engine = ThumbnailStatisticsEngine(
        cold_gpu_threshold_bytes=1,
        warm_gpu_threshold_bytes=1,
    )

    decision = engine.select(
        ThumbnailStatisticsRequest(values, compute_mode=ComputeMode.AUTO)
    )

    assert decision.backend is ThumbnailStatisticsBackend.CPU_NUMPY
    assert decision.reason_code == "auto_noncontiguous_host_staging"
    assert decision.host_staging_bytes == values.nbytes


def test_float_percentile_progress_names_noninterruptible_inner_pass():
    values = np.asarray([np.nan, -2.0, 0.0, 1.5, 9.0], dtype=np.float64)
    engine = ThumbnailStatisticsEngine(cpu_chunk_elements=2)
    updates = []

    result = engine.calculate(
        ThumbnailStatisticsRequest(values, compute_mode=ComputeMode.CPU),
        progress=ProgressContext(reporter=updates.append),
    )

    assert result.limits == thumbnail_contrast_limits(values)
    assert any(
        "cancel applies after this pass" in update.message
        for update in updates
    )
    assert updates[-1].current == updates[-1].total


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_float_channel_percentiles_preserve_existing_semantics(axis):
    rng = np.random.default_rng(540)
    values = rng.normal(size=(4, 5, 6)).astype(np.float64)
    values[0, 0, 0] = np.nan
    values[1, 1, 1] = -9.0
    engine = ThumbnailStatisticsEngine(cpu_chunk_elements=7)

    result = engine.calculate(
        ThumbnailStatisticsRequest(
            values,
            channel_axis=axis,
            compute_mode=ComputeMode.CPU,
        )
    )
    expected = thumbnail_channel_contrast_limits(values, channel_axis=axis)

    assert len(result.limits) == len(expected)
    for actual_limits, expected_limits in zip(result.limits, expected, strict=True):
        np.testing.assert_array_equal(
            np.asarray(actual_limits).view(np.uint64),
            np.asarray(expected_limits).view(np.uint64),
        )


def test_large_float_percentile_fails_before_workspace_when_host_admission_rejects(
    monkeypatch,
):
    values = np.broadcast_to(np.float32(1.0), (20_000_000,))
    monkeypatch.setattr(
        "napari_vipp.core.thumbnail_statistics._host_allocation_rejection",
        lambda required_bytes, *, purpose: (
            f"Skipped {purpose}: need {required_bytes} bytes."
        ),
    )

    with pytest.raises(ThumbnailStatisticsMemoryError, match="need") as error:
        ThumbnailStatisticsEngine().calculate(
            ThumbnailStatisticsRequest(values, compute_mode=ComputeMode.CPU)
        )

    assert error.value.required_bytes >= values.size * np.dtype(np.float32).itemsize


def test_none_and_invalid_channel_metadata_preserve_existing_empty_contracts():
    engine = ThumbnailStatisticsEngine()

    none_result = engine.calculate(
        ThumbnailStatisticsRequest(None, compute_mode=ComputeMode.CPU)
    )
    invalid_channel = engine.calculate(
        ThumbnailStatisticsRequest(
            np.arange(12, dtype=np.float32).reshape(3, 4),
            channel_axis=4,
            compute_mode=ComputeMode.CPU,
        )
    )

    assert none_result.limits is None
    assert none_result.decision.reason_code == "scan_free"
    assert invalid_channel.limits == ()
    assert invalid_channel.decision.reason_code == "scan_free"


class _Scope(AbstractContextManager):
    def __init__(self, runtime, *, body_error=None, cleanup_error=None):
        self.runtime = runtime
        self.body_error = body_error
        self.cleanup_error = cleanup_error

    def __enter__(self):
        self.runtime.scope_entered = True
        if self.body_error is not None:
            raise self.body_error
        return None

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback
        self.runtime.scope_exited = True
        if self.cleanup_error is not None:
            raise self.cleanup_error
        return None


class _SuccessfulRuntime:
    scope_entered = False
    scope_exited = False

    def execution_scope(self, **_kwargs):
        return _Scope(self)

    def classify_exception(self, exc):
        return SimpleNamespace(
            reason_code="gpu_execution_failed",
            message=str(exc),
            kind=SimpleNamespace(value="unknown"),
        )


class _SuccessfulRegistry:
    def __init__(self):
        self.runtime_instance = _SuccessfulRuntime()
        self.closed = False

    def probe_runtime(self, _runtime_id):
        return SimpleNamespace(
            available=True,
            selected_device_id="cuda:0",
            reason_code="",
            message="",
        )

    def runtime(self, _runtime_id):
        return self.runtime_instance

    def close(self):
        self.closed = True


class _UnavailableRegistry:
    def __init__(self):
        self.closed = False

    def probe_runtime(self, _runtime_id):
        return SimpleNamespace(
            available=False,
            selected_device_id="",
            reason_code="dependency_missing",
            message="CuPy is not installed.",
        )

    def close(self):
        self.closed = True


class _ExecutionFailure(RuntimeError):
    pass


class _ExecutionFailureRuntime(_SuccessfulRuntime):
    def __init__(self, reason_code):
        self.reason_code = reason_code
        self.calculation_error = _ExecutionFailure("GPU attempt failed")
        self.scope_entered = False
        self.scope_exited = False

    def execution_scope(self, **_kwargs):
        return _Scope(self)

    def classify_exception(self, exc):
        return SimpleNamespace(
            reason_code=self.reason_code,
            message=str(exc),
            kind=SimpleNamespace(value="out_of_memory"),
        )


class _ExecutionFailureRegistry(_SuccessfulRegistry):
    def __init__(self, reason_code):
        self.runtime_instance = _ExecutionFailureRuntime(reason_code)
        self.closed = False


class _CleanupFailure(RuntimeError):
    pass


class _CleanupFailureRuntime(_SuccessfulRuntime):
    def __init__(self, fail_scope):
        self.fail_scope = fail_scope
        self.scope_entered = False
        self.scope_exited = False

    def execution_scope(self, **_kwargs):
        return _Scope(
            self,
            cleanup_error=(
                _CleanupFailure("scope cleanup failed") if self.fail_scope else None
            ),
        )

    def classify_exception(self, exc):
        return SimpleNamespace(
            reason_code="cuda_cleanup_incomplete",
            message=str(exc),
            kind=SimpleNamespace(value="kernel_failure"),
        )


class _CleanupFailureRegistry(_SuccessfulRegistry):
    def __init__(self, failure_site):
        self.failure_site = failure_site
        self.runtime_instance = _CleanupFailureRuntime(failure_site == "scope")
        self.closed = False

    def close(self):
        self.closed = True
        if self.failure_site == "registry":
            raise _CleanupFailure("registry cleanup failed")


@pytest.fixture(autouse=True)
def _replace_cupy_calculation(monkeypatch):
    """Keep core lifecycle tests provider-free; real CUDA has its own test file."""

    def fake_counts(_runtime, arr, **kwargs):
        progress = kwargs.get("progress")
        if progress is not None:
            progress.check_cancelled()
        if getattr(_runtime, "calculation_error", None) is not None:
            raise _runtime.calculation_error
        level_count = 256 if arr.dtype == np.uint8 else 65_536
        return np.bincount(
            arr.reshape(-1).astype(np.intp),
            minlength=level_count,
        ).astype(np.uint64)

    monkeypatch.setattr(
        "napari_vipp.core.thumbnail_statistics._calculate_cupy_counts",
        fake_counts,
    )


def test_default_threshold_contracts_remain_conservative():
    assert DEFAULT_COLD_UINT8_GPU_THRESHOLD_BYTES == 384 * 1024**2
    assert DEFAULT_COLD_UINT16_GPU_THRESHOLD_BYTES == 512 * 1024**2
    assert DEFAULT_COLD_GPU_THRESHOLD_BYTES == DEFAULT_COLD_UINT16_GPU_THRESHOLD_BYTES
    assert DEFAULT_WARM_GPU_THRESHOLD_BYTES == 32 * 1024**2
