from __future__ import annotations

from contextlib import AbstractContextManager
from types import SimpleNamespace

import numpy as np
import pytest

from napari_vipp.core.compute import ComputeMode
from napari_vipp.core.gpu.cupy_thumbnail_statistics import (
    _float32_radix_histogram_kernel,
    _linear_percentile_from_selected,
    _linear_percentile_indices,
    _radix_select_float32_bits,
)
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
    EXACT_FLOAT32_MINMAX_GPU_ALGORITHM_ID,
    EXACT_FLOAT32_PERCENTILE_GPU_ALGORITHM_ID,
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


@pytest.mark.parametrize("size", [1, 2, 101, 201, 257, 1_001])
def test_float32_gpu_host_interpolation_is_bitwise_numpy_linear(size):
    values = np.linspace(0.0, 19.0, size, dtype=np.float32)
    values[-1] = np.finfo(np.float32).max
    ordered = np.sort(values)
    virtual, previous, following = _linear_percentile_indices(size)
    selected = np.asarray(
        [
            ordered[0],
            ordered[-1],
            ordered[previous[0]],
            ordered[following[0]],
            ordered[previous[1]],
            ordered[following[1]],
        ],
        dtype=np.float32,
    )

    actual = _linear_percentile_from_selected(
        selected,
        virtual=virtual,
        previous_indices=previous,
    )
    expected = np.percentile(values, (0.5, 99.9))

    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint64),
        np.asarray(expected).view(np.uint64),
    )


def test_float32_radix_selector_reuses_one_fixed_program_across_values_and_sizes():
    class FakeKernel:
        def __init__(self):
            self.calls = []

        def __call__(self, grid, block, arguments):
            self.calls.append((grid, block, arguments[5], arguments[6]))
            (
                values,
                _size,
                include_negative_infinity,
                prefixes,
                _prefix_count,
                prefix_bits,
                byte_shift,
                histograms,
            ) = arguments
            for bits in np.asarray(values, dtype=np.float32).view(np.uint32):
                magnitude = int(bits) & 0x7FFFFFFF
                finite = magnitude & 0x7F800000 != 0x7F800000
                negative_finite = bool(finite and int(bits) & 0x80000000 and magnitude)
                if finite:
                    key = 0 if negative_finite else magnitude
                elif int(bits) == 0xFF800000 and int(include_negative_infinity):
                    key = 0
                else:
                    continue
                resolved_bits = int(prefix_bits)
                prefix = 0 if resolved_bits == 0 else key >> (32 - resolved_bits)
                bucket = (key >> int(byte_shift)) & 0xFF
                for target, expected_prefix in enumerate(prefixes):
                    if int(expected_prefix) == prefix:
                        histograms[target, bucket] += np.uint64(1)

    class FakeCuPy:
        uint64 = np.uint64

        def __init__(self):
            self.kernel = FakeKernel()
            self.factories = []

        def RawKernel(self, source, name, options):
            self.factories.append((source, name, options))
            return self.kernel

        @staticmethod
        def zeros(shape, dtype):
            return np.zeros(shape, dtype=dtype)

    class FakeRuntime:
        @staticmethod
        def to_device(value, *, device_id):
            del device_id
            return np.asarray(value).copy()

        @staticmethod
        def to_host(value):
            return np.asarray(value).copy()

        @staticmethod
        def synchronize(*, device_id):
            del device_id

        @staticmethod
        def release(value):
            assert isinstance(value, np.ndarray)

    fake_cupy = FakeCuPy()
    runtime = FakeRuntime()
    _float32_radix_histogram_kernel.cache_clear()
    try:
        for values, ranks in (
            (
                np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
                np.asarray([0, 1, 2, 3], dtype=np.uint64),
            ),
            (
                np.asarray([0.0, 0.5, 1.5, 7.0, 9.0], dtype=np.float32),
                np.asarray([0, 1, 3, 4], dtype=np.uint64),
            ),
        ):
            selected = _radix_select_float32_bits(
                runtime,
                fake_cupy,
                values,
                device_id="cuda:0",
                include_negative_infinity=False,
                target_ranks=ranks,
                block_count=1,
                progress=None,
                completed_work=0,
                total_work=values.size * 4,
                channel_size=values.size,
                channel_number=1,
                channel_count=1,
            )
            np.testing.assert_array_equal(
                selected.bits,
                values.view(np.uint32)[ranks.astype(np.intp)],
            )
            assert 16 <= selected.auxiliary_host_to_device_bytes <= 64
            assert selected.device_to_host_values == (
                selected.auxiliary_host_to_device_bytes // 4 * 256
            )
            assert selected.device_to_host_bytes == selected.device_to_host_values * 8

        assert len(fake_cupy.factories) == 1
        source, name, options = fake_cupy.factories[0]
        assert name == "vipp_thumbnail_float32_radix_histogram"
        assert options == ("--std=c++11",)
        assert "prefix_bits" in source
        assert "byte_shift" in source
        assert len(fake_cupy.kernel.calls) == 8
        assert [int(call[2]) for call in fake_cupy.kernel.calls] == [
            0,
            8,
            16,
            24,
        ] * 2
        assert [int(call[3]) for call in fake_cupy.kernel.calls] == [
            24,
            16,
            8,
            0,
        ] * 2
    finally:
        _float32_radix_histogram_kernel.cache_clear()


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
    assert cpu.requested_compute_mode is ComputeMode.CPU
    assert automatic.requested_compute_mode is ComputeMode.AUTO
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
    assert preferred.requested_compute_mode is ComputeMode.PREFER_GPU
    assert preferred.algorithm_id == EXACT_UINT_HISTOGRAM_ALGORITHM_ID
    assert preferred.input_path == "host_upload"
    assert preferred.logical_input_host_to_device_bytes == values.nbytes
    assert preferred.auxiliary_host_to_device_bytes == 0
    assert preferred.device_to_host_values == 65_536
    assert preferred.device_to_host_bytes == 65_536 * np.dtype(np.uint64).itemsize
    assert engine.gpu_warm
    assert engine.select(automatic_request).backend is (
        ThumbnailStatisticsBackend.GPU_CUPY
    )
    assert engine.select(automatic_request).reason_code == "auto_gpu_threshold_met"


def test_float32_minmax_warmth_does_not_claim_percentile_is_warm():
    values = np.arange(32, dtype=np.float32)
    engine = ThumbnailStatisticsEngine(registry_factory=lambda: _SuccessfulRegistry())

    engine.calculate(
        ThumbnailStatisticsRequest(
            values,
            contrast_mode="Min-max",
            compute_mode=ComputeMode.PREFER_GPU,
        )
    )

    minmax = engine.select(
        ThumbnailStatisticsRequest(
            values,
            contrast_mode="Min-max",
            compute_mode=ComputeMode.AUTO,
        )
    )
    percentile = engine.select(
        ThumbnailStatisticsRequest(
            values,
            contrast_mode="Percentile",
            compute_mode=ComputeMode.AUTO,
        )
    )
    assert minmax.gpu_warm
    assert minmax.threshold_bytes == DEFAULT_WARM_GPU_THRESHOLD_BYTES
    assert not percentile.gpu_warm
    assert percentile.threshold_bytes == DEFAULT_COLD_GPU_THRESHOLD_BYTES


def test_resident_gpu_warm_evidence_is_exact_and_preserves_contract_separation():
    engine = ThumbnailStatisticsEngine()

    assert not engine.gpu_contract_is_warm(np.float32, "Min-max")
    assert not engine.gpu_contract_is_warm(np.float32, "Percentile")
    engine.record_resident_gpu_success(np.dtype(np.float32), "Min-max")

    assert engine.gpu_warm
    assert engine.gpu_contract_is_warm(np.float32, "Min-max")
    assert not engine.gpu_contract_is_warm(np.float32, "Percentile")
    engine.record_resident_gpu_success(np.float32, "Percentile")
    engine.record_resident_gpu_success(np.uint16, "Percentile")
    assert engine.gpu_contract_is_warm(np.float32, "Percentile")
    assert engine.gpu_contract_is_warm(np.dtype("uint16"), "Percentile")


@pytest.mark.parametrize(
    ("dtype", "contrast_mode"),
    [
        (np.uint8, "Min-max"),
        (np.uint16, "Raw"),
        (np.float64, "Percentile"),
        (np.float32, "Raw"),
    ],
)
def test_resident_gpu_warm_evidence_rejects_unsupported_contracts(
    dtype,
    contrast_mode,
):
    engine = ThumbnailStatisticsEngine()

    assert not engine.gpu_contract_is_warm(dtype, contrast_mode)
    with pytest.raises(ValueError, match="GPU thumbnail warmth supports"):
        engine.record_resident_gpu_success(dtype, contrast_mode)
    assert not engine.gpu_warm


@pytest.mark.parametrize("contrast_mode", ["Percentile", "Min-max"])
def test_float32_memory_cap_includes_input_channel_copy_and_fixed_workspace(
    contrast_mode,
):
    values = np.zeros((4, 3, 5), dtype=np.float32)
    fixed_workspace = 5 * 8 + 4 * 256 * 8 + 4 * 4 + 8 * 1024**2
    required = values.nbytes + values.nbytes // values.shape[1] + fixed_workspace
    engine = ThumbnailStatisticsEngine()

    rejected = engine.select(
        ThumbnailStatisticsRequest(
            values,
            contrast_mode=contrast_mode,
            channel_axis=1,
            compute_mode=ComputeMode.PREFER_GPU,
            accelerator_memory_cap_bytes=required - 1,
        )
    )
    admitted = engine.select(
        ThumbnailStatisticsRequest(
            values,
            contrast_mode=contrast_mode,
            channel_axis=1,
            compute_mode=ComputeMode.PREFER_GPU,
            accelerator_memory_cap_bytes=required,
        )
    )

    assert rejected.backend is ThumbnailStatisticsBackend.CPU_NUMPY
    assert rejected.reason_code == "gpu_memory_cap_insufficient"
    assert admitted.backend is ThumbnailStatisticsBackend.GPU_CUPY


@pytest.mark.parametrize(
    ("contrast_mode", "algorithm_id"),
    [
        ("Percentile", EXACT_FLOAT32_PERCENTILE_GPU_ALGORITHM_ID),
        ("Min-max", EXACT_FLOAT32_MINMAX_GPU_ALGORITHM_ID),
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        np.asarray([np.nan, -np.inf, np.inf], dtype=np.float32),
        np.asarray([-0.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([-3.0, -np.inf, -0.0, 2.0, np.inf, np.nan], dtype=np.float32),
        np.asarray(
            [
                np.nextafter(np.float32(0.0), np.float32(1.0)),
                np.float32(1.0),
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        ),
    ],
)
def test_prefer_gpu_float32_matches_cpu_contract_with_typed_provenance(
    contrast_mode,
    algorithm_id,
    values,
):
    engine = ThumbnailStatisticsEngine(registry_factory=lambda: _SuccessfulRegistry())
    expected = ThumbnailStatisticsEngine().calculate(
        ThumbnailStatisticsRequest(
            values,
            contrast_mode=contrast_mode,
            compute_mode=ComputeMode.CPU,
        )
    )

    actual = engine.calculate(
        ThumbnailStatisticsRequest(
            values,
            contrast_mode=contrast_mode,
            compute_mode=ComputeMode.PREFER_GPU,
        )
    )

    assert actual.actual_backend is ThumbnailStatisticsBackend.GPU_CUPY
    assert actual.algorithm_id == algorithm_id
    assert actual.input_path == "host_upload"
    assert actual.logical_input_host_to_device_bytes == values.nbytes
    assert actual.auxiliary_host_to_device_bytes == 17
    assert actual.device_to_host_values == 29
    assert actual.device_to_host_bytes == 23
    assert not actual.used_fallback
    np.testing.assert_array_equal(
        np.asarray(actual.limits).view(np.uint64),
        np.asarray(expected.limits).view(np.uint64),
    )


@pytest.mark.parametrize("contrast_mode", ["Percentile", "Min-max"])
@pytest.mark.parametrize("channel_axis", [0, 1, -1])
def test_prefer_gpu_float32_channel_limits_match_cpu_bitwise(
    contrast_mode,
    channel_axis,
):
    rng = np.random.default_rng(541)
    values = rng.normal(size=(4, 3, 5)).astype(np.float32)
    values[0, 0, 0] = np.nan
    values[1, 1, 1] = -np.inf
    values[2, 2, 2] = np.inf
    engine = ThumbnailStatisticsEngine(registry_factory=lambda: _SuccessfulRegistry())
    cpu = ThumbnailStatisticsEngine().calculate(
        ThumbnailStatisticsRequest(
            values,
            contrast_mode=contrast_mode,
            channel_axis=channel_axis,
            compute_mode=ComputeMode.CPU,
        )
    )

    gpu = engine.calculate(
        ThumbnailStatisticsRequest(
            values,
            contrast_mode=contrast_mode,
            channel_axis=channel_axis,
            compute_mode=ComputeMode.PREFER_GPU,
        )
    )

    np.testing.assert_array_equal(
        np.asarray(gpu.limits).view(np.uint64),
        np.asarray(cpu.limits).view(np.uint64),
    )
    assert gpu.auxiliary_host_to_device_bytes == 17
    assert gpu.device_to_host_values == 29
    assert gpu.device_to_host_bytes == 23


@pytest.mark.parametrize(
    ("values", "contrast_mode"),
    [
        (np.arange(32, dtype=np.uint16), "Min-max"),
        (np.arange(32, dtype=np.float32), "Raw"),
        (np.arange(32, dtype=np.float64), "Percentile"),
    ],
)
def test_prefer_gpu_keeps_unsupported_thumbnail_contracts_on_cpu(
    values,
    contrast_mode,
):
    result = ThumbnailStatisticsEngine().calculate(
        ThumbnailStatisticsRequest(
            values,
            contrast_mode=contrast_mode,
            compute_mode=ComputeMode.PREFER_GPU,
        )
    )

    assert result.actual_backend is ThumbnailStatisticsBackend.CPU_NUMPY
    assert result.fallback_reason_code == "gpu_ineligible"


def test_prefer_gpu_ineligible_dtype_and_memory_cap_decisions_are_visible_fallbacks():
    engine = ThumbnailStatisticsEngine()
    float64_result = engine.calculate(
        ThumbnailStatisticsRequest(
            np.linspace(0, 1, 20, dtype=np.float64),
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

    assert float64_result.fallback_reason_code == "gpu_ineligible"
    assert capped_result.fallback_reason_code == "gpu_memory_cap_insufficient"
    assert float64_result.fallback_message
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


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_gpu_failure_falls_back_only_after_successful_runtime_cleanup(dtype):
    registry = _ExecutionFailureRegistry(reason_code="cuda_out_of_memory")
    engine = ThumbnailStatisticsEngine(registry_factory=lambda: registry)

    result = engine.calculate(
        ThumbnailStatisticsRequest(
            np.arange(100, dtype=dtype),
            compute_mode=ComputeMode.PREFER_GPU,
        )
    )

    assert result.actual_backend is ThumbnailStatisticsBackend.CPU_NUMPY
    assert result.requested_compute_mode is ComputeMode.PREFER_GPU
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
        update.current / update.total for update in updates if update.total > 0
    ]
    assert fractions == sorted(fractions)
    assert fractions[0] == pytest.approx(0.75)
    assert fractions[-1] == pytest.approx(1.0)
    assert any(update.message.startswith("CPU fallback") for update in updates)
    assert result.actual_backend is ThumbnailStatisticsBackend.CPU_NUMPY
    assert result.fallback_reason_code == "cuda_out_of_memory"


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_gpu_cancellation_closes_runtime_and_never_falls_back(dtype):
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
                np.arange(100, dtype=dtype),
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


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_integer_chunked_native_minmax_is_exact_and_never_probes_gpu(dtype):
    calls = 0

    def forbidden_registry():
        nonlocal calls
        calls += 1
        raise AssertionError("Min-max must use the cheaper CPU reduction.")

    values = np.asarray([0, 3, 0, 8], dtype=dtype)
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
    assert any("cancel applies after this pass" in update.message for update in updates)
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

    def fake_float32_limits(_runtime, arr, **kwargs):
        progress = kwargs.get("progress")
        if progress is not None:
            progress.check_cancelled()
        if getattr(_runtime, "calculation_error", None) is not None:
            raise _runtime.calculation_error
        channel_axis = kwargs.get("channel_axis")
        request = ThumbnailStatisticsRequest(
            arr,
            contrast_mode=kwargs.get("contrast_mode", "Percentile"),
            channel_axis=channel_axis,
            compute_mode=ComputeMode.CPU,
        )
        limits = ThumbnailStatisticsEngine().calculate(request).limits
        return SimpleNamespace(
            limits=np.asarray(limits, dtype=np.float64),
            auxiliary_host_to_device_bytes=17,
            device_to_host_bytes=23,
            device_to_host_values=29,
        )

    monkeypatch.setattr(
        "napari_vipp.core.thumbnail_statistics._calculate_cupy_counts",
        fake_counts,
    )
    monkeypatch.setattr(
        "napari_vipp.core.thumbnail_statistics._calculate_cupy_float32_limits",
        fake_float32_limits,
    )


def test_default_threshold_contracts_remain_conservative():
    assert DEFAULT_COLD_UINT8_GPU_THRESHOLD_BYTES == 384 * 1024**2
    assert DEFAULT_COLD_UINT16_GPU_THRESHOLD_BYTES == 512 * 1024**2
    assert DEFAULT_COLD_GPU_THRESHOLD_BYTES == DEFAULT_COLD_UINT16_GPU_THRESHOLD_BYTES
    assert DEFAULT_WARM_GPU_THRESHOLD_BYTES == 32 * 1024**2
