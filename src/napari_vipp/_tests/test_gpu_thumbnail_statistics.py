from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

from napari_vipp.core.compute_registry import ComputeRegistry
from napari_vipp.core.gpu import cupy_thumbnail_statistics as thumbnail_provider
from napari_vipp.core.gpu.cupy_runtime import CUDACleanupError
from napari_vipp.core.gpu.cupy_thumbnail_statistics import (
    ThumbnailStatisticsProviderCleanupError,
    exact_float32_thumbnail_limits,
    exact_float32_thumbnail_limits_from_device,
    exact_uint_histogram_counts,
)
from napari_vipp.core.progress import OperationCancelled, ProgressContext
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


class _FakeDeviceValue:
    def __init__(self, values, *, owner, allocation=None) -> None:
        self.values = np.asarray(values)
        self.owner = owner
        self.allocation = allocation if allocation is not None else object()

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def ndim(self):
        return self.values.ndim

    @property
    def shape(self):
        return self.values.shape

    @property
    def size(self):
        return self.values.size

    def __getitem__(self, selection):
        return _FakeDeviceValue(
            self.values[selection],
            owner=self.owner,
            allocation=self.allocation,
        )


class _FakeResidentRuntime:
    def __init__(self) -> None:
        self.active = True
        self.fail_release = False
        self.uploads: list[int] = []
        self.releases: list[object] = []

    def allocation_identity(self, value):
        if not self.active:
            raise RuntimeError("an active execution scope is required")
        if not isinstance(value, _FakeDeviceValue) or value.owner is not self:
            raise TypeError("the device value is not owned by this runtime")
        return value.allocation

    def to_device(self, value, *, device_id):
        del device_id
        values = np.asarray(value)
        self.uploads.append(int(values.nbytes))
        return _FakeDeviceValue(values.copy(), owner=self)

    def release(self, value):
        self.releases.append(self.allocation_identity(value))
        if self.fail_release:
            raise RuntimeError("injected fake release failure")


class _ObservedResidentRuntime:
    def __init__(
        self,
        runtime,
        *,
        borrowed_identity,
        fail_first_scratch_release: bool = False,
    ) -> None:
        self.runtime = runtime
        self.borrowed_identity = borrowed_identity
        self.fail_first_scratch_release = fail_first_scratch_release
        self.failed_scratch_release = False
        self.to_device_bytes: list[int] = []
        self.to_host_bytes: list[int] = []
        self.to_host_values: list[int] = []
        self.release_identities: list[object] = []

    def __getattr__(self, name):
        return getattr(self.runtime, name)

    def to_device(self, value, *, device_id):
        host_value = np.asarray(value)
        self.to_device_bytes.append(int(host_value.nbytes))
        return self.runtime.to_device(host_value, device_id=device_id)

    def to_host(self, value):
        self.to_host_bytes.append(int(value.nbytes))
        self.to_host_values.append(int(value.size))
        return self.runtime.to_host(value)

    def release(self, value):
        identity = self.runtime.allocation_identity(value)
        self.release_identities.append(identity)
        if (
            self.fail_first_scratch_release
            and not self.failed_scratch_release
            and identity != self.borrowed_identity
        ):
            self.failed_scratch_release = True
            raise RuntimeError("injected resident scratch release failure")
        return self.runtime.release(value)


def _fake_float32_channel_limits(
    _runtime,
    _cupy,
    device_channel,
    **_kwargs,
):
    finite = device_channel.values[np.isfinite(device_channel.values)]
    limits = (
        (0.0, 0.0) if not finite.size else (float(finite.min()), float(finite.max()))
    )
    return thumbnail_provider._Float32ChannelStatistics(
        limits,
        40,
        40,
        5,
    )


def test_fake_resident_float32_path_shares_host_implementation_without_upload_or_alias(
    monkeypatch,
):
    runtime = _FakeResidentRuntime()
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    monkeypatch.setattr(thumbnail_provider, "_cupy_module", lambda: object())
    monkeypatch.setattr(
        thumbnail_provider,
        "_exact_float32_channel_limits",
        _fake_float32_channel_limits,
    )

    host_result = exact_float32_thumbnail_limits(
        runtime,
        values,
        device_id="cuda:0",
        channel_axis=0,
        contrast_mode="Min-max",
    )
    assert runtime.uploads == [values.nbytes]
    assert len(runtime.releases) == 1

    runtime.uploads.clear()
    runtime.releases.clear()
    resident = _FakeDeviceValue(values, owner=runtime)
    resident_reference = weakref.ref(resident)
    resident_result = exact_float32_thumbnail_limits_from_device(
        runtime,
        resident,
        device_id="cuda:0",
        channel_axis=0,
        contrast_mode="Min-max",
    )

    np.testing.assert_array_equal(resident_result.limits, host_result.limits)
    assert resident_result.auxiliary_host_to_device_bytes == 80
    assert resident_result.device_to_host_bytes == 80
    assert resident_result.device_to_host_values == 10
    assert runtime.uploads == []
    assert runtime.releases == []
    del resident
    gc.collect()
    assert resident_reference() is None


@pytest.mark.parametrize(
    ("condition", "error", "message"),
    [
        ("inactive", RuntimeError, "active execution scope"),
        ("foreign", TypeError, "not owned"),
        ("wrong_dtype", TypeError, "native float32"),
    ],
)
def test_fake_resident_float32_path_validates_active_ownership_and_dtype(
    monkeypatch,
    condition,
    error,
    message,
):
    runtime = _FakeResidentRuntime()
    owner = _FakeResidentRuntime() if condition == "foreign" else runtime
    dtype = np.uint16 if condition == "wrong_dtype" else np.float32
    resident = _FakeDeviceValue(np.arange(8, dtype=dtype), owner=owner)
    if condition == "inactive":
        runtime.active = False
    monkeypatch.setattr(thumbnail_provider, "_cupy_module", lambda: object())

    with pytest.raises(error, match=message):
        exact_float32_thumbnail_limits_from_device(
            runtime,
            resident,
            device_id="cuda:0",
        )

    assert runtime.uploads == []
    assert runtime.releases == []


def test_fake_resident_float32_failure_drops_borrowed_alias_without_releasing_it(
    monkeypatch,
):
    runtime = _FakeResidentRuntime()
    resident = _FakeDeviceValue(np.arange(8, dtype=np.float32), owner=runtime)
    resident_reference = weakref.ref(resident)

    def fail_channel(*_args, **_kwargs):
        raise ThumbnailStatisticsProviderCleanupError(
            "injected scratch cleanup failure"
        )

    monkeypatch.setattr(thumbnail_provider, "_cupy_module", lambda: object())
    monkeypatch.setattr(
        thumbnail_provider,
        "_exact_float32_channel_limits",
        fail_channel,
    )

    with pytest.raises(
        ThumbnailStatisticsProviderCleanupError,
        match="injected scratch cleanup failure",
    ) as raised:
        exact_float32_thumbnail_limits_from_device(
            runtime,
            resident,
            device_id="cuda:0",
        )

    assert raised.value.cleanup_succeeded is False
    assert runtime.uploads == []
    assert runtime.releases == []
    del raised
    del resident
    gc.collect()
    assert resident_reference() is None


def test_fake_host_float32_release_failure_is_typed_cleanup_failure(monkeypatch):
    runtime = _FakeResidentRuntime()
    runtime.fail_release = True
    values = np.arange(8, dtype=np.float32)
    monkeypatch.setattr(thumbnail_provider, "_cupy_module", lambda: object())
    monkeypatch.setattr(
        thumbnail_provider,
        "_exact_float32_channel_limits",
        _fake_float32_channel_limits,
    )

    with pytest.raises(
        ThumbnailStatisticsProviderCleanupError,
        match="injected fake release failure",
    ) as raised:
        exact_float32_thumbnail_limits(
            runtime,
            values,
            device_id="cuda:0",
            contrast_mode="Min-max",
        )

    assert raised.value.cleanup_succeeded is False
    assert runtime.uploads == [values.nbytes]
    assert len(runtime.releases) == 1


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


@pytest.mark.parametrize("contrast_mode", ["Percentile", "Min-max"])
@pytest.mark.parametrize("channel_axis", [None, 0, 1, 2, -1])
def test_real_gpu_float32_limits_match_cpu_bitwise_and_clean_up(
    contrast_mode,
    channel_axis,
):
    registry, probe = _working_cuda_registry()
    runtime = registry.runtime("cuda-cupy")
    rng = np.random.default_rng(542)
    values = rng.normal(size=(3, 7, 11)).astype(np.float32)
    values[0, 0, :8] = np.asarray(
        [
            np.nan,
            -np.inf,
            np.inf,
            -0.0,
            0.0,
            np.nextafter(np.float32(0.0), np.float32(1.0)),
            np.finfo(np.float32).max,
            -np.finfo(np.float32).max,
        ],
        dtype=np.float32,
    )
    original_bits = values.view(np.uint32).copy()
    expected = ThumbnailStatisticsEngine().calculate(
        ThumbnailStatisticsRequest(
            values,
            contrast_mode=contrast_mode,
            channel_axis=channel_axis,
            compute_mode="cpu",
        )
    )

    try:
        with runtime.execution_scope(
            device_id=probe.selected_device_id,
            memory_limit_bytes=512 * 1024**2,
            safety_reserve_bytes=256 * 1024**2,
        ):
            actual = exact_float32_thumbnail_limits(
                runtime,
                values,
                device_id=probe.selected_device_id,
                channel_axis=channel_axis,
                contrast_mode=contrast_mode,
            )

        np.testing.assert_array_equal(values.view(np.uint32), original_bits)
        np.testing.assert_array_equal(
            np.asarray(actual.limits).view(np.uint64),
            np.asarray(expected.limits).view(np.uint64),
        )
        channel_count = 1 if channel_axis is None else values.shape[channel_axis]
        if contrast_mode == "Min-max":
            assert actual.auxiliary_host_to_device_bytes == channel_count * 40
            assert actual.device_to_host_bytes == channel_count * 40
            assert actual.device_to_host_values == channel_count * 5
        else:
            assert channel_count * 56 <= actual.auxiliary_host_to_device_bytes
            assert actual.auxiliary_host_to_device_bytes <= channel_count * 104
            assert channel_count * 8_232 <= actual.device_to_host_bytes
            assert actual.device_to_host_bytes <= channel_count * 32_808
            assert channel_count * 1_029 <= actual.device_to_host_values
            assert actual.device_to_host_values <= channel_count * 4_101
        terminal = runtime.memory_snapshot(device_id=probe.selected_device_id)
        assert terminal.runtime_live_bytes == 0
        assert terminal.runtime_reserved_bytes == 0
    finally:
        registry.close()


def test_real_gpu_float32_percentile_matches_gamma_edges_and_degenerate_inputs():
    registry, probe = _working_cuda_registry()
    runtime = registry.runtime("cuda-cupy")
    cases = []
    for size in (1, 2, 101, 201, 1_001):
        values = np.linspace(0.0, 17.0, size, dtype=np.float32)
        if size > 1:
            values[0] = -0.0
            values[-1] = np.finfo(np.float32).max
        cases.append(values)
    cases.extend(
        [
            np.full(257, -0.0, dtype=np.float32),
            np.asarray([np.nan, -np.inf, np.inf], dtype=np.float32),
            np.asarray([-np.inf, 1.0, 2.0], dtype=np.float32),
            np.asarray([-3.0, -np.inf, 0.0, 2.0], dtype=np.float32),
        ]
    )

    try:
        with runtime.execution_scope(
            device_id=probe.selected_device_id,
            memory_limit_bytes=512 * 1024**2,
            safety_reserve_bytes=256 * 1024**2,
        ):
            actual = [
                exact_float32_thumbnail_limits(
                    runtime,
                    values,
                    device_id=probe.selected_device_id,
                ).limits
                for values in cases
            ]

        for values, limits in zip(cases, actual, strict=True):
            expected = (
                ThumbnailStatisticsEngine()
                .calculate(ThumbnailStatisticsRequest(values, compute_mode="cpu"))
                .limits
            )
            np.testing.assert_array_equal(
                np.asarray(limits).view(np.uint64),
                np.asarray(expected).view(np.uint64),
            )
        terminal = runtime.memory_snapshot(device_id=probe.selected_device_id)
        assert terminal.runtime_live_bytes == 0
        assert terminal.runtime_reserved_bytes == 0
    finally:
        registry.close()


@pytest.mark.parametrize("contrast_mode", ["Percentile", "Min-max"])
@pytest.mark.parametrize("layout", ["fortran", "strided"])
def test_real_gpu_float32_stages_noncontiguous_host_layout_exactly(
    contrast_mode,
    layout,
):
    registry, probe = _working_cuda_registry()
    runtime = registry.runtime("cuda-cupy")
    rng = np.random.default_rng(543)
    base = rng.normal(size=(17, 31, 37)).astype(np.float32)
    values = np.asfortranarray(base) if layout == "fortran" else base[:, :, ::2]
    assert not values.flags.c_contiguous
    original_bits = values.view(np.uint32).copy()
    expected = ThumbnailStatisticsEngine().calculate(
        ThumbnailStatisticsRequest(
            values,
            contrast_mode=contrast_mode,
            compute_mode="cpu",
        )
    )

    try:
        with runtime.execution_scope(
            device_id=probe.selected_device_id,
            memory_limit_bytes=512 * 1024**2,
            safety_reserve_bytes=256 * 1024**2,
        ):
            actual = exact_float32_thumbnail_limits(
                runtime,
                values,
                device_id=probe.selected_device_id,
                contrast_mode=contrast_mode,
            )

        np.testing.assert_array_equal(values.view(np.uint32), original_bits)
        np.testing.assert_array_equal(
            np.asarray(actual.limits).view(np.uint64),
            np.asarray(expected.limits).view(np.uint64),
        )
        terminal = runtime.memory_snapshot(device_id=probe.selected_device_id)
        assert terminal.runtime_live_bytes == 0
        assert terminal.runtime_reserved_bytes == 0
    finally:
        registry.close()


def test_real_gpu_float32_cancellation_releases_every_private_allocation():
    registry, probe = _working_cuda_registry()
    runtime = registry.runtime("cuda-cupy")
    values = np.linspace(-1.0, 1.0, 257 * 263, dtype=np.float32)
    cancelled = False

    def report(update):
        nonlocal cancelled
        if "radix pass 1/4" in update.message:
            cancelled = True

    progress = ProgressContext(cancelled=lambda: cancelled, reporter=report)
    try:
        with pytest.raises(OperationCancelled):
            with runtime.execution_scope(
                device_id=probe.selected_device_id,
                memory_limit_bytes=512 * 1024**2,
                safety_reserve_bytes=256 * 1024**2,
            ):
                exact_float32_thumbnail_limits(
                    runtime,
                    values,
                    device_id=probe.selected_device_id,
                    progress=progress,
                )

        terminal = runtime.memory_snapshot(device_id=probe.selected_device_id)
        assert terminal.runtime_live_bytes == 0
        assert terminal.runtime_reserved_bytes == 0
    finally:
        registry.close()


def test_real_gpu_float32_engine_reports_host_upload_and_bounded_transfers():
    registry, probe = _working_cuda_registry()
    registry.close()
    values = np.linspace(-2.0, 5.0, 4_097, dtype=np.float32)

    result = ThumbnailStatisticsEngine().calculate(
        ThumbnailStatisticsRequest(
            values,
            compute_mode="prefer_gpu",
            device_id=probe.selected_device_id,
            accelerator_memory_cap_bytes=512 * 1024**2,
            accelerator_safety_reserve_bytes=256 * 1024**2,
        )
    )

    expected = ThumbnailStatisticsEngine().calculate(
        ThumbnailStatisticsRequest(values, compute_mode="cpu")
    )
    assert result.actual_backend is ThumbnailStatisticsBackend.GPU_CUPY
    assert not result.used_fallback
    assert result.input_path == "host_upload"
    assert result.logical_input_host_to_device_bytes == values.nbytes
    assert 56 <= result.auxiliary_host_to_device_bytes <= 104
    assert 8_232 <= result.device_to_host_bytes <= 32_808
    assert 1_029 <= result.device_to_host_values <= 4_101
    np.testing.assert_array_equal(
        np.asarray(result.limits).view(np.uint64),
        np.asarray(expected.limits).view(np.uint64),
    )


@pytest.mark.parametrize("contrast_mode", ["Percentile", "Min-max"])
@pytest.mark.parametrize("layout", ["contiguous", "strided"])
def test_real_gpu_resident_float32_is_exact_nonowning_and_returns_to_baseline(
    contrast_mode,
    layout,
):
    registry, probe = _working_cuda_registry()
    runtime = registry.runtime("cuda-cupy")
    rng = np.random.default_rng(544)
    base_values = rng.normal(size=(5, 4, 33)).astype(np.float32)
    base_values[0, 0, :8] = np.asarray(
        [
            np.nan,
            -np.inf,
            np.inf,
            -0.0,
            0.0,
            np.nextafter(np.float32(0.0), np.float32(1.0)),
            np.finfo(np.float32).max,
            -np.finfo(np.float32).max,
        ],
        dtype=np.float32,
    )
    values = base_values if layout == "contiguous" else base_values[:, :, ::2]
    expected = ThumbnailStatisticsEngine().calculate(
        ThumbnailStatisticsRequest(
            values,
            contrast_mode=contrast_mode,
            channel_axis=1,
            compute_mode="cpu",
        )
    )

    try:
        with runtime.execution_scope(
            device_id=probe.selected_device_id,
            memory_limit_bytes=512 * 1024**2,
            safety_reserve_bytes=256 * 1024**2,
        ):
            device_base = runtime.to_device(
                base_values,
                device_id=probe.selected_device_id,
            )
            device_values = (
                device_base if layout == "contiguous" else device_base[:, :, ::2]
            )
            borrowed_identity = runtime.allocation_identity(device_values)
            baseline = runtime.memory_snapshot(device_id=probe.selected_device_id)
            observed_runtime = _ObservedResidentRuntime(
                runtime,
                borrowed_identity=borrowed_identity,
            )

            actual = exact_float32_thumbnail_limits_from_device(
                observed_runtime,
                device_values,
                device_id=probe.selected_device_id,
                channel_axis=1,
                contrast_mode=contrast_mode,
            )

            after_statistics = runtime.memory_snapshot(
                device_id=probe.selected_device_id
            )
            assert after_statistics.runtime_live_bytes == baseline.runtime_live_bytes
            assert borrowed_identity not in observed_runtime.release_identities
            assert sum(observed_runtime.to_device_bytes) == (
                actual.auxiliary_host_to_device_bytes
            )
            assert sum(observed_runtime.to_host_bytes) == actual.device_to_host_bytes
            assert sum(observed_runtime.to_host_values) == actual.device_to_host_values
            assert observed_runtime.to_device_bytes
            assert max(observed_runtime.to_device_bytes) < int(device_values.nbytes)
            np.testing.assert_array_equal(
                np.asarray(actual.limits).view(np.uint64),
                np.asarray(expected.limits).view(np.uint64),
            )

            observed_runtime = None
            device_values = None
            runtime.release(device_base)
            device_base = None
            gc.collect()
            after_borrower_release = runtime.memory_snapshot(
                device_id=probe.selected_device_id
            )
            assert after_borrower_release.runtime_live_bytes == 0

        terminal = runtime.memory_snapshot(device_id=probe.selected_device_id)
        assert terminal.runtime_live_bytes == 0
        assert terminal.runtime_reserved_bytes == 0
    finally:
        registry.close()


def test_real_gpu_resident_float32_cleanup_failure_is_fatal_and_nonowning():
    registry, probe = _working_cuda_registry()
    runtime = registry.runtime("cuda-cupy")
    values = np.linspace(-2.0, 3.0, 4_097, dtype=np.float32)
    provider_cleanup_seen = False

    try:
        with pytest.raises(CUDACleanupError) as scope_error:
            with runtime.execution_scope(
                device_id=probe.selected_device_id,
                memory_limit_bytes=512 * 1024**2,
                safety_reserve_bytes=256 * 1024**2,
            ):
                device_values = runtime.to_device(
                    values,
                    device_id=probe.selected_device_id,
                )
                borrowed_identity = runtime.allocation_identity(device_values)
                observed_runtime = _ObservedResidentRuntime(
                    runtime,
                    borrowed_identity=borrowed_identity,
                    fail_first_scratch_release=True,
                )
                try:
                    exact_float32_thumbnail_limits_from_device(
                        observed_runtime,
                        device_values,
                        device_id=probe.selected_device_id,
                        contrast_mode="Min-max",
                    )
                except ThumbnailStatisticsProviderCleanupError as exc:
                    provider_cleanup_seen = True
                    assert exc.cleanup_succeeded is False
                    assert observed_runtime.failed_scratch_release
                    assert borrowed_identity not in (
                        observed_runtime.release_identities
                    )
                    # Drop the caller's source alias without asking the provider
                    # to release it.  The injected scratch-release failure must
                    # remain fatal and poison the enclosing private scope.
                    device_values = None
                    observed_runtime = None
                    gc.collect()
                    raise

        assert provider_cleanup_seen
        assert isinstance(
            scope_error.value.__cause__,
            ThumbnailStatisticsProviderCleanupError,
        )
        terminal = runtime.memory_snapshot(device_id=probe.selected_device_id)
        assert terminal.runtime_live_bytes > 0
        assert terminal.runtime_reserved_bytes >= terminal.runtime_live_bytes
        with pytest.raises(RuntimeError, match="cleanup was incomplete"):
            with runtime.execution_scope(device_id=probe.selected_device_id):
                pass
    finally:
        registry.close()


def test_real_gpu_resident_float32_cancellation_cleans_only_scratch():
    registry, probe = _working_cuda_registry()
    runtime = registry.runtime("cuda-cupy")
    values = np.linspace(-1.0, 1.0, 257 * 263, dtype=np.float32)
    cancelled = False

    def report(update):
        nonlocal cancelled
        if "radix pass 1/4" in update.message:
            cancelled = True

    progress = ProgressContext(cancelled=lambda: cancelled, reporter=report)
    try:
        with runtime.execution_scope(
            device_id=probe.selected_device_id,
            memory_limit_bytes=512 * 1024**2,
            safety_reserve_bytes=256 * 1024**2,
        ):
            device_values = runtime.to_device(
                values,
                device_id=probe.selected_device_id,
            )
            borrowed_identity = runtime.allocation_identity(device_values)
            baseline = runtime.memory_snapshot(device_id=probe.selected_device_id)
            observed_runtime = _ObservedResidentRuntime(
                runtime,
                borrowed_identity=borrowed_identity,
            )

            with pytest.raises(OperationCancelled):
                exact_float32_thumbnail_limits_from_device(
                    observed_runtime,
                    device_values,
                    device_id=probe.selected_device_id,
                    progress=progress,
                )

            assert borrowed_identity not in observed_runtime.release_identities
            after_cancellation = runtime.memory_snapshot(
                device_id=probe.selected_device_id
            )
            assert after_cancellation.runtime_live_bytes == baseline.runtime_live_bytes

            observed_runtime = None
            runtime.release(device_values)
            device_values = None
            gc.collect()
            after_borrower_release = runtime.memory_snapshot(
                device_id=probe.selected_device_id
            )
            assert after_borrower_release.runtime_live_bytes == 0

        terminal = runtime.memory_snapshot(device_id=probe.selected_device_id)
        assert terminal.runtime_live_bytes == 0
        assert terminal.runtime_reserved_bytes == 0
    finally:
        registry.close()
