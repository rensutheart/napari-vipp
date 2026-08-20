"""Bounded CuPy histograms for presentation-only thumbnail statistics."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import cache
from types import ModuleType

import numpy as np

from napari_vipp.core.progress import ProgressContext

_THREADS_PER_BLOCK = 256
_MAXIMUM_BLOCKS = 65_535
_FLOAT32_METADATA_INITIAL = np.asarray(
    [0, 0, 0, np.iinfo(np.uint32).max, 0],
    dtype=np.uint64,
)


class ThumbnailStatisticsProviderCleanupError(RuntimeError):
    """Raised when this provider cannot relinquish a runtime-owned allocation."""

    cleanup_succeeded = False


@dataclass(frozen=True, slots=True)
class Float32ThumbnailStatistics:
    """Host limits and exact auxiliary transfer observations for one request."""

    limits: np.ndarray
    auxiliary_host_to_device_bytes: int
    device_to_host_bytes: int
    device_to_host_values: int


@dataclass(frozen=True, slots=True)
class _Float32ChannelStatistics:
    limits: tuple[float, float]
    auxiliary_host_to_device_bytes: int
    device_to_host_bytes: int
    device_to_host_values: int


@dataclass(frozen=True, slots=True)
class _RadixSelection:
    bits: np.ndarray
    auxiliary_host_to_device_bytes: int
    device_to_host_bytes: int
    device_to_host_values: int


@cache
def _cupy_module() -> ModuleType:
    """Load CuPy only after the core selector explicitly chooses CUDA."""

    return importlib.import_module("cupy")


@cache
def _histogram_kernel(cupy: ModuleType, dtype_name: str):
    if dtype_name == "uint8":
        native_type = "unsigned char"
        kernel_name = "vipp_thumbnail_histogram_uint8"
    elif dtype_name == "uint16":
        native_type = "unsigned short"
        kernel_name = "vipp_thumbnail_histogram_uint16"
    else:  # pragma: no cover - guarded by the public adapter
        raise TypeError(f"Unsupported thumbnail histogram dtype {dtype_name!r}.")
    return cupy.RawKernel(
        rf"""
        extern "C" __global__
        void {kernel_name}(
            const {native_type}* values,
            const unsigned long long size,
            const unsigned long long channel_count,
            const unsigned long long channel_stride,
            const unsigned long long level_count,
            unsigned long long* counts)
        {{
            unsigned long long index =
                (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
            const unsigned long long grid_stride =
                (unsigned long long)blockDim.x * gridDim.x;
            for (; index < size; index += grid_stride) {{
                const unsigned long long channel =
                    channel_count == 1ULL
                    ? 0ULL
                    : (index / channel_stride) % channel_count;
                const unsigned long long level =
                    (unsigned long long)values[index];
                atomicAdd(
                    counts + channel * level_count + level,
                    1ULL);
            }}
        }}
        """,
        kernel_name,
        options=("--std=c++11",),
    )


@cache
def _float32_metadata_kernel(cupy: ModuleType):
    return cupy.RawKernel(
        r"""
        extern "C" __global__
        void vipp_thumbnail_float32_metadata(
            const unsigned int* values,
            const unsigned long long size,
            unsigned long long* metadata)
        {
            unsigned long long index =
                (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
            const unsigned long long grid_stride =
                (unsigned long long)blockDim.x * gridDim.x;
            for (; index < size; index += grid_stride) {
                const unsigned int bits = values[index];
                const unsigned int magnitude = bits & 0x7fffffffU;
                const bool finite = (magnitude & 0x7f800000U) != 0x7f800000U;
                if (finite) {
                    atomicAdd(metadata + 0, 1ULL);
                    const bool negative_finite =
                        (bits & 0x80000000U) != 0U && magnitude != 0U;
                    if (negative_finite) {
                        atomicMax(metadata + 2, 1ULL);
                    }
                    const unsigned long long clipped_bits =
                        negative_finite ? 0ULL : (unsigned long long)magnitude;
                    atomicMin(metadata + 3, clipped_bits);
                    atomicMax(metadata + 4, clipped_bits);
                } else if (bits == 0xff800000U) {
                    atomicAdd(metadata + 1, 1ULL);
                }
            }
        }
        """,
        "vipp_thumbnail_float32_metadata",
        options=("--std=c++11",),
    )


@cache
def _float32_radix_histogram_kernel(cupy: ModuleType):
    return cupy.RawKernel(
        r"""
        extern "C" __global__
        void vipp_thumbnail_float32_radix_histogram(
            const unsigned int* values,
            const unsigned long long size,
            const unsigned long long include_negative_infinity,
            const unsigned int* prefixes,
            const unsigned long long prefix_count,
            const unsigned int prefix_bits,
            const unsigned int byte_shift,
            unsigned long long* histograms)
        {
            unsigned long long index =
                (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
            const unsigned long long grid_stride =
                (unsigned long long)blockDim.x * gridDim.x;
            for (; index < size; index += grid_stride) {
                const unsigned int bits = values[index];
                const unsigned int magnitude = bits & 0x7fffffffU;
                const bool finite = (magnitude & 0x7f800000U) != 0x7f800000U;
                unsigned int key = 0U;
                bool valid = false;
                if (finite) {
                    const bool negative_finite =
                        (bits & 0x80000000U) != 0U && magnitude != 0U;
                    key = negative_finite ? 0U : magnitude;
                    valid = true;
                } else if (
                    bits == 0xff800000U && include_negative_infinity != 0ULL
                ) {
                    key = 0U;
                    valid = true;
                }
                if (!valid) {
                    continue;
                }
                const unsigned int prefix =
                    prefix_bits == 0U ? 0U : key >> (32U - prefix_bits);
                const unsigned int bucket = (key >> byte_shift) & 0xffU;
                for (unsigned long long target = 0ULL;
                     target < prefix_count;
                     ++target) {
                    if (prefixes[target] == prefix) {
                        atomicAdd(
                            histograms + target * 256ULL + bucket,
                            1ULL);
                    }
                }
            }
        }
        """,
        "vipp_thumbnail_float32_radix_histogram",
        options=("--std=c++11",),
    )


def exact_uint_histogram_counts(
    runtime,
    data,
    *,
    device_id: str,
    channel_axis: int | None = None,
    progress: ProgressContext | None = None,
) -> np.ndarray:
    """Count every uint8/uint16 level within an active private runtime scope.

    The input is uploaded once.  A channel-aware kernel writes only bounded
    uint64 counts, and only those counts return to the host.  Device arrays are
    released and all Python aliases dropped before the caller exits its runtime
    scope, allowing the established CuPy lifecycle checks to verify zero live
    private allocations.
    """

    arr = np.asarray(data)
    if arr.dtype not in {np.dtype(np.uint8), np.dtype(np.uint16)}:
        raise TypeError("CuPy thumbnail histograms require native uint8 or uint16.")
    if arr.size == 0:
        level_count = 256 if arr.dtype == np.dtype(np.uint8) else 65_536
        if channel_axis is None:
            return np.zeros(level_count, dtype=np.uint64)
        axis = _normalized_channel_axis(channel_axis, arr.ndim)
        return np.zeros((arr.shape[axis], level_count), dtype=np.uint64)

    axis = (
        None
        if channel_axis is None
        else _normalized_channel_axis(channel_axis, arr.ndim)
    )
    channel_count = 1 if axis is None else int(arr.shape[axis])
    channel_stride = (
        1
        if axis is None or axis == arr.ndim - 1
        else int(np.prod(arr.shape[axis + 1 :], dtype=np.int64))
    )
    level_count = 256 if arr.dtype == np.dtype(np.uint8) else 65_536
    total = int(arr.size)
    _report(
        progress,
        0,
        total,
        "Uploading thumbnail statistics to GPU · cancel applies after this pass",
    )

    cupy = _cupy_module()
    device_values = None
    device_counts = None
    try:
        device_values = runtime.to_device(arr, device_id=device_id)
        _report(
            progress,
            max(1, total // 4),
            total,
            "Counting exact thumbnail intensity levels on GPU · "
            "cancel applies after this pass",
        )
        device_counts = cupy.zeros(
            (channel_count, level_count),
            dtype=cupy.uint64,
        )
        block_count = min(
            (total + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK,
            _MAXIMUM_BLOCKS,
        )
        kernel = _histogram_kernel(cupy, np.dtype(arr.dtype).name)
        kernel(
            (block_count,),
            (_THREADS_PER_BLOCK,),
            (
                device_values,
                np.uint64(total),
                np.uint64(channel_count),
                np.uint64(channel_stride),
                np.uint64(level_count),
                device_counts,
            ),
        )
        runtime.synchronize(device_id=device_id)
        _report(
            progress,
            max(1, 3 * total // 4),
            total,
            "Returning bounded thumbnail histogram from GPU · "
            "cancel applies after this pass",
        )
        host_counts = np.asarray(runtime.to_host(device_counts), dtype=np.uint64)
        _report(progress, total, total, "Thumbnail GPU histogram ready")
    finally:
        release_errors = []
        for value in (device_counts, device_values):
            if value is None:
                continue
            try:
                runtime.release(value)
            except Exception as exc:
                release_errors.append(exc)
        value = None
        device_counts = None
        device_values = None
        if release_errors:
            detail = "; ".join(f"{type(exc).__name__}: {exc}" for exc in release_errors)
            raise ThumbnailStatisticsProviderCleanupError(
                "Thumbnail GPU allocations could not be relinquished cleanly: " + detail
            ) from release_errors[0]
    return host_counts[0] if axis is None else host_counts


def exact_float32_thumbnail_limits(
    runtime,
    data,
    *,
    device_id: str,
    channel_axis: int | None = None,
    contrast_mode: str = "Percentile",
    progress: ProgressContext | None = None,
) -> Float32ThumbnailStatistics:
    """Upload host float32 values and return exact bounded GPU limits.

    Non-finite filtering, policy clipping, and ordering operate on the original
    IEEE-754 bit patterns.  This avoids CUDA floating-point contraction or
    subnormal handling changing the selected values.  Percentiles return bounded
    radix histograms and metadata (at most about 32 KiB per channel) while only
    the selected scalar keys escape the provider.  NumPy performs the final
    linear interpolation so its arithmetic order remains identical to the CPU
    path.
    """

    arr = np.asarray(data)
    mode, axis = _float32_thumbnail_contract(
        arr,
        channel_axis=channel_axis,
        contrast_mode=contrast_mode,
    )
    total_work = _float32_total_work(arr, mode)
    _check_cancelled(progress)
    _report(
        progress,
        0,
        total_work,
        "Uploading float32 thumbnail statistics to GPU · "
        "cancel applies after this pass",
    )

    cupy = _cupy_module()
    uploaded_values = None
    device_values_holder: list[object] = []
    try:
        uploaded_values = runtime.to_device(arr, device_id=device_id)
        device_values_holder.append(uploaded_values)
        uploaded_values = None
        _check_cancelled(progress)
        return _exact_float32_thumbnail_limits_from_device_owned(
            runtime,
            cupy,
            device_values_holder,
            device_id=device_id,
            axis=axis,
            mode=mode,
            progress=progress,
            total_work=total_work,
        )
    finally:
        release_errors = []
        if device_values_holder:
            try:
                runtime.release(device_values_holder[0])
            except Exception as exc:
                release_errors.append(exc)
        device_values_holder.clear()
        uploaded_values = None
        if release_errors:
            detail = "; ".join(f"{type(exc).__name__}: {exc}" for exc in release_errors)
            raise ThumbnailStatisticsProviderCleanupError(
                "Float32 thumbnail GPU allocations could not be relinquished "
                "cleanly: " + detail
            ) from release_errors[0]


def exact_float32_thumbnail_limits_from_device(
    runtime,
    device_values,
    *,
    device_id: str,
    channel_axis: int | None = None,
    contrast_mode: str = "Percentile",
    progress: ProgressContext | None = None,
) -> Float32ThumbnailStatistics:
    """Return exact limits from a borrowed runtime-owned float32 device value.

    The caller must already be inside the value's active private execution
    scope.  This adapter validates that ownership, but never uploads, releases,
    or retains the borrowed input.  It returns only host limits and bounded
    auxiliary-transfer observations; all provider-owned scratch is relinquished
    before return.
    """

    # Keep the borrowed value in a clearable holder.  An exception traceback
    # may retain this frame until scope cleanup; clearing the holder prevents
    # the provider from extending the caller-owned alias lifetime.
    device_values_holder = [device_values]
    device_values = None
    try:
        runtime.allocation_identity(device_values_holder[0])
        mode, axis = _float32_thumbnail_contract(
            device_values_holder[0],
            channel_axis=channel_axis,
            contrast_mode=contrast_mode,
        )
        total_work = _float32_total_work(device_values_holder[0], mode)
        _check_cancelled(progress)
        _report(
            progress,
            0,
            total_work,
            "Using resident float32 thumbnail values on GPU · "
            "cancel applies after this pass",
        )
        return _exact_float32_thumbnail_limits_from_device_owned(
            runtime,
            _cupy_module(),
            device_values_holder,
            device_id=device_id,
            axis=axis,
            mode=mode,
            progress=progress,
            total_work=total_work,
        )
    finally:
        device_values_holder.clear()


def _exact_float32_thumbnail_limits_from_device_owned(
    runtime,
    cupy: ModuleType,
    device_values_holder: list[object],
    *,
    device_id: str,
    axis: int | None,
    mode: str,
    progress: ProgressContext | None,
    total_work: int,
) -> Float32ThumbnailStatistics:
    """Apply the shared bounded implementation to one validated device value."""

    device_values = device_values_holder[0]
    channel_count = 1 if axis is None else int(device_values.shape[axis])
    pass_count = 1 if mode == "minmax" else 5
    completed_work = 0
    auxiliary_host_to_device_bytes = 0
    device_to_host_bytes = 0
    device_to_host_values = 0
    device_channel = None
    channel_result = None
    limits: list[tuple[float, float]] = []
    try:
        for channel in range(channel_count):
            device_channel = (
                device_values
                if axis is None
                else _device_axis_view(device_values, axis, channel)
            )
            channel_size = int(device_channel.size)
            channel_result = _exact_float32_channel_limits(
                runtime,
                cupy,
                device_channel,
                device_id=device_id,
                mode=mode,
                progress=progress,
                completed_work=completed_work,
                total_work=total_work,
                channel_number=channel + 1,
                channel_count=channel_count,
            )
            limits.append(channel_result.limits)
            auxiliary_host_to_device_bytes += (
                channel_result.auxiliary_host_to_device_bytes
            )
            device_to_host_bytes += channel_result.device_to_host_bytes
            device_to_host_values += channel_result.device_to_host_values
            completed_work += channel_size * pass_count
        _report(
            progress,
            total_work,
            total_work,
            "Float32 thumbnail GPU statistics ready",
        )
    finally:
        channel_result = None
        device_channel = None
        device_values = None
    host_limits = np.asarray(limits, dtype=np.float64)
    return Float32ThumbnailStatistics(
        limits=host_limits[0] if axis is None else host_limits,
        auxiliary_host_to_device_bytes=auxiliary_host_to_device_bytes,
        device_to_host_bytes=device_to_host_bytes,
        device_to_host_values=device_to_host_values,
    )


def _exact_float32_channel_limits(
    runtime,
    cupy: ModuleType,
    device_channel,
    *,
    device_id: str,
    mode: str,
    progress: ProgressContext | None,
    completed_work: int,
    total_work: int,
    channel_number: int,
    channel_count: int,
) -> _Float32ChannelStatistics:
    channel_size = int(device_channel.size)
    block_count = min(
        max((channel_size + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK, 1),
        _MAXIMUM_BLOCKS,
    )
    contiguous_channel = None
    channel_copy = None
    device_metadata = None
    try:
        contiguous_channel = cupy.ascontiguousarray(device_channel)
        if contiguous_channel is not device_channel:
            channel_copy = contiguous_channel
        device_metadata = runtime.to_device(
            _FLOAT32_METADATA_INITIAL,
            device_id=device_id,
        )
        _report(
            progress,
            completed_work,
            total_work,
            _channel_message(
                "Classifying exact float32 thumbnail values on GPU",
                channel_number,
                channel_count,
            )
            + " · cancel applies after this pass",
        )
        _float32_metadata_kernel(cupy)(
            (block_count,),
            (_THREADS_PER_BLOCK,),
            (
                contiguous_channel,
                np.uint64(channel_size),
                device_metadata,
            ),
        )
        runtime.synchronize(device_id=device_id)
        host_metadata = np.asarray(
            runtime.to_host(device_metadata),
            dtype=np.uint64,
        )
        _check_cancelled(progress)
        after_metadata = completed_work + channel_size
        _report(
            progress,
            after_metadata,
            total_work,
            _channel_message(
                "Float32 thumbnail value classification ready",
                channel_number,
                channel_count,
            ),
        )

        finite_count = int(host_metadata[0])
        negative_infinity_count = int(host_metadata[1])
        has_negative_finite = bool(host_metadata[2])
        if not finite_count:
            return _Float32ChannelStatistics(
                (0.0, 0.0),
                _FLOAT32_METADATA_INITIAL.nbytes,
                _FLOAT32_METADATA_INITIAL.nbytes,
                _FLOAT32_METADATA_INITIAL.size,
            )
        minimum, maximum = _float32_extrema_from_metadata(host_metadata)
        if mode == "minmax":
            return _Float32ChannelStatistics(
                (minimum, maximum),
                _FLOAT32_METADATA_INITIAL.nbytes,
                _FLOAT32_METADATA_INITIAL.nbytes,
                _FLOAT32_METADATA_INITIAL.size,
            )

        valid_count = finite_count + (
            negative_infinity_count if has_negative_finite else 0
        )
        virtual, previous_indices, next_indices = _linear_percentile_indices(
            valid_count
        )
        target_ranks = np.asarray(
            [
                previous_indices[0],
                next_indices[0],
                previous_indices[1],
                next_indices[1],
            ],
            dtype=np.uint64,
        )
        selection = _radix_select_float32_bits(
            runtime,
            cupy,
            contiguous_channel,
            device_id=device_id,
            include_negative_infinity=has_negative_finite,
            target_ranks=target_ranks,
            block_count=block_count,
            progress=progress,
            completed_work=after_metadata,
            total_work=total_work,
            channel_size=channel_size,
            channel_number=channel_number,
            channel_count=channel_count,
        )
        extrema_bits = np.asarray([host_metadata[3], host_metadata[4]], dtype=np.uint32)
        selected = np.concatenate((extrema_bits, selection.bits)).view(np.float32)
        limits = _linear_percentile_from_selected(
            selected,
            virtual=virtual,
            previous_indices=previous_indices,
        )
        if limits[1] <= limits[0]:
            limits = (minimum, maximum)
        return _Float32ChannelStatistics(
            limits,
            _FLOAT32_METADATA_INITIAL.nbytes + selection.auxiliary_host_to_device_bytes,
            _FLOAT32_METADATA_INITIAL.nbytes + selection.device_to_host_bytes,
            _FLOAT32_METADATA_INITIAL.size + selection.device_to_host_values,
        )
    finally:
        release_errors = []
        for value in (
            device_metadata,
            channel_copy,
        ):
            if value is None:
                continue
            try:
                runtime.release(value)
            except Exception as exc:
                release_errors.append(exc)
        value = None
        device_metadata = None
        channel_copy = None
        contiguous_channel = None
        device_channel = None
        if release_errors:
            detail = "; ".join(f"{type(exc).__name__}: {exc}" for exc in release_errors)
            raise ThumbnailStatisticsProviderCleanupError(
                "Float32 thumbnail channel allocations could not be relinquished "
                "cleanly: " + detail
            ) from release_errors[0]


def _radix_select_float32_bits(
    runtime,
    cupy: ModuleType,
    device_values,
    *,
    device_id: str,
    include_negative_infinity: bool,
    target_ranks: np.ndarray,
    block_count: int,
    progress: ProgressContext | None,
    completed_work: int,
    total_work: int,
    channel_size: int,
    channel_number: int,
    channel_count: int,
) -> _RadixSelection:
    # A cancellation exception keeps traceback frames alive until the runtime
    # scope classifies it.  Store the device alias only in a mutable holder so
    # the wrapper can clear that alias before scope cleanup inspects the pool.
    device_values_holder = [device_values]
    device_values = None
    try:
        return _radix_select_float32_bits_owned(
            runtime,
            cupy,
            device_values_holder,
            device_id=device_id,
            include_negative_infinity=include_negative_infinity,
            target_ranks=target_ranks,
            block_count=block_count,
            progress=progress,
            completed_work=completed_work,
            total_work=total_work,
            channel_size=channel_size,
            channel_number=channel_number,
            channel_count=channel_count,
        )
    finally:
        device_values_holder.clear()


def _radix_select_float32_bits_owned(
    runtime,
    cupy: ModuleType,
    device_values_holder: list[object],
    *,
    device_id: str,
    include_negative_infinity: bool,
    target_ranks: np.ndarray,
    block_count: int,
    progress: ProgressContext | None,
    completed_work: int,
    total_work: int,
    channel_size: int,
    channel_number: int,
    channel_count: int,
) -> _RadixSelection:
    """Select exact non-negative float32 order keys with bounded histograms."""

    ranks = np.asarray(target_ranks, dtype=np.uint64)
    prefixes = np.zeros(ranks.shape, dtype=np.uint32)
    auxiliary_host_to_device_bytes = 0
    device_to_host_bytes = 0
    device_to_host_values = 0
    for pass_index, byte_shift in enumerate((24, 16, 8, 0)):
        _check_cancelled(progress)
        unique_prefixes, inverse = np.unique(prefixes, return_inverse=True)
        unique_prefixes = np.asarray(unique_prefixes, dtype=np.uint32)
        device_prefixes = None
        device_histograms = None
        try:
            device_prefixes = runtime.to_device(
                unique_prefixes,
                device_id=device_id,
            )
            device_histograms = cupy.zeros(
                (int(unique_prefixes.size), 256),
                dtype=cupy.uint64,
            )
            _report(
                progress,
                completed_work + pass_index * channel_size,
                total_work,
                _channel_message(
                    "Selecting exact float32 thumbnail percentiles on GPU "
                    f"· radix pass {pass_index + 1}/4",
                    channel_number,
                    channel_count,
                )
                + " · cancel applies after this pass",
            )
            _float32_radix_histogram_kernel(cupy)(
                (block_count,),
                (_THREADS_PER_BLOCK,),
                (
                    device_values_holder[0],
                    np.uint64(channel_size),
                    np.uint64(1 if include_negative_infinity else 0),
                    device_prefixes,
                    np.uint64(unique_prefixes.size),
                    np.uint32(pass_index * 8),
                    np.uint32(byte_shift),
                    device_histograms,
                ),
            )
            runtime.synchronize(device_id=device_id)
            host_histograms = np.asarray(
                runtime.to_host(device_histograms),
                dtype=np.uint64,
            )
            auxiliary_host_to_device_bytes += int(unique_prefixes.nbytes)
            device_to_host_bytes += int(host_histograms.nbytes)
            device_to_host_values += int(host_histograms.size)
            _check_cancelled(progress)
        finally:
            release_errors = []
            for value in (device_histograms, device_prefixes):
                if value is None:
                    continue
                try:
                    runtime.release(value)
                except Exception as exc:
                    release_errors.append(exc)
            value = None
            device_histograms = None
            device_prefixes = None
            if release_errors:
                detail = "; ".join(
                    f"{type(exc).__name__}: {exc}" for exc in release_errors
                )
                raise ThumbnailStatisticsProviderCleanupError(
                    "Float32 thumbnail radix allocations could not be "
                    "relinquished cleanly: " + detail
                ) from release_errors[0]

        next_prefixes = np.empty_like(prefixes)
        next_ranks = np.empty_like(ranks)
        for target_index, row_index in enumerate(inverse):
            cumulative = np.cumsum(host_histograms[int(row_index)], dtype=np.uint64)
            rank = ranks[target_index]
            bucket = int(np.searchsorted(cumulative, rank, side="right"))
            if bucket >= 256:
                raise RuntimeError(
                    "Float32 thumbnail radix selection observed an impossible rank."
                )
            preceding = np.uint64(0) if bucket == 0 else cumulative[bucket - 1]
            next_ranks[target_index] = rank - preceding
            next_prefixes[target_index] = np.uint32(
                (int(prefixes[target_index]) << 8) | bucket
            )
        prefixes = next_prefixes
        ranks = next_ranks
        _report(
            progress,
            completed_work + (pass_index + 1) * channel_size,
            total_work,
            _channel_message(
                "Exact float32 thumbnail radix pass ready",
                channel_number,
                channel_count,
            ),
        )
    return _RadixSelection(
        bits=prefixes,
        auxiliary_host_to_device_bytes=auxiliary_host_to_device_bytes,
        device_to_host_bytes=device_to_host_bytes,
        device_to_host_values=device_to_host_values,
    )


def _float32_extrema_from_metadata(metadata: np.ndarray) -> tuple[float, float]:
    bits = np.asarray([metadata[3], metadata[4]], dtype=np.uint32)
    values = bits.view(np.float32)
    return (float(values[0]), float(values[1]))


def _linear_percentile_indices(
    value_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Preserve NumPy's exact percent-to-quantile conversion.  In particular,
    # 99.9 / 100 is not bitwise interchangeable with a pre-rounded 0.999
    # literal at gamma boundaries next to an extreme float32 value.
    quantiles = np.true_divide(
        np.asarray((0.5, 99.9), dtype=np.float64),
        100.0,
    )
    virtual = (int(value_count) - 1) * quantiles
    previous = np.floor(virtual).astype(np.int64)
    following = np.minimum(previous + 1, int(value_count) - 1)
    return virtual, previous, following


def _linear_percentile_from_selected(
    selected: np.ndarray,
    *,
    virtual: np.ndarray,
    previous_indices: np.ndarray,
) -> tuple[float, float]:
    values = np.asarray(selected, dtype=np.float32)
    if values.shape != (6,):
        raise ValueError("Float32 percentile selection must contain six values.")
    previous = np.asarray((values[2], values[4]), dtype=np.float32)
    following = np.asarray((values[3], values[5]), dtype=np.float32)
    gamma = np.asarray(virtual, dtype=np.float64) - np.asarray(
        previous_indices,
        dtype=np.float64,
    )
    difference = np.subtract(following, previous)
    interpolated = np.add(previous, difference * gamma)
    np.subtract(
        following,
        difference * (1.0 - gamma),
        out=interpolated,
        where=gamma >= 0.5,
        casting="unsafe",
        dtype=interpolated.dtype,
    )
    return (float(interpolated[0]), float(interpolated[1]))


def _float32_thumbnail_contract(
    values,
    *,
    channel_axis: int | None,
    contrast_mode: str,
) -> tuple[str, int | None]:
    try:
        dtype = np.dtype(values.dtype)
    except (AttributeError, TypeError) as exc:
        raise TypeError(
            "CuPy float thumbnail statistics require native float32."
        ) from exc
    if dtype != np.dtype(np.float32):
        raise TypeError("CuPy float thumbnail statistics require native float32.")
    mode = _contrast_mode_key(contrast_mode)
    if mode not in {"percentile", "minmax"}:
        raise ValueError(
            "Float32 GPU thumbnail statistics require Percentile or Min-max."
        )
    axis = (
        None
        if channel_axis is None
        else _normalized_channel_axis(channel_axis, int(values.ndim))
    )
    return mode, axis


def _float32_total_work(values, mode: str) -> int:
    pass_count = 1 if mode == "minmax" else 5
    return max(int(values.size) * pass_count, 1)


def _normalized_channel_axis(channel_axis: int, ndim: int) -> int:
    if ndim <= 0:
        raise ValueError("A scalar array cannot have a channel axis.")
    axis = int(channel_axis)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(
            f"Channel axis {channel_axis} is outside an array with {ndim} axes."
        )
    return axis


def _device_axis_view(values, axis: int, index: int):
    selection = [slice(None)] * int(values.ndim)
    selection[axis] = index
    return values[tuple(selection)]


def _contrast_mode_key(contrast_mode: str) -> str:
    text = str(contrast_mode or "").strip().lower()
    if text in {"min-max", "minmax", "minimum-maximum", "minimum maximum"}:
        return "minmax"
    if text == "raw":
        return "raw"
    return "percentile"


def _channel_message(message: str, channel: int, channel_count: int) -> str:
    if channel_count <= 1:
        return message
    return f"{message} · channel {channel}/{channel_count}"


def _check_cancelled(progress: ProgressContext | None) -> None:
    if progress is not None:
        progress.check_cancelled()


def _report(
    progress: ProgressContext | None,
    current: int,
    total: int,
    message: str,
) -> None:
    if progress is not None:
        progress.report(current, total, message)


__all__ = [
    "Float32ThumbnailStatistics",
    "ThumbnailStatisticsProviderCleanupError",
    "exact_float32_thumbnail_limits",
    "exact_float32_thumbnail_limits_from_device",
    "exact_uint_histogram_counts",
]
