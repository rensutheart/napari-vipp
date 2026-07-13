"""Exact, Qt-free diagnostic calculations for scientific image arrays.

These helpers inspect every relevant value. They do not sample arrays or infer
RGB/channel semantics from shape; callers must provide an explicit channel axis
when a per-channel histogram is required.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from numbers import Rational

import numpy as np
from scipy import ndimage as ndi

from napari_vipp.core.operations import exact_integer_percentiles

DIAGNOSTIC_CHUNK_ELEMENTS = 1_048_576
DISPLAY_HISTOGRAM_BINS = 128


@dataclass(frozen=True, slots=True)
class ExactFiniteStats:
    """Exact finite-value count and extrema for one array."""

    count: int
    minimum: int | float
    maximum: int | float


def exact_finite_stats(data) -> ExactFiniteStats:
    """Return exact finite count and extrema using bounded temporaries."""
    arr = np.asarray(data)
    integer_data = np.issubdtype(arr.dtype, np.integer)
    count = 0
    minimum: int | float | None = None
    maximum: int | float | None = None
    for values in _iter_finite_numeric_chunks(arr):
        if values.size == 0:
            continue
        count += int(values.size)
        chunk_minimum = (
            int(values.min()) if integer_data else float(values.min())
        )
        chunk_maximum = (
            int(values.max()) if integer_data else float(values.max())
        )
        minimum = chunk_minimum if minimum is None else min(minimum, chunk_minimum)
        maximum = chunk_maximum if maximum is None else max(maximum, chunk_maximum)
    if count == 0:
        return ExactFiniteStats(0, 0.0, 0.0)
    assert minimum is not None and maximum is not None
    return ExactFiniteStats(count, minimum, maximum)


def exact_finite_percentiles(
    data,
    percentiles: tuple[float, ...],
) -> tuple[int | float | Rational, ...] | None:
    """Calculate NumPy-linear percentiles over every finite input value.

    Extrema avoid a full-data copy. Interior percentiles necessarily retain
    all finite values because fixed display bins cannot recover exact order
    statistics. Integer inputs use exact rational interpolation so native
    int64/uint64 levels are never collapsed through float conversion.
    """
    requested = _validated_percentiles(percentiles)
    stats = exact_finite_stats(data)
    if stats.count == 0:
        return None
    if stats.minimum == stats.maximum:
        return tuple(stats.minimum for _value in requested)
    if all(value in {0.0, 100.0} for value in requested):
        return tuple(
            stats.minimum if value == 0.0 else stats.maximum
            for value in requested
        )

    arr = np.asarray(data)
    if arr.dtype == np.dtype(bool):
        raise ValueError(
            "Interior percentiles are undefined for boolean data; convert it "
            "to an explicit numeric dtype first."
        )
    if np.issubdtype(arr.dtype, np.integer) and arr.dtype != np.dtype(bool):
        return exact_integer_percentiles(arr, requested)
    values = np.empty(stats.count, dtype=arr.dtype)
    offset = 0
    for chunk in _iter_finite_numeric_chunks(arr):
        if chunk.size == 0:
            continue
        stop = offset + int(chunk.size)
        values[offset:stop] = chunk
        offset = stop
    if offset != stats.count:
        raise RuntimeError("Finite-value count changed during percentile calculation.")
    result = np.percentile(values, requested, overwrite_input=True)
    return tuple(float(value) for value in np.asarray(result).ravel())


def exact_histogram(
    data,
    *,
    channel_axis: int | None = None,
) -> tuple[
    np.ndarray | None,
    tuple[int | float, int | float] | None,
]:
    """Return an exact display histogram and its numeric range.

    ``channel_axis=None`` always treats the complete array as one scalar-value
    population. Supplying a channel axis returns one row per channel using a
    shared range. No trailing dimension is implicitly interpreted as RGB.
    """
    arr = np.asarray(data)
    if channel_axis is None:
        return _single_histogram(arr)
    axis = _normalized_channel_axis(channel_axis, arr.ndim)
    return _multichannel_histogram(arr, axis)


def exact_generated_layer_contrast_limits(
    data,
) -> tuple[float, float] | None:
    """Return display limits containing every finite value and zero."""
    stats = exact_finite_stats(data)
    if stats.count == 0:
        return None
    low = min(float(stats.minimum), 0.0)
    high = max(float(stats.maximum), 0.0)
    if low == high:
        return (0.0, 1.0)
    return (float(low), float(high))


def provisional_generated_layer_contrast_limits(
    data,
) -> tuple[float, float]:
    """Return scan-free temporary limits while exact extrema are calculated."""
    try:
        dtype = np.dtype(data.dtype)
    except (AttributeError, TypeError):
        dtype = np.asarray(data).dtype
    if dtype == np.dtype(bool):
        return (0.0, 1.0)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        low = min(float(info.min), 0.0)
        high = max(float(info.max), 0.0)
        if low < high:
            return (low, high)
    return (0.0, 1.0)


def label_volumes(labels, spatial_ndim: int) -> np.ndarray:
    """Count each positive label within each independent leading-axis block.

    The final ``spatial_ndim`` axes form one spatial block. Leading axes (for
    example time or channel) are evaluated independently, so the same numeric
    label ID in two leading-axis positions represents two separate objects.
    """
    arr = np.asarray(labels)
    spatial_ndim = _validated_spatial_ndim(arr, spatial_ndim)
    if arr.size == 0:
        return np.array([], dtype=np.int64)
    volumes: list[np.ndarray] = []
    for block in _spatial_blocks(arr, spatial_ndim):
        foreground = np.asarray(block)
        foreground = foreground[foreground > 0]
        if foreground.size == 0:
            continue
        _labels, counts = np.unique(foreground, return_counts=True)
        volumes.append(counts.astype(np.int64, copy=False))
    if not volumes:
        return np.array([], dtype=np.int64)
    return np.concatenate(volumes)


def largest_label_volume(labels, spatial_ndim: int) -> int:
    """Return the largest positive-label count across spatial blocks."""
    volumes = label_volumes(labels, spatial_ndim)
    return int(volumes.max()) if volumes.size else 0


def largest_object_size(
    objects,
    spatial_ndim: int,
    connectivity: str,
) -> int:
    """Return the largest labeled object or boolean connected component.

    Non-boolean inputs retain their existing positive label IDs. Boolean masks
    are connected-component labeled within each spatial block; ``Face
    connected`` uses rank-1 connectivity and ``Full connectivity`` uses the
    full spatial rank.
    """
    arr = np.asarray(objects)
    spatial_ndim = _validated_spatial_ndim(arr, spatial_ndim)
    if arr.size == 0:
        return 0
    if arr.dtype != np.dtype(bool):
        return largest_label_volume(arr, spatial_ndim)

    normalized_connectivity = str(connectivity).strip().lower()
    if normalized_connectivity == "face connected":
        rank = 1
    elif normalized_connectivity in {"full connectivity", "fully connected"}:
        rank = spatial_ndim
    else:
        raise ValueError(
            "Connectivity must be 'Face connected' or 'Full connectivity'."
        )
    structure = ndi.generate_binary_structure(spatial_ndim, rank)
    largest = 0
    for block in _spatial_blocks(arr, spatial_ndim):
        component_labels, count = ndi.label(block, structure=structure)
        if count:
            largest = max(
                largest,
                int(np.bincount(component_labels.ravel())[1:].max()),
            )
    return largest


def _iter_finite_numeric_chunks(data) -> Iterator[np.ndarray]:
    """Yield every finite numeric value without a full-size temporary mask."""
    if data is None:
        return
    try:
        iterator = np.nditer(
            np.asarray(data),
            flags=["buffered", "external_loop", "refs_ok", "zerosize_ok"],
            op_flags=[["readonly"]],
            order="K",
            buffersize=DIAGNOSTIC_CHUNK_ELEMENTS,
        )
    except (TypeError, ValueError):
        return
    for chunk in iterator:
        values = np.asarray(chunk)
        try:
            finite = np.isfinite(values)
        except TypeError:
            return
        if not finite.all():
            values = values[finite]
        yield values


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


def _validated_percentiles(
    percentiles: tuple[float, ...],
) -> tuple[float, ...]:
    requested: list[float] = []
    for value in percentiles:
        try:
            percentile = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Percentiles must be finite numeric values.") from exc
        if not np.isfinite(percentile):
            raise ValueError("Percentiles must be finite numeric values.")
        if not 0.0 <= percentile <= 100.0:
            raise ValueError("Percentiles must be between 0 and 100.")
        requested.append(percentile)
    return tuple(requested)


def _validated_spatial_ndim(arr: np.ndarray, spatial_ndim: int) -> int:
    if arr.ndim == 0:
        raise ValueError("Label diagnostics require at least one array axis.")
    try:
        resolved = int(spatial_ndim)
    except (TypeError, ValueError) as exc:
        raise ValueError("Spatial dimensionality must be an integer.") from exc
    if resolved < 1 or resolved > arr.ndim:
        raise ValueError(
            f"Spatial dimensionality {resolved} is outside an array with "
            f"{arr.ndim} axes."
        )
    return resolved


def _single_histogram(
    arr: np.ndarray,
) -> tuple[np.ndarray | None, tuple[int | float, int | float] | None]:
    if arr.dtype == np.dtype(bool):
        true_count = int(np.count_nonzero(arr))
        return (
            np.array([arr.size - true_count, true_count], dtype=np.int64),
            (0.0, 1.0),
        )

    stats = exact_finite_stats(arr)
    if stats.count == 0:
        return None, None
    if stats.minimum == stats.maximum:
        return np.array([stats.count], dtype=np.int64), (
            stats.minimum,
            stats.maximum,
        )
    if np.issubdtype(arr.dtype, np.integer):
        return _exact_integer_display_histogram(arr, stats)
    edges, value_range = _display_histogram_edges(stats)
    return _exact_histogram_counts(arr, edges), value_range


def _multichannel_histogram(
    arr: np.ndarray,
    channel_axis: int,
) -> tuple[np.ndarray | None, tuple[int | float, int | float] | None]:
    channels = [
        _axis_index_view(arr, channel_axis, channel)
        for channel in range(arr.shape[channel_axis])
    ]
    channel_stats = [exact_finite_stats(values) for values in channels]
    valid_stats = [stats for stats in channel_stats if stats.count]
    if not valid_stats:
        return None, None

    if arr.dtype == np.dtype(bool):
        counts = []
        for values in channels:
            true_count = int(np.count_nonzero(values))
            counts.append(
                np.array(
                    [values.size - true_count, true_count],
                    dtype=np.int64,
                )
            )
        return np.vstack(counts), (0.0, 1.0)

    combined = ExactFiniteStats(
        sum(stats.count for stats in valid_stats),
        min(stats.minimum for stats in valid_stats),
        max(stats.maximum for stats in valid_stats),
    )
    if combined.minimum == combined.maximum:
        counts = np.array(
            [[stats.count] for stats in channel_stats],
            dtype=np.int64,
        )
        return counts, (combined.minimum, combined.maximum)
    if np.issubdtype(arr.dtype, np.integer):
        histogram_minimum, histogram_maximum, bin_count = (
            _integer_display_histogram_configuration(arr.dtype, combined)
        )
        counts = [
            _exact_integer_display_histogram_counts(
                values,
                histogram_minimum=histogram_minimum,
                histogram_maximum=histogram_maximum,
                bin_count=bin_count,
            )
            for values in channels
        ]
        return np.vstack(counts), (histogram_minimum, histogram_maximum)

    edges, value_range = _display_histogram_edges(combined)
    counts = [
        (
            _exact_histogram_counts(values, edges)
            if stats.count
            else np.zeros(edges.size - 1, dtype=np.int64)
        )
        for values, stats in zip(channels, channel_stats, strict=True)
    ]
    return np.vstack(counts), value_range


def _exact_integer_display_histogram(
    data,
    stats: ExactFiniteStats,
) -> tuple[np.ndarray, tuple[int, int]]:
    histogram_minimum, histogram_maximum, bin_count = (
        _integer_display_histogram_configuration(np.asarray(data).dtype, stats)
    )
    counts = _exact_integer_display_histogram_counts(
        data,
        histogram_minimum=histogram_minimum,
        histogram_maximum=histogram_maximum,
        bin_count=bin_count,
    )
    return counts, (histogram_minimum, histogram_maximum)


def _integer_display_histogram_configuration(
    dtype: np.dtype,
    stats: ExactFiniteStats,
) -> tuple[int, int, int]:
    del dtype
    minimum = int(stats.minimum)
    maximum = int(stats.maximum)
    if 0 <= minimum and maximum <= 255:
        return 0, 255, 256
    level_span = maximum - minimum + 1
    return minimum, maximum, min(DISPLAY_HISTOGRAM_BINS, level_span)


def _exact_integer_display_histogram_counts(
    data,
    *,
    histogram_minimum: int,
    histogram_maximum: int,
    bin_count: int,
) -> np.ndarray:
    """Count integer levels exactly after subtracting a Python-int offset."""
    level_span = histogram_maximum - histogram_minimum + 1
    counts = np.zeros(bin_count, dtype=np.int64)
    if level_span <= 65_536:
        native_counts = np.zeros(level_span, dtype=np.int64)
        for values in _iter_finite_numeric_chunks(data):
            if values.size == 0:
                continue
            levels, level_counts = np.unique(values, return_counts=True)
            indices = np.fromiter(
                (int(level) - histogram_minimum for level in levels),
                dtype=np.intp,
                count=levels.size,
            )
            native_counts[indices] += level_counts.astype(np.int64, copy=False)
        if bin_count == level_span:
            return native_counts
        boundaries = (
            np.arange(bin_count + 1, dtype=np.int64) * level_span // bin_count
        )
        return np.asarray(
            [
                native_counts[start:stop].sum(dtype=np.int64)
                for start, stop in zip(
                    boundaries[:-1],
                    boundaries[1:],
                    strict=True,
                )
            ],
            dtype=np.int64,
        )

    for values in _iter_finite_numeric_chunks(data):
        if values.size == 0:
            continue
        levels, level_counts = np.unique(values, return_counts=True)
        indices = np.fromiter(
            (
                min(
                    ((int(level) - histogram_minimum) * bin_count) // level_span,
                    bin_count - 1,
                )
                for level in levels
            ),
            dtype=np.intp,
            count=levels.size,
        )
        np.add.at(counts, indices, level_counts.astype(np.int64, copy=False))
    return counts


def _display_histogram_edges(
    stats: ExactFiniteStats,
) -> tuple[np.ndarray, tuple[float, float]]:
    return (
        np.linspace(
            stats.minimum,
            stats.maximum,
            DISPLAY_HISTOGRAM_BINS + 1,
        ),
        (stats.minimum, stats.maximum),
    )


def _exact_histogram_counts(data, edges: np.ndarray) -> np.ndarray:
    counts = np.zeros(int(edges.size) - 1, dtype=np.int64)
    for values in _iter_finite_numeric_chunks(data):
        if values.size:
            counts += np.histogram(values, bins=edges)[0]
    return counts


def _axis_index_view(arr: np.ndarray, axis: int, index: int) -> np.ndarray:
    selection = [slice(None)] * arr.ndim
    selection[int(axis)] = int(index)
    return arr[tuple(selection)]


def _spatial_blocks(arr: np.ndarray, spatial_ndim: int) -> Iterator[np.ndarray]:
    leading_shape = arr.shape[: arr.ndim - spatial_ndim]
    if leading_shape:
        for index in np.ndindex(leading_shape):
            yield arr[index]
        return
    yield arr


__all__ = [
    "DIAGNOSTIC_CHUNK_ELEMENTS",
    "DISPLAY_HISTOGRAM_BINS",
    "ExactFiniteStats",
    "exact_finite_percentiles",
    "exact_finite_stats",
    "exact_generated_layer_contrast_limits",
    "exact_histogram",
    "label_volumes",
    "largest_label_volume",
    "largest_object_size",
    "provisional_generated_layer_contrast_limits",
]
