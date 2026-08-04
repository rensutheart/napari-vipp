"""Device-resident cuCIM provider for VIPP's basic object measurements.

The provider returns one C-contiguous ``float64`` CuPy matrix.  VIPP's mixed
Python table is deliberately reconstructed only after that matrix crosses the
host boundary by :func:`napari_vipp.core.measurements.finalize_basic_measurement_table`.

Arbitrary positive ``int32`` object IDs are compacted independently in each
leading block.  cuCIM then computes batched area, bounding box, and centroid
properties for the dense private IDs.  Euler number uses a pinned private
cuCIM kernel because the public region-properties route currently has an
unrelated feature-probe side effect.  Intensity statistics use two grouped
float64 passes so standard deviation is not formed from ``E[x^2]-E[x]^2``.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from functools import cache
from types import ModuleType

import numpy as np

from napari_vipp.core.measurements import (
    BasicMeasurementLayout,
    basic_measurement_layout,
    validate_basic_measurement_options,
)

_SUPPORTED_CUCIM_VERSION = "26.06.00"
_MAXIMUM_SPATIAL_BLOCK_ELEMENTS = 2**31 - 2
_LABEL_DTYPE = np.dtype(np.int32)
_SUPPORTED_INTENSITY_DTYPES = frozenset(
    {
        np.dtype(bool),
        np.dtype(np.uint8),
        np.dtype(np.uint16),
        np.dtype(np.float32),
    }
)
_MORPHOLOGY_PROPERTIES = ("label", "num_pixels", "bbox", "centroid")


@dataclass(frozen=True, slots=True)
class _MorphologyColumns:
    size: object
    centroids: tuple[object, ...]
    bbox_minimums: tuple[object, ...]
    bbox_maximums: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _IntensityFirstPass:
    group_ids: object
    values: object
    counts: object
    means: object
    minimums: object
    maximums: object
    sums: object


@cache
def _cupy_module() -> ModuleType:
    """Load CuPy only after this implementation has been selected."""

    return importlib.import_module("cupy")


@cache
def _cupyx_ndimage_module() -> ModuleType:
    """Load grouped extrema support only for an explicit provider call."""

    return importlib.import_module("cupyx.scipy.ndimage")


@cache
def _validated_cucim_modules() -> tuple[ModuleType, object]:
    """Load and gate the private cuCIM topology dependency."""

    cucim = importlib.import_module("cucim")
    version = str(getattr(cucim, "__version__", ""))
    if version != _SUPPORTED_CUCIM_VERSION:
        raise RuntimeError(
            "The cuCIM measurement provider is pinned to cuCIM "
            f"{_SUPPORTED_CUCIM_VERSION} because it uses a reviewed private "
            f"Euler-number API; received {version or 'an unknown version'}."
        )
    measure = importlib.import_module("cucim.skimage.measure")
    private_module = importlib.import_module(
        "cucim.skimage.measure._regionprops_gpu_misc_kernels"
    )
    euler = getattr(private_module, "regionprops_euler", None)
    if not callable(euler):
        raise RuntimeError(
            "cuCIM 26.06.00 does not expose the reviewed private "
            "regionprops_euler API required by VIPP."
        )
    return measure, euler


def measure_objects(
    data,
    spatial_mode: str = "Auto from axes",
    measurement_set: str = "Basic morphology",
    include_shape_descriptors: bool = False,
    include_axis_descriptors: bool = False,
    include_2d_boundary_descriptors: bool = False,
    include_derived_shape_ratios: bool = False,
    include_2d_shape_moments: bool = False,
    resolved_spatial_ndim: int | None = None,
    axis_names: tuple[str, ...] | None = None,
    axis_types: tuple[str, ...] | None = None,
    axis_scales: tuple[float, ...] | None = None,
    axis_units: tuple[str | None, ...] | None = None,
    source_name: str = "",
    progress=None,
):
    """Return packed resident basic morphology for an ``int32`` label image."""

    del measurement_set, source_name
    validate_basic_measurement_options(
        include_shape_descriptors=include_shape_descriptors,
        include_axis_descriptors=include_axis_descriptors,
        include_2d_boundary_descriptors=include_2d_boundary_descriptors,
        include_derived_shape_ratios=include_derived_shape_ratios,
        include_2d_shape_moments=include_2d_shape_moments,
    )
    cupy = _cupy_module()
    labels = _validated_labels(data, cupy=cupy)
    layout = basic_measurement_layout(
        labels.shape,
        spatial_mode=spatial_mode,
        resolved_spatial_ndim=resolved_spatial_ndim,
        axis_names=axis_names,
        axis_types=axis_types,
        axis_scales=axis_scales,
        axis_units=axis_units,
        include_intensity=False,
    )
    return _measure_basic(
        labels,
        None,
        layout=layout,
        progress=progress,
        cupy=cupy,
    )


def measure_objects_with_intensity(
    inputs,
    spatial_mode: str = "Auto from axes",
    measurement_set: str = "Basic morphology + intensity",
    include_shape_descriptors: bool = False,
    include_axis_descriptors: bool = False,
    include_2d_boundary_descriptors: bool = False,
    include_derived_shape_ratios: bool = False,
    include_2d_shape_moments: bool = False,
    resolved_spatial_ndim: int | None = None,
    axis_names: tuple[str, ...] | None = None,
    axis_types: tuple[str, ...] | None = None,
    axis_scales: tuple[float, ...] | None = None,
    axis_units: tuple[str | None, ...] | None = None,
    source_name: str = "",
    progress=None,
):
    """Return packed morphology and intensity statistics on the device."""

    del measurement_set, source_name
    validate_basic_measurement_options(
        include_shape_descriptors=include_shape_descriptors,
        include_axis_descriptors=include_axis_descriptors,
        include_2d_boundary_descriptors=include_2d_boundary_descriptors,
        include_derived_shape_ratios=include_derived_shape_ratios,
        include_2d_shape_moments=include_2d_shape_moments,
    )
    labels_data, intensity_data = _measurement_inputs(inputs)
    cupy = _cupy_module()
    labels = _validated_labels(labels_data, cupy=cupy)
    intensity = _validated_intensity(intensity_data, cupy=cupy)
    if tuple(int(size) for size in intensity.shape) != tuple(
        int(size) for size in labels.shape
    ):
        raise ValueError(
            "Intensity-aware measurements require labels and intensity image "
            "with the same shape."
        )
    layout = basic_measurement_layout(
        labels.shape,
        spatial_mode=spatial_mode,
        resolved_spatial_ndim=resolved_spatial_ndim,
        axis_names=axis_names,
        axis_types=axis_types,
        axis_scales=axis_scales,
        axis_units=axis_units,
        include_intensity=True,
    )
    return _measure_basic(
        labels,
        intensity,
        layout=layout,
        progress=progress,
        cupy=cupy,
    )


def _measure_basic(
    labels,
    intensity,
    *,
    layout: BasicMeasurementLayout,
    progress,
    cupy: ModuleType,
):
    measure, euler_function = _validated_cucim_modules()
    ndimage = _cupyx_ndimage_module() if intensity is not None else None
    spatial_elements = math.prod(layout.spatial_shape)
    if spatial_elements >= _MAXIMUM_SPATIAL_BLOCK_ELEMENTS:
        raise ValueError(
            "Each GPU measurement spatial block must contain fewer than "
            f"{_MAXIMUM_SPATIAL_BLOCK_ELEMENTS:,} elements so compact int32 "
            "object IDs remain valid."
        )
    if spatial_elements == 0 and layout.block_count:
        raise ValueError("GPU measurements require non-empty spatial blocks.")

    working_labels = cupy.ascontiguousarray(cupy.transpose(labels, layout.permutation))
    working_intensity = (
        cupy.ascontiguousarray(cupy.transpose(intensity, layout.permutation))
        if intensity is not None
        else None
    )
    stages_per_block = 6 if intensity is not None else 4
    total = max(layout.block_count * stages_per_block + 1, 1)
    completed = 0
    _progress_initial(progress, total)
    packed_blocks: list[object] = []
    indexes = np.ndindex(layout.leading_shape) if layout.leading_shape else ((),)

    for block_number, leading_index in enumerate(indexes, start=1):
        block = (
            working_labels[leading_index]
            if layout.leading_shape
            else working_labels
        )
        intensity_block = (
            working_intensity[leading_index]
            if working_intensity is not None and layout.leading_shape
            else working_intensity
        )

        _progress_before(progress)
        object_ids, dense_labels = _compact_labels(block, cupy=cupy)
        object_count = int(object_ids.size)
        completed = _progress_after(
            progress,
            cupy,
            completed,
            total,
            _stage_message("compacting labels", block_number, layout.block_count),
        )

        _progress_before(progress)
        morphology = _morphology_columns(
            dense_labels,
            object_count,
            layout.spatial_ndim,
            measure=measure,
            cupy=cupy,
        )
        completed = _progress_after(
            progress,
            cupy,
            completed,
            total,
            _stage_message("measuring morphology", block_number, layout.block_count),
        )

        _progress_before(progress)
        euler_numbers = _euler_numbers(
            dense_labels,
            object_count,
            euler_function=euler_function,
            cupy=cupy,
        )
        completed = _progress_after(
            progress,
            cupy,
            completed,
            total,
            _stage_message("measuring topology", block_number, layout.block_count),
        )

        intensity_columns = None
        if intensity_block is not None:
            assert ndimage is not None
            _progress_before(progress)
            first_pass = _intensity_first_pass(
                dense_labels,
                intensity_block,
                object_count,
                ndimage=ndimage,
                cupy=cupy,
            )
            completed = _progress_after(
                progress,
                cupy,
                completed,
                total,
                _stage_message(
                    "measuring intensity ranges and means",
                    block_number,
                    layout.block_count,
                ),
            )

            _progress_before(progress)
            standard_deviations = _intensity_second_pass(
                first_pass,
                object_count,
                cupy=cupy,
            )
            intensity_columns = (
                first_pass.means,
                first_pass.minimums,
                first_pass.maximums,
                first_pass.sums,
                standard_deviations,
            )
            completed = _progress_after(
                progress,
                cupy,
                completed,
                total,
                _stage_message(
                    "measuring intensity variation",
                    block_number,
                    layout.block_count,
                ),
            )

        _progress_before(progress)
        packed_blocks.append(
            _pack_block(
                leading_index,
                object_ids,
                morphology,
                euler_numbers,
                intensity_columns,
                layout=layout,
                cupy=cupy,
            )
        )
        completed = _progress_after(
            progress,
            cupy,
            completed,
            total,
            _stage_message("packing rows", block_number, layout.block_count),
        )

    _progress_before(progress)
    if packed_blocks:
        result = cupy.ascontiguousarray(
            cupy.concatenate(packed_blocks, axis=0),
            dtype=cupy.float64,
        )
    else:
        result = cupy.empty((0, layout.packed_width), dtype=cupy.float64)
    completed = _progress_after(
        progress,
        cupy,
        completed,
        total,
        "Object measurements: assembling packed table",
    )
    if completed != total:  # pragma: no cover - private arithmetic guard
        raise RuntimeError("GPU measurement progress accounting is inconsistent.")
    return result


def _validated_labels(data, *, cupy: ModuleType):
    source_dtype = getattr(data, "dtype", None)
    if source_dtype is not None:
        source_dtype = np.dtype(source_dtype)
        if source_dtype != _LABEL_DTYPE or not source_dtype.isnative:
            raise ValueError(
                "GPU measurements require native int32 labels; received "
                f"{source_dtype}."
            )
    labels = cupy.asarray(data)
    dtype = np.dtype(labels.dtype)
    if dtype != _LABEL_DTYPE or not dtype.isnative:
        raise ValueError(
            "GPU measurements require native int32 labels; received "
            f"{dtype}."
        )
    if int(labels.size) and _device_scalar_bool(cupy.any(labels < 0)):
        raise ValueError("GPU measurements require non-negative label IDs.")
    return labels


def _validated_intensity(data, *, cupy: ModuleType):
    source_dtype = getattr(data, "dtype", None)
    if source_dtype is not None:
        source_dtype = np.dtype(source_dtype)
        if source_dtype not in _SUPPORTED_INTENSITY_DTYPES or not source_dtype.isnative:
            raise ValueError(
                "GPU intensity measurements support native bool, uint8, uint16, "
                f"and finite float32 input; received {source_dtype}."
            )
    intensity = cupy.asarray(data)
    dtype = np.dtype(intensity.dtype)
    if dtype not in _SUPPORTED_INTENSITY_DTYPES or not dtype.isnative:
        raise ValueError(
            "GPU intensity measurements support native bool, uint8, uint16, "
            f"and finite float32 input; received {dtype}."
        )
    if dtype == np.dtype(np.float32) and int(intensity.size):
        if not _device_scalar_bool(cupy.all(cupy.isfinite(intensity))):
            raise ValueError("GPU float32 intensity measurements require finite data.")
    return intensity


def _measurement_inputs(inputs) -> tuple[object, object]:
    try:
        values = list(inputs)
    except Exception as exc:
        raise ValueError(
            "Intensity-aware measurements require labels and intensity image inputs."
        ) from exc
    if len(values) < 2:
        raise ValueError(
            "Intensity-aware measurements require labels and intensity image inputs."
        )
    return values[0], values[1]


def _compact_labels(block, *, cupy: ModuleType) -> tuple[object, object]:
    object_ids = cupy.unique(block)
    object_ids = object_ids[object_ids > 0]
    if int(object_ids.size) == 0:
        return object_ids, cupy.zeros(block.shape, dtype=cupy.int32)
    dense = cupy.where(
        block > 0,
        cupy.searchsorted(object_ids, block) + 1,
        0,
    )
    return object_ids, cupy.ascontiguousarray(dense, dtype=cupy.int32)


def _morphology_columns(
    dense_labels,
    object_count: int,
    spatial_ndim: int,
    *,
    measure: ModuleType,
    cupy: ModuleType,
) -> _MorphologyColumns:
    if object_count == 0:
        empty = cupy.empty((0,), dtype=cupy.float64)
        return _MorphologyColumns(
            size=empty,
            centroids=tuple(empty for _ in range(spatial_ndim)),
            bbox_minimums=tuple(empty for _ in range(spatial_ndim)),
            bbox_maximums=tuple(empty for _ in range(spatial_ndim)),
        )
    raw = measure.regionprops_table(
        dense_labels,
        properties=_MORPHOLOGY_PROPERTIES,
        batch_processing=True,
    )
    required = (
        "num_pixels",
        *(f"centroid-{index}" for index in range(spatial_ndim)),
        *(f"bbox-{index}" for index in range(2 * spatial_ndim)),
    )
    missing = tuple(name for name in required if name not in raw)
    if missing:
        raise RuntimeError(
            "cuCIM basic morphology omitted required properties: "
            + ", ".join(missing)
        )
    columns = {
        name: cupy.asarray(raw[name], dtype=cupy.float64) for name in required
    }
    malformed = tuple(
        name
        for name, values in columns.items()
        if tuple(int(size) for size in values.shape) != (object_count,)
    )
    if malformed:
        raise RuntimeError(
            "cuCIM basic morphology returned malformed properties: "
            + ", ".join(malformed)
        )
    return _MorphologyColumns(
        size=columns["num_pixels"],
        centroids=tuple(columns[f"centroid-{index}"] for index in range(spatial_ndim)),
        bbox_minimums=tuple(columns[f"bbox-{index}"] for index in range(spatial_ndim)),
        bbox_maximums=tuple(
            columns[f"bbox-{spatial_ndim + index}"]
            for index in range(spatial_ndim)
        ),
    )


def _euler_numbers(
    dense_labels,
    object_count: int,
    *,
    euler_function,
    cupy: ModuleType,
):
    if object_count == 0:
        return cupy.empty((0,), dtype=cupy.float64)
    values = euler_function(
        dense_labels,
        connectivity=None,
        max_label=object_count,
        robust=True,
    )
    values = cupy.asarray(values)
    if tuple(int(size) for size in values.shape) != (object_count,):
        raise RuntimeError("cuCIM Euler-number output has an unexpected shape.")
    return values.astype(cupy.float64, copy=False)


def _intensity_first_pass(
    dense_labels,
    intensity,
    object_count: int,
    *,
    ndimage: ModuleType,
    cupy: ModuleType,
) -> _IntensityFirstPass:
    if object_count == 0:
        empty = cupy.empty((0,), dtype=cupy.float64)
        empty_groups = cupy.empty((0,), dtype=cupy.int32)
        return _IntensityFirstPass(
            empty_groups,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
        )
    flattened_labels = dense_labels.reshape(-1)
    positive = flattened_labels > 0
    group_ids = flattened_labels[positive]
    values = intensity.reshape(-1)[positive].astype(cupy.float64, copy=False)
    # A stable grouping retains each object's C-order sample sequence.  CuPy's
    # float64 ufunc reductions then follow the same pairwise reduction shape
    # as NumPy on the validated platforms, while avoiding nondeterministic
    # atomic weighted-bincount accumulation under severe cancellation.
    order = cupy.argsort(group_ids, kind="stable")
    group_ids = group_ids[order]
    values = values[order]
    integer_counts = cupy.bincount(
        group_ids,
        minlength=object_count + 1,
    )[1:]
    starts = cupy.cumsum(integer_counts, dtype=cupy.int64) - integer_counts
    sums = cupy.add.reduceat(values, starts).astype(cupy.float64, copy=False)
    counts = integer_counts.astype(
        cupy.float64,
        copy=False,
    )
    means = sums / counts
    indexes = cupy.arange(1, object_count + 1, dtype=cupy.int32)
    minimums = ndimage.minimum(values, labels=group_ids, index=indexes).astype(
        cupy.float64,
        copy=False,
    )
    maximums = ndimage.maximum(values, labels=group_ids, index=indexes).astype(
        cupy.float64,
        copy=False,
    )
    return _IntensityFirstPass(
        group_ids=group_ids,
        values=values,
        counts=counts,
        means=means,
        minimums=minimums,
        maximums=maximums,
        sums=sums,
    )


def _intensity_second_pass(
    first_pass: _IntensityFirstPass,
    object_count: int,
    *,
    cupy: ModuleType,
):
    if object_count == 0:
        return cupy.empty((0,), dtype=cupy.float64)
    deviations = first_pass.values - first_pass.means[first_pass.group_ids - 1]
    squared_deviations = deviations * deviations
    integer_counts = first_pass.counts.astype(cupy.int64, copy=False)
    starts = cupy.cumsum(integer_counts, dtype=cupy.int64) - integer_counts
    summed_squared_deviations = cupy.add.reduceat(
        squared_deviations,
        starts,
    ).astype(cupy.float64, copy=False)
    return cupy.sqrt(summed_squared_deviations / first_pass.counts)


def _pack_block(
    leading_index: tuple[int, ...],
    object_ids,
    morphology: _MorphologyColumns,
    euler_numbers,
    intensity_columns,
    *,
    layout: BasicMeasurementLayout,
    cupy: ModuleType,
):
    object_count = int(object_ids.size)
    if object_count == 0:
        return cupy.empty((0, layout.packed_width), dtype=cupy.float64)
    columns = [
        cupy.full(object_count, value, dtype=cupy.float64)
        for value in leading_index
    ]
    columns.extend(
        (
            object_ids.astype(cupy.float64, copy=False),
            morphology.size,
            *morphology.centroids,
            *morphology.bbox_minimums,
            *morphology.bbox_maximums,
            euler_numbers,
        )
    )
    if intensity_columns is not None:
        columns.extend(intensity_columns)
    if len(columns) != layout.packed_width:
        raise RuntimeError("GPU measurement packing violated its declared layout.")
    return cupy.ascontiguousarray(
        cupy.column_stack(columns),
        dtype=cupy.float64,
    )


def _device_scalar_bool(value) -> bool:
    """Transfer one validation scalar; image-sized values stay resident."""

    return bool(value.item())


def _progress_initial(progress, total: int) -> None:
    if progress is None:
        return
    progress.check_cancelled()
    progress.report(0, total, "Object measurements: preparing")


def _progress_before(progress) -> None:
    if progress is not None:
        progress.check_cancelled()


def _progress_after(
    progress,
    cupy: ModuleType,
    completed: int,
    total: int,
    message: str,
) -> int:
    completed += 1
    if progress is None:
        return completed
    cupy.cuda.get_current_stream().synchronize()
    progress.check_cancelled()
    progress.report(completed, total, message)
    return completed


def _stage_message(stage: str, block_number: int, block_count: int) -> str:
    return f"Object measurements: {stage} (block {block_number}/{block_count})"


__all__ = ["measure_objects", "measure_objects_with_intensity"]
