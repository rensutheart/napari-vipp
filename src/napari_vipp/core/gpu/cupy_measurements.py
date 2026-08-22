"""Device-resident CuPy provider for VIPP's basic object measurements.

The provider returns one C-contiguous ``float64`` CuPy matrix. VIPP's mixed
Python table is reconstructed only after that matrix crosses the host boundary
by :func:`napari_vipp.core.measurements.finalize_basic_measurement_table`.

Arbitrary positive ``int32`` object IDs are compacted independently in each
leading block. Custom CuPy raw kernels compute area, bounding boxes, centroids,
and per-object Euler numbers without cuCIM. Intensity statistics use grouped
CuPy/CuPyX reductions and two float64 passes so standard deviation is not
formed from ``E[x^2]-E[x]^2``.
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

# Coefficients and bit order match skimage.measure.euler_number for the full
# connectivity used by VIPP (8-connected in 2D and 26-connected in 3D).
_EULER_COEFFICIENTS_2D = (
    0,
    0,
    0,
    0,
    0,
    0,
    -1,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    -1,
    0,
)
_EULER_COEFFICIENTS_3D = (
    0,
    1,
    1,
    0,
    1,
    0,
    -2,
    -1,
    1,
    -2,
    0,
    -1,
    0,
    -1,
    -1,
    0,
    1,
    0,
    -2,
    -1,
    -2,
    -1,
    -1,
    -2,
    -6,
    -3,
    -3,
    -2,
    -3,
    -2,
    0,
    -1,
    1,
    -2,
    0,
    -1,
    -6,
    -3,
    -3,
    -2,
    -2,
    -1,
    -1,
    -2,
    -3,
    0,
    -2,
    -1,
    0,
    -1,
    -1,
    0,
    -3,
    -2,
    0,
    -1,
    -3,
    0,
    -2,
    -1,
    0,
    1,
    1,
    0,
    1,
    -2,
    -6,
    -3,
    0,
    -1,
    -3,
    -2,
    -2,
    -1,
    -3,
    0,
    -1,
    -2,
    -2,
    -1,
    0,
    -1,
    -3,
    -2,
    -1,
    0,
    0,
    -1,
    -3,
    0,
    0,
    1,
    -2,
    -1,
    1,
    0,
    -2,
    -1,
    -3,
    0,
    -3,
    0,
    0,
    1,
    -1,
    4,
    0,
    3,
    0,
    3,
    1,
    2,
    -1,
    -2,
    -2,
    -1,
    -2,
    -1,
    1,
    0,
    0,
    3,
    1,
    2,
    1,
    2,
    2,
    1,
    1,
    -6,
    -2,
    -3,
    -2,
    -3,
    -1,
    0,
    0,
    -3,
    -1,
    -2,
    -1,
    -2,
    -2,
    -1,
    -2,
    -3,
    -1,
    0,
    -1,
    0,
    4,
    3,
    -3,
    0,
    0,
    1,
    0,
    1,
    3,
    2,
    0,
    -3,
    -1,
    -2,
    -3,
    0,
    0,
    1,
    -1,
    0,
    0,
    -1,
    -2,
    1,
    -1,
    0,
    -1,
    -2,
    -2,
    -1,
    0,
    1,
    3,
    2,
    -2,
    1,
    -1,
    0,
    1,
    2,
    2,
    1,
    0,
    -3,
    -3,
    0,
    -1,
    -2,
    0,
    1,
    -1,
    0,
    -2,
    1,
    0,
    -1,
    -1,
    0,
    -1,
    -2,
    0,
    1,
    -2,
    -1,
    3,
    2,
    -2,
    1,
    1,
    2,
    -1,
    0,
    2,
    1,
    -1,
    0,
    -2,
    1,
    -2,
    1,
    1,
    2,
    -2,
    3,
    -1,
    2,
    -1,
    2,
    0,
    1,
    0,
    -1,
    -1,
    0,
    -1,
    0,
    2,
    1,
    -1,
    2,
    0,
    1,
    0,
    1,
    1,
    0,
)


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
    starts: object
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
    """Load grouped extrema support only for an explicit intensity call."""

    return importlib.import_module("cupyx.scipy.ndimage")


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
    return _measure_basic(labels, None, layout=layout, progress=progress, cupy=cupy)


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
            working_labels[leading_index] if layout.leading_shape else working_labels
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
        euler_numbers = _euler_numbers(dense_labels, object_count, cupy=cupy)
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
            f"GPU measurements require native int32 labels; received {dtype}."
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


def _launch(kernel, size: int, args: tuple[object, ...]) -> None:
    if size <= 0:
        return
    threads = 256
    blocks = (int(size) + threads - 1) // threads
    kernel((blocks,), (threads,), args)


@cache
def _morphology_kernel(cupy: ModuleType, ndim: int):
    if ndim == 2:
        source = r"""
extern "C" __global__
void vipp_measure_morphology_2d(
    const int* labels,
    const unsigned long long size,
    const unsigned int width,
    unsigned long long* counts,
    unsigned long long* sum_y,
    unsigned long long* sum_x,
    unsigned int* min_y,
    unsigned int* min_x,
    unsigned int* max_y,
    unsigned int* max_x)
{
    const unsigned long long index =
        (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= size) return;
    const int label = labels[index];
    if (label <= 0) return;
    const unsigned int target = (unsigned int)(label - 1);
    const unsigned int y = (unsigned int)(index / width);
    const unsigned int x =
        (unsigned int)(index - (unsigned long long)y * width);
    atomicAdd(&counts[target], 1ULL);
    atomicAdd(&sum_y[target], (unsigned long long)y);
    atomicAdd(&sum_x[target], (unsigned long long)x);
    atomicMin(&min_y[target], y);
    atomicMin(&min_x[target], x);
    atomicMax(&max_y[target], y + 1U);
    atomicMax(&max_x[target], x + 1U);
}
"""
        return cupy.RawKernel(source, "vipp_measure_morphology_2d")
    if ndim == 3:
        source = r"""
extern "C" __global__
void vipp_measure_morphology_3d(
    const int* labels,
    const unsigned long long size,
    const unsigned int height,
    const unsigned int width,
    unsigned long long* counts,
    unsigned long long* sum_z,
    unsigned long long* sum_y,
    unsigned long long* sum_x,
    unsigned int* min_z,
    unsigned int* min_y,
    unsigned int* min_x,
    unsigned int* max_z,
    unsigned int* max_y,
    unsigned int* max_x)
{
    const unsigned long long index =
        (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= size) return;
    const int label = labels[index];
    if (label <= 0) return;
    const unsigned int target = (unsigned int)(label - 1);
    const unsigned long long plane =
        (unsigned long long)height * (unsigned long long)width;
    const unsigned int z = (unsigned int)(index / plane);
    const unsigned long long rem = index - (unsigned long long)z * plane;
    const unsigned int y = (unsigned int)(rem / width);
    const unsigned int x =
        (unsigned int)(rem - (unsigned long long)y * width);
    atomicAdd(&counts[target], 1ULL);
    atomicAdd(&sum_z[target], (unsigned long long)z);
    atomicAdd(&sum_y[target], (unsigned long long)y);
    atomicAdd(&sum_x[target], (unsigned long long)x);
    atomicMin(&min_z[target], z);
    atomicMin(&min_y[target], y);
    atomicMin(&min_x[target], x);
    atomicMax(&max_z[target], z + 1U);
    atomicMax(&max_y[target], y + 1U);
    atomicMax(&max_x[target], x + 1U);
}
"""
        return cupy.RawKernel(source, "vipp_measure_morphology_3d")
    raise ValueError("CuPy basic morphology supports only 2D and 3D blocks.")


def _morphology_columns(
    dense_labels,
    object_count: int,
    spatial_ndim: int,
    *,
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

    labels = cupy.ascontiguousarray(dense_labels, dtype=cupy.int32)
    counts = cupy.zeros(object_count, dtype=cupy.uint64)
    coordinate_sums = [
        cupy.zeros(object_count, dtype=cupy.uint64) for _ in range(spatial_ndim)
    ]
    minimums = [
        cupy.full(object_count, np.iinfo(np.uint32).max, dtype=cupy.uint32)
        for _ in range(spatial_ndim)
    ]
    maximums = [
        cupy.zeros(object_count, dtype=cupy.uint32) for _ in range(spatial_ndim)
    ]
    kernel = _morphology_kernel(cupy, spatial_ndim)
    if spatial_ndim == 2:
        arguments = (
            labels,
            np.uint64(labels.size),
            np.uint32(labels.shape[1]),
            counts,
            *coordinate_sums,
            *minimums,
            *maximums,
        )
    else:
        arguments = (
            labels,
            np.uint64(labels.size),
            np.uint32(labels.shape[1]),
            np.uint32(labels.shape[2]),
            counts,
            *coordinate_sums,
            *minimums,
            *maximums,
        )
    _launch(kernel, int(labels.size), arguments)
    counts_f64 = counts.astype(cupy.float64)
    return _MorphologyColumns(
        size=counts_f64,
        centroids=tuple(
            values.astype(cupy.float64) / counts_f64 for values in coordinate_sums
        ),
        bbox_minimums=tuple(values.astype(cupy.float64) for values in minimums),
        bbox_maximums=tuple(values.astype(cupy.float64) for values in maximums),
    )


@cache
def _euler_kernel(cupy: ModuleType, ndim: int):
    if ndim == 2:
        coefficients = ",".join(str(value) for value in _EULER_COEFFICIENTS_2D)
        source = rf"""
extern "C" __global__
void vipp_measure_euler_2d(
    const int* labels,
    const int height,
    const int width,
    long long* output)
{{
    const unsigned long long padded_width = (unsigned long long)width + 1ULL;
    const unsigned long long size =
        ((unsigned long long)height + 1ULL) * padded_width;
    const unsigned long long index =
        (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= size) return;
    const int py = (int)(index / padded_width) - 1;
    const int px = (int)(index % padded_width) - 1;
    int values[4];
    const int ys[4] = {{py, py, py + 1, py + 1}};
    const int xs[4] = {{px, px + 1, px, px + 1}};
    #pragma unroll
    for (int lane = 0; lane < 4; ++lane) {{
        const int y = ys[lane];
        const int x = xs[lane];
        values[lane] = (y >= 0 && y < height && x >= 0 && x < width)
            ? labels[(unsigned long long)y * width + x]
            : 0;
    }}
    const int weights[4] = {{1, 4, 2, 8}};
    const int coefficients[16] = {{{coefficients}}};
    #pragma unroll
    for (int lane = 0; lane < 4; ++lane) {{
        const int label = values[lane];
        if (label <= 0) continue;
        bool duplicate = false;
        #pragma unroll
        for (int earlier = 0; earlier < lane; ++earlier) {{
            duplicate = duplicate || values[earlier] == label;
        }}
        if (duplicate) continue;
        int code = 0;
        #pragma unroll
        for (int bit = 0; bit < 4; ++bit) {{
            if (values[bit] == label) code += weights[bit];
        }}
        const long long contribution = (long long)coefficients[code];
        atomicAdd(
            reinterpret_cast<unsigned long long*>(&output[label - 1]),
            (unsigned long long)contribution
        );
    }}
}}
"""
        return cupy.RawKernel(
            source,
            "vipp_measure_euler_2d",
            options=("--std=c++11",),
        )
    if ndim == 3:
        coefficients = ",".join(str(value) for value in _EULER_COEFFICIENTS_3D)
        source = rf"""
extern "C" __global__
void vipp_measure_euler_3d(
    const int* labels,
    const int depth,
    const int height,
    const int width,
    long long* output)
{{
    const unsigned long long padded_width = (unsigned long long)width + 1ULL;
    const unsigned long long padded_height = (unsigned long long)height + 1ULL;
    const unsigned long long padded_plane = padded_height * padded_width;
    const unsigned long long size =
        ((unsigned long long)depth + 1ULL) * padded_plane;
    const unsigned long long index =
        (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= size) return;
    const int pz = (int)(index / padded_plane) - 1;
    const unsigned long long remainder = index % padded_plane;
    const int py = (int)(remainder / padded_width) - 1;
    const int px = (int)(remainder % padded_width) - 1;
    const int zs[8] = {{
        pz, pz, pz, pz, pz + 1, pz + 1, pz + 1, pz + 1
    }};
    const int ys[8] = {{
        py, py, py + 1, py + 1, py, py, py + 1, py + 1
    }};
    const int xs[8] = {{
        px, px + 1, px, px + 1, px, px + 1, px, px + 1
    }};
    const int weights[8] = {{1, 4, 2, 8, 16, 64, 32, 128}};
    const int coefficients[256] = {{{coefficients}}};
    int values[8];
    #pragma unroll
    for (int lane = 0; lane < 8; ++lane) {{
        const int z = zs[lane];
        const int y = ys[lane];
        const int x = xs[lane];
        values[lane] = (
            z >= 0 && z < depth
            && y >= 0 && y < height
            && x >= 0 && x < width
        ) ? labels[((unsigned long long)z * height + y) * width + x] : 0;
    }}
    #pragma unroll
    for (int lane = 0; lane < 8; ++lane) {{
        const int label = values[lane];
        if (label <= 0) continue;
        bool duplicate = false;
        #pragma unroll
        for (int earlier = 0; earlier < lane; ++earlier) {{
            duplicate = duplicate || values[earlier] == label;
        }}
        if (duplicate) continue;
        int code = 0;
        #pragma unroll
        for (int bit = 0; bit < 8; ++bit) {{
            if (values[bit] == label) code += weights[bit];
        }}
        const long long contribution = (long long)coefficients[code];
        atomicAdd(
            reinterpret_cast<unsigned long long*>(&output[label - 1]),
            (unsigned long long)contribution
        );
    }}
}}
"""
        return cupy.RawKernel(
            source,
            "vipp_measure_euler_3d",
            options=("--std=c++11",),
        )
    raise ValueError("CuPy Euler measurement supports only 2D and 3D blocks.")


def _euler_numbers(
    dense_labels,
    object_count: int,
    *,
    cupy: ModuleType,
):
    if object_count == 0:
        return cupy.empty((0,), dtype=cupy.float64)
    labels = cupy.ascontiguousarray(dense_labels, dtype=cupy.int32)
    output = cupy.zeros(object_count, dtype=cupy.int64)
    kernel = _euler_kernel(cupy, int(labels.ndim))
    if labels.ndim == 2:
        size = (int(labels.shape[0]) + 1) * (int(labels.shape[1]) + 1)
        arguments = (
            labels,
            np.int32(labels.shape[0]),
            np.int32(labels.shape[1]),
            output,
        )
    elif labels.ndim == 3:
        size = (
            (int(labels.shape[0]) + 1)
            * (int(labels.shape[1]) + 1)
            * (int(labels.shape[2]) + 1)
        )
        arguments = (
            labels,
            np.int32(labels.shape[0]),
            np.int32(labels.shape[1]),
            np.int32(labels.shape[2]),
            output,
        )
    else:  # pragma: no cover - guarded by basic_measurement_layout
        raise ValueError("CuPy Euler measurement supports only 2D and 3D blocks.")
    _launch(kernel, size, arguments)
    if labels.ndim == 3:
        output //= 8
    return output.astype(cupy.float64, copy=False)


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
            empty_groups,
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
    # A stable grouping retains each object's C-order sample sequence. The
    # deterministic segmented kernels below avoid both atomic accumulation and
    # CuPy reduceat edge cases for adjacent short groups.
    order = cupy.argsort(group_ids, kind="stable")
    group_ids = group_ids[order]
    values = values[order]
    integer_counts = cupy.ascontiguousarray(
        cupy.bincount(
            group_ids,
            minlength=object_count + 1,
        )[1:],
        dtype=cupy.int64,
    )
    starts = cupy.ascontiguousarray(
        cupy.cumsum(integer_counts, dtype=cupy.int64) - integer_counts,
        dtype=cupy.int64,
    )
    sums = _segmented_sums(
        values,
        starts,
        integer_counts,
        object_count,
        cupy=cupy,
    )
    counts = integer_counts.astype(cupy.float64, copy=False)
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
        starts=starts,
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
    integer_counts = first_pass.counts.astype(cupy.int64, copy=False)
    summed_squared_deviations = _segmented_squared_deviation_sums(
        first_pass.values,
        first_pass.starts,
        integer_counts,
        first_pass.means,
        object_count,
        cupy=cupy,
    )
    variances = summed_squared_deviations / first_pass.counts
    # Population standard deviation is exactly zero for a singleton object.
    # Explicitly masking that case also makes the mathematical contract clear.
    variances = cupy.where(
        first_pass.counts <= 1.0,
        cupy.float64(0.0),
        cupy.maximum(variances, cupy.float64(0.0)),
    )
    return cupy.sqrt(variances)


@cache
def _segmented_sum_kernel(cupy: ModuleType):
    source = r"""
extern "C" __global__
void vipp_segmented_sum_f64(
    const double* values,
    const long long* starts,
    const long long* counts,
    double* output,
    const int segment_count
) {
    const int segment = (int)blockIdx.x;
    if (segment >= segment_count) return;
    const int lane = (int)threadIdx.x;
    const long long start = starts[segment];
    const long long stop = start + counts[segment];
    double subtotal = 0.0;
    for (long long index = start + lane; index < stop; index += blockDim.x) {
        subtotal += values[index];
    }
    __shared__ double partial[256];
    partial[lane] = subtotal;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    if (lane == 0) output[segment] = partial[0];
}
"""
    return cupy.RawKernel(source, "vipp_segmented_sum_f64")


@cache
def _segmented_squared_deviation_sum_kernel(cupy: ModuleType):
    source = r"""
extern "C" __global__
void vipp_segmented_squared_deviation_sum_f64(
    const double* values,
    const long long* starts,
    const long long* counts,
    const double* means,
    double* output,
    const int segment_count
) {
    const int segment = (int)blockIdx.x;
    if (segment >= segment_count) return;
    const int lane = (int)threadIdx.x;
    const long long start = starts[segment];
    const long long stop = start + counts[segment];
    const double mean = means[segment];
    double subtotal = 0.0;
    for (long long index = start + lane; index < stop; index += blockDim.x) {
        const double deviation = values[index] - mean;
        subtotal += deviation * deviation;
    }
    __shared__ double partial[256];
    partial[lane] = subtotal;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    if (lane == 0) output[segment] = partial[0];
}
"""
    return cupy.RawKernel(
        source,
        "vipp_segmented_squared_deviation_sum_f64",
    )


def _segmented_sums(values, starts, counts, object_count: int, *, cupy: ModuleType):
    output = cupy.empty(object_count, dtype=cupy.float64)
    if object_count:
        _segmented_sum_kernel(cupy)(
            (object_count,),
            (256,),
            (values, starts, counts, output, np.int32(object_count)),
        )
    return output


def _segmented_squared_deviation_sums(
    values,
    starts,
    counts,
    means,
    object_count: int,
    *,
    cupy: ModuleType,
):
    output = cupy.empty(object_count, dtype=cupy.float64)
    if object_count:
        _segmented_squared_deviation_sum_kernel(cupy)(
            (object_count,),
            (256,),
            (values, starts, counts, means, output, np.int32(object_count)),
        )
    return output


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
        cupy.full(object_count, value, dtype=cupy.float64) for value in leading_index
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
