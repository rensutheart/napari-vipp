"""Exact device-resident CuPy provider for ``Analyze Skeleton``.

Only already-skeletonized boolean inputs are accepted.  CuPyX performs
full-connectivity component labeling, while CuPy reductions reproduce VIPP's
voxel graph after removing the same diagonal shortcuts as the CPU reference.
The provider returns one versioned ``uint8`` payload; public table construction
is deferred to the declared host finalizer.
"""

from __future__ import annotations

import importlib
import math
from functools import cache
from itertools import combinations, product
from types import ModuleType

import numpy as np

from napari_vipp.core.skeleton_measurements import (
    SKELETON_PAYLOAD_HEADER_BYTES,
    SkeletonAnalysisLayout,
    skeleton_analysis_layout,
    skeleton_payload_header,
)

_MAXIMUM_SPATIAL_BLOCK_ELEMENTS = 2**31 - 2
_BOOLEAN_DTYPE = np.dtype(bool)


@cache
def _cupy_module() -> ModuleType:
    """Load CuPy only after this implementation has been selected."""

    return importlib.import_module("cupy")


@cache
def _cupyx_ndimage_module() -> ModuleType:
    """Load CuPyX ndimage only for an explicit GPU execution."""

    return importlib.import_module("cupyx.scipy.ndimage")


def analyze_skeleton(
    data,
    spatial_mode: str = "Auto from axes",
    input_mode: str = "Already skeletonized",
    resolved_spatial_ndim: int | None = None,
    axis_names: tuple[str, ...] | None = None,
    axis_types: tuple[str, ...] | None = None,
    axis_scales: tuple[float, ...] | None = None,
    axis_units: tuple[str | None, ...] | None = None,
    source_name: str = "",
    progress=None,
):
    """Return a resident packed payload of exact skeleton-component metrics."""

    del source_name
    _validate_input_mode(input_mode)
    cupy = _cupy_module()
    cupyx_ndimage = _cupyx_ndimage_module()
    skeleton = _validated_skeleton(data, cupy=cupy)
    layout = skeleton_analysis_layout(
        skeleton.shape,
        spatial_mode=spatial_mode,
        resolved_spatial_ndim=resolved_spatial_ndim,
        axis_names=axis_names,
        axis_types=axis_types,
        axis_scales=axis_scales,
        axis_units=axis_units,
    )
    return _pack_skeleton_payload(
        skeleton,
        layout=layout,
        progress=progress,
        cupy=cupy,
        cupyx_ndimage=cupyx_ndimage,
    )


def _validated_skeleton(data, *, cupy: ModuleType):
    source_dtype = getattr(data, "dtype", None)
    if source_dtype is not None and np.dtype(source_dtype) != _BOOLEAN_DTYPE:
        raise ValueError(
            "GPU Analyze Skeleton requires a boolean skeleton mask; "
            f"received {np.dtype(source_dtype)}."
        )
    skeleton = cupy.asarray(data)
    if np.dtype(skeleton.dtype) != _BOOLEAN_DTYPE:
        raise ValueError(
            "GPU Analyze Skeleton requires a boolean skeleton mask; "
            f"received {np.dtype(skeleton.dtype)}."
        )
    return skeleton


def _pack_skeleton_payload(
    skeleton,
    *,
    layout: SkeletonAnalysisLayout,
    progress,
    cupy: ModuleType,
    cupyx_ndimage: ModuleType,
):
    spatial_elements = math.prod(layout.spatial_shape)
    if spatial_elements >= _MAXIMUM_SPATIAL_BLOCK_ELEMENTS:
        raise ValueError(
            "Each GPU skeleton-analysis spatial block must contain fewer than "
            f"{_MAXIMUM_SPATIAL_BLOCK_ELEMENTS:,} elements so CuPyX int32 "
            "component IDs remain exact."
        )

    working = cupy.ascontiguousarray(cupy.transpose(skeleton, layout.permutation))
    structure = cupy.ones((3,) * layout.spatial_ndim, dtype=cupy.bool_)
    total = max(layout.block_count * 2 + 1, 1)
    completed = 0
    _progress_initial(progress, total)
    directory_blocks: list[object] = []
    row_count = 0
    indexes = np.ndindex(layout.leading_shape) if layout.leading_shape else ((),)

    for block_number, leading_index in enumerate(indexes, start=1):
        block = working[leading_index] if layout.leading_shape else working

        _progress_before(progress)
        labels = cupy.empty(block.shape, dtype=cupy.int32)
        if int(block.size):
            component_count_value = cupyx_ndimage.label(
                block,
                structure=structure,
                output=labels,
            )
            component_count = _scalar_integer(component_count_value)
        else:
            labels.fill(0)
            component_count = 0
        completed = _progress_after(
            progress,
            cupy,
            completed,
            total,
            _stage_message("labeling components", block_number, layout.block_count),
        )

        _progress_before(progress)
        directory = _measure_block(
            block,
            labels,
            leading_index,
            component_count=component_count,
            layout=layout,
            cupy=cupy,
        )
        directory_blocks.append(directory)
        row_count += component_count
        completed = _progress_after(
            progress,
            cupy,
            completed,
            total,
            _stage_message("measuring graph", block_number, layout.block_count),
        )

    _progress_before(progress)
    if directory_blocks:
        directory = cupy.ascontiguousarray(
            cupy.concatenate(directory_blocks, axis=0),
            dtype=cupy.uint64,
        )
    else:
        directory = cupy.empty((0, layout.record_words), dtype=cupy.uint64)
    if directory.shape != (row_count, layout.record_words):
        raise RuntimeError("GPU skeleton directory violated its declared layout.")
    header = cupy.asarray(skeleton_payload_header(layout=layout, row_count=row_count))
    directory_bytes = cupy.ascontiguousarray(directory).reshape(-1).view(cupy.uint8)
    result = cupy.ascontiguousarray(
        cupy.concatenate((header, directory_bytes)),
        dtype=cupy.uint8,
    )
    expected_size = SKELETON_PAYLOAD_HEADER_BYTES + row_count * layout.record_words * 8
    if int(result.size) != expected_size:
        raise RuntimeError("GPU skeleton payload assembly is inconsistent.")
    completed = _progress_after(
        progress,
        cupy,
        completed,
        total,
        "Analyze Skeleton: assembling payload",
    )
    if completed != total:  # pragma: no cover - private arithmetic guard
        raise RuntimeError("GPU skeleton progress accounting is inconsistent.")
    return result


def _measure_block(
    block,
    labels,
    leading_index: tuple[int, ...],
    *,
    component_count: int,
    layout: SkeletonAnalysisLayout,
    cupy: ModuleType,
):
    if component_count == 0:
        return cupy.empty((0, layout.record_words), dtype=cupy.uint64)

    flattened_labels = labels.reshape(-1)
    counts = cupy.bincount(
        flattened_labels,
        minlength=component_count + 1,
    )[1:].astype(cupy.uint64, copy=False)
    degrees = cupy.zeros(block.shape, dtype=cupy.uint8)
    edge_counts = cupy.zeros(component_count, dtype=cupy.uint64)
    pixel_lengths = cupy.zeros(component_count, dtype=cupy.float64)
    physical_lengths = cupy.zeros(component_count, dtype=cupy.float64)

    for offset in _half_neighbor_offsets(layout.spatial_ndim):
        source_slices, neighbor_slices = _neighbor_slices(block.shape, offset)
        valid = block[source_slices] & block[neighbor_slices]
        nonzero_axes = tuple(axis for axis, value in enumerate(offset) if value)
        if len(nonzero_axes) > 1:
            for subset_size in range(1, len(nonzero_axes)):
                for axes in combinations(nonzero_axes, subset_size):
                    intermediate_offset = tuple(
                        offset[axis] if axis in axes else 0
                        for axis in range(layout.spatial_ndim)
                    )
                    intermediate_slices = _shifted_slices(
                        block.shape,
                        source_slices,
                        intermediate_offset,
                    )
                    valid &= ~block[intermediate_slices]

        increment = valid.astype(cupy.uint8, copy=False)
        degrees[source_slices] += increment
        degrees[neighbor_slices] += increment
        edge_labels = labels[source_slices][valid]
        counts_for_offset = _component_bincount(
            edge_labels,
            component_count=component_count,
            cupy=cupy,
        )
        edge_counts += counts_for_offset
        pixel_lengths += counts_for_offset * _offset_length(
            offset,
            (1.0,) * layout.spatial_ndim,
        )
        physical_lengths += counts_for_offset * _offset_length(
            offset,
            layout.units.scales,
        )

    flattened_degrees = degrees.reshape(-1)
    endpoint_counts = _degree_category_counts(
        flattened_labels,
        flattened_degrees == 1,
        component_count=component_count,
        cupy=cupy,
    )
    degree_two_counts = _degree_category_counts(
        flattened_labels,
        flattened_degrees == 2,
        component_count=component_count,
        cupy=cupy,
    )
    junction_counts = _degree_category_counts(
        flattened_labels,
        flattened_degrees >= 3,
        component_count=component_count,
        cupy=cupy,
    )
    isolated_counts = _degree_category_counts(
        flattened_labels,
        flattened_degrees == 0,
        component_count=component_count,
        cupy=cupy,
    )
    foreground = flattened_labels > 0
    key_voxels = foreground & (flattened_degrees != 2)
    key_degree_sums = cupy.zeros(component_count + 1, dtype=cupy.uint64)
    cupy.add.at(
        key_degree_sums,
        flattened_labels[key_voxels],
        flattened_degrees[key_voxels].astype(cupy.uint64, copy=False),
    )
    key_degree_sums = key_degree_sums[1:]
    branch_counts = cupy.where(
        counts == 1,
        cupy.uint64(0),
        cupy.where(
            degree_two_counts == counts,
            cupy.uint64(1),
            key_degree_sums // cupy.uint64(2),
        ),
    ).astype(cupy.uint64, copy=False)
    graph_node_counts = endpoint_counts + junction_counts + isolated_counts
    cycle_counts_signed = (
        edge_counts.astype(cupy.int64) - counts.astype(cupy.int64) + cupy.int64(1)
    )
    cycle_counts = cycle_counts_signed.astype(cupy.uint64)

    component_ids = cupy.arange(1, component_count + 1, dtype=cupy.uint64)
    component_counts = cupy.full(
        component_count,
        np.uint64(component_count),
        dtype=cupy.uint64,
    )
    total_voxel_count = cupy.sum(counts, dtype=cupy.uint64)
    total_voxel_counts = cupy.full(
        component_count,
        total_voxel_count,
        dtype=cupy.uint64,
    )
    columns: list[object] = [
        cupy.full(component_count, np.uint64(value), dtype=cupy.uint64)
        for value in leading_index
    ]
    columns.extend(
        (
            component_ids,
            component_counts,
            total_voxel_counts,
            counts,
            endpoint_counts,
            junction_counts,
            isolated_counts,
            branch_counts,
            graph_node_counts,
            branch_counts,
            edge_counts,
            cycle_counts,
            cupy.ascontiguousarray(pixel_lengths).view(cupy.uint64),
            cupy.ascontiguousarray(physical_lengths).view(cupy.uint64),
        )
    )
    if len(columns) != layout.record_words:
        raise RuntimeError("GPU skeleton record packing violated its layout.")
    return cupy.ascontiguousarray(
        cupy.column_stack(columns),
        dtype=cupy.uint64,
    )


def _degree_category_counts(
    labels,
    category,
    *,
    component_count: int,
    cupy: ModuleType,
):
    selected = category & (labels > 0)
    return _component_bincount(
        labels[selected],
        component_count=component_count,
        cupy=cupy,
    )


def _component_bincount(values, *, component_count: int, cupy: ModuleType):
    if int(values.size) == 0:
        return cupy.zeros(component_count, dtype=cupy.uint64)
    return cupy.bincount(
        values,
        minlength=component_count + 1,
    )[1:].astype(cupy.uint64, copy=False)


def _half_neighbor_offsets(ndim: int) -> tuple[tuple[int, ...], ...]:
    zero = (0,) * ndim
    return tuple(
        offset
        for offset in product((-1, 0, 1), repeat=ndim)
        if offset != zero and offset > zero
    )


def _neighbor_slices(
    shape: tuple[int, ...],
    offset: tuple[int, ...],
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    source: list[slice] = []
    neighbor: list[slice] = []
    for size, delta in zip(shape, offset, strict=True):
        start = max(0, -delta)
        stop = min(int(size), int(size) - delta)
        source.append(slice(start, stop))
        neighbor.append(slice(start + delta, stop + delta))
    return tuple(source), tuple(neighbor)


def _shifted_slices(
    shape: tuple[int, ...],
    source_slices: tuple[slice, ...],
    offset: tuple[int, ...],
) -> tuple[slice, ...]:
    del shape  # Source endpoints already prove every partial offset is in bounds.
    return tuple(
        slice(int(axis_slice.start) + delta, int(axis_slice.stop) + delta)
        for axis_slice, delta in zip(source_slices, offset, strict=True)
    )


def _offset_length(
    offset: tuple[int, ...],
    scales: tuple[float, ...],
) -> float:
    return float(
        np.sqrt(
            np.sum(
                [
                    (float(offset[index]) * float(scales[index])) ** 2
                    for index in range(len(offset))
                ]
            )
        )
    )


def _scalar_integer(value) -> int:
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def _validate_input_mode(input_mode: object) -> None:
    if str(input_mode).strip().casefold() != "already skeletonized":
        raise ValueError(
            "GPU Analyze Skeleton supports only 'Already skeletonized'; "
            "skeletonization remains on CPU."
        )


def _progress_initial(progress, total: int) -> None:
    if progress is None:
        return
    progress.check_cancelled()
    progress.report(0, total, "Analyze Skeleton: preparing")


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
    return f"Analyze Skeleton: {stage} (block {block_number}/{block_count})"


__all__ = ["analyze_skeleton"]
