"""Lazy CuPy payload provider for 3D mesh morphology measurements.

CuPy discovers sorted objects, voxel counts, and bounding boxes, then packs each
eligible bounding-box-local mask directly on the device.  The returned value is
one C-contiguous ``uint8`` CuPy array.  Dense crops use bitmasks while sparse
crops use uint32 local indices, strictly bounding encoded data to four bytes
per positive voxel.  Marching cubes, convex hulls, and public table construction
intentionally happen only in the declared host finalizer.
"""

from __future__ import annotations

import importlib
import math
from functools import cache
from types import ModuleType

import numpy as np

from napari_vipp.core.gpu import cupy_measurements
from napari_vipp.core.mesh_measurements import (
    MESH_ENCODING_BITMASK,
    MESH_ENCODING_NONE,
    MESH_ENCODING_SPARSE_UINT32,
    MESH_PAYLOAD_HEADER_BYTES,
    MeshMorphologyLayout,
    mesh_morphology_layout,
    mesh_payload_header,
)

_LABEL_DTYPE = np.dtype(np.int32)
_MAXIMUM_SPATIAL_BLOCK_ELEMENTS = 2**31 - 2


@cache
def _cupy_module() -> ModuleType:
    """Load CuPy only after this implementation has been selected."""

    return importlib.import_module("cupy")


def measure_3d_mesh_morphology(
    data,
    spatial_mode: str = "Auto from axes",
    minimum_voxel_count: int = 16,
    include_convex_hull_metrics: bool = True,
    resolved_spatial_ndim: int | None = None,
    axis_names: tuple[str, ...] | None = None,
    axis_types: tuple[str, ...] | None = None,
    axis_scales: tuple[float, ...] | None = None,
    axis_units: tuple[str | None, ...] | None = None,
    source_name: str = "",
    progress=None,
):
    """Return one resident, bit-packed payload for true-3D ``int32`` labels."""

    del source_name
    cupy = _cupy_module()
    labels = _validated_labels(data, cupy=cupy)
    layout = mesh_morphology_layout(
        labels.shape,
        spatial_mode=spatial_mode,
        resolved_spatial_ndim=resolved_spatial_ndim,
        axis_names=axis_names,
        axis_types=axis_types,
        axis_scales=axis_scales,
        axis_units=axis_units,
        include_convex_hull_metrics=include_convex_hull_metrics,
    )
    return _pack_mesh_payload(
        labels,
        layout=layout,
        minimum_voxel_count=max(int(minimum_voxel_count), 1),
        progress=progress,
        cupy=cupy,
    )


def _validated_labels(data, *, cupy: ModuleType):
    source_dtype = getattr(data, "dtype", None)
    if source_dtype is not None:
        source_dtype = np.dtype(source_dtype)
        if source_dtype != _LABEL_DTYPE or not source_dtype.isnative:
            raise ValueError(
                "GPU 3D mesh morphology requires native int32 labels; "
                f"received {source_dtype}."
            )
    labels = cupy.asarray(data)
    dtype = np.dtype(labels.dtype)
    if dtype != _LABEL_DTYPE or not dtype.isnative:
        raise ValueError(
            f"GPU 3D mesh morphology requires native int32 labels; received {dtype}."
        )
    if int(labels.size) and bool(cupy.any(labels < 0).item()):
        raise ValueError("GPU 3D mesh morphology requires non-negative label IDs.")
    return labels


def _pack_mesh_payload(
    labels,
    *,
    layout: MeshMorphologyLayout,
    minimum_voxel_count: int,
    progress,
    cupy: ModuleType,
):
    spatial_elements = math.prod(layout.spatial_shape)
    if spatial_elements >= _MAXIMUM_SPATIAL_BLOCK_ELEMENTS:
        raise ValueError(
            "Each GPU mesh morphology spatial block must contain fewer than "
            f"{_MAXIMUM_SPATIAL_BLOCK_ELEMENTS:,} elements so compact int32 "
            "object IDs remain valid."
        )
    if any(size > np.iinfo(np.uint32).max for size in layout.spatial_shape):
        raise ValueError(
            "GPU mesh morphology spatial dimensions must fit native uint32."
        )
    # The authored threshold is an unrestricted Python integer on the CPU path.
    # Values above one complete spatial block all have the same skip-everything
    # meaning, so canonicalize them before the device-side uint64 comparison.
    # This preserves that CPU meaning without overflowing ``np.uint64``.
    minimum_voxel_count = min(
        int(minimum_voxel_count),
        int(spatial_elements) + 1,
    )

    working = cupy.ascontiguousarray(cupy.transpose(labels, layout.permutation))
    total = max(layout.block_count * 3 + 1, 1)
    completed = 0
    _progress_initial(progress, total)
    directory_blocks: list[object] = []
    mask_blocks: list[object] = []
    global_mask_offset = 0
    row_count = 0
    indexes = np.ndindex(layout.leading_shape) if layout.leading_shape else ((),)

    for block_number, leading_index in enumerate(indexes, start=1):
        block = working[leading_index] if layout.leading_shape else working

        _progress_before(progress)
        object_ids, dense_labels = cupy_measurements._compact_labels(
            block,
            cupy=cupy,
        )
        object_count = int(object_ids.size)
        row_count += object_count
        completed = _progress_after(
            progress,
            cupy,
            completed,
            total,
            _stage_message("compacting labels", block_number, layout.block_count),
        )

        _progress_before(progress)
        morphology = cupy_measurements._morphology_columns(
            dense_labels,
            object_count,
            3,
            cupy=cupy,
        )
        completed = _progress_after(
            progress,
            cupy,
            completed,
            total,
            _stage_message("measuring bounds", block_number, layout.block_count),
        )

        _progress_before(progress)
        directory, mask_data = _pack_block(
            dense_labels,
            leading_index,
            object_ids,
            morphology,
            minimum_voxel_count=minimum_voxel_count,
            global_mask_offset=global_mask_offset,
            layout=layout,
            cupy=cupy,
        )
        directory_blocks.append(directory)
        mask_blocks.append(mask_data)
        global_mask_offset += int(mask_data.size)
        completed = _progress_after(
            progress,
            cupy,
            completed,
            total,
            _stage_message("packing masks", block_number, layout.block_count),
        )

    _progress_before(progress)
    if directory_blocks:
        directory = cupy.ascontiguousarray(
            cupy.concatenate(directory_blocks, axis=0),
            dtype=cupy.uint64,
        )
    else:  # pragma: no cover - every valid layout has one logical block
        directory = cupy.empty((0, layout.record_words), dtype=cupy.uint64)
    if mask_blocks:
        mask_data = cupy.ascontiguousarray(
            cupy.concatenate(mask_blocks),
            dtype=cupy.uint8,
        )
    else:  # pragma: no cover - every valid layout has one logical block
        mask_data = cupy.empty((0,), dtype=cupy.uint8)
    if directory.shape != (row_count, layout.record_words):
        raise RuntimeError("GPU mesh morphology directory violated its layout.")

    header = cupy.asarray(
        mesh_payload_header(
            layout=layout,
            row_count=row_count,
            mask_data_bytes=int(mask_data.size),
        )
    )
    directory_bytes = cupy.ascontiguousarray(directory).reshape(-1).view(cupy.uint8)
    result = cupy.ascontiguousarray(
        cupy.concatenate((header, directory_bytes, mask_data)),
        dtype=cupy.uint8,
    )
    expected_size = (
        MESH_PAYLOAD_HEADER_BYTES + int(directory_bytes.size) + int(mask_data.size)
    )
    if int(result.size) != expected_size:
        raise RuntimeError("GPU mesh morphology payload assembly is inconsistent.")
    completed = _progress_after(
        progress,
        cupy,
        completed,
        total,
        "3D mesh morphology: assembling payload",
    )
    if completed != total:  # pragma: no cover - private arithmetic guard
        raise RuntimeError("GPU mesh morphology progress accounting is inconsistent.")
    return result


def _pack_block(
    dense_labels,
    leading_index: tuple[int, ...],
    object_ids,
    morphology,
    *,
    minimum_voxel_count: int,
    global_mask_offset: int,
    layout: MeshMorphologyLayout,
    cupy: ModuleType,
) -> tuple[object, object]:
    object_count = int(object_ids.size)
    if object_count == 0:
        return (
            cupy.empty((0, layout.record_words), dtype=cupy.uint64),
            cupy.empty((0,), dtype=cupy.uint8),
        )

    counts = morphology.size.astype(cupy.uint64)
    minimums = tuple(values.astype(cupy.uint32) for values in morphology.bbox_minimums)
    maximums = tuple(values.astype(cupy.uint32) for values in morphology.bbox_maximums)
    bbox_bits = cupy.ones(object_count, dtype=cupy.uint64)
    for minimum, maximum in zip(minimums, maximums, strict=True):
        bbox_bits *= maximum.astype(cupy.uint64) - minimum.astype(cupy.uint64)
    eligible = counts >= np.uint64(minimum_voxel_count)
    minimum_bitmask_bytes = (bbox_bits + np.uint64(7)) // np.uint64(8)
    aligned_bitmask_bytes = (
        (minimum_bitmask_bytes + np.uint64(3)) // np.uint64(4)
    ) * np.uint64(4)
    sparse_bytes = counts * np.uint64(4)
    use_bitmask = eligible & (aligned_bitmask_bytes <= sparse_bytes)
    use_sparse = eligible & ~use_bitmask
    encodings = cupy.where(
        use_bitmask,
        np.uint64(MESH_ENCODING_BITMASK),
        cupy.where(
            use_sparse,
            np.uint64(MESH_ENCODING_SPARSE_UINT32),
            np.uint64(MESH_ENCODING_NONE),
        ),
    )
    data_nbytes = cupy.where(
        use_bitmask,
        aligned_bitmask_bytes,
        cupy.where(use_sparse, sparse_bytes, np.uint64(0)),
    )
    data_counts = cupy.where(
        use_bitmask,
        bbox_bits,
        cupy.where(use_sparse, counts, np.uint64(0)),
    )
    local_offsets = cupy.empty(object_count, dtype=cupy.uint64)
    local_offsets[0] = 0
    if object_count > 1:
        local_offsets[1:] = cupy.cumsum(data_nbytes[:-1], dtype=cupy.uint64)
    total_data_bytes = int(cupy.sum(data_nbytes, dtype=cupy.uint64).item())
    encoded_data = cupy.zeros(total_data_bytes, dtype=cupy.uint8)
    _launch_bitmask_kernel(
        dense_labels,
        minimums,
        maximums,
        encodings,
        local_offsets,
        data_nbytes,
        encoded_data,
        cupy=cupy,
    )
    _launch_sparse_kernel(
        dense_labels,
        minimums,
        maximums,
        encodings,
        local_offsets,
        encoded_data,
        cupy=cupy,
    )

    columns: list[object] = [
        cupy.full(object_count, np.uint64(value), dtype=cupy.uint64)
        for value in leading_index
    ]
    columns.extend(
        (
            object_ids.astype(cupy.uint64, copy=False),
            counts,
            *(values.astype(cupy.uint64) for values in minimums),
            *(values.astype(cupy.uint64) for values in maximums),
            encodings,
            local_offsets + np.uint64(global_mask_offset),
            data_nbytes,
            data_counts,
        )
    )
    if len(columns) != layout.record_words:
        raise RuntimeError("GPU mesh morphology record packing violated its layout.")
    directory = cupy.ascontiguousarray(
        cupy.column_stack(columns),
        dtype=cupy.uint64,
    )
    return directory, encoded_data


def _launch_bitmask_kernel(
    dense_labels,
    minimums,
    maximums,
    encodings,
    offsets,
    sizes,
    output,
    *,
    cupy: ModuleType,
) -> None:
    output_size = int(output.size)
    if output_size == 0:
        return
    kernel = _mask_pack_kernel(cupy)
    threads = 256
    blocks = (output_size + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            cupy.ascontiguousarray(dense_labels, dtype=cupy.int32),
            np.uint32(dense_labels.shape[0]),
            np.uint32(dense_labels.shape[1]),
            np.uint32(dense_labels.shape[2]),
            *minimums,
            *maximums,
            encodings,
            offsets,
            sizes,
            np.uint32(offsets.size),
            output,
            np.uint64(output_size),
        ),
    )


def _launch_sparse_kernel(
    dense_labels,
    minimums,
    maximums,
    encodings,
    offsets,
    output,
    *,
    cupy: ModuleType,
) -> None:
    if not int(output.size) or not int(dense_labels.size):
        return
    object_count = int(encodings.size)
    counters = cupy.zeros(object_count, dtype=cupy.uint32)
    kernel = _sparse_pack_kernel(cupy)
    threads = 256
    blocks = (int(dense_labels.size) + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            cupy.ascontiguousarray(dense_labels, dtype=cupy.int32),
            np.uint64(dense_labels.size),
            np.uint32(dense_labels.shape[1]),
            np.uint32(dense_labels.shape[2]),
            *minimums,
            *maximums,
            encodings,
            offsets,
            counters,
            output,
        ),
    )


@cache
def _mask_pack_kernel(cupy: ModuleType):
    source = r"""
extern "C" __global__
void vipp_pack_mesh_masks(
    const int* labels,
    const unsigned int depth,
    const unsigned int height,
    const unsigned int width,
    const unsigned int* min_z,
    const unsigned int* min_y,
    const unsigned int* min_x,
    const unsigned int* max_z,
    const unsigned int* max_y,
    const unsigned int* max_x,
    const unsigned long long* encodings,
    const unsigned long long* offsets,
    const unsigned long long* sizes,
    const unsigned int object_count,
    unsigned char* output,
    const unsigned long long output_size)
{
    const unsigned long long byte_index =
        (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (byte_index >= output_size) return;

    unsigned int low = 0U;
    unsigned int high = object_count;
    while (low < high) {
        const unsigned int mid = low + (high - low) / 2U;
        const unsigned long long end = offsets[mid] + sizes[mid];
        if (byte_index < end) high = mid;
        else low = mid + 1U;
    }
    if (low >= object_count || sizes[low] == 0ULL || encodings[low] != 1ULL) return;

    const unsigned long long local_byte = byte_index - offsets[low];
    const unsigned int box_height = max_y[low] - min_y[low];
    const unsigned int box_width = max_x[low] - min_x[low];
    const unsigned long long box_plane =
        (unsigned long long)box_height * (unsigned long long)box_width;
    const unsigned long long box_size =
        (unsigned long long)(max_z[low] - min_z[low]) * box_plane;
    unsigned char packed = 0U;
    #pragma unroll
    for (unsigned int bit = 0U; bit < 8U; ++bit) {
        const unsigned long long local = local_byte * 8ULL + bit;
        if (local >= box_size) break;
        const unsigned int z = (unsigned int)(local / box_plane);
        const unsigned long long remainder = local - (unsigned long long)z * box_plane;
        const unsigned int y = (unsigned int)(remainder / box_width);
        const unsigned int x = (unsigned int)(
            remainder - (unsigned long long)y * box_width);
        const unsigned long long source =
            ((unsigned long long)(z + min_z[low]) * height + (y + min_y[low]))
                * width + (x + min_x[low]);
        if (labels[source] == (int)(low + 1U)) {
            packed = (unsigned char)(packed | (unsigned char)(1U << bit));
        }
    }
    output[byte_index] = packed;
}
"""
    return cupy.RawKernel(source, "vipp_pack_mesh_masks")


@cache
def _sparse_pack_kernel(cupy: ModuleType):
    source = r"""
extern "C" __global__
void vipp_pack_sparse_mesh_masks(
    const int* labels,
    const unsigned long long size,
    const unsigned int height,
    const unsigned int width,
    const unsigned int* min_z,
    const unsigned int* min_y,
    const unsigned int* min_x,
    const unsigned int* max_z,
    const unsigned int* max_y,
    const unsigned int* max_x,
    const unsigned long long* encodings,
    const unsigned long long* offsets,
    unsigned int* counters,
    unsigned char* output)
{
    const unsigned long long index =
        (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= size) return;
    const int dense_label = labels[index];
    if (dense_label <= 0) return;
    const unsigned int object = (unsigned int)(dense_label - 1);
    if (encodings[object] != 2ULL) return;

    const unsigned long long plane =
        (unsigned long long)height * (unsigned long long)width;
    const unsigned int z = (unsigned int)(index / plane);
    const unsigned long long remainder = index - (unsigned long long)z * plane;
    const unsigned int y = (unsigned int)(remainder / width);
    const unsigned int x =
        (unsigned int)(remainder - (unsigned long long)y * width);
    const unsigned int box_height = max_y[object] - min_y[object];
    const unsigned int box_width = max_x[object] - min_x[object];
    const unsigned int local =
        ((z - min_z[object]) * box_height + (y - min_y[object]))
            * box_width + (x - min_x[object]);
    const unsigned int rank = atomicAdd(&counters[object], 1U);
    unsigned int* target =
        (unsigned int*)(output + offsets[object]);
    target[rank] = local;
}
"""
    return cupy.RawKernel(source, "vipp_pack_sparse_mesh_masks")


def _progress_initial(progress, total: int) -> None:
    if progress is None:
        return
    progress.check_cancelled()
    progress.report(0, total, "3D mesh morphology: preparing")


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
    return f"3D mesh morphology: {stage} (block {block_number}/{block_count})"


__all__ = ["measure_3d_mesh_morphology"]
