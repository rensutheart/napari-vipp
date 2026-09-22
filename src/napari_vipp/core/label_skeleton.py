"""Skeletonize and measure each original label without losing object identity.

Thinning uses scikit-image's Zhang (2D) or Lee (2D/3D) implementation on a
zero-padded bounding-box mask for each label. Graph measurements use the same
full-neighborhood components and diagonal-edge rules as Analyze Skeleton.
Spatial blocks are independent; disconnected fragments sharing a label remain
one object. Cropping changes neither foreground voxels nor boundary conditions.
Cancellation is cooperative between labels and around atomic library calls.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
from skimage import morphology

from .grid import _unit_dimension_and_factor
from .operations import (
    _analyze_skeleton_block,
    _measurement_axis_names,
    _safe_axis_column_names,
    _skeleton_units,
    _skeletonize_method,
    _spatial_mode_dimension,
    _validated_resolved_spatial_ndim,
)
from .tables import TableData, table_from_columns

_SCAN_CHUNK_SIZE = 1_048_576
_COUNT_COLUMNS = (
    "skeleton_voxel_count",
    "endpoint_voxel_count",
    "junction_voxel_count",
    "isolated_node_count",
    "branch_count",
    "graph_node_count",
    "graph_edge_count",
    "voxel_graph_edge_count",
    "cycle_count",
)


def _checkpoint(progress) -> None:
    if progress is not None:
        progress.check_cancelled()


def _report(progress, current, total, message) -> None:
    if progress is not None:
        progress.report(current, total, message)
        progress.check_cancelled()


def _labels(data, *, name="Labels") -> np.ndarray:
    array = np.asarray(data)
    if array.dtype.kind not in "ui":
        raise ValueError(
            f"{name} must contain nonnegative integer label IDs; "
            "label a binary mask first. Boolean and floating-point inputs "
            "are not label images."
        )
    if array.ndim < 2:
        raise ValueError(f"{name} require at least two spatial dimensions.")
    if array.dtype.kind == "i" and array.size and array.min() < 0:
        raise ValueError(f"{name} must contain nonnegative integer label IDs.")
    return array


def _spatial_layout(array, spatial_mode, resolved_spatial_ndim, axis_names, axis_types):
    for name, values in (("axis_names", axis_names), ("axis_types", axis_types)):
        if values is not None and len(values) != array.ndim:
            raise ValueError(f"{name} must have one entry per array axis.")
    names = _measurement_axis_names(array.ndim, axis_names)
    types = (
        tuple(str(value).strip().casefold() for value in axis_types)
        if axis_types is not None
        else None
    )
    spatial_ndim = _spatial_mode_dimension(spatial_mode)
    if spatial_ndim is None and resolved_spatial_ndim is not None:
        spatial_ndim = _validated_resolved_spatial_ndim(resolved_spatial_ndim)
    if spatial_ndim is None:
        if all(names.count(name) == 1 for name in ("z", "y", "x")):
            spatial_ndim = 3
        elif all(names.count(name) == 1 for name in ("y", "x")):
            spatial_ndim = 2
        elif types is not None and types.count("space") in {2, 3}:
            spatial_ndim = types.count("space")
        elif array.ndim == 2 and axis_names is None and axis_types is None:
            spatial_ndim = 2
        else:
            raise ValueError(
                "Auto from axes requires explicit spatial axis semantics. "
                "Supply axis names/types or select an explicit 2D/3D mode."
            )
    if spatial_ndim not in {2, 3} or spatial_ndim > array.ndim:
        raise ValueError("Label skeleton operations require 2D or 3D spatial blocks.")
    desired = ("z", "y", "x")[-spatial_ndim:]
    if all(names.count(name) == 1 for name in desired):
        axes = tuple(names.index(name) for name in desired)
        if types is not None and any(types[index] != "space" for index in axes):
            raise ValueError("Named spatial axes must have spatial axis types.")
    elif any(name in {"z", "y", "x"} for name in names):
        raise ValueError("Declared spatial axis names do not match the requested mode.")
    elif types is not None:
        candidates = tuple(i for i, value in enumerate(types) if value == "space")
        if len(candidates) < spatial_ndim:
            raise ValueError("Declared axes have too few spatial dimensions.")
        axes = candidates[-spatial_ndim:]
    else:
        axes = tuple(range(array.ndim - spatial_ndim, array.ndim))
    leading_names = _safe_axis_column_names(
        tuple(names[index] for index in range(array.ndim) if index not in axes),
        fallback=tuple(f"axis_{index}" for index in range(array.ndim - spatial_ndim)),
    )
    return axes, leading_names


def _move_spatial_last(array, spatial_axes):
    return np.moveaxis(
        array, spatial_axes, tuple(range(array.ndim - len(spatial_axes), array.ndim))
    )


def _label_bounds(block, progress) -> Iterator[tuple[int, tuple[slice, ...]]]:
    """Discover sparse/wide label bounds in one bounded-memory block scan.

    Never allocate by the largest label ID or compare every label against the
    full image. Chunked reductions also support noncontiguous spatial views.
    """
    bounds = {}
    for start in range(0, block.size, _SCAN_CHUNK_SIZE):
        _checkpoint(progress)
        values = block.flat[start : start + _SCAN_CHUNK_SIZE]
        foreground = np.flatnonzero(values)
        if not foreground.size:
            continue
        ids, inverse = np.unique(values[foreground], return_inverse=True)
        coordinates = np.unravel_index(foreground + start, block.shape)
        lower = np.full((ids.size, block.ndim), np.iinfo(np.intp).max, dtype=np.intp)
        upper = np.zeros((ids.size, block.ndim), dtype=np.intp)
        for axis, coords in enumerate(coordinates):
            np.minimum.at(lower[:, axis], inverse, coords)
            np.maximum.at(upper[:, axis], inverse, coords)
        for index, value in enumerate(ids):
            label_id = int(value)
            if label_id in bounds:
                old_lower, old_upper = bounds[label_id]
                np.minimum(old_lower, lower[index], out=old_lower)
                np.maximum(old_upper, upper[index], out=old_upper)
            else:
                bounds[label_id] = (lower[index].copy(), upper[index].copy())
        _checkpoint(progress)
    for label_id in sorted(bounds):
        lower, upper = bounds[label_id]
        yield (
            label_id,
            tuple(
                slice(int(low), int(high) + 1)
                for low, high in zip(lower, upper, strict=True)
            ),
        )


def _thin(mask, method, progress):
    _checkpoint(progress)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    skeleton = morphology.skeletonize(padded, method=method)
    _checkpoint(progress)
    return np.asarray(skeleton[(slice(1, -1),) * mask.ndim], dtype=bool)


def skeletonize_labels(
    data,
    spatial_mode: str = "Auto from axes",
    method: str = "Auto",
    resolved_spatial_ndim: int | None = None,
    axis_names: tuple[str, ...] | None = None,
    axis_types: tuple[str, ...] | None = None,
    progress=None,
) -> np.ndarray:
    """Thin each positive label independently, preserving IDs, dtype and axes.

    Zero is background. The output owns its buffer and is a voxel subset of
    the original label image. Lee thinning may erase particular small/even
    objects; Analyze Skeleton per Label retains those objects as empty rows.
    """
    _checkpoint(progress)
    labels = _labels(data)
    axes, _ = _spatial_layout(
        labels, spatial_mode, resolved_spatial_ndim, axis_names, axis_types
    )
    method_value = _skeletonize_method(method, spatial_ndim=len(axes))
    moved = _move_spatial_last(labels, axes)
    output = np.zeros(labels.shape, dtype=labels.dtype)
    moved_output = _move_spatial_last(output, axes)
    leading_shape = moved.shape[: labels.ndim - len(axes)]
    total = math.prod(leading_shape) + 1
    _report(progress, 0, total, "Skeletonize Labels: preparing objects")
    for completed, index in enumerate(np.ndindex(leading_shape), start=1):
        block = moved[index]
        destination = moved_output[index]
        for label_id, bounds in _label_bounds(block, progress):
            _checkpoint(progress)
            skeleton = _thin(block[bounds] == label_id, method_value, progress)
            destination[bounds][skeleton] = label_id
            _report(
                progress,
                completed - 1,
                total,
                f"Skeletonize Labels: skeletonized label {label_id}",
            )
        _report(
            progress, completed, total, "Skeletonize Labels: finished spatial block"
        )
    _report(progress, total, total, "Skeletonize Labels complete")
    return output


def _measurement_units(ndim, axes, axis_scales, axis_units):
    for name, values in (("axis_scales", axis_scales), ("axis_units", axis_units)):
        if values is not None and len(values) != ndim:
            raise ValueError(f"{name} must have one entry per array axis.")
    scales = (
        tuple(axis_scales[index] for index in axes) if axis_scales is not None else ()
    )
    if scales and any(
        value is None or not math.isfinite(float(value)) or float(value) <= 0
        for value in scales
    ):
        raise ValueError("Spatial voxel spacing must be finite and positive.")
    units = tuple(axis_units[index] for index in axes) if axis_units is not None else ()
    dimensions = [
        _unit_dimension_and_factor(
            "pixel" if str(value).strip().casefold() in {"voxel", "voxels"} else value
        )
        for value in units
    ]
    if not dimensions or all(dimension == "index" for dimension, _ in dimensions):
        return _skeleton_units(len(axes), (), ())
    if len({dimension for dimension, _ in dimensions}) != 1:
        raise ValueError("Spatial axes require complete, compatible length units.")
    if dimensions[0][0] != "length_micrometer":
        raise ValueError("Spatial axes require recognized physical length units.")
    if not scales:
        raise ValueError(
            "Physical length units require explicit spatial voxel spacing."
        )
    # Match the X (last resolved spatial axis) unit, including aliases and nm/um.
    target_factor = dimensions[-1][1]
    converted = tuple(
        float(scale) * factor / target_factor
        for scale, (_, factor) in zip(scales, dimensions, strict=True)
    )
    if not all(math.isfinite(value) and value > 0 for value in converted):
        raise ValueError("Converted spatial voxel spacing must be finite and positive.")
    return _skeleton_units(len(axes), converted, (str(units[-1]),) * len(axes))


def _validate_skeleton_subset(labels, skeleton, progress):
    if skeleton.shape != labels.shape:
        raise ValueError(
            "Labeled skeleton and original labels must have identical shapes."
        )
    for start in range(0, labels.size, _SCAN_CHUNK_SIZE):
        _checkpoint(progress)
        values = skeleton.flat[start : start + _SCAN_CHUNK_SIZE]
        source = labels.flat[start : start + _SCAN_CHUNK_SIZE]
        present = values != 0
        # Compare via Python integers when signed/unsigned promotion could round
        # wide integer IDs to float64 (NumPy's uint64/int64 common dtype).
        if np.result_type(values.dtype, source.dtype).kind == "f":
            agrees = np.array_equal(
                values[present].astype(object), source[present].astype(object)
            )
        else:
            agrees = np.array_equal(values[present], source[present])
        if not agrees:
            raise ValueError(
                "Labeled skeleton must be a subset of the original labels and "
                "preserve each original label ID at every nonzero voxel."
            )
    _checkpoint(progress)


def analyze_skeleton_per_label(
    labels,
    skeleton=None,
    spatial_mode: str = "Auto from axes",
    method: str = "Auto",
    resolved_spatial_ndim: int | None = None,
    axis_names: tuple[str, ...] | None = None,
    axis_types: tuple[str, ...] | None = None,
    axis_scales: tuple[float, ...] | None = None,
    axis_units: tuple[str | None, ...] | None = None,
    source_name: str = "",
    progress=None,
) -> tuple[TableData, TableData]:
    """Return per-object summaries and per-fragment skeleton measurements.

    Without ``skeleton``, original labels are thinned independently. A supplied
    skeleton must already be skeletonized, label-valued, on the identical grid,
    and an exact ID-preserving subset; ``method`` is then ignored. The pipeline
    verifies physical grid
    alignment; this array API verifies shape and voxel identity. Every original
    positive label appears once per spatial block, including empty skeletons.
    Physical lengths require explicit compatible spacing/units on every spatial
    axis; unknown pixel/voxel calibration produces discrete lengths only.
    """
    _checkpoint(progress)
    labels = _labels(labels)
    axes, leading_names = _spatial_layout(
        labels, spatial_mode, resolved_spatial_ndim, axis_names, axis_types
    )
    units = _measurement_units(labels.ndim, axes, axis_scales, axis_units)
    if skeleton is not None:
        skeleton = _labels(skeleton, name="Labeled skeleton")
        _validate_skeleton_subset(labels, skeleton, progress)
        moved_skeleton = _move_spatial_last(skeleton, axes)
    else:
        moved_skeleton = None
        method_value = _skeletonize_method(method, spatial_ndim=len(axes))
    moved = _move_spatial_last(labels, axes)
    leading_shape = moved.shape[: labels.ndim - len(axes)]
    index_columns = tuple(f"{name}_index" for name in leading_names)
    metrics = _COUNT_COLUMNS + (units.length_column,)
    if units.physical_column:
        metrics += (units.physical_column,)
    summary = {
        name: []
        for name in (
            *index_columns,
            "label_id",
            "skeleton_status",
            "skeleton_component_count",
            *metrics,
        )
    }
    components = {
        name: []
        for name in (
            *index_columns,
            "label_id",
            "component_id",
            "component_count_in_label",
            "component_voxel_fraction_in_label",
            *metrics,
        )
    }
    if units.physical_column:
        summary["physical_unit"] = []
        components["physical_unit"] = []
    total = math.prod(leading_shape) + 1
    _report(progress, 0, total, "Analyze Skeleton per Label: preparing objects")
    for completed, index in enumerate(np.ndindex(leading_shape), start=1):
        block = moved[index]
        for label_id, bounds in _label_bounds(block, progress):
            _checkpoint(progress)
            object_skeleton = (
                _thin(block[bounds] == label_id, method_value, progress)
                if moved_skeleton is None
                else moved_skeleton[index][bounds] == label_id
            )
            values = _analyze_skeleton_block(object_skeleton, units)
            _checkpoint(progress)
            count = len(values["component_id"])
            for name, value in zip(index_columns, index, strict=True):
                summary[name].append(int(value))
                components[name].extend([int(value)] * count)
            summary["label_id"].append(label_id)
            summary["skeleton_status"].append("ok" if count else "empty")
            summary["skeleton_component_count"].append(count)
            components["label_id"].extend([label_id] * count)
            components["component_id"].extend(values["component_id"])
            components["component_count_in_label"].extend([count] * count)
            components["component_voxel_fraction_in_label"].extend(
                values["component_voxel_fraction"]
            )
            for name in metrics:
                summary[name].append(sum(values[name]))
                components[name].extend(values[name])
            if units.physical_column:
                summary["physical_unit"].append(units.unit_label)
                components["physical_unit"].extend([units.unit_label] * count)
            _report(
                progress,
                completed - 1,
                total,
                f"Analyze Skeleton per Label: analyzed label {label_id}",
            )
        _report(
            progress, completed, total, "Analyze Skeleton per Label: finished block"
        )
    _report(progress, total, total, "Analyze Skeleton per Label complete")
    column_units = {
        **units.column_units,
        "label_id": "label",
        "component_id": "index",
        "skeleton_status": "text",
        "skeleton_component_count": "count",
        "component_count_in_label": "count",
        "component_voxel_fraction_in_label": "fraction",
    }
    # table_from_columns otherwise infers a float dtype for Python integer
    # lists combining small values and uint64 IDs above the signed range.
    summary["label_id"] = np.asarray(summary["label_id"], dtype=labels.dtype)
    components["label_id"] = np.asarray(components["label_id"], dtype=labels.dtype)
    return (
        table_from_columns(
            summary,
            name="Skeleton measurements per label",
            table_kind="Skeleton per-label summary",
            source_name=source_name,
            column_units=column_units,
        ),
        table_from_columns(
            components,
            name="Skeleton components per label",
            table_kind="Skeleton per-label components",
            source_name=source_name,
            column_units=column_units,
        ),
    )
