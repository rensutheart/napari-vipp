"""Scientific contracts for basic object-measurement accelerators.

The authoritative CPU operations ultimately expose :class:`TableData`.  A
device implementation cannot construct that mixed-type host object while it
is inside a private accelerator allocation scope, so the GPU boundary uses one
strictly typed, two-dimensional ``float64`` matrix.  This module owns the
axis/layout contract and the only public conversion from that private matrix
to VIPP's stable table schema.

Only the basic morphology schemas are represented here.  Extended shape,
axis, boundary, ratio, and moment properties remain authoritative CPU regions
until a provider validates every advertised column.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from napari_vipp.core.connected_components import resolve_spatial_ndim
from napari_vipp.core.tables import TableData, table_from_columns

BASIC_MEASUREMENT_FLOAT_RTOL = 2.0e-7
BASIC_MEASUREMENT_FLOAT_ATOL = 1.0e-12
INTEGER_INTENSITY_REDUCTION_RTOL = 2.0e-13
INTEGER_INTENSITY_REDUCTION_ATOL = 2.0e-12
FLOAT32_INTENSITY_REDUCTION_RTOL = 5.0e-7
FLOAT32_INTENSITY_REDUCTION_ATOL = 2.0e-5

INTENSITY_COLUMNS = (
    "intensity_mean",
    "intensity_min",
    "intensity_max",
    "intensity_sum",
    "intensity_std",
)

MEASUREMENT_TABLE_PARITY_POLICY_ID = "basic-measurement-table-v1"
MEASUREMENT_TABLE_PARITY_OPERATION_IDS = frozenset(
    {"measure_objects", "measure_objects_intensity"}
)
MESH_MORPHOLOGY_TABLE_PARITY_POLICY_ID = "mesh-morphology-table-exact-v1"
MESH_MORPHOLOGY_TABLE_PARITY_OPERATION_IDS = frozenset({"measure_3d_mesh_morphology"})
SKELETON_MEASUREMENT_TABLE_PARITY_POLICY_ID = "skeleton-measurement-table-v1"
SKELETON_MEASUREMENT_TABLE_PARITY_OPERATION_IDS = frozenset({"analyze_skeleton"})


@dataclass(frozen=True, slots=True)
class MeasurementUnits:
    """Resolved basic-measurement calibration and ordered unit metadata."""

    size_column: str
    equivalent_diameter_column: str
    physical_size_column: str
    scale_product: float
    spatial_scales: tuple[float, ...]
    length_unit: str
    physical_unit: str
    calibrated: bool
    column_units: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class BasicMeasurementLayout:
    """Frozen axis, packed-matrix, and public-table layout for one call."""

    input_shape: tuple[int, ...]
    spatial_ndim: int
    spatial_axes: tuple[int, ...]
    leading_axes: tuple[int, ...]
    permutation: tuple[int, ...]
    leading_shape: tuple[int, ...]
    spatial_shape: tuple[int, ...]
    leading_axis_names: tuple[str, ...]
    spatial_axis_names: tuple[str, ...]
    units: MeasurementUnits
    include_intensity: bool
    packed_columns: tuple[str, ...]
    public_columns: tuple[str, ...]

    @property
    def packed_width(self) -> int:
        return len(self.packed_columns)

    @property
    def block_count(self) -> int:
        if not self.leading_shape:
            return 1
        return int(np.prod(self.leading_shape, dtype=np.int64))

    def packed_index(self, column: str) -> int:
        try:
            return self.packed_columns.index(column)
        except ValueError as exc:
            raise KeyError(column) from exc


def basic_measurement_layout(
    shape: Sequence[int],
    *,
    spatial_mode: str = "Auto from axes",
    resolved_spatial_ndim: int | None = None,
    axis_names: Sequence[str] | None = None,
    axis_types: Sequence[str] | None = None,
    axis_scales: Sequence[float | None] | None = None,
    axis_units: Sequence[str | None] | None = None,
    include_intensity: bool = False,
) -> BasicMeasurementLayout:
    """Resolve the exact basic measurement layout without inspecting values."""

    normalized_shape = _validated_shape(shape)
    ndim = len(normalized_shape)
    spatial_ndim = resolve_spatial_ndim(
        ndim,
        spatial_mode,
        resolved_spatial_ndim,
    )
    if spatial_ndim not in {2, 3}:
        raise ValueError(
            "Basic GPU measurements require a resolved 2D or 3D spatial rank."
        )

    normalized_axis_names = _measurement_axis_names(ndim, axis_names)
    normalized_axis_types = _measurement_axis_types(ndim, axis_types)
    spatial_axes = _measurement_spatial_axes(
        ndim,
        spatial_ndim,
        normalized_axis_names,
        normalized_axis_types,
    )
    leading_axes = tuple(index for index in range(ndim) if index not in spatial_axes)
    permutation = leading_axes + spatial_axes
    moved_axis_names = tuple(normalized_axis_names[index] for index in permutation)
    moved_scales = _reordered_axis_values(axis_scales, ndim, spatial_axes)
    moved_units = _reordered_axis_values(axis_units, ndim, spatial_axes)
    leading_axis_names = _safe_axis_column_names(
        moved_axis_names[:-spatial_ndim],
        tuple(f"axis_{index}" for index in range(ndim - spatial_ndim)),
    )
    spatial_axis_names = _safe_axis_column_names(
        moved_axis_names[-spatial_ndim:],
        ("z", "y", "x")[-spatial_ndim:],
    )
    units = measurement_units(
        spatial_ndim,
        moved_scales[-spatial_ndim:],
        moved_units[-spatial_ndim:],
        spatial_axis_names=spatial_axis_names,
        include_intensity=bool(include_intensity),
    )
    leading_shape = tuple(normalized_shape[index] for index in leading_axes)
    spatial_shape = tuple(normalized_shape[index] for index in spatial_axes)
    packed_columns = _packed_columns(
        leading_axis_names,
        spatial_axis_names,
        units,
        include_intensity=bool(include_intensity),
    )
    public_columns = _public_columns(
        leading_axis_names,
        spatial_axis_names,
        units,
        include_intensity=bool(include_intensity),
    )
    return BasicMeasurementLayout(
        input_shape=normalized_shape,
        spatial_ndim=spatial_ndim,
        spatial_axes=spatial_axes,
        leading_axes=leading_axes,
        permutation=permutation,
        leading_shape=leading_shape,
        spatial_shape=spatial_shape,
        leading_axis_names=leading_axis_names,
        spatial_axis_names=spatial_axis_names,
        units=units,
        include_intensity=bool(include_intensity),
        packed_columns=packed_columns,
        public_columns=public_columns,
    )


def validate_basic_measurement_options(
    *,
    include_shape_descriptors: bool = False,
    include_axis_descriptors: bool = False,
    include_2d_boundary_descriptors: bool = False,
    include_derived_shape_ratios: bool = False,
    include_2d_shape_moments: bool = False,
) -> None:
    """Reject every extended schema not yet implemented by the GPU provider."""

    options = {
        "include_shape_descriptors": include_shape_descriptors,
        "include_axis_descriptors": include_axis_descriptors,
        "include_2d_boundary_descriptors": include_2d_boundary_descriptors,
        "include_derived_shape_ratios": include_derived_shape_ratios,
        "include_2d_shape_moments": include_2d_shape_moments,
    }
    enabled = tuple(name for name, value in options.items() if bool(value))
    if enabled:
        joined = ", ".join(enabled)
        raise ValueError(
            "The basic GPU measurement provider requires all extended "
            f"measurement options to be disabled; enabled: {joined}."
        )


def finalize_basic_measurement_table(
    packed,
    *,
    layout: BasicMeasurementLayout,
    measurement_set: str | None = None,
    source_name: str = "",
) -> TableData:
    """Convert a validated packed host matrix to an exact typed ``TableData``.

    Integer-valued fields are checked before conversion so a malformed device
    result cannot be silently rounded.  Float-derived public fields use the
    same formulas and ordering as the authoritative CPU basic schema.
    """

    matrix = np.asarray(packed)
    if matrix.dtype != np.dtype(np.float64):
        raise TypeError(
            "Packed basic measurements must use native float64 storage; "
            f"received {matrix.dtype}."
        )
    if matrix.ndim != 2 or matrix.shape[1] != layout.packed_width:
        raise ValueError(
            "Packed basic measurements must have shape (rows, "
            f"{layout.packed_width}); received {matrix.shape}."
        )
    if not matrix.dtype.isnative:
        raise TypeError("Packed basic measurements must use native float64 storage.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Packed basic measurements must contain only finite values.")

    packed_columns = {
        name: matrix[:, index]
        for index, name in enumerate(layout.packed_columns)
    }
    integer_names = (
        *(f"{name}_index" for name in layout.leading_axis_names),
        "label_id",
        layout.units.size_column,
        *(f"bbox_{name}_min" for name in layout.spatial_axis_names),
        *(f"bbox_{name}_max" for name in layout.spatial_axis_names),
        "euler_number",
    )
    for name in integer_names:
        _validate_integer_column(packed_columns[name], name)
    _validate_packed_semantics(packed_columns, layout)

    row_count = int(matrix.shape[0])
    columns: dict[str, Sequence[object] | np.ndarray] = {}
    for name in layout.leading_axis_names:
        packed_name = f"{name}_index"
        columns[packed_name] = packed_columns[packed_name].astype(np.int64)
    columns["label_id"] = packed_columns["label_id"].astype(np.int64)
    sizes = packed_columns[layout.units.size_column].astype(np.int64)
    columns[layout.units.size_column] = sizes
    if layout.units.calibrated:
        columns[layout.units.physical_size_column] = [
            float(np.float64(value) * layout.units.scale_product)
            for value in sizes
        ]
        columns["physical_unit"] = [layout.units.physical_unit] * row_count

    for axis_name in layout.spatial_axis_names:
        columns[f"centroid_{axis_name}"] = packed_columns[f"centroid_{axis_name}"]
    for axis_name in layout.spatial_axis_names:
        columns[f"bbox_{axis_name}_min"] = packed_columns[
            f"bbox_{axis_name}_min"
        ].astype(np.int64)
    for axis_name in layout.spatial_axis_names:
        columns[f"bbox_{axis_name}_max"] = packed_columns[
            f"bbox_{axis_name}_max"
        ].astype(np.int64)

    discrete_equivalent = [
        _equivalent_diameter(np.float64(value), layout.spatial_ndim)
        for value in sizes
    ]
    columns[layout.units.equivalent_diameter_column] = discrete_equivalent
    if layout.units.calibrated:
        columns["equivalent_diameter_physical"] = [
            _equivalent_diameter(
                np.float64(value) * layout.units.scale_product,
                layout.spatial_ndim,
            )
            for value in sizes
        ]
        for axis_index, axis_name in enumerate(layout.spatial_axis_names):
            scale = layout.units.spatial_scales[axis_index]
            columns[f"centroid_{axis_name}_physical"] = [
                float(np.float64(value) * scale)
                for value in packed_columns[f"centroid_{axis_name}"]
            ]
            columns[f"bbox_{axis_name}_min_physical"] = [
                float(np.float64(value) * scale)
                for value in packed_columns[f"bbox_{axis_name}_min"]
            ]
            columns[f"bbox_{axis_name}_max_physical"] = [
                float(np.float64(value) * scale)
                for value in packed_columns[f"bbox_{axis_name}_max"]
            ]

    columns["extent"] = [
        _extent_for_row(packed_columns, layout, row_index)
        for row_index in range(row_count)
    ]
    columns["euler_number"] = packed_columns["euler_number"].astype(np.int64)
    if layout.include_intensity:
        for name in INTENSITY_COLUMNS:
            columns[name] = packed_columns[name]

    if tuple(columns) != layout.public_columns:
        raise RuntimeError("The basic measurement finalizer violated its schema.")
    base = str(
        measurement_set
        or (
            "Basic morphology + intensity"
            if layout.include_intensity
            else "Basic morphology"
        )
    )
    return table_from_columns(
        columns,
        name=(
            "Object intensity measurements"
            if layout.include_intensity
            else "Object measurements"
        ),
        table_kind=base,
        source_name=str(source_name),
        column_units=dict(layout.units.column_units),
    )


def finalize_basic_measurement_outputs(host_outputs, *, call):
    """Finalize the one-output resident provider ABI after device cleanup.

    Device execution deliberately sanitizes ``call.inputs`` before this hook
    runs.  The original label shape is therefore recovered from carried input
    state; the direct-input fallback exists only for isolated tests and other
    callers that have not sanitized an otherwise ordinary prepared call.
    """

    payloads = tuple(host_outputs)
    if len(payloads) != 1:
        raise ValueError(
            "Basic measurement host finalization requires exactly one packed "
            f"payload; received {len(payloads)}."
        )
    operation_id = str(getattr(call, "operation_id", "")).strip()
    operation_intensity = {
        "measure_objects": False,
        "measure_objects_intensity": True,
    }
    if operation_id not in operation_intensity:
        raise ValueError(
            "Basic measurement host finalization cannot handle operation "
            f"{operation_id!r}."
        )
    kwargs = dict(getattr(call, "kwargs", {}))
    validate_basic_measurement_options(
        include_shape_descriptors=kwargs.get("include_shape_descriptors", False),
        include_axis_descriptors=kwargs.get("include_axis_descriptors", False),
        include_2d_boundary_descriptors=kwargs.get(
            "include_2d_boundary_descriptors",
            False,
        ),
        include_derived_shape_ratios=kwargs.get(
            "include_derived_shape_ratios",
            False,
        ),
        include_2d_shape_moments=kwargs.get("include_2d_shape_moments", False),
    )
    shape = _prepared_measurement_input_shape(call)
    layout = basic_measurement_layout(
        shape,
        spatial_mode=kwargs.get("spatial_mode", "Auto from axes"),
        resolved_spatial_ndim=kwargs.get("resolved_spatial_ndim"),
        axis_names=kwargs.get("axis_names"),
        axis_types=kwargs.get("axis_types"),
        axis_scales=kwargs.get("axis_scales"),
        axis_units=kwargs.get("axis_units"),
        include_intensity=operation_intensity[operation_id],
    )
    return finalize_basic_measurement_table(
        payloads[0],
        layout=layout,
        measurement_set=kwargs.get("measurement_set"),
        source_name=str(kwargs.get("source_name", "")),
    )


def measurement_table_parity(
    reference: object,
    candidate: object,
    *,
    intensity_dtype: str | np.dtype | None = None,
    exact_float_columns: bool = False,
    default_float_tolerance: tuple[float, float] | None = None,
    float_tolerance_overrides: Mapping[str, tuple[float, float]] | None = None,
):
    """Compare complete public measurement tables under the v1 contract.

    The table boundary is intentionally stricter than a numeric array
    comparison: public metadata, column and row order, units, and the Python
    scalar type of every cell are part of the scientific result.  Integer and
    text fields compare exactly.  Floating fields first require identical
    finite/NaN/infinity masks and then use the named operation-owned bounds.

    ``intensity_dtype`` lets callers select the tighter integer-reduction
    profile.  When it is unavailable, the admitted finite-float32 profile is
    used conservatively for intensity reductions. ``exact_float_columns`` is
    reserved for hybrid providers whose public floats are authored by the same
    authoritative CPU algorithms after an exact accelerator preprocessing
    boundary.
    """

    # Keep this scientific module importable while compute policy is being
    # initialized; the generic result type is needed only when parity runs.
    from napari_vipp.core.compute_benchmark import ParityResult

    if not isinstance(reference, TableData) or not isinstance(candidate, TableData):
        return ParityResult(False, "measurement outputs must both be TableData")
    metadata_fields = (
        "columns",
        "name",
        "table_kind",
        "source_name",
        "column_units",
    )
    for field_name in metadata_fields:
        expected = getattr(reference, field_name)
        actual = getattr(candidate, field_name)
        if actual != expected:
            return ParityResult(
                False,
                f"table {field_name} differs: CPU {expected!r}, candidate {actual!r}",
            )
    if reference.row_count != candidate.row_count:
        return ParityResult(
            False,
            "table row count differs: "
            f"CPU {reference.row_count}, candidate {candidate.row_count}",
        )

    normalized_intensity_dtype = _parity_intensity_dtype(intensity_dtype)
    normalized_default_tolerance = _normalized_parity_tolerance(
        default_float_tolerance,
        name="default_float_tolerance",
    )
    normalized_overrides = {
        str(column_name): _normalized_parity_tolerance(
            tolerance,
            name=f"float_tolerance_overrides[{column_name!r}]",
        )
        for column_name, tolerance in dict(float_tolerance_overrides or {}).items()
    }
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    for row_index, (expected_row, actual_row) in enumerate(
        zip(reference.rows, candidate.rows, strict=True)
    ):
        if len(expected_row) != len(reference.columns) or len(actual_row) != len(
            candidate.columns
        ):
            return ParityResult(False, f"row {row_index} violates the table schema")
        for column_index, column_name in enumerate(reference.columns):
            expected = expected_row[column_index]
            actual = actual_row[column_index]
            if type(actual) is not type(expected):
                return ParityResult(
                    False,
                    f"Python scalar type differs at row {row_index}, column "
                    f"{column_name!r}: CPU {type(expected).__name__}, "
                    f"candidate {type(actual).__name__}",
                )
            if isinstance(expected, (int, str, bool)):
                if actual != expected:
                    return ParityResult(
                        False,
                        f"exact value differs at row {row_index}, column "
                        f"{column_name!r}: CPU {expected!r}, candidate {actual!r}",
                    )
                continue
            if not isinstance(expected, float):
                return ParityResult(
                    False,
                    f"unsupported measurement scalar type at row {row_index}, "
                    f"column {column_name!r}: {type(expected).__name__}",
                )
            masks_match, mask_detail = _float_masks_match(expected, actual)
            if not masks_match:
                return ParityResult(
                    False,
                    f"{mask_detail} at row {row_index}, column {column_name!r}",
                )
            if not np.isfinite(expected):
                continue
            if exact_float_columns:
                rtol, atol = (0.0, 0.0)
            elif column_name in normalized_overrides:
                rtol, atol = normalized_overrides[column_name]
            elif normalized_default_tolerance is not None:
                rtol, atol = normalized_default_tolerance
            else:
                rtol, atol = _measurement_column_tolerance(
                    column_name,
                    normalized_intensity_dtype,
                )
            absolute_error = abs(actual - expected)
            denominator = max(abs(expected), atol)
            relative_error = (
                absolute_error / denominator
                if denominator
                else (0.0 if absolute_error == 0.0 else float("inf"))
            )
            maximum_absolute_error = max(maximum_absolute_error, absolute_error)
            maximum_relative_error = max(maximum_relative_error, relative_error)
            if not bool(np.isclose(actual, expected, rtol=rtol, atol=atol)):
                return ParityResult(
                    False,
                    f"float value differs at row {row_index}, column "
                    f"{column_name!r}: CPU {expected:.17g}, candidate "
                    f"{actual:.17g}, abs={absolute_error:.9g}, rtol={rtol:.9g}, "
                    f"atol={atol:.9g}",
                )
    return ParityResult(
        True,
        "exact table schema/order/units/scalar types and exact integer/text "
        f"columns; max_float_abs={maximum_absolute_error:.9g}; "
        f"max_float_rel={maximum_relative_error:.9g}",
    )


def skeleton_measurement_table_parity(reference: object, candidate: object):
    """Compare Analyze Skeleton tables with an edge-count error bound.

    Every schema, metadata, scalar type, integer, text, and non-length float is
    exact. Pixel and calibrated physical lengths allow only the forward-error
    bound implied by summing the same positive float64 edge lengths in a
    different order on CPU and GPU. The bound scales with the largest authored
    component edge count instead of granting a broad fixed table tolerance.
    """

    from napari_vipp.core.compute_benchmark import ParityResult

    if not isinstance(reference, TableData) or not isinstance(candidate, TableData):
        return measurement_table_parity(reference, candidate)
    edge_column = "voxel_graph_edge_count"
    if edge_column not in reference.columns:
        return ParityResult(
            False,
            "Analyze Skeleton parity requires voxel_graph_edge_count.",
        )
    edge_index = reference.columns.index(edge_column)
    maximum_edges = 0
    for row_index, row in enumerate(reference.rows):
        if len(row) != len(reference.columns):
            return ParityResult(False, f"row {row_index} violates the table schema")
        value = row[edge_index]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return ParityResult(
                False,
                "Analyze Skeleton voxel_graph_edge_count must be a "
                "non-negative Python int.",
            )
        maximum_edges = max(maximum_edges, value)

    epsilon = float(np.finfo(np.float64).eps)
    guarded_additions = maximum_edges + 64
    denominator = 1.0 - guarded_additions * epsilon
    if denominator <= 0.0:
        return ParityResult(
            False,
            "Analyze Skeleton edge count exceeds the declared float64 "
            "summation parity domain.",
        )
    relative_bound = 2.0 * guarded_additions * epsilon / denominator
    absolute_bound = 16.0 * epsilon
    length_columns = {
        name
        for name in reference.columns
        if name
        in {
            "skeleton_length_pixels",
            "skeleton_length_voxels",
            "skeleton_length_physical",
        }
    }
    return measurement_table_parity(
        reference,
        candidate,
        default_float_tolerance=(0.0, 0.0),
        float_tolerance_overrides={
            name: (relative_bound, absolute_bound) for name in length_columns
        },
    )


def measurement_units(
    spatial_ndim: int,
    axis_scales: Sequence[float | None],
    axis_units: Sequence[str | None],
    *,
    spatial_axis_names: Sequence[str],
    include_intensity: bool,
) -> MeasurementUnits:
    """Return CPU-compatible basic column units in their stable order."""

    if spatial_ndim == 3:
        size_column = "volume_voxels"
        equivalent_column = "equivalent_diameter_voxels"
        physical_size_column = "volume_physical"
        default_physical_unit = "voxel^3"
        discrete_unit = "voxels"
        length_default = "voxel"
    elif spatial_ndim == 2:
        size_column = "area_pixels"
        equivalent_column = "equivalent_diameter_pixels"
        physical_size_column = "area_physical"
        default_physical_unit = "pixel^2"
        discrete_unit = "pixels"
        length_default = "pixel"
    else:
        raise ValueError("Basic measurement units require a 2D or 3D spatial rank.")

    scales = _normalized_spatial_scales(spatial_ndim, axis_scales)
    units = tuple(
        str(value).strip()
        for value in tuple(axis_units)[-spatial_ndim:]
        if value not in {None, ""}
    )
    calibrated = any(abs(scale - 1.0) > 1.0e-12 for scale in scales) or bool(
        units
    )
    scale_product = float(np.prod(scales)) if scales else 1.0
    length_unit = _physical_unit_label(units, 1, length_default)
    physical_unit = _physical_unit_label(
        units,
        spatial_ndim,
        default_physical_unit,
    )

    # Preserve the authoritative CPU dictionary insertion order.  Entries for
    # non-selected columns are omitted by ``table_from_columns``.
    ordered_units: dict[str, str] = {
        size_column: discrete_unit,
        equivalent_column: discrete_unit,
        _size_descriptor_column("bbox", spatial_ndim): discrete_unit,
        _size_descriptor_column("filled", spatial_ndim): discrete_unit,
        "intensity_mean": "intensity",
        "intensity_min": "intensity",
        "intensity_max": "intensity",
        "intensity_sum": "intensity",
        "intensity_std": "intensity",
    }
    if calibrated:
        ordered_units[physical_size_column] = physical_unit
        ordered_units["physical_unit"] = "text"
        ordered_units["equivalent_diameter_physical"] = length_unit
        for axis_name in spatial_axis_names:
            ordered_units[f"centroid_{axis_name}_physical"] = length_unit
            ordered_units[f"bbox_{axis_name}_min_physical"] = length_unit
            ordered_units[f"bbox_{axis_name}_max_physical"] = length_unit

    # Keep the argument meaningful and explicit even though absent intensity
    # keys are filtered only when the final TableData is built.
    del include_intensity
    return MeasurementUnits(
        size_column=size_column,
        equivalent_diameter_column=equivalent_column,
        physical_size_column=physical_size_column,
        scale_product=scale_product,
        spatial_scales=scales,
        length_unit=length_unit,
        physical_unit=physical_unit if calibrated else "",
        calibrated=calibrated,
        column_units=tuple(ordered_units.items()),
    )


def _packed_columns(
    leading_axis_names: tuple[str, ...],
    spatial_axis_names: tuple[str, ...],
    units: MeasurementUnits,
    *,
    include_intensity: bool,
) -> tuple[str, ...]:
    columns = [f"{name}_index" for name in leading_axis_names]
    columns.extend(("label_id", units.size_column))
    columns.extend(f"centroid_{name}" for name in spatial_axis_names)
    columns.extend(f"bbox_{name}_min" for name in spatial_axis_names)
    columns.extend(f"bbox_{name}_max" for name in spatial_axis_names)
    columns.append("euler_number")
    if include_intensity:
        columns.extend(INTENSITY_COLUMNS)
    return tuple(columns)


def _public_columns(
    leading_axis_names: tuple[str, ...],
    spatial_axis_names: tuple[str, ...],
    units: MeasurementUnits,
    *,
    include_intensity: bool,
) -> tuple[str, ...]:
    columns = [f"{name}_index" for name in leading_axis_names]
    columns.extend(("label_id", units.size_column))
    if units.calibrated:
        columns.extend((units.physical_size_column, "physical_unit"))
    columns.extend(f"centroid_{name}" for name in spatial_axis_names)
    columns.extend(f"bbox_{name}_min" for name in spatial_axis_names)
    columns.extend(f"bbox_{name}_max" for name in spatial_axis_names)
    columns.append(units.equivalent_diameter_column)
    if units.calibrated:
        columns.append("equivalent_diameter_physical")
        for name in spatial_axis_names:
            columns.extend(
                (
                    f"centroid_{name}_physical",
                    f"bbox_{name}_min_physical",
                    f"bbox_{name}_max_physical",
                )
            )
    columns.extend(("extent", "euler_number"))
    if include_intensity:
        columns.extend(INTENSITY_COLUMNS)
    return tuple(columns)


def _validated_shape(shape: Sequence[int]) -> tuple[int, ...]:
    result: list[int] = []
    for value in tuple(shape):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise ValueError("Measurement shapes require non-negative integers.")
        size = int(value)
        if size < 0:
            raise ValueError("Measurement shapes require non-negative integers.")
        result.append(size)
    return tuple(result)


def _prepared_measurement_input_shape(call) -> tuple[int, ...]:
    input_states = tuple(getattr(call, "input_states", ()))
    if input_states:
        state_shape = getattr(input_states[0], "shape", None)
        if state_shape is not None:
            return _validated_shape(state_shape)
    inputs = tuple(getattr(call, "inputs", ()))
    if inputs and inputs[0] is not None:
        input_shape = getattr(inputs[0], "shape", None)
        if input_shape is not None:
            return _validated_shape(input_shape)
    raise ValueError(
        "Basic measurement host finalization requires the original label "
        "shape in call.input_states[0].shape."
    )


def _measurement_axis_names(
    ndim: int,
    axis_names: Sequence[str] | None,
) -> tuple[str, ...]:
    if axis_names is not None and len(tuple(axis_names)) == ndim:
        return tuple(
            str(name).strip().lower() or f"axis_{index}"
            for index, name in enumerate(axis_names)
        )
    return tuple(f"axis_{index}" for index in range(ndim))


def _measurement_axis_types(
    ndim: int,
    axis_types: Sequence[str] | None,
) -> tuple[str, ...]:
    if axis_types is not None and len(tuple(axis_types)) == ndim:
        return tuple(str(value).strip().lower() for value in axis_types)
    return tuple("space" if index >= ndim - 2 else "unknown" for index in range(ndim))


def _measurement_spatial_axes(
    ndim: int,
    spatial_ndim: int,
    axis_names: tuple[str, ...],
    axis_types: tuple[str, ...],
) -> tuple[int, ...]:
    desired_names = ("z", "y", "x")[-spatial_ndim:]
    if all(axis_names.count(name) == 1 for name in desired_names):
        return tuple(axis_names.index(name) for name in desired_names)
    spatial = tuple(
        index for index, axis_type in enumerate(axis_types) if axis_type == "space"
    )
    if len(spatial) >= spatial_ndim:
        return spatial[-spatial_ndim:]
    return tuple(range(ndim - spatial_ndim, ndim))


def _reordered_axis_values(
    values: Sequence | None,
    ndim: int,
    spatial_axes: tuple[int, ...],
) -> tuple:
    if values is None or len(tuple(values)) != ndim:
        values = tuple(None for _ in range(ndim))
    values = tuple(values)
    leading = tuple(
        values[index] for index in range(ndim) if index not in spatial_axes
    )
    spatial = tuple(values[index] for index in spatial_axes)
    return leading + spatial


def _safe_axis_column_names(
    names: tuple[str, ...],
    fallback: tuple[str, ...],
) -> tuple[str, ...]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for index, name in enumerate(names):
        fallback_name = fallback[index] if index < len(fallback) else f"axis_{index}"
        candidate = _safe_column_fragment(name or fallback_name)
        if not candidate:
            candidate = _safe_column_fragment(fallback_name)
        if candidate in seen:
            candidate = f"{candidate}_{index}"
        seen.add(candidate)
        cleaned.append(candidate)
    return tuple(cleaned)


def _safe_column_fragment(value: str) -> str:
    text = str(value).strip().lower()
    chars = [character if character.isalnum() else "_" for character in text]
    return "_".join(part for part in "".join(chars).split("_") if part)


def _normalized_spatial_scales(
    spatial_ndim: int,
    axis_scales: Sequence[float | None],
) -> tuple[float, ...]:
    scales = [
        float(value) if value not in {None, ""} else 1.0
        for value in tuple(axis_scales)[-spatial_ndim:]
    ]
    if len(scales) < spatial_ndim:
        scales = [1.0] * (spatial_ndim - len(scales)) + scales
    return tuple(scales)


def _physical_unit_label(
    units: Sequence[str],
    spatial_ndim: int,
    default_unit: str,
) -> str:
    if not units:
        return default_unit
    if len(set(units)) == 1:
        unit = units[0]
        return unit if spatial_ndim == 1 else f"{unit}^{spatial_ndim}"
    return "*".join(units)


def _size_descriptor_column(prefix: str, spatial_ndim: int) -> str:
    if spatial_ndim >= 3:
        return f"{prefix}_volume_voxels"
    return f"{prefix}_area_pixels"


def _equivalent_diameter(area: np.float64, spatial_ndim: int) -> float:
    return float((2 * spatial_ndim * area / np.pi) ** (1 / spatial_ndim))


def _extent_for_row(
    packed_columns: dict[str, np.ndarray],
    layout: BasicMeasurementLayout,
    row_index: int,
) -> float:
    bbox_size = 1
    for axis_name in layout.spatial_axis_names:
        minimum = int(packed_columns[f"bbox_{axis_name}_min"][row_index])
        maximum = int(packed_columns[f"bbox_{axis_name}_max"][row_index])
        bbox_size *= maximum - minimum
    size = np.float64(packed_columns[layout.units.size_column][row_index])
    return float(size / bbox_size)


def _validate_integer_column(values: np.ndarray, name: str) -> None:
    if np.any(np.abs(values) > 2**53) or np.any(values != np.trunc(values)):
        raise ValueError(
            f"Packed measurement column {name!r} contains a non-exact integer."
        )


def _validate_packed_semantics(
    packed: dict[str, np.ndarray],
    layout: BasicMeasurementLayout,
) -> None:
    if np.any(packed["label_id"] <= 0):
        raise ValueError("Packed measurement label IDs must be positive.")
    if np.any(packed[layout.units.size_column] <= 0):
        raise ValueError("Packed measurement object sizes must be positive.")
    for axis_index, axis_name in enumerate(layout.leading_axis_names):
        values = packed[f"{axis_name}_index"]
        if np.any(values < 0) or np.any(values >= layout.leading_shape[axis_index]):
            raise ValueError(
                f"Packed leading index {axis_name!r} is outside its input axis."
            )
    for axis_index, axis_name in enumerate(layout.spatial_axis_names):
        minimums = packed[f"bbox_{axis_name}_min"]
        maximums = packed[f"bbox_{axis_name}_max"]
        if (
            np.any(minimums < 0)
            or np.any(maximums <= minimums)
            or np.any(maximums > layout.spatial_shape[axis_index])
        ):
            raise ValueError(
                f"Packed bounding boxes are invalid along axis {axis_name!r}."
            )


def _parity_intensity_dtype(value: str | np.dtype | None) -> np.dtype:
    if value is None:
        return np.dtype(np.float32)
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Measurement intensity parity requires bool, uint8, uint16, or "
            "float32 dtype metadata."
        ) from exc
    if dtype not in {
        np.dtype(bool),
        np.dtype(np.uint8),
        np.dtype(np.uint16),
        np.dtype(np.float32),
    }:
        raise ValueError(
            "Measurement intensity parity requires bool, uint8, uint16, or "
            f"float32 dtype metadata; received {dtype}."
        )
    return dtype


def _measurement_column_tolerance(
    column_name: str,
    intensity_dtype: np.dtype,
) -> tuple[float, float]:
    if column_name not in INTENSITY_COLUMNS:
        return BASIC_MEASUREMENT_FLOAT_RTOL, BASIC_MEASUREMENT_FLOAT_ATOL
    if column_name in {"intensity_min", "intensity_max"}:
        return 0.0, 0.0
    if intensity_dtype in {
        np.dtype(bool),
        np.dtype(np.uint8),
        np.dtype(np.uint16),
    }:
        if column_name == "intensity_sum":
            return 0.0, 0.0
        return (
            INTEGER_INTENSITY_REDUCTION_RTOL,
            INTEGER_INTENSITY_REDUCTION_ATOL,
        )
    return FLOAT32_INTENSITY_REDUCTION_RTOL, FLOAT32_INTENSITY_REDUCTION_ATOL


def _normalized_parity_tolerance(
    value: tuple[float, float] | None,
    *,
    name: str,
) -> tuple[float, float] | None:
    if value is None:
        return None
    try:
        rtol, atol = tuple(value)
        normalized = (float(rtol), float(atol))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite non-negative rtol/atol.") from exc
    if any(not np.isfinite(item) or item < 0.0 for item in normalized):
        raise ValueError(f"{name} must contain finite non-negative rtol/atol.")
    return normalized


def _float_masks_match(expected: float, actual: float) -> tuple[bool, str]:
    if bool(np.isfinite(expected)) != bool(np.isfinite(actual)):
        return False, "finite/non-finite masks differ"
    if bool(np.isnan(expected)) != bool(np.isnan(actual)):
        return False, "NaN masks differ"
    if bool(np.isposinf(expected)) != bool(np.isposinf(actual)):
        return False, "positive-infinity masks differ"
    if bool(np.isneginf(expected)) != bool(np.isneginf(actual)):
        return False, "negative-infinity masks differ"
    return True, ""


__all__ = [
    "BASIC_MEASUREMENT_FLOAT_ATOL",
    "BASIC_MEASUREMENT_FLOAT_RTOL",
    "FLOAT32_INTENSITY_REDUCTION_ATOL",
    "FLOAT32_INTENSITY_REDUCTION_RTOL",
    "INTEGER_INTENSITY_REDUCTION_ATOL",
    "INTEGER_INTENSITY_REDUCTION_RTOL",
    "INTENSITY_COLUMNS",
    "MEASUREMENT_TABLE_PARITY_OPERATION_IDS",
    "MEASUREMENT_TABLE_PARITY_POLICY_ID",
    "MESH_MORPHOLOGY_TABLE_PARITY_OPERATION_IDS",
    "MESH_MORPHOLOGY_TABLE_PARITY_POLICY_ID",
    "SKELETON_MEASUREMENT_TABLE_PARITY_OPERATION_IDS",
    "SKELETON_MEASUREMENT_TABLE_PARITY_POLICY_ID",
    "BasicMeasurementLayout",
    "MeasurementUnits",
    "basic_measurement_layout",
    "finalize_basic_measurement_table",
    "finalize_basic_measurement_outputs",
    "measurement_units",
    "measurement_table_parity",
    "skeleton_measurement_table_parity",
    "validate_basic_measurement_options",
]
