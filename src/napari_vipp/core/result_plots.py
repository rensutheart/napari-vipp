"""Immutable, Qt-free descriptive plots of measurement tables.

This module prepares exact values, shared histogram bins and empirical cumulative
distributions. It performs no hypothesis tests or implicit biological replication.
Only display marks may be sampled; all summaries and export rows use full data.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields, replace
from numbers import Real

import numpy as np

from napari_vipp.core.progress import OperationCancelled, ProgressContext
from napari_vipp.core.tables import TableData

DISPLAY_POINT_LIMIT = 10_000


@dataclass(frozen=True)
class PlotRecipe:
    """Versioned author intent; auto field choices are resolved in the result."""

    recipe_version: int = 1
    plot_type: str = "Compare groups"
    y_column: str = "auto"
    x_column: str = "auto"
    group_column: str = ""
    point_unit: str = "Objects"
    image_column: str = ""
    summary: str = "Mean"
    distribution: str = "Histogram"
    bins: int = 20
    normalization: str = "Count"
    log_x: bool = False
    log_y: bool = False
    title: str = ""
    point_size: float = 5.0
    show_grid: bool = True

    def __post_init__(self) -> None:
        if self.recipe_version != 1 or isinstance(self.recipe_version, bool):
            raise ValueError("Unsupported Plot Results recipe version.")
        choices = {
            "plot_type": ("Compare groups", "Distribution", "Scatter"),
            "point_unit": ("Objects", "Mean per image"),
            "summary": ("Mean", "Median", "None"),
            "distribution": ("Histogram", "Cumulative"),
            "normalization": ("Count", "Percent"),
        }
        for key, allowed in choices.items():
            if getattr(self, key) not in allowed:
                raise ValueError(f"Invalid {key}: choose one of {', '.join(allowed)}.")
        if (
            isinstance(self.bins, bool)
            or not isinstance(self.bins, int)
            or not 2 <= self.bins <= 512
        ):
            raise ValueError("Histogram bins must be an integer between 2 and 512.")
        if (
            not isinstance(self.point_size, Real)
            or isinstance(self.point_size, bool)
            or not math.isfinite(self.point_size)
            or not 1 <= self.point_size <= 20
        ):
            raise ValueError("Point size must be between 1 and 20.")
        for key in ("log_x", "log_y", "show_grid"):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"{key} must be true or false.")
        for key in ("y_column", "x_column", "group_column", "image_column", "title"):
            if not isinstance(getattr(self, key), str):
                raise ValueError(f"{key} must be text.")
        if self.point_unit == "Mean per image" and not self.image_column:
            raise ValueError(
                "Choose the image identity column before using Mean per image."
            )

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> PlotRecipe:
        names = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in params.items() if key in names})

    def to_params(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PlotCounts:
    input_rows: int
    eligible_rows: int
    missing_rows: int
    nonfinite_rows: int
    nonnumeric_rows: int
    nonpositive_rows: int
    plotted_points: int
    displayed_points: int
    nonpositive_points: int = 0

    @property
    def excluded_rows(self) -> int:
        return self.input_rows - self.eligible_rows

    @property
    def finite_rows(self) -> int:
        return self.eligible_rows + self.nonpositive_rows

    def to_dict(self) -> dict[str, int]:
        return {
            **asdict(self),
            "excluded_rows": self.excluded_rows,
            "finite_rows": self.finite_rows,
        }


@dataclass(frozen=True)
class PlotSeries:
    name: str
    x: tuple[float, ...]
    y: tuple[float, ...]
    source_rows: tuple[tuple[int, ...], ...]
    display_indices: tuple[int, ...]
    mean: float | None
    median: float | None
    bin_edges: tuple[float, ...] = ()
    histogram_values: tuple[float, ...] = ()
    ecdf_x: tuple[float, ...] = ()
    ecdf_y: tuple[float, ...] = ()


@dataclass(frozen=True)
class PlotData:
    recipe: PlotRecipe
    source_table: TableData
    plotted_table: TableData
    series: tuple[PlotSeries, ...]
    counts: PlotCounts
    x_label: str
    y_label: str
    warnings: tuple[str, ...] = ()

    @property
    def nbytes(self) -> int:
        """Conservative resident accounting, including retained source rows."""
        import sys

        seen: set[int] = set()

        def size(value: object) -> int:
            if id(value) in seen:
                return 0
            seen.add(id(value))
            total = sys.getsizeof(value)
            if isinstance(value, tuple):
                return total + sum(size(item) for item in value)
            if hasattr(value, "__dataclass_fields__"):
                return total + sum(
                    size(getattr(value, field.name)) for field in fields(value)
                )
            return total

        return size(self)


@dataclass(frozen=True)
class PlotState:
    plot_type: str
    point_count: int
    input_row_count: int
    excluded_row_count: int
    source_name: str = ""
    history: tuple[str, ...] = ()
    kind: str = "plot"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def is_plot_data(value: object) -> bool:
    return isinstance(value, PlotData)


def plot_state_from_data(data: PlotData, *, history: tuple[str, ...] = ()) -> PlotState:
    return PlotState(
        data.recipe.plot_type,
        data.counts.plotted_points,
        data.counts.input_rows,
        data.counts.excluded_rows,
        data.source_table.source_name,
        tuple(history),
    )


def numeric_columns(
    table: TableData, *, include_identifiers: bool = False
) -> tuple[str, ...]:
    """Numeric fields, without numeric-looking strings or auto-selected IDs."""
    result = []
    for index, column in enumerate(table.columns):
        name = column.lower()
        identifier = (
            name in {"label", "id", "index", "object", "image", "frame", "row"}
            or name.endswith(("_id", "_index", "_label"))
            or name.startswith(("vipp_", "_vipp_"))
        )
        if identifier and not include_identifiers:
            continue
        if any(
            isinstance(row[index], Real)
            and not isinstance(row[index], (bool, np.bool_))
            for row in table.rows
        ):
            result.append(column)
    return tuple(result)


def _scalar(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, tuple):
        return tuple(_scalar(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        "Plot Results requires immutable scalar table cells, "
        "not embedded arrays or objects."
    )


def _identity(value: object) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def _missing(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _mean(values: tuple[float, ...] | list[float]) -> float:
    # Scale before summation to avoid overflow for otherwise finite data.
    n = len(values)
    return math.fsum(value / n for value in values)


def _median(values: tuple[float, ...]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return _mean(ordered[middle - 1 : middle + 1])


def build_plot_result(
    table: TableData,
    *,
    recipe: PlotRecipe | None = None,
    cancel_callback: Callable[[], bool] | None = None,
    **params: object,
) -> PlotData:
    """Prepare exact descriptive data, preserving each original row identity.

    Exclusion reasons are mutually exclusive in this order: missing required
    cell, nonnumeric value, NaN/Inf. Per-image means use all finite paired rows
    in scatter mode, then give every image equal weight. Logarithmic eligibility
    is applied to those prepared means (or object points), never by removing
    negative contributors from an otherwise positive image mean. The separate
    nonpositive point count reports rejected means; nonpositive_rows counts the
    source rows contributing to those rejected points. No missing image is zero.
    """
    if not isinstance(table, TableData):
        raise TypeError("Plot Results expects a measurement table.")
    if recipe is not None and params:
        raise ValueError("Supply either a PlotRecipe or parameter values, not both.")
    recipe = recipe or PlotRecipe.from_params(params)

    def cancelled() -> None:
        if cancel_callback is not None and cancel_callback():
            raise OperationCancelled("Plot preparation cancelled.")

    cancelled()
    if len(set(table.columns)) != len(table.columns):
        raise ValueError("Plot Results requires unique table column names.")
    frozen_rows = []
    for index, row in enumerate(table.rows):
        if index % 1024 == 0:
            cancelled()
        if len(row) != len(table.columns):
            raise ValueError("A measurement row does not match its table columns.")
        frozen_rows.append(tuple(_scalar(value) for value in row))
    table = replace(
        table,
        columns=tuple(table.columns),
        rows=tuple(frozen_rows),
        column_units=tuple(tuple(item) for item in table.column_units),
    )
    candidates = numeric_columns(table)

    def resolve(column: str, *, exclude: str = "") -> str:
        if column in ("", "auto"):
            eligible = tuple(name for name in candidates if name != exclude)
            if not eligible:
                raise ValueError(
                    "Choose a numeric measurement column; "
                    "no suitable automatic field is available."
                )
            return eligible[0]
        if column not in table.columns:
            raise ValueError(
                f"Measurement column {column!r} is not present in the input table."
            )
        return column

    y_column = resolve(recipe.y_column)
    x_column = (
        resolve(recipe.x_column, exclude=y_column)
        if recipe.plot_type == "Scatter"
        else recipe.x_column
    )
    recipe = replace(recipe, y_column=y_column, x_column=x_column)
    for column in (recipe.group_column, recipe.image_column):
        if column and column not in table.columns:
            raise ValueError(f"Column {column!r} is not present in the input table.")
    y_index = table.columns.index(y_column)
    x_index = table.columns.index(x_column) if recipe.plot_type == "Scatter" else None
    group_index = (
        table.columns.index(recipe.group_column) if recipe.group_column else None
    )
    image_index = (
        table.columns.index(recipe.image_column) if recipe.image_column else None
    )
    groups: dict[
        tuple[str, str], tuple[str, list[tuple[object, float, float, tuple[int, ...]]]]
    ] = {}
    missing = nonnumeric = nonfinite = nonpositive = 0
    image_groups: dict[tuple[str, str], tuple[str, str]] = {}
    # Log X applies to the measured distribution axis, not categorical groups.
    y_log = recipe.log_x if recipe.plot_type == "Distribution" else recipe.log_y
    for row_index, row in enumerate(table.rows):
        if row_index % 1024 == 0:
            cancelled()
        group = row[group_index] if group_index is not None else "All objects"
        image = row[image_index] if image_index is not None else ""
        raw = [row[y_index]] + ([row[x_index]] if x_index is not None else [])
        required = (
            raw
            + ([group] if group_index is not None else [])
            + ([image] if recipe.point_unit == "Mean per image" else [])
        )
        if any(_missing(value) for value in required):
            missing += 1
            continue
        if any(not isinstance(value, Real) or isinstance(value, bool) for value in raw):
            nonnumeric += 1
            continue
        if any(isinstance(value, int) and abs(value) > 2**53 for value in raw):
            raise ValueError(
                "A selected integer exceeds exact floating-point plotting precision "
                "(2^53). Select another measurement or explicitly rescale "
                "the data first."
            )
        if any(not math.isfinite(value) for value in raw) or any(
            isinstance(value, float) and not math.isfinite(value) for value in required
        ):
            nonfinite += 1
            continue
        y = float(raw[0])
        x = float(raw[1]) if len(raw) == 2 else float(row_index)
        key = _identity(group)
        if recipe.point_unit == "Mean per image":
            image_key = _identity(image)
            if image_key in image_groups and image_groups[image_key] != key:
                raise ValueError(
                    f"Image {image!r} occurs in several groups. Choose a unique image "
                    "identity and a group that is constant within each image."
                )
            image_groups[image_key] = key
        if key not in groups:
            groups[key] = (str(group), [])
        groups[key][1].append((image, x, y, (row_index,)))
    eligible_count = table.row_count - missing - nonnumeric - nonfinite
    nonpositive_points = 0
    prepared = []
    names: set[str] = set()
    for key, (name, points) in groups.items():
        cancelled()
        if name in names:
            name = f"{name} ({key[0]})"
        names.add(name)
        if recipe.point_unit == "Mean per image":
            by_image: dict[tuple[str, str], list] = {}
            for point in points:
                by_image.setdefault(_identity(point[0]), []).append(point)
            points = [
                (
                    rows[0][0],
                    _mean([row[1] for row in rows]),
                    _mean([row[2] for row in rows]),
                    tuple(index for row in rows for index in row[3]),
                )
                for rows in by_image.values()
            ]
        visible_points = []
        for point in points:
            if (y_log and point[2] <= 0) or (
                x_index is not None and recipe.log_x and point[1] <= 0
            ):
                nonpositive_points += 1
                nonpositive += len(point[3])
            else:
                visible_points.append(point)
        if visible_points:
            prepared.append((name, visible_points))
    eligible_count -= nonpositive
    total_points = sum(len(points) for _, points in prepared)
    displayed = min(DISPLAY_POINT_LIMIT, total_points)
    display_global = (
        set(np.linspace(0, total_points - 1, displayed, dtype=int).tolist())
        if displayed
        else set()
    )
    all_y = [point[2] for _, points in prepared for point in points]
    bin_edges: tuple[float, ...] = ()
    if (
        recipe.plot_type == "Distribution"
        and recipe.distribution == "Histogram"
        and all_y
    ):
        low, high = min(all_y), max(all_y)
        if low == high:
            delta = abs(low) * 0.05 if low else 0.5
            low, high = low - delta, high + delta
        # Axis scale changes presentation, not the histogram binning policy.
        # Convex interpolation avoids overflow in high-low for finite extremes.
        weights = np.linspace(0, 1, recipe.bins + 1)
        edges = low * (1 - weights) + high * weights
        if not np.isfinite(edges).all() or not np.all(np.diff(edges) > 0):
            raise ValueError(
                "The selected range cannot form distinct finite histogram bins. "
                "Reduce the bin count or rescale the measurement."
            )
        bin_edges = tuple(float(value) for value in edges)
    series = []
    output_rows = []
    offset = 0
    identity_columns = tuple(
        (index, column)
        for index, column in enumerate(table.columns)
        if column.lower() in {"label", "id", "index", "image", "source", "batch"}
        or column.lower().endswith(("_id", "_index", "_label", "_name"))
        or column.lower().startswith(("vipp_", "_vipp_", "source_"))
        or column in {recipe.image_column, recipe.group_column}
    )
    for group_number, (name, points) in enumerate(prepared):
        cancelled()
        ys = tuple(point[2] for point in points)
        xs = (
            tuple(point[1] for point in points)
            if recipe.plot_type == "Scatter"
            else tuple(float(group_number) for _ in points)
        )
        ids = tuple(point[3] for point in points)
        histogram = ()
        ecdf_x = ecdf_y = ()
        if bin_edges:
            counts, _ = np.histogram(ys, bins=bin_edges)
            histogram = tuple(
                float(value)
                * (100.0 / len(ys) if recipe.normalization == "Percent" else 1.0)
                for value in counts
            )
        if recipe.plot_type == "Distribution" and recipe.distribution == "Cumulative":
            values, counts = np.unique(np.asarray(ys), return_counts=True)
            ecdf_x = tuple(float(value) for value in values)
            ecdf_y = tuple(float(value) / len(ys) * 100 for value in np.cumsum(counts))
        series.append(
            PlotSeries(
                name,
                xs,
                ys,
                ids,
                tuple(i for i in range(len(points)) if offset + i in display_global),
                _mean(ys),
                _median(ys),
                bin_edges,
                histogram,
                ecdf_x,
                ecdf_y,
            )
        )
        for i, point in enumerate(points):
            identity_values = []
            for column_index, _ in identity_columns:
                values = tuple(
                    dict.fromkeys(table.rows[j][column_index] for j in point[3])
                )
                identity_values.append(values[0] if len(values) == 1 else values)
            output_rows.append(
                (name, point[0], xs[i], ys[i], point[3], *identity_values)
            )
        offset += len(points)

    def label(column: str) -> str:
        unit = table.unit_for(column)
        return f"{column} ({unit})" if unit else column

    x_label = (
        label(x_column)
        if recipe.plot_type == "Scatter"
        else (recipe.group_column or "Objects")
    )
    y_label = label(y_column)
    if recipe.point_unit == "Mean per image":
        y_label = f"Mean per image: {y_label}"
        if recipe.plot_type == "Scatter":
            x_label = f"Mean per image: {x_label}"
    if recipe.plot_type == "Distribution":
        x_label = y_label
        y_label = (
            "Cumulative percent (%)"
            if recipe.distribution == "Cumulative"
            else (
                "Within-group percent (%)"
                if recipe.normalization == "Percent"
                else "Count"
            )
        )
    warnings = []
    if y_log or (recipe.plot_type == "Scatter" and recipe.log_x):
        warnings.append(
            "Image means use all finite values first; logarithmic axes then "
            "exclude nonpositive object points or image means. Summaries use "
            "the plotted points only. "
            "Histogram bins remain shared, linearly spaced measurement intervals."
        )
    if eligible_count != table.row_count:
        warnings.append(
            f"Excluded {table.row_count - eligible_count} rows: {missing} missing, "
            f"{nonnumeric} nonnumeric, {nonfinite} NaN/infinite, "
            f"{nonpositive} contributing to {nonpositive_points} nonpositive "
            "points on logarithmic axes."
        )
    if total_points > displayed:
        warnings.append(
            f"Display shows a deterministic sample of {displayed:,} of "
            f"{total_points:,} points. Summaries, bins, cumulative distributions "
            "and data exports use all eligible values."
        )
    if recipe.point_unit == "Mean per image":
        warnings.append(
            "Each image contributes one mean from eligible rows; images with no "
            "eligible rows are absent, not zero. An image is not automatically "
            "an independent biological sample."
        )
    else:
        warnings.append(
            "Each point represents one table row/object, not necessarily "
            "an independent biological sample."
        )
    if not total_points:
        warnings.append(
            "No eligible values for this plot. Check the selected fields and scale."
        )
    cancelled()
    return PlotData(
        recipe,
        table,
        TableData(
            (
                "group",
                "image",
                "x",
                "y",
                "source_rows",
                *(f"source:{column}" for _, column in identity_columns),
            ),
            tuple(output_rows),
            name="Plotted measurements",
            table_kind="plot values",
            source_name=table.source_name,
            column_units=(
                (
                    "x",
                    table.unit_for(x_column) if recipe.plot_type == "Scatter" else "",
                ),
                ("y", table.unit_for(y_column)),
            ),
        ),
        tuple(series),
        PlotCounts(
            table.row_count,
            eligible_count,
            missing,
            nonfinite,
            nonnumeric,
            nonpositive,
            total_points,
            displayed,
            nonpositive_points,
        ),
        x_label,
        y_label,
        tuple(warnings),
    )


def plot_results(
    table: TableData, *, progress: ProgressContext | None = None, **params: object
) -> PlotData:
    """Public operation entry point, available to headless execution."""
    return build_plot_result(
        table, cancel_callback=progress.is_cancelled if progress else None, **params
    )
