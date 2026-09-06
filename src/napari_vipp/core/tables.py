"""Small table model for non-image VIPP outputs."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HistogramSeriesMetadata:
    """Exact accounting and presentation identity for one histogram series."""

    series_index: int
    series_name: str
    series_color: str
    input_value_count: int
    finite_value_count: int
    nan_value_count: int
    positive_infinite_value_count: int
    negative_infinite_value_count: int
    binned_value_count: int
    underflow_count: int
    overflow_count: int
    nonpositive_excluded_count: int

    def __post_init__(self) -> None:
        if int(self.series_index) < 0:
            raise ValueError("Histogram series indices must be non-negative.")
        count_fields = (
            "input_value_count",
            "finite_value_count",
            "nan_value_count",
            "positive_infinite_value_count",
            "negative_infinite_value_count",
            "binned_value_count",
            "underflow_count",
            "overflow_count",
            "nonpositive_excluded_count",
        )
        if any(int(getattr(self, name)) < 0 for name in count_fields):
            raise ValueError("Histogram series counts must be non-negative.")
        nonfinite = (
            int(self.nan_value_count)
            + int(self.positive_infinite_value_count)
            + int(self.negative_infinite_value_count)
        )
        if int(self.input_value_count) != int(self.finite_value_count) + nonfinite:
            raise ValueError(
                "Histogram series input count must equal finite plus non-finite "
                "values."
            )
        classified_finite = (
            int(self.binned_value_count)
            + int(self.underflow_count)
            + int(self.overflow_count)
            + int(self.nonpositive_excluded_count)
        )
        if int(self.finite_value_count) != classified_finite:
            raise ValueError(
                "Histogram series finite count must equal all finite-value outcomes."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "series_index": int(self.series_index),
            "series_name": str(self.series_name),
            "series_color": str(self.series_color),
            "input_value_count": int(self.input_value_count),
            "finite_value_count": int(self.finite_value_count),
            "nan_value_count": int(self.nan_value_count),
            "positive_infinite_value_count": int(
                self.positive_infinite_value_count
            ),
            "negative_infinite_value_count": int(
                self.negative_infinite_value_count
            ),
            "binned_value_count": int(self.binned_value_count),
            "underflow_count": int(self.underflow_count),
            "overflow_count": int(self.overflow_count),
            "nonpositive_excluded_count": int(self.nonpositive_excluded_count),
        }


@dataclass(frozen=True)
class HistogramResultMetadata:
    """Exact population accounting carried by an intensity-histogram table.

    Histogram bins alone cannot say how many source values were excluded as
    non-finite, outside a custom range, or non-positive for logarithmic bins.
    Keeping those facts beside the table lets the inspector and detached plot
    describe the authored calculation without rescanning a possibly large
    source image.
    """

    input_value_count: int
    finite_value_count: int
    nan_value_count: int
    positive_infinite_value_count: int
    negative_infinite_value_count: int
    binned_value_count: int
    underflow_count: int
    overflow_count: int
    nonpositive_excluded_count: int
    effective_minimum: float | None
    effective_maximum: float | None
    bin_count: int
    bin_spacing: str
    series: tuple[HistogramSeriesMetadata, ...] = ()

    def __post_init__(self) -> None:
        count_fields = (
            "input_value_count",
            "finite_value_count",
            "nan_value_count",
            "positive_infinite_value_count",
            "negative_infinite_value_count",
            "binned_value_count",
            "underflow_count",
            "overflow_count",
            "nonpositive_excluded_count",
            "bin_count",
        )
        if any(int(getattr(self, name)) < 0 for name in count_fields):
            raise ValueError("Histogram result counts must be non-negative.")
        if int(self.bin_count) < 2:
            raise ValueError("Histogram result metadata requires at least two bins.")
        range_missing = (
            self.effective_minimum is None,
            self.effective_maximum is None,
        )
        if range_missing[0] != range_missing[1]:
            raise ValueError(
                "Histogram result range must have both bounds or neither bound."
            )
        if not any(range_missing):
            if not np.isfinite(self.effective_minimum) or not np.isfinite(
                self.effective_maximum
            ):
                raise ValueError("Histogram result range must be finite.")
            if float(self.effective_maximum) <= float(self.effective_minimum):
                raise ValueError("Histogram result maximum must exceed its minimum.")
        if self.bin_spacing not in {"Linear", "Logarithmic"}:
            raise ValueError(
                "Histogram result spacing must be 'Linear' or 'Logarithmic'."
            )
        nonfinite = (
            int(self.nan_value_count)
            + int(self.positive_infinite_value_count)
            + int(self.negative_infinite_value_count)
        )
        if int(self.input_value_count) != int(self.finite_value_count) + nonfinite:
            raise ValueError(
                "Histogram input count must equal finite plus non-finite values."
            )
        classified_finite = (
            int(self.binned_value_count)
            + int(self.underflow_count)
            + int(self.overflow_count)
            + int(self.nonpositive_excluded_count)
        )
        if int(self.finite_value_count) != classified_finite:
            raise ValueError(
                "Histogram finite count must equal all finite-value outcomes."
            )
        if self.series:
            expected_indices = tuple(range(len(self.series)))
            actual_indices = tuple(int(item.series_index) for item in self.series)
            if actual_indices != expected_indices:
                raise ValueError(
                    "Histogram series metadata must use contiguous zero-based indices."
                )
            aggregate_fields = (
                "input_value_count",
                "finite_value_count",
                "nan_value_count",
                "positive_infinite_value_count",
                "negative_infinite_value_count",
                "binned_value_count",
                "underflow_count",
                "overflow_count",
                "nonpositive_excluded_count",
            )
            for name in aggregate_fields:
                if int(getattr(self, name)) != sum(
                    int(getattr(item, name)) for item in self.series
                ):
                    raise ValueError(
                        f"Histogram aggregate {name} must equal its series totals."
                    )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible structural description."""

        data = {
            "input_value_count": int(self.input_value_count),
            "finite_value_count": int(self.finite_value_count),
            "nan_value_count": int(self.nan_value_count),
            "positive_infinite_value_count": int(
                self.positive_infinite_value_count
            ),
            "negative_infinite_value_count": int(
                self.negative_infinite_value_count
            ),
            "binned_value_count": int(self.binned_value_count),
            "underflow_count": int(self.underflow_count),
            "overflow_count": int(self.overflow_count),
            "nonpositive_excluded_count": int(self.nonpositive_excluded_count),
            "effective_minimum": (
                None
                if self.effective_minimum is None
                else float(self.effective_minimum)
            ),
            "effective_maximum": (
                None
                if self.effective_maximum is None
                else float(self.effective_maximum)
            ),
            "bin_count": int(self.bin_count),
            "bin_spacing": self.bin_spacing,
        }
        if self.series:
            data["series"] = [item.to_dict() for item in self.series]
        return data


@dataclass(frozen=True)
class TableData:
    """Column-oriented result table with stable row order."""

    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]
    name: str = ""
    table_kind: str = "table"
    source_name: str = ""
    column_units: tuple[tuple[str, str], ...] = ()
    histogram_metadata: HistogramResultMetadata | None = None

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def column_count(self) -> int:
        return len(self.columns)

    def unit_for(self, column: str) -> str:
        for name, unit in self.column_units:
            if name == column:
                return unit
        return ""

    def records(self, limit: int | None = None) -> list[dict[str, object]]:
        rows = self.rows if limit is None else self.rows[: max(int(limit), 0)]
        return [
            {column: row[index] for index, column in enumerate(self.columns)}
            for row in rows
        ]


@dataclass(frozen=True)
class TableState:
    """Metadata carried alongside a table output."""

    row_count: int
    column_count: int
    columns: tuple[str, ...]
    kind: str = "measurement table"
    table_kind: str = "object measurements"
    metadata_source: str = "VIPP table"
    source_name: str = ""
    history: tuple[str, ...] = ()
    column_units: tuple[tuple[str, str], ...] = ()
    numeric_value_count: int | None = None
    nan_value_count: int | None = None
    infinite_value_count: int | None = None
    missing_value_count: int | None = None
    nonfinite_row_count: int | None = None
    nonfinite_columns: tuple[str, ...] = ()
    histogram_metadata: HistogramResultMetadata | None = None

    def to_dict(self) -> dict[str, object]:
        data = {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": list(self.columns),
            "kind": self.kind,
            "table_kind": self.table_kind,
            "metadata_source": self.metadata_source,
            "source_name": self.source_name,
            "history": list(self.history),
            "column_units": dict(self.column_units),
        }
        if self.numeric_value_count is not None:
            data["numeric_value_count"] = self.numeric_value_count
            data["nan_value_count"] = self.nan_value_count
            data["infinite_value_count"] = self.infinite_value_count
            data["missing_value_count"] = self.missing_value_count
            data["nonfinite_row_count"] = self.nonfinite_row_count
            data["nonfinite_columns"] = list(self.nonfinite_columns)
        if self.histogram_metadata is not None:
            data["histogram_metadata"] = self.histogram_metadata.to_dict()
        return data


@dataclass(frozen=True)
class TableQuality:
    """Exact resident-value quality counts for a table output."""

    numeric_values: int
    nan_values: int
    infinite_values: int
    missing_values: int
    nonfinite_rows: int
    nonfinite_columns: tuple[str, ...]


def table_from_columns(
    columns: Mapping[str, Sequence[Any] | np.ndarray],
    *,
    name: str = "",
    table_kind: str = "table",
    source_name: str = "",
    column_units: Mapping[str, str] | None = None,
    histogram_metadata: HistogramResultMetadata | None = None,
) -> TableData:
    """Build a :class:`TableData` from equally sized column vectors."""
    names = tuple(str(name) for name in columns.keys())
    vectors = [_as_column_values(columns[name]) for name in columns]
    lengths = {len(vector) for vector in vectors}
    if len(lengths) > 1:
        raise ValueError("All table columns must have the same row count.")
    row_count = next(iter(lengths), 0)
    rows = tuple(
        tuple(
            _python_value(vectors[column_index][row_index])
            for column_index in range(len(names))
        )
        for row_index in range(row_count)
    )
    units = tuple(
        (str(column), str(unit))
        for column, unit in (column_units or {}).items()
        if column in names and str(unit)
    )
    return TableData(
        columns=names,
        rows=rows,
        name=name,
        table_kind=table_kind,
        source_name=source_name,
        column_units=units,
        histogram_metadata=histogram_metadata,
    )


def table_state_from_data(
    table: TableData,
    *,
    history: tuple[str, ...] = (),
    metadata_source: str = "VIPP table",
    source_name: str = "",
) -> TableState:
    """Create carried table metadata from a table output."""
    quality = table_quality_from_data(table)
    return TableState(
        row_count=table.row_count,
        column_count=table.column_count,
        columns=table.columns,
        table_kind=table.table_kind,
        metadata_source=metadata_source,
        source_name=source_name or table.source_name,
        history=history,
        column_units=table.column_units,
        numeric_value_count=quality.numeric_values,
        nan_value_count=quality.nan_values,
        infinite_value_count=quality.infinite_values,
        missing_value_count=quality.missing_values,
        nonfinite_row_count=quality.nonfinite_rows,
        nonfinite_columns=quality.nonfinite_columns,
        histogram_metadata=table.histogram_metadata,
    )


def table_quality_from_data(table: TableData) -> TableQuality:
    """Inspect a resident table once and return exact non-finite diagnostics."""

    numeric_values = 0
    missing_values = 0
    nan_values = 0
    infinite_values = 0
    affected_rows: set[int] = set()
    affected_fields: set[int] = set()

    for row_index, row in enumerate(table.rows):
        for field_index, value in enumerate(row):
            if value is None:
                missing_values += 1
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value,
                (int, float, complex, np.number),
            ):
                continue
            numeric_values += 1
            try:
                is_nan = bool(np.isnan(value))
                is_infinite = bool(np.isinf(value))
            except (TypeError, ValueError):
                continue
            if is_nan:
                nan_values += 1
            if is_infinite:
                infinite_values += 1
            if is_nan or is_infinite:
                affected_rows.add(row_index)
                affected_fields.add(field_index)

    return TableQuality(
        numeric_values=numeric_values,
        nan_values=nan_values,
        infinite_values=infinite_values,
        missing_values=missing_values,
        nonfinite_rows=len(affected_rows),
        nonfinite_columns=tuple(
            table.columns[index]
            for index in sorted(affected_fields)
            if index < len(table.columns)
        ),
    )


def is_table_data(value) -> bool:
    return isinstance(value, TableData)


def save_table_output(
    table: TableData,
    path: str | Path,
    *,
    format: str = "auto",
    overwrite: bool = True,
) -> Path:
    """Write a table output as CSV or TSV."""
    if not isinstance(table, TableData):
        raise TypeError("save_table_output expects a TableData object.")
    raw_path = str(path).strip()
    if not raw_path:
        raise ValueError("Save path cannot be blank.")
    target = Path(raw_path).expanduser()
    delimiter, suffix = _table_delimiter(format, target)
    if not target.suffix:
        target = target.with_suffix(suffix)
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerow(table.columns)
        writer.writerows(table.rows)
    return target


def _as_column_values(values) -> list[object]:
    arr = np.asarray(values)
    if arr.ndim == 0:
        return [_python_value(arr.item())]
    if arr.ndim > 1:
        return [_python_value(item) for item in arr.reshape(arr.shape[0], -1).tolist()]
    return [_python_value(item) for item in arr.tolist()]


def _python_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _table_delimiter(format: str, path: Path) -> tuple[str, str]:
    name = str(format).strip().lower()
    if name == "auto":
        suffix = path.suffix.lower()
        name = "tsv" if suffix == ".tsv" else "csv"
    if name == "tsv":
        return "\t", ".tsv"
    if name == "csv":
        return ",", ".csv"
    raise ValueError(f"Unsupported table format: {format!r}.")
