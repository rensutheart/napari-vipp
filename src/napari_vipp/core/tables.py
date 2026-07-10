"""Small table model for non-image VIPP outputs."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TableData:
    """Column-oriented result table with stable row order."""

    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]
    name: str = ""
    table_kind: str = "table"
    source_name: str = ""
    column_units: tuple[tuple[str, str], ...] = ()

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

    def to_dict(self) -> dict[str, object]:
        return {
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


def table_from_columns(
    columns: Mapping[str, Sequence[Any] | np.ndarray],
    *,
    name: str = "",
    table_kind: str = "table",
    source_name: str = "",
    column_units: Mapping[str, str] | None = None,
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
    )


def table_state_from_data(
    table: TableData,
    *,
    history: tuple[str, ...] = (),
    metadata_source: str = "VIPP table",
    source_name: str = "",
) -> TableState:
    """Create carried table metadata from a table output."""
    return TableState(
        row_count=table.row_count,
        column_count=table.column_count,
        columns=table.columns,
        table_kind=table.table_kind,
        metadata_source=metadata_source,
        source_name=source_name or table.source_name,
        history=history,
        column_units=table.column_units,
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
