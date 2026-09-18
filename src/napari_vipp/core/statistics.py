"""Versioned, Qt-free descriptive statistics for immutable measurement tables.

Version 2 describes objects, image means, or sample means, never independent
biological replication inferred from row counts. Quantiles use linear
interpolation (NumPy's default definition); standard deviation uses ddof=1.
No inference, uncertainty intervals, implicit type coercion, or sampling occurs.
"""

from __future__ import annotations

import json
import math
import statistics as stdlib_statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from fractions import Fraction
from numbers import Integral, Real

import numpy as np

from napari_vipp.core.progress import ProgressContext
from napari_vipp.core.tables import TableData

DEFAULT_STATISTICS = "count,mean,std"
STATISTICS = (
    "count",
    "mean",
    "median",
    "std",
    "q25",
    "q75",
    "iqr",
    "min",
    "max",
    "sum",
)
AUTO_GROUP_COLUMNS = (
    "condition",
    "treatment",
    "group",
    "replicate",
    "batch",
    "plate",
    "well",
    "field",
    "timepoint",
    "time_point",
    "t_index",
    "time_index",
    "c_index",
    "channel_index",
    "z_index",
)
_METADATA_COLUMNS = frozenset(AUTO_GROUP_COLUMNS) | {
    "id",
    "label",
    "index",
    "object",
    "image",
    "sample",
    "frame",
    "row",
    "source",
    "source_name",
    "source_file",
    "filename",
    "file",
    "path",
    "series",
    "channel",
    "experiment",
    "dataset",
    "subject",
    "donor",
    "animal",
    "patient",
    "slide",
    "tile",
    "position",
    "category",
    "class",
    "summary_version",
    "summary_level",
    "sample_weighting",
    "group_by",
    "image_column",
    "sample_column",
    "missing_policy",
    "statistics_recipe",
}
_PROVENANCE_COLUMNS = (
    "summary_version",
    "summary_level",
    "sample_weighting",
    "group_by",
    "image_column",
    "sample_column",
    "missing_policy",
    "statistics_recipe",
    "row_count",
)
_COUNT_SUFFIXES = (
    "object_total",
    "object_valid",
    "object_excluded",
    "missing_count",
    "nonfinite_count",
    "nonnumeric_count",
    "n",
)


@dataclass(frozen=True)
class StatisticsRecipe:
    """Persistable author intent; identity choices may be incomplete while editing."""

    summary_version: int = 2
    group_by: str = ""
    value_columns: str = "auto"
    statistics: str = DEFAULT_STATISTICS
    summary_level: str = "Objects"
    image_column: str = ""
    sample_column: str = ""
    sample_weighting: str = "Equal images"
    missing_policy: str = "Exclude and report"

    def __post_init__(self) -> None:
        _validate_version(self.summary_version)
        if self.summary_version == 1:
            return
        for field in fields(self):
            if field.name != "summary_version" and not isinstance(
                getattr(self, field.name), str
            ):
                raise ValueError(f"Statistics {field.name} must be text.")
        choices = {
            "summary_level": ("Objects", "Image averages", "Sample averages"),
            "sample_weighting": ("Equal images", "Equal objects"),
            "missing_policy": ("Exclude and report", "Stop and review"),
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(
                    f"Invalid Statistics {name}: choose {', '.join(allowed)}."
                )
        _column_list(self.group_by)
        _column_list(self.value_columns)
        _statistic_names(self.statistics)

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> StatisticsRecipe:
        names = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in params.items() if key in names})

    def to_params(self) -> dict[str, object]:
        return asdict(self)


def _validate_version(version: object) -> None:
    if type(version) is not int or version not in (1, 2):
        raise ValueError(
            "Unsupported Statistics summary_version; choose integer 1 or 2."
        )


def _column_list(text: str) -> tuple[str, ...]:
    """Comma-separated names or a JSON list for comma/reserved-word names."""
    cleaned = text.strip()
    if not cleaned or cleaned.lower() == "auto":
        return ()
    if cleaned.startswith("["):
        try:
            parts = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Statistics columns must be CSV names or a JSON list."
            ) from exc
        if not isinstance(parts, list) or any(
            not isinstance(part, str) or not part.strip() for part in parts
        ):
            raise ValueError(
                "Statistics column lists must contain nonempty text names."
            )
    else:
        parts = [part.strip() for part in cleaned.split(",")]
        if any(not part for part in parts):
            raise ValueError("Statistics column lists cannot contain empty names.")
    if len(set(parts)) != len(parts):
        raise ValueError("Statistics column lists cannot contain duplicate names.")
    return tuple(parts)


def _statistic_names(text: str) -> tuple[str, ...]:
    aliases = {
        "n": "count",
        "average": "mean",
        "avg": "mean",
        "sd": "std",
        "stdev": "std",
        "standard_deviation": "std",
        "standard deviation": "std",
        "minimum": "min",
        "maximum": "max",
        "total": "sum",
        "p25": "q25",
        "25%": "q25",
        "p75": "q75",
        "75%": "q75",
        "interquartile range": "iqr",
        "interquartile_range": "iqr",
    }
    result = []
    for part in text.replace(";", ",").replace("\n", ",").split(","):
        name = part.strip().lower().replace("-", "_")
        if not name:
            continue
        name = aliases.get(name, name)
        if name not in STATISTICS:
            raise ValueError(f"Unsupported Statistics statistic: {part.strip()!r}.")
        if name not in result:
            result.append(name)
    if not result:
        raise ValueError("Choose at least one descriptive statistic.")
    return tuple(result)


def _is_measurement_name(column: str) -> bool:
    name = column.lower()
    return not (
        name in _METADATA_COLUMNS
        or name.endswith(("_id", "_index", "_label", "_name", "_path", "_file"))
        or name.startswith(("vipp_", "_vipp_"))
    )


def measurement_columns(table: TableData) -> tuple[str, ...]:
    """Auto-select real numeric fields, excluding known IDs and grouping metadata.

    NaN/Inf remain recognizable as numeric, so an all-nonfinite selected field
    is reported. Un-typed empty/all-null fields require explicit selection.
    Numeric-looking strings and booleans are never converted into measurements.
    """
    _validate_table(table)
    return tuple(
        column
        for index, column in enumerate(table.columns)
        if _is_measurement_name(column)
        and any(
            isinstance(row[index], Real)
            and not isinstance(row[index], (bool, np.bool_))
            for row in table.rows
        )
    )


def _validate_table(table: TableData) -> None:
    if not isinstance(table, TableData):
        raise TypeError("Statistics expects a VIPP TableData input.")
    if any(not isinstance(name, str) or not name for name in table.columns):
        raise ValueError("Statistics requires nonempty text column names.")
    if len(set(table.columns)) != len(table.columns):
        raise ValueError("Statistics requires unique input column names.")
    if any(len(row) != len(table.columns) for row in table.rows):
        raise ValueError("Statistics input rows do not match the table columns.")


def _identity(value: object, column: str, row: int, *, group: bool = False) -> tuple:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(
            f"Statistics: missing identity/group {column!r} at row {row + 1}."
        )
    if isinstance(value, bool) and not group:
        raise ValueError(f"Statistics identity {column!r} cannot contain booleans.")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Statistics identity/group {column!r} must be finite.")
    if type(value) not in (str, int, float, bool):
        raise ValueError(
            f"Statistics identity/group {column!r} must be scalar text/numbers."
        )
    # A type tag prevents True/1/1.0 and numeric-looking text from merging.
    # In particular, integer IDs never travel through a float64 array.
    return type(value).__name__, value


def _numeric(value: object, column: str) -> tuple[float | None, str]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, "missing_count"
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None, "nonnumeric_count"
    try:
        converted = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(
            f"Statistics {column!r} has a value outside float64 range."
        ) from exc
    if isinstance(value, Integral):
        if not math.isfinite(converted) or int(converted) != int(value):
            raise ValueError(
                f"Statistics {column!r} contains an integer not exactly representable "
                "in float64; rescale explicitly before summarizing."
            )
    elif not math.isfinite(converted):
        if bool(np.isfinite(value)):
            raise ValueError(
                f"Statistics {column!r} has a value outside float64 range."
            )
        return None, "nonfinite_count"
    return converted, ""


def _finite(value: float, statistic: str) -> float:
    if not math.isfinite(value):
        raise ValueError(
            f"Statistics {statistic} exceeds float64 range; rescale explicitly."
        )
    return value


def _mean(values: Sequence[float]) -> float:
    # Exact-ratio intermediates avoid overflow and large-offset cancellation.
    return _finite(float(stdlib_statistics.mean(values)), "mean")


def _quantile(ordered: Sequence[float], fraction: float) -> float:
    position = (len(ordered) - 1) * fraction
    low = int(position)
    weight = position - low
    if not weight:
        return ordered[low]
    # Interpolate exact endpoint ratios before the final float conversion.
    # Multiplying endpoints by float weights first can underflow even when
    # both endpoints (and their quantile) are the same nonzero subnormal value.
    lower = Fraction.from_float(ordered[low])
    upper = Fraction.from_float(ordered[low + 1])
    exact_weight = Fraction.from_float(weight)
    return _finite(
        float(lower + exact_weight * (upper - lower)),
        "quantile",
    )


def _statistics(values: Sequence[float], names: tuple[str, ...]) -> dict[str, object]:
    count = len(values)
    result: dict[str, object] = {}
    ordered = (
        sorted(values)
        if any(n in names for n in ("median", "q25", "q75", "iqr"))
        else []
    )
    for name in names:
        if name == "count":
            result[name] = count
        elif not count or (name == "std" and count < 2):
            result[name] = None
        elif name == "mean":
            result[name] = _mean(values)
        elif name == "std":
            try:
                result[name] = _finite(
                    stdlib_statistics.stdev(values), "standard deviation"
                )
            except OverflowError as exc:
                raise ValueError(
                    "Statistics standard deviation exceeds float64 range; "
                    "rescale explicitly."
                ) from exc
        elif name == "min":
            result[name] = min(values)
        elif name == "max":
            result[name] = max(values)
        elif name == "sum":
            try:
                try:
                    total = math.fsum(values)
                except OverflowError:
                    # fsum can overflow before later cancellation yields a
                    # representable result. Exact ratios resolve that rare case.
                    total = float(sum(Fraction.from_float(value) for value in values))
                result[name] = _finite(total, "sum")
            except OverflowError as exc:
                raise ValueError(
                    "Statistics sum exceeds float64 range; rescale explicitly."
                ) from exc
        elif name == "iqr":
            result[name] = _finite(
                _quantile(ordered, 0.75) - _quantile(ordered, 0.25), "IQR"
            )
        else:
            result[name] = _quantile(
                ordered, {"median": 0.5, "q25": 0.25, "q75": 0.75}[name]
            )
    return result


def _observations(
    valid: list[tuple[int, float]],
    recipe: StatisticsRecipe,
    image_keys: Sequence[tuple],
    sample_keys: Sequence[tuple],
) -> list[float]:
    if recipe.summary_level == "Objects":
        return [value for _, value in valid]
    if recipe.summary_level == "Image averages":
        images: dict[tuple, list[float]] = {}
        for row, value in valid:
            images.setdefault(image_keys[row], []).append(value)
        return [_mean(values) for values in images.values()]
    if recipe.sample_weighting == "Equal objects":
        samples: dict[tuple, list[float]] = {}
        for row, value in valid:
            samples.setdefault(sample_keys[row], []).append(value)
        return [_mean(values) for values in samples.values()]
    nested: dict[tuple, dict[tuple, list[float]]] = {}
    for row, value in valid:
        nested.setdefault(sample_keys[row], {}).setdefault(image_keys[row], []).append(
            value
        )
    return [
        _mean([_mean(values) for values in images.values()])
        for images in nested.values()
    ]


def summarize_statistics(
    data: TableData,
    *,
    recipe: StatisticsRecipe | None = None,
    progress: ProgressContext | None = None,
    **params: object,
) -> TableData:
    """Describe complete selected populations with explicit experimental units.

    Equal images first averages valid objects in each image, then valid image
    means in each sample. Equal objects pools valid objects within each sample.
    Both modes describe the resulting sample means with equal sample weight.
    Missing values are counted independently for each measurement; identities
    must be complete and consistent even on rows whose measurement is invalid.
    """
    if recipe is not None and params:
        raise ValueError("Supply a Statistics recipe or flat parameters, not both.")
    recipe = recipe or StatisticsRecipe.from_params(params)
    if not isinstance(recipe, StatisticsRecipe):
        raise TypeError("Statistics recipe must be a StatisticsRecipe.")
    if progress is not None:
        progress.check_cancelled()
    if recipe.summary_version == 1:
        from napari_vipp.core.operations import summarize_measurements

        return summarize_measurements(data, **recipe.to_params(), progress=progress)
    table = data
    _validate_table(table)
    groups = (
        tuple(column for column in AUTO_GROUP_COLUMNS if column in table.columns)
        if recipe.group_by.strip().lower() == "auto"
        else _column_list(recipe.group_by)
    )
    identities = tuple(
        name for name in (recipe.image_column, recipe.sample_column) if name
    )
    if recipe.summary_level == "Image averages" and not recipe.image_column:
        raise ValueError("Choose an image identity column for Image averages.")
    if recipe.summary_level == "Sample averages":
        if not recipe.sample_column:
            raise ValueError("Choose a sample identity column for Sample averages.")
        if recipe.sample_weighting == "Equal images" and not recipe.image_column:
            raise ValueError(
                "Choose an image identity column for Equal images sample weighting."
            )
    values = (
        tuple(
            column
            for column in measurement_columns(table)
            if column not in (*groups, *identities)
        )
        if recipe.value_columns.strip().lower() == "auto"
        else _column_list(recipe.value_columns)
    )
    if not values:
        raise ValueError(
            "No numeric measurement columns selected; choose value columns explicitly."
        )
    missing = [
        column
        for column in (*groups, *identities, *values)
        if column not in table.columns
    ]
    if missing:
        raise ValueError(
            "Statistics could not find column(s): " + ", ".join(dict.fromkeys(missing))
        )
    if set(values) & set((*groups, *identities)):
        raise ValueError(
            "Statistics measurements cannot also be group or identity columns."
        )
    stats = _statistic_names(recipe.statistics)
    extra_counts = (("image_total", "image_valid") if recipe.image_column else ()) + (
        ("sample_total", "sample_valid") if recipe.sample_column else ()
    )
    output_columns = list(groups) + list(_PROVENANCE_COLUMNS)
    if recipe.image_column:
        output_columns.append("image_count")
    if recipe.sample_column:
        output_columns.append("sample_count")
    suffixes = _COUNT_SUFFIXES + extra_counts + stats + ("status",)
    for column in values:
        output_columns.extend(f"{column}_{suffix}" for suffix in suffixes)
    if len(set(output_columns)) != len(output_columns):
        raise ValueError(
            "Statistics output column names conflict; "
            "rename selected/group columns explicitly."
        )
    units = [
        (column, table.unit_for(column)) for column in groups if table.unit_for(column)
    ]
    for column in values:
        if table.unit_for(column):
            units.extend(
                (f"{column}_{stat}", table.unit_for(column))
                for stat in stats
                if stat != "count"
            )
    index = {name: position for position, name in enumerate(table.columns)}
    grouped: dict[tuple, list[int]] = {}
    raw_groups: dict[tuple, tuple] = {}
    image_keys: list[tuple] = []
    sample_keys: list[tuple] = []
    image_group: dict[tuple, tuple] = {}
    sample_group: dict[tuple, tuple] = {}
    image_sample: dict[tuple, tuple] = {}
    for row_number, row in enumerate(table.rows):
        if progress is not None and row_number % 1024 == 0:
            progress.report(
                row_number, table.row_count, "Checking Statistics identities"
            )
        raw = tuple(row[index[column]] for column in groups)
        key = tuple(
            _identity(value, column, row_number, group=True)
            for column, value in zip(groups, raw, strict=True)
        )
        grouped.setdefault(key, []).append(row_number)
        raw_groups.setdefault(
            key,
            tuple(
                value.item() if isinstance(value, np.generic) else value
                for value in raw
            ),
        )
        for column, keys, ownership, label, aggregate in (
            (
                recipe.image_column,
                image_keys,
                image_group,
                "Image",
                recipe.summary_level == "Image averages"
                or (
                    recipe.summary_level == "Sample averages"
                    and recipe.sample_weighting == "Equal images"
                ),
            ),
            (
                recipe.sample_column,
                sample_keys,
                sample_group,
                "Sample",
                recipe.summary_level == "Sample averages",
            ),
        ):
            if column:
                identity = _identity(row[index[column]], column, row_number)
                keys.append(identity)
                if aggregate and ownership.setdefault(identity, key) != key:
                    raise ValueError(
                        f"{label} identity {row[index[column]]!r} "
                        "spans multiple groups; "
                        "pairing is not supported."
                    )
        if recipe.image_column and recipe.sample_column:
            if (
                image_sample.setdefault(image_keys[-1], sample_keys[-1])
                != sample_keys[-1]
            ):
                raise ValueError(
                    "An image identity spans multiple samples; "
                    "provide globally unique image IDs."
                )
    if not groups and not table.rows:
        grouped[()] = []
        raw_groups[()] = ()
    recipe_json = json.dumps(
        {
            **recipe.to_params(),
            "resolved_group_columns": groups,
            "resolved_value_columns": values,
            "std_ddof": 1,
            "quantile_method": "linear",
        },
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    rows: list[tuple[object, ...]] = []
    for group_number, (key, row_indices) in enumerate(grouped.items()):
        output: list[object] = list(raw_groups[key]) + [
            2,
            recipe.summary_level,
            recipe.sample_weighting,
            json.dumps(groups, ensure_ascii=False),
            recipe.image_column,
            recipe.sample_column,
            recipe.missing_policy,
            recipe_json,
            len(row_indices),
        ]
        image_total = (
            len({image_keys[row] for row in row_indices}) if recipe.image_column else 0
        )
        sample_total = (
            len({sample_keys[row] for row in row_indices})
            if recipe.sample_column
            else 0
        )
        if recipe.image_column:
            output.append(image_total)
        if recipe.sample_column:
            output.append(sample_total)
        for column in values:
            if progress is not None:
                progress.report(group_number, len(grouped), f"Summarizing {column}")
            counts = dict.fromkeys(
                ("missing_count", "nonfinite_count", "nonnumeric_count"), 0
            )
            valid: list[tuple[int, float]] = []
            for number, row in enumerate(row_indices):
                if progress is not None and number % 1024 == 0:
                    progress.check_cancelled()
                value, issue = _numeric(table.rows[row][index[column]], column)
                if issue:
                    counts[issue] += 1
                else:
                    valid.append((row, value))
            excluded = len(row_indices) - len(valid)
            if excluded and recipe.missing_policy == "Stop and review":
                raise ValueError(
                    f"Statistics Stop and review: {column!r} has {excluded} "
                    f"excluded objects ({counts})."
                )
            observations = _observations(valid, recipe, image_keys, sample_keys)
            output.extend(
                (
                    len(row_indices),
                    len(valid),
                    excluded,
                    *counts.values(),
                    len(observations),
                )
            )
            if recipe.image_column:
                output.extend((image_total, len({image_keys[row] for row, _ in valid})))
            if recipe.sample_column:
                output.extend(
                    (sample_total, len({sample_keys[row] for row, _ in valid}))
                )
            output.extend(_statistics(observations, stats).values())
            status = []
            if excluded:
                status.append(
                    f"Excluded {excluded} objects: "
                    + ", ".join(
                        f"{name}={count}" for name, count in counts.items() if count
                    )
                )
            if not observations:
                status.append(
                    "No contributing observations; descriptive values are unavailable"
                )
            if "std" in stats and len(observations) < 2:
                status.append("Sample standard deviation undefined for n < 2 (ddof=1)")
            output.append("; ".join(status) or "ok")
        rows.append(tuple(output))
    if progress is not None:
        progress.report(len(grouped), len(grouped), "Statistics complete")
    return TableData(
        columns=tuple(output_columns),
        rows=tuple(rows),
        name="Statistics",
        table_kind="Descriptive Statistics v2",
        source_name=table.source_name,
        column_units=tuple(units),
    )
