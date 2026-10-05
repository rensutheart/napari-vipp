"""Immutable, JSON-safe coordinate and population evidence for detections.

This module intentionally has no image/table dependencies: both carried image
states and ordinary result tables can persist these small schema values.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from numbers import Integral, Real


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _finite(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite numeric data.")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be finite{' and positive' if positive else ''}.")
    return result


def _grid(instance, prefix):
    axes = tuple(instance.axes)
    if axes not in (("y", "x"), ("z", "y", "x")):
        raise ValueError("Detection coordinates require canonical YX or ZYX axes.")
    object.__setattr__(instance, "axes", axes)
    for suffix in ("shape", "scale", "origin", "units"):
        name = f"{prefix}_{suffix}"
        values = tuple(getattr(instance, name))
        if len(values) != len(axes):
            raise ValueError(f"{name} must match the spatial rank.")
        if suffix == "shape":
            values = tuple(_integer(v, name, 1) for v in values)
        elif suffix in ("scale", "origin"):
            values = tuple(_finite(v, name, positive=suffix == "scale") for v in values)
        elif any(v is not None and not isinstance(v, str) for v in values):
            raise ValueError(f"{name} must contain unit strings or null.")
        object.__setattr__(instance, name, values)
    if not isinstance(instance.source_frame, str) or not instance.source_frame:
        raise ValueError("Detection metadata requires a nonempty source frame.")


def _digest(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 revision.")


class _Schema:
    def to_dict(self):
        def json_value(value):
            if isinstance(value, (tuple, list)):
                return [json_value(item) for item in value]
            return value

        return {
            "schema_version": 1,
            **{k: json_value(v) for k, v in asdict(self).items()},
        }

    @classmethod
    def from_dict(cls, data):
        if (
            not isinstance(data, dict)
            or type(data.get("schema_version")) is not int
            or data.get("schema_version") != 1
        ):
            raise ValueError("Unsupported detection metadata schema.")
        expected = {field.name for field in fields(cls)} | {"schema_version"}
        if set(data) != expected:
            raise ValueError("Detection metadata fields do not match its schema.")
        return cls(
            **{key: value for key, value in data.items() if key != "schema_version"}
        )


@dataclass(frozen=True)
class TemplateMatchMetadata(_Schema):
    """Exact source lattice, template footprint and paired output revisions."""

    axes: tuple[str, ...]
    search_shape: tuple[int, ...]
    template_shape: tuple[int, ...]
    search_scale: tuple[float, ...]
    search_origin: tuple[float, ...]
    search_units: tuple[str | None, ...]
    source_frame: str
    search_revision: str
    template_revision: str
    score_revision: str
    valid_mask_revision: str
    implementation: str

    def __post_init__(self):
        _grid(self, "search")
        shape = tuple(_integer(v, "template_shape", 1) for v in self.template_shape)
        if len(shape) != len(self.axes) or any(
            t > s for t, s in zip(shape, self.search_shape, strict=True)
        ):
            raise ValueError("Template shape must fit its search grid.")
        object.__setattr__(self, "template_shape", shape)
        for name in (
            "search_revision",
            "template_revision",
            "score_revision",
            "valid_mask_revision",
        ):
            _digest(getattr(self, name), name)
        if not isinstance(self.implementation, str) or not self.implementation:
            raise ValueError("Template matching requires implementation identity.")

    @property
    def center_offset(self):
        return tuple((size - 1) / 2 for size in self.template_shape)

    @property
    def score_shape(self):
        return tuple(
            s - t + 1
            for s, t in zip(self.search_shape, self.template_shape, strict=True)
        )


@dataclass(frozen=True)
class DetectionMetadata(_Schema):
    """Coordinate mapping and exact pre-cap detection accounting."""

    axes: tuple[str, ...]
    source_shape: tuple[int, ...]
    source_scale: tuple[float, ...]
    source_origin: tuple[float, ...]
    source_units: tuple[str | None, ...]
    source_frame: str
    coordinate_columns: tuple[str, ...]
    score_column: str
    template_shape: tuple[int, ...] | None
    minimum_value: float
    minimum_separation: float
    separation_units: str
    maximum_detections: int
    border_exclusion: int
    candidate_count: int
    accepted_count: int
    returned_count: int
    truncated: bool
    source_revision: str
    implementation: str

    def __post_init__(self):
        _grid(self, "source")
        columns = tuple(self.coordinate_columns)
        if columns != tuple(f"{axis}_index" for axis in self.axes):
            raise ValueError("Detection coordinate columns must name source indices.")
        object.__setattr__(self, "coordinate_columns", columns)
        if self.score_column != "score":
            raise ValueError("Detection score column must be 'score'.")
        if self.template_shape is not None:
            shape = tuple(_integer(v, "template_shape", 1) for v in self.template_shape)
            if len(shape) != len(self.axes) or any(
                t > s for t, s in zip(shape, self.source_shape, strict=True)
            ):
                raise ValueError("Detection template must fit the source grid.")
            object.__setattr__(self, "template_shape", shape)
        for name in ("minimum_value", "minimum_separation"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.minimum_separation < 0:
            raise ValueError("Minimum separation cannot be negative.")
        if self.separation_units not in ("Pixels", "Physical (micrometers)"):
            raise ValueError("Unsupported detection separation units.")
        for name in (
            "maximum_detections",
            "border_exclusion",
            "candidate_count",
            "accepted_count",
            "returned_count",
        ):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name), name, 1 if name == "maximum_detections" else 0
                ),
            )
        if not 0 <= self.returned_count <= self.accepted_count <= self.candidate_count:
            raise ValueError("Detection population counts are inconsistent.")
        if self.returned_count != min(self.accepted_count, self.maximum_detections):
            raise ValueError(
                "Returned detections must reflect the explicit result cap."
            )
        if not isinstance(self.truncated, bool) or self.truncated != (
            self.accepted_count > self.returned_count
        ):
            raise ValueError("Detection truncation flag disagrees with its counts.")
        _digest(self.source_revision, "source_revision")
        if not isinstance(self.implementation, str) or not self.implementation:
            raise ValueError("Detections require implementation identity.")
