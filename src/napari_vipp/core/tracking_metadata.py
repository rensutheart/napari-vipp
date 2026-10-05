"""Immutable, strict JSON evidence for one uniformly sampled observation series.

No image/table dependencies are permitted here: carried states and table
artifacts share these schemas without importing the scientific linker.
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


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text.")
    return value


def _schema_fields(cls, data, *, version=True):
    if not isinstance(data, dict) or set(data) != {f.name for f in fields(cls)}:
        raise ValueError(f"{cls.__name__} fields do not match its schema.")
    if version and (
        type(data.get("schema_version")) is not int or data["schema_version"] != 1
    ):
        raise ValueError(f"Unsupported {cls.__name__} schema.")


def _json_value(value):
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class FramePopulation:
    """Exact accepted-before-cap and returned populations of one frame."""

    frame_index: int
    eligible_count: int
    retained_count: int
    truncated: bool = False

    def __post_init__(self):
        for name in ("frame_index", "eligible_count", "retained_count"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if self.retained_count > self.eligible_count:
            raise ValueError(
                "Retained observations cannot exceed eligible observations."
            )
        if type(self.truncated) is not bool or self.truncated != (
            self.retained_count < self.eligible_count
        ):
            raise ValueError("Frame truncation must agree with its population counts.")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        _schema_fields(cls, data, version=False)
        return cls(**data)


@dataclass(frozen=True)
class ObservationSeriesMetadata:
    """One source, fixed spatial lattice, explicit T, and source-index centers.

    IDs are local to each frame; they are not persistent track identities.
    Empty frames must have an explicit zero-count population entry. Unknown
    physical or time units remain null/empty rather than receiving a guess.
    """

    spatial_axes: tuple[str, ...]
    source_shape: tuple[int, ...]
    source_scale: tuple[float, ...]
    source_origin: tuple[float, ...]
    source_units: tuple[str | None, ...]
    source_frame: str
    source_revision: str
    frame_count: int
    time_scale: float
    time_origin: float
    time_unit: str | None
    coordinate_columns: tuple[str, ...]
    id_column: str
    frame_populations: tuple[FramePopulation, ...]
    frame_column: str = "t_index"
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported observation series metadata schema.")
        axes = tuple(self.spatial_axes)
        if axes not in (("y", "x"), ("z", "y", "x")):
            raise ValueError("Observation series require canonical spatial YX or ZYX.")
        object.__setattr__(self, "spatial_axes", axes)
        for suffix in ("shape", "scale", "origin", "units"):
            name = f"source_{suffix}"
            values = tuple(getattr(self, name))
            if len(values) != len(axes):
                raise ValueError(f"{name} must match the spatial rank.")
            if suffix == "shape":
                values = tuple(_integer(v, name, 1) for v in values)
            elif suffix in ("scale", "origin"):
                values = tuple(
                    _finite(v, name, positive=suffix == "scale") for v in values
                )
            elif any(v is not None and not isinstance(v, str) for v in values):
                raise ValueError("Source units must be text or null.")
            object.__setattr__(self, name, values)
        _text(self.source_frame, "source_frame")
        if (
            not isinstance(self.source_revision, str)
            or len(self.source_revision) != 64
            or any(c not in "0123456789abcdef" for c in self.source_revision)
        ):
            raise ValueError("source_revision must be a lowercase SHA-256 revision.")
        object.__setattr__(
            self, "frame_count", _integer(self.frame_count, "frame_count", 1)
        )
        object.__setattr__(
            self, "time_scale", _finite(self.time_scale, "time_scale", positive=True)
        )
        object.__setattr__(
            self, "time_origin", _finite(self.time_origin, "time_origin")
        )
        if self.time_unit is not None and not isinstance(self.time_unit, str):
            raise ValueError("time_unit must be text or null.")
        columns = tuple(self.coordinate_columns)
        if columns not in (
            tuple(f"{axis}_index" for axis in axes),
            tuple(f"centroid_{axis}" for axis in axes),
        ):
            raise ValueError(
                "Coordinates must be named source indices or label centroids."
            )
        object.__setattr__(self, "coordinate_columns", columns)
        if self.id_column not in ("detection_id", "label_id"):
            raise ValueError("Observation identity must be detection_id or label_id.")
        if self.frame_column != "t_index":
            raise ValueError("Observation frame column must be t_index.")
        populations = tuple(self.frame_populations)
        if len(populations) != self.frame_count or any(
            not isinstance(item, FramePopulation) or item.frame_index != index
            for index, item in enumerate(populations)
        ):
            raise ValueError(
                "Frame populations must cover every frame in order, "
                "including empty frames."
            )
        object.__setattr__(self, "frame_populations", populations)

    @property
    def truncated(self):
        return any(item.truncated for item in self.frame_populations)

    def to_dict(self):
        return _json_value(asdict(self))

    @classmethod
    def from_dict(cls, data):
        _schema_fields(cls, data)
        values = dict(data)
        if not isinstance(data["frame_populations"], (tuple, list)):
            raise ValueError("Frame populations must be an ordered array.")
        values["frame_populations"] = tuple(
            FramePopulation.from_dict(item) for item in data["frame_populations"]
        )
        return cls(**values)


ASSIGNMENT_POLICY = "adjacent-first maximum-cardinality minimum-total-distance"


@dataclass(frozen=True)
class TrackingMetadata:
    """Settings and exact backend identity accompanying tracks and summaries."""

    source_observations: ObservationSeriesMetadata
    maximum_displacement: float
    distance_units: str
    maximum_gap: int
    implementation: str
    implementation_version: str
    assignment_policy: str = ASSIGNMENT_POLICY
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported tracking metadata schema.")
        if not isinstance(self.source_observations, ObservationSeriesMetadata):
            raise ValueError("Tracking requires observation-series metadata.")
        if self.source_observations.truncated:
            raise ValueError("Tracking rejects truncated observation populations.")
        object.__setattr__(
            self,
            "maximum_displacement",
            _finite(self.maximum_displacement, "maximum_displacement", positive=True),
        )
        object.__setattr__(
            self, "maximum_gap", _integer(self.maximum_gap, "maximum_gap")
        )
        if self.distance_units not in ("Pixels", "Physical (micrometers)"):
            raise ValueError("Choose Pixels or Physical (micrometers) distance units.")
        if self.assignment_policy != ASSIGNMENT_POLICY:
            raise ValueError("Unsupported tracking assignment policy.")
        _text(self.implementation, "implementation")
        _text(self.implementation_version, "implementation_version")

    def to_dict(self):
        return _json_value(asdict(self))

    @classmethod
    def from_dict(cls, data):
        _schema_fields(cls, data)
        values = dict(data)
        values["source_observations"] = ObservationSeriesMetadata.from_dict(
            data["source_observations"]
        )
        return cls(**values)
