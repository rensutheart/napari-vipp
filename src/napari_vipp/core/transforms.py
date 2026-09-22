"""Immutable, portable global spatial transforms, never backend pickles.

Matrices map moving physical coordinates to reference physical coordinates in
canonical NumPy spatial order (YX or ZYX). Length coordinates are micrometres;
uncalibrated coordinates are explicitly pixels. Array axes are never inferred.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from napari_vipp.core.atomic_io import atomic_write_json
from napari_vipp.core.grid import _unit_dimension_and_factor
from napari_vipp.core.metadata import AxisMetadata, ImageState, image_state_from_array

TRANSFORM_SCHEMA_VERSION = 1
TRANSFORM_DIRECTION = "moving-to-reference"


@dataclass(frozen=True)
class RegistrationGrid:
    """Canonical spatial lattice plus the acquisition coordinate-frame identity."""

    axes: tuple[str, ...]
    shape: tuple[int, ...]
    spacing: tuple[float, ...]
    origin: tuple[float, ...]
    unit: str
    frame_id: str

    def __post_init__(self):
        for name in ("axes", "shape", "spacing", "origin"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.axes not in (("y", "x"), ("z", "y", "x")):
            raise ValueError(
                "Registration grids require explicit YX or ZYX spatial axes."
            )
        rank = len(self.axes)
        if any(
            len(getattr(self, key)) != rank for key in ("shape", "spacing", "origin")
        ):
            raise ValueError(
                "Registration grid calibration rank does not match its axes."
            )
        if any(
            isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in self.shape
        ):
            raise ValueError("Registration grid extents must be positive integers.")
        if not np.isfinite(self.spacing).all() or any(v <= 0 for v in self.spacing):
            raise ValueError("Registration grid spacing must be finite and positive.")
        if not np.isfinite(self.origin).all():
            raise ValueError("Registration grid origins must be finite.")
        if self.unit not in ("micrometer", "pixel") or not self.frame_id:
            raise ValueError(
                "Registration grids need a coordinate unit and source frame identity."
            )

    @property
    def index_to_physical(self):
        result = np.eye(len(self.axes) + 1)
        result[:-1, :-1] = np.diag(self.spacing)
        result[:-1, -1] = self.origin
        return result

    def to_dict(self):
        return dict(
            axes=list(self.axes),
            shape=list(self.shape),
            spacing=list(self.spacing),
            origin=list(self.origin),
            unit=self.unit,
            frame_id=self.frame_id,
        )

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {
            "axes",
            "shape",
            "spacing",
            "origin",
            "unit",
            "frame_id",
        }:
            raise ValueError("Invalid registration grid fields.")
        return cls(**value)


@dataclass(frozen=True)
class TransformState:
    model: str
    spatial_axes: tuple[str, ...]
    transform_count: int
    coordinate_unit: str
    source_name: str = ""
    reference_name: str = ""
    history: tuple[str, ...] = ()
    kind: str = "spatial transform"
    metadata_source: str = "VIPP registration"

    def __post_init__(self):
        object.__setattr__(self, "spatial_axes", tuple(self.spatial_axes))
        object.__setattr__(self, "history", tuple(self.history))
        if self.model not in ("Translation", "Rigid", "Affine"):
            raise ValueError("Transform model must be Translation, Rigid or Affine.")
        if self.spatial_axes not in (("y", "x"), ("z", "y", "x")):
            raise ValueError("Transform summary requires YX or ZYX spatial axes.")
        if (
            isinstance(self.transform_count, bool)
            or not isinstance(self.transform_count, int)
            or self.transform_count < 1
            or self.coordinate_unit not in ("micrometer", "pixel")
        ):
            raise ValueError("Transform summary needs a positive count and valid unit.")

    def to_dict(self):
        return dict(
            model=self.model,
            spatial_axes=list(self.spatial_axes),
            transform_count=self.transform_count,
            coordinate_unit=self.coordinate_unit,
            source_name=self.source_name,
            reference_name=self.reference_name,
            history=list(self.history),
            kind=self.kind,
            metadata_source=self.metadata_source,
            direction=TRANSFORM_DIRECTION,
            schema_version=TRANSFORM_SCHEMA_VERSION,
        )


@dataclass(frozen=True)
class TransformData:
    """One matrix or one matrix per complete time volume; recursively immutable."""

    matrices: tuple[tuple[tuple[float, ...], ...], ...]
    moving_grid: RegistrationGrid
    reference_grid: RegistrationGrid
    state: TransformState
    time_axis: AxisMetadata | None = None
    reference_time: int | None = None
    settings: tuple[tuple[str, str | int | float | bool | None], ...] = ()
    implementation: str = ""
    schema_version: int = TRANSFORM_SCHEMA_VERSION
    direction: str = TRANSFORM_DIRECTION

    def __post_init__(self):
        if (
            isinstance(self.schema_version, bool)
            or self.schema_version != TRANSFORM_SCHEMA_VERSION
            or self.direction != TRANSFORM_DIRECTION
        ):
            raise ValueError("Unsupported transform schema or coordinate direction.")
        if not isinstance(self.moving_grid, RegistrationGrid) or not isinstance(
            self.reference_grid, RegistrationGrid
        ):
            raise TypeError("Transforms require typed moving and reference grids.")
        if (
            self.moving_grid.axes != self.reference_grid.axes
            or self.moving_grid.unit != self.reference_grid.unit
        ):
            raise ValueError(
                "Transform grid spatial axes and coordinate units must agree."
            )
        rank = len(self.moving_grid.axes)
        matrices = np.asarray(self.matrices, dtype=np.float64)
        if (
            matrices.ndim != 3
            or matrices.shape[1:] != (rank + 1, rank + 1)
            or not len(matrices)
        ):
            raise ValueError(
                "Transforms require one or more homogeneous spatial matrices."
            )
        if not np.isfinite(matrices).all():
            raise ValueError("Transform matrices must contain only finite values.")
        expected = np.zeros(rank + 1)
        expected[-1] = 1
        if not np.all(matrices[:, -1] == expected):
            raise ValueError("Invalid homogeneous matrix final row.")
        for matrix in matrices:
            linear = matrix[:-1, :-1]
            if np.linalg.det(linear) < 1e-12:
                raise ValueError(
                    "A registration transform must be invertible "
                    "and preserve orientation."
                )
            if self.state.model == "Translation" and not np.allclose(
                linear, np.eye(rank), rtol=0, atol=1e-10
            ):
                raise ValueError(
                    "A Translation transform cannot rotate, scale or shear."
                )
            if self.state.model == "Rigid" and not np.allclose(
                linear.T @ linear, np.eye(rank), rtol=0, atol=1e-6
            ):
                raise ValueError("A Rigid transform cannot scale or shear.")
        if self.time_axis is None:
            if len(matrices) != 1 or self.reference_time is not None:
                raise ValueError(
                    "A pairwise transform contains one matrix and no time index."
                )
        elif (
            self.time_axis.name.lower() != "t"
            or len(matrices) < 2
            or self.time_axis.type != "time"
            or not self.time_axis.is_explicit
            or isinstance(self.reference_time, bool)
            or not isinstance(self.reference_time, int)
            or not 0 <= self.reference_time < len(matrices)
        ):
            raise ValueError(
                "Transform series require explicit T calibration "
                "and a valid reference time."
            )
        if (
            self.state.transform_count != len(matrices)
            or self.state.spatial_axes != self.moving_grid.axes
            or self.state.coordinate_unit != self.moving_grid.unit
        ):
            raise ValueError(
                "Transform summary does not match its matrix/grid payload."
            )
        settings = tuple(tuple(item) for item in self.settings)
        if any(
            len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], (str, int, float, bool, type(None)))
            for item in settings
        ):
            raise ValueError(
                "Transform settings must be immutable scalar key/value pairs."
            )
        json.dumps(dict(settings), allow_nan=False)
        object.__setattr__(self, "settings", settings)
        object.__setattr__(
            self,
            "matrices",
            tuple(
                tuple(tuple(float(v) for v in row) for row in matrix)
                for matrix in matrices
            ),
        )

    @property
    def nbytes(self):
        return len(self.matrices) * (len(self.moving_grid.axes) + 1) ** 2 * 8

    @property
    def is_time_series(self):
        return self.time_axis is not None

    def to_dict(self):
        return dict(
            schema_version=self.schema_version,
            direction=self.direction,
            matrices=[[list(row) for row in matrix] for matrix in self.matrices],
            moving_grid=self.moving_grid.to_dict(),
            reference_grid=self.reference_grid.to_dict(),
            state=self.state.to_dict(),
            time_axis=self.time_axis.to_dict() if self.time_axis else None,
            reference_time=self.reference_time,
            settings=dict(self.settings),
            implementation=self.implementation,
        )

    @classmethod
    def from_dict(cls, value):
        fields = {
            "schema_version",
            "direction",
            "matrices",
            "moving_grid",
            "reference_grid",
            "state",
            "time_axis",
            "reference_time",
            "settings",
            "implementation",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("Invalid transform document fields.")
        state = dict(value["state"])
        if (
            state.pop("direction", None) != TRANSFORM_DIRECTION
            or state.pop("schema_version", None) != TRANSFORM_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported transform state version or direction.")
        return cls(
            matrices=value["matrices"],
            moving_grid=RegistrationGrid.from_dict(value["moving_grid"]),
            reference_grid=RegistrationGrid.from_dict(value["reference_grid"]),
            state=TransformState(**state),
            time_axis=AxisMetadata.from_dict(value["time_axis"])
            if value["time_axis"] is not None
            else None,
            reference_time=value["reference_time"],
            settings=tuple(value["settings"].items()),
            implementation=value["implementation"],
            schema_version=value["schema_version"],
            direction=value["direction"],
        )


def is_transform_data(value):
    return isinstance(value, TransformData)


def save_transform_output(transform: TransformData, path: str | Path):
    if not isinstance(transform, TransformData):
        raise TypeError("Transform export requires a spatial transform.")
    return atomic_write_json(path, transform.to_dict())


def load_transform(path: str | Path):
    def invalid_constant(value):
        raise ValueError(
            f"Non-finite JSON constant {value} is not a transform coordinate."
        )

    with Path(path).open(encoding="utf-8") as stream:
        return TransformData.from_dict(
            json.load(stream, parse_constant=invalid_constant)
        )


def registration_grid(state: ImageState) -> RegistrationGrid:
    """Resolve explicit semantics and canonical physical coordinates, not pixels."""
    if not isinstance(state, ImageState) or not state.axes_explicit:
        raise ValueError(
            "Registration requires explicit axis names. "
            "Declare YX, ZYX or time/channel axes first."
        )
    names = tuple(a.name.lower() for a in state.axes)
    if (
        len(names) != len(set(names))
        or len(names) != len(state.shape)
        or not set(names) <= set("tczyx")
    ):
        raise ValueError(
            "Registration requires unique explicit T/C/Z/Y/X axes "
            "matching the array rank."
        )
    for axis in state.axes:
        expected = (
            "time"
            if axis.name.lower() == "t"
            else "channel"
            if axis.name.lower() == "c"
            else "space"
        )
        if axis.type != expected:
            raise ValueError("Registration axis names and semantic types disagree.")
    spatial = tuple(name for name in ("z", "y", "x") if name in names)
    if spatial not in (("y", "x"), ("z", "y", "x")):
        raise ValueError(
            "Registration requires both X and Y, optionally Z, as spatial axes."
        )
    axes = tuple(state.axes[names.index(name)] for name in spatial)
    units = tuple(_unit_dimension_and_factor(axis.unit) for axis in axes)
    if len({unit[0] for unit in units}) != 1 or units[0][0] not in (
        "index",
        "length_micrometer",
    ):
        raise ValueError(
            "Registration spatial axes need compatible length units "
            "or explicit pixel coordinates."
        )
    frame = state.source.source_uuid or (
        f"{state.source.uri}#series={state.source.series_index}"
        if state.source.uri
        else state.source_name
    )
    if not frame:
        raise ValueError(
            "Registration needs an identified source frame; "
            "name the image source before estimating."
        )
    return RegistrationGrid(
        spatial,
        tuple(int(state.shape[names.index(name)]) for name in spatial),
        tuple(a.scale * unit[1] for a, unit in zip(axes, units, strict=True)),
        tuple(a.translation * unit[1] for a, unit in zip(axes, units, strict=True)),
        "micrometer" if units[0][0] == "length_micrometer" else "pixel",
        frame,
    )


def apply_transform_output_state(image_state, transform, data, mask=False):
    """Keep input T/C and acquisition facts; replace the spatial lattice explicitly."""
    grid = transform.reference_grid
    axes = []
    for axis in image_state.axes:
        name = axis.name.lower()
        if name in grid.axes:
            index = grid.axes.index(name)
            axes.append(
                replace(
                    axis,
                    scale=grid.spacing[index],
                    translation=grid.origin[index],
                    unit=grid.unit,
                )
            )
        else:
            axes.append(axis)
    state = image_state_from_array(
        data,
        axes=tuple(axes),
        source_name=image_state.source_name,
        source=replace(image_state.source, source_uuid=grid.frame_id),
        acquisition=image_state.acquisition,
        channels=image_state.channels,
        history=(
            *image_state.history,
            f"Apply Transform: {transform.state.model}; "
            "moving-to-reference; reference grid; original pixels resampled once",
        ),
        metadata_source="VIPP registered reference grid",
        defer_statistics=True,
    )
    return replace(state, kind="binary mask" if mask else image_state.kind)
