"""Immutable object-aware meshes, full-resolution extraction and publication.

Vertices use voxel-centre Z/Y/X coordinates. Calibration is separate, so
inspection overlays the source without baking scale into pixels twice. OBJ
uses calibrated X/Y/Z coordinates in a common unit and right-handed winding.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from skimage.measure import marching_cubes

from napari_vipp.core.atomic_io import atomic_replace
from napari_vipp.core.grid import _unit_dimension_and_factor
from napari_vipp.core.metadata import AxisMetadata, ImageState


@dataclass(frozen=True)
class MeshObject:
    """Stable triangle-object identity and appearance, never cached measurements."""

    object_id: int
    name: str = ""
    color: tuple[float, float, float, float] = (0.2, 0.7, 1.0, 1.0)
    source_object_id: int | None = None
    parent_object_id: int | None = None
    source_name: str = ""
    source_axes: tuple[str, ...] = ()
    source_scale: tuple[float, ...] = ()
    source_units: tuple[str | None, ...] = ()
    source_origin: tuple[float, ...] = ()
    source_history: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("object_id", "source_object_id", "parent_object_id"):
            value = getattr(self, name)
            if value is None and name != "object_id":
                continue
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not 0 < value <= np.iinfo(np.int64).max
            ):
                raise ValueError("Mesh object IDs must be positive int64 integers.")
            object.__setattr__(self, name, int(value))
        color = tuple(float(v) for v in self.color)
        if len(color) != 4 or not all(np.isfinite(v) and 0 <= v <= 1 for v in color):
            raise ValueError(
                "Mesh object colours require four finite RGBA values in [0, 1]."
            )
        object.__setattr__(self, "color", color)
        object.__setattr__(self, "name", str(self.name or f"Object {self.object_id}"))
        object.__setattr__(self, "source_name", str(self.source_name))
        for name in ("source_axes", "source_history"):
            object.__setattr__(
                self, name, tuple(str(item) for item in getattr(self, name))
            )
        object.__setattr__(
            self,
            "source_units",
            tuple(None if value is None else str(value) for value in self.source_units),
        )
        for name in ("source_scale", "source_origin"):
            values = tuple(float(value) for value in getattr(self, name))
            if not all(
                np.isfinite(value) and (name != "source_scale" or value > 0)
                for value in values
            ):
                raise ValueError(
                    "Mesh source calibration must be finite, with positive scales."
                )
            object.__setattr__(self, name, values)
        if any(
            len(getattr(self, name)) not in {0, 3}
            for name in ("source_axes", "source_scale", "source_units", "source_origin")
        ):
            raise ValueError(
                "Mesh source calibration must describe three spatial axes."
            )

    def to_dict(self):
        from dataclasses import asdict

        return asdict(self)


@dataclass(frozen=True)
class MeshState:
    vertex_count: int
    triangle_count: int
    spatial_axes: tuple[AxisMetadata, ...]
    source_shape: tuple[int, ...]
    boundary: str
    source_name: str = ""
    history: tuple[str, ...] = ()
    kind: str = "3D surface mesh"
    metadata_source: str = "VIPP marching cubes"
    object_count: int = 1
    processing_history: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("spatial_axes", "source_shape", "history", "processing_history"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def to_dict(self):
        return dict(
            kind=self.kind,
            vertex_count=self.vertex_count,
            triangle_count=self.triangle_count,
            object_count=self.object_count,
            spatial_axes=[axis.to_dict() for axis in self.spatial_axes],
            source_shape=list(self.source_shape),
            boundary=self.boundary,
            source_name=self.source_name,
            history=list(self.history),
            metadata_source=self.metadata_source,
            algorithm="Lewiner marching cubes",
            level=0.5,
            step_size=1,
            vertex_coordinates="ZYX voxel centres",
            smoothing=next(
                (
                    item
                    for item in reversed(self.processing_history)
                    if "smooth" in item.lower()
                ),
                "none",
            ),
            simplification=next(
                (
                    item
                    for item in reversed(self.processing_history)
                    if "simplif" in item.lower()
                ),
                "none",
            ),
            processing_history=list(self.processing_history),
        )


@dataclass(frozen=True, eq=False)
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray
    state: MeshState
    face_object_ids: np.ndarray | None = None
    objects: tuple[MeshObject, ...] | None = None

    def __post_init__(self):
        vertices = np.array(self.vertices, dtype=np.float64, copy=True, order="C")
        raw_faces = np.asarray(self.faces)
        if not np.issubdtype(raw_faces.dtype, np.integer):
            raise ValueError("Mesh face indices must be integers.")
        faces = np.array(raw_faces, dtype=np.int64, copy=True, order="C")
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or not np.isfinite(vertices).all()
        ):
            raise ValueError("Mesh vertices must be finite N by 3 coordinates.")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("Mesh faces must be M by 3 triangle indices.")
        if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("Mesh face index is outside the vertex array.")
        if len(self.state.spatial_axes) != 3:
            raise ValueError("Mesh metadata requires three spatial axes.")
        raw_ids = (
            np.ones(len(faces), dtype=np.int64)
            if self.face_object_ids is None
            else np.asarray(self.face_object_ids)
        )
        if (
            raw_ids.shape != (len(faces),)
            or not np.issubdtype(raw_ids.dtype, np.integer)
            or (
                raw_ids.size
                and (np.any(raw_ids <= 0) or np.any(raw_ids > np.iinfo(np.int64).max))
            )
        ):
            raise ValueError(
                "Mesh face object IDs require one positive int64 integer per triangle."
            )
        ids = np.array(raw_ids, dtype=np.int64, copy=True)
        unique_ids = tuple(int(v) for v in np.unique(ids))
        records = (
            tuple(
                MeshObject(
                    v,
                    source_name=self.state.source_name,
                    source_axes=tuple(a.name for a in self.state.spatial_axes),
                    source_scale=tuple(a.scale for a in self.state.spatial_axes),
                    source_units=tuple(a.unit for a in self.state.spatial_axes),
                    source_origin=tuple(a.translation for a in self.state.spatial_axes),
                    source_history=self.state.history,
                )
                for v in unique_ids
            )
            if self.objects is None
            else tuple(self.objects)
        )
        if (
            any(not isinstance(item, MeshObject) for item in records)
            or len({item.object_id for item in records}) != len(records)
            or set(unique_ids) != {item.object_id for item in records}
        ):
            raise ValueError(
                "Mesh object records must match the unique face object IDs exactly."
            )
        vertices.setflags(write=False)
        faces.setflags(write=False)
        ids.setflags(write=False)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "face_object_ids", ids)
        object.__setattr__(
            self, "objects", tuple(sorted(records, key=lambda item: item.object_id))
        )
        object.__setattr__(
            self,
            "state",
            replace(
                self.state,
                vertex_count=len(vertices),
                triangle_count=len(faces),
                object_count=len(records),
            ),
        )

    @property
    def nbytes(self):
        return self.vertices.nbytes + self.faces.nbytes + self.face_object_ids.nbytes

    @property
    def object_count(self):
        return len(self.objects)

    def __deepcopy__(self, memo):
        # Snapshotting an immutable result can share it. NumPy's default deep
        # copy would both duplicate a large mesh and make its buffers writable.
        memo[id(self)] = self
        return self


def is_mesh_data(data):
    return isinstance(data, MeshData)


def mask_mesh_axes(shape, image_state=None):
    """Require one explicitly identified volume; never collapse time/channels."""
    if len(shape) != 3 or any(size < 2 for size in shape):
        raise ValueError(
            "Mask to 3D Mesh needs one 3D volume with at least two voxels per "
            "spatial axis. Use Select Axis Slice to select/remove time or channel "
            "axes first; a 2D mask cannot be made volumetric by this node."
        )
    if image_state is None:
        axes = tuple(AxisMetadata(name, "space") for name in "zyx")
    else:
        if not isinstance(image_state, ImageState) or image_state.shape != tuple(shape):
            raise ValueError("Mask to 3D Mesh needs matching image metadata.")
        axes = image_state.axes
        if (
            len(axes) != 3
            or {a.name.lower() for a in axes} != set("zyx")
            or any(a.type != "space" or not a.is_explicit for a in axes)
        ):
            raise ValueError(
                "Mask to 3D Mesh requires explicitly declared Z/Y/X spatial "
                "axes. Set the image stack axes before calculating the mesh."
            )
    order = tuple(
        next(i for i, a in enumerate(axes) if a.name.lower() == n) for n in "zyx"
    )
    return order, tuple(axes[i] for i in order)


def mask_to_3d_mesh(
    data,
    boundary="Close at image border",
    object_mode="Single object",
    *,
    image_state=None,
    progress=None,
):
    """Extract all foreground at level 0.5, step 1, without mesh smoothing.

    Raw array calls use ZYX; pipeline calls must carry explicit spatial axes.
    Closing pads one background voxel beyond the acquisition border. Open
    extraction does not assume anything outside the acquired volume.
    """
    if boundary not in {"Close at image border", "Leave open at image border"}:
        raise ValueError("Unknown Mask to 3D Mesh boundary policy.")
    arr = np.asarray(data)
    if object_mode not in {"Single object", "Connected objects", "Label IDs"}:
        raise ValueError("Unknown mesh object extraction mode.")
    if object_mode != "Single object":
        return _extract_mesh_objects(arr, boundary, object_mode, image_state, progress)
    if arr.dtype != np.bool_:
        raise TypeError(
            "Mask to 3D Mesh requires a binary mask. Add Binary Threshold first."
        )
    order, axes = mask_mesh_axes(arr.shape, image_state)
    if progress is not None:
        progress.check_cancelled()
        progress.report(0, 3, "Preparing full-resolution mask")
    volume = arr.transpose(order)
    closed = boundary == "Close at image border"
    work = np.pad(volume, 1).astype(np.float32) if closed else volume.astype(np.float32)
    if progress is not None:
        progress.check_cancelled()
        progress.report(
            1,
            3,
            "Extracting full-resolution mesh; this library call "
            "finishes before cancellation can be observed",
        )
    if not work.any() or work.all():
        vertices = np.empty((0, 3), dtype=np.float64)
        faces = np.empty((0, 3), dtype=np.int64)
    else:
        vertices, faces, _, _ = marching_cubes(
            work,
            level=0.5,
            method="lewiner",
            step_size=1,
            allow_degenerate=False,
            gradient_direction="ascent",
        )
        if closed:
            vertices -= 1
    if progress is not None:
        progress.check_cancelled()
        progress.report(2, 3, "Finalizing mesh and calibration")
    history = tuple(getattr(image_state, "history", ())) + (
        f"Mask to 3D Mesh: Lewiner marching cubes; level 0.5; full resolution; "
        f"{boundary}; no smoothing",
    )
    state = MeshState(
        len(vertices),
        len(faces),
        axes,
        tuple(volume.shape),
        boundary,
        getattr(image_state, "source_name", ""),
        history,
    )
    result = MeshData(vertices, faces, state)
    if progress is not None:
        progress.check_cancelled()
        progress.report(
            3, 3, f"Mesh ready: {len(vertices):,} vertices, {len(faces):,} triangles"
        )
    return result


def _extract_mesh_objects(arr, boundary, object_mode, image_state, progress):
    from scipy import ndimage

    from napari_vipp.core.mesh_objects import object_id_color

    order, axes = mask_mesh_axes(arr.shape, image_state)
    volume = arr.transpose(order)
    if progress is not None:
        progress.check_cancelled()
    if object_mode == "Connected objects":
        if volume.dtype != np.bool_:
            raise TypeError(
                "Connected objects requires a binary mask; "
                "choose Label IDs for integer labels."
            )
        labels, _count = ndimage.label(
            volume, structure=ndimage.generate_binary_structure(3, 1)
        )
    else:
        if (
            not np.issubdtype(volume.dtype, np.integer)
            or volume.dtype == np.bool_
            or (
                volume.size
                and (volume.min() < 0 or volume.max() > np.iinfo(np.int64).max)
            )
        ):
            raise TypeError(
                "Label IDs requires non-negative integer labels within the "
                "int64 range; zero is background."
            )
        labels = volume
    # Dense temporary IDs avoid allocation proportional to sparse label IDs.
    # Bounding boxes then keep meshing work proportional to object regions,
    # rather than repeating marching cubes over the entire acquisition.
    unique, inverse = np.unique(labels, return_inverse=True)
    dense = inverse.reshape(labels.shape) + 1
    regions = ndimage.find_objects(dense)
    ids = [
        (int(value), regions[index]) for index, value in enumerate(unique) if value != 0
    ]
    vertices, faces, face_ids, objects = [], [], [], []
    offset = 0
    for index, (object_id, region) in enumerate(ids):
        if progress is not None:
            progress.check_cancelled()
            progress.report(
                index, max(1, len(ids)), f"Extracting object {int(object_id)}"
            )
        # Explicit object extraction preserves cavity shells as part of their
        # foreground object. It never infers objects from surface containment.
        region = tuple(
            slice(max(0, item.start - 1), min(size, item.stop + 1))
            for item, size in zip(region, labels.shape, strict=True)
        )
        start = np.asarray([item.start for item in region])
        result = mask_to_3d_mesh(labels[region] == object_id, boundary)
        if not len(result.faces):
            continue
        vertices.append(result.vertices + start)
        faces.append(result.faces + offset)
        offset += len(result.vertices)
        face_ids.append(np.full(len(result.faces), int(object_id), dtype=np.int64))
        objects.append(
            MeshObject(
                int(object_id),
                color=object_id_color(int(object_id)),
                source_object_id=int(object_id),
                source_name=getattr(image_state, "source_name", ""),
                source_axes=tuple(a.name for a in axes),
                source_scale=tuple(a.scale for a in axes),
                source_units=tuple(a.unit for a in axes),
                source_origin=tuple(a.translation for a in axes),
                source_history=tuple(getattr(image_state, "history", ())),
            )
        )
    history = tuple(getattr(image_state, "history", ())) + (
        "Mask to 3D Mesh: Lewiner marching cubes; level 0.5; full resolution; "
        f"{boundary}; "
        f"{object_mode}; face-connected foreground for Connected objects; no smoothing",
    )
    state = MeshState(
        offset,
        sum(len(v) for v in faces),
        axes,
        tuple(volume.shape),
        boundary,
        getattr(image_state, "source_name", ""),
        history,
    )
    result = MeshData(
        np.concatenate(vertices) if vertices else np.empty((0, 3)),
        np.concatenate(faces) if faces else np.empty((0, 3), dtype=np.int64),
        state,
        np.concatenate(face_ids) if face_ids else np.empty(0, dtype=np.int64),
        tuple(objects),
    )
    if progress is not None:
        progress.check_cancelled()
        progress.report(
            max(1, len(ids)),
            max(1, len(ids)),
            f"Mesh ready: {result.object_count} objects",
        )
    return result


def labels_to_3d_mesh(
    data, boundary="Close at image border", *, image_state=None, progress=None
):
    """Extract each positive integer label as one object, preserving source IDs."""
    return mask_to_3d_mesh(
        data,
        boundary,
        object_mode="Label IDs",
        image_state=image_state,
        progress=progress,
    )


def _mesh_calibration(state):
    dimensions = [_unit_dimension_and_factor(a.unit) for a in state.spatial_axes]
    if (
        len({dimension for dimension, _ in dimensions}) != 1
        or dimensions[0][0] == "time_second"
    ):
        raise ValueError(
            "Mesh operations require compatible units on Z/Y/X. "
            "Set Pixel Size / Units before creating the mesh."
        )
    dimension = dimensions[0][0]
    # Keep the X unit when possible, converting the other two coordinates to
    # it. Unknown but identical units follow the grid contract's exact identity.
    target_factor = dimensions[2][1]
    unit = state.spatial_axes[2].unit or "voxel (uncalibrated)"
    factors = np.array([factor / target_factor for _, factor in dimensions])
    return factors, str(unit), dimension


def measure_mesh_geometry(mesh, *, include_convex_hull_metrics=True, progress=None):
    """Measure each explicit mesh object without remeshing or implicit union."""
    from napari_vipp.core.mesh_objects import iter_mesh_objects
    from napari_vipp.core.tables import TableData

    if not isinstance(mesh, MeshData):
        raise TypeError("Mesh measurements require MeshData.")
    parts = iter_mesh_objects(mesh) if mesh.object_count else iter((mesh,))
    tables = []
    for index, part in enumerate(parts):
        if progress is not None:
            progress.check_cancelled()
            progress.report(
                index, max(1, mesh.object_count), "Measuring mesh objects; no remeshing"
            )
        tables.append(
            _measure_single_mesh_geometry(
                part,
                include_convex_hull_metrics=include_convex_hull_metrics,
                progress=(
                    _ObjectMeasurementProgress(
                        progress, index, max(1, mesh.object_count)
                    )
                    if progress is not None
                    else None
                ),
            )
        )
    template = tables[0]
    result = TableData(
        columns=template.columns,
        rows=tuple(row for table in tables for row in table.rows),
        name=template.name,
        table_kind=template.table_kind,
        source_name=mesh.state.source_name,
        column_units=template.column_units,
    )
    if progress is not None:
        progress.check_cancelled()
        progress.report(
            max(1, mesh.object_count),
            max(1, mesh.object_count),
            "Mesh measurements complete",
        )
    return result


class _ObjectMeasurementProgress:
    """Map each object's three phases into one monotonically advancing bar."""

    def __init__(self, parent, index, count):
        self.parent, self.index, self.count = parent, index, count

    def check_cancelled(self):
        self.parent.check_cancelled()

    def report(self, current, total, message=""):
        phase = round(3 * current / total) if total else 0
        self.parent.report(
            3 * self.index + phase,
            3 * self.count,
            f"Object {self.index + 1}/{self.count} · {message}",
        )


def _measure_single_mesh_geometry(
    mesh, *, include_convex_hull_metrics=True, progress=None
):
    """Measure one object's supplied triangles, without remeshing or repair.

    Calibration is converted to the X-axis unit. An open, non-manifold,
    degenerate or inconsistently wound surface has no reported enclosed volume.
    Edge checks are not a general self-intersection test. Voxel counts cannot
    be recovered from a surface and are deliberately not invented.
    """
    from scipy.spatial import QhullError

    from napari_vipp.core import operations as ops
    from napari_vipp.core.tables import table_from_columns

    if progress is not None:
        progress.check_cancelled()
        progress.report(0, 3, "Measuring existing mesh; no remeshing")
    factors, _unit, _dimension = _mesh_calibration(mesh.state)
    scales = np.array([axis.scale for axis in mesh.state.spatial_axes]) * factors
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Mesh measurements require finite, positive spatial scales.")
    unit = mesh.state.spatial_axes[2].unit or "voxel"
    names = ("z", "y", "x")
    units = ops._mesh_units(
        scales,
        (unit,) * 3,
        names,
        include_convex_hull_metrics=include_convex_hull_metrics,
    )
    columns = ops._mesh_morphology_empty_columns(
        (),
        names,
        include_convex_hull_metrics=include_convex_hull_metrics,
    )
    # Keep existing metric names, but distinguish a mesh row from a label row.
    for name in ("label_id", "voxel_count", "voxel_volume_physical"):
        columns.pop(name)
    row = {name: float("nan") for name in columns}
    row.update(
        mesh_status="empty_mesh",
        mesh_error="Mesh has no triangles.",
        physical_unit=unit,
    )
    watertight = False
    if mesh.faces.size:
        vertices = mesh.vertices[np.unique(mesh.faces)]
        # Translation does not affect these measurements. Centre coordinates
        # before the signed-volume sum to avoid cancellation far from origin.
        centre = vertices.mean(axis=0)
        vertices = (mesh.vertices - centre) * scales
        if not np.isfinite(vertices).all():
            raise ValueError("Calibrated mesh coordinates exceed finite numeric range.")
        triangles = vertices[mesh.faces]
        areas = (
            np.linalg.norm(
                np.cross(
                    triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
                ),
                axis=1,
            )
            * 0.5
        )
        area = float(areas.sum())
        if not np.isfinite(area):
            raise ValueError("Mesh surface area exceeds finite numeric range.")
        edges = np.concatenate(
            [mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]]
        )
        _edges, inverse, counts = np.unique(
            np.sort(edges, axis=1),
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        winding = np.bincount(
            inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1)
        )
        unique_faces = len(np.unique(np.sort(mesh.faces, axis=1), axis=0)) == len(
            mesh.faces
        )
        watertight = bool(
            np.all(counts == 2)
            and np.all(winding == 0)
            and np.all(areas > 0)
            and unique_faces
        )
        volume = (
            abs(ops._signed_triangle_mesh_volume(vertices, mesh.faces))
            if watertight
            else float("nan")
        )
        if watertight and not np.isfinite(volume):
            raise ValueError("Mesh volume exceeds finite numeric range.")
        referenced_vertices = vertices[np.unique(mesh.faces)]
        extents = np.ptp(referenced_vertices, axis=0)
        row.update(
            mesh_status="ok" if watertight else "open_or_invalid_surface",
            mesh_error=""
            if watertight
            else (
                "Surface is open, degenerate, non-manifold or inconsistently wound; "
                "volume-based measurements are unavailable. Geometry was not repaired."
            ),
            mesh_surface_area_physical=area,
            mesh_volume_physical=volume,
            surface_area_to_volume=ops._nan_ratio(area, volume),
            equivalent_sphere_radius_physical=ops._equivalent_sphere_radius(volume),
            equivalent_sphere_diameter_physical=2
            * ops._equivalent_sphere_radius(volume),
            sphericity=ops._sphericity(volume, area),
        )
        for i, name in enumerate(names):
            row[f"mesh_extent_{name}_physical"] = float(extents[i])
            for j in range(i + 1, 3):
                row[f"mesh_extent_ratio_{name}_{names[j]}"] = ops._nan_ratio(
                    extents[i], extents[j]
                )
        if progress is not None:
            progress.check_cancelled()
            progress.report(
                2,
                3,
                "Measuring mesh convex hull"
                if include_convex_hull_metrics
                else "Finalizing mesh measurements",
            )
        if include_convex_hull_metrics:
            try:
                hull = ops._convex_hull_metrics(referenced_vertices)
                row.update(hull)
                row["solidity_3d"] = ops._nan_ratio(
                    volume, hull["convex_hull_volume_physical"]
                )
                row["surface_area_to_convex_hull_area"] = ops._nan_ratio(
                    area, hull["convex_hull_surface_area_physical"]
                )
            except (QhullError, ValueError) as error:
                row["mesh_status"] = (
                    "convex_hull_failed" if watertight else row["mesh_status"]
                )
                row["mesh_error"] = (
                    row["mesh_error"] + " Convex hull: " + str(error)
                ).strip()
    result = table_from_columns(
        {
            "mesh_id": [mesh.objects[0].object_id if mesh.objects else 1],
            "vertex_count": [len(mesh.vertices)],
            "triangle_count": [len(mesh.faces)],
            "watertight": [watertight],
            **{name: [value] for name, value in row.items()},
        },
        name="3D mesh morphology measurements",
        table_kind="3D mesh morphology",
        source_name=mesh.state.source_name,
        column_units={
            name: value for name, value in units.column_units.items() if name in row
        },
    )
    if progress is not None:
        progress.check_cancelled()
        progress.report(3, 3, "Mesh measurements complete")
    return result


def save_mesh_output(mesh, path, *, format="auto", overwrite=True, progress=None):
    """Atomically publish OBJ groups or object-coloured, calibrated 3MF.

    OBJ itself has no standard units. The JSON comment records calibration and
    history in the same atomic artifact rather than a detached optional file.
    Export does not repair, smooth, re-centre, or rescale to printing dimensions.
    """
    if not isinstance(mesh, MeshData):
        raise TypeError("Mesh export requires MeshData.")
    if not mesh.faces.size:
        raise ValueError(
            "The mesh has no triangles to export. Check the mask and border policy."
        )
    if not str(path).strip():
        raise ValueError("Mesh save path cannot be blank.")
    target = Path(path).expanduser()
    fmt = str(format).strip().lower()
    suffix = target.suffix.lower()
    if fmt == "auto":
        fmt = suffix.lstrip(".") if suffix else "obj"
    if fmt not in {"obj", "3mf"} or (suffix and suffix != f".{fmt}"):
        raise ValueError(
            "Mesh export requires matching OBJ (.obj) or 3MF (.3mf) format."
        )
    if fmt == "3mf":
        from napari_vipp.core.mesh_3mf import write_mesh_3mf

        return write_mesh_3mf(mesh, target, overwrite=overwrite, progress=progress)
    target = target.with_suffix(".obj")
    factors, unit, _ = _mesh_calibration(mesh.state)
    scales = np.array([a.scale for a in mesh.state.spatial_axes]) * factors
    offsets = np.array([a.translation for a in mesh.state.spatial_axes]) * factors
    if not overwrite and target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    total = len(mesh.vertices) + len(mesh.faces)
    metadata = {
        **mesh.state.to_dict(),
        "export_coordinates": "XYZ",
        "export_unit": unit,
        "export_scale_applied": True,
        "objects": [item.to_dict() for item in mesh.objects],
        "object_colours": "metadata only; use 3MF for standard display colours",
    }
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("# VIPP surface mesh; units: " + json.dumps(unit) + "\n")
            handle.write(
                "# VIPP metadata: "
                + json.dumps(metadata, ensure_ascii=True, allow_nan=False)
                + "\n"
            )
            handle.write("o VIPP_mesh\n")
            for start in range(0, len(mesh.vertices), 8192):
                if progress is not None:
                    progress.check_cancelled()
                    progress.report(start, total, "Writing mesh vertices")
                xyz = (mesh.vertices[start : start + 8192] * scales + offsets)[:, ::-1]
                if not np.isfinite(xyz).all():
                    raise ValueError(
                        "Calibrated mesh coordinates exceed finite precision."
                    )
                np.savetxt(handle, xyz, fmt="v %.17g %.17g %.17g")
            # ZYX -> XYZ reverses handedness. Reverse each face so its outward
            # orientation is retained in OBJ's conventional XYZ coordinates.
            order = np.argsort(mesh.face_object_ids, kind="stable")
            grouped_ids = mesh.face_object_ids[order]
            for item in mesh.objects:
                start = int(np.searchsorted(grouped_ids, item.object_id, side="left"))
                end = int(np.searchsorted(grouped_ids, item.object_id, side="right"))
                handle.write(f"o VIPP_object_{item.object_id}\n")
                for offset in range(int(start), end, 8192):
                    if progress is not None:
                        progress.check_cancelled()
                        progress.report(
                            len(mesh.vertices) + offset,
                            total,
                            "Writing mesh object faces",
                        )
                    indices = order[offset : min(offset + 8192, end)]
                    np.savetxt(handle, mesh.faces[indices, ::-1] + 1, fmt="f %d %d %d")
            handle.flush()
            os.fsync(handle.fileno())
        if progress is not None:
            progress.check_cancelled()
        if overwrite:
            atomic_replace(temporary, target)
        else:
            os.link(temporary, target)  # Atomic create-if-absent, including races.
    finally:
        temporary.unlink(missing_ok=True)
    return target
