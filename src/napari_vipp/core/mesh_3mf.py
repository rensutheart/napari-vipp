"""Atomic, streamed 3MF Core surface publication without geometry repair.

The 3MF Consortium Core specification, sections 3.4, 4.1, 4.2 and 5.1,
defines the package, objects, calibration and sRGB display materials used here:
https://github.com/3MFConsortium/spec_core/blob/master/3MF%20Core%20Specification.md

Scientific surfaces are exported as ``surface``, not manufacturing-ready
``model`` solids. This expressly permits open meshes without inventing a
closure or claiming print validation. One component assembly preserves the
objects' relative positions; it does not perform a geometric union. Display
colours are 8-bit sRGB/RGBA, while the original floating-point colours and
source calibration remain in the per-object VIPP metadata.
"""

from __future__ import annotations

import io
import json
import os
import uuid
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr
from zipfile import ZIP64_LIMIT, ZIP_DEFLATED, ZipFile

import numpy as np

from napari_vipp.core.atomic_io import atomic_replace
from napari_vipp.core.grid import _unit_dimension_and_factor
from napari_vipp.core.meshes import MeshData, _mesh_calibration

CORE_NAMESPACE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
VIPP_NAMESPACE = "https://github.com/rensutheart/napari-vipp/3mf/metadata/1"
MODEL_CONTENT_TYPE = "application/vnd.ms-package.3dmanufacturing-3dmodel+xml"
MODEL_RELATIONSHIP = "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"
_CHUNK_SIZE = 8192
_MAX_3MF_COUNT = 2**31
_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" '
    'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    f'<Default Extension="model" ContentType="{MODEL_CONTENT_TYPE}"/>'
    "</Types>\n"
)
_RELATIONSHIPS = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    f'<Relationship Id="rel0" Type="{MODEL_RELATIONSHIP}" Target="/3D/3dmodel.model"/>'
    "</Relationships>\n"
)


def _valid_xml_text(value):
    text = str(value)
    if any(
        not (
            code in (9, 10, 13)
            or 0x20 <= code <= 0xD7FF
            or 0xE000 <= code <= 0xFFFD
            or 0x10000 <= code <= 0x10FFFF
        )
        for code in map(ord, text)
    ):
        raise ValueError("3MF names cannot contain XML control characters.")
    return text


def _metadata(name, value):
    """Custom metadata is namespaced; no custom core XML elements are used."""
    return (
        f'<metadata name={quoteattr(name)} preserve="true">'
        f"{escape(_valid_xml_text(value))}</metadata>\n"
    )


def _export_calibration(state):
    """Convert compatible physical axes to a supported common 3MF unit."""
    if tuple(axis.name.lower() for axis in state.spatial_axes) != (
        "z",
        "y",
        "x",
    ) or any(
        axis.type != "space" or not axis.is_explicit for axis in state.spatial_axes
    ):
        raise ValueError("3MF export requires explicitly ordered Z/Y/X mesh axes.")
    factors, _unit, dimension = _mesh_calibration(state)
    if dimension != "length_micrometer":
        raise ValueError(
            "3MF requires physical length units, not uncalibrated voxels. "
            "Set Pixel Size / Units before creating the mesh, then recalculate; "
            "or export OBJ to retain voxel coordinates."
        )
    x_factor = _unit_dimension_and_factor(state.spatial_axes[2].unit)[1]
    # Retain native common units supported by Core. Smaller physical units
    # (nm, pm and angstrom) are represented in microns, never interpreted as mm.
    unit = {
        1.0: "micron",
        1000.0: "millimeter",
        10000.0: "centimeter",
        1000000.0: "meter",
    }.get(x_factor, "micron")
    target_factor = {
        "micron": 1.0,
        "millimeter": 1000.0,
        "centimeter": 10000.0,
        "meter": 1000000.0,
    }[unit]
    factors = factors * (x_factor / target_factor)
    scales = np.array([axis.scale for axis in state.spatial_axes]) * factors
    offsets = np.array([axis.translation for axis in state.spatial_axes]) * factors
    if (
        not np.isfinite(scales).all()
        or np.any(scales <= 0)
        or not np.isfinite(offsets).all()
    ):
        raise ValueError(
            "3MF export requires finite, positive scales and finite origins."
        )
    return unit, scales, offsets


def _display_color(rgba):
    # Core displaycolor is explicitly an 8-bit sRGB value, not a printer
    # material. Keep the original unquantized RGBA in the object's metadata.
    return "#" + "".join(
        f"{int(np.floor(float(value) * 255 + 0.5)):02X}" for value in rgba
    )


def _write_model(stream, mesh, unit, scales, offsets, progress):
    from napari_vipp.core.mesh_objects import iter_mesh_objects

    stream.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    stream.write(
        f'<model xmlns="{CORE_NAMESPACE}" xmlns:vipp="{VIPP_NAMESPACE}" '
        f'unit="{unit}" xml:lang="en-US">\n'
    )
    stream.write(_metadata("Application", "VIPP"))
    stream.write(_metadata("Title", mesh.state.source_name or "VIPP mesh collection"))
    stream.write(
        _metadata(
            "Description",
            "Scientific surfaces; not validated or repaired for printing.",
        )
    )
    stream.write(
        _metadata(
            "vipp:MeshState",
            json.dumps(
                {
                    **mesh.state.to_dict(),
                    "export_coordinates": "XYZ",
                    "export_unit": unit,
                    "export_scale_applied": True,
                    "export_object_type": "surface",
                    "print_validation": "not performed",
                    "geometry_repair": "none",
                    "display_color_encoding": (
                        "8-bit sRGB RGBA; originals in object metadata"
                    ),
                },
                ensure_ascii=True,
                allow_nan=False,
            ),
        )
    )
    stream.write('<resources>\n<basematerials id="1">\n')
    for record in mesh.objects:
        name = _valid_xml_text(f"{record.name} (VIPP object {record.object_id})")
        stream.write(
            f"<base name={quoteattr(name)} "
            f'displaycolor="{_display_color(record.color)}"/>\n'
        )
    stream.write("</basematerials>\n")
    for index, part in enumerate(iter_mesh_objects(mesh)):
        if progress is not None:
            progress.check_cancelled()
            progress.report(
                index,
                mesh.object_count,
                f"Writing 3MF object {index + 1} of {mesh.object_count}",
            )
        record = part.objects[0]
        if (
            not 3 <= len(part.vertices) < _MAX_3MF_COUNT
            or not 1 <= len(part.faces) < _MAX_3MF_COUNT
        ):
            raise ValueError(
                "Each 3MF surface needs at least three vertices and one triangle, "
                "and fewer than 2^31 of either."
            )
        # Core resource IDs are 31-bit. VIPP's int64 identity is carried in
        # partnumber and metadata instead of being truncated into a resource ID.
        stream.write(
            f'<object id="{index + 2}" type="surface" '
            f"name={quoteattr(_valid_xml_text(record.name))} "
            f'partnumber="VIPP-{record.object_id}" pid="1" pindex="{index}">\n'
        )
        stream.write("<metadatagroup>\n")
        stream.write(
            _metadata(
                "vipp:Object",
                json.dumps(record.to_dict(), ensure_ascii=True, allow_nan=False),
            )
        )
        stream.write("</metadatagroup>\n<mesh>\n<vertices>\n")
        for start in range(0, len(part.vertices), _CHUNK_SIZE):
            if progress is not None:
                progress.check_cancelled()
            xyz = (part.vertices[start : start + _CHUNK_SIZE] * scales + offsets)[
                :, ::-1
            ]
            if not np.isfinite(xyz).all():
                raise ValueError(
                    "Calibrated mesh coordinates exceed finite numeric range."
                )
            stream.writelines(
                f'<vertex x="{x:.17g}" y="{y:.17g}" z="{z:.17g}"/>\n' for x, y, z in xyz
            )
        stream.write("</vertices>\n<triangles>\n")
        for start in range(0, len(part.faces), _CHUNK_SIZE):
            if progress is not None:
                progress.check_cancelled()
            # The ZYX -> XYZ reflection must preserve outward orientation.
            stream.writelines(
                f'<triangle v1="{v1:d}" v2="{v2:d}" v3="{v3:d}"/>\n'
                for v1, v2, v3 in part.faces[start : start + _CHUNK_SIZE, ::-1]
            )
        stream.write("</triangles>\n</mesh>\n</object>\n")
    assembly_id = mesh.object_count + 2
    stream.write(
        f'<object id="{assembly_id}" name="VIPP mesh collection"><components>\n'
    )
    for index in range(mesh.object_count):
        stream.write(f'<component objectid="{index + 2}"/>\n')
    stream.write("</components></object>\n</resources>\n")
    stream.write(f'<build><item objectid="{assembly_id}"/></build>\n</model>\n')
    if progress is not None:
        progress.report(mesh.object_count, mesh.object_count, "Finalizing 3MF archive")


def write_mesh_3mf(mesh, path, *, overwrite=True, progress=None):
    """Write a self-contained 3MF with units, distinct objects and RGBA colours.

    Objects remain in calibrated physical positions, grouped as one assembly.
    Open surfaces are retained as surfaces; no printing scale, placement,
    welding, union, topology repair or material prescription is inferred.
    A cancellation or write failure removes the private temporary archive and
    leaves the destination unchanged. ``overwrite=False`` is race-safe.
    """
    if not isinstance(mesh, MeshData):
        raise TypeError("3MF export requires MeshData.")
    if not mesh.faces.size:
        raise ValueError(
            "The mesh has no triangles to export. Check the mask and filter settings."
        )
    if not str(path).strip():
        raise ValueError("Mesh save path cannot be blank.")
    target = Path(path).expanduser()
    if target.suffix and target.suffix.lower() != ".3mf":
        raise ValueError("3MF export requires a .3mf filename.")
    target = target.with_suffix(".3mf")
    if mesh.object_count + 2 >= _MAX_3MF_COUNT:
        raise ValueError("The mesh has too many objects for 3MF resource IDs.")
    unit, scales, offsets = _export_calibration(mesh.state)
    if progress is not None:
        progress.check_cancelled()
    if not overwrite and target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    # A conservative bound allows ordinary-sized files to use plain ZIP for
    # older consumers, and enables ZIP64 before very large streamed XML starts.
    xml_bound = len(mesh.faces) * (3 * 128 + 128) + len(mesh.objects) * 4096
    try:
        with temporary.open("xb") as raw:
            with ZipFile(
                raw, "w", compression=ZIP_DEFLATED, compresslevel=6
            ) as archive:
                archive.writestr("[Content_Types].xml", _CONTENT_TYPES)
                archive.writestr("_rels/.rels", _RELATIONSHIPS)
                with archive.open(
                    "3D/3dmodel.model", "w", force_zip64=xml_bound >= ZIP64_LIMIT
                ) as part:
                    with io.TextIOWrapper(
                        part, encoding="utf-8", newline="\n"
                    ) as stream:
                        _write_model(stream, mesh, unit, scales, offsets, progress)
            raw.flush()
            os.fsync(raw.fileno())
        if progress is not None:
            progress.check_cancelled()
        if overwrite:
            atomic_replace(temporary, target)
        else:
            os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


__all__ = ["write_mesh_3mf"]
