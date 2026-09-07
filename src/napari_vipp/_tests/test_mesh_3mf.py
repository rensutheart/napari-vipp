"""3MF package, geometry, calibration and atomic-publication contracts."""

import json
from dataclasses import replace
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from napari_vipp.core import mesh_3mf
from napari_vipp.core.mesh_3mf import write_mesh_3mf
from napari_vipp.core.meshes import MeshData, MeshObject, MeshState
from napari_vipp.core.metadata import AxisMetadata
from napari_vipp.core.progress import OperationCancelled

NS = {"c": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}


def _mesh(*, unit="um", open_surface=False):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    if open_surface:
        faces = faces[:1]
    state = MeshState(
        0,
        0,
        tuple(
            AxisMetadata(n, "space", unit, s, t)
            for n, s, t in zip("zyx", [2, 3, 4], [10, 20, -30], strict=True)
        ),
        (10, 10, 10),
        "Leave open at image border" if open_surface else "Close at image border",
        source_name="μ specimen & <reference>",
        history=("test extraction",),
    )
    records = (
        MeshObject(
            7,
            'Red & "first"',
            (1, 0, 0, 1),
            source_object_id=3,
            source_name="red source",
            source_axes=("z", "y", "x"),
            source_scale=(1.0, 2.0, 3.0),
            source_units=("um", "um", "um"),
            source_origin=(0.0, 5.0, -10.0),
            source_history=("Mask extracted at level 0.5", "Split by connectivity"),
        ),
        MeshObject(
            2**40,
            "Blue α",
            (0, 0.25, 1, 0.5),
            parent_object_id=5,
            source_name="blue source",
        ),
    )
    return MeshData(
        np.concatenate([vertices, vertices + [0, 0, 10]]),
        np.concatenate([faces, faces + len(vertices)]),
        state,
        np.repeat([record.object_id for record in records], len(faces)),
        records,
    )


def _read(path):
    """Read through stdlib ZIP/XML, independently of the export implementation."""
    with ZipFile(path) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {
            "[Content_Types].xml",
            "_rels/.rels",
            "3D/3dmodel.model",
        }
        relationships = ET.fromstring(archive.read("_rels/.rels"))
        start = relationships[0]
        assert (
            start.attrib["Type"]
            == "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"
        )
        root = ET.fromstring(archive.read(start.attrib["Target"].lstrip("/")))
        types = ET.fromstring(archive.read("[Content_Types].xml"))
        assert {node.attrib["Extension"]: node.attrib["ContentType"] for node in types}[
            "model"
        ] == "application/vnd.ms-package.3dmanufacturing-3dmodel+xml"
        assert all(info.compress_type == ZIP_DEFLATED for info in archive.infolist())
        # Small files need not require ZIP64 merely because XML was streamed.
        assert all(info.extract_version < 45 for info in archive.infolist())
    objects = []
    for node in root.findall("c:resources/c:object", NS):
        geometry = node.find("c:mesh", NS)
        if geometry is None:
            continue
        vertices = np.array(
            [
                [float(vertex.attrib[n]) for n in "xyz"]
                for vertex in geometry.find("c:vertices", NS)
            ]
        )
        faces = np.array(
            [
                [int(face.attrib[n]) for n in ("v1", "v2", "v3")]
                for face in geometry.find("c:triangles", NS)
            ]
        )
        metadata = json.loads(node.find("c:metadatagroup/c:metadata", NS).text)
        objects.append((node, vertices, faces, metadata))
    return root, objects


def _signed_volume(vertices, faces):
    triangles = vertices[faces]
    return (
        np.einsum(
            "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
        ).sum()
        / 6
    )


def test_3mf_preserves_objects_calibration_colors_and_winding(tmp_path):
    mesh = _mesh()
    before_vertices, before_faces = mesh.vertices.copy(), mesh.faces.copy()
    path = write_mesh_3mf(mesh, tmp_path / "specimen")
    assert path.suffix == ".3mf"
    root, objects = _read(path)
    assert root.attrib["unit"] == "micron"
    assert len(objects) == 2
    assert root.find("c:metadata[@name='Title']", NS).text == mesh.state.source_name
    metadata = json.loads(root.find("c:metadata[@name='vipp:MeshState']", NS).text)
    assert metadata["print_validation"] == "not performed"
    assert metadata["geometry_repair"] == "none"
    assert metadata["export_object_type"] == "surface"
    assert metadata["spatial_axes"] == mesh.state.to_dict()["spatial_axes"]
    base = root.findall("c:resources/c:basematerials/c:base", NS)
    assert [element.attrib["displaycolor"] for element in base] == [
        "#FF0000FF",
        "#0040FF80",
    ]
    for index, (node, vertices, faces, record) in enumerate(objects):
        assert node.attrib["type"] == "surface"
        assert node.attrib["name"] == mesh.objects[index].name
        assert node.attrib["partnumber"] == f"VIPP-{mesh.objects[index].object_id}"
        assert int(node.attrib["id"]) < 2**31
        assert node.attrib["pid"] == "1"
        assert int(node.attrib["pindex"]) == index
        assert record == json.loads(json.dumps(mesh.objects[index].to_dict()))
        expected = (
            mesh.vertices[index * 4 : index * 4 + 4] * [2, 3, 4] + [10, 20, -30]
        )[:, ::-1]
        np.testing.assert_allclose(vertices, expected)
        np.testing.assert_array_equal(
            faces, mesh.faces[index * 4 : index * 4 + 4, ::-1] - index * 4
        )
        assert _signed_volume(vertices, faces) == pytest.approx(4)
    assembly = root.findall("c:resources/c:object", NS)[-1]
    assert assembly.find("c:mesh", NS) is None
    components = assembly.findall("c:components/c:component", NS)
    assert [c.attrib["objectid"] for c in components] == [
        o[0].attrib["id"] for o in objects
    ]
    assert all("transform" not in component.attrib for component in components)
    items = root.findall("c:build/c:item", NS)
    assert len(items) == 1 and items[0].attrib == {"objectid": assembly.attrib["id"]}
    np.testing.assert_array_equal(mesh.vertices, before_vertices)
    np.testing.assert_array_equal(mesh.faces, before_faces)
    assert not mesh.vertices.flags.writeable and not mesh.faces.flags.writeable


@pytest.mark.parametrize(
    "unit,encoded,factor",
    [
        ("mm", "millimeter", 1),
        ("cm", "centimeter", 1),
        ("m", "meter", 1),
        ("nm", "micron", 0.001),
        ("pm", "micron", 0.000001),
        ("angstrom", "micron", 0.0001),
    ],
)
def test_3mf_native_and_submicron_units(tmp_path, unit, encoded, factor):
    mesh = _mesh(unit=unit)
    root, objects = _read(write_mesh_3mf(mesh, tmp_path / "units.3mf"))
    assert root.attrib["unit"] == encoded
    np.testing.assert_allclose(objects[0][1][0], np.array([-30, 20, 10]) * factor)


def test_3mf_mixed_compatible_units_convert_scale_and_origin(tmp_path):
    mesh = _mesh()
    axes = (
        AxisMetadata("z", "space", "nm", 2000, 10000),
        AxisMetadata("y", "space", "mm", 0.003, 0.020),
        AxisMetadata("x", "space", "um", 4, -30),
    )
    mesh = replace(mesh, state=replace(mesh.state, spatial_axes=axes))
    root, objects = _read(write_mesh_3mf(mesh, tmp_path / "mixed.3mf"))
    assert root.attrib["unit"] == "micron"
    expected = (mesh.vertices[:4] * [2, 3, 4] + [10, 20, -30])[:, ::-1]
    np.testing.assert_allclose(objects[0][1], expected)


@pytest.mark.parametrize("unit", [None, "pixel", "voxel", "furlong", "second"])
def test_3mf_never_invents_physical_calibration(tmp_path, unit):
    with pytest.raises(ValueError, match="physical length units|compatible units"):
        write_mesh_3mf(_mesh(unit=unit), tmp_path / "uncalibrated.3mf")
    assert not list(tmp_path.iterdir())


def test_3mf_open_surfaces_are_not_closed_or_promoted_to_solids(tmp_path):
    mesh = _mesh(open_surface=True)
    _root, objects = _read(write_mesh_3mf(mesh, tmp_path / "open.3mf"))
    assert [node.attrib["type"] for node, *_ in objects] == ["surface", "surface"]
    assert [len(faces) for _, _, faces, _ in objects] == [1, 1]
    assert [len(vertices) for _, vertices, *_ in objects] == [3, 3]


def test_3mf_rejects_invalid_axes_and_overflow_without_replacing(tmp_path):
    mesh = _mesh()
    target = tmp_path / "previous.3mf"
    target.write_bytes(b"previous")
    changed = replace(
        mesh, state=replace(mesh.state, spatial_axes=mesh.state.spatial_axes[::-1])
    )
    with pytest.raises(ValueError, match="ordered Z/Y/X"):
        write_mesh_3mf(changed, target)
    axes = tuple(replace(axis, scale=1e308) for axis in mesh.state.spatial_axes)
    changed = replace(mesh, state=replace(mesh.state, spatial_axes=axes))
    with (
        np.errstate(over="ignore"),
        pytest.raises(ValueError, match="finite numeric range"),
    ):
        write_mesh_3mf(changed, target)
    assert target.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("name", ["nul\x00", "control\x01", "surrogate\ud800"])
def test_3mf_rejects_non_xml_names_instead_of_emitting_broken_document(tmp_path, name):
    mesh = _mesh()
    changed = replace(
        mesh, objects=(replace(mesh.objects[0], name=name), mesh.objects[1])
    )
    with pytest.raises(ValueError, match="XML control"):
        write_mesh_3mf(changed, tmp_path / "broken.3mf")
    assert not list(tmp_path.iterdir())


def test_3mf_empty_and_wrong_inputs_are_actionable(tmp_path):
    mesh = _mesh()
    empty = MeshData(np.empty((0, 3)), np.empty((0, 3), dtype=int), mesh.state)
    with pytest.raises(ValueError, match="no triangles"):
        write_mesh_3mf(empty, tmp_path / "empty.3mf")
    with pytest.raises(TypeError, match="MeshData"):
        write_mesh_3mf(np.zeros((3, 3)), tmp_path / "array.3mf")
    with pytest.raises(ValueError, match="blank"):
        write_mesh_3mf(mesh, " ")
    with pytest.raises(ValueError, match=".3mf filename"):
        write_mesh_3mf(mesh, tmp_path / "wrong.obj")
    assert not list(tmp_path.iterdir())


def test_3mf_cancellation_during_streaming_keeps_previous_output(tmp_path, monkeypatch):
    monkeypatch.setattr(mesh_3mf, "_CHUNK_SIZE", 1)
    target = tmp_path / "cancel.3mf"
    target.write_bytes(b"previous")

    class CancelDuringVertices:
        checks = 0

        def check_cancelled(self):
            self.checks += 1
            if self.checks == 5:
                raise OperationCancelled("cancelled while streaming")

        def report(self, *_args):
            pass

    with pytest.raises(OperationCancelled, match="streaming"):
        write_mesh_3mf(_mesh(), target, progress=CancelDuringVertices())
    assert target.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [target]


def test_3mf_failed_atomic_replace_keeps_previous_output(tmp_path, monkeypatch):
    target = tmp_path / "failure.3mf"
    target.write_bytes(b"previous")

    def fail(*_args):
        raise OSError("publication failed")

    monkeypatch.setattr(mesh_3mf, "atomic_replace", fail)
    with pytest.raises(OSError, match="publication failed"):
        write_mesh_3mf(_mesh(), target)
    assert target.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [target]


def test_3mf_no_overwrite_is_race_safe(tmp_path, monkeypatch):
    target = tmp_path / "race.3mf"
    original_link = mesh_3mf.os.link

    def competing_publication(source, destination):
        destination.write_bytes(b"other writer")
        return original_link(source, destination)

    monkeypatch.setattr(mesh_3mf.os, "link", competing_publication)
    with pytest.raises(FileExistsError):
        write_mesh_3mf(_mesh(), target, overwrite=False)
    assert target.read_bytes() == b"other writer"
    assert list(tmp_path.iterdir()) == [target]


def test_3mf_no_overwrite_existing_and_successful_publication(tmp_path):
    target = tmp_path / "existing.3mf"
    write_mesh_3mf(_mesh(), target, overwrite=False)
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        write_mesh_3mf(_mesh(open_surface=True), target, overwrite=False)
    assert target.read_bytes() == before
    write_mesh_3mf(_mesh(open_surface=True), target, overwrite=True)
    assert [len(part[2]) for part in _read(target)[1]] == [1, 1]
    assert list(tmp_path.iterdir()) == [target]
