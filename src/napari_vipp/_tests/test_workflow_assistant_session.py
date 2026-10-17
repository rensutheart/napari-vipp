from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import struct
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import napari_vipp.core.workflow_assistant_session as archive_module
from napari_vipp.core.workflow_assistant_session import (
    AssistantSessionError,
    SavedAssistantSession,
    load_assistant_session,
    make_checkpoint,
    save_assistant_session,
)


def _npy(data):
    stream = io.BytesIO()
    np.save(stream, data, allow_pickle=data.dtype.hasobject)
    return stream.getvalue()


def _png():
    stream = io.BytesIO()
    Image.new("RGB", (8, 7), "blue").save(stream, format="PNG")
    return stream.getvalue()


def _name(payload, kind="arrays"):
    extension = "npy" if kind == "arrays" else "png"
    return f"{kind}/{hashlib.sha256(payload).hexdigest()}.{extension}"


@pytest.fixture
def checkpoint():
    array, preview = _npy(np.arange(30, dtype=np.uint8).reshape(2, 3, 5)), _png()
    return make_checkpoint(
        {
            "kind": "synthetic-passive-session",
            "version": 1,
            "goal": "Synthetic goal",
            "usage": {"calls": 4},
        },
        {_name(array): array, _name(preview, "previews"): preview},
    )


def _members(path):
    with zipfile.ZipFile(path) as stream:
        return [(info.filename, stream.read(info)) for info in stream.infolist()]


def _write_zip(path, members, *, compression=zipfile.ZIP_STORED):
    with zipfile.ZipFile(path, "w", compression=compression) as stream:
        for name, payload in members:
            stream.writestr(name, payload)


def _replace_member(members, name, payload):
    return [(key, payload if key == name else value) for key, value in members]


def test_roundtrip_is_passive_immutable_hash_bound_and_deterministic(
    tmp_path, checkpoint, monkeypatch
):
    first, second = tmp_path / "first.vipp-session", tmp_path / "second.vipp-session"
    receipt = save_assistant_session(first, checkpoint)
    again = save_assistant_session(second, checkpoint)
    assert receipt.sha256 == again.sha256
    assert receipt.byte_count == first.stat().st_size
    assert receipt.blob_count == 2
    assert receipt.path == first
    assert receipt.to_dict()["sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    monkeypatch.setattr(
        np,
        "load",
        lambda *args, **kwargs: pytest.fail(
            "No pickle/array execution on passive open"
        ),
    )
    loaded = load_assistant_session(first)
    assert loaded.document == checkpoint.document
    assert loaded.blob_names == checkpoint.blob_names
    assert all(
        loaded.blob_bytes(key) == checkpoint.blob_bytes(key)
        for key in loaded.blob_names
    )
    loaded.document["usage"]["calls"] = 0
    assert loaded.document["usage"]["calls"] == 4
    assert len(list(tmp_path.iterdir())) == 2  # No extracted members or profile state.


def test_empty_blob_inventory_is_valid(tmp_path):
    checkpoint = make_checkpoint({"history": []}, {})
    target = tmp_path / "empty.vipp-session"
    save_assistant_session(target, checkpoint)
    assert load_assistant_session(target).document == {"history": []}


def test_default_save_never_overwrites_existing_bytes(tmp_path, checkpoint):
    target = tmp_path / "original"
    target.write_bytes(b"original user bytes")
    with pytest.raises(FileExistsError):
        save_assistant_session(target, checkpoint)
    assert target.read_bytes() == b"original user bytes"
    receipt = save_assistant_session(target, checkpoint, overwrite=True)
    assert receipt.sha256 == hashlib.sha256(target.read_bytes()).hexdigest()
    assert load_assistant_session(target).document == checkpoint.document


def test_racing_no_overwrite_publication_preserves_intervening_file(
    tmp_path, checkpoint, monkeypatch
):
    target = tmp_path / "race"
    publication_name = "rename" if os.name == "nt" else "link"
    original = getattr(os, publication_name)

    def race(source, destination, **kwargs):
        Path(destination).write_bytes(b"intervening user file")
        return original(source, destination, **kwargs)

    monkeypatch.setattr(os, publication_name, race)
    with pytest.raises(FileExistsError):
        save_assistant_session(target, checkpoint)
    assert target.read_bytes() == b"intervening user file"
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_replace_failure_keeps_original_and_cleans_temporary(
    tmp_path, checkpoint, monkeypatch
):
    target = tmp_path / "target"
    target.write_bytes(b"keep")
    monkeypatch.setattr(
        archive_module,
        "atomic_replace",
        lambda *_: (_ for _ in ()).throw(PermissionError("locked")),
    )
    with pytest.raises(PermissionError):
        save_assistant_session(target, checkpoint, overwrite=True)
    assert target.read_bytes() == b"keep"
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize(
    "payload",
    [
        {"data": np.zeros(3)},
        {"object": object()},
        {"nan": float("nan")},
        {"bad": {1: "key"}},
        {"infinity": float("inf")},
    ],
)
def test_non_json_document_values_are_rejected(payload):
    with pytest.raises(AssistantSessionError):
        make_checkpoint(payload, {})


@pytest.mark.parametrize(
    "name",
    [
        "../escape.npy",
        "/absolute.npy",
        "arrays/../file.npy",
        "arrays\\file.npy",
        "auth.json",
        "arrays/" + "a" * 64 + ".png",
        "previews/" + "a" * 64 + ".npy",
        "arrays/" + "A" * 64 + ".npy",
    ],
)
def test_blob_name_allowlist_rejects_paths_and_unknown_types(name):
    with pytest.raises(AssistantSessionError):
        make_checkpoint({}, {name: b"data"})


def test_blob_digest_and_immutable_byte_type_are_required():
    array = _npy(np.zeros(2, dtype=np.uint8))
    with pytest.raises(AssistantSessionError, match="filename hash"):
        make_checkpoint({}, {"arrays/" + "0" * 64 + ".npy": array})
    with pytest.raises(AssistantSessionError, match="immutable"):
        make_checkpoint({}, {_name(array): bytearray(array)})


@pytest.mark.parametrize(
    "data",
    [
        np.array([object()], dtype=object),
        np.array(["text"]),
        np.zeros(2, dtype=[("x", "i4")]),
        np.zeros(2, dtype=np.complex128),
    ],
)
def test_object_structured_text_and_complex_npy_are_rejected_without_pickle(data):
    payload = _npy(data)
    with pytest.raises(AssistantSessionError):
        make_checkpoint({}, {_name(payload): payload})


@pytest.mark.parametrize(
    "data",
    [
        np.array(True),
        np.arange(7, dtype=np.int64),
        np.array([[1.0, float("nan")]], dtype=np.float64),
        np.asfortranarray(np.arange(12, dtype=np.float32).reshape(3, 4)),
        np.arange(4, dtype=">u2"),
    ],
)
def test_numeric_npy_bytes_are_preserved_without_casting(data):
    payload = _npy(data)
    saved = make_checkpoint({}, {_name(payload): payload})
    assert saved.blob_bytes(_name(payload)) == payload


@pytest.mark.parametrize(
    "mutation", ["truncated", "trailing", "huge-shape", "bad-magic"]
)
def test_npy_header_shape_and_exact_payload_length_are_bounded(mutation):
    payload = _npy(np.arange(3, dtype=np.uint8))
    if mutation == "truncated":
        payload = payload[:-1]
    elif mutation == "trailing":
        payload += b"extra"
    elif mutation == "bad-magic":
        payload = b"unsafe" + payload[6:]
    else:
        stream = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            stream, {"descr": "|u1", "fortran_order": False, "shape": (2**50,)}
        )
        payload = stream.getvalue()
    with pytest.raises(AssistantSessionError):
        make_checkpoint({}, {_name(payload): payload})


@pytest.mark.parametrize(
    "mutation", ["truncated", "trailing", "crc", "dimensions", "metadata"]
)
def test_png_structure_crc_pixels_and_metadata_are_checked(mutation):
    payload = _png()
    if mutation == "truncated":
        payload = payload[:-1]
    elif mutation == "trailing":
        payload += b"extra"
    elif mutation == "crc":
        payload = payload[:45] + bytes([payload[45] ^ 1]) + payload[46:]
    elif mutation == "dimensions":
        payload = bytearray(payload)
        struct.pack_into("!I", payload, 16, 2**30)
        struct.pack_into(
            "!I", payload, 29, archive_module.zlib.crc32(payload[12:29]) & 0xFFFFFFFF
        )
        payload = bytes(payload)
    else:
        from PIL.PngImagePlugin import PngInfo

        metadata = PngInfo()
        metadata.add_text("private", "not a preview-only chunk")
        stream = io.BytesIO()
        Image.new("RGB", (2, 2)).save(stream, format="PNG", pnginfo=metadata)
        payload = stream.getvalue()
    with pytest.raises(AssistantSessionError):
        make_checkpoint({}, {_name(payload, "previews"): payload})


def test_direct_constructor_rejects_noncanonical_json_and_duplicate_blobs():
    with pytest.raises(AssistantSessionError, match="canonical"):
        SavedAssistantSession(b'{ "x": 1 }', ())
    array = _npy(np.zeros(2, dtype=np.uint8))
    row = (_name(array), array)
    with pytest.raises(AssistantSessionError, match="unique"):
        SavedAssistantSession(b"{}", (row, row))


@pytest.mark.parametrize(
    "mutation",
    [
        "compressed",
        "duplicate",
        "unknown",
        "missing",
        "bad-hash",
        "bad-count",
        "unknown-schema",
        "duplicate-json-key",
        "payload-corrupt",
        "comment",
        "hidden-prefix",
    ],
)
def test_malicious_or_corrupt_zip_is_rejected(tmp_path, checkpoint, mutation):
    target = tmp_path / "archive"
    save_assistant_session(target, checkpoint)
    members = _members(target)
    compression = zipfile.ZIP_STORED
    manifest = json.loads(members[0][1])
    if mutation == "compressed":
        compression = zipfile.ZIP_DEFLATED
    elif mutation == "duplicate":
        members.append(members[1])
    elif mutation == "unknown":
        members.append(("extra.py", b"must not execute"))
    elif mutation == "missing":
        members.pop()
    elif mutation == "bad-hash":
        manifest["session"]["sha256"] = "0" * 64
    elif mutation == "bad-count":
        manifest["payload_bytes"] += 1
    elif mutation == "unknown-schema":
        manifest["version"] = 999
    elif mutation == "duplicate-json-key":
        members[1] = ("session.json", b'{"x":1,"x":2}')
        manifest["session"]["byte_count"] = len(members[1][1])
        manifest["session"]["sha256"] = hashlib.sha256(members[1][1]).hexdigest()
        manifest["payload_bytes"] = sum(
            len(data) for name, data in members if name != "manifest.json"
        )
    elif mutation == "payload-corrupt":
        members[1] = ("session.json", members[1][1] + b" ")
    members = _replace_member(members, "manifest.json", json.dumps(manifest).encode())
    _write_zip(target, members, compression=compression)
    if mutation == "comment":
        with zipfile.ZipFile(target, "a") as stream:
            stream.comment = b"hidden comment"
    elif mutation == "hidden-prefix":
        target.write_bytes(b"prefix" + target.read_bytes())
    with pytest.raises(AssistantSessionError):
        load_assistant_session(target)


def test_central_directory_count_is_rejected_before_zipfile_parser(
    tmp_path, checkpoint, monkeypatch
):
    target = tmp_path / "archive"
    save_assistant_session(target, checkpoint)
    payload = bytearray(target.read_bytes())
    struct.pack_into("<HH", payload, len(payload) - 14, 65535, 65535)
    target.write_bytes(payload)
    monkeypatch.setattr(
        zipfile,
        "ZipFile",
        lambda *_args, **_kwargs: pytest.fail("Preflight must reject count first"),
    )
    with pytest.raises(AssistantSessionError, match="directory/count"):
        load_assistant_session(target)


@pytest.mark.parametrize("operation", ["load", "save"])
def test_hardlinked_archive_target_is_rejected(tmp_path, checkpoint, operation):
    target, alias = tmp_path / "archive", tmp_path / "alias"
    save_assistant_session(target, checkpoint)
    os.link(target, alias)
    with pytest.raises(AssistantSessionError, match="single regular"):
        if operation == "load":
            load_assistant_session(target)
        else:
            save_assistant_session(target, checkpoint, overwrite=True)


@pytest.mark.parametrize("operation", ["load", "save"])
def test_windows_reparse_attribute_is_rejected_without_privilege(
    tmp_path, checkpoint, monkeypatch, operation
):
    target = tmp_path / "archive"
    save_assistant_session(target, checkpoint)
    original = Path.lstat

    def lstat(path, *args, **kwargs):
        info = original(path, *args, **kwargs)
        if path == target:
            return SimpleNamespace(
                st_mode=info.st_mode, st_nlink=info.st_nlink, st_file_attributes=0x400
            )
        return info

    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(AssistantSessionError, match="Redirected"):
        if operation == "load":
            load_assistant_session(target)
        else:
            save_assistant_session(target, checkpoint, overwrite=True)


def test_zip_symlink_member_is_not_a_blob(tmp_path, checkpoint):
    target = tmp_path / "archive"
    save_assistant_session(target, checkpoint)
    members = _members(target)
    with zipfile.ZipFile(target, "w") as stream:
        for name, payload in members:
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            stream.writestr(info, payload)
    with pytest.raises(AssistantSessionError, match="unsupported"):
        load_assistant_session(target)


def test_size_guards_and_relative_paths_fail_before_io(tmp_path, monkeypatch):
    with pytest.raises(AssistantSessionError, match="absolute"):
        save_assistant_session("relative", make_checkpoint({}, {}))
    monkeypatch.setattr(archive_module, "MAX_DOCUMENT_BYTES", 10)
    with pytest.raises(AssistantSessionError, match="limit"):
        make_checkpoint({"text": "a" * 12}, {})
    monkeypatch.setattr(archive_module, "MAX_BLOBS", 0)
    array = _npy(np.zeros(2, dtype=np.uint8))
    with pytest.raises(AssistantSessionError, match="bounded mapping"):
        make_checkpoint({}, {_name(array): array})


def test_oversized_npy_header_is_rejected_before_numpy_parser(monkeypatch):
    payload = b"\x93NUMPY\x02\x00" + struct.pack("<I", 2**30) + b"{}\n"
    monkeypatch.setattr(
        np.lib.format,
        "read_magic",
        lambda *_: pytest.fail("Header bound must precede NumPy parser"),
    )
    with pytest.raises(AssistantSessionError, match="header exceeds"):
        make_checkpoint({}, {_name(payload): payload})


def test_passive_loader_rejects_object_blob_even_with_matching_manifest_hash(tmp_path):
    payload = _npy(np.array([object()], dtype=object))
    name = _name(payload)
    document = b"{}"
    manifest = {
        "kind": "napari-vipp-assistant-session-container",
        "version": 1,
        "session": {
            "name": "session.json",
            "byte_count": 2,
            "sha256": hashlib.sha256(document).hexdigest(),
        },
        "blobs": [
            {
                "name": name,
                "byte_count": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
        "payload_bytes": len(payload) + 2,
    }
    target = tmp_path / "malicious"
    _write_zip(
        target,
        [
            ("manifest.json", json.dumps(manifest).encode()),
            ("session.json", document),
            (name, payload),
        ],
    )
    with pytest.raises(AssistantSessionError):
        load_assistant_session(target)


def test_mismatched_declared_zip_entry_count_is_rejected(tmp_path, checkpoint):
    target = tmp_path / "archive"
    save_assistant_session(target, checkpoint)
    payload = bytearray(target.read_bytes())
    struct.pack_into("<HH", payload, len(payload) - 14, 2, 2)
    target.write_bytes(payload)
    with pytest.raises(AssistantSessionError, match="entries"):
        load_assistant_session(target)


def test_total_archive_limit_failure_never_publishes_target(tmp_path, monkeypatch):
    checkpoint = make_checkpoint({}, {})
    monkeypatch.setattr(archive_module, "MAX_ARCHIVE_BYTES", 10)
    target = tmp_path / "too-small"
    with pytest.raises(AssistantSessionError, match="total byte limit"):
        save_assistant_session(target, checkpoint)
    assert not target.exists()
    assert not list(tmp_path.glob(".*.tmp"))
