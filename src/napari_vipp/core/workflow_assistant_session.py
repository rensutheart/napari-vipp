"""Passive, bounded local archives for typed workflow-assistant checkpoints.

This container is not a session factory. Its consumer must validate semantic
state and acquire fresh source/graph ownership and sharing consent. No archive
member is extracted, imported as Python, unpickled, or given execution authority.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
import struct
import uuid
import zipfile
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .atomic_io import atomic_replace

MAX_DOCUMENT_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_BLOBS = 1024
MAX_ARRAY_BYTES = 64 * 1024 * 1024
MAX_ARRAY_HEADER_BYTES = 16 * 1024
MAX_PNG_BYTES = 16 * 1024 * 1024
MAX_PNG_PIXELS = 4096 * 4096
MAX_MANIFEST_BYTES = 256 * 1024
_MAX_CENTRAL_BYTES = 512 * 1024
_NAME = re.compile(r"(arrays|previews)/([a-f0-9]{64})\.(npy|png)\Z")
_SHA = re.compile(r"[a-f0-9]{64}\Z")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class AssistantSessionError(ValueError):
    """Invalid, unsafe, unsupported or incompletely written passive archive."""


def _json_bytes(value: object, maximum: int) -> bytes:
    remaining = 1_000_000

    def visit(item: object, depth: int = 0) -> None:
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 64:
            raise AssistantSessionError(
                "Saved-session JSON structure exceeds its limit."
            )
        if item is None or type(item) in (str, int, bool):
            if type(item) is str and len(item) > maximum:
                raise AssistantSessionError("Saved-session text exceeds its limit.")
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                visit(child, depth + 1)
            return
        if type(item) is dict:
            for key, child in item.items():
                if type(key) is not str:
                    raise AssistantSessionError("Saved-session JSON keys must be text.")
                visit(key, depth + 1)
                visit(child, depth + 1)
            return
        raise AssistantSessionError("Saved sessions accept finite JSON values only.")

    if type(value) is not dict:
        raise AssistantSessionError("Saved-session document must be a JSON object.")
    visit(value)
    try:
        payload = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (ValueError, TypeError, RecursionError) as exc:
        raise AssistantSessionError(
            "Saved-session document is not finite JSON."
        ) from exc
    if not 0 < len(payload) <= maximum:
        raise AssistantSessionError("Saved-session JSON exceeds its byte limit.")
    return payload


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AssistantSessionError("Saved-session JSON contains duplicate keys.")
        result[key] = value
    return result


def _decode(payload: bytes, maximum: int) -> dict[str, Any]:
    if type(payload) is not bytes or not 0 < len(payload) <= maximum:
        raise AssistantSessionError("Saved-session JSON byte count is invalid.")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(
                AssistantSessionError("Saved-session JSON must contain finite numbers.")
            ),
        )
        _json_bytes(value, maximum)
        return value
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise AssistantSessionError("Saved-session JSON is malformed.") from exc


def _blob_kind(name: object) -> str:
    match = _NAME.fullmatch(name) if type(name) is str else None
    if match is None or (match[1], match[3]) not in {
        ("arrays", "npy"),
        ("previews", "png"),
    }:
        raise AssistantSessionError(
            "Use only content-hashed array or preview blob names."
        )
    return match[1]


def _check_npy(payload: bytes) -> None:
    if len(payload) > MAX_ARRAY_BYTES + MAX_ARRAY_HEADER_BYTES:
        raise AssistantSessionError("Saved array exceeds its byte limit.")
    if len(payload) < 10 or payload[:6] != b"\x93NUMPY":
        raise AssistantSessionError("Saved array magic/header is invalid.")
    if payload[6:8] == b"\x01\x00":
        prefix, header_size = 10, struct.unpack_from("<H", payload, 8)[0]
    elif payload[6:8] == b"\x02\x00" and len(payload) >= 12:
        prefix, header_size = 12, struct.unpack_from("<I", payload, 8)[0]
    else:
        raise AssistantSessionError("Saved arrays require NPY format 1.0 or 2.0.")
    if (
        not 0 < header_size <= MAX_ARRAY_HEADER_BYTES - prefix
        or prefix + header_size > len(payload)
    ):
        raise AssistantSessionError("Saved array header exceeds its fixed limit.")
    stream = io.BytesIO(payload)
    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(
                stream, max_header_size=MAX_ARRAY_HEADER_BYTES
            )
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(
                stream, max_header_size=MAX_ARRAY_HEADER_BYTES
            )
        else:
            raise AssistantSessionError("Saved arrays require NPY format 1.0 or 2.0.")
        if (
            type(shape) is not tuple
            or len(shape) > 8
            or any(type(size) is not int or size < 0 for size in shape)
            or type(fortran) is not bool
            or dtype.hasobject
            or dtype.fields is not None
            or dtype.subdtype is not None
            or dtype.kind not in "biuf"
            or not 1 <= dtype.itemsize <= 16
        ):
            raise AssistantSessionError("Saved array shape/dtype is not supported.")
        byte_count = math.prod(shape) * dtype.itemsize
        if (
            byte_count > MAX_ARRAY_BYTES
            or stream.tell() > MAX_ARRAY_HEADER_BYTES
            or stream.tell() + byte_count != len(payload)
        ):
            raise AssistantSessionError(
                "Saved array header/size/trailing bytes are invalid."
            )
    except (ValueError, TypeError, EOFError, OSError, OverflowError) as exc:
        raise AssistantSessionError(
            "Saved array is not a bounded, non-pickle NPY."
        ) from exc


def _check_png(payload: bytes) -> None:
    if not 45 <= len(payload) <= MAX_PNG_BYTES or not payload.startswith(
        _PNG_SIGNATURE
    ):
        raise AssistantSessionError("Saved preview is not a bounded PNG.")
    # Inspect dimensions and chunk boundaries before a decoder allocates pixels.
    offset, first, ended, idat = 8, True, False, False
    width = height = 0
    allowed = {b"IHDR", b"PLTE", b"tRNS", b"IDAT", b"IEND", b"sRGB", b"gAMA", b"cHRM"}
    while offset < len(payload):
        if offset + 12 > len(payload):
            raise AssistantSessionError("Saved PNG chunk is truncated.")
        size, kind = struct.unpack_from("!I4s", payload, offset)
        end = offset + 12 + size
        if end > len(payload) or kind not in allowed or ended:
            raise AssistantSessionError("Saved PNG contains an unsupported chunk.")
        data = payload[offset + 8 : end - 4]
        crc = struct.unpack_from("!I", payload, end - 4)[0]
        if zlib.crc32(kind + data) & 0xFFFFFFFF != crc:
            raise AssistantSessionError("Saved PNG checksum is invalid.")
        if first:
            if kind != b"IHDR" or size != 13:
                raise AssistantSessionError("Saved PNG must begin with one IHDR.")
            width, height, depth, color, compression, filtering, interlace = (
                struct.unpack("!2I5B", data)
            )
            if (
                not 0 < width <= 4096
                or not 0 < height <= 4096
                or width * height > MAX_PNG_PIXELS
                or compression != 0
                or filtering != 0
                or interlace != 0
                or depth not in {1, 2, 4, 8, 16}
                or color not in {0, 2, 3, 4, 6}
            ):
                raise AssistantSessionError(
                    "Saved PNG dimensions/encoding exceed the bound."
                )
        elif kind == b"IHDR":
            raise AssistantSessionError("Saved PNG contains duplicate IHDR.")
        if kind == b"IDAT":
            idat = True
        if kind == b"IEND":
            if size != 0 or end != len(payload):
                raise AssistantSessionError("Saved PNG has trailing data.")
            ended = True
        first, offset = False, end
    if not ended or not idat:
        raise AssistantSessionError("Saved PNG is incomplete.")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.format != "PNG" or image.size != (width, height):
                raise AssistantSessionError("Saved preview decoder identity differs.")
            image.load()
    except (
        ValueError,
        OSError,
        SyntaxError,
        struct.error,
        IndexError,
        zlib.error,
    ) as exc:
        raise AssistantSessionError(
            "Saved preview pixels could not be decoded."
        ) from exc


def _blob(name: object, payload: object) -> tuple[str, bytes]:
    kind = _blob_kind(name)
    if type(payload) is not bytes:
        raise AssistantSessionError("Saved blobs must be immutable bytes.")
    if hashlib.sha256(payload).hexdigest() != name.split("/")[1].split(".")[0]:
        raise AssistantSessionError(
            "Saved blob bytes do not match their filename hash."
        )
    if kind == "arrays":
        _check_npy(payload)
    else:
        _check_png(payload)
    return name, payload


@dataclass(frozen=True)
class SavedAssistantSession:
    _document: bytes
    _blobs: tuple[tuple[str, bytes], ...]

    def __post_init__(self):
        document = _decode(self._document, MAX_DOCUMENT_BYTES)
        if _json_bytes(document, MAX_DOCUMENT_BYTES) != self._document:
            raise AssistantSessionError(
                "Saved-session JSON must use canonical encoding."
            )
        if type(self._blobs) is not tuple or len(self._blobs) > MAX_BLOBS:
            raise AssistantSessionError(
                "Saved-session blob collection exceeds its limit."
            )
        total, names = len(self._document), set()
        for row in self._blobs:
            if type(row) is not tuple or len(row) != 2:
                raise AssistantSessionError("Saved-session blob record is invalid.")
            name, payload = _blob(*row)
            if name in names:
                raise AssistantSessionError("Saved-session blob names must be unique.")
            names.add(name)
            total += len(payload)
            if total > MAX_ARCHIVE_BYTES:
                raise AssistantSessionError(
                    "Saved-session payload exceeds its total limit."
                )
        if tuple(sorted(names)) != self.blob_names:
            raise AssistantSessionError(
                "Saved-session blobs must have canonical name order."
            )

    @property
    def document(self) -> dict[str, Any]:
        return json.loads(self._document)

    @property
    def blob_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self._blobs)

    def blob_bytes(self, reference: str) -> bytes:
        _blob_kind(reference)
        for name, payload in self._blobs:
            if name == reference:
                return payload
        raise AssistantSessionError("Saved-session blob reference is missing.")


@dataclass(frozen=True)
class AssistantSessionReceipt:
    path: Path
    sha256: str
    byte_count: int
    blob_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "blob_count": self.blob_count,
        }


def make_checkpoint(
    document: dict, blobs: Mapping[str, bytes]
) -> SavedAssistantSession:
    if not isinstance(blobs, Mapping) or len(blobs) > MAX_BLOBS:
        raise AssistantSessionError(
            "Saved-session blobs need an explicit bounded mapping."
        )
    rows = []
    for name, payload in blobs.items():
        if type(name) is not str:
            raise AssistantSessionError("Saved-session blob names must be text.")
        rows.append((name, payload))
    return SavedAssistantSession(
        _json_bytes(document, MAX_DOCUMENT_BYTES), tuple(sorted(rows))
    )


def _safe_path(path: Path, *, directory: bool | None = None) -> None:
    for component in (*reversed(path.parents), path):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise AssistantSessionError(
                "Redirected saved-session paths are unsupported."
            )
        if component != path and not stat.S_ISDIR(info.st_mode):
            raise AssistantSessionError("Saved-session parent must be a directory.")
        if component == path:
            if directory is True and not stat.S_ISDIR(info.st_mode):
                raise AssistantSessionError(
                    "Saved-session directory is not a directory."
                )
            if directory is False and (
                not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            ):
                raise AssistantSessionError(
                    "Saved session requires a single regular file."
                )


def _absolute(path: str | Path) -> Path:
    target = Path(path)
    if not target.is_absolute():
        raise AssistantSessionError("Saved-session path must be explicitly absolute.")
    _safe_path(target, directory=False)
    return target


def _sync_directory(path: Path) -> None:
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _manifest(checkpoint: SavedAssistantSession) -> bytes:
    return _json_bytes(
        {
            "kind": "napari-vipp-assistant-session-container",
            "version": 1,
            "session": {
                "name": "session.json",
                "byte_count": len(checkpoint._document),
                "sha256": hashlib.sha256(checkpoint._document).hexdigest(),
            },
            "blobs": [
                {
                    "name": name,
                    "byte_count": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
                for name, payload in checkpoint._blobs
            ],
            "payload_bytes": len(checkpoint._document)
            + sum(len(data) for _, data in checkpoint._blobs),
        },
        MAX_MANIFEST_BYTES,
    )


def _write_member(archive, name, payload):
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o600) << 16
    archive.writestr(info, payload)


def _file_receipt(path: Path, blob_count: int) -> AssistantSessionReceipt:
    _safe_path(path, directory=False)
    digest, count = hashlib.sha256(), 0
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= MAX_ARCHIVE_BYTES
        ):
            raise AssistantSessionError("Saved archive exceeds its byte limit.")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while block := stream.read(1024 * 1024):
                count += len(block)
                if count > MAX_ARCHIVE_BYTES:
                    raise AssistantSessionError("Saved archive grew during hashing.")
                digest.update(block)
        _safe_path(path, directory=False)
        after = path.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or count != before.st_size:
            raise AssistantSessionError("Saved archive changed during hashing.")
        return AssistantSessionReceipt(path, digest.hexdigest(), count, blob_count)
    finally:
        os.close(descriptor)


def save_assistant_session(
    path: str | Path, checkpoint: SavedAssistantSession, *, overwrite: bool = False
) -> AssistantSessionReceipt:
    """Atomically save an explicit local file; never overwrite by default."""
    if type(checkpoint) is not SavedAssistantSession or type(overwrite) is not bool:
        raise AssistantSessionError(
            "Use a validated checkpoint and explicit overwrite choice."
        )
    checkpoint.__post_init__()
    target = _absolute(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Saved-session file already exists: {target}")
    _safe_path(target.parent, directory=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    _safe_path(target.parent, directory=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            with zipfile.ZipFile(
                stream, "w", compression=zipfile.ZIP_STORED, allowZip64=False
            ) as archive:
                _write_member(archive, "manifest.json", _manifest(checkpoint))
                _write_member(archive, "session.json", checkpoint._document)
                for name, payload in checkpoint._blobs:
                    _write_member(archive, name, payload)
            stream.flush()
            if stream.tell() > MAX_ARCHIVE_BYTES:
                raise AssistantSessionError(
                    "Saved archive exceeds its total byte limit."
                )
            os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        _safe_path(temporary, directory=False)
        _safe_path(target, directory=False)
        if overwrite:
            atomic_replace(temporary, target)
        else:
            _publish_no_replace(temporary, target)
        _sync_directory(target.parent)
        return _file_receipt(target, len(checkpoint._blobs))
    finally:
        if descriptor != -1:
            os.close(descriptor)
        _safe_path(temporary, directory=False)
        temporary.unlink(missing_ok=True)


def _publish_no_replace(temporary: Path, target: Path) -> None:
    # Windows rename refuses an existing destination. POSIX rename does not,
    # so there use adjacent atomic link publication and unlink our temporary.
    # Neither route has an exists-check/replace race that can clobber user bytes.
    if os.name == "nt":
        os.rename(temporary, target)
        return
    owned = temporary.stat()
    os.link(temporary, target, follow_symlinks=False)
    try:
        temporary.unlink()
    except OSError:
        # If temporary retirement fails, roll back only the exact new inode
        # published by this call, never a replacement that appeared later.
        current = target.lstat()
        if (
            stat.S_ISREG(current.st_mode)
            and current.st_nlink == 2
            and (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino)
        ):
            target.unlink()
        raise


def _preflight_zip(stream, size: int) -> tuple[int, int]:
    if not 22 <= size <= MAX_ARCHIVE_BYTES:
        raise AssistantSessionError("Saved archive size is outside its limit.")
    stream.seek(size - 22)
    tail = stream.read(22)
    (
        signature,
        disk,
        start_disk,
        on_disk,
        count,
        central_size,
        central_offset,
        comment,
    ) = struct.unpack("<4s4H2LH", tail)
    if (
        signature != b"PK\x05\x06"
        or disk
        or start_disk
        or on_disk != count
        or not 2 <= count <= MAX_BLOBS + 2
        or comment
        or central_size > _MAX_CENTRAL_BYTES
        or central_offset + central_size + 22 != size
    ):
        raise AssistantSessionError(
            "Saved archive directory/count/format is unsupported."
        )
    return central_offset, count


def _zip_infos(stream, archive, central_offset, expected_count):
    infos = archive.infolist()
    if not 2 <= len(infos) <= MAX_BLOBS + 2 or len(infos) != expected_count:
        raise AssistantSessionError("Saved archive has too many entries.")
    names, ranges, total = set(), [], 0
    for info in infos:
        name = info.filename
        if name not in {"manifest.json", "session.json"}:
            _blob_kind(name)
        if (
            name in names
            or info.is_dir()
            or info.compress_type != zipfile.ZIP_STORED
            or info.compress_size != info.file_size
            or info.flag_bits != 0
            or info.extra
            or info.comment
            or info.extract_version > 20
            or stat.S_IFMT(info.external_attr >> 16) not in {0, stat.S_IFREG}
        ):
            raise AssistantSessionError(
                "Saved archive entry is duplicate or unsupported."
            )
        names.add(name)
        maximum = (
            MAX_MANIFEST_BYTES
            if name == "manifest.json"
            else MAX_DOCUMENT_BYTES
            if name == "session.json"
            else MAX_PNG_BYTES
            if name.startswith("previews/")
            else MAX_ARRAY_BYTES + MAX_ARRAY_HEADER_BYTES
        )
        if (
            not 0 < info.file_size <= maximum
            or not 0 <= info.header_offset < central_offset
        ):
            raise AssistantSessionError("Saved archive entry exceeds its fixed limit.")
        total += info.file_size
        if total > MAX_ARCHIVE_BYTES:
            raise AssistantSessionError(
                "Saved archive payload exceeds its total limit."
            )
        stream.seek(info.header_offset)
        header = stream.read(30)
        if len(header) != 30:
            raise AssistantSessionError("Saved archive local header is truncated.")
        (
            sig,
            version,
            flags,
            method,
            _,
            _,
            crc,
            compressed,
            raw,
            name_size,
            extra_size,
        ) = struct.unpack("<4s5H3L2H", header)
        try:
            encoded = name.encode("ascii")
        except UnicodeError as exc:
            raise AssistantSessionError(
                "Saved archive entry names must be ASCII."
            ) from exc
        if (
            sig != b"PK\x03\x04"
            or version > 20
            or flags
            or method
            or crc != info.CRC
            or compressed != info.file_size
            or raw != info.file_size
            or name_size != len(encoded)
            or extra_size
            or stream.read(name_size) != encoded
        ):
            raise AssistantSessionError(
                "Saved archive local and central headers disagree."
            )
        ranges.append(
            (info.header_offset, info.header_offset + 30 + name_size + info.file_size)
        )
    if infos[0].filename != "manifest.json" or infos[1].filename != "session.json":
        raise AssistantSessionError(
            "Saved archive must start with manifest/session JSON."
        )
    end = 0
    for start, stop in sorted(ranges):
        if start != end or stop > central_offset:
            raise AssistantSessionError(
                "Saved archive contains hidden or overlapping bytes."
            )
        end = stop
    if end != central_offset:
        raise AssistantSessionError("Saved archive contains unaccounted data.")
    return infos


def _manifest_records(manifest, names):
    if (
        type(manifest) is not dict
        or set(manifest) != {"kind", "version", "session", "blobs", "payload_bytes"}
        or manifest["kind"] != "napari-vipp-assistant-session-container"
        or type(manifest["version"]) is not int
        or manifest["version"] != 1
        or type(manifest["blobs"]) is not list
        or len(manifest["blobs"]) > MAX_BLOBS
        or type(manifest["payload_bytes"]) is not int
    ):
        raise AssistantSessionError("Saved-session manifest schema is unsupported.")
    rows = [manifest["session"], *manifest["blobs"]]
    seen, total = set(), 0
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != {"name", "byte_count", "sha256"}:
            raise AssistantSessionError("Saved-session manifest record is invalid.")
        name = row["name"]
        if index == 0:
            if name != "session.json":
                raise AssistantSessionError(
                    "Saved-session manifest document reference is invalid."
                )
        else:
            _blob_kind(name)
        if (
            name in seen
            or type(row["byte_count"]) is not int
            or row["byte_count"] <= 0
            or type(row["sha256"]) is not str
            or not _SHA.fullmatch(row["sha256"])
        ):
            raise AssistantSessionError("Saved-session manifest hash/size is invalid.")
        seen.add(name)
        total += row["byte_count"]
    if (
        seen != names - {"manifest.json"}
        or total != manifest["payload_bytes"]
        or total > MAX_ARCHIVE_BYTES
    ):
        raise AssistantSessionError(
            "Saved-session manifest inventory/total does not match."
        )
    return rows


def load_assistant_session(path: str | Path) -> SavedAssistantSession:
    """Read verified bytes only; never bind a source, resume work, or share data."""
    source = _absolute(path)
    descriptor = os.open(
        source, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise AssistantSessionError("Saved archive must be a single regular file.")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            central_offset, count = _preflight_zip(stream, before.st_size)
            stream.seek(0)
            with zipfile.ZipFile(stream, "r", allowZip64=False) as archive:
                infos = _zip_infos(stream, archive, central_offset, count)
                by_name = {info.filename: info for info in infos}
                manifest_payload = archive.read(by_name["manifest.json"])
                manifest = _decode(manifest_payload, MAX_MANIFEST_BYTES)
                rows = _manifest_records(manifest, set(by_name))
                payloads = {}
                for row in rows:
                    info = by_name[row["name"]]
                    if info.file_size != row["byte_count"]:
                        raise AssistantSessionError(
                            "Saved archive manifest byte count differs."
                        )
                    payload = archive.read(info)
                    if (
                        len(payload) != row["byte_count"]
                        or hashlib.sha256(payload).hexdigest() != row["sha256"]
                    ):
                        raise AssistantSessionError(
                            "Saved archive content hash differs."
                        )
                    payloads[row["name"]] = payload
        _safe_path(source, directory=False)
        after = source.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise AssistantSessionError("Saved archive changed while it was read.")
        session = payloads.pop("session.json")
        return SavedAssistantSession(session, tuple(sorted(payloads.items())))
    except (zipfile.BadZipFile, EOFError, struct.error, UnicodeError) as exc:
        raise AssistantSessionError("Saved archive is malformed or corrupt.") from exc
    finally:
        os.close(descriptor)


__all__ = [
    "AssistantSessionError",
    "AssistantSessionReceipt",
    "SavedAssistantSession",
    "make_checkpoint",
    "save_assistant_session",
    "load_assistant_session",
]
