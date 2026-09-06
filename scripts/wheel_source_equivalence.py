"""Compare clean-source wheels without changing their exact payload identities.

Pinned setuptools writes generated METADATA with platform-native newlines.
Only that file's CRLF/LF representation and its valid RECORD hash/size may
differ here. Raw wheel/content hashes used for distribution and frozen-payload
provenance must continue to include every original byte, including RECORD.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import re
import stat
import zipfile
from pathlib import Path


class WheelSourceEquivalenceError(ValueError):
    """A wheel is invalid or differs from its clean-source rebuild."""


def _read_wheel(path: Path) -> tuple[dict[str, bytes], dict[str, str], str, str]:
    payloads: dict[str, bytes] = {}
    types: dict[str, str] = {}
    folded_names: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name = info.filename
            parts = (name[:-1] if info.is_dir() else name).split("/")
            if (
                not name
                or "\\" in name
                or any(part in {"", ".", ".."} or ":" in part for part in parts)
                or any(ord(character) < 32 for character in name)
                or name.casefold() in folded_names
            ):
                raise WheelSourceEquivalenceError(
                    f"Unsafe or duplicate wheel member: {name!r}"
                )
            folded_names.add(name.casefold())
            mode_type = stat.S_IFMT(info.external_attr >> 16)
            if info.is_dir() and mode_type in {0, stat.S_IFDIR}:
                kind = "directory"
            elif not info.is_dir() and mode_type in {0, stat.S_IFREG}:
                kind = "file"
            else:
                raise WheelSourceEquivalenceError(
                    f"Unsupported wheel member type: {name!r}"
                )
            types[name] = kind
            payloads[name] = archive.read(info)
            if kind == "directory" and payloads[name]:
                raise WheelSourceEquivalenceError("Wheel directory has a payload.")

    roots = {
        name.split("/", 1)[0]
        for name in payloads
        if name.split("/", 1)[0].endswith(".dist-info")
    }
    if len(roots) != 1:
        raise WheelSourceEquivalenceError("Expected exactly one dist-info directory.")
    root = roots.pop()
    metadata_name, record_name = f"{root}/METADATA", f"{root}/RECORD"
    if any(types.get(name) != "file" for name in (metadata_name, record_name)):
        raise WheelSourceEquivalenceError("Wheel METADATA or RECORD is missing.")
    return payloads, types, metadata_name, record_name


def _validated_record(
    payloads: dict[str, bytes],
    types: dict[str, str],
    metadata_name: str,
    record_name: str,
) -> bytes:
    """Validate every row; mask only the generated METADATA hash and size."""
    record = payloads[record_name]
    rows = list(
        csv.reader(io.StringIO(record.decode("utf-8"), newline=""), strict=True)
    )
    names: set[str] = set()
    metadata_digest = metadata_size = ""
    for row in rows:
        if len(row) != 3:
            raise WheelSourceEquivalenceError("RECORD rows must have three fields.")
        name, digest, size = row
        if name in names or types.get(name) != "file":
            raise WheelSourceEquivalenceError(
                f"Duplicate or extra RECORD entry: {name!r}"
            )
        names.add(name)
        if name == record_name:
            if digest or size:
                raise WheelSourceEquivalenceError("RECORD self-entry must be unhashed.")
            continue
        # The pinned release builder writes SHA-256. Accept no weak, missing,
        # noncanonical, or unverifiable digest representation in this comparison.
        expected_digest = "sha256=" + base64.urlsafe_b64encode(
            hashlib.sha256(payloads[name]).digest()
        ).rstrip(b"=").decode("ascii")
        if digest != expected_digest or size != str(len(payloads[name])):
            raise WheelSourceEquivalenceError(
                f"RECORD hash or size does not match {name!r}."
            )
        if name == metadata_name:
            metadata_digest, metadata_size = digest, size
    if names != {name for name, kind in types.items() if kind == "file"}:
        raise WheelSourceEquivalenceError(
            "RECORD omits wheel members or its self-entry."
        )

    # Preserve RECORD byte order, line endings, quoting and every other entry.
    # Generated dist-info names cannot contain CSV escapes in our pinned build.
    # Reject those forms instead of accepting broader arbitrary RECORD changes.
    metadata_row = f"{metadata_name},{metadata_digest},{metadata_size}".encode()
    expression = rb"(?m)^" + re.escape(metadata_row) + rb"(?=\r?$)"
    masked, replacements = re.subn(
        expression, lambda _match: metadata_name.encode() + b",<hash>,<size>", record
    )
    if replacements != 1:
        raise WheelSourceEquivalenceError("Unexpected generated METADATA RECORD row.")
    return masked


def require_source_equivalent_wheels(supplied: Path, rebuilt: Path) -> None:
    """Require source equivalence; read both archives, never rewrite either."""
    try:
        left, left_types, metadata_name, record_name = _read_wheel(supplied)
        right, right_types, right_metadata, right_record = _read_wheel(rebuilt)
        if (
            left_types != right_types
            or metadata_name != right_metadata
            or record_name != right_record
        ):
            raise WheelSourceEquivalenceError("Wheel member names or types differ.")
        # Validate RECORD even when the two archives or content hashes are equal.
        left_record = _validated_record(left, left_types, metadata_name, record_name)
        right_record = _validated_record(right, right_types, metadata_name, record_name)
        if left_record != right_record:
            raise WheelSourceEquivalenceError("Wheel RECORD entries differ.")
        for name, payload in left.items():
            other = right[name]
            if name == record_name:
                continue
            if name == metadata_name:
                payload = payload.replace(b"\r\n", b"\n")
                other = other.replace(b"\r\n", b"\n")
            if payload != other:
                raise WheelSourceEquivalenceError(f"Wheel contents differ: {name}")
    except (OSError, UnicodeError, csv.Error, zipfile.BadZipFile) as error:
        raise WheelSourceEquivalenceError(
            f"Could not validate wheel: {error}"
        ) from error
