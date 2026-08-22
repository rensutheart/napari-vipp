"""Rewrite a source distribution as a deterministic, safe ``.tar.gz``.

The build backend can emit semantically identical source archives with
different tar-member or gzip timestamps.  Release builds pass their immutable
commit timestamp through ``SOURCE_DATE_EPOCH`` and run this script before
comparing or publishing artifacts.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import os
import re
import stat
import sys
import tarfile
import tempfile
import zlib
from collections.abc import Sequence
from pathlib import Path

_MAX_GZIP_MTIME = (1 << 32) - 1
_SAFE_MEMBER_TYPES = frozenset({tarfile.REGTYPE, tarfile.DIRTYPE})
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")
_WINDOWS_RESERVED_NAMES = frozenset(
    {"aux", "con", "nul", "prn"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)
_SAFE_PAX_KEYS = frozenset({"mtime"})


class SdistCanonicalizationError(ValueError):
    """Raised when an sdist cannot be canonicalized safely."""


def source_date_epoch_from_environment() -> int:
    """Return a gzip-compatible ``SOURCE_DATE_EPOCH`` from the environment."""

    raw_value = os.environ.get("SOURCE_DATE_EPOCH")
    if raw_value is None:
        raise SdistCanonicalizationError(
            "SOURCE_DATE_EPOCH is required for deterministic sdist output"
        )
    if not re.fullmatch(r"[0-9]+", raw_value):
        raise SdistCanonicalizationError(
            "SOURCE_DATE_EPOCH must be an unsigned decimal integer"
        )
    value = int(raw_value)
    if value > _MAX_GZIP_MTIME:
        raise SdistCanonicalizationError(
            f"SOURCE_DATE_EPOCH must not exceed {_MAX_GZIP_MTIME}"
        )
    return value


def _member_path_parts(name: str) -> tuple[str, ...]:
    if not name or "\0" in name:
        raise SdistCanonicalizationError("archive members must have a name")
    if name.startswith("/") or "\\" in name or _WINDOWS_DRIVE.match(name):
        raise SdistCanonicalizationError(
            f"archive member path is not a safe relative POSIX path: {name!r}"
        )

    candidate = name[:-1] if name.endswith("/") else name
    parts = tuple(candidate.split("/"))
    if not candidate or any(part in {"", ".", ".."} for part in parts):
        raise SdistCanonicalizationError(
            f"archive member path is not canonical and relative: {name!r}"
        )
    for part in parts:
        windows_stem = part.split(".", maxsplit=1)[0].casefold()
        if (
            ":" in part
            or part.endswith((" ", "."))
            or windows_stem in _WINDOWS_RESERVED_NAMES
            or any(ord(character) < 32 for character in part)
        ):
            raise SdistCanonicalizationError(
                f"archive member path is not portable to Windows: {name!r}"
            )
    return parts


def _validate_pax_headers(member: tarfile.TarInfo) -> None:
    unexpected = set(member.pax_headers) - _SAFE_PAX_KEYS
    if unexpected:
        raise SdistCanonicalizationError(
            f"archive member {member.name!r} has unsupported PAX metadata: "
            + ", ".join(sorted(unexpected))
        )


def _validate_members(members: Sequence[tarfile.TarInfo]) -> None:
    if not members:
        raise SdistCanonicalizationError("source archive is empty")

    archive_root: str | None = None
    normalized_names: dict[str, str] = {}
    normalized_members: dict[str, tarfile.TarInfo] = {}
    for member in members:
        parts = _member_path_parts(member.name)
        _validate_pax_headers(member)
        if archive_root is None:
            archive_root = parts[0]
        elif parts[0] != archive_root:
            raise SdistCanonicalizationError(
                "source archive must contain exactly one top-level root; "
                f"found {archive_root!r} and {parts[0]!r}"
            )
        normalized_name = "/".join(parts)
        collision_key = normalized_name.casefold()
        if collision_key in normalized_names:
            raise SdistCanonicalizationError(
                "source archive contains duplicate or case-colliding members: "
                f"{normalized_names[collision_key]!r} and {member.name!r}"
            )
        normalized_names[collision_key] = member.name
        normalized_members[normalized_name] = member
        if member.name.endswith("/") and not member.isdir():
            raise SdistCanonicalizationError(
                f"non-directory archive member ends in '/': {member.name!r}"
            )
        if member.type not in _SAFE_MEMBER_TYPES:
            raise SdistCanonicalizationError(
                f"archive member {member.name!r} has unsafe type {member.type!r}"
            )

    assert archive_root is not None
    root_member = normalized_members.get(archive_root)
    if root_member is None or not root_member.isdir():
        raise SdistCanonicalizationError(
            "source archive root must be present as an explicit directory member"
        )
    for member in members:
        member_parts = _member_path_parts(member.name)
        for depth in range(1, len(member_parts)):
            ancestor = normalized_members.get("/".join(member_parts[:depth]))
            if ancestor is not None and not ancestor.isdir():
                raise SdistCanonicalizationError(
                    f"archive member {member.name!r} descends through non-directory "
                    f"member {ancestor.name!r}"
                )


def _canonical_member(member: tarfile.TarInfo, epoch: int) -> tarfile.TarInfo:
    canonical = copy.copy(member)
    canonical.mtime = epoch
    canonical.pax_headers = {}
    return canonical


def _regular_payload_hashes(
    archive: tarfile.TarFile,
    members: Sequence[tarfile.TarInfo],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for member in members:
        if not member.isreg():
            continue
        file_object = archive.extractfile(member)
        if file_object is None:
            raise SdistCanonicalizationError(
                f"could not read archive member {member.name!r}"
            )
        digest = hashlib.sha256()
        total_size = 0
        with file_object:
            while chunk := file_object.read(1024 * 1024):
                digest.update(chunk)
                total_size += len(chunk)
        if total_size != member.size:
            raise SdistCanonicalizationError(
                f"archive member {member.name!r} is truncated"
            )
        hashes[member.name] = digest.hexdigest()
    return hashes


def _validate_single_gzip_stream(path: Path) -> None:
    decompressor = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
    with path.open("rb") as raw_archive:
        while chunk := raw_archive.read(1024 * 1024):
            decompressor.decompress(chunk)
            if decompressor.eof:
                if decompressor.unused_data or raw_archive.read(1):
                    raise SdistCanonicalizationError(
                        "gzip archive contains trailing bytes or another gzip member"
                    )
                break
    if not decompressor.eof:
        raise SdistCanonicalizationError(
            "gzip archive contains a truncated gzip member"
        )


def _validate_canonical_output(
    path: Path,
    *,
    source_members: Sequence[tarfile.TarInfo],
    source_payload_hashes: dict[str, str],
    epoch: int,
) -> None:
    with path.open("rb") as raw_archive:
        header = raw_archive.read(10)
    if (
        len(header) != 10
        or header[:3] != b"\x1f\x8b\x08"
        or header[3] & gzip.FNAME
        or int.from_bytes(header[4:8], "little") != epoch
    ):
        raise SdistCanonicalizationError(
            "canonical output has a nondeterministic gzip header"
        )

    _validate_single_gzip_stream(path)
    with tarfile.open(path, mode="r:gz") as archive:
        output_members = archive.getmembers()
        _validate_members(output_members)
        expected = source_members
        if len(output_members) != len(expected):
            raise SdistCanonicalizationError(
                "canonical output member count differs from its source"
            )
        for actual, original in zip(output_members, expected, strict=True):
            if (
                actual.name != original.name
                or actual.type != original.type
                or actual.mode != original.mode
                or actual.uid != original.uid
                or actual.gid != original.gid
                or actual.uname != original.uname
                or actual.gname != original.gname
                or actual.linkname != original.linkname
                or actual.size != original.size
            ):
                raise SdistCanonicalizationError(
                    f"canonical output changed archive member {original.name!r}"
                )
            if actual.mtime != epoch or actual.pax_headers:
                raise SdistCanonicalizationError(
                    f"canonical metadata validation failed for {actual.name!r}"
                )
            if actual.isreg():
                file_object = archive.extractfile(actual)
                if file_object is None:
                    raise SdistCanonicalizationError(
                        f"canonical output truncated {actual.name!r}"
                    )
                digest = hashlib.sha256()
                with file_object:
                    while chunk := file_object.read(1024 * 1024):
                        digest.update(chunk)
                if digest.hexdigest() != source_payload_hashes[actual.name]:
                    raise SdistCanonicalizationError(
                        f"canonical output changed payload {actual.name!r}"
                    )


def canonicalize_sdist(
    archive: Path | str,
    *,
    source_date_epoch: int,
    output: Path | str | None = None,
) -> Path:
    """Canonicalize ``archive`` atomically and return the output path.

    When ``output`` is omitted, the input archive is replaced in place.  The
    caller supplies an already validated ``SOURCE_DATE_EPOCH`` value when using
    this library entry point directly.
    """

    if not isinstance(source_date_epoch, int) or isinstance(source_date_epoch, bool):
        raise SdistCanonicalizationError("source_date_epoch must be an integer")
    if not 0 <= source_date_epoch <= _MAX_GZIP_MTIME:
        raise SdistCanonicalizationError(
            f"source_date_epoch must be between 0 and {_MAX_GZIP_MTIME}"
        )

    source_path = Path(archive)
    output_path = source_path if output is None else Path(output)
    if not source_path.is_file():
        raise SdistCanonicalizationError(
            f"source archive does not exist: {source_path}"
        )
    if not output_path.parent.is_dir():
        raise SdistCanonicalizationError(
            f"output directory does not exist: {output_path.parent}"
        )

    temporary_fd, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        try:
            _validate_single_gzip_stream(source_path)
            with tarfile.open(source_path, mode="r:gz") as source:
                if source.pax_headers:
                    raise SdistCanonicalizationError(
                        "source archive has unsupported global PAX metadata"
                    )
                members = source.getmembers()
                _validate_members(members)
                source_payload_hashes = _regular_payload_hashes(source, members)
                with os.fdopen(temporary_fd, "wb") as raw_output:
                    temporary_fd = -1
                    with gzip.GzipFile(
                        filename="",
                        mode="wb",
                        fileobj=raw_output,
                        mtime=source_date_epoch,
                    ) as compressed_output:
                        with tarfile.open(
                            fileobj=compressed_output,
                            mode="w",
                            format=tarfile.PAX_FORMAT,
                        ) as target:
                            for member in members:
                                canonical = _canonical_member(member, source_date_epoch)
                                file_object = (
                                    source.extractfile(member)
                                    if member.isreg()
                                    else None
                                )
                                if member.isreg() and file_object is None:
                                    raise SdistCanonicalizationError(
                                        f"could not read archive member {member.name!r}"
                                    )
                                try:
                                    target.addfile(canonical, file_object)
                                finally:
                                    if file_object is not None:
                                        file_object.close()
                    raw_output.flush()
                    os.fsync(raw_output.fileno())
            _validate_canonical_output(
                temporary_path,
                source_members=members,
                source_payload_hashes=source_payload_hashes,
                epoch=source_date_epoch,
            )
        except (OSError, tarfile.TarError, zlib.error) as error:
            raise SdistCanonicalizationError(
                f"could not canonicalize {source_path}: {error}"
            ) from error

        os.chmod(
            temporary_path,
            stat.S_IMODE(source_path.stat().st_mode),
        )
        os.replace(temporary_path, output_path)
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        temporary_path.unlink(missing_ok=True)
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Input .tar.gz source archive")
    parser.add_argument(
        "--output",
        type=Path,
        help="Write atomically to this path instead of replacing the input",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        epoch = source_date_epoch_from_environment()
        result = canonicalize_sdist(
            arguments.archive,
            source_date_epoch=epoch,
            output=arguments.output,
        )
    except SdistCanonicalizationError as error:
        parser.error(str(error))
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
