from __future__ import annotations

import gzip
import os
import struct
import subprocess
import sys
import tarfile
from io import BytesIO
from pathlib import Path

import pytest

from scripts.canonicalize_sdist import (
    SdistCanonicalizationError,
    _validate_single_gzip_stream,
    canonicalize_sdist,
)

SCRIPT = Path(__file__).parents[3] / "scripts" / "canonicalize_sdist.py"


def _member(
    name: str,
    *,
    variant: int,
    member_type: bytes = tarfile.REGTYPE,
    linkname: str = "",
    mode: int = 0o644,
    content: bytes = b"",
) -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name)
    info.type = member_type
    info.linkname = linkname
    info.mode = mode
    info.size = len(content) if member_type == tarfile.REGTYPE else 0
    info.mtime = 1_000 + variant
    info.uid = 100
    info.gid = 200
    info.uname = "release-user"
    info.gname = "release-group"
    info.pax_headers = {"mtime": str(1_000 + variant)}
    return info, content


def _write_sdist(
    path: Path,
    *,
    variant: int,
    members: list[tuple[tarfile.TarInfo, bytes]] | None = None,
) -> None:
    if members is None:
        members = [
            _member(
                "example-1.0/bin/run.py",
                variant=variant,
                mode=0o755,
                content=b"print('unchanged')\n",
            ),
            _member(
                "example-1.0/example/__init__.py",
                variant=variant,
                content=b"VALUE = 42\n",
            ),
            _member(
                "example-1.0/example",
                variant=variant,
                member_type=tarfile.DIRTYPE,
                mode=0o755,
            ),
            _member(
                "example-1.0",
                variant=variant,
                member_type=tarfile.DIRTYPE,
                mode=0o755,
            ),
        ]
    with path.open("wb") as raw_archive:
        with gzip.GzipFile(
            filename=f"variant-{variant}.tar",
            mode="wb",
            fileobj=raw_archive,
            mtime=4_000 + variant,
        ) as compressed_archive:
            with tarfile.open(
                fileobj=compressed_archive,
                mode="w",
                format=tarfile.PAX_FORMAT,
            ) as archive:
                for info, content in members:
                    archive.addfile(info, BytesIO(content) if info.isreg() else None)


def test_semantically_identical_sdists_canonicalize_byte_identically(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    _write_sdist(first, variant=1)
    _write_sdist(second, variant=2)
    assert first.read_bytes() != second.read_bytes()

    epoch = 1_725_000_123
    canonicalize_sdist(first, source_date_epoch=epoch)
    canonicalize_sdist(second, source_date_epoch=epoch)

    assert first.read_bytes() == second.read_bytes()
    raw = first.read_bytes()
    canonicalize_sdist(first, source_date_epoch=epoch)
    assert first.read_bytes() == raw
    assert raw[:3] == b"\x1f\x8b\x08"
    assert raw[3] & gzip.FNAME == 0
    assert struct.unpack("<I", raw[4:8])[0] == epoch

    with tarfile.open(first, "r:gz") as archive:
        members = archive.getmembers()
        assert [member.name for member in members] == [
            "example-1.0/bin/run.py",
            "example-1.0/example/__init__.py",
            "example-1.0/example",
            "example-1.0",
        ]
        assert all(member.mtime == epoch for member in members)
        assert all(member.uid == 100 and member.gid == 200 for member in members)
        assert all(member.uname == "release-user" for member in members)
        assert all(member.gname == "release-group" for member in members)
        assert all(member.pax_headers == {} for member in members)
        assert archive.extractfile("example-1.0/bin/run.py").read() == (
            b"print('unchanged')\n"
        )
        executable = archive.getmember("example-1.0/bin/run.py")
        assert executable.mode == 0o755


@pytest.mark.parametrize(
    ("members", "match"),
    [
        ([_member("../escape", variant=1, content=b"bad")], "relative"),
        ([_member("/absolute", variant=1, content=b"bad")], "relative"),
        (
            [
                _member("one/file", variant=1, content=b"one"),
                _member("two/file", variant=1, content=b"two"),
            ],
            "one top-level root",
        ),
        (
            [
                _member("example-1.0/Package.py", variant=1, content=b"one"),
                _member("example-1.0/package.py", variant=1, content=b"two"),
            ],
            "case-colliding",
        ),
        (
            [_member("example-1.0/CON.txt", variant=1, content=b"bad")],
            "portable to Windows",
        ),
        (
            [
                _member(
                    "example-1.0/link",
                    variant=1,
                    member_type=tarfile.SYMTYPE,
                    linkname="../../escape",
                )
            ],
            "unsafe type",
        ),
    ],
)
def test_unsafe_archives_are_rejected(
    tmp_path: Path,
    members: list[tuple[tarfile.TarInfo, bytes]],
    match: str,
) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    _write_sdist(archive, variant=2, members=members)

    with pytest.raises(SdistCanonicalizationError, match=match):
        canonicalize_sdist(archive, source_date_epoch=123)


def test_cli_requires_valid_environment_and_supports_explicit_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.tar.gz"
    output = tmp_path / "canonical.tar.gz"
    _write_sdist(source, variant=1)
    output.write_bytes(b"old output")
    original = source.read_bytes()
    environment = os.environ.copy()
    environment.pop("SOURCE_DATE_EPOCH", None)

    missing = subprocess.run(
        [sys.executable, str(SCRIPT), str(source)],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert missing.returncode == 2
    assert "SOURCE_DATE_EPOCH is required" in missing.stderr

    environment["SOURCE_DATE_EPOCH"] = "not-an-integer"
    invalid = subprocess.run(
        [sys.executable, str(SCRIPT), str(source)],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert invalid.returncode == 2
    assert "unsigned decimal integer" in invalid.stderr

    environment["SOURCE_DATE_EPOCH"] = "1725000123"
    success = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(source),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert success.returncode == 0, success.stderr
    assert output.is_file()
    assert source.read_bytes() == original
    assert str(output) in success.stdout


def test_corrupt_input_is_rejected_without_replacing_it(tmp_path: Path) -> None:
    source = tmp_path / "corrupt.tar.gz"
    output = tmp_path / "existing.tar.gz"
    source.write_bytes(b"not a gzip archive")
    output.write_bytes(b"existing output")
    original = source.read_bytes()

    with pytest.raises(SdistCanonicalizationError, match="could not canonicalize"):
        canonicalize_sdist(source, source_date_epoch=123, output=output)

    assert source.read_bytes() == original
    assert output.read_bytes() == b"existing output"


@pytest.mark.parametrize("suffix", [b"trailing", gzip.compress(b"second")])
def test_single_gzip_validation_rejects_trailing_data(
    tmp_path: Path, suffix: bytes
) -> None:
    archive = tmp_path / "multiple.gz"
    archive.write_bytes(gzip.compress(b"first", mtime=123) + suffix)

    with pytest.raises(SdistCanonicalizationError, match="trailing bytes"):
        _validate_single_gzip_stream(archive)


def test_source_with_another_gzip_member_is_rejected_unchanged(
    tmp_path: Path,
) -> None:
    source = tmp_path / "concatenated.tar.gz"
    _write_sdist(source, variant=1)
    source.write_bytes(source.read_bytes() + gzip.compress(b"unexpected"))
    original = source.read_bytes()

    with pytest.raises(SdistCanonicalizationError, match="trailing bytes"):
        canonicalize_sdist(source, source_date_epoch=123)

    assert source.read_bytes() == original


def test_unexpected_pax_metadata_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "unexpected-pax.tar.gz"
    member = _member("example-1.0/file", variant=1, content=b"payload")
    member[0].pax_headers["VENDOR.example"] = "semantic metadata"
    _write_sdist(source, variant=2, members=[member])

    with pytest.raises(SdistCanonicalizationError, match="unsupported PAX"):
        canonicalize_sdist(source, source_date_epoch=123)
