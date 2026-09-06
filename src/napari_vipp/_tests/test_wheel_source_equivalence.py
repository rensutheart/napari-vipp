from __future__ import annotations

import base64
import csv
import hashlib
import io
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import package_macos_installer as macos
from scripts import package_windows_installer as windows
from scripts.wheel_source_equivalence import (
    WheelSourceEquivalenceError,
    require_source_equivalent_wheels,
)

VERSION = "0.15.0a1"
DIST_INFO = f"napari_vipp-{VERSION}.dist-info"
METADATA = f"{DIST_INFO}/METADATA"
RECORD = f"{DIST_INFO}/RECORD"
WHEEL = f"{DIST_INFO}/WHEEL"
CODE = "napari_vipp/example.py"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _payloads(*, crlf: bool = False) -> dict[str, bytes]:
    metadata = (
        f"Metadata-Version: 2.4\nName: napari-vipp\nVersion: {VERSION}\n"
        "Requires-Dist: numpy>=1.24\n\nA description.\n"
    ).encode()
    return {
        METADATA: metadata.replace(b"\n", b"\r\n") if crlf else metadata,
        WHEEL: b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        CODE: b"VALUE = 123\n",
        "napari_vipp/data.txt": b"unchanged asset\n",
    }


def _record(payloads: dict[str, bytes]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    for name, contents in sorted(payloads.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(contents).digest())
        writer.writerow((name, "sha256=" + digest.rstrip(b"=").decode(), len(contents)))
    writer.writerow((RECORD, "", ""))
    return stream.getvalue().encode()


def _wheel(path, payloads=None, *, record=None, member_type=None):
    payloads = _payloads() if payloads is None else payloads
    with zipfile.ZipFile(path, "w") as archive:
        for name, contents in payloads.items():
            info = zipfile.ZipInfo(name)
            if member_type and name == CODE:
                info.create_system = 3
                info.external_attr = member_type << 16
            archive.writestr(info, contents)
        archive.writestr(RECORD, _record(payloads) if record is None else record)
    return path


def test_only_metadata_newlines_are_accepted_without_rewriting_or_rehashing(tmp_path):
    left = _wheel(tmp_path / "left.whl", _payloads(crlf=True))
    right = _wheel(tmp_path / "right.whl")
    original = (left.read_bytes(), right.read_bytes())
    for packager in (windows, macos):
        assert packager._wheel_contents_sha256(left) != packager._wheel_contents_sha256(
            right
        )
    require_source_equivalent_wheels(left, right)
    require_source_equivalent_wheels(right, left)
    assert (left.read_bytes(), right.read_bytes()) == original


def test_identical_valid_wheels_are_accepted(tmp_path):
    wheel = _wheel(tmp_path / "valid.whl")
    require_source_equivalent_wheels(wheel, wheel)


@pytest.mark.parametrize("member", [METADATA, WHEEL, CODE, "napari_vipp/data.txt"])
def test_semantic_change_with_correct_record_is_rejected(tmp_path, member):
    original = _wheel(tmp_path / "original.whl")
    payloads = _payloads()
    payloads[member] += b"semantic change\n"
    changed = _wheel(tmp_path / "changed.whl", payloads)
    with pytest.raises(WheelSourceEquivalenceError):
        require_source_equivalent_wheels(original, changed)


@pytest.mark.parametrize("member", [WHEEL, CODE, "napari_vipp/data.txt"])
def test_newlines_outside_generated_metadata_are_not_normalized(tmp_path, member):
    original = _wheel(tmp_path / "original.whl")
    payloads = _payloads(crlf=True)
    payloads[member] = payloads[member].replace(b"\n", b"\r\n")
    changed = _wheel(tmp_path / "changed.whl", payloads)
    with pytest.raises(WheelSourceEquivalenceError):
        require_source_equivalent_wheels(original, changed)


@pytest.mark.parametrize(
    "corruption",
    [
        "hash",
        "size",
        "missing",
        "duplicate",
        "extra",
        "missing_self",
        "hashed_self",
        "sized_self",
        "weak_hash",
        "blank_hash",
        "malformed",
        "invalid_utf8",
        "duplicate_self",
        "noncanonical_size",
        "metadata_hash",
    ],
)
def test_invalid_record_is_rejected_even_when_both_wheels_are_identical(
    tmp_path, corruption
):
    payloads = _payloads()
    rows = list(csv.reader(io.StringIO(_record(payloads).decode())))
    code_row = next(row for row in rows if row[0] == CODE)
    if corruption == "hash":
        code_row[1] = "sha256=" + "a" * 43
    elif corruption == "size":
        code_row[2] = "9999"
    elif corruption == "missing":
        rows.remove(code_row)
    elif corruption == "duplicate":
        rows.append(code_row)
    elif corruption == "extra":
        rows.append(["not-in-wheel", "sha256=" + "a" * 43, "1"])
    elif corruption == "missing_self":
        rows = [row for row in rows if row[0] != RECORD]
    elif corruption == "hashed_self":
        rows[-1][1] = "sha256=" + "a" * 43
    elif corruption == "sized_self":
        rows[-1][2] = "0"
    elif corruption == "weak_hash":
        code_row[1] = "md5=" + "a" * 22
    elif corruption == "blank_hash":
        code_row[1] = ""
    elif corruption == "duplicate_self":
        rows.append(rows[-1])
    elif corruption == "noncanonical_size":
        code_row[2] = "0" + code_row[2]
    elif corruption == "metadata_hash":
        next(row for row in rows if row[0] == METADATA)[1] = code_row[1]
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(rows)
    record = stream.getvalue().encode()
    if corruption == "malformed":
        record += b"four,fields,not,three\n"
    elif corruption == "invalid_utf8":
        record += b"\xff"
    broken = _wheel(tmp_path / "broken.whl", record=record)
    with pytest.raises(WheelSourceEquivalenceError):
        require_source_equivalent_wheels(broken, broken)


def test_missing_record_file_is_rejected(tmp_path):
    path = tmp_path / "missing.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for name, contents in _payloads().items():
            archive.writestr(name, contents)
    with pytest.raises(WheelSourceEquivalenceError, match="RECORD is missing"):
        require_source_equivalent_wheels(path, path)


@pytest.mark.parametrize("kind", [stat.S_IFLNK, stat.S_IFDIR, stat.S_IFIFO])
def test_unsupported_or_conflicting_member_type_is_rejected(tmp_path, kind):
    broken = _wheel(tmp_path / "broken.whl", member_type=kind)
    with pytest.raises(WheelSourceEquivalenceError, match="member type"):
        require_source_equivalent_wheels(broken, broken)


def test_duplicate_archive_member_is_rejected(tmp_path):
    broken = _wheel(tmp_path / "broken.whl")
    with zipfile.ZipFile(broken, "a") as archive, pytest.warns(UserWarning):
        archive.writestr(CODE, b"VALUE = 123\n")
    with pytest.raises(WheelSourceEquivalenceError, match="duplicate wheel"):
        require_source_equivalent_wheels(broken, broken)


def test_other_record_bytes_are_not_normalized(tmp_path):
    left = _wheel(tmp_path / "left.whl")
    payloads = _payloads(crlf=True)
    right = _wheel(
        tmp_path / "right.whl",
        payloads,
        record=_record(payloads).replace(b"\n", b"\r\n"),
    )
    with pytest.raises(WheelSourceEquivalenceError, match="RECORD entries differ"):
        require_source_equivalent_wheels(left, right)


@pytest.mark.parametrize("change", ["added", "removed", "renamed", "case"])
def test_member_set_changes_with_valid_records_are_rejected(tmp_path, change):
    left = _wheel(tmp_path / "left.whl")
    payloads = _payloads()
    if change == "added":
        payloads["unexpected.txt"] = b"extra"
    elif change == "removed":
        del payloads[CODE]
    else:
        payloads[CODE.upper() if change == "case" else "renamed.py"] = payloads.pop(
            CODE
        )
    right = _wheel(tmp_path / "right.whl", payloads)
    with pytest.raises(WheelSourceEquivalenceError, match="member names or types"):
        require_source_equivalent_wheels(left, right)


@pytest.mark.parametrize(
    "name",
    ["../escape", "/absolute", "C:/drive", "back\\slash", "with\nnewline", "a//b"],
)
def test_unsafe_archive_paths_are_rejected(tmp_path, name):
    payloads = _payloads()
    payloads[name] = b"unsafe"
    broken = _wheel(tmp_path / "broken.whl", payloads)
    with pytest.raises(WheelSourceEquivalenceError):
        require_source_equivalent_wheels(broken, broken)


def test_second_dist_info_directory_is_not_arbitrarily_normalized(tmp_path):
    payloads = _payloads()
    payloads["another.dist-info/METADATA"] = b"arbitrary metadata\n"
    broken = _wheel(tmp_path / "broken.whl", payloads)
    with pytest.raises(WheelSourceEquivalenceError, match="exactly one"):
        require_source_equivalent_wheels(broken, broken)


def test_record_reordering_is_rejected_despite_valid_hashes(tmp_path):
    left = _wheel(tmp_path / "left.whl")
    payloads = _payloads(crlf=True)
    reordered = b"".join(reversed(_record(payloads).splitlines(keepends=True)))
    right = _wheel(tmp_path / "right.whl", payloads, record=reordered)
    with pytest.raises(WheelSourceEquivalenceError, match="RECORD entries differ"):
        require_source_equivalent_wheels(left, right)


@pytest.mark.parametrize("packager", [windows, macos], ids=["windows", "macos"])
@pytest.mark.parametrize(
    "valid", [True, False], ids=["metadata-newlines", "tampered-record"]
)
def test_official_packager_checks_source_equivalence_before_staging(
    tmp_path, monkeypatch, packager, valid
):
    supplied = _wheel(tmp_path / f"napari_vipp-{VERSION}-py3-none-any.whl")
    rebuilt_payload = _payloads(crlf=True)
    rebuilt = _wheel(
        tmp_path / f"rebuilt-{VERSION}-py3-none-any.whl",
        rebuilt_payload,
        record=None if valid else b"",
    )
    state = SimpleNamespace(
        version=VERSION,
        commit="1" * 40,
        commit_count=123,
        expected_tag=f"v{VERSION}",
        exact_tags=(f"v{VERSION}",),
        dirty=False,
        officially_releasable=True,
    )
    state.as_dict = lambda: {
        key: value for key, value in vars(state).items() if key != "as_dict"
    }
    monkeypatch.setattr(packager, "inspect_source", lambda _root: state)
    monkeypatch.setattr(packager, "_build_release_wheel", lambda *_args: rebuilt)

    class StagingReached(Exception):
        pass

    def stop_staging(*_args, **_kwargs):
        raise StagingReached

    if packager is windows:
        for name in (
            "_require_windows_amd64",
            "_require_pyinstaller_version",
            "_require_build_tool_versions",
        ):
            monkeypatch.setattr(packager, name, lambda: None)
        monkeypatch.setattr(packager, "_render_icon", stop_staging)
        expected_error = windows.InstallerPackagingError
    else:
        monkeypatch.setattr(
            packager, "_macos_architecture", lambda: ("arm64", "osx-arm64")
        )
        monkeypatch.setattr(packager, "_require_macos", lambda: None)
        monkeypatch.setattr(packager, "_require_builder_tools", lambda _conda: {})
        monkeypatch.setattr(packager, "_render_menu_metadata", stop_staging)
        expected_error = macos.MacOSInstallerPackagingError
    original = supplied.read_bytes()
    with pytest.raises(StagingReached if valid else expected_error):
        packager.build_installer(
            repository_root=REPO_ROOT,
            wheel_path=supplied,
            output_directory=tmp_path / "output",
            development=False,
        )
    assert supplied.read_bytes() == original


@pytest.mark.parametrize(
    "script", ["package_windows_installer.py", "package_macos_installer.py"]
)
def test_packager_remains_directly_executable(script):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), "--help"],
        cwd=REPO_ROOT.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "finalize-unsigned" in result.stdout
