from __future__ import annotations

import hashlib
import json
import re
import stat
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BUILDER = REPO_ROOT / "scripts" / "build_cucim_windows.ps1"


def _payload_program() -> str:
    text = BUILDER.read_text(encoding="utf-8")
    match = re.search(
        r"\$wheelPayloadHashProgram = @'\n(?P<program>.*?)\n'@",
        text,
        re.DOTALL,
    )
    assert match is not None
    return match.group("program")


def _write_wheel(
    path: Path,
    entries: list[tuple[str, bytes]],
    *,
    timestamp: tuple[int, int, int, int, int, int],
) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, contents in entries:
            info = zipfile.ZipInfo(name, timestamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, contents)


def _run_payload_program(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _payload_program(), str(path)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_builder_is_pinned_to_the_qualified_local_recipe() -> None:
    text = BUILDER.read_text(encoding="utf-8")
    expected_fragments = (
        '$SourceTag = "v26.06.00"',
        '$SourceCommit = "3c15781c207eab93a317dd9803a6e726fe01f7c4"',
        '$BuildRecipeId = "napari-vipp-cucim-windows-v1"',
        '$ManifestSchemaVersion = 2',
        '$PayloadHashAlgorithm = "sha256-wheel-payload-length-prefix-v1"',
        (
            '$ExpectedWheelPayloadSha256 = "'
            'd640d1e17bcce15d32d03841997252bf915b63da855e406c35f0d70c5a5ea667"'
        ),
        'distribution = "cucim-cu13"',
        'distribution_version = "26.6.0"',
        '"numpy" = "2.5.1"',
        '"scipy" = "1.18.0"',
        '"scikit-image" = "0.26.0"',
        '"cupy-cuda13x" = "14.1.1"',
        '"nvidia-nvimgcodec-cu13" = "0.8.0.22"',
        '"attrs" = "26.1.0"',
        '"colorama" = "0.4.6"',
        '"imageio" = "2.37.4"',
        '"jsonschema" = "4.26.0"',
        '"jsonschema-specifications" = "2025.9.1"',
        '"networkx" = "3.6.1"',
        '"packaging" = "26.3"',
        '"pillow" = "12.3.0"',
        '"pyproject-hooks" = "1.2.0"',
        '"pyyaml" = "6.0.3"',
        '"rapids-dependency-file-generator" = "1.22.0"',
        '"referencing" = "0.37.0"',
        '"rpds-py" = "2026.6.3"',
        '"tifffile" = "2026.7.31"',
        '"tomlkit" = "0.15.1"',
        '"typing-extensions" = "4.16.0"',
    )
    for fragment in expected_fragments:
        assert fragment in text
    assert "[string]$CucimTag" not in text
    # The public Windows instructions invoke the inbox Windows PowerShell 5.1.
    # ``ConvertFrom-Json -AsHashtable`` exists only in newer PowerShell.
    assert "ConvertFrom-Json -AsHashtable" not in text
    assert text.count("--no-deps") >= 3
    assert "installed package inventory differs from the complete lock" in text
    assert 'exact_package_inventory = "passed"' in text


def test_canonical_payload_ignores_record_order_and_zip_timestamps(
    tmp_path: Path,
) -> None:
    first_entries = [
        ("pkg/module.py", b"answer = 42\n"),
        ("pkg-1.0.dist-info/METADATA", b"Name: pkg\nVersion: 1.0\n"),
        ("pkg-1.0.dist-info/RECORD", b"first,ignored\n"),
    ]
    second_entries = [
        ("pkg-1.0.dist-info/RECORD", b"different,ignored\n"),
        ("pkg-1.0.dist-info/METADATA", b"Name: pkg\nVersion: 1.0\n"),
        ("pkg/module.py", b"answer = 42\n"),
    ]
    first = tmp_path / "first.whl"
    second = tmp_path / "second.whl"
    _write_wheel(first, first_entries, timestamp=(2020, 1, 1, 0, 0, 0))
    _write_wheel(second, second_entries, timestamp=(2025, 6, 1, 1, 2, 4))

    first_result = _run_payload_program(first)
    second_result = _run_payload_program(second)
    assert first_result.returncode == second_result.returncode == 0
    first_report = json.loads(first_result.stdout)
    second_report = json.loads(second_result.stdout)
    assert first_report == second_report
    assert first_report["file_count"] == 2

    digest = hashlib.sha256()
    for name, contents in sorted(first_entries[:2]):
        name_bytes = name.encode("utf-8")
        digest.update(struct.pack(">Q", len(name_bytes)))
        digest.update(name_bytes)
        digest.update(struct.pack(">Q", len(contents)))
        digest.update(contents)
    assert first_report["sha256"] == digest.hexdigest()


def test_canonical_payload_rejects_unsafe_and_symlink_entries(
    tmp_path: Path,
) -> None:
    unsafe = tmp_path / "unsafe.whl"
    _write_wheel(
        unsafe,
        [("../escape.py", b"pass\n")],
        timestamp=(2020, 1, 1, 0, 0, 0),
    )
    assert _run_payload_program(unsafe).returncode != 0

    symlink = tmp_path / "symlink.whl"
    with zipfile.ZipFile(symlink, "w") as archive:
        info = zipfile.ZipInfo("pkg/link.py")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "target.py")
    assert _run_payload_program(symlink).returncode != 0
