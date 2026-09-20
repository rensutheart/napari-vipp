"""Verify the narrowly scoped no-relink policy for pinned Centrosome wheels.

These upstream extension bundles already load only macOS system libraries. Adding
an unused RPATH can overflow the Intel wheel's Mach-O header padding. Refuse any
changed wheel or load-command policy instead of silently skipping a needed fixup.
This is a read-only check, not a general Mach-O loader or binary repair tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from zipfile import ZipFile

WHEELS = {
    "osx-arm64": (
        "centrosome-1.3.4-cp312-cp312-macosx_11_0_arm64.whl",
        "ee9190b8514e329972cfaa601a3ed66ec1a4f1968fed310dedb70d8a9236d667",
        0x0100000C,
    ),
    "osx-64": (
        "centrosome-1.3.4-cp312-cp312-macosx_10_13_x86_64.whl",
        "b43e878fd0916b8a40b10294ee366a809a4c81e73eae9bec15684e6291368b0b",
        0x01000007,
    ),
}
EXTENSIONS = frozenset(
    f"centrosome/{name}.cpython-312-darwin.so"
    for name in (
        "_propagate",
        "_cpmorphology2",
        "_convex_hull",
        "_filter",
        "_lapjv",
        "_fastemd",
    )
)
SYSTEM_LIBRARIES = frozenset(("/usr/lib/libSystem.B.dylib", "/usr/lib/libc++.1.dylib"))
# Mach-O constants/layout: Apple's EXTERNAL_HEADERS/mach-o/loader.h.
# Only commands present in the two reviewed wheels are permitted. In particular,
# RPATH, weak/re-exported dylibs, dylinker/environment overrides and unknown
# commands fail closed; no prefix/relative dependency can bypass LC_LOAD_DYLIB.
PASSIVE_COMMANDS = frozenset(
    (0x19, 0x80000022, 0x2, 0xB, 0x1B, 0x24, 0x2A, 0x26, 0x29, 0x32, 0x1D)
)
MACHO_MAGICS = frozenset(
    bytes.fromhex(value)
    for value in (
        "feedface",
        "cefaedfe",
        "feedfacf",
        "cffaedfe",
        "cafebabe",
        "bebafeca",
        "cafebabf",
        "bfbafeca",
    )
)


def verify_macho(data: bytes, *, cpu: int) -> list[str]:
    """Require an architecture-correct bundle with absolute system-only loads."""
    if len(data) < 32:
        raise ValueError("Truncated Mach-O header")
    magic, actual_cpu, _, filetype, count, size, _, _ = struct.unpack_from("<8I", data)
    if magic != 0xFEEDFACF or actual_cpu != cpu or filetype != 8:
        raise ValueError("Expected a native thin 64-bit Mach-O extension bundle")
    end = 32 + size
    if end > len(data) or not count or count > size // 8:
        raise ValueError("Invalid Mach-O load-command table")
    offset = 32
    libraries = []
    for _ in range(count):
        if offset + 8 > end:
            raise ValueError("Truncated Mach-O load command")
        command, command_size = struct.unpack_from("<2I", data, offset)
        if command_size < 8 or command_size % 8 or offset + command_size > end:
            raise ValueError("Invalid Mach-O load-command size")
        if command == 0xC:  # LC_LOAD_DYLIB, with a 24-byte dylib_command header.
            if command_size < 24:
                raise ValueError("Truncated Mach-O dylib command")
            name_offset = struct.unpack_from("<I", data, offset + 8)[0]
            if not 24 <= name_offset < command_size:
                raise ValueError("Invalid Mach-O dylib name offset")
            raw_name = data[offset + name_offset : offset + command_size]
            if b"\0" not in raw_name:
                raise ValueError("Unterminated Mach-O dylib name")
            library = raw_name.split(b"\0", 1)[0].decode("utf-8")
            if library not in SYSTEM_LIBRARIES:
                raise ValueError(f"Unreviewed native dependency: {library!r}")
            libraries.append(library)
        elif command not in PASSIVE_COMMANDS:
            raise ValueError(f"Unreviewed Mach-O load command: {command:#x}")
        offset += command_size
    if offset != end or "/usr/lib/libSystem.B.dylib" not in libraries:
        raise ValueError("Incomplete Mach-O load-command table or system dependency")
    return libraries


def verify_wheel(wheel: Path, target: str, installed: Path | None = None) -> dict:
    """Validate exact upstream bytes and every native payload before packaging."""
    filename, digest, cpu = WHEELS[target]
    if (
        wheel.name != filename
        or hashlib.sha256(wheel.read_bytes()).hexdigest() != digest
    ):
        raise ValueError(f"Expected the hash-pinned {target} Centrosome 1.3.4 wheel")
    records = {}
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate wheel members")
        for name in names:
            data = archive.read(name)
            native = name.endswith((".so", ".dylib")) or data[:4] in MACHO_MAGICS
            if not native:
                continue
            if name not in EXTENSIONS:
                raise ValueError(f"Unreviewed native wheel member: {name}")
            libraries = verify_macho(data, cpu=cpu)
            if installed is not None and (installed / name).read_bytes() != data:
                raise ValueError(
                    f"Installed extension differs from upstream wheel: {name}"
                )
            records[name] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "libraries": libraries,
            }
    if set(records) != EXTENSIONS:
        raise ValueError("Expected all six reviewed Centrosome native extensions")
    return {"target": target, "wheel_sha256": digest, "extensions": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=WHEELS, required=True)
    parser.add_argument("--installed", type=Path)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_wheel(args.wheel, args.target, args.installed), indent=2))


if __name__ == "__main__":
    main()
