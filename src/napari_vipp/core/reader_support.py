"""Fixed reader catalog and isolated diagnostics; never opens a user's image.

Import this module freely in the UI. Call ``probe_reader`` only in a disposable
process: native library imports can fail, hang, or crash outside Python.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from packaging.requirements import Requirement


@dataclass(frozen=True)
class ReaderSpec:
    key: str
    name: str
    formats: str
    suffixes: tuple[str, ...]
    requirements: tuple[str, ...]
    modules: tuple[tuple[str, str], ...]
    optional: bool = False


READERS = (
    ReaderSpec(
        "tiff",
        "TIFF / OME-TIFF",
        "TIFF, OME-TIFF, LSM",
        (".tif", ".tiff", ".lsm"),
        ("tifffile>=2023.8", "imagecodecs>=2026.8.16,<2027"),
        (("tifffile", "TiffFile"), ("imagecodecs", "imread")),
    ),
    ReaderSpec(
        "czi",
        "Zeiss CZI",
        "CZI",
        (".czi",),
        ("czifile>=2026.8.16,<2027", "imagecodecs>=2026.8.16,<2027"),
        (("czifile", "CziFile"), ("imagecodecs", "imread")),
    ),
    ReaderSpec(
        "lif",
        "Leica LIF",
        "LIF, LOF, XLIF",
        (".lif", ".lof", ".xlif"),
        ("liffile>=2026.7.14,<2027",),
        (("liffile", "LifFile"),),
    ),
    ReaderSpec(
        "nd2",
        "Nikon ND2",
        "ND2, including legacy compression",
        (".nd2",),
        ("nd2[legacy]>=0.11.1,<0.12", "imagecodecs>=2026.8.16,<2027"),
        (("nd2", "ND2File"), ("imagecodecs", "jpeg2k_decode")),
    ),
    ReaderSpec(
        "oif",
        "Olympus OIF / OIB",
        "OIF, OIB",
        (".oif", ".oib"),
        ("oiffile>=2026.2.8,<2027",),
        (("oiffile", "OifFile"),),
    ),
    ReaderSpec(
        "oir",
        "Olympus OIR",
        "OIR",
        (".oir",),
        ("oirfile>=2026.7.28,<2027",),
        (("oirfile", "OirFile"),),
    ),
    ReaderSpec(
        "bioformats",
        "Bio-Formats",
        "IMS, VSI and fallback formats",
        (".ims", ".vsi"),
        ("bioio>=3.4,<4", "bioio-bioformats>=2,<3"),
        (("bioio", "BioImage"), ("bioio_bioformats", "Reader")),
        True,
    ),
)


def reader_spec(key: str) -> ReaderSpec:
    """Only a catalog identifier can select an installation recipe."""
    for reader in READERS:
        if reader.key == key:
            return reader
    raise ValueError(f"Unknown reader: {key!r}")


def reader_for_path(path: str) -> ReaderSpec | None:
    suffix = Path(path).suffix.casefold()
    return next((r for r in READERS if suffix in r.suffixes), None)


@dataclass(frozen=True)
class ReaderStatus:
    key: str
    state: str  # ready, missing, incompatible, broken
    detail: str


def dependency_status(key: str) -> ReaderStatus:
    """Cheap metadata-only check, safe on the UI thread; no reader imports."""
    spec = reader_spec(key)
    versions = []
    for requirement in spec.requirements:
        required = Requirement(requirement)
        try:
            version = metadata.version(required.name)
        except metadata.PackageNotFoundError:
            return ReaderStatus(key, "missing", f"{required.name} is not installed.")
        versions.append(f"{required.name} {version}")
        if version not in required.specifier:
            return ReaderStatus(
                key,
                "incompatible",
                f"{required.name} {version}; needs {required.specifier}.",
            )
    return ReaderStatus(key, "unchecked", ", ".join(versions))


def probe_reader(key: str) -> ReaderStatus:
    spec = reader_spec(key)
    available = dependency_status(key)
    if available.state != "unchecked":
        return available
    try:
        importlib.invalidate_caches()
        for module, symbol in spec.modules:
            getattr(importlib.import_module(module), symbol)
        if key in {"tiff", "czi", "nd2"}:
            import imagecodecs
            import numpy as np

            # Exercise native codec loading, not just its Python wrapper.
            pixels = np.arange(16, dtype=np.uint16).reshape(4, 4)
            encoded = imagecodecs.jpeg2k_encode(pixels, level=0)
            if not np.array_equal(imagecodecs.jpeg2k_decode(encoded), pixels):
                raise RuntimeError("Lossless JPEG 2000 codec check failed.")
    except Exception as error:
        return ReaderStatus(key, "broken", f"{type(error).__name__}: {error}"[:800])
    detail = available.detail
    if spec.optional:
        detail += (
            ". Python reader available; Java/Bio-Formats is initialized on first "
            "use and may download its runtime."
        )
    return ReaderStatus(key, "ready", detail)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check a VIPP reader without opening an image"
    )
    parser.add_argument("reader", choices=[r.key for r in READERS] + ["all-native"])
    args = parser.parse_args()
    if args.reader == "all-native":
        results = [probe_reader(r.key) for r in READERS if not r.optional]
        print(json.dumps([asdict(result) for result in results], indent=2))
        sys.exit(0 if all(result.state == "ready" for result in results) else 1)
    print(json.dumps(asdict(probe_reader(args.reader))))


if __name__ == "__main__":
    main()
