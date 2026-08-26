"""Generate the tiny, deterministic SourceItem v1 contract fixtures.

The committed fixtures are derived from one checksum-frozen, CC-BY-4.0 Zeiss
LSM source in the public corpus.  The generator never downloads data.  It
verifies the supplied source, takes fixed pixel windows, and writes:

* a mixed-member NPZ with canonical member order and fixed ZIP metadata; and
* two equivalent two-series OME-TIFFs with opposite physical series order.

NPZ and TIFF container bytes are deterministic for one generator/library
contract.  Canonical per-item semantic hashes are also recorded so a reviewed
library update can distinguish harmless container-encoding drift from changed
fixture meaning.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import tifffile

SCHEMA = "napari-vipp-sourceitem-contract-fixtures"
SCHEMA_VERSION = 1
GENERATOR_ID = "napari-vipp-sourceitem-contract-fixture-generator-v1"
SOURCE_DATASET_ID = "zenodo-14510432-zeiss-lsm"
SOURCE_SERIES = {
    "primary": 0,
    "associated": 1,
}
NPZ_NAME = "sourceitem-mixed-members-v1.npz"
OME_TIFF_NAME = "sourceitem-two-series-v1.ome.tiff"
OME_TIFF_REVERSED_NAME = "sourceitem-two-series-reversed-v1.ome.tiff"
RECORD_NAME = "sourceitem-contract-fixtures-v1.json"
NPZ_MEMBER_ORDER = (
    "outline_bool",
    "thumbnail_uint8",
    "primary_uint16",
    "normalized_float32",
)
OME_SERIES_ORDER = ("primary-uint16", "associated-uint8")
OME_UUIDS = {
    "forward": "urn:uuid:5d88b66e-4622-52a5-ae89-207bf4fd7001",
    "reversed": "urn:uuid:5d88b66e-4622-52a5-ae89-207bf4fd7002",
}
_ARRAY_IDENTITY_DOMAIN = b"napari-vipp-sourceitem-fixture-array-v1\0"
_SERIES_SET_IDENTITY_DOMAIN = b"napari-vipp-sourceitem-series-set-v1\0"
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_array(value: object) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError("Contract fixtures cannot contain object arrays.")
    if array.dtype.itemsize > 1:
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return np.ascontiguousarray(array)


def array_semantic_record(
    name: str,
    axes: str,
    value: object,
) -> dict[str, object]:
    """Return the canonical semantic identity of one named fixture array."""
    array = _canonical_array(value)
    header = json.dumps(
        {
            "axes": str(axes),
            "dtype": array.dtype.name,
            "name": str(name),
            "shape": list(array.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(_ARRAY_IDENTITY_DOMAIN)
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(array.tobytes(order="C"))
    return {
        "name": str(name),
        "axes": str(axes),
        "shape": list(array.shape),
        "dtype": array.dtype.name,
        "sha256": digest.hexdigest(),
        "minimum": bool(array.min())
        if array.dtype == np.dtype(bool)
        else array.min().item(),
        "maximum": bool(array.max())
        if array.dtype == np.dtype(bool)
        else array.max().item(),
    }


def semantic_series_set_sha256(records: Sequence[Mapping[str, object]]) -> str:
    """Hash logical series independent of their physical container order."""
    digest = hashlib.sha256()
    digest.update(_SERIES_SET_IDENTITY_DOMAIN)
    for record in sorted(records, key=lambda item: str(item["name"])):
        name = str(record["name"]).encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(bytes.fromhex(str(record["sha256"])))
    return digest.hexdigest()


def _npy_bytes(value: object) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        _canonical_array(value),
        version=(1, 0),
        allow_pickle=False,
    )
    return stream.getvalue()


def build_deterministic_npz(members: Mapping[str, object]) -> bytes:
    """Build a byte-stable NPZ with canonical members and ZIP headers."""
    if tuple(members) != NPZ_MEMBER_ORDER:
        raise ValueError(
            "Mixed-member NPZ keys must use the canonical order: "
            + ", ".join(NPZ_MEMBER_ORDER)
        )
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name, value in members.items():
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(value))
    return stream.getvalue()


def build_two_series_ome_tiff(
    series: Mapping[str, object],
    *,
    reverse: bool = False,
) -> bytes:
    """Build one two-series OME-TIFF in forward or reversed physical order."""
    if tuple(series) != OME_SERIES_ORDER:
        raise ValueError(
            "OME-TIFF series must use the canonical order: "
            + ", ".join(OME_SERIES_ORDER)
        )
    ordered_names = OME_SERIES_ORDER[::-1] if reverse else OME_SERIES_ORDER
    stream = io.BytesIO()
    with tifffile.TiffWriter(stream, ome=True, bigtiff=False, byteorder="<") as writer:
        for index, name in enumerate(ordered_names):
            metadata: dict[str, object] = {"axes": "YX", "Name": name}
            if index == 0:
                metadata.update(
                    {
                        "Creator": GENERATOR_ID,
                        "UUID": OME_UUIDS["reversed" if reverse else "forward"],
                    }
                )
            writer.write(
                _canonical_array(series[name]),
                photometric="minisblack",
                compression=None,
                contiguous=False,
                datetime=False,
                software=False,
                metadata=metadata,
            )
    return stream.getvalue()


def inspect_npz_semantics(value: bytes | Path) -> tuple[dict[str, object], ...]:
    """Read the member semantic records from a generated NPZ."""
    source = io.BytesIO(value) if isinstance(value, bytes) else value
    with np.load(source, allow_pickle=False) as archive:
        return tuple(
            array_semantic_record(name, "YX", archive[name]) for name in archive.files
        )


def inspect_ome_tiff_semantics(value: bytes | Path) -> tuple[dict[str, object], ...]:
    """Read physical-order semantic records from a generated OME-TIFF."""
    source = io.BytesIO(value) if isinstance(value, bytes) else value
    with tifffile.TiffFile(source) as container:
        return tuple(
            array_semantic_record(series.name, series.axes, series.asarray())
            for series in container.series
        )


def _source_contract(manifest_path: Path) -> dict[str, object]:
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    dataset = next(
        (
            record
            for record in manifest["datasets"]
            if record["id"] == SOURCE_DATASET_ID
        ),
        None,
    )
    if dataset is None:
        raise ValueError(f"Manifest does not contain {SOURCE_DATASET_ID!r}.")
    artifact = dataset["artifact"]
    return {
        "manifest": manifest_path.name,
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "dataset_id": SOURCE_DATASET_ID,
        "artifact_sha256": artifact["sha256"],
        "artifact_bytes": artifact["bytes"],
        "attribution": dataset["license"]["attribution"],
        "license": dataset["license"]["spdx"],
        "landing_page": dataset["source"]["landing_page"],
    }


def _derived_arrays(source_path: Path, source_contract: Mapping[str, object]):
    if source_path.stat().st_size != source_contract["artifact_bytes"]:
        raise ValueError("LSM source byte count does not match the frozen manifest.")
    if _sha256_file(source_path) != source_contract["artifact_sha256"]:
        raise ValueError("LSM source SHA-256 does not match the frozen manifest.")

    with tifffile.TiffFile(source_path) as container:
        primary = np.asarray(container.series[SOURCE_SERIES["primary"]].asarray())
        associated = np.asarray(container.series[SOURCE_SERIES["associated"]].asarray())
    if primary.shape != (4, 1024, 1024) or primary.dtype != np.dtype("uint16"):
        raise ValueError(
            "Frozen LSM primary series no longer has the expected contract."
        )
    if associated.shape != (3, 128, 128) or associated.dtype != np.dtype("uint8"):
        raise ValueError(
            "Frozen LSM associated series no longer has the expected contract."
        )

    primary_patch = np.ascontiguousarray(primary[0, 496:528, 496:528], dtype="<u2")
    thumbnail_patch = np.ascontiguousarray(associated[0, 48:72, 48:72], dtype="u1")
    members = {
        "outline_bool": np.ascontiguousarray(thumbnail_patch >= 96),
        "thumbnail_uint8": thumbnail_patch,
        "primary_uint16": primary_patch,
        "normalized_float32": np.ascontiguousarray(
            primary_patch.astype("<f4") / np.float32(65535.0)
        ),
    }
    series = {
        "primary-uint16": primary_patch,
        "associated-uint8": thumbnail_patch,
    }
    return members, series


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_fixture_set(
    source_path: Path,
    output_dir: Path,
    *,
    manifest_path: Path,
) -> dict[str, object]:
    """Verify one frozen LSM source and write the complete fixture set."""
    source_path = source_path.resolve(strict=True)
    output_dir = output_dir.resolve()
    contract = _source_contract(manifest_path.resolve(strict=True))
    members, series = _derived_arrays(source_path, contract)

    npz_bytes = build_deterministic_npz(members)
    forward_bytes = build_two_series_ome_tiff(series)
    reversed_bytes = build_two_series_ome_tiff(series, reverse=True)
    npz_records = inspect_npz_semantics(npz_bytes)
    forward_records = inspect_ome_tiff_semantics(forward_bytes)
    reversed_records = inspect_ome_tiff_semantics(reversed_bytes)
    if semantic_series_set_sha256(forward_records) != semantic_series_set_sha256(
        reversed_records
    ):
        raise RuntimeError("Forward and reversed OME-TIFF semantics disagree.")

    artifacts = {
        NPZ_NAME: {
            "bytes": len(npz_bytes),
            "sha256": _sha256_bytes(npz_bytes),
            "byte_contract": "exact-v1",
            "member_order": list(NPZ_MEMBER_ORDER),
            "members": list(npz_records),
        },
        OME_TIFF_NAME: {
            "bytes": len(forward_bytes),
            "sha256": _sha256_bytes(forward_bytes),
            "byte_contract": "exact-for-recorded-generator;semantic-hash-is-portable",
            "physical_series_order": [record["name"] for record in forward_records],
            "semantic_set_sha256": semantic_series_set_sha256(forward_records),
            "series": list(forward_records),
        },
        OME_TIFF_REVERSED_NAME: {
            "bytes": len(reversed_bytes),
            "sha256": _sha256_bytes(reversed_bytes),
            "byte_contract": "exact-for-recorded-generator;semantic-hash-is-portable",
            "physical_series_order": [record["name"] for record in reversed_records],
            "semantic_set_sha256": semantic_series_set_sha256(reversed_records),
            "series": list(reversed_records),
        },
    }
    record = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "generator": {
            "id": GENERATOR_ID,
            "numpy": np.__version__,
            "tifffile": tifffile.__version__,
        },
        "source": contract,
        "recipe": {
            "primary_series_index": SOURCE_SERIES["primary"],
            "primary_window": "C=0,Y=496:528,X=496:528",
            "associated_series_index": SOURCE_SERIES["associated"],
            "associated_window": "S=0,Y=48:72,X=48:72",
            "bool_rule": "associated uint8 >= 96",
            "float32_rule": "primary uint16 / float32(65535)",
            "npy_version": "1.0",
            "zip_compression": "stored",
            "zip_timestamp": "1980-01-01T00:00:00",
            "ome_series_semantics": "name+axes+shape+dtype+C-order pixels",
        },
        "artifacts": artifacts,
    }

    _write_bytes(output_dir / NPZ_NAME, npz_bytes)
    _write_bytes(output_dir / OME_TIFF_NAME, forward_bytes)
    _write_bytes(output_dir / OME_TIFF_REVERSED_NAME, reversed_bytes)
    encoded = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_bytes(output_dir / RECORD_NAME, encoded)
    return record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_lsm", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    record = write_fixture_set(
        arguments.source_lsm,
        arguments.output_dir,
        manifest_path=arguments.manifest,
    )
    print(
        f"Wrote {len(record['artifacts'])} SourceItem contract fixtures to "
        f"{arguments.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
