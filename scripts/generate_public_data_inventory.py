"""Generate a frozen inventory for one directory-backed public data store.

This maintainer tool records every relative regular-file path, byte count, and
SHA-256 digest.  The aggregate identity is the same privacy-safe local-source
identity used by VIPP itself.  It does not download data and must write outside
the source store so that inventory generation cannot change the identity being
recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath

from napari_vipp.core.source_identity import capture_local_source_identity


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_regular_files(root: Path) -> tuple[tuple[str, Path], ...]:
    records: list[tuple[str, Path]] = []
    for candidate in root.rglob("*"):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(root).as_posix()
        parsed = PurePosixPath(relative)
        if parsed.is_absolute() or ".." in parsed.parts or not parsed.parts:
            raise ValueError(f"Unsafe store path: {relative!r}")
        records.append((relative, candidate))
    records.sort(key=lambda item: item[0])
    return tuple(records)


def build_inventory(
    root: Path,
    *,
    dataset_id: str,
    endpoint: str,
    bucket: str,
    prefix: str,
    retrieved_on: str,
) -> dict[str, object]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Store root must be a directory: {root}")
    if not dataset_id.strip():
        raise ValueError("Dataset ID must not be empty.")
    if not endpoint.startswith("https://"):
        raise ValueError("The source endpoint must use HTTPS.")
    if not bucket.strip() or not prefix.strip().endswith("/"):
        raise ValueError("Bucket must be non-empty and prefix must end with '/'.")

    objects = []
    for relative, path in _relative_regular_files(root):
        objects.append(
            {
                "bytes": path.stat().st_size,
                "key": relative,
                "sha256": _sha256(path),
            }
        )
    identity = capture_local_source_identity(root).to_dict()
    if identity["regular_file_count"] != len(objects):
        raise RuntimeError("Object inventory and VIPP source identity disagree.")
    if identity["size_bytes"] != sum(item["bytes"] for item in objects):
        raise RuntimeError("Object byte total and VIPP source identity disagree.")

    return {
        "schema": "napari-vipp-public-zarr-object-inventory",
        "schema_version": 1,
        "dataset_id": dataset_id,
        "retrieved_on": retrieved_on,
        "source": {
            "bucket": bucket,
            "endpoint": endpoint.rstrip("/"),
            "prefix": prefix,
        },
        "content_identity": identity,
        "objects": objects,
    }


def write_inventory(document: dict[str, object], output: Path) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--retrieved-on", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    root = arguments.root.resolve(strict=True)
    output = arguments.output.resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError("Write the inventory outside the source store.")
    document = build_inventory(
        root,
        dataset_id=arguments.dataset_id,
        endpoint=arguments.endpoint,
        bucket=arguments.bucket,
        prefix=arguments.prefix,
        retrieved_on=arguments.retrieved_on,
    )
    write_inventory(document, output)
    print(
        f"Recorded {len(document['objects'])} objects at "
        f"{document['content_identity']['size_bytes']} bytes in {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
