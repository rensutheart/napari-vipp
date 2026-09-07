"""Verify (or explicitly download) the frozen native-reader release corpus.

Ordinary tests remain network-free. Only --download permits public downloads;
existing cache entries, archive members, and scientific expectations are never
rewritten to make a qualification pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen

REPOSITORY = Path(__file__).resolve().parents[1]
MANIFEST = REPOSITORY / "docs/validation/public-data/corpus-v4.json"
MANIFEST_SHA256 = "3365b4cec7a220f6399f3c030eb3ac751581bde730fac60f01c9fca3b823714d"
NATIVE_DATASET_IDS = (
    "ome-bf007-nikon-nd2",
    "bia-s-biad2080-nikon-nd2",
    "bia-s-biad1390-leica-lif",
    "bia-s-biad1305-zeiss-czi",
    "zenodo-7015307-zeiss-czi-multiscene",
    "ome-imagesc-105684-olympus-oir",
    "ome-imagesc-71616-olympus-oib",
    "zenodo-14510432-zeiss-lsm",
)


def native_artifacts():
    payload = MANIFEST.read_bytes()
    if hashlib.sha256(payload).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Frozen corpus-v4 manifest changed; review a new baseline.")
    datasets = {item["id"]: item for item in json.loads(payload)["datasets"]}
    return [(name, datasets[name]["artifact"]) for name in NATIVE_DATASET_IDS]


def verify_artifact(path, artifact):
    if path.stat().st_size != artifact["bytes"]:
        raise ValueError(f"Frozen artifact size mismatch: {path}")
    with path.open("rb") as stream:
        observed = hashlib.file_digest(stream, "sha256").hexdigest()
    if observed != artifact["sha256"]:
        raise ValueError(f"Frozen artifact SHA-256 mismatch: {path}")


def cache_artifact(cache, artifact, *, download=False):
    cache = cache.resolve()
    target = (cache / artifact["relative_path"]).resolve()
    if not target.is_relative_to(cache) or target == cache:
        raise ValueError("Artifact path escapes the selected cache.")
    if target.exists():
        verify_artifact(target, artifact)
        return target
    if not download:
        raise FileNotFoundError(f"Missing frozen artifact: {target}; use --download.")
    if not artifact["url"].startswith("https://"):
        raise ValueError("Public artifact downloads must use HTTPS.")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".vipp-corpus-", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            with urlopen(artifact["url"], timeout=60) as response:
                if not response.url.startswith("https://"):
                    raise ValueError("Public artifact redirected away from HTTPS.")
                size = 0
                while block := response.read(1024 * 1024):
                    size += len(block)
                    if size > artifact["bytes"]:
                        raise ValueError("Download exceeds frozen artifact size.")
                    output.write(block)
        verify_artifact(temporary, artifact)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    verified = []
    for dataset_id, artifact in native_artifacts():
        print(f"Verifying {dataset_id}", file=sys.stderr, flush=True)
        cache_artifact(args.cache, artifact, download=args.download)
        verified.append({"dataset_id": dataset_id, **artifact})
    print(
        json.dumps(
            {
                "status": "verified cache; decoding must be qualified separately",
                "platform": platform.platform(),
                "architecture": platform.machine(),
                "manifest_sha256": MANIFEST_SHA256,
                "artifacts": verified,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
