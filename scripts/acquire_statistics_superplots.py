"""Acquire the six additional fascin wells used in the authors' superplots.

This supplements, without changing, the original four-well acquisition.
Source filenames come from the pinned authors' Image.csv, not guessed URLs.
"""

from __future__ import annotations

import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

import requests
import tifffile

ROOT = Path("D:/VIPP-paper-reproductions")
REFERENCE = (
    ROOT / "statistics/reference/data_subsets/cell_profiler_outputs/idr0139/Image.csv"
)
DESTINATION = ROOT / "statistics/idr0139-superplots"
BASE = "https://ftp.ebi.ac.uk/pub/databases/IDR/idr0139-lawson-fascin/20220707-box/1093711385/"
WELLS = {"B02", "N12", "G15", "I19", "H13", "L18"}


def acquire(name: str) -> dict:
    path = DESTINATION / "raw" / name
    receipt = path.with_suffix(".tif.receipt.json")
    url = BASE + quote(name)
    if path.exists() and receipt.exists():
        prior = json.loads(receipt.read_text(encoding="utf-8"))
        if (
            prior["source_url"] != url
            or hashlib.sha256(path.read_bytes()).hexdigest() != prior["sha256"]
        ):
            raise RuntimeError(f"Existing file failed verification: {path}")
        return prior
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite unverified file: {path}")
    response = requests.get(url, timeout=(30, 120))
    response.raise_for_status()
    partial = path.with_suffix(".tif.part")
    partial.write_bytes(response.content)
    with tifffile.TiffFile(partial) as image:
        pixels = image.asarray()
        assert pixels.shape == (996, 996) and pixels.dtype.name == "uint16"
    entry = {
        "source_url": url,
        "path": str(path),
        "bytes": len(response.content),
        "sha256": hashlib.sha256(response.content).hexdigest(),
        "retrieved_utc": datetime.now(UTC).isoformat(),
        "shape": list(pixels.shape),
        "dtype": pixels.dtype.name,
        "well": name.split("_")[1],
        "pixels_fully_decoded": True,
    }
    partial.replace(path)
    receipt.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
    return entry


def main() -> None:
    with REFERENCE.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    names = sorted(
        {
            value
            for row in rows
            for key, value in row.items()
            if key.startswith("FileName_")
            and value.endswith(".tif")
            and value.split("_")[1] in WELLS
        }
    )
    assert len(names) == 96, f"Unexpected reference inventory: {len(names)}"
    (DESTINATION / "raw").mkdir(parents=True, exist_ok=True)
    manifest = {
        "scope": "Six additional same-plate fascin wells for published superplots",
        "license": "CC BY 4.0",
        "wells": sorted(WELLS),
        "files": [],
        "reference": str(REFERENCE),
        "reference_sha256": hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        "status": "downloading",
    }
    checkpoint = DESTINATION / "manifest.json"
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(acquire, name) for name in names]
        for job in as_completed(jobs):
            manifest["files"].append(job.result())
            checkpoint.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            print(f"Verified {len(manifest['files'])}/96", flush=True)
    manifest["files"].sort(key=lambda item: item["path"])
    manifest["status"] = "complete"
    manifest["bytes"] = sum(item["bytes"] for item in manifest["files"])
    checkpoint.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
