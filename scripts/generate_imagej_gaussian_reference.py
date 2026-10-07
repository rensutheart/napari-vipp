"""Generate synthetic Gaussian references with pinned, unmodified ImageJ bytecode.

Run with --imagej-jar /path/to/ij-1.54p.jar and a JDK's java/javac on PATH.
No research images or VIPP operation code enter the reference calculation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "scripts/imagej_gaussian_reference/ImageJGaussianReference.java"
JAR_SHA256 = "2e1a09961dfb41cee66ddc821b2577a41a072566ce45a49bae69267099741e20"
OUTPUT = ROOT / "src/napari_vipp/_tests/fixtures/imagej_gaussian_reference_v1.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagej-jar", type=Path, required=True)
    parser.add_argument("--java", default=shutil.which("java"))
    parser.add_argument("--javac", default=shutil.which("javac"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    jar = args.imagej_jar.resolve(strict=True)
    if digest(jar) != JAR_SHA256:
        raise ValueError("Reference requires the pinned official ij-1.54p.jar SHA-256.")
    if not args.java or not args.javac:
        raise ValueError("Supply a JDK's java and javac paths.")
    with tempfile.TemporaryDirectory(prefix="vipp-imagej-gaussian-") as temporary:
        subprocess.run(
            [args.javac, "-cp", str(jar), "-d", temporary, str(HARNESS)], check=True
        )
        separator = ";" if platform.system() == "Windows" else ":"
        result = subprocess.run(
            [
                args.java, "-Djava.awt.headless=true", "-cp",
                separator.join((str(jar), temporary)), "ImageJGaussianReference",
            ],
            check=True, capture_output=True, text=True,
        )
    document = json.loads(result.stdout)
    if document["imagej_version"] != "1.54p":
        raise ValueError("Unexpected ImageJ bytecode version.")
    document.update(
        schema="napari-vipp-imagej-gaussian-reference-v1",
        reference_kind="independently executed unmodified ImageJ GaussianBlur bytecode",
        provenance={
            "imagej_jar_sha256": JAR_SHA256,
            "imagej_jar_url": "https://repo1.maven.org/maven2/net/imagej/ij/1.54p/ij-1.54p.jar",
            "harness": str(HARNESS.relative_to(ROOT)).replace("\\", "/"),
            "harness_sha256": digest(HARNESS),
            "generator": str(Path(__file__).resolve().relative_to(ROOT)).replace(
                "\\", "/"
            ),
            "generator_sha256": digest(Path(__file__)),
            "host_platform": platform.platform(),
            "java_version": subprocess.run(
                [args.java, "-version"], check=True, capture_output=True, text=True
            ).stderr.strip(),
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(document['cases'])} independent cases to {args.output}")


if __name__ == "__main__":
    main()
