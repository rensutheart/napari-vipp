"""Generate frozen Sigma Filter Plus outputs from official ImageJ bytecode.

This is a maintainer tool, not part of the normal test run. It downloads and
hash-verifies the published plugin source/class plus a versioned ImageJ jar,
compiles only the small Java adapter in ``scripts/sigma_reference``, executes
the external plugin implementation, and writes a self-describing JSON fixture.

The adapter bypasses ImageJ's dialog, ROI, and stack orchestration. The actual
filter is ``Sigma_Filter_Plus.doFiltering`` and integer restoration is ImageJ's
``ByteProcessor.setPixels`` or ``ShortProcessor.setPixels``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import struct
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = (
    REPOSITORY_ROOT / "scripts" / "sigma_reference" / "SigmaGoldenRunner.java"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "src"
    / "napari_vipp"
    / "_tests"
    / "fixtures"
    / "sigma_filter_fiji_reference_v1.json"
)

REFERENCE_FILES = {
    "plugin_source": {
        "url": "https://imagej.net/ij/plugins/download/Sigma_Filter_Plus.java",
        "filename": "Sigma_Filter_Plus.java",
        "bytes": 14_823,
        "sha256": "d1ae5b9c6ed9f41117f691f3661b73553315055624bce39564422f57c2d6dce1",
        "last_modified": "2022-10-14T11:27:23Z",
    },
    "plugin_class": {
        "url": "https://imagej.net/ij/plugins/download/Sigma_Filter_Plus.class",
        "filename": "Sigma_Filter_Plus.class",
        "bytes": 5_174,
        "sha256": "fc1292bb06ac21e21e81b3a26401fa7f3b70cc90752938da37f3d9135d114e2c",
        "last_modified": "2022-10-14T11:27:23Z",
    },
    "imagej_jar": {
        "url": "https://repo1.maven.org/maven2/net/imagej/ij/1.54p/ij-1.54p.jar",
        "filename": "ij-1.54p.jar",
        "bytes": 2_586_953,
        "sha256": "2e1a09961dfb41cee66ddc821b2577a41a072566ce45a49bae69267099741e20",
        "version": "1.54p",
    },
}


@dataclass(frozen=True)
class InputImage:
    dtype: str
    shape: tuple[int, int]
    values: tuple[int, ...]


@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    input_id: str
    radius: float
    sigma_width: float
    minimum_pixel_fraction: float
    outlier_aware: bool
    coverage: tuple[str, ...]
    expectation: str = "fiji_parity"
    deviation_id: str = ""


def _inputs() -> dict[str, InputImage]:
    gradient_u8 = tuple(
        (row * 31 + column * 7) % 256 for row in range(4) for column in range(6)
    )
    gradient_u16 = tuple(
        (row * 8_111 + column * 997) % 65_536 for row in range(5) for column in range(7)
    )
    hard_edge = tuple(
        10 if column < 4 else 240 for _row in range(5) for column in range(8)
    )
    hot_dead = [
        1_000 + row * 71 + column * 13 for row in range(7) for column in range(7)
    ]
    hot_dead[3 * 7 + 3] = 65_535
    hot_dead[1 * 7 + 5] = 0
    return {
        "u8_constant_tiny": InputImage("uint8", (1, 1), (37,)),
        "u16_constant_high": InputImage("uint16", (2, 3), (65_535,) * 6),
        "u8_gradient": InputImage("uint8", (4, 6), gradient_u8),
        "u16_gradient": InputImage("uint16", (5, 7), gradient_u16),
        "u8_hard_edge": InputImage("uint8", (5, 8), hard_edge),
        "u16_hot_dead": InputImage("uint16", (7, 7), tuple(hot_dead)),
        "u8_fallback_21": InputImage("uint8", (3, 3), (0, 1, 0, 1, 100, 1, 0, 1, 0)),
        "u8_half_up": InputImage("uint8", (3, 3), (0, 1, 0, 0, 100, 0, 0, 1, 0)),
        "u16_half_up": InputImage("uint16", (3, 3), (0, 1, 0, 0, 100, 0, 0, 1, 0)),
        "u8_border_alias": InputImage("uint8", (2, 2), (100, 0, 0, 0)),
        "u8_threshold": InputImage("uint8", (3, 3), (0, 0, 0, 0, 0, 2, 0, 0, 0)),
        "u16_negative_variance": InputImage("uint16", (3, 3), (65_535,) * 9),
    }


def _cases() -> tuple[GoldenCase, ...]:
    return (
        GoldenCase(
            "u8_constant_tiny_radius10",
            "u8_constant_tiny",
            10.0,
            2.0,
            0.2,
            True,
            ("constant", "tiny-plane", "clamped-border", "radius-10"),
        ),
        GoldenCase(
            "u16_constant_high_radius10",
            "u16_constant_high",
            10.0,
            2.0,
            0.2,
            True,
            ("constant", "uint16-high", "clamped-border", "radius-10"),
        ),
        GoldenCase(
            "u8_gradient_radius_half",
            "u8_gradient",
            0.5,
            1.0,
            0.2,
            False,
            ("gradient", "clamped-border", "radius-0.5"),
        ),
        GoldenCase(
            "u16_gradient_radius2",
            "u16_gradient",
            2.0,
            2.0,
            0.8,
            False,
            ("gradient", "uint16", "radius-2"),
        ),
        GoldenCase(
            "u8_hard_edge",
            "u8_hard_edge",
            2.0,
            1.0,
            0.2,
            True,
            ("hard-edge", "edge-preservation"),
        ),
        GoldenCase(
            "u16_hot_dead_outlier_aware",
            "u16_hot_dead",
            2.0,
            1.0,
            0.8,
            True,
            ("hot-pixel", "dead-pixel", "exclude-center-fallback"),
        ),
        GoldenCase(
            "u16_hot_dead_full_mean",
            "u16_hot_dead",
            2.0,
            1.0,
            0.8,
            False,
            ("hot-pixel", "dead-pixel", "full-mean-fallback"),
        ),
        GoldenCase(
            "u8_full_mean_fallback_rounds_20_8_to_21",
            "u8_fallback_21",
            0.5,
            0.0,
            1.0,
            False,
            ("full-mean-fallback", "integer-restoration"),
        ),
        GoldenCase(
            "u8_exclude_center_half_up",
            "u8_half_up",
            0.5,
            0.0,
            1.0,
            True,
            ("exclude-center-fallback", "half-up-restoration"),
        ),
        GoldenCase(
            "u16_exclude_center_half_up",
            "u16_half_up",
            0.5,
            0.0,
            1.0,
            True,
            ("exclude-center-fallback", "half-up-restoration", "uint16"),
        ),
        GoldenCase(
            "u8_border_alias_exclude_one_center",
            "u8_border_alias",
            0.5,
            0.0,
            1.0,
            True,
            ("clamped-border", "repeated-alias", "exclude-center-fallback"),
        ),
        GoldenCase(
            "u8_border_alias_full_mean",
            "u8_border_alias",
            0.5,
            0.0,
            1.0,
            False,
            ("clamped-border", "repeated-alias", "full-mean-fallback"),
        ),
        GoldenCase(
            "u8_inclusive_threshold_equality",
            "u8_threshold",
            0.5,
            2.5,
            1.0,
            True,
            ("inclusive-threshold", "branch-decision"),
        ),
        GoldenCase(
            "u8_threshold_next_sigma_below",
            "u8_threshold",
            0.5,
            2.4999999999999996,
            1.0,
            True,
            ("inclusive-threshold", "branch-decision", "next-below"),
        ),
        GoldenCase(
            "u8_exact_ceil_reviewed_divergence",
            "u8_threshold",
            0.5,
            0.0,
            0.8000001,
            True,
            ("minimum-count", "reviewed-divergence"),
            expectation="intentional_vipp_divergence",
            deviation_id="exact_ceil",
        ),
        GoldenCase(
            "u16_negative_variance_fraction_zero_reviewed_divergence",
            "u16_negative_variance",
            0.5,
            2.0,
            0.0,
            True,
            ("negative-variance", "reviewed-divergence"),
            expectation="intentional_vipp_divergence",
            deviation_id="negative_variance_clamp",
        ),
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _raw(image: InputImage) -> bytes:
    if image.dtype == "uint8":
        return bytes(image.values)
    if image.dtype == "uint16":
        return struct.pack(f"<{len(image.values)}H", *image.values)
    raise ValueError(f"Unsupported dtype: {image.dtype}")


def _decode(dtype: str, raw: bytes) -> list[int]:
    if dtype == "uint8":
        return list(raw)
    if dtype == "uint16":
        return list(struct.unpack(f"<{len(raw) // 2}H", raw))
    raise ValueError(f"Unsupported dtype: {dtype}")


def _download(reference: dict[str, object], target: Path) -> None:
    request = urllib.request.Request(
        str(reference["url"]),
        headers={"User-Agent": "napari-vipp-sigma-reference-generator/1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read()
    if len(content) != int(reference["bytes"]):
        raise RuntimeError(
            f"Unexpected size for {reference['url']}: {len(content)} bytes"
        )
    if _sha256(content) != reference["sha256"]:
        raise RuntimeError(f"SHA-256 mismatch for {reference['url']}")
    target.write_bytes(content)


def _tool(name: str, explicit: str | None) -> str:
    resolved = explicit or shutil.which(name)
    if not resolved:
        raise RuntimeError(f"{name} is required; pass --{name} explicitly")
    return resolved


def generate(output: Path, *, java: str, javac: str) -> None:
    inputs = _inputs()
    cases = _cases()
    with tempfile.TemporaryDirectory(prefix="vipp-sigma-fiji-") as temp_name:
        work = Path(temp_name)
        downloaded: dict[str, Path] = {}
        for key, reference in REFERENCE_FILES.items():
            target = work / str(reference["filename"])
            _download(reference, target)
            downloaded[key] = target

        classpath = os.pathsep.join((str(downloaded["imagej_jar"]), str(work)))
        compile_result = subprocess.run(
            [javac, "-classpath", classpath, "-d", str(work), str(HARNESS_PATH)],
            check=True,
            capture_output=True,
            text=True,
        )
        javac_version = subprocess.run(
            [javac, "-version"],
            check=True,
            capture_output=True,
            text=True,
        )
        execution_environment: dict[str, str] | None = None
        encoded_cases: list[dict[str, object]] = []
        for case in cases:
            image = inputs[case.input_id]
            input_raw = _raw(image)
            input_path = work / f"{case.case_id}.input.raw"
            output_path = work / f"{case.case_id}.output.raw"
            input_path.write_bytes(input_raw)
            command = [
                java,
                "-Djava.awt.headless=true",
                "-classpath",
                classpath,
                "SigmaGoldenRunner",
                image.dtype,
                str(image.shape[1]),
                str(image.shape[0]),
                repr(case.radius),
                repr(case.sigma_width),
                repr(case.minimum_pixel_fraction),
                str(case.outlier_aware).lower(),
                str(input_path),
                str(output_path),
            ]
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            observed_environment = json.loads(result.stdout.strip())
            if execution_environment is None:
                execution_environment = observed_environment
            elif observed_environment != execution_environment:
                raise RuntimeError(
                    "Reference execution environment changed between cases"
                )
            output_raw = output_path.read_bytes()
            if len(output_raw) != len(input_raw):
                raise RuntimeError(f"Unexpected output length for {case.case_id}")
            encoded_cases.append(
                {
                    "id": case.case_id,
                    "input_id": case.input_id,
                    "parameters": {
                        "radius": case.radius,
                        "sigma_width": case.sigma_width,
                        "minimum_pixel_fraction": case.minimum_pixel_fraction,
                        "outlier_aware": case.outlier_aware,
                    },
                    "coverage": list(case.coverage),
                    "expectation": case.expectation,
                    **(
                        {"deviation_id": case.deviation_id} if case.deviation_id else {}
                    ),
                    "external_output_sha256": _sha256(output_raw),
                    "external_output_values": _decode(image.dtype, output_raw),
                }
            )

    encoded_inputs = {
        input_id: {
            "dtype": image.dtype,
            "shape": list(image.shape),
            "input_sha256": _sha256(_raw(image)),
            "values": list(image.values),
        }
        for input_id, image in inputs.items()
    }
    document = {
        "schema": "napari-vipp-sigma-fiji-reference-v1",
        "reference_kind": (
            "independently executed official ImageJ Sigma Filter Plus bytecode; "
            "not the VIPP Python oracle"
        ),
        "observed_date": "2026-08-02",
        "provenance": {
            key: dict(reference) for key, reference in REFERENCE_FILES.items()
        },
        "generation": {
            "script": "scripts/generate_sigma_fiji_reference.py",
            "script_sha256": _sha256(Path(__file__).read_bytes()),
            "harness": "scripts/sigma_reference/SigmaGoldenRunner.java",
            "harness_sha256": _sha256(HARNESS_PATH.read_bytes()),
            "adapter_scope": (
                "Calls official Sigma_Filter_Plus.doFiltering and ImageJ unsigned "
                "restoration; bypasses dialog, ROI, and stack orchestration only."
            ),
            "javac_version": (javac_version.stdout or javac_version.stderr).strip(),
            "compile_stdout": compile_result.stdout.strip(),
            "host_python": platform.python_version(),
            "execution_environment": execution_environment,
        },
        "reviewed_contract_deviations": {
            "exact_ceil": (
                "VIPP uses exact ceil(N*fraction). The published Java plugin uses "
                "(int)(N*fraction+0.999999), which differs in a narrow boundary region."
            ),
            "negative_variance_clamp": (
                "VIPP clamps cancellation-induced negative population variance to +0. "
                "Published Java can form a NaN interval; at fraction 0 it may restore "
                "NaN to unsigned zero. VIPP's stabilized result is intentional."
            ),
            "roi": "Fiji ROI/mask behavior is outside the VIPP v1 node contract.",
        },
        "inputs": encoded_inputs,
        "cases": encoded_cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--java", help="Path to java executable")
    parser.add_argument("--javac", help="Path to javac executable")
    args = parser.parse_args()
    generate(
        args.output.resolve(),
        java=_tool("java", args.java),
        javac=_tool("javac", args.javac),
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
