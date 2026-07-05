"""Generate an analytical phantom validation report for object morphology.

This script validates calibrated VIPP object measurements against deterministic
phantoms with known geometry. Grid-aligned rectangles and cuboids have exact
expected physical measurements. Spheres and ellipsoids are voxelized from
continuous equations, so their checks use explicit tolerances for discretization
and marching-cubes surface approximation.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from napari_vipp.core.operations import (
    measure_3d_mesh_morphology,
    measure_objects,
)

UNIT = "micrometer"
REPORT_DATE = "2026-07-04"


@dataclass(frozen=True)
class Check:
    case: str
    quantity: str
    observed: float
    expected: float
    tolerance: float
    tolerance_kind: str
    unit: str
    status: str
    notes: str = ""

    @property
    def error(self) -> float:
        if math.isnan(self.observed) and math.isnan(self.expected):
            return 0.0
        if self.tolerance_kind == "relative":
            if self.expected == 0:
                return abs(self.observed - self.expected)
            return abs(self.observed - self.expected) / abs(self.expected)
        return abs(self.observed - self.expected)


def run_validation() -> list[Check]:
    """Run all analytical phantom checks."""
    checks: list[Check] = []
    checks.extend(_rectangle_2d_isotropic_checks())
    checks.extend(_rectangle_2d_anisotropic_checks())
    checks.extend(_cuboid_3d_anisotropic_checks())
    checks.extend(_sphere_3d_checks())
    checks.extend(_ellipsoid_3d_anisotropic_checks())
    return checks


def _rectangle_2d_isotropic_checks() -> list[Check]:
    labels = np.zeros((20, 30), dtype=np.int32)
    labels[4:10, 6:16] = 1
    scale = (0.5, 0.5)
    record = _measure_object_record(labels, scale)
    area = 6 * 10 * scale[0] * scale[1]
    return [
        _abs_check(
            "2D isotropic rectangle",
            "area_physical",
            record["area_physical"],
            area,
            1e-12,
            UNIT + "^2",
            "Exact grid-aligned area.",
        ),
        _abs_check(
            "2D isotropic rectangle",
            "centroid_y_physical",
            record["centroid_y_physical"],
            ((4 + 9) / 2) * scale[0],
            1e-12,
            UNIT,
            "Pixel-center centroid scaled by y pixel size.",
        ),
        _abs_check(
            "2D isotropic rectangle",
            "centroid_x_physical",
            record["centroid_x_physical"],
            ((6 + 15) / 2) * scale[1],
            1e-12,
            UNIT,
            "Pixel-center centroid scaled by x pixel size.",
        ),
        _abs_check(
            "2D isotropic rectangle",
            "bbox_axis_0_length_physical",
            record["bbox_axis_0_length_physical"],
            6 * scale[0],
            1e-12,
            UNIT,
            "Exact physical bbox height.",
        ),
        _abs_check(
            "2D isotropic rectangle",
            "bbox_axis_1_length_physical",
            record["bbox_axis_1_length_physical"],
            10 * scale[1],
            1e-12,
            UNIT,
            "Exact physical bbox width.",
        ),
        _abs_check(
            "2D isotropic rectangle",
            "equivalent_diameter_physical",
            record["equivalent_diameter_physical"],
            2.0 * math.sqrt(area / math.pi),
            1e-12,
            UNIT,
            "Diameter of equal-area circle.",
        ),
        _abs_check(
            "2D isotropic rectangle",
            "perimeter_physical",
            record["perimeter_physical"],
            record["perimeter_pixels"] * scale[0],
            1e-12,
            UNIT,
            "Spacing-aware scikit-image perimeter for isotropic pixels.",
        ),
    ]


def _rectangle_2d_anisotropic_checks() -> list[Check]:
    labels = np.zeros((18, 24), dtype=np.int32)
    labels[2:8, 3:11] = 1
    scale = (0.25, 0.75)
    record = _measure_object_record(labels, scale)
    area = 6 * 8 * scale[0] * scale[1]
    return [
        _abs_check(
            "2D anisotropic rectangle",
            "area_physical",
            record["area_physical"],
            area,
            1e-12,
            UNIT + "^2",
            "Exact area with unequal y/x pixel sizes.",
        ),
        _abs_check(
            "2D anisotropic rectangle",
            "bbox_axis_0_length_physical",
            record["bbox_axis_0_length_physical"],
            6 * scale[0],
            1e-12,
            UNIT,
            "Exact physical bbox height.",
        ),
        _abs_check(
            "2D anisotropic rectangle",
            "bbox_axis_1_length_physical",
            record["bbox_axis_1_length_physical"],
            8 * scale[1],
            1e-12,
            UNIT,
            "Exact physical bbox width.",
        ),
        _nan_check(
            "2D anisotropic rectangle",
            "perimeter_physical",
            record["perimeter_physical"],
            UNIT,
            "Physical 2D perimeter is intentionally not estimated for "
            "anisotropic pixels.",
        ),
        _nan_check(
            "2D anisotropic rectangle",
            "perimeter_crofton_physical",
            record["perimeter_crofton_physical"],
            UNIT,
            "scikit-image perimeter estimators require isotropic spacing.",
        ),
    ]


def _cuboid_3d_anisotropic_checks() -> list[Check]:
    labels = np.zeros((14, 18, 24), dtype=np.int32)
    labels[3:9, 4:12, 5:15] = 1
    scale = (1.2, 0.5, 0.25)
    record = _measure_object_record(labels, scale, spatial_ndim=3)
    volume = 6 * 8 * 10 * math.prod(scale)
    return [
        _abs_check(
            "3D anisotropic cuboid",
            "volume_physical",
            record["volume_physical"],
            volume,
            1e-12,
            UNIT + "^3",
            "Exact grid-aligned cuboid volume.",
        ),
        _abs_check(
            "3D anisotropic cuboid",
            "bbox_axis_0_length_physical",
            record["bbox_axis_0_length_physical"],
            6 * scale[0],
            1e-12,
            UNIT,
            "Exact z extent.",
        ),
        _abs_check(
            "3D anisotropic cuboid",
            "bbox_axis_1_length_physical",
            record["bbox_axis_1_length_physical"],
            8 * scale[1],
            1e-12,
            UNIT,
            "Exact y extent.",
        ),
        _abs_check(
            "3D anisotropic cuboid",
            "bbox_axis_2_length_physical",
            record["bbox_axis_2_length_physical"],
            10 * scale[2],
            1e-12,
            UNIT,
            "Exact x extent.",
        ),
        _abs_check(
            "3D anisotropic cuboid",
            "equivalent_diameter_physical",
            record["equivalent_diameter_physical"],
            2.0 * ((3.0 * volume) / (4.0 * math.pi)) ** (1.0 / 3.0),
            1e-12,
            UNIT,
            "Diameter of equal-volume sphere.",
        ),
    ]


def _sphere_3d_checks() -> list[Check]:
    radius = 8.0
    scale = (0.5, 0.5, 0.5)
    labels = _ellipsoid_labels((45, 45, 45), scale, (radius, radius, radius))
    object_record = _measure_object_record(labels, scale, spatial_ndim=3)
    mesh_record = _measure_mesh_record(labels, scale)
    volume = (4.0 / 3.0) * math.pi * radius**3
    surface = 4.0 * math.pi * radius**2
    return [
        _relative_check(
            "3D isotropic sphere",
            "volume_physical",
            object_record["volume_physical"],
            volume,
            0.035,
            UNIT + "^3",
            "Voxel-count volume vs continuous sphere volume.",
        ),
        _relative_check(
            "3D isotropic sphere",
            "equivalent_diameter_physical",
            object_record["equivalent_diameter_physical"],
            2.0 * radius,
            0.015,
            UNIT,
            "Equivalent diameter from voxelized volume.",
        ),
        _relative_check(
            "3D isotropic sphere",
            "mesh_volume_physical",
            mesh_record["mesh_volume_physical"],
            volume,
            0.06,
            UNIT + "^3",
            "Marching-cubes mesh volume vs continuous sphere volume.",
        ),
        _relative_check(
            "3D isotropic sphere",
            "mesh_surface_area_physical",
            mesh_record["mesh_surface_area_physical"],
            surface,
            0.08,
            UNIT + "^2",
            "Marching-cubes surface area vs continuous sphere surface.",
        ),
    ]


def _ellipsoid_3d_anisotropic_checks() -> list[Check]:
    scale = (0.5, 0.25, 0.25)
    semiaxes = (5.0, 4.0, 7.0)
    labels = _ellipsoid_labels((31, 41, 65), scale, semiaxes)
    object_record = _measure_object_record(labels, scale, spatial_ndim=3)
    mesh_record = _measure_mesh_record(labels, scale)
    volume = (4.0 / 3.0) * math.pi * math.prod(semiaxes)
    surface = _knud_thomsen_ellipsoid_surface(*semiaxes)
    return [
        _relative_check(
            "3D anisotropic ellipsoid",
            "volume_physical",
            object_record["volume_physical"],
            volume,
            0.05,
            UNIT + "^3",
            "Voxel-count volume vs continuous ellipsoid volume.",
        ),
        _relative_check(
            "3D anisotropic ellipsoid",
            "equivalent_diameter_physical",
            object_record["equivalent_diameter_physical"],
            2.0 * ((3.0 * volume) / (4.0 * math.pi)) ** (1.0 / 3.0),
            0.02,
            UNIT,
            "Equivalent sphere diameter from ellipsoid volume.",
        ),
        _abs_check(
            "3D anisotropic ellipsoid",
            "bbox_axis_0_length_physical",
            object_record["bbox_axis_0_length_physical"],
            _voxelized_diameter(semiaxes[0], scale[0]),
            1e-12,
            UNIT,
            "Expected z span of center-inclusion voxelization.",
        ),
        _abs_check(
            "3D anisotropic ellipsoid",
            "bbox_axis_1_length_physical",
            object_record["bbox_axis_1_length_physical"],
            _voxelized_diameter(semiaxes[1], scale[1]),
            1e-12,
            UNIT,
            "Expected y span of center-inclusion voxelization.",
        ),
        _abs_check(
            "3D anisotropic ellipsoid",
            "bbox_axis_2_length_physical",
            object_record["bbox_axis_2_length_physical"],
            _voxelized_diameter(semiaxes[2], scale[2]),
            1e-12,
            UNIT,
            "Expected x span of center-inclusion voxelization.",
        ),
        _relative_check(
            "3D anisotropic ellipsoid",
            "mesh_volume_physical",
            mesh_record["mesh_volume_physical"],
            volume,
            0.08,
            UNIT + "^3",
            "Marching-cubes mesh volume vs continuous ellipsoid volume.",
        ),
        _relative_check(
            "3D anisotropic ellipsoid",
            "mesh_surface_area_physical",
            mesh_record["mesh_surface_area_physical"],
            surface,
            0.12,
            UNIT + "^2",
            "Compared with Knud Thomsen ellipsoid surface approximation.",
        ),
    ]


def _measure_object_record(
    labels: np.ndarray,
    scale: tuple[float, ...],
    *,
    spatial_ndim: int = 2,
) -> dict[str, object]:
    axis_names = ("z", "y", "x")[-spatial_ndim:]
    table = measure_objects(
        labels,
        resolved_spatial_ndim=spatial_ndim,
        axis_names=axis_names,
        axis_types=("space",) * spatial_ndim,
        axis_scales=scale,
        axis_units=(UNIT,) * spatial_ndim,
        include_shape_descriptors=True,
        include_axis_descriptors=True,
        include_2d_boundary_descriptors=True,
        include_derived_shape_ratios=True,
        include_2d_shape_moments=True,
    )
    records = table.records()
    if len(records) != 1:
        raise AssertionError(f"expected one object row, found {len(records)}")
    return records[0]


def _measure_mesh_record(
    labels: np.ndarray,
    scale: tuple[float, float, float],
) -> dict[str, object]:
    table = measure_3d_mesh_morphology(
        labels,
        resolved_spatial_ndim=3,
        axis_names=("z", "y", "x"),
        axis_types=("space", "space", "space"),
        axis_scales=scale,
        axis_units=(UNIT, UNIT, UNIT),
        minimum_voxel_count=16,
    )
    records = table.records()
    if len(records) != 1:
        raise AssertionError(f"expected one mesh row, found {len(records)}")
    if records[0]["mesh_status"] != "ok":
        raise AssertionError(records[0]["mesh_error"])
    return records[0]


def _ellipsoid_labels(
    shape: tuple[int, int, int],
    scale: tuple[float, float, float],
    semiaxes: tuple[float, float, float],
) -> np.ndarray:
    center = tuple((size - 1) / 2.0 for size in shape)
    coords = np.indices(shape, dtype=np.float64)
    normalized = np.zeros(shape, dtype=np.float64)
    for axis, (axis_center, axis_scale, semiaxis) in enumerate(
        zip(center, scale, semiaxes, strict=True),
    ):
        physical = (coords[axis] - axis_center) * axis_scale
        normalized += (physical / semiaxis) ** 2
    labels = np.zeros(shape, dtype=np.int32)
    labels[normalized <= 1.0] = 1
    return labels


def _knud_thomsen_ellipsoid_surface(a: float, b: float, c: float) -> float:
    p = 1.6075
    return 4.0 * math.pi * (
        ((a * b) ** p + (a * c) ** p + (b * c) ** p) / 3.0
    ) ** (1.0 / p)


def _voxelized_diameter(semiaxis: float, spacing: float) -> float:
    return (2 * math.floor(float(semiaxis) / float(spacing)) + 1) * float(spacing)


def _abs_check(
    case: str,
    quantity: str,
    observed: object,
    expected: float,
    tolerance: float,
    unit: str,
    notes: str = "",
) -> Check:
    value = float(observed)
    status = "PASS" if abs(value - expected) <= tolerance else "FAIL"
    return Check(
        case,
        quantity,
        value,
        expected,
        tolerance,
        "absolute",
        unit,
        status,
        notes,
    )


def _relative_check(
    case: str,
    quantity: str,
    observed: object,
    expected: float,
    tolerance: float,
    unit: str,
    notes: str = "",
) -> Check:
    value = float(observed)
    error = abs(value - expected) / abs(expected) if expected else abs(value - expected)
    status = "PASS" if error <= tolerance else "FAIL"
    return Check(
        case,
        quantity,
        value,
        expected,
        tolerance,
        "relative",
        unit,
        status,
        notes,
    )


def _nan_check(
    case: str,
    quantity: str,
    observed: object,
    unit: str,
    notes: str = "",
) -> Check:
    value = float(observed)
    status = "PASS" if math.isnan(value) else "FAIL"
    return Check(case, quantity, value, float("nan"), 0.0, "nan", unit, status, notes)


def render_markdown(checks: Iterable[Check]) -> str:
    checks = list(checks)
    passed = sum(check.status == "PASS" for check in checks)
    failed = len(checks) - passed
    lines = [
        "# Analytical Phantom Validation: Calibrated Morphology",
        "",
        f"Last generated: {REPORT_DATE}",
        "",
        "This report validates VIPP calibrated object and mesh morphology",
        "measurements against deterministic analytical phantoms. It is intended",
        "as an internal scientific validation artifact, not as a substitute for",
        "cross-tool benchmarking on biological microscopy datasets.",
        "",
        "## Summary",
        "",
        f"- Checks: {len(checks)}",
        f"- Passed: {passed}",
        f"- Failed: {failed}",
        "- Status: " + ("PASS" if failed == 0 else "FAIL"),
        "",
        "## Scope",
        "",
        "Validated here:",
        "",
        "- grid-aligned 2D rectangle measurements with isotropic pixels;",
        "- grid-aligned 2D rectangle measurements with anisotropic pixels;",
        "- grid-aligned 3D cuboid measurements with anisotropic voxels;",
        "- voxelized 3D sphere measurements against continuous analytical volume",
        "  and surface references;",
        "- voxelized 3D ellipsoid measurements against continuous analytical",
        "  volume and approximate analytical surface references;",
        "- the intentional `NaN` behavior for anisotropic 2D physical perimeter",
        "  columns.",
        "",
        "Not validated here:",
        "",
        "- biological segmentation correctness;",
        "- equivalence to Fiji/ImageJ, CellProfiler, or another external package;",
        "- whole-slide or very large volume performance;",
        "- downstream biological interpretation of morphology values.",
        "",
        "## Results",
        "",
        "| Case | Quantity | Observed | Expected | Error | Tolerance | Unit | "
        "Status | Notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for check in checks:
        lines.append(
            f"| {check.case} | `{check.quantity}` | "
            f"{_format_number(check.observed)} | "
            f"{_format_number(check.expected)} | "
            f"{_format_number(check.error)} | {_format_tolerance(check)} | "
            f"{check.unit} | {check.status} | {check.notes} |",
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Rectangles and cuboids are aligned to the voxel grid, so area,",
            "  volume, centroids, bounding-box coordinates, and bounding-box side",
            "  lengths have exact analytical expectations.",
            "- Sphere and ellipsoid masks are generated by center-inclusion",
            "  voxelization of continuous equations. Their voxel-count and mesh",
            "  measurements are compared to continuous analytical references with",
            "  explicit relative tolerances.",
            "- The ellipsoid surface reference uses the Knud Thomsen approximation,",
            "  so that surface check validates broad scale correctness rather than",
            "  exact closed-form equality.",
            "- `perimeter_physical` and `perimeter_crofton_physical` intentionally",
            "  remain `NaN` for anisotropic 2D pixels because scikit-image's",
            "  perimeter estimators require isotropic spacing. This is preferable",
            "  to reporting a silently misleading scalar-scaled perimeter.",
            "",
            "## Reproduction",
            "",
            "Run from the repository root:",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe "
            "scripts\\validate_calibrated_morphology_phantoms.py --output "
            "docs\\analytical-phantom-validation.md",
            "```",
            "",
            "The script exits with a non-zero status if any check fails.",
            "",
        ],
    )
    return "\n".join(lines)


def _format_number(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    return f"{value:.6g}"


def _format_tolerance(check: Check) -> str:
    if check.tolerance_kind == "relative":
        return f"{100.0 * check.tolerance:.3g}%"
    if check.tolerance_kind == "nan":
        return "must be NaN"
    return _format_number(check.tolerance)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional markdown report path. Prints to stdout when omitted.",
    )
    args = parser.parse_args()

    checks = run_validation()
    report = render_markdown(checks)
    if args.output is None:
        print(report)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")

    failed = [check for check in checks if check.status != "PASS"]
    if failed:
        print(f"{len(failed)} validation check(s) failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
