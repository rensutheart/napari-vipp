# Analytical Phantom Validation: Calibrated Morphology

Last generated: 2026-07-04

This report validates VIPP calibrated object and mesh morphology
measurements against deterministic analytical phantoms. It is intended
as an internal scientific validation artifact, not as a substitute for
cross-tool benchmarking on biological microscopy datasets.

## Summary

- Checks: 28
- Passed: 28
- Failed: 0
- Status: PASS

## Scope

Validated here:

- grid-aligned 2D rectangle measurements with isotropic pixels;
- grid-aligned 2D rectangle measurements with anisotropic pixels;
- grid-aligned 3D cuboid measurements with anisotropic voxels;
- voxelized 3D sphere measurements against continuous analytical volume
  and surface references;
- voxelized 3D ellipsoid measurements against continuous analytical
  volume and approximate analytical surface references;
- the intentional `NaN` behavior for anisotropic 2D physical perimeter
  columns.

Not validated here:

- biological segmentation correctness;
- equivalence to Fiji/ImageJ, CellProfiler, or another external package;
- whole-slide or very large volume performance;
- downstream biological interpretation of morphology values.

## Results

| Case | Quantity | Observed | Expected | Error | Tolerance | Unit | Status | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 2D isotropic rectangle | `area_physical` | 15 | 15 | 0 | 1e-12 | micrometer^2 | PASS | Exact grid-aligned area. |
| 2D isotropic rectangle | `centroid_y_physical` | 3.25 | 3.25 | 0 | 1e-12 | micrometer | PASS | Pixel-center centroid scaled by y pixel size. |
| 2D isotropic rectangle | `centroid_x_physical` | 5.25 | 5.25 | 0 | 1e-12 | micrometer | PASS | Pixel-center centroid scaled by x pixel size. |
| 2D isotropic rectangle | `bbox_axis_0_length_physical` | 3 | 3 | 0 | 1e-12 | micrometer | PASS | Exact physical bbox height. |
| 2D isotropic rectangle | `bbox_axis_1_length_physical` | 5 | 5 | 0 | 1e-12 | micrometer | PASS | Exact physical bbox width. |
| 2D isotropic rectangle | `equivalent_diameter_physical` | 4.37019 | 4.37019 | 0 | 1e-12 | micrometer | PASS | Diameter of equal-area circle. |
| 2D isotropic rectangle | `perimeter_physical` | 14 | 14 | 0 | 1e-12 | micrometer | PASS | Spacing-aware scikit-image perimeter for isotropic pixels. |
| 2D anisotropic rectangle | `area_physical` | 9 | 9 | 0 | 1e-12 | micrometer^2 | PASS | Exact area with unequal y/x pixel sizes. |
| 2D anisotropic rectangle | `bbox_axis_0_length_physical` | 1.5 | 1.5 | 0 | 1e-12 | micrometer | PASS | Exact physical bbox height. |
| 2D anisotropic rectangle | `bbox_axis_1_length_physical` | 6 | 6 | 0 | 1e-12 | micrometer | PASS | Exact physical bbox width. |
| 2D anisotropic rectangle | `perimeter_physical` | NaN | NaN | 0 | must be NaN | micrometer | PASS | Physical 2D perimeter is intentionally not estimated for anisotropic pixels. |
| 2D anisotropic rectangle | `perimeter_crofton_physical` | NaN | NaN | 0 | must be NaN | micrometer | PASS | scikit-image perimeter estimators require isotropic spacing. |
| 3D anisotropic cuboid | `volume_physical` | 72 | 72 | 0 | 1e-12 | micrometer^3 | PASS | Exact grid-aligned cuboid volume. |
| 3D anisotropic cuboid | `bbox_axis_0_length_physical` | 7.2 | 7.2 | 0 | 1e-12 | micrometer | PASS | Exact z extent. |
| 3D anisotropic cuboid | `bbox_axis_1_length_physical` | 4 | 4 | 0 | 1e-12 | micrometer | PASS | Exact y extent. |
| 3D anisotropic cuboid | `bbox_axis_2_length_physical` | 2.5 | 2.5 | 0 | 1e-12 | micrometer | PASS | Exact x extent. |
| 3D anisotropic cuboid | `equivalent_diameter_physical` | 5.16152 | 5.16152 | 0 | 1e-12 | micrometer | PASS | Diameter of equal-volume sphere. |
| 3D isotropic sphere | `volume_physical` | 2134.62 | 2144.66 | 0.00467933 | 3.5% | micrometer^3 | PASS | Voxel-count volume vs continuous sphere volume. |
| 3D isotropic sphere | `equivalent_diameter_physical` | 15.975 | 16 | 0.00156222 | 1.5% | micrometer | PASS | Equivalent diameter from voxelized volume. |
| 3D isotropic sphere | `mesh_volume_physical` | 2130.4 | 2144.66 | 0.00665129 | 6% | micrometer^3 | PASS | Marching-cubes mesh volume vs continuous sphere volume. |
| 3D isotropic sphere | `mesh_surface_area_physical` | 866.9 | 804.248 | 0.0779018 | 8% | micrometer^2 | PASS | Marching-cubes surface area vs continuous sphere surface. |
| 3D anisotropic ellipsoid | `volume_physical` | 583.156 | 586.431 | 0.00558357 | 5% | micrometer^3 | PASS | Voxel-count volume vs continuous ellipsoid volume. |
| 3D anisotropic ellipsoid | `equivalent_diameter_physical` | 10.3656 | 10.385 | 0.00186467 | 2% | micrometer | PASS | Equivalent sphere diameter from ellipsoid volume. |
| 3D anisotropic ellipsoid | `bbox_axis_0_length_physical` | 10.5 | 10.5 | 0 | 1e-12 | micrometer | PASS | Expected z span of center-inclusion voxelization. |
| 3D anisotropic ellipsoid | `bbox_axis_1_length_physical` | 8.25 | 8.25 | 0 | 1e-12 | micrometer | PASS | Expected y span of center-inclusion voxelization. |
| 3D anisotropic ellipsoid | `bbox_axis_2_length_physical` | 14.25 | 14.25 | 0 | 1e-12 | micrometer | PASS | Expected x span of center-inclusion voxelization. |
| 3D anisotropic ellipsoid | `mesh_volume_physical` | 581.932 | 586.431 | 0.00767071 | 8% | micrometer^3 | PASS | Marching-cubes mesh volume vs continuous ellipsoid volume. |
| 3D anisotropic ellipsoid | `mesh_surface_area_physical` | 392.181 | 352.858 | 0.111444 | 12% | micrometer^2 | PASS | Compared with Knud Thomsen ellipsoid surface approximation. |

## Interpretation Notes

- Rectangles and cuboids are aligned to the voxel grid, so area,
  volume, centroids, bounding-box coordinates, and bounding-box side
  lengths have exact analytical expectations.
- Sphere and ellipsoid masks are generated by center-inclusion
  voxelization of continuous equations. Their voxel-count and mesh
  measurements are compared to continuous analytical references with
  explicit relative tolerances.
- The ellipsoid surface reference uses the Knud Thomsen approximation,
  so that surface check validates broad scale correctness rather than
  exact closed-form equality.
- `perimeter_physical` and `perimeter_crofton_physical` intentionally
  remain `NaN` for anisotropic 2D pixels because scikit-image's
  perimeter estimators require isotropic spacing. This is preferable
  to reporting a silently misleading scalar-scaled perimeter.

## Reproduction

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\validate_calibrated_morphology_phantoms.py --output docs\analytical-phantom-validation.md
```

The script exits with a non-zero status if any check fails.
