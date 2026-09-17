# Seeded 3D segmentation evidence — 2026-09-17

The existing VIPP **Marker-Controlled Watershed** passes exact reference checks
on an analytic 3D phantom and seven acquired mitochondrial volumes. This is
implementation validation and a runnable demonstration, not biological ground
truth, a complete paper reproduction, or Fiji equivalence.

The accompanying [report.json](report.json) records source-file and numerical
array SHA256s, exact workflow hashes, calibration, package versions, histories,
checks, timings, and sampled process RSS. Raw pixels, numerical results and
overlays remain on the user's HDD. The generated file-bound review workflows
include canonical SourceItem records and exact-value calibrated OME-TIFF inputs.

## Reproduction and acceptance

Run [the validation driver](../../../scripts/validate_seeded_segmentation.py)
from a VIPP development environment with the acquired starter collection:

```powershell
python scripts/validate_seeded_segmentation.py --data-root D:/VIPP-paper-reproductions/mitochondria
python -m pytest src/napari_vipp/_tests/test_seeded_3d_watershed_example.py -q
```

Each run creates a new timestamped directory under
`D:/VIPP-paper-reproductions/validation/seeded-segmentation/`; it does not overwrite
original images or earlier evidence. It archives its own source. Every case
contains `workflow.json`, `review-workflow.json`, exported `workflow.py`,
`input-calibrated.ome.tif`, `watershed-labels.ome.tif`, `intermediates.npz`,
`result.json`, and `overlay.png`. The calibrated review input is checked for exact
pixel equality against its source. `review-workflow.json` is ready to open locally
in VIPP; its absolute paths intentionally refer to that run's HDD artifacts.

The distributable [seeded-3d-watershed.json](../../../examples/seeded-3d-watershed.json)
uses VIPP's bundled synthetic volume and requires no acquired pixels. The focused
tests execute that graph, restore its persisted form, and execute its generated
Python through the shared runtime. The separate sphere test checks analytical
truth, reference equality, true 3D behavior, read-only input preservation, label
metadata and anisotropic calibration. Both tests pass. Qt layout checks report
no card, wire, tunnel, or note intersections in initial and ready states.

## Frozen numerical workflow

All seven acquired volumes use identical authored settings, with no per-image
adjustment to improve agreement:

1. Convert uint16 to float32 with `scaling="preserve"`; no normalization.
2. Gaussian Blur 3D, sigma Z/Y/X = 0.5/1/1 **voxels**.
3. Otsu threshold, one complete stack histogram, 256 bins.
4. Remove components smaller than 16 voxels, 3D face connectivity.
5. 3D Euclidean Distance Transform; distances are in voxel-index coordinates.
6. H-Maxima Markers, prominence 1 voxel-index unit, full marker connectivity.
7. 3D Marker-Controlled Watershed, inverted distance, compactness 0, no watershed
   line. The underlying watershed uses default face connectivity (6 neighbors).

Anisotropic physical calibration is **preserved but does not weight the existing
EDT or watershed**. No resampling occurs. Nellie uses embedded Z/Y/X calibration
0.25/0.0655/0.0655 µm. Mammalian originals lack embedded calibration; the pinned
upstream README supplies 0.2/0.1667/0.1667 µm, explicitly written into the review
copies. A future physically weighted distance-transform option is a separate
scientific change, not a claim supported by this run.

Acceptance requires exact equality between shared VIPP execution and
`skimage.segmentation.watershed(-distance, markers, mask=mask, compactness=0,
watershed_line=False)`, an exact repeat of the direct VIPP node, and an exact
second full graph run from a verified file snapshot. All pass, with unchanged
inputs, retained calibration, retained seed IDs, and zero labels outside the
mask. Each 3D result differs from processing independent YX planes with the same
seeds, demonstrating that it is actually using Z connectivity.

The analytic phantom is the union of equal radius-15 voxel spheres centered at
ZYX (17,24,22) and (17,24,41) in a 35×49×64 grid. Its known division is the halfway
X plane; **all 25,770 foreground voxels are assigned correctly**. It uses the
binary mask directly, without the real-image preprocessing chain.

## Real-data and performance results

| Input | Shape ZYX | Regions | Full workflow (s) | Peak sampled RSS / increment (MiB) | Watershed only (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| analytic-overlapping-spheres | 35 x 49 x 64 | 2 | 0.086 | 248.4 / 7.9 | 0.0076 |
| nellie-yeast-t000 | 17 x 192 x 279 | 5 | 0.578 | 373.7 / 85.4 | 0.0162 |
| Drp1_B4_hmec1_0_w1MCherry_CAM_000 | 26 x 512 x 512 | 18 | 5.235 | 935.8 / 610.8 | 0.1351 |
| Drp1_B4_hmec1_0_w1MCherry_CAM_001 | 26 x 512 x 512 | 41 | 5.363 | 1088.0 / 608.8 | 0.1248 |
| Drp1_B4_hmec1_1_w1MCherry_CAM_002 | 24 x 512 x 512 | 37 | 4.641 | 1057.2 / 564.4 | 0.1075 |
| WT_hmec1_3_w1MCherry_CAM_026 | 26 x 512 x 512 | 59 | 4.792 | 1106.1 / 609.3 | 0.1449 |
| WT_hmec1_5_w1MCherry_CAM_030 | 21 x 512 x 512 | 89 | 3.702 | 1033.8 / 498.2 | 0.1075 |
| WT_hmec1_7_w1MCherry_CAM_032 | 19 x 512 x 512 | 39 | 5.956 | 976.3 / 453.8 | 0.1779 |

Authoritative artifacts: `D:/VIPP-paper-reproductions/validation/seeded-segmentation/20260917T174020Z`.
The driver SHA256 matches the source archived in that directory. Tested NumPy
2.5.1, SciPy 1.18.0, scikit-image 0.26.0 and VIPP 0.15.0a5.

Times are one CPU run per case, in a sequential warm process on a 12-logical-CPU
Intel Family 6 Model 165 machine with 31.9 GiB RAM. Other desktop and validation
jobs ran concurrently; this was not dedicated performance qualification.
The measurements are descriptive, not a
cross-hardware performance guarantee. Full-workflow timings include shared
execution preparation and all graph nodes, but exclude input-file reads and
artifact/figure writing. Process RSS includes native allocations and is sampled
every 10 ms, so reported peaks are sampled lower bounds. Absolute RSS includes
imports and retained process state; the increment is relative to that case's
baseline, not an independently isolated process peak.

The complete reference report also includes fixed central 96×96 XY crops from
Nellie and WT CAM026 for the Random Walker experiment. The unseeded foreground
counts are expected consequences of H-Maxima suppression: CAM001 retains 18
foreground voxels without a reachable seed, and CAM002 retains 105. They remain
label 0 in both VIPP and the reference. Other full cases have none. No hidden
fallback seed generation or label filling is performed.

Central XY/XZ overlays were inspected. Bright structures are captured, but
faint mitochondrial branches can be missed and dense adjacent signal can enter
one foreground component. Watershed labels divide the supplied mask into seeded
regions; those regions are not asserted to be individual mitochondria. A
publication-quality morphology workflow still needs a task-specific foreground
model, cell ROI, parameter review, and annotated or Fiji-reference comparison.
No parameter was tuned after seeing held-out images to force a result.

## Random Walker decision

**Defer a separate VIPP node.** This experiment establishes that the installed
scikit-image implementation works in 3D with spacing, but does not establish a
quality improvement for these examples. The analytic geometry is already exact
with watershed. The acquired images have no authoritative label truth, so
differences between methods cannot be interpreted as improved segmentation.

| Input | Classes / active voxels | Solve time (s) | Total wall (s) | Peak tree RSS (MiB) | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| analytic-overlapping-spheres | 2 / 25,770 | 0.249 | 1.225 | 98.0 | 100% synthetic truth; same as watershed |
| nellie-yeast-t000-center96 | 4 / 6,052 | 0.466 | 1.513 | 95.3 | 4,673 voxels differ; no truth |
| nellie-yeast-t000-full | 5 / 8,642 | 0.982 | 1.992 | 188.8 | 4,995 voxels differ; no truth |
| WT_hmec1_3_w1MCherry_CAM_026-center96 | 17 / 36,833 | not completed | 45.019 | 111.9 | Stopped at 45 s; no result |

Random Walker was run as a separate feasibility experiment, not as a VIPP node:
`mode="cg_j"`, `beta=130`, `tol=0.001`, `copy=True`, `return_full_prob=False`,
`channel_axis=None`, and the recorded physical Z/Y/X spacing. Its input is native
intensity converted explicitly to float32; watershed instead uses the distance
elevation. These are different segmentation criteria, not interchangeable
implementations. Both start with the same seed regions. Background and any
unreached watershed foreground are explicitly inactive (`-1`); seed IDs are
densely remapped for the solver and restored for comparisons.

Every solver ran in a separate process with a **45-second wall limit** and
**3 GiB process-tree RSS ceiling**, including Windows venv launcher children.
Completed solves retained all seed IDs, assigned all active voxels, and emitted
no solver warnings. The WT crop did not complete within the wall limit; its
process tree was stopped, no segmentation was published, and a full WT solve
was not attempted. No convergence claim is made for the stopped solve. Solver
times exclude imports/IO; wall time and tree RSS include them. Process-tree RSS
is the resource-ceiling metric; the report also retains sampled solver-process
RSS and baseline/increment for completed cases.

Revisit Random Walker with annotated low-contrast examples and a defined class/
seed policy. Compare quality against the existing watershed before deciding on
a node. A prospective node would also need iterative-solver convergence reporting,
memory preflight, cancellation, spacing and label-ID contracts, benchmark coverage
for many seed classes, and clear distinction from CellProfiler Propagation.
Alternative solver/preconditioner benchmarking was outside this bounded check.

## Sources and limits

- Reference algorithms: scikit-image's official
  [watershed](https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.watershed)
  and [Random Walker](https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.random_walker)
  documentation; tested package versions are pinned by the receipt, rather than
  inferred from the moving documentation URL.
- Nellie: [official sample](https://github.com/aelefebv/nellie/tree/d638a7c2951149963ac52229d7da49c97e594d48/sample_data),
  commit `d638a7c2951149963ac52229d7da49c97e594d48`, CC BY 4.0, Austin E. Y. T.
  Lefebvre et al. The first timepoint retains all 17 Z planes.
- Mammalian examples: [Hill Lab / MitoGraph](https://github.com/Hill-Lab/MitoGraph-Contrib-RScripts/tree/2f762c40a21208eb552937ddc89c7d40aa08825d/samples),
  commit `2f762c40a21208eb552937ddc89c7d40aa08825d`. Download for testing is invited,
  but redistribution licensing is unspecified; no acquired image/overlay is
  bundled in this repository. The WT/Drp1 filenames do not establish independent
  biological replicate identities.
- Source acquisition and calibration evidence reside in
  `D:/VIPP-paper-reproductions/mitochondria/manifest.json` and its README. These
  unmodified downloaded analysis inputs have incomplete upstream acquisition/
  preprocessing histories and are not the original Chaudhry 2020 cohort.
- The earlier exploratory evidence directory `20260917T173215Z` is superseded:
  its external Random Walker RSS sampler observed only the Windows launcher;
  the authoritative receipt here measures the full process tree. Later runs
  retain standalone copies of the exact validation driver.
