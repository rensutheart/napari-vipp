# Seeded segmentation implementation and evidence

Unreleased after 0.15.0a5. Tracked in
[issue 66](https://github.com/rensutheart/napari-vipp/issues/66). Public instructions
belong in the companion manual's `how-to/grow-regions-from-seeds.md` and
`reference/seeded-segmentation.md`; this document records the code contract.

## CellProfiler Propagation

`cellprofiler_propagation` is **Grow Regions from Seeds — CellProfiler
Propagation**, a manual CPU node in Segmentation / Object Separation. Its three
required ports are Guidance image, Seed labels and Foreground mask. Distance
regularization defaults to 0.05. It calls `centrosome.propagate.propagate` from
the pinned BSD-licensed Centrosome 1.3.4 distribution. The full CellProfiler
application is not a dependency, and its kernel has not been reimplemented.

The reference is Jones, Carpenter and Golland (2005),
[Voronoi-Based Segmentation of Cells on Image Manifolds](https://doi.org/10.1007/11569541_54).
The [Centrosome source](https://github.com/CellProfiler/centrosome) is authoritative
for neighborhood, tie and border behavior. Shortest accumulated paths combine
3-by-3 local intensity differences and eight-neighbor pixel displacement.
Increasing regularization increases the distance contribution relative to image
appearance. Intensities are passed unchanged: changing the intensity scale
changes this tradeoff. Physical spacing is carried as metadata, not used in
the distance calculation. No normalization, resampling or projection is implicit.

The numerical module accepts exactly three same-shape 2D arrays; shared graph
execution additionally checks YX spatial semantics on **every** input and rejects
different physical grids. Even singleton T, C or Z dimensions require an explicit
selection upstream. An ordinary uncalibrated 2D source can use VIPP's inferred
YX axes. Explicit XY, TX, channel or unknown axis declarations are not reinterpreted.

Guidance must be finite real Boolean, integer or floating-point values exactly
representable in float64. Seed IDs must be nonnegative integers fitting int32;
Boolean/floating seeds are rejected. The mask must be Boolean. Private contiguous
buffers protect read-only and strided inputs. The output is a new int32 label
image with original IDs. Seeds outside the mask remain in the result but cannot
initiate growth; disconnected mask regions without a reachable seed remain zero.
An empty image yields an empty label image. Finite nonnegative regularization,
int32-safe array coordinates and conservative float64 cost/path overflow bounds
are checked before native execution; unsafe ranges fail with explicit guidance.

Cancellation is checked before and after the blocking native call, and before
publishing its result. The kernel cannot report internal progress or stop midway.
Its declaration admits CPU only, with 2D spatial support. Cache identities include
the actual Centrosome version; a backend version change invalidates reuse. Image
history combines all three branches and records the backend, regularization,
intensity and mask rules. Batch manifests and reproducibility environments carry
the dependency version. Saved workflows and generated Python use the same shared
execution path; no schema version change is needed for a new operation ID.

The wrapper establishes kernel parity for the same guidance, seeds, mask and
weight. It does not establish equivalence to a complete CellProfiler pipeline:
thresholding, object declumping/filtering, excluded nuclei, hole filling and
tertiary-object construction remain separate stages. The 2026 statistics paper's
parameters cannot be transferred to native uint16 guidance without accounting
for its preprocessing and intensity scaling.

## Existing 3D watershed and Random Walker assessment

Marker-Controlled Watershed already runs on full ZYX volumes. Its elevation,
marker and foreground-mask inputs use the shared aligned-grid contract. The
existing Euclidean Distance Transform, compactness and watershed use voxel-index
geometry; carrying anisotropic calibration does not make them spacing-aware.
Default 3D watershed connectivity is face adjacency. Existing Sobel Edges is a
2D slice operation and is not described as a volumetric gradient.

`scripts/validate_seeded_segmentation.py` records acquired-volume reference parity,
repeatability, a true-3D analytical phantom, and comparison with independent slice
processing. Random Walker is evaluated separately with explicit spacing, solver
parameters, seed-ID handling, process memory and runtime limits. It is a different
segmentation method; its presence in scikit-image is not a reason to substitute it
for Propagation or claim improvement without evidence.

## Reproducible evidence

- [Development qualification](evidence/seeded-segmentation/qualification.md):
  clean installed-wheel checks, source regressions and full-suite limitations.
- `test_cellprofiler_propagation.py`: analytical assignments, direct Centrosome
  parity, ties/borders/masks, dtype/range, immutable buffers and cancellation.
- `test_cellprofiler_propagation_integration.py`: graph/roundtrip/export/reference
  parity, CPU execution, physical grids, metadata and actual batch publication.
- `test_compute_cache.py`: required backend identity and upgrade invalidation.
- `scripts/validate_cellprofiler_propagation.py` and
  [acquired-image evidence](evidence/cellprofiler-propagation/README.md): four
  996-by-996 fields, zero differing pixels against Centrosome for shared execution
  and generated Python (3,968,064 pixels per path).
- `scripts/validate_seeded_segmentation.py` and
  [3D evidence](evidence/seeded-segmentation/README.md): seven acquired mitochondria
  volumes, two crops and an analytical phantom pass exact reference/repeat checks.
  Random Walker adds no demonstrated improvement; the mammalian crop exceeds its
  45-second wall limit. Its public node is deferred.

The source images and bulky outputs remain under `D:/VIPP-paper-reproductions`,
with hashes and source manifests. Code and compact evidence do not redistribute
source images with unclear licensing. Reference agreement is implementation
validation, not biological ground truth, Fiji agreement or full paper reproduction.
The datasets are substitutes for the original mitochondria cohort.

Future scope, if needed: spacing-aware distance transforms, an independently
specified 3D geodesic method, complete CellProfiler object-stage parity, Fiji
reference execution, and the papers' statistical/inferential analyses. These
require separate evidence and are not implied by the new node.
