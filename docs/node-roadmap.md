# VIPP Node Roadmap

Status: current node-family planning document

Last reviewed: 2026-07-10

This document tracks the node catalogue at the level of workflow capability:
what VIPP can already do, which node families are still worth building, and
which requests should wait for stronger product evidence. It is intentionally
shorter than the old implementation diary. The previous long-form discussion
has been archived as [node-roadmap-history.md](node-roadmap-history.md).

Use [planning.md](planning.md) for versioned alpha milestones. Use the
specialist docs for implementation detail:

- [measurement-workflows.md](measurement-workflows.md): object, intensity,
  mesh, skeleton, table, and colocalization table workflows.
- [skeleton-nodes.md](skeleton-nodes.md): skeleton and network node behaviour.
- [object-mesh-morphology-plan.md](object-mesh-morphology-plan.md): 3D mesh
  morphology and deferred mesh export/preview.
- [colocalization-racc-plan.md](colocalization-racc-plan.md): colocalization,
  RACC-like outputs, object association, and interop decisions.
- [psf-and-deconvolution-plan.md](psf-and-deconvolution-plan.md): PSF
  generation and deconvolution.
- [ome-io-plan.md](ome-io-plan.md) and
  [cache-and-memory.md](cache-and-memory.md): I/O, lazy data, preview, cache,
  and memory policy.

## Scope

VIPP should prioritize complete, inspectable bioimage workflows over broad
library feature parity. The important workflow families remain:

- segmentation and label cleanup for 2D images and true 3D `ZYX` volumes;
- object, intensity, mesh, skeleton, and network measurements;
- two-channel pixel colocalization, object-restricted colocalization, and
  object association;
- multi-channel, z-stack, and time-lapse fluorescence workflows;
- native PSF generation and PSF-aware restoration/deconvolution;
- microscope acquisition import across common vendor formats, with normalized
  metadata;
- reproducible export and batch execution with explicit outputs and
  provenance.

PSF generation, deconvolution foundations, and optional microscope-reader
routing are implemented foundations. Their remaining work is real-data
validation, metadata coverage, and performance polish. Registration,
model-backed segmentation, stitching, object tracking, and specialist
mitochondrial indices remain later work unless a current validation or
publication workflow needs them.

## Priority Definitions

| Priority | Meaning |
| --- | --- |
| P0 | Platform capability needed before several node families can be correct. |
| P1 | High-value node or polish that fits current workflows and dependencies. |
| P2 | Useful after the current platform, batch, preview, and validation work settles. |
| P3 | Specialized, expensive, or dependent on additional product decisions. |
| Defer | Avoid unless a concrete workflow makes it necessary. |

## Decision Principles

1. Prefer complete workflows over adding isolated algorithms.
2. Treat `T`, `C`, `Z`, `Y`, and `X` axes explicitly.
3. Make slice-wise versus volumetric processing visible.
4. Use physical spacing when distance, size, volume, or surface area is being
   measured.
5. Keep images, masks, labels, tables, RGB views, and future points/surfaces
   semantically distinct.
6. Prefer SciPy and scikit-image while they cover the requirement. Add OpenCV or
   heavier libraries only for a demonstrated workflow, performance, or file
   format gap.
7. Every node should be deterministic, headless-testable, serializable in
   workflow JSON, and exportable to Python.
8. Expensive nodes should use manual/cached execution with stale-state feedback,
   progress where practical, and deterministic recomputation in export/batch
   contexts.
9. Validation, documentation, and example workflows are part of the feature for
   scientific analysis nodes.

## Current Baseline By Node Family

This is a compact baseline, not a second full user guide. The code source of
truth is `NODE_LIBRARY` in `src/napari_vipp/core/pipeline.py`.

| Family | Implemented enough to build on | Important remaining work |
| --- | --- | --- |
| Sources, outputs, and persistence | `Image Source`, `Save Image`, `Batch Output`, workflow JSON, Python export, local collection batch execution, OME-TIFF/ImageJ TIFF/TIFF/OME-Zarr/NumPy/common raster I/O. | Proprietary microscope readers, saved batch configs, richer output manifests, per-item provenance, OME-Zarr pyramids, preview-level selection, label colours/properties, HCS traversal, remote reads, and operation-level lazy execution. |
| Axes, calibration, and channels | Crop, axis slice, split/reorder axes, `Set Pixel Size / Units`, `Rescale Axes`, channel extraction/splitting/combining, channel colour assignment, RGB compositing. | Resample to target physical spacing, pad/crop to target shape, flip/rotate, richer channel/probe/objective metadata from real microscope files. |
| Intensity and math | Linear scale/offset, gamma, rescale intensity, normalize, clip, weighted image calculation, add/subtract/ratio, mask image, logical operations, invert, dtype conversion. | Clarify old names where needed; avoid growing a generic image-calculator surface until reproducibility and safety rules are clear. |
| Filtering, enhancement, and restoration foundation | Average/Gaussian/3D Gaussian/median/bilateral/non-local-means filtering, rolling-ball background, subtract background, DoG, unsharp mask, Sobel, Canny, Laplace, `Born-Wolf PSF`, `Prepare / Validate PSF`, baseline Richardson-Lucy, and Richardson-Lucy TV deconvolution. | Real microscopy/PSF validation, reflect-padded edge-policy follow-up, performance profiling for large 3D restoration, wavelet denoising, noise estimation, Laplacian-of-Gaussian/blob-oriented helpers, and clearer performance/progress guidance on expensive restoration steps. |
| Thresholding and segmentation | Otsu, Triangle, Li, Yen, Isodata, Minimum, Binary, Hysteresis, Adaptive Mean/Gaussian, Sauvola, Niblack, Auto Watershed From Mask, distance transform, H-maxima markers, marker-controlled watershed, and expand labels. | Watershed validation and marker QC summaries, better defaults from microscopy examples, optional mask-port semantics where needed, and possible consolidation into selector nodes only if the palette becomes hard to scan. |
| Binary morphology and label cleanup | Binary erosion/dilation/opening/closing/top-hat/black-hat/gradient, fill holes, remove small objects, connected-component labels, clear border, filter by volume, filter by property, relabel sequential. | Find label boundaries, kept/removed object count reporting, direct calibrated-unit size filtering if the table-based path proves too indirect, grayscale morphology, convex hull per object, and object boundary/thickness maps. |
| Tables and object measurements | First-class table outputs, object morphology, intensity measurements, calibrated physical variants, 3D mesh morphology, table merge, column selection, metadata annotation, grouped summaries, CSV/TSV output. | Optional mesh export/preview, specialist mesh repair/smoothing only if validated, richer intensity distribution columns if requested, and continued analytical validation. |
| Skeleton and network analysis | Skeletonize, skeleton keypoints, graph overlay, component labels, branch labels, branch pruning, component analysis, branch tables, branch summaries, graph node/edge tables, and whole-network metrics. | Skeleton/network validation report, specialist mitochondrial network indices, and broader progress/cancellation coverage for dense networks. |
| Colocalization and spatial association | Whole-image and ROI-masked Pearson/Manders/overlap/Costes metrics, colocalized-voxel RGB views, RACC-like index images, object colocalization metrics, label overlap, nearest-object distance, and event localization tables. | Validation figures/notebooks, RACC core/interop decision, and publication-facing example artifacts. |
| Graph platform | Typed ports, named heterogeneous inputs, dynamic multi-outputs, tunnels, graph search, notes, undo/redo, duplicate/delete, insert-on-wire mapping, manual/cached nodes, cache modes, memory guard. | Large-workflow navigation only if user workflows demand it; AI-assisted authoring after batch/provenance and validation are stronger. |

## P0: Platform Foundations

These are not all "nodes", but they decide whether future nodes can be correct
and reproducible.

| Item | Status | Why it matters | Next action |
| --- | --- | --- | --- |
| First-class labels | Implemented | Keeps object IDs separate from intensity arrays and binary masks. | Continue using nearest-neighbor interpolation for label geometry and registration work. |
| First-class tables | Implemented | Lets measurements, colocalization, and summaries produce analysis outputs instead of fake images. | Keep improving table provenance and validation examples. |
| Named heterogeneous ports | Implemented | Required for labels + intensity, image + mask, image + PSF, and channel-pair analyses. | Add optional input-port semantics only where a real workflow needs them. |
| Manual/cached execution | Baseline implemented | Prevents expensive measurements from live-recomputing on every parameter change. | Broaden cooperative cancellation and determinate progress coverage. |
| Spatial scope and units | Partially implemented | Users must know whether operations are per-plane or volumetric and whether units are pixels or physical units. | Normalize wording across nodes and add direct physical-unit controls where the table route is awkward. |
| Batch configuration and provenance | Partial | Batch outputs exist, but reproducibility needs saved run configuration and per-item manifests. | Keep as the next major workflow-hardening milestone after the 0.11 PSF/import foundation. |
| OME-Zarr preview and lazy execution strategy | Partial | Large OME data must not be accidentally materialized for previews or eager-only nodes. | Add pyramids, preview controls, operation capability declarations, and materialization warnings. |
| Optional microscope reader boundary | Initial foundation implemented | Nikon, Zeiss, Leica, Olympus, and other proprietary readers need optional dependencies and common metadata mapping without bloating the core install. | Validate the new optional reader routes against real sample files and expand per-format metadata extraction. |
| Acquisition metadata normalization | Partial | PSF generation and publication provenance need objective, channel, wavelength, scale, scene, source identity, and upstream processing flags regardless of file format. | Extend every reader to populate the same normalized `ImageState`/source metadata fields where possible. |
| Points output type | Not implemented | Spot detection, peak finding, puncta workflows, and nearest-neighbor analyses need coordinates as first-class outputs. | Design a points + table contract before adding blob/peak nodes. |
| Transform output type | Not implemented | Registration needs reusable estimated transforms and interpolation policy by semantic type. | Design before adding several registration algorithms. |
| Surface/mesh output type | Not implemented | Mesh morphology currently outputs tables only; mesh preview/export needs a proper graph contract. | Design before adding mesh preview/export nodes. |

## P1: Near-Term Node And Workflow Work

P1 work should stay aligned with the active release themes:
PSF/deconvolution, microscope import, batch provenance, large-data preview, and
scientific validation. New nodes should be small, clearly useful additions to
existing workflows.

### Label And Segmentation Polish

| Node or feature | Suggested backend | Notes |
| --- | --- | --- |
| Find Label Boundaries | `skimage.segmentation.find_boundaries` | Useful for QC overlays, boundary masks, and measurement sanity checks. |
| Kept/removed counts for label filters | Existing label-count paths | Inspector/status polish for `Filter Labels By Volume`, `Filter Labels By Property`, and related cleanup nodes. |
| Marker QC summaries | Existing maxima/watershed outputs | Report marker count, empty markers, labels without markers, and split/merge hints for watershed workflows. |
| Watershed validation workflow | Synthetic and microscopy-like phantoms | Not a new node, but a release-quality requirement for trusting the existing watershed family. |

### Restoration And PSF

The restoration direction is now covered by
[psf-and-deconvolution-plan.md](psf-and-deconvolution-plan.md).

| Node or feature | Suggested backend | Notes |
| --- | --- | --- |
| Born-Wolf PSF polish | Existing native implementation | Keep generated PSFs inspectable/saveable graph images with fresh `YX` or `ZYX` micrometer axes and explicit metadata override behavior. |
| Prepare / Validate PSF | NumPy/SciPy | Implemented first pass: center, clip negatives, force odd shape, normalize, and reject invalid kernels. Follow up with richer validation summaries if users need them. |
| Richardson-Lucy Deconvolution | `skimage.restoration.richardson_lucy`-compatible update | Implemented first pass with named `Image` and `PSF` inputs, manual/cached execution, leading-axis block processing, and Python export. |
| Richardson-Lucy TV Deconvolution | NumPy/SciPy local implementation | Implemented first pass as the primary microscopy restoration target. Real data validation, edge-policy options, and performance profiling remain. |
| PSF/deconvolution validation workflow | Synthetic bead and blurred-image phantoms | Synthetic 2D and 3D example workflows exist as `examples/synthetic-deconvolution-rl-tv.json` and `examples/synthetic-3d-deconvolution-rl-tv.json`; next step is real bead/microscopy validation before publication positioning. |

### Microscope Format Import

The goal is broad microscope acquisition import without making the core install
heavy, license-confusing, or tied to one vendor SDK. Readers should normalize
metadata into the same `ImageDataset`/`ImageState` contract used by OME-TIFF,
OME-Zarr, raster, and NumPy sources.

| Format family | Candidate route | Notes |
| --- | --- | --- |
| Nikon ND2 | Optional `nd2` reader first | Initial dispatch and metadata normalization exist; validate axes, scale, channel names/wavelengths, objective metadata, scenes/positions, and lazy access against real files. |
| Zeiss CZI / LSM | Optional `czifile` for CZI, TIFF/LSM path for LSM, BioIO fallback | Initial dispatch exists; validate scene, pyramid, compression, and metadata behavior against real CZI/LSM files. |
| Leica LIF / LOF / XLIF | Optional `liffile` first, BioIO/Bio-Formats fallback | Initial dispatch exists; preserve series/scene identity, scale, channels, and acquisition metadata where available. |
| Olympus OIR / OIB / OIF / VSI | Optional `oirfile`/`oiffile` where available, BioIO/Bio-Formats fallback | Initial dispatch exists; treat multi-series and tiled/large acquisitions as source-inspection problems, not hidden graph lists. |
| Other common microscope formats | Reader-plugin or Bio-Formats-backed fallback | Hamamatsu, PerkinElmer, whole-slide, and facility-specific formats should enter through the same optional boundary when demand appears. |

Reader requirements:

- content or metadata-based dispatch where possible, not suffix-only guessing;
- source inspection that can select series, scene, position, plate, well, field,
  or resolution level when the file contains multiple images;
- normalized axes, scale/units, origin, channel display metadata, wavelengths,
  objective NA/magnification, immersion/refractive index, detector/acquisition
  metadata where available, and raw metadata provenance;
- clear "supported", "experimental", and "metadata incomplete" states in the UI
  and docs;
- lightweight tests for dispatch and metadata normalization, with large or
  proprietary samples stored outside the repository when needed.

### Batch And Output Nodes

| Feature | Notes |
| --- | --- |
| Saved `batch_config.yaml` or equivalent | Should capture source bindings, selected outputs, naming templates, formats, and overwrite policy. |
| Per-item provenance manifest | Include workflow hash, VIPP/package versions, input identity, source metadata, outputs, and status. |
| Richer batch failure summary | Separate skipped, failed, completed, and partial outputs. |
| Semantic-axis iteration | Run the same graph over timepoints, channels, z-slices, or selected combinations when that is explicit and reproducible. |

## P2: Add After Core Platform And Validation Work

### Denoising And Restoration

| Node | Suggested backend | Why not P1 |
| --- | --- | --- |
| Total Variation Denoise | `skimage.restoration.denoise_tv_chambolle` | Useful, but parameters and nD semantics need careful docs. |
| Wavelet Denoise | `skimage.restoration.denoise_wavelet` | Useful for fluorescence noise, but channel/dtype semantics need care. |
| Estimate Noise Sigma | `skimage.restoration.estimate_sigma` | Best paired with denoising guidance and table/scalar output design. |
| Wiener Deconvolution | `skimage.restoration.wiener` | Should follow PSF handling and first deconvolution validation. |

`Non-Local Means` is already implemented as a slice-wise denoising node.

### Registration And Drift Correction

| Node | Suggested backend | Prerequisite |
| --- | --- | --- |
| Estimate Translation | `skimage.registration.phase_cross_correlation` | Transform output type. |
| Apply Translation | `scipy.ndimage.shift` | Interpolation policy for images, masks, and labels. |
| Register Stack To Reference | repeated phase cross-correlation | Batch/axis iteration semantics. |
| Affine Transform | `scipy.ndimage.affine_transform` or `skimage.transform.warp` | Transform representation plus metadata updates. |

Registration should not start as several disconnected image-output nodes. It
needs a transform contract, label-safe interpolation, metadata updates, and
validation phantoms.

### Geometry And Sampling

Already implemented: `Set Pixel Size / Units`, `Rescale Axes`, and calibrated
orthogonal projections.

Candidate nodes:

- resample to target pixel/voxel spacing;
- pad/crop to target shape;
- flip and rotate by right angles;
- arbitrary rotation;
- isotropic 3D resampling for visualization or analysis.

All geometry nodes must update axes, scale, translation, and units correctly.
Label inputs require nearest-neighbor interpolation.

### Shape And Structure

Candidate nodes:

- grayscale erosion, dilation, opening, and closing;
- label-safe erosion and label-safe expansion variants where `Expand Labels`
  is not enough;
- medial axis;
- convex hull per object;
- object boundary distance or local thickness.

Do not add these as a flat catalogue. Add them when they complete a documented
workflow such as boundary QC, thickness measurement, or cytoplasm/nucleus
region construction.

### Spot And Blob Detection

Candidate nodes:

- Blob LoG;
- Blob DoG;
- peak local maximum;
- spot intensity measurement;
- count spots per labeled object.

These should wait for the `points` output contract unless an interim
labels/table implementation is needed for a concrete puncta workflow.

### Colocalization Follow-Up

The main nodes are implemented. P2 work should be evidence and interop rather
than more metrics by default:

- deterministic validation report using known overlap and threshold scenarios;
- object-association validation for overlap, nearest distance, and event
  localization assumptions;
- RACC numerical-core decision: keep VIPP-owned implementation, share a small
  core with the standalone RACC plugin, or document intentional separation;
- optional action to hand selected VIPP channel outputs to the RACC plugin when
  both plugins are installed.

## P3: Specialized Or Expensive

These are plausible future families, but they should wait for stronger demand,
better platform contracts, or optional dependency boundaries.

- random walker segmentation;
- active contours and morphological snakes;
- graph-cut or graph-based segmentation;
- superpixels such as SLIC;
- optical flow;
- non-rigid registration;
- stitching and mosaics;
- object tracking across time;
- model-backed segmentation such as Cellpose, StarDist, or ilastik;
- mesh export, mesh preview/rendering, oriented bounding boxes, mesh repair, or
  specialist mesh inertia metrics;
- specialist mitochondrial fission/fusion/event tracking metrics;
- frequency-domain notch filtering;
- blind deconvolution;
- learned denoisers.

Model-backed nodes should be optional integrations with isolated dependencies,
model provenance, device selection, and reproducible model/version metadata.
They should not make the core plugin installation heavy.

## Defer

The following are common in general computer-vision libraries but are not good
near-term VIPP priorities:

- many separate edge operators beyond the existing Sobel, Canny, and Laplace
  family, such as Scharr, Prewitt, and Roberts;
- Hough line and circle transforms;
- contour hierarchy operations;
- polygon approximation and rotated boxes;
- ORB/SIFT-like keypoints and descriptors;
- template matching;
- face/object detection APIs;
- camera calibration and video-stream processing;
- broad colour-space conversion catalogues;
- artistic transforms, image pyramids as ordinary processing nodes, and
  inpainting;
- duplicate OpenCV versions of operations already handled well by SciPy or
  scikit-image.

## Palette Direction

The current palette is close to the desired shape. Keep it stable unless a
workflow becomes hard to find.

```text
Image Data
  Source & Output
  Axes & Regions
  Channels & Composites
  Utilities
  Math & Logic

Intensity & Contrast

Filtering
  Background Correction
  Smoothing & Denoising
  Edge & Detail
  Restoration & PSF

Projection

Segmentation
  Global Thresholds
  Local Thresholds
  Object Separation

Morphology
  Skeleton / Network QC

Label Operations
  Skeleton / Network QC

Measurements
  Skeleton / Network QC
  Tables

Colocalization & Spatial Analysis
```

Potential naming cleanup:

- Keep `Linear Scale + Offset` as explicit math. Use `Rescale Intensity` for
  contrast stretching.
- Rename binary `Top Hat` and `Black Hat` to `Binary Top Hat` and
  `Binary Black Hat` if grayscale morphology is added.
- Keep `Maximum Projection` as a quick shortcut, but prefer `Project Image`
  for semantic axes and reducer choice.
- Continue showing stack-processing notes for slice-wise filters, thresholds,
  edge detectors, and morphology operations.

## Example Target Workflows

### Fluorescent Nuclei

```text
Extract Channel
  -> Rolling-Ball Background or Subtract Background
  -> Gaussian Blur
  -> threshold
  -> Remove Small Objects
  -> Fill Holes
  -> Label Connected Components or watershed branch
  -> Measure Objects
```

### Split Touching Objects

```text
binary mask
  -> Euclidean Distance Transform
  -> H-Maxima Markers
  -> Marker-Controlled Watershed
  -> Filter Labels By Volume / Property
  -> Relabel Sequential
```

### Cell Regions From Nuclear Seeds

```text
nuclear labels
  -> Expand Labels
  -> optional image/mask constraint
  -> Measure Objects + Intensity
```

### Fluorescent Puncta

```text
Extract Channel
  -> background correction
  -> Difference of Gaussians or future LoG/blob detection
  -> future points/table output
  -> count or measure spots per labeled object
```

### Mitochondrial Morphology

```text
Extract mitochondrial channel
  -> background correction
  -> optional ridge/tubeness enhancement later
  -> threshold and binary cleanup
  -> labels for fragment/object analysis
  -> skeleton/network analysis for connectedness and branches
  -> optional mesh morphology for 3D object shape
```

Specialist mitochondrial indices should build on the generic object, mesh, and
skeleton table nodes rather than duplicating them.

### Colocalization

```text
Pixel/ROI workflow:
channel A + channel B + optional ROI mask
  -> colocalization metrics / RACC / colocalized voxels

Object workflow:
labels + channel A + channel B
  -> Object Colocalization Metrics
  -> Merge Tables with object morphology/intensity

Object association:
labels A + labels B or event labels + region labels
  -> overlap, nearest distance, or event localization table
```

### PSF And Deconvolution

```text
Image Source
PSF Image Source or Born-Wolf PSF
  -> Prepare / Validate PSF
  -> Richardson-Lucy Deconvolution
  -> Richardson-Lucy TV Deconvolution
```

## Decisions

Decided:

1. Labels, tables, and named heterogeneous ports are first-class platform
   contracts.
2. SciPy and scikit-image remain the default implementation stack.
3. OpenCV is not a required dependency unless profiling or a concrete workflow
   proves it is needed.
4. Registration requires a transform contract before adding several algorithms.
5. Deconvolution is active near-term work, but it must use first-class PSF
   images, named `Image`/`PSF` ports, and manual/cached execution.
6. Mesh tables are implemented; mesh preview/export waits for a surface-output
   contract.
7. Model-backed segmentation should be optional and dependency-isolated.
8. Proprietary microscope readers should be optional and normalized through the
   shared I/O model instead of becoming ad hoc widget code.

Still open:

1. Should `Filter Labels By Volume` gain direct calibrated-unit controls, or is
   `Measure Objects` plus `Filter Labels By Property` the better explicit path?
2. What is the exact `points` output contract: napari Points layer only,
   points plus table, or table-first with point inspection?
3. What is the minimal transform representation for translation and affine
   registration?
4. What is the minimal surface/mesh representation for preview/export without
   adding a heavy dependency?
5. Should threshold nodes remain named, or should an `Automatic Threshold`
   selector node replace part of the palette once node count grows further?
6. Should VIPP and the standalone RACC plugin share a small numerical core, or
   remain intentionally separate?
7. Which proprietary reader should be the first release gate after the
   OME/TIFF/Zarr foundation: ND2, CZI, Leica, Olympus, or Bio-Formats-backed
   fallback?

## Library Direction

Keep core dependencies focused:

- SciPy: multidimensional filtering, morphology, distance transforms,
  interpolation, and labeling.
- scikit-image: segmentation, labels, measurements, restoration, feature
  detection, registration, and transforms.
- ome-types, tifffile, ome-zarr, zarr, dask, fsspec, imageio, and pillow:
  normalized I/O and large-data foundations.

Optional or deferred dependencies should be attached to a workflow:

- OpenCV only for a clear capability/performance gap.
- `trimesh` only when mesh export/preview, oriented bounding boxes, inertia, or
  repair become first-class.
- `porespy` only for explicit pore/local-thickness/chord/network analysis.
- Nikon ND2, Zeiss CZI/LSM, Leica LIF/LOF/XLIF, Olympus OIR/OIB/OIF/VSI, and
  other proprietary microscope readers only behind optional reader boundaries
  with reviewed licensing and metadata fidelity.
- Bio-Formats can be a validation reference and broad fallback, but it should
  stay optional rather than a mandatory Java dependency.

## Primary References

- [scikit-image segmentation API](https://scikit-image.org/docs/stable/api/skimage.segmentation.html)
- [scikit-image morphology API](https://scikit-image.org/docs/stable/api/skimage.morphology.html)
- [scikit-image measurement API](https://scikit-image.org/docs/stable/api/skimage.measure.html)
- [scikit-image filters API](https://scikit-image.org/docs/stable/api/skimage.filters.html)
- [scikit-image restoration API](https://scikit-image.org/docs/stable/api/skimage.restoration.html)
- [scikit-image feature API](https://scikit-image.org/docs/stable/api/skimage.feature.html)
- [scikit-image registration API](https://scikit-image.org/docs/stable/api/skimage.registration.html)
- [scikit-image transform API](https://scikit-image.org/docs/stable/api/skimage.transform.html)
- [SciPy multidimensional image processing API](https://docs.scipy.org/doc/scipy/reference/ndimage.html)
- [OpenCV image-processing overview](https://docs.opencv.org/4.x/d7/da8/tutorial_table_of_content_imgproc.html)
