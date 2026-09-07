# Registration, image comparison, and template matching

Status: planned feature scope for the **0.16 release series**; not implemented.
No release date or alpha milestone is assigned. Later phases are not release gates.
Reviewed: 2026-09-07.

## Recommendation

Build two small, complete workflows first: **align the same structure between
images** and **find structures that resemble a supplied example**. These solve
different problems; template matching should not be presented as general image
registration or segmentation.

The proposed first delivery contains five nodes, all supporting explicit 2D
`YX` images and true 3D `ZYX` volumes:

| Node | Inputs → outputs | Main purpose |
| --- | --- | --- |
| **Estimate Translation** | Reference image + Moving image → Transform + Diagnostics table | Measure a spatial shift, including subpixel shifts, without changing either image. |
| **Apply Transform** | Image/mask/labels + Transform + Reference grid image → Aligned data + Valid-region mask | Apply a measured alignment to the original pixels, related channels, or labels. |
| **Compare Images** | Reference image + Comparison image + optional Valid mask → Metrics table | Quantify structural similarity, correlation, and intensity error on an explicit shared region. |
| **Template Match** | Search image + Template image → Match-score image + Valid-score mask | Show where a supplied patch resembles the image. |
| **Find Peaks** | Score/intensity image + optional Valid mask → Locations and values table | Turn responses into reviewable detections using a threshold and minimum separation. |

Use reusable inspector overlays for alignment and detections, not additional
processing nodes just to draw a checkerboard or markers. Start table-first for
locations; a general points port is a separate platform milestone. Do not
represent detections as fake segmentation labels.

**0.16 core:** transform foundation → translation and comparison workflow →
template/detection workflow. Alpha milestones may deliver these complete
workflows incrementally once their gates pass. Time-lapse drift is a stretch
candidate; rigid/affine registration and the other later phases must not hold
up the core release.

## Fit with the existing roadmap

The [active roadmap](planning.md) assigns this work to 0.16. The
[node roadmap](node-roadmap.md) retains transforms as a prerequisite for
registration and uses table-first detection for the template workflow. Existing
correctness, source, memory, and reproducibility gates still apply; assigning a
release theme does not claim that the implementation is ready.

Existing building blocks include named multi-input/output ports, first-class
tables and labels, Crop Stack, channel/axis selection, calibrated metadata,
manual/cached execution, and shared interactive/batch/Python execution. There
is no first-class transform or points output yet. Equal-grid validation exists,
but registration needs its own contract: inputs being aligned must not already
be required to have identical origins.

Assumed first use cases are fluorescence-stack alignment, fiducial/reference
channel alignment, and repeatable structures with similar appearance. Different
stains without shared structure, tissue deformation, and generic object
recognition are not covered by this first scope.

For colocalization workflows, do not treat a higher colocalization value after
alignment as proof of correctness. Prefer shared fiducials or a stable reference
channel and independent alignment evidence over fitting unrelated structures
to resemble one another.

## Phase 0 — Reusable transform foundation

This is the main platform change, not merely an algorithm wrapper.

- Add an immutable, versioned transform value and state, with 2D/3D rank,
  named spatial axes, a homogeneous matrix, coordinate units, and moving and
  reference grid descriptors. Retain estimator settings, exact source revisions,
  implementation identity, and diagnostics in provenance.
- Define one public direction: **moving coordinates → reference coordinates**.
  Store the direction explicitly. For calibrated axes, map physical coordinates;
  for a user-selected pixel-coordinate mode, clearly record the absence of
  physical calibration. Never manufacture micrometer units.
- Convert explicitly at the sampling boundary. With grid maps `Gmoving` and
  `Greference` and forward transform `F`, the sampler uses
  `inverse(Gmoving) * inverse(F) * Greference`. SciPy expects an output-to-input
  mapping, so passing the forward matrix directly would reverse the operation.
  See [SciPy's affine sampler](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html).
- Extend typed wiring, cache ownership, memory accounting, inspector rendering,
  workflow validation, generated Python, and batch provenance. Reject transform
  connections to image-only consumers. Preserve old workflow loading and test
  any required schema migration; do not assume a new type is UI-only.
- Reuse requires compatible coordinate frames, not identical image content.
  A channel extracted from the moving acquisition can reuse its transform;
  an unrelated image with the same array shape cannot be assumed compatible.
- Support inspectable, schema-validated transform JSON export/import through
  the existing save/source surfaces. No pickled backend objects. Exported
  registration workflows must retain the transform, diagnostics, and source
  identities alongside any aligned images.

Start with translation matrices. Keep the representation extensible to global
rigid/affine transforms; defer displacement fields and transform-series payloads.
Audit `core/pipeline.py`, `core/execution.py`, `core/metadata.py`, `core/grid.py`,
cache/export/batch paths, and UI type dispatch before implementation. Follow the
[developer recipe](developer-notes.md#add-or-change-a-scientific-operation).

## Phase 1 — Translation alignment

### Estimate Translation

Use [scikit-image phase cross-correlation](https://scikit-image.org/docs/stable/api/skimage.registration.html#skimage.registration.phase_cross_correlation).
It supplies a moving-to-reference pixel shift and subpixel refinement. Its FFT
estimate is periodic, so unwrap/disambiguate candidates against real overlap;
do not let a wrapped shift appear trustworthy. Normalized error is a diagnostic,
not a calibrated confidence percentage.

Initial contract:

- Accept one finite scalar spatial image/volume per input, with equal spatial
  shape and compatible sampling. Allow different origins and incorporate them
  in the transform. Reject unresolved axes or inconsistent calibration. Unequal
  sampling requires an explicit resampling operation, not an automatic repair.
- Require users to select a channel/time point upstream. Never optimize across
  `C` or `T`, treat Z as time, or silently project a volume. A 2D shift applied
  across Z is a later explicit mode, not implicit 3D registration.
- Expose **Spatial scope**, **Subpixel precision**, **Maximum displacement**,
  and **Minimum overlap** with valid bounds. Put phase versus unnormalized
  correlation under advanced settings and explain the noise trade-off. Specify
  and test VIPP's overlap calculation; the backend's overlap-ratio setting is
  not an unmasked-overlap gate.
- Defer masks initially. The backend's masked route does not use subpixel
  refinement and lacks the same error output; future mask support must show
  these differences instead of leaving ineffective controls enabled.
- Reject constant/empty/non-finite input and impossible overlap. Report ambiguous
  peaks, implausible shifts, and weak structural evidence. Block application
  when configured quality bounds fail; never substitute an identity transform
  and report success. Numerical cutoffs require phantom/real-data evidence.

### Apply Transform

Keep estimation and resampling separate. A workflow can estimate on a clearly
preprocessed reference channel and apply that transform once to original data:

```text
Reference + moving reference channel → Estimate Translation → Transform
Original moving image / sibling channel / labels + Transform + Reference
                                                    → Apply Transform
```

- The reference input defines output spatial extent, origin, scale, and units;
  validate it against the transform's target frame. Its intensities are not
  sampled. Preserve the moving input's non-spatial
  axes/channel identity. Offer an explicit **Apply the same transform to every
  channel/time point** scope with compatibility checks, not inferred broadcasting.
- Show **Interpolation**, **Output grid: Reference**, **Outside-image fill**,
  and **Output type**. Start with nearest-neighbor and linear interpolation.
  Masks and labels require nearest-neighbor and retain their semantic type/IDs.
  Linear image interpolation returns an explicitly described floating result;
  no implicit integer rounding, intensity normalization, or unsafe wide-integer
  conversion. Higher-order interpolation and expanded canvases can follow later.
- Return a valid-region mask describing pixels with complete supported sampling
  coverage. Define it against the interpolation footprint. Report lost field of
  view and exclude synthetic fill from registration quality measurements.
- Resample pixels onto the reference grid and carry the transform as history;
  do not also apply it as a viewer transform, which would align the image twice.

Inspector acceptance includes reference/moving identification, shift per named
axis in pixels and known physical units, overlap and diagnostic status, and an
overlay/checkerboard comparison on the valid region. Reuse existing inspection
where possible; display controls must not affect scientific outputs.

### Compare Images: Similarity And Quality Evidence

**Include SSIM, but do not label it registration confidence.** The
[original SSIM work](https://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf) concerns
full-reference perceptual image quality, not proof of geometric correspondence.
Our recommendation is to combine several measurements with overlap and visual
inspection. These comparisons can also support restoration workflows when a
meaningful reference exists; the noisy input is not automatically ground truth.

Use one reusable **Compare Images** node rather than a node for every metric.
Keep these outputs separate from the estimator's optimization score and from
biological colocalization measurements.

| Metric | Scope | Interpretation and boundary |
| --- | --- | --- |
| **Structural similarity (SSIM)** | 0.16 core | Local structural resemblance; higher is more similar. Sensitive to the chosen intensity range and spatial window. Not a confidence percentage or a universal pass threshold. |
| **Correlation (Pearson / zero-mean normalized correlation)** | 0.16 core | Signed intensity-pattern agreement. Useful alongside SSIM; report constant-input correlation as undefined, not perfect alignment. |
| **Intensity error (RMSE)** | 0.16 core | Lower means less pixelwise disagreement. Retain intensity units; comparable intensities are required. |
| **Peak signal-to-noise ratio (PSNR)** | 0.16 core, optional selection | A range-dependent error measure in dB, useful for reference-based restoration comparisons; not an independent geometric check. |
| **Mutual information / normalized MI** | Later, alongside different-contrast registration | Candidate when intensities do not correspond directly. Qualify the exact definition, histogram settings, sampling, and small-overlap behavior first. |
| **Dice/IoU, surface distances, landmark error** | Separate follow-up | Compare masks, boundaries, or paired landmarks rather than treating label IDs as intensities. Require explicit correspondence, units, and empty-set policies. |

The [scikit-image metrics API](https://scikit-image.org/docs/stable/api/skimage.metrics.html)
provides SSIM and PSNR without another dependency. SSIM supports same-shaped
n-dimensional inputs, but a volumetric implementation still needs VIPP's own
3D evidence. The [SimpleITK registration framework](https://simpleitk.readthedocs.io/en/master/registrationOverview.html#similarity-metric)
is a later reference for correlation and mutual-information objectives; its
optimizer values must not be confused with this node's reported metrics.

Scientific contract:

- Compare scalar `YX` or `ZYX` inputs on the same physical/pixel grid. Select
  channels/time upstream. No hidden registration, projection, resampling, or
  normalization. Reuse reviewed correlation primitives where appropriate,
  without inheriting unrelated colocalization thresholds.
- Require a persisted **Intensity range** for SSIM/PSNR; never take it from
  display contrast or independently fit each image. Expose **SSIM window size**
  in pixels/voxels: odd, at least 3, and fitting every spatial axis. Start with
  an explicitly recorded uniform window and covariance convention. A cubic
  voxel window is not physically isotropic on anisotropic data; record scales
  and defer physical-radius/Gaussian alternatives until qualified.
- Accept a same-grid valid mask, including Apply Transform's coverage mask.
  Exclude synthetic fill. For SSIM, retain only window centers whose entire
  footprint lies inside the valid region and image boundary; masking the
  center alone or replacing invalid pixels with zero is insufficient.
- A before/after comparison must use the same reference-grid region, metric
  settings, and intensity range for both measurements. Intersect their valid
  regions, report retained coverage, and do not claim improvement by comparing
  different populations. If grids differ, explicit Apply Transform operations
  prepare the baseline and aligned comparisons; Compare Images never does so.
- Preserve buffers and reject non-finite selected data. Report undefined
  metrics or insufficient valid windows with a reason and sample count, not
  a fabricated zero. Exact identity has zero RMSE and infinite PSNR; represent
  the latter with an explicit status rather than non-standard JSON Infinity.

Return a compact table with metric, value, units, direction, valid pixel/window
count, coverage, and status; retain settings and source revisions in provenance.
The inspector shows readable values and warnings, with a before/after preset
using the same table outputs. An SSIM map plus its valid-window mask is a useful
optional follow-up, not a reason to allocate full-volume maps when only a table
was requested. Use bounded reductions/SSIM tiles with tested halo parity.

Acceptance adds identity, known shifts, gain/offset, blur/noise, contrast
inversion, constant data, small/empty overlap, anisotropic 3D, masked-window
boundaries, and matched before/after regions. Include misleading-score examples:
high similarity alone must never produce an automatic "registration verified"
badge. Reuse the normal batch/table export and cancellation contracts.

## Phase 2 — Template matching and useful detections

### Template Match

Use [scikit-image normalized template correlation](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.match_template),
already available in the dependency set. Supply the template using Crop Stack
plus explicit channel/time selection, or a separate Image Source. A graph
preset can demonstrate this; no dedicated template-drawing subsystem is needed.

V1 searches at the template's supplied size and orientation. It does not search
rotations/scales, tolerate arbitrary deformation, or generate an object boundary.
Check equal spatial rank and sampling, template extents no larger than the search
image, and nonconstant finite template content. No hidden resizing or color
conversion. Template origin need not match the search image origin.

Use full-template placements only initially, avoiding artificial padded edge
matches. The raw valid response is smaller than the search image. Give that
score grid a center-based origin offset `(template_shape - 1) / 2` pixels from
the search origin; preserve half-pixel centers for even sizes. Carry template
extent and search-grid identity so detections can report both center coordinates
and bounding boxes without an off-by-one interpretation.

Scores describe similarity, not probability. Return a valid-score mask for
undefined zero-variance search windows and connect it to Find Peaks;
never turn a backend placeholder zero into a valid detection. Keep signed
scores visible, with contrast-inverted matches outside the default positive
threshold. Benchmark memory for full 3D responses before claiming large-volume
support. FFT working memory matters as much as the output image.

### Find Peaks

Use local maxima as candidates, with a VIPP-defined deterministic ordering and
suppression policy. The core reference is
[scikit-image peak detection](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.peak_local_max).

Expose **Minimum value**, **Minimum separation**, **Maximum detections**, and
**Border exclusion**. Separation must distinguish pixel distance from calibrated
physical distance; use axis scales for anisotropic volumes. Define plateau/tie
handling, apply score thresholds before suppression, and record when a result
cap truncates detections. An empty table is a valid "no matches" result.

Return IDs, named pixel-center coordinates, physical coordinates where known,
and scores; add template bounds when that metadata exists. Define optional
mask-port semantics as part of this phase; require the matching valid-score mask
for template responses, and wire it in the supplied graph preset. Show
non-editing detection markers in
the inspector and export ordinary tables. Do not promise editable napari Points
or arbitrary points-input nodes until the points contract is implemented.

This node also prepares a useful path toward puncta detection. If the real goal
is generic round bright spots rather than resemblance to a specific patch,
**Blob LoG/DoG** is a better follow-up candidate than an ever-larger template bank.

## Later phases and deliberate exclusions

| Phase | Candidate | Reason and boundary |
| --- | --- | --- |
| 3 | **Estimate Drift** | One translation per explicit time point of `TYX`/`TZYX`, estimated from a selected stable channel against one fixed reference frame. Extend the transform type to an indexed series, output a drift/quality table, and apply once per frame across related channels. No arbitrary batch-axis iterator required. |
| 3 follow-up | Previous-frame drift estimation | Compose transforms relative to an anchor and resample the original frames only once. Expose accumulated-error risk and failed-frame policy; no silent interpolation over failures. |
| 4 | **Estimate Rigid Transform**, then **Estimate Affine Transform** | Rigid adds rotation without shape change; affine additionally changes scale/shear and can affect morphology measurements. Reuse Apply Transform. Require real cases that translation cannot solve. |
| Later | **Estimate Transform from Landmarks** | Useful for fiducials or different stains, but requires reproducible paired-point inputs, correspondence IDs, residuals, degeneracy checks, and outlier policy first. |
| Defer | Deformable registration, optical flow, stitching, tracking, ORB/SIFT, rotation/scale template banks | Each introduces separate scientific assumptions, outputs, costs, or UX. Deformation can change the morphology being measured; these are not simple extensions of a shift node. |

For rigid/affine, evaluate an optional SimpleITK integration in a bounded spike.
Its [registration framework](https://simpleitk.readthedocs.io/en/master/registrationOverview.html)
supports global transforms, intensity metrics, and multiresolution optimization.
Qualification must cover physical-axis conversion, initialization, sampling
seed/threading reproducibility, convergence/failure reporting, package size,
licensing, and supported desktop wheels. Correlation for similar contrast and
mutual information for different contrast are candidates, not promises of
successful registration. Do not add SimpleITK or OpenCV to core dependencies now.

## Delivery gates

1. **Agree scope and open an implementation issue.** Confirm one alignment
   example and one template-detection example with the user before tuning defaults.
2. **Prove numerical and coordinate contracts headlessly.** Test analytical
   asymmetric 2D/3D phantoms with known integer/subpixel shifts; anisotropic
   scales, different origins, inverse/sign conventions, ambiguous periodic
   patterns, noise, zero variance, NaN/Inf, no overlap, and invalid parameters.
3. **Prove outputs and detection semantics.** Test read-only inputs, exact label
   IDs including wide integers, dtype/range policies, fill/valid masks, even/odd
   template centers, border placements, repeated/absent matches, tied peaks,
   calibrated separation, and threshold/cap behavior. Define tolerances against
   ground truth, not only agreement with the same backend call.
4. **Prove the whole workflow.** Verify multi-output wiring, source changes,
   cache invalidation, undo/redo, schema round trips, batch decisions/failures,
   transform/table persistence, and shared Python/interactive execution.
   Exercise the supported minimum dependency versions as well as the qualified
   desktop environment; do not assume every scikit-image version has identical
   signatures or edge-case behavior.
5. **Make expensive work understandable.** Use existing manual/cached execution,
   elapsed time, operation/volume progress, bounded memory preflight, and stale
   result protection. Native FFT calls may not be interruptible; report that
   boundary honestly and check cancellation between calls/blocks. Never publish
   a cancelled transform, partial detection table, or partially aligned volume.
6. **Validate the user journey.** Supply 2D and 3D example workflows with known
   answers plus licensed representative microscopy evidence. Update the
   dedicated vipp-mkdocs manual with short alignment and template tutorials,
   screenshots, and limitations when implemented; retain this engineering plan
   here. CPU is the initial reference. GPU support needs its own correctness,
   memory, cancellation, and end-to-end benefit evidence.

The first useful milestone is **a reusable measured translation, correctly
applied and visually reviewable**, not a large registration algorithm menu.
