# PSF Generation And Deconvolution Plan

Status: first implementation complete; follow-up validation and polish tracked below
Last reviewed: 2026-07-10

This plan tracks VIPP's restoration work: first-class point-spread functions
and PSF-aware deconvolution. The immediate implementation target is
Richardson-Lucy total-variation deconvolution for microscopy, with ordinary
Richardson-Lucy included as the baseline comparator and simpler first node.

This document is intended as a handoff for an implementation agent. It keeps
the deconvolution work separate from the concurrent microscope-file import work:
the deconvolution nodes consume normalized `ImageState` metadata, but they
should not implement Nikon/Zeiss/Leica/Olympus readers themselves.

## Product Contract

PSFs are graph images. Users should be able to generate, inspect, pin, save,
load, prepare, and reuse a PSF before wiring it into a deconvolution node.
Measured or externally generated PSFs should remain supported.

The target workflow is:

```text
Image Source
  -> optional background correction / crop / channel selection
  -> Born-Wolf PSF or measured PSF Image Source
  -> Prepare / Validate PSF
  -> Richardson-Lucy TV Deconvolution
  -> Save Image / Batch Output
```

The first implementation should keep the nodes explicit rather than hiding
preparation inside deconvolution. That makes it possible to inspect a generated
or measured PSF before spending time on an expensive restoration.

## Current Baseline

Implemented foundation:

- OME-TIFF import preserves objective NA, nominal magnification, immersion, and
  objective-settings refractive index when OME-XML provides them.
- `Born-Wolf PSF` consumes normalized axis, channel, and objective metadata.
- Generated PSFs carry fresh `YX` or `ZYX` micrometer axes rather than blindly
  inheriting the reference image axes.
- Microscope acquisition import now has an optional-reader boundary for ND2,
  CZI/LSM, Leica, Olympus, and BioIO/Bio-Formats-style fallback routes.
- Acquisition metadata can now carry a nullable upstream deconvolution flag plus
  a detected deconvolution method/source note.
- The graph already supports named heterogeneous input ports.
- Manual/cached execution, progress reporting, and cooperative cancellation
  exist for expensive nodes.

Implemented first pass:

- `Prepare / Validate PSF` node.
- Baseline `Richardson-Lucy Deconvolution` node.
- `Richardson-Lucy TV Deconvolution` node.
- Named `Image` and `PSF` ports with manual/cached execution.
- Deterministic synthetic deconvolution samples plus 2D and 3D review workflows:
  `examples/synthetic-deconvolution-rl-tv.json` and
  `examples/synthetic-3d-deconvolution-rl-tv.json`.
- User-facing restoration documentation and parameter caveats in
  `docs/user-guide.md`.

Still missing or intentionally deferred:

- real microscopy/PSF validation datasets and comparison reports;
- reflect-padded or otherwise explicit edge/boundary policy options;
- performance tuning for large 3D volumes;
- release-facing screenshots or tutorial walkthroughs;
- GPU acceleration, blind deconvolution, spatially variant PSFs, and
  vendor-specific file import remain out of first-pass scope.

## Source Notes

Use official and primary sources for implementation choices:

- scikit-image's `richardson_lucy` is n-dimensional and exposes `num_iter`,
  `clip`, and `filter_epsilon`. Its docs warn that default clipping can lose
  information unless the image is normalized to the expected range:
  <https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.richardson_lucy>
- RL-TV is commonly described as an RL multiplicative update with an additional
  total-variation denominator term. ImageJ Ops documents the same family and
  shows the update form used by Dey et al.:
  <https://imagej.net/libs/imagej-ops/deconvolution>
- Dey, Blanc-Feraud, Zimmer, Kam, Roux, Olivo-Marin, and Zerubia proposed
  Richardson-Lucy deconvolution with total-variation regularization for 3D
  microscopy to suppress unstable oscillations while preserving edges:
  <https://pubmed.ncbi.nlm.nih.gov/16586486/>

Do not copy GPL implementation code from Fiji/ImageJ plugins or Bio-Formats
components. Reimplement the small numerical core from the published algorithm
and cite the method.

## Scope Boundaries

In scope for the deconvolution agent:

- pure NumPy/SciPy/scikit-image operations;
- new operation specs under `Filtering -> Restoration & PSF`;
- named `Image` and `PSF` ports;
- manual/cached execution for deconvolution;
- progress/cancellation per iteration;
- output metadata that follows the deconvolved image, not the PSF;
- unit tests and a small deterministic synthetic validation workflow.

Out of scope for this agent:

- deeper proprietary microscope-reader validation beyond the first optional
  reader foundation;
- GPU/CUDA/OpenCL acceleration;
- blind deconvolution;
- spatially variant PSFs;
- adaptive optics or aberration estimation;
- non-circulant edge handling beyond a simple padding policy;
- UI beyond normal palette/parameter/inspector behavior already provided by
  `OperationSpec`.

## Node Contracts

### Prepare / Validate PSF

Purpose: turn a generated or measured PSF image into a clean scalar kernel.

Input/output:

```text
image -> image
```

Parameters:

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| Center mode | choice | `Peak` | `None`, `Peak`, `Centroid`. Peak is safest for generated PSFs; centroid is useful for measured bead stacks. |
| Clip negatives | bool | `true` | PSFs should be non-negative for RL. |
| Normalize sum | bool | `true` | Required for stable intensity behavior. |
| Minimum valid sum | float | `1e-12` | Raise a clear error if the PSF is empty after clipping. |
| Force odd shape | bool | `true` | Pad/crop by one pixel where needed so the PSF has a central sample. |
| Crop empty border | bool | `false` | Optional, conservative default off. |

First-pass behavior:

- Convert to `float32`.
- Replace non-finite values with zero.
- Optionally clip negatives to zero.
- Center the PSF by integer shift only. Do not do subpixel interpolation in the
  first pass.
- Normalize by sum when requested.
- Return only the prepared PSF image. A table/report output can be added later
  if users need persistent validation metrics.

Important errors/warnings:

- empty PSF after clipping;
- PSF max not finite;
- PSF has too few spatial dimensions;
- PSF has channel/time axes and cannot be interpreted as a scalar kernel.

### Richardson-Lucy Deconvolution

Purpose: baseline PSF-aware deconvolution, useful by itself and as the
comparison point for RL-TV.

Input/output:

```text
Image + PSF -> image
```

Input ports:

| Port | Type | Title |
| --- | --- | --- |
| `image` | `image` or `array` | Image |
| `psf` | `image` or `array` | PSF |

Parameters:

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| Spatial processing | choice | `Auto from axes` | Same contract as segmentation: `2D YX`, `3D ZYX`, or auto. |
| Iterations | int | `25` | Keep modest; iteration count itself regularizes ordinary RL. |
| Normalize PSF | bool | `true` | Normalize defensively inside the node even if users skipped preparation. |
| Clip negative input | bool | `true` | Microscopy intensities for RL should be non-negative. |
| Clip output negative | bool | `true` | Keep restored fluorescence physically meaningful. |
| Preserve input scale | bool | `true` | Scale output by the original input max/scale after internal normalization. |
| Filter epsilon | float | `1e-12` | Passed to scikit-image or native denominator guard. |

Execution:

- Manual/cached by default.
- Report progress per spatial block and iteration where possible.
- Support cooperative cancellation between iterations.

Algorithm:

- First pass may call `skimage.restoration.richardson_lucy` for each spatial
  block.
- Pass `clip=False` to avoid scikit-image's `[-1, 1]` compatibility clipping,
  then apply VIPP's own negative clipping/output scaling policy.
- Normalize the PSF to sum 1 before calling the algorithm.
- Process leading non-spatial axes independently, so `TCZYX` data can run per
  timepoint/channel while each `ZYX` block is deconvolved volumetrically.

Shape rules:

- PSF dimensionality must equal the resolved spatial dimensionality.
- A 2D PSF can only be used in `2D YX` processing.
- A 3D PSF can only be used in `3D ZYX` processing.
- For channel-specific PSFs, first-pass workflow should use `Split Channels`
  and one deconvolution node per channel. Generated Born-Wolf PSFs may expose
  one output port per channel, but deconvolution should still consume one
  scalar PSF per branch; do not implement multi-channel PSF broadcasting yet.

### Richardson-Lucy TV Deconvolution

Purpose: primary microscopy deconvolution target. Ordinary RL can amplify noise;
the TV term should suppress unstable oscillations while preserving stronger
edges.

Input/output:

```text
Image + PSF -> image
```

Input ports are the same as baseline Richardson-Lucy.

Parameters:

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| Spatial processing | choice | `Auto from axes` | 2D or 3D per spatial block. |
| Iterations | int | `25` | Same starting range as baseline RL. |
| TV regularization | float | `0.002` | Also called lambda. Needs conservative bounds and docs. |
| TV epsilon | float | `1e-6` | Smooths gradient norm near zero. |
| Normalize PSF | bool | `true` | Same as baseline. |
| Clip negative input | bool | `true` | Same as baseline. |
| Clip output negative | bool | `true` | Same as baseline. |
| Preserve input scale | bool | `true` | Same as baseline. |
| Filter epsilon | float | `1e-12` | Denominator guard for convolution estimate. |
| Denominator floor | float | `0.05` | Prevent TV denominator from crossing zero. |

Native update sketch:

```text
estimate = initial non-negative image
for iteration:
    blurred = convolve(estimate, psf)
    ratio = observed / max(blurred, filter_epsilon)
    correction = convolve(ratio, flipped_psf)
    tv = div(grad(estimate) / sqrt(sum(grad(estimate)^2) + tv_epsilon^2))
    denom = max(1 - tv_weight * tv, denominator_floor)
    estimate = estimate * correction / denom
    estimate = max(estimate, 0)
```

Implementation notes:

- Use `scipy.signal.fftconvolve` or a local helper for same-size convolution.
  Add a single boundary policy first; `reflect pad then crop` is usually a
  better user default than pure zero padding, but the implementation should be
  tested for small arrays and odd/even PSFs.
- Flip the PSF across all spatial axes for the RL correction pass.
- Compute gradient/divergence over spatial axes only.
- Use `float32` output unless internal calculations need `float64` for
  stability.
- Keep all guard values explicit; never allow NaN/Inf to propagate silently.
- If `tv_weight == 0`, RL-TV should numerically reduce to the native ordinary
  RL update.

## Axis And Metadata Policy

The deconvolution output should inherit axes, channel metadata, source identity,
and acquisition metadata from the image input. It should not inherit axes from
the PSF, because the result lives in the image coordinate system.

Implementation touchpoints:

- `core/operations.py`: add pure functions and helpers.
- `core/pipeline.py`: import functions, add `OperationSpec`s, named input
  ports, manual execution policy, and include deconvolution operations in
  `SPATIAL_OPERATIONS`.
- `core/metadata.py`: add operation-history strings. Default
  `transform_multi_input_image_state` should be sufficient for axes because it
  uses the first input state.
- `core/export.py`: no special handling should be needed if the operation
  signatures use serializable parameters.
- `_tests/test_operations.py`: operation, pipeline, metadata, export, and
  cancellation tests.
- `_tests/test_widget.py`: palette grouping and manual-node UI tests if needed.

Potential additional pipeline helper:

- If deconvolution needs image-axis names/scales in the operation function,
  add a small metadata injection set similar to
  `LABEL_METADATA_MULTI_INPUT_OPERATIONS`, but do not couple it to label logic.

## Internal Data Policy

Recommended first-pass numeric policy:

- Convert image blocks to `float32`.
- Replace non-finite image values with zero.
- Clip negative image values when requested.
- Normalize each spatial block by a robust or absolute scale before
  deconvolution, then restore scale if `Preserve input scale` is on.
- Normalize PSF by sum.
- Return `float32`, not the input integer dtype. Deconvolution is analytical
  restoration, not a display-preserving dtype transform.

Open decision:

- Whether normalization should use block max, dtype range, or robust percentile
  range. Start with block max because it is deterministic and easy to explain;
  add robust scaling only if real images make max scaling brittle.

## Boundary Handling

Boundary policy is a major source of deconvolution artifacts. Do not overbuild
it in the first pass, but document it clearly.

Recommended first implementation:

1. Pad each spatial block by half the PSF size.
2. Use reflect padding by default.
3. Run same-size convolution on the padded block.
4. Crop the restored estimate back to the original spatial block.

Initial implementation note: the first RL and RL-TV nodes use the same
`scipy.signal.convolve(..., mode="same")` convolution semantics as
scikit-image's Richardson-Lucy baseline. That keeps `TV regularization = 0`
close to ordinary RL for validation fixtures. Reflect-padded edge handling
remains a follow-up boundary-policy improvement.

Candidate future choices:

- `Reflect` default;
- `Edge`;
- `Constant zero`;
- `None / circular`.

Non-circulant edge handling and vector acceleration are explicitly later scope.

## Validation Plan

### Unit Tests

Add deterministic tests for:

- PSF preparation clips negatives, centers peak, forces odd shape, and
  normalizes sum.
- Baseline RL accepts 2D image + 2D PSF and returns finite non-negative
  `float32` with the original shape.
- Baseline RL improves a simple blurred point/blob image compared with the
  blurred input, using a small tolerance and a deterministic PSF.
- RL-TV with `tv_weight=0` closely matches native ordinary RL for a small
  fixture.
- RL-TV with nonzero TV produces finite output and lower total variation than
  unregularized RL on a noisy blurred fixture, without asserting excessive
  exactness.
- 3D mode accepts `ZYX` image + 3D PSF.
- 2D mode processes leading axes independently.
- mismatched PSF dimensionality raises a clear `ValueError`.
- progress reports per iteration and cancellation stops between iterations.
- pipeline output metadata follows the image input axes/channels.
- generated Python export includes the deconvolution operation call.

### Example Workflow

Add an example such as:

```text
VIPP synthetic deconvolution sample
  -> Born-Wolf PSF or synthetic Gaussian PSF source
  -> Prepare / Validate PSF
  -> Richardson-Lucy Deconvolution
  -> Richardson-Lucy TV Deconvolution
  -> Batch Output
```

The sample should include:

- a clean bead/blob ground truth;
- blurred/noisy observation;
- known PSF;
- optional anisotropic 3D variant.

Example assertions:

- deconvolution increases peak sharpness versus blurred input;
- RL-TV has lower noise/TV than ordinary RL at the same iteration count;
- restored image remains non-negative and finite;
- axes and physical scale are preserved.

### External Reference Checks

Later, compare against Fiji/ImageJ Ops or another established package using
user-provided reference data. Do not include large or license-restricted files
in the repository.

## Implementation Order

Completed:

1. Add `Prepare / Validate PSF`.
2. Add baseline `Richardson-Lucy Deconvolution` using scikit-image-compatible
   per-block iteration with VIPP's normalization and clipping policy.
3. Add native `Richardson-Lucy TV Deconvolution`.
4. Add tests for operation behavior, pipeline ports, manual execution,
   metadata, export, progress, and cancellation.
5. Add deterministic synthetic deconvolution samples and 2D/3D example workflows.
6. Update user docs with measured-PSF workflow guidance and caveats.

Follow-up backlog:

1. Add release-note/changelog wording for the new nodes and example workflow.
2. Add a short screenshot/tutorial walkthrough after the UI copy settles.
3. Validate against real bead PSFs and microscopy images from at least one 2D
   and one 3D acquisition.
4. Compare numerical behavior against established references such as Fiji/ImageJ
   Ops using user-provided or public data.
5. Add boundary-policy options, starting with reflect padding and documented
   crop-margin guidance.
6. Profile large 3D volumes and decide whether chunking, vector acceleration,
   or optional GPU work is warranted.
7. Consider Wiener/unsupervised Wiener after PSF handling and RL-TV are stable.

## Later Deconvolution Nodes

Candidate later nodes:

- Wiener-Hunt deconvolution via `skimage.restoration.wiener`;
- unsupervised Wiener via `skimage.restoration.unsupervised_wiener`;
- vector-accelerated RL-TV;
- non-circulant edge-aware RL;
- blind deconvolution;
- GPU-backed deconvolution as an optional acceleration path.

These should not block the first RL-TV milestone.

## Open Questions

1. Should `Prepare / Validate PSF` eventually output both an image and a
   validation table?
2. Should baseline RL and RL-TV be separate nodes, or one node with a
   regularization selector? Separate nodes are clearer for early validation.
3. What default TV weight is safest for fluorescence microscopy examples?
4. Should output scaling preserve block max, dtype range, or robust percentile
   range?
5. Should reflect padding be fixed initially, or exposed as an advanced
   parameter immediately?
6. Should the release gate require both baseline RL and RL-TV, or is baseline
   RL acceptable as an intermediate implementation checkpoint?
