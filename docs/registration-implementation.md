# Registration implementation contract

Development status: 2026-09-22, unreleased after 0.16.0a1. Tracking issue:
[napari-vipp#68](https://github.com/rensutheart/napari-vipp/issues/68).
Public steps and interpretation belong to the companion manual's unreleased
registration how-to/reference, not this engineering record.

## Scope and ownership

- `core/transforms.py`: immutable `RegistrationGrid`, `TransformData`,
  `TransformState`; strict version-1 JSON import/export and output-state propagation.
- `core/registration.py`: CPU translation (scikit-image), rigid and affine
  (SimpleITK 2.5.6), selected-channel estimation, fixed-reference time series,
  exact-label or floating-point resampling, interpolation-valid coverage.
- `core/registration_nodes.py`: small operation declarations, execution adapters
  and revision-bound identities for anonymous graph sources.
- `core/registration_planning.py`: metadata-only grid/type projection for batch
  and compute planning, using a distinct `TransformPlan` with no invented matrices.
- `core/image_comparison.py`: same-grid, per-spatial-volume RMSE, Pearson
  correlation, SSIM and PSNR without hidden alignment or normalization.
- `core/pipeline.py`: dynamic pair/time and optional-mask ports, state injection
  and heterogeneous output metadata; shared execution/export/batch remain authoritative.
- `_widget.py` and existing inspector/graph helpers: transform cards, metadata,
  JSON saving, ordinary diagnostic-table access and background dispatch.

No registration GPU path is advertised. The 0.16.0a2 integration also includes
the independently qualified exact CPU median acceleration, sharing one SimpleITK
pin. Gaussian, background subtraction and deconvolution remain unchanged.
See the [combined readiness record](release-readiness-0.16.0a2.md) for integration
checks; the original qualification below applies to its recorded source snapshot.

## Inspector and next-step editing

Estimate Registration exposes both outputs together instead of using the generic
**Displayed output** selector. Its primary inspector result is the diagnostics
table, accompanied by a compact transform summary; the node card summarizes both
outputs. Full technical metadata remains available in a collapsed section.
Explicit export actions distinguish the diagnostics table from the versioned
transform JSON and allow both to be saved. A transform is not an image preview,
and these controls do not add a GUI transform-import path.

The shared **Next step** presentation starts with Estimate Registration only.
Its neutral **Add Apply Transform** action is an authored graph edit, not an
execution fallback: connect the exact moving-image source output already used
by the estimator and its transform output to a newly positioned, selected Apply
Transform node. Adding and wiring it form one Undo operation; the action does
not calculate, replace an existing connection or alter another branch.
When an Apply Transform already consumes that same image output and transform,
offer **Show Apply Transform** and retain an explicit option to add another.
This supports reuse without implying that every downstream image is the same
source or that an unrelated Apply Transform is the suggested continuation.

## Scientific contract

All spatial axes must be explicit YX or ZYX, with optional explicit T/C in any
order. T and C never enter a spatial optimizer. Time-series mode estimates one
transform for each whole volume against a chosen fixed reference time (default 0),
using one selected channel; application reuses it over compatible channels.
Pair mode requires an explicit time selection upstream if the input contains T.

Matrices map moving physical coordinates to reference physical coordinates in
canonical NumPy YX/ZYX order. Compatible length units become micrometres; unknown
physical calibration stays explicitly pixel coordinates. SimpleITK's fixed-to-moving
XYZ convention is inverted/permuted only at the boundary. The inverse mapping
used for resampling is `inverse(Gmoving) @ inverse(F) @ Greference`.
Reference spacing, origin and spatial extent govern output; input T/C are retained.

Translation requires equal spatial shape and sampling and uses subpixel
cross-correlation with explicit displacement/overlap-constrained periodic
disambiguation. Rigid/affine accept different compatible physical grids.
Initialization, multiresolution, full-voxel metric sampling and estimator mean
conditioning are implementation facts, not changes to the original image data.
SimpleITK optimization is not promised bitwise-identical across thread/hardware
configurations. Affine can alter measured shape and is never the default drift model.

Transforms carry physical grid/frame identities, time calibration, settings,
backend version and exact estimation-input hashes. Reuse accepts different content
(e.g. sibling channels/derived labels) only in the same frame and lattice.
Anonymous graph sources get a deterministic source-node/metadata/pixel-revision
identity inherited by derived outputs; equal names and shapes alone do not match.
Separate imported sibling data must carry a shared declared frame identity.
Connecting registration to previously cached anonymous sources refreshes their
source-frame metadata and invalidates sibling branches. Headless callers with
old identity-free cached image states receive an explicit refresh instruction.

Inputs remain unchanged. Automatic application uses nearest-neighbour for masks
and labels; nearest samples original integer values, including IDs beyond float64's
exact range. Linear intensity interpolation explicitly produces float64.
Coverage excludes coordinates without a full interpolation footprint; outside
fill is authored and never counted as valid coverage. Arrays are resampled once,
with no additional viewer affine. Outward boundary-coordinate excursions are
snapped only within eight float64 epsilons times the larger per-axis source/index-
map extent. This corrects matrix-composition roundoff, not genuine outside support;
the same coordinates govern interpolation and coverage. Constant/nonfinite estimation images, invalid
axes, excessive displacement, insufficient overlap and ambiguous shifts fail
with actionable errors rather than silently substituting identity.

## Comparison and progress

Compare Images requires identical physical grids, shapes and declared axes.
An optional Boolean mask must match that grid. Before/after comparisons should
share the same valid mask and authored intensity range. T/C blocks produce
separate rows; scalar reductions and SSIM slabs do not sample the population.
SSIM accepts only centres with a complete valid square/cubic window.
Constant-image correlation is undefined. Exact-match infinite PSNR is represented
by a status plus missing numeric value, never non-finite JSON. Metrics are image
QC, not proof of biological correspondence or improved colocalization.

The existing background/stale-result boundary is reused. Progress and cancellation
are checked between volumes, resampling slabs and optimizer iterations; an already
running FFT/native operation is not forcibly killed. No incomplete transform series
is published. Batch transform JSON uses the existing private staging and source-
verification gates; generated Python runs the same graph executor.

## Evidence and remaining qualification

The [synthetic record](registration-synthetic-qualification.md) reports only
the fixtures actually measured, including independent landmark errors. Tests
also cover source kinds, dynamic ports, workflow/Python roundtrips, immutable
cache payloads, batch JSON, progress and cancellation. The native install smoke
checks an independently generated 3D rigid phantom and exact wide-label sampling.
Final Windows qualification completed on 2026-09-22: 11,877 passed, 30 skipped,
two known expected failures, and verified unchanged source. Packaging, private
installed native smoke and the independent analytical fixtures also passed.
That recorded full-suite run predates the inspector and Next step follow-up
described above; it does not qualify those later UI changes.

The subsequent inspector/Next step follow-up passed 19 dedicated authoring tests
and 209 registration UI, inspector, multi-output, result-table, architecture and
example integration tests. Three affected guidance/layout cases were rerun after
final copy and icon adjustments and passed. Ruff, the plugin manifest and diff
whitespace checks passed. Narrow dark/light inspector captures were reviewed.
Coverage includes exact nonzero/tunnel ports, existing-consumer matching, one-step
Undo/Redo, busy guards, rollback, explicit exports, and a diagnostics pop-out that
stays attached to its own output while recalculating or inspecting another node.
These are focused UI checks, not a rerun of the full scientific qualification.

Passing synthetic examples is not universal registration validation. Large
rotations, weak texture, changing morphology, partial field overlap and differing
modalities remain user-review cases. Native Linux/macOS execution remains untested
locally and must be qualified separately before release. This development work
does not create an installer, publish a release or merge branches automatically.

## References

- [scikit-image registration](https://scikit-image.org/docs/stable/api/skimage.registration.html#skimage.registration.phase_cross_correlation)
- [SimpleITK registration overview](https://simpleitk.readthedocs.io/en/master/registrationOverview.html)
- [scikit-image structural similarity](https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity)
