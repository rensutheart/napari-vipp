# Template matching and detections: implementation contract

Status: unreleased development after 0.16.0a2. This is engineering evidence,
not a second user guide or a microscopy-validation claim. The Phase 2 scope
in [the registration/template plan](registration-and-template-matching-plan.md)
is implemented by `core/detection.py`, the immutable schemas in
`core/detection_metadata.py`, and `core/detection_nodes.py`.

## Scientific population and algorithm

Both nodes consume one explicit, finite scalar `YX` image or `ZYX` volume.
No channel/time selection, axis permutation, projection, resizing, or image
alignment occurs. Array shape/dtype and carried state must agree. Boolean,
signed/unsigned integer values within the exact float64 integer interval
`[-2**53, 2**53]`, and floating types through float64 are supported. Complex,
empty, NaN/Inf, extended-precision floats, and unsafe wide integers fail.
This includes nonfinite pixels outside a supplied Find Peaks mask.

Template Match uses
[scikit-image `match_template`](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.match_template)
for zero-mean normalized correlation, with `pad_input=False`. Its documented
reference is J. P. Lewis, *Fast Normalized Cross-Correlation* (1995).
Template and search ranks and sampling must agree; compatible length units
are converted for that comparison, with relative spacing tolerance `1e-9`.
Their origins need not agree. All spatial units must be compatible lengths,
or all must describe index/pixel coordinates.

The template must be nonconstant and fit every search dimension. The supplied
size/orientation are fixed. Only complete placements are returned, with shape
`search_shape - template_shape + 1`. Search-window constancy is tested by exact
local min/max equality, not cancellation-prone integral variance. Undefined
windows have placeholder score zero **and false validity**. A constant search
is legal and produces no valid scores. Contrast-inverted patterns have negative
scores; neither positive scores nor detection counts are probabilities.

Computation uses owned float64 buffers with positive affine numerical
conditioning (midrange subtraction and division by maximum absolute deviation),
which is intrinsic to the same normalized-correlation definition. Original
buffers are never modified. When a window's raw contrast divided by global
amplitude is at most `sqrt(64 * eps_float64 * padded_sample_count)`, its score
is recomputed directly from its individually conditioned, mean-subtracted
window and template. Nonfinite or out-of-bound FFT scores (`abs(score) >
1 + 1e-10`) also take that direct route. This deterministic numerical fallback
visits every affected window; it is not sampling or a scientific score cutoff.
It protects small signals beside unrelated enormous intensities. Remaining
nonfinite/unstable outputs fail, while floating roundoff at the mathematical
`[-1, 1]` endpoints is clipped. The fallback can be slower for high-offset or
extreme-dynamic-range images; no large-volume throughput claim is made.

## Coordinate and validity contract

Score origin on each axis is `search_origin + (template_size - 1)/2 * scale`.
Even templates therefore retain half-pixel centers. `TemplateMatchMetadata`
records the original search grid, template extent, source frame, exact input
and paired output SHA-256 revisions, and backend implementation identity.
Score and Boolean validity outputs share the same center lattice and metadata.

Find Peaks requires the matching validity mask for a template response. It
verifies paired metadata, bytes, shape, and expected center-grid calibration;
recalibrated or modified responses cannot retain stale coordinate evidence.
A missing template tag with retained template kind/history fails clearly.
V1 does not support processing a response before peak extraction. Ordinary
intensity inputs may omit the mask, or use a Boolean same-grid mask with no
broadcasting or implicit conversion. Mask-false samples are absent from the
local neighborhood, not synthetic zero-valued competitors.

The persisted schema does not make a detached, untagged image's semantic origin
recoverable. Save/import paths must retain the carried template state when
scores are to be reused as template responses. A plain external array without
that state is an ordinary intensity image, not an asserted template response.

VIPP TIFF JSON, OME-TIFF annotations and OME-Zarr 0.4/0.5 metadata carry the
paired state. Save/read checks retain exact float64/Boolean storage, named axes,
center-grid calibration and paired revisions; conflicting native calibration
or malformed carried state fails. NGFF records the center translation in its
native coordinate transformations as well. ImageJ TIFF refuses tagged responses
because its storage path cannot preserve those dtypes. Both TIFF variants refuse
tagged ZYX grids with singleton Z because current readers squeeze that axis;
OME-Zarr preserves them. External readers are not claimed to recover VIPP's
additional provenance contract.

## Find Peaks ordering and population evidence

The general reference is
[scikit-image's local-maximum model](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.peak_local_max),
but VIPP defines its own deterministic policy rather than inheriting a backend's
version-dependent plateau selection or separation norm:

1. Candidates meet the inclusive authored minimum value and equal the maximum
   of their full `3 x 3` or `3 x 3 x 3` valid neighborhood.
2. Border exclusion removes the authored integer number of score-grid samples
   on every axis. Zero allows every full-template border placement.
3. Each full-connectivity, equal-valued candidate plateau retains its
   lexicographically first coordinate. Ordering is descending score, then
   lexicographic coordinate. A constant image above threshold is one plateau.
4. Greedy Euclidean suppression rejects distances **strictly below** the
   authored separation; equality is retained. Pixel mode uses index distance.
   Physical mode converts each axis scale to micrometers and therefore respects
   anisotropic sampling; it requires known length calibration. Zero separation
   is allowed, but does not undo local-maximum or plateau selection.
5. The result cap applies after suppression. All candidates are processed to
   retain exact candidate, accepted-before-cap, and returned counts plus an
   explicit truncation flag. An empty table is an ordinary successful result.

The table has `detection_id`, named source-image `<axis>_index` centers,
`<axis>_physical` values when length units are known, and `score`. Native
per-axis units accompany physical columns. Template detections additionally
have integer `template_<axis>_start` and exclusive `template_<axis>_stop` bounds.
`DetectionMetadata` supplies the source lattice, revision, coordinate-column
names, parameters, and exact cap evidence for linked read-only inspection.
It does not create label IDs, segmentation boundaries, or a general Points type.

Add Metadata Columns and Select Table Columns preserve this evidence when
they only annotate, reorder, or remove unrelated columns. Explicitly overwriting
or dropping a detection ID, score, source-center/physical coordinate, or template
bound clears the detection evidence and marks that loss in the carried table
kind. The result remains an ordinary editable/exportable table, but must no
longer assert the original geometry or pre-cap detection population. Other table
transforms do not yet preserve per-detection evidence; they require separate
contracts rather than assigning one source grid to merged/summarized results.

Batch CSV/TSV manifests retain the per-image detection schema, parameters and
pre-cap counts beside the exact exported cells. Collect Measurements retains
that evidence in each source item's inventory, including saved collection
snapshots. Its combined table deliberately has no single-image detection tag:
multiple sources do not share one coordinate frame. A detached CSV without its
manifest is an ordinary table, not complete detection provenance.

## Execution and memory boundary

Both nodes are CPU manual/cached operations with injected `input_states`.
Template Match's adapter returns exactly the two declared arrays. Graph
finalization calls `template_match_states_from_outputs`; the direct scientific
API wraps that same kernel/helper and returns arrays plus states. This keeps
host-output normalization and generated/batch execution on the standard path.

`template_match_grid_contract` supports metadata-only preflight. The numerical
kernel separately checks values before native work. Additional-allocation
admission uses the existing native host-memory snapshot and conservative gate,
including RAM reserve and Windows commit headroom. The estimate includes full
padded FFT grids, backend work arrays, owned float64 buffers, extrema/mask arrays,
and direct fallback buffers. Find Peaks admits worst-case candidate/label/sort,
spatial-bucket, and table-row storage before allocating its all-valid mask.
These are conservative estimates, not measured peak-memory qualification.

Cancellation is checked during input/revision chunks, between native kernels,
in direct fallback windows, during suppression and row construction, and before
returning complete results. A native FFT, filter, connected-component or sort
call cannot be interrupted internally. Cancelled operations return no partial
scientific result. Hashing is full-population, C-order, dtype/shape-bound and
works with noncontiguous/read-only arrays.

## Validation evidence and remaining gates

`test_detection.py` uses independent planted 2D/3D patterns and a direct
window-by-window Pearson oracle. Cases include even/odd templates, first/last
complete placements, noise, missing and contrast-inverted matches, constants,
tiny amplitudes, large integer offsets, extreme dynamic range, anisotropic
physical separation, native units, exact distance/threshold boundaries,
plateau ties, border exclusion, masks, truncation, immutable inputs, malformed
states and numeric values, memory refusal, and post-FFT cancellation.
`test_detection_node_adapters.py` covers runtime injection and the declared
two-output/table-first adapter boundary. Workflow/persistence/inspection and
example evidence are tested separately by their owning integration modules.

On Windows 11 / Python 3.12, the final focused selection of all detection tests
plus image IO, preview/source loading, collection persistence and Python export
passed **391 tests**, with **20 optional public-image tests skipped** because
their external corpus/strict acceptance flags were not enabled. There were no
failures. The seeded packaged workflows recovered all five 2D and four 3D
centers with no missed/extra detections and no input modification. The companion
manual passed its 16 content contracts, 50 cross-repository routes and strict
build. Eight normal/narrow, light/dark Qt inspection screenshots were reviewed.
These focused checks supplement the clean full regression and installed-package
verification recorded below; overlapping selections are not additive counts.

The first broad run (2026-09-28) built and privately installed its wheel and
passed the installed 2D/3D known-center checks. Full pytest stopped at its
eight-failure limit: 12,667 passed, 31 skipped and 2 expected failures before
the stop. All eight reported failures were outdated example inventories,
planning/bypass coverage inventories or historical showcase projections that
had not excluded the newly added independent detection lane. This is **not**
a clean full-suite pass. The follow-up updates must preserve old scientific
hashes, prove that the previous graph is unchanged, and explicitly exercise
new planning/bypass boundaries before one fresh full run.

Those bounded test-coverage repairs passed 158 planning/bypass/integration
tests and 83 example/golden tests. The new 2D/3D matrix cases check independently
planted centers in both CPU and Prefer GPU modes, not just agreement between
the modes. Explicit no-pixel planning and connected/persisted bypass-denial
checks preserve the detection boundaries. No production algorithm, resource
guard or scientific numerical acceptance criterion changed for this repair.
The subsequent clean full rerun passed, as recorded below.

Direct comparison with both `v0.16.0a2` and the checkout base confirmed that
removing the six named detection nodes and their seven internal connections
and two output tunnels leaves the previous 152 node records, 185 connection
records and 13 tunnel records exactly unchanged, including order. Historical
scientific hashes are retained rather than regenerated to accommodate the
addition. The current exhaustive-showcase digest changes only because its
independent detection lane is now included.

### Completed local full validation — 2026-09-28

Run `validation-20260928-153724-21180` completed at 14:28:07 UTC on Windows 11
x86_64, Python 3.12.9, NumPy 2.5.1, SciPy 1.18.0 and scikit-image 0.26.0.
Full pytest exited successfully: **12,711 passed, 31 skipped, 2 xfailed and
304 warnings**, in 2,994.21 seconds. JUnit independently records 12,744 cases,
zero failures/errors and 33 skipped entries (including the two xfails).
The xfails describe pre-existing CuPy integer Gaussian parity gaps, not
detection failures. Skips retain optional CUDA/public-corpus and host-dependent
coverage limitations; they do not qualify those unexecuted paths.

Ruff, manifest validation, wheel/source build and private wheel installation
all passed. The installed known-answer smoke imported exactly
`validation-20260928-153724-21180/installed-site/napari_vipp/__init__.py`, with
distribution metadata from that same private target. Its two packaged examples
recovered all five 2D and four 3D planted centers, with zero misses/extras,
zero index-coordinate error, calibrated physical coordinates and unchanged
inputs. The validity populations were 7,752 and 66,640 score-grid positions.
The full suite separately imported this development checkout's `src`, not an
older installed package.

The runner verified unchanged source after the full suite. A final independent
audit, before this documentation update, matched all **1,058** source-file
fingerprints, with no added, removed or modified paths, and rechecked both
artifact sizes and SHA-256 values:

- `napari_vipp-0.16.0a2-py3-none-any.whl`: 2,283,084 bytes;
  `F0DDC4EFCE978AF7ECD7D70EC60EE81E17732DD430DC490CCD07B622C41CDF04`.
- `napari_vipp-0.16.0a2.tar.gz`: 9,153,245 bytes;
  `5C086995F4E3CDA9DA5E9A006FA4A78DB09FC68B029D20D0D2D00D995789F958`.

These retain the checkout's unchanged version identifier and are private
development artifacts, **not** replacements for published 0.16.0a2 files.
Subsequent pre-commit updates are limited to engineering/status documentation;
the qualified implementation and tests were not changed. The earlier failed
run remains preserved as historical evidence, not part of the clean pass.

The local dated evidence is recorded under `VIPP-local-tests/detection-20260928`
outside the application checkout. The background runner records exact build
hashes, imported package location, known-center evidence, full pytest outcomes
and a before/after source fingerprint. No release or active-installation update
is implied by that private validation.

Native Linux/macOS execution, representative licensed microscopy accuracy,
minimum-version dependency qualification, large-volume measured memory and
performance, and release gates remain untested/separate. This local validation
is Windows-only and does not establish those broader claims. Synthetic
agreement is not biological detection validation. No GPU implementation,
rotation/scale bank, editable Points contract, tracking, or object-boundary
inference is included.
