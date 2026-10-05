# Bounded tracking: implementation contract

Status: unreleased development for issue #71 after 0.16.0a2. This is an
engineering contract and synthetic evidence, not acquired-microscopy validation
or a second public manual. `core/tracking_metadata.py` owns the immutable v1
schemas; `core/tracking.py` owns the CPU linker; `core/tracking_nodes.py` declares
the manual/cached, table-first Build Tracks node.

## Canonical names

Current declarations use **Detect Spots per Frame** (`detect_spots_per_frame`)
and **Build Tracks** (`build_tracks`). Their canonical functions are
`core/time_detection.py:detect_spots_per_frame` and `core/tracking.py:build_tracks`.
This rename does not change detection, assignment, calibration or scientific
limits. Dated qualification narratives below retain the original labels and
source snapshots they verified; they are not validation evidence for the rename.

## Accepted population and coordinate evidence

One table describes one explicit scalar `TYX` or `TZYX` acquisition, with a fixed
spatial grid and uniform temporal interval. Select channels upstream. No mixed
sources/channels, inferred time axis, irregular timestamps, axis permutations,
resampling or drift correction are introduced by linking. Observation metadata
retains spatial axes, shape, scale, origin and native units; exact source frame
and SHA-256 revision; frame count and time calibration; source-index coordinate
column names; and frame-local `detection_id` or `label_id` identity.

`FramePopulation` records exact eligible-before-cap and retained counts for
every frame, including empty frames. Linking rejects any truncated frame; no
allow-truncated switch is provided. Empty detections and missing observations
are not missing acquisition frames. Tables with missing, duplicate or invalid
identities, conflicting coordinate evidence, out-of-grid/NaN/Inf coordinates,
or populations disagreeing with retained rows fail without dropping data.
Frame-local IDs may exceed float64's exact integer range and remain integers.
Only source coordinates are converted to owned float64 computation buffers.
Integer coordinates outside the exact `[-2**53, 2**53]` interval and extended-
precision floating coordinates fail rather than silently losing precision.

Pixel mode measures Euclidean distance directly in source indices, independent
of physical scale. Physical mode requires known compatible length units on all
spatial axes and converts scales to micrometers, including anisotropic Z.
Origin remains in the evidence, but cancels in displacement on a fixed grid.
Known time calibration is converted to seconds; absent calibration reports
frames explicitly. Explicit unsupported units or invalid scale/origin fail.
No unit or coordinate is inferred from table shape or presentation settings.

## Assignment and ambiguity policy

The backend is SciPy's
[`linear_sum_assignment`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html),
a rectangular minimum-cost assignment implementation using modified
Jonker-Volgenant. VIPP specifies the candidate/gap/unmatched policy around it:

1. Sort observations by integer frame, then frame-local ID. Tracks receive
   increasing IDs when an unassigned observation first appears.
2. For each current frame, consider endpoints last seen in the adjacent frame
   first, then older endpoints in increasing elapsed-frame order. Already
   assigned current observations are unavailable to older tiers. This prevents
   a long-gap candidate from stealing an eligible adjacent association.
3. A candidate is feasible when Euclidean displacement is **at most** the
   authored maximum displacement per frame multiplied by elapsed frames. A
   maximum gap of `g` permits at most `g` missing observations/frames between
   linked endpoints. There is no velocity model, prediction or interpolation.
4. Within each tier, connected candidate components independently maximize the
   number of real one-to-one links, then minimize their total Euclidean
   distance. Real costs are normalized by that tier's distance gate. A private
   unmatched column per left endpoint costs `min(left_count, right_count) + 1`,
   exceeding every possible sum of normalized real costs. Infeasible edges have
   infinite cost. This is not a global across-time trajectory optimum.
5. Sorting gives stable ties in the same SciPy environment, not a claimed
   cross-version tie convention. No perturbation changes the distance objective.
   Any endpoint or current observation participating in multiple same-tier
   feasible alternatives receives `review_flag=True`, even if total-cost
   assignment is unique. This flag is ambiguity evidence, not confidence or a
   probability, and does not establish biological identity at crossings.

Track IDs are separate from local observation IDs. Divisions, mergers, lineage
trees, appearance models, edited identities, automatic drift compensation and
inference from disappeared objects are outside this slice.

## Outputs and reproducibility

The observations output retains all original columns and every original row,
sorted by frame and local ID. It appends `track_id`, `previous_frame`,
`gap_frames`, `displacement`, `speed` and `review_flag`. First observations have
no previous frame, displacement or speed (`None`), not fabricated zero motion.
Across a gap, speed is endpoint displacement divided by elapsed calibrated
time/frames. It does not measure motion along an unseen path.

The summary reports observation count, first/last frame, duration between first
and last observations, sum of observed endpoint displacements (`path_length`),
first-to-last net displacement, number of gap links, total skipped-frame count
and any review flag. Singleton duration and path length are zero. No rows are
inserted for empty frames. Summary path length is not an interpolated or
continuous-motion measurement. Distances and durations have explicit table
units; output metadata carries the original series, settings, fixed assignment
policy, VIPP linker version and installed SciPy version. Both JSON schemas
strictly reject unsupported versions and unknown/missing fields.

Exact table reload also validates protected output columns and units, positive
track identities, Boolean review flags, per-track previous-frame chronology,
gap bounds, and displacement/speed against recorded coordinates. Summaries are
checked for unique identities, finite nonnegative metrics, calibrated durations,
path/net bounds, total population, endpoint capacities and temporal-span
coverage of declared frame populations. These consistency checks do not rerun
assignment or prove biological identity, nor can a standalone summary recover
every original trajectory from aggregate values alone. Conflicting evidence is
rejected rather than silently cleared on import.

## Resource, cancellation and ownership boundary

Candidate discovery uses a spatial tree and exact Euclidean neighborhoods,
counted before allocating each neighbor list. A tier exceeding 1,000,000
candidate edges fails explicitly. A connected assignment component exceeding
2,000,000 matrix entries, including unmatched columns, also fails before dense
allocation. These are refusal limits, never scientific caps, approximations,
sampling or candidate pruning. Independent small components remain admissible
without a full all-tracks-by-all-observations matrix. Host-memory admission
checks conservatively include identity sets, row maps, candidate structures,
owned coordinates, result rows and native assignment workspace. These estimates
are not measured large-data performance qualification.

Input tables/arrays are never mutated; calculation and output structures are
owned separately. Cancellation is checked during validation, frames, candidate
building, component traversal, matrix filling, before/after native assignment,
and during output construction. An individual native spatial query or assignment
cannot be interrupted internally. Cancellation returns no partial result.

## Focused evidence and remaining gates

Integration uses the existing manual-node, shared-executor and table-output
contracts. Detect Spots per Frame has one source input in Local peaks mode and
an additional fixed-template input in Template match mode. Build Tracks has two
table outputs; generated Python's existing primary-result mapping can expose
the summary through an explicit all-columns table consumer. No generic export
contract is widened to accommodate these nodes.

Object measurements acquire observation evidence only for explicitly declared
whole `TYX`/`TZYX` series. Original label IDs are frame-local and source-index
centroids retain their historical definition. Unqualified layouts remain
ordinary measurement tables, not implicitly trackable observations. Table
annotations and column selection either retain the protected evidence or
explicitly clear it. Batch manifests and collection snapshots retain evidence
per item; aggregate multi-source tables never acquire a synthetic common frame.
Snapshot reads validate each included item's cells against its own evidence,
without rerunning scientific assignment.

Trajectory review is a separate read-only presentation surface. It navigates
one actual T/Z plane, links table selection to current-frame markers, and draws
only observed endpoints (dashed across gaps). Original-source residency,
readiness, geometry and frame identity are required. Pending work or changed
results disable stale identification/export. Sorting, trail length, window
size and decimal display never modify scientific values or workflow hashes.
Ordinary static measurement inspectors do not show the time-series section.

Preview buffers are optional: their conservative host-memory admission precedes
full-series coordinate conversion and frame indexing. Refusal or allocation
failure leaves the complete table, sorting and current-result export available,
with overlay navigation disabled and an explanation. T/Z navigation reuses the
frame index and admits its marker/trail buffers separately; it never samples
observations to fit memory. Stale-result export remains disabled in either mode.

CPU, Prefer GPU fallback and host-finalized device measurements propagate the
same immutable observation evidence before downstream linking. A resident-only
label result requires one admitted temporary host transfer for its exact source
revision; an existing host representation is reused. Retained/exit transfers are
not repeated, and cancellation/refusal publishes no partially qualified table.

`test_tracking.py` covers independently known 2D/3D motion, a permutation oracle
for minimum total distance, cardinality taking priority over a tempting short
link, adjacent-before-gap policy, explicit empty frames, gap distance/duration,
crossing ambiguity, input-row permutation stability, anisotropic mixed-unit
calibration, integer identity preservation, read-only source arrays, invalid
coordinates/frames/populations/calibration, strict immutable JSON, resource
refusal, and cancellation after native assignment. Node/integration, persistence
and presentation checks belong to their focused companion test modules.

On 2026-09-29, the final combined focused run passed **271 tests** across
previous-frame registration/UI, time detection, tracking, examples, standalone
smoke, shared-executor/batch integration, snapshot persistence, review UI and
measurement execution. Its JUnit record is
`VIPP-local-tests/tracking-20260929/focused-final.xml`. Ruff and diff whitespace
checks passed. This count is one combined selection, not a sum of overlapping
earlier checks. GPU lifecycle/fallback tests use a provider-free runtime harness;
they do not qualify actual CUDA numerical execution.

Offscreen Windows review captures include 2D/3D, light/dark and 1120x700/600x400
layouts. Representative final compact dark 3D and full-size light 2D captures
were visually inspected for readable controls and overlap. Manual content
contracts (16), app/manual routes (50) and the strict MkDocs build passed. Local
browser review covered the tracking how-to at desktop/narrow light widths and
the narrow reference in light/dark; it is not a complete visual browser matrix.

The first full Windows run completed on 2026-09-29 with **13,008 passed,
5 failed, 30 skipped and 2 expected failures** in 3046.04 seconds. Its failures
were missing documentation index/example entries, the comprehensive showcase
missing the two new nodes, and outdated sample-command/inspector inventories.
No algorithm acceptance failed, but this is **not a clean full-suite pass**.
Bounded fixture/docs/inventory repairs were followed by focused checks and one
fresh fingerprinted run; scientific assertions and production guards remained
unchanged. The failed run and logs remain preserved.

The repairs passed 26 documentation tests, 63 sample/inspector checks (including
real-widget static/time-qualified visibility), 115 showcase/golden/guidance
checks and two initial/ready layout audits. These are separate focused
selections, not an additional full-suite result. Ruff and diff checks passed;
the manual's 16 content contracts, 50 app/manual routes and strict build also
passed again. The comprehensive example now has a genuine TCYX-to-TYX tracking
lane with 24 observations/four tracks, exact gap/review accounting and a separate
summary consumer. Removing only its five nodes/four connections reproduces the
entire old eleven-lane document; existing scientific hashes remain tested
through that strict projection. The current complete showcase hash necessarily
changes for the authored new lane. No algorithm or production guard was changed
to address these failures.

That run's private installed wheel imported from its isolated target and passed
the standalone synthetic checks: 24/4 spot observations/tracks and 14/3 object
observations/tracks, zero coordinate error, exact frame counts and unchanged
sources. Previous-frame analytical translation had zero matrix error and
maximum valid-alignment error `1.164670302474663e-21` against tolerance `1e-12`.
Environment: Windows AMD64, Python 3.12.9, NumPy 2.5.1, SciPy 1.18.0 and
scikit-image 0.26.0. These private development artifacts retain version 0.16.0a2
but are not the published release artifacts and are not approved for upload.

## Completed local Windows qualification

The clean rerun `validation-20260929-182928-35132` completed on 2026-09-29 at
17:22:29 UTC: **13,015 passed, 31 skipped, 2 expected failures**, 304 warnings,
3114.62 seconds. JUnit records 13,048 cases, zero failures, zero errors and 33
skips (including the two expected failures). Environment/import checks, Ruff,
manifest validation, wheel/source build, private wheel installation, installed
known-answer smoke and the full suite all exited successfully. Final source
verification passed; an independent post-exit audit found all 1,083 recorded
source files and the file inventory unchanged. This completion record is a
documentation-only edit after that audit, not a change during validation.

The standalone acceptance imported `napari_vipp` and its distribution metadata
from this run's private `installed-site`, not the source checkout or active VIPP
installation. All 321 packaged module/resource files matched the wheel byte for
byte, and installed distribution provenance matched the wheel hash. The final
installed smoke independently repeated these known answers:

- 2D spots: 24 observations, four tracks, frame counts `4,4,4,0,4,4,4`, no
  missing/extra observations or coordinate error, four crossing-review
  observations across two tracks, and unchanged source. Spatial calibration
  `(0.5, 0.4)` micrometers, origin `(4, 11)`, time interval `0.5` seconds and
  time origin `2` seconds were retained; this case explicitly uses pixel gates.
- 3D labels: 14 observations, three tracks of lengths `6,5,3`, frame counts
  `2,2,1,3,3,3`, no missing/extra observations or coordinate error, and unchanged
  source. Physical gates retain anisotropic spacing `(1.5,0.5,0.4)` micrometers,
  origin `(-2,4,8)`, time interval `2.5` seconds and time origin `11` seconds.
  Known path lengths are `7.762087348130011`, `2` and `0` micrometers.
- Previous-frame analytical translation: five frames with anchor 2, two frames
  on each side, zero matrix error and maximum valid-alignment absolute error
  `1.164670302474663e-21` against tolerance `1e-12`. Valid coverage counts are
  `1800,1938,2080,1938,1800`; anisotropic scale/origin and time calibration are
  retained, with unchanged source.

Final artifact inventories and independently verified SHA-256 values:

- Wheel, 2,320,780 bytes:
  `03F2FACD63A6EF873BF868708EF0212C163431AF2514F460FD90D45F095747AE`.
- Source archive, 9,205,009 bytes:
  `0500B8B4FB4FB934023A3C0D34B0B1224E159D737953C88C34AA1A2DA250A9C0`.

The private development artifacts still identify as 0.16.0a2; they are not the
published release bytes and must not be uploaded. Exact logs, JUnit, installed
smoke JSON, environment, fingerprints and artifacts are retained under
`VIPP-local-tests/tracking-20260929/validation-20260929-182928-35132`.

The 31 skips include opt-in CUDA/public-image checks and platform/UI limitations.
One Tk console lifecycle test skipped because this test environment lacks a Tk
scrollbar resource; that lifecycle is not qualified by this run. The two expected
failures are existing CuPy 14.1.1 integer Gaussian parity cases, not tracking
acceptance failures. Provider-free device lifecycle tests are not actual CUDA
numerical qualification.

This completes **local native Windows** qualification only. Native Linux/macOS,
minimum dependencies, large-volume memory/performance, physical CUDA and
acquired-microscopy accuracy remain untested. The visual review remains partial;
no release or installer qualification is claimed. No shared/active environment,
open VIPP session, commit, push or publication was changed. The external
`VIPP-local-tests/tracking-20260929/CHECKPOINT.md` records completion and monitor
closure; these future gates do not prolong this local validation task. The
standalone acceptance script and [synthetic record](tracking-synthetic-evidence.md)
specify the known answers for installed-package tracking and previous-frame
registration.

## Inspector tuning windows — 2026-09-30

Tracking and the shared detection declarations now use fixed practical slider
windows with the original numeric-entry limits, defaults, steps and precision.
Displacement/separation sliders end at 100 and missing-frame sliders at 10;
the numeric fields retain larger valid values. Registration's broad reference,
optimization, intensity-range and fill controls follow the same declaration
contract. Detect Over Time inherits the Find Peaks windows. Slider bounds are
presentation metadata and do not enter authored values or scientific hashes.
Typing outside a window pins only its thumb; reopening the inspector retains
the exact entered value. Existing input-dependent channel bounds remain active.

The focused slider/control/precision/workflow-hash regression selection passed
**888 tests** (17 existing dependency warnings, 35.48 seconds); Ruff and diff
checks passed. Documentation checks passed 26 tests; the companion manual passed
16 content contracts, 50 app/manual routes and its strict build. This UI change
follows the clean full run above and is verified
separately, rather than describing its newer source as the fingerprinted full-run
snapshot. No scientific algorithm, accepted parameter range or original result
was changed.

## Bounded release preparation — 2026-10-04

The clean wheel/source installation job now invokes the standalone installed
detection and tracking acceptance scripts. Both native macOS installed-prefix
gates invoke them with the application's own interpreter, and the development
workflow watches changes to those scripts. Windows frozen setup remains a
packaging gate; the setup executable is not an installed scientific interpreter.
Twelve new workflow-contract cases cover this wiring. Focused workflow/packaging
verification passed **126 tests**, with one native macOS-tooling skip. Ruff and
whitespace checks passed. This wires future native CI coverage; it does not
claim those hosted jobs ran locally.

Current UI checks passed **44 result-action/trajectory-review tests** and
**43 practical-slider tests**, separately from the historical full-suite
fingerprint above. A separate synthetic validation graph adds image-intensity
histogram and observed-position plot branches to the known 2D spot example.
Its CPU preflight recovered 24 observations/four tracks, and eight tracks with
missing-frame reconnection disabled. Histogram accounting retained 48,384
pixels and the plot retained 24 values. Save/reopen retained identical tracking
tables, both CSV exports retained exact rows/columns, and the read-only source
was unchanged. The checked graph was opened in a visible development session
with read-only trajectory review for human acceptance. The deliberately
ambiguous crossing and frame-local detection identities remain unchanged.

### Actual CUDA measurement handoff

A bounded RTX 5090 canary passed **eight cases**: TYX/TZYX, basic morphology or
morphology-plus-intensity, and host-produced or native GPU Connected Components
labels. It exercised public `cupy-measure-objects-basic-v1`,
`cupy-measure-objects-intensity-basic-v1` and
`cupyx-connected-components-v1` implementations, with strict GPU measurement
selection and CPU `cpu-numpy` linking, not synthetic runtime providers.

Each case recovered six observations/two tracks with frame counts `2,2,0,2`,
exact CPU measurement/tracking parity and zero centroid/speed error. Read-only
sources were unchanged; all eight explicit finalized-host replacements retained
exact source revisions. Host finalizers received detached inputs after private
live/reserved GPU memory reached zero, with no fallback and clean runtime
closure. Environment: Windows, Python 3.12.9, RTX 5090 driver 610.74, CUDA
runtime 13.2/driver API 13.3, CuPy 14.1.1, NumPy 2.5.1, SciPy 1.18.0,
scikit-image 0.26.0 and Centrosome 1.3.4. Evidence and preserved harness repair
notes are under `VIPP-local-tests/tracking-gpu-canary-20261004`.

This closes the affected small CUDA measurement-to-tracking handoff gap only.
Detection and tracking remain CPU operations. It is not full GPU admission,
large-volume memory/performance, acquired-microscopy accuracy or other-platform
qualification; elapsed canary timings are not speed claims.

### Current private packaged candidate

A fresh wheel built from an isolated, byte-verified source copy passed installed
detection/tracking and analytical previous-frame registration acceptance.
All 1,085 original source fingerprints and inventory remained unchanged, and
all 321 installed module/resource files matched the wheel. Package imports and
distribution metadata came from the private target, not the source checkout or
open application environment. Detection recovered five 2D/four 3D centres with
zero missing/extra centres or index error; tracking recovered 24/four spot and
14/three object observations/tracks. Previous-frame translation retained zero
matrix error and maximum valid-alignment error `1.164670302474663e-21` against
tolerance `1e-12`, with unchanged sources.

The wheel is 2,321,553 bytes, SHA-256
`4B3B654A28C59DA2DDCC254B5E154630B2A5741D9042A9FA85F32789D1DE03C5`.
It still identifies as 0.16.0a2 and is **private and non-publishable**; it must
never replace published release bytes. Existing dependencies were reused
read-only: this is local native Windows evidence, not a clean resolver solve,
native Linux/macOS, installer or release qualification. Logs, source manifests,
provenance and installed known-answer JSON are under
`VIPP-local-tests/release-prep-20261004/package-check`.

Human acceptance, reviewed integration/version metadata, successful exact-final
main native CI and fresh release artifact/installer audits remain required.
No full suite was repeated, shared environment changed, commit pushed or release
published by this preparation. This record was added after the package worker's
unchanged-source verification.

## Inspector tuning stability follow-up — 2026-10-04

Detect Over Time exposed two shared presentation defects: the live settings
summary was intentionally elided after three lines, and each parameter edit
recreated identical connected-input rows. A second transient movement remained
after retaining those rows: the execution-status refresh unconditionally
remeasured and activated the full responsive inspector through the isolation
controls, even when their presentation was unchanged.

The settings summary now uses complete native word wrapping. Equal input
bindings retain their existing widgets; changed source names, ports, roles,
types and scientific metadata still refresh normally, with independent live
theme updates. Isolation refresh requests responsive layout work only when its
checkbox/panel visibility, status text or bypass visibility actually changes.
Width and node-selection updates retain their separate responsive paths.
No layout is frozen and no scientific parameter, accepted range or algorithm
has changed.

The final focused selection passed **251 tests**, zero failures/errors/skips,
in 132.96 seconds (67 existing dependency deprecation warnings). It includes
eight real scroll-surface scrubbing cases: widths 340/800 pixels, light/dark
napari styles, and 10/14-point text. All retain the active control's viewport
position and scroll offset during intermediate ancestor resize/move events,
not just after settling. They also verify committed slider values, unchanged
input-row ownership, complete wrapped text and sufficient rendered height.
Additional checks cover long summaries in a scroll area, narrow/wide reflow,
name drafts, input-context changes, theme updates, isolation/bypass, responsive
forms, practical slider windows and documentation contracts. Ruff, manifest
validation and whitespace checks passed. The companion manual passed its 16
content contracts, 50 application/manual routes and strict build.

This is focused native Windows/offscreen Qt evidence, not a repeated full suite
or Linux/macOS GUI qualification. Evidence is in
`VIPP-local-tests/inspector-stability-20261004/focused.xml` and its checkpoint.
These UI changes postdate the private candidate fingerprint above and are not
claimed to be included in that wheel. No open VIPP session was restarted, active
environment changed, commit pushed or release published by this follow-up.

## Direct trajectory-review card actions — 2026-10-04

Detect Spots per Frame and Build Tracks have a **Review trajectories…** card
button beside their existing table-opening action. Measure Objects and Measure
Objects Intensity expose the same action only when their primary result table
carries time-series observation metadata; ordinary static measurements do not.

The shortcut uses the existing read-only tracking review controller and the
clicked node's primary observations, independently of the selected node or
remembered inspector table output. It reuses the review window without
calculating, changing selection or modifying source/results. Existing guards
for missing, stale, pending and unavailable source data remain in force.
Changes in card height refresh port geometry and connection routing; unchanged
visibility does not request another geometry update.

Focused validation passed **279 unique tests**: 257 graph, tracking review,
inspector integration and documentation tests, plus 22 node-result-action
tests. Actual viewport clicks in light and dark themes retain selection and
card position. Checks also cover exact result ownership, shared-window reuse,
primary observations versus summary output, temporal measurement visibility,
and rejection of unavailable results without recalculation. Repository Ruff,
manifest validation and whitespace checks passed. The companion manual passed
16 content contracts, 50 cross-repository routes and a strict build. Offscreen
light/dark card renders were inspected for complete labels and separated
buttons; evidence is under
`VIPP-local-tests/node-trajectory-shortcuts-20261004`.

This is focused native Windows/offscreen Qt evidence, not a full-suite,
installed-wheel, other-platform or release qualification. This shortcut
postdates the private candidate fingerprint above and is not included in that
wheel. The open VIPP session was not restarted or changed, and no environment
installation, commit, push or publication was performed.

## Initial recalculation and canonical names — 2026-10-04

The unreleased nodes now use **Detect Spots per Frame** and **Build Tracks**
throughout the catalogue, inspector/review, development examples and manual.
Their operation IDs, domain functions, node adapters, compute implementation
identities and generated-code operations use `detect_spots_per_frame` and
`build_tracks`. Module names `time_detection` and `tracking`, observation/track
schemas, scientific parameters, linking rules and output ports remain descriptive
and unchanged. The user confirmed there are no legacy workflows for these
unreleased nodes: no old-name aliases or migration map are retained. Normal
custom node naming and unrelated released workflow compatibility are unchanged.

The fresh launch's explicit synchronous CPU calculation previously recorded
display decisions but not authentic chained cache provenance. The first selected
Build Tracks recalculation consequently rejected the cached manual detector and
waited upstream. The synchronous path now uses the shared exact source capture,
fail-closed cache admission and CPU provenance publication for each invocation,
including autodefault reruns and safely completed prefixes after an error. The
small in-place CPU path remains; neither admission nor manual authorization is
relaxed. Stale decisions from a subsequent rerun are not reported as current.

Immediate and post-debounce fresh seven-node example checks now retain the exact
READY detector table/provenance. Setting maximum missing frames from one to zero
and recalculating only Build Tracks gives **24 observations and eight tracks**,
without Calculate all. Source arrays remain unchanged. Changed bytes, source
metadata, upstream parameters, compute policy and unauthenticated READY/completed
flags still prevent cache admission in direct and detached paths.

Focused verification covers **993 distinct passing cases across recorded runs**:
807 domain/UI/workflow/documentation cases, 141 shared execution/cache cases,
29 selected existing widget cases, 11 canonical naming cases and five additional
handoff regressions. This is a deduplicated count, not a fresh full-suite run.
The initial domain selection had three outdated example-hash goldens; reversing
only the approved operation/generated-node identities exactly reproduced each
preceding hash before updating constants. All 85 golden cases then passed. The
initial widget selection had three invalid cache fixtures: two switched AUTO to
CPU directly after startup, and one fabricated a manual completion without
provenance. Correct initial CPU setup and a real one-iteration 8-by-9 authorized
deconvolution preserve the original cache-retention assertions. The final widget
and new handoff selection passed all 44 cases. No production guard or scientific
acceptance assertion was weakened; failed logs remain preserved.

Repository Ruff, manifest validation and whitespace checks passed. The companion
manual passed 16 content contracts, 50 cross-repository routes and its strict
build. Evidence, JUnit summaries, identity-only hash audit and the exact external
startup probe are in `VIPP-local-tests/tracking-recalc-rename-20261004`.

Coverage is source-imported native Windows dependencies with offscreen Qt, not
new installed-wheel/frozen-GUI qualification or manual acceptance of the open
session. Linux/macOS, minimum dependencies, large-volume memory/performance,
physical CUDA and acquired-microscopy accuracy remain untested by this follow-up.
These changes postdate the private candidate above and are not included in it.
No live VIPP session, active environment, commit, push or publication was changed.
