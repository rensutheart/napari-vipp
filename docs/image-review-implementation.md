# Review Images implementation and qualification

Status: implemented on `feat/review-images`, intentionally uncommitted and
unreleased after 0.16.0a3. The companion public instructions are in
`vipp-mkdocs` under `how-to/review-images.md` and `reference/image-review.md`.
This record defines implementation/test boundaries, not a duplicate user guide.

## Contract

`review_images` is a presentation-only sink with required array input `a`
(Image A), optional array input `b` (Image B) and no scientific outputs.
Opening the review consumes current cached upstream results without running a
scientific operation. There is no batch image output or new numerical image to
export. Numerical comparison remains the separate `compare_images` operation.

Supported carried kinds are scalar intensity, Boolean binary mask,
non-negative categorical integer labels and explicitly encoded RGB/RGBA.
Canonical YX/ZYX, optionally preceded by T, are required. Encoded colour uses
an explicit trailing `rgb`/`rgba` channel axis of length 3/4. Ordinary C stacks
need an upstream channel selection or explicit RGB composition; a short final
dimension is not sufficient evidence of RGB. Inferred/ambiguous axes fail.

Paired physical/time grids must match after compatible-unit interpretation:
axis names/types, shape, spacing and translation. Review performs no hidden
registration, resampling, broadcasting, axis reordering, clipping or scientific
normalization/casting. Descriptors retain non-writeable source views. Separate
napari layer objects prevent shared style mutation and label painting.

Display recipes use schema version 1 under
`metadata.vipp.inspector.image_reviews`, keyed by review-node ID. Arrangement,
pane composition, ndisplay, orientation preset, axes visibility,
viewpoint/contrast linking and per-input display
settings are presentation metadata, not operation parameters. Scientific
workflow hashes omit the presentation sink and its display recipe. Malformed
recipes fail validation before replacing a workflow. Workflow ownership and
source-current checks prevent cross-tab display leakage or stale results being
presented as current.

The revised window separates global controls from input styling: arrangement,
display dimensionality, Link viewpoint and Axes are in the top toolbar;
each pane has its own A/B/A+B selector; time and spatial-slice positions use a
shared bottom strip. Those dimension positions remain shared even with camera
linking disabled. Link viewpoint controls only camera rotation, pan and zoom.
XY/XZ/YZ select orthogonal 2D display planes or axis-aligned 3D viewpoints, and
Oblique selects an angled 3D view when supported. The spatial-slice control
follows the undisplayed Z/Y/X axis. An actual 3D-to-2D toolbar/native mode
change starts in XY with Z navigation, rather than carrying a 3D camera
preset into an unexpected slice plane. Deliberate XZ/YZ slice choices and
saved already-2D recipes remain supported. The one-based position editor
contains only the current number; a separate `/ N` label shows the total,
with the calibrated world-coordinate caption beside it.
The orientation preset belongs to the
display recipe; current
camera position, timepoint and slice position remain runtime-only navigation.
These camera presets do not transform or resample scientific data.
Fit views sits beside the bottom Oblique/XY/XZ/YZ buttons. The recipe's
`show_axes` Boolean defaults to true and controls both panes' native orientation
axes in 3D only. Both native axes and scale bars disable the public background
`box` property when available, preserving the image behind the arrows and
labels. Older scene axes without that property are unchanged; no private
shader or main-viewer overlay is modified. Palette-aware icons retain text
labels and do not introduce
new scientific settings.

## Renderer boundary

The window owns two ViewerModels/QtViewers, distinct from the main viewer. Scalar
display uses black/white limits, colormaps and optional alpha cutoff; mask
background is transparent and labels retain categorical IDs. Changing display
must not mutate scientific buffers or dispatch processing.

Scalar 3D display recipes additionally retain `rendering` (`mip`, the default;
`attenuated_mip`; or `iso`), finite non-negative `attenuation` (default 0.05)
and finite source-value `iso_threshold` (default None, meaning the overflow-safe
black/white midpoint). Schema-1 recipes missing these fields receive defaults;
malformed values are rejected before replacement. These fields are presentation
metadata and remain excluded from scientific hashes and cache identity. Public
napari layer properties are applied to both copies of each scalar input in 3D
only; 2D slicing, RGB component-MIP, Boolean mask iso and categorical Labels
remain unchanged.

The inspector names these choices Maximum intensity (MIP), Depth-weighted
intensity and Surface (isosurface), with conditional depth-weight/surface-level
entries. Depth weighting emphasizes nearer signal, not opaque physical
occlusion. A surface hides interior detail and is neither segmentation nor
exact measured geometry. Shading can alter apparent brightness/colour; the
colour bar is explicitly a Base colour scale (before shading) outside MIP.
Lock colour scale protects the black/white limits, not a quantitative shaded
rendering. Two scalar inputs retain additive overlay blending with no physical
inter-layer occlusion. No private shader hooks or analysis operations are added.

The binary-mask renderer no longer uses the image layer's default MIP, which
collapses foreground depth into a flat silhouette. Canonical Boolean 0/1
storage borrows a zero-copy uint8 view for native 3D iso rendering at threshold
0.5 with linear display interpolation. A black-to-foreground ramp preserves
native iso lighting; the shader discards background. The full uint8 texture
normalization range is renderer plumbing, while display contrast remains 0–1.
Noncanonical True bytes, such as 255, require an explicit read-only presentation
buffer with canonical logical foreground, costing one byte per voxel and
prepared once per input/window for reuse across panes and modes. Existing
host-memory preflight guards that allocation and reports an actionable error
when headroom is insufficient. The original Boolean bytes remain unchanged;
this is not a scientific conversion. Native categorical Labels are not used as
a mask fallback because adversarial rendering exposed a shading seam. There
are no private colormap or shader hooks. 2D keeps the original Boolean buffer,
nearest interpolation and transparent zero. Ordinary label inputs keep
categorical IDs. Interpolated surfaces and
shading are display approximations, not smoothing, exact geometry or scientific
segmentation results; voxel facets can remain.

The numeric colour bar retains the black/white mapping when values are hidden.
The optional Lock colour scale setting keeps scalar limits fixed during review.
The cutoff uses graphics presentation precision, not an exact numerical mask or
scientific threshold. Red/green colour overlap is not a colocalization metric.

RGB/RGBA slices use native encoded-colour display. RGB 3D uses an explicit
separate R/G/B component MIP adapter with additive blending. Independent
component maxima may occur at different depths; this is neither voxelwise RGB
ray-casting nor napari-racc custom shader parity. RGBA 3D is unavailable rather
than silently losing per-voxel alpha. RGB example recipes open in 2D.

## Deterministic executable examples

Eight independently constructed samples are registered in the normal sample
catalogue and a separate napari image-review sample entry. Four mirrored
workflow JSON files (repository and packaged resource) have saved recipes:

| Launcher ID | Source geometry | Presentation acceptance |
| --- | --- | --- |
| `review-channels-2d` | Two aligned float32 YX images, 80×104, spacing 0.5/0.4 µm, origin 4/−3 µm | Independent contrast, red/green overlay and linked pan/zoom |
| `review-mask-3d` | Two analytical physical spheres, radii 5.4/5.0 µm, on 48×64×80 ZYX, spacing 0.6/0.5/0.4 µm, nonzero origin | Scalar plus Boolean mask, linked 3D and 2D, transparent background |
| `review-labels-time-series` | Four 12×40×48 volumes, 2 s/frame, nonzero time origin, explicit labels 7 and 42 | Shared time/slices, optional linked viewpoints and categorical IDs; no tracking computation |
| `review-rgb-index-3d` | Explicit uint8 ZYXC RGB plus aligned float32 0–1 index | Preserved RGB slices, fixed scalar colour scale and independent hide-below cutoff |

The index is directly constructed as an X-coordinate gradient inside the
spheres, not a RACC, RACC-like or colocalization computation. This distinction
is repeated in sample metadata, on-canvas notes and chooser guidance.

The initial fixture used 16 Z planes and two near-physical-spherical ellipsoids
only five to seven samples across Z, so stepped voxel boundaries were expected.
After the independent renderer investigation identified the flat mask MIP
silhouette, the authored demonstration was separately replaced with analytical
physical spheres sampled across at least 16.7 voxels per diameter on every axis.
The mask is the exact integer-coordinate radius condition on 0.1 µm ticks, not
a smoothed mask. The paired intensity is an analytical radial Gaussian; carried
truth records radii, world-coordinate centres and formula. Tests reconstruct
the full Boolean mask and float32 intensity independently and include an exact
boundary voxel. This demonstration change does not alter production arrays or
serve as evidence that the renderer's original projection bug was fixed.

`scripts/launch_vipp_image_review.py` selects a review fixture in a new session;
it does not install into, close or mutate an existing VIPP session. The ordinary
launcher also accepts each ID with node `review` selected. User acceptance
should exercise both panes, mask/label styles, scalar cutoff, RGB slices,
workflow save/reopen, stale input warnings and two different workflow tabs.

## Evidence and remaining gates

Focused example/documentation checks passed 50 tests across sample data, six
new deterministic sample/recipe tests, packaged-resource parity, chooser guidance
and repository documentation contracts. A separate focused layout run passed
the same six fixture tests and ten initial/ready Qt card/notes/wire cases for
the four new workflows and expanded exhaustive showcase. Checks verify explicit
metadata, byte-identical packaged JSON, aligned grids, repeatable input buffers
and the honest synthetic-index label. The exhaustive generator retains twelve
source-sample lanes and adds Review Images beside quantitative registration
comparison (165 nodes, 199 connections, 143 distinct operations).
Six launcher-selection tests also passed, covering each registered ID, the
default mask review and rejection of unknown/external paths without launching
an application. The separate helper run rechecked the six fixture tests.

Changed-file lint passed. Companion manual checks passed 20 content contracts
and 50 cross-repository routes; the strict documentation build passed. A native
desktop/narrow-width visual review of those manual pages is not recorded here.
Final combined integration/window counts are recorded by the coordinating
implementation check after its source is stable; this is not an already
completed full suite.

The coordinating checks passed 27 new core contracts, 25 isolated window/model
contracts and six widget integration cases (including two tabs with identical
node IDs and different image sizes). Existing inspector/graph checks passed
175 tests; inspector integration passed 23 and history/name integration passed
20. The full example compute matrix passed 38 with one explicit physical-CUDA
skip. Optimizer/compute contracts, policy and review-core checks passed 184;
these focused totals overlap and must not be added as unique-suite counts.

An earlier bounded native Windows Qt/OpenGL probe rendered all four paired examples,
checked nonblank framebuffers, verified red/green RGB slice and component-MIP
pixels, linked camera/time navigation, read-only labels, unchanged source
SHA-256 digests and an untouched independent main viewer. Screenshots were
visually inspected, including the right-hand inspector and readable tiny
contrast values. This is graphics smoke coverage, not exact quantitative
renderer equivalence. Extreme floating-point texture/contrast accuracy remains
unqualified. Evidence is in `VIPP-local-tests/review-images-20261008` outside
the repository. The source-stable qualification runner builds wheel/source,
checks a private installed wheel and then runs the combined focused and full
regression suites; its external state is authoritative while those gates run.

The first qualification worker was interrupted for the user-approved toolbar,
shared-navigation and mask-rendering revisions. Its partial results are not a
clean source-stable final pass. The focused results above describe their
recorded source revisions; final combined and installed-package qualification
must be recorded against the revised source after it is stable.

A subsequent source-unchanged qualification run stopped at pytest's failure
limit with 6,949 passed, nine skipped, two expected failures and three failures.
This was a natural test failure, not an interrupted run, and early-stop counts
are not complete full-suite coverage. The three recorded integration mismatches
were repaired within scope. A fresh clean full run remains pending against
the final source, including the later Axes/toolbar revision.

The Axes/toolbar-icons revision passed 51 window/model tests, including saved
global axes intent through 3D/2D changes, fit-button placement, palette refresh,
labelled accessibility and narrow-layout containment. A separate glyph/core run
passed 47 tests (36 core, five glyph, six existing decimal-icon cases). These
overlap other focused totals and are not unique cumulative coverage. The
integration repair run passed 1,099 tests across seven modules with clean lint.
Its strengthened presentation-boundary test also exposed and fixed broad
diagnostic callers allowing display sinks into compute workloads. Workload
assembly now intersects caller IDs with scientific-node IDs; exact scientific
descriptors, facts, lineage and source bytes remain identical with or without
the connected review sink.

The new native Windows source-tree probe passed all four examples and the
independent RGB/sphere/cuboid display oracles. It toggled axes off/on in both
panes, with identical cameras and mask foreground extent (178×176 display
pixels), verified labelled icons and bottom Fit views, and checked unchanged
source digests/main-viewer state. Both axes-on and axes-off screenshots were
visually inspected. Evidence is under external `native-axes-source-confirmed`.
This native smoke result does not replace the pending fresh source-stable
private installed-wheel/full-suite qualification. No clean full pass is
claimed for the preceding failed or interrupted revisions.

The user subsequently exposed an opaque axes background in XZ view. The
installed napari 0.9 canvas axes inherit `box=True` by default. A native bright
volume oracle reproduced 69,943/69,942 obscured pixels in the two panes and
zero after setting the public property false; 1,385/1,381 coloured axes pixels
remained, with identical cameras/source digest. Both screenshots were visually
inspected. A strengthened model regression failed against the old production
code before the bounded fix. A legacy-overlay test ensures the property is not
invented on versions lacking it.

Run `qualification-20261008-213132` was explicitly stopped for this new
user-requested display repair before changing application source. Its 1,160
passing focused cases, one physical-CUDA skip and installed/native checks are
retained, but its unfinished full suite has no clean final result. A fresh
source-stable qualification includes the bright-volume axes oracle and the
new default-transparency assertion. Actual external state is authoritative;
no completed clean full pass is claimed until its final summary and source
verification are recorded.

After the transparency repair, 99 focused window/widget/core/icon tests passed,
including axes visibility, save/load/reset, image/cache isolation and the
feature-detected legacy-overlay path. The native source-tree bright-volume
probe verified both production axes boxes are false on construction, zero
obscured pixels and retained coloured axes, with identical camera/source digest.
App documentation checks passed 26 cases; changed-file lint and whitespace
checks passed. The companion manual again passed 20 contracts, 50 routes and
a strict build. These are bounded repair checks, not a completed full-suite
or non-Windows graphics qualification.

After the revised geometry, shared-navigation guidance and meaningful input
node titles were added, a focused run passed 43 fixture, chooser-guidance and
documentation tests. This includes seven review-fixture tests, with exact
physical-sphere construction and packaged-resource/name checks. A separate
eight-case Qt layout run passed the initial and ready states of all four paired
examples. The sample/fixture/guidance run passed 20 tests; these totals overlap,
not unique cumulative coverage. Companion checks again passed 20 manual content
contracts and 50 routes, with a strict build. These checks do not replace the
new stable-source installed/full qualification or the renderer's native probe.

The final toolbar/sidebar revision's coordinating focused checks passed 32
review-core contracts, 49 component/model cases and seven mask-rendering helper cases.
These are focused, overlapping qualification domains, not cumulative full-suite
coverage. The window sets calibrated units before layer insertion to prevent
the native 2D RGB/scalar slices from appearing blank. Revised native probe and
stable-source installed/full results are recorded separately when complete;
these focused results alone are not a native-graphics qualification.

Before the Axes/toolbar-icons revision, the revised source-tree native Windows
Qt/OpenGL probe passed all four paired
examples, startup nonempty calibrated slices, sidebar viewport containment,
shared camera/time navigation, read-only categorical labels, source SHA-256
invariance and independent main-viewer isolation. An independent RGB-primary
volume verified red/green/blue foreground pixels in both panes through
2D→3D→2D switching; the original RGB slice also matched its source bytes.
Independent spherical occupancy projected with width/height ratios
1.004/1.004/0.997/0.972 in XY/XZ/YZ/Oblique views. An asymmetric cuboid
disambiguated those plane directions with measured ratios 1.494/3.000/2.000
against expected 1.5/3/2. The recorded framebuffer checks are bounded display
smoke acceptance, not scientific morphology measurements or exact rendering
equivalence. Screenshots were visually inspected. Evidence is under external
`native-revised-source-confirmed`, distinct from earlier partial/failed probes.
Revised stable-source private-wheel and full-suite qualification remains pending;
no completed clean full run is claimed here.

### Slice-control repair and qualification status — 2026-10-09

The transparency qualification `qualification-20261008-214541` naturally
stopped at the three-failure limit: 2,880 passed, three failed, two skipped,
183.53 seconds. JUnit contains 2,885 cases and zero errors; final source
fingerprints match. All failures were detection-example allocations rejected
with 4.5–4.9 GiB commit headroom below the unchanged 5.1 GiB safety reserve.
Read-only comparison with released `v0.16.0a3` confirms identical detection
code, tests and memory guards; replaying the logged headroom produces identical
denials. No baseline full suite was run, resource guards were not weakened,
and this failed/early-stop result is not clean full-suite qualification.

After that worker exited, the user-requested XY/Z default and separate total
labels were implemented. Focused window/widget/core/main-dimension checks pass
110 tests, zero failures/errors, 14.11 seconds (`slice-controls-final-focused.xml`).
Regression-first evidence is retained. Native napari centers a newly hidden
spatial axis before its mode notification, so the native transition tests
require shared resulting positions and preserved time; exact prior spatial
positions are checked on the toolbar path. Saved deliberate 2D planes remain
honoured. Keyboard entry, endpoints and narrow-layout totals are tested.

A bounded native Windows source-tree probe passed 16 toolbar/native transitions
from all four 3D presets across the mask and time-label examples. Both panes
use XY/Z, positions stay shared, source digests are unchanged and visible mask/
scalar slices equal their source arrays. Totals and bottom controls fit at
900/1280 logical pixels; screenshots were visually inspected. Evidence:
external `native-slice-controls-source/slice-controls-result.json`. Its own
probe windows exited; no user session was changed or closed.

App documentation checks passed 26 tests, repository lint/whitespace passed,
and the companion manual passed 20 contracts/50 routes and strict build.
These overlapping focused results are not a completed full-suite or installed
qualification of the latest slice revision. No competing full worker was
started under memory pressure. Free additional headroom with user approval
before one fresh stable-source qualification; preserve research sessions and
all prior artifacts. No shared-environment install, commit or release occurred.

### Scalar rendering test build — 2026-10-09

The user requested reopening the test with scalar 3D rendering options and the
external totals. Core recipe checks passed 74 cases; UI model checks passed
76, including save/reset, public renderer properties, broad valid numeric
entry, separate A/B styles and 2D/native-mode transitions. Those counts overlap
the combined check rather than forming cumulative coverage. The companion
manual passed 20 content contracts, 50 routes and a strict build.

Bounded private installed test build `rendering-test-20261008-221121` passed
lint, manifest validation, 206 combined focused cases (zero failures/errors/
skips; JUnit 18.122 seconds), wheel/source build, isolated private installation
and exact private package/UI import. Source-before/source-after fingerprints
match. It deliberately did not rerun the full suite; the preceding environmental
failure remains accurately retained, not promoted to a pass. The worker exited
before this evidence-only documentation append.

Native Windows Qt/OpenGL smoke passed all three scalar render modes on the
same camera, with both copies of each scalar layer configured identically,
unchanged mask iso rendering, unchanged source SHA-256 and retained selection
through 3D→2D→3D. Visible 2D slices equal their source arrays; the default plane
is XY with Z navigation and a separate `/ 48` label. Hidden native layer
slices are intentionally deferred and are not used as a source-slice oracle.
Screenshots were visually inspected. A separate saved-screenshot central-image
population excludes corner axes/scale bars and all controls, confirming actual
image foreground in each mode. Depth/iso changed 184,721/400,296 framebuffer
pixels relative to MIP, but these are only display-smoke differences, not
numerical renderer or physical-occlusion equivalence.

External evidence: the run's `state.json`, `focused.xml`, `installed-import.log`,
`native-rendering/rendering-result.json`, `central-foreground-result.json` and
four native screenshots. Private development wheel: 2,372,573 bytes, SHA-256
`d284bce8b80ac2bc392a56c0128e366980f51b806952a4356bf04a1c82f54665`.
Private source: 9,275,310 bytes, SHA-256
`eea80df5a5b01a41c3a6d960190a08f7d17b9775eb5e006a01da8176b82d5373`.
They inherit base version 0.16.0a3 but are unreleased development artifacts,
never public release replacements. User acceptance and a clean full run remain
pending; no shared installation, commit, push, publication or research-session
mutation occurred.

The first updated user test imported that exact private wheel, but its inherited
napari stylesheet exposed clipped first/last lines in the new rendering warning.
The plain-style native probe did not cover that themed layout. The intermediate
v5 receipt/screenshot are retained. Only that newly created disposable test was
closed; policy-only changes were insufficient under the themed form. The bounded
local explanation label explicitly recomputes wrapped height from the owning
form's actual width, clearing prior constraints so it can shrink again. It
retains identical-text constraints through opacity/depth edits to avoid layout
jitter. All 80 focused UI cases passed, including four dark/light × 10/12pt
regressions covering mode changes, narrow containment, shrink on widening and
unchanged-text opacity/depth edits without geometry jitter. Own-file lint passed.
Updated bounded test build `rendering-test-20261008-222201` passed lint, manifest,
106 UI/documentation cases (zero failures/errors/skips; JUnit 11.001 seconds),
build/private installation/import and native rendering/slice smoke with napari's
dark 12pt stylesheet. All three explanations fit their required wrapped height;
cameras and source SHA-256 remain unchanged, visible XY/Z slices are exact and
the total remains outside the editor. Actual foreground in saved central-image
ROIs was confirmed separately from corner overlays. Screenshots were visually
inspected. Source-before/source-after fingerprints match and the worker exited
before this evidence-only append. Earlier records remain intact.

The final private testing wheel is 2,373,201 bytes, SHA-256
`522511dc604ed7e0a044bdaddc20d0be358ba8e6cf49ffbe5c9c1c2a0920346f`;
source is 9,277,068 bytes, SHA-256
`b9cfa53b6b71c6019e5b9247d941062de43d8ec15449845abf4d797d23de28be`.
These are still unreleased development bytes, not replacements for public a3.
This bounded layout qualification does not replace the pending clean full suite
or qualify non-Windows rendering, scientific intensity accuracy or acquired data.

The corrected independent testing session v6 was reopened from that final
private wheel with all four workflow tabs, the physical-sphere review active
and depth weighting 0.1 selected. External `user-launch-v6.json` records exact
package path, runtime 65476/wrapper 53348, visible state, all three rendering
choices and external totals; a separate exact-process check confirmed it is
responding. Actual inherited-style warning height
and required height are both 80; `user-review-v6.png` was visually inspected and
all explanation lines are readable. The session is left open for user acceptance;
pre-existing research sessions and shared environments remain unchanged.

### Pane cards and scale-bar visibility — 2026-10-09

The user requested the mockup's joined pane-card appearance rather than floating
selectors/captions, and a global Scale bar checkbox beside Axes. The card is a
palette-derived presentation frame around the existing native canvas, header
selector and footer. It does not change image buffers, physical grids or camera
semantics. Scale-bar visibility is a strict saved Boolean, defaulting to true
for earlier schema-1 recipes; hiding it is independent of axes and leaves the
existing compatible-unit/calibration eligibility rule and transparent overlay
background intact. Both 2D and 3D panes use the same saved display choice.

The user closed the preceding v6 disposable testing instance. A refreshed
private test will be opened after changed-domain checks, without closing or
changing unrelated research sessions or installing into shared environments.
Bounded private installed build `rendering-test-20261009-100745` passed lint,
manifest validation, 237 focused cases (zero failures/errors/skips, 67 upstream
warnings, JUnit 17.443 seconds), build/private installation/exact import, and
changed-domain native display acceptance. Its wrapper57032/worker31128 exited
with matching source-before/source-after fingerprints before this append.
Prior full-suite failure remains retained and is not promoted to a pass.

Native dark/light 12pt checks at 900/1280 logical pixels show one joined frame
around the header, canvas and footer, with the new toolbar control contained.
Framebuffer differences in the lower-right scale-bar population establish
that hiding/showing removes/restores bars in both panes in 2D and 3D. Toggling
preserves cameras, axes visibility, layer identities and source SHA-256;
reopened saved recipes retain hidden bars. Visible XY/Z slices are exact source
slices, with totals outside the editor. Time-series footer checks use actual
four-frame counts and shared T/Z positions. Native screenshots were visually
inspected. Standalone styled probes log missing napari theme-resource icons;
they do not establish native checkbox-tick resource rendering. The normal
napari-parent user test is checked separately before handoff.

Evidence: the run's `state.json`, `focused.xml`, `installed-import.log`,
`native-rendering/card-scale-result.json` and 18 native screenshots. Private
development wheel: 2,374,013 bytes, SHA-256
`024cbf52205503a1b524f77ab6ea440c520b3911903397644031612c91721c51`.
Private source: 9,278,904 bytes, SHA-256
`c1d1e13e66aae401428198719d813e8283de1846f83b1445dbaf3ee5531014cc`.
These inherit the base a3 version but are never public release replacements.
Manual checks passed 20 contracts, 50 routes and strict build. No new full
suite, shared installation, commit, push or publication was performed.

Refreshed isolated v7 user test is visible/responding, runtime64032 with
wrapper27076, importing that exact private target. `user-launch-v7.json`
records four paired workflow tabs, active physical-sphere review, joined
cards, visible checked Scale bar, both calibrated bars, all scalar rendering
choices and external totals. `user-review-v7.png` was visually inspected:
normal napari theme resources render actual checkbox ticks/dropdown arrows,
the card frame joins selector/canvas/footer, and the rendering explanation
is unclipped (height/required height80). Executable UI SHA-256 matches the
installed private copy. The test remains open for user acceptance; other
sessions and shared environments remain untouched.

Native coverage is local Windows development only. Native Linux/macOS, minimum
dependency versions, large-volume memory/performance, acquired-microscopy
accuracy, renderer equivalence and release/installer artifacts remain untested.
Two graphics panes can allocate two texture populations despite CPU array
sharing. Automated Qt model tests do not prove a native GPU framebuffer's
numeric/rendering equivalence. Neither fixtures nor review screenshots prove
biological segmentation, correspondence or colocalization accuracy.

### Compact outputless graph card — 2026-10-09

The user found a blank thumbnail and `No output` on the Review Images card.
Initial graph construction already hid the image placeholder, but a later
generic thumbnail refresh used the node's fallback output type and enabled it
again. Outputless cards now have a distinct presentation capability: neither
thumbnail nor output metadata can be restored by later preview/metadata writes.
The capability follows resolved output-port changes, separately from the user's
thumbnail preference. Ordinary image previews and table/mesh/transform metadata
retain their behavior, including dynamic output restoration. The widget skips
thumbnail rendering/statistics for zero-port nodes and hides their inapplicable
thumbnail checkbox. Review's two inputs and Open review action remain intact.
Scientific execution, parameters, arrays, cache identities and hashes are not
changed by this presentation repair.

Evidence is external in `review-images-20261008/outputless-card-focused.xml`:
192 focused Windows offscreen Qt checks passed, zero failures/errors/skips,
67 upstream warnings, JUnit 10.939 seconds. Modules cover graph cards, review
widget integration, non-image bypass previews, graph display settings and
documentation. New cases cover dark/light compact cards, forced preview writes,
Off/Slice/MIP refresh after graph rebuild, no pixel rendering/statistics on the
sink, metadata preservation, dynamic ports, action mouse interaction, and
unchanged upstream output identities/scientific workflow hash. Repository lint
and whitespace checks pass. Two offscreen dark/light component images were
visually inspected: a 279×122 card has both input labels and Open review, without
the empty preview/output label. This is component evidence, not a native
renderer, full-suite or cross-platform qualification. A read-only second review
found no new blocker.

The red regression run is retained as `outputless-card-red.xml`: five expected
regression failures plus one teardown error from an overly broad test trap
catching unrelated queued source rendering. The trap was restricted to the
direct sink refresh before the clean focused run; no production assertions or
resource guards were weakened. Pre-existing whole-file formatting differences
were reported by the format check and were not mechanically rewritten. The
manual's two review pages were updated; 20 contracts, 50 routes and strict build
passed. No full suite, new wheel/install, commit/push or live-session restart
was performed. The already-open private demo therefore retains its previous
installed card until a separately requested refreshed launch.

### Shared bottom status and viewpoint row — 2026-10-09

The read-only/native-resolution status now occupies the left side of one
full-width bottom toolbar, with the existing Oblique/XY/XZ/YZ/Fit controls on
the right. It no longer reserves its own permanent row. Shared T/spatial sliders
remain immediately under the pane cards, wrap only when their own groups need
more room, and hide altogether when a plain 3D volume has no hidden dimensions.
Actionable stale-input, RGB and RGBA messages retain their full text and wrap
within the status area, without overlapping or replacing the view buttons.
Stale inputs disable the moved buttons while leaving their explanation readable.
A parent-owned timer rechecks navigator layout after font/style inheritance.
This changes only layout and control availability, not settings, source arrays,
calibration, scientific calculations or rendering algorithms.

`review-footer-focused.xml` records 136 focused Windows offscreen Qt tests,
zero failures/errors/skips, 67 upstream warnings, JUnit20.232s. The new layout
regression failed in all four dark/light×900/1280 cases before the repair
(`review-footer-red.xml`). Final UI/icon/widget/documentation checks cover the
one-row normal footer, both themes at 12pt, long stale-message wrapping and
shrink on widening, disabled stale actions/no settings callback, unchanged
source/camera, 2D→3D→2D slider visibility and retained time navigation. Lint and
whitespace checks pass. Offscreen component screenshots show a 34px normal
bottom bar at both widths; the complete stale message needs 74px at900 and51px
at1280. Normal dark900/light1280 and stale dark900 screenshots were visually
inspected. This is layout evidence, not native GL or cross-platform validation.
Manual20 contracts/50 routes/strict build pass. Existing sessions, installed
artifacts and older qualification records remain unchanged; no full suite,
new wheel/install, restart, commit/push or publication was performed.

A read-only second review found no blocking issue and suggested explicitly
retaining QLabel's height-for-width policy instead of relying on theme
inheritance. That final sizing safeguard was applied; nine affected layout
checks pass (`review-footer-final-wrap.xml`,17 upstream warnings,pytest4.26s),
including an added unstyled long-warning case. These overlap the preceding
136 checks and are not additional unique full-suite coverage. Final lint and
whitespace checks pass; no live session or installed build was changed.

### Accepted live input replacement — 2026-10-09

The user reported instability when changing smoothing with Review Images open.
The original controller treated every new output identity as an unavailable
snapshot and hid the panes instead of refreshing them. Its direct READY check
also could read a transient new A from a running worker beside an old accepted
B, and did not reject READY buffers whose completed-cache identity was cleared.

The controller now waits through active/queued source and pipeline work, rejects
pending/manual/inflight dirtiness in either input's ancestors (including the
full-run sentinel), and requires completed READY resident endpoints. It reads
accepted pipeline caches explicitly, never per-node background preview overrides.
Only a complete valid A+B pair can advance the window's input token. A completed
accepted pair can remain valid when an unrelated CPU branch fails; this does not
claim whole-workflow success. Failure, cancellation, supersession and rejected
source revisions affecting the review keep its previous pair noncurrent.

Same-count/kind/calibrated-grid replacement preflights all descriptors, exact
reset extrema and guarded mask display buffers before mutating either pane.
Existing independent layer wrappers receive borrowed read-only data together,
with notifications and painting suppressed during assignment. Notifications
publish only a complete pair. Camera poses, time/slice positions, recipe, styles,
layout and settings callbacks are preserved; no fit, layer rebuild or scientific
calculation is triggered by review refresh. Names and Reset display extrema
follow the new inputs. Labels remain noneditable even though napari's data setter
can re-enable editing. Normal failures restore the complete previous state;
unrecoverable native restoration hard-hides both canvases until explicit reopen.
Changed count/type/layout/grid also requires explicit reopen, without resampling.

Pending updates keep the last complete panes visible with disabled recipe edits
and an explicit previous-results message. Native layout diagnostics found the
initial verbose stale paragraph enlarged the footer from 35 to 52px at 900/12pt
and shrank the panes 17px. Compact routine stale/updating messages remove that
transition; detailed actionable errors still wrap. Visible current-session review
sinks participate in the existing one-level cache retention rule, retaining
only their bound A/B endpoints when the user selects another node. This preserves
Low-memory live refresh without retaining the whole upstream graph or weakening
resource guards. Hidden/off-tab/closed reviews are not active retention targets.

External evidence in `review-images-20261008`:

- `review-live-focused.xml`: 275 focused Windows offscreen Qt checks, zero
  failures/errors/skips, 67 upstream warnings, JUnit 29.250s. Covers review UI,
  controller/widget, mask rendering, core descriptors/settings, icons and docs.
- `review-live-cache-neighbors.xml`: 27 separate existing low-memory, detached
  histogram and synchronous/detached cache-handoff checks, zero failures/errors/
  skips, 17 upstream warnings, JUnit 11.491s.
- `review-live-final-status.xml`: 20 affected checks pass, 126 deselected,
  zero failures/errors/skips, 17 warnings, pytest 12.12s. This overlaps the earlier
  focused run and adds two 900/1280 routine-status geometry regressions; counts
  are not additive full-suite qualification.
- `native-live-refresh-widths-20261009/result.json`: source-tree native Windows
  Qt/GL smoke, 2D/3D × 900/1280, three scalar+mask replacements per case. All twelve
  replacements preserve wrapper/camera/navigation/recipe identities, source
  bytes and read-only borrowing. Actual framebuffers change in both panes;
  visible 2D slices equal the requested source planes. The 1280px 3D-after image
  was visually inspected. `pending-geometry-final.json` separately verifies native
  900px/12pt stale→updating→ready transitions retain exact pane geometry.

The initial controller test record retains one overly insensitive mask-fixture
failure (23 other tests passed); the fixture's threshold/sigma were changed to
produce a known visible mask change rather than weakening the assertion. Final
terminal acceptance cases pass. Exploratory native probes retain incomplete
outputs: a wrong slice control key, an unlocalized pose assertion before settled
pending transitions/resource setup, and an assertion against intentionally
deferred hidden-layer slicing. None is called a clean pass. Final initialized,
settled probes check visible requested slices and exact final poses.

Repository lint/changed review-file formatting/whitespace checks pass. Manual 20
content contracts, 50 routes and strict build pass. No full suite, new wheel,
installation, user-session restart, commit/push or publication was performed.
The user's existing private demo still imports its preceding installed build.
Native Linux/macOS, minimum dependencies, large-volume memory/performance,
extreme-float graphics precision and acquired-microscopy accuracy remain
unqualified. This bounded rendering smoke does not prove numerical rendering
equivalence, biological accuracy, or installed/frozen-app lifecycle qualification.
