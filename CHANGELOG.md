# Changelog

## Unreleased

### Batch Workspace

- The main `Batch workspace...` action now sits between workflow loading and
  the separated export group. The duplicate action in the representative strip
  was removed; the retained workspace still reopens from the main toolbar.
  Both the main toolbar and Batch workspace now consistently place Load before
  Save.
- Saving a workflow while a Batch workspace is active now offers Yes/No/Cancel.
  Yes embeds the validated versioned batch config in the same workflow JSON and
  loading that workflow restores the workspace without running a preview. No
  keeps the ordinary graph-only workflow and standalone batch-config behavior.
- `Run batch` now performs a plan-only preflight and starts an unpreviewed or
  deliberately edited batch in the same click. `Preview batch` remains optional
  and is the only action that calculates a live graph representative; an
  unexpectedly changed, already reviewed plan still stops for confirmation.
- The output folder is now suggested as an `output` subdirectory of the first
  bound batch source. The amber field follows source changes until the user
  focuses, clicks, edits, or explicitly chooses the destination; saved config
  paths remain explicit and are never replaced by the suggestion.
- An item whose resolved `Skip` destinations all exist is now finalized without
  loading its source pixels or calculating the graph. Mixed existing/missing
  output items still calculate normally so missing outputs remain correct.
- Atomic artifact replacement now tolerates several seconds of transient
  Windows, cloud-sync, or indexing locks. Redundant item-sidecar rewrites were
  removed, and an exhausted final item-sidecar write is recorded as a partial
  item that obeys `continue_on_error` instead of aborting the entire run; final
  run-manifest persistence remains mandatory.

## 0.12.0a2 - 2026-07-16

### Release Overview

This second 0.12 alpha makes expensive interactive workflows easier to tune
and trust without changing workflow schema version 3. It adds isolated node
tuning, clearer actionable-versus-waiting execution states, progressive
run-scoped previews, responsive exact-pixel presentation paths, graph port
label controls, and more legible PSF/deconvolution guidance. Scientific cache
publication remains atomic, and existing schema-3 workflows remain
structurally compatible.

### Isolated Node Tuning

- Added `Tune node in isolation` to the node context menu and the top of the
  inspector, with a persistent `Downstream paused` panel.
- Parameter edits recalculate only the tuned node while every downstream node
  is held in the darker-amber waiting state and its prior cached output is
  retained; the tuned root remains the actionable bright-amber frontier.
- `Apply and continue` reuses the latest tuned output and resumes from its
  direct children; `Cancel tuning` restores the session-start parameters and
  cached result.
- Toolbar `Calculate all` now releases isolated tuning before normal execution,
  including for fully automatic graphs with no manual nodes.
- Apply stops any pending parameter debounce, and any graph/history edit safely
  commits the active tuning session before mutating the saved workflow.
- The isolated execution boundary is shared by synchronous and detached
  background runs and remains transient rather than entering workflow JSON.

### Responsive Result Presentation

- Generated inspector, pinned-label, and RGB-channel layers now use exact
  non-writeable views of cached scientific arrays instead of copying complete
  volumes for display. Boolean masks are converted only when a Labels layer
  requires an integer representation.
- Compatible napari Image layers are reused across image/mask dtype and
  same-rank shape changes. Mask blending, colormap, and contrast are reset
  explicitly when returning to a normal image, while Image/Labels, rank, RGB,
  and channel-layout changes still replace the layer.
- Reused layers invalidate old contrast tokens, and node selection rejects
  stale contrast or histogram results from previously selected outputs.
- Thumbnail sources are reduced to display resolution before rendering, exact
  stack contrast remains background-calculated, and the shared progress area
  stays visible through post-pipeline presentation work.
- Background runs now publish each completed node to its card immediately. The
  first thumbnail uses the exact completed pixels with scan-free presentation
  limits; partial worker results never enter the live scientific cache, and
  stale run/source revisions are ignored.
- Rescale Intensity now exposes cutoff and voxel-processing phases through the
  pipeline progress UI. Floating-point rescaling uses bounded float64 work
  chunks with unchanged output arithmetic, and exact 0/100-percentile cutoffs
  use a direct finite-extrema path instead of an unnecessary order statistic.

### Graph Port Labels

- Added a Settings > Port labels preference with `Ambiguous only` (default),
  `Show all`, and `Hide all` modes. Visible labels reserve horizontal gutters,
  multi-port rows reserve vertical space, and overlong names are elided with
  the complete name in a tooltip.
- Label-mode changes resize cards without moving manually arranged nodes. VIPP
  reports any resulting overlap and points to `Auto structure graph`, whose
  layout now consumes the expanded card dimensions.

### Deconvolution Safety And Guidance

- During background execution, each completed downstream node now leaves its
  dark-amber waiting state immediately instead of waiting for the entire branch
  to finish. This progressive display state remains separate from the live
  scientific cache until the final run result is accepted.
- Manual nodes that have never been calculated now use the same bright-amber
  action styling as stale manual barriers. The toolbar `Calculate all` action
  also turns amber whenever an uncalculated or stale manual frontier needs
  attention; waiting descendants remain dark amber.
- Every stale manual/cached node now acts as an execution barrier across VIPP.
  The actionable barrier remains bright amber, while stale descendants use a
  darker amber waiting state, retain their last coherent cached outputs when
  present, and resume in dependency order when the barrier is recalculated.
- Born-Wolf support fields now identify their user-set physical spans and the
  inspector separates Nyquist sampling, tail containment, and image-extent
  checks into concise statuses with direct actions. A documentation link carries
  the underlying support-selection and boundary-model guidance.
- Fixed wrapped RL/RL-TV guidance reserving too little rendered height while
  the Parameters group absorbed unused vertical space. Long PSF preflight
  notes now remain fully visible through normal inspector scrolling.
- Float parameter fields now use compact scientific notation for non-zero
  magnitudes below `0.001`, and every numeric node field keeps its standard
  edit menu while adding a right-click `Reset to default` action.
- Added a cached, read-only PSF preflight to both Richardson-Lucy inspectors.
  It reports rank, metadata-known physical sampling, finite/non-negative values,
  positive sum, approximate normalization, odd/even shape, peak and centroid
  offsets, and support relative to the image as `Ready`, `Warning`, `Invalid`,
  or `Unknown`. Missing calibration is explicit; no PSF is silently recentered,
  cropped, padded, normalized, or resampled by the diagnostic.
- Reworked PSF preflight presentation into separately colored passed checks,
  attention items, and next actions. Support warnings now name the affected
  axis and exact PSF/image sample counts, boundary intensity is explicitly
  distinguished from out-of-array intensity, and generated-PSF workflows
  explain why Prepare / Validate does not resize support.
- Added a conventional-widefield Nyquist estimate to Born-Wolf and downstream
  RL/RL-TV inspectors, kept it separate from image-extent/support checks, and
  made the fixed PSF support-window controls explicit. The inspector now states
  that a kernel matching an image axis has at most one centered fully supported
  position and no interior margin, while a larger kernel has none. Both can
  still calculate under the current zero-outside-image boundary assumption.
  Boundary-tail mass is broken down by Z, Y, and X.
- Kept wrong-rank and metadata-known sampling mismatch as hard failures while
  reporting absent physical calibration as a warning instead of inventing unit
  pixel spacing.
- Expanded RL/RL-TV parameter guidance and added one concise RL-TV scientific
  note covering under-convergence, feature loss from excessive TV, and PSF
  validation order. Reconstruction math, constant initialization, boundary
  handling, numerical-guard defaults, and the `0.002` TV default are unchanged.
- Changed the bundled 2D and 3D RL-TV comparisons to 25 iterations, TV
  regularization `0.002`, and denominator floor `0.05`. Their annotations now
  identify `0.008-0.012` as comparatively strong rather than a recommended
  default.

## 0.12.0a1 - 2026-07-14

### Release Overview

This alpha is a major reproducibility and architecture release. It replaces
several implicit scientific assumptions with persisted, validated contracts;
routes interactive, generated-Python, and batch work through shared headless
services; turns collection processing into a reviewable, provenance-rich
workspace; and decomposes the former widget-heavy implementation into focused
core and UI modules. Existing workflow JSON from schema versions 1 and 2 is not
silently upgraded because doing so would invent choices that can change output.

### Important Compatibility And Scientific Behavior

- Advanced workflow JSON to schema version 3. Versions 1 and 2 are
  intentionally rejected rather than receiving inferred scientific defaults.
  Keep the VIPP release that created an older workflow to inspect it, then
  recreate and verify it in 0.12; changing only the JSON version number is not
  a valid migration.
- Made `channel_axis` a required persisted choice for Crop Stack; average,
  Gaussian 2D/3D, median, bilateral, non-local-means, rolling-ball,
  background-subtraction, difference-of-Gaussians, unsharp, Sobel, Canny, and
  Laplace operations; and every automatic/manual/adaptive threshold family.
  `-1` now explicitly means scalar/no-channel data instead of shape-based RGB
  detection.
- Made Composite to RGB configuration explicit. `Channel axis mode` and `RGB
  mapping mode` each persist `Auto` or `Manual`; auto mode shows its resolved
  axis/mapping in disabled controls, while manual mode enables the relevant
  selectors. Legacy numeric axis and per-plane selectors remain hidden only
  for compatibility with existing schema-3 files.
- Replaced three fixed RGB-plane selectors with one dynamic colour assignment
  per detected source channel. Manual mapping supports arbitrary channel counts
  and `Unassigned`, Red, Green, Blue, Magenta, Cyan, or Yellow; unassigned
  channels contribute nothing and several channels can contribute additively
  to one or more RGB planes.
- Kept Composite to RGB intensity mapping explicit. New nodes preserve native
  numeric values by default; independent per-channel 1st-to-99th-percentile
  normalization remains available only as an explicitly selected lossy mode.
- Limited Composite to RGB mapping edits to invalidating/recalculating that
  node and its downstream dependants. Every already calculated upstream manual
  result—including a deconvolution several hops away—is retained in Keep-all,
  Smart, and Low-memory modes. Automatic upstream intermediates are not
  invalidated by the edit but remain subject to the selected cache mode's
  intentional pruning policy.
- Stopped treating a trailing length-three or length-four axis as implicit
  RGB/RGBA. Generic `C` axes remain scientific fluorescence channels unless
  the axis is explicitly declared `rgb` or `rgba`.
- Generated Python exports record the exact VIPP version that created them and
  refuse a different runtime version. Regenerate and revalidate an export under
  the release that will execute it.
- File sources are frozen to one verified path/series revision until Refresh.
  Live lazy arrays, rotation, shear, unsupported affine transforms, and source
  revisions that change during calculation now fail explicitly.
- Same array shape no longer implies physical-grid compatibility. Multi-image,
  image/mask, and image/PSF operations validate axis meaning, size, scale,
  compatible units, and origin instead of silently resampling or registering.
- Batch workflows reject enabled `Save Image` side effects; use explicit
  `Batch Output` nodes so every planned write is collision-checked and recorded.

### Deterministic Batch Configuration And Provenance

- Added loadable, versioned `vipp_batch_config.json` files preserving source
  bindings, resolved selected outputs, naming/format choices, the required
  workflow companion, optional runner choice, scientific workflow hash, and
  `Error`/`Skip`/`Overwrite` existing-file policy.
- Unified preview and execution around one deterministic sorted positional
  source-pairing and output-planning service, including duplicate, existing,
  input-overlap, and within-plan collision state before graph execution.
- Kept explicit `Batch Output` nodes authoritative. Terminal outputs retain a
  warned compatibility fallback only when every terminal has one output port.
- Added latest-run and archived `vipp_batch_manifest.json` provenance plus
  atomic per-item sidecars/checkpoints containing workflow/config hashes,
  software versions, input identities, source metadata, planned outputs,
  errors, and `pending`/`completed`/`skipped`/`failed` records.
- Isolated failures by item and output. Successful writes and provenance remain
  available, later items continue by default, and the final summary distinguishes
  completed, partial, skipped, and failed items.
- Stage each item's outputs privately, reverify every bound source identity,
  and only then promote outputs atomically. A source revision that changes
  during execution publishes no outputs; later promotion failures are recorded
  as partial rather than being hidden.
- Changed batch-created `vipp_batch_pipeline.py` into a thin command-line
  launcher that defaults to its sibling config, resolves the recorded workflow,
  and delegates to the shared headless batch core.
- Added a deterministic generated validation bundle with three paired items,
  two sources, nine explicit NPY/TIFF/TSV outputs, exact scientific ground
  truth, portable workflow/config/runner artifacts, manifests, archives, and
  per-item sidecars.
- Renamed the ambiguous demo entry to `Open batch demo...`. It creates a safe
  working copy, loads the first paired field through the interactive graph,
  opens an already configured collision-aware preview, explains the three-item/
  nine-output plan, and provides a direct `Run demo batch` next step.
- Kept semantic-axis iteration and plate/well/field HCS traversal explicitly
  outside this local-collection release instead of inferring them from array
  positions or directory names.

### Batch Workspace And Representative Navigation

- Replaced the transient `Run batch...` dialog with a retained `Batch
  workspace...` that keeps setup, representative selection, item-level run
  progress, final statuses, validation, and the manifest path inspectable.
- Added a persistent Previous/Next/slider navigator for the complete batch
  plan. Selecting an item atomically swaps every paired collection Image Source
  and recalculates that representative through the graph without saving batch
  outputs or changing serialized source parameters and the workflow hash.
- Connected preview-table selection and double-click activation to the same
  representative session, while retaining limited table rendering for large
  plans and full-plan navigation through the slider.
- Made the deterministic paired demo auto-load its data into this session and
  clarified throughout the UI that one graph representative is distinct from
  running the complete collection.
- Added requested-versus-committed representative tracking, bounded materialized
  source caching, stale-workflow invalidation, and exact reviewed-plan checks so
  failed or changed inputs cannot be presented or run as the prior preview.
- Retained failed and completed run evidence with truthful progress, historical
  preflight labelling, and a required fresh review before replay.
- Made the retained workspace responsive on smaller displays: setup and results
  scroll vertically beneath a fixed Run/Close footer, preview paths no longer
  force oversized table columns, and long representative details stack in a
  narrow main dock across supported Qt platforms.

### Stable Scientific Sources And Physical Grids

- Added exact file and directory identities based on path revision and bytes,
  checked before and after source inspection/materialization. Interactive file
  arrays are owned, read-only snapshots pinned until Refresh.
- Moved OME-Zarr, microscope, and large local-file materialization into typed
  background workers, rejecting stale worker results and changed on-disk
  revisions.
- Snapshot live NumPy-backed napari layers with revision tokens. Data,
  metadata, RGB, axis, scale, translation, unit, rotation, shear, and affine
  events invalidate the source; stale background results are discarded.
- Preserve supported napari axis labels, scale, translation, and units at the
  source boundary while rejecting transforms that cannot be represented
  without changing pixels.
- Give inspect and pinned napari layers detached data so display edits cannot
  mutate graph caches or later scientific results.
- Added reusable grid validation and semantic mask broadcasting. Broadcasting
  is based on unique axis/calibration correspondence, never coincident sizes.
- Require image and PSF sampling compatibility before deconvolution and reject
  hidden resampling, origin repair, or dimension guessing.

### Explicit Axes, Channels, And Operation Contracts

- Added per-axis semantic confidence distinguishing explicit metadata from
  shape inference. Automatic spatial, channel, projection, and PSF decisions
  reject inferred-only ambiguity when scientific meaning matters.
- Reject malformed or duplicated axes, stale shapes, non-finite scale/origin,
  and non-positive calibration instead of silently rebuilding metadata.
- Reorder Axes now moves each complete axis record with its pixels. Named crop,
  projection, rescale, and measurement operations follow semantic axes;
  positional kernels reject explicit non-canonical layouts they cannot support.
- Projection's non-YX mode uses the named non-YX spatial axes, and measurement
  operations select named YX/ZYX axes after reordering rather than assuming
  trailing dimensions.
- Denoisers use one reversible whole-input intensity transform, preserve native
  floating scale and constants, retain valid unsharp-mask overshoot, support
  declared non-trailing channel axes, and reject invalid/non-finite ranges.
- Composite to RGB auto axis resolution requires explicit carried channel
  semantics. Auto RGB mapping exposes its resolved mapping: declared RGB/RGBA
  preserves encoded RGB order (alpha ignored), while fluorescence stacks blend
  every channel by carried pseudo-colour or the documented repeating default
  colour order, including stacks with more than three channels.
- Composite to RGB validates wide-integer precision, floating overflow,
  non-finite/complex/object data, duplicate axes, invalid manual indices, and
  ambiguous automatic axis resolution; provenance records resolved
  axis/mapping, dtype, and intensity mapping. Manual axis mode deliberately
  permits any selected axis, including a spatial axis such as Z, even when
  metadata declares a separate C axis.
- Tightened crop margins, projection/split/range grammar, ordered Canny,
  hysteresis and difference-of-Gaussians parameters, exact Image Calculator
  operand counts/weights, finite gamma/linear parameters, and dtype conversion.
  Invalid values fail instead of being clamped, swapped, rounded, or repaired.
- Kept mask conversion elementwise and shape-preserving, and ensured scientific
  kernels accept read-only inputs without mutating source data.
- Fixed disconnected Born-Wolf PSF editor context and rejected invalid requested
  dimensionality instead of silently selecting another rank.

### Interactive Parameter And Histogram Controls

- Added effect-oriented tooltips to every Richardson-Lucy TV parameter and its
  label, slider, spinner, checkbox, or choice control.
- Reduced the iteration slider to a practical 1-100 window and changed TV
  regularization, TV epsilon, filter epsilon, and denominator floor to true
  geometric sliders with parameter-specific ranges.
- Decoupled those slider windows from spinner entry: valid zero/off and
  out-of-window values remain directly enterable without expanding the slider.
- Made both Richardson-Lucy parameter forms responsive: long spatial-mode
  choices and numeric controls can shrink or wrap instead of forcing the
  inspector panel wide, while spinner entry remains fully available.
- Made both Rescale Intensity input guides draggable. Dragging a
  percentile-derived guide switches the node to explicit values, preserves the
  other exact cutoff, persists the edit, and queues interactive recalculation.
- Prevented a click on an input-histogram marker from changing its parameter
  unless the pointer actually moves through a drag.
- Added real mouse-drag and re-selection regression coverage for Rescale
  Intensity and Binary Threshold histogram guides.

### Exact Diagnostics And Presentation Isolation

- Added a Qt-free diagnostics core for exact finite statistics, extrema,
  percentiles, histograms, contrast limits, and label-volume summaries.
- Use bounded chunks only for memory control, never hidden sampling; preserve
  wide-integer levels and require explicit channel behavior.
- Added typed, stale-safe diagnostic workers. Provisional contrast is display
  only, and generated viewer-layer presentation cannot feed back into scientific
  source selection or graph caches.

### Workflow Execution, Export, And Persistence

- Added detached validated `GraphSnapshot` and `WorkflowSnapshot` values for
  history and background execution, including full graph/port/type/cycle
  validation when materialized and canonical tunnel references.
- Added atomic, fsynced JSON/text replacement and rejected non-finite JSON so a
  failed workflow/config write leaves the prior file intact.
- Extracted typed headless execution requests/results; Qt workers are signal
  adapters, and widget shutdown now rejects late callbacks safely.
- Rebuilt `Export Python...` around the same shared executor used by VIPP.
  Generated scripts embed immutable validated workflow JSON, reconstruct a
  fresh pipeline per call, preserve `ImageState` through load/save, and support
  explicit multiple-source bindings.
- Generated exports fail on missing, duplicate, or unknown source bindings and
  on VIPP runtime-version mismatch instead of approximating metadata-dependent
  behavior with incomplete direct operation calls.
- Preserved the simple generated CLI as a primary-source convenience while the
  saved-config batch runner remains the complete multi-source collection path.

### Runtime, UI, Launching, Architecture, And Contributor Experience

- Bundled examples now select their first Image Source before computation;
  ordinary saved workflows still restore an explicitly saved inspector node.
  Keep-all thumbnails populate after background completion without another
  selection, and the processing status is no longer overwritten prematurely.
- Added `Focus` beside Refresh to recover the center of an infinite graph
  canvas without changing zoom, selection, layout, cache state, or undo history.
- Added the installed `vipp` command, `python -m napari_vipp`, and repository
  `./vipp` development launcher.
- Added explicit Windows, macOS, and POSIX memory-reporting branches for cache
  status and the automatic memory guard. Windows uses its native memory API and
  never assumes that POSIX `os.sysconf` exists; missing platform counters fall
  back safely without interrupting graph execution.
- Hardened the batch representative navigator's compact layout so long batch
  identifiers, paired filenames, progress details, and buttons can shrink or
  wrap inside a 420 px dock under platform-specific Qt font metrics.
- Reduced `_widget.py` by roughly 4,500 net lines and made it the composition
  root. Extracted controls, axis editors, dialogs, plots, examples, sources,
  workers, history, lifecycle, view dimensions, and batch UI/services into
  focused `ui/` modules.
- Added Qt-free core boundaries for grids, diagnostics, snapshots, execution,
  atomic I/O, source identity, file snapshots, and batch setup.
- Added architecture tests enforcing dependency direction, expanded scientific
  contract/golden/example tests, and documented contributor boundaries and the
  proportional verification ladder.

### Upgrade Checklist

- Keep an environment containing the VIPP version that created a schema 1 or 2
  workflow. Use it to inspect the old graph, then recreate and scientifically
  verify the workflow under 0.12 schema 3; do not edit only the JSON version.
- Re-export generated Python under 0.12. Exported programs deliberately require
  the exact VIPP version that generated them.
- Review every formerly inferred channel axis, spatial mode, cutoff mode, RGB
  declaration, Composite-to-RGB intensity mapping, and physical grid before
  accepting regenerated outputs.
- Replace enabled `Save Image` side effects in collection workflows with
  explicit `Batch Output` nodes, preview the complete plan, inspect
  representatives, and retain the final manifest and item sidecars.

### Deliberate Limits In This Alpha

- Collection batching is local-folder and sorted-positional. Semantic-axis
  iteration and plate/well/field HCS traversal are not inferred in this release.
- Most processing remains eager. Large OME-Zarr data still needs deliberate
  cache and materialization choices; pyramid-aware interactive previews and
  broader lazy execution are later milestones.
- Proprietary microscope-reader coverage depends on optional third-party
  readers and remains experimental across real facility datasets.
- Validation is strongest for deterministic operation tests and calibrated
  morphology phantoms; broader real-data validation remains necessary for
  restoration, colocalization, watershed, skeleton/network, vendor formats,
  batch deployments, and OME-Zarr round-tripping.

## 0.11.0a3 - 2026-07-12

- Standardized the public name as VIPP, the Visual Image Processing Platform,
  and added the reusable logo/mark asset set and concise README presentation.
- Automatically dispatch pipeline updates to the background for image data at
  least 32 MiB or four million values, while retaining the explicit
  `Run all in BG` override for smaller work.
- Moved large input-histogram and automatic-threshold diagnostics off the Qt
  thread, coalesced repeated dimension refreshes, and cached completed results.
- Reduced global-threshold, histogram, thumbnail, and segmentation memory
  pressure with exact bounded chunks, dtype and boolean-mask reuse, zero-copy
  mask inspection, and known mask contrast limits. Histogram calculations still
  count every finite pixel and do not introduce hidden large-array sampling.
- Added dtype-aware automatic-threshold handling: explicit boolean-mask
  passthrough, exact native integer levels, a saved `Float histogram bins`
  setting, actionable rejection of integer spans above 65,536 levels, and
  background treatment of all non-finite pixels.
- Added an explicit saved `Input cutoffs` mode to Rescale Intensity. New nodes
  default to exact all-finite-value percentiles.
- Added an explicit saved cutoff mode to Clip Intensity. New nodes default to
  `Data range`.
- Preserved adjacent int64/uint64 levels in Rescale Intensity by calculating
  integer percentiles as exact native order statistics and applying the mapping
  in translated coordinates. Clip now clamps integer data in its native dtype.
  Unrepresentable fractional integer bounds, rounded GUI values above 2^53,
  and rescale spans too wide for level-faithful float64 arithmetic fail with an
  actionable error instead of silently corrupting the result.
- Boolean inputs to automatic thresholds are now preserved explicitly as
  already-segmented masks. Minimum Threshold exposes its saved smoothing limit
  and reports failure instead of substituting an unrelated cutoff. Empty or
  all-nonfinite threshold inputs also fail instead of receiving a fabricated
  zero cutoff.
- Made selected-node auto contrast and generated napari-layer contrast exact
  over all finite values while moving large calculations off the Qt thread;
  provisional layer limits are display-only and never affect graph data.
- Replaced sampled metadata range/pattern inference with exact bounded scans.
- Made colocalization scatter density, ROI counts, and colocalized counts exact
  over every ROI voxel using bounded background accumulation; removed the old
  hidden stride/sample display paths.
- Increased the exact colocalization scatter-density display grid from
  192 x 192 to 255 x 255 cells for finer visual detail without changing
  thresholds or reported counts.
- Preserved float64 precision when RGB inputs are converted to luminance for
  automatic thresholding instead of silently downcasting them to float32.
- Preserved exact native threshold decisions for large-magnitude int64/uint64
  images by calculating on translated levels and restoring Python-integer
  cutoffs. Li similarly uses exact native offsets and rejects only relative
  spans that exceed float64's exact integer range.
- Advanced workflow JSON to version 2 so histogram-bin and cutoff-mode controls
  are explicit required scientific parameters. Version 1 files are rejected
  rather than silently receiving defaults that could change scientific output;
  keep `0.11.0a2` to run them unchanged or recreate them in the current release.
- Made `Split Channels` present its sole distinct downstream-used output across
  the thumbnail, inspect/pin, histogram, metadata, dimension, and selected-save
  surfaces. Nodes with zero or multiple distinct used outputs still use the
  saved `Thumbnail channel`; this display choice does not mutate that setting
  or any scientific graph output.
- Decoupled input-histogram distributions from parameter-dependent guide
  markers. Dragging Binary/Hysteresis thresholds or explicit Rescale/Clip
  cutoffs now reuses unchanged counts immediately, while computed automatic
  markers refresh independently and real input, scope, or slice changes still
  invalidate the distribution. Label-volume filters likewise reuse their
  unchanged object-volume population while minimum/maximum guides move.
- Let the wrapped colocalization description expand instead of clipping it, and
  added the ROI-based percentage beside exact colocalized/ROI voxel counts in
  the description, plot annotation, and tooltip. Empty ROIs report `n/a`.

## 0.11.0a2 - 2026-07-11

- Corrected the package's Python requirement to 3.12 or newer, matching the
  installable OME-Zarr dependency baseline.
- Added cross-platform Linux/Windows CI for manifest validation, linting,
  packaging, and Qt-aware tests on representative supported Python versions.
- Added contributor, support, security, conduct, issue, and pull-request
  guidance for safer and more reviewable community participation.
- Hardened graph restoration so invalid cycles, ports, duplicate input targets,
  source-node targets, and tunnel definitions are rejected atomically.
- Preserved dynamic output-count hints during workflow restore and stopped
  invalid dynamic source ports from silently falling back to port 0, including
  saved multi-channel Born-Wolf PSF outputs before runtime shape inference.
- Fixed generated Python for incomplete multi-input nodes, invalid or colliding
  identifiers, source-only graphs, custom entry-point names, generated-helper
  collisions, and workflows whose node ids normalize to the same variable.
- Tightened NumPy, workflow, and table output path validation, including NPY
  suffix normalization and overwrite protection.
- Improved background clipboard retry behavior and input-count consistency.
- Added graph, operation-registry, export, I/O, documentation-link, and workflow
  contract tests while replacing a slow redundant widget smoke test with a
  focused palette invariant.
- Reworked the README and user documentation, added a documentation index, and
  reconciled planning, architecture, research, and release records with the
  current 0.11 baseline.
- Expanded the development example launcher to cover every bundled workflow,
  list valid example ids, accept external JSON files, and reject misspelled ids
  instead of silently opening an unrelated workflow.
- Added a deterministic dark-theme documentation screenshot generator for
  full-context, floating-workflow, focused-inspector, and 3D napari views.
- Corrected the OME-Zarr label-export error so it points to the already
  available image-linked analysis-dataset export.
- Preserved saved `Select Table Columns` choices when its inspector opens
  before an upstream manual measurement has been calculated.

## 0.11.0a1 - 2026-07-09

- Changed the project license to BSD 3-Clause for compatibility with napari and
  the broader scientific Python ecosystem.
- Added optional microscope-reader routing for ND2, CZI/LSM, Leica LIF/LOF/
  XLIF, Olympus OIR/OIB/OIF/VSI, and BioIO/Bio-Formats-backed fallback paths.
- Added a missing optional-reader dialog that reports the required file reader
  extra and lets users copy the install command from the UI.
- Added normalized acquisition metadata fields for objective, refractive index,
  channel wavelength, and conservative upstream-deconvolution detection.
- Added `Born-Wolf PSF`, `Prepare / Validate PSF`, baseline
  `Richardson-Lucy Deconvolution`, and `Richardson-Lucy TV Deconvolution`.
- Added deterministic 2D and 3D deconvolution samples plus example workflows
  that compare ordinary Richardson-Lucy with RL-TV using an explicit prepared
  PSF input.
- Refined Born-Wolf PSF auto-parameter visibility, channel-specific PSF outputs,
  and background execution defaults for slower PSF/restoration workflows.
- Added slice/stack thumbnail contrast range handling with cached stack limits,
  and a linked/unlinked napari/VIPP slider setting for large data review.
- Improved background cancellation for rerun requests and simplified redundant
  long-running progress labels.
- Fixed channel-aware previews for `Extract Channel` and retained-output
  `Split Channels` workflows.
- Added a grouped `Open example...` chooser for bundled workflow templates and
  packaged the example workflow JSON files with the plugin.
- Moved napari-layer selection into the `Image Source` inspector so the toolbar
  is not carrying a confusing global input dropdown.
- Fixed the red-channel label-cleanup examples so they use the red/TRITC-like
  channel, and made the clear-border plus volume-filter cleanup visibly remove
  real labels.
- Updated restoration, microscope import, example-workflow, and release-roadmap
  documentation for the 0.11 alpha scope.

## 0.10.0a1 - 2026-07-07

- Added graph search/focus for node titles, operation IDs, named tunnels, and
  `Batch Output` tags.
- Added an ambiguous insert-on-wire chooser so users can select the inserted
  node input/output mapping when several compatible port mappings are possible.
- Added dynamic `Split Channels` output inference before execution so large
  channel counts expose the right graph ports while editing.
- Added `Split Axis` for explicit splitting of time, Z, or other non-channel
  stack axes.
- Added workflow UI metadata for selected inspector state, with optional
  per-node thumbnail visibility persistence controlled by Settings.
- Added explicit cache modes, cache/RAM status, auto memory guard, per-node
  `Keep output cached`, and low-memory batch retention.
- Added saved graph notes, tunnel reveal/highlight, and a tunnel manager for
  filtering, renaming, deleting, focusing, and auditing named sources.
- Fixed OME-Zarr analysis package validation so mismatched label shapes are
  rejected before writing the reference image store.

## 0.9.0a1 - 2026-07-05

- Added colocalization workflows for masked pixel metrics, threshold scatter
  inspection, RACC-style outputs, object colocalization metrics, and object
  association tables.
- Added publication-facing colocalization method notes for Pearson, Manders,
  Costes, RACC, and object-association assumptions.
- Added analytical phantom validation for calibrated 2D/3D morphology on
  rectangles, cuboids, spheres, ellipsoids, and anisotropic voxel sizes.
- Added a local collection batch runner that executes workflows over matched
  folder inputs and writes reproducibility artifacts.
- Added explicit `Batch Output` nodes so workflows can mark exactly which
  images, labels, masks, or tables should be saved during batch runs.
- Added background-run cancellation controls, cooperative progress/cancellation
  for long operations, and determinate toolbar progress where operations report
  work units.
- Added draggable histogram threshold markers for rescale intensity, clip,
  binary threshold, hysteresis threshold, and label-volume filtering.
- Added focused tests for the new colocalization, validation, batch-output,
  progress, and histogram-marker workflows.

## 0.8.3a1 - 2026-07-02

- Added `Summarize Measurements`, a table node that groups measurement rows by
  metadata or axis-index columns and calculates count, mean, median, standard
  deviation, min/max, and quartiles for selected numeric columns.
- Added a deterministic `VIPP synthetic measurement summary` sample plus an
  example workflow for validating grouped object-count and area summaries.
- Added derived object morphology groups to `Measure Objects` and
  `Measure Objects + Intensity`: shape ratios, bounding-box side lengths/aspect
  ratios, fill fraction, inertia eigenvalue ratios, Crofton-based circularity,
  perimeter-to-area ratio, and Hu moments.
- Added a deterministic `VIPP synthetic object morphology` sample plus an
  example workflow for validating derived shape ratios, circularity, and Hu
  moments.
- Added `Measure 3D Mesh Morphology`, a manual/cached true-3D label measurement
  node for mesh surface area, mesh volume, sphericity, surface-to-volume ratio,
  convex-hull metrics, 3D solidity, and per-object mesh status/error reporting.
- Added a deterministic anisotropic `VIPP synthetic 3D mesh morphology` sample
  plus an example workflow that merges standard object measurements with mesh
  morphology measurements.
- Added a responsive `View dims` bar with VIPP-local T/Z/C-style controls that
  synchronize with napari dims and remain usable when napari hides slice sliders
  in 3D view.
- Fixed `View dims` synchronization for downstream nodes that drop axes, such as
  Split Channels -> Gaussian Blur, so the VIPP Z/T/C controls continue driving
  the same source napari dimension.
- Added skeleton QC and cleanup nodes: `Skeleton Keypoints`, `Label Skeleton
  Components`, `Label Skeleton Branches`, and `Prune Skeleton Branches`.
- Added `Skeleton Graph Overlay` for RGB edge/node visualization and
  `Measure Skeleton Branches` for row-per-branch length, endpoint, and
  tortuosity tables.
- Added `Skeleton Graph Tables`, a manual table node that exports explicit
  graph-node and graph-edge tables from skeleton masks.
- Added `Summarize Skeleton Branches`, a table node that converts
  row-per-branch skeleton measurements into grouped length/tortuosity
  distributions and branch-type count/fraction summaries.
- Added `Measure Overall Skeleton Network`, a manual table node for per-block
  connectedness, fragmentation, branch-count, and branch-length whole-network
  metrics, including normalized per-component and per-length connectivity
  columns.
- Added pixel/voxel versus physical-unit thresholding to `Prune Skeleton
  Branches` when pixel-size metadata is available.
- Fixed `Skeleton Graph Overlay` output metadata so 2D napari inspect/pin
  layers display the result as channel-last RGB instead of grayscale.
- Display volumetric RGB outputs as separate additive red/green/blue napari
  layers, avoiding napari's 3D RGB-volume scalar-field status/rendering path.
- Regrouped skeleton nodes by output type: visual mask/RGB skeleton QC under
  Morphology, skeleton label images under Label Operations, and skeleton tables
  under Measurements.
- Added a dedicated skeleton-node user guide covering expected inputs, outputs,
  and intended use for each skeleton/network node.
- Added a deterministic `VIPP synthetic skeleton network` sample plus an
  example workflow for validating skeleton keypoint masks, branch labels,
  component labels, pruning, and before/after skeleton analysis.
- Added a deterministic `VIPP synthetic advanced skeleton network` sample plus
  an example workflow for validating time-indexed 3D skeleton graph overlays,
  branch tables, branch-summary tables, explicit graph node/edge tables,
  network summaries, pruning, loops, disconnected fragments, and anisotropic
  physical calibration.
- Adjusted thumbnail percentile contrast so sparse bright foreground objects
  are not dropped as outliers while low-amplitude background ramps are stretched.
- Added manual/cached execution for expensive table nodes. `Measure Objects`,
  `Measure Objects + Intensity`, `Measure 3D Mesh Morphology`,
  `Analyze Skeleton`, `Measure Skeleton Branches`, `Skeleton Graph Tables`, and
  `Measure Overall Skeleton Network` now expose `Calculate`/`Recalculate`, keep
  the last result available downstream when stale, and recompute
  deterministically in headless/export runs.
- Added per-node `Auto Recalculate` for manual nodes, with a warning that it
  can be slow on large inputs. Manual node cards now use gray, green, orange,
  and red state colors for not calculated, ready, stale, and error results.
- Added a toolbar `Calculate all` button that calculates all manual nodes whose
  cached results are missing, stale, or errored.

## 0.8.2a2 - 2026-06-30

- Changed the project license for this alpha line to PolyForm Shield License
  1.0.0. This was superseded by the BSD 3-Clause license in `0.11.0a1`.
- Added project-specific commercial-permission guidance, required notice text,
  and citation metadata for that license experiment.
- Documented that versions published through `0.8.2a1` remain under BSD
  3-Clause terms, while later releases use their declared distribution license.

## 0.8.2a1 - 2026-06-30

- Added insertion of already-existing loose nodes by dragging them onto a graph
  wire, sharing the connector glow, local make-room, and single-step undo
  behavior used by palette and right-click insertion.
- Made nodes translucent while dragging so connector insertion targets remain
  visible underneath the moving card.
- Kept loose-node drags from live-rerouting unrelated wires before drop; wires
  now route around the node only after it is placed.
- Made insert-on-wire spacing gap-aware so downstream nodes move only by the
  extra space needed for the inserted card and padding.
- Improved numeric spinbox editing so partial decimal input such as `1.` is not
  reformatted before the user can finish typing, and floats display without
  unnecessary trailing zeros.

## 0.8.1a1 - 2026-06-29

- Added right-click connector insertion with a compatible-node picker.
- Added one-shot `Auto structure graph` layout cleanup with undo.
- Added staged toolbar compaction for narrow dock layouts, moving crowded
  controls into `Settings` as space runs out.
- Improved graph layout planning and test coverage for connector insertion and
  auto-structure behavior.

## 0.8.0a1 - 2026-06-29

- Drag nodes onto existing wires to insert them into a pipeline.
- Newly inserted nodes are centered between their connected neighbors.
- Graph wires route more cleanly around nearby nodes and stay readable as nodes
  move.
- Napari preview scaling and RGB/composite thumbnail slice tracking are more
  reliable.
- Spatial processing options now show the resolved 2D or 3D mode.

## 0.7.2a1 - 2026-06-26

- Restored the global `Run all in BG` workflow toggle after it was dropped in a
  later widget refactor.
- Added a visible VIPP version badge in the workflow header so running builds
  are easier to identify during testing.

## 0.7.1a1 - 2026-06-25

- Added `Auto Watershed From Mask` as a single-node object-separation workflow
  that chains distance transform, h-maxima marker detection, and watershed.
- Kept the advanced watershed building blocks available separately and improved
  inspector guidance for marker-controlled watershed inputs.
- Improved watershed parameter UX with a saner `H` slider range, explicit
  auto spatial-mode resolution feedback, and a clearer explanation that `H`
  is a distance-map prominence measured in pixels/voxels.
- Made the graph canvas auto-expand as nodes or the viewport approach the
  scene edge, so long pipelines are no longer constrained by a fixed canvas.

## 0.7.0a1 - 2026-06-25

- Marked the package release maturity as Alpha in project metadata.
- Updated package version metadata to `0.7.0a1`.
- Added an explicit alpha disclaimer and acknowledgement guidance to the README.
- Added a release runbook for PyPI, GitHub Releases, and napari hub listing.

- Added a shared headless image I/O registry used by Image Source, quick save,
  Save Image, and generated Python scripts.
- Added OME-TIFF import/export with series discovery, semantic axes, physical
  scale, channel metadata, source identity, and VIPP workflow provenance.
- Added explicit ImageJ TIFF and conventional TIFF export modes. ImageJ mode
  writes calibrated hyperstacks; conventional TIFF preserves 32-bit label IDs.
- Added local OME-Zarr 0.4/0.5 image import/export with lazy Dask-backed reads,
  semantic axes, scale, channel names, and namespaced VIPP provenance.
- OME-Zarr export now explicitly stores channel/display metadata so channel
  names round-trip across current `ome-zarr` writer behavior.
- Added graph-aware Export OME Analysis Dataset for a reference image plus
  OME-Zarr `image-label` groups, preserving integer label IDs and label-node
  provenance.
- OME-Zarr label groups now import as VIPP label images rather than ordinary
  intensity images.
- Image Source now adapts to multi-image files/stores with a series selector,
  OME-Zarr folder browser, source summary, and stored single-item/collection
  binding mode.
- Added toolbar graph zoom controls with a calibrated 40%-250% slider, compact
  reset icon for the `100%` default, and synced Ctrl/trackpad wheel zoom that
  can continue beyond the slider range. The `100%` default corresponds to the
  previous calibrated 125% graph size.
- Extended `ImageState` with normalized channel, acquisition, and source
  metadata, and allowed source nodes to inject reader-built state directly.
- Added source-axis tracking to image metadata so thumbnails, slice histograms,
  and current-view labels follow the correct napari sliders after nodes remove
  axes, such as `Split Channels` removing C from `TCZYX`. Napari-layer sources
  now also right-align their axes to the current viewer dimensions, and thumbnail
  refreshes listen to durable dims callbacks.
- Added independent slice-vs-stack selectors for input histograms on
  cutoff nodes such as `Rescale Intensity`, `Clip`, and `Binary Threshold`.
- Reorganized the palette so intensity remapping nodes live under `Intensity &
  Contrast`, `Convert Dtype` is the first `Image Data > Utilities` node, and
  alpha/beta contrast now uses the `linear_scale_offset` operation and appears
  as `Linear Scale + Offset`.
- Expanded filtering and segmentation nodes with filtering subgroups for
  smoothing/denoising versus edge/detail operations, and segmentation subgroups
  for global versus local thresholding. Added Li, Yen, Isodata, Minimum,
  Sauvola, and Niblack threshold nodes plus Difference of Gaussians, Unsharp
  Mask, Sobel, Laplace, and Non-Local Means filtering nodes.
- Added `Hysteresis Threshold` under global threshold segmentation and
  `Canny Edges` under `Filtering > Edge & Detail`. Hysteresis exposes low/high
  input-histogram markers and 2D/3D spatial processing; Canny runs slice-wise
  with quantile thresholds.
- Added a reusable stack-processing notice for slice-wise nodes. XY-only
  filters, local thresholds, edge detectors, and XY morphology now warn on stack
  inputs that they process each `YX` slice independently and suggest
  `Reorder Axes` when another plane is intended.
- Global automatic threshold nodes now expose `Threshold uses` on stack inputs,
  defaulting to `Stack histogram` with an optional `Slice histogram` mode. The
  control changes how the cutoff is computed, not whether the output is a stack,
  and the input histogram now shows a live marker at the chosen threshold.
- Added `Reorder Axes`, which transposes arrays from a draggable axis-order
  list and reinterprets spatial metadata by output position so downstream
  nodes treat the result as a reoriented volume. Its graph thumbnail remains
  state-aware and follows napari slice sliders after the transpose.
- Added node right-click context menus with Delete, Inspect Code, Duplicate
  Node, and contextual Pin/Unpin actions. Node-card pin buttons were removed;
  pinning remains available from the inspector and context menu. Inspect Code
  now uses lightweight Python syntax highlighting.
- Channel pseudo-colours are now carried through `ChannelMetadata.color`.
  Image Source exposes channel colour controls when a channel axis is known,
  Combine Channels writes its colour choices into metadata, and the new
  `Assign Channel Colors` node can reassign colours mid-workflow without
  changing pixel data.
- Composite → RGB auto mode now blends source channels by carried
  pseudo-colours, so yellow/cyan/magenta channels contribute to the appropriate
  RGB planes. Manual red/green/blue channel selectors still force explicit
  single-channel plane mapping.
- Added a toolbar `Mono` colormap selector for monochrome graph thumbnails,
  including gray, perceptual-style maps, and common fluorescence colours.
- Added global thumbnail display controls: a `Thumbnails` show/hide checkbox
  and a `Contrast` selector with `Percentile`, `Min-max`, and `Raw` modes.
  Per-node thumbnail disabling remains available as an additional opt-out.
- Composite → RGB auto-mapping now matches graph thumbnail fluorescence colour
  order for ordinary channel stacks while preserving true RGB/RGBA inputs, uses
  robust per-channel contrast for display-like RGB output, and keeps time/Z
  slider mapping aligned with the source image.
- Split Channels now has an inspector `Thumbnail channel` control for choosing
  which output port is shown on the node card without changing the actual
  channel outputs or downstream wiring.
- Added first-class table outputs, CSV/TSV table saving, table metadata
  summaries, generated-script table export, and an inspector table preview.
- Table-only nodes now hide the per-node thumbnail toggle instead of showing a
  disabled image-preview option.
- Added a first-class `labels` graph type and a new Label Operations palette
  category. Label outputs inspect and pin as napari Labels layers.
- Added `Label Connected Components`, `Filter Labels By Volume`, and
  `Relabel Sequential` nodes. They support explicit 2D/3D spatial processing,
  process leading time/channel dimensions independently, and preserve label IDs
  until relabeling is requested.
- Added `Measure Objects`, which measures label images with
  `skimage.measure.regionprops_table` and outputs one row per object with label
  ID, pixel/voxel size, calibrated physical area/volume when axis scale is
  available, centroid, bounding box, equivalent diameter, extent, and Euler
  number.
- Added named typed input slots and `Measure Objects + Intensity`, which accepts
  separate label and intensity-image inputs and outputs basic object morphology
  plus per-object mean, minimum, maximum, sum, and standard deviation intensity.
- Added `Merge Tables` and `Add Metadata Columns` for assembling measurement
  branches into one PCA-ready table with object-identity joins and explicit
  treatment/replicate/batch annotations.
- Added generic skeleton/network analysis nodes. `Skeletonize` thins binary
  masks in metadata-aware 2D or 3D spatial blocks, and `Analyze Skeleton`
  outputs a per-component table with skeleton voxel count, endpoint voxels,
  junction voxels, isolated nodes, branch/graph edge counts, voxel-graph edge
  count, cycle count, connected-component context, and calibrated length when
  spatial scale metadata is available.
- Added `Clear Border Objects` for binary masks and integer labels. It preserves
  label IDs, supports all-volume or lateral-only boundaries in 3D, and exposes
  an optional data-aware border buffer.
- Expanded `Fill Holes` with metadata-aware 2D/3D processing, an advanced
  per-slice mode, maximum hole area/volume filtering, connectivity control,
  input-aware mode choices, and a warning when slice-wise filling is selected
  for a z-stack.
- Replaced the obsolete `Volume Filter` operation with `Remove Small Objects`.
  The new node accepts masks or labels, preserves their semantic type, supports
  metadata-aware 2D/3D processing and mask connectivity, and uses contextual
  logarithmic area/volume controls.
- Workflow loading now uses one strict versioned schema. Unknown operations,
  malformed records, duplicate ids, invalid positions, and dangling
  connections are rejected instead of being silently ignored.
- Label TIFF saves now preserve 32-bit integer IDs using standard TIFF because
  ImageJ TIFF does not support 32-bit integer label data.
- Made Split Channels preserve upstream image/mask/label port types so
  thresholded channels connect directly to label operations.
- Added a prebuilt Otsu red-channel labeling workflow and manual launch script.
- Made label-volume filter sliders logarithmic and data-aware, using the largest
  incoming object while preserving exact numeric entry up to the hard limit.
- Added an incoming label-volume distribution to the volume-filter inspector,
  with live minimum and enabled maximum threshold markers plus a log-scale
  toggle that defaults on.
- Detached VIPP windows now use standard top-level window controls, including a
  maximize button. Double-clicking the detached title bar toggles maximized
  state instead of re-docking the panel.
- Renamed the `Channel Composite` node to `Combine Channels` to make it the
  clear complement of channel splitting; it still stacks its connected inputs
  into a multichannel image.
- Added a generic `Split Channels` node that emits one output port per channel
  in the image (replacing the fixed three-port `Split RGB`). The split is
  lossless and preserves dtype, and the port count adjusts to the true channel
  count once the node processes an image (a grayscale image yields a single
  port). `Combine Channels` and `Split Channels` are inverse operations and sit
  next to each other in the node palette.
- Added a single configurable `Composite → RGB` display node (merging the two
  earlier RGB nodes) that maps a multichannel composite to a channel-last RGB
  image. By default the channel axis is auto-detected and channels map in order
  (0→R, 1→G, 2→B; single channel→white); the channel axis and per-plane channel
  selections can be set explicitly.
- Fixed `Composite → RGB` controls so `-1 auto` remains selectable, auto channel
  detection uses carried axis metadata such as `CZYX`, and constant nonzero
  channels render visibly instead of becoming black. Generated RGB outputs are
  now added to napari with `rgb=True` so the RGB axis is not mistaken for a
  spatial/data slider.
- VIPP now maps napari slider positions back into canonical source-axis
  coordinates before updating thumbnails, histograms, and current-view labels.
  This keeps Z stepping linked between CZYX source images and ZYX RGB outputs
  even though napari exposes different slider counts for those layers.
- Added true multi-output support to the graph model, canvas, persistence, and
  Python export: connections now carry a `source_port`, nodes can declare static
  `OutputSpec` ports or a dynamic `output_factory`, and downstream wires resolve
  the selected port (stale wires to removed ports are trimmed automatically).

## 0.1.0

- Initial napari plugin scaffold.
- Added prototype visual workflow widget with node thumbnails and inspect/pin
  behavior.
