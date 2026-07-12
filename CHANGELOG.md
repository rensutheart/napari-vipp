# Changelog

## Unreleased

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
