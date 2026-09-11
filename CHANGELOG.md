# Changelog

## Unreleased

- Review the full example catalogue for readable graph placement. Separate crowded nodes, tunnel labels and explanatory notes; leave room for calculated-result controls, and widen the exhaustive showcase's seven logical lanes. Keep the examples' analysis parameters, connections, saved display profiles and operation coverage intact.

- Space out the **RACC Colocalization** example so channel tunnels clear neighbouring nodes. Use a 30,000 binary ROI threshold and manual thresholds of 43,970.51 / 48,073.03 for both RACC variants. Display all RACC index images with Magma in VIPP thumbnails, Inspect and pinned views; metric and voxel-overlay threshold modes are unchanged.

- Give the Intensity Histogram inspector a minimum drawable plot height plus adaptive space for its title, channel legend and axis labels. Narrow inspectors and larger fonts expand vertically instead of compressing the graph; horizontal titles use the available label bands. Histogram values, log scales, pop-out plots and exports are unchanged.

- Avoid resetting unchanged mesh display calibration during node selection, pinning and refresh, reducing unnecessary napari transform callbacks. Keep genuine calibration changes exact. Contain delayed selection display failures without changing calculated results; show a concise warning with **Details…** and **Dismiss** instead of an inline traceback filling the workflow dock.

- Simplify desktop launch to one **VIPP** app/shortcut using Auto, for both Windows CPU/CUDA installations and macOS. Remove the separate CPU/Prefer-GPU graphical entry points and the normal splash's Automatic badge/profile explanation; compute controls remain inside VIPP. Upgrades retain ownership and rollback checks when retiring old profile shortcuts. GPU dependencies remain a separate installation capability, not a separate app icon.

- Check for updates on every startup (unless disabled), retry a transient GitHub failure once, and explain connection failures without presenting a cached release as freshly checked. Separate installed/latest versions, status, update actions and preferences in the update dialog.

- Add an explicit **Download & open update** action for owned Windows desktop installations. Download official installer assets in the background, verify SHA-256 and the active installation, then open guided setup with the current location and CPU/CUDA profile selected. Setup still requires review; VIPP never silently updates a Python environment or closes unsaved work. Other installation types retain browser/manual update routes.

## 0.15.0a4 - 2026-09-10

- Make overlapping channels easier to compare in the detailed histogram window and PNG/TIFF exports: draw faint fills first, then continuous stepped channel outlines, without opaque vertical borders around every bin. Single-channel bars, bin values, hover details, logarithmic zero gaps and dense-plot peak preservation remain unchanged.

- Keep backward and same-column wires outside their source and destination cards, including wide multi-input nodes and while dragging. Avoid nearby cards even with close ports, check rounded detours against the whole graph, and use a bounded alternate-route search for crowded layouts. Moving or resizing cards updates their routes; workflow connections and calculations are unchanged.

## 0.15.0a3 - 2026-09-09

- Fix a malformed splash-screen style rule so Qt applies the intended profile badge and minimize-button styling without a stylesheet parsing warning.

- Prevent an intermittent splash-startup failure when a generated authentication token begins with a hyphen. Pass the token as an explicitly bound command-line value; keep the token validation and startup handshake unchanged.

- Keep the macOS installer’s bundled YAML dependency within conda’s declared compatible range, preventing an installed-environment `pip check` failure. Both native architecture checks remain required after rebuilding.

- Restore complete palette coverage in **Exhaustive Inspector Showcase** by adding label-boundary QC, and keep its packaged, standard and manual workflow copies synchronized when regenerating the example.

- Compact the Batch reproduction banner by placing its actions to the right of the status text, without a separate full-width button row.

- Show **Run complete · Original inputs verified** in green after a fully successful reproduction with matching VIPP versions. The banner describes the completed run, not permission for another: a new run still needs **Check batch**, and changing folders or settings clears the completed-run badge. Cancelled, partial, failed, or version-exception runs do not receive the green completion state.

- Preserve the latest original-input verification result when handing a reproduction batch from Run preflight to its worker. Verified original files no longer fail as "not checked" at startup, including after correcting a previously mismatched input folder. Execution still rechecks current inputs before processing and publication.

- Open a local workflow JSON dragged onto VIPP's workflow canvas, tab strip, or inspector in a new tab, preserving existing workflows and unsaved edits. Dropped files use the same validation and recorded-run reproduction choice as **Open**.

- Make Batch **Check details…** a compact result view with input-check totals, relevant issues and explicit VIPP version status. Its review button opens all inputs, affected inputs or Setup as appropriate; navigation never rechecks files, approves a version exception or starts a run.

- Preserve authored Image Source parameters when showing the inspector. Opening a portable recorded workflow on Windows no longer changes path separators and falsely rejects its attached Batch Setup as a different workflow.

- Clarify the reproducibility opening dialog with grouped choices, a separate VIPP version status and **Continue to batch setup**. Show release/setup links only where version help is needed; links open web pages without downloading or installing software. Keep explicit version acknowledgement and original-input checks unchanged.

- Add an explicit **Reproduce original run / Use workflow on new data** choice when opening recorded reproducibility workflows. Reproduction checks compare input contents and image selections, show per-item findings and aggregate mismatch counts, and prevent running with unverified inputs. Different VIPP versions need an explicit, recorded acknowledgement; reusing the workflow on new data removes the original comparison requirement. Input data are still shared separately.

- Add **Export reproducibility package…** for the current workflow recipe or a recorded batch run. Review an offline human-readable report, exact file contents and sharing omissions before saving a ZIP. Packages contain portable workflow/settings, a Python runner, software records and redacted run evidence, never raw images, result files or thumbnails. Local folders are hidden and filenames can be anonymised. Batch reports use the archived run snapshot; redacted evidence is explicitly not resumable and does not guarantee identical results on another machine.

- Make package repeat instructions VIPP-first: open the linked `workflow.json`, restore portable Batch Setup, choose input folders and a new output folder, then Check and Run. Include official recorded-version release/setup guidance with an explicit development-build caveat; no installer or locked environment is bundled. Keep Python runners as an advanced alternative and omit empty outcome-message markers.

- Keep explanatory workflow notes in reproducibility packages, with the same folder hiding and optional filename anonymisation. Explain exclusions and sharing checks in plain language. Images, intermediate results, output data and previews remain excluded; review the retained notes for sensitive text before sharing.

- Simplify Batch **Run & results** to one toolbar row for **Resume saved run…**, **Output folder**, and **Refresh file status**. Move **Export reproducibility package…** into the run report card in place of **Find manifest JSON**, with a concise contents/exclusions explanation. After a finished run, the results footer offers **Export package…** to open the same review; **Items & outputs** and **Overrides** retain **View run report**, and **Setup** keeps **Check batch** for a new run. No entry point exports without review.

- Add **Find Label Boundaries** to Label Operations for QC overlays and saveable Boolean boundary masks from labels or Boolean masks. Choose inside objects (default), outside objects, or both sides, with explicit 2D/3D processing and Face/Full connectivity. It preserves the image grid, calibration and upstream labels, processes time/channel blocks independently on CPU, and does not invent background beyond the image edge. Outside placement also marks touching-label interfaces; boundary masks do not retain object IDs or create meshes.

- Add a compact **Filter result** inspector section, after the input/object histogram, showing exact input, kept and removed object counts for Remove Small Objects, Clear Border Objects, Filter Labels by Volume and Filter Labels by Property. Counts use the current calculated input/output, respect processing blocks and mask connectivity, and run in a cached background diagnostic. Stale, bypassed and uncached results do not display misleading counts; no segmentation settings or pixels are changed.

- Add **Resume saved run…** to Batch workflow and `--resume MANIFEST` to newly exported batch runners. Continue the saved workflow/settings without altering the open graph, reusing only whole completed items with verified input/output contents, effective parameters, runtime and checkpoint evidence. Schema 6 manifests preserve run lineage and distinguish verified reuse from new writes. Older or incompatible evidence and unverified existing outputs are refused; Skip existing remains separate. Normal and resumed runs share a crash-released destination lock.

- Use VIPP icons for installed Windows shortcuts and the installed desktop application's running window/taskbar (macOS keeps its branded app bundle). Plugin/manual launches retain napari branding. Remove the startup window's native title bar while keeping it draggable and non-topmost, with a small top-right minimize button and position preserved when restored. Initially allocate about two-thirds of the available height to the workflow dock, leaving room for napari and allowing normal resizing afterward.

- Add **Separate Overlapping Objects** under **Segmentation & Labels**: an annotated, authored workflow combining intensity segmentation, XOR/OR mask reconstruction and two separately measured, coloured meshes. Preserve its parameters and layout; disabled Save Image nodes keep calculation free of output files.

- Separate **Batch workflow** and **Display settings** with a matching vertical toolbar divider, including compact layouts.

- Keep one **Mesh Objects, Colours & Refinement** example, using the saved interactive workflow unchanged. Remove the duplicate example and its generator while preserving the authored parameters, layout and inspector settings.

### Broader Richardson-Lucy GPU execution

- Allow ordinary Richardson-Lucy GPU execution for 1–500 iterations and filter
  epsilon 0–1, and RL-TV for 1–100 iterations, regularization 0–0.1, filter
  epsilon 0–0.001, TV epsilon 1e-12–0.01 and denominator floor 1e-6–1.
  Even-sized/larger PSFs and authored normalization, clipping and scale options
  also remain eligible. No CPU comparison is required before running these
  settings. Finite float32 inputs, valid axes, positive PSF mass, environment,
  memory and cancellation requirements remain enforced.
- Show non-blocking numerical-difference advisories in the selected node's
  Compute section and retain them in execution provenance version 2. Advisories
  describe uncertainty, never CPU equivalence; CPU fallback clears GPU result
  advisories. Benchmark/optimizer comparison criteria remain separate from
  permission to execute on GPU.

## 0.15.0a2 - 2026-09-08

### Compatibility fixes

- Preserve legacy batch-workflow hashes when restoring the additive Binary Threshold and Rescale Intensity defaults. Explicit non-default choices still change scientific workflow identity; the deterministic batch demo can be restored and previewed without a false mismatch.
- Record Binary Threshold range endpoints and inclusive/exclusive rules accurately in operation history instead of describing an inactive single cutoff.
- Keep the update dialog usable at narrow widths and larger system fonts: installer and checksum buttons stack when needed instead of forcing horizontal scrolling.

### Mesh creation, object management and export

- Add **Mesh Objects — Tuned Refinement** under **Open example → 3D Meshes**, captured from an interactive workflow with its authored parameters, layout and inspector settings. It demonstrates Turbo triangle-count colours, two smoothing iterations at strength 1, and 10% simplification at aggressiveness 4 without automatic file output.
- Extend **Save Image** to write meshes as OBJ or 3MF without converting geometry to an image. Its output retains mesh identity, colours and calibration for downstream nodes. **Save Image** and **Batch Output** now list formats for the connected image, mesh or batch table; incompatible saved selections require an explicit replacement. Overwrite safeguards and batch restrictions on auto-saving nodes remain in place.
- Make mesh-refinement failures actionable: distinguish invalid input from defects introduced by smoothing or simplification, identify the object/component and requested settings, and suggest the relevant controls to adjust. Duplicate-face errors include the count; rejected results do not replace the input or silently repair geometry.
- Audit all numeric node parameters for decimal precision: **Simplify Mesh** now retains fractional percentages and aggressiveness, and **Filter Mesh Objects** declares fractional limits for shared editors and batch overrides. Catalog-wide regression tests cover defaults, limits, steps and resets; integer and odd-only controls retain their constraints.
- Right-click a workflow tab to **Open in File Explorer** (Windows), **Open in Finder** (macOS), or **Open containing folder** (Linux). The action locates that tab's saved workflow without switching tabs or saving pending changes; it is disabled until the workflow has a file path.
- Fix **Smooth Mesh → Strength** rounding to 0 or 1: the slider and numeric entry now accept 0.01 steps and show the default 0.5 correctly. A tooltip explains per-iteration smoothing and possible changes to shape and volume.
- Add an input-object histogram to **Filter Mesh Objects**, with the selected measurement's units, hover bin counts and linked minimum/maximum markers. Background measurements use the cached input mesh and match the filter; unavailable measurements are excluded and reported. The plot follows the data range instead of the default `1e12` upper bound, and logarithmic display does not change filtering. Physical units follow the mesh's X-axis calibration, including non-micrometer units; uncalibrated mesh area/volume are geometric voxel-coordinate measures, not mask voxel counts.

- Add **3D Meshes** beside Morphology, grouped into surface creation, object controls and geometry refinement. **Labels to 3D Mesh** retains label IDs; **Mask to 3D Mesh** can identify disconnected foreground objects. **Colour Mesh Objects** uses distinct object colours or volume, surface area, sphericity or triangle count. **Combine Meshes** preserves separate objects, colours and compatible calibration without geometric union. **Split Mesh Objects** separates disconnected surfaces; **Filter Mesh Objects** selects by size, measurement or ID.
- Add explicit **Smooth Mesh** and **Simplify Mesh** nodes with boundary controls and recorded settings. They create new geometry; their source meshes remain unchanged. Measurements use the actual resulting surface, one row per object. Native napari surfaces display the same object colours, including after geometry changes. A bundled **Mesh objects, colours and refinement** example demonstrates the workflow.
- Add **3MF** saving and Batch Output support for physical units, separate objects and colours. Scientific IDs, calibration and processing history are embedded in the file; OBJ remains available for geometry and object groups. Uncalibrated meshes require physical calibration for 3MF—VIPP never guesses a printing scale. Neither export claims that a surface is repaired or print-ready.

- Fix **Measure 3D Mesh Morphology** rejecting an already-calculated mesh during compute preflight. Cached meshes and Boolean masks now correctly use CPU measurement under Auto or Prefer GPU, without rebuilding existing geometry.

- Hide empty thumbnail placeholders on table- and mesh-only graph nodes, including before their first calculation and while bypassed. Bypassed mesh nodes no longer request image-only what-if previews. **Measure 3D Mesh Morphology** now also accepts Boolean masks and existing meshes: supplied triangles are measured directly with calibration, one row per object, no remeshing or invented voxel counts. Open/invalid surfaces retain area and extent measurements but report unavailable volume-derived fields.

- Add **Mask to 3D Mesh**: manual/cached, full-resolution Lewiner marching cubes for one explicitly identified Boolean Z/Y/X volume. Choose whether to close foreground at the image border, inspect/pin a native napari Surface, and export calibrated geometry. Extraction never silently smooths or simplifies the surface.

### Intensity, threshold and display controls

- Rescale Intensity now keeps **Output min ≤ Output max** and has an explicit **Invert intensity** checkbox. Inversion reverses the cutoff-to-output mapping without changing the bounds, and is retained in history, saved workflows, batch runs and Python exports. Old reversed bounds remain authored but require explicit correction before calculation; they are not silently swapped.

- Add an explicit **Channel display** choice to Image Source: **Stack (C slider)** or **Separate coloured layers**. It applies immediately and is saved as workflow presentation metadata. Colour edits no longer silently remove the source's C slider; image pixels, axes and processing are unchanged.

- Binary Threshold adds **In range** and **Outside range** foreground choices with linked low/high sliders and draggable histogram limits. In range includes both endpoints; Outside range excludes them, and NaN remains background. CPU, supported CuPy execution, saved workflows, batch runs, and Python exports share these rules; Above/Below behavior is unchanged.
- Clarify that graph contrast settings refresh automatically without Calculate, and explain why different choices can look identical when their contrast ranges coincide. Pixel-level checks cover immediate refresh, the startup example, and unchanged scientific results.
### Readers, updates and workspace improvements

- Move Image Source reader diagnostics into a separate, collapsible **Reader support** inspector section marked **System information**, available on every Image Source, including new and live-layer/sample nodes. Checks and cached results are shared for the VIPP session, with explicit rechecking. Format names stay bold in the normal text colour; status text shows green when loading succeeds, yellow for missing/outdated support, and red for failed loading. Expanded statuses use the inspector's full scroll area; file-specific install/retry actions remain beside Source.
- Fix 3D inspection of interleaved microscope axes such as Olympus OIR's ZCYX: render spatial Z/Y/X and keep channels as a working slider. Viewer ordering changes only presentation, preserving pixel storage, calibration, and linked VIPP/napari channel navigation.
- Make the update dialog content-sized, keep its controls together, and show the official VIPP wordmark; longer guidance remains scrollable on small screens.
- Keep all Graph display settings dropdowns the same width, including after host font styling, without clipping the longer thumbnail-view option.
- Include native CZI, Leica LIF/LOF/XLIF, Nikon ND2 (with legacy codecs), and Olympus OIF/OIB/OIR readers in ordinary plugin and managed desktop installations. Java/Bio-Formats remains optional.
- Add a collapsed **Reader support** section to Image Source, with isolated reader checks, in-context missing-reader setup and retry actions. Explicit setup reviews hash-verified packages and waits for VIPP/napari to close; it adds missing packages without replacing the installed scientific stack.

- Binary Threshold now offers **Foreground: Above / Below** for bright or dark foreground. Both comparisons are strict: equal values and NaN stay background. Existing workflows retain Above, and the choice is preserved in workflow, batch, generated Python, and supported GPU execution.
- Compact inspector histograms show cached counts on hover, with size/property ranges where available, including log-size distributions. Tiny and empty bars are reachable across their bin column; hover pauses during marker edits and when a narrow plot combines bins for drawing. Hovering never rescans image data or changes analysis.
- Node searches recognize common alternative names, verb forms, abbreviations, and British spellings (such as dilate/erode, thinning, NLM, clipping, and normalisation) across the palette, insert-node picker, and Find in workflow. Node names and saved workflows are unchanged.
- Added **Convex Hull** under Morphology: form one Boolean hull per 2D YX slice or 3D ZYX volume, with axis-aware automatic selection and independent time/channel blocks. Sparse, line and flat 3D masks are supported; separate foreground objects can be joined.
- Fixed numeric controls that could author invalid node parameters: odd-only filter/window and PSF sizes, linked low/high limits, bounded percentages and counts, and positive minima at very small scales. Editing one bound does not move its partner; legitimate intensity inversion and label-filter sentinel values remain available.

- Added quiet, daily GitHub update checks: the version badge highlights new releases without opening a dialog. Click for update/download guidance, or right-click for release notes and a manual check. Automatic checks and prerelease inclusion are configurable; running environments are never modified automatically.

- Consolidated public guidance in the vipp-mkdocs manual and replaced local user guides with topic links. Planning, architecture, execution contracts, and validation evidence remain beside the application code.

- Added full-contrast magnifiers inside workflow and node-library search fields. Workflow search has more room on wide windows and supports Ctrl+F when napari or another host shortcut has not claimed it.
- Centred the Settings menu's section headings with balanced margins and clearer separation from the left-aligned commands.
- Renamed the toolbar's Preview menu to **Display settings** (**Display** at narrower widths), with clearer graph-display labels and expanded option names. Contrast scope uses **Current slice**, **Current projection**, or **Current image** as appropriate; **Entire stack** is unavailable when no available graph preview contains a stack or series. Existing display preferences and scientific processing remain unchanged.

## 0.15.0a1 - 2026-09-06

### Workflow workspace and node library

- Rebuilt the node library around a compact category-icon rail and category popups, with operation counts, clearer icons and tooltips, and accessible labels. Global search spans the library and restores the previous category expansion state when cleared.
- Reorganized the main toolbar into grouped workflow commands, **Preview**, compute controls, graph navigation, and Settings, with responsive alternatives for narrower windows and a persistent status/activity footer.
- Dropping a compatible library node onto a terminal output now appends and automatically connects it in one undoable edit.
- **Image Source** accepts local image-file drops directly on its graph card and file or pixel paste with Ctrl+V/Cmd+V. File-backed sources retain their metadata; pasted colour pixels retain explicit RGB/RGBA semantics.
- Added total and current-stage elapsed timers to **Find fastest**, independent of worker progress updates. The dialog explains that some CPU calls report only on completion and that time limits/cancellation wait for safe checkpoints; the timers do not claim measurable progress within an opaque library call.

### Operation-aware inspector

- Rebuilt the responsive, theme-aware inspector around operation-specific profiles, with consistent spacing, persistent labels, and controls appropriate to each operation rather than one generic parameter layout.
- Connected-input summaries now explain scientific roles as well as connections. Deconvolution distinguishes the observed image from the point-spread function, while **Calculate New Image** shows the weighted equation being constructed.
- Added more relevant diagnostic views, including object-size and measurement distributions, alongside image histograms and operation-specific guidance.
- Background diagnostics and lightweight presentation refreshes keep graph dragging and existing threshold-guide interactions responsive, and preserve the inspector's scroll position instead of repeatedly rebuilding heavy content.
- Added a seven-lane synthetic inspector showcase covering the current node palette and deterministic threshold phantoms for repeatable UI review.

### Plots, analysis, and measurement

- Extended **Normalize** with four modes: robust z-score using median/MAD, maximum-absolute scaling, reference z-score using a saved mean and standard deviation, and percentile scaling to 0–1. Existing min-max and z-score calculations retain their behaviour; fitted statistics apply across the supplied array, so split channels first when independent channel normalization is required.
- Added the **Intensity Histogram** analysis node for reproducible scalar or multichannel distributions with shared bin edges, explicit limits, linear or logarithmic bin spacing, counts, fractions, densities, and cumulative values.
- Intensity Histogram records non-finite exclusions, custom-range underflow/overflow, and non-positive logarithmic exclusions in metadata. Its inspector and pop-out reuse the calculated table rather than rescanning source pixels for display changes.
- Added resizable, sortable result-table windows with units, clearer non-finite/missing-value diagnostics, and background CSV/TSV export. Sorting changes the view only; export preserves the workflow's original row order and values.
- Improved histogram and colocalization pop-outs, including channel-aware legends, less opaque overlapping histogram bars, zero-inclusive shared scatter axes, equal-axis and populated-data zoom controls, and clearer axis titles. Scatter review is available directly from colocalization metric nodes.
- Detached scatter plots support up to 4096 bins per axis with background, memory-gated calculation. Compact and interactive estimates use bounded mass-preserving derivatives; display-only changes do not replace scientific calculations.
- Reduced CPU measurement work through tight per-object mesh regions, cropped skeleton component graphs, and spatial-indexed nearest-object distances. These optimizations preserve physical calibration, global coordinates, result ordering, and deterministic distance ties; sparse label IDs no longer require mesh preparation proportional to the largest ID.
- Added GPU-assisted **Measure 3D Mesh Morphology** for eligible non-negative int32 3D labels. GPU label preparation feeds the authoritative CPU marching-cubes and convex-hull finalizer; this is a hybrid implementation, not an all-GPU mesh algorithm.
- Added GPU **Analyze Skeleton** measurement for eligible already-skeletonized boolean 2D/3D inputs, retaining the CPU reference's voxel-graph and physical calibration rules. This does not add GPU skeleton thinning.
- Extended compute support, memory admission, cancellation/progress, and table finalization contracts for those measurement providers. Proven label-array facts now survive validated label filtering, avoiding unnecessary rescans when deciding downstream GPU eligibility.

### Batch workflow

- Rebuilt the batch window around four task-based tabs: **Setup**, **Items & outputs**, **Overrides**, and **Run & results**, with a persistent activity strip and a context-appropriate next action. Setup brings workflow identity, source pairing, destination, and run policy together.
- Source discovery appears immediately in the item list, followed by background metadata and exact-content checks with per-file activity and checked-file counts. **Check batch** validates a plan without calculating the representative image or saving outputs; **Preview selected** remains an explicit, optional graph calculation.
- Added actionable recheck warnings and clearer distinctions between a full batch check, a selected-item recheck, and the read-only **Refresh file status** action. The main graph's simplified representative controls provide a direct route back to inspect that sample in the batch window.
- Added per-item **Keep existing outputs**, **Rerun and overwrite outputs**, and **Use batch default** context-menu choices. Individual choices can coexist within one batch and are saved against the exact source pairing and destination, not a row number. Generated batch runners preserve the same policies.
- Changing only the existing-file policy updates the checked plan without repeating source inspection. Keeping existing files preserves outputs already present while still creating missing outputs; planned, existing, overwrite, and unresolved-decision counts now reflect that intent.
- Reworked the parameter override matrix with persistent field labels, centered values, readable node/parameter/default headers, explicit page and matching-sample selection, and separate reset-selected and reset-all actions. The resizable **Edit selected** editor guides bulk changes: only checked parameters change, and other overrides remain intact.
- Integrated the existing whole-batch Run/Bypass controls into a matching collapsible section. Distinct colours identify forced Run and Bypass choices, and the Overrides page scrolls as a whole.
- Planned-output details now lead with the producing node, output kind, and format; exact file paths are secondary. Refined table typography, spacing, column widths, and link hit areas: selecting a result row reveals its outputs, while clicking its underlined name explicitly navigates to item review. Source and output reveal actions use platform-appropriate labels and icons.
- Added preparation feedback, elapsed time, per-sample node counts, and friendly node names during execution. Progress labels reserve two bottom-aligned lines to prevent ordinary wrapping from shifting the tables.
- Reused the freshly validated Run plan at the worker handoff, removing an unnecessary second full collection scan. Exact source-content validation before use and destination checks remain in place.
- Added an inline, human-readable final run report with elapsed time, completed/kept/failed/cancelled outcomes, output counts, and expandable failure reasons. The manifest JSON remains a separately labelled technical provenance artifact. Completed runs stay in report review; reviewing Setup is separate from explicitly checking and starting another run.

### Important fixes

- Corrected mixed CPU/GPU execution planning where a CPU operation on another branch could incorrectly reunite GPU segments and report an execution-unit cycle in an otherwise valid workflow.
- Improved cache reuse across **Calculate all**, bypass, and source-loading transitions while preserving scientific invalidation. Calculate all can retain an in-progress upstream calculation, composite edits preserve compatible manual deconvolution results, and napari reslicing no longer masquerades as a source-pixel edit.
- Prevented failed replacement-source calculations from presenting unrelated cached downstream pixels as current results. Provenance-compatible completed boundaries and unaffected source branches remain available. Presentation errors also no longer leave the application indefinitely reporting Processing.
- Avoided duplicate CZI container opens by reading metadata and pixels within one container lifetime, while retaining source-mutation checks and saved item identity.
- Corrected mixed-rank T/C navigation, hidden Crop/Inspect layer lifetime, scalar images incorrectly inferred as RGB, and encoded-colour axis validation. Axis reductions and slices retain appropriate channel metadata; rendering Select/Reorder controls no longer silently changes saved parameters.
- Corrected **Combine Channels** colour invalidation and saving so cached thumbnails, histograms, Inspect/Pinned layers, downstream colour composites, and reopened workflows consistently reflect authored colours without unnecessary pixel restacking.
- Fixed object-association pair counting for large unsigned label IDs, including values beyond the signed 64-bit range, without changing their identity through integer conversion.
- Kept detached plots associated with their originating node and workflow after selection changes, including threshold edits and stale-result warnings. Late background table sorts can no longer replace newer results, exports retain the selected result snapshot, and themed histogram/table startup is corrected.
- Corrected batch cancellation handling so a cooperative stop is not mistaken for a CPU/GPU cleanup failure. Genuine unverified cleanup still blocks further calculation until recovery.

### Compatibility and remaining limits

- This is an alpha release. Preserve original data and copies of workflows/configurations before resaving, and revalidate analyses affected by the changed controls or scientific contracts.
- Batch configuration version 6 records individual file decisions and continues to read versions 1–5; older VIPP releases supporting only version 5 cannot read the new configurations. Workflow version 6 and manifest version 5 remain unchanged.
- Replaced the public **ImageJ Auto Threshold (8-bit)** method selector with a fixed **ImageJ Default Threshold (8-bit)** node. Previously saved ImageJ Triangle nodes retain that separate calculation as fixed legacy compatibility, not a silent conversion to Default or generic Triangle. The implementation remains experimental and source-aligned to ImageJ 1.54p; independent ImageJ-generated golden parity is not claimed. Generic Triangle and Isodata retain their distinct contracts.
- Clarified **Minimum Threshold** as a valley-between-peaks method. Its histogram smoothing pass limit is a convergence safety bound, not an image-blurring strength, and unsuccessful convergence remains an explicit error. Renamed **Clip** to **Clamp Intensity**, with explicit bound semantics and no intended change to its calculation.
- The older standalone colocalization scatter-raster nodes are no longer offered in the node library, but remain loadable and executable in saved workflows; metric-node plot review is the current interface.
- Check and Run validate exact source contents, so large containers or slow storage can still take time. A timer or busy indicator shows application activity, not proof that a non-cooperative CPU/GPU operation is making numerical progress.
- Cancellation is cooperative; an active kernel, library call, or output write may need to reach a safe boundary. Existing files kept by policy are not claimed as newly calculated or scientifically verified by the current run.
- Most workflow operations still materialize inputs. General lazy/chunked graph execution is not introduced; exact source-window pushdown retains its previously documented direct local OME-Zarr Crop Stack limits.
- GPU acceleration remains operation-, dtype-, shape-, memory-, and environment-dependent. The optional Windows installer is explicitly unsigned; native macOS packages are explicitly unsigned, unnotarized, CPU-only, and architecture-specific.

## 0.14.0a3 - 2026-08-29

### Features

- Completed responsive volumetric Crop Stack support for issue
  [#50](https://github.com/rensutheart/napari-vipp/issues/50). Persisted Z-start
  and Z-end margins apply only to one explicitly declared Z spatial axis; old
  workflows retain zero-Z behavior, inferred QYX is never promoted to Z, and
  T/C axes, physical calibration, origins, history, workflow hashes, export,
  generated execution, and batch behavior remain reproducible.
- Crop controls now present a lightweight translucent ROI over cached pixels
  while the user drags. Release commits one undoable scientific edit and one
  calculation; a pause while the mouse remains held is never an Undo boundary.
  Pending drafts are flushed before
  calculation, save, export, batch, workflow-tab, and close boundaries. The
  bundled `responsive-crop` example provides a numbered TCZYX acceptance path.
- Crop Stack now hides its channel-axis override when explicit metadata already
  identifies the preserved channel axis, and also for explicitly scalar data.
  The fallback remains visible for unresolved axes and saved manual overrides,
  with wording that makes clear it protects an axis rather than cropping
  channels.
- Refined the Crop ROI overlay for each napari display mode: a thinner
  current-plane guide in 2D and a transparent, depth-independent wireframe in
  3D, avoiding the thick embedded-looking faces seen against rendered volumes.
- Added a separate presentation-only outline-thickness control so Crop ROI
  guides remain legible without overwhelming low-resolution data.
- Added conventional workflow saving with `Ctrl+S`. The toolbar save action now
  overwrites the current workflow by default, while first saves and `Save as`
  request a destination and confirm before replacing an existing file. A
  persistent Settings choice can instead confirm every overwrite or create a
  separately timestamped workflow on every save. Bundled examples remain
  templates, so saving one never overwrites the packaged example.
- Added topology-aware Safe Node Bypass across callable, fixed-single-output
  processing nodes. Bypass forwards the exact first/main input while preserving
  the authored node and its settings; multi-input restoration nodes therefore
  forward Image/intensity rather than PSF. Sources, writers, unused terminal
  nodes, dynamic/multi-output operations, and type-incompatible splices fail
  closed. Bypassed cards retain a presentation-only would-run thumbnail, a
  faded dotted treatment, badge, and pass-through cue without exposing those
  hypothetical pixels to downstream analysis.
- Added matching whole-batch **Use workflow / Run / Bypass** profiles. Profiles
  are applied atomically to detached effective workflows, remain separate from
  authored graph intent, and are recorded in saved configuration, generated
  runners, manifests, hashes, and execution provenance. Workflow schema 6 and
  batch config/manifest schema 5 carry the new intent while continuing to read
  their supported earlier schema versions.
- Added exact level-0 source-window pushdown for a direct **Image Source → Crop Stack** path. Supported local OME-Zarr sources now read only the proven crop window while retaining complete T/C axes, exact source revision and crop identity, full-resolution scientific semantics, and fail-closed fallback when graph topology, axes, reader evidence, or source bytes do not match the verified plan.
- When an eligible complete source exceeds the safe RAM budget, Image Source can offer an explicit centred fitted Crop Stack as a reviewable starting point. VIPP never crops silently, preserves every T/C position, and retains ordinary actionable memory refusal when an exact safe read cannot be proved.
- Added a bounded napari compatibility lane covering the retained minimum
  `napari==0.6.0`, exact `napari==0.9.0`, and the latest supported napari. It
  validates the plugin manifest and exercises import, start/close, and focused
  viewer integration with the platform bindings VIPP distributes: PyQt6 on
  Windows/Linux and PySide6 on macOS. The declared `napari>=0.6` range remains
  unchanged.

### Bug Fixes

- Hardened Crop selection and presentation-preview updates against napari layer-model re-entrancy, preventing the crash previously triggered by selecting a changed Crop Stack after switching compute preference.
- Updated generated Image and Labels presentation layers to carry VIPP's axis
  names into napari when that layer API is available. Labels stay aligned to
  displayed dimensionality, omit a trailing RGB/RGBA component axis, update on
  in-place layer reuse, and follow the selected scientific presentation layer;
  hidden or provisional source previews cannot replace the active viewer
  labels.
- Replaced documentation capture's direct use of napari's legacy camera and Qt
  window layout with feature-detected compatibility seams. Camera access
  prefers `viewer.scene.camera` and falls back to `viewer.camera`; native-window
  lookup prefers ordinary Qt parent traversal and retains only a bounded legacy
  fallback for older supported napari releases. Both paths are binding-neutral
  across PySide6 and PyQt6.
- Advanced Image Source axis text now remains an uncommitted editor draft until
  it is a complete valid mapping and the user presses Enter or leaves the field.
  Partial keystrokes can no longer enter workflow history or trip strict source
  validation during an otherwise valid edit.
- The napari compatibility work is presentation-only: it changes no workflow
  schema, scientific operation, generated result, provenance, or GPU contract.
- Undo and Redo now restore a single node's parameter change in
  place, retain graph cards, thumbnails, viewer layers, and unaffected cached
  branches, and recalculate only that node and its descendants. Image Source
  edits also use this path while still refreshing source-owned I/O and preview
  state. Topology, layout, compute-policy, and multi-node changes retain the
  audited whole-workflow restore path.
- Slider, axis-range, histogram-marker, and colocalization-threshold scrubbing
  now creates one Undo step per completed press-drag-release gesture, even when
  a recalculation or idle timer finishes while the pointer remains held. Undo
  returns directly to the value present at mouse-down rather than a calculated
  intermediate value.
- `Ctrl+S` is now caught at the active VIPP/napari window boundary, so graph,
  inspector, and viewer focus cannot prevent workflow saving. VIPP no longer
  registers a second Qt action shortcut that can collide with napari's own
  `Ctrl+S`; the window-level handler claims the key before napari resolves its
  actions. A successful save clears the tab's dirty asterisk and reports
  `Saved workflow` in the status strip.

### Remaining limitations

- Exact source-window pushdown is deliberately limited to one direct Image Source to Crop Stack path and readers that can prove an exact local level-0 window. Other graph shapes and formats still use the ordinary full-source load path.
- The macOS installers remain unsigned, unnotarized, CPU-only, and current-user-only. macOS may require the documented **Open Anyway** confirmation after checksum verification.

## 0.14.0a2 - 2026-08-27

### Features

- Added separate native macOS installers for Apple Silicon and Intel. Each
  offline, CPU-only package installs its own managed Python environment for the
  current user and creates a normal `VIPP.app`, so Python and command-line
  setup are not required.
- Added exact-wheel macOS packaging, architecture-specific checksum and build
  records, and native Cocoa install-and-launch checks. The macOS package joins
  the Windows executable as an explicitly unsigned alpha convenience artifact.

### Bug Fixes

- Detached VIPP windows now release stale width and height constraints across
  the complete Qt dock-widget chain, so they can be maximized or resized freely
  in either direction. Reattaching the window restores napari's original dock
  constraints and size policies.
- Corrected PySide6 compatibility in graph dialogs, rendered colocalization
  previews, settings submenus, dock callbacks, and queued thumbnail work while
  retaining PyQt6 behavior. Callbacks now avoid Qt objects whose native owners
  have already been destroyed, and menu wrappers remain alive for as long as
  their actions are displayed.

### Remaining limitations

- The macOS installers are unsigned and not notarized. macOS may require the
  documented **Open Anyway** confirmation after checksum verification.
- macOS installation is CPU-only and current-user-only. Automatic update and a
  graphical uninstaller are not included in this alpha.

## 0.14.0a1 - 2026-08-26

### Features

- Added a durable SourceItem v1 record for selecting scientific images inside
  multi-item containers. Workflow schema 5 and batch config/manifest schema 4
  carry the same stable selector, reader evidence, source revision, axes, and
  metadata through interactive, batch, generated, replay, export, checkpoint,
  and provenance paths while retaining deterministic legacy migrations.
- Added truthful reader inspection and typed source failures for the frozen
  public microscope corpus. Inspection/read metadata now agree for the claimed
  ND2, LIF, CZI, OIR/OIB, VSI/IMS, and LSM routes; OIB pixels and ImageState
  agree on CZYX; multifile VSI/ETS and OIF companion trees participate in the
  exact source revision; and optional Java/Bio-Formats failures are actionable.
- Corrected the frozen Leica LIF contract to preserve its semantic Alexa dye
  names rather than confusing them with display LUT colours, and normalize
  combined objective labels such as `63x, 1.3NA` from Imaris metadata.
- Added a local OME-Zarr 0.4/0.5 presentation preview that enumerates declared
  levels and transforms, slices requested axes and regions before computing,
  preserves label semantics, reports observable I/O where available, and keeps
  scientific analysis fixed to level 0.
- Added reviewed numeric per-SourceItem batch overrides. Blank cells visibly
  inherit the saved workflow value, typed preflight rejects stale or ambiguous
  rows, and effective values/hashes are consistent across preview, batch,
  generated execution, checkpoints, manifests, and provenance without
  modifying the base workflow.
- Added source-load memory preflight, truthful progress, cooperative
  cancellation, and stale-generation protection for large local reads.
- Improved the retained Batch workspace: attached configurations redetect their
  samples in the background, effective per-sample values appear only where an
  override applies, compact activity remains visible beside the toolbar, and
  ordinary existing-output collisions offer a safe one-run overwrite prompt.
  Cancel is the default; duplicate destinations, input overlaps, and explicitly
  protected outputs remain hard errors.
- Completed installer issue
  [#42](https://github.com/rensutheart/napari-vipp/issues/42): the reviewed setup
  now shows separate approximate download, installed, and peak working sizes
  for CPU and CUDA; preserves the distinct enforced installation/temp-drive
  minimums without presenting them as VRAM; names each phase and elapsed time;
  uses determinate byte progress only when trustworthy totals exist; keeps a
  quiet-work heartbeat and latest concrete activity visible; and exposes the
  setup log from Advanced details so slow, stalled, and failed states are not
  conflated.

### Remaining limitations

- Scientific graph execution still materializes the selected full-resolution
  level-0 image; the OME-Zarr lower level is presentation-only.
- Preview support is limited to local OME-Zarr 0.4/0.5 image and label groups.
  Remote stores, HCS plate/well/field traversal, IMS pyramid claims, and
  operation-level lazy/chunked execution remain deferred.
- Native LIF, CZI, OIR, OIB, and LSM pixel access is eager. VIPP records that
  capability rather than presenting it as lazy.
- VIPP pins the selected reader/backend and refuses an unreviewed topology
  change; it does not infer equivalence between `liffile`'s combined PR2729
  view and Bio-Formats' four logical views.

## 0.13.0a9 - 2026-08-25

### Features

- Skeletonize now makes volumetric processing explicit and verifiable. Auto
  selects Lee thinning for declared ZYX data, leading dimensions are processed
  as independent ZYX blocks, Zhang is rejected for 3D, and the resolved method
  and dimensionality are preserved in provenance. Ambiguous page axes fail
  closed until their spatial meaning is declared.
- GPU VRAM admission failures now identify the CUDA device and every affected
  graph node, format the estimated peak, available memory, and shortfall in
  readable units, explain whether free VRAM/reserve or the configured cap is
  binding, retain exact byte counts as structured diagnostics, and suggest
  concrete ways to make the graph fit.

### Bug Fixes

- Prefer GPU planning now carries exact shape, dtype, and axis facts through
  CPU-only nodes such as Rescale Axes, Rescale Intensity, and Unsharp Mask.
  Reviewed downstream CuPy and CuPyX providers therefore remain eligible after
  a host boundary instead of being deferred to CPU because their inputs appear
  unresolved.
- Changing Image Source from QYX to ZYX now immediately projects the effective
  axes through the selected branch, so Gaussian Blur 3D exposes Sigma Z without
  waiting for a successful full-graph pixel run. Unrelated source or execution
  failures cannot force the inspector back to stale QYX metadata, and reverting
  the declaration hides—without deleting—the stored Z parameter.
- Re-materializing the same file revision no longer changes its source identity
  merely because its array wrapper is new. Sequential sibling measurements can
  therefore remain ready together and feed Merge Tables in either execution
  order, including after Low-memory cache pruning; genuine upstream changes
  still invalidate the affected descendants.

## 0.13.0a8 - 2026-08-22

### Features

- Replaced the optional cuCIM-backed basic object-measurement providers with
  resident CuPy implementations available through the normal CUDA
  installation. Exact saved `cucim-measure-objects-basic-v1` and
  `cucim-measure-objects-intensity-basic-v1` pins migrate to the corresponding
  `cupy-*` IDs, while an ambiguous broad `library:cucim` preference remains
  visibly unavailable. The separate cuCIM source-build bundle, installer, and
  hosted canary are retired; historical implementation and benchmark evidence
  remain preserved as a dated record.
- Added **Remove Outliers (Binary)**, a source-aligned ImageJ/Fiji mask-cleanup
  node for removing foreground specks or filling background notches. It uses
  ImageJ's exact circular-footprint construction and nearest-edge behavior,
  processes each trailing YX plane independently with progress/cancellation,
  and accepts only boolean or canonical uint8 0/1 or 0/255 masks. The
  authoritative CPU implementation is paired with an exact, resident CuPy
  candidate whose fixed kernel takes radius and polarity at runtime, so
  parameter sweeps do not create radius-specific compilation pauses and Find
  Fastest can measure CPU against GPU for the active mask.
- Compute Setup now lists qualified accelerators and lets each workflow tab
  choose Automatic or one exact runtime/device for that machine. The choice is
  carried through scientific execution and asynchronous thumbnail statistics,
  remains independent between tabs, and is deliberately excluded from
  workflow files and undo history.
- Image Source now shares the batch workspace's reviewed **Image stack**
  control. A saved `QYX -> ZYX` choice reinterprets TIFF pages as depth without
  transposing pixels, is preserved by workflow undo/save/load, export, and
  headless execution, and makes Rescale Axes expose its Z scale and Z output
  size. Batch source rows inherit that choice but retain an explicit
  file-labels-unchanged override.

### Bug Fixes

- The exact CuPy 14.1.1 `cupyx.jit.rawkernel is experimental` import notice is
  no longer surfaced as an end-user napari warning. VIPP filters only that
  known upstream `FutureWarning` around its lazy CuPyX signal import; other
  warnings and all import/execution failures remain visible, and the pinned
  behavior is recorded in the developer notes for upgrade review.
- **Rescale Axes** now accepts unique carried Y/X names even when their
  confidence came from shape inference. The inspector warns before relying on
  those names, the reviewed inferred X/Y plane is recorded explicitly in output
  provenance, and unrelated Q/T/C axes remain untouched. Inferred or missing Z
  still requires an explicit axis declaration before Z can be resized, so a
  generic QYX TIFF is never silently treated as ZYX. True no-op rescaling is
  allowed without changing axis confidence, and batch preflight now predicts
  the same resized shape and metadata as real execution.
- Dropping a completely disconnected node onto a compatible green-highlighted
  wire now inserts it between the wire's source and target. The splice keeps
  exact source/target ports, rolls back atomically on failure, and is restored
  as one undo/redo action; dropping in open space remains a layout-only move.
- Every numeric **Intensity & Contrast** node now shows the shared exact input
  histogram as well as its output histogram. Linear Scale + Offset, Gamma
  Correction, and Normalize gain the same slice/stack, cache, cancellation,
  and stale-result behavior already used by Rescale Intensity and Clip, while
  only cutoff-bearing nodes expose draggable guides.
- Clip bounds, Rescale Intensity output bounds, and Mask Image's outside value
  now use whole-number controls when their connected image has a non-boolean
  integer dtype, while floating-point and boolean images retain their
  scientifically valid fractional controls. The inspector rebuilds safely when
  the upstream dtype changes, preserves invalid legacy values for explicit
  correction, and avoids overflowing Qt controls for wide integer dtypes.
- Parameter specifications can now declare a practical slider window separately
  from their wider numeric-entry range. Sigma Filter uses this for a useful
  `0..10` Sigma-width slider while retaining direct entry through `1,000,000`.
  The same separation now keeps other extreme-range controls—such as histogram
  bins, numerical epsilons, label distances, mesh size, skeleton length, and
  calculator offsets—comfortable to drag without reducing their valid entry
  ranges.
- **Find fastest pipeline** now accepts cache-retention scopes containing
  connected or disconnected **Batch Output** and **Save Image** nodes. The
  complete requested retention set still identifies and stales proposals, but
  detached optimization and validation retain only the writer-free scientific
  subgraph, so writers cannot run or create files during analysis.
- **Find fastest pipeline** now attests the requested, planned, device-segment,
  and actually executed implementation before comparing outputs. An unavailable
  proposed backend is excluded and the remaining graph choices are solved
  again, while a genuine small numerical CPU/GPU difference is measured and
  shown for explicit user acceptance instead of being mislabeled as an
  assignment failure. Shape, dtype, non-finite classification, and larger
  differences still fail closed; reviewed differences never change provider
  parity policy or authorize cache sharing.
- Applying a clean **Find fastest pipeline** result no longer treats the fixed
  CPU Image Source row as a changed runtime assignment. Source identity is
  reconstructed independently from its authoritative declaration, while every
  executable node still requires its accepted measured backend to match.
- Graph thumbnails now keep the last complete preview visible while full-stack
  contrast statistics are recalculated. A first preview shows an explicit
  calculating state, and failed, cancelled, or superseded work cannot replace
  a valid thumbnail with a provisional or white frame.
- Thumbnail-contrast backend information now uses a concise
  CPU/GPU/fallback/cached status-bar summary. Full selection, algorithm,
  timing, byte-count, runtime/device, and fallback details remain available in
  the wrapped inspector tooltip and to assistive technology; cached limits now
  identify their producing policy instead of appearing to be a new decision
  under the current policy.
- Scientific parameter edits now stop queued or active thumbnail-only work
  immediately while retaining the normal 150 ms edit debounce. Thumbnail
  cleanup hands control back exactly once, a newer edit keeps ownership of its
  debounce window, and progressive results from an older affected calculation
  cannot briefly replace the current preview.
- Reviewed float32 GPU Gaussian blur now uses one radius-independent CUDA
  kernel. Tuning sigma to a previously unseen radius no longer triggers a new
  multi-second CuPyX kernel compilation; the scientific reflect boundary,
  output dtype, and CPU/GPU agreement contract are unchanged.
- Rolling-Ball Background and Subtract Background now use a radius-independent
  CuPy kernel. Changing to an unseen radius no longer compiles a new erosion
  program; the reviewed boundary, dtype, smoothing, and CPU/GPU agreement
  contracts are unchanged.
- Median Filter now uses a size-independent CuPy radix-selection kernel.
  Changing to an unseen supported odd size no longer compiles another filter
  program, while the reviewed reflect boundary and bitwise agreement contract
  remain unchanged.
- Prefer GPU thumbnail contrast now supports exact `float32` Percentile and
  Min-max statistics with bounded CuPy radix reductions. NaN, infinity,
  clipping, channel-axis, signed-zero, subnormal, interpolation, and degenerate
  cases match the CPU contract bit-for-bit; the inspector records the actual
  host-upload and bounded metadata transfers without implying scientific GPU
  residency.
- Warm Prefer GPU stack previews can now reuse the selected retained
  `float32` image output while it is still inside its scientific CUDA scope.
  The measured shortcut is limited to outputs of at least 128 MiB, returns
  only immutable host limits, records zero logical input upload, and leaves
  small or cold previews on the existing pre-emptible worker. Recoverable
  presentation failures cannot change the scientific result; GPU scratch
  cleanup failure remains fatal and quarantines accelerator work.
- Proactive GPU repair hints now use a structural workload contract instead of
  hashing every source byte on each parameter edit. Exact benchmark,
  qualification, optimizer, cache, and provenance identities continue to use
  the full byte/layout fingerprint.
- Repeated runs of an unchanged, revision-stable source now reuse its accepted
  exact display range and value-pattern metadata. Different file series,
  revised live sources, arbitrary mutable revision tokens, and already-exact
  current metadata cannot borrow the cached statistics.
- Downstream-only edits can now reuse the accepted exact scientific source
  context for the identical cached, read-only array under a VIPP-owned stable
  revision. The bounded provenance sidecar avoids repeatedly hashing every
  source byte while preserving the exact context fingerprint; any changed
  revision, binding, metadata, state, object, mutability, or provenance falls
  back to a fresh exact hash.
- Device plans are now bound to the exact compute request used to build them.
  A stale runtime/device request, an unreported explicit device, a mismatched
  capability snapshot, or an injected planner returning another device is
  rejected before scientific transfer or node execution.

### Validation

- Added opt-in, bounded interaction-latency reports for standard scientific
  parameter controls. Reports correlate parameter invalidation, the real
  debounce window, worker queue and delivery delays, scientific execution,
  thumbnail-statistics queue and execution, final rendering, publication, and
  discarded generations. Provider-neutral device spans retain exact runtime,
  device, transfer, synchronization, and implementation identity, while the
  tracing-disabled path remains clock-free and the reports stay out of
  workflows, scientific provenance, compute history, and policy decisions.
- Detached execution telemetry now separates graph restoration, cache and
  workload preparation, accelerator setup, runtime/library probing, planning,
  and device-plan construction before the existing transfer/operation spans.
  Partial preparation remains visible on failure or cancellation, uses the
  same monotonic clock as the UI report, and never changes scientific identity.
- Added a fresh-process UI interaction diagnostic that commits real parameter
  controls and records debounce, worker scheduling, preparation, scientific
  device work, thumbnail calculation, rendering, publication, supersession,
  parity, fallback, and cleanup. Large profiles include a deterministic
  target-node-start cancellation/reuse gate; synchronized phase timing is kept
  in a separate diagnostic run.
- Opt-in device observations now include a timed terminal memory checkpoint
  for every used runtime/device after all private execution scopes have
  cleaned up while the exact accelerator lease is still held. Evidence fails
  closed on missing or nonzero private live/reserved memory; device-wide
  provider/JIT cache bytes remain a diagnostic rather than a false cleanup
  failure.
- The final RTX 5090 interaction matrix passed the exact sample, a 108 MiB
  representative stack, a 432 MiB resident-thumbnail stack, and a separately
  synchronized diagnostic on explicit `cuda:0`. Every accepted GPU run kept
  exact CPU output parity, used no fallback, published only the newest result,
  reported zero terminal private memory, and remained reusable after the
  deterministic started-work cancellation. The 432 MiB resident path retained
  zero logical thumbnail upload. The final artifacts cover all 137 production
  package Python files plus four exact harness/policy/workflow/project anchors
  under one deterministic tree digest, distinguish explicit affinity from real
  multi-device validation, and show later Prefer-GPU large-source preparation
  falling to roughly 0.02-0.04 s from the second same-process warm edit onward
  under the fail-closed exact source-context reuse contract.
- A reconstructed pre-fix provider baseline used commit `91a05a9`, the same
  bundled Subtract Background input, and isolated empty CuPy caches. Unseen
  Ball radii took a 1.287 s median versus 0.0155 s for revisits; the current
  radius-independent provider took 0.0133 s for both and retained identical
  output hashes, directly demonstrating and removing the intermittent
  parameter-specialized compilation cliff.
- The interactive-tuning diagnostic now requests synchronized device-phase
  observations under Prefer GPU and refuses to publish incomplete GPU evidence.
  This machine-local sidecar reports transfer counts and bytes, operation and
  synchronization spans, fallback, and cleanup without treating the
  instrumented timings as portable performance or Auto-selection evidence.
- Added a manifest-locked interactive GPU parameter sweep covering all 18
  admitted implementations: 14 production-executed parameter or branch
  profiles, two dedicated RL/RL-TV profiles, and two fixed-contract
  measurement profiles. It records exact backend identity, fallback, cleanup,
  and matched first-use/revisit timing without creating a portable performance
  claim or a new per-release gate.
- Added a production-path RL/RL-TV parameter sweep for changed and revisited
  2D/3D PSF dimensions, iteration counts, and TV regularization, with an
  authoritative CPU parity run for every GPU result. The initial RTX 5090
  diagnostic found no avoidable PSF-shape stall, fallback, cleanup failure, or
  parity failure; this result remains machine-local screening evidence.
- Added float32 thumbnail-statistics calibration and transfer evidence. The
  protected RTX matrix passed 26 cases with exact parity, cancellation, and
  zero-residue cleanup; machine-local 2/32/128 MiB timing retained the existing
  512 MiB cold and 32 MiB warm Auto thresholds. A separate resident comparison
  at 2/32/128/512 MiB saved about 0.6/5.1/24.8/92.0 ms by avoiding the redundant
  full-image upload, establishing a conservative 128 MiB hook threshold. The
  protected resident matrix also passed exact parity, cancellation, alias
  ownership, transfer-accounting, and cleanup/quarantine checks.

### Fixed

- Simplified the Windows GPU installer status text to state its 15 GiB free
  disk-space requirement directly without referring to VRAM.

## 0.13.0a7 - 2026-08-14

### Release Overview

This seventh 0.13 alpha makes GPU acceleration easier to understand and apply
without silently changing a workflow's scientific settings. VIPP can now point
out when a visible, lossless dtype conversion is the only thing preventing a
node from using a reviewed GPU implementation, insert that ordinary conversion
in the correct place with one click, and undo it as one graph action.

The release also makes optimizer evidence much easier to inspect, including
when CPU and GPU timings are too close to justify changing the saved backend.
It adds a portable, reviewed GPU segmentation path through thresholding,
boolean mask cleanup, and connected components, and broadens the documented
CPU/GPU agreement region for Richardson-Lucy deconvolution. CPU remains the
portable reference and visible fallback outside every exact admitted region.

### Features Added

#### Clearer GPU Workflow Repairs

- Added a reviewed GPU-resident implementation for the existing **Convert
  Dtype** node in its lossless `uint8`/`uint16` to `float32`, **Preserve**
  region. It can stay in the same CUDA segment as a following GPU node instead
  of forcing a device-to-host-to-device round trip.
- Added a subtle **GPU tip** on nodes where input dtype is the only remaining
  GPU blocker. Selecting the node explains the exact conversion and memory
  trade-off; **Add conversion** inserts a normal visible node on the affected
  input only, with one-step Undo and no silent change to shared branches.

#### Richardson-Lucy Backend Agreement

- Versioned the ordinary RL and RL-TV CPU/GPU comparison as the explicit
  `rl-scientific-equivalence-v2` / `rl-tv-scientific-equivalence-v2` policy:
  equal shape and `float32` dtype, identical finite masks, completely finite
  and nonnegative clipped outputs, NRMSE no greater than `0.005`, and maximum
  absolute error no greater than `1e-6 + 0.005 * CPU reference peak`.
- The former `2e-6` near-identity result and maximum-ULP observations remain
  diagnostics only. The 0.5% limits establish backend agreement against VIPP's
  CPU implementation; they are not a universal image-quality threshold and do
  not validate the PSF, iteration count, recovered resolution, or biological
  interpretation.
- Expanded ordinary RL's checkpoint-backed, odd-PSF/default-safe envelope to
  finite authored `filter_epsilon` values from `1e-12` through `1e-6` and 1
  through 100 iterations; lambda-zero RL-TV inherits it. Exact-workload
  comparison still runs before optimizer selection, and VIPP never rewrites
  epsilon or iteration count to qualify a GPU call. Positive-TV runs retain
  their narrow shipped-profile points.
- Updated the bundled 3D RL/RL-TV comparison so one visible `float32` Preserve
  conversion feeds both branches without rescaling the sample. Both branches
  retain 25 iterations and `filter_epsilon=1e-12`.

#### A Connected GPU Segmentation Path

- Added reviewed Custom/Prefer-GPU implementations for **Binary Threshold**
  on scalar `float32` images and **Extract Channel** on explicitly described
  channel axes. Binary Threshold creates an exact resident boolean mask;
  Extract Channel exposes the selected channel as an allocation-sharing GPU
  view when its input is already resident.
- Prefer GPU now keeps a host-entry Extract Channel on CPU so VIPP uploads only
  the selected channel, rather than moving and retaining the complete
  multichannel source merely to perform a cheap slice. An explicit Custom GPU
  choice can still request whole-input residency.
- Added the annotated **Portable GPU Segmentation Bridge** example, which runs
  from Extract Channel through the exact Preserve conversion, Gaussian Blur,
  Binary Threshold, boolean **Remove Small Objects**, **Fill Holes**, and
  Connected Components while remaining usable with transparent CPU fallback.
- Added reviewed Custom/Prefer-GPU CuPyX implementations for boolean **Remove
  Small Objects** in resolved 2D/3D Face/Full connectivity regions and **Fill
  Holes** in the exact `max_hole_size = 0` region. Integer-label cleanup and
  bounded-hole-size cleanup remain on the authoritative CPU path with a visible
  explanation.

#### Results You Can Actually Read

- **Find fastest pipeline** now keeps a completed comparison visible even when
  CPU and GPU are too close to name a safe winner. In that case the current
  settings stay unchanged, but the scientific checks and measured timings
  remain available for inspection.
- Replaced the dense one-row-per-node timing cell with a grouped result view:
  each node has one clear heading and a separate subrow for every tested CPU,
  CuPy, or cuCIM implementation. The compact view emphasizes total time,
  scientific agreement, and the outcome; optional details expose compute time,
  data movement, first-run cost, memory, and evidence provenance.

### Bug Fixes

- The Windows one-click installer now accepts only the exact per-track roots
  below the canonical Local App Data directory returned by
  `SHGetKnownFolderPath(FOLDERID_LocalAppData)`: `VIPP\environments\cpu` and
  `VIPP\environments\cuda13`. Custom managed roots are not accepted. If the
  canonical path contains non-ASCII characters, CUDA one-click setup is
  unavailable before environment creation or package download and the UI
  offers CPU; the fixed CPU root remains Unicode-safe. Expert-selected existing
  environments remain a separate non-mutating route. An installer-owned CUDA
  copy already in an incompatible path is not updated or repaired in place,
  and setup does not claim that a second managed CUDA copy can coexist.
- On Windows, when the effective temporary directory used by Python contains a
  non-ASCII character, VIPP now forces CuPy's in-memory compilation cache for
  that process. This avoids the affected temporary-source filename path, but
  disables CuPy's disk kernel cache for that process, so Compute Doctor or the
  first GPU work can incur compilation again in each new process. An RTX 5090
  development observation was about 52 seconds cold versus 0.87 seconds for a
  same-process refresh; these are reference measurements, not guarantees. The
  scientific kernels and results are unchanged.
- A failed CuPy kernel compilation now preserves the real `CompileException`
  and its cause. Probe cleanup no longer masks that failure with a false
  512-byte private-pool leak report caused by traceback-held arrays.
- GPU conversion suggestions now fail closed when their saved candidate no
  longer exists, the input changed, or Custom mode explicitly selects CPU or a
  different GPU implementation.
- Tunnel-backed repairs are placed beside the affected subscriber without
  moving a distant tunnel source or unrelated downstream branches.
- A scientifically successful but speed-inconclusive optimizer run is no
  longer presented like a GPU eligibility failure with its measurements
  hidden. In particular, ordinary finite decimal Binary Threshold values stay
  eligible for the reviewed CuPy implementation even when whole-pipeline
  timing cannot justify changing the saved backend.

### Validation

- Expanded the strict admission catalogue to 18 public GPU implementations
  and 23 executable evidence owners. Added exact parity, adversarial input,
  metadata, integrity, memory, cancellation, cleanup, fallback, provenance,
  and end-to-end timing coverage for Convert Dtype and the boolean mask-cleanup
  corridor.
- Added a real-CUDA corridor check proving **Convert Dtype → Gaussian Blur**
  executes as one resident segment with one upload and one download.
- Added exact and margin-safe real-CUDA segmentation corridor checks proving
  the channel view, threshold, Remove Small Objects, and Fill Holes can remain
  resident through Connected Components with one upload and one terminal
  download. A separate retained-intermediate check records the intentional
  second download instead of hiding it.
- Versioned the ordinary RL admission-evidence schema and generator for the
  v2 agreement gate. The next auditable matrix includes the authored `1e-12`
  epsilon at 10, 25, 26, 50, and 100 iterations while retaining the old
  epsilon/iteration matrices as near-identity diagnostics; the historical v1
  artifact remains labeled as such rather than being relabeled as v2 evidence.

## 0.13.0a6 - 2026-08-13

### Release Overview

This sixth 0.13 alpha makes workflow editing substantially easier and turns
GPU setup and qualification into something users can inspect without reading
developer logs. Nodes and connected graph fragments can now be copied and
pasted, settings can be transferred between matching nodes, and a processing
step can be inserted before an existing tunnel without rebuilding the graph.
Compute Doctor now explains CUDA, optional libraries, and the GPU operations
VIPP can actually use as three separate questions with one recommended next
step.

The release also expands real-world image handling with Imaris `.ims` import,
multi-series batch items, and a node for recording missing microscope metadata.
It adds clean-package checks and a reproducible GPU qualification harness, but
does not pretend that automated checks replace fresh-computer installation
testing. The included field checklist records those remaining human acceptance
steps explicitly.

### Features Added

#### Faster Graph Editing And Reuse

- Added Windows-style multi-node selection, group movement, group deletion,
  and copy/paste for complete graph fragments. Pasted nodes receive fresh
  identities while keeping their internal connections, applicable tunnels,
  notes, positions, and authored settings.
- Added **Paste Values** for transferring settings between two nodes of the
  exact same operation. The transfer is validated before it changes the live
  workflow, refreshes the interface once, and can be undone in one step.
- Added **Insert node before tunnel** from the tunnel menu, compatible palette
  drops onto a tunnel, and insertion by dragging a loose node onto a tunnel.
  Existing tunnel subscribers remain connected to the same named route.
- Added the self-explaining **Graph Editing Acceptance Check** example so these
  behaviors can be tested without preparing images or a workflow first.

#### Clearer GPU Diagnosis And Qualification

- Rebuilt Compute Doctor around three separate answers: whether CUDA can run,
  whether optional CuPyX/cuCIM providers are usable, and which of VIPP's 13
  current reviewed GPU regions are actually admitted. The window now gives one
  recommended next step, keeps technical detail collapsed, and can save an
  atomic privacy-redacted support report.
- Added one strict, spec-driven GPU admission runner that maps every public GPU
  implementation to parity, adversarial-input, metadata, input-integrity,
  memory, cancellation, cleanup, fallback, provenance, and complete-workflow
  timing evidence. The former Phase-1 benchmark now covers the current public
  background, Gaussian 2D/3D, and median implementations.

#### Image Sources, Series, And Microscope Metadata

- Added Imaris `.ims` files to the shared microscope/BioIO source path.
- Added **Set Microscope Metadata**, a pixel-preserving node for recording up
  to three emission wavelengths, objective numerical aperture, and immersion
  refractive index when a reader cannot recover them reliably.
- Batch sources now expand a multi-series container into distinct, clearly
  named items; interactive representative browsing, output names, manifests,
  and provenance retain both the container and selected series identity.
  These changes incorporate and harden community contributions from Tom Naber
  in pull requests 8, 10, and 13.

### Bug Fixes

- Prefer-GPU planning now ignores connected graph fragments whose upstream
  processing chain is deliberately loose, matching CPU behavior and allowing
  the Graph Editing Acceptance Check example to calculate normally.

### Installation And Release Qualification

- Added clean wheel and source-archive installation jobs across Windows,
  Linux, and macOS, covering CPython 3.12/3.13 and verifying packaged resources
  plus the headless Compute Doctor and installer-planner entry points.
- Added a weekly Windows cuCIM release canary. Hosted CI reproducibly verifies
  the published no-wheel bundle; an explicitly enabled protected self-hosted
  job performs the lengthy real CUDA build, installation, provenance, Doctor,
  and representative-library probes without treating a skipped GPU job as
  acceptance evidence.
- Made the exact `0.13.0a6` cuCIM add-on download and checksum-first extraction
  route visible in the README, Quick Start, and GPU guide. The bundled add-on
  guide remains version-neutral so it cannot contain a stale or circular
  archive checksum.
- Added a plain-language Windows field-acceptance checklist for recording a
  fresh CPU or CUDA install, paths containing spaces or non-ASCII characters,
  rollback after interruption, repair/update/uninstall, and a timed novice
  first-workflow test without marking unperformed checks as passed.

## 0.13.0a5 - 2026-08-12

### Release Overview

This fifth 0.13 alpha makes the Windows installer the recommended path for
ordinary users. It adds branded Automatic, CPU-only, and Prefer-GPU launchers;
a novice-facing managed setup flow; transactional install, update, and repair;
independently removable CPU and CUDA installations; and a separate optional
cuCIM local-build add-on. The manually managed pip route remains available for
advanced and non-Windows installations.

### Desktop Startup And Installation Foundation

- Added a lightweight packaged launcher with a branded splash, elapsed time,
  retained diagnostics, and progress tied to real napari/VIPP startup
  milestones. Installed GUI entry points provide Automatic, CPU-only, and
  Prefer-GPU sessions without claiming that every eligible workflow node will
  execute on an accelerator.
- Opening VIPP from napari now returns a branded loading host before importing
  the full scientific composition root. The real editor is constructed on the
  GUI thread, the initial workflow runs exactly once, and import/construction
  failures are visible and retryable.
- Packaged the canonical VIPP wordmark/mark in the wheel and added wheel smoke
  coverage for branding and the new console/GUI entry points.
- Added a separate Windows cuCIM installer coordinator and deterministic
  no-wheel ZIP bundle. It asks for an existing released VIPP CUDA environment,
  performs the pinned verified build locally, delegates artifact and
  environment admission to the existing reviewed helpers, and retains logs,
  manifests, artifacts, and resumable run state.
- Added the first production-installer core: a standard-library-only,
  non-mutating Windows discovery and planning service for managed CPU, managed
  CUDA 13, and explicitly selected existing napari virtual environments. Its
  deterministic schema-v1 JSON records stable blockers, exact release intent,
  disk reserve, future action argument arrays, acceptance, shortcuts, and a
  rollback ownership boundary without running pip, creating files, importing
  scientific/GPU libraries, downloading content, or changing the registry.
  Discovery is fingerprint-bound to its exact request, rejects remote or
  redirected interpreter/target/shortcut paths before content inspection,
  validates target-parent feasibility, and refuses existing venvs that inherit
  system site-packages. CUDA plans retain device ordinals and match the current
  runtime's all-visible-device probe contract.
- Added `vipp-install-plan plan` for inspecting that contract. GPU planning
  uses the packaged public driver/compute-capability policy without a device
  model allowlist; cuCIM and transitive dependency resolution remain explicit
  separate follow-up stages.
- Added the standalone Windows setup packaging boundary: a pinned PyInstaller
  configuration embeds the exact same-tag VIPP wheel, public policy, VIPP icon,
  and CPython/Tcl-Tk/PyInstaller notices. A verified Authenticode signature is
  still mandatory for the reserved signed filename. This alpha additionally
  permits an intentional unsigned release only under an explicit `-UNSIGNED`
  filename, with a release manifest, SHA-256 sidecars, exact-tag and frozen-
  payload verification, and `Unknown publisher` guidance.
  The optional no-wheel cuCIM local-build ZIP remains a separate, hash-bound
  GitHub release companion.
- Replaced the setup window's provisional drawn header with a raster generated
  from the official horizontal VIPP SVG and the canonical product tagline.
  Computer checks now keep visible milestone history, elapsed time, and a plain
  warning that exact-package review can take several minutes while the moving
  bar confirms setup is active. Advanced details show the live stage, progress,
  requested route/location, resolved target, technical report, and activity
  history instead of an empty placeholder.
- Added the novice managed-install lifecycle behind that packaging boundary:
  new installs, transactional updates, repairs, independently coexisting CPU
  and CUDA installations, hash-owned shortcuts, Windows Apps & Features
  registration, and ownership-safe uninstall. Manual and unrelated napari
  environments are never overwritten or removed.
- Bound the setup confirmation to every install-relevant selection. Editing the
  compute route, location, existing environment, or desktop-shortcut choice now
  requires **Check these settings** again before Install can be enabled.
  Multi-minute GPU downloads use bounded retries with a 120-second network-idle
  timeout, and transient network failures roll back the incomplete candidate.
  **Try again** rechecks the exact current selection and requires a fresh review
  rather than reusing a failed transaction.
- Reordered the installation documentation around an installer-first quick
  start. The explicitly unsigned Windows `.exe` is the primary ordinary-user
  path for this alpha; checksum verification and exact SmartScreen instructions
  precede execution, while pip
  commands, existing-napari installation, and the headless planner remain
  manual or advanced routes.
- Reduced the main README to a product-facing landing page and moved GPU setup,
  modes, qualification, supported operation families, benchmarking, fallback,
  cuCIM, and reproducibility guidance into a dedicated user-facing GPU guide.
  Exact admission matrices and machine-local timing evidence remain in their
  technical records.

## 0.13.0a4 - 2026-08-09

### Release Overview

This fourth 0.13 alpha removes the RTX-5090 model allowlist from normal public
GPU admission. Auto, Prefer GPU, and explicit Custom GPU choices can now use a
compatible NVIDIA CUDA device when the exact supported native-Windows,
CPython 3.12, CUDA 13, scientific-stack, provider, memory, and workload gates
pass. CPU remains the authoritative scientific reference, and this release
does not change workflow or batch schemas.

### Compatible CUDA 13 Admission

- Public GPU admission now requires an NVIDIA CUDA device with compute
  capability 7.5 or newer, CUDA runtime API 13.2, driver API 13.3 or newer,
  CuPy/CuPyX 14.1.1, and the pinned NumPy, SciPy, and scikit-image versions.
  Device model names are recorded for provenance rather than used as an
  allowlist.
- Auto remains workload- and evidence-driven and may correctly select CPU.
  Prefer GPU requests every scientifically and operationally eligible public
  GPU implementation even when CPU is faster. Unsupported calls remain on CPU
  with an explained decision, and no mode silently changes authored data or
  parameters.
- Immutable compute-policy artifact v8 records the compatible-device rule,
  minimum driver and compute capability, and the RTX 5090 and RTX 4050 Laptop
  GPU systems as reference validation devices rather than exclusive targets.

### Validation

- Source-current validation on an NVIDIA GeForce RTX 4050 Laptop GPU (compute
  capability 8.9) passed real Auto and Prefer-GPU execution through cuCIM and
  CuPyX with CPU parity, no runtime fallback, successful cleanup, and zero
  retained VIPP-owned device memory.
- All 13 bundled workflows then completed independently in CPU, Auto, and
  Prefer-GPU modes. Both accelerated modes selected real CUDA nodes, every
  selected node passed its declared production parity contract against the
  fresh CPU run, no fallback record was emitted, and every graph completed
  with ready outputs and successful cleanup.

### Cross-device Reproducibility

- Compatible does not imply bit-for-bit identity across GPU models. VIPP still
  enforces each implementation's declared parity contract, but reviewed
  floating-point regions can show minor device-dependent differences from
  CUDA hardware, drivers, compiler paths, or reduction order.
- Consequential work should retain the exact GPU model and compute capability,
  NVIDIA driver, CUDA driver/runtime and toolkit-package versions, Python,
  CuPy/CuPyX/cuCIM, NumPy, SciPy, scikit-image, actual implementation IDs,
  workflow, and input identities. Validate important results against the CPU
  reference and review results before combining runs from different
  environments.

## 0.13.0a3 - 2026-08-08

### Release Overview

This third 0.13 alpha fixes local GPU qualification on compatible secondary
NVIDIA hardware without widening the portable reviewed-host policy used by
Auto and Prefer GPU. It also fixes Sigma Filter compilation with CuPy 14.1.1.
CPU remains the authoritative scientific reference, and this release does not
change workflow or batch schemas.

### GPU Qualification And Execution

- Node benchmarking and `Find fastest pipeline…` can now test a secondary
  NVIDIA GPU after the exact supported native-Windows, CPython 3.12, CUDA 13,
  and scientific stack passes its probes. The device must report compute
  capability 7.5 or newer, the driver API must be at least 13.3, and the exact
  workload must still pass provider, memory, and scientific admission.
- `Find fastest pipeline…` still requires exact node parity and changed-output
  whole-pipeline parity before it offers a proposal. Accepting the proposal
  writes an explicit Custom implementation choice. A manually authored Custom
  GPU choice remains an expert override rather than proof that parity was run;
  users should requalify after changing the device, environment, data, or
  workload.
- Auto and Prefer GPU retain the narrower recorded native-Windows RTX 5090
  gate. Local qualification is not reusable performance or scientific evidence
  for a different machine.
- Sigma Filter no longer supplies an `--ftz=false` NVRTC option that conflicts
  with the option appended by CuPy 14.1.1. Its bit-level subnormal handling is
  retained without the duplicate compiler flag.

### Validation

- Source-current validation passed on native Windows with an NVIDIA GeForce
  RTX 4050 Laptop GPU (compute capability 8.9), CUDA runtime API 13.2, driver
  API 13.3, CPython 3.12.10, CuPy 14.1.1, and the pinned scientific stack. This
  complements rather than relabels the retained historical RTX 5090 records
  and is not a portable performance claim or final tagged-artifact
  qualification.
- Eighty-seven real-device provider cases passed, followed by the formerly
  failing Sigma compile/parity case. Real median and Canny-to-Otsu Find-Fastest
  transactions passed node parity, whole-pipeline validation, application,
  fallback, and cleanup checks; the optimizer retained CPU Otsu when it was
  faster.
- All 39 bundled-example executions passed: 13 CPU, 13 actual Prefer GPU, and
  13 explicit Custom attempts. A separate chained cuCIM/Gaussian/median trial
  was correctly rejected when tolerated upstream float differences produced a
  downstream bitwise mismatch. Every terminal device-memory snapshot was
  clean.

## 0.13.0a2 - 2026-08-08

### Release Overview

This second 0.13 alpha fixes compute preflight for fresh workflows whose
dynamic multi-output host operations feed accelerator-capable nodes. Such
workflows no longer need a preliminary CPU calculation to establish exact
planning descriptors. The release does not change workflow schemas or weaken
scientific input validation.

### Compute Planning

- Fresh planning now projects an exact descriptor for every exposed output of
  `Split Channels`, including the selected source port's shape, dtype, axes,
  and channel metadata.
- Shape-preserving accelerator projections no longer turn unresolved
  `shape=()` / `dtype=object` placeholders into apparently resolved values.
  When no exact deterministic projection is available, transitive descendants
  remain unresolved so compute planning safely defers them to CPU.
- The bundled red-channel object-intensity example now plans and runs under
  `Prefer GPU` without a CPU warm-up. When cuCIM is absent, its measurement
  node receives the normal explained CPU decision instead of preflight
  aborting at the downstream Otsu dtype guard.

## 0.13.0a1 - 2026-08-06

### Release Overview

This first 0.13 alpha packages VIPP's progress-to-date GPU work as a usable,
evidence-gated feature without claiming complete GPU coverage. CPU remains the
portable scientific reference implementation. Supported nodes may use reviewed
CuPy, CuPyX, or cuCIM regions through `Auto`, `Prefer GPU`, or explicit
`Custom` choices; unsupported hardware, dependencies, dtypes, parameters,
shapes, and workloads remain on CPU with a visible reason.

The release also unifies interactive, batch, generated-Python/CLI, and exported
execution; adds independent workflow tabs and new colocalization tools; fixes
ordered ND2 axis metadata, the batch `QYX`/`ZYX` ambiguity, and napari viewport
resets; and includes extensive cache, optimizer, progress, cancellation,
memory, provenance, and atomic-output hardening accumulated since 0.12.0a3.

### Compute Policy And GPU Execution

- Added the main-toolbar `CPU`/`Auto`/`Prefer GPU`/`Custom` compute policy, a
  Settings mirror, per-node CPU/CuPy/cuCIM choices, actual-run badges, and
  visible CPU fallback reasons. New sessions default to `Auto`; an authored
  node choice is not an optimizer lock unless the user explicitly locks it.
- Made Auto start from reviewed safe GPU defaults and learn only from exact
  compatible complete-pipeline timings recorded by successful, fallback-free
  completed runs. When compatible history is accelerated-only, the next global
  Auto run measures CPU once on the same execution surface. Once both exist,
  the accelerated assignment must clear the reviewed 1.20x/20-ms benefit
  margin or Auto selects CPU. Auto never silently benchmarks multiple
  implementations or combines incompatible interactive, batch, or
  registry-lifecycle timing surfaces.
- Renamed the per-node policy from **Selective** to **Custom**. Development
  builds that predate 0.13.0a1 may display or serialize `selective`; current
  builds accept that legacy spelling and write `custom` without changing its
  behavior.
- Added `Prefer GPU` for users who want every scientifically eligible reviewed
  GPU implementation even when it is no faster than CPU. It considers both
  `public_custom` and `public_auto_candidate` providers and bypasses only the
  CPU-versus-GPU performance gate; scientific, dtype, parameter, shape,
  dependency, environment, and memory gates still apply, and VIPP never inserts
  a cast. Complete comparable GPU timings choose the fastest GPU; otherwise a
  stable implementation-ID order is deterministic. Unsupported nodes receive
  an explained CPU decision. Prefer GPU requires visible fallback, ignores but
  preserves dormant per-node preferences, and does not expose Custom-only
  benchmarking or `Find fastest pipeline…`. Developer-hidden providers remain
  excluded unless experimental admission is explicitly enabled.
- Added worker-based compute setup diagnostics, copyable environment repair,
  system RAM plus discrete VRAM reporting, and unified-memory presentation for
  platforms such as Apple silicon. macOS remains CPU-only in this alpha.
- Added exact-workload node benchmarking and a review-before-apply whole-
  pipeline optimizer. It compares every scientifically eligible implementation
  for unlocked nodes, reuses complete exact evidence, models transfers across
  the graph, reports nested progress and deadline exhaustion, and changes
  nothing until the measured assignment is revalidated and accepted.
- Added a versioned evidence policy, exact implementation/environment identity,
  device-resident execution, memory admission, a fair process-wide accelerator
  lease, structured OOM fallback, cooperative cancellation, and cleanup-gated
  cache/publication behavior.
- Added public-candidate accelerated regions for Rolling-Ball/Subtract
  Background, median, 2D/3D Gaussian, ordinary Richardson-Lucy,
  Richardson-Lucy TV, Canny, Otsu, Sigma Filter, connected components, basic
  object measurements, and basic object-plus-intensity measurements. Each
  provider is admitted only for its documented scientific region.
- CUDA extras now pin the exact NumPy, SciPy, scikit-image, CuPy, and CUDA stack
  used by admission. Public GPU evidence in this alpha remains limited to the
  recorded native-Windows CPython 3.12 / CUDA 13 / RTX 5090 environment;
  installation on another system is not an acceleration claim.
- The ordinary CUDA extra does not distribute or require cuCIM. Windows users
  can optionally build the exact cuCIM 26.6.0 tag/commit with the release's
  fixed local recipe, then use the manifest-verifying setup helper to install
  and approve their own wheel in an existing released VIPP environment. Policy
  pins the source, recipe, canonical payload, CUDA/scientific stack, and
  workload while allowing each local ZIP archive to have its own verified
  SHA-256. Without an approved build, background and basic-measurement
  candidates remain on CPU.

### Durable Workflows, Batch, CLI, And Export

- Advanced workflow persistence to schema 4. It stores portable compute mode,
  fallback policy, per-node preferences, precision policy, and workload policy,
  including the serialized/CLI `prefer_gpu` value, while excluding
  machine-local devices, memory limits, runtime state, and benchmark evidence.
  Schema-3 workflows load with explicit CPU intent.
- Advanced collection batch configs and manifests to schema 3. Source bindings
  can now carry a guarded raw-to-effective axis declaration such as `QYX ->
  ZYX`. Successfully read source records include the raw axes, effective axes,
  and declaration; the embedded config retains declarations for items skipped
  or failed before source reading.
  Version-1 and version-2 configs remain loadable: version 1 migrates to CPU
  because it carried no accelerator intent, while version 2 keeps its saved
  compute request. Both older versions contain no declaration until reviewed
  and saved as version 3.
- Added the novice-facing `Image stack` choice. A new unsaved row starts at
  `Automatic (recommended)`; when an exact `QYX` representative reaches a
  demonstrated `ZYX` workflow requirement, VIPP visibly selects
  `Pages are depth slices (Z stack)` and retries with the guarded declaration.
  `Use the file's labels unchanged` is an explicit opt-out, and uncommon
  mappings remain under `Something else (advanced)...`.
- Kept automatic page interpretation UI-only until it resolves. Saving an
  unresolved automatic row stores no declaration and reloads as file-unchanged,
  as do historic/headless blank configs. A resolved suggestion saves concrete
  `QYX -> ZYX`, so later GUI and headless runs reproduce the reviewed choice.
- Added a representative scientific-contract preflight before output-directory
  creation, run artifacts, or compute-device setup. It applies the same source
  declarations used by interactive representative loading and stops
  deterministic axis failures regardless of `continue_on_error`. The check is
  representative-only; unreadable or differently shaped later files retain
  normal item-level failure handling and every item revalidates its declaration.
- Routed interactive calculation, saved batch runners, generated Python/CLI,
  and standalone export through the same execution service. All paths now carry
  the configured and effective request, exact per-node implementation
  provenance, structured fallback/OOM records, nested operation progress,
  cooperative cancellation, and cleanup state.
- Added durable saved-runner overrides for compute mode, fallback policy, and
  per-node preference. Cancellation finalizes manifests/checkpoints, returns
  exit code 130 from the CLI, and blocks unpublished output from escaping.
- Strengthened private staging and atomic publication. An interruption during a
  multi-output item's final promotions can still leave a manifest-recorded
  `partial` item; automatic checkpoint resume and semantic T/C/Z/HCS or
  container-series iteration remain outside this alpha.

### Image I/O, Execution, And Interface Fixes

- Fixed the collection-batch `QYX`/`ZYX` bug without blindly treating every
  generic TIFF page axis as Z. An exact `QYX` source receives one visible,
  guarded Z-stack suggestion only after the workflow demonstrates a `ZYX`
  requirement. The saved declaration relabels semantics in place, while
  `Reorder Axes` remains a pure pixel-and-metadata transpose and cannot rename Q
  to Z.
- Corrected ND2 metadata normalization to follow the reader's actual ordered
  dimensions, restoring the proper T/Z/C sliders and slice updates for affected
  files.
- Preserved napari camera, zoom, translation, displayed dimensions, and slice
  positions when recalculating and replacing an inspected layer during isolated
  node tuning.
- Added independent workflow tabs with retained graph, cache, history,
  inspection, and batch-origin state; live source-binding subtitles; draggable
  tunnel rerouting; and per-output persisted display styles.
- Added responsive high-resolution and pop-out colocalization scatter views,
  cached threshold-independent densities, exact full-ROI recounting, memory-
  bounded histogram construction, and export at the selected display size.
- Added persistent Low (90 × 55), Standard (180 × 110), High (360 × 220), and
  Very High (720 × 440) thumbnail render detail without changing card size or
  scientific results. High and Very High retain larger backing images for HiDPI
  display, downsampling, or zoomed graph inspection rather than guaranteeing a
  larger on-screen card. Changing
  detail rerenders cards while retaining cached exact Stack contrast limits.
  Stack statistics remain full-output and resolution-independent; responsive
  Slice contrast normalizes the spatially sampled current view, so its display
  limits may change slightly with detail.
- Replaced full-stack `uint8`/`uint16` thumbnail percentile sorting with exact
  native-dtype histograms on CPU or eligible CuPy GPU. Min-max uses a faster
  exact native reduction instead of building a histogram. The presentation-only
  Auto policy uses full output dtype/bytes—not render resolution—with
  conservative cold crossovers of 384 MiB for `uint8` and 512 MiB for `uint16`,
  then 32 MiB after the GPU path is warm. These measured heuristics are not a
  universal fastest guarantee; CPU and Prefer GPU remain explicit controls, and
  float/other dtypes retain exact NumPy-compatible CPU percentile behavior.
- Added separate per-node thumbnail-contrast pending/backend/fallback/error
  status in a compact selected-node inspector row, plus shared progress and
  cooperative cancellation for thumbnail statistics. Scientific CPU/GPU
  badges remain on cards; presentation success is muted, while fallback and
  error remain emphasized. CPU integer
  histogram and min-max work stop between bounded chunks; an active GPU
  kernel/synchronization or exact NumPy percentile for another dtype may have a
  non-interruptible inner pass. Main compute CPU hard-forces presentation CPU,
  main Prefer GPU biases presentation Auto, and these display decisions never
  replace scientific CPU/CuPy/cuCIM provenance. CPU-selected micro-workloads
  up to 1 MiB, eight requests, and eight aggregate channel lanes complete
  inline to avoid worker-queue overhead; larger, high-channel, and GPU work
  remains asynchronous and cancellable.
- Hardened structural cache identity, stale-source rejection, optimizer/manual
  barriers, memory accounting, cancellation cleanup, measurement assembly,
  and generated-output provenance/publication.

### Compatibility And Known Alpha Limits

- Colocalization results may change materially. Intensities, thresholds, and
  sums now remain in native units; the Costes search and Pearson/Manders domains
  are an experimental source-aligned Fiji Coloc 2 3.1.0 target with independent
  golden parity still pending; and existing
  `manders_m1`/`manders_m2` names now alias thresholded tM1/tM2. Do not combine
  0.12 and 0.13 numerical results without review.
- `ImageJ Auto Threshold (8-bit)` is a separate explicit conversion/threshold
  node; generic scikit-image threshold nodes are unchanged.
- Generated Python is exact-version locked and should be regenerated for
  0.13.0a1. The generated `batch_process()` folder loop remains a warned,
  non-durable convenience; use the saved batch runner for production work.
- A source-axis declaration preserves existing calibration positionally; it
  does not discover a missing Z step, unit, or origin. Verify Output Metadata
  and use `Set Pixel Size / Units` before calibration-dependent analysis.
- CPython 3.12 and 3.13 are the supported CPU interpreters. GPU extras and the
  reviewed CUDA environment remain CPython 3.12-only.
- Linux GPU qualification, RTX 40-series evidence, Apple GPU acceleration,
  a hosted downstream cuCIM distribution, Clara support, remaining GPU nodes,
  semantic collection iteration, HCS traversal, and scalable OME-Zarr previews
  remain post-alpha work.

### Colocalization Correctness

- Corrected pixel and object colocalization to retain finite native
  intensities instead of jointly rescaling and clipping the two channels to
  0..255. Thresholds and intensity sums now report native units.
- Replaced the approximate automatic-threshold search with an experimental,
  source-aligned implementation targeting Fiji Coloc 2 3.1.0's classic Costes
  `SimpleStepper`, including Java rounding, one-unit native threshold steps,
  any-channel-below-threshold (OR) Pearson populations, last-tested threshold
  retention, the cursor-offset quirk, and sequential variance accumulation.
  Independent Fiji-generated golden parity validation remains pending.
- Added source-aligned Pearson no-threshold and threshold-domain outputs. New
  canonical column names explicitly distinguish `any_channel` OR populations
  from the `both_channels` intersection; the shorter ambiguous names remain
  compatibility aliases for existing table consumers.
- Added Fiji Manders M1/M2 and thresholded tM1/tM2 definitions. Existing
  `manders_m1`/`manders_m2` columns now alias tM1/tM2 for workflow
  compatibility, while the previous above-threshold intersection fractions
  remain available under descriptive non-Manders column names.
- Pixel and object tables retain `coloc_semantics=fiji_coloc2_3.1` as the target
  contract identity and now separately record
  `coloc_validation_status=experimental_source_aligned_golden_parity_pending`.

### Workflow Compatibility

- Crop Stack now preserves image, mask, and label graph-port types so cropped
  ROI masks restore correctly when connected to masked analysis nodes.
- Added an explicit ImageJ Auto Threshold (8-bit) node with an experimental,
  source-aligned ImageJ 1.54p target for scalar uint8, uint16, and float32
  per-plane ScaleConversions plus `Default`/`Triangle` AutoThresholder behavior.
  Independent ImageJ-generated golden parity validation remains pending. Bool
  handling and RGB/RGBA luma reduction are VIPP extensions and are not claimed
  as ImageJ-exact. Infinite float inputs are rejected deliberately instead of
  preserving ImageJ's non-informative all-zero conversion. VIPP's generic
  scikit-image threshold nodes remain unchanged.

### Connected Components CPU/GPU Vertical Slice

- Promoted `Label Connected Components` as a complete CPU/GPU vertical slice.
  The authoritative CPU path retains nonzero foreground semantics, SciPy face
  or full connectivity, independent 2D/3D leading blocks whose IDs restart at
  one, and shape-preserving native `int32` output.
- Added lazy `cupyx-connected-components-v1` execution for the exact reviewed
  boolean-mask 2D/3D region. GPU output must match SciPy label IDs bit for bit;
  partition equivalence with different numbering is rejected. Numeric-mask
  conversion, 1D labeling, oversized blocks, and unqualified environments
  visibly remain on CPU.
- Added resident Otsu-to-label planning, exact output/fact projection, distinct
  scientific cache identity, synchronized leading-block progress and
  cancellation, transactional cleanup, and a conservative memory model that
  holds the full bool input plus `int32` output and one active block's workspace.
  A single plane or volume is one atomic CuPyX operation, so it cannot report
  finer progress or cancel mid-volume in this implementation.
- Added immutable compute-policy artifact v5 and source-current RTX 5090
  admission/performance evidence. The validated region is a normal public
  `Auto`/`Custom` candidate in the pinned environment; the timing matrix is
  machine-local screening rather than a portable speed promise or durable Auto
  assignment.

### Sigma Filter CPU/GPU Vertical Slice

- Added `Sigma Filter` under `Filtering > Smoothing & Denoising` as a public,
  edge-preserving Lee filter compatible with the documented behavior of Fiji
  Sigma Filter Plus. The node processes resolved `YX` planes slice-wise,
  handles channels and leading axes independently, uses nearest/clamped borders,
  and preserves finite native-endian `uint8`, `uint16`, or `float32` shape and
  dtype; non-native-endian arrays fail closed before accelerator transfer.
- Froze the radius-dependent circular footprint, float32 sample/square and
  ordered float64 accumulation rules, inclusive center-relative sigma interval,
  exact minimum-count ceiling, both fallback modes, and Fiji-compatible
  unsigned half-up restoration. ROI/mask behavior remains explicitly outside
  the version-1 node contract.
- Added independently generated unsigned-integer fixtures from the official
  ImageJ Sigma Filter Plus bytecode. Two narrow VIPP stabilizations are recorded
  as intentional differences rather than mislabeled as Fiji parity: exact
  `ceil(N * fraction)` and deterministic clamping of negative floating-point
  variance to positive zero.
- Added a lazy CuPy `RawKernel` implementation with resident device execution,
  explicit contiguous-axis staging, bounded 64-row launches, synchronized
  progress, cooperative cancellation, subnormal-preserving float conversions,
  conservative memory admission, dtype-specific parity, exact provenance, and
  visible CPU decisions outside the validated region. The source-current RTX
  5090 record passed 10 exact admission cases, 10 matched rejections, all 18
  timed workloads bitwise, synchronized cancellation, and zero-residue cleanup,
  so the exact region is now a normal public `Auto`/`Custom` candidate.
  Transfer-inclusive examples ranged from 23.57x for a 512² radius-0.5 plane to
  170.95x for a 2048² radius-10 plane. Radius 0.5 first cleared both gates at
  512²: its 20.13-ms saving exceeded the 20-ms gate and its paired 95% speedup
  lower bound was 19.58x against the 1.20x gate. Radius 2 also cleared at 512²,
  and radii 5/10 at the smallest tested 256². Timings are machine-local rather
  than portable guarantees.

### Graph Authoring

- Image Source cards now show a live, elided binding subtitle for napari layers,
  files, samples, and collection representatives. The complete binding remains
  available in the card tooltip and follows collection-item changes.
- Named output tunnels can now be rerouted by dragging their source badge to a
  different compatible output. Preview and commit share the same type, cycle,
  and topology validation; the atomic edit is undoable.
- Existing insert-on-wire support remains the in-canvas path for splitting a
  connection around a newly dropped compatible node.

### Workflow Sessions

- Added a movable workflow tab bar with independent live pipelines, calculated
  results, ancillary caches, undo/redo histories, inspector state, paths, and
  dirty baselines. New and Load create sessions without replacing another open
  workflow; tabs support rename, reorder, and Save/Discard/Cancel close handling.
- Tab clicks now acknowledge the selected workflow immediately with a dedicated
  indeterminate loading state while VIPP restores its retained graph, inspector,
  thumbnails, and cached results. Switching no longer rebuilds the entire tab
  bar, and the loading text makes clear that no scientific recalculation occurs.
- A collection batch now runs in a single background worker tagged to its
  originating workflow. Other tabs remain editable while it runs, progress and
  completion return only to the origin, and closing the origin, launching a
  second batch, or closing VIPP is blocked until the run finishes or cooperative
  cancellation completes and final state is persisted.

### Compute Policy Interface

- Added a main-toolbar `CPU`/`Auto`/`Prefer GPU`/`Custom` compute policy with
  an actual-run summary, a Settings-menu mirror, and per-node choices for nodes
  that declare GPU implementations. Workflow-v3 files deliberately reopen in
  CPU mode; workflow-v4 files persist portable compute intent.
- Simplified Custom node choices to `Auto for this node`, `CPU`, and one
  entry per declared GPU library. `Best GPU` appears only when multiple
  libraries compete, while loaded exact pins remain honestly visible as an
  advanced compatibility entry until replaced.
- Accepted execution reports now render compact CPU, CuPy, cuCIM, or amber CPU
  fallback badges on node cards. Stale badges are muted and distinguish an
  active update from an idle previous result.
- Consolidated VIPP feedback into one severity-aware message-strip component.
  Routine status remains lightweight, while only actionable errors receive a
  filled full-width alert.
- Exact optimizer validation now evaluates the runnable frontier before manual
  barriers. Nodes beyond a deliberately skipped manual/cache boundary no longer
  request a nonexistent private CPU parity target.

### Colocalization Inspector

- Added a colormap selector to the resizable scatter pop-out. It is linked in
  both directions with the inspector selector and redraws from cached density
  without changing or recalculating scientific results.
- Threshold scrubbing now keeps a compatible scatter density visible and moves
  its guides immediately while exact counts are recalculated. Stale counts are
  replaced with a calculating state, rapid requests are coalesced, and a density
  is retained only while the channel, ROI, and intensity context still matches.
- Recalculating the same VIPP Inspect output now preserves user-selected napari
  display styling, including its colormap. Switching to a different output or
  an incompatible layer representation still initializes VIPP's safe defaults.
- VIPP Inspect display styling is now remembered independently for each node,
  output port, and RGB display surface, and is restored from saved workflows.
  Intensity-domain settings remain dtype-specific, and the inspector header now
  provides an explicit reset-to-defaults action for the selected node.
- Added real `Colocalization Scatter Plot` and masked graph nodes with
  independently configurable histogram bins and square output size up to 4096,
  independent native populated axis ranges, and optional symmetric percentile
  clipping. A reusable resizable scatter dialog provides live threshold
  feedback plus PNG/TIFF export at the current display resolution.
- High-resolution scatter inspectors now schedule by histogram memory cost,
  reuse threshold-independent densities while recounting the exact full ROI,
  and enforce a byte-budgeted cache. Interactive rendering is explicitly
  capped at 1024 bins per axis (and reports that cap); graph nodes retain their
  requested resolution up to 4096. Graph renders aggregate density before
  downsampling and draw threshold guides afterward so neither sparse bins nor
  guides disappear.
- Masked high-resolution graph renders now accumulate their histogram in bounded
  chunks, avoid simultaneous full-ROI channel copies, release intermediate
  density buffers promptly, and preserve native log-density contrast when a
  histogram is enlarged for output.

## 0.12.0a3 - 2026-07-20

### Release Overview

This third 0.12 alpha makes collection batching faster, clearer, and more
dependable without changing workflow schema version 3 or batch-config schema
version 1. It adds direct one-click execution with a fresh plan-only preflight,
safer destination guidance, fast all-existing skips, more resilient atomic
artifact handling, and an optional one-file workflow-plus-batch configuration.
No scientific image-processing kernel or cached-pixel contract changed.

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
