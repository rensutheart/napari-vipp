# VIPP 0.15.0a1

VIPP 0.15.0a1 is a substantial update to both everyday workflow building and batch analysis. Since **0.14.0a3**, the main toolbar and node library have been redesigned, the inspector has become more closely tailored to each operation, and plots and measurement tables have gained richer ways to explore results. New normalization methods, measurement optimizations, a reworked batch workflow, and important correctness fixes round out the release.

This is alpha software. Keep original data and copies of your workflows and batch configurations, and validate important analyses on representative data before drawing scientific conclusions.

## A redesigned workspace and node library

- **Clearer toolbar organization.** Workflow commands, preview settings, compute controls, and graph navigation have distinct groups, with consistent icons and spacing, narrower-window layouts, and a persistent status/activity footer. Presentation options live in the Preview menu; general configuration is collected in Settings.
- **A node library that can give space back to the graph.** Collapse it to a category-icon rail, then use category and global-search popups to find nodes. Category counts, operation icons, explanatory tooltips, and accessible labels improve discovery. Clearing a search restores the previous expansion state, and context menus provide add-node and expand/collapse actions.
- **Quicker workflow building.** Drop an image file onto an Image Source card, or paste a copied file or image with Ctrl+V/Cmd+V. Dropping a compatible palette node onto a terminal output appends and connects it as one undoable edit.
- **More informative Find fastest.** Total and current-stage elapsed timers continue updating during long benchmarks. They show elapsed time, not an estimated finish or proof of numerical progress inside an opaque CPU/GPU call; cancellation and time limits still wait for safe checkpoints.

## An inspector built around the selected operation

The inspector overhaul goes beyond appearance. Operation-specific layouts organize connected inputs, parameters, distributions, results, metadata, and compute evidence according to what is useful for the selected node.

- **Understand the inputs.** Connected-input cards identify each input's role, source, and image or table characteristics. Image, label, table, source, and writer nodes expose appropriate controls and output actions.
- **See relevant scientific feedback.** Object filters emphasize object-size or measurement distributions instead of label-ID histograms. Deconvolution compares the observed image with its output, rather than treating the point-spread function as the comparison image. Calculate New Image shows its weighted equation alongside the connected inputs.
- **Inspect comfortably.** Collapsible sections, responsive parameter forms, consistent sliders, theme-aware colours, and clearer spacing make narrow docks and detached windows easier to use.
- **Keep graph interaction responsive.** Moving a selected node preserves its inspector and scroll position. Heavy diagnostics and routing work are deferred during dragging; large diagnostics run in the background, reuse cached results, and discard obsolete requests.

## Richer plots, tables, and analysis

- **Complete measurement tables.** Resizable windows offer numeric/natural sorting, units, missing and non-finite value diagnostics, and background CSV/TSV export. Sorting changes only the view; exports preserve the original scientific rows and values.
- **New Intensity Histogram node.** Calculate reproducible scalar or multichannel distributions with shared bin edges, explicit ranges, and linear or logarithmic spacing. Results include counts, fractions, densities, and cumulative values, with excluded and out-of-range values recorded in metadata. Display changes reuse the calculated table instead of rescanning the image.
- **Improved colocalization and histogram exploration.** Metric-node scatter pop-outs gain smoother threshold-guide interaction, zero-inclusive shared axes, equal-axis and populated-data zoom, and up to 4096 density bins per axis through background, memory-gated calculation. Plot controls do not change the full ROI used for exact metrics. Overlapping channel histograms have clearer colours and translucent fills.
- **Four additional Normalize methods.** Robust z-score (median/MAD), maximum-absolute scaling, reference z-score using saved mean/standard deviation, and percentile-based 0–1 scaling join the existing min–max and z-score methods. Fitted statistics use the whole input, not separate per-channel fits; signed methods should not be mistaken for 0–1 scaling. Existing methods retain their previous calculation.
- **Faster object measurements.** CPU mesh and skeleton analysis reduce work to relevant object regions, and nearest-object calculations use spatial indexing while retaining coordinate, calibration, and tie-breaking rules.
- **New GPU measurement paths.** Eligible NVIDIA workflows gain hybrid **Measure 3D Mesh Morphology** for non-negative int32 3D labels and GPU **Analyze Skeleton** measurement for already-skeletonized boolean 2D/3D inputs. Mesh preparation is accelerated while marching cubes and convex hulls remain CPU work. Skeleton measurement retains the CPU reference's voxel-graph and calibration rules; it does not add GPU thinning.

## A clearer batch workflow, from checking to the final report

**Setup → Items & outputs → Overrides → Run & results** separates preparing a collection from calculating it, with a persistent next action and activity strip.

- **Review files while checks continue.** The source inventory appears before slower metadata/content checks finish, with active-file feedback and checked counts. One container may contain several samples, so inspection determines the final item list. Check batch does not calculate images; Preview selected remains a separate, optional graph calculation.
- **Understand outputs and edit exceptions.** Output details emphasize the producing node, data kind, and format, with paths available separately. The labelled override matrix has clearer workflow defaults, a guided bulk editor, separate selection/reset actions, and a matching collapsible Run/Bypass section with highlighted exceptions. The main workflow's sample navigator links directly back to batch details.
- **Keep some outputs and rerun others.** Right-click an item to choose **Keep existing outputs**, **Rerun and overwrite outputs**, or **Use batch default**. Choices follow exact sources and destinations, not row numbers. Policy-only changes reuse source checks; keeping existing files still permits missing outputs to be created. Counts distinguish files to create, keep, overwrite, or review, while protected inputs and duplicate destinations remain safety errors.
- **Follow preparation and each sample.** Run reports its final validation before processing; the worker reuses that fresh plan instead of repeating the collection scan. Progress shows the sample and **Running (node 12/32)**, then the readable node name below. Two bottom-aligned text lines reduce layout movement.
- **Read the outcome in the window.** Completed and stopped runs show elapsed time, item outcomes, newly saved versus kept files, and expandable failure reasons. Row selection reveals outputs; clicking the underlined name explicitly opens item review. Completed runs remain in report-review state. The JSON manifest is secondary technical provenance; Refresh file status checks disk presence without revalidating or restarting the batch. Keeping an existing file does not claim this run calculated or scientifically verified it.

## Important fixes

- **Source loading and stale results:** CZI metadata and pixels share one container-open lifetime, avoiding duplicate opens while retaining mutation checks. A failed source replacement no longer presents incompatible downstream pixels as current results, while independent branches remain available.
- **Axis and colour correctness:** fixes cover mixed-rank time/channel navigation, scalar images incorrectly inferred as RGB, encoded-colour axis validation, and Crop/Inspect layer lifetime. Rendering Select/Reorder axis controls no longer silently changes saved parameters. Combine Channels colour edits now propagate consistently to previews, layers, and saved workflows.
- **Large object-label IDs:** association pair counting preserves unsigned label IDs beyond the signed 64-bit range instead of corrupting them through integer conversion.
- **Execution and cache reuse:** mixed CPU/GPU branches no longer produce a false execution-unit cycle. Calculate All preserves already-running upstream work, and downstream composite edits retain valid expensive manual deconvolution results. Presentation failures no longer leave the application indefinitely showing Processing.
- **Stable result windows:** detached plots stay associated with their originating node/workflow, outdated results are marked stale, and late background table work cannot replace a newer result. Themed histogram/table startup is corrected.
- **Safe batch cancellation:** a cooperative stop is no longer mistaken for a CPU/GPU cleanup failure. Genuine cleanup failures still block unsafe reuse.

## Upgrading: changes worth reviewing

- **Batch configuration version 6** stores individual existing-file choices and reads supported earlier versions 1–5. Workflow version 6 and manifest version 5 remain unchanged. Older VIPP releases that only understand version-5 batch configurations cannot read a newly saved version-6 configuration.
- **ImageJ Default Threshold (8-bit)** replaces the public ImageJ Auto Threshold method dropdown with a fixed Default method. Previously saved ImageJ Triangle nodes keep their distinct legacy calculation; they are not silently changed to Default or to VIPP's generic Triangle Threshold.
- The ImageJ implementation remains experimental and source-aligned to ImageJ 1.54p. Independent ImageJ-generated golden parity is not claimed; generic Triangle and Isodata remain separate scientific methods.
- **Clip** is now labelled **Clamp Intensity**. The name clarifies its existing bound behavior rather than introducing a new operation.
- **Minimum Threshold** explains histogram valley detection and its smoothing-pass convergence limit more explicitly. Encoded RGB/RGBA inputs now receive clearer validation when a scalar-only axis choice would be misleading.
- Legacy scatter-raster nodes remain loadable and executable but are hidden from the palette; use the corresponding metrics-node pop-out for new interactive plots. Regenerate exported Python after upgrading, because exported programs require the exact VIPP runtime version.

## Installation and limits

Use the [0.15.0a1 release page](https://github.com/rensutheart/napari-vipp/releases/tag/v0.15.0a1) for Python packages and installer assets. Windows setup is explicitly unsigned. The separate Apple Silicon and Intel macOS packages are unsigned, unnotarized, and CPU-only. Verify the matching SHA-256 checksum and follow the platform installation instructions.

Manual installation remains available in a dedicated CPython 3.12 or 3.13 environment. On Windows or Linux:

```bash
python -m pip install "napari[pyqt6]>=0.6" "napari-vipp==0.15.0a1"
vipp
```

On macOS, use `"napari[pyside6]>=0.6"` in place of `"napari[pyqt6]>=0.6"`; macOS execution remains CPU-only.

Exact source-content checks can still take time on large containers or slow disks. Cancellation is cooperative: a running kernel, library call, or output write may need to finish a safe unit of work. Most graph operations still materialize their inputs in memory; this release does not introduce general lazy/chunked execution. GPU eligibility and speed depend on the operation, data, memory, and supported environment.

The [0.15.0a1 manual](https://rensutheart.github.io/vipp-mkdocs/0.15.0a1/) describes the updated interface and the distinction between checking, previewing, running, and reviewing results.

[Full source comparison: 0.14.0a3 → 0.15.0a1](https://github.com/rensutheart/napari-vipp/compare/v0.14.0a3...v0.15.0a1)
