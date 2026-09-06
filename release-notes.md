# VIPP 0.15.0a1

VIPP 0.15.0a1 brings a redesigned batch workflow, a clearer inspector, richer plots and measurement tables, and new GPU-assisted measurement paths. The largest change is the journey from checking a collection to deciding what to save, following each sample, and understanding the final result.

This is alpha software. Keep original data and copies of your workflows and batch configurations, and validate important analyses on representative data before drawing scientific conclusions.

## A batch workflow built around the task

The batch window now follows four stages: **Setup → Items & outputs → Overrides → Run & results**. The next action and current activity remain visible at the bottom of the window.

- **See the files before waiting for the checks.** The source list appears as it is discovered, then shows background metadata/content checks and a checked-file counter. One image container can contain several samples, so the final item list is established after inspection.
- **Check without calculating the images.** Check batch validates the collection and its output plan. Preview selected is a separate, optional calculation when you want to inspect a sample's pixels in the graph.
- **Understand what will be saved.** Planned outputs lead with the producing node, output type, and format. Exact paths remain available, with clearer actions to locate source and result files.
- **Edit exceptions deliberately.** The override table has persistent search labels, clearer workflow defaults, centered values, and distinct selection, bulk editing, and reset actions. The existing Run/Bypass controls now form a matching collapsible section, with colour highlighting for explicit batch choices.
- **Move between sample and collection.** A compact representative navigator in the main workflow provides a direct route back to that sample's batch details and overrides.

## Keep some outputs, overwrite others

Existing-output decisions no longer require treating every sample the same way. Right-click an item to choose **Keep existing outputs**, **Rerun and overwrite outputs**, or **Use batch default**.

Those choices are saved against the exact sources and destinations, not their position in the table. Changing only a file policy updates the checked plan without repeating source inspection. Keeping existing outputs still allows missing outputs to be created; duplicate destinations and protected inputs remain safety errors.

The UI now distinguishes files to create, existing files to keep, files to overwrite, and decisions still needed. Keeping an old file does not claim that this run calculated or scientifically verified it.

## Follow the run—and read the result

- Preparation now reports what is being checked before the first item starts. The worker reuses the fresh Run validation instead of repeating the same collection scan.
- The upper progress label identifies the sample and its current graph step, such as **Running (node 12/32)**. The lower label uses the node's readable name and describes the current operation without repeating the filename.
- Progress text reserves two bottom-aligned lines, reducing layout movement as messages wrap.
- A cooperative stop is no longer incorrectly classified as a CPU/GPU cleanup failure. Genuine cleanup failures still prevent unsafe reuse.
- Completed and stopped runs show an **inline run report**: elapsed time, item outcomes, newly saved versus kept outputs, and expandable failure reasons. Selecting a row reveals that sample's outputs; clicking its underlined name explicitly opens item review.
- The manifest JSON remains available as a technical provenance record, not as the main human-readable report. Refresh file status checks disk presence; it does not revalidate or restart the batch.

## A clearer inspector and more useful plots

The inspector has been rebuilt with clearer connected-input summaries, context-aware controls, consistent theme-aware spacing, and background diagnostics. Heavy inspection work is kept out of graph dragging and unnecessary refreshes.

Measurement results open in resizable, sortable table windows with units and CSV/TSV export. Sorting changes only the view; exported data retains the workflow's original row order and values.

The new **Intensity Histogram** node produces reproducible scalar or multichannel distributions with shared bin edges, explicit ranges, and linear or logarithmic spacing. Its table includes counts, fractions, densities, and cumulative values, with excluded/underflow/overflow values accounted for in metadata. Plot display changes reuse that table rather than rescan the source image.

Histogram and colocalization pop-outs are easier to read and explore. Scatter plots use zero-inclusive shared axes by default, offer equal-axis and populated-data zoom controls, and support up to 4096 bins per axis through background, memory-gated calculation. Overlapping channel histograms use clearer colours and more transparent fills.

## Measurement acceleration and workflow reliability

Eligible NVIDIA workflows gain two measurement paths:

- **Measure 3D Mesh Morphology:** GPU preparation of eligible non-negative int32 3D labels, followed by the authoritative CPU marching-cubes and convex-hull calculation. This is a hybrid acceleration, not an entirely GPU-based mesh analysis.
- **Analyze Skeleton:** GPU measurement of eligible already-skeletonized boolean 2D/3D inputs, with the same voxel-graph and calibration rules as the CPU reference. This does not add GPU skeleton thinning.

The toolbar is reorganized around workflow commands, preview, compute, graph navigation, and status. Image Source also accepts an image file dropped onto its card, or a copied file/image pasted with Ctrl+V or Cmd+V.

**Find fastest** now shows total elapsed time and time spent in the current stage, even when a CPU library call cannot provide intermediate progress. The clocks are not completion estimates or proof that the current calculation is advancing. The time limit and cancellation take effect at safe stopping points, so an in-progress CPU/GPU call may need to finish first.

Reliability fixes cover source-loading and cache reuse, mixed-rank time/channel navigation, Crop/Inspect layer lifetime, RGB/RGBA interpretation, and Combine Channels colour saving/invalidation. A failed source replacement no longer leaves unrelated downstream pixels presented as current results.

## Upgrading: changes worth reviewing

- **Batch configuration version 6** stores individual existing-file choices and reads supported earlier versions 1–5. Workflow version 6 and manifest version 5 remain unchanged. Older VIPP releases that only understand version-5 batch configurations cannot read a newly saved version-6 configuration.
- **ImageJ Default Threshold (8-bit)** replaces the public ImageJ Auto Threshold method dropdown with a fixed Default method. Previously saved ImageJ Triangle nodes keep their distinct legacy calculation; they are not silently changed to Default or to VIPP's generic Triangle Threshold.
- The ImageJ implementation remains experimental and source-aligned to ImageJ 1.54p. Independent ImageJ-generated golden parity is not claimed; generic Triangle and Isodata remain separate scientific methods.
- **Clip** is now labelled **Clamp Intensity**. The name clarifies its existing bound behavior rather than introducing a new operation.
- **Minimum Threshold** explains histogram valley detection and its smoothing-pass convergence limit more explicitly. Encoded RGB/RGBA inputs now receive clearer validation when a scalar-only axis choice would be misleading.

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
