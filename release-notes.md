# VIPP 0.15.0a2

VIPP 0.15.0a2 makes microscope files easier to open, adds a complete mesh-object workflow, and improves everyday searching, display and parameter editing. It also expands Binary Threshold and makes intensity inversion explicit.

This is alpha software. Preserve original data, workflows and environments, and compare consequential analyses with representative reference data after upgrading.

## Microscope readers and updates

- **Native microscope readers are included.** Ordinary plugin and managed desktop installations include Zeiss CZI, Leica LIF/LOF/XLIF, Nikon ND2 with legacy codecs, and Olympus OIF/OIB/OIR support. Bio-Formats, including the IMS/VSI route, remains optional and can require Java downloads on first use.
- **Reader support is visible in Image Source.** A separate, collapsed system-information section checks installed reader support and shares results across the session. Missing-reader setup offers a package review and waits for VIPP/napari to close before adding missing packages; it does not replace the installed scientific stack. A successful dependency check is not validation of a particular acquisition or its metadata.
- **A quiet update badge.** Daily GitHub checks highlight newer releases without opening a dialog. Click for release notes and download guidance, or check manually. Automatic checks and prerelease inclusion are configurable; VIPP never updates a running environment automatically.

## A more comfortable workspace

- **Display settings** replaces the toolbar’s Preview menu, with clearer thumbnail, contrast, colour and input/output-label choices. Display changes apply immediately without Calculate and do not change scientific pixels or processing.
- **Find nodes using familiar words.** Palette, insert-node and workflow searches recognize alternatives such as dilate, erode, thinning, NLM, clipping and British spellings. Search fields have clearer icons and more room; Ctrl+F focuses workflow search when the host has not claimed the shortcut.
- **Choose channel presentation explicitly.** Image Source offers **Stack (C slider)** or **Separate coloured layers**. Colour edits preserve that choice. Interleaved microscope axes such as ZCYX display spatial Z/Y/X correctly while retaining working channel navigation and unchanged underlying data.
- **Inspect and navigate with less clutter.** Compact histograms show cached bin counts and ranges on hover. Table- and mesh-only cards omit empty image placeholders. Workflow-tab menus can reveal the saved file in Explorer, Finder or the containing folder without switching tabs or saving edits.

## Thresholding, intensity and mask morphology

- **Binary Threshold gains four foreground choices:** strict **Above** and **Below**, inclusive **In range**, and strict **Outside range**. Linked bounds and draggable histogram guides support range selection; equal endpoints are included only by In range, and NaN stays background. Supported CuPy execution uses the same comparison rules.
- **Rescale Intensity adds Invert intensity.** Output bounds remain ordered; inversion reverses their mapping explicitly. Older reversed bounds are retained on load but require correction before calculation rather than being silently swapped.
- **Safer numeric editing.** Odd-only windows and PSF sizes, paired bounds, percentages, counts and small positive minima retain their constraints. Fractional mesh settings no longer round to whole numbers. Editing one bound does not move its partner, and loading a workflow does not silently repair its parameters.
- **New Convex Hull.** Fill one Boolean hull per 2D YX slice or 3D ZYX volume, with independent time/channel blocks and support for sparse or flat masks. One hull can join separate foreground objects; it is not a per-label operation.

## Create, manage, refine and export mesh objects

The new **3D Meshes** family works with explicit, calibrated surfaces. Mask and label extraction use full-resolution Lewiner marching cubes without implicit smoothing or simplification. Select one Z/Y/X volume before extraction; these manual/cached operations run on CPU.

- **Create and organize objects.** Labels retain their IDs; masks can form one object or separate connected objects. Colour by identity, volume, surface area, sphericity or triangle count. Combine retains separate objects rather than performing registration or geometric union; split separates disconnected surfaces, including cavity walls.
- **Filter and refine deliberately.** Filter by measurement or ID, using an input-object histogram with units and linked limits. Smooth Mesh and Simplify Mesh produce new geometry while preserving the input. Fractional controls, boundary options and actionable failure messages help tune results; refinement can change shape, volume and measurements.
- **Measure the resulting surface.** Measure 3D Mesh Morphology accepts Boolean masks and existing meshes as well as labels. Existing triangles are measured directly on CPU, one row per object, without remeshing or inventing voxel counts. Open or invalid surfaces retain available measurements but report unavailable volume-derived fields.
- **Export through the inspector, Save Image or Batch Output.** OBJ retains geometry and object groups, with VIPP metadata in comments. 3MF carries physical units, separate objects and display colours; uncalibrated data requires explicit calibration. Connected-input format menus prevent silently retaining an incompatible format. Neither format certifies a repaired or print-ready solid.
- **Two bundled examples** cover mesh objects, colours and refinement, including **Mesh Objects — Tuned Refinement** with authored layout, triangle-count colours and refinement settings. They create no files automatically.

## Upgrading and limits

Two compatibility fixes protect saved analysis records: legacy threshold and rescaling defaults no longer cause a false batch-workflow hash mismatch when a workflow is restored, and Binary Threshold history now records both range bounds with the correct inclusive/exclusive endpoint rules. Non-default authored values remain part of workflow identity.

Workflow schema 6, batch configuration 6 and manifest schema 5 remain unchanged. Supported older workflows retain migration paths; new operations and parameters require this release. Preserve old records, review threshold and rescaling choices, recalculate representative results, and regenerate Python exports for the exact installed VIPP version.

Large images and full-resolution meshes can require substantial memory. Cancellation is cooperative and may wait for the current library call. Mesh extraction, object management and refinement are not GPU operations; the existing label-morphology GPU path remains hybrid. GPU availability depends on supported input, parameters, memory and environment, not merely the selected compute preference. This release does not add general lazy execution, Apple GPU support or broad biological validation.

## Install and read more

Use the [canonical 0.15.0a2 GitHub release](https://github.com/rensutheart/napari-vipp/releases/tag/v0.15.0a2) for assets, checksums and qualification evidence, or [PyPI 0.15.0a2](https://pypi.org/project/napari-vipp/0.15.0a2/) for an exact package pin. Windows installers are explicitly unsigned; separate Apple Silicon and Intel macOS packages are unsigned, unnotarized and CPU-only. Verify the matching checksum before installation.

In a dedicated supported Python environment on Windows or Linux:

```bash
python -m pip install "napari[pyqt6]>=0.6" "napari-vipp==0.15.0a2"
vipp
```

On macOS, use `"napari[pyside6]>=0.6"` instead of `"napari[pyqt6]>=0.6"`. Follow the [0.15.0a2 manual](https://rensutheart.github.io/vipp-mkdocs/0.15.0a2/) for installation, reader diagnostics, mesh workflows and scientific boundaries.

[Full source comparison: 0.15.0a1 → 0.15.0a2](https://github.com/rensutheart/napari-vipp/compare/v0.15.0a1...v0.15.0a2)
