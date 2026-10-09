# VIPP 0.16.0a4

VIPP 0.16.0a4 adds linked, read-only image review, a bounded ImageJ-compatible Gaussian filter, clearer Batch/Results controls and Windows workspace-window fixes, plus OME export and loading compatibility repairs.

**Alpha software.** Preserve original images, saved workflows and decisive outputs. Check representative data before scientific interpretation.

## Compare calculated images without changing the analysis

- Review Images opens one or two already calculated, explicitly declared scalar, RGB/RGBA, mask or label inputs in linked side-by-side or overlay views. Both inputs must share their physical/time grid; the node never silently resamples, reorders, normalizes or casts scientific arrays.
- Choose each pane's content, independent scalar black/white levels, colours, opacity and visibility from the inspector sidebar. Link viewpoints separately from shared time/slice navigation, use Oblique/XY/XZ/YZ and Fit controls, and toggle calibrated scale bars and 3D axes globally. Framed pane cards and compact status messages keep the display stable.
- Scalar 3D review offers maximum intensity, depth-weighted intensity and a display-only isosurface. Binary-mask surfaces use display interpolation, not scientific mask smoothing. These shaded modes are not quantitative intensity/colour-scale review or physically correct inter-layer occlusion. RGB 3D uses an explicitly labelled per-component MIP adapter; RGBA 3D and RACC custom-shader parity are not supported.
- Open reviews retain clearly marked previous results during upstream work, then replace the accepted same-grid A+B pair together while preserving viewpoints, navigation and the display recipe. Rejected or incomplete work stays noncurrent; changed grids, input kinds or input presence require explicit reopening.
- Display recipes persist with the workflow but do not enter scientific cache/hash identity. Review Images has no scientific output, batch output or generated-Python kernel. Low-memory mode retains only a visible review's directly bound inputs, not its entire upstream graph. Four synthetic workflows cover channels, calibrated masks, labelled time series and RGB plus a constructed scalar index.

## A separate ImageJ Gaussian option

- ImageJ Gaussian Blur reproduces the reviewed direct-convolution branch of ImageJ 1.54p for scalar uint8, uint16 and finite float32 YX planes, independently over leading planes, at sigma 0–8.5 pixels. Existing Gaussian Blur and Gaussian Blur 3D retain their behavior.
- The filter preserves source buffers, dtype, shape and calibration, with explicit history and conservative planning facts. Independent ImageJ bytecode fixtures match all 37 recorded cases bit-for-bit; this bounded evidence is not a guarantee for every input or platform. Larger sigma/downsampling, encoded RGB, physical-unit sigma and a true 3D ImageJ blur are unsupported. Numerical reproduction is not biological validation, and whole-image memory-bounded execution is not claimed.

## Batch, Results and workspace improvements

- Use consistent Select all / Deselect all controls across Batch items and Overrides, parameter columns, Results visible columns, Statistics measurements and Select Table Columns, retaining existing filtering and scientific selection rules.
- Blocked batch outputs explain duplicate names, input-file overlaps and overwrite protection. Find problem focuses the affected output node without calculating or saving; changed output settings require destination rechecking, and matching-item selection spans pages without dropping filtered-out selections.
- Detached VIPP, Batch and Results workspaces maximize or restore once and minimize independently from their Windows host, while keeping their workflow/data ownership. Reopening a minimized workspace restores its prior state. Native Linux/macOS window-manager ownership and minimization remain unqualified.
- Threshold sliders use the displayed float-image histogram while preserving saved cutoffs and the wider numeric-entry domain. Histogram guide edits remain synchronized after constrained or rounded changes. GPU-conversion advisories wrap fully, and a defensive napari layer-list decoration guard avoids errors from temporarily missing size hints.

## File compatibility and scope

OME-Zarr 0.4/0.5 image and analysis-dataset export supports the newer ome-zarr writer API while retaining display metadata, calibration and provenance. Legacy OME-TIFF spatial unit aliases micrometer/micrometre are normalized only in the parser copy; numeric calibration, pixels and original file bytes are unchanged. NRF Thuthuka Post PhD Track support is acknowledged in the repository.

Runtime/build dependencies, supported Python routes, installer engines and existing GPU kernels/admission regions are unchanged. Review Images affects shared execution, persistence and packaged resources, so exact integrated CI, installed/native checks, reproducible distributions and a bounded real-device interoperability gate remain release requirements; historical private development wheels are not release bytes. Earlier memory-pressure or incomplete full runs are not clean integrated qualification.

Synthetic and bounded native Windows checks do not establish acquired-microscopy accuracy, large-volume memory/performance, minimum-dependency coverage or native Linux/macOS renderer equivalence. Exact supported-platform CI and native artifact gates are recorded as they complete. macOS remains CPU-only. Desktop installers are explicitly unsigned alpha builds, and macOS packages are unnotarized.

Preserve workflow copies before resaving; new nodes and recipes require this version, and reproduction acknowledgements and verified-resume restrictions remain in force. See the [qualification declaration](https://github.com/rensutheart/napari-vipp/blob/v0.16.0a4/docs/release-qualification-baseline.md), [Review Images contract](https://github.com/rensutheart/napari-vipp/blob/v0.16.0a4/docs/image-review-implementation.md), [ImageJ evidence](https://github.com/rensutheart/napari-vipp/blob/v0.16.0a4/docs/validation/imagej-gaussian-compatibility.md) and [Windows workspace evidence](https://github.com/rensutheart/napari-vipp/blob/v0.16.0a4/docs/workspace-window-qualification.md). Verify downloads using their release checksum files.
