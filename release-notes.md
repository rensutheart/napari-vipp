# VIPP 0.16.0a2

VIPP 0.16.0a2 adds calibrated image registration and whole-volume time-lapse drift correction, label-preserving skeleton measurements, and faster exactly compatible CPU median filtering.

**Alpha software.** Preserve original images, saved workflows and decisive outputs. Check representative data before scientific interpretation.

## Image registration and comparison

- Estimate Registration supports translation, rigid and advanced affine models for 2D/3D image pairs or fixed-reference time series. Each time point moves as one complete spatial volume; a selected estimation channel determines the transform shared across channels.
- Apply Transform resamples onto an explicit calibrated reference grid and returns a valid-coverage mask. Labels retain their original IDs with exact nearest-neighbour sampling; intensity interpolation and other scientific choices remain explicit.
- Review diagnostics and the transform summary together in the inspector. Export the diagnostic table and reusable transform JSON separately or together. The Next step suggestion adds a connected Apply Transform as one undoable edit, or locates an existing matching node, without calculating automatically.
- Compare Images reports same-grid RMSE, correlation, SSIM and PSNR with explicit range and coverage handling. Three synthetic examples demonstrate translation, anisotropic 3D rigid alignment and multichannel whole-volume time-lapse correction.

## Preserve object identity through skeleton analysis

- Skeletonize Labels thins each object independently while preserving its label ID. Analyze Skeleton per Label reports object-level measurements and component details, including isolated voxels and objects with no remaining skeleton.
- Join skeleton and morphology measurements using original labels and leading-axis identity. A calibrated Per-label Skeleton & Morphology example demonstrates the connected workflow. The existing binary skeleton/component workflow remains available.
- Automatic joins include valid shared custom axis-index columns, preventing repeated label IDs at different image positions from being combined accidentally.

## Performance and desktop refinements

- Qualified CPU Median Filter calls use SimpleITK 2.5.6 while preserving the existing XY-only footprint, reflected borders and dtype. Other input domains retain SciPy, and GPU implementations are unchanged. Local qualified warm benchmarks measured 2.02–6.41× speedups; this is not a promise for every image or computer.
- Closed inspector dropdowns no longer change parameters when the mouse wheel passes over them. The inspector keeps scrolling; opened lists and keyboard selection retain their normal behavior.
- Logarithmic plot tick labels render as powers of ten in previews and exported figures, while titles and category labels remain literal text.

## Scope and compatibility

Registration is CPU-only and globally aligns shared structure; it does not establish biological accuracy or solve arbitrary deformation. Inspect diagnostics, coverage and representative results. Deformable registration, template matching, previous-frame chaining and GPU registration are not included. Gaussian filtering, background subtraction and deconvolution methods are unchanged. Statistics remains descriptive; no hypothesis tests or significance analysis are added.

New nodes and parameters require the new version; retain copies before resaving older workflows. Reproduction version acknowledgements and verified-resume restrictions remain in force. macOS remains CPU-only. Desktop installers are unsigned alpha builds, and macOS packages are unnotarized.

See the [qualification declaration](https://github.com/rensutheart/napari-vipp/blob/v0.16.0a2/docs/release-qualification-baseline.md), [registration evidence](https://github.com/rensutheart/napari-vipp/blob/v0.16.0a2/docs/registration-synthetic-qualification.md) and [median qualification](https://github.com/rensutheart/napari-vipp/blob/v0.16.0a2/docs/simpleitk-cpu-qualification.md) for the scope of scientific checks. Verify platform downloads using the release checksum files.
