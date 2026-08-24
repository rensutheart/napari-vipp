# VIPP 0.13.0a9

VIPP 0.13.0a9 is a focused correctness release for workflows that combine
declared 3D image axes, CPU-only processing steps, reviewed GPU providers, and
multiple downstream measurements. It packages the recent fixes found while
testing real microscopy workflows rather than waiting for the broader 0.14
feature line.

This remains alpha software. Keep the original data and workflow, test
representative images, and review important outputs before using them for
scientific conclusions or publication.

## Prefer GPU now survives CPU-only nodes

- **Prefer GPU** planning now preserves exact shape, dtype, and axis facts
  through CPU-only operations including Rescale Axes, Rescale Intensity, and
  Unsharp Mask.
- A reviewed downstream CuPy or CuPyX provider is no longer deferred to CPU
  merely because an earlier node must execute on the host.
- The requested global compute intent remains distinct from saved per-node
  preferences, and CPU-only nodes continue to execute on CPU as intended.

## More reliable 3D workflows

- Changing Image Source from `QYX` to `ZYX` immediately propagates the effective
  axes through the active branch, allowing Gaussian Blur 3D to expose and retain
  Sigma Z without waiting for a successful full pixel run.
- Skeletonize now treats declared ZYX data as a real volume. Auto selects Lee
  thinning, leading dimensions are processed as independent ZYX blocks, and the
  resolved method and dimensionality are retained in provenance. Zhang remains
  explicitly limited to 2D.
- Ambiguous page axes continue to fail closed until their spatial meaning is
  declared; VIPP does not silently reinterpret an unknown page axis as depth.

## Measurements remain usable together

- Re-materializing the same unchanged file no longer creates a different source
  identity simply because its array wrapper is new.
- Sequential sibling measurements can therefore remain ready together and feed
  Merge Tables in either execution order, including after low-memory cache
  pruning. Genuine upstream changes still invalidate affected descendants.

## Actionable GPU-memory errors

- GPU admission failures identify the CUDA device and every affected graph
  node.
- Estimated peak use, available memory, and the shortfall are displayed in
  readable MiB or GiB while exact byte counts remain available in structured
  diagnostics.
- The message distinguishes free-VRAM reserve from a configured limit and
  suggests concrete ways to make the graph fit.

## Windows installer and compatibility

The normal Windows installer continues to offer CPU and managed NVIDIA CUDA 13
routes from one executable. The CUDA route includes the complete pinned CuPy,
CUDA runtime, cuBLAS, cuFFT, and related dependency set; no separate GPU add-on
is required.

Pre-publication installer candidates are deliberately labelled
**DEVELOPMENT BUILD — local testing only**. A final alpha installer is produced
only from a clean immutable `v0.13.0a9` tag and is published together with its
SHA-256 checksum and release manifest.

Workflow schema remains version 4, and batch configuration and manifest schema
remain version 3. Existing 0.13 workflows remain supported. Preserve copies of
important workflows before updating a managed installation.

## Qualification scope

This alpha changes core/UI, workflow-planning, and GPU/scientific behavior. Its
acceptance focuses on the four reported workflow regressions, exact metadata
propagation through CPU-only boundaries, real-device execution of the affected
CuPy/CuPyX nodes, and an update using the newly built Windows installer. The
unchanged installer transaction model and dependency pins carry forward from
the qualified 0.13.0a8 baseline.
