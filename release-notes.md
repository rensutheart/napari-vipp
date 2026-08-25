# VIPP 0.13.0a9

VIPP 0.13.0a9 is a focused correctness release for workflows that combine
declared 3D image axes, CPU-only processing steps, reviewed GPU providers, and
multiple downstream measurements. It packages the recent fixes found while
testing real microscopy workflows rather than waiting for the broader 0.14
feature line.

This remains alpha software. Keep the original data and workflow, test
representative images, and review important outputs before using them for
scientific conclusions or publication.

## Prefer GPU planning preserves eligibility across CPU-only nodes

- **Prefer GPU** planning now preserves exact shape, dtype, and axis facts
  through CPU-only operations including Rescale Axes, Rescale Intensity, and
  Unsharp Mask.
- A reviewed downstream CuPy or CuPyX provider is no longer deferred to CPU
  merely because an earlier node must execute on the host.
- The requested global compute intent remains distinct from saved per-node
  preferences, and CPU-only nodes continue to execute on CPU as intended.
- GPU execution still requires a healthy, compatible runtime and installed
  provider. Compute details report the exact eligibility or fallback reason
  when a machine cannot use an otherwise reviewed GPU implementation.

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

VIPP 0.13.0a9 includes its own Windows installer:
`VIPP-Setup-0.13.0a9-Windows-x86_64-UNSIGNED.exe`. It offers CPU and managed
NVIDIA CUDA 13 routes from one executable. The CUDA route includes the complete
pinned CuPy, CUDA runtime, cuBLAS, cuFFT, and related dependency set; no
separate GPU add-on is required.

The published `0.13.0a9` installer is the release-specific build created from
the clean immutable `v0.13.0a9` tag. It is intentionally unsigned, so Windows
displays **Unknown publisher**; its SHA-256 checksum and release manifest are
supplied alongside it.

Workflow schema remains version 4, and batch configuration and manifest schema
remain version 3. Existing 0.13 workflows remain supported. Preserve copies of
important workflows before updating a managed installation.

## Qualification scope

This alpha changes core/UI, workflow-planning, and GPU/scientific behavior. Its
acceptance focuses on the four reported workflow regressions, exact metadata
propagation through CPU-only boundaries, real-device execution of the affected
CuPy/CuPyX nodes, and exact-tag Windows installer integrity. The unchanged
installer transaction model and dependency pins carry forward from the
qualified 0.13.0a8 baseline.

The release-domain declaration is:

- changed: core/UI, workflow/provenance, GPU/scientific planning, and
  documentation;
- unchanged: installer transaction behavior, dependency/toolchain pins, and
  package/release infrastructure; and
- carried forward: the 0.13.0a8 installer lifecycle, dependency, and packaging
  baselines, while exact a9 tag, artifact, checksum, and publication facts are
  regenerated.

On the release workstation, the original student workflow completed under
global Prefer GPU across CPU-only Rescale Axes, Rescale Intensity, and Unsharp
Mask boundaries. Subtract Background, Convert Dtype, Gaussian Blur, Otsu,
Remove Outliers, and Remove Small Objects executed on `cuda-cupy` with no
fallback. This bounded result verifies the planner and reviewed providers on
that environment; it is not a claim that every separate installation has a
healthy CUDA runtime.
