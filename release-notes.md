# VIPP 0.13.0a8

VIPP 0.13.0a8 makes interactive GPU workflows more responsive and easier to
trust, adds Fiji-style binary outlier cleanup, improves everyday graph and
intensity editing, and removes the separate cuCIM installation route. Every
reviewed GPU implementation in this release is available through the normal
CuPy/CuPyX CUDA installation.

This remains alpha software. Keep the original data and workflow, test
representative images, and review important outputs before using them for
scientific conclusions or publication. A CPU/GPU agreement check establishes
agreement with VIPP's CPU implementation in a stated region; it does not by
itself validate the input data, parameters, PSF, or biological interpretation.

## Features added

### Clean fuzzy binary edges with Remove Outliers

- Added **Remove Outliers (Binary)**, aligned with ImageJ/Fiji's circular
  neighborhood and nearest-edge behavior. It can remove isolated foreground
  specks or fill small background notches on each YX plane.
- The node accepts Boolean masks and canonical `uint8` masks containing 0/1 or
  0/255. Invalid grayscale inputs fail visibly instead of being silently
  thresholded.
- The authoritative CPU implementation has an exact resident CuPy candidate,
  so **Find fastest pipeline** can compare both implementations for the active
  image and parameters.

### Use one normal CUDA installation

- **Measure Objects** and **Measure Objects + Intensity** now use resident CuPy
  implementations. A separate cuCIM source build, add-on ZIP, and provider
  installer are no longer needed or shipped.
- Exact saved `cucim-measure-objects-basic-v1` and
  `cucim-measure-objects-intensity-basic-v1` pins migrate to their corresponding
  `cupy-*` implementation IDs. A broad saved `library:cucim` preference remains
  visibly unavailable because it does not identify one unambiguous replacement.
- Preserved RTX 5090 evidence records 11 admission cases, 11 rejection cases,
  two lifecycle cases, and 15 performance cases for the CuPy replacement. CuPy
  won all 14 matched transfer-inclusive comparisons with the historical cuCIM
  implementation in that bounded environment. These measurements are evidence
  for that machine and workload, not a portable speed guarantee.

### Interpret unknown TIFF pages deliberately

- **Image Source** now shares the batch workspace's reviewed **Image stack**
  control. Choosing `QYX -> ZYX` reinterprets pages as depth without transposing
  pixels and remains intact through undo, workflow save/load, export, batch, and
  headless execution.
- **Rescale Axes** exposes Z scaling and output size after that explicit
  declaration. It can also use unique carried X/Y names inferred from shape,
  with a visible warning and explicit output provenance.
- VIPP never silently assumes that an unknown Q axis is depth. Inferred or
  missing Z still requires the source declaration before Z can be resized.

## Workflow and control improvements

- Dropping a disconnected compatible node onto a green-highlighted wire now
  inserts it between the wire's source and target as one undoable edit.
- Every numeric **Intensity & Contrast** node now shows the shared exact input
  histogram as well as its output histogram.
- Clip bounds, Rescale Intensity output bounds, and Mask Image's outside value
  use whole-number controls for integer images. Floating-point images retain
  fractional controls, and invalid legacy values remain visible for correction.
- Parameter sliders can use a practical window without narrowing direct numeric
  entry. Sigma Filter now has a useful `0..10` Sigma-width slider while its
  numeric field still accepts the full supported range.

## Faster, more stable GPU interaction

- Gaussian Blur, Rolling-Ball/Subtract Background, and Median Filter now use
  radius- or size-independent CuPy kernels. Tuning to a previously unseen
  supported value no longer creates a parameter-specific compilation pause.
- Thumbnail contrast work keeps the last complete preview visible while new
  statistics are calculated, cancels stale work promptly, and reports concise
  CPU/GPU/fallback/cached status without losing detailed provenance.
- Qualified `float32` percentile and min-max thumbnail statistics can run on
  CuPy with exact CPU agreement. Large retained GPU-resident outputs can avoid
  a redundant thumbnail upload in the reviewed region.
- Compute Setup can bind each workflow tab to Automatic or one exact qualified
  runtime/device on the current machine. This machine choice is intentionally
  excluded from portable workflow files and undo history.
- The exact CuPy 14.1.1 `cupyx.jit.rawkernel is experimental` FutureWarning is
  suppressed only around the known lazy CuPyX import. Other warnings and all
  import or execution failures remain visible.

## Optimizer fixes

- **Find fastest pipeline** now separates genuine small numerical CPU/GPU
  differences from assignment failures and presents reviewed differences for
  explicit acceptance. Shape, dtype, non-finite classification, and larger
  differences still fail closed.
- Save Image and Batch Output nodes may remain in the requested retention scope,
  but detached scientific analysis excludes writers so benchmarking cannot
  create files.
- Applying measured assignments no longer treats the fixed CPU Image Source row
  as a changed runtime assignment. Every executable node still has to match its
  accepted measured backend.

## Windows installation

The official `v0.13.0a8` GitHub prerelease and checksum sidecars are public.
Use only that release surface; do not treat a guessed asset URL as a download.

For the shortest route, download
`VIPP-Setup-0.13.0a8-Windows-x86_64-UNSIGNED.exe` and
`SHA256SUMS-Windows-0.13.0a8.txt` from that release. Verify the SHA-256 value
before opening the installer. There is no a8 cuCIM add-on asset: the normal
CUDA installation contains every current reviewed CuPy/CuPyX implementation.

This alpha is intentionally not Authenticode-signed. Windows will show
**Unknown publisher** and may show **Windows protected your PC**. After
verifying the official checksum, select **More info > Run anyway**. Stop if the
checksum differs or antivirus reports a threat; never disable Windows security.
If organizational policy does not allow the unsigned installer, use the manual
installation route in the Quick Start.

The managed installer can keep CPU and compatible NVIDIA CUDA 13 installations
side by side, create launch shortcuts, repair or update an owned installation,
and remove it without touching unrelated Python or napari environments.
One-click setup continues to accept only the fixed CPU and CUDA roots beneath
canonical Windows Local App Data. The complete CUDA path must contain ASCII
characters because of the pinned CuPy 14.1.1 NVRTC path boundary; spaces remain
supported.

## Manual installation or upgrade

VIPP supports CPython 3.12 and 3.13 for CPU use. In PowerShell, Command Prompt,
or a terminal with the intended environment activated:

```text
python -m pip install --upgrade "napari[pyqt6]>=0.6" "napari-vipp==0.13.0a8"
vipp
```

For the supported native-Windows CUDA 13 route, use CPython 3.12 and follow the
versioned Windows CUDA guide instead of mixing CUDA packages manually. Preserve
the old environment and copies of important workflows first. This release
continues to read workflow schema 3 and writes schema 4; batch configuration and
manifest schema remain version 3.

When moving from 0.13.0a7, use the normal a8 installer or create a clean,
version-pinned environment. Do not reuse the a7 cuCIM add-on bundle or copy its
private wheel into a8. Existing workflows with exact cuCIM measurement pins are
migrated as described above; VIPP no longer imports or executes cuCIM providers.

## Qualification status

The CuPy measurement replacement and its preserved comparison artifact have
passed the bounded scientific, lifecycle, and performance checks summarized
above. The changed Remove Outliers, source-axis, rescaling, optimizer, graph,
control, thumbnail, installer, packaging, and documentation paths have focused
automated coverage in the source tree.

The GPU qualification source at `7189cf40280d` passed all 13 jobs in
[CI run 32578260799](https://github.com/rensutheart/napari-vipp/actions/runs/32578260799),
including clean wheel and source-archive installs across Windows, Linux, and
macOS. Its normal unsigned-installer build smoke also passed in
[run 32578260812](https://github.com/rensutheart/napari-vipp/actions/runs/32578260812).

The same clean source passed the full RTX 5090 admission catalogue: 19
public implementations and all 24 evidence owners completed successfully on
`cuda:0`. The retained
[aggregate](docs/benchmarks/gpu-admission-0.13.0a8-windows-rtx5090.json)
has SHA-256
`0365366dc23750e000c6e9c4f8b384cdf706afdcb338ae3a9f80cfad3d1d8506`.
The refreshed Remove Outliers and CuPy measurement reports are linked from the
architecture and GPU guides. Production GPU code was unchanged between that
source and the final release commit; the later production delta was limited to
deterministic source-archive canonicalization.

The tagged release commit
`5a66ae9d1098ca5a8d409a4075c585692e3c3638` passed all 13 jobs in
[exact-main CI run 32584690313](https://github.com/rensutheart/napari-vipp/actions/runs/32584690313),
and its normal installer smoke passed in
[run 32585512509](https://github.com/rensutheart/napari-vipp/actions/runs/32585512509).
The final CUDA installer updated an owned a7 environment to a8 with no retired
cuCIM/nvimgcodec residue, healthy dependencies, Doctor 19/19, qualified CuPy
measurements, and the a8 GUI. The exact six GitHub assets and matching PyPI
wheel/source bytes are recorded in the
[release qualification baseline](docs/release-qualification-baseline.md).

See the [Quick Start](https://github.com/rensutheart/napari-vipp/blob/v0.13.0a8/docs/quick-start.md),
[GPU Guide](https://github.com/rensutheart/napari-vipp/blob/v0.13.0a8/docs/gpu-guide.md),
[full changelog](https://github.com/rensutheart/napari-vipp/blob/v0.13.0a8/CHANGELOG.md#0130a8---2026-08-22),
and [roadmap](https://github.com/rensutheart/napari-vipp/blob/v0.13.0a8/docs/planning.md)
for the complete delivered scope and remaining product milestones.
