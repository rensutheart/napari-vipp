# VIPP 0.13.0a7

VIPP 0.13.0a7 makes GPU acceleration easier to understand, inspect, and apply. It can explain when a visible dtype conversion is the only thing preventing a node from using a reviewed GPU implementation, add that ordinary conversion in the correct place with one click, and keep the change fully reviewable and undoable.

This remains alpha software. Keep the original data and workflow, test representative images, and review important outputs before using them for scientific conclusions or publication. A GPU agreement check establishes agreement with VIPP's CPU implementation in a stated region; it does not by itself validate the input data, parameters, PSF, or biological interpretation.

## Features added

### A clearer path from a GPU blocker to a safe repair

- Nodes now show a subtle **GPU tip** when input dtype is the only remaining blocker for an otherwise reviewed GPU implementation.
- Selecting the node explains the exact conversion and memory trade-off. **Add conversion** inserts a normal, visible **Convert Dtype** node on the affected input only. The insertion is one Undo action and does not silently alter shared branches or authored processing parameters.
- The tip remains available in **Prefer GPU** after calculation, rather than disappearing when a CPU result has been cached.
- The lossless `uint8`/`uint16` to `float32` **Preserve** conversion itself can remain GPU-resident with the following reviewed operation, avoiding an unnecessary device round trip.

### Optimizer results that can actually be read

- **Find fastest pipeline** now keeps completed scientific and timing results visible even when CPU and GPU are too close to name a safe winner. In that case VIPP leaves the saved backend unchanged and explains why.
- Results are grouped by node, with one subrow per tested CPU, CuPy, or cuCIM implementation. The main view emphasizes total time, scientific agreement, and outcome; optional detail shows compute, transfer, first-run cost, memory, and evidence information.

### A connected GPU segmentation path

- Added reviewed GPU implementations for scalar `float32` **Binary Threshold** and explicitly described **Extract Channel** inputs.
- Added reviewed boolean-mask GPU implementations for **Remove Small Objects** in resolved 2D/3D Face/Full connectivity regions and **Fill Holes** when `max_hole_size` is zero.
- The annotated **Portable GPU Segmentation Bridge** example connects channel selection, dtype conversion, Gaussian blur, thresholding, mask cleanup, and connected components. It remains usable on CPU when an accelerator or an exact GPU region is unavailable.
- Integer-label small-object cleanup and positive bounded-hole-size cleanup deliberately remain on CPU with a visible explanation.

### Broader Richardson-Lucy agreement without parameter rewriting

- Ordinary Richardson-Lucy GPU eligibility now covers finite authored `filter_epsilon` values from `1e-12` through `1e-6` and 1 through 100 iterations in the reviewed odd-PSF/default-safe region. Lambda-zero RL-TV inherits that region; positive-TV runs retain their narrower reviewed points.
- CPU/GPU comparison now uses a documented 0.5% agreement policy with finite, nonnegative outputs and matching shape/dtype requirements. The former much tighter near-identity observations remain useful diagnostics rather than an arbitrary scientific validity boundary.
- VIPP never changes the authored epsilon or iteration count merely to qualify a GPU call. The bundled 3D RL/RL-TV example keeps 25 iterations and `filter_epsilon=1e-12` on both branches.

## Bug fixes

- Ordinary finite decimal Binary Threshold values no longer look GPU-ineligible merely because whole-pipeline timings cannot justify a backend change.
- GPU conversion suggestions now fail closed if their saved candidate no longer exists, the input changed, or Custom mode explicitly selected another implementation.
- Conversion repairs placed through a named tunnel stay beside the affected subscriber instead of moving an unrelated source or branch.
- A scientifically successful but speed-inconclusive optimizer run is no longer presented as an eligibility failure with its measurements hidden.

## Windows installation

The a7-specific links and downloads are valid only after the official `v0.13.0a7` GitHub prerelease is published. Until that release and its checksum sidecars exist, use the public 0.13.0a6 release or an explicitly marked development checkout; do not treat a guessed a7 asset URL as a download.

For the shortest route, download `VIPP-Setup-0.13.0a7-Windows-x86_64-UNSIGNED.exe` and `SHA256SUMS-Windows-0.13.0a7.txt` from this GitHub release. Verify the SHA-256 value before opening the installer.

This alpha is intentionally not Authenticode-signed. Windows will show **Unknown publisher** and may show **Windows protected your PC**. After verifying the official checksum, select **More info > Run anyway**. Stop if the checksum differs or antivirus reports a threat; never disable Windows security. If organizational policy does not allow the unsigned installer, use the manual installation route in the Quick Start.

The managed installer can install CPU or compatible NVIDIA CUDA 13 environments, keep CPU and GPU installations side by side, create launch shortcuts, repair or update an owned installation, and remove it without touching unrelated Python or napari environments. The standard CUDA installation includes the reviewed CuPy/CuPyX route and works without optional cuCIM.

One-click setup accepts only the exact per-track roots beneath the canonical Windows Local App Data directory returned by `SHGetKnownFolderPath(FOLDERID_LocalAppData)`: `VIPP\environments\cpu` and `VIPP\environments\cuda13`. Custom managed roots are not accepted. CUDA requires the complete canonical path to contain ASCII characters in this release. If it does not, one-click CUDA is unavailable and the UI offers CPU; the fixed CPU root remains supported. Expert-selected existing environments remain separate and unchanged.

The CUDA root and Windows temporary directory are separate. If Python's effective temporary directory contains a non-ASCII character, VIPP uses process-local in-memory CuPy compilation. CuPy's disk kernel cache is then off for that process, so Compute Doctor or the first GPU work may pay the cold compilation cost again in a new process; scientific kernels and results are unchanged. One RTX 5090 reference check took about 52 seconds cold and about 0.87 seconds when refreshed in the same process; those observations are not a performance guarantee. A failed kernel compile now preserves the real CuPy `CompileException` instead of masking it as a false 512-byte private-pool leak.

## Optional cuCIM

After a standard CUDA installation passes Compute Doctor, users who need the cuCIM-backed operations can download `napari-vipp-cucim-installer-0.13.0a7-windows.zip` from the same release. The bundle contains no cuCIM wheel: it builds the pinned source locally, verifies the result, and installs the private wheel only into the selected VIPP environment. Rebuild it with the matching a7 bundle after upgrading; do not reuse an a6 private wheel or copy one between environments.

## Manual installation or upgrade

VIPP supports CPython 3.12 and 3.13 for CPU use. In PowerShell, Command Prompt, or a terminal with the intended environment activated:

```text
python -m pip install --upgrade "napari[pyqt6]>=0.6" "napari-vipp==0.13.0a7"
vipp
```

For the supported native-Windows CUDA 13 route, use CPython 3.12 and follow the versioned Windows CUDA guide instead of mixing CUDA packages manually. Preserve the old environment and copies of important workflows first. This release continues to read workflow schema 3 and writes schema 4; batch configuration and manifest schema remain version 3.

When upgrading an installer-managed 0.13.0a6 installation, run the a7 installer, select the same CPU or CUDA route, review the copy detected at that track's fixed root, and let setup retain the old working copy until a7 passes its checks. CPU and CUDA tracks can coexist, but two managed versions of the same track cannot. An installer-owned CUDA copy under an incompatible root is not moved or repaired in place; after any separately recorded recovery from a prior interrupted transaction, the newly blocked selection performs no new mutation. Remove that copy only through its ownership-bound uninstaller. Do not point setup at an unrelated manually managed napari environment.

## What we validated

The complete source suite for the release candidate passed with 5,084 tests passing, five environment-dependent tests skipped, and two documented expected failures. All 13 pull-request CI jobs passed across Windows, Linux, and macOS on supported Python versions, including clean wheel and source archive installation. The strict admission catalogue accounts for 18 public GPU implementations and 23 executable evidence owners.

RTX 5090 development evidence covers the connected segmentation corridor with one upload and one terminal download when only the final result is retained, plus exact CPU agreement in the admitted mask-cleanup regions. This does not claim qualification on every NVIDIA model or platform. The public support matrix remains scoped, and exact tagged-installer acceptance is a separate release gate that editable-checkout or pull-request testing cannot satisfy.

See the [Quick Start](https://github.com/rensutheart/napari-vipp/blob/v0.13.0a7/docs/quick-start.md), [GPU Guide](https://github.com/rensutheart/napari-vipp/blob/v0.13.0a7/docs/gpu-guide.md), [full changelog](https://github.com/rensutheart/napari-vipp/blob/v0.13.0a7/CHANGELOG.md#0130a7---2026-08-14), and [roadmap](https://github.com/rensutheart/napari-vipp/blob/v0.13.0a7/docs/planning.md) for complete scopes and remaining qualification work.
