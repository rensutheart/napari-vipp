# cuCIM native-Windows source-build and benchmark evaluation

Initial evaluation: 2026-07-15
Phase 1 checksum/install refresh: 2026-07-27
Status: successful research build and public background candidate in the exact
validated environment; ordinary installation and wider-platform gates remain open

## Bottom line

No credible native-Windows cuCIM binary distribution was found. The official
packages remain Linux-only, the upstream Windows-support issue remains open,
and the current release has no GitHub binary assets. A source build is therefore
still necessary.

The useful part of cuCIM **can** be built for native Windows. The pinned
`v26.06.00` source produced this artifact and passed real RTX 5090 kernels:

```text
cucim_cu13-26.6.0-cp312-cp312-win_amd64.whl
```

On 2026-07-27 the builder was rerun with every compiler-side CUDA component
aligned to 13.2.86. The exact emitted wheel was checksum-verified and installed
through VIPP's setup helper into `.venv-gpu-cu13`; `pip check`, the CuPy/cuCIM
probes, all 98 background-adapter tests, and all 45 real RTX background cases
passed there. Integer public outputs are bitwise exact; float32 uses the
versioned bounded-error policy with exact shape, dtype, zero/sign, and
non-finite masks. This closes the Phase 1 local packaging/evidence slice, not
the public distribution or multi-platform admission gate.

This build provides `cucim.skimage` and `cucim.core`, including rolling ball,
filters, feature detection, labeling, measurements, morphology, restoration,
segmentation, and transforms. It does not provide `cucim.clara`: the native
`libcucim/_cucim` whole-slide image-I/O library was not ported or packaged.

The first benchmark gives a clear library-primitive answer:

- **Promising primitive coverage:** rolling ball (265-529x faster end-to-end),
  Canny (17x), region-properties tables (10x), connected components (2.8x),
  and Otsu thresholding (2.4x) versus the CPU primitive baseline.
- **Keep the existing CuPy path:** Gaussian, ordinary median, Sobel, and binary
  closing were within about 5-10% of their direct CuPy equivalents. The 31x31
  uint16 histogram median reached a clear 1.42x primitive win that still needs
  production-adapter crossover mapping.
- **Do not adopt cuCIM Richardson-Lucy for speed:** it took about 4.5x as long
  as the existing explicit CuPy/CuPyX loop in both 2D and 3D.

The evidence supports continuing cuCIM as a narrow provider for the operations
where it adds substantial value. It does not support replacing CuPy as the
primary array/runtime layer or importing all cuCIM APIs wholesale.

These timings compare optional-library primitives, not complete VIPP nodes.
Rolling-Ball/Subtract Background must additionally preserve VIPP's smoothing,
inversion, clipping, block/channel, non-finite, dtype-restoration, progress, and
metadata behavior. Canny, Otsu, connected-components, and measurement adapters
likewise need their complete VIPP parameter and output-schema contracts. The
"exact" labels below therefore describe only the recorded fixtures and cannot
admit a production implementation by themselves.

## Windows binary search

The search covered package indexes, release assets, the upstream issue tracker,
repository searches, and current forks.

| Source checked | Result |
|---|---|
| [PyPI `cucim-cu13` 26.6.0 JSON](https://pypi.org/pypi/cucim-cu13/26.6.0/json) | Eight wheels: CPython 3.11-3.14 for manylinux x86-64 and aarch64 only. No `win_amd64` file. |
| [RAPIDS conda channel](https://anaconda.org/channels/rapidsai/packages/cucim/overview) and nightly channel | Linux x86-64/aarch64 builds only. No Windows subdirectory. |
| PyPI package-name variants | No `cucim-windows`, `cucim-win`, `cucim-cu12-windows`, or `cucim-cu13-windows` project. |
| conda-forge and NVIDIA channels | No alternative native-Windows cuCIM package found. |
| [Upstream releases](https://github.com/rapidsai/cucim/releases) | No binary release assets. |
| [Upstream Windows compatibility issue #454](https://github.com/rapidsai/cucim/issues/454) | Still open; no linked Windows port or pull request. |
| GitHub repository/fork audit | No credible Windows-wheel repository was found. An API scan of all 83 current forks found no Windows/MSVC-named branch; this is supporting evidence, not proof that no private or obscure build exists. |

The [RAPIDS platform requirements](https://docs.rapids.ai/install/) continue to
direct Windows users to WSL. The absence of a third-party binary should be
rechecked before each pinned upgrade, but it is not a reason to defer the
source-built `skimage` experiment now that the pinned procedure is repeatable.

## What was built

| Item | Pinned value |
|---|---|
| cuCIM tag | `v26.06.00` |
| cuCIM commit | `3c15781c207eab93a317dd9803a6e726fe01f7c4` |
| Python | 3.12.9 |
| CuPy | `cupy-cuda13x[ctk] == 14.1.1` |
| CUDA toolkit package | `cuda-toolkit == 13.2.2` |
| CUDA compiler/runtime/CRT/NVVM/nvJitLink build components | `13.2.86` |
| nvImageCodec package | `nvidia-nvimgcodec-cu13 == 0.8.0.22` |
| CUDA driver/runtime APIs | 13.3 (`13030`) / 13.2 (`13020`) |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, 32 GiB |
| NumPy / SciPy / scikit-image | 2.5.1 / 1.18.0 / 0.26.0 |
| Wheel size | 8,654,879 bytes |
| Exact installed-artifact SHA-256 | `586D3443091EEA67CE2C697BE2C490CA51977A5DBDF894B9318B270977134CF8` |

The wheel is intentionally not committed. It is a locally built research
artifact without upstream Windows support. The machine-readable build record is
[`benchmarks/cucim-source-windows-rtx5090-build.json`](benchmarks/cucim-source-windows-rtx5090-build.json).

The digest identifies this exact artifact; it is not yet a deterministic-build
claim. The earlier July build had the same 8,654,879-byte size but a different
digest. Wheel archive normalization and a byte-for-byte two-build comparison
remain packaging work before publication.

### Required adaptations

The Python/skimage wheel required three small source adaptations:

1. `rapids-build-backend 0.4.1` invokes Unix `which`. Put Git for Windows'
   `usr/bin` directory on `PATH` during the build.
2. `python/cucim/src/cucim/VERSION` is a relative symbolic link in Git. A
   normal Windows checkout materializes the link target as text, so replace it
   with the root `VERSION` contents and explicitly include it in `MANIFEST.in`.
3. NumPy 2.5 deprecates assigning `ndarray.shape`. Replace the one occurrence
   in cuCIM's vendored padding code with `reshape`; otherwise cuCIM's strict
   pytest configuration reports 252 histogram-median failures because it turns
   the deprecation warning into an error.

No image-processing formula was changed. The reproducible implementation is
[`scripts/build_cucim_windows.ps1`](../scripts/build_cucim_windows.ps1). A clean
second build from a separate source checkout completed, installed the wheel, and
passed Gaussian, rolling-ball, and labeling kernels:

```powershell
.\scripts\build_cucim_windows.ps1
```

The refreshed builder also pins CUDA compiler, runtime, CRT, NVVM, and
nvJitLink build components to 13.2.86 and installs nvImageCodec 0.8.0.22. It
then installs the exact local wheel with `--no-deps` so pip cannot silently
re-resolve the admitted CUDA stack, and finishes with a real-GPU probe and
`pip check`. Installing nvImageCodec satisfies the upstream wheel metadata; it
does not create the absent native `cucim.clara/libcucim` extension.

## Verification

The initial July evaluation of the Windows wheel produced these upstream test
results (they were not rerun during the 2026-07-27 checksum/install refresh):

- complete `filters/tests/test_median.py`: **707 passed, 4 skipped**, with two
  expected warnings from tests that intentionally request an impossible CUDA
  block size and verify sorting fallback;
- selected Gaussian, rolling-ball, Richardson-Lucy, labeling, Canny, and
  region-properties tests: **172 passed, 8 skipped, 6 deselected**;
- skipped restoration cases required downloadable test data that was not
  available; rolling-ball `nansafe=True` is explicitly unsupported upstream.

Before the NumPy compatibility patch, the broader run was 641 passed, 12
skipped, and 252 strict-warning failures. This result is retained as a
maintenance warning: pinned cuCIM upgrades must be tested against the exact
NumPy version that VIPP will ship.

The Phase 1 application-environment verification, extended on 2026-07-28,
additionally produced:

- checksum-aware setup-helper install into `.venv-gpu-cu13`: **passed**;
- CuPy Gaussian/median and cuCIM rolling-ball setup probes: **passed**;
- final application-environment `pip check`: **passed**;
- complete `test_gpu_background.py`: **98 passed**; and
- real RTX subset: **45 passed**, covering `uint8`/`uint16`/`float32`, 2D/3D,
  leading blocks and channel axes, background/subtract operations, non-finite
  handling, exact integer output, bounded float32 parity, and common
  private-pool array-domain behavior.

## Benchmark method

[`scripts/benchmark_cucim.py`](../scripts/benchmark_cucim.py) separates:

- **optimization comparisons**, where cuCIM and a ready CuPy/CuPyX path both
  operate on device arrays; and
- **coverage comparisons**, where cuCIM supplies a GPU implementation but CuPy
  has no equivalent high-level API, so the current CPU scikit-image operation is
  the primitive baseline. It is not necessarily the complete VIPP adapter.

The standard profile used two warmups, five synchronized GPU repetitions, and
three CPU repetitions. Resident times exclude transfers. End-to-end times
include one host-to-device and one device-to-host transfer. The table reports
end-to-end medians; a speedup greater than 1 means cuCIM was faster. The first
call in a process is recorded separately, but the persistent CuPy compiler cache
is not cleared, so it must not be interpreted as clean-install JIT latency.

Full sizes, min/median/max ranges, resident timings, first-call timings, output
schemas, and numerical comparisons are in
[`benchmarks/cucim-source-windows-rtx5090-standard.json`](benchmarks/cucim-source-windows-rtx5090-standard.json).

## Standard benchmark results

| Primitive workload | Primitive baseline | Baseline ms | cuCIM ms | cuCIM speedup | Recorded fixture comparison |
|---|---|---:|---:|---:|---|
| Gaussian 2D | CuPyX Gaussian | 2.134 | 2.066 | 1.03x | exact |
| Gaussian 3D | CuPyX Gaussian | 3.054 | 3.224 | 0.95x | exact |
| Median float32 5x5, sorting | CuPyX median | 2.602 | 2.417 | 1.08x | exact |
| Median uint16 31x31, histogram | CuPyX median | 81.257 | 57.158 | 1.42x | exact |
| Sobel 2D | normalized CuPyX composition | 2.131 | 2.166 | 0.98x | allclose; max error below 1e-6 |
| Binary closing 2D | CuPyX binary closing | 0.900 | 0.887 | 1.01x | exact |
| Richardson-Lucy 2D, 15 iterations | explicit CuPyX loop | 3.680 | 16.764 | 0.22x | allclose; max error below 1e-6 |
| Richardson-Lucy 3D, 15 iterations | explicit CuPyX loop | 4.055 | 18.418 | 0.22x | allclose; max error below 1e-6 |
| Rolling ball 2D, radius 15 | scikit-image CPU | 950.982 | 3.582 | **265.46x** | exact |
| Rolling ball 3D, radius 5 | scikit-image CPU | 3062.291 | 5.793 | **528.66x** | exact |
| Connected components 2D | scikit-image CPU | 28.264 | 9.863 | **2.87x** | exact; both `int32` |
| Connected components 3D | scikit-image CPU | 9.954 | 3.504 | **2.84x** | exact; both `int32` |
| Canny 2D | scikit-image CPU | 81.676 | 4.797 | **17.03x** | exact |
| Otsu threshold 2D | scikit-image CPU | 8.313 | 3.490 | **2.38x** | exact; both `float32` |
| Region-properties table, 4096 objects | scikit-image CPU | 219.475 | 20.919 | **10.49x** | exact values; dtype caveat below |

These are single-host measurements, not portable thresholds. The large effects
justify wider multi-device testing; the near-1x results do not justify another
provider dependency on their own.

### Scientific and API caveats

- The benchmark's coverage outputs were value-exact for these fixtures. Sobel
  and Richardson-Lucy were numerically allclose rather than bitwise equal.
- `regionprops_table` returned the same values but different storage dtypes:
  CPU area/labels/bounds were `float64`/`int64`, while cuCIM returned
  `float32`/`uint16`/`uint32` for this fixture. A production adapter must either
  restore the CPU table schema and include that conversion in timing, or define
  and migrate a new public schema. It must also guard label-count overflow.
- The fast histogram median accepts dense rectangular footprints; a disk
  footprint is rejected. This matches VIPP's current square-size median node,
  but it is not a general replacement for arbitrary-footprint median filtering.
- cuCIM's Richardson-Lucy is convenient, but convenience is not a performance
  reason to use it here. VIPP's explicit loop also provides better progress and
  cancellation boundaries.
- First use and clean-install compilation still need dedicated measurement.
  Auto-selection must use warm operation-specific thresholds and account for
  whether neighboring nodes already keep the array resident.

## Recommendation

| Operation family | Decision from this host | Reason |
|---|---|---|
| Rolling ball/background subtraction | **Public candidate in the exact recorded environment** | The complete wrapper preserves smoothing, inversion, clipping, blocks/channels, non-finite handling, dtype restoration, cancellation boundaries, and metadata across 98 adapter tests and 45 real RTX cases. Integer output is exact and float32 uses bounded v2 parity. Unsupported environments and regions visibly remain on CPU; ordinary packaging and wider-platform qualification remain open. |
| Canny | **Raw cuCIM route rejected; exact CuPyX adapter implemented** | Later adversarial testing found raw cuCIM mask disagreements. VIPP now uses its exact CuPyX implementation and retains raw cuCIM only as rejected feasibility evidence. |
| Connected components | **Advance to VIPP-adapter validation** | Primitive output/schema matched with 2.8x benefit in both 2D and 3D. |
| Otsu threshold | **Exact CuPy adapter implemented** | GPU histogram/mask work plus the bounded authoritative host finalizer preserve VIPP's bool, integer, floating, non-finite, luma, scope, and exact-mask contract. |
| Region-properties table | **Advance with schema work** | 10x primitive benefit, but output dtypes and VIPP's production table schema need an explicit adapter/overflow policy. |
| Histogram median | **Map production crossover** | The 1.42x primitive result is worth retaining as a Selective candidate study; compare the complete adapter and neighboring-node residency before any Auto policy. |
| Gaussian, ordinary median, Sobel, binary morphology | **Keep CuPy** | No material cuCIM advantage on the tested workload. |
| Richardson-Lucy | **Keep explicit CuPy** | cuCIM was about 4.5x slower and offers worse progress/cancellation control. |
| `cucim.clara` image I/O | **Deferred from Phase 1; investigate soon** | Requires a separate native C++/codec port. A named feature-completeness/upstream review should decide a maintainable full-cuCIM path rather than normalize a permanently hobbled build. |

cuCIM should therefore remain an **optional implementation library on the CuPy
runtime**, not a replacement array runtime. Before release it still needs the Linux
comparison, a second Windows GPU tier, CUDA 12/13 policy, clean-install/JIT and
memory measurements, production-schema adapters, and a decision about whether
VIPP will maintain downstream Windows patches or seek their inclusion upstream.

macOS remains CPU-only for this NVIDIA-specific plan. Building the Python cuCIM
sources on Apple Silicon would not supply a CUDA runtime or execute these
kernels.

## Reproduce the benchmark

After the build script completes, use the isolated Python path printed in its
JSON result:

```powershell
$python = Join-Path $env:TEMP "napari-vipp-cucim-windows\venv\Scripts\python.exe"
& $python scripts\benchmark_cucim.py --profile smoke
& $python scripts\benchmark_cucim.py --profile standard --output docs\benchmarks\cucim-source-windows-rtx5090-standard.json
```

Install that exact local artifact into the dedicated application environment
through the checksum-verifying helper (substitute the digest printed by the
builder for every new artifact):

```powershell
$wheel = Join-Path $env:TEMP "napari-vipp-cucim-windows\cucim-v26.06.00\python\cucim\dist\cucim_cu13-26.6.0-cp312-cp312-win_amd64.whl"
python scripts\setup_gpu_dev.py --track cuda13 --venv .venv-gpu-cu13 `
  --cucim-wheel $wheel `
  --cucim-sha256 586D3443091EEA67CE2C697BE2C490CA51977A5DBDF894B9318B270977134CF8
```

That temporary interpreter validates the standalone build; it is not VIPP's
dedicated GPU development environment. The 2026-07-27 refresh separately used
the explicit setup-helper option to install the checksum-recorded wheel into
`.venv-gpu-cu13` and run the background-adapter suite there. Every replacement
artifact must repeat that separation and verification. The builder must never
silently mutate the application environment.

Clara remains explicitly outside Phase 1. The dedicated Windows port plan owns
the investigate-soon feature-completeness/upstream review; the desired end state
remains maintainable full cuCIM where feasible, not a permanently hobbled
skimage-only fork.

The source procedure is adapted from the upstream
[cuCIM contributor guide](https://github.com/rapidsai/cucim/blob/main/CONTRIBUTING.md#setting-up-your-build-environment),
which remains Ubuntu-tested rather than a declaration of native-Windows
support.
