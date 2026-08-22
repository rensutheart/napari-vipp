# VIPP GPU Guide

This page is the user-facing reference for GPU setup, compute modes,
eligibility, fallback, benchmarking, and reproducibility. Detailed scientific
contracts and implementation evidence remain in the linked technical records.

## Scope At A Glance

VIPP always retains the CPU implementation as the portable scientific
reference. GPU acceleration is optional and deliberately selective: each
operation must pass its datatype, dimensionality, parameter, dependency,
environment, memory, and scientific-parity gates. Work outside those gates
runs on CPU with an explanation; VIPP does not silently cast data or change an
authored parameter merely to use a GPU.

The public `0.13.0a8` CUDA route targets native 64-bit Windows and CPython
3.12. Linux and macOS continue to use the CPU path in this alpha while their
accelerator policies are qualified separately.

## Compute Modes

| Mode | Behavior |
| --- | --- |
| **Auto** | Chooses between reviewed CPU and GPU assignments using compatible machine-local evidence and workload gates. CPU can correctly win. |
| **CPU** | Uses the authoritative host implementations throughout the workflow. |
| **Prefer GPU** | Requests reviewed public GPU implementations without requiring them to beat CPU. VIPP can still keep a lightweight selection step on CPU when that avoids uploading data that would immediately be discarded; every such choice is shown. |
| **Custom** | Exposes per-node CPU/library choices, node benchmarking, locks, and **Find fastest pipeline…**. |

The badge on a completed node reports what actually ran, not merely what was
requested. Prefer GPU always allows visible CPU fallback. Custom can use the
configured visible or strict fallback contract where supported.

## Current Windows CUDA Qualification

The released public policy has no GPU-model allowlist. Device names are
recorded for provenance. Qualification instead requires all of the following:

- native 64-bit Windows;
- 64-bit CPython 3.12 with the CPython 3.12 ABI;
- an NVIDIA CUDA device with compute capability 7.5 or newer;
- CUDA runtime API 13.2 (`13020`) and NVIDIA driver API 13.3 (`13030`) or
  newer;
- the pinned NumPy 2.5.1, SciPy 1.18.0, and scikit-image 0.26.0 stack;
- CuPy/CuPyX 14.1.1 and the exact CUDA 13 component packages;
- successful synchronized runtime/provider probes and zero retained VIPP-owned
  device memory after the probe; and
- the operation-specific data, parameter, scientific, and memory gates.

The released runtime probes every visible CUDA ordinal before selecting its
default device. Consequently, every currently visible GPU must meet the 7.5
architecture floor. A mixed workstation containing an older visible device is
blocked for now even if another GPU qualifies. Proper persisted per-device
selection is required before that restriction can be relaxed safely.

Only the NVIDIA display driver is a system-wide CUDA prerequisite. The normal
VIPP CUDA route installs its CUDA libraries inside the virtual environment and
does not require a separately installed CUDA Toolkit, `nvcc`, Visual Studio, or
CMake.

## Install And Verify

The explicitly unsigned Windows installer is the normal 0.13.0a8 route. Follow
the checksum and Windows-warning steps in the [Quick Start](quick-start.md).

One-click setup derives Windows Local App Data through
`SHGetKnownFolderPath(FOLDERID_LocalAppData)` and accepts only the exact
per-track roots below it: `VIPP\environments\cpu` and
`VIPP\environments\cuda13`. It does not accept a custom managed root. The
complete CUDA path must use ASCII characters in this release because CuPy
14.1.1 cannot reliably compile CUDA kernels from a Windows environment path
containing non-ASCII characters. Spaces are supported. If canonical Local App
Data contains a non-ASCII character, one-click CUDA is unavailable before
environment creation or package download and the UI offers CPU. The fixed CPU
root remains Unicode-safe.

An expert-selected existing environment remains a separate, non-mutating
route. The installer can inspect it but does not move, rename, edit, or convert
it into a managed installation.

### Non-ASCII Windows temporary directories

The installation root and Python's effective temporary directory are separate
paths. If the effective temporary directory contains a non-ASCII character,
VIPP sets `CUPY_CACHE_IN_MEMORY=1` before CuPy compiles kernels. This keeps the
affected NVRTC temporary source operation in memory. It also turns off CuPy's
disk kernel cache for that process, so Compute Doctor or the first GPU work can
pay the compilation cost again after VIPP is closed and reopened.

On the development RTX 5090, one reference run took about 52 seconds from a
cold process and about 0.87 seconds for a refresh in that same process. These
figures describe that machine and workload; they are not a speed guarantee.
Only where compiled kernels are cached changes. The scientific kernels and
their results do not change. If compilation itself fails, Compute Doctor now
preserves the real CuPy `CompileException` instead of replacing it with a false
512-byte private-pool leak message caused by traceback-held probe arrays.

For advanced use, verify a manual dedicated CUDA 13 environment with:

```powershell
& ".\.venv-vipp-gpu-cu13\Scripts\vipp-compute-doctor.exe" --track cuda13
```

A passing report identifies the selected device and confirms that CuPy GPU
execution is available. `python -m pip check` must also report no broken
requirements.

Compute Doctor answers two separate questions instead of treating a working
CUDA import as the whole result:

1. **CUDA and GPU** — can the pinned CUDA/CuPy runtime really allocate and run;
2. **VIPP GPU coverage** — how many of the current reviewed operation regions
   VIPP will actually admit on this machine.

The **Compute setup and memory** window presents those two short rows and one
next step. Memory, provider messages, and other technical evidence begin under
**Show advanced details** so a new user does not have to interpret them first.
After a check, **Save privacy-redacted support report…** writes an atomic JSON
report that is suitable to attach to a support request. The equivalent Windows
PowerShell command is:

```powershell
& ".\.venv-vipp-gpu-cu13\Scripts\vipp-compute-doctor.exe" --track cuda13 --support-bundle ".\vipp-compute-support.json"
```

The support report excludes local repair commands, workflow and node names,
filesystem paths, credentials, and raw environment/workload fingerprints. The
separate `--json` option prints a detailed *local* diagnostic and may include
machine-local provenance; do not share that raw output without reviewing it.

Source developers can inspect or create the pinned qualification environment
with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_gpu_dev.ps1 --track cuda13 --plan-only
powershell -ExecutionPolicy Bypass -File scripts/setup_gpu_dev.ps1 --track cuda13
.\.venv-gpu-cu13\Scripts\python.exe -m napari_vipp.core.compute_diagnostics --track cuda13
```

The setup helper refuses mixed CuPy CUDA majors and writes a strict local
environment record after real-kernel checks. CUDA 12 remains a separate
qualification-only track and must never share an environment with CUDA 13.

## Accelerated Operation Families

Public GPU candidates currently cover reviewed regions of:

- Rolling-Ball Background and Subtract Background;
- Extract Channel and exact Preserve conversion to `float32`;
- Gaussian and median filtering;
- Richardson-Lucy and Richardson-Lucy TV deconvolution;
- fixed Binary, Canny, and Otsu thresholding;
- Sigma Filter;
- boolean Remove Small Objects and Fill Holes in their reviewed cleanup regions;
- Connected Components; and
- the basic **Measure Objects** and **Measure Objects + Intensity** schemas.

Coverage is region-specific rather than node-wide. Examples include finite
float32 requirements for some deconvolution/Gaussian paths, exact-mask or
exact-label promises for reviewed integer operations, and narrow validated
parameter profiles for nonlinear RL-TV. Consult the operation's CPU decision
reason and the implementation records below rather than changing scientific
parameters solely to unlock acceleration.

### Richardson-Lucy agreement policy

Ordinary RL uses the versioned `rl-scientific-equivalence-v2` CPU/GPU gate.
For the default clipped contract, CPU and GPU results must have the same shape
and `float32` dtype, identical finite masks, completely finite and nonnegative
values, NRMSE no greater than `0.005`, and maximum absolute error no greater
than `1e-6 + 0.005 * max(abs(CPU reference))`. RL-TV uses the corresponding
`rl-tv-scientific-equivalence-v2` policy. The older ordinary-RL `2e-6` NRMSE
result and maximum-ULP observations remain visible diagnostics only.

The 0.5% values are backend-agreement margins. They do not prove that a
restoration is accurate, that the PSF matches the acquisition, that the chosen
iteration is appropriate, or that sharpened structures are real. Those
questions require image- and experiment-specific checks such as local error or
artifact maps, forward-model residuals, resolution/noise behavior, intensity or
flux measurements, and the downstream biological measurement.

The checkpoint-backed ordinary-RL workload envelope covers finite authored
`filter_epsilon` values from `1e-12` through `1e-6` and 1 through 100
iterations when the other finite-`float32`, odd-PSF, and default-safe gates
pass. Exact-workload comparison is still required before optimizer selection.
VIPP does not raise epsilon, truncate iterations, or otherwise change an
authored restoration to make it GPU eligible.

The checkpoint matrix is not an exhaustive sample of every epsilon/iteration
pair inside that envelope. The lambda-zero RL-TV profile inherits this envelope;
positive-TV runs remain limited to the exact shipped profile at 10 and 25
iterations, pending their own exact-workload test.

Softly out-of-envelope values retain their exact authored parameters and may be
considered by **Find fastest pipeline...** through an exact-workload comparison.
Invalid input facts, rank or dtype, even PSFs, and nondefault safety controls
remain hard exclusions.

An explicit **Convert Dtype** node may make a longer GPU-resident segment
eligible, but conversion changes the public workflow representation and can
change scaling, storage, thresholds, and downstream output semantics. VIPP
never inserts that conversion silently.

When dtype is the only blocker on a qualified GPU setup, VIPP can offer
**Add conversion**. The initial safe action inserts a visible **Convert Dtype**
node on the affected wire, set to `float32` and `Scaling = Preserve`, for a
`uint8` or `uint16` input. Those integer values are exactly representable in
`float32`; the action does not rescale them. The edit is saved normally and one
Undo removes it. Other branches remain unchanged. VIPP also states the memory
trade-off before the edit: the converted image uses four times the bytes of
`uint8`, or twice the bytes of `uint16`.

The refreshed result says **GPU eligible**, not guaranteed. Compute mode,
workload policy, memory, runtime health, or another eligibility gate may still
select CPU or produce a visible fallback. The shortcut is not offered on a
CPU-only/unqualified environment or for a lossy, clipping, or rescaling
conversion that needs scientific review.

## Portable Segmentation Bridge Example

Open **Portable GPU Segmentation Bridge** from **Open example...**, or launch
`gpu-segmentation`. Its annotated path is Extract Channel, Convert Dtype,
Gaussian Blur, Binary Threshold, boolean Remove Small Objects, Fill Holes, and
3D Connected Components. It opens in Prefer GPU mode, so the same saved
workflow remains usable on CPU-only and partially eligible systems with a
visible explanation for every fallback. Its dedicated sample contains four
objects, one 19-voxel speck removed by the 22-voxel boundary, and 31 enclosed
cavity voxels restored by Fill Holes.

The initial Extract Channel and Binary Threshold implementations are reviewed
public Custom/Prefer-GPU candidates, identified as
`cupy-extract-channel-view-v1` and
`cupy-binary-threshold-f32-exact-v1`. The boolean cleanup implementations are
`cupyx-remove-small-objects-bool-v1` and `cupyx-fill-holes-all-v1`. Remove Small
Objects initially accepts boolean masks only; integer labels stay on CPU
so label identities are never silently changed. Fill Holes initially accepts
only `max_hole_size = 0`, which means fill every enclosed cavity; positive
size-limited cleanup stays on CPU. Passing these local eligibility gates means
that Prefer GPU may select them; it does not guarantee that Auto or every
workload will do so. The example's threshold is deliberately placed inside a
wide gap in its deterministic sample. That makes the review result stable but
does not make the threshold appropriate for unrelated images.

Where channel extraction happens changes the cost. If Extract Channel runs on
CPU at the host entry, VIPP can upload only the selected ZYX channel. If it runs
inside an already resident GPU segment, it creates an allocation-sharing view
without copying the channel, while the full CZYX device allocation remains
live. The resident choice is therefore not automatically faster at the first
node even though it can keep a longer GPU segment connected.

The example can use one host-to-device and one device-to-host boundary only
when the final label image is the single retained terminal result. Previewing,
branching to, or retaining intermediate outputs can require additional
downloads. VIPP reports the actual implementations and transfers rather than
turning this best case into a blanket promise.

## CuPy-only basic measurements

The reviewed **Measure Objects** and **Measure Objects + Intensity** providers
now use CuPy directly, so every current GPU operation is available through the
standard CUDA installation. No separately built provider is needed.

The replacement's full production evidence passed 11 admission cases, 11
rejection cases, two lifecycle cases, and 15 performance cases with zero
private-pool residue. Against the preserved historical artifact, CuPy won all
14 matched transfer-inclusive cases with a 1.78x geometric-mean speedup. A
final replay of the supplied acceptance workflow passed the complete CPU table
parity contract for both nodes. One-pixel objects now correctly report
population standard deviation `0` instead of a nonfinite or rounded artifact.

See the [generated CuPy evidence](benchmarks/measurements-cupy-windows-rtx5090.md)
and its [reproduction protocol](benchmarks/measurements-cupy-evidence-protocol.md).

Exact saved pins migrate from `cucim-measure-objects-basic-v1` to
`cupy-measure-objects-basic-v1`, and from
`cucim-measure-objects-intensity-basic-v1` to
`cupy-measure-objects-intensity-basic-v1`. A broad saved `library:cucim`
preference remains unavailable because it does not identify one unambiguous
replacement. The old phase record and benchmark artifacts remain linked below
as dated evidence; they are not current installation instructions.

## Benchmark Node And Find Fastest Pipeline

In Custom mode, **Benchmark node...** compares the exact captured workload,
including ordered inputs and transfers, and requires the implementation's
scientific parity contract before presenting timing evidence.

**Find fastest pipeline…** evaluates unlocked nodes, models CPU/GPU segments
and transfer costs, and then runs fresh whole-pipeline parity and paired timing
before offering a changed assignment. Analysis changes nothing by itself. A
reviewed proposal is revalidated against the graph, inputs, parameters,
compute intent, locks, source bytes/metadata, and current accelerator
environment before one undoable Apply action.

The result view groups implementations beneath each node and keeps the main
comparison concise. Optional timing details separate compute, observed data
movement, first-run cost, memory, and evidence provenance. If final paired
timing is too close to call, VIPP keeps the current assignment and disables
Apply, but it still shows the completed CPU/GPU results. **No clear winner** is
a speed conclusion, not a claim that the GPU implementation could not run.

Timing evidence is machine- and workload-local. A censored early-stopped CPU
measurement is reported as a lower bound rather than an exact reusable time.
Reaching the analysis time limit means that no fastest assignment was proven;
VIPP keeps the current configuration.

## Batch, Export, Memory, And Failure Behavior

Interactive execution, collection batch, generated Python/CLI, and export use
the same compute request and execution service. Batch artifacts record the
configured and effective request, actual implementation identities, fallback,
memory policy, cleanup, and output provenance. CPU-only installations can load
the same portable workflows safely.

VIPP accounts for RAM, Windows commit headroom, VRAM, transfers, workspaces, and
retained outputs before admitting work. One runtime/device is protected by a
fair process-wide lease so execution and benchmarking do not unknowingly
contend for it.

Cancellation is cooperative and accepted only after synchronized cleanup.
Runtime failure can visibly retry on CPU when the fallback contract permits.
If accelerator cleanup itself fails, the process is treated as unsafe and new
compute is disabled until VIPP restarts. Partial or unreported accelerator
values never replace the previous coherent scientific result.

For durable command-line and collection behavior, see
[Durable GPU execution](durable-gpu-execution.md).

## Cross-device Reproducibility

Compatible GPU models can produce minor floating-point differences because
hardware, drivers, compiler/JIT paths, and reduction order differ. VIPP still
enforces each implementation's declared parity contract—bitwise where
promised and bounded tolerance for reviewed floating-point regions—but
compatibility is not a promise of bitwise identity across every GPU.

For consequential analyses and publications, retain:

- VIPP version and exact workflow/input identities;
- actual implementation IDs and versions for every node;
- GPU model, ordinal, compute capability, and VRAM;
- NVIDIA driver and CUDA driver/runtime/component versions;
- Python, CuPy/CuPyX, NumPy, SciPy, and scikit-image versions; and
- parameters, fallback decisions, and validation against the CPU reference.

Do not treat a speedup measured on one device as a portable performance
promise. The retained RTX 5090 and RTX 4050 Laptop GPU records are bounded
reference evidence for their exact environments and revisions.

## Technical Records

- [Production GPU implementation plan](gpu-production-implementation-plan.md)
- [Durable GPU execution](durable-gpu-execution.md)
- [Phase 1 implementation record](gpu-phase1-implementation-report.md)
- [Richardson-Lucy implementation record](gpu-phase2b-rl-implementation-report.md)
- [Richardson-Lucy TV implementation record](gpu-phase2c-rl-tv-implementation-report.md)
- [Canny and Otsu implementation record](gpu-phase3-canny-otsu-implementation-report.md)
- [Sigma Filter implementation record](gpu-phase4-sigma-filter-implementation-report.md)
- [Connected Components implementation record](gpu-phase5-connected-components-implementation-report.md)
- [Basic Measurements implementation record](gpu-phase6-measurements-implementation-report.md)
- [Historical Windows cuCIM source evaluation](cucim-windows-source-evaluation.md)
- [GPU benchmark records](benchmarks/)

These records contain exact operation matrices, formulas, parity policies,
timings, lifecycle evidence, and deferred gates. They do not override the
current executable policy or turn machine-local measurements into general
performance claims.
