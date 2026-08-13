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

The public `0.13.0a6` CUDA route currently targets native 64-bit Windows and
CPython 3.12. Linux and macOS continue to use the CPU path in this alpha while
their accelerator policies are qualified separately.

## Compute Modes

| Mode | Behavior |
| --- | --- |
| **Auto** | Chooses between reviewed CPU and GPU assignments using compatible machine-local evidence and workload gates. CPU can correctly win. |
| **CPU** | Uses the authoritative host implementations throughout the workflow. |
| **Prefer GPU** | Requests every scientifically and operationally eligible public GPU implementation, even when CPU may be faster; ineligible nodes visibly use CPU. |
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

The explicitly unsigned Windows installer is the normal 0.13.0a6 route. Follow
the checksum and Windows-warning steps in the [Quick Start](quick-start.md), or
use this manual dedicated CUDA 13 environment route:

```powershell
& ".\.venv-vipp-gpu-cu13\Scripts\vipp-compute-doctor.exe" --track cuda13
```

A passing report identifies the selected device and confirms that CuPy GPU
execution is available. `python -m pip check` must also report no broken
requirements.

Compute Doctor answers three separate questions instead of treating a working
CUDA import as the whole result:

1. **CUDA and GPU** — can the pinned CUDA/CuPy runtime really allocate and run;
2. **Optional cuCIM** — is the separately built add-on usable; and
3. **VIPP GPU coverage** — how many of the current reviewed operation regions
   VIPP will actually admit on this machine.

The **Compute setup and memory** window presents those three short rows and one
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
- Gaussian and median filtering;
- Richardson-Lucy and Richardson-Lucy TV deconvolution;
- Canny and Otsu thresholding;
- Sigma Filter;
- Connected Components; and
- the basic **Measure Objects** and **Measure Objects + Intensity** schemas.

Coverage is region-specific rather than node-wide. Examples include finite
float32 requirements for some deconvolution/Gaussian paths, exact-mask or
exact-label promises for reviewed integer operations, and narrow validated
parameter profiles for nonlinear RL-TV. Consult the operation's CPU decision
reason and the implementation records below rather than changing scientific
parameters solely to unlock acceleration.

An explicit **Convert Dtype** node may make a longer GPU-resident segment
eligible, but conversion changes the public workflow representation and can
change scaling, storage, thresholds, and downstream output semantics. VIPP
never inserts that conversion automatically.

## Optional cuCIM Add-on

cuCIM is not required for VIPP or for the other qualified CuPy/CuPyX
operations. It supplies the currently reviewed rolling-ball/background and
basic-measurement candidates in its exact admitted environment.

VIPP does not redistribute a private cuCIM wheel. Windows users may download
the exact
[`0.13.0a6` optional cuCIM local-build add-on](https://github.com/rensutheart/napari-vipp/releases/download/v0.13.0a6/napari-vipp-cucim-installer-0.13.0a6-windows.zip)
only after the standard CUDA Compute Doctor passes. Verify its SHA-256 against
the release's `SHA256SUMS-Windows-0.13.0a6.txt`, extract it, and double-click
**Install VIPP cuCIM.cmd**. The bundle contains no wheel: it performs the pinned
build locally, verifies the resulting bytes and provenance, runs real GPU
probes, and records approval in the selected released VIPP environment. The
first build and kernel warm-up can take a long time; later probes normally
reuse the compiled cache. See the bundle's
[complete instructions](../scripts/README-cucim-windows-installer.md) for the
expected checksum, retained records, and command-line recovery route.

When cuCIM is absent or rejected, affected nodes remain scientifically usable
on CPU.

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
- Python, CuPy/CuPyX/cuCIM, NumPy, SciPy, and scikit-image versions; and
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
- [Windows cuCIM source evaluation](cucim-windows-source-evaluation.md)
- [GPU benchmark records](benchmarks/)

These records contain exact operation matrices, formulas, parity policies,
timings, lifecycle evidence, and deferred gates. They do not override the
current executable policy or turn machine-local measurements into general
performance claims.
