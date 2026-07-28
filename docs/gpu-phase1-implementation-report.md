# GPU Phase 1 implementation record

Status: implemented on `codex/gpu-cross-platform-support` as a headless,
developer-hidden vertical slice. It is not exposed by the production toolbar
and is not yet a public GPU support promise.

This record describes the code that exists, the environment in which it was
validated, and the gates that intentionally remain closed. The normative
product sequence and promotion criteria remain in the
[production GPU implementation plan](gpu-production-implementation-plan.md).

## Delivered substrate

- Provider-neutral CPU, Auto, and Selective requests, including per-node
  `auto`, `cpu`, `best_gpu`, implementation-library, and exact-implementation
  preferences.
- Visible or strict fallback decisions with requested, planned, and actual
  execution provenance.
- A Qt-free execution service shared by headless runs, transactional device
  segments, opaque device values, liveness-based release, cancellation, and
  classified out-of-memory recovery.
- Lazy runtime and implementation-library discovery. Importing VIPP or using
  its CPU path does not import CuPy, CuPyX, or cuCIM.
- Candidate-driven, cancellable complete-array facts. Persistent fact reuse is
  allowed only when every upstream source has an authoritative revision token;
  otherwise facts are scoped to the current transaction.
- Scientific result-cache contracts that bind exact implementation, runtime,
  library, dependency, policy, and result semantics. GPU results are detached
  before publication and cannot alias device or mutable producer state.
- A parity-before-timing per-node benchmark service with adaptive paired rounds,
  confidence bounds, transfer/resident timing, peak memory, quarantine, and a
  separate benchmark cache identity.
- A reproducible evidence command, `scripts/benchmark_gpu_phase1.py`, that runs
  fixed production-adapter cases and atomically writes strict JSON.

## Implemented operation regions

| VIPP operation | GPU library | Admitted public dtypes | Spatial region | Phase 1 exclusions |
| --- | --- | --- | --- | --- |
| Rolling-Ball Background | cuCIM | `uint8`, `uint16`, `float32` | reviewed 2D and 3D semantics | other dtypes and out-of-bound radii use CPU |
| Subtract Background | cuCIM | `uint8`, `uint16`, `float32` | reviewed 2D and 3D semantics | other dtypes and out-of-bound radii use CPU |
| Median Filter | CuPyX | `uint8`, `uint16`, finite `float32` | 2D | float64, non-finite float32, and negative-zero-sensitive inputs use CPU |
| Gaussian Blur | CuPyX | finite `float32` | 2D | integer, float64, and non-finite inputs use CPU |
| Gaussian Blur 3D | CuPyX | finite `float32` | 3D | integer, float64, and non-finite inputs use CPU |

Integer background and median regions are bitwise gated. Float32 background
preserves dtype, shape, finite/non-finite masks, exact zero masks, and zero sign,
then applies the documented bounded-error policy. Gaussian uses its separately
versioned float32 tolerance. Unsupported regions produce typed CPU decisions;
they are not silently cast.

## Exact Phase 1 environment admission

The executable Phase 1 policy is deliberately narrower than the eventual
cross-platform product:

- native Windows, CPython 3.12 with the `cpython-312` ABI;
- CuPy and CuPyX 14.1.1 with a working CUDA 12 or CUDA 13 runtime probe,
  numeric driver metadata, a selected NVIDIA device, and compute-capability
  metadata;
- for cuCIM, the CUDA 13 `cucim-cu13` 26.6.0 wheel with SHA-256
  `586d3443091eea67ce2c697be2c490ca51977a5dbdf894b9318b270977134cf8`;
- a strict environment record written only after the setup kernels and
  `pip check` pass. Runtime admission re-verifies the installed CuPy
  distribution and the cuCIM PEP 610 archive digest, so a later same-version
  replacement invalidates the record.

Native Linux setup assets exist, but GPU admission fails closed until a clean
host supplies the required evidence. macOS remains on the authoritative CPU
path while a separate Metal/MPS/MLX provider and unified-memory accounting are
investigated. WSL2 is not treated as native-Windows evidence.

## Recreate the validated development environment

Use a base CPython 3.12 interpreter. The helper modifies only the named virtual
environment and supports `--plan-only` for inspection.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_gpu_dev.ps1 --track cuda13 --plan-only
powershell -ExecutionPolicy Bypass -File scripts/setup_gpu_dev.ps1 --track cuda13
```

The CuPyX operations need no cuCIM wheel. To enable the developer-hidden
background implementations, first build the reviewed wheel with
`scripts/build_cucim_windows.ps1`, then supply the exact reported path and
digest together:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_gpu_dev.ps1 `
  --track cuda13 `
  --cucim-wheel C:\path\to\cucim_cu13-26.6.0-cp312-cp312-win_amd64.whl `
  --cucim-sha256 586d3443091eea67ce2c697be2c490ca51977a5dbdf894b9318b270977134cf8

.\.venv-gpu-cu13\Scripts\python.exe -m napari_vipp.core.compute_diagnostics --track cuda13
.\.venv-gpu-cu13\Scripts\python.exe scripts\benchmark_gpu_phase1.py `
  --output docs\benchmarks\phase1-production-node-benchmark-windows-rtx5090.json
```

The benchmark document is machine-local evidence, not a portable performance
promise. Auto requires compatible evidence for the exact workload and hardware;
without it, or when its confidence/noise gate fails, CPU remains selected.

## Local validation target

The primary Phase 1 machine is native Windows with Python 3.12.9, CuPy 14.1.1,
CUDA runtime 13.2, driver API 13.3, cuCIM 26.6.0/26.06.00, and an NVIDIA GeForce
RTX 5090 (compute capability 12.0, 34,190,458,880 bytes VRAM). Validation covers
CPU compatibility, exact environment admission, scientific parity, read-only
inputs, one-transfer device chains, cancellation, result/fact cache isolation,
benchmark identity, memory accounting, terminal zero-allocation checks, and a
real classified CUDA OOM followed by successful runtime reuse.

The final repository run completed with **2,375 passed and 2 expected xfails**.
The xfails document two narrow integer Gaussian parity gaps that remain
CPU-only; they are not advertised GPU regions. A real-device-focused run
completed with **167 passed and no skips** after the verified environment record
was created. A clean wheel build also contained the v2 policy resource and
benchmark adapter; a dependency-free isolated Python 3.12 install loaded the
packaged policy and its five operation entries successfully.

The fixed production benchmark used 21 paired warm rounds for every case. These
small inputs deliberately demonstrate the conservative Auto gate:

| Operation | CPU median | GPU end-to-end median | Paired median speedup | Lower 95% bound | Auto choice |
| --- | ---: | ---: | ---: | ---: | --- |
| Subtract Background, 31 x 37 | 0.588 ms | 2.567 ms | 0.235x | 0.221x | CPU |
| Gaussian Blur, 128 x 160 | 0.303 ms | 1.630 ms | 0.197x | 0.169x | CPU |
| Median Filter, 96 x 112 | 3.524 ms | 1.500 ms | 2.357x | 2.090x | CPU |

Median is faster by ratio, but its absolute saving is below the local 10 ms
noise floor, so Auto correctly retains CPU. Larger workloads and resident
pipelines require their own exact benchmark evidence; these figures are not
extrapolated.

The generated evidence document is
[`benchmarks/phase1-production-node-benchmark-windows-rtx5090.json`](benchmarks/phase1-production-node-benchmark-windows-rtx5090.json).

The [representative real-acquisition ND2 benchmark](benchmarks/representative-nd2-phase1-benchmark.md)
applies the same registered production-node adapter to two full-resolution
planes from a 647 MB, two-channel ND2 time series. At native `uint16`, Auto
selected cuCIM for Subtract Background, CuPyX for Median Filter, and CPU for
Gaussian Blur. The run also exposed an ND2 Z/channel metadata-order defect;
the benchmark used direct reader indexing so the timing inputs remained
scientifically unambiguous.

## Deliberately deferred gates

- Toolbar controls, per-node badges, whole-pipeline benchmark controls, durable
  user choices, and RAM/VRAM presentation begin in Phase 2.
- RTX 40-series laptop, native Linux, CUDA 12 clean-host, and M1 Max evidence is
  still required before broader Auto calibration or platform claims.
- Deconvolution, Otsu, morphology, segmentation, measurement, batch, workflow,
  and generated-Python surfaces follow the ordered rollout in the production
  plan.
- Clara functionality is outside Phase 1. A named near-term investigation must
  decide how to reach a maintainable, feature-complete cuCIM path rather than
  treating the current skimage-only Windows wheel as the end state.
