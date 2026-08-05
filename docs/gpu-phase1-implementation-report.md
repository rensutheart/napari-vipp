# GPU Phase 1 implementation record

Status: implemented on `codex/gpu-cross-platform-support` as a headless and
interactive public-candidate vertical slice, with the first Phase 2
interactive controls, workflow-v4 intent, node benchmarking, and Custom-only
whole-pipeline optimizer now connected to that service. This remains a
development-branch surface and is not yet a released cross-platform GPU support
promise. Each validated region is visible normally; regions outside its
scientific or environment contract visibly remain on CPU.

This record describes the code that exists, the environment in which it was
validated, and the gates that intentionally remain closed. The normative
product sequence and promotion criteria remain in the
[production GPU implementation plan](gpu-production-implementation-plan.md).

## Delivered substrate

- Provider-neutral Phase 1-era CPU, Auto, and Custom requests, including
  per-node `auto`, `cpu`, `best_gpu`, implementation-library, and
  exact-implementation preferences. Prefer GPU was added by the later policy
  update.
- Distinct optimizer locks persisted as validated, non-scientific workflow UI
  metadata. Missing lock metadata means every node is unlocked; changing a lock
  does not alter normal execution or scientific cache identity.
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
- Structural pipeline-cache provenance now binds exact source bytes, dtype,
  shape, metadata, image state and revision; public node parameters, ports and
  incoming topology; chained upstream result contexts; and the actual versioned
  implementation. Missing or mismatched provenance fails closed without
  needlessly invalidating an exact upstream cache for a downstream-only policy
  edit.
- A parity-before-timing per-node benchmark service with adaptive paired rounds,
  confidence bounds, transfer/resident timing, peak memory, quarantine, and a
  separate benchmark cache identity. Its machine-local JSON index publishes only
  complete records with atomic replacement plus same- and cross-process locking.
  Pipeline search reuses an exact complete hit; otherwise it screens at 3 paired
  rounds and extends to 7 or 15 only while the timing remains close or uncertain.
  The key binds the CPU scientific stack and excludes the cancellation time
  budget, which does not change the meaning of a completed measurement.
- The first Phase 2 Custom-only whole-pipeline optimizer: detached private
  source/workflow execution, exact workload/environment evidence, measured
  directional transfers, VRAM/liveness-constrained graph assignment,
  operation-specific parity before paired end-to-end validation, review before
  apply, explicit per-node optimizer locks, and one scoped undoable
  authored-intent edit. The current backend is not itself a lock: `Find fastest`
  searches every eligible implementation for every unlocked node. Fresh
  whole-pipeline parity is mandatory before offering a changed assignment even
  when node evidence is reused, and paired validation advances through 5, 7, or
  15 rounds only as needed.
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

### Dtype-sensitive GPU eligibility

The current Gaussian GPU admission is deliberately finite-`float32` only.
Consequently, native `uint16` Gaussian is not yet a scientifically admitted GPU
region and remains on CPU; this is not evidence that Gaussian is inherently a
poor GPU workload. A user may add an explicit **Convert Dtype** node to
`float32` when that conversion is appropriate for the scientific workflow. It
can make Gaussian eligible and can help a longer sequence remain GPU-resident,
but VIPP never inserts the conversion on the user's behalf. Use
`Scaling = Preserve` when numeric values should remain unchanged; the node's
default `Rescale` intentionally remaps the intensity range.

IEEE-754 `float32` represents every integer whose magnitude is at most 2^24
exactly under a Preserve conversion, including the complete `uint8` and
`uint16` ranges. That fact describes only the conversion of individual input
values; it does not make a
float32 pipeline semantically interchangeable with an integer one. The public
dtype, intermediate/output range, later thresholds and rounding, writer
behavior, cache identity, and memory footprint all change. In particular,
`float32` uses twice the RAM/VRAM of `uint16`. Users should review those effects
and benchmark the exact converted pipeline before accepting the tradeoff.

## Exact Phase 1 environment admission

The executable Phase 1 policy is deliberately narrower than the eventual
cross-platform product:

- native Windows, CPython 3.12 with the `cpython-312` ABI;
- NumPy 2.5.1, SciPy 1.18.0, and scikit-image 0.26.0 as the authoritative CPU
  scientific stack;
- CuPy and CuPyX 14.1.1 with the recorded CUDA runtime API 13.2 (`13020`) and
  driver API 13.3 (`13030`);
- an NVIDIA GeForce RTX 5090 with compute capability 12.0;
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
CUDA 12 and secondary NVIDIA devices remain qualification-only tracks outside
public admission; their setup assets do not make them normal Auto/Custom
candidates.

## Recreate the validated development environment

Use a base CPython 3.12 interpreter. The helper modifies only the named virtual
environment and supports `--plan-only` for inspection.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_gpu_dev.ps1 --track cuda13 --plan-only
powershell -ExecutionPolicy Bypass -File scripts/setup_gpu_dev.ps1 --track cuda13
```

The CuPyX operations need no cuCIM wheel. To enable the public background
candidates inside their exact validated environment, first build the reviewed wheel with
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
promise, and an isolated-node record does not teach Auto. It can inform a
reviewed packaged default or an explicit Custom choice. Auto uses reviewed
defaults with no compatible history. Accelerated-only history causes the next
global Auto run to measure CPU once on the same execution surface; a completed
pair then selects under the 1.20x/20-ms gate. Auto never silently benchmarks
multiple implementations.

## Local validation target

The primary Phase 1 machine is native Windows with Python 3.12.9, CuPy 14.1.1,
CUDA runtime API 13.2 (`13020`), driver API 13.3 (`13030`), cuCIM
26.6.0/26.06.00, and an NVIDIA GeForce RTX 5090 (compute capability 12.0,
34,190,458,880 bytes VRAM). Validation covers
CPU compatibility, exact environment admission, scientific parity, read-only
inputs, one-transfer device chains, cancellation, result/fact cache isolation,
benchmark identity, memory accounting, terminal zero-allocation checks, and a
real classified CUDA OOM followed by successful runtime reuse.

The final Phase 1/pre-UI repository run completed with **2,375 passed and 2
expected xfails**.
The xfails document two narrow integer Gaussian parity gaps that remain
CPU-only; they are not advertised GPU regions. A real-device-focused run
completed with **167 passed and no skips** after the verified environment record
was created. A clean wheel build also contained the v2 policy resource and
benchmark adapter; a dependency-free isolated Python 3.12 install loaded the
packaged policy and its five operation entries successfully.

After the first interactive slice and its production-UI hardening, the full
repository run completed with **2,431 passed and 2 expected xfails**. After the
whole-pipeline optimizer, cache-provenance, and apply-time environment checks
were integrated, the full repository run completed with **2,539 passed and 2
expected xfails** on 2026-07-28. The xfails are the same documented CuPy
integer-Gaussian parity gaps.

After the core whole-pipeline optimizer landed, its focused optimizer and graph
suite completed with **39 passed**. This focused result supplements rather than
replaces the last recorded branch-wide and real-device runs above.

A final adversarial hardening pass added exact structural cache provenance,
observable-boundary parity, exact private decision/environment/cleanup checks,
bitwise signed-zero handling, benchmark-equivalent background input scaling,
and fixed-row preference regressions. The focused
optimizer/coordinator/dialog suite completed with **34 passed** and repository
Ruff was clean. That branch-wide run completed with **2,549 passed and 2
expected xfails** on 2026-07-28.

The exhaustive-search/lock/cache/adaptive-validation update, including terminal
dialog and cleanup hardening, completed a clean branch-wide run with **2,579
passed and 2 expected xfails** on 2026-07-28. This is the current
application-wide validation record; the xfails remain the two deliberately
CPU-only integer-Gaussian parity regions.

The fixed production benchmark used 21 paired warm rounds for every case. In
the Phase 1 implementation, these small inputs exercised the then-current
candidate-admission screen; they do not describe Auto's current learning
mechanism:

| Operation | CPU median | GPU end-to-end median | Paired median speedup | Lower 95% bound | Phase 1 screening choice |
| --- | ---: | ---: | ---: | ---: | --- |
| Subtract Background, 31 x 37 | 0.588 ms | 2.567 ms | 0.235x | 0.221x | CPU |
| Gaussian Blur, 128 x 160 | 0.303 ms | 1.630 ms | 0.197x | 0.169x | CPU |
| Median Filter, 96 x 112 | 3.524 ms | 1.500 ms | 2.357x | 2.090x | CPU |

Median is faster by ratio, but its absolute saving is below the then-current
local 10 ms noise floor, so the Phase 1 screen retained CPU. These figures are
not extrapolated. Raw node timings remain Custom/optimizer evidence; current
Auto uses reviewed defaults first, performs one same-surface CPU exploration
run after accelerated-only history, and then uses an exact compatible completed
timing pair under the 1.20x/20-ms gate.

The generated evidence document is
[`benchmarks/phase1-production-node-benchmark-windows-rtx5090.json`](benchmarks/phase1-production-node-benchmark-windows-rtx5090.json).

The [representative real-acquisition ND2 benchmark](benchmarks/representative-nd2-phase1-benchmark.md)
applies the same registered production-node adapter to two full-resolution
planes from a 647 MB, two-channel ND2 time series. At native `uint16`, the
isolated benchmark winner was cuCIM for Subtract Background, CuPyX for Median
Filter, and CPU for Gaussian Blur. Those local winners are Custom evidence,
not graph-global Auto admission. The metadata-order defect exposed by the first
run is now fixed on both `main` and this GPU branch. A later graph-global check
on the exact central `CYX` plane validated Median Filter changing from CPU to
CuPyX in an intentionally mixed CPU/cuCIM/CPU pipeline whose background and
Gaussian choices were deliberately constrained for that historical safety
test: paired medians improved from 321.365 ms to 100.754 ms with a 2.524x lower
confidence bound. The full analysis completed in 27.352 seconds. It predates
the explicit-lock product update and is not an unlocked `Find fastest` timing
claim. It also predates exact benchmark reuse and the progressive 3/7/15 node
and 5/7/15 pipeline measurement checkpoints. The updated unlocked run on the
same central plane took 123.396 seconds from an all-CPU starting assignment:
Background and Median both stopped at three node rounds, and five fresh
whole-pipeline rounds validated 10.049 seconds -> 0.162 seconds with a 39.148x
lower confidence bound. A repeat before apply reused both node records but still
took 72.248 seconds because fresh validation had to compare the unchanged slow
all-CPU assignment. In the measured post-apply assignment, an exact repeat took
0.209 seconds, reused both records, and correctly skipped a redundant
current-versus-identical pipeline validation.

## Deliberately deferred gates

- The first toolbar policy, Custom per-node choices, actual CPU/CuPy/cuCIM
  badges, visible fallback, and the single message-strip component are
  implemented; major/actionable paths are severity-classified. Durable
  workflow-v4 authored intent, worker-based compute setup, RAM/VRAM presentation,
  review-first selected-node benchmarking, and the conservative Custom-only
  whole-pipeline optimizer are also connected. Batch/generated/export consumers
  preserve workflow-v4 intent but still execute on CPU.
- The optimizer currently requires a calculated coherent graph, one shared
  accelerator runtime, exact evidence for every variable candidate, known usable
  VRAM, and node shapes supported by one-input/one-output node benchmarking. It
  refuses unsafe retained writer paths, incomplete/stale identity or timings,
  parity/synchronization failures, infeasible memory, and proposals that do not
  clear the greater of 5% or 10 ms with a lower confidence bound above 1.0.
  `Find fastest` compares every scientifically eligible implementation for each
  unlocked node, regardless of which backend that node currently uses. Only an
  explicit node lock constrains the search; applying the result does not create
  locks. Private parity/timing runs must echo the exact request, environment,
  implementation map, safe decision scope, no fallback, and successful cleanup.
  Apply refuses a changed graph, exact source bytes/metadata/image state,
  compute request or lock state, cache-retention scope, current actual assignment,
  candidate environment, or VIPP/NumPy/SciPy/scikit-image benchmark stack.
  Deterministic typed parity rejections can be reused and explained; transient
  runtime, OOM, cleanup, and timing failures are retried and cannot support an
  exhaustive-optimum claim. A final paired-validation win for the current
  assignment is reported as success with the reverse confidence bound.
- Before broader use, capture workflow/source/retention/compute/assignment as one
  immutable application snapshot, preserve the optimizer's exact evidence
  envelope through every future consumer, retain a single end-to-end deadline
  across all nested benchmark work, and expose transfer/VRAM/refusal details in
  the review UI. The current interactive surface queues normal pipeline work
  during the optimizer's evidence window; a process-wide, device-keyed lease is
  still required for headless, multi-window, and future concurrent GPU callers.
- RTX 40-series laptop, native Linux, CUDA 12 clean-host, and M1 Max evidence is
  still required before broader Auto calibration or platform claims.
- Deconvolution, Otsu, morphology, segmentation, measurement, batch, workflow,
  and generated-Python surfaces follow the ordered rollout in the production
  plan.
- Clara functionality is outside Phase 1. A named near-term investigation must
  decide how to reach a maintainable, feature-complete cuCIM path rather than
  treating the current skimage-only Windows wheel as the end state.
