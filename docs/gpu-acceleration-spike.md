# GPU acceleration architecture and benchmark spike

Date: 2026-07-15

Status: capability/benchmark spike only. No production pipeline, workflow,
batch, generated-Python, or scientific-default behavior was changed.

## Technical summary

CuPy/CuPyX is the recommended primary GPU provider for native Windows and
supported Linux distributions. It is the closest fit for the existing
NumPy/SciPy-shaped operations, can be imported lazily, and was usable on the
available NVIDIA GeForce RTX 4050 Laptop GPU. It has no current CUDA/macOS path;
VIPP must remain CPU-only on macOS during the NVIDIA-only phase. Median
filtering is the strongest first operation to promote: the standard workloads
measured 33.4x (2D) and 35.6x (3D) end-to-end speedups with exact CPU/GPU
parity. Gaussian filtering is a lower-risk integration proof at 2.0-2.1x for
standard inputs. Ordinary Richardson-Lucy (RL) and RL-TV also passed the
standard-size performance and numerical-difference gates, but should follow
filtering because they need iterative progress, cancellation, memory guards,
and a wider scientific validation matrix.

Device residency matters. A six-item Gaussian-to-median batch took 142.6 ms
when every node copied to and from the GPU, versus 99.9 ms when the intermediate
stayed on-device: residency was 1.43x faster than per-node GPU wrapping and
31.6x faster than CPU for that workload.

Promotion must be size-aware. The smoke profile found Gaussian 2D (0.62x),
Gaussian 3D (1.37x), RL-TV 2D (0.28x), and RL-TV 3D (0.75x) slower than CPU
end-to-end. `Auto` therefore cannot mean “use an available GPU for every
supported operation.” It needs validated thresholds keyed by operation,
dimensionality, shape, dtype, and important parameters.

cuCIM remains a gated source-build candidate for operations such as rolling-
ball background estimation. RAPIDS publishes Linux/WSL packages, while the
cuCIM contributor guide documents source and wheel builds but says its procedure
is tested on Ubuntu and only suggests that other operating systems may be
compatible. A later
[native-Windows source evaluation](cucim-windows-source-evaluation.md) produced
a working `cucim.skimage` wheel and found large benefits for rolling ball,
Canny, labeling, Otsu, and region properties; Linux/multi-device packaging and
production gates remain. PyTorch should not be selected or added as an optional dependency
in this phase: its package is cross-platform but NVIDIA CUDA is not available on
macOS, and it adds a second large array/runtime API without filling the current
coverage gap.

Overall assessment: **share with caveats**. The results are strong enough to
choose the architecture and implementation order, but they come from one
laptop GPU and do not yet authorize production GPU execution.

## Cross-platform correction

The application support target and the accelerator support target are
different:

| Target | Windows | Linux | macOS |
|---|---|---|---|
| VIPP base/CPU path | Required | Required | Required |
| CuPy/NVIDIA acceleration | Eligible after validation | Eligible on named supported distributions after validation | Ineligible; CPU-only |

NVIDIA identifies CUDA 10.2 as the last toolkit release with macOS support.
That legacy stack is incompatible with this repository's Python 3.12, CuPy 14,
and CUDA 12/13 direction. This is not fixable by compiling the Python library
from source. If NVIDIA GPU acceleration itself must work on all three operating
systems, the NVIDIA-only scope is infeasible. The production plan therefore
requires platform-marked optional CUDA extras, a macOS CPU/package CI job, and
real GPU validation on native Windows and supported Linux targets.

## Scope and metric definitions

The standard profile used deterministic float32 data (seed `20260715`):

- 2D filters: 1024 x 1024;
- 3D filters: 32 x 192 x 192;
- 2D RL/RL-TV: 512 x 512, 15 iterations;
- 3D RL/RL-TV: 24 x 128 x 128, 15 iterations;
- repeated batch: six 1024 x 1024 items, Gaussian followed by median;
- one unmeasured warm-up and three measured repeats.

`CPU ms` and `GPU ms` below are medians. GPU end-to-end time includes input
host-to-device transfer, synchronized device execution, and result
device-to-host transfer. GPU work was explicitly synchronized before stopping
each timer. `Max abs difference` compares the final CPU and GPU float32 arrays.
The memory value is the CuPy device memory-pool allocation high-water proxy
after a clean run; it is conservative for pooled allocation but is not a
device-wide sampled peak.

## Standard benchmark results

| Workload | CPU ms | GPU end-to-end ms | Speedup | Max abs difference | GPU pool proxy MiB |
|---|---:|---:|---:|---:|---:|
| Gaussian 2D | 23.679 | 11.179 | 2.118x | 2.384e-7 | 20.0 |
| Gaussian 3D | 27.575 | 14.109 | 1.954x | 2.384e-7 | 22.5 |
| Median 2D | 414.477 | 12.420 | 33.373x | 0 | 17.0 |
| Median 3D | 491.306 | 13.817 | 35.559x | 0 | 18.0 |
| Ordinary RL 2D | 246.113 | 31.579 | 7.794x | 1.311e-6 | 9.0 |
| RL-TV 2D | 522.622 | 54.117 | 9.657x | 9.537e-7 | 20.0 |
| Ordinary RL 3D | 597.117 | 291.588 | 2.048x | 5.007e-6 | 12.5 |
| RL-TV 3D | 1211.556 | 365.465 | 3.315x | 6.676e-6 | 35.8 |
| Six-item Gaussian + median, per-node round trips | 3152.868 | 142.642 | 22.103x | 2.384e-7 | n/a |
| Six-item Gaussian + median, device-resident | 3152.868 | 99.916 | 31.555x | 2.384e-7 | n/a |

The corresponding normalized RMSE values were zero for both median cases,
about `0.7e-6` for 2D RL/RL-TV, and about `1.7e-6` for 3D RL/RL-TV. These are
promising, not final acceptance tolerances. The production gate should compare
representative scientific images, boundary modes, PSFs, iteration counts, and
parameter extremes before declaring parity.

The complete machine-readable results, including min/median/max ranges and
separate transfer/compute timings, are in
[`benchmarks/gpu-spike-windows-rtx4050-standard.json`](benchmarks/gpu-spike-windows-rtx4050-standard.json).
The smoke profile is in
[`benchmarks/gpu-spike-windows-rtx4050-smoke.json`](benchmarks/gpu-spike-windows-rtx4050-smoke.json).

## Backend contract

The Qt-free contract is implemented in `src/napari_vipp/core/compute.py` without
changing normal execution:

```python
class ComputeBackend(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"
```

The public selection behavior is:

| Requested backend | GPU available and operation validated | Otherwise |
|---|---|---|
| `cpu` | CPU | CPU |
| `auto` | GPU only when the workload policy predicts a promoted benefit | CPU with a recorded reason |
| `gpu` | GPU | fail before execution with a useful reason |
| `gpu` plus explicit fallback opt-in | GPU | CPU with a recorded fallback reason |

The spike's `select_compute_backend` implements availability/operation support
selection now. A later performance policy should add the shape/dtype/parameter
crossover check before production `Auto` can resolve to GPU. An explicit GPU
request is fail-closed by default so user intent is never silently weakened.

### Capability report

`detect_compute_capabilities()` returns immutable, JSON-safe CPU and GPU
records with:

- backend availability;
- provider and installed version;
- device name;
- validated supported operation IDs;
- a diagnostic reason when unavailable or not yet promoted.

Optional packages are imported only inside detection. The GPU probe creates a
CUDA context and executes a real one-element kernel; import success alone is
not considered availability. CPU capability discovery relies only on the base
NumPy/SciPy/scikit-image environment. The GPU supported-operation list defaults
to empty in this spike, even on the tested machine, so capability discovery
cannot advertise benchmark prototypes as production implementations.

CPU-only import and plugin discovery therefore remain independent of CuPy,
cuCIM, CUDA libraries, and a working GPU driver. Missing or broken optional
packages become capability reasons rather than module-import failures.

### Later `OperationSpec` capability field

Do not put GPU callables or import providers directly in the already-large
`core/pipeline.py`. In a later integration, add a small immutable declaration
to each `OperationSpec`, defaulting to CPU-only, for example:

```python
@dataclass(frozen=True, slots=True)
class ComputeImplementationSpec:
    backend: ComputeBackend
    implementation_id: str
    provider: str
    supported_ndim: frozenset[int]
    supported_dtypes: frozenset[str]
    parity_policy_id: str
    cancellation_granularity: str
    allows_device_residency: bool

# OperationSpec field, added later:
compute_implementations: tuple[ComputeImplementationSpec, ...] = ()
```

Keep implementation callables in a separate provider registry such as
`core/gpu/cupy_ops.py`; resolve that registry lazily only after backend
selection. Workload thresholds and memory estimates belong to the execution
policy/registry rather than static UI metadata.

## Provider evaluation and optional dependencies

### CuPy/CuPyX: select as primary

CuPyX exposes SciPy-compatible ndimage operations, including Gaussian and
median filtering, and reuses the same device-array model for custom RL/RL-TV
loops. The official CuPy guidance also requires CUDA-event or stream
synchronization for reliable timing because GPU work is asynchronous, and it
documents device and pinned memory pools. The harness follows those constraints.

Recommended optional extras, split by CUDA major:

```toml
gpu-cuda12 = [
    "cupy-cuda12x[ctk]>=14,<15; platform_system == 'Windows' or platform_system == 'Linux'",
]
gpu-cuda13 = [
    "cupy-cuda13x[ctk]>=14,<15; platform_system == 'Windows' or platform_system == 'Linux'",
]
```

Do not add either to base dependencies, and do not create one ambiguous `gpu`
extra: the user must choose a CUDA-major-compatible wheel. Platform markers
must prevent either CUDA distribution from resolving on macOS. Prefer the
`[ctk]` variant for the supported installation path because it brings the CUDA
component wheels while still requiring an NVIDIA driver. A driver-only
`cupy-cuda13x` wheel was enough for the ndimage benchmark here, but a
`cupyx.scipy.signal` import exposed a missing cuBLAS DLL; the supported install
should avoid that partial configuration.

### cuCIM: retain as a gated source-build candidate

cuCIM exposes compatible `richardson_lucy` and `rolling_ball` functions and may
avoid reimplementing some scikit-image-shaped paths. However, `cucim-cu13==26.4.0`
could not be installed on native Windows because no compatible distribution was
published for this environment. RAPIDS documents Windows support through WSL2.
The contributor guide documents source and wheel builds, so absence of a wheel
does not by itself reject the library. However, the guide says its instructions
are tested on Ubuntu, relies on GCC/NVCC and shell scripts, and shows Linux
`.so` artifacts. The subsequent
[native-Windows source evaluation](cucim-windows-source-evaluation.md) pinned
`v26.06.00`, produced a CPython 3.12 `win_amd64` wheel for the Python/skimage
layer, passed selected upstream and real-device tests, and measured material
benefit for several operations. It did not build the native `cucim.clara` image-
I/O layer and does not complete the supported-Linux, multi-device, memory, or
cancellation gates. Building from source still cannot overcome the missing
macOS CUDA runtime. Do not add a production provider until every advertised
target clears its remaining gates.

### PyTorch: do not select in this phase

PyTorch provides CUDA tensors and convolution primitives, but using it here
would require another array conversion layer and custom implementations for
coverage already shaped around SciPy/scikit-image. It does not justify its
dependency weight for the current candidate list. Revisit only if future work
introduces trained models, differentiable pipelines, or a Torch-native
operation family large enough to amortize that dependency.

No GPU dependency was added to `pyproject.toml` by this spike.

## Ranked implementation candidates

1. **Device-resident execution segments.** Build this substrate first so the
   operation implementations do not hard-code wasteful node-by-node transfers.
2. **Median filtering.** Best measured gain (33-36x), exact parity in both
   profiles, straightforward CuPyX ndimage mapping, and small memory proxy.
3. **Gaussian filtering.** Simple and low-risk integration proof; standard
   inputs pass at about 2x, but Auto needs a minimum-work threshold because
   small inputs lose end-to-end.
4. **Ordinary Richardson-Lucy.** Worth implementing after filtering (7.8x 2D,
   2.0x 3D standard), using an explicit iteration loop so progress and
   cancellation are observable.
5. **Richardson-Lucy TV.** Strong standard results (9.7x 2D, 3.3x 3D), but its
   small workloads lost and its scientific/cancellation validation surface is
   larger. Preserve all current CPU defaults and formulas.
6. **Rolling-ball/background subtraction.** Unranked for promotion until cuCIM
   and/or a CuPy implementation establishes end-to-end performance, parity,
   memory, packaging, and cancellation behavior on native Windows and supported
   Linux. Subtraction itself is cheap; background estimation determines value.

An operation is promoted only after all of these hold for its supported
workload region:

- at least 1.5x median end-to-end speedup, including transfers;
- bounded memory with a documented estimate and guard;
- operation-specific numerical parity acceptance;
- progress and cancellation behavior appropriate to its kernel structure;
- stable failure behavior and provenance coverage.

## Execution, fallback, and failure behavior

### Missing dependencies or unusable hardware

- Capability detection reports `available=false` and preserves the import or
  CUDA-probe error as `reason`.
- `Auto` runs CPU and records why GPU was not selected.
- Explicit `GPU` fails at preflight. It may retry CPU only when an explicit
  fallback flag was requested.
- Plugin import/discovery and CPU execution never import optional providers.

### Unsupported nodes and mixed graphs

- `Auto` partitions a graph into maximal compatible GPU segments and CPU
  segments, materializing arrays at segment boundaries.
- An explicit GPU request fails preflight with the unsupported operation IDs by
  default. Explicit fallback opt-in may partition or run the whole request on
  CPU, but must be recorded.
- GPU arrays must never leak into existing CPU operation contracts.

### GPU out of memory

- Estimate live input, output, temporary, and provider-workspace bytes before
  launching. Compare against runtime free memory, a configurable GPU cap, and a
  safety reserve; this guard is separate from the existing host-RAM policy.
- On an actual CuPy OOM, synchronize, release live intermediates, clear relevant
  memory-pool blocks, and emit a diagnostic containing the estimate and runtime
  memory state.
- `Auto` may retry the affected segment once on CPU after cleanup. Explicit GPU
  errors by default; CPU retry remains opt-in.
- Never retry indefinitely and never treat an unrelated CUDA exception as OOM.

### Progress and cancellation

- RL and RL-TV report progress and check cancellation at least once per
  iteration, with a stream synchronization/cleanup boundary before returning.
- Gaussian and median are monolithic provider kernels: check cancellation
  immediately before and after each kernel. Chunk only if later measurements
  show acceptable boundary handling and overhead.
- Batch progress is item plus node/iteration progress, not opaque whole-batch
  GPU work.

### Cache and provenance separation

Cache identity and provenance must include:

- requested and resolved backend;
- fallback reason, if any;
- provider and provider version;
- device name and relevant compute capability/runtime identity;
- GPU implementation/kernel version;
- dtype and precision policy;
- operation parameters and existing input/cache identities.

CPU and GPU entries must never alias merely because their final array shapes
match. Cache lookup may reuse an entry only under its declared parity policy;
default to backend-specific separation. Provenance should distinguish an Auto
decision from an explicit request and record any OOM/unsupported fallback.

### Batch cleanup and residency

- Keep compatible intermediates on-device across consecutive supported nodes
  and, when bounded, reuse allocations across batch items.
- Drop live per-item arrays and synchronize at each item boundary so cancellation
  and failures are attributable.
- Do not empty the CuPy pool after every operation or item; that destroys useful
  reuse. Clear free blocks on OOM recovery, at batch end, on memory pressure, or
  when the configured pool cap is exceeded.
- A batch failure must release live GPU references in `finally`, then perform
  the synchronization/pool policy before reporting completion or cancellation.

## Later integration plan

1. Add `ComputeImplementationSpec` and a lazy provider registry, without
   changing scientific CPU implementations.
2. Add requested backend and explicit-fallback policy to `PipelineRunRequest`;
   resolve once during preflight and produce a capability/decision record.
3. Add the GPU memory estimator and maximal compatible-segment planner to core
   execution. Arrays enter/leave the device only at segment boundaries.
4. Implement median and Gaussian with operation-specific workload thresholds,
   parity tests, cancellation boundaries, and provenance.
5. Implement iterative RL, then RL-TV, with per-iteration progress/cancellation
   and dedicated scientific regression tests. Do not change RL-TV defaults or
   its CPU algorithm.
6. Extend batch configuration and cleanup after single-run execution is stable;
   validate cancellation and OOM recovery over repeated items.
7. Separate CPU/GPU cache keys and provenance, then add UI backend controls from
   the Qt-free capability report.
8. Only after the contract stabilizes, carry backend intent into workflows and
   generated Python. Preserve CPU as the portable default.
9. Run clean base/package tests on Windows, macOS, and Linux, then run the real-
   GPU matrix on native Windows and named supported Linux targets with multiple
   GPU tiers, both supported CUDA majors, and representative small/medium/large
   datasets before enabling Auto GPU by default for any operation. In the same
   pass, attempt pinned cuCIM source/package builds on native Windows and the
   supported Linux targets and issue a separate promote/defer/reject result.

## Commands and observed results

Environment and provider probes:

```powershell
nvidia-smi
.\.venv\Scripts\python.exe -c "import importlib.util; print(importlib.util.find_spec('cupy')); print(importlib.util.find_spec('cucim')); print(importlib.util.find_spec('torch'))"
.\.venv\Scripts\python.exe -m pip install "cucim-cu13==26.4.0"
.\.venv\Scripts\python.exe -m pip install "cupy-cuda13x==14.1.1"
.\.venv\Scripts\python.exe -c "import cupy as cp; print(cp.__version__); print(cp.cuda.runtime.getDeviceCount()); print(cp.arange(1))"
```

Observed environment: Windows 11, Python 3.12.7, NVIDIA GeForce RTX 4050 Laptop
GPU, driver 596.08, 6141 MiB VRAM, CUDA driver/runtime API 13.2, CuPy 14.1.1.
CuPy's real kernel probe passed. cuCIM installation reported no matching native
Windows distribution. PyTorch was not installed or benchmarked.

Reproducible benchmarks and checks:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_gpu.py --profile smoke --output docs/benchmarks/gpu-spike-windows-rtx4050-smoke.json
.\.venv\Scripts\python.exe scripts/benchmark_gpu.py --profile standard --output docs/benchmarks/gpu-spike-windows-rtx4050-standard.json
.\.venv\Scripts\python.exe -m pytest src/napari_vipp/_tests/test_compute.py -q
.\.venv\Scripts\python.exe -m ruff check src/napari_vipp/core/compute.py src/napari_vipp/_tests/test_compute.py scripts/benchmark_gpu.py
```

Observed spike checks: seven capability-contract tests passed and Ruff passed.
The benchmark exits successfully with a structured `gpu_unavailable` result if
CuPy or usable hardware is absent, which provides reproducible graceful-failure
evidence on CPU-only systems.

## Limitations and further validation

- One Windows laptop GPU is insufficient for general thresholds. WDDM,
  power/thermal state, driver, and CUDA versions may materially change results.
- Three repeats identify large effects but are not a full performance study;
  promotion CI should use more repeats or robust confidence intervals on stable
  benchmark hosts.
- The memory-pool value is not a sampled device-wide peak. Production work
  should combine pool used/total bytes with CUDA free-memory snapshots and
  operation-specific live-allocation estimates.
- The benchmark-only RL/RL-TV implementations demonstrate feasibility; they
  are not production code and have not passed the full image/PSF/boundary test
  matrix.
- A CuPy rolling-ball/background implementation has not been designed. The
  follow-up [cuCIM evaluation](cucim-windows-source-evaluation.md) established a
  native-Windows `cucim.skimage` build and a large rolling-ball benefit on one
  RTX 5090, but broader hardware/Linux, memory, and cancellation validation is
  still required.
- Thresholds should be recalibrated by operation, dimensions, dtype, shape,
  kernel/PSF size, iteration count, device tier, and whether neighboring nodes
  are already resident.

## Primary technical references

- [CuPy installation](https://docs.cupy.dev/en/stable/install.html)
- [CuPy performance timing guidance](https://docs.cupy.dev/en/v14.1.1/user_guide/performance.html)
- [CuPy memory management](https://docs.cupy.dev/en/latest/user_guide/memory.html)
- [CuPyX Gaussian filter API](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.ndimage.gaussian_filter.html)
- [CuPy NumPy/SciPy comparison table](https://docs.cupy.dev/en/stable/reference/comparison.html)
- [NVIDIA CUDA 10.2 release notes (last macOS-supporting toolkit)](https://docs.nvidia.com/cuda/archive/10.2/pdf/CUDA_Toolkit_Release_Notes.pdf)
- [cuCIM API reference](https://docs.rapids.ai/api/cucim/stable/api/)
- [cuCIM source-build guide](https://github.com/rapidsai/cucim/blob/main/CONTRIBUTING.md#setting-up-your-build-environment)
- [RAPIDS platform requirements](https://docs.rapids.ai/install/)
- [PyTorch local installation matrix](https://docs.pytorch.org/get-started/locally/)
- [PyTorch macOS Metal backend](https://docs.pytorch.org/docs/stable/notes/mps)
- [PyTorch CUDA API](https://docs.pytorch.org/docs/stable/cuda)
- [PyTorch `conv2d` API](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.conv2d)
