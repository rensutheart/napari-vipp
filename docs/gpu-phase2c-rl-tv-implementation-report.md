# GPU Phase 2C Richardson-Lucy TV implementation record

- Date: 2026-07-29
- Branch: `codex/gpu-cross-platform-support`
- Status: exact validated profiles are normal public Auto/Selective candidates
  on the GPU branch; broader biological-data, release, and cross-platform
  qualification remains open

## Outcome

Phase 2C adds Richardson-Lucy total-variation deconvolution as a normal VIPP
accelerator implementation. It reuses the ordinary RL device substrate without
changing the CPU operation, shipped defaults, workflow schema, or output type.
The slice includes:

- CuPy/CuPyX execution for resolved 2D and 3D spatial data with arbitrary
  leading blocks;
- separate, versioned lambda-zero and positive-TV scientific profiles;
- exact ordered `[Image, PSF]` node and pipeline benchmarking;
- conservative TV-aware memory admission;
- synchronized per-block/per-iteration progress and cancellation;
- deterministic cleanup and exact implementation provenance; and
- selected-node benchmark support for resolved multi-input, single-output
  operations in the application UI.

The validated profiles are no longer `developer_hidden`: they are visible in
normal pipelines as public `Selective` candidates and may participate in
`Auto` where applicable workload/runtime benefit evidence exists. Unsupported
parameters, data, shapes, or runtimes visibly remain on CPU. This report is
still branch-scoped development-host evidence, not a blanket released-package
or cross-platform GPU-support statement.

## Frozen scientific contract

The authoritative reference remains
`napari_vipp.core.operations.richardson_lucy_tv_deconvolution`. For each cleaned
float32 spatial block, both providers start from a constant `0.5` estimate and
perform two zero-extension `same` convolutions per iteration. The correction
uses the PSF flipped over every spatial axis.

The accelerated TV term preserves the existing repeated `gradient` definition:
unit spacing, centered interior differences, first-order one-sided boundary
differences, and the divergence formed by applying `gradient` again to each
normalized component. The update keeps the exact minus sign and denominator
placement:

```text
estimate *= correction / max(1 - lambda * divergence, denominator_floor)
```

The provider also preserves the `1e-12` blur guard, authored filter threshold,
TV epsilon, denominator floor, PSF cleaning/normalization, optional input-scale
normalization, per-leading-block independence, per-iteration finite
sanitization and non-negativity clamp, and fixed contiguous `float32` output.
It does not introduce physical voxel spacing, reflect padding, a
forward/backward-adjoint stencil, observed-image initialization, implicit PSF
preparation, synthesized dtype conversion, or changed defaults.

## Implementation and admission profiles

The operation is `richardson_lucy_tv_deconvolution`; its accelerator is
`rl-tv-cupy-f32-v1` version 1, provided by
`napari_vipp.core.gpu.cupy_rl_tv:richardson_lucy_tv_deconvolution`. Optional
CUDA modules remain lazy imports, so CPU-only plugin discovery, workflow
loading, generated Python, and CPU execution remain usable without CuPy.
The dedicated `rl-tv-zero-fill-same-central-gradient-edge1-v1` boundary policy
is declared on the implementation and every array port, so provenance captures
both convolution padding and the scientifically essential TV edge stencil.

Both initial profiles require two ordered finite `float32` inputs, resolved 2D
or 3D spatial rank, non-empty compatible image/PSF extents, positive PSF mass,
odd PSF extents, and the default-safe normalization, clipping, and
scale-preservation options. Every active TV spatial axis must contain at least
two samples.

The profiles then diverge:

1. **Lambda zero:** `tv_regularization == 0` disables the TV branch. This
   profile inherits ordinary RL's `filter_epsilon == 1e-8`, 1 through 25
   iterations, and strict ordinary-RL parity gate. `tv_epsilon` and
   `denominator_floor` remain authored and finite but are scientifically
   inactive.
2. **Positive TV:** the first admitted region is exactly the shipped tuple
   `tv_regularization == 0.002`, `tv_epsilon == 1e-6`,
   `filter_epsilon == 1e-12`, and `denominator_floor == 0.05`, at exactly 10 or
   25 iterations. Other positive iteration counts, lambdas, guards, and floors
   remain visibly on CPU; VIPP never changes them to obtain GPU eligibility.

An explicit **Convert Dtype** node can make Image and PSF finite `float32` when
that representation change is scientifically appropriate. The planner and
optimizer never synthesize a cast to improve a benchmark.

## Why positive TV has a separate parity gate

Lambda-zero RL-TV uses ordinary RL's gate: equal shape and `float32` dtype,
equal finite masks with completely finite output, NRMSE `<= 2e-6`, and
`max_abs <= 1e-6 + 5e-6 * CPU_peak`. It was also tested for direct equivalence
with the ordinary GPU RL provider.

Positive TV is nonlinear. Small CPU/GPU convolution and 3D reduction-order
differences feed back through the gradient normalization and denominator on
every iteration. Applying the ordinary RL screen to the exact positive shipped
profile rejected 113 of the inherited 164 adversarial fixtures at 25
iterations, even though maintained phantom morphology, MSE, flux, and boundary
metrics agreed closely. Loosening ordinary RL's gate would have weakened a
different operation, so Phase 2C instead introduces
`rl-tv-float32-tolerance-v1` only for positive TV:

- equal shape and `float32` dtype;
- equal finite masks, completely finite output, and non-negative CPU/GPU
  results;
- NRMSE `<= 0.005`; and
- `max_abs <= 1e-6 + 0.005 * CPU_peak`.

Maximum float32 ULP distance is retained for diagnosis and does not separately
pass or fail the output. Exact-workload benchmarking still runs CPU/GPU parity
before timing; the broader fixed evidence admits only the implementation
region, never an unverified image.

## Admission evidence

The fixed production-path study covers the inherited 164 deterministic
non-negative float32 2D/3D adversarial fixtures plus an independently generated
96-fixture holdout. Shapes, symmetric/asymmetric PSFs, Poisson and mixed noise,
sparse and zero-heavy signal, ramps, borders, checker structure, high dynamic
range, and dim structures near bright objects are represented.

At the positive shipped profile, all 164 inherited fixtures passed at 10 and 25
iterations. Worst normalized gate scores were `0.45744` and `0.44384`, leaving
more than 54% margin. All 96 holdout fixtures also passed at both iteration
counts, with worst scores `0.22686` and `0.24209`. Across the same 260 cases,
lambda-zero CPU RL-TV was bitwise equal to CPU RL and GPU RL-TV was bitwise
equal to ordinary GPU RL at both iteration boundaries: 520/520 comparisons
passed the strict gate.

The maintained 2D and 3D phantoms had NRMSE `5.249e-7` and `7.592e-7`; maximum
absolute differences were `1.192e-6` and `2.623e-6`. Their largest feature
delta was `2.094e-7`, and their largest relative MSE/border-MSE/flux delta was
`2.103e-6`. Threshold branching was active in 173/260 positive cases at 10
iterations and 181/260 at 25. The default denominator floor did not activate;
the minimum raw denominator was `0.992868`. A dedicated floor-active diagnostic
test covers the branch without claiming that such a regime is publicly
validated.

The reproducible runner, raw evidence, and readable summary are:

- [`scripts/benchmark_gpu_rl_tv_admission.py`](../scripts/benchmark_gpu_rl_tv_admission.py)
- [raw admission evidence](benchmarks/rl-tv-cupy-admission-windows-rtx5090.json)
- [readable admission summary](benchmarks/rl-tv-cupy-admission-windows-rtx5090.md)

The artifact records environment and scientific-source fingerprints and can be
validated on a CPU-only system without importing CuPy. It supports a fresh GPU
run on the named development host; changing any hashed implementation or policy
source makes the checked-in result stale rather than silently reusable.

This is sufficient to expose the exact profiles as public candidates on this
branch. Broadening the admitted region or making stronger biological-restoration,
release, or cross-platform claims still requires bead data and at least three
calibrated biological datasets covering sparse points, dim structures near
bright signal, anisotropic 3D, and boundary objects, with blinded review and
cross-platform replication.

## Memory, progress, cancellation, and cleanup

The versioned `cupyx-richardson-lucy-tv-fft-memory-v1` model inherits ordinary
RL's resident inputs, output, logical iteration buffers, padded real/complex FFT
arrays, cuFFT-plan allowance, and first-use uncertainty. Positive TV adds a
conservative `3 * spatial_ndim + 4` image-sized float32 buffers for gradients,
norm construction, normalized components, divergence, and denominator. The
lambda-zero estimate collapses to the ordinary RL estimate because the TV
branch is inactive.

Progress is `leading block count * iteration count`. A GPU checkpoint is
reported only after synchronization, and cancellation is checked before each
iteration. The runtime's private allocator and scoped FFT-plan policy retain the
ordinary RL cleanup contract: success, parity failure, cancellation, error, and
benchmark exit must leave no live or reserved VIPP-owned device bytes.

## Large-stack CPU/GPU timing

The production-path timing runner uses the exact positive shipped profile at 25
iterations, one warmup, and three paired warm rounds. Every workload must pass
the production parity gate before timing. GPU end-to-end samples include both
input transfers, synchronized resident compute, output transfer, and private
scope cleanup; disk I/O and input generation are excluded.

The machine-local evidence uses one private real-acquisition `ZYX` volume
selected lazily from the representative ND2 file (`T=0`, `C=1`) plus a
deterministic 16.78-million-voxel 3D shape stress. The private path, filename,
pixels, content hash, and content-derived workload identity are not published.

| Workload | Voxels | CPU median | GPU end-to-end | GPU resident | Transfer | Paired median speedup |
|---|---:|---:|---:|---:|---:|---:|
| Private real-acquisition single-channel `ZYX` volume | 8,507,700 | 34.830 s | 0.529 s | 0.461 s | 0.024 s | 66.15x |
| Medium 3D shape stress | 16,777,216 | 55.527 s | 0.511 s | 0.446 s | 0.028 s | 108.63x |

Both cases passed exact production parity and terminal-zero private allocator
cleanup. Observed device peaks were 0.934 GiB and 1.873 GiB. The memory model
estimated 1.501 GiB and 2.502 GiB before its uncertainty allowance; final
admitted bounds were 1.876 GiB and 3.127 GiB, respectively.

Results and raw paired samples are recorded in:

- [`scripts/benchmark_gpu_rl_tv_performance.py`](../scripts/benchmark_gpu_rl_tv_performance.py)
- [readable timing summary](benchmarks/rl-tv-cupy-performance-windows-rtx5090.md)
- [raw timing evidence](benchmarks/rl-tv-cupy-performance-windows-rtx5090.json)

These short RTX 5090 measurements are descriptive machine-local evidence, not
a portable speed promise or durable optimizer record.

## UI and exact benchmarking

The selected-node benchmark eligibility check now accepts one or more fully
resolved ordered inputs when the operation has exactly one output. RL-TV can
therefore benchmark both Image and PSF from the inspector. Every input is
detached, byte-identified, transferred, included in memory admission, and used
for CPU reference and GPU candidate calls. Changing only the PSF invalidates
the evidence. Writers, side effects, unresolved ports, and multi-output nodes
still fail closed.

After an accepted run, the existing node badge/provenance path reports the
actual CPU or GPU · CuPy implementation. A CPU fallback remains explicit rather
than being represented as a successful GPU execution.

## Release-gate validation

The final source tree passed the complete automated suite: 2,745 tests passed,
with two expected integer-Gaussian parity gaps marked `xfail`. Repository-wide
Ruff and whitespace checks passed. The napari plugin manifest validates, the
environment has no broken Python requirements, and both the wheel and source
distribution build successfully. The source distribution was also inspected to
confirm that the RL-TV provider, admission and performance runners, maintained
phantom validator, machine-readable evidence, and validation CSVs are included.

## Current limitations and required hardening

- RL-TV's exact validated profiles are publicly visible in Selective and are
  Auto candidates. This does not admit parameters outside those profiles or
  claim portable performance on unmeasured hardware.
- Positive TV is admitted only for the exact shipped tuple at 10 or 25
  iterations. Lambda zero requires ordinary RL's explicit `1e-8` filter
  profile. Wider parameters require new versioned evidence, not a relaxed test.
- The study does not yet cover calibrated biological volumes, denominator-floor
  activation as a scientifically useful regime, alternative TV stencils,
  physical-spacing-aware TV, reflect padding, or observed initialization.
- Public admission remains the exact native-Windows CUDA runtime API 13.2
  (`13020`), driver API 13.3 (`13030`), and RTX 5090 compute capability 12.0
  region. CUDA 12 is qualification-only. Native Linux, RTX 40-series laptop,
  clean-environment, and Apple M1 Max provider studies remain open; WSL2 is
  secondary evidence.
- Exact node benchmarking still requires one output; writers, multi-output
  operations, multiple accelerator runtimes, batch, generated Python/CLI, and
  export GPU execution remain outside this slice.
- The immutable optimizer snapshot, benchmark-dialog lifecycle, cleanup-failure
  publication, final source-byte Apply check, and uninterrupted paired-evidence
  lease remain tracked promotion hardening.

## Ordered next suggested steps

The next phase has implemented Canny and Otsu; see the
[Canny/Otsu implementation record](gpu-phase3-canny-otsu-implementation-report.md).

1. **Connected components** — preserve connectivity, leading-block semantics,
   `int32` output, and exact deterministic label numbering.
2. **Measurements** — support ordered labels-plus-intensity inputs and an exact
   typed host-table finalizer with schema, order, units, and missing-value
   parity.
3. **Convert Dtype and inexpensive residency bridges** — accelerate only
   explicit authored conversion/bridge nodes that preserve CPU semantics; never
   synthesize casts to improve a benchmark.
4. **Native Linux and Windows laptop validation** — collect clean-environment,
   parity, memory, cancellation, cleanup, and end-to-end evidence on supported
   Linux hosts and available RTX 40-series Windows laptops; treat WSL2 as
   secondary evidence.
5. **Apple M1 Max study** — evaluate Metal/MPS/MLX providers with unified-memory
   accounting and retain CPU fallback unless operation-level gates pass.
6. **cuCIM/Clara feature-complete investigation** — perform the named near-term,
   time-boxed Windows packaging/upstream review, aiming for a maintainable
   feature-complete integration rather than a permanent skimage-only fork.
7. **Batch, generated Python/CLI, and export** — add durable GPU execution and
   provenance only after core interactive coverage and lifecycle contracts are
   stable.

The cross-cutting optimizer lifecycle and immutable-snapshot hardening remains
active alongside these operation families; Phase 2C does not mark it complete.
