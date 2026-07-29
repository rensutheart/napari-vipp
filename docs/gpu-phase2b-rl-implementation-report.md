# GPU Phase 2B ordinary Richardson-Lucy implementation record

- Date: 2026-07-29
- Branch: `codex/gpu-cross-platform-support`
- Status: developer-hidden headless implementation complete; public and
  cross-platform promotion gates remain open

## Outcome

Phase 2B adds ordinary Richardson-Lucy as a normal VIPP accelerator
implementation rather than a special-case demonstration. The slice also adds
the substrate that a scientifically meaningful multi-input node needs:

- a CuPy/CuPyX provider for 2D and 3D spatial data, including arbitrary leading
  blocks;
- ordered-multi-input, single-output exact node benchmarking;
- a fair process-wide accelerator lease keyed by runtime and device;
- explicit workload, memory, parity, progress, cancellation, and typed-output
  planning policies; and
- exact implementation provenance through headless device execution and the
  optimizer.

The implementation remains `developer_hidden`. This record is development
evidence, not a released GPU-support statement.

## Implemented contracts

The operation is `richardson_lucy_deconvolution`; the accelerator implementation
is `rl-cupy-f32-v1` version 1, provided by
`napari_vipp.core.gpu.cupy_rl:richardson_lucy_deconvolution`. CuPy supplies the
resident array domain and `cupyx.scipy.signal` supplies convolution. Optional
CUDA modules remain lazy imports, so plugin discovery, workflows, generated
Python, and the CPU path do not require a GPU package.

The provider receives the established ordered `[Image, PSF]` input, applies the
CPU prepared-call parameters, and returns an Image with the input shape and
fixed `float32` dtype. It covers resolved 2D and 3D spatial modes and processes
each leading block independently. Image, PSF, output, and image-sized
intermediates remain in the CuPy array domain until the execution service's
planned host boundary.

The initial workload region requires:

- exactly two ordered inputs: Image then PSF;
- explicit `float32` dtype and complete finite facts for both inputs;
- a resolved 2D or 3D spatial rank and a PSF whose rank matches it;
- non-empty inputs, with every PSF extent fitting the corresponding image
  extent and positive PSF mass above the validation floor;
- valid existing normalization, clipping, and scale-preservation parameters;
- odd PSF extents;
- the default-safe PSF normalization, input/output clipping, and input-scale
  preservation options;
- `filter_epsilon` exactly `1e-8`; and
- 1 through 25 iterations.

Unsupported data receives a typed CPU/fallback decision. VIPP never inserts a
cast to win a benchmark. An authored **Convert Dtype** node may unlock ordinary
GPU RL when converting both Image and PSF to finite `float32` is scientifically
appropriate. `Scaling = Preserve` keeps representable numeric values, whereas
the node's default `Rescale` intentionally changes the range. Users must review
downstream thresholds, rounding, writers, cache identity, and RAM/VRAM effects.

The authoritative CPU default remains `filter_epsilon=1e-12`. That default and
every value other than the validated `1e-8` point are outside the first GPU
region and therefore visibly remain on CPU. VIPP never changes the epsilon, and
it never truncates a run above 25 iterations. These are scientific parameters,
not performance hints.

Planning now propagates the fixed-`float32`, shape-preserving RL result and its
conservative facts without inspecting a device output. The versioned
`cupyx-richardson-lucy-fft-memory-v2` policy accounts for resident inputs and
output, six logical block buffers, PSF preparation, 2/3/5-smooth padded real
and complex FFT arrays, four padded cuFFT workspace pairs, and a 32 MiB
first-use/out-of-pool allowance. On the measured 512×512 image with a 13×13
PSF, its 55,973,460-byte admitted total bounds the 33,554,432-byte observed
private-pool-plus-out-of-pool peak.

## Scientific parity, progress, and cancellation

The authoritative comparison is the production CPU prepared-call path. The GPU
provider preserves constant initialization, zero-fill `same` convolution, PSF
cleaning and optional normalization, epsilon behavior, input/output clipping,
scale preservation, per-block behavior, and float32 output.

The versioned `rl-float32-tolerance-v1` exact-workload gate requires:

- equal shape and `float32` dtype;
- equal finite/non-finite masks, with the admitted output completely finite;
- NRMSE `<= 2e-6`; and
- `max_abs <= 1e-6 + 5e-6 * max(abs(CPU reference))`.

Maximum float32 ULP distance is recorded for diagnosis but does not independently
pass or fail the result. Wider calibrated real-data morphology, flux, recovery,
and performance gates remain prerequisites for public promotion.

The narrower region is evidence-driven. On the RTX 5090 development host, an
adversarial matrix of 164 normalized nonnegative float32 2D/3D fixtures covered
Gaussian and asymmetric positive PSFs, dark fields, high dynamic range, sparse
and noisy beads, and multiple epsilon/iteration boundaries. At exactly `1e-8`,
all 164 fixtures passed through 25 iterations under the production gate; the
worst normalized gate score was 0.864348. The threshold response was not
monotonic: `1e-7` failed one fixture at 25 iterations, and `1e-6` failed one as
early as 10 iterations. At 50 iterations, `1e-8` failed four fixtures. A
separate 36-fixture matrix rejected the provisional `1e-10` point, and the
unchanged `1e-12` default failed decisively. Forty even-PSF comparison fixtures
had 14 failures at 25 iterations for every tested epsilon. The initial
admission therefore uses exactly `1e-8`, at most 25 iterations, odd PSF extents,
and default-safe options. Optimizer selection still requires exact-workload
parity evidence. Broadening any bound requires new versioned evidence;
loosening the parity tolerance does not.

The fixed matrices and production-path runner are preserved in
[`scripts/benchmark_gpu_rl_admission.py`](../scripts/benchmark_gpu_rl_admission.py).
The committed [raw JSON evidence](benchmarks/rl-cupy-admission-windows-rtx5090.json)
retains all 1,980 per-fixture comparisons, environment and source fingerprints,
while its [readable summary](benchmarks/rl-cupy-admission-windows-rtx5090.md)
shows the condition-level failures and worst gate scores. Reproduce or refresh
both on a CUDA host with:

```powershell
.\.venv-gpu-cu13\Scripts\python.exe scripts\benchmark_gpu_rl_admission.py
```

CPU-only systems can verify that the committed artifact still matches the
generator and scientific source snapshot without importing CuPy:

```powershell
.\.venv-gpu-cu13\Scripts\python.exe scripts\benchmark_gpu_rl_admission.py `
  --validate-existing docs\benchmarks\rl-cupy-admission-windows-rtx5090.json
```

Progress is measured as `leading block count × iteration count`. A checkpoint
is reported only after the completed GPU work is synchronized, and cancellation
is checked at iteration boundaries. This supplies truthful optimizer progress
for ordinary RL while retaining deterministic cleanup on cancellation.

## Ordered-input exact benchmarking

The production adapter and application coordinator now benchmark a pure node
with one or more ordered inputs and exactly one output. For Richardson-Lucy this
means both Image and PSF are:

- detached from mutable live arrays;
- hashed into exact workload identity;
- transferred through the measured runtime path;
- included in byte and memory admission calculations; and
- checked for staleness while evidence is captured and whenever the
  coordinator's exact freshness validation is invoked.

Changing only the PSF therefore produces a different benchmark identity and
invalidates the earlier result. CPU reference, cold/warm candidate calls,
resident GPU timing, transfer modeling, parity, and cleanup all use the same
detached ordered call. Writer nodes and multi-output calls are refused rather
than approximated. The selected-node UI still needs to invoke the exact
source-byte freshness check immediately before Apply; that promotion blocker is
tracked explicitly below.

## Process-wide accelerator lease

Accelerator activity is serialized through a fair lease for each
`(runtime_id, device_id)` key. The lease:

- queues same-device work fairly while allowing different device keys to
  proceed independently;
- is reentrant for the owning thread, so a coordinator transaction can contain
  adapter invocations safely;
- observes cancellation and the caller's absolute deadline while waiting; and
- releases reliably across success, error, cancellation, and OOM cleanup paths.

Device execution, transfer measurement, selected-node benchmarking, each GPU
subtransaction in pipeline optimization, and relevant runtime probe/memory
operations use the same coordination contract. Lease wait time is not hidden or
granted a new budget; it consumes the existing end-to-end deadline. The
optimizer does not yet retain one lease across its entire multi-subtransaction
evidence window, so preventing unrelated work from interleaving between paired
measurements remains a tracked hardening item below.

FFT ownership is also scoped. CuPy's per-thread/per-device plan cache can retain
cuFFT work areas allocated from a private VIPP pool after a large convolution.
The runtime now preserves any pre-existing external plan entries and limits,
disables new cache entries only for the VIPP-owned scope, proves the private pool
is empty, and restores the exact earlier cache state on every exit path. A real
512×512 two-source RL regression reproduced the former 8,821,760-byte residue;
two consecutive runs now finish with zero live, reserved, and out-of-pool bytes.

## Validation evidence

The final focused GPU/execution integration run on 2026-07-29 completed with
**315 passes** and 18 warnings. It covered:

- fake and real-CUDA ordinary RL provider/parity paths;
- 2D, 3D, leading-block, parameter, progress, cancellation, and read-only input
  behavior;
- exact two-input adapter and coordinator execution, PSF-only invalidation,
  transfer/memory accounting, and benchmark reuse;
- same-device lease contention, independent keys, reentrancy, cancellation,
  deadline, and guaranteed release;
- optimizer integration, typed metadata/fact planning, device cleanup, and
  runtime reuse; and
- real CUDA provider and exact-benchmark smoke on the development RTX host; and
- a real 512×512 public headless pipeline run with distinct Image/PSF sources,
  exact provenance, CPU parity, synchronized progress, FFT cleanup, and
  same-runtime reuse.

The final branch-wide run completed with **2,675 passes, 2 documented xfails,
and 83 warnings**. The xfails are the existing narrow integer-Gaussian CuPy
parity gaps; they are explicit scientific exclusions rather than Phase 2B
regressions. Clean-host installation, native Linux, secondary Windows GPU, and
public Auto calibration remain separate gates and must not be inferred from
this development-host result.

## Current limitations and open hardening

- Ordinary GPU RL is developer-hidden; broad public Selective and Auto admission
  are not claimed.
- The CPU default epsilon (`1e-12`), all epsilon values other than `1e-8`, and
  authored runs above 25 iterations are intentionally CPU-only in the first GPU
  region. A future numerical study should broaden this without changing CPU
  defaults or tolerances.
- RL-TV is not implemented.
- Exact node benchmarking supports ordered multi-input calls only when there is
  exactly one output. Writers, side effects, and multi-output operations fail
  closed.
- The pipeline optimizer supports one accelerator runtime and does not yet cover
  batch, generated Python/CLI, or export execution.
- The UI still captures workflow, sources, retention, compute request, locks,
  and accepted assignment as separate inputs. Replacing these with one immutable
  application snapshot under a coherent capture boundary remains required; this
  lifecycle hardening is not complete.
- The admitted environment remains the validated native-Windows CPython 3.12 /
  CUDA/CuPy matrix. Native Linux, RTX 40-series laptop, Apple accelerator, and
  wider clean-install evidence remain open.
- The current Windows cuCIM research wheel is skimage-focused and omits Clara
  I/O. That temporary limitation is not the desired final cuCIM scope.

## Required hardening tracked before broad promotion

These items are deliberately tracked separately from the operation-family
queue below; completing Phase 2B does not silently close them:

- [ ] Broaden ordinary RL away from `filter_epsilon=1e-8`, beyond 25 iterations,
  to even PSFs, or to nondefault safety options only through a new numerical
  policy study. The CPU default and
  production parity tolerance must not be changed merely to expand GPU use.
- [ ] Replace the UI optimizer's separately captured workflow, sources,
  retention, compute request, locks, and baseline assignment with one immutable
  capture object created at one coherent boundary.
- [ ] Make benchmark/optimizer dialogs close safely with their owner, roll back
  all modal state if worker dispatch fails, and ignore late Apply signals after
  shutdown.
- [ ] Treat registry/runtime cleanup failure as a failed analysis result rather
  than publishing or applying otherwise successful evidence.
- [ ] Revalidate selected-node source bytes as well as workflow history before
  Apply, and move expensive environment/source verification off the modal GUI
  thread with a final cheap generation guard.
- [ ] Prevent unrelated same-device work from biasing paired optimizer timing
  between subtransactions, either with one cancellable evidence window or an
  equally strong measured-wait exclusion contract.

## Ordered next suggested steps

This is the maintained order after Phase 2B:

1. **Richardson-Lucy TV** — preserve the existing formula, sign, stencil,
   initialization, floor, padding, defaults, and phantom validation while
   reusing the ordinary RL substrate.
2. **Canny and Otsu** — declare and validate operation-specific dtype,
   parameter, boundary, memory, and parity regions, comparing CuPyX and cuCIM
   where both are scientifically eligible.
3. **Connected components** — preserve connectivity, leading-block semantics,
   `int32` output, and exact deterministic label numbering.
4. **Measurements** — support ordered labels-plus-intensity inputs and an exact
   typed host-table finalizer with schema, order, units, and missing-value
   parity.
5. **Convert Dtype and inexpensive residency bridges** — accelerate only
   explicit authored conversion/bridge nodes that preserve their CPU semantics;
   never synthesize casts to improve a benchmark.
6. **Native Linux and Windows laptop validation** — collect clean-environment,
   parity, memory, cancellation, cleanup, and end-to-end evidence on supported
   Linux hosts and available RTX 40-series Windows laptops; treat WSL2 as
   secondary evidence.
7. **Apple M1 Max study** — evaluate Metal/MPS/MLX providers with unified-memory
   accounting and retain CPU fallback unless operation-level gates pass.
8. **cuCIM/Clara feature-complete investigation** — perform the named near-term,
   time-boxed Windows packaging/upstream review, aiming for a maintainable
   feature-complete integration rather than a permanent skimage-only fork.
9. **Batch, generated Python/CLI, and export** — add durable GPU execution and
   provenance only after core interactive coverage and lifecycle contracts are
   stable.

The immutable UI optimizer snapshot should be hardened alongside the first
operation waves and before broad optimizer claims. Completion of ordinary RL
does not mark that work done.
