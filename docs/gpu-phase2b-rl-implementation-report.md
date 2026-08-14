# GPU Phase 2B ordinary Richardson-Lucy implementation record

- Date: 2026-07-29
- Branch: `codex/gpu-cross-platform-support`
- Status: exact admitted region is a normal public Auto/Custom candidate on
  the GPU branch; release-wide and cross-platform qualification remains open

Phase 2C has since implemented the RL-TV slice described in
this record's former first next step. See the
[Phase 2C implementation record](gpu-phase2c-rl-tv-implementation-report.md)
for its current contracts and evidence. The Phase 2B performance study below
was fully rerun on 2026-08-06 after the release-source hardening; its generated
JSON and Markdown retain the new measurements and current source fingerprints.

On 2026-08-14, ordinary RL backend comparison moved to
`rl-scientific-equivalence-v2`. The policy update below supersedes the original
v1 admission bounds while retaining the 2026-07-29/2026-08-06 measurements as
historical near-identity diagnostics. It does not change the RL formula or
authored workflow parameters.

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

The checkpoint-backed region is no longer `developer_hidden`: it is visible in normal
pipelines as a public candidate for `Custom`, and can participate in `Auto`
when the exact workload/runtime has applicable benefit evidence. Unsupported
dtype, parameter, shape, or runtime regions visibly remain on CPU. This record
is still branch-scoped development evidence, not a blanket released-package or
cross-platform GPU-support statement.

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

The current checkpoint-backed workload envelope requires:

- exactly two ordered inputs: Image then PSF;
- explicit `float32` dtype and complete finite facts for both inputs;
- a resolved 2D or 3D spatial rank and a PSF whose rank matches it;
- non-empty inputs, with every PSF extent fitting the corresponding image
  extent and positive PSF mass above the validation floor;
- valid existing normalization, clipping, and scale-preservation parameters;
- odd PSF extents;
- the default-safe PSF normalization, input/output clipping, and input-scale
  preservation options;
- finite authored `filter_epsilon` from `1e-12` through `1e-6`; and
- 1 through 100 iterations.

Unsupported data receives a typed CPU/fallback decision. VIPP never inserts a
cast to win a benchmark. An authored **Convert Dtype** node may unlock ordinary
GPU RL when converting both Image and PSF to finite `float32` is scientifically
appropriate. `Scaling = Preserve` keeps representable numeric values, whereas
the node's default `Rescale` intentionally changes the range. Users must review
downstream thresholds, rounding, writers, cache identity, and RAM/VRAM effects.

The authoritative CPU default remains `filter_epsilon=1e-12`, and it is now
inside the v2 reviewed range. VIPP never changes epsilon or truncates an
authored run to enter the GPU region. Values outside the range and runs above
100 iterations visibly remain outside ordinary RL's prequalified region; they
may be considered only by an explicit exact-workload comparison where the
surrounding optimizer policy permits it. These are scientific parameters, not
performance hints.

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

The versioned `rl-scientific-equivalence-v2` exact-workload gate requires:

- equal shape and `float32` dtype;
- equal finite/non-finite masks, with the admitted output completely finite;
- nonnegative CPU and GPU output for the default clipped contract;
- NRMSE `<= 0.005`; and
- `max_abs <= 1e-6 + 0.005 * max(abs(CPU reference))`.

NRMSE here is the L2 norm of the CPU/GPU difference divided by the L2 norm of
the CPU reference. Maximum float32 ULP distance and the former v1 NRMSE
`2e-6`/maximum-error result are recorded for diagnosis but do not independently
pass or fail v2.

This is an engineering non-inferiority margin between two VIPP backends. It is
not a universal NRMSE threshold, and passing does not establish restoration
accuracy, PSF suitability, optimal stopping, recovered resolution, absence of
artifacts, or biological validity. Those claims need separate reference images,
forward-model residuals, local artifact maps, frequency/noise measures, flux or
feature measurements, and downstream task validation as appropriate.

This separation follows three focused sources: the official scikit-image
metric documentation notes that NRMSE has no standard normalization across the
literature; Liu et al. characterize how RL's iteration-dependent recovery is
coupled to signal, spatial frequency, and noise; and the Deconwolf benchmark
documents the PSF, iteration, and boundary conditions needed when comparing
accelerated deconvolution outputs. A spatially varying microscopy study also
shows that standard deconvolution can become noisy and low-contrast where the
PSF measured at the field center mismatches edge PSFs. See
[scikit-image NRMSE](https://scikit-image.org/docs/stable/api/skimage.metrics.html),
[Liu et al., 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11751374/),
[Wernersson et al., 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11239506/),
and [Ring deconvolution microscopy](https://pmc.ncbi.nlm.nih.gov/articles/PMC12165846/).

The historical v1 region was evidence-driven. On the RTX 5090 development host, an
adversarial matrix of 164 normalized nonnegative float32 2D/3D fixtures covered
Gaussian and asymmetric positive PSFs, dark fields, high dynamic range, sparse
and noisy beads, and multiple epsilon/iteration boundaries. At exactly `1e-8`,
all 164 fixtures passed through 25 iterations under the production gate; the
worst normalized gate score was 0.864348. The threshold response was not
monotonic: `1e-7` failed one fixture at 25 iterations, and `1e-6` failed one as
early as 10 iterations. At 50 iterations, `1e-8` failed four fixtures. A
separate 36-fixture matrix rejected the provisional `1e-10` point, and the
nearby unchanged `1e-12` CPU default was not directly included in that matrix.
Under v1 it therefore remained unvalidated for GPU rather than being described
as a measured failure; VIPP never silently changed it to qualify an authored run.
Forty even-PSF comparison fixtures had 14 failures at 25 iterations for every
tested epsilon. Those failures describe near-identity sensitivity; they are not
v2 failures and are not being erased or relabeled.

The v2 generator adds the authored `1e-12` epsilon on the same 164 odd-PSF
fixtures at iteration checkpoints 10, 25, 26, 50, and 100. It retains the
`1e-8`/`1e-7`/`1e-6`, provisional `1e-10`, and even-PSF matrices as diagnostic
characterization under the old near-identity limits. Schema v2 records both v2
pass/fail and v1 diagnostic scores per case. The committed v1 JSON/Markdown
below remain historical evidence until a complete v2 artifact is generated;
they must not be presented as if they contain the new checkpoint records.
The checkpoint design does not exhaust every epsilon/iteration combination
inside the envelope.
Optimizer selection still requires exact-workload backend agreement, and
broader release/platform claims require their own regenerated evidence.

The fixed matrices and production-path runner are preserved and versioned in
[`scripts/benchmark_gpu_rl_admission.py`](../scripts/benchmark_gpu_rl_admission.py).
The committed [raw v1 JSON evidence](benchmarks/rl-cupy-admission-windows-rtx5090.json)
retains all 1,980 historical per-fixture comparisons, environment and source
fingerprints, while its [readable v1 summary](benchmarks/rl-cupy-admission-windows-rtx5090.md)
shows the former condition-level failures and worst gate scores. Generate the
complete v2 replacement on a CUDA host with:

```powershell
.\.venv-gpu-cu13\Scripts\python.exe scripts\benchmark_gpu_rl_admission.py
```

After a v2 artifact is generated, CPU-only systems can verify its generator
contract and scientific source snapshot without importing CuPy:

```powershell
.\.venv-gpu-cu13\Scripts\python.exe scripts\benchmark_gpu_rl_admission.py `
  --validate-existing docs\benchmarks\rl-cupy-admission-windows-rtx5090.json
```

## Large-stack CPU/GPU timing

A separate production-path performance screen used the admitted 3D operation
at 25 iterations and `filter_epsilon=1e-8`. It compared the same CPU and GPU
parameters after exact-workload parity, then measured one warmup and three
paired warm calls. GPU end-to-end samples include both Image and PSF transfers,
synchronized resident compute, output transfer, and private-scope cleanup.
Disk I/O and input generation are excluded.

| Workload | Voxels | CPU median | GPU end-to-end | GPU resident | Transfer | Paired median speedup |
|---|---:|---:|---:|---:|---:|---:|
| Private real-acquisition single-channel `ZYX` volume | 8,507,700 | 24.381 s | 0.551 s | 0.518 s | 0.025 s | 45.03x |
| Medium 3D shape stress | 16,777,216 | 34.968 s | 0.411 s | 0.373 s | 0.029 s | 85.06x |
| Large 3D shape stress | 67,108,864 | 144.137 s | 1.524 s | 1.284 s | 0.112 s | 94.58x |

All three exact workloads passed production parity, synchronized execution,
and terminal-zero private allocator cleanup. Observed device peaks were 0.697,
1.098, and 4.502 GiB, within conservative admitted bounds of 1.361, 2.111, and
7.720 GiB, respectively. These are machine-local RTX 5090 results and a short
descriptive screen, not a portable speed promise or reusable optimizer record.
The synthetic volumes deliberately
stress realistic stack shapes and memory; they are not claimed to reproduce
confocal image statistics. The private ND2 case supplies the real-acquisition
anchor without publishing its path, filename, pixel data, or content-derived
fingerprints. Its generated Gaussian PSF is a timing kernel, not a measured
restoration-quality PSF.

The [readable result](benchmarks/rl-cupy-performance-windows-rtx5090.md) and
[raw evidence](benchmarks/rl-cupy-performance-windows-rtx5090.json) retain the
three paired samples, transfer/resident breakdown, memory observations, cleanup
snapshots, environment, and source fingerprints. Re-run or validate them with
[`scripts/benchmark_gpu_rl_performance.py`](../scripts/benchmark_gpu_rl_performance.py).
The standalone validator passes against the 0.13.0a1 release source and checks
both current source fingerprints and canonical JSON-to-Markdown rendering.

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
portable Auto calibration and broader release/platform qualification remain
separate gates and must not be inferred from this development-host result.

## Current limitations and open hardening

- Ordinary GPU RL's checkpoint-backed envelope is publicly visible in Custom and
  is an Auto candidate. This does not admit parameters outside that region or
  claim portable performance on unmeasured hardware.
- The v2 ordinary-RL envelope includes the CPU default epsilon and extends through
  `1e-6` and 100 iterations only for the declared finite-float32, odd-PSF,
  default-safe contract. Even PSFs, nondefault safety options, out-of-range
  epsilon, and longer runs remain outside the prequalified region. Exact-workload
  agreement does not validate restoration quality.
- RL-TV is implemented separately in Phase 2C and its validated profiles are
  public candidates under the same region-specific rule.
- Exact node benchmarking supports ordered multi-input calls only when there is
  exactly one output. Writers, side effects, and multi-output operations fail
  closed.
- The pipeline optimizer supports one accelerator runtime and does not yet cover
  batch, generated Python/CLI, or export execution.
- The UI still captures workflow, sources, retention, compute request, locks,
  and accepted assignment as separate inputs. Replacing these with one immutable
  application snapshot under a coherent capture boundary remains required; this
  lifecycle hardening is not complete.
- The public environment remains the exact validated native-Windows CPython
  3.12 and software/provenance matrix with CUDA runtime API 13.2 (`13020`). It
  admits any successfully probed NVIDIA CUDA device at compute capability 7.5
  or newer with a matching numeric driver API at least 13.3 (`13030`) across
  Auto, Prefer GPU, and Custom. CUDA 12 is qualification-only; native Linux,
  Apple accelerator, broader multi-device performance characterization, and
  wider clean-install evidence remain open.
- The current Windows cuCIM research wheel is skimage-focused and omits Clara
  I/O. That temporary limitation is not the desired final cuCIM scope.

## Required hardening tracked before broader release/platform claims

These items are deliberately tracked separately from the operation-family
queue below; completing Phase 2B does not silently close them:

- [ ] Generate, review, and commit the complete schema-v2 ordinary-RL artifact
  with `filter_epsilon=1e-12` checkpoints at 10, 25, 26, 50, and 100
  iterations. Keep the v1 near-identity matrices as diagnostics rather than
  rewriting their historical results.
- [ ] Broaden ordinary RL to even PSFs, nondefault safety options, epsilon
  outside `1e-12..1e-6`, or runs beyond 100 only through a separately versioned
  numerical study. Never alter CPU defaults or authored parameters merely to
  expand GPU use.
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
- [ ] Make Extract Channel and Select Axis Slice apply their selection to lazy
  resource-backed arrays before NumPy materialization. The representative ND2
  source is `TZCYX`; selecting one `T,C` volume should decode roughly 32 MiB of
  float32 image data, not first materialize the complete 617 MiB uint16
  acquisition.

## Ordered next suggested steps

Phase 2C completed the former first item, and the next phase has implemented
Canny and Otsu. See the
[Canny/Otsu implementation record](gpu-phase3-canny-otsu-implementation-report.md).
The maintained order is now:

1. **Connected components** — preserve connectivity, leading-block semantics,
   `int32` output, and exact deterministic label numbering.
2. **Measurements** — support ordered labels-plus-intensity inputs and an exact
   typed host-table finalizer with schema, order, units, and missing-value
   parity.
3. **Convert Dtype and inexpensive residency bridges** — accelerate only
   explicit authored conversion/bridge nodes that preserve their CPU semantics;
   never synthesize casts to improve a benchmark.
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

The immutable UI optimizer snapshot should be hardened alongside the first
operation waves and before broad optimizer claims. Completion of ordinary RL or
RL-TV does not mark that work done.
