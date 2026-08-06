# GPU Phase 5 Connected Components implementation record

- Date: 2026-08-02
- Implementation branch: `codex/gpu-connected-components`; integration target:
  `codex/gpu-cross-platform-support`
- CPU status: public node and authoritative scientific contract preserved
- GPU status: exact CuPyX region is a normal public Custom and Auto
  candidate in the pinned environment

## Outcome and visibility rule

Phase 5 completes `Label Connected Components`
(`label_connected_components`) as a CPU/GPU vertical slice. The existing public
node remains usable in ordinary workflows without CUDA. Its accelerator
implementation, `cupyx-connected-components-v1`, uses CuPyX on VIPP's CuPy
runtime, preserves device residency, and participates in the same admission,
planning, benchmarking, provenance, memory, progress, cancellation, fallback,
and cleanup contracts as the earlier GPU families.

Accelerator visibility is region-specific. The reviewed boolean-mask 2D/3D
region is declared `public_auto_candidate`, so it is shown in normal Custom
pipelines and is a reviewed Auto default in the exact pinned environment. Auto
uses that default with no compatible history; accelerated-only history causes
one same-surface CPU exploration run before a later matching run applies the
1.20x/20-ms gate. It never silently benchmarks multiple implementations.
Numeric
nonzero-mask conversion, 1D labeling, oversized spatial blocks, unqualified
runtimes/platforms, and any other unsupported region visibly remain on CPU.
VIPP does not coerce an authored numeric image to bool, change its spatial
interpretation, or renumber an approximately equivalent GPU partition merely
to make it eligible.

The full RTX 5090 study was rerun on 2026-08-06 after the release-source
hardening. Its current-source record passed every exact admission case plus the
synchronized lifecycle and private-pool cleanup checks. It remains
machine-local evidence, not a released-package, cross-platform, or
universal-performance claim.

## Frozen CPU scientific contract

### Foreground, connectivity, axes, and output

The authoritative CPU operation accepts an array and treats every nonzero value
as foreground. Boolean input is already a mask; numeric input is compared with
zero without changing the public CPU behavior. Background is label zero.

The public connectivity choices map directly to SciPy binary structures:

- `Face connected` uses rank-one connectivity: 4-neighbor in 2D and 6-neighbor
  in 3D.
- `Full connectivity` uses the full spatial rank: 8-neighbor in 2D and
  26-neighbor in 3D.

`Auto from axes` uses the resolved axis metadata. Explicit 2D mode processes
the final `YX` axes; explicit 3D mode processes the final `ZYX` axes. Every
remaining leading axis is a batch axis. Each leading block is labeled
independently, and label IDs restart at one in every block. Thus two objects in
different timepoints or channels may both have ID one; no component connects
across those leading axes.

The output is shape-preserving native `int32`. Label IDs must be identical to
the authoritative SciPy result, including scan-order numbering. Equality only
up to a permutation of IDs is scientifically insufficient because downstream
measurements and tables use the actual label values. The CPU path asks SciPy to
write `int32` directly so an excessive component count raises rather than
silently narrowing.

### Exact GPU region

The first public accelerator region requires all of the following:

- boolean mask input, including empty shapes supported by the CPU contract;
- resolved spatial rank 2 or 3;
- `Face connected` or `Full connectivity` with an authored/resolved spatial
  mode that agrees with the array rank;
- each active spatial block contains fewer than 2,147,483,646 elements;
- the exact reviewed NumPy/SciPy/CuPyX/runtime/device environment; and
- sufficient memory under `cupyx-connected-components-memory-v1`.

The boolean-only public region is deliberate. The provider can form
`data != 0` on the device, but that numeric conversion has not been promoted as
part of the version-1 accelerator workload contract. Numeric inputs therefore
take the visible CPU route. CPU-only 1D semantics are likewise retained rather
than being silently reinterpreted as 2D.

Invalid authored settings that also violate the CPU contract remain errors;
they are not disguised as accelerator fallback. Valid-but-unreviewed dtypes,
rank, block size, environment, or platform regions produce a typed CPU
decision. Strict Custom GPU intent reports an actionable unsupported or
unavailable decision instead of silently switching implementations.

## GPU implementation and library choice

`cupyx-connected-components-v1` is provided by
`napari_vipp.core.gpu.cupy_connected_components:label_connected_components`.
CuPy owns the CUDA device, stream, allocator, and resident array domain;
`cupyx.scipy.ndimage.label` supplies the labeling primitive. Optional CuPy and
CuPyX imports remain lazy and occur only after the accelerator implementation
has been selected.

The adapter resolves the spatial rank with the shared CPU helper, constructs
the exact SciPy connectivity structure, allocates one shape-preserving `int32`
device output, and labels each spatial block into its corresponding contiguous
output view. No host image transfer or post-hoc canonicalizer is needed inside
a resident segment. CPU and GPU cache/provenance identities remain distinct
because no cross-implementation cache-equivalence group is claimed.

CuPyX is the production choice for this first region because the complete
adapter produced exact SciPy label IDs while sharing the existing CuPy runtime
and residency substrate. The earlier cuCIM source-build screen showed a useful
connected-components primitive, but primitive parity on a few fixtures does not
establish VIPP's complete axis, connectivity, label-ID, overflow, lifecycle,
memory, and fallback contract. A production cuCIM adapter remains a later
comparator if it can satisfy the same contract and provides a material complete-
pipeline advantage; its availability alone does not displace CuPyX.

## Parity, metadata, residency, and provenance

`labels-bitwise-int32-v1` requires equal shape, native `int32` dtype, and
bitwise-identical label values. Swapped-but-equivalent IDs fail. Deterministic
repeats are part of admission, and leading blocks must independently restart at
one.

The output state is projected without scanning device values: shape and axes
are preserved, dtype is fixed to `int32`, and the semantic kind is a label
image. Complete nonnegative/integer facts can therefore flow to downstream
planning without a host materialization. A validated CuPy/CuPyX predecessor,
such as GPU Otsu, can feed Connected Components within one resident CUDA
segment; only the public retained boundary is transferred back to the host.

Scientific result keys retain implementation ID/version, runtime/library,
parity and memory policy IDs, exact parameters, input bytes/shape/dtype, axis
identity, dependencies, device, and environment. Workflow files preserve
portable authored compute intent, not resolved hardware or local timing
samples.

## Memory model

Let `E` be the number of elements in the complete workload and `B` the number
of elements in one active spatial block. The provider-specific peak model is:

```text
E * (1 byte boolean input + 4 bytes int32 output) + B * 7 bytes workspace
```

For one 2D plane or one 3D volume, `E == B`, so the modeled peak is exactly
12 bytes per spatial element. With multiple leading blocks, the full input and
output remain resident while only one block's seven-byte workspace is charged
at a time. Host materialization separately reserves four bytes per output
element. The coordinator also applies its standard uncertainty and runtime
reserve; it does not treat the formula as device-wide free-memory telemetry.

The canonical real-device run measured an isolated private CuPy allocator
reserved high-water for every workload. The model covered every observation,
and success/cancellation cleanup returned both used and reserved private-pool
bytes to zero.

## Progress, cancellation, and the atomic-volume limit

CPU and GPU progress use complete leading spatial blocks as truthful work
units. Both report zero before the first block and check cancellation before
starting every block and again after each atomic block finishes. The GPU path
synchronizes the current CUDA stream before that post-block check and before it
reports completion, so a displayed increment never claims unfinished device
work and a cancellation requested during the final block cannot become a
successful result. Cancellation, error, parity failure, benchmark exit, and
success all use the shared transactional cleanup contract.

`cupyx.scipy.ndimage.label` is monolithic for one spatial block. Consequently,
a single 2D plane or single 3D volume has one atomic progress unit: its progress
bar may remain at zero until the CuPyX call returns, and cancellation can only
be observed after that call before another block or boundary begins. A stack
with independent leading blocks advances and can cancel between those blocks.
This is an honest current limitation, not simulated sub-kernel progress.
Finer-grained progress or mid-volume cancellation would require a different
chunkable algorithm whose seam merging and exact SciPy ID order pass a new
scientific contract.

## Real-device evidence and timing interpretation

The canonical runner is `scripts/benchmark_gpu_connected_components.py`. Its
full profile covers sparse, dense, checkerboard, boundary-touching, empty,
single-component, 2D, 3D, face/full-connectivity, and leading-block cases.
Every admission output is compared as exact native `int32`, and deterministic
repeats must retain identical output hashes. Timing begins only after parity.

CPU timing records a case-cold call and warm host medians. GPU timing records
resident synchronized compute, transfer-inclusive host-to-device/compute/
device-to-host execution, and a case-cold private-pool call after process-level
runtime warmup. The VRAM observation is the isolated CuPy pool's reserved
high-water, not device-wide telemetry.

The finalized machine-local matrix contains 18 timed workloads and is retained
in the
[readable evidence](benchmarks/connected-components-cupyx-windows-rtx5090.md)
and
[machine-readable record](benchmarks/connected-components-cupyx-windows-rtx5090.json).
Its main product conclusion is workload sensitivity: small planes can favor
SciPy, while larger dense/checkerboard planes, 3D volumes, and resident stacks
can favor CuPyX; sparse masks with many isolated components can cross over at a
different point. A size-only rule is therefore not justified.

For the largest tested sparse volume (`64×512×512`), the case-cold CPU call was
87.56 ms and the case-cold transfer-inclusive GPU call was 42.66 ms, a 2.05x
speedup after process-level CUDA warmup with an empty private allocator pool.
Warm medians were 89.01 ms on CPU, 30.19 ms GPU transfer-inclusive (2.95x),
and 1.48 ms for resident GPU compute. The large gap between resident and
transfer-inclusive timing is why complete-pipeline residency must remain part
of selection. Among the tested plane extents, the first transfer-inclusive GPU
crossover was 512² for dense/full and checkerboard/face masks, but 1024² for
sparse/face masks.

The evidence table's `Choice` column identifies only the faster observed warm
transfer-inclusive median on this host. It is not a durable optimizer record
and does not by itself satisfy Auto's confidence and absolute-saving gates.
VIPP must benchmark or reuse a source-current record for the exact workload,
environment, and complete pipeline, including transfers and neighboring-node
residency. CPU remains a correct and expected Auto result for small or
insufficiently decisive calls.

### Release-source evidence refresh

Connected Components, ordinary RL, and RL-TV were all fully rerun on
2026-08-06 after the 0.13.0a1 cuCIM provenance hardening changed shared policy,
specification, and registry files. Each generator wrote new measurements and
new source fingerprints together; all three standalone validators now pass.
This avoids associating historical timings with newer source bytes. Longer
term, RL/RL-TV should adopt operation-owned CPU, parity, and compute-contract
modules like Sigma Filter so unrelated registrations create less provenance
churn while real operation changes still fail closed.

## Packaged compute-policy artifact

`phase1-gpu-public-v5.json` is the current immutable packaged compute-policy
record. It is a strict extension of v4: the six historical operations are
byte-for-byte represented in the same order and the connected-components
record is appended. The record mirrors the live public candidate's bool dtype,
2D/3D support, policy IDs, limitations, and derived
`resolved_spatial_ndim` bound of 2 through 3. Historical v1-v4 resource bytes
remain pinned and unchanged.

The v5 canonical content digest is:

```text
061fa1d02b89c1dbed47e1d5061836a9d92edb9221cff821eb30a4afe5e3d756
```

## Platform and packaging limits

The first admitted environment is native Windows, CPython 3.12, NumPy 2.5.1,
SciPy 1.18.0, CuPy/CuPyX 14.1.1, CUDA runtime API 13.2, driver API 13.3, and the
recorded RTX 5090/compute-capability-12.0 device. Changed scientific packages,
runtime, driver, GPU, compute capability, Python ABI, OS, or execution mode fail
closed until their own evidence is reviewed.

Native Linux and RTX 40-series Windows laptops remain named validation targets;
WSL2 is secondary evidence. CUDA has no macOS target, so this provider is CPU-
only on macOS. The Apple M1 Max Metal/MPS/MLX study may introduce a different
provider only after the same operation-level scientific and lifecycle gates
pass. Base installation, plugin discovery, workflow loading, and CPU execution
must continue to work without CuPy.

## Deferred scope

- Numeric nonzero-mask GPU conversion and 1D GPU labeling.
- A chunked single-volume algorithm with exact seam merging, label order,
  progress, and cancellation.
- A cuCIM production comparator under the complete VIPP contract.
- Broader runtime, driver, GPU, OS, Python, and scientific-stack admission.
- GPU execution and provenance in batch, generated Python/CLI, and export.
- CPU/GPU cache equivalence or partition-only label equivalence.

## Ordered next suggested steps

1. **Measurements, preceded by the operation-owned RL provenance boundary** —
   isolate the inherited RL/RL-TV evidence owners described above, then support
   ordered labels-plus-intensity inputs and an exact typed host-table finalizer
   preserving schema, row/column order, units, calibration, missing values, and
   public scalar/storage types.
2. **Convert Dtype and inexpensive residency bridges** — accelerate only
   explicit authored operations that preserve CPU semantics; never synthesize
   a cast or bridge to improve a benchmark.
3. **Native Linux and Windows laptop evidence** — collect clean-install,
   parity, memory, progress, cancellation, cleanup, and end-to-end evidence on
   supported Linux hosts and available RTX 40-series Windows laptops.
4. **Apple M1 Max study** — evaluate Metal/MPS/MLX with unified-memory
   accounting and retain CPU fallback unless operation-level gates pass.
5. **cuCIM/Clara feature-complete investigation** — perform the named
   time-boxed packaging/upstream review toward a maintainable complete route,
   including a Connected Components comparator where justified.
6. **Batch, generated Python/CLI, and export** — add durable GPU execution and
   provenance after interactive coverage and lifecycle contracts are stable.
