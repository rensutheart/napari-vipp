# GPU Phase 6: Basic Measurements

Status: implemented on `codex/gpu-cross-platform-support` and qualified on one
native-Windows RTX 5090 development environment. This is branch-scoped evidence,
not a portable speed guarantee or a released-package support claim.

## Outcome

VIPP now has public cuCIM candidates for the basic schemas of **Measure
Objects** and **Measure Objects + Intensity**. They preserve the authoritative
CPU table contract while allowing labels, and an optional intensity image, to
arrive through the shared CuPy device domain. The two registered implementation
IDs are:

- `cucim-measure-objects-basic-v1`
- `cucim-measure-objects-intensity-basic-v1`

Both are normal `Selective` candidates in their validated region and are
eligible for workload-specific `Auto`/whole-pipeline comparison. CPU remains a
normal possible winner. In particular, GPU setup and host-table construction
can dominate a small measurement, and the measured float32 plane stack did not
benefit on the qualification host.

## Exact public region

The promoted region accepts:

- native-endian, non-negative `int32` labels;
- resolved 2D or 3D spatial measurement blocks, including ordered leading
  blocks and explicitly resolved spatial-axis layouts;
- arbitrary sparse positive object IDs and an object ID spanning disconnected
  components;
- empty label content, which produces the correct zero-row table schema;
- for **Measure Objects + Intensity**, a same-shape native-endian `bool`,
  `uint8`, `uint16`, or finite `float32` intensity image; and
- the existing basic morphology schema, optionally with the existing five basic
  intensity columns.

The provider preserves row and column ordering, calibrated axis names and
units, physical-size and centroid formulas, bounding boxes, Euler number,
equivalent diameter, public Python scalar/storage types, and empty-table
behavior. Integer-valued fields are validated before conversion. Floating
columns use the operation-owned `basic-measurement-table-v1` parity policy,
including separate reduction tolerances for integer and float32 intensity
inputs.

The first region deliberately leaves the following on CPU with an explicit
reason:

- otherwise valid non-negative integer label arrays outside native-endian
  `int32`, or facts that cannot yet prove the non-negative label contract;
- other intensity dtypes, non-finite float32 intensity, or mismatched label and
  intensity shapes;
- unresolved ranks outside 2D/3D, empty spatial blocks, or a spatial block at
  or above the private compact-label `int32` bound; and
- any extended shape, axis, 2D boundary, derived-ratio, or 2D moment column
  group.

Invalid authored shapes or parameters retain the CPU operation's error
semantics. Boolean or non-integer label domains and any negative label value are
invalid for the authoritative CPU operation as well as the GPU provider; they
raise the existing input-validation error and are not described as CPU
fallbacks. Missing or unqualified CUDA/CuPy/cuCIM, insufficient VRAM, and other
typed availability failures follow the existing visible/strict fallback policy;
they do not silently coerce data or remove requested columns.

## Mandatory typed host-table boundary

The accelerator provider intentionally does not construct `TableData` on the
GPU. It emits one private, C-contiguous `float64` matrix containing the packed
rows. VIPP then enforces this sequence:

1. finish and synchronize resident measurement work;
2. copy the packed matrix to the host;
3. clean up and exit the private CUDA allocation scope;
4. validate and convert the packed matrix into the exact typed `TableData`;
5. expose the public table to caches, callbacks, viewers, and downstream code.

The finalizer is part of the scientific implementation contract and its time
and host staging memory are part of benchmark/optimizer cost. A measurement
table is therefore a mandatory device-segment boundary: the optimizer cannot
pretend that the private packed matrix remains a public resident result. No
CuPy array, allocator handle, or other private device value may escape through
the table.

## Provider and lifecycle design

Positive object IDs are compacted independently per leading block so sparse
authored IDs do not allocate by maximum label value. cuCIM provides the reviewed
basic morphology properties; a pinned cuCIM 26.06 private kernel supplies the
Euler-number calculation. Intensity statistics use ordered float64 reductions,
including a separate deviation pass rather than `E[x^2] - E[x]^2`.

Progress is reported only after synchronized stages within each leading block,
and cancellation is checked between those stages. A canceled call is reusable
and must leave its private allocator at zero used and zero reserved bytes.

cuCIM lazily creates tiny rank-specific region-properties/Euler lookup caches.
VIPP primes the 2D and 3D caches once in a dedicated process-lifetime,
module-owned pool before transaction pools are opened. This keeps library-owned
immutable lookup state distinct from per-call allocations; every subsequent
private 2D/3D execution pool must still drain to zero. The cache warm is not a
measurement result cache and does not relax cleanup checks.

## Native RTX 5090 evidence

The source-current full profile passed 11 admission cases, 11 matched rejection
cases, two progress/cancellation lifecycle cases, deterministic repeats, and
zero private-pool residue after every case. The timing screen includes authored
input transfer, resident compute, mandatory result transfer, and typed-table
finalization.

Representative paired medians on that machine were:

| Workload | CPU | Full public GPU | Speedup | Faster screen |
| --- | ---: | ---: | ---: | --- |
| 256² morphology | 1.04 ms | 6.85 ms | 0.15x | CPU |
| 1024² morphology | 10.74 ms | 5.73 ms | 1.87x | GPU-cuCIM |
| 2048² morphology + uint16 intensity | 308.38 ms | 18.84 ms | 16.37x | GPU-cuCIM |
| 32×256×256 morphology + uint16 intensity | 170.78 ms | 23.46 ms | 7.28x | GPU-cuCIM |
| 6×512² plane-wise morphology + float32 intensity | 101.51 ms | 115.01 ms | 0.88x | CPU |
| 64×512×512 confocal-like volume + uint16 intensity | 1.133 s | 47.42 ms | 23.90x | GPU-cuCIM |
| 16×1024² plane-wise morphology | 713.71 ms | 122.48 ms | 5.83x | GPU-cuCIM |

These results are deliberately not encoded as a size-only rule. `Auto` and
`Find fastest pipeline…` must compare the exact workload and neighboring
residency/transfer context; a small or float32 workload can correctly remain on
CPU while a larger basic measurement selects cuCIM.

See the [canonical evidence](benchmarks/measurements-cucim-windows-rtx5090.md)
and its [reproduction protocol](benchmarks/measurements-cucim-evidence-protocol.md).

## Deferred work

- Add the extended measurement-column groups only after their complete public
  schema, missing-value behavior, units, parity, memory, and lifecycle gates
  pass; do not substitute a convenient `regionprops_table` subset.
- Qualify native Linux and additional Windows NVIDIA devices before making
  portable platform or crossover claims.
- Keep generated Python/CLI, batch, and export execution on their tracked
  durable-surface path rather than implying that this interactive/headless
  provider already activates every surface.
- The maintained next implementation step is **Convert Dtype and inexpensive
  residency bridges**, restricted to explicit authored, scientifically faithful
  operations. VIPP must never synthesize a cast merely to win a benchmark.
