# Representative real-acquisition ND2 GPU benchmark

Date: 2026-07-28
Status: machine-local Phase 1 evidence; not a portable performance promise

## Source and benchmark selection

The source was a representative local two-channel ND2 acquisition. It remains
outside the repository and is not redistributed. This report deliberately
omits its filename, workstation path, project identifiers, and content digest;
it does not contain image pixels.

Direct `nd2` 0.11.3 metadata described one `uint16` acquisition with actual
axis order `TZCYX` and shape `19 x 9 x 2 x 1150 x 822`: 19 time points, 9 Z
planes, 2 channels, and 1150 x 822 pixels per plane. The channels were `488`
(488 nm excitation, 515 nm emission) and `561` (561 nm excitation, 620 nm
emission). X/Y sampling was 0.11 micrometers and Z spacing was 0.2
micrometers. The decoded array contains 646,585,200 bytes; the source file was
648,404,992 bytes.

The production-profile comparison used the full-resolution central plane at
`T=9, Z=4` from each channel. This is real acquisition data, not a crop or
synthetic fixture. It avoids presenting a repeated 647 MB CPU run as routine
local benchmarking: one default CPU rolling-ball call already took about 9.6
seconds per plane. Full-stack figures below are explicitly projections, not
benchmark records for another workload shape.

The code revision was `c3e96daff5788a359e00df5ed0d012b34efe561e` on
`codex/gpu-cross-platform-support`. The exact admitted environment fingerprint
was `893868c40b0d305a035b44e9d346caafd7dc6946b6e9ed9f5c4aa3271a25ef69`:
native Windows, CPython 3.12, CuPy/CuPyX 14.1.1, cuCIM 26.6.0/26.06.00, CUDA
runtime 13.2, driver API 13.3, and an NVIDIA GeForce RTX 5090.

## Method

The registered production-node adapter benchmarked these UI defaults:

- Subtract Background: radius 50 px, dark background, smoothing enabled,
  negative clipping enabled, and `2D YX` processing;
- Median Filter: 5 x 5; and
- Gaussian Blur: sigma 1.2.

Every admitted GPU comparison performed scientific parity before warm timing,
one untimed warmup, randomized paired CPU/GPU rounds, 2,000 paired-bootstrap
resamples, and a 95% confidence bound. Adaptive stability checks reached 21
paired rounds for every CPU/GPU comparison. Timings include a fresh detached
host input, H2D transfer, synchronized GPU work, D2H transfer, host result, and
private allocator cleanup. ND2 disk I/O is excluded. Every GPU result passed
its operation's Phase 1 parity policy, and every private runtime pool returned
to zero live and reserved bytes.

The paired median speedup is `median(CPU_i / GPU_i)` over randomized paired
rounds. It is intentionally not the ratio of the independently rounded median
columns.

Native `uint16` Gaussian is an explicit CPU region in the Phase 1 policy because
the current reviewed CuPyX implementation admits only finite `float32`. Its CPU
value below is the median of seven warm calls; no unsupported GPU result was
timed or used to expand the admitted scientific region. This should not be read
as “Gaussian does not benefit from a GPU”: dtype is part of the candidate's
scientific eligibility.

The original benchmark harness called its isolated-node performance result an
`Auto choice`. Under the current product contract this is local Selective
evidence only: it can support an explicit reviewed node/pipeline choice, but it
does not by itself admit a graph-global Auto assignment. Auto requires compatible
whole-segment evidence for the exact graph context.

## Native uint16 results and isolated-node choices

| Node | Channel | CPU median | GPU end-to-end median | Paired median speedup | Lower 95% bound | Isolated-node choice |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Subtract Background | 488 | 9,594.934 ms | 96.597 ms | 99.130x | 93.571x | cuCIM GPU |
| Subtract Background | 561 | 9,584.920 ms | 79.910 ms | 120.920x | 86.718x | cuCIM GPU |
| Median Filter | 488 | 292.046 ms | 4.353 ms | 67.130x | 52.511x | CuPyX GPU |
| Median Filter | 561 | 293.292 ms | 5.709 ms | 52.106x | 41.920x | CuPyX GPU |
| Gaussian Blur | 488 | 11.086 ms | not admitted | n/a | n/a | CPU |
| Gaussian Blur | 561 | 10.764 ms | not admitted | n/a | n/a | CPU |

The first cuCIM background call included about 4.15 seconds of cold provider
and kernel initialization. Its steady-state end-to-end median was 96.6 ms;
later cold calls in the initialized process were much smaller. Interactive UI
guidance should distinguish first-use latency from steady-state local evidence.

For a native-`uint16` chain of Subtract Background -> Gaussian Blur -> Median
Filter, the measured per-node selection is therefore:

`cuCIM GPU -> CPU -> CuPyX GPU`

Summing the independently measured end-to-end node medians gives about 112 ms
per 488 plane and 96 ms per 561 plane, compared with about 9.90 seconds per
plane for an all-CPU chain. This sum is conservative about GPU transfers but is
not a measurement of the resident graph executor as a single transaction.

## Explicit float32 Gaussian comparison

Converting the same planes explicitly to `float32` creates a different admitted
workload. The value-preserving conversion used here is exact for the source's
12-bit integer values but is not implicit VIPP behavior. More generally,
`float32` exactly represents every
integer with magnitude up to 2^24, including all `uint16` values. This exact
input-value conversion does not make the converted workflow equivalent in every
respect: its public dtype, later range/threshold/rounding behavior, output
writers, cache identity, and memory footprint differ. `float32` uses twice the
RAM/VRAM of `uint16`. Add an explicit **Convert Dtype** node only after reviewing
those consequences. Select `Scaling = Preserve` if the numeric values must stay
unchanged; the node's default `Rescale` intentionally remaps the range. VIPP does
not silently convert an integer image to make a GPU implementation eligible.

| Channel | CPU median | GPU end-to-end median | Paired median speedup | Lower 95% bound | Isolated-node choice |
| --- | ---: | ---: | ---: | ---: | --- |
| 488 | 11.691 ms | 7.812 ms | 1.542x | 1.263x | CPU |
| 561 | 12.252 ms | 8.302 ms | 1.443x | 1.321x | CPU |

Parity and confidence passed, but the absolute saving was only about 4 ms per
plane. That is below the local 10 ms noise floor, so the isolated-node gate
correctly retained CPU. A larger float32 stack can still benefit by avoiding
transfers across several eligible nodes, but requires its own exact
whole-pipeline evidence rather than an assumption based on this isolated result.

## Full-acquisition projection

The acquisition contains 342 channel planes. A strictly linear projection of
the measured plane medians gives approximately 56.4 minutes for the example
three-node chain entirely on CPU and 35.6 seconds with the measured selective
choices. This is useful only as an order-of-magnitude planning estimate. It
does not account for full-array batching, resident graph segments, disk I/O,
thermal behavior, or memory pressure, and it must not be persisted as the Auto
record for the full acquisition.

## Whole-pipeline Selective optimizer validation

The review-first whole-pipeline optimizer was subsequently exercised on the
same acquisition and machine using the exact full-resolution `CYX` workload at
`T=9, Z=4` (`2 x 1150 x 822`, `uint16`). The test pipeline was Image Source ->
Extract Channel 0 -> Subtract Background -> Gaussian Blur -> Median Filter.
This was an intentionally constrained safety validation. In the build under
test, authored choices deliberately kept Extract Channel on CPU, Subtract
Background on cuCIM, and native-`uint16` Gaussian on CPU; Median Filter remained
Auto so the optimizer had one measured choice to make. Under the current product
contract, such a constraint is represented by a separate explicit node lock.
Merely running a node on CPU, CuPy, or cuCIM does not lock it: `Find fastest`
compares every eligible implementation for every unlocked node.

The complete analysis took 27.352 seconds, including source detachment, three
node benchmarks, directional transfer profiling, graph solving, scientific
parity, and seven paired end-to-end validation rounds. It proposed only Median
Filter changing from `cpu-median_filter-v1` to
`cupyx-median-filter-v1`. Fixed/excluded nodes retained their authored intent;
in particular, the source remained Auto instead of acquiring an accidental CPU
pin.

This duration is retained as historical safety evidence, not as a latency target
for an unlocked comprehensive search. Approximately 17.57 seconds of the run was
spent benchmarking Subtract Background, mainly through repeated CPU reference
rounds, even though the old authored-choice constraint prevented that node from
changing. The current search contract distinguishes an explicit lock from the
current backend: a locked node has no alternatives to search, while every
eligible alternative of an unlocked node is considered. The updated design
reuses an exact complete node record, screens new node timings at 3/7/15
paired-round checkpoints, and validates the complete pipeline freshly at
5/7/15 checkpoints. Reuse never skips fresh whole-pipeline parity before a
changed assignment is offered. Updated timings are reported below.

| Measurement | Current mixed assignment | Proposed mixed assignment |
| --- | ---: | ---: |
| Cost-model estimate | 228.931 ms | 9.802 ms |
| Paired whole-pipeline median | 321.365 ms | 100.754 ms |

The paired speedup lower confidence bound was **2.524x**, comfortably above the
1.0 confidence gate, and the absolute/relative saving cleared the greater of
10 ms or 5%. Every private run echoed the exact requested implementation map
and environment, reported successful accelerator cleanup, and passed parity at
the changed node plus affected observable boundaries.

This validation also caught a real transfer-profiler lifetime defect before
commit: its final CuPy sample alias survived until the private allocator's leak
check, leaving 1,890,816 live/reserved bytes. The profiler now drops that alias
inside the scope; the successful result above is from the corrected path. This
is precisely why zero-allocation terminal checks remain part of the admission
contract.

### Updated unlocked `Find fastest` timing

The implemented unlocked-search contract was then tested on the same central
`CYX` workload and parameter defaults. ND2 decoding occurred before the analysis
timer. Starting from a clean all-CPU Selective assignment, both eligible nodes
were screened across all admitted alternatives and stopped decisively after
three paired node rounds. Native-`uint16` Gaussian remained scientifically
ineligible for GPU and therefore remained CPU. The optimizer proposed:

`cuCIM Subtract Background -> CPU Gaussian -> CuPyX Median`

The cold comprehensive analysis took **123.396 seconds**. Its fresh five-round
whole-pipeline validation measured **10.049 seconds** for the all-CPU starting
assignment and **0.162 seconds** for the proposal, with a **39.148x** paired
speedup lower confidence bound. The wall time is dominated by the 9.6-10.2
second CPU Background reference: a three-round screen still needs its parity/
cold call, warmup, and three paired calls, while the end-to-end gate needs a
baseline, full parity, and five current-assignment timing calls.

Repeating the analysis immediately **before applying** took **72.248 seconds**.
It reused exact complete records for Background and Median, but correctly reran
fresh pipeline parity and five paired timings against the still-authored slow
all-CPU assignment. This is why an evidence-cache hit is not equivalent to a
fast repeat when the user has not yet applied the winner.

The intended post-apply case was tested separately with the measured cuCIM and
CuPyX preferences authored but still unlocked. With an empty evidence store, the
analysis took **51.378 seconds** because it still had to compare the expensive
CPU Background alternative. The current assignment won, so no redundant
current-versus-identical pipeline validation ran. An immediate exact repeat then
took **0.209 seconds**, reused both node records, performed zero validation
rounds, and confirmed the same assignment with no preference change. These
figures demonstrate the product contract: the first comprehensive search may be
expensive, especially when a viable CPU reference is intrinsically slow; after
the measured winner is applied and exact evidence remains current, confirmation
is fast. Changes to input bytes, dtype, parameters, software stack, candidates,
device/environment, memory scope, or measurement policy deliberately invalidate
that reuse.

## ND2 metadata finding

At the original benchmarked revision, VIPP reported this array as `TCZYX` even though
the direct reader's ordered sizes and array layout were `TZCYX`. `nd2` exposes
`ND2File.sizes` as a read-only mapping proxy, while VIPP's ND2 axis helper
accepted only a concrete `dict` and therefore used its five-dimensional
fallback order. The benchmark bypassed the faulty metadata with explicit
`data[t, z, c, y, x]` indexing.

The defect is mostly invisible to operations that use only the trailing Y/X
axes, but it is unsafe for channel-aware or 3D processing because it swaps the
semantic Z and channel positions. It was fixed and regression-tested on `main`
(`79d4e3d`) and merged into this GPU branch (`52e959a`). The whole-pipeline
validation above still uses explicit `T/Z` indexing so its benchmark workload
remains independently unambiguous.
