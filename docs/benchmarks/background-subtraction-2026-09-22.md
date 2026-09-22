# Background subtraction: speed and scientific differences

Completed 2026-09-22. **Keep current rolling-ball subtraction unchanged.** A
separately named morphological background method could be useful, but the matched
SciPy box implementation was faster than SimpleITK on this machine. SimpleITK's
ball and reconstruction variants did not establish a general speed advantage.

## Evidence and scope

- 66 configurations, six methods, 396 attempted backend measurements.
- 395 completed with three warm repetitions each; one bounded timeout.
- CPU/GPU rolling-ball outputs had identical hashes in all 66 configurations.
- SciPy/SimpleITK box outputs had identical hashes in all 66 configurations.
- Identical input hashes across the six methods for every configuration.
- All five measured source hashes were verified and preserved in the local
  `background-comparison-20260922/measured-source` snapshot.
- [Reviewed machine-readable evidence](background-subtraction-2026-09-22.json)
  retains timings, cases, versions, hashes, numerical differences and quality
  diagnostics. Its header identifies and hashes the complete local raw record.
- [Protocol and primary references](background-subtraction-protocol-2026-09-22.md).

Windows 11; Intel Core i5-10500 (6 cores / 12 logical processors); RTX 5090;
32 GiB RAM. Python 3.12.9, NumPy 2.5.1, SciPy 1.18.0, scikit-image 0.26.0,
SimpleITK 2.5.6, CuPy CUDA 13 package 14.1.1; GPU driver 610.74.
The serialized run lasted approximately 2 h 3 min, including worker setup,
first calls, warm repetitions and quality calculations.

These are controlled synthetic workloads, not biological validation or universal
performance guarantees. Flat opening and reconstruction are different algorithms
from non-flat rolling ball, not alternate implementations of the same operation.

## Complete-operation timings

Warm median seconds; uint16 mixed phantoms; presmoothing enabled. GPU timings
include upload, download and synchronization. Fixture creation and quality checks
are excluded. Columns labelled ITK use SimpleITK. All methods use original-input
subtraction and the same clipping/rounding policies.

| Shape and spatial mode; radius | VIPP CPU | VIPP GPU | ITK ball | ITK box | ITK reconstruction | SciPy box |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 × 256 XY; 15 | 0.0632 | 0.00247 | 0.140 | 0.0208 | 0.118 | 0.00292 |
| 1024 × 1024 XY; 15 | 1.028 | 0.00561 | 0.835 | 0.125 | 0.692 | 0.0616 |
| 2048 × 2048 XY; 15 | 4.190 | 0.0128 | 2.434 | 0.464 | 2.246 | 0.358 |
| 2048 × 2048 XY; 50 | 41.968 | 0.0368 | 42.896 | 0.620 | 33.800 | 0.371 |
| 16 × 256 × 256, independent XY; 15 | 1.025 | 0.0232 | 2.193 | 0.320 | 1.851 | 0.0508 |
| 32 × 512 × 512, independent XY; 15 | 8.324 | 0.0771 | 10.091 | 1.419 | 8.749 | 0.472 |
| 32 × 512 × 512, independent XY; 50 | 83.605 | 0.135 | Incomplete | 3.046 | 180.309 | 0.519 |
| 16 × 128 × 128, true 3D; 7 | 1.560 | 0.00258 | 5.872 | 0.256 | 3.243 | 0.0261 |
| 16 × 128 × 128, true 3D; 15 | 10.167 | 0.00471 | 96.503 | 0.317 | 28.519 | 0.0471 |

At 2048²/radius 50, box opening is about **68× faster with SimpleITK** or
**113× faster with SciPy** than CPU rolling ball, but it produces different
scientific results. For the genuinely matched box algorithm, SciPy had the lower
median in every configuration; ITK took 1.079–15.444 times as long. The smallest
gap may be within run variability, not an established meaningful advantage.

Across this specific matrix, ITK ball was faster than CPU rolling ball in 11 of
65 complete cases; reconstruction in 9 of 66; ITK box in 64 of 66. SciPy box was
faster than CPU rolling ball in all 66. These counts are coverage summaries, not
probabilities for arbitrary user data. Current GPU rolling ball was the strongest
tested acceleration while retaining the original method's exact outputs.

The incomplete ITK-ball case hit the **900-second total worker deadline** during
its third warm repetition. First call: 265.949 s; two completed warm calls:
261.677 and 262.646 s. These partial samples are retained but excluded from
complete-case ratios. The deadline includes all calls and worker setup; it is
not a claim that one call took 900 seconds.

### Variability, first-call latency and memory

The median relative three-run spread, `(max - min) / median`, was 5.19%.
104/395 complete rows exceeded 10%; 28 exceeded 20%. Tiny millisecond timings
are particularly sensitive. Large CPU rolling ball at 2048²/radius 50 was steadier:
41.861–42.042 s. Three repetitions do not support confidence intervals or precise
universal rankings for near-ties.

First calls include roughly three seconds of common lazy VIPP/scientific imports.
GPU first calls were 3.512–4.177 s. This is neither GPU-only initialization cost
nor normal latency after the application has initialized. Disk caches were not
cleared. Warm timings describe already initialized complete array operations.

**Memory results are invalid and intentionally omitted.** The Windows virtual
environment launches an interpreter child; the original monitor sampled only its
small launcher process. It therefore did not reliably enforce the nominal 6 GiB
runtime cap either. Conservative memory preflight was separate, not a measured
guarantee. In-worker operation timers and output evidence remain valid. The
process-tree monitor was corrected separately, with 22 runner tests passing,
including a real child-process memory-allocation check. The corrected runner uses
a new schema and an explicit summed-host-RSS field; shared pages may be counted
more than once and GPU memory is excluded. This fix cannot retroactively validate
the original run's memory data; no memory ranking is claimed.

## How much do the outputs differ?

On the 1024² uint16 mixed phantom, radius 15:

| Alternative versus CPU rolling ball | Mean absolute difference | Maximum difference |
| --- | ---: | ---: |
| SciPy / ITK box | 367.70 counts | 6,852 counts |
| ITK ball | 454.25 counts | 7,554 counts |
| ITK reconstruction | 498.18 counts | 7,565 counts |

Box differs at 93.325% of pixels; RMSE is 722.96 counts. This is substantive,
not an integer-rounding difference. Correlation alone does not establish
equivalence. Agreement with today's method and recovery of known truth are
separate questions: neither algorithm is a universal ground-truth reference.

### A morphology benefit and a failure mode

256² uint16 phantoms, radius 15, presmoothing on. Retention is the sum of corrected
values within known foreground divided by the true foreground sum. **Above 100%
can mean residual background/noise, not additional signal recovered.**

| Phantom | Method | Foreground retained | Background mean | Fixed-threshold Dice |
| --- | --- | ---: | ---: | ---: |
| Sparse objects | CPU/GPU rolling ball | 108.04% | 565.37 | 0.94519 |
| Sparse objects | SciPy / ITK box | 100.20% | 86.96 | 0.98946 |
| Broad object | CPU/GPU rolling ball | 82.47% | 578.21 | 0.95623 |
| Broad object | SciPy / ITK box | 36.76% | 83.18 | 0.46428 |
| Broad object | ITK ball | 23.30% | 85.98 | 0.32594 |
| Broad object | ITK reconstruction | 22.96% | 57.03 | 0.32678 |

Box opening reduces residual background and preserves the small sparse objects
well here, but removes much of the broad feature at this radius. That broad
object alone retained **76.56% with rolling ball, 22.10% with box, 7.57% with
ITK ball, and 7.02% with reconstruction**. Conversely, the dim object's apparent
retention was 159.25%, 101.64%, 95.07% and 62.54%, respectively: rolling ball
overestimated it through remaining background. An overall average would hide
these distinct problems.

There is also a positive true-3D morphology example: at 16 × 128 × 128, radius 7,
box gives 103.67% retention, background mean 181.76 and Dice 0.92597, versus
142.76%, 899.38 and 0.72322 for rolling ball. This supports an alternative method,
not replacing one universal winner with another.

### Normalized float input needs particular care

Same 512² float32 scene, radius 15, presmoothing on; only intensity scale changes:

| Method | Count-scale retention | Normalized [0, 1] retention | Count-scale / normalized Dice |
| --- | ---: | ---: | ---: |
| CPU/GPU rolling ball | 73.1099% | 1.7917% | 0.63240 / 0 |
| SciPy / ITK box | 43.0305% | 43.0305% | 0.58053 / 0.58053 |
| ITK ball | 35.9793% | 35.9793% | 0.52728 / 0.52728 |
| ITK reconstruction | 31.9857% | 31.9857% | 0.48847 / 0.48847 |

All six 512² normalized rolling-ball cases (radii 5/15/50, smoothing off/on)
produced no regions at the common truth-derived threshold. This is an
intensity-dependent rolling-ball geometry issue, not a CPU/GPU disagreement.
Do not silently normalize or change saved workflows. A future explicitly
intensity-aware rolling-ball option and clear guidance deserve investigation.

### Raw values, polarity and interpretation limits

The 12 selected dark-background preview configurations retain raw pre-clipping
diagnostics as well as final outputs. For normalized broad input, reconstruction
differs from rolling ball at 12.920% of raw pixels, versus 7.768% after clipping;
47.96% of its raw values are negative. A raw background signed mean near zero
(-0.000015) coexists with MAE 0.001916. Clipping and signed averages can hide
errors; they were not used to declare equivalence.

All 72 matched light/dark method pairs have identical Dice and region counts;
the maximum absolute retention-fraction change is 1.28325e-6.

Important limits:

- A common numeric radius does not imply equivalent geometry or equivalent
  radius-to-object size. Objects scale with the fixture dimensions.
- Kernel support and edge conventions differ; edge-only error is not isolated.
- Truth precedes common noise and input rounding. Overlap labels have one owning
  object; their per-object summaries do not deblend overlapping signal.
- One fixed truth-derived threshold tests sensitivity, not biological counting
  accuracy or optimized downstream settings. For example, CPU rolling ball gives
  378 connected regions versus 36 truth regions on the 1024² mixed phantom.
- These include explicit 2D, slice-wise XY and true 3D cases, not time/channel
  metadata qualification or physical anisotropy testing.

## Recommendation

1. Keep current rolling ball, its defaults and the existing GPU implementation.
2. If a fast CPU alternative is added, name the algorithm clearly, for example
   **Morphological Background Subtraction**, with an explicit footprint/method.
   Start by considering the SciPy box implementation, not an "ITK version" label.
3. Explain the size/background trade-off and validate on representative real data
   before choosing defaults. Do not silently substitute it into existing graphs.
4. Investigate an explicit intensity-aware rolling-ball option separately; the
   normalized-input finding is scientifically more important than a small speed
   ranking. No such behavior change was made in this work.
5. Retain ITK ball/reconstruction as candidates only if their distinct behavior
   meets a concrete need. The benchmark does not justify adding them for speed.

No background node, active installation, commit, push or release was changed.
The separately approved exact-median implementation still awaits its clean full
repository regression run; this background benchmark is not that qualification.
