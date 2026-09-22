# SimpleITK CPU qualification

Status: unreleased after **0.16.0a1**. Scope approved 2026-09-22: exact CPU
median acceleration and a bounded assessment of further heavy operations.
Registration, new filtering nodes and GPU changes are outside this change.

## Scientific contract

`core/simpleitk_filters.py` implements the existing `Median Filter` operation,
not a new filtering method. `operations.median_filter` retains its parameter
canonicalization and channel-axis validation. The footprint is the final two
spatial axes (XY); Z, time, channel and other leading dimensions are independent.
Metadata, units, input shape and output dtype do not change.

The authoritative reference remains
[`scipy.ndimage.median_filter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html) with an XY
box footprint and default half-sample-symmetric `reflect` boundaries. The adapter
adds a symmetric halo before calling SimpleITK and discards it afterwards;
SimpleITK's own edge extension cannot affect retained pixels. A scalar 3D buffer
with radius zero along the plane axis prevents cross-plane/channel mixing.
Both implementations select an existing sample, with no intensity conversion.
The accelerated primitive is
[`SimpleITK.MedianImageFilter`](https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1MedianImageFilter.html);
VIPP's padding and axis adapter, not the filter name, establishes compatibility.

The qualified dtype family is native-endian uint8/16/32, int8/16/32 and finite
float32/64 without negative zero. Boolean, 64-bit integer, non-native-endian,
nonfinite and negative-zero inputs retain the original SciPy behavior. Unsupported
SciPy inputs do not become implicitly converted inputs. Small/degenerate XY
planes and unqualified footprints also retain SciPy. Production dispatch requires
at least 65,536 XY pixels, both XY extents at least the canonical footprint size,
and an odd footprint between 3 and 51. These are conservative eligibility gates,
not a guarantee of a speed gain for every input or hardware configuration.

This is an exact compatibility claim for the tested domain, not a claim that
every possible image or native build has been exhaustively verified. Original
SciPy handling of wide integers and NaN/Inf is preserved, not newly certified.

## Runtime and reproducibility

- SimpleITK 2.5.6 is pinned as a normal dependency and imported lazily. Native
  import or execution failures remain visible; they are not silently hidden by
  a fallback after execution begins.
- Inputs are never mutated. Non-contiguous/read-only layouts are copied into
  private work buffers, and output owns its contiguous memory.
- Each filter uses at most 12 CPU threads/work units without changing global
  ITK settings. Padded plane chunks target 64 MiB; this is not a total process
  memory cap. Output, conversion copies and at least one complete plane still
  require additional memory.
- The composite CPU dispatcher has a new implementation identity. SimpleITK's
  package version participates in CPU median cache identity and environment,
  batch and reproducibility inventories. The CPU library label describes the
  dispatcher; it does not falsely claim that a fallback call executed in ITK.
- GPU implementations and their scientific policies are unchanged. GPU parity
  checks must continue to cover the original SciPy reference, not only compare
  two new implementations to each other.

## Qualification layers

`test_simpleitk_median.py` compares bytes, shape and dtype against the independent
SciPy path. It covers every odd footprint from 1 through 51, eight dtypes,
extrema/subnormals, edge-only and degenerate images, explicit/absent channel axes,
leading dimensions, read-only strided input, chunk boundaries, lazy imports,
unchanged global settings and visible backend failures. Private candidate tests
exercise tiny inputs even when production dispatch would retain SciPy.

Shared-execution integration checks cover workflow round trips, generated Python,
calibration, batch output and the real CPU execution path. Cache tests require
the SimpleITK version and invalidate median/downstream evidence on upgrades.

`scripts/smoke_simpleitk_install.py` checks actual native import and admitted
2D/3D read-only uint16 medians against SciPy, including borders. Existing
clean-install CI and native macOS installer jobs run this check. Windows local
execution is not evidence of native macOS or Linux success; those remain CI
gates before a release. The macOS conda package can link a different ITK build
than the PyPI wheel, so the smoke output records the reported ITK version.

## Further operations

The bounded benchmark compares candidate outputs before recording timings,
including conversion/padding costs, warmup and repeated calls. Candidate code
is not production dispatch until both exactness and a useful speed gain pass.

- Binary opening/closing need an identical footprint, origin and boundary
  convention at both primitive stages. A radius alone does not reproduce an
  even-size SciPy footprint.
- Euclidean distance requires matching foreground/background and contour
  conventions. An ITK distance-map name alone is not sufficient evidence.
- Rolling-ball background uses rolling-ball processing with mean presmoothing,
  not a median filter. Faster median calls therefore do not accelerate it.
  A morphological opening is not an equivalent replacement.
  The [completed expanded comparison](benchmarks/background-subtraction-2026-09-22.md)
  covers 66 configurations: current GPU output matched CPU exactly, while a
  separately named morphological method could be useful. Matched SciPy box
  opening was faster than SimpleITK. Memory measurements were invalid and are
  excluded; no background method was switched.
- Recursive Gaussian filtering differs from the existing finite-kernel Gaussian.
  Keep the current method. Keep connected components and RL/RL-TV unchanged;
  previous evaluation did not establish an appropriate replacement.

## Local results — 2026-09-22

The [completed benchmark report](benchmarks/simpleitk-cpu-qualification-2026-09-22.md)
and its raw JSON retain all 50 cases and three repetitions per backend. All
checked candidate outputs were bitwise identical. The 26 admitted median cases
showed **2.02–6.41×** warm speedups, including all conversion/halo/copy costs.
The existing 65,536-pixel gate is retained. Smaller images keep SciPy.

| Median workload, uint16, width 5 | Original SciPy | New CPU dispatcher |
| --- | ---: | ---: |
| 2048 × 2048 | 1.329 s | 0.295 s |
| 32 × 512 × 512, independent XY planes | 2.677 s | 0.637 s |

These are single-machine Windows results, not cross-platform speed promises.
First-call lazy loading is separate: a 512 × 512 uint16 call took 124 ms in a
fresh process, versus 23 ms immediately afterward. Further-hotspot assessment
found promising large-image distance-transform gains but insufficient production
qualification; morphology gains were inconsistent, and mean presmoothing differed
numerically. No additional operation has been switched.

Completed local checks:

- 354 focused median cases, including bytewise comparison and fallback contracts.
- 10 shared workflow/export/executor/batch integration cases.
- 511 focused compute/cache/policy/execution/UI/GPU cases: two stale test dependency
  inventories were corrected, followed by a clean rerun of all 127 cache/planning
  cases. The GPU group exercised the real device and the independent SciPy oracle.
- 27 benchmark-harness cases; native packaging and installed-wheel smoke checks.
- Repository Ruff, plugin manifest validation, source/wheel build and private
  wheel installation. No changes to the user's active installed environment.
- Companion manual content checks, strict build and desktop/narrow light/dark QA.

### Final local confirmation

The clean confirmation completed on **2026-09-22 at 16:51 UTC**: **12,134 passed,
30 skipped and two expected failures**, with no unexpected failures, in
**2,683.14 seconds (44m43s)**. Ruff and plugin-manifest validation also exited
successfully. The expected failures remain the two documented CuPy integer
Gaussian parity cases, not median failures. Native-platform, opt-in public-data,
OpenGL and other explicitly skipped checks are not claimed as tested.

Machine-local evidence is retained in
`simpleitk-qualification-20260922/confirmed-20260922-180605-42252/`:
`full-pytest.log`, `full-pytest.xml`, `ruff.log`, `manifest.log`, `state.json` and
`source-files.json`; `validation-confirmed-state.json` is the final state pointer.
JUnit records 12,166 cases, zero errors/failures and 32 skipped entries because
its skipped total includes the two expected failures. The confirmation runner
compared the full source fingerprint before and after the checks and accepted
the run only when unchanged. A read-only follow-up verified all 998 recorded
files and the complete file inventory still matched before this documentation
update; the source-fingerprint file SHA-256 is
`6c9f1abaa7ad9c7703c6e6d16cc9c9d54c1be6502b087f860e7a60b970f3cb6f`.

The earlier source/wheel build, private wheel installation and native installed-
package smoke remain applicable to the unchanged production implementation:
all 252 production Python files match the installed wheel byte-for-byte.
That smoke passed on Windows AMD64 with SimpleITK 2.5.6 / ITK 5.4, checking
read-only uint16 arrays of shape 256 × 257 and 3 × 256 × 257 at width 5 against
SciPy. It was not rerun during this evidence-only follow-up, and no active
installation or environment was changed.

The final local outcome was **delivered to the user in the scheduled 16:59 UTC
check on 2026-09-22**. Local confirmation is complete; native Linux and macOS
qualification remains **untested here** and a separate release-CI gate. This
does not authorize or claim a commit, push, installer publication or release.

### Historical runs and resolved blockers

The first complete repository run finished with 12,017 passes, 30 skips, two
known expected failures and two failures. One was the interpreter home path
included in benchmark documentation; it has been removed and a privacy
regression test added. Documentation and harness checks then passed (54 tests).
The other was a test subprocess dropping the private Centrosome dependency
directory from its environment, before batch execution; the test is identical
to the released baseline. A disposable validation environment corrected this,
and the batch restart plus median integration group passed (11 tests). No
application algorithm changes were required.

The second complete run finished with 12,017 passes, 31 skips, two known expected
failures and two failures in installer import checks. Those checks use isolated
Python mode, which also needed the source worktree in the disposable environment's
private `.pth`. After that environment-only correction, both installer checks and
the batch restart check passed together (3 tests). No application changes were
needed. Neither of these historical runs was wholly green; their targeted
rechecks alone did not establish the later clean full-suite result above.

The third verification stopped after five failures, with 6,184 passes, nine skips
and two expected failures. The failures depended on live Windows commit headroom:
the unchanged safety guard legitimately declined optional object counts or a CPU
timing comparison. The object-count behavior was reproduced from released-baseline
source with low versus admitted memory snapshots. Test-only deterministic admitted-
memory fixtures now isolate the intended asynchronous/cache/timing checks; explicit
low-memory refusal regressions keep the real production guard covered.
All 66 object-count/diagnostic/feedback checks passed in each private environment,
and both CPU-timing admission/refusal checks passed. No algorithm or allocation
policy was relaxed.

The background benchmark is complete. The then-required fresh full-suite check
was subsequently completed by the clean confirmation above, run sequentially
after the independent registration worker exited. Earlier logs and failure
diagnoses are retained as historical evidence, not current pending work.
Native macOS/Linux qualification remains a separate untested release gate.
