# Richardson-Lucy GPU execution with numerical advisories

Implemented, unreleased after 0.15.0a2; maintainer decision, 2026-09-08.

The maintainer explicitly prefers broader GPU execution with warnings over
requiring CPU agreement before execution. This supersedes the v2 admission
envelope and the unmerged August 13 experiment at `3965f12`.

## Execution contract

The existing CuPy/CuPyX kernels, axes, authored parameters, precision, boundary
handling and iterative algorithm are unchanged. Parameter/workload policies are
v3 and implementation declarations are version 2, so older benchmark and cache
identities cannot stand in for this admission contract.

Ordinary RL accepts 1–500 iterations and filter epsilon 0–1. RL-TV accepts
1–100 iterations, regularization 0–0.1, filter epsilon 0–0.001, TV epsilon
1e-12–0.01 and denominator floor 1e-6–1. These are the authored UI ranges.
Even-sized PSFs, PSFs larger than their image, and nondefault normalization,
clipping and scale-preservation options no longer force CPU execution.

Hard requirements still include complete finite float32 input facts, matching
2D/3D image/PSF axes, nonempty dimensions, positive PSF mass, supported runtime
and sufficient memory. Positive TV requires at least two samples per spatial
axis. Invalid parameters must not be silently clamped by GPU selection.

## Numerical interpretation and presentation

The v2 prequalified envelope remains a reference for advisory coverage, not an
execution gate. Even PSFs get a specific centering advisory; oversized PSFs and
broader parameter settings get coverage advisories. These warnings are distinct
from runtime failures and do not turn a GPU result badge into a CPU fallback.
The selected node's Compute section labels stale advisories as belonging to a
previous result. Execution provenance version 2 retains the messages, including
in batch/generated execution; actual CPU fallback clears GPU advisories.

The existing CPU/GPU comparison remains available as diagnostic/benchmark
evidence. A failed comparison does not prevent an explicit GPU execution choice.
Auto and Find fastest retain their separate performance/comparison criteria;
this change does not certify unlike results as equivalent or share CPU/GPU caches.
Users who want broad execution can select Prefer GPU or pin GPU in Custom mode.

Historical GPU qualification reports retain their original measurements and
dates. This change claims broader executable support, not fresh cross-platform
scientific equivalence or new speed measurements.

The [companion manual change](https://github.com/rensutheart/vipp-mkdocs/commit/5263dc7d90e6d26786f19c893bb21051eb4391e8)
updates choosing compute and GPU validation status, marked unreleased. Public
user instructions remain in the manual.

## Validation record

Windows, 2026-09-08: the focused policy/planner/contract/provenance/inspector
suite passed 193 tests. The dedicated warning-policy suite passed all 30 tests,
including eight real CUDA pipeline executions on an NVIDIA GeForce RTX 4050
Laptop GPU with the installed VIPP 0.15.0a2 CUDA 13 runtime. The application
source under test was this checkout; the installed application was not changed.

Those real runs covered both Prefer GPU and strict Custom selection, a
500-iteration ordinary RL run, even and oversized PSFs, and a 100-iteration
3D RL-TV run with broader parameters. They checked GPU selection without a
CPU qualification record, finite float32 output with the original shape,
unchanged read-only inputs and authored parameters, and carried advisories.
They establish executable behavior for these cases, not CPU equivalence or
restoration quality throughout every newly admitted parameter combination.

Manifest validation, lint, wheel/sdist build, manual content/route checks and
the strict manual build passed. Inspector advisories were visually checked at
360 pixels in light and dark themes; the manual was inspected at desktop and
phone widths in both themes.

The GPU admission manifest names the new implementation versions and runs the
advisory-execution suite in both profiles. Its 22 regression tests passed, as
did 23 benchmark-coordinator tests (one real-CUDA case skipped in the development
environment) and all 19 Measurements evidence-validator tests.

Changing the shared policy source invalidates the source-freshness check for
the historical RTX 5090 Measurements artifact. The measured JSON and Markdown
remain unchanged. Its regression verifies preserved contract/content integrity
and requires the strict freshness validator to reject the stale source claim;
it does not rewrite hashes to make an old measurement look current.

The full 9,161-test development run was stopped at 88% after cascading Qt/CuPy
failures and mesh-provider failures made that environment unsuitable for a
clean full-suite result. Six representative display/CUDA failures were
reproduced against an untouched `main` snapshot, including CuPy FFT cache
cleanup exceptions. This is an incomplete, non-green regression run, not a
release qualification. The directly affected historical-evidence, benchmark
coordinator and admission-manifest regressions were corrected and passed their
focused reruns. A final planner/batch rerun passed all 81 tests, including both
CPU fallback paths clearing GPU warnings and restoring the CPU version.

Windows, 2026-09-09: the interactive GPU sweep catalogue still pinned both RL
implementation versions to 1, causing four provider-free catalogue tests to
reject the current version-2 admission manifest. The catalogue now explicitly
reviews version 2 for both operations while continuing to reject stale and
unknown versions. Its delegated rows distinguish bounded diagnostic CPU/GPU
comparisons from the broader v3 advisory-execution contracts; these rows do not
execute RL sweeps, certify broader CPU equivalence, or refresh historical
measurements. No provider, execution policy, comparison gate, or historical
benchmark artifact was changed.

All 27 sweep tests passed (the original 19 plus eight version/policy regressions).
The related warning-policy, support-policy, planner, benchmark-coordinator,
RL characterization, diagnostic PSF-harness, and admission-manifest tests passed
211 low-cost tests across a combined run and a separate admission skip-integrity
regression. Nine real-CUDA cases were deliberately not rerun. Targeted lint
passed. This catalogue-only correction required no new CUDA execution and is
not a full-suite or cross-platform release qualification.
