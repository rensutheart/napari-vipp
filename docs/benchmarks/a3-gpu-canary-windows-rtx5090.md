# 0.15.0a3 current-source GPU canary — Windows RTX 5090

**2026-09-09: 119 passed, no failures, errors or skips.** Of these, 35 test
cases execute real CUDA providers; 84 are additional CPU/injected-provider
contract checks. This is a bounded release canary, not full GPU qualification.

The [portable evidence JSON](a3-gpu-canary-windows-rtx5090.json) records exact
selectors, normalized commands/environment, per-stage results, real-CUDA test
IDs, source SHA-256 fingerprints and hashes of the retained logs/XML. The raw
machine-local records remain in `.cache/a3-release-gpu-current/`; they are not
published because they include local paths. No installed VIPP application was
replaced or restarted.

## Source and environment

The run started at **11:59:45 UTC** and finished at **12:01:18 UTC** using
checkout HEAD `88d7cdfbe0a248ced319a0995a363465542280b5`, unchanged throughout.
All 622 captured source/configuration file fingerprints remained identical.
The only uncommitted application-code change was the unrelated startup-splash
layout update, with its test; release documentation/metadata were also dirty.
The tested core files match HEAD. This record therefore does not pretend the
whole checkout was clean or identify a later release tag as the tested commit.

Runtime: Windows 11, Python 3.12.9 in `.venv-gpu-cu13`, package version
`0.15.0a3`, NVIDIA GeForce RTX 5090 (`cuda:0`), CuPy 14.1.1, CUDA runtime 13.2,
driver API 13.3 and display driver 610.74. `PYTHONPATH` selected this checkout's
`src`; Qt was offscreen. `VIPP_RUN_REAL_CUDA_BATCH=1` enabled the normally
opt-in batch/generated-runner/resume tests, and `VIPP_EXPECT_CUDA_DEVICE`
required the RTX 5090. Any selected-test skip would have failed this run.

Compared with the earlier RL review at
`7d4940633a53d57ac275ac2ade5ee8b719b6dd10`, GPU kernels, RL policies, device
execution and the execution-provenance serializer are unchanged. The relevant
new shared scope is batch manifest v6, sealed receipts, output verification,
reproduction guards and batch `--resume`; the execution/pipeline delta also
adds the CPU label-boundary operation. The checks below target those shared
boundaries without repeating full historical numerical/performance sweeps.

## Executed scope

| Stage | Passed | Real CUDA cases | Coverage |
| --- | ---: | ---: | --- |
| RL/RL-TV provider checks | 19 | 19 | Bounded CPU comparisons, leading/spatial dimensions, default phantom checks, synchronized cancellation, residency, private FFT allocation and memory estimates |
| RL advisory contracts | 30 | 8 | Prefer GPU/strict Custom, broad iterations/settings, even/oversized PSFs, finite float32 outputs, unchanged inputs/parameters, warnings and hard-input rejection |
| Shared GPU execution | 12 | 7 | RL/RL-TV cleanup/reuse, batch and generated Python provenance, cancellation, generated cleanup runner, non-RL corridors/table finalization, injected fallback and cleanup failures |
| Batch/resume provenance | 58 | 1 | Complete batch-compute/resume/fresh-process-restart modules: receipts, environment/input/output refusal, no-overwrite publication and completed GPU-result reuse |
| **Total** | **119** | **35** | **Zero failures, errors or skips** |

Counts refer to parameterized test cases, not individual GPU launches. The
first stage selects `real_gpu` tests from the two provider modules; it does
not claim those entire modules ran. Other exact selections are in the JSON.

## Scientific and release limits

Broad RL advisory executions establish executable behavior with recorded
warnings, not CPU/GPU equivalence or restoration quality across the expanded
region. Existing diagnostic CPU comparisons and phantom checks remain strict
and apply only to their tested fixtures; no acceptance threshold was changed.

The real-CUDA resume case closes its registry and reuses an already completed
Gaussian GPU result **within the same Python process**. The separate killed-
process and fresh generated-batch-runner resume checks use CPU fixtures. These
complementary checks do not establish recovery of unfinished GPU work after
process death through the generated batch CLI.

This evidence supplies a current shared-execution canary, not full-catalogue
numerical admission, cross-platform GPU coverage, installed-build validation
or new speed claims. Historical RL and Measurements artifacts remain unchanged
and are not relabelled as source-fresh. Final-source CI, native installers and
release publication remain separately recorded release gates.
