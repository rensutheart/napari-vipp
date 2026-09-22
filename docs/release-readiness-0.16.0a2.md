# 0.16.0a2 integration readiness

Integration review: 2026-09-22 UTC. This is an unreleased development integration,
not a published 0.16.0a2 artifact. Package metadata remains 0.16.0a1 until release
preparation. No additional feature is required for this alpha's agreed scope.

## Included work

- Registration: Estimate Registration, Apply Transform and Compare Images;
  whole-volume time series, three synthetic examples, diagnostics and explicit
  transform exports, and the undoable Add/Show Apply Transform suggestion.
- Exact CPU Median Filter acceleration, preserving the qualified SciPy behavior,
  fallback domain and existing GPU implementation. Other filters are unchanged.
- Skeletonize Labels and Analyze Skeleton per Label, original-object identity
  and morphology joins, and the calibrated per-label skeleton example.
- Closed inspector dropdown wheel protection and logarithmic plot tick rendering.
- Companion manual at `rensutheart/vipp-mkdocs` commit `e7a706b`, with unreleased
  behavior labelled separately from the 0.16.0a1 manual.

Feature commits: median `6b5c1b8`, registration `7506199`, per-label skeleton
`3d3d067`, plot labels `64b4253`. The combined catalogue has 27 examples.

## Evidence and integration checks

Original independent full-suite results are retained, not re-labelled as combined
validation: registration **11,877 passed / 30 skipped / 2 expected failures**;
median **12,134 passed / 30 skipped / 2 expected failures**. Registration's later
inspector follow-up has its own focused evidence. Per-label skeleton and log-label
changes have focused passing evidence, not a completed clean independent full run.

On the combined source:

- 891 core, cache, median, registration, skeleton and plot tests passed.
- 157 example/inventory/scientific-golden tests passed. Preservation regressions
  recover both independent feature goldens and the original pre-addition golden.
- 130 inspector/Next step/dropdown, safe-bypass and dependency-packaging checks
  passed; 28 documentation and architectural checks passed.
- All three exhaustive example copies match their generator: 152 nodes and
  185 connections, including 70 tunnel subscriptions and 115 direct connections.
- Ruff, plugin manifest and source/wheel build checks passed. The combined wheel
  installed into a private disposable environment passed both native Windows
  SimpleITK smoke checks (median exactness and registration, including calibrated
  3D rigid resampling, wide label IDs and transform JSON). SimpleITK was 2.5.6 / ITK
  5.4; the registration smoke's maximum landmark error was 0.00640336 micrometers.
  This is not a clean dependency solve or native Linux/macOS qualification.
- Combined manual content contracts, 50 application routes, strict build and
  changed-page desktop/narrow light/dark review passed.

Machine-local logs and JUnit are retained under
`VIPP-local-tests/a2-integration-20260922`; manual QA is under
`VIPP-local-tests/manual-a2-integration-20260923`. The first directory uses UTC
dating; the manual review crossed midnight locally. Different test runs may
overlap; their counts must not be added into a claimed complete-suite total.

## Release gates still required

1. Normal exact-commit CI on the integrated main branch: supported OS/Python
   suites, clean wheel/source installs and both native SimpleITK smoke checks.
   Do not substitute the earlier branch-local full suites for this result.
2. Native macOS package validation on both Apple Silicon and Intel for the new
   SimpleITK dependency. Windows evidence does not qualify either architecture.
   The affected Windows installer dependency-install path must also pass when
   its a2 artifact is prepared; the current installed VIPP was not modified.
3. Version/changelog/release notes and numbered manual preparation, followed by
   exact final main CI, immutable tag, artifact metadata/reproducibility/hash
   checks and publication according to the alpha release runbook.

Changed domains are core/UI, scientific CPU dispatch, workflow/export/provenance,
dependency/toolchain, packaging/native installers and documentation. GPU kernels,
Gaussian/background subtraction/deconvolution methods and installer UX are not
replaced by this integration. Carry forward unchanged-domain evidence only within
the release runbook's explicit limits. This review does not authorize publication.

## Preserved historical work

Older dirty detached review checkouts and retained release artifacts are not
current feature branches. They are preserved outside this integration, not
blindly committed or deleted. Branch cleanup removes only heads proven merged
into main; source worktrees can remain detached at their recorded commits.
