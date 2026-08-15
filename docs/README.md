# VIPP Documentation

**Visual image processing made approachable.**

The supporting scientific promise is **visual workflows for reproducible
bioimage analysis**.

This index separates current user guidance from implementation references and
historical planning. Start with the first section unless you are developing or
reviewing VIPP itself.

## Use VIPP

- [Quick start](quick-start.md): the signed-Windows-installer experience,
  manual Linux/macOS and advanced routes, CPU/GPU choices, and a first
  workflow. Start here if you are installing VIPP rather than developing it.
- [User guide](user-guide.md): build, inspect, save, export, and batch-run
  workflows.
- [GPU guide](gpu-guide.md): optional GPU qualification, compute modes,
  supported operation families, fallback, benchmarking, cuCIM, and
  reproducibility.
- [Image import and export](io-user-guide.md): supported formats, optional
  microscope readers, batch input binding, and output choices.
- [Durable GPU execution](durable-gpu-execution.md): batch/config migration,
  generated Python and CLI overrides, exact provenance, OOM fallback,
  two-level progress, cancellation, and publication safety.
- [Full collection-batch cancellation verification](full-batch-cancellation-verification.md):
  implementation status, Apple Silicon evidence, Windows/GPU evidence, and the
  shared manual smoke procedure.
- [Cache and memory](cache-and-memory.md): cache modes, memory guard, and
  large-data tradeoffs.
- [Operator tips](operator-tips.md): background work, cancellation, previews,
  and responsive operation.
- [Example workflows](../examples/README.md): every bundled workflow and its
  intended review purpose.

## Scientific Workflows And Methods

- [Measurement workflows](measurement-workflows.md): object, intensity, mesh,
  skeleton, colocalization, and table contracts.
- [Skeleton nodes](skeleton-nodes.md): skeleton inputs, visual QC, and graph
  measurements.
- [Colocalization method notes](colocalization-method-notes.md): definitions,
  assumptions, and publication cautions.
- [Analytical phantom validation](analytical-phantom-validation.md): generated
  validation results for calibrated morphology.
- [Richardson-Lucy TV validation](rl-tv-validation-report.md): deterministic
  convergence, feature-recovery, PSF-sensitivity, and parameter evidence.

## Develop And Release

- [Developer notes](developer-notes.md): contributor entry point and local
  checks.
- [Architecture](architecture.md): runtime model, metadata, UI, persistence,
  export, and known gaps.
- [Release notes](../CHANGELOG.md): categorized compatibility, architecture,
  scientific-behavior, workflow, UI, and validation changes by version.
- [Release runbook](release-runbook.md): risk-based alpha, release-candidate,
  and production publication with change-triggered gates.
- [Release qualification baseline](release-qualification-baseline.md):
  reusable installer, GPU, schema, packaging, and documentation evidence plus
  exact invalidation rules.
- [Windows installer field acceptance](windows-installer-field-acceptance.md):
  a short record for exact-artifact CPU, CUDA, rollback, path, and novice checks.
- [Windows setup packaging](../packaging/windows/README.md): the same-tag wheel,
  PyInstaller, Authenticode, licence, checksum, and release-asset boundary.
- [Research and publication record](research-and-publication.md): evidence
  boundaries, evaluation plan, and reproducibility artifacts.

## Current Planning

- [Planning and roadmap](planning.md) is the source of truth for release order
  and active priorities.
- [Desktop startup and installer plan](desktop-startup-and-installer-plan.md)
  defines the branded launch profiles, napari loading host, separate local-build
  cuCIM bundle, and staged Windows/Linux/macOS installer path.
- [Windows installer and planning contract](windows-installation-planner.md)
  documents the read-only plan schema, transactional managed
  executor, update/repair ownership rules, and signed-release boundary.
- [Production GPU implementation plan](gpu-production-implementation-plan.md)
  defines CPU/Auto/Prefer-GPU/Custom behavior, per-node and graph-global
  benchmarking,
  the Windows/Linux CUDA path, Apple-provider investigation, implementation-
  library choices, installation UX, and release gates.
- [cuCIM native-Windows port plan](cucim-windows-port-plan.md) defines the
  upstream-tracking fork, MSVC/Clara port, CUDA/Python wheel matrix, validation,
  distribution, installation, and upstream contribution work.
- [App improvements plan](app-improvements-plan.md) records the completed UI,
  graph-feedback, and RL-TV safety work packages.
- [Node roadmap](node-roadmap.md) tracks current capability gaps by node family.
- [MitoMorph feature parity](mitomorph-feature-parity.md) tracks remaining
  measurement goals.
- [Durable GPU execution](durable-gpu-execution.md) records the current public
  Auto/Prefer-GPU/Custom behavior, CPU fallback, environment qualification,
  provenance, and batch/generated-Python surfaces. Historical GPU phase pages
  below remain implementation evidence rather than installation instructions.

## Implementation Records

These pages preserve accepted architecture, completed phases, scientific
reasoning, and deferred work. They are useful design records, but they do not
override the current planning documents above.

- [OME import and export](ome-io-plan.md)
- [PSF and deconvolution](psf-and-deconvolution-plan.md)
- [GPU feasibility spike](gpu-acceleration-spike.md)
- [GPU Phase 1 implementation record](gpu-phase1-implementation-report.md)
- [GPU Phase 2B Richardson-Lucy implementation record](gpu-phase2b-rl-implementation-report.md)
- [GPU Phase 2C Richardson-Lucy TV implementation record](gpu-phase2c-rl-tv-implementation-report.md)
- [GPU Phase 3A Canny and Otsu implementation record](gpu-phase3-canny-otsu-implementation-report.md)
- [GPU Phase 4 Sigma Filter implementation record](gpu-phase4-sigma-filter-implementation-report.md)
- [GPU Phase 5 Connected Components implementation record](gpu-phase5-connected-components-implementation-report.md)
- [GPU Phase 6 basic Measurements implementation record](gpu-phase6-measurements-implementation-report.md)
- [Representative real-acquisition ND2 GPU benchmark](benchmarks/representative-nd2-phase1-benchmark.md)
- [Richardson-Lucy large-stack CPU/GPU timing](benchmarks/rl-cupy-performance-windows-rtx5090.md)
- [Richardson-Lucy TV admission matrix](benchmarks/rl-tv-cupy-admission-windows-rtx5090.md)
- [Richardson-Lucy TV large-stack CPU/GPU timing](benchmarks/rl-tv-cupy-performance-windows-rtx5090.md)
- [Canny/Otsu exact-mask CPU/GPU evidence](benchmarks/canny-otsu-cupy-windows-rtx5090.md)
- [Connected Components exact-label CPU/GPU evidence](benchmarks/connected-components-cupyx-windows-rtx5090.md)
- [Basic Measurements CPU/GPU evidence](benchmarks/measurements-cucim-windows-rtx5090.md)
- [cuCIM native-Windows source evaluation](cucim-windows-source-evaluation.md)
- [Context-aware controls audit](context-aware-controls-audit.md)
- [Object and mesh morphology](object-mesh-morphology-plan.md)
- [Colocalization and RACC](colocalization-racc-plan.md)
- [Historical node-roadmap discussion](node-roadmap-history.md)
