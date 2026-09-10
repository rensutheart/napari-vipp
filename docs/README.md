# Repository documentation

**Visual image processing made approachable.**

VIPP builds **visual workflows for reproducible bioimage analysis**.
The [public manual](https://rensutheart.github.io/vipp-mkdocs/stable/) is maintained
only in vipp-mkdocs. This directory is for contributors and coding agents.

## Start here

- [Documentation ownership and migration map](documentation.md): where to edit,
  what stays here, and where the former user guides moved.
- [Developer notes](developer-notes.md): setup, extension points, and local checks.
- [Architecture](architecture.md): runtime, schema, scientific and UI contracts.
- [Planning](planning.md): active priorities and release order.
- [Durable execution contract](durable-gpu-execution.md): shared execution,
  provenance, fallback, cancellation, and publication invariants.

## Validation and release

- [Release notes](../CHANGELOG.md)
- [Release runbook](release-runbook.md)
- [Release qualification baseline](release-qualification-baseline.md)
- [Windows installer field acceptance](windows-installer-field-acceptance.md)
- [Windows setup packaging](../packaging/windows/README.md)
- [macOS PKG packaging](../packaging/macos/README.md)
- [Full-batch cancellation verification](full-batch-cancellation-verification.md)
- [Verified batch resume contract](verified-batch-resume.md)
- [Public data corpus](public-data-corpus.md)
- [Source-aware loading qualification](source-aware-loading-0.14.0a1.md)
- [Analytical phantom validation](analytical-phantom-validation.md)
- [Richardson-Lucy TV validation](rl-tv-validation-report.md)
- [Research and publication record](research-and-publication.md)

## Current Planning

- [Measurement plots and statistics](measurement-plots-and-statistics-plan.md):
  planned 0.16 results workflow with two general-purpose nodes, batch-table
  collection, independent-sample safeguards, editable pop-outs, and figure export.
- [Registration, image comparison, and template matching](registration-and-template-matching-plan.md):
  planned 0.16 scope for 2D/3D alignment, SSIM and related comparisons, template
  detection, and transform contracts; drift/rigid/affine follow-ups are separate.
- [Update discovery and reader packaging](update-and-reader-plan.md): quiet
  update checks and the proposed default-reader dependency set.

- [Planning and roadmap](planning.md) is the source of truth for release order
  and active priorities.
- [Product ideas](product-ideas.md) preserves promising concepts that are not
  committed to a release, including conditional batch routing.
- [Planning history through 0.13.0a7](planning-history-0.13.md) preserves the
  delivered chronology and detailed qualification rationale removed from the
  active roadmap.
- [Desktop startup and installer plan](desktop-startup-and-installer-plan.md)
  defines the branded launch profiles, napari loading host, and staged
  Windows/Linux/macOS installer path.
- [Windows installer and planning contract](windows-installation-planner.md)
  documents the read-only plan schema, transactional managed
  executor, update/repair ownership rules, and signed-release boundary.
- [Production GPU implementation plan](gpu-production-implementation-plan.md)
  defines CPU/Auto/Prefer-GPU/Custom behavior, per-node and graph-global
  benchmarking,
  the Windows/Linux CUDA path, Apple-provider investigation, implementation-
  library choices, installation UX, and release gates.
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

- [Reproducibility-package export](reproducibility-package.md): portable recipes,
  archived run evidence, privacy review, readable reports and explicit limits.

- [OME import and export](ome-io-plan.md)
- [PSF and deconvolution](psf-and-deconvolution-plan.md)
- [GPU feasibility spike](gpu-acceleration-spike.md)
- [GPU Phase 1 implementation record](gpu-phase1-implementation-report.md)
- [GPU Phase 2B Richardson-Lucy implementation record](gpu-phase2b-rl-implementation-report.md)
- [GPU Phase 2C Richardson-Lucy TV implementation record](gpu-phase2c-rl-tv-implementation-report.md)
- [Richardson-Lucy GPU execution with numerical advisories](rl-gpu-warning-policy.md)
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
- [Current CuPy Basic Measurements CPU/GPU evidence](benchmarks/measurements-cupy-windows-rtx5090.md)
- [0.13.0a8 full GPU admission aggregate](benchmarks/gpu-admission-0.13.0a8-windows-rtx5090.json)
- [Preserved historical cuCIM Measurements evidence](benchmarks/measurements-cucim-windows-rtx5090.md)
- [cuCIM native-Windows source evaluation](cucim-windows-source-evaluation.md)
- [Archived cuCIM native-Windows port plan](cucim-windows-port-plan.md)
- [Context-aware controls audit](context-aware-controls-audit.md)
- [Object and mesh morphology](object-mesh-morphology-plan.md)
- [Colocalization and RACC](colocalization-racc-plan.md)
- [Historical node-roadmap discussion](node-roadmap-history.md)
