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
- [0.16.0a2 integration readiness](release-readiness-0.16.0a2.md)
- [Release qualification baseline](release-qualification-baseline.md)
- [Windows installer field acceptance](windows-installer-field-acceptance.md)
- [Windows setup packaging](../packaging/windows/README.md)
- [macOS PKG packaging](../packaging/macos/README.md)
- [Full-batch cancellation verification](full-batch-cancellation-verification.md)
- [Verified batch resume contract](verified-batch-resume.md)
- [Public data corpus](public-data-corpus.md)
- [Source-aware loading qualification](source-aware-loading-0.14.0a1.md)
- [Analytical phantom validation](analytical-phantom-validation.md)
- [SimpleITK CPU qualification](simpleitk-cpu-qualification.md): exact median
  compatibility, bounded dispatch, packaging and further-hotspot assessment.
- [Background-subtraction comparison protocol](benchmarks/background-subtraction-protocol-2026-09-22.md):
  full-operation CPU/GPU timings and scientific differences for exploratory
  SimpleITK estimators; not production replacements.
- [Completed background-subtraction comparison](benchmarks/background-subtraction-2026-09-22.md):
  66 configurations, speed/quality trade-offs, exact GPU and box-control agreement,
  normalized-input caveat and invalid-memory-measurement disclosure.
- [Seeded segmentation](seeded-segmentation.md): 2D CellProfiler Propagation,
  acquired-image examples, existing 3D watershed and Random Walker evaluation.
- [Statistics-paper reproduction](validation/statistics-paper-reproduction.md):
  the recorded CellProfiler compartment profile, independent executable checks,
  acquired-input provenance and comparison with archived paper measurements.
- [Paper statistics/plot integration](validation/paper-statistics-plots-integration.md):
  existing descriptive tools alongside the compartment profile, fixture bridge
  checks and inherited batch UI qualification limits.
- [Richardson-Lucy TV validation](rl-tv-validation-report.md)
- [Research and publication record](research-and-publication.md)

## Current Planning

- [Batch measurement collection](measurement-collection.md): unreleased
  collection/export contract, typed datasets, Table Source, and validation record.
- [Measurement plots and statistics](measurement-plots-and-statistics-plan.md):
  approved descriptive-only 0.16 scope; implemented, unreleased collection and
  plotting plus versioned object/image/sample summaries. No inference phase.
- [Statistics contract](statistics.md): descriptive methods, weighting,
  exclusions, output counts/units, and explicit legacy-recipe migration.
- [Registration, image comparison, and template matching](registration-and-template-matching-plan.md):
  earlier design rationale and revised delivery scope; template detection is deferred.
- [Registration implementation](registration-implementation.md): unreleased reusable
  transforms, pairwise and whole-volume time-series alignment, label-safe application
  and comparison contracts. [Synthetic qualification](registration-synthetic-qualification.md)
  records independently known motion and measured landmark error.
- [Update discovery and reader packaging](update-and-reader-plan.md): quiet
  update checks and the proposed default-reader dependency set.

- [Planning and roadmap](planning.md) is the source of truth for release order
  and active priorities.
- [Product ideas](product-ideas.md) preserves promising concepts that are not
  committed to a release, including conditional batch routing.
- [AI-assisted nodes, workflows, and iterative analysis](ai-assisted-authoring-plan.md):
  future, unassigned scope for description-driven authoring, system-generated
  custom-node integration, and bounded image-guided refinement.
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
- [Per-label skeletons](per-label-skeleton.md): original-object identity,
  thinning and graph conventions, optional skeleton input, calibration,
  empty/fragment accounting, resource boundaries and regression evidence.
- [Colocalization and RACC](colocalization-racc-plan.md)
- [Historical node-roadmap discussion](node-roadmap-history.md)
