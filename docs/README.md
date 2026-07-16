# VIPP Documentation

This index separates current user guidance from implementation references and
historical planning. Start with the first section unless you are developing or
reviewing VIPP itself.

## Use VIPP

- [User guide](user-guide.md): build, inspect, save, export, and batch-run
  workflows.
- [Image import and export](io-user-guide.md): supported formats, optional
  microscope readers, batch input binding, and output choices.
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
- [Release runbook](release-runbook.md): alpha release verification and
  publication steps.
- [Research and publication record](research-and-publication.md): evidence
  boundaries, evaluation plan, and reproducibility artifacts.

## Current Planning

- [Planning and roadmap](planning.md) is the source of truth for release order
  and active priorities.
- [Production GPU implementation plan](gpu-production-implementation-plan.md)
  defines the Windows/Linux NVIDIA path, macOS CPU contract, provider choices,
  and release gates.
- [cuCIM native-Windows port plan](cucim-windows-port-plan.md) defines the
  upstream-tracking fork, MSVC/Clara port, CUDA/Python wheel matrix, validation,
  distribution, installation, and upstream contribution work.
- [App improvements plan](app-improvements-plan.md) records the completed UI,
  graph-feedback, and RL-TV safety work packages.
- [Node roadmap](node-roadmap.md) tracks current capability gaps by node family.
- [MitoMorph feature parity](mitomorph-feature-parity.md) tracks remaining
  measurement goals.
- Optional GPU backend research is intentionally isolated on the
  [`codex/gpu-cross-platform-support`](https://github.com/rensutheart/napari-vipp/tree/codex/gpu-cross-platform-support)
  branch. Main remains the CPU production baseline; that branch contains
  feasibility benchmarks, cross-platform provider evidence, and future-looking
  implementation plans rather than a supported GPU execution mode.

## Implementation Records

These pages preserve accepted architecture, completed phases, scientific
reasoning, and deferred work. They are useful design records, but they do not
override the current planning documents above.

- [OME import and export](ome-io-plan.md)
- [PSF and deconvolution](psf-and-deconvolution-plan.md)
- [GPU feasibility spike](gpu-acceleration-spike.md)
- [cuCIM native-Windows source evaluation](cucim-windows-source-evaluation.md)
- [Context-aware controls audit](context-aware-controls-audit.md)
- [Object and mesh morphology](object-mesh-morphology-plan.md)
- [Colocalization and RACC](colocalization-racc-plan.md)
- [Historical node-roadmap discussion](node-roadmap-history.md)
