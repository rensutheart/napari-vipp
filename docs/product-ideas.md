# VIPP Product Ideas

Last reviewed: 2026-09-30

This page preserves promising product concepts that are not committed to an
active release. An idea moves into the [active roadmap](planning.md) only after
it has a clear user outcome, dependency boundary, scientific and persistence
semantics, acceptance evidence, and a release owner. Presence here is not a
delivery promise.

The batch-routing ideas originated in early tester Tom Naber's feedback on
[pull request 13](https://github.com/rensutheart/napari-vipp/pull/13). The active
roadmap adopts the immediately actionable SourceItem, per-sample parameter,
volume-crop, and pyramid-preview outcomes while preserving the broader routing
concept here for later review.

## Conditional Batch Routing And Convergence

### User need

A batch may need different processing branches for different SourceItems. For
example, one structure may use Otsu while another uses an item-specific manual
threshold, after which both routes should continue through the same watershed
and measurement chain.

The first committed solution is explicit
[per-sample batch parameters](planning.md#014-c-per-sample-batch-parameters),
which covers the common threshold case without changing graph topology.
Conditional routing remains an idea for cases where parameters alone are
insufficient.

### Possible safe shape

- `Route by batch item` consumes data plus immutable `BatchItemContext` and
  chooses among named, typed branches using explicit SourceItem fields.
- An inactive route carries a typed inactive-branch state, not `None`, because
  `None` already represents missing, failed, or not-calculated data.
- `Select active branch` or `Conditional merge` has named inputs, requires
  exactly one active compatible value, and rejects zero or multiple active
  values.
- Static validation checks every possible branch for compatible port type,
  semantic image/table kind, axes, grid, and metadata contracts.
- The scheduler prunes inactive work before scientific operations, writers,
  transfers, or optimizer timing. Inactive outputs are not applicable rather
  than failed.
- The selected route and predicate inputs are recorded per item in provenance,
  checkpoints, and effective scientific hashes.

### Reasons it is not on the active roadmap

VIPP can preserve its current one-connection-per-target-port invariant because a
future merge would use distinct named input ports. The missing architecture is
the typed inactive value, scheduler pruning, hashing/cache behavior, error and
side-effect semantics, GPU planning, replay, and provenance. That remains a
substantial control-flow and schema feature and should be reconsidered only
after SourceItem identity and typed per-sample overrides have been used in real
batches.

Open questions include nested routes, branch-specific outputs, UI readability,
route predicates beyond exact SourceItem fields, and whether route decisions may
depend on scientific measurements rather than immutable batch context.

## Rule-Based Batch Overrides

After the explicit per-sample parameter table is stable, a rule builder might
populate values from exact metadata fields such as series name, well, channel,
or an explicitly normalized relative filename. It must not begin as arbitrary
Python, substring evaluation, or implicit directory inference.

A future design would need deterministic precedence, preview of every matched
item, duplicate/unmatched detection, saved normalization rules, privacy-safe
provenance, and a way to materialize the resolved table before execution. CSV
import/export is a simpler possible intermediate step.

## Branch-Aware Region Loading And Progressive Scientific Outputs

A future source planner could union exact regions required by several branches,
read only those chunks, and progressively refine scientific results. This goes
beyond the implemented narrow first steps of presentation pyramids and exact
sole direct-Crop pushdown.

Before adoption it would need:

- operation-level region and halo propagation;
- global-reduction and connected-component boundaries;
- deterministic results independent of chunk order;
- branch and cache invalidation after ROI changes;
- partial-result provenance and publication rules; and
- CPU/GPU transfer economics that do not turn many small chunks into a slower
  workflow.

Presentation may refine progressively; scientific results must not silently
change resolution or population while appearing complete.

## Workspace Density And Action Hierarchy

A future 0.14 UI pass may reduce visual chrome in the main toolbars, selected-
node inspector, and retained Batch workspace so common scientific controls and
results remain visible on ordinary laptop screens. This is exploratory cleanup,
not a committed layout or release gate.

Candidate directions include tighter spacing and grouping, a denser inspector
and Batch workspace, and moving infrequent demo/example actions out of the
primary Batch action row. The demo entry could become a subtler secondary
control, possibly near the top-right corner, without making onboarding examples
hard to discover. Any later design should preserve accessibility, keyboard
navigation, narrow-window behavior, and the clear separation between preview,
edit, and run actions, and should be validated with representative workflows
before implementation.

## AI-Assisted Authoring And Iterative Analysis

The [detailed future plan](ai-assisted-authoring-plan.md) records the 2026-09-10
request for a connected AI authoring system, not a release commitment:

1. Describe an analysis and assemble a reviewable workflow from existing nodes,
   with sensible connections, parameters, layout, and explanatory notes.
2. Describe a missing node, optionally supply code snippets, and declare
   inputs/outputs and parameters. A cloud model can draft the operation while
   VIPP owns the integration boilerplate, validation, and approval flow.
3. Inspect selected images/results, suggest changes, and compare candidate
   workflows, including parameter edits and adding/reconnecting nodes.
4. Progress to authorized, bounded self-iteration over workflow structure,
   layout, and parameters, with evaluation criteria, budgets, stopping rules,
   checkpoints, and a reproducible accepted result.

Reuse the operation and graph contracts, but do not mistake the current node
registry for a safe custom-code runtime. Generated code needs explicit trust,
isolated testing, versioned packaging, and scientific validation; cloud image
access needs informed opt-in. Prefer existing nodes before inventing new ones.
Saved accepted workflows must run without re-generating code or calling an AI
model again. These capabilities remain outside the committed 0.16 scope.

## Richer Tracking And Mitochondrial Analysis

User direction recorded 2026-09-30: the current basic tracking capability is
sufficient for now. Keep detection, one-to-one linking and read-only trajectory
review focused; do not turn the current release into a reimplementation of
TrackMate for Fiji or a comprehensive tracking package. This is a future,
unassigned direction, not authorization to implement the candidates below.

Longer term, the user would like richer, integrated tracking that can consider
object morphology alongside motion and support mitochondrial analysis:

- **Feature-assisted linking:** combine position with selected object
  measurements such as size, shape, elongation and intensity. Reuse the generic
  object, mesh and skeleton measurements rather than create a parallel
  mitochondrial measurement system. Feature choice, scaling, weights and
  rejection rules must be explicit and reproducible.
- **Motion-aware linking:** consider trajectory continuity and previous
  velocity, potentially with an optional prediction model, to improve difficult
  crossings and gap reconnections. Do not equate frame-local detection/label
  numbers with persistent identity or promise that prediction resolves every
  ambiguous crossing.
- **Richer trajectory analysis:** connect morphology changes over time with
  calibrated velocity, direction and motion-persistence measures, and shared
  table/plot review. The current linker already reports displacement, endpoint
  speed, path length and net displacement; extend those capabilities rather
  than duplicate them. Keep missing observations, units and the chosen
  coordinate/reference frame explicit.
- **Mitochondrial dynamics:** build organelle-focused workflows on those generic
  capabilities. Split/merge, lineage and mitochondrial fission/fusion analysis
  are possible later, separately designed extensions; a segmentation change or
  disappearance alone must not be reported as a biological event.

TrackMate and MitoMeter are user-mentioned reference tools for a future review,
not verified feature comparisons or parity targets. Investigate their primary
documentation, scientific methods, licensing and integration options before
choosing an implementation or claiming equivalent capabilities. Prefer small,
workflow-driven additions or suitable optional integrations over reproducing an
entire specialist application. Any promotion needs a concrete user workflow,
known-answer crossing/gap and morphology-change tests, and representative
acquired-microscopy evidence, following the [promotion checklist](#promotion-checklist).
No release version or delivery date is assigned.

## Longer-Horizon Product Concepts

These remain worthwhile but are not all release commitments. Translation,
image comparison (including SSIM), and template matching have moved into the
[planned 0.16 scope](registration-and-template-matching-plan.md). Drift remains
a stretch candidate; the broader concepts below are not core 0.16 gates.

- first-class points followed by puncta/spot detection and point measurements;
- rigid/affine alignment and separately qualified non-rigid registration,
  building on the transform foundation planned for 0.16;
- rotation/scale-aware template detection beyond the fixed-template 0.16 scope;
- first-class surfaces followed by mesh preview/export and specialist surface
  analysis;
- full plate/well/field browsing and broader HCS traversal after SourceItem;
- model-backed segmentation such as Cellpose, StarDist, or ilastik through
  isolated optional dependencies and exact model provenance;
- Apple acceleration after a time-boxed provider study, with CPU retained as
  the honest fallback until admission passes;
- stitching and mosaics;
- [richer tracking and mitochondrial analysis](#richer-tracking-and-mitochondrial-analysis)
  beyond the current basic tracking capability;
- [AI-assisted custom nodes, workflow authoring, and bounded iterative analysis](ai-assisted-authoring-plan.md)
  through the staged plan above, with explicit trust, cloud privacy, review,
  versioned replay, and execution boundaries.

## Promotion Checklist

Before moving an idea into `planning.md`, record:

1. the concrete user workflow and outcome;
2. why existing nodes, parameters, or batch tables are insufficient;
3. persisted schema, migration, hashing, and provenance behavior;
4. CPU/GPU, cancellation, cache, side-effect, and error semantics;
5. smallest coherent implementation slice and explicit non-goals; and
6. ordinary and protected acceptance evidence proportional to the claim.
