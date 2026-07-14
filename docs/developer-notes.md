# VIPP Developer and Architecture Notes

Last reviewed: 2026-07-13

This is the shortest technical entry point for contributors. It identifies the
owner of each behavior, the scientific contracts that must survive a change,
and the tests expected for common extensions. The detailed design is in the
[architecture reference](architecture.md).

## Start Here

- [Scientific behavior requirements](../CONTRIBUTING.md#scientific-behavior-requirements)
- [Core architecture](architecture.md)
- [Implementation planning and priorities](planning.md)
- [Node roadmap and migration choices](node-roadmap.md)
- [I/O behavior contract](io-user-guide.md)
- [OME architecture and constraints](ome-io-plan.md)

## Dependency Direction

```text
napari / npe2
    -> _widget.py       composition root and compatibility facade
        -> _graph.py    graph presentation and user-intent signals
        -> ui/          reusable Qt components, adapters, controllers
            -> core/    Qt-free scientific and persistence services
```

- `src/napari_vipp/core/` must not import napari, Qt, `_widget.py`, `_graph.py`,
  or `ui/`.
- `src/napari_vipp/ui/` may import Qt and `core/`, but must not import
  `_widget.py`.
- `_widget.py` may depend on all layers because `napari.yaml` constructs it. It
  should wire dependencies and coordinate the viewer, not become the default
  home for reusable algorithms or controls.
- `_graph.py` is presentational. It emits user intent; `VippWidget` currently
  applies graph mutations to the headless model.

`test_architecture.py` enforces the first two rules.

## Ownership Map

| Question | Start in |
| --- | --- |
| What does a node calculate? | `core/operations.py` |
| How is a node declared, connected, or executed? | `core/pipeline.py` |
| How are axes, calibration, kind, channels, and history propagated? | `core/metadata.py` |
| May several images or an image/PSF be combined on their physical grids? | `core/grid.py` |
| How are exact statistics, histograms, percentiles, contrast extrema, or label sizes calculated? | `core/diagnostics.py` |
| How is a local file/store revision identified and frozen? | `core/source_identity.py`, `core/file_sources.py` |
| How is a live napari layer revision frozen and invalidated? | `ui/source_adapter.py` |
| How does background graph execution cross the Qt boundary? | `core/execution.py`, `ui/workers.py` |
| How do diagnostic requests/results cross Qt workers? | `ui/diagnostic_workers.py` |
| How is graph/workflow state detached for history or execution? | `core/snapshots.py`, `core/workflow.py` |
| How are workflow and text artifacts replaced safely? | `core/atomic_io.py` |
| How are batch configs built and batches planned/run? | `core/batch_setup.py`, `core/batch.py` |
| How is collection batch UI coordinated? | `ui/batch.py`, `ui/batch_controller.py`, `ui/batch_navigator.py` |
| Where are reusable controls and dialogs? | `ui/controls.py`, `ui/axis_controls.py`, `ui/view_dims.py`, `ui/dialogs.py`, `ui/palette.py` |
| Where are plots and their Qt rendering? | `ui/plots.py` |
| Where are undo/redo and shutdown semantics? | `ui/history.py`, `ui/lifecycle.py` |
| Where is remaining application/viewer orchestration? | `_widget.py` |

## Non-Negotiable Scientific Contracts

1. **Inputs are stable revisions.** File/store sources are hashed before and
   after inspection/materialization, copied to owned read-only arrays, and
   pinned until `Refresh`. Live NumPy layers are copied and revision-tokened;
   stale results are discarded. A live source that cannot be detached is
   rejected.
2. **Physical coordinates and semantic confidence are data.** Explicit versus
   shape-inferred confidence determines whether an operation may make an
   automatic spatial/channel/projection choice. Carried axis name/type, scale,
   unit, origin, and sample count define multi-image grid compatibility. VIPP
   never inserts a hidden resampling, registration, projection, reorder, or
   RGB/Z interpretation to make inputs fit. Invalid declared axes, stale carried
   shapes, and corrupt calibration are errors, not invitations to infer
   replacements. A kernel that still processes trailing positions must pass the
   central canonical `YX`/`ZYX` layout gate; a kernel that supports arbitrary
   layouts must receive and test explicit semantic axis mappings.
3. **Diagnostic populations are explicit and exact.** Core diagnostics process
   every finite value in the declared slice, stack, channel, or spatial block.
   Chunking is a memory strategy, not sampling. Channel axes are explicit; a
   trailing dimension of length three or four is not sufficient evidence of
   RGB.
4. **Invalid parameters fail visibly.** Do not silently clamp, swap, normalize,
   skip, fill, or substitute scientific parameters. If a safe policy exists,
   expose and persist it; otherwise raise an actionable error.
5. **Operations preserve upstream buffers.** Kernels must accept read-only
   inputs and must not mutate cached arrays. An intentional view or pass-through
   alias needs an explicit test and contract.
6. **Presentation is detached.** Inspect/pin layers receive copies. Contrast,
   colormaps, thumbnails, and viewer edits are display state unless an explicit
   operation parameter says otherwise. Provisional contrast is replaced by an
   exact stale-safe result and never enters calculation.
7. **Persistence is validated and atomic per artifact.** Typed workflow
   snapshots are defensively copied and graph-validated. JSON rejects
   non-finite values and is replaced through a flushed, fsynced temporary file.
   Related files are not one multi-file transaction.
8. **Batch publication follows source verification.** Outputs are fully staged
   privately, every input source is reverified, and only then are outputs
   promoted atomically one by one. Source changes publish nothing for that
   item; later promotion failures are recorded as partial rather than hidden.
9. **Export uses the same executor.** Generated Python embeds a validated
   workflow snapshot and reconstructs `PrototypePipeline` per call. It carries
   source/output states and records the VIPP runtime version; do not replace
   this with hand-written direct operation calls that omit semantic injection.

See [Scientific Integrity Boundaries](architecture.md#scientific-integrity-boundaries)
for the exact owning modules and limitations.

## Contributor Recipes

### Add Or Change A Scientific Operation

1. State the input population, supported ranks/axes, dtype and unit behavior,
   parameter domain, output dtype/shape, edge cases, and reference algorithm.
2. Implement the numerical kernel in `core/operations.py`. Keep viewer and Qt
   concepts out of the function.
3. Add or update its `OperationSpec` in `core/pipeline.py`.
4. Add metadata propagation in `core/metadata.py` when axes, calibration, kind,
   channels, confidence, or history change. Do not promote shape-inferred axes
   to explicit unless an explicit user action or authoritative metadata did so.
5. For a multi-image node, add the correct physical-grid contract in
   `SAME_SHAPE_GRID_OPERATIONS`, `PSF_SAMPLING_GRID_OPERATIONS`, or an explicit
   operation-specific validator. Never solve mismatch by implicit resampling.
6. Test a numerical oracle or analytical phantom, invalid parameters,
   dtype/range and NaN/Inf behavior, read-only inputs, metadata, graph wiring,
   workflow round-trip, and export when affected.
7. Add UI controls only after the headless contract passes. Prefer an existing
   reusable control in `ui/`; keep `_widget.py` to composition and viewer glue.

### Add A Diagnostic

1. Put the population reduction in `core/diagnostics.py` and define what counts,
   how non-finite values are handled, and whether channel/spatial axes are
   required.
2. Keep it exact unless the user-visible contract explicitly declares an
   approximation. Use bounded chunks when temporary memory matters.
3. Test small reference populations, empty/all-nonfinite inputs, booleans,
   signed and wide integers, multichannel behavior, and chunk-boundary cases.
4. Put Qt drawing in `ui/plots.py` and reusable typed Qt runnables in
   `ui/diagnostic_workers.py`. Keep slice selection, request/cache keys,
   submission, and stale-result application at the UI/application boundary.

### Add Or Change A Source

- Reuse `core/io/` for format normalization and `ImageDataset` construction.
- Local files and directory stores must pass through exact identity capture and
  post-materialization verification. Preserve reader-built `ImageState` when
  making the owned snapshot.
- Qt scheduling belongs in `ui/file_sources.py`; the verification and ownership
  policy belongs in `core/file_sources.py`.
- Live viewer inputs belong in `ui/source_adapter.py`. Subscribe to public
  revision events, detach NumPy data/metadata, carry axis-aligned transforms,
  and reject unsupported transforms or data that cannot be frozen.
- Test a source changing during inspection/read, multi-series identity reuse,
  explicit Refresh, stale workers, and metadata preservation.

### Extract A UI Controller Or Component

1. Name one cohesive responsibility and its inputs/outputs before moving code.
2. Put reusable Qt widgets/adapters/controllers in `ui/`; do not import
   `_widget.py` from them.
3. Inject providers, callbacks, or action dataclasses from the composition root
   instead of reaching back into the widget. `CollectionBatchController` and
   `CollectionBatchActions` are examples of this direction; diagnostic workers
   similarly receive narrow calculation callables.
4. Keep scientific computation in `core/`, even when only one plot currently
   uses it.
5. Add focused component/controller tests and retain a small widget integration
   test for signal wiring. Preserve transitional imports only when a real caller
   still needs them.

### Change Workflow Or Batch Persistence

- Update schema validation before UI restoration. A malformed document must not
  partially replace the current graph.
- Use `GraphSnapshot`/`WorkflowSnapshot` for runtime ownership and
  `core.atomic_io` for file replacement.
- Breaking workflow changes are acceptable during alpha only when intentional:
  bump or explicitly reject the old schema, update golden tests and bundled
  examples, and avoid an implicit migration that changes scientific meaning.
- Batch configuration must retain the exact scientific workflow hash and
  deterministic source/output plan.
- Never publish a batch output before all source-dependent bytes are staged and
  every source identity has been reverified. Preserve collision policy and
  per-item/output provenance.

## Testing Ladder

Run the smallest relevant contract tests while developing, then the complete
suite before handoff.

```bash
python -m pytest -q src/napari_vipp/_tests/test_operation_input_contracts.py
python -m pytest -q src/napari_vipp/_tests/test_axis_semantics.py
python -m pytest -q src/napari_vipp/_tests/test_grid.py
python -m pytest -q \
  src/napari_vipp/_tests/test_diagnostics.py \
  src/napari_vipp/_tests/test_diagnostic_workers.py
python -m pytest -q \
  src/napari_vipp/_tests/test_source_identity.py \
  src/napari_vipp/_tests/test_file_sources.py \
  src/napari_vipp/_tests/test_source_adapter.py
python -m pytest -q \
  src/napari_vipp/_tests/test_snapshots.py \
  src/napari_vipp/_tests/test_execution.py \
  src/napari_vipp/_tests/test_workflow.py
python -m pytest -q \
  src/napari_vipp/_tests/test_batch_setup.py \
  src/napari_vipp/_tests/test_batch_controller.py \
  src/napari_vipp/_tests/test_batch.py
```

Required repository checks:

```bash
python -m npe2 validate src/napari_vipp/napari.yaml
python -m ruff check .
python -m pytest
python -m build
```

Use pytest-qt/widget tests only for behavior that crosses Qt, napari, event
ordering, or the final composition boundary. Documentation links are checked by
`test_documentation.py`; dependency direction is checked by
`test_architecture.py`.

## Remaining Seams

- `_widget.py` is still large. Inspector composition, graph commands, source
  caching/inspection, diagnostic cache/submission/result coordination,
  generated-layer presentation, and application commands remain there. Typed
  diagnostic worker adapters have moved to `ui/diagnostic_workers.py`.
- Synchronous execution calls `PrototypePipeline.run()` directly; background
  execution uses `core.execution`.
- Exact colocalization scatter-density computation still lives in
  `ui/plots.py`, not `core/diagnostics.py`.
- `core/operations.py` and `core/pipeline.py` remain large registries. A domain
  split needs registry, workflow, metadata, export, and numerical parity tests;
  file size alone is not a reason to move code.
- Atomicity is per artifact/output, not across a related set of files or every
  output in a batch item.

Operational recommendations for users are documented in
[operator-tips.md](operator-tips.md).
