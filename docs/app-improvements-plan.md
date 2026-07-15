# App Improvements Implementation Plan

## Purpose

This plan turns five requested improvements into independently assignable work
packages. It is based on the current `src/napari_vipp` architecture and keeps
scientific behavior, workflow compatibility, and CPU-only installations safe.

## Current architecture and important findings

- Operation and parameter declarations live in `core/pipeline.py`.
  `ParameterSpec` already carries labels and tooltips, while `OperationSpec`
  already carries named `InputSpec` and `OutputSpec` ports.
- Inspector parameter widgets are rebuilt in `_widget.py` and already support
  operation-specific rendering. The pipeline exposes cached upstream arrays via
  `input_data_by_port_for_node()` and metadata via `input_state_for_node()`.
- Graph ports in `_graph.py` already store labels and expose them in tooltips,
  but no persistent text items are drawn beside ports. The generic graph helper
  currently replaces most declared input names with `Input 1`, `Input 2`, etc.;
  this is why the RL-TV node's declared `Image` and `PSF` names are not visible
  on the canvas.
- The toolbar is one long `QHBoxLayout` in `_widget.py`. Its compact behavior
  moves controls into the Settings menu as width decreases.
- Execution is headless and NumPy/SciPy/scikit-image based. Background and batch
  execution serialize and reconstruct the same pipeline, so backend selection
  must be part of the execution request/workflow contract rather than UI-only
  state.
- RL-TV is a local implementation in `core/operations.py`. It starts the
  estimate at `0.5`, uses zero-extension through `scipy.signal.convolve(...,
  mode="same")`, applies the TV term through `1 - lambda * div(...)`, and floors
  that denominator at `0.05`. The PSF is optionally normalized, but the current
  operation does not pad the image, validate physical sampling compatibility,
  or resample the PSF. These are plausible sources of edge artifacts or
  structure loss and must be tested separately from the default lambda.

## Recommended sequence

1. Implement the shared input-aware parameter visibility mechanism.
2. Implement port labels and toolbar layout in parallel after agreeing on the
   small global-settings contract.
3. Run the RL-TV validation work before changing scientific defaults.
4. Build the GPU capability/benchmark spike, then implement only the backends
   that pass the benchmark and parity gates.
5. Finish with integrated UI, workflow, batch, documentation, and packaging QA.

The UI packages are suitable for separate agents. The RL-TV and GPU packages
should not be combined: one is scientific validation and the other is execution
infrastructure.

## Work package A: input-aware parameter visibility

### Goal

Show parameters only when they are meaningful for the selected node's current
primary input, without deleting or changing their persisted values.

### Implementation

1. Extend `ParameterSpec` in `core/pipeline.py` with a declarative visibility
   key (for example `visibility="floating_input"` or
   `visibility="rgb_or_rgba_input"`). Keep the default unconditional so existing
   operations and saved workflows are unchanged.
2. Add a Qt-free resolver, preferably in `core/pipeline.py`, which accepts the
   parameter spec, connected input array/state, and operation context and
   returns visible/hidden plus an optional reason. Do not embed dtype and axis
   rules directly in widget construction.
3. In `_widget.py`, resolve the selected node's port-zero input from cached
   pipeline data. Re-evaluate visibility after selection, connection changes,
   source replacement, completed execution, workflow restore, and operations
   that change dtype or axis semantics. If the upstream input is not yet known,
   show the control rather than guessing and optionally explain that it will be
   refined after input resolution.
4. Mark `HISTOGRAM_BINS_PARAMETER` as floating-input-only. Apply it to all nodes
   that use it: Otsu, Triangle, Yen, Isodata, and Minimum. Keep the parameter in
   node params and generated Python even while hidden; integer execution already
   has native-level behavior and must not be altered by this UI change.
5. Mark the RGB/RGBA-to-luma channel-axis control as relevant only when explicit
   metadata identifies an RGB/RGBA axis, or when the user must resolve genuinely
   ambiguous color input. Do not infer RGB from a dimension of length 3 or 4;
   that would violate the repository's explicit-axis policy. Add this tooltip:
   "Select the axis containing encoded RGB or RGBA channels so the operation can
   reduce color to luminance. Leave at -1 for scalar images. Change this only
   when the input metadata does not already identify the color axis."
6. Audit other conditional candidates, but limit the first patch to defensible
   rules. Likely follow-ups are spatial-mode controls for strictly 2D inputs and
   channel selectors for known scalar inputs. Record candidates rather than
   introducing speculative hiding.

### Tests

- `test_widget.py`: bins visible for `float32` and hidden for `uint8`, `uint16`,
  signed integers, and bool; visibility updates when the source or upstream
  dtype changes; hidden values survive save/load and generated-code export.
- `test_widget.py` / `test_axis_semantics.py`: RGB control visibility follows
  explicit `rgb`/`rgba` metadata, remains scalar for ordinary `ZYX`, and handles
  unresolved inputs conservatively.
- `test_pipeline_restore_safety.py`: old workflow documents restore without a
  schema migration.

### Acceptance criteria

- No irrelevant bins slider is shown for a resolved integer input.
- No parameter value is reset merely because its row is hidden.
- UI visibility never changes numerical execution or workflow compatibility.

## Work package B: optional graph port labels

### Goal

Add three global display modes: `Hide all`, `Ambiguous only`, and `Show all`,
with `Ambiguous only` as the default.

### Implementation

1. Define a `PortLabelMode` enum/string contract in `_graph.py` (or a small UI
   settings module) and add a `PipelineGraphView.set_port_label_mode()` method.
2. Fix label sourcing first. Resolve static names from
   `OperationSpec.input_ports`/`output_ports` and dynamic names from pipeline
   port factories. RL-TV must resolve to `Image` and `PSF`, not `Input 1` and
   `Input 2`. Preserve the specialized channel-color labels.
3. Add child `QGraphicsTextItem` labels to each `PortItem` or `NodeProxy`.
   Position input labels just inside the left edge and output labels just inside
   the right edge. Use horizontal text, elide very long labels with the full
   name in a tooltip, and reserve vertical rows per port. Rotated text is not
   recommended for the first implementation because it harms scanning.
4. Mode behavior:
   - `Hide all`: no persistent labels.
   - `Ambiguous only`: label all ports when a node has more than one input or
     more than one output; otherwise hide them.
   - `Show all`: label every existing port.
5. Recompute the card/proxy minimum height from port count and reserve adaptive
   left/right card gutters from label metrics so labels never cover node UI.
   On a mode toggle, update geometry, call `prepareGeometryChange()` as needed,
   refresh ports and wires, and expand the scene bounds.
6. Do not silently move manually arranged nodes on a toggle. Detect overlaps
   and offer/use the existing `Auto structure graph` action for reflow. Ensure
   the layered layout consumes the new proxy sizes so an explicit re-layout is
   collision-free.
7. Add the selector to the Settings menu in `_widget.py`. Treat it as an app
   display preference (default per installation/session), not scientific
   workflow content. If persistence is added, use Qt application settings; do
   not bump workflow schema version 3 for a canvas-only preference.

### Tests

- `test_graph.py`: all three modes, exact RL-TV labels, dynamic multi-output
  labels, elision/tooltips, wire attachment after geometry changes, and no node
  position changes on toggle.
- `test_graph_layout.py`: auto-layout does not overlap expanded multi-port
  nodes.
- `test_widget.py`: Settings menu selects and restores the mode; default is
  `Ambiguous only`.

### Acceptance criteria

- RL-TV visibly distinguishes Image from PSF in the default mode.
- Labels remain readable at supported graph zoom levels and do not cover ports.
- Toggling labels does not corrupt connections or saved node positions.

## Work package C: main toolbar layout

### Goal

Make every label-control relationship visually explicit while preserving the
existing responsive compact menu.

### Implementation

1. Replace the flat toolbar `QHBoxLayout` in `_widget.py::_build_layout()` with
   small field-pair widgets. Each pair should use a two-column `QGridLayout` or
   compact `QFormLayout`: right-aligned label, immediately adjacent control,
   consistent 4-6 px internal spacing.
2. Keep `Preview` as the leftmost field/section anchor. Group Preview, Contrast,
   Contrast Range, and Mono together; group Zoom separately; group graph actions
   separately. Add stretch only after logical groups, never between a label and
   its control.
3. Preserve `_apply_toolbar_compact_stage()` behavior by registering entire
   field pairs instead of independent labels and combos. When hidden, the same
   controls must still appear in Settings exactly as they do now.
4. Give the four combos stable minimum/content widths so translations or longer
   values do not make adjacent labels appear attached to the wrong control.

### Tests

- Add a Qt layout test at wide, medium, and narrow widths. Assert each label's
  right edge is nearer its own control than any neighboring control and that
  `Preview` remains the leftmost visible label.
- Retain/extend existing compact-toolbar Settings menu tests.
- Capture one documentation screenshot at normal DPI and inspect at high-DPI
  scale to catch clipping.

### Acceptance criteria

- Field labels are right-aligned within their groups with small, consistent
  label-to-control gaps.
- The responsive toolbar and Settings fallbacks remain usable at narrow widths.

## Work package D: RL-TV scientific validation and safer behavior

### Goal

Determine the cause of reported structure loss with Born-Wolf PSFs before
changing defaults, then make the node safer and easier to tune.

### Phase 1: reproducible diagnosis

1. Add deterministic 2D and 3D phantoms containing bright points, thin lines,
   dim lines near bright structures, and objects near borders. Blur them with a
   known normalized PSF and add controlled Poisson/Gaussian noise.
2. Define metrics: reconstruction MSE/PSNR, SSIM where appropriate, recovered
   line/point intensity, false loss of dim structures, total variation, flux
   ratio, and border-region error. Save metric tables, not only screenshots.
3. Sweep iterations, TV lambda, TV epsilon, denominator floor, and filter
   epsilon. Compare RL-TV at lambda zero with ordinary RL; they should agree
   within a documented tolerance. This isolates implementation differences.
4. Test PSF sum, non-negativity, odd shape, centering, spatial rank, and physical
   sampling. Born-Wolf outputs carry calibrated axes; compare PSF Y/X/Z spacing
   to the image state and fail or warn on mismatch rather than silently treating
   samples as equivalent.
5. Compare current zero-extension convolution with reflect-padding and explicit
   crop-back. Measure whether border artifacts propagate inward. Also compare
   initialization (`0.5`, observed image, and positive mean) because a constant
   unscaled start can dominate early iterations.
6. Cross-check a small fixture against a documented reference implementation or
   paper equation. Confirm the sign and discretization of the TV divergence and
   the denominator convention before tuning lambda.

### Phase 2: product changes after evidence

- If padding is the cause, add an `Edge handling` choice (`Reflect` recommended,
  `Current/zero` for reproducibility) and document crop-back behavior.
- If sampling mismatch is material, add validation in `Prepare / Validate PSF`
  and at the deconvolution boundary. A later explicit PSF resampling feature can
  be separate; do not silently resample in the initial fix.
- If initialization is the cause, change it only with regression fixtures and a
  changelog entry because numerical output will change.
- Change the `0.002` TV default only if the phantom sweep and at least one real
  microscopy dataset show a broadly safer value. Otherwise retain it and make
  the warning more prominent.
- Add presets as named parameter bundles, not separate algorithms: `Light`,
  `Moderate`, `Strong`, plus `Custom`. Selecting a preset writes explicit
  parameters into the node so workflows remain reproducible. Preset values must
  come from the validation sweep.
- Keep the existing detailed tooltips and add an inspector warning near TV
  regularization: excessive values can erase dim/fine structures. Add PSF
  spacing/normalization status to the inspector.

### Tests and acceptance criteria

- Unit tests for PSF validation, padding/crop shape, lambda-zero parity,
  cancellation, finite/non-negative output, and scale preservation.
- Regression thresholds for thin/dim structure recovery and border error.
- Defaults are not changed without quantitative evidence attached to the PR.
- The documentation clearly distinguishes iteration artifacts, TV suppression,
  PSF mismatch, and edge handling.

## Work package E: optional GPU execution

### Goal

Provide meaningful acceleration where measured, with transparent capability
reporting and safe CPU fallback.

### Phase 1: capability and benchmark spike

1. Introduce a Qt-free compute-backend contract (`Auto`, `CPU`, `GPU`) and a
   capability report containing availability, provider/version, device name,
   supported operation IDs, and unavailability reason. Keep it in a new module
   such as `core/compute.py`.
2. Evaluate CuPy/CuPyX first because current kernels are NumPy/SciPy shaped;
   evaluate cuCIM for scikit-image-like filters where it provides compatible
   operations. Do not depend on PyTorch only for array math unless benchmarks
   justify the extra dependency.
3. Add GPU dependencies as optional extras, never base dependencies. Detect them
   lazily so CPU-only import, plugin discovery, and tests remain clean.
4. Benchmark representative 2D and 3D sizes, including host-to-device and
   device-to-host transfer. Priority candidates: ordinary RL, RL-TV, Gaussian
   and median filters, rolling-ball/background subtraction, and batch reuse.
   Record peak GPU memory, speedup, and numerical difference. An RTX 5090-class
   device should be tested if available, but support must be capability-based,
   not GPU-model-specific.
5. Promotion gate: implement a GPU path only where end-to-end speedup is
   meaningful (suggested >=1.5x on a representative workload), memory behavior
   is bounded, cancellation/progress remain functional, and numerical parity is
   within operation-specific tolerance.

### Phase 2: execution integration

1. Add backend selection to `PipelineRunRequest`, batch requests/config, and
   generated Python. `Auto` may choose GPU only for supported nodes; `CPU` must
   be deterministic and preserve current behavior; explicit `GPU` should either
   fail clearly before execution or fall back only when the user has enabled a
   documented fallback policy.
2. Add per-operation capability metadata to `OperationSpec` rather than a
   scattered title/ID list. Display support in the palette/inspector and explain
   why a selected node will run on CPU.
3. Keep arrays on the device across consecutive supported nodes within one run.
   Transfer at source/unsupported-node/output boundaries. A per-function
   `cp.asarray`/`cp.asnumpy` wrapper would erase much of the gain and should not
   be the final design.
4. Include backend and device identity in cache keys/provenance so CPU results
   are not confused with GPU results. Ensure memory guards account for GPU
   memory separately from RAM.
5. Map progress/cancellation into iterative GPU implementations. Catch GPU OOM
   with a clear node/device message; if fallback is enabled, release device
   allocations before retrying on CPU.

### Tests and acceptance criteria

- CPU-only CI: missing optional packages, `Auto` fallback, explicit-GPU error,
  serialization, generated code, and batch behavior.
- GPU CI/manual matrix: supported CUDA/provider versions, CPU/GPU parity per
  operation, cancellation, OOM handling, device residency across a chain, and
  memory cleanup between batch items.
- No GPU dependency is required to import or use VIPP on CPU.
- UI never claims acceleration for an unsupported node.
- Published benchmark results identify workloads where GPU mode helps and where
  transfer overhead makes CPU preferable.

## Agent assignments and dependency boundaries

### Agent 1: conditional controls

Own work package A. Touch `core/pipeline.py`, `_widget.py`, focused widget/axis
tests, and user-guide parameter documentation. Avoid graph label and toolbar
layout edits to minimize merge conflicts.

### Agent 2: graph port labels

Own work package B. Touch `_graph.py`, graph/layout tests, and the Settings menu
hook in `_widget.py`. Coordinate the small Settings menu edit with Agent 3.

### Agent 3: toolbar layout

Own work package C. Touch the toolbar construction/responsive helpers and their
widget tests. Do not change inspector parameter rendering.

### Agent 4: RL-TV validation

Own work package D Phase 1 first. Deliver fixtures, metrics, findings, and a
recommendation PR before a numerical behavior PR. Scientific default changes
must be reviewed separately from UI polish.

### Agent 5: GPU spike and infrastructure

Own work package E Phase 1. Deliver the backend interface, capability detection,
benchmark harness/results, packaging recommendation, and a ranked node list.
Implementation of promoted node backends can then be divided by operation
family.

### Integration agent

After the above branches land, resolve the two Settings menu additions, run the
full suite, test workflow v3 round trips, execute synthetic examples in CPU
mode, run narrow/high-DPI UI checks, and update `docs/user-guide.md`,
`docs/operator-tips.md`, `docs/architecture.md`, `docs/cache-and-memory.md`, and
the changelog.

## Suggested PR boundaries

1. Declarative parameter visibility plus bins behavior.
2. RGB/RGBA relevance rules and tooltip after the shared resolver lands.
3. Port-name sourcing and label modes.
4. Toolbar field-pair layout.
5. RL-TV diagnostic fixtures and report (no changed defaults).
6. RL-TV behavior/default changes justified by the report.
7. GPU capability API, optional packaging, and benchmarks.
8. One PR per promoted GPU operation family, followed by batch/device-residency
   integration.

This split keeps reviews small and prevents scientific-output changes from being
hidden inside presentation or infrastructure work.
