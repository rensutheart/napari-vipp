# AI-Assisted Nodes, Workflows, And Iterative Analysis

Status: **future product direction**, not implemented or committed to 0.16.
Requested and reviewed: 2026-09-10. No release date, provider, or model selected.
See [product ideas](product-ideas.md#ai-assisted-authoring-and-iterative-analysis)
and [active planning](planning.md) for scope and promotion criteria.

## Intended User Experience

A biologist should eventually be able to describe an analysis, have VIPP build
an inspectable workflow, and improve it together with an AI assistant. If an
operation is missing, they should be able to describe a new node, optionally
provide example code, and declare its inputs, outputs, and parameters. VIPP
would generate the reusable node and handle the integration boilerplate.

The longer-term goal goes beyond chat suggestions: the assistant can inspect
selected intermediate/output images and measurements, change parameters,
add or reconnect nodes, arrange the graph, run another candidate, and compare
the results. This should develop in stages, with progressively broader but
explicitly bounded authority. These are proposed capabilities, not claims that
AI-generated code or visually appealing results are scientifically correct.

## Staged Delivery

### 1. Describe a workflow; assemble existing nodes

Start with VIPP's known node catalogue and validated graph fragments. A user
describes the biological aim, available channels/data, desired measurements,
and constraints. The assistant asks for missing scientific choices rather
than silently guessing axes, units, channel meaning, or the target objects.

Return a proposed workflow with named nodes, compatible connections, initial
parameters, readable layout, and explanatory notes. Show the graph changes
and why they were proposed before applying them. Prefer existing operations
over generating code for functionality VIPP already provides. Use normal
graph editing, validation, undo/redo, and source binding; do not bypass them
by having the model directly mutate arbitrary workflow JSON.

Apply a proposal as one validated, undoable transaction against the graph
revision it inspected. If the user has edited that graph meanwhile, revalidate
instead of overwriting newer work. Preserve named ports, tunnels, and manual
or bypass states; distinguish layout-only changes from scientific edits.

Initial execution is user-approved. Saving the accepted graph produces an
ordinary reproducible workflow that can run without calling the model again.

### 2. Describe a missing operation; create a reusable custom node

Offer an **AI-assisted node builder**, rather than requiring users to edit
VIPP source files or write every registration and UI hook. Its inputs include:

- a plain-language description and intended biological use;
- optional example code snippets, an existing function, or a method reference;
- named input/output ports and supported data kinds;
- accepted dimensions, semantic axes, calibration/units, and dtype behavior;
- parameters, defaults, valid ranges/steps, and help text;
- small example inputs and expected outputs or other acceptance criteria.

For example: "I have an image and a same-grid label image. Produce a table of
the median signal inside each object, retaining object IDs and intensity
units. Here is a function I used before." The builder should first check
whether an existing measurement node already meets that request, then help
define the missing behavior if it does not.

The preferred interaction can use a user-configured **cloud model** to draft
the scientific function, documentation, and candidate tests. VIPP itself owns
the versioned node specification, trusted templates, port/parameter controls,
registration adapters, metadata interfaces, packaging, and validation. Do not
rely on the model to invent or reproduce integration boilerplate correctly.
Do not patch VIPP's installed core files to add a user node.

Proposed lifecycle: **Describe -> Review contract -> Generate -> Test in
isolation -> Review results -> Approve and add to My nodes**. Show generated
code, assumptions, dependencies, warnings, and test outcomes in understandable
terms, with technical detail available. Test success means the tested contract
passed, not that the biological method is proven. Tests suggested by the same
model need independent expected values/reference checks where applicable.

The custom-node runtime/package interface is a prerequisite: today's
OperationSpec registry and UI parameter machinery are building blocks, not
an already-safe user extension system. Start with pure CPU operations and a
bounded set of existing input/output types. New types, arbitrary UI widgets,
device-specific code, and extra dependency installation need separate support
and qualification, not model-generated shortcuts.

Installed custom nodes should have a discoverable **My nodes** grouping,
namespaced IDs, versions, code/dependency fingerprints, editable documentation,
and explicit upgrade/remove actions. Pin the implementation used by a saved
workflow; never silently regenerate it when reopened. Missing or untrusted
nodes should be identified without executing their code. Reuse, packaging,
and sharing can follow once licensing, trust, and compatibility are reviewable.

### 3. Inspect results; suggest and test improvements

Let the assistant inspect user-selected source/intermediate/output views and
numeric diagnostics. For example, identify apparent merged objects, propose
a watershed parameter change, run a candidate, and show the before/after
masks with counts and the reason for the change. Extend this to proposing
different operations, additional nodes, connections, and clearer layouts.

Keep candidate workflows separate from the accepted baseline, with reversible
changes and explicit acceptance. The assistant acts through the same bounded
edit/run/inspect interfaces as ordinary UI actions. It must respect invalid
inputs, manual computation, cancellation, stale results, and resource limits.

Distinguish model-visible renderings from scientific data: record crop/slice,
projection, scale, downsampling, and display contrast. A prettier thumbnail
must not masquerade as improved segmentation or change the underlying data.
Use quantitative checks and representative examples alongside visual review;
ask the user when success is ambiguous.

### 4. Bounded self-directed iteration

Eventually the user can authorize an iterative session: **inspect -> propose
edits -> validate -> run -> compare -> retain or revert -> repeat**. Within
that permission, the assistant can control layout, add/reconnect approved
nodes, and tune declared parameters without asking about every minor change.

Before starting, agree on the goal, representative data, evaluation criteria,
allowed nodes/parameter ranges, iteration and time limits, compute/memory and
cloud-cost budgets, and stopping conditions. Provide pause/cancel, a visible
history, checkpoints, and a final comparison against the starting workflow.
New custom-code execution, dependency installation, broader data access, and
file publication remain separate permissions unless specifically authorized.

Do not optimize toward a desired biological conclusion or p-value. Evaluate
on held-out/reference data where appropriate; record exclusions, failures,
and per-image adaptations. Freeze the accepted workflow before production
batch execution, or explicitly record any authorized adaptive policy and its
resolved per-item choices. Batch reproduction must not silently become a new
model-guided optimization session.

## Cross-Cutting Requirements

- **Trust and execution:** user snippets and AI output are untrusted code.
  Validate the structured contract and test in a qualified execution boundary
  with resource limits and restricted filesystem/network access. A subprocess,
  syntax check, or model review alone is not a security sandbox. Until that
  boundary exists on a platform, offer proposal/export without automatic code
  execution. Enforced runtime limits must also cover stuck/native calls, not
  rely solely on cooperative cancellation. Opening a workflow never grants
  code-execution permission.
- **Cloud privacy:** before transmission, show exactly which text, code,
  metadata, images/crops, or results will leave the computer and which provider
  receives them. Do not upload raw datasets or private paths by default.
  Keep credentials outside workflow files and reproducibility packages.
  Local/offline providers can be evaluated later; ordinary saved-workflow
  execution must remain independent of cloud access.
- **Controlled actions:** reference documents, image text, and workflow notes
  are data, not permission to expand the assistant's scope. Models propose
  typed actions; VIPP validates and enforces user permissions locally. Unknown
  packages, file writes, and network requests are not implicitly authorized
  by a request to create or improve a node.
- **Scientific contracts:** preserve inputs, units, physical grids, axis
  semantics, object identities, and explicit approximation/normalization
  policies. Reject unsupported inputs rather than silently repairing them.
  Generated nodes need numerical, dtype/NaN, metadata, read-only-buffer,
  serialization, cancellation, and export tests appropriate to their claims.
- **Replay and provenance:** retain the accepted graph, exact custom-node
  implementation/dependencies, effective parameters, data references,
  evaluated candidates, decisions, and relevant model/provider identifiers
  and generation settings. Keep a privacy-reviewed request/action record;
  reproducibility must not depend on re-prompting a model. Respect existing
  exclusions of raw images/results from shareable packages by default, and
  make any custom-code inclusion explicit and inert until trusted.

## Smallest Coherent Starting Point

The first implementation should assemble existing nodes from descriptions,
show a validated graph diff, and require approval to apply/run it. In parallel,
design the reusable custom-node specification and safe execution boundary.
AI-assisted custom-node creation follows that foundation; visual suggestions
and then bounded self-iteration follow reliable edit/run/inspect interfaces.

Before promoting a stage into a release, demonstrate wrong-axis/type handling,
unsafe generated code rejection, cloud opt-in/cancellation, interrupted runs,
undo/reopen, missing-node/version handling, and headless/batch replay of accepted
workflows. Review the builder and iteration experience with nontechnical users.
No provider-specific integration or autonomous research claim is committed by
this plan.
