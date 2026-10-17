# AI-Assisted Nodes, Workflows, And Iterative Analysis

Status: **registry-wide planning, run-bound assessment, finite refinement,
shared investigation tools/loop, resumable allowances, live local MCP,
graph recovery, reviewed same-topology edits and local session archives,
unreleased** (2026-10-17).
Broader custom-node generation, review-frequency controls and automatic
best-candidate selection remain future product direction, not implemented or
committed to a release. See the [prototype contract](workflow-assistant-prototype.md) and
[issue 80](https://github.com/rensutheart/napari-vipp/issues/80).
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

### Next implementation focus

The next milestone is a first-class **improve this existing workflow** path:
select the branch/results and what must be preserved, describe one defect, test
small alternatives, inspect matched visual/measurement evidence, then review a
localized edit in the native window without helper scripts or hand-selected
processing choices. Existing same-topology patching is the implemented starting
point, not general graph-edit authority.

Extend that path in this order:

1. Bounded insert/remove/bypass/reconnect patches using existing nodes, with
   exact source/graph ownership, unaffected-branch preservation, a readable diff,
   checkpoint, one Undo and a separate explicit Run.
2. User-marked problematic/protected regions and understandable acceptance
   checks tied to the actual comparison views and measurements. Retain explicit
   uncertainty about inferred reconstruction and scientific object boundaries.
3. Blind, multi-task evaluation of experiment choice, regression recognition,
   best-result retention, stopping and human intervention. Compare providers
   under the same tools and allowances; infrastructure passes are not evidence
   of model quality or generalization from the circular/wavy demonstration.

These are next implementation/acceptance priorities, not shipped capabilities
or a release commitment. The [active roadmap](planning.md#next-ai-assistant-milestones)
owns their product order. Custom-node generation remains a later, separately
reviewed capability rather than the next dependency.

### Current implementation

The current shared-agent slice uses the same continuing core loop for a new
Goal and accepted-result refinement. Bounded native-region probes let the model
compare its own methods/parameters before a full candidate trial. The optional
local MCP/stdio host reuses that tool/execution/evidence layer. It now supports
an explicitly enabled connection to the current GUI workflow as well as the
separate human-configured isolated NPY source mode. Codex sign-in, selected
image-capable local models and explicit OpenAI API access share the embedded
investigation harness. Questions and selected allowance boundaries can pause
and resume the same in-memory session, without discarding evidence or resetting
usage. Local `.vipp-session` files now provide passive history/evidence reopening
and explicitly re-authorized paused continuation when exact source, graph,
compute, operation contracts and supported evidence still match. Ordinary tab
recovery is separately graph-only and never restarts an agent or calculation.
A completed tested alternative can preview and apply a bounded same-topology
parameter/compatible-operation patch to its live-owned baseline, with a required
recovery checkpoint and one Undo step. Wider graph edits remain separate Add;
Run is always a separate owner action. Neither archive history nor MCP grants
unrestricted in-place mutation or automatic cross-version continuation.
These are procedural capabilities, not identical model reasoning, scientific
accuracy, unrestricted computer access or automatic best-result selection.

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

The approved native testing workspace separates **Goal → Discussion → Plans**.
Navigation does not send, add or run. Discussion retains the original objective,
answers and revisions; Plans retains text-only history, with earlier drafts
read-only. A visible sharing disclosure accompanies explicit Send/revision
actions, which approve that request without a repeated permission checkbox;
input-image previews remain a separate default-off choice. Provider/model
changes are staged in Settings until Apply. Adding the latest eligible draft
is one guarded undoable edit; a separate Run action invokes the ordinary guarded
Calculate-all path. Output assessment requires its own default-off choice and
an explicit assessment request after a completed run; it does not itself grant
execution. The separately started bounded continuation below grants only its
declared finite assess/revise/Add/Run authority.

Planning attempts now default to and are capped at 600 seconds for both
providers; at most two attempts can use up to 20 minutes of model-request time.
A live local elapsed clock makes the wait visible without claiming remote
model liveness, token production or reasoning progress. Cancellation remains
available; this changes neither execution authority nor output sharing.

#### Current continuation: library discovery and broader composition

The continuation now uses registry-wide authoring contracts instead of the
earlier 26-operation adapter. Catalogue version 4 lists all 146 operations and
permits 143 existing operations. The three concrete exclusions are Image Source
and Table Source, which need explicit user-selected bindings, and Batch Output,
which needs the separate collection/publication setup. The assistant must
distinguish an authoring boundary from an operation missing from VIPP. Ordinary
node definitions, safe current optional controls and metadata validation remain
authoritative; authorability is not evidence that a method meets a biological aim.

The adapter supports branching/convergence, fixed and metadata-dependent outputs,
optional/mode-dependent inputs, up to 64 new nodes/128 connections and a 256-port
cap checked before dynamic port expansion. A cached finite owned source remains
limited to 64 MiB, with explicit Y/X spatial axes and any actual Z, time or channel
axes. Numeric images, Boolean masks and integer labels use the normal node-specific
metadata contracts, not shape-based dimension guesses. Scientific preconditions
may still reject a proposed connection. Image previews remain scalar YX/ZYX only.
Freshness, undo, manual execution and compute/resource guards remain.

Save Image is conditionally authorable with a user-selected local folder. The
exact answer **use the same folder as input** can resolve one verified physical
file folder; sample/live inputs or files from different folders require an
explicit picker choice. Only USER_OUTPUT_DIRECTORY and approval state enter the
model request, not the actual path. Generated filenames and formats are bound
locally after validation; no existing file may be overwritten. Send, navigation
and Add do not save. An explicit Run invokes the ordinary guarded executor;
the model gains no filesystem or execution tool. Batch publication, arbitrary
new source binding, external programs and custom code remain outside this slice.
The serialized public metadata request has a separate 192 KiB budget for the
full catalogue plus complete allowed dialogue. Shared parameter/port tables and
verbatim text pools losslessly encode every catalogue entry and its help rather
than selecting a subset or shortening answers. Proposal responses remain
bounded to 128 KiB. Existing history limits and provider transport caps are
unchanged. Oversized serialized requests fail before sending, not by silently
truncating answers.

The two-structure demonstration is the first topology acceptance target:
24 new nodes and 28 connections using preprocessing, independent mask branches,
mask logic/cleanup, mesh combination and measurement. Its sanitized fixture
must leave acquisition-border meshes open rather than reproducing the original
closed-border choice, and uses non-identifying synthetic inputs. Passing graph
construction, persistence and replay does not establish equivalent geometry,
correct separation, biological accuracy or success on the original images.
The registry adapter alone does not assess results; explicit run-bound
assessment and finite refinement are implemented separately below; no model
request is needed to qualify the registry adapter.

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

#### Bounded investigation before a final draft

The separately selected **Investigate with trial runs** option now supports a
model-directed investigation over one loaded explicit YX/ZYX source. Unlike
post-run refinement, trials execute detached candidate graphs without adding
them to the user's workflow. The model can query registered node contracts,
request selected source/intermediate views and exact diagnostics, inspect its
tested recipes, compare explicit output roles and retain a preferred candidate.
Brief observations are evidence-linked model interpretations, not measured facts.
Declared reconstruction hypotheses are reviewable assumptions, never proof
that inferred material was observed. The current continuation requires a
prospective experiment plan for each UI trial and an evidence-owned assessment
before the next trial or finish. Plans declare predictions, risks and numerical/
visual checks; unresolved checks stay explicit. Compact safe-node discovery and
batched exact contracts reduce lookup overhead. Optional calibrated native ROI
statistics/comparisons test a selected region without changing scientific data.

Start requires the existing output-assessment choice for output measurements
and previews, with independent input-image opt-in. It grants only the displayed
trial/action/model-call/time bounds; it does not enable general model tools,
writers, custom code, new source access or publication. With the unreleased
**Review before finishing** choice enabled (the native ChatGPT default), a
completed model selection first pauses for user review. **Keep improving** can
use optional feedback and the same remaining bounded allowance to investigate
known addressable issues under the original goal. It does not reset the session,
notebook, protected baseline or counters. Human pause/review time is excluded
from the shared native continuation's active-time allowance; used work is not
refunded. The model need not label a partial result successful or
remove honest limitations in order to stop.

The initial review-enabled phase reserves one of the selected trial attempts
when the total is at least two, plus usable action/call room for a continuation.
That capacity is released only by a successful **Keep improving** in the same
session; it is not an extra trial or renewed allowance. Initial closing must
leave room for assessed results and the model's explicit nomination, without
auto-selecting a candidate. A one-trial setting has no further trial to reserve,
and time, context, sharing and resource limits can still prevent continuation.

**Good enough** accepts only the exact completed, still-current nomination as a
normal checked draft; Add and Run remain separate. This can accept completed
work after the work deadline without permitting a new request or calculation.
Selecting the accepted starting result keeps it without adding a duplicate.
**Stop** preserves the accepted graph. The opt-in core flag defaults false for
existing callers, and the local connector's earlier refinement series is
unchanged. Exact source/calibration, stale/cancellation/cleanup, image-budget and replay checks
remain necessary. See [the acceptance protocol](workflow-investigation-acceptance.md).

The 2026-10-16 blinded test used the original circular/wavy prompt and both
preview permissions, without a human recipe or reference. The product completed
28 actions and three CPU trials and explicitly selected a partial result.
After fixing an acceptance-driver consumed-ID bug, a separate zero-inference
continuation verified the unchanged selected recipe through normal Add/Run:
both masks, three mesh roles, full parameters/wiring and metadata matched.
Original failed acceptance records were preserved, not relabelled. This proves
the bounded interaction/execution/replay path, not complete reconstruction:
the central wavy gap, open/nonmanifold mesh geometry and uncertain overlap
remained. The next scientific-planning gap is targeted hypothesis testing rather
than additional cosmetic revisions. Dated local evidence is in
`VIPP-local-tests/ai-assistant-investigation-loop-20261016/REPORT.txt`.

A separate reconstruction-planning continuation on the same date exercised the
new prospective experiment ledger, native-region diagnostics and compact node
discovery. A fresh original-input run tested four candidates and the model
explicitly nominated candidate4; all seven masks/labels/mesh roles then replayed
exactly through ordinary Add/Run under the original qualified execution package.
The full history reached the unchanged request-size bound before nomination.
Lossless reference encoding was fixed and mechanically round-trip checked;
one FINISH/question-only recovery turn used the original session, complete own
evidence and original absolute deadline. This is not an uninterrupted clean
product-run claim, and that recovery harness is not an automatic GUI recovery
feature. Original interrupted receipts remain unchanged. Independent post-lock
scientific review and detailed evidence are recorded separately in
`VIPP-local-tests/ai-assistant-reconstruction-investigation-20261016/REPORT.txt`.

Numerical convergence, calibrated scientific confidence, larger datasets,
local image-capable investigation and unrestricted custom-node execution are
not established by this increment.

The first result-feedback increment is implemented alongside planning. A checked
proposal or completed Run must not be presented as meeting the goal: an explicit
run-bound assessment can now interpret calculated outputs, but its verdict is
not proof. It uses existing catalogue operations; the custom-node builder in
stage 2 is not a prerequisite for this bounded path.

#### First increment: explicit run-bound result feedback

Goal offers prominent independent **Include input image previews** and **Assess
workflow outputs** choices, both off by default. After an explicit ordinary Run
completes, **Preview results shared** discloses actual source/output diagnostics,
candidate topology and permitted bounded images. **Assess results** then sends
one run-bound request. The report explains goal checks, failures, recommended
changes and limitations. Source/graph/cache changes invalidate old evidence;
diagnostics use full supported arrays, not display sampling. Mask connectivity,
label IDs and mesh display groups remain distinct. The local connector receives
measurements only; Codex output images do not grant input-image permission.

**Revise using results** separately sends measured feedback, the report and
preview manifests with a fresh proposal context. Codex revision resends the
same previously approved bounded output PNGs under the current image consents
and shared budget; local revision remains measurements-only. The resulting
draft still requires explicit Add and Run in Manual mode; a separately started
finite series can delegate these ordinary guarded steps. Model confidence does
not establish correctness. See the
[implemented contract](workflow-assistant-prototype.md#completed-run-output-assessment).
The existing [result card](#assessment-after-every-iteration) and
[bounded iteration policy](#4-bounded-self-directed-iteration) below describe
the broader future policy, including controls beyond the finite continuation.

#### Current increment: comparable evidence and finite rounds

Actual captured meshes now contribute two fixed calibrated oblique renders,
with unchanged triangles and disclosed display approximations. Each assessment
can receive the last completed comparable run's bounded diagnostics and
accepted report, not its images or recursive history. Source SHA/grid, original
dialogue and image choices bind that comparison; relative improvement remains
separate from accuracy or overall goal achievement.

The default-zero **Refine using the results** slider permits at most five
additional calculation attempts. Separate Start in Plans after the first ordinary run
delegates checked assess/revise/Add/Run for at most 30 minutes and
`2 * selected limit + 1` logical requests with existing bounded format repairs. A
positive whole-graph inventory refuses writers, other sources, external/custom
or unreviewed operations. Earlier completed branches remain intact. Questions, non-repairable errors,
stale/changed work, unavailable evidence, cancellation, repeated drafts, limits
and explicitly labelled AI goal-achievement judgments stop the series. Up to
six recent reports remain visible. Review-frequency controls, numerical
convergence criteria and best-candidate selection below remain future work.
See the [finite controller contract](workflow-assistant-prototype.md#separately-started-finite-refinement).

The failure-aware continuation counts failed Run dispatches against the selected
attempt limit, separately from completed/assessed results. At most two positively
identified, owned mesh-topology repair transitions use the same request/time/run
budgets. Only the exact failed own-Add is rolled back before fresh proposal
capture; prior completed branches and failed-attempt history are retained.
Unknown/resource/cancel/stale/security/provider/I/O or cleanup failures stop for
manual review. A failed run is not assessed or represented by partial outputs:
repair shares safe error/operation/recipe context and separately allowed input
previews, never failed-output PNGs or a fabricated report. Last completed
assessment remains historical, not current authority or a best-candidate pick;
the next completed result compares with the last completed result. No scientific
geometry patching or weakened mesh admission is introduced. Goal checks now
request explicit coverage, unwanted-signal, continuity, separation and mesh
quality trade-offs; prettier or less fragmented is not necessarily more correct.

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

#### Assessment after every iteration

Each calculated candidate needs a plain-language result card, retained in the
session history even when the attempt failed:

- **What it tried:** the intended improvement and the changed nodes/parameters.
- **How it did:** goal-by-goal **achieved**, **partly achieved**, **not achieved**
  or **cannot assess**, supported by observed output views and relevant measured
  checks, with the previous/best candidate available for comparison.
- **What remains wrong or uncertain:** missed signal, merged structures,
  fragments, incomplete volume coverage or an unavailable operation, where
  supported by the evidence. A run error is a failed attempt, not a usable
  segmentation. Missing evidence must not become an invented observation.
- **What happens next:** the proposed change and its reason, a request for
  feedback when needed, or the reason for stopping.

The user's wavy structure and central bright rounded structure are a concrete
acceptance case: check whether the masks correspond to those two independently
identified structures across the relevant 3D volume, and whether the meshes
retain their separate identities and colours. Merely producing two labels,
colouring a combined mask differently or showing a plausible projection does
not establish success. Record split/merge errors, missing regions and view
coverage. Completeness remains uncertain without sufficient evidence or an
independent reference; a model's confidence is not proof of biological accuracy.

The smallest testable increment is **run one approved candidate → assess its
actual output → propose one revision → compare the next approved result**.
Use a reviewed solvable workflow and synthetic/reference data with known
structures to test recognition of failure and improvement before granting
broader execution authority. Failure to separate structures alone must not be
reported as proof that the required nodes are missing from the catalogue.

### 4. Bounded self-directed iteration

The finite existing-node continuation above implements the first part of an
iterative session. The broader target is **inspect -> propose
edits -> validate -> run -> compare -> retain or revert -> repeat**. Within
that permission, the assistant can control layout, add/reconnect approved
nodes, and tune declared parameters without asking about every minor change.

Before starting, agree on the goal, representative data, evaluation criteria,
allowed nodes/parameter ranges, iteration and time limits, compute/memory and
cloud-cost budgets, and stopping conditions. Provide pause/cancel, a visible
history, checkpoints, and a final comparison against the starting workflow.
New custom-code execution, dependency installation, broader data access, and
file publication remain separate permissions unless specifically authorized.

#### User control over review frequency

The unreleased finish-review checkbox above is a narrower implemented slice:
it pauses at a model's proposed Finish. It is not the broader per-iteration or
checkpoint scheduler below, and it does not promise unrestricted continuation.

Add a separate review-frequency control, with a slider or similarly simple
choice: **Review every iteration → Review at checkpoints → Review at the end**.
This controls when the session pauses for comments or acceptance; it is distinct
from the existing assistant-initiative slider, which controls clarification
questions while drafting a plan. Preserve the original goal and all feedback
when proceeding to the next candidate. Every iteration still records its result
card, even when the user chooses not to review it immediately.

At the end-review setting, proceed only within the approved operations,
parameter ranges, dataset and compute/cloud-cost budgets. This is not permission
for unlimited retries, silent data uploads, new custom-code execution or
overwriting the accepted workflow. Input/output previews and numeric results
sent to a provider require a disclosed sharing choice; the current input-preview
checkbox is not blanket approval to send all future outputs. Permission changes
and genuine scope/evidence blockers still pause the session for the user.

#### Stop decisions and the best observed result

VIPP must enforce finite iteration, elapsed-time, resource and model-use budgets,
independently of the model. Stop on agreed goal criteria, no meaningful
improvement, repeated/oscillating candidates, repeated failures, insufficient
evidence, cancellation or a budget limit. The model can recommend stopping
earlier but cannot extend the limits itself.

Keep the accepted baseline and best observed candidate separate from the latest
attempt; a worse final attempt must not silently replace the better result.
Finish with a comparison, the evidence behind the outcome assessment, remaining
limitations, effective parameters and a clear stop reason. Failure to meet the
goal should be visible as failure or partial progress, not as success merely
because execution completed. Accepting the selected workflow is a separate
reviewable action, with reproducible graph/run provenance and unchanged source,
axis, calibration, cancellation and stale-result guards.

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
Explicit output-aware assessment now follows that foundation without waiting
for custom-node generation. An approved run/assessment/revision comparison now
supports a bounded existing-node iteration controller; review-frequency control
remains future work. AI-assisted custom-node creation follows its separate
execution/trust prerequisites, rather than being a dependency for evaluating
existing workflows.

Before promoting a stage into a release, demonstrate wrong-axis/type handling,
unsafe generated code rejection, cloud opt-in/cancellation, interrupted runs,
undo/reopen, missing-node/version handling, and headless/batch replay of accepted
workflows. Review the builder and iteration experience with nontechnical users.
No provider-specific integration or autonomous research claim is committed by
this plan.
