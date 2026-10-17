# Workflow assistant: planning, result assessment and bounded refinement

Status: unreleased local testing, continued 2026-10-15; issue 80. Public task instructions
live only in `vipp-mkdocs/docs/how-to/workflow-assistant.md` on
`docs/ai-workflow-proposals`. No publication or release is implied.
This continuation is after 0.16.0a4, not a feature of that numbered release.

## Contract and ownership

`core/workflow_proposal_catalogue.py` owns catalogue version 4;
`core/workflow_proposals.py` owns the strict proposal contract. A compact index
is derived from all 146 registered operations, with
an explicit authoring status and reason for each. Detailed contracts now cover
143 operations, derived from the registry minus three concrete boundaries:
Image Source and Table Source need explicit user-selected bindings, while Batch
Output needs separate collection/publication setup. The model cannot grant
execution authority merely by naming a node. There is no registered external
program/Fiji operation in this inventory; this change adds no external tool or
custom-code runner.

All ordinary existing in-memory families are authorable, including axes/channel
selection, projection, preprocessing, deconvolution/PSFs, segmentation, morphology,
meshes, measurements, skeletons, colocalization, registration/detection/tracking,
tables and plots. Save Image is conditional on the private local destination
binding below. "Authorable" means that a graph can be proposed and checked,
not that every operation is compatible with every source or newly scientifically
validated. A proposal remains bounded to 64 new nodes, 128 connections, 256 ports
and 128 KiB proposal JSON. Dynamic outputs require actual connected metadata;
port counts are not guessed from a default or expanded beyond the cap.

`ProposalDialogue` is a frozen, Qt-free snapshot of the original objective and
an immutable tuple of question/answer pairs. The goal remains limited to 4,000
characters; accumulated answers are not concatenated into that field. Structured
answered history preserves exact text, with at most 32 unique question/answer
pairs, 2,000 characters per question or answer and a 64 KiB clarification-text
UTF-8 budget. A
detached public JSON metadata request is separately bounded to 192 KiB before
provider discovery or transmission, leaving room for the registry-backed
catalogue and complete allowed clarification history. `pack_catalogue` and
`unpack_catalogue` use lossless shared parameter/port tables and verbatim text
pools, preserving the complete registry, parameter constraints, help and
source-specific choices. The wire format names its columns/defaults/references;
it is not heuristic node selection or a summary. Proposal responses retain
their 128 KiB bound; outer provider transport limits are unchanged. Escaping
counts toward the serialized request budget: an oversized request fails before
sending, never by truncating the goal, answers or catalogue.
Plain-string calls remain supported. History is untrusted user data, not
provider instructions; oversized or malformed text fails explicitly without
truncation. Response field structure and scientific restrictions are unchanged;
the selected planning preference supplies the per-response question bound.

`PlanningPreferences(mode=...)` is a frozen, client-owned planning policy with
`ask_first`, `balanced` and `use_best_judgment` modes. Their locally enforced
question caps are 3, 2 and 1 per response, separate from the 32-pair answered
history limit. The core constructor defaults to `ask_first`; the
Goal card's Assistant initiative slider explicitly defaults to Use best judgment.
Its compact endpoint labels stay close to the slider track. The chosen mode is
bold at the start of the descriptive line below, not repeated at the top right.
The planning-only reminder remains in the fixed footer rather than being
repeated in the card.
Derived preferences are disclosed in the public request and used by both
providers; objective text or model output cannot choose a larger question cap.

Ask first prioritizes consequential clarification; Balanced prefers a starting
candidate with reviewable assumptions and asks only when an answer materially
changes a safe draft. Use best judgment asks the model to make routine choices
as explicit reviewable assumptions and aim for zero routine questions, with at
most one essential blocker if no safe, honest in-catalogue candidate is possible.
Zero questions is an intent, not a guarantee. A conservative observed-signal
candidate and stated method/parameter conventions do not establish complete
masks, biological identity, calibrated measurements or objective success.
Missing calibration/axes/units and unavailable operations cannot be invented.
This policy is not model reasoning effort or temperature and grants no tools,
transmission, calculation or scientific execution authority.

Shared public-request guidance asks future summaries, step reasons and
assumptions for everyday language explaining each choice, why it matters and
what to check, without internal parameter keys or axis codes in prose.
Uncertainty, negations, numerical qualifications and scientific caveats must
remain; exact parameters stay in their parameter
fields. This writing guidance does not change schema, caps, validation or
authority and does not rewrite existing responses.

Catalogue titles, required/safe optional parameters, types and policies come
from the actual registry. Normal choices include explicit 2D processing of a
ZYX input, mesh-border closure, metric colouring and alternative watershed
elevations when the operation's actual input contract permits them. Closure is
an explicit flat image-border cap, not recovered missing anatomy. Numeric
minimum/maximum/step metadata describes registry UI bounds, not universal
scientific limits; shared typed validation, finite/odd constraints and ordinary
execution/resource checks remain authoritative. Derived hints, private controls,
legacy node identity and model-authored auto-recalculation are excluded.

`input_ports_for`, `required_input_ports_for` and bound `output_ports_for` query
the actual detached pipeline policy. Optional Review Images/skeleton inputs,
registration/detection mode switches and metadata-dependent split/PSF outputs
are not flattened to a fixed one-input/one-output model. Mesh combination still
permits up to 16 inputs; channel/arithmetic/logical combinations use their
registry counts. Every required port must connect. Ordinary restore plus
metadata preflight checks the candidate without running scientific kernels.
Equal shapes do not establish equal physical grids. Type-preserving and
rank-changing chains still depend on normal graph/metadata capabilities; this
does not change the global registry or silently repair a scientific request.
Review Images remains array-only; meshes use the native Surface viewer.

Input capture requires a cached, owned read-only finite NumPy array of at most
64 MiB, with explicit Y/X axes and correctly typed actual Z, T and C axes.
Boolean masks, label images, up-to-32-bit integers and up-to-64-bit real floats
are supported. Complex/object/wide integer data, ambiguous axes, or mismatched
shape/dtype metadata remain unsupported. Time/channel support is conditional on
each proposed operation, not a blanket ability to process any rank.
No source file is loaded or lazy data materialized by this boundary. Exact
finite-range and local content fingerprint checks scan the bounded cached data.
Paths, source names, acquisition history and existing graph parameters remain
private. Source, graph, workflow session and editor settings are rechecked
before accepting/applying a response. Questions block Add; validation checks
graph safety, not image accuracy.

## Locally bound output destinations

`core/workflow_proposal_outputs.py` resolves a user-authored folder choice to
a private `OutputDestination`. Goal offers Ask me when saving, Same folder as
input and Choose folder. The exact **use the same folder as input** answer may
resolve the folder of one verified physical file or OME-Zarr store. Sample/live
inputs and multiple-file bindings must use the picker, even if those files share
one folder. Source display names are never
treated as paths. Destination changes invalidate stale proposal authority.

Only the literal USER_OUTPUT_DIRECTORY and selection state enter the public
request. The actual folder remains local. Save Image proposals use an explicit
format, enabled writing and overwrite=no; generated destination filenames are
bound locally after proposal validation. Unresolved destinations keep saving
plans blocked rather than choosing a path or silently dropping the writer.
Send, Preview, navigation and Add create no files. A separate explicit Run
invokes ordinary guarded calculation and the existing no-overwrite publication
contract. Rechecking a path does not itself authorize writing or replace the
writer's atomic no-overwrite guarantees. Assistant-added Save nodes retain a
persisted per-node manual barrier: parameter edits do not automatically save.
Existing ordinary Save nodes keep their prior behavior. Saved workflows retain
their local output parameters; the model receives no filesystem tool.

## Explicit image context

`capture_proposal_context(..., include_images=False)` shares no pixels by default.
With permission, `core/workflow_proposal_images.py` creates immutable owned PNG
previews from the same verified source revision: at most seven representative
slices/orthogonal views, with integer-stride sampling when an edge exceeds 512
pixels. Captions disclose selections, native axes, source-range grayscale
mapping and sampling. Pixel aspect is not physically rescaled; calibration is
supplied separately. No MIP is misrepresented as a complete 3D observation.
These are presentation-only bytes, never scientific results or cache mutations.
Preview generation remains limited to scalar YX/ZYX sources. Other supported
source axes can be described in metadata; enabling unsupported previews fails
explicitly instead of guessing a timepoint, channel or spatial interpretation.

`ProposalContext.image_previews` is a tuple of `ImagePreview` values
(`label`, `png_bytes`, `description`). `public_request()` remains JSON metadata
plus a preview manifest with labelled selections, byte counts and hashes, not pixel
payloads or file paths. Codex attaches the approved PNG bytes separately.
The preview dialog shows those exact bytes, not a later recapture. Settings,
source or goal changes invalidate approval. No bytes are sent merely by opening
the assistant, checking an option or preparing the preview.
The preview uses a distinct framed outer scroll area and framed read-only
Advanced JSON. Its bounded initial height follows the content: short goals do
not leave a blank stretch before Close preview, while long text, approved images
and expanded details scroll without truncating the complete request. Opening
the preview does not send data or copy it to the clipboard.

## Completed-run output assessment

Manual result review remains explicit. A separate bounded refinement controller
can delegate a finite assess/revise/Add/Run series only after the user chooses a
limit and starts it; neither a sharing checkbox nor ordinary Run starts that series.
Goal presents **Include input image previews** and **Assess workflow outputs**
as independent prominent choices, both off by default. Selecting either sends
nothing. The input choice governs input pixels, including source-intensity
backgrounds in output overlays; output assessment never implicitly enables it.
The output choice allows preparation of result diagnostics and, with Codex,
bounded output previews. A trusted local connector remains measurements-only.

After the latest added candidate completes an ordinary explicit Run, the
composition root binds `core/workflow_assessment.py`'s `AssessmentSnapshot` to
that workflow session, dispatch and completed run ID. It captures detached
source/output data and carried metadata; hashes plus cache identities, execution
states and the graph fingerprint reject replacement, mutation and stale runs.
It neither calculates a node nor changes scientific caches. The caller repeats
ownership and `assert_current(pipeline)` checks before accepting a reply or
using its revision feedback. A failed, cancelled, active or changed run cannot
inherit a prior successful run's evidence.

`AssessmentContext` exposes a privacy-minimised public request using local
`input`/`n1` aliases rather than private node IDs/names. It includes the original
objective/dialogue, candidate topology/parameters, source calibration/grid and
selected actual output diagnostics. At most eight output ports are selected,
prioritising terminal meshes and their actual upstream mesh-input masks;
unsupported or omitted ports remain explicit. Source/output image adapters
currently require scalar explicit YX/ZYX grids, at most 4,000,000 elements per
array and a 64 MiB detached snapshot budget. Large/lazy or time/channel arrays
are not silently sampled or projected to fit. Mesh-edge diagnostics are bounded
to 200,000 triangles; numeric table summaries omit row text and disclose
truncated column coverage.

Finite extrema, foreground occupancy, border contact, face/full foreground
component counts, label identity counts and available mesh geometry summaries
describe the complete supported population, never PNG pixels. One label can
have disconnected pieces; connected nonzero foreground can join different
labels. Mesh display groups/colours are not connected topology or biological
identities. Mesh edge incidence does not validate orientation, self-intersection
or completeness. These diagnostics are evidence, not goal-success assertions.

**Preview results shared** shows the exact bounded evidence prepared for the
next assessment; it sends nothing. Codex can receive up to seven PNGs sharing
the existing 2 MiB image budget, with declared slices, display ranges, palette
and integer-stride sampling. Source pixels/overlays appear only with the
independent input-image opt-in. `core/workflow_assessment_mesh.py` renders the
actual captured mesh with two fixed opposing orthographic oblique cameras,
carried compatible Z/Y/X scale and origin, source-grid framing where compatible,
and original per-object colours. All admitted triangles are drawn without
simplification; triangle-centroid back-to-front ordering, flat lighting,
transparency and subpixel visibility are explicit display approximations, not
geometry/topology validation. These views and exact mesh-input mask slices share
the same seven-preview/2 MiB budget, with omissions disclosed. No live viewer
capture or unrelated layer can leak into them. No full volume,
source path/name, acquisition history, table-row text or unrelated older
candidate is transmitted. Unsupported coverage and omissions are disclosed.

**Assess results** explicitly sends that run-bound request. The provider has
no tools, execution or graph-mutation authority. The strict frozen
`AssessmentReport` contains matching `assessment_id`/`run_id`, an overall
`achieved`/`partly_achieved`/`not_achieved`/`cannot_assess` interpretation, bounded
criterion/status/evidence checks, findings, suggested changes and limitations.
Its comparison capsule separates relative change from overall goal status:
`improved`/`mixed`/`worse`/`unchanged`/`cannot_compare` names the exact predecessor,
or `first_run` has a null predecessor. A new comparative response cannot omit
that capsule or claim `first_run`; earlier no-predecessor saved reports remain
readable. Node aliases are candidate-local: comparison guidance requires output
roles/lineage rather than equating `n1` across different drafts.
Unknown fields, confidence scores, executable graphs, paths, duplicate/non-finite
JSON, wrong identities and contradictory achieved/unmet checks are rejected.
At most one format repair follows the selected attempt budget; transport/auth
failures are terminal, not automatic retries. The independent accuracy caveat
remains visible: a model's verdict is not validation or biological ground truth.

The existing bounded goal-check fields now request separate applicable criteria
for coverage/missed wanted parts, unwanted foreground, continuity versus false
bridges, intended separation versus false splits, and mesh quality/admissibility.
Trade-offs must be explained: deleting true structures can reduce foreground or
component counts, and smoothing can hide features. More review nodes or a
mask-preserving mesh change cannot establish better segmentation. Comparison
guidance distinguishes gains, losses and unavailable evidence, without inventing
reference masks, accuracy/overlap scores, biological identities or a best result.
The report schema and provider authority are unchanged. These are assessment
instructions, not evidence that a model reliably detects each defect. The local
progress wording says results are being checked against the goal; it does not
claim measured scientific progress or remote-model liveness.

**Revise using results** is a separate explicit request. It sends the complete
bounded measured feedback, accepted report and preview manifests alongside a
fresh ordinary proposal context/catalogue. Codex revision resends the same
previously approved bounded output PNGs, still subject to current image consents
and the shared image budget; local revision remains measurements-only. Original
goals and answers are retained. The reply is a proposed
new candidate checked by the existing proposal validator, not an already
improved scientific result. In Manual mode it requires a separate Add, Run and
review; it neither replaces the accepted baseline nor reruns indefinitely.

`core/workflow_assessment_history.py` freezes the accepted report and selected
public measurements/graph from only the last completed, assessed run into immutable
JSON bytes (96 KiB maximum). No PNGs, geometry buffers, private owner or recursive
history are retained. Comparison binds source SHA-256, full physical grid,
original objective/answers/deferred questions and both image choices. Attachment
and every provider send recheck that binding; a changed source, goal or sharing
choice fails closed rather than comparing unrelated evidence. Current evidence,
previous record, schema and instructions must fit the existing 192 KiB request
limit without truncation. Provider/configuration/session history reset remains
the composition owner's responsibility. A previous interpretation is evidence to
criticize, not ground truth to repeat.
Failed attempts never replace that completed comparison record or create an
assessment. After recovery, the next completed result compares with the last
completed result; failed or partial caches are not a new baseline.

## Separately started finite refinement

`core/workflow_refinement.py` owns Qt-free bounded state transitions and semantic
candidate signatures; `ui/workflow_assistant_refinement.py` schedules only the
normal assessment, proposal, Add and ordinary Calculate-all paths. Goal's
**Refine using the results** slider defaults to **Manual (0)** and permits at
most five **additional calculation attempts**. It grants no authority when moved. **Start
refinement** in Plans requires a completed first ordinary Run with output
assessment enabled, a connected provider, an exact current snapshot and a safe
whole graph. The adjacent disclosure explicitly delegates model requests,
checked candidate Add and Run for this finite session. The input-image choice
remains independent and both sharing choices remain default off.

Each attempted round is revision → checked Add → Run, followed by assessment only
after successful completion. A starting assessment is requested if needed.
The selected limit `N` caps attempted additional Run dispatches, including failed
calculations; completed, assessed results are counted separately. Hard bounds are
five attempts, 30 minutes total and `2 * N + 1` logical requests; each request retains at most two
format attempts and the existing per-attempt timeout, capped by remaining loop
time. Normal executor memory/device/resource policy is unchanged. Stop and Close
revoke late replies and request cancellation only of the loop-owned run;
scientific kernels stop at their normal safe checkpoints. Already sent data
cannot be recalled.

The whole Calculate-all graph, including earlier or disabled nodes, must use a
reviewed positive operation inventory and exactly the selected source. Save
writers, another source, external/custom or future unreviewed operations stop
the series rather than being dropped, bypassed or repaired silently. Each new
candidate is an ordinary undoable Add and earlier completed branches remain intact. The
loop stops on questions, invalid/stale/unavailable results, non-repairable errors, denial,
source/goal/settings/permission changes, cancellation, repeated semantic drafts,
budget limits, `cannot_assess`, or `achieved`. The last is explicitly an AI
judgment, never scientific validation or proof of completeness. Later results
are not automatically superior. Up to six reports remain visible in Result
history; no automatic best-candidate selector, wider review-frequency policy,
custom-node generation or unlimited optimization is introduced.

### Owned failure and limited repair

An assistant-owned failed scientific run reports its actual operation and local
error instead of becoming a generic graph-change warning. It creates no
`AssessmentContext`, report or output-preview approval. Its failed draft and
message remain history; the last completed assessment may remain visible only
as historical evidence, not current Run/Add authority or an automatic best pick.

Only positively identified structured mesh-input topology errors are eligible
for this repair path. At most two repair transitions occur per started series,
inside the same attempted-Run, logical-request and wall-clock budgets; a repair
never resets a counter/deadline. Unknown errors, resource denial, cancellation,
staleness, security/provider/I/O failures and unsafe cleanup stop for manual
review. The composition root verifies the exact failed own-Add/session/graph
revision and rolls back only that Add before capturing a fresh replacement
proposal. This prevents the broken branch from rerunning; prior completed
candidate branches and unrelated edits are not deleted. Any ownership mismatch
or rollback failure stops rather than reconstructing a guessed workflow.

The fresh repair proposal receives a safe structured code, operation and failed
recipe, not a fabricated result assessment, raw traceback or private paths.
`core/workflow_run_failure.py` owns that immutable, canonical feedback capsule
(48 KiB maximum), candidate-local node aliases and source/rollback-context
guards. It admits only registry-typed finite numeric/Boolean/static-choice
parameters; private, derived, dynamic/free-text and unknown values do not leak
through the failed recipe. The complete proposal request retains its existing
192 KiB bound without trimming evidence to fit.
Only separately opted-in input previews may accompany it; no failed-output or
older output PNG is included in that input-only repair request. Normal proposal validation,
checked Add, whole-graph inventory and ordinary calculation/resource guards
still apply. `MeshInputTopologyError` classifies known rejections without
weakening them: no face deduplication, welding, inferred gap filling or scientific
geometry change is introduced. The UI separates attempted/completed counts and
names the bounded repair step. A repaired workflow is not claimed improved until
it completes and its actual results are assessed.

## Review a bounded investigation finish

Unreleased finish review applies to the ChatGPT investigation engine, including
its accepted-result warm start. The native **Review before finishing** checkbox
defaults on. The core option `review_before_finish=False` preserves legacy
callers, while the local connector's earlier bounded refinement series is
unchanged. Changing the checkbox starts no request or calculation.

An otherwise valid model Finish is an exact completed nomination for review,
not automatic user acceptance. Pending review blocks concurrent model work and
offers **Keep improving**, **Good enough** and **Stop**. Optional feedback can
accompany Keep improving; the original goal, protected accepted baseline and
complete notebook/experiment records stay in the same session. Resuming retains
all consumed trial/action/model-call/image budgets and the original absolute
active-time allowance. Human pause/review time is excluded by the shared native
continuation; consumed work is not refunded. The model may test remaining
addressable problems within those bounds, using ordinary prospective plans and
actual evidence; it cannot extend limits or erase limitations to manufacture
success.

Good enough rechecks ownership/freshness and accepts only the exact completed
nomination into the existing checked-draft path. It performs no new model or
scientific work, so an elapsed work deadline alone need not discard an otherwise
fresh completed result. A changed source, workflow, permission or other authority
still blocks stale acceptance. Selecting the already accepted warm-start baseline
keeps that graph without a duplicate Add. Add and Run remain separate for a
selected alternative; no trial is silently published and the accepted graph is
unchanged during review. Stop preserves
that graph and ends further investigation. A user acceptance or model preference
does not certify scientific quality, a simple solid or anatomical correctness.

## Providers and bounded repair

`core/workflow_proposal_model.py` retains the metadata-only trusted loopback
Ollama/OpenAI-compatible adapter. It rejects external endpoints, credentials,
redirects, proxy routing, cloud-model identifiers and filesystem/model tools.
An arbitrary local proxy may itself forward data; trust in its configuration
is explicitly required. This adapter does not gain vision from the image option.
The Ollama path still requests `num_ctx=16384` with `truncate=False`. The full
catalogue increases token requirements; a local server/model may reject a
request rather than accept its complete context. No catalogue trimming, context
increase, local-model inference qualification or Codex inference qualification
is implied by this provider-free composition milestone.

`core/workflow_proposal_codex.py` is the optional installed Codex connector.
It uses an app-owned isolated profile, official user-initiated browser sign-in,
a managed configuration and a capability-gated stdio app-server. Existing Codex
credentials, personal configuration and chat history are not copied. The
connector exposes no execution environments or dynamic tools; shell, web,
plugins/MCP, skills and other unrelated features are disabled. A deny-by-default
read-only permission profile covers only the empty temporary request workspace.
Tool/permission requests are declined and the request stops, never falling back
to full-access execution. Unsupported runtimes/configurations fail visibly.

The native Codex output grammar uses a shallow, strict `proposal_json` string
envelope for runtime compatibility. The complete unchanged domain schema is
supplied in developer instructions; the decoded proposal still goes through
the same local scientific/graph validator and 128 KiB UTF-8 limit. The bounded
outer envelope is a transport detail, not a relaxed graph format, and is never
shown as the proposal or sent back as repair feedback. Malformed envelopes are
terminal protocol errors; invalid inner proposals use the ordinary repair
budget. There is no automatic schema fallback or extra planning turn.

Connection state is separate from proposal validation. The assistant performs one
read-only connection check when opened and offers an explicit refresh. It tracks
only the login child it started, then checks authoritative status once after
that child exits; it does not repeatedly query authentication while the browser
is open. Login waiting is bounded and cancellable. Cancellation/closing stops
only its own pending login child, never signs out or removes saved credentials.
Goal edits do not lose sign-in completion feedback or restore proposal approval.
Neither login completion nor a connection check captures/sends images or starts
a plan. A signed-in status is not proof of model entitlement or scientific success.

Both providers return the same locally validated proposal. The normal UI
allows two total attempts by default: a rejected response and exact local
validation feedback can be sent for one automatic repair. Both providers now
default to 600 seconds per attempt, with a 600-second maximum. Two attempts
can therefore use up to 20 minutes of model-request time; this is not a new
scientific execution or iteration budget. Cancellation remains available.
The live elapsed indicator measures local wall-clock time for the current
planning request, including its bounded repair, independently of provider
progress events. It is not proof that the remote model is alive, generating
tokens or making reasoning progress, and is not a completion percentage.
Graph/source guards
are not relaxed. Terminal transport/authentication failures stop rather than
starting another VIPP turn. Codex's explicit `willRetry: true` error events for
the acknowledged active thread/turn allow recovery in that same child and turn,
within the original deadline and stream-byte limits. Preflight, mismatched or
malformed recovery events fail closed. Interrupted answer text is invalidated
until a fresh completed answer is returned; recovery is not JSON repair and
never resets the deadline. Cancellation revokes response acceptance and future
repair; already transmitted requests cannot be recalled. No scientific execution is part of
generation, validation, repair or adding a graph.

Attempts explicitly distinguish `request` failures from returned `proposal`
content. An empty but completed text response is checked like other returned
content; missing/incomplete responses, terminal transport errors and disabled
capabilities are terminal request failures, not grounds for an automatic retry. Request
failures must not be presented as exhausted JSON repair or blamed on the goal.
`SafeModelDiagnostic` is a frozen allowlisted capsule containing only provider,
stage, category, bounded numeric code/HTTP status and a fixed reason enum. Raw
provider error messages, additional details, prompts, URLs and credentials are
never retained in that capsule. Optional diagnostic copying must copy only this
capsule, not the Advanced request or response trace.

Codex-authenticated requests are cloud transmissions subject to the applicable
ChatGPT policies. API no-training defaults are not a promise about ChatGPT
usage or zero retention. The application cannot establish retention by writing
an instruction into a prompt. Provider notices and consent must reflect this.

## UI and application transaction

`ui/workflow_assistant.py` presents a dedicated non-modal native workspace,
with the same independently usable window pattern as Batch. Goal, Discussion
and Plans are product tabs, not alternative mockups. Back and tab changes
only navigate: they neither transmit a request nor add/calculate nodes. The
input selection, goal, unsent discussion text and clarification history survive
navigation. Original text and answers remain readable in the discussion rather
than being replaced by the latest technical response.
Discussion uses one main scrolling page with natural-height conversation and
question sections. Original goal and Conversation history start collapsed behind
assistant-local disclosure headers using the shared right/down chevrons.
Collapsed frames remove only outer vertical padding; expanded spacing, shared
chevrons and font-aware header click targets are unchanged.
The global clarification warning directs answers to Discussion and keeps Add
blocked; the local Ask first/Balanced instruction refers to questions below.
Under Use best judgment, explicit Send answers may delegate fields left blank
to a fresh best-judgment request. Blanks are not fabricated user answers and do
not remove questions locally or make the existing proposal eligible for Add.
Only a newly accepted, locally validated response can replace those questions;
genuine unresolved blockers may remain.
Current question cards stay open; wrapped questions, readable answer rows
and the bounded message box follow the page's natural flow. Only the current assistant
reply's duplicated question appendix is suppressed in the rendered conversation;
its full literal `_chat` and draft records are retained. Earlier questions and
answers remain in readable history. Assistant replies and other non-user blocks
align left on neutral bordered surfaces; user messages align right in blue-accent
blocks derived from the same information accent as the stage tabs. Sender labels
stay inside their own aligned block, with readable user-label text in both themes.
A recorded clarification reply keeps the Assistant’s question caption and
literal referenced question together in a blue-tinted neutral inset with a blue
outline inside the user reply. The quote and sender labels use normal readable
text for light-theme contrast. Your answer is a separate bold heading before the
literal answer. Display-only pairing uses the exact
recorded question and answer, not guessed boundaries from splitting text.
Free messages render without their generated question anchor, which remains
in literal history and request data. Disclosure and display formatting do not
alter dialogue limits, request content, consent or Add/Run authority; expanding
or scrolling a section does not send a request.

`ui/workflow_assistant_style.py` owns presentation-only full-width, equally sized
numbered stage tabs, palette-derived surfaces and action-icon colours. The native
UI frames the source/goal, discussion composer and plan summary. The title and
workflow-toolbar action use the AI sparkle icon. The navigation row keeps Back
left, the current step centred and a passive View plan action in Discussion
when a draft exists. Close occupies the very bottom strip beside
Planning only · Add and Run are separate actions, not the navigation row.
Goal initially has no Next button: Send goal is its single planning action
(Show example plan for the fixed no-AI recipe). Once draft history exists,
Goal's Next: Discussion returns to retained discussion without sending or
renewing sign-in/approval. Discussion keeps View plan; Plans has no Next.
These shortcuts are disabled while busy. An initial failed/no-response request
without a draft leaves Goal's Next hidden. All tabs remain available for passive
browsing,
including empty states and Advanced error details before the first draft.
Each stage keeps its action strip outside the scrolling content: Goal pins
Preview what is shared and Send goal (or Show example plan); Discussion pins
Preview what is shared, Send message and Send answers when questions exist;
Plans pins Add to workflow and Run workflow. The Discussion message box remains
inside the main scroll with the question and history sections, not in a pinned
composer. Goal starts with a smaller bounded editor; both editors grow modestly
with available viewport space and retain a font-aware minimum. Plans heading,
draft selector, review, Advanced details, scientific caveat and refinement remain scrollable
above Add/Run. Resizing and scrolling change presentation, not literal text,
request approval, privacy policy or execution authority.
Retained history remains browsable with a blank goal. A separate Settings gear and
header connection icon do not send requests. The green header check appears only
for confirmed Codex sign-in, not for local/no-AI selection or scientific success.
Live palette/style refresh updates presentation without changing consent, input
captures, clarification fields, draft selection or Add/Run authority. Navigation
and provider/source/scientific guards remain in their existing owners.
Palette/style change notifications queue the existing dialog-owned single-shot
appearance timer, as the inherited-theme probe already does. The deferred local
stylesheet update avoids nesting inside Qt's ongoing global repolish cascade;
this bounded handling change does not establish that all native Qt crashes are
resolved.
Assistant buttons use a scoped `AssistantButton` subclass of the existing
`ToolbarCommandButton` for a paint-only icon-to-label gap, including busy send
spinners and left-to-right/right-to-left layouts. Stored captions, mnemonics,
signals and request authority are unchanged.

Goal owns the selected input, objective and the separately opt-in, off-by-default
Include input image previews setting.
Discussion owns Send message/Send answers actions and the readable conversation;
Goal uses Send goal (or Show example plan for the fixed no-AI recipe). A pending
model request replaces the sending button's shared paper-plane icon with an
inline loading spinner and a clear sending label in Goal or Discussion. The
spinner is presentation-only and does not claim a completion percentage; the
workspace title and toolbar menu keep their distinct AI sparkle. An accepted
result opens Discussion
when clarification questions remain, otherwise Plans. Rejected, failed,
cancelled or stale results cannot create an accepted draft.
The sharing disclosure names the destination and whether bounded previews are
included. Clicking Send, Update plan or a retry action approves that specific request
and its bounded validation repair; there is no separate permission checkbox to
re-tick after each answer. Choosing a provider, applying settings, navigating,
signing in or opening a sharing preview is not permission to transmit.
Cancellation still revokes late response acceptance and future repairs.

Plans retains text-only draft records: summaries, questions/answers, steps and
proposal text, not image arrays, PNGs, credentials or reusable stale captures.
Source metadata and live context/owner objects are not historical records.
Earlier drafts are read-only and cannot be added or run. Only the current,
locally validated latest draft can become eligible for Add. Goal/input/workflow
or relevant provider/settings changes revoke live eligibility; retaining a
history entry does not make its old scientific context current. Failed or
cancelled revisions do not discard the user's current answers or accepted
earlier text.
An initiative-only change retains goal, current answer text and history while
cancelling the old planning request, revoking approval and stale Add/Run
eligibility. It does not transmit or grant eligibility merely by moving the
slider; a fresh explicit request and all source/workflow guards are required.

The pinned global status row identifies the current initiative or image-preview
include/exclude choice that invalidated a draft or cancelled an in-flight request.
This is a choice-change warning, not a failed AI response. With no draft or
in-flight request, the same change is informational without an out-of-date-plan
warning. With the input still current, Update plan uses the normal guarded Send
goal path: it retains the
literal goal, committed/current answers and allowed best-judgment blank choices,
but excludes and keeps an unsent Discussion message. New source/image context
is captured by an explicit Update, Preview or normal Send action, never by the
option change.
Preview alone does not transmit. Required unanswered fields instead expose the
passive Answer questions action; a changed source/workflow exposes Review goal,
without reusing stale assumptions or transmitting.

Genuine failed requests offer Retry plan, Retry message or Retry answers as
fresh explicit requests with the same selected attempt limit and normal bounded
validation/repair. Retry message uses the current visible composer; if edited,
the action says Send edited message. Separate retries do not carry prior
invalid-response feedback or increase limits; no new automatic retry loop is
introduced. Unauthorized/unsupported-model/image failures expose Check connection
to open Settings; request-format, capability, configuration or protocol failures
expose Show details, not a blind AI repair. Those navigation/settings actions do
not send. The no-AI provider uses Refresh example instead of model transmission.
Recovery has a busy spinner and normal double-click/cancellation/generation
guards. An accepted result clears the recovery action. Updating/retrying does
not replace or rerun previously added nodes; Add and Run still require explicit
actions on the latest eligible, locally validated draft.

Conversation/draft history belongs to the retained native workspace in the
current session; it is not a disk transcript or a workflow-file extension.
Closing the workspace cancels pending assistant work and revokes live Add/Run
authority. Reopening retains goal/chat/draft text, but the latest draft is
read-only until a fresh request is accepted. Saved accepted nodes still use
ordinary workflow persistence, independently of this transient conversation.

The connection indicator is status-only. A separate modal Settings editor
stages provider/model choices until Apply; Cancel discards those edits. An
explicit browser sign-in is still an external account action, not an inference
request; cancelling settings does not sign out or erase saved credentials.
Changing provider/settings revokes the live draft and any pending request.

Settings uses a compact, theme-aware layout and a distinct live connection
card rather than a plain error embedded among model controls. Sign in with
ChatGPT / Sign in again is an explicit browser action; Check connection is a
separate read-only status action. Pending sign-in/status work is shown in place,
and completing browser sign-in still uses the owned-child lifecycle and one
authoritative status check described above. No second polling loop or implicit
model request is introduced by this presentation. Optional model overrides,
thinking effort and the existing bounded attempt/time limits are initially
collapsed under Advanced. Its expanded inner containers are transparent so the
outer raised card remains visually continuous; input backgrounds stay distinct.
This is presentation only, without settings or authentication changes.
Local model discovery remains an explicit action;
changing providers does not start or install a local service.

A recognized managed runtime-feature inventory drift is labelled Connection
setup needs an update and offers an explicit Repair connection setup action in
the connection card. `CodexRuntime.repairable` identifies only that recognized
drift; `repair_codex_profile` performs the explicit migration. Repair accepts only
an intact, canonical VIPP-owned template
with the expected marker; it retains a backup of that managed configuration and
marker before updating only the owned feature flags to the current disabled
inventory. It then rechecks connection status. It does not clear or replace
cached credentials, initiate browser sign-in, capture an image or send a model
request. This is an explicit, narrowly verified compatibility migration, not an
automatic profile reset, general-purpose TOML repair or permission change.
This recognized update does not require the user to edit JSON or configuration.
Refresh and sign-in never perform this migration automatically. A persistent
repair lock prevents uncertain/interrupted work from being silently resumed;
that state stays blocked for diagnosis rather than deleting credentials or
guessing which partial files to adopt. Backups contain only managed
configuration/marker bytes, not authentication data. The subsequent official
status command may inspect its own cached sign-in; VIPP does not copy that data.

Other managed-profile verification failures remain Connection setup needs
attention. Redirected, noncanonical, extra or arbitrarily modified configuration
is still rejected rather than overwritten. Sign-in is paused while setup is
blocked; Check connection remains a read-only status action. Advanced holds
Connection details for diagnosis, not a hidden manual fix or credentials.
Provider edits are still staged, but explicit repair and browser sign-in are
immediate profile actions, not deferred until Apply. Cancel does not undo a
completed repair or sign-in. Closing Settings cancels only its owned pending
connection work, preserving completed credentials.

Numbered step cards, questions and automatic local checks are the default
review. JSON/request/raw attempt traces are hidden under Advanced details.
`ui/workflow_assistant_details.py` owns a display-only AssistantDetailsEditor
using the system fixed-width font family for Proposal JSON, Request, Attempts
and the sharing preview's Advanced shared-data JSON. Proposal/request/shared
JSON get dependency-free lexical colour cues for keys, strings, numbers,
boolean/null literals and punctuation, adapting to the actual light/dark
background. Attempts is fixed-width only, not parsed as a JSON transcript;
Plan text, Goal and ordinary chat retain their normal fonts. Incomplete JSON
may still be colourized without acceptance: highlighting does not validate,
repair or reformat content. Literal text, wrapping, read-only/editor roles,
copying, undo/history, request data and Add/Run guards remain unchanged.
The editor explicitly combines the fixed-width family with its actual immediate
parent's point/pixel font size, following parent FontChange and ParentChange
events. The actual parent is authoritative; this does not override an existing
whole-dialog scoped stylesheet font cascade or redesign product-wide font
accessibility. The whole font-size/style and highlighting refresh blocks and
restores editor/document signals rather than triggering proposal edits,
validation or transmission.
Assumptions to check and What to check after running use independent chevron
disclosures, collapsed by default. Expanding them reveals the complete true
hanging-bullet lists; disclosure changes neither data nor eligibility and sends
no request. Current clarification questions and blocking warnings remain
prominent outside these disclosures. Every accepted assumption stays unchanged
rather than being translated, classified or dropped. Original assumptions also
remain in Advanced details, draft history and the graph note after Add.
The separate What to check after running section gives
three conditional human-review prompts: compare wanted parts and background
with the original; inspect joins/splits for separate masks and all slices of a
Z stack; check scale, units and partial edge objects when measuring size.
This generic, non-exhaustive guidance is not model-authored assumptions,
candidate facts or automatic scientific validation. It never enters proposal
JSON, raw plans, history or graph notes.
The UI explicitly says VIPP has not automatically assessed the result and that
the plan's assumptions are starting choices, not confirmed facts, including for
the no-AI example.
Optional expert JSON edits are debounce-validated. There is no Validate JSON
action and no requirement to repair JSON by hand.
Status uses icon plus text: green means structurally ready to add, amber needs
information/permission, red cannot accept. Neither green nor a generated
explanation certifies scientific success. Progress exposes bounded attempt/
repair events, not private model chain-of-thought.

Generation tokens and detached background inputs discard closed, cancelled,
stale and superseded results. Clarification answers regenerate a proposal;
they do not erase questions from untrusted JSON to bypass approval. Failed or
cancelled requests retain the visible answers for correction or retry.
Only accepted returned plans replace the question fields and commit history.
AI-authored content remains literal text within trusted UI formatting; model
markup is not interpreted. A new explicit request
freezes its approved context; cancellation and relevant changes invalidate the
pending approval rather than accepting a response for superseded data.

Changing image-preview sharing invalidates the sendable capture and plan,
cancels pending work and revokes request approval. A subsequent explicit Send
approves the fresh capture; no request is made by the checkbox itself. The original goal and
answers survive only while the captured source/workflow freshness guard still
passes. A pixel-free old context plus owner guards repeated toggles and fresh
recapture; it is never sent. Goal/source/provider/settings changes clear this
guard and live answer context. Historical text may remain readable, but it is
not proof that the old source is current or permission to reuse a capture.
A changed source cannot inherit earlier live approval by toggling sharing;
late cancelled results remain rejected.

`_widget.py` owns the guarded live add transaction. Only validated additions
are appended, preserving existing nodes, edges, layout, caches and counters.
Summary/assumptions become a graph note. Failure rolls back graph, caches, dirty
state and history. Adding is one undoable edit without scheduler execution;
undo/redo retain that non-running behavior. Accepted graphs persist as normal
workflows, without model credentials or image previews.

Run is a separate user action enabled only after an eligible draft has been
added to that same workflow. It routes through the ordinary guarded Calculate-all
entry point, with current workflow/source/compute state rechecked; it does not
call the model or bypass production resource guards. Normal manual-node and
calculation policies still apply. Add, navigation and planning never implicitly
run, and Run does not inspect results, refine parameters or start a scientific
optimization loop. Running changes the workflow, so older proposal captures
must not be treated as fresh authoring context.

## Evidence and remaining scope

The 2026-10-11 first-stage core/transport/UI, native synthetic and three-case
Qwen records remain under `VIPP-local-tests/ai-authoring-20261011`. They used the
earlier catalogue/harness and are not re-scored or claimed as evidence for this
new Codex/image integration. Curated references are not API inference results.
The Qwen campaign established structural first-attempt results only, including
a scientific explanation error that structural checks missed. No general model
ranking, cloud parity or acquired-image accuracy is established.

Earlier continuation evidence lives under
`VIPP-local-tests/ai-inspect-plan-20261012`. The 2026-10-13 send-first flow
documentation checks are under `VIPP-local-tests/ai-assistant-flow-20261013/docs`:
26 application documentation tests, 21 manual content contracts, 50
application-to-manual routes and a strict offline content/link build passed.
The subsequent spinner/footer/sharing-preview polish repeats these documentation
checks under `VIPP-local-tests/ai-assistant-polish-20261013/docs`. Neither
documentation follow-up launched GUI windows or qualified responsive rendering.
The button-spacing documentation follow-up has separate records under
`VIPP-local-tests/ai-assistant-spacing-20261013/docs`.
The Discussion layout documentation follow-up has separate records under
`VIPP-local-tests/ai-assistant-discussion-20261013/docs`.
The disclosure/duplicate-question documentation follow-up has separate records
under `VIPP-local-tests/ai-assistant-disclosures-20261013/docs`.
Only checks actually recorded are claimed. Source tests, mock
provider protocol and native synthetic UI are
separate from live inference, installation and release qualification.

Current registry/destination metadata evidence is recorded separately under
`VIPP-local-tests/ai-assistant-full-library-20261015`. Source-aligned metadata
checks do not establish model plan quality, acquired-image accuracy, native GPU
qualification or output-aware refinement. Do not carry the earlier Qwen scores
forward as scores for this larger catalogue.

Explicit run-bound output-image/measurement critique is now implemented as
described above; provider-free contract tests do not establish model accuracy.
Deferred: unbounded scientific optimization, large/lazy-volume handling and arbitrary source creation,
cloud BYOK providers, macro translation, custom nodes and unattended batches.
The launcher is synthetic and performs no provider inference. No active/shared
environment install, live-session restart, merge, push or publication is part
of this development handoff.
