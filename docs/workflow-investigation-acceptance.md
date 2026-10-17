# Blinded product investigation acceptance

This protocol tests the opt-in product investigation controller, not a separate
agent-written segmentation workflow. Execution and plausible coloured surfaces
are not scientific success. The final candidate must be selected and locked
before any independent reference is opened or used for a comparison.

## Scope and blinding

The sole scientific input is the pre-existing `input.npy` with its approved
`metadata.json` in
`VIPP-local-tests/ai-assistant-independent-solver-20261016/input`.
The input is uint8 ZYX, 12 × 96 × 128, with 0.45 micrometer spacing on all three
axes and zero origin. Raw C-order pixel SHA-256 is
`628b47bd3d23e70f380344e834b6d16c983ea16c34d14e56db75f9551e0ea341`;
NPY-file SHA-256 is
`3d67f1eaa1a22166b3e9733e1af9e5f3f1c0aea4ba95a1fce313421a6662d287`.

The exact original goal, including spelling, is:

> in this z-stack there is a wavy structure and a bright circular structure in the middle. Can you please create complete 3D masks (and ultimately meshes) of these structures independently and combine the labeled structures with different colours together, so that I can see the meshese overlaid in different colors.

Do not open the human workflow, any prior solved recipe, old comparison results,
reference masks, sample generator or its formulas during planning, investigation,
execution, inspection, result selection or locking. Do not import an old
experiment's driver. The model receives the original goal, ordinary product
contracts/catalogue, approved source metadata and previews, and the evidence
produced by its own bounded product actions. The harness supplies no algorithm,
preset nodes, thresholds, expected object geometry or target mask.

Wrap those approved pixels in a new OME-TIFF with explicit ZYX and X/Y/Z = 0.45
micrometer metadata, preserving the original input files. This is a file-format
wrapper, not a sample generator or scientific transform. Verify the wrapper's
raw pixel equality and record its file hash. The starting graph contains only a
normal file Image Source pointed at that wrapper; the ordinary reader must
return the approved axes, spacing and zero origin before any Send. Do not patch
cache metadata. Calibration is input metadata, not a solver hint.

## Gates before a live run

1. Finish offline unit and controller tests, review their changes, and qualify
   the clean packaged source in an isolated copy. Record the package/source
   identity, interpreter, dependency provenance and test results. No installs or
   changes to the shared environment belong to this acceptance run.
2. Receive explicit authorization to launch the UI and make bounded model
   requests. Do not repair or modify authentication, the model profile or user
   configuration as an acceptance workaround. Report a failed gate instead.
3. Use a fresh evidence directory beneath
   `VIPP-local-tests/ai-assistant-investigation-loop-20261016`.
   Keep timing history and presentation settings in this run's local state.
4. Verify both input hashes, dtype, axes, shape and calibration. Record the
   verified starting workflow before opening the assistant.
5. Use the product Goal investigation opt-in with separate input-preview and
   output-evidence permissions. Select Use best judgment without supplying
   additional human answers or segmentation instructions. The product's
   default request timeout stays unchanged. Record all finite budgets.

## Offline acceptance matrix

| Boundary | Required observation |
| --- | --- |
| Passive controls | Opening Goal, changing a limit or viewing sharing details starts no model call or calculation. Investigation defaults off. |
| Consent off | Source inspection cannot expose source pixels without input consent; trial inspection cannot expose result evidence without output consent. Withdrawn consent revokes in-flight result authority. |
| Cancel and close | Cancel reaches the active product session and ordinary executor checkpoints; queued/late model or calculation completion cannot create a current draft or selected result. |
| Freshness | Source-byte mutation, source revision, graph edits, settings/dialogue changes and replaced output buffers reject stale evidence and final Add eligibility. |
| Hard budgets | Model calls, actions, trials, elapsed time, data/image volume and candidate retention remain bounded; repair/failure paths do not reset budgets. Trial dispatch failure still consumes the applicable attempt. |
| Manual completion | The model may ask a genuine blocker or stop without success. No unselected/unfinished result is silently promoted; final Add and final normal Run remain explicit. |
| Roles and comparison | Candidate outputs have explicit structure/role mappings. A comparison cannot infer roles from array order, use a failed/partial run, substitute the latest candidate for the retained one, or call a smoother/smaller mask an established improvement. |
| Idea selection and testing | The compact eligible-node index and bounded multi-node contract lookup reveal capabilities, not a recipe. Experiment plans record a model hypothesis and a disconfirming observation before a linked trial; assessments bind that exact completed candidate and product-issued evidence. Planned, failed, tested and assessed experiments remain distinguishable. |
| Native regional evidence | Explicit native-grid ROIs validate axis names and integer half-open bounds. Counts/diagnostics cover the recorded ROI, retain grid/measurement hashes and never become truth labels for wanted or missing anatomy. Presentation crops, projections or montages cannot replace those numerical populations. |
| Bounded notebook request v2 | A complete immutable session-local public journal remains indexed and hash-bound while each wire request declares a finite working view, not complete inline history. Exact lookup pages and bounded record pins preserve full payloads; changed/cross-session references, missing pages and unsafe payloads fail closed. Local export and speculative request construction are pure and cannot grant sent-evidence authority. |
| Actual supplied evidence | Only the exact request/current PNG board bound by production `before_call` can supply scientific evidence. A record index, page fragment, quotation or pin alone cannot authorize a measurement or unseen image. Format repair consumes its ordinary call/image allowance under the same immutable request generation. |
| Efficient exact decision context | Exact current-test facts and the latest comparison take priority over duplicate baseline content. After a completed or product-rejected executed trial, the top-level accepted entry becomes an exact baseline descriptor with its unchanged full record retrievable; failed/partial trials alone do not trigger it. Optional retained-baseline content is packed last. Current recipe/roles, whole contracts and complete typed numeric payloads remain bounded. Issued references require actual transmission. Changed identities/payloads/private paths fail closed. No scientific choice follows from packing. |
| Preview reuse and closure | Exact cached view identities bind source/candidate/grid/content and display choices. Pinned plus new plus protected previews fit seven PNGs/2 MiB without silent eviction; repeated sends consume cumulative bytes. Closing-phase restrictions reserve pending assessments and explicit Finish inside unchanged action/call/time/context limits; no default nomination or invented assessment is allowed. |
| First-review allocation | With finish review enabled and at least two selected attempts, one original trial and usable action/call room remain held until a successful same-session Continue. Initial dispatch and response-schema restrictions both enforce the hold; closing reads cannot drain it. Verify the first genuine nomination is eligible, the actual Keep improving transition releases only held capacity, and the next trial/assessment/Finish use unchanged session identity, counters, deadline, evidence and protection. Failed/stale/context-denied Continue rolls back; one-trial and small-limit cases disclose unavailable reservation without inventing allowance. Time, image and hard limits still apply. |
| Legacy request-v1 lossless history | Explicit shared references expand back to every original history record, in order, with full experiment plans/assessments and historical preview captions. Dangling, changed or unsupported references fail closed. The complete compact request still fits the unchanged 192 KiB limit; no cap increase, omitted fields or silently discarded evidence may substitute for lossless encoding. |
| Safe graph | Writers, other sources, external/custom code and unsupported operations remain outside delegated trials. Whole-graph and ordinary dimensional/resource checks apply. |

Use synthetic fixtures to test these boundaries. Their recipes and arrays are
test mechanics only and must not enter the blinded live model dialogue.

## Live controller run

The driver invokes normal Goal Send. It observes the product provider and
session boundaries without replacing a model decision, creating a trial recipe
or choosing an analysis parameter. Each discovery, inspection, trial,
comparison, retention and finish action must come from the product's validated
model action response. Every trial uses ordinary VIPP execution under the
session's ownership, freshness, cancellation and resource policy.

This is a fresh full investigation, not a continuation of an earlier nomination.
Start with the original goal unchanged and a new source-only graph. Do not replay
old requests, pass a suggested reconstruction family, or prepopulate an experiment
plan. The product may choose an explicit reconstruction as a testable hypothesis;
inferred structure is not observed signal, and a successful calculation alone
does not confirm that hypothesis. Discovery, planning and assessment consume
their ordinary finite action/model/time budgets, not new allowances.

Capture these records locally, without leaking local paths into model payloads:

- exact metadata request, native prompt/contracts and response/action text;
- provider resolved model, request attempts, logical and actual model-call
  counts, timestamps, elapsed duration and stop reason;
- every image sent: original product PNG bytes, caption/selection manifest,
  SHA-256, byte count, kind and count per request;
- every action and result, candidate ID, role mapping, provenance/run ID,
  source/graph/output hashes, trial failure and comparison/retention reason;
- compact index/batch-contract responses, model-authored experiment plans and
  assessments, issued numerical evidence references and the terminal experiment
  ledger; measurements and model judgments must remain separately identifiable;
- every retained candidate's workflow recipe, metadata states and actual
  full-resolution masks/labels and mesh vertices/faces/object colours;
- final product-selected candidate, final checked draft, explicit Add/Run
  actions, complete final workflow/state and final output hashes;
- a final receipt stating what was and was not achieved without calling model
  judgment or execution a validated scientific result.

The driver records its own source-file hashes, interpreter, unchanged per-request
timeout and initial/terminal session limits. The package qualification inventories
every module and asset: a newly added experiment-ledger module must appear in both
the wheel and isolated package, and any subsequent byte or inventory change fails
reverification. A connection check, valid action, completed trial or model FINISH
is not a completed acceptance run.

The harness may save local evidence arrays/meshes directly; this is not model
filesystem authority or a Save node in the delegated trial. Preserve scientific
buffers and store the unmodified output data with their carried metadata.

## Version-2 bounded working notebook

The unreleased v2 request deliberately does not include complete inline history.
Keep the complete immutable **public** action requests/results, candidate records
and full experiment-ledger revisions in the bounded session-local notebook;
never rewrite them into summaries to claim a lossless wire. The record index
contains exact identities, hashes, owners, byte/page counts and complete retrieval
references. `read_notebook(record_id, sha256, page)` retrieves exact 8 KiB pages;
`pin_notebook` selects at most four complete records with 32 KiB total payload.
The 64 KiB working-payload/96 KiB view bound is separate from the finite complete
notebook (256 records, 512 KiB per record, 16 MiB total). These are developer
testing boundaries, not a larger scientific or provider allowance.

Also save the exact `decision_context` in each outgoing request. Check that it
prioritizes current completed-test facts and the latest exact comparison over
duplicate unchanged baseline content. After a completed or product-rejected
executed post-baseline trial, verify the top-level `accepted_candidate` exact
compact descriptor/full-record reference; failed/partial trials alone must not
trigger it. Keep the original full record unchanged and queryable. Pack optional
retained-baseline recipe/facts last, not ahead of current preservation/mesh facts
and the latest actual comparison/inspection. Current scientific recipe/roles
and whole registered active/recently requested
contracts remain exact context, not measurement authority. Only complete
numerical payloads carry typed numeric citation references; recipe, contract
and record hashes cannot substitute for them. Literal numeric payloads actually
supplied in `decision_context` or notebook working records can grant their
matching references at the ordinary actual-call boundary, not during request
construction. A descriptor alone must not grant authority for its omitted body.
A fresh efficient-run preflight must verify the actual qualified core supplies
these entries without executing a
pipeline or calling a provider. Repeated bookkeeping actions must not invent
or change measurements. Explicit omission records disclose whole entries left
out of the finite context; the full local notebook is never replaced. Numeric
references for protection or delivered mesh roles become usable only when the
exact matching payload crosses the ordinary actual-call boundary.

Each page encodes its exact canonical-ASCII JSON bytes as canonical unpadded
base64url (`encoding='base64url-canonical-json'`, `data_base64url`). Its readable
`text` uses the exact reversible `ascii-tilde-v1-display-only` projection: fixed
display marker `json|`, then `~b` for an original backslash and `~~` for an original
tilde. Strip the marker before reversing escapes; it is not part of the original
bytes or hash. The projection is neither a free-form summary nor byte-identical
text. Verify decoded-byte hashes, Merkle proofs, offsets and counts, require text
to equal the projection, and bound encoded/readable lengths before decoding.
Keep original whole structured-record path guards unchanged. Partial pages
authenticate bytes and index membership, not privacy of unseen record content;
do not invent string boundaries in fragment checks. The marker prevents false
absolute-path boundaries without weakening those original guards. Malformed
encoding, changed readable text or an invalid proof must fail before provider
discovery.

For each actual action, save pure `export_notebook()` snapshots before and after,
including failures, as well as the original action result and candidate caches.
Save the complete terminal export even when there is no FINISH. Store each actual
wire request and PNG bytes/manifests separately; only production `before_call`
with that exact request may record scientific evidence as supplied. Merely
constructing a request, exporting, listing, retrieving or pinning a record must
not mark measurements/images as sent. A rejected or partially serialized action
must not erase already completed trial evidence or promote it to selection.

Check that cached inspections reuse only an exact source/candidate/grid/content
and display-choice match. `pin_previews` can place existing original PNG views on
the current board; it cannot rebind another candidate's image, silently evict a
pin, bypass either permission or avoid repeated-send image accounting. Pinned,
new and protected views share the unchanged seven-image/2 MiB request bound.
Scientific arrays and actual mesh geometry remain unmodified by reuse.
An empty `pin_previews` request clears explicit pins only; eligible exact cached
source/retained/latest previews may still be reused automatically under the
same permissions and limits. This is not an all-images-off action or a new render.

The finite working request targets 128 KiB under the unchanged 192 KiB hard cap.
Record its explicit closing phase and complete experiment index. Context pressure
or low remaining actions/calls/time must restrict new trials/plans/discovery and
reserve room for pending assessments and a genuine Finish. The full internal
ledger remains authoritative; missing visual evidence can remain unresolved,
not fabricated. Closing restrictions grant no extra action, call, image, trial
or wall time and do not choose a candidate. A fresh notebook test has fresh
authorization and a fresh evidence root, not a continuation or budget reset of
a permanently closed prior experiment. A real nomination and exact owned final
Add/Run (or retained accepted baseline) must be locked before new comparisons.

## Fresh matched-assessment recording

The separately sealed `scripts/run_workflow_paired_assessment.py` keeps the
efficient driver's exact goal (including its sole uneven-background sentence),
unchanged feedback, original accepted own baseline and seven starting roles.
It pins that frozen driver and all its old recorder helpers. Normal baseline
Add/Run, exact 41-buffer/full-metadata replay, 2,340 protected foreground voxels,
three trials/32 actions/40 actual calls/1,800 seconds, 600-second requests,
192 KiB wire, seven PNGs/2 MiB board and cumulative image limits are unchanged.
No old trial parameters/results, human workflow, automatic repair, reset or
continuation enter the new dialogue. Root-owned qualification and one explicit
live launch are separate from offline harness checks.

At each successful normal `record_model_call` binding, save an additive receipt
of the exact outgoing request hash, current original-PNG manifests, literal
`comparison_previews` groups and complete typed `mesh-topology-review` rows.
Record whether both declared supplied hashes actually occur on that board;
retain omitted/unavailable reasons rather than claiming an unseen comparison.
Match source/grid/ROI/view/display identity, not captions or array shape alone.
A speculative request/export is not this actual binding, and the existing
transport recorder separately proves whether the native call began. Observer
receipts neither select pairs nor change a request, image, recipe or assessment.

Require offline regressions for pair persistence after a later mesh inspection,
both-consent revocation, exact candidate/role/view binding, unchanged PNG bytes,
finite-board pressure, truthful omissions and actual-send-only evidence
authority. Topology reviews must preserve before/after role-bound counts and
state limitations even when boundary edges are zero; no simple-solid, genus,
self-intersection or anatomy claim follows. Genuine model assessments/Finish,
exact normal final replay or accepted-baseline retention and the immutable
selection lock remain required. This protocol or a passing offline gate does
not establish live completion or improved scientific quality.

## Legacy request-v1 lossless history encodings

The following codecs describe earlier request-v1 recordings, not the v2 bounded
working-view protocol above. Their internal history-encoding version numbers
must not be confused with the outer request version.

When `history_encoding` is present, the model request contains explicit lossless
references to complete records in `experiment_ledger` and
`history_preview_manifests`. The preview table is keyed by each full manifest's
hash. Offline checks must reconstruct the complete original ordered history and
reject altered references, hashes or table records. Save the actual compact wire
request without expanding it in the observer or replacing it with a summary.
Historical captions retain their native view, crop, calibration, display
transforms and limitations; they are descriptions of prior evidence, not newly
shared PNGs. Only the current approved `image_previews` disclose images in a call.

History-encoding version 3 adds an optional lossless outer layer when the
ordinary history-encoding-v2 request would exceed the same 192 KiB limit. Repeated complete JSON
values in `history`, `accepted_candidate`, `history_preview_manifests`, `source`,
`node_index` and `response_schema` may use
`{"$ref":"history_shared_values","id":"vN"}`. The consecutive short ID names
an entry containing exactly `sha256` and `value`; the SHA-256 covers the complete
original value's canonical JSON. Entries contain literal original JSON, not
recursive references. Verify every hash and resolve this outer layer before
the version-2 ledger, observation and historical-preview references. Literal
objects using the reserved reference namespace have an explicit reversible
escape; malformed, changed, dangling, unused or excessively nested values fail
closed. Public root container types, list order, unknown fields, full captions,
measurements and model reasoning are preserved, not summarized or truncated.

The short IDs are representation lookups, not PNG hashes, numerical evidence
handles, candidate identities or execution authority. The original goal and
current `image_previews` remain full and unchanged. Offline tests must reproduce
the complete pre-compaction request and ordered journal exactly, while the
observer saves the actual compact wire form. No cap increase, extra model turn,
new image disclosure, trial or counter/deadline reset follows from this encoding.
If unique content still cannot fit, stop honestly; a passing codec check does
not establish FINISH, a selected result, completed recovery or scientific success.

The observer saves a returned action's result, previews and candidate artifacts
before capturing the next public request. A subsequent serialization/budget
failure is recorded separately and still propagates; it cannot erase a completed
trial or promote it to FINISH, a selected lock, or scientific success. Preserve a
failed run unchanged and use a new evidence directory for any authorized fresh
full investigation after a source fix.

## Selection lock and independent comparison

Close the pre-comparison journal with `selected-result-lock.json` containing the
product-selected candidate ID, completed run ID, explicit role mapping and all
artifact/event hashes. The append-only event chain and immutable file creation
make accidental replacement or post-selection change detectable. Do not choose
the last trial or strongest-looking preview in the harness. Failure to obtain a
completed identified selection is an incomplete acceptance result, not a reason
to invent one.

Before locking, the fresh driver freezes the selected recipe's key-to-node mapping
before the dialog consumes its temporary Run authority. It binds the one normal
Run to its dispatch serial, pipeline serial and exact Add revision, requires a
fresh successful cleanup report, and checks the complete executed graph. Selected
arrays must retain identical shape, dtype/byte order and raw bytes; meshes must
retain geometry, face/object IDs, colours and complete carried metadata. The
driver preserves the product's explicit output roles rather than interpreting
array order or silently substituting a later result.

Only after this lock exists may a separate, explicitly authorized comparison
read independent references. Record that comparison in a separate directory
keyed to the locked selection's SHA-256. It cannot change the selected recipe,
masks, roles, trial parameters or original evidence. Report boundary and
completeness errors, including missed wanted signal and unwanted foreground;
two label IDs, two colours or fewer connected components are not sufficient.

No live or scientific-success claim is implied by the presence of this protocol
or a passing offline harness test. Native UI inspection, packaged-source
qualification and the live blinded experiment require their own dated evidence.

## Replay-only continuation of a recorded selection

If an evidence-driver error interrupts the final normal Run after an explicit
successful FINISH action, preserve that entire run unchanged. The separate
`scripts/replay_workflow_investigation_selection.py` defaults to read-only bundle
validation. An explicitly approved replay requires a fresh output directory
outside the original run, its exact qualified wheel and exact original source
file/path. It never requests another model, reselects a candidate or changes
scientific parameters. Only the fresh proposal context identifier is rebound.

The continuation records all original-file hashes and the FINISH hash, verifies
the complete selected prepared graph before Add and executed graph after Run,
and freezes a nonempty key-to-node mapping before the dialog may consume its
temporary Run permission. Completion must belong to the one accepted dispatch:
the product's every-dispatch serial, pipeline serial and exact Add revision must
remain unchanged. All selected full-resolution mask/mesh buffers, buffer dtypes,
object IDs/colours and complete carried metadata must replay exactly. Interpreter
guards measure and block sample-generator, inference, native RPC and auth entry.

Only after verification does the new journal lock the original nomination and
its replay; all original files are rehashed before and after. Optional native
inspection hold is limited to 120 seconds, defaults to zero, and begins only
after the lock. The partial scientific interpretation and limitations remain
unchanged; exact replay is not proof of scientific completeness. No references
or comparison results are read by this continuation.

## One remaining recorded finalization action

This is a separately approved developer recovery harness, not automatic native
GUI recovery or a general resumable-investigation feature.

An explicitly approved serialization-only recovery can use
`scripts/finalize_workflow_investigation.py` when the original journal contains
completed action receipts but no FINISH. Its default is read-only verification.
It requires the independently frozen original file-inventory SHA-256, verifies
the event chain, exact source/candidate/role/context identities and original
qualification, and reconstructs every original returned history record with the
current lossless codec. It does not replay trials or supply a preferred candidate.

The remaining authority is narrower, not renewed: one actual model turn without
format repair, one remaining action, no new images or scientific calculations,
and only a validated FINISH or genuine question. Cumulative action/model/image
budgets remain unchanged. The original earliest journal timestamp plus its
original time limit is the conservative wall deadline; an expired deadline stops
the recovery rather than granting a fresh allowance. Every completed candidate's
linked experiment must already be assessed before FINISH can nominate one.

The new provider/codec qualification and the exact original execution
qualification remain separate. A successful FINISH is saved only in a fresh
sibling evidence directory. The replay driver's `--finalization` adapter verifies
that nomination and both immutable inventories, retains the original source path
and scientific artifact paths, then uses the original qualified execution
package in a separate process for the ordinary checked Add/Run. It does not copy
or fabricate a FINISH into the original tree. Only exact owned replay can create
the fresh selected-result lock; a question or failed/expired finalization cannot.

## Feedback-refinement recorder contract

An explicitly authorized refinement of the product model's own prior result is
not a fresh blinded reconstruction. Pin that prior recipe, input, evidence
inventory and selection/limited-recording lock before replay. A prior incomplete
recording remains incomplete: a new normal baseline Add/Run may establish fresh
complete metadata, but cannot retroactively prove missing old states, tables,
mesh-object records or execution cleanup. Compare all persisted old scientific
buffers exactly before making any new model request. Only administrative context
and owned node identifiers may be rebound; scientific parameters and wiring stay
unchanged.

Readiness distinguishes computation from presentation using the exact qualified
operation catalogue. A display-only operation with no scientific output may be
recorded as a presentation exception; the current positive allowlist is only
`review_images`, and its catalogue entry must also confirm `presentation_only`
and `output_type="none"` with no scientific cached value. It must not keep an
otherwise completed
ordinary Run waiting for a nonexistent cached result. Unknown operations,
scientific outputs, errors and blocked nodes are not presentation exceptions.
The owned node set must be nonempty. Require every scientific node's ready and
completed state, no active or pending calculation, the exact owned dispatch and
Add revision, and a fresh successful execution report/cleanup. Persist this
report and the actual presentation exceptions before terminal model assessment.
Record final assessment adoption by the current UI owner separately from receipt
of a valid provider response.

Undefined scientific measurements are not zero and are not an exporter failure
that may erase already produced results. The local evidence writer uses an
explicit versioned, reversible encoding for nonfinite floating-point table/state
values, including NaN and signed infinities, and escapes colliding reserved
mapping keys. Preserve scientific NPY buffers unchanged. Finite legacy JSON
documents retain their existing encoding; provider prompts and responses retain
their strict-JSON contracts. Local evidence-state consumers use the explicit
`read_recording_json` / `decode_recording_document` boundary; raw provider JSON
readers must not interpret these evidence envelopes. Tests must cover round-trip nonfinite values,
reserved-key collisions, unchanged arrays, full mesh-object/state recording and
an undefined mesh-quality table in an otherwise completed run.

Lock the new model-authored result before opening comparison methods. Human and
earlier independent-assistant workflows may be read only for a separately
authorized post-lock comparison; neither is automatically anatomical ground
truth. Reuse an old result only after exact source, grid, method and output-role
verification. A reference rerun, if necessary and authorized, is a separate
post-lock calculation and cannot feed another revision of the locked model
result. Record differences in border policy or calibration rather than silently
recapping, resampling or tuning a reference to make the comparison look better.

## Explicit protected-region refinement continuation

A separately authorized testing continuation may clarify the objective while
retaining the literal original goal for provenance. Record the changed wording;
do not claim an unchanged-prompt benchmark or assign ambiguity-related failures
to model capability. Outcome clarification must not supply a reference recipe,
scientific parameters, sample generator or hidden anatomical target.

An accepted foreground constraint comes only from explicitly selected current
owned mask/label outputs, optionally limited by another aligned mask/label
foreground. Record the selection, exact source/grid/output hashes, semantics
and protected count. It is an accepted inference, not an accuracy reference.
The revised model proposal must explicitly nominate its protected structure;
an unused preserved copy is not evidence that the delivered structure retained
it. Check all protected foreground after the ordinary owned Run and before
accepting its refinement assessment. Missing foreground, changed binding/grid
or stale output stops the continuation, not silent OR repair or resampling.
Keep earlier branches/evidence without reviving historical current-run authority.

Protected previews remain within existing consent, image/byte and request limits,
with displaced views disclosed. Final meshes feeding explicitly known
measurement-only consumers remain assessment deliverables, alongside their
actual creator-mask lineage and available measurement tables. A scientific
table cannot silently hide the mesh's topology or geometry views. Freeze and
record the actual accepted/rejected product outcome before reference comparison;
never promote a failed protected revision to a successful selected result.
