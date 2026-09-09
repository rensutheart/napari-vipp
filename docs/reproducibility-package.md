# Reproducibility package contract

Status: implemented, unreleased after 0.15.0a2.

This is documentation export, not scientific execution, recovery or a data
archive. Public steps live in the companion manual's
[package-export task](https://rensutheart.github.io/vipp-mkdocs/nightly/how-to/export-reproducibility-package/).

## Evidence ownership

`core/reproducibility.py` builds a `ReproducibilityPackage` with detached report
data and immutable member bytes. Recipe mode receives a detached workflow;
recorded-run mode selects the archived manifest's workflow/configuration and
validated adjacent JSON evidence. It never substitutes a later edited graph.
Only bounded selected run records are read, not source/result images or output
tables. Stored input/output identities are reported, not recomputed or asserted
to match current files.

The report distinguishes recipe intent from recorded item outcomes and export
environment versions from available run versions. Failure, cancellation,
partial publication and skipped outputs remain explicit. Missing/incompatible
evidence fails preparation rather than being repaired or fabricated.

## Privacy and portable copies

`core/reproducibility_privacy.py` owns removal of original folder locations and
optional filename anonymisation. It records omissions and substitutions while
preserving supported scientific values or rejecting an unsafe transformation.
Portable `workflow.json` retains freeform explanatory notes from the selected
workflow, applying the same location redaction and optional filename
anonymisation as other shared text. Recorded-run notes come only from the
archived workflow snapshot when available, never from a later edited graph.
These authored explanations are distinct from generated analysis data: raw
images, cached/intermediate results, output data, meshes, tables, thumbnails,
presentation previews and unknown embedded metadata remain excluded. Retained
scientific names and free text may still identify people or samples; authors
must review the notes before sharing. Automatic sanitization is not an anonymity
claim.

Portable workflows/runners require explicit source relinking and new output
destinations. Batch copies use recorded effective compute intent and receive new
portable-workflow/configuration digests. Sanitized evidence has distinct view
schemas and identifies original digest references without retaining original
seals. These views must never be accepted as original recovery receipts.

For a recorded batch or recipe with valid attached batch settings,
`workflow.json` embeds the same portable `batch_config` document that is
provided separately as `batch-config.json`. This reuses the ordinary
workflow-open/Batch Setup restoration path; the user should not need
to import a separate config, edit JSON, or repair workflow/configuration hashes.
The attachment is not part of scientific graph identity. Restoration must not
calculate a representative, run the batch, or treat sanitized source records
as fresh verification of local data. The intended collection route is explicit
input-folder selection per source, a new output folder, Check batch, review,
then Run. Python runners remain an advanced headless alternative.

## Reproduce or reuse

An explicit recorded-run export may embed a typed `batch_config.reproduction`
reference. It contains original input content/selection/calibration identities
and the recorded VIPP version, not image data or original folder paths. Recipe
exports and ordinary new-data saves do not create a reference. A recorded run
with insufficient input evidence remains exportable as documentation, but must
not claim that original-input verification is available.

Opening a reference-bearing workflow prompts before tab mutation, source
discovery or graph execution. Reproduce requires comparison with the reference;
new-data use removes the comparison requirement. Cancellation leaves the current
workspace intact. A different or unknown VIPP version requires explicit consent
for reproduction, bound to the recorded/current version pair. A prior saved
acknowledgement is not silently accepted on reopening. Links are generated from
the version and opened only on an explicit click; no software is fetched or run.

Core preflight owns the comparison and supplies all item findings plus an
aggregate result to the batch UI. Changed, missing, unexpected, ambiguous or
unverifiable inputs cannot authorize a reproduction run. GUI controls are not
the safety boundary: headless execution and plan handoff must enforce the same
rule, retaining ordinary input-revision checks before publication. Moving a
folder does not change identity; changing a logical image selection does.
Version overrides remain visible and recorded; they never disable input checks.
The desktop Run handoff must carry `fresh_preview.reproduction` into its
`BatchPlan`, alongside the exact config and items. It must not omit the check
or substitute the earlier displayed check after a folder/settings change.
Worker preparation requires that record; execution independently rebuilds the
input comparison before processing and retains publication-time revision checks.
Matching inputs and VIPP versions do not prove matching outputs, dependencies,
hardware, or scientific validity. Verified batch resume remains a separate
operation with its existing original-receipt requirements.

The executable portable config omits frozen source inventories: redacted
locations and metadata are not current verification evidence. Check rediscovers
the collection and validates source contents, logical selectors, axis
declarations and source-bound numeric overrides through the ordinary batch
planner. The sanitized workflow/evidence retains available selection and
calibration context. Path-bound per-item overwrite permissions are also omitted;
the original run cannot authorize replacing a recipient's files. Whole-batch
and output policies remain visible settings for explicit review.

Fixed unbound reference inputs remain fixed and require individual source
relinking; export must not convert them into collection bindings or invent
replacement logical selectors. Attachment restoration validates structure while
allowing missing fixed-reference locations, so the Batch Setup survives until
the user selects those references in their Image Source inspectors. This
exception applies only to attachment restoration: normal Check and Run retain
strict source existence, content and selection validation. Live-layer/sample
inputs still require their separately supplied data. Any other failed source
or attachment restoration remains an explicit limitation, not permission to
execute an incomplete recipe.

Environment JSON and README guidance describe available versions and relinking.
The report distinguishes the recorded run's VIPP version from the exporter and
offers official version-specific release/setup links when the recorded version
supports them. A development/local version is not silently mapped to a matching
public installer or claimed to identify an unmodified release. No installer or
environment is bundled, downloaded, or executed by package preparation/export;
the version record is not a lockfile or a guarantee of identical results.
`SHA256SUMS.json` hashes each other packaged member's exact bytes. It does not
authenticate an author or prove scientific validity.

## Review and publication boundary

The batch results toolbar owns recovery and file-location/status actions only.
The inline run report card exposes **Export reproducibility package…** instead
of the former manifest-locator action, and the completed **Run & results**
footer exposes the same entry point as **Export package…**. **Items & outputs**
and **Overrides** retain **View run report** navigation; **Setup** retains
**Check batch** for a new run. Both export entry points open the review dialog
for the recorded manifest; they never bypass review or directly publish a ZIP.

`core/reproducibility_report.py` renders escaped self-contained HTML and README
text. Repeat guidance leads with the VIPP GUI and includes a relative
`workflow.json` link usable beside the extracted report. Optional absent
messages do not add empty explanation markers to outcome summaries; actual
failure/cancellation evidence remains visible. `ui/reproducibility.py` uses the
actual prepared `report.html` bytes for its preview. Its text browser blocks
all resource loads. Only explicit clicks on generated official release/setup
links may open an external browser; arbitrary evidence URLs are not permitted.
Clicking the relative workflow link in the preview explains export, extraction
and Open workflow rather than writing a file or executing a recipe. Extracted
HTML keeps the ordinary relative link. Preparation/export perform no uploads.
The file-inventory tab exposes each included member for review, alongside
privacy/omission details.

The dialog accepts one workflow snapshot or manifest path. Application-owned
background jobs retain no widgets. Generation tokens reject stale or queued
preparation results; changing title, notes or filename policy clears the preview
and sharing acknowledgment. Closing during preparation discards completion
without joining the worker. Export requires current prepared evidence plus the
explicit review checkbox and uses exactly those immutable members.

The UI refuses existing destinations and holds close while writing completes.
The core writer validates archive members and destination links, creates a
same-directory temporary ZIP, flushes it, and atomically promotes it with
race-safe no-overwrite semantics by default. Publication never rebuilds the
report or rereads the evidence. Core-only explicit overwrite replaces a regular
file atomically; this is not offered silently by the dialog.

## Focused evidence

`test_reproducibility_ui.py` covers review authorization, detached workflow
ownership, off-GUI-thread work, invalidation and queued stale results, close and
native deletion during preparation, resource blocking, export cancellation,
existing-file refusal and exact reviewed-snapshot publication. Core tests own
record validation, sanitization, portable-copy correctness and archive writing;
UI success is not independent scientific validation.
