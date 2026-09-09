# Verified interrupted-batch resume

Implemented for the post-0.15.0a2 development line; issue [#58](https://github.com/rensutheart/napari-vipp/issues/58).
Public instructions live in the manual's batch-processing tutorial.

## Contract and ownership

`core/batch_resume.py` owns recovery validation and the OS-held output-directory
lock. `core/batch.py` owns execution, publication and durable receipts. The same
contract is exposed by `run_batch(..., resume_manifest_path=...)`,
`run_batch_resume_from_manifest(...)`, the generated batch runner's `--resume`,
and a detached GUI worker. The UI never interprets file presence as completion.

New batch manifest schema **6** adds a recovery snapshot and document checksum.
Recovery version 1 retains the complete workflow, config resolution base,
effective intent, all fixed/collection input content identities, scientific
source-code fingerprint and installed dependency versions. Completed outputs
carry full content identities; sidecars are checksummed and bound to their run.
Earlier manifests remain historical and cannot provide verified resume.

## Restart and refusal boundaries

1. Read the selected manifest and reconcile its run-bound atomic item sidecars.
   A crash may leave the main manifest older than its completed sidecars.
2. Acquire the same destination lock used by ordinary batch runs. OS process
   termination releases it; the presence of a lock file alone is not ownership.
3. Reconstruct the archive's exact request, not mutable workflow/config files.
   Check schema/checksums, effective per-item workflow and overrides, source
   bindings/selectors/content, compute request, code/dependencies/runtime/device,
   and output declarations/destinations.
4. Reuse only **whole completed items** with successful execution and cleanup
   evidence, matching provenance, and independently verified output contents.
   Recheck reused inputs and outputs immediately before accepting their receipt.
5. Start a new continuation run with `resumed_from_run_id` lineage; keep the
   original archive and sidecars. Reused items stay completed, are explicitly
   identified, and do not enter `saved_paths` (new writes only).

Changed or missing completed output bytes cause refusal, even when file size
and modification time still match. Partial/skipped/failed work is not proof of
completion. Unfinished items with any unverified existing destination require
manual review; resume never inherits Skip/Overwrite permission for them. Files
appearing later are guarded by atomic no-replace publication. Recovery does not
delete private staging files or claim sentinels left by a crashed process.

Checksums establish integrity, not authorship: use trusted local manifests.
Run IDs, sidecar directories and destination paths are constrained; links or
redirects must not bypass destination/source collision checks. This first
implementation is local-machine, original-location and conservative about
software changes. It does not promise distributed locking, moved-run rebasing,
partial-item recovery, or compatibility across upgraded environments.

## Presentation and checks

**Resume saved run…** is an explicit Run & results action. Its confirmation
states that the saved workflow/settings are used and the open workflow is
unchanged. Verification is cancellable background work. Archived results do
not navigate into unrelated items in the open workflow. The report distinguishes
verified reused items/files from newly written outputs.

Regression owners: `test_batch_resume.py`, `test_batch_resume_restart.py`, and
`test_batch_resume_ui.py`; existing batch/publication, UI worker, and generated
runner suites remain required. The restart test terminates a process with
`os._exit` after a completed sidecar, then resumes through a fresh generated
runner process, checking lock release, no recalculation of the completed item,
and preserved original evidence. Broader release qualification is separate.

### Implementation validation (2026-09-08)

- Dedicated resume/UI/fresh-process suite: 36 passed; the opt-in real-CUDA
  smoke was separately exercised successfully on an RTX 5090.
- Existing batch suite: 82 passed; compute/override/policy/mesh compatibility
  suite: 158 passed. The synthetic batch validator's 12 tests also pass with
  the sealed, run-bound sidecar format.
- Broad UI/schema check: 396 passed initially. Four synthetic-demo failures
  were corrected and retested. All 18 remaining layout failures reproduced
  identically on clean `main` at `7d49406`; they are not resume regressions.
- Ruff and diff checks pass. Public-manual checks (12 content contracts,
  50 application routes) and the strict MkDocs build pass.

Targeted qualification is not a new release or a full-suite pass. The changes
remain unreleased, with existing desktop layout qualification still outstanding.
