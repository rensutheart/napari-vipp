# 0.16.0a10 recent workflows, reader recovery and Crop display patch

This non-AI iterative alpha covers the full delta since public `v0.16.0a9`
at `268087906df7c9f8d17fca7b2fdc726b021d3025`, not just its version bump.
Follow the [release runbook](release-runbook.md).

## Selected changes and gates

- **Open Recent:** full-path, bounded, persistent history of successful opens
  and saves; normal new-tab opening safeguards; no filesystem probe on hover;
  clearing history never deletes files. Qualify persistence, safe callbacks,
  save/reopen and excluded failures/examples, plus a native hover/open smoke.
- **Reader recovery:** classify reader-version/implementation mismatch without
  accepting a changed binding. Require explicit Refresh file sources. Check
  synchronous/worker inspection, error details, stale tabs, superseded work and
  batch representative rollback; native warning/refresh on a synthetic TIFF.
  Schema and source-verification equality rules do not change.
- **Crop presentation:** guarded Windows Intel/unknown-renderer route uses
  ordinary Inspect input and a passive raster overlay, not another GL volume
  or Shapes layer. Qualify 2D/slice/slab/3D projection, calibrated bounds,
  drafts-versus-commits, source/result immutability, layer ordering, lifecycle
  and cold results. Add the changed helpers to the existing minimum/pinned
  napari compatibility jobs. Native smoke uses only isolated synthetic data;
  exercise this host's natural route and a per-process forced guarded route.
- **Internal planning:** the Apple GPU proposal is planning only, with no
  implementation, dependency change, supported-platform expansion or new
  hardware qualification claim.

All AI work, including Allen/AI planning lineage, is excluded and preserved
for 0.17. Historical uncommitted experiments and dirty worktrees are not
silently assimilated. Old installer/recovery branch tips already ancestral
to a9 need no duplicate merge.

## Carry-forward and publication

Scientific kernels, GPU providers/dispatch/admission, workflow/batch/manifest
schemas, serializers, runtime pins, installer engines, entry points and asset
composition are unchanged. Carry a9/a8 behavioral evidence only within its
recorded boundaries; exact artifact, source, embedded-version, signature-state
and public-byte facts are regenerated. The CI change extends coverage only.

Use one app PR and one manual PR, ordinary history-preserving merges, candidate
and exact final-main CI. Tag the exact clean passing main immutably. Retain the
exact-main reproducible Python pair, build unsigned Windows and separate CPU-only
unnotarized macOS architecture installers once from that tag, audit originals,
publish one audited GitHub draft prerelease, verify public bytes, publish the
same pair to PyPI, then deploy and verify the numbered/default manual once.

Preserve existing user sessions, installations, data and worktrees. Record
SHAs, run/process IDs, receipts and changed-domain qualification in local release
evidence. No completed native or lifecycle claim follows from this plan alone.
