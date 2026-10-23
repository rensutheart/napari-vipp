# 0.16.0a7 startup, recovery and path-saving patch

The user authorized this incremental alpha patch on 2026-10-23. It contains
only the source-loading/recovery fixes and workflow path-saving preference
developed after public `v0.16.0a6` at
`543b0704b8512054fb306a5a4346a825ab2be0c0`. Unfinished assistant, webhook bot,
agent and MCP work stays on its separate branches and is not released.

Follow the [release runbook](release-runbook.md) and
[qualification declaration](release-qualification-baseline.md#0160a7-qualification-declaration).
Use one application PR and one companion manual PR, preserving their histories.
Long-running CI and installer builds receive occasional scheduled follow-ups,
not repeated polling, duplicated suites or duplicated transfers.

## Included behavior

- Load the starter sample independently of interrupted working copies; allow
  Calculate all to calculate an untouched, uncalculated automatic graph.
- Recover graph copies as unsaved tabs and rebuild automatic results from
  verified sources. Keep manual steps (even authored Auto Recalculate on),
  writers and their descendants paused until explicit calculation. Retain that
  boundary across asynchronous reads, background continuations, cache policies,
  tab switches and bypass what-if preview handling.
- Preserve saved graph selection and show a source when an ordinary newly
  opened workflow has an unavailable manual result selected.
- Add persistent Settings → Workflow paths: Relative to workflow file
  (existing default) or Absolute file locations. Apply only on workflow Save,
  Save As or timestamped Save, including a valid attached Batch configuration.
  Preserve original input/output targets, source revision evidence, checksums,
  selectors, atomic writes and overwrite confirmation. Do not move/copy data or
  change separate Python-export and Batch Save config behavior.

## Changed-domain checks and limits

Core/UI, source-load/recovery orchestration, workflow persistence and
documentation change. Focused regressions cover actual Gaussian/Otsu CPU
reference results, source revisions/axes/RGB, all cache modes, asynchronous
ownership, manual/writer/bypass safety, relative/absolute save-open round trips,
raw original-base Save As, Windows roots/shares, and attached Batch hashes.
Require the affected representative batch path and a manual smoke of recovery
and the path setting before publication; automated source checks alone do not
establish native installed behavior. Existing owned synthetic fixtures are
used; original workflows, open sessions and installed environments stay intact.

The saved working-copy store, lock/retention and crash-checkpoint code is
unchanged; carry only those unchanged storage/crash behaviors from the prior
baseline. Automatic post-recovery calculation is new and requires its own
guarded recovery check. Recovery never restores scientific arrays or Batch
workspace/configuration. Missing or changed sources remain explicit errors.

Schema numbers, scientific kernels and image interpretation, core execution
engine, GPU provider algorithms/catalogue/dispatch, dependency/build pins,
installer engines, entry points and publication workflows remain unchanged.
Do not rerun a full GPU catalogue or installer lifecycle solely for this patch;
carry their qualified unchanged boundaries from a6/a5. This does not make a
new full frozen-GUI, uninstall, fresh-user or scientific-CUDA qualification.

## Release gates

Require the existing full supported-platform candidate and exact-final-main CI
matrix. Retain its exact wheel/canonical-source artifacts, recorded sizes and
SHA-256 hashes, metadata/resources/entry points and assistant/MCP exclusions.
CI already checks independent clean-source reproducibility; do not duplicate
matching builds. After all changed-domain gates pass, tag clean final main
immutably and build the selected exact-tag unsigned Windows x86_64 and separate
CPU-only unsigned/unnotarized macOS arm64/x86_64 artifacts once. Verify exact
embedded/input wheel identity, version, architecture, unsigned state and
checksums. Carry old behavioral facts only, never old artifact byte facts.

Audit one draft prerelease before publishing original qualified assets. Publish
the same Python bytes to PyPI, verify actual public hashes/sizes, then deploy
and verify the numbered/default companion manual once. Never move a published
tag or overwrite public bytes. An immutable-source failure requires a new
version decision. This plan is not a claim of completed CI or publication.
