# Release Qualification Baseline

Last reviewed: 2026-08-15

This page records expensive behavioral evidence that future releases may carry
forward. It prevents a version-only release change from triggering unrelated
installer, GPU, schema, or manual-UI qualification.

The baseline is evidence, not a claim that untested platforms or scenarios are
supported. A release may carry an entry forward only while none of its listed
invalidators changed.

## Starting Baseline: `v0.13.0a7`

The first ledger snapshot is a7. Each domain can later advance independently;
the qualification tag/commit is therefore recorded on every row rather than
assuming one global version forever.

- Commit: `dc8a63912110a75ab1daad0e7f81c2b20e5001e6`
- Exact-main CI: 13 jobs passed across Windows, Linux, macOS, CPython 3.12,
  and CPython 3.13.
- Source suite: 5,084 passed, 5 skipped, 2 expected failures.
- GitHub release: <https://github.com/rensutheart/napari-vipp/releases/tag/v0.13.0a7>
- PyPI release: <https://pypi.org/project/napari-vipp/0.13.0a7/>
- Published bundle canary: 31872248297 passed.

## Domain Records

| Domain | Qualified at | Baseline evidence | Carry forward when | Invalidate when |
| --- | --- | --- | --- | --- |
| Core/UI | `v0.13.0a7` / `dc8a639` / 2026-08-15 | Exact-main CI and focused widget regressions covered the a7 graph, inspector, GPU-tip, optimizer-result, and example-workflow changes; the operator accepted the final changed UI path. | The specific UI components, worker/ownership behavior, examples, and interaction contracts being relied on are unchanged. | Affected UI, interaction, worker, or example behavior changes. Smoke only the changed journey. |
| Windows installer | `v0.13.0a7` / `dc8a639` / 2026-08-15 | The exact hash-locked a7 wheel/finalized EXE exercised production-backend CPU and CUDA new-install, same-version repair, health/scientific smoke, ownership, shortcuts/registration, and uninstall. The operator separately attested that the final visible setup and installed UI passed. | `src/napari_vipp/installer/**`, `packaging/windows/**`, installer packaging/signing scripts, installer dependency resolution, PyInstaller/Python support, ownership layout, and Windows support assumptions are unchanged. | Any install/update/repair/rollback/network/path/ownership behavior changes; installer runtime/build pins change; signing policy changes; a relevant canary or regression fails. Version and embedded app-feature changes alone do not invalidate lifecycle behavior. |
| GPU/scientific catalogue | `v0.13.0a7` / `dc8a639` / 2026-08-15 | Full RTX 5090 admission passed all 23 evidence owners for all 18 public implementations. Aggregate SHA-256: `3ad655f7d3e36055449bda3e8bb41c914e010fd7607ced26763e23045dcee7ae`. Installed CUDA workflows also passed segmentation and 3D RL/RL-TV checks. | GPU provider implementations, scientific contracts, compute policy/specs, dispatch/fallback/cleanup/provenance/memory code, admission catalogue, NumPy/SciPy/scikit-image/CuPy/cuCIM/CUDA pins, and supported hardware assumptions are unchanged. | Any listed input changes or a real-GPU canary fails. Qualify affected providers only unless a shared layer or the catalogue changed. |
| Workflow/schema/provenance | `v0.13.0a7` / `dc8a639` / 2026-08-15 | CI and the release suite covered workflow schema 4, batch/config/manifest schema 3, generated/export paths, examples, migration, and provenance contracts current in a7. | Schema versions, serializers, migrations, graph/source identity, batch planning, manifest/checkpoint, and generated-runner contracts are unchanged. | Any schema, migration, source identity, serializer, batch, provenance, or generated-runner contract changes. |
| Packaging | `v0.13.0a7` / `dc8a639` / 2026-08-15 | Exact a7 wheel and source archive passed metadata, clean-install, resource, entry-point, manifest, and hash checks on every supported OS/Python lane. The optional no-wheel cuCIM ZIP reproduced byte-for-byte in the hosted canary. | Build backend, package-data/resource layout, entry points, publication workflows, dependency declarations, and optional asset composition are unchanged. | Any of those inputs changes or the hosted bundle canary fails. Exact tag/version/hash/public-URL facts are still regenerated for every release. |
| Documentation | `v0.13.0a7` / `532ba128` / 2026-08-15 | Strict build passed and the numbered plus stable a7 manuals were deployed and checked. | Site configuration, navigation, installation/upgrade behavior, UI shown in screenshots, and changed feature documentation are unchanged. | Relevant documentation or pictured behavior changes. Inspect only changed pages. |

## Explicitly Unqualified At This Baseline

The a7 release did not establish every planned field target. In particular,
downloaded-installer SmartScreen execution, fresh-account/path matrices,
network/cancellation rollback, an older-release update path, RTX 40-series
evidence, native-Linux CUDA evidence, and a timed novice pilot remained outside
the completed baseline. These are roadmap or production-maturity targets. They
do not become blockers for an unrelated iterative alpha unless that alpha makes
one of those claims.

## Updating One Domain

When a release invalidates a domain, append or replace only that domain's row
after its focused qualification passes. Record:

- release/tag and commit;
- relevant environment or hardware;
- focused procedure and result;
- retained artifact/report identifier; and
- the new invalidation boundary.

Do not rewrite unaffected rows merely to replace the release version.

## Expected `0.14.0a1` Selection

For the planned first SourceItem slice, the expected changed domains are
`core_ui`, `workflow_schema`, and `documentation`. Its release checks should
cover SourceItem migration, save/reopen, batch naming, generated execution,
and provenance compatibility.

Unless that implementation also changes their listed inputs, carry the a7
Windows-installer and GPU/scientific-catalogue baselines forward. Do not run a
full installer lifecycle or full GPU admission merely because the package
version becomes `0.14.0a1`.
