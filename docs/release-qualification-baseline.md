# Release Qualification Baseline

Last reviewed: 2026-08-27

> **0.14.0a1 app release record (2026-08-26):** The immutable app tag, GitHub
> prerelease, PyPI packages, Windows installer, and changed-domain gates are
> complete. Companion documentation source is merged and strict-build clean,
> but public deployment is still blocked; `/0.14.0a1/` is not online and
> `stable` still selects `0.13.0a9`. The a8 and a7 records below remain
> historical carry-forward evidence.

This page records expensive behavioral evidence that future releases may carry
forward. It prevents a version-only release change from triggering unrelated
installer, GPU, schema, or manual-UI qualification.

The baseline is evidence, not a claim that untested platforms or scenarios are
supported. A release may carry an entry forward only while none of its listed
invalidators changed.

## `v0.14.0a1` App Release Qualification

- Immutable annotated tag and release commit:
  `e16ca87161ec7b1041a5e98c5a2bf786b11a1ec8`.
- [Exact-main CI run 32975633514](https://github.com/rensutheart/napari-vipp/actions/runs/32975633514):
  all 13 jobs passed at that exact commit.
- [Exact-main Windows-installer smoke run 32975659650](https://github.com/rensutheart/napari-vipp/actions/runs/32975659650):
  passed at that exact commit.
- [GitHub prerelease](https://github.com/rensutheart/napari-vipp/releases/tag/v0.14.0a1)
  and [PyPI 0.14.0a1](https://pypi.org/project/napari-vipp/0.14.0a1/):
  published and verified.
- [PyPI publication run 32982120913](https://github.com/rensutheart/napari-vipp/actions/runs/32982120913):
  passed; downloaded public wheel and source-archive bytes matched the qualified
  exact-tag artifacts.

Changed since `v0.13.0a9`:

```yaml
core_ui: true
workflow_schema_provenance: true
gpu_scientific_shared_axes_metadata: true
windows_installer: true
dependencies_toolchain: false
packaging_release_metadata: true
documentation: true
```

### Focused changed-domain evidence

Release PR [#47](https://github.com/rensutheart/napari-vipp/pull/47) recorded
393 focused source/schema/UI tests, a native RTX 5090 ZYX
Gaussian/Otsu/Remove-Small-Objects corridor with exact GPU assignments and no
skips, and 164 installer frontend/backend/engine/GUI tests with one expected
account-level symlink-capability skip. Its CI and Windows installer smoke passed
before merge.

The strict hash-pinned public corpus v4 gate passed all 10 vendor cases. Its
[fixture manifest](validation/public-data/fixtures/sourceitem-contract-fixtures-v1.json)
records SHA-256
`3365b4cec7a220f6399f3c030eb3ac751581bde730fac60f01c9fca3b823714d`;
the IMS case used portable Temurin 21. This qualifies the changed SourceItem,
reader, schema/migration, preview, per-sample override, and shared-axis/metadata
boundaries. The unchanged full GPU catalogue, installer transactional
lifecycle, and dependencies/toolchain evidence carry forward from `v0.13.0a8`.

### Published artifacts

The GitHub prerelease contains exactly these six qualified assets:

| Asset | SHA-256 |
| --- | --- |
| `napari_vipp-0.14.0a1-py3-none-any.whl` | `ebc46ac87d90b1b5a82821920b13f75e75f306b3301882745969e7081236be70` |
| `napari_vipp-0.14.0a1.tar.gz` | `bad176628678d00b6cd58965c29022a55c6c7d40f414a259f2aa6e5bb70fa53d` |
| `VIPP-Setup-0.14.0a1-Windows-x86_64-UNSIGNED.exe` | `24ca7e8a0ea80bb21f849a30a5946e83e9e55be785047a5164305fab84d9aeea` |
| `VIPP-Setup-0.14.0a1-Windows-x86_64-UNSIGNED-release.json` | `066726da2b5f6a0ec130f09d005f696abdca5d5c72ddbbf211b6aae71713ca66` |
| `VIPP-Setup-0.14.0a1-Windows-x86_64-UNSIGNED-THIRD-PARTY-NOTICES.txt` | `95ca668d0977be347ab39af820f97652ed3565ed7ea4218205bb3a716a847e2d` |
| `SHA256SUMS-Windows-0.14.0a1.txt` | `bf2ad89fb5e78b7f068d7ce7ee694260304535f85fe8a439d4e7fd8bd3e50032` |

The independent CPython 3.12.10 exact-tag A/B package builds matched, and the
PyPI wheel and source archive match the first two rows byte for byte.

### Companion documentation status

Companion documentation PR
[`vipp-mkdocs#18`](https://github.com/rensutheart/vipp-mkdocs/pull/18) merged as
`4f213f27fdabbe7ee26d66e3d34f677d31320302`; its build check
[32970058466](https://github.com/rensutheart/vipp-mkdocs/actions/runs/32970058466)
and a pristine local strict build passed. Public deployment did not complete:
run [32984340385](https://github.com/rensutheart/vipp-mkdocs/actions/runs/32984340385)
ended in a zero-job startup failure, and the single replacement
[32985831392](https://github.com/rensutheart/vipp-mkdocs/actions/runs/32985831392)
failed before any deployment step acquired a hosted runner. As verified on
2026-08-27, `/0.14.0a1/` returns 404 and `versions.json` still aliases `stable`
to `0.13.0a9`. This is a documentation-hosting blocker, not an app artifact or
scientific-workflow failure. Issue
[#42](https://github.com/rensutheart/napari-vipp/issues/42) remains open pending
release closure.

## Current Domain Records: `v0.14.0a1`

| Domain | Qualified at | Baseline evidence | Carry forward when | Invalidate when |
| --- | --- | --- | --- | --- |
| Core/UI | `v0.14.0a1` / `e16ca87` / 2026-08-26 | PR #47 focused source/schema/UI tests, strict public corpus v4, and exact-main CI covered SourceItems, readers, OME-Zarr preview, batch overrides, restore/activity/overwrite UI, and changed graph journeys. | The specific source, UI, worker/ownership, example, and interaction contracts being relied on are unchanged. | Affected source, UI, interaction, worker, or example behavior changes. Smoke only the changed journey. |
| Windows installer | focused presentation at `v0.14.0a1` / `e16ca87`; transactional lifecycle at `v0.13.0a8` / `5a66ae9` | 164 focused tests plus exact-main installer smoke qualified capacity estimates/minima, phase/activity/heartbeat, and log presentation. The unchanged transactional install/update/rollback lifecycle carries forward from a8. | Presentation code and the carried transactional inputs remain unchanged. | Any relevant capacity/progress/log UI change, or install/update/repair/rollback/uninstall/network/path/ownership/dependency/signing change. |
| GPU/scientific catalogue | shared axes/metadata at `v0.14.0a1` / `e16ca87`; full catalogue at `v0.13.0a8` / `7189cf4` | The focused RTX 5090 ZYX corridor passed exact assignments/no skips; strict corpus v4 qualified the shared source-axis boundary. Full a8 admission remains 19 public implementations and 24 evidence owners, aggregate SHA-256 `0365366dc23750e000c6e9c4f8b384cdf706afdcb338ae3a9f80cfad3d1d8506`. | GPU implementations, policy/dispatch/fallback/cleanup, dependencies, and supported regions are unchanged. | Any listed input or shared layer changes, or a real-GPU canary fails. Requalify affected providers unless a shared/catalogue-wide boundary changes. |
| Workflow/schema/provenance | `v0.14.0a1` / `e16ca87` / 2026-08-26 | Exact-main CI, 393 focused tests, and strict corpus v4 covered workflow schema 5, batch config/manifest schema 4, SourceItem migrations, generated/replay/export paths, per-sample overrides, checkpoints, and provenance. | Schema versions, serializers, migrations, source identity, planning, manifests/checkpoints, and generated-runner contracts are unchanged. | Any schema, migration, source identity, serializer, batch, provenance, or generated-runner contract changes. |
| Dependencies/toolchain | `v0.13.0a8` / `5a66ae9` / 2026-08-22 | a8 qualified the current pins and CuPy-only public route; a1 exact-main clean wheel/sdist installs and exact-tag CPython 3.12.10 builds reconfirmed the unchanged boundary. | Runtime/build pins, supported Python/OS/toolchain matrix, dependency groups, and accelerator stack are unchanged. | Any relevant runtime, build, installer, Python, OS, or accelerator dependency changes. |
| Packaging/release | `v0.14.0a1` / `e16ca87` / 2026-08-26 | Independent exact-tag package builds matched; exact-main clean installs passed; exactly six hash-qualified GitHub assets and matching PyPI bytes were published. | Build backend, package-data/resource layout, entry points, publication workflows, dependency declarations, and asset composition are unchanged. | Any listed input changes or publication canary failure. Exact tag/version/hash/public-URL facts are regenerated for every release. |
| Documentation | source merged at `v0.14.0a1` / `4f213f2`; public deployment pending | PR #18 and strict builds passed, but both release deployment attempts failed before publication; numbered/stable/version-index surfaces remain unqualified. | Not yet eligible for carry-forward as a completed 0.14 deployment baseline. | Retry only after the external hosted-runner/deployment blocker is resolved, then verify numbered, stable, and `versions.json` surfaces. |

## Historical `v0.13.0a8` Release Qualification

- Tag and release commit:
  `5a66ae9d1098ca5a8d409a4075c585692e3c3638`.
- [Exact-main CI run 32584690313](https://github.com/rensutheart/napari-vipp/actions/runs/32584690313):
  all 13 jobs passed.
- [Normal Windows-installer smoke run 32585512509](https://github.com/rensutheart/napari-vipp/actions/runs/32585512509):
  passed.
- [GitHub prerelease](https://github.com/rensutheart/napari-vipp/releases/tag/v0.13.0a8)
  and [PyPI 0.13.0a8](https://pypi.org/project/napari-vipp/0.13.0a8/):
  published and verified.
- [PyPI publication run 32592042093](https://github.com/rensutheart/napari-vipp/actions/runs/32592042093):
  passed; the public wheel and source archive matched the qualified bytes.

Changed since `v0.13.0a7`:

```yaml
core_ui: true
workflow_schema_provenance: true
gpu_scientific: true
windows_installer: true
dependencies_toolchain: true
packaging_release: true
documentation: true
```

### GPU and installed-update evidence

The clean GPU qualification source was
`7189cf40280d895b61b061f1468767164ccfbcf4`. On an RTX 5090 at `cuda:0`, all
19 public implementations and all 24 evidence owners passed. The retained
[aggregate](benchmarks/gpu-admission-0.13.0a8-windows-rtx5090.json) has SHA-256
`0365366dc23750e000c6e9c4f8b384cdf706afdcb338ae3a9f80cfad3d1d8506`.
Production GPU code did not change between that source and the release commit;
the later production delta was the packaging-only deterministic source-archive
canonicalizer.

The installed CUDA route then passed a focused a7-to-a8 update with the final
unsigned installer. The active a8 environment contained no cuCIM,
`nvidia-nvimgcodec`, or old VIPP cuCIM-provider residue; `pip check` passed;
Doctor admitted 19 of 19 implementations; and the two CuPy measurement
implementations passed parity, cancellation, resident-output reuse, and zero
cleanup checks without fallback. The GUI showed VIPP 0.13.0a8. The healthy a7
rollback environment was deliberately preserved under the risk-based
minor-release scope instead of repeating the unchanged CPU, cancellation,
repair, and uninstall matrix. The retained installed-acceptance summary has
SHA-256
`661e7d3fdb84393161e684c2a4f2a6a273a5d52b4456a50d697a11f522c58532`.

### Published artifacts

The GitHub prerelease contains exactly these six qualified assets:

| Asset | SHA-256 |
| --- | --- |
| `napari_vipp-0.13.0a8-py3-none-any.whl` | `77f18e4f34541847c93aa0a54b230a586b9967cc9a66d97370ea0f15ff593ec5` |
| `napari_vipp-0.13.0a8.tar.gz` | `4e1a7b40669832ad8e19e7e0297bae0c3d19fc175f7a99ba3d5284425dbbb433` |
| `VIPP-Setup-0.13.0a8-Windows-x86_64-UNSIGNED.exe` | `5b8233a05696efbf8fea7557012934385021ea3e9018befdd7789bf624740528` |
| `VIPP-Setup-0.13.0a8-Windows-x86_64-UNSIGNED-release.json` | `cac0276a7b0e4556bf93ba1c7bb270f209460070906be061157f783beee33d58` |
| `VIPP-Setup-0.13.0a8-Windows-x86_64-UNSIGNED-THIRD-PARTY-NOTICES.txt` | `95ca668d0977be347ab39af820f97652ed3565ed7ea4218205bb3a716a847e2d` |
| `SHA256SUMS-Windows-0.13.0a8.txt` | `93ad582d443dd95625b1d512d0f3b572d909676565eeb694d53d5b4ffe0b0ab6` |

The PyPI wheel and source archive matched the first two rows byte for byte.
No cuCIM add-on asset was published.

### Companion documentation deployment

Companion documentation PR
[`vipp-mkdocs#16`](https://github.com/rensutheart/vipp-mkdocs/pull/16) merged as
`b6567eeb5a8921926c7b446c4997c42513d0ec34`. Nightly/deployment checks
[32592434921](https://github.com/rensutheart/vipp-mkdocs/actions/runs/32592434921)
and
[32592434943](https://github.com/rensutheart/vipp-mkdocs/actions/runs/32592434943)
passed, followed by successful stable-release deployment
[32592472705](https://github.com/rensutheart/vipp-mkdocs/actions/runs/32592472705).
The public `/0.13.0a8/`, `/stable/`, and `versions.json` surfaces were verified.

### Historical a8 Domain Records

| Domain | Qualified at | Baseline evidence | Carry forward when | Invalidate when |
| --- | --- | --- | --- | --- |
| Core/UI | `v0.13.0a8` / `5a66ae9` / 2026-08-22 | Exact-main CI and focused regressions covered the changed graph, inspector, optimizer, thumbnail, source-axis, numeric-control, and workflow journeys. | The specific UI components, worker/ownership behavior, examples, and interaction contracts being relied on are unchanged. | Affected UI, interaction, worker, or example behavior changes. Smoke only the changed journey. |
| Windows installer | `v0.13.0a8` / `5a66ae9` / 2026-08-22 | Exact-main installer smoke passed. The final hash-locked unsigned EXE updated the installed CUDA a7 environment to a8 with zero retired-provider residue, healthy dependencies, Doctor 19/19, qualified CuPy measurements, and a visible a8 GUI. The healthy a7 rollback was preserved; unchanged CPU/cancellation/repair/uninstall behavior carries forward from a7 and was deliberately not repeated. | Installer code, runtime/build pins, dependency routes, ownership layout, update cleanup, and Windows support assumptions are unchanged. | Any relevant install/update/repair/rollback/uninstall/network/path/ownership behavior or installer dependency changes; signing policy changes; or a canary fails. Version and unrelated embedded feature changes alone do not invalidate lifecycle behavior. |
| GPU/scientific catalogue | `v0.13.0a8` / `7189cf4` plus tag `5a66ae9` / 2026-08-22 | Full RTX 5090 admission passed 19 public implementations and 24 evidence owners on `cuda:0`; aggregate SHA-256 `0365366dc23750e000c6e9c4f8b384cdf706afdcb338ae3a9f80cfad3d1d8506`. Production GPU code was unchanged through the tag, and installed a8 repeated the changed measurement path without fallback. | GPU implementations, scientific contracts, compute policy/specs, dispatch/fallback/cleanup/provenance/memory code, admission catalogue, NumPy/SciPy/scikit-image/CuPy/CUDA pins, and supported hardware assumptions are unchanged. | Any listed input changes or a real-GPU canary fails. Qualify affected providers only unless a shared layer or the catalogue changed. |
| Workflow/schema/provenance | `v0.13.0a8` / `5a66ae9` / 2026-08-22 | Exact-main CI covered workflow schema 4, batch/config/manifest schema 3, source-axis declarations, generated/export paths, migrations, batch planning, and implementation provenance. | Schema versions, serializers, migrations, graph/source identity, batch planning, manifest/checkpoint, and generated-runner contracts are unchanged. | Any schema, migration, source identity, serializer, batch, provenance, or generated-runner contract changes. |
| Dependencies/toolchain | `v0.13.0a8` / `5a66ae9` / 2026-08-22 | Cross-platform exact-main CI, clean wheel/sdist installs, final installer smoke, and the installed CUDA update qualified the a8 pins and the removal of cuCIM/nvimgcodec from the public route. | Runtime/build pins, supported Python/OS/toolchain matrix, dependency groups, and accelerator stack are unchanged. | Any relevant runtime, build, installer, Python, OS, or accelerator dependency changes. Requalify only the affected routes. |
| Packaging/release | `v0.13.0a8` / `5a66ae9` / 2026-08-22 | Independent builds produced identical wheels and canonical source archives; clean installs, metadata, resources, entry points, manifests, and archive guards passed. Exactly six hash-locked GitHub assets were published, and PyPI run 32592042093 published matching wheel/sdist bytes. | Build backend, canonicalizer, package-data/resource layout, entry points, publication workflows, dependency declarations, and asset composition are unchanged. | Any listed input changes or a publication canary fails. Exact tag/version/hash/public-URL facts are regenerated for every release. |
| Documentation | `v0.13.0a8` / `b6567ee` / 2026-08-22 | Companion PR #16 merged; strict/nightly checks passed; release deployment 32592472705 succeeded; numbered, stable, and version-index surfaces were verified. | Site configuration, navigation, installation/upgrade behavior, UI shown in screenshots, and changed feature documentation are unchanged. | Relevant documentation or pictured behavior changes. Inspect only changed pages. |

## Historical Starting Baseline: `v0.13.0a7`

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

### Historical a7 Domain Records

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

## `0.14.0a1` Selection Outcome

The planned a1 selection was executed as recorded in the current ledger above:
core/UI, workflow/schema/provenance, the shared GPU axis/metadata boundary,
focused Windows-installer presentation, packaging metadata, and documentation
changed. The unchanged full GPU catalogue, installer transactional lifecycle,
and dependencies/toolchain remained carried from `v0.13.0a8`. App publication
completed; only the companion public documentation deployment remains open.
