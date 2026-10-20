# 0.16.0a5 non-AI release plan

This iterative alpha integrates the non-AI work after public `v0.16.0a4`
(`86aee0f4056766828c53fd66067f4e67ea2570dc`). It is not an AI feature release.
Use the [release runbook](release-runbook.md) and the union declaration in the
[qualification baseline](release-qualification-baseline.md#0160a5-qualification-declaration).
The declaration states requirements, not completed publication.

## Included histories and independent slices

| Group | Source checkpoint | Integration boundary |
| --- | --- | --- |
| Installer registration/recovery | `a3003ab5ff8209186c721fa87775015fb39b4480` | Exact ownership/preflight and actionable help; retain transaction and active-environment guards. |
| Wires, tunnels and floating-window reuse | `86d96025d63515c96ff179e15ef7b5174a193db2` | Eight unmerged commits; preserve computational graph edges, output lanes and a4 window ownership fixes. |
| Thumbnail busy cleanup | `159052b3a88f38dabdd388ea938600e1ce8ea5ec` | Presentation work queue/progress only. |
| Portable workflow/batch/Python paths | `39833cfdeb11075f4bea1fd640cf5643c3c42c2a` | Document-relative binding/rebasing; preserve revisions, checksums and output guards. |
| In-app exact-build GPU setup | `c7564ad4250e5ac3c725b00be32951bbd0e49071` | Separate managed Windows environment and matching dialogs, not new scientific GPU support. |
| Image-writer no-clobber safety | `91331d40e10d74ed54ddde9040a3f7fbe91f6e62` | Independent writer/test extraction; fail closed when exclusive publication is unsupported. |
| Standalone workflow recovery | `70e6c7f1306297b9164b57b2dbe7fa8f72232301` plus `12dbdc8d42e39050c59f8e5ce0e87396ccf57440` | Independent graph-only store/controller and selective widget hooks; resolved bindings survive recovery into unsaved tabs. |

The thumbnail, portable-path and GPU work originated as copied dirty stacks.
Their scoped snapshots are committed and history-preserving merges integrate
each change once. The combined GPU adapter needs the follow-up private helper
rename `619024e`: its `_inspect` name collided with the newly merged installer
base-class recovery helper. This fixes delegation without weakening checks.

Independent review also found that newly authored relative live bindings could
be silently retargeted to a save/export destination folder. Follow-up
`f91945abefdba0fa12c9983cf5c2af205a8d0951` freezes their actual runtime base before
ordinary Save, Python export and included-Batch hash generation. Three new
regressions fail before the fix; explicit raw-document/source-path semantics
remain unchanged.

Recovery captures graph recipes, titles and executable bindings, not source
data, calculated caches, Undo history, ongoing work or Batch configuration/UI.
It restores copies, does not overwrite originals, and does not auto-calculate.
Save Batch configuration separately. The combined relocation/recovery tests
cover image source-container identities, table checksums and Save Image targets.

## Explicit exclusions

The app AI checkpoint `eb0d428bdd081a389660743a65dfb8b5a3009376` remains on
`feat/ai-workflow-proposals`; its dirty work is preserved. Do not merge that
mixed checkpoint wholesale. Assistant/proposal/investigation/refinement,
output assessment, protected editing, saved conversations/sessions, AI toolbar
actions and the MCP entry point are excluded. Only the two independently
audited non-AI slices above cross that boundary.

The companion manual remains separate from `docs/ai-workflow-proposals`.
Research/prototypes, generated sites, cached diagnostics, private wheels and
gh-pages are not release source. No separate bot/webhook refs were found in this
application repository. Already-released Batch/Results, Review Images, OME-Zarr,
legacy OME/layer, original detached-window and NRF changes are not merged again.

## Verification and order

1. Review the combined app/manual scope and ensure AI modules, entry points and
   user-facing navigation are absent. Keep existing histories and dirty work.
2. Run affected persistence/path/export/writer, graph/thumbnail/window, GPU
   setup and installer checks. Retain prior failures and deliberate skips;
   neither incomplete full suites nor deselected native tests are clean passes.
3. Prove normal autosave followed by abrupt termination and explicit recovery
   using only an owned synthetic producer process. Check original workflow/data
   hashes, unsaved restored tabs, usable file bindings and no automatic run.
4. Require exact candidate CI, clean package installs and the affected native
   installer/GPU setup scenarios. Use an isolated supported Windows account/VM
   for registration/upgrade recovery; do not modify active/shared environments.
   Windows frozen payload/version checks alone are not lifecycle qualification.
5. Merge the reviewed release histories once; require exact final-main CI.
   Independently reproduce and retain its wheel/canonical source pair. Verify
   exact artifact modules, entry points, versions, sizes and SHA-256 hashes.
6. Only after gates pass, tag the exact clean main immutably, build selected
   exact-tag installers once, qualify their exact artifacts, publish identical
   GitHub/PyPI Python bytes, then deploy/verify the numbered/default manual.

No tag, publication or docs deployment is implied by this plan. Native Linux/
macOS window-manager parity, a full frozen Windows GUI lifecycle, broad
filesystem support, source-data backup and scientific accuracy are not claimed.
