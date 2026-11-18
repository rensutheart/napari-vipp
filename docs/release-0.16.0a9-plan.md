# 0.16.0a9 recovery, canvas and Crop patch

This non-AI iterative alpha combines the reviewed fixes since public
`v0.16.0a8` at `0d33dc6c7bd48fa308c9d33a6f93cdc545b15efa`.
Follow the [release runbook](release-runbook.md) and
[qualification declaration](release-qualification-baseline.md#0160a9-qualification-declaration).

## Changed domains

- **Recovery orchestration:** omit only an untouched startup workflow; keep edited
  recipes recoverable, preserve pre-edit protection, and do not delete historical
  recovery data. Cover starter-preview, source-load and launcher handoffs.
- **Canvas and footer:** require canvas-owned mouse press/held-button state for
  panning, cancel interrupted gestures, retain normal drop/slider behavior and
  deliberate navigation during calculation. Check stable progress height at
  narrow/wide widths and with host insets; keep wrapped failures readable.
- **Crop presentation:** guard the complete viewer-layer transition against
  re-entry, reuse an unchanged source image/ROI presentation, and publish real
  image/calibration changes. Exercise selection, margin edits, dimension changes,
  other-node selection and actual napari layer-list ownership.

Run focused automated regressions and one new isolated native synthetic smoke
of these changed journeys. Do not change installed environments, global settings,
user windows, workflows or original data to qualify the patch.

## Unchanged domains and publication

Workflow, batch and manifest schemas remain 6. No scientific kernels, GPU
providers/dispatch, runtime dependencies, installer code or publishing workflow
changes are included. Carry forward a8 qualification only within its recorded
boundaries; there is no claim of a new full GPU/installer lifecycle matrix.

Use one app and one companion manual PR with history-preserving normal merges.
Require candidate and exact final-main CI before an immutable tag. Reuse the
exact-main CI qualified reproducible Python pair. Audit newly built unsigned
Windows x86_64 and separate CPU-only, unnotarized macOS arm64/x86_64 artifacts.
Publish audited GitHub assets, verify actual public bytes, publish that same
Python pair on PyPI, then deploy and verify numbered/default documentation.
Record exact SHAs, runs, hashes and public outcomes in release evidence;
these requirements are not completed-gate claims.
