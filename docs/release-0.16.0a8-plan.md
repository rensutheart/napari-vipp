# 0.16.0a8 UI and Windows updater release

This non-AI iterative alpha covers the complete reviewed app and manual changes
since public `v0.16.0a7` at `a538c260b9c43b6c2e6282727f4d7ef7e9b90591`.
Keep assistant/MCP work, user data, research/demo outputs, and local evidence
out of the release. Follow the [release runbook](release-runbook.md) and
[qualification declaration](release-qualification-baseline.md#0160a8-qualification-declaration).

## Changed behavior and gates

- **Core/UI and recovery:** verify the reviewed graph, inspector, search,
  loading, dimensions, node details, multi-session recovery, result tables,
  and Review Images journeys. Preserve manual/writer holds and unselected or
  failed working copies. Exercise changed presentation and recovery in a
  native synthetic workflow; leave user sessions and files alone.
- **Science and graphics:** compare Hysteresis Threshold against its existing
  connectivity and mask contract, including mixed values and axes. Exercise
  affected native 3D/oblique, GPU renderer and reduced-intensity preview paths.
  Keep masks/labels and workflow arrays exact, report display failures, and
  distinguish OpenGL renderer evidence from CUDA compute. No full CUDA
  catalogue rerun for unchanged providers and pins.
- **Windows setup:** validate legacy admission and changed-package In-place,
  Replacement/Fresh, ownership/registration repair, cache reuse, running-app
  exclusion, interruption/rollback, and runtime-isolated Open VIPP. Reuse
  earlier exact unchanged-engine evidence where applicable. Exercise the
  changed update/rollback and launch journey with disposable managed fixtures,
  not the user's installed environments. EXE version output alone is not
  lifecycle proof.
- **macOS:** carry only unchanged constructor/installer behavior, dependencies
  and platform assumptions. Audit new separate unsigned/unnotarized native
  arm64 and x86_64 packages from the exact tag.

## Publication

Use one app and one companion manual PR. Require exact candidate and final-main
CI, reproducible qualified Python artifacts, metadata and AI-exclusion checks,
and the changed-domain gates before tagging. Verify the new Windows frozen
wheel and platform installer inventories from the immutable tag. Publish the
audited GitHub prerelease once, verify actual public bytes, then publish the
same Python files on PyPI and verify those bytes. Deploy the numbered manual
and verify the default alias after package publication. No existing installation
or tag changes are needed. These requirements are not completed-release claims.
