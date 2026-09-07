# Documentation ownership

**One public manual, with implementation context beside the code.**

| Content | Authoritative home |
| --- | --- |
| Installation, user tasks, controls, formats, scientific interpretation | [vipp-mkdocs](https://github.com/rensutheart/vipp-mkdocs/tree/main/docs) |
| Plans, architecture, execution/schema invariants, implementation decisions | This repository's [documentation index](README.md) |
| Tests, benchmarks, qualification records, fixture generators | This repository, beside the implementation they verify |
| Contribution, support/security policy, packaging and release procedures | This repository |
| Executable examples and developer acceptance instructions | This repository's examples, scripts, and packaged registry |
| Example tours, user screenshots, and worked tutorials | vipp-mkdocs |
| Release metadata and chronological changelog | This repository; the manual publishes a reader-oriented release summary |

## When behavior changes

1. Update code/tests and any affected local implementation contract.
2. Update the relevant task/reference page in vipp-mkdocs. Link the companion
   documentation change in the application PR; do not recreate a local guide.
3. Mark unreleased behavior clearly in the nightly manual. Keep README user
   links on `stable`; keep release-specific instructions on their numbered version.
4. Run the application documentation tests, the manual's content check, and
   `mkdocs build --strict`. Inspect changed pages and links.

If the manual checkout is unavailable, record the required companion change
in the handoff. Do not claim documentation is finished or invent a second copy.

## Where the former public guides went

The old paths contain only relocation notes. `public-guide-redirects.json`
records their topic destinations; tests enforce that these stay small links.
Historical Git revisions retain the original text.

| Old guide | Public destinations |
| --- | --- |
| [Quick start](quick-start.md) | Installation, macOS, first-workflow tour |
| [User guide](user-guide.md) | How-to guides, interface/display reference, batch tutorial |
| [GPU guide](gpu-guide.md) | Windows NVIDIA setup, choosing compute, validation status |
| [Import/export](io-user-guide.md) | Format reference, optional readers, batch and export contracts |
| [Cache/memory](cache-and-memory.md) | Memory reference, performance troubleshooting, crop how-to |
| [Operator tips](operator-tips.md) | Performance, display, batch, and restoration guides |
| [Measurement workflows](measurement-workflows.md) | Object tutorial, table/units reference, skeleton and colocalization tutorials |
| [Skeleton nodes](skeleton-nodes.md) | Skeleton tutorial and node/output reference |
| [Colocalization methods](colocalization-method-notes.md) | Metric definitions, RACC-like index, and object-association reference |

Additional detail from the former user guide lives in the manual's intensity/
threshold reference, channel/axis reference, and crop how-to. Recent display
wording and search changes are marked as post-0.15.0a1 nightly behavior.

Architecture and durable execution pages stay here because they document
implementation invariants. Public overviews should link to them for engineering
detail, not copy their full specifications. Plans, implementation reports,
and dated benchmarks are not current installation instructions.

## Keep it readable

Use a short task page for steps and a focused reference page for exact details.
Put the action or answer first. Prefer short paragraphs, numbered procedures,
and compact comparison tables. Keep scientific caveats beside the relevant
choice; move optional derivations and implementation history out of the main
path. The manual's contributor guide owns the detailed writing checklist.
