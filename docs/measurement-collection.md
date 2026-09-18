# Batch measurement collection

Status: implemented and integration-tested, unreleased after 0.15.0a5.
Scope agreed: 2026-09-15. This is the data foundation for the
[0.16 results plan](measurement-plots-and-statistics-plan.md), not a plotting
or statistical-inference implementation. Public instructions belong in the
[companion manual](https://github.com/rensutheart/vipp-mkdocs).

## Decision and boundary

Add an explicit **Collect measurement results…** action to Batch **Run &
results**. A user selects one saved table output across the recorded batch,
reviews its item inventory, and uses **Export results…** for direct CSV, TSV or
Excel export. Export does not require another workflow or a native dataset
save. **Save VIPP collection…** separately saves a reusable `.vipp-results.json`
dataset; **Open VIPP collection in a results workflow** is optional and unchecked
by default. Both actions leave the review open and report success inline.
Opening the native dataset creates a results workflow with a file-backed
**Table Source** feeding existing table tools. Its
**Review collection…** action reopens the saved inventory read-only, including
zero-row and excluded items.

This is vertical collection of compatible per-image observations. **Merge
Tables** remains a horizontal join of matching observations; **Add Metadata
Columns** remains the tool for adding constant annotations within a graph.
Neither operation performs hidden cross-item batch execution. Existing numeric
per-item parameter overrides are not repurposed as a text-metadata editor.

## Collection contract

- Start from recorded batch output evidence, not a folder scan of arbitrary
  CSVs. Verify each candidate's saved content hash before decoding it.
- Record a versioned typed table schema with new batch table outputs. Rebuild
  values from that evidence, preserving scalar types, units, column order,
  missing/non-finite distinctions, and existing annotation columns. Do not
  infer types from CSV spelling or silently convert incompatible schemas/units.
- Preserve batch/run, source-item/image and output identities alongside each
  row. Local object IDs remain local: object 1 in two images represents two
  observations, not one object. Retain the recorded analysis and effective
  parameter provenance; collection does not rerun the measurement node.
- Keep a complete per-item inventory, including valid zero-row tables and
  failed, missing, changed, unsupported, skipped or explicitly excluded results.
  An absent result is not a zero-object image. Empty valid tables contribute no
  invented measurement row, but remain visible in the inventory.
- Exclusion is an explicit, reviewed decision. Never drop a problem item to
  make the dataset appear complete, silently substitute a different output,
  or append the same source/output twice after a resumed run.
- Offer bulk annotation of missing condition, sample and replicate fields.
  Preserve authored values and require deliberate identity assignment; do not
  infer biological independence from filenames, row count, or image count.
  Annotation is post-run metadata, not a change to the recorded measurements.

Historical CSV/TSV files without the required typed output evidence cannot be
collected safely by guessing. Explain the missing record and offer the user
the option to rerun the original analysis with a supporting VIPP build. Opening
the review must never start that rerun or modify old outputs/evidence.

## Direct export contract

- CSV/TSV publishes the collected measurement rows. An optional companion
  `<stem>-image-summary.csv` or `.tsv`, selected by default, retains the complete
  per-image inventory, inclusion decisions and annotations. Omitting it must
  not imply that a measurement-only file accounts for zero-row or excluded
  images.
- Excel publishes one `.xlsx` workbook with **Measurements**, **Image summary**
  and **About this collection** sheets. The image summary is always included;
  About this collection carries column units and run information. These are
  exports of reviewed results, not newly calculated statistics.
- Preserve empty, excluded, missing, changed and failed outcomes as distinct
  inventory entries. Never manufacture object rows to represent zero objects.
- Keep the native typed snapshot a separate, lossless round-trip route. Do
  not route spreadsheet exports through a saved workflow or reprocess images.
  The ordinary Table Source **Export table…** action remains CSV/TSV-only and
  does not imply Excel or collection-inventory support for generic tables.
- Apply explicit overwrite, cancellation and safe-publication handling to
  exports, including both delimited files when a companion is requested.
  Spreadsheet output must not silently turn authored text into formulas or
  round identifiers that exceed spreadsheet numeric precision.

## Persistence and execution

The dataset is self-contained **for measurements**: typed rows, units, reviewed
annotations, item inventory, inclusion decisions and provenance. It contains
no source pixels, segmentation arrays, mesh geometry or previews. It is a
result artifact, not a reproducibility package or a resumable batch manifest.
Source names, identifiers and authored annotations may be sensitive.

A saved results workflow stores the external dataset path and expected content
hash, not embedded measurement rows. Table Source loads the same checked file
through the Qt-free execution path; a missing or changed dataset is an
actionable error, not permission to accept another revision. Moving a workflow
therefore requires sharing the dataset separately. Reproducibility-package
export must continue to exclude result data.

Initial Python export is deliberately unavailable for workflows containing
Table Source; the exporter reports that limit instead of writing a broken
image-oriented runner. Reproducibility packages still include the portable
recipe and expected dataset hash, omit the unsupported runner, and explain how
to reopen the separately shared dataset in VIPP. The core source and ordinary
table execution remain Qt-free and are tested through CPU and Auto requests.

Collection and opening do not modify the image workflow, rerun segmentation,
change source inputs, or publish files without the user's save action. File
publication must retain the application's atomic-save and cancellation rules;
background results must not replace newer selections or closed UI state.

## Acceptance and deferred scope

Verify typed round trips (including text-like numbers, Boolean values, wide
integers and missing/non-finite values), compatible units/schema enforcement,
changed/missing files, valid empty tables, mixed item outcomes, explicit
exclusions, annotation preservation, repeated local IDs, and resumed-run
identity. Check dataset/workflow save and reopen, headless execution, stale
source detection, and preservation of the existing table operations. For
direct exports, check all three formats, workbook sheet contents and units,
optional companion publication, empty/excluded image coverage, annotations,
numeric/text fidelity and overwrite/cancellation behavior. Opening a results
workflow must stay opt-in and separate from spreadsheet export.

Keep **Plot Results**, expanded **Statistics**, inferential tests, imported
metadata spreadsheets, arbitrary historical CSV inference and automatic
biological-replicate assignment outside this initial implementation. Current
**Summarize Measurements** retains its existing semantics; collection alone
does not establish a valid experimental design or independently validate an
analysis.

## Implementation validation

The focused integration run passed 124 tests spanning collection, its Qt review,
Table Source execution, results-workflow tabs, batch actions and catalogue
coverage. Further source-widget regression checks cover explicit hash adoption,
stale loads, changed files and Windows path spelling. The real batch test runs
three synthetic images (including one with no objects), saves their tables,
collects and reopens the dataset, then resumes the completed batch and confirms
that no measurements are recalculated or duplicated. Existing workflow,
reproducibility-package and batch UI regression sets also passed.

The direct-export extension passed 164 combined feature tests, including 45 Qt
collector checks, 39 exporter checks and 15 native-save destination guards.
The real three-image batch also exports CSV, TSV and XLSX without rereading
inputs, retaining the empty image's annotation in the summary. Separate batch,
packaging and documentation checks passed, as did Ruff, manifest validation,
the strict manual build and wheel construction with its Excel dependency.

Dark and light/narrow large-font Qt renders were inspected. These checks do not
constitute a release qualification or installer build.
