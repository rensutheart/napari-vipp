# Descriptive Statistics implementation contract

Status: unreleased after **0.15.0a5**, within the approved descriptive-only
0.16 scope. The saved operation ID is `summarize_measurements`; its palette
title is **Statistics**. This is an implementation contract, not a second user
guide. Public instructions live in the companion manual's
[Summarize measurements](https://rensutheart.github.io/vipp-mkdocs/nightly/how-to/summarize-measurements/)
page. See the [results scope](measurement-plots-and-statistics-plan.md) for
the explicit exclusion of inferential analysis.

## Version and recipe boundary

The core `StatisticsRecipe` and `summarize_statistics` implementation are
Qt-free and use immutable `TableData`. Scientific choices are operation
parameters, not widget state:

| Parameter | Version-2 meaning |
| --- | --- |
| `summary_version` | Integer `2`; only versions 1 and 2 are accepted. |
| `group_by` | Empty string means no groups. Explicit columns are comma-separated names or a JSON list. `auto` resolves known metadata/leading-index fields. |
| `value_columns` | Explicit measurement columns or `auto`; selected group/identity columns cannot also be measurements. |
| `statistics` | Selected descriptive reductions: `count`, `mean`, `median`, `std`, `q25`, `q75`, `iqr`, `min`, `max`, `sum`. |
| `summary_level` | `Objects`, `Image averages`, or `Sample averages`. |
| `image_column`, `sample_column` | Explicit identity fields; no filename-derived sample inference. |
| `sample_weighting` | `Equal images` or `Equal objects`, within each sample. |
| `missing_policy` | `Exclude and report` or `Stop and review`, applying only to measurement values. |

Automatic measurement selection requires actual real numeric values, excludes
booleans and known metadata/ID-style names, and never parses numeric-looking
strings. Non-finite numeric columns remain selectable so their exclusions
can be reported. Empty/all-null untyped columns require explicit selection.
The resolved columns are retained with the authored recipe in the output.
Column lists support JSON encoding so a literal name containing a comma or
the reserved word `auto` can be selected unambiguously.

New palette nodes use version 2. Deserialization of an old, unversioned
`summarize_measurements` node pins `summary_version=1`, preserving legacy
grouping, conversion/exclusion rules, output shape and singleton SD of zero.
Opening or saving a workflow is not an analysis upgrade. The legacy inspector
offers an explicit, undoable **Upgrade to descriptive Statistics** action.

New recipes default to `group_by=""`, `value_columns="auto"`,
`statistics="count,mean,std"`, `summary_level="Objects"`, blank identities,
`sample_weighting="Equal images"`, and `missing_policy="Exclude and report"`.

`operations.summarize_measurements(...)` retains its historical version-1
default for existing direct calls and generated programs. A direct modern call
must request version 2; use `group_by=""` explicitly for no grouping because
that wrapper retains its historical `group_by="auto"` default. The modern
`StatisticsRecipe` itself defaults to no grouping. Workflow saves, undo/redo,
generated Python and shared execution carry the chosen version. Batch
scientific identity canonicalizes only inert version-1 migration fields;
upgrading to version 2 changes the scientific identity.

## Populations and weighting

Each measurement is filtered independently within each group. Reductions use
the full eligible population, never a preview sample. Let `x` denote eligible
object values, `m_i` the arithmetic mean for image `i`, and `s_j` the mean for
declared sample `j`.

- **Objects:** reduce all eligible `x`, giving each object equal weight.
- **Image averages:** calculate `m_i` within each image, then reduce the
  contributing image means with equal weight. Image identity is required.
- **Sample averages / Equal images:** calculate `m_i`, average the contributing
  image means within each sample to obtain `s_j`, then reduce the sample means
  with equal weight. Both image and sample identity are required.
- **Sample averages / Equal objects:** pool eligible objects within each
  sample to obtain `s_j`, then reduce the sample means with equal weight.
  Sample identity is required; image identity is optional for reporting.

For images `[2, 4]` and `[12]` in one sample, equal-image weighting yields
`(3 + 12) / 2 = 7.5`; equal-object weighting yields `(2 + 4 + 12) / 3 = 6`.
At sample level, both have final `n=1` and undefined sample SD. Neither is an
inferential claim. No identity declaration proves biological independence.

Groups retain their first-appearance order. Identity keys preserve scalar
types: integer `1`, float `1.0` and text `"1"` do not silently become one
identity. Large integer IDs are not converted through float64. Group values
may be Boolean; image/sample identities may not. All selected grouping and
identity fields must be nonmissing finite scalar text/numbers, even on rows
whose measurement is excluded.

An image must belong to one group when calculating image averages or
equal-image sample averages. A sample must belong to one group when
calculating sample averages. Object-level summaries may describe different
categories within one image/sample; image averages may describe groups
sharing a sample. These are descriptive groupings, not paired-sample inference.
When both identities are supplied, each image must belong to one sample.
Reused local image IDs across samples are rejected; users must supply globally
distinguishing identities. Object IDs are not silently substituted for image
IDs.

## Numerical definitions

Numeric measurements are real values calculated in float64. Signed values
are supported; booleans, complex values and text are not coerced. Reject a
measurement integer when conversion to float64 changes its exact integer
value. Reject finite values or computed results outside float64 range instead
of publishing infinity. Means and sample SD use Python standard-library
`statistics` reductions with exact-ratio intermediate accumulation to avoid
unnecessary overflow and large-offset cancellation; sums use `math.fsum` with
an exact-ratio fallback if an intermediate sum overflows before cancellation.
Results remain float64, not arbitrary-precision output or a biological
validation claim.

For the final eligible units `u[0] ... u[n-1]`:

- `count = n`; `mean = sum(u) / n`; `sum` adds these units, not necessarily
  original objects.
- Sample `std = sqrt(sum((u - mean)**2) / (n - 1))`, corresponding to
  [NumPy `std` with `ddof=1`](https://numpy.org/doc/stable/reference/generated/numpy.std.html).
  It describes spread, not a confidence interval or an assertion of
  independent sampling.
- Sort values and use `h=(n-1)*q` with linear interpolation between the
  neighbouring sorted values. `median` uses `q=0.5`, `q25` uses `0.25`, and
  `q75` uses `0.75`; `iqr=q75-q25`. This matches the definition of
  [NumPy `quantile(method="linear")`](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html).
- `min` and `max` are extrema of the final eligible units.

No silent normalization, unit conversion, analytical log transform, outlier
removal, bootstrap, statistical test, p-value, ANOVA, significance annotation
or confidence interval is part of this operation.

## Exclusions and empty populations

Version 2 distinguishes three mutually exclusive measurement exclusions:

| Count | Values |
| --- | --- |
| `missing_count` | `None` or empty/whitespace-only text. |
| `nonfinite_count` | Numeric NaN or positive/negative infinity. |
| `nonnumeric_count` | Other nonnumeric values, including numeric strings and booleans. |

`Exclude and report` keeps counts and descriptive status text. `Stop and
review` raises when a selected measurement has excluded values. Neither
policy suppresses missing-column, malformed-table, ambiguous-identity,
conflicting-output-name or numeric-range errors. No measurements are replaced
by zero.

Groups with no contributing units remain in the result: `n` and `count` are
zero; every selected numeric statistic, including `sum`, is `None`. With one
unit, sample SD is `None` and status explains `n < 2`; other reductions are
defined. Two or more identical units have SD zero. An ungrouped empty input
with explicitly selected measurement columns produces one zero-count row;
an empty grouped input has no groups and hence no output rows. Automatic
selection on an empty/untyped table requests explicit measurement columns.

An image or sample with no eligible value does not receive an invented zero
mean. Per-measurement identity totals count represented identities before
exclusions; valid counts include identities with at least one eligible object.
Zero-row batch images have no object rows in the input: their status remains
in the collection inventory/image summary, not a fabricated Statistics row.

## Ordinary wide result table

Output is named **Statistics**, with `table_kind="Descriptive Statistics v2"`.
It keeps the input source name and is an ordinary `TableData`, not a special
analysis object. Rows begin with group values, then method/count fields:

| Fields | Meaning |
| --- | --- |
| `summary_version`, `summary_level`, `sample_weighting` | Version, units summarized and within-sample weighting. |
| `group_by`, `image_column`, `sample_column`, `missing_policy` | Resolved groups and authored identities/policy. |
| `statistics_recipe` | JSON recipe plus resolved group/value columns, `std_ddof=1`, and `quantile_method="linear"`. |
| `row_count`, optional `image_count` / `sample_count` | Input objects and distinct supplied identities before measurement exclusions. |

For each selected measurement `m`, append:

- `m_object_total`, `m_object_valid`, `m_object_excluded`;
- `m_missing_count`, `m_nonfinite_count`, `m_nonnumeric_count`;
- `m_n`, the actual number of final units summarized;
- `m_image_total`, `m_image_valid` when image identity is supplied, and
  `m_sample_total`, `m_sample_valid` when sample identity is supplied;
- `m_<statistic>` for each requested reduction; and
- `m_status`, with exclusions and insufficient-data reasons, or `ok`.

Invariants: `object_total = object_valid + object_excluded` and
`object_excluded = missing_count + nonfinite_count + nonnumeric_count`.
`m_count = m_n`; it need not equal `m_object_valid` at image/sample level.

Statistic columns retain the corresponding measurement unit except for
`count`. Counts/provenance have no physical unit; grouping columns preserve
their input unit if one exists. Different measurement units remain separate
columns. Name collisions are rejected instead of overwriting data.

Ordinary table inspection, CSV/TSV export, table composition and Plot Results
accept this result. Plotting a summary does not recover raw observations or
automatically interpret SD as error bars. Excel export remains specific to
the batch-collection export surface, not ordinary Statistics table export.

## Execution and evidence boundary

The nonmodal **Results Workspace** is another editor for the same Statistics
node, not another calculation or persistence model. It starts from one table
and creates Statistics/Plot Results nodes only through explicit add actions.
Inspector and workspace edits share the saved recipe and normal undo/dirty
state. Unrelated summary nodes remain independent. Source collection/joining
stays upstream; browsing/search/column visibility is not a scientific filter.

When plotting this result in the workspace, the selected source is explicitly
**Summary table** rather than **Original measurements**. The connected rows
are already summaries, not the original objects or images. The workspace
must not infer raw replication, add SD error bars, or silently average these
rows again by image. Result export remains the full ordinary table, including
counts and method fields; preview column selection is presentation only.

The operation reads but never mutates input rows/buffers. It reports progress
and checks cancellation through the shared execution context. It does not
write files, infer physical axes, resample or access image pixels. 2D, 3D and
leading-axis measurements use the same table contract; leading indices matter
only when explicitly grouped or resolved by the saved `auto` grouping rule.
No physical calibration is inferred or repaired here.

Acceptance covers analytical known answers, unequal image/sample yields,
invalid/empty/singleton populations, identity conflicts, wide numeric values,
units, input immutability, ordinary export/plotting and execution routes.
Compatibility tests must cover both the old numerical behavior and explicit
upgrade. Repository tests establish these bounded computational contracts;
they do not validate an experimental design or guarantee biological validity.
