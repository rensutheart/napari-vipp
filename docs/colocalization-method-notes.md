# Colocalization Method Documentation

Last reviewed: 2026-07-03

This document describes the colocalization and object-association calculations
implemented in VIPP. It is intended as methods-text source material for a
manuscript, validation report, or public technical documentation.

Implemented reference workflows:

- `examples/synthetic-colocalization-racc.json`;
- `examples/synthetic-object-colocalization-association.json`.

## Dependency And Scope Statement

VIPP does not depend on the standalone RACC napari plugin. RACC is not listed
in `pyproject.toml`; the `RACC Index` and `Masked RACC Index` nodes use VIPP's
own implementation in `napari_vipp.core.operations`.

The VIPP RACC node should therefore be described as a VIPP-implemented
RACC-like index rather than as a wrapper around the external RACC plugin. The
standalone RACC plugin remains a separate package for its focused interactive
workflow. If future releases add direct RACC-plugin interoperation or extract a
shared numerical core, that dependency and any relevant license or patent
notices should be documented separately.

## Input Conventions

Colocalization nodes operate on two same-shaped scalar image channels. If an
RGB/RGBA image is supplied to a colocalization node, VIPP reduces it to a
single intensity image by taking the maximum of the first three colour
channels. Multichannel fluorescence data should usually be split into scalar
channels first with `Split Channels`.

All colocalization threshold parameters are expressed on a normalized
`0..255` intensity scale. For each two-channel calculation, VIPP converts both
channels to floating point and jointly normalizes them as follows:

1. Negative values are clipped to zero.
2. If the joint maximum of both channels is less than or equal to `1`, both
   channels are interpreted as normalized float images and multiplied by `255`.
3. If the joint maximum exceeds `255`, both channels are scaled by
   `255 / joint_maximum`.
4. Otherwise, the original intensity scale is retained.
5. The result is clipped to `[0, 255]`.

This joint normalization preserves the relative intensity scale between the two
channels while keeping manual and automatic thresholds comparable across
`uint8`, `uint16`, and normalized floating-point images. Any normalization
adjustments are reported in the `normalization_warnings` table column.

Masked colocalization nodes accept a third ROI input. Voxels with ROI values
greater than zero are included; all other voxels are excluded. When no ROI mask
is supplied, all voxels are included.

## Thresholds And Positive Sets

Let `I1(i)` and `I2(i)` be the normalized channel intensities at voxel `i`.
Let `R` be the analysis population: either all voxels, ROI-restricted voxels,
or one labeled object. Let `T1` and `T2` be the selected channel thresholds.

VIPP defines:

```text
P1 = {i in R | I1(i) >= T1}
P2 = {i in R | I2(i) >= T2}
C  = P1 intersect P2
```

`P1` and `P2` are the threshold-positive voxels for each channel. `C` is the
colocalized voxel set. Table columns report `channel_1_positive_voxels`,
`channel_2_positive_voxels`, `colocalized_voxels`, and
`colocalized_fraction`. For whole-image and ROI analyses,
`colocalized_fraction = |C| / |R|`. For object-restricted analysis,
`colocalized_fraction = |C| / object_voxels`.

Manual thresholds use the threshold values supplied by the user. `Costes auto`
calculates `T1` and `T2` automatically, as described below, and writes the
calculated values back into the visible node parameters.

## Pearson Correlation

VIPP reports Pearson correlation for the selected intensity population:

```text
pearson(X, Y) =
  sum((X - mean(X)) * (Y - mean(Y)))
  / sqrt(sum((X - mean(X))^2) * sum((Y - mean(Y))^2))
```

For whole-image or ROI-restricted metrics, `pearson_all` is calculated over
all voxels in the analysis population `R`. For object-restricted metrics,
`pearson_object` is calculated over all voxels in the labeled object.

`pearson_colocalized` is calculated only over the colocalized voxel set `C`.
If fewer than two voxels are available, or if either channel has effectively
zero variance in the selected population, VIPP reports `NaN`.

Pearson values are intensity-correlation measurements. They do not by
themselves imply spatial overlap unless the analysis population and thresholds
are also reported.

## Manders Coefficients

VIPP reports thresholded Manders-style intensity fractions:

```text
M1 = sum(I1(i) for i in C) / sum(I1(i) for i in P1)
M2 = sum(I2(i) for i in C) / sum(I2(i) for i in P2)
```

`M1` is the fraction of threshold-positive channel-1 intensity that is also
threshold-positive in channel 2. `M2` is the reciprocal fraction for channel 2.
Both are therefore threshold-dependent. If the denominator is zero, VIPP
reports `NaN`.

The numerator and denominator values are also exposed through the
`channel_1_positive_sum`, `channel_2_positive_sum`,
`colocalized_channel_1_sum`, and `colocalized_channel_2_sum` columns.

## Intensity Overlap Coefficient

VIPP reports an intensity overlap coefficient:

```text
overlap_coefficient(X, Y) =
  sum(X * Y) / sqrt(sum(X^2) * sum(Y^2))
```

For whole-image or ROI-restricted metrics, `overlap_coefficient_all` is
calculated over the full analysis population `R`. For object-restricted
metrics, `overlap_coefficient_object` is calculated over each labeled object.

`overlap_coefficient_colocalized` is calculated only over the colocalized voxel
set `C`. If the selected population is empty or both intensity norms are zero,
VIPP reports `NaN`.

This value is not the same as binary overlap or intersection-over-union. It is
an intensity-space similarity measure and should be interpreted alongside the
thresholded voxel counts and Manders coefficients.

## Costes Automatic Thresholding

When `Costes auto` is selected, VIPP estimates a two-channel intensity
relationship on the normalized `0..255` intensities and searches for thresholds
that drive correlation in the below-threshold population toward zero.

For the selected threshold population, VIPP computes channel means,
variances, and covariance. It fits a line:

```text
I2 = slope * I1 + intercept
```

The slope is estimated from the variance/covariance terms using a
Deming-style relationship. If covariance is effectively zero but both channel
variances are non-zero, the fallback slope is `sqrt(var(I2) / var(I1))`; if the
relationship is degenerate, the fallback slope is `1`.

Threshold search proceeds along the fitted line. If `-1 < slope < 1`, VIPP
searches along the channel-1 axis and maps each candidate channel-1 threshold
to channel 2 using the fitted line. Otherwise, it searches along the channel-2
axis and maps back to channel 1. Candidate thresholds are clamped to
`[0, 255]`.

At each iteration, VIPP calculates Pearson correlation for the population:

```text
B = {i | I1(i) < T1 or I2(i) < T2}
```

If the below-threshold Pearson value is finite and positive, the search moves
the threshold downward; otherwise it moves upward. The search stops when the
step size is below `1` normalized intensity unit or after 100 iterations. Final
thresholds are rounded to the nearest integer and clamped to `[0, 255]`.

The output table records:

- `channel_1_threshold`;
- `channel_2_threshold`;
- `costes_slope`;
- `costes_intercept`;
- `costes_pearson_below`;
- `costes_iterations`.

For object-restricted colocalization, Costes thresholds are calculated once
over all foreground voxels in the supplied label image and then reused for each
object. This makes object rows comparable and avoids unstable per-object
threshold estimates for small objects.

## Colocalized-Voxel Visual Outputs

`Colocalized Voxels` and `Masked Colocalized Voxels` are visual feedback nodes.
They use the same normalized intensities, thresholds, and ROI restriction as
the metric nodes. Voxels in `C` can be rendered as white on the channel-colour
composite, white on black, or in channel colours only. Voxels outside an ROI
mask are set to black.

These outputs are intended for threshold review and figure generation; the
table outputs are the primary quantitative record.

## Scatter-Density Inspector

The colocalization inspector scatter panel plots channel-1 intensity against
channel-2 intensity over the active analysis population. VIPP uses a 2D
histogram over `[0, 255] x [0, 255]`; display counts can be log-transformed
with `log1p`. The threshold guide lines are drawn at `T1` and `T2`.

Dragging either guide line switches the node back to manual thresholds and
updates the corresponding threshold parameter. The scatter plot is therefore a
threshold review and tuning view, not an additional metric.

## RACC-Like Index

The `RACC Index` and `Masked RACC Index` nodes calculate a scalar image over
the threshold-positive overlap set `C`. Voxels outside `C`, or outside the ROI
for masked analysis, are assigned zero.

The RACC-like calculation requires at least two voxels in `C`. VIPP fits a
positive Deming-style regression line through the overlap intensities:

```text
I2 = slope * I1 + intercept
```

The line is anchored between two points:

- `p0`, where the fitted line intersects the lower threshold boundary defined
  by `T1` and `T2`;
- `p1`, where the line intersects the upper normalized intensity boundary
  at `255`.

For every voxel in `C`, VIPP projects the intensity point `(I1, I2)` onto the
line from `p0` to `p1`, producing a fractional position `t`. It also measures
the perpendicular distance from the point to the line, normalized by `255`.

The `include_percentile` parameter defines the high-intensity and distance
population used to scale the output. VIPP calculates:

- `t_max`, the selected quantile of projected positions;
- `pmax = p0 + t_max * (p1 - p0)`;
- `distance_threshold`, the selected quantile of normalized perpendicular
  distances.

The output value for each overlap voxel is then:

```text
value = min(t_to_pmax, 1) - distance_to_line * tan(theta)
```

Values are set to zero when `t_to_pmax <= 0` or when the normalized distance
is greater than `distance_threshold`, and the final result is clipped to
`[0, 1]`. The default output is `float32`; optional `uint8` output scales the
clipped result to `0..255`.

The `theta` parameter therefore controls how strongly off-axis points are
penalized. A larger angle applies a stronger distance penalty. The
`include_percentile` parameter limits the influence of extreme overlap
intensities and distances.

If the overlap set is too small, the regression line is degenerate, no
positive regression slope can be fit, or the percentile-based scale cannot be
calculated, VIPP raises an error instead of returning a misleading index.

## Object-Restricted Colocalization

`Object Colocalization Metrics` accepts a label image and two matching channel
images. Positive label IDs define the objects. Binary label inputs are
converted to connected-component labels per spatial block, so separate
timepoints or channels are not accidentally connected through leading axes.

VIPP processes each leading non-spatial axis independently. For example, a
`TZYX` label image produces rows with `t_index` plus `label_id`. These identity
columns allow object colocalization tables to merge cleanly with `Measure
Objects` and `Measure Objects + Intensity`.

For each label object, VIPP reports the same metric family used by
whole-image/ROI analysis, but restricted to the object's voxels:

- object voxel count;
- threshold-positive voxel counts for each channel;
- colocalized voxel count and fraction;
- Pearson correlation over the object;
- Pearson correlation over colocalized voxels;
- Manders M1 and M2;
- intensity overlap coefficients;
- threshold-positive and colocalized intensity sums;
- Costes diagnostics when Costes thresholding is used.

Manual thresholds are reused for all objects. Costes thresholds are calculated
once over all labeled foreground voxels, then reused for all objects.

## Label Overlap Association

`Label Overlap Association` accepts two matching label-like images: a
reference label image and a target label image. Binary inputs are converted to
connected-component labels per spatial block. Positive label IDs are treated
as objects; background is zero.

For every reference/target label pair with at least one shared voxel, VIPP
outputs one row with:

- `label_id`: reference label ID;
- `target_label_id`: target label ID;
- `reference_voxels`;
- `target_voxels`;
- `overlap_voxels`;
- `reference_overlap_fraction = overlap_voxels / reference_voxels`;
- `target_overlap_fraction = overlap_voxels / target_voxels`;
- `intersection_over_union = overlap_voxels /
  (reference_voxels + target_voxels - overlap_voxels)`.

The node reports only overlapping pairs. Non-overlapping labels can still be
summarized by combining this table with object measurement tables.

## Nearest-Object Distance

`Nearest Object Distance` accepts reference and target label-like images. For
each reference label, VIPP calculates the centroid of the reference object and
the centroid of each target object using voxel coordinates. It then reports
the nearest target label and the Euclidean centroid-to-centroid distance in
pixels.

When axis scales and physical units are available, VIPP also reports
`centroid_distance_physical`, calculated after multiplying each spatial-axis
coordinate difference by the corresponding physical scale. If no target labels
exist in the spatial block, `nearest_label_id` is `0` and distances are `NaN`.

Nearest-centroid distance is an association heuristic. It does not imply
overlap, containment, or biological interaction without additional context.

## Event Localization

`Event Localization` assigns event or puncta objects to labels, masks, or ROI
regions. The first input defines events. Binary event inputs are converted to
connected-component labels per spatial block, yielding one event row per
connected component. Integer event labels are used directly.

The second input defines regions. Integer labels are used directly. Binary
region masks are treated as a single ROI with region label ID `1`, rather than
as separate connected components.

For each event, VIPP counts the overlap with each positive region label. The
reported `region_label_id` is the region with the largest overlap. Ties are
resolved by the smaller region ID. If the event overlaps no positive region,
`region_label_id` is `0`, `overlap_voxels` is `0`, and `in_region` is `False`.

The table reports:

- `event_id`;
- `event_voxels`;
- `region_label_id`;
- `overlap_voxels`;
- `event_overlap_fraction = overlap_voxels / event_voxels`;
- `in_region`.

This node is suitable for puncta-in-cell, foci-in-nucleus, event-in-ROI, or
similar localization summaries where event objects and region masks/labels are
already defined.

## Reporting Recommendations

When reporting VIPP colocalization results, include:

- the input channels and any preprocessing steps;
- the analysis population: whole image, ROI mask, or objects;
- the threshold mode and final threshold values;
- whether intensities were normalized or clipped, as reported by
  `normalization_warnings`;
- the spatial dimensionality and whether leading axes such as time or channel
  were analyzed independently;
- for object tables, the label-generation method and merge keys used;
- for RACC-like outputs, `theta`, `include_percentile`, threshold mode, and
  output dtype;
- for object association, whether association is based on overlap, nearest
  centroid distance, or dominant event-region overlap.

## Validation Status

The implementation is covered by automated tests for whole-image/ROI metrics,
Costes threshold write-back, inspector scatter interaction, RACC outputs,
object-level metrics, label-overlap association, nearest-object distances,
event localization, and the bundled example workflows.

The bundled synthetic samples are deterministic and intended for software
regression and workflow demonstration. They do not replace biological
validation on real microscopy datasets. A manuscript should still include
representative biological datasets, parameter mappings, and comparison results
against relevant reference tools when making external numerical or biological
claims.
