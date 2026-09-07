"""Scientific-domain limits for interactive node controls, not model repair.

Core operations still validate loaded, scripted and batch values. Slider ranges
may be ergonomic windows; only known hard limits and linked ranges constrain
the wider numeric editor here. Merely rendering never changes authored values.
"""

from __future__ import annotations

import math
from dataclasses import replace
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from napari_vipp.ui.controls import ParameterBounds

# (lower parameter, upper parameter, strictly ordered). Do not infer these from
# names: label maxima have sentinel semantics rather than an ordinary ordered
# interval. Rescale inversion is an explicit boolean, not reversed bounds.
ORDERED_PARAMETERS = {
    "binary_threshold": (("low_threshold", "high_threshold", False),),
    "rescale_intensity": (
        ("in_low_percentile", "in_high_percentile", False),
        ("in_low_value", "in_high_value", False),
        ("out_min", "out_max", False),
    ),
    "normalize_image": (("low_percentile", "high_percentile", True),),
    "clip_intensity": (("minimum", "maximum", False),),
    "filter_mesh_objects": (("minimum", "maximum", False),),
    "difference_of_gaussians": (("low_sigma", "high_sigma", True),),
    "canny_edges": (("low_quantile", "high_quantile", False),),
    "hysteresis_threshold": (("low_threshold", "high_threshold", False),),
    "intensity_histogram": (("custom_min", "custom_max", True),),
    "born_wolf_psf": (("numerical_aperture", "refractive_index", True),),
}

HARD_RANGE_NAMES = frozenset(
    {
        "histogram_bins",
        "bin_count",
        "max_iterations",
        "input_count",
        "minimum_pixel_fraction",
        "low_quantile",
        "high_quantile",
        "in_low_percentile",
        "in_high_percentile",
        "low_percentile",
        "high_percentile",
        "range_percentile",
        "include_percentile",
        "theta_degrees",
        "bins",
        "output_size",
    }
)


def _entry_limits(bounds):
    minimum = bounds.minimum if bounds.entry_minimum is None else bounds.entry_minimum
    maximum = bounds.maximum if bounds.entry_maximum is None else bounds.entry_maximum
    if bounds.expandable:
        if bounds.entry_minimum is None and minimum < 0:
            minimum = min(minimum, -1_000_000)
        if bounds.entry_maximum is None:
            maximum = max(maximum, 1_000_000)
    return minimum, maximum


def _intersect(bounds, minimum, maximum):
    entry_min, entry_max = _entry_limits(bounds)
    entry_min, entry_max = max(entry_min, minimum), min(entry_max, maximum)
    if entry_min > entry_max:
        # An invalid imported peer can leave no possible value for this field.
        # Keep its correction editor usable; do not repair either model value.
        return bounds
    slider_min = min(max(bounds.minimum, entry_min), entry_max)
    slider_max = max(min(bounds.maximum, entry_max), slider_min)
    return replace(
        bounds,
        minimum=slider_min,
        maximum=slider_max,
        entry_minimum=entry_min,
        entry_maximum=entry_max,
    )


def _representable_limit(peer, decimals, *, upper, strict):
    # Round toward the valid side, including peers loaded at higher precision
    # than this editor. Decimal avoids 98.99 becoming 98.99000000000001.
    quantum = Decimal(1).scaleb(-decimals)
    value = Decimal(str(peer)) / quantum
    rounding = ROUND_FLOOR if upper else ROUND_CEILING
    units = value.to_integral_value(rounding=rounding)
    if strict and units == value:
        units += -1 if upper else 1
    result = float(units * quantum)
    if strict and (result >= peer if upper else result <= peer):
        result = math.nextafter(peer, -math.inf if upper else math.inf)
    return result


def constrain_parameter_bounds(
    operation_id,
    spec,
    params,
    bounds: ParameterBounds,
) -> ParameterBounds:
    """Intersect a numeric control with known valid limits and peer values."""
    if spec.kind not in {"int", "float"}:
        return bounds
    if spec.name in HARD_RANGE_NAMES:
        bounds = _intersect(bounds, spec.minimum, spec.maximum)
    if operation_id in {"smooth_mesh", "simplify_mesh"}:
        bounds = _intersect(bounds, spec.minimum, spec.maximum)
    if operation_id == "filter_mesh_objects" and spec.name in {"minimum", "maximum"}:
        bounds = _intersect(bounds, 0, math.inf)
    if (
        operation_id == "intensity_histogram"
        and params.get("bin_spacing") == "Logarithmic"
        and spec.name in {"custom_min", "custom_max"}
    ):
        bounds = _intersect(bounds, 10**-bounds.decimals, math.inf)
    if (
        operation_id in {"h_maxima_markers", "auto_watershed_from_mask"}
        and spec.name == "h"
    ):
        bounds = _intersect(bounds, 0, math.inf)
    for low, high, strict in ORDERED_PARAMETERS.get(operation_id, ()):
        if spec.name not in {low, high}:
            continue
        upper = spec.name == low
        peer = params.get(high if upper else low)
        if not isinstance(peer, (int, float)) or not math.isfinite(peer):
            continue
        if operation_id == "born_wolf_psf" and peer <= 0:
            # Zero means unresolved/auto, not a physical refractive index or NA.
            continue
        limit = _representable_limit(
            peer,
            bounds.decimals,
            upper=upper,
            strict=strict,
        )
        bounds = _intersect(
            bounds,
            -math.inf if upper else limit,
            limit if upper else math.inf,
        )
    return bounds


def has_linked_bounds(operation_id, name):
    if operation_id == "intensity_histogram" and name == "bin_spacing":
        return True
    return any(
        name in (low, high) for low, high, _ in ORDERED_PARAMETERS.get(operation_id, ())
    )
