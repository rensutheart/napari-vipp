"""Scalar YX/ZYX normalized template correlation and deterministic detections.

Only complete, supplied-size/orientation template placements are searched.
Scores are signed resemblance, not probabilities or segmentation boundaries.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import math
from dataclasses import replace
from numbers import Integral, Real

import numpy as np
from scipy import ndimage
from scipy.fft import next_fast_len
from skimage.feature import match_template

from napari_vipp.core.detection_metadata import DetectionMetadata, TemplateMatchMetadata
from napari_vipp.core.grid import (
    _unit_dimension_and_factor,
    validate_aligned_image_states,
)
from napari_vipp.core.host_memory import capture_host_memory, preflight_host_allocation
from napari_vipp.core.metadata import ImageState, image_state_from_array
from napari_vipp.core.tables import TableData


def _progress(context, current=0, total=1, message=""):
    if context is not None:
        context.check_cancelled()
        context.report(current, total, message)


def _guard_memory(required, purpose):
    decision = preflight_host_allocation(
        capture_host_memory(), required_bytes=int(required), purpose=purpose
    )
    if not decision.allowed:
        raise MemoryError(decision.reason)


def template_match_required_bytes(search_shape, template_shape):
    """Conservative additional bytes including padded FFT and backend workspace.

    Inputs are already resident. This is an admission estimate, not a measured
    peak-memory claim; use complete padded FFT grids even for real FFT storage.
    """
    padded = tuple(s + 2 * t for s, t in zip(search_shape, template_shape, strict=True))
    fft_shape = tuple(
        next_fast_len(s + t - 1) for s, t in zip(padded, template_shape, strict=True)
    )
    return (
        192 * math.prod(fft_shape)
        + 96 * math.prod(padded)
        + 64 * math.prod(search_shape)
        + 32 * math.prod(template_shape)
    )


def template_match_grid_contract(
    search_shape, template_shape, search_state, template_state
):
    """Validate metadata-only planning inputs; return score shape and axes.

    This deliberately does not inspect or allocate image values. Value-domain
    checks (finite intensities and nonconstant template) remain execution gates.
    """
    for shape, state, label in (
        (search_shape, search_state, "Search"),
        (template_shape, template_state, "Template"),
    ):
        if (
            not isinstance(state, ImageState)
            or not state.axes_explicit
            or tuple(shape) != state.shape
            or len(state.axes) != len(shape)
        ):
            raise ValueError(
                f"{label} requires explicit axes and matching carried shape metadata."
            )
        if tuple(axis.name.lower() for axis in state.axes) not in (
            ("y", "x"),
            ("z", "y", "x"),
        ) or any(axis.type != "space" for axis in state.axes):
            raise ValueError(
                "Template Match requires one scalar YX image or ZYX volume; "
                "select channel/time and reorder axes explicitly upstream."
            )
        if (
            not all(
                isinstance(size, Integral) and not isinstance(size, bool) and size > 0
                for size in shape
            )
            or np.dtype(state.dtype).kind not in "biuf"
            or (
                np.dtype(state.dtype).kind == "f" and np.dtype(state.dtype).itemsize > 8
            )
        ):
            raise ValueError(
                "Template Match requires nonempty real numeric images "
                "with floating-point precision at most float64."
            )
        dimensions = tuple(_unit_dimension_and_factor(axis.unit) for axis in state.axes)
        if len({item[0] for item in dimensions}) != 1 or dimensions[0][0] not in (
            "index",
            "length_micrometer",
        ):
            raise ValueError(
                "Template Match spatial units must all be compatible lengths "
                "or pixel coordinates."
            )
    if len(search_shape) != len(template_shape) or any(
        t > s for s, t in zip(search_shape, template_shape, strict=True)
    ):
        raise ValueError(
            "Template Match needs equal rank and a template no larger "
            "than the search image."
        )
    for left, right in zip(search_state.axes, template_state.axes, strict=True):
        lu, lf = _unit_dimension_and_factor(left.unit)
        ru, rf = _unit_dimension_and_factor(right.unit)
        values = (
            left.scale * lf,
            right.scale * rf,
            left.translation * lf,
            right.translation * rf,
        )
        if not all(math.isfinite(value) for value in values) or min(values[:2]) <= 0:
            raise ValueError(
                "Template Match calibration cannot be represented in canonical units."
            )
        if lu != ru or not math.isclose(values[0], values[1], rel_tol=1e-9, abs_tol=0):
            raise ValueError(
                "Template Match requires equal spatial sampling and compatible "
                "units; resample explicitly first."
            )
    shape = tuple(s - t + 1 for s, t in zip(search_shape, template_shape, strict=True))
    axes = tuple(
        replace(axis, translation=axis.translation + (size - 1) / 2 * axis.scale)
        for axis, size in zip(search_state.axes, template_shape, strict=True)
    )
    return shape, axes


def _array(image, state, label, context=None):
    array = np.asarray(image)
    if (
        not isinstance(state, ImageState)
        or not state.axes_explicit
        or tuple(array.shape) != state.shape
        or array.dtype.name != state.dtype
        or len(state.axes) != array.ndim
    ):
        raise ValueError(
            f"{label} requires explicit axes and matching carried shape/dtype metadata."
        )
    names = tuple(axis.name.lower() for axis in state.axes)
    if names not in (("y", "x"), ("z", "y", "x")) or any(
        axis.type != "space" for axis in state.axes
    ):
        raise ValueError(
            f"{label} requires one scalar YX image or ZYX volume. "
            "Select channel/time and reorder axes explicitly upstream."
        )
    if array.dtype.kind not in "biuf" or not array.size:
        raise ValueError(f"{label} requires a nonempty real numeric image.")
    if array.dtype.kind == "f" and array.dtype.itemsize > 8:
        raise ValueError(
            f"{label} supports floating-point inputs up to float64. "
            "Explicitly convert extended-precision input first."
        )
    dimensions = tuple(_unit_dimension_and_factor(axis.unit) for axis in state.axes)
    if len({item[0] for item in dimensions}) != 1 or dimensions[0][0] not in (
        "index",
        "length_micrometer",
    ):
        raise ValueError(
            f"{label} spatial units must all be compatible lengths "
            "or pixel coordinates."
        )
    for axis, (_, factor) in zip(state.axes, dimensions, strict=True):
        if (
            not math.isfinite(axis.scale * factor)
            or axis.scale * factor <= 0
            or not math.isfinite(axis.translation * factor)
        ):
            raise ValueError(
                f"{label} calibration cannot be represented in canonical units."
            )
    for block in np.nditer(
        array,
        flags=["external_loop", "buffered"],
        op_flags=["readonly"],
        buffersize=131072,
    ):
        if context is not None:
            context.check_cancelled()
        if not np.isfinite(block).all():
            raise ValueError(
                f"{label} requires finite values; resolve NaN/Inf explicitly first."
            )
        if array.dtype.kind in "iu" and (
            int(block.min()) < -(2**53) or int(block.max()) > 2**53
        ):
            raise ValueError(
                f"{label} cannot exactly represent these wide integer intensities. "
                "Explicitly convert an estimation image first."
            )
    return array


def _revision(array, context=None):
    digest = hashlib.sha256(f"{array.dtype.str}:{array.shape}:C".encode("ascii"))
    for block in np.nditer(
        array,
        flags=["external_loop", "buffered"],
        op_flags=["readonly"],
        order="C",
        buffersize=131072,
    ):
        if context is not None:
            context.check_cancelled()
        digest.update(block.tobytes(order="C"))
    return digest.hexdigest()


def _frame(state, revision):
    return state.source.source_uuid or (
        f"{state.source.uri}#series={state.source.series_index}"
        if state.source.uri
        else f"sha256:{revision}"
    )


def _condition(array):
    """Positive affine conditioning intrinsic to ZNCC; never changes inputs."""
    result = array.astype(np.float64, copy=True)
    low, high = float(result.min()), float(result.max())
    if low == high:
        return np.zeros_like(result)
    result -= low / 2 + high / 2
    result /= max(abs(float(result.min())), abs(float(result.max())))
    return result


def _repair_ill_conditioned_windows(
    search, template, scores, valid, low, high, context
):
    """Use bounded direct Pearson windows where integral/FFT error can dominate.

    The conservative trigger compares local range to global amplitude using a
    roundoff allowance for the padded integral-sum population. It changes the
    numerical evaluation, never the definition, threshold, or sampled population.
    """
    amplitude = max(abs(float(search.min())), abs(float(search.max())))
    if amplitude == 0:
        return
    contrast = high.astype(np.float64) / amplitude - low.astype(np.float64) / amplitude
    padded_count = math.prod(
        s + 2 * t for s, t in zip(search.shape, template.shape, strict=True)
    )
    allowance = math.sqrt(64 * np.finfo(np.float64).eps * padded_count)
    repair = valid & (
        (contrast <= allowance) | ~np.isfinite(scores) | (np.abs(scores) > 1 + 1e-10)
    )
    indices = np.flatnonzero(repair)
    if not len(indices):
        return
    centered_template = _condition(template)
    centered_template -= centered_template.mean()
    template_norm = np.linalg.norm(centered_template)
    for index, flat in enumerate(indices):
        if index % 256 == 0:
            _progress(
                context,
                2,
                4,
                "Direct correlation for ill-conditioned windows: "
                f"{index}/{len(indices)}",
            )
        coordinate = np.unravel_index(flat, scores.shape)
        window = search[
            tuple(
                slice(start, start + width)
                for start, width in zip(coordinate, template.shape, strict=True)
            )
        ]
        centered_window = _condition(window)
        centered_window -= centered_window.mean()
        denominator = np.linalg.norm(centered_window) * template_norm
        if not denominator or not np.isfinite(denominator):
            raise ValueError(
                "Template correlation cannot resolve this window's intensity variation."
            )
        scores[coordinate] = np.sum(centered_window * centered_template) / denominator


def template_match(
    search, template, *, search_state, template_state, progress_context=None
):
    """Return scores, Boolean validity, and their centre-calibrated image states."""
    scores, valid = template_match_arrays(
        search,
        template,
        search_state=search_state,
        template_state=template_state,
        progress_context=progress_context,
    )
    score_state, mask_state = template_match_states_from_outputs(
        search,
        template,
        search_state,
        template_state,
        scores,
        valid,
        progress_context=progress_context,
    )
    return scores, valid, score_state, mask_state


def template_match_arrays(
    search, template, *, search_state, template_state, progress_context=None
):
    """Scientific two-array kernel shared by the graph and direct API."""
    _progress(progress_context, 0, 4, "Validating scalar template inputs")
    search = _array(search, search_state, "Template Match search", progress_context)
    template = _array(
        template, template_state, "Template Match template", progress_context
    )
    template_match_grid_contract(
        search.shape, template.shape, search_state, template_state
    )
    if np.min(template) == np.max(template):
        raise ValueError("Template Match requires a nonconstant template.")
    _guard_memory(
        template_match_required_bytes(search.shape, template.shape),
        "Template Match padded FFT and validity workspace",
    )
    _progress(
        progress_context,
        1,
        4,
        "Normalized correlation (native FFT cannot be interrupted)",
    )
    conditioned_search, conditioned_template = _condition(search), _condition(template)
    scores = match_template(conditioned_search, conditioned_template, pad_input=False)
    _progress(
        progress_context,
        2,
        4,
        "Checking complete-placement variance and score validity",
    )
    # Extrema test is exact for accepted numeric inputs, unlike subtraction of
    # integral squared sums. It does not declare backend placeholder zeros valid.
    start = tuple(size // 2 for size in template.shape)
    crop = tuple(slice(i, i + n) for i, n in zip(start, scores.shape, strict=True))
    low = ndimage.minimum_filter(search, size=template.shape, mode="nearest")[crop]
    high = ndimage.maximum_filter(search, size=template.shape, mode="nearest")[crop]
    valid = np.array(low != high, dtype=bool, copy=True)
    _repair_ill_conditioned_windows(
        search, template, scores, valid, low, high, progress_context
    )
    if not np.isfinite(scores).all() or np.any(np.abs(scores[valid]) > 1 + 1e-7):
        raise ValueError(
            "Template correlation was numerically unstable; "
            "explicitly rescale the input dynamic range."
        )
    scores = np.array(np.clip(scores, -1, 1), dtype=np.float64, copy=True)
    scores[~valid] = 0.0
    _progress(progress_context, 3, 4, "Template correlation complete")
    return scores, valid


def template_match_states_from_outputs(
    search,
    template,
    search_state,
    template_state,
    scores,
    valid,
    progress_context=None,
):
    """Create paired state/revision evidence after the host executor returns."""
    search, template = np.asarray(search), np.asarray(template)
    shape, _ = template_match_grid_contract(
        search.shape, template.shape, search_state, template_state
    )
    if (
        scores.shape != shape
        or valid.shape != shape
        or scores.dtype != np.dtype(np.float64)
        or valid.dtype != np.dtype(bool)
    ):
        raise ValueError("Template Match output data violates its score-grid contract.")
    search_revision = _revision(search, progress_context)
    metadata = TemplateMatchMetadata(
        axes=tuple(axis.name.lower() for axis in search_state.axes),
        search_shape=search.shape,
        template_shape=template.shape,
        search_scale=tuple(axis.scale for axis in search_state.axes),
        search_origin=tuple(axis.translation for axis in search_state.axes),
        search_units=tuple(axis.unit for axis in search_state.axes),
        source_frame=_frame(search_state, search_revision),
        search_revision=search_revision,
        template_revision=_revision(template, progress_context),
        score_revision=_revision(scores, progress_context),
        valid_mask_revision=_revision(valid, progress_context),
        implementation=(
            f"scikit-image {importlib.metadata.version('scikit-image')} "
            "match_template; VIPP detection v1"
        ),
    )
    score_state, mask_state = template_match_output_states(
        search_state, template_state, scores, valid, metadata=metadata
    )
    _progress(progress_context, 4, 4, "Template matching complete")
    return score_state, mask_state


def template_match_output_states(
    search_state, template_state, scores, valid, *, metadata
):
    """Build both states on the reduced complete-placement centre lattice."""
    axes = tuple(
        replace(axis, translation=axis.translation + offset * axis.scale)
        for axis, offset in zip(search_state.axes, metadata.center_offset, strict=True)
    )
    history = (
        f"Template Match: supplied size {metadata.template_shape} and orientation; "
        "complete placements; signed ZNCC, not probability; "
        "positive affine float64 numerical conditioning; "
        "direct Pearson fallback for ill-conditioned windows; "
        f"template={template_state.source_name!r}",
        metadata.implementation,
    )
    state = image_state_from_array(
        scores,
        axes=axes,
        source_name=search_state.source_name,
        source=search_state.source,
        acquisition=search_state.acquisition,
        history=history,
        metadata_source="VIPP template-center score grid",
        defer_statistics=True,
    )
    score_state = replace(
        state, kind="template match score image", template_match_metadata=metadata
    )
    mask_state = replace(
        image_state_from_array(
            valid,
            axes=axes,
            source_name=search_state.source_name,
            source=search_state.source,
            history=history,
            metadata_source="VIPP template-center validity grid",
            defer_statistics=True,
        ),
        kind="binary mask",
        template_match_metadata=metadata,
    )
    return score_state, mask_state


def _real(value, label, minimum=None):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or (minimum is not None and value < minimum)
    ):
        raise ValueError(
            f"{label} must be finite{' and non-negative' if minimum == 0 else ''}."
        )
    return float(value)


def _integer(value, label, minimum):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < minimum
    ):
        raise ValueError(f"{label} must be an integer >= {minimum}.")
    return int(value)


def _candidate_indices(array, valid, minimum, border, context):
    # Masked samples are absent, not zeros that could suppress negative maxima.
    values = np.where(valid, array, -np.inf)
    local = ndimage.maximum_filter(values, size=3, mode="constant", cval=-np.inf)
    candidates = valid & (array >= minimum) & (values == local)
    if border:
        for axis in range(array.ndim):
            first, last = [slice(None)] * array.ndim, [slice(None)] * array.ndim
            first[axis], last[axis] = (
                slice(0, border),
                slice(max(0, array.shape[axis] - border), None),
            )
            candidates[tuple(first)] = False
            candidates[tuple(last)] = False
    _progress(context, 1, 3, "Resolving local-maximum plateaus")
    labels, _ = ndimage.label(
        candidates, structure=np.ones((3,) * array.ndim, dtype=bool)
    )
    indices = np.flatnonzero(candidates)
    if not len(indices):
        return np.empty((0, array.ndim), dtype=np.int64)
    # Adjacent candidate maxima necessarily have equal values. One lexical
    # first representative is retained per full-connectivity plateau.
    _, first = np.unique(labels.reshape(-1)[indices], return_index=True)
    indices = indices[first]
    order = np.lexsort((indices, -array.reshape(-1)[indices].astype(np.float64)))
    return np.column_stack(np.unravel_index(indices[order], array.shape))


def _suppress(coordinates, scales, separation, maximum, context):
    if separation <= min(scales):
        return coordinates[:maximum], len(coordinates)
    buckets = {}
    retained = []
    accepted = 0
    offsets = tuple(itertools.product((-1, 0, 1), repeat=len(scales)))
    for index, coordinate in enumerate(coordinates):
        if index % 1024 == 0:
            _progress(
                context,
                2,
                3,
                f"Checking minimum separation: {index}/{len(coordinates)} candidates",
            )
        location = coordinate * scales
        bucket = tuple(math.floor(value / separation) for value in location)
        suppressed = False
        for offset in offsets:
            for other in buckets.get(
                tuple(v + d for v, d in zip(bucket, offset, strict=True)), ()
            ):
                if math.dist(location, other) < separation:
                    suppressed = True
                    break
            if suppressed:
                break
        if not suppressed:
            buckets.setdefault(bucket, []).append(location)
            accepted += 1
            if len(retained) < maximum:
                retained.append(coordinate)
    return np.asarray(retained, dtype=np.int64).reshape(-1, len(scales)), accepted


def find_peaks(
    image,
    mask=None,
    *,
    image_state,
    mask_state=None,
    minimum_value=0.5,
    minimum_separation=1.0,
    separation_units="Pixels",
    maximum_detections=1000,
    border_exclusion=0,
    progress_context=None,
):
    """Return deterministic, table-first detections in source-centre coordinates.

    Threshold is inclusive. Plateaus use full 3^rank connectivity and their
    lexicographically first coordinate. Order is decreasing score then coordinate.
    Greedy Euclidean suppression rejects distances strictly below the separation;
    the result cap is applied last with exact pre-cap population evidence.
    """
    minimum_value = _real(minimum_value, "Minimum value")
    minimum_separation = _real(minimum_separation, "Minimum separation", 0)
    maximum_detections = _integer(maximum_detections, "Maximum detections", 1)
    border_exclusion = _integer(border_exclusion, "Border exclusion", 0)
    if separation_units not in ("Pixels", "Physical (micrometers)"):
        raise ValueError("Choose Pixels or Physical (micrometers) separation.")
    _progress(progress_context, 0, 3, "Validating peak inputs and coordinates")
    array = _array(image, image_state, "Find Peaks image", progress_context)
    # Includes the optional all-valid mask: admit before any volume workspace.
    _guard_memory(
        array.size * (256 + 64 * array.ndim)
        + min(array.size, maximum_detections) * (512 + 256 * array.ndim),
        "Find Peaks local maxima, sorting, suppression and table workspace",
    )
    metadata = image_state.template_match_metadata
    if metadata is None and (
        "template match" in image_state.kind.lower()
        or any(step.startswith("Template Match:") for step in image_state.history)
    ):
        raise ValueError(
            "Template response coordinate/validity metadata is missing. "
            "Recalculate Template Match and connect its matching valid-score mask."
        )
    if metadata is not None and mask is None:
        raise ValueError(
            "Template Match scores require their matching valid-score mask. "
            "Enable Use valid mask and connect it."
        )
    if mask is not None:
        valid = _array(mask, mask_state, "Find Peaks mask", progress_context)
        if valid.dtype != np.dtype(bool):
            raise ValueError(
                "Find Peaks valid mask must be Boolean; no implicit thresholding."
            )
        validate_aligned_image_states(
            (image_state, mask_state), operation_title="Find Peaks"
        )
    else:
        if mask_state is not None:
            raise ValueError("A mask state was supplied without mask data.")
        valid = np.ones(array.shape, dtype=bool)
    revision = _revision(array, progress_context)
    if metadata is not None:
        if (
            not isinstance(metadata, TemplateMatchMetadata)
            or mask_state.template_match_metadata != metadata
            or array.shape != metadata.score_shape
            or revision != metadata.score_revision
            or _revision(valid, progress_context) != metadata.valid_mask_revision
        ):
            raise ValueError(
                "Template scores and valid-score mask do not belong to the same "
                "unchanged result. Recalculate and reconnect both outputs."
            )
        for index, axis in enumerate(image_state.axes):
            dimension, factor = _unit_dimension_and_factor(axis.unit)
            expected_dimension, expected_factor = _unit_dimension_and_factor(
                metadata.search_units[index]
            )
            expected_scale = metadata.search_scale[index] * expected_factor
            expected_origin = (
                metadata.search_origin[index]
                + metadata.center_offset[index] * metadata.search_scale[index]
            ) * expected_factor
            if (
                axis.name.lower() != metadata.axes[index]
                or dimension != expected_dimension
                or not math.isclose(
                    axis.scale * factor, expected_scale, rel_tol=1e-12, abs_tol=0
                )
                or not math.isclose(
                    axis.translation * factor,
                    expected_origin,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    "Template score calibration changed after matching. "
                    "Recalculate Template Match and reconnect both direct outputs."
                )
    units = tuple(_unit_dimension_and_factor(axis.unit) for axis in image_state.axes)
    calibrated = units[0][0] == "length_micrometer"
    if separation_units == "Physical (micrometers)" and not calibrated:
        raise ValueError(
            "Physical separation requires known length calibration "
            "on every spatial axis."
        )
    scales = np.asarray(
        [
            axis.scale * unit[1]
            for axis, unit in zip(image_state.axes, units, strict=True)
        ]
        if separation_units != "Pixels"
        else [1.0] * array.ndim
    )
    if not np.isfinite(np.asarray(array.shape) * scales).all():
        raise ValueError(
            "Peak coordinates exceed the finite physical-coordinate range."
        )
    coordinates = _candidate_indices(
        array, valid, minimum_value, border_exclusion, progress_context
    )
    selected, accepted = _suppress(
        coordinates, scales, minimum_separation, maximum_detections, progress_context
    )
    axes = tuple(axis.name.lower() for axis in image_state.axes)
    source_shape = metadata.search_shape if metadata is not None else image_state.shape
    source_scale = (
        metadata.search_scale
        if metadata is not None
        else tuple(axis.scale for axis in image_state.axes)
    )
    source_origin = (
        metadata.search_origin
        if metadata is not None
        else tuple(axis.translation for axis in image_state.axes)
    )
    source_units = (
        metadata.search_units
        if metadata is not None
        else tuple(axis.unit for axis in image_state.axes)
    )
    coordinate_columns = tuple(f"{axis}_index" for axis in axes)
    physical_columns = tuple(f"{axis}_physical" for axis in axes) if calibrated else ()
    bound_columns = (
        tuple(f"template_{axis}_{side}" for axis in axes for side in ("start", "stop"))
        if metadata is not None
        else ()
    )
    columns = (
        "detection_id",
        *coordinate_columns,
        *physical_columns,
        "score",
        *bound_columns,
    )
    rows = []
    for index, coordinate in enumerate(selected):
        if index % 1024 == 0 and progress_context is not None:
            progress_context.check_cancelled()
        center = coordinate.astype(float) + (
            np.asarray(metadata.center_offset) if metadata is not None else 0
        )
        physical = (
            tuple(float(v) for v in np.asarray(source_origin) + center * source_scale)
            if calibrated
            else ()
        )
        if physical and not np.isfinite(physical).all():
            raise ValueError(
                "Detection physical coordinates overflow; correct the calibration."
            )
        bounds = (
            tuple(
                value
                for start, width in zip(
                    coordinate, metadata.template_shape, strict=True
                )
                for value in (int(start), int(start + width))
            )
            if metadata is not None
            else ()
        )
        rows.append(
            (
                index + 1,
                *tuple(float(v) for v in center),
                *physical,
                float(array[tuple(coordinate)]),
                *bounds,
            )
        )
    evidence = DetectionMetadata(
        axes=axes,
        source_shape=source_shape,
        source_scale=source_scale,
        source_origin=source_origin,
        source_units=source_units,
        source_frame=metadata.source_frame
        if metadata is not None
        else _frame(image_state, revision),
        coordinate_columns=coordinate_columns,
        score_column="score",
        template_shape=metadata.template_shape if metadata is not None else None,
        minimum_value=minimum_value,
        minimum_separation=minimum_separation,
        separation_units=separation_units,
        maximum_detections=maximum_detections,
        border_exclusion=border_exclusion,
        candidate_count=len(coordinates),
        accepted_count=accepted,
        returned_count=len(rows),
        truncated=accepted > len(rows),
        source_revision=metadata.search_revision if metadata is not None else revision,
        implementation=(
            "VIPP deterministic local maxima v1; full-neighborhood plateaus; "
            "score-descending/lexical ties; strict Euclidean separation; cap last"
        ),
    )
    table = TableData(
        columns=columns,
        rows=tuple(rows),
        name="Detections",
        table_kind="detections",
        source_name=image_state.source_name,
        column_units=tuple(
            (column, "pixel") for column in (*coordinate_columns, *bound_columns)
        )
        + tuple(zip(physical_columns, source_units, strict=True))
        if calibrated
        else tuple(
            (column, "pixel") for column in (*coordinate_columns, *bound_columns)
        ),
        detection_metadata=evidence,
    )
    _progress(
        progress_context,
        3,
        3,
        "Peak detection complete"
        + (
            f"; capped at {maximum_detections} of {accepted}"
            if evidence.truncated
            else ""
        ),
    )
    return table
