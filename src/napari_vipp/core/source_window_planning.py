"""Qt-free planning for exact Image Source to Crop Stack window reads.

This module contains policy and geometry only.  It never opens a source and it
never mutates a graph.  A caller can therefore use the same decision in the
interactive UI, batch preflight, exported execution, and the source loader.

The planner is intentionally narrow: a scientific source window is eligible
only for one exact, visible ``Image Source -> Crop Stack`` edge.  Any competing
consumer, tunnel, bypass, ambiguous spatial axis, or stale SourceItem causes a
typed refusal and leaves full-source loading as the only legal interpretation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from numbers import Integral, Real

import numpy as np

from napari_vipp.core.metadata import AxisDeclaration, ImageState
from napari_vipp.core.operations import crop_stack_selection
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.source_item_persistence import source_item_from_params
from napari_vipp.core.source_items import SourceItem
from napari_vipp.core.source_window import (
    SourceWindowRequest,
    estimate_exact_window_read,
)


class SourceWindowPlanReason(StrEnum):
    """Stable reasons returned by :func:`plan_exact_source_crop_window`."""

    ELIGIBLE = "eligible"
    UNKNOWN_SOURCE_NODE = "unknown_source_node"
    NOT_IMAGE_SOURCE = "not_image_source"
    SOURCE_ITEM_MISMATCH = "source_item_mismatch"
    SOURCE_CONTRACT_MISMATCH = "source_contract_mismatch"
    EXACT_REGION_UNAVAILABLE = "exact_region_unavailable"
    SOURCE_TUNNEL_PRESENT = "source_tunnel_present"
    DIRECT_TOPOLOGY_REQUIRED = "direct_topology_required"
    CROP_BYPASSED = "crop_bypassed"
    CROP_GEOMETRY_INVALID = "crop_geometry_invalid"


@dataclass(frozen=True, slots=True)
class CropWindowMargins:
    """Authored Crop Stack margins in semantic spatial coordinates."""

    z_start: int = 0
    z_end: int = 0
    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0

    def __post_init__(self) -> None:
        for name in (
            "z_start",
            "z_end",
            "top",
            "bottom",
            "left",
            "right",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_integer(getattr(self, name), label=name),
            )

    def as_params(self) -> dict[str, int]:
        """Return values accepted by the visible Crop Stack node."""

        return {
            "z_start": self.z_start,
            "z_end": self.z_end,
            "top": self.top,
            "bottom": self.bottom,
            "left": self.left,
            "right": self.right,
        }


@dataclass(frozen=True, slots=True)
class CropWindowGeometry:
    """Canonical full-rank level-0 selection for one Crop Stack."""

    source_shape: tuple[int, ...]
    axis_names: tuple[str, ...]
    selection: tuple[slice, ...]
    margins: CropWindowMargins
    z_axis: int | None
    y_axis: int
    x_axis: int

    def __post_init__(self) -> None:
        shape = tuple(int(size) for size in self.source_shape)
        axes = tuple(str(name).strip().casefold() for name in self.axis_names)
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("Crop window source dimensions must be positive.")
        if len(axes) != len(shape) or len(self.selection) != len(shape):
            raise ValueError(
                "Crop window shape, axes, and selection must have equal rank."
            )
        normalized = SourceWindowRequest(tuple(self.selection)).normalized_selection(
            shape
        )
        if not isinstance(self.margins, CropWindowMargins):
            raise TypeError("Crop window margins must be CropWindowMargins.")
        spatial_indices = tuple(
            index
            for index in (self.z_axis, self.y_axis, self.x_axis)
            if index is not None
        )
        if len(set(spatial_indices)) != len(spatial_indices) or any(
            not 0 <= int(index) < len(shape) for index in spatial_indices
        ):
            raise ValueError("Crop window spatial axis indices must be distinct.")
        object.__setattr__(self, "source_shape", shape)
        object.__setattr__(self, "axis_names", axes)
        object.__setattr__(self, "selection", normalized)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return tuple(
            int(selector.stop) - int(selector.start) for selector in self.selection
        )

    @property
    def bounds(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (int(selector.start), int(selector.stop)) for selector in self.selection
        )

    @property
    def requires_crop(self) -> bool:
        return self.output_shape != self.source_shape


@dataclass(frozen=True, slots=True)
class ExactSourceCropWindowPlan:
    """One graph-proven exact source-window request."""

    source_node_id: str
    crop_node_id: str
    source_item_digest: str
    geometry: CropWindowGeometry
    request: SourceWindowRequest
    decoded_output_bytes: int

    def __post_init__(self) -> None:
        if not str(self.source_node_id) or not str(self.crop_node_id):
            raise ValueError("A source-window plan requires source and crop nodes.")
        if not str(self.source_item_digest):
            raise ValueError("A source-window plan requires a SourceItem digest.")
        if not isinstance(self.geometry, CropWindowGeometry):
            raise TypeError("A source-window plan requires CropWindowGeometry.")
        if not isinstance(self.request, SourceWindowRequest):
            raise TypeError("A source-window plan requires SourceWindowRequest.")
        if self.request.normalized_selection(self.geometry.source_shape) != (
            self.geometry.selection
        ):
            raise ValueError("The source-window request and crop geometry disagree.")
        _nonnegative_integer(self.decoded_output_bytes, label="decoded_output_bytes")


@dataclass(frozen=True, slots=True)
class SourceCropWindowDecision:
    """Typed eligibility result that never mutates or executes the graph."""

    reason_code: SourceWindowPlanReason
    reason: str
    plan: ExactSourceCropWindowPlan | None = None

    def __post_init__(self) -> None:
        code = SourceWindowPlanReason(self.reason_code)
        reason = str(self.reason).strip()
        if not reason:
            raise ValueError("A source-window decision requires a reason.")
        if (code is SourceWindowPlanReason.ELIGIBLE) != (self.plan is not None):
            raise ValueError("Only an eligible source-window decision has a plan.")
        object.__setattr__(self, "reason_code", code)
        object.__setattr__(self, "reason", reason)

    @property
    def eligible(self) -> bool:
        return self.plan is not None


@dataclass(frozen=True, slots=True)
class CenteredCropSuggestion:
    """Conservative, content-agnostic crop sized to one host byte budget."""

    geometry: CropWindowGeometry
    full_decoded_bytes: int
    decoded_output_bytes: int
    estimated_peak_bytes: int
    available_byte_budget: int
    planned_byte_budget: int
    utilization_fraction: float
    peak_multiplier: float

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, CropWindowGeometry):
            raise TypeError("A fitted crop requires CropWindowGeometry.")
        for name in (
            "full_decoded_bytes",
            "decoded_output_bytes",
            "estimated_peak_bytes",
            "available_byte_budget",
            "planned_byte_budget",
        ):
            _nonnegative_integer(getattr(self, name), label=name)
        if self.estimated_peak_bytes > self.planned_byte_budget:
            raise ValueError("The fitted crop exceeds its planned byte budget.")
        if not 0.0 < float(self.utilization_fraction) <= 1.0:
            raise ValueError("Crop utilization_fraction must be in (0, 1].")
        if float(self.peak_multiplier) < 1.0:
            raise ValueError("Crop peak_multiplier must be at least 1.")

    @property
    def requires_crop(self) -> bool:
        return self.geometry.requires_crop

    @property
    def retained_fraction(self) -> float:
        if self.full_decoded_bytes <= 0:
            return 0.0
        return self.decoded_output_bytes / self.full_decoded_bytes


class CropFitUnavailableError(ValueError):
    """Raised when even the smallest semantics-preserving crop cannot fit."""


def derive_crop_window_geometry(
    params: Mapping[str, object],
    full_state: ImageState,
) -> CropWindowGeometry:
    """Derive the exact full-rank selection implemented by Crop Stack.

    Pushdown is stricter than the ordinary interactive crop: Y and X must be
    explicit spatial axes.  Z is cropped only when it is one explicit spatial
    axis.  Every other dimension, including time and channels, remains complete.
    """

    if not isinstance(full_state, ImageState):
        raise TypeError("Crop window planning requires a full ImageState.")
    shape = tuple(int(size) for size in full_state.shape)
    if not shape or any(size <= 0 for size in shape):
        raise ValueError("Crop window planning requires positive source dimensions.")
    if len(full_state.axes) != len(shape):
        raise ValueError("Crop window ImageState axes do not match its rank.")

    y_axis = _one_explicit_spatial_axis(full_state, "y", required=True)
    x_axis = _one_explicit_spatial_axis(full_state, "x", required=True)
    z_axis = _one_explicit_spatial_axis(full_state, "z", required=False)
    assert y_axis is not None and x_axis is not None
    if y_axis == x_axis:
        raise ValueError("Crop window Y and X axes must be distinct.")

    channel_axis = _crop_channel_axis(params.get("channel_axis", -1), len(shape))
    if channel_axis is not None and channel_axis in {y_axis, x_axis, z_axis}:
        raise ValueError(
            "Crop Stack channel-axis override conflicts with a spatial axis."
        )

    margins = CropWindowMargins(
        z_start=params.get("z_start", 0),
        z_end=params.get("z_end", 0),
        top=params.get("top", 0),
        bottom=params.get("bottom", 0),
        left=params.get("left", 0),
        right=params.get("right", 0),
    )
    if margins.z_start or margins.z_end:
        if z_axis is None:
            raise ValueError(
                "Z crop margins require one explicitly declared Z spatial axis."
            )

    # Give the operation helper only axis semantics proven above.  In
    # particular, an inferred Q-like leading axis must never acquire implicit Z
    # semantics merely because of its position, and a non-spatial axis carrying
    # a misleading x/y/z name must not affect geometry.
    runtime_axis_names = tuple(
        "y"
        if index == y_axis
        else "x"
        if index == x_axis
        else "z"
        if z_axis is not None and index == z_axis
        else (
            axis.name
            if axis.is_explicit and axis.name.strip().casefold() not in {"x", "y", "z"}
            else f"axis:{index}"
        )
        for index, axis in enumerate(full_state.axes)
    )
    selection = crop_stack_selection(
        shape,
        top=margins.top,
        bottom=margins.bottom,
        left=margins.left,
        right=margins.right,
        channel_axis=channel_axis,
        axis_names=runtime_axis_names,
        z_start=margins.z_start,
        z_end=margins.z_end,
        z_axis_explicit=z_axis is not None,
    )
    return CropWindowGeometry(
        source_shape=shape,
        axis_names=tuple(axis.name for axis in full_state.axes),
        selection=selection,
        margins=margins,
        z_axis=z_axis,
        y_axis=y_axis,
        x_axis=x_axis,
    )


def plan_exact_source_crop_window(
    pipeline: PrototypePipeline,
    source_node_id: str,
    source_item: SourceItem,
    full_state: ImageState,
) -> SourceCropWindowDecision:
    """Prove one exact direct Image Source -> Crop Stack pushdown."""

    if not isinstance(pipeline, PrototypePipeline):
        raise TypeError("source-window planning requires a PrototypePipeline.")
    if not isinstance(source_item, SourceItem):
        raise TypeError("source-window planning requires a SourceItem.")
    if not isinstance(full_state, ImageState):
        raise TypeError("source-window planning requires a full ImageState.")

    source_id = str(source_node_id)
    source = pipeline.nodes.get(source_id)
    if source is None:
        return _refusal(
            SourceWindowPlanReason.UNKNOWN_SOURCE_NODE,
            f"Image Source node {source_id!r} is not present in the graph.",
        )
    if source.operation_id != "input":
        return _refusal(
            SourceWindowPlanReason.NOT_IMAGE_SOURCE,
            f"Node {source_id!r} is not an Image Source.",
        )

    try:
        bound_item = source_item_from_params(source.params)
    except (TypeError, ValueError) as exc:
        return _refusal(
            SourceWindowPlanReason.SOURCE_ITEM_MISMATCH,
            f"The Image Source has invalid saved SourceItem evidence: {exc}",
        )
    if bound_item is not None and bound_item.digest != source_item.digest:
        return _refusal(
            SourceWindowPlanReason.SOURCE_ITEM_MISMATCH,
            "The inspected SourceItem no longer matches the Image Source node.",
        )

    contract_reason = _source_contract_mismatch(source_item, full_state)
    if contract_reason:
        return _refusal(
            SourceWindowPlanReason.SOURCE_CONTRACT_MISMATCH,
            contract_reason,
        )
    if not source_item.capabilities.exact_region_read:
        return _refusal(
            SourceWindowPlanReason.EXACT_REGION_UNAVAILABLE,
            "The selected reader does not advertise exact level-0 region reads.",
        )

    if any(tunnel.source_id == source_id for tunnel in pipeline.output_tunnel_list()):
        return _refusal(
            SourceWindowPlanReason.SOURCE_TUNNEL_PRESENT,
            "The Image Source feeds an output tunnel that may require the full image.",
        )

    outgoing = tuple(
        connection
        for connection in pipeline.connections
        if connection.source_id == source_id
    )
    if len(outgoing) != 1:
        return _refusal(
            SourceWindowPlanReason.DIRECT_TOPOLOGY_REQUIRED,
            "Exact pushdown requires Crop Stack to be the Image Source's only "
            "direct consumer.",
        )
    connection = outgoing[0]
    crop = pipeline.nodes.get(connection.target_id)
    if (
        crop is None
        or crop.operation_id != "crop_stack"
        or connection.source_port != 0
        or connection.target_port != 0
        or bool(connection.tunnel_name)
    ):
        return _refusal(
            SourceWindowPlanReason.DIRECT_TOPOLOGY_REQUIRED,
            "Exact pushdown requires one ordinary output-0 to input-0 Image "
            "Source -> Crop Stack connection.",
        )
    crop_inputs = tuple(
        item for item in pipeline.connections if item.target_id == crop.id
    )
    if crop_inputs != (connection,):
        return _refusal(
            SourceWindowPlanReason.DIRECT_TOPOLOGY_REQUIRED,
            "Crop Stack must receive exactly the direct Image Source input.",
        )
    if pipeline.node_is_bypassed(crop.id):
        return _refusal(
            SourceWindowPlanReason.CROP_BYPASSED,
            "A bypassed Crop Stack requires the complete Image Source.",
        )

    try:
        geometry = derive_crop_window_geometry(crop.params, full_state)
    except (TypeError, ValueError, IndexError) as exc:
        return _refusal(
            SourceWindowPlanReason.CROP_GEOMETRY_INVALID,
            f"Crop Stack cannot define an exact source window: {exc}",
        )
    if not geometry.requires_crop:
        return _refusal(
            SourceWindowPlanReason.CROP_GEOMETRY_INVALID,
            "Crop Stack retains the complete source, so no source-window "
            "pushdown is needed.",
        )

    declaration = _source_item_axis_declaration(source_item)
    request = SourceWindowRequest(
        geometry.selection,
        axis_declaration=declaration,
        preserve_time_and_channels=True,
        source_revision=source_item.container.revision.sha256,
        source_item_digest=source_item.digest,
    )
    decoded_bytes = (
        math.prod(geometry.output_shape) * np.dtype(source_item.resolved.dtype).itemsize
    )
    plan = ExactSourceCropWindowPlan(
        source_node_id=source.id,
        crop_node_id=crop.id,
        source_item_digest=source_item.digest,
        geometry=geometry,
        request=request,
        decoded_output_bytes=decoded_bytes,
    )
    return SourceCropWindowDecision(
        SourceWindowPlanReason.ELIGIBLE,
        "Crop Stack is the sole exact level-0 consumer of this Image Source.",
        plan,
    )


def suggest_centered_memory_fit_crop(
    source_item: SourceItem,
    full_state: ImageState,
    *,
    available_byte_budget: int,
    utilization_fraction: float = 0.5,
    peak_multiplier: float = 2.0,
    analysis_chunk_grid: Sequence[Sequence[int]] | None = None,
) -> CenteredCropSuggestion:
    """Return a centered, content-agnostic crop with conservative headroom.

    ``available_byte_budget`` is the caller's already-reserved additional host
    allocation budget.  Only ``utilization_fraction`` of it is used so source
    chunks, Python/reader overhead, and later graph work retain headroom.  The
    estimate governs source loading only; it does not promise that downstream
    operations will fit.  When ``analysis_chunk_grid`` is supplied, the chosen
    crop also includes every intersecting decoded storage chunk in the same
    conservative peak estimate enforced by the exact reader.
    """

    if not isinstance(source_item, SourceItem):
        raise TypeError("A fitted crop requires a SourceItem.")
    if not isinstance(full_state, ImageState):
        raise TypeError("A fitted crop requires a full ImageState.")
    contract_reason = _source_contract_mismatch(source_item, full_state)
    if contract_reason:
        raise ValueError(contract_reason)
    if not source_item.capabilities.exact_region_read:
        raise CropFitUnavailableError(
            "The selected reader cannot apply a fitted crop before full "
            "materialization because it does not advertise exact level-0 "
            "region reads."
        )
    available = _positive_integer(
        available_byte_budget,
        label="available_byte_budget",
    )
    if (
        isinstance(utilization_fraction, bool)
        or not isinstance(utilization_fraction, Real)
        or not 0.0 < float(utilization_fraction) <= 1.0
    ):
        raise ValueError("utilization_fraction must be in (0, 1].")
    if (
        isinstance(peak_multiplier, bool)
        or not isinstance(peak_multiplier, Real)
        or not math.isfinite(float(peak_multiplier))
        or float(peak_multiplier) < 1.0
    ):
        raise ValueError("peak_multiplier must be finite and at least 1.")

    fraction = float(utilization_fraction)
    multiplier = float(peak_multiplier)
    planned_budget = max(1, math.floor(available * fraction))
    shape = tuple(int(size) for size in full_state.shape)
    dtype = np.dtype(source_item.resolved.dtype)
    full_bytes = math.prod(shape) * dtype.itemsize
    allowed_elements = math.floor(planned_budget / (dtype.itemsize * multiplier))

    empty_geometry = derive_crop_window_geometry({}, full_state)
    crop_axes = tuple(
        index
        for index in (
            empty_geometry.z_axis,
            empty_geometry.y_axis,
            empty_geometry.x_axis,
        )
        if index is not None
    )
    fixed_elements = math.prod(
        size for index, size in enumerate(shape) if index not in crop_axes
    )
    if allowed_elements < fixed_elements:
        minimum_peak = math.ceil(fixed_elements * dtype.itemsize * multiplier)
        raise CropFitUnavailableError(
            "Even a one-sample spatial crop cannot fit while preserving every "
            "time, channel, and non-spatial dimension: estimated minimum peak "
            f"{minimum_peak} bytes exceeds the planned {planned_budget}-byte budget."
        )

    allowed_spatial_elements = max(1, allowed_elements // fixed_elements)
    full_spatial_elements = math.prod(shape[index] for index in crop_axes)

    def geometry_for_limit(spatial_limit: int) -> CropWindowGeometry:
        spatial_limit = max(1, min(int(spatial_limit), full_spatial_elements))
        if spatial_limit >= full_spatial_elements:
            retained = {index: shape[index] for index in crop_axes}
        else:
            scale = (spatial_limit / full_spatial_elements) ** (
                1.0 / len(crop_axes)
            )
            retained = {
                index: max(1, min(shape[index], math.floor(shape[index] * scale)))
                for index in crop_axes
            }
            while math.prod(retained.values()) > spatial_limit:
                reducible = tuple(index for index in crop_axes if retained[index] > 1)
                if not reducible:
                    raise CropFitUnavailableError(
                        "No positive semantics-preserving crop fits the planned budget."
                    )
                selected = max(
                    reducible,
                    key=lambda index: retained[index] / shape[index],
                )
                retained[selected] -= 1
        params = _centered_margin_params(shape, empty_geometry, retained)
        return derive_crop_window_geometry(params, full_state)

    maximum_limit = min(allowed_spatial_elements, full_spatial_elements)
    chunk_grid = tuple(
        tuple(int(size) for size in axis_chunks)
        for axis_chunks in (analysis_chunk_grid or ())
    )
    if chunk_grid:
        best_geometry: CropWindowGeometry | None = None
        best_peak = 0
        low = 1
        high = maximum_limit
        while low <= high:
            midpoint = (low + high) // 2
            candidate = geometry_for_limit(midpoint)
            estimate = estimate_exact_window_read(
                shape,
                dtype,
                candidate.selection,
                chunk_grid=chunk_grid,
            )
            if estimate.estimated_peak_bytes <= planned_budget:
                best_geometry = candidate
                best_peak = estimate.estimated_peak_bytes
                low = midpoint + 1
            else:
                high = midpoint - 1
        if best_geometry is None:
            smallest = geometry_for_limit(1)
            minimum = estimate_exact_window_read(
                shape,
                dtype,
                smallest.selection,
                chunk_grid=chunk_grid,
            )
            raise CropFitUnavailableError(
                "Even the smallest centered crop cannot fit its touched storage "
                f"chunks: estimated minimum peak {minimum.estimated_peak_bytes} "
                f"bytes exceeds the planned {planned_budget}-byte budget."
            )
        geometry = best_geometry
        estimated_peak = best_peak
    else:
        geometry = geometry_for_limit(maximum_limit)
        output_bytes = math.prod(geometry.output_shape) * dtype.itemsize
        estimated_peak = math.ceil(output_bytes * multiplier)

    output_bytes = math.prod(geometry.output_shape) * dtype.itemsize
    if estimated_peak > planned_budget:
        raise AssertionError("Centered crop planner exceeded its byte budget.")
    return CenteredCropSuggestion(
        geometry=geometry,
        full_decoded_bytes=full_bytes,
        decoded_output_bytes=output_bytes,
        estimated_peak_bytes=estimated_peak,
        available_byte_budget=available,
        planned_byte_budget=planned_budget,
        utilization_fraction=fraction,
        peak_multiplier=multiplier,
    )


def _centered_margin_params(
    shape: tuple[int, ...],
    geometry: CropWindowGeometry,
    retained: Mapping[int, int],
) -> dict[str, int]:
    def pair(axis: int | None) -> tuple[int, int]:
        if axis is None:
            return 0, 0
        extent = int(retained[axis])
        removed = int(shape[axis]) - extent
        first = removed // 2
        return first, removed - first

    z_start, z_end = pair(geometry.z_axis)
    top, bottom = pair(geometry.y_axis)
    left, right = pair(geometry.x_axis)
    return CropWindowMargins(
        z_start=z_start,
        z_end=z_end,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
    ).as_params()


def _source_contract_mismatch(
    source_item: SourceItem,
    full_state: ImageState,
) -> str:
    if source_item.resolved.analysis_level != 0:
        return "Scientific source-window planning is fixed to analysis level 0."
    if tuple(full_state.shape) != tuple(source_item.resolved.shape):
        return "The full ImageState shape does not match the inspected SourceItem."
    try:
        state_dtype = np.dtype(full_state.dtype).name
    except (TypeError, ValueError):
        return "The full ImageState dtype is invalid."
    if state_dtype != np.dtype(source_item.resolved.dtype).name:
        return "The full ImageState dtype does not match the inspected SourceItem."
    state_axes = tuple(axis.name.strip().casefold() for axis in full_state.axes)
    item_axes = tuple(
        str(name).strip().casefold() for name in source_item.resolved.axes
    )
    if len(state_axes) != len(full_state.shape) or state_axes != item_axes:
        return "The full ImageState axes do not match the inspected SourceItem."
    return ""


def _source_item_axis_declaration(source_item: SourceItem) -> AxisDeclaration | None:
    selector = source_item.selector
    if not selector.source_axes:
        return None
    return AxisDeclaration(
        ",".join(selector.source_axes),
        ",".join(selector.effective_axes),
    )


def _one_explicit_spatial_axis(
    state: ImageState,
    name: str,
    *,
    required: bool,
) -> int | None:
    named = tuple(
        index
        for index, axis in enumerate(state.axes)
        if axis.name.strip().casefold() == name
    )
    if not named:
        if required:
            raise ValueError(
                f"Crop window planning requires an explicit {name.upper()} axis."
            )
        return None
    if len(named) != 1:
        raise ValueError(f"Crop window planning found multiple {name.upper()} axes.")
    index = named[0]
    axis = state.axes[index]
    if not axis.is_explicit or axis.type.strip().casefold() != "space":
        if required:
            raise ValueError(
                f"Crop window {name.upper()} must be an explicit spatial axis."
            )
        return None
    return index


def _crop_channel_axis(value: object, ndim: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("Crop Stack channel-axis override must be an integer.")
    axis = int(value)
    if axis == -1:
        return None
    if not 0 <= axis < ndim:
        raise ValueError(
            f"Crop Stack channel-axis override {axis} is outside {ndim} dimensions."
        )
    return axis


def _refusal(
    reason_code: SourceWindowPlanReason,
    reason: str,
) -> SourceCropWindowDecision:
    return SourceCropWindowDecision(reason_code, reason)


def _nonnegative_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer.")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{label} must be non-negative.")
    return parsed


def _positive_integer(value: object, *, label: str) -> int:
    parsed = _nonnegative_integer(value, label=label)
    if parsed <= 0:
        raise ValueError(f"{label} must be positive.")
    return parsed


__all__ = [
    "CenteredCropSuggestion",
    "CropFitUnavailableError",
    "CropWindowGeometry",
    "CropWindowMargins",
    "ExactSourceCropWindowPlan",
    "SourceCropWindowDecision",
    "SourceWindowPlanReason",
    "derive_crop_window_geometry",
    "plan_exact_source_crop_window",
    "suggest_centered_memory_fit_crop",
]
