"""Physical-grid contracts for scientific multi-image operations.

These checks are deliberately separate from array-shape validation.  Equal
array shapes do not establish that samples describe the same physical points,
and VIPP never resamples an input implicitly to make grids agree.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from napari_vipp.core.metadata import AxisMetadata, ImageState

_RELATIVE_TOLERANCE = 1e-9
_ABSOLUTE_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class GridAxis:
    """One sampled coordinate axis in an :class:`ImageGrid`."""

    name: str
    type: str
    size: int
    scale: float
    translation: float
    unit: str | None
    explicit: bool

    @classmethod
    def from_metadata(cls, axis: AxisMetadata, size: int) -> GridAxis:
        return cls(
            name=str(axis.name).strip().casefold(),
            type=str(axis.type).strip().casefold(),
            size=int(size),
            scale=float(axis.scale),
            translation=float(axis.translation),
            unit=str(axis.unit).strip() if axis.unit else None,
            explicit=bool(axis.is_explicit),
        )


@dataclass(frozen=True, slots=True)
class ImageGrid:
    """Immutable coordinate-grid view of carried image metadata."""

    axes: tuple[GridAxis, ...]

    @classmethod
    def from_image_state(cls, state: ImageState) -> ImageGrid:
        if len(state.shape) != len(state.axes):
            raise ValueError(
                "Image-state shape and axis metadata have different ranks: "
                f"{len(state.shape)}D shape versus {len(state.axes)} axes."
            )
        return cls(
            tuple(
                GridAxis.from_metadata(axis, size)
                for axis, size in zip(state.axes, state.shape, strict=True)
            )
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(axis.size for axis in self.axes)


@dataclass(frozen=True, slots=True)
class GridIssue:
    """One reason two grids cannot safely be treated as aligned."""

    code: str
    detail: str


@dataclass(frozen=True, slots=True)
class GridCompatibility:
    """Structured result of a grid comparison."""

    issues: tuple[GridIssue, ...] = ()

    @property
    def compatible(self) -> bool:
        return not self.issues


@dataclass(frozen=True, slots=True)
class MaskBroadcastCompatibility:
    """Semantic mapping and issues for broadcasting a mask onto an image."""

    mask_to_image_axes: tuple[int, ...] = ()
    issues: tuple[GridIssue, ...] = ()

    @property
    def compatible(self) -> bool:
        return not self.issues


def compare_aligned_grids(
    reference: ImageGrid,
    candidate: ImageGrid,
) -> GridCompatibility:
    """Compare grids whose array elements must refer to the same coordinates."""
    issues: list[GridIssue] = []
    if len(reference.axes) != len(candidate.axes):
        return GridCompatibility(
            (
                GridIssue(
                    "rank",
                    f"ranks differ ({len(reference.axes)}D versus "
                    f"{len(candidate.axes)}D)",
                ),
            )
        )

    for index, (left, right) in enumerate(
        zip(reference.axes, candidate.axes, strict=True)
    ):
        label = _axis_label(left, index)
        if left.name != right.name or left.type != right.type:
            issues.append(
                GridIssue(
                    "axis_semantics",
                    f"{label} semantics differ ({_semantics(left)!r} versus "
                    f"{_semantics(right)!r})",
                )
            )
        if left.size != right.size:
            issues.append(
                GridIssue(
                    "axis_size",
                    f"{label} sizes differ ({left.size} versus {right.size})",
                )
            )
        issues.extend(
            _axis_calibration_issues(
                left,
                right,
                axis_label=label,
                compare_translation=True,
            )
        )
    return GridCompatibility(tuple(issues))


def compare_mask_broadcast_grids(
    image: ImageGrid,
    mask: ImageGrid,
) -> MaskBroadcastCompatibility:
    """Resolve a mask-to-image axis mapping without using coincident sizes.

    Same-rank inputs retain the ordinary pointwise-grid contract. A lower-rank
    mask may omit image axes only when all remaining semantics are explicit and
    uniquely match image axes with equal sizes and physical calibration.
    """
    if len(image.axes) == len(mask.axes):
        aligned = compare_aligned_grids(image, mask)
        return MaskBroadcastCompatibility(
            (tuple(range(len(mask.axes))) if aligned.compatible else ()),
            aligned.issues,
        )
    if len(mask.axes) > len(image.axes):
        return MaskBroadcastCompatibility(
            issues=(
                GridIssue(
                    "rank",
                    f"Mask rank {len(mask.axes)}D exceeds Image rank "
                    f"{len(image.axes)}D",
                ),
            )
        )

    inferred_axes = [
        f"Image axis {index} ({axis.name or 'unnamed'})"
        for index, axis in enumerate(image.axes)
        if not axis.explicit
    ] + [
        f"Mask axis {index} ({axis.name or 'unnamed'})"
        for index, axis in enumerate(mask.axes)
        if not axis.explicit
    ]
    if inferred_axes:
        return MaskBroadcastCompatibility(
            issues=(
                GridIssue(
                    "axis_confidence",
                    "mismatched-rank mask broadcasting requires explicit axis "
                    "semantics; inferred semantics remain on "
                    + ", ".join(inferred_axes),
                ),
            )
        )

    issues: list[GridIssue] = []
    mapping: list[int] = []
    for mask_index, mask_axis in enumerate(mask.axes):
        matches = [
            image_index
            for image_index, image_axis in enumerate(image.axes)
            if (image_axis.name == mask_axis.name and image_axis.type == mask_axis.type)
        ]
        mask_label = _axis_label(mask_axis, mask_index)
        if not matches:
            issues.append(
                GridIssue(
                    "axis_semantics",
                    f"Mask {mask_label} semantics {_semantics(mask_axis)!r} "
                    "have no matching Image axis",
                )
            )
            continue
        if len(matches) > 1:
            issues.append(
                GridIssue(
                    "ambiguous_axis",
                    f"Mask {mask_label} semantics {_semantics(mask_axis)!r} "
                    f"match multiple Image axes {matches}",
                )
            )
            continue

        image_index = matches[0]
        image_axis = image.axes[image_index]
        mapping.append(image_index)
        image_label = _axis_label(image_axis, image_index)
        if mask_axis.size != image_axis.size:
            issues.append(
                GridIssue(
                    "axis_size",
                    f"{image_label} sizes differ ({image_axis.size} versus "
                    f"{mask_axis.size})",
                )
            )
        issues.extend(
            _axis_calibration_issues(
                image_axis,
                mask_axis,
                axis_label=image_label,
                compare_translation=True,
            )
        )

    if len(mapping) == len(mask.axes) and len(set(mapping)) != len(mapping):
        issues.append(
            GridIssue(
                "ambiguous_axis",
                "multiple Mask axes resolve to the same Image axis",
            )
        )
    return MaskBroadcastCompatibility(tuple(mapping), tuple(issues))


def compare_psf_sampling(
    image: ImageGrid,
    psf: ImageGrid,
    *,
    spatial_ndim: int,
) -> GridCompatibility:
    """Compare image/PSF sampling while allowing different kernel extents.

    Deconvolution treats a PSF as a centered kernel, so its array shape and
    coordinate translation need not equal the image's.  Axis semantics and
    sample spacing must agree, however; this function never resamples a PSF.
    """
    spatial_ndim = int(spatial_ndim)
    if spatial_ndim not in {2, 3}:
        return GridCompatibility(
            (
                GridIssue(
                    "spatial_rank",
                    "deconvolution sampling can only be checked for 2D or 3D "
                    f"processing, not {spatial_ndim}D",
                ),
            )
        )
    if len(image.axes) < spatial_ndim:
        return GridCompatibility(
            (
                GridIssue(
                    "image_rank",
                    f"image rank {len(image.axes)}D is smaller than the resolved "
                    f"{spatial_ndim}D spatial processing rank",
                ),
            )
        )
    if len(psf.axes) != spatial_ndim:
        return GridCompatibility(
            (
                GridIssue(
                    "psf_rank",
                    f"PSF rank is {len(psf.axes)}D but deconvolution is resolved "
                    f"as {spatial_ndim}D",
                ),
            )
        )

    image_axes = image.axes[-spatial_ndim:]
    issues: list[GridIssue] = []
    for index, (image_axis, psf_axis) in enumerate(
        zip(image_axes, psf.axes, strict=True)
    ):
        label = _axis_label(image_axis, len(image.axes) - spatial_ndim + index)
        if image_axis.type != "space" or psf_axis.type != "space":
            issues.append(
                GridIssue(
                    "non_spatial_axis",
                    f"{label} must be spatial in both image and PSF metadata "
                    f"({_semantics(image_axis)!r} versus "
                    f"{_semantics(psf_axis)!r})",
                )
            )
        elif image_axis.name != psf_axis.name:
            issues.append(
                GridIssue(
                    "axis_semantics",
                    f"{label} name differs ({image_axis.name!r} versus "
                    f"{psf_axis.name!r})",
                )
            )
        issues.extend(
            _axis_calibration_issues(
                image_axis,
                psf_axis,
                axis_label=label,
                compare_translation=False,
            )
        )
    return GridCompatibility(tuple(issues))


def validate_aligned_image_states(
    states: Sequence[ImageState],
    *,
    input_labels: Sequence[str] = (),
    operation_title: str,
) -> None:
    """Reject inputs that a pointwise operation cannot align without resampling."""
    if len(states) < 2:
        return
    labels = _input_labels(len(states), input_labels)
    reference = ImageGrid.from_image_state(states[0])
    for index, state in enumerate(states[1:], start=1):
        compatibility = compare_aligned_grids(
            reference,
            ImageGrid.from_image_state(state),
        )
        if not compatibility.compatible:
            _raise_grid_error(
                operation_title,
                labels[0],
                labels[index],
                compatibility,
                remedy=(
                    "Reorder or explicitly resample one input onto the other "
                    "grid before combining them."
                ),
            )


def validate_mask_broadcast_image_states(
    image_state: ImageState,
    mask_state: ImageState,
    *,
    operation_title: str = "Mask Image",
) -> tuple[int, ...]:
    """Return the only scientifically supported mask-to-image axis mapping."""
    compatibility = compare_mask_broadcast_grids(
        ImageGrid.from_image_state(image_state),
        ImageGrid.from_image_state(mask_state),
    )
    if not compatibility.compatible:
        _raise_grid_error(
            operation_title,
            "Image",
            "Mask",
            GridCompatibility(compatibility.issues),
            remedy=(
                "Supply explicit, unique axis metadata with matching sizes and "
                "calibration. Reorder or explicitly resample the mask when its "
                "sampled grid differs; VIPP does not align masks by dimension "
                "size."
            ),
        )
    return compatibility.mask_to_image_axes


def validate_psf_image_states(
    image_state: ImageState,
    psf_state: ImageState,
    *,
    spatial_ndim: int,
    operation_title: str,
) -> None:
    """Reject an image/PSF pair with incompatible physical sampling."""
    compatibility = compare_psf_sampling(
        ImageGrid.from_image_state(image_state),
        ImageGrid.from_image_state(psf_state),
        spatial_ndim=spatial_ndim,
    )
    if not compatibility.compatible:
        _raise_grid_error(
            operation_title,
            "Image",
            "PSF",
            compatibility,
            remedy=(
                "Supply a PSF measured or explicitly resampled at the image's "
                "spatial sampling; VIPP does not resample PSFs implicitly."
            ),
        )


def _axis_calibration_issues(
    left: GridAxis,
    right: GridAxis,
    *,
    axis_label: str,
    compare_translation: bool,
) -> tuple[GridIssue, ...]:
    issues: list[GridIssue] = []
    left_calibration = _normalized_calibration(left)
    right_calibration = _normalized_calibration(right)
    if left_calibration is None:
        issues.append(
            GridIssue(
                "invalid_calibration",
                f"{axis_label} has invalid reference calibration "
                f"(scale={left.scale!r}, translation={left.translation!r})",
            )
        )
        return tuple(issues)
    if right_calibration is None:
        issues.append(
            GridIssue(
                "invalid_calibration",
                f"{axis_label} has invalid candidate calibration "
                f"(scale={right.scale!r}, translation={right.translation!r})",
            )
        )
        return tuple(issues)

    left_dimension, left_scale, left_translation = left_calibration
    right_dimension, right_scale, right_translation = right_calibration
    if left_dimension != right_dimension:
        issues.append(
            GridIssue(
                "unit",
                f"{axis_label} units are incompatible "
                f"({_unit_label(left.unit)!r} versus {_unit_label(right.unit)!r})",
            )
        )
        return tuple(issues)
    if not _close(left_scale, right_scale):
        issues.append(
            GridIssue(
                "scale",
                f"{axis_label} sample spacing differs "
                f"({_calibration_label(left)} versus "
                f"{_calibration_label(right)})",
            )
        )
    if compare_translation and not _close(left_translation, right_translation):
        issues.append(
            GridIssue(
                "translation",
                f"{axis_label} origins differ "
                f"({_origin_label(left)} versus {_origin_label(right)})",
            )
        )
    return tuple(issues)


def _normalized_calibration(axis: GridAxis) -> tuple[str, float, float] | None:
    if (
        not math.isfinite(axis.scale)
        or axis.scale <= 0
        or not math.isfinite(axis.translation)
    ):
        return None
    dimension, factor = _unit_dimension_and_factor(axis.unit)
    return dimension, axis.scale * factor, axis.translation * factor


def _unit_dimension_and_factor(unit: str | None) -> tuple[str, float]:
    normalized = _normalized_unit(unit)
    if normalized in {"", "pixel"}:
        return "index", 1.0

    length_factors = {
        "meter": 1_000_000.0,
        "centimeter": 10_000.0,
        "millimeter": 1_000.0,
        "micrometer": 1.0,
        "nanometer": 0.001,
        "picometer": 0.000001,
        "angstrom": 0.0001,
    }
    if normalized in length_factors:
        return "length_micrometer", length_factors[normalized]

    time_factors = {
        "second": 1.0,
        "millisecond": 0.001,
        "microsecond": 0.000001,
        "nanosecond": 0.000000001,
        "minute": 60.0,
        "hour": 3600.0,
    }
    if normalized in time_factors:
        return "time_second", time_factors[normalized]
    return f"unit:{normalized}", 1.0


def _normalized_unit(unit: str | None) -> str:
    text = str(unit or "").strip().casefold().replace("μ", "µ")
    aliases = {
        "px": "pixel",
        "pixels": "pixel",
        "m": "meter",
        "metre": "meter",
        "metres": "meter",
        "meters": "meter",
        "cm": "centimeter",
        "centimetre": "centimeter",
        "centimetres": "centimeter",
        "centimeters": "centimeter",
        "mm": "millimeter",
        "millimetre": "millimeter",
        "millimetres": "millimeter",
        "millimeters": "millimeter",
        "µm": "micrometer",
        "um": "micrometer",
        "micron": "micrometer",
        "microns": "micrometer",
        "micrometre": "micrometer",
        "micrometres": "micrometer",
        "micrometers": "micrometer",
        "nm": "nanometer",
        "nanometre": "nanometer",
        "nanometres": "nanometer",
        "nanometers": "nanometer",
        "pm": "picometer",
        "picometre": "picometer",
        "picometres": "picometer",
        "picometers": "picometer",
        "å": "angstrom",
        "angström": "angstrom",
        "angstroms": "angstrom",
        "s": "second",
        "sec": "second",
        "seconds": "second",
        "ms": "millisecond",
        "milliseconds": "millisecond",
        "us": "microsecond",
        "µs": "microsecond",
        "microseconds": "microsecond",
        "ns": "nanosecond",
        "nanoseconds": "nanosecond",
        "min": "minute",
        "minutes": "minute",
        "h": "hour",
        "hr": "hour",
        "hours": "hour",
    }
    return aliases.get(text, text)


def _raise_grid_error(
    operation_title: str,
    reference_label: str,
    candidate_label: str,
    compatibility: GridCompatibility,
    *,
    remedy: str,
) -> None:
    details = "; ".join(issue.detail for issue in compatibility.issues)
    raise ValueError(
        f"{operation_title} cannot combine {reference_label} and "
        f"{candidate_label}: their physical grids are incompatible ({details}). "
        f"{remedy}"
    )


def _input_labels(count: int, labels: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        str(labels[index]).strip() or f"Input {index + 1}"
        if index < len(labels)
        else f"Input {index + 1}"
        for index in range(count)
    )


def _axis_label(axis: GridAxis, index: int) -> str:
    return f"axis {index} ({axis.name or 'unnamed'})"


def _semantics(axis: GridAxis) -> str:
    return f"{axis.name or 'unnamed'}:{axis.type or 'unknown'}"


def _unit_label(unit: str | None) -> str:
    return _normalized_unit(unit) or "index/pixel"


def _calibration_label(axis: GridAxis) -> str:
    return f"{axis.scale:g} {_unit_label(axis.unit)} per sample"


def _origin_label(axis: GridAxis) -> str:
    return f"{axis.translation:g} {_unit_label(axis.unit)}"


def _close(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=_RELATIVE_TOLERANCE,
        abs_tol=_ABSOLUTE_TOLERANCE,
    )
