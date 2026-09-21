"""Faithful, CPU-only 2D CellProfiler seed propagation through Centrosome.

The reference uses eight neighbors and shortest accumulated paths combining
3x3 local intensity differences and pixel displacement. Intensities are passed
unchanged: regularization therefore depends on the authored intensity scale.
Physical pixel spacing does not enter this pixel-grid algorithm. This wrapper
does not reproduce CellProfiler's thresholding, seed filtering or hole filling.

References:
    https://doi.org/10.1007/11569541_54
    https://github.com/CellProfiler/centrosome
"""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
from centrosome.propagate import propagate as _propagate

_NAME = "Grow Regions from Seeds — CellProfiler Propagation"
_MAX_LABEL = np.iinfo(np.int32).max
_MAX_FLOAT = np.finfo(np.float64).max


def cellprofiler_propagation(
    inputs,
    regularization: float = 0.05,
    progress=None,
) -> np.ndarray:
    """Grow integer seed IDs inside a Boolean mask on one aligned YX image.

    ``inputs`` are guidance image, seed labels and foreground mask, in that
    order. They must have identical 2D shapes; the execution layer additionally
    verifies semantic YX axes and physical grids. No stack slicing is implicit.
    Guidance must be finite real numeric data represented exactly in float64;
    seeds must be nonnegative integers fitting int32; the mask must be Boolean.
    Return a new same-shape int32 label array, preserving original seed IDs.

    This calls the installed Centrosome reference kernel without normalization
    or algorithm substitution. Seeds outside the mask remain in the output but
    do not start growth. Mask components without reachable seeds remain zero.
    The fixed eight-neighbor and tie-handling behavior belongs to Centrosome.
    Conservative float64 cost-overflow checks reject unsafe numeric ranges.

    All backend buffers are private, including for read-only/noncontiguous
    inputs. Cancellation is checked before and after the blocking backend call;
    the reference kernel does not support interruption within that call.
    """
    if progress is not None:
        progress.check_cancelled()
        progress.report(0, 1, "CellProfiler Propagation")
    try:
        count = len(inputs)
    except TypeError as error:
        raise ValueError(f"{_NAME} requires image, seeds and mask inputs.") from error
    if count != 3:
        raise ValueError(
            f"{_NAME} requires exactly three inputs: image, seeds and mask."
        )
    image, seeds, mask = (np.asarray(value) for value in inputs)
    if any(array.ndim != 2 for array in (image, seeds, mask)):
        raise ValueError(
            f"{_NAME} accepts one 2D YX image only; select a plane explicitly "
            "or use Marker-Controlled Watershed for 3D segmentation."
        )
    if seeds.shape != image.shape or mask.shape != image.shape:
        raise ValueError(f"{_NAME} requires matching image, seed and mask shapes.")
    if any(length > _MAX_LABEL for length in image.shape):
        raise ValueError("Propagation axis lengths must fit int32 backend coordinates.")
    if image.dtype.kind not in "buif" or image.dtype.itemsize > 8:
        raise ValueError(
            "Propagation guidance must be real numeric data representable "
            "exactly in float64."
        )
    if seeds.dtype.kind not in "ui":
        raise ValueError(
            "Propagation seeds must be nonnegative integer label IDs fitting "
            "int32; Boolean and floating-point seeds are not supported."
        )
    if mask.dtype.kind != "b":
        raise ValueError("Propagation foreground mask must have Boolean dtype.")
    if seeds.size and (int(seeds.min()) < 0 or int(seeds.max()) > _MAX_LABEL):
        raise ValueError(
            f"Propagation seed IDs must lie between 0 and {_MAX_LABEL} (int32)."
        )
    if isinstance(regularization, (bool, np.bool_)) or not isinstance(
        regularization, Real
    ):
        raise ValueError(
            "Propagation regularization must be a finite nonnegative number."
        )
    try:
        weight = float(regularization)
    except (OverflowError, ValueError) as error:
        raise ValueError(
            "Propagation regularization must be a finite nonnegative number."
        ) from error
    if not math.isfinite(weight) or weight < 0:
        raise ValueError(
            "Propagation regularization must be a finite nonnegative number."
        )

    guidance = _exact_float64_guidance(image)
    _validate_cost_range(guidance, weight)
    if progress is not None:
        progress.check_cancelled()
    if image.size:
        labels, _distances = _propagate(
            guidance,
            np.array(seeds, dtype=np.int32, order="C", copy=True),
            np.array(mask, dtype=bool, order="C", copy=True),
            weight,
        )
        result = np.array(labels, dtype=np.int32, order="C", copy=True)
    else:
        # Avoid the reference Cython code's indexing assumptions for zero axes.
        result = np.empty(image.shape, dtype=np.int32)
    if progress is not None:
        progress.check_cancelled()
        progress.report(1, 1, "CellProfiler Propagation")
        progress.check_cancelled()
    return result


def _exact_float64_guidance(image: np.ndarray) -> np.ndarray:
    """Make a private buffer without rounding wide integer intensity values."""
    guidance = np.array(image, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(guidance).all():
        raise ValueError("Propagation guidance must contain only finite intensities.")
    if image.dtype.kind in "ui" and image.dtype.itemsize == 8 and image.size:
        # Prevent an undefined/out-of-range float-to-integer cast at the upper
        # boundary, then check exact round-trip in the original integer dtype.
        upper_exclusive = float(2**64 if image.dtype.kind == "u" else 2**63)
        if np.any(guidance >= upper_exclusive) or not np.array_equal(
            guidance.astype(image.dtype), image
        ):
            raise ValueError(
                "Propagation guidance contains integer intensities that cannot "
                "be represented exactly in float64. Rescale explicitly upstream."
            )
    return guidance


def _validate_cost_range(guidance: np.ndarray, weight: float) -> None:
    """Bound the reference's squared local terms and accumulated path length."""
    span = float(guidance.max()) - float(guidance.min()) if guidance.size else 0.0
    # Up to nine absolute pixel differences contribute per 3x3 neighborhood;
    # a diagonal step contributes twice weight**2. hypot avoids overflow in
    # this guard, although the unmodified reference explicitly squares terms.
    edge_bound = math.hypot(9.0 * span, math.sqrt(2.0) * weight)
    if not math.isfinite(edge_bound) or edge_bound > math.sqrt(_MAX_FLOAT) / 2:
        raise ValueError(
            "Propagation intensity range or regularization risks float64 cost "
            "overflow. Rescale the guidance or reduce regularization explicitly."
        )
    if edge_bound > _MAX_FLOAT / max(1, guidance.size - 1):
        raise ValueError("Propagation accumulated path costs risk float64 overflow.")


__all__ = ["cellprofiler_propagation"]
