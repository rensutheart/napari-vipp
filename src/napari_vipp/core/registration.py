"""Whole-volume CPU registration and single-pass image/label resampling.

Translation uses Guizar-Sicairos et al., Optics Letters 33 (2008), as exposed
by scikit-image phase_cross_correlation. Rigid/affine optimization uses
SimpleITK ImageRegistrationMethod with deterministic full-voxel sampling,
multiresolution and physical-shift optimizer scaling. Scores are diagnostics,
not a calibrated confidence or evidence of biological correctness.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import math
import os

import numpy as np
from scipy import ndimage
from skimage.registration import phase_cross_correlation

from napari_vipp.core.progress import OperationCancelled
from napari_vipp.core.tables import TableData
from napari_vipp.core.transforms import TransformData, TransformState, registration_grid


def _progress(context, current=0, total=1, message=""):
    if context is not None:
        context.check_cancelled()
        context.report(current, total, message)


def _integer(value, name, low, high):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not low <= value <= high
    ):
        raise ValueError(f"{name} must be an integer from {low} to {high}.")
    return int(value)


def _fraction(value, name, low, high):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite fraction.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite fraction.") from exc
    if not np.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{name} must be between {low} and {high}.")
    return result


def _array(array, state):
    result = np.asarray(array)
    if (
        state is None
        or tuple(result.shape) != tuple(state.shape)
        or result.dtype.name != state.dtype
    ):
        raise ValueError("Registration data and carried shape/dtype metadata disagree.")
    if result.dtype.kind not in "biuf" or not result.size:
        raise ValueError("Registration requires a nonempty real numeric image.")
    registration_grid(state)
    if not np.isfinite(result).all():
        raise ValueError(
            "Registration requires finite input values; "
            "resolve NaN/Inf explicitly first."
        )
    return result


def _content_revision(array, context):
    """Exact estimation-input revision in bounded C-order buffers, including dtype."""
    digest = hashlib.sha256()
    digest.update(f"{array.dtype.str}:{array.shape}:C".encode("ascii"))
    iterator = np.nditer(
        array,
        flags=["external_loop", "buffered"],
        order="C",
        op_flags=["readonly"],
        buffersize=131072,
    )
    for block in iterator:
        if context is not None:
            context.check_cancelled()
        digest.update(block.tobytes(order="C"))
    return digest.hexdigest()


def _volume(array, state, channel, time=None):
    names = [a.name.lower() for a in state.axes]
    indices = []
    spatial_names = []
    for index, name in enumerate(names):
        if name == "c":
            if channel >= array.shape[index]:
                raise ValueError(
                    "The selected registration channel is outside the input image."
                )
            indices.append(channel)
        elif name == "t":
            if time is None:
                raise ValueError(
                    "Two images mode does not consume T. "
                    "Select a time point or use Time series mode."
                )
            indices.append(time)
        else:
            indices.append(slice(None))
            spatial_names.append(name)
    if "c" not in names and channel != 0:
        raise ValueError("A single-channel image only has channel index 0.")
    spatial = tuple(name for name in ("z", "y", "x") if name in spatial_names)
    volume = array[tuple(indices)].transpose(
        tuple(spatial_names.index(name) for name in spatial)
    )
    if min(volume.shape) < 4:
        raise ValueError(
            "Registration needs at least four pixels/voxels along every spatial axis."
        )
    if np.ptp(volume.astype(np.float64)) == 0:
        raise ValueError(
            "Registration cannot estimate alignment from a constant image."
        )
    if volume.dtype.kind in "iu" and (
        int(volume.min()) < -(2**53) or int(volume.max()) > 2**53
    ):
        raise ValueError(
            "Registration cannot represent wide integer intensities exactly. "
            "Use a separate floating-point estimation image; "
            "Apply Transform preserves nearest-neighbour label IDs."
        )
    return volume


def _same_sampling(moving, reference):
    return (
        moving.shape == reference.shape
        and moving.axes == reference.axes
        and moving.unit == reference.unit
        and np.allclose(moving.spacing, reference.spacing, rtol=1e-9, atol=1e-12)
    )


def _translation(
    fixed,
    moving,
    moving_grid,
    reference_grid,
    precision,
    max_shift=0.25,
    minimum_overlap=0.25,
):
    if not _same_sampling(moving_grid, reference_grid):
        raise ValueError(
            "Translation needs equal spatial shape and sampling. "
            "Use Rigid/Affine for different grids, or explicitly resample first."
        )
    # Subtracting a scalar offset changes no spatial information and is explicit
    # estimator conditioning. Original arrays are never normalized or modified.
    fixed_float = fixed.astype(np.float64) - float(np.mean(fixed))
    moving_float = moving.astype(np.float64) - float(np.mean(moving))
    # Detect indistinguishable separated integer peaks before subpixel fitting.
    # The local 5-voxel neighbourhood belongs to one smooth correlation peak;
    # a distinct equally high peak indicates repeated/periodic structure.
    spectrum = np.fft.fftn(fixed_float) * np.conj(np.fft.fftn(moving_float))
    correlation_surface = np.abs(np.fft.ifftn(spectrum))
    peak = np.unravel_index(np.argmax(correlation_surface), fixed.shape)
    peak_value = float(correlation_surface[peak])
    neighbours = np.ix_(
        *(
            np.mod(np.arange(p - 2, p + 3), n)
            for p, n in zip(peak, fixed.shape, strict=True)
        )
    )
    correlation_surface[neighbours] = 0
    if float(np.max(correlation_surface)) >= peak_value * (1 - 1e-6):
        raise ValueError(
            "Translation is ambiguous: repeated structures give indistinguishable "
            "correlation peaks. Use a more distinctive registration channel or region."
        )
    shift, error, _phase = phase_cross_correlation(
        fixed_float,
        moving_float,
        upsample_factor=precision,
        normalization=None,
        disambiguate=False,
    )
    if not np.isfinite(shift).all() or not np.isfinite(error):
        raise ValueError(
            "Translation failed to find a finite, unambiguous spatial shift."
        )
    # A tiny background-only overlap can win skimage's unconstrained periodic
    # disambiguation. Compare only candidates satisfying the user's displacement
    # and valid-overlap limits, using full real-space overlap for each candidate.
    candidates = []
    for wrap in itertools.product((-1, 0, 1), repeat=fixed.ndim):
        candidate = np.asarray(shift) + np.asarray(wrap) * fixed.shape
        if np.any(np.abs(candidate) / fixed.shape > max_shift + 1e-10):
            continue
        overlap = float(
            np.prod(
                np.maximum(0, np.asarray(fixed.shape) - np.ceil(np.abs(candidate)))
                / fixed.shape
            )
        )
        if overlap < minimum_overlap:
            continue
        shifted = ndimage.shift(
            moving_float, candidate, order=1, mode="constant", cval=0.0, prefilter=False
        )
        valid = tuple(
            slice(max(0, int(np.ceil(v))), min(n, int(np.floor(n + v))))
            for v, n in zip(candidate, fixed.shape, strict=True)
        )
        correlation = _correlation(fixed_float[valid].ravel(), shifted[valid].ravel())
        if correlation is not None:
            candidates.append((correlation, candidate))
    if not candidates:
        raise ValueError(
            "No translation satisfies Maximum displacement and Minimum overlap. "
            "Review the images or explicitly change these limits."
        )
    candidates.sort(key=lambda value: value[0], reverse=True)
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 1e-6:
        raise ValueError(
            "Translation is ambiguous: several periodic shifts have "
            "indistinguishable overlap scores."
        )
    matrix = np.eye(fixed.ndim + 1)
    matrix[:-1, -1] = (
        np.asarray(reference_grid.origin)
        - np.asarray(moving_grid.origin)
        + candidates[0][1] * reference_grid.spacing
    )
    return (
        matrix,
        float(error),
        "subpixel cross-correlation; "
        "displacement/overlap-constrained periodic disambiguation",
    )


def _sitk_estimate(
    fixed,
    moving,
    moving_grid,
    reference_grid,
    model,
    iterations,
    metric,
    precision,
    context,
    max_shift=0.25,
    minimum_overlap=0.25,
):
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError(
            "Rigid/Affine registration needs SimpleITK. "
            "Install the VIPP registration dependency."
        ) from exc
    dim = fixed.ndim
    fixed_image = sitk.GetImageFromArray(fixed.astype(np.float64))
    moving_image = sitk.GetImageFromArray(moving.astype(np.float64))
    for image, grid in ((fixed_image, reference_grid), (moving_image, moving_grid)):
        image.SetSpacing(tuple(reversed(grid.spacing)))
        image.SetOrigin(tuple(reversed(grid.origin)))
    initial = sitk.Euler2DTransform() if dim == 2 else sitk.Euler3DTransform()
    center = (
        np.asarray(reference_grid.origin)
        + (np.asarray(reference_grid.shape) - 1) * reference_grid.spacing / 2
    )
    initial.SetCenter(tuple(center[::-1]))
    if _same_sampling(moving_grid, reference_grid):
        translation, _error, _stop = _translation(
            fixed,
            moving,
            moving_grid,
            reference_grid,
            precision,
            max_shift=max_shift,
            minimum_overlap=minimum_overlap,
        )
        initial.SetTranslation(tuple(-translation[:-1, -1][::-1]))
    else:
        moving_center = (
            np.asarray(moving_grid.origin)
            + (np.asarray(moving_grid.shape) - 1) * moving_grid.spacing / 2
        )
        initial.SetTranslation(tuple((moving_center - center)[::-1]))

    def optimize(start):
        method = sitk.ImageRegistrationMethod()
        method.SetNumberOfThreads(min(os.cpu_count() or 1, 12))
        if metric == "Correlation":
            method.SetMetricAsCorrelation()
        else:
            method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
        method.SetMetricSamplingStrategy(method.NONE)
        method.SetInterpolator(sitk.sitkLinear)
        if isinstance(start, sitk.AffineTransform):
            method.SetOptimizerAsGradientDescentLineSearch(
                learningRate=1.0,
                numberOfIterations=iterations,
                convergenceMinimumValue=1e-7,
                convergenceWindowSize=10,
                lineSearchUpperLimit=2.0,
                maximumStepSizeInPhysicalUnits=min(reference_grid.spacing),
            )
        else:
            method.SetOptimizerAsRegularStepGradientDescent(
                learningRate=2.0,
                minStep=1e-4,
                numberOfIterations=iterations,
                relaxationFactor=0.5,
                gradientMagnitudeTolerance=1e-8,
            )
        method.SetOptimizerScalesFromPhysicalShift()
        factors = [4, 2, 1] if min(fixed.shape + moving.shape) >= 16 else [2, 1]
        method.SetShrinkFactorsPerLevel(factors)
        method.SetSmoothingSigmasPerLevel(
            [float(v // 2) * min(reference_grid.spacing) for v in factors]
        )
        method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        method.SetInitialTransform(start, inPlace=False)
        cancelled = [False]

        def iteration():
            if context is not None and context.is_cancelled():
                cancelled[0] = True
                method.StopRegistration()
                return
            if context is not None:
                context.report(
                    method.GetOptimizerIteration(),
                    iterations,
                    f"{model}: optimizing level "
                    f"{method.GetCurrentLevel() + 1}/{len(factors)}",
                )

        method.AddCommand(sitk.sitkIterationEvent, iteration)
        try:
            result = method.Execute(fixed_image, moving_image)
        except RuntimeError as exc:
            if cancelled[0] or (context is not None and context.is_cancelled()):
                raise OperationCancelled("Registration cancelled.") from exc
            raise ValueError(
                "Registration optimization failed. "
                "Check overlap, image contrast and the chosen model. "
                + str(exc).split("\n")[-1]
            ) from exc
        if cancelled[0]:
            raise OperationCancelled("Registration cancelled.")
        score = float(method.GetMetricValue())
        if (
            not np.isfinite(score)
            or score >= np.finfo(float).max / 2
            or method.GetMetricNumberOfValidPoints() < 4
        ):
            raise ValueError(
                "Registration has insufficient overlap "
                "or an invalid optimization score."
            )
        return result, score, method.GetOptimizerStopConditionDescription()

    result, score, stop = optimize(initial)
    if model == "Affine":
        # Initialize affine from the rigid optimum, not a second image resample.
        affine = sitk.AffineTransform(dim)
        zero = np.asarray(result.TransformPoint((0.0,) * dim))
        linear = np.column_stack(
            [
                np.asarray(result.TransformPoint(tuple(np.eye(dim)[i]))) - zero
                for i in range(dim)
            ]
        )
        affine.SetMatrix(tuple(linear.ravel()))
        affine.SetCenter(tuple(center[::-1]))
        affine.SetTranslation(tuple(zero + linear @ center[::-1] - center[::-1]))
        result, score, stop = optimize(affine)
    # SimpleITK returns fixed->moving in XYZ; public contract is its inverse in ZYX.
    zero = np.asarray(result.TransformPoint((0.0,) * dim))
    linear = np.column_stack(
        [
            np.asarray(result.TransformPoint(tuple(np.eye(dim)[i]))) - zero
            for i in range(dim)
        ]
    )
    backward = np.eye(dim + 1)
    backward[:-1, :-1] = linear[::-1, ::-1]
    backward[:-1, -1] = zero[::-1]
    try:
        forward = np.linalg.inv(backward)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Registration produced a singular transform; "
            "no aligned image was published."
        ) from exc
    if np.linalg.det(forward[:-1, :-1]) <= 0:
        raise ValueError(
            "Registration produced an orientation-reversing transform; "
            "review the images."
        )
    return forward, score, stop


def _resample_volume(
    volume, matrix, moving_grid, reference_grid, *, nearest, outside, context
):
    """Bounded coordinates; nearest gathers preserve uint64/int64 IDs exactly."""
    backward = (
        np.linalg.inv(moving_grid.index_to_physical)
        @ np.linalg.inv(matrix)
        @ reference_grid.index_to_physical
    )
    shape = reference_grid.shape
    source_last = np.asarray(volume.shape, dtype=np.float64) - 1
    mapped_extent = np.abs(backward[:-1, :-1]) @ (
        np.asarray(shape, dtype=np.float64) - 1
    ) + np.abs(backward[:-1, -1])
    # Physical/index matrix composition can put an exact boundary a few float64
    # ULPs outside (e.g. -2e-15 instead of 0). Snap only those roundoff-sized
    # excursions: 8 eps times the larger source/index-map extent, per axis.
    # This is not geometric padding or np.isclose's broad relative tolerance.
    boundary_tolerance = (
        8
        * np.finfo(np.float64).eps
        * np.maximum(1, np.maximum(source_last, mapped_extent))
    )[:, None]
    source_last = source_last[:, None]
    output = np.empty(shape, dtype=volume.dtype if nearest else np.float64)
    coverage = np.empty(shape, dtype=bool)
    flat_output, flat_coverage = output.reshape(-1), coverage.reshape(-1)
    count = math.prod(shape)
    for start in range(0, count, 131072):
        _progress(
            context, start, count, "Resampling original pixels to the reference grid"
        )
        end = min(start + 131072, count)
        indices = np.array(np.unravel_index(np.arange(start, end), shape), dtype=float)
        coordinates = backward[:-1, :-1] @ indices + backward[:-1, -1, None]
        # Use the same corrected coordinates for both coverage and sampling.
        # Marking coverage alone would still make SciPy return its exterior fill.
        coordinates = np.where(
            (coordinates < 0) & (coordinates >= -boundary_tolerance), 0, coordinates
        )
        coordinates = np.where(
            (coordinates > source_last)
            & (coordinates <= source_last + boundary_tolerance),
            source_last,
            coordinates,
        )
        if nearest:
            rounded = np.floor(coordinates + 0.5).astype(np.int64)
            valid = np.all(
                (rounded >= 0) & (rounded < np.asarray(volume.shape)[:, None]), axis=0
            )
            block = np.full(end - start, outside, dtype=volume.dtype)
            block[valid] = volume[tuple(rounded[:, valid])]
        else:
            # A linear result is valid only when its whole interpolation support
            # is inside the source. Constant exterior never counts as measured.
            valid = np.all(
                (coordinates >= 0) & (coordinates <= source_last),
                axis=0,
            )
            block = ndimage.map_coordinates(
                volume,
                coordinates,
                order=1,
                mode="constant",
                cval=float(outside),
                prefilter=False,
                output=np.float64,
            )
        flat_output[start:end] = block
        flat_coverage[start:end] = valid
    _progress(context, count, count, "Resampling complete")
    return output, coverage


def _correlation(left, right):
    x, y = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator else None


def estimate_registration(
    moving,
    reference=None,
    *,
    moving_state,
    reference_state=None,
    mode="Two images",
    model="Translation",
    channel=0,
    reference_channel=0,
    reference_time=0,
    precision=10,
    max_shift=0.25,
    minimum_overlap=0.25,
    iterations=200,
    metric="Correlation",
    progress_context=None,
):
    """Estimate one spatial transform or one per T against a fixed T anchor.

    max_shift limits displacement of the moving grid centre, measured as a
    fraction of reference spatial extents (relative to corresponding centres).
    It is not a pixel-wise rotation/affine displacement bound. The overlap gate
    is the fraction of reference voxels with valid linear interpolation support.
    """
    if mode not in ("Two images", "Time series") or model not in (
        "Translation",
        "Rigid",
        "Affine",
    ):
        raise ValueError(
            "Choose Two images/Time series and Translation/Rigid/Affine registration."
        )
    if metric not in ("Correlation", "Mutual information"):
        raise ValueError(
            "Choose Correlation or Mutual information as the registration metric."
        )
    channel = _integer(channel, "Channel", 0, 100000)
    reference_channel = _integer(reference_channel, "Reference channel", 0, 100000)
    reference_time = _integer(reference_time, "Reference time", 0, 1000000)
    precision = _integer(precision, "Subpixel precision", 1, 1000)
    iterations = _integer(iterations, "Iterations", 1, 10000)
    max_shift = _fraction(max_shift, "Maximum displacement fraction", 0.001, 1.0)
    minimum_overlap = _fraction(minimum_overlap, "Minimum overlap fraction", 0.001, 1.0)
    moving_array = _array(moving, moving_state)
    moving_grid = registration_grid(moving_state)
    names = tuple(a.name.lower() for a in moving_state.axes)
    if mode == "Time series":
        if reference is not None:
            raise ValueError(
                "Time series uses its selected time point as reference; "
                "disconnect the second image."
            )
        if "t" not in names:
            raise ValueError("Time series registration requires an explicit T axis.")
        count = moving_array.shape[names.index("t")]
        if count < 2 or reference_time >= count:
            raise ValueError(
                "Select a valid reference time from an image "
                "with at least two time points."
            )
        reference_state = moving_state
        reference_grid = moving_grid
        fixed = _volume(moving_array, moving_state, channel, reference_time)
        time_axis = moving_state.axes[names.index("t")]
    else:
        if reference is None or reference_state is None:
            raise ValueError(
                "Two images registration needs Moving and Reference "
                "inputs with metadata."
            )
        fixed_array = _array(reference, reference_state)
        reference_grid = registration_grid(reference_state)
        fixed = _volume(fixed_array, reference_state, reference_channel)
        count, time_axis = 1, None
    if (
        moving_grid.axes != reference_grid.axes
        or moving_grid.unit != reference_grid.unit
    ):
        raise ValueError(
            "Moving and reference images need the same spatial rank "
            "and compatible coordinate units."
        )
    matrices, rows = [], []
    for t in range(count):
        _progress(
            progress_context,
            t,
            count,
            f"Estimating {model.lower()} for time {t + 1}/{count}"
            if time_axis
            else "Estimating image registration",
        )
        volume = _volume(moving_array, moving_state, channel, t if time_axis else None)
        if time_axis and t == reference_time:
            matrix, score, stop = (
                np.eye(fixed.ndim + 1),
                None,
                "reference time: identity by definition",
            )
        elif model == "Translation":
            matrix, score, stop = _translation(
                fixed,
                volume,
                moving_grid,
                reference_grid,
                precision,
                max_shift=max_shift,
                minimum_overlap=minimum_overlap,
            )
        else:
            matrix, score, stop = _sitk_estimate(
                fixed,
                volume,
                moving_grid,
                reference_grid,
                model,
                iterations,
                metric,
                precision,
                progress_context,
                max_shift=max_shift,
                minimum_overlap=minimum_overlap,
            )
        _progress(
            progress_context, t, count, "Checking estimated alignment and valid overlap"
        )
        moving_center = (
            np.asarray(moving_grid.origin)
            + (np.asarray(moving_grid.shape) - 1) * moving_grid.spacing / 2
        )
        reference_center = (
            np.asarray(reference_grid.origin)
            + (np.asarray(reference_grid.shape) - 1) * reference_grid.spacing / 2
        )
        displacement = (
            matrix[:-1, :-1] @ moving_center + matrix[:-1, -1] - reference_center
        )
        fractions = np.abs(displacement) / (
            np.asarray(reference_grid.shape) * reference_grid.spacing
        )
        if np.any(fractions > max_shift + 1e-10):
            raise ValueError(
                f"Estimated displacement exceeds the {max_shift:g} image-extent limit. "
                "Review the images or explicitly increase Maximum displacement."
            )
        aligned, valid = _resample_volume(
            volume,
            matrix,
            moving_grid,
            reference_grid,
            nearest=False,
            outside=0.0,
            context=progress_context,
        )
        overlap = float(np.mean(valid))
        if overlap < minimum_overlap or np.count_nonzero(valid) < 4:
            raise ValueError(
                f"Estimated alignment has only {overlap:.1%} valid overlap; "
                f"required {minimum_overlap:.1%}."
            )
        correlation = _correlation(fixed[valid], aligned[valid])
        if correlation is None:
            raise ValueError(
                "Registration overlap has no intensity variation; "
                "alignment cannot be checked."
            )
        if (model == "Translation" or metric == "Correlation") and abs(
            correlation
        ) < 0.1:
            raise ValueError(
                "Registration has weak shared structure "
                "(absolute correlation below 0.1). Review the registration channel "
                "and overlap; no transform was accepted."
            )
        warning = (
            "Maximum iterations reached; inspect alignment"
            if "maximum" in stop.lower()
            else "Review alignment; scores do not establish biological correspondence"
        )
        rows.append(
            (
                t if time_axis else 0,
                reference_time if time_axis else 0,
                model,
                score,
                correlation,
                overlap,
                *tuple(float(v) for v in displacement),
                stop,
                warning,
            )
        )
        matrices.append(matrix)
    _progress(progress_context, count, count, "Registration complete")
    settings = tuple(
        dict(
            mode=mode,
            model=model,
            channel=channel,
            reference_channel=reference_channel,
            reference_time=reference_time,
            precision=precision,
            max_shift=max_shift,
            minimum_overlap=minimum_overlap,
            iterations=iterations,
            metric=metric,
            estimator_conditioning=(
                "mean subtraction for translation; no input-array modification"
            ),
            sampling="all voxels; no random samples",
            interpolation_for_diagnostics="linear float64",
            moving_content_sha256=_content_revision(moving_array, progress_context),
            reference_content_sha256=_content_revision(
                moving_array if time_axis else fixed_array, progress_context
            ),
        ).items()
    )
    package = "scikit-image" if model == "Translation" else "SimpleITK"
    implementation = (
        f"{package} {importlib.metadata.version(package)}; VIPP registration v1"
    )
    state = TransformState(
        model,
        moving_grid.axes,
        count,
        moving_grid.unit,
        moving_state.source_name,
        reference_state.source_name,
        (
            f"{mode}: {model}; CPU; whole spatial volume; fixed reference",
            implementation,
        ),
    )
    transform = TransformData(
        tuple(matrices),
        moving_grid,
        reference_grid,
        state,
        time_axis=time_axis,
        reference_time=reference_time if time_axis else None,
        settings=settings,
        implementation=implementation,
    )
    columns = (
        "time_index",
        "reference_time",
        "model",
        "optimizer_score",
        "correlation",
        "valid_overlap_fraction",
        *(f"center_displacement_{axis}" for axis in moving_grid.axes),
        "stop_reason",
        "review_note",
    )
    table = TableData(
        columns,
        tuple(rows),
        name="Registration diagnostics",
        table_kind="registration diagnostics",
        source_name=moving_state.source_name,
        column_units=tuple(
            (f"center_displacement_{axis}", moving_grid.unit)
            for axis in moving_grid.axes
        ),
    )
    return transform, table


def apply_transform(
    image,
    transform,
    *,
    image_state,
    interpolation="Automatic",
    outside_value=0.0,
    progress_context=None,
):
    """Apply measured transforms once to original pixels, retaining input T/C order.

    Automatic is nearest for masks/labels and float64 linear for intensity data.
    Nearest always preserves the source dtype and exact IDs. Coverage has the
    same axes/shape as aligned data, repeated consistently across channels.
    """
    if not isinstance(transform, TransformData):
        raise TypeError(
            "Apply Transform needs a registration transform, not an image/table."
        )
    array = _array(image, image_state)
    grid = registration_grid(image_state)
    expected = transform.moving_grid
    if grid.frame_id != expected.frame_id:
        raise ValueError(
            "The image belongs to a different coordinate frame. Use the original "
            "moving image or a sibling channel/mask retaining its source identity."
        )
    if (
        grid.axes != expected.axes
        or grid.shape != expected.shape
        or grid.unit != expected.unit
        or not np.allclose(grid.spacing, expected.spacing, rtol=1e-9, atol=1e-12)
        or not np.allclose(grid.origin, expected.origin, rtol=1e-9, atol=1e-12)
    ):
        raise ValueError(
            "The image grid differs from the transform's moving grid. "
            "Re-estimate after cropping or changing calibration."
        )
    if interpolation not in ("Automatic", "Nearest neighbour", "Nearest", "Linear"):
        raise ValueError("Choose Automatic, Nearest neighbour or Linear interpolation.")
    label_data = (
        array.dtype.kind == "b"
        or "label" in image_state.kind.lower()
        or "mask" in image_state.kind.lower()
    )
    nearest = interpolation in ("Nearest neighbour", "Nearest") or (
        interpolation == "Automatic" and label_data
    )
    if label_data and not nearest:
        raise ValueError(
            "Masks and labels require Nearest neighbour interpolation to preserve IDs."
        )
    if (
        not nearest
        and array.dtype.kind in "iu"
        and (int(array.min()) < -(2**53) or int(array.max()) > 2**53)
    ):
        raise ValueError(
            "Linear float64 interpolation cannot exactly represent these wide "
            "integers. Choose Nearest neighbour."
        )
    try:
        if not np.isfinite(outside_value):
            raise ValueError
        if nearest and array.dtype.kind in "biu":
            if outside_value != int(outside_value):
                raise ValueError
            lower, upper = (
                (0, 1)
                if array.dtype.kind == "b"
                else (np.iinfo(array.dtype).min, np.iinfo(array.dtype).max)
            )
            if not lower <= outside_value <= upper:
                raise ValueError
        outside = np.asarray(
            outside_value, dtype=array.dtype if nearest else np.float64
        ).item()
        if not np.isfinite(outside):
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Outside value must be finite and representable in the output dtype "
            "(integer for nearest integer data)."
        ) from exc
    names = tuple(a.name.lower() for a in image_state.axes)
    if transform.is_time_series:
        if "t" not in names or array.shape[names.index("t")] != len(transform.matrices):
            raise ValueError(
                "Transform series requires the same time-point count "
                "as its moving image."
            )
        axis = image_state.axes[names.index("t")]
        anchor = transform.time_axis
        if (
            axis.unit != anchor.unit
            or axis.scale != anchor.scale
            or axis.translation != anchor.translation
        ):
            raise ValueError("Time calibration does not match the transform series.")
    elif "t" in names:
        raise ValueError(
            "A pairwise transform cannot be silently broadcast over time; "
            "estimate a Time series transform."
        )
    shape = tuple(
        transform.reference_grid.shape[grid.axes.index(name)]
        if name in grid.axes
        else array.shape[i]
        for i, name in enumerate(names)
    )
    output = np.empty(shape, dtype=array.dtype if nearest else np.float64)
    coverage = np.empty(shape, dtype=bool)
    leading = tuple(i for i, name in enumerate(names) if name not in grid.axes)
    spatial_names = tuple(name for name in names if name in grid.axes)
    permutation = tuple(spatial_names.index(name) for name in grid.axes)
    undo = tuple(np.argsort(permutation))
    for block in itertools.product(*(range(array.shape[i]) for i in leading)):
        index = [slice(None)] * array.ndim
        for axis, value in zip(leading, block, strict=True):
            index[axis] = value
        t = index[names.index("t")] if transform.is_time_series else 0
        volume = array[tuple(index)].transpose(permutation)
        result, valid = _resample_volume(
            volume,
            np.asarray(transform.matrices[t]),
            grid,
            transform.reference_grid,
            nearest=nearest,
            outside=outside,
            context=progress_context,
        )
        output[tuple(index)] = result.transpose(undo)
        coverage[tuple(index)] = valid.transpose(undo)
    return output, coverage
