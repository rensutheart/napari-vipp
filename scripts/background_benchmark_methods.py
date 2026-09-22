"""Exploratory background methods, not registered VIPP operations.

Every candidate uses the current VIPP preparation and output policies. Flat
opening and reconstruction are intentionally different algorithms from the
non-flat rolling-ball reference. Numerical providers load only when called.
"""

from __future__ import annotations

METHODS = (
    "vipp_cpu",
    "vipp_gpu",
    "itk_ball",
    "itk_box",
    "itk_reconstruction",
    "scipy_box",
)
METHOD_DESCRIPTIONS = {
    "vipp_cpu": {
        "algorithm": "VIPP non-flat rolling-ball background subtraction",
        "provider": "scikit-image / SciPy CPU",
        "boundary": "Rolling-ball ignores out-of-image samples (+infinity padding)",
    },
    "vipp_gpu": {
        "algorithm": "VIPP non-flat rolling-ball background subtraction",
        "provider": "VIPP CuPy CUDA; transfer-inclusive, synchronized host output",
        "boundary": "Rolling-ball ignores out-of-image samples",
    },
    "itk_ball": {
        "algorithm": "Flat grayscale ball opening, then original minus estimate",
        "provider": "SimpleITK GrayscaleMorphologicalOpeningImageFilter",
        "boundary": "SafeBorder=True; pad r with working-dtype maximum, then crop",
        "kernel": "sitkBall; ITK non-parametric radius convention (+0.5 voxel)",
    },
    "itk_box": {
        "algorithm": "Flat grayscale box opening, then original minus estimate",
        "provider": "SimpleITK GrayscaleMorphologicalOpeningImageFilter",
        "boundary": "SafeBorder=True; pad r with working-dtype maximum, then crop",
        "kernel": "sitkBox; side length 2*r+1 on each spatial axis",
    },
    "itk_reconstruction": {
        "algorithm": "Opening by reconstruction, then original minus estimate",
        "provider": "SimpleITK OpeningByReconstructionImageFilter",
        "boundary": "Native erosion boundary and reconstruction within the image",
        "kernel": "sitkBall; ITK non-parametric radius convention (+0.5 voxel)",
        "settings": "FullyConnected=False; PreserveIntensities=False",
    },
    "scipy_box": {
        "algorithm": "Flat grayscale box opening, then original minus estimate",
        "provider": "SciPy grey_opening; same-method box control",
        "boundary": "Pad r with working-dtype maximum, open, then crop (safe border)",
        "kernel": "Full box of side length 2*r+1 on each spatial axis",
    },
}


def _validate(array, method, radius, spatial_ndim, threads):
    import numpy as np

    if method not in METHODS:
        raise ValueError(f"Unknown background method: {method!r}")
    arr = np.asarray(array)
    if arr.dtype.kind not in "biuf":
        raise TypeError("Background benchmarks require real numeric arrays.")
    if spatial_ndim not in (2, 3) or arr.ndim < spatial_ndim:
        raise ValueError("Use spatial_ndim=2 or 3, not exceeding the array dimensions.")
    if not np.isfinite(float(radius)):
        raise ValueError("Radius must be finite.")
    if isinstance(threads, bool) or int(threads) != threads or threads < 1:
        raise ValueError("Thread count must be a positive integer.")
    return arr, max(int(round(float(radius))), 1), int(spatial_ndim), int(threads)


def _mode(spatial_ndim):
    return "2D YX" if spatial_ndim == 2 else "3D ZYX"


def _candidate_block(block, *, method, radius, smoothing, light_background, threads):
    import numpy as np
    from scipy import ndimage as ndi

    dtype = np.float64 if block.dtype == np.float64 else np.float32
    values = np.asarray(block, dtype=dtype)
    if not values.size:
        return values.copy()
    finite = values[np.isfinite(values)]
    if not finite.size:
        return np.zeros_like(values)
    low, high = float(finite.min()), float(finite.max())
    safe = np.nan_to_num(values, nan=low, posinf=high, neginf=low)
    if smoothing:
        safe = ndi.uniform_filter(safe, size=3, mode="nearest")
    if light_background:
        low, high = float(safe.min()), float(safe.max())
        offset = low + high
        safe = offset - safe
    if method == "scipy_box":
        maximum = np.finfo(dtype).max
        padded = np.pad(safe, radius, constant_values=maximum)
        opened = ndi.grey_opening(
            padded,
            size=(2 * radius + 1,) * safe.ndim,
            mode="constant",
            cval=maximum,
        )
        background = opened[(slice(radius, -radius),) * safe.ndim].copy()
    else:
        import SimpleITK as sitk

        if method == "itk_reconstruction":
            filter_ = sitk.OpeningByReconstructionImageFilter()
            filter_.SetFullyConnected(False)
            filter_.SetPreserveIntensities(False)
        else:
            filter_ = sitk.GrayscaleMorphologicalOpeningImageFilter()
            filter_.SetSafeBorder(True)
        filter_.SetKernelType(sitk.sitkBox if method == "itk_box" else sitk.sitkBall)
        filter_.SetKernelRadius([radius] * safe.ndim)
        filter_.SetNumberOfThreads(threads)
        filter_.SetNumberOfWorkUnits(threads)
        image = sitk.GetImageFromArray(np.ascontiguousarray(safe), isVector=False)
        background = sitk.GetArrayFromImage(filter_.Execute(image))
    if light_background:
        background = offset - background
    return np.asarray(background, dtype=dtype)


def estimate_background(
    array,
    *,
    method,
    radius,
    smoothing=True,
    light_background=False,
    spatial_ndim=2,
    threads=12,
):
    """Return a float estimate before subtraction/rounding; CPU methods only.

    Float64 inputs retain float64; other numeric inputs use float32, as in VIPP.
    Every leading dimension is independent. No channel inference is performed.
    """
    import numpy as np

    from napari_vipp.core import operations as ops

    arr, radius, spatial_ndim, threads = _validate(
        array, method, radius, spatial_ndim, threads
    )
    if method == "vipp_gpu":
        raise ValueError(
            "Background diagnostics use CPU methods; GPU is timed end-to-end."
        )
    dtype = np.float64 if arr.dtype == np.float64 else np.float32
    if arr.dtype == bool:
        return np.zeros_like(arr, dtype=dtype)
    if method == "vipp_cpu":
        return ops._estimate_rolling_ball_background(
            arr,
            radius=radius,
            light_background=light_background,
            disable_smoothing=not smoothing,
            spatial_mode=_mode(spatial_ndim),
            resolved_spatial_ndim=spatial_ndim,
            channel_axis=None,
        )
    output = np.empty(arr.shape, dtype=dtype)
    # Separate calls are essential for reconstruction: a zero kernel radius on
    # Z would not prevent the reconstruction itself propagating between planes.
    for index in np.ndindex(arr.shape[:-spatial_ndim]):
        output[index] = _candidate_block(
            arr[index],
            method=method,
            radius=radius,
            smoothing=smoothing,
            light_background=light_background,
            threads=threads,
        )
    return output


def raw_corrected(
    array,
    *,
    method,
    radius,
    smoothing=True,
    light_background=False,
    clip_negative=True,
    spatial_ndim=2,
    threads=12,
):
    """Return unrounded, unclipped original-minus-estimate diagnostic values.

    ``clip_negative`` is accepted for call-site symmetry but intentionally does
    not affect this raw diagnostic. Nonfinite original values remain nonfinite.
    """
    import numpy as np

    arr = np.asarray(array)
    background = estimate_background(
        arr,
        method=method,
        radius=radius,
        smoothing=smoothing,
        light_background=light_background,
        spatial_ndim=spatial_ndim,
        threads=threads,
    )
    values = arr.astype(background.dtype, copy=False)
    if arr.dtype == bool:
        return values.copy()
    return background - values if light_background else values - background


def run_method(
    array,
    *,
    method,
    radius,
    smoothing=True,
    light_background=False,
    clip_negative=True,
    spatial_ndim=2,
    threads=12,
):
    """Execute the complete operation; GPU includes upload, download and sync."""
    import numpy as np

    from napari_vipp.core import operations as ops

    arr, radius, spatial_ndim, threads = _validate(
        array, method, radius, spatial_ndim, threads
    )
    kwargs = {
        "radius": radius,
        "light_background": light_background,
        "disable_smoothing": not smoothing,
        "clip_negative": clip_negative,
        "spatial_mode": _mode(spatial_ndim),
        "resolved_spatial_ndim": spatial_ndim,
    }
    if method == "vipp_cpu":
        return ops.subtract_background(arr, **kwargs)
    if method == "vipp_gpu":
        import cupy

        from napari_vipp.core.gpu import cupy_background

        result = cupy_background.subtract_background(cupy.asarray(arr), **kwargs)
        host = cupy.asnumpy(result, blocking=True)
        cupy.cuda.get_current_stream().synchronize()
        return host
    corrected = raw_corrected(
        arr,
        method=method,
        radius=radius,
        smoothing=smoothing,
        light_background=light_background,
        spatial_ndim=spatial_ndim,
        threads=threads,
    )
    if clip_negative:
        corrected = np.maximum(corrected, 0)
    return ops._restore_numeric_dtype(corrected, arr)
