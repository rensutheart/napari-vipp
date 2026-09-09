"""Presentation-only alternative names shared by node search surfaces.

Keep alternatives operation-specific: generic stemming can conflate distinct
scientific methods. These terms aid discovery, never rename operations, change
workflow IDs, or imply that two algorithms are interchangeable.
"""

from __future__ import annotations

from functools import lru_cache
from types import MappingProxyType

OPERATION_SEARCH_ALIASES = MappingProxyType(
    {
        "dilate": ("dilate", "dilating", "grow mask", "expand foreground"),
        "erode": ("erode", "eroding", "shrink mask", "shrink foreground"),
        "opening": ("open mask", "binary opening", "morphological opening"),
        "closing": ("close mask", "binary closing", "morphological closing"),
        "convex_hull": ("convex envelope", "convexhull"),
        "mask_to_3d_mesh": ("marching cubes", "surface mesh", "export obj", "3mf"),
        "labels_to_3d_mesh": (
            "marching cubes",
            "label surfaces",
            "object meshes",
            "3mf",
        ),
        "combine_meshes": ("merge meshes", "mesh collection", "collect surfaces"),
        "split_mesh_objects": (
            "separate mesh",
            "disconnected components",
            "split surfaces",
        ),
        "filter_mesh_objects": (
            "select mesh objects",
            "mesh size",
            "remove small meshes",
        ),
        "color_mesh_objects": (
            "mesh colors",
            "object colours",
            "colour by measurement",
        ),
        "smooth_mesh": ("surface smoothing", "taubin", "mesh smoothing"),
        "simplify_mesh": (
            "decimate",
            "decimation",
            "reduce triangles",
            "quadric error",
        ),
        "fill_holes": ("hole filling", "fill cavities"),
        "remove_small_objects": ("small object removal", "remove specks"),
        "remove_binary_outliers": ("outlier removal",),
        "clear_border_objects": ("remove border objects", "clear border"),
        "find_label_boundaries": (
            "label outlines", "object outlines", "boundary mask", "contours",
        ),
        "skeletonize": ("skeletonization", "thinning"),
        "label_connected_components": (
            "connected component labeling",
            "component labeling",
            "ccl",
        ),
        "relabel_sequential": ("relabeling", "compact labels"),
        "average_blur": ("mean filter", "box blur", "averaging"),
        "gaussian_blur": ("gaussian smoothing",),
        "gaussian_blur_3d": ("3d gaussian smoothing",),
        "median_filter": ("median filtering", "median denoising", "median denoise"),
        "bilateral_filter": (
            "bilateral smoothing",
            "bilateral denoising",
            "bilateral denoise",
        ),
        "non_local_means_filter": (
            "nlm",
            "nonlocal means",
            "non local means denoising",
            "non local means denoise",
        ),
        "subtract_background": ("background subtraction", "background removal"),
        "rolling_ball_background": ("background estimation",),
        "difference_of_gaussians": ("dog",),
        "unsharp_mask": ("sharpen", "sharpening", "unsharp masking"),
        "laplace_filter": ("laplacian", "laplacian filter"),
        "mip": ("max projection", "maximum intensity projection"),
        "project_image": ("projection",),
        "euclidean_distance_transform": ("edt", "distance map"),
        "binary_threshold": ("binarize", "binarization", "binarizing"),
        "richardson_lucy_deconvolution": ("rl deconvolution",),
        "richardson_lucy_tv_deconvolution": ("rl tv", "total variation deconvolution"),
        "normalize_image": ("normalization", "normalizing"),
        "rescale_intensity": ("rescaling intensity", "contrast stretching"),
        "clip_intensity": ("clip", "clipping", "clamping"),
        "invert": ("inversion", "inverting"),
        "add_images": ("addition", "sum images"),
        "subtract_images": ("subtraction",),
        "ratio_image": ("divide", "division", "divide images"),
        "convert_dtype": ("data type", "dtype conversion", "bit depth"),
        "extract_channel": ("channel extraction",),
        "combine_channels": ("merge channels", "channel merge"),
        "split_channels": ("channel splitting",),
        "assign_channel_colors": ("channel colors",),
        "rescale_axes": ("resize", "resizing", "resampling"),
        "set_pixel_size": ("voxel size", "pixel spacing", "calibration"),
        "measure_objects": ("object measurements", "region properties", "regionprops"),
        "measure_objects_intensity": ("object intensity measurements",),
        "intensity_histogram": ("intensity distribution",),
        "save_output": ("save output", "write image", "export image"),
    }
)


@lru_cache(maxsize=256)
def operation_search_aliases(operation_id: str) -> tuple[str, ...]:
    """Return curated aliases plus common British spelling equivalents."""
    aliases = OPERATION_SEARCH_ALIASES.get(operation_id, ())
    spellings = []
    for phrase in (operation_id.replace("_", " "), *aliases):
        british = phrase
        for american, alternate in (
            ("color", "colour"),
            ("analyz", "analys"),
            ("localiz", "localis"),
            ("normaliz", "normalis"),
            ("summariz", "summaris"),
            ("skeletoniz", "skeletonis"),
            ("binariz", "binaris"),
            ("labeling", "labelling"),
        ):
            british = british.replace(american, alternate)
        if british != phrase:
            spellings.append(british)
    return tuple(dict.fromkeys((*aliases, *spellings)))
