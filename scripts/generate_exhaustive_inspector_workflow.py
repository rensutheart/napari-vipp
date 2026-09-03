"""Generate the manual, exhaustive inspector showcase workflow.

The resulting graph is intentionally broad rather than a single linear
analysis.  Each lane uses a small bundled sample that suits the represented
operations, and every operation exposed in the node palette appears at least
once. Long shared routes use named tunnels while nearby connections remain
visible, so the graph stays readable without hiding its processing structure.
Keep this generator deterministic so the checked-in JSON remains easy to
review and regenerate after the palette changes.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from napari_vipp.core.pipeline import (
    PALETTE_NODE_LIBRARY,
    GraphNode,
    PrototypePipeline,
)
from napari_vipp.core.workflow import save_workflow

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    REPOSITORY_ROOT / "examples" / "manual" / "exhaustive-inspector-showcase.json"
)


def build_workflow() -> tuple[
    PrototypePipeline,
    dict[str, tuple[float, float]],
    list[dict[str, Any]],
]:
    """Return the exhaustive graph, positions, and explanatory canvas notes."""

    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    positions: dict[str, tuple[float, float]] = {}
    notes: list[dict[str, Any]] = []

    def place(
        operation_id: str,
        x: float,
        y: float,
        /,
        **params: Any,
    ) -> GraphNode:
        node = pipeline.add_node(operation_id)
        node.params.update(params)
        positions[node.id] = (x, y)
        return node

    def source(
        sample_name: str,
        x: float,
        y: float,
        /,
        *,
        channel_colors: str = "",
        first: bool = False,
    ) -> GraphNode:
        if first:
            node = pipeline.nodes["input"]
        else:
            node = pipeline.add_node("input")
        node.params.update(
            {
                "source_mode": "sample",
                "layer_name": "",
                "file_path": "",
                "sample_name": sample_name,
                "series_index": 0,
                "channel_colors": channel_colors,
                "binding_mode": "single item",
                "axis_declaration": "",
            }
        )
        positions[node.id] = (x, y)
        return node

    def wire(
        upstream: GraphNode,
        downstream: GraphNode,
        /,
        *,
        target_port: int = 0,
        source_port: int = 0,
        tunnel_name: str = "",
    ) -> None:
        result = pipeline.connect(
            upstream.id,
            downstream.id,
            target_port=target_port,
            source_port=source_port,
            tunnel_name=tunnel_name,
        )
        if not result.success:
            raise RuntimeError(
                f"Could not connect {upstream.id}:{source_port} to "
                f"{downstream.id}:{target_port}: {result.message}"
            )

    def add_tunnel(name: str, upstream: GraphNode, source_port: int = 0) -> str:
        return pipeline.add_output_tunnel(name, upstream.id, source_port).name

    def lane_note(identifier: str, text: str, x: float, y: float) -> None:
        notes.append(
            {
                "id": identifier,
                "text": text,
                "position": (x, y),
                "width": 360.0,
            }
        )

    # Lane 1: axes, regions, metadata, projections, and a generated PSF.
    lane_note(
        "lane_axes",
        "1. AXES, REGIONS, METADATA, PROJECTIONS, AND GENERATED PSF\n"
        "A TCZYX sample is cropped, reduced to one time point, split by channel, "
        "and inspected through spatial metadata and projection operations.",
        -440,
        -40,
    )
    time_source = source(
        "VIPP synthetic time-lapse multichannel",
        0,
        0,
        channel_colors="Blue,Green,Red",
        first=True,
    )
    crop = place(
        "crop_stack",
        340,
        0,
        z_start=1,
        z_end=1,
        top=4,
        bottom=4,
        left=6,
        right=6,
        channel_axis=-1,
    )
    time_slice = place(
        "select_axis_slice",
        680,
        0,
        axis=0,
        index=2,
        axes="0",
        indices="2",
        ranges="",
        range_mode=True,
        remove_axes="0",
        remove_indices="2",
    )
    split_axis = place("split_axis", 1020, 0, axis="axis:0")
    reorder = place("reorder_axes", 1360, 0, order="YXZ")
    wire(time_source, crop)
    wire(crop, time_slice)
    wire(time_slice, split_axis)
    wire(split_axis, reorder, source_port=0)

    metadata = place(
        "set_microscope_metadata",
        1020,
        330,
        channel_1_wavelength_nm=405.0,
        channel_2_wavelength_nm=488.0,
        channel_3_wavelength_nm=561.0,
        numerical_aperture=1.4,
        refractive_index=1.518,
    )
    pixel_size = place(
        "set_pixel_size",
        1360,
        330,
        x_size=0.12,
        y_size=0.12,
        z_size=0.35,
        unit="micrometer",
    )
    rescale_axes = place(
        "rescale_axes",
        1700,
        330,
        x_scale=0.75,
        y_scale=0.75,
        z_scale=1.0,
        lock_xy=True,
        interpolation="Auto",
        anti_aliasing=True,
    )
    generated_psf = place(
        "born_wolf_psf",
        2040,
        330,
        spatial_mode="Auto from axes",
        auto_parameters=True,
        wavelength_nm=0.0,
        numerical_aperture=0.0,
        refractive_index=0.0,
        pixel_size_xy_um=0.0,
        z_step_um=0.0,
        xy_size=17,
        z_size=7,
        channel=-1,
        pupil_samples=64,
        normalize=True,
    )
    generated_psf_tunnel = add_tunnel("Born-Wolf PSF", generated_psf)
    wire(time_slice, metadata)
    wire(metadata, pixel_size)
    wire(pixel_size, rescale_axes)
    wire(pixel_size, generated_psf)

    mip = place("mip", 1700, -10, axis=2)
    project = place("project_image", 2040, -10, axes="auto", method="Mean")
    orthogonal = place(
        "orthogonal_projection",
        2380,
        -10,
        method="Maximum",
        use_physical_scale=True,
    )
    wire(reorder, mip)
    wire(reorder, project)
    wire(reorder, orthogonal)

    # Lane 2: intensity transformations and the full filtering family.
    lane_note(
        "lane_filtering",
        "2. INTENSITY AND FILTERING\n"
        "A small ZYX fluorescence volume drives an intensity-adjustment chain "
        "plus parallel smoothing, background, edge, and detail branches.",
        -440,
        980,
    )
    volume_source = source("VIPP synthetic volume", 0, 1020)
    raw_volume_tunnel = add_tunnel("Raw volume", volume_source)
    scale_offset = place("linear_scale_offset", 340, 1020, alpha=1.15, beta=4.0)
    gamma = place("gamma_correction", 680, 1020, gamma=0.85)
    rescale_intensity = place(
        "rescale_intensity",
        1020,
        1020,
        cutoff_mode="Percentiles",
        in_low_value=0.0,
        in_high_value=1.0,
        in_low_percentile=1.0,
        in_high_percentile=99.0,
        out_min=0.0,
        out_max=1.0,
    )
    normalize = place("normalize_image", 1360, 1020, method="z-score")
    clamp = place(
        "clip_intensity",
        1700,
        1020,
        cutoff_mode="Values",
        minimum=-2.0,
        maximum=2.0,
    )
    wire(volume_source, scale_offset)
    wire(scale_offset, gamma)
    wire(gamma, rescale_intensity)
    wire(rescale_intensity, normalize)
    wire(normalize, clamp)

    average = place("average_blur", 340, 1320, size=3, channel_axis=-1)
    median = place("median_filter", 680, 1320, size=3, channel_axis=-1)
    sigma = place(
        "sigma_filter",
        1020,
        1320,
        radius=1.5,
        sigma_width=2.0,
        minimum_pixel_fraction=0.2,
        outlier_aware=True,
        channel_axis=-1,
    )
    bilateral = place(
        "bilateral_filter",
        1360,
        1320,
        diameter=5,
        sigma_color=12.0,
        sigma_space=3.0,
        channel_axis=-1,
    )
    non_local = place(
        "non_local_means_filter",
        1700,
        1320,
        patch_size=3,
        patch_distance=4,
        h=0.08,
        fast_mode=True,
        channel_axis=-1,
    )
    wire(volume_source, average)
    wire(average, median)
    wire(median, sigma)
    wire(sigma, bilateral)
    wire(bilateral, non_local)

    gaussian = place("gaussian_blur", 340, 1620, sigma=1.2, channel_axis=-1)
    unsharp = place(
        "unsharp_mask",
        680,
        1620,
        radius=1.0,
        amount=1.25,
        channel_axis=-1,
    )
    sobel = place("sobel_filter", 1020, 1620, channel_axis=-1)
    laplace = place("laplace_filter", 1360, 1620, kernel_size=3, channel_axis=-1)
    wire(volume_source, gaussian)
    wire(gaussian, unsharp)
    wire(unsharp, sobel)
    wire(sobel, laplace)

    gaussian_3d = place(
        "gaussian_blur_3d",
        340,
        1920,
        sigma_z=0.8,
        sigma_y=1.2,
        sigma_x=1.2,
        lock_xy=True,
        channel_axis=-1,
    )
    rolling = place(
        "rolling_ball_background",
        680,
        1920,
        radius=12.0,
        light_background=False,
        disable_smoothing=False,
        spatial_mode="2D YX",
        channel_axis=-1,
    )
    subtract_background = place(
        "subtract_background",
        1020,
        1920,
        radius=12.0,
        light_background=False,
        disable_smoothing=False,
        clip_negative=True,
        spatial_mode="2D YX",
        channel_axis=-1,
    )
    dog = place(
        "difference_of_gaussians",
        1360,
        1920,
        low_sigma=1.0,
        high_sigma=2.5,
        channel_axis=-1,
    )
    canny = place(
        "canny_edges",
        1700,
        1920,
        sigma=1.0,
        low_quantile=0.1,
        high_quantile=0.25,
        channel_axis=-1,
    )
    wire(volume_source, gaussian_3d)
    for filtered in (rolling, subtract_background, dog, canny):
        wire(volume_source, filtered, tunnel_name=raw_volume_tunnel)

    # Lane 3: channels, RGB, arithmetic, all threshold families, and logic.
    lane_note(
        "lane_channels",
        "3. CHANNELS, RGB, THRESHOLDS, AND IMAGE MATH\n"
        "Two registered fluorescence channels feed composites, arithmetic, "
        "logical masks, and a multichannel histogram; a compact bimodal "
        "phantom keeps the Minimum-threshold demonstration deterministic.",
        -440,
        2720,
    )
    coloc_source = source(
        "VIPP synthetic colocalization",
        0,
        2760,
        channel_colors="Red,Green",
    )
    split_channels = place("split_channels", 340, 2760, preview_channel=0)
    wire(coloc_source, split_channels)
    red_channel_tunnel = add_tunnel("Red channel", split_channels, 0)
    green_channel_tunnel = add_tunnel("Green channel", split_channels, 1)

    combine = place("combine_channels", 680, 2760, input_count=2)
    colors = place("assign_channel_colors", 1020, 2760, channel_colors="Red,Green")
    rgb = place(
        "composite_to_rgb",
        1360,
        2760,
        channel_axis=-1,
        red_channel=-1,
        green_channel=-1,
        blue_channel=-1,
        intensity_mapping="Preserve numeric values",
    )
    histogram = place(
        "intensity_histogram",
        1700,
        2760,
        bin_count=128,
        range_mode="Data range",
        custom_min=0.0,
        custom_max=1.0,
        bin_spacing="Linear",
    )
    safe_save = place(
        "save_output",
        1700,
        3060,
        enabled="off",
        path="",
        format="auto",
        overwrite="no",
    )
    wire(split_channels, combine, target_port=0, source_port=0)
    wire(split_channels, combine, target_port=1, source_port=1)
    wire(combine, colors)
    wire(colors, rgb)
    wire(colors, histogram)
    wire(rgb, safe_save)

    extract = place("extract_channel", 680, 3060, channel=0)
    converted = place(
        "convert_dtype", 1020, 3060, output_dtype="uint8", scaling="rescale"
    )
    imagej = place(
        "imagej_auto_threshold", 1360, 3060, method="Default", channel_axis=-1
    )
    wire(colors, extract)
    wire(extract, converted)
    wire(converted, imagej)

    weighted = place(
        "calculate_weighted_image",
        680,
        3380,
        input_count=2,
        weights="0.65,0.35",
        offset=0.0,
    )
    added = place("add_images", 1020, 3380, input_count=2)
    subtracted = place("subtract_images", 1360, 3380, input_count=2)
    ratio = place("ratio_image", 1700, 3380, input_count=2, epsilon=1e-6)
    for math_node in (weighted, added, subtracted, ratio):
        wire(
            split_channels,
            math_node,
            target_port=0,
            source_port=0,
            tunnel_name=red_channel_tunnel,
        )
        wire(
            split_channels,
            math_node,
            target_port=1,
            source_port=1,
            tunnel_name=green_channel_tunnel,
        )

    threshold_source = source("VIPP synthetic threshold gallery", 340, 3700)
    binary = place("binary_threshold", 680, 3700, threshold=12000.0, channel_axis=-1)
    roi_mask_tunnel = add_tunnel("ROI mask", binary)
    otsu = place(
        "otsu_threshold",
        1020,
        3700,
        threshold_scope="Stack histogram",
        histogram_bins=256,
        channel_axis=-1,
    )
    triangle = place(
        "triangle_threshold",
        1360,
        3700,
        threshold_scope="Stack histogram",
        histogram_bins=256,
        channel_axis=-1,
    )
    li = place(
        "li_threshold",
        1700,
        3700,
        threshold_scope="Stack histogram",
        channel_axis=-1,
    )
    isodata = place(
        "isodata_threshold",
        2040,
        3700,
        threshold_scope="Stack histogram",
        histogram_bins=256,
        channel_axis=-1,
    )
    minimum = place(
        "minimum_threshold",
        2380,
        3700,
        threshold_scope="Stack histogram",
        histogram_bins=256,
        max_iterations=10000,
        channel_axis=-1,
    )
    wire(
        split_channels,
        binary,
        source_port=0,
        tunnel_name=red_channel_tunnel,
    )
    wire(
        split_channels,
        otsu,
        source_port=0,
        tunnel_name=red_channel_tunnel,
    )
    wire(
        split_channels,
        triangle,
        source_port=1,
        tunnel_name=green_channel_tunnel,
    )
    wire(
        split_channels,
        li,
        source_port=0,
        tunnel_name=red_channel_tunnel,
    )
    wire(
        split_channels,
        isodata,
        source_port=0,
        tunnel_name=red_channel_tunnel,
    )
    wire(threshold_source, minimum)

    hysteresis = place(
        "hysteresis_threshold",
        680,
        4010,
        low_threshold=8000.0,
        high_threshold=22000.0,
        spatial_mode="Auto from axes",
        channel_axis=-1,
    )
    adaptive_mean = place(
        "adaptive_mean_threshold",
        1020,
        4010,
        block_size=15,
        c=2.0,
        channel_axis=-1,
    )
    adaptive_gaussian = place(
        "adaptive_gaussian_threshold",
        1360,
        4010,
        block_size=15,
        c=2.0,
        channel_axis=-1,
    )
    sauvola = place(
        "sauvola_threshold",
        1700,
        4010,
        window_size=15,
        k=0.2,
        dynamic_range=0.0,
        channel_axis=-1,
    )
    niblack = place(
        "niblack_threshold",
        2040,
        4010,
        window_size=15,
        k=0.2,
        channel_axis=-1,
    )
    wire(
        split_channels,
        hysteresis,
        source_port=0,
        tunnel_name=red_channel_tunnel,
    )
    wire(
        split_channels,
        adaptive_mean,
        source_port=0,
        tunnel_name=red_channel_tunnel,
    )
    wire(
        split_channels,
        adaptive_gaussian,
        source_port=1,
        tunnel_name=green_channel_tunnel,
    )
    wire(
        split_channels,
        sauvola,
        source_port=0,
        tunnel_name=red_channel_tunnel,
    )
    wire(
        split_channels,
        niblack,
        source_port=1,
        tunnel_name=green_channel_tunnel,
    )

    mask_image = place("mask_image", 2040, 3380, outside_value=0.0, invert_mask="no")
    inverted = place("invert", 2380, 3380)
    logical_and = place("logical_and", 2380, 4010, input_count=2)
    logical_or = place("logical_or", 2720, 4010, input_count=2)
    logical_xor = place("logical_xor", 3060, 4010, input_count=2)
    wire(weighted, mask_image, target_port=0)
    wire(binary, mask_image, target_port=1)
    wire(mask_image, inverted)
    wire(otsu, logical_and, target_port=0)
    wire(li, logical_and, target_port=1)
    wire(triangle, logical_or, target_port=0)
    wire(isodata, logical_or, target_port=1)
    wire(adaptive_mean, logical_xor, target_port=0)
    wire(adaptive_gaussian, logical_xor, target_port=1)

    # Lane 4: segmentation cleanup, morphology, labels, measurements, and tables.
    lane_note(
        "lane_objects",
        "4. MORPHOLOGY, OBJECT SEPARATION, LABELS, AND TABLES\n"
        "A thresholded 3D fluorescence channel is cleaned and separated, then "
        "measured through object, intensity, mesh, and table-summary paths.",
        -440,
        4750,
    )
    dilate = place("dilate", 680, 4790, size=3, iterations=1)
    erode = place("erode", 1020, 4790, size=3, iterations=1)
    opening = place("opening", 1360, 4790, size=2)
    closing = place("closing", 1700, 4790, size=2)
    top_hat = place("top_hat", 2040, 4790, size=2)
    black_hat = place("black_hat", 2380, 4790, size=2)
    gradient = place("morphological_gradient", 2720, 4790, size=2)
    for morph in (dilate, erode, opening, closing, top_hat, black_hat, gradient):
        wire(binary, morph, tunnel_name=roi_mask_tunnel)

    outliers = place(
        "remove_binary_outliers",
        680,
        5110,
        radius=1.5,
        which_outliers="Foreground (remove)",
    )
    holes = place(
        "fill_holes",
        1020,
        5110,
        max_hole_size=0,
        spatial_mode="Auto from axes",
        connectivity="Face connected",
    )
    small_objects = place(
        "remove_small_objects",
        1360,
        5110,
        min_size=24,
        spatial_mode="Auto from axes",
        connectivity="Face connected",
    )
    wire(binary, outliers)
    wire(outliers, holes)
    wire(holes, small_objects)

    distance = place(
        "euclidean_distance_transform",
        1700,
        5110,
        spatial_mode="Auto from axes",
    )
    markers = place(
        "h_maxima_markers",
        2040,
        5110,
        h=1.0,
        spatial_mode="Auto from axes",
        connectivity="Full connectivity",
    )
    watershed = place(
        "marker_controlled_watershed",
        2380,
        5110,
        image_mode="Distance map (invert)",
        compactness=0.0,
        watershed_line=False,
        spatial_mode="Auto from axes",
    )
    expanded = place(
        "expand_labels",
        2720,
        5110,
        distance=2.0,
        spatial_mode="Auto from axes",
    )
    expanded_labels_tunnel = add_tunnel("Expanded labels", expanded)
    auto_watershed = place(
        "auto_watershed_from_mask",
        2380,
        5430,
        h=1.0,
        connectivity="Full connectivity",
        image_mode="Distance map (invert)",
        compactness=0.0,
        watershed_line=False,
        spatial_mode="Auto from axes",
    )
    watershed_labels_tunnel = add_tunnel("Watershed labels", auto_watershed)
    wire(small_objects, distance)
    wire(distance, markers)
    wire(distance, watershed, target_port=0)
    wire(markers, watershed, target_port=1)
    wire(small_objects, watershed, target_port=2)
    wire(watershed, expanded)
    wire(small_objects, auto_watershed)

    components = place(
        "label_connected_components",
        1700,
        5430,
        spatial_mode="Auto from axes",
        connectivity="Full connectivity",
    )
    clear_border = place(
        "clear_border_objects",
        2040,
        5750,
        border_buffer=0,
        boundary_mode="All spatial borders",
    )
    volume_filter = place(
        "filter_labels_by_volume",
        2380,
        5750,
        min_volume=24,
        max_volume=0,
        spatial_mode="Auto from axes",
    )
    relabel = place("relabel_sequential", 2720, 5750, spatial_mode="Auto from axes")
    object_labels_tunnel = add_tunnel("Object labels", relabel)
    wire(small_objects, components)
    wire(components, clear_border)
    wire(clear_border, volume_filter)
    wire(volume_filter, relabel)

    object_table = place(
        "measure_objects",
        3060,
        5750,
        spatial_mode="Auto from axes",
        include_shape_descriptors=True,
        include_axis_descriptors=True,
        include_2d_boundary_descriptors=False,
        include_derived_shape_ratios=True,
        include_2d_shape_moments=False,
    )
    property_filter = place(
        "filter_labels_by_property",
        3400,
        5750,
        property_column="volume_voxels",
        min_value=24.0,
        max_value=0.0,
        keep_mode="Keep inside range",
        unmatched_labels="Remove unmatched labels",
        spatial_mode="Auto from axes",
    )
    intensity_table = place(
        "measure_objects_intensity",
        3740,
        5750,
        spatial_mode="Auto from axes",
        include_shape_descriptors=False,
        include_axis_descriptors=False,
        include_2d_boundary_descriptors=False,
        include_derived_shape_ratios=False,
        include_2d_shape_moments=False,
    )
    mesh_table = place(
        "measure_3d_mesh_morphology",
        3740,
        6070,
        spatial_mode="Auto from axes",
        minimum_voxel_count=16,
        include_convex_hull_metrics=True,
    )
    wire(relabel, object_table)
    wire(relabel, property_filter, target_port=0)
    wire(object_table, property_filter, target_port=1)
    wire(property_filter, intensity_table, target_port=0)
    wire(
        split_channels,
        intensity_table,
        target_port=1,
        source_port=0,
        tunnel_name=red_channel_tunnel,
    )
    wire(property_filter, mesh_table)

    merged = place(
        "merge_tables",
        4080,
        5750,
        input_count=2,
        join_mode="Left join",
        join_keys="auto",
    )
    add_metadata = place(
        "add_metadata_columns",
        4420,
        5750,
        metadata_columns="condition=inspector_demo,sample=synthetic_colocalization",
        overwrite="no",
    )
    select_columns = place("select_table_columns", 4760, 5750, columns="auto")
    summarize = place(
        "summarize_measurements",
        5100,
        5750,
        group_by="auto",
        value_columns="auto",
        statistics="count,mean,median,std,min,max,q25,q75",
    )
    batch = place(
        "batch_output",
        5440,
        5750,
        tag="inspector_demo_summary",
        format="batch default",
        subfolder="inspector-demo",
        filename_template="{source_stem}__{tag}",
        overwrite="batch default",
    )
    wire(object_table, merged, target_port=0)
    wire(intensity_table, merged, target_port=1)
    wire(merged, add_metadata)
    wire(add_metadata, select_columns)
    wire(select_columns, summarize)
    wire(summarize, batch)

    # Lane 5: colocalization and spatial relationships on the same registered grid.
    lane_note(
        "lane_colocalization",
        "5. COLOCALIZATION AND SPATIAL ASSOCIATION\n"
        "The same two channels and ROI mask are reused across unmasked, masked, "
        "voxel, object, overlap, nearest-distance, and event-localization views.",
        -440,
        6550,
    )
    coloc_metrics = place(
        "colocalization_metrics",
        680,
        6590,
        threshold_mode="Manual",
        channel_1_threshold=12000.0,
        channel_2_threshold=12000.0,
    )
    masked_metrics = place(
        "masked_colocalization_metrics",
        1020,
        6590,
        threshold_mode="Manual",
        channel_1_threshold=12000.0,
        channel_2_threshold=12000.0,
    )
    scatter = place(
        "colocalization_scatter_plot",
        1360,
        6590,
        threshold_mode="Manual",
        channel_1_threshold=12000.0,
        channel_2_threshold=12000.0,
        bins=96,
        output_size=512,
        range_percentile=99.5,
        log_counts=True,
    )
    masked_scatter = place(
        "masked_colocalization_scatter_plot",
        1700,
        6590,
        threshold_mode="Manual",
        channel_1_threshold=12000.0,
        channel_2_threshold=12000.0,
        bins=96,
        output_size=512,
        range_percentile=99.5,
        log_counts=True,
    )
    coloc_voxels = place(
        "colocalized_voxels",
        2040,
        6590,
        threshold_mode="Manual",
        channel_1_threshold=12000.0,
        channel_2_threshold=12000.0,
        display_mode="White overlay on channels",
        channel_1_color="Red",
        channel_2_color="Green",
    )
    masked_voxels = place(
        "masked_colocalized_voxels",
        2380,
        6590,
        threshold_mode="Manual",
        channel_1_threshold=12000.0,
        channel_2_threshold=12000.0,
        display_mode="White overlay on channels",
        channel_1_color="Red",
        channel_2_color="Green",
    )
    racc = place(
        "racc_index",
        2720,
        6590,
        threshold_mode="Manual",
        channel_1_threshold=12000.0,
        channel_2_threshold=12000.0,
        theta_degrees=45.0,
        include_percentile=99.0,
        output_dtype="float32",
    )
    masked_racc = place(
        "masked_racc_index",
        3060,
        6590,
        threshold_mode="Manual",
        channel_1_threshold=12000.0,
        channel_2_threshold=12000.0,
        theta_degrees=45.0,
        include_percentile=99.0,
        output_dtype="float32",
    )

    unmasked_pairs = (coloc_metrics, scatter, coloc_voxels, racc)
    masked_triples = (masked_metrics, masked_scatter, masked_voxels, masked_racc)
    for analysis in unmasked_pairs:
        wire(
            split_channels,
            analysis,
            target_port=0,
            source_port=0,
            tunnel_name=red_channel_tunnel,
        )
        wire(
            split_channels,
            analysis,
            target_port=1,
            source_port=1,
            tunnel_name=green_channel_tunnel,
        )
    for analysis in masked_triples:
        wire(
            split_channels,
            analysis,
            target_port=0,
            source_port=0,
            tunnel_name=red_channel_tunnel,
        )
        wire(
            split_channels,
            analysis,
            target_port=1,
            source_port=1,
            tunnel_name=green_channel_tunnel,
        )
        wire(binary, analysis, target_port=2, tunnel_name=roi_mask_tunnel)

    object_coloc = place(
        "object_colocalization_metrics",
        3400,
        6590,
        spatial_mode="Auto from axes",
        threshold_mode="Manual",
        channel_1_threshold=12000.0,
        channel_2_threshold=12000.0,
    )
    overlap = place(
        "label_overlap_association",
        3740,
        6590,
        spatial_mode="Auto from axes",
    )
    nearest = place(
        "nearest_object_distance",
        4080,
        6590,
        spatial_mode="Auto from axes",
    )
    events = place(
        "event_localization",
        4420,
        6590,
        spatial_mode="Auto from axes",
    )
    wire(relabel, object_coloc, target_port=0, tunnel_name=object_labels_tunnel)
    wire(
        split_channels,
        object_coloc,
        target_port=1,
        source_port=0,
        tunnel_name=red_channel_tunnel,
    )
    wire(
        split_channels,
        object_coloc,
        target_port=2,
        source_port=1,
        tunnel_name=green_channel_tunnel,
    )
    wire(relabel, overlap, target_port=0, tunnel_name=object_labels_tunnel)
    wire(
        auto_watershed,
        overlap,
        target_port=1,
        tunnel_name=watershed_labels_tunnel,
    )
    wire(relabel, nearest, target_port=0, tunnel_name=object_labels_tunnel)
    wire(
        auto_watershed,
        nearest,
        target_port=1,
        tunnel_name=watershed_labels_tunnel,
    )
    wire(expanded, events, target_port=0, tunnel_name=expanded_labels_tunnel)
    wire(relabel, events, target_port=1, tunnel_name=object_labels_tunnel)

    # Lane 6: skeleton QC, labels, graph tables, and summaries.
    lane_note(
        "lane_skeleton",
        "6. SKELETON QC AND NETWORK MEASUREMENTS\n"
        "A sparse 3D network is thresholded and skeletonized before parallel "
        "topology, labeling, pruning, overlay, graph-table, and summary outputs.",
        -440,
        7430,
    )
    skeleton_source = source("VIPP synthetic skeleton network", 0, 7470)
    yen = place(
        "yen_threshold",
        340,
        7470,
        threshold_scope="Stack histogram",
        histogram_bins=256,
        channel_axis=-1,
    )
    skeleton = place(
        "skeletonize",
        680,
        7470,
        spatial_mode="Auto from axes",
        method="Auto",
    )
    skeleton_mask_tunnel = add_tunnel("Skeleton mask", skeleton)
    wire(skeleton_source, yen)
    wire(yen, skeleton)

    analyze = place(
        "analyze_skeleton",
        1020,
        7470,
        spatial_mode="Auto from axes",
        input_mode="Already skeletonized",
    )
    branches = place(
        "measure_skeleton_branches",
        1360,
        7470,
        spatial_mode="Auto from axes",
        input_mode="Already skeletonized",
    )
    branch_summary = place(
        "summarize_skeleton_branches",
        1700,
        7470,
        group_by="auto",
        statistics="mean,median,std,min,max,q25,q75",
    )
    graph_tables = place(
        "skeleton_graph_tables",
        2040,
        7470,
        spatial_mode="Auto from axes",
        input_mode="Already skeletonized",
    )
    overall = place(
        "measure_overall_skeleton_network",
        2380,
        7470,
        spatial_mode="Auto from axes",
        input_mode="Already skeletonized",
    )
    for measurement in (analyze, branches, graph_tables, overall):
        wire(skeleton, measurement)
    wire(branches, branch_summary)

    keypoints = place("skeleton_keypoints", 1020, 7790, spatial_mode="Auto from axes")
    overlay = place(
        "skeleton_graph_overlay",
        1360,
        7790,
        spatial_mode="Auto from axes",
        display_mode="Colored edges + colored nodes",
        node_size=2,
    )
    skeleton_components = place(
        "label_skeleton_components",
        1700,
        7790,
        spatial_mode="Auto from axes",
    )
    skeleton_branches = place(
        "label_skeleton_branches",
        2040,
        7790,
        spatial_mode="Auto from axes",
    )
    prune = place(
        "prune_skeleton_branches",
        2380,
        7790,
        min_branch_length=3.0,
        length_units="Pixels/voxels",
        iterations=1,
        remove_isolated=True,
        spatial_mode="Auto from axes",
    )
    for qc_node in (
        keypoints,
        overlay,
        skeleton_components,
        skeleton_branches,
        prune,
    ):
        wire(skeleton, qc_node, tunnel_name=skeleton_mask_tunnel)

    # Lane 7: measured and generated PSF restoration. Deconvolution nodes remain
    # manual by schema, so loading the showcase does not immediately run them.
    lane_note(
        "lane_restoration",
        "7. PSF PREPARATION AND DECONVOLUTION\n"
        "A measured 3D PSF is prepared for Richardson-Lucy restoration, while "
        "the metadata-derived Born-Wolf PSF drives the TV-regularized variant.",
        -440,
        8650,
    )
    deconv_source = source("VIPP synthetic 3D deconvolution volume", 0, 8690)
    measured_psf_source = source("VIPP synthetic 3D measured PSF", 0, 9010)
    prepared_psf = place(
        "prepare_validate_psf",
        340,
        9010,
        center_mode="Peak",
        clip_negatives=True,
        normalize_sum=True,
        minimum_valid_sum=1e-12,
        force_odd_shape=True,
        crop_empty_border=False,
    )
    rl = place(
        "richardson_lucy_deconvolution",
        680,
        8690,
        spatial_mode="Auto from axes",
        iterations=5,
        normalize_psf=True,
        clip_negative_input=True,
        clip_output_negative=True,
        preserve_input_scale=True,
        filter_epsilon=1e-12,
    )
    rl_tv = place(
        "richardson_lucy_tv_deconvolution",
        1020,
        8690,
        spatial_mode="Auto from axes",
        iterations=5,
        tv_regularization=0.002,
        tv_epsilon=1e-6,
        normalize_psf=True,
        clip_negative_input=True,
        clip_output_negative=True,
        preserve_input_scale=True,
        filter_epsilon=1e-12,
        denominator_floor=0.05,
    )
    wire(measured_psf_source, prepared_psf)
    wire(deconv_source, rl, target_port=0)
    wire(prepared_psf, rl, target_port=1)
    wire(deconv_source, rl_tv, target_port=0)
    wire(
        generated_psf,
        rl_tv,
        target_port=1,
        source_port=0,
        tunnel_name=generated_psf_tunnel,
    )

    operation_counts = Counter(node.operation_id for node in pipeline.nodes.values())
    expected = {spec.id for spec in PALETTE_NODE_LIBRARY}
    actual = set(operation_counts)
    if actual != expected:
        raise RuntimeError(
            "Exhaustive workflow coverage mismatch: "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    duplicates = {
        operation_id: count
        for operation_id, count in operation_counts.items()
        if operation_id != "input" and count != 1
    }
    if duplicates:
        raise RuntimeError(f"Non-source operation duplicates: {duplicates}")
    if set(positions) != set(pipeline.nodes):
        raise RuntimeError("Every showcase node must have a canvas position.")

    return pipeline, positions, notes


def main() -> None:
    pipeline, positions, notes = build_workflow()
    metadata = {
        "vipp": {
            "inspector": {
                "selected_node_id": "input",
                "right_panel_visible": True,
            }
        }
    }
    target = save_workflow(
        OUTPUT_PATH,
        pipeline,
        positions=positions,
        notes=notes,
        metadata=metadata,
    )
    print(
        f"Wrote {target} with {len(pipeline.nodes)} nodes, "
        f"{len(pipeline.connections)} connections, and "
        f"{len({node.operation_id for node in pipeline.nodes.values()})} operations."
    )


if __name__ == "__main__":
    main()
