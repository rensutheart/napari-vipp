"""Editable prototype graph model and executor."""

from __future__ import annotations

import inspect
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from numbers import Integral, Real
from typing import Any

from napari_vipp.core.grid import (
    validate_aligned_image_states,
    validate_psf_image_states,
)
from napari_vipp.core.metadata import (
    DEFERRED_VALUE_RANGE,
    ImageState,
    image_state_from_array,
    transform_image_state,
    transform_multi_input_image_state,
    transform_split_output_state,
    with_channel_colors,
)
from napari_vipp.core.operations import (
    BORN_WOLF_PSF_AUTO_PARAMETERS,
    adaptive_gaussian_threshold,
    adaptive_mean_threshold,
    add_images,
    add_metadata_columns,
    analyze_skeleton,
    assign_channel_colors,
    auto_watershed_from_mask,
    average_blur,
    batch_output,
    bilateral_filter,
    binary_threshold,
    black_hat,
    born_wolf_psf_outputs,
    calculate_weighted_image,
    canny_edges,
    clear_border_objects,
    clip_intensity,
    closing,
    colocalization_metrics,
    colocalization_threshold_values,
    colocalized_voxels,
    combine_channels,
    composite_to_rgb,
    convert_dtype,
    crop_stack,
    difference_of_gaussians_filter,
    dilate,
    erode,
    euclidean_distance_transform,
    event_localization,
    expand_labels_image,
    extract_channel,
    fill_holes,
    filter_labels_by_property,
    filter_labels_by_volume,
    gamma_correction,
    gaussian_blur,
    gaussian_blur_3d,
    h_maxima_markers,
    hysteresis_threshold,
    invert,
    isodata_threshold,
    label_connected_components,
    label_overlap_association,
    label_skeleton_branches,
    label_skeleton_components,
    laplace_filter,
    li_threshold,
    linear_scale_offset,
    logical_and,
    logical_or,
    logical_xor,
    marker_controlled_watershed,
    mask_image,
    max_intensity_projection,
    measure_3d_mesh_morphology,
    measure_objects,
    measure_objects_with_intensity,
    measure_overall_skeleton_network,
    measure_skeleton_branches,
    median_filter,
    merge_tables,
    minimum_threshold,
    morphological_gradient,
    nearest_object_distance,
    niblack_threshold,
    non_local_means_filter,
    normalize_image,
    object_colocalization_metrics,
    opening,
    orthogonal_projection,
    otsu_threshold,
    prepare_validate_psf,
    project_image,
    prune_skeleton_branches,
    racc_index,
    ratio_image,
    relabel_sequential,
    remove_small_objects,
    reorder_axes,
    rescale_axes,
    rescale_intensity,
    resolve_born_wolf_psf_parameters,
    richardson_lucy_deconvolution,
    richardson_lucy_tv_deconvolution,
    rolling_ball_background,
    sauvola_threshold,
    save_output,
    select_axis_slice,
    select_table_columns,
    set_pixel_size,
    skeleton_graph_overlay,
    skeleton_graph_tables,
    skeleton_keypoints,
    skeletonize_mask,
    sobel_filter,
    split_axis,
    split_channels,
    subtract_background,
    subtract_images,
    summarize_measurements,
    summarize_skeleton_branches,
    top_hat,
    triangle_threshold,
    unsharp_mask_filter,
    yen_threshold,
)
from napari_vipp.core.progress import ProgressContext
from napari_vipp.core.tables import TableState, table_state_from_data


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    label: str
    kind: str
    default: float | int | str | bool
    minimum: float | int
    maximum: float | int
    step: float | int
    decimals: int = 0
    choices: tuple[str, ...] = ()
    choice_labels: tuple[str, ...] = ()
    dynamic_choices: bool = False


_RESOLVED_SPATIAL_NDIM_PARAMETER = ParameterSpec(
    "resolved_spatial_ndim",
    "Resolved spatial dimensions",
    "int",
    2,
    1,
    3,
    1,
)

_OPTIONAL_PERSISTED_PARAMETER_SPECS: dict[str, tuple[ParameterSpec, ...]] = {
    "select_axis_slice": (
        ParameterSpec("axes", "Selected axes", "text", "", 0, 0, 1),
        ParameterSpec("indices", "Selected indices", "text", "", 0, 0, 1),
        ParameterSpec("ranges", "Selected ranges", "text", "", 0, 0, 1),
        ParameterSpec("range_mode", "Use ranges", "bool", True, 0, 1, 1),
        ParameterSpec("remove_axes", "Removed axes", "text", "", 0, 0, 1),
        ParameterSpec(
            "remove_indices",
            "Removed indices",
            "text",
            "",
            0,
            0,
            1,
        ),
    ),
    "rescale_axes": (
        ParameterSpec(
            "resize_mode",
            "Resize mode",
            "choice",
            "Scale factor",
            0,
            0,
            1,
            choices=("Scale factor", "Output size"),
        ),
        ParameterSpec("x_size", "X output size", "int", 1, 1, 2**31 - 1, 1),
        ParameterSpec("y_size", "Y output size", "int", 1, 1, 2**31 - 1, 1),
        ParameterSpec("z_size", "Z output size", "int", 1, 1, 2**31 - 1, 1),
    ),
    "combine_channels": (
        ParameterSpec("channel_axis", "Channel axis", "int", -1, -1, 64, 1),
        ParameterSpec("channel_colors", "Channel colours", "text", "", 0, 0, 1),
    ),
}


def optional_persisted_parameter_spec(
    operation_spec: OperationSpec,
    name: object,
) -> ParameterSpec | None:
    """Return an explicitly supported, non-required serialized parameter."""
    if not isinstance(name, str):
        return None
    if name == "resolved_spatial_ndim":
        if operation_spec.id in _RESOLVED_SPATIAL_PARAMETER_OPERATION_IDS:
            return _RESOLVED_SPATIAL_NDIM_PARAMETER
        return None
    return next(
        (
            parameter
            for parameter in _OPTIONAL_PERSISTED_PARAMETER_SPECS.get(
                operation_spec.id,
                (),
            )
            if parameter.name == name
        ),
        None,
    )


def validate_optional_persisted_parameter(
    operation_spec: OperationSpec,
    parameter: ParameterSpec,
    value: Any,
    *,
    context: str,
) -> None:
    """Validate saved UI/derived state that is not a required node control."""
    validate_parameter_value(parameter, value, context=context)
    label = f"{context} {parameter.name!r}"
    if parameter.name == "resolved_spatial_ndim" and int(value) not in {1, 2, 3}:
        raise ValueError(f"{label} must be 1, 2, or 3.")
    if operation_spec.id == "rescale_axes" and parameter.name.endswith("_size"):
        if int(value) < 1:
            raise ValueError(f"{label} must be a positive integer.")


def validate_parameter_value(
    spec: ParameterSpec,
    value: Any,
    *,
    context: str = "Parameter",
) -> None:
    """Reject malformed persisted or programmatic parameter values.

    UI bounds remain presentation hints for most numeric parameters because
    several controls intentionally expand to the connected data range. Choice
    membership, scalar type, finiteness, and the scientific histogram-bin
    range are invariant and are therefore validated centrally.
    """
    label = f"{context} {spec.name!r}"
    if spec.kind == "choice":
        if spec.dynamic_choices:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must be a non-empty choice value.")
            return
        if value not in spec.choices:
            choices = ", ".join(repr(choice) for choice in spec.choices)
            raise ValueError(f"{label} must be one of: {choices}.")
        return
    if spec.kind == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{label} must be a boolean.")
        return
    if spec.kind == "int":
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{label} must be an integer.")
        if spec.name in {"histogram_bins", "max_iterations"} and not (
            spec.minimum <= value <= spec.maximum
        ):
            raise ValueError(
                f"{label} must be between {int(spec.minimum):,} and "
                f"{int(spec.maximum):,}."
            )
        return
    if spec.kind == "float":
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{label} must be a finite number.")
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} must be a finite number.")
        return
    if spec.kind == "text" and not isinstance(value, str):
        raise ValueError(f"{label} must be text.")


@dataclass(frozen=True)
class OutputSpec:
    name: str
    output_type: str
    title: str = ""

    @property
    def label(self) -> str:
        return self.title or self.name


@dataclass(frozen=True)
class InputSpec:
    name: str
    input_type: str
    title: str = ""

    @property
    def label(self) -> str:
        return self.title or self.name


@dataclass(frozen=True)
class OperationSpec:
    id: str
    title: str
    category: str
    input_type: str | None
    output_type: str
    parameters: tuple[ParameterSpec, ...] = ()
    function: Callable[..., Any] | None = None
    max_inputs: int | None = 1
    subcategory: str = ""
    outputs: tuple[OutputSpec, ...] = ()
    output_factory: Callable[[int], tuple[OutputSpec, ...]] | None = None
    inputs: tuple[InputSpec, ...] = ()
    preserves_input_type: bool = False
    stack_processing_note: str = ""
    execution_policy: str = "auto"

    @property
    def has_input(self) -> bool:
        return self.input_type is not None or bool(self.inputs)

    @property
    def input_ports(self) -> tuple[InputSpec, ...]:
        if self.inputs:
            return self.inputs
        if self.input_type is None:
            return ()
        return (InputSpec("in", self.input_type, "Input"),)

    @property
    def is_multi_output(self) -> bool:
        """Whether this node can produce more than one output port."""
        return bool(self.outputs) or self.output_factory is not None

    @property
    def output_ports(self) -> tuple[OutputSpec, ...]:
        """Return declared static output ports, or a single default port.

        Nodes with a dynamic ``output_factory`` resolve their ports per node
        instance via ``PrototypePipeline.output_ports`` instead.
        """
        if self.outputs:
            return self.outputs
        return (OutputSpec("out", self.output_type),)


@dataclass(frozen=True)
class SourcePayload:
    data: Any
    metadata: dict | None = None
    name: str = ""
    image_state: ImageState | None = None
    revision_token: object | None = None


@dataclass
class GraphNode:
    id: str
    operation_id: str
    title: str
    category: str
    input_type: str | None
    output_type: str
    params: dict[str, Any] = field(default_factory=dict)
    max_inputs: int | None = 1

    @property
    def has_input(self) -> bool:
        return self.input_type is not None


@dataclass(frozen=True)
class GraphConnection:
    source_id: str
    target_id: str
    target_port: int = 0
    source_port: int = 0
    tunnel_name: str = ""


@dataclass(frozen=True)
class OutputTunnel:
    name: str
    source_id: str
    source_port: int = 0


@dataclass(frozen=True)
class ConnectionResult:
    success: bool
    message: str
    removed: tuple[GraphConnection, ...] = ()
    connection: GraphConnection | None = None


EXECUTION_READY = "ready"
EXECUTION_RUNNING = "running"
EXECUTION_STALE = "stale"
EXECUTION_ERROR = "error"
EXECUTION_NOT_CALCULATED = "not_calculated"
EXECUTION_POLICIES = {"auto", "manual"}
MANUAL_RUN_CALCULATE = "calculate"
MANUAL_RUN_SKIP = "skip"
MANUAL_AUTO_RECALCULATE_PARAM = "_vipp_auto_recalculate"


IMAGE_DATA_CATEGORY = "Image Data"
INTENSITY_CONTRAST_CATEGORY = "Intensity & Contrast"
SOURCE_OUTPUT_GROUP = "Source & Output"
AXES_REGIONS_GROUP = "Axes & Regions"
CHANNELS_COMPOSITES_GROUP = "Channels & Composites"
UTILITIES_GROUP = "Utilities"
MATH_LOGIC_GROUP = "Math & Logic"
LABEL_OPERATIONS_CATEGORY = "Label Operations"
MEASUREMENTS_CATEGORY = "Measurements"
SKELETON_NETWORK_GROUP = "Skeleton / Network QC"
COLOCALIZATION_CATEGORY = "Colocalization & Spatial Analysis"
FILTERING_CATEGORY = "Filtering"
SEGMENTATION_CATEGORY = "Segmentation"
SMOOTHING_DENOISING_GROUP = "Smoothing & Denoising"
EDGE_DETAIL_GROUP = "Edge & Detail"
BACKGROUND_CORRECTION_GROUP = "Background Correction"
RESTORATION_PSF_GROUP = "Restoration & PSF"
GLOBAL_THRESHOLDS_GROUP = "Global Thresholds"
LOCAL_THRESHOLDS_GROUP = "Local Thresholds"
OBJECT_SEPARATION_GROUP = "Object Separation"
SLICE_WISE_STACK_NOTICE = (
    "Stack notice: this node processes each YX slice independently and does not "
    "use 3D neighborhoods. If another plane should be processed, use Reorder "
    "Axes first so the intended plane is YX."
)
GLOBAL_THRESHOLD_OPERATIONS = {
    "otsu_threshold",
    "triangle_threshold",
    "li_threshold",
    "yen_threshold",
    "isodata_threshold",
    "minimum_threshold",
}
COLOCALIZATION_THRESHOLD_OPERATIONS = {
    "colocalization_metrics",
    "masked_colocalization_metrics",
    "colocalized_voxels",
    "masked_colocalized_voxels",
    "racc_index",
    "masked_racc_index",
}
LABEL_METADATA_MULTI_INPUT_OPERATIONS = {
    "event_localization",
    "label_overlap_association",
    "measure_objects_intensity",
    "nearest_object_distance",
    "object_colocalization_metrics",
}
SAME_SHAPE_GRID_OPERATIONS = {
    "add_images",
    "calculate_weighted_image",
    "colocalization_metrics",
    "colocalized_voxels",
    "combine_channels",
    "event_localization",
    "label_overlap_association",
    "logical_and",
    "logical_or",
    "logical_xor",
    "marker_controlled_watershed",
    "mask_image",
    "masked_colocalization_metrics",
    "masked_colocalized_voxels",
    "masked_racc_index",
    "measure_objects_intensity",
    "nearest_object_distance",
    "object_colocalization_metrics",
    "racc_index",
    "ratio_image",
    "subtract_images",
}
PSF_SAMPLING_GRID_OPERATIONS = {
    "richardson_lucy_deconvolution",
    "richardson_lucy_tv_deconvolution",
}

SOURCE_PARAMETERS = (
    ParameterSpec(
        "source_mode",
        "Source",
        "choice",
        "napari layer",
        0,
        0,
        1,
        choices=("napari layer", "file path", "sample"),
    ),
    ParameterSpec("layer_name", "Napari layer", "text", "", 0, 0, 1),
    ParameterSpec("file_path", "File path", "text", "", 0, 0, 1),
    ParameterSpec("sample_name", "Sample", "text", "", 0, 0, 1),
    ParameterSpec("series_index", "Series", "int", 0, 0, 100000, 1),
    ParameterSpec("channel_colors", "Channel colours", "text", "", 0, 0, 1),
    ParameterSpec(
        "binding_mode",
        "Binding",
        "choice",
        "single item",
        0,
        0,
        1,
        choices=("single item", "collection"),
    ),
)

SPATIAL_MODE_PARAMETER = ParameterSpec(
    "spatial_mode",
    "Spatial processing",
    "choice",
    "Auto from axes",
    0,
    0,
    1,
    choices=("Auto from axes", "2D YX", "3D ZYX"),
    dynamic_choices=True,
)

BACKGROUND_SPATIAL_MODE_PARAMETER = ParameterSpec(
    "spatial_mode",
    "Spatial processing",
    "choice",
    "2D YX",
    0,
    0,
    1,
    choices=("2D YX", "3D ZYX", "Auto from axes"),
    dynamic_choices=True,
)

THRESHOLD_SCOPE_PARAMETER = ParameterSpec(
    "threshold_scope",
    "Threshold uses",
    "choice",
    "Stack histogram",
    0,
    0,
    1,
    choices=("Stack histogram", "Slice histogram"),
)
HISTOGRAM_BINS_PARAMETER = ParameterSpec(
    "histogram_bins",
    "Float histogram bins",
    "int",
    256,
    2,
    65_536,
    1,
)

SPATIAL_OPERATIONS = {
    "auto_watershed_from_mask",
    "born_wolf_psf",
    "clear_border_objects",
    "euclidean_distance_transform",
    "expand_labels",
    "event_localization",
    "fill_holes",
    "hysteresis_threshold",
    "h_maxima_markers",
    "label_overlap_association",
    "label_skeleton_branches",
    "label_skeleton_components",
    "measure_skeleton_branches",
    "skeleton_graph_tables",
    "measure_overall_skeleton_network",
    "measure_3d_mesh_morphology",
    "nearest_object_distance",
    "object_colocalization_metrics",
    "label_connected_components",
    "marker_controlled_watershed",
    "filter_labels_by_property",
    "filter_labels_by_volume",
    "analyze_skeleton",
    "measure_objects",
    "relabel_sequential",
    "remove_small_objects",
    "richardson_lucy_deconvolution",
    "richardson_lucy_tv_deconvolution",
    "rolling_ball_background",
    "prune_skeleton_branches",
    "subtract_background",
    "skeletonize",
    "skeleton_graph_overlay",
    "skeleton_keypoints",
}


DEFAULT_DYNAMIC_OUTPUT_PORTS = 3
DYNAMIC_OUTPUT_COUNT_PARAM = "_vipp_dynamic_output_count"


def _split_channels_outputs(count: int) -> tuple[OutputSpec, ...]:
    """Build the output ports for a Split Channels node.

    ``count`` is the number of channels discovered when the node last ran; the
    pipeline supplies the default port count for a node that has not yet
    processed an image.
    """
    count = max(int(count), 0)
    return tuple(
        OutputSpec(f"channel_{index + 1}", "image", f"Ch {index + 1}")
        for index in range(count)
    )


def _split_axis_outputs(count: int) -> tuple[OutputSpec, ...]:
    count = max(int(count), 0)
    return tuple(
        OutputSpec(f"slice_{index + 1}", "image", f"Slice {index + 1}")
        for index in range(count)
    )


def _born_wolf_psf_outputs(count: int) -> tuple[OutputSpec, ...]:
    count = max(int(count), 0)
    return tuple(
        OutputSpec(f"channel_{index + 1}_psf", "image", f"Channel {index + 1} PSF")
        for index in range(count)
    )


NODE_LIBRARY: tuple[OperationSpec, ...] = (
    OperationSpec(
        "input",
        "Image Source",
        IMAGE_DATA_CATEGORY,
        None,
        "image",
        SOURCE_PARAMETERS,
        subcategory=SOURCE_OUTPUT_GROUP,
    ),
    OperationSpec(
        "crop_stack",
        "Crop Stack",
        IMAGE_DATA_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec("top", "Top", "int", 0, 0, 256, 1),
            ParameterSpec("bottom", "Bottom", "int", 0, 0, 256, 1),
            ParameterSpec("left", "Left", "int", 0, 0, 256, 1),
            ParameterSpec("right", "Right", "int", 0, 0, 256, 1),
        ),
        crop_stack,
        subcategory=AXES_REGIONS_GROUP,
    ),
    OperationSpec(
        "linear_scale_offset",
        "Linear Scale + Offset",
        INTENSITY_CONTRAST_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec("alpha", "Scale", "float", 3.0, 0.0, 1000.0, 0.0001, 4),
            ParameterSpec(
                "beta",
                "Offset",
                "float",
                1.0,
                -100000.0,
                100000.0,
                1.0,
                2,
            ),
        ),
        linear_scale_offset,
    ),
    OperationSpec(
        "gamma_correction",
        "Gamma Correction",
        INTENSITY_CONTRAST_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec("gamma", "Gamma", "float", 0.5, 0.01, 5.0, 0.01, 3),
        ),
        gamma_correction,
    ),
    OperationSpec(
        "average_blur",
        "Average Blur",
        FILTERING_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec("size", "Kernel size", "int", 5, 1, 51, 1),
        ),
        average_blur,
        subcategory=SMOOTHING_DENOISING_GROUP,
        stack_processing_note=SLICE_WISE_STACK_NOTICE,
    ),
    OperationSpec(
        "gaussian_blur",
        "Gaussian Blur",
        FILTERING_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec("sigma", "Sigma", "float", 1.2, 0.0, 12.0, 0.1, 2),
        ),
        gaussian_blur,
        subcategory=SMOOTHING_DENOISING_GROUP,
        stack_processing_note=SLICE_WISE_STACK_NOTICE,
    ),
    OperationSpec(
        "gaussian_blur_3d",
        "Gaussian Blur 3D",
        FILTERING_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec("sigma_z", "Sigma Z", "float", 2.0, 0.0, 12.0, 0.1, 2),
            ParameterSpec("sigma_y", "Sigma Y", "float", 2.0, 0.0, 12.0, 0.1, 2),
            ParameterSpec("sigma_x", "Sigma X", "float", 2.0, 0.0, 12.0, 0.1, 2),
            ParameterSpec("lock_xy", "Lock X/Y sigma", "bool", True, 0, 1, 1),
        ),
        gaussian_blur_3d,
        subcategory=SMOOTHING_DENOISING_GROUP,
    ),
    OperationSpec(
        "born_wolf_psf",
        "Born-Wolf PSF",
        FILTERING_CATEGORY,
        "array",
        "image",
        (
            SPATIAL_MODE_PARAMETER,
            ParameterSpec(
                "auto_parameters",
                "Auto from metadata",
                "bool",
                True,
                0,
                1,
                1,
            ),
            ParameterSpec(
                "wavelength_nm",
                "Emission wavelength nm",
                "float",
                0.0,
                0.0,
                2000.0,
                1.0,
                1,
            ),
            ParameterSpec(
                "numerical_aperture",
                "Numerical aperture",
                "float",
                0.0,
                0.0,
                2.0,
                0.01,
                3,
            ),
            ParameterSpec(
                "refractive_index",
                "Refractive index",
                "float",
                0.0,
                0.0,
                2.0,
                0.001,
                4,
            ),
            ParameterSpec(
                "pixel_size_xy_um",
                "XY pixel size um",
                "float",
                0.0,
                0.0,
                10.0,
                0.001,
                4,
            ),
            ParameterSpec(
                "z_step_um",
                "Z step um",
                "float",
                0.0,
                0.0,
                20.0,
                0.001,
                4,
            ),
            ParameterSpec("xy_size", "PSF XY size", "int", 65, 9, 1025, 2),
            ParameterSpec("z_size", "PSF Z size", "int", 33, 1, 1025, 2),
            ParameterSpec("channel", "Channel", "int", -1, -1, 64, 1),
            ParameterSpec(
                "pupil_samples",
                "Pupil samples",
                "int",
                256,
                16,
                2048,
                16,
            ),
            ParameterSpec("normalize", "Normalize sum to 1", "bool", True, 0, 1, 1),
        ),
        born_wolf_psf_outputs,
        subcategory=RESTORATION_PSF_GROUP,
        output_factory=_born_wolf_psf_outputs,
    ),
    OperationSpec(
        "prepare_validate_psf",
        "Prepare / Validate PSF",
        FILTERING_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec(
                "center_mode",
                "Center mode",
                "choice",
                "Peak",
                0,
                0,
                1,
                choices=("None", "Peak", "Centroid"),
            ),
            ParameterSpec("clip_negatives", "Clip negatives", "bool", True, 0, 1, 1),
            ParameterSpec("normalize_sum", "Normalize sum", "bool", True, 0, 1, 1),
            ParameterSpec(
                "minimum_valid_sum",
                "Minimum valid sum",
                "float",
                1e-12,
                0.0,
                1.0,
                1e-12,
                12,
            ),
            ParameterSpec("force_odd_shape", "Force odd shape", "bool", True, 0, 1, 1),
            ParameterSpec(
                "crop_empty_border",
                "Crop empty border",
                "bool",
                False,
                0,
                1,
                1,
            ),
        ),
        prepare_validate_psf,
        subcategory=RESTORATION_PSF_GROUP,
    ),
    OperationSpec(
        "richardson_lucy_deconvolution",
        "Richardson-Lucy Deconvolution",
        FILTERING_CATEGORY,
        "image",
        "image",
        (
            SPATIAL_MODE_PARAMETER,
            ParameterSpec("iterations", "Iterations", "int", 25, 1, 500, 1),
            ParameterSpec("normalize_psf", "Normalize PSF", "bool", True, 0, 1, 1),
            ParameterSpec(
                "clip_negative_input",
                "Clip negative input",
                "bool",
                True,
                0,
                1,
                1,
            ),
            ParameterSpec(
                "clip_output_negative",
                "Clip output negative",
                "bool",
                True,
                0,
                1,
                1,
            ),
            ParameterSpec(
                "preserve_input_scale",
                "Preserve input scale",
                "bool",
                True,
                0,
                1,
                1,
            ),
            ParameterSpec(
                "filter_epsilon",
                "Filter epsilon",
                "float",
                1e-12,
                0.0,
                1.0,
                1e-12,
                12,
            ),
        ),
        richardson_lucy_deconvolution,
        max_inputs=2,
        subcategory=RESTORATION_PSF_GROUP,
        inputs=(
            InputSpec("image", "image", "Image"),
            InputSpec("psf", "image", "PSF"),
        ),
        execution_policy="manual",
    ),
    OperationSpec(
        "richardson_lucy_tv_deconvolution",
        "Richardson-Lucy TV Deconvolution",
        FILTERING_CATEGORY,
        "image",
        "image",
        (
            SPATIAL_MODE_PARAMETER,
            ParameterSpec("iterations", "Iterations", "int", 25, 1, 500, 1),
            ParameterSpec(
                "tv_regularization",
                "TV regularization",
                "float",
                0.002,
                0.0,
                1.0,
                0.0001,
                6,
            ),
            ParameterSpec(
                "tv_epsilon",
                "TV epsilon",
                "float",
                1e-6,
                1e-12,
                1.0,
                1e-6,
                8,
            ),
            ParameterSpec("normalize_psf", "Normalize PSF", "bool", True, 0, 1, 1),
            ParameterSpec(
                "clip_negative_input",
                "Clip negative input",
                "bool",
                True,
                0,
                1,
                1,
            ),
            ParameterSpec(
                "clip_output_negative",
                "Clip output negative",
                "bool",
                True,
                0,
                1,
                1,
            ),
            ParameterSpec(
                "preserve_input_scale",
                "Preserve input scale",
                "bool",
                True,
                0,
                1,
                1,
            ),
            ParameterSpec(
                "filter_epsilon",
                "Filter epsilon",
                "float",
                1e-12,
                0.0,
                1.0,
                1e-12,
                12,
            ),
            ParameterSpec(
                "denominator_floor",
                "Denominator floor",
                "float",
                0.05,
                1e-6,
                1.0,
                0.01,
                4,
            ),
        ),
        richardson_lucy_tv_deconvolution,
        max_inputs=2,
        subcategory=RESTORATION_PSF_GROUP,
        inputs=(
            InputSpec("image", "image", "Image"),
            InputSpec("psf", "image", "PSF"),
        ),
        execution_policy="manual",
    ),
    OperationSpec(
        "median_filter",
        "Median Filter",
        FILTERING_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec("size", "Kernel size", "int", 5, 1, 51, 2),
        ),
        median_filter,
        subcategory=SMOOTHING_DENOISING_GROUP,
        stack_processing_note=SLICE_WISE_STACK_NOTICE,
    ),
    OperationSpec(
        "bilateral_filter",
        "Bilateral Filtering",
        FILTERING_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec("diameter", "Diameter", "int", 5, 3, 31, 2),
            ParameterSpec(
                "sigma_color",
                "Sigma color",
                "float",
                5.0,
                0.1,
                50.0,
                0.1,
                2,
            ),
            ParameterSpec(
                "sigma_space",
                "Sigma space",
                "float",
                5.0,
                0.1,
                50.0,
                0.1,
                2,
            ),
        ),
        bilateral_filter,
        subcategory=SMOOTHING_DENOISING_GROUP,
        stack_processing_note=SLICE_WISE_STACK_NOTICE,
    ),
    OperationSpec(
        "non_local_means_filter",
        "Non-Local Means",
        FILTERING_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec("patch_size", "Patch size", "int", 5, 3, 15, 2),
            ParameterSpec("patch_distance", "Patch distance", "int", 6, 1, 20, 1),
            ParameterSpec("h", "Filter strength", "float", 0.08, 0.0, 1.0, 0.01, 3),
            ParameterSpec("fast_mode", "Fast mode", "bool", True, 0, 1, 1),
        ),
        non_local_means_filter,
        subcategory=SMOOTHING_DENOISING_GROUP,
        stack_processing_note=SLICE_WISE_STACK_NOTICE,
    ),
    OperationSpec(
        "rolling_ball_background",
        "Rolling-Ball Background",
        FILTERING_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec("radius", "Radius (px)", "float", 50.0, 1.0, 500.0, 1.0, 1),
            ParameterSpec(
                "light_background",
                "Light background",
                "bool",
                False,
                0,
                1,
                1,
            ),
            ParameterSpec(
                "disable_smoothing",
                "Disable smoothing",
                "bool",
                False,
                0,
                1,
                1,
            ),
            BACKGROUND_SPATIAL_MODE_PARAMETER,
        ),
        rolling_ball_background,
        subcategory=BACKGROUND_CORRECTION_GROUP,
        stack_processing_note=(
            "Stack notice: the default processes each YX slice independently. "
            "3D rolling-ball background estimation can be slow for large radii."
        ),
    ),
    OperationSpec(
        "subtract_background",
        "Subtract Background",
        FILTERING_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec("radius", "Radius (px)", "float", 50.0, 1.0, 500.0, 1.0, 1),
            ParameterSpec(
                "light_background",
                "Light background",
                "bool",
                False,
                0,
                1,
                1,
            ),
            ParameterSpec(
                "disable_smoothing",
                "Disable smoothing",
                "bool",
                False,
                0,
                1,
                1,
            ),
            ParameterSpec(
                "clip_negative",
                "Clip negative values",
                "bool",
                True,
                0,
                1,
                1,
            ),
            BACKGROUND_SPATIAL_MODE_PARAMETER,
        ),
        subtract_background,
        subcategory=BACKGROUND_CORRECTION_GROUP,
        stack_processing_note=(
            "Stack notice: the default processes each YX slice independently. "
            "3D rolling-ball background subtraction can be slow for large radii."
        ),
    ),
    OperationSpec(
        "difference_of_gaussians",
        "Difference of Gaussians",
        FILTERING_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec("low_sigma", "Low sigma", "float", 1.0, 0.0, 50.0, 0.1, 2),
            ParameterSpec("high_sigma", "High sigma", "float", 3.0, 0.0, 50.0, 0.1, 2),
        ),
        difference_of_gaussians_filter,
        subcategory=EDGE_DETAIL_GROUP,
        stack_processing_note=SLICE_WISE_STACK_NOTICE,
    ),
    OperationSpec(
        "unsharp_mask",
        "Unsharp Mask",
        FILTERING_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec("radius", "Radius", "float", 1.0, 0.0, 50.0, 0.1, 2),
            ParameterSpec("amount", "Amount", "float", 1.0, 0.0, 10.0, 0.1, 2),
        ),
        unsharp_mask_filter,
        subcategory=EDGE_DETAIL_GROUP,
        stack_processing_note=SLICE_WISE_STACK_NOTICE,
    ),
    OperationSpec(
        "sobel_filter",
        "Sobel Edges",
        FILTERING_CATEGORY,
        "array",
        "image",
        (),
        sobel_filter,
        subcategory=EDGE_DETAIL_GROUP,
        stack_processing_note=SLICE_WISE_STACK_NOTICE,
    ),
    OperationSpec(
        "canny_edges",
        "Canny Edges",
        FILTERING_CATEGORY,
        "array",
        "mask",
        (
            ParameterSpec("sigma", "Sigma", "float", 1.0, 0.0, 12.0, 0.1, 2),
            ParameterSpec(
                "low_quantile",
                "Low quantile",
                "float",
                0.1,
                0.0,
                1.0,
                0.01,
                3,
            ),
            ParameterSpec(
                "high_quantile",
                "High quantile",
                "float",
                0.2,
                0.0,
                1.0,
                0.01,
                3,
            ),
        ),
        canny_edges,
        subcategory=EDGE_DETAIL_GROUP,
        stack_processing_note=SLICE_WISE_STACK_NOTICE,
    ),
    OperationSpec(
        "laplace_filter",
        "Laplace Filter",
        FILTERING_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec("kernel_size", "Kernel size", "int", 3, 3, 15, 2),
        ),
        laplace_filter,
        subcategory=EDGE_DETAIL_GROUP,
        stack_processing_note=SLICE_WISE_STACK_NOTICE,
    ),
    OperationSpec(
        "mip",
        "Maximum Projection",
        "Projection",
        "array",
        "image",
        (
            ParameterSpec("axis", "Axis", "int", 0, 0, 5, 1),
        ),
        max_intensity_projection,
    ),
    OperationSpec(
        "project_image",
        "Project Image",
        "Projection",
        "array",
        "image",
        (
            ParameterSpec(
                "axes",
                "Project axis / axes",
                "choice",
                "auto",
                0,
                0,
                1,
                choices=(
                    "auto",
                    "non_yx_spatial",
                ),
                choice_labels=(
                    "Auto (Z if present)",
                    "All non-YX spatial axes",
                ),
                dynamic_choices=True,
            ),
            ParameterSpec(
                "method",
                "Projection method",
                "choice",
                "Maximum",
                0,
                0,
                1,
                choices=(
                    "Maximum",
                    "Mean",
                    "Minimum",
                    "Median",
                    "Sum",
                    "Standard deviation",
                ),
            ),
        ),
        project_image,
    ),
    OperationSpec(
        "orthogonal_projection",
        "Orthogonal Projection",
        "Projection",
        "array",
        "image",
        (
            ParameterSpec(
                "method",
                "Projection method",
                "choice",
                "Maximum",
                0,
                0,
                1,
                choices=(
                    "Maximum",
                    "Mean",
                    "Minimum",
                    "Median",
                    "Sum",
                    "Standard deviation",
                ),
            ),
            ParameterSpec(
                "use_physical_scale",
                "Use physical pixel size",
                "bool",
                True,
                0,
                1,
                1,
            ),
        ),
        orthogonal_projection,
    ),
    OperationSpec(
        "select_axis_slice",
        "Select Axis Slice",
        IMAGE_DATA_CATEGORY,
        "array",
        "any",
        (
            ParameterSpec("axis", "Axis", "int", 0, 0, 5, 1),
            ParameterSpec("index", "Index", "int", 0, 0, 1024, 1),
        ),
        select_axis_slice,
        subcategory=AXES_REGIONS_GROUP,
    ),
    OperationSpec(
        "split_axis",
        "Split Axis",
        IMAGE_DATA_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec(
                "axis",
                "Axis to split",
                "choice",
                "axis:0",
                0,
                0,
                1,
                choices=("axis:0",),
                choice_labels=("Axis 0",),
                dynamic_choices=True,
            ),
        ),
        split_axis,
        subcategory=AXES_REGIONS_GROUP,
        output_factory=_split_axis_outputs,
        preserves_input_type=True,
    ),
    OperationSpec(
        "reorder_axes",
        "Reorder Axes",
        IMAGE_DATA_CATEGORY,
        "array",
        "any",
        (
            ParameterSpec(
                "order",
                "Output axis order",
                "text",
                "",
                0,
                0,
                1,
            ),
        ),
        reorder_axes,
        subcategory=AXES_REGIONS_GROUP,
        preserves_input_type=True,
    ),
    OperationSpec(
        "set_pixel_size",
        "Set Pixel Size / Units",
        IMAGE_DATA_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec(
                "x_size",
                "X pixel size",
                "float",
                1.0,
                0.0001,
                10000.0,
                0.01,
                4,
            ),
            ParameterSpec(
                "y_size",
                "Y pixel size",
                "float",
                1.0,
                0.0001,
                10000.0,
                0.01,
                4,
            ),
            ParameterSpec(
                "z_size",
                "Z step size",
                "float",
                1.0,
                0.0001,
                10000.0,
                0.01,
                4,
            ),
            ParameterSpec(
                "unit",
                "Unit",
                "choice",
                "micrometer",
                0,
                0,
                1,
                choices=(
                    "micrometer",
                    "nanometer",
                    "millimeter",
                    "meter",
                    "pixel",
                ),
            ),
        ),
        set_pixel_size,
        subcategory=AXES_REGIONS_GROUP,
        preserves_input_type=True,
    ),
    OperationSpec(
        "rescale_axes",
        "Rescale Axes",
        IMAGE_DATA_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec(
                "x_scale",
                "X scale factor",
                "float",
                1.0,
                0.01,
                20.0,
                0.01,
                3,
            ),
            ParameterSpec(
                "y_scale",
                "Y scale factor",
                "float",
                1.0,
                0.01,
                20.0,
                0.01,
                3,
            ),
            ParameterSpec(
                "z_scale",
                "Z scale factor",
                "float",
                1.0,
                0.01,
                20.0,
                0.01,
                3,
            ),
            ParameterSpec(
                "lock_xy",
                "Lock X/Y aspect ratio",
                "bool",
                True,
                0,
                1,
                1,
            ),
            ParameterSpec(
                "interpolation",
                "Interpolation",
                "choice",
                "Auto",
                0,
                0,
                1,
                choices=(
                    "Auto",
                    "Nearest neighbor",
                    "Linear (bilinear/trilinear)",
                    "Cubic (bicubic/tricubic)",
                    "Quadratic spline",
                    "Quartic spline",
                    "Quintic spline",
                ),
            ),
            ParameterSpec(
                "anti_aliasing",
                "Anti-alias downsampling",
                "bool",
                True,
                0,
                1,
                1,
            ),
        ),
        rescale_axes,
        subcategory=AXES_REGIONS_GROUP,
        preserves_input_type=True,
    ),
    OperationSpec(
        "otsu_threshold",
        "Otsu Threshold",
        SEGMENTATION_CATEGORY,
        "array",
        "mask",
        (THRESHOLD_SCOPE_PARAMETER, HISTOGRAM_BINS_PARAMETER),
        otsu_threshold,
        subcategory=GLOBAL_THRESHOLDS_GROUP,
    ),
    OperationSpec(
        "triangle_threshold",
        "Triangle Threshold",
        SEGMENTATION_CATEGORY,
        "array",
        "mask",
        (THRESHOLD_SCOPE_PARAMETER, HISTOGRAM_BINS_PARAMETER),
        triangle_threshold,
        subcategory=GLOBAL_THRESHOLDS_GROUP,
    ),
    OperationSpec(
        "li_threshold",
        "Li Threshold",
        SEGMENTATION_CATEGORY,
        "array",
        "mask",
        (THRESHOLD_SCOPE_PARAMETER,),
        li_threshold,
        subcategory=GLOBAL_THRESHOLDS_GROUP,
    ),
    OperationSpec(
        "yen_threshold",
        "Yen Threshold",
        SEGMENTATION_CATEGORY,
        "array",
        "mask",
        (THRESHOLD_SCOPE_PARAMETER, HISTOGRAM_BINS_PARAMETER),
        yen_threshold,
        subcategory=GLOBAL_THRESHOLDS_GROUP,
    ),
    OperationSpec(
        "isodata_threshold",
        "Isodata Threshold",
        SEGMENTATION_CATEGORY,
        "array",
        "mask",
        (THRESHOLD_SCOPE_PARAMETER, HISTOGRAM_BINS_PARAMETER),
        isodata_threshold,
        subcategory=GLOBAL_THRESHOLDS_GROUP,
    ),
    OperationSpec(
        "minimum_threshold",
        "Minimum Threshold",
        SEGMENTATION_CATEGORY,
        "array",
        "mask",
        (
            THRESHOLD_SCOPE_PARAMETER,
            HISTOGRAM_BINS_PARAMETER,
            ParameterSpec(
                "max_iterations",
                "Maximum smoothing iterations",
                "int",
                10_000,
                1,
                10_000,
                25,
            ),
        ),
        minimum_threshold,
        subcategory=GLOBAL_THRESHOLDS_GROUP,
    ),
    OperationSpec(
        "binary_threshold",
        "Binary Threshold",
        SEGMENTATION_CATEGORY,
        "array",
        "mask",
        (
            ParameterSpec("threshold", "Threshold", "float", 0.5, 0.0, 1.0, 0.01, 3),
        ),
        binary_threshold,
        subcategory=GLOBAL_THRESHOLDS_GROUP,
    ),
    OperationSpec(
        "hysteresis_threshold",
        "Hysteresis Threshold",
        SEGMENTATION_CATEGORY,
        "array",
        "mask",
        (
            ParameterSpec(
                "low_threshold",
                "Low threshold",
                "float",
                0.25,
                0.0,
                1.0,
                0.01,
                3,
            ),
            ParameterSpec(
                "high_threshold",
                "High threshold",
                "float",
                0.7,
                0.0,
                1.0,
                0.01,
                3,
            ),
            SPATIAL_MODE_PARAMETER,
        ),
        hysteresis_threshold,
        subcategory=GLOBAL_THRESHOLDS_GROUP,
    ),
    OperationSpec(
        "adaptive_mean_threshold",
        "Adaptive Mean Threshold",
        SEGMENTATION_CATEGORY,
        "array",
        "mask",
        (
            ParameterSpec("block_size", "Block size", "int", 11, 3, 101, 2),
            ParameterSpec("c", "C", "float", 2.0, -50.0, 50.0, 0.1, 2),
        ),
        adaptive_mean_threshold,
        subcategory=LOCAL_THRESHOLDS_GROUP,
        stack_processing_note=SLICE_WISE_STACK_NOTICE,
    ),
    OperationSpec(
        "adaptive_gaussian_threshold",
        "Adaptive Gaussian Threshold",
        SEGMENTATION_CATEGORY,
        "array",
        "mask",
        (
            ParameterSpec("block_size", "Block size", "int", 11, 3, 101, 2),
            ParameterSpec("c", "C", "float", 2.0, -50.0, 50.0, 0.1, 2),
        ),
        adaptive_gaussian_threshold,
        subcategory=LOCAL_THRESHOLDS_GROUP,
        stack_processing_note=SLICE_WISE_STACK_NOTICE,
    ),
    OperationSpec(
        "sauvola_threshold",
        "Sauvola Threshold",
        SEGMENTATION_CATEGORY,
        "array",
        "mask",
        (
            ParameterSpec("window_size", "Window size", "int", 15, 3, 151, 2),
            ParameterSpec("k", "k", "float", 0.2, -2.0, 2.0, 0.01, 3),
            ParameterSpec(
                "dynamic_range",
                "Dynamic range (0 auto)",
                "float",
                0.0,
                0.0,
                1_000_000.0,
                0.1,
                3,
            ),
        ),
        sauvola_threshold,
        subcategory=LOCAL_THRESHOLDS_GROUP,
        stack_processing_note=SLICE_WISE_STACK_NOTICE,
    ),
    OperationSpec(
        "niblack_threshold",
        "Niblack Threshold",
        SEGMENTATION_CATEGORY,
        "array",
        "mask",
        (
            ParameterSpec("window_size", "Window size", "int", 15, 3, 151, 2),
            ParameterSpec("k", "k", "float", 0.2, -2.0, 2.0, 0.01, 3),
        ),
        niblack_threshold,
        subcategory=LOCAL_THRESHOLDS_GROUP,
        stack_processing_note=SLICE_WISE_STACK_NOTICE,
    ),
    OperationSpec(
        "auto_watershed_from_mask",
        "Auto Watershed From Mask",
        SEGMENTATION_CATEGORY,
        "array",
        "labels",
        (
            ParameterSpec(
                "h",
                "H / prominence in px/voxels (0 = local maxima)",
                "float",
                1.0,
                0.0,
                5.0,
                0.1,
                3,
            ),
            ParameterSpec(
                "connectivity",
                "Marker connectivity",
                "choice",
                "Full connectivity",
                0,
                0,
                1,
                choices=("Face connected", "Full connectivity"),
            ),
            ParameterSpec(
                "image_mode",
                "Image meaning",
                "choice",
                "Distance map (invert)",
                0,
                0,
                1,
                choices=("Distance map (invert)", "Elevation image"),
            ),
            ParameterSpec(
                "compactness",
                "Compactness",
                "float",
                0.0,
                0.0,
                100.0,
                0.1,
                3,
            ),
            ParameterSpec(
                "watershed_line",
                "Watershed line",
                "bool",
                False,
                0,
                1,
                1,
            ),
            SPATIAL_MODE_PARAMETER,
        ),
        auto_watershed_from_mask,
        subcategory=OBJECT_SEPARATION_GROUP,
    ),
    OperationSpec(
        "euclidean_distance_transform",
        "Euclidean Distance Transform",
        SEGMENTATION_CATEGORY,
        "array",
        "image",
        (SPATIAL_MODE_PARAMETER,),
        euclidean_distance_transform,
        subcategory=OBJECT_SEPARATION_GROUP,
    ),
    OperationSpec(
        "h_maxima_markers",
        "H-Maxima Markers",
        SEGMENTATION_CATEGORY,
        "array",
        "labels",
        (
            ParameterSpec(
                "h",
                "H / prominence in px/voxels (0 = local maxima)",
                "float",
                1.0,
                0.0,
                5.0,
                0.1,
                3,
            ),
            SPATIAL_MODE_PARAMETER,
            ParameterSpec(
                "connectivity",
                "Marker connectivity",
                "choice",
                "Full connectivity",
                0,
                0,
                1,
                choices=("Face connected", "Full connectivity"),
            ),
        ),
        h_maxima_markers,
        subcategory=OBJECT_SEPARATION_GROUP,
    ),
    OperationSpec(
        "marker_controlled_watershed",
        "Marker-Controlled Watershed",
        SEGMENTATION_CATEGORY,
        "array",
        "labels",
        (
            ParameterSpec(
                "image_mode",
                "Image meaning",
                "choice",
                "Distance map (invert)",
                0,
                0,
                1,
                choices=("Distance map (invert)", "Elevation image"),
            ),
            ParameterSpec(
                "compactness",
                "Compactness",
                "float",
                0.0,
                0.0,
                100.0,
                0.1,
                3,
            ),
            ParameterSpec(
                "watershed_line",
                "Watershed line",
                "bool",
                False,
                0,
                1,
                1,
            ),
            SPATIAL_MODE_PARAMETER,
        ),
        marker_controlled_watershed,
        max_inputs=3,
        inputs=(
            InputSpec("image", "array", "Image / distance"),
            InputSpec("markers", "labels", "Markers"),
            InputSpec("mask", "array", "Mask"),
        ),
        subcategory=OBJECT_SEPARATION_GROUP,
    ),
    OperationSpec(
        "expand_labels",
        "Expand Labels",
        SEGMENTATION_CATEGORY,
        "labels",
        "labels",
        (
            ParameterSpec(
                "distance",
                "Distance (pixels/voxels)",
                "float",
                5.0,
                0.0,
                10_000.0,
                0.1,
                2,
            ),
            SPATIAL_MODE_PARAMETER,
        ),
        expand_labels_image,
        subcategory=OBJECT_SEPARATION_GROUP,
    ),
    OperationSpec(
        "dilate",
        "Dilation",
        "Morphology",
        "array",
        "mask",
        (
            ParameterSpec("size", "Kernel size", "int", 10, 1, 101, 1),
            ParameterSpec("iterations", "Iterations", "int", 1, 1, 25, 1),
        ),
        dilate,
        stack_processing_note=SLICE_WISE_STACK_NOTICE,
    ),
    OperationSpec(
        "erode",
        "Erosion",
        "Morphology",
        "array",
        "mask",
        (
            ParameterSpec("size", "Kernel size", "int", 10, 1, 101, 1),
            ParameterSpec("iterations", "Iterations", "int", 1, 1, 25, 1),
        ),
        erode,
        stack_processing_note=SLICE_WISE_STACK_NOTICE,
    ),
    OperationSpec(
        "opening",
        "Opening",
        "Morphology",
        "array",
        "mask",
        (ParameterSpec("size", "Kernel size", "int", 2, 1, 101, 1),),
        opening,
        stack_processing_note=SLICE_WISE_STACK_NOTICE,
    ),
    OperationSpec(
        "closing",
        "Closing",
        "Morphology",
        "array",
        "mask",
        (ParameterSpec("size", "Kernel size", "int", 2, 1, 101, 1),),
        closing,
        stack_processing_note=SLICE_WISE_STACK_NOTICE,
    ),
    OperationSpec(
        "top_hat",
        "Top Hat",
        "Morphology",
        "array",
        "mask",
        (ParameterSpec("size", "Kernel size", "int", 2, 1, 101, 1),),
        top_hat,
        stack_processing_note=SLICE_WISE_STACK_NOTICE,
    ),
    OperationSpec(
        "black_hat",
        "Black Hat",
        "Morphology",
        "array",
        "mask",
        (ParameterSpec("size", "Kernel size", "int", 2, 1, 101, 1),),
        black_hat,
        stack_processing_note=SLICE_WISE_STACK_NOTICE,
    ),
    OperationSpec(
        "morphological_gradient",
        "Morphological Gradient",
        "Morphology",
        "array",
        "mask",
        (ParameterSpec("size", "Kernel size", "int", 2, 1, 101, 1),),
        morphological_gradient,
        stack_processing_note=SLICE_WISE_STACK_NOTICE,
    ),
    OperationSpec(
        "fill_holes",
        "Fill Holes",
        "Morphology",
        "mask",
        "mask",
        (
            ParameterSpec(
                "max_hole_size",
                "Maximum hole size (0 = fill all)",
                "int",
                0,
                0,
                1_000_000_000,
                1,
            ),
            SPATIAL_MODE_PARAMETER,
            ParameterSpec(
                "connectivity",
                "Hole connectivity",
                "choice",
                "Face connected",
                0,
                0,
                1,
                choices=("Face connected", "Full connectivity"),
            ),
        ),
        fill_holes,
    ),
    OperationSpec(
        "remove_small_objects",
        "Remove Small Objects",
        "Morphology",
        "mask_or_labels",
        "mask",
        (
            ParameterSpec(
                "min_size",
                "Minimum object size (pixels/voxels)",
                "int",
                10,
                0,
                1_000_000_000,
                1,
            ),
            SPATIAL_MODE_PARAMETER,
            ParameterSpec(
                "connectivity",
                "Mask connectivity",
                "choice",
                "Face connected",
                0,
                0,
                1,
                choices=("Face connected", "Full connectivity"),
            ),
        ),
        remove_small_objects,
        preserves_input_type=True,
    ),
    OperationSpec(
        "skeletonize",
        "Skeletonize",
        "Morphology",
        "mask",
        "mask",
        (
            SPATIAL_MODE_PARAMETER,
            ParameterSpec(
                "method",
                "Method",
                "choice",
                "Auto",
                0,
                0,
                1,
                choices=("Auto", "Lee", "Zhang 2D"),
            ),
        ),
        skeletonize_mask,
        subcategory=SKELETON_NETWORK_GROUP,
    ),
    OperationSpec(
        "label_connected_components",
        "Label Connected Components",
        LABEL_OPERATIONS_CATEGORY,
        "mask",
        "labels",
        (
            SPATIAL_MODE_PARAMETER,
            ParameterSpec(
                "connectivity",
                "Connectivity",
                "choice",
                "Full connectivity",
                0,
                0,
                1,
                choices=("Face connected", "Full connectivity"),
            ),
        ),
        label_connected_components,
    ),
    OperationSpec(
        "clear_border_objects",
        "Clear Border Objects",
        LABEL_OPERATIONS_CATEGORY,
        "mask_or_labels",
        "mask",
        (
            ParameterSpec(
                "border_buffer",
                "Border buffer (pixels/voxels)",
                "int",
                0,
                0,
                10_000,
                1,
            ),
            ParameterSpec(
                "boundary_mode",
                "Boundaries",
                "choice",
                "All spatial borders",
                0,
                0,
                1,
                choices=(
                    "All spatial borders",
                    "Lateral borders only (YX)",
                ),
            ),
        ),
        clear_border_objects,
        preserves_input_type=True,
    ),
    OperationSpec(
        "filter_labels_by_volume",
        "Filter Labels By Volume",
        LABEL_OPERATIONS_CATEGORY,
        "labels",
        "labels",
        (
            ParameterSpec(
                "min_volume",
                "Minimum volume (pixels/voxels)",
                "int",
                10,
                0,
                1_000_000_000,
                1,
            ),
            ParameterSpec(
                "max_volume",
                "Maximum volume (0 = none)",
                "int",
                0,
                0,
                1_000_000_000,
                1,
            ),
            SPATIAL_MODE_PARAMETER,
        ),
        filter_labels_by_volume,
    ),
    OperationSpec(
        "filter_labels_by_property",
        "Filter Labels By Property",
        LABEL_OPERATIONS_CATEGORY,
        "labels",
        "labels",
        (
            ParameterSpec(
                "property_column",
                "Property column (auto or name)",
                "text",
                "auto",
                0,
                0,
                1,
            ),
            ParameterSpec(
                "min_value",
                "Minimum value",
                "float",
                0.0,
                -1_000_000_000.0,
                1_000_000_000.0,
                1.0,
                3,
            ),
            ParameterSpec(
                "max_value",
                "Maximum value (0 = none)",
                "float",
                0.0,
                -1_000_000_000.0,
                1_000_000_000.0,
                1.0,
                3,
            ),
            ParameterSpec(
                "keep_mode",
                "Action",
                "choice",
                "Keep inside range",
                0,
                0,
                1,
                choices=("Keep inside range", "Remove inside range"),
            ),
            ParameterSpec(
                "unmatched_labels",
                "Labels without table row",
                "choice",
                "Remove unmatched labels",
                0,
                0,
                1,
                choices=("Remove unmatched labels", "Keep unmatched labels"),
            ),
            SPATIAL_MODE_PARAMETER,
        ),
        filter_labels_by_property,
        max_inputs=2,
        inputs=(
            InputSpec("labels", "labels", "Labels"),
            InputSpec("table", "table", "Measurements table"),
        ),
    ),
    OperationSpec(
        "relabel_sequential",
        "Relabel Sequential",
        LABEL_OPERATIONS_CATEGORY,
        "labels",
        "labels",
        (SPATIAL_MODE_PARAMETER,),
        relabel_sequential,
    ),
    OperationSpec(
        "measure_objects",
        "Measure Objects",
        MEASUREMENTS_CATEGORY,
        "labels",
        "table",
        (
            SPATIAL_MODE_PARAMETER,
            ParameterSpec(
                "include_shape_descriptors",
                "Shape descriptors",
                "bool",
                False,
                0,
                1,
                1,
            ),
            ParameterSpec(
                "include_axis_descriptors",
                "Axis/inertia descriptors",
                "bool",
                False,
                0,
                1,
                1,
            ),
            ParameterSpec(
                "include_2d_boundary_descriptors",
                "2D boundary descriptors",
                "bool",
                False,
                0,
                1,
                1,
            ),
            ParameterSpec(
                "include_derived_shape_ratios",
                "Derived shape ratios",
                "bool",
                False,
                0,
                1,
                1,
            ),
            ParameterSpec(
                "include_2d_shape_moments",
                "2D shape moments",
                "bool",
                False,
                0,
                1,
                1,
            ),
        ),
        measure_objects,
        execution_policy="manual",
    ),
    OperationSpec(
        "measure_objects_intensity",
        "Measure Objects + Intensity",
        MEASUREMENTS_CATEGORY,
        "labels",
        "table",
        (
            SPATIAL_MODE_PARAMETER,
            ParameterSpec(
                "include_shape_descriptors",
                "Shape descriptors",
                "bool",
                False,
                0,
                1,
                1,
            ),
            ParameterSpec(
                "include_axis_descriptors",
                "Axis/inertia descriptors",
                "bool",
                False,
                0,
                1,
                1,
            ),
            ParameterSpec(
                "include_2d_boundary_descriptors",
                "2D boundary descriptors",
                "bool",
                False,
                0,
                1,
                1,
            ),
            ParameterSpec(
                "include_derived_shape_ratios",
                "Derived shape ratios",
                "bool",
                False,
                0,
                1,
                1,
            ),
            ParameterSpec(
                "include_2d_shape_moments",
                "2D shape moments",
                "bool",
                False,
                0,
                1,
                1,
            ),
        ),
        measure_objects_with_intensity,
        max_inputs=2,
        inputs=(
            InputSpec("labels", "labels", "Labels"),
            InputSpec("intensity", "image", "Intensity image"),
        ),
        execution_policy="manual",
    ),
    OperationSpec(
        "measure_3d_mesh_morphology",
        "Measure 3D Mesh Morphology",
        MEASUREMENTS_CATEGORY,
        "labels",
        "table",
        (
            SPATIAL_MODE_PARAMETER,
            ParameterSpec(
                "minimum_voxel_count",
                "Minimum voxel count",
                "int",
                16,
                1,
                100000,
                1,
            ),
            ParameterSpec(
                "include_convex_hull_metrics",
                "Convex hull metrics",
                "bool",
                True,
                0,
                1,
                1,
            ),
        ),
        measure_3d_mesh_morphology,
        execution_policy="manual",
    ),
    OperationSpec(
        "analyze_skeleton",
        "Analyze Skeleton",
        MEASUREMENTS_CATEGORY,
        "mask",
        "table",
        (
            SPATIAL_MODE_PARAMETER,
            ParameterSpec(
                "input_mode",
                "Input",
                "choice",
                "Already skeletonized",
                0,
                0,
                1,
                choices=("Already skeletonized", "Skeletonize first"),
            ),
        ),
        analyze_skeleton,
        subcategory=SKELETON_NETWORK_GROUP,
        execution_policy="manual",
    ),
    OperationSpec(
        "measure_skeleton_branches",
        "Measure Skeleton Branches",
        MEASUREMENTS_CATEGORY,
        "mask",
        "table",
        (
            SPATIAL_MODE_PARAMETER,
            ParameterSpec(
                "input_mode",
                "Input",
                "choice",
                "Already skeletonized",
                0,
                0,
                1,
                choices=("Already skeletonized", "Skeletonize first"),
            ),
        ),
        measure_skeleton_branches,
        subcategory=SKELETON_NETWORK_GROUP,
        execution_policy="manual",
    ),
    OperationSpec(
        "summarize_skeleton_branches",
        "Summarize Skeleton Branches",
        MEASUREMENTS_CATEGORY,
        "table",
        "table",
        (
            ParameterSpec(
                "group_by",
                "Group by columns (auto or comma-separated)",
                "text",
                "auto",
                0,
                0,
                1,
            ),
            ParameterSpec(
                "statistics",
                "Length/tortuosity statistics",
                "text",
                "mean,median,std,min,max,q25,q75",
                0,
                0,
                1,
            ),
        ),
        summarize_skeleton_branches,
        subcategory=SKELETON_NETWORK_GROUP,
    ),
    OperationSpec(
        "skeleton_graph_tables",
        "Skeleton Graph Tables",
        MEASUREMENTS_CATEGORY,
        "mask",
        "table",
        (
            SPATIAL_MODE_PARAMETER,
            ParameterSpec(
                "input_mode",
                "Input",
                "choice",
                "Already skeletonized",
                0,
                0,
                1,
                choices=("Already skeletonized", "Skeletonize first"),
            ),
        ),
        skeleton_graph_tables,
        subcategory=SKELETON_NETWORK_GROUP,
        outputs=(
            OutputSpec("nodes", "table", "Graph nodes"),
            OutputSpec("edges", "table", "Graph edges"),
        ),
        execution_policy="manual",
    ),
    OperationSpec(
        "measure_overall_skeleton_network",
        "Measure Overall Skeleton Network",
        MEASUREMENTS_CATEGORY,
        "mask",
        "table",
        (
            SPATIAL_MODE_PARAMETER,
            ParameterSpec(
                "input_mode",
                "Input",
                "choice",
                "Already skeletonized",
                0,
                0,
                1,
                choices=("Already skeletonized", "Skeletonize first"),
            ),
        ),
        measure_overall_skeleton_network,
        subcategory=SKELETON_NETWORK_GROUP,
        execution_policy="manual",
    ),
    OperationSpec(
        "skeleton_keypoints",
        "Skeleton Keypoints",
        "Morphology",
        "mask",
        "mask",
        (SPATIAL_MODE_PARAMETER,),
        skeleton_keypoints,
        subcategory=SKELETON_NETWORK_GROUP,
        outputs=(
            OutputSpec("endpoints", "mask", "Endpoints"),
            OutputSpec("junctions", "mask", "Junctions"),
            OutputSpec("isolated", "mask", "Isolated nodes"),
        ),
    ),
    OperationSpec(
        "skeleton_graph_overlay",
        "Skeleton Graph Overlay",
        "Morphology",
        "mask",
        "image",
        (
            SPATIAL_MODE_PARAMETER,
            ParameterSpec(
                "display_mode",
                "Display mode",
                "choice",
                "Colored edges + colored nodes",
                0,
                0,
                1,
                choices=(
                    "Colored edges + colored nodes",
                    "Colored edges",
                    "White edges + colored nodes",
                ),
            ),
            ParameterSpec("node_size", "Node size", "int", 1, 1, 5, 1),
        ),
        skeleton_graph_overlay,
        subcategory=SKELETON_NETWORK_GROUP,
    ),
    OperationSpec(
        "label_skeleton_components",
        "Label Skeleton Components",
        LABEL_OPERATIONS_CATEGORY,
        "mask",
        "labels",
        (SPATIAL_MODE_PARAMETER,),
        label_skeleton_components,
        subcategory=SKELETON_NETWORK_GROUP,
    ),
    OperationSpec(
        "label_skeleton_branches",
        "Label Skeleton Branches",
        LABEL_OPERATIONS_CATEGORY,
        "mask",
        "labels",
        (SPATIAL_MODE_PARAMETER,),
        label_skeleton_branches,
        subcategory=SKELETON_NETWORK_GROUP,
    ),
    OperationSpec(
        "prune_skeleton_branches",
        "Prune Skeleton Branches",
        "Morphology",
        "mask",
        "mask",
        (
            ParameterSpec(
                "min_branch_length",
                "Minimum terminal branch length",
                "float",
                3.0,
                0.0,
                100_000.0,
                1.0,
                decimals=1,
            ),
            ParameterSpec(
                "length_units",
                "Length units",
                "choice",
                "Pixels/voxels",
                0,
                0,
                1,
                choices=("Pixels/voxels", "Physical units"),
            ),
            ParameterSpec(
                "iterations",
                "Pruning passes",
                "int",
                1,
                1,
                100,
                1,
            ),
            ParameterSpec(
                "remove_isolated",
                "Remove isolated skeleton voxels",
                "bool",
                True,
                0,
                1,
                1,
            ),
            SPATIAL_MODE_PARAMETER,
        ),
        prune_skeleton_branches,
        subcategory=SKELETON_NETWORK_GROUP,
    ),
    OperationSpec(
        "colocalization_metrics",
        "Colocalization Metrics",
        COLOCALIZATION_CATEGORY,
        "array",
        "table",
        (
            ParameterSpec(
                "threshold_mode",
                "Thresholds",
                "choice",
                "Manual",
                0,
                0,
                1,
                choices=("Manual", "Costes auto"),
            ),
            ParameterSpec(
                "channel_1_threshold",
                "Channel 1 threshold (0-255)",
                "float",
                25.0,
                0.0,
                255.0,
                1.0,
                2,
            ),
            ParameterSpec(
                "channel_2_threshold",
                "Channel 2 threshold (0-255)",
                "float",
                25.0,
                0.0,
                255.0,
                1.0,
                2,
            ),
        ),
        colocalization_metrics,
        max_inputs=2,
        inputs=(
            InputSpec("channel_1", "array", "Channel 1 image"),
            InputSpec("channel_2", "array", "Channel 2 image"),
        ),
        execution_policy="manual",
    ),
    OperationSpec(
        "masked_colocalization_metrics",
        "Masked Colocalization Metrics",
        COLOCALIZATION_CATEGORY,
        "array",
        "table",
        (
            ParameterSpec(
                "threshold_mode",
                "Thresholds",
                "choice",
                "Manual",
                0,
                0,
                1,
                choices=("Manual", "Costes auto"),
            ),
            ParameterSpec(
                "channel_1_threshold",
                "Channel 1 threshold (0-255)",
                "float",
                25.0,
                0.0,
                255.0,
                1.0,
                2,
            ),
            ParameterSpec(
                "channel_2_threshold",
                "Channel 2 threshold (0-255)",
                "float",
                25.0,
                0.0,
                255.0,
                1.0,
                2,
            ),
        ),
        colocalization_metrics,
        max_inputs=3,
        inputs=(
            InputSpec("channel_1", "array", "Channel 1 image"),
            InputSpec("channel_2", "array", "Channel 2 image"),
            InputSpec("roi_mask", "mask_or_labels", "ROI mask"),
        ),
        execution_policy="manual",
    ),
    OperationSpec(
        "colocalized_voxels",
        "Colocalized Voxels",
        COLOCALIZATION_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec(
                "threshold_mode",
                "Thresholds",
                "choice",
                "Manual",
                0,
                0,
                1,
                choices=("Manual", "Costes auto"),
            ),
            ParameterSpec(
                "channel_1_threshold",
                "Channel 1 threshold (0-255)",
                "float",
                25.0,
                0.0,
                255.0,
                1.0,
                2,
            ),
            ParameterSpec(
                "channel_2_threshold",
                "Channel 2 threshold (0-255)",
                "float",
                25.0,
                0.0,
                255.0,
                1.0,
                2,
            ),
            ParameterSpec(
                "display_mode",
                "Display mode",
                "choice",
                "White overlay on channels",
                0,
                0,
                1,
                choices=(
                    "White overlay on channels",
                    "White on black",
                    "Channel colors only",
                ),
            ),
            ParameterSpec(
                "channel_1_color",
                "Channel 1 colour",
                "choice",
                "Red",
                0,
                0,
                1,
                choices=("Red", "Green", "Blue", "Magenta", "Cyan", "Yellow"),
            ),
            ParameterSpec(
                "channel_2_color",
                "Channel 2 colour",
                "choice",
                "Green",
                0,
                0,
                1,
                choices=("Red", "Green", "Blue", "Magenta", "Cyan", "Yellow"),
            ),
        ),
        colocalized_voxels,
        max_inputs=2,
        inputs=(
            InputSpec("channel_1", "array", "Channel 1 image"),
            InputSpec("channel_2", "array", "Channel 2 image"),
        ),
    ),
    OperationSpec(
        "masked_colocalized_voxels",
        "Masked Colocalized Voxels",
        COLOCALIZATION_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec(
                "threshold_mode",
                "Thresholds",
                "choice",
                "Manual",
                0,
                0,
                1,
                choices=("Manual", "Costes auto"),
            ),
            ParameterSpec(
                "channel_1_threshold",
                "Channel 1 threshold (0-255)",
                "float",
                25.0,
                0.0,
                255.0,
                1.0,
                2,
            ),
            ParameterSpec(
                "channel_2_threshold",
                "Channel 2 threshold (0-255)",
                "float",
                25.0,
                0.0,
                255.0,
                1.0,
                2,
            ),
            ParameterSpec(
                "display_mode",
                "Display mode",
                "choice",
                "White overlay on channels",
                0,
                0,
                1,
                choices=(
                    "White overlay on channels",
                    "White on black",
                    "Channel colors only",
                ),
            ),
            ParameterSpec(
                "channel_1_color",
                "Channel 1 colour",
                "choice",
                "Red",
                0,
                0,
                1,
                choices=("Red", "Green", "Blue", "Magenta", "Cyan", "Yellow"),
            ),
            ParameterSpec(
                "channel_2_color",
                "Channel 2 colour",
                "choice",
                "Green",
                0,
                0,
                1,
                choices=("Red", "Green", "Blue", "Magenta", "Cyan", "Yellow"),
            ),
        ),
        colocalized_voxels,
        max_inputs=3,
        inputs=(
            InputSpec("channel_1", "array", "Channel 1 image"),
            InputSpec("channel_2", "array", "Channel 2 image"),
            InputSpec("roi_mask", "mask_or_labels", "ROI mask"),
        ),
    ),
    OperationSpec(
        "racc_index",
        "RACC Index",
        COLOCALIZATION_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec(
                "threshold_mode",
                "Thresholds",
                "choice",
                "Manual",
                0,
                0,
                1,
                choices=("Manual", "Costes auto"),
            ),
            ParameterSpec(
                "channel_1_threshold",
                "Channel 1 threshold (0-255)",
                "float",
                25.0,
                0.0,
                255.0,
                1.0,
                2,
            ),
            ParameterSpec(
                "channel_2_threshold",
                "Channel 2 threshold (0-255)",
                "float",
                25.0,
                0.0,
                255.0,
                1.0,
                2,
            ),
            ParameterSpec(
                "theta_degrees",
                "Theta (degrees)",
                "float",
                45.0,
                0.0,
                89.0,
                1.0,
                2,
            ),
            ParameterSpec(
                "include_percentile",
                "Included percentile",
                "float",
                99.0,
                1.0,
                100.0,
                0.5,
                2,
            ),
            ParameterSpec(
                "output_dtype",
                "Output dtype",
                "choice",
                "float32",
                0,
                0,
                1,
                choices=("float32", "uint8"),
            ),
        ),
        racc_index,
        max_inputs=2,
        inputs=(
            InputSpec("channel_1", "array", "Channel 1 image"),
            InputSpec("channel_2", "array", "Channel 2 image"),
        ),
        execution_policy="manual",
    ),
    OperationSpec(
        "masked_racc_index",
        "Masked RACC Index",
        COLOCALIZATION_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec(
                "threshold_mode",
                "Thresholds",
                "choice",
                "Manual",
                0,
                0,
                1,
                choices=("Manual", "Costes auto"),
            ),
            ParameterSpec(
                "channel_1_threshold",
                "Channel 1 threshold (0-255)",
                "float",
                25.0,
                0.0,
                255.0,
                1.0,
                2,
            ),
            ParameterSpec(
                "channel_2_threshold",
                "Channel 2 threshold (0-255)",
                "float",
                25.0,
                0.0,
                255.0,
                1.0,
                2,
            ),
            ParameterSpec(
                "theta_degrees",
                "Theta (degrees)",
                "float",
                45.0,
                0.0,
                89.0,
                1.0,
                2,
            ),
            ParameterSpec(
                "include_percentile",
                "Included percentile",
                "float",
                99.0,
                1.0,
                100.0,
                0.5,
                2,
            ),
            ParameterSpec(
                "output_dtype",
                "Output dtype",
                "choice",
                "float32",
                0,
                0,
                1,
                choices=("float32", "uint8"),
            ),
        ),
        racc_index,
        max_inputs=3,
        inputs=(
            InputSpec("channel_1", "array", "Channel 1 image"),
            InputSpec("channel_2", "array", "Channel 2 image"),
            InputSpec("roi_mask", "mask_or_labels", "ROI mask"),
        ),
        execution_policy="manual",
    ),
    OperationSpec(
        "object_colocalization_metrics",
        "Object Colocalization Metrics",
        COLOCALIZATION_CATEGORY,
        "array",
        "table",
        (
            SPATIAL_MODE_PARAMETER,
            ParameterSpec(
                "threshold_mode",
                "Thresholds",
                "choice",
                "Manual",
                0,
                0,
                1,
                choices=("Manual", "Costes auto"),
            ),
            ParameterSpec(
                "channel_1_threshold",
                "Channel 1 threshold (0-255)",
                "float",
                25.0,
                0.0,
                255.0,
                1.0,
                2,
            ),
            ParameterSpec(
                "channel_2_threshold",
                "Channel 2 threshold (0-255)",
                "float",
                25.0,
                0.0,
                255.0,
                1.0,
                2,
            ),
        ),
        object_colocalization_metrics,
        max_inputs=3,
        inputs=(
            InputSpec("labels", "mask_or_labels", "Object labels"),
            InputSpec("channel_1", "array", "Channel 1 image"),
            InputSpec("channel_2", "array", "Channel 2 image"),
        ),
        execution_policy="manual",
    ),
    OperationSpec(
        "label_overlap_association",
        "Label Overlap Association",
        COLOCALIZATION_CATEGORY,
        "mask_or_labels",
        "table",
        (SPATIAL_MODE_PARAMETER,),
        label_overlap_association,
        max_inputs=2,
        inputs=(
            InputSpec("reference", "mask_or_labels", "Reference labels"),
            InputSpec("target", "mask_or_labels", "Target labels"),
        ),
        execution_policy="manual",
    ),
    OperationSpec(
        "nearest_object_distance",
        "Nearest Object Distance",
        COLOCALIZATION_CATEGORY,
        "mask_or_labels",
        "table",
        (SPATIAL_MODE_PARAMETER,),
        nearest_object_distance,
        max_inputs=2,
        inputs=(
            InputSpec("reference", "mask_or_labels", "Reference labels"),
            InputSpec("target", "mask_or_labels", "Target labels"),
        ),
        execution_policy="manual",
    ),
    OperationSpec(
        "event_localization",
        "Event Localization",
        COLOCALIZATION_CATEGORY,
        "mask_or_labels",
        "table",
        (SPATIAL_MODE_PARAMETER,),
        event_localization,
        max_inputs=2,
        inputs=(
            InputSpec("events", "mask_or_labels", "Events / puncta"),
            InputSpec("regions", "mask_or_labels", "Regions / ROIs"),
        ),
        execution_policy="manual",
    ),
    OperationSpec(
        "merge_tables",
        "Merge Tables",
        MEASUREMENTS_CATEGORY,
        "table",
        "table",
        (
            ParameterSpec("input_count", "Input tables", "int", 2, 2, 8, 1),
            ParameterSpec(
                "join_mode",
                "Join mode",
                "choice",
                "Left join",
                0,
                0,
                1,
                choices=("Left join", "Inner join", "Outer join"),
            ),
            ParameterSpec(
                "join_keys",
                "Join keys (auto or comma-separated)",
                "text",
                "auto",
                0,
                0,
                1,
            ),
        ),
        merge_tables,
        max_inputs=8,
        subcategory="Tables",
    ),
    OperationSpec(
        "add_metadata_columns",
        "Add Metadata Columns",
        MEASUREMENTS_CATEGORY,
        "table",
        "table",
        (
            ParameterSpec(
                "metadata_columns",
                "Metadata (name=value, ...)",
                "text",
                "condition=control",
                0,
                0,
                1,
            ),
            ParameterSpec(
                "overwrite",
                "Overwrite existing columns",
                "choice",
                "no",
                0,
                0,
                1,
                choices=("no", "yes"),
            ),
        ),
        add_metadata_columns,
        subcategory="Tables",
    ),
    OperationSpec(
        "select_table_columns",
        "Select Table Columns",
        MEASUREMENTS_CATEGORY,
        "table",
        "table",
        (
            ParameterSpec(
                "columns",
                "Columns (comma-separated, auto = all)",
                "text",
                "auto",
                0,
                0,
                1,
            ),
        ),
        select_table_columns,
        subcategory="Tables",
    ),
    OperationSpec(
        "summarize_measurements",
        "Summarize Measurements",
        MEASUREMENTS_CATEGORY,
        "table",
        "table",
        (
            ParameterSpec(
                "group_by",
                "Group by columns (auto or comma-separated)",
                "text",
                "auto",
                0,
                0,
                1,
            ),
            ParameterSpec(
                "value_columns",
                "Value columns (auto or comma-separated)",
                "text",
                "auto",
                0,
                0,
                1,
            ),
            ParameterSpec(
                "statistics",
                "Statistics",
                "text",
                "count,mean,median,std,min,max,q25,q75",
                0,
                0,
                1,
            ),
        ),
        summarize_measurements,
        subcategory="Tables",
    ),
    OperationSpec(
        "extract_channel",
        "Extract Channel",
        IMAGE_DATA_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec("channel", "Channel", "int", 0, 0, 5, 1),
        ),
        extract_channel,
        subcategory=CHANNELS_COMPOSITES_GROUP,
    ),
    OperationSpec(
        "combine_channels",
        "Combine Channels",
        IMAGE_DATA_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec("input_count", "Input channels", "int", 2, 2, 12, 1),
        ),
        combine_channels,
        max_inputs=12,
        subcategory=CHANNELS_COMPOSITES_GROUP,
    ),
    OperationSpec(
        "calculate_weighted_image",
        "Calculate New Image",
        IMAGE_DATA_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec("input_count", "Inputs", "int", 2, 2, 12, 1),
            ParameterSpec("weights", "Weights", "text", "1,1", 0, 0, 1),
            ParameterSpec(
                "offset",
                "Offset",
                "float",
                0.0,
                -100000.0,
                100000.0,
                1.0,
                3,
            ),
        ),
        calculate_weighted_image,
        max_inputs=12,
        subcategory=MATH_LOGIC_GROUP,
    ),
    OperationSpec(
        "add_images",
        "Add",
        IMAGE_DATA_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec("input_count", "Inputs", "int", 2, 2, 12, 1),
        ),
        add_images,
        max_inputs=12,
        subcategory=MATH_LOGIC_GROUP,
    ),
    OperationSpec(
        "subtract_images",
        "Subtract",
        IMAGE_DATA_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec("input_count", "Inputs", "int", 2, 2, 12, 1),
        ),
        subtract_images,
        max_inputs=12,
        subcategory=MATH_LOGIC_GROUP,
    ),
    OperationSpec(
        "ratio_image",
        "Ratio",
        IMAGE_DATA_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec("input_count", "Inputs", "int", 2, 2, 2, 1),
            ParameterSpec("epsilon", "Epsilon", "float", 1e-6, 0.0, 1.0, 1e-6, 6),
        ),
        ratio_image,
        max_inputs=2,
        subcategory=MATH_LOGIC_GROUP,
    ),
    OperationSpec(
        "mask_image",
        "Mask Image",
        IMAGE_DATA_CATEGORY,
        "image",
        "image",
        (
            ParameterSpec(
                "outside_value",
                "Outside value",
                "float",
                0.0,
                -100000.0,
                100000.0,
                1.0,
                3,
            ),
            ParameterSpec(
                "invert_mask",
                "Invert mask",
                "choice",
                "no",
                0,
                0,
                1,
                choices=("no", "yes"),
            ),
        ),
        mask_image,
        max_inputs=2,
        inputs=(
            InputSpec("image", "image", "Image"),
            InputSpec("mask", "mask_or_labels", "Mask"),
        ),
        subcategory=MATH_LOGIC_GROUP,
    ),
    OperationSpec(
        "logical_and",
        "Logical AND",
        IMAGE_DATA_CATEGORY,
        "array",
        "mask",
        (
            ParameterSpec("input_count", "Inputs", "int", 2, 2, 12, 1),
        ),
        logical_and,
        max_inputs=12,
        subcategory=MATH_LOGIC_GROUP,
    ),
    OperationSpec(
        "logical_or",
        "Logical OR",
        IMAGE_DATA_CATEGORY,
        "array",
        "mask",
        (
            ParameterSpec("input_count", "Inputs", "int", 2, 2, 12, 1),
        ),
        logical_or,
        max_inputs=12,
        subcategory=MATH_LOGIC_GROUP,
    ),
    OperationSpec(
        "logical_xor",
        "Logical XOR",
        IMAGE_DATA_CATEGORY,
        "array",
        "mask",
        (
            ParameterSpec("input_count", "Inputs", "int", 2, 2, 12, 1),
        ),
        logical_xor,
        max_inputs=12,
        subcategory=MATH_LOGIC_GROUP,
    ),
    OperationSpec(
        "split_channels",
        "Split Channels",
        IMAGE_DATA_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec(
                "preview_channel",
                "Thumbnail channel",
                "int",
                0,
                0,
                15,
                1,
            ),
        ),
        split_channels,
        subcategory=CHANNELS_COMPOSITES_GROUP,
        output_factory=_split_channels_outputs,
        preserves_input_type=True,
    ),
    OperationSpec(
        "composite_to_rgb",
        "Composite \u2192 RGB",
        IMAGE_DATA_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec(
                "channel_axis", "Channel axis (-1 auto)", "int", -1, -1, 6, 1
            ),
            ParameterSpec("red_channel", "Red channel (-1 auto)", "int", -1, -1, 15, 1),
            ParameterSpec(
                "green_channel", "Green channel (-1 auto)", "int", -1, -1, 15, 1
            ),
            ParameterSpec(
                "blue_channel", "Blue channel (-1 auto)", "int", -1, -1, 15, 1
            ),
        ),
        composite_to_rgb,
        subcategory=CHANNELS_COMPOSITES_GROUP,
    ),
    OperationSpec(
        "assign_channel_colors",
        "Assign Channel Colors",
        IMAGE_DATA_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec(
                "channel_colors",
                "Channel colours",
                "text",
                "",
                0,
                0,
                1,
            ),
        ),
        assign_channel_colors,
        subcategory=CHANNELS_COMPOSITES_GROUP,
        preserves_input_type=True,
    ),
    OperationSpec(
        "rescale_intensity",
        "Rescale Intensity",
        INTENSITY_CONTRAST_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec(
                "cutoff_mode",
                "Input cutoffs",
                "choice",
                "Percentiles",
                0,
                0,
                1,
                choices=("Percentiles", "Values"),
                choice_labels=("Percentiles (exact)", "Explicit values"),
            ),
            ParameterSpec(
                "in_low_value",
                "Low value",
                "float",
                0.0,
                -100000.0,
                100000.0,
                0.01,
                3,
            ),
            ParameterSpec(
                "in_high_value",
                "High value",
                "float",
                1.0,
                -100000.0,
                100000.0,
                0.01,
                3,
            ),
            ParameterSpec(
                "in_low_percentile",
                "Low percentile",
                "float",
                0.0,
                0.0,
                100.0,
                0.1,
                2,
            ),
            ParameterSpec(
                "in_high_percentile",
                "High percentile",
                "float",
                100.0,
                0.0,
                100.0,
                0.1,
                2,
            ),
            ParameterSpec(
                "out_min",
                "Output min",
                "float",
                0.0,
                -100000.0,
                100000.0,
                0.01,
                3,
            ),
            ParameterSpec(
                "out_max",
                "Output max",
                "float",
                1.0,
                -100000.0,
                100000.0,
                0.01,
                3,
            ),
        ),
        rescale_intensity,
    ),
    OperationSpec(
        "normalize_image",
        "Normalize",
        INTENSITY_CONTRAST_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec(
                "method",
                "Method",
                "choice",
                "min-max",
                0,
                0,
                1,
                choices=("min-max", "z-score"),
            ),
        ),
        normalize_image,
    ),
    OperationSpec(
        "clip_intensity",
        "Clip",
        INTENSITY_CONTRAST_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec(
                "cutoff_mode",
                "Input cutoffs",
                "choice",
                "Data range",
                0,
                0,
                1,
                choices=("Data range", "Values"),
                choice_labels=("Full data range", "Explicit values"),
            ),
            ParameterSpec(
                "minimum",
                "Minimum",
                "float",
                0.0,
                -100000.0,
                100000.0,
                1.0,
                3,
            ),
            ParameterSpec(
                "maximum",
                "Maximum",
                "float",
                255.0,
                -100000.0,
                100000.0,
                1.0,
                3,
            ),
        ),
        clip_intensity,
    ),
    OperationSpec(
        "save_output",
        "Save Image",
        IMAGE_DATA_CATEGORY,
        "array",
        "any",
        (
            ParameterSpec(
                "enabled",
                "Auto-save on update",
                "choice",
                "off",
                0,
                0,
                1,
                choices=("off", "on"),
            ),
            ParameterSpec("path", "Path", "text", "", 0, 0, 1),
            ParameterSpec(
                "format",
                "Format",
                "choice",
                "auto",
                0,
                0,
                1,
                choices=(
                    "auto",
                    "ome-zarr",
                    "ome-zarr-0.5",
                    "ome-tiff",
                    "imagej-tiff",
                    "tiff",
                    "npy",
                    "png",
                    "jpeg",
                    "bmp",
                    "gif",
                    "webp",
                    "tga",
                    "pnm",
                ),
            ),
            ParameterSpec(
                "overwrite",
                "Overwrite",
                "choice",
                "no",
                0,
                0,
                1,
                choices=("no", "yes"),
            ),
        ),
        save_output,
        subcategory=SOURCE_OUTPUT_GROUP,
    ),
    OperationSpec(
        "batch_output",
        "Batch Output",
        IMAGE_DATA_CATEGORY,
        "any",
        "any",
        (
            ParameterSpec("tag", "Tag", "text", "output", 0, 0, 1),
            ParameterSpec(
                "format",
                "Format",
                "choice",
                "batch default",
                0,
                0,
                1,
                choices=(
                    "batch default",
                    "ome-tiff",
                    "imagej-tiff",
                    "tiff",
                    "npy",
                    "csv",
                    "tsv",
                ),
            ),
            ParameterSpec("subfolder", "Subfolder", "text", "", 0, 0, 1),
            ParameterSpec(
                "filename_template",
                "Filename template",
                "text",
                "{source_stem}__{tag}",
                0,
                0,
                1,
            ),
            ParameterSpec(
                "overwrite",
                "Overwrite",
                "choice",
                "batch default",
                0,
                0,
                1,
                choices=("batch default", "yes", "no"),
            ),
        ),
        batch_output,
        subcategory=SOURCE_OUTPUT_GROUP,
        preserves_input_type=True,
    ),
    OperationSpec(
        "convert_dtype",
        "Convert Dtype",
        IMAGE_DATA_CATEGORY,
        "array",
        "any",
        (
            ParameterSpec(
                "output_dtype",
                "Output dtype",
                "choice",
                "uint8",
                0,
                0,
                1,
                choices=("uint8", "uint16", "float32", "bool"),
            ),
            ParameterSpec(
                "scaling",
                "Scaling",
                "choice",
                "rescale",
                0,
                0,
                1,
                choices=("rescale", "clip", "preserve"),
            ),
        ),
        convert_dtype,
        subcategory=UTILITIES_GROUP,
    ),
    OperationSpec(
        "invert",
        "Invert",
        IMAGE_DATA_CATEGORY,
        "any",
        "any",
        (),
        invert,
        subcategory=MATH_LOGIC_GROUP,
    ),
)

NODE_LIBRARY_BY_ID = {spec.id: spec for spec in NODE_LIBRARY}
_RESOLVED_SPATIAL_PARAMETER_OPERATION_IDS = frozenset(
    spec.id
    for spec in NODE_LIBRARY
    if spec.function is not None
    and "resolved_spatial_ndim" in inspect.signature(spec.function).parameters
)
PALETTE_NODE_LIBRARY = NODE_LIBRARY


def graph_node_from_persisted_params(
    node_id: object,
    operation_id: object,
    saved_params: object,
    *,
    index: int,
) -> GraphNode:
    """Build one graph node using the schema-v2 parameter validation rules."""
    context = f"Node {index}"
    if not isinstance(operation_id, str) or not operation_id.strip():
        raise ValueError(f"{context} requires non-empty 'operation_id'.")
    spec = NODE_LIBRARY_BY_ID.get(operation_id)
    if spec is None:
        raise ValueError(
            f"Node {index} uses unknown operation {operation_id!r}."
        )
    if not isinstance(node_id, str) or not node_id.strip():
        raise ValueError(f"{context} requires non-empty 'id'.")
    if not isinstance(saved_params, dict):
        raise ValueError(f"Parameters for node {node_id!r} must be an object.")

    required_params = {parameter.name for parameter in spec.parameters}
    missing_params = required_params - saved_params.keys()
    if missing_params:
        missing = ", ".join(sorted(missing_params))
        raise ValueError(
            f"Node {node_id!r} is missing required parameters: {missing}."
        )
    unknown_params = [
        name
        for name in saved_params
        if not isinstance(name, str)
        or (
            name not in required_params
            and not name.startswith("_vipp_")
            and optional_persisted_parameter_spec(spec, name) is None
        )
    ]
    if unknown_params:
        unknown = ", ".join(
            repr(name) for name in sorted(unknown_params, key=str)
        )
        raise ValueError(f"Node {node_id!r} has unknown parameters: {unknown}.")

    # Keep this shallow copy for the established deserializer contract. Runtime
    # snapshots pass a defensive deep copy into this shared validator.
    params = dict(saved_params)
    for parameter in spec.parameters:
        validate_parameter_value(
            parameter,
            params[parameter.name],
            context=f"Node {node_id!r} parameter",
        )
    for name, value in params.items():
        optional_spec = optional_persisted_parameter_spec(spec, name)
        if optional_spec is not None:
            validate_optional_persisted_parameter(
                spec,
                optional_spec,
                value,
                context=f"Node {node_id!r} parameter",
            )
    return GraphNode(
        node_id,
        spec.id,
        spec.title,
        spec.category,
        spec.input_type,
        spec.output_type,
        params,
        spec.max_inputs,
    )

PROTOTYPE_NODES = [
    GraphNode(
        "input",
        "input",
        "Image Source",
        IMAGE_DATA_CATEGORY,
        None,
        "image",
        {param.name: param.default for param in SOURCE_PARAMETERS},
    ),
    GraphNode(
        "gaussian",
        "gaussian_blur",
        "Gaussian Blur",
        "Filtering",
        "array",
        "image",
        {"sigma": 1.2},
    ),
    GraphNode(
        "threshold",
        "otsu_threshold",
        "Otsu Threshold",
        "Segmentation",
        "image",
        "mask",
        {"threshold_scope": "Stack histogram", "histogram_bins": 256},
    ),
]
PROTOTYPE_CONNECTIONS = [
    GraphConnection("input", "gaussian"),
    GraphConnection("gaussian", "threshold"),
]


class PrototypePipeline:
    """A small executable single-input graph for the interaction prototype."""

    def __init__(self) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self.connections: list[GraphConnection] = []
        self.outputs: dict[str, Any] = {}
        self.output_states: dict[str, ImageState | TableState | None] = {}
        self.node_outputs: dict[str, list[Any]] = {}
        self.node_output_states: dict[str, list[ImageState | TableState | None]] = {}
        self.output_tunnels: dict[str, OutputTunnel] = {}
        self.completed_node_ids: set[str] = set()
        self.node_execution_states: dict[str, str] = {}
        self.node_execution_messages: dict[str, str] = {}
        self._counters: Counter[str] = Counter()
        self.reset_starter_graph()

    def reset_starter_graph(self) -> None:
        self.nodes = {node.id: _clone_node(node) for node in PROTOTYPE_NODES}
        self.connections = list(PROTOTYPE_CONNECTIONS)
        self.outputs = {}
        self.output_states = {}
        self.node_outputs = {}
        self.node_output_states = {}
        self.output_tunnels = {}
        self.completed_node_ids = set()
        self.node_execution_states = {
            node_id: EXECUTION_NOT_CALCULATED for node_id in self.nodes
        }
        self.node_execution_messages = {node_id: "" for node_id in self.nodes}
        self._counters = Counter()
        for node in self.nodes.values():
            self._counters[node.operation_id] += 1

    def reset_empty_graph(self) -> None:
        """Reset to one unbound Image Source node."""
        input_node = _clone_node(PROTOTYPE_NODES[0])
        input_node.params.update(
            {
                "source_mode": "file path",
                "layer_name": "",
                "file_path": "",
                "sample_name": "",
                "series_index": 0,
                "binding_mode": "single item",
                "channel_colors": "",
            }
        )
        self.nodes = {input_node.id: input_node}
        self.connections = []
        self.outputs = {}
        self.output_states = {}
        self.node_outputs = {}
        self.node_output_states = {}
        self.output_tunnels = {}
        self.completed_node_ids = set()
        self.node_execution_states = {
            node_id: EXECUTION_NOT_CALCULATED for node_id in self.nodes
        }
        self.node_execution_messages = {node_id: "" for node_id in self.nodes}
        self._counters = Counter({input_node.operation_id: 1})

    def restore_graph(
        self,
        nodes: Iterable[GraphNode],
        connections: Iterable[GraphConnection],
        output_tunnels: Iterable[OutputTunnel] | None = None,
    ) -> None:
        """Replace the current graph with deserialized nodes and connections."""
        node_list = list(nodes)
        connection_list = list(connections)
        tunnel_list = list(output_tunnels or ())
        restored = object.__new__(type(self))
        restored.nodes = {}
        for index, node in enumerate(node_list):
            validated = graph_node_from_persisted_params(
                node.id,
                node.operation_id,
                node.params,
                index=index,
            )
            restored.nodes[validated.id] = validated
        if len(restored.nodes) != len(node_list):
            raise ValueError("Cannot restore a graph with duplicate node ids.")
        valid = set(restored.nodes)
        restored.connections = connection_list
        restored.outputs = {node_id: None for node_id in restored.nodes}
        restored.output_states = {node_id: None for node_id in restored.nodes}
        restored.node_outputs = {node_id: [] for node_id in restored.nodes}
        restored.node_output_states = {node_id: [] for node_id in restored.nodes}
        restored.output_tunnels = {}
        restored.completed_node_ids = set()
        restored.node_execution_states = {
            node_id: EXECUTION_NOT_CALCULATED for node_id in restored.nodes
        }
        restored.node_execution_messages = {
            node_id: "" for node_id in restored.nodes
        }
        restored._counters = Counter()

        occupied_inputs: set[tuple[str, int]] = set()
        for connection in restored.connections:
            if (
                connection.source_id not in valid
                or connection.target_id not in valid
            ):
                raise ValueError(
                    "Cannot restore a connection that references a missing node: "
                    f"{connection.source_id!r} -> {connection.target_id!r}."
                )
            if connection.source_id == connection.target_id:
                raise ValueError("Cannot restore a connection from a node to itself.")
            if any(
                isinstance(port, bool) or not isinstance(port, int)
                for port in (connection.target_port, connection.source_port)
            ):
                raise ValueError("Cannot restore a connection with a non-integer port.")
            if connection.target_port < 0 or connection.source_port < 0:
                raise ValueError("Cannot restore a connection with a negative port.")
            target_slot = (connection.target_id, connection.target_port)
            if target_slot in occupied_inputs:
                raise ValueError(
                    "Cannot restore multiple connections to "
                    f"{connection.target_id!r} input {connection.target_port}."
                )
            occupied_inputs.add(target_slot)
        cyclic_nodes = restored._cyclic_node_ids()
        if cyclic_nodes:
            names = ", ".join(repr(node_id) for node_id in cyclic_nodes)
            raise ValueError(f"Cannot restore a graph containing a cycle: {names}.")

        for tunnel in tunnel_list:
            if isinstance(tunnel.source_port, bool) or not isinstance(
                tunnel.source_port,
                int,
            ):
                raise ValueError(
                    f"Tunnel '{tunnel.name}' references a non-integer output port."
                )
            if tunnel.source_port < 0:
                raise ValueError(
                    f"Tunnel '{tunnel.name}' references a negative output port."
                )

        restored._ensure_dynamic_output_hints(connection_list, tunnel_list)
        for tunnel in tunnel_list:
            restored._restore_output_tunnel(tunnel)
        canonical_connections: list[GraphConnection] = []
        for connection in restored.connections:
            restored._validate_restored_connection(connection)
            if connection.tunnel_name:
                declared_tunnel = restored.output_tunnel(connection.tunnel_name)
                connection = replace(
                    connection,
                    tunnel_name=declared_tunnel.name,
                )
            canonical_connections.append(connection)
        restored.connections = canonical_connections

        for node in restored.nodes.values():
            restored._counters[node.operation_id] += 1
        self.nodes = restored.nodes
        self.connections = restored.connections
        self.outputs = restored.outputs
        self.output_states = restored.output_states
        self.node_outputs = restored.node_outputs
        self.node_output_states = restored.node_output_states
        self.output_tunnels = restored.output_tunnels
        self.completed_node_ids = restored.completed_node_ids
        self.node_execution_states = restored.node_execution_states
        self.node_execution_messages = restored.node_execution_messages
        self._counters = restored._counters

    def topological_order(self) -> list[str]:
        """Return node ids in dependency order (sources first).

        Declaration order is preserved among nodes whose inputs are ready.
        Any nodes left in a cycle are appended in declaration order so the
        result always contains every node exactly once.
        """
        order: list[str] = []
        done: set[str] = set()
        remaining = list(self.nodes)
        while remaining:
            progressed = False
            for node_id in list(remaining):
                if all(src in done for src in self._input_sources(node_id)):
                    order.append(node_id)
                    done.add(node_id)
                    remaining.remove(node_id)
                    progressed = True
            if not progressed:
                order.extend(remaining)
                break
        return order

    def add_node(self, operation_id: str) -> GraphNode:
        spec = self.operation_spec(operation_id)
        self._counters[operation_id] += 1
        node_id = _node_id(operation_id, self._counters[operation_id])
        while node_id in self.nodes:
            self._counters[operation_id] += 1
            node_id = _node_id(operation_id, self._counters[operation_id])

        node = GraphNode(
            node_id,
            spec.id,
            spec.title,
            spec.category,
            spec.input_type,
            spec.output_type,
            {param.name: param.default for param in spec.parameters},
            spec.max_inputs,
        )
        self.nodes[node.id] = node
        self.outputs[node.id] = None
        self.output_states[node.id] = None
        self.node_outputs[node.id] = []
        self.node_output_states[node.id] = []
        self.node_execution_states[node.id] = EXECUTION_NOT_CALCULATED
        self.node_execution_messages[node.id] = ""
        return node

    def remove_node(self, node_id: str) -> bool:
        if node_id not in self.nodes:
            return False
        removed_tunnels = {
            tunnel.name
            for tunnel in self.output_tunnels.values()
            if tunnel.source_id == node_id
        }
        del self.nodes[node_id]
        self.outputs.pop(node_id, None)
        self.output_states.pop(node_id, None)
        self.node_outputs.pop(node_id, None)
        self.node_output_states.pop(node_id, None)
        self.node_execution_states.pop(node_id, None)
        self.node_execution_messages.pop(node_id, None)
        for tunnel_name in removed_tunnels:
            self.output_tunnels.pop(_tunnel_key(tunnel_name), None)
        self.connections = [
            connection
            for connection in self.connections
            if connection.source_id != node_id and connection.target_id != node_id
            and connection.tunnel_name not in removed_tunnels
        ]
        return True

    def connect(
        self,
        source_id: str,
        target_id: str,
        target_port: int | None = None,
        source_port: int = 0,
        tunnel_name: str = "",
    ) -> ConnectionResult:
        if source_id not in self.nodes or target_id not in self.nodes:
            return ConnectionResult(False, "Cannot connect missing nodes.")
        if source_id == target_id:
            return ConnectionResult(False, "Cannot connect a node to itself.")

        target = self.nodes[target_id]
        if not target.has_input:
            return ConnectionResult(False, "That node does not accept an input.")
        source_ports = self.output_ports(source_id)
        if not 0 <= source_port < len(source_ports):
            return ConnectionResult(False, "That node does not have that output.")
        source_output_type = source_ports[source_port].output_type
        tunnel_name = _clean_tunnel_name(tunnel_name)
        if tunnel_name:
            tunnel = self.output_tunnel(tunnel_name)
            if tunnel is None:
                return ConnectionResult(False, f"Unknown tunnel '{tunnel_name}'.")
            if tunnel.source_id != source_id or tunnel.source_port != source_port:
                return ConnectionResult(
                    False,
                    (
                        f"Tunnel '{tunnel.name}' is not assigned to that output "
                        "port."
                    ),
                )
        if self._would_create_cycle(source_id, target_id):
            return ConnectionResult(False, "Cannot connect nodes in a cycle.")

        existing_targets = [
            existing
            for existing in self.connections
            if existing.target_id == target_id
        ]
        removed: tuple[GraphConnection, ...] = ()
        if self._node_accepts_multiple_inputs(target):
            maximum = self._max_inputs_for(target)
            port = self._target_port_for_connection(
                target,
                existing_targets,
                target_port,
                source_output_type,
            )
            if port is None:
                if target_port is not None:
                    expected_type = self._input_type_for_port(target, target_port)
                    return ConnectionResult(
                        False,
                        (
                            f"Cannot connect {source_output_type} output to "
                            f"{expected_type} input."
                        ),
                    )
                return ConnectionResult(
                    False,
                    f"That node already has {maximum} connected inputs.",
                )
            removed = tuple(
                existing
                for existing in existing_targets
                if existing.target_port == port
            )
            self.connections = [
                existing
                for existing in self.connections
                if not (
                    existing.target_id == target_id
                    and existing.target_port == port
                )
            ]
        else:
            port = 0
            expected_type = self._input_type_for_port(target, port)
            if not self._types_compatible(source_output_type, expected_type):
                return ConnectionResult(
                    False,
                    (
                        f"Cannot connect {source_output_type} output to "
                        f"{expected_type} input."
                    ),
                )
            removed = tuple(existing_targets)
            self.connections = [
                existing
                for existing in self.connections
                if existing.target_id != target_id
            ]
        connection = GraphConnection(
            source_id,
            target_id,
            port,
            source_port,
            tunnel_name,
        )
        if connection in self.connections:
            return ConnectionResult(
                True,
                "Those nodes are already connected.",
                connection=connection,
            )
        self.connections.append(connection)
        return ConnectionResult(
            True,
            f"Connected input to tunnel '{tunnel_name}'."
            if tunnel_name
            else "Connected nodes.",
            removed,
            connection,
        )

    def add_output_tunnel(
        self,
        name: str,
        source_id: str,
        source_port: int = 0,
    ) -> OutputTunnel:
        tunnel = OutputTunnel(_clean_tunnel_name(name), source_id, int(source_port))
        self._validate_new_output_tunnel(tunnel)
        self.output_tunnels[_tunnel_key(tunnel.name)] = tunnel
        return tunnel

    def rename_output_tunnel(self, old_name: str, new_name: str) -> OutputTunnel:
        old_key = _tunnel_key(old_name)
        current = self.output_tunnels.get(old_key)
        if current is None:
            raise ValueError(f"Unknown tunnel '{_clean_tunnel_name(old_name)}'.")
        renamed = OutputTunnel(
            _clean_tunnel_name(new_name),
            current.source_id,
            current.source_port,
        )
        self._validate_new_output_tunnel(renamed, allow_replace_key=old_key)
        self.output_tunnels.pop(old_key, None)
        self.output_tunnels[_tunnel_key(renamed.name)] = renamed
        self.connections = [
            replace(connection, tunnel_name=renamed.name)
            if connection.tunnel_name == current.name
            else connection
            for connection in self.connections
        ]
        return renamed

    def remove_output_tunnel(self, name: str) -> tuple[GraphConnection, ...]:
        key = _tunnel_key(name)
        tunnel = self.output_tunnels.pop(key, None)
        if tunnel is None:
            return ()
        removed = tuple(
            connection
            for connection in self.connections
            if connection.tunnel_name == tunnel.name
        )
        if removed:
            self.connections = [
                connection
                for connection in self.connections
                if connection not in removed
            ]
        return removed

    def connect_to_tunnel(
        self,
        name: str,
        target_id: str,
        target_port: int | None = None,
    ) -> ConnectionResult:
        tunnel = self.output_tunnel(name)
        if tunnel is None:
            return ConnectionResult(False, f"Unknown tunnel '{name}'.")
        return self.connect(
            tunnel.source_id,
            target_id,
            target_port,
            tunnel.source_port,
            tunnel.name,
        )

    def output_tunnel(self, name: str) -> OutputTunnel | None:
        return self.output_tunnels.get(_tunnel_key(name))

    def output_tunnel_for_port(
        self,
        source_id: str,
        source_port: int = 0,
    ) -> OutputTunnel | None:
        for tunnel in self.output_tunnels.values():
            if tunnel.source_id == source_id and tunnel.source_port == int(source_port):
                return tunnel
        return None

    def output_tunnel_list(self) -> tuple[OutputTunnel, ...]:
        return tuple(sorted(self.output_tunnels.values(), key=lambda item: item.name))

    def tunnel_connection_for_input(
        self,
        target_id: str,
        target_port: int = 0,
    ) -> GraphConnection | None:
        for connection in self.connections:
            if (
                connection.target_id == target_id
                and connection.target_port == int(target_port)
                and connection.tunnel_name
            ):
                return connection
        return None

    def compatible_output_tunnels(
        self,
        target_id: str,
        target_port: int = 0,
    ) -> tuple[OutputTunnel, ...]:
        target = self.nodes.get(target_id)
        if target is None or not target.has_input:
            return ()
        target_type = self._input_type_for_port(target, int(target_port))
        compatible: list[OutputTunnel] = []
        for tunnel in self.output_tunnel_list():
            if tunnel.source_id == target_id:
                continue
            source_ports = self.output_ports(tunnel.source_id)
            if not 0 <= tunnel.source_port < len(source_ports):
                continue
            source_type = source_ports[tunnel.source_port].output_type
            if self._types_compatible(source_type, target_type):
                compatible.append(tunnel)
        return tuple(compatible)

    def disconnect(
        self,
        source_id: str,
        target_id: str,
        target_port: int | None = None,
    ) -> bool:
        before = len(self.connections)
        self.connections = [
            connection
            for connection in self.connections
            if not (
                connection.source_id == source_id
                and connection.target_id == target_id
                and (
                    target_port is None
                    or connection.target_port == int(target_port)
                )
            )
        ]
        return len(self.connections) != before

    def trim_invalid_connections(self, node_id: str) -> tuple[GraphConnection, ...]:
        node = self.nodes.get(node_id)
        if node is None:
            return ()
        count = self.input_port_count(node_id)
        removed = tuple(
            connection
            for connection in self.connections
            if connection.target_id == node_id and connection.target_port >= count
        )
        if removed:
            self.connections = [
                connection
                for connection in self.connections
                if connection not in removed
            ]
        return removed

    def trim_invalid_output_connections(
        self, node_id: str
    ) -> tuple[GraphConnection, ...]:
        """Drop edges whose source port no longer exists on ``node_id``.

        Used when a dynamic multi-output node (e.g. Split Channels) produces
        fewer ports than before, so downstream wires to removed ports are
        cleaned up rather than silently falling back to port 0.
        """
        if node_id not in self.nodes:
            return ()
        count = len(self.output_ports(node_id))
        removed = tuple(
            connection
            for connection in self.connections
            if connection.source_id == node_id and connection.source_port >= count
        )
        removed_tunnels = tuple(
            tunnel
            for tunnel in self.output_tunnels.values()
            if tunnel.source_id == node_id and tunnel.source_port >= count
        )
        if removed_tunnels:
            removed_tunnel_names = {tunnel.name for tunnel in removed_tunnels}
            removed = removed + tuple(
                connection
                for connection in self.connections
                if connection.tunnel_name in removed_tunnel_names
                and connection not in removed
            )
            for tunnel in removed_tunnels:
                self.output_tunnels.pop(_tunnel_key(tunnel.name), None)
        if removed:
            self.connections = [
                connection
                for connection in self.connections
                if connection not in removed
            ]
        return removed

    def set_param(self, node_id: str, name: str, value: Any) -> None:
        node = self.nodes[node_id]
        spec = next(
            (
                parameter
                for parameter in self.operation_spec(node.operation_id).parameters
                if parameter.name == name
            ),
            None,
        )
        if spec is None:
            operation_spec = self.operation_spec(node.operation_id)
            optional_spec = optional_persisted_parameter_spec(
                operation_spec,
                name,
            )
            if optional_spec is not None:
                validate_optional_persisted_parameter(
                    operation_spec,
                    optional_spec,
                    value,
                    context=f"Node {node_id!r} parameter",
                )
            elif not isinstance(name, str) or not name.startswith("_vipp_"):
                raise ValueError(
                    f"Node {node_id!r} has no public parameter {name!r}."
                )
        else:
            validate_parameter_value(
                spec,
                value,
                context=f"Node {node_id!r} parameter",
            )
        node.params[name] = value

    def node_auto_recalculate(self, node_id: str) -> bool:
        node = self.nodes.get(node_id)
        if node is None or not self.is_manual_node(node_id):
            return False
        return bool(node.params.get(MANUAL_AUTO_RECALCULATE_PARAM, False))

    def set_node_auto_recalculate(self, node_id: str, enabled: bool) -> None:
        if node_id not in self.nodes or not self.is_manual_node(node_id):
            return
        self.nodes[node_id].params[MANUAL_AUTO_RECALCULATE_PARAM] = bool(enabled)

    def _validate_restored_connection(self, connection: GraphConnection) -> None:
        target = self.nodes[connection.target_id]
        if connection.source_id == connection.target_id:
            raise ValueError("Cannot restore a connection from a node to itself.")
        if connection.target_port < 0 or connection.source_port < 0:
            raise ValueError("Cannot restore a connection with a negative port.")
        if not target.has_input:
            raise ValueError(
                f"Cannot restore a connection to source node {connection.target_id!r}."
            )
        input_count = self.input_port_count(connection.target_id)
        if connection.target_port >= input_count:
            raise ValueError(
                f"Cannot restore connection to {connection.target_id!r} input "
                f"{connection.target_port}; node has {input_count} input port(s)."
            )
        source_ports = self.output_ports(connection.source_id)
        if connection.source_port >= len(source_ports):
            raise ValueError(
                f"Cannot restore connection from {connection.source_id!r} output "
                f"{connection.source_port}; node has {len(source_ports)} output "
                "port(s)."
            )
        source_type = source_ports[connection.source_port].output_type
        target_type = self._input_type_for_port(target, connection.target_port)
        if not self._types_compatible(source_type, target_type):
            raise ValueError(
                f"Cannot restore {source_type} output to {target_type} input: "
                f"{connection.source_id!r} -> {connection.target_id!r}."
            )
        if connection.tunnel_name:
            tunnel = self.output_tunnel(connection.tunnel_name)
            if tunnel is None:
                raise ValueError(
                    f"Connection references unknown tunnel "
                    f"{connection.tunnel_name!r}."
                )
            if (
                tunnel.source_id != connection.source_id
                or tunnel.source_port != connection.source_port
            ):
                raise ValueError(
                    f"Connection tunnel {connection.tunnel_name!r} does not match "
                    "its declared source output."
                )

    def _ensure_dynamic_output_hints(
        self,
        connections: Iterable[GraphConnection],
        output_tunnels: Iterable[OutputTunnel],
    ) -> None:
        """Keep referenced dynamic ports available until source data is loaded."""
        required_counts: dict[str, int] = {}
        for item in (*tuple(connections), *tuple(output_tunnels)):
            source_id = item.source_id
            source_port = int(item.source_port)
            if source_id not in self.nodes or source_port < 0:
                continue
            required_counts[source_id] = max(
                required_counts.get(source_id, 0),
                source_port + 1,
            )
        for source_id, required in required_counts.items():
            node = self.nodes[source_id]
            spec = self.operation_spec(node.operation_id)
            if spec.output_factory is None:
                continue
            current = _dynamic_output_count_hint(node.params)
            if required > max(current or 0, DEFAULT_DYNAMIC_OUTPUT_PORTS):
                node.params[DYNAMIC_OUTPUT_COUNT_PARAM] = required

    def _cyclic_node_ids(self) -> tuple[str, ...]:
        indegree = {node_id: 0 for node_id in self.nodes}
        downstream: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}
        for connection in self.connections:
            indegree[connection.target_id] += 1
            downstream[connection.source_id].append(connection.target_id)
        ready = [node_id for node_id in self.nodes if indegree[node_id] == 0]
        visited: set[str] = set()
        while ready:
            node_id = ready.pop()
            visited.add(node_id)
            for target_id in downstream[node_id]:
                indegree[target_id] -= 1
                if indegree[target_id] == 0:
                    ready.append(target_id)
        return tuple(node_id for node_id in self.nodes if node_id not in visited)

    def _restore_output_tunnel(self, tunnel: OutputTunnel) -> None:
        restored = OutputTunnel(
            _clean_tunnel_name(tunnel.name),
            tunnel.source_id,
            tunnel.source_port,
        )
        self._validate_new_output_tunnel(restored)
        self.output_tunnels[_tunnel_key(restored.name)] = restored

    def _validate_new_output_tunnel(
        self,
        tunnel: OutputTunnel,
        *,
        allow_replace_key: str | None = None,
    ) -> None:
        if not tunnel.name:
            raise ValueError("Tunnel name cannot be blank.")
        key = _tunnel_key(tunnel.name)
        if key in self.output_tunnels and key != allow_replace_key:
            raise ValueError(f"Tunnel '{tunnel.name}' already exists.")
        if tunnel.source_id not in self.nodes:
            raise ValueError(
                f"Tunnel '{tunnel.name}' references missing source "
                f"{tunnel.source_id!r}."
            )
        source_ports = self.output_ports(tunnel.source_id)
        if not 0 <= tunnel.source_port < len(source_ports):
            raise ValueError(
                f"Tunnel '{tunnel.name}' references output {tunnel.source_port}, "
                f"but {tunnel.source_id!r} has {len(source_ports)} output port(s)."
            )
        for existing_key, existing in self.output_tunnels.items():
            if existing_key == allow_replace_key:
                continue
            if (
                existing.source_id == tunnel.source_id
                and existing.source_port == tunnel.source_port
            ):
                raise ValueError(
                    f"Output {tunnel.source_id!r} port {tunnel.source_port} "
                    f"already has tunnel '{existing.name}'."
                )

    def operation_spec(self, operation_id: str) -> OperationSpec:
        return NODE_LIBRARY_BY_ID[operation_id]

    def output_ports(self, node_id: str) -> tuple[OutputSpec, ...]:
        node = self.nodes.get(node_id)
        if node is None:
            return ()
        spec = self.operation_spec(node.operation_id)
        if spec.output_factory is not None:
            inferred = self._inferred_dynamic_output_count_for_node(node_id, spec)
            if inferred is None:
                count = len(self.node_outputs.get(node_id, ()))
                if count <= 0:
                    count = (
                        _dynamic_output_count_hint(node.params)
                        or DEFAULT_DYNAMIC_OUTPUT_PORTS
                    )
            else:
                count = inferred
            ports = spec.output_factory(count)
        else:
            ports = spec.output_ports
        if spec.id == "split_axis":
            ports = self._labeled_split_axis_ports(node_id, ports)
        if spec.id == "born_wolf_psf":
            ports = self._labeled_born_wolf_psf_ports(node_id, ports)
        return self._resolved_output_port_types(node_id, ports)

    def inferred_dynamic_output_count(
        self,
        operation_id: str,
        source_id: str,
        source_port: int = 0,
        params: dict[str, Any] | None = None,
    ) -> int | None:
        spec = self.operation_spec(operation_id)
        if spec.output_factory is None:
            return None
        if operation_id == "split_channels":
            return self._split_channel_output_count(source_id, source_port)
        if operation_id == "split_axis":
            return self._split_axis_output_count(source_id, source_port, params or {})
        if operation_id == "born_wolf_psf":
            return self._born_wolf_psf_output_count(
                source_id,
                source_port,
                params or {},
            )
        return None

    def input_ports(self, node_id: str) -> tuple[InputSpec, ...]:
        node = self.nodes.get(node_id)
        if node is None or not node.has_input:
            return ()
        spec = self.operation_spec(node.operation_id)
        if spec.inputs:
            return spec.input_ports
        count = self.input_port_count(node_id)
        input_type = node.input_type or "any"
        return tuple(
            InputSpec(f"input_{index + 1}", input_type, f"Input {index + 1}")
            for index in range(count)
        )

    def _resolved_output_port_types(
        self,
        node_id: str,
        ports: tuple[OutputSpec, ...],
    ) -> tuple[OutputSpec, ...]:
        spec = self.operation_spec(self.nodes[node_id].operation_id)
        if not spec.preserves_input_type:
            return ports
        connections = self._input_connections(node_id)
        if not connections:
            return ports
        source = connections[0]
        source_ports = self.output_ports(source.source_id)
        if not 0 <= source.source_port < len(source_ports):
            return ports
        output_type = source_ports[source.source_port].output_type
        return tuple(replace(port, output_type=output_type) for port in ports)

    def _inferred_dynamic_output_count_for_node(
        self,
        node_id: str,
        spec: OperationSpec,
    ) -> int | None:
        connections = self._input_connections(node_id)
        if not connections:
            return None
        source = connections[0]
        if spec.id == "split_channels":
            return self._split_channel_output_count(
                source.source_id,
                source.source_port,
            )
        if spec.id == "split_axis":
            node = self.nodes.get(node_id)
            return self._split_axis_output_count(
                source.source_id,
                source.source_port,
                self._public_params(node.params) if node is not None else {},
            )
        if spec.id == "born_wolf_psf":
            node = self.nodes.get(node_id)
            return self._born_wolf_psf_output_count(
                source.source_id,
                source.source_port,
                self._public_params(node.params) if node is not None else {},
            )
        return None

    def _split_channel_output_count(
        self,
        source_id: str,
        source_port: int,
    ) -> int | None:
        shape = self._resolved_output_shape(source_id, source_port)
        if not shape:
            return None
        state = self._resolved_output_state(source_id, source_port)
        axis = _strict_channel_axis_for_shape(shape, state)
        if axis is None:
            return 0
        try:
            return max(int(shape[axis]), 0)
        except (TypeError, ValueError, IndexError):
            return None

    def _split_axis_output_count(
        self,
        source_id: str,
        source_port: int,
        params: dict[str, Any],
    ) -> int | None:
        shape = self._resolved_output_shape(source_id, source_port)
        if not shape:
            return None
        axis = _split_axis_index_from_params(params, len(shape))
        try:
            return max(int(shape[axis]), 0)
        except (TypeError, ValueError, IndexError):
            return None

    def _born_wolf_psf_output_count(
        self,
        source_id: str,
        source_port: int,
        params: dict[str, Any],
    ) -> int | None:
        if not bool(params.get("auto_parameters", True)):
            return 1
        try:
            requested = int(params.get("channel", -1))
        except Exception:
            requested = -1
        if requested >= 0:
            return 1
        state = self._resolved_output_state(source_id, source_port)
        shape = self._resolved_output_shape(source_id, source_port)
        count = _state_channel_count(state, shape)
        return max(count, 1) if count is not None else None

    def _labeled_split_axis_ports(
        self,
        node_id: str,
        ports: tuple[OutputSpec, ...],
    ) -> tuple[OutputSpec, ...]:
        connections = self._input_connections(node_id)
        if not connections:
            return ports
        source = connections[0]
        shape = self._resolved_output_shape(source.source_id, source.source_port)
        if not shape:
            return ports
        node = self.nodes.get(node_id)
        axis_index = _split_axis_index_from_params(
            self._public_params(node.params) if node is not None else {},
            len(shape),
        )
        label = _split_axis_label(
            self._resolved_output_state(source.source_id, source.source_port),
            axis_index,
        )
        return tuple(
            replace(
                port,
                name=f"{label.lower()}_{index + 1}",
                title=f"{label} {index + 1}",
            )
            for index, port in enumerate(ports)
        )

    def _labeled_born_wolf_psf_ports(
        self,
        node_id: str,
        ports: tuple[OutputSpec, ...],
    ) -> tuple[OutputSpec, ...]:
        connections = self._input_connections(node_id)
        if not connections:
            return ports
        node = self.nodes.get(node_id)
        if node is None:
            return ports
        source = connections[0]
        state = self._resolved_output_state(source.source_id, source.source_port)
        channels = tuple(getattr(state, "channels", ()) or ())
        try:
            requested = int(node.params.get("channel", -1))
        except Exception:
            requested = -1
        labels = (
            [_born_wolf_psf_channel_label(channels, requested)]
            if requested >= 0
            else [
                _born_wolf_psf_channel_label(channels, index)
                for index in range(len(ports))
            ]
        )
        return tuple(
            replace(
                port,
                name=f"channel_{index + 1}_psf",
                title=f"{labels[index]} PSF",
            )
            for index, port in enumerate(ports)
            if index < len(labels)
        )

    def _resolved_output_shape(
        self,
        source_id: str,
        source_port: int,
    ) -> tuple[int, ...] | None:
        output = self._resolved_output(source_id, source_port)
        shape = getattr(output, "shape", None)
        if shape is None:
            state = self._resolved_output_state(source_id, source_port)
            shape = getattr(state, "shape", None)
        if shape is None:
            return None
        try:
            return tuple(int(size) for size in shape)
        except (TypeError, ValueError):
            return None

    def _resolved_output(self, source_id: str, source_port: int):
        outputs = self.node_outputs.get(source_id)
        if outputs:
            if 0 <= source_port < len(outputs):
                return outputs[source_port]
            return None
        return self.outputs.get(source_id)

    def _resolved_output_state(
        self, source_id: str, source_port: int
    ) -> ImageState | TableState | None:
        states = self.node_output_states.get(source_id)
        if states:
            if 0 <= source_port < len(states):
                return states[source_port]
            return None
        return self.output_states.get(source_id)

    def node_parameter_specs(self, node_id: str) -> tuple[ParameterSpec, ...]:
        node = self.nodes[node_id]
        return self.operation_spec(node.operation_id).parameters

    def input_data_for_node(self, node_id: str):
        connections = self._input_connections(node_id)
        if not connections:
            return None
        primary = connections[0]
        return self._resolved_output(primary.source_id, primary.source_port)

    def input_data_by_port_for_node(self, node_id: str) -> dict[int, Any]:
        return {
            int(connection.target_port): self._resolved_output(
                connection.source_id,
                connection.source_port,
            )
            for connection in self._input_connections(node_id)
        }

    def input_state_for_node(self, node_id: str) -> ImageState | None:
        connections = self._input_connections(node_id)
        if not connections:
            return None
        primary = connections[0]
        return self._resolved_output_state(primary.source_id, primary.source_port)

    def is_manual_node(self, node_id: str) -> bool:
        node = self.nodes.get(node_id)
        if node is None:
            return False
        return self.operation_spec(node.operation_id).execution_policy == "manual"

    def manual_node_ids(self) -> set[str]:
        return {node_id for node_id in self.nodes if self.is_manual_node(node_id)}

    def auto_recalculate_node_ids(self) -> set[str]:
        return {
            node_id
            for node_id in self.manual_node_ids()
            if self.node_auto_recalculate(node_id)
        }

    def set_node_execution_error(self, node_id: str | None, message: str) -> None:
        if node_id is None or node_id not in self.nodes:
            return
        self.node_execution_states[node_id] = EXECUTION_ERROR
        self.node_execution_messages[node_id] = str(message)

    def mark_manual_descendants_stale(self, node_ids: Iterable[str]) -> set[str]:
        """Mark manual nodes downstream of an edit as stale, preserving caches."""
        affected = self.descendants_inclusive(node_ids)
        changed: set[str] = set()
        for node_id in affected:
            if not self.is_manual_node(node_id):
                continue
            if self._has_cached_output(node_id):
                self.node_execution_states[node_id] = EXECUTION_STALE
                self.node_execution_messages[node_id] = (
                    "Upstream data or parameters changed. Recalculate to refresh "
                    "this cached result."
                )
            else:
                self.node_execution_states[node_id] = EXECUTION_NOT_CALCULATED
                self.node_execution_messages[node_id] = (
                    "This manual node has no result yet."
                )
            changed.add(node_id)
        return changed

    def _has_cached_output(self, node_id: str) -> bool:
        if self.outputs.get(node_id) is not None:
            return True
        return any(output is not None for output in self.node_outputs.get(node_id, ()))

    def _manual_nodes_to_skip(
        self,
        candidates: Iterable[str],
        manual_mode: str,
        manual_node_ids: Iterable[str] | None,
    ) -> set[str]:
        if manual_mode not in {MANUAL_RUN_CALCULATE, MANUAL_RUN_SKIP}:
            raise ValueError(f"Unknown manual execution mode: {manual_mode!r}.")
        manual_nodes = self.manual_node_ids() & set(candidates)
        if manual_mode == MANUAL_RUN_SKIP:
            selected = set(manual_node_ids or ())
            return manual_nodes - selected
        if manual_node_ids is None:
            return set()
        return manual_nodes - set(manual_node_ids)

    def _mark_skipped_manual_node(self, node_id: str, *, dirty: bool) -> bool:
        """Return whether a skipped manual node can satisfy downstream inputs."""
        has_cached_output = self._has_cached_output(node_id)
        if has_cached_output:
            self.node_execution_states[node_id] = (
                EXECUTION_STALE if dirty else self.node_execution_states.get(
                    node_id,
                    EXECUTION_READY,
                )
            )
            if self.node_execution_states[node_id] == EXECUTION_STALE:
                self.node_execution_messages[node_id] = (
                    "Cached result is stale. Recalculate to refresh this node."
                )
            else:
                self.node_execution_messages[node_id] = ""
            return True
        self.node_execution_states[node_id] = EXECUTION_NOT_CALCULATED
        self.node_execution_messages[node_id] = (
            "Click Calculate to produce this result."
        )
        return False

    @staticmethod
    def _public_params(params: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in params.items()
            if not key.startswith("_vipp_")
        }

    def _operation_kwargs(self, node: GraphNode) -> dict[str, Any]:
        kwargs = self._public_params(node.params)
        operation_spec = self.operation_spec(node.operation_id)
        required_specs = {
            parameter.name: parameter for parameter in operation_spec.parameters
        }
        for name, value in kwargs.items():
            parameter = required_specs.get(name)
            if parameter is not None:
                validate_parameter_value(
                    parameter,
                    value,
                    context=f"Node {node.id!r} parameter",
                )
                continue
            parameter = optional_persisted_parameter_spec(operation_spec, name)
            if parameter is None:
                raise ValueError(f"Node {node.id!r} has no public parameter {name!r}.")
            validate_optional_persisted_parameter(
                operation_spec,
                parameter,
                value,
                context=f"Node {node.id!r} parameter",
            )
        return kwargs

    def _sync_born_wolf_psf_resolution(
        self,
        node: GraphNode,
        source_output: Any,
        kwargs: dict[str, Any],
    ) -> None:
        shape = getattr(source_output, "shape", ())
        channel_indices = _born_wolf_psf_output_channel_indices(kwargs)
        resolutions = [
            resolve_born_wolf_psf_parameters(
                shape,
                kwargs.get("spatial_mode", "Auto from axes"),
                auto_parameters=bool(kwargs.get("auto_parameters", True)),
                wavelength_nm=kwargs.get("wavelength_nm", 0.0),
                numerical_aperture=kwargs.get("numerical_aperture", 0.0),
                refractive_index=kwargs.get("refractive_index", 0.0),
                pixel_size_xy_um=kwargs.get("pixel_size_xy_um", 0.0),
                z_step_um=kwargs.get("z_step_um", 0.0),
                channel=channel_index,
                resolved_spatial_ndim=kwargs.get("resolved_spatial_ndim"),
                axis_types=kwargs.get("axis_types", ()),
                axis_names=kwargs.get("axis_names", ()),
                axis_scales=kwargs.get("axis_scales", ()),
                axis_units=kwargs.get("axis_units", ()),
                channel_emission_wavelengths=kwargs.get(
                    "channel_emission_wavelengths",
                    (),
                ),
                channel_emission_wavelength_units=kwargs.get(
                    "channel_emission_wavelength_units",
                    (),
                ),
                channel_excitation_wavelengths=kwargs.get(
                    "channel_excitation_wavelengths",
                    (),
                ),
                channel_excitation_wavelength_units=kwargs.get(
                    "channel_excitation_wavelength_units",
                    (),
                ),
                objective_lens_na=kwargs.get("objective_lens_na"),
                objective_refractive_index=kwargs.get("objective_refractive_index"),
            )
            for channel_index in channel_indices
        ]
        resolution = resolutions[0]
        node.params["_vipp_psf_resolution"] = {
            name: {
                "value": result.value,
                "source": result.source,
                "message": result.message,
                "required": result.required,
            }
            for name, result in resolution.parameters.items()
        }
        node.params["_vipp_psf_channel_resolutions"] = [
            {
                "values": dict(item.values),
                "unresolved": list(item.unresolved),
                "parameters": {
                    name: {
                        "value": result.value,
                        "source": result.source,
                        "message": result.message,
                        "required": result.required,
                    }
                    for name, result in item.parameters.items()
                },
            }
            for item in resolutions
        ]
        unresolved_by_channel = [
            (index, item.unresolved)
            for index, item in enumerate(resolutions)
            if item.unresolved
        ]
        node.params["_vipp_psf_unresolved_parameters"] = sorted(
            {
                name
                for _index, unresolved in unresolved_by_channel
                for name in unresolved
            }
        )
        if unresolved_by_channel:
            unresolved = "; ".join(
                f"channel {index + 1}: {', '.join(unresolved)}"
                for index, unresolved in unresolved_by_channel
            )
            mode = "auto" if bool(kwargs.get("auto_parameters", True)) else "manual"
            raise ValueError(
                f"Cannot generate Born-Wolf PSF in {mode} mode; unresolved "
                f"parameter(s): {unresolved}."
            )
        if len(channel_indices) > 1:
            return
        for name in (*BORN_WOLF_PSF_AUTO_PARAMETERS, "channel"):
            if name in resolution.values:
                kwargs[name] = resolution.values[name]

    def _sync_colocalization_costes_thresholds(
        self,
        node: GraphNode,
        inputs: list[Any],
        kwargs: dict[str, Any],
    ) -> None:
        if node.operation_id not in COLOCALIZATION_THRESHOLD_OPERATIONS:
            return
        if not str(kwargs.get("threshold_mode", "Manual")).lower().startswith("costes"):
            return
        threshold_1, threshold_2 = colocalization_threshold_values(
            inputs,
            threshold_mode=kwargs.get("threshold_mode", "Costes auto"),
            channel_1_threshold=kwargs.get("channel_1_threshold", 25.0),
            channel_2_threshold=kwargs.get("channel_2_threshold", 25.0),
            intensity_max=kwargs.get("intensity_max", 255.0),
        )
        kwargs["channel_1_threshold"] = float(threshold_1)
        kwargs["channel_2_threshold"] = float(threshold_2)
        node.params["channel_1_threshold"] = float(threshold_1)
        node.params["channel_2_threshold"] = float(threshold_2)

    def run(
        self,
        input_data,
        input_metadata: dict | None = None,
        input_name: str = "",
        source_payloads: dict[str, SourcePayload] | None = None,
        dirty_node_ids: Iterable[str] | None = None,
        node_started_callback: Callable[[str], None] | None = None,
        progress_callback: Callable[[str, int, int, str], None] | None = None,
        cancel_callback: Callable[[], bool] | None = None,
        manual_mode: str = MANUAL_RUN_CALCULATE,
        manual_node_ids: Iterable[str] | None = None,
        retain_node_ids: Iterable[str] | None = None,
        prune_unretained: bool = False,
    ) -> dict[str, Any]:
        retained_nodes = {
            node_id for node_id in (retain_node_ids or ()) if node_id in self.nodes
        }
        dirty_nodes = self._validated_dirty_nodes(dirty_node_ids)
        if dirty_nodes is None:
            prior_outputs = dict(self.outputs)
            prior_output_states = dict(self.output_states)
            prior_node_outputs = {
                node_id: list(outputs)
                for node_id, outputs in self.node_outputs.items()
            }
            prior_node_output_states = {
                node_id: list(states)
                for node_id, states in self.node_output_states.items()
            }
            prior_execution_states = dict(self.node_execution_states)
            prior_execution_messages = dict(self.node_execution_messages)
            self.outputs = {node_id: None for node_id in self.nodes}
            self.output_states = {node_id: None for node_id in self.nodes}
            self.node_outputs = {node_id: [] for node_id in self.nodes}
            self.node_output_states = {node_id: [] for node_id in self.nodes}
            self.node_execution_states = {
                node_id: EXECUTION_NOT_CALCULATED for node_id in self.nodes
            }
            self.node_execution_messages = {node_id: "" for node_id in self.nodes}
            remaining = set(self.nodes)
            completed: set[str] = set()
            self.completed_node_ids = set()
        else:
            remaining = self.descendants_inclusive(dirty_nodes)
            self.completed_node_ids.difference_update(remaining)
            prior_outputs = {}
            prior_output_states = {}
            prior_node_outputs = {}
            prior_node_output_states = {}
            prior_execution_states = {}
            prior_execution_messages = {}
            skipped = self._manual_nodes_to_skip(
                remaining,
                manual_mode,
                manual_node_ids,
            )
            remaining.difference_update(skipped)
            for node_id in remaining:
                self.outputs[node_id] = None
                self.output_states[node_id] = None
                self.node_outputs[node_id] = []
                self.node_output_states[node_id] = []
                self.node_execution_states[node_id] = EXECUTION_NOT_CALCULATED
                self.node_execution_messages[node_id] = ""
            for node_id in skipped:
                if self._mark_skipped_manual_node(node_id, dirty=True):
                    self.completed_node_ids.add(node_id)
            completed = set(self.nodes) - remaining

        if dirty_nodes is None:
            skipped = self._manual_nodes_to_skip(
                remaining,
                manual_mode,
                manual_node_ids,
            )
            remaining.difference_update(skipped)
            for node_id in skipped:
                if node_id in prior_outputs:
                    self.outputs[node_id] = prior_outputs[node_id]
                    self.output_states[node_id] = prior_output_states.get(node_id)
                    self.node_outputs[node_id] = prior_node_outputs.get(node_id, [])
                    self.node_output_states[node_id] = prior_node_output_states.get(
                        node_id,
                        [],
                    )
                    self.node_execution_states[node_id] = prior_execution_states.get(
                        node_id,
                        EXECUTION_NOT_CALCULATED,
                    )
                    self.node_execution_messages[node_id] = (
                        prior_execution_messages.get(node_id, "")
                    )
                if self._mark_skipped_manual_node(
                    node_id,
                    dirty=self._has_cached_output(node_id),
                ):
                    completed.add(node_id)
                    self.completed_node_ids.add(node_id)

        while remaining:
            runnable = [
                node_id
                for node_id in remaining
                if all(source in completed for source in self._input_sources(node_id))
            ]
            if not runnable:
                break

            for node_id in runnable:
                if node_started_callback is not None:
                    node_started_callback(node_id)
                self.node_execution_states[node_id] = EXECUTION_RUNNING
                self.node_execution_messages[node_id] = ""
                try:
                    results = self._run_node(
                        node_id,
                        input_data,
                        input_metadata,
                        input_name,
                        source_payloads or {},
                        progress_callback,
                        cancel_callback,
                    )
                except Exception as exc:
                    self.set_node_execution_error(node_id, str(exc))
                    raise
                self.node_outputs[node_id] = [data for data, _ in results]
                self.node_output_states[node_id] = [state for _, state in results]
                primary_output, primary_state = results[0]
                self.outputs[node_id] = primary_output
                self.output_states[node_id] = primary_state
                self.node_execution_states[node_id] = (
                    EXECUTION_READY
                    if primary_output is not None
                    else EXECUTION_NOT_CALCULATED
                )
                self.node_execution_messages[node_id] = ""
                remaining.remove(node_id)
                completed.add(node_id)
                self.completed_node_ids.add(node_id)
                if prune_unretained:
                    self._prune_completed_outputs(
                        completed,
                        remaining,
                        retained_nodes,
                    )
        if prune_unretained:
            self.prune_cached_outputs(retained_nodes)
        return self.outputs

    def prune_cached_outputs(self, retain_node_ids: Iterable[str]) -> None:
        """Drop cached output data for nodes outside ``retain_node_ids``."""
        retained = {node_id for node_id in retain_node_ids if node_id in self.nodes}
        for node_id in list(self.nodes):
            if node_id not in retained:
                self._clear_cached_output(node_id)

    def _prune_completed_outputs(
        self,
        completed: set[str],
        remaining: set[str],
        retain_node_ids: set[str],
    ) -> None:
        needed_sources = {
            source_id
            for node_id in remaining
            for source_id in self._input_sources(node_id)
        }
        for node_id in list(completed):
            if node_id in retain_node_ids or node_id in needed_sources:
                continue
            self._clear_cached_output(node_id)
            completed.discard(node_id)

    def _clear_cached_output(self, node_id: str) -> None:
        if node_id not in self.nodes:
            return
        output_count = len(self.node_outputs.get(node_id, ()))
        state_count = len(self.node_output_states.get(node_id, ()))
        self.outputs[node_id] = None
        self.output_states[node_id] = None
        self.node_outputs[node_id] = [None] * output_count
        self.node_output_states[node_id] = [None] * state_count
        self.completed_node_ids.discard(node_id)

    def descendants_inclusive(self, node_ids: Iterable[str]) -> set[str]:
        targets = {node_id for node_id in node_ids if node_id in self.nodes}
        if not targets:
            return set()
        descendants = set(targets)
        changed = True
        while changed:
            changed = False
            for connection in self.connections:
                if (
                    connection.source_id in descendants
                    and connection.target_id not in descendants
                ):
                    descendants.add(connection.target_id)
                    changed = True
        return descendants

    def _validated_dirty_nodes(
        self,
        dirty_node_ids: Iterable[str] | None,
    ) -> set[str] | None:
        if dirty_node_ids is None:
            return None
        dirty_nodes = {node_id for node_id in dirty_node_ids if node_id in self.nodes}
        if not dirty_nodes:
            return None
        if set(self.outputs) != set(self.nodes):
            return None
        if set(self.output_states) != set(self.nodes):
            return None
        if set(self.node_outputs) != set(self.nodes):
            return None
        if set(self.node_output_states) != set(self.nodes):
            return None
        if set(self.node_execution_states) != set(self.nodes):
            return None
        if set(self.node_execution_messages) != set(self.nodes):
            return None
        cached_nodes = {
            node_id
            for node_id in self.completed_node_ids & set(self.nodes)
            if self._has_cached_output(node_id)
        }
        while True:
            nodes_to_run = self.descendants_inclusive(dirty_nodes)
            required_cached_sources = {
                source_id
                for node_id in nodes_to_run
                for source_id in self._input_sources(node_id)
                if source_id not in nodes_to_run
            }
            missing_upstream = required_cached_sources - cached_nodes
            if not missing_upstream:
                break
            dirty_nodes.update(missing_upstream)
        return dirty_nodes

    def _run_node(
        self,
        node_id: str,
        input_data,
        input_metadata: dict | None,
        input_name: str,
        source_payloads: dict[str, SourcePayload],
        progress_callback: Callable[[str, int, int, str], None] | None = None,
        cancel_callback: Callable[[], bool] | None = None,
    ) -> list[tuple[Any, ImageState | TableState | None]]:
        node = self.nodes[node_id]
        spec = self.operation_spec(node.operation_id)
        port_count = len(self.output_ports(node_id))
        if not spec.has_input:
            payload = source_payloads.get(node_id)
            if payload is None:
                payload = SourcePayload(input_data, input_metadata, input_name)
            state = payload.image_state
            if state is None or state.value_range == DEFERRED_VALUE_RANGE:
                state = image_state_from_array(
                    payload.data,
                    layer_metadata=payload.metadata,
                    source_name=payload.name,
                    axes=(state.axes if state is not None else None),
                    metadata_source=(
                        state.metadata_source if state is not None else None
                    ),
                    history=(state.history if state is not None else ()),
                    channels=(state.channels if state is not None else None),
                    acquisition=(state.acquisition if state is not None else None),
                    source=(state.source if state is not None else None),
                )
            state = with_channel_colors(state, node.params.get("channel_colors", ""))
            return [
                (
                    payload.data,
                    state,
                )
            ]

        connections = self._input_connections(node_id)
        if not connections:
            return [(None, None)] * port_count
        if self._node_accepts_multiple_inputs(node):
            required = self._required_inputs_for(node)
            input_connections = {
                connection.target_port: connection
                for connection in connections
            }
            if any(port not in input_connections for port in range(required)):
                return [(None, None)] * port_count
            ordered = [input_connections[port] for port in range(required)]
            source_outputs = [
                self._resolved_output(conn.source_id, conn.source_port)
                for conn in ordered
            ]
            if (
                any(output is None for output in source_outputs)
                or spec.function is None
            ):
                return [(None, None)] * port_count

            input_states = [
                self._resolved_output_state(conn.source_id, conn.source_port)
                for conn in ordered
            ]
            kwargs = self._operation_kwargs(node)
            self._inject_progress_context(
                spec,
                kwargs,
                node_id,
                progress_callback,
                cancel_callback,
            )
            if node.operation_id in SPATIAL_OPERATIONS:
                spatial_mode = kwargs.get("spatial_mode", "Auto from axes")
                resolved_spatial_ndim = _resolved_spatial_ndim(
                    input_states[0],
                    source_outputs[0],
                    spatial_mode,
                )
                kwargs["resolved_spatial_ndim"] = resolved_spatial_ndim
                node.params["resolved_spatial_ndim"] = resolved_spatial_ndim
            if node.operation_id == "combine_channels":
                derived_axis = _default_combined_channel_axis(
                    input_states[0],
                )
                kwargs["channel_axis"] = derived_axis
                node.params["channel_axis"] = derived_axis
            if node.operation_id in LABEL_METADATA_MULTI_INPUT_OPERATIONS:
                labels_state = input_states[0]
                spatial_mode = kwargs.get("spatial_mode", "Auto from axes")
                resolved_spatial_ndim = _resolved_spatial_ndim(
                    labels_state,
                    source_outputs[0],
                    spatial_mode,
                )
                kwargs["resolved_spatial_ndim"] = resolved_spatial_ndim
                node.params["resolved_spatial_ndim"] = resolved_spatial_ndim
                if isinstance(labels_state, ImageState):
                    kwargs["axis_names"] = tuple(
                        axis.name for axis in labels_state.axes
                    )
                    kwargs["axis_types"] = tuple(
                        axis.type for axis in labels_state.axes
                    )
                    kwargs["axis_scales"] = tuple(
                        axis.scale for axis in labels_state.axes
                    )
                    kwargs["axis_units"] = tuple(
                        axis.unit for axis in labels_state.axes
                    )
                    kwargs["source_name"] = labels_state.source_name
            if node.operation_id == "filter_labels_by_property":
                labels_state = input_states[0]
                spatial_mode = kwargs.get("spatial_mode", "Auto from axes")
                resolved_spatial_ndim = _resolved_spatial_ndim(
                    labels_state,
                    source_outputs[0],
                    spatial_mode,
                )
                kwargs["resolved_spatial_ndim"] = resolved_spatial_ndim
                node.params["resolved_spatial_ndim"] = resolved_spatial_ndim
                if isinstance(labels_state, ImageState):
                    kwargs["axis_names"] = tuple(
                        axis.name for axis in labels_state.axes
                    )
                    kwargs["axis_types"] = tuple(
                        axis.type for axis in labels_state.axes
                    )
            self._validate_multi_input_grids(node, input_states, kwargs)
            self._sync_colocalization_costes_thresholds(
                node,
                source_outputs,
                kwargs,
            )
            output = spec.function(source_outputs, **kwargs)
            if spec.output_type == "table":
                history = _table_history(input_states, node.title, output)
                state = table_state_from_data(
                    output,
                    history=history,
                    source_name=_combined_source_name(input_states),
                )
                return [(output, state)]
            state = transform_multi_input_image_state(
                output,
                input_states,
                operation_id=node.operation_id,
                operation_title=node.title,
                params=kwargs,
            )
            return [(output, state)]

        primary = connections[0]
        source_output = self._resolved_output(primary.source_id, primary.source_port)
        if source_output is None or spec.function is None:
            return [(None, None)] * port_count

        input_state = self._resolved_output_state(
            primary.source_id, primary.source_port
        )
        kwargs = self._operation_kwargs(node)
        self._inject_progress_context(
            spec,
            kwargs,
            node_id,
            progress_callback,
            cancel_callback,
        )
        if node.operation_id == "save_output":
            kwargs["image_state"] = input_state
        if node.operation_id in SPATIAL_OPERATIONS:
            spatial_mode = kwargs.get("spatial_mode", "Auto from axes")
            if node.operation_id == "clear_border_objects":
                spatial_mode = "Auto from axes"
            resolved_spatial_ndim = _resolved_spatial_ndim(
                input_state,
                source_output,
                spatial_mode,
            )
            kwargs["resolved_spatial_ndim"] = resolved_spatial_ndim
            node.params["resolved_spatial_ndim"] = resolved_spatial_ndim
        if node.operation_id in {
            "measure_objects",
            "analyze_skeleton",
            "measure_skeleton_branches",
            "skeleton_graph_tables",
            "measure_overall_skeleton_network",
            "measure_3d_mesh_morphology",
        } and isinstance(input_state, ImageState):
            kwargs["axis_names"] = tuple(axis.name for axis in input_state.axes)
            kwargs["axis_types"] = tuple(axis.type for axis in input_state.axes)
            kwargs["axis_scales"] = tuple(axis.scale for axis in input_state.axes)
            kwargs["axis_units"] = tuple(axis.unit for axis in input_state.axes)
            kwargs["source_name"] = input_state.source_name
        if node.operation_id == "prune_skeleton_branches" and isinstance(
            input_state,
            ImageState,
        ):
            kwargs["axis_scales"] = tuple(axis.scale for axis in input_state.axes)
            kwargs["axis_units"] = tuple(axis.unit for axis in input_state.axes)
        if node.operation_id == "reorder_axes" and isinstance(input_state, ImageState):
            kwargs["axis_names"] = tuple(axis.name for axis in input_state.axes)
        if (
            node.operation_id
            in {
                "project_image",
                "orthogonal_projection",
                "rescale_axes",
                "split_axis",
                "split_channels",
                "extract_channel",
            }
            and isinstance(input_state, ImageState)
        ):
            kwargs["axis_names"] = tuple(axis.name for axis in input_state.axes)
            kwargs["axis_types"] = tuple(axis.type for axis in input_state.axes)
            if node.operation_id == "orthogonal_projection":
                kwargs["axis_scales"] = tuple(axis.scale for axis in input_state.axes)
                kwargs["axis_units"] = tuple(axis.unit for axis in input_state.axes)
            if node.operation_id == "rescale_axes":
                kwargs["input_kind"] = input_state.kind
        if node.operation_id == "born_wolf_psf" and isinstance(
            input_state,
            ImageState,
        ):
            kwargs["axis_names"] = tuple(axis.name for axis in input_state.axes)
            kwargs["axis_types"] = tuple(axis.type for axis in input_state.axes)
            kwargs["axis_scales"] = tuple(axis.scale for axis in input_state.axes)
            kwargs["axis_units"] = tuple(axis.unit for axis in input_state.axes)
            kwargs["channel_emission_wavelengths"] = tuple(
                channel.emission_wavelength for channel in input_state.channels
            )
            kwargs["channel_emission_wavelength_units"] = tuple(
                channel.emission_wavelength_unit for channel in input_state.channels
            )
            kwargs["channel_excitation_wavelengths"] = tuple(
                channel.excitation_wavelength for channel in input_state.channels
            )
            kwargs["channel_excitation_wavelength_units"] = tuple(
                channel.excitation_wavelength_unit for channel in input_state.channels
            )
            kwargs["objective_lens_na"] = input_state.acquisition.objective_na
            kwargs["objective_refractive_index"] = (
                input_state.acquisition.refractive_index
            )
        if node.operation_id == "born_wolf_psf":
            self._sync_born_wolf_psf_resolution(node, source_output, kwargs)
        if node.operation_id == "composite_to_rgb" and isinstance(
            input_state,
            ImageState,
        ):
            try:
                requested_channel_axis = int(kwargs.get("channel_axis", -1))
            except Exception:
                requested_channel_axis = -1
            if requested_channel_axis < 0:
                channel_axis = _image_state_channel_axis(input_state)
                if channel_axis is not None:
                    kwargs["channel_axis"] = channel_axis
            try:
                resolved_channel_axis = int(kwargs.get("channel_axis", -1))
            except Exception:
                resolved_channel_axis = -1
            if 0 <= resolved_channel_axis < len(input_state.axes):
                kwargs["channel_axis_semantics"] = input_state.axes[
                    resolved_channel_axis
                ].name
            if input_state.channels:
                kwargs["channel_colors"] = tuple(
                    channel.color if channel.color is not None else ""
                    for channel in input_state.channels
                )
        output = spec.function(source_output, **kwargs)
        if spec.is_multi_output:
            return self._split_node_outputs(node, spec, output, input_state)
        if node.operation_id == "batch_output":
            return [(output, input_state)]
        if spec.output_type == "table":
            history = _table_history(input_state, node.title, output)
            state = table_state_from_data(
                output,
                history=history,
                source_name=getattr(input_state, "source_name", ""),
            )
            return [(output, state)]
        transform_params = self._public_params(node.params)
        if node.operation_id == "born_wolf_psf":
            transform_params = {
                **transform_params,
                **{
                    name: kwargs.get(name, transform_params.get(name))
                    for name in BORN_WOLF_PSF_AUTO_PARAMETERS
                },
                "channel": kwargs.get("channel", transform_params.get("channel")),
            }
        state = transform_image_state(
            output,
            input_state,
            operation_id=node.operation_id,
            operation_title=node.title,
            params=transform_params,
        )
        return [(output, state)]

    def _validate_multi_input_grids(
        self,
        node: GraphNode,
        input_states: list[ImageState | TableState | None],
        kwargs: dict[str, Any],
    ) -> None:
        """Enforce explicit physical-grid contracts before array operations."""
        if node.operation_id in PSF_SAMPLING_GRID_OPERATIONS:
            if (
                len(input_states) >= 2
                and isinstance(input_states[0], ImageState)
                and isinstance(input_states[1], ImageState)
            ):
                validate_psf_image_states(
                    input_states[0],
                    input_states[1],
                    spatial_ndim=int(kwargs["resolved_spatial_ndim"]),
                    operation_title=node.title,
                )
            return

        if node.operation_id not in SAME_SHAPE_GRID_OPERATIONS:
            return
        ports = self.input_ports(node.id)
        image_inputs = [
            (state, ports[index].label if index < len(ports) else f"Input {index + 1}")
            for index, state in enumerate(input_states)
            if isinstance(state, ImageState)
        ]
        if len(image_inputs) < 2:
            return

        # Some operations intentionally support a projected/broadcast mask or
        # RGB reduction.  Their operation-level shape checks remain authoritative;
        # this contract covers every group that claims the same sampled shape.
        shape_groups: dict[
            tuple[int, ...],
            list[tuple[ImageState, str]],
        ] = {}
        for state, label in image_inputs:
            shape_groups.setdefault(state.shape, []).append((state, label))
        for same_shape_inputs in shape_groups.values():
            validate_aligned_image_states(
                tuple(state for state, _label in same_shape_inputs),
                input_labels=tuple(label for _state, label in same_shape_inputs),
                operation_title=node.title,
            )

    def _inject_progress_context(
        self,
        spec: OperationSpec,
        kwargs: dict[str, Any],
        node_id: str,
        progress_callback: Callable[[str, int, int, str], None] | None,
        cancel_callback: Callable[[], bool] | None,
    ) -> None:
        if spec.function is None:
            return
        if "progress" not in inspect.signature(spec.function).parameters:
            return
        kwargs["progress"] = ProgressContext(
            cancelled=cancel_callback,
            reporter=(
                None
                if progress_callback is None
                else lambda update: progress_callback(
                    node_id,
                    update.current,
                    update.total,
                    update.message,
                )
            ),
        )

    def _split_node_outputs(
        self,
        node: GraphNode,
        spec: OperationSpec,
        outputs_seq: Any,
        input_state: ImageState | None,
    ) -> list[tuple[Any, ImageState | TableState | None]]:
        arrays = list(outputs_seq)
        if spec.output_factory is not None:
            ports: tuple[OutputSpec, ...] = spec.output_factory(len(arrays))
        else:
            ports = spec.output_ports
        if node.operation_id == "born_wolf_psf":
            ports = self._labeled_born_wolf_psf_ports(node.id, ports)
        ports = self._resolved_output_port_types(node.id, ports)
        results: list[tuple[Any, ImageState | None]] = []
        for index, port in enumerate(ports):
            data = arrays[index] if index < len(arrays) else None
            if data is None:
                results.append((None, None))
                continue
            if port.output_type == "table" or spec.output_type == "table":
                label = f"{node.title} ({port.label})"
                state = table_state_from_data(
                    data,
                    history=_table_history(input_state, label, data),
                    source_name=getattr(input_state, "source_name", ""),
                )
                results.append((data, state))
                continue
            params = self._public_params(node.params)
            if node.operation_id == "born_wolf_psf":
                params = self._born_wolf_output_transform_params(node, index)
            state = transform_split_output_state(
                data,
                input_state,
                operation_id=node.operation_id,
                operation_title=node.title,
                port_name=port.label,
                params=params,
            )
            results.append((data, state))
        return results

    def _born_wolf_output_transform_params(
        self,
        node: GraphNode,
        index: int,
    ) -> dict[str, Any]:
        params = self._public_params(node.params)
        channel_resolutions = node.params.get("_vipp_psf_channel_resolutions")
        if not isinstance(channel_resolutions, list):
            return params
        if not 0 <= index < len(channel_resolutions):
            return params
        item = channel_resolutions[index]
        if not isinstance(item, dict):
            return params
        values = item.get("values")
        if not isinstance(values, dict):
            return params
        return {
            **params,
            **{
                name: values[name]
                for name in (*BORN_WOLF_PSF_AUTO_PARAMETERS, "channel")
                if name in values
            },
        }

    def _input_sources(self, node_id: str) -> list[str]:
        return [
            connection.source_id
            for connection in self._input_connections(node_id)
        ]

    def _input_connections(self, node_id: str) -> list[GraphConnection]:
        return sorted(
            (
                connection
                for connection in self.connections
                if connection.target_id == node_id
            ),
            key=lambda connection: connection.target_port,
        )

    def input_port_count(self, node_id: str) -> int:
        node = self.nodes.get(node_id)
        if node is None or not node.has_input:
            return 0
        spec = self.operation_spec(node.operation_id)
        if spec.inputs:
            return len(spec.inputs)
        if self._node_accepts_multiple_inputs(node):
            return self._required_inputs_for(node)
        return 1

    def _node_accepts_multiple_inputs(self, node: GraphNode) -> bool:
        spec = self.operation_spec(node.operation_id)
        return bool(spec.inputs) or node.max_inputs is None or node.max_inputs != 1

    def _max_inputs_for(self, node: GraphNode) -> int | None:
        if node.max_inputs is None:
            return None
        return max(int(node.max_inputs), 1)

    def _required_inputs_for(self, node: GraphNode) -> int:
        spec = self.operation_spec(node.operation_id)
        if spec.inputs:
            return len(spec.inputs)
        if "input_count" in node.params:
            maximum = self._max_inputs_for(node)
            requested = max(int(node.params.get("input_count", 1)), 1)
            return min(requested, maximum) if maximum is not None else requested
        return 1

    def _target_port_for_connection(
        self,
        target: GraphNode,
        existing_targets: list[GraphConnection],
        requested_port: int | None,
        source_output_type: str,
    ) -> int | None:
        maximum = self._max_inputs_for(target)
        if requested_port is not None:
            port = int(requested_port)
            if port < 0:
                return None
            if maximum is not None and port >= maximum:
                return None
            if not self._types_compatible(
                source_output_type,
                self._input_type_for_port(target, port),
            ):
                return None
            return port

        used = {connection.target_port for connection in existing_targets}
        limit = maximum if maximum is not None else len(used) + 1
        for port in range(limit):
            if port not in used and self._types_compatible(
                source_output_type,
                self._input_type_for_port(target, port),
            ):
                return port
        return None

    def _input_type_for_port(self, node: GraphNode, port: int | None) -> str | None:
        spec = self.operation_spec(node.operation_id)
        port_index = 0 if port is None else int(port)
        if spec.inputs and 0 <= port_index < len(spec.inputs):
            return spec.inputs[port_index].input_type
        return node.input_type

    def _would_create_cycle(self, source_id: str, target_id: str) -> bool:
        downstream = {target_id}
        frontier = [target_id]
        while frontier:
            current = frontier.pop()
            for connection in self.connections:
                if connection.source_id != current:
                    continue
                if connection.target_id == source_id:
                    return True
                if connection.target_id not in downstream:
                    downstream.add(connection.target_id)
                    frontier.append(connection.target_id)
        return False

    def _types_compatible(self, output_type: str, input_type: str | None) -> bool:
        if input_type is None or input_type == "any" or output_type == "any":
            return True
        if input_type == "array":
            return output_type in {"array", "image", "mask", "labels"}
        if input_type == "mask_or_labels":
            return output_type in {"mask", "labels"}
        if input_type == "table":
            return output_type == "table"
        return output_type == input_type


def _table_history(input_states, operation_title: str, table) -> tuple[str, ...]:
    if isinstance(input_states, (list, tuple)):
        states = input_states
    else:
        states = (input_states,)
    prior = _combined_history(states)
    row_count = getattr(table, "row_count", 0)
    table_kind = str(getattr(table, "table_kind", "")).lower()
    if "graph node" in table_kind:
        noun = "node" if row_count == 1 else "nodes"
        action = "exported"
    elif "graph edge" in table_kind:
        noun = "edge" if row_count == 1 else "edges"
        action = "exported"
    elif "branch summary" in table_kind:
        noun = "group" if row_count == 1 else "groups"
        action = "summarized"
    elif "summary" in table_kind:
        if "skeleton" in table_kind or "network" in table_kind:
            noun = "block" if row_count == 1 else "blocks"
        else:
            noun = "group" if row_count == 1 else "groups"
        action = "summarized"
    elif "overall skeleton network" in table_kind:
        noun = "block" if row_count == 1 else "blocks"
        action = "measured"
    elif "branch" in table_kind:
        noun = "branch" if row_count == 1 else "branches"
        action = "measured"
    elif "skeleton" in table_kind:
        noun = "component" if row_count == 1 else "components"
        action = "analyzed"
    elif "column selection" in table_kind:
        column_count = getattr(table, "column_count", 0)
        noun = "column" if column_count == 1 else "columns"
        return prior + (f"{operation_title}: selected {column_count} {noun}",)
    elif "metadata" in table_kind:
        noun = "row" if row_count == 1 else "rows"
        action = "annotated"
    elif "merge" in table_kind:
        noun = "row" if row_count == 1 else "rows"
        action = "merged"
    else:
        noun = "object" if row_count == 1 else "objects"
        action = "measured"
    return prior + (f"{operation_title}: {action} {row_count} {noun}",)


def _combined_history(states) -> tuple[str, ...]:
    history: list[str] = []
    seen: set[str] = set()
    for state in states:
        for item in tuple(getattr(state, "history", ()) or ()):
            if item in seen:
                continue
            history.append(item)
            seen.add(item)
    return tuple(history)


def _combined_source_name(states) -> str:
    names: list[str] = []
    seen: set[str] = set()
    for state in states:
        name = str(getattr(state, "source_name", "") or "").strip()
        if not name or name in seen:
            continue
        names.append(name)
        seen.add(name)
    return ", ".join(names)


def grouped_palette_specs() -> dict[str, dict[str, list[OperationSpec]]]:
    groups: dict[str, dict[str, list[OperationSpec]]] = {}
    for spec in PALETTE_NODE_LIBRARY:
        subcategory = spec.subcategory or ""
        groups.setdefault(spec.category, {}).setdefault(subcategory, []).append(spec)
    return groups


def _node_id(operation_id: str, index: int) -> str:
    if operation_id == "gaussian_blur" and index == 1:
        return "gaussian"
    if operation_id == "otsu_threshold" and index == 1:
        return "threshold"
    return f"{operation_id}_{index}"


def _resolved_spatial_ndim(
    state: ImageState | None,
    data,
    spatial_mode,
) -> int:
    try:
        ndim = max(int(getattr(data, "ndim", 0)), 1)
    except Exception:
        ndim = 1
    mode = str(spatial_mode).strip().lower()
    if mode.startswith("2d"):
        requested = 2
    elif mode.startswith("3d"):
        requested = 3
    elif state is not None:
        spatial_count = sum(axis.type == "space" for axis in state.axes)
        if spatial_count >= 3:
            requested = 3
        elif spatial_count >= 2:
            requested = 2
        else:
            requested = 3 if ndim >= 3 else 2
    else:
        requested = 3 if ndim >= 3 else 2
    return min(requested, ndim)


def _clone_node(node: GraphNode) -> GraphNode:
    return GraphNode(
        node.id,
        node.operation_id,
        node.title,
        node.category,
        node.input_type,
        node.output_type,
        dict(node.params),
        node.max_inputs,
    )


def _clean_tunnel_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name or "").strip())


def _tunnel_key(name: str) -> str:
    return _clean_tunnel_name(name).casefold()


def _dynamic_output_count_hint(params: dict[str, Any]) -> int | None:
    value = params.get(DYNAMIC_OUTPUT_COUNT_PARAM)
    if isinstance(value, bool):
        return None
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return count if count >= 0 else None


def _default_combined_channel_axis(input_state: ImageState | None) -> int:
    if input_state is None:
        return 0
    for index, axis in enumerate(input_state.axes):
        if axis.type == "space":
            return index
    return 0


def _strict_channel_axis_for_shape(
    shape: tuple[int, ...],
    state: ImageState | TableState | None,
) -> int | None:
    if isinstance(state, ImageState):
        axis = _image_state_channel_axis(state)
        if axis is not None and 0 <= axis < len(shape):
            return axis
    if len(shape) >= 3 and shape[-1] in (3, 4):
        return len(shape) - 1
    return None


def _state_channel_count(
    state: ImageState | TableState | None,
    shape: tuple[int, ...] | None,
) -> int | None:
    if isinstance(state, ImageState) and state.channels:
        return len(state.channels)
    if shape:
        axis = _strict_channel_axis_for_shape(shape, state)
        if axis is not None and 0 <= axis < len(shape):
            return int(shape[axis])
    return None


def _born_wolf_psf_output_channel_indices(kwargs: dict[str, Any]) -> tuple[int, ...]:
    try:
        requested = int(kwargs.get("channel", -1))
    except Exception:
        requested = -1
    count = max(
        len(tuple(kwargs.get("channel_emission_wavelengths", ()) or ())),
        len(tuple(kwargs.get("channel_excitation_wavelengths", ()) or ())),
    )
    if bool(kwargs.get("auto_parameters", True)) and requested < 0 and count > 1:
        return tuple(range(count))
    return (requested,)


def _born_wolf_psf_channel_label(
    channels: tuple[Any, ...],
    index: int,
) -> str:
    if 0 <= index < len(channels):
        name = str(getattr(channels[index], "name", "") or "").strip()
        if name:
            return name
    return f"Channel {index + 1}"


def _split_axis_index_from_params(params: dict[str, Any], ndim: int) -> int:
    value = params.get("axis", "axis:0")
    text = str(value).strip().lower()
    if text.startswith("axis:"):
        text = text.split(":", 1)[1]
    try:
        axis = int(text)
    except ValueError:
        axis = 0
    if ndim <= 0:
        return 0
    if axis < 0:
        axis += ndim
    return max(0, min(axis, ndim - 1))


def _split_axis_label(
    state: ImageState | TableState | None,
    axis_index: int,
) -> str:
    if isinstance(state, ImageState) and 0 <= axis_index < len(state.axes):
        name = str(state.axes[axis_index].name or "").strip()
        if name:
            return name.upper() if len(name) == 1 else name
    return f"Axis {axis_index}"


def _image_state_channel_axis(input_state: ImageState) -> int | None:
    for index, axis in enumerate(input_state.axes):
        if axis.type == "channel" or axis.name.lower() == "c":
            return index
    return None


def connection_pairs(connections: Iterable[GraphConnection]) -> set[tuple[str, str]]:
    return {(connection.source_id, connection.target_id) for connection in connections}
