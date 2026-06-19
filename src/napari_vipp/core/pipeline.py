"""Editable prototype graph model and executor."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Any

from napari_vipp.core.metadata import (
    ImageState,
    image_state_from_array,
    transform_image_state,
    transform_multi_input_image_state,
    transform_split_output_state,
)
from napari_vipp.core.operations import (
    adaptive_gaussian_threshold,
    adaptive_mean_threshold,
    add_images,
    add_metadata_columns,
    analyze_skeleton,
    average_blur,
    bilateral_filter,
    binary_threshold,
    black_hat,
    calculate_weighted_image,
    clear_border_objects,
    clip_intensity,
    closing,
    combine_channels,
    composite_to_rgb,
    contrast_stretch,
    convert_dtype,
    crop_stack,
    dilate,
    erode,
    extract_channel,
    fill_holes,
    filter_labels_by_volume,
    gamma_correction,
    gaussian_blur,
    gaussian_blur_3d,
    invert,
    label_connected_components,
    logical_and,
    logical_or,
    logical_xor,
    mask_image,
    max_intensity_projection,
    measure_objects,
    measure_objects_with_intensity,
    median_filter,
    merge_tables,
    morphological_gradient,
    normalize_image,
    opening,
    otsu_threshold,
    ratio_image,
    relabel_sequential,
    remove_small_objects,
    rescale_intensity,
    save_output,
    select_axis_slice,
    skeletonize_mask,
    split_channels,
    subtract_images,
    top_hat,
    triangle_threshold,
)
from napari_vipp.core.tables import TableState, table_state_from_data


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    label: str
    kind: str
    default: float | int | str
    minimum: float | int
    maximum: float | int
    step: float | int
    decimals: int = 0
    choices: tuple[str, ...] = ()


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


@dataclass(frozen=True)
class ConnectionResult:
    success: bool
    message: str
    removed: tuple[GraphConnection, ...] = ()
    connection: GraphConnection | None = None


IMAGE_DATA_CATEGORY = "Image Data"
SOURCE_OUTPUT_GROUP = "Source & Output"
AXES_REGIONS_GROUP = "Axes & Regions"
CHANNELS_COMPOSITES_GROUP = "Channels & Composites"
TYPE_SCALING_GROUP = "Type & Scaling"
MATH_LOGIC_GROUP = "Math & Logic"
LABEL_OPERATIONS_CATEGORY = "Label Operations"
MEASUREMENTS_CATEGORY = "Measurements"

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
)

SPATIAL_OPERATIONS = {
    "clear_border_objects",
    "fill_holes",
    "label_connected_components",
    "filter_labels_by_volume",
    "analyze_skeleton",
    "measure_objects",
    "relabel_sequential",
    "remove_small_objects",
    "skeletonize",
}


DEFAULT_DYNAMIC_OUTPUT_PORTS = 3


def _split_channels_outputs(count: int) -> tuple[OutputSpec, ...]:
    """Build the output ports for a Split Channels node.

    ``count`` is the number of channels discovered when the node last ran; the
    pipeline supplies the default port count for a node that has not yet
    processed an image.
    """
    count = max(int(count), 1)
    return tuple(
        OutputSpec(f"channel_{index + 1}", "image", f"Ch {index + 1}")
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
        "contrast_stretch",
        "Contrast Stretching",
        "Contrast",
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
        contrast_stretch,
    ),
    OperationSpec(
        "gamma_correction",
        "Gamma Correction",
        "Contrast",
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
        "Filtering",
        "array",
        "image",
        (
            ParameterSpec("size", "Kernel size", "int", 5, 1, 51, 1),
        ),
        average_blur,
    ),
    OperationSpec(
        "gaussian_blur",
        "Gaussian Blur",
        "Filtering",
        "array",
        "image",
        (
            ParameterSpec("sigma", "Sigma", "float", 1.2, 0.0, 12.0, 0.1, 2),
        ),
        gaussian_blur,
    ),
    OperationSpec(
        "gaussian_blur_3d",
        "Gaussian Blur 3D",
        "Filtering",
        "array",
        "image",
        (
            ParameterSpec("sigma_z", "Sigma Z", "float", 2.0, 0.0, 12.0, 0.1, 2),
            ParameterSpec("sigma_y", "Sigma Y", "float", 2.0, 0.0, 12.0, 0.1, 2),
            ParameterSpec("sigma_x", "Sigma X", "float", 2.0, 0.0, 12.0, 0.1, 2),
        ),
        gaussian_blur_3d,
    ),
    OperationSpec(
        "median_filter",
        "Median Filter",
        "Filtering",
        "array",
        "image",
        (
            ParameterSpec("size", "Kernel size", "int", 5, 1, 51, 2),
        ),
        median_filter,
    ),
    OperationSpec(
        "bilateral_filter",
        "Bilateral Filtering",
        "Filtering",
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
        "otsu_threshold",
        "Otsu Threshold",
        "Segmentation",
        "array",
        "mask",
        (),
        otsu_threshold,
    ),
    OperationSpec(
        "triangle_threshold",
        "Triangle Threshold",
        "Segmentation",
        "array",
        "mask",
        (),
        triangle_threshold,
    ),
    OperationSpec(
        "binary_threshold",
        "Binary Threshold",
        "Segmentation",
        "array",
        "mask",
        (
            ParameterSpec("threshold", "Threshold", "float", 0.5, 0.0, 1.0, 0.01, 3),
        ),
        binary_threshold,
    ),
    OperationSpec(
        "adaptive_mean_threshold",
        "Adaptive Mean Threshold",
        "Segmentation",
        "array",
        "mask",
        (
            ParameterSpec("block_size", "Block size", "int", 11, 3, 101, 2),
            ParameterSpec("c", "C", "float", 2.0, -50.0, 50.0, 0.1, 2),
        ),
        adaptive_mean_threshold,
    ),
    OperationSpec(
        "adaptive_gaussian_threshold",
        "Adaptive Gaussian Threshold",
        "Segmentation",
        "array",
        "mask",
        (
            ParameterSpec("block_size", "Block size", "int", 11, 3, 101, 2),
            ParameterSpec("c", "C", "float", 2.0, -50.0, 50.0, 0.1, 2),
        ),
        adaptive_gaussian_threshold,
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
    ),
    OperationSpec(
        "opening",
        "Opening",
        "Morphology",
        "array",
        "mask",
        (ParameterSpec("size", "Kernel size", "int", 2, 1, 101, 1),),
        opening,
    ),
    OperationSpec(
        "closing",
        "Closing",
        "Morphology",
        "array",
        "mask",
        (ParameterSpec("size", "Kernel size", "int", 2, 1, 101, 1),),
        closing,
    ),
    OperationSpec(
        "top_hat",
        "Top Hat",
        "Morphology",
        "array",
        "mask",
        (ParameterSpec("size", "Kernel size", "int", 2, 1, 101, 1),),
        top_hat,
    ),
    OperationSpec(
        "black_hat",
        "Black Hat",
        "Morphology",
        "array",
        "mask",
        (ParameterSpec("size", "Kernel size", "int", 2, 1, 101, 1),),
        black_hat,
    ),
    OperationSpec(
        "morphological_gradient",
        "Morphological Gradient",
        "Morphology",
        "array",
        "mask",
        (ParameterSpec("size", "Kernel size", "int", 2, 1, 101, 1),),
        morphological_gradient,
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
                "measurement_set",
                "Measurement set",
                "choice",
                "Basic morphology",
                0,
                0,
                1,
                choices=("Basic morphology",),
            ),
        ),
        measure_objects,
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
                "measurement_set",
                "Measurement set",
                "choice",
                "Basic morphology + intensity",
                0,
                0,
                1,
                choices=("Basic morphology + intensity",),
            ),
        ),
        measure_objects_with_intensity,
        max_inputs=2,
        inputs=(
            InputSpec("labels", "labels", "Labels"),
            InputSpec("intensity", "image", "Intensity image"),
        ),
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
        "array",
        "image",
        (
            ParameterSpec("input_count", "Inputs", "int", 2, 2, 2, 1),
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
        (),
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
        "rescale_intensity",
        "Rescale Intensity",
        IMAGE_DATA_CATEGORY,
        "array",
        "image",
        (
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
        subcategory=TYPE_SCALING_GROUP,
    ),
    OperationSpec(
        "normalize_image",
        "Normalize",
        IMAGE_DATA_CATEGORY,
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
        subcategory=TYPE_SCALING_GROUP,
    ),
    OperationSpec(
        "clip_intensity",
        "Clip",
        IMAGE_DATA_CATEGORY,
        "array",
        "image",
        (
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
        subcategory=TYPE_SCALING_GROUP,
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
        subcategory=TYPE_SCALING_GROUP,
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
PALETTE_NODE_LIBRARY = NODE_LIBRARY

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
        self._counters: Counter[str] = Counter()
        self.reset_starter_graph()

    def reset_starter_graph(self) -> None:
        self.nodes = {node.id: _clone_node(node) for node in PROTOTYPE_NODES}
        self.connections = list(PROTOTYPE_CONNECTIONS)
        self.outputs = {}
        self.output_states = {}
        self.node_outputs = {}
        self.node_output_states = {}
        self._counters = Counter()
        for node in self.nodes.values():
            self._counters[node.operation_id] += 1

    def restore_graph(
        self,
        nodes: Iterable[GraphNode],
        connections: Iterable[GraphConnection],
    ) -> None:
        """Replace the current graph with deserialized nodes and connections."""
        node_list = list(nodes)
        self.nodes = {node.id: _clone_node(node) for node in node_list}
        if len(self.nodes) != len(node_list):
            raise ValueError("Cannot restore a graph with duplicate node ids.")
        valid = set(self.nodes)
        self.connections = list(connections)
        self.outputs = {node_id: None for node_id in self.nodes}
        self.output_states = {node_id: None for node_id in self.nodes}
        self.node_outputs = {node_id: [] for node_id in self.nodes}
        self.node_output_states = {node_id: [] for node_id in self.nodes}
        for connection in self.connections:
            if (
                connection.source_id not in valid
                or connection.target_id not in valid
            ):
                raise ValueError(
                    "Cannot restore a connection that references a missing node: "
                    f"{connection.source_id!r} -> {connection.target_id!r}."
                )
            self._validate_restored_connection(connection)
        self._counters = Counter()
        for node in self.nodes.values():
            self._counters[node.operation_id] += 1

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
        return node

    def remove_node(self, node_id: str) -> bool:
        if node_id not in self.nodes:
            return False
        del self.nodes[node_id]
        self.outputs.pop(node_id, None)
        self.output_states.pop(node_id, None)
        self.node_outputs.pop(node_id, None)
        self.node_output_states.pop(node_id, None)
        self.connections = [
            connection
            for connection in self.connections
            if connection.source_id != node_id and connection.target_id != node_id
        ]
        return True

    def connect(
        self,
        source_id: str,
        target_id: str,
        target_port: int | None = None,
        source_port: int = 0,
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
        connection = GraphConnection(source_id, target_id, port, source_port)
        if connection in self.connections:
            return ConnectionResult(
                True,
                "Those nodes are already connected.",
                connection=connection,
            )
        self.connections.append(connection)
        return ConnectionResult(
            True,
            "Connected nodes.",
            removed,
            connection,
        )

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
        if removed:
            self.connections = [
                connection
                for connection in self.connections
                if connection not in removed
            ]
        return removed

    def set_param(self, node_id: str, name: str, value: Any) -> None:
        self.nodes[node_id].params[name] = value

    def _validate_restored_connection(self, connection: GraphConnection) -> None:
        target = self.nodes[connection.target_id]
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

    def operation_spec(self, operation_id: str) -> OperationSpec:
        return NODE_LIBRARY_BY_ID[operation_id]

    def output_ports(self, node_id: str) -> tuple[OutputSpec, ...]:
        node = self.nodes.get(node_id)
        if node is None:
            return ()
        spec = self.operation_spec(node.operation_id)
        if spec.output_factory is not None:
            count = len(self.node_outputs.get(node_id, ()))
            if count <= 0:
                count = DEFAULT_DYNAMIC_OUTPUT_PORTS
            ports = spec.output_factory(count)
        else:
            ports = spec.output_ports
        return self._resolved_output_port_types(node_id, ports)

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

    def _resolved_output(self, source_id: str, source_port: int):
        outputs = self.node_outputs.get(source_id)
        if outputs:
            if 0 <= source_port < len(outputs):
                return outputs[source_port]
            return outputs[0]
        return self.outputs.get(source_id)

    def _resolved_output_state(
        self, source_id: str, source_port: int
    ) -> ImageState | TableState | None:
        states = self.node_output_states.get(source_id)
        if states:
            if 0 <= source_port < len(states):
                return states[source_port]
            return states[0]
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

    def input_state_for_node(self, node_id: str) -> ImageState | None:
        connections = self._input_connections(node_id)
        if not connections:
            return None
        primary = connections[0]
        return self._resolved_output_state(primary.source_id, primary.source_port)

    def run(
        self,
        input_data,
        input_metadata: dict | None = None,
        input_name: str = "",
        source_payloads: dict[str, SourcePayload] | None = None,
    ) -> dict[str, Any]:
        self.outputs = {node_id: None for node_id in self.nodes}
        self.output_states = {node_id: None for node_id in self.nodes}
        self.node_outputs = {node_id: [] for node_id in self.nodes}
        self.node_output_states = {node_id: [] for node_id in self.nodes}
        remaining = set(self.nodes)
        completed: set[str] = set()

        while remaining:
            runnable = [
                node_id
                for node_id in remaining
                if all(source in completed for source in self._input_sources(node_id))
            ]
            if not runnable:
                break

            for node_id in runnable:
                results = self._run_node(
                    node_id,
                    input_data,
                    input_metadata,
                    input_name,
                    source_payloads or {},
                )
                self.node_outputs[node_id] = [data for data, _ in results]
                self.node_output_states[node_id] = [state for _, state in results]
                primary_output, primary_state = results[0]
                self.outputs[node_id] = primary_output
                self.output_states[node_id] = primary_state
                remaining.remove(node_id)
                completed.add(node_id)
        return self.outputs

    def _run_node(
        self,
        node_id: str,
        input_data,
        input_metadata: dict | None,
        input_name: str,
        source_payloads: dict[str, SourcePayload],
    ) -> list[tuple[Any, ImageState | None]]:
        node = self.nodes[node_id]
        spec = self.operation_spec(node.operation_id)
        port_count = len(self.output_ports(node_id))
        if not spec.has_input:
            payload = source_payloads.get(node_id)
            if payload is None:
                payload = SourcePayload(input_data, input_metadata, input_name)
            return [
                (
                    payload.data,
                    payload.image_state
                    or image_state_from_array(
                        payload.data,
                        layer_metadata=payload.metadata,
                        source_name=payload.name,
                    ),
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
            kwargs = dict(node.params)
            if node.operation_id == "combine_channels":
                derived_axis = _default_combined_channel_axis(
                    input_states[0],
                )
                kwargs["channel_axis"] = derived_axis
                node.params["channel_axis"] = derived_axis
            if node.operation_id == "measure_objects_intensity":
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
        kwargs = dict(node.params)
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
        if node.operation_id in {"measure_objects", "analyze_skeleton"} and isinstance(
            input_state,
            ImageState,
        ):
            kwargs["axis_names"] = tuple(axis.name for axis in input_state.axes)
            kwargs["axis_types"] = tuple(axis.type for axis in input_state.axes)
            kwargs["axis_scales"] = tuple(axis.scale for axis in input_state.axes)
            kwargs["axis_units"] = tuple(axis.unit for axis in input_state.axes)
            kwargs["source_name"] = input_state.source_name
        output = spec.function(source_output, **kwargs)
        if spec.is_multi_output:
            return self._split_node_outputs(node, spec, output, input_state)
        if spec.output_type == "table":
            history = _table_history(input_state, node.title, output)
            state = table_state_from_data(
                output,
                history=history,
                source_name=getattr(input_state, "source_name", ""),
            )
            return [(output, state)]
        state = transform_image_state(
            output,
            input_state,
            operation_id=node.operation_id,
            operation_title=node.title,
            params=node.params,
        )
        return [(output, state)]

    def _split_node_outputs(
        self,
        node: GraphNode,
        spec: OperationSpec,
        outputs_seq: Any,
        input_state: ImageState | None,
    ) -> list[tuple[Any, ImageState | None]]:
        arrays = list(outputs_seq)
        if spec.output_factory is not None:
            ports: tuple[OutputSpec, ...] = spec.output_factory(len(arrays))
        else:
            ports = spec.output_ports
        ports = self._resolved_output_port_types(node.id, ports)
        results: list[tuple[Any, ImageState | None]] = []
        for index, port in enumerate(ports):
            data = arrays[index] if index < len(arrays) else None
            if data is None:
                results.append((None, None))
                continue
            state = transform_split_output_state(
                data,
                input_state,
                operation_id=node.operation_id,
                operation_title=node.title,
                port_name=port.label,
                params=node.params,
            )
            results.append((data, state))
        return results

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
    if "skeleton" in table_kind:
        noun = "component" if row_count == 1 else "components"
        action = "analyzed"
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


def _default_combined_channel_axis(input_state: ImageState | None) -> int:
    if input_state is None:
        return 0
    for index, axis in enumerate(input_state.axes):
        if axis.type == "space":
            return index
    return 0


def connection_pairs(connections: Iterable[GraphConnection]) -> set[tuple[str, str]]:
    return {(connection.source_id, connection.target_id) for connection in connections}
