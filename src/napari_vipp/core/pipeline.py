"""Editable prototype graph model and executor."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from napari_vipp.core.metadata import (
    ImageState,
    image_state_from_array,
    transform_image_state,
    transform_multi_input_image_state,
)
from napari_vipp.core.operations import (
    adaptive_gaussian_threshold,
    adaptive_mean_threshold,
    add_images,
    average_blur,
    bilateral_filter,
    binary_threshold,
    black_hat,
    calculate_weighted_image,
    channel_composite,
    clip_intensity,
    closing,
    contrast_stretch,
    convert_dtype,
    crop_stack,
    dilate,
    erode,
    extract_channel,
    fill_holes,
    gamma_correction,
    gaussian_blur,
    gaussian_blur_3d,
    invert,
    logical_and,
    logical_or,
    logical_xor,
    mask_image,
    max_intensity_projection,
    median_filter,
    morphological_gradient,
    normalize_image,
    opening,
    otsu_threshold,
    ratio_image,
    rescale_intensity,
    rgb_composite,
    save_output,
    select_axis_slice,
    subtract_images,
    top_hat,
    triangle_threshold,
    volume_filter,
)


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

    @property
    def has_input(self) -> bool:
        return self.input_type is not None


@dataclass(frozen=True)
class SourcePayload:
    data: Any
    metadata: dict | None = None
    name: str = ""


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


@dataclass(frozen=True)
class ConnectionResult:
    success: bool
    message: str
    removed: tuple[GraphConnection, ...] = ()


IMAGE_DATA_CATEGORY = "Image Data"
SOURCE_OUTPUT_GROUP = "Source & Output"
AXES_REGIONS_GROUP = "Axes & Regions"
CHANNELS_COMPOSITES_GROUP = "Channels & Composites"
TYPE_SCALING_GROUP = "Type & Scaling"
MATH_LOGIC_GROUP = "Math & Logic"

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
        "array",
        "mask",
        (),
        fill_holes,
    ),
    OperationSpec(
        "volume_filter",
        "Volume Filter",
        "Morphology",
        "array",
        "mask",
        (
            ParameterSpec("min_volume", "Minimum volume", "int", 10, 1, 5000, 1),
        ),
        volume_filter,
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
        "channel_composite",
        "Channel Composite",
        IMAGE_DATA_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec("input_count", "Input channels", "int", 2, 2, 12, 1),
        ),
        channel_composite,
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
        "rgb_composite",
        "RGB Composite",
        IMAGE_DATA_CATEGORY,
        "array",
        "image",
        (
            ParameterSpec("channel_axis", "Channel axis", "int", 0, 0, 5, 1),
            ParameterSpec("red_channel", "Red", "int", 2, 0, 15, 1),
            ParameterSpec("green_channel", "Green", "int", 1, 0, 15, 1),
            ParameterSpec("blue_channel", "Blue", "int", 0, 0, 15, 1),
        ),
        rgb_composite,
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
                choices=("auto", "npy", "tiff"),
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
        self.output_states: dict[str, ImageState | None] = {}
        self._counters: Counter[str] = Counter()
        self.reset_starter_graph()

    def reset_starter_graph(self) -> None:
        self.nodes = {node.id: _clone_node(node) for node in PROTOTYPE_NODES}
        self.connections = list(PROTOTYPE_CONNECTIONS)
        self.outputs = {}
        self.output_states = {}
        self._counters = Counter()
        for node in self.nodes.values():
            self._counters[node.operation_id] += 1

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
        return node

    def remove_node(self, node_id: str) -> bool:
        if node_id not in self.nodes:
            return False
        del self.nodes[node_id]
        self.outputs.pop(node_id, None)
        self.output_states.pop(node_id, None)
        self.connections = [
            connection
            for connection in self.connections
            if connection.source_id != node_id and connection.target_id != node_id
        ]
        return True

    def connect(self, source_id: str, target_id: str) -> ConnectionResult:
        if source_id not in self.nodes or target_id not in self.nodes:
            return ConnectionResult(False, "Cannot connect missing nodes.")
        if source_id == target_id:
            return ConnectionResult(False, "Cannot connect a node to itself.")

        source = self.nodes[source_id]
        target = self.nodes[target_id]
        if not target.has_input:
            return ConnectionResult(False, "That node does not accept an input.")
        if not self._types_compatible(source.output_type, target.input_type):
            return ConnectionResult(
                False,
                (
                    f"Cannot connect {source.output_type} output to "
                    f"{target.input_type} input."
                ),
            )
        connection = GraphConnection(source_id, target_id)
        if connection in self.connections:
            return ConnectionResult(True, "Those nodes are already connected.")
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
            if maximum is not None and len(existing_targets) >= maximum:
                return ConnectionResult(
                    False,
                    f"That node already has {maximum} connected inputs.",
                )
        else:
            removed = tuple(existing_targets)
            self.connections = [
                existing
                for existing in self.connections
                if existing.target_id != target_id
            ]
        self.connections.append(connection)
        return ConnectionResult(True, "Connected nodes.", removed)

    def disconnect(self, source_id: str, target_id: str) -> bool:
        before = len(self.connections)
        self.connections = [
            connection
            for connection in self.connections
            if not (
                connection.source_id == source_id
                and connection.target_id == target_id
            )
        ]
        return len(self.connections) != before

    def set_param(self, node_id: str, name: str, value: Any) -> None:
        self.nodes[node_id].params[name] = value

    def operation_spec(self, operation_id: str) -> OperationSpec:
        return NODE_LIBRARY_BY_ID[operation_id]

    def node_parameter_specs(self, node_id: str) -> tuple[ParameterSpec, ...]:
        node = self.nodes[node_id]
        return self.operation_spec(node.operation_id).parameters

    def input_data_for_node(self, node_id: str):
        sources = self._input_sources(node_id)
        if not sources:
            return None
        return self.outputs.get(sources[0])

    def input_state_for_node(self, node_id: str) -> ImageState | None:
        sources = self._input_sources(node_id)
        if not sources:
            return None
        return self.output_states.get(sources[0])

    def run(
        self,
        input_data,
        input_metadata: dict | None = None,
        input_name: str = "",
        source_payloads: dict[str, SourcePayload] | None = None,
    ) -> dict[str, Any]:
        self.outputs = {node_id: None for node_id in self.nodes}
        self.output_states = {node_id: None for node_id in self.nodes}
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
                output, state = self._run_node(
                    node_id,
                    input_data,
                    input_metadata,
                    input_name,
                    source_payloads or {},
                )
                self.outputs[node_id] = output
                self.output_states[node_id] = state
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
    ):
        node = self.nodes[node_id]
        spec = self.operation_spec(node.operation_id)
        if not spec.has_input:
            payload = source_payloads.get(node_id)
            if payload is None:
                payload = SourcePayload(input_data, input_metadata, input_name)
            return payload.data, image_state_from_array(
                payload.data,
                layer_metadata=payload.metadata,
                source_name=payload.name,
            )

        sources = self._input_sources(node_id)
        if not sources:
            return None, None
        if self._node_accepts_multiple_inputs(node):
            required = self._required_inputs_for(node)
            if len(sources) < required:
                return None, None
            sources = sources[:required]
            source_outputs = [self.outputs.get(source) for source in sources]
            if (
                any(output is None for output in source_outputs)
                or spec.function is None
            ):
                return None, None

            input_states = [self.output_states.get(source) for source in sources]
            kwargs = dict(node.params)
            if node.operation_id == "channel_composite":
                kwargs["channel_axis"] = _default_combined_channel_axis(
                    input_states[0],
                )
            output = spec.function(source_outputs, **kwargs)
            return output, transform_multi_input_image_state(
                output,
                input_states,
                operation_id=node.operation_id,
                operation_title=node.title,
                params=kwargs,
            )

        source_output = self.outputs.get(sources[0])
        if source_output is None or spec.function is None:
            return None, None

        input_state = self.output_states.get(sources[0])
        kwargs = dict(node.params)
        if node.operation_id == "save_output":
            kwargs["image_state"] = input_state
        output = spec.function(source_output, **kwargs)
        return output, transform_image_state(
            output,
            input_state,
            operation_id=node.operation_id,
            operation_title=node.title,
            params=node.params,
        )

    def _input_sources(self, node_id: str) -> list[str]:
        return [
            connection.source_id
            for connection in self.connections
            if connection.target_id == node_id
        ]

    def _node_accepts_multiple_inputs(self, node: GraphNode) -> bool:
        return node.max_inputs is None or node.max_inputs != 1

    def _max_inputs_for(self, node: GraphNode) -> int | None:
        if node.max_inputs is None:
            return None
        return max(int(node.max_inputs), 1)

    def _required_inputs_for(self, node: GraphNode) -> int:
        if "input_count" in node.params:
            maximum = self._max_inputs_for(node)
            requested = max(int(node.params.get("input_count", 1)), 1)
            return min(requested, maximum) if maximum is not None else requested
        return 1

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
            return output_type in {"array", "image", "mask"}
        return output_type == input_type


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
