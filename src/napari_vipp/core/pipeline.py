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
)
from napari_vipp.core.operations import (
    adaptive_gaussian_threshold,
    adaptive_mean_threshold,
    average_blur,
    bilateral_filter,
    binary_threshold,
    black_hat,
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
    max_intensity_projection,
    median_filter,
    morphological_gradient,
    opening,
    otsu_threshold,
    select_axis_slice,
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

    @property
    def has_input(self) -> bool:
        return self.input_type is not None


@dataclass
class GraphNode:
    id: str
    operation_id: str
    title: str
    category: str
    input_type: str | None
    output_type: str
    params: dict[str, Any] = field(default_factory=dict)

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


NODE_LIBRARY: tuple[OperationSpec, ...] = (
    OperationSpec("input", "Input Layer", "Input", None, "image"),
    OperationSpec(
        "crop_stack",
        "Crop Stack",
        "Input",
        "array",
        "image",
        (
            ParameterSpec("top", "Top", "int", 0, 0, 256, 1),
            ParameterSpec("bottom", "Bottom", "int", 0, 0, 256, 1),
            ParameterSpec("left", "Left", "int", 0, 0, 256, 1),
            ParameterSpec("right", "Right", "int", 0, 0, 256, 1),
        ),
        crop_stack,
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
        "Input",
        "array",
        "any",
        (
            ParameterSpec("axis", "Axis", "int", 0, 0, 5, 1),
            ParameterSpec("index", "Index", "int", 0, 0, 1024, 1),
        ),
        select_axis_slice,
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
        "Channels",
        "array",
        "image",
        (
            ParameterSpec("channel", "Channel", "int", 0, 0, 5, 1),
        ),
        extract_channel,
    ),
    OperationSpec(
        "convert_dtype",
        "Convert Dtype",
        "Utility",
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
    ),
    OperationSpec("invert", "Invert", "Utility", "any", "any", (), invert),
)

NODE_LIBRARY_BY_ID = {spec.id: spec for spec in NODE_LIBRARY}
PALETTE_NODE_LIBRARY = tuple(spec for spec in NODE_LIBRARY if spec.id != "input")

PROTOTYPE_NODES = [
    GraphNode("input", "input", "Input Layer", "Input", None, "image"),
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
        )
        self.nodes[node.id] = node
        self.outputs[node.id] = None
        self.output_states[node.id] = None
        return node

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

        removed = tuple(
            existing
            for existing in self.connections
            if existing.target_id == target_id
        )
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
    ):
        node = self.nodes[node_id]
        spec = self.operation_spec(node.operation_id)
        if not spec.has_input:
            return input_data, image_state_from_array(
                input_data,
                layer_metadata=input_metadata,
                source_name=input_name,
            )

        sources = self._input_sources(node_id)
        if not sources:
            return None, None
        source_output = self.outputs.get(sources[0])
        if source_output is None or spec.function is None:
            return None, None

        output = spec.function(source_output, **node.params)
        return output, transform_image_state(
            output,
            self.output_states.get(sources[0]),
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


def grouped_palette_specs() -> dict[str, list[OperationSpec]]:
    groups: dict[str, list[OperationSpec]] = {}
    for spec in PALETTE_NODE_LIBRARY:
        groups.setdefault(spec.category, []).append(spec)
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
    )


def connection_pairs(connections: Iterable[GraphConnection]) -> set[tuple[str, str]]:
    return {(connection.source_id, connection.target_id) for connection in connections}
