"""Complete registry authoring metadata, without scientific or file execution.

Authorability follows the real operation registry minus explicit effect/source
boundaries. It means a node can be proposed and checked, not that its method is
suitable for a particular image or that execution/publication is authorized.
The lossless shared-parameter encoding keeps the complete library in a bounded
request without truncating parameter help or the user's discussion history.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from typing import Any

from napari_vipp.core.pipeline import (
    _OPTIONAL_PERSISTED_PARAMETER_SPECS,
    NODE_LIBRARY,
    NODE_LIBRARY_BY_ID,
    SAME_SHAPE_GRID_OPERATIONS,
    InputSpec,
    OperationSpec,
    OutputSpec,
    ParameterSpec,
    PrototypePipeline,
)

CATALOGUE_VERSION = 4
MAX_AUTHORING_PORTS = 256
AUTHORING_EXCLUSIONS = {
    "input": "Image Source requires an explicit user-selected file or layer binding.",
    "table_source": "A Table Source requires an explicit user-selected table binding.",
    "batch_output": (
        "Batch Output requires the separate batch collection and publication setup. "
        "Use the reviewed Save Image destination binding for individual outputs."
    ),
}
AUTHORABLE_OPERATION_IDS = tuple(
    spec.id for spec in NODE_LIBRARY if spec.id not in AUTHORING_EXCLUSIONS
)
MESH_OPERATION_IDS = frozenset(
    spec.id
    for spec in NODE_LIBRARY
    if "mesh" in spec.output_type
    or any("mesh" in port.input_type for port in spec.input_ports)
)
THREE_D_ONLY_OPERATION_IDS = frozenset(
    {"mask_to_3d_mesh", "labels_to_3d_mesh", "measure_3d_mesh_morphology"}
)
PROPERTY_COLUMNS = ("label_id", "area_pixels", "volume_voxels")

_LIMITATIONS = {
    "gaussian_blur": (
        "2D XY Gaussian smoothing, independently per Z plane; sigma is pixels, "
        "not physical units. Smoothing changes intensities and may merge objects. "
        "CPU integer inputs retain their dtype. Not isotropic 3D smoothing. "
        "Smoothing a Boolean mask deliberately makes a floating intensity image; "
        "threshold it explicitly to obtain a new mask. Never smooth label IDs."
    ),
    "gaussian_blur_3d": (
        "Explicit ZYX Gaussian smoothing; sigmas are axis-specific voxel units, "
        "not physical units. Equal sigmas need not mean physically isotropic "
        "smoothing on anisotropic grids. CPU integer inputs retain their dtype. "
        "Smoothing a Boolean mask deliberately makes a floating intensity image; "
        "threshold it explicitly to obtain a new mask. Never smooth label IDs."
    ),
    "otsu_threshold": (
        "Foreground is brighter than the estimated threshold. Stack histogram "
        "uses one threshold for the entire image/volume; slice histogram uses "
        "separate XY thresholds. Floating histograms use explicit bin counts. "
        "A starting point, not evidence of accurate segmentation; illumination "
        "gradients, weak objects and overlapping distributions may invalidate it."
    ),
    "label_connected_components": (
        "Labels connected mask components in the explicitly selected spatial "
        "rank. Touching objects are not separated. Connectivity is a scientific "
        "choice; full connectivity can merge diagonal contacts."
    ),
    "binary_threshold": (
        "Explicit native-intensity threshold or inclusive range, not normalized "
        "0–1 values. Threshold selection is a proposed scientific choice, not "
        "object identity evidence. Range limits must be ordered."
    ),
    "auto_watershed_from_mask": (
        "Distance-based h-maxima watershed separates touching foreground, not "
        "semantic structures. H is pixels/voxels; calibration anisotropy is not "
        "corrected. It may split one wavy structure into several objects. The "
        "chosen image_mode determines the elevation interpretation; do not "
        "substitute a different interpretation silently."
    ),
    "measure_objects": (
        "Measures connected label objects, not biological identities. Physical "
        "measurements depend on trusted image calibration; without calibration "
        "use pixel/voxel measures, not invented physical units. Manual execution "
        "is preserved: this proposal does not enable auto-recalculation."
    ),
    "review_images": (
        "Read-only comparison of connected images/masks/labels. Display changes "
        "do not change analysis pixels. A visual review cannot by itself prove "
        "segmentation correctness. Image B is optional. An explicit channel "
        "stack from Combine Channels may be reviewed, not treated as scalar."
    ),
    "filter_labels_by_volume": (
        "Keeps existing labels by exact pixel/voxel count, not physical volume "
        "or biological identity. Zero maximum means no upper limit. Cannot "
        "invent which label is the requested structure."
    ),
    "filter_labels_by_property": (
        "Selects existing labels using the connected Measure Objects table. "
        "A property column must exist and have the intended units and meaning. "
        "Selection by label_id requires observed/user-confirmed IDs, not guessed "
        "biological identities. Zero maximum means no upper bound."
    ),
    "mask_to_3d_mesh": (
        "Manual CPU full-resolution marching cubes on one explicit ZYX mask. "
        "No smoothing or hole filling. Leave open makes no assumption outside "
        "the volume. Close explicitly adds flat image-border caps; it does not "
        "recover missing anatomy. Uses carried calibration, never invented units."
    ),
    "labels_to_3d_mesh": (
        "Manual CPU mesh of each positive label in one explicit ZYX volume; "
        "preserves label identity and cavities. Close deliberately caps image "
        "borders, not reconstruction of missing anatomy. No smoothing, hole "
        "filling or identity inference."
    ),
    "combine_meshes": (
        "Collects two to sixteen meshes on their carried compatible coordinates; "
        "retains object lineage and colours, no welding, union or registration. "
        "All requested mesh inputs must be connected. Manual calculation."
    ),
    "color_mesh_objects": (
        "Colours by Object ID or freshly measured geometry. No biological "
        "class identity inference, geometry change or invented measurements. "
        "Unavailable measurements remain unavailable. Meshes are displayed "
        "in the main napari Surface view, not Review Images."
    ),
    "rescale_intensity": (
        "Explicit affine intensity rescaling and clipping, using exact whole-input "
        "percentiles or ordered native-value cutoffs; not a viewer contrast change. "
        "Output bounds remain ordered; invert_intensity deliberately reverses the "
        "mapping. A Boolean mask remains Boolean (inversion is logical NOT), not "
        "a continuous intensity rescale. Label IDs must not be rescaled."
    ),
    "subtract_background": (
        "Rolling-ball background subtraction changes intensities; radius is "
        "pixels/voxels, not physical length. Explicit 2D YX processes planes "
        "independently; explicit 3D ZYX uses a volume neighbourhood. Light "
        "background, smoothing and negative-value clipping are scientific choices. "
        "No inferred channel axis or invented background model accuracy."
    ),
    "non_local_means_filter": (
        "Denoises independently in each YX plane, not volumetric 3D. Odd patch_size "
        "and search distance are pixels. h uses a shared global normalized scale: "
        "full integer dtype span, 1 for floats in 0..1, otherwise finite whole-array "
        "min-to-max span. Restores input units. Can remove weak real structures."
    ),
    "logical_xor": (
        "Boolean exclusive OR over explicitly connected aligned masks; with more "
        "than two inputs it retains odd membership parity, not general subtraction "
        "or semantic separation. Overlap is removed for two masks; shared evidence "
        "may belong to a real structure. No registration or grid repair."
    ),
    "logical_or": (
        "Boolean union over explicitly connected aligned masks. Touching or "
        "overlapping structures lose independent identity in the result; use "
        "separate branches, labels or meshes to keep identities. No registration."
    ),
    "remove_binary_outliers": (
        "Binary circular-neighbourhood filtering independently in each YX plane, "
        "not a 3D operation. Radius is pixels. Foreground removal can erase thin "
        "real structures; Background fill deliberately adds foreground pixels, "
        "not evidence of a complete or accurately reconstructed object."
    ),
    "erode": (
        "Binary erosion independently in each YX plane using a square kernel, "
        "not a 3D ball or physical-distance operation. Kernel size is pixels. "
        "May erase thin structures or split objects. Nonzero array values become "
        "foreground; threshold intensities first when that is not intended. "
        "This operation does not preserve integer label identities."
    ),
    "dilate": (
        "Binary dilation independently in each YX plane using a square kernel, "
        "not a 3D ball or physical-distance operation. Kernel size is pixels. "
        "May merge structures or extend beyond measured signal. Nonzero array "
        "values become foreground; threshold first when that is not intended. "
        "This operation does not preserve integer label identities."
    ),
    "remove_small_objects": (
        "Filters masks by connected-component count or labels by each existing "
        "label's count in the explicit spatial rank. Size is pixels/voxels, not "
        "physical volume. Connectivity matters for masks. Preserves mask dtype "
        "or existing integer IDs, not semantic identity; can discard real objects."
    ),
    "measure_3d_mesh_morphology": (
        "Manual 3D-only measurement. Masks/labels reconstruct full-resolution "
        "surfaces using carried ZYX spacing; existing meshes are measured directly, "
        "not remeshed. minimum_voxel_count skips surface metrics for small voxel "
        "objects without removing their rows. Open/invalid mesh volumes may be "
        "unavailable. Numeric geometry is not evidence of correct segmentation; "
        "never report uncalibrated values as trusted physical measurements."
    ),
    "combine_channels": (
        "Stacks two to twelve same-shape, compatible-grid scalar inputs, adding "
        "an explicit channel axis immediately before the spatial axes. "
        "Does not label, merge, register or resample structures. Mixed dtypes use "
        "NumPy promotion. Output is a channel image, not a scalar mask or labels. "
        "Downstream operations must respect the carried channel-axis metadata."
    ),
    "intensity_histogram": (
        "Manual exact all-value histogram, not a viewer-slice sample or automatic "
        "threshold chooser. Explicit linear/log bin spacing and native-value "
        "custom range change interpretation; logarithmic bins exclude nonpositive "
        "values and report their count. Explicit channel stacks are counted per "
        "carried channel axis, not silently treated as a scalar spatial volume."
    ),
    "save_output": (
        "Save Image is authorable only with a separately approved local output "
        "destination binding. The model uses USER_OUTPUT_DIRECTORY, never a "
        "real path. No existing file overwrite or publication is authorized by "
        "a proposal. Local approval binds a manual writer; automatic parameter "
        "updates do not save. Explicit Run or Calculate can write its new result. "
        "Format support depends on the "
        "connected image/mesh and its axes/calibration. Add does not write files."
    ),
}


def _port_pipeline(
    operation_id: str,
    params: Mapping[str, Any] | None,
    *,
    pipeline: PrototypePipeline | None = None,
    node_id: str | None = None,
) -> tuple[PrototypePipeline, str]:
    """Resolve graph metadata without executing or changing a supplied graph."""
    spec = NODE_LIBRARY_BY_ID[operation_id]
    supplied = dict(params or {})
    if pipeline is not None or node_id is not None:
        if not isinstance(pipeline, PrototypePipeline) or node_id not in pipeline.nodes:
            raise ValueError("Bound port lookup needs an existing pipeline node.")
        node = pipeline.nodes[node_id]
        if node.operation_id != operation_id or any(
            node.params.get(key) != value for key, value in supplied.items()
        ):
            raise ValueError("Bound port lookup does not match the node's parameters.")
        lookup_params = node.params
    else:
        lookup_params = supplied
    count_spec = next(
        (parameter for parameter in spec.parameters if parameter.name == "input_count"),
        None,
    )
    if count_spec is not None:
        count = lookup_params.get("input_count", count_spec.default)
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not count_spec.minimum <= count <= count_spec.maximum
        ):
            raise ValueError(
                f"{operation_id} input_count is outside its registry bounds."
            )
    if pipeline is not None or node_id is not None:
        return pipeline, node_id
    pipeline = PrototypePipeline()
    node = pipeline.add_node(operation_id)
    node.params.update(supplied)
    return pipeline, node.id


def input_ports_for(
    operation_id: str,
    params: Mapping[str, Any] | None = None,
    *,
    pipeline: PrototypePipeline | None = None,
    node_id: str | None = None,
) -> tuple[InputSpec, ...]:
    """Return real parameter-resolved ports without clamping an invalid count."""
    pipeline, node_id = _port_pipeline(
        operation_id, params, pipeline=pipeline, node_id=node_id
    )
    return pipeline.input_ports(node_id)


def output_ports_for(
    operation_id: str,
    params: Mapping[str, Any] | None = None,
    *,
    pipeline: PrototypePipeline | None = None,
    node_id: str | None = None,
) -> tuple[OutputSpec, ...]:
    """Return actual ports, rejecting guessed metadata-dependent output counts."""
    bound = pipeline is not None
    pipeline, node_id = _port_pipeline(
        operation_id, params, pipeline=pipeline, node_id=node_id
    )
    spec = NODE_LIBRARY_BY_ID[operation_id]
    if spec.output_factory is not None:
        count = (
            pipeline._inferred_dynamic_output_count_for_node(node_id, spec)
            if bound
            else None
        )
        if count is None:
            raise ValueError(
                f"{operation_id} outputs require an explicitly bound "
                "source metadata contract."
            )
        if not 1 <= count <= MAX_AUTHORING_PORTS:
            raise ValueError(
                f"{operation_id} has {count} outputs; authoring supports at most "
                f"{MAX_AUTHORING_PORTS} ports, without truncation."
            )
    return pipeline.output_ports(node_id)


def required_input_ports_for(
    operation_id: str,
    params: Mapping[str, Any] | None = None,
    *,
    pipeline: PrototypePipeline | None = None,
    node_id: str | None = None,
) -> tuple[int, ...]:
    """Use the executor's required-input policy, including optional bindings."""
    pipeline, node_id = _port_pipeline(
        operation_id, params, pipeline=pipeline, node_id=node_id
    )
    node = pipeline.nodes[node_id]
    return tuple(range(pipeline._required_inputs_for(node))) if node.has_input else ()


def authorable_parameter_specs(operation_id: str) -> tuple[ParameterSpec, ...]:
    """Return required parameters from the actual operation definition."""
    return NODE_LIBRARY_BY_ID[operation_id].parameters


def optional_authorable_parameter_specs(
    operation_id: str,
) -> tuple[ParameterSpec, ...]:
    """Expose current optional controls, not derived state or legacy identity.

    The same persistence registry owns these controls. The hidden ImageJ
    method is a legacy node identity, not a current control; Combine Channels'
    channel axis is derived by the executor from carried metadata. Neither may
    be authored as an assistant override. Manual auto-recalculation and private
    derived hints are not supplied by this registry and remain locally owned.
    """
    return tuple(
        parameter
        for parameter in _OPTIONAL_PERSISTED_PARAMETER_SPECS.get(operation_id, ())
        if not parameter.name.startswith("_")
        and not (operation_id == "imagej_auto_threshold" and parameter.name == "method")
        and not (
            operation_id == "combine_channels" and parameter.name == "channel_axis"
        )
    )


def _parameter_entry(operation_id: str, parameter: ParameterSpec) -> dict[str, Any]:
    # JSON round-tripping normalizes tuple fields without changing their values.
    entry = json.loads(json.dumps(asdict(parameter)))
    entry["help"] = entry.pop("tooltip")
    if operation_id == "save_output" and parameter.name in {"enabled", "overwrite"}:
        # Only the approved local binding can supply an enabled writer. The
        # proposal never grants replacement of existing files.
        entry.update(
            choices=["on"] if parameter.name == "enabled" else ["no"],
            default="on" if parameter.name == "enabled" else "no",
            registry_choices=list(parameter.choices),
            registry_default=parameter.default,
        )
    if operation_id == "save_output" and parameter.name == "format":
        entry.update(
            choices=[choice for choice in parameter.choices if choice != "auto"],
            default="ome-tiff",
            registry_choices=list(parameter.choices),
            registry_default=parameter.default,
        )
    if operation_id == "save_output" and parameter.name == "path":
        entry["default"] = "USER_OUTPUT_DIRECTORY"
        entry["registry_default"] = parameter.default
        entry["binding"] = (
            "Explicit locally approved destination; never a real model path."
        )
    return entry


def _authoring_status(spec: OperationSpec) -> tuple[str, str]:
    if spec.id in AUTHORABLE_OPERATION_IDS:
        return (
            "supported",
            "Registry-authorable; method suitability remains reviewable.",
        )
    return (
        "requires_explicit_binding",
        AUTHORING_EXCLUSIONS[spec.id],
    )


OPTIONAL_INPUT_PORTS = {
    spec.id: frozenset(
        set(range(len(input_ports_for(spec.id))))
        - set(required_input_ports_for(spec.id))
    )
    for spec in NODE_LIBRARY
    if spec.has_input
    and len(input_ports_for(spec.id)) > len(required_input_ports_for(spec.id))
}


def _registry_entry(spec: OperationSpec) -> dict[str, Any]:
    status, reason = _authoring_status(spec)
    entry = {
        "operation_id": spec.id,
        "title": spec.title,
        "category": spec.category,
        "input_kinds": [port.input_type for port in spec.input_ports],
        "output_kinds": [port.output_type for port in spec.output_ports],
        "execution_policy": spec.execution_policy,
        "authoring_status": status,
        "reason": reason,
    }
    if any(parameter.name == "input_count" for parameter in spec.parameters):
        entry["input_port_policy"] = "parameter_dependent"
        entry["maximum_inputs"] = spec.max_inputs
    elif spec.id in {
        "estimate_registration",
        "compare_images",
        "find_peaks",
        "detect_spots_per_frame",
    }:
        entry["input_port_policy"] = "parameter_dependent"
    if spec.output_factory is not None:
        entry["output_port_policy"] = "bound_source_metadata_dependent"
    if spec.id in OPTIONAL_INPUT_PORTS:
        entry["optional_input_ports"] = sorted(OPTIONAL_INPUT_PORTS[spec.id])
    if spec.preserves_input_type:
        entry["output_kind_rule"] = "preserves_bound_primary_input_kind"
    if spec.id == "save_output":
        entry["requires_output_binding"] = True
    return entry


def _grid_effect(operation_id: str) -> str:
    if operation_id == "combine_channels":
        return "Insert leading C axis; preserve all compatible spatial grids."
    if operation_id in {"logical_or", "logical_xor"}:
        return "Preserve shape and grid; all input grids must be compatible."
    if operation_id == "combine_meshes":
        return "Retain mesh positions; convert compatible units, never register."
    if operation_id in {"mask_to_3d_mesh", "labels_to_3d_mesh"}:
        return (
            "Voxel boundaries to calibrated surfaces; the chosen border policy "
            "leaves borders open or adds explicit flat caps."
        )
    if operation_id == "color_mesh_objects":
        return "No geometry or coordinate change."
    if operation_id in {
        "measure_objects",
        "measure_3d_mesh_morphology",
        "intensity_histogram",
    }:
        return "Produce a table, not an image grid; retain calibration provenance."
    if operation_id == "review_images":
        return "Presentation only; no scientific output or grid changes."
    effects = {
        "crop_stack": "Crop selected voxel margins; shape and carried origin change.",
        "select_axis_slice": "Select indices/ranges; removed axes change rank.",
        "split_axis": "One output per selected-axis index; removes that axis.",
        "split_channels": "One output per carried channel; removes its channel axis.",
        "reorder_axes": "Explicitly permute axes; shape, axes and calibration follow.",
        "rescale_axes": "Explicit resampling changes sizes and grid spacing.",
        "set_pixel_size": "Replace physical calibration without pixel resampling.",
        "set_microscope_metadata": "Assign explicit, not measured, metadata.",
        "mip": "Project the selected axis; reduces rank and discards its coordinate.",
        "project_image": "Reduce chosen axes by the explicit projection statistic.",
        "orthogonal_projection": "Extract a spatial-plane projection and its axes.",
        "born_wolf_psf": "Generate PSF sampling grid from explicit optics/metadata.",
        "prepare_validate_psf": "Normalize, centre and check PSF sampling explicitly.",
        "estimate_registration": "Estimate a transform; does not apply registration.",
        "apply_transform": "Resample moving data onto the connected fixed grid.",
        "template_match": "Source-centred scores with explicit valid-score support.",
        "composite_to_rgb": "Map carried channels to encoded RGB; changes colour axes.",
        "colocalization_scatter_plot": "Scatter image, not an acquisition grid.",
        "masked_colocalization_scatter_plot": "Scatter image, not an acquisition grid.",
    }
    if operation_id in effects:
        return effects[operation_id]
    spec = NODE_LIBRARY_BY_ID[operation_id]
    if operation_id in SAME_SHAPE_GRID_OPERATIONS:
        return "All contributing inputs must have compatible shapes and physical grids."
    if spec.output_type in {"table", "plot", "transform"}:
        return "Structured result, not an image grid; retain calibration provenance."
    return (
        "Use the node's ordinary carried-axis/grid contract. Never infer alignment "
        "from array shape alone or silently resample/reorder/replace calibration."
    )


def _output_kind_rule(operation_id: str) -> str:
    if operation_id == "remove_small_objects":
        return "Boolean mask stays mask; integer labels retain IDs and label kind."
    if operation_id == "rescale_intensity":
        return "Intensity becomes image; Boolean mask stays mask. Labels disallowed."
    if operation_id in {"gaussian_blur", "gaussian_blur_3d"}:
        return (
            "Boolean mask becomes float32 intensity; numeric CPU input retains "
            "dtype. Label IDs disallowed."
        )
    if operation_id == "combine_channels":
        return (
            "Inputs become an explicitly channel-stacked image, never label identities."
        )
    if NODE_LIBRARY_BY_ID[operation_id].preserves_input_type:
        return (
            "Preserve the primary connected input kind through normal graph resolution."
        )
    return "Use declared output-port kinds."


def _input_variants(spec: OperationSpec) -> list[dict[str, Any]]:
    """Discover parameter-dependent port alternatives from the graph's policy.

    Only Boolean/choice parameters can change the finite mode-based policies;
    numeric input_count is represented separately as a bounded count contract.
    The scan changes one parameter at a time and retains an alternative only
    when the real input/required port signature changes. No kernel is called.
    """
    base_ports = input_ports_for(spec.id)
    base_required = required_input_ports_for(spec.id)
    baseline = (base_ports, base_required)
    variants = []
    for parameter in spec.parameters:
        values = (
            (False, True)
            if parameter.kind == "bool"
            else parameter.choices
            if parameter.kind == "choice"
            else ()
        )
        for value in values:
            if value == parameter.default:
                continue
            params = {parameter.name: value}
            ports = input_ports_for(spec.id, params)
            required = required_input_ports_for(spec.id, params)
            if (ports, required) == baseline:
                continue
            variants.append(
                {
                    "when": params,
                    "inputs": [
                        {"port": index, "name": port.label, "type": port.input_type}
                        for index, port in enumerate(ports)
                    ],
                    "required_input_ports": list(required),
                }
            )
    return variants


def build_catalogue() -> dict[str, Any]:
    """Return the complete registry and every authorable parameter contract."""
    operations = []
    for operation_id in AUTHORABLE_OPERATION_IDS:
        spec = NODE_LIBRARY_BY_ID[operation_id]
        count_spec = next(
            (
                parameter
                for parameter in spec.parameters
                if parameter.name == "input_count"
            ),
            None,
        )
        params = {}
        input_ports = []
        for index, port in enumerate(input_ports_for(operation_id, params)):
            item = {
                "port": index,
                "name": port.label,
                "type": port.input_type,
                "required": index not in OPTIONAL_INPUT_PORTS.get(operation_id, ()),
            }
            input_ports.append(item)
        outputs = (
            spec.output_factory(1)
            if spec.output_factory is not None
            else output_ports_for(operation_id)
        )
        entry = {
            "operation_id": spec.id,
            "title": spec.title,
            "category": spec.category,
            "inputs": input_ports,
            "outputs": [
                {"port": index, "name": port.name, "type": port.output_type}
                for index, port in enumerate(outputs)
            ],
            "parameters": [
                _parameter_entry(operation_id, parameter)
                for parameter in authorable_parameter_specs(operation_id)
            ],
            "optional_parameters": [
                _parameter_entry(operation_id, parameter)
                for parameter in optional_authorable_parameter_specs(operation_id)
            ],
            "execution_policy": spec.execution_policy,
            "authoring_status": "supported",
            "limitations": _LIMITATIONS.get(
                operation_id,
                spec.stack_processing_note
                or "Follow the operation's parameter and input metadata contracts. "
                "Graph validity is not evidence of scientific goal achievement.",
            ),
            "axis_contract": (
                "Requires a resolved 3D spatial input or existing 3D mesh."
                if operation_id in THREE_D_ONLY_OPERATION_IDS
                else "Resolve dimensional, channel and time semantics from connected "
                "metadata using this operation's ordinary executor contract."
            ),
            "required_input_ports": list(required_input_ports_for(operation_id)),
            "grid_effect": _grid_effect(operation_id),
            "output_kind_rule": _output_kind_rule(operation_id),
        }
        if spec.stack_processing_note:
            entry["stack_processing_note"] = spec.stack_processing_note
        if operation_id in OPTIONAL_INPUT_PORTS:
            entry["optional_input_ports"] = sorted(OPTIONAL_INPUT_PORTS[operation_id])
        if count_spec is not None:
            entry["input_count"] = {
                "parameter": "input_count",
                "minimum": count_spec.minimum,
                "maximum": count_spec.maximum,
                "type": spec.input_type,
            }
            entry["input_count_rule"] = (
                "Connect exactly ports 0 through input_count - 1; "
                "other ports are unavailable."
            )
        variants = _input_variants(spec)
        if variants:
            entry["input_variants"] = variants
        if spec.output_factory is not None:
            entry["dynamic_outputs"] = {
                "binding": "Connected source metadata determines the exact port count.",
                "rule": (
                    "Outputs above are a type template, not a one-port limit. "
                    "Ports are numbered from zero. Never invent a cached count."
                ),
            }
        if operation_id == "save_output":
            entry["requires_output_binding"] = True
            entry["proposal_execution_policy"] = "manual; auto-recalculation disabled"
        operations.append(entry)
    return {
        "kind": "napari-vipp-proposal-catalogue",
        "version": CATALOGUE_VERSION,
        "scope": (
            "Complete VIPP registry authoring except explicit source/batch bindings. "
            "All ordinary in-memory node families and locally bound Save Image "
            "proposals use normal graph/metadata validation. Authorability does "
            "not prove scientific suitability or authorize execution/publication."
        ),
        "coverage": {
            "registry_count": len(NODE_LIBRARY),
            "authorable_count": len(AUTHORABLE_OPERATION_IDS),
            "exclusions": deepcopy(AUTHORING_EXCLUSIONS),
        },
        "numeric_bounds_policy": (
            "minimum/maximum/step are registry UI bounds, not universal scientific "
            "limits. data_dependent_bounds marks values resolved from connected "
            "data. Use the shared typed parameter and metadata validators; never "
            "silently clamp values. Ordinary execution resource guards still apply."
        ),
        "operations": operations,
        "registry_index": [_registry_entry(spec) for spec in NODE_LIBRARY],
    }


_SHARED_ENCODING = "shared-parameters-v1"
_PARAMETER_COLUMNS = ("name", "label", "kind", "default", "minimum", "maximum", "step")
_PARAMETER_DEFAULTS = {
    "decimals": 0,
    "choices": [],
    "choice_labels": [],
    "dynamic_choices": False,
    "dynamic_choice_kind": "",
    "help": "",
    "visibility": "always",
    "visibility_parameter": "",
    "visibility_values": [],
    "visibility_ports": [],
    "slider_minimum": None,
    "slider_maximum": None,
    "data_dependent_bounds": False,
    "odd_only": False,
}
_OPERATION_COLUMNS = (
    "operation_id",
    "title",
    "category",
    "inputs",
    "outputs",
    "parameters",
)
_REGISTRY_COLUMNS = ("operation_id", "title", "category", "input_kinds", "output_kinds")
_PORT_COLUMNS = ("port", "name", "type")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _shared_rows(
    entries: list[dict[str, Any]], columns: tuple[str, ...]
) -> tuple[dict[str, Any], list[list[Any]]]:
    """Factor identical, universally present metadata without dropping fields."""
    common_keys = (
        set.intersection(*(set(entry) for entry in entries)) if entries else set()
    )
    defaults = {}
    for key in sorted(common_keys - set(columns)):
        groups: dict[str, list[Any]] = {}
        for entry in entries:
            token = _canonical(entry[key])
            group = groups.setdefault(token, [entry[key], 0])
            group[1] += 1
        value, count = max(groups.values(), key=lambda group: group[1])
        if count > 1:
            defaults[key] = deepcopy(value)
    rows = []
    for entry in entries:
        overrides = {
            key: deepcopy(value)
            for key, value in entry.items()
            if key not in columns and (key not in defaults or value != defaults[key])
        }
        rows.append([*(deepcopy(entry[key]) for key in columns), overrides])
    return defaults, rows


def pack_catalogue(catalogue: Mapping[str, Any]) -> dict[str, Any]:
    """Losslessly encode the full catalogue for bounded model transmission.

    Each table names its columns. The last cell of every row is a dictionary
    overriding that table's shared defaults. Operation parameter integers index
    the shared parameter pool; optional parameters use the same pool. No help,
    parameter constraint, library entry or source-specific choice is omitted.
    """
    if catalogue.get("encoding") is not None:
        raise ValueError("Pack an expanded catalogue, not an already encoded one.")
    expanded = deepcopy(dict(catalogue))
    operations = expanded.pop("operations")
    registry = expanded.pop("registry_index")
    pool: list[list[Any]] = []
    indices = {}
    port_pool = []
    port_indices = {}

    def port_refs(ports):
        refs = []
        for port in ports:
            token = _canonical(port)
            if token not in port_indices:
                port_indices[token] = len(port_pool)
                port_pool.append(
                    [
                        *(deepcopy(port[key]) for key in _PORT_COLUMNS),
                        {
                            key: deepcopy(value)
                            for key, value in port.items()
                            if key not in _PORT_COLUMNS
                        },
                    ]
                )
            refs.append(port_indices[token])
        return refs

    for operation in operations:
        operation["inputs"] = port_refs(operation["inputs"])
        operation["outputs"] = port_refs(operation["outputs"])
        for variant in operation.get("input_variants", ()):
            variant["inputs"] = port_refs(variant["inputs"])
        for field in ("parameters", "optional_parameters"):
            refs = []
            for parameter in operation[field]:
                token = _canonical(parameter)
                if token not in indices:
                    indices[token] = len(pool)
                    extras = {
                        key: deepcopy(value)
                        for key, value in parameter.items()
                        if key not in _PARAMETER_COLUMNS
                        and (
                            key not in _PARAMETER_DEFAULTS
                            or value != _PARAMETER_DEFAULTS[key]
                        )
                    }
                    pool.append(
                        [
                            *(deepcopy(parameter[key]) for key in _PARAMETER_COLUMNS),
                            extras,
                        ]
                    )
                refs.append(indices[token])
            operation[field] = refs
    operation_defaults, operation_rows = _shared_rows(operations, _OPERATION_COLUMNS)
    registry_defaults, registry_rows = _shared_rows(registry, _REGISTRY_COLUMNS)
    packed = {
        **expanded,
        "encoding": _SHARED_ENCODING,
        "encoding_help": (
            "Tables are lossless: each row follows its named columns; its final "
            "dictionary overrides that table's shared defaults. Parameters and "
            "optional_parameters contain zero-based parameter_pool indices. "
            "Input/output lists contain port_pool indices. Repeated strings "
            "may be encoded as a one-key {$text: index} object into text_pool. "
            "Expand text, defaults and pool references before interpreting rows."
        ),
        "port_columns": list(_PORT_COLUMNS),
        "port_pool": port_pool,
        "parameter_columns": list(_PARAMETER_COLUMNS),
        "parameter_defaults": deepcopy(_PARAMETER_DEFAULTS),
        "parameter_pool": pool,
        "operation_columns": list(_OPERATION_COLUMNS),
        "operation_defaults": operation_defaults,
        "operations": operation_rows,
        "registry_columns": list(_REGISTRY_COLUMNS),
        "registry_defaults": registry_defaults,
        "registry_index": registry_rows,
    }
    return _pack_repeated_text(packed)


def _pack_repeated_text(value: dict[str, Any]) -> dict[str, Any]:
    counts: Counter[str] = Counter()

    def count(item):
        if isinstance(item, str):
            counts[item] += 1
        elif isinstance(item, list):
            for child in item:
                count(child)
        elif isinstance(item, dict):
            for child in item.values():
                count(child)

    count(value)
    # Store a string only when its repeated encoded bytes outweigh the index
    # markers and pool entry. The encoding preserves every word verbatim.
    texts = sorted(
        text
        for text, count_ in counts.items()
        if count_ > 1
        and (count_ - 1) * len(_canonical(text).encode("utf-8")) > count_ * 16 + 8
    )
    indices = {text: index for index, text in enumerate(texts)}

    def replace(item):
        if isinstance(item, str) and item in indices:
            return {"$text": indices[item]}
        if isinstance(item, list):
            return [replace(child) for child in item]
        if isinstance(item, dict):
            return {key: replace(child) for key, child in item.items()}
        return item

    result = replace(value)
    result["text_pool"] = texts
    return result


def _unpack_repeated_text(value: Mapping[str, Any]) -> dict[str, Any]:
    texts = value.get("text_pool")
    if not isinstance(texts, list) or any(type(text) is not str for text in texts):
        raise ValueError("Invalid catalogue text pool.")

    def replace(item):
        if isinstance(item, dict):
            if set(item) == {"$text"}:
                index = item["$text"]
                if type(index) is not int or not 0 <= index < len(texts):
                    raise ValueError("Invalid catalogue text reference.")
                return texts[index]
            return {key: replace(child) for key, child in item.items()}
        if isinstance(item, list):
            return [replace(child) for child in item]
        return item

    return replace({key: item for key, item in value.items() if key != "text_pool"})


def _expand_rows(
    rows: Any, columns: tuple[str, ...], defaults: Any
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not isinstance(defaults, dict):
        raise ValueError("Encoded catalogue tables need rows and shared defaults.")
    result = []
    for row in rows:
        if (
            not isinstance(row, list)
            or len(row) != len(columns) + 1
            or not isinstance(row[-1], dict)
            or set(row[-1]) & set(columns)
        ):
            raise ValueError("Invalid encoded catalogue row.")
        result.append(
            {
                **deepcopy(defaults),
                **dict(zip(columns, deepcopy(row[:-1]), strict=True)),
                **deepcopy(row[-1]),
            }
        )
    return result


def unpack_catalogue(catalogue: Mapping[str, Any]) -> dict[str, Any]:
    """Expand a packed local catalogue; reject malformed tables or references."""
    if catalogue.get("encoding") != _SHARED_ENCODING:
        raise ValueError("Unsupported catalogue encoding.")
    result = _unpack_repeated_text(catalogue)
    for key, expected in (
        ("parameter_columns", _PARAMETER_COLUMNS),
        ("operation_columns", _OPERATION_COLUMNS),
        ("registry_columns", _REGISTRY_COLUMNS),
        ("port_columns", _PORT_COLUMNS),
    ):
        if result.pop(key, None) != list(expected):
            raise ValueError("Unsupported catalogue columns.")
    pool = _expand_rows(
        result.pop("parameter_pool"),
        _PARAMETER_COLUMNS,
        result.pop("parameter_defaults"),
    )
    ports = _expand_rows(result.pop("port_pool"), _PORT_COLUMNS, {})
    operations = _expand_rows(
        result["operations"], _OPERATION_COLUMNS, result.pop("operation_defaults")
    )
    for operation in operations:
        for field in ("inputs", "outputs"):
            operation[field] = _expand_refs(operation[field], ports, "port")
        for variant in operation.get("input_variants", ()):
            variant["inputs"] = _expand_refs(variant["inputs"], ports, "port")
        for field in ("parameters", "optional_parameters"):
            operation[field] = _expand_refs(operation[field], pool, "parameter")
    result["operations"] = operations
    result["registry_index"] = _expand_rows(
        result["registry_index"], _REGISTRY_COLUMNS, result.pop("registry_defaults")
    )
    del result["encoding"]
    result.pop("encoding_help", None)
    return result


def _expand_refs(
    refs: Any, pool: list[dict[str, Any]], label: str
) -> list[dict[str, Any]]:
    if not isinstance(refs, list) or any(
        type(index) is not int or not 0 <= index < len(pool) for index in refs
    ):
        raise ValueError(f"Invalid catalogue {label} reference.")
    return [deepcopy(pool[index]) for index in refs]


# A readable alias for callers displaying the fully expanded local catalogue.
expand_catalogue = unpack_catalogue


__all__ = [
    "AUTHORING_EXCLUSIONS",
    "AUTHORABLE_OPERATION_IDS",
    "CATALOGUE_VERSION",
    "MAX_AUTHORING_PORTS",
    "MESH_OPERATION_IDS",
    "OPTIONAL_INPUT_PORTS",
    "PROPERTY_COLUMNS",
    "THREE_D_ONLY_OPERATION_IDS",
    "authorable_parameter_specs",
    "build_catalogue",
    "expand_catalogue",
    "input_ports_for",
    "optional_authorable_parameter_specs",
    "output_ports_for",
    "pack_catalogue",
    "required_input_ports_for",
    "unpack_catalogue",
]
