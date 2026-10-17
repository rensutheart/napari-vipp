"""Full registry authoring contracts, without model or scientific execution.

These are structural/capability checks on synthetic metadata. They do not
establish numerical correctness, method suitability, segmentation quality or
an ability to reconstruct information absent from an image.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from napari_vipp.core.metadata import (
    AcquisitionMetadata,
    AxisMetadata,
    ChannelMetadata,
    image_state_from_array,
)
from napari_vipp.core.node_execution import CPUNodeExecutor
from napari_vipp.core.pipeline import (
    MANUAL_AUTO_RECALCULATE_PARAM,
    NODE_LIBRARY_BY_ID,
    PrototypePipeline,
)
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow
from napari_vipp.core.workflow_proposal_catalogue import (
    output_ports_for,
    pack_catalogue,
    unpack_catalogue,
)
from napari_vipp.core.workflow_proposal_outputs import (
    OUTPUT_DIRECTORY_PLACEHOLDER,
    OutputDestination,
)
from napari_vipp.core.workflow_proposals import (
    MAX_DIALOGUE_HISTORY_BYTES,
    MAX_DIALOGUE_PAIRS,
    MAX_REQUEST_BYTES,
    ProposalDialogue,
    ProposalError,
    build_catalogue,
    capture_proposal_context,
    validate_proposal,
)

_BINDING_OPERATIONS = {"input", "table_source", "save_output", "batch_output"}
_TIME_OPERATIONS = {"detect_spots_per_frame", "build_tracks"}


def full_source(*, volume=True, axes=None, dtype=np.float32, kind=None):
    axis_names = axes or ("zyx" if volume else "yx")
    sizes = {"t": 3, "c": 2, "z": 10, "y": 20, "x": 24}
    shape = tuple(sizes[name] for name in axis_names)
    data = (
        np.arange(np.prod(shape), dtype=np.float32)
        .reshape(shape)
        .astype(dtype, copy=True)
    )
    data.flags.writeable = False
    state = image_state_from_array(
        data,
        axes=tuple(
            AxisMetadata(
                name,
                "time" if name == "t" else "channel" if name == "c" else "space",
                "s" if name == "t" else "" if name == "c" else "um",
                scale=1.0 if name == "c" else 0.5,
            )
            for name in axis_names
        ),
    )
    if kind is not None:
        state = replace(state, kind=kind)
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    pipeline.outputs["input"] = data
    pipeline.node_outputs["input"] = [data]
    pipeline.output_states["input"] = state
    pipeline.node_output_states["input"] = [state]
    pipeline.completed_node_ids.add("input")
    return pipeline


def full_node(key, operation_id, **overrides):
    params = {
        parameter.name: deepcopy(parameter.default)
        for parameter in NODE_LIBRARY_BY_ID[operation_id].parameters
    }
    # Make choices explicit without restricting the normal node's other modes.
    if "spatial_mode" in params:
        params["spatial_mode"] = "3D ZYX"
    if "boundary" in params:
        params["boundary"] = "Leave open at image border"
    if "property_column" in params:
        params["property_column"] = "label_id"
    params.update(overrides)
    return {
        "key": key,
        "operation_id": operation_id,
        "params": params,
        "reason": "Synthetic metadata-only capability example; not executed.",
    }


def full_response(context, nodes, links):
    return {
        "kind": "napari-vipp-workflow-proposal",
        "version": 1,
        "context_id": context.context_id,
        "summary": "An unexecuted candidate composed from ordinary VIPP nodes.",
        "assumptions": [
            "Numerical results and scientific suitability remain unassessed."
        ],
        "questions": [],
        "nodes": nodes,
        "connections": [
            {
                "source": link[0],
                "source_port": link[3] if len(link) == 4 else 0,
                "target": link[1],
                "target_port": link[2],
            }
            for link in links
        ],
    }


def ordinary_operation_response(context, operation_id, **overrides):
    """Bind one representative leaf through typed, reachable normal nodes."""
    nodes = []
    links = []
    plane_only = operation_id.startswith("cellprofiler_")
    bindings = {"image": "input", "array": "input", "any": "input"}

    def add(key, operation, inputs, **params):
        if plane_only and any(
            parameter.name == "spatial_mode"
            for parameter in NODE_LIBRARY_BY_ID[operation].parameters
        ):
            params.setdefault("spatial_mode", "2D YX")
        nodes.append(full_node(key, operation, **params))
        links.extend(
            (source, key, port, source_port)
            for port, (source, source_port) in enumerate(inputs)
        )
        return key

    base_source = "input"
    if plane_only:
        base_source = add(
            "plane",
            "select_axis_slice",
            [("input", 0)],
            axis=0,
            index=0,
            range_mode=False,
        )
        bindings.update(image=base_source, array=base_source, any=base_source)

    def bind(kind):
        if kind in bindings:
            return bindings[kind]
        if kind == "mask":
            result = add("mask", "binary_threshold", [(base_source, 0)])
        elif kind in {"labels", "mask_or_labels"}:
            result = add("labels", "label_connected_components", [(bind("mask"), 0)])
            bindings["labels"] = result
            bindings["mask_or_labels"] = result
        elif kind in {"mesh", "mask_or_labels_or_mesh"}:
            result = add("mesh", "labels_to_3d_mesh", [(bind("labels"), 0)])
            bindings["mesh"] = result
            bindings["mask_or_labels_or_mesh"] = result
        elif kind == "table":
            result = add("objects", "measure_objects", [(bind("labels"), 0)])
        elif kind == "transform":
            result = add(
                "registration", "estimate_registration", [("input", 0), ("input", 0)]
            )
        else:
            raise AssertionError(f"Missing test binding for registry input kind {kind}")
        bindings[kind] = result
        return result

    def channels(count=3):
        if "channels" not in bindings:
            bindings["channels"] = add(
                "channels",
                "combine_channels",
                [("input", 0)] * count,
                input_count=count,
            )
        return bindings["channels"]

    spec = NODE_LIBRARY_BY_ID[operation_id]
    params = {}
    if operation_id == "reorder_axes":
        params["order"] = "zyx"
    elif operation_id == "project_image":
        params["axes"] = "name:z"
    elif operation_id == "split_axis":
        params["axis"] = "axis:0"
    elif operation_id == "born_wolf_psf":
        params.update(
            auto_parameters=False,
            wavelength_nm=510.0,
            numerical_aperture=1.2,
            refractive_index=1.33,
            pixel_size_xy_um=0.5,
            z_step_um=0.5,
            xy_size=9,
            z_size=5,
        )
    elif operation_id == "assign_channel_colors":
        params["channel_colors"] = "Red,Green,Blue"
    elif operation_id == "composite_to_rgb":
        params.update(channel_axis=0, red_channel=0, green_channel=1, blue_channel=2)
    params.update(overrides)
    if operation_id in {
        "split_channels",
        "extract_channel",
        "assign_channel_colors",
        "composite_to_rgb",
    }:
        inputs = [(channels(), 0)]
    elif operation_id in {
        "richardson_lucy_deconvolution",
        "richardson_lucy_tv_deconvolution",
    }:
        psf = add(
            "psf",
            "born_wolf_psf",
            [("input", 0)],
            auto_parameters=False,
            wavelength_nm=510.0,
            numerical_aperture=1.2,
            refractive_index=1.33,
            pixel_size_xy_um=0.5,
            z_step_um=0.5,
            xy_size=9,
            z_size=5,
        )
        inputs = [("input", 0), (psf, 0)]
    elif operation_id == "filter_labels_by_property":
        inputs = [(bind("labels"), 0), (bind("table"), 0)]
    else:
        ports = list(spec.input_ports)
        if operation_id in {"compare_images", "find_peaks"}:
            ports = ports[
                :-1
            ]  # use_mask=False, rather than inventing a mask requirement.
        if not spec.inputs and any(p.name == "input_count" for p in spec.parameters):
            ports *= int(
                next(p.default for p in spec.parameters if p.name == "input_count")
            )
        inputs = [(bind(port.input_type), 0) for port in ports]
    add("candidate", operation_id, inputs, **params)
    return full_response(context, nodes, links)


_ORDINARY_OPERATIONS = sorted(
    set(NODE_LIBRARY_BY_ID) - _BINDING_OPERATIONS - _TIME_OPERATIONS
)


@pytest.fixture
def no_execution(monkeypatch):
    monkeypatch.setattr(
        PrototypePipeline,
        "run",
        lambda *_a, **_k: pytest.fail(
            "Proposal checks must never run scientific kernels"
        ),
    )
    monkeypatch.setattr(
        CPUNodeExecutor,
        "execute",
        lambda *_a, **_k: pytest.fail("Metadata checks must not dispatch a CPU kernel"),
    )


@pytest.mark.parametrize("operation_id", _ORDINARY_OPERATIONS)
def test_ordinary_registry_family_has_a_saveable_metadata_only_candidate(
    operation_id, no_execution
):
    pipeline = full_source()
    original = GraphSnapshot.from_pipeline(pipeline)
    data = pipeline.outputs["input"]
    context = capture_proposal_context(pipeline, "input")
    raw = ordinary_operation_response(context, operation_id)
    prepared = validate_proposal(raw, context)
    assert prepared.can_apply
    candidate = prepared.candidate.to_pipeline()
    restored = deserialize_workflow(serialize_workflow(candidate))
    replay = PrototypePipeline()
    replay.restore_graph(
        restored["nodes"], restored["connections"], restored["output_tunnels"]
    )
    assert serialize_workflow(replay) == serialize_workflow(candidate)
    for node_id in prepared.added_node_ids:
        node = candidate.nodes[node_id]
        if NODE_LIBRARY_BY_ID[node.operation_id].execution_policy == "manual":
            assert node.params[MANUAL_AUTO_RECALCULATE_PARAM] is False
        assert candidate.outputs[node_id] is None
    assert GraphSnapshot.from_pipeline(pipeline) == original
    assert pipeline.outputs["input"] is data
    assert not data.flags.writeable


def test_full_registry_discovery_and_authoring_do_not_mean_source_binding_permission():
    catalogue = build_catalogue()
    detailed = {entry["operation_id"] for entry in catalogue["operations"]}
    assert (
        set(NODE_LIBRARY_BY_ID) - {"input", "table_source", "batch_output"} <= detailed
    )
    assert "input" not in detailed
    assert "table_source" not in detailed
    assert "batch_output" not in detailed
    assert {entry["operation_id"] for entry in catalogue["registry_index"]} == set(
        NODE_LIBRARY_BY_ID
    )


def test_broad_catalogue_preserves_normal_static_choices_and_parameter_limits():
    catalogue = build_catalogue()
    for entry in catalogue["operations"]:
        spec = NODE_LIBRARY_BY_ID[entry["operation_id"]]
        by_name = {parameter["name"]: parameter for parameter in entry["parameters"]}
        for parameter in spec.parameters:
            declared = by_name[parameter.name]
            if spec.id == "save_output" and parameter.name in {
                "enabled",
                "path",
                "format",
                "overwrite",
            }:
                continue  # Explicit user-bound/no-overwrite publication contract.
            if parameter.choices:
                assert declared["choices"] == list(parameter.choices), (
                    spec.id,
                    parameter.name,
                )
            if parameter.kind in {"int", "float"}:
                assert declared["minimum"] == parameter.minimum, (
                    spec.id,
                    parameter.name,
                )
                assert declared["maximum"] == parameter.maximum, (
                    spec.id,
                    parameter.name,
                )


def test_full_catalogue_request_encoding_preserves_every_expanded_contract():
    expanded = build_catalogue()
    packed = pack_catalogue(expanded)
    restored = unpack_catalogue(packed)
    assert restored == expanded
    assert len(restored["operations"]) == 143
    assert len(restored["registry_index"]) == len(NODE_LIBRARY_BY_ID)
    restored["operations"][0]["title"] = "A changed detached copy"
    assert unpack_catalogue(packed) == expanded


@pytest.mark.parametrize(
    "operation_id,axes,expected_count,params",
    [
        ("split_axis", "zyx", 10, {"axis": "axis:0"}),
        ("split_channels", "czyx", 2, {}),
        ("born_wolf_psf", "czyx", 2, {}),
    ],
)
def test_dynamic_port_helpers_require_real_bound_metadata_not_a_fallback_count(
    operation_id, axes, expected_count, params, no_execution
):
    with pytest.raises(ValueError):
        output_ports_for(operation_id, params)
    pipeline = full_source(axes=axes)
    node = pipeline.add_node(operation_id)
    node.params.update(params)
    assert pipeline.connect("input", node.id).success
    ports = output_ports_for(
        operation_id, node.params, pipeline=pipeline, node_id=node.id
    )
    assert len(ports) == expected_count


def test_full_library_and_maximum_discussion_fit_without_losing_history(no_execution):
    pipeline = full_source()
    context = capture_proposal_context(pipeline, "input")
    pairs = tuple(
        (f"Q{index:02d}" + "q" * 1021, "a" * 1024)
        for index in range(MAX_DIALOGUE_PAIRS)
    )
    assert (
        sum(len(text.encode()) for pair in pairs for text in pair)
        == MAX_DIALOGUE_HISTORY_BYTES
    )
    dialogue = ProposalDialogue("g" * 4000, pairs)
    request = context.public_request(dialogue)
    encoded = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()
    assert len(encoded) <= MAX_REQUEST_BYTES
    assert request["objective"] == dialogue.objective
    assert request["clarifications"] == [
        {"question": question, "answer": answer} for question, answer in pairs
    ]
    catalogue = unpack_catalogue(request["catalogue"])
    assert len(catalogue["operations"]) == 143
    assert len(catalogue["registry_index"]) == len(NODE_LIBRARY_BY_ID)


@pytest.mark.parametrize(
    "dtype,kind", [(np.bool_, "binary mask"), (np.uint16, "label image")]
)
def test_cached_mask_and_label_source_kinds_are_retained_not_reinterpreted(dtype, kind):
    pipeline = full_source(dtype=dtype, kind=kind)
    context = capture_proposal_context(pipeline, "input")
    assert context.source_summary["kind"] == kind
    assert context.source_summary["dtype"] == np.dtype(dtype).name
    assert not pipeline.outputs["input"].flags.writeable


@pytest.mark.parametrize(
    "operation_id,overrides",
    [
        ("mask_to_3d_mesh", {"boundary": "Close at image border"}),
        ("color_mesh_objects", {"color_by": "mesh_volume_physical"}),
        ("color_mesh_objects", {"color_by": "sphericity", "color_map": "plasma"}),
        ("auto_watershed_from_mask", {"image_mode": "Elevation image"}),
    ],
)
def test_normal_explicit_choices_are_not_blocked_by_the_old_narrow_catalogue(
    operation_id, overrides, no_execution
):
    pipeline = full_source()
    context = capture_proposal_context(pipeline, "input")
    raw = ordinary_operation_response(context, operation_id, **overrides)
    assert validate_proposal(raw, context).can_apply


@pytest.mark.parametrize(
    "operation_id", ["input", "table_source", "batch_output", "external_script"]
)
def test_broad_catalogue_cannot_create_unbound_sources_batch_or_external_code(
    operation_id, no_execution
):
    pipeline = full_source()
    context = capture_proposal_context(pipeline, "input")
    raw = full_response(
        context,
        [
            {
                "key": "unbound",
                "operation_id": operation_id,
                "params": {},
                "reason": "Not authorized",
            }
        ],
        [("input", "unbound", 0)],
    )
    with pytest.raises(ProposalError):
        validate_proposal(raw, context)


def _save_response(context, *, questions=()):
    raw = full_response(
        context,
        [
            full_node(
                "save",
                "save_output",
                enabled="on",
                path=OUTPUT_DIRECTORY_PLACEHOLDER,
                format="npy",
                overwrite="no",
            )
        ],
        [("input", "save", 0)],
    )
    raw["questions"] = list(questions)
    return raw


def test_unbound_save_is_a_question_not_authority_to_choose_a_path(no_execution):
    pipeline = full_source()
    context = capture_proposal_context(pipeline, "input")
    raw = _save_response(context, questions=["Where should the results be saved?"])
    prepared = validate_proposal(raw, context)
    assert not prepared.can_apply
    assert prepared.candidate.nodes[-1].params["path"] == OUTPUT_DIRECTORY_PLACEHOLDER
    raw["questions"] = []
    with pytest.raises(ProposalError, match="folder|destination"):
        validate_proposal(raw, context)


@pytest.mark.parametrize("volume", [False, True])
def test_save_destination_is_bound_privately_without_writing_or_sharing_its_path(
    volume, tmp_path, no_execution
):
    pipeline = full_source(volume=volume)
    destination = OutputDestination.from_directory(tmp_path)
    context = capture_proposal_context(
        pipeline, "input", output_destination=destination
    )
    raw = _save_response(context)
    prepared = validate_proposal(raw, context)
    assert prepared.can_apply
    params = prepared.candidate.nodes[-1].params
    assert params["path"] == str(tmp_path / "vipp-save.npy")
    assert params["overwrite"] == "no"
    assert raw["nodes"][0]["params"]["path"] == OUTPUT_DIRECTORY_PLACEHOLDER
    assert tuple(tmp_path.iterdir()) == ()
    public = json.dumps(context.public_request("Save the proposed result"))
    assert str(tmp_path) not in public
    assert tmp_path.as_posix() not in public
    prepared.assert_current(pipeline)


def test_gaussian_volume_filter_respects_normal_graph_zyx_axis_requirement(
    no_execution,
):
    pipeline = full_source(volume=False)
    context = capture_proposal_context(pipeline, "input")
    raw = full_response(
        context,
        [full_node("smooth", "gaussian_blur_3d")],
        [("input", "smooth", 0)],
    )
    # The low-level callable can process lower-rank arrays, but the authored
    # graph explicitly requires ZYX. Follow that normal graph contract.
    with pytest.raises(ProposalError, match="ZYX"):
        validate_proposal(raw, context)


def test_mesh_generation_cannot_reinterpret_a_2d_plane_as_a_volume(no_execution):
    pipeline = full_source(volume=False)
    context = capture_proposal_context(pipeline, "input")
    raw = full_response(
        context,
        [full_node("mask", "binary_threshold"), full_node("mesh", "mask_to_3d_mesh")],
        [("input", "mask", 0), ("mask", "mesh", 0)],
    )
    with pytest.raises(ProposalError):
        validate_proposal(raw, context)


@pytest.mark.parametrize(
    "overrides",
    [
        {"path": "C:/assistant-chosen/output.npy"},
        {"path": "../output.npy"},
        {"overwrite": "yes"},
        {"format": "auto"},
    ],
)
def test_bound_save_cannot_accept_model_paths_overwrite_or_ambiguous_format(
    overrides, tmp_path, no_execution
):
    pipeline = full_source()
    destination = OutputDestination.from_directory(tmp_path)
    context = capture_proposal_context(
        pipeline, "input", output_destination=destination
    )
    raw = _save_response(context)
    raw["nodes"][0]["params"].update(overrides)
    with pytest.raises(ProposalError):
        validate_proposal(raw, context)
    assert tuple(tmp_path.iterdir()) == ()


def test_two_save_nodes_cannot_resolve_to_the_same_casefolded_filename(
    tmp_path, no_execution
):
    pipeline = full_source()
    destination = OutputDestination.from_directory(tmp_path)
    context = capture_proposal_context(
        pipeline, "input", output_destination=destination
    )
    raw = _save_response(context)
    duplicate = deepcopy(raw["nodes"][0])
    duplicate["key"] = "SAVE"
    raw["nodes"].append(duplicate)
    raw["connections"].append(
        {"source": "input", "source_port": 0, "target": "SAVE", "target_port": 0}
    )
    with pytest.raises(ProposalError):
        validate_proposal(raw, context)
    assert tuple(tmp_path.iterdir()) == ()


def test_image_writer_cannot_select_a_mesh_format(tmp_path, no_execution):
    pipeline = full_source()
    context = capture_proposal_context(
        pipeline, "input", output_destination=OutputDestination.from_directory(tmp_path)
    )
    raw = _save_response(context)
    raw["nodes"][0]["params"]["format"] = "obj"
    with pytest.raises(ProposalError):
        validate_proposal(raw, context)
    assert tuple(tmp_path.iterdir()) == ()


def test_mesh_writer_cannot_select_an_image_format(tmp_path, no_execution):
    pipeline = full_source()
    context = capture_proposal_context(
        pipeline, "input", output_destination=OutputDestination.from_directory(tmp_path)
    )
    raw = _save_response(context)
    raw["nodes"][0:0] = [
        full_node("mask", "binary_threshold"),
        full_node("mesh", "mask_to_3d_mesh"),
    ]
    raw["connections"] = [
        {"source": "input", "source_port": 0, "target": "mask", "target_port": 0},
        {"source": "mask", "source_port": 0, "target": "mesh", "target_port": 0},
        {"source": "mesh", "source_port": 0, "target": "save", "target_port": 0},
    ]
    with pytest.raises(ProposalError):
        validate_proposal(raw, context)
    assert tuple(tmp_path.iterdir()) == ()


@pytest.mark.parametrize(
    "operation_id", ["gaussian_blur", "add_images", "mask_to_3d_mesh"]
)
def test_strict_registry_kinds_still_reject_tables_and_meshes_as_arrays(
    operation_id, no_execution
):
    pipeline = full_source()
    context = capture_proposal_context(pipeline, "input")
    raw = full_response(
        context,
        [
            full_node("threshold", "binary_threshold"),
            full_node("labels", "label_connected_components"),
            full_node("measure", "measure_objects"),
            full_node("target", operation_id),
        ],
        [
            ("input", "threshold", 0),
            ("threshold", "labels", 0),
            ("labels", "measure", 0),
            ("measure", "target", 0),
        ]
        + ([("input", "target", 1)] if operation_id == "add_images" else []),
    )
    with pytest.raises(ProposalError):
        validate_proposal(raw, context)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_values_remain_rejected_with_full_library_access(bad, no_execution):
    pipeline = full_source()
    context = capture_proposal_context(pipeline, "input")
    raw = ordinary_operation_response(context, "gamma_correction", gamma=bad)
    with pytest.raises(ProposalError):
        validate_proposal(raw, context)


@pytest.mark.parametrize("port", [0, 9])
def test_split_axis_outputs_are_bound_to_actual_z_extent_not_default_port_count(
    port, no_execution
):
    pipeline = full_source()
    context = capture_proposal_context(pipeline, "input")
    raw = full_response(
        context,
        [
            full_node("split", "split_axis", axis="axis:0"),
            full_node("consumer", "gaussian_blur"),
        ],
        [("input", "split", 0), ("split", "consumer", 0, port)],
    )
    assert validate_proposal(raw, context).can_apply
    raw["connections"][-1]["source_port"] = 10
    with pytest.raises(ProposalError):
        validate_proposal(raw, context)


def test_split_channel_outputs_follow_in_graph_channel_count(no_execution):
    pipeline = full_source()
    context = capture_proposal_context(pipeline, "input")
    raw = full_response(
        context,
        [
            full_node("channels", "combine_channels", input_count=2),
            full_node("split", "split_channels"),
            full_node("consumer", "gaussian_blur_3d"),
        ],
        [
            ("input", "channels", 0),
            ("input", "channels", 1),
            ("channels", "split", 0),
            ("split", "consumer", 0, 1),
        ],
    )
    assert validate_proposal(raw, context).can_apply
    raw["connections"][-1]["source_port"] = 2
    with pytest.raises(ProposalError):
        validate_proposal(raw, context)


def test_crop_shape_mismatch_cannot_enter_multi_image_math(no_execution):
    pipeline = full_source()
    context = capture_proposal_context(pipeline, "input")
    raw = full_response(
        context,
        [full_node("cropped", "crop_stack", left=1), full_node("sum", "add_images")],
        [("input", "cropped", 0), ("input", "sum", 0), ("cropped", "sum", 1)],
    )
    with pytest.raises(ProposalError):
        validate_proposal(raw, context)


def test_equal_shape_different_physical_grid_cannot_enter_multi_image_math(
    no_execution,
):
    pipeline = full_source()
    context = capture_proposal_context(pipeline, "input")
    raw = full_response(
        context,
        [
            full_node(
                "calibrated", "set_pixel_size", x_size=2.0, y_size=2.0, z_size=2.0
            ),
            full_node("sum", "add_images"),
        ],
        [("input", "calibrated", 0), ("input", "sum", 0), ("calibrated", "sum", 1)],
    )
    with pytest.raises(ProposalError):
        validate_proposal(raw, context)


def test_equivalent_physical_grids_in_different_length_units_remain_compatible(
    no_execution,
):
    pipeline = full_source()
    context = capture_proposal_context(pipeline, "input")
    raw = full_response(
        context,
        [
            full_node(
                "nanometers",
                "set_pixel_size",
                x_size=500.0,
                y_size=500.0,
                z_size=500.0,
                unit="nanometer",
            ),
            full_node("sum", "add_images"),
        ],
        [("input", "nanometers", 0), ("input", "sum", 0), ("nanometers", "sum", 1)],
    )
    assert validate_proposal(raw, context).can_apply


def test_parameter_dependent_optional_mask_is_not_required_until_selected(no_execution):
    pipeline = full_source()
    context = capture_proposal_context(pipeline, "input")
    raw = ordinary_operation_response(context, "compare_images")
    assert validate_proposal(raw, context).can_apply
    raw["nodes"][-1]["params"]["use_mask"] = True
    with pytest.raises(ProposalError):
        validate_proposal(raw, context)
    raw["nodes"].insert(0, full_node("mask", "binary_threshold"))
    raw["connections"].extend(
        [
            {"source": "input", "source_port": 0, "target": "mask", "target_port": 0},
            {
                "source": "mask",
                "source_port": 0,
                "target": "candidate",
                "target_port": 2,
            },
        ]
    )
    assert validate_proposal(raw, context).can_apply


@pytest.mark.parametrize("axes", ["tyx", "tzyx", "cyx", "czyx", "tczyx"])
def test_explicit_time_and_channel_sources_keep_declared_axes_without_preview_guessing(
    axes, no_execution
):
    pipeline = full_source(axes=axes)
    context = capture_proposal_context(pipeline, "input")
    assert context.source_summary["axes"] == axes.upper()
    assert context.source_summary["shape"] == list(pipeline.outputs["input"].shape)
    assert context.image_previews == ()
    raw = full_response(
        context, [full_node("review", "review_images")], [("input", "review", 0)]
    )
    assert validate_proposal(raw, context).can_apply
    assert not pipeline.outputs["input"].flags.writeable


def test_time_detection_then_tracking_is_structurally_bound_to_complete_series(
    no_execution,
):
    pipeline = full_source(axes="tyx")
    context = capture_proposal_context(pipeline, "input")
    raw = full_response(
        context,
        [
            full_node("detect", "detect_spots_per_frame"),
            full_node("tracks", "build_tracks"),
        ],
        [("input", "detect", 0), ("detect", "tracks", 0)],
    )
    prepared = validate_proposal(raw, context)
    candidate = prepared.candidate.to_pipeline()
    assert prepared.can_apply
    assert len(candidate.output_ports(prepared.added_node_ids[-1])) == 2


def test_z_is_not_reinterpreted_as_time_for_per_frame_detection(no_execution):
    pipeline = full_source()
    context = capture_proposal_context(pipeline, "input")
    raw = full_response(
        context,
        [full_node("detect", "detect_spots_per_frame")],
        [("input", "detect", 0)],
    )
    with pytest.raises(ProposalError):
        validate_proposal(raw, context)


def test_static_spatial_object_measurements_cannot_be_reinterpreted_as_tracks(
    no_execution,
):
    pipeline = full_source()
    context = capture_proposal_context(pipeline, "input")
    raw = ordinary_operation_response(context, "build_tracks")
    with pytest.raises(ProposalError):
        validate_proposal(raw, context)


def test_channel_extraction_preserves_time_and_supports_per_frame_detection(
    no_execution,
):
    pipeline = full_source(axes="tczyx")
    context = capture_proposal_context(pipeline, "input")
    raw = full_response(
        context,
        [
            full_node("channel", "extract_channel", channel=1),
            full_node("detect", "detect_spots_per_frame"),
            full_node("tracks", "build_tracks"),
        ],
        [("input", "channel", 0), ("channel", "detect", 0), ("detect", "tracks", 0)],
    )
    assert validate_proposal(raw, context).can_apply


def test_time_registration_transform_and_coverage_outputs_have_real_ports(no_execution):
    pipeline = full_source(axes="tyx")
    context = capture_proposal_context(pipeline, "input")
    raw = full_response(
        context,
        [
            full_node("registration", "estimate_registration", mode="Time series"),
            full_node("aligned", "apply_transform"),
            full_node("review", "review_images"),
        ],
        [
            ("input", "registration", 0),
            ("input", "aligned", 0),
            ("registration", "aligned", 1),
            ("aligned", "review", 0),
            ("aligned", "review", 1, 1),
        ],
    )
    assert validate_proposal(raw, context).can_apply
    raw["connections"][2]["source_port"] = 1  # Diagnostics table is not a transform.
    with pytest.raises(ProposalError):
        validate_proposal(raw, context)


def test_template_score_and_validity_ports_can_feed_masked_peak_detection(no_execution):
    pipeline = full_source(volume=False)
    context = capture_proposal_context(pipeline, "input")
    raw = full_response(
        context,
        [
            full_node("match", "template_match"),
            full_node("peaks", "find_peaks", use_mask=True),
        ],
        [
            ("input", "match", 0),
            ("input", "match", 1),
            ("match", "peaks", 0),
            ("match", "peaks", 1, 1),
        ],
    )
    assert validate_proposal(raw, context).can_apply


def test_cellprofiler_static_multioutput_labels_bind_seeds_without_execution(
    no_execution,
):
    pipeline = full_source(volume=False)
    context = capture_proposal_context(pipeline, "input")
    raw = full_response(
        context,
        [
            full_node("nuclei", "cellprofiler_primary_objects"),
            full_node("seeds", "cellprofiler_propagation_seeds"),
        ],
        [("input", "nuclei", 0), ("nuclei", "seeds", 0, 1), ("nuclei", "seeds", 1, 0)],
    )
    assert validate_proposal(raw, context).can_apply


def test_multichannel_psf_ports_derive_from_actual_channel_count(no_execution):
    pipeline = full_source(axes="czyx")
    state = replace(
        pipeline.output_states["input"],
        channels=(
            ChannelMetadata(emission_wavelength=520.0, emission_wavelength_unit="nm"),
            ChannelMetadata(emission_wavelength=600.0, emission_wavelength_unit="nm"),
        ),
        acquisition=AcquisitionMetadata(objective_na=1.2, refractive_index=1.33),
    )
    pipeline.output_states["input"] = state
    pipeline.node_output_states["input"] = [state]
    context = capture_proposal_context(pipeline, "input")
    raw = ordinary_operation_response(context, "born_wolf_psf")
    # Normal manual PSF mode intentionally has one output, even on CZYX.
    # Auto mode uses actual channel optics to generate one PSF per channel.
    raw["nodes"][0]["params"]["auto_parameters"] = True
    raw["nodes"].append(full_node("prepare", "prepare_validate_psf"))
    raw["connections"].append(
        {"source": "candidate", "source_port": 1, "target": "prepare", "target_port": 0}
    )
    assert validate_proposal(raw, context).can_apply
    raw["connections"][-1]["source_port"] = 2
    with pytest.raises(ProposalError):
        validate_proposal(raw, context)


def test_tiny_cpu_axis_channel_math_composition_matches_numpy_and_preserves_source():
    """A small numerical wiring check, not qualification of the full library."""
    pipeline = full_source()
    data = pipeline.outputs["input"]
    before = data.copy()
    state = pipeline.output_states["input"]
    context = capture_proposal_context(pipeline, "input")
    raw = full_response(
        context,
        [
            full_node(
                "crop",
                "crop_stack",
                z_start=1,
                z_end=1,
                top=2,
                bottom=1,
                left=1,
                right=2,
            ),
            full_node("mean", "project_image", axes="name:z", method="Mean"),
            full_node("maximum", "project_image", axes="name:z", method="Maximum"),
            full_node("combined", "combine_channels", input_count=2),
            full_node("split", "split_channels"),
            full_node("difference", "subtract_images", input_count=2),
        ],
        [
            ("input", "crop", 0),
            ("crop", "mean", 0),
            ("crop", "maximum", 0),
            ("mean", "combined", 0),
            ("maximum", "combined", 1),
            ("combined", "split", 0),
            ("split", "difference", 0, 1),
            ("split", "difference", 1, 0),
        ],
    )
    prepared = validate_proposal(raw, context)
    candidate = prepared.candidate.to_pipeline()
    ids = {
        node["key"]: node_id
        for node, node_id in zip(raw["nodes"], prepared.added_node_ids, strict=True)
    }
    outputs = candidate.run(data, input_metadata={"vipp_image_state": state.to_dict()})
    cropped = before[1:-1, 2:-1, 1:-2]
    expected_mean = cropped.mean(axis=0)
    expected_max = cropped.max(axis=0)
    np.testing.assert_array_equal(outputs[ids["crop"]], cropped)
    np.testing.assert_array_equal(outputs[ids["mean"]], expected_mean)
    np.testing.assert_array_equal(outputs[ids["maximum"]], expected_max)
    np.testing.assert_array_equal(
        outputs[ids["combined"]], np.stack([expected_mean, expected_max])
    )
    np.testing.assert_array_equal(
        outputs[ids["difference"]], expected_max - expected_mean
    )
    np.testing.assert_array_equal(data, before)
    assert not data.flags.writeable


def test_input_count_cannot_exceed_real_port_limit_even_when_all_ports_are_wired(
    no_execution,
):
    pipeline = full_source()
    context = capture_proposal_context(pipeline, "input")
    raw = full_response(
        context,
        [full_node("sum", "add_images", input_count=13)],
        [("input", "sum", port) for port in range(12)],
    )
    with pytest.raises(ProposalError, match="input contract"):
        validate_proposal(raw, context)
    raw["nodes"][0]["params"]["input_count"] = 12
    assert validate_proposal(raw, context).can_apply
