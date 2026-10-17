"""Bounded broader-library authoring, with synthetic—not biological—evidence.

The 24-node acceptance graph preserves the topology of a manually solved
two-branch example, not its private metadata, pixel data, titles or parameters.
It deliberately leaves image borders open: it does not claim equivalence to a
workflow that explicitly caps surfaces at the image border.
"""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage as ndi

from napari_vipp.core.meshes import MeshData
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.pipeline import (
    MANUAL_AUTO_RECALCULATE_PARAM,
    NODE_LIBRARY_BY_ID,
    PrototypePipeline,
)
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.core.tables import TableData
from napari_vipp.core.workflow import (
    load_workflow,
    save_workflow,
    serialize_workflow,
)
from napari_vipp.core.workflow_proposal_catalogue import (
    input_ports_for,
    output_ports_for,
    unpack_catalogue,
)
from napari_vipp.core.workflow_proposals import (
    MAX_DIALOGUE_HISTORY_BYTES,
    MAX_DIALOGUE_PAIRS,
    MAX_PROPOSAL_BYTES,
    MAX_PROPOSAL_CONNECTIONS,
    MAX_PROPOSAL_NODES,
    MAX_REQUEST_BYTES,
    ProposalDialogue,
    ProposalError,
    build_catalogue,
    capture_proposal_context,
    validate_proposal,
)


def synthetic_source_pipeline(*, volume=True):
    """Owned read-only pixels, explicit axes, and no real-world metadata."""
    shape = (10, 20, 24) if volume else (20, 24)
    data = np.zeros(shape, np.float32)
    if volume:
        data[2:7, 3:9, 3:9] = 4
        data[3:8, 11:17, 14:20] = 10
    else:
        data[3:9, 3:9] = 4
        data[11:17, 14:20] = 10
    data.flags.writeable = False
    state = image_state_from_array(
        data,
        axes=tuple(
            AxisMetadata(name, "space", "um", scale=0.5)
            for name in ("zyx" if volume else "yx")
        ),
    )
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    pipeline.outputs["input"] = data
    pipeline.node_outputs["input"] = [data]
    pipeline.output_states["input"] = state
    pipeline.node_output_states["input"] = [state]
    pipeline.completed_node_ids.add("input")
    return pipeline


def library_node(key, operation_id, *, volume=True, **overrides):
    params = {
        parameter.name: deepcopy(parameter.default)
        for parameter in NODE_LIBRARY_BY_ID[operation_id].parameters
    }
    if "spatial_mode" in params:
        params["spatial_mode"] = "3D ZYX" if volume else "2D YX"
    if "boundary" in params:
        params["boundary"] = "Leave open at image border"
    params.update(overrides)
    return {
        "key": key,
        "operation_id": operation_id,
        "params": params,
        "reason": "Explicit synthetic trial choice; calculation is separate.",
    }


def library_proposal(context, nodes, links):
    """Links are (source, target, target_port[, source_port])."""
    return {
        "kind": "napari-vipp-workflow-proposal",
        "version": 1,
        "context_id": context.context_id,
        "summary": "A synthetic branched candidate, not a quality assessment.",
        "assumptions": [
            "Surface boundaries remain open; no missing tissue is inferred."
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


def solved_topology_proposal(context):
    """Sanitized 24 added nodes, 28 connections, 18 added operation kinds."""
    nodes = [
        library_node("background", "subtract_background", radius=6.0),
        library_node(
            "rescale",
            "rescale_intensity",
            cutoff_mode="Values",
            in_low_value=0.0,
            in_high_value=10.0,
            out_min=0.0,
            out_max=255.0,
        ),
        library_node(
            "denoise", "non_local_means_filter", patch_size=3, patch_distance=1
        ),
        library_node("all_signal", "otsu_threshold"),
        library_node("bright", "binary_threshold", threshold=180.0),
        library_node("difference", "logical_xor"),
        library_node("union", "logical_or"),
        library_node("bridge", "binary_threshold", threshold=240.0),
        library_node("outliers", "remove_binary_outliers", radius=0.5),
        library_node("erode_a", "erode", size=1),
        library_node("dilate_a", "dilate", size=1),
        library_node("cleanup", "remove_small_objects", min_size=2),
        library_node("mesh_a", "mask_to_3d_mesh"),
        library_node("mesh_b", "mask_to_3d_mesh"),
        library_node(
            "measure_a", "measure_3d_mesh_morphology", include_convex_hull_metrics=False
        ),
        library_node(
            "measure_b", "measure_3d_mesh_morphology", include_convex_hull_metrics=False
        ),
        library_node("comparison", "combine_channels"),
        library_node("histogram", "intensity_histogram", bin_count=8),
        library_node("dilate_b", "dilate", size=1),
        library_node("erode_b", "erode", size=1),
        library_node(
            "smooth_mask", "gaussian_blur_3d", sigma_z=0.4, sigma_y=0.4, sigma_x=0.4
        ),
        library_node("smooth_threshold", "otsu_threshold"),
        library_node("surfaces", "combine_meshes"),
        library_node("colours", "color_mesh_objects"),
    ]
    links = [
        ("input", "background", 0),
        ("background", "rescale", 0),
        ("rescale", "denoise", 0),
        ("denoise", "all_signal", 0),
        ("all_signal", "difference", 0),
        ("bright", "difference", 1),
        ("bridge", "outliers", 0),
        ("outliers", "erode_a", 0),
        ("erode_a", "union", 1),
        ("difference", "cleanup", 0),
        ("cleanup", "dilate_a", 0),
        ("mesh_a", "measure_a", 0),
        ("mesh_b", "measure_b", 0),
        ("denoise", "bright", 0),
        ("bright", "mesh_b", 0),
        ("input", "comparison", 0),
        ("background", "comparison", 1),
        ("comparison", "histogram", 0),
        ("rescale", "bridge", 0),
        ("dilate_a", "union", 0),
        ("dilate_b", "erode_b", 0),
        ("union", "smooth_mask", 0),
        ("smooth_mask", "smooth_threshold", 0),
        ("smooth_threshold", "mesh_a", 0),
        ("union", "dilate_b", 0),
        ("mesh_b", "surfaces", 0),
        ("mesh_a", "surfaces", 1),
        ("surfaces", "colours", 0),
    ]
    return library_proposal(context, nodes, links)


def _id_map(prepared, raw):
    return {
        node["key"]: node_id
        for node, node_id in zip(raw["nodes"], prepared.added_node_ids, strict=True)
    }


def test_full_registry_discovery_is_distinct_from_reviewed_authoring_authority():
    catalogue = build_catalogue()
    assert catalogue["version"] == 4
    entries = catalogue["registry_index"]
    assert {entry["operation_id"] for entry in entries} == set(NODE_LIBRARY_BY_ID)
    assert len(entries) == len(NODE_LIBRARY_BY_ID)
    detailed = {entry["operation_id"] for entry in catalogue["operations"]}
    assert len(detailed) == 143
    statuses = {entry["operation_id"]: entry["authoring_status"] for entry in entries}
    assert {
        key for key, status in statuses.items() if status == "supported"
    } == detailed
    for operation_id in ("input", "table_source", "batch_output"):
        assert statuses[operation_id] == "requires_explicit_binding"
        assert operation_id not in detailed
    assert statuses["save_output"] == "supported"
    assert next(entry for entry in entries if entry["operation_id"] == "save_output")[
        "requires_output_binding"
    ]
    assert statuses["richardson_lucy_deconvolution"] == "supported"
    for entry in entries:
        spec = NODE_LIBRARY_BY_ID[entry["operation_id"]]
        assert entry["title"] == spec.title
        assert entry["reason"]
    entries[0]["title"] = "Changed detached copy"
    assert build_catalogue()["registry_index"][0]["title"] != entries[0]["title"]


@pytest.mark.parametrize("volume", [False, True])
def test_complete_history_goal_and_full_catalogue_fit_independent_request_budget(
    volume,
):
    assert MAX_PROPOSAL_BYTES == 128 * 1024
    assert MAX_REQUEST_BYTES == 192 * 1024
    pipeline = synthetic_source_pipeline(volume=volume)
    context = capture_proposal_context(pipeline, "input")
    pairs = tuple(
        (f"Q{index:02d}" + "q" * 1021, "a" * 1024)
        for index in range(MAX_DIALOGUE_PAIRS)
    )
    assert len(pairs) == 32
    assert (
        sum(len(text.encode("utf-8")) for pair in pairs for text in pair)
        == MAX_DIALOGUE_HISTORY_BYTES
    )
    objective = "g" * 4000
    request = context.public_request(ProposalDialogue(objective, pairs))
    encoded = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()
    assert MAX_PROPOSAL_BYTES < len(encoded) <= MAX_REQUEST_BYTES
    assert request["objective"] == objective
    assert request["clarifications"] == [
        {"question": question, "answer": answer} for question, answer in pairs
    ]
    catalogue = unpack_catalogue(request["catalogue"])
    assert len(catalogue["operations"]) == 143
    assert len(catalogue["registry_index"]) == len(NODE_LIBRARY_BY_ID)
    assert request["image_previews"] == []


def test_json_escaping_cannot_bypass_the_192_kib_request_budget():
    pipeline = synthetic_source_pipeline()
    context = capture_proposal_context(pipeline, "input")
    pairs = tuple(
        (f"Q{index:02d}" + "\t" * 1021, "\t" * 1023 + "a")
        for index in range(MAX_DIALOGUE_PAIRS)
    )
    assert (
        sum(len(text.encode("utf-8")) for pair in pairs for text in pair)
        == MAX_DIALOGUE_HISTORY_BYTES
    )
    dialogue = ProposalDialogue("g" * 4000, pairs)
    with pytest.raises(ProposalError, match="request exceeds 192 KiB"):
        context.public_request(dialogue)
    assert dialogue.clarifications == pairs  # No silent history truncation.


@pytest.mark.parametrize(
    "raw",
    [" " * (128 * 1024 + 1), b" " * (128 * 1024 + 1)],
    ids=["text", "bytes"],
)
def test_larger_request_budget_does_not_expand_untrusted_response_budget(raw):
    pipeline = synthetic_source_pipeline()
    context = capture_proposal_context(pipeline, "input")
    with pytest.raises(ProposalError, match="128 KiB"):
        validate_proposal(raw, context)


@pytest.mark.parametrize("volume", [False, True])
def test_every_catalogue_parameter_default_matches_its_contextual_contract(volume):
    pipeline = synthetic_source_pipeline(volume=volume)
    context = capture_proposal_context(pipeline, "input")
    catalogue = unpack_catalogue(
        context.public_request("Prepare a synthetic trial")["catalogue"]
    )
    for entry in catalogue["operations"]:
        for parameter in entry["parameters"]:
            default = parameter["default"]
            label = f"{entry['operation_id']}.{parameter['name']}"
            if parameter["kind"] == "choice" and parameter["choices"]:
                assert default in parameter["choices"], label
            if parameter["kind"] in {"int", "float"}:
                assert not isinstance(default, bool), label
                assert parameter["minimum"] <= default <= parameter["maximum"], label
            if parameter.get("odd_only"):
                assert default % 2 == 1, label
            if parameter["name"] == "spatial_mode":
                assert default == ("3D ZYX" if volume else "2D YX"), label
            if parameter["name"] == "boundary":
                assert default in parameter["choices"], label
            if parameter["name"] == "channel_axis":
                assert default == -1, label


@pytest.mark.parametrize(
    "operation_id", ["combine_channels", "logical_or", "logical_xor", "combine_meshes"]
)
@pytest.mark.parametrize("count", [0, 1, True, 2.5, 32])
def test_dynamic_port_discovery_rejects_invalid_counts_without_clamping(
    operation_id, count
):
    for resolver in (input_ports_for, output_ports_for):
        with pytest.raises(ValueError, match="registry bounds"):
            resolver(operation_id, {"input_count": count})


@pytest.mark.parametrize("port_name", ["source_port", "target_port"])
def test_unavailable_port_32_is_rejected_without_clamping_to_a_valid_port(port_name):
    pipeline = synthetic_source_pipeline()
    context = capture_proposal_context(pipeline, "input")
    raw = library_proposal(
        context,
        [library_node("blur", "gaussian_blur")],
        [("input", "blur", 0)],
    )
    raw["connections"][0][port_name] = 32
    with pytest.raises(ProposalError):
        validate_proposal(raw, context)


def test_solved_topology_validates_detached_and_roundtrips_normal_persistence(tmp_path):
    pipeline = synthetic_source_pipeline()
    data = pipeline.outputs["input"]
    before = GraphSnapshot.from_pipeline(pipeline)
    pixels = data.copy()
    context = capture_proposal_context(pipeline, "input")
    raw = solved_topology_proposal(context)
    assert len(raw["nodes"]) == 24
    assert len(raw["connections"]) == 28
    assert len({node["operation_id"] for node in raw["nodes"]} | {"input"}) == 19
    assert MAX_PROPOSAL_NODES >= 24
    assert MAX_PROPOSAL_CONNECTIONS >= 28

    prepared = validate_proposal(raw, context)
    assert prepared.can_apply
    candidate = prepared.candidate.to_pipeline()
    assert len(candidate.nodes) == 25
    assert len(candidate.connections) == 28
    for node_id in prepared.added_node_ids:
        node = candidate.nodes[node_id]
        if NODE_LIBRARY_BY_ID[node.operation_id].execution_policy == "manual":
            assert node.params[MANUAL_AUTO_RECALCULATE_PARAM] is False
        assert candidate.outputs[node_id] is None
    path = save_workflow(tmp_path / "synthetic-branches.json", candidate)
    restored = load_workflow(path)
    replay = PrototypePipeline()
    replay.restore_graph(
        restored["nodes"], restored["connections"], restored["output_tunnels"]
    )
    assert serialize_workflow(replay) == serialize_workflow(candidate)
    assert GraphSnapshot.from_pipeline(pipeline) == before
    assert pipeline.outputs["input"] is data
    np.testing.assert_array_equal(data, pixels)
    assert not data.flags.writeable
    public = json.dumps(raw)
    assert "file_path" not in public
    assert "source_name" not in public
    assert "resolved_spatial_ndim" not in public


def test_full_topology_uses_normal_cpu_execution_and_keeps_source_readonly():
    pipeline = synthetic_source_pipeline()
    data = pipeline.outputs["input"]
    state = pipeline.output_states["input"]
    original = data.copy()
    context = capture_proposal_context(pipeline, "input")
    raw = solved_topology_proposal(context)
    prepared = validate_proposal(raw, context)
    candidate = prepared.candidate.to_pipeline()
    ids = _id_map(prepared, raw)
    outputs = candidate.run(data, input_metadata={"vipp_image_state": state.to_dict()})
    assert set(prepared.added_node_ids) <= candidate.completed_node_ids
    assert isinstance(outputs[ids["measure_a"]], TableData)
    assert isinstance(outputs[ids["measure_b"]], TableData)
    assert isinstance(outputs[ids["histogram"]], TableData)
    assert outputs[ids["comparison"]].shape == (2, *data.shape)
    assert isinstance(outputs[ids["colours"]], MeshData)
    assert outputs[ids["colours"]].object_count == 2
    np.testing.assert_array_equal(data, original)
    assert not data.flags.writeable


@pytest.mark.parametrize("volume", [False, True])
def test_twelve_input_mask_composition_uses_exact_active_ports(volume):
    pipeline = synthetic_source_pipeline(volume=volume)
    context = capture_proposal_context(pipeline, "input")
    nodes = [library_node("mask", "binary_threshold", volume=volume)]
    nodes.append(library_node("union", "logical_or", input_count=12))
    links = [("input", "mask", 0)] + [("mask", "union", port) for port in range(12)]
    raw = library_proposal(context, nodes, links)
    assert validate_proposal(raw, context).can_apply
    raw["connections"].pop()
    with pytest.raises(ProposalError, match="required input"):
        validate_proposal(raw, context)


@pytest.mark.parametrize("count", [2, 12])
def test_combine_channels_has_exact_dynamic_input_ports_and_histogram_support(count):
    pipeline = synthetic_source_pipeline()
    context = capture_proposal_context(pipeline, "input")
    raw = library_proposal(
        context,
        [
            library_node("comparison", "combine_channels", input_count=count),
            library_node("histogram", "intensity_histogram", bin_count=8),
        ],
        [("input", "comparison", port) for port in range(count)]
        + [("comparison", "histogram", 0)],
    )
    assert validate_proposal(raw, context).can_apply
    raw["connections"].pop(0)
    with pytest.raises(ProposalError, match="required input"):
        validate_proposal(raw, context)


def test_sixteen_mesh_inputs_use_the_actual_registry_limit():
    pipeline = synthetic_source_pipeline()
    context = capture_proposal_context(pipeline, "input")
    raw = library_proposal(
        context,
        [
            library_node("mask", "binary_threshold"),
            library_node("mesh", "mask_to_3d_mesh"),
            library_node("combined", "combine_meshes", input_count=16),
        ],
        [("input", "mask", 0), ("mask", "mesh", 0)]
        + [("mesh", "combined", port) for port in range(16)],
    )
    assert validate_proposal(raw, context).can_apply
    raw["connections"].pop()
    with pytest.raises(ProposalError, match="required input"):
        validate_proposal(raw, context)


def test_boolean_rescale_preserves_mask_kind_through_a_longer_branch():
    pipeline = synthetic_source_pipeline()
    context = capture_proposal_context(pipeline, "input")
    raw = library_proposal(
        context,
        [
            library_node("mask", "binary_threshold"),
            library_node("rescale", "rescale_intensity"),
            library_node("erode", "erode", size=1),
            library_node("mesh", "mask_to_3d_mesh"),
        ],
        [
            ("input", "mask", 0),
            ("mask", "rescale", 0),
            ("rescale", "erode", 0),
            ("erode", "mesh", 0),
        ],
    )
    assert validate_proposal(raw, context).can_apply


@pytest.mark.parametrize("patch_size", [4, 6, 14])
def test_nonlocal_means_even_patch_sizes_fail_before_execution(patch_size, monkeypatch):
    pipeline = synthetic_source_pipeline()
    context = capture_proposal_context(pipeline, "input")
    before = GraphSnapshot.from_pipeline(pipeline)
    monkeypatch.setattr(
        PrototypePipeline,
        "run",
        lambda *_a, **_k: pytest.fail("Parameter checks must not execute science"),
    )
    raw = library_proposal(
        context,
        [library_node("denoise", "non_local_means_filter", patch_size=patch_size)],
        [("input", "denoise", 0)],
    )
    with pytest.raises(ProposalError, match="odd"):
        validate_proposal(raw, context)
    assert GraphSnapshot.from_pipeline(pipeline) == before


@pytest.mark.parametrize("patch_size", [3, 5, 15])
def test_nonlocal_means_declared_odd_patch_sizes_validate_without_execution(
    patch_size, monkeypatch
):
    pipeline = synthetic_source_pipeline()
    context = capture_proposal_context(pipeline, "input")
    monkeypatch.setattr(
        PrototypePipeline,
        "run",
        lambda *_a, **_k: pytest.fail("Parameter checks must not execute science"),
    )
    raw = library_proposal(
        context,
        [library_node("denoise", "non_local_means_filter", patch_size=patch_size)],
        [("input", "denoise", 0)],
    )
    prepared = validate_proposal(raw, context)
    assert prepared.can_apply
    assert prepared.candidate.nodes[-1].params["patch_size"] == patch_size


def test_boolean_rescale_direct_to_strict_mesh_port_fails_closed_without_execution(
    monkeypatch,
):
    """Known conservative binding limit, not a reason to weaken mesh ports.

    The Boolean kernel preserves values, but the ordinary unexecuted registry
    edge still declares Rescale Intensity as image. The assistant must not
    silently override that general graph contract or execute to infer a type.
    """
    pipeline = synthetic_source_pipeline()
    context = capture_proposal_context(pipeline, "input")
    before = GraphSnapshot.from_pipeline(pipeline)
    monkeypatch.setattr(
        PrototypePipeline,
        "run",
        lambda *_a, **_k: pytest.fail("Graph validation must not execute science"),
    )
    raw = library_proposal(
        context,
        [
            library_node("mask", "binary_threshold"),
            library_node("rescale", "rescale_intensity"),
            library_node("mesh", "mask_to_3d_mesh"),
        ],
        [("input", "mask", 0), ("mask", "rescale", 0), ("rescale", "mesh", 0)],
    )
    with pytest.raises(ProposalError):
        validate_proposal(raw, context)
    assert GraphSnapshot.from_pipeline(pipeline) == before


def test_binary_morphology_can_explicitly_coerce_nonzero_smoothed_values():
    pipeline = synthetic_source_pipeline()
    context = capture_proposal_context(pipeline, "input")
    raw = library_proposal(
        context,
        [
            library_node("mask", "binary_threshold"),
            library_node("smooth", "gaussian_blur_3d"),
            library_node("dilate", "dilate", size=1),
        ],
        [("input", "mask", 0), ("mask", "smooth", 0), ("smooth", "dilate", 0)],
    )
    raw["assumptions"].append(
        "Dilation treats every nonzero smoothed value as foreground. This is "
        "an explicit coercion, not proof of a thresholded or accurate mask."
    )
    assert validate_proposal(raw, context).can_apply
    raw["nodes"].append(library_node("threshold", "otsu_threshold"))
    raw["connections"][-1]["source"] = "threshold"
    raw["connections"].append(
        {"source": "smooth", "source_port": 0, "target": "threshold", "target_port": 0}
    )
    assert validate_proposal(raw, context).can_apply


def test_manual_auto_recalculation_cannot_be_enabled_by_the_model():
    pipeline = synthetic_source_pipeline()
    context = capture_proposal_context(pipeline, "input")
    raw = solved_topology_proposal(context)
    measure = next(node for node in raw["nodes"] if node["key"] == "measure_a")
    measure["params"][MANUAL_AUTO_RECALCULATE_PARAM] = True
    with pytest.raises(ProposalError):
        validate_proposal(raw, context)


@pytest.mark.parametrize(
    "table_key,accepted",
    [
        ("measure_a", True),
        ("measure_b", False),
        ("histogram", False),
        ("mesh_measure", False),
    ],
)
def test_property_filters_require_object_measurements_from_the_exact_label_branch(
    table_key, accepted, monkeypatch
):
    pipeline = synthetic_source_pipeline()
    context = capture_proposal_context(pipeline, "input")
    before = GraphSnapshot.from_pipeline(pipeline)
    monkeypatch.setattr(
        PrototypePipeline,
        "run",
        lambda *_a, **_k: pytest.fail("Table lineage checks must not execute science"),
    )
    raw = library_proposal(
        context,
        [
            library_node("mask_a", "binary_threshold", threshold=1.0),
            library_node("mask_b", "binary_threshold", threshold=7.0),
            library_node("labels_a", "label_connected_components"),
            library_node("labels_b", "label_connected_components"),
            library_node("measure_a", "measure_objects"),
            library_node("measure_b", "measure_objects"),
            library_node("histogram", "intensity_histogram"),
            library_node("mesh", "labels_to_3d_mesh"),
            library_node("mesh_measure", "measure_3d_mesh_morphology"),
            library_node(
                "select", "filter_labels_by_property", property_column="label_id"
            ),
        ],
        [
            ("input", "mask_a", 0),
            ("input", "mask_b", 0),
            ("mask_a", "labels_a", 0),
            ("mask_b", "labels_b", 0),
            ("labels_a", "measure_a", 0),
            ("labels_b", "measure_b", 0),
            ("input", "histogram", 0),
            ("labels_a", "mesh", 0),
            ("mesh", "mesh_measure", 0),
            ("labels_a", "select", 0),
            (table_key, "select", 1),
        ],
    )
    if accepted:
        assert validate_proposal(raw, context).can_apply
    else:
        with pytest.raises(ProposalError, match="same labels input"):
            validate_proposal(raw, context)
    assert GraphSnapshot.from_pipeline(pipeline) == before


@pytest.mark.parametrize(
    "operation_id",
    ["save_output", "batch_output", "input", "table_source", "run_script"],
)
def test_discovery_does_not_authorize_sources_writers_or_external_execution(
    operation_id,
):
    pipeline = synthetic_source_pipeline()
    context = capture_proposal_context(pipeline, "input")
    before = GraphSnapshot.from_pipeline(pipeline)
    node = {
        "key": "forbidden",
        "operation_id": operation_id,
        "params": {},
        "reason": "Must not execute.",
    }
    raw = library_proposal(context, [node], [("input", "forbidden", 0)])
    with pytest.raises(ProposalError):
        validate_proposal(raw, context)
    assert GraphSnapshot.from_pipeline(pipeline) == before


@pytest.mark.parametrize(
    "operation_id,overrides",
    [
        ("subtract_background", {"radius": float("inf")}),
        (
            "rescale_intensity",
            {
                "cutoff_mode": "Percentiles",
                "in_low_percentile": 90.0,
                "in_high_percentile": 10.0,
            },
        ),
        ("rescale_intensity", {"out_min": 2.0, "out_max": 1.0}),
        ("non_local_means_filter", {"patch_size": True}),
        ("non_local_means_filter", {"channel_axis": 0}),
        ("remove_small_objects", {"min_size": -1}),
        ("dilate", {"size": 0}),
        ("erode", {"iterations": 0}),
        ("logical_xor", {"input_count": 13}),
        ("combine_channels", {"input_count": 13}),
        ("combine_meshes", {"input_count": 17}),
        (
            "intensity_histogram",
            {"range_mode": "Custom range", "custom_min": 2, "custom_max": 1},
        ),
    ],
)
def test_bad_new_operation_parameters_are_rejected_without_live_mutation(
    operation_id, overrides
):
    pipeline = synthetic_source_pipeline()
    context = capture_proposal_context(pipeline, "input")
    before = GraphSnapshot.from_pipeline(pipeline)
    # Keep every port/type valid, so rejection cannot be an unrelated missing
    # input or intensity-to-mask failure that would also reject valid params.
    raw = solved_topology_proposal(context)
    node = next(node for node in raw["nodes"] if node["operation_id"] == operation_id)
    node["params"].update(overrides)
    with pytest.raises(ProposalError):
        validate_proposal(raw, context)
    assert GraphSnapshot.from_pipeline(pipeline) == before


@pytest.mark.parametrize(
    "operation_id",
    ["remove_small_objects"],
)
def test_strict_mask_or_label_ports_do_not_accept_raw_intensities(operation_id):
    pipeline = synthetic_source_pipeline()
    context = capture_proposal_context(pipeline, "input")
    links = [("input", "mask_only", 0)]
    if operation_id in {"logical_or", "logical_xor"}:
        links.append(("input", "mask_only", 1))
    raw = library_proposal(context, [library_node("mask_only", operation_id)], links)
    with pytest.raises(ProposalError):
        validate_proposal(raw, context)


@pytest.mark.parametrize(
    "operation_id",
    [
        "label_connected_components",
        "mask_to_3d_mesh",
    ],
)
def test_channel_stacks_cannot_silently_enter_strict_mask_only_stages(operation_id):
    pipeline = synthetic_source_pipeline()
    context = capture_proposal_context(pipeline, "input")
    raw = library_proposal(
        context,
        [
            library_node("channels", "combine_channels"),
            library_node("scalar_only", operation_id),
        ],
        [
            ("input", "channels", 0),
            ("input", "channels", 1),
            ("channels", "scalar_only", 0),
        ],
    )
    with pytest.raises(ProposalError):
        validate_proposal(raw, context)


@pytest.mark.parametrize(
    "operation_id", ["dilate", "erode", "logical_or", "logical_xor"]
)
def test_normal_array_mask_operations_keep_explicit_nonzero_coercion(operation_id):
    pipeline = synthetic_source_pipeline()
    context = capture_proposal_context(pipeline, "input")
    links = [("input", "coerce", 0)]
    if operation_id in {"logical_or", "logical_xor"}:
        links.append(("input", "coerce", 1))
    raw = library_proposal(context, [library_node("coerce", operation_id)], links)
    raw["assumptions"].append(
        "This binary operation treats nonzero input values as foreground; "
        "no intensity threshold or object-accuracy claim is implied."
    )
    assert validate_proposal(raw, context).can_apply


@pytest.mark.parametrize("operation_id", ["binary_threshold", "gaussian_blur"])
def test_declared_channel_arrays_keep_normal_channel_aware_processing(operation_id):
    pipeline = synthetic_source_pipeline()
    context = capture_proposal_context(pipeline, "input")
    raw = library_proposal(
        context,
        [
            library_node("channels", "combine_channels"),
            library_node("process", operation_id),
        ],
        [
            ("input", "channels", 0),
            ("input", "channels", 1),
            ("channels", "process", 0),
        ],
    )
    assert validate_proposal(raw, context).can_apply


def test_maximum_node_budget_is_usable_but_not_unbounded():
    assert MAX_PROPOSAL_NODES == 64
    assert MAX_PROPOSAL_CONNECTIONS == 128
    pipeline = synthetic_source_pipeline()
    context = capture_proposal_context(pipeline, "input")
    nodes = [library_node(f"blur_{index}", "gaussian_blur") for index in range(64)]
    links = [
        ("input" if index == 0 else f"blur_{index - 1}", f"blur_{index}", 0)
        for index in range(64)
    ]
    raw = library_proposal(context, nodes, links)
    assert validate_proposal(raw, context).can_apply
    raw["nodes"].append(library_node("one_too_many", "gaussian_blur"))
    with pytest.raises(ProposalError):
        validate_proposal(raw, context)
    raw = library_proposal(context, nodes[:1], links[:1])
    raw["connections"] *= 129
    with pytest.raises(ProposalError):
        validate_proposal(raw, context)


def test_broader_topology_still_rejects_cycles_and_stale_source():
    pipeline = synthetic_source_pipeline()
    context = capture_proposal_context(pipeline, "input")
    raw = solved_topology_proposal(context)
    prepared = validate_proposal(raw, context)
    loop = deepcopy(raw)
    loop["connections"][0]["source"] = "rescale"
    with pytest.raises(ProposalError):
        validate_proposal(loop, context)
    data = pipeline.outputs["input"].copy()
    data[0, 0, 0] = 1
    data.flags.writeable = False
    pipeline.outputs["input"] = data
    pipeline.node_outputs["input"] = [data]
    with pytest.raises(ProposalError, match="source changed"):
        prepared.assert_current(pipeline)


def test_small_branched_execution_matches_exact_synthetic_masks_and_two_objects():
    """Independent array/ndimage oracle checks wiring, not scientific suitability."""
    pipeline = synthetic_source_pipeline()
    data = pipeline.outputs["input"]
    state = pipeline.output_states["input"]
    context = capture_proposal_context(pipeline, "input")
    raw = library_proposal(
        context,
        [
            library_node("all", "binary_threshold", threshold=1.0),
            library_node("bright", "binary_threshold", threshold=7.0),
            library_node("dim", "logical_xor"),
            library_node("eroded", "erode", size=3),
            library_node("expanded", "dilate", size=3),
            library_node("mesh_dim", "mask_to_3d_mesh"),
            library_node("mesh_bright", "mask_to_3d_mesh"),
            library_node("combined", "combine_meshes"),
            library_node("colours", "color_mesh_objects"),
        ],
        [
            ("input", "all", 0),
            ("input", "bright", 0),
            ("all", "dim", 0),
            ("bright", "dim", 1),
            ("dim", "eroded", 0),
            ("eroded", "expanded", 0),
            ("expanded", "mesh_dim", 0),
            ("bright", "mesh_bright", 0),
            ("mesh_dim", "combined", 0),
            ("mesh_bright", "combined", 1),
            ("combined", "colours", 0),
        ],
    )
    prepared = validate_proposal(raw, context)
    candidate = prepared.candidate.to_pipeline()
    ids = _id_map(prepared, raw)
    outputs = candidate.run(data, input_metadata={"vipp_image_state": state.to_dict()})
    expected_bright = data > 7.0
    expected_dim = np.logical_xor(data > 1.0, expected_bright)
    # VIPP Erode/Dilate explicitly operate slice by slice in XY, not in 3D.
    structure = np.ones((1, 3, 3), bool)
    expected_eroded = ndi.binary_erosion(expected_dim, structure=structure)
    expected_expanded = ndi.binary_dilation(expected_eroded, structure=structure)
    np.testing.assert_array_equal(outputs[ids["bright"]], expected_bright)
    np.testing.assert_array_equal(outputs[ids["dim"]], expected_dim)
    np.testing.assert_array_equal(outputs[ids["eroded"]], expected_eroded)
    np.testing.assert_array_equal(outputs[ids["expanded"]], expected_expanded)
    mesh = outputs[ids["colours"]]
    assert isinstance(mesh, MeshData)
    assert mesh.object_count == 2
    assert len({tuple(obj.color) for obj in mesh.objects}) == 2
    assert len(mesh.vertices) > 0 and len(mesh.faces) > 0
    assert not mesh.vertices.flags.writeable
    assert not mesh.faces.flags.writeable
    assert not data.flags.writeable


def test_add_full_topology_is_one_undoable_non_running_edit(
    qtbot, monkeypatch, tmp_path
):
    from napari.components import ViewerModel

    from napari_vipp._widget import VippWidget
    from napari_vipp.core import workflow_proposal_codex
    from napari_vipp.core.compute import ComputeMode

    monkeypatch.setattr(
        workflow_proposal_codex,
        "discover_codex_runtime",
        lambda: SimpleNamespace(
            supported=True, authenticated=False, detail="Offline test"
        ),
    )
    monkeypatch.setattr(
        workflow_proposal_codex,
        "generate_codex_proposal",
        lambda *_a, **_k: pytest.fail("No provider request during validation/Add"),
    )
    widget = VippWidget(
        ViewerModel(), defer_initial_run=True, initial_compute_mode=ComputeMode.CPU
    )
    qtbot.addWidget(widget)
    monkeypatch.setattr(widget, "_confirm_close_dirty_workflow_tabs", lambda: True)
    widget.pipeline = synthetic_source_pipeline()
    widget._build_graph_from_pipeline()
    calls = []
    monkeypatch.setattr(widget, "run_pipeline", lambda **kwargs: calls.append(kwargs))
    widget._history.clear()
    before = GraphSnapshot.from_pipeline(widget.pipeline)
    data = widget.pipeline.outputs["input"]
    context, owner = widget._capture_workflow_proposal_context("input")
    prepared = validate_proposal(solved_topology_proposal(context), context)
    ids = widget._apply_workflow_proposal(prepared, owner)
    assert len(ids) == 24
    assert len(widget.pipeline.nodes) == 25
    assert len(widget._history.undo_stack) == 1
    assert widget.pipeline.outputs["input"] is data
    assert all(widget.pipeline.outputs[node_id] is None for node_id in ids)
    assert calls == []
    assert not widget._debounce_timer.isActive()
    accepted = GraphSnapshot.from_pipeline(widget.pipeline)
    save_workflow(tmp_path / "accepted.json", widget.pipeline)
    widget.undo()
    assert GraphSnapshot.from_pipeline(widget.pipeline) == before
    assert calls == []
    widget.redo()
    assert GraphSnapshot.from_pipeline(widget.pipeline) == accepted
    assert calls == []
