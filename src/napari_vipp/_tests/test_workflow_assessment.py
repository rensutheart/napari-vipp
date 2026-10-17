"""Analytical cache-only evidence and privacy/freshness contracts; no model/Qt."""

import json
from dataclasses import replace
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from napari_vipp.core.meshes import mask_to_3d_mesh
from napari_vipp.core.metadata import (
    AcquisitionMetadata,
    AxisMetadata,
    SourceMetadata,
    image_state_from_array,
)
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.tables import TableData, table_state_from_data
from napari_vipp.core.workflow_assessment import (
    MAX_DIAGNOSTIC_ELEMENTS,
    AssessmentCancelled,
    AssessmentError,
    build_assessment_context,
    capture_assessment_snapshot,
)
from napari_vipp.core.workflow_proposals import ProposalDialogue


def state_for(data, *, kind=None, axes=None):
    axes = axes or tuple(
        AxisMetadata(name, "space", "um", scale=0.5)
        for name in ("zyx" if data.ndim == 3 else "yx")
    )
    state = image_state_from_array(
        data,
        axes=axes,
        source_name="PRIVATE_SAMPLE_NAME",
        source=SourceMetadata(uri="C:/PRIVATE_PATIENT/input.tif"),
        acquisition=AcquisitionMetadata(description="PRIVATE_ACQUISITION"),
        history=("PRIVATE_HISTORY",),
    )
    return replace(state, kind=kind) if kind else state


def commit_cache(pipeline, node_id, data, state=None):
    if state is None and type(data) is np.ndarray:
        state = state_for(data)
    pipeline.outputs[node_id] = data
    pipeline.output_states[node_id] = state
    pipeline.node_outputs[node_id] = [data]
    pipeline.node_output_states[node_id] = [state]
    pipeline.node_execution_states[node_id] = "ready"
    pipeline.completed_node_ids.add(node_id)


def fixture(*, mask=None, shape=(5, 6, 7), mesh=False):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    source = np.arange(np.prod(shape), dtype=np.float32).reshape(shape).copy()
    source.setflags(write=False)
    commit_cache(pipeline, "input", source)
    pipeline.nodes["input"].params["file_path"] = "C:/PRIVATE_PATIENT/input.tif"
    if mask is None:
        mask = np.zeros(shape, dtype=bool)
        mask[(slice(1, 3),) * len(shape)] = True
    threshold = pipeline.add_node("binary_threshold")
    assert pipeline.connect("input", threshold.id).success
    commit_cache(pipeline, threshold.id, mask, state_for(mask, kind="binary mask"))
    node_ids = [threshold.id]
    if mesh:
        node = pipeline.add_node("mask_to_3d_mesh")
        assert pipeline.connect(threshold.id, node.id).success
        output = mask_to_3d_mesh(mask, image_state=pipeline.output_states[threshold.id])
        commit_cache(pipeline, node.id, output, output.state)
        node_ids.append(node.id)
    return pipeline, tuple(node_ids)


def context_for(pipeline, nodes, *, input_images=False, output_images=False):
    snapshot = capture_assessment_snapshot(pipeline, "input", nodes, "run-7")
    return build_assessment_context(
        snapshot, include_input_images=input_images, include_output_images=output_images
    )


def outputs(context):
    return context.public_request("Separate the two structures")["selected_outputs"]


def test_exact_3d_face_and_full_components_and_border_counts():
    mask = np.zeros((5, 6, 7), dtype=bool)
    mask[0, 0, 0] = mask[1, 1, 1] = mask[4, 5, 6] = True
    pipeline, nodes = fixture(mask=mask)
    result = outputs(context_for(pipeline, nodes))[0]["diagnostics"]
    assert result["sampling"] == "exact full array"
    assert result["foreground_count"] == 3
    assert result["foreground_border_count"] == 2
    face, full = (result["foreground_components"][name] for name in ("face", "full"))
    assert face["neighbours"] == 6 and face["component_count"] == 3
    assert full["neighbours"] == 26 and full["component_count"] == 2
    assert full["largest_sizes"] == [2, 1]
    assert face["border_touching_components"] == full["border_touching_components"] == 2


def test_2d_connectivity_is_not_mislabelled_6_or_26():
    mask = np.eye(4, dtype=bool)
    pipeline, nodes = fixture(mask=mask, shape=mask.shape)
    result = outputs(context_for(pipeline, nodes))[0]["diagnostics"]
    assert result["foreground_components"]["face"]["neighbours"] == 4
    assert result["foreground_components"]["face"]["component_count"] == 4
    assert result["foreground_components"]["full"]["neighbours"] == 8
    assert result["foreground_components"]["full"]["component_count"] == 1


def test_grouped_mesh_is_not_mistaken_for_one_connected_object():
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[1, 1, 1] = mask[4, 4, 4] = True
    pipeline, nodes = fixture(mask=mask, shape=mask.shape, mesh=True)
    records = outputs(context_for(pipeline, nodes))
    mesh, mask = (record["diagnostics"] for record in records)
    assert mesh["display_object_group_count"] == 1
    assert mesh["topological_component_count"] == 2
    assert mesh["surface_component_count"] == 2
    assert mesh["euler_characteristic"] == 4
    assert mesh["boundary_edge_count"] == 0
    assert mask["foreground_components"]["face"]["component_count"] == 2
    assert records[1]["role"] == "exact mesh input"


def test_actual_mesh_input_not_better_looking_old_candidate():
    pipeline, nodes = fixture(mesh=True)
    old = pipeline.add_node("binary_threshold")
    assert pipeline.connect("input", old.id).success
    commit_cache(pipeline, old.id, np.ones((5, 6, 7), dtype=bool))
    request = context_for(pipeline, nodes).public_request("goal")
    assert len(request["candidate_graph"]["nodes"]) == 2
    assert request["selected_outputs"][1]["diagnostics"]["foreground_count"] == 8
    assert old.id not in json.dumps(request)


def test_readonly_detached_cache_preserves_original_and_detects_mutation():
    pipeline, nodes = fixture()
    original = pipeline.outputs[nodes[0]].copy()
    snapshot = capture_assessment_snapshot(pipeline, "input", nodes, "run-7")
    owned = snapshot._outputs[0].binding.data
    assert owned is not pipeline.outputs[nodes[0]]
    assert not owned.flags.writeable
    context = build_assessment_context(snapshot, include_input_images=False)
    context.assert_current(pipeline)
    np.testing.assert_array_equal(pipeline.outputs[nodes[0]], original)
    pipeline.outputs[nodes[0]][0, 0, 0] = True
    with pytest.raises(AssessmentError):
        context.assert_current(pipeline)
    np.testing.assert_array_equal(owned, original)


@pytest.mark.parametrize(
    "change", ["replace", "state", "graph", "execution", "completed", "source"]
)
def test_stale_replacement_state_graph_run_or_source_is_rejected(change):
    pipeline, nodes = fixture()
    context = context_for(pipeline, nodes)
    node_id = nodes[0]
    if change == "replace":
        commit_cache(pipeline, node_id, pipeline.outputs[node_id].copy())
    elif change == "state":
        pipeline.node_output_states[node_id][0] = replace(
            pipeline.output_states[node_id], history=("new provenance",)
        )
    elif change == "graph":
        pipeline.nodes[node_id].params["threshold"] = 17
    elif change == "execution":
        pipeline.node_execution_states[node_id] = "stale"
    elif change == "completed":
        pipeline.completed_node_ids.remove(node_id)
    else:
        pipeline.outputs["input"].setflags(write=True)
        pipeline.outputs["input"][0, 0, 0] = -1
        pipeline.outputs["input"].setflags(write=False)
    with pytest.raises(AssessmentError):
        context.assert_current(pipeline)


@pytest.mark.parametrize(
    "input_images,output_images,count",
    [
        (False, False, 0),
        (True, False, 1),
        (False, True, 3),
        (True, True, 4),
    ],
)
def test_independent_image_permissions_are_explicit_and_no_hidden_source_overlay(
    input_images, output_images, count
):
    pipeline, nodes = fixture()
    context = context_for(
        pipeline, nodes, input_images=input_images, output_images=output_images
    )
    assert len(context.image_previews) == count
    if output_images:
        preview = context.image_previews[int(input_images)]
        assert ("Source-intensity background" in preview.description) == input_images
        with Image.open(BytesIO(preview.png_bytes)) as image:
            pixels = np.array(image)
            assert image.info == {}
        if not input_images:
            assert not pixels[0, 0].any()
    request = context.public_request("goal")
    assert request["image_permissions"] == {
        "input_images": input_images,
        "output_images": output_images,
    }
    assert "png_bytes" not in json.dumps(request)


def test_same_shape_different_grid_never_overlaid_as_aligned():
    pipeline, nodes = fixture()
    state = pipeline.output_states[nodes[0]]
    axes = (replace(state.axes[0], translation=1), *state.axes[1:])
    commit_cache(
        pipeline, nodes[0], pipeline.outputs[nodes[0]], replace(state, axes=axes)
    )
    context = context_for(pipeline, nodes, input_images=True, output_images=True)
    assert outputs(context)[0]["diagnostics"]["same_source_grid"] is False
    assert "No input-intensity overlay" in context.image_previews[1].description


def test_no_private_paths_names_history_old_graph_or_raw_arrays_in_request():
    pipeline, nodes = fixture(mesh=True)
    save = pipeline.add_node("save_output")
    save.params.update(output_directory="C:/PRIVATE_OUTPUT", filename="PRIVATE_RESULT")
    assert pipeline.connect(nodes[-1], save.id).success
    # A completed Save is deliberately not a second, guessed scientific result.
    commit_cache(
        pipeline,
        save.id,
        pipeline.outputs[nodes[-1]],
        pipeline.output_states[nodes[-1]],
    )
    context = context_for(
        pipeline, (*nodes, save.id), input_images=True, output_images=True
    )
    request = context.public_request(
        ProposalDialogue("goal", (("question", "answer"),))
    )
    text = json.dumps(request)
    for marker in ("PRIVATE", "C:/", "png_bytes", "vertices", "PRIVATE_HISTORY"):
        assert marker not in text
    assert request["source_id"] == "input"
    assert request["candidate_node_ids"] == ["n1", "n2", "n3"]
    assert request["clarifications"] == [{"question": "question", "answer": "answer"}]
    assert request["kind"] == "napari-vipp-output-assessment-request"
    assert request["assessment_id"] == context.assessment_id == context.context_id
    assert request["run_id"] == "run-7"
    request["source"]["grid"]["shape"][0] = -1
    assert context.public_request("goal")["source"]["grid"]["shape"][0] == 5


def test_empty_and_full_masks_are_described_without_false_success():
    for value, expected in ((False, 0), (True, 1)):
        pipeline, nodes = fixture(mask=np.full((5, 6, 7), value))
        result = outputs(context_for(pipeline, nodes))[0]["diagnostics"]
        assert result["foreground_fraction"] == float(value)
        assert result["foreground_components"]["face"]["component_count"] == expected
        assert "success" not in result


def test_sparse_label_ids_and_disconnected_single_id_are_distinguished():
    pipeline, nodes = fixture()
    labels = np.zeros((5, 6, 7), dtype=np.uint64)
    labels[1, 1, 1] = labels[3, 3, 3] = 2**63 + 7
    commit_cache(pipeline, nodes[0], labels, state_for(labels, kind="label image"))
    result = outputs(context_for(pipeline, nodes))[0]["diagnostics"]
    assert result["maximum"] == 2**63 + 7
    assert result["label_ids_count"] == 1
    assert result["foreground_components"]["face"]["component_count"] == 2
    assert result["largest_label_sizes"] == [2]


def test_nonfinite_intensity_is_counted_and_visibly_marked_not_repaired():
    pipeline, nodes = fixture()
    values = pipeline.outputs["input"].copy()
    values[2, 0, 0], values[2, 0, 1] = np.nan, np.inf
    commit_cache(pipeline, nodes[0], values, state_for(values, kind="intensity image"))
    context = context_for(pipeline, nodes, output_images=True)
    result = outputs(context)[0]["diagnostics"]
    assert result["nonfinite_count"] == 2
    assert result["finite_count"] == values.size - 2
    with Image.open(BytesIO(context.image_previews[0].png_bytes)) as image:
        assert tuple(np.array(image)[0, 0]) == (255, 0, 255)
    assert np.isnan(pipeline.outputs[nodes[0]][2, 0, 0])


def test_table_numeric_summaries_do_not_share_row_strings_or_arbitrary_names():
    pipeline, nodes = fixture()
    table = TableData(
        columns=("voxel_count", "C:/PRIVATE_COLUMN"),
        rows=((7, "PRIVATE_ROW"), (13, "C:/PRIVATE_PATH")),
        name="PRIVATE_TABLE",
        source_name="PRIVATE_SOURCE",
    )
    commit_cache(pipeline, nodes[0], table, table_state_from_data(table))
    result = outputs(context_for(pipeline, nodes))[0]["diagnostics"]
    assert result["row_count"] == 2
    assert result["column_summaries"][0]["minimum"] == 7
    assert result["column_summaries"][0]["maximum"] == 13
    assert result["column_summaries"][1]["text_count"] == 2
    assert "PRIVATE" not in json.dumps(result)


def test_cancelled_exact_analysis_never_returns_partial_context():
    pipeline, nodes = fixture(mesh=True)
    snapshot = capture_assessment_snapshot(pipeline, "input", nodes, "run-7")
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls >= 3

    with pytest.raises(AssessmentCancelled):
        build_assessment_context(
            snapshot, include_input_images=True, cancelled=cancelled
        )
    assert calls == 3


@pytest.mark.parametrize(
    "status", ["error", "running", "stale", "blocked", "not_calculated"]
)
def test_old_cache_is_not_assessable_when_latest_run_did_not_finish(status):
    pipeline, nodes = fixture()
    pipeline.node_execution_states[nodes[0]] = status
    with pytest.raises(AssessmentError, match="Run the complete candidate"):
        capture_assessment_snapshot(pipeline, "input", nodes, "run-7")


def test_output_free_review_sink_does_not_block_successful_scientific_run():
    pipeline, nodes = fixture()
    review = pipeline.add_node("review_images")
    assert pipeline.connect(nodes[0], review.id).success
    context = context_for(pipeline, (*nodes, review.id))
    assert len(outputs(context)) == 1
    pipeline.node_execution_states[review.id] = "error"
    with pytest.raises(AssessmentError):
        context_for(pipeline, (*nodes, review.id))


def test_oversized_full_source_refused_without_implicit_diagnostic_sampling():
    pipeline, nodes = fixture()
    large = np.zeros((2001, 2000), dtype=np.uint8)
    assert large.size > MAX_DIAGNOSTIC_ELEMENTS
    large.setflags(write=False)
    commit_cache(pipeline, "input", large, state_for(large))
    with pytest.raises(AssessmentError, match="no implicit sampling"):
        capture_assessment_snapshot(pipeline, "input", nodes, "run-7")


def test_more_than_eight_terminal_results_are_disclosed_not_silently_replaced():
    pipeline, nodes = fixture()
    node_ids = list(nodes)
    for _ in range(9):
        node = pipeline.add_node("binary_threshold")
        assert pipeline.connect("input", node.id).success
        commit_cache(pipeline, node.id, np.zeros((5, 6, 7), dtype=bool))
        node_ids.append(node.id)
    request = context_for(pipeline, tuple(node_ids)).public_request("goal")
    assert len(request["selected_outputs"]) == 8
    assert len(request["omitted_outputs"]) == 2
    context = context_for(
        pipeline, tuple(node_ids), input_images=True, output_images=True
    )
    assert len(context.image_previews) == 7


def test_revision_feedback_is_full_detached_evidence_with_private_guard():
    pipeline, nodes = fixture(mesh=True)
    context = context_for(pipeline, nodes)
    report = {
        "assessment_id": context.assessment_id,
        "run_id": context.run_id,
        "summary": "The mask is fragmented.",
    }
    feedback = context.revision_feedback(report)
    assert len(feedback["evidence"]["selected_outputs"]) == 2
    assert context.snapshot.assessment_id == context.assessment_id
    feedback["assessment"]["summary"] = "mutated"
    assert report["summary"] == "The mask is fragmented."
    with pytest.raises(AssessmentError):
        context.revision_feedback({**report, "run_id": "another-run"})


def test_unknown_unsupported_output_is_reported_when_another_output_is_supported():
    pipeline, nodes = fixture()
    extra = pipeline.add_node("binary_threshold")
    assert pipeline.connect("input", extra.id).success
    commit_cache(pipeline, extra.id, object(), None)
    records = outputs(context_for(pipeline, (*nodes, extra.id)))
    assert records[1]["availability"] == "unavailable"
    assert "adapter" in records[1]["reason"]


def test_candidate_cannot_reuse_unrelated_cached_old_nodes_as_evidence():
    pipeline, nodes = fixture()
    next_node = pipeline.add_node("remove_small_objects")
    assert pipeline.connect(nodes[0], next_node.id).success
    commit_cache(pipeline, next_node.id, pipeline.outputs[nodes[0]])
    with pytest.raises(AssessmentError, match="outside"):
        context_for(pipeline, (next_node.id,))


def test_unselected_intermediate_pixel_mutation_invalidates_the_assessment():
    pipeline, nodes = fixture(mesh=True)
    middle = pipeline.add_node("gaussian_blur_3d")
    pipeline.connections = [
        edge for edge in pipeline.connections if edge.target_id != nodes[0]
    ]
    assert pipeline.connect("input", middle.id).success
    assert pipeline.connect(middle.id, nodes[0]).success
    commit_cache(pipeline, middle.id, pipeline.outputs["input"].copy())
    # Only the mesh and its exact mask input are selected for diagnostics; the
    # intermediate blur still enters the complete candidate freshness gate.
    context = context_for(pipeline, (middle.id, *nodes))
    assert len(outputs(context)) == 2
    pipeline.outputs[middle.id][0, 0, 0] = -100
    with pytest.raises(AssessmentError):
        context.assert_current(pipeline)


@pytest.mark.parametrize("dtype,base", [(np.uint64, 2**63 + 10), (np.int64, -(2**63))])
def test_wide_integer_previews_do_not_lose_adjacent_native_values(dtype, base):
    pipeline, nodes = fixture(shape=(1, 2))
    values = np.array([[base, base + 1]], dtype=dtype)
    commit_cache(pipeline, nodes[0], values, state_for(values, kind="intensity image"))
    context = context_for(pipeline, nodes, output_images=True)
    with Image.open(BytesIO(context.image_previews[0].png_bytes)) as image:
        np.testing.assert_array_equal(np.array(image), [[0, 255]])


def test_preview_limit_discloses_every_omitted_output_section():
    pipeline, nodes = fixture()
    node_ids = list(nodes)
    for _ in range(2):
        node = pipeline.add_node("binary_threshold")
        assert pipeline.connect("input", node.id).success
        commit_cache(pipeline, node.id, np.zeros((5, 6, 7), dtype=bool))
        node_ids.append(node.id)
    context = context_for(
        pipeline, tuple(node_ids), input_images=True, output_images=True
    )
    request = context.public_request("goal")
    assert len(context.image_previews) == 7
    assert len(request["image_preview_omissions"]) == 3


def test_checkbox_flags_must_be_booleans_and_lazy_source_is_never_materialized():
    pipeline, nodes = fixture()
    snapshot = capture_assessment_snapshot(pipeline, "input", nodes, "run-7")
    with pytest.raises(TypeError):
        build_assessment_context(snapshot, include_input_images=1)

    class Lazy:
        def __array__(self, *args, **kwargs):
            raise AssertionError("Must not materialise a lazy source")

    pipeline.node_outputs["input"] = [Lazy()]
    with pytest.raises(AssessmentError, match="already cached CPU image"):
        capture_assessment_snapshot(pipeline, "input", nodes, "run-7")


def test_oversized_and_ambiguous_output_is_explicit_not_treated_as_a_sample():
    pipeline, nodes = fixture()
    data = np.zeros((1, 5, 6, 7), dtype=bool)
    axes = (AxisMetadata("c", "channel"), *pipeline.output_states["input"].axes)
    commit_cache(pipeline, nodes[0], data, state_for(data, axes=axes))
    with pytest.raises(AssessmentError, match="no channel/time projection"):
        context_for(pipeline, nodes)


@pytest.mark.parametrize("run_id", ["", 3, "C:/private/run", "a" * 129])
def test_run_binding_is_a_bounded_opaque_identifier_not_a_path(run_id):
    pipeline, nodes = fixture()
    with pytest.raises(AssessmentError, match="Run ID"):
        capture_assessment_snapshot(pipeline, "input", nodes, run_id)


def test_uncalculated_ancillary_branch_is_disclosed_never_fabricated():
    pipeline, nodes = fixture(mesh=True)
    extra = pipeline.add_node("logical_and")
    assert pipeline.connect(nodes[0], extra.id, target_port=0).success
    assert pipeline.connect(nodes[0], extra.id, target_port=1).success
    review = pipeline.add_node("review_images")
    assert pipeline.connect(extra.id, review.id).success
    context = context_for(pipeline, (*nodes, extra.id, review.id))
    request = context.public_request("goal")
    assert request["uncalculated_nodes"][0]["node"] == "n3"
    absent = request["selected_outputs"][-1]
    assert absent["output_ref"] == {"node": "n3", "port": 0}
    assert absent["availability"] == "unavailable"
    assert "Not calculated" in absent["reason"]
    assert "diagnostics" not in absent
    assert extra.id not in pipeline.completed_node_ids
    assert pipeline.outputs[extra.id] is None


def test_not_calculated_with_old_cache_or_completed_marker_still_rejected():
    for cached in (False, True):
        pipeline, nodes = fixture(mesh=True)
        extra = pipeline.add_node("logical_and")
        assert pipeline.connect(nodes[0], extra.id, target_port=0).success
        assert pipeline.connect(nodes[0], extra.id, target_port=1).success
        if cached:
            commit_cache(pipeline, extra.id, pipeline.outputs[nodes[0]])
            pipeline.completed_node_ids.remove(extra.id)
        else:
            pipeline.completed_node_ids.add(extra.id)
        pipeline.node_execution_states[extra.id] = "not_calculated"
        with pytest.raises(AssessmentError, match="Run the complete candidate"):
            context_for(pipeline, (*nodes, extra.id))
