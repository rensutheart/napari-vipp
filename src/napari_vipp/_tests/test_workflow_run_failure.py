"""Qualified failures are distinct from outputs and expose only safe recipe data."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest

from napari_vipp.core.execution import PipelineExecutionFailure
from napari_vipp.core.mesh_refinement import MeshInputTopologyError
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.pipeline import GraphConnection, OutputTunnel, PrototypePipeline
from napari_vipp.core.snapshots import GraphSnapshot, NodeSnapshot
from napari_vipp.core.workflow_proposals import capture_proposal_context
from napari_vipp.core.workflow_run_failure import (
    MAX_FAILURE_FEEDBACK_BYTES,
    RunFailureFeedback,
    build_run_failure_feedback,
)


def fixture(*, extra_nodes=(), extra_edges=()):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    data = np.zeros((3, 8, 9), np.uint8)
    data[:, 2:6, 2:7] = 100
    data.flags.writeable = False
    state = image_state_from_array(
        data,
        axes=tuple(AxisMetadata(axis, "space", "um", scale=0.45) for axis in "zyx"),
        source_name="private-source-name",
    )
    pipeline.outputs["input"] = data
    pipeline.output_states["input"] = state
    pipeline.nodes["input"].params["file_path"] = "C:/private/source-image.tif"
    context = capture_proposal_context(pipeline, "input")
    ids = ("private-mask-name", "private-mesh-name", "private-smooth-name")
    nodes = (
        NodeSnapshot(ids[0], "binary_threshold", {"threshold": 70.0}),
        NodeSnapshot(ids[1], "mask_to_3d_mesh", {"boundary": "Leave open"}),
        NodeSnapshot(ids[2], "smooth_mesh", {"iterations": 3, "strength": 0.2}),
        *extra_nodes,
    )
    failed_graph = GraphSnapshot(
        (*context.baseline.nodes, *nodes),
        (
            GraphConnection("input", ids[0]),
            GraphConnection(ids[0], ids[1]),
            GraphConnection(ids[1], ids[2]),
            *extra_edges,
        ),
    )
    ids += tuple(node.id for node in extra_nodes)
    failure = PipelineExecutionFailure(
        kind="execution_error",
        error_type="MeshInputTopologyError",
        message=str(MeshInputTopologyError("smooth_mesh", 2, "duplicate_faces", 2)),
        cleanup_succeeded=True,
    )
    return pipeline, context, failed_graph, ids, failure


def build(values, **overrides):
    _, context, failed_graph, ids, failure = values
    kwargs = dict(
        context=context,
        failed_graph=failed_graph,
        candidate_ids=ids,
        failed_node_ids=(ids[2],),
        run_id="background-8",
        failure=failure,
    )
    kwargs.update(overrides)
    return build_run_failure_feedback(**kwargs)


def test_only_safe_failed_recipe_and_known_diagnosis_are_public():
    values = fixture()
    feedback = build(values)
    public = feedback.public_feedback()
    assert public["status"] == "failed_no_assessment"
    assert public["run_id"] == "background-8"
    (failed,) = public["failed_nodes"]
    assert failed["node_key"] == "n3"
    assert failed["code"] == "duplicate_faces"
    assert failed["object_id"] == failed["count"] == 2
    assert len(public["failed_candidate"]["nodes"]) == 3
    assert public["failed_candidate"]["nodes"][2]["parameters"] == {
        "iterations": 3,
        "strength": 0.2,
    }
    assert public["failed_candidate"]["connections"][0] == {
        "source": "input",
        "target": "n1",
        "source_port": 0,
        "target_port": 0,
    }
    encoded = json.dumps(public)
    for private in (
        "private",
        "file_path",
        "source-image",
        "source_name",
        "source_fingerprint",
        "baseline_fingerprint",
        "image_previews",
        "assessment_id",
        "node_outputs",
    ):
        assert private not in encoded
    feedback.assert_current(values[0])


@pytest.mark.parametrize(
    "code",
    [
        "nonfinite_coordinates",
        "degenerate_faces",
        "nonmanifold_or_inconsistent_winding",
    ],
)
def test_other_known_input_mesh_codes_keep_unknown_counts_unavailable(code):
    values = fixture()
    failure = replace(
        values[4], message=str(MeshInputTopologyError("smooth_mesh", 2, code))
    )
    public = build(values, failure=failure).public_feedback()
    assert public["failed_nodes"][0]["code"] == code
    assert public["failed_nodes"][0]["count"] is None


def test_text_dynamic_choices_unknown_keys_and_nonfinite_values_are_redacted():
    secret = "C:/private/results.csv sk-private-api-key"
    values = fixture(
        extra_nodes=(
            NodeSnapshot(
                "private-note",
                "add_metadata_columns",
                {
                    "metadata_columns": secret,
                    "overwrite": "no",
                    secret: secret,
                },
            ),
            NodeSnapshot(
                "private-project",
                "project_image",
                {
                    "axes": "name:private-axis",
                    "method": "Maximum",
                },
            ),
            NodeSnapshot(
                "private-threshold",
                "binary_threshold",
                {
                    "threshold": float("nan"),
                    "foreground": secret,
                },
            ),
        )
    )
    public = build(values).public_feedback()
    note, project, threshold = public["failed_candidate"]["nodes"][3:]
    assert note["parameters"] == {"overwrite": "no"}
    assert note["redacted_parameters"] == ["metadata_columns"]
    assert project["parameters"] == {"method": "Maximum"}
    assert project["redacted_parameters"] == ["axes"]
    assert threshold["parameters"] == {}
    assert threshold["redacted_parameters"] == ["foreground", "threshold"]
    assert "private" not in json.dumps(public)
    assert "NaN" not in json.dumps(public)


@pytest.mark.parametrize("cleanup", [False, None, 1])
def test_unknown_or_failed_cleanup_never_grants_failure_feedback(cleanup):
    values = fixture()
    if cleanup == 1 and type(cleanup) is int:
        with pytest.raises(TypeError, match="cleanup"):
            replace(values[4], cleanup_succeeded=cleanup)
        return
    with pytest.raises(ValueError, match="not eligible"):
        build(values, failure=replace(values[4], cleanup_succeeded=cleanup))


@pytest.mark.parametrize(
    "kind",
    [
        "cancelled",
        "host_memory_oom",
        "memory_preflight",
        "compute_preflight",
        "provider_error",
        "security_error",
    ],
)
def test_resource_cancel_provider_or_security_failure_never_enters_repair(kind):
    values = fixture()
    with pytest.raises(ValueError, match="not eligible"):
        build(values, failure=replace(values[4], kind=kind))


@pytest.mark.parametrize(
    "changes",
    [
        {"retryable": True},
        {"runtime_id": "cuda"},
        {"segment_id": "segment-1"},
        {"error_type": "ValueError"},
        {"message": "unknown mesh diagnosis"},
        {"message": "C:/private/key.txt sk-api-key"},
        {"node_ids": ("input",)},
    ],
)
def test_unknown_or_untrusted_failures_are_not_public(changes):
    values = fixture()
    with pytest.raises(ValueError):
        build(values, failure=replace(values[4], **changes))


def test_appended_private_exception_text_is_rejected_not_sanitized_into_success():
    values = fixture()
    failure = replace(values[4], message=values[4].message + " C:/private/password.txt")
    with pytest.raises(ValueError, match="known safe"):
        build(values, failure=failure)


@pytest.mark.parametrize("run_id", ["", "C:/private/file", "has spaces", "a" * 129, 8])
def test_public_run_identity_must_be_a_bounded_safe_token(run_id):
    with pytest.raises(ValueError):
        build(fixture(), run_id=run_id)


@pytest.mark.parametrize("bad", [(), ("x", "x"), ["x"], ("x",) * 65])
def test_invalid_candidate_identity_sets_are_rejected(bad):
    with pytest.raises(ValueError, match="exact candidate"):
        build(fixture(), candidate_ids=bad)


@pytest.mark.parametrize("bad", [(), ("input",), ("unknown",), ("x", "x"), ["x"]])
def test_failure_must_belong_to_unique_nodes_of_current_candidate(bad):
    with pytest.raises(ValueError, match="exact candidate"):
        build(fixture(), failed_node_ids=bad)


def test_safe_diagnosis_from_different_operation_is_rejected():
    values = fixture()
    failure = replace(
        values[4],
        message=str(MeshInputTopologyError("simplify_mesh", 2, "duplicate_faces", 2)),
    )
    with pytest.raises(ValueError, match="known safe"):
        build(values, failure=failure)


def test_whole_graph_writer_or_other_source_refused_even_outside_candidate():
    values = fixture()
    for operation in ("save_output", "input", "custom_code"):
        bad = GraphSnapshot(
            (*values[2].nodes, NodeSnapshot("unrelated", operation, {})),
            values[2].connections,
        )
        with pytest.raises(ValueError, match="whole workflow"):
            build(values, failed_graph=bad)


def test_changed_source_or_baseline_parameters_cannot_be_hidden_by_rollback():
    values = fixture()
    source = values[2].nodes[0]
    changed = NodeSnapshot(
        source.id, source.operation_id, {**source.params, "file_path": "other"}
    )
    bad = GraphSnapshot((changed, *values[2].nodes[1:]), values[2].connections)
    with pytest.raises(ValueError, match="baseline changed"):
        build(values, failed_graph=bad)


def test_only_reviewed_derived_rank_cache_can_differ_from_rollback_baseline():
    from napari_vipp.core.workflow_proposals import _graph_fingerprint

    values = fixture()
    old = NodeSnapshot("old", "remove_small_objects", {"resolved_spatial_ndim": None})
    baseline = GraphSnapshot((*values[1].baseline.nodes, old))
    context = replace(
        values[1], baseline=baseline, baseline_fingerprint=_graph_fingerprint(baseline)
    )
    for resolved in (3, 4):
        failed = GraphSnapshot(
            (
                *values[2].nodes,
                NodeSnapshot(
                    "old", "remove_small_objects", {"resolved_spatial_ndim": resolved}
                ),
            ),
            values[2].connections,
        )
        if resolved == 3:
            assert build(values, context=context, failed_graph=failed)
        else:
            with pytest.raises(ValueError, match="baseline changed"):
                build(values, context=context, failed_graph=failed)


def test_candidate_tunnel_names_and_preexisting_output_tunnel_edits_are_refused():
    values = fixture()
    edges = (
        replace(values[2].connections[0], tunnel_name="private-tunnel"),
        *values[2].connections[1:],
    )
    with pytest.raises(ValueError, match="unrelated"):
        build(values, failed_graph=GraphSnapshot(values[2].nodes, edges))
    for target in ("input", values[3][0]):
        failed = GraphSnapshot(
            values[2].nodes,
            values[2].connections,
            (OutputTunnel("private-output-name", target),),
        )
        with pytest.raises(ValueError):
            build(values, failed_graph=failed)


def test_candidate_may_not_modify_or_depend_on_preexisting_processing_nodes():
    values = fixture()
    baseline = GraphSnapshot(
        (*values[1].baseline.nodes, NodeSnapshot("old", "invert", {}))
    )
    context = replace(values[1], baseline=baseline)
    from napari_vipp.core.workflow_proposals import _graph_fingerprint

    context = replace(context, baseline_fingerprint=_graph_fingerprint(baseline))
    failed = GraphSnapshot(
        (*values[2].nodes, baseline.nodes[-1]),
        (*values[2].connections, GraphConnection("old", values[3][0], 1)),
    )
    with pytest.raises(ValueError, match="unrelated node"):
        build(values, context=context, failed_graph=failed)
    failed = GraphSnapshot(
        (*values[2].nodes, baseline.nodes[-1]),
        (*values[2].connections, GraphConnection(values[3][0], "old")),
    )
    with pytest.raises(ValueError, match="pre-existing"):
        build(values, context=context, failed_graph=failed)


def test_graph_and_source_revisions_are_rechecked_before_feedback_use():
    values = fixture()
    feedback = build(values)
    source = values[0].outputs["input"].copy()
    source[0, 0, 0] = 1
    source.flags.writeable = False
    values[0].outputs["input"] = source
    with pytest.raises(ValueError, match="source changed"):
        feedback.assert_current(values[0])
    values = fixture()
    feedback = build(values)
    values[0].nodes["input"].params["file_path"] = "different.tif"
    with pytest.raises(ValueError, match="workflow changed"):
        feedback.assert_current(values[0])


def test_forged_context_and_mismatched_guard_bindings_are_rejected():
    values = fixture()
    with pytest.raises(TypeError, match="exact fresh"):
        build(
            values,
            context=SimpleNamespace(
                source_id=values[1].source_id,
                baseline=values[1].baseline,
                baseline_fingerprint=values[1].baseline_fingerprint,
                source_fingerprint=values[1].source_fingerprint,
            ),
        )
    with pytest.raises(ValueError, match="fresh rollback"):
        build(values, context=replace(values[1], baseline_fingerprint="0" * 64))
    feedback = build(values)
    for change in (
        {"source_id": "other"},
        {"source_fingerprint": "0" * 64},
        {"baseline_fingerprint": "0" * 64},
    ):
        with pytest.raises(ValueError, match="rollback context"):
            replace(feedback, **change)


def test_feedback_is_frozen_detached_and_contains_no_success_checkpoint():
    values = fixture()
    feedback = build(values)
    with pytest.raises(FrozenInstanceError):
        feedback.run_id = "changed"
    public = feedback.public_feedback()
    public["failed_nodes"][0]["safe_message"] = "C:/private/stolen"
    public["failed_candidate"]["nodes"][0]["parameters"]["threshold"] = 900
    assert feedback.public_feedback()["failed_candidate"]["nodes"][0]["parameters"] == {
        "threshold": 70.0
    }
    assert not hasattr(feedback, "snapshot")
    assert not hasattr(feedback, "assessment_id")
    assert not hasattr(feedback, "image_previews")


@pytest.mark.parametrize(
    "mutate",
    ["field", "text", "raw_id", "diagnosis", "noncanonical", "writer", "bool_count"],
)
def test_forged_public_json_cannot_leak_extra_text_or_bypass_known_diagnosis(mutate):
    feedback = build(fixture())
    public = feedback.public_feedback()
    if mutate == "field":
        public["private"] = "C:/password.txt"
    elif mutate == "text":
        public["failed_candidate"]["nodes"][0]["parameters"]["threshold"] = "secret"
    elif mutate == "raw_id":
        public["failed_candidate"]["nodes"][0]["key"] = "private-node-id"
    elif mutate == "diagnosis":
        public["failed_nodes"][0]["code"] = "invented"
    elif mutate == "writer":
        public["failed_candidate"]["nodes"][0]["operation_id"] = "save_output"
    elif mutate == "bool_count":
        public["failed_nodes"][0]["count"] = True
    payload = json.dumps(
        public, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    if mutate == "noncanonical":
        payload += " "
    with pytest.raises(ValueError):
        replace(feedback, _payload_json=payload)


def test_failure_recipe_budget_is_separate_and_never_truncated():
    values = fixture(
        extra_nodes=tuple(
            NodeSnapshot(f"large-{index}", "smooth_mesh", {"iterations": 10**1000})
            for index in range(61)
        )
    )
    assert len(values[3]) == 64
    with pytest.raises(ValueError, match="48 KiB"):
        build(values)
    assert MAX_FAILURE_FEEDBACK_BYTES == 48 * 1024


def test_constructor_validates_canonical_payload_instead_of_trusting_factory():
    feedback = build(fixture())
    with pytest.raises(ValueError, match="canonical"):
        RunFailureFeedback(
            feedback.run_id,
            feedback.source_id,
            feedback.baseline_fingerprint,
            feedback.source_fingerprint,
            feedback._context,
            "not json",
        )


def test_proposal_context_embeds_only_exact_detached_failed_run_feedback():
    values = fixture()
    feedback = build(values)
    context = replace(
        values[1],
        _failure_feedback=feedback.public_feedback(),
        _failure_feedback_guard=feedback,
    )
    public = context.public_request("Keep both structures separate.")
    assert public["calculation_failure"] == feedback.public_feedback()
    assert "result_feedback" not in public
    context.assert_current(values[0])
    public["calculation_failure"]["failed_nodes"][0]["count"] = 999
    assert (
        context.public_request("Keep both structures separate.")["calculation_failure"][
            "failed_nodes"
        ][0]["count"]
        == 2
    )
    changed = feedback.public_feedback()
    changed["failed_nodes"][0]["count"] = 999
    with pytest.raises(ValueError, match="exact failed-run"):
        replace(context, _failure_feedback=changed)
    with pytest.raises(ValueError, match="exact failed-run"):
        replace(context, source_fingerprint="0" * 64)
    with pytest.raises(ValueError, match="exact failed-run"):
        replace(context, _failure_feedback_guard=None)


def test_failure_feedback_cannot_be_recursively_built_from_output_backed_context():
    values = fixture()
    feedback = build(values)
    context = replace(
        values[1],
        _failure_feedback=feedback.public_feedback(),
        _failure_feedback_guard=feedback,
    )
    with pytest.raises(ValueError, match="fresh rollback"):
        build(values, context=context)


def test_private_feedback_mutation_after_context_capture_is_rejected_before_use():
    values = fixture()
    feedback = build(values)
    context = replace(
        values[1],
        _failure_feedback=feedback.public_feedback(),
        _failure_feedback_guard=feedback,
    )
    context._failure_feedback["failed_nodes"][0]["count"] = 999
    with pytest.raises(ValueError, match="feedback.*changed|changed.*feedback"):
        context.public_request("Keep both structures separate.")
    with pytest.raises(ValueError, match="feedback.*changed|changed.*feedback"):
        context.assert_current(values[0])
