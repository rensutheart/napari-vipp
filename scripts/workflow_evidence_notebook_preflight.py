"""Read-only qualified-core compatibility for the accepted notebook baseline."""

from __future__ import annotations

from copy import deepcopy

from workflow_evidence_notebook_observers import (
    NotebookObservers,
    assert_pure_export,
    public_bytes,
)
from workflow_investigation_evidence import sha256
from workflow_microedit_evidence import exact_baseline_replay, restore_recorded_port


def offline_notebook_compatibility(bundle, goal, feedback):
    """Restore recorded caches only; no reader, calculation, UI or provider call."""
    from napari_vipp.core.compute import ComputeRequest
    from napari_vipp.core.pipeline import PrototypePipeline
    from napari_vipp.core.workflow import deserialize_workflow
    from napari_vipp.core.workflow_assessment import capture_assessment_snapshot
    from napari_vipp.core.workflow_investigation import (
        InvestigationLimits,
        InvestigationSession,
        capture_completed_candidate,
    )
    from napari_vipp.core.workflow_investigation_model import _request
    from napari_vipp.core.workflow_proposals import ProposalDialogue
    from napari_vipp.core.workflow_refinement import capture_preservation_constraint

    graph = deserialize_workflow(bundle.executed_graph)
    pipeline = PrototypePipeline()
    pipeline.restore_graph(
        graph["nodes"], graph["connections"], graph["output_tunnels"]
    )
    for node, port, saved, expected in bundle.expected:
        data, state = restore_recorded_port(saved, expected)
        values, states = pipeline.node_outputs[node], pipeline.node_output_states[node]
        if port != len(values):
            raise ValueError("Persisted baseline ports are not in exact port order.")
        values.append(data)
        states.append(state)
        if port == 0:
            pipeline.outputs[node], pipeline.output_states[node] = data, state
    pipeline.completed_node_ids = set(bundle.full_state["completed_node_ids"])
    for name in (
        "node_execution_states",
        "node_execution_messages",
        "node_compute_provenance",
        "node_cache_lineage",
    ):
        setattr(pipeline, name, deepcopy(bundle.full_state[name]))
    mapping = {node: node for node in bundle.key_to_node.values()}
    exact_baseline_replay(bundle, pipeline, mapping)
    ids = tuple(bundle.key_to_node.values())
    snapshot = capture_assessment_snapshot(
        pipeline, "input", ids, "offline-notebook-compatibility-not-owned"
    )
    constraint = capture_preservation_constraint(
        snapshot,
        (bundle.key_to_node["refine_reference_labels"], 0),
        (bundle.key_to_node["refine_circle_labels"], 0),
        output_sharing=True,
    )
    captured = capture_completed_candidate(
        pipeline,
        snapshot,
        bundle.roles,
        preservation_constraint=constraint,
        protected_role="band_mask",
    )
    detached = snapshot.baseline.to_pipeline()
    detached.outputs["input"] = pipeline.outputs["input"]
    detached.output_states["input"] = pipeline.output_states["input"]
    session = InvestigationSession.from_completed(
        detached,
        captured,
        ProposalDialogue(goal, (("Additional instructions 1", feedback),)),
        compute_request=ComputeRequest(),
        limits=InvestigationLimits(max_attempts=3),
        include_input_images=True,
        include_output_images=True,
        require_experiment_plan=True,
        assert_current=lambda: None,
    )
    before = assert_pure_export(session)
    request = session.public_request()
    checker = NotebookObservers(None)
    checker.preservation_contract = constraint.public_contract()
    checker.assert_request(request)
    _request(
        request, session.preview_images()
    )  # Strict production wire validation, NOT a provider call.
    after = assert_pure_export(session)
    if public_bytes(before) != public_bytes(after):
        raise ValueError(
            "Speculative request publication changed notebook evidence authority."
        )
    if session.attempted_trials or session.model_calls_used or session.actions_used:
        raise ValueError(
            "A read-only notebook preflight consumed live session authority."
        )
    return {
        "qualified_core_compatible": True,
        "native_run_ownership_proven": False,
        "cache_only_no_execution_or_provider": True,
        "exact_restored_ports": len(bundle.expected),
        "exact_restored_buffers": bundle.buffer_count,
        "accepted_node_count": len(ids),
        "preservation": constraint.public_contract(),
        "warm_start_binding": session.warm_start_binding,
        "key_to_node": session.candidates[0].key_to_node,
        "candidate0_retained": session.retained_candidate_id == "candidate0",
        "wire_version": request["version"],
        "wire_bytes": len(public_bytes(request)),
        "notebook_export_sha256": sha256(public_bytes(after)),
        "notebook_export_pure": True,
        "request_publication_grants_no_evidence_authority": True,
    }
