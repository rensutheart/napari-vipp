"""Shared in-app/MCP tools are the same source-bound, finite native actions."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from napari_vipp.core.compute import ComputeRequest
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.pipeline import NODE_LIBRARY_BY_ID, PrototypePipeline
from napari_vipp.core.workflow_agent_probes import (
    MAX_PROBE_SNAPSHOT_BYTES,
    DiagnosticProbeError,
    validate_probe_context,
)
from napari_vipp.core.workflow_agent_tools import (
    WorkflowAgentTools,
    canonical_action_schema,
    tool_definitions_for_action_schema,
)
from napari_vipp.core.workflow_investigation import (
    InvestigationAction,
    InvestigationError,
    InvestigationLimits,
    InvestigationSession,
    InvestigationStopped,
)


def make_session(*, shape=(4, 8, 10), **kwargs):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape).copy()
    data.flags.writeable = False
    axes = tuple(
        AxisMetadata(
            axis.lower(), "space", "um", scale=index + 0.5, translation=10 + index
        )
        for index, axis in enumerate("YX" if len(shape) == 2 else "ZYX")
    )
    pipeline.outputs["input"] = data
    pipeline.output_states["input"] = image_state_from_array(
        data, axes=axes, source_name="PRIVATE name"
    )
    session = InvestigationSession(
        pipeline,
        "input",
        "A generic requested result.",
        compute_request=ComputeRequest(),
        assert_current=kwargs.pop("assert_current", lambda: None),
        include_output_images=kwargs.pop("include_output_images", True),
        **kwargs,
    )
    return pipeline, session, WorkflowAgentTools(session)


def threshold_proposal(session, thresholds=(100,)):
    nodes, edges, roles = [], [], []
    for index, threshold in enumerate(thresholds):
        key = f"mask{index}"
        params = {
            p.name: p.default for p in NODE_LIBRARY_BY_ID["binary_threshold"].parameters
        }
        params["threshold"] = threshold
        nodes.append(
            {
                "key": key,
                "operation_id": "binary_threshold",
                "params": params,
                "reason": "Explicit diagnostic choice.",
            }
        )
        edges.append(
            {"source": "input", "source_port": 0, "target": key, "target_port": 0}
        )
        roles.append({"role": key, "node": key, "port": 0})
    return {
        "kind": "napari-vipp-workflow-proposal",
        "version": 1,
        "context_id": session.proposal_context_id,
        "summary": "Explicit branched experiment.",
        "assumptions": ["A diagnostic, not wanted anatomy."],
        "questions": [],
        "nodes": nodes,
        "connections": edges,
    }, roles


def probe(tools, thresholds=(100,), *, roi=None, parent=None):
    proposal, roles = threshold_proposal(tools.session, thresholds)
    return tools.call_tool(
        "vipp_diagnostic_probe",
        {
            "input": parent or {"kind": "source"},
            "roi": roi or {"axis_order": "ZYX", "bounds": [[1, 3], [2, 7], [3, 9]]},
            "proposal": proposal,
            "output_roles": roles,
        },
    )


def test_tools_canonical_discovery_is_detached_and_has_no_session_argument():
    _, session, tools = make_session()
    definitions = tools.list_tools()
    assert {row["name"] for row in definitions} == {
        "vipp_" + name
        for name in session.public_request()["response_schema"]["actions"]
    }
    for row in definitions:
        assert set(row) == {"name", "description", "inputSchema"}
        assert row["inputSchema"]["additionalProperties"] is False
        assert "session_id" not in row["inputSchema"]["properties"]
    definitions[0]["name"] = "changed"
    assert tools.list_tools()[0]["name"] != "changed"
    assert canonical_action_schema() == tools.action_schema()
    assert tools.session_id == session.session_id
    assert "PRIVATE" not in json.dumps(definitions)


def test_tools_dispatch_exactly_one_validated_session_action(monkeypatch):
    _, session, tools = make_session()
    original, seen = session.perform, []

    def perform(action):
        seen.append(action)
        return original(action)

    monkeypatch.setattr(session, "perform", perform)
    result = tools.call_tool("vipp_find_nodes", {"query": "threshold"})
    assert result.public_result()["status"] == "ok"
    assert len(seen) == 1 and type(seen[0]) is InvestigationAction
    assert seen[0].session_id == session.session_id
    assert session.actions_used == 1
    with pytest.raises(InvestigationError):
        tools.call_tool(
            "vipp_find_nodes", {"query": "threshold", "session_id": "wrong"}
        )
    with pytest.raises(InvestigationError):
        tools.perform_action(
            InvestigationAction("wrong", "find_nodes", {"query": "threshold"})
        )
    assert session.actions_used == 1


@pytest.mark.parametrize(
    "name", ["shell", "vipp_shell", "vipp_save_output", "vipp_start_session", None]
)
def test_tools_do_not_expose_other_authority(name):
    _, session, tools = make_session()
    with pytest.raises(InvestigationError):
        tools.call_tool(name, {})
    assert session.actions_used == 0


def test_context_preparation_is_pure_actual_delivery_meters_and_grants():
    _, session, tools = make_session(include_input_images=True)
    transmission = tools.prepare_transmission()
    assert session.model_calls_used == 0 and not session._sent_previews
    transmission.request["objective"] = "changed"
    assert transmission.request["objective"] == session.dialogue.objective
    tools.record_transmission(transmission)
    assert session.model_calls_used == 1
    assert session._image_bytes == sum(len(p.png_bytes) for p in transmission.previews)
    assert session._sent_previews == {
        p.manifest()["sha256"] for p in transmission.previews
    }
    tools.record_transmission(transmission)
    assert session.model_calls_used == 2


def test_stale_and_foreign_transmission_does_not_grant_evidence():
    _, session, tools = make_session()
    transmission = tools.prepare_transmission()
    other = WorkflowAgentTools(session)
    with pytest.raises(InvestigationError):
        other.record_transmission(transmission)
    tools.call_tool("vipp_find_nodes", {"query": "threshold"})
    with pytest.raises(InvestigationError):
        tools.record_transmission(transmission)
    assert session.model_calls_used == 0


def test_concurrent_facades_share_nonblocking_operation_lease():
    _, session, tools = make_session()
    other = WorkflowAgentTools(session)
    entered, release = threading.Event(), threading.Event()

    def hold():
        with tools.exclusive():
            entered.set()
            assert release.wait(3)

    thread = threading.Thread(target=hold)
    thread.start()
    assert entered.wait(3)
    try:
        with pytest.raises(InvestigationError, match="busy"):
            other.call_tool("vipp_find_nodes", {"query": "threshold"})
        assert session.actions_used == 0
    finally:
        release.set()
        thread.join(3)
    assert (
        other.call_tool("vipp_find_nodes", {"query": "threshold"}).public_result()[
            "status"
        ]
        == "ok"
    )


def test_probe_runs_native_branched_graph_with_grid_and_no_candidate_promotion():
    pipeline, session, tools = make_session()
    before = pipeline.outputs["input"].copy()
    result = probe(tools, (100, 200)).public_result()
    assert result["status"] == "ok", result
    record = result["result"]
    assert record["status"] == "completed"
    assert record["input_grid"]["shape"] == [2, 5, 6]
    assert record["population"]["bounds"] == [[1, 3], [2, 7], [3, 9]]
    assert record["population"]["sample_coordinates"][0]["first_sample"] == 10.5
    expected = before[1:3, 2:7, 3:9]
    for role, binding, _operation in session._probes[0].outputs:
        threshold = 100 if role == "mask0" else 200
        np.testing.assert_array_equal(binding.data, expected > threshold)
        assert not binding.data.flags.writeable
    np.testing.assert_array_equal(pipeline.outputs["input"], before)
    assert len(pipeline.nodes) == 1
    assert session.attempted_probes == 1 and session.attempted_trials == 0
    assert not session.candidates and session.retained_candidate_id is None
    assert session.actions_used == 1 and session.model_calls_used == 0
    for name in ("vipp_retain_candidate", "vipp_finish"):
        arguments = (
            {"candidate_id": "probe1", "reason": "not candidate"}
            if name.endswith("retain_candidate")
            else {
                "candidate_id": "probe1",
                "summary": "not candidate",
                "limitations": [],
            }
        )
        assert tools.call_tool(name, arguments).public_result()["status"] == "error"
    assert not session.finished


def test_probe_exact_outputs_can_be_inspected_but_not_cited_before_delivery():
    _, session, tools = make_session()
    result = probe(tools).public_result()["result"]
    with pytest.raises(InvestigationError, match="not transmitted"):
        session._validate_evidence_refs([result["evidence_ref"]])
    assert validate_probe_context(session.public_request()["diagnostic_probes"])
    tools.record_transmission(tools.prepare_transmission())
    session._validate_evidence_refs([result["evidence_ref"]])
    inspection = tools.call_tool(
        "vipp_inspect_probe",
        {
            "probe_id": "probe1",
            "role": "mask0",
            "histogram": True,
            "views": [{"axis": "Z", "indices": [0, 1], "montage": True}],
        },
    )
    public = inspection.public_result()
    assert public["status"] == "ok", public
    assert public["result"]["diagnostics"]["element_count"] == 60
    assert inspection.image_previews
    assert "probe1/mask0" in inspection.image_previews[0].label
    with pytest.raises(InvestigationError):
        session._validate_evidence_refs([public["result"]["evidence_ref"]])
    tools.record_transmission(tools.prepare_transmission())
    session._validate_evidence_refs([public["result"]["evidence_ref"]])
    with pytest.raises(InvestigationError):
        session._validate_evidence_refs(
            [public["result"]["evidence_ref"]], candidate_id="candidate1"
        )


@pytest.mark.parametrize(
    "roi",
    [
        None,
        {"axis_order": "ZYX", "bounds": [[0, 5], [0, 8], [0, 10]]},
        {"axis_order": "XYZ", "bounds": [[0, 4], [0, 8], [0, 10]]},
    ],
)
def test_invalid_probe_roi_consumes_probe_not_candidate_and_produces_no_output(roi):
    _, session, tools = make_session()
    proposal, roles = threshold_proposal(session)
    result = tools.call_tool(
        "vipp_diagnostic_probe",
        {
            "input": {"kind": "source"},
            "roi": roi,
            "proposal": proposal,
            "output_roles": roles,
        },
    ).public_result()
    assert result["status"] == "error"
    assert result["probe_receipt"]["status"] == "failed"
    assert session.attempted_probes == 1 and session.attempted_trials == 0
    assert not session._probes[0].outputs
    assert validate_probe_context(session.public_request()["diagnostic_probes"])


def test_probe_attempts_have_independent_hard_limit_without_retry_reset():
    _, session, tools = make_session()
    for _ in range(6):
        assert probe(tools).public_result()["status"] == "ok"
    seventh = probe(tools).public_result()
    assert seventh["status"] == "error" and "six" in seventh["error"]
    assert session.attempted_probes == 6 and session.attempted_trials == 0
    assert session.public_request()["diagnostic_probes"]["attempts_remaining"] == 0


def test_probe_output_consent_cannot_be_bypassed_by_source_input():
    _, session, tools = make_session(include_output_images=False)
    result = probe(tools).public_result()
    assert result["status"] == "error"
    assert session.attempted_probes == 0


def test_probe_respects_original_closing_budget_and_cancellation():
    _, session, tools = make_session(limits=InvestigationLimits(max_actions=3))
    with pytest.raises(InvestigationError, match="reserved"):
        probe(tools)
    assert session.attempted_probes == 0
    _, session, tools = make_session()
    session.cancel()
    with pytest.raises(InvestigationStopped):
        probe(tools)


@pytest.mark.parametrize(
    "change",
    [
        {"maximum_attempts": 7},
        {"attempts_used": True},
        {"attempts_remaining": 5},
        {"max_input_voxels": 262145},
        {"snapshot_bytes_used": -1},
        {"index": [{}]},
        {"extra": 1},
    ],
)
def test_probe_context_strictly_rejects_mutated_limits_and_descriptors(change):
    _, session, _ = make_session()
    with pytest.raises(DiagnosticProbeError):
        validate_probe_context(session.public_request()["diagnostic_probes"] | change)


def test_tool_schema_cannot_advertise_a_non_native_action():
    schema = canonical_action_schema()
    schema["actions"]["shell"] = {}
    with pytest.raises(InvestigationError):
        tool_definitions_for_action_schema(schema)


def test_transmission_stale_check_is_pure_after_its_own_delivery():
    _, session, tools = make_session(include_input_images=True)
    transmission = tools.prepare_transmission()
    tools.assert_transmission_current(transmission)
    assert session.model_calls_used == 0
    tools.record_transmission(transmission)
    tools.assert_transmission_current(transmission)
    assert session.model_calls_used == 1
    tools.call_tool("vipp_find_nodes", {"query": "threshold"})
    with pytest.raises(InvestigationError, match="current"):
        tools.assert_transmission_current(transmission)
    assert session.model_calls_used == 1


def test_all_advertised_tools_have_valid_json_schemas():
    import jsonschema

    _, _, tools = make_session()
    for definition in tools.list_tools():
        jsonschema.Draft202012Validator.check_schema(definition["inputSchema"])


def test_probe_can_use_only_an_explicit_completed_candidate_role():
    _, session, tools = make_session()
    proposal, roles = threshold_proposal(session)
    trial = tools.call_tool(
        "vipp_trial_candidate",
        {
            "proposal": proposal,
            "output_roles": roles,
        },
    ).public_result()
    assert trial["status"] == "ok"
    tools.call_tool(
        "vipp_retain_candidate",
        {"candidate_id": "candidate1", "reason": "Explicit test baseline."},
    )
    result = probe(
        tools,
        thresholds=(0,),
        parent={"kind": "candidate", "candidate_id": "candidate1", "role": "mask0"},
    ).public_result()
    assert result["status"] == "ok", result
    parent = session.candidates[0]._outputs[0][1]
    output = session._probes[0].outputs[0][1]
    np.testing.assert_array_equal(output.data, parent.data[1:3, 2:7, 3:9])
    assert result["result"]["parent_sha256"] == parent.fingerprint
    assert session.attempted_trials == 1
    assert session.retained_candidate_id == "candidate1"


@pytest.mark.parametrize("nodes,outputs", [(9, 4), (5, 5)])
def test_probe_node_and_output_limits_reject_before_executor(
    monkeypatch, nodes, outputs
):
    from napari_vipp.core import workflow_agent_probes as module

    _, session, tools = make_session()
    calls = []
    monkeypatch.setattr(
        module, "execute_pipeline_request", lambda request: calls.append(request)
    )
    proposal, roles = threshold_proposal(session, thresholds=tuple(range(nodes)))
    result = tools.call_tool(
        "vipp_diagnostic_probe",
        {
            "input": {"kind": "source"},
            "roi": {"axis_order": "ZYX", "bounds": [[0, 4], [0, 8], [0, 10]]},
            "proposal": proposal,
            "output_roles": roles[:outputs],
        },
    ).public_result()
    assert result["status"] == "error" and not calls
    assert session.attempted_probes == 1 and session.attempted_trials == 0


def test_probe_input_voxel_and_cumulative_snapshot_caps_are_independent():
    _, session, tools = make_session(shape=(5, 256, 256))
    result = probe(
        tools, roi={"axis_order": "ZYX", "bounds": [[0, 5], [0, 256], [0, 256]]}
    ).public_result()
    assert result["status"] == "error" and "262144" in result["error"]
    assert session.attempted_probes == 1 and not session._probes[0].outputs
    session._probe_snapshot_bytes = MAX_PROBE_SNAPSHOT_BYTES
    result = probe(tools).public_result()
    assert result["status"] == "error" and "64 MiB" in result["error"]
    assert session.attempted_probes == 2 and not session._probes[1].outputs


@pytest.mark.parametrize(
    "failure",
    ["cancelled", "execution_error", "cleanup_unknown", "no_pipeline", "exception"],
)
def test_probe_executor_failure_or_unknown_cleanup_never_produces_evidence(
    monkeypatch, failure
):
    from napari_vipp.core import workflow_agent_probes as module

    _, session, tools = make_session()
    calls = []

    def execute(request):
        calls.append(request)
        if failure == "exception":
            raise RuntimeError("PRIVATE path C:/private/secret")
        return SimpleNamespace(
            cancelled=failure == "cancelled",
            error="error" if failure == "execution_error" else None,
            failure=None,
            execution_report=SimpleNamespace(
                cleanup_succeeded=None if failure == "cleanup_unknown" else True
            ),
            pipeline=None,
        )

    monkeypatch.setattr(module, "execute_pipeline_request", execute)
    with pytest.raises(InvestigationStopped) as caught:
        probe(tools)
    assert "PRIVATE" not in str(caught.value)
    assert len(calls) == 1 and calls[0].cancel_event is not None
    assert calls[0].compute_request == session.compute_request
    assert session.attempted_probes == 1 and session.attempted_trials == 0
    assert not session._probes[0].outputs and session._probes[0].status == "failed"


def test_changed_permission_stops_probe_and_old_transmission_before_new_work():
    _, session, tools = make_session()
    transmission = tools.prepare_transmission()
    session.include_output_images = False
    with pytest.raises(InvestigationStopped):
        probe(tools)
    with pytest.raises(InvestigationStopped):
        tools.assert_transmission_current(transmission)
    assert session.attempted_probes == 0 and session.model_calls_used == 0


def test_probe_rejects_writer_without_executing_or_loading_source(monkeypatch):
    from napari_vipp.core import workflow_agent_probes as module

    _, session, tools = make_session()
    calls = []
    monkeypatch.setattr(
        module, "execute_pipeline_request", lambda request: calls.append(request)
    )
    proposal, roles = threshold_proposal(session)
    proposal["nodes"][0]["operation_id"] = "save_output"
    proposal["nodes"][0]["params"] = {}
    result = tools.call_tool(
        "vipp_diagnostic_probe",
        {
            "input": {"kind": "source"},
            "roi": {"axis_order": "ZYX", "bounds": [[0, 4], [0, 8], [0, 10]]},
            "proposal": proposal,
            "output_roles": roles,
        },
    ).public_result()
    assert result["status"] == "error" and not calls
    assert session.attempted_probes == 1 and not session._probes[0].outputs
