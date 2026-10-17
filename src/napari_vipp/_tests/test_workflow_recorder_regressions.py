"""Recorder mechanics only; synthetic objects never enter live model requests."""

from __future__ import annotations

import importlib
import json
import math
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    return (
        importlib.import_module("workflow_investigation_evidence"),
        importlib.import_module("run_workflow_feedback_refinement"),
    )


def test_ieee_nonfinite_evidence_roundtrip_and_reserved_collisions(modules):
    evidence, _ = modules
    nan = struct.unpack(">d", bytes.fromhex("7ff8000012345678"))[0]
    value = {
        "values": [nan, math.inf, -math.inf],
        "ordinary": {"$vipp_float64_bits": "original-text"},
        "escaped": {"$vipp_mapping_items": {"original": True}},
    }
    encoded = evidence.recording_document(value)
    evidence.json_bytes(encoded)
    decoded = evidence.decode_recording_document(encoded)
    assert struct.pack(">d", decoded["values"][0]).hex() == "7ff8000012345678"
    assert decoded["values"][1:] == [math.inf, -math.inf]
    assert decoded["ordinary"] == value["ordinary"]
    assert decoded["escaped"] == value["escaped"]
    collision = {
        "kind": "napari-vipp-nonfinite-recording",
        "version": 1,
        "encoding": "ieee754-binary64-hex",
        "value": {"ordinary": 4},
    }
    assert (
        evidence.decode_recording_document(evidence.recording_document(collision))
        == collision
    )
    assert evidence.recording_document({"ordinary": 4}) == {"ordinary": 4}
    with pytest.raises(ValueError):
        evidence.json_bytes(
            {"provider": math.nan}
        )  # Provider/public JSON remains strict.
    bad = {**encoded, "version": 2}
    with pytest.raises(ValueError, match="Unsupported"):
        evidence.decode_recording_document(bad)
    with pytest.raises(ValueError, match="Unsupported"):
        evidence.decode_recording_document({**encoded, "version": True})


def _completion(evidence):
    binding = evidence.OwnedReplayBinding(
        ("science", "review"), (("science", "science"), ("review", "review")), "hash"
    )
    pipeline = SimpleNamespace(
        nodes={
            "science": SimpleNamespace(operation_id="binary_threshold"),
            "review": SimpleNamespace(operation_id="review_images"),
        },
        node_execution_states={"science": "ready", "review": "not_calculated"},
        completed_node_ids={"science"},
        node_outputs={"science": [np.ones(2)], "review": [None]},
        outputs={"science": np.ones(2), "review": None},
    )
    return binding, pipeline


def test_only_catalogued_noncomputing_review_nodes_are_exceptions(modules):
    evidence, _ = modules
    binding, pipeline = _completion(evidence)
    registry = {
        "review_images": SimpleNamespace(presentation_only=True, output_type="none")
    }
    result = evidence.owned_completion_details(binding, pipeline, registry=registry)
    assert result["status"] == "ready" and result["presentation_exceptions"] == [
        "review"
    ]
    pipeline.node_execution_states["science"] = "not_calculated"
    assert (
        evidence.owned_completion_details(binding, pipeline, registry=registry)[
            "status"
        ]
        == "waiting"
    )
    pipeline.node_execution_states["science"] = "ready"
    pipeline.node_execution_states["review"] = "error"
    assert (
        evidence.owned_completion_details(binding, pipeline, registry=registry)[
            "status"
        ]
        == "failed"
    )
    pipeline.node_execution_states["review"] = "not_calculated"
    pipeline.node_outputs["review"] = [np.ones(2)]
    with pytest.raises(ValueError, match="scientific cache"):
        evidence.owned_completion_details(binding, pipeline, registry=registry)
    pipeline.node_outputs["review"] = [None]
    with pytest.raises(ValueError, match="catalogue-confirmed"):
        evidence.owned_completion_details(binding, pipeline, registry={})
    pipeline.nodes["review"].operation_id = "unknown_presentation"
    assert (
        evidence.owned_completion_details(binding, pipeline, registry=registry)[
            "status"
        ]
        == "waiting"
    )


def test_capture_full_state_after_real_mesh_and_nonfinite_table(
    modules, monkeypatch, tmp_path
):
    evidence, _ = modules
    driver = importlib.import_module("run_workflow_investigation_acceptance")
    from napari_vipp.core.meshes import MeshData, MeshObject, MeshState
    from napari_vipp.core.metadata import AxisMetadata
    from napari_vipp.core.tables import TableData, TableState

    axes = tuple(
        AxisMetadata(name=axis, type="space", scale=0.45, unit="micrometer")
        for axis in "ZYX"
    )
    state = MeshState(
        3,
        1,
        axes,
        (2, 2, 2),
        "Close at image border",
        history=("synthetic provenance",),
    )
    objects = (
        MeshObject(
            1, "owned object", (0.2, 0.4, 0.8, 1), source_history=("full provenance",)
        ),
    )
    mesh = MeshData(
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[0, 1, 2]]),
        state,
        objects=objects,
    )
    table = TableData(
        ("nan", "positive", "negative"),
        ((math.nan, math.inf, -math.inf),),
        name="quality",
        source_name="preserve this",
    )
    table_state = TableState(1, 3, table.columns, history=("retain all metadata",))
    pipeline = SimpleNamespace(
        nodes={"mesh": object(), "table": object()},
        node_outputs={"mesh": [mesh], "table": [table]},
        node_output_states={"mesh": [state], "table": [table_state]},
        outputs={"mesh": mesh, "table": table},
        output_states={"mesh": state, "table": table_state},
        completed_node_ids={"mesh", "table"},
        node_execution_states={"mesh": "ready", "table": "ready"},
        node_execution_messages={},
        node_compute_provenance={"mesh": {"complete": "provenance"}},
        node_cache_lineage={"table": "lineage"},
    )
    monkeypatch.setattr(
        "napari_vipp.core.workflow.serialize_workflow",
        lambda value: {"synthetic_graph": True},
    )
    journal = evidence.EvidenceJournal(tmp_path / "recorded")
    driver.capture_pipeline(journal, "complete", pipeline)
    document = json.loads(
        (journal.directory / "complete/full-state.json").read_text("utf-8")
    )
    decoded = evidence.read_recording_json(
        journal.directory / "complete/full-state.json"
    )
    assert document["kind"] == "napari-vipp-nonfinite-recording"
    mesh_record, table_record = decoded["ports"]
    assert mesh_record["mesh"]["objects"] == evidence.public_value(mesh.objects)
    assert mesh_record["state"] == evidence.public_value(state)
    assert (
        np.load(journal.directory / mesh_record["mesh"]["vertices"]["path"]).tobytes()
        == mesh.vertices.tobytes()
    )
    assert math.isnan(table_record["data"]["rows"][0][0])
    assert table_record["data"]["rows"][0][1:] == [math.inf, -math.inf]
    assert table_record["state"] == evidence.public_value(table_state)
    assert decoded["node_compute_provenance"] == pipeline.node_compute_provenance


def test_fresh_cleanup_and_final_ui_adoption_are_required(modules):
    evidence, runner = modules
    binding, pipeline = _completion(evidence)
    revision, initial = object(), object()
    widget = SimpleNamespace(
        pipeline=pipeline,
        _workflow_assistant_assessment_serial=2,
        _pipeline_run_serial=2,
        _workflow_assistant_run_revision=revision,
        _last_execution_report=SimpleNamespace(
            cleanup_succeeded=True, to_dict=lambda: {"cleanup_succeeded": True}
        ),
        _active_pipeline_run_id=None,
        _pipeline_run_pending=False,
        _pending_dirty_node_ids=set(),
        _pending_manual_node_ids=set(),
    )
    ticket = evidence.OwnedDispatchTicket(2, 2, "background-2", revision, initial)
    receipt = runner.execution_completion_receipt(widget, ticket, binding)
    assert receipt["cleanup_succeeded"] and receipt["presentation_exceptions"] == [
        "review"
    ]
    widget._last_execution_report = initial
    with pytest.raises(ValueError, match="fresh"):
        runner.execution_completion_receipt(widget, ticket, binding)
    widget._last_execution_report = SimpleNamespace(
        cleanup_succeeded=True, to_dict=lambda: {"cleanup_succeeded": True}
    )
    checks = []
    report = SimpleNamespace(
        run_id="background-2",
        assessment_id="accepted",
        to_dict=lambda: {"report": "unchanged"},
    )
    context, owner = SimpleNamespace(assessment_id="accepted"), object()
    reply = SimpleNamespace(
        context=context,
        owner=owner,
        choices=(True, True),
        result=SimpleNamespace(assessment=report, request_sha256="request"),
    )
    assessment = SimpleNamespace(
        reply=reply,
        run_id="background-2",
        _choices=lambda: (True, True),
        assert_current=lambda ctx, own: checks.append((ctx, own)),
    )
    loop = SimpleNamespace(
        active=False,
        attempted_runs=1,
        completed_rounds=1,
        history=[SimpleNamespace(assessment_id="accepted")],
    )
    dialog = SimpleNamespace(
        _busy=False, _assessment=assessment, _refinement=SimpleNamespace(loop=loop)
    )
    assert runner.final_assessment_adoption(
        dialog, widget, {"ticket": ticket, "binding": binding}
    )["idle_ui_adoption_verified"]
    assert checks == [(context, owner)]
    assessment._choices = lambda: (False, True)
    with pytest.raises(ValueError, match="exact owned run"):
        runner.final_assessment_adoption(
            dialog, widget, {"ticket": ticket, "binding": binding}
        )


def test_completion_capture_precedes_assessment_worker_and_fails_closed(modules):
    _evidence, runner = modules
    calls = []
    wrapped = runner.before_assessment_capture(
        lambda **kwargs: calls.append("worker"),
        lambda: calls.append("durable completion"),
    )
    wrapped(preview=False)
    assert calls == ["durable completion", "worker"]

    def denied():
        raise ValueError("No current cleanup")

    with pytest.raises(ValueError, match="cleanup"):
        runner.before_assessment_capture(
            lambda: calls.append("unauthorized worker"), denied
        )()
    assert "unauthorized worker" not in calls


def test_recording_reader_is_explicit_not_provider_decoder(modules, tmp_path):
    evidence, _runner = modules
    replay = importlib.import_module("replay_workflow_investigation_selection")
    path = tmp_path / "provider.json"
    original = {
        "kind": "napari-vipp-nonfinite-recording",
        "version": 1,
        "encoding": "ieee754-binary64-hex",
        "value": {"ordinary": 4},
    }
    path.write_text(json.dumps(original), encoding="utf-8")
    assert replay.read_json(path) == original
    journal = evidence.EvidenceJournal(tmp_path / "recorded")
    journal.document("scientific-state.json", {"measurement": math.nan})
    assert math.isnan(
        evidence.read_recording_json(journal.directory / "scientific-state.json")[
            "measurement"
        ]
    )


def test_prior_recipe_must_be_source_local_and_missing_metadata_not_claimed(modules):
    evidence, runner = modules
    baseline = importlib.import_module("workflow_feedback_baseline")
    graph = {
        "nodes": [{"id": "input"}, {"id": "old-context"}, {"id": "owned"}],
        "connections": [
            {"source": "input", "target": "owned"},
            {"source": "input", "target": "old-context"},
        ],
    }
    assert len(baseline.source_local_graph(graph, ["owned"])["nodes"]) == 2
    graph["connections"][0]["source"] = "old-context"
    with pytest.raises(ValueError, match="earlier branch"):
        baseline.source_local_graph(graph, ["owned"])
    data = np.arange(6).reshape(2, 3)
    role = {"role": "model-node:chosen", "port": 0}
    bundle = SimpleNamespace(
        recorded_buffers_only=True, expected=[(role, "old", {"array": {}}, data)]
    )
    pipe = SimpleNamespace(
        node_outputs={"new": [data.copy()]},
        node_output_states={"new": ["fresh metadata not old"]},
    )
    result = runner.exact_baseline_outputs(
        bundle, pipe, SimpleNamespace(node_mapping={"old": "new"})
    )
    assert result[0]["persisted_scientific_buffers_exactly_verified"] is True
    assert result[0]["prior_complete_metadata_exactly_verified"] is False
    pipe.node_outputs["new"][0][0, 0] = 1
    with pytest.raises(ValueError, match="array buffer"):
        runner.exact_baseline_outputs(
            bundle, pipe, SimpleNamespace(node_mapping={"old": "new"})
        )
