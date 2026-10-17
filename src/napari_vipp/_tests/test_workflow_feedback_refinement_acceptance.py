"""Pure mechanical checks for one-round feedback acceptance, no real model."""

from __future__ import annotations

import importlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def driver(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    return importlib.import_module("run_workflow_feedback_refinement")


def test_exact_original_goal_and_literal_new_feedback_required(driver):
    request = {
        "objective": driver.ORIGINAL_GOAL,
        "clarifications": [
            {"question": "Additional instructions 1", "answer": driver.USER_FEEDBACK}
        ],
    }
    driver.assert_feedback_request(request)
    for key, value in (("objective", "another objective"), ("clarifications", [])):
        changed = deepcopy(request)
        changed[key] = value
        with pytest.raises(ValueError, match="original goal"):
            driver.assert_feedback_request(changed)


def test_explicit_clarified_goal_does_not_silently_reuse_original(driver):
    goal, feedback = "Explicitly clarified user goal.", "Exact latest instruction."
    request = {
        "objective": goal,
        "clarifications": [
            {"question": "Additional instructions 1", "answer": feedback}
        ],
    }
    driver.assert_feedback_request(request, feedback, goal)
    with pytest.raises(ValueError, match="original goal"):
        driver.assert_feedback_request(request, feedback)
    with pytest.raises(ValueError, match="original goal"):
        driver.assert_feedback_request(
            {**request, "objective": driver.ORIGINAL_GOAL}, feedback, goal
        )


def test_pinned_lock_binds_event_bytes_chain_count_and_terminal_hash(driver, tmp_path):
    event = {"sequence": 1, "previous_event_sha256": "0" * 64, "kind": "recorded"}
    data = (json.dumps(event) + "\n").encode()
    (tmp_path / "events.jsonl").write_bytes(data)
    lock = {
        "events_file_sha256": driver.sha256(data),
        "event_count": 1,
        "last_event_sha256": driver.sha256(driver.json_bytes(event)),
    }
    driver.verify_locked_events(tmp_path, lock)
    for key, value in (
        ("events_file_sha256", "0" * 64),
        ("event_count", 2),
        ("last_event_sha256", "0" * 64),
    ):
        changed = {**lock, key: value}
        with pytest.raises(ValueError, match="locked baseline event"):
            driver.verify_locked_events(tmp_path, changed)
    event["sequence"] = 2
    data = (json.dumps(event) + "\n").encode()
    (tmp_path / "events.jsonl").write_bytes(data)
    lock["events_file_sha256"] = driver.sha256(data)
    with pytest.raises(ValueError, match="event chain"):
        driver.verify_locked_events(tmp_path, lock)


def _graph():
    return {
        "nodes": [
            {"id": "input", "operation_id": "input", "params": {"path": "owned"}},
            {"id": "old", "operation_id": "gaussian_blur", "params": {"sigma": 2}},
            {"id": "new", "operation_id": "opening", "params": {"radius": 4}},
        ],
        "connections": [
            {"source": "input", "target": "old", "source_port": 0, "target_port": 0},
            {"source": "input", "target": "new", "source_port": 0, "target_port": 0},
        ],
    }


def test_additive_graph_preserves_old_and_new_exactly(driver):
    expected, actual = _graph(), _graph()
    actual["nodes"][2]["params"]["resolved_spatial_ndim"] = 3
    driver.verify_authored_graph(expected, actual, {"opening"})
    assert "resolved_spatial_ndim" in actual["nodes"][2]["params"]
    assert "resolved_spatial_ndim" not in expected["nodes"][2]["params"]
    for mutation in ("old", "new", "wire", "source", "rank"):
        changed = deepcopy(actual)
        if mutation == "old":
            changed["nodes"][1]["params"]["sigma"] = 3
        elif mutation == "new":
            changed["nodes"][2]["params"]["radius"] = 5
        elif mutation == "wire":
            changed["connections"][1]["source"] = "old"
        elif mutation == "source":
            changed["nodes"][0]["params"]["path"] = "another"
        else:
            changed["nodes"][2]["params"]["resolved_spatial_ndim"] = True
        with pytest.raises(ValueError, match="authored parameters"):
            driver.verify_authored_graph(expected, changed, {"opening"})


def test_additive_binding_requires_exact_model_keys_ids_and_wires(driver):
    graph = _graph()
    context = "abcdef1234567890"
    new_id = "proposal_abcdef123456_new"
    graph["nodes"][2]["id"] = new_id
    graph["connections"][1]["target"] = new_id
    recipe = {
        "context_id": context,
        "nodes": [{"key": "new", "operation_id": "opening", "params": {"radius": 4}}],
        "connections": [
            {"source": "input", "target": "new", "source_port": 0, "target_port": 0}
        ],
    }
    binding, mapping = driver.bind_additive_graph(
        recipe, graph, graph, [new_id], {"opening"}
    )
    assert mapping == {"new": new_id}
    assert binding.added_node_ids == (new_id,)
    assert binding.node_mapping["old"] == "old"
    for ids in ([], ["old"], [new_id, new_id]):
        with pytest.raises(ValueError, match="owned Add IDs"):
            driver.bind_additive_graph(recipe, graph, graph, ids, {"opening"})
    changed_recipe = deepcopy(recipe)
    changed_recipe["connections"][0]["source_port"] = 1
    with pytest.raises(ValueError, match="wiring"):
        driver.bind_additive_graph(changed_recipe, graph, graph, [new_id], {"opening"})


def test_exact_baseline_array_and_complete_state(driver):
    import numpy as np

    source = np.ones((2, 3), dtype=np.uint8)
    state = {"axes": "YX", "scale": [0.5, 0.5], "history": ["original"]}
    saved = {"array": {}, "state": state}
    role = {"role": "explicit-model-role", "port": 0}
    bundle = SimpleNamespace(expected=[(role, "old", saved, source)])
    pipeline = SimpleNamespace(
        node_outputs={"new": [source.copy()]},
        node_output_states={"new": [deepcopy(state)]},
    )
    binding = SimpleNamespace(node_mapping={"old": "new"})
    assert (
        driver.exact_baseline_outputs(bundle, pipeline, binding)[0]["role"]
        == role["role"]
    )
    pipeline.node_output_states["new"][0]["history"] = []
    with pytest.raises(ValueError, match="full output metadata"):
        driver.exact_baseline_outputs(bundle, pipeline, binding)


def test_six_actual_turns_are_stopped_before_transport(driver, monkeypatch, tmp_path):
    from napari_vipp.core import workflow_proposal_codex as codex

    calls = []
    monkeypatch.setattr(
        codex, "_call_codex", lambda *args, **kwargs: calls.append(args) or "reply"
    )
    journal = driver.EvidenceJournal(tmp_path / "evidence")
    observer = driver.FeedbackObservers(journal)
    observer.install()
    try:
        for _ in range(6):
            assert codex._call_codex({}, (), {}, None, None, None) == "reply"
        with pytest.raises(RuntimeError, match="six-turn"):
            codex._call_codex({}, (), {}, None, None, None)
        assert len(calls) == 6
        assert observer.counts["model_calls"] == 6
    finally:
        observer.close()


def test_changed_dialogue_is_rejected_before_provider(driver, monkeypatch, tmp_path):
    from napari_vipp.core import workflow_assessment_model as assessment

    calls = []
    monkeypatch.setattr(
        assessment,
        "generate_codex_assessment",
        lambda *args, **kwargs: calls.append(args),
    )
    observer = driver.FeedbackObservers(driver.EvidenceJournal(tmp_path / "evidence"))
    observer.install()
    try:
        context = SimpleNamespace(
            public_request=lambda objective: {
                "objective": objective,
                "clarifications": [],
            }
        )
        with pytest.raises(ValueError, match="original goal"):
            assessment.generate_codex_assessment(context, driver.ORIGINAL_GOAL, {})
        assert not calls
        assert observer.counts["model_calls"] == 0
    finally:
        observer.close()


def test_result_record_never_serializes_private_context_or_png_bytes(driver):
    result = SimpleNamespace(
        assessment={"overall": "partly_achieved"},
        attempts=(),
        provider="codex",
        model="recorded",
        request_sha256="request-hash",
        context=SimpleNamespace(image_previews=(b"private original PNG bytes",)),
    )
    record = driver.provider_result_record(result)
    assert record == {
        "assessment": {"overall": "partly_achieved"},
        "attempts": [],
        "provider": "codex",
        "model": "recorded",
        "request_sha256": "request-hash",
    }
    assert "context" not in record
    driver.json_bytes(record)


@pytest.fixture
def recovery_fixture(driver, tmp_path):
    from napari_vipp.core.workflow_proposal_images import ImagePreview

    preview = ImagePreview(
        "approved synthetic view", b"\x89PNG\r\n\x1a\nmechanicalfixture", "No science"
    )
    old = {
        "assessment_id": "old-assessment",
        "run_id": "old-run",
        "objective": driver.ORIGINAL_GOAL,
        "clarifications": [
            {"question": "Additional instructions 1", "answer": driver.USER_FEEDBACK}
        ],
        "image_previews": [preview.manifest()],
        "source": {"value": 1},
    }
    report = {
        "assessment_id": old["assessment_id"],
        "run_id": old["run_id"],
        "overall": "partly_achieved",
        "summary": "Observed synthetic result",
        "goal_checks": [
            {
                "criterion": "Review",
                "status": "partly_met",
                "evidence": "Recorded evidence",
            }
        ],
        "findings": [],
        "suggested_changes": [],
        "limitations": [],
        "comparison": {
            "status": "first_run",
            "previous_assessment_id": None,
            "explanation": "First view",
        },
    }
    original = driver.EvidenceJournal(tmp_path / "original")
    original.previews("requests/request-001", (preview,))
    recovery = {
        "root": original.directory,
        "request": old,
        "text": json.dumps(report),
        "original_response_sha256": "recorded-response",
        "model": "recorded",
    }
    current = {**deepcopy(old), "assessment_id": "new-assessment", "run_id": "new-run"}
    context = SimpleNamespace(
        assessment_id="new-assessment",
        run_id="new-run",
        _evidence={},
        image_previews=(preview,),
        public_request=lambda _: deepcopy(current),
    )
    return recovery, context, current, tmp_path


def test_recorded_assessment_rebinds_only_ids_with_same_evidence(
    driver, recovery_fixture
):
    recovery, context, current, tmp_path = recovery_fixture
    journal = driver.EvidenceJournal(tmp_path / "continued")
    result = driver.recorded_assessment_result(
        recovery, context, driver.ORIGINAL_GOAL, journal
    )
    assert result.assessment.assessment_id == "new-assessment"
    assert result.assessment.run_id == "new-run"
    assert result.assessment.summary == "Observed synthetic result"
    assert result.attempts == ()
    receipt = json.loads(
        (journal.directory / "recorded-assessment-reuse.json").read_text("utf-8")
    )
    assert receipt["changed_report_fields"] == ["assessment_id", "run_id"]
    assert not receipt["is_new_model_call"]
    assert receipt["current_png_bytes_identical"]
    assert current["source"] == {"value": 1}


def test_recorded_assessment_rejects_scientific_evidence_change(
    driver, recovery_fixture
):
    recovery, context, current, tmp_path = recovery_fixture
    current["source"]["value"] = 2
    with pytest.raises(ValueError, match="Fresh evidence differs"):
        driver.recorded_assessment_result(
            recovery,
            context,
            driver.ORIGINAL_GOAL,
            driver.EvidenceJournal(tmp_path / "continued"),
        )


def test_recorded_assessment_validates_original_identity_first(
    driver, recovery_fixture
):
    recovery, context, _current, tmp_path = recovery_fixture
    report = json.loads(recovery["text"])
    report["assessment_id"] = "wrong"
    recovery["text"] = json.dumps(report)
    with pytest.raises(ValueError, match="different captured run"):
        driver.recorded_assessment_result(
            recovery,
            context,
            driver.ORIGINAL_GOAL,
            driver.EvidenceJournal(tmp_path / "continued"),
        )


def test_original_deadline_blocks_transport_before_new_call(
    driver, monkeypatch, tmp_path
):
    from napari_vipp.core import workflow_proposal_codex as codex

    calls = []
    monkeypatch.setattr(
        codex, "_call_codex", lambda *args, **kwargs: calls.append(args)
    )
    observer = driver.FeedbackObservers(
        driver.EvidenceJournal(tmp_path / "evidence"),
        {"original_deadline_utc": driver.time.time() - 1},
    )
    observer.install()
    try:
        with pytest.raises(RuntimeError, match="original refinement deadline"):
            codex._call_codex({}, (), {}, None, None, None)
        assert not calls
        assert observer.counts["model_calls"] == 1
        assert observer.historical_model_calls == 1
    finally:
        observer.close()
