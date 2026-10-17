"""Algorithm-neutral acceptance observers; fake boundaries, no Qt or model calls."""

from __future__ import annotations

import importlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def observed_product(monkeypatch, tmp_path):
    scripts = Path(__file__).resolve().parents[3] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    driver = importlib.import_module("run_workflow_investigation_acceptance")
    support = importlib.import_module("workflow_investigation_evidence")
    from napari_vipp.core import workflow_investigation as engine

    journal = support.EvidenceJournal(tmp_path / "evidence")
    observers = driver.ProductObservers(journal)
    return engine, journal, observers


def _document(journal, path):
    return json.loads((journal.directory / path).read_text("utf-8"))


def test_observer_captures_new_actions_ledger_and_exact_previews_unchanged(
    observed_product, monkeypatch
):
    engine, journal, observers = observed_product
    preview = SimpleNamespace(
        png_bytes=b"original-product-preview", manifest=lambda: {"kind": "source-roi"}
    )
    called = []

    class Session:
        candidates = ()
        finished = False
        stop_reason = ""
        retained_candidate_id = None

        def public_request(self):
            return {"experiment_ledger": [{"experiment_id": "e1", "state": "planned"}]}

        def perform(self, action):
            called.append(action)
            return SimpleNamespace(
                public_result=lambda: {"action": action, "evidence_ref": "issued-hash"},
                image_previews=(preview,),
            )

    monkeypatch.setattr(engine, "InvestigationSession", Session)
    session = Session()
    actions = [
        {"action": "describe_nodes", "operation_ids": ["ordinary-a", "ordinary-b"]},
        {"action": "plan_experiment", "experiment_id": "e1", "hypothesis": "test"},
        {"action": "inspect_source", "roi": {"Z": [0, 1], "Y": [1, 3], "X": [2, 4]}},
        {"action": "assess_experiment", "experiment_id": "e1", "candidate_id": "c1"},
    ]
    observers.install()
    try:
        for index, action in enumerate(actions, 1):
            result = session.perform(action)
            assert called[-1] is action
            assert result.image_previews[0] is preview
            prefix = f"actions/action-{index:03d}"
            assert _document(journal, f"{prefix}/action.json") == action
            assert _document(journal, f"{prefix}/result.json")["action"] == action
            assert _document(journal, f"{prefix}/session-request-after.json") == (
                session.public_request()
            )
            assert (
                journal.directory / f"{prefix}/result-previews/preview-01.png"
            ).read_bytes() == preview.png_bytes
        assert observers.counts == {
            "logical_requests": 0,
            "model_calls": 0,
            "actions": len(actions),
            "trial_executions": 0,
        }
    finally:
        observers.close()


def test_observer_preserves_compact_history_and_full_lossless_roundtrip(
    observed_product, monkeypatch
):
    from napari_vipp.core.workflow_investigation_history import (
        compact_history,
        expand_request_history,
    )

    engine, journal, observers = observed_product
    plan = {
        "baseline_candidate_id": None,
        "evidence_refs": [{"kind": "numeric", "sha256": "a" * 64, "alias": "input"}],
        "observed_failure": "Synthetic evidence only; no scientific target.",
        "hypothesis": "Model judgment retained verbatim.",
        "prior_assumptions": ["Explicit test assumption"],
        "predicted_geometric_change": "Synthetic predeclared prediction.",
        "risks": ["Synthetic risk"],
        "intent": "reconstruction",
        "evaluation_checks": [{"check_id": "check1", "criterion": "synthetic check"}],
    }
    assessment_request = {
        "experiment_id": "experiment1",
        "candidate_id": "candidate1",
        "check_results": [
            {
                "check_id": "check1",
                "outcome": "unresolved",
                "evidence_refs": [],
                "assessment": "Model interpretation, not a measured truth.",
            }
        ],
        "unresolved_criteria": ["No scientific success established."],
    }
    entry = {
        "experiment_id": "experiment1",
        "state": "assessed",
        "plan": plan | {"meaning": "Model judgment"},
        "candidate_id": "candidate1",
        "assessment": {
            "check_results": [
                {
                    **{
                        key: value
                        for key, value in result.items()
                        if key != "assessment"
                    },
                    "model_assessment": result["assessment"],
                }
                for result in assessment_request["check_results"]
            ],
            "unresolved_criteria": assessment_request["unresolved_criteria"],
        },
    }
    manifest = {
        "label": "Synthetic native view",
        "description": "Native crop; μm calibration; display transform; stated limits.",
        "format": "image/png",
        "byte_count": 123,
        "sha256": "b" * 64,
    }
    history = [
        {
            "action": "plan_experiment",
            "status": "ok",
            "request": plan,
            "result": entry
            | {"state": "planned", "candidate_id": None, "assessment": None},
            "image_previews": [manifest],
        },
        {
            "action": "assess_experiment",
            "status": "ok",
            "request": assessment_request,
            "result": entry,
            "image_previews": [manifest],
        },
        {
            "action": "future_action",
            "status": "error",
            "error": "Keep errors and unknown fields.",
            "unknown": {"ordered": [True, None, 0, 0.25, "μm"]},
            "image_previews": [manifest | {"future_field": "must stay inline"}],
        },
    ]
    before_history, before_ledger = deepcopy(history), deepcopy([entry])
    request = {
        "experiment_ledger": [entry],
        "image_previews": [],
        **compact_history(history, [entry]),
    }
    before_request = deepcopy(request)

    class Session:
        candidates = ()
        finished = False
        stop_reason = ""
        retained_candidate_id = None

        def public_request(self):
            return request

        def perform(self, _action):
            return SimpleNamespace(
                public_result=lambda: {"status": "ok"}, image_previews=()
            )

    monkeypatch.setattr(engine, "InvestigationSession", Session)
    observers.install()
    try:
        Session().perform({"action": "synthetic_observation"})
        saved = _document(journal, "actions/action-001/session-request-after.json")
        assert saved == before_request
        assert saved["history"][0]["request"]["$ref"] == "experiment_ledger"
        assert (
            saved["history"][0]["image_previews"][0]["$ref"]
            == "history_preview_manifests"
        )
        assert len(saved["history_preview_manifests"]) == 1
        assert expand_request_history(saved) == before_history
        assert saved["image_previews"] == []
        assert request == before_request
        assert history == before_history
        assert [entry] == before_ledger
    finally:
        observers.close()


def test_failed_action_preserves_pending_candidate_and_explicit_role_binding(
    observed_product, monkeypatch
):
    engine, journal, observers = observed_product
    failure = RuntimeError("ordinary dispatch failed")
    roles = [{"role": "requested-object", "node": "declared-node-key", "port": 1}]

    class Session:
        candidates = (
            SimpleNamespace(
                candidate_id="candidate1",
                evidence={"status": "attempted", "run_id": "run-1"},
                recipe={"nodes": []},
                output_roles=roles,
                graph=None,
            ),
        )

        def perform(self, _action):
            raise failure

    monkeypatch.setattr(engine, "InvestigationSession", Session)
    observers.install()
    try:
        with pytest.raises(RuntimeError) as caught:
            Session().perform({"action": "trial_candidate"})
        assert caught.value is failure
        prefix = "actions/action-001/candidates/candidate1"
        assert _document(journal, f"{prefix}/output-roles.json") == roles
        assert _document(journal, f"{prefix}/candidate.json")["status"] == "attempted"
    finally:
        observers.close()


def test_next_request_failure_preserves_returned_action_and_candidate_artifacts(
    observed_product, monkeypatch
):
    engine, journal, observers = observed_product
    failure = ValueError("bounded request cannot be serialized")
    roles = [{"role": "declared-output", "node": "node-key", "port": 0}]
    preview = SimpleNamespace(
        png_bytes=b"completed-action-preview", manifest=lambda: {"kind": "result"}
    )
    completed = {
        "action": "trial_candidate",
        "status": "ok",
        "result": {"candidate_id": "candidate1", "status": "completed"},
    }

    class Session:
        candidates = (
            SimpleNamespace(
                candidate_id="candidate1",
                evidence={"status": "completed", "run_id": "run-1"},
                recipe={"nodes": []},
                output_roles=roles,
                graph=None,
            ),
        )
        finished = False
        stop_reason = ""
        retained_candidate_id = None

        def public_request(self):
            raise failure

        def perform(self, _action):
            return SimpleNamespace(
                public_result=lambda: completed,
                image_previews=(preview,),
            )

    monkeypatch.setattr(engine, "InvestigationSession", Session)
    observers.install()
    try:
        with pytest.raises(ValueError) as caught:
            Session().perform({"action": "trial_candidate"})
        assert caught.value is failure
        prefix = "actions/action-001"
        assert _document(journal, f"{prefix}/result.json") == completed
        assert _document(journal, f"{prefix}/candidates/candidate1/candidate.json") == {
            "status": "completed",
            "run_id": "run-1",
        }
        assert (
            _document(journal, f"{prefix}/candidates/candidate1/output-roles.json")
            == roles
        )
        assert (
            journal.directory / f"{prefix}/result-previews/preview-01.png"
        ).read_bytes() == preview.png_bytes
        events = [
            json.loads(line)
            for line in (journal.directory / "events.jsonl")
            .read_text("utf-8")
            .splitlines()
        ]
        assert [event["kind"] for event in events] == [
            "product_action",
            "product_action_returned",
            "next_request_capture_failed",
        ]
        assert events[-1]["details"]["action_returned"] is True
        assert not (journal.directory / f"{prefix}/session-request-after.json").exists()
        assert not (journal.directory / "selected-result-lock.json").exists()
    finally:
        observers.close()


def test_failed_executor_attempt_is_counted_and_journalled_without_substitution(
    observed_product, monkeypatch
):
    engine, journal, observers = observed_product
    failure = RuntimeError("ordinary execution failed")

    def execute(_request):
        raise failure

    monkeypatch.setattr(engine, "execute_pipeline_request", execute)
    observers.install()
    try:
        request = SimpleNamespace(workflow={"nodes": [], "connections": []})
        with pytest.raises(RuntimeError) as caught:
            engine.execute_pipeline_request(request)
        assert caught.value is failure
        assert observers.counts["trial_executions"] == 1
        assert _document(journal, "trials/trial-001/workflow-request.json") == (
            request.workflow
        )
        events = (journal.directory / "events.jsonl").read_text("utf-8").splitlines()
        assert json.loads(events[-1])["kind"] == "trial_execution_failed"
        assert not (
            journal.directory / "trials/trial-001/execution-result.json"
        ).exists()
    finally:
        observers.close()


def test_trial_port_lookup_uses_nominated_candidate_and_explicit_port(observed_product):
    _engine, _journal, observers = observed_product
    first = {
        "node_id": "same-node",
        "port": 1,
        "mesh": {"objects": [{"name": "first"}]},
    }
    latest = {
        "node_id": "same-node",
        "port": 1,
        "mesh": {"objects": [{"name": "latest"}]},
    }
    observers.trial_ports_by_candidate.update(candidate1=[first], candidate2=[latest])
    assert observers.trial_port("candidate1", "same-node", 1) is first
    with pytest.raises(ValueError, match="exact executed port"):
        observers.trial_port("candidate1", "same-node", 0)
    with pytest.raises(ValueError, match="exact executed port"):
        observers.trial_port("missing", "same-node", 1)


def test_successful_action_binds_its_own_executed_manifest_before_later_trials(
    observed_product, monkeypatch
):
    engine, _journal, observers = observed_product
    driver = importlib.import_module("run_workflow_investigation_acceptance")

    def execute(request):
        from napari_vipp.core.pipeline import PrototypePipeline

        pipeline = PrototypePipeline()
        pipeline.reset_empty_graph()
        pipeline._fixture_trial_number = request.run_id
        return SimpleNamespace(
            run_id=request.run_id,
            workflow=request.workflow,
            error=None,
            cancelled=False,
            source_revisions={},
            execution_report={"cleanup_succeeded": True},
            failure=None,
            pipeline=pipeline,
        )

    monkeypatch.setattr(engine, "execute_pipeline_request", execute)
    monkeypatch.setattr(
        driver,
        "capture_pipeline",
        lambda _journal, _prefix, pipeline: [
            {
                "node_id": "mesh-node",
                "port": 0,
                "mesh": {"objects": [{"trial": pipeline._fixture_trial_number}]},
            }
        ],
    )

    class Session:
        candidates = ()
        finished = False
        stop_reason = ""
        retained_candidate_id = None

        def public_request(self):
            return {}

        def perform(self, action):
            engine.execute_pipeline_request(
                SimpleNamespace(run_id=action["number"], workflow={"nodes": []})
            )
            return SimpleNamespace(
                public_result=lambda: {
                    "status": "ok",
                    "result": {
                        "candidate_id": action["candidate_id"],
                        "status": "completed",
                    },
                },
                image_previews=(),
            )

    monkeypatch.setattr(engine, "InvestigationSession", Session)
    observers.install()
    try:
        session = Session()
        session.perform({"candidate_id": "candidate1", "number": 1})
        session.perform({"candidate_id": "candidate2", "number": 2})
        assert observers.trial_port("candidate1", "mesh-node", 0)["mesh"][
            "objects"
        ] == [{"trial": 1}]
        assert observers.trial_port("candidate2", "mesh-node", 0)["mesh"][
            "objects"
        ] == [{"trial": 2}]
    finally:
        observers.close()
