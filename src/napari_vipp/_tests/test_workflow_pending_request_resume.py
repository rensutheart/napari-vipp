"""Unsent-slot recovery mechanics, with no provider, UI or calculation."""

from __future__ import annotations

import ast
import importlib
import inspect
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    return (
        importlib.import_module("workflow_pending_request_resume"),
        importlib.import_module("run_workflow_feedback_refinement"),
    )


def test_actual_product_contract_tuple_is_json_equivalent_not_scientific_change(
    modules,
):
    from napari_vipp.core.grid import GridAxis, ImageGrid
    from napari_vipp.core.workflow_refinement import PreservationConstraint

    helper, runner = modules
    grid = ImageGrid(
        tuple(
            GridAxis(
                name=name,
                type="space",
                unit="um",
                scale=0.5,
                translation=0.0,
                size=2,
                explicit=True,
            )
            for name in "zyx"
        )
    )
    frozen = SimpleNamespace(
        _snapshot=SimpleNamespace(
            candidate_node_ids=("band", "region"),
            run_id="old",
            _source=SimpleNamespace(fingerprint="source"),
        ),
        primary_ref=("band", 0),
        region_ref=("region", 0),
        _grid=grid,
        _protected_hash="protected",
        constraint_id="issued",
        protected_count=3,
    )
    raw = PreservationConstraint.public_contract(frozen)
    recorded = json.loads(helper.strict_public_json(raw))
    assert isinstance(raw["grid"]["axes"], tuple)
    assert raw != recorded
    request = {
        "preservation_constraint": raw,
        "image_permissions": {"input_images": True, "output_images": True},
    }
    runner.assert_preservation_request(request, "assessment", 1, recorded)
    for key, value in (
        ("constraint_id", "other"),
        ("protected_count", 4),
        ("required_retention_fraction", True),
    ):
        with pytest.raises(ValueError, match="exact product"):
            runner.assert_preservation_request(
                {**request, "preservation_constraint": {**raw, key: value}},
                "assessment",
                1,
                recorded,
            )
    for nonfinite in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            helper.strict_public_json({"invalid": nonfinite})


def test_preprovider_validation_error_is_persisted_before_reraise(
    modules, monkeypatch, tmp_path
):
    from napari_vipp.core import workflow_assessment_model as provider

    _helper, runner = modules
    calls = []
    monkeypatch.setattr(
        provider, "generate_codex_assessment", lambda *a, **k: calls.append(a)
    )
    journal = runner.EvidenceJournal(tmp_path / "evidence")
    observer = runner.FeedbackObservers(journal, preservation_required=True)
    observer.preservation_contract = {"constraint_id": "expected"}
    context = SimpleNamespace(
        image_previews=(),
        public_request=lambda objective: {
            "objective": objective,
            "clarifications": [
                {
                    "question": "Additional instructions 1",
                    "answer": runner.USER_FEEDBACK,
                }
            ],
            "preservation_constraint": {"constraint_id": "other"},
            "image_permissions": {"input_images": True, "output_images": True},
        },
    )
    observer.install()
    try:
        with pytest.raises(ValueError, match="exact product"):
            provider.generate_codex_assessment(context, runner.ORIGINAL_GOAL, {})
    finally:
        observer.close()
    prefix = journal.directory / "requests/request-001"
    assert (prefix / "public-request.json").is_file()
    error = json.loads((prefix / "observer-validation-failure.json").read_text("utf-8"))
    assert error["type"] == "ValueError" and "exact product" in error["reason"]
    assert not calls and observer.counts["model_calls"] == 0


@pytest.fixture
def pending_fixture(modules, monkeypatch, tmp_path):
    helper, _runner = modules
    root, qualified = tmp_path / "failed", tmp_path / "qualified"
    root.mkdir()
    qualified.mkdir()
    qualification = {"wheel_sha256": "exact"}
    selection = {"authorized": "test-fixture-only"}
    configuration = {
        "authorized_model_key_selection": selection,
        "product_contract": {"constraint_id": "old", "protected_sha256": "same"},
        "primary_ref": ["old-band", 0],
        "region_ref": ["old-region", 0],
    }
    own = {"persisted_buffers_bound_including_source": 41}
    recipe = {"context_id": "old", "nodes": [{"key": "fixture"}]}
    graph = {"nodes": [{"id": "input", "params": {"path": "owned"}}]}
    bundle = SimpleNamespace(
        own_model_baseline=own,
        recipe={**recipe, "context_id": "new"},
        prepared_graph=graph,
        expected=["one-fixture-output"],
    )
    protocol = dict(
        requested_goal="goal",
        new_user_feedback="feedback",
        requested_preservation_selection=selection,
        include_input_previews=True,
        include_output_previews=True,
        maximum_logical_model_requests=3,
        maximum_actual_model_turns=6,
        maximum_additional_scientific_runs=1,
        maximum_refinement_seconds=1800,
        configured_per_attempt_timeout_seconds=600,
        qualified_wheel_sha256="exact",
        own_model_baseline=own,
        qualified_source=str(qualified),
    )
    receipt = dict(
        status="incomplete",
        normal_runs=1,
        additional_run_attempts=0,
        completed_refinement_assessments=0,
        provider_counts={"logical_requests": 1, "model_calls": 0},
        loop_history=[],
        preservation={"configuration": configuration, "product_result": None},
    )
    documents = {
        "receipt.json": receipt,
        "protocol.json": protocol,
        "preservation-configuration.json": configuration,
        "qualification-receipt.json": qualification,
        "baseline/proposal.json": recipe,
        "baseline/prepared-workflow.json": graph,
        "baseline/exact-output-verification.json": [
            {"persisted_scientific_buffers_exactly_verified": True}
        ],
    }
    for relative, value in documents.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(helper.strict_public_json(value))
    previous, events = "0" * 64, []
    start = "2026-10-16T20:15:57.297224+00:00"
    for number, kind in enumerate(
        ("normal_product_add", "normal_product_run", "explicit_one_round_start"), 1
    ):
        event = dict(
            sequence=number,
            previous_event_sha256=previous,
            utc=start,
            kind=kind,
            details={"user_feedback": "feedback"} if number == 3 else {},
        )
        events.append(event)
        previous = helper.sha256(helper.json_bytes(event))
    (root / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n", "utf-8"
    )
    monkeypatch.setattr(
        helper.time, "time", lambda: datetime.fromisoformat(start).timestamp() + 60
    )

    def load():
        digest = helper.sha256(helper.strict_public_json(helper.run_inventory(root)))
        return helper.load_pending_request(
            root,
            digest,
            qualified_source=qualified,
            qualification=qualification,
            bundle=bundle,
            objective="goal",
            feedback="feedback",
            preservation=selection,
        )

    return SimpleNamespace(root=root, load=load, documents=documents)


def test_pending_slot_retains_original_deadline_and_zero_actual_calls(pending_fixture):
    pending = pending_fixture.load()
    assert pending["historical_logical_reservations"] == 1
    assert pending["historical_actual_model_calls"] == 0
    assert pending["no_original_provider_payload_or_response_recorded"] is True
    assert (
        datetime.fromtimestamp(pending["original_deadline_utc"], UTC).isoformat()
        == "2026-10-16T20:45:57.297224+00:00"
    )


@pytest.mark.parametrize(
    "change",
    [
        "actual_call",
        "additional_run",
        "goal",
        "permission",
        "package",
        "scientific_parameter",
        "recorded_request",
        "expired",
    ],
)
def test_pending_resume_rejects_changed_authority(
    modules, pending_fixture, monkeypatch, change
):
    helper, _runner = modules
    fixture = pending_fixture
    if change == "expired":
        monkeypatch.setattr(helper.time, "time", lambda: float("inf"))
    elif change == "recorded_request":
        (fixture.root / "requests").mkdir()
        (fixture.root / "requests/request.json").write_text("{}", "utf-8")
    else:
        name = (
            "receipt.json"
            if change in {"actual_call", "additional_run"}
            else "protocol.json"
        )
        if change == "scientific_parameter":
            name = "baseline/proposal.json"
        value = deepcopy(fixture.documents[name])
        if change == "actual_call":
            value["provider_counts"]["model_calls"] = 1
        elif change == "additional_run":
            value["additional_run_attempts"] = 1
        elif change == "goal":
            value["requested_goal"] = "changed"
        elif change == "permission":
            value["include_input_previews"] = False
        elif change == "package":
            value["qualified_wheel_sha256"] = "changed"
        elif change == "scientific_parameter":
            value["nodes"][0]["threshold"] = 2
        (fixture.root / name).write_bytes(helper.strict_public_json(value))
    with pytest.raises(ValueError):
        fixture.load()


def test_resumed_protection_allows_only_admin_identity_rebinding(modules):
    helper, _runner = modules
    old = {
        "authorized_model_key_selection": {"explicit": "same"},
        "product_contract": {
            "constraint_id": "old",
            "baseline_run_id": "old-run",
            "protected_sha256": "same",
            "protected_count": 3,
        },
        "primary_ref": ["old-band", 0],
        "region_ref": ["old-region", 0],
    }
    current = deepcopy(old)
    current["product_contract"].update(
        constraint_id="fresh", baseline_run_id="fresh-run"
    )
    current["primary_ref"] = ["fresh-band", 0]
    current["region_ref"] = ["fresh-region", 0]
    receipt = helper.bind_resumed_preservation(old, current)
    assert receipt["changed_contract_fields"] == ["baseline_run_id", "constraint_id"]
    current["product_contract"]["protected_count"] = 2
    with pytest.raises(ValueError, match="scientific preservation"):
        helper.bind_resumed_preservation(old, current)


def test_pending_request_reuses_slot_one_then_only_slots_two_and_three(
    modules, monkeypatch, tmp_path
):
    from napari_vipp.core import workflow_assessment_model as assessment
    from napari_vipp.core import workflow_proposal_codex as codex

    _helper, runner = modules
    calls = []
    reply = SimpleNamespace(attempts=(), provider="codex", model="offline-fixture")
    monkeypatch.setattr(
        assessment,
        "generate_codex_assessment",
        lambda *a, **k: calls.append("assessment") or reply,
    )
    monkeypatch.setattr(
        codex,
        "generate_codex_proposal",
        lambda *a, **k: calls.append("revision") or reply,
    )
    context = SimpleNamespace(
        image_previews=(),
        public_request=lambda objective: {
            "objective": objective,
            "clarifications": [
                {
                    "question": "Additional instructions 1",
                    "answer": runner.USER_FEEDBACK,
                }
            ],
        },
    )
    observer = runner.FeedbackObservers(
        runner.EvidenceJournal(tmp_path / "evidence"),
        pending={"original_deadline_utc": runner.time.time() + 60},
    )
    assert observer.counts["logical_requests"] == 1
    observer.install()
    try:
        assessment.generate_codex_assessment(context, runner.ORIGINAL_GOAL, {})
        assert observer.pending_resume_used
        assert observer.counts["logical_requests"] == 1
        codex.generate_codex_proposal(context, runner.ORIGINAL_GOAL, {})
        assessment.generate_codex_assessment(context, runner.ORIGINAL_GOAL, {})
        assert observer.counts["logical_requests"] == 3
        with pytest.raises(RuntimeError, match="three-request"):
            assessment.generate_codex_assessment(context, runner.ORIGINAL_GOAL, {})
    finally:
        observer.close()
    assert calls == ["assessment", "revision", "assessment"]
    assert observer.counts["model_calls"] == 0  # Offline provider stubs do not infer.


def test_pending_deadline_is_checked_before_any_native_transport(
    modules, monkeypatch, tmp_path
):
    from napari_vipp.core import workflow_proposal_codex as codex

    _helper, runner = modules
    calls = []
    monkeypatch.setattr(codex, "_call_codex", lambda *a, **k: calls.append(a))
    observer = runner.FeedbackObservers(
        runner.EvidenceJournal(tmp_path / "evidence"),
        pending={"original_deadline_utc": 0},
    )
    observer.install()
    try:
        with pytest.raises(RuntimeError, match="deadline expired before transport"):
            codex._call_codex({}, (), {}, None, None, None)
    finally:
        observer.close()
    assert not calls and observer.counts["model_calls"] == 0


def test_resume_provenance_parameter_is_not_dispatch_pending(modules):
    _helper, runner = modules
    source = ast.parse(inspect.getsource(runner.run_live))
    function = source.body[0]
    assert "pending_resume" in {arg.arg for arg in function.args.args}
    stored = {
        node.id
        for node in ast.walk(function)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    assert "pending" in stored  # Separate ordinary Add/Run dispatch bookkeeping.
    assert "pending_resume" not in stored


def test_original_deadline_is_anchored_before_settings_worker(modules, monkeypatch):
    helper, runner = modules
    monkeypatch.setattr(helper.time, "time", lambda: 100.0)
    monkeypatch.setattr(helper, "monotonic", lambda: 10.0)
    loop = SimpleNamespace(deadline=1810.0, started_at=10.0)
    calls = []
    wrapped = runner.before_assessment_capture(
        lambda: calls.append(("settings-worker", loop.deadline)),
        lambda: helper.anchor_continuation_loop(loop, {"original_deadline_utc": 160.0}),
    )
    wrapped()
    assert calls == [("settings-worker", 70.0)]
    assert loop.started_at == -1730.0
    with pytest.raises(ValueError, match="deadline expired"):
        helper.anchor_continuation_loop(loop, {"original_deadline_utc": 99.0})
