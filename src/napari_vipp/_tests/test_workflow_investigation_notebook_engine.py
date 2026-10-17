"""Finite working-context/actual-disclosure tests, synthetic provider-free data."""

import copy
from dataclasses import replace

import pytest

from napari_vipp._tests.test_workflow_investigation import act, session, trial
from napari_vipp.core.workflow_investigation import (
    InvestigationError,
    InvestigationLimits,
    InvestigationStopped,
    _json,
)


def send(engine):
    request = engine.public_request()
    engine.record_model_call(request=request)
    return request


def ref(descriptor):
    return {key: descriptor[key] for key in ("record_id", "sha256")}


def pending_experiment(engine):
    """A real tiny synthetic trial, not an invented pending-ledger placeholder."""
    from napari_vipp._tests.test_workflow_investigation import proposal

    send(engine)
    plan = act(
        engine,
        "plan_experiment",
        baseline_candidate_id=None,
        evidence_refs=[
            {"kind": "numeric", "alias": "input", "sha256": engine._source.fingerprint}
        ],
        observed_failure="Synthetic question.",
        hypothesis="A bounded threshold may help.",
        prior_assumptions=["A test only."],
        predicted_geometric_change="Unknown.",
        risks=["May be inaccurate."],
        intent="cleanup",
        evaluation_checks=[
            {"check_id": "coverage", "kind": "numeric", "criterion": "Unknown."}
        ],
    ).public_result()["result"]
    trial_result = act(
        engine,
        "trial_candidate",
        proposal=proposal(engine),
        output_roles=[{"role": "wanted", "node": "mask", "port": 0}],
        experiment_id=plan["experiment_id"],
    ).public_result()
    assert trial_result["status"] == "ok"
    candidate_id = trial_result["result"]["candidate_id"]
    inspected = act(
        engine, "inspect_candidate", candidate_id=candidate_id, role="wanted"
    ).public_result()["result"]
    send(engine)
    return plan["experiment_id"], candidate_id, inspected["evidence_ref"]


def test_v2_working_view_is_not_full_history_and_export_is_pure():
    engine = session()
    before = engine.export_notebook()
    request = engine.public_request()
    assert request["version"] == 2
    assert not {"history", "history_encoding", "history_shared_values"} & request.keys()
    assert request["evidence_notebook"]["record_index"] == before["record_index"]
    assert engine.export_notebook() == before
    assert engine._sent_numeric == engine._sent_previews == set()
    assert request["closing_phase"]["allowed_actions"] == sorted(
        request["response_schema"]["actions"]
    )


@pytest.mark.parametrize("field", ["objective", "source", "image_permissions"])
def test_actual_call_requires_exact_literal_outgoing_payload(field):
    engine = session()
    request = copy.deepcopy(engine.public_request())
    request[field] = "changed"
    with pytest.raises(InvestigationError, match="outgoing request"):
        engine.record_model_call(request=request)
    assert engine.model_calls_used == 0
    assert engine._sent_numeric == set()
    with pytest.raises(InvestigationError, match="exact outgoing request"):
        engine.record_model_call()


def test_only_seconds_may_vary_and_format_repair_reuses_exact_generation():
    engine = session()
    request = engine.public_request()
    request["budgets"]["seconds_remaining"] = 0.01
    engine.record_model_call(request=request)
    engine.record_model_call(request=request)  # Same actual frozen repair request.
    assert engine.model_calls_used == 2
    act(engine, "find_nodes", query="threshold")
    with pytest.raises(InvestigationError, match="outgoing request"):
        engine.record_model_call(request=request)
    assert engine.model_calls_used == 2


def test_generated_numeric_handle_is_not_issued_until_its_typed_payload_is_sent():
    engine = session()
    output = act(engine, "inspect_source").public_result()["result"]
    evidence = output["evidence_ref"]
    with pytest.raises(InvestigationError, match="not transmitted"):
        engine._validate_evidence_refs([evidence])
    engine.public_request()
    engine.export_notebook()
    assert (evidence["alias"], evidence["sha256"]) not in engine._sent_numeric
    send(engine)
    engine._validate_evidence_refs([evidence])


def test_index_history_pages_and_pinned_quoted_handles_never_mint_authority():
    engine = session()
    receipt = act(engine, "inspect_source").public_result()
    evidence = receipt["result"]["evidence_ref"]
    old_record = engine._latest_action_record
    # Persistent decision context legitimately supplies the latest typed source
    # measurement. Replace that alias with another actual ROI so the older
    # quoted/page/pinned measurement remains genuinely UNSUPPLIED.
    replacement = act(
        engine,
        "inspect_source",
        roi={"axis_order": "ZYX", "bounds": [[0, 1], [0, 2], [0, 2]]},
    ).public_result()["result"]
    act(engine, "find_nodes", query="threshold")
    act(engine, "pin_notebook", refs=[old_record])
    assert evidence not in [
        row["evidence_ref"]
        for row in engine.public_request()["decision_context"]["numeric_evidence"]
    ]
    send(engine)
    engine._validate_evidence_refs([replacement["evidence_ref"]])
    with pytest.raises(InvestigationError, match="not transmitted"):
        engine._validate_evidence_refs([evidence])
    page = act(engine, "read_notebook", **old_record, page=0).public_result()
    assert "evidence_ref" in page["result"]["text"]
    send(engine)
    with pytest.raises(InvestigationError, match="not transmitted"):
        engine._validate_evidence_refs([evidence])
    # An explicit real inspection is the way to supply the actual numeric payload.
    repeated = act(engine, "inspect_source").public_result()
    assert repeated["inspection_reused"] is True
    assert repeated["result"] == receipt["result"]
    send(engine)
    engine._validate_evidence_refs([evidence])


def test_cached_candidate_fingerprints_are_not_blanket_registered():
    engine = session()
    assert trial(engine).public_result()["status"] == "ok"
    candidate = engine.candidates[0]
    key = (
        f"{candidate.candidate_id}/wanted",
        dict(candidate._outputs)["wanted"].fingerprint,
    )
    act(engine, "find_nodes", query="threshold")
    send(engine)
    assert key not in engine._sent_numeric
    output = act(
        engine, "inspect_candidate", candidate_id=candidate.candidate_id, role="wanted"
    ).public_result()["result"]
    send(engine)
    engine._validate_evidence_refs(
        [output["evidence_ref"]], candidate_id=candidate.candidate_id
    )


def test_read_notebook_rejects_changed_hash_bool_page_and_other_session():
    engine = session()
    descriptor = engine.export_notebook()["record_index"][0]
    assert (
        act(engine, "read_notebook", **ref(descriptor), page=True).public_result()[
            "status"
        ]
        == "error"
    )
    assert (
        act(
            engine,
            "read_notebook",
            record_id=descriptor["record_id"],
            sha256="0" * 64,
            page=0,
        ).public_result()["status"]
        == "error"
    )
    other = session()
    result = act(engine, "inspect_source").public_result()["result"]
    with pytest.raises(InvestigationError, match="not transmitted"):
        other._validate_evidence_refs([result["evidence_ref"]])


def test_read_page_is_supplied_or_returns_explicit_atomic_unpin_error():
    engine = session(include_input_images=True)
    act(engine, "inspect_source", views=[{"axis": "Z", "indices": [0]}])
    preview = engine.preview_images()[0]
    # Seven valid synthetic PNG manifests make the exact receipt larger than
    # 32 KiB, but still within the unchanged 64 KiB complete working allowance.
    engine._images = tuple(
        replace(preview, label=f"synthetic {i}", description="Display note. " * 120)
        for i in range(7)
    )
    payload = {"help": ("\\" + "n~") * 9000}
    descriptor = engine._notebook.append("node_index", payload)
    pin = engine._notebook.append("node_index", {"contract": "x" * (31 * 1024)})
    act(engine, "pin_notebook", refs=[ref(pin)])
    pins_before = engine._notebook.pinned_record_refs
    denied = act(engine, "read_notebook", **ref(descriptor), page=0).public_result()
    assert denied["status"] == "error" and "Unpin" in denied["error"]
    assert "result" not in denied
    assert engine._notebook.pinned_record_refs == pins_before
    assert engine._notebook.lookup(**ref(descriptor)) == payload
    send(engine)  # The actionable error, not an omitted successful page.
    act(engine, "pin_notebook", refs=[])
    retrieved = act(engine, "read_notebook", **ref(descriptor), page=0).public_result()
    assert retrieved["status"] == "ok"
    assert len(_json(retrieved)) > 32 * 1024
    request = send(engine)
    assert any(
        row["record_id"] == engine._latest_action_record["record_id"]
        and row["payload"] == retrieved
        for row in request["evidence_notebook"]["working_records"]
    )


def test_same_exact_inspection_reuses_numeric_and_pixels_but_counts_each_send(
    monkeypatch,
):
    engine = session(include_input_images=True)
    calls = []
    original = engine._render_views

    def render(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "_render_views", render)
    options = {"views": [{"axis": "Z", "indices": [0]}]}
    first = act(engine, "inspect_source", **options).public_result()
    images = engine.preview_images()
    size = sum(len(p.png_bytes) for p in images)
    send(engine)
    second = act(engine, "inspect_source", **options).public_result()
    assert calls == [1]
    assert first["result"] == second["result"]
    assert second["inspection_reused"] is True
    assert engine.preview_images() == images
    send(engine)
    assert engine._image_bytes == 2 * size


def test_pinned_visual_board_persists_over_noninspection_and_is_atomic():
    engine = session(include_input_images=True)
    view_ids = []
    for index in range(7):
        result = act(
            engine, "inspect_source", views=[{"axis": "Y", "indices": [index]}]
        ).public_result()
        view_ids.append(result["result"]["view_id"])
    assert (
        act(engine, "pin_previews", view_ids=view_ids).public_result()["status"] == "ok"
    )
    before = engine.preview_images()
    assert len(before) == 7
    act(engine, "find_nodes", query="threshold")
    assert engine.preview_images() == before
    denied = act(
        engine, "inspect_source", views=[{"axis": "X", "indices": [0]}]
    ).public_result()
    assert denied["status"] == "error"
    assert engine._pinned_view_ids == tuple(view_ids)
    assert engine.preview_images() == before
    assert act(engine, "pin_previews", view_ids=[]).public_result()["status"] == "ok"
    assert engine._pinned_view_ids == ()
    # Clearing explicit pins leaves the documented automatic source context;
    # it neither rerenders nor changes scientific/transmission budgets.
    assert 0 < len(engine.preview_images()) <= 7
    assert engine.model_calls_used == 0 and engine._image_bytes == 0


def test_preview_cache_keys_bind_options_and_current_session():
    engine = session(include_input_images=True)
    first = act(
        engine, "inspect_source", views=[{"axis": "Z", "indices": [0]}]
    ).public_result()["result"]
    changed = act(
        engine, "inspect_source", views=[{"axis": "Z", "indices": [1]}]
    ).public_result()["result"]
    assert first["view_id"] != changed["view_id"]
    other = session(include_input_images=True)
    assert (
        act(other, "pin_previews", view_ids=[first["view_id"]]).public_result()[
            "status"
        ]
        == "error"
    )


def test_permission_change_cancellation_and_staleness_block_retrieval_and_pixels():
    engine = session(include_input_images=True)
    original = engine.export_notebook()
    engine.include_input_images = False
    assert engine.preview_images() == ()
    with pytest.raises(InvestigationStopped, match="permissions changed"):
        engine.public_request()
    assert engine.export_notebook() == original
    cancelled = session(include_input_images=True)
    cancelled.cancel()
    assert cancelled.preview_images() == ()
    with pytest.raises(InvestigationStopped):
        act(cancelled, "read_notebook", record_id="record1", sha256="0" * 64, page=0)


def test_sixty_four_unique_long_receipts_fit_without_deleting_any_journal_bytes():
    engine = session(limits=InvestigationLimits(max_actions=64, max_model_calls=80))
    expected = []
    for number in range(64):
        receipt = {
            "action": "describe_node",
            "status": "ok",
            "request": {"operation_id": "binary_threshold"},
            "result": {"unique_contract": f"{number}:" + "abcdefgh" * 3750},
            "image_previews": [],
        }
        encoded = _json(receipt)
        expected.append(encoded)
        engine._history.append(encoded)
        engine._latest_action_record = ref(
            engine._notebook.append(
                "action_receipt", receipt, owner={"action_index": number + 1}
            )
        )
    engine.actions_used = 64
    request = engine.public_request()
    assert len(_json(request)) < 128 * 1024
    assert engine._history == expected
    records = engine.export_notebook()["records"]
    assert [
        _json(r["payload"]) for r in records if "action" in r["payload"]
    ] == expected
    assert request["closing_phase"]["active"] is True
    assert "trial_candidate" not in request["response_schema"]["actions"]


def test_closing_phase_enforces_real_assessment_and_finish_slots():
    engine = session(limits=InvestigationLimits(max_actions=6))
    send(engine)
    plan = act(
        engine,
        "plan_experiment",
        baseline_candidate_id=None,
        evidence_refs=[
            {"kind": "numeric", "alias": "input", "sha256": engine._source.fingerprint}
        ],
        observed_failure="Synthetic question.",
        hypothesis="A bounded threshold may help.",
        prior_assumptions=["A test only."],
        predicted_geometric_change="Unknown.",
        risks=["May be inaccurate."],
        intent="cleanup",
        evaluation_checks=[
            {
                "check_id": "coverage",
                "kind": "numeric",
                "criterion": "Inspect real coverage.",
            }
        ],
    ).public_result()["result"]
    from napari_vipp._tests.test_workflow_investigation import proposal

    result = act(
        engine,
        "trial_candidate",
        proposal=proposal(engine),
        output_roles=[{"role": "wanted", "node": "mask", "port": 0}],
        experiment_id=plan["experiment_id"],
    ).public_result()
    assert result["status"] == "ok"
    candidate = result["result"]["candidate_id"]
    phase = engine.public_request()["closing_phase"]
    assert phase["active"] is True
    assert phase["pending_experiment_ids"] == [plan["experiment_id"]]
    with pytest.raises(InvestigationError, match="reserved"):
        act(engine, "trial_edit", candidate_id=candidate, edits=[])
    for _ in range(2):
        inspected = act(
            engine, "inspect_candidate", candidate_id=candidate, role="wanted"
        ).public_result()["result"]
        send(engine)
    assert (
        "inspect_candidate"
        not in engine.public_request()["closing_phase"]["allowed_actions"]
    )
    with pytest.raises(InvestigationError, match="reserved"):
        act(engine, "inspect_candidate", candidate_id=candidate, role="wanted")
    assert engine.finished is False
    assessed = act(
        engine,
        "assess_experiment",
        experiment_id=plan["experiment_id"],
        candidate_id=candidate,
        check_results=[
            {
                "check_id": "coverage",
                "outcome": "unresolved",
                "evidence_refs": [inspected["evidence_ref"]],
                "assessment": "Not established; no invented success.",
            }
        ],
        unresolved_criteria=["Coverage remains unvalidated."],
    ).public_result()
    assert assessed["status"] == "ok"
    finished = act(
        engine,
        "finish",
        candidate_id=candidate,
        summary="Explicit preference with unresolved evidence, not accuracy.",
        limitations=["Scientific coverage is unknown."],
    ).public_result()
    assert finished["status"] == "ok" and engine.finished is True
    assert engine.actions_used == 6
    assert engine.export_notebook()["records"]


def test_elapsed_reserve_blocks_new_experiments_without_extending_deadline():
    clock = [0.0]
    engine = session(clock=lambda: clock[0])
    deadline = engine.deadline
    clock[0] = 1700
    assert engine.public_request()["closing_phase"]["active"] is True
    with pytest.raises(InvestigationError, match="reserved"):
        act(engine, "find_nodes", query="threshold")
    assert engine.actions_used == 0 and engine.deadline == deadline


@pytest.mark.parametrize("format_attempts", [1, 2])
def test_closing_calls_reserve_two_format_attempts_without_rejecting_advertised_read(
    format_attempts,
):
    engine = session(limits=InvestigationLimits(max_actions=20, max_model_calls=20))
    experiment_id, candidate_id, evidence = pending_experiment(engine)
    # Every simulated earlier attempt is actually request-bound and charged.
    while engine.limits.max_model_calls - engine.model_calls_used > 8:
        send(engine)
    request = engine.public_request()
    assert request["closing_phase"]["reserved_model_calls"] == 6
    assert "inspect_candidate" in request["response_schema"]["actions"]
    for _ in range(format_attempts):
        engine.record_model_call(request=request)
    inspected = act(
        engine, "inspect_candidate", candidate_id=candidate_id, role="wanted"
    ).public_result()
    assert inspected["status"] == "ok"  # No request/perform threshold mismatch.
    # Four calls are enough for assessment + FINISH, each with one format repair.
    while engine.limits.max_model_calls - engine.model_calls_used > 4:
        send(engine)
    request = engine.public_request()
    assert "inspect_candidate" not in request["response_schema"]["actions"]
    assert "trial_candidate" not in request["response_schema"]["actions"]
    for _ in range(2):
        engine.record_model_call(request=request)
    assessed = act(
        engine,
        "assess_experiment",
        experiment_id=experiment_id,
        candidate_id=candidate_id,
        check_results=[
            {
                "check_id": "coverage",
                "outcome": "unresolved",
                "evidence_refs": [evidence],
                "assessment": "Unknown; no invented scientific pass.",
            }
        ],
        unresolved_criteria=["Coverage is unknown."],
    ).public_result()
    assert assessed["status"] == "ok"
    request = engine.public_request()
    assert "finish" in request["response_schema"]["actions"]
    for _ in range(2):
        engine.record_model_call(request=request)
    finished = act(
        engine,
        "finish",
        candidate_id=candidate_id,
        summary="Explicit preference, not accuracy.",
        limitations=["Coverage is unknown."],
    ).public_result()
    assert finished["status"] == "ok" and engine.finished
    assert engine.model_calls_used == engine.limits.max_model_calls


def test_after_charge_phase_cannot_widen_the_exact_outgoing_action_schema():
    engine = session(limits=InvestigationLimits(max_model_calls=8))
    send(engine)
    send(engine)
    request = engine.public_request()
    assert "trial_candidate" not in request["response_schema"]["actions"]
    engine.record_model_call(request=request)
    with pytest.raises(InvestigationError, match="exact outgoing request"):
        trial(engine)
    assert engine.attempted_trials == 0


def test_repair_capsule_does_not_ignore_changed_notebook_pins():
    engine = session()
    act(engine, "inspect_source")
    request = send(engine)
    engine._notebook.set_pins([engine._latest_action_record])
    with pytest.raises(InvestigationError, match="outgoing request"):
        engine.record_model_call(request=request)
    assert engine.model_calls_used == 1


def test_soft_context_pressure_keeps_fresh_numeric_needed_for_assessment(monkeypatch):
    import napari_vipp.core.workflow_investigation as investigation

    # Reduced test-only soft target forces the pressure branch; no cap is raised.
    monkeypatch.setattr(investigation, "WORKING_REQUEST_TARGET_BYTES", 4000)
    engine = session()
    result = act(engine, "inspect_source").public_result()["result"]
    request = engine.public_request()
    assert request["closing_phase"]["active"] is True
    assert any(
        row["payload"].get("action") == "inspect_source"
        for row in request["evidence_notebook"]["working_records"]
    )
    engine.record_model_call(request=request)
    engine._validate_evidence_refs([result["evidence_ref"]])


def test_every_ledger_check_and_contradiction_stays_visible_with_latest_projection():
    engine = session()
    fake_ref = {"kind": "numeric", "alias": "input", "sha256": "a" * 64}
    for number in range(2):
        plan = engine._experiments.plan(
            validate_evidence_refs=lambda *_a, **_kw: None,
            baseline_candidate_id=None,
            evidence_refs=[fake_ref],
            observed_failure="Synthetic ledger fixture, no scientific calculation.",
            hypothesis="Unproven.",
            prior_assumptions=["Unproven."],
            predicted_geometric_change="Unknown.",
            risks=["Unknown."],
            intent="cleanup",
            evaluation_checks=[
                {"check_id": "geometry", "kind": "numeric", "criterion": "Unknown."}
            ],
        )
        candidate_id = f"candidate{number + 1}"
        engine._experiments.bind_trial(plan["experiment_id"], candidate_id)
        engine._experiments.assess(
            experiment_id=plan["experiment_id"],
            candidate_id=candidate_id,
            check_results=[
                {
                    "check_id": "geometry",
                    "outcome": "contradicts",
                    "evidence_refs": [fake_ref],
                    "assessment": "Synthetic contradiction.",
                }
            ],
            unresolved_criteria=["Fixture only."],
            validate_evidence_refs=lambda *_a, **_kw: None,
        )
    engine._sync_notebook_records()
    request = engine.public_request()
    assert len(engine.experiment_ledger) == 2
    assert len(request["experiment_ledger"]) == 1
    assert len(request["experiment_index"]) == 2
    for header in request["experiment_index"]:
        assert header["checks"] == [
            {"check_id": "geometry", "kind": "numeric", "outcome": "contradicts"}
        ]
        full = engine._notebook.lookup(**header["full_record"])
        assert full["assessment"]["check_results"][0]["outcome"] == "contradicts"
