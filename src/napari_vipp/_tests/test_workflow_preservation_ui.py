"""Normal guarded UI paths enforce optional preservation, with offline adapters."""

from types import SimpleNamespace

import pytest

from napari_vipp._tests.test_workflow_assessment_ui import (
    add_and_run,
)
from napari_vipp._tests.test_workflow_assessment_ui import (
    review as review,
)
from napari_vipp._tests.test_workflow_assessment_widget import (
    assessment_editor as assessment_editor,
)
from napari_vipp._tests.test_workflow_refinement_ui import (
    complete_job,
)
from napari_vipp._tests.test_workflow_refinement_ui import (
    refining as refining,
)
from napari_vipp.core import workflow_proposal_codex
from napari_vipp.core.workflow_proposals import reference_proposal


def configure(review):
    ids = add_and_run(review)
    node = next(
        node
        for node in review.widget.pipeline.nodes.values()
        if node.id in ids and node.operation_id == "label_connected_components"
    )
    return review.dialog._refinement.configure_preservation((node.id, 0))


def proposal(context, *, nominate=True, reject=False):
    raw = reference_proposal(context)
    raw["nodes"] = [node for node in raw["nodes"] if node["key"] != "measure"]
    raw["connections"] = [
        edge for edge in raw["connections"] if edge["target"] != "measure"
    ]
    # A different declared connectivity on a simple phantom changes semantics,
    # not its mask. This is a test adapter, never a live-model substitute.
    next(node for node in raw["nodes"] if node["key"] == "objects")["params"][
        "connectivity"
    ] = "Full connectivity"
    if reject:
        mask = next(node for node in raw["nodes"] if node["key"] == "mask")
        from napari_vipp.core.pipeline import NODE_LIBRARY_BY_ID

        mask["operation_id"] = "binary_threshold"
        mask["params"] = {
            p.name: p.default for p in NODE_LIBRARY_BY_ID["binary_threshold"].parameters
        }
        mask["params"]["threshold"] = 999
    if nominate:
        next(node for node in raw["nodes"] if node["key"] == "objects")["key"] = (
            "protected_structure"
        )
        for edge in raw["connections"]:
            for end in ("source", "target"):
                if edge[end] == "objects":
                    edge[end] = "protected_structure"
    return SimpleNamespace(proposal=raw, attempts=())


def test_off_by_default_and_configuration_is_passive(refining):
    controller = refining.dialog._refinement
    assert not controller.preservation_checkbox.isChecked()
    constraint = configure(refining)
    assert controller.preservation_constraint is constraint
    assert not refining.workers and not refining.provider_calls
    assert len(refining.runs) == 1
    assert controller.preservation_result is None
    assert constraint.public_contract()["required_retention_fraction"] == 1


@pytest.mark.parametrize("reject", [False, True])
def test_real_owned_run_checks_exact_nominated_mask_before_assessment(
    refining, monkeypatch, qtbot, reject
):
    constraint = configure(refining)
    d = refining.dialog
    monkeypatch.setattr(
        workflow_proposal_codex,
        "generate_codex_proposal",
        lambda context, *args, **kwargs: proposal(context, reject=reject),
    )
    d._refinement.slider.setValue(1)
    d._refinement.start_button.click()
    complete_job(refining, qtbot)
    assert (
        refining.requests[0]["preservation_constraint"] == constraint.public_contract()
    )
    complete_job(refining, qtbot)
    assert len(refining.runs) == 2, (
        d._refinement.note.text() + d.trace_edit.toPlainText()
    )
    result = d._refinement.preservation_result
    assert result is not None
    assert result.passed is not reject
    if reject:
        assert not refining.workers and not d._refinement.active
        assert "rejected" in d._refinement.note.text()
        assert d._assessment.run_id is None
        assert d._refinement.loop.completed_rounds == 0
        assert d._refinement.loop.attempted_runs == 1
    else:
        complete_job(refining, qtbot)
        assert refining.requests[-1]["preservation_check"]["passed"] is True
        assert d._refinement.loop.completed_rounds == 1
        assert not d._refinement.active


@pytest.mark.parametrize("mode", ["missing", "unused_mesh"])
def test_invalid_nomination_rejects_before_add_or_run(
    refining, monkeypatch, qtbot, mode
):
    configure(refining)
    d = refining.dialog
    monkeypatch.setattr(
        workflow_proposal_codex,
        "generate_codex_proposal",
        lambda context, *args, **kwargs: proposal(context, nominate=mode != "missing"),
    )
    if mode == "unused_mesh":
        from napari_vipp.ui import workflow_assistant_refinement

        def unused(*_args, **_kwargs):
            raise ValueError("An unused preserved copy is not accepted")

        monkeypatch.setattr(
            workflow_assistant_refinement, "preservation_mesh_creators", unused
        )
    d._refinement.slider.setValue(1)
    d._refinement.start_button.click()
    complete_job(refining, qtbot)
    complete_job(refining, qtbot)
    assert len(refining.runs) == 1
    assert not d._refinement.active and not refining.workers
    assert (
        "did not nominate" if mode == "missing" else "unused preserved copy"
    ) in d._refinement.note.text()
    assert not d._draft_authorized
    assert not d.apply_button.isEnabled()
    assert not d.run_button.isEnabled()


def test_invalid_selector_recapture_revokes_old_feedback_and_result(
    refining, monkeypatch, qtbot
):
    configure(refining)
    d = refining.dialog
    d._assessment.assess_button.click()
    complete_job(refining, qtbot)
    assert d._assessment.reply is not None
    d._draft_authorized = True
    d._refinement.preservation_result = object()

    def unavailable():
        raise ValueError("The new region has no foreground in common")

    monkeypatch.setattr(d._refinement, "_load_preservation_outputs", unavailable)
    d._refinement._preservation_selection_changed()
    assert d._assessment.reply is None
    assert not d._assessment.revision_requested
    assert d._refinement.preservation_constraint is None
    assert d._refinement.preservation_result is None
    assert not d._draft_authorized
    assert not d._assessment.revise_button.isEnabled()
    assert not refining.workers


@pytest.mark.parametrize("job", ["assessment", "revision"])
def test_changed_selector_discards_late_manual_request(refining, qtbot, job):
    configure(refining)
    d = refining.dialog
    d._assessment.assess_button.click()
    if job == "revision":
        complete_job(refining, qtbot)
        d._assessment.revise_button.click()
    worker = refining.workers.pop()
    reply = worker.call()
    d._refinement._preservation_selection_changed()
    assert not d._busy
    d._ready(worker.generation, reply)
    qtbot.wait(5)
    assert d._assessment.reply is None
    assert not d._assessment.revise_button.isEnabled()
    assert not refining.workers and len(refining.runs) == 1


def test_all_postrun_constraint_errors_revoke_current_acceptance(
    refining, monkeypatch, qtbot
):
    constraint = configure(refining)
    d = refining.dialog
    monkeypatch.setattr(
        workflow_proposal_codex,
        "generate_codex_proposal",
        lambda context, *args, **kwargs: proposal(context),
    )

    def wrong_grid(*_args, **_kwargs):
        raise ValueError("The nominated output has a different grid")

    monkeypatch.setattr(type(constraint), "check", wrong_grid)
    d._refinement.slider.setValue(1)
    d._refinement.start_button.click()
    complete_job(refining, qtbot)
    complete_job(refining, qtbot)
    assert len(refining.runs) == 2
    assert not refining.workers and not d._refinement.active
    assert d._assessment.run_id is None
    assert not d._draft_authorized
    assert d._refinement.preservation_result is None
    assert "different grid" in d._refinement.note.text()


@pytest.mark.parametrize("change", ["permission", "goal", "deadline", "cancel"])
def test_revocation_ignores_late_protected_request(refining, qtbot, change):
    configure(refining)
    d = refining.dialog
    d._refinement.slider.setValue(1)
    d._refinement.start_button.click()
    worker = refining.workers.pop()
    reply = worker.call()
    if change == "permission":
        d.output_assessment_checkbox.setChecked(False)
        assert d._refinement.preservation_constraint is None
    elif change == "goal":
        d.goal_edit.setPlainText("Changed objective")
    elif change == "deadline":
        d._refinement.loop.deadline = 0
        d._refinement._tick()
    else:
        d._refinement.stop_button.click()
    d._ready(worker.generation, reply)
    qtbot.wait(5)
    assert len(refining.runs) == 1
    assert not d._refinement.active and not refining.workers
