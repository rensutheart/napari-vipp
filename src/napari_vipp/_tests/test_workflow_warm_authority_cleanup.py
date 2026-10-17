"""Warm authority cleanup is local and happens before any worker submission."""

from types import SimpleNamespace

import pytest

from napari_vipp.ui.workflow_assistant import WorkflowAssistantDialog
from napari_vipp.ui.workflow_assistant_refinement import AssistantRefinementController


class StopBeforeWorker(Exception):
    pass


@pytest.mark.parametrize("busy", [False, True])
def test_withdrawn_protection_clears_pending_promotion_and_old_authority(busy):
    cancelled = []
    dialog = SimpleNamespace(
        _busy=busy,
        cancel_request=lambda **kwargs: cancelled.append(kwargs),
        _assessment=SimpleNamespace(reply=object(), revision_requested=True),
        _draft_authorized=True,
    )
    controller = SimpleNamespace(
        dialog=dialog,
        preservation_result=object(),
        _preserved_candidate_ref=("old-mask", 0),
        _pending_investigative_promotion=("old-promoted-mask", 0),
    )
    AssistantRefinementController._withdraw_preservation_draft_authority(controller)
    assert cancelled == ([{"announce": False}] if busy else [])
    assert dialog._assessment.reply is None
    assert not dialog._assessment.revision_requested
    assert not dialog._draft_authorized
    assert controller.preservation_result is None
    assert controller._preserved_candidate_ref is None
    assert controller._pending_investigative_promotion is None


@pytest.mark.parametrize("kind", ["proposal", "assessment", "assessment_preview"])
def test_only_new_proposal_replaces_selected_trial_authority_before_submission(kind):
    selected = ("old-context", "old-tested-fingerprint")
    pending = ("old-proposal_protected_structure", 0)
    calls = []

    def stop_before_worker():
        calls.append("before worker")
        raise StopBeforeWorker

    dialog = SimpleNamespace(
        _refinement=SimpleNamespace(
            before_request=lambda _kind: True,
            _pending_investigative_promotion=pending,
        ),
        _investigation=SimpleNamespace(_selected_draft=selected),
        _recovering=False,
        _generation=0,
        _busy=False,
        _draft_authorized=True,
        _start_request_clock=lambda: calls.append("clock"),
        _stop_request_clock=lambda: calls.append("clock stopped"),
        _invalidate_proposal=lambda: calls.append("invalidate"),
        _clear_error_details=lambda: None,
        trace_edit=SimpleNamespace(clear=lambda: None),
        _set_status=lambda *_args: None,
        _sync_buttons=stop_before_worker,
        attempts_spin=SimpleNamespace(value=lambda: 1),
    )
    with pytest.raises(StopBeforeWorker):
        WorkflowAssistantDialog._start(dialog, kind, lambda: pytest.fail("worker ran"))
    assert calls[-1] == "before worker"
    if kind == "proposal":
        assert dialog._investigation._selected_draft is None
        assert dialog._refinement._pending_investigative_promotion is None
        assert not dialog._draft_authorized
        assert "invalidate" in calls
    else:
        assert dialog._investigation._selected_draft == selected
        assert dialog._refinement._pending_investigative_promotion == pending
        assert dialog._draft_authorized
        assert "invalidate" not in calls
