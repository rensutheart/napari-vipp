"""Recorded-assessment scope checks, no provider, source image or execution."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def continuation(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    return importlib.import_module("continue_workflow_microedit_assessment")


def test_continuation_scope_excludes_any_new_scientific_work(continuation):
    assert {"inspect_candidate", "inspect_source", "assess_experiment", "finish"} <= (
        continuation.ALLOWED_ACTIONS
    )
    assert (
        not {"trial_candidate", "trial_edit", "plan_experiment", "record_observation"}
        & continuation.ALLOWED_ACTIONS
    )


def test_entire_original_tree_is_pinned_before_any_restore(
    continuation, monkeypatch, tmp_path
):
    monkeypatch.setattr(continuation, "run_inventory", lambda _: {"changed": "hash"})
    with pytest.raises(ValueError, match="entire frozen incomplete"):
        continuation.load_recorded(tmp_path)


def test_execution_and_auth_are_forbidden_before_any_dispatch(
    continuation, monkeypatch
):
    from napari_vipp.core import workflow_investigation as core
    from napari_vipp.core import workflow_proposal_codex as codex

    monkeypatch.setattr(continuation.ProductObservers, "install", lambda _: None)
    monkeypatch.setattr(core.InvestigationSession, "perform", lambda *_: "allowed")
    observer = continuation.AssessmentOnlyObservers(None)
    observer.install()
    try:
        for kind in ("trial_candidate", "trial_edit", "plan_experiment"):
            with pytest.raises(ValueError, match="Only recorded"):
                core.InvestigationSession.perform(None, SimpleNamespace(action=kind))
        assert (
            core.InvestigationSession.perform(
                None, SimpleNamespace(action="inspect_candidate")
            )
            == "allowed"
        )
        for call in (
            core.execute_pipeline_request,
            codex.start_codex_login,
            codex.repair_codex_profile,
        ):
            with pytest.raises(RuntimeError, match="forbids execution/auth"):
                call()
    finally:
        observer.close()


def test_historical_numeric_hash_codec_is_strict(continuation):
    with pytest.raises(ValueError):
        continuation.compact({"number": float("nan")})
    assert continuation.compact({"number": True}) != continuation.compact({"number": 1})
