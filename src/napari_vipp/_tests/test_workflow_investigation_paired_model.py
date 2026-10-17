"""Strict matched-pixel metadata crosses the usual isolated provider seam."""

import json
from copy import deepcopy

import pytest

from napari_vipp._tests.test_workflow_investigation import act
from napari_vipp._tests.test_workflow_investigation_model import (
    action_text,
    fake_provider,
)
from napari_vipp._tests.test_workflow_investigation_warmstart import (
    accepted,
    engine_for,
    update,
)
from napari_vipp.core import workflow_investigation_model as model
from napari_vipp.core import workflow_proposal_codex as codex
from napari_vipp.core import workflow_proposal_model as local


def paired():
    pipeline, _, bundle = accepted(protection=True, mesh=True)
    engine = engine_for(pipeline, bundle, include_input_images=True)
    views = [{"axis": "Z", "indices": [0], "montage": True, "overlay_source": True}]
    assert (
        act(
            engine,
            "inspect_candidate",
            candidate_id="candidate0",
            role="band_mask",
            views=views,
        ).public_result()["status"]
        == "ok"
    )
    assert update(engine, 90)["status"] == "ok"
    assert (
        act(
            engine,
            "inspect_candidate",
            candidate_id="candidate1",
            role="band_mask",
            views=views,
        ).public_result()["status"]
        == "ok"
    )
    request = engine.public_request()
    assert any(row["status"] == "supplied" for row in request["comparison_previews"])
    return engine, request


def test_actual_pairs_and_warnings_are_sent_without_housekeeping(monkeypatch):
    engine, request = paired()
    original = deepcopy(request)
    calls = fake_provider(monkeypatch, [action_text(session_id=engine.session_id)])
    result = model.generate_codex_investigation_action(
        request,
        codex.CodexSettings(max_attempts=1),
        image_previews=engine.preview_images(),
        before_call=lambda: engine.record_model_call(request=request),
    )
    assert result.action.session_id == engine.session_id and request == original
    assert engine.model_calls_used == 1 and engine.actions_used == 3
    assert any(
        row["payload"].get("kind") == "mesh-topology-review"
        for row in request["decision_context"]["numeric_evidence"]
    )
    assert "Zero boundary edges does not cancel" in model._COMPARISON_GUIDANCE
    assert "warnings neither repair" in model._COMPARISON_GUIDANCE
    assert model._COMPARISON_GUIDANCE in calls[0][1]["investigation_contract"][0]


@pytest.mark.parametrize(
    "mutation",
    [
        "extra",
        "source",
        "half",
        "owner",
        "role",
        "description",
        "status",
        "too_many",
        "too_large",
    ],
)
def test_invalid_pair_metadata_stops_before_provider_discovery(monkeypatch, mutation):
    engine, request = paired()
    row = next(
        row for row in request["comparison_previews"] if row["status"] == "supplied"
    )
    if mutation == "extra":
        row["execute"] = True
    elif mutation == "source":
        row["selection"]["source_sha256"] = "f" * 64
    elif mutation == "half":
        row["views"].pop()
    elif mutation == "owner":
        row["views"][1]["candidate_id"] = row["views"][0]["candidate_id"]
    elif mutation == "role":
        row["role"] = "another role"
    elif mutation == "description":
        row["selection"]["display_policy"]["render_description_sha256"] = "a" * 64
    elif mutation == "status":
        row["status"] = "omitted"
    elif mutation == "too_many":
        request["comparison_previews"] = [row] * 5
    else:
        row["reason"] = "x" * (8 * 1024)
    monkeypatch.setattr(
        codex, "discover_codex_runtime", lambda *_: pytest.fail("discovery")
    )
    with pytest.raises(local.ModelProposalError):
        model.generate_codex_investigation_action(
            request, codex.CodexSettings(), image_previews=engine.preview_images()
        )
    assert engine.model_calls_used == 0


def test_legacy_v2_projection_stays_accepted():
    engine, request = paired()
    del request["comparison_previews"]
    decoded, encoded = model._request(request, engine.preview_images())
    assert encoded == local._json_bytes(request)
    assert decoded == json.loads(encoded)


def test_identical_png_bytes_do_not_supply_an_absent_candidate_owner():
    # These genuinely different thresholds have the same pixels in native Z0.
    # Do not ask for a duplicate unchanged recipe, which the engine rejects.
    engine, request = paired()
    row = next(
        row for row in request["comparison_previews"] if row["status"] == "supplied"
    )
    assert row["views"][0]["preview_sha256"] == row["views"][1]["preview_sha256"]
    current_prefix = f"{row['candidate_ids'][1]}/{row['role']} "
    images = tuple(
        preview
        for preview in engine.preview_images()
        if not preview.label.startswith(current_prefix)
    )
    assert len(images) < len(engine.preview_images())
    request["image_previews"] = [preview.manifest() for preview in images]
    with pytest.raises(local.ModelProposalError, match="matched preview groups"):
        model._request(request, images)
    assert engine.model_calls_used == 0
